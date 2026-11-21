"""POST /v1/audio/transcriptions for a speech-recognition model, end to end against the stub engine, and the rules around it.

A real TranscriptionEngine drives tools/asr_stub.py over its serve protocol, and a real APIServer answers HTTP on a free port:
nothing between the socket and the engine's stdin is mocked. The stub "hears" a file by its length and echoes the language and the
context it was given, which is what lets these tests see that the upload was decoded to 16 kHz mono, that the fields were passed on,
and what the gateway does with the answer.
"""
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import uuid
import wave
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import asr_engine  # noqa: E402
import numpy as np  # noqa: E402
from asr_engine import (LANGUAGES, TranscriptionEngine, transcription_payload, transcription_request)  # noqa: E402
from embedding_engine import EmbeddingEngineError, EmbeddingRequestError  # noqa: E402
from family_registry import default_model_id, display_for, resolve_model  # noqa: E402
from openai_server import APIServer  # noqa: E402

TOOLS = Path(__file__).resolve().parent.parent / "tools"
STUB = str(TOOLS / "asr_stub.py")
MODEL_ID = "qwen3-asr-0.6b"
KEY = "secret"


def stub_env(**extra):
    env = dict(os.environ)
    env.update({key: str(value) for key, value in extra.items()})
    return env


def wav_bytes(seconds, rate=16000, tone=True, channels=1):
    """A WAV file of `seconds` of a 440 Hz tone (or silence), at `rate`, as the bytes an upload would be."""
    t = np.arange(int(seconds * rate)) / rate
    samples = (np.sin(2 * np.pi * 440 * t) * 0.3 if tone else np.zeros(len(t))).astype(np.float32)
    pcm = (samples * 32767).astype("<i2")
    if channels == 2:
        pcm = np.repeat(pcm[:, None], 2, axis=1).reshape(-1)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


def multipart(fields, filename="clip.wav", data=None):
    boundary = "----coli" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    if data is not None:
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                      "Content-Type: audio/wav\r\n\r\n").encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class RequestRulesTest(unittest.TestCase):
    def refused(self, body, param, fragment=""):
        with self.assertRaises(EmbeddingRequestError) as caught:
            transcription_request(body)
        self.assertEqual(caught.exception.param, param, str(caught.exception))
        self.assertIn(fragment, str(caught.exception))

    def test_the_defaults(self):
        self.assertEqual(transcription_request({}), {"format": "json", "language": None, "language_code": None, "context": ""})

    def test_a_language_code_becomes_the_name_the_model_is_told(self):
        for code, name in (("en", "English"), ("EN", "English"), ("zh", "Chinese"), ("yue", "Cantonese"), ("pt-BR", "Portuguese"),
                           ("tl", "Filipino"), ("mk", "Macedonian")):
            with self.subTest(code=code):
                self.assertEqual(transcription_request({"language": code})["language"], name)

    def test_every_name_the_model_knows_has_a_code(self):
        self.assertEqual(set(asr_engine.ISO_NAMES.values()), set(LANGUAGES))

    def test_a_language_the_model_does_not_have_is_refused_with_what_it_has(self):
        self.refused({"language": "xx"}, "language", "not one this model transcribes")
        self.refused({"language": "he"}, "language", "yue")
        self.refused({"language": "english"}, "language", "ISO-639")
        self.refused({"language": 5}, "language", "ISO-639")

    def test_an_empty_language_means_detect(self):
        self.assertIsNone(transcription_request({"language": ""})["language"])

    def test_formats(self):
        for fmt in ("json", "text", "verbose_json"):
            self.assertEqual(transcription_request({"response_format": fmt})["format"], fmt)
        self.refused({"response_format": "srt"}, "response_format", "timestamps")
        self.refused({"response_format": "vtt"}, "response_format", "timestamps")
        self.refused({"response_format": "xml"}, "response_format", "one of")

    def test_word_timestamps_and_streaming_are_refused(self):
        self.refused({"timestamp_granularities": ["word"]}, "timestamp_granularities", "Word timestamps")
        self.refused({"timestamp_granularities[]": "word"}, "timestamp_granularities")
        self.assertEqual(transcription_request({"timestamp_granularities": ["segment"]})["format"], "json")
        self.refused({"stream": "true"}, "stream")

    def test_the_prompt_is_the_context_and_has_a_limit(self):
        self.assertEqual(transcription_request({"prompt": "  Kennedy, Berlin  "})["context"], "Kennedy, Berlin")
        self.refused({"prompt": "x" * 1001}, "prompt", "1000")
        self.refused({"prompt": 3}, "prompt")

    def test_temperature_is_checked_and_ignored(self):
        self.assertEqual(transcription_request({"temperature": 0.4})["format"], "json")
        self.refused({"temperature": 2}, "temperature")
        self.refused({"temperature": True}, "temperature")

    def test_the_payloads(self):
        request = transcription_request({"response_format": "verbose_json"})
        result = {"text": "hello there", "language": "English"}
        payload = transcription_payload(request, result, 2.6, MODEL_ID)
        self.assertEqual((payload["task"], payload["language"], payload["duration"], payload["text"]), ("transcribe", "english", 2.6, "hello there"))
        self.assertEqual(payload["segments"], [{"id": 0, "seek": 0, "start": 0.0, "end": 2.6, "text": "hello there"}])
        self.assertEqual(payload["usage"], {"type": "duration", "seconds": 3})
        self.assertEqual(transcription_payload(request, {"text": "", "language": ""}, 1.0, MODEL_ID)["segments"], [])
        self.assertNotIn("language", transcription_payload(request, {"text": "x", "language": ""}, 1.0, MODEL_ID))
        plain = transcription_payload(transcription_request({}), result, 2.6, MODEL_ID)
        self.assertEqual(plain, {"text": "hello there", "usage": {"type": "duration", "seconds": 3}})
        forced = transcription_request({"response_format": "verbose_json", "language": "fr"})
        self.assertEqual(transcription_payload(forced, {"text": "x", "language": ""}, 1.0, MODEL_ID)["language"], "french")


class EngineClientTest(unittest.TestCase):
    def engine(self, **env):
        engine = TranscriptionEngine(STUB, "model-dir", env=stub_env(**env), on_log=lambda _m: None)
        self.addCleanup(engine.close)
        return engine

    def test_ready_tells_what_the_engine_does(self):
        engine = self.engine()
        self.assertEqual((engine.info["task"], engine.info["sample_rate"], engine.info["piece_seconds"]), ("transcribe", 16000, 1200))
        self.assertEqual((engine.modality, engine.chats, engine.decides), ("transcription", False, False))
        self.assertTrue(engine.alive)

    def test_a_file_is_heard_and_the_fields_arrive(self):
        engine = self.engine()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            path.write_bytes(wav_bytes(2.0))
            result = engine.transcribe(path, language="French", context="Berlin")
        self.assertEqual(result["text"], "French | 2.0 s | Berlin")
        self.assertEqual((result["language"], result["tokens"], result["pieces"], result["truncated"]), ("French", 5, 1, False))
        self.assertAlmostEqual(result["audio_seconds"], 2.0, places=3)

    def test_the_engine_names_a_language_when_none_is_forced(self):
        self.assertEqual(self.engine().transcribe_samples(wav_bytes(1.0))["language"], "English")
        self.assertEqual(self.engine(ASR_STUB_LANGUAGE="").transcribe_samples(wav_bytes(1.0))["language"], "")

    def test_the_temporary_file_is_gone_afterwards(self):
        engine = self.engine()
        before = set(Path(tempfile.gettempdir()).glob("coli-asr-*.wav"))
        engine.transcribe_samples(wav_bytes(1.0))
        self.assertEqual(set(Path(tempfile.gettempdir()).glob("coli-asr-*.wav")), before)

    def test_an_error_from_the_engine_is_an_exception_and_it_goes_on(self):
        engine = self.engine(ASR_STUB_FAIL_ON="bad")
        with self.assertRaises(EmbeddingEngineError) as caught:
            engine.transcribe_samples(wav_bytes(1.0), context="bad")
        self.assertIn("could not be read", str(caught.exception))
        self.assertTrue(engine.alive)
        self.assertEqual(engine.transcribe_samples(wav_bytes(1.0), context="fine")["text"], "English | 1.0 s | fine")

    def test_a_dead_engine_says_so(self):
        engine = self.engine(ASR_STUB_DIE_ON="boom")
        with self.assertRaises(EmbeddingEngineError):
            engine.transcribe_samples(wav_bytes(1.0), context="boom")
        self.assertFalse(engine.alive)
        with self.assertRaises(EmbeddingEngineError):
            engine.transcribe_samples(wav_bytes(1.0))


class GatewayHarness:
    def __init__(self, model_id=MODEL_ID, **env):
        self.engine = TranscriptionEngine(STUB, "model-dir", env=stub_env(**env), on_log=lambda _m: None)
        self.server = APIServer(("127.0.0.1", 0), self.engine, model_id, KEY, 16, max_queue=8, queue_timeout=30)
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def close(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.engine.close()

    def call(self, path, data=None, content_type="application/json", key=KEY, method=None):
        headers = {"Content-Type": content_type}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = Request(f"http://127.0.0.1:{self.port}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=30) as response:
                return response.status, dict(response.headers), response.read()
        except HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def json(self, path, body=None, **kw):
        data = None if body is None else json.dumps(body).encode()
        status, headers, raw = self.call(path, data, **kw)
        return status, headers, json.loads(raw or b"{}")

    def transcribe(self, audio, filename="clip.wav", **fields):
        fields.setdefault("model", MODEL_ID)
        body, content_type = multipart(fields, filename, audio)
        return self.call("/v1/audio/transcriptions", body, content_type)


class EndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def body(self, response):
        status, headers, raw = response
        return status, headers, json.loads(raw)

    def test_a_clip_comes_back_as_text_in_openais_shape(self):
        status, headers, body = self.body(self.gw.transcribe(wav_bytes(2.0)))
        self.assertEqual(status, 200, body)
        self.assertEqual(body["text"], "English | 2.0 s")
        self.assertEqual(body["usage"], {"type": "duration", "seconds": 2})
        self.assertIn("x-colibri-queue-wait-ms", {k.lower() for k in headers})

    def test_a_json_body_with_the_file_in_base64_works_too(self):
        import base64
        status, _, body = self.gw.json("/v1/audio/transcriptions", {"model": MODEL_ID, "file": base64.b64encode(wav_bytes(1.0)).decode()})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["text"], "English | 1.0 s")

    def test_the_language_and_the_prompt_reach_the_engine(self):
        status, _, body = self.body(self.gw.transcribe(wav_bytes(1.0), language="de", prompt="Brandenburger Tor"))
        self.assertEqual(status, 200, body)
        self.assertEqual(body["text"], "German | 1.0 s | Brandenburger Tor")

    def test_the_upload_is_decoded_to_16_khz_mono_whatever_it_was(self):
        for rate, channels in ((8000, 1), (44100, 2), (48000, 1)):
            with self.subTest(rate=rate, channels=channels):
                status, _, body = self.body(self.gw.transcribe(wav_bytes(1.5, rate=rate, channels=channels)))
                self.assertEqual(status, 200, body)
                self.assertEqual(body["text"], "English | 1.5 s")

    def test_text_and_verbose_json(self):
        status, headers, raw = self.gw.transcribe(wav_bytes(2.0), response_format="text", language="en")
        self.assertEqual((status, raw), (200, b"English | 2.0 s\n"))
        self.assertIn("text/plain", headers.get("Content-Type", ""))
        status, _, body = self.body(self.gw.transcribe(wav_bytes(2.0), response_format="verbose_json"))
        self.assertEqual((body["task"], body["language"], body["duration"]), ("transcribe", "english", 2.0))
        self.assertEqual(body["segments"][0]["text"], body["text"])
        self.assertEqual((body["segments"][0]["start"], body["segments"][0]["end"]), (0.0, 2.0))

    def test_silence_is_not_asked_about(self):
        status, _, body = self.body(self.gw.transcribe(wav_bytes(2.0, tone=False)))
        self.assertEqual((status, body["text"]), (200, ""))
        status, _, body = self.body(self.gw.transcribe(wav_bytes(2.0, tone=False), response_format="verbose_json"))
        self.assertEqual((body["text"], body["segments"]), ("", []))

    def test_refusals_carry_the_field(self):
        cases = [({"response_format": "srt"}, "response_format"), ({"language": "xx"}, "language"), ({"prompt": "x" * 1001}, "prompt"),
                 ({"temperature": "9"}, "temperature"), ({"stream": "true"}, "stream"), ({"timestamp_granularities[]": "word"}, "timestamp_granularities")]
        for fields, param in cases:
            with self.subTest(fields=fields):
                status, _, body = self.body(self.gw.transcribe(wav_bytes(1.0), **fields))
                self.assertEqual(status, 400, body)
                self.assertEqual(body["error"]["param"], param)

    def test_a_missing_or_unreadable_file_is_a_400(self):
        body, content_type = multipart({"model": MODEL_ID})
        status, _, raw = self.gw.call("/v1/audio/transcriptions", body, content_type)
        self.assertEqual((status, json.loads(raw)["error"]["code"]), (400, "missing_required_parameter"))
        status, _, body = self.body(self.gw.transcribe(b"this is not audio at all"))
        self.assertEqual((status, body["error"]["code"]), (400, "invalid_audio"))

    def test_a_file_over_the_limit_is_refused(self):
        import openai_server
        original = openai_server.TRANSCRIBE_MAX_SECONDS
        openai_server.TRANSCRIBE_MAX_SECONDS = 3
        try:
            status, _, body = self.body(self.gw.transcribe(wav_bytes(5.0)))
        finally:
            openai_server.TRANSCRIBE_MAX_SECONDS = original
        self.assertEqual((status, body["error"]["code"]), (400, "invalid_audio"))
        self.assertIn("limit", body["error"]["message"])

    def test_the_key_is_checked(self):
        body, content_type = multipart({"model": MODEL_ID}, data=wav_bytes(1.0))
        status, _, _raw = self.gw.call("/v1/audio/transcriptions", body, content_type, key="wrong")
        self.assertEqual(status, 401)

    def test_the_model_card_says_what_it_takes_and_gives(self):
        status, _, body = self.gw.json("/v1/models", method="GET")
        card = next(item for item in body["data"] if item["id"] == MODEL_ID)
        self.assertEqual(card["capabilities"], ["transcription"])
        self.assertEqual(card["input_modalities"], ["audio"])
        self.assertEqual(card["transcription"]["sample_rate"], 16000)
        self.assertEqual(card["transcription"]["languages"], list(LANGUAGES))
        self.assertEqual(card["transcription"]["formats"], ["json", "text", "verbose_json"])
        self.assertEqual(card["transcription"]["device"], "cpu")

    def test_the_other_routes_say_where_to_go(self):
        status, _, body = self.gw.json("/v1/chat/completions", {"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 400, body)
        self.assertIn("speech recognition model", body["error"]["message"])
        self.assertIn("/v1/audio/transcriptions", body["error"]["message"])
        for path, what in (("/v1/images/generations", "generate images"), ("/v1/audio/music", "make music"),
                           ("/v1/embeddings", "make embeddings"), ("/v1/rerank", "rank documents")):
            with self.subTest(path=path):
                body_in = {"model": MODEL_ID, "prompt": "p", "input": "i", "query": "q", "documents": ["d"]}
                status, _, body = self.gw.json(path, body_in)
                self.assertEqual(status, 400, body)
                self.assertIn(f"does not {what}", body["error"]["message"])
                self.assertIn("transcribes speech", body["error"]["message"])


class DeadEngineTest(unittest.TestCase):
    def test_a_dead_engine_is_a_503_not_a_hang(self):
        gw = GatewayHarness(ASR_STUB_DIE_ON="boom")
        try:
            status, _, raw = gw.transcribe(wav_bytes(1.0), prompt="boom")
            self.assertEqual(status, 500, raw)
            self.assertIn("speech recognition engine failed", json.loads(raw)["error"]["message"])
            status, _, raw = gw.transcribe(wav_bytes(1.0), prompt="after")
            self.assertEqual((status, json.loads(raw)["error"]["code"]), (503, "engine_unavailable"))
        finally:
            gw.close()


class OtherModelsRefuseTest(unittest.TestCase):
    def test_a_transcription_request_to_a_non_speech_server_is_refused_with_its_purpose(self):
        from embedding_engine import EmbeddingEngine
        engine = EmbeddingEngine(str(TOOLS / "embedding_stub.py"), "model-dir", env=stub_env(), on_log=lambda _m: None)
        server = APIServer(("127.0.0.1", 0), engine, "qwen3-embedding-0.6b", KEY, 16, max_queue=4, queue_timeout=10)
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        try:
            body, content_type = multipart({"model": "qwen3-embedding-0.6b"}, data=wav_bytes(1.0))
            req = Request(f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions", data=body,
                          headers={"Content-Type": content_type, "Authorization": f"Bearer {KEY}"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(req, timeout=30)
            self.assertEqual(caught.exception.code, 400)
        finally:
            server.scheduler.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            engine.close(timeout=5)


class FamilyTest(unittest.TestCase):
    def checkpoint(self, root, hidden):
        root = Path(root)
        root.mkdir(parents=True)
        (root / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3ASRForConditionalGeneration"], "model_type": "qwen3_asr",
            "thinker_config": {"audio_config": {"encoder_layers": 18, "d_model": 896},
                               "text_config": {"hidden_size": hidden, "num_hidden_layers": 28, "model_type": "qwen3"}}}))
        return root

    def test_the_checkpoints_are_told_apart_by_the_width_of_the_decoder(self):
        for hidden, model_id, scale in ((1024, "qwen3-asr-0.6b", "0.6B"), (2048, "qwen3-asr-1.7b", "1.7B")):
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as tmp:
                resolved = resolve_model(self.checkpoint(Path(tmp) / "m", hidden))
                family = resolved.descriptor
                self.assertEqual((family.id, family.modality, family.engine_artifact), ("qwen3_asr", "transcription", "qwen3asr"))
                self.assertEqual(default_model_id(resolved), model_id)
                self.assertEqual(display_for(resolved), ("Qwen3-ASR", scale))
                self.assertEqual(resolved.family_config["hidden_size"], hidden)

    def test_a_config_without_the_thinker_text_config_is_read_as_it_stands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"
            root.mkdir()
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3_asr", "thinker_config": {}}))
            resolved = resolve_model(root)
            self.assertEqual((resolved.descriptor.id, default_model_id(resolved)), ("qwen3_asr", "qwen3-asr"))
            self.assertEqual(resolved.family_config, resolved.config)


if __name__ == "__main__":
    unittest.main()
