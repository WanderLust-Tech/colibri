"""POST /v1/audio/speech for a speech-synthesis model, end to end against the stub engine, and the rules around it.

A real SpeechEngine drives tools/tts_stub.py over its serve protocol, and a real APIServer answers HTTP on a free port: nothing between
the socket and the engine's stdin is mocked. The stub writes a WAV whose first samples spell out the fields it was given
(`speaker|language|seed|greedy|max_frames`), which is how these tests see what the gateway passed on.
"""
import io
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import speech_engine  # noqa: E402
from embedding_engine import EmbeddingEngineError, EmbeddingRequestError  # noqa: E402
from family_registry import default_model_id, display_for, resolve_model  # noqa: E402
from openai_server import APIServer  # noqa: E402
from speech_engine import SpeechEngine, encode_audio, speech_request, wav_pcm  # noqa: E402

TOOLS = Path(__file__).resolve().parent.parent / "tools"
STUB = str(TOOLS / "tts_stub.py")
MODEL_ID = "qwen3-tts-0.6b"
KEY = "secret"
INFO = {"speakers": ["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"],
        "languages": ["chinese", "english", "german", "italian", "portuguese", "spanish", "japanese", "korean", "french", "russian",
                      "beijing_dialect", "sichuan_dialect"]}


def stub_env(**extra):
    env = dict(os.environ)
    env.update({key: str(value) for key, value in extra.items()})
    return env


def read_wav(data):
    """(rate, channels, int16 samples as a tuple) of a WAV file's bytes."""
    with wave.open(io.BytesIO(data), "rb") as clip:
        frames = clip.readframes(clip.getnframes())
        return clip.getframerate(), clip.getnchannels(), struct.unpack(f"<{len(frames) // 2}h", frames)


def echo_of(data):
    """What the stub says it was given: the string its WAV spells out."""
    _rate, _channels, samples = read_wav(data)
    return bytes(samples[1:1 + samples[0]]).decode()


class RequestRulesTest(unittest.TestCase):
    def refused(self, body, param, fragment=""):
        with self.assertRaises(EmbeddingRequestError) as caught:
            speech_request(body, INFO)
        self.assertEqual(caught.exception.param, param, str(caught.exception))
        self.assertIn(fragment, str(caught.exception))

    def test_the_defaults(self):
        self.assertEqual(speech_request({"input": "hello"}, INFO),
                         {"input": "hello", "speaker": "ryan", "language": "auto", "format": "wav", "seed": None, "greedy": False})

    def test_input_is_required_and_limited(self):
        self.refused({}, "input", "required")
        self.refused({"input": "   "}, "input", "required")
        self.refused({"input": 5}, "input", "required")
        self.refused({"input": "x" * 4097}, "input", "4096")
        self.assertEqual(speech_request({"input": "x" * 4096}, INFO)["input"], "x" * 4096)

    def test_a_voice_is_a_speaker_of_the_model_or_one_of_openais_names(self):
        for voice, speaker in (("vivian", "vivian"), ("Vivian", "vivian"), ("uncle fu", "uncle_fu"), ("uncle-fu", "uncle_fu"),
                               ("alloy", "ryan"), ("nova", "vivian"), ("shimmer", "serena"), ("onyx", "uncle_fu")):
            with self.subTest(voice=voice):
                self.assertEqual(speech_request({"input": "x", "voice": voice}, INFO)["speaker"], speaker)
        self.assertEqual(speech_request({"input": "x", "voice": {"id": "sohee"}}, INFO)["speaker"], "sohee")

    def test_every_openai_name_stands_for_a_speaker_the_model_has(self):
        self.assertTrue(set(speech_engine.OPENAI_VOICES.values()) <= set(INFO["speakers"]))

    def test_an_unknown_voice_is_refused_with_what_there_is(self):
        self.refused({"input": "x", "voice": "morgan"}, "voice", "uncle_fu")
        self.refused({"input": "x", "voice": ""}, "voice")
        self.refused({"input": "x", "voice": 3}, "voice")

    def test_a_language_is_a_name_or_a_code(self):
        for language, name in (("en", "english"), ("EN", "english"), ("zh", "chinese"), ("en-US", "english"), ("German", "german"),
                               ("beijing dialect", "beijing_dialect"), ("auto", "auto")):
            with self.subTest(language=language):
                self.assertEqual(speech_request({"input": "x", "language": language}, INFO)["language"], name)
        self.refused({"input": "x", "language": "xx"}, "language", "not one this model speaks")
        self.refused({"input": "x", "language": "klingon"}, "language", "french")
        self.refused({"input": "x", "language": ""}, "language")

    def test_formats(self):
        for fmt in ("wav", "pcm", "mp3", "opus", "aac", "flac"):
            self.assertEqual(speech_request({"input": "x", "response_format": fmt}, INFO)["format"], fmt)
        self.refused({"input": "x", "response_format": "ogg"}, "response_format", "one of")

    def test_speed_instructions_and_streaming_are_refused(self):
        self.assertEqual(speech_request({"input": "x", "speed": 1}, INFO)["format"], "wav")
        self.refused({"input": "x", "speed": 1.5}, "speed", "not supported")
        self.refused({"input": "x", "speed": 9}, "speed", "0.25")
        self.refused({"input": "x", "speed": True}, "speed")
        self.refused({"input": "x", "instructions": "cheerful"}, "instructions", "not supported")
        self.refused({"input": "x", "instructions": 4}, "instructions")
        self.assertEqual(speech_request({"input": "x", "instructions": ""}, INFO)["format"], "wav")
        self.refused({"input": "x", "stream_format": "sse"}, "stream_format")
        self.assertEqual(speech_request({"input": "x", "stream_format": "audio"}, INFO)["format"], "wav")

    def test_a_model_that_takes_instructions_gets_them_through_the_check(self):
        self.assertEqual(speech_request({"input": "x", "instructions": "calm"}, dict(INFO, instruct=True))["speaker"], "ryan")

    def test_a_seed_is_a_non_negative_integer(self):
        self.assertEqual(speech_request({"input": "x", "seed": 7}, INFO)["seed"], 7)
        self.assertIsNone(speech_request({"input": "x", "seed": 0}, INFO)["seed"])
        self.refused({"input": "x", "seed": -1}, "seed")
        self.refused({"input": "x", "seed": 1.5}, "seed")
        self.refused({"input": "x", "seed": True}, "seed")

    def test_without_the_engines_tables_the_names_pass_through(self):
        self.assertEqual(speech_request({"input": "x", "voice": "anyone", "language": "elvish"})["speaker"], "anyone")

    def test_a_body_that_is_not_an_object(self):
        with self.assertRaises(EmbeddingRequestError):
            speech_request([], INFO)


class EncodingTest(unittest.TestCase):
    WAV = None

    @classmethod
    def setUpClass(cls):
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(struct.pack("<5h", 1, -2, 3, -4, 5))
        cls.WAV = out.getvalue()

    def test_pcm_is_the_data_chunk(self):
        self.assertEqual(wav_pcm(self.WAV), struct.pack("<5h", 1, -2, 3, -4, 5))
        self.assertEqual(encode_audio(self.WAV, "pcm"), struct.pack("<5h", 1, -2, 3, -4, 5))

    def test_wav_is_left_alone(self):
        self.assertIs(encode_audio(self.WAV, "wav"), self.WAV)

    def test_a_chunk_before_data_is_skipped_and_a_non_wav_is_refused(self):
        extra = self.WAV[:12] + b"LIST" + struct.pack("<I", 3) + b"abc\x00" + self.WAV[12:]
        self.assertEqual(wav_pcm(extra), struct.pack("<5h", 1, -2, 3, -4, 5))
        with self.assertRaises(ValueError):
            wav_pcm(b"not a wav file at all")
        with self.assertRaises(ValueError):
            wav_pcm(self.WAV[:12] + b"fmt " + struct.pack("<I", 0))

    def test_a_format_that_needs_ffmpeg_is_refused_when_there_is_none(self):
        with mock.patch.object(speech_engine, "ffmpeg_path", return_value=None):
            with self.assertRaises(EmbeddingRequestError) as caught:
                encode_audio(self.WAV, "mp3")
        self.assertEqual(caught.exception.param, "response_format")
        self.assertIn("needs ffmpeg", str(caught.exception))

    def test_a_failing_ffmpeg_is_an_engine_error_not_silence(self):
        failed = mock.Mock(returncode=1, stdout=b"", stderr=b"Unknown encoder 'libmp3lame'")
        with mock.patch.object(speech_engine.subprocess, "run", return_value=failed):
            with self.assertRaises(EmbeddingEngineError) as caught:
                encode_audio(self.WAV, "mp3", ffmpeg="ffmpeg")
        self.assertIn("ffmpeg could not make mp3", str(caught.exception))
        self.assertIn("libmp3lame", str(caught.exception))
        silent = mock.Mock(returncode=0, stdout=b"", stderr=b"")           # exit 0 and nothing written is a failure too
        with mock.patch.object(speech_engine.subprocess, "run", return_value=silent):
            with self.assertRaises(EmbeddingEngineError):
                encode_audio(self.WAV, "flac", ffmpeg="ffmpeg")

    def test_real_ffmpeg_makes_the_formats(self):
        exe = speech_engine.ffmpeg_path()
        if not exe:
            self.skipTest("no ffmpeg here")
        wav = tone_wav()
        for fmt, magic in (("mp3", (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")), ("flac", (b"fLaC",)), ("opus", (b"OggS",)), ("aac", (b"\xff\xf1", b"\xff\xf9"))):
            with self.subTest(fmt=fmt):
                data = encode_audio(wav, fmt, ffmpeg=exe)
                self.assertTrue(data.startswith(magic), data[:8])


def tone_wav(seconds=1.0, rate=24000):
    """A WAV file of a square tone, as the bytes an engine would have written."""
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{int(seconds * rate)}h", *[8000 * ((i // 20) % 2 * 2 - 1) for i in range(int(seconds * rate))]))
    return out.getvalue()


class EngineClientTest(unittest.TestCase):
    def engine(self, **env):
        engine = SpeechEngine(STUB, "model-dir", env=stub_env(**env), on_log=lambda _m: None)
        self.addCleanup(engine.close)
        return engine

    def test_ready_tells_what_the_engine_does(self):
        engine = self.engine()
        self.assertEqual((engine.info["task"], engine.info["sample_rate"]), ("speech", 24000))
        self.assertIn("ryan", engine.info["speakers"])
        self.assertIn("english", engine.info["languages"])
        self.assertEqual((engine.modality, engine.chats, engine.decides), ("speech", False, False))
        self.assertTrue(engine.alive)

    def test_a_text_is_spoken_and_the_fields_arrive(self):
        result = self.engine().speak("hello there", speaker="vivian", language="french", seed=9, greedy=True, max_frames=100)
        self.assertEqual(echo_of(result["wav"]), "vivian|french|9|1|100")
        rate, channels, samples = read_wav(result["wav"])
        self.assertEqual((rate, channels, len(samples)), (24000, 1, int(len("hello there") * 0.05 * 24000)))
        self.assertAlmostEqual(result["audio_seconds"], len("hello there") * 0.05, places=3)
        self.assertEqual((result["sample_rate"], result["truncated"]), (24000, False))

    def test_unset_fields_are_not_sent(self):
        self.assertEqual(echo_of(self.engine().speak("hello there", speaker="ryan")["wav"]), "ryan|auto|0|0|0")

    def test_the_temporary_file_is_gone_afterwards(self):
        engine = self.engine()
        before = set(Path(tempfile.gettempdir()).glob("coli-tts-*.wav"))
        engine.speak("hello there")
        self.assertEqual(set(Path(tempfile.gettempdir()).glob("coli-tts-*.wav")), before)

    def test_a_truncated_answer_says_so(self):
        self.assertTrue(self.engine(TTS_STUB_TRUNCATE=1).speak("hello there")["truncated"])

    def test_an_error_from_the_engine_is_an_exception_and_it_goes_on(self):
        engine = self.engine(TTS_STUB_FAIL_ON="bad")
        with self.assertRaises(EmbeddingEngineError) as caught:
            engine.speak("bad")
        self.assertIn("not one this model has", str(caught.exception))
        self.assertTrue(engine.alive)
        self.assertEqual(echo_of(engine.speak("fine", speaker="ryan")["wav"]), "ryan|auto|0|0|0")

    def test_an_unknown_speaker_reaches_the_engine_and_comes_back_as_its_error(self):
        with self.assertRaises(EmbeddingEngineError) as caught:
            self.engine().speak("hello", speaker="nobody")
        self.assertIn("nobody", str(caught.exception))

    def test_an_engine_that_wrote_nothing_is_an_error(self):
        with self.assertRaises(EmbeddingEngineError):
            self.engine(TTS_STUB_NOFILE=1).speak("hello there")

    def test_a_dead_engine_says_so(self):
        engine = self.engine(TTS_STUB_DIE_ON="boom")
        with self.assertRaises(EmbeddingEngineError):
            engine.speak("boom")
        self.assertFalse(engine.alive)
        with self.assertRaises(EmbeddingEngineError):
            engine.speak("after")


class GatewayHarness:
    def __init__(self, model_id=MODEL_ID, **env):
        self.engine = SpeechEngine(STUB, "model-dir", env=stub_env(**env), on_log=lambda _m: None)
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

    def speak(self, text="hello there", **fields):
        fields.setdefault("model", MODEL_ID)
        fields["input"] = text
        return self.call("/v1/audio/speech", json.dumps(fields).encode())


class EndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def error(self, response):
        status, _headers, raw = response
        return status, json.loads(raw)["error"]

    def test_a_text_comes_back_as_a_wav_in_openais_shape(self):
        status, headers, raw = self.gw.speak("hello there")
        self.assertEqual(status, 200, raw[:200])
        self.assertEqual(headers["Content-Type"], "audio/wav")
        self.assertEqual(echo_of(raw), "ryan|auto|0|0|0")
        self.assertEqual(read_wav(raw)[0], 24000)
        self.assertIn("x-colibri-queue-wait-ms", {k.lower() for k in headers})
        self.assertAlmostEqual(float(headers["x-colibri-audio-seconds"]), len("hello there") * 0.05, places=3)
        self.assertEqual(headers["x-colibri-sample-rate"], "24000")
        self.assertNotIn("x-colibri-truncated", headers)

    def test_the_voice_the_language_and_the_seed_reach_the_engine(self):
        _status, _headers, raw = self.gw.speak(voice="Vivian", language="zh", seed=11)
        self.assertEqual(echo_of(raw), "vivian|chinese|11|0|0")
        _status, _headers, raw = self.gw.speak(voice="alloy")
        self.assertEqual(echo_of(raw), "ryan|auto|0|0|0")
        _status, _headers, raw = self.gw.speak(voice="fable")
        self.assertEqual(echo_of(raw), "dylan|auto|0|0|0")

    def test_pcm_is_raw_samples(self):
        status, headers, raw = self.gw.speak("hello there", response_format="pcm")
        self.assertEqual((status, headers["Content-Type"]), (200, "audio/pcm"))
        self.assertEqual(len(raw), int(len("hello there") * 0.05 * 24000) * 2)
        self.assertEqual(struct.unpack_from("<h", raw)[0], len("ryan|auto|0|0|0"))

    def test_a_format_that_needs_ffmpeg_is_refused_before_the_work_when_there_is_none(self):
        before = self.gw.engine._seq
        with mock.patch.object(speech_engine, "ffmpeg_path", return_value=None):
            status, error = self.error(self.gw.speak("hello", response_format="mp3"))
        self.assertEqual((status, error["param"]), (400, "response_format"))
        self.assertIn("needs ffmpeg", error["message"])
        self.assertEqual(self.gw.engine._seq, before, "the engine was asked anyway")

    def test_mp3_through_a_real_ffmpeg(self):
        if not speech_engine.ffmpeg_path():
            self.skipTest("no ffmpeg here")
        status, headers, raw = self.gw.speak("hello there " * 8, response_format="mp3")
        self.assertEqual((status, headers["Content-Type"]), (200, "audio/mpeg"), raw[:200])
        self.assertTrue(raw.startswith((b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")), raw[:8])

    def test_a_cut_short_answer_is_marked(self):
        gw = GatewayHarness(TTS_STUB_TRUNCATE=1)
        try:
            status, headers, _raw = gw.speak("hello there")
            self.assertEqual((status, headers["x-colibri-truncated"]), (200, "true"))
        finally:
            gw.close()

    def test_refusals_carry_the_field(self):
        cases = [({"voice": "morgan"}, "voice"), ({"language": "xx"}, "language"), ({"response_format": "ogg"}, "response_format"),
                 ({"speed": 2}, "speed"), ({"instructions": "gently"}, "instructions"), ({"seed": -3}, "seed"),
                 ({"stream_format": "sse"}, "stream_format")]
        for fields, param in cases:
            with self.subTest(fields=fields):
                status, error = self.error(self.gw.speak("hello", **fields))
                self.assertEqual((status, error["param"]), (400, param), error)

    def test_input_is_required(self):
        status, _headers, raw = self.gw.call("/v1/audio/speech", json.dumps({"model": MODEL_ID}).encode())
        self.assertEqual((status, json.loads(raw)["error"]["param"]), (400, "input"))
        status, error = self.error(self.gw.speak("x" * 5000))
        self.assertEqual((status, error["param"]), (400, "input"))

    def test_an_engine_error_is_a_500_that_says_so(self):
        gw = GatewayHarness(TTS_STUB_FAIL_ON="bad")
        try:
            status, error = self.error(gw.speak("bad"))
            self.assertEqual((status, error["code"]), (500, "engine_error"))
            self.assertIn("speech synthesis engine failed", error["message"])
            self.assertEqual(gw.speak("good")[0], 200)
        finally:
            gw.close()

    def test_the_key_is_checked(self):
        status, _headers, _raw = self.gw.call("/v1/audio/speech", json.dumps({"model": MODEL_ID, "input": "x"}).encode(), key="wrong")
        self.assertEqual(status, 401)

    def test_a_model_name_that_is_not_this_one_is_refused(self):
        status, error = self.error(self.gw.speak("hello", model="some-other-model"))
        self.assertEqual(status, 404, error)

    def test_the_model_card_says_what_it_takes_and_gives(self):
        _status, _headers, body = self.gw.json("/v1/models", method="GET")
        card = next(item for item in body["data"] if item["id"] == MODEL_ID)
        self.assertEqual(card["capabilities"], ["speech"])
        self.assertEqual(card["input_modalities"], ["text"])
        self.assertEqual(card["speech"]["sample_rate"], 24000)
        self.assertEqual(card["speech"]["speakers"], INFO["speakers"])
        self.assertEqual(card["speech"]["languages"], INFO["languages"])
        self.assertEqual(card["speech"]["formats"], list(speech_engine.FORMATS))
        self.assertEqual(card["speech"]["voices"]["alloy"], "ryan")
        self.assertEqual(card["speech"]["max_input_chars"], 4096)
        self.assertEqual(card["speech"]["device"], "cpu")

    def test_health_reports_the_engine(self):
        _status, _headers, body = self.gw.json("/health", method="GET")
        self.assertEqual(body["speech"]["task"], "speech")

    def test_the_other_routes_say_where_to_go(self):
        status, _headers, body = self.gw.json("/v1/chat/completions", {"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 400, body)
        self.assertIn("speech synthesis model", body["error"]["message"])
        self.assertIn("/v1/audio/speech", body["error"]["message"])
        for path, what in (("/v1/images/generations", "generate images"), ("/v1/audio/music", "make music"),
                           ("/v1/embeddings", "make embeddings"), ("/v1/rerank", "rank documents")):
            with self.subTest(path=path):
                body_in = {"model": MODEL_ID, "prompt": "p", "input": "i", "query": "q", "documents": ["d"]}
                status, _headers, body = self.gw.json(path, body_in)
                self.assertEqual(status, 400, body)
                self.assertIn(f"does not {what}", body["error"]["message"])
                self.assertIn("speaks text", body["error"]["message"])


class DeadEngineTest(unittest.TestCase):
    def test_a_dead_engine_is_a_503_not_a_hang(self):
        gw = GatewayHarness(TTS_STUB_DIE_ON="boom")
        try:
            status, _headers, raw = gw.speak("boom")
            self.assertEqual(status, 500, raw)
            status, _headers, raw = gw.speak("after")
            self.assertEqual((status, json.loads(raw)["error"]["code"]), (503, "engine_unavailable"))
        finally:
            gw.close()


class OtherModelsRefuseTest(unittest.TestCase):
    def test_a_speech_request_to_a_non_speech_server_is_refused_with_its_purpose(self):
        from embedding_engine import EmbeddingEngine
        engine = EmbeddingEngine(str(TOOLS / "embedding_stub.py"), "model-dir", env=stub_env(), on_log=lambda _m: None)
        server = APIServer(("127.0.0.1", 0), engine, "qwen3-embedding-0.6b", KEY, 16, max_queue=4, queue_timeout=10)
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        try:
            req = Request(f"http://127.0.0.1:{server.server_port}/v1/audio/speech",
                          data=json.dumps({"model": "qwen3-embedding-0.6b", "input": "hi"}).encode(),
                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(req, timeout=30)
            self.assertEqual(caught.exception.code, 400)
            self.assertIn("does not speak", json.loads(caught.exception.read())["error"]["message"])
        finally:
            server.scheduler.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            engine.close(timeout=5)


class FamilyTest(unittest.TestCase):
    def checkpoint(self, root, hidden, kind="custom_voice"):
        root = Path(root)
        root.mkdir(parents=True)
        (root / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3TTSForConditionalGeneration"], "model_type": "qwen3_tts", "tts_model_type": kind,
            "talker_config": {"hidden_size": hidden, "num_hidden_layers": 28, "model_type": "qwen3_tts_talker"}}))
        return root

    def test_the_checkpoints_are_told_apart_by_the_width_of_the_talker(self):
        for hidden, model_id, scale in ((1024, "qwen3-tts-0.6b", "0.6B"), (2048, "qwen3-tts-1.7b", "1.7B")):
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as tmp:
                resolved = resolve_model(self.checkpoint(Path(tmp) / "m", hidden))
                family = resolved.descriptor
                self.assertEqual((family.id, family.modality, family.engine_artifact), ("qwen3_tts", "speech", "qwen3tts"))
                self.assertEqual(default_model_id(resolved), model_id)
                self.assertEqual(display_for(resolved), ("Qwen3-TTS", scale))
                self.assertEqual(resolved.family_config["hidden_size"], hidden)

    def test_a_config_without_the_talker_section_is_read_as_it_stands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m"
            root.mkdir()
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3_tts"}))
            resolved = resolve_model(root)
            self.assertEqual((resolved.descriptor.id, default_model_id(resolved)), ("qwen3_tts", "qwen3-tts"))
            self.assertEqual(resolved.family_config, resolved.config)


if __name__ == "__main__":
    unittest.main()
