"""Voice cloning in the gateway, the router and `coli`: the request rules, the saved voices, the HTTP routes, end to end against the stub engine.

The stub (tools/tts_stub.py, TTS_STUB_KIND=base) is a cloning model: it needs `ref_wav` and its echo ends with
`|ref=<bytes of the reference file>|<reference text>|<x_vector_only 0 or 1>`, which is how these tests see what the gateway handed the engine.
A reference of 2 s at 16 kHz becomes a 24 kHz mono 16-bit WAV: 44 + 2 * 24000 * 2 = 96044 bytes.
"""
import base64
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))

import speech_cli  # noqa: E402
import voices  # noqa: E402
from embedding_engine import EmbeddingRequestError  # noqa: E402
from speech_engine import speech_request  # noqa: E402
from test_asr_gateway import wav_bytes  # noqa: E402
from test_router import Rig, entry  # noqa: E402
from test_speech_cli import Plain, clients, fetcher  # noqa: E402
import test_tts_gateway  # noqa: E402
from test_tts_gateway import GatewayHarness, echo_of  # noqa: E402

CLONE_INFO = {"clone": True, "speakers": [], "languages": ["english", "french"]}
REF_BYTES = 96044


def b64(data):
    return base64.b64encode(data).decode()


class VoiceDir:
    """A scratch COLI_VOICES for one test."""

    def use(self, test):
        self.tmp = tempfile.TemporaryDirectory()
        test.addCleanup(self.tmp.cleanup)
        old = os.environ.get("COLI_VOICES")
        os.environ["COLI_VOICES"] = self.tmp.name

        def restore():
            if old is None:
                os.environ.pop("COLI_VOICES", None)
            else:
                os.environ["COLI_VOICES"] = old
        test.addCleanup(restore)
        return self.tmp.name


class RequestRulesTest(unittest.TestCase):
    def test_a_cloning_model_takes_a_saved_voice_by_name_and_has_no_default_speaker(self):
        request = speech_request({"input": "hi", "voice": "My-Voice"}, CLONE_INFO)
        self.assertEqual((request["voice"], request["speaker"], request["reference"]), ("my-voice", "", None))
        request = speech_request({"input": "hi"}, CLONE_INFO)
        self.assertEqual((request["voice"], request["speaker"]), (None, ""))

    def test_an_inline_reference_is_decoded_and_its_text_kept(self):
        request = speech_request({"input": "hi", "reference_audio": "data:audio/wav;base64," + b64(b"RIFFabc"), "reference_text": "  what it says "}, CLONE_INFO)
        self.assertEqual((request["reference"], request["reference_text"], request["x_vector_only"]), (b"RIFFabc", "what it says", False))
        request = speech_request({"input": "hi", "reference_audio": b64(b"xyz"), "x_vector_only": True}, CLONE_INFO)
        self.assertEqual((request["reference"], request["reference_text"], request["x_vector_only"]), (b"xyz", None, True))

    def test_a_reference_is_refused_by_a_model_that_does_not_clone(self):
        with self.assertRaises(EmbeddingRequestError) as caught:
            speech_request({"input": "hi", "reference_audio": b64(b"xyz")}, test_tts_gateway.INFO)
        self.assertEqual(caught.exception.param, "reference_audio")

    def test_bad_reference_fields_are_refused_with_the_field(self):
        for body, param in (({"reference_audio": "!!not base64!!"}, "reference_audio"), ({"reference_audio": 5}, "reference_audio"),
                            ({"reference_audio": b64(b"x"), "reference_text": 3}, "reference_text"),
                            ({"reference_audio": b64(b"x"), "x_vector_only": "yes"}, "x_vector_only")):
            with self.assertRaises(EmbeddingRequestError, msg=body) as caught:
                speech_request(dict(body, input="hi"), CLONE_INFO)
            self.assertEqual(caught.exception.param, param)

    def test_the_openai_names_are_not_speakers_of_a_cloning_model(self):
        self.assertEqual(speech_request({"input": "hi", "voice": "alloy"}, CLONE_INFO)["voice"], "alloy")     # a saved voice may be called that


class VoiceStoreTest(unittest.TestCase):
    def setUp(self):
        VoiceDir().use(self)

    def test_a_voice_is_saved_listed_found_replaced_and_forgotten(self):
        info = voices.add("Ana", wav_bytes(2.0), "hello there")
        self.assertEqual((info["name"], info["text"], info["replaced"]), ("ana", "hello there", False))
        self.assertAlmostEqual(info["seconds"], 2.0, places=2)
        self.assertEqual(os.path.getsize(info["path"]), REF_BYTES)                         # 24 kHz mono 16-bit
        self.assertEqual([v["name"] for v in voices.list_voices()], ["ana"])
        self.assertEqual(voices.get("ANA")["text"], "hello there")
        self.assertTrue(voices.add("ana", wav_bytes(3.0), "")["replaced"])
        self.assertEqual((voices.get("ana")["text"], voices.payload(voices.get("ana"))["mode"]), ("", "x-vector"))
        self.assertTrue(voices.delete("ana"))
        self.assertEqual((voices.list_voices(), voices.get("ana"), voices.delete("ana")), ([], None, False))

    def test_names_are_checked(self):
        for bad in ("", " ", "-a", "a b", "A/../B", "x" * 65, None, 5):
            with self.assertRaises(voices.VoiceError, msg=repr(bad)) as caught:
                voices.add(bad, wav_bytes(2.0))
            self.assertEqual(caught.exception.param, "name")
        self.assertIsNone(voices.get("../etc"))

    def test_a_clip_that_is_too_short_or_not_audio_is_refused(self):
        with self.assertRaises(voices.VoiceError) as caught:
            voices.add("short", wav_bytes(0.4))
        self.assertIn("at least", str(caught.exception))
        with self.assertRaises(voices.VoiceError):
            voices.add("junk", b"this is not audio at all, not even close to it")
        self.assertEqual(voices.list_voices(), [])


class CloneEndpointTest(unittest.TestCase):
    def setUp(self):
        VoiceDir().use(self)
        self.gw = GatewayHarness(TTS_STUB_KIND="base")
        self.addCleanup(self.gw.close)

    def speak(self, **fields):
        status, headers, raw = self.gw.speak(**fields)
        return status, (echo_of(raw) if status == 200 else json.loads(raw)["error"])

    def test_an_inline_reference_with_its_text_is_cloned_in_context(self):
        status, echo = self.speak(reference_audio=b64(wav_bytes(2.0)), reference_text="and so, my fellow Americans")
        self.assertEqual(status, 200, echo)
        self.assertTrue(echo.endswith(f"|ref={REF_BYTES}|and so, my fellow Americans|0"), echo)

    def test_an_inline_reference_without_its_text_is_the_speaker_embedding_alone(self):
        status, echo = self.speak(reference_audio=b64(wav_bytes(2.0)))
        self.assertEqual(status, 200, echo)
        self.assertTrue(echo.endswith(f"|ref={REF_BYTES}||1"), echo)
        status, echo = self.speak(reference_audio=b64(wav_bytes(2.0)), reference_text="words", x_vector_only=True)
        self.assertTrue(echo.endswith(f"|ref={REF_BYTES}||1"), echo)                        # x_vector_only wins over the text

    def test_a_saved_voice_brings_its_clip_and_its_text(self):
        voices.add("ana", wav_bytes(2.0), "hello there")
        status, echo = self.speak(voice="Ana", language="en", seed=3)
        self.assertEqual(status, 200, echo)
        self.assertEqual(echo, f"|english|3|0|0|ref={REF_BYTES}|hello there|0")
        voices.add("mute", wav_bytes(2.0), "")
        self.assertTrue(self.speak(voice="mute")[1].endswith(f"|ref={REF_BYTES}||1"))

    def test_a_request_text_overrides_the_saved_text_for_that_request(self):
        voices.add("ana", wav_bytes(2.0), "hello there")
        self.assertTrue(self.speak(voice="ana", reference_text="other words")[1].endswith("|other words|0"))

    def test_no_voice_and_no_reference_says_what_to_send_and_what_is_saved(self):
        voices.add("ana", wav_bytes(2.0), "hi")
        status, error = self.speak()
        self.assertEqual((status, error["param"]), (400, "voice"))
        self.assertIn("ana", error["message"])
        status, error = self.speak(voice="nobody")
        self.assertEqual((status, error["param"]), (400, "voice"))
        self.assertIn("nobody", error["message"])

    def test_a_reference_that_is_not_audio_is_a_400_on_the_field(self):
        status, error = self.speak(reference_audio=b64(b"definitely not audio, sorry about that"))
        self.assertEqual((status, error["param"]), (400, "reference_audio"))

    def test_the_card_says_it_clones_and_lists_the_saved_voices(self):
        voices.add("ana", wav_bytes(2.0), "hi")
        card = next(item for item in self.gw.json("/v1/models", method="GET")[2]["data"] if item["id"] == test_tts_gateway.MODEL_ID)
        self.assertEqual((card["speech"]["clone"], card["speech"]["kind"], card["speech"]["speakers"], card["speech"]["saved_voices"]),
                         (True, "base", [], ["ana"]))

    def test_a_model_that_does_not_clone_refuses_a_reference(self):
        gw = GatewayHarness()
        self.addCleanup(gw.close)
        status, _headers, raw = gw.speak(reference_audio=b64(wav_bytes(2.0)))
        self.assertEqual((status, json.loads(raw)["error"]["param"]), (400, "reference_audio"))
        card = next(item for item in gw.json("/v1/models", method="GET")[2]["data"] if item["id"] == test_tts_gateway.MODEL_ID)
        self.assertFalse(card["speech"]["clone"])


class RouterVoicesTest(unittest.TestCase):
    def setUp(self):
        VoiceDir().use(self)
        self.rig = Rig(self, [entry("text-a")])

    def test_a_voice_is_saved_listed_and_deleted_over_http(self):
        status, _, _, body = self.rig.call("POST", "/v1/audio/voices", {"name": "Ana", "audio": b64(wav_bytes(2.0)), "text": "hello there"})
        self.assertEqual((status, body["status"], body["name"], body["mode"]), (200, "created", "ana", "in-context"))
        status, _, _, body = self.rig.call("POST", "/v1/audio/voices", {"name": "ana", "audio": "data:audio/wav;base64," + b64(wav_bytes(3.0))})
        self.assertEqual((status, body["status"], body["mode"]), (200, "replaced", "x-vector"))
        status, _, _, body = self.rig.call("GET", "/v1/audio/voices")
        self.assertEqual((status, [v["name"] for v in body["data"]], "path" in body), (200, ["ana"], True))
        self.assertNotIn("path", body["data"][0])                                           # no file path in a voice
        status, _, _, body = self.rig.call("POST", "/v1/audio/voices/delete", {"name": "ana"})
        self.assertEqual((status, body), (200, {"status": "deleted", "name": "ana"}))
        self.assertEqual(self.rig.call("GET", "/v1/audio/voices")[3]["data"], [])
        self.assertEqual(self.rig.call("POST", "/v1/audio/voices/delete", {"name": "ana"})[0], 404)

    def test_bad_requests_say_which_field(self):
        for body, param in (({"name": "a b", "audio": b64(wav_bytes(2.0))}, "name"), ({"name": "ok"}, "audio"),
                            ({"name": "ok", "audio": "!!"}, "audio"), ({"name": "ok", "audio": b64(wav_bytes(2.0)), "text": 5}, "text"),
                            ({"name": "ok", "audio": b64(wav_bytes(0.3))}, "audio")):
            status, _, _, answer = self.rig.call("POST", "/v1/audio/voices", body)
            self.assertEqual((status, answer["error"]["param"]), (400, param), body)

    def test_it_wants_the_key_when_there_is_one(self):
        rig = Rig(self, [entry("text-a")], api_key="secret")
        self.assertEqual(rig.call("GET", "/v1/audio/voices")[0], 401)
        self.assertEqual(rig.call("GET", "/v1/audio/voices", headers={"Authorization": "Bearer secret"})[0], 200)


class CommandTest(unittest.TestCase):
    """`coli voice` against a router's endpoints, and `coli speak --reference` / `--voice NAME` against a cloning gateway."""

    def setUp(self):
        VoiceDir().use(self)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rig = Rig(self, [entry("text-a")])
        self.api, self.upload = clients(self.rig.port, key=None)

    def clip(self, name="clip.wav", seconds=2.0):
        path = os.path.join(self.tmp.name, name)
        Path(path).write_bytes(wav_bytes(seconds))
        return path

    def voice(self, action, name=None, file=None, **kw):
        out = io.StringIO()
        base = dict(action=action, name=name, file=file, text=None, no_text=False, json=False)
        base.update(kw)
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = speech_cli.voice(SimpleNamespace(**base), self.api, self.upload, Plain)
        return code, out.getvalue()

    def test_add_list_and_remove(self):
        code, text = self.voice("add", "ana", self.clip(), text="hello there")
        self.assertEqual(code, 0, text)
        self.assertIn("created ana", text)
        code, text = self.voice("list")
        self.assertEqual(code, 0)
        self.assertIn("ana", text)
        self.assertIn("in-context", text)
        self.assertIn("hello there", text)
        self.assertEqual(self.voice("remove", "ana")[0], 0)
        self.assertIn("no saved voices", self.voice("list")[1])
        self.assertEqual(self.voice("remove", "ana")[0], 1)

    def test_add_without_words_and_without_a_recogniser_saves_the_embedding_alone(self):
        code, text = self.voice("add", "mute", self.clip())                                  # the rig has no speech recognition model
        self.assertEqual(code, 0, text)
        self.assertIn("x-vector", text)
        self.assertIn("without the words", text)

    def test_usage_and_bad_input(self):
        self.assertEqual(self.voice("add", "x", None)[0], 1)
        self.assertEqual(self.voice("add", "Bad Name", self.clip(), text="t")[0], 1)
        self.assertEqual(self.voice("add", "ok", os.path.join(self.tmp.name, "missing.wav"), text="t")[0], 1)

    def test_speak_with_a_reference_clip_and_with_a_saved_voice(self):
        gw = GatewayHarness(TTS_STUB_KIND="base")
        self.addCleanup(gw.close)
        api, _upload = clients(gw.port)
        voices.add("ana", wav_bytes(2.0), "hello there")
        out = os.path.join(self.tmp.name, "out.wav")

        def speak(**kw):
            base = dict(text=["hello", "there"], voice=None, instruct=None, language=None, out=out, format=None, seed=None, model=None,
                        reference=None, reference_text=None, x_vector=False)
            base.update(kw)
            buffer = io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(io.StringIO()):
                code = speech_cli.speak(SimpleNamespace(**base), api, fetcher(gw.port), Plain)
            return code, buffer.getvalue()

        code, text = speak(reference=self.clip(), reference_text="the words")
        self.assertEqual(code, 0, text)
        self.assertTrue(echo_of(Path(out).read_bytes()).endswith(f"|ref={REF_BYTES}|the words|0"))
        code, text = speak(reference=self.clip(), x_vector=True)
        self.assertTrue(echo_of(Path(out).read_bytes()).endswith(f"|ref={REF_BYTES}||1"))
        code, text = speak(voice="ana")
        self.assertEqual(code, 0, text)
        self.assertTrue(echo_of(Path(out).read_bytes()).endswith(f"|ref={REF_BYTES}|hello there|0"))
        self.assertEqual(speak(reference=os.path.join(self.tmp.name, "missing.wav"))[0], 1)


if __name__ == "__main__":
    unittest.main()
