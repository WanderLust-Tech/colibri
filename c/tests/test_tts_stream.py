"""Streaming speech: the engine wrapper's generator, the request rules, and POST /v1/audio/speech with `stream_format` "sse" or `stream: true`, end
to end against the stub engine (tools/tts_stub.py), which sends the samples of a request as three PART lines and then AUDIO.
"""
import base64
import json
import os
import struct
import sys
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))

import speech_engine  # noqa: E402
from embedding_engine import EmbeddingEngineError, EmbeddingRequestError  # noqa: E402
from speech_engine import SpeechEngine, speech_request  # noqa: E402
from test_tts_gateway import KEY, MODEL_ID, STUB, GatewayHarness, stub_env  # noqa: E402

STREAMING_INFO = {"speakers": ["ryan"], "languages": ["english"], "stream": True}


def echo_of_pcm(pcm):
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    return bytes(samples[1:1 + samples[0]]).decode()


def post(gw, body, key=KEY):
    """The raw HTTP exchange: (status, headers, the body read to the end of the connection)."""
    connection = HTTPConnection("127.0.0.1", gw.port, timeout=30)
    try:
        connection.request("POST", "/v1/audio/speech", json.dumps(dict(body, model=MODEL_ID)), {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        response = connection.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
    finally:
        connection.close()


def events(raw):
    out = []
    for block in raw.decode().split("\n\n"):
        if block.startswith("data: "):
            out.append(json.loads(block[6:]))
    return out


class RulesTest(unittest.TestCase):
    def test_sse_and_stream_are_the_stream_modes(self):
        self.assertEqual(speech_request({"input": "x", "stream_format": "sse"}, STREAMING_INFO)["stream"], "sse")
        self.assertEqual(speech_request({"input": "x", "stream_format": "sse", "response_format": "pcm"}, STREAMING_INFO)["stream"], "sse")
        self.assertEqual(speech_request({"input": "x", "stream": True}, STREAMING_INFO)["stream"], "audio")
        self.assertEqual(speech_request({"input": "x", "stream": True, "response_format": "pcm"}, STREAMING_INFO)["stream"], "audio")
        self.assertIsNone(speech_request({"input": "x", "stream_format": "audio"}, STREAMING_INFO)["stream"])
        self.assertIsNone(speech_request({"input": "x", "stream": False}, STREAMING_INFO)["stream"])

    def test_what_cannot_be_cut_into_pieces_is_refused_with_the_field(self):
        for body, param in (({"stream": True, "response_format": "mp3"}, "response_format"), ({"stream_format": "sse", "response_format": "flac"}, "response_format"),
                            ({"stream": "yes"}, "stream"), ({"stream_format": "bogus"}, "stream_format")):
            with self.assertRaises(EmbeddingRequestError, msg=body) as caught:
                speech_request(dict(body, input="x"), STREAMING_INFO)
            self.assertEqual(caught.exception.param, param)

    def test_an_engine_that_does_not_say_it_streams_refuses(self):
        with self.assertRaises(EmbeddingRequestError) as caught:
            speech_request({"input": "x", "stream": True}, {"speakers": ["ryan"]})
        self.assertEqual(caught.exception.param, "stream")


class EngineStreamTest(unittest.TestCase):
    def engine(self, **env):
        engine = SpeechEngine(STUB, "model-dir", env=stub_env(**env), on_log=lambda _m: None)
        self.addCleanup(engine.close)
        return engine

    def test_the_parts_come_in_order_and_make_the_whole(self):
        engine = self.engine()
        items = list(engine.speak_stream("hello there", speaker="ryan", language="english", seed=3))
        parts = [value for kind, value in items if kind == "part"]
        self.assertEqual([kind for kind, _ in items], ["part", "part", "part", "done"])
        pcm = b"".join(parts)
        self.assertEqual(echo_of_pcm(pcm), "ryan|english|3|0|0")
        done = items[-1][1]
        self.assertEqual((done["sample_rate"], done["samples"] * 2), (24000, len(pcm)))
        whole = engine.speak("hello there", speaker="ryan", language="english", seed=3)["wav"]
        self.assertEqual(whole[44:], pcm)                                   # the same samples as the one-piece answer

    def test_an_error_after_the_first_part_is_an_exception_and_the_engine_goes_on(self):
        engine = self.engine(TTS_STUB_FAIL_AFTER_PARTS=1)
        seen = []
        with self.assertRaises(EmbeddingEngineError) as caught:
            for kind, value in engine.speak_stream("hello there"):
                seen.append(kind)
        self.assertEqual(seen, ["part"])
        self.assertIn("while the speech was going out", str(caught.exception))
        self.assertTrue(engine.alive)

    def test_a_caller_that_stops_early_leaves_the_engine_in_step(self):
        engine = self.engine(TTS_STUB_PART_DELAY=0.05)
        generator = engine.speak_stream("first request")
        self.assertEqual(next(generator)[0], "part")
        generator.close()                                                   # the engine is still working on the rest of it
        result = engine.speak("second request")
        self.assertEqual(echo_of_pcm(result["wav"][44:]), "|auto|0|0|0")
        self.assertEqual(result["frames"], len("second request") * 1200 // 1920)

    def test_an_engine_that_cannot_stream_says_so(self):
        engine = self.engine(TTS_STUB_NOSTREAM=1)
        with self.assertRaises(EmbeddingEngineError):
            list(engine.speak_stream("hello"))
        self.assertEqual(engine.info.get("stream"), False)


class EndpointStreamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def test_sse_sends_deltas_of_pcm_and_a_done_event(self):
        status, headers, raw = post(self.gw, {"input": "hello there", "voice": "vivian", "stream_format": "sse"})
        self.assertEqual(status, 200, raw[:200])
        self.assertEqual(headers["content-type"], "text/event-stream")
        self.assertEqual(headers["x-colibri-sample-rate"], "24000")
        self.assertNotIn("content-length", headers)
        sent = events(raw)
        self.assertEqual([e["type"] for e in sent], ["speech.audio.delta"] * 3 + ["speech.audio.done"])
        pcm = b"".join(base64.b64decode(e["audio"]) for e in sent[:3])
        self.assertEqual(echo_of_pcm(pcm), "vivian|auto|0|0|0")
        self.assertAlmostEqual(sent[-1]["audio_seconds"], len("hello there") * 0.05, places=2)
        self.assertEqual((sent[-1]["sample_rate"], sent[-1]["truncated"]), (24000, False))

    def test_stream_true_sends_a_wav_with_an_open_header_then_the_samples(self):
        status, headers, raw = post(self.gw, {"input": "hello there", "stream": True})
        self.assertEqual((status, headers["content-type"]), (200, "audio/wav"))
        self.assertEqual((raw[:4], raw[8:16]), (b"RIFF", b"WAVEfmt "))
        self.assertEqual(struct.unpack_from("<I", raw, 4)[0], 0xFFFFFFFF)               # length not known yet
        self.assertEqual(struct.unpack_from("<IHHIIHH", raw, 16), (16, 1, 1, 24000, 48000, 2, 16))
        self.assertEqual(echo_of_pcm(raw[44:]), "ryan|auto|0|0|0")

    def test_stream_true_with_pcm_is_raw_samples(self):
        status, headers, raw = post(self.gw, {"input": "hello there", "stream": True, "response_format": "pcm"})
        self.assertEqual((status, headers["content-type"]), (200, "audio/pcm"))
        self.assertEqual(echo_of_pcm(raw), "ryan|auto|0|0|0")

    def test_the_stream_is_the_same_speech_as_the_one_piece_answer(self):
        _s, _h, streamed = post(self.gw, {"input": "same words", "seed": 9, "stream": True, "response_format": "pcm"})
        _s, _h, whole = post(self.gw, {"input": "same words", "seed": 9})
        self.assertEqual(streamed, whole[44:])

    def test_a_request_the_engine_refuses_is_a_json_error_not_a_stream(self):
        gw = GatewayHarness(TTS_STUB_FAIL_ON="bad")
        self.addCleanup(gw.close)
        status, headers, raw = post(gw, {"input": "bad", "stream_format": "sse"})
        self.assertEqual((status, headers["content-type"]), (500, "application/json"))
        self.assertEqual(json.loads(raw)["error"]["type"], "server_error")

    def test_a_failure_after_the_first_part_is_an_error_event(self):
        gw = GatewayHarness(TTS_STUB_FAIL_AFTER_PARTS=1)
        self.addCleanup(gw.close)
        status, _headers, raw = post(gw, {"input": "hello there", "stream_format": "sse"})
        self.assertEqual(status, 200)
        sent = events(raw)
        self.assertEqual([e["type"] for e in sent], ["speech.audio.delta", "error"])
        self.assertIn("graphics card", sent[-1]["error"]["message"])
        status, _headers, raw = post(gw, {"input": "hello again", "stream": True, "response_format": "pcm"})   # and the engine is fine afterwards
        self.assertEqual(status, 200)

    def test_a_client_that_hangs_up_does_not_break_the_next_request(self):
        gw = GatewayHarness(TTS_STUB_PART_DELAY=0.05)
        self.addCleanup(gw.close)
        connection = HTTPConnection("127.0.0.1", gw.port, timeout=30)
        connection.request("POST", "/v1/audio/speech", json.dumps({"model": MODEL_ID, "input": "hello there", "stream": True, "response_format": "pcm"}),
                           {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        response = connection.getresponse()
        response.read(8)
        connection.close()                                                  # gone after the first bytes
        status, _headers, raw = post(gw, {"input": "after", "response_format": "pcm"})
        self.assertEqual(status, 200)
        self.assertEqual(echo_of_pcm(raw), "ryan|auto|0|0|0")

    def test_the_card_says_it_streams(self):
        card = next(item for item in self.gw.json("/v1/models", method="GET")[2]["data"] if item["id"] == MODEL_ID)
        self.assertTrue(card["speech"]["stream"])
        gw = GatewayHarness(TTS_STUB_NOSTREAM=1)
        self.addCleanup(gw.close)
        card = next(item for item in gw.json("/v1/models", method="GET")[2]["data"] if item["id"] == MODEL_ID)
        self.assertFalse(card["speech"]["stream"])
        status, _headers, raw = post(gw, {"input": "x", "stream": True})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["error"]["param"], "stream")

    def test_a_voice_that_does_not_exist_is_refused_before_anything_is_sent(self):
        status, _headers, raw = post(self.gw, {"input": "x", "voice": "nobody", "stream": True})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["error"]["param"], "voice")


if __name__ == "__main__":
    unittest.main()
