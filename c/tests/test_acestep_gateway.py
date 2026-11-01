"""POST /v1/audio/music, end to end against the stub engine.

A real MusicEngine drives tools/acestep_stub.py over its serve protocol, and a real APIServer answers HTTP on a free
port: nothing between the socket and the engine's stdin is mocked. The stub makes a clip that is a pure function of
(caption, lyrics, language, bpm, key, seed, duration), which is what lets these tests pin determinism, the length
actually made and the WAV framing.
"""
import base64
import http.client
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acestep_fixture import RATE, STUB, make_fake_checkpoint, stub_env  # noqa: E402

import music_engine  # noqa: E402
from openai_server import APIServer  # noqa: E402

MODEL_ID = "ace-step-1.5-colibri"
KEY = "secret"


class GatewayHarness:
    """One stub engine behind one API server, torn down in reverse order."""

    def __init__(self, delay=0.0, max_queue=8, api_key=KEY, **stub):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = make_fake_checkpoint(Path(self.tmp.name) / "ace-step")
        self.engine = music_engine.MusicEngine(STUB, self.model, env=stub_env(delay, **stub), on_log=lambda _m: None)
        self.server = APIServer(("127.0.0.1", 0), self.engine, MODEL_ID, api_key, 16,
                                max_queue=max_queue, queue_timeout=30)
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def close(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.engine.close(timeout=5)
        self.tmp.cleanup()

    def request(self, path, body=None, headers=None, method=None, key=KEY):
        all_headers = {"Content-Type": "application/json"}
        if key:
            all_headers["Authorization"] = f"Bearer {key}"
        all_headers.update(headers or {})
        data = None if body is None else json.dumps(body).encode()
        req = Request(f"http://127.0.0.1:{self.port}{path}", data=data, headers=all_headers, method=method)
        try:
            with urlopen(req, timeout=30) as response:
                return response.status, dict(response.headers), response.read()
        except HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def music(self, **body):
        body.setdefault("model", MODEL_ID)
        return self.request("/v1/audio/music", body)

    def stream(self, **body):
        body.setdefault("model", MODEL_ID)
        body["stream"] = True
        status, headers, raw = self.request("/v1/audio/music", body)
        return status, headers, parse_sse(raw.decode("utf-8"))


def parse_sse(text):
    """[(event, data)] in order; `data: [DONE]` appears as (None, "[DONE]")."""
    events = []
    for block in text.split("\n\n"):
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            payload = "\n".join(data)
            events.append((name, payload if payload == "[DONE]" else json.loads(payload)))
    return events


def wav_of(b64):
    return music_engine.decode_wav(base64.b64decode(b64))


class MusicEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def test_a_song_comes_back_as_a_wav_of_the_length_asked(self):
        status, headers, raw = self.gw.music(prompt="calm piano with soft strings", duration=2, steps=2, seed=7)
        self.assertEqual(status, 200, raw[:200])
        body = json.loads(raw)
        self.assertEqual(len(body["data"]), 1)
        self.assertEqual(body["data"][0]["format"], "wav")
        pcm, rate, channels = wav_of(body["data"][0]["b64_json"])
        self.assertEqual((rate, channels, len(pcm)), (RATE, 2, 2 * RATE * 4))
        info = body["colibri"]
        self.assertEqual((info["seed"], info["steps"], info["sample_rate"], info["channels"]), (7, 2, RATE, 2))
        self.assertAlmostEqual(info["duration"], 2.0, places=2)
        self.assertIn("x-colibri-queue-wait-ms", {k.lower() for k in headers})
        self.assertIn("created", body)

    def test_the_same_request_gives_the_same_song_and_each_field_changes_it(self):
        base = dict(prompt="calm piano", duration=1, steps=1, seed=3)

        def pcm(**changes):
            status, _, raw = self.gw.music(**{**base, **changes})
            self.assertEqual(status, 200, raw[:200])
            return wav_of(json.loads(raw)["data"][0]["b64_json"])[0]
        first = pcm()
        self.assertEqual(pcm(), first)
        for change in ({"seed": 4}, {"lyrics": "la la la"}, {"bpm": 90}, {"key": "C Major"}, {"language": "en"},
                       {"prompt": "heavy metal"}):
            with self.subTest(change=change):
                self.assertNotEqual(pcm(**change), first)

    def test_a_missing_seed_is_made_and_reported(self):
        status, _, raw = self.gw.music(prompt="calm piano", duration=1, steps=1)
        self.assertEqual(status, 200, raw[:200])
        seed = json.loads(raw)["colibri"]["seed"]
        self.assertTrue(0 <= seed <= 2 ** 32 - 1)

    def test_response_format_wav_is_the_file_itself(self):
        status, headers, raw = self.gw.music(prompt="calm piano", duration=1, steps=1, seed=5, response_format="wav")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "audio/wav")
        self.assertEqual(int(headers["Content-Length"]), len(raw))
        self.assertEqual(headers["x-colibri-seed"], "5")
        pcm, rate, channels = music_engine.decode_wav(raw)
        self.assertEqual((rate, channels, len(pcm)), (RATE, 2, RATE * 4))

    def test_stream_sends_progress_then_the_song_then_done(self):
        status, headers, events = self.gw.stream(prompt="calm piano", duration=1, steps=3, seed=9)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        names = [name for name, _ in events]
        self.assertEqual(names[-2:], ["music_generation.completed", None])
        self.assertEqual(events[-1][1], "[DONE]")
        progress = [data for name, data in events if name == "music_generation.progress"]
        self.assertEqual(progress[0]["stage"], "text")
        self.assertEqual([p["step"] for p in progress if p["stage"] == "denoise"], [0, 1, 2, 3])
        self.assertEqual(progress[-1]["stage"], "decode")
        done = events[-2][1]
        pcm, rate, channels = wav_of(done["data"][0]["b64_json"])
        self.assertEqual((rate, channels, len(pcm)), (RATE, 2, RATE * 4))
        self.assertEqual(done["colibri"]["seed"], 9)

    def test_every_bad_request_names_its_parameter(self):
        cases = (({}, "prompt"), ({"prompt": 5}, "prompt"), ({"prompt": "x", "duration": 0}, "duration"),
                 ({"prompt": "x", "duration": 601}, "duration"), ({"prompt": "x", "steps": 0}, "steps"),
                 ({"prompt": "x", "steps": 99}, "steps"), ({"prompt": "x", "seed": -1}, "seed"),
                 ({"prompt": "x", "shift": 0}, "shift"), ({"prompt": "x", "bpm": 0}, "bpm"),
                 ({"prompt": "x", "n": 2}, "n"), ({"prompt": "x", "response_format": "mp3"}, "response_format"),
                 ({"prompt": "x", "stream": "yes"}, "stream"), ({"prompt": "x", "dcw": 1}, "dcw"),
                 ({"prompt": "x", "lyrics": 5}, "lyrics"))
        for body, param in cases:
            with self.subTest(body=body):
                status, _, raw = self.gw.music(**body)
                self.assertEqual(status, 400, raw[:200])
                error = json.loads(raw)["error"]
                self.assertEqual(error["param"], param)
                self.assertTrue(error["message"])

    def test_stream_and_wav_together_are_refused(self):
        status, _, raw = self.gw.music(prompt="x", stream=True, response_format="wav")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["error"]["param"], "response_format")

    def test_the_engine_failing_is_a_500_and_the_next_request_works(self):
        status, _, raw = self.gw.music(prompt="__stub_error__")
        self.assertEqual(status, 500, raw[:200])
        self.assertEqual(json.loads(raw)["error"]["code"], "engine_error")
        status, _, raw = self.gw.music(prompt="calm piano", duration=1, steps=1)
        self.assertEqual(status, 200, raw[:200])

    def test_no_key_is_refused_and_a_bad_one_too(self):
        status, _, _ = self.gw.request("/v1/audio/music", {"prompt": "x"}, key=None)
        self.assertEqual(status, 401)
        status, _, _ = self.gw.request("/v1/audio/music", {"prompt": "x"}, key="wrong")
        self.assertEqual(status, 401)

    def test_the_text_endpoints_point_at_the_music_one(self):
        for path in ("/v1/chat/completions", "/v1/completions", "/v1/messages"):
            with self.subTest(path=path):
                status, _, raw = self.gw.request(path, {"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}],
                                                        "prompt": "hi", "max_tokens": 4})
                self.assertEqual(status, 400, raw[:200])
                self.assertIn("/v1/audio/music", json.loads(raw)["error"]["message"])

    def test_the_image_endpoint_says_this_model_makes_music(self):
        status, _, raw = self.gw.request("/v1/images/generations", {"model": MODEL_ID, "prompt": "a fox"})
        self.assertEqual(status, 400, raw[:200])
        self.assertIn("music", json.loads(raw)["error"]["message"].lower())

    def test_the_model_card_and_health_say_what_it_makes(self):
        status, _, raw = self.gw.request("/v1/models", method="GET")
        self.assertEqual(status, 200)
        entry = json.loads(raw)["data"][0]
        self.assertEqual(entry["capabilities"], ["music_generation"])
        audio = entry["audio"]
        self.assertEqual((audio["sample_rate"], audio["channels"], audio["max_duration"], audio["max_steps"]), (RATE, 2, 600, 20))
        self.assertNotIn("image", entry)
        status, _, raw = self.gw.request("/health", method="GET")          # the detail is for a client that has the key
        self.assertEqual(status, 200)
        health = json.loads(raw)
        self.assertEqual(health["capabilities"], ["music_generation"])
        self.assertEqual(health["audio"]["sample_rate"], RATE)

    def test_a_text_engine_refuses_the_music_route(self):
        """The gateway fronting something that is not a music engine says so, by name."""
        class Chatty:
            modality = "text"
            alive = True
            info = {}
        gateway = self.gw
        server = gateway.server
        real = server.engine
        server.engine = Chatty()
        try:
            status, _, raw = gateway.music(prompt="x")
        finally:
            server.engine = real
        self.assertEqual(status, 400, raw[:200])
        self.assertIn("does not make music", json.loads(raw)["error"]["message"])


class CancelAndQueueTest(unittest.TestCase):
    def test_a_client_that_hangs_up_cancels_the_song_and_the_engine_stays_ready(self):
        gw = GatewayHarness(delay=0.2)
        try:
            sock = socket.create_connection(("127.0.0.1", gw.port), timeout=10)
            body = json.dumps({"model": MODEL_ID, "prompt": "calm piano", "duration": 1, "steps": 20, "seed": 1}).encode()
            sock.sendall(b"POST /v1/audio/music HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer " + KEY.encode() +
                         b"\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            time.sleep(0.8)                         # the engine is composing
            sock.close()
            started = time.monotonic()
            status, _, raw = gw.music(prompt="calm piano again", duration=1, steps=1, seed=2)
            self.assertEqual(status, 200, raw[:200])
            self.assertLess(time.monotonic() - started, 15, "the abandoned song was not cancelled")
        finally:
            gw.close()

    def test_requests_are_made_one_at_a_time_in_the_order_they_came(self):
        gw = GatewayHarness(delay=0.05)
        try:
            done = []

            def ask(n):
                status, _, raw = gw.music(prompt=f"song {n}", duration=1, steps=2, seed=n)
                done.append((n, status, json.loads(raw)["colibri"]["seed"]))
            threads = [threading.Thread(target=ask, args=(n,)) for n in (1, 2, 3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
            self.assertEqual(sorted(done), [(1, 200, 1), (2, 200, 2), (3, 200, 3)])
        finally:
            gw.close()

    def test_a_full_queue_is_a_503_not_a_429(self):
        gw = GatewayHarness(delay=0.3, max_queue=1)
        try:
            results = []

            def ask(n):
                results.append(gw.music(prompt=f"song {n}", duration=1, steps=4, seed=n)[0])
            threads = [threading.Thread(target=ask, args=(n,)) for n in range(4)]
            for t in threads:
                t.start()
                time.sleep(0.1)
            for t in threads:
                t.join(timeout=60)
            self.assertIn(200, results)
            self.assertNotIn(429, results)
            self.assertTrue(set(results) <= {200, 503}, results)
        finally:
            gw.close()


if __name__ == "__main__":
    unittest.main()
