"""The router: one door in front of per-model workers (c/router.py).

The workers here are tests/router_fake_worker.py, real processes that answer like `coli serve` without a model, so
what is checked is the router's own behaviour: routing, loading on demand, eviction, the one-model-at-a-time gate, the
guards, streaming, and what happens when a worker fails or a client leaves."""
import http.client
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))

import router  # noqa: E402

FAKE = str(Path(__file__).resolve().parent / "router_fake_worker.py")
GB = 1 << 30


def entry(name, modality="text", size=1 * GB, reason=None):
    return {"key": "k" + name.replace("-", "")[:8], "path": tempfile.gettempdir(), "id": f"{name}-colibri",
            "name": name, "family": "fake", "modality": modality, "size_bytes": size, "verified": True,
            "current": False, "switchable": reason is None, "reason": reason}


class Rig:
    """A router on a free port over fake workers."""

    def __init__(self, test, entries, *, total=None, keep_alive=600.0, max_loaded=3, api_key=None, parallel=False,
                 max_queue=8, queue_timeout=10.0, env=None, measure=None, state_path=None, allowed_hosts=(), gpu=None,
                 settings=None):
        self.test = test
        self.tmp = tempfile.TemporaryDirectory()
        test.addCleanup(self.tmp.cleanup)
        self.log = os.path.join(self.tmp.name, "events.log")
        self.catalog = router.Catalog(entries=entries)
        self.entries = entries
        self.total = total

        def command(item, port):
            return [sys.executable, FAKE, "--id", item["id"], "--port", str(port), "--modality", item["modality"]]

        def worker_env(item):
            extra = {"FAKE_LOG": self.log}
            extra.update((env or {}).get(item["id"], {}))
            return extra

        self.pool = router.Pool(self.catalog, keep_alive=keep_alive, max_loaded=max_loaded, worker_command=command,
                                worker_env=worker_env, state=router.State(state_path),
                                memory=self.free if total else (lambda: None), measure=measure or (lambda pid, job: 0),
                                grace=0.5, queue_timeout=queue_timeout, gpu=gpu, settings=settings)
        self.gate = router.Gate(max_queue=max_queue, timeout=queue_timeout, parallel=parallel)
        self.server = router.RouterServer(("127.0.0.1", 0), self.pool, self.catalog, self.gate, api_key,
                                          ("http://127.0.0.1:1",), allowed_hosts, queue_timeout, settings=settings)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        test.addCleanup(self.stop)

    def free(self):
        used = sum(w.entry["size_bytes"] for w in self.pool.snapshot() if w.state in ("starting", "ready", "stopping"))
        return self.total - used

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.pool.close()

    def call(self, method, path, body=None, headers=None, raw=None, host=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        try:
            data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
            sent = dict(headers or {})
            if body is not None and raw is None:
                sent.setdefault("Content-Type", "application/json")
            if host:
                connection.putrequest(method, path, skip_host=True)
                connection.putheader("Host", host)
                for name, value in sent.items():
                    connection.putheader(name, value)
                connection.putheader("Content-Length", str(len(data or b"")))
                connection.endheaders(data)
            else:
                connection.request(method, path, body=data, headers=sent)
            response = connection.getresponse()
            payload = response.read()
            try:
                parsed = json.loads(payload)
            except ValueError:
                parsed = None
            return response.status, dict(response.getheaders()), payload, parsed
        finally:
            connection.close()

    def chat(self, model, **extra):
        return self.call("POST", "/v1/chat/completions", dict({"model": model, "messages": []}, **extra))

    def events(self):
        if not os.path.exists(self.log):
            return []
        out = []
        for line in open(self.log, encoding="utf-8"):
            stamp, kind, name = line.split()
            out.append((float(stamp), kind, name))
        return out

    def states(self):
        status, _, _, body = self.call("GET", "/v1/models/installed")
        self.test.assertEqual(status, 200)
        return {item["id"]: item["state"] for item in body["data"]}


def overlap(events, first, second):
    """Did a request to `first` and one to `second` run at the same time?"""
    spans = {}
    for stamp, kind, name in events:
        spans.setdefault(name, []).append((stamp, kind))
    def runs(name):
        starts = [t for t, k in spans.get(name, []) if k == "start"]
        ends = [t for t, k in spans.get(name, []) if k in ("end", "cut")]
        return list(zip(sorted(starts), sorted(ends)))
    return any(a0 < b1 and b0 < a1 for a0, a1 in runs(first) for b0, b1 in runs(second))


class RoutingTest(unittest.TestCase):
    def test_a_request_goes_to_the_worker_that_holds_its_model(self):
        rig = Rig(self, [entry("text-a"), entry("image-b", "image")])
        status, _, _, body = rig.chat("text-a-colibri")
        self.assertEqual((status, body["model"]), (200, "text-a-colibri"))
        status, _, _, body = rig.call("POST", "/v1/images/generations", {"model": "image-b-colibri", "prompt": "x"})
        self.assertEqual((status, body["model"], body["path"]), (200, "image-b-colibri", "/v1/images/generations"))

    def test_the_model_is_found_by_id_key_folder_name_or_the_id_without_the_suffix(self):
        rig = Rig(self, [entry("text-a")])
        for name in ("text-a-colibri", "TEXT-A-COLIBRI", "text-a", rig.entries[0]["key"]):
            status, _, _, body = rig.call("POST", "/v1/chat/completions", {"model": name})
            self.assertEqual(status, 200, name)
            self.assertEqual(body["model"], "text-a-colibri")

    def test_a_model_loads_when_first_asked_for_and_not_before(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")])
        before = rig.call("GET", "/v1/models")[3]["data"]
        self.assertEqual({m["id"]: m["state"] for m in before}, {"text-a-colibri": "idle", "text-b-colibri": "idle"})
        self.assertEqual(rig.states(), {"text-a-colibri": "idle", "text-b-colibri": "idle"})
        rig.chat("text-b-colibri")
        listing = rig.call("GET", "/v1/models")[3]
        self.assertEqual({m["id"]: m["state"] for m in listing["data"]},
                         {"text-a-colibri": "idle", "text-b-colibri": "loaded"})
        loaded = next(m for m in listing["data"] if m["id"] == "text-b-colibri")
        self.assertEqual(loaded["capabilities"], ["chat", "systemone"])           # what the worker announced
        self.assertEqual([c["name"] for c in listing["models"]], ["text-b-colibri"])
        self.assertEqual(rig.states(), {"text-a-colibri": "idle", "text-b-colibri": "ready"})

    def test_a_model_that_is_not_loaded_is_listed_from_what_it_said_last_time(self):
        path = os.path.join(tempfile.mkdtemp(), "router.json")
        rig = Rig(self, [entry("text-a"), entry("image-b", "image"), entry("music-m", "audio")], state_path=path,
                  measure=lambda pid, job: 1 * GB)
        listing = {m["id"]: m for m in rig.call("GET", "/v1/models")[3]["data"]}
        self.assertEqual(listing["image-b-colibri"]["capabilities"], ["image_generation"])      # from the modality
        self.assertEqual(listing["music-m-colibri"]["capabilities"], ["music_generation"])
        self.assertEqual(listing["music-m-colibri"]["audio"], {})
        rig.call("POST", "/v1/images/generations", {"model": "image-b-colibri", "prompt": "x"})
        rig.pool.unload(rig.entries[1]["key"])
        listing = {m["id"]: m for m in rig.call("GET", "/v1/models")[3]["data"]}
        self.assertEqual(listing["image-b-colibri"]["state"], "idle")
        self.assertEqual(listing["image-b-colibri"]["owned_by"], "colibri")
        self.assertEqual(listing["image-b-colibri"]["image"], {"default_width": 640})          # only its engine said so
        idle = rig.call("GET", "/v1/models/image-b-colibri")[3]
        self.assertEqual((idle["state"], idle["image"]), ("idle", {"default_width": 640}))
        self.assertEqual(listing["music-m-colibri"]["audio"], {})                                 # never ran: from the modality

    def test_the_same_worker_serves_the_next_request(self):
        rig = Rig(self, [entry("text-a")])
        first = rig.chat("text-a-colibri")[3]["pid"]
        second = rig.chat("text-a-colibri")[3]["pid"]
        self.assertEqual(first, second)

    def test_errors_have_the_workers_shape(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")])
        status, _, _, body = rig.chat("nope")
        self.assertEqual((status, body["error"]["code"], body["error"]["param"]), (404, "model_not_found", "model"))
        status, _, _, body = rig.call("POST", "/v1/chat/completions", {"messages": []})
        self.assertEqual((status, body["error"]["param"]), (400, "model"))
        self.assertEqual(rig.call("POST", "/v1/nothing", {"model": "text-a"})[0], 404)
        self.assertEqual(rig.call("GET", "/v1/models/nope")[0], 404)

    def test_a_single_model_is_used_when_the_request_names_none(self):
        rig = Rig(self, [entry("text-a")])
        status, _, _, body = rig.call("POST", "/v1/chat/completions", {"messages": []})
        self.assertEqual((status, body["model"]), (200, "text-a-colibri"))

    def test_a_model_that_cannot_start_here_says_why(self):
        rig = Rig(self, [entry("text-a", reason="the engine for this model is not built")])
        status, _, _, body = rig.chat("text-a-colibri")
        self.assertEqual((status, body["error"]["code"]), (409, "not_switchable"))
        self.assertIn("not built", body["error"]["message"])

    def test_multipart_bodies_are_routed_by_their_model_field(self):
        rig = Rig(self, [entry("text-a"), entry("music-m", "audio")])
        boundary = "xyz"
        raw = (f'--{boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nhello\r\n'
               f'--{boundary}\r\nContent-Disposition: form-data; name="src_audio"; filename="a.wav"\r\n\r\n'.encode()
               + bytes(5000) + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
               f'music-m-colibri\r\n--{boundary}--\r\n'.encode())
        status, _, _, body = rig.call("POST", "/v1/audio/music", raw=raw,
                                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        self.assertEqual((status, body["model"], body["parts"]), (200, "music-m-colibri", 3))
        self.assertEqual(body["size"], len(raw))

    def test_a_transcription_is_routed_by_its_model_field_and_may_carry_a_long_file(self):
        rig = Rig(self, [entry("text-a"), entry("music-m", "audio")])
        boundary = "xyz"
        raw = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'.encode()
               + bytes(6 << 20) + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
               f'text-a-colibri\r\n--{boundary}--\r\n'.encode())          # 6 MiB: over the 4 MiB of a chat body
        status, _, _, body = rig.call("POST", "/v1/audio/transcriptions", raw=raw,
                                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        self.assertEqual((status, body["model"]), (200, "text-a-colibri"))
        self.assertEqual(body["size"], len(raw))

    def test_the_workers_get_the_model_by_its_own_id(self):
        rig = Rig(self, [entry("text-a"), entry("music-m", "audio")])
        status, _, _, body = rig.call("POST", "/v1/chat/completions", {"model": "text-a", "keep": "me"})
        self.assertEqual((status, body["model"]), (200, "text-a-colibri"))
        raw = (b'--b\r\nContent-Disposition: form-data; name="model"\r\n\r\nmusic-m\r\n'
               b'--b\r\nContent-Disposition: form-data; name="prompt"\r\n\r\nhi\r\n--b--\r\n')
        sent = router.with_model("multipart/form-data; boundary=b", raw, "music-m-colibri")
        self.assertEqual(router.extract_model("multipart/form-data; boundary=b", sent), "music-m-colibri")
        self.assertIn(b'name="prompt"\r\n\r\nhi', sent)
        json_body = json.loads(router.with_model("application/json", '{"model":"x","a":[1,{"b":"\u00e9"}]}'.encode(), "y"))
        self.assertEqual(json_body, {"model": "y", "a": [1, {"b": "\u00e9"}]})
        same = b'{"model": "y"}'
        self.assertIs(router.with_model("application/json", same, "y"), same)

    def test_extract_model(self):
        self.assertEqual(router.extract_model("application/json", b'{"model": "m", "x": 1}'), "m")
        self.assertIsNone(router.extract_model("application/json", b'{"x": 1}'))
        self.assertIsNone(router.extract_model("application/json", b"[1]"))
        self.assertIsNone(router.extract_model("application/json", b"not json"))
        self.assertIsNone(router.extract_model("application/json", b'{"model": 5}'))
        body = b'--b\r\nContent-Disposition: form-data; name="model"\r\n\r\nm2\r\n--b--\r\n'
        self.assertEqual(router.extract_model("multipart/form-data; boundary=b", body), "m2")
        self.assertIsNone(router.extract_model("multipart/form-data; boundary=b", b"--b--\r\n"))


class ProxyTest(unittest.TestCase):
    def test_the_workers_headers_are_not_passed_on_and_the_clients_secrets_are_not_forwarded(self):
        rig = Rig(self, [entry("text-a")], api_key="sekrit")
        status, headers, _, body = rig.call("POST", "/v1/chat/completions", {"model": "text-a"},
                                            headers={"Authorization": "Bearer sekrit", "X-Test": "kept",
                                                     "Origin": "http://127.0.0.1:1"})
        self.assertEqual(status, 200)
        self.assertIsNone(body["authorization"], "the key is the router's: workers do not see it")
        self.assertIsNone(body["origin"])
        self.assertEqual(body["x_test"], "kept")
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://127.0.0.1:1")     # the router's own, not "*"
        self.assertEqual(headers["x-colibri-model"], "text-a-colibri")

    def test_the_client_gets_one_cors_header_and_it_is_the_routers(self):
        rig = Rig(self, [entry("text-a")])
        connection = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=30)
        connection.request("POST", "/v1/chat/completions", body=json.dumps({"model": "text-a"}).encode(),
                           headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1:1"})
        response = connection.getresponse()
        response.read()
        sent = [value for name, value in response.getheaders() if name.lower() == "access-control-allow-origin"]
        connection.close()
        self.assertEqual(sent, ["http://127.0.0.1:1"], "the worker's own `*` must not reach the client")

    def test_a_streamed_answer_arrives_as_it_is_made(self):
        rig = Rig(self, [entry("text-a")])
        rig.chat("text-a-colibri")                                      # load it first
        connection = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=30)
        body = json.dumps({"model": "text-a-colibri", "stream": True, "sleep": 0.5, "chunks": 3}).encode()
        began = time.monotonic()
        connection.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        first = response.read1(65536)
        first_at = time.monotonic() - began
        rest = response.read()
        total = time.monotonic() - began
        connection.close()
        self.assertEqual(response.status, 200)
        self.assertIn(b'"n": 0', first)
        self.assertLess(first_at, 0.4, "the first chunk waited for the whole answer")
        self.assertGreater(total, 1.2)
        self.assertTrue((first + rest).rstrip().endswith(b"data: [DONE]"))

    def test_a_client_that_leaves_cancels_the_workers_request(self):
        rig = Rig(self, [entry("text-a")])
        rig.chat("text-a-colibri")
        worker = rig.pool.get(rig.entries[0]["key"])
        body = json.dumps({"model": "text-a-colibri", "stream": True, "sleep": 0.2, "chunks": 40}).encode()
        sock = socket.create_connection(("127.0.0.1", rig.port))
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        got = b""
        while b'"n": 0' not in got:
            got += sock.recv(4096)
        sock.close()                                                    # gone, mid-answer
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline and (worker.inflight or not any(k == "cut" for _, k, _ in rig.events())):
            time.sleep(0.1)
        self.assertEqual(worker.inflight, 0, "the router kept the request open for nobody")
        self.assertTrue(any(k == "cut" for _, k, _ in rig.events()), "the worker kept computing for nobody")

    def test_telemetry_comes_from_the_loaded_model(self):
        rig = Rig(self, [entry("text-a")])
        self.assertEqual(rig.call("GET", "/profile")[3], {"seq": 0, "turns": []})
        rig.chat("text-a-colibri")
        self.assertEqual(rig.call("GET", "/profile")[3]["from"], "text-a-colibri")


class GuardsTest(unittest.TestCase):
    def test_the_key_is_checked_once_at_the_door(self):
        rig = Rig(self, [entry("text-a")], api_key="sekrit")
        self.assertEqual(rig.chat("text-a-colibri")[0], 401)
        self.assertEqual(rig.call("GET", "/v1/models")[0], 401)
        self.assertEqual(rig.call("GET", "/v1/models/installed")[0], 401)
        self.assertEqual(rig.call("POST", "/v1/models/switch", {"key": "x"})[0], 401)
        self.assertEqual(rig.call("POST", "/v1/chat/completions", {"model": "text-a"},
                                  headers={"Authorization": "Bearer sekrit"})[0], 200)
        self.assertEqual(rig.call("POST", "/v1/chat/completions", {"model": "text-a"},
                                  headers={"x-api-key": "sekrit"})[0], 200)
        self.assertEqual(rig.call("POST", "/v1/chat/completions", {"model": "text-a"},
                                  headers={"Authorization": "Bearer wrong"})[0], 401)

    def test_health_is_public_but_shows_nothing_without_the_key(self):
        rig = Rig(self, [entry("text-a")], api_key="sekrit")
        self.assertEqual(rig.call("GET", "/health")[3], {"status": "ok"})
        body = rig.call("GET", "/health", headers={"Authorization": "Bearer sekrit"})[3]
        self.assertEqual(body["status"], "ok")
        self.assertIn("router", body)
        self.assertEqual(rig.call("GET", "/experts")[3]["rows"], 0)

    def test_an_unknown_host_header_is_refused(self):
        rig = Rig(self, [entry("text-a")])
        self.assertEqual(rig.call("GET", "/v1/models", host="evil.example:8000")[0], 403)
        self.assertEqual(rig.call("GET", "/v1/models", host=f"127.0.0.1:{rig.port}")[0], 200)
        allowed = Rig(self, [entry("text-a")], allowed_hosts=("evil.example",))
        self.assertEqual(allowed.call("GET", "/v1/models", host="evil.example:8000")[0], 200)

    def test_a_chunked_request_is_refused(self):
        rig = Rig(self, [entry("text-a")])
        connection = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=10)
        connection.putrequest("POST", "/v1/chat/completions")
        connection.putheader("Transfer-Encoding", "chunked")
        connection.putheader("Content-Type", "application/json")
        connection.endheaders(b"5\r\nhello\r\n0\r\n\r\n")
        self.assertEqual(connection.getresponse().status, 400)
        connection.close()
        # both a length and a chunked encoding: the two readings of the body disagree, so it is refused too
        body = b'{"model": "text-a"}'
        connection = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=10)
        connection.putrequest("POST", "/v1/chat/completions")
        connection.putheader("Transfer-Encoding", "chunked")
        connection.putheader("Content-Length", str(len(body)))
        connection.putheader("Content-Type", "application/json")
        connection.endheaders(body)
        self.assertEqual(connection.getresponse().status, 400)
        connection.close()

    def test_a_cross_origin_switch_is_refused(self):
        rig = Rig(self, [entry("text-a")])
        status, _, _, body = rig.call("POST", "/v1/models/switch", {"key": rig.entries[0]["key"]},
                                      headers={"Origin": "http://evil.example"})
        self.assertEqual((status, body["error"]["code"]), (403, "forbidden"))


class OneModelAtATimeTest(unittest.TestCase):
    def run_two(self, rig, first, second, sleep=0.6):
        results = []
        def go(model):
            results.append(rig.chat(model, sleep=sleep)[0])
        threads = [threading.Thread(target=go, args=(m,)) for m in (first, second)]
        for thread in threads:
            thread.start()
            time.sleep(0.15)
        for thread in threads:
            thread.join()
        return results

    def test_two_models_never_compute_together(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")])
        rig.chat("text-a-colibri")
        rig.chat("text-b-colibri")                                      # both loaded
        open(rig.log, "w").close()
        self.assertEqual(self.run_two(rig, "text-a-colibri", "text-b-colibri"), [200, 200])
        self.assertFalse(overlap(rig.events(), "text-a-colibri", "text-b-colibri"))
        self.assertEqual(rig.states(), {"text-a-colibri": "ready", "text-b-colibri": "ready"})

    def test_requests_for_the_same_model_may_run_together(self):
        rig = Rig(self, [entry("text-a")])
        rig.chat("text-a-colibri")
        open(rig.log, "w").close()
        self.assertEqual(self.run_two(rig, "text-a-colibri", "text-a-colibri", sleep=1.5), [200, 200])
        self.assertTrue(overlap(rig.events(), "text-a-colibri", "text-a-colibri"))

    def test_parallel_models_lets_them_overlap(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")], parallel=True)
        rig.chat("text-a-colibri")
        rig.chat("text-b-colibri")
        open(rig.log, "w").close()
        self.run_two(rig, "text-a-colibri", "text-b-colibri", sleep=1.5)
        self.assertTrue(overlap(rig.events(), "text-a-colibri", "text-b-colibri"))

    def test_waiting_requests_are_served_in_order_and_the_queue_is_bounded(self):
        rig = Rig(self, [entry("text-a"), entry("text-b"), entry("text-c")], max_queue=1)
        for name in ("text-a", "text-b", "text-c"):
            rig.chat(name + "-colibri")
        statuses = {}
        def go(model):
            statuses[model] = rig.chat(model, sleep=1.0)
        threads = []
        for model in ("text-a-colibri", "text-b-colibri", "text-c-colibri"):
            thread = threading.Thread(target=go, args=(model,))
            thread.start()
            threads.append(thread)
            time.sleep(0.3)
        for thread in threads:
            thread.join()
        self.assertEqual(statuses["text-a-colibri"][0], 200)
        self.assertEqual(statuses["text-b-colibri"][0], 200)
        self.assertEqual(statuses["text-c-colibri"][0], 429)
        self.assertEqual(statuses["text-c-colibri"][3]["error"]["code"], "queue_full")
        self.assertEqual(statuses["text-c-colibri"][1].get("Retry-After"), "1")

    def test_a_request_that_waits_too_long_times_out(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")], queue_timeout=0.8)
        rig.chat("text-a-colibri")
        rig.chat("text-b-colibri")
        threading.Thread(target=lambda: rig.chat("text-a-colibri", sleep=2.0), daemon=True).start()
        time.sleep(0.3)
        status, _, _, body = rig.chat("text-b-colibri")
        self.assertEqual((status, body["error"]["code"]), (429, "queue_timeout"))


class MakingRoomTest(unittest.TestCase):
    def test_a_model_that_does_not_fit_stops_the_idle_one(self):
        rig = Rig(self, [entry("text-a", size=6 * GB), entry("text-b", size=6 * GB)], total=10 * GB)
        rig.chat("text-a-colibri")
        self.assertEqual(rig.states(), {"text-a-colibri": "ready", "text-b-colibri": "idle"})
        self.assertEqual(rig.chat("text-b-colibri")[0], 200)
        self.assertEqual(rig.states(), {"text-a-colibri": "idle", "text-b-colibri": "ready"})
        self.assertEqual(rig.chat("text-a-colibri")[0], 200)
        self.assertEqual(rig.states(), {"text-a-colibri": "ready", "text-b-colibri": "idle"})

    def test_models_that_fit_together_stay_loaded(self):
        rig = Rig(self, [entry("text-a", size=2 * GB), entry("text-b", size=2 * GB)], total=10 * GB)
        rig.chat("text-a-colibri")
        rig.chat("text-b-colibri")
        self.assertEqual(rig.states(), {"text-a-colibri": "ready", "text-b-colibri": "ready"})

    def test_the_headroom_counts(self):
        # 10 GB free, 8 GB model: fits in 80% of 10 exactly; a 8.5 GB model does not
        rig = Rig(self, [entry("text-a", size=1 * GB), entry("text-b", size=int(8.5 * GB))], total=10 * GB)
        rig.chat("text-a-colibri")
        rig.chat("text-b-colibri")
        self.assertEqual(rig.states(), {"text-a-colibri": "idle", "text-b-colibri": "ready"})

    def test_the_least_recently_used_goes_first(self):
        rig = Rig(self, [entry("text-a", size=int(3.5 * GB)), entry("text-b", size=int(3.5 * GB)),
                         entry("text-c", size=int(3.5 * GB))], total=10 * GB)
        rig.chat("text-a-colibri")
        rig.chat("text-b-colibri")
        rig.chat("text-a-colibri")                                      # a is now newer than b
        rig.chat("text-c-colibri")                                      # 3.5 GB does not fit in 80% of the 3 GB left: evict
        self.assertEqual(rig.states()["text-b-colibri"], "idle")
        self.assertEqual(rig.states()["text-a-colibri"], "ready")

    def test_a_limit_on_loaded_models_is_kept(self):
        rig = Rig(self, [entry("text-a"), entry("text-b"), entry("text-c")], max_loaded=2)
        for name in ("text-a", "text-b", "text-c"):
            rig.chat(name + "-colibri")
        loaded = [k for k, v in rig.states().items() if v == "ready"]
        self.assertEqual(len(loaded), 2)
        self.assertIn("text-c-colibri", loaded)

    def test_a_model_in_use_is_never_stopped_to_make_room(self):
        rig = Rig(self, [entry("text-a", size=6 * GB), entry("text-b", size=6 * GB)], total=10 * GB, parallel=True)
        rig.chat("text-a-colibri")
        answer = {}
        thread = threading.Thread(target=lambda: answer.update(r=rig.chat("text-a-colibri", sleep=1.5)))
        thread.start()
        time.sleep(0.4)
        began = time.monotonic()
        status = rig.chat("text-b-colibri")[0]
        waited = time.monotonic() - began
        thread.join()
        self.assertEqual(answer["r"][0], 200, "the running request was cut off")
        self.assertEqual(status, 200)
        self.assertGreater(waited, 0.8, "b did not wait for a to finish")
        self.assertEqual(rig.states(), {"text-a-colibri": "idle", "text-b-colibri": "ready"})

    def test_a_model_bigger_than_memory_is_loaded_anyway_when_nothing_else_is(self):
        rig = Rig(self, [entry("text-a", size=50 * GB)], total=10 * GB)
        self.assertEqual(rig.chat("text-a-colibri")[0], 200)

    def test_what_a_model_really_took_is_remembered(self):
        path = os.path.join(tempfile.mkdtemp(), "router.json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        rig = Rig(self, [entry("text-a", size=40 * GB)], measure=lambda pid, job: 3 * GB, state_path=path)
        rig.chat("text-a-colibri")
        saved = json.load(open(path))
        self.assertEqual(saved[rig.entries[0]["key"]]["resident_bytes"], 3 * GB)
        again = router.Pool(rig.catalog, state=router.State(path), worker_command=lambda e, p: [], grace=1)
        self.addCleanup(again.close)
        self.assertEqual(again.estimate(rig.entries[0]), 3 * GB)


class KeepAliveTest(unittest.TestCase):
    def test_an_idle_model_is_stopped_after_its_keep_alive(self):
        rig = Rig(self, [entry("text-a")], keep_alive=1.0)
        rig.chat("text-a-colibri")
        worker = rig.pool.get(rig.entries[0]["key"])
        self.assertEqual(rig.states()["text-a-colibri"], "ready")
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and rig.states()["text-a-colibri"] != "idle":
            time.sleep(0.3)
        self.assertEqual(rig.states()["text-a-colibri"], "idle")
        self.assertFalse(worker.alive(), "the process is still running")

    def test_a_model_in_use_is_not_stopped(self):
        rig = Rig(self, [entry("text-a")], keep_alive=0.5)
        status = rig.chat("text-a-colibri", sleep=2.5)[0]
        self.assertEqual(status, 200)

    def test_forever_keeps_it(self):
        rig = Rig(self, [entry("text-a")], keep_alive=None)
        rig.chat("text-a-colibri")
        time.sleep(2.5)
        self.assertEqual(rig.states()["text-a-colibri"], "ready")

    def test_a_request_starts_the_clock_again(self):
        rig = Rig(self, [entry("text-a")], keep_alive=2.5)
        rig.chat("text-a-colibri")
        time.sleep(1.5)
        rig.chat("text-a-colibri")
        time.sleep(1.5)
        self.assertEqual(rig.states()["text-a-colibri"], "ready")

    def test_the_clock_runs_from_the_end_of_a_request_not_its_start(self):
        rig = Rig(self, [entry("text-a")], keep_alive=2.0)
        rig.chat("text-a-colibri", sleep=1.6)                 # 1.6 s of work, then 1 s idle: 2.6 s since it began
        time.sleep(1.0)
        self.assertEqual(rig.states()["text-a-colibri"], "ready")

    def test_parse_duration(self):
        self.assertEqual(router.parse_duration("30s"), 30)
        self.assertEqual(router.parse_duration("10m"), 600)
        self.assertEqual(router.parse_duration("2h"), 7200)
        self.assertEqual(router.parse_duration("45"), 45)
        self.assertEqual(router.parse_duration("0"), 0)
        self.assertIsNone(router.parse_duration("-1"))
        self.assertIsNone(router.parse_duration("forever"))
        with self.assertRaises(ValueError):
            router.parse_duration("soon")


def multipart_raw(fields, boundary="bb"):
    out = b""
    for name, value in fields:
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    return out + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


class PerRequestKeepAliveTest(unittest.TestCase):
    def wait_idle(self, rig, seconds=6.0):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and rig.states()["text-a-colibri"] != "idle":
            time.sleep(0.2)
        return rig.states()["text-a-colibri"]

    def view(self, rig):
        return rig.call("GET", "/health")[3]["router"]["loaded"][0]

    def test_a_request_says_how_long_its_model_stays(self):
        rig = Rig(self, [entry("text-a")], keep_alive=600.0)
        rig.chat("text-a-colibri", keep_alive="1s")
        self.assertEqual(self.view(rig)["keep_alive"], 1.0)
        self.assertEqual(self.wait_idle(rig), "idle")

    def test_a_number_is_seconds_and_minus_one_is_forever(self):
        rig = Rig(self, [entry("text-a")], keep_alive=0.5)
        rig.chat("text-a-colibri", keep_alive=-1)
        time.sleep(2.5)
        self.assertEqual(rig.states()["text-a-colibri"], "ready")
        self.assertIsNone(self.view(rig)["keep_alive"])
        self.assertIsNone(self.view(rig)["expires_in"])
        rig.chat("text-a-colibri", keep_alive=1.5)
        self.assertEqual(self.view(rig)["keep_alive"], 1.5)

    def test_zero_unloads_as_soon_as_the_model_is_idle(self):
        rig = Rig(self, [entry("text-a")], keep_alive=600.0)
        began = time.monotonic()
        rig.chat("text-a-colibri", keep_alive=0)
        self.assertEqual(self.wait_idle(rig, 3.0), "idle")
        self.assertLess(time.monotonic() - began, 3.5)

    def test_it_sticks_until_a_request_says_otherwise(self):
        rig = Rig(self, [entry("text-a")], keep_alive=600.0)
        rig.chat("text-a-colibri", keep_alive="1s")
        rig.chat("text-a-colibri")                                   # says nothing: the 1 s stays
        self.assertEqual(self.view(rig)["keep_alive"], 1.0)
        self.assertEqual(self.wait_idle(rig), "idle")
        rig.chat("text-a-colibri")                                   # a new load starts from the server's default
        self.assertEqual(self.view(rig)["keep_alive"], 600.0)

    def test_a_bad_value_is_refused_before_anything_loads(self):
        rig = Rig(self, [entry("text-a")])
        for bad in ("soon", True, [1], {"a": 1}, "5x"):
            status, _, _, body = rig.chat("text-a-colibri", keep_alive=bad)
            self.assertEqual((status, body["error"]["param"], body["error"]["type"]),
                             (400, "keep_alive", "invalid_request_error"), bad)
        self.assertEqual(rig.states(), {"text-a-colibri": "idle"})
        self.assertEqual(rig.call("GET", "/health")[3]["router"]["loaded"], [])
        self.assertEqual(rig.chat("text-a-colibri", keep_alive=None)[0], 200)      # null is the same as absent

    def test_the_worker_never_sees_it(self):
        rig = Rig(self, [entry("text-a"), entry("music-m", "audio")])
        keys = rig.chat("text-a-colibri", keep_alive="5m")[3]["keys"]
        self.assertEqual(keys, ["messages", "model"])
        raw, kind = multipart_raw([("prompt", "hi"), ("keep_alive", "-1"), ("model", "music-m")])
        status, _, _, body = rig.call("POST", "/v1/audio/music", raw=raw, headers={"Content-Type": kind})
        self.assertEqual((status, body["keys"]), (200, ["model", "prompt"]))
        worker = rig.pool.get(rig.entries[1]["key"])
        self.assertIsNone(worker.keep_alive, "the multipart field still applies: -1 is forever")

    def test_a_preload_can_set_it_and_an_already_loaded_model_can_change_it(self):
        rig = Rig(self, [entry("text-a")], keep_alive=600.0)
        status, _, _, body = rig.call("POST", "/v1/models/switch", {"model": "text-a", "keep_alive": "2m"})
        self.assertEqual((status, body["status"], body["id"]), (200, "loaded", "text-a-colibri"))
        self.assertEqual(self.view(rig)["keep_alive"], 120.0)
        self.assertEqual(rig.call("POST", "/v1/models/switch", {"model": "text-a"})[0], 409)
        self.assertEqual(rig.call("POST", "/v1/models/switch", {"model": "text-a", "keep_alive": "1s"})[0], 200)
        self.assertEqual(self.view(rig)["keep_alive"], 1.0)
        self.assertEqual(rig.call("POST", "/v1/models/switch", {"model": "text-a", "keep_alive": "soon"})[0], 400)

    def test_keep_alive_value(self):
        self.assertIs(router.keep_alive_value(None), router.UNSET)
        self.assertEqual(router.keep_alive_value(30), 30.0)
        self.assertEqual(router.keep_alive_value(0), 0.0)
        self.assertIsNone(router.keep_alive_value(-1))
        self.assertIsNone(router.keep_alive_value("-1"))
        self.assertEqual(router.keep_alive_value("5m"), 300.0)
        for bad in (True, False, [], {}, "x", "1d"):
            with self.assertRaises(router.ApiError):
                router.keep_alive_value(bad)

    def test_a_multipart_part_is_removed_whole_and_the_rest_is_kept(self):
        raw, kind = multipart_raw([("a", "1"), ("keep_alive", "5m"), ("b", "2")])
        body = router.Body(kind, raw)
        self.assertEqual(body.keep_alive, 300.0)
        sent = body.for_worker("m")
        self.assertEqual(sent, b'--bb\r\nContent-Disposition: form-data; name="a"\r\n\r\n1\r\n'
                               b'--bb\r\nContent-Disposition: form-data; name="b"\r\n\r\n2\r\n--bb--\r\n')
        only, kind = multipart_raw([("keep_alive", "5m")])
        self.assertEqual(router.Body(kind, only).for_worker("m"), b"--bb--\r\n")
        same, kind = multipart_raw([("a", "1")])
        self.assertIs(router.Body(kind, same).for_worker("m"), same)


class UnloadTest(unittest.TestCase):
    def test_an_idle_model_is_stopped(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")])
        rig.chat("text-b-colibri")                      # b first: the one to stop is not the first the router started
        rig.chat("text-a-colibri")
        worker = rig.pool.get(rig.entries[0]["key"])
        status, _, _, body = rig.call("POST", "/v1/models/unload", {"model": "text-a"})
        self.assertEqual((status, body["status"], body["id"]), (200, "unloaded", "text-a-colibri"))
        self.assertEqual(rig.states(), {"text-a-colibri": "idle", "text-b-colibri": "ready"})
        self.assertFalse(worker.alive())

    def test_by_key_and_for_a_model_that_is_not_loaded(self):
        rig = Rig(self, [entry("text-a")])
        status, _, _, body = rig.call("POST", "/v1/models/unload", {"key": rig.entries[0]["key"]})
        self.assertEqual((status, body["status"]), (200, "not loaded"))
        rig.chat("text-a-colibri")
        self.assertEqual(rig.call("POST", "/v1/models/unload", {"key": rig.entries[0]["key"]})[3]["status"], "unloaded")

    def test_a_model_that_is_answering_goes_when_it_has_finished(self):
        rig = Rig(self, [entry("text-a")])
        rig.chat("text-a-colibri")
        answer = {}
        thread = threading.Thread(target=lambda: answer.update(r=rig.chat("text-a-colibri", sleep=1.5)))
        thread.start()
        time.sleep(0.5)
        status, _, _, body = rig.call("POST", "/v1/models/unload", {"model": "text-a"})
        self.assertEqual((status, body["status"]), (202, "unloading when idle"))
        self.assertEqual(rig.states()["text-a-colibri"], "ready", "it must not be cut off")
        thread.join()
        self.assertEqual(answer["r"][0], 200, "the running request was cut off")
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and rig.states()["text-a-colibri"] != "idle":
            time.sleep(0.2)
        self.assertEqual(rig.states()["text-a-colibri"], "idle")

    def test_it_says_what_it_cannot_do(self):
        rig = Rig(self, [entry("text-a")], api_key="sekrit")
        auth = {"Authorization": "Bearer sekrit"}
        self.assertEqual(rig.call("POST", "/v1/models/unload", {"model": "text-a"})[0], 401)
        self.assertEqual(rig.call("POST", "/v1/models/unload", {"model": "nope"}, headers=auth)[0], 404)
        self.assertEqual(rig.call("POST", "/v1/models/unload", {}, headers=auth)[0], 400)
        self.assertEqual(rig.call("POST", "/v1/models/unload", raw=b"{}",
                                  headers=dict(auth, **{"Content-Type": "text/plain"}))[0], 415)
        status, _, _, body = rig.call("POST", "/v1/models/unload", {"model": "text-a"},
                                      headers=dict(auth, Origin="http://evil.example"))
        self.assertEqual((status, body["error"]["code"]), (403, "forbidden"))


class FailureTest(unittest.TestCase):
    def test_a_worker_that_does_not_start_is_reported_with_its_reason(self):
        rig = Rig(self, [entry("text-a")], env={"text-a-colibri": {"FAKE_FAIL": "out of cheese"}})
        status, _, _, body = rig.chat("text-a-colibri")
        self.assertEqual((status, body["error"]["code"]), (503, "model_load_failed"))
        self.assertIn("out of cheese", body["error"]["message"])
        self.assertEqual(rig.states(), {"text-a-colibri": "idle"})
        installed = rig.call("GET", "/v1/models/installed")[3]["data"][0]
        self.assertIn("out of cheese", installed["last_error"])

    def test_a_failed_load_is_tried_again_by_the_next_request(self):
        rig = Rig(self, [entry("text-a")], env={"text-a-colibri": {"FAKE_FAIL": "once"}})
        self.assertEqual(rig.chat("text-a-colibri")[0], 503)
        rig.pool.worker_env = lambda item: {"FAKE_LOG": rig.log}
        self.assertEqual(rig.chat("text-a-colibri")[0], 200)

    def test_a_worker_that_dies_is_started_again(self):
        rig = Rig(self, [entry("text-a")])
        first = rig.chat("text-a-colibri")[3]["pid"]
        worker = rig.pool.get(rig.entries[0]["key"])
        worker.proc.kill()
        worker.proc.wait(10)
        status, _, _, body = rig.chat("text-a-colibri")             # at once: before the reaper has looked
        self.assertEqual(status, 200)
        self.assertNotEqual(body["pid"], first)

    def test_requests_that_wait_for_a_loading_model_share_the_load(self):
        rig = Rig(self, [entry("text-a")], env={"text-a-colibri": {"FAKE_LOAD_DELAY": "1.5"}})
        pids, statuses = [], []
        def go():
            status, _, _, body = rig.chat("text-a-colibri")
            statuses.append(status)
            pids.append(body["pid"])
        threads = [threading.Thread(target=go) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(statuses, [200, 200, 200])
        self.assertEqual(len(set(pids)), 1)

    def test_the_router_stops_its_workers_when_it_closes(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")])
        rig.chat("text-a-colibri")
        rig.chat("text-b-colibri")
        workers = rig.pool.snapshot()
        rig.stop()
        self.assertTrue(all(not w.alive() for w in workers))


class SwitchTest(unittest.TestCase):
    def test_switch_loads_a_model_and_answers_when_it_is_ready(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")])
        key = rig.entries[1]["key"]
        status, _, _, body = rig.call("POST", "/v1/models/switch", {"key": key})
        self.assertEqual((status, body["id"], body["status"]), (200, "text-b-colibri", "loaded"))
        self.assertEqual(rig.states(), {"text-a-colibri": "idle", "text-b-colibri": "ready"})
        self.assertEqual(rig.call("POST", "/v1/models/switch", {"key": key})[3]["error"]["code"], "already_current")
        self.assertEqual(rig.call("POST", "/v1/models/switch", {"key": "nope"})[0], 404)
        self.assertEqual(rig.call("POST", "/v1/models/switch", {})[0], 400)
        self.assertEqual(rig.call("POST", "/v1/models/switch", raw=b"{}", headers={"Content-Type": "text/plain"})[0], 415)

    def test_switch_that_fails_says_why(self):
        rig = Rig(self, [entry("text-a")], env={"text-a-colibri": {"FAKE_FAIL": "bad weights"}})
        status, _, _, body = rig.call("POST", "/v1/models/switch", {"key": rig.entries[0]["key"]})
        self.assertEqual(status, 503)
        self.assertIn("bad weights", body["error"]["message"])

    def test_the_installed_list_is_what_the_dashboard_reads(self):
        rig = Rig(self, [entry("text-a", size=5 * GB), entry("music-m", "audio", reason="the engine is not built")])
        rig.chat("text-a-colibri")
        body = rig.call("GET", "/v1/models/installed")[3]
        self.assertTrue(body["switch"])
        self.assertTrue(body["router"])
        by_id = {item["id"]: item for item in body["data"]}
        self.assertTrue(by_id["text-a-colibri"]["current"])
        self.assertEqual(by_id["text-a-colibri"]["state"], "ready")
        self.assertIsNotNone(by_id["text-a-colibri"]["expires_in"])
        self.assertNotIn("path", by_id["text-a-colibri"])
        self.assertFalse(by_id["music-m-colibri"]["switchable"])
        self.assertEqual(by_id["music-m-colibri"]["reason"], "the engine is not built")
        self.assertEqual(body["data"][0]["id"], "text-a-colibri")       # the loaded model first

    def test_health_shows_the_loaded_models_and_the_queue(self):
        rig = Rig(self, [entry("text-a")])
        rig.chat("text-a-colibri")
        body = rig.call("GET", "/health")[3]
        self.assertEqual(body["arch"], "fake")                           # the loaded model's own health
        self.assertEqual([w["id"] for w in body["router"]["loaded"]], ["text-a-colibri"])
        self.assertEqual(body["router"]["gate"], {"active": 0, "queued": 0})
        self.assertFalse(body["router"]["parallel_models"])


class WorkerKvSlotsTest(unittest.TestCase):
    """The router's --kv-slots is what a chat model gets, never more than its family allows: a picture or a music
    model has one slot, and was refused (or marked unavailable) when every worker was told the same number."""

    def slots(self, family, wanted):
        return router.worker_kv_slots({"family": family}, wanted)

    def test_a_chat_model_gets_what_was_asked(self):
        self.assertEqual(self.slots("qwen36", 3), 3)

    def test_a_chat_model_is_held_to_its_family_limit(self):
        from family_registry import family_by_id
        limit = family_by_id("qwen36").limits.max_kv_slots
        self.assertEqual(self.slots("qwen36", limit + 5), limit)

    def test_picture_and_music_models_keep_one_slot(self):
        self.assertEqual(self.slots("qwen_image", 3), 1)
        self.assertEqual(self.slots("ace_step", 3), 1)

    def test_an_unknown_family_or_a_nonsense_count_gives_one(self):
        self.assertEqual(self.slots("no-such-family", 3), 1)
        self.assertEqual(router.worker_kv_slots({}, 3), 1)
        self.assertEqual(self.slots("qwen36", 0), 1)


if __name__ == "__main__":
    unittest.main()
