"""Tags a client puts on a request (client_tags.py), and what the router does with them."""
import json
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import client_tags  # noqa: E402
import request_log  # noqa: E402
from test_router import Rig, entry  # noqa: E402
from test_router_jobs import AUTH, KEY, wait_for  # noqa: E402
from test_library import Folders  # noqa: E402

TAGS = client_tags.HEADER


class ParseTest(unittest.TestCase):
    def test_a_small_object_is_taken_as_it_is(self):
        self.assertEqual(client_tags.parse('{"app": "dashboard", "chat": "c-7f3a", "n": 3, "pinned": true}'),
                         {"app": "dashboard", "chat": "c-7f3a", "n": 3, "pinned": True})

    def test_no_header_or_an_empty_one_is_no_tags(self):
        for value in (None, "", "   ", "{}"):
            self.assertIsNone(client_tags.parse(value))

    def test_what_does_not_fit_is_refused_and_says_what_to_change(self):
        cases = {
            "not json": "JSON object",
            "[1, 2]": "JSON object",
            '{"1bad": "x"}': "must start with a letter",
            '{"a b": "x"}': "must start with a letter",
            '{"k": {"nested": 1}}': "string, a number or true or false",
            '{"k": null}': "string, a number or true or false",
            '{"k": "' + "x" * 121 + '"}': "up to 120 characters",
            '{"k": "line\\nbreak"}': "control characters",
            json.dumps({f"k{i}": i for i in range(9)}): "at most 8",
            '{"k": "' + "x" * 1100 + '"}': "over 1024 bytes",
        }
        for text, words in cases.items():
            with self.assertRaises(client_tags.TagError, msg=text[:40]) as caught:
                client_tags.parse(text)
            self.assertIn(words, str(caught.exception), text[:40])

    def test_a_filter_takes_the_first_colon_only(self):
        self.assertEqual(client_tags.wanted(["chat:c-7f3a", "url:http://x/y"]), {"chat": "c-7f3a", "url": "http://x/y"})
        for bad in ("chat", ":x", "1a:x"):
            with self.assertRaises(client_tags.TagError):
                client_tags.wanted([bad])

    def test_matching_compares_as_text_and_needs_every_one(self):
        tags = {"chat": "c1", "n": 3, "pinned": True}
        self.assertTrue(client_tags.matches(tags, {"n": "3", "pinned": "true"}))
        self.assertFalse(client_tags.matches(tags, {"chat": "c1", "n": "4"}))
        self.assertFalse(client_tags.matches(None, {"chat": "c1"}))
        self.assertTrue(client_tags.matches(None, {}))


class RouterTagsTest(unittest.TestCase):
    def setUp(self):
        self.dirs = Folders().make(self)
        self.rig = Rig(self, [entry("text-a"), entry("image-b", "image")], api_key=KEY)
        self.log_path = os.path.join(self.rig.tmp.name, "requests.jsonl")
        self.rig.server.request_log = request_log.RequestLog(self.log_path)

    def chat(self, tags=None, model="text-a-colibri"):
        headers = dict(AUTH)
        if tags is not None:
            headers[TAGS] = tags if isinstance(tags, str) else json.dumps(tags)
        return self.rig.call("POST", "/v1/chat/completions", {"model": model, "messages": []}, headers)

    def logged(self, query=""):
        status, _h, _raw, body = self.rig.call("GET", "/v1/requests" + query, headers=AUTH)
        self.assertEqual(status, 200, body)
        return body["data"]

    def test_the_tags_are_written_on_the_requests_line(self):
        self.assertEqual(self.chat({"app": "dashboard", "chat": "c1"})[0], 200)
        self.chat()
        lines = [r for r in self.logged() if r.get("path") == "/v1/chat/completions"]
        self.assertEqual(sorted((r.get("tags") or {}).get("chat", "-") for r in lines), ["-", "c1"])

    def test_requests_can_be_listed_by_tag(self):
        self.chat({"chat": "c1"})
        self.chat({"chat": "c2"})
        self.chat({"chat": "c1", "app": "extension"})
        both = self.logged("?tag=chat:c1")
        self.assertEqual(len(both), 2)
        self.assertEqual(len(self.logged("?tag=chat:c1&tag=app:extension")), 1)
        self.assertEqual(self.logged("?tag=chat:nope"), [])

    def test_a_header_that_does_not_fit_is_a_400_before_anything_loads(self):
        status, _h, _raw, body = self.chat("not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_tags")
        self.assertEqual(body["error"]["param"], TAGS)
        installed = self.rig.call("GET", "/v1/models/installed", headers=AUTH)[3]["data"]
        self.assertEqual({m["id"]: m["state"] for m in installed}["text-a-colibri"], "idle")   # the model was not started for it

    def test_a_bad_filter_is_a_400(self):
        self.assertEqual(self.rig.call("GET", "/v1/requests?tag=chat", headers=AUTH)[0], 400)
        self.assertEqual(self.rig.call("GET", "/v1/jobs?tag=chat", headers=AUTH)[0], 400)

    def test_a_background_job_keeps_the_tags_and_can_be_listed_by_them(self):
        def submit(tags):
            headers = {**AUTH, "Prefer": "respond-async"}
            if tags:
                headers[TAGS] = json.dumps(tags)
            return self.rig.call("POST", "/v1/images/generations",
                                 {"model": "image-b-colibri", "prompt": "a red fox", "job_demo": True, "steps": 1}, headers)
        tagged = submit({"chat": "c1"})
        submit(None)
        self.assertEqual(tagged[0], 202)
        self.assertEqual(tagged[3]["tags"], {"chat": "c1"})
        found = self.rig.call("GET", "/v1/jobs?tag=chat:c1", headers=AUTH)[3]
        self.assertEqual([job["id"] for job in found["data"]], [tagged[3]["id"]])
        self.assertEqual(self.rig.call("GET", "/v1/jobs", headers=AUTH)[3]["total"], 2)
        self.assertEqual(wait_for(lambda: self.rig.call("GET", f"/v1/jobs/{tagged[3]['id']}", headers=AUTH)[3]["status"] == "succeeded"
                                  or None), True)
        again = self.rig.call("GET", f"/v1/jobs/{tagged[3]['id']}", headers=AUTH)[3]
        self.assertEqual(again["tags"], {"chat": "c1"})                                    # still there when it has finished

    def test_a_page_on_another_origin_may_send_the_header(self):
        status, headers, _raw, _body = self.rig.call("OPTIONS", "/v1/chat/completions", headers={
            "Origin": "http://127.0.0.1:1", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": TAGS})
        self.assertIn(TAGS.lower(), headers.get("Access-Control-Allow-Headers", "").lower())

    def test_the_features_block_says_so(self):
        features = self.rig.call("GET", "/health", headers=AUTH)[3]["features"]
        self.assertTrue(features["tags"])


class GatewayCorsTest(unittest.TestCase):
    def test_a_gateway_lets_a_page_on_another_origin_send_the_header_too(self):
        import threading
        from urllib.request import Request, urlopen
        from openai_server import APIServer

        class Engine:
            pass
        server = APIServer(("127.0.0.1", 0), Engine(), "m", cors_origins=("http://example.test",))
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        request = Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", method="OPTIONS",
                          headers={"Origin": "http://example.test", "Access-Control-Request-Method": "POST",
                                   "Access-Control-Request-Headers": TAGS})
        with urlopen(request, timeout=10) as response:
            self.assertIn(TAGS.lower(), response.headers.get("Access-Control-Allow-Headers", "").lower())


if __name__ == "__main__":
    unittest.main()
