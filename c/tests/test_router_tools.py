"""The router's built-in tools: GET /v1/tools, POST /v1/tools/<name>, the approvals, and chat completions where the router runs the loop.

The model is a scripted server (`Model`) that the router's loop reaches instead of its own chat endpoint, so each round of the conversation is
exactly what the test says; the files and the policy are real."""
import http.client
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import router  # noqa: E402
import tools_loop  # noqa: E402
from test_router import Rig, entry  # noqa: E402


class Model:
    """A chat endpoint that answers from a script. A step is {"text": ..., "calls": [(name, arguments)], "status": 200, "error": {...}}; the last
    step repeats. `requests` keeps what the router sent: (headers, body)."""

    def __init__(self, steps):
        self.steps, self.requests, self.lock = steps, [], threading.Lock()
        model = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with model.lock:
                    index = len(model.requests)
                    model.requests.append((dict(self.headers), body))
                step = model.steps[min(index, len(model.steps) - 1)]
                model.answer(self, body, step)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def answer(self, h, body, step):
        if "error" in step:
            raw = json.dumps({"error": step["error"]}).encode()
            h.send_response(step.get("status", 500))
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(raw)))
            h.end_headers()
            h.wfile.write(raw)
            return
        text, calls = step.get("text", ""), step.get("calls", [])
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if not body.get("stream"):
            message = {"role": "assistant", "content": text}
            if calls:
                message["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
            raw = json.dumps({"id": "x", "object": "chat.completion", "model": "scripted", "usage": usage,
                              "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}]}).encode()
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(raw)))
            h.end_headers()
            h.wfile.write(raw)
            return
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()

        def send(obj):
            h.wfile.write(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")
            h.wfile.flush()

        def chunk(delta, finish=None):
            return {"id": "x", "object": "chat.completion.chunk", "model": "scripted", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        send(chunk({"role": "assistant", "content": ""}))
        half = len(text) // 2
        for piece in (text[:half], text[half:]):
            if piece:
                send(chunk({"content": piece}))
        for i, (name, arguments) in enumerate(calls):
            raw = json.dumps(arguments)
            send(chunk({"tool_calls": [{"index": i, "id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": ""}}]}))
            send(chunk({"tool_calls": [{"index": i, "function": {"arguments": raw[:len(raw) // 2]}}]}))
            send(chunk({"tool_calls": [{"index": i, "function": {"arguments": raw[len(raw) // 2:]}}]}))
        send(chunk({}, "tool_calls" if calls else "stop"))
        if (body.get("stream_options") or {}).get("include_usage"):
            send({"id": "x", "object": "chat.completion.chunk", "model": "scripted", "choices": [], "usage": usage})
        send(b"[DONE]")


def sse_events(raw):
    out = []
    for block in raw.decode("utf-8").split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                out.append(line[6:])
    return out


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = os.path.realpath(self.temp.name)
        self.proj = os.path.join(self.root, "proj")
        self.scratch = os.path.join(self.root, "scratch")
        for d in (self.proj, self.scratch):
            os.makedirs(d)
        Path(self.proj, "notes.txt").write_text("the notes say hello\n", encoding="utf-8")
        self.notes = os.path.join(self.proj, "notes.txt")
        self.policy_file = os.path.join(self.root, "policy.json")
        patcher = patch.dict(os.environ, {"COLI_TOOLS_POLICY": self.policy_file})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.rig = Rig(self, [entry("text-a")])

    def policy(self, **settings):
        data = {"folders": [{"path": self.proj}, {"path": self.scratch, "write": True}]}
        data.update(settings)
        Path(self.policy_file).write_text(json.dumps(data))

    def model(self, steps):
        model = Model(steps)
        self.addCleanup(model.close)
        patcher = patch.object(tools_loop, "loopback", lambda port: urllib.parse.urlsplit(f"http://127.0.0.1:{model.port}"))
        patcher.start()
        self.addCleanup(patcher.stop)
        return model

    def chat(self, body, headers=None):
        return self.rig.call("POST", "/v1/chat/completions", dict({"model": "text-a-colibri", "messages": [{"role": "user", "content": "hi"}]}, **body), headers=headers)

    def stream(self, body, on_event=None, headers=None):
        """A streamed chat request read as it arrives: the parsed events. `on_event(event)` runs as each arrives (it may answer an approval)."""
        connection = http.client.HTTPConnection("127.0.0.1", self.rig.port, timeout=30)
        payload = json.dumps(dict({"model": "text-a-colibri", "messages": [{"role": "user", "content": "hi"}], "stream": True}, **body))
        connection.request("POST", "/v1/chat/completions", payload, dict({"Content-Type": "application/json"}, **(headers or {})))
        response = connection.getresponse()
        events, buffer = [], b""
        while True:
            piece = response.read1(4096)
            if not piece:
                break
            buffer += piece
            while b"\n\n" in buffer:
                block, buffer = buffer.split(b"\n\n", 1)
                for line in block.decode("utf-8").split("\n"):
                    if line.startswith("data: "):
                        value = line[6:]
                        event = value if value == "[DONE]" else json.loads(value)
                        events.append(event)
                        if on_event:
                            on_event(event)
        connection.close()
        return response.status, events


class EndpointTest(ToolsTest):
    def test_with_no_policy_everything_is_off(self):
        status, _, _, body = self.rig.call("GET", "/v1/tools")
        self.assertEqual(status, 200)
        self.assertEqual((body["problem"], body["policy"]["folders"], body["policy"]["web"]), (None, [], False))
        self.assertTrue(body["data"])
        self.assertFalse(any(t["colibri"]["enabled"] for t in body["data"]))
        self.assertEqual(self.rig.call("POST", "/v1/tools/files_read", {"arguments": {"path": self.notes}})[0], 404)

    def test_the_policy_decides_what_is_on(self):
        self.policy(web=True)
        body = self.rig.call("GET", "/v1/tools")[3]
        on = {t["function"]["name"] for t in body["data"] if t["colibri"]["enabled"]}
        self.assertEqual(on, {"web_fetch", "files_list", "files_read", "files_search", "git_status", "git_log", "git_diff", "git_show", "git_branches",
                              "files_write", "files_edit", "git_add", "git_commit"})
        self.assertEqual(body["data"][0]["type"], "function")
        self.assertIn("parameters", body["data"][0]["function"])
        self.assertEqual([f["write"] for f in body["policy"]["folders"]], [False, True])
        self.assertEqual(body["path"], self.policy_file)

    def test_a_bad_policy_file_is_reported_and_turns_everything_off(self):
        Path(self.policy_file).write_text('{"surprise": 1}')
        body = self.rig.call("GET", "/v1/tools")[3]
        self.assertIn("surprise", body["problem"])
        self.assertFalse(any(t["colibri"]["enabled"] for t in body["data"]))
        status, _, _, got = self.rig.call("POST", "/v1/tools/files_read", {"arguments": {"path": "x"}})
        self.assertEqual((status, got["error"]["code"]), (500, "tools_policy_error"))

    def test_a_read_only_tool_runs_without_confirm(self):
        self.policy()
        status, _, _, body = self.rig.call("POST", "/v1/tools/files_read", {"arguments": {"path": self.notes}})
        self.assertEqual((status, body["object"], body["name"], body["is_error"]), (200, "tool_result", "files_read", False))
        self.assertIn("the notes say hello", body["text"])

    def test_a_tool_that_fails_is_a_200_with_is_error_and_the_reason(self):
        self.policy()
        status, _, _, body = self.rig.call("POST", "/v1/tools/files_read", {"arguments": {"path": os.path.join(self.root, "elsewhere", "x.txt")}})
        self.assertEqual((status, body["is_error"]), (200, True))
        self.assertIn("outside", body["text"])

    def test_a_tool_that_changes_things_needs_confirm(self):
        self.policy()
        target = os.path.join(self.scratch, "made.txt")
        for extra in ({}, {"confirm": False}, {"confirm": "yes"}, {"confirm": 1}):
            with self.subTest(extra=extra):
                status, _, _, body = self.rig.call("POST", "/v1/tools/files_write", dict({"arguments": {"path": target, "content": "x"}}, **extra))
                self.assertEqual((status, body["error"]["code"]), (409, "confirmation_required"))
                self.assertIn("write 1 characters", body["error"]["message"])
                self.assertFalse(os.path.exists(target))
        status, _, _, body = self.rig.call("POST", "/v1/tools/files_write", {"arguments": {"path": target, "content": "hello"}, "confirm": True})
        self.assertEqual((status, body["is_error"]), (200, False))
        self.assertEqual(Path(target).read_text(), "hello")

    def test_a_write_outside_the_writable_folder_is_an_error_even_when_confirmed(self):
        self.policy()
        target = os.path.join(self.proj, "no.txt")
        body = self.rig.call("POST", "/v1/tools/files_write", {"arguments": {"path": target, "content": "x"}, "confirm": True})[3]
        self.assertTrue(body["is_error"])
        self.assertFalse(os.path.exists(target))

    def test_what_is_off_or_unknown_is_a_404_that_says_where_the_switch_is(self):
        self.policy()
        for name in ("web_fetch", "nonesuch", "policy", "pending2"):
            with self.subTest(name=name):
                status, _, _, body = self.rig.call("POST", f"/v1/tools/{name}", {"arguments": {}})
                self.assertEqual((status, body["error"]["code"]), (404, "tool_not_found"))
        self.assertIn("coli tools", body["error"]["message"])
        self.policy(disabled=["files_read"])
        self.assertEqual(self.rig.call("POST", "/v1/tools/files_read", {"arguments": {"path": self.notes}})[0], 404)

    def test_no_route_changes_the_policy(self):
        self.policy()
        before = Path(self.policy_file).read_text()
        for method, path, body in (("POST", "/v1/tools", {"folders": ["/"]}), ("POST", "/v1/tools/policy", {"folders": ["/"]}), ("POST", "/v1/tools/allow", {"path": "/"}),
                                   ("GET", "/v1/tools/allow", None)):
            self.rig.call(method, path, body)
        self.assertEqual(Path(self.policy_file).read_text(), before)

    def test_bad_arguments(self):
        self.policy()
        status, _, _, body = self.rig.call("POST", "/v1/tools/files_read", {"arguments": [1]})
        self.assertEqual((status, body["error"]["param"]), (400, "arguments"))
        self.assertEqual(self.rig.call("POST", "/v1/tools/files_read", raw=b"not json", headers={"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.rig.call("POST", "/v1/tools/files_read", raw=b"x=1", headers={"Content-Type": "text/plain"})[0], 415)
        body = self.rig.call("POST", "/v1/tools/files_read", {})[3]
        self.assertTrue(body["is_error"], "no arguments is the tool's own complaint (path is needed)")

    def test_it_needs_the_key_and_the_same_origin(self):
        self.policy()
        self.rig.server.api_key = "secret"
        for method, path, body in (("GET", "/v1/tools", None), ("GET", "/v1/tools/pending", None), ("POST", "/v1/tools/files_read", {"arguments": {"path": self.notes}}),
                                   ("POST", "/v1/tools/pending", {"id": "x", "allow": True})):
            self.assertEqual(self.rig.call(method, path, body)[0], 401, path)
        keyed = {"Authorization": "Bearer secret"}
        self.assertEqual(self.rig.call("POST", "/v1/tools/files_read", {"arguments": {"path": self.notes}}, headers=dict(keyed, Origin="http://evil.example"))[0], 403)
        self.assertEqual(self.rig.call("POST", "/v1/tools/files_read", {"arguments": {"path": self.notes}}, headers=dict(keyed, **{"Sec-Fetch-Site": "cross-site"}))[0], 403)
        self.assertEqual(self.rig.call("POST", "/v1/tools/files_read", {"arguments": {"path": self.notes}}, headers=keyed)[0], 200)

    def test_the_pending_list_and_answers(self):
        status, _, _, body = self.rig.call("GET", "/v1/tools/pending")
        self.assertEqual((status, body["data"]), (200, []))
        for payload in ({}, {"id": "x"}, {"id": 5, "allow": True}, {"id": "x", "allow": "yes"}):
            self.assertEqual(self.rig.call("POST", "/v1/tools/pending", payload)[0], 400, payload)
        status, _, _, body = self.rig.call("POST", "/v1/tools/pending", {"id": "nope", "allow": True})
        self.assertEqual((status, body["error"]["code"]), (404, "approval_not_found"))


class LoopTest(ToolsTest):
    def test_the_model_asks_for_a_tool_and_answers_from_its_result(self):
        self.policy()
        model = self.model([{"text": "", "calls": [("files_read", {"path": self.notes})]}, {"text": "The notes say hello."}])
        status, _, _, body = self.chat({"colibri_tools": True})
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"], {"role": "assistant", "content": "The notes say hello."})
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"], {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}, "every round is counted")
        calls = body["colibri"]["tool_calls"]
        self.assertEqual([(c["name"], c["ok"], c["how"]) for c in calls], [("files_read", True, "read-only")])
        self.assertIn("the notes say hello", calls[0]["result"])
        first, second = model.requests[0][1], model.requests[1][1]
        self.assertNotIn("colibri_tools", first)
        self.assertEqual(model.requests[0][1]["stream"], False)
        self.assertIn("files_read", [t["function"]["name"] for t in first["tools"]])
        roles = [m["role"] for m in second["messages"]]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        self.assertIn("the notes say hello", second["messages"][2]["content"])
        self.assertEqual(second["messages"][1]["tool_calls"][0]["function"]["name"], "files_read")

    def test_the_callers_key_and_tags_go_on_to_the_model_request(self):
        self.policy()
        self.rig.server.api_key = "secret"
        model = self.model([{"text": "ok"}])
        self.chat({"colibri_tools": True}, headers={"Authorization": "Bearer secret", "X-Colibri-Tags": "app:test"})
        sent = {k.lower(): v for k, v in model.requests[0][0].items()}
        self.assertEqual((sent["authorization"], sent["x-colibri-tags"]), ("Bearer secret", "app:test"))

    def test_the_other_request_fields_reach_the_model(self):
        self.policy()
        model = self.model([{"text": "ok"}])
        self.chat({"colibri_tools": True, "temperature": 0.2, "max_tokens": 77})
        body = model.requests[0][1]
        self.assertEqual((body["temperature"], body["max_tokens"], body["model"]), (0.2, 77, "text-a-colibri"))

    def test_a_group_or_a_list_of_names_chooses_what_the_model_is_shown(self):
        self.policy(web=True)
        model = self.model([{"text": "ok"}])
        self.chat({"colibri_tools": ["web"]})
        self.chat({"colibri_tools": ["git_log", "files_read"]})
        self.assertEqual([t["function"]["name"] for t in model.requests[0][1]["tools"]], ["web_fetch"])
        self.assertEqual([t["function"]["name"] for t in model.requests[1][1]["tools"]], ["git_log", "files_read"])

    def test_what_cannot_be_served_is_a_refusal_before_any_model_is_asked(self):
        self.model([{"text": "ok"}])
        cases = ((dict(colibri_tools=True), 409, "tools_not_enabled"),)
        for body, status, code in cases:
            got = self.chat(body)
            self.assertEqual((got[0], got[3]["error"]["code"]), (status, code))
        self.policy()
        for body, status, word in ((dict(colibri_tools=True, tools=[{"type": "function", "function": {"name": "x"}}]), 400, "not both"),
                                   (dict(colibri_tools=True, tool_choice="auto"), 400, "not both"),
                                   (dict(colibri_tools=["nonesuch"]), 400, "nonesuch"), (dict(colibri_tools=[5]), 400, "colibri_tools"),
                                   (dict(colibri_tools=True, n=2), 400, "n"), (dict(colibri_tools=True, messages=[]), 400, "messages"),
                                   (dict(colibri_tools=True, messages="hi"), 400, "messages")):
            with self.subTest(body=body):
                got = self.chat(body)
                self.assertEqual(got[0], status)
                self.assertIn(word, got[3]["error"]["message"])

    def test_false_means_an_ordinary_chat_request(self):
        self.policy()
        model = self.model([{"text": "ok"}])
        patcher = patch.object(tools_loop, "serve", side_effect=AssertionError("the loop must not run"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.chat({"colibri_tools": False})            # goes to the router's own worker (a fake one), not the scripted model
        self.assertEqual(model.requests, [])

    def test_a_tool_that_fails_is_a_result_the_model_can_correct(self):
        self.policy()
        model = self.model([{"text": "", "calls": [("files_read", {"path": os.path.join(self.root, "elsewhere", "x.txt")})]}, {"text": "I could not read it."}])
        body = self.chat({"colibri_tools": True})[3]
        self.assertEqual(body["choices"][0]["message"]["content"], "I could not read it.")
        self.assertFalse(body["colibri"]["tool_calls"][0]["ok"])
        tool_message = model.requests[1][1]["messages"][-1]
        self.assertTrue(tool_message["content"].startswith("Error: "))
        self.assertIn("outside", tool_message["content"])

    def test_a_tool_the_model_invents_is_answered_with_the_real_names(self):
        self.policy()
        model = self.model([{"text": "", "calls": [("files_reed", {})]}, {"text": "done"}])
        self.chat({"colibri_tools": ["files"]})
        content = model.requests[1][1]["messages"][-1]["content"]
        self.assertIn("files_read", content)

    def test_a_model_that_keeps_asking_is_stopped_and_asked_to_answer(self):
        self.policy()
        model = self.model([{"text": "", "calls": [("files_list", {})]}] * 3 + [{"text": "enough"}])
        with patch.object(tools_loop, "MAX_ROUNDS", 2):
            body = self.chat({"colibri_tools": True})[3]
        self.assertEqual(body["colibri"]["rounds_stopped"], True)
        self.assertEqual(len(model.requests), 3)
        self.assertNotIn("tools", model.requests[2][1], "the last round offers no tools")

    def test_an_error_from_the_model_is_passed_on_with_its_status(self):
        self.policy()
        self.model([{"error": {"message": "the model is busy", "type": "server_error", "code": "busy"}, "status": 503}])
        status, _, _, body = self.chat({"colibri_tools": True})
        self.assertEqual((status, body["error"]["message"]), (503, "the model is busy"))

    def test_an_unreachable_model_is_a_502(self):
        self.policy()
        free = socket.socket()
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
        free.close()
        with patch.object(tools_loop, "loopback", lambda p: urllib.parse.urlsplit(f"http://127.0.0.1:{port}")):
            status, _, _, body = self.chat({"colibri_tools": True})
        self.assertEqual(status, 502)

    def test_a_streamed_answer_carries_the_text_the_events_and_no_tool_calls_of_the_models_own(self):
        self.policy()
        self.model([{"text": "Let me look. ", "calls": [("files_read", {"path": self.notes})]}, {"text": "It says hello."}])
        status, events = self.stream({"colibri_tools": True, "stream_options": {"include_usage": True}})
        self.assertEqual(status, 200)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [e for e in events if e != "[DONE]"]
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        self.assertEqual(text, "Let me look. It says hello.")
        self.assertFalse(any("tool_calls" in c["choices"][0]["delta"] for c in chunks if c["choices"]), "the router runs them; the client must not")
        self.assertEqual(len({c["id"] for c in chunks}), 1)
        roles = [c for c in chunks if c["choices"] and c["choices"][0]["delta"].get("role")]
        self.assertEqual(len(roles), 1, "the role is said once")
        kinds = [c["colibri"]["event"] for c in chunks if "colibri" in c]
        self.assertEqual(kinds, ["tool_call", "tool_result"])
        call = next(c["colibri"] for c in chunks if c.get("colibri", {}).get("event") == "tool_call")
        self.assertEqual((call["name"], call["arguments"], call["read_only"], call["how"]), ("files_read", {"path": self.notes}, True, "read-only"))
        result = next(c["colibri"] for c in chunks if c.get("colibri", {}).get("event") == "tool_result")
        self.assertTrue(result["ok"])
        self.assertIn("the notes say hello", result["text"])
        finals = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
        self.assertEqual([f["choices"][0]["finish_reason"] for f in finals], ["stop"], "one finish, at the very end")
        self.assertEqual(chunks[-1]["usage"], {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
        order = [("t" if c["choices"] else c["colibri"]["event"]) for c in chunks if c["choices"] or "colibri" in c]
        self.assertLess(order.index("tool_call"), order.index("tool_result"))
        self.assertEqual(order[0], "t")

    def test_a_streamed_answer_without_usage_asked_for_has_none(self):
        self.policy()
        self.model([{"text": "plain"}])
        _, events = self.stream({"colibri_tools": True})
        self.assertFalse(any(isinstance(e, dict) and "usage" in e for e in events))

    def test_a_streamed_model_error_before_anything_is_a_json_error(self):
        self.policy()
        self.model([{"error": {"message": "no model", "type": "server_error", "code": "x"}, "status": 500}])
        connection = http.client.HTTPConnection("127.0.0.1", self.rig.port, timeout=30)
        connection.request("POST", "/v1/chat/completions", json.dumps({"model": "text-a-colibri", "messages": [{"role": "user", "content": "x"}], "stream": True, "colibri_tools": True}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 500)
        self.assertIn("no model", response.read().decode())

    def test_the_two_requests_share_nothing_between_conversations(self):
        self.policy()
        model = self.model([{"text": "one"}])
        self.chat({"colibri_tools": True, "messages": [{"role": "user", "content": "first"}]})
        self.chat({"colibri_tools": True, "messages": [{"role": "user", "content": "second"}]})
        self.assertEqual([r[1]["messages"][0]["content"] for r in model.requests], ["first", "second"])


class ApprovalTest(ToolsTest):
    def write_call(self, content="written by the model", name="made.txt"):
        return ("files_write", {"path": os.path.join(self.scratch, name), "content": content})

    def answer(self, id_, allow, always=False):
        return self.rig.call("POST", "/v1/tools/pending", {"id": id_, "allow": allow, "always": always})

    def test_a_streamed_write_waits_for_a_yes_and_then_goes_on(self):
        self.policy()
        model = self.model([{"text": "", "calls": [self.write_call()]}, {"text": "I made the file."}])
        seen = {}

        def on_event(event):
            if isinstance(event, dict) and event.get("colibri", {}).get("event") == "approval":
                seen["approval"] = event["colibri"]
                seen["listed"] = self.rig.call("GET", "/v1/tools/pending")[3]["data"]
                seen["file_before"] = os.path.exists(os.path.join(self.scratch, "made.txt"))
                seen["answer"] = self.answer(event["colibri"]["id"], True)

        status, events = self.stream({"colibri_tools": True}, on_event)
        approval = seen["approval"]
        self.assertEqual((approval["tool"], approval["arguments"]["path"]), ("files_write", os.path.join(self.scratch, "made.txt")))
        self.assertIn("write 20 characters", approval["summary"])
        self.assertEqual([p["id"] for p in seen["listed"]], [approval["id"]])
        self.assertFalse(seen["file_before"], "nothing is written while the router waits")
        self.assertEqual(seen["answer"][0], 200)
        self.assertEqual(Path(self.scratch, "made.txt").read_text(), "written by the model")
        kinds = [e["colibri"]["event"] for e in events if isinstance(e, dict) and "colibri" in e]
        self.assertEqual(kinds, ["approval", "approval_answered", "tool_call", "tool_result"])
        self.assertTrue(next(e for e in events if isinstance(e, dict) and e.get("colibri", {}).get("event") == "approval_answered")["colibri"]["allowed"])
        self.assertEqual(self.rig.call("GET", "/v1/tools/pending")[3]["data"], [], "nothing is left waiting")
        self.assertIn("Wrote 20 characters", model.requests[1][1]["messages"][-1]["content"])

    def test_a_no_goes_back_to_the_model_and_the_file_is_not_written(self):
        self.policy()
        model = self.model([{"text": "", "calls": [self.write_call()]}, {"text": "Understood, I will not."}])

        def on_event(event):
            if isinstance(event, dict) and event.get("colibri", {}).get("event") == "approval":
                self.answer(event["colibri"]["id"], False)

        _, events = self.stream({"colibri_tools": True}, on_event)
        self.assertFalse(os.path.exists(os.path.join(self.scratch, "made.txt")))
        self.assertIn("declined", model.requests[1][1]["messages"][-1]["content"])
        kinds = [e["colibri"]["event"] for e in events if isinstance(e, dict) and "colibri" in e]
        self.assertEqual(kinds, ["approval", "approval_answered", "tool_call"], "declined: the call is shown, never run")
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events if isinstance(e, dict) and e["choices"])
        self.assertEqual(text, "Understood, I will not.")

    def test_no_answer_in_time_is_a_no(self):
        self.policy(approvalSeconds=1)
        model = self.model([{"text": "", "calls": [self.write_call()]}, {"text": "No answer came."}])
        started = time.monotonic()
        status, events = self.stream({"colibri_tools": True})
        self.assertLess(time.monotonic() - started, 10)
        self.assertFalse(os.path.exists(os.path.join(self.scratch, "made.txt")))
        self.assertIn("declined", model.requests[1][1]["messages"][-1]["content"])
        self.assertEqual(self.rig.call("GET", "/v1/tools/pending")[3]["data"], [])

    def test_an_answer_after_the_time_or_twice_is_a_404(self):
        self.policy(approvalSeconds=1)
        self.model([{"text": "", "calls": [self.write_call()]}, {"text": "x"}])
        ids = []
        self.stream({"colibri_tools": True}, lambda e: ids.append(e["colibri"]["id"]) if isinstance(e, dict) and e.get("colibri", {}).get("event") == "approval" else None)
        self.assertEqual(self.answer(ids[0], True)[0], 404)

        self.policy()
        self.model([{"text": "", "calls": [self.write_call()]}, {"text": "x"}])
        codes = []

        def twice(event):
            if isinstance(event, dict) and event.get("colibri", {}).get("event") == "approval":
                codes.append(self.answer(event["colibri"]["id"], True)[0])
                codes.append(self.answer(event["colibri"]["id"], False)[0])

        self.stream({"colibri_tools": True}, twice)
        self.assertEqual(codes[0], 200)
        self.assertIn(codes[1], (200, 404))
        self.assertTrue(os.path.exists(os.path.join(self.scratch, "made.txt")), "the first answer stands")

    def test_always_stops_the_questions_for_that_tool_in_this_conversation(self):
        self.policy()
        self.model([{"text": "", "calls": [self.write_call(name="one.txt")]}, {"text": "", "calls": [self.write_call(name="two.txt")]}, {"text": "both done"}])
        asked = []

        def on_event(event):
            if isinstance(event, dict) and event.get("colibri", {}).get("event") == "approval":
                asked.append(event["colibri"]["id"])
                self.answer(event["colibri"]["id"], True, always=True)

        _, events = self.stream({"colibri_tools": True}, on_event)
        self.assertEqual(len(asked), 1, "asked once, then the second write ran")
        self.assertTrue(os.path.exists(os.path.join(self.scratch, "one.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.scratch, "two.txt")))
        calls = [e["colibri"] for e in events if isinstance(e, dict) and e.get("colibri", {}).get("event") == "tool_call"]
        self.assertEqual([c["how"] for c in calls], ["approved", "approved for this session"])

    def test_always_does_not_carry_to_the_next_request(self):
        self.policy()
        self.model([{"text": "", "calls": [self.write_call(name="one.txt")]}, {"text": "done"}])
        asked = []

        def on_event(event):
            if isinstance(event, dict) and event.get("colibri", {}).get("event") == "approval":
                asked.append(1)
                self.answer(event["colibri"]["id"], True, always=True)

        self.stream({"colibri_tools": True}, on_event)
        self.model([{"text": "", "calls": [self.write_call(name="two.txt")]}, {"text": "done"}])
        self.stream({"colibri_tools": True}, on_event)
        self.assertEqual(len(asked), 2)

    def test_a_not_streamed_request_waits_too_and_another_connection_answers(self):
        self.policy()
        self.model([{"text": "", "calls": [self.write_call()]}, {"text": "done"}])
        result = {}
        thread = threading.Thread(target=lambda: result.update(response=self.chat({"colibri_tools": True})))
        thread.start()
        deadline = time.monotonic() + 15
        listed = []
        while not listed and time.monotonic() < deadline:
            listed = self.rig.call("GET", "/v1/tools/pending")[3]["data"]
            time.sleep(0.05)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["tool"], "files_write")
        self.assertIn("expires_in", listed[0])
        self.answer(listed[0]["id"], True)
        thread.join(15)
        status, _, _, body = result["response"]
        self.assertEqual(status, 200)
        self.assertEqual([(c["name"], c["how"], c["ok"]) for c in body["colibri"]["tool_calls"]], [("files_write", "approved", True)])
        self.assertTrue(os.path.exists(os.path.join(self.scratch, "made.txt")))

    def test_a_caller_who_leaves_while_the_router_waits_drops_the_call(self):
        patcher = patch.object(tools_loop, "HEARTBEAT", 0.1)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.policy(approvalSeconds=30)
        import request_log
        log_path = os.path.join(self.root, "left.jsonl")
        self.rig.server.request_log = request_log.RequestLog(log_path)
        self.model([{"text": "", "calls": [self.write_call()]}, {"text": "x"}])
        sock = socket.create_connection(("127.0.0.1", self.rig.port), timeout=30)
        body = json.dumps({"model": "text-a-colibri", "messages": [{"role": "user", "content": "x"}], "stream": True, "colibri_tools": True}).encode()
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        buffer = b""
        while b"approval" not in buffer:
            piece = sock.recv(4096)
            self.assertTrue(piece, "the stream ended before the approval")
            buffer += piece
        sock.close()                                                    # the caller goes away
        deadline = time.monotonic() + 15
        listed = [1]
        while listed and time.monotonic() < deadline:
            listed = self.rig.call("GET", "/v1/tools/pending")[3]["data"]
            time.sleep(0.2)
        self.assertEqual(listed, [], "the waiting call was dropped")
        self.assertFalse(os.path.exists(os.path.join(self.scratch, "made.txt")))
        deadline = time.monotonic() + 5
        notes = []
        while not notes and time.monotonic() < deadline:                 # the log line is written as the request ends
            if os.path.exists(log_path):
                notes = [r for r in map(json.loads, Path(log_path).read_text().splitlines()) if r.get("note") == "client left"]
            time.sleep(0.05)
        self.assertEqual(len(notes), 1, "the log says the caller left, and that nothing ran")
        self.assertEqual(notes[0]["tool_calls"], [], "no tool ran: it was still waiting for a yes")

    def test_a_write_outside_the_writable_folders_is_refused_by_the_tool_even_if_a_person_says_yes(self):
        self.policy()
        self.model([{"text": "", "calls": [("files_write", {"path": os.path.join(self.proj, "no.txt"), "content": "x"})]}, {"text": "x"}])

        def on_event(event):
            if isinstance(event, dict) and event.get("colibri", {}).get("event") == "approval":
                self.answer(event["colibri"]["id"], True)

        _, events = self.stream({"colibri_tools": True}, on_event)
        self.assertFalse(os.path.exists(os.path.join(self.proj, "no.txt")))
        result = next(e["colibri"] for e in events if isinstance(e, dict) and e.get("colibri", {}).get("event") == "tool_result")
        self.assertFalse(result["ok"])


class LogTest(ToolsTest):
    def test_the_log_says_which_tools_ran_and_on_what_but_not_what_was_read(self):
        import request_log
        self.policy()
        self.model([{"text": "", "calls": [("files_read", {"path": self.notes})]}, {"text": "done"}])
        log_path = os.path.join(self.root, "requests.jsonl")
        self.rig.server.request_log = request_log.RequestLog(log_path)
        self.chat({"colibri_tools": True})
        self.rig.call("POST", "/v1/tools/files_read", {"arguments": {"path": self.notes}})
        deadline = time.monotonic() + 5
        while True:                                  # the line is written just after the answer goes out
            text = Path(log_path).read_text() if os.path.exists(log_path) else ""
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            if any(r.get("path") == "/v1/tools/files_read" for r in records) or time.monotonic() > deadline:
                break
            time.sleep(0.05)
        loop = next(r for r in records if r.get("path") == "/v1/chat/completions" and r.get("tool_calls"))
        self.assertEqual([(c["tool"], c["ok"]) for c in loop["tool_calls"]], [("files_read", True)])
        direct = next(r for r in records if r.get("path") == "/v1/tools/files_read")
        self.assertEqual((direct["tool"], direct["ok"]), ("files_read", True))
        self.assertNotIn("the notes say hello", text, "the contents of what was read are not logged")


class PendingTest(unittest.TestCase):
    def test_a_call_is_answered_once_and_only_in_time(self):
        pending = tools_loop.Pending()
        item = pending.add("files_write", {"path": "x"}, "write x", 60)
        self.assertEqual([p["id"] for p in pending.listing()], [item["id"]])
        self.assertTrue(pending.answer(item["id"], True))
        self.assertFalse(pending.answer(item["id"], False), "the first answer stands")
        self.assertEqual(item["answer"], "yes")
        self.assertEqual(pending.listing(), [], "an answered call is no longer waiting")
        late = pending.add("files_write", {"path": "y"}, "write y", 60)
        late["expires"] = time.time() - 1
        self.assertFalse(pending.answer(late["id"], True), "too late")
        self.assertIsNone(late["answer"])
        self.assertEqual(pending.listing(), [])
        self.assertFalse(pending.answer("nonesuch", True))

    def test_always_is_only_for_a_yes(self):
        pending = tools_loop.Pending()
        for allow, always, want in ((True, True, "always"), (True, False, "yes"), (False, True, "no"), (False, False, "no")):
            item = pending.add("t", {}, "s", 60)
            pending.answer(item["id"], allow, always)
            self.assertEqual(item["answer"], want, (allow, always))

    def test_waiting_ends_with_the_answer_or_a_no_when_time_runs_out(self):
        pending = tools_loop.Pending()
        item = pending.add("t", {}, "s", 60)
        threading.Timer(0.1, lambda: pending.answer(item["id"], True)).start()
        self.assertEqual(pending.wait(item), "yes")
        slow = pending.add("t", {}, "s", 0.3)
        self.assertEqual(pending.wait(slow), "no")
        self.assertEqual(pending.listing(), [], "a finished wait leaves nothing behind")

    def test_a_wait_stops_when_the_caller_has_gone(self):
        pending = tools_loop.Pending()
        item = pending.add("t", {}, "s", 60)

        def gone():
            raise tools_loop.ClientGone()

        with patch.object(tools_loop, "HEARTBEAT", 0.05), self.assertRaises(tools_loop.ClientGone):
            pending.wait(item, tick=gone)
        self.assertEqual(pending.listing(), [])


class CoveredTest(unittest.TestCase):
    def test_the_routes_are_in_the_spec_and_the_docs(self):
        import test_api_spec
        for route in (("GET", "/v1/tools"), ("GET", "/v1/tools/pending"), ("POST", "/v1/tools/pending"), ("POST", "/v1/tools/")):
            self.assertIn(route, test_api_spec.source_routes())


if __name__ == "__main__":
    unittest.main()
