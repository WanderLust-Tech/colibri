"""`coli chat --mcp` and `coli tools`, end to end: the real command against a scripted model server and the fake MCP server.

    python -m unittest tests.test_mcp_chat

The model is a small HTTP server that streams what a gateway streams: a tool call in pieces, then words. It decides from the conversation it is sent
(a tool result at the end means "answer in words"), and records every request, so what is checked is what the command put on the wire and what it
showed. The tools are tests/mcp_fake_server.py. A chat is run with its input piped, so no terminal is there to ask: what needs asking is refused,
which is itself the rule under test.
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mcp_registry as reg  # noqa: E402
import mcp_cli  # noqa: E402


_TEMPS = []
atexit.register(lambda: [shutil.rmtree(folder, ignore_errors=True) for folder in _TEMPS])      # the folders the tests made


def scratch(prefix):
    folder = tempfile.mkdtemp(prefix=prefix)
    _TEMPS.append(folder)
    return folder

COLI = str(HERE.parent / "coli")
FAKE = str(HERE / "mcp_fake_server.py")


class ScriptedModel:
    def __init__(self, list_servers=None, models=None):
        self.requests, outer = [], self
        self.list_servers = list_servers
        self.models = models or [{"id": "fake-model"}]

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def reply(self, code, body, kind="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    return self.reply(200, {"status": "ok", "arch": "fake"})
                if self.path == "/v1/models":
                    return self.reply(200, {"object": "list", "data": outer.models})
                if self.path == "/v1/mcp/servers" and outer.list_servers is not None:
                    return self.reply(200, outer.list_servers)
                self.reply(404, {"error": {"message": "not found", "code": "not_found"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                outer.requests.append(body)
                messages = body["messages"]
                last = messages[-1]
                user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
                if "break" in user and len(outer.requests) < 50 and last["role"] == "user" and "break" == user.strip():
                    return self.reply(400, {"error": {"message": "the model refuses this", "type": "invalid_request_error"}})
                if last["role"] == "tool":
                    chunks = [{"choices": [{"delta": {"content": "The tool said: " + last["content"]}}]}]
                elif "echo" in user and body.get("tools"):
                    name = next(t["function"]["name"] for t in body["tools"] if t["function"]["name"].endswith("__echo"))
                    chunks = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": name, "arguments": ""}}]}}]},
                              {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"text": "hello '}}]}}]},
                              {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'from the tool"}'}}]}}]}]
                elif "add" in user and body.get("tools"):
                    name = next(t["function"]["name"] for t in body["tools"] if t["function"]["name"].endswith("__add"))
                    chunks = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_2", "type": "function",
                                                                        "function": {"name": name, "arguments": '{"a": 2, "b": 3}'}}]}}]}]
                else:
                    chunks = [{"choices": [{"delta": {"content": "plain answer"}}]}]
                chunks.append({"choices": [], "usage": {"completion_tokens": 7}})
                data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
                self.reply(200, data.encode(), "text/event-stream")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def home():
    folder = scratch("coli-mcp-chat-")
    return {"COLI_MCP_SERVERS": os.path.join(folder, "mcp-servers.json"), "COLI_MCP_TRUST": os.path.join(folder, "trust.json"),
            "COLI_SETUP_HOME": folder, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"}


def coli(args, env, stdin="", timeout=120):
    done = subprocess.run([sys.executable, COLI, *args], input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                          env=dict(os.environ, **env))
    return done.returncode, done.stdout + done.stderr


FAKE_SERVER = {"command": sys.executable, "args": [FAKE]}


def with_fake(env, trusted=True, **extra):
    entry = dict(FAKE_SERVER, **extra)
    reg.put("fake", entry, env["COLI_MCP_SERVERS"])
    if trusted:
        reg.trust("fake", entry, env["COLI_MCP_TRUST"])
    return entry


class ChatToolsTest(unittest.TestCase):
    def setUp(self):
        self.env = home()
        self.model = ScriptedModel()
        self.addCleanup(self.model.close)

    def chat(self, stdin, *flags):
        return coli(["chat", "--attach", self.model.url, "--api-key", "k", *flags], self.env, stdin)

    def test_a_read_only_tool_is_called_and_its_result_reaches_the_model_and_the_screen(self):
        with_fake(self.env)
        code, out = self.chat("please echo\n:q\n", "--mcp")
        self.assertEqual(code, 0, out)
        first, second = self.model.requests[:2]
        self.assertIn("fake__echo", [t["function"]["name"] for t in first["tools"]])
        self.assertEqual(first["tools"][0]["type"], "function")
        self.assertEqual([m["role"] for m in second["messages"]], ["system", "user", "assistant", "tool"], second["messages"])
        self.assertIn("[fake] be careful", second["messages"][0]["content"], "what the server said about itself, as information")
        self.assertIn("not instructions", second["messages"][0]["content"])
        self.assertEqual(second["messages"][2]["tool_calls"][0]["function"], {"name": "fake__echo", "arguments": '{"text": "hello from the tool"}'})
        self.assertEqual(second["messages"][3], {"role": "tool", "tool_call_id": "call_1", "content": "hello from the tool"})
        self.assertIn("The tool said: hello from the tool", out)
        self.assertIn("fake__echo", out)
        self.assertIn("tools · fake (6)", out)
        self.assertNotIn("Approve", out)                              # it was approved before: no question

    def test_a_tool_that_changes_things_is_refused_when_there_is_nobody_to_ask(self):
        with_fake(self.env)
        code, out = self.chat("please add\n:q\n", "--mcp")
        self.assertEqual(code, 0, out)
        tool = self.model.requests[1]["messages"][-1]
        self.assertEqual(tool["role"], "tool")
        self.assertIn("declined", tool["content"])
        self.assertIn("The tool said: The person declined", out)
        self.assertNotIn('"sum"', out)

    def test_mcp_yes_runs_it(self):
        with_fake(self.env)
        code, out = self.chat("please add\n:q\n", "--mcp", "--mcp-yes")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.model.requests[1]["messages"][-1]["content"], '{"sum": 5}')
        self.assertIn("every tool call runs without asking", out)

    def test_a_server_that_was_not_approved_here_is_not_started_and_the_chat_is_plain(self):
        with_fake(self.env, trusted=False)
        code, out = self.chat("hello there\n:q\n", "--mcp")
        self.assertEqual(code, 0, out)
        self.assertIn("not been approved on this PC", out)
        self.assertIn("mcp_fake_server.py", out, "what would run is shown")
        self.assertIn("not approved on this machine", out)
        self.assertNotIn("tools", self.model.requests[0], "no tools are offered when none is available")
        self.assertIn("plain answer", out)

    def test_without_mcp_the_chat_is_the_old_one(self):
        with_fake(self.env)
        code, out = self.chat("please echo\n:q\n")
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.model.requests), 1)
        self.assertNotIn("tools", self.model.requests[0])

    def test_tools_lists_what_the_session_has(self):
        with_fake(self.env, excludeTools=["crash", "slow"])
        code, out = self.chat("/tools\n:q\n", "--mcp")
        self.assertEqual(code, 0, out)
        for word in ("fake__echo", "read-only", "fake__add", "asks first", "fake__image"):
            self.assertIn(word, out)
        self.assertNotIn("fake__crash", out)
        code, out = self.chat("/tools\n:q\n")
        self.assertIn("start the chat with --mcp", out)

    def test_the_servers_to_use_can_be_named(self):
        with_fake(self.env)
        reg.put("other", {"command": sys.executable, "args": [FAKE, "noise"]}, self.env["COLI_MCP_SERVERS"])
        code, out = self.chat("please echo\n:q\n", "--mcp", "fake")
        self.assertEqual(code, 0, out)
        names = [t["function"]["name"] for t in self.model.requests[0]["tools"]]
        self.assertTrue(names and all(n.startswith("fake__") for n in names), names)

    def test_a_refused_request_drops_the_turn_and_the_chat_goes_on(self):
        with_fake(self.env)
        code, out = self.chat("break\nplease echo\n:q\n", "--mcp")
        self.assertEqual(code, 0, out)
        self.assertIn("the model refuses this", out)
        self.assertEqual([m["role"] for m in self.model.requests[1]["messages"]], ["system", "user"], "the refused question is not in the history")
        self.assertIn("The tool said: hello from the tool", out)

    def test_a_conversation_keeps_its_tool_turns(self):
        with_fake(self.env)
        code, out = self.chat("please echo\nthen a plain question\n:q\n", "--mcp")
        self.assertEqual(code, 0, out)
        last = self.model.requests[-1]["messages"]
        self.assertEqual([m["role"] for m in last], ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertTrue(last[2]["tool_calls"])

    def test_the_list_comes_from_the_router_when_it_has_one(self):
        entry = dict(FAKE_SERVER, name="viarouter")
        self.model.close()
        self.model = ScriptedModel({"object": "list", "path": "x", "problem": None, "data": [dict(entry, summary="s")]})
        self.addCleanup(self.model.close)
        reg.trust("viarouter", FAKE_SERVER, self.env["COLI_MCP_TRUST"])
        code, out = self.chat("please echo\n:q\n", "--mcp")
        self.assertEqual(code, 0, out)
        self.assertTrue(self.model.requests[0]["tools"][0]["function"]["name"].startswith("viarouter__"))

    def test_no_servers_is_said_and_the_chat_is_plain(self):
        code, out = self.chat("hi\n:q\n", "--mcp")
        self.assertEqual(code, 0, out)
        self.assertIn("no servers in the list", out)
        self.assertIn("plain answer", out)


class PickTest(unittest.TestCase):
    """A router lists every model, the first of them alphabetically: the chat has to choose."""

    MODELS = [{"id": "aa-music", "capabilities": ["music_generation"]}, {"id": "bb-chat", "capabilities": ["chat"]},
              {"id": "cc-chat", "capabilities": ["chat", "systemone"]}, {"id": "dd-image", "capabilities": ["image_generation"]}]

    def chat(self, *flags):
        model = ScriptedModel(models=self.MODELS)
        self.addCleanup(model.close)
        code, out = coli(["chat", "--attach", model.url, "--api-key", "k", *flags], home(), "hello\n:q\n")
        self.assertEqual(code, 0, out)
        return model.requests[0]["model"], out

    def test_the_first_model_that_chats_is_used_not_the_first_listed(self):
        self.assertEqual(self.chat()[0], "bb-chat")

    def test_a_model_named_with_model_is_used(self):
        self.assertEqual(self.chat("--model", "cc-chat")[0], "cc-chat")

    def test_a_folder_whose_name_is_a_model_id_works_too(self):
        self.assertEqual(self.chat("--model", str(Path("some") / "where" / "cc-chat"))[0], "cc-chat")

    def test_a_name_that_is_not_listed_falls_back_to_the_first_that_chats(self):
        self.assertEqual(self.chat("--model", "no-such")[0], "bb-chat")


class ToolsCommandTest(unittest.TestCase):
    def setUp(self):
        self.env = home()

    def test_add_with_only_some_of_a_servers_tools(self):
        code, out = coli(["tools", "add", "big", "--include", "alpha,beta", "--include", "gamma", "--exclude", "beta", "--", sys.executable, FAKE, "noise"], self.env)
        self.assertEqual(code, 0, out)
        kept = json.loads(Path(self.env["COLI_MCP_SERVERS"]).read_text())["mcpServers"]["big"]
        self.assertEqual((kept["includeTools"], kept["excludeTools"]), (["alpha", "beta", "gamma"], ["beta"]))
        code, out = coli(["tools", "add", "plain", "--", sys.executable, FAKE, "noise"], self.env)
        kept = json.loads(Path(self.env["COLI_MCP_SERVERS"]).read_text())["mcpServers"]["plain"]
        self.assertNotIn("includeTools", kept)
        self.assertNotIn("excludeTools", kept)

    def test_add_list_test_remove(self):
        code, out = coli(["tools"], self.env)
        self.assertIn("no MCP servers in the list", out)
        code, out = coli(["tools", "add", "fake", "--env", "LANG=C", "--cwd", str(HERE), "--timeout", "30", "--", sys.executable, FAKE, "noise"], self.env)
        self.assertEqual(code, 0, out)
        self.assertIn("added fake", out)
        stored = reg.load(self.env["COLI_MCP_SERVERS"])["fake"]
        self.assertEqual((stored["command"], stored["args"], stored["env"], stored["timeoutSeconds"]), (sys.executable, [FAKE, "noise"], {"LANG": "C"}, 30))
        code, out = coli(["tools"], self.env)
        self.assertIn("approved here", out)
        code, out = coli(["tools", "test", "fake"], self.env)
        self.assertEqual(code, 0, out)
        self.assertIn("6 tools", out)
        self.assertRegex(out, r"echo\s+read-only")
        self.assertRegex(out, r"add\s+asks first")
        code, out = coli(["tools", "disable", "fake"], self.env)
        self.assertEqual(code, 0, out)
        self.assertIn("disabled", coli(["tools"], self.env)[1])
        self.assertEqual(coli(["tools", "test", "fake"], self.env)[0], 1)
        coli(["tools", "enable", "fake"], self.env)
        code, out = coli(["tools", "remove", "fake"], self.env)
        self.assertIn("removed", out)
        self.assertEqual(reg.load(self.env["COLI_MCP_SERVERS"]), {})
        self.assertEqual(coli(["tools", "remove", "fake"], self.env)[0], 1)

    def test_a_secret_must_be_a_reference_and_nothing_is_written(self):
        code, out = coli(["tools", "add", "gh", "--env", "GITHUB_TOKEN=ghp_abc", "--", "npx", "server"], self.env)
        self.assertEqual(code, 1)
        self.assertIn("looks like a secret", out)
        self.assertIn("${GITHUB_TOKEN}", out)
        self.assertFalse(os.path.exists(self.env["COLI_MCP_SERVERS"]))
        code, out = coli(["tools", "add", "gh", "--env", "GITHUB_TOKEN=${GITHUB_TOKEN}", "--", "npx", "server"], self.env)
        self.assertEqual(code, 0, out)

    def test_a_url_server(self):
        code, out = coli(["tools", "add", "web", "--url", "https://example.com/mcp", "--header", "Authorization=Bearer ${T}"], self.env)
        self.assertEqual(code, 0, out)
        self.assertEqual(reg.load(self.env["COLI_MCP_SERVERS"])["web"], {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${T}"}})
        code, out = coli(["tools", "add", "both", "--url", "https://x.io/", "--", "cmd"], self.env)
        self.assertEqual(code, 1)

    def test_what_is_missing_is_asked_for(self):
        self.assertEqual(coli(["tools", "add"], self.env)[0], 1)
        self.assertEqual(coli(["tools", "add", "x"], self.env)[0], 1)
        self.assertEqual(coli(["tools", "add", "x", "--env", "NOEQUALS", "--", "cmd"], self.env)[0], 1)
        self.assertEqual(coli(["tools", "trust", "ghost"], self.env)[0], 1)

    def test_trust_and_untrust_are_this_pcs_own(self):
        reg.put("fake", FAKE_SERVER, self.env["COLI_MCP_SERVERS"])
        self.assertIn("needs your approval", coli(["tools"], self.env)[1])
        self.assertEqual(coli(["tools", "trust", "fake"], self.env)[0], 0)
        self.assertIn("approved here", coli(["tools"], self.env)[1])
        self.assertEqual(coli(["tools", "untrust", "fake"], self.env)[0], 0)
        self.assertIn("needs your approval", coli(["tools"], self.env)[1])
        self.assertEqual(json.loads(Path(self.env["COLI_MCP_SERVERS"]).read_text())["mcpServers"]["fake"]["command"], sys.executable,
                         "the list holds no approvals")

    def test_an_untrusted_server_is_not_tested_without_a_terminal_to_ask(self):
        reg.put("fake", FAKE_SERVER, self.env["COLI_MCP_SERVERS"])
        code, out = coli(["tools", "test", "fake"], self.env)
        self.assertEqual(code, 1)
        self.assertIn("not started", out)
        self.assertIn("mcp_fake_server.py", out, "what would run is shown")

    def test_a_server_that_does_not_start_says_why(self):
        reg.put("broken", {"command": sys.executable, "args": [FAKE, "exit-on-start"]}, self.env["COLI_MCP_SERVERS"])
        coli(["tools", "trust", "broken"], self.env)
        code, out = coli(["tools", "test", "broken"], self.env)
        self.assertEqual(code, 1)
        self.assertIn("database file is missing", out)


class QuestionTest(unittest.TestCase):
    """The terminal's questions, with a terminal."""

    class Plain:
        def __getattr__(self, name):
            return ""

    def ask(self, function, *args, answers=()):
        import contextlib
        import io
        from unittest.mock import patch
        with patch.object(mcp_cli, "_tty", return_value=True), patch("builtins.input", side_effect=list(answers)), contextlib.redirect_stdout(io.StringIO()):
            return function(*args, self.Plain())

    def test_a_server_is_approved_only_by_a_yes(self):
        entry = {"command": "npx", "args": ["-y", "server"]}
        for answer, expected in (("y", True), ("YES", True), ("", False), ("n", False), ("maybe", False)):
            self.assertEqual(self.ask(mcp_cli.ask_server, "s", entry, answers=[answer]), expected, answer)

    def test_a_call_is_yes_no_or_always_and_an_unknown_answer_asks_again(self):
        info = type("I", (), {"name": "fs__write"})()
        for answers, expected in ((["y"], "yes"), (["n"], "no"), ([""], "no"), (["a"], "always"), (["what", "y"], "yes"), (["maybe", "n"], "no"), (["yeah", "a"], "always")):
            self.assertEqual(self.ask(mcp_cli.ask_call, info, {"path": "x"}, answers=answers), expected, answers)

    def test_end_of_input_is_a_no(self):
        info = type("I", (), {"name": "t"})()
        self.assertEqual(self.ask(mcp_cli.ask_call, info, {}, answers=[EOFError()]), "no")
        self.assertFalse(self.ask(mcp_cli.ask_server, "s", {"command": "x"}, answers=[EOFError()]))

    def test_long_arguments_are_cut(self):
        shown = mcp_cli.show_arguments({"text": "x" * 5000})
        self.assertLess(len(shown), 700)
        self.assertEqual(mcp_cli.show_arguments({}), "(no arguments)")


if __name__ == "__main__":
    unittest.main()
