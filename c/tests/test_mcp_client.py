"""The MCP client (c/mcp_client.py) against a fake stdio server, fake HTTP servers, and colibri's own `coli mcp`.

    python -m unittest tests.test_mcp_client
"""
import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mcp_client as mc  # noqa: E402

FAKE = str(HERE / "mcp_fake_server.py")


def stdio(*flags, **extra):
    cfg = {"command": sys.executable, "args": [FAKE, *flags]}
    cfg.update(extra)
    return mc.connect("fake", cfg, connect_timeout=20)


class ContentTextTest(unittest.TestCase):
    def test_text_is_joined_and_what_a_model_cannot_read_is_named(self):
        text = mc.content_text({"content": [{"type": "text", "text": "one"}, {"type": "image", "mimeType": "image/png", "data": "AAAA" * 100},
                                            {"type": "resource", "resource": {"uri": "file:///x", "text": "inside"}},
                                            {"type": "resource", "resource": {"uri": "file:///y", "mimeType": "application/pdf", "blob": "AA"}},
                                            {"type": "resource_link", "uri": "http://h/z", "name": "Z"}, "junk"]})
        self.assertEqual(text.split("\n")[0], "one")
        self.assertIn("[image: image/png, about 300 bytes, not shown]", text)
        self.assertIn("inside", text)
        self.assertIn("[resource file:///y: application/pdf]", text)
        self.assertIn("[link: http://h/z (Z)]", text)

    def test_a_structured_result_stands_in_for_missing_text(self):
        self.assertEqual(mc.content_text({"content": [], "structuredContent": {"sum": 3}}), '{"sum": 3}')
        self.assertEqual(mc.content_text({"content": [{"type": "text", "text": "t"}], "structuredContent": {"sum": 3}}), "t")
        self.assertEqual(mc.content_text({}), "")


class StdioTest(unittest.TestCase):
    def test_start_list_call_close(self):
        client = stdio()
        try:
            self.assertEqual(client.info.get("name"), "fake")
            self.assertEqual(client.instructions, "be careful")
            tools = client.list_tools()
            self.assertEqual([t["name"] for t in tools], ["echo", "add", "fail", "slow", "crash", "image"])
            self.assertTrue(tools[0]["annotations"]["readOnlyHint"])
            self.assertEqual(client.call_tool("echo", {"text": "hello"}).text, "hello")
            added = client.call_tool("add", {"a": 2, "b": 3})
            self.assertEqual((added.text, added.is_error), ('{"sum": 5}', False))
            failed = client.call_tool("fail", {})
            self.assertEqual((failed.text, failed.is_error), ("it did not work", True))
            self.assertIn("a picture", client.call_tool("image", {}).text)
            with self.assertRaisesRegex(mc.McpError, "unknown tool"):
                client.call_tool("nothing", {})
        finally:
            client.close()
        self.assertIsNotNone(client.transport.process.poll(), "the server is stopped on close")

    def test_listing_follows_the_pages(self):
        client = stdio("page")
        try:
            self.assertEqual(len(client.list_tools()), 6)
        finally:
            client.close()

    def test_lines_that_are_not_messages_are_ignored(self):
        client = stdio("noise")
        try:
            self.assertEqual(client.call_tool("echo", {"text": "ok"}).text, "ok")
        finally:
            client.close()

    def test_what_a_server_asks_of_us_is_answered_and_does_not_get_in_the_way(self):
        client = stdio("asks-ping")
        try:
            self.assertEqual(client.call_tool("echo", {"text": "ok"}).text, "ok")
            deadline = time.time() + 5
            while "sampling refused" not in client.transport.stderr_text() and time.time() < deadline:
                time.sleep(0.1)
            seen = client.transport.stderr_text()
            self.assertIn("ping answered", seen)
            self.assertIn("sampling refused: -32601", seen, "a request we do not offer is refused, not left hanging")
        finally:
            client.close()

    def test_a_slow_call_times_out_and_the_server_is_told_to_stop_it(self):
        client = stdio()
        try:
            started = time.time()
            with self.assertRaises(mc.McpTimeout):
                client.call_tool("slow", {"seconds": 2}, timeout=0.5)
            self.assertLess(time.time() - started, 1.5)
            deadline = time.time() + 8
            while "cancelled" not in client.transport.stderr_text() and time.time() < deadline:
                time.sleep(0.1)
            self.assertIn("cancelled", client.transport.stderr_text())
        finally:
            client.close()

    def test_a_server_that_dies_in_a_call_says_what_it_printed(self):
        client = stdio()
        try:
            with self.assertRaises(mc.McpError) as caught:
                client.call_tool("crash", {})
            self.assertIn("segmentation fault", str(caught.exception))
            self.assertIn("code 7", str(caught.exception))
            with self.assertRaisesRegex(mc.McpError, "not running|ended"):
                client.call_tool("echo", {"text": "x"})
        finally:
            client.close()

    def test_a_server_that_cannot_start_says_why(self):
        with self.assertRaises(mc.McpError) as caught:
            stdio("exit-on-start")
        self.assertIn("database file is missing", str(caught.exception))

    def test_a_server_that_refuses_the_handshake(self):
        with self.assertRaisesRegex(mc.McpError, "unsupported protocol version"):
            stdio("bad-init")

    def test_a_command_that_is_not_there(self):
        with self.assertRaisesRegex(mc.McpError, "was not found"):
            mc.connect("x", {"command": "definitely-not-a-command-xyz"})

    def test_environment_and_folder_reach_the_server(self):
        folder = str(HERE)
        cfg = {"command": sys.executable, "args": ["-c", "import os,sys,json\n"
               "for l in sys.stdin:\n m=json.loads(l)\n i=m.get('id')\n"
               " if m.get('method')=='initialize': print(json.dumps({'jsonrpc':'2.0','id':i,'result':{'serverInfo':{'name':os.environ.get('COLI_T')+'|'+os.getcwd()}}}),flush=True)\n"],
               "env": {"COLI_T": "value"}, "cwd": folder}
        client = mc.connect("x", cfg, connect_timeout=20)
        try:
            self.assertEqual(client.info["name"], "value|" + folder)
        finally:
            client.close()

    def test_a_server_that_never_answers_the_handshake_times_out_with_its_stderr(self):
        cfg = {"command": sys.executable, "args": ["-c", "import sys,time\nsys.stderr.write('warming up\\n');sys.stderr.flush()\ntime.sleep(30)"],
               "connectTimeoutSeconds": 1}
        started = time.time()
        with self.assertRaisesRegex(mc.McpError, "did not start in 1 s.*warming up"):
            mc.connect("slow", cfg)
        self.assertLess(time.time() - started, 15)


class FakeHttp:
    """A Streamable HTTP server: `mode` is "json" or "sse"; it hands out a session id and requires it afterwards."""

    def __init__(self, mode="json", token=None):
        outer = self
        self.mode, self.seen, self.deleted = mode, [], False

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_DELETE(self):
                outer.deleted = self.headers.get("Mcp-Session-Id") == "sess-1"
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                outer.seen.append((body.get("method") or f"answer:{body.get('id')}", self.headers.get("Mcp-Session-Id"),
                                   self.headers.get("Authorization"), self.headers.get("MCP-Protocol-Version")))
                if token and self.headers.get("Authorization") != f"Bearer {token}":
                    self.send_response(401)
                    self.send_header("Content-Length", "6")
                    self.end_headers()
                    self.wfile.write(b"denied")
                    return
                method = body.get("method")
                if method == "notifications/initialized":
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if "method" in body and "id" not in body:
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if "result" in body and "method" not in body:             # our answer to a server request
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if method == "initialize":
                    result = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "http-fake"}}
                elif method == "tools/list":
                    result = {"tools": [{"name": "echo", "description": "d", "inputSchema": {"type": "object"}}]}
                elif method == "tools/call":
                    result = {"content": [{"type": "text", "text": "http:" + body["params"]["arguments"].get("text", "")}]}
                else:
                    result = {}
                reply = {"jsonrpc": "2.0", "id": body["id"], "result": result}
                if outer.mode == "sse":
                    data = ("event: message\ndata: " + json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {}}) + "\n\n"
                            + "data: " + json.dumps({"jsonrpc": "2.0", "id": 77, "method": "ping"}) + "\n\n"
                            + "event: message\ndata: " + json.dumps(reply) + "\n\n").encode()
                    kind = "text/event-stream"
                else:
                    data, kind = json.dumps(reply).encode(), "application/json"
                self.send_response(200)
                self.send_header("Content-Type", kind)
                if method == "initialize":
                    self.send_header("Mcp-Session-Id", "sess-1")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/mcp"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class HttpTest(unittest.TestCase):
    def run_mode(self, mode):
        server = FakeHttp(mode, token="tok")
        try:
            client = mc.connect("h", {"url": server.url, "headers": {"Authorization": "Bearer tok"}}, connect_timeout=10)
            self.assertEqual(client.info["name"], "http-fake")
            self.assertEqual([t["name"] for t in client.list_tools()], ["echo"])
            self.assertEqual(client.call_tool("echo", {"text": "hi"}).text, "http:hi")
            client.close()
            methods = [m for m, *_ in server.seen if not m.startswith("answer:")]
            self.assertEqual(methods[:3], ["initialize", "notifications/initialized", "tools/list"])
            self.assertIsNone(server.seen[0][1])                                          # no session yet at the handshake
            self.assertTrue(all(s[1] == "sess-1" for s in server.seen[1:] if s[0] in ("tools/list", "tools/call")))
            self.assertTrue(all(s[2] == "Bearer tok" for s in server.seen))
            self.assertTrue(all(s[3] == "2025-06-18" for s in server.seen))
            self.assertTrue(server.deleted, "the session is ended on close")
            return server
        finally:
            server.close()

    def test_json_answers(self):
        self.run_mode("json")

    def test_event_stream_answers_with_a_ping_in_the_stream(self):
        server = self.run_mode("sse")
        self.assertIn("answer:77", [m for m, *_ in server.seen], "the ping the server asked for is answered")

    def test_a_refused_key_is_an_error_with_the_status(self):
        server = FakeHttp("json", token="right")
        try:
            with self.assertRaisesRegex(mc.McpError, "HTTP 401.*denied"):
                mc.connect("h", {"url": server.url, "headers": {"Authorization": "Bearer wrong"}}, connect_timeout=10)
        finally:
            server.close()

    def test_nothing_listening(self):
        with self.assertRaisesRegex(mc.McpError, "cannot be reached"):
            mc.connect("h", {"url": "http://127.0.0.1:9/mcp"}, connect_timeout=5)


class ColiMcpTest(unittest.TestCase):
    """The repository's own MCP server, started the way a user would."""

    def test_coli_mcp_lists_tools_and_answers_one(self):
        client = mc.connect("colibri", {"command": sys.executable, "args": [str(HERE.parent / "mcp_server.py")]}, connect_timeout=30)
        try:
            names = [t["name"] for t in client.list_tools()]
            self.assertTrue(names, "the server has tools")
            self.assertTrue(any("hardware" in n or "detect" in n for n in names), names)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
