"""The shim's event stream is framed for the handler it runs on: chunks for HTTP/1.1, the raw body for HTTP/1.0 (colibri's router)."""
import http.client
import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

spec = importlib.util.spec_from_file_location("responses_framing_shim", Path(__file__).resolve().parents[1] / "responses_shim.py")
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)


def chat_stream():
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
              {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
              {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}]
    return "".join("data: " + json.dumps(c) + "\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


class Backend(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        body = chat_stream()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(body)


def proxy_for(version, backend_url):
    class Proxy(BaseHTTPRequestHandler):
        protocol_version = version

        def log_message(self, *args):
            pass

        def do_POST(self):
            shim.serve(self, urlsplit(backend_url))
    return Proxy


class FramingTest(unittest.TestCase):
    def run_against(self, version):
        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), proxy_for(version, "http://127.0.0.1:%d" % backend.server_address[1]))
        for server in (backend, proxy):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=30)
        self.addCleanup(connection.close)
        connection.request("POST", "/v1/responses", body=json.dumps({"model": "m", "input": "x", "stream": True}), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        headers = {k.lower(): v for k, v in response.getheaders()}
        return headers, response.read().decode()

    def names(self, payload):
        return [line[7:] for line in payload.split("\n") if line.startswith("event: ")]

    def test_an_http_1_1_handler_gets_chunks(self):
        headers, payload = self.run_against("HTTP/1.1")
        self.assertEqual(headers.get("transfer-encoding"), "chunked")
        self.assertEqual(self.names(payload)[0], "response.created")      # http.client has taken the chunk framing off
        self.assertEqual(self.names(payload)[-1], "response.completed")

    def test_an_http_1_0_handler_gets_the_raw_stream_and_a_closed_connection(self):
        headers, payload = self.run_against("HTTP/1.0")
        self.assertNotIn("transfer-encoding", headers)
        self.assertEqual(headers.get("connection"), "close")
        self.assertEqual(self.names(payload)[0], "response.created")
        self.assertEqual(self.names(payload)[-1], "response.completed")
        self.assertNotRegex(payload, r"^[0-9a-f]+\r\n", "no chunk sizes in the body")


if __name__ == "__main__":
    unittest.main()
