"""POST /v1/responses on the router: the Responses API (Codex) over the router's own chat endpoint (responses_shim.py).

The router answers HTTP/1.0 (a stream ends when the connection closes), so the shim's event stream must arrive as the raw body, not
in chunks; and the request it makes to /v1/chat/completions goes through the router like any other, with the caller's key and tags."""
import http.client
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_router import GB, Rig, entry  # noqa: E402

MODEL = "text-a-colibri"
CHAT = {MODEL: {"FAKE_CHAT": "1"}}


def make(test, **kw):
    return Rig(test, [entry("text-a", size=1 * GB)], env=CHAT, **kw)


def events(payload):
    """The `event: name` / `data: json` pairs of a server-sent stream."""
    out = []
    for block in payload.decode().split("\n\n"):
        name, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        if name is not None and data is not None:
            out.append((name, json.loads(data)))
    return out


def stream(rig, body, headers=None):
    """POST /v1/responses with a raw connection: (status, headers, raw payload)."""
    connection = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=60)
    try:
        connection.request("POST", "/v1/responses", body=json.dumps(body), headers={"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
    finally:
        connection.close()


class RouterResponsesTest(unittest.TestCase):
    def test_a_streamed_response_arrives_as_the_raw_body_of_an_http_1_0_answer(self):
        rig = make(self)
        status, headers, payload = stream(rig, {"model": MODEL, "input": "hi", "stream": True})
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "text/event-stream")
        self.assertNotIn("transfer-encoding", headers, "the router has no chunking: the body is the stream")
        self.assertEqual(headers.get("connection"), "close")
        names = [name for name, _ in events(payload)]
        self.assertEqual(names[0], "response.created")
        self.assertEqual(names[-1], "response.completed")
        text = "".join(data.get("delta", "") for name, data in events(payload) if name == "response.output_text.delta")
        self.assertTrue(text.startswith("Hello from text-a-colibri"), text)
        done = events(payload)[-1][1]["response"]
        self.assertEqual(done["status"], "completed")

    def test_without_stream_the_whole_response_comes_as_json(self):
        rig = make(self)
        status, headers, payload = stream(rig, {"model": MODEL, "input": "hi"})
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json")
        body = json.loads(payload)
        self.assertEqual((body["object"], body["status"]), ("response", "completed"))
        message = next(item for item in body["output"] if item["type"] == "message")
        self.assertTrue(message["content"][0]["text"].startswith("Hello from text-a-colibri"))
        self.assertEqual(body["usage"]["input_tokens"], 7)

    def test_a_function_call_comes_back_as_a_function_call_item(self):
        rig = make(self)
        tools = [{"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]
        status, _headers, payload = stream(rig, {"model": MODEL, "input": "TOOL please", "tools": tools})
        self.assertEqual(status, 200)
        calls = [item for item in json.loads(payload)["output"] if item["type"] == "function_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["arguments"]), {"city": "Rome"})

    def test_the_router_key_is_required_and_is_not_passed_on_to_the_model_worker(self):
        rig = make(self, api_key="secret")
        status, _h, payload = stream(rig, {"model": MODEL, "input": "hi"})
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(payload)["error"]["code"], "invalid_api_key")
        status, _h, payload = stream(rig, {"model": MODEL, "input": "hi"}, {"Authorization": "Bearer secret"})
        self.assertEqual(status, 200)
        self.assertIn("auth=None", payload.decode(), "the router checks the key itself and does not hand it to a worker")
        status, _h, payload = stream(rig, {"model": MODEL, "input": "hi"}, {"x-api-key": "secret"})
        self.assertEqual(status, 200, "a key in x-api-key is accepted too (the chat request made for it carries it)")
        status, _h, _p = stream(rig, {"model": MODEL, "input": "hi"}, {"Authorization": "Bearer wrong"})
        self.assertEqual(status, 401)

    def test_client_tags_go_on_to_the_chat_request(self):
        rig = make(self)
        tags = json.dumps({"app": "codex", "chat": "c-1"})
        status, _h, payload = stream(rig, {"model": MODEL, "input": "hi"}, {"X-Colibri-Tags": tags})
        self.assertEqual(status, 200)
        self.assertIn('"app": "codex"', payload.decode().replace('\\"', '"'))

    def test_the_hosts_the_router_allows_are_the_same_as_for_any_request(self):
        rig = make(self)
        connection = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=30)
        try:
            connection.putrequest("POST", "/v1/responses", skip_host=True)
            connection.putheader("Host", "evil.example")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", "2")
            connection.endheaders(b"{}")
            self.assertEqual(connection.getresponse().status, 403)
        finally:
            connection.close()

    def test_a_request_that_is_not_a_responses_request_is_a_400_and_an_unknown_model_the_routers_404(self):
        rig = make(self)
        connection = http.client.HTTPConnection("127.0.0.1", rig.port, timeout=30)
        try:
            connection.request("POST", "/v1/responses", body="not json", headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn("not a Responses request", response.read().decode())
        finally:
            connection.close()
        status, _h, payload = stream(rig, {"model": "nope-colibri", "input": "hi"})
        self.assertEqual(status, 404)
        self.assertIn("nope-colibri", payload.decode())

    def test_the_chat_endpoint_is_as_it_was(self):
        rig = make(self)
        status, _h, _p, body = rig.chat(MODEL, messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], MODEL)


if __name__ == "__main__":
    unittest.main()
