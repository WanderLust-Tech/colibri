"""Gemma 4 through the gateway: what the engine is sent, and what a reply becomes.

    python -m unittest tests.test_gemma4_gateway

The engine is a double (the real one is tests/test_gemma4.py's subject): what is checked here is the
prompt the gateway builds from a request, the engine's frames turned into text (Gemma's thinking
channel in the gateway's words), the tool calls read back out of the reply, and the refusals.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import gemma4_chat  # noqa: E402
import openai_server  # noqa: E402
from openai_server import Engine, read_engine_turn  # noqa: E402
from test_openai_server import (FakeProcess, ScriptedEngine, _error_body, _post_chat,  # noqa: E402
                                _post_json, _spawn_test_server)

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}
CALL = '<|tool_call>call:get_weather{city:<|"|>Rome<|"|>,days:3}<tool_call|>'


class GatewayTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("openai_server.ARCH", "gemma4")
        patcher.start()
        self.addCleanup(patcher.stop)

    def server(self, chunks, **kw):
        self.engine = ScriptedEngine(chunks=chunks, supports_logprobs_echo=False, supports_tok_ids=False, **kw)
        return _spawn_test_server(self, self.engine)

    def chat(self, base, **body):
        body.setdefault("model", "test-model")
        body.setdefault("messages", [{"role": "user", "content": "Weather in Rome?"}])
        with _post_chat(base, body) as response:
            return json.load(response)

    def test_the_engine_is_sent_the_templates_prompt(self):
        base = self.server(["Hi"])
        reply = self.chat(base, messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}])
        self.assertEqual(self.engine.calls[0][0], "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nHello<turn|>\n"
                                                  "<|turn>model\n<|channel>thought\n<channel|>")
        self.assertEqual(reply["choices"][0]["message"]["content"], "Hi")
        self.assertEqual(reply["choices"][0]["finish_reason"], "stop")

    def test_thinking_on_leaves_the_turn_open_and_splits_the_reasoning(self):
        base = self.server(["Let me ", "think", "</think>", "The answer"])
        reply = self.chat(base, enable_thinking=True)
        self.assertTrue(self.engine.calls[0][0].startswith("<bos><|turn>system\n<|think|>\n<turn|>\n"))
        self.assertTrue(self.engine.calls[0][0].endswith("<|turn>model\n"))
        message = reply["choices"][0]["message"]
        self.assertEqual(message["content"], "The answer")
        self.assertEqual(message["reasoning_content"], "Let me think")

    def test_thinking_off_is_all_answer(self):
        base = self.server(["Plain ", "answer"])
        message = self.chat(base)["choices"][0]["message"]
        self.assertEqual(message["content"], "Plain answer")
        self.assertNotIn("reasoning_content", message)

    def test_a_tool_call_becomes_tool_calls(self):
        base = self.server(["Checking. ", CALL])
        reply = self.chat(base, tools=[WEATHER])
        self.assertIn("<|tool>declaration:get_weather{description:<|\"|>Weather of a city.<|\"|>", self.engine.calls[0][0])
        choice = reply["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        calls = choice["message"]["tool_calls"]
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Rome", "days": 3})
        self.assertEqual(choice["message"]["content"].strip(), "Checking.")

    def test_a_tool_call_after_reasoning(self):
        base = self.server(["I need the weather", "</think>", CALL])
        reply = self.chat(base, tools=[WEATHER], enable_thinking=True)
        choice = reply["choices"][0]
        self.assertEqual(choice["message"]["reasoning_content"], "I need the weather")
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "get_weather")

    def test_a_tool_call_streams_as_tool_calls_not_as_text(self):
        base = self.server(["Checking. ", "<|tool_call>call:get_weather{city:", '<|"|>Rome<|"|>}<tool_call|>'])
        body = {"model": "test-model", "stream": True, "tools": [WEATHER],
                "messages": [{"role": "user", "content": "Weather?"}]}
        with _post_json(base, "/v1/chat/completions", body) as response:
            text = response.read().decode()
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
        content = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e["choices"])
        self.assertNotIn("<|tool_call>", content)
        self.assertNotIn("tool_call|>", content)
        names = [c["function"]["name"] for e in events if e["choices"]
                 for c in e["choices"][0]["delta"].get("tool_calls") or [] if c.get("function", {}).get("name")]
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(events[-1]["choices"][0]["finish_reason"] if events[-1]["choices"] else
                         events[-2]["choices"][0]["finish_reason"], "tool_calls")

    def test_a_tool_result_goes_back_in_the_models_syntax(self):
        base = self.server(["Sunny."])
        self.chat(base, tools=[WEATHER], messages=[
            {"role": "user", "content": "Weather in Rome?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome", "days": 3}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "24 C"}])
        prompt = self.engine.calls[0][0]
        self.assertIn('<|tool_call>call:get_weather{city:<|"|>Rome<|"|>,days:3}<tool_call|>', prompt)
        self.assertTrue(prompt.endswith('<|tool_response>response:get_weather{value:<|"|>24 C<|"|>}<tool_response|>'))

    def test_a_picture_is_a_clear_400(self):
        base = self.server(["x"])
        status, error = _error_body(self, lambda: self.chat(base, messages=[{"role": "user", "content": [
            {"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]))
        self.assertEqual(status, 400)
        self.assertEqual(error["code"], "unsupported_content_type")
        self.assertEqual(self.engine.calls, [])

    def test_the_model_is_listed_under_its_family_name(self):
        self.assertEqual(openai_server.family_by_id("gemma4").default_model_id, "gemma-4-26b-a4b-colibri")


class DispatcherTest(unittest.TestCase):
    """The real Engine class on canned frames: the exact SUBMIT, and the channel's translation."""

    def test_request_and_response_transcript_is_byte_exact(self):
        prompt = gemma4_chat.render([{"role": "user", "content": "hi"}], enable_thinking=True)
        payload = prompt.encode("utf-8")
        expected = f"SUBMIT 1 0 {len(payload)} 4 0.25 0.9\n".encode() + payload + b"\n"

        def respond(process, frame):
            self.assertEqual(frame, expected)
            process.stdout.feed(
                b"ACCEPT 1 12\n"
                b"DATA 1 10\n<|channel>\n" b"DATA 1 7\nthought\n" b"DATA 1 1\n\n\n"
                b"DATA 1 5\nHmm, \n" b"DATA 1 4\nokay\n" b"DATA 1 10\n<channel|>\n"
                b"DATA 1 6\nHello!\n"
                b"DONE 1 STAT 4 2.500 50.0 1.25 12 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "gemma4"), patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("gemma4", "model")
            chunks = []
            stats = engine.generate(prompt, 4, 0.25, 0.9, chunks.append)
        engine.close()
        self.assertEqual("".join(chunks), "Hmm, okay</think>Hello!")
        self.assertEqual(stats["prompt_tokens"], 12)
        self.assertEqual(process.writes, [expected])

    def test_a_reply_that_ends_inside_the_opening_is_not_lost(self):
        def respond(process, frame):
            process.stdout.feed(b"DATA 1 10\n<|channel>\n" b"DATA 1 3\ntho\n" b"DONE 1 STAT 2 1.0 50.0 1.25 5 1\n")

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "gemma4"), patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("gemma4", "model")
            chunks = []
            engine.generate("<bos>x", 4, 0.0, 1.0, chunks.append)
        engine.close()
        self.assertEqual("".join(chunks), "tho")

    def test_other_families_are_not_touched(self):
        def respond(process, frame):
            process.stdout.feed(b"DATA 1 10\n<|channel>\n" b"DONE 1 STAT 1 1.0 50.0 1.25 5 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "olmoe"), patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("olmoe", "model")
            chunks = []
            engine.generate("x", 4, 0.0, 1.0, chunks.append)
        engine.close()
        self.assertEqual("".join(chunks), "<|channel>")


if __name__ == "__main__":
    unittest.main()
