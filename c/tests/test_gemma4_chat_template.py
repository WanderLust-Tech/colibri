#!/usr/bin/env python3
"""Gemma 4's chat format: the gateway's renderer against the release's own template, and the reply parsers.

The gateway renders prompts by hand instead of running jinja per request, and a hand copy can drift from
the original while the model keeps answering. Here the real template (tests/fixtures/gemma4_chat_template.jinja,
Apache-2.0, from google/gemma-4-26B-A4B-it, sha256 ae53464b...) is rendered with jinja2 the way transformers
does and compared byte for byte with gemma4_chat.render.

One documented difference: an OpenAI client sends tool-call arguments as a JSON string, which the template
refuses; the gateway reads it as the object it is. The reference side of each case gets the object.
"""
import copy
import hashlib
import itertools
import json
import random
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import gemma4_chat as g  # noqa: E402

TEMPLATE = HERE / "fixtures" / "gemma4_chat_template.jinja"
TEMPLATE_SHA256 = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"

try:
    import jinja2
    import jinja2.sandbox
except ImportError:                                   # pragma: no cover
    jinja2 = None

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "the city"},
        "days": {"type": "integer", "description": "how many"},
        "units": {"type": "string", "enum": ["C", "F"], "description": "temperature unit"},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "labels"},
        "where": {"type": "object", "description": "a place", "properties": {
            "lat": {"type": "number"}, "lon": {"type": "number", "nullable": True}}, "required": ["lat"]}},
        "required": ["city", "days"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search", "description": "Search the web.", "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}}, "required": ["query"]}}}
BARE = {"type": "function", "function": {"name": "ping", "description": "no parameters"}}
RECORDS = {"type": "function", "function": {
    "name": "records", "description": "list of objects", "parameters": {"type": "object", "properties": {
        "rows": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "name": {"type": "string"}}, "required": ["id"]}}}}}}


def call(name, args, ident=None):
    c = {"type": "function", "function": {"name": name, "arguments": args}}
    if ident:
        c["id"] = ident
    return c


CASES = {
    "one user turn": {"messages": [{"role": "user", "content": "hello"}]},
    "system and user": {"messages": [{"role": "system", "content": "  Be brief.  "}, {"role": "user", "content": "Capital of France?"}]},
    "developer role": {"messages": [{"role": "developer", "content": "Rules."}, {"role": "user", "content": "hi"}]},
    "system as parts": {"messages": [{"role": "system", "content": [{"type": "text", "text": " one "}, {"type": "text", "text": "two"}]},
                                     {"role": "user", "content": "hi"}]},
    "history": {"messages": [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"},
                             {"role": "user", "content": "and 2+2?"}]},
    "user parts": {"messages": [{"role": "user", "content": [{"type": "text", "text": " a "}, {"type": "text", "text": " b "}]}]},
    "a picture in a user turn": {"messages": [{"role": "user", "content": [
        {"type": "text", "text": " what is this? "}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": " and this "}, {"type": "image", "image": "x"}]}]},
    "two pictures and a system turn": {"messages": [
        {"role": "system", "content": "Describe pictures."},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u1"}}, {"type": "image_url", "image_url": {"url": "u2"}}]},
        {"role": "assistant", "content": "Two pictures."}, {"role": "user", "content": "more"}]},
    "pictures in a tool result": {"tools": [SEARCH], "messages": [
        {"role": "user", "content": "find it"},
        {"role": "assistant", "content": "", "tool_calls": [call("search", {"query": "cats"}, "c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": [
            {"type": "text", "text": "two hits"}, {"type": "image_url", "image_url": {"url": "u1"}},
            {"type": "image_url", "image_url": {"url": "u2"}}]}]},
    "an old turn's reasoning is dropped": {"messages": [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "reasoning_content": "old thoughts"},
        {"role": "user", "content": "q2"}]},
    "reasoning after the last user turn is kept": {"messages": [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "reasoning_content": "fresh thoughts"}]},
    "thinking channels inside an answer are stripped": {"messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "<|channel>thought\nhmm<channel|>  the answer  "},
        {"role": "user", "content": "more"}]},
    "tools, no system": {"tools": [WEATHER, SEARCH], "messages": [{"role": "user", "content": "Weather in Rome?"}]},
    "tools and a system": {"tools": [WEATHER], "messages": [{"role": "system", "content": "Use the tools."},
                                                            {"role": "user", "content": "Weather?"}]},
    "a tool without parameters": {"tools": [BARE], "messages": [{"role": "user", "content": "ping"}]},
    "array of objects": {"tools": [RECORDS], "messages": [{"role": "user", "content": "go"}]},
    "an agent loop": {"tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "Weather in Rome and Milan?"},
        {"role": "assistant", "content": "Checking both.", "tool_calls": [
            call("get_weather", {"city": "Rome", "days": 3, "tags": ["a", "b"], "where": {"lat": 41.9, "lon": None}}, "a"),
            call("get_weather", {"city": "Milan", "days": 3}, "b")]},
        {"role": "tool", "tool_call_id": "a", "content": "sun, 24 C"},
        {"role": "tool", "tool_call_id": "b", "content": "rain, 17 C"},
        {"role": "assistant", "content": "Rome is sunny, Milan rainy."},
        {"role": "user", "content": "And Naples?"}]},
    "a tool result last": {"tools": [SEARCH], "messages": [
        {"role": "user", "content": "Find colibri"},
        {"role": "assistant", "content": "", "tool_calls": [call("search", {"query": "colibri"}, "x")]},
        {"role": "tool", "tool_call_id": "x", "content": "a pure-C engine"}]},
    "a tool result with no id": {"tools": [SEARCH], "messages": [
        {"role": "user", "content": "Find colibri"},
        {"role": "assistant", "content": "", "tool_calls": [call("search", {"query": "colibri"})]},
        {"role": "tool", "name": "search", "content": "found"}]},
    "a call nobody answered": {"tools": [SEARCH], "messages": [
        {"role": "user", "content": "Find colibri"},
        {"role": "assistant", "content": "", "tool_calls": [call("search", {"query": "colibri"}, "x")]}]},
    "tool result as parts": {"tools": [SEARCH], "messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [call("search", {"query": "q"}, "x")]},
        {"role": "tool", "tool_call_id": "x", "content": [{"type": "text", "text": "one "}, {"type": "text", "text": "two"}]}]},
    "arguments of every kind": {"tools": [SEARCH], "messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [call("search", {
            "z": "text", "a": 1, "m": 2.5, "flag": True, "off": False, "nothing": None,
            "nested": {"b": [1, "two", {"c": False}], "A": "x"}, "empty": {}, "list": []}, "x")]},
        {"role": "tool", "tool_call_id": "x", "content": "ok"}]},
    "two assistant messages in a row": {"messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "part one"},
        {"role": "assistant", "content": "part two"}, {"role": "user", "content": "next"}]},
    "a system message in the middle": {"messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
        {"role": "system", "content": "new rule"}, {"role": "user", "content": "q2"}]},
    "an empty assistant turn": {"messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "q2"}]},
}


def reference_render(case, enable_thinking=False, add_generation_prompt=True, preserve_thinking=False):
    """The template, rendered the way transformers' apply_chat_template does."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
    template = env.from_string(TEMPLATE.read_text(encoding="utf-8"))
    kwargs = {"messages": case["messages"], "add_generation_prompt": add_generation_prompt,
              "enable_thinking": enable_thinking, "preserve_thinking": preserve_thinking, "bos_token": "<bos>"}
    if case.get("tools"):
        kwargs["tools"] = case["tools"]
    return template.render(**kwargs)


def as_client_sends(messages):
    """The same history with tool-call arguments as JSON strings, the OpenAI shape."""
    out = copy.deepcopy(messages)
    for message in out:
        for c in message.get("tool_calls") or []:
            fn = c["function"]
            if isinstance(fn.get("arguments"), dict):
                fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
    return out


@unittest.skipUnless(jinja2, "jinja2 is not installed")
class TemplateTest(unittest.TestCase):

    def test_template_fixture_is_the_release(self):
        self.assertEqual(hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(), TEMPLATE_SHA256)

    def test_every_case_matches_the_template_byte_for_byte(self):
        for (name, case), thinking, generation in itertools.product(CASES.items(), (False, True), (True, False)):
            with self.subTest(case=name, thinking=thinking, generation_prompt=generation):
                want = reference_render(case, thinking, generation)
                got = g.render(as_client_sends(case["messages"]), case.get("tools"), thinking, generation)
                self.assertEqual(got, want)

    def test_preserved_thinking_matches_too(self):
        for name in ("an agent loop", "reasoning after the last user turn is kept", "an old turn's reasoning is dropped"):
            case = copy.deepcopy(CASES[name])
            if name == "an agent loop":
                case["messages"][1]["reasoning_content"] = "I should look up both"
            for thinking in (False, True):
                with self.subTest(case=name, thinking=thinking):
                    want = reference_render(case, thinking, True, preserve_thinking=True)
                    got = g.render(as_client_sends(case["messages"]), case.get("tools"), thinking, True, True)
                    self.assertEqual(got, want)

    def test_reasoning_field_is_read_under_either_name(self):
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "reasoning": "r"}]
        want = reference_render({"messages": messages})
        self.assertEqual(g.render(messages), want)
        self.assertIn("<|channel>thought\nr\n<channel|>", want)

    def test_audio_and_video_are_refused_not_dropped(self):
        for part in ({"type": "input_audio", "input_audio": {}}, {"type": "audio"}, {"type": "video"}):
            with self.assertRaises(g.ChatError) as caught:
                g.render([{"role": "user", "content": [{"type": "text", "text": "what is this"}, part]}])
            self.assertEqual(caught.exception.code, "unsupported_content_type")
            self.assertEqual(caught.exception.param, "messages.0.content.1")
            self.assertIn("audio", str(caught.exception))

    def test_a_picture_becomes_what_the_processor_makes_of_the_templates_marker(self):
        """The template writes <|image|>; the processor turns each into begin, one placeholder per soft token, end."""
        part = {"type": "image_url", "_soft_tokens": 3, "_image_index": 0}
        bare = g.render([{"role": "user", "content": [{"type": "text", "text": "what?"}, {"type": "image_url"}]}])
        self.assertIn("<|turn>user\nwhat?<|image|><turn|>", bare)
        order = []
        got = g.render([{"role": "user", "content": [{"type": "text", "text": "what?"}, part]}], image_order=order)
        self.assertEqual(got, bare.replace("<|image|>", "<|image><|image|><|image|><|image|><image|>"))
        self.assertEqual(order, [0])
        # a tool result's pictures follow the response block, as the template writes them
        messages = [{"role": "user", "content": "find"},
                    {"role": "assistant", "content": "", "tool_calls": [call("search", {"query": "x"}, "c1")]},
                    {"role": "tool", "tool_call_id": "c1", "content": [
                        {"type": "text", "text": "hit"}, {"type": "image_url", "_soft_tokens": 1, "_image_index": 0},
                        {"type": "image_url", "_soft_tokens": 2, "_image_index": 1}]}]
        order = []
        got = g.render(messages, [SEARCH], image_order=order)
        self.assertIn("<tool_response|><|image><|image|><image|><|image><|image|><|image|><image|>", got)
        self.assertEqual(order, [0, 1])

    def test_a_count_that_is_not_a_positive_integer_is_a_bare_marker(self):
        for count in (0, -1, True, "3", None, 2.5):
            part = {"type": "image_url", "_soft_tokens": count}
            self.assertEqual(g.image_marker(part), "<|image|>", count)

    def test_bad_input_is_refused_by_name(self):
        for bad, param in (([], "messages"), (None, "messages"), ("hi", "messages"),
                           ([{"role": "wizard", "content": "x"}], "messages.0.role"),
                           ([{"role": "user", "content": 5}], "messages.0.content"),
                           (["x"], "messages.0")):
            with self.subTest(bad=bad), self.assertRaises(g.ChatError) as caught:
                g.render(bad)
            self.assertEqual(caught.exception.param, param)
        with self.assertRaises(g.ChatError) as caught:
            g.render([{"role": "user", "content": "x"},
                      {"role": "assistant", "content": "", "tool_calls": [call("f", "{not json")]}])
        self.assertEqual(caught.exception.param, "messages.1.tool_calls.0.function.arguments")
        with self.assertRaises(g.ChatError):
            g.render([{"role": "user", "content": "x"}], tools=[{"type": "function"}])

    def test_the_prompt_ends_where_the_model_starts(self):
        got = g.render([{"role": "user", "content": "hi"}])
        self.assertTrue(got.endswith("<|turn>model\n<|channel>thought\n<channel|>"))
        got = g.render([{"role": "user", "content": "hi"}], enable_thinking=True)
        self.assertTrue(got.endswith("<|turn>model\n"))
        self.assertTrue(got.startswith("<bos><|turn>system\n<|think|>\n<turn|>\n"))


def values(rng, depth=0):
    """A random JSON value a tool call might carry; no string holds the string delimiter."""
    kind = rng.choice(("str", "int", "float", "bool", "null", "list", "dict") if depth < 3 else ("str", "int", "bool"))
    if kind == "str":
        return "".join(rng.choice("abc xyz,:{}[]\n\"'é日") for _ in range(rng.randint(0, 8)))
    if kind == "int":
        return rng.randint(-50, 50)
    if kind == "float":
        return rng.choice((0.5, -2.25, 3.0, 1e-3))
    if kind == "bool":
        return rng.choice((True, False))
    if kind == "null":
        return None
    if kind == "list":
        return [values(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {f"k{i}_{rng.choice('abc')}": values(rng, depth + 1) for i in range(rng.randint(0, 3))}


class ReplyTest(unittest.TestCase):

    def test_arguments_round_trip_through_the_templates_own_syntax(self):
        rng = random.Random(5)
        for _ in range(300):
            args = {f"p{i}": values(rng) for i in range(rng.randint(0, 4))}
            text = g.render([{"role": "user", "content": "x"},
                             {"role": "assistant", "content": "", "tool_calls": [call("f", args, "i")]}], enable_thinking=False,
                            add_generation_prompt=False)
            start = text.index("<|tool_call>call:f") + len("<|tool_call>call:f")
            body = text[start:text.index(g.CALL_CLOSE)]
            self.assertEqual(g.parse_arguments(body), json.loads(json.dumps(args)), body)

    def test_a_reply_with_calls_becomes_openai_tool_calls(self):
        reply = ('Let me look. <|tool_call>call:get_weather{city:<|"|>Rome<|"|>,days:3,where:{lat:41.9,lon:null},'
                 'tags:[<|"|>a<|"|>,<|"|>b, c<|"|>]}<tool_call|><|tool_call>call:ping{}<tool_call|>')
        content, calls = g.parse_tool_calls(reply)
        self.assertEqual(content, "Let me look.")
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather", "ping"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Rome", "days": 3, "where": {"lat": 41.9, "lon": None}, "tags": ["a", "b, c"]})
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {})
        self.assertTrue(all(c["type"] == "function" and c["id"].startswith("call_") for c in calls))
        self.assertEqual(len({c["id"] for c in calls}), 2)

    def test_a_reply_with_no_calls_is_untouched(self):
        self.assertEqual(g.parse_tool_calls("just words"), ("just words", []))
        self.assertEqual(g.parse_tool_calls(""), ("", []))
        self.assertEqual(g.parse_tool_calls(None), ("", []))

    def test_a_call_that_does_not_parse_stays_in_the_text(self):
        reply = "<|tool_call>call:f{a:<|\"|>unterminated}<tool_call|>"
        content, calls = g.parse_tool_calls(reply)
        self.assertEqual(calls, [])
        self.assertEqual(content, reply)

    def test_quoted_keys_and_whitespace_are_read(self):
        self.assertEqual(g.parse_arguments('{ <|"|>a b<|"|> : 1 , c : <|"|>x<|"|> }'), {"a b": 1, "c": "x"})

    def test_the_thinking_channel_is_spoken_in_the_gateways_words(self):
        d = g.StreamDialect()
        frames = [b"<|channel>", b"thought", b"\n", b"Let me ", b"think", b"<channel|>", b"The answer"]
        self.assertEqual(b"".join(d.feed(f) for f in frames), b"Let me think</think>The answer")

    def test_the_opening_can_arrive_in_other_pieces(self):
        for pieces in ([b"<|channel>", b"thought\n", b"x"], [b"<|channel>", b"tho", b"ught\nx"], [b"<|channel>", b"thought\nx"]):
            d = g.StreamDialect()
            self.assertEqual(b"".join(d.feed(p) for p in pieces), b"x", pieces)

    def test_a_channel_that_is_not_a_thought_is_passed_on(self):
        d = g.StreamDialect()
        self.assertEqual(b"".join(d.feed(f) for f in (b"<|channel>", b"final", b" text")), b"final text")
        d = g.StreamDialect()
        d.feed(b"<|channel>"), d.feed(b"tho")
        self.assertEqual(d.flush(), b"tho")

    def test_ordinary_frames_pass_through(self):
        d = g.StreamDialect()
        self.assertEqual(d.feed("plain é".encode()), "plain é".encode())
        self.assertEqual(d.feed(b""), b"")


if __name__ == "__main__":
    unittest.main()
