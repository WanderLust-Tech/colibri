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

    def test_video_is_a_clear_400(self):
        base = self.server(["x"])
        status, error = _error_body(self, lambda: self.chat(base, messages=[{"role": "user", "content": [
            {"type": "text", "text": "what is this"}, {"type": "video", "video": {"url": "u"}}]}]))
        self.assertEqual(status, 400)
        self.assertEqual(error["code"], "unsupported_content_type")
        self.assertIn("video", error["message"])
        self.assertEqual(self.engine.calls, [])

    def test_the_model_is_listed_under_its_family_name(self):
        self.assertEqual(openai_server.family_by_id("gemma4").default_model_id, "gemma-4-26b-a4b-colibri")


try:
    import numpy  # noqa: F401
    from PIL import Image
    HAVE_PIL = True
except ImportError:                                   # pragma: no cover
    HAVE_PIL = False


def data_uri(width, height, colour=(200, 30, 30), kind="PNG"):
    import base64
    import io
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, kind)
    return f"data:image/{kind.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()


class PictureEngine(ScriptedEngine):
    """An engine that says it loaded a vision tower, and keeps the pictures it is handed."""
    model_dir = None

    def __init__(self, *args, vision=True, tokens="280", audio=False, audio_tokens="750", **kw):
        super().__init__(*args, **kw)
        self.vision = vision
        self.audio = audio
        self.caps = {"vision": "1" if vision else "0", "img_tokens": tokens,
                     "audio": "1" if audio else "0", "audio_tokens": audio_tokens}
        self.images = []

    def generate(self, *args, image=None, **kw):
        self.images.append(image)
        return super().generate(*args, image=image, **kw)


@unittest.skipUnless(HAVE_PIL, "needs Pillow and numpy")
class PictureTest(unittest.TestCase):
    """Pictures in a request: run through the image processor, placeholders in the prompt, IMAGE frames to the engine."""

    def setUp(self):
        patcher = patch("openai_server.ARCH", "gemma4")
        patcher.start()
        self.addCleanup(patcher.stop)

    def server(self, **kw):
        self.engine = PictureEngine(chunks=["It is red."], supports_logprobs_echo=False, supports_tok_ids=False, **kw)
        return _spawn_test_server(self, self.engine)

    def chat(self, base, content, **body):
        body.setdefault("model", "test-model")
        body.setdefault("messages", [{"role": "user", "content": content}])
        with _post_chat(base, body) as response:
            return json.load(response)

    def test_a_picture_is_sized_by_the_processors_rule_and_becomes_its_placeholders(self):
        base = self.server()
        reply = self.chat(base, [{"type": "text", "text": "What colour?"},
                                 {"type": "image_url", "image_url": {"url": data_uri(96, 48)}}])
        self.assertEqual(reply["choices"][0]["message"]["content"], "It is red.")
        # 96 x 48 pixels, scaled to fill at most 2520 patches of 16 pixels in multiples of 48: 1104 x 528
        grid_h, grid_w = 528 // 16, 1104 // 16
        tokens = (grid_h // 3) * (grid_w // 3)
        self.assertEqual(tokens, 253)
        prompt = self.engine.calls[0][0]
        self.assertIn("<|turn>user\nWhat colour?<|image>" + "<|image|>" * tokens + "<image|><turn|>", prompt)
        (patches, gh, gw), = [self.engine.images[0]]
        self.assertEqual((gh, gw), (grid_h, grid_w))
        self.assertEqual(len(patches.tobytes()), gh * gw * 16 * 16 * 3)
        self.assertTrue((patches.reshape(-1, 3) == (200, 30, 30)).all())      # a flat colour stays flat through the resize

    def test_two_pictures_go_in_order(self):
        base = self.server()
        self.chat(base, [{"type": "image_url", "image_url": {"url": data_uri(48, 48, (255, 0, 0))}},
                         {"type": "text", "text": "and"},
                         {"type": "image_url", "image_url": {"url": data_uri(96, 48, (0, 0, 255))}}])
        first, second = self.engine.images[0]
        self.assertEqual(len(self.engine.images[0]), 2)
        self.assertTrue((first[0].reshape(-1, 3) == (255, 0, 0)).all())
        self.assertTrue((second[0].reshape(-1, 3) == (0, 0, 255)).all())
        prompt = self.engine.calls[0][0]
        self.assertEqual(prompt.count("<|image>"), 2)
        self.assertEqual(prompt.count("<image|>"), 2)
        runs = [len(run) // len("<|image|>") for run in prompt.replace("<|image>", " ").replace("<image|>", " ").split(" ") if run.startswith("<|image|>")]
        self.assertEqual(runs, [(first[1] // 3) * (first[2] // 3), (second[1] // 3) * (second[2] // 3)])

    def test_the_engines_limit_chooses_the_budget(self):
        base = self.server(tokens="70")
        self.chat(base, [{"type": "image_url", "image_url": {"url": data_uri(400, 300)}}])
        patches, gh, gw = self.engine.images[0]
        self.assertLessEqual((gh // 3) * (gw // 3), 70)
        self.assertGreater((gh // 3) * (gw // 3), 40)

    def test_an_engine_without_the_tower_says_so(self):
        base = self.server(vision=False)
        status, error = _error_body(self, lambda: self.chat(base, [{"type": "image_url", "image_url": {"url": data_uri(48, 48)}}]))
        self.assertEqual(status, 400)
        self.assertIn("no vision tower", error["message"])
        self.assertIn("--add-vision", error["message"])
        self.assertEqual(self.engine.calls, [])

    def test_something_that_is_not_a_picture_is_a_400(self):
        base = self.server()
        status, error = _error_body(self, lambda: self.chat(base, [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]))
        self.assertEqual(status, 400)
        self.assertEqual(error["code"], "invalid_image")
        self.assertEqual(self.engine.calls, [])

    def test_a_sliver_is_one_column_wide_as_the_processor_makes_it(self):
        """A 2 x 2000 strip: the processor gives its width one 48-pixel column and its height all it can have."""
        base = self.server()
        self.chat(base, [{"type": "image_url", "image_url": {"url": data_uri(2, 2000)}}])
        patches, gh, gw = self.engine.images[0]
        self.assertEqual((gh, gw), (840, 3))
        self.assertEqual((gh // 3) * (gw // 3), 280)

    def test_pictures_that_the_template_would_place_in_another_order_are_refused(self):
        """The template writes a tool result's pictures at the assistant turn that called the tool, before that
        turn's own content: the engine takes pictures in the order the gateway sends them."""
        base = self.server()
        picture = {"type": "image_url", "image_url": {"url": data_uri(48, 48)}}
        status, error = _error_body(self, lambda: self.chat(base, None, messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "text", "text": "here"}, picture],
             "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "ok"}, picture]}]))
        self.assertEqual(status, 400)
        self.assertEqual(error["code"], "unsupported_content_order")

    def test_a_tool_result_picture_is_placed_after_its_response(self):
        base = self.server()
        self.chat(base, None, messages=[
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [
                {"type": "text", "text": "ok"}, {"type": "image_url", "image_url": {"url": data_uri(48, 48)}}]}])
        prompt = self.engine.calls[0][0]
        self.assertIn("<tool_response|><|image>", prompt)
        self.assertEqual(len(self.engine.images[0:1]), 1)

    def test_the_model_card_lists_image_input_when_the_tower_is_loaded(self):
        from urllib.request import Request, urlopen
        for vision, want in ((True, True), (False, False)):
            base = self.server(vision=vision)
            with urlopen(Request(base + "/v1/models", headers={"Authorization": "Bearer secret"}), timeout=5) as response:
                card = json.load(response)["data"][0]
            self.assertEqual("image" in card["input_modalities"], want, card)

    def audio_part(self, seconds=1.0, **kw):
        import base64
        import io
        import wave
        import numpy as np
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(kw.get("rate", 16000))
            n = int(seconds * kw.get("rate", 16000))
            w.writeframes((0.3 * np.sin(np.arange(n) * 0.2) * 32767).astype("<i2").tobytes())
        return {"type": "input_audio", "input_audio": {"data": base64.b64encode(buffer.getvalue()).decode(), "format": kw.get("format", "wav")}}

    def test_a_clip_is_run_through_the_front_end_and_becomes_its_placeholders(self):
        base = self.server(audio=True)
        reply = self.chat(base, [{"type": "text", "text": "Transcribe."}, self.audio_part(1.0)])
        self.assertEqual(reply["choices"][0]["message"]["content"], "It is red.")
        prompt = self.engine.calls[0][0]
        self.assertIn("<|turn>user\nTranscribe.<|audio>" + "<|audio|>" * 25 + "<audio|><turn|>", prompt)    # 99 mel frames -> 25 soft tokens
        mel, frames, bins, kind = self.engine.images[0]
        self.assertEqual((frames, bins, kind), (99, 128, "audio"))
        self.assertEqual(len(mel), 99 * 128 * 4)

    def test_pictures_and_clips_go_in_the_order_they_come(self):
        base = self.server(audio=True)
        self.chat(base, [{"type": "image_url", "image_url": {"url": data_uri(48, 48)}}, self.audio_part(0.5),
                         {"type": "text", "text": "both"}])
        pictures = self.engine.images[0]
        self.assertEqual(len(pictures), 2)
        self.assertEqual(len(pictures[0]), 3)                         # (patches, gh, gw)
        self.assertEqual(pictures[1][3], "audio")
        prompt = self.engine.calls[0][0]
        self.assertLess(prompt.index("<|image>"), prompt.index("<|audio>"))

    def test_an_engine_without_the_tower_says_so_for_audio(self):
        base = self.server(audio=False)
        status, error = _error_body(self, lambda: self.chat(base, [self.audio_part(0.5)]))
        self.assertEqual(status, 400)
        self.assertIn("no audio tower", error["message"])
        self.assertIn("--add-audio", error["message"])
        self.assertEqual(self.engine.calls, [])

    def test_only_wav_is_read_and_a_bad_file_is_a_400(self):
        base = self.server(audio=True)
        for part, what in ((self.audio_part(0.5, format="mp3"), "WAV only"),
                           ({"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}, "not a WAV"),
                           ({"type": "input_audio", "input_audio": {"data": "!!!", "format": "wav"}}, "base64"),
                           ({"type": "input_audio", "input_audio": "x"}, "must be an object")):
            with self.subTest(what=what):
                status, error = _error_body(self, lambda: self.chat(base, [part]))
                self.assertEqual(status, 400)
                self.assertEqual(error["code"], "invalid_audio")
                self.assertIn(what, error["message"])
        self.assertEqual(self.engine.calls, [])

    def test_the_engines_limit_cuts_a_long_clip(self):
        base = self.server(audio=True, audio_tokens="50")
        self.chat(base, [self.audio_part(5.0)])
        self.assertLessEqual(self.engine.images[0][1], 50 * 4 + 3)
        prompt = self.engine.calls[0][0]
        self.assertLessEqual(prompt.count("<|audio|>"), 50)

    def test_a_clip_at_another_rate_is_resampled(self):
        base = self.server(audio=True)
        self.chat(base, [self.audio_part(1.0, rate=44100)])
        self.assertAlmostEqual(self.engine.images[0][1], 99, delta=2)

    def test_the_model_card_lists_audio_when_the_tower_is_loaded(self):
        from urllib.request import Request, urlopen
        for audio, want in ((True, True), (False, False)):
            base = self.server(audio=audio)
            with urlopen(Request(base + "/v1/models", headers={"Authorization": "Bearer secret"}), timeout=5) as response:
                card = json.load(response)["data"][0]
            self.assertEqual("audio" in card["input_modalities"], want, card)

    def test_the_anthropic_endpoint_takes_a_picture_too(self):
        import base64
        import io
        base = self.server()
        buffer = io.BytesIO()
        Image.new("RGB", (48, 48), (1, 2, 3)).save(buffer, "PNG")
        body = {"model": "test-model", "max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what"}, {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": base64.b64encode(buffer.getvalue()).decode()}}]}]}
        with _post_json(base, "/v1/messages", body) as response:
            reply = json.load(response)
        self.assertEqual(reply["content"][0]["text"], "It is red.")
        self.assertIn("<|image>", self.engine.calls[0][0])
        patches, gh, gw = self.engine.images[0]
        self.assertEqual(len(patches.tobytes()), gh * gw * 16 * 16 * 3)
        self.assertTrue((patches.reshape(-1, 3) == (1, 2, 3)).all())


class DispatcherTest(unittest.TestCase):
    """The real Engine class on canned frames: the exact SUBMIT, and the channel's translation."""

    def test_several_pictures_are_sent_one_frame_each_before_the_submit(self):
        first, second = b"\x01" * 12, b"\x02" * 24

        def respond(process, frame):
            if frame.startswith(b"SUBMIT"):
                process.stdout.feed(b"ACCEPT 1 5\nDATA 1 2\nok\n" b"DONE 1 STAT 1 1.0 50.0 1.25 5 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "gemma4"), patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("gemma4", "model")
            engine.generate("<bos>x", 4, 0.0, 1.0, lambda text: None, image=[(first, 2, 3), (second, 4, 3)])
        engine.close()
        self.assertEqual(process.writes[0], b"IMAGE 1 12 2 3\n" + first + b"\n")
        self.assertEqual(process.writes[1], b"IMAGE 1 24 4 3\n" + second + b"\n")
        self.assertTrue(process.writes[2].startswith(b"SUBMIT 1 "))

    def test_a_clip_goes_as_an_audio_frame(self):
        mel = b"\x01" * 40

        def respond(process, frame):
            if frame.startswith(b"SUBMIT"):
                process.stdout.feed(b"ACCEPT 1 5\nDATA 1 2\nok\n" b"DONE 1 STAT 1 1.0 50.0 1.25 5 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "gemma4"), patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("gemma4", "model")
            engine.generate("<bos>x", 4, 0.0, 1.0, lambda text: None, image=[(b"\x02" * 12, 2, 3), (mel, 5, 2, "audio")])
        engine.close()
        self.assertEqual(process.writes[0], b"IMAGE 1 12 2 3\n" + b"\x02" * 12 + b"\n")
        self.assertEqual(process.writes[1], b"AUDIO 1 40 5 2\n" + mel + b"\n")
        self.assertTrue(process.writes[2].startswith(b"SUBMIT 1 "))

    def test_the_engines_refusal_of_a_clip_is_a_400_too(self):
        error = openai_server._engine_error(["BAD_AUDIO", "5", "mel", "bins"], "ERROR 1 BAD_AUDIO 5 mel bins")
        self.assertEqual((error.status, error.code), (400, "invalid_audio"))
        self.assertIn("5 mel bins", error.message)

    def test_the_engines_refusal_of_a_picture_is_a_400(self):
        error = openai_server._engine_error(["BAD_IMAGE", "bytes=3", "expected=12"], "ERROR 1 BAD_IMAGE bytes=3 expected=12")
        self.assertEqual(error.status, 400)
        self.assertEqual(error.code, "invalid_image")
        self.assertIn("bytes=3", error.message)

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
