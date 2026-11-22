"""The voice loop (voice_loop.py) and `coli talk` (talk_cli.py): the text rules, the voice-activity detector, and whole turns against a fake server that
speaks the three routes (transcriptions, a streamed chat, streamed speech)."""
import array
import base64
import io
import json
import math
import os
import sys
import tempfile
import threading
import unittest
import wave
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import talk_cli  # noqa: E402
import voice_loop  # noqa: E402
from voice_loop import SentenceSplitter, VoiceError, VoiceLoop, spoken_text  # noqa: E402


class TextRulesTest(unittest.TestCase):
    def test_markdown_code_links_and_emoji_are_not_spoken(self):
        text = "# Title\n\n- **bold** item\n- `code` and [a link](http://x.y/z)\n\n```python\nprint(1)\n```\nDone \U0001F600."
        self.assertEqual(spoken_text(text), "Title bold item code and a link Done .")

    def test_an_unclosed_code_fence_drops_the_rest(self):
        self.assertEqual(spoken_text("Here: ```x = 1\nmore"), "Here:")

    def sentences(self, chunks, **kw):
        splitter = SentenceSplitter(**kw)
        out = []
        for chunk in chunks:
            out += splitter.feed(chunk)
        return out, splitter.flush()

    def test_sentences_come_out_as_they_complete(self):
        done, rest = self.sentences(["Hello there, this is the first sentence. ", "And here is a second", " one. Third"])
        self.assertEqual(done, ["Hello there, this is the first sentence.", "And here is a second one."])
        self.assertEqual(rest, ["Third"])

    def test_a_short_sentence_is_joined_to_the_next(self):
        done, rest = self.sentences(["Yes. I think the answer is quite clearly the number forty two. "])
        self.assertEqual(done, ["Yes. I think the answer is quite clearly the number forty two."])

    def test_abbreviations_and_decimals_do_not_cut(self):
        done, rest = self.sentences(["Dr. Smith paid 3.5 dollars for it today. "])
        self.assertEqual(done, ["Dr. Smith paid 3.5 dollars for it today."])

    def test_the_first_sentence_may_be_shorter_so_the_voice_starts_early(self):
        done, _ = self.sentences(["Sure thing, boss. And now the long part of the answer follows here. "])
        self.assertEqual(done[0], "Sure thing, boss.")

    def test_a_line_break_ends_a_sentence_and_nothing_is_lost(self):
        done, rest = self.sentences(["First line of the answer is here\nsecond line of the answer is here too"])
        self.assertEqual((done, rest), (["First line of the answer is here"], ["second line of the answer is here too"]))

    def test_text_with_no_ending_is_flushed(self):
        self.assertEqual(self.sentences(["no punctuation at all"]), ([], ["no punctuation at all"]))


def tone(seconds, level, rate=16000, freq=300):
    n = int(seconds * rate)
    return array.array("h", (int(level * 32767 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n))).tobytes()


def frames_of(pcm):
    size = talk_cli.FRAME * 2
    return [pcm[i:i + size] for i in range(0, len(pcm) - size + 1, size)]


class VadTest(unittest.TestCase):
    def run_vad(self, pcm, **kw):
        vad = talk_cli.Vad(**kw)
        out = []
        for frame in frames_of(pcm):
            utterance = vad.feed(frame)
            if utterance is not None:
                out.append(utterance)
        return out

    def test_speech_between_silences_is_one_utterance_with_lead_in_and_a_short_tail(self):
        pcm = tone(0.6, 0.002) + tone(1.5, 0.3) + tone(1.2, 0.002)
        out = self.run_vad(pcm)
        self.assertEqual(len(out), 1)
        seconds = len(out[0]) / 2 / 16000
        self.assertGreater(seconds, 1.5)
        self.assertLess(seconds, 2.3)

    def test_two_utterances_come_out_separately(self):
        pcm = tone(0.6, 0.002) + tone(1.0, 0.3) + tone(1.2, 0.002) + tone(1.0, 0.25) + tone(1.2, 0.002)
        self.assertEqual(len(self.run_vad(pcm)), 2)

    def test_a_click_is_not_speech(self):
        pcm = tone(0.6, 0.002) + tone(0.04, 0.5) + tone(2.0, 0.002)
        self.assertEqual(self.run_vad(pcm), [])

    def test_a_short_pause_inside_a_sentence_does_not_end_it(self):
        pcm = tone(0.6, 0.002) + tone(0.8, 0.3) + tone(0.4, 0.002) + tone(0.8, 0.3) + tone(1.2, 0.002)
        out = self.run_vad(pcm)
        self.assertEqual(len(out), 1)
        self.assertGreater(len(out[0]) / 2 / 16000, 2.0)

    def test_a_noisy_room_raises_the_bar(self):
        noise = 0.02
        pcm = tone(0.6, noise) + tone(1.0, 0.4) + tone(1.2, noise)
        self.assertEqual(len(self.run_vad(pcm)), 1)
        self.assertEqual(self.run_vad(tone(0.6, noise) + tone(2.0, noise)), [])

    def test_speech_that_never_stops_is_cut_at_the_limit(self):
        out = self.run_vad(tone(0.6, 0.002) + tone(5.0, 0.3), max_seconds=2.0)
        self.assertEqual(len(out), 2)


class FakeServer:
    """The three routes, enough for a turn: what the chat model is asked is kept in `chats`, what the speech model is asked in `spoken`."""

    def __init__(self, heard="what is the weather", reply="It is sunny today. Bring sunglasses, though.", fail=None):
        outer = self
        self.heard, self.reply, self.fail = heard, reply, fail or {}
        self.chats, self.spoken, self.keys = [], [], []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def body(self):
                return self.rfile.read(int(self.headers.get("Content-Length", "0")))

            def reply_json(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def sse(self, events):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for event in events:
                    self.wfile.write(b"data: " + (event if isinstance(event, bytes) else json.dumps(event).encode()) + b"\n\n")
                    self.wfile.flush()
                self.close_connection = True

            def do_POST(self):
                outer.keys.append(self.headers.get("Authorization"))
                raw = self.body()
                if self.path == "/v1/audio/transcriptions":
                    if "transcribe" in outer.fail:
                        return self.reply_json(500, {"error": {"message": "the recogniser is down"}})
                    return self.reply_json(200, {"text": outer.heard})
                body = json.loads(raw)
                if self.path == "/v1/chat/completions":
                    outer.chats.append(body)
                    if "chat" in outer.fail:
                        return self.reply_json(404, {"error": {"message": "no such model"}})
                    words = outer.reply.split(" ")
                    events = [{"choices": [{"delta": {"content": w + (" " if i < len(words) - 1 else "")}}]} for i, w in enumerate(words)]
                    return self.sse(events + [b"[DONE]"])
                if self.path == "/v1/audio/speech":
                    outer.spoken.append(body)
                    if outer.fail.get("speak") == "early":
                        return self.reply_json(400, {"error": {"message": "no such voice"}})
                    if outer.fail.get("speak") == "late":
                        return self.sse([{"type": "speech.audio.delta", "audio": base64.b64encode(b"\x01\x00" * 4).decode()},
                                         {"type": "error", "error": {"message": "the card failed"}}])
                    pcm = b"\x01\x00" * (len(body["input"]) * 10)
                    half = len(pcm) // 2
                    return self.sse([{"type": "speech.audio.delta", "audio": base64.b64encode(pcm[:half]).decode()},
                                     {"type": "speech.audio.delta", "audio": base64.b64encode(pcm[half:]).decode()},
                                     {"type": "speech.audio.done", "audio_seconds": 1.0}])
                self.reply_json(404, {"error": {"message": "not found"}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    @property
    def server(self):
        return voice_loop.Server(f"http://127.0.0.1:{self.port}", "secret")


def wav(seconds=1.0):
    out = io.BytesIO()
    with wave.open(out, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(16000)
        clip.writeframes(tone(seconds, 0.2))
    return out.getvalue()


class TurnTest(unittest.TestCase):
    def fake(self, **kw):
        fake = FakeServer(**kw)
        self.addCleanup(fake.close)
        return fake

    def loop(self, fake, **kw):
        return VoiceLoop(fake.server, "chat-m", "asr-m", "tts-m", **kw)

    def test_a_turn_hears_answers_and_speaks_in_that_order(self):
        fake = self.fake()
        events = []
        result = self.loop(fake, voice="ryan").turn(audio=wav(), emit=lambda kind, value: events.append((kind, value)))
        kinds = [k for k, _ in events]
        self.assertEqual(kinds[0], "heard")
        self.assertLess(kinds.index("text"), kinds.index("audio"))
        self.assertEqual(kinds[-2:], ["reply", "timing"])
        self.assertEqual(result["heard"], "what is the weather")
        self.assertEqual(result["reply"], "It is sunny today. Bring sunglasses, though.")
        text = "".join(v for k, v in events if k == "text")
        self.assertEqual(text, result["reply"])
        self.assertEqual([c["input"] for c in fake.spoken], ["It is sunny today.", "Bring sunglasses, though."])      # a request a sentence
        self.assertTrue(all(c["voice"] == "ryan" and c["model"] == "tts-m" and c["stream_format"] == "sse" for c in fake.spoken))
        self.assertGreater(sum(len(v) for k, v in events if k == "audio"), 0)
        timing = result["timing"]
        self.assertLessEqual(timing["heard"], timing["first_text"])
        self.assertLessEqual(timing["first_text"], timing["done"])
        self.assertIn("first_audio", timing)
        self.assertEqual(set(fake.keys), {"Bearer secret"})

    def test_the_chat_model_gets_the_system_prompt_the_history_and_the_words(self):
        fake = self.fake()
        loop = self.loop(fake, system="Be brief.")
        loop.turn(audio=wav())
        loop.turn(text="and tomorrow?")
        first, second = fake.chats
        self.assertEqual([m["role"] for m in first["messages"]], ["system", "user"])
        self.assertEqual(first["messages"][0]["content"], "Be brief.")
        self.assertEqual([m["role"] for m in second["messages"]], ["system", "user", "assistant", "user"])
        self.assertEqual(second["messages"][-1]["content"], "and tomorrow?")
        self.assertTrue(first["stream"])

    def test_the_history_is_bounded(self):
        fake = self.fake()
        loop = self.loop(fake)
        for i in range(30):
            loop.turn(text=f"question {i}", speak=False)
        self.assertEqual(len(loop.history), voice_loop.MAX_HISTORY)
        self.assertEqual(loop.history[-2]["content"], "question 29")

    def test_text_turn_without_speech_makes_no_speech_request(self):
        fake = self.fake()
        result = self.loop(fake).turn(text="hello", speak=False)
        self.assertEqual((fake.spoken, result["reply"] != ""), ([], True))

    def test_silence_ends_the_turn_before_the_chat_model_is_asked(self):
        fake = self.fake(heard="  ")
        result = self.loop(fake).turn(audio=wav())
        self.assertEqual((result["heard"], result["reply"], fake.chats), ("", "", []))

    def test_a_recogniser_that_fails_is_named(self):
        fake = self.fake(fail={"transcribe": 1})
        with self.assertRaises(VoiceError) as caught:
            self.loop(fake).turn(audio=wav())
        self.assertEqual((caught.exception.step, caught.exception.message), ("transcribe", "the recogniser is down"))

    def test_a_chat_model_that_fails_is_named_and_nothing_is_spoken(self):
        fake = self.fake(fail={"chat": 1})
        loop = self.loop(fake)
        with self.assertRaises(VoiceError) as caught:
            loop.turn(audio=wav())
        self.assertEqual((caught.exception.step, fake.spoken, loop.history), ("chat", [], []))

    def test_a_voice_that_fails_early_still_leaves_the_text_and_the_history(self):
        fake = self.fake(fail={"speak": "early"})
        loop = self.loop(fake)
        events = []
        with self.assertRaises(VoiceError) as caught:
            loop.turn(audio=wav(), emit=lambda k, v: events.append(k))
        self.assertEqual(caught.exception.step, "speak")
        self.assertIn("reply", events)
        self.assertEqual(loop.history[-1]["content"], "It is sunny today. Bring sunglasses, though.")
        self.assertEqual(len(fake.spoken), 1)                                    # after the first failure the rest is not asked

    def test_a_voice_that_fails_mid_stream_is_an_error_after_the_audio_so_far(self):
        fake = self.fake(fail={"speak": "late"})
        got = []
        with self.assertRaises(VoiceError) as caught:
            self.loop(fake).turn(audio=wav(), emit=lambda k, v: got.append(k))
        self.assertEqual(caught.exception.message, "the card failed")
        self.assertIn("audio", got)

    def test_nothing_listening_is_a_clear_error(self):
        loop = VoiceLoop(voice_loop.Server("http://127.0.0.1:1"), "c", "a", "t")
        with self.assertRaises(VoiceError) as caught:
            loop.turn(audio=wav())
        self.assertIn("nothing answers", caught.exception.message)


class Plain:
    yel = dim = r = ""


def api_for(fake):
    """coli's api(method, path, body) for the fake: the model lists `coli talk` asks for."""
    models = {"data": [{"id": "big-chat", "capabilities": ["chat"]}, {"id": "gemma-e2b", "capabilities": ["chat"]},
                       {"id": "ears", "capabilities": ["transcription"]}, {"id": "mouth", "capabilities": ["speech"]}]}

    def api(method, path, body=None):
        if path == "/v1/models":
            return 200, models
        if path == "/v1/audio/voices":
            return 200, {"data": [{"name": "ana"}]}
        return 404, None
    return api


def talk_args(fake, *files, **kw):
    base = dict(files=list(files), text=None, model=None, asr=None, tts=None, voice=None, language=None, system=None, max_tokens=50, mic=None,
                list_mics=False, silence=0.8, threshold=0.012, out=None, no_play=True, host="127.0.0.1", port=fake.port, api_key="secret")
    base.update(kw)
    return SimpleNamespace(**base)


class TalkCommandTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeServer()
        self.addCleanup(self.fake.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_talk(self, a, api=None):
        out = io.StringIO()
        with redirect_stdout(out):
            code = talk_cli.talk(a, api or api_for(self.fake), Plain)
        return code, out.getvalue()

    def test_models_are_picked_a_small_chat_model_first(self):
        chat, asr, tts = talk_cli.pick_models(talk_args(self.fake), api_for(self.fake), Plain)
        self.assertEqual((chat, asr, tts), ("gemma-e2b", "ears", "mouth"))
        self.assertEqual(talk_cli.pick_models(talk_args(self.fake, model="big-chat", tts="x"), api_for(self.fake), Plain), ("big-chat", "ears", "x"))

    def test_a_file_turn_prints_the_exchange_and_saves_the_answer(self):
        clip = os.path.join(self.tmp.name, "q.wav")
        Path(clip).write_bytes(wav())
        out_dir = os.path.join(self.tmp.name, "out")
        code, text = self.run_talk(talk_args(self.fake, clip, out=out_dir))
        self.assertEqual(code, 0, text)
        self.assertIn("you: what is the weather", text)
        self.assertIn("colibri: It is sunny today. Bring sunglasses, though.", text)
        self.assertIn("first audio", text)
        with wave.open(os.path.join(out_dir, "reply-1.wav")) as clip_out:
            self.assertEqual((clip_out.getframerate(), clip_out.getnchannels(), clip_out.getsampwidth()), (24000, 1, 2))
            self.assertGreater(clip_out.getnframes(), 100)

    def test_typed_words_are_one_turn_and_a_conversation_keeps_its_history(self):
        code, text = self.run_talk(talk_args(self.fake, text="hello there", out=os.path.join(self.tmp.name, "o")))
        self.assertEqual(code, 0, text)
        self.assertEqual(self.fake.chats[0]["messages"][-1]["content"], "hello there")
        clip = os.path.join(self.tmp.name, "q.wav")
        Path(clip).write_bytes(wav())
        code, _ = self.run_talk(talk_args(self.fake, clip, text="and then?", out=os.path.join(self.tmp.name, "o2")))
        self.assertEqual(code, 0)
        last = self.fake.chats[-1]["messages"]
        self.assertEqual([m["content"] for m in last if m["role"] == "user"], ["what is the weather", "and then?"])      # the file's turn came first

    def test_a_saved_voice_goes_to_the_speech_model_as_voice(self):
        self.run_talk(talk_args(self.fake, text="hi", voice="ana", out=os.path.join(self.tmp.name, "o")))
        self.assertTrue(all(c["voice"] == "ana" for c in self.fake.spoken))

    def test_a_failing_step_is_said_and_the_exit_code_is_one(self):
        fake = FakeServer(fail={"chat": 1})
        self.addCleanup(fake.close)
        clip = os.path.join(self.tmp.name, "q.wav")
        Path(clip).write_bytes(wav())
        code, text = self.run_talk(talk_args(fake, clip, out=os.path.join(self.tmp.name, "o")), api=api_for(fake))
        self.assertEqual(code, 1)
        self.assertIn("chat failed: no such model", text)

    def test_no_server_models_is_a_clear_refusal(self):
        code, text = self.run_talk(talk_args(self.fake, text="x"), api=lambda method, path, body=None: (200, {"data": []}))
        self.assertEqual(code, 1)
        self.assertIn("no chat model", text)

    def test_a_missing_file_is_a_refusal(self):
        code, text = self.run_talk(talk_args(self.fake, os.path.join(self.tmp.name, "nope.wav"), out=os.path.join(self.tmp.name, "o")))
        self.assertEqual(code, 1)
        self.assertIn("cannot read", text)


if __name__ == "__main__":
    unittest.main()
