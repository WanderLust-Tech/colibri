"""Forced alignment (docs/qwen3-aligner.md): the word rules, the subtitles, POST /v1/audio/alignments end to end against the stub engine, and `coli align` /
`coli transcribe --srt` against two gateways (a recogniser and an aligner)."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))

import align_engine  # noqa: E402
import speech_cli  # noqa: E402
import subtitles  # noqa: E402
from align_engine import AlignmentEngine, alignment_request, split_words, split_words_marked  # noqa: E402
from embedding_engine import EmbeddingRequestError  # noqa: E402
from openai_server import APIServer  # noqa: E402
from test_asr_gateway import KEY, MODEL_ID, STUB, GatewayHarness as AsrHarness, multipart, stub_env, wav_bytes  # noqa: E402
import threading  # noqa: E402


class WordsTest(unittest.TestCase):
    def test_a_unit_is_a_run_of_letters_digits_and_apostrophes(self):
        self.assertEqual(split_words("Hello, world! It's 3.5 now..."), ["Hello", "world", "It's", "35", "now"])

    def test_each_cjk_character_is_a_unit_of_its_own(self):
        self.assertEqual(split_words("你好 world 世界"), ["你", "好", "world", "世", "界"])
        self.assertEqual(split_words("abc世def"), ["abc", "世", "def"])

    def test_punctuation_only_is_nothing(self):
        self.assertEqual(split_words(" ... -- !? "), [])

    def test_marks_say_where_a_sentence_or_a_clause_ends(self):
        words, marks = split_words_marked("And so, my fellow Americans; ask not what you can do. Done!")
        self.assertEqual(words, ["And", "so", "my", "fellow", "Americans", "ask", "not", "what", "you", "can", "do", "Done"])
        self.assertEqual(marks, [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2])

    def test_a_closing_quote_after_the_full_stop_still_ends_the_sentence(self):
        _words, marks = split_words_marked('He said "go." Then left')
        self.assertEqual(marks, [0, 0, 2, 0, 0])

    def test_japanese_without_nagisa_is_a_unit_a_character(self):
        words, marks = split_words_marked("こんにちは。", "Japanese")
        if "nagisa" not in sys.modules:
            try:
                import nagisa  # noqa: F401
            except ImportError:
                self.assertEqual(words, list("こんにちは"))
        self.assertEqual(len(words), len(marks))

    def test_a_language_is_guessed_from_kana_and_hangul(self):
        self.assertEqual(align_engine.guess_language("こんにちは"), "Japanese")
        self.assertEqual(align_engine.guess_language("안녕하세요"), "Korean")
        self.assertIsNone(align_engine.guess_language("hello 你好"))


class RequestRulesTest(unittest.TestCase):
    def refused(self, body, param):
        with self.assertRaises(EmbeddingRequestError) as caught:
            alignment_request(body)
        self.assertEqual(caught.exception.param, param, str(caught.exception))

    def test_the_defaults(self):
        request = alignment_request({"text": "Hello there, friend."})
        self.assertEqual((request["words"], request["language"], request["format"], request["max_chars"], request["max_seconds"]),
                         (["Hello", "there", "friend"], None, "json", 42, 7.0))
        self.assertEqual(request["marks"], [0, 1, 2])

    def test_the_text_is_required_and_has_to_have_words_in_it(self):
        self.refused({}, "text")
        self.refused({"text": "   "}, "text")
        self.refused({"text": "?! ..."}, "text")
        self.refused({"text": "x" * (align_engine.MAX_TEXT_CHARS + 1)}, "text")

    def test_a_language_is_a_name_or_a_code_of_the_aligners_languages(self):
        self.assertEqual(alignment_request({"text": "hi", "language": "en"})["language"], "English")
        self.assertEqual(alignment_request({"text": "hi", "language": "zh"})["language"], "Chinese")
        self.assertEqual(alignment_request({"text": "hi", "language": "German"})["language"], "German")
        self.assertEqual(alignment_request({"text": "hi", "language": "ja-JP"})["language"], "Japanese")
        self.refused({"text": "hi", "language": "xx"}, "language")
        self.refused({"text": "hi", "language": "Thai"}, "language")                 # the recogniser has Thai, the aligner not
        self.refused({"text": "hi", "language": 5}, "language")

    def test_formats_and_limits(self):
        for fmt in ("json", "verbose_json", "srt", "vtt", "text"):
            self.assertEqual(alignment_request({"text": "hi", "response_format": fmt})["format"], fmt)
        self.refused({"text": "hi", "response_format": "ass"}, "response_format")
        self.assertEqual(alignment_request({"text": "hi", "max_chars": "30", "max_seconds": "5"})["max_chars"], 30)
        self.refused({"text": "hi", "max_chars": 3}, "max_chars")
        self.refused({"text": "hi", "max_seconds": 99}, "max_seconds")
        self.refused({"text": "hi", "max_seconds": "soon"}, "max_seconds")
        self.refused({"text": "hi", "max_chars": True}, "max_chars")


class SubtitlesTest(unittest.TestCase):
    def words(self, spec):
        return [{"word": w, "start": s, "end": e} for w, s, e in spec]

    def test_a_line_is_closed_by_length_by_a_sentence_end_and_by_a_pause(self):
        items = self.words([("one", 0.0, 0.4), ("two", 0.5, 0.9), ("three", 1.0, 1.4), ("four", 1.5, 1.9), ("five", 4.0, 4.4), ("six", 4.5, 4.9)])
        self.assertEqual([c["text"] for c in subtitles.build_cues(items, [0, 0, 0, 0, 0, 0], max_chars=14)], ["one two three", "four", "five six"])
        self.assertEqual([c["text"] for c in subtitles.build_cues(items, [0, 2, 0, 0, 0, 0], max_chars=80)], ["one two", "three four", "five six"])   # a stop, a pause

    def test_a_comma_closes_a_line_once_it_is_half_full(self):
        items = self.words([("aaaa", 0, 0.3), ("bbbb", 0.4, 0.7), ("cccc", 0.8, 1.1), ("dddd", 1.2, 1.5)])
        self.assertEqual([c["text"] for c in subtitles.build_cues(items, [1, 1, 0, 0], max_chars=12)], ["aaaa bbbb", "cccc dddd"])

    def test_a_line_longer_than_the_time_limit_is_cut(self):
        items = self.words([(f"w{i}", i * 0.9, i * 0.9 + 0.8) for i in range(10)])
        cues = subtitles.build_cues(items, None, max_chars=200, max_seconds=3.0)
        self.assertTrue(all(c["end"] - c["start"] <= 3.0 for c in cues))
        self.assertEqual(sum(len(c["text"].split()) for c in cues), 10)

    def test_cjk_characters_are_joined_without_spaces(self):
        self.assertEqual(subtitles.join_words(["你", "好", "world", "ok", "世"]), "你好world ok世")

    def test_srt_and_vtt(self):
        cues = [{"start": 0.32, "end": 2.16, "text": "And so, my fellow Americans"}, {"start": 3600.5, "end": 3601.0, "text": "ask not"}]
        self.assertEqual(subtitles.srt(cues), "1\n00:00:00,320 --> 00:00:02,160\nAnd so, my fellow Americans\n\n2\n01:00:00,500 --> 01:00:01,000\nask not\n\n")
        self.assertEqual(subtitles.vtt(cues), "WEBVTT\n\n00:00:00.320 --> 00:00:02.160\nAnd so, my fellow Americans\n\n01:00:00.500 --> 01:00:01.000\nask not\n\n")


class AdoptTest(unittest.TestCase):
    """The launcher starts every engine of this family as a recogniser; the READY line decides what it is."""

    def test_an_engine_that_says_it_aligns_becomes_an_aligner_and_works(self):
        from asr_engine import TranscriptionEngine
        engine = TranscriptionEngine(STUB, "model-dir", env=stub_env(ASR_STUB_ALIGN=1), on_log=lambda _m: None)
        self.addCleanup(engine.close)
        self.assertFalse(hasattr(engine, "align_samples"))
        self.assertIs(align_engine.adopt(engine), engine)
        self.assertIsInstance(engine, AlignmentEngine)
        self.assertEqual(engine.align_samples(wav_bytes(1.0), ["a", "b"])["starts"], [0.0, 0.4])
        self.assertEqual(align_engine.adopt(engine).__class__, AlignmentEngine)                 # a second time changes nothing

    def test_a_recogniser_stays_one(self):
        from asr_engine import TranscriptionEngine
        engine = TranscriptionEngine(STUB, "model-dir", env=stub_env(), on_log=lambda _m: None)
        self.addCleanup(engine.close)
        self.assertIs(align_engine.adopt(engine).__class__, TranscriptionEngine)


class AlignHarness:
    """A gateway over the stub started as a forced aligner."""

    def __init__(self, model_id="qwen3-forcedaligner-0.6b", **env):
        self.engine = AlignmentEngine(STUB, "model-dir", env=stub_env(ASR_STUB_ALIGN=1, **env), on_log=lambda _m: None)
        self.server = APIServer(("127.0.0.1", 0), self.engine, model_id, KEY, 16, max_queue=8, queue_timeout=30)
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def close(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.engine.close()

    def call(self, path, data=None, content_type="application/json", key=KEY, method=None):
        headers = {"Content-Type": content_type}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = Request(f"http://127.0.0.1:{self.port}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=30) as response:
                return response.status, dict(response.headers), response.read()
        except HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def align(self, text="Hello there, my friend.", seconds=2.0, **fields):
        data, content_type = multipart(dict({"model": "qwen3-forcedaligner-0.6b", "text": text}, **fields), data=wav_bytes(seconds))
        return self.call("/v1/audio/alignments", data, content_type)


class EndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = AlignHarness()

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def test_the_words_come_back_with_their_times(self):
        status, headers, raw = self.gw.align()
        self.assertEqual(status, 200, raw[:200])
        self.assertEqual(json.loads(raw), {"words": [{"word": "Hello", "start": 0.0, "end": 0.3}, {"word": "there", "start": 0.4, "end": 0.7},
                                                     {"word": "my", "start": 0.8, "end": 1.1}, {"word": "friend", "start": 1.2, "end": 1.5}]})
        self.assertIn("x-colibri-queue-wait-ms", {k.lower() for k in headers})

    def test_verbose_json_says_what_it_was_about(self):
        status, _headers, raw = self.gw.align(response_format="verbose_json", language="en")
        body = json.loads(raw)
        self.assertEqual((status, body["task"], body["language"], body["text"], len(body["words"])), (200, "align", "english", "Hello there, my friend.", 4))
        self.assertAlmostEqual(body["duration"], 2.0, places=2)

    def test_srt_vtt_and_text(self):
        status, headers, raw = self.gw.align(response_format="srt")
        self.assertEqual((status, headers["Content-Type"].split(";")[0]), (200, "application/x-subrip"))
        self.assertEqual(raw.decode(), "1\n00:00:00,000 --> 00:00:01,500\nHello there, my friend.\n\n".replace("Hello there, my friend.", "Hello there my friend"))
        status, headers, raw = self.gw.align(response_format="vtt", text="One two. Three four.")
        self.assertEqual((status, headers["Content-Type"].split(";")[0]), (200, "text/vtt"))
        self.assertTrue(raw.decode().startswith("WEBVTT\n\n00:00:00.000 --> 00:00:00.700\nOne two\n\n00:00:00.800 --> 00:00:01.500\nThree four\n\n"), raw.decode())
        status, headers, raw = self.gw.align(response_format="text")
        self.assertEqual((status, raw.decode()), (200, "0.00 0.30 Hello\n0.40 0.70 there\n0.80 1.10 my\n1.20 1.50 friend\n"))

    def test_a_json_body_with_base64_audio_works_too(self):
        import base64
        body = json.dumps({"model": "qwen3-forcedaligner-0.6b", "text": "one two", "file": base64.b64encode(wav_bytes(1.0)).decode()}).encode()
        status, _headers, raw = self.gw.call("/v1/audio/alignments", body)
        self.assertEqual((status, len(json.loads(raw)["words"])), (200, 2))

    def test_refusals_carry_the_field(self):
        for fields, param in (({"text": ""}, "text"), ({"text": "?!"}, "text"), ({"language": "xx"}, "language"), ({"response_format": "ass"}, "response_format")):
            with self.subTest(fields=fields):
                status, _headers, raw = self.gw.align(**fields)
                self.assertEqual((status, json.loads(raw)["error"]["param"]), (400, param))
        data, content_type = multipart({"model": "qwen3-forcedaligner-0.6b", "text": "hello"})
        status, _headers, raw = self.gw.call("/v1/audio/alignments", data, content_type)
        self.assertEqual((status, json.loads(raw)["error"]["param"]), (400, "file"))
        data, content_type = multipart({"model": "qwen3-forcedaligner-0.6b", "text": "hello"}, data=b"this is not audio at all, not even a little")
        status, _headers, raw = self.gw.call("/v1/audio/alignments", data, content_type)
        self.assertEqual((status, json.loads(raw)["error"]["param"]), (400, "file"))

    def test_audio_longer_than_the_limit_is_refused(self):
        status, _headers, raw = self.gw.align(seconds=align_engine.MAX_SECONDS + 5)
        self.assertEqual((status, json.loads(raw)["error"]["param"]), (400, "file"))
        self.assertIn("up to 300 s", json.loads(raw)["error"]["message"])

    def test_the_aligner_does_not_transcribe_and_says_where_to_go(self):
        data, content_type = multipart({"model": "qwen3-forcedaligner-0.6b"}, data=wav_bytes(1.0))
        status, _headers, raw = self.gw.call("/v1/audio/transcriptions", data, content_type)
        self.assertEqual(status, 400)
        self.assertIn("/v1/audio/alignments", json.loads(raw)["error"]["message"])

    def test_the_card_lists_alignment_and_not_transcription(self):
        status, _headers, raw = self.gw.call("/v1/models", method="GET")
        card = next(item for item in json.loads(raw)["data"] if item["id"] == "qwen3-forcedaligner-0.6b")
        self.assertEqual(card["capabilities"], ["alignment"])
        self.assertEqual((card["alignment"]["max_seconds"], "English" in card["alignment"]["languages"], "srt" in card["alignment"]["formats"]), (300, True, True))
        self.assertNotIn("transcription", card)

    def test_an_engine_error_is_a_500_and_it_goes_on(self):
        gw = AlignHarness(ASR_STUB_FAIL_ALIGN=1)
        self.addCleanup(gw.close)
        status, _headers, raw = gw.align()
        self.assertEqual((status, json.loads(raw)["error"]["type"]), (500, "server_error"))
        self.assertTrue(gw.engine.alive)

    def test_a_recogniser_refuses_alignments_and_says_what_it_does(self):
        gw = AsrHarness()
        self.addCleanup(gw.close)
        data, content_type = multipart({"model": MODEL_ID, "text": "hello"}, data=wav_bytes(1.0))
        status, _headers, raw = gw.call("/v1/audio/alignments", data, content_type)
        self.assertEqual(status, 400)
        self.assertIn("transcribes speech", json.loads(raw)["error"]["message"])
        card = next(item for item in json.loads(gw.call("/v1/models", method="GET")[2])["data"] if item["id"] == MODEL_ID)
        self.assertEqual(card["capabilities"], ["transcription"])


class RealUploadHelperTest(unittest.TestCase):
    """`coli`'s own upload helper (not the tests' stand-in): subtitles are not JSON and must come back as text."""

    @classmethod
    def setUpClass(cls):
        import importlib.machinery
        import importlib.util
        loader = importlib.machinery.SourceFileLoader("coli_script", str(HERE.parent / "coli"))
        spec = importlib.util.spec_from_loader("coli_script", loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)
        cls.gw = AlignHarness()

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def upload(self, fmt):
        a = SimpleNamespace(host="127.0.0.1", port=self.gw.port, api_key=KEY)
        return self.coli._api_upload(a, "/v1/audio/alignments", {"model": "qwen3-forcedaligner-0.6b", "text": "One two. Three four.", "response_format": fmt},
                                     "clip.wav", wav_bytes(2.0))

    def test_srt_vtt_and_text_come_back_as_text_and_json_as_json(self):
        status, body = self.upload("srt")
        self.assertEqual((status, body.startswith("1\n00:00:00,000 --> ")), (200, True))
        status, body = self.upload("vtt")
        self.assertEqual((status, body.startswith("WEBVTT")), (200, True))
        status, body = self.upload("text")
        self.assertEqual((status, body.splitlines()[0]), (200, "0.00 0.30 One"))
        status, body = self.upload("json")
        self.assertEqual((status, [w["word"] for w in body["words"]]), (200, ["One", "two", "Three", "four"]))

    def test_an_error_is_still_json(self):
        a = SimpleNamespace(host="127.0.0.1", port=self.gw.port, api_key=KEY)
        status, body = self.coli._api_upload(a, "/v1/audio/alignments", {"model": "qwen3-forcedaligner-0.6b", "text": "?!"}, "clip.wav", wav_bytes(1.0))
        self.assertEqual((status, body["error"]["param"]), (400, "text"))


class Plain:
    yel = dim = r = ""


class CommandTest(unittest.TestCase):
    """`coli align` and `coli transcribe --srt` against a recogniser and an aligner: the client's api and upload route by path."""

    def setUp(self):
        self.asr, self.aligner = AsrHarness(), AlignHarness()
        self.addCleanup(self.asr.close)
        self.addCleanup(self.aligner.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clip = os.path.join(self.tmp.name, "talk.wav")
        Path(self.clip).write_bytes(wav_bytes(2.0))

    def api(self, method, path, body=None):
        if path == "/v1/models":
            return 200, {"data": [{"id": MODEL_ID, "capabilities": ["transcription"]}, {"id": "qwen3-forcedaligner-0.6b", "capabilities": ["alignment"]}]}
        return 404, None

    def upload(self, path, fields, filename, data):
        gw = self.aligner if path == "/v1/audio/alignments" else self.asr
        body, content_type = multipart(fields, filename, data)
        status, _headers, raw = gw.call(path, body, content_type)
        try:
            return status, json.loads(raw)
        except ValueError:
            return status, raw.decode()

    def run_command(self, function, **kw):
        base = dict(files=[self.clip], text=None, text_file=None, language=None, aligner=None, srt=False, vtt=False, words=False, json=False, out=None,
                    model=None, context=None, verbose=False)
        base.update(kw)
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = function(SimpleNamespace(**base), self.api, self.upload, Plain)
        return code, out.getvalue()

    def test_align_prints_each_word_with_its_times(self):
        code, text = self.run_command(speech_cli.align, text="Hello there friend")
        self.assertEqual((code, text), (0, "0.00 0.30 Hello\n0.40 0.70 there\n0.80 1.10 friend\n"))

    def test_align_writes_subtitles_beside_the_audio(self):
        code, text = self.run_command(speech_cli.align, text="One two. Three four.", srt=True)
        self.assertEqual(code, 0, text)
        written = Path(os.path.splitext(self.clip)[0] + ".srt").read_text(encoding="utf-8")
        self.assertEqual(written, "1\n00:00:00,000 --> 00:00:00,700\nOne two\n\n2\n00:00:00,800 --> 00:00:01,500\nThree four\n\n")
        out = os.path.join(self.tmp.name, "x.vtt")
        code, _text = self.run_command(speech_cli.align, text="One two.", vtt=True, out=out)
        self.assertTrue(Path(out).read_text(encoding="utf-8").startswith("WEBVTT"))

    def test_align_reads_the_text_from_a_file_and_json_is_the_servers_answer(self):
        words = os.path.join(self.tmp.name, "words.txt")
        Path(words).write_text("alpha beta", encoding="utf-8")
        code, text = self.run_command(speech_cli.align, text_file=words, json=True)
        body = json.loads(text)
        self.assertEqual((code, [w["word"] for w in body["words"]], body["task"]), (0, ["alpha", "beta"], "align"))

    def test_align_usage_and_server_refusals_are_said(self):
        self.assertEqual(self.run_command(speech_cli.align)[0], 1)                                   # no text
        self.assertEqual(self.run_command(speech_cli.align, text="x", files=[])[0], 1)
        code, text = self.run_command(speech_cli.align, text="?!")
        self.assertEqual(code, 1)
        self.assertIn("no words", text)
        code, text = self.run_command(speech_cli.align, text="hello", language="xx")
        self.assertEqual(code, 1)
        self.assertIn("language", text)

    def test_transcribe_with_srt_hears_then_times_what_it_heard(self):
        code, text = self.run_command(speech_cli.transcribe, srt=True)
        self.assertEqual(code, 0, text)
        srt = Path(os.path.splitext(self.clip)[0] + ".srt").read_text(encoding="utf-8")
        self.assertIn("English 20 s", srt.replace("English | 2.0 s", "English 20 s"))               # the stub's transcript, cut into words and timed
        self.assertIn("-->", srt)

    def test_transcribe_with_words_prints_the_timed_words(self):
        code, text = self.run_command(speech_cli.transcribe, words=True)
        self.assertEqual(code, 0, text)
        lines = text.strip().splitlines()
        self.assertEqual(lines[0], "0.00 0.30 English")
        self.assertTrue(all(len(line.split()) == 3 for line in lines))

    def test_transcribe_without_an_aligner_says_so(self):
        def only_asr(method, path, body=None):
            return 200, {"data": [{"id": MODEL_ID, "capabilities": ["transcription"]}]}
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = speech_cli.transcribe(SimpleNamespace(files=[self.clip], language=None, context=None, model=None, verbose=False, json=False, srt=True, vtt=False,
                                                         words=False, aligner=None, out=None), only_asr, self.upload, Plain)
        self.assertEqual(code, 1)
        self.assertIn("qwen3-forcedaligner-0.6b", out.getvalue())


if __name__ == "__main__":
    unittest.main()
