"""`coli transcribe` (speech_cli.py): what a person sees, end to end against a real gateway over the stub speech engine."""
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import speech_cli  # noqa: E402
from test_asr_gateway import KEY, MODEL_ID, GatewayHarness, multipart, wav_bytes  # noqa: E402
import test_tts_gateway  # noqa: E402


class Plain:
    """coli's colour table with the colours turned off."""
    yel = dim = r = ""


def clients(port, key=KEY):
    def api(method, path, body=None):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(), headers=headers, method=method)
        try:
            with urlopen(req, timeout=30) as response:
                return response.status, json.loads(response.read() or b"null")
        except HTTPError as error:
            return error.code, json.loads(error.read() or b"null")
        except (URLError, OSError):
            return None, None

    def upload(path, fields, filename, data):
        body, content_type = multipart(fields, filename, data)
        headers = {"Content-Type": content_type}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = Request(f"http://127.0.0.1:{port}{path}", data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=60) as response:
                return response.status, json.loads(response.read() or b"null")
        except HTTPError as error:
            return error.code, json.loads(error.read() or b"null")
        except (URLError, OSError):
            return None, None
    return api, upload


def args(*files, **kw):
    base = dict(files=list(files), language=None, context=None, model=None, verbose=False, json=False)
    base.update(kw)
    return SimpleNamespace(**base)


class TranscribeCommandTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness()
        api, upload = clients(cls.gw.port)
        cls.api, cls.upload = staticmethod(api), staticmethod(upload)       # plain functions, not methods
        cls.tmp = tempfile.TemporaryDirectory()
        cls.clip = os.path.join(cls.tmp.name, "two.wav")
        Path(cls.clip).write_bytes(wav_bytes(2.0))
        cls.quiet = os.path.join(cls.tmp.name, "quiet.wav")
        Path(cls.quiet).write_bytes(wav_bytes(2.0, tone=False))

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()
        cls.tmp.cleanup()

    def run_command(self, a, api=None, upload=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = speech_cli.transcribe(a, api or self.api, upload or self.upload, Plain)
        return code, out.getvalue(), err.getvalue()

    def test_the_text_is_printed_alone(self):
        code, out, _ = self.run_command(args(self.clip))
        self.assertEqual((code, out), (0, "English | 2.0 s\n"))

    def test_language_and_context_go_to_the_server(self):
        code, out, _ = self.run_command(args(self.clip, language="fr", context="Paris"))
        self.assertEqual((code, out), (0, "French | 2.0 s | Paris\n"))

    def test_verbose_says_the_length_and_the_language_heard(self):
        code, out, _ = self.run_command(args(self.clip, verbose=True))
        self.assertEqual(code, 0)
        self.assertIn("two.wav  (2.0 s, english)", out)
        self.assertIn("English | 2.0 s", out)

    def test_several_files_are_each_named(self):
        code, out, _ = self.run_command(args(self.clip, self.clip))
        self.assertEqual(code, 0)
        self.assertEqual(out.count("two.wav  ("), 2)

    def test_json_is_the_servers_answer(self):
        code, out, _ = self.run_command(args(self.clip, json=True))
        self.assertEqual(code, 0)
        answer = json.loads(out)
        self.assertEqual((answer["task"], answer["text"]), ("transcribe", "English | 2.0 s"))

    def test_a_clip_with_nothing_in_it_says_so(self):
        code, out, _ = self.run_command(args(self.quiet))
        self.assertEqual(code, 0)
        self.assertIn("nothing was heard in quiet.wav", out)

    def test_a_file_that_is_not_there_is_one_line_and_the_others_still_run(self):
        code, out, _ = self.run_command(args(os.path.join(self.tmp.name, "missing.wav"), self.clip))
        self.assertEqual(code, 1)
        self.assertIn("cannot read", out)
        self.assertIn("English | 2.0 s", out)

    def test_an_empty_file_is_refused_before_it_is_sent(self):
        empty = os.path.join(self.tmp.name, "empty.wav")
        Path(empty).write_bytes(b"")
        code, out, _ = self.run_command(args(empty))
        self.assertEqual(code, 1)
        self.assertIn("is empty", out)

    def test_the_server_refusing_is_its_own_words(self):
        code, out, _ = self.run_command(args(self.clip, language="xx"))
        self.assertEqual(code, 1)
        self.assertIn("the server said 400", out)
        self.assertIn("not one this model transcribes", out)

    def test_no_files_is_a_usage_line(self):
        code, out, _ = self.run_command(args())
        self.assertEqual(code, 1)
        self.assertIn("usage: coli transcribe", out)

    def test_a_wrong_key_and_a_server_that_is_not_there(self):
        api, upload = clients(self.gw.port, key="wrong")
        code, out, _ = self.run_command(args(self.clip), api, upload)
        self.assertEqual(code, 1)
        self.assertIn("wants an API key", out)
        api, upload = clients(1)
        code, out, _ = self.run_command(args(self.clip), api, upload)
        self.assertEqual(code, 1)
        self.assertIn("nothing answers", out)

    def test_a_server_without_a_speech_model_says_what_to_install(self):
        def api(method, path, body=None):
            return 200, {"data": [{"id": "chat", "capabilities": ["chat"]}]}
        code, out, _ = self.run_command(args(self.clip), api, self.upload)
        self.assertEqual(code, 1)
        self.assertIn("coli model install qwen3-asr-0.6b", out)

    def test_a_file_over_the_limit_is_refused_before_it_is_sent(self):
        big = os.path.join(self.tmp.name, "big.wav")
        with open(big, "wb") as handle:
            handle.truncate(speech_cli.MAX_BYTES + 1)
        code, out, _ = self.run_command(args(big))
        self.assertEqual(code, 1)
        self.assertIn("takes up to 25 MB", out)

    def test_the_model_can_be_named(self):
        code, out, _ = self.run_command(args(self.clip, model=MODEL_ID))
        self.assertEqual((code, out), (0, "English | 2.0 s\n"))


if __name__ == "__main__":
    unittest.main()


def fetcher(port, key=KEY):
    """coli's _api_fetch, for a gateway on `port`: (status, raw body, headers) of a JSON POST."""
    def fetch(path, body):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urlopen(req, timeout=60) as response:
                return response.status, response.read(), {k.lower(): v for k, v in response.headers.items()}
        except HTTPError as error:
            return error.code, error.read(), {k.lower(): v for k, v in error.headers.items()}
        except (URLError, OSError):
            return None, None, None
    return fetch


def speak_args(*words, **kw):
    base = dict(text=list(words), voice=None, language=None, out=None, format=None, seed=None, model=None)
    base.update(kw)
    return SimpleNamespace(**base)


class SpeakCommandTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = test_tts_gateway.GatewayHarness(model_id=test_tts_gateway.MODEL_ID)
        api, _upload = clients(cls.gw.port)
        cls.api, cls.fetch = staticmethod(api), staticmethod(fetcher(cls.gw.port))
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()
        cls.tmp.cleanup()

    def run_command(self, a, api=None, fetch=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = speech_cli.speak(a, api or self.api, fetch or self.fetch, Plain)
        return code, out.getvalue(), err.getvalue()

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_the_audio_is_written_and_the_length_said(self):
        out = self.path("a.wav")
        code, text, _ = self.run_command(speak_args("hello", "there", out=out))
        self.assertEqual(code, 0, text)
        self.assertIn("a.wav", text)
        self.assertIn("0.6 s of speech", text)
        self.assertEqual(test_tts_gateway.echo_of(Path(out).read_bytes()), "ryan|auto|0|0|0")

    def test_the_voice_the_language_and_the_seed_go_to_the_server(self):
        out = self.path("b.wav")
        code, text, _ = self.run_command(speak_args("hello there", voice="vivian", language="fr", seed=5, out=out))
        self.assertEqual(code, 0, text)
        self.assertEqual(test_tts_gateway.echo_of(Path(out).read_bytes()), "vivian|french|5|0|0")

    def test_an_instruction_goes_to_the_server_as_instructions(self):
        gw = test_tts_gateway.GatewayHarness(model_id=test_tts_gateway.MODEL_ID, TTS_STUB_INSTRUCT=1)
        self.addCleanup(gw.close)
        api, _upload = clients(gw.port)
        out = self.path("styled.wav")
        code, text, _ = self.run_command(speak_args("hello there", voice="vivian", instruct="Speak slowly.", out=out), api=api, fetch=fetcher(gw.port))
        self.assertEqual(code, 0, text)
        self.assertEqual(test_tts_gateway.echo_of(Path(out).read_bytes()), "vivian|auto|0|0|0|Speak slowly.")

    def test_a_model_without_styles_says_so_when_one_is_asked_for(self):
        code, text, _ = self.run_command(speak_args("hello there", instruct="Speak slowly.", out=self.path("no.wav")))
        self.assertEqual(code, 1)
        self.assertIn("instructions", text)

    def test_the_suffix_picks_the_format_and_a_flag_overrides_it(self):
        out = self.path("c.pcm")
        self.assertEqual(self.run_command(speak_args("hello there", out=out))[0], 0)
        self.assertEqual(len(Path(out).read_bytes()), int(len("hello there") * 0.05 * 24000) * 2)
        out = self.path("d.dat")
        self.assertEqual(self.run_command(speak_args("hello there", out=out, format="pcm"))[0], 0)
        self.assertEqual(len(Path(out).read_bytes()), int(len("hello there") * 0.05 * 24000) * 2)

    def test_the_default_file_name_follows_the_format(self):
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        try:
            self.assertEqual(self.run_command(speak_args("hello there"))[0], 0)
            self.assertEqual(self.run_command(speak_args("hello there", format="pcm"))[0], 0)
        finally:
            os.chdir(cwd)
        self.assertTrue(os.path.exists(self.path("speech.wav")))
        self.assertTrue(os.path.exists(self.path("speech.pcm")))

    def test_the_words_may_come_from_standard_input(self):
        out = self.path("e.wav")
        stdin = sys.stdin
        sys.stdin = io.StringIO("read from the pipe\n")
        try:
            code, _text, _ = self.run_command(speak_args("-", out=out))
        finally:
            sys.stdin = stdin
        self.assertEqual(code, 0)
        self.assertAlmostEqual(len(test_tts_gateway.read_wav(Path(out).read_bytes())[2]), int(len("read from the pipe") * 0.05 * 24000), delta=1)

    def test_no_words_is_a_usage_line(self):
        code, text, _ = self.run_command(speak_args())
        self.assertEqual(code, 1)
        self.assertIn("usage: coli speak", text)

    def test_the_servers_reason_is_shown_and_nothing_is_written(self):
        out = self.path("f.wav")
        code, text, _ = self.run_command(speak_args("hello", voice="morgan", out=out))
        self.assertEqual(code, 1)
        self.assertIn("the server said 400", text)
        self.assertIn("voice", text)
        self.assertFalse(os.path.exists(out))

    def test_a_server_that_is_not_there(self):
        code, text, _ = self.run_command(speak_args("hello"), api=lambda *a, **k: (None, None), fetch=lambda *a: (None, None, None))
        self.assertEqual(code, 1)
        self.assertIn("nothing answers", text)

    def test_a_server_with_no_speech_model_says_how_to_get_one(self):
        def api(method, path, body=None):
            return 200, {"data": [{"id": "chat-model", "capabilities": ["chat"]}]}
        code, text, _ = self.run_command(speak_args("hello"), api=api)
        self.assertEqual(code, 1)
        self.assertIn("coli model install qwen3-tts-0.6b-customvoice", text)

    def test_a_wrong_key_is_said(self):
        _api, _ = clients(self.gw.port, key="wrong")
        code, text, _ = self.run_command(speak_args("hello"), api=_api, fetch=fetcher(self.gw.port, key="wrong"))
        self.assertEqual(code, 1)
        self.assertIn("API key", text)

    def test_an_unwritable_path_is_said(self):
        code, text, _ = self.run_command(speak_args("hello", out=self.path("no/such/dir/x.wav")))
        self.assertEqual(code, 1)
        self.assertIn("cannot write", text)
