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
