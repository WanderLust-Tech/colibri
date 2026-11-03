"""`coli ps` and `coli stop MODEL`, run as a user runs them, against a router over fake workers (and against a plain
single-model server, which has no model to unload on its own)."""
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_router import GB, Rig, entry  # noqa: E402

CLI = C_DIR / "coli"


def coli(*args, env_extra=None):
    env = dict(os.environ, PYTHONUTF8="1", COLI_COLOR="0")
    env.pop("COLI_API_KEY", None)
    env.update(env_extra or {})
    result = subprocess.run([sys.executable, str(CLI), *args], env=env, capture_output=True, text=True, timeout=60,
                            cwd=str(C_DIR))
    return result.returncode, result.stdout + result.stderr


class PsAndStopTest(unittest.TestCase):
    def test_ps_lists_what_is_loaded(self):
        rig = Rig(self, [entry("text-a", size=5 * GB), entry("text-b")], keep_alive=600.0)
        code, out = coli("ps", "--port", str(rig.port))
        self.assertEqual(code, 0, out)
        self.assertIn("no models loaded", out)
        rig.chat("text-a-colibri")
        code, out = coli("ps", "--port", str(rig.port))
        self.assertEqual(code, 0, out)
        header = next(line for line in out.splitlines() if "NAME" in line)
        row = next(line for line in out.splitlines() if "text-a-colibri" in line)
        self.assertEqual(header.split(), ["NAME", "STATE", "MEMORY", "UNLOADS", "IN"])
        self.assertEqual(row.split()[1], "loaded")
        self.assertRegex(row, r"\d+m \d\ds")                           # about 10 minutes left
        self.assertNotIn("text-b-colibri", out, "a model that is not loaded is not listed")

    def test_ps_says_never_for_a_model_that_stays(self):
        rig = Rig(self, [entry("text-a")], keep_alive=600.0)
        rig.chat("text-a-colibri", keep_alive=-1)
        row = next(line for line in coli("ps", "--port", str(rig.port))[1].splitlines() if "text-a-colibri" in line)
        self.assertTrue(row.rstrip().endswith("never"), row)

    def test_ps_needs_the_key_when_there_is_one(self):
        rig = Rig(self, [entry("text-a")], api_key="sekrit")
        code, out = coli("ps", "--port", str(rig.port))
        self.assertEqual(code, 1)
        self.assertIn("API key", out)
        code, out = coli("ps", "--port", str(rig.port), "--api-key", "sekrit")
        self.assertEqual(code, 0, out)
        code, out = coli("ps", "--port", str(rig.port), env_extra={"COLI_API_KEY": "sekrit"})
        self.assertEqual(code, 0, out)

    def test_ps_when_nothing_answers(self):
        code, out = coli("ps", "--port", "1")
        self.assertEqual(code, 1)
        self.assertIn("no server answers", out)

    def test_stop_with_a_model_unloads_just_that_one(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")])
        rig.chat("text-a-colibri")
        rig.chat("text-b-colibri")
        code, out = coli("stop", "text-a", "--port", str(rig.port))
        self.assertEqual(code, 0, out)
        self.assertIn("unloaded text-a-colibri", out)
        self.assertEqual(rig.states(), {"text-a-colibri": "idle", "text-b-colibri": "ready"})
        code, out = coli("stop", "text-a", "--port", str(rig.port))
        self.assertEqual(code, 0, out)
        self.assertIn("was not loaded", out)

    def test_stop_says_when_there_is_no_such_model(self):
        rig = Rig(self, [entry("text-a")])
        code, out = coli("stop", "nope", "--port", str(rig.port))
        self.assertEqual(code, 1)
        self.assertIn("no installed model is called nope", out)

    def test_stop_with_a_model_needs_the_key_too(self):
        rig = Rig(self, [entry("text-a")], api_key="sekrit")
        self.assertEqual(coli("stop", "text-a", "--port", str(rig.port))[0], 1)
        code, out = coli("stop", "text-a", "--port", str(rig.port), "--api-key", "sekrit")
        self.assertEqual(code, 0, out)

    def test_plain_stop_still_means_the_whole_server(self):
        code, out = coli("stop", "--dry-run", "--port", "59999")
        self.assertEqual(code, 0, out)
        self.assertIn("nothing running", out)

    def test_a_router_is_found_by_coli_stop(self):
        loader = importlib.machinery.SourceFileLoader("coli_router_test", str(CLI))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        match = module._serve_cmdline_matches_port
        self.assertTrue(match(b"python\0/x/coli\0router\0--port\0 8123".replace(b" ", b""), 8123))
        self.assertTrue(match(b"python\0/x/coli\0router\0", 8000))
        self.assertTrue(match(b"python\0/x/coli\0serve\0--port\0" b"8123", 8123))
        self.assertFalse(match(b"python\0/x/coli\0chat\0--port\0" b"8123", 8123))
        self.assertFalse(match(b"python\0/x/coli\0router\0--port\0" b"8124", 8123))


class PlainServerTest(unittest.TestCase):
    """A `coli serve` has one model for as long as it runs: ps says so, and stop MODEL says what to do instead."""

    def setUp(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self.reply(200, {"object": "list", "data": [{"id": "only-colibri"}]})
                self.reply(404, {"error": {"message": "Not found.", "code": "not_found"}})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.reply(404, {"error": {"message": "Not found.", "code": "not_found"}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (self.server.shutdown(), self.server.server_close()))
        self.port = str(self.server.server_address[1])

    def test_ps_names_the_one_model(self):
        code, out = coli("ps", "--port", self.port)
        self.assertEqual(code, 0, out)
        self.assertIn("single-model server", out)
        self.assertIn("only-colibri", out)

    def test_stop_with_a_model_says_this_is_not_a_router(self):
        code, out = coli("stop", "only", "--port", self.port)
        self.assertEqual(code, 1)
        self.assertIn("not a `coli router`", out)


if __name__ == "__main__":
    unittest.main()
