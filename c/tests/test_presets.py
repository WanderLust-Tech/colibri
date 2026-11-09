"""Named presets (c/presets.py, applied by c/router.py) and the request log (c/request_log.py).

    python -m unittest tests.test_presets

The router runs over tests/router_fake_worker.py, which echoes what it received, so what is checked is what the router
changed in a request on its way to the model. The gateway half of the request log runs over the scripted engine of
tests/test_openai_server.py.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import presets  # noqa: E402
import request_log  # noqa: E402
import router  # noqa: E402
from test_router import Rig, entry  # noqa: E402

TERSE = {"model": "text-a", "system": "Answer in one sentence.", "description": "short answers",
         "defaults": {"temperature": 0.2, "max_tokens": 300}}
PHOTO = {"model": "image-b", "defaults": {"size": "1024x768", "steps": 20}}
SONG = {"model": "music-m", "defaults": {"duration": 30, "steps": 8}}


def wait_for_lines(path, count, seconds=5.0):
    """The lines of a request log once it has `count` of them (a line is written after its response is sent), or what is there
    when the time is up. count=0 waits a moment for nothing to appear."""
    deadline = time.monotonic() + (seconds if count else 0.5)
    lines = []
    while True:
        lines = [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []
        if (count and len(lines) >= count) or time.monotonic() >= deadline:
            return lines
        time.sleep(0.05)


def write(path, **named):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"presets": named}, handle)


class FileTest(unittest.TestCase):
    """presets.py on its own: what a file may hold, and how it is read."""

    def parse(self, **named):
        return presets.parse(json.dumps({"presets": named}))

    def test_a_good_file_makes_presets(self):
        found, problems = self.parse(terse=TERSE, photo=PHOTO)
        self.assertEqual(problems, [])
        self.assertEqual(sorted(found), ["photo", "terse"])
        self.assertEqual(found["terse"].defaults, {"temperature": 0.2, "max_tokens": 300})
        self.assertEqual(found["terse"].system, "Answer in one sentence.")
        self.assertEqual(found["photo"].public(), {"name": "photo", "model": "image-b", "defaults": PHOTO["defaults"]})

    def test_a_bad_entry_is_reported_and_the_others_stand(self):
        found, problems = self.parse(good=TERSE, Bad=TERSE, nomodel={"defaults": {}}, **{
            "typo": {"model": "x", "default": {}}, "forbidden": {"model": "x", "defaults": {"messages": []}},
            "weird name": TERSE, "text": {"model": "x", "system": 5}})
        self.assertEqual(sorted(found), ["good"])
        why = dict(problems)
        self.assertIn("name must be", why["Bad"])
        self.assertIn("`model` must name", why["nomodel"])
        self.assertIn("unknown key(s): default", why["typo"])
        self.assertIn("cannot be a default", why["forbidden"])
        self.assertIn("name must be", why["weird name"])
        self.assertIn("`system` must be text", why["text"])

    def test_a_file_that_is_not_a_presets_file_is_refused(self):
        for text in ("not json", "[]", '{"presets": []}'):
            with self.subTest(text=text), self.assertRaises(presets.PresetError):
                presets.parse(text)
        self.assertEqual(presets.parse("  "), ({}, []))                  # an empty file is no presets

    def test_a_missing_file_is_no_presets_and_a_written_one_reads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "presets.json")
            self.assertEqual(presets.read(path), ({}, []))
            found, _ = self.parse(terse=TERSE, photo=PHOTO)
            presets.write(path, found)
            again, problems = presets.read(path)
            self.assertEqual(problems, [])
            self.assertEqual({n: p.public() for n, p in again.items()}, {n: p.public() for n, p in found.items()})

    def test_the_file_is_read_again_when_it_changes_and_a_broken_one_keeps_the_last_good_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "presets.json")
            logged = []
            live = presets.Presets(path, log=logged.append)
            self.assertEqual(live.all(), [])
            write(path, terse=TERSE)
            self.assertEqual([p.name for p in live.all()], ["terse"])
            self.assertEqual(live.get("TERSE").name, "terse")              # names are not case sensitive
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{ broken")
            self.assertEqual([p.name for p in live.all()], ["terse"])      # the last good set stands
            self.assertIn("not valid JSON", live.error)
            self.assertTrue(any("keeping the presets read before" in m for m in logged))
            write(path, photo=PHOTO)
            self.assertEqual([p.name for p in live.all()], ["photo"])
            self.assertIsNone(live.error)


class RouterPresetTest(unittest.TestCase):
    """What the router does with a request that names a preset."""

    def rig(self, **named):
        rig = Rig(self, [entry("text-a"), entry("image-b", "image"), entry("music-m", "audio")])
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "presets.json")
        write(self.path, **named)
        rig.server.presets = presets.Presets(self.path, log=lambda message: None)
        return rig

    def chat(self, rig, model, **extra):
        status, _, _, body = rig.call("POST", "/v1/chat/completions", dict(
            {"model": model, "messages": [{"role": "user", "content": "hi"}]}, **extra))
        return status, body

    def test_a_preset_runs_its_model_with_its_defaults_and_system_prompt(self):
        rig = self.rig(terse=TERSE)
        status, body = self.chat(rig, "terse")
        self.assertEqual((status, body["model"]), (200, "text-a-colibri"))
        got = body["received"]
        self.assertEqual((got["model"], got["temperature"], got["max_tokens"]), ("text-a-colibri", 0.2, 300))
        self.assertEqual(got["messages"], [{"role": "system", "content": "Answer in one sentence."},
                                           {"role": "user", "content": "hi"}])

    def test_what_the_request_carries_wins_and_null_is_nothing(self):
        rig = self.rig(terse=TERSE)
        got = self.chat(rig, "terse", temperature=1.0, max_tokens=None)[1]["received"]
        self.assertEqual((got["temperature"], got["max_tokens"]), (1.0, 300))

    def test_a_system_message_of_the_request_keeps_the_preset_out(self):
        rig = self.rig(terse=TERSE)
        for role in ("system", "developer"):
            with self.subTest(role=role):
                messages = [{"role": role, "content": "mine"}, {"role": "user", "content": "hi"}]
                got = self.chat(rig, "terse", messages=messages)[1]["received"]
                self.assertEqual(got["messages"], messages)

    def test_the_anthropic_system_field_is_filled_when_absent(self):
        rig = self.rig(terse=TERSE)
        status, _, _, body = rig.call("POST", "/v1/messages", {"model": "terse", "max_tokens": 9,
                                                                "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(body["received"]["system"], "Answer in one sentence.")
        _, _, _, body = rig.call("POST", "/v1/messages", {"model": "terse", "max_tokens": 9, "system": "mine",
                                                           "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(body["received"]["system"], "mine")

    def test_other_endpoints_take_the_defaults_and_no_system_prompt(self):
        rig = self.rig(**{"photo": PHOTO, "terse": TERSE})
        status, _, _, body = rig.call("POST", "/v1/images/generations", {"model": "photo", "prompt": "a cat", "steps": 4})
        self.assertEqual(body["received"], {"model": "image-b-colibri", "prompt": "a cat", "size": "1024x768", "steps": 4})
        status, _, _, body = rig.call("POST", "/v1/completions", {"model": "terse", "prompt": "once"})
        self.assertNotIn("system", body["received"])
        self.assertEqual(body["received"]["temperature"], 0.2)

    def test_multipart_requests_get_the_defaults_as_fields(self):
        rig = self.rig(song=SONG)
        boundary = "xyz"
        raw = (f'--{boundary}\r\nContent-Disposition: form-data; name="prompt"\r\n\r\ncalm piano\r\n'
               f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nsong\r\n'
               f'--{boundary}\r\nContent-Disposition: form-data; name="duration"\r\n\r\n12\r\n'
               f'--{boundary}--\r\n').encode()
        status, _, _, body = rig.call("POST", "/v1/audio/music", raw=raw,
                                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], "music-m-colibri")
        self.assertEqual(body["received"], {"prompt": "calm piano", "model": "music-m-colibri", "duration": "12",
                                            "steps": "8"})                      # the request's duration kept, the steps added

    def test_the_presets_keep_alive_applies_when_the_request_has_none(self):
        rig = self.rig(quick=dict(TERSE, keep_alive="2m"))
        self.chat(rig, "quick")
        self.assertEqual(rig.pool.get(rig.entries[0]["key"]).keep_alive, 120.0)
        self.chat(rig, "quick", keep_alive="5m")
        self.assertEqual(rig.pool.get(rig.entries[0]["key"]).keep_alive, 300.0)

    def test_the_preset_is_listed_as_a_model_of_its_own(self):
        rig = self.rig(terse=TERSE)
        status, _, _, body = rig.call("GET", "/v1/models")
        by_id = {m["id"]: m for m in body["data"]}
        self.assertIn("terse", by_id)
        self.assertEqual(by_id["terse"]["preset_of"], "text-a-colibri")
        self.assertEqual(by_id["terse"]["preset"]["defaults"], TERSE["defaults"])
        self.assertEqual(by_id["terse"]["capabilities"], by_id["text-a-colibri"]["capabilities"])
        status, _, _, one = rig.call("GET", "/v1/models/terse")
        self.assertEqual((status, one["id"], one["preset_of"]), (200, "terse", "text-a-colibri"))

    def test_a_preset_of_a_model_that_is_not_here_is_not_listed_and_says_so_when_asked_for(self):
        rig = self.rig(ghost={"model": "no-such-model"}, terse=TERSE)
        ids = [m["id"] for m in rig.call("GET", "/v1/models")[3]["data"]]
        self.assertNotIn("ghost", ids)
        status, body = self.chat(rig, "ghost")
        self.assertEqual((status, body["error"]["code"]), (404, "model_not_found"))
        self.assertIn("preset `ghost` names the model `no-such-model`", body["error"]["message"])
        listing = rig.call("GET", "/v1/presets")[3]
        rows = {p["name"]: p for p in listing["data"]}
        self.assertFalse(rows["ghost"]["available"])
        self.assertIn("not installed", rows["ghost"]["reason"])
        self.assertTrue(rows["terse"]["available"])
        self.assertEqual(rows["terse"]["resolved"], "text-a-colibri")

    def test_a_real_models_name_is_never_taken_by_a_preset(self):
        rig = self.rig(**{"text-a": dict(TERSE, model="text-a")})
        status, body = self.chat(rig, "text-a")
        self.assertEqual(status, 200)
        self.assertNotIn("temperature", body["received"])                   # the model, not the preset
        row = rig.call("GET", "/v1/presets")[3]["data"][0]
        self.assertFalse(row["available"])
        self.assertIn("the name is a model's own", row["reason"])

    def test_a_file_edited_while_the_router_runs_is_taken_up(self):
        rig = self.rig(terse=TERSE)
        self.assertEqual(self.chat(rig, "terse")[0], 200)
        write(self.path, photo=PHOTO, terse=dict(TERSE, defaults={"temperature": 0.9}))
        self.assertEqual(self.chat(rig, "terse")[1]["received"]["temperature"], 0.9)
        self.assertEqual(rig.call("POST", "/v1/images/generations", {"model": "photo", "prompt": "x"})[0], 200)

    def test_switch_and_unload_take_a_preset_name(self):
        rig = self.rig(terse=TERSE)
        status, _, _, body = rig.call("POST", "/v1/models/switch", {"model": "terse"})
        self.assertEqual((status, body["id"]), (200, "text-a-colibri"))
        self.assertEqual(rig.states()["text-a-colibri"], "ready")
        status, _, _, body = rig.call("POST", "/v1/models/unload", {"model": "terse"})
        self.assertEqual(status, 200)

    def test_a_router_without_presets_is_what_it_was(self):
        rig = Rig(self, [entry("text-a")])
        self.assertIsNone(rig.server.presets)
        status, _, _, body = rig.chat("text-a-colibri")
        self.assertEqual(status, 200)
        self.assertEqual(rig.call("GET", "/v1/presets")[3]["data"], [])
        self.assertEqual(rig.call("POST", "/v1/chat/completions", {"model": "terse", "messages": []})[0], 404)


class LogFileTest(unittest.TestCase):
    """request_log.py on its own."""

    def test_a_line_per_record_with_a_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = request_log.RequestLog(os.path.join(tmp, "deep", "req.log"))
            log.write({"path": "/v1/x", "status": 200})
            log.write({"path": "/v1/y", "ts": "kept"})
            lines = [json.loads(line) for line in open(log.path, encoding="utf-8")]
            self.assertEqual([r["path"] for r in lines], ["/v1/x", "/v1/y"])
            self.assertRegex(lines[0]["ts"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
            self.assertEqual(lines[1]["ts"], "kept")

    def test_a_file_over_the_limit_moves_to_dot_1_and_one_old_file_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = request_log.RequestLog(os.path.join(tmp, "req.log"), max_bytes=300)
            for i in range(12):
                log.write({"n": i, "pad": "x" * 60})
            self.assertTrue(os.path.exists(log.path + ".1"))
            self.assertFalse(os.path.exists(log.path + ".2"))
            kept = [json.loads(l)["n"] for p in (log.path + ".1", log.path) for l in open(p, encoding="utf-8")]
            self.assertEqual(kept, sorted(kept))
            self.assertEqual(kept[-1], 11)
            self.assertLessEqual(os.path.getsize(log.path), 300)

    def test_a_log_that_cannot_be_written_does_not_raise_and_says_so_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = os.path.join(tmp, "file")
            open(blocker, "w").close()
            log = request_log.RequestLog(os.path.join(blocker, "req.log"))        # a directory below a file: impossible
            with patch("sys.stderr") as stderr:
                log.write({"a": 1})
                log.write({"a": 2})
            self.assertEqual(sum("cannot write" in str(c) for c in stderr.write.call_args_list), 1)

    def test_long_strings_are_cut_and_bytes_are_counted(self):
        out = request_log.shorten({"a": "x" * 2000, "b": [1, {"c": "y" * 600}], "d": b"1234"}, limit=100)
        self.assertEqual(len(out["a"]), 100 + len("...[1900 more characters]"))
        self.assertTrue(out["a"].endswith("...[1900 more characters]"))
        self.assertEqual(out["b"][0], 1)
        self.assertTrue(out["b"][1]["c"].endswith("[500 more characters]"))
        self.assertEqual(out["d"], "<4 bytes>")

    def test_the_environment_decides(self):
        self.assertIsNone(request_log.from_env({}))
        self.assertIsNone(request_log.from_env({"COLI_REQUEST_LOG": "  "}))
        log = request_log.from_env({"COLI_REQUEST_LOG": "a.log", "COLI_REQUEST_LOG_BODIES": "1", "COLI_REQUEST_LOG_MAX": "99"})
        self.assertEqual((os.path.basename(log.path), log.bodies, log.max_bytes), ("a.log", True, 99))
        self.assertEqual(request_log.from_env({"COLI_REQUEST_LOG": "a.log", "COLI_REQUEST_LOG_MAX": "x"}).max_bytes,
                         request_log.DEFAULT_MAX_BYTES)
        self.assertFalse(request_log.from_env({"COLI_REQUEST_LOG": "a.log"}).bodies)


class RouterLogTest(unittest.TestCase):
    def rig(self):
        rig = Rig(self, [entry("text-a")])
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "req.log")
        return rig

    def lines(self, count=1):
        return wait_for_lines(self.path, count)

    def test_nothing_is_written_unless_asked(self):
        rig = self.rig()
        rig.chat("text-a-colibri")
        self.assertEqual(self.lines(0), [])

    def test_a_request_leaves_one_line_the_worker_can_be_joined_to(self):
        rig = self.rig()
        rig.server.request_log = request_log.RequestLog(self.path)
        write(os.path.join(self.tmp.name, "p.json"), terse=TERSE)
        rig.server.presets = presets.Presets(os.path.join(self.tmp.name, "p.json"), log=lambda m: None)
        status, _, _, body = rig.call("POST", "/v1/chat/completions", {"model": "terse", "messages": [], "stream": False})
        self.assertEqual(status, 200)
        rig.call("GET", "/v1/models")                                     # not a request to compute: not logged
        (line,) = self.lines()
        self.assertEqual((line["source"], line["path"], line["model"], line["preset"], line["resolved"], line["status"]),
                         ("router", "/v1/chat/completions", "terse", "terse", "text-a-colibri", 200))
        self.assertGreaterEqual(line["seconds"], 0)
        self.assertEqual(body["trace"], line["trace"])                    # the worker was told the same id
        self.assertNotIn("request", line)                                  # no text of the request without the opt-in

    def test_a_refused_request_is_logged_with_its_error(self):
        rig = self.rig()
        rig.server.request_log = request_log.RequestLog(self.path)
        self.assertEqual(rig.chat("nope")[0], 404)
        (line,) = self.lines()
        self.assertEqual((line["status"], line["error"], line["model"]), (404, "model_not_found", "nope"))
        self.assertNotIn("resolved", line)

    def test_the_text_of_the_request_is_logged_only_when_asked_and_cut(self):
        rig = self.rig()
        rig.server.request_log = request_log.RequestLog(self.path, bodies=True)
        rig.call("POST", "/v1/chat/completions", {"model": "text-a-colibri", "messages": [
            {"role": "user", "content": "q" * 3000}]})
        (line,) = self.lines()
        self.assertTrue(line["request"]["messages"][0]["content"].endswith("more characters]"))
        self.assertLess(len(json.dumps(line)), 1500)


class GatewayLogTest(unittest.TestCase):
    """The gateway's line: the same file, with the tokens only it knows."""

    def setUp(self):
        from test_openai_server import ScriptedEngine, _spawn_test_server
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "req.log")
        self.spawn = lambda **kw: _spawn_test_server(self, ScriptedEngine(
            chunks=["Hel", "lo"], supports_logprobs_echo=False, supports_tok_ids=False, prompt_tokens=7, **kw))

    def lines(self, count=1):
        return wait_for_lines(self.path, count)

    def post(self, base, body, headers=None):
        from urllib.request import Request, urlopen
        request = Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=dict(
            {"Authorization": "Bearer secret", "Content-Type": "application/json"}, **(headers or {})))
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    def test_nothing_is_written_unless_asked(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COLI_REQUEST_LOG", None)
            base = self.spawn()
            self.post(base, {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(self.lines(0), [])

    def test_a_request_leaves_a_line_with_its_tokens(self):
        with patch.dict(os.environ, {"COLI_REQUEST_LOG": self.path}):
            base = self.spawn()
            self.post(base, {"model": "test-model", "stream": False, "messages": [{"role": "user", "content": "hi"}]},
                      {"x-colibri-trace": "abc123"})
        (line,) = self.lines()
        self.assertEqual((line["source"], line["path"], line["model"], line["status"], line["trace"]),
                         ("gateway", "/v1/chat/completions", "test-model", 200, "abc123"))
        self.assertEqual((line["prompt_tokens"], line["completion_tokens"], line["finish"]), (7, 2, "stop"))
        self.assertTrue(line["id"].startswith("req_"))
        self.assertEqual(line["stream"], False)
        self.assertNotIn("request", line)

    def test_a_refused_request_is_logged_with_its_error_and_no_tokens(self):
        from urllib.error import HTTPError
        with patch.dict(os.environ, {"COLI_REQUEST_LOG": self.path}):
            base = self.spawn()
            with self.assertRaises(HTTPError) as caught:
                self.post(base, {"model": "test-model", "messages": "not a list"})
            caught.exception.close()
        (line,) = self.lines()
        self.assertEqual(line["status"], 400)
        self.assertTrue(line.get("error"))
        self.assertNotIn("prompt_tokens", line)

    def test_the_bodies_only_with_the_opt_in(self):
        with patch.dict(os.environ, {"COLI_REQUEST_LOG": self.path, "COLI_REQUEST_LOG_BODIES": "1"}):
            base = self.spawn()
            self.post(base, {"model": "test-model", "messages": [{"role": "user", "content": "a secret question"}]})
        (line,) = self.lines()
        self.assertEqual(line["request"]["messages"][0]["content"], "a secret question")


class CommandTest(unittest.TestCase):
    """`coli preset`: it edits the file, and what it writes is what the router reads."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = os.path.join(self.tmp.name, "presets.json")

    def coli(self, *args):
        import subprocess
        env = dict(os.environ, PYTHONUTF8="1", COLI_COLOR="0")
        return subprocess.run([sys.executable, str(HERE.parent / "coli"), "preset", *args, "--file", self.file],
                              capture_output=True, text=True, env=env, errors="replace", timeout=120)

    def test_add_list_show_replace_and_remove(self):
        out = self.coli("list")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("no presets yet", out.stdout)
        out = self.coli("add", "Terse", "--model", "qwen3-coder-30b-a3b", "--system", "Be brief.", "--set", "temperature=0.2",
                        "--set", "max_tokens=300", "--set", "size=1024x768", "--set", "stop_thinking=true",
                        "--description", "short", "--keep-alive", "10m")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("added terse", out.stdout)
        found, problems = presets.read(self.file)
        self.assertEqual(problems, [])
        terse = found["terse"]                                              # the name is lower-cased
        self.assertEqual(terse.defaults, {"temperature": 0.2, "max_tokens": 300, "size": "1024x768", "stop_thinking": True})
        self.assertEqual((terse.system, terse.description, terse.keep_alive), ("Be brief.", "short", "10m"))
        out = self.coli("list")
        self.assertIn("terse", out.stdout)
        self.assertIn("temperature=0.2", out.stdout)
        self.assertIn("system prompt", out.stdout)
        shown = self.coli("show", "terse").stdout                              # the banner comes first
        self.assertEqual(json.loads(shown[shown.index("{"):])["model"], "qwen3-coder-30b-a3b")
        self.assertIn("replaced terse", self.coli("add", "terse", "--model", "other").stdout)
        self.assertEqual(presets.read(self.file)[0]["terse"].model, "other")
        self.assertEqual(self.coli("rm", "terse").returncode, 0)
        self.assertEqual(presets.read(self.file)[0], {})
        self.assertEqual(self.coli("rm", "terse").returncode, 1)
        self.assertEqual(self.coli("show", "terse").returncode, 1)

    def test_a_bad_preset_is_refused_with_the_reason_and_the_file_is_not_touched(self):
        write(self.file, kept=TERSE)
        before = open(self.file, encoding="utf-8").read()
        for args, why in ((("add", "x"), "--model"), (("add", "Bad Name", "--model", "m"), "name must be"),
                          (("add", "x", "--model", "m", "--set", "messages=[]"), "cannot be a default"),
                          (("add", "x", "--model", "m", "--set", "novalue"), "KEY=VALUE")):
            with self.subTest(args=args):
                out = self.coli(*args)
                self.assertEqual(out.returncode, 1)
                self.assertIn(why, out.stderr + out.stdout)
        self.assertEqual(open(self.file, encoding="utf-8").read(), before)

    def test_an_entry_this_version_cannot_read_survives_an_edit(self):
        with open(self.file, "w", encoding="utf-8") as handle:
            json.dump({"presets": {"future": {"model": "m", "newkey": 1}}}, handle)
        self.assertEqual(self.coli("add", "terse", "--model", "m").returncode, 0)
        raw = json.load(open(self.file, encoding="utf-8"))
        self.assertEqual(sorted(raw["presets"]), ["future", "terse"])       # not dropped for being unreadable
        out = self.coli("list")
        self.assertEqual(out.returncode, 1)                                  # and the list says it is not usable
        self.assertIn("future", out.stdout)
        self.assertIn("unknown key(s): newkey", out.stdout)


if __name__ == "__main__":
    unittest.main()
