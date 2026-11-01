"""The dashboard's model switcher: which models are installed, which of them this
server can start, and the one-way trip from POST /v1/models/switch to a relaunch.

A server process serves exactly one model, so a switch is a restart. What these
tests pin is everything that must be true for that restart to be safe: the browser
names a model by key and never by path, a foreign web page cannot trigger one, a
model the launcher could not start is not offered, a half-downloaded model is not
listed, and the relaunch command drops the flags that named the old model."""
import importlib.machinery
import importlib.util
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import model_switch
from family_registry import family_by_id
from openai_server import APIServer

GLM = {"model_type": "glm_moe_dsa"}
MIMO_FLASH = {"model_type": "mimo_v2", "hidden_size": 4096, "num_hidden_layers": 48,
              "n_routed_experts": 256}


_loader = importlib.machinery.SourceFileLoader("coli_cli_model_switch", str(Path(__file__).resolve().parent.parent / "coli"))
_spec = importlib.util.spec_from_loader("coli_cli_model_switch", _loader)
coli = importlib.util.module_from_spec(_spec)
_loader.exec_module(coli)


def write_model(root, name, config, *, manifest=None):
    """A model folder: a config, and optionally the downloader's manifest with one
    real file so is_complete() can stat it."""
    path = Path(root) / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if manifest is not None:
        (path / "weights.bin").write_bytes(b"1234")
        (path / ".colibri-download.json").write_text(json.dumps(
            {"complete": manifest, "files": [{"path": "weights.bin", "size": 4}]}), encoding="utf-8")
    return str(path)


def engines_built(_family):
    return True


class ReplaceModelTest(unittest.TestCase):
    def test_replaces_the_model_flag_in_both_spellings(self):
        for args in (["web", "--model", "/old", "--port", "8000"],
                     ["web", "--model=/old", "--port", "8000"]):
            with self.subTest(args=args):
                out = model_switch.replace_model(args, "/new")
                self.assertEqual(out[:3], ["web", "--model", "/new"])
                self.assertNotIn("/old", out)
                self.assertEqual(out[3:5], ["--port", "8000"])

    def test_drops_the_model_id_that_named_the_old_model(self):
        for args in (["serve", "--model", "/old", "--model-id", "mine"],
                     ["serve", "--model-id=mine", "--model", "/old"]):
            with self.subTest(args=args):
                out = model_switch.replace_model(args, "/new")
                self.assertNotIn("--model-id", out)
                self.assertNotIn("mine", out)
                self.assertEqual(out.count("--model"), 1)

    def test_adds_the_flag_after_the_subcommand_when_the_model_came_from_the_environment(self):
        self.assertEqual(model_switch.replace_model(["serve", "--port", "9"], "/new"),
                         ["serve", "--model", "/new", "--port", "9"])

    def test_web_does_not_open_a_second_browser_tab(self):
        self.assertIn("--no-browser", model_switch.replace_model(["web", "--model", "/old"], "/new"))
        self.assertNotIn("--no-browser", model_switch.replace_model(["serve", "--model", "/old"], "/new"))

    def test_environment_carries_the_way_back_and_not_the_old_id(self):
        env = model_switch.relaunch_env("/new", "/old", {"COLI_MODEL_ID": "mine", "COLI_MODEL": "/old", "X": "1"})
        self.assertEqual(env[model_switch.FALLBACK_ENV], "/old")
        self.assertNotIn("COLI_MODEL_ID", env)
        self.assertEqual(env["COLI_MODEL"], "/new")
        self.assertEqual(env["X"], "1")
        # the way back has no way back of its own: no loop between two bad models
        self.assertNotIn(model_switch.FALLBACK_ENV, model_switch.relaunch_env("/old", None, env))


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.current = write_model(self.root, "glm", GLM)
        self.flash = write_model(self.root, "mimo-flash", MIMO_FLASH, manifest=True)
        write_model(self.root, "half-downloaded", MIMO_FLASH, manifest=False)
        write_model(self.root, "not-ours", {"model_type": "no_such_family"})
        (self.root / "empty-dir").mkdir()
        self.ctx = model_switch.SwitchContext(self.current, roots=[str(self.root)], environ={})

    def scan(self, ctx=None):
        return model_switch.scan_installed(ctx or self.ctx, engines_built)

    def test_lists_complete_models_current_first(self):
        entries = self.scan()
        self.assertEqual([e["name"] for e in entries][0], entries[0]["name"])
        self.assertTrue(entries[0]["current"])
        self.assertEqual(entries[0]["id"], "glm-5.2-colibri")
        flash = next(e for e in entries if e["id"] == "mimo-v2.6-flash")
        self.assertEqual((flash["family"], flash["size_bytes"], flash["verified"]), ("mimo", 4, True))
        self.assertTrue(flash["switchable"])
        self.assertEqual(len(entries), 2)

    def test_leaves_out_downloads_in_progress_and_folders_we_do_not_serve(self):
        paths = {e["path"] for e in self.scan()}
        self.assertEqual(paths, {os.path.realpath(self.current), os.path.realpath(self.flash)})

    def test_a_model_fetched_by_hand_is_listed_as_unverified(self):
        by_hand = write_model(self.root, "by-hand", MIMO_FLASH)
        entry = next(e for e in self.scan() if e["path"] == os.path.realpath(by_hand))
        self.assertFalse(entry["verified"])
        self.assertIsNone(entry["size_bytes"])

    def test_the_served_model_is_listed_even_outside_every_root(self):
        elsewhere = tempfile.TemporaryDirectory()
        self.addCleanup(elsewhere.cleanup)
        current = write_model(elsewhere.name, "served", GLM)
        ctx = model_switch.SwitchContext(current, roots=[str(self.root)], environ={})
        entries = self.scan(ctx)
        self.assertEqual(entries[0]["path"], os.path.realpath(current))
        self.assertTrue(entries[0]["current"])

    def test_the_current_model_is_never_a_switch_target(self):
        self.assertFalse(self.scan()[0]["switchable"])

    def test_public_entries_carry_no_filesystem_path(self):
        self.assertNotIn("path", model_switch.public(self.scan()[0]))

    def test_the_key_is_stable_and_differs_per_folder(self):
        keys = [e["key"] for e in self.scan()]
        self.assertEqual(keys, [e["key"] for e in self.scan()])
        self.assertEqual(len(set(keys)), len(keys))

    def test_roots_come_from_the_environment_and_the_served_models_parent(self):
        extra = tempfile.TemporaryDirectory()
        self.addCleanup(extra.cleanup)
        roots = model_switch.candidate_roots(self.current, {model_switch.ROOT_ENV: extra.name})
        self.assertIn(os.path.realpath(extra.name), roots)
        self.assertIn(os.path.realpath(self.root), roots)


class WhyNotTest(unittest.TestCase):
    def ctx(self, **kw):
        return model_switch.SwitchContext(tempfile.gettempdir(), environ={}, **kw)

    def test_a_model_the_flags_allow_has_no_reason(self):
        self.assertIsNone(model_switch.why_not(family_by_id("mimo"), self.ctx(), engines_built))

    def test_an_engine_that_is_not_built(self):
        self.assertIn("not built", model_switch.why_not(family_by_id("mimo"), self.ctx(), lambda f: False))

    def test_more_kv_slots_than_the_family_allows(self):
        family = family_by_id("mimo")
        reason = model_switch.why_not(family, self.ctx(kv_slots=family.limits.max_kv_slots + 1), engines_built)
        self.assertIn("KV slot", reason)

    def test_a_music_model_can_be_switched_to_like_any_other(self):
        self.assertIsNone(model_switch.why_not(family_by_id("ace_step"), self.ctx(), engines_built))

    def test_cluster_mode_is_for_glm_only(self):
        self.assertIsNone(model_switch.why_not(family_by_id("glm"), self.ctx(cluster=True), engines_built))
        self.assertIn("cluster", model_switch.why_not(family_by_id("mimo"), self.ctx(cluster=True), engines_built))

    def test_a_pinned_engine_disables_switching_altogether(self):
        ctx = model_switch.SwitchContext(tempfile.gettempdir(), environ={"COLI_ENGINE": "/x/glm"})
        self.assertFalse(ctx.enabled)
        self.assertIn("COLI_ENGINE", ctx.reason)


class SwitchHTTPTest(unittest.TestCase):
    KEY = "secret"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.current = write_model(root, "glm", GLM)
        self.flash = write_model(root, "mimo-flash", MIMO_FLASH, manifest=True)
        patcher = patch("openai_server.default_engine", lambda family: types.SimpleNamespace(exists=lambda: True))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.start(model_switch.SwitchContext(self.current, roots=[str(root)], environ={}))

    def start(self, switch):
        self.server = APIServer(("127.0.0.1", 0), None, "glm-5.2-colibri", self.KEY, 64)
        self.server.switch = switch
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.stop)

    def stop(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def call(self, path, body=None, headers=None, auth=True):
        head = {"Content-Type": "application/json"}
        if auth:
            head["Authorization"] = f"Bearer {self.KEY}"
        head.update(headers or {})
        data = None if body is None else json.dumps(body).encode()
        try:
            with urlopen(Request(self.base + path, data=data, headers=head), timeout=3) as response:
                return response.status, json.loads(response.read() or b"{}")
        except HTTPError as error:
            with error:
                return error.code, json.loads(error.read() or b"{}")

    def flash_key(self):
        _, listing = self.call("/v1/models/installed")
        return next(m["key"] for m in listing["data"] if m["id"] == "mimo-v2.6-flash")

    def test_lists_installed_models_without_paths(self):
        status, listing = self.call("/v1/models/installed")
        self.assertEqual(status, 200)
        self.assertTrue(listing["switch"])
        self.assertEqual(len(listing["data"]), 2)
        self.assertNotIn("path", json.dumps(listing))
        self.assertTrue(listing["data"][0]["current"])

    def test_listing_and_switching_need_the_api_key(self):
        self.assertEqual(self.call("/v1/models/installed", auth=False)[0], 401)
        self.assertEqual(self.call("/v1/models/switch", {"key": self.flash_key()}, auth=False)[0], 401)
        self.assertIsNone(self.server.relaunch_to)

    def test_switch_answers_202_and_stops_the_server_for_the_launcher(self):
        status, body = self.call("/v1/models/switch", {"key": self.flash_key()})
        self.assertEqual((status, body["status"], body["id"]), (202, "switching", "mimo-v2.6-flash"))
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive(), "serve_forever should have returned")
        self.assertEqual(self.server.relaunch_to, os.path.realpath(self.flash))

    def test_a_foreign_origin_cannot_switch(self):
        status, body = self.call("/v1/models/switch", {"key": self.flash_key()},
                                 headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertIsNone(self.server.relaunch_to)

    def test_a_plain_text_body_is_refused(self):
        status, _ = self.call("/v1/models/switch", {"key": self.flash_key()},
                              headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        self.assertIsNone(self.server.relaunch_to)

    def test_refuses_what_it_cannot_or_should_not_start(self):
        _, listing = self.call("/v1/models/installed")
        current = next(m["key"] for m in listing["data"] if m["current"])
        for body, expected in (({"key": "nope"}, 404), ({"key": current}, 409),
                               ({"key": 5}, 400), ({}, 400)):
            with self.subTest(body=body):
                self.assertEqual(self.call("/v1/models/switch", body)[0], expected)
        self.assertIsNone(self.server.relaunch_to)

    def test_does_not_cut_off_a_response_in_progress(self):
        self.server.scheduler.active = 1
        status, body = self.call("/v1/models/switch", {"key": self.flash_key()})
        self.assertEqual((status, body["error"]["code"]), (409, "busy"))
        self.assertIsNone(self.server.relaunch_to)

    def test_a_server_the_launcher_did_not_start_reports_it_cannot_switch(self):
        self.stop()
        self.start(None)
        status, listing = self.call("/v1/models/installed")
        self.assertEqual((status, listing["switch"], listing["data"]), (200, False, []))
        self.assertEqual(self.call("/v1/models/switch", {"key": "x"})[0], 501)


class LauncherTest(unittest.TestCase):
    """cmd_serve: serve, then relaunch on what the dashboard asked for, or go back
    to the previous model when the one it switched to did not start."""

    def run_serve(self, serve, *, fallback=None, argv=("web", "--model", "/old", "--model-id", "mine")):
        env = {model_switch.FALLBACK_ENV: fallback} if fallback else {}
        relaunched = []
        with patch.dict(os.environ, env, clear=False),                 patch.object(sys, "argv", ["coli", *argv]),                 patch.object(coli, "_cmd_serve", serve),                 patch.object(model_switch, "relaunch", lambda cmd, env: relaunched.append((cmd, env))):
            if not fallback:
                os.environ.pop(model_switch.FALLBACK_ENV, None)
            try:
                coli.cmd_serve(types.SimpleNamespace(model="/old"))
            except SystemExit as error:
                return relaunched, error
        return relaunched, None

    def test_a_requested_switch_relaunches_on_the_new_model_with_a_way_back(self):
        relaunched, _ = self.run_serve(lambda a: "/new")
        (cmd, env), = relaunched
        self.assertEqual(cmd[cmd.index("--model") + 1], "/new")
        self.assertEqual(cmd[-1], "--no-browser")
        self.assertNotIn("/old", cmd)
        self.assertNotIn("--model-id", cmd)
        self.assertEqual(env[model_switch.FALLBACK_ENV], "/old")

    def test_a_normal_exit_relaunches_nothing(self):
        self.assertEqual(self.run_serve(lambda a: None), ([], None))

    def test_a_new_model_that_fails_to_start_sends_us_back_without_a_further_way_back(self):
        def fails(a):
            raise SystemExit(1)
        relaunched, error = self.run_serve(fails, fallback="/previous")
        self.assertIsNone(error)
        (cmd, env), = relaunched
        self.assertIn("/previous", cmd)
        self.assertNotIn(model_switch.FALLBACK_ENV, env)

    def test_a_failure_with_no_way_back_still_fails(self):
        def fails(a):
            raise SystemExit(1)
        relaunched, error = self.run_serve(fails)
        self.assertEqual((relaunched, error.code), ([], 1))

    def test_a_clean_stop_of_a_switched_server_does_not_go_back(self):
        def stops(a):
            raise SystemExit(0)
        relaunched, error = self.run_serve(stops, fallback="/previous")
        self.assertEqual((relaunched, error.code), ([], 0))


if __name__ == "__main__":
    unittest.main()
