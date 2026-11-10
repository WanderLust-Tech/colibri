"""Where a model runs, and which models are on: gpu.py, model_settings.py, and how the router uses them (docs/gpu-offload.md).

    python -m unittest tests.test_gpu

The card is a double (a function that says how much memory is free), the workers are tests/router_fake_worker.py. What is checked:
the policy's decisions at the edges, that a worker is started with the environment the decision made, that the choice is
made again at every load, the settings file (read, changed, read again, broken), and a model turned off: refused with a 403 that
says how to turn it on, unloaded when it is turned off, absent from presets, and back when turned on."""
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

import gpu  # noqa: E402
import model_settings  # noqa: E402
import presets as presets_mod  # noqa: E402
import router  # noqa: E402
from test_router import Rig, entry  # noqa: E402


class CapturingRig(Rig):
    """A Rig that keeps the environment each worker was started with."""

    def __init__(self, *args, **kw):
        self.started = []
        super().__init__(*args, **kw)
        original = self.pool._command

        def command(item):
            result = original(item)
            self.started.append((item["id"], {k: v for k, v in result[1].items()
                                              if k in ("COLI_VULKAN", "COLI_VK_TIER_RESERVE_GB", "G4_VK_RESERVE_GB")}))
            return result

        self.pool._command = command


class FreeCard:
    """A card whose free memory a test sets."""

    def __init__(self, free=24.0, total=32.0):
        self.free, self.total, self.reads = free, total, 0

    def __call__(self):
        self.reads += 1
        return None if self.free is None else (self.free, self.total)


def policy(mode, card, **kw):
    made = gpu.GpuPolicy(mode, probe=card, **kw)
    made.memory = lambda: card()          # no caching in tests that change the card between loads
    return made


class PolicyTest(unittest.TestCase):
    def test_no_policy_leaves_the_environment_alone(self):
        decision = gpu.GpuPolicy(None, probe=FreeCard()).decide()
        self.assertEqual((decision.mode, decision.env), ("unchanged", {}))

    def test_off_and_on(self):
        card = FreeCard(1.0)
        self.assertEqual(gpu.GpuPolicy("off", probe=card).decide().env, {"COLI_VULKAN": "0"})
        decision = gpu.GpuPolicy("on", reserve_gb=4, probe=card).decide()
        self.assertEqual((decision.mode, decision.env), ("vulkan", {"COLI_VULKAN": "1", "COLI_VK_TIER_RESERVE_GB": "4",
                                                                      "G4_VK_RESERVE_GB": "4"}))

    def test_auto_follows_the_free_memory_at_the_edges(self):
        for free, want in ((20.0, "vulkan"), (9.0, "vulkan"), (8.99, "cpu"), (6.0, "cpu"), (2.0, "cpu"), (0.0, "cpu")):
            decision = gpu.GpuPolicy("auto", reserve_gb=6, min_gb=3, probe=FreeCard(free)).decide()
            self.assertEqual(decision.mode, want, free)
            self.assertEqual(decision.env["COLI_VULKAN"], "1" if want == "vulkan" else "0")
        self.assertEqual(gpu.GpuPolicy("auto", reserve_gb=0, min_gb=0, probe=FreeCard(0.0)).decide().mode, "vulkan")

    def test_the_reserve_and_the_minimum_are_the_users(self):
        card = FreeCard(12.0)
        self.assertEqual(gpu.GpuPolicy("auto", reserve_gb=10, min_gb=3, probe=card).decide().mode, "cpu")
        self.assertEqual(gpu.GpuPolicy("auto", reserve_gb=6, min_gb=3, probe=card).decide().mode, "vulkan")
        self.assertEqual(gpu.GpuPolicy("auto", reserve_gb=6, min_gb=7, probe=card).decide().mode, "cpu")
        self.assertEqual(gpu.GpuPolicy("auto", reserve_gb=2.5, probe=card).decide().env["G4_VK_RESERVE_GB"], "2.5")

    def test_auto_without_a_reading_says_so_and_changes_nothing(self):
        decision = gpu.GpuPolicy("auto", probe=FreeCard(None)).decide()
        self.assertEqual((decision.mode, decision.env), ("unchanged", {}))
        self.assertIn("nvidia-smi", decision.why)

    def test_a_pinned_model_ignores_the_policy(self):
        low = FreeCard(1.0)
        self.assertEqual(gpu.GpuPolicy("auto", probe=low).decide(override="vulkan").mode, "vulkan")
        self.assertEqual(gpu.GpuPolicy("auto", probe=FreeCard(30.0)).decide(override="cpu").env, {"COLI_VULKAN": "0"})
        self.assertEqual(gpu.GpuPolicy("on", probe=FreeCard(30.0)).decide(override="cpu").mode, "cpu")
        self.assertEqual(gpu.GpuPolicy(None, probe=FreeCard(30.0)).decide(override="vulkan").mode, "vulkan")
        self.assertEqual(gpu.GpuPolicy("off", probe=FreeCard(30.0)).decide(override="auto").mode, "cpu")

    def test_bad_values_are_refused(self):
        for args in (("maybe",), ("auto", -1), ("auto", 6, -1)):
            with self.assertRaises(ValueError):
                gpu.GpuPolicy(*args)

    def test_the_reading_is_cached_for_a_moment(self):
        card = FreeCard(20.0)
        made = gpu.GpuPolicy("auto", probe=card)
        made.decide(); made.decide(); made.decide()
        self.assertEqual(card.reads, 1)

    def test_a_fake_card_for_trying_it_out(self):
        with patch.dict(os.environ, {"COLI_GPU_FAKE_FREE_GB": "11.5", "COLI_GPU_FAKE_TOTAL_GB": "32"}):
            self.assertEqual(gpu.read_memory(), (11.5, 32.0))
        with patch.dict(os.environ, {"COLI_GPU_FAKE_FREE_GB": "lots"}):
            self.assertIsNone(gpu.read_memory())

    def test_nvidia_smi_output_is_read_in_gib(self):
        class Done:
            returncode, stdout = 0, "26613, 32607\n"
        with patch.dict(os.environ, {"COLI_GPU_FAKE_FREE_GB": ""}), patch("gpu.subprocess.run", return_value=Done()):
            free, total = gpu.read_memory()
        self.assertAlmostEqual(free, 26613 / 1024)
        self.assertAlmostEqual(total, 32607 / 1024)
        for broken in ("", "oops", "1, two"):
            class Bad:
                returncode, stdout = 0, broken
            with patch.dict(os.environ, {"COLI_GPU_FAKE_FREE_GB": ""}), patch("gpu.subprocess.run", return_value=Bad()):
                self.assertIsNone(gpu.read_memory(), broken)
        with patch.dict(os.environ, {"COLI_GPU_FAKE_FREE_GB": ""}), patch("gpu.subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(gpu.read_memory())


class SettingsFileTest(unittest.TestCase):
    def path(self):
        folder = tempfile.mkdtemp()
        return os.path.join(folder, "model-settings.json")

    def test_a_missing_file_is_no_settings(self):
        self.assertEqual(model_settings.read(self.path()), {})
        store = model_settings.ModelSettings(self.path())
        self.assertTrue(store.enabled({"id": "m-colibri", "key": "k1"}))
        self.assertIsNone(store.gpu({"id": "m-colibri", "key": "k1"}))

    def test_update_writes_and_cleans_up(self):
        path = self.path()
        model_settings.update(path, "m-colibri", enabled=False)
        model_settings.update(path, "m-colibri", gpu="cpu")
        self.assertEqual(model_settings.read(path), {"m-colibri": {"enabled": False, "gpu": "cpu"}})
        model_settings.update(path, "m-colibri", enabled=True)
        self.assertEqual(model_settings.read(path), {"m-colibri": {"gpu": "cpu"}})
        model_settings.update(path, "m-colibri", gpu="auto")
        self.assertEqual(model_settings.read(path), {})                 # nothing left to say: no record
        self.assertEqual(json.loads(Path(path).read_text(encoding="utf-8")), {"models": {}})

    def test_a_model_is_found_by_its_id_or_its_key(self):
        path = self.path()
        model_settings.update(path, "m-colibri", enabled=False)
        model_settings.update(path, "kabc", gpu="vulkan")
        store = model_settings.ModelSettings(path)
        self.assertFalse(store.enabled({"id": "m-colibri", "key": "zzz"}))
        self.assertEqual(store.gpu({"id": "other", "key": "kabc"}), "vulkan")
        self.assertTrue(store.enabled({"id": "other", "key": "kabc"}))

    def test_the_file_is_read_again_when_it_changes(self):
        path = self.path()
        store = model_settings.ModelSettings(path)
        item = {"id": "m-colibri", "key": "k1"}
        self.assertTrue(store.enabled(item))
        model_settings.update(path, "m-colibri", enabled=False)
        self.assertFalse(store.enabled(item))
        os.remove(path)
        self.assertTrue(store.enabled(item))

    def test_a_broken_file_keeps_the_last_good_settings_and_says_so_once(self):
        path = self.path()
        said = []
        model_settings.update(path, "m-colibri", enabled=False)
        store = model_settings.ModelSettings(path, log=said.append)
        item = {"id": "m-colibri", "key": "k1"}
        self.assertFalse(store.enabled(item))
        time.sleep(0.01)
        Path(path).write_text("{ not json", encoding="utf-8")
        self.assertFalse(store.enabled(item))
        self.assertFalse(store.enabled(item))
        self.assertEqual(len(said), 1)
        self.assertIn("keeping the last good", said[0])

    def test_wrong_values_are_errors_that_say_which(self):
        for text, word in (('{"models": {"m": {"enabled": "no"}}}', "enabled"), ('{"models": {"m": {"gpu": "gpu"}}}', "gpu"),
                           ('{"models": {"m": {"speed": 1}}}', "unknown setting 'speed'"), ('{"models": {"m": 3}}', "expected an object"),
                           ('[1]', "must hold"), ('{"models": []}', "must hold")):
            path = self.path()
            Path(path).write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(model_settings.SettingsError, word, msg=text):
                model_settings.read(path)
        with self.assertRaises(model_settings.SettingsError):
            model_settings.update(self.path(), "m", gpu="gpu")


class RouterGpuTest(unittest.TestCase):
    def test_a_worker_is_started_with_the_decision(self):
        card = FreeCard(20.0)
        rig = CapturingRig(self, [entry("text-a")], gpu=policy("auto", card, reserve_gb=6, min_gb=3))
        self.assertEqual(rig.chat("text-a-colibri")[0], 200)
        self.assertEqual(rig.started, [("text-a-colibri", {"COLI_VULKAN": "1", "COLI_VK_TIER_RESERVE_GB": "6", "G4_VK_RESERVE_GB": "6"})])

    def test_the_choice_is_made_again_at_every_load(self):
        card = FreeCard(20.0)
        rig = CapturingRig(self, [entry("text-a")], gpu=policy("auto", card))
        rig.chat("text-a-colibri")
        rig.call("POST", "/v1/models/unload", {"model": "text-a"})
        card.free = 4.0                                        # a render started: the card is nearly full
        rig.chat("text-a-colibri")
        rig.call("POST", "/v1/models/unload", {"model": "text-a"})
        card.free = 26.0
        rig.chat("text-a-colibri")
        self.assertEqual([env["COLI_VULKAN"] for _id, env in rig.started], ["1", "0", "1"])

    def test_a_loaded_model_keeps_its_choice_and_says_what_it_is(self):
        card = FreeCard(20.0)
        rig = CapturingRig(self, [entry("text-a")], gpu=policy("auto", card), api_key="k")
        auth = {"Authorization": "Bearer k"}
        rig.call("POST", "/v1/chat/completions", {"model": "text-a-colibri", "messages": []}, headers=auth)
        card.free = 1.0
        rig.call("POST", "/v1/chat/completions", {"model": "text-a-colibri", "messages": []}, headers=auth)
        self.assertEqual(len(rig.started), 1)
        _status, _, _, health = rig.call("GET", "/health", headers=auth)
        view = health["router"]["loaded"][0]["gpu"]
        self.assertEqual(view["mode"], "vulkan")
        self.assertIn("20.0 GB free", view["why"])
        self.assertEqual(health["router"]["gpu"]["mode"], "auto")
        self.assertEqual(health["router"]["gpu"]["free_gb"], 1.0)

    def test_no_policy_no_change(self):
        rig = CapturingRig(self, [entry("text-a")])
        rig.chat("text-a-colibri")
        self.assertEqual(rig.started, [("text-a-colibri", {})])

    def test_a_pinned_model_beats_the_policy_and_the_others_follow_it(self):
        path = os.path.join(tempfile.mkdtemp(), "model-settings.json")
        model_settings.update(path, "text-a-colibri", gpu="cpu")
        card = FreeCard(30.0)
        rig = CapturingRig(self, [entry("text-a"), entry("text-b")], gpu=policy("auto", card),
                           settings=model_settings.ModelSettings(path))
        rig.chat("text-a-colibri")
        rig.chat("text-b-colibri")
        self.assertEqual({i: env["COLI_VULKAN"] for i, env in rig.started}, {"text-a-colibri": "0", "text-b-colibri": "1"})

    def test_a_pin_works_without_any_policy(self):
        path = os.path.join(tempfile.mkdtemp(), "model-settings.json")
        model_settings.update(path, "text-a-colibri", gpu="vulkan")
        rig = CapturingRig(self, [entry("text-a"), entry("text-b")], settings=model_settings.ModelSettings(path))
        rig.chat("text-a-colibri")
        rig.chat("text-b-colibri")
        self.assertEqual(rig.started[0][1]["COLI_VULKAN"], "1")
        self.assertEqual(rig.started[1][1], {})


class DisabledModelTest(unittest.TestCase):
    def rig(self, **kw):
        self.path = os.path.join(tempfile.mkdtemp(), "model-settings.json")
        self.settings = model_settings.ModelSettings(self.path)
        return CapturingRig(self, [entry("text-a"), entry("text-b")], settings=self.settings, **kw)

    def test_a_model_that_is_off_is_refused_with_the_way_back(self):
        rig = self.rig()
        model_settings.update(self.path, "text-a-colibri", enabled=False)
        status, _, _, body = rig.chat("text-a-colibri")
        self.assertEqual((status, body["error"]["code"]), (403, "model_disabled"))
        self.assertIn("coli model enable text-a-colibri", body["error"]["message"])
        self.assertEqual(rig.started, [])                                   # nothing was loaded for it
        self.assertEqual(rig.chat("text-b-colibri")[0], 200)                 # the others are not affected

    def test_switching_to_it_and_preloading_it_are_refused_too(self):
        rig = self.rig()
        model_settings.update(self.path, "text-a-colibri", enabled=False)
        status, _, _, body = rig.call("POST", "/v1/models/switch", {"model": "text-a"})
        self.assertEqual((status, body["error"]["code"]), (403, "model_disabled"))
        with self.assertRaises(router.ApiError):
            rig.pool.acquire(rig.entries[0])

    def test_the_list_marks_it_and_it_comes_back_when_turned_on(self):
        rig = self.rig()
        model_settings.update(self.path, "text-a-colibri", enabled=False)
        _s, _, _, models = rig.call("GET", "/v1/models")
        states = {m["id"]: m["state"] for m in models["data"]}
        self.assertEqual(states, {"text-a-colibri": "disabled", "text-b-colibri": "idle"})
        _s, _, _, installed = rig.call("GET", "/v1/models/installed")
        row = {m["id"]: m for m in installed["data"]}["text-a-colibri"]
        self.assertFalse(row["enabled"])
        self.assertFalse(row["switchable"])
        model_settings.update(self.path, "text-a-colibri", enabled=True)
        self.assertEqual(rig.chat("text-a-colibri")[0], 200)

    def test_the_api_turns_a_loaded_model_off_and_unloads_it(self):
        rig = self.rig()
        rig.chat("text-a-colibri")
        worker = rig.pool.get(rig.entries[0]["key"])
        status, _, _, body = rig.call("POST", "/v1/models/settings", {"model": "text-a", "enabled": False})
        self.assertEqual((status, body["enabled"], body["gpu"], body["unload"]), (200, False, "auto", "unloaded"))
        self.assertFalse(worker.alive())
        self.assertEqual(model_settings.read(self.path), {"text-a-colibri": {"enabled": False}})
        self.assertEqual(rig.chat("text-a-colibri")[0], 403)
        status, _, _, body = rig.call("POST", "/v1/models/settings", {"model": "text-a", "enabled": True, "gpu": "cpu"})
        self.assertEqual((status, body["enabled"], body["gpu"]), (200, True, "cpu"))
        self.assertEqual(rig.chat("text-a-colibri")[0], 200)

    def test_a_model_that_is_answering_is_not_cut_off_when_turned_off(self):
        import threading
        rig = self.rig()
        rig.chat("text-a-colibri")
        answer = {}
        thread = threading.Thread(target=lambda: answer.update(r=rig.chat("text-a-colibri", sleep=1.5)))
        thread.start()
        time.sleep(0.5)
        status, _, _, body = rig.call("POST", "/v1/models/settings", {"model": "text-a", "enabled": False})
        self.assertEqual((status, body["unload"]), (200, "unloading when idle"))
        thread.join()
        self.assertEqual(answer["r"][0], 200)
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and rig.states()["text-a-colibri"] != "idle":
            time.sleep(0.2)
        self.assertEqual(rig.states()["text-a-colibri"], "idle", "it must go once it has finished")

    def test_presets_of_a_model_that_is_off_are_not_offered(self):
        rig = self.rig()
        folder = tempfile.mkdtemp()
        file = os.path.join(folder, "presets.json")
        Path(file).write_text(json.dumps({"presets": {"terse": {"model": "text-a", "defaults": {"temperature": 0.2}},
                                                         "brief": {"model": "text-b", "defaults": {"temperature": 0.2}}}}), encoding="utf-8")
        rig.server.presets = presets_mod.Presets(file)
        _s, _, _, models = rig.call("GET", "/v1/models")
        self.assertEqual(sorted(m["id"] for m in models["data"]), ["brief", "terse", "text-a-colibri", "text-b-colibri"])
        model_settings.update(self.path, "text-a-colibri", enabled=False)
        _s, _, _, models = rig.call("GET", "/v1/models")
        self.assertEqual(sorted(m["id"] for m in models["data"]), ["brief", "text-a-colibri", "text-b-colibri"])

    def test_the_settings_list_says_what_is_set_and_what_runs_where(self):
        card = FreeCard(20.0)
        rig = self.rig(gpu=policy("auto", card))
        rig.chat("text-a-colibri")
        model_settings.update(self.path, "text-b-colibri", enabled=False, gpu="cpu")
        _s, _, _, body = rig.call("GET", "/v1/models/settings")
        rows = {m["id"]: m for m in body["data"]}
        self.assertEqual((rows["text-a-colibri"]["enabled"], rows["text-a-colibri"]["gpu"], rows["text-a-colibri"]["state"]),
                         (True, "auto", "ready"))
        self.assertEqual(rows["text-a-colibri"]["running"]["mode"], "vulkan")
        self.assertEqual((rows["text-b-colibri"]["enabled"], rows["text-b-colibri"]["gpu"]), (False, "cpu"))
        self.assertEqual(body["gpu_policy"]["mode"], "auto")
        self.assertEqual(body["file"], self.path)

    def test_the_api_refuses_what_it_cannot_do(self):
        rig = self.rig()
        for body, code in (({"model": "text-a"}, 400), ({"model": "text-a", "enabled": "no"}, 400),
                           ({"model": "text-a", "gpu": "fast"}, 400), ({"model": "nope", "enabled": False}, 404),
                           ({"enabled": False}, 400)):
            self.assertEqual(rig.call("POST", "/v1/models/settings", body)[0], code, body)
        self.assertEqual(rig.call("POST", "/v1/models/settings", raw=b"{}", headers={"Content-Type": "text/plain"})[0], 415)
        status, _, _, body = rig.call("POST", "/v1/models/settings", {"model": "text-a", "enabled": False},
                                      headers={"Origin": "http://evil.example"})
        self.assertEqual((status, body["error"]["code"]), (403, "forbidden"))
        self.assertEqual(model_settings.read(self.path), {})                    # nothing refused changed the file

    def test_the_api_needs_the_key(self):
        path = os.path.join(tempfile.mkdtemp(), "model-settings.json")
        rig = CapturingRig(self, [entry("text-a")], settings=model_settings.ModelSettings(path), api_key="sekrit")
        self.assertEqual(rig.call("POST", "/v1/models/settings", {"model": "text-a", "enabled": False})[0], 401)
        self.assertEqual(rig.call("GET", "/v1/models/settings")[0], 401)
        self.assertEqual(rig.call("POST", "/v1/models/settings", {"model": "text-a", "enabled": False},
                                  headers={"Authorization": "Bearer sekrit"})[0], 200)


if __name__ == "__main__":
    unittest.main()
