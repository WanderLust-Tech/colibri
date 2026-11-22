"""A context window of a model's own (docs/model-settings.md): the settings file, the router's worker environment, the endpoint, `coli model context`."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import model_settings as ms  # noqa: E402
import router  # noqa: E402
from test_router import Rig, entry  # noqa: E402


def real(name, family, modality="text"):
    """A router entry of a family the registry knows."""
    item = entry(name, modality)
    item["family"] = family
    return item


class FileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "model-settings.json")

    def test_a_context_is_stored_replaced_and_cleared(self):
        self.assertEqual(ms.update(self.path, "m", context=32768), {"context": 32768})
        self.assertEqual(ms.read(self.path), {"m": {"context": 32768}})
        self.assertEqual(ms.update(self.path, "m", context=65536, gpu="cpu"), {"context": 65536, "gpu": "cpu"})
        self.assertEqual(ms.update(self.path, "m", context=0), {"gpu": "cpu"})
        self.assertEqual(ms.update(self.path, "m", gpu="auto"), {})
        self.assertEqual(ms.read(self.path), {})                               # an empty record is removed

    def test_the_other_settings_are_left_alone(self):
        ms.update(self.path, "m", enabled=False, gpu="vulkan")
        self.assertEqual(ms.update(self.path, "m", context=8192), {"enabled": False, "gpu": "vulkan", "context": 8192})

    def test_bad_values_are_refused_and_name_the_model(self):
        for bad in (100, -5, 1.5, "big", True):
            with self.assertRaises(ms.SettingsError, msg=repr(bad)):
                ms.update(self.path, "m", context=bad)
        Path(self.path).write_text(json.dumps({"models": {"m": {"context": "lots"}}}), encoding="utf-8")
        with self.assertRaises(ms.SettingsError) as caught:
            ms.read(self.path)
        self.assertIn("'m'", str(caught.exception))
        self.assertIn("context", str(caught.exception))

    def test_the_accessor_gives_a_number_or_none(self):
        settings = ms.ModelSettings(self.path)
        item = {"id": "m-colibri", "key": "km"}
        self.assertIsNone(settings.context(item))
        ms.update(self.path, "m-colibri", context=16384)
        self.assertEqual(settings.context(item), 16384)
        ms.update(self.path, "km", context=4096)                                # by key too; the id's entry wins
        self.assertEqual(settings.context(item), 16384)


class WorkerEnvironmentTest(unittest.TestCase):
    def rig(self, *entries):
        rig = Rig(self, list(entries))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "model-settings.json")
        settings = ms.ModelSettings(self.path)
        rig.pool.settings = settings
        return rig

    def env_of(self, rig, item):
        return rig.pool._command(item)[1]

    def test_a_models_window_goes_to_its_worker_in_the_variable_its_family_reads(self):
        qwen, gemma = real("qwen36", "qwen36"), real("gemma", "gemma4")
        rig = self.rig(qwen, gemma)
        ms.update(self.path, qwen["id"], context=32768)
        ms.update(self.path, gemma["id"], context=16384)
        self.assertEqual(self.env_of(rig, qwen).get("Q36_MAXT"), "32768")
        self.assertEqual(self.env_of(rig, gemma).get("CTX"), "16384")

    def test_it_wins_over_the_routers_own_environment(self):
        gemma = real("gemma", "gemma4")
        rig = self.rig(gemma)
        old = os.environ.get("CTX")
        os.environ["CTX"] = "65536"
        self.addCleanup(lambda: os.environ.pop("CTX", None) if old is None else os.environ.__setitem__("CTX", old))
        self.assertEqual(self.env_of(rig, gemma).get("CTX"), "65536")
        ms.update(self.path, gemma["id"], context=8192)
        self.assertEqual(self.env_of(rig, gemma).get("CTX"), "8192")
        ms.update(self.path, gemma["id"], context=0)
        self.assertEqual(self.env_of(rig, gemma).get("CTX"), "65536")

    def test_a_window_over_the_familys_limit_is_cut_to_it(self):
        gemma = real("gemma", "gemma4")
        rig = self.rig(gemma)
        ms.update(self.path, gemma["id"], context=10_000_000)
        self.assertEqual(self.env_of(rig, gemma).get("CTX"), str(router.context_limits(gemma)[1]))

    def test_a_model_without_a_window_or_of_an_unknown_family_ignores_it(self):
        speech, unknown = real("tts", "qwen3_tts", "speech"), entry("odd")
        rig = self.rig(speech, unknown)
        for item in (speech, unknown):
            ms.update(self.path, item["id"], context=4096)
            env = self.env_of(rig, item)
            self.assertNotEqual(env.get("CTX"), "4096")
            self.assertNotEqual(env.get("Q36_MAXT"), "4096")
        self.assertIsNone(router.context_limits(speech))
        self.assertIsNone(router.context_limits(unknown))


class EndpointTest(unittest.TestCase):
    def setUp(self):
        self.qwen, self.tts = real("qwen36", "qwen36"), real("tts", "qwen3_tts", "speech")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "model-settings.json")
        self.rig = Rig(self, [self.qwen, self.tts], settings=ms.ModelSettings(self.path))

    def set(self, model, **body):
        return self.rig.call("POST", "/v1/models/settings", dict({"model": model}, **body))

    def test_a_window_is_set_listed_and_cleared(self):
        status, _, _, body = self.set(self.qwen["id"], context=32768)
        self.assertEqual((status, body["context"]), (200, 32768))
        listing = self.rig.call("GET", "/v1/models/settings")[3]["data"]
        row = next(r for r in listing if r["id"] == self.qwen["id"])
        self.assertEqual((row["context"], row["context_max"] > 32768), (32768, True))
        self.assertEqual(ms.read(self.path), {self.qwen["id"]: {"context": 32768}})
        status, _, _, body = self.set(self.qwen["id"], context="auto")
        self.assertEqual((status, body["context"]), (200, None))
        self.assertEqual(ms.read(self.path), {})

    def test_a_bad_window_is_a_400_that_names_the_field(self):
        for value, fragment in ((100, "at least"), (10 ** 9, "at most"), ("lots", "number"), (True, "number"), (1.5, "number")):
            status, _, _, body = self.set(self.qwen["id"], context=value)
            self.assertEqual((status, body["error"]["param"]), (400, "context"), value)
            self.assertIn(fragment, body["error"]["message"])

    def test_a_model_with_no_window_refuses_one_and_says_so(self):
        status, _, _, body = self.set(self.tts["id"], context=4096)
        self.assertEqual((status, body["error"]["param"]), (400, "context"))
        self.assertIn("no context window", body["error"]["message"])
        self.assertEqual(self.set(self.tts["id"], context="auto")[0], 200)                 # clearing nothing is fine

    def test_the_listing_says_a_model_has_no_window(self):
        listing = self.rig.call("GET", "/v1/models/settings")[3]["data"]
        self.assertEqual(next(r for r in listing if r["id"] == self.tts["id"])["context_max"], 0)

    def test_changing_the_window_of_a_loaded_model_unloads_it(self):
        self.rig.chat(self.qwen["id"])
        self.assertEqual(self.rig.states()[self.qwen["id"]], "ready")
        status, _, _, body = self.set(self.qwen["id"], context=16384)
        self.assertEqual((status, body.get("unload")), (200, "unloaded"))
        self.assertEqual(self.rig.states()[self.qwen["id"]], "idle")
        status, _, _, body = self.set(self.qwen["id"], context=16384)                       # the same again: nothing to restart
        self.assertEqual((status, body.get("unload")), (200, None))

    def test_the_same_request_still_needs_something_to_change(self):
        status, _, _, body = self.rig.call("POST", "/v1/models/settings", {"model": self.qwen["id"]})
        self.assertEqual((status, body["error"]["param"]), (400, "enabled"))


class CommandTest(unittest.TestCase):
    """`coli model context`: what is typed, and the file it edits when no server answers."""

    @classmethod
    def setUpClass(cls):
        import importlib.machinery
        import importlib.util
        loader = importlib.machinery.SourceFileLoader("coli_script_ctx", str(HERE.parent / "coli"))
        spec = importlib.util.spec_from_loader("coli_script_ctx", loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def test_what_is_typed_becomes_tokens(self):
        for typed, tokens in (("32768", 32768), ("32k", 32768), ("128K", 131072), ("1.5k", 1536), ("auto", 0), ("default", 0), ("0", 0)):
            self.assertEqual(self.coli._context_tokens(typed), tokens, typed)
        for typed in ("", None, "lots", "k", "12x"):
            self.assertIsNone(self.coli._context_tokens(typed), repr(typed))


if __name__ == "__main__":
    unittest.main()
