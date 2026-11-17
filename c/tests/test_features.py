"""The `features` block of /health: its names are the ones docs/api.md lists, and a router turns on what a server lacks."""
import sys
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))

import features  # noqa: E402

API_MD = C_DIR.parent / "docs" / "api.md"


class FeaturesTest(unittest.TestCase):
    def test_a_single_server_has_none_of_the_router_features(self):
        found = features.describe()
        self.assertEqual(found["api"], features.API_LEVEL)
        self.assertIs(found["router"], False)
        self.assertEqual(found["model_switch"], "restart")
        for name in ("model_unload", "model_settings", "presets", "request_log", "server_log", "priority"):
            self.assertIs(found[name], False, name)

    def test_a_router_has_them_and_the_files_it_was_started_with_decide_the_rest(self):
        bare = features.describe(router=True)
        self.assertEqual(bare["model_switch"], "load")
        for name in ("model_unload", "server_log", "priority"):
            self.assertIs(bare[name], True, name)
        for name in ("model_settings", "presets", "request_log"):
            self.assertIs(bare[name], False, name)
        full = features.describe(router=True, presets=True, request_log=True, model_settings=True)
        for name in ("model_settings", "presets", "request_log"):
            self.assertIs(full[name], True, name)

    def test_only_a_router_can_turn_a_router_file_feature_on(self):
        found = features.describe(presets=True, request_log=True, model_settings=True)
        for name in ("model_settings", "presets", "request_log"):
            self.assertIs(found[name], False, name)

    def test_every_name_is_written_up_in_the_api_docs(self):
        text = API_MD.read_text(encoding="utf-8")
        missing = [name for name in features.FEATURES if f"`{name}`" not in text]
        self.assertEqual(missing, [], "feature names docs/api.md does not describe")


if __name__ == "__main__":
    unittest.main()
