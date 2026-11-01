"""tools/acestep_ref.py: the case table and the guard that a capture leaves its model directory alone.

The capture itself needs upstream's code and a checkpoint (see the tool's docstring); what is checked here is the part
that decides what it captures and whether it can be trusted, which needs torch only to import the module.
"""
import importlib.util
import os
import sys
import tempfile
import unittest

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")


def load():
    if importlib.util.find_spec("torch") is None:
        raise unittest.SkipTest("needs torch (the reference venv, tools/requirements-acestep-ref.txt)")
    sys.path.insert(0, TOOLS)
    try:
        import acestep_ref
    finally:
        sys.path.remove(TOOLS)
    return acestep_ref


class Cases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = load()

    def test_dcw_is_always_explicit_and_the_suffix_says_which(self):
        """Upstream turns DCW on for turbo when nothing is said; a case that leaves it open would not pin a path."""
        for name, params in self.ref.CASES.items():
            with self.subTest(case=name):
                self.assertIsInstance(params["dcw_enabled"], bool)
                self.assertEqual(name.endswith("_nodcw"), not params["dcw_enabled"])

    def test_every_nodcw_case_has_a_dcw_twin_that_differs_only_in_dcw(self):
        """The pair is how the sampler is isolated from the correction: nothing else may differ."""
        cases = self.ref.CASES
        for name, params in cases.items():
            if name.endswith("_nodcw"):
                twin = cases[name[: -len("_nodcw")]]
                self.assertEqual({k: v for k, v in params.items() if k != "dcw_enabled"},
                                 {k: v for k, v in twin.items() if k != "dcw_enabled"}, name)

    def test_the_seed_is_fixed_so_a_capture_can_be_repeated(self):
        for name, params in self.ref.CASES.items():
            self.assertFalse(params["use_random_seed"], name)
            self.assertEqual(params["seed"], 42, name)

    def test_the_cases_cover_what_the_engine_must_get_right(self):
        c = self.ref.CASES
        self.assertEqual(c["inst10"]["audio_duration"], 10.0)
        self.assertEqual((c["odd9"]["audio_duration"] * 25) % 2, 1.0)            # an odd number of latent frames
        self.assertIn(c["shift3"]["shift"], (3.0,))                              # a table schedule other than shift 1
        self.assertNotIn(c["steps5"]["inference_steps"], (8,))                   # a linspace schedule
        self.assertGreater(c["long30"]["audio_duration"] * 25, 256 + 64)         # more than one tiled VAE chunk
        self.assertTrue(c["vocal10"]["lyrics"].count("\n") >= 3)                 # real lyrics, not the instrumental tag
        self.assertNotEqual(c["vocal10"]["vocal_language"], c["inst10"]["vocal_language"])


class Fingerprint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = load()

    def test_a_changed_added_or_removed_file_shows_in_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "vae"))
            for rel, text in (("a.txt", "one"), (os.path.join("vae", "config.json"), "{}")):
                with open(os.path.join(d, rel), "w") as f:
                    f.write(text)
            before = self.ref.fingerprint(d)
            self.assertEqual(before, self.ref.fingerprint(d))                    # reading changes nothing
            with open(os.path.join(d, "a.txt"), "w") as f:
                f.write("longer than before")                                    # size differs
            with open(os.path.join(d, "new.bin"), "w") as f:
                f.write("x")
            os.remove(os.path.join(d, "vae", "config.json"))
            after = self.ref.fingerprint(d)
            changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
            self.assertEqual(changed, sorted(["a.txt", "new.bin", os.path.join("vae", "config.json")]))


if __name__ == "__main__":
    unittest.main()
