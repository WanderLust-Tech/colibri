"""tools/mutate_acestep_vae.py: every mutant still applies to acestep_vae.h.

The mutation run proves the VAE oracle can fail, and it compiles and runs the engine (see its docstring). What is
checked here is cheap and needs neither: that every mutant's text still occurs exactly once in the header, so a
refactor of acestep_vae.h cannot quietly turn a mutant into a no-op that the oracle then "catches" by the build
failing, or "survives" by not having changed anything.
"""
import os
import sys
import unittest

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(C_DIR, "tools"))
import mutate_acestep_vae as mut  # noqa: E402


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(C_DIR, "acestep_vae.h"), encoding="utf-8", newline="") as f:
            cls.source = f.read()

    def test_every_mutant_text_occurs_exactly_once(self):
        for name, (old, new) in mut.MUTANTS.items():
            with self.subTest(mutant=name):
                self.assertEqual(self.source.count(old), 1, f"{old!r} in acestep_vae.h")

    def test_every_mutant_changes_something(self):
        for name, (old, new) in mut.MUTANTS.items():
            with self.subTest(mutant=name):
                self.assertNotEqual(old, new)

    def test_the_mutants_are_distinct_edits(self):
        edits = [(old, new) for old, new in mut.MUTANTS.values()]
        self.assertEqual(len(edits), len(set(edits)))

    def test_the_families_the_oracle_must_cover_are_all_present(self):
        names = " ".join(mut.MUTANTS)
        for family in ("dilation", "convt", "im2col", "weight-norm", "snake", "residual", "tile", "peak-limit"):
            self.assertIn(family, names)


if __name__ == "__main__":
    unittest.main()
