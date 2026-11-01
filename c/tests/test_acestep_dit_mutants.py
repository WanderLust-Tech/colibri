"""tools/mutate_acestep_dit.py: every mutant still applies to acestep_dit.h.

The mutation run proves the DiT oracle can fail, and it compiles and runs the engine (see its docstring). What is checked
here is cheap and needs neither: that every mutant's text still occurs exactly as many times as the mutant says, so a
refactor of acestep_dit.h cannot quietly turn a mutant into a no-op that the oracle then "survives" by not having changed
anything, or into a build error that it "catches" for the wrong reason.
"""
import os
import sys
import unittest

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(C_DIR, "tools"))
import mutate_acestep_dit as mut  # noqa: E402


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(C_DIR, "acestep_dit.h"), encoding="utf-8", newline="") as f:
            cls.source = f.read()

    def test_every_mutant_text_occurs_as_often_as_it_says(self):
        for name, edit in mut.MUTANTS.items():
            old, count = edit[0], (edit[2] if len(edit) == 3 else 1)
            with self.subTest(mutant=name):
                self.assertEqual(self.source.count(old), count, f"{old!r} in acestep_dit.h")

    def test_every_mutant_changes_something(self):
        for name, edit in mut.MUTANTS.items():
            with self.subTest(mutant=name):
                self.assertNotEqual(edit[0], edit[1])

    def test_the_mutants_are_distinct_edits(self):
        edits = [(e[0], e[1]) for e in mut.MUTANTS.values()]
        self.assertEqual(len(edits), len(set(edits)))

    def test_the_families_the_oracle_must_cover_are_all_present(self):
        names = " ".join(mut.MUTANTS)
        for family in ("timbre", "window", "rope", "norm", "kv-head", "adaln", "gate", "swiglu", "embedding", "patch", "euler",
                       "schedule", "dcw", "output"):
            self.assertIn(family, names)

    def test_the_cases_the_docstring_asks_for_are_the_ones_that_matter(self):
        """Documented, not assumed: the odd-tail DCW mutant is the one that needs an odd-length case."""
        self.assertIn("dcw-odd-tail", mut.MUTANTS)
        self.assertIn("odd9", mut.__doc__)
        self.assertIn("long30", mut.__doc__)


if __name__ == "__main__":
    unittest.main()
