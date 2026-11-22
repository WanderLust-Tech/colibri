"""tools/mutate_acestep_meta.py: every mutant still applies to the header it names.

The mutation run proves the first phase's oracles can fail (YAML emitter, token tables, the decoder's machine, the parse of the model's text), and it
compiles and runs the engine on a checkpoint whose planner has learned the metadata blocks (see its docstring). What is checked here is cheap and
needs neither: that every mutant's text still occurs exactly as often as the mutant says, so a refactor of acestep_meta.h cannot quietly turn a
mutant into a no-op that the oracle then "survives" by not having changed anything, or into a build error that it "catches" for the wrong reason.
"""
import os
import sys
import unittest

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(C_DIR, "tools"))
import mutate_acestep_meta as mut  # noqa: E402


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {}
        for header, *_ in mut.MUTANTS.values():
            if header not in cls.sources:
                with open(os.path.join(C_DIR, header), encoding="utf-8", newline="") as f:
                    cls.sources[header] = f.read()

    def test_every_mutant_text_occurs_as_often_as_it_says(self):
        for name, (header, old, new, *count) in mut.MUTANTS.items():
            with self.subTest(mutant=name):
                self.assertEqual(self.sources[header].count(old), count[0] if count else 1, f"{old!r} in {header}")

    def test_every_mutant_changes_something(self):
        for name, (header, old, new, *count) in mut.MUTANTS.items():
            with self.subTest(mutant=name):
                self.assertNotEqual(old, new)

    def test_the_mutants_are_distinct_edits(self):
        edits = [(m[0], m[1], m[2]) for m in mut.MUTANTS.values()]
        self.assertEqual(len(edits), len(set(edits)))

    def test_the_families_the_oracle_must_cover_are_all_present(self):
        names = " ".join(mut.MUTANTS)
        for family in ("yaml-width", "yaml-single-quote", "yaml-double", "yaml-resolver", "yaml-int", "yaml-float", "yaml-timestamp", "yaml-colon", "parse-",
                       "tree-bpm", "tree-duration", "tree-keyscale", "tree-language", "tab-fixed", "tab-caption", "fsm-language",
                       "fsm-caption-line", "fsm-caption-limit", "fsm-field-name", "fsm-skipped", "fsm-injected", "fsm-end-token", "meta-prompt", "meta-stops"):
            self.assertIn(family, names)


if __name__ == "__main__":
    unittest.main()
