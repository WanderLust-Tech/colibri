"""Constrained decoding (c/constrain.h): what it makes is JSON, judged by Python's parser, not by the engine's own grammar walker.

tests/test_constrain.c holds the mechanism to its grammar (masking, rejection, the fallback scan, the end token). Here the same
program makes hundreds of generations from random logits, which left alone would be noise, and every answer that ended by itself
must be accepted by json.loads, and by the schema's shape when a schema was given.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
EXE = ".exe" if os.name == "nt" else ""


def compiler():
    return os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")


@unittest.skipUnless(compiler(), "needs a C compiler")
class ConstrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.exe = os.path.join(cls.tmp.name, "test_constrain" + EXE)
        build = subprocess.run([compiler(), "-O2", "-Wall", "-Wextra", "-Wno-unused-function", "-Wno-unused-parameter",
                                str(C_DIR / "tests" / "test_constrain.c"), "-o", cls.exe, "-lm"],
                               capture_output=True, text=True, cwd=str(C_DIR))
        if build.returncode:
            raise RuntimeError("test_constrain.c did not build:\n" + build.stdout + build.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_exe(self, *args):
        result = subprocess.run([self.exe, *args], capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def generations(self, which, count, seed):
        out = []
        for line in self.run_exe("gen", which, str(count), str(seed)).splitlines():
            kind, hexed = line.split(" ", 1)
            out.append((kind == "C", bytes.fromhex(hexed)))
        return out

    def test_the_mechanism_checks_pass(self):
        self.assertIn("constrain: ok", self.run_exe())

    def test_random_logits_under_the_json_grammar_make_json(self):
        done = 0
        for seed in (1, 2, 3):
            for ended, data in self.generations("generic", 150, seed):
                if ended:
                    json.loads(data.decode("utf-8"))             # raises when it is not JSON
                    done += 1
        self.assertGreater(done, 200, "too few answers ended by themselves for the check to mean anything")

    def test_random_logits_under_a_schema_make_that_shape(self):
        done = 0
        for seed in (4, 5):
            for ended, data in self.generations("schema", 150, seed):
                if not ended:
                    continue
                value = json.loads(data.decode("utf-8"))
                self.assertEqual(list(value), ["name", "n", "tags", "ok"])
                self.assertIsInstance(value["name"], str)
                self.assertTrue(isinstance(value["n"], int) and not isinstance(value["n"], bool))
                self.assertIsInstance(value["tags"], list)
                self.assertTrue(all(isinstance(tag, str) for tag in value["tags"]))
                self.assertIsInstance(value["ok"], bool)
                done += 1
        self.assertGreater(done, 40, "too few schema answers ended by themselves")


if __name__ == "__main__":
    unittest.main()
