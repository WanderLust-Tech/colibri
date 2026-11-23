"""The planner loads a checkpoint whose tensors are named `model.layers.N...` (the 0.6B's and the 4B's, saved as a Qwen3ForCausalLM) as well as the 1.7B's `layers.N...`.

    ACESTEP_TINY_C=/path/to/acestep_tiny_c ACESTEP_TINY_LM_REF=/path/to/lm_ref python -m unittest tests.test_acestep_lm_prefix

(a tiny fixture made with `tools/make_acestep_tiny.py --audio-in --base --lm`, and the planner's captures of it, `tools/acestep_ref_lm.py`.) The tiny planner
is saved the 1.7B's way. The test copies the fixture, renames every tensor of the planner's file with the prefix by rewriting the safetensors header (the data
and its offsets stay as they are), and holds the engine's planner to the same captures through the copy: the same codes and logits as through the original.
The real 0.6B and 4B planners were held to upstream's captures the same way, 2026-11-23 (docs/acestep.md).
"""
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.environ.get("ACESTEP_EXE") or os.path.join(HERE, "..", "acestep.exe" if os.name == "nt" else "acestep")
TINY = os.environ.get("ACESTEP_TINY_C")
REF = os.environ.get("ACESTEP_TINY_LM_REF")
LM = "acestep-5Hz-lm-1.7B"


def add_prefix(path, prefix):
    """Rename every tensor of a safetensors file: only the JSON header changes (offsets count from the end of it)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    renamed = {(k if k == "__metadata__" else prefix + k): v for k, v in header.items()}
    text = json.dumps(renamed, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + data)


def oracle(model, case):
    done = subprocess.run([EXE, "--model", model, "--lm-ref", os.path.join(REF, case)], capture_output=True, timeout=600,
                          env={k: v for k, v in os.environ.items() if k != "COLI_VULKAN"})
    return done.returncode, (done.stdout + done.stderr).decode("utf-8", "replace")


@unittest.skipUnless(TINY and os.path.isdir(TINY) and REF and os.path.isdir(REF) and os.path.isfile(EXE),
                     "needs ACESTEP_TINY_C, ACESTEP_TINY_LM_REF and a built acestep (ACESTEP_EXE)")
class Prefix(unittest.TestCase):
    CASE = "lm10"

    def test_the_planner_of_a_prefixed_checkpoint_makes_the_same_codes(self):
        code, plain = oracle(TINY, self.CASE)
        self.assertEqual(code, 0, plain[-600:])
        self.assertIn("identical", plain)
        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "m")
            shutil.copytree(TINY, model)
            add_prefix(os.path.join(model, LM, "model.safetensors"), "model.")
            code, prefixed = oracle(model, self.CASE)
            self.assertEqual(code, 0, prefixed[-600:])
            self.assertIn("identical", prefixed)
            self.assertIn("within their limits", prefixed)

    def test_a_checkpoint_with_neither_naming_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "m")
            shutil.copytree(TINY, model)
            add_prefix(os.path.join(model, LM, "model.safetensors"), "decoder.")
            code, out = oracle(model, self.CASE)
            self.assertNotEqual(code, 0)
            self.assertIn("missing tensor", out)


if __name__ == "__main__":
    unittest.main()
