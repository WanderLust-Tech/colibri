"""Gemma 4's decode step as a dense chain on the Vulkan device (gemma4_chain.h) against the CPU's.

    python -m unittest tests.test_gemma4_chain        (from c/)

Needs a gemma4 built with VK=1 and a Vulkan device; skipped without them. The models are the tiny random Gemma 4s of
test_gemma4 (tools/make_gemma4_tiny.py): the mixture of experts, the dense one and the one with per-layer embeddings, each
with sliding layers over a ring of 6 rows (the run below passes it several times) and full layers.

What is held, for each of the three:
  - the chain runs (the engine says "dense chain on") and its greedy continuation is the CPU's, the last logits within
    float rounding of it (the device adds in another order; the bits are not the CPU's);
  - COLI_VK_CHAIN_CHECK=1, which runs every chained step on the CPU path as well, reports the residual of each step within
    float rounding of the CPU's;
  - a device lost in the middle of a decode step (COLI_VK_CHAIN_FAULT=n fakes it at the n-th frame) is not seen in the answer:
    the host's caches hold every position before the step, and the CPU carries on with the same continuation;
  - COLI_VK_CHAIN=0 keeps the per-matrix card path, with no chain.
"""
import json
import os
import re
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_gemma4 as t4  # noqa: E402

IDS = [1, 5, 9, 12, 3, 7, 2, 8, 4, 6, 10, 11, 13, 14]
GREEDY = 24                      # past the ring of 6 rows several times, and past a full layer's first rows


def run(snap, mode, extra=None, greedy=GREEDY):
    """The engine's test mode on a prompt of token ids; mode: cpu, permatrix (the card without the chain), chain.
    Returns (the JSON it prints, its stderr)."""
    env = dict(os.environ, SNAP=snap, G4_TEST="1", G4_IDS=",".join(map(str, IDS)), G4_GREEDY=str(greedy),
               OMP_NUM_THREADS="2", G4_SLACK="0", G4_IMG_TOKENS="1",
               COLI_VK_GRP_COOP="0")   # a prompt chunk's grouped expert GEMMs in floats: the parity checks below are tight; the matrix units' f16 form has its own test
    for name in ("COLI_VULKAN", "COLI_VK_CHAIN", "COLI_VK_CHAIN_CHECK", "COLI_VK_CHAIN_FAULT", "G4_VK_DENSE", "G4_VK_CHUNK"):
        env.pop(name, None)
    if mode == "permatrix":
        env.update(COLI_VULKAN="1", COLI_VK_CHAIN="0")
    elif mode == "chain":
        env.update(COLI_VULKAN="1")
    env.update(extra or {})
    done = subprocess.run([t4.EXE, "128"], env=env, capture_output=True, text=True, timeout=300, cwd=os.path.join(HERE, ".."))
    if done.returncode:
        raise AssertionError(done.stdout[-400:] + done.stderr[-1500:])
    return json.loads(done.stdout.strip().splitlines()[-1]), done.stderr


def rel_diff(a, b):
    return max(abs(x - y) / max(1.0, abs(x), abs(y)) for x, y in zip(a, b))


def have_chain():
    """A VK build with a device that runs the chain: its stderr says so."""
    if not t4.HAVE_VARIANTS:
        return False
    try:
        _, err = run(t4.VARIANT_DIRS["dense"], "chain", greedy=2)
    except (AssertionError, OSError, subprocess.SubprocessError):
        return False
    return "dense chain on" in err


HAVE_CHAIN = have_chain()
SNAPS = {"moe": t4.TINY, "dense": t4.VARIANT_DIRS["dense"], "ple": t4.VARIANT_DIRS["ple"]}


@unittest.skipUnless(HAVE_CHAIN, "needs gemma4 built with VK=1 and a Vulkan device")
class ChainTest(unittest.TestCase):
    cpu = {}

    @classmethod
    def reference(cls, variant):
        if variant not in cls.cpu:
            cls.cpu[variant] = run(SNAPS[variant], "cpu")[0]
        return cls.cpu[variant]

    def test_the_continuation_is_the_cpus(self):
        for variant, snap in SNAPS.items():
            with self.subTest(variant):
                cpu = self.reference(variant)
                out, err = run(snap, "chain")
                self.assertIn("gemma4 chain:", err)                       # it really ran
                self.assertIn("decode steps", err)
                self.assertEqual(out["greedy"], cpu["greedy"])
                self.assertLess(rel_diff(cpu["last"], out["last"]), 2e-3)

    def test_every_step_matches_the_cpus_residual(self):
        for variant, snap in SNAPS.items():
            with self.subTest(variant):
                _, err = run(snap, "chain", {"COLI_VK_CHAIN_CHECK": "1"})
                found = re.search(r"chain check: (\d+) steps compared with the CPU's, the worst residual difference ([0-9.e+-]+)", err)
                self.assertTrue(found, err[-800:])
                self.assertGreaterEqual(int(found.group(1)), GREEDY)       # every decode step was compared
                self.assertLess(float(found.group(2)), 5e-4)
                self.assertNotIn("differs from the CPU's", err)

    def test_without_the_chain_the_card_still_runs_the_matrices(self):
        for variant, snap in SNAPS.items():
            with self.subTest(variant):
                out, err = run(snap, "permatrix")
                self.assertNotIn("gemma4 chain:", err)
                self.assertEqual(out["greedy"], self.reference(variant)["greedy"])

    def test_the_prompt_is_read_in_chunks_on_the_device(self):
        """The 14 prompt rows go through the chain as one chunk (the CPU's chunk before): the same tokens and logits as the CPU's, and
        with G4_VK_CHUNK=0 (decode rows only) no chunk is stepped there."""
        for variant, snap in SNAPS.items():
            with self.subTest(variant):
                cpu = self.reference(variant)
                out, err = run(snap, "chain")
                found = re.search(r"(\d+) prompt chunks \((\d+) rows", err)
                self.assertTrue(found, err[-600:])
                self.assertEqual((int(found.group(1)), int(found.group(2))), (1, len(IDS)))
                self.assertEqual(out["greedy"], cpu["greedy"])
                self.assertLess(rel_diff(cpu["last"], out["last"]), 2e-3)
                out, err = run(snap, "chain", {"G4_VK_CHUNK": "0"})
                self.assertRegex(err, r"0 prompt chunks \(0 rows")
                self.assertEqual(out["greedy"], cpu["greedy"])

    def test_a_chunk_on_the_matrix_units_is_the_cpus_to_f16(self):
        """The chunk's grouped expert GEMMs on the cooperative matrices (f16 operands, the form a real prompt takes): the same tokens, the
        last logits within what f16 operands reach (a tiny model's random weights amplify it, a real one's do not)."""
        cpu = self.reference("moe")
        out, err = run(SNAPS["moe"], "chain", {"COLI_VK_GRP_COOP": "1"})
        self.assertIn("1 prompt chunks", err)
        self.assertEqual(out["greedy"], cpu["greedy"])
        self.assertLess(rel_diff(cpu["last"], out["last"]), 5e-2)

    def test_a_device_lost_in_the_middle_of_a_step_does_not_change_the_answer(self):
        for variant, snap in SNAPS.items():
            with self.subTest(variant):
                cpu = self.reference(variant)
                lost_mid_step = 0
                # the first frames set the chain up; later ones are decode steps (a mixture layer has two or three a layer)
                for frame in ("30", "45", "52", "70", "110"):
                    out, err = run(snap, "chain", {"COLI_VK_CHAIN_FAULT": frame})
                    self.assertEqual(out["greedy"], cpu["greedy"], f"fault at frame {frame}")
                    self.assertLess(rel_diff(cpu["last"], out["last"]), 2e-3, f"fault at frame {frame}")
                    lost_mid_step += "the device was lost at position" in err
                self.assertGreater(lost_mid_step, 0, "no fault landed in a decode step: the fallback was not exercised")


if __name__ == "__main__":
    unittest.main()
