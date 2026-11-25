"""Gemma 4's routed experts on the Vulkan device (the shared expert tier, vk_tier.c) against the CPU's.

    python -m unittest tests.test_gemma4_tier        (from c/)

Needs a gemma4 built with VK=1 and a Vulkan device; skipped without them. The model is the tiny random mixture-of-experts Gemma 4
of test_gemma4 (tools/make_gemma4_tiny.py). Its experts are in the engine's own int4 planar blocks of 64, so the device reads them
as they are, and its activation is GeGLU (gelu_pytorch_tanh), which the tier's shaders take as act 3.

What is held:
  - the tier takes every routed expert (the engine says "tier gemma4: on" and the run's line says 100% on the device) and the
    greedy continuation is the CPU's, the last logits within float rounding (the device adds in another order);
  - a budget too small for every expert leaves some on the CPU, and the rows the two computed are added in one rank order: the
    answer is still the CPU's, and the run's line says so (resident fewer than the experts, the rest on the CPU);
  - COLI_VK_TIER=0 keeps the experts with the CPU: no tier, the same answer;
  - a batch that fails in the middle of a step (COLI_VK_TIER_FAULT=n fakes it at the n-th) is recomputed on the CPU, the tier
    stops, and the answer is the CPU's;
  - the history (.coli_usage) is written beside the model and read back: the second run starts from it.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_gemma4 as t4  # noqa: E402

IDS = [1, 5, 9, 12, 3, 7, 2, 8, 4, 6, 10, 11, 13, 14]
GREEDY = 24


def run(snap, mode, extra=None, greedy=GREEDY, usage=None):
    """The engine's test mode on a prompt of token ids; mode: cpu, card (the card with the tier on unless `extra` turns it off).
    Returns (the JSON it prints, its stderr)."""
    env = dict(os.environ, SNAP=snap, G4_TEST="1", G4_IDS=",".join(map(str, IDS)), G4_GREEDY=str(greedy),
               OMP_NUM_THREADS="2", G4_SLACK="0", G4_IMG_TOKENS="1")
    for name in ("COLI_VULKAN", "COLI_VK_CHAIN", "COLI_VK_TIER", "COLI_VK_TIER_GB", "COLI_VK_TIER_FAULT", "COLI_VK_TIER_WARM",
                 "COLI_USAGE", "G4_VK_DENSE"):
        env.pop(name, None)
    if mode == "card":
        env["COLI_VULKAN"] = "1"
    if usage:
        env["COLI_USAGE"] = usage
    env.update(extra or {})
    done = subprocess.run([t4.EXE, "128"], env=env, capture_output=True, text=True, timeout=300, cwd=os.path.join(HERE, ".."))
    if done.returncode:
        raise AssertionError(done.stdout[-400:] + done.stderr[-1500:])
    return json.loads(done.stdout.strip().splitlines()[-1]), done.stderr


def rel_diff(a, b):
    return max(abs(x - y) / max(1.0, abs(x), abs(y)) for x, y in zip(a, b))


def have_tier():
    """A VK build with a device that takes the experts: its stderr says so."""
    if not t4.HAVE_VARIANTS:
        return False
    try:
        with tempfile.TemporaryDirectory() as d:
            _, err = run(t4.TINY, "card", greedy=2, usage=os.path.join(d, "u"))
    except (AssertionError, OSError, subprocess.SubprocessError):
        return False
    return "tier gemma4: on" in err


HAVE_TIER = have_tier()


@unittest.skipUnless(HAVE_TIER, "needs gemma4 built with VK=1 and a Vulkan device")
class TierTest(unittest.TestCase):
    cpu = None

    @classmethod
    def reference(cls):
        if cls.cpu is None:
            cls.cpu = run(t4.TINY, "cpu")[0]
        return cls.cpu

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.usage = os.path.join(self.dir, "usage")          # the history stays out of the fixture's directory

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def card(self, extra=None, **kw):
        return run(t4.TINY, "card", extra, usage=self.usage, **kw)

    def same_as_cpu(self, out, what):
        cpu = self.reference()
        self.assertEqual(out["greedy"], cpu["greedy"], what)
        self.assertLess(rel_diff(cpu["last"], out["last"]), 2e-3, what)

    def test_every_routed_expert_runs_on_the_device(self):
        out, err = self.card()
        self.assertIn("tier gemma4: on", err)
        found = re.search(r"tier gemma4 run: device (\d+) of (\d+) routed experts \(100\.0%", err)
        self.assertTrue(found, err[-1200:])
        self.assertEqual(found.group(1), found.group(2))
        self.assertGreater(int(found.group(1)), 0)
        self.same_as_cpu(out, "all on the device")

    def test_a_budget_that_holds_a_part_of_the_experts_shares_them_with_the_cpu(self):
        # the tiny experts are about 7 KiB each: 0.00004 GiB is a few
        # (COLI_VK_TIER_HOST=0: with the host pool, the experts the card cannot hold are read from system RAM by the device, so all
        # of them would run on it; this test is of the other split, the CPU taking the rest)
        out, err = self.card({"COLI_VK_TIER_GB": "0.00004", "COLI_VK_TIER_RATE": "0", "COLI_VK_TIER_HOST": "0"})
        found = re.search(r"tier gemma4 run: device (\d+) of (\d+) routed experts", err)
        self.assertTrue(found, err[-1200:])
        self.assertGreater(int(found.group(1)), 0)
        self.assertLess(int(found.group(1)), int(found.group(2)))
        self.same_as_cpu(out, "device and CPU experts in one layer")

    def test_the_tier_can_be_turned_off(self):
        out, err = self.card({"COLI_VK_TIER": "0"})
        self.assertNotIn("tier gemma4", err)
        self.same_as_cpu(out, "COLI_VK_TIER=0")

    def test_a_batch_that_fails_is_recomputed_on_the_cpu(self):
        failed = 0
        for n in ("1", "7", "40", "150"):
            out, err = self.card({"COLI_VK_TIER_FAULT": n})
            self.same_as_cpu(out, f"batch {n} failed")
            failed += "a batch failed, its experts are recomputed on the CPU and the tier stops" in err
        self.assertGreater(failed, 0, "no fault landed: the fallback was not exercised")

    def test_the_history_is_kept_and_read_back(self):
        self.card()
        self.assertTrue(os.path.exists(self.usage), "no .coli_usage written")
        with open(self.usage) as f:
            text = f.read()
        self.assertIn("-2 1 ", text)                                    # route_trace.h's header record: the engine's identity
        out, err = self.card()
        self.assertIn("[USAGE] expert history:", err)
        self.assertIn("warm start", err)
        self.same_as_cpu(out, "started from the history")


if __name__ == "__main__":
    unittest.main()
