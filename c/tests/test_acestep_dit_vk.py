"""The DiT on a Vulkan device makes the song the CPU makes (acestep_dit_vk.h), on the tiny guided fixture (4 layers, 4 query heads over 2 key/value heads, a window of 8).

    ACESTEP_VK_EXE=/path/to/acestep (built with VK=1) ACESTEP_TINY_C=/path/to/make_acestep_tiny.py --audio-in --base --lm fixture python -m unittest tests.test_acestep_dit_vk

Skipped without both, and without a Vulkan device (the engine says on stderr when the DiT is on the card). The real DiT's numbers on the card are held to
upstream's by the oracles the CPU is (`acestep --dit-ref`, `--gen-ref`, with COLI_VULKAN=1: docs/acestep.md); what is under test here is that the two paths
agree through a whole request, as far as float rounding lets them (the audio within 1e-3 of its norm; the tiny fixture's is 1e-5), for the turbo DiT and the
guided base one, with a clip longer than the window (the sliding layers' ranges), and that a failing step and the switch leave the CPU's song.
"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

VK_EXE = os.environ.get("ACESTEP_VK_EXE")
TINY = os.environ.get("ACESTEP_TINY_C")
ON_CARD = b"[VK] acestep dit:"
ON_CARD_UP = b" on the card ("


def make(model, name, env, *args):
    """One song through the CLI: (the samples, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, name + ".wav")
        clean = {k: v for k, v in os.environ.items() if k not in ("COLI_VULKAN", "COLI_VK_DIT", "COLI_VK_LM", "COLI_VK_VAE", "ACESTEP_DIT_VK_FAULT")}
        done = subprocess.run([VK_EXE, "--model", model, "--caption", "calm piano", "--seed", "5", *args, "--out", out], env={**clean, **env},
                              capture_output=True, timeout=600)
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-800:]
        with wave.open(out) as w:
            data = w.readframes(w.getnframes())
        return struct.unpack("<%dh" % (len(data) // 2), data), done.stderr


def rel(a, b):
    assert len(a) == len(b), (len(a), len(b))
    num = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    den = sum(x * x for x in a) ** 0.5
    return num / den if den else num


@unittest.skipUnless(VK_EXE and os.path.isfile(VK_EXE) and TINY and os.path.isdir(TINY), "needs ACESTEP_VK_EXE (a VK=1 build) and ACESTEP_TINY_C")
class OnTheCard(unittest.TestCase):
    CARD = {"COLI_VULKAN": "1"}

    @classmethod
    def setUpClass(cls):
        cls.cpu, err = make(TINY, "cpu", {}, "--duration", "3", "--steps", "2")
        assert ON_CARD not in err, "the DiT ran on a card nobody asked for"
        cls.card, err = make(TINY, "card", cls.CARD, "--duration", "3", "--steps", "2")
        cls.engaged = ON_CARD in err and ON_CARD_UP in err

    def setUp(self):
        if not self.engaged:
            self.skipTest("no Vulkan device: the DiT stayed on the CPU")

    def test_the_turbo_dit_makes_the_cpus_song(self):
        self.assertLess(rel(self.cpu, self.card), 1e-3)
        self.assertGreater(sum(abs(x) for x in self.card), 0)

    def test_the_guided_base_dit_makes_the_cpus_song(self):
        cpu, _ = make(TINY, "bcpu", {}, "--dit", "base", "--duration", "3", "--steps", "4")
        card, err = make(TINY, "bcard", self.CARD, "--dit", "base", "--duration", "3", "--steps", "4")
        self.assertIn(b"acestep-v15-base on the card", err)
        self.assertLess(rel(cpu, card), 2e-3)

    def test_a_clip_longer_than_the_window_makes_the_cpus_song(self):
        cpu, _ = make(TINY, "lcpu", {}, "--duration", "12", "--steps", "3")          # 150 tokens against a window of 8: the ranges of the sliding layers matter
        card, _ = make(TINY, "lcard", self.CARD, "--duration", "12", "--steps", "3")
        self.assertLess(rel(cpu, card), 1e-3)

    def test_an_odd_length_makes_the_cpus_song(self):
        cpu, _ = make(TINY, "ocpu", {}, "--duration", "2.7", "--steps", "2")
        card, _ = make(TINY, "ocard", self.CARD, "--duration", "2.7", "--steps", "2")
        self.assertLess(rel(cpu, card), 1e-3)

    def test_a_step_that_fails_on_the_card_is_made_on_the_cpu(self):
        """Fault injection (ACESTEP_DIT_VK_FAULT): the card fails its 2nd step; that step is the CPU's and the song is the same."""
        card, err = make(TINY, "fault", {**self.CARD, "ACESTEP_DIT_VK_FAULT": "2"}, "--duration", "3", "--steps", "4")
        cpu, _ = make(TINY, "faultcpu", {}, "--duration", "3", "--steps", "4")
        self.assertEqual(err.count(b"a step failed; the CPU runs it"), 1, err[-400:])
        self.assertLess(rel(cpu, card), 1e-3)

    def test_the_switch_keeps_the_dit_on_the_cpu(self):
        song, err = make(TINY, "off", {**self.CARD, "COLI_VK_DIT": "0", "COLI_VK_VAE": "0"}, "--duration", "3", "--steps", "2")
        self.assertNotIn(ON_CARD, err)
        self.assertEqual(song, self.cpu)


if __name__ == "__main__":
    unittest.main()
