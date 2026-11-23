"""The text and condition encoders on a Vulkan device make the song the CPU makes (acestep_enc_vk.h), on the tiny fixture.

    ACESTEP_VK_EXE=/path/to/acestep (built with VK=1) ACESTEP_TINY_C=/path/to/make_acestep_tiny.py --audio-in --base --lm fixture python -m unittest tests.test_acestep_enc_vk

Skipped without both, and without a Vulkan device (the engine says on stderr when an encoder stack is on the card). The real encoders' numbers on the card are held to
upstream's by the oracles the CPU's are (`acestep --text-ref`, `--gen-ref` with COLI_VULKAN=1: docs/acestep.md); what is under test here is that the two paths agree
through a whole request, with the DiT, the VAE and the planner left on the CPU in both so that only the encoders differ (the audio within 1e-3 of its norm), for the
turbo DiT's lyric and timbre encoders and the guided base DiT's own, with lyrics (the lyric encoder's sliding layers see more than one window's rows), and that a failing
call and the switch leave the CPU's song.
"""
import os
import struct
import subprocess
import tempfile
import unittest
import wave

VK_EXE = os.environ.get("ACESTEP_VK_EXE")
TINY = os.environ.get("ACESTEP_TINY_C")
ON_CARD = b"[VK] acestep encoder:"
DROP = ("COLI_VULKAN", "COLI_VK_DIT", "COLI_VK_LM", "COLI_VK_VAE", "COLI_VK_ENC", "ACESTEP_ENC_VK_FAULT")
LYRICS = "[Verse]\n" + "the quick brown fox jumps over the lazy dog and keeps on running through the night\n" * 12


def make(env, name, *args):
    """One song through the CLI: (the samples, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, name + ".wav")
        clean = {k: v for k, v in os.environ.items() if k not in DROP}
        done = subprocess.run([VK_EXE, "--model", TINY, "--caption", "calm piano", "--seed", "5", *args, "--out", out], env={**clean, **env},
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
    CARD = {"COLI_VULKAN": "1", "COLI_VK_DIT": "0", "COLI_VK_LM": "0", "COLI_VK_VAE": "0"}      # only the encoders are under test

    @classmethod
    def setUpClass(cls):
        cls.cpu, err = make({}, "cpu", "--duration", "3", "--steps", "2")
        assert ON_CARD not in err, "an encoder ran on a card nobody asked for"
        cls.card, err = make(cls.CARD, "card", "--duration", "3", "--steps", "2")
        cls.engaged = ON_CARD in err
        cls.stacks = err.count(ON_CARD)

    def setUp(self):
        if not self.engaged:
            self.skipTest("no Vulkan device: the encoders stayed on the CPU")

    def test_the_three_stacks_make_the_cpus_song(self):
        self.assertEqual(self.stacks, 3, "the text, lyric and timbre stacks")
        self.assertLess(rel(self.cpu, self.card), 1e-3)
        self.assertGreater(sum(abs(x) for x in self.card), 0)

    def test_lyrics_make_the_cpus_song(self):
        cpu, _ = make({}, "lcpu", "--lyrics", LYRICS, "--duration", "3", "--steps", "2")
        card, _ = make(self.CARD, "lcard", "--lyrics", LYRICS, "--duration", "3", "--steps", "2")
        self.assertLess(rel(cpu, card), 1e-3)

    def test_the_guided_base_dits_own_encoders_make_the_cpus_song(self):
        cpu, _ = make({}, "bcpu", "--dit", "base", "--duration", "3", "--steps", "4")
        card, err = make(self.CARD, "bcard", "--dit", "base", "--duration", "3", "--steps", "4")
        self.assertEqual(err.count(ON_CARD), 3, err[-400:])                          # the text stack, and the base DiT's lyric and timbre stacks
        self.assertLess(rel(cpu, card), 1e-3)

    def test_a_call_that_fails_on_the_card_is_made_on_the_cpu(self):
        """Fault injection (ACESTEP_ENC_VK_FAULT): the card fails its 2nd call, the lyric encoder's; the CPU makes that one and the song is the same."""
        card, err = make({**self.CARD, "ACESTEP_ENC_VK_FAULT": "2"}, "fault", "--duration", "3", "--steps", "2")
        self.assertEqual(err.count(b"a call failed; the CPU runs it"), 1, err[-400:])
        self.assertLess(rel(self.cpu, card), 1e-3)

    def test_the_switch_keeps_the_encoders_on_the_cpu(self):
        song, err = make({**self.CARD, "COLI_VK_ENC": "0"}, "off", "--duration", "3", "--steps", "2")
        self.assertNotIn(ON_CARD, err)
        self.assertEqual(song, self.cpu)


if __name__ == "__main__":
    unittest.main()
