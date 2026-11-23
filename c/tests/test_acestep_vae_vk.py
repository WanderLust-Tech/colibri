"""The VAE decoder on a Vulkan device makes the song the CPU makes (acestep_vae_vk.h), on the tiny fixture.

    ACESTEP_VK_EXE=/path/to/acestep (built with VK=1) ACESTEP_TINY_C=/path/to/make_acestep_tiny.py --audio-in --base --lm fixture python -m unittest tests.test_acestep_vae_vk

Skipped without both, and without a Vulkan device (the engine says on stderr when the VAE is on the card). The real decoder's numbers on the card are
held to upstream's by the oracle the CPU is (`acestep --vae-ref`, with COLI_VULKAN=1: docs/acestep.md); what is under test here is that the two paths agree
through a whole request, with the DiT left on the CPU in both so that only the VAE differs (the audio within 1e-4 of its norm), a clip long enough for
several tiled windows, and that a failing window and the switch leave the CPU's song.
"""
import os
import struct
import subprocess
import tempfile
import unittest
import wave

VK_EXE = os.environ.get("ACESTEP_VK_EXE")
TINY = os.environ.get("ACESTEP_TINY_C")
ON_CARD = b"[VK] acestep vae: the decoder is on the card"
DROP = ("COLI_VULKAN", "COLI_VK_DIT", "COLI_VK_LM", "COLI_VK_VAE", "COLI_VK_ENC", "ACESTEP_DIT_VK_FAULT", "ACESTEP_VAE_VK_FAULT")


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
    CARD = {"COLI_VULKAN": "1", "COLI_VK_DIT": "0", "COLI_VK_LM": "0", "COLI_VK_ENC": "0"}          # the DiT, the planner and the encoders stay on the CPU: only the VAE is under test

    @classmethod
    def setUpClass(cls):
        cls.cpu, err = make({}, "cpu", "--duration", "3", "--steps", "2")
        assert ON_CARD not in err, "the VAE ran on a card nobody asked for"
        cls.card, err = make(cls.CARD, "card", "--duration", "3", "--steps", "2")
        cls.engaged = ON_CARD in err

    def setUp(self):
        if not self.engaged:
            self.skipTest("no Vulkan device: the VAE stayed on the CPU")

    def test_the_card_makes_the_cpus_song(self):
        self.assertLess(rel(self.cpu, self.card), 1e-4)
        self.assertGreater(sum(abs(x) for x in self.card), 0)

    def test_a_clip_of_several_tiled_windows_makes_the_cpus_song(self):
        cpu, _ = make({}, "lcpu", "--duration", "30", "--steps", "2")             # 750 frames: windows of 256 with an overlap of 64, six of them
        card, _ = make(self.CARD, "lcard", "--duration", "30", "--steps", "2")
        self.assertLess(rel(cpu, card), 1e-4)

    def test_an_odd_length_makes_the_cpus_song(self):
        cpu, _ = make({}, "ocpu", "--duration", "2.7", "--steps", "2")
        card, _ = make(self.CARD, "ocard", "--duration", "2.7", "--steps", "2")
        self.assertLess(rel(cpu, card), 1e-4)

    def test_a_window_that_fails_on_the_card_is_made_on_the_cpu(self):
        """Fault injection (ACESTEP_VAE_VK_FAULT): the card fails its 2nd window; that window is the CPU's and the song is the same."""
        card, err = make({**self.CARD, "ACESTEP_VAE_VK_FAULT": "2"}, "fault", "--duration", "30", "--steps", "2")
        cpu, _ = make({}, "faultcpu", "--duration", "30", "--steps", "2")
        self.assertEqual(err.count(b"a window failed; the CPU runs it"), 1, err[-400:])
        self.assertLess(rel(cpu, card), 1e-4)

    def test_the_switch_keeps_the_vae_on_the_cpu(self):
        song, err = make({**self.CARD, "COLI_VK_VAE": "0"}, "off", "--duration", "3", "--steps", "2")
        self.assertNotIn(ON_CARD, err)
        self.assertEqual(song, self.cpu)


if __name__ == "__main__":
    unittest.main()
