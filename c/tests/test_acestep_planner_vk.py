"""The planner on a Vulkan device gives the song the CPU gives (acestep_lm_vk.h), on the tiny fixture whose planner has learned a few metadata blocks.

    ACESTEP_VK_EXE=/path/to/acestep (built with VK=1: make VK=1 acestep) ACESTEP_TINY_TEXT=/path/to/fixture python -m unittest tests.test_acestep_planner_vk

Skipped without both, and without a Vulkan device (the engine says on stderr when the planner is on the card). The real planner's numbers on the card
are held to upstream's by the same oracles the CPU is (`acestep --meta-ref`, `--lm-ref`, with COLI_VULKAN=1: docs/acestep.md); what is under test here is
that the two paths agree through the whole request: phase 1's metadata, the codes, and so the audio the DiT makes from them, byte for byte, and that
COLI_VK_LM=0 keeps the planner on the CPU.
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_acestep_serve as serve  # noqa: E402

VK_EXE = os.environ.get("ACESTEP_VK_EXE")
TEXT = os.environ.get("ACESTEP_TINY_TEXT")
BARE = "bright ukulele and light percussion, summery"
ON_CARD = b" on the card ("


class Run(serve.Engine):
    """An engine process of a chosen binary and environment, with its stderr kept."""

    def __init__(self, model, env):
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen([VK_EXE, "--model", model, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, bufsize=0,
                                     env={**{k: v for k, v in os.environ.items() if k not in ("COLI_VULKAN", "COLI_VK_LM")}, **env})
        import queue
        import threading
        self.frames = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def stderr(self):
        self.err.seek(0)
        return self.err.read()

    def close(self):
        super().close()
        self.err.close()


def song(env, **fields):
    engine = Run(TEXT, env)
    try:
        kind, ready, _ = engine.next(300)
        assert kind == "READY", (kind, ready)
        engine.send("GEN", {"id": "a", "caption": BARE, "plan": True, "lm_temperature": 0, "steps": 2, "seed": 3, **fields})
        _, kind, frame = engine.until_final("a")
        assert kind == "AUDIO", frame
        return frame, engine.stderr()
    finally:
        engine.close()


@unittest.skipUnless(VK_EXE and os.path.isfile(VK_EXE) and TEXT and os.path.isdir(TEXT), "needs ACESTEP_VK_EXE (a VK=1 build) and ACESTEP_TINY_TEXT")
class OnTheCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu, err = song({})
        assert ON_CARD not in err, "the planner ran on a card nobody asked for"
        cls.card, err = song({"COLI_VULKAN": "1"})
        cls.engaged = ON_CARD in err

    def setUp(self):
        if not self.engaged:
            self.skipTest("no Vulkan device: the planner stayed on the CPU")

    def test_a_bare_caption_gives_the_same_song(self):
        for key in ("meta", "plan_codes", "duration", "samples", "bytes"):
            self.assertEqual(self.card[key], self.cpu[key], key)
        self.assertEqual(self.card["_pcm"], self.cpu["_pcm"])

    def test_the_codes_phase_alone_gives_the_same_song(self):
        fields = dict(bpm=100, key="C major", timesig="4", duration=4)         # all four given: phase 1 is skipped
        cpu, _ = song({}, **fields)
        card, _ = song({"COLI_VULKAN": "1"}, **fields)
        self.assertNotIn("meta", cpu)
        self.assertEqual((card["plan_codes"], card["_pcm"]), (cpu["plan_codes"], cpu["_pcm"]))

    def test_a_step_that_fails_on_the_card_is_made_again_on_the_cpu(self):
        """Fault injection (ACESTEP_LM_VK_FAULT): the card fails at its 40th step, in phase 1; the song is the CPU's, and so is the next one."""
        engine = Run(TEXT, {"COLI_VULKAN": "1", "ACESTEP_LM_VK_FAULT": "40"})
        try:
            kind, _, _ = engine.next(300)
            self.assertEqual(kind, "READY")
            for rid in ("a", "b"):
                engine.send("GEN", {"id": rid, "caption": BARE, "plan": True, "lm_temperature": 0, "steps": 2, "seed": 3})
                _, kind, frame = engine.until_final(rid)
                self.assertEqual(kind, "AUDIO", frame)
                self.assertEqual((frame["meta"], frame["plan_codes"], frame["_pcm"]), (self.cpu["meta"], self.cpu["plan_codes"], self.cpu["_pcm"]), rid)
            err = engine.stderr()
        finally:
            engine.close()
        self.assertIn(b"the CPU runs the planner from here on", err)
        self.assertEqual(err.count(ON_CARD), 1, "the planner went back to the card after it failed")

    def test_the_switch_keeps_the_planner_on_the_cpu(self):
        frame, err = song({"COLI_VULKAN": "1", "COLI_VK_LM": "0"})
        self.assertNotIn(ON_CARD, err)
        self.assertEqual(frame["_pcm"], self.cpu["_pcm"])


if __name__ == "__main__":
    unittest.main()
