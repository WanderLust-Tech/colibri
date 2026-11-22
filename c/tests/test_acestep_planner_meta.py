"""The planner's first phase through the engine's line protocol (`acestep --model DIR --serve`), on the tiny fixture whose planner has learned a few metadata blocks.

    ACESTEP_TINY_TEXT=/path/to/make_acestep_tiny.py --lm --lm-text fixture python -m unittest tests.test_acestep_planner_meta

Needs the built engine (make acestep) and that fixture; skipped without them. The first phase's own numbers are held to upstream by `acestep --meta-ref`
(tools/acestep_ref_meta.py); what is under test here is how a request reaches it and what comes back: which requests run the phase at all, what is taken from
the model and what from the caller, the `meta` stage, and the metadata in the AUDIO frame.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_acestep_serve import EXE, Engine  # noqa: E402

TEXT = os.environ.get("ACESTEP_TINY_TEXT")
BARE = "bright ukulele and light percussion, summery"       # one of the blocks the tiny planner learned: 100 bpm, 20 s, C major, 4, unknown


@unittest.skipUnless(EXE and TEXT and os.path.isdir(TEXT), "needs the built engine and ACESTEP_TINY_TEXT")
class Meta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(TEXT)
        kind, cls.ready, _ = cls.engine.next()
        assert kind == "READY", kind
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def gen(self, **fields):
        type(self).n += 1
        rid = f"m{self.n}"
        body = {"id": rid, "caption": BARE, "plan": True, "lm_temperature": 0, "steps": 2, "seed": 3, **fields}
        self.engine.send("GEN", body)
        return self.engine.until_final(rid)

    def stages(self, progress):
        out = []
        for frame in progress:
            if not out or out[-1] != frame["stage"]:
                out.append(frame["stage"])
        return out

    def test_the_planner_is_announced(self):
        self.assertTrue(self.ready["planner"]["available"])

    def test_a_bare_caption_is_enough(self):
        progress, kind, frame = self.gen()
        self.assertEqual(kind, "AUDIO", frame)
        meta = frame["meta"]
        self.assertEqual((meta["bpm"], meta["keyscale"], meta["timesignature"], meta["duration"], meta["language"]), (100, "C major", "4", 20, "unknown"))
        self.assertTrue(meta["caption"].startswith("A bright and cheerful ukulele-led"))
        self.assertEqual((frame["plan_codes"], frame["duration"]), (100, 20.0))              # the length the planner chose
        self.assertEqual(self.stages(progress)[:2], ["meta", "plan"])

    def test_what_is_given_is_kept(self):
        _, kind, frame = self.gen(bpm=90, duration=6)
        self.assertEqual(kind, "AUDIO", frame)
        self.assertEqual((frame["meta"]["bpm"], frame["meta"]["duration"], frame["plan_codes"], frame["duration"]), (90, 6, 30, 6.0))

    def test_all_four_given_skips_the_first_phase(self):
        progress, kind, frame = self.gen(bpm=90, key="D minor", timesig="3", duration=4)
        self.assertEqual(kind, "AUDIO", frame)
        self.assertNotIn("meta", frame)
        self.assertEqual((frame["plan_codes"], self.stages(progress)[0]), (20, "plan"))

    def test_the_caption_and_the_language_can_be_kept(self):
        _, kind, frame = self.gen(lm_caption=False, lm_language=False)
        self.assertEqual(kind, "AUDIO", frame)
        self.assertEqual((frame["meta"]["caption"], frame["meta"]["language"]), ("", ""))
        self.assertEqual(frame["meta"]["bpm"], 100)

    def test_the_same_seed_makes_the_same_song(self):
        a = self.gen(duration=3)[2]
        b = self.gen(duration=3)[2]
        self.assertEqual((a["meta"], a["plan_codes"]), (b["meta"], b["plan_codes"]))

    def test_the_caption_the_dit_follows_is_the_planners(self):
        a = self.gen(duration=3)[2]
        b = self.gen(duration=3, lm_caption=False)[2]
        self.assertGreater(a["caption_tokens"], b["caption_tokens"])                         # the rewrite is longer than the caption it replaced

    def test_without_the_planner_a_zero_duration_is_refused(self):
        self.engine.send("GEN", {"id": "z1", "caption": BARE, "duration": 0})
        _, kind, frame = self.engine.until_final("z1")
        self.assertEqual(kind, "ERROR")
        self.assertIn("duration", frame["message"])

    def test_the_engine_is_still_ready(self):
        self.engine.send("GEN", {"id": "z2", "caption": "calm piano", "duration": 2, "steps": 2})
        _, kind, frame = self.engine.until_final("z2")
        self.assertEqual(kind, "AUDIO", frame)


if __name__ == "__main__":
    unittest.main()
