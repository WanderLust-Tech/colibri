"""A small model that started on the CPU only because the card was full is stopped once the card has room, to load there next time
(router.Pool.promote)."""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu  # noqa: E402
import router  # noqa: E402
from test_gpu import CapturingRig  # noqa: E402
from test_router import GB, entry  # noqa: E402

BIG, SMALL = "big-colibri", "small-colibri"
ON_CARD = {BIG: 20.0, SMALL: 2.0}          # what each takes of a 24 GB card when it runs there


class PromoteTest(unittest.TestCase):
    """A big model that holds most of a 24 GB card, and a small one that finds the card full and starts on the CPU."""

    def setUp(self):
        self._modalities = router.PROMOTE_MODALITIES
        router.PROMOTE_MODALITIES = ("text",)               # the fake workers only answer chat: stand in for embedding and rerank

    def tearDown(self):
        router.PROMOTE_MODALITIES = self._modalities

    def make(self, **pool_kw):
        holder = {}

        def card():
            used = sum(ON_CARD[w.entry["id"]] for w in holder["rig"].pool.snapshot()
                       if w.card and w.state in ("starting", "ready", "stopping", "demoting"))
            return 24.0 - used, 24.0
        policy = gpu.GpuPolicy("auto", reserve_gb=2.0, min_gb=3.0, probe=card)
        policy.memory = card
        kw = {"card_min_idle": 9999.0, "promote_min_idle": 0.0, "promote_min_age": 0.0, "watch_pressure": False}
        kw.update(pool_kw)
        rig = CapturingRig(self, [entry("big", size=20 * GB), entry("small", size=1 * GB)], gpu=policy, pool_kw=kw)
        holder["rig"] = rig
        return rig

    def on_card(self, rig, name):
        return [env["COLI_VULKAN"] for model, env in rig.started if model == name][-1] == "1"

    def worker(self, rig, name):
        return rig.pool.workers[next(item["key"] for item in rig.entries if item["id"] == name)]

    def full_card_then_free(self, rig):
        """big takes the card, small is started beside it (on the CPU), and big is unloaded."""
        rig.chat(BIG)
        rig.chat(SMALL)
        self.assertTrue(self.on_card(rig, BIG))
        self.assertFalse(self.on_card(rig, SMALL))                                       # the card was full for it
        self.assertEqual(rig.pool.unload(self.worker(rig, BIG).key), "unloaded")

    def later(self):
        return time.monotonic() + 1000.0

    def test_a_cpu_model_is_stopped_once_the_card_has_room_and_loads_there_next_time(self):
        rig = self.make()
        self.full_card_then_free(rig)
        self.assertEqual(rig.pool.promote(self.later()), SMALL)
        self.assertEqual(rig.states(), {BIG: "idle", SMALL: "idle"})                     # both unloaded
        self.assertEqual(rig.chat(SMALL)[0], 200)
        self.assertTrue(self.on_card(rig, SMALL))                                        # the next load took the card
        self.assertEqual(rig.pool.promotions["stops"], 1)
        self.assertEqual(rig.pool.promotions["last"]["model"], SMALL)

    def test_nothing_happens_while_the_card_is_still_full(self):
        rig = self.make()
        rig.chat(BIG)
        rig.chat(SMALL)
        self.assertIsNone(rig.pool.promote(self.later()))
        self.assertEqual(rig.states(), {BIG: "ready", SMALL: "ready"})

    def test_the_card_must_have_room_for_the_model_and_not_only_for_a_start(self):
        rig = self.make()
        self.full_card_then_free(rig)
        rig.pool.workers[self.worker(rig, SMALL).key].resident = 30 * GB                 # it would not fit even on an empty card
        self.assertIsNone(rig.pool.promote(self.later()))

    def test_a_model_that_is_answering_is_left_alone(self):
        rig = self.make()
        self.full_card_then_free(rig)
        self.worker(rig, SMALL).inflight = 1
        self.assertIsNone(rig.pool.promote(self.later()))
        self.worker(rig, SMALL).inflight = 0

    def test_a_model_used_a_moment_ago_or_started_a_moment_ago_is_left_alone(self):
        rig = self.make(promote_min_idle=30.0, promote_min_age=60.0)
        self.full_card_then_free(rig)
        self.assertIsNone(rig.pool.promote(time.monotonic() + 10))                       # too soon after use
        self.assertIsNone(rig.pool.promote(time.monotonic() + 45))                       # idle enough, but not running long enough
        self.assertEqual(rig.pool.promote(time.monotonic() + 100), SMALL)

    def test_a_model_idle_for_less_than_the_wait_is_left_alone_however_long_it_has_run(self):
        rig = self.make(promote_min_idle=30.0, promote_min_age=60.0)
        self.full_card_then_free(rig)
        worker = self.worker(rig, SMALL)
        worker.started = time.monotonic() - 1000                                         # old enough
        worker.last_used = time.monotonic() - 5                                          # but just used
        self.assertIsNone(rig.pool.promote())
        worker.last_used = time.monotonic() - 100
        self.assertEqual(rig.pool.promote(), SMALL)

    def test_a_model_pinned_to_the_cpu_or_that_failed_on_the_card_stays_where_it_is(self):
        rig = self.make()
        self.full_card_then_free(rig)
        worker = self.worker(rig, SMALL)
        for why in ("pinned to the CPU", "the card failed to start it: lost device"):
            worker.gpu = gpu.Decision("cpu", why, env={"COLI_VULKAN": "0"})              # room is False: the card was not why
            self.assertIsNone(rig.pool.promote(self.later()), why)

    def test_a_model_on_the_card_or_that_gave_it_back_is_not_touched(self):
        rig = self.make()
        rig.chat(SMALL)                                                                  # the card is free: it starts there
        self.assertTrue(self.on_card(rig, SMALL))
        self.assertIsNone(rig.pool.promote(self.later()))
        # each guard on its own: the decision would allow it (the card was why), so only the worker's state says no
        worker = self.worker(rig, SMALL)
        worker.gpu = gpu.Decision("cpu", "only 1 GB free", env={"COLI_VULKAN": "0"}, room=True)
        self.assertTrue(worker.card)
        self.assertIsNone(rig.pool.promote(self.later()))                                # still on the card
        worker.card, worker.demoted = False, True
        self.assertIsNone(rig.pool.promote(self.later()))                                # gave the card back: nothing to move

    def test_only_the_models_that_load_quickly_are_moved_by_default(self):
        router.PROMOTE_MODALITIES = self._modalities                                     # embedding and rerank, not text
        rig = self.make()
        self.full_card_then_free(rig)
        self.assertIsNone(rig.pool.promote(self.later()))
        self.assertEqual(rig.states(), {BIG: "idle", SMALL: "ready"})

    def test_it_can_be_turned_off_and_does_nothing_unless_the_policy_is_auto(self):
        rig = self.make(card_promote=False)
        self.full_card_then_free(rig)
        self.assertIsNone(rig.pool.promote(self.later()))
        rig = self.make()
        self.full_card_then_free(rig)
        rig.pool.gpu.mode = "off"
        self.assertIsNone(rig.pool.promote(self.later()))

    def test_health_shows_what_it_did(self):
        rig = self.make()
        self.full_card_then_free(rig)
        rig.pool.promote(self.later())
        body = rig.call("GET", "/health")[3]["router"]["card_promote"]
        self.assertEqual((body["on"], body["stops"], body["last"]["model"]), (True, 1, SMALL))

    def test_the_reaper_does_it_by_itself(self):
        rig = self.make(watch_pressure=True)
        self.full_card_then_free(rig)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and rig.states()[SMALL] != "idle":
            time.sleep(0.25)
        self.assertEqual(rig.states()[SMALL], "idle")                                    # it was stopped without a call
        self.assertEqual(rig.pool.promotions["stops"], 1)


if __name__ == "__main__":
    unittest.main()
