"""Who the router stops (cost-aware eviction), when memory runs low between requests, and what it learns about a model's size."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu  # noqa: E402
import router  # noqa: E402
from test_gpu import CapturingRig  # noqa: E402
from test_router import GB, Rig, entry  # noqa: E402

KEY = {"a": "text-a-colibri", "b": "text-b-colibri", "c": "text-c-colibri"}


def worker(key, resident, idle, now):
    """A stand-in for a ready, idle Worker: what the scoring reads."""
    return SimpleNamespace(entry={"key": key, "id": key, "size_bytes": resident}, resident=resident, last_used=now - idle)


class StateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "router.json")

    def test_a_load_time_is_the_slowest_of_the_last_three(self):
        state = router.State(self.path)
        self.assertIsNone(state.load_seconds("k"))
        for seconds in (300.0, 20.0, 18.0):
            state.record_load("k", "m", seconds)
        self.assertEqual(state.load_seconds("k"), 300.0)                 # a cold read is what a wrong eviction would cost again
        state.record_load("k", "m", 22.0)                                # the 300 falls out of the last three
        self.assertEqual(state.load_seconds("k"), 22.0)

    def test_the_peak_only_rises_and_is_written_for_a_rise_that_matters(self):
        state = router.State(self.path)
        self.assertTrue(state.record_peak("k", "m", 10 * GB))
        self.assertFalse(state.record_peak("k", "m", 10 * GB + 1000))    # under 3%: not worth a write
        self.assertFalse(state.record_peak("k", "m", 5 * GB))            # a smaller reading never lowers it
        self.assertTrue(state.record_peak("k", "m", 12 * GB))
        self.assertEqual(state.peak("k"), 12 * GB)

    def test_both_come_back_after_a_restart_and_the_old_fields_are_kept(self):
        state = router.State(self.path)
        state.record("k", "m", 8 * GB)
        state.record_load("k", "m", 40.0)
        state.record_peak("k", "m", 9 * GB)
        again = router.State(self.path)
        self.assertEqual((again.resident("k"), again.load_seconds("k"), again.peak("k")), (8 * GB, 40.0, 9 * GB))

    def test_garbage_in_the_file_is_ignored(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"k": {"loads": ["x", -3, None, 12], "peak_bytes": "big", "resident_bytes": 0}}, handle)
        state = router.State(self.path)
        self.assertEqual(state.load_seconds("k"), 12)
        self.assertIsNone(state.peak("k"))
        self.assertIsNone(state.resident("k"))


class ScoreTest(unittest.TestCase):
    """The choice of victim, on its own."""

    def pool(self, loads=None):
        rig_state = router.State()
        for key, seconds in (loads or {}).items():
            rig_state.record_load(key, key, seconds)
        pool = router.Pool(router.Catalog(entries=[]), state=rig_state, memory=lambda: None)
        self.addCleanup(pool.close)
        return pool

    def test_a_model_that_is_slow_to_load_again_stays(self):
        now = time.monotonic()
        pool = self.pool({"usb": 300.0, "nvme": 8.0})
        usb, nvme = worker("usb", 20 * GB, 600, now), worker("nvme", 20 * GB, 600, now)      # same size, same idle time
        self.assertIs(pool.pick_victim([usb, nvme]), nvme)

    def test_longer_idle_and_bigger_go_first_at_equal_cost(self):
        now = time.monotonic()
        pool = self.pool({"x": 30.0, "y": 30.0, "z": 30.0})
        old, new, big = worker("x", 8 * GB, 3000, now), worker("y", 8 * GB, 60, now), worker("z", 16 * GB, 3000, now)
        self.assertIs(pool.pick_victim([old, new]), old)
        self.assertIs(pool.pick_victim([old, big]), big)

    def test_when_memory_is_short_one_that_frees_enough_is_preferred_to_a_string_of_small_stops(self):
        now = time.monotonic()
        pool = self.pool({"small": 5.0, "large": 60.0})
        small, large = worker("small", 2 * GB, 900, now), worker("large", 12 * GB, 900, now)
        self.assertIs(pool.pick_victim([small, large]), small)                         # no need given: the cheap one scores higher
        self.assertIs(pool.pick_victim([small, large], need=6 * GB), large)            # six GB are needed: only the large one frees that
        self.assertIs(pool.pick_victim([small, large], need=40 * GB), small)           # nothing frees enough: the best score, and the planner asks again

    def test_a_model_never_timed_is_guessed_from_its_size_and_a_tie_goes_to_the_longest_idle(self):
        now = time.monotonic()
        pool = self.pool()
        self.assertGreaterEqual(pool.reload_seconds(worker("q", 30 * GB, 1, now)), 100)     # 30 GB at a slow disk's speed
        self.assertEqual(pool.reload_seconds(worker("q", 1000, 1, now)), 5.0)
        first, second = worker("a", 0, 0.1, now), worker("b", 0, 0.5, now)                  # inside the one-second floor: equal scores
        self.assertIs(pool.pick_victim([first, second]), second)


class EvictionTest(unittest.TestCase):
    def test_the_cheap_to_reload_model_is_stopped_though_the_other_is_older(self):
        rig = Rig(self, [entry("text-a"), entry("text-b"), entry("text-c")], max_loaded=2)
        rig.chat(KEY["a"])                                                                # a is the older
        rig.chat(KEY["b"])
        by_id = {item["id"]: item["key"] for item in rig.entries}
        rig.pool.state.data[by_id[KEY["a"]]]["loads"] = [300.0]                           # a came off a slow drive
        rig.pool.state.data[by_id[KEY["b"]]]["loads"] = [4.0]
        time.sleep(0.1)
        rig.chat(KEY["c"])                                                                # no room: one of them has to go
        states = rig.states()
        self.assertEqual((states[KEY["a"]], states[KEY["b"]], states[KEY["c"]]), ("ready", "idle", "ready"))

    def test_the_load_time_is_recorded_when_a_model_starts(self):
        rig = Rig(self, [entry("text-a")])
        rig.chat(KEY["a"])
        key = rig.entries[0]["key"]
        self.assertGreater(rig.pool.state.load_seconds(key), 0)

    def test_health_says_how_long_a_loaded_model_takes_to_load_again(self):
        rig = Rig(self, [entry("text-a")], api_key="k")
        rig.chat(KEY["a"], ) if False else rig.call("POST", "/v1/chat/completions", {"model": KEY["a"], "messages": []}, {"Authorization": "Bearer k"})
        loaded = rig.call("GET", "/health", headers={"Authorization": "Bearer k"})[3]["router"]["loaded"]
        self.assertGreater(loaded[0]["reload_seconds"], 0)


class CardEvictionTest(unittest.TestCase):
    """A model that would run on the CPU only because an idle one holds the graphics card: the idle one goes first."""

    def make(self, **pool_kw):
        holder = {}

        def card():                                           # each model on the card takes 20 of its 24 GB
            used = sum(20.0 for w in holder["rig"].pool.snapshot() if w.card and w.state in ("starting", "ready", "stopping"))
            return 24.0 - used, 24.0
        policy = gpu.GpuPolicy("auto", reserve_gb=2.0, min_gb=6.0, probe=card)
        policy.memory = card
        kw = {"card_min_idle": 0.0, "card_settle": 2.0}
        kw.update(pool_kw)
        rig = CapturingRig(self, [entry("text-a"), entry("text-b")], gpu=policy, pool_kw=kw, parallel=kw.pop("parallel", False))
        holder["rig"] = rig
        return rig

    def on_card(self, rig, name):
        return [env["COLI_VULKAN"] for model, env in rig.started if model == name][-1] == "1"

    def test_the_idle_model_that_holds_the_card_is_stopped_so_the_next_one_can_use_it(self):
        rig = self.make()
        rig.chat(KEY["a"])
        self.assertTrue(self.on_card(rig, KEY["a"]))
        rig.chat(KEY["b"])
        self.assertTrue(self.on_card(rig, KEY["b"]))
        self.assertEqual(rig.states(), {KEY["a"]: "idle", KEY["b"]: "ready"})

    def test_it_can_be_turned_off(self):
        rig = self.make(card_evict=False)
        rig.chat(KEY["a"])
        rig.chat(KEY["b"])
        self.assertFalse(self.on_card(rig, KEY["b"]))                                    # the CPU, as before
        self.assertEqual(rig.states(), {KEY["a"]: "ready", KEY["b"]: "ready"})

    def test_a_model_used_a_moment_ago_keeps_the_card(self):
        rig = self.make(card_min_idle=60.0)
        rig.chat(KEY["a"])
        rig.chat(KEY["b"])
        self.assertFalse(self.on_card(rig, KEY["b"]))
        self.assertEqual(rig.states()[KEY["a"]], "ready")

    def test_a_model_that_is_answering_keeps_the_card(self):
        rig = self.make(parallel=True)
        rig.chat(KEY["a"])
        answer = {}
        thread = threading.Thread(target=lambda: answer.update(r=rig.chat(KEY["a"], sleep=1.5)))
        thread.start()
        time.sleep(0.4)
        rig.chat(KEY["b"])
        thread.join(5)
        self.assertEqual(answer["r"][0], 200)
        self.assertFalse(self.on_card(rig, KEY["b"]))
        self.assertEqual(rig.states()[KEY["a"]], "ready")

    def test_a_model_pinned_to_the_cpu_stops_nobody(self):
        import model_settings
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "s.json")
        settings = model_settings.ModelSettings(path)
        holder = {}

        def card():
            used = sum(20.0 for w in holder["rig"].pool.snapshot() if w.card and w.state in ("starting", "ready", "stopping"))
            return 24.0 - used, 24.0
        policy = gpu.GpuPolicy("auto", reserve_gb=2.0, min_gb=6.0, probe=card)
        policy.memory = card
        rig = CapturingRig(self, [entry("text-a"), entry("text-b")], gpu=policy, settings=settings,
                           pool_kw={"card_min_idle": 0.0, "card_settle": 1.0})
        holder["rig"] = rig
        rig.chat(KEY["a"])
        model_settings.update(path, KEY["b"], gpu="cpu")
        rig.chat(KEY["b"])
        self.assertFalse(self.on_card(rig, KEY["b"]))
        self.assertEqual(rig.states()[KEY["a"]], "ready")

    def test_the_best_victim_is_the_one_that_costs_least_to_bring_back(self):
        holder = {}

        def card():                                           # room for two on the card; a third needs one of them to go
            used = sum(11.0 for w in holder["rig"].pool.snapshot() if w.card and w.state in ("starting", "ready", "stopping"))
            return 24.0 - used, 24.0
        policy = gpu.GpuPolicy("auto", reserve_gb=2.0, min_gb=6.0, probe=card)
        policy.memory = card
        rig = CapturingRig(self, [entry("text-a"), entry("text-b"), entry("text-c")], gpu=policy,
                           pool_kw={"card_min_idle": 0.0, "card_settle": 2.0})
        holder["rig"] = rig
        rig.chat(KEY["a"])
        rig.chat(KEY["b"])
        by_id = {item["id"]: item["key"] for item in rig.entries}
        rig.pool.state.data[by_id[KEY["a"]]]["loads"] = [250.0]                           # a came off a slow drive
        rig.pool.state.data[by_id[KEY["b"]]]["loads"] = [5.0]
        time.sleep(0.1)
        rig.chat(KEY["c"])
        self.assertTrue(self.on_card(rig, KEY["c"]))
        states = rig.states()
        self.assertEqual((states[KEY["a"]], states[KEY["b"]], states[KEY["c"]]), ("ready", "idle", "ready"))


class PressureTest(unittest.TestCase):
    """relieve() is called directly: the reaper calls it once a second."""

    def rig(self, free, watch=False, **kw):
        """Two models loaded and idle. Memory reads `free` only once they are: the reaper's own loop is off unless asked for, so
        the tests are the ones that call relieve()."""
        self.free = 64 * GB
        pool_kw = {"pressure_floor": 2 * GB, "pressure_min_idle": 0.0, "pressure_cooldown": 0.3, "watch_pressure": watch}
        pool_kw.update(kw)
        rig = Rig(self, [entry("text-a"), entry("text-b"), entry("text-c")], memory=lambda: self.free, pool_kw=pool_kw)
        for name in ("a", "b"):
            rig.chat(KEY[name])
        self.free = free
        return rig

    def test_nothing_happens_while_there_is_enough_memory(self):
        rig = self.rig(free=8 * GB)
        self.assertIsNone(rig.pool.relieve())
        self.assertEqual(rig.states(), {KEY["a"]: "ready", KEY["b"]: "ready", KEY["c"]: "idle"})

    def test_when_memory_runs_low_the_best_idle_model_is_stopped_and_the_reason_is_said(self):
        rig = self.rig(free=int(1.2 * GB))
        for item in rig.entries[:2]:
            rig.pool.state.data[item["key"]]["loads"] = [200.0 if item["id"] == KEY["a"] else 3.0]
        time.sleep(0.05)
        stopped = rig.pool.relieve()
        self.assertEqual(stopped, KEY["b"])                                               # a is the slow one to bring back
        self.assertEqual(rig.states()[KEY["b"]], "idle")
        self.assertEqual(rig.states()[KEY["a"]], "ready")
        self.assertEqual(rig.pool.pressure["stops"], 1)
        self.assertEqual(rig.pool.pressure["last"]["model"], KEY["b"])
        self.assertEqual(rig.pool.pressure["last"]["free_bytes"], int(1.2 * GB))

    def test_one_stop_then_a_wait_for_the_memory_to_show_up(self):
        rig = self.rig(free=int(1.2 * GB))
        self.assertIsNotNone(rig.pool.relieve())
        self.assertIsNone(rig.pool.relieve())                                             # still low, but the cooldown has not passed
        self.assertEqual(sum(1 for s in rig.states().values() if s == "ready"), 1)
        time.sleep(0.35)
        self.assertIsNotNone(rig.pool.relieve())                                          # still low after the wait: the next one goes
        self.assertEqual(sum(1 for s in rig.states().values() if s == "ready"), 0)
        time.sleep(0.35)
        self.assertIsNone(rig.pool.relieve())                                             # nothing left to stop is not an error

    def test_a_model_that_is_answering_is_never_stopped(self):
        rig = self.rig(free=int(1.2 * GB), pressure_cooldown=0.0)
        answer = {}
        thread = threading.Thread(target=lambda: answer.update(r=rig.chat(KEY["a"], sleep=1.5)))
        thread.start()
        time.sleep(0.4)
        rig.pool.relieve()
        rig.pool.relieve()
        thread.join(5)
        self.assertEqual(answer["r"][0], 200)                                             # a was not cut off
        self.assertEqual(rig.states()[KEY["a"]], "ready")
        self.assertEqual(rig.states()[KEY["b"]], "idle")                                  # the idle one was the one to go

    def test_a_model_used_a_moment_ago_waits_unless_memory_is_nearly_gone(self):
        rig = self.rig(free=int(1.5 * GB), pressure_min_idle=60.0)
        self.assertIsNone(rig.pool.relieve())                                             # both were used seconds ago
        self.free = int(0.5 * GB)                                                         # under half the floor: urgent
        self.assertIsNotNone(rig.pool.relieve())

    def test_it_is_off_unless_asked_for_and_does_nothing_when_memory_cannot_be_read(self):
        rig = self.rig(free=int(0.1 * GB), pressure_floor=None)
        self.assertIsNone(rig.pool.relieve())
        rig2 = self.rig(free=None)
        self.assertIsNone(rig2.pool.relieve())
        self.assertEqual(rig.states()[KEY["a"]], "ready")

    def test_the_reapers_own_loop_does_it(self):
        rig = self.rig(free=int(1.2 * GB), watch=True)
        rig.pool._wake.set()
        end = time.monotonic() + 8
        while time.monotonic() < end and rig.pool.pressure["stops"] == 0:
            time.sleep(0.1)
        self.assertEqual(rig.pool.pressure["stops"], 1)

    def test_health_shows_the_floor_and_what_it_did(self):
        free = {"now": 8 * GB}
        rig = Rig(self, [entry("text-a")], api_key="k", memory=lambda: free["now"],
                  pool_kw={"pressure_floor": 2 * GB, "pressure_min_idle": 0.0, "watch_pressure": False})
        auth = {"Authorization": "Bearer k"}
        health = lambda: rig.call("GET", "/health", headers=auth)[3]["router"]["memory_pressure"]      # noqa: E731
        rig.call("POST", "/v1/chat/completions", {"model": KEY["a"], "messages": []}, auth)
        self.assertEqual(health(), {"floor_bytes": 2 * GB, "stops": 0, "last": None})
        free["now"] = int(1.0 * GB)
        rig.pool.relieve()
        found = health()
        self.assertEqual((found["stops"], found["last"]["model"], found["last"]["free_bytes"]), (1, KEY["a"], GB))

    def test_the_default_floor_is_the_larger_of_a_few_gb_and_a_share_of_the_total(self):
        original = router.total_memory
        try:
            router.total_memory = lambda: 64 * GB
            self.assertEqual(router.pressure_floor(), int(0.03 * 64 * GB))
            router.total_memory = lambda: 16 * GB
            self.assertEqual(router.pressure_floor(), int(1.5 * GB))
            router.total_memory = lambda: None
            self.assertIsNone(router.pressure_floor())
            os.environ["COLI_MEMORY_FLOOR_GB"] = "3"
            self.assertEqual(router.pressure_floor(), 3 * GB)
            os.environ["COLI_MEMORY_FLOOR_GB"] = "0"
            self.assertIsNone(router.pressure_floor())
        finally:
            router.total_memory = original
            os.environ.pop("COLI_MEMORY_FLOOR_GB", None)


class PeakTest(unittest.TestCase):
    def test_a_model_that_grows_while_it_answers_is_remembered_at_its_peak(self):
        reading = {"bytes": 4 * GB}
        rig = Rig(self, [entry("text-a", size=2 * GB)], measure=lambda pid, job: reading["bytes"], pool_kw={"peak_sample": 0.0})
        rig.chat(KEY["a"])
        key = rig.entries[0]["key"]
        self.assertEqual(rig.pool.state.resident(key), 4 * GB)                           # what it took after loading
        answer = {}
        thread = threading.Thread(target=lambda: answer.update(r=rig.chat(KEY["a"], sleep=1.0)))
        thread.start()
        time.sleep(0.3)
        reading["bytes"] = 7 * GB                                                         # its caches filled while it worked
        rig.pool.sample_peaks()
        thread.join(5)
        self.assertEqual(rig.pool.state.peak(key), 7 * GB)
        self.assertEqual(rig.pool.estimate(rig.entries[0]), 7 * GB)                      # the next "will it fit" uses that
        self.assertEqual(rig.pool.get(key).resident, 7 * GB)                             # and so does the planner for the running one

    def test_an_idle_model_is_not_sampled_and_a_slow_sample_rate_is_respected(self):
        reading = {"bytes": 4 * GB}
        rig = Rig(self, [entry("text-a")], measure=lambda pid, job: reading["bytes"], pool_kw={"peak_sample": 3600.0})
        rig.chat(KEY["a"])
        key = rig.entries[0]["key"]
        reading["bytes"] = 9 * GB
        rig.pool.sample_peaks()                                                           # nothing is answering: nothing to read
        self.assertIsNone(rig.pool.state.peak(key))

    def test_the_estimate_falls_back_as_before_without_a_peak(self):
        rig = Rig(self, [entry("text-a", size=3 * GB)])
        self.assertEqual(rig.pool.estimate(rig.entries[0]), 3 * GB)                      # its size on disk, nothing seen yet


if __name__ == "__main__":
    unittest.main()
