"""A model that holds the graphics card gives it back and keeps running, instead of being stopped (router.Pool.demote)."""
import os
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpu  # noqa: E402
from test_gpu import CapturingRig  # noqa: E402
from test_router import entry  # noqa: E402

KEY = {"a": "text-a-colibri", "b": "text-b-colibri"}


class DemoteTest(unittest.TestCase):
    """Two models, a card with room for one: each on the card takes 20 of its 24 GB."""

    def make(self, answer="ok", **pool_kw):
        holder = {}

        def card():
            used = sum(20.0 for w in holder["rig"].pool.snapshot() if w.card and w.state in ("starting", "ready", "stopping", "demoting"))
            return 24.0 - used, 24.0
        policy = gpu.GpuPolicy("auto", reserve_gb=2.0, min_gb=6.0, probe=card)
        policy.memory = card
        kw = {"card_min_idle": 0.0, "card_settle": 2.0}
        kw.update(pool_kw)
        env = {KEY["a"]: {"FAKE_DEMOTE": answer}} if answer else {}
        rig = CapturingRig(self, [entry("text-a"), entry("text-b")], gpu=policy, pool_kw=kw, env=env,
                           parallel=kw.pop("parallel", False))
        holder["rig"] = rig
        return rig

    def on_card(self, rig, name):
        return [env["COLI_VULKAN"] for model, env in rig.started if model == name][-1] == "1"

    def worker(self, rig, name):
        return rig.pool.workers[next(item["key"] for item in rig.entries if item["id"] == name)]

    def test_the_holder_gives_the_card_back_and_keeps_running(self):
        rig = self.make("ok")
        first = rig.chat(KEY["a"])[3]["pid"]
        self.assertTrue(self.on_card(rig, KEY["a"]))
        rig.chat(KEY["b"])
        self.assertTrue(self.on_card(rig, KEY["b"]))                                     # the card was free for it
        self.assertEqual(rig.states(), {KEY["a"]: "ready", KEY["b"]: "ready"})           # a was not stopped
        self.assertEqual([kind for _, kind, name in rig.events() if kind == "demote"], ["demote"])
        again = rig.chat(KEY["a"])[3]["pid"]
        self.assertEqual(again, first)                                                   # the same process answers: no reload

    def test_the_pool_counts_it_as_a_cpu_model_from_then_on(self):
        rig = self.make("ok")
        rig.chat(KEY["a"])
        rig.chat(KEY["b"])
        worker = self.worker(rig, KEY["a"])
        self.assertFalse(worker.card)
        self.assertTrue(worker.demoted)
        self.assertEqual(worker.gpu.mode, "cpu")
        self.assertIn("gave the graphics card back", worker.gpu.why)
        status, _, _, body = rig.call("GET", "/v1/models/installed")
        item = next(i for i in body["data"] if i["id"] == KEY["a"])
        self.assertEqual(item["state"], "ready")

    def test_health_says_which_worker_was_demoted(self):
        rig = self.make("ok")
        rig.chat(KEY["a"])
        rig.chat(KEY["b"])
        view = {w["id"]: w for w in self.loaded(rig)}
        self.assertTrue(view[KEY["a"]]["demoted"])
        self.assertFalse(view[KEY["b"]]["demoted"])
        self.assertEqual(view[KEY["a"]]["gpu"]["mode"], "cpu")

    def loaded(self, rig):
        # the same view /health gives, without a key
        handler = type("H", (), {"server": rig.server})()
        return [router_worker_view(handler, w) for w in rig.pool.snapshot()]

    def test_an_engine_that_cannot_is_stopped_as_before(self):
        rig = self.make("no")
        rig.chat(KEY["a"])
        rig.chat(KEY["b"])
        self.assertTrue(self.on_card(rig, KEY["b"]))
        self.assertEqual(rig.states(), {KEY["a"]: "idle", KEY["b"]: "ready"})

    def test_an_older_worker_with_no_such_route_is_stopped_as_before(self):
        rig = self.make(None)                                                            # FAKE_DEMOTE unset: a 404
        rig.chat(KEY["a"])
        rig.chat(KEY["b"])
        self.assertTrue(self.on_card(rig, KEY["b"]))
        self.assertEqual(rig.states(), {KEY["a"]: "idle", KEY["b"]: "ready"})

    def test_a_request_that_got_there_first_keeps_the_card_with_its_model(self):
        rig = self.make("busy")
        rig.chat(KEY["a"])
        rig.chat(KEY["b"])
        self.assertFalse(self.on_card(rig, KEY["b"]))                                    # the CPU this time
        self.assertEqual(rig.states(), {KEY["a"]: "ready", KEY["b"]: "ready"})
        self.assertTrue(self.worker(rig, KEY["a"]).card)                                 # a still holds it

    def test_it_can_be_turned_off(self):
        rig = self.make("ok", card_demote=False)
        rig.chat(KEY["a"])
        rig.chat(KEY["b"])
        self.assertEqual([kind for _, kind, _ in rig.events() if kind == "demote"], [])  # never asked
        self.assertEqual(rig.states(), {KEY["a"]: "idle", KEY["b"]: "ready"})

    def test_a_request_for_the_holder_waits_while_it_gives_the_card_back(self):
        rig = self.make("ok")
        rig.chat(KEY["a"])
        worker = self.worker(rig, KEY["a"])
        worker.state = "demoting"                                                        # as the router leaves it meanwhile
        answer = {}
        thread = threading.Thread(target=lambda: answer.update(r=rig.chat(KEY["a"])))
        thread.start()
        time.sleep(0.5)
        self.assertTrue(thread.is_alive())                                               # held, not sent to a half-demoted engine
        with rig.pool.cv:
            worker.state = "ready"
            worker.settled.set()
            rig.pool.cv.notify_all()
        thread.join(10)
        self.assertEqual(answer["r"][0], 200)


def router_worker_view(handler, worker):
    import router
    return router.RouterHandler.worker_view(handler, worker)


if __name__ == "__main__":
    unittest.main()
