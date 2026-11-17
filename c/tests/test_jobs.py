"""The job lane (jobs.py): order, states, cancel in every state, ids, the records on disk and what a restart does to them."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jobs  # noqa: E402


def wait_for(condition, seconds=5.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


class Lane(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "jobs.jsonl")
        self.finished = []
        self.manager = self.make()

    def make(self, **kw):
        manager = jobs.JobManager(jobs.JobStore(self.path), on_finish=self.finished.append, **kw)
        self.addCleanup(manager.close)
        return manager

    def submit(self, run, job_id=None, kind="image", model="m", request=None):
        return self.manager.submit(kind, model, "/v1/images/generations", request or {"prompt": "a"}, run, job_id)

    def status(self, job_id):
        return self.manager.get(job_id)["status"]


class StatesTest(Lane):
    def test_a_job_that_returns_succeeds_with_its_result(self):
        snapshot, created = self.submit(lambda ctx: {"file": "a.png"})
        self.assertTrue(created)
        self.assertEqual((snapshot["object"], snapshot["kind"], snapshot["model"]), ("job", "image", "m"))
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "succeeded"))
        done = self.manager.get(snapshot["id"])
        self.assertEqual(done["result"], {"file": "a.png"})
        self.assertIsNotNone(done["started"])
        self.assertIsNotNone(done["finished"])
        self.assertNotIn("error", done)
        self.assertNotIn("position", done)

    def test_a_job_that_raises_a_job_error_fails_with_its_words(self):
        def run(ctx):
            raise jobs.JobError(503, "The image engine is not running.", "engine_unavailable")
        snapshot, _ = self.submit(run)
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "failed"))
        self.assertEqual(self.manager.get(snapshot["id"])["error"],
                         {"code": "engine_unavailable", "message": "The image engine is not running."})

    def test_a_bug_in_a_job_fails_it_and_the_lane_goes_on(self):
        first, _ = self.submit(lambda ctx: 1 / 0)
        second, _ = self.submit(lambda ctx: {"ok": True})
        self.assertTrue(wait_for(lambda: self.status(second["id"]) == "succeeded"))
        self.assertEqual(self.status(first["id"]), "failed")
        self.assertIn("division by zero", self.manager.get(first["id"])["error"]["message"])

    def test_jobs_run_one_at_a_time_in_the_order_they_came(self):
        order, release = [], threading.Event()

        def run(name):
            def go(ctx):
                order.append(f"start {name}")
                release.wait(5)
                order.append(f"end {name}")
                return {}
            return go
        ids = [self.submit(run(name))[0]["id"] for name in "abc"]
        self.assertTrue(wait_for(lambda: order == ["start a"]))
        self.assertEqual([self.manager.get(i).get("position") for i in ids[1:]], [1, 2])
        release.set()
        self.assertTrue(wait_for(lambda: self.status(ids[2]) == "succeeded"))
        self.assertEqual(order, ["start a", "end a", "start b", "end b", "start c", "end c"])

    def test_progress_is_shown_while_running_and_gone_after(self):
        seen, release = threading.Event(), threading.Event()

        def run(ctx):
            ctx.progress({"stage": "denoise", "step": 3, "steps": 8})
            seen.set()
            release.wait(5)
            return {}
        snapshot, _ = self.submit(run)
        self.assertTrue(seen.wait(5))
        self.assertEqual(self.manager.get(snapshot["id"])["progress"], {"stage": "denoise", "step": 3, "steps": 8})
        release.set()
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "succeeded"))
        self.assertNotIn("progress", self.manager.get(snapshot["id"]))

    def test_the_finish_callback_gets_the_final_snapshot_once(self):
        snapshot, _ = self.submit(lambda ctx: {"n": 1})
        self.assertTrue(wait_for(lambda: len(self.finished) == 1))
        self.assertEqual((self.finished[0]["id"], self.finished[0]["status"]), (snapshot["id"], "succeeded"))


class CancelTest(Lane):
    def blocker(self):
        started, release = threading.Event(), threading.Event()

        def run(ctx):
            started.set()
            release.wait(5)
            return {}
        return run, started, release

    def test_a_waiting_job_is_removed_and_never_runs(self):
        run, started, release = self.blocker()
        first, _ = self.submit(run)
        ran = []
        second, _ = self.submit(lambda ctx: ran.append(1))
        self.assertTrue(started.wait(5))
        self.assertEqual(self.manager.cancel(second["id"]), (True, "cancelled"))
        self.assertEqual(self.status(second["id"]), "cancelled")
        release.set()
        self.assertTrue(wait_for(lambda: self.status(first["id"]) == "succeeded"))
        time.sleep(0.1)
        self.assertEqual(ran, [])
        self.assertTrue(any(f["id"] == second["id"] and f["status"] == "cancelled" for f in self.finished))

    def test_a_running_job_is_interrupted_through_what_it_registered(self):
        stopped = threading.Event()

        def run(ctx):
            ctx.on_abort(stopped.set)                       # as the router registers "close the connection to the model"
            stopped.wait(5)
            raise jobs.JobError(502, "The model went away.", "worker_unavailable")
        snapshot, _ = self.submit(run)
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "running"))
        time.sleep(0.05)
        self.assertEqual(self.manager.cancel(snapshot["id"]), (True, "cancelling"))
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "cancelled"))
        self.assertNotIn("error", self.manager.get(snapshot["id"]))         # the failure the interrupt caused is not the story

    def test_a_cancel_that_arrives_before_the_job_registered_its_interrupt_still_interrupts(self):
        gate, stopped = threading.Event(), threading.Event()

        def run(ctx):
            gate.wait(5)
            ctx.on_abort(stopped.set)                       # registered late: called at once
            stopped.wait(5)
            return {}
        snapshot, _ = self.submit(run)
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "running"))
        self.manager.cancel(snapshot["id"])
        gate.set()
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "cancelled"))
        self.assertTrue(stopped.is_set())

    def test_cancel_is_the_same_call_in_every_state_and_never_an_error(self):
        snapshot, _ = self.submit(lambda ctx: {})
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "succeeded"))
        self.assertEqual(self.manager.cancel(snapshot["id"]), (False, "succeeded"))
        self.assertEqual(self.manager.cancel("job_nobody_knows"), (False, None))
        self.assertEqual(self.manager.cancel(snapshot["id"]), (False, "succeeded"))

    def test_a_cancel_does_not_leak_onto_the_next_job(self):
        run, started, release = self.blocker()
        first, _ = self.submit(run)
        self.assertTrue(started.wait(5))
        self.manager.cancel(first["id"])
        release.set()
        seen = []
        second, _ = self.submit(lambda ctx: seen.append(ctx.cancelled.is_set()) or {})
        self.assertTrue(wait_for(lambda: self.status(second["id"]) == "succeeded"))
        self.assertEqual(seen, [False])
        self.assertEqual(self.status(first["id"]), "cancelled")


class IdsAndLimitsTest(Lane):
    def test_a_client_id_makes_a_resubmit_the_same_job(self):
        runs = []
        first, created = self.submit(lambda ctx: runs.append(1) or {}, job_id="my-render_1")
        again, created_again = self.submit(lambda ctx: runs.append(2) or {}, job_id="my-render_1")
        self.assertEqual((first["id"], created, again["id"], created_again), ("my-render_1", True, "my-render_1", False))
        self.assertTrue(wait_for(lambda: self.status("my-render_1") == "succeeded"))
        time.sleep(0.1)
        self.assertEqual(runs, [1])
        _, still = self.submit(lambda ctx: runs.append(3) or {}, job_id="my-render_1")      # also after it finished
        self.assertFalse(still)

    def test_a_bad_client_id_is_refused(self):
        for bad in ("", "has space", "a/b", "../x", "x" * 65, "é"):
            with self.assertRaises(jobs.JobError, msg=bad) as caught:
                self.submit(lambda ctx: {}, job_id=bad)
            self.assertEqual(caught.exception.status, 400)

    def test_too_many_waiting_is_a_429_and_room_comes_back(self):
        manager = self.make(max_pending=2)
        self.manager = manager
        release = threading.Event()
        a, _ = self.submit(lambda ctx: release.wait(5) and {})
        self.submit(lambda ctx: {})
        with self.assertRaises(jobs.JobError) as caught:
            self.submit(lambda ctx: {})
        self.assertEqual((caught.exception.status, caught.exception.code), (429, "queue_full"))
        release.set()
        self.assertTrue(wait_for(lambda: self.status(a["id"]) == "succeeded"))
        self.assertTrue(wait_for(lambda: self.manager.running is None and not self.manager.queue))
        self.submit(lambda ctx: {})

    def test_the_list_is_newest_first_with_a_status_filter_and_pages(self):
        ids = []
        for n in range(5):
            snapshot, _ = self.submit(lambda ctx, n=n: {"n": n} if n != 1 else (_ for _ in ()).throw(jobs.JobError(500, "x", "boom")))
            ids.append(snapshot["id"])
            self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) in jobs.FINAL))
            time.sleep(0.002)
        found = self.manager.listing()
        self.assertEqual([j["id"] for j in found["data"]], ids[::-1])
        self.assertEqual((found["total"], found["has_more"]), (5, False))
        failed = self.manager.listing(status="failed")
        self.assertEqual([j["id"] for j in failed["data"]], [ids[1]])
        page = self.manager.listing(limit=2, offset=1)
        self.assertEqual(([j["id"] for j in page["data"]], page["has_more"]), (ids[::-1][1:3], True))

    def test_only_finished_jobs_are_forgotten_past_the_limit(self):
        manager = self.make(keep=3)
        self.manager = manager
        release = threading.Event()
        pinned, _ = self.submit(lambda ctx: release.wait(5) and {})
        for _ in range(5):
            self.submit(lambda ctx: {})
        self.assertIn(pinned["id"], [j["id"] for j in manager.listing(limit=200)["data"]])
        release.set()
        self.assertTrue(wait_for(lambda: manager.running is None and not manager.queue))


class RecordsTest(Lane):
    def lines(self):
        return [json.loads(line) for line in Path(self.path).read_text(encoding="utf-8").splitlines()]

    def test_every_change_of_state_is_a_line_and_progress_is_not(self):
        release = threading.Event()

        def run(ctx):
            ctx.progress({"step": 1})
            release.wait(5)
            return {"file": "a.png"}
        snapshot, _ = self.submit(run, request={"prompt": "a fox"})
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "running"))
        release.set()
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "succeeded"))
        mine = [r for r in self.lines() if r["id"] == snapshot["id"]]
        self.assertEqual([r["status"] for r in mine], ["queued", "running", "succeeded"])
        self.assertTrue(all("progress" not in r for r in mine))
        self.assertEqual(mine[-1]["result"], {"file": "a.png"})
        self.assertEqual(mine[0]["request"], {"prompt": "a fox"})

    def test_the_list_comes_back_after_a_restart(self):
        snapshot, _ = self.submit(lambda ctx: {"file": "a.png"})
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "succeeded"))
        self.manager.close()
        again = jobs.JobManager(jobs.JobStore(self.path))
        self.addCleanup(again.close)
        back = again.get(snapshot["id"])
        self.assertEqual((back["status"], back["result"]), ("succeeded", {"file": "a.png"}))

    def test_a_job_that_was_waiting_or_running_when_the_router_stopped_comes_back_failed_and_says_why(self):
        Path(self.path).write_text("\n".join(json.dumps(r) for r in (
            {"id": "w", "kind": "image", "model": "m", "path": "/p", "status": "queued", "created": 1, "request": {}},
            {"id": "r", "kind": "image", "model": "m", "path": "/p", "status": "queued", "created": 2, "request": {}},
            {"id": "r", "kind": "image", "model": "m", "path": "/p", "status": "running", "created": 2, "started": 3, "request": {}},
            {"id": "d", "kind": "music", "model": "m", "path": "/p", "status": "succeeded", "created": 0, "request": {}})) + "\n",
            encoding="utf-8")
        again = jobs.JobManager(jobs.JobStore(self.path))
        self.addCleanup(again.close)
        self.assertEqual(again.get("w")["error"]["code"], "router_restarted")
        self.assertIn("waiting", again.get("w")["error"]["message"])
        self.assertIn("running", again.get("r")["error"]["message"])
        self.assertEqual(again.get("d")["status"], "succeeded")
        self.assertEqual([j["id"] for j in again.listing()["data"]], ["r", "w", "d"])
        stored = {r["id"]: r["status"] for r in self.lines()}                 # and that is written down, so it says it once
        self.assertEqual(stored, {"w": "failed", "r": "failed", "d": "succeeded"})

    def test_a_file_with_garbage_in_it_still_loads_what_is_good(self):
        Path(self.path).write_text('not json\n{"id": 5}\n[1]\n{"id": "ok", "kind": "image", "model": "m", "path": "/p", '
                                   '"status": "succeeded", "created": 1, "request": {}}\n\n', encoding="utf-8")
        again = jobs.JobManager(jobs.JobStore(self.path))
        self.addCleanup(again.close)
        self.assertEqual([j["id"] for j in again.listing()["data"]], ["ok"])

    def test_a_file_that_grew_is_cut_back_to_what_is_kept(self):
        store = jobs.JobStore(self.path, keep=3)
        for n in range(40):
            store.write({"id": f"j{n % 5}", "kind": "image", "model": "m", "path": "/p", "status": "succeeded", "created": n,
                         "request": {}})
        kept = store.load()
        self.assertEqual(len(kept), 3)
        self.assertEqual(len(self.lines()), 3)

    def test_a_disk_that_cannot_be_written_does_not_stop_a_job(self):
        blocker = os.path.join(self.tmp.name, "file")
        Path(blocker).write_text("x")
        manager = jobs.JobManager(jobs.JobStore(os.path.join(blocker, "inside", "jobs.jsonl")))
        self.addCleanup(manager.close)
        snapshot, _ = manager.submit("image", "m", "/p", {}, lambda ctx: {"ok": 1})
        self.assertTrue(wait_for(lambda: manager.get(snapshot["id"])["status"] == "succeeded"))

    def test_closing_cancels_what_is_running_and_stops_the_lane(self):
        stopped = threading.Event()

        def run(ctx):
            ctx.on_abort(stopped.set)
            stopped.wait(5)
            return {}
        snapshot, _ = self.submit(run)
        self.assertTrue(wait_for(lambda: self.status(snapshot["id"]) == "running"))
        time.sleep(0.05)
        self.manager.close()
        self.assertTrue(stopped.is_set())
        with self.assertRaises(jobs.JobError):
            self.submit(lambda ctx: {})


if __name__ == "__main__":
    unittest.main()
