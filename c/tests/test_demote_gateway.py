"""The gateway's half of giving the graphics card back: the DEMOTE frame, the scheduler hold, POST /v1/demote."""
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai_server import APIError, APIServer, Engine, GenerationScheduler  # noqa: E402
from test_openai_server import READY, BlockingStream, FakeProcess  # noqa: E402


def engine_with(caps, answer):
    """An Engine over a fake process that said `caps` in its handshake and answers a DEMOTE with `answer` (None: says nothing)."""
    def respond(process, frame):
        if frame == b"DEMOTE demote\n" and answer is not None:
            process.stdout.feed(answer)

    process = FakeProcess(respond)
    process.stdout = BlockingStream(READY + b"CAPS " + caps.encode() + b"\nSTAT 0 0 0 0\n")
    with patch("openai_server.subprocess.Popen", return_value=process):
        engine = Engine("qwen36", "model")
    return engine, process


class EngineDemoteTest(unittest.TestCase):
    def test_an_engine_that_said_demote_is_asked_and_answers(self):
        engine, process = engine_with("vision=0 demote=1", b"DEMOTED demote 1 ok\n")
        self.addCleanup(engine.close)
        self.assertTrue(engine.can_demote)
        self.assertEqual(engine.demote(timeout=5), (True, "ok"))
        self.assertEqual(process.writes, [b"DEMOTE demote\n"])
        self.assertFalse(engine.can_demote)                                           # nothing left on the card to give
        done, why = engine.demote(timeout=5)
        self.assertFalse(done)
        self.assertEqual(process.writes, [b"DEMOTE demote\n"])                        # not sent again

    def test_a_refusal_carries_its_reason_and_leaves_it_able_to_try_again(self):
        engine, _ = engine_with("demote=1", b"DEMOTED demote 0 a request is running\n")
        self.addCleanup(engine.close)
        self.assertEqual(engine.demote(timeout=5), (False, "a request is running"))
        self.assertTrue(engine.can_demote)

    def test_an_engine_that_did_not_say_demote_is_never_sent_the_frame(self):
        engine, process = engine_with("vision=0", b"DEMOTED demote 1 ok\n")
        self.addCleanup(engine.close)
        self.assertFalse(engine.can_demote)
        self.assertFalse(engine.demote(timeout=5)[0])
        self.assertEqual(process.writes, [])

    def test_an_engine_that_does_not_answer_is_not_waited_for_forever(self):
        engine, _ = engine_with("demote=1", None)
        self.addCleanup(engine.close)
        started = time.monotonic()
        self.assertEqual(engine.demote(timeout=0.3), (False, "the engine did not answer"))
        self.assertLess(time.monotonic() - started, 3)


class ExclusiveTest(unittest.TestCase):
    def test_every_slot_is_held_while_nothing_runs_and_nothing_is_admitted_meanwhile(self):
        scheduler = GenerationScheduler(max_queue=0, queue_timeout=1, capacity=2)
        with scheduler.exclusive() as idle:
            self.assertTrue(idle)
            with self.assertRaises(APIError):                                          # no slot, no queue: refused at once
                with scheduler.admit():
                    pass
        with scheduler.admit() as (_, slot):                                           # released afterwards
            self.assertIn(slot, (0, 1))

    def test_it_is_refused_while_a_request_runs(self):
        scheduler = GenerationScheduler(capacity=2)
        with scheduler.admit():
            with scheduler.exclusive() as idle:
                self.assertFalse(idle)
            self.assertEqual(len(scheduler.idle_slots()), 1)                           # the one slot still free is still free

    def test_it_does_not_wait_for_a_request_to_finish(self):
        scheduler = GenerationScheduler()
        release = threading.Event()

        def hold():
            with scheduler.admit():
                release.wait(5)
        thread = threading.Thread(target=hold, daemon=True)
        thread.start()
        time.sleep(0.2)
        started = time.monotonic()
        with scheduler.exclusive() as idle:
            self.assertFalse(idle)
        self.assertLess(time.monotonic() - started, 1)
        release.set()


class FakeEngine:
    """Enough of an engine for the model routes: answers a demotion as told."""

    def __init__(self, result):
        self.result, self.asked = result, 0

    def demote(self):
        self.asked += 1
        return self.result


class DemoteRouteTest(unittest.TestCase):
    def serve(self, engine):
        self.engine = engine
        self.server = APIServer(("127.0.0.1", 0), engine, "test-model")
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def post(self):
        request = Request(self.base + "/v1/demote", data=b"{}", headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())

    def test_an_idle_engine_is_asked_and_the_answer_is_passed_on(self):
        self.serve(FakeEngine((True, "ok")))
        self.assertEqual(self.post(), (200, {"demoted": True, "reason": None}))
        self.assertEqual(self.engine.asked, 1)

    def test_a_refusal_says_why(self):
        self.serve(FakeEngine((False, "it is not on the card")))
        self.assertEqual(self.post(), (200, {"demoted": False, "reason": "it is not on the card"}))

    def test_a_running_request_is_never_waited_for(self):
        self.serve(FakeEngine((True, "ok")))
        with self.server.scheduler.admit():
            status, body = self.post()
        self.assertEqual(status, 200)
        self.assertEqual(body["demoted"], False)
        self.assertIn("request is running", body["reason"])
        self.assertEqual(self.engine.asked, 0)                                            # the engine was not asked at all


if __name__ == "__main__":
    unittest.main()
