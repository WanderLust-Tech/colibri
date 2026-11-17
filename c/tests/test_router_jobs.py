"""Background jobs through the router: Prefer: respond-async, GET /v1/jobs, POST /v1/jobs/cancel (jobs.py + router.py)."""
import json
import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jobs  # noqa: E402
import router  # noqa: E402
from test_acestep_gateway import GatewayHarness as MusicHarness  # noqa: E402
from test_library import Folders  # noqa: E402
from test_qwen_image_gateway import GatewayHarness as ImageHarness, MODEL_ID as IMAGE_MODEL_ID  # noqa: E402
from test_router import Rig, entry  # noqa: E402

KEY = "k"
AUTH = {"Authorization": f"Bearer {KEY}"}
ASYNC = {**AUTH, "Prefer": "respond-async"}
IMAGE, MUSIC, TEXT = "image-b-colibri", "audio-c-colibri", "text-a-colibri"


def wait_for(condition, seconds=15.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        found = condition()
        if found:
            return found
        time.sleep(0.02)
    return None


class JobRig(unittest.TestCase):
    keep_files = True

    def setUp(self):
        self.dirs = Folders().make(self)
        env = {} if self.keep_files else {"FAKE_KEEP": "0"}
        self.rig = Rig(self, [entry("text-a"), entry("image-b", "image"), entry("audio-c", "audio")], api_key=KEY,
                       env={IMAGE: env, MUSIC: env})

    def submit(self, path="/v1/images/generations", model=IMAGE, headers=None, **body):
        body = {"model": model, "prompt": "a red fox", "job_demo": True, **body}
        return self.rig.call("POST", path, body, {**ASYNC, **(headers or {})})

    def job(self, job_id):
        status, _h, _raw, body = self.rig.call("GET", f"/v1/jobs/{job_id}", headers=AUTH)
        self.assertEqual(status, 200, body)
        return body

    def until(self, job_id, *states):
        return wait_for(lambda: self.job(job_id) if self.job(job_id)["status"] in states else None)


class SubmitTest(JobRig):
    def test_it_answers_202_with_a_job_at_once_and_the_picture_is_made_without_anyone_waiting(self):
        status, headers, _raw, job = self.submit(sleep=0.1, steps=3, seed=7)
        self.assertEqual(status, 202, job)
        self.assertEqual(headers["Location"], f"/v1/jobs/{job['id']}")
        self.assertEqual(headers["Preference-Applied"], "respond-async")
        self.assertEqual((job["object"], job["kind"], job["model"], job["status"]), ("job", "image", IMAGE, job["status"]))
        self.assertIn(job["status"], ("queued", "running"))
        self.assertEqual(job["request"], {"prompt": "a red fox", "steps": 3, "seed": 7})
        done = self.until(job["id"], "succeeded", "failed")
        self.assertEqual(done["status"], "succeeded", done)
        result = done["result"]
        self.assertEqual((result["kind"], result["seed"], result["steps"], result["width"]), ("images", 7, 3, 8))
        self.assertTrue(result["file"].endswith("-a-red-fox-s7.png"))
        self.assertTrue((self.dirs.images / result["file"]).is_file())
        status, _h, raw, _ = self.rig.call("GET", result["url"], headers=AUTH)                # and the library serves it
        self.assertEqual((status, raw[:4]), (200, b"\x89PNG"))
        self.assertNotIn("cut", [kind for _t, kind, _n in self.rig.events()])

    def test_progress_is_visible_while_it_runs(self):
        _s, _h, _r, job = self.submit(sleep=0.25, steps=6)
        seen = wait_for(lambda: (lambda j: j if j["status"] == "running" and j.get("progress") else None)(self.job(job["id"])))
        self.assertIsNotNone(seen)
        self.assertEqual(seen["progress"]["stage"], "denoise")
        self.assertEqual(seen["progress"]["steps"], 6)
        self.assertTrue(seen["started"])
        self.until(job["id"], "succeeded")

    def test_a_song_is_a_job_too(self):
        _s, _h, _r, job = self.submit("/v1/audio/music", MUSIC, prompt="calm piano", duration=2, seed=3)
        self.assertEqual((job["kind"], job["request"]["prompt"]), ("music", "calm piano"))
        done = self.until(job["id"], "succeeded", "failed")
        self.assertEqual(done["status"], "succeeded", done)
        self.assertEqual((done["result"]["kind"], done["result"]["duration"]), ("music", 0.05))
        self.assertTrue((self.dirs.music / done["result"]["file"]).is_file())

    def test_jobs_come_one_after_another_and_a_waiting_one_says_where_it_is(self):
        _s, _h, _r, first = self.submit(sleep=0.3, steps=3, prompt="first")
        _s, _h, _r, second = self.submit(sleep=0.01, steps=1, prompt="second")
        self.assertEqual(self.job(second["id"]).get("position"), 1)
        done = self.until(second["id"], "succeeded")
        self.assertEqual(done["status"], "succeeded")
        self.assertLessEqual(self.job(first["id"])["finished"], done["finished"])

    def test_a_job_that_the_engine_fails_says_what_it_said(self):
        _s, _h, _r, job = self.submit(fail="out of memory on the card")
        done = self.until(job["id"], "failed", "succeeded")
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["error"], {"code": "engine_error", "message": "out of memory on the card"})
        self.assertNotIn("result", done)

    def test_a_model_that_does_not_exist_is_refused_at_once_not_as_a_failed_job(self):
        status, _h, _r, body = self.submit(model="nothing-colibri")
        self.assertEqual((status, body["error"]["code"]), (404, "model_not_found"))
        self.assertEqual(self.rig.call("GET", "/v1/jobs", headers=AUTH)[3]["total"], 0)

    def test_only_pictures_and_songs_can_go_to_the_background(self):
        status, _h, _r, body = self.rig.call("POST", "/v1/chat/completions", {"model": TEXT, "messages": []}, ASYNC)
        self.assertEqual((status, body["error"]["code"]), (400, "invalid_value"))
        self.assertIn("pictures and songs", body["error"]["message"])

    def test_without_the_preference_nothing_changes(self):
        status, _h, _raw, body = self.rig.call("POST", "/v1/images/generations",
                                               {"model": IMAGE, "prompt": "x", "job_demo": True, "steps": 1}, AUTH)
        self.assertEqual(status, 200)
        self.assertIn("data", body)
        self.assertEqual(self.rig.call("GET", "/v1/jobs", headers=AUTH)[3]["total"], 0)

    def test_a_client_id_makes_a_second_send_the_same_job(self):
        sent = [self.submit(headers={"X-Colibri-Job-Id": "render-1"}, sleep=0.05, steps=2) for _ in range(2)]
        self.assertEqual([s[0] for s in sent], [202, 202])
        self.assertEqual([s[3]["id"] for s in sent], ["render-1", "render-1"])
        self.until("render-1", "succeeded")
        self.assertEqual(len(list(self.dirs.images.iterdir())), 1)
        status, _h, _r, body = self.submit(headers={"X-Colibri-Job-Id": "no good!"})
        self.assertEqual((status, body["error"]["param"]), (400, "job_id"))

    def test_the_prefer_header_may_carry_other_preferences(self):
        status, *_ = self.submit(headers={"Prefer": "wait=10, respond-async"})
        self.assertEqual(status, 202)
        status, _h, _r, _b = self.rig.call("POST", "/v1/images/generations", {"model": IMAGE, "prompt": "x", "job_demo": True, "steps": 1},
                                           {**AUTH, "Prefer": "wait=10"})
        self.assertEqual(status, 200)


class NothingKeptTest(JobRig):
    keep_files = False

    def test_a_worker_that_keeps_nothing_still_leaves_a_job_its_result(self):
        _s, _h, _r, job = self.submit(prompt="a lighthouse", seed=5)
        done = self.until(job["id"], "succeeded", "failed")
        self.assertEqual(done["status"], "succeeded", done)
        self.assertTrue(done["result"]["file"].endswith("-a-lighthouse-s5.png"))
        self.assertTrue((self.dirs.images / done["result"]["file"]).is_file())


class CancelTest(JobRig):
    def cancel(self, job_id):
        status, _h, _r, body = self.rig.call("POST", "/v1/jobs/cancel", {"id": job_id}, AUTH)
        self.assertEqual(status, 200, body)
        return body

    def test_a_running_job_is_stopped_and_the_model_is_told_to_stop_computing(self):
        _s, _h, _r, job = self.submit(sleep=0.2, steps=50)
        self.assertTrue(wait_for(lambda: "start" in [kind for _t, kind, _n in self.rig.events()]))      # the model is working on it
        self.assertEqual(self.cancel(job["id"]), {"id": job["id"], "object": "job.cancel", "cancelled": True, "status": "cancelling"})
        done = self.until(job["id"], "cancelled", "succeeded")
        self.assertEqual(done["status"], "cancelled")
        self.assertTrue(wait_for(lambda: "cut" in [kind for _t, kind, _n in self.rig.events()]))      # the worker saw it go
        self.assertEqual(list(self.dirs.images.iterdir()), [])
        status, _h, _r, body = self.rig.call("POST", "/v1/images/generations",                         # and the lane is free again
                                             {"model": IMAGE, "prompt": "x", "job_demo": True, "steps": 1}, AUTH)
        self.assertEqual(status, 200)

    def test_a_job_cancelled_while_its_model_is_still_loading_never_reaches_the_model(self):
        self.rig = Rig(self, [entry("image-b", "image")], api_key=KEY, env={IMAGE: {"FAKE_LOAD_DELAY": "1.0"}})
        _s, _h, _r, job = self.submit(steps=3)
        self.assertTrue(wait_for(lambda: self.job(job["id"])["status"] == "running"))
        self.cancel(job["id"])
        self.assertEqual(self.until(job["id"], "cancelled", "succeeded")["status"], "cancelled")
        time.sleep(1.5)
        self.assertNotIn("start", [kind for _t, kind, _n in self.rig.events()])
        self.assertEqual(list(self.dirs.images.iterdir()), [])

    def test_a_waiting_job_is_removed_and_the_next_one_moves_up(self):
        _s, _h, _r, first = self.submit(sleep=0.2, steps=3)
        _s, _h, _r, second = self.submit(prompt="second", steps=1)
        _s, _h, _r, third = self.submit(prompt="third", steps=1)
        self.assertEqual(self.job(third["id"])["position"], 2)
        self.assertEqual(self.cancel(second["id"])["status"], "cancelled")
        self.assertEqual(self.job(second["id"])["status"], "cancelled")
        self.assertEqual(self.job(third["id"])["position"], 1)
        self.until(third["id"], "succeeded")
        names = sorted(p.name for p in self.dirs.images.iterdir())
        self.assertFalse(any("second" in name for name in names), names)

    def test_cancelling_a_finished_or_unknown_job_is_not_an_error(self):
        _s, _h, _r, job = self.submit(steps=1)
        self.until(job["id"], "succeeded")
        self.assertEqual(self.cancel(job["id"])["cancelled"], False)
        self.assertEqual(self.cancel(job["id"])["status"], "succeeded")
        self.assertEqual(self.cancel("job_nobody_knows"), {"id": "job_nobody_knows", "object": "job.cancel", "cancelled": False, "status": None})

    def test_cancel_needs_an_id(self):
        for raw in (b"{}", b"[]", b"not json", b'{"id": 5}'):
            status, _h, _r, body = self.rig.call("POST", "/v1/jobs/cancel", raw=raw, headers={**AUTH, "Content-Type": "application/json"})
            self.assertEqual((status, body["error"]["param"]), (400, "id"), raw)


class ListTest(JobRig):
    def test_the_list_is_newest_first_with_filters_and_pages(self):
        ids = []
        for n in range(3):
            _s, _h, _r, job = self.submit(prompt=f"p{n}", steps=1)
            ids.append(job["id"])
            self.until(job["id"], "succeeded")
            time.sleep(0.01)
        _s, _h, _r, body = self.rig.call("GET", "/v1/jobs", headers=AUTH)
        self.assertEqual([j["id"] for j in body["data"]], ids[::-1])
        self.assertEqual((body["object"], body["total"], body["has_more"]), ("list", 3, False))
        _s, _h, _r, page = self.rig.call("GET", "/v1/jobs?limit=1&offset=1&status=succeeded", headers=AUTH)
        self.assertEqual(([j["id"] for j in page["data"]], page["has_more"]), ([ids[1]], True))
        _s, _h, _r, none = self.rig.call("GET", "/v1/jobs?status=failed", headers=AUTH)
        self.assertEqual(none["data"], [])

    def test_bad_arguments_and_unknown_jobs(self):
        for query in ("status=done", "limit=0", "limit=201", "offset=-1", "limit=x"):
            status, _h, _r, body = self.rig.call("GET", "/v1/jobs?" + query, headers=AUTH)
            self.assertEqual(status, 400, query)
            self.assertIn(body["error"]["param"], ("status", "limit", "offset"))
        status, _h, _r, body = self.rig.call("GET", "/v1/jobs/job_nothing", headers=AUTH)
        self.assertEqual((status, body["error"]["code"]), (404, "job_not_found"))

    def test_every_job_route_needs_the_key(self):
        for method, path in (("GET", "/v1/jobs"), ("GET", "/v1/jobs/x"), ("POST", "/v1/jobs/cancel")):
            status, *_ = self.rig.call(method, path, {"id": "x"} if method == "POST" else None)
            self.assertEqual(status, 401, path)
        status, *_ = self.rig.call("POST", "/v1/images/generations", {"model": IMAGE, "prompt": "x"}, {"Prefer": "respond-async"})
        self.assertEqual(status, 401)
        self.assertEqual(self.rig.call("GET", "/v1/jobs", headers={**AUTH, "Sec-Fetch-Site": "cross-site", "Origin": "http://evil.example"})[0], 403)

    def test_health_says_jobs_and_library_are_here(self):
        _s, _h, _r, health = self.rig.call("GET", "/health", headers=AUTH)
        self.assertIs(health["features"]["jobs"], True)
        self.assertIs(health["features"]["library"], True)


class RealGatewayTest(unittest.TestCase):
    """follow_worker against the real gateway with a stub engine behind it: the real event names, the real way it ends an
    answer, and the real way it notices a client that went away."""

    class Progress:
        def __init__(self):
            self.seen = []

        def _set_progress(self, job_id, data):
            self.seen.append(data)

    def setUp(self):
        self.dirs = Folders().make(self)

    def work(self, kind, path, **fields):
        body = router.Body("application/json", json.dumps(fields).encode())
        return {"kind": kind, "path": path, "raw": router.streaming_body(path, body, body.raw), "content_type": "application/json",
                "summary": router.request_summary(body)}

    def context(self):
        self.progress = self.Progress()
        return jobs.Context(self.progress, "job_x")

    def test_a_picture_is_followed_to_the_file_the_gateway_kept(self):
        gateway = ImageHarness(api_key=None)
        self.addCleanup(gateway.close)
        gateway.server.save_images_to = str(self.dirs.images)
        worker = SimpleNamespace(port=gateway.port, entry={"id": IMAGE_MODEL_ID})
        work = self.work("image", "/v1/images/generations", model=IMAGE_MODEL_ID, prompt="a lighthouse", size="256x256", steps=3, seed=5,
                         response_format="b64_json", partial_images=2)
        result = router.follow_worker(worker, work, self.context())
        self.assertEqual((result["kind"], result["seed"], result["steps"], result["width"], result["height"]), ("images", 5, 3, 256, 256))
        self.assertTrue(result["file"].endswith("-a-lighthouse-s5.png"))
        self.assertEqual([p.name for p in self.dirs.images.iterdir()], [result["file"]])
        self.assertTrue(self.progress.seen)
        self.assertEqual(self.progress.seen[-1]["steps"], 3)

    def test_a_song_is_followed_to_the_file_the_gateway_kept(self):
        gateway = MusicHarness(api_key=None)
        self.addCleanup(gateway.close)
        gateway.server.save_songs_to = str(self.dirs.music)
        worker = SimpleNamespace(port=gateway.port, entry={"id": "ace-step-1.5-colibri"})
        work = self.work("music", "/v1/audio/music", model="ace-step-1.5-colibri", prompt="calm piano", duration=1, steps=1, seed=3,
                         response_format="wav")
        result = router.follow_worker(worker, work, self.context())
        self.assertEqual((result["kind"], result["seed"]), ("music", 3))
        self.assertTrue(result["file"].endswith("-calm-piano-s3.wav"))
        self.assertEqual([p.name for p in self.dirs.music.iterdir()], [result["file"]])

    def test_a_gateway_that_keeps_nothing_leaves_the_job_a_file_anyway(self):
        gateway = ImageHarness(api_key=None)
        self.addCleanup(gateway.close)
        gateway.server.save_images_to = None
        worker = SimpleNamespace(port=gateway.port, entry={"id": IMAGE_MODEL_ID})
        result = router.follow_worker(worker, self.work("image", "/v1/images/generations", model=IMAGE_MODEL_ID, prompt="a fox",
                                                        size="256x256", steps=2, seed=1), self.context())
        self.assertTrue((self.dirs.images / result["file"]).is_file())
        import media_meta
        self.assertEqual(media_meta.inspect_png(self.dirs.images / result["file"])["settings"]["prompt"], "a fox")

    def test_an_error_from_the_gateway_is_the_jobs_failure(self):
        gateway = ImageHarness(api_key=None)
        self.addCleanup(gateway.close)
        worker = SimpleNamespace(port=gateway.port, entry={"id": IMAGE_MODEL_ID})
        with self.assertRaises(jobs.JobError) as caught:
            router.follow_worker(worker, self.work("image", "/v1/images/generations", model=IMAGE_MODEL_ID, prompt="a fox", size="100x100"),
                                 self.context())
        self.assertEqual((caught.exception.status, caught.exception.code), (400, "invalid_value"))
        self.assertIn("multiple of", caught.exception.message)

    def test_cancelling_closes_the_connection_and_the_gateway_stops_computing(self):
        gateway = ImageHarness(api_key=None, delay=0.3)
        self.addCleanup(gateway.close)
        gateway.server.save_images_to = str(self.dirs.images)
        worker = SimpleNamespace(port=gateway.port, entry={"id": IMAGE_MODEL_ID})
        work = self.work("image", "/v1/images/generations", model=IMAGE_MODEL_ID, prompt="a fox", size="256x256", steps=40, seed=1)
        ctx, outcome = self.context(), {}

        def run():
            try:
                outcome["result"] = router.follow_worker(worker, work, ctx)
            except BaseException as error:           # noqa: BLE001
                outcome["error"] = error
        thread = threading.Thread(target=run)
        started = time.monotonic()
        thread.start()
        self.assertTrue(wait_for(lambda: self.progress.seen))
        ctx.cancel()
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome.get("error"), jobs.Cancelled)
        self.assertTrue(wait_for(lambda: gateway.server.scheduler.snapshot()["active"] == 0, 5))
        self.assertLess(time.monotonic() - started, 8)               # forty steps at 0.3 s would take twelve
        self.assertEqual(list(self.dirs.images.iterdir()), [])


class HelpersTest(unittest.TestCase):
    def body(self, **fields):
        return router.Body("application/json", json.dumps(fields).encode())

    def test_the_streaming_variant_asks_for_events_and_no_previews_and_keeps_the_rest(self):
        body = self.body(model="m", prompt="a", steps=4, response_format="b64_json", stream=False, partial_images=3)
        changed = json.loads(router.streaming_body("/v1/images/generations", body, body.raw))
        self.assertEqual(changed, {"model": "m", "prompt": "a", "steps": 4, "stream": True, "partial_images": 0})
        music = self.body(model="m", prompt="a", response_format="wav")
        self.assertEqual(json.loads(router.streaming_body("/v1/audio/music", music, music.raw)),
                         {"model": "m", "prompt": "a", "stream": True})

    def test_the_streaming_variant_of_a_form_replaces_those_fields_and_leaves_the_files(self):
        boundary = "XB"

        def part(name, value):
            return f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        raw = (part("model", "m") + part("prompt", "edit it") + part("stream", "false") + part("response_format", "b64_json")
               + f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="a.png"\r\n\r\nPNGDATA\r\n'.encode()
               + f"--{boundary}--\r\n".encode())
        body = router.Body(f"multipart/form-data; boundary={boundary}", raw)
        out = router.streaming_body("/v1/images/edits", body, raw)
        self.assertEqual(router._multipart_value(out, "stream"), "true")
        self.assertEqual(router._multipart_value(out, "partial_images"), "0")
        self.assertIsNone(router._multipart_value(out, "response_format"))
        self.assertEqual(router._multipart_value(out, "prompt"), "edit it")
        self.assertIn(b"PNGDATA", out)
        self.assertEqual(out.count(b'name="stream"'), 1)
        self.assertTrue(out.rstrip().endswith(f"--{boundary}--".encode()))

    def test_the_summary_is_the_prompt_and_the_numbers_and_nothing_else(self):
        body = self.body(model="m", prompt="x" * 400, steps=4, seed=9, size="512x512", images=["data:..."], keep_alive=5, user="me")
        found = router.request_summary(body)
        self.assertEqual(sorted(found), ["prompt", "seed", "size", "steps"])
        self.assertEqual(len(found["prompt"]), 303)
        self.assertEqual(router.request_summary(self.body(model="m", caption="calm piano", duration=30)), {"prompt": "calm piano", "duration": 30})

    def test_prefer_is_read_as_a_list_of_preferences(self):
        class Headers:
            def __init__(self, *values):
                self.values = values

            def get_all(self, name):
                return list(self.values) if name == "Prefer" else None
        self.assertTrue(router.wants_async(Headers("respond-async")))
        self.assertTrue(router.wants_async(Headers("wait=5, Respond-Async")))
        self.assertTrue(router.wants_async(Headers("wait=5", "respond-async")))
        self.assertFalse(router.wants_async(Headers("wait=5")))
        self.assertFalse(router.wants_async(Headers("respond-async-not")))
        self.assertFalse(router.wants_async(Headers()))

    def test_events_are_read_as_they_arrive_across_chunk_edges(self):
        class Stream:
            def __init__(self, pieces):
                self.pieces = list(pieces)

            def read1(self, _n):
                return self.pieces.pop(0) if self.pieces else b""
        pieces = [b"event: a.progress\ndata: {\"step\":1}\n\n: keep-alive\n\nevent: a.comp", b"leted\r\ndata: {\"x\":2}\r\n\r\ndata: [DONE]\n\n"]
        self.assertEqual(list(router.sse_events(Stream(pieces))),
                         [("a.progress", '{"step":1}'), ("a.completed", '{"x":2}'), (None, "[DONE]")])

    def test_the_end_of_a_job_is_a_line_in_the_request_log(self):
        class Log:
            def __init__(self):
                self.lines = []

            def write(self, record):
                self.lines.append(record)
        log = Log()
        finished = router.job_logger(log)
        finished({"id": "j1", "status": "succeeded", "model": "m", "endpoint": "/v1/images/generations", "created": 10, "started": 12, "finished": 20})
        finished({"id": "j2", "status": "failed", "model": "m", "endpoint": "/v1/audio/music", "created": 10, "started": 12, "finished": 13,
                  "error": {"code": "engine_error", "message": "boom"}})
        finished({"id": "j3", "status": "cancelled", "model": "m", "endpoint": "/v1/audio/music", "created": 10, "started": None, "finished": 11})
        self.assertEqual([(r["job"], r["status"], r["seconds"], r.get("error")) for r in log.lines],
                         [("j1", 200, 8, None), ("j2", 500, 1, "engine_error"), ("j3", 499, 1, "cancelled")])
        router.job_logger(None)({"id": "x", "status": "succeeded", "created": 1})        # no log: nothing happens


if __name__ == "__main__":
    unittest.main()
