"""The router over real `coli serve` workers, with the stub engines (an image model and a music model).

test_router.py checks the router against fake workers; this one checks the whole path a user takes: the catalog finds the
checkpoints, each worker is a real gateway on its own port, and one port answers a picture and a song."""
import base64
import http.client
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import router  # noqa: E402
from acestep_fixture import STUB as MUSIC_STUB, make_fake_checkpoint as make_music, stub_env as music_env  # noqa: E402
from qwen_image_fixture import STUB as IMAGE_STUB, make_fake_pipeline as make_image, stub_env as image_env  # noqa: E402

IMAGE_ID, MUSIC_ID = "qwen-image-2.1-colibri", "ace-step-1.5-colibri"


class RealGatewaysTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.models = root / "models"
        make_image(self.models / "qwen-image")
        make_music(self.models / "ace-step")
        self.saved = {"images": str(root / "images"), "music": str(root / "music")}

    def rig(self, **options):
        catalog = router.Catalog(roots=[str(self.models)], engine_ready=lambda family: True)

        def worker_env(entry):
            if entry["modality"] == "image":
                env = image_env(COLI_ENGINE=str(IMAGE_STUB), COLI_IMAGE_DIR=self.saved["images"])
            else:
                env = music_env(COLI_ENGINE=str(MUSIC_STUB), COLI_MUSIC_DIR=self.saved["music"])
            return {k: v for k, v in env.items() if v != os.environ.get(k)} | {"COLI_ENGINE": env["COLI_ENGINE"]}

        pool = router.Pool(catalog, worker_env=worker_env, grace=8.0, state=router.State(), **options)
        gate = router.Gate()
        server = router.RouterServer(("127.0.0.1", 0), pool, catalog, gate, None, ("http://127.0.0.1:1",))
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def stop():
            server.shutdown()
            server.server_close()
            pool.close()

        self.addCleanup(stop)
        self.pool, self.port = pool, server.server_address[1]

    def call(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        try:
            data = json.dumps(body).encode() if body is not None else None
            connection.request(method, path, body=data, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = response.read()
            return response.status, json.loads(payload) if payload else None
        finally:
            connection.close()

    def states(self):
        return {m["id"]: m["state"] for m in self.call("GET", "/v1/models/installed")[1]["data"]}

    def test_one_port_answers_a_picture_and_a_song(self):
        self.rig()
        installed = self.call("GET", "/v1/models/installed")[1]
        self.assertEqual({m["id"] for m in installed["data"]}, {IMAGE_ID, MUSIC_ID})
        self.assertTrue(all(m["switchable"] for m in installed["data"]), installed)
        self.assertEqual(self.states(), {IMAGE_ID: "idle", MUSIC_ID: "idle"})

        status, body = self.call("POST", "/v1/audio/music", {"model": MUSIC_ID, "prompt": "calm piano", "duration": 1,
                                                             "steps": 2, "seed": 1})
        self.assertEqual(status, 200, body)
        self.assertTrue(base64.b64decode(body["data"][0]["b64_json"]).startswith(b"RIFF"))
        status, body = self.call("POST", "/v1/images/generations", {"model": IMAGE_ID, "prompt": "a hummingbird"})
        self.assertEqual(status, 200, body)
        self.assertTrue(base64.b64decode(body["data"][0]["b64_json"]).startswith(b"\x89PNG"))

        self.assertEqual(self.states(), {IMAGE_ID: "ready", MUSIC_ID: "ready"})
        listing = {m["id"]: m for m in self.call("GET", "/v1/models")[1]["data"]}
        self.assertEqual(listing[MUSIC_ID]["state"], "loaded")
        self.assertIn("music_generation", listing[MUSIC_ID]["capabilities"])
        self.assertIn("tasks", listing[MUSIC_ID]["audio"])                    # the engine's own limits, not a stub
        self.assertIn("default_width", listing[IMAGE_ID]["image"])
        health = self.call("GET", "/health")[1]
        self.assertEqual({w["id"] for w in health["router"]["loaded"]}, {IMAGE_ID, MUSIC_ID})

    def test_the_pictures_it_makes_are_kept_by_the_gateway_that_made_them(self):
        self.rig()
        self.call("POST", "/v1/images/generations", {"model": IMAGE_ID, "prompt": "x"})
        self.assertTrue(any(Path(self.saved["images"]).glob("*.png")))

    def test_a_limit_of_one_model_swaps_them(self):
        self.rig(max_loaded=1)
        self.assertEqual(self.call("POST", "/v1/audio/music", {"model": MUSIC_ID, "prompt": "x", "duration": 1,
                                                               "steps": 2})[0], 200)
        self.assertEqual(self.call("POST", "/v1/images/generations", {"model": IMAGE_ID, "prompt": "x"})[0], 200)
        self.assertEqual(self.states(), {IMAGE_ID: "ready", MUSIC_ID: "idle"})
        listing = {m["id"]: m for m in self.call("GET", "/v1/models")[1]["data"]}
        self.assertEqual(listing[MUSIC_ID]["state"], "idle")
        self.assertIn("tasks", listing[MUSIC_ID]["audio"], "an unloaded model is listed from what it said last time")

    def test_the_dashboards_switch_loads_a_model(self):
        self.rig()
        key = next(m["key"] for m in self.call("GET", "/v1/models/installed")[1]["data"] if m["id"] == MUSIC_ID)
        status, body = self.call("POST", "/v1/models/switch", {"key": key})
        self.assertEqual((status, body["id"]), (200, MUSIC_ID), body)
        self.assertEqual(self.states(), {IMAGE_ID: "idle", MUSIC_ID: "ready"})

    def test_a_model_the_gateway_refuses_is_reported_by_it(self):
        self.rig()
        status, body = self.call("POST", "/v1/images/generations", {"model": IMAGE_ID, "prompt": "x", "size": "7x7"})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["param"], "size")


if __name__ == "__main__":
    unittest.main()
