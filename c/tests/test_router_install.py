"""The router's model install and uninstall endpoints (GET /v1/models/catalog, POST /v1/models/install, /install/cancel,
/uninstall, GET /v1/models/installs): their shapes, their refusals, and that a loaded model is never removed.

The installing itself is tested in test_model_installer.py; here a real Installer (over a fake hub) sits behind the
router's handlers, with fake workers for the "loaded" cases."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import model_installer  # noqa: E402
import setup_catalog  # noqa: E402
import setup_download  # noqa: E402
import test_router  # noqa: E402
from fake_hub import FakeHub  # noqa: E402

REPO = "someone/tiny-image"
AUTH = {"Authorization": "Bearer sekrit"}


class InstallRig(test_router.Rig):
    """A router whose models are real folders under one models root, with an installer over a fake hub."""

    def __init__(self, test, names=("text-a",), with_installer=True, **kw):
        self.root = tempfile.mkdtemp()
        test.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))
        entries = []
        for name in names:
            folder = os.path.join(self.root, name)
            os.makedirs(folder)
            Path(folder, "config.json").write_text("{}")
            Path(folder, "weights.bin").write_bytes(b"x" * 5000)
            item = test_router.entry(name)
            item.update(path=os.path.realpath(folder))
            entries.append(item)
        super().__init__(test, entries, **kw)
        self.src = os.path.join(self.root, "_repo")
        os.makedirs(self.src)
        Path(self.src, "model_index.json").write_text(json.dumps({"_class_name": "ZImagePipeline"}))
        Path(self.src, "weights.safetensors").write_bytes(os.urandom(200_000))
        self.hub = FakeHub({REPO: self.src})
        self.hub.__enter__()
        test.addCleanup(self.hub.__exit__)
        patch = mock.patch.dict(os.environ, {"HF_ENDPOINT": self.hub.base})
        patch.start()
        test.addCleanup(patch.stop)
        self.fetchable = setup_catalog.CatalogModel(
            id="tiny-image", family="zimage", name="Tiny image", repo=REPO, disk_gb=0.001, ram_min_gb=0.1,
            ram_good_gb=0.2, dense_gb=0.01, rank=1, size_class="small", summary="fixture", modality="image",
            license_note="Apache-2.0")
        self.refreshed = 0
        if with_installer:
            self.server.installer = model_installer.Installer(
                roots=lambda: [self.root], installed=self.installed_now, refresh=self.refresh,
                catalog=lambda: (self.fetchable,), memory=lambda: 64 * 10 ** 9)

    def refresh(self):
        self.refreshed += 1

    def installed_now(self):
        out = []
        for found in self.catalog.entries():
            worker = self.pool.get(found["key"])
            found["state"] = worker.state if worker is not None else "idle"
            out.append(found)
        return out

    def wait_job(self, model_id, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            status, _, _, body = self.call("GET", f"/v1/models/installs/{model_id}", headers=AUTH)
            if status == 200 and body["state"] in ("done", "failed", "cancelled"):
                return body
            time.sleep(0.05)
        self.test.fail("the install did not finish")


class CatalogAndFeatures(unittest.TestCase):
    def test_features_say_whether_the_router_can_install(self):
        rig = InstallRig(self, api_key="sekrit")
        self.assertTrue(rig.call("GET", "/health", headers=AUTH)[3]["features"]["model_install"])
        bare = InstallRig(self, api_key="sekrit", with_installer=False)
        self.assertFalse(bare.call("GET", "/health", headers=AUTH)[3]["features"]["model_install"])
        self.assertEqual(bare.call("GET", "/v1/models/catalog", headers=AUTH)[0], 404)

    def test_the_catalog_lists_what_can_be_fetched(self):
        rig = InstallRig(self, api_key="sekrit")
        status, _, _, body = rig.call("GET", "/v1/models/catalog", headers=AUTH)
        self.assertEqual(status, 200)
        row = body["data"][0]
        self.assertEqual((row["id"], row["kind"], row["license_note"], row["installed"], row["fits"]),
                         ("tiny-image", "image", "Apache-2.0", None, True))
        self.assertEqual(body["roots"], [rig.root])
        self.assertIsNotNone(body["free_gb"])

    def test_every_endpoint_wants_the_key(self):
        rig = InstallRig(self, api_key="sekrit")
        for method, path in (("GET", "/v1/models/catalog"), ("GET", "/v1/models/installs"),
                             ("POST", "/v1/models/install"), ("POST", "/v1/models/install/cancel"),
                             ("POST", "/v1/models/uninstall")):
            self.assertEqual(rig.call(method, path, {} if method == "POST" else None)[0], 401, path)


class Install(unittest.TestCase):
    def test_install_needs_confirm_and_a_known_id(self):
        rig = InstallRig(self, api_key="sekrit")
        status, _, _, body = rig.call("POST", "/v1/models/install", {"id": "tiny-image"}, headers=AUTH)
        self.assertEqual((status, body["error"]["code"]), (400, "confirmation_required"))
        status, _, _, body = rig.call("POST", "/v1/models/install", {"id": "nope", "confirm": True}, headers=AUTH)
        self.assertEqual((status, body["error"]["code"]), (404, "model_not_found"))
        self.assertEqual(rig.call("POST", "/v1/models/install", {"confirm": True}, headers=AUTH)[0], 400)
        self.assertEqual(rig.hub.requests, [], "something was fetched without a confirmation")

    def test_it_is_a_management_request_like_a_switch(self):
        rig = InstallRig(self, api_key="sekrit")
        self.assertEqual(rig.call("POST", "/v1/models/install", raw=b"{}",
                                  headers=dict(AUTH, **{"Content-Type": "text/plain"}))[0], 415)
        status, _, _, body = rig.call("POST", "/v1/models/install", {"id": "tiny-image", "confirm": True},
                                      headers=dict(AUTH, Origin="http://evil.example"))
        self.assertEqual((status, body["error"]["code"]), (403, "forbidden"))
        self.assertEqual(rig.hub.requests, [])

    def test_a_job_that_can_be_followed(self):
        rig = InstallRig(self, api_key="sekrit")
        status, _, _, job = rig.call("POST", "/v1/models/install", {"id": "tiny-image", "confirm": True}, headers=AUTH)
        self.assertEqual(status, 202)
        self.assertEqual((job["id"], job["name"], job["license_note"]), ("tiny-image", "Tiny image", "Apache-2.0"))
        done = rig.wait_job("tiny-image")
        self.assertEqual(done["state"], "done", done)
        self.assertTrue(setup_download.is_complete(os.path.join(rig.root, "tiny-image")))
        listing = rig.call("GET", "/v1/models/installs", headers=AUTH)[3]
        self.assertEqual([j["id"] for j in listing["data"]], ["tiny-image"])
        self.assertGreaterEqual(rig.refreshed, 1)

    def test_a_job_that_does_not_exist(self):
        rig = InstallRig(self, api_key="sekrit")
        self.assertEqual(rig.call("GET", "/v1/models/installs/nope", headers=AUTH)[0], 404)
        status, _, _, body = rig.call("POST", "/v1/models/install/cancel", {"id": "tiny-image"}, headers=AUTH)
        self.assertEqual((status, body["error"]["code"]), (404, "not_installing"))

    def test_a_failed_install_is_a_job_that_says_why(self):
        rig = InstallRig(self, api_key="sekrit")
        rig.fetchable = setup_catalog.CatalogModel(**{**rig.fetchable.__dict__, "repo": "someone/not-there"})
        with mock.patch.object(setup_download.time, "sleep"):
            rig.call("POST", "/v1/models/install", {"id": "tiny-image", "confirm": True}, headers=AUTH)
            job = rig.wait_job("tiny-image")
        self.assertEqual(job["state"], "failed")
        self.assertIn("not found", job["error"])


class Uninstall(unittest.TestCase):
    def test_it_needs_a_known_model_and_a_confirmation(self):
        rig = InstallRig(self, api_key="sekrit")
        self.assertEqual(rig.call("POST", "/v1/models/uninstall", {"model": "nope", "confirm": True}, headers=AUTH)[0], 404)
        status, _, _, body = rig.call("POST", "/v1/models/uninstall", {"model": "text-a"}, headers=AUTH)
        self.assertEqual((status, body["error"]["code"]), (400, "confirmation_required"))
        self.assertTrue(os.path.isdir(os.path.join(rig.root, "text-a")))

    def test_an_idle_model_is_removed(self):
        rig = InstallRig(self, names=("text-a", "text-b"), api_key="sekrit")
        status, _, _, body = rig.call("POST", "/v1/models/uninstall", {"model": "text-a", "confirm": True}, headers=AUTH)
        self.assertEqual(status, 200, body)
        self.assertEqual((body["status"], body["id"], body["presets"]), ("removed", "text-a-colibri", []))
        self.assertGreaterEqual(body["freed_bytes"], 5000)
        self.assertFalse(os.path.exists(os.path.join(rig.root, "text-a")))
        self.assertTrue(os.path.isdir(os.path.join(rig.root, "text-b")), "the other model was touched")

    def test_a_loaded_model_is_refused_and_kept(self):
        rig = InstallRig(self, api_key="sekrit")
        self.assertEqual(rig.call("POST", "/v1/chat/completions", {"model": "text-a-colibri", "messages": []},
                                  headers=AUTH)[0], 200)
        status, _, _, body = rig.call("POST", "/v1/models/uninstall", {"model": "text-a", "confirm": True}, headers=AUTH)
        self.assertEqual((status, body["error"]["code"]), (409, "model_loaded"))
        self.assertTrue(os.path.isdir(os.path.join(rig.root, "text-a")))
        # unloaded, it can go
        rig.call("POST", "/v1/models/unload", {"model": "text-a"}, headers=AUTH)
        self.assertEqual(rig.call("POST", "/v1/models/uninstall", {"model": "text-a", "confirm": True}, headers=AUTH)[0], 200)

    def test_a_model_outside_the_models_folder_is_refused(self):
        rig = InstallRig(self, api_key="sekrit")
        other = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(other, ignore_errors=True))
        Path(other, "config.json").write_text("{}")
        rig.entries[0]["path"] = os.path.realpath(other)
        rig.catalog = router_catalog = type(rig.catalog)(entries=rig.entries)
        rig.server.catalog = router_catalog
        status, _, _, body = rig.call("POST", "/v1/models/uninstall", {"model": "text-a", "confirm": True}, headers=AUTH)
        self.assertEqual((status, body["error"]["code"]), (409, "outside_models_folder"))
        self.assertTrue(os.path.isfile(os.path.join(other, "config.json")))


if __name__ == "__main__":
    unittest.main()
