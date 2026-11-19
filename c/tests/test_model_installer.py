"""Installing and uninstalling models while colibri runs (model_installer.py).

A fake Hugging Face hub serves a tiny repository; the installer fetches it as a job, and the tests hold it to what the
router's endpoints and `coli model` rely on: the confirmation, one install at a time, progress, cancel, a failure that
leaves the partial files, a lock between processes, and an uninstall that never leaves a half-deleted model behind or
touches a folder outside the models folders."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
import model_installer  # noqa: E402
import model_switch  # noqa: E402
import setup_catalog  # noqa: E402
import setup_download  # noqa: E402
from fake_hub import FakeHub  # noqa: E402

REPO = "someone/tiny-image"


def entry(**overrides):
    values = dict(id="tiny-image", family="zimage", name="Tiny image", repo=REPO, disk_gb=0.001, ram_min_gb=0.1,
                  ram_good_gb=0.2, dense_gb=0.01, rank=1, size_class="small", summary="fixture", modality="image",
                  license_note="Apache-2.0")
    values.update(overrides)
    return setup_catalog.CatalogModel(**values)


class InstallerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = self.tmp.name
        self.src = os.path.join(base, "repo")
        os.makedirs(self.src)
        Path(self.src, "model_index.json").write_text(json.dumps({"_class_name": "ZImagePipeline"}))
        Path(self.src, "tokenizer.json").write_text("{}")
        self.shard = os.urandom(300_000)
        Path(self.src, "weights.safetensors").write_bytes(self.shard)
        self.hub = FakeHub({REPO: self.src})
        self.hub.__enter__()
        self.addCleanup(self.hub.__exit__)
        self.root = os.path.join(base, "models")
        os.makedirs(self.root)
        self.refreshed = 0
        self.entries = [entry()]
        patcher = mock.patch.dict(os.environ, {"HF_ENDPOINT": self.hub.base})
        patcher.start()
        self.addCleanup(patcher.stop)
        sleeper = mock.patch.object(setup_download.time, "sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)
        self.installer = model_installer.Installer(
            roots=lambda: [self.root], installed=self.installed, refresh=self.refresh,
            catalog=lambda: tuple(self.entries), memory=lambda: 64 * 10 ** 9, launcher_dir=str(HERE))

    def refresh(self):
        self.refreshed += 1

    def installed(self):
        out = []
        for name in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, name)
            if os.path.isdir(path) and not name.endswith(model_installer.DELETING) \
                    and os.path.isfile(os.path.join(path, "model_index.json")):
                manifest = setup_download.read_manifest(path)
                out.append({"key": model_switch.model_key(path), "path": os.path.realpath(path), "id": name, "name": name,
                            "unfinished": bool(manifest and not manifest.get("complete")), "state": "idle"})
        return out

    def wait(self, model_id, states=("done", "failed", "cancelled"), timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            job = self.installer.job(model_id)
            if job and job["state"] in states:
                return job
            time.sleep(0.02)
        self.fail(f"the job did not finish: {self.installer.job(model_id)}")


class Install(InstallerCase):
    def test_nothing_is_fetched_without_confirm(self):
        for confirm in (False, None, "yes", 1):
            with self.assertRaises(model_installer.InstallError) as caught:
                self.installer.start("tiny-image", confirm=confirm)
            self.assertEqual((caught.exception.status, caught.exception.code), (400, "confirmation_required"))
        self.assertEqual(os.listdir(self.root), [])
        self.assertEqual(self.hub.requests, [])

    def test_unknown_model(self):
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.start("nope", confirm=True)
        self.assertEqual((caught.exception.status, caught.exception.code), (404, "model_not_found"))

    def test_an_install_runs_to_the_end(self):
        job = self.installer.start("tiny-image", confirm=True, background=False)
        self.assertEqual(job["state"], "done", job)
        folder = os.path.join(self.root, "tiny-image")
        self.assertEqual(Path(folder, "weights.safetensors").read_bytes(), self.shard)
        self.assertTrue(setup_download.is_complete(folder))
        self.assertFalse(os.path.exists(os.path.join(folder, model_installer.LOCK_NAME)), "the lock was left behind")
        self.assertGreaterEqual(self.refreshed, 1)
        self.assertEqual(job["done"], job["total"])

    def test_the_catalog_says_what_is_here(self):
        before = self.installer.catalog_rows()[0]
        self.assertEqual((before["id"], before["installed"], before["license_note"], before["kind"]),
                         ("tiny-image", None, "Apache-2.0", "image"))
        self.assertTrue(before["fits"])
        self.installer.start("tiny-image", confirm=True, background=False)
        after = self.installer.catalog_rows()[0]
        self.assertEqual(after["installed"]["id"], "tiny-image")
        self.assertEqual(after["job"]["state"], "done")

    def test_a_second_install_of_the_same_model_is_refused(self):
        self.installer.start("tiny-image", confirm=True, background=False)
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.start("tiny-image", confirm=True)
        self.assertEqual(caught.exception.code, "already_installed")

    def test_one_install_at_a_time(self):
        self.entries.append(entry(id="other", name="Other"))
        self.installer.jobs["other"] = {"id": "other", "name": "Other", "state": "downloading"}
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.start("tiny-image", confirm=True)
        self.assertEqual((caught.exception.status, caught.exception.code), (409, "install_busy"))

    def test_two_requests_at_once_start_only_one(self):
        # two different models asked for at the same moment: the router's handler threads race, and only one may win
        self.entries.append(entry(id="other", name="Other"))
        gate = threading.Event()
        real = setup_download.download_repo

        def held(repo, revision, model_dir, files, **kw):
            gate.wait(10)                      # the winner stays "downloading" until the test lets it go
            return real(repo, revision, model_dir, files, **kw)

        outcomes = []

        def ask(model_id):
            try:
                outcomes.append(self.installer.start(model_id, confirm=True)["state"])
            except model_installer.InstallError as error:
                outcomes.append(error.code)

        with mock.patch.object(setup_download, "download_repo", held):
            threads = [threading.Thread(target=ask, args=(m,)) for m in ("tiny-image", "other") * 3]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            gate.set()
            for model_id in ("tiny-image", "other"):
                job = self.installer.job(model_id)
                if job:
                    self.wait(model_id)
        started = [o for o in outcomes if o in ("starting", "downloading", "finishing", "done")]
        self.assertEqual(len(started), 1, outcomes)
        self.assertEqual(sorted(set(o for o in outcomes if o not in started)), ["install_busy"])

    def test_progress_is_reported_while_it_downloads(self):
        seen = []
        real = setup_download.download_repo

        def slow(repo, revision, model_dir, files, **kw):
            def watch(event):
                seen.append(self.installer.job("tiny-image")["state"])
                kw["progress"](event)
            return real(repo, revision, model_dir, files, **{**kw, "progress": watch})

        with mock.patch.object(setup_download, "download_repo", slow):
            job = self.installer.start("tiny-image", confirm=True, background=False)
        self.assertEqual(job["state"], "done")
        self.assertIn("downloading", seen)

    def test_cancel_stops_it_and_keeps_the_part_that_came(self):
        real = setup_download.download_repo

        def cancelling(repo, revision, model_dir, files, **kw):
            def watch(event):
                self.installer.cancel_job("tiny-image")        # what POST /v1/models/install/cancel does
                kw["progress"](event)
            return real(repo, revision, model_dir, files, **{**kw, "progress": watch})

        with mock.patch.object(setup_download, "download_repo", cancelling):
            job = self.installer.start("tiny-image", confirm=True, background=False)
        self.assertEqual(job["state"], "cancelled", job)
        folder = os.path.join(self.root, "tiny-image")
        self.assertFalse(os.path.exists(os.path.join(folder, model_installer.LOCK_NAME)))
        self.assertFalse(setup_download.is_complete(folder))
        # asking again resumes and finishes
        again = self.installer.start("tiny-image", confirm=True, background=False)
        self.assertEqual(again["state"], "done", again)
        self.assertTrue(setup_download.is_complete(folder))

    def test_cancel_of_nothing_says_so(self):
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.cancel_job("tiny-image")
        self.assertEqual(caught.exception.status, 404)

    def test_a_failure_is_the_jobs_state_not_an_exception(self):
        self.entries[0] = entry(repo="someone/not-there")
        job = self.installer.start("tiny-image", confirm=True, background=False)
        self.assertEqual(job["state"], "failed")
        self.assertIn("not found", job["error"])
        self.assertFalse(os.path.exists(os.path.join(self.root, "tiny-image", model_installer.LOCK_NAME)))
        self.assertIsNone(self.installer.active())

    def test_a_download_in_the_background_can_be_watched(self):
        job = self.installer.start("tiny-image", confirm=True)
        self.assertIn(job["state"], ("starting", "downloading", "finishing", "done"))
        self.assertEqual(self.wait("tiny-image")["state"], "done")
        self.assertEqual([j["id"] for j in self.installer.job_list()], ["tiny-image"])

    def test_not_enough_disk_is_refused_before_anything_is_written(self):
        self.entries[0] = entry(disk_gb=10 ** 9)
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.start("tiny-image", confirm=True)
        self.assertEqual(caught.exception.code, "insufficient_disk")
        self.assertFalse(os.path.exists(os.path.join(self.root, "tiny-image")))

    def test_a_folder_that_is_not_ours_is_not_written_into(self):
        os.makedirs(os.path.join(self.root, "tiny-image"))
        Path(self.root, "tiny-image", "notes.txt").write_text("mine")
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.start("tiny-image", confirm=True)
        self.assertEqual(caught.exception.code, "folder_exists")

    def test_another_live_process_holding_the_folder_is_respected(self):
        folder = os.path.join(self.root, "tiny-image")
        os.makedirs(folder)
        Path(folder, model_installer.LOCK_NAME).write_text(json.dumps({"pid": os.getppid()}))
        with mock.patch.object(model_installer, "_pid_alive", return_value=True):
            with self.assertRaises(model_installer.InstallError) as caught:
                model_installer.take_lock(folder)
        self.assertEqual(caught.exception.code, "model_busy")

    def test_a_lock_left_by_a_dead_process_is_taken_over(self):
        folder = os.path.join(self.root, "tiny-image")
        os.makedirs(folder)
        Path(folder, model_installer.LOCK_NAME).write_text(json.dumps({"pid": 2 ** 22 + 12345}))
        with mock.patch.object(model_installer, "_pid_alive", return_value=False):
            model_installer.take_lock(folder)
        self.assertEqual(model_installer.lock_holder(folder), os.getpid())
        model_installer.release_lock(folder)
        self.assertIsNone(model_installer.lock_holder(folder))


class Uninstall(InstallerCase):
    def setUp(self):
        super().setUp()
        self.installer.start("tiny-image", confirm=True, background=False)
        self.item = self.installed()[0]
        self.folder = self.item["path"]

    def test_confirm_is_required(self):
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.uninstall(self.item)
        self.assertEqual(caught.exception.code, "confirmation_required")
        self.assertTrue(os.path.isdir(self.folder))

    def test_the_files_go_and_the_size_is_reported(self):
        before = model_installer.folder_bytes(self.folder)
        result = self.installer.uninstall(self.item, confirm=True)
        self.assertEqual(result["status"], "removed")
        self.assertEqual(result["freed_bytes"], before)
        self.assertGreater(before, 300_000)
        self.assertEqual(os.listdir(self.root), [])
        self.assertGreaterEqual(self.refreshed, 2)

    def test_it_can_be_installed_again_afterwards(self):
        self.installer.uninstall(self.item, confirm=True)
        self.assertEqual(self.installer.start("tiny-image", confirm=True, background=False)["state"], "done")

    def test_a_read_only_file_does_not_stop_it(self):
        os.chmod(os.path.join(self.folder, "tokenizer.json"), 0o444)
        self.installer.uninstall(self.item, confirm=True)
        self.assertEqual(os.listdir(self.root), [])

    def test_a_folder_outside_the_models_folders_is_refused(self):
        outside = os.path.join(self.tmp.name, "elsewhere", "precious")
        os.makedirs(outside)
        Path(outside, "model_index.json").write_text("{}")
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.uninstall({"path": outside, "key": "x", "name": "precious"}, confirm=True)
        self.assertEqual(caught.exception.code, "outside_models_folder")
        self.assertTrue(os.path.isfile(os.path.join(outside, "model_index.json")))

    def test_a_models_folder_itself_is_never_removed(self):
        with self.assertRaises(model_installer.InstallError):
            self.installer.uninstall({"path": self.root, "key": "x", "name": "root"}, confirm=True)
        self.assertTrue(os.path.isdir(self.root))

    def test_a_link_is_refused_and_its_target_is_left(self):
        target = os.path.join(self.tmp.name, "target")
        os.makedirs(target)
        Path(target, "keep.txt").write_text("keep")
        link = os.path.join(self.root, "linked")
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("this account cannot make symbolic links")
        with self.assertRaises(model_installer.InstallError):
            self.installer.uninstall({"path": link, "key": "x", "name": "linked"}, confirm=True)
        self.assertTrue(os.path.isfile(os.path.join(target, "keep.txt")))

    def test_a_download_in_progress_is_not_removed_under_it(self):
        self.installer.jobs["tiny-image"] = {"id": "tiny-image", "name": "Tiny", "state": "downloading"}
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.uninstall(self.item, confirm=True)
        self.assertEqual(caught.exception.code, "model_busy")
        self.assertTrue(os.path.isdir(self.folder))

    def test_files_another_process_holds_leave_the_model_where_it_was(self):
        with mock.patch.object(model_installer.os, "rename", side_effect=PermissionError(13, "in use")):
            with self.assertRaises(model_installer.InstallError) as caught:
                self.installer.uninstall(self.item, confirm=True)
        self.assertEqual((caught.exception.status, caught.exception.code), (409, "model_in_use"))
        self.assertTrue(os.path.isfile(os.path.join(self.folder, "weights.safetensors")))

    def test_a_deletion_that_was_cut_short_is_finished_and_never_listed(self):
        left = os.path.join(self.root, f".tiny-image.1{model_installer.DELETING}")
        os.makedirs(left)
        Path(left, "model_index.json").write_text("{}")
        names = [os.path.basename(p) for p in model_switch._children(self.root)]
        self.assertNotIn(os.path.basename(left), names)
        self.assertEqual(self.installer.sweep(), 1)
        self.assertFalse(os.path.exists(left))


if __name__ == "__main__":
    unittest.main()
