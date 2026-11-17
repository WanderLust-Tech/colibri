"""The router's catalogue keeps what it has seen: a drive that is unplugged or a download that is half done stays visible, with why."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import router  # noqa: E402
import setup_download  # noqa: E402
from qwen_image_fixture import make_fake_pipeline  # noqa: E402


class Known(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name, "models")
        self.root.mkdir()
        self.known = os.path.join(self.tmp.name, "state", "known-models.json")
        self.folder = Path(make_fake_pipeline(self.root / "qwen-image"))

    def catalog(self, known=True):
        return router.Catalog(roots=[str(self.root)], engine_ready=lambda family: True, known_path=self.known if known else None, ttl=0)

    def by_folder(self, entries):
        return {os.path.basename(e["path"]): e for e in entries}


class RememberingTest(Known):
    def test_a_model_found_is_written_down(self):
        entries = self.catalog().entries()
        self.assertEqual(len(entries), 1)
        saved = json.loads(Path(self.known).read_text(encoding="utf-8"))["models"]
        self.assertEqual(list(saved), [entries[0]["key"]])
        self.assertEqual(saved[entries[0]["key"]]["id"], entries[0]["id"])
        self.assertNotIn("reason", saved[entries[0]["key"]])

    def test_without_a_file_to_keep_it_in_nothing_changes(self):
        catalog = self.catalog(known=False)
        catalog.entries()
        shutil.rmtree(self.folder)
        self.assertEqual(catalog.entries(), [])
        self.assertFalse(os.path.exists(self.known))

    def test_a_model_on_a_drive_that_is_not_there_stays_listed_switched_off_with_the_reason(self):
        before = self.catalog().entries()[0]
        shutil.rmtree(self.folder)
        with mock.patch.object(router, "why_missing", lambda path: ("the drive E: is not connected", False)):
            entries = self.catalog().entries()
        self.assertEqual(len(entries), 1)
        gone = entries[0]
        self.assertEqual((gone["key"], gone["id"], gone["name"], gone["modality"]), (before["key"], before["id"], before["name"], before["modality"]))
        self.assertEqual((gone["reason"], gone["missing"], gone["switchable"], gone["current"], gone["verified"]),
                         ("the drive E: is not connected", True, False, False, False))
        self.assertEqual(gone["path"], before["path"])

    def test_it_comes_back_as_itself_when_the_drive_does(self):
        self.catalog().entries()
        moved = self.root.parent / "elsewhere"
        shutil.move(str(self.folder), str(moved))
        with mock.patch.object(router, "why_missing", lambda path: ("the drive E: is not connected", False)):
            self.assertTrue(self.catalog().entries()[0]["missing"])
        shutil.move(str(moved), str(self.folder))
        entries = self.catalog().entries()
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["reason"])
        self.assertNotIn("missing", entries[0])

    def test_a_folder_that_was_deleted_from_a_place_that_is_there_is_forgotten(self):
        self.catalog().entries()
        shutil.rmtree(self.folder)
        self.assertEqual(self.catalog().entries(), [])                       # the parent is there and the folder is not: deleted on purpose
        self.assertEqual(json.loads(Path(self.known).read_text(encoding="utf-8"))["models"], {})

    def test_a_folder_that_is_there_but_is_no_longer_a_model_says_so(self):
        self.catalog().entries()
        (self.folder / "model_index.json").unlink()
        entries = self.catalog().entries()
        self.assertEqual(len(entries), 1)
        self.assertIn("no longer recognised as a model", entries[0]["reason"])
        self.assertIn("coli model verify", entries[0]["reason"])

    def test_a_missing_model_cannot_be_asked_for_and_says_why(self):
        self.catalog().entries()
        shutil.rmtree(self.folder)
        with mock.patch.object(router, "why_missing", lambda path: ("the drive E: is not connected", False)):
            catalog = self.catalog()
            found = catalog.find("qwen-image-2.1-colibri")
        self.assertIsNotNone(found)
        self.assertEqual(found["reason"], "the drive E: is not connected")

    def test_a_damaged_known_file_is_ignored_and_rewritten(self):
        os.makedirs(os.path.dirname(self.known))
        Path(self.known).write_text("{not json", encoding="utf-8")
        self.assertEqual(len(self.catalog().entries()), 1)
        self.assertEqual(len(json.loads(Path(self.known).read_text(encoding="utf-8"))["models"]), 1)


class WhyMissingTest(unittest.TestCase):
    def have(self, *paths):
        return lambda p: p.replace("\\", "/") in set(paths)

    def test_a_windows_drive_that_is_not_there(self):
        self.assertEqual(router.why_missing("E:\\models\\gemma", self.have("C:/")), ("the drive E: is not connected", False))

    def test_a_windows_drive_that_is_there_means_the_folder_was_deleted(self):
        self.assertEqual(router.why_missing("E:\\models\\gemma", self.have("E:/", "E:/models")), (None, True))
        self.assertEqual(router.why_missing("E:\\models\\gemma", self.have("E:/")), (None, True))

    def test_an_empty_mount_point_is_a_drive_that_is_not_plugged_in(self):
        reason, forget = router.why_missing("/mnt/usb/models/gemma", self.have("/", "/mnt", "/mnt/usb"), empty=lambda p: True)
        self.assertEqual((reason, forget), ("the drive or share at /mnt/usb is not mounted", False))

    def test_an_ordinary_folder_is_not_taken_for_a_mount_point(self):
        self.assertEqual(router.why_missing("/home/me/models/gemma", self.have("/", "/home", "/home/me"), empty=lambda p: False), (None, True))
        self.assertEqual(router.why_missing("/mnt/usb/models/gemma", self.have("/", "/mnt", "/mnt/usb"), empty=lambda p: False), (None, True))

    def test_a_folder_that_is_there_is_not_missing_just_changed(self):
        reason, forget = router.why_missing("/x/y", self.have("/x/y"))
        self.assertFalse(forget)
        self.assertIn("no longer recognised", reason)


class UnfinishedTest(Known):
    def unfinished_copy(self):
        copy = self.root / "qwen-image-partial"
        shutil.copytree(self.folder, copy)
        size = (copy / "model_index.json").stat().st_size
        (copy / setup_download.MANIFEST).write_text(json.dumps({"complete": False, "files": [
            {"path": "model_index.json", "size": size}, {"path": "transformer/never-arrived.safetensors", "size": 1000}]}))
        return copy

    def test_a_download_in_progress_is_listed_with_how_far_it_got_and_is_not_remembered(self):
        copy = self.unfinished_copy()
        found = self.by_folder(self.catalog().entries())[copy.name]
        self.assertEqual((found["switchable"], found["unfinished"], found["verified"]), (False, True, False))
        self.assertEqual(found["reason"], "download incomplete (1 of 2 files here); run `coli setup` to finish it")
        saved = json.loads(Path(self.known).read_text(encoding="utf-8"))["models"]
        self.assertNotIn(found["key"], saved)
        shutil.rmtree(copy)
        self.assertEqual(len(self.catalog().entries()), 1)                       # and when it is deleted nothing is left behind

    def test_the_usable_copy_is_found_by_name_whichever_sorts_first(self):
        copy = self.unfinished_copy()
        catalog = self.catalog()
        entries = catalog.entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual({e["id"] for e in entries}, {entries[0]["id"]})                    # the same model id twice
        found = catalog.find(entries[0]["id"])
        self.assertIsNone(found["reason"])
        self.assertEqual(os.path.basename(found["path"]), "qwen-image")
        self.assertTrue(copy.exists())


class DetailsTest(Known):
    def test_the_details_come_from_the_folders_small_files_and_are_read_once(self):
        catalog = self.catalog()
        entry = catalog.entries()[0]
        details = catalog.details(entry)
        self.assertIsInstance(details["architecture"], str)
        self.assertGreater(details["shards"], 0)
        self.assertNotIn("problems", details)
        with mock.patch("model_info.inspect", side_effect=AssertionError("read again")):
            self.assertIs(catalog.details(entry), details)                         # cached: the folder has not changed

    def test_a_changed_folder_is_read_again_and_a_missing_model_has_no_details(self):
        catalog = self.catalog()
        entry = catalog.entries()[0]
        catalog.details(entry)
        os.utime(self.folder, (1, 1))
        with mock.patch("model_info.inspect", return_value={"problems": ["x"], "architecture": "Y"}) as inspected:
            self.assertEqual(catalog.details(entry), {"architecture": "Y", "problems": ["x"]})
        inspected.assert_called_once()
        self.assertIsNone(catalog.details({"key": "k", "missing": True, "path": "/gone"}))
        self.assertIsNone(catalog.details({"key": "k", "path": os.path.join(self.tmp.name, "nowhere")}))


if __name__ == "__main__":
    unittest.main()
