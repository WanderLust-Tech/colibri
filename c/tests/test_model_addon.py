"""Add-ons of the model installer (model_installer.py): files that go into the folder of a model that is already installed.

ACE-Step's planner is a subfolder of the main repository and its base DiT a repository of its own; both land beside the turbo DiT
in the checkpoint's folder, not in a folder of their own. A fake hub serves a parent model and the two kinds of add-on, and the
tests hold the installer to: a parent first, the subfolder's paths (a prefix taken off, the files fetched from where they are), an
install that is its own job with its own manifest and leaves the parent's alone, a folder put together by hand counting as
installed, a removal that takes only the add-on, and `coli setup` never offering one."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))
import model_installer  # noqa: E402
import setup_catalog  # noqa: E402
import setup_download  # noqa: E402
from fake_hub import FakeHub  # noqa: E402
from test_model_installer import InstallerCase, entry  # noqa: E402

PARENT = "someone/tiny-parent"
SEPARATE = "someone/tiny-dit"
WITHIN = "someone/tiny-parent-with-lm"


class AddonCase(InstallerCase):
    def setUp(self):
        super().setUp()
        base = self.tmp.name
        self.dit = os.path.join(base, "dit")
        os.makedirs(self.dit)
        self.dit_weights = os.urandom(120_000)
        Path(self.dit, "config.json").write_text("{}")
        Path(self.dit, "model.safetensors").write_bytes(self.dit_weights)
        Path(self.dit, "silence_latent.pt").write_bytes(b"s" * 100)
        Path(self.dit, "modeling.py").write_text("# never read")
        self.within = os.path.join(base, "within")
        os.makedirs(os.path.join(self.within, "lm-1.7B"))
        os.makedirs(os.path.join(self.within, "other-1.7B"))
        self.lm_weights = os.urandom(90_000)
        Path(self.within, "model_index.json").write_text("{}")
        Path(self.within, "lm-1.7B", "config.json").write_text("{}")
        Path(self.within, "lm-1.7B", "model.safetensors").write_bytes(self.lm_weights)
        Path(self.within, "lm-1.7B", "tokenizer.json").write_text("{}")
        Path(self.within, "other-1.7B", "config.json").write_text("{}")
        self.hub.repos.update({PARENT: self.src, SEPARATE: self.dit, WITHIN: self.within})
        self.entries = [
            entry(id="parent", repo=PARENT),
            entry(id="parent-dit", repo=SEPARATE, name="DiT add-on", disk_gb=0.001, into="parent", subdir="dit",
                  include=("config.json", "model.safetensors", "silence_latent.pt"),
                  check=("config.json", "model.safetensors", "silence_latent.pt")),
            entry(id="parent-lm", repo=WITHIN, name="LM add-on", disk_gb=0.001, into="parent", subdir="lm",
                  include=("lm-1.7B/*",), strip="lm-1.7B/", check=("config.json", "model.safetensors", "tokenizer.json")),
        ]

    def rows(self):
        return {r["id"]: r for r in self.installer.catalog_rows()}

    def install_parent(self):
        self.assertEqual(self.installer.start("parent", confirm=True, background=False)["state"], "done")


class Installing(AddonCase):
    def test_without_the_parent_it_is_refused_and_the_row_says_what_to_do(self):
        row = self.rows()["parent-dit"]
        self.assertEqual((row["status"], row["fits"], row["into"]), ("needs-parent", False, "parent"))
        self.assertIn("coli model install parent", row["reason"])
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.start("parent-dit", confirm=True)
        self.assertEqual((caught.exception.status, caught.exception.code), (409, "needs_parent"))
        self.assertEqual(os.listdir(self.root), [])

    def test_a_separate_repository_lands_in_the_subfolder_and_only_what_is_included(self):
        self.install_parent()
        job = self.installer.start("parent-dit", confirm=True, background=False)
        self.assertEqual(job["state"], "done", job)
        folder = os.path.join(self.root, "parent", "dit")
        self.assertEqual(Path(folder, "model.safetensors").read_bytes(), self.dit_weights)
        self.assertEqual(sorted(n for n in os.listdir(folder) if not n.startswith(".")),
                         ["config.json", "model.safetensors", "silence_latent.pt"])
        self.assertTrue(setup_download.is_complete(folder))
        self.assertFalse(os.path.exists(os.path.join(folder, model_installer.LOCK_NAME)))

    def test_a_subfolder_of_a_repository_comes_without_its_prefix_from_where_it_is(self):
        self.install_parent()
        job = self.installer.start("parent-lm", confirm=True, background=False)
        self.assertEqual(job["state"], "done", job)
        folder = os.path.join(self.root, "parent", "lm")
        self.assertEqual(Path(folder, "model.safetensors").read_bytes(), self.lm_weights)
        self.assertFalse(os.path.exists(os.path.join(folder, "other-1.7B")), "a sibling folder came along")
        self.assertFalse(os.path.exists(os.path.join(folder, "lm-1.7B")), "the prefix was kept")
        self.assertEqual(sorted(f["path"] for f in setup_download.read_manifest(folder)["files"]),
                         ["config.json", "model.safetensors", "tokenizer.json"])

    def test_the_parents_own_files_and_manifest_are_not_touched(self):
        self.install_parent()
        parent = os.path.join(self.root, "parent")
        before = Path(parent, setup_download.MANIFEST).read_text()
        self.installer.start("parent-dit", confirm=True, background=False)
        self.assertEqual(Path(parent, setup_download.MANIFEST).read_text(), before)
        self.assertTrue(setup_download.is_complete(parent))

    def test_the_parent_is_told_when_an_add_on_has_landed(self):
        told = []
        self.installer.on_addon = told.append
        self.install_parent()
        self.assertEqual(told, [], "a model of its own is not an add-on")
        self.installer.start("parent-dit", confirm=True, background=False)
        self.assertEqual([t["id"] for t in told], ["parent"])
        self.installer.on_addon = lambda parent: 1 / 0          # a failing courtesy does not undo the install
        self.assertEqual(self.installer.start("parent-lm", confirm=True, background=False)["state"], "done")

    def test_the_catalog_shows_it_installed_and_a_second_install_is_refused(self):
        self.install_parent()
        self.installer.start("parent-dit", confirm=True, background=False)
        row = self.rows()["parent-dit"]
        self.assertEqual(row["installed"]["id"], "parent-dit")
        self.assertGreater(row["installed"]["size_bytes"], 100_000)
        self.assertIsNone(self.rows()["parent-lm"]["installed"])
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.start("parent-dit", confirm=True)
        self.assertEqual(caught.exception.code, "already_installed")

    def test_a_folder_put_together_by_hand_counts_as_installed(self):
        self.install_parent()
        folder = os.path.join(self.root, "parent", "dit")
        os.makedirs(folder)
        for name in ("config.json", "model.safetensors", "silence_latent.pt"):
            Path(folder, name).write_bytes(b"x")
        self.assertIsNotNone(self.rows()["parent-dit"]["installed"])

    def test_a_cancelled_download_is_partial_and_resumes(self):
        self.install_parent()
        folder = os.path.join(self.root, "parent", "dit")
        os.makedirs(folder)
        Path(folder, "model.safetensors.part").write_bytes(self.dit_weights[:50_000])
        row = self.rows()["parent-dit"]
        self.assertTrue(row["unfinished"])
        self.assertIsNone(row["installed"])
        self.assertEqual(self.installer.start("parent-dit", confirm=True, background=False)["state"], "done")
        self.assertEqual(Path(folder, "model.safetensors").read_bytes(), self.dit_weights)

    def test_a_folder_that_is_not_ours_is_not_written_into(self):
        self.install_parent()
        folder = os.path.join(self.root, "parent", "dit")
        os.makedirs(folder)
        Path(folder, "notes.txt").write_text("mine")
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.start("parent-dit", confirm=True)
        self.assertEqual(caught.exception.code, "folder_exists")
        self.assertEqual(os.listdir(folder), ["notes.txt"])


class Removing(AddonCase):
    def setUp(self):
        super().setUp()
        self.install_parent()
        self.installer.start("parent-dit", confirm=True, background=False)
        self.installer.start("parent-lm", confirm=True, background=False)

    def test_confirm_is_required(self):
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.uninstall_addon("parent-dit")
        self.assertEqual(caught.exception.code, "confirmation_required")
        self.assertTrue(os.path.isdir(os.path.join(self.root, "parent", "dit")))

    def test_only_the_add_on_goes_and_the_size_is_reported(self):
        plan = self.installer.addon_plan("parent-dit")
        self.assertGreater(plan["bytes"], 100_000)
        result = self.installer.uninstall_addon("parent-dit", confirm=True)
        self.assertEqual((result["status"], result["freed_bytes"]), ("removed", plan["bytes"]))
        parent = os.path.join(self.root, "parent")
        self.assertFalse(os.path.exists(os.path.join(parent, "dit")))
        self.assertTrue(os.path.isdir(os.path.join(parent, "lm")), "the other add-on went too")
        self.assertTrue(setup_download.is_complete(parent))
        self.assertEqual([n for n in os.listdir(parent) if n.endswith(model_installer.DELETING)], [])
        self.assertIsNone(self.rows()["parent-dit"]["installed"])

    def test_it_can_be_installed_again_afterwards(self):
        self.installer.uninstall_addon("parent-lm", confirm=True)
        self.assertEqual(self.installer.start("parent-lm", confirm=True, background=False)["state"], "done")

    def test_what_is_not_installed_or_not_an_add_on_is_refused(self):
        self.installer.uninstall_addon("parent-dit", confirm=True)
        for name, status, code in (("parent-dit", 409, "model_missing"), ("parent", 404, "model_not_found"),
                                   ("nope", 404, "model_not_found")):
            with self.assertRaises(model_installer.InstallError) as caught:
                self.installer.addon_plan(name)
            self.assertEqual((caught.exception.status, caught.exception.code), (status, code), name)

    def test_a_link_is_refused_and_its_target_is_left(self):
        target = os.path.join(self.tmp.name, "elsewhere")
        os.makedirs(target)
        Path(target, "keep.txt").write_text("keep")
        folder = os.path.join(self.root, "parent", "lm")
        self.installer.uninstall_addon("parent-lm", confirm=True)
        try:
            os.symlink(target, folder, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("no symbolic links here")
        Path(target, "config.json").write_text("{}")
        Path(target, "model.safetensors").write_text("{}")
        Path(target, "tokenizer.json").write_text("{}")
        with self.assertRaises(model_installer.InstallError) as caught:
            self.installer.uninstall_addon("parent-lm", confirm=True)
        self.assertEqual(caught.exception.code, "model_is_link")
        self.assertTrue(os.path.exists(os.path.join(target, "keep.txt")))


class Setup(unittest.TestCase):
    def test_coli_setup_never_offers_an_add_on(self):
        rows = setup_catalog.recommend(64 * 10 ** 9, 800 * 10 ** 9)
        offered = {r["entry"].id for r in rows}
        addons = {e.id for e in setup_catalog.CATALOG if e.into}
        self.assertTrue(addons, "the catalog has no add-on")
        self.assertFalse(offered & addons)

    def test_every_add_on_names_a_catalog_model_a_folder_and_a_way_to_tell_it_is_there(self):
        ids = {e.id for e in setup_catalog.CATALOG}
        for e in setup_catalog.CATALOG:
            if e.into:
                with self.subTest(entry=e.id):
                    self.assertIn(e.into, ids)
                    self.assertFalse(setup_catalog.by_id(e.into).into, "an add-on of an add-on")
                    self.assertRegex(e.subdir, r"^[\w.-]+$")
                    self.assertTrue(e.check)
                    self.assertEqual(setup_catalog.by_id(e.into).family, e.family)
                    if e.strip:
                        self.assertTrue(e.strip.endswith("/"))
                        self.assertTrue(all(p.startswith(e.strip) for p in e.include))


if __name__ == "__main__":
    unittest.main()
