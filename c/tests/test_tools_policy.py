"""tools_policy.py: what the router's built-in tools may touch, and the check every path goes through."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))

import tools_policy as tp  # noqa: E402


def scratch():
    folder = tempfile.TemporaryDirectory()
    return folder, os.path.realpath(folder.name)


def link(target, name, directory=True):
    """A link at `name` to `target`: a symbolic link, or on Windows a junction when links are not allowed. None when neither works."""
    try:
        os.symlink(target, name, target_is_directory=directory)
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt" and directory:
        done = subprocess.run(["cmd", "/c", "mklink", "/J", name, target], capture_output=True)
        return done.returncode == 0
    return False


class ParseTest(unittest.TestCase):
    def test_nothing_is_on_by_default(self):
        policy = tp.parse("{}")
        self.assertEqual((policy.folders, policy.web, policy.allow_secrets, policy.readable, policy.writable), ([], False, False, False, False))
        self.assertEqual(tp.Policy().summary()["folders"], [])

    def test_a_missing_file_is_everything_off_not_an_error(self):
        folder, root = scratch()
        self.addCleanup(folder.cleanup)
        self.assertFalse(tp.load(os.path.join(root, "nope.json")).readable)

    def test_what_is_wrong_is_named(self):
        folder, root = scratch()
        self.addCleanup(folder.cleanup)
        for text, word in (("{nope", "JSON"), ("[1]", "object"), ('{"surprise": 1}', "surprise"), ('{"folders": [{"nopath": 1}]}', "path"),
                           ('{"folders": [{"path": "' + root.replace("\\", "/") + '", "writ": true}]}', "writ"),
                           ('{"folders": [{"path": "' + root.replace("\\", "/") + '", "write": "yes"}]}', "write"),
                           ('{"web": "on"}', "web"), ('{"approvalSeconds": 0}', "approvalSeconds"), ('{"approvalSeconds": true}', "approvalSeconds"),
                           ('{"disabled": "git_log"}', "disabled"), ('{"allowSecrets": 1}', "allowSecrets")):
            with self.subTest(text=text), self.assertRaises(tp.PolicyError) as caught:
                tp.parse(text)
            self.assertIn(word, str(caught.exception))

    def test_the_top_of_a_drive_is_refused(self):
        top = os.path.abspath(os.sep)
        with self.assertRaises(tp.PolicyError) as caught:
            tp.parse(json.dumps({"folders": [top]}))
        self.assertIn("top of a drive", str(caught.exception))

    def test_a_folder_may_be_a_plain_string_and_is_made_absolute(self):
        folder, root = scratch()
        self.addCleanup(folder.cleanup)
        policy = tp.parse(json.dumps({"folders": [root + os.sep + "." + os.sep]}))
        self.assertEqual(policy.folders, [{"path": os.path.normpath(root), "write": False}])


class FileTest(unittest.TestCase):
    def setUp(self):
        self.folder, self.root = scratch()
        self.addCleanup(self.folder.cleanup)
        self.file = os.path.join(self.root, "policy.json")

    def test_allow_deny_and_web_are_kept_in_the_file(self):
        folder, kept = tp.allow(self.root, file=self.file)[1], None
        policy = tp.load(self.file)
        self.assertEqual(policy.folders, [{"path": os.path.normpath(self.root), "write": False}])
        policy, _ = tp.allow(self.root, write=True, file=self.file)
        self.assertEqual([f["write"] for f in policy.folders], [True], "the same folder is replaced, not doubled")
        self.assertTrue(tp.set_web(True, self.file).web)
        self.assertTrue(tp.load(self.file).web)
        policy, was = tp.deny(self.root, file=self.file)
        self.assertEqual((policy.folders, was), ([], True))
        self.assertFalse(tp.deny(self.root, file=self.file)[1], "a folder that was not there says so")
        self.assertTrue(tp.load(self.file).web, "and the other settings stay")

    def test_a_folder_that_does_not_exist_is_refused_before_anything_is_written(self):
        with self.assertRaises(tp.PolicyError):
            tp.allow(os.path.join(self.root, "missing"), file=self.file)
        self.assertFalse(os.path.exists(self.file))

    def test_a_file_that_is_not_valid_is_reported_and_not_overwritten(self):
        Path(self.file).write_text('{"surprise": 1}')
        with self.assertRaises(tp.PolicyError):
            tp.allow(self.root, file=self.file)
        self.assertIn("surprise", Path(self.file).read_text())

    def test_the_written_file_is_what_a_reader_accepts(self):
        tp.allow(self.root, write=True, file=self.file)
        tp.set_web(True, self.file)
        data = json.loads(Path(self.file).read_text())
        self.assertEqual(set(data), {"folders", "web"})
        tp.parse(json.dumps(data))


class ResolveTest(unittest.TestCase):
    """Every path a tool is given goes through Policy.resolve: this is the line between the allowed folders and the rest of the PC."""

    def setUp(self):
        self.folder, self.root = scratch()
        self.addCleanup(self.folder.cleanup)
        self.work = os.path.join(self.root, "work")
        self.evil = os.path.join(self.root, "work-evil")          # a sibling whose name starts with the allowed folder's
        self.other = os.path.join(self.root, "other")
        for d in (self.work, self.evil, self.other, os.path.join(self.work, "sub")):
            os.makedirs(d, exist_ok=True)
        Path(self.work, "a.txt").write_text("a")
        Path(self.evil, "x.txt").write_text("x")
        Path(self.other, "o.txt").write_text("o")
        self.policy = tp.Policy([{"path": self.work, "write": False}])

    def test_inside_is_allowed_by_full_or_relative_path(self):
        self.assertEqual(self.policy.resolve(os.path.join(self.work, "a.txt")), os.path.join(self.work, "a.txt"))
        self.assertEqual(self.policy.resolve("a.txt"), os.path.join(self.work, "a.txt"))
        self.assertEqual(self.policy.resolve("sub"), os.path.join(self.work, "sub"))
        self.assertEqual(self.policy.resolve(self.work), self.work)
        self.assertEqual(self.policy.resolve("."), self.work)

    def test_a_new_file_inside_resolves_too(self):
        self.assertEqual(self.policy.resolve("sub/new.txt"), os.path.join(self.work, "sub", "new.txt"))

    def test_outside_is_refused_however_it_is_written(self):
        for raw in (os.path.join(self.other, "o.txt"), self.root, os.path.join(self.evil, "x.txt"), self.evil, "..", "../other/o.txt",
                    "sub/../../other/o.txt", os.path.join(self.work, "..", "other", "o.txt"), "..\\other\\o.txt"):
            with self.subTest(raw=raw), self.assertRaises(tp.PolicyError):
                self.policy.resolve(raw)

    def test_a_sibling_with_the_same_start_of_its_name_is_not_inside(self):
        with self.assertRaises(tp.PolicyError):
            self.policy.resolve(self.evil + os.sep + "x.txt")

    @unittest.skipUnless(os.name == "nt", "case-insensitive file names")
    def test_the_case_of_a_windows_path_does_not_matter(self):
        self.assertEqual(os.path.normcase(self.policy.resolve(self.work.upper() + "\\A.TXT")), os.path.normcase(os.path.join(self.work, "a.txt")))

    def test_a_link_inside_that_points_out_does_not_lead_out(self):
        if not link(self.other, os.path.join(self.work, "escape")):
            self.skipTest("this machine cannot make links")
        with self.assertRaises(tp.PolicyError):
            self.policy.resolve("escape/o.txt")
        with self.assertRaises(tp.PolicyError):
            self.policy.resolve("escape")
        self.assertEqual(self.policy.resolve("a.txt"), os.path.join(self.work, "a.txt"), "and a plain file still works")

    def test_a_link_inside_that_points_inside_is_fine(self):
        if not link(os.path.join(self.work, "sub"), os.path.join(self.work, "alias")):
            self.skipTest("this machine cannot make links")
        self.assertEqual(self.policy.resolve("alias"), os.path.join(self.work, "sub"))

    def test_the_allowed_folder_may_itself_be_reached_through_a_link(self):
        if not link(self.work, os.path.join(self.root, "via")):
            self.skipTest("this machine cannot make links")
        policy = tp.Policy([{"path": os.path.join(self.root, "via"), "write": False}])
        self.assertEqual(policy.resolve("a.txt"), os.path.join(self.work, "a.txt"))
        with self.assertRaises(tp.PolicyError):
            policy.resolve(os.path.join(self.other, "o.txt"))

    def test_odd_input_is_refused(self):
        for raw in ("", "   ", None, 5, ["a"], "a\0b", "//server/share/x", "\\\\server\\share\\x", "\\\\?\\C:\\Windows", "\\\\.\\pipe\\x"):
            with self.subTest(raw=raw), self.assertRaises(tp.PolicyError):
                self.policy.resolve(raw)

    def test_a_network_path_is_refused_before_anything_resolves_it(self):
        """Resolving \\\\server\\share can make the machine open a connection (and send credentials), so the refusal must come first."""
        touched = []
        real = os.path.realpath

        def watching(path, *args, **kwargs):
            touched.append(path)
            return real(path, *args, **kwargs)

        with unittest.mock.patch.object(tp.os.path, "realpath", watching):
            for raw in ("\\\\server\\share\\f", "//server/share/f", "\\\\?\\UNC\\server\\share", "\\\\.\\pipe\\x",
                        "/\\server\\share\\f", "\\/server/share/f"):
                if raw in ("//server/share/f",) or os.name == "nt" or raw.startswith("\\\\"):
                    with self.subTest(raw=raw), self.assertRaises(tp.PolicyError):
                        self.policy.resolve(raw)
        self.assertEqual(touched, [], "a network path was handed to realpath")

    @unittest.skipUnless(os.name == "nt", "NTFS streams")
    def test_an_ntfs_stream_is_refused(self):
        with self.assertRaises(tp.PolicyError):
            self.policy.resolve("a.txt:hidden")

    def test_writing_needs_a_folder_that_may_be_written(self):
        with self.assertRaises(tp.PolicyError) as caught:
            self.policy.resolve("a.txt", write=True)
        self.assertIn("--write", str(caught.exception))
        both = tp.Policy([{"path": self.work, "write": False}, {"path": self.other, "write": True}])
        self.assertEqual(both.resolve(os.path.join(self.other, "o.txt"), write=True), os.path.join(self.other, "o.txt"))
        with self.assertRaises(tp.PolicyError):
            both.resolve(os.path.join(self.work, "a.txt"), write=True)
        self.assertEqual(both.resolve(os.path.join(self.work, "a.txt")), os.path.join(self.work, "a.txt"), "reading is still fine")

    def test_no_folder_means_no_path(self):
        with self.assertRaises(tp.PolicyError) as caught:
            tp.Policy().resolve("a.txt")
        self.assertIn("coli tools allow", str(caught.exception))

    def test_a_relative_path_with_several_folders_asks_for_a_full_one(self):
        both = tp.Policy([{"path": self.work, "write": False}, {"path": self.other, "write": False}])
        with self.assertRaises(tp.PolicyError) as caught:
            both.resolve("a.txt")
        self.assertIn("full path", str(caught.exception))
        self.assertIsNone(both.default_folder())
        self.assertEqual(self.policy.default_folder(), self.work)


class SecretTest(unittest.TestCase):
    def test_names_that_say_credential(self):
        for name in (".env", ".env.local", "server.pem", "my.key", "id_rsa", "id_ed25519", "ID_RSA", ".npmrc", ".netrc", ".git-credentials",
                     "credentials", "credentials.json", "secrets.yaml", "terraform.tfstate", "x.kdbx", "app.p12"):
            with self.subTest(name=name):
                self.assertTrue(tp.looks_secret(os.path.join("C:", os.sep, "work", name)))
        for folder in (".ssh", ".aws", ".gnupg", ".kube"):
            self.assertTrue(tp.looks_secret(os.path.join("C:", os.sep, "home", folder, "config")), folder)

    def test_names_that_do_not(self):
        for name in ("main.c", "README.md", "environment.md", "keys.txt", "monkey.py", "id_rsa.txt", "envoy.yaml", "secretary.md", ".envrc.md"):
            with self.subTest(name=name):
                self.assertFalse(tp.looks_secret(os.path.join("C:", os.sep, "work", name)))


if __name__ == "__main__":
    unittest.main()
