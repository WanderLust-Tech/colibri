"""`coli tools allow | deny | web` and the built-in section of `coli tools`: the policy of the router's own tools, set on this PC."""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
COLI = str(C_DIR / "coli")


class ToolsCliTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = os.path.realpath(temp.name)
        self.policy = os.path.join(self.root, "policy.json")
        self.folder = os.path.join(self.root, "project")
        os.makedirs(self.folder)
        self.env = dict(os.environ, COLI_TOOLS_POLICY=self.policy, COLI_MCP_SERVERS=os.path.join(self.root, "mcp.json"),
                        COLI_SETUP_HOME=os.path.join(self.root, "setup"), PYTHONIOENCODING="utf-8", NO_COLOR="1")

    def coli(self, *args):
        done = subprocess.run([sys.executable, COLI, "tools", *args], capture_output=True, env=self.env, timeout=120)
        out = (done.stdout + done.stderr).decode("utf-8", "replace")
        return done.returncode, re.sub(r"\x1b\[[0-9;]*m", "", out)

    def saved(self):
        return json.loads(Path(self.policy).read_text())

    def test_nothing_is_on_and_the_list_says_how_to_turn_it_on(self):
        code, out = self.coli()
        self.assertEqual(code, 0)
        self.assertIn("built in", out)
        self.assertIn("on: nothing", out)
        self.assertIn("coli tools allow FOLDER", out)
        self.assertFalse(os.path.exists(self.policy), "looking changes nothing")

    def test_allow_read_only_then_with_write_then_web_then_deny(self):
        code, out = self.coli("allow", self.folder)
        self.assertEqual(code, 0, out)
        self.assertIn("read only", out)
        self.assertEqual(self.saved(), {"folders": [{"path": os.path.normpath(self.folder), "write": False}], "web": False})
        listed = self.coli()[1]
        for tool in ("files_read", "git_log"):
            self.assertIn(tool, listed)
        self.assertNotIn("files_write", listed, "no write tool until a folder may be written")
        self.assertIn("read only", listed)

        code, out = self.coli("allow", self.folder, "--write")
        self.assertEqual(code, 0, out)
        self.assertIn("read and write", out)
        self.assertEqual([f["write"] for f in self.saved()["folders"]], [True], "the same folder is replaced")
        self.assertIn("files_write", self.coli()[1])

        code, out = self.coli("web", "on")
        self.assertEqual(code, 0, out)
        self.assertTrue(self.saved()["web"])
        self.assertIn("web_fetch", self.coli()[1])
        self.assertEqual(self.coli("web", "off")[0], 0)
        self.assertFalse(self.saved()["web"])

        code, out = self.coli("deny", self.folder)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.saved()["folders"], [])
        code, out = self.coli("deny", self.folder)
        self.assertEqual(code, 1)
        self.assertIn("was not allowed", out)

    def test_what_cannot_be_allowed_is_refused_with_the_reason_and_nothing_is_written(self):
        top = os.path.abspath(os.sep)
        for args, word in ((("allow", top), "top of a drive"), (("allow", os.path.join(self.root, "missing")), "not a folder that exists"),
                           (("allow",), "FOLDER"), (("web",), "on|off"), (("web", "maybe"), "on|off"), (("deny",), "FOLDER")):
            with self.subTest(args=args):
                code, out = self.coli(*args)
                self.assertEqual(code, 1, out)
                self.assertIn(word, out)
        self.assertFalse(os.path.exists(self.policy))

    def test_a_policy_file_that_is_not_valid_is_reported_and_left_alone(self):
        Path(self.policy).write_text('{"surprise": 1}')
        code, out = self.coli("allow", self.folder)
        self.assertEqual(code, 1)
        self.assertIn("surprise", out)
        self.assertEqual(Path(self.policy).read_text(), '{"surprise": 1}')
        self.assertIn("surprise", self.coli()[1])

    def test_the_policy_is_changed_on_the_routers_pc_only(self):
        code, out = self.coli("allow", self.folder, "--host", "192.0.2.7")
        self.assertEqual(code, 1)
        self.assertIn("PC the router runs on", out)
        self.assertFalse(os.path.exists(self.policy))


if __name__ == "__main__":
    unittest.main()
