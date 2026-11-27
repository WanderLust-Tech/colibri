"""The list of MCP servers (c/mcp_registry.py) and the router's endpoints for it.

    python -m unittest tests.test_mcp_registry
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mcp_registry as reg  # noqa: E402


_TEMPS = []
atexit.register(lambda: [shutil.rmtree(folder, ignore_errors=True) for folder in _TEMPS])      # the folders the tests made


def scratch(prefix):
    folder = tempfile.mkdtemp(prefix=prefix)
    _TEMPS.append(folder)
    return folder


def tmp(name="mcp-servers.json"):
    return os.path.join(scratch("coli-mcp-"), name)


GIT = {"command": "uvx", "args": ["mcp-server-git", "--repository", "."]}
WEB = {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${GITHUB_TOKEN}"}}


class CheckTest(unittest.TestCase):
    def test_a_command_server_and_a_url_server_are_kept_as_given(self):
        self.assertEqual(reg.check("git", GIT), GIT)
        self.assertEqual(reg.check("web", WEB), WEB)

    def test_it_is_one_or_the_other(self):
        with self.assertRaisesRegex(reg.RegistryError, "both"):
            reg.check("x", {"command": "a", "url": "http://h/"})
        with self.assertRaisesRegex(reg.RegistryError, "needs a `command`"):
            reg.check("x", {})

    def test_names(self):
        for bad in ("", "has space", "-leading", "a" * 41, "dot.dot", "builtin", "BUILTIN", None, 3):
            with self.subTest(name=bad), self.assertRaises(reg.RegistryError):
                reg.check(bad, GIT)
        for good in ("git", "a", "my-server_2", "A" * 40):
            self.assertEqual(reg.check(good, GIT), GIT)

    def test_unknown_settings_and_wrong_types_name_the_problem(self):
        cases = [({"command": "a", "argz": []}, "argz"), ({"command": "a", "args": "x"}, "args"), ({"command": "a", "args": [1]}, "args"),
                 ({"command": "a", "env": {"A": 1}}, "env"), ({"command": "a", "disabled": "yes"}, "disabled"),
                 ({"command": "a", "timeoutSeconds": 0}, "timeoutSeconds"), ({"command": "a", "timeoutSeconds": True}, "timeoutSeconds"),
                 ({"command": "a", "includeTools": "x"}, "includeTools"), ({"command": 5}, "command"), ({"command": " "}, "command"),
                 ({"url": "ftp://x"}, "url"), ({"url": "http://h/", "args": []}, "url server"), ({"command": "a", "headers": {}}, "command server"),
                 ({"command": "a", "cwd": 3}, "cwd")]
        for cfg, word in cases:
            with self.subTest(cfg=cfg), self.assertRaisesRegex(reg.RegistryError, word):
                reg.check("x", cfg)

    def test_the_server_is_a_copy(self):
        cfg = {"command": "a", "args": ["1"]}
        kept = reg.check("x", cfg)
        kept["args"].append("2")
        self.assertEqual(cfg["args"], ["1"])


class SecretsTest(unittest.TestCase):
    def test_a_secret_must_be_a_reference(self):
        for cfg in ({"command": "a", "env": {"GITHUB_TOKEN": "ghp_abc"}}, {"command": "a", "env": {"api_key": "x"}},
                    {"url": "http://h/", "headers": {"Authorization": "Bearer abc"}}, {"url": "http://h/x?token=abc"},
                    {"url": "http://user:pw@h/"}, {"command": "a", "args": ["--api-key", "abc"]}, {"command": "a", "args": ["--token=abc"]},
                    {"command": "a", "env": {"DB_PASSWORD": "hunter2"}}, {"command": "a", "env": {"SESSION_COOKIE": "x"}},
                    {"command": "a", "env": {"CLIENT_SECRET": "x"}}, {"command": "a", "env": {"PRIVATE_CREDENTIALS": "x"}},
                    {"url": "http://h/", "headers": {"Cookie": "a=b"}}):
            with self.subTest(cfg=cfg), self.assertRaisesRegex(reg.RegistryError, "looks like a secret"):
                reg.check("x", cfg)

    def test_references_and_plain_values_pass(self):
        for cfg in ({"command": "a", "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}}, {"command": "a", "env": {"PATH_EXTRA": "C:/tools", "LANG": "C"}},
                    {"url": "http://h/", "headers": {"Authorization": "Bearer ${T}", "X-Other": "yes"}},
                    {"url": "http://h/x?token=${T}&page=2"}, {"command": "a", "args": ["--token", "${T}"]},
                    {"command": "a", "args": ["--token", "--verbose"]}):
            with self.subTest(cfg=cfg):
                reg.check("x", cfg)

    def test_the_message_says_what_to_write(self):
        with self.assertRaisesRegex(reg.RegistryError, r"\$\{GITHUB_TOKEN\}"):
            reg.check("x", {"command": "a", "env": {"GITHUB_TOKEN": "ghp_abc"}})


class ExpandTest(unittest.TestCase):
    def test_references_are_taken_from_the_environment(self):
        cfg = {"command": "a", "args": ["--t", "${TOK}", "plain"], "env": {"X": "pre-${TOK}-post"}, "cwd": "${HOME_DIR}/p"}
        out = reg.expand(cfg, {"TOK": "abc", "HOME_DIR": "/h"})
        self.assertEqual(out, {"command": "a", "args": ["--t", "abc", "plain"], "env": {"X": "pre-abc-post"}, "cwd": "/h/p"})
        self.assertEqual(cfg["args"][1], "${TOK}", "the server given is not changed")

    def test_a_missing_variable_is_named_and_not_guessed(self):
        with self.assertRaisesRegex(reg.RegistryError, "A, B are not set"):
            reg.expand({"command": "a", "args": ["${B}"], "env": {"X": "${A}"}}, {})
        with self.assertRaisesRegex(reg.RegistryError, "A is not set"):
            reg.expand({"url": "http://h/", "headers": {"Authorization": "Bearer ${A}"}}, {})

    def test_the_process_environment_is_the_default(self):
        with patch.dict(os.environ, {"COLI_TEST_TOK": "v"}):
            self.assertEqual(reg.expand({"url": "http://h/?t=${COLI_TEST_TOK}"})["url"], "http://h/?t=v")


class FileTest(unittest.TestCase):
    def test_put_load_remove(self):
        file = tmp()
        self.assertEqual(reg.load(file), {})
        kept, replaced = reg.put("git", GIT, file)
        self.assertEqual((kept, replaced), (GIT, False))
        self.assertEqual(reg.put("git", dict(GIT, disabled=True), file)[1], True)
        reg.put("web", WEB, file)
        self.assertEqual(sorted(reg.load(file)), ["git", "web"])
        self.assertTrue(reg.load(file)["git"]["disabled"])
        self.assertTrue(reg.remove("git", file))
        self.assertFalse(reg.remove("git", file))
        self.assertEqual(list(reg.load(file)), ["web"])

    def test_the_file_has_claudes_shape(self):
        file = tmp()
        reg.put("git", GIT, file)
        self.assertEqual(json.loads(Path(file).read_text()), {"mcpServers": {"git": GIT}})

    def test_a_server_that_is_refused_changes_nothing(self):
        file = tmp()
        reg.put("git", GIT, file)
        before = Path(file).read_text()
        with self.assertRaises(reg.RegistryError):
            reg.put("bad", {"command": "a", "env": {"MY_TOKEN": "x"}}, file)
        self.assertEqual(Path(file).read_text(), before)

    def test_a_file_that_is_not_valid_says_which_file_and_what(self):
        file = tmp()
        Path(file).write_text('{"mcpServers": {"x": {"command": "a", "nope": 1}}}')
        with self.assertRaisesRegex(reg.RegistryError, "mcp-servers.json.*nope"):
            reg.load(file)
        Path(file).write_text("{broken")
        with self.assertRaisesRegex(reg.RegistryError, "not JSON"):
            reg.load(file)
        Path(file).write_text("   ")
        self.assertEqual(reg.load(file), {})

    def test_a_save_leaves_no_temporary_file(self):
        file = tmp()
        reg.put("git", GIT, file)
        self.assertEqual(os.listdir(os.path.dirname(file)), ["mcp-servers.json"])

    def test_disable_and_enable(self):
        file = tmp()
        reg.put("git", GIT, file)
        reg.set_disabled("git", True, file)
        self.assertTrue(reg.load(file)["git"]["disabled"])
        reg.set_disabled("git", False, file)
        self.assertFalse(reg.load(file)["git"]["disabled"])
        with self.assertRaisesRegex(reg.RegistryError, "no server"):
            reg.set_disabled("nope", True, file)

    def test_the_path_follows_the_environment(self):
        with patch.dict(os.environ, {"COLI_MCP_SERVERS": "~/x/list.json"}):
            self.assertTrue(reg.path().endswith("list.json"))
        with patch.dict(os.environ, {"COLI_MCP_SERVERS": "", "COLI_SETUP_HOME": "/tmp/h"}):
            self.assertEqual(os.path.basename(reg.path()), "mcp-servers.json")
            self.assertIn("h", reg.path())


class TrustTest(unittest.TestCase):
    def test_what_runs_decides_the_digest(self):
        base = reg.digest("git", GIT)
        self.assertEqual(base, reg.digest("git", dict(GIT)))
        for changed in (dict(GIT, command="other"), dict(GIT, args=["x"]), dict(GIT, env={"A": "1"}), dict(GIT, cwd="/elsewhere")):
            self.assertNotEqual(base, reg.digest("git", changed), changed)
        self.assertNotEqual(base, reg.digest("renamed", GIT))
        self.assertNotEqual(reg.digest("w", WEB), reg.digest("w", dict(WEB, url="https://evil.example/mcp")))
        self.assertNotEqual(reg.digest("w", WEB), reg.digest("w", dict(WEB, headers={"Authorization": "Bearer ${OTHER}"})))

    def test_what_does_not_run_does_not(self):
        base = reg.digest("git", GIT)
        for same in (dict(GIT, disabled=True), dict(GIT, includeTools=["a"]), dict(GIT, excludeTools=["b"]), dict(GIT, timeoutSeconds=5),
                     dict(GIT, connectTimeoutSeconds=9)):
            self.assertEqual(base, reg.digest("git", same), same)

    def test_trust_is_kept_per_client_and_lost_when_the_definition_changes(self):
        file = tmp("trust.json")
        self.assertFalse(reg.is_trusted("git", GIT, file))
        reg.trust("git", GIT, file)
        self.assertTrue(reg.is_trusted("git", GIT, file))
        self.assertFalse(reg.is_trusted("git", dict(GIT, args=["--evil"]), file))
        self.assertFalse(reg.is_trusted("other", GIT, file))
        reg.forget("git", file)
        self.assertFalse(reg.is_trusted("git", GIT, file))
        reg.forget("git", file)                                   # again: no error

    def test_a_broken_trust_file_trusts_nothing(self):
        file = tmp("trust.json")
        Path(file).write_text("{nope")
        self.assertEqual(reg.trusted(file), {})
        Path(file).write_text("[1]")
        self.assertEqual(reg.trusted(file), {})


class DescribeTest(unittest.TestCase):
    def test_one_line_a_person_can_check(self):
        self.assertEqual(reg.describe(GIT), "uvx mcp-server-git --repository .")
        self.assertEqual(reg.describe({"command": "C:/Program Files/x/server.exe", "args": ["a b", "c"]}), '"C:/Program Files/x/server.exe" "a b" c')
        self.assertEqual(reg.describe(WEB), "https://example.com/mcp")


class RouterMcpTest(unittest.TestCase):
    """GET /v1/mcp/servers, POST /v1/mcp/servers and POST /v1/mcp/servers/delete on a router over fake workers."""

    def setUp(self):
        sys.path.insert(0, str(HERE))
        from test_router import Rig, entry
        self.file = tmp()
        patcher = patch.dict(os.environ, {"COLI_MCP_SERVERS": self.file})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.rig = Rig(self, [entry("text-a")])

    def test_an_empty_list_and_then_a_server_that_is_listed_with_what_it_runs(self):
        status, _, _, body = self.rig.call("GET", "/v1/mcp/servers")
        self.assertEqual((status, body["data"], body["problem"], body["path"]), (200, [], None, self.file))
        status, _, _, body = self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="git"))
        self.assertEqual((status, body["status"], body["name"]), (200, "created", "git"))
        self.assertIn("uvx mcp-server-git", body["summary"])
        listed = self.rig.call("GET", "/v1/mcp/servers")[3]["data"]
        self.assertEqual([(s["name"], s["command"], s["args"]) for s in listed], [("git", "uvx", GIT["args"])])
        self.assertIn("mcp-server-git", listed[0]["summary"])
        self.assertEqual(self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="git", disabled=True))[3]["status"], "replaced")
        self.assertTrue(self.rig.call("GET", "/v1/mcp/servers")[3]["data"][0]["disabled"])
        self.assertEqual(reg.load(self.file)["git"]["disabled"], True, "the file is what was changed")

    def test_toolmode_is_kept_listed_and_checked(self):
        status, _, _, body = self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="big", toolMode="search"))
        self.assertEqual((status, body["status"]), (200, "created"))
        listed = {s["name"]: s for s in self.rig.call("GET", "/v1/mcp/servers")[3]["data"]}
        self.assertEqual(listed["big"]["toolMode"], "search")
        self.assertEqual(reg.load(self.file)["big"]["toolMode"], "search")
        before = Path(self.file).read_text()
        status, _, _, got = self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="odd", toolMode="lazy"))
        self.assertEqual(status, 400)
        self.assertIn("toolMode", got["error"]["message"])
        self.assertEqual(Path(self.file).read_text(), before)

    def test_what_is_refused_is_a_400_with_the_reason_and_the_file_is_untouched(self):
        self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="git"))
        before = Path(self.file).read_text()
        for body, word in ((dict(GIT, name="has space"), "name"), ({"name": "x"}, "needs a `command`"),
                           ({"name": "x", "command": "a", "env": {"MY_TOKEN": "plain"}}, "looks like a secret"),
                           ({"name": "x", "command": "a", "surprise": 1}, "surprise"), ({"command": "a"}, "name")):
            with self.subTest(body=body):
                status, _, _, got = self.rig.call("POST", "/v1/mcp/servers", body)
                self.assertEqual(status, 400)
                self.assertIn(word, got["error"]["message"])
        self.assertEqual(Path(self.file).read_text(), before)

    def test_delete(self):
        self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="git"))
        status, _, _, body = self.rig.call("POST", "/v1/mcp/servers/delete", {"name": "git"})
        self.assertEqual((status, body["status"]), (200, "deleted"))
        self.assertEqual(self.rig.call("GET", "/v1/mcp/servers")[3]["data"], [])
        got = self.rig.call("POST", "/v1/mcp/servers/delete", {"name": "git"})
        self.assertEqual((got[0], got[3]["error"]["code"]), (404, "mcp_server_not_found"))
        self.assertEqual(self.rig.call("POST", "/v1/mcp/servers/delete", {})[0], 400)

    def test_a_list_file_that_is_not_valid_is_reported_not_hidden(self):
        Path(self.file).write_text('{"mcpServers": {"x": {"command": "a", "nope": 1}}}')
        body = self.rig.call("GET", "/v1/mcp/servers")[3]
        self.assertEqual(body["data"], [])
        self.assertIn("nope", body["problem"])
        self.assertEqual(self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="git"))[0], 400, "and it is not overwritten blindly")
        self.assertIn("nope", Path(self.file).read_text())

    def test_it_needs_the_key_json_and_the_same_origin(self):
        self.rig.server.api_key = "secret"
        self.assertEqual(self.rig.call("GET", "/v1/mcp/servers")[0], 401)
        self.assertEqual(self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="git"))[0], 401)
        keyed = {"Authorization": "Bearer secret"}
        self.assertEqual(self.rig.call("POST", "/v1/mcp/servers", raw=b"name=t", headers=dict(keyed, **{"Content-Type": "text/plain"}))[0], 415)
        self.assertEqual(self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="git"), headers=dict(keyed, Origin="http://evil.example"))[0], 403)
        self.assertFalse(os.path.exists(self.file))
        self.assertEqual(self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="git"), headers=keyed)[0], 200)

    def test_another_machine_may_read_the_list_but_not_change_it(self):
        self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="git"))
        import router
        with patch.object(router, "LOOPBACK", set()):
            self.assertEqual(self.rig.call("GET", "/v1/mcp/servers")[0], 200)
            for path, body in (("/v1/mcp/servers", dict(GIT, name="evil")), ("/v1/mcp/servers/delete", {"name": "git"})):
                status, _, _, got = self.rig.call("POST", path, body)
                self.assertEqual((status, got["error"]["code"]), (403, "forbidden"))
            self.assertEqual(list(reg.load(self.file)), ["git"])
            with patch.dict(os.environ, {"COLI_MCP_REMOTE_EDIT": "1"}):
                self.assertEqual(self.rig.call("POST", "/v1/mcp/servers", dict(GIT, name="other"))[0], 200)


if __name__ == "__main__":
    unittest.main()
