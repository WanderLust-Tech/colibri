"""builtin_tools.py: the files and git tools over a real folder and a real repository, and the rules that keep them inside it."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import builtin_tools as bt  # noqa: E402
import tools_policy as tp  # noqa: E402

HAVE_GIT = shutil.which("git") is not None


def git(repo, *args):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    done = subprocess.run(["git", "-C", repo, *args], capture_output=True, env=env, text=True, encoding="utf-8")
    assert done.returncode == 0, (args, done.stdout, done.stderr)
    return done.stdout.strip()


def make_repo(repo):
    os.makedirs(repo, exist_ok=True)
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test Person")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "commit.gpgsign", "false")
    git(repo, "config", "core.autocrlf", "false")


class Folder(unittest.TestCase):
    """A scratch folder with a project in it, allowed for reading (and a second, allowed for writing)."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = os.path.realpath(temp.name)
        self.proj = os.path.join(self.root, "proj")
        self.scratch = os.path.join(self.root, "scratch")
        self.outside = os.path.join(self.root, "outside")
        for d in (self.proj, self.scratch, self.outside, os.path.join(self.proj, "src"), os.path.join(self.proj, ".git-not"), os.path.join(self.proj, "node_modules")):
            os.makedirs(d, exist_ok=True)
        self.write("proj/src/main.c", "int main() {\n  return 0;\n}\n")
        self.write("proj/README.md", "# Project\nhello world\nHELLO again\n")
        self.write("proj/.env", "TOKEN=abc123\n")
        self.write("proj/server.pem", "-----BEGIN PRIVATE KEY-----\n")
        self.write("proj/node_modules/dep.js", "hello from a dependency\n")
        Path(self.proj, "blob.bin").write_bytes(b"\x00\x01\x02hello\x00")
        self.write("outside/secret.txt", "outside the folders\n")
        self.read_only = tp.Policy([{"path": self.proj, "write": False}])
        self.both = tp.Policy([{"path": self.proj, "write": False}, {"path": self.scratch, "write": True}])

    def write(self, rel, text):
        path = os.path.join(self.root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return path

    def ok(self, name, args, policy=None):
        text, error = bt.run(name, args, policy or self.read_only)
        self.assertFalse(error, text)
        return text

    def refused(self, name, args, word, policy=None):
        text, error = bt.run(name, args, policy or self.read_only)
        self.assertTrue(error, f"{name} {args} should have been refused, it said: {text[:200]}")
        self.assertIn(word, text)
        return text


class SelectionTest(Folder):
    def names(self, policy, wanted=None):
        return [t.name for t in bt.pick(policy, wanted)]

    def test_nothing_is_on_with_no_policy(self):
        self.assertEqual(bt.enabled(tp.Policy()), [])
        self.assertEqual(self.names(tp.Policy(), True), [])

    def test_groups_come_from_what_the_policy_allows(self):
        self.assertEqual(set(t.group for t in bt.enabled(tp.Policy(web=True))), {"web"})
        self.assertEqual(set(t.group for t in bt.enabled(self.read_only)), {"files", "git"})
        self.assertEqual(set(t.group for t in bt.enabled(self.both)), {"files", "git", "write"})
        self.assertEqual(set(t.group for t in bt.enabled(tp.Policy(self.both.folders, web=True))), {"web", "files", "git", "write"})

    def test_a_request_names_tools_or_groups(self):
        policy = tp.Policy(self.both.folders, web=True)
        self.assertEqual(self.names(policy, ["web"]), ["web_fetch"])
        self.assertEqual(self.names(policy, ["git_log", "files"]), ["git_log", "files_list", "files_read", "files_search"])
        self.assertEqual(self.names(policy, "git"), ["git_status", "git_log", "git_diff", "git_show", "git_branches"])
        self.assertEqual(self.names(self.read_only, ["write"]), [], "a group that is not on gives nothing, not an error")

    def test_an_unknown_name_is_an_error_that_lists_the_real_ones(self):
        with self.assertRaises(bt.ToolError) as caught:
            bt.pick(self.read_only, ["git_logg"])
        self.assertIn("git_logg", str(caught.exception))
        self.assertIn("git_log", str(caught.exception))
        for bad in (5, [1], {"a": 1}):
            with self.assertRaises(bt.ToolError):
                bt.pick(self.read_only, bad)

    def test_a_disabled_tool_is_off_even_when_asked_for_by_name(self):
        policy = tp.Policy(self.read_only.folders, disabled=["files_read"])
        self.assertNotIn("files_read", self.names(policy, True))
        self.assertNotIn("files_read", self.names(policy, ["files_read", "git_log"]))
        self.refused("files_read", {"path": "README.md"}, "no tool", policy)

    def test_a_tool_that_is_not_on_is_not_run(self):
        self.refused("files_write", {"path": "x", "content": "y"}, "no tool", self.read_only)
        self.refused("web_fetch", {"url": "http://example.com"}, "no tool", self.read_only)
        self.refused("nonesuch", {}, "no tool")

    def test_every_tool_has_a_short_definition(self):
        for tool in bt.TOOLS:
            size = len(tool.description) + len(str(tool.schema))
            self.assertLess(size, 500, f"{tool.name}: the model reads every definition in every conversation")
            self.assertEqual(tool.schema["type"], "object")
        self.assertEqual({t.group for t in bt.TOOLS if not t.read_only}, {"write"})
        self.assertTrue(all(t.read_only for t in bt.TOOLS if t.group != "write"))

    def test_the_listing_is_in_the_openai_shape_and_says_what_is_on(self):
        listed = bt.listing(self.read_only)
        self.assertEqual({"type", "function", "colibri"}, set(listed[0]))
        on = {i["function"]["name"] for i in listed if i["colibri"]["enabled"]}
        self.assertEqual(on, {t.name for t in bt.enabled(self.read_only)})
        self.assertEqual(len(listed), len(bt.TOOLS))


class FilesTest(Folder):
    def test_list(self):
        text = self.ok("files_list", {})
        self.assertIn("src/", text)
        self.assertIn("README.md (", text)
        self.assertIn(".env", text)
        self.assertIn("not readable", text.split(".env")[1].split("\n")[0], "a secret-looking file is listed and marked")
        self.assertLess(text.index("src/"), text.index("README.md"), "folders first")
        deeper = self.ok("files_list", {"path": "src", "depth": 2})
        self.assertIn("main.c", deeper)
        self.assertIn("main.c", self.ok("files_list", {"depth": 3}))

    def test_list_stops_at_a_limit_and_says_so(self):
        with patch.object(bt, "LIST_LIMIT", 3):
            self.assertIn("stops at 3 entries", self.ok("files_list", {}))

    def test_read_whole_and_in_pieces(self):
        text = self.ok("files_read", {"path": "src/main.c"})
        self.assertIn("lines 1-3 of 3", text)
        self.assertIn("return 0;", text)
        piece = self.ok("files_read", {"path": "README.md", "start_line": 2, "max_lines": 1})
        self.assertIn("lines 2-2 of 3", piece)
        self.assertIn("hello world", piece)
        self.assertNotIn("# Project", piece)
        self.assertIn("start_line=3", piece, "it says how to go on")
        self.refused("files_read", {"path": "README.md", "start_line": 9}, "past the end")

    def test_read_a_full_path_inside(self):
        self.assertIn("hello world", self.ok("files_read", {"path": os.path.join(self.proj, "README.md")}))

    def test_read_outside_is_refused_however_it_is_asked(self):
        for path in (os.path.join(self.outside, "secret.txt"), "../outside/secret.txt", "..\\outside\\secret.txt", "src/../../outside/secret.txt",
                     "C:/Windows/win.ini", "/etc/passwd", "//server/share/f"):
            with self.subTest(path=path):
                text = self.refused("files_read", {"path": path}, "")
                self.assertNotIn("outside the folders\n", text, "the refusal does not hand over the file")

    def test_secrets_are_not_read_unless_the_policy_says(self):
        for name in (".env", "server.pem"):
            text = self.refused("files_read", {"path": name}, "secret")
            self.assertNotIn("abc123", text)
        allowed = tp.Policy(self.read_only.folders, allow_secrets=True)
        self.assertIn("abc123", self.ok("files_read", {"path": ".env"}, allowed))

    def test_binary_and_big_files_are_not_dumped(self):
        self.refused("files_read", {"path": "blob.bin"}, "not a text file")
        with patch.object(bt, "MAX_FILE_BYTES", 10):
            self.refused("files_read", {"path": "README.md"}, "larger than")
        self.refused("files_read", {"path": "src"}, "not a file")
        self.refused("files_read", {"path": "missing.txt"}, "not a file")

    def test_a_long_result_is_cut_and_the_cut_is_said(self):
        self.write("proj/long.txt", "\n".join("line %d %s" % (i, "x" * 60) for i in range(2000)))
        text = self.ok("files_read", {"path": "long.txt", "max_lines": 2000})
        self.assertIn("cut at", text)
        self.assertLess(len(text), bt.READ_CHARS + 600)

    def test_missing_and_wrong_arguments_say_what_is_needed(self):
        self.refused("files_read", {}, "path")
        self.refused("files_read", {"path": 5}, "path")
        self.refused("files_read", {"path": "README.md", "start_line": "abc"}, "start_line")
        self.refused("files_read", {"path": "README.md", "start_line": True}, "start_line")
        self.assertIn("lines 1-3", self.ok("files_read", {"path": "src/main.c", "start_line": "1", "max_lines": "5"}), "numbers sent as text are taken")

    def test_search(self):
        text = self.ok("files_search", {"text": "hello"})
        self.assertIn("README.md:2: hello world", text)
        self.assertIn("README.md:3: HELLO again", text, "the case does not matter")
        self.assertNotIn("dep.js", text.replace("hello from a dependency", ""), "node_modules is not searched")
        self.assertNotIn("abc123", text)
        self.assertIn("secret-looking files skipped", text)
        self.assertNotIn("blob.bin", text)
        only = self.ok("files_search", {"text": "return", "glob": "*.c"})
        self.assertIn("main.c:2", only)
        self.assertIn("1 match", only)

    def test_search_is_a_plain_text_search_not_a_pattern(self):
        self.write("proj/p.txt", "a.*b\nxxx\n")
        self.assertIn("p.txt:1", self.ok("files_search", {"text": "a.*b"}))
        self.assertIn("0 matches", self.ok("files_search", {"text": "(a+)+$"}), "no pattern is compiled, so none can hang")

    def test_search_stops_at_a_limit_and_says_so(self):
        self.write("proj/many.txt", "hit\n" * 100)
        text = self.ok("files_search", {"text": "hit", "max_results": 5})
        self.assertIn("stopped at 5 matches", text)
        with patch.object(bt, "SEARCH_FILES", 1):
            self.assertIn("stopped after 1 files", self.ok("files_search", {"text": "zzz-nothing"}))

    def test_files_can_be_found_by_name_without_any_text(self):
        self.write("proj/src/deep/tools_loop.py", "x = 1\n")
        self.write("proj/docs/notes.md", "tools_loop.py is mentioned in here\n")
        by_name = self.ok("files_search", {"glob": "tools_loop.py"})
        self.assertIn("1 match", by_name)
        self.assertIn("files named like 'tools_loop.py'", by_name)
        self.assertIn(os.path.join("src", "deep", "tools_loop.py"), by_name)
        self.assertNotIn("notes.md", by_name, "the text of other files is not what was asked for")
        self.assertIn("main.c", self.ok("files_search", {"glob": "*.c"}))
        self.assertIn("1 match", self.ok("files_search", {"glob": "**/main.c"}), "a pattern with a folder in it matches the path from the folder searched")
        self.assertIn("src/deep/tools_loop.py".replace("/", os.sep), self.ok("files_search", {"glob": "src/deep/*.py"}))
        both = self.ok("files_search", {"text": "tools_loop", "glob": "*.md"})
        self.assertIn("notes.md:1", both)
        self.assertNotIn("tools_loop.py:", both)

    def test_a_search_with_nothing_to_look_for_says_what_to_give(self):
        for args in ({}, {"path": "src"}, {"text": ""}):
            with self.subTest(args=args):
                self.refused("files_search", args, "glob")

    def test_a_name_search_skips_secrets_and_stays_in_the_folder(self):
        self.assertIn("0 matches", self.ok("files_search", {"glob": ".env*"}))
        self.assertIn("secret-looking files skipped", self.ok("files_search", {"glob": ".env*"}))
        self.refused("files_search", {"glob": "*", "path": self.outside}, "outside")
        with patch.object(bt, "SEARCH_FILES", 2):
            self.assertIn("stopped after 2 files", self.ok("files_search", {"glob": "*"}))
        self.write("proj/many1.dat", "x")
        self.write("proj/many2.dat", "x")
        self.write("proj/many3.dat", "x")
        self.assertIn("stopped at 2 files", self.ok("files_search", {"glob": "*.dat", "max_results": 2}))

    def test_the_box_has_the_face_the_loop_needs_for_tools_found_on_demand(self):
        """mcp_agent.run_turn asks the box for `deferred`: the built-in tools are all listed, so it is empty and no search tools are offered."""
        box = bt.BuiltinBox(tp.Policy([{"path": self.root}]))
        self.assertEqual(box.deferred, {})
        self.assertEqual([s["function"]["name"] for s in box.specs()], [t.name for t in bt.enabled(tp.Policy([{"path": self.root}]))])
        messages = [{"role": "user", "content": "go"}]
        steps = iter([("", [{"id": "c1", "name": "files_list", "arguments": json.dumps({"path": self.root})}]), ("done", [])])
        outcome = bt.mcp_agent.run_turn(lambda m, t: next(steps), messages, box)
        self.assertEqual(outcome, "done")
        self.assertEqual(messages[2]["role"], "tool")

    def test_search_outside_is_refused(self):
        self.refused("files_search", {"text": "x", "path": self.outside}, "outside")
        self.refused("files_search", {"text": ""}, "text")


class LinkTest(Folder):
    def test_a_link_to_outside_is_neither_read_listed_into_nor_searched(self):
        from test_tools_policy import link
        if not link(self.outside, os.path.join(self.proj, "escape")):
            self.skipTest("this machine cannot make links")
        self.refused("files_read", {"path": "escape/secret.txt"}, "outside")
        self.assertIn("0 matches", self.ok("files_search", {"text": "outside the folders"}))
        listing = self.ok("files_list", {"depth": 3})
        self.assertNotIn("secret.txt", listing)


class WriteTest(Folder):
    def test_write_a_new_file_and_replace_it(self):
        text = self.ok("files_write", {"path": os.path.join(self.scratch, "new", "a.txt"), "content": "one\ntwo\n"}, self.both)
        self.assertIn("a new file", text)
        self.assertEqual(Path(self.scratch, "new", "a.txt").read_bytes(), b"one\ntwo\n", "the text is written as given, no newline changes")
        again = self.ok("files_write", {"path": os.path.join(self.scratch, "new", "a.txt"), "content": "three"}, self.both)
        self.assertIn("replacing a file of 8 characters", again)
        self.assertEqual(Path(self.scratch, "new", "a.txt").read_text(), "three")

    def test_write_is_refused_outside_the_writable_folders(self):
        for path in (os.path.join(self.proj, "x.txt"), os.path.join(self.outside, "x.txt"), "../outside/x.txt"):
            with self.subTest(path=path):
                self.refused("files_write", {"path": path, "content": "x"}, "", self.both)
        self.assertFalse(os.path.exists(os.path.join(self.proj, "x.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.outside, "x.txt")))

    def test_write_never_goes_inside_a_dot_git_folder(self):
        os.makedirs(os.path.join(self.scratch, ".git", "hooks"))
        for rel in (".git/hooks/pre-commit", ".git/config", "sub/.GIT/x"):
            with self.subTest(rel=rel):
                self.refused("files_write", {"path": os.path.join(self.scratch, *rel.split("/")), "content": "#!/bin/sh\nevil"}, ".git", self.both)
                self.refused("files_edit", {"path": os.path.join(self.scratch, *rel.split("/")), "old": "a", "new": "b"}, "", self.both)
        self.assertFalse(os.path.exists(os.path.join(self.scratch, ".git", "hooks", "pre-commit")))

    def test_write_validates_its_arguments(self):
        self.refused("files_write", {"path": os.path.join(self.scratch, "a"), "content": 5}, "content", self.both)
        self.refused("files_write", {"path": os.path.join(self.scratch, "a")}, "content", self.both)
        with patch.object(bt, "MAX_WRITE_CHARS", 5):
            self.refused("files_write", {"path": os.path.join(self.scratch, "a"), "content": "123456"}, "longer", self.both)
        self.refused("files_write", {"path": self.scratch, "content": "x"}, "folder", self.both)

    def test_edit_replaces_exactly_one_place(self):
        path = self.write("scratch/code.py", "a = 1\nb = 2\nc = 1\n")
        self.refused("files_edit", {"path": path, "old": "= 1", "new": "= 9"}, "2 times", self.both)
        self.assertEqual(Path(path).read_text(), "a = 1\nb = 2\nc = 1\n", "nothing changed")
        self.refused("files_edit", {"path": path, "old": "zzz", "new": "y"}, "not in", self.both)
        self.refused("files_edit", {"path": path, "old": "", "new": "y"}, "old", self.both)
        self.ok("files_edit", {"path": path, "old": "b = 2", "new": "b = 3"}, self.both)
        self.assertEqual(Path(path).read_text(), "a = 1\nb = 3\nc = 1\n")
        self.assertIn("2 places", self.ok("files_edit", {"path": path, "old": "= 1", "new": "= 7", "replace_all": True}, self.both))
        self.assertEqual(Path(path).read_text(), "a = 7\nb = 3\nc = 7\n")

    def test_edit_keeps_the_files_own_line_endings(self):
        path = self.write("scratch/crlf.txt", "one\r\ntwo\r\nthree\r\n")
        self.ok("files_edit", {"path": path, "old": "one\ntwo", "new": "uno\ndos"}, self.both)
        self.assertEqual(Path(path).read_bytes(), b"uno\r\ndos\r\nthree\r\n")

    def test_edit_will_not_touch_a_secret_looking_file_or_a_binary(self):
        env = self.write("scratch/.env", "K=1\n")
        self.refused("files_edit", {"path": env, "old": "K=1", "new": "K=2"}, "secret", self.both)
        Path(self.scratch, "b.bin").write_bytes(b"\0\0abc")
        self.refused("files_edit", {"path": os.path.join(self.scratch, "b.bin"), "old": "abc", "new": "x"}, "not a text file", self.both)


@unittest.skipUnless(HAVE_GIT, "git is not installed")
class GitTest(Folder):
    def setUp(self):
        super().setUp()
        make_repo(self.proj)
        git(self.proj, "add", "src", "README.md")
        git(self.proj, "commit", "-q", "-m", "first commit")
        self.write("proj/README.md", "# Project\nhello world\nHELLO again\nmore\n")
        git(self.proj, "commit", "-q", "-am", "add a line to the readme")
        git(self.proj, "branch", "topic")

    def test_status_log_diff_show_branches(self):
        self.write("proj/src/main.c", "int main() { return 1; }\n")
        self.write("proj/new.txt", "untracked\n")
        status = self.ok("git_status", {})
        self.assertIn("M src/main.c", status.replace(" M", "M"))
        self.assertIn("?? new.txt", status)
        log = self.ok("git_log", {"max_count": 5})
        self.assertIn("Test Person: add a line to the readme", log)
        self.assertEqual(len(log.splitlines()), 2)
        self.assertIn("first commit", self.ok("git_log", {"max_count": 1, "rev": "HEAD~1"}))
        self.assertIn("first commit", self.ok("git_log", {"path": "src/main.c"}))
        diff = self.ok("git_diff", {})
        self.assertIn("-int main() {", diff)
        self.assertIn("+int main() { return 1; }", diff)
        self.assertIn("+more", self.ok("git_diff", {"target": "HEAD~1..HEAD"}))
        self.assertIn("nothing", self.ok("git_diff", {"staged": True}))
        show = self.ok("git_show", {})
        self.assertIn("add a line to the readme", show)
        self.assertIn("README.md", show)
        self.assertIn("first commit", self.ok("git_show", {"rev": "HEAD~1"}))
        self.assertNotIn("src/main.c", self.ok("git_show", {"path": "README.md"}))
        branches = self.ok("git_branches", {})
        self.assertIn("topic", branches)

    def test_a_revision_or_path_that_looks_like_an_option_is_refused(self):
        for rev in ("--output=evil.txt", "-c", "--upload-pack=x", "-p", "--all", "main;rm -rf", "a b", "HEAD`id`", "$(id)", "x\ny", ""):
            with self.subTest(rev=rev):
                self.refused("git_log", {"rev": rev} if rev else {"rev": "--"}, "rev")
        self.refused("git_diff", {"target": "--output=evil.txt"}, "target")
        self.refused("git_show", {"rev": "--format=%H"}, "rev")
        self.assertFalse(os.path.exists(os.path.join(self.proj, "evil.txt")))
        self.assertFalse(os.path.exists("evil.txt"))

    def test_a_path_inside_the_repository_cannot_leave_it(self):
        for path in ("../outside/secret.txt", "..", "src/../../outside"):
            with self.subTest(path=path):
                self.refused("git_log", {"path": path}, "outside the repository")
                self.refused("git_diff", {"path": path}, "outside the repository")
        for path in ("C:/Windows", "/etc/passwd", "C:\\Windows"):
            with self.subTest(path=path):
                self.refused("git_log", {"path": path}, "")

    def test_outside_the_allowed_folder_is_refused(self):
        make_repo(self.outside)
        self.refused("git_status", {"repo": self.outside}, "outside")
        self.refused("git_log", {"repo": "../outside"}, "outside")

    def test_a_repository_whose_top_is_above_the_allowed_folder_is_refused(self):
        top = os.path.join(self.root, "bigrepo")
        make_repo(top)
        sub = os.path.join(top, "sub")
        os.makedirs(sub)
        Path(sub, "f.txt").write_text("f")
        git(top, "add", "sub")
        git(top, "commit", "-q", "-m", "history of files outside the folder")
        policy = tp.Policy([{"path": sub, "write": False}])
        self.refused("git_log", {}, "top folder", policy)
        self.refused("git_status", {"repo": sub}, "top folder", policy)

    def test_a_folder_that_is_not_a_repository_says_so(self):
        folder = os.path.join(self.scratch, "plain")
        os.makedirs(folder)
        self.refused("git_status", {"repo": folder}, "not a git repository", self.both)

    def test_git_environment_cannot_steer_it(self):
        other = os.path.join(self.root, "other")
        make_repo(other)
        Path(other, "o.txt").write_text("o")
        git(other, "add", "o.txt")
        git(other, "commit", "-q", "-m", "the other repository")
        with patch.dict(os.environ, {"GIT_DIR": os.path.join(other, ".git"), "GIT_WORK_TREE": other, "GIT_EXTERNAL_DIFF": "echo pwned", "GIT_PAGER": "echo pwned"}):
            self.assertIn("add a line to the readme", self.ok("git_log", {"max_count": 1}))
            self.assertNotIn("pwned", self.ok("git_diff", {"target": "HEAD~1"}))

    def test_a_repository_configured_to_run_a_program_does_not_run_it(self):
        marker = os.path.join(self.root, "ran.txt")
        git(self.proj, "config", "core.fsmonitor", f"echo ran > \"{marker}\"")
        git(self.proj, "config", "diff.external", f"echo ran > \"{marker}\"")
        self.ok("git_status", {})
        self.write("proj/src/main.c", "changed\n")
        self.ok("git_diff", {})
        self.ok("git_show", {})
        self.assertFalse(os.path.exists(marker), "core.fsmonitor and diff.external are not run")

    def test_a_big_diff_is_cut_and_the_cut_is_said(self):
        self.write("proj/big.txt", "".join(f"line {i}\n" for i in range(5000)))
        git(self.proj, "add", "big.txt")
        text = self.ok("git_diff", {"staged": True, "max_chars": 1000})
        self.assertIn("cut at 1000", text)
        self.assertLess(len(text), 1400)
        with patch.object(bt, "GIT_BYTES", 500):
            self.assertIn("larger than the tool reads", self.ok("git_diff", {"staged": True}))

    def test_git_that_takes_too_long_is_stopped_with_what_it_started(self):
        bin_dir = os.path.join(self.root, "fakebin")
        os.makedirs(bin_dir)
        if os.name == "nt":
            Path(bin_dir, "git.cmd").write_text("@echo off\r\nping -n 30 127.0.0.1 >nul\r\n")
        else:
            script = Path(bin_dir, "git")
            script.write_text("#!/bin/sh\nsleep 30\n")
            script.chmod(0o755)
        started = time.monotonic()
        with patch.dict(os.environ, {"PATH": bin_dir + os.pathsep + os.environ["PATH"]}), patch.object(bt, "GIT_SECONDS", 0.5):
            self.refused("git_status", {}, "did not finish")
        self.assertLess(time.monotonic() - started, 8, "a child of git that keeps the output open did not hold the call")


@unittest.skipUnless(HAVE_GIT, "git is not installed")
class GitWriteTest(Folder):
    def setUp(self):
        super().setUp()
        self.repo = os.path.join(self.scratch, "repo")
        make_repo(self.repo)
        Path(self.repo, "a.txt").write_text("a\n")
        git(self.repo, "add", "a.txt")
        git(self.repo, "commit", "-q", "-m", "start")

    def test_add_and_commit(self):
        Path(self.repo, "b.txt").write_text("b\n")
        Path(self.repo, "c.txt").write_text("c\n")
        added = self.ok("git_add", {"repo": self.repo, "paths": ["b.txt"]}, self.both)
        self.assertIn("A  b.txt", added)
        self.assertIn("?? c.txt", added)
        done = self.ok("git_commit", {"repo": self.repo, "message": "add b"}, self.both)
        self.assertIn("add b", done)
        self.assertEqual(git(self.repo, "log", "-1", "--format=%s"), "add b")
        self.assertEqual(git(self.repo, "show", "--stat", "--format=", "HEAD").split()[0], "b.txt", "only what was staged is in it")

    def test_commit_with_nothing_staged_says_so_and_makes_no_commit(self):
        self.refused("git_commit", {"repo": self.repo, "message": "nothing"}, "git commit failed", self.both)
        self.assertEqual(git(self.repo, "log", "--oneline").count("\n"), 0)

    def test_the_repository_must_be_in_a_folder_that_may_be_written(self):
        Path(self.proj, "x.txt").write_text("x")
        make_repo(self.proj)
        self.refused("git_add", {"repo": self.proj, "paths": ["x.txt"]}, "", self.both)
        self.refused("git_commit", {"repo": self.proj, "message": "m"}, "", self.both)
        self.assertIn("?? x.txt", git(self.proj, "status", "--short"))

    def test_paths_to_add_stay_in_the_repository(self):
        self.refused("git_add", {"repo": self.repo, "paths": ["../outside/secret.txt"]}, "", self.both)
        self.refused("git_add", {"repo": self.repo, "paths": []}, "paths", self.both)
        self.refused("git_add", {"repo": self.repo, "paths": "../x"}, "", self.both)
        self.refused("git_add", {"repo": self.repo}, "paths", self.both)

    def test_a_path_that_starts_with_a_dash_is_a_path_not_an_option(self):
        Path(self.repo, "-A").write_text("a file called -A\n")
        Path(self.repo, "other.txt").write_text("other\n")
        self.ok("git_add", {"repo": self.repo, "paths": ["-A"]}, self.both)
        self.assertEqual(sorted(git(self.repo, "diff", "--cached", "--name-only").split()), ["-A"], "only that file, not everything")

    def test_hooks_in_the_repository_are_not_run(self):
        marker = os.path.join(self.root, "hook-ran.txt")
        hook = os.path.join(self.repo, ".git", "hooks", "pre-commit")
        Path(hook).write_text(f"#!/bin/sh\necho ran > '{marker.replace(os.sep, '/')}'\n")
        os.chmod(hook, 0o755)
        msg = os.path.join(self.repo, ".git", "hooks", "commit-msg")
        Path(msg).write_text(f"#!/bin/sh\necho ran > '{marker.replace(os.sep, '/')}'\nexit 1\n")
        os.chmod(msg, 0o755)
        Path(self.repo, "h.txt").write_text("h\n")
        self.ok("git_add", {"repo": self.repo, "paths": ["h.txt"]}, self.both)
        self.ok("git_commit", {"repo": self.repo, "message": "hooked"}, self.both)
        self.assertFalse(os.path.exists(marker))
        self.assertEqual(git(self.repo, "log", "-1", "--format=%s"), "hooked")

    def test_a_commit_message_that_looks_like_an_option_is_only_a_message(self):
        Path(self.repo, "m.txt").write_text("m\n")
        self.ok("git_add", {"repo": self.repo, "paths": ["m.txt"]}, self.both)
        self.ok("git_commit", {"repo": self.repo, "message": "--amend --no-edit"}, self.both)
        self.assertEqual(git(self.repo, "log", "-1", "--format=%s"), "--amend --no-edit")
        self.assertEqual(git(self.repo, "log", "--oneline").count("\n"), 1, "a new commit, not an amended one")
        self.refused("git_commit", {"repo": self.repo, "message": ""}, "message", self.both)


class BoxTest(Folder):
    def test_it_has_the_face_run_turn_needs(self):
        box = bt.BuiltinBox(self.both)
        self.assertEqual({s["function"]["name"] for s in box.specs()}, {t.name for t in bt.enabled(self.both)})
        info = box.tools["files_read"]
        self.assertTrue(info.read_only)
        self.assertFalse(box.tools["files_write"].read_only)
        self.assertEqual(box.always, set())
        self.assertEqual(box.instructions(), {})

    def test_call_is_the_result_or_an_error_with_its_reason(self):
        box = bt.BuiltinBox(self.read_only)
        self.assertIn("hello world", box.call("files_read", {"path": "README.md"}))
        error = box.call("files_read", {"path": "nope.txt"})
        self.assertTrue(error.startswith("Error: "), error)
        self.assertTrue(box.call("files_read", "not an object").startswith("Error: "))

    def test_the_description_of_a_write_says_what_will_happen(self):
        box = bt.BuiltinBox(self.both)
        text = box.describe("files_write", {"path": os.path.join(self.scratch, "n.txt"), "content": "12345"})
        self.assertIn("5 characters", text)
        self.assertIn("a new file", text)
        Path(self.scratch, "n.txt").write_text("old")
        self.assertIn("replacing", box.describe("files_write", {"path": os.path.join(self.scratch, "n.txt"), "content": "x"}))
        edit = box.describe("files_edit", {"path": os.path.join(self.scratch, "n.txt"), "old": "old", "new": "new"})
        self.assertIn("'old'", edit)
        self.assertIn("'new'", edit)
        self.assertIn("files_write", box.describe("files_write", {"path": "/elsewhere/x", "content": "x"}), "never raises, even for a path that is refused")

    def test_the_target_for_the_log_has_no_query_and_no_content(self):
        policy = tp.Policy(self.both.folders, web=True)
        box = bt.BuiltinBox(policy)
        self.assertEqual(box.target("web_fetch", {"url": "https://example.com/a/b?token=SECRET&x=1#frag"}), "https://example.com/a/b")
        self.assertNotIn("SECRET", str(box.target("web_fetch", {"url": "https://example.com/?token=SECRET"})))
        self.assertEqual(box.target("files_write", {"path": "a.txt", "content": "PRIVATE"}), "a.txt")


if __name__ == "__main__":
    unittest.main()
