"""The tools the router has built in: fetch a page, read a repository, read and (when allowed) change files (docs/builtin-tools.md).

These run on the router's PC for whoever holds its API key, inside the limits of `tools_policy.py`: nothing is on until a person turns it on
there, files and repositories only inside the folders allowed, writes only inside folders allowed for writing, and the web only the
public internet (`web_fetch.py`).

Groups and tools (a request names tools or groups):

    web    web_fetch
    files  files_list  files_read  files_search
    git    git_status  git_log  git_diff  git_show  git_branches
    write  files_write  files_edit  git_add  git_commit          (each asks for a yes first; see tools_loop.py)

The definitions are short on purpose: the model reads every one of them in each conversation. Results are cut to a size a context can hold,
and the cut is said. A tool never raises for a bad request: it answers with the reason, so the model can correct itself.

`BuiltinBox` has the same face as `mcp_agent.ToolBox`, so `mcp_agent.run_turn` runs a conversation over either. The standard library only.
"""
import fnmatch
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.parse

import mcp_agent
import tools_policy
import web_fetch

MAX_FILE_BYTES = 1 << 20
MAX_WRITE_CHARS = 1 << 20
LIST_LIMIT = 200
SEARCH_FILES = 5000
SEARCH_SECONDS = 15.0
DEFAULT_LINES = 200
READ_CHARS = 20000
GIT_SECONDS = 20.0
GIT_BYTES = 200000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
REV = re.compile(r"^[A-Za-z0-9_@{}^~:.,/+-]{1,200}$")


class ToolError(Exception):
    """A request the tool cannot do, with the reason the model should be told."""


def _clip(text, limit, hint=""):
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... cut at {limit} characters{hint}]"


# ---- arguments
def _text(args, key, required=True, default=None, limit=4000):
    value = args.get(key, default)
    if value is None or value == "":
        if required:
            raise ToolError(f"`{key}` is needed")
        return default
    if not isinstance(value, str):
        raise ToolError(f"`{key}` must be text")
    if len(value) > limit:
        raise ToolError(f"`{key}` is longer than {limit} characters")
    return value


def _int(args, key, default, low, high):
    value = args.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ToolError(f"`{key}` must be a number")
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ToolError(f"`{key}` must be a number")
    return max(low, min(high, value))


def _flag(args, key):
    value = args.get(key, False)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


# ---- files
def _folder(args, policy, key="path"):
    given = args.get(key)
    if given in (None, ""):
        given = policy.default_folder()
        if given is None:
            raise ToolError(f"`{key}` is needed (more than one folder is allowed)")
    return policy.resolve(given)


def _contained(path, root_real):
    """Does `path` really lie under `root_real`? (A junction or a link inside a folder can lead out of it; os.walk does not always say so.)"""
    return tools_policy.inside(os.path.realpath(path), root_real)


def _binary(chunk):
    return b"\0" in chunk


def files_list(args, policy):
    root = _folder(args, policy)
    if not os.path.isdir(root):
        raise ToolError(f"{root} is not a folder")
    depth = _int(args, "depth", 1, 1, 3)
    lines, shown = [], [0]
    root_real = os.path.realpath(root)

    def walk(folder, level):
        try:
            entries = sorted(os.scandir(folder), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except OSError as error:
            lines.append("  " * level + f"[{error.strerror or error}]")
            return
        for entry in entries:
            if shown[0] >= LIST_LIMIT:
                return
            shown[0] += 1
            pad = "  " * level
            if entry.is_symlink() or (entry.is_dir(follow_symlinks=False) and not _contained(entry.path, root_real)):
                lines.append(f"{pad}{entry.name} -> (link)")
            elif entry.is_dir(follow_symlinks=False):
                lines.append(f"{pad}{entry.name}/")
                if level + 1 < depth and entry.name not in SKIP_DIRS:
                    walk(entry.path, level + 1)
            else:
                try:
                    size = entry.stat(follow_symlinks=False).st_size
                except OSError:
                    size = 0
                secret = not policy.allow_secrets and tools_policy.looks_secret(entry.path)
                lines.append(f"{pad}{entry.name} ({size} bytes)" + (" [secret-looking: not readable]" if secret else ""))

    walk(root, 0)
    more = f"\n[the list stops at {LIST_LIMIT} entries]" if shown[0] >= LIST_LIMIT else ""
    return f"{root}\n" + ("\n".join(lines) if lines else "(empty)") + more


def _read_text(real, policy, limit=None):
    limit = limit or MAX_FILE_BYTES
    if not os.path.isfile(real):
        raise ToolError(f"{real} is not a file")
    if not policy.allow_secrets and tools_policy.looks_secret(real):
        raise ToolError(f"{os.path.basename(real)} looks like a secret (a key, a credential or an .env file): the tools do not read it")
    try:
        with open(real, "rb") as handle:
            data = handle.read(limit + 1)
    except OSError as error:
        raise ToolError(f"{real} cannot be read ({error.strerror or error})")
    if len(data) > limit:
        raise ToolError(f"{os.path.basename(real)} is larger than {limit // 1024} KB")
    if _binary(data[:8192]):
        raise ToolError(f"{os.path.basename(real)} is not a text file")
    return data.decode("utf-8", "replace")


def files_read(args, policy):
    real = policy.resolve(_text(args, "path"))
    text = _read_text(real, policy)
    lines = text.splitlines()
    start = _int(args, "start_line", 1, 1, 10**9)
    count = _int(args, "max_lines", DEFAULT_LINES, 1, 2000)
    chunk = lines[start - 1:start - 1 + count]
    if not chunk and lines:
        raise ToolError(f"{os.path.basename(real)} has {len(lines)} lines; start_line {start} is past the end")
    last = start - 1 + len(chunk)
    body = _clip("\n".join(chunk), READ_CHARS, "; ask for a later start_line")
    head = f"{real} (lines {start}-{last} of {len(lines)})" if lines else f"{real} (empty)"
    return head + ("\n" + body if body else "") + (f"\n[more: call again with start_line={last + 1}]" if last < len(lines) else "")


def files_search(args, policy):
    needle = _text(args, "text", required=False, limit=300)
    root = _folder(args, policy)
    if not os.path.isdir(root):
        raise ToolError(f"{root} is not a folder")
    pattern = args.get("glob")
    if pattern is not None and not isinstance(pattern, str):
        raise ToolError("`glob` must be text")
    if not needle and not pattern:
        raise ToolError("give `text` to look for inside files, or `glob` (like *.py) to find files by name, or both")
    limit = _int(args, "max_results", 40, 1, 200)
    lowered = (needle or "").lower()
    root_real = os.path.realpath(root)
    found, scanned, deadline, skipped = [], 0, time.monotonic() + SEARCH_SECONDS, 0
    stopped = ""
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not os.path.islink(os.path.join(folder, d))
                         and _contained(os.path.join(folder, d), root_real))
        for name in sorted(files):
            path = os.path.join(folder, name)
            if pattern and not (fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(os.path.relpath(path, root).replace(os.sep, "/"), pattern)):
                continue
            if os.path.islink(path):
                continue
            if not policy.allow_secrets and tools_policy.looks_secret(path):
                skipped += 1
                continue
            scanned += 1
            if scanned > SEARCH_FILES:
                stopped = f"stopped after {SEARCH_FILES} files"
                break
            if time.monotonic() > deadline:
                stopped = f"stopped after {SEARCH_SECONDS:.0f} s"
                break
            if not needle:                                           # by name only: no need to open anything
                found.append(os.path.relpath(path, root))
                if len(found) >= limit:
                    stopped = f"stopped at {limit} files"
                    break
                continue
            try:
                if os.path.getsize(path) > MAX_FILE_BYTES:
                    continue
                with open(path, "rb") as handle:
                    data = handle.read()
            except OSError:
                continue
            if _binary(data[:8192]):
                continue
            for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if lowered in line.lower():
                    found.append(f"{os.path.relpath(path, root)}:{number}: {line.strip()[:200]}")
                    if len(found) >= limit:
                        break
            if len(found) >= limit:
                stopped = f"stopped at {limit} matches"
                break
        if stopped:
            break
    what = f"for {needle!r}" if needle else f"files named like {pattern!r}"
    head = f"{len(found)} match{'es' if len(found) != 1 else ''} {what} under {root} ({scanned} files looked at"
    head += f"; {skipped} secret-looking files skipped" if skipped else ""
    head += f"; {stopped}" if stopped else ""
    return head + ")" + ("\n" + "\n".join(found) if found else "")


def _no_git_dir(real):
    if any(part.lower() == ".git" for part in re.split(r"[\\/]+", real)):
        raise ToolError("the tools do not write inside a .git folder")


def files_write(args, policy):
    real = policy.resolve(_text(args, "path"), write=True)
    content = args.get("content")
    if not isinstance(content, str):
        raise ToolError("`content` must be text")
    if len(content) > MAX_WRITE_CHARS:
        raise ToolError(f"`content` is longer than {MAX_WRITE_CHARS} characters")
    _no_git_dir(real)
    if os.path.isdir(real):
        raise ToolError(f"{real} is a folder")
    before = None
    if os.path.exists(real):
        try:
            with open(real, "rb") as handle:
                before = len(handle.read(MAX_FILE_BYTES).decode("utf-8", "replace"))
        except OSError:
            before = 0
    try:
        os.makedirs(os.path.dirname(real), exist_ok=True)
        with open(real, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
    except OSError as error:
        raise ToolError(f"{real} cannot be written ({error.strerror or error})")
    return f"Wrote {len(content)} characters to {real} (" + ("a new file" if before is None else f"replacing a file of {before} characters") + ")."


def files_edit(args, policy):
    real = policy.resolve(_text(args, "path"), write=True)
    old, new = args.get("old"), args.get("new")
    if not isinstance(old, str) or not old:
        raise ToolError("`old` must be the text to replace (not empty)")
    if not isinstance(new, str):
        raise ToolError("`new` must be text")
    _no_git_dir(real)
    _read_text(real, policy)                                      # the checks: a text file, not a secret, not too large
    try:
        with open(real, "rb") as handle:
            raw = handle.read().decode("utf-8", "replace")
    except OSError as error:
        raise ToolError(f"{real} cannot be read ({error.strerror or error})")
    crlf = "\r\n" in raw
    if crlf and "\r\n" not in old:                               # the file keeps its line endings; the model wrote plain newlines
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    count = raw.count(old)
    everywhere = _flag(args, "replace_all")
    if count == 0:
        raise ToolError(f"`old` is not in {os.path.basename(real)}; read the file again and copy the text exactly")
    if count > 1 and not everywhere:
        raise ToolError(f"`old` is in {os.path.basename(real)} {count} times; add more around it to make it one place, or set replace_all")
    changed = raw.replace(old, new) if everywhere else raw.replace(old, new, 1)
    if len(changed) > MAX_WRITE_CHARS:
        raise ToolError("the file would be larger than 1 MB")
    try:
        with open(real, "w", encoding="utf-8", newline="") as handle:
            handle.write(changed)
    except OSError as error:
        raise ToolError(f"{real} cannot be written ({error.strerror or error})")
    return f"Edited {real}: {count if everywhere else 1} place{'s' if everywhere and count != 1 else ''} replaced."


# ---- git
def _git_executable():
    found = shutil.which("git")
    if not found:
        raise ToolError("git is not installed on the router's PC")
    return found


def _git_env():
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}      # GIT_DIR, GIT_EXTERNAL_DIFF... must not steer it
    env.update(GIT_TERMINAL_PROMPT="0", GIT_PAGER="cat", GIT_OPTIONAL_LOCKS="0", GIT_ASKPASS="")
    return env


def _kill_tree(process):
    """Stop git and anything it started: a child that outlives it would hold the output open."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, timeout=10)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        process.kill()
    except Exception:
        pass


def _git(repo, args, write=False, seconds=None, limit=None):
    """Run git in `repo`: (exit code, output as text, whether it was cut). No shell, no pager, no hooks, no prompts; the output is read only up
    to `limit` bytes (the rest is not wanted and git is stopped), and git, with anything it started, is stopped after `seconds`."""
    seconds, limit = seconds or GIT_SECONDS, limit or GIT_BYTES
    command = [_git_executable(), "--no-pager", "-c", "core.fsmonitor=false", "-c", "core.quotepath=false", "-c", "core.pager=cat", "-C", repo]
    if write:
        command += ["-c", "core.hooksPath=" + os.path.join(tempfile.gettempdir(), "colibri-no-hooks"), "-c", "commit.gpgsign=false"]
    extra = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(command + list(args), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=_git_env(), **extra)
    got = []

    def pump():
        try:
            got.append(process.stdout.read(limit + 1))
        except (OSError, ValueError):
            got.append(b"")

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    reader.join(seconds)
    if reader.is_alive():
        _kill_tree(process)
        reader.join(2)
        try:
            process.stdout.close()
        except (OSError, ValueError):
            pass
        raise ToolError(f"git did not finish in {seconds:g} s")
    data = got[0] if got else b""
    cut = len(data) > limit
    if cut:
        _kill_tree(process)
    try:
        process.wait(timeout=max(1.0, seconds))
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        raise ToolError(f"git did not finish in {seconds:g} s")
    finally:
        try:
            process.stdout.close()
        except OSError:
            pass
    return (0 if cut else process.returncode), data[:limit].decode("utf-8", "replace"), cut


def _repo(args, policy, write=False):
    """The repository the call is about: inside the allowed folders, and so is the top of the repository (a repository whose root is
    above the folder would show history of files outside it)."""
    given = args.get("repo")
    if given in (None, ""):
        given = policy.default_folder()
        if given is None:
            raise ToolError("`repo` is needed (more than one folder is allowed)")
    real = policy.resolve(given, write=write)
    if not os.path.isdir(real):
        raise ToolError(f"{real} is not a folder")
    code, out, _ = _git(real, ["rev-parse", "--show-toplevel"])
    if code != 0:
        raise ToolError(f"{real} is not a git repository" + (f" ({out.strip()[:200]})" if out.strip() else ""))
    top = os.path.realpath(out.strip())
    try:
        policy.resolve(top, write=write)
    except tools_policy.PolicyError:
        raise ToolError(f"the repository's top folder {top} is outside the allowed folders")
    return top


def _rev(value, key="rev"):
    if not isinstance(value, str) or not REV.match(value) or value.startswith("-"):
        raise ToolError(f"`{key}` must be a branch, tag, commit or range (like main, v1.2, abc123 or main..topic)")
    return value


def _inside_path(top, value):
    """A path given as a path in the repository, checked to stay in it."""
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ToolError("a path must be text")
    clean = value.strip()
    if os.path.isabs(clean) or ":" in clean:
        raise ToolError(f"{clean} must be a path inside the repository, like src/main.c")
    joined = os.path.realpath(os.path.join(top, clean))
    if not tools_policy.inside(joined, top):
        raise ToolError(f"{clean} is outside the repository")
    return clean


def _git_text(code, out, cut, what, max_chars=None):
    if code != 0:
        raise ToolError(f"git {what} failed: {out.strip()[:500] or 'no message'}")
    out = out.strip("\n")
    limit = max_chars or 12000
    text = _clip(out, limit, "; ask for less, or for a path")
    return (text or f"(nothing: git {what} printed nothing)") + ("\n[the output was larger than the tool reads]" if cut else "")


def git_status(args, policy):
    top = _repo(args, policy)
    code, out, cut = _git(top, ["status", "--short", "--branch"])
    return f"{top}\n" + _git_text(code, out, cut, "status")


def git_log(args, policy):
    top = _repo(args, policy)
    count = _int(args, "max_count", 10, 1, 50)
    command = ["log", f"-{count}", "--date=short", "--no-color", "--pretty=format:%h %ad %an: %s"]
    if args.get("rev"):
        command.append(_rev(args["rev"]))
    if args.get("path"):
        command += ["--", _inside_path(top, args["path"])]
    return _git_text(*_git(top, command), "log")


def git_diff(args, policy):
    top = _repo(args, policy)
    command = ["diff", "--no-color", "--no-ext-diff", "--no-textconv"]
    if _flag(args, "staged"):
        command.append("--cached")
    if args.get("target"):
        command.append(_rev(args["target"], "target"))
    if args.get("path"):
        command += ["--", _inside_path(top, args["path"])]
    return _git_text(*_git(top, command), "diff", _int(args, "max_chars", 12000, 500, 40000))


def git_show(args, policy):
    top = _repo(args, policy)
    command = ["show", "--no-color", "--no-ext-diff", "--no-textconv", "--stat", "--patch", _rev(args.get("rev") or "HEAD")]
    if args.get("path"):
        command += ["--", _inside_path(top, args["path"])]
    return _git_text(*_git(top, command), "show", _int(args, "max_chars", 12000, 500, 40000))


def git_branches(args, policy):
    top = _repo(args, policy)
    return _git_text(*_git(top, ["branch", "--all", "--no-color", "-vv"]), "branch")


def git_add(args, policy):
    top = _repo(args, policy, write=True)
    paths = args.get("paths")
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list) or not paths or len(paths) > 100:
        raise ToolError("`paths` must be a list of paths in the repository (up to 100); \".\" is the whole repository")
    clean = ["." if p == "." else _inside_path(top, p) for p in paths]
    code, out, cut = _git(top, ["add", "--", *clean], write=True)
    if code != 0:
        raise ToolError(f"git add failed: {out.strip()[:500]}")
    return f"Added to the index: {', '.join(clean)}.\n" + _git_text(*_git(top, ["status", "--short", "--branch"]), "status")


def git_commit(args, policy):
    top = _repo(args, policy, write=True)
    message = _text(args, "message", limit=2000)
    code, out, cut = _git(top, ["commit", "-m", message], write=True)
    if code != 0:
        raise ToolError(f"git commit failed: {out.strip()[:500]}")
    return _git_text(code, out, cut, "commit")


# ---- the web
def web_fetch_tool(args, policy):
    return web_fetch.get(_text(args, "url", limit=2000), start=_int(args, "start", 0, 0, 10**9))


# ---- the list
def _schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required)}


_STR, _INT = {"type": "string"}, {"type": "integer"}


def _prop(kind, text):
    return dict(kind, description=text)


class Tool:
    def __init__(self, name, group, read_only, description, schema, run, describe=None, target=None):
        self.name, self.group, self.read_only, self.description, self.schema = name, group, read_only, description, schema
        self.run, self._describe, self._target = run, describe, target

    def describe(self, args, policy):
        """What a person is asked to approve: the call in words. Never raises."""
        try:
            return self._describe(args, policy) if self._describe else self.name
        except Exception:
            return f"{self.name} {args}"

    def target(self, args):
        """What the request log says it was about: a path or an address without its query."""
        try:
            return self._target(args) if self._target else None
        except Exception:
            return None


def _url_target(args):
    parts = urllib.parse.urlsplit(str(args.get("url", "")))
    return f"{parts.scheme}://{parts.hostname}{parts.path}"[:200]


def _path_target(key="path"):
    return lambda args: str(args.get(key) or args.get("repo") or "")[:200]


def _describe_write(args, policy):
    real = policy.resolve(args.get("path"), write=True)
    content = args.get("content") if isinstance(args.get("content"), str) else ""
    state = "a new file" if not os.path.exists(real) else "replacing the file that is there"
    return f"write {len(content)} characters to {real} ({state})"


def _describe_edit(args, policy):
    real = policy.resolve(args.get("path"), write=True)
    old, new = str(args.get("old", "")), str(args.get("new", ""))
    return f"edit {real}: replace {_clip(old, 200)!r} with {_clip(new, 200)!r}" + (" (everywhere)" if _flag(args, "replace_all") else "")


def _describe_add(args, policy):
    return f"git add {args.get('paths')} in {_repo(args, policy, write=True)}"


def _describe_commit(args, policy):
    return f"git commit in {_repo(args, policy, write=True)}: {_clip(str(args.get('message', '')), 300)!r}"


REPO = _prop(_STR, "repository folder (default: the allowed folder)")
PATH = _prop(_STR, "path inside the repository")

TOOLS = [
    Tool("web_fetch", "web", True, "Fetch a web page or text file (public internet only) as text.",
         _schema({"url": _STR, "start": _prop(_INT, "character to continue from")}, ["url"]), web_fetch_tool, target=_url_target),
    Tool("files_list", "files", True, "List a folder.", _schema({"path": _STR, "depth": _prop(_INT, "1 to 3")}), files_list, target=_path_target()),
    Tool("files_read", "files", True, "Read a text file.",
         _schema({"path": _STR, "start_line": _INT, "max_lines": _INT}, ["path"]), files_read, target=_path_target()),
    Tool("files_search", "files", True, "Find text inside files (not a regex), and/or find files by name with a glob like *.py or **/main.c.",
         _schema({"text": _STR, "path": _prop(_STR, "folder"), "glob": _prop(_STR, "file name pattern")}), files_search,
         target=_path_target()),
    Tool("git_status", "git", True, "Git status and branch.", _schema({"repo": REPO}), git_status, target=_path_target("repo")),
    Tool("git_log", "git", True, "Recent commits.",
         _schema({"repo": REPO, "max_count": _prop(_INT, "up to 50"), "rev": _STR, "path": PATH}), git_log, target=_path_target("repo")),
    Tool("git_diff", "git", True, "Git diff: working tree, or staged, or against a revision.",
         _schema({"repo": REPO, "target": _prop(_STR, "revision or a..b"), "staged": {"type": "boolean"}, "path": PATH}), git_diff,
         target=_path_target("repo")),
    Tool("git_show", "git", True, "A commit: message, files, patch.",
         _schema({"repo": REPO, "rev": _prop(_STR, "default HEAD"), "path": PATH}), git_show, target=_path_target("repo")),
    Tool("git_branches", "git", True, "Branches.", _schema({"repo": REPO}), git_branches, target=_path_target("repo")),
    Tool("files_write", "write", False, "Write a whole file (creates it or replaces it).",
         _schema({"path": _STR, "content": _STR}, ["path", "content"]), files_write, _describe_write, _path_target()),
    Tool("files_edit", "write", False, "Replace text in a file: `old` must be in it exactly once.",
         _schema({"path": _STR, "old": _STR, "new": _STR, "replace_all": {"type": "boolean"}}, ["path", "old", "new"]), files_edit,
         _describe_edit, _path_target()),
    Tool("git_add", "write", False, "Stage files.",
         _schema({"repo": REPO, "paths": {"type": "array", "items": _STR}}, ["paths"]), git_add, _describe_add, _path_target("repo")),
    Tool("git_commit", "write", False, "Commit what is staged.",
         _schema({"repo": REPO, "message": _STR}, ["message"]), git_commit, _describe_commit, _path_target("repo")),
]
BY_NAME = {tool.name: tool for tool in TOOLS}
GROUPS = ("web", "files", "git", "write")


def enabled(policy):
    """The tools the policy turns on: the web one when `web` is set; files and git when a folder is allowed; the write ones when one may be written."""
    on = set()
    if policy.web:
        on.add("web")
    if policy.readable:
        on.update(("files", "git"))
    if policy.writable:
        on.add("write")
    return [t for t in TOOLS if t.group in on and t.name not in policy.disabled]


def pick(policy, wanted=None):
    """The enabled tools a request asks for: all of them (`wanted` is None or True), or those named (tool names and group names)."""
    available = enabled(policy)
    if wanted is None or wanted is True:
        return available
    if isinstance(wanted, str):
        wanted = [wanted]
    if not isinstance(wanted, list) or not all(isinstance(n, str) for n in wanted):
        raise ToolError("`colibri_tools` must be true or a list of tool and group names")
    chosen, unknown = [], []
    for name in wanted:
        if name not in BY_NAME and name not in GROUPS:
            unknown.append(name)
            continue
        for tool in available:
            if (tool.name == name or tool.group == name) and tool not in chosen:
                chosen.append(tool)
    if unknown:
        raise ToolError("no such tool or group: " + ", ".join(unknown) + " (tools: " + ", ".join(BY_NAME) + "; groups: " + ", ".join(GROUPS) + ")")
    return chosen


def listing(policy):
    """GET /v1/tools: each tool in the OpenAI function shape, with what the router knows about it."""
    on = {t.name for t in enabled(policy)}
    return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema},
             "colibri": {"group": t.group, "read_only": t.read_only, "enabled": t.name in on}} for t in TOOLS]


def run(name, arguments, policy):
    """(text, is_error) for a call of `name`. The tool must be one the policy turns on."""
    tool = BY_NAME.get(name)
    if tool is None or tool.name not in {t.name for t in enabled(policy)}:
        return f"There is no tool `{name}` available here.", True
    if not isinstance(arguments, dict):
        return "The arguments must be a JSON object.", True
    try:
        return tool.run(arguments, policy), False
    except (ToolError, tools_policy.PolicyError, web_fetch.FetchError) as error:
        return str(error), True
    except OSError as error:
        return f"{error.strerror or error}", True
    except subprocess.SubprocessError as error:
        return f"git did not finish ({error})", True


class BuiltinBox:
    """The same face as `mcp_agent.ToolBox`, over the built-in tools, so `mcp_agent.run_turn` runs a conversation with them."""

    def __init__(self, policy, wanted=None):
        self.policy = policy
        self.chosen = pick(policy, wanted)
        self.tools = {t.name: mcp_agent.ToolInfo(t.name, "router", t.name, t.description, t.schema, t.read_only) for t in self.chosen}
        self.always, self.problems, self.clients = set(), {}, {}

    def specs(self):
        return [t.spec() for t in self.tools.values()]

    def instructions(self):
        return {}

    def describe(self, name, arguments):
        return BY_NAME[name].describe(arguments, self.policy)

    def target(self, name, arguments):
        return BY_NAME[name].target(arguments)

    def call(self, name, arguments):
        text, is_error = run(name, arguments, self.policy)
        return ("Error: " if is_error else "") + text

    def close(self):
        pass
