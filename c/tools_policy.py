"""What the router's built-in tools may touch (docs/builtin-tools.md).

One file, `tools-policy.json` in the setup folder (`COLI_TOOLS_POLICY` names another):

    {"folders": [{"path": "C:/work/project", "write": false}, {"path": "C:/work/scratch", "write": true}],
     "web": true, "approvalSeconds": 120}

The router runs these tools for anyone who holds its API key, so the key is not the only limit: **nothing is on until a person says so**
here, on this PC. The file is changed by `coli tools allow|deny|web` or by hand, never through the API, so a key holder cannot widen it.

- a folder is a place the file and git tools may read (and, with `"write": true`, change), and nothing outside it: every path is resolved
  through links and `..` first, so a link inside a folder that points out of it does not lead out;
- `web` turns the fetch tool on (it still refuses addresses that are not on the public internet);
- `allowSecrets` lets the file tools read files that look like secrets (`.env`, keys, credentials), which they refuse by default;
- `disabled` names tools to switch off.

The standard library only.
"""
import json
import os
import re
import tempfile
import threading

KEYS = ("folders", "web", "approvalSeconds", "allowSecrets", "disabled")
FILE_LOCK = threading.Lock()
DEFAULT_APPROVAL_SECONDS = 120.0

# a file whose name says it holds a credential: the file tools do not read it unless `allowSecrets` is set (a best effort, not a guarantee)
SECRET_FILE = re.compile(
    r"^(\.env(\..*)?|.*\.(pem|key|pfx|p12|kdbx|ppk|jks|keystore)|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|\.npmrc|\.pypirc|\.netrc|_netrc"
    r"|\.git-credentials|credentials(\.json)?|.*secrets?\.(json|ya?ml|toml|txt)|\.htpasswd|.*\.tfstate|terraform\.tfvars)$", re.I)
SECRET_DIRS = {".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker"}


class PolicyError(ValueError):
    """A policy file or a request the policy refuses, with the reason a person can act on."""


def path():
    explicit = os.environ.get("COLI_TOOLS_POLICY")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    try:
        import setup_flow
        return os.path.join(setup_flow.setup_home(), "tools-policy.json")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".colibri-tools-policy.json")


def normal(folder):
    """A folder as the file keeps it: absolute, `~` expanded, no `..` or doubled separators."""
    return os.path.normpath(os.path.abspath(os.path.expanduser(folder)))


def _key(p):
    return os.path.normcase(p)


def inside(child, parent):
    """Is `child` the folder `parent` or something under it? (Both already resolved; the case does not matter on Windows.)"""
    try:
        return os.path.commonpath([_key(child), _key(parent)]) == _key(parent)
    except ValueError:                                  # another drive
        return False


def is_root(folder):
    return os.path.dirname(folder) == folder


def looks_secret(real_path):
    """True when the file's name, or a folder above it, says it holds a credential."""
    parts = re.split(r"[\\/]+", real_path)
    return bool(SECRET_FILE.match(parts[-1])) or any(p.lower() in SECRET_DIRS for p in parts[:-1])


class Policy:
    def __init__(self, folders=None, web=False, approval_seconds=DEFAULT_APPROVAL_SECONDS, allow_secrets=False, disabled=()):
        self.folders = [dict(f) for f in (folders or [])]
        self.web = bool(web)
        self.approval_seconds = approval_seconds
        self.allow_secrets = bool(allow_secrets)
        self.disabled = sorted(set(disabled))

    # ---- what the file says
    def to_json(self):
        out = {"folders": [{"path": f["path"], "write": bool(f.get("write"))} for f in self.folders], "web": self.web}
        if self.approval_seconds != DEFAULT_APPROVAL_SECONDS:
            out["approvalSeconds"] = self.approval_seconds
        if self.allow_secrets:
            out["allowSecrets"] = True
        if self.disabled:
            out["disabled"] = list(self.disabled)
        return out

    def summary(self):
        """What the policy lets the tools do, for a person (and for GET /v1/tools)."""
        return {"folders": [{"path": f["path"], "write": bool(f.get("write"))} for f in self.folders], "web": self.web,
                "allowSecrets": self.allow_secrets, "approvalSeconds": self.approval_seconds, "disabled": list(self.disabled)}

    @property
    def readable(self):
        return bool(self.folders)

    @property
    def writable(self):
        return any(f.get("write") for f in self.folders)

    # ---- the check every path goes through
    def resolve(self, raw, write=False):
        """The real path `raw` stands for, when it is inside a folder the policy allows (a folder that may be written, for a write).

        A relative path is taken from the one folder there is; with several, the tool is told to give a full path. The answer has had
        links and `..` resolved, so the check is on where the path really goes."""
        if not isinstance(raw, str) or not raw.strip() or "\0" in raw:
            raise PolicyError("the path must be text")
        raw = raw.strip()
        spelled = raw.replace("/", "\\") if os.name == "nt" else raw          # on Windows `/\server\share` is a network path too
        if spelled.startswith("\\\\") or raw.startswith("//"):
            raise PolicyError("a network or device path is not allowed")
        if not self.folders:
            raise PolicyError("no folder is allowed: on the PC the router runs on, `coli tools allow FOLDER` allows one")
        if os.path.isabs(raw):
            candidate = raw
        elif len(self.folders) == 1:
            candidate = os.path.join(self.folders[0]["path"], raw)
        else:
            raise PolicyError("give a full path (more than one folder is allowed: " + ", ".join(f["path"] for f in self.folders) + ")")
        candidate = os.path.abspath(candidate)
        if ":" in os.path.splitdrive(candidate)[1]:      # an NTFS stream (`file.txt:hidden`)
            raise PolicyError("a path with a colon in it is not allowed")
        real = os.path.realpath(candidate)
        if write and _key(real) == _key(os.path.realpath(path())):
            raise PolicyError("the tools policy file is changed with `coli tools` or the dashboard's folder list, not by a tool")
        for folder in self.folders:
            if write and not folder.get("write"):
                continue
            if inside(real, os.path.realpath(folder["path"])):
                return real
        if write and not self.writable:
            raise PolicyError("no folder may be written: `coli tools allow FOLDER --write` on the router's PC allows one")
        where = ", ".join(f["path"] for f in self.folders if f.get("write") or not write)
        raise PolicyError(f"{raw} is outside the {'writable ' if write else ''}folders allowed ({where})")

    def default_folder(self):
        """The folder a tool uses when it is not told one: the only one there is."""
        return self.folders[0]["path"] if len(self.folders) == 1 else None


# ---- the file
def parse(text):
    try:
        data = json.loads(text)
    except ValueError as error:
        raise PolicyError(f"not valid JSON ({error})")
    if not isinstance(data, dict):
        raise PolicyError("must be an object")
    unknown = [k for k in data if k not in KEYS]
    if unknown:
        raise PolicyError("unknown setting " + ", ".join(f"`{k}`" for k in unknown) + " (the settings are " + ", ".join(KEYS) + ")")
    folders = []
    for item in data.get("folders", []):
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"].strip():
            raise PolicyError("each of `folders` needs a `path`")
        extra = [k for k in item if k not in ("path", "write")]
        if extra:
            raise PolicyError(f"a folder has no setting `{extra[0]}` (it has `path` and `write`)")
        if not isinstance(item.get("write", False), bool):
            raise PolicyError("`write` must be true or false")
        folder = normal(item["path"])
        if is_root(folder):
            raise PolicyError(f"{folder} is the top of a drive: allow a folder, not everything")
        folders.append({"path": folder, "write": bool(item.get("write", False))})
    for key in ("web", "allowSecrets"):
        if not isinstance(data.get(key, False), bool):
            raise PolicyError(f"`{key}` must be true or false")
    seconds = data.get("approvalSeconds", DEFAULT_APPROVAL_SECONDS)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 1 <= seconds <= 3600:
        raise PolicyError("`approvalSeconds` must be a number between 1 and 3600")
    disabled = data.get("disabled", [])
    if not isinstance(disabled, list) or not all(isinstance(n, str) for n in disabled):
        raise PolicyError("`disabled` must be a list of tool names")
    return Policy(folders, data.get("web", False), seconds, data.get("allowSecrets", False), disabled)


def load(file=None):
    """The policy in the file; none (everything off) when there is no file. A file that is not valid is an error naming what to fix."""
    file = file or path()
    try:
        with open(file, encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        return Policy()
    except OSError as error:
        raise PolicyError(f"{file} cannot be read: {error}")
    try:
        return parse(text)
    except PolicyError as error:
        raise PolicyError(f"{file}: {error}")


def save(policy, file=None):
    """Write the policy, all or nothing (a temporary file, then a rename)."""
    file = file or path()
    folder = os.path.dirname(file) or "."
    os.makedirs(folder, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".tools-policy-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(policy.to_json(), out, indent=2)
            out.write("\n")
        os.replace(temporary, file)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def edit(change, file=None):
    """Load, apply `change(policy)`, check it is still a valid policy, save. Returns the new policy."""
    with FILE_LOCK:
        policy = load(file)
        change(policy)
        parse(json.dumps(policy.to_json()))             # what is written is what a reader will accept
        save(policy, file)
        return policy


def allow(folder, write=False, file=None):
    """Allow `folder` (replacing its earlier entry). Returns (policy, the folder as kept)."""
    kept = normal(folder)
    if is_root(kept):
        raise PolicyError(f"{kept} is the top of a drive: allow a folder, not everything")
    if not os.path.isdir(kept):
        raise PolicyError(f"{kept} is not a folder that exists")

    def change(policy):
        policy.folders = [f for f in policy.folders if _key(f["path"]) != _key(kept)] + [{"path": kept, "write": bool(write)}]

    return edit(change, file), kept


def deny(folder, file=None):
    """Take `folder` out. Returns (policy, whether it was there)."""
    kept = normal(folder)
    seen = []

    def change(policy):
        seen.append(any(_key(f["path"]) == _key(kept) for f in policy.folders))
        policy.folders = [f for f in policy.folders if _key(f["path"]) != _key(kept)]

    return edit(change, file), seen[0]


def set_web(on, file=None):
    def change(policy):
        policy.web = bool(on)
    return edit(change, file)
