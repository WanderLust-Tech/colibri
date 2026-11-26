"""The list of MCP servers that colibri's clients use (docs/mcp-client.md).

One file, `mcp-servers.json` in the setup folder (`COLI_MCP_SERVERS` names another), in the shape of Claude's `.mcp.json`:

    {"mcpServers": {"git": {"command": "uvx", "args": ["mcp-server-git"]},
                    "github": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${GITHUB_TOKEN}"}}}}

The router keeps it and serves it (GET /v1/mcp/servers); the clients that run tools (`coli chat --mcp`, the VS Code extension) read it
and run the servers themselves, on their own machine. Two rules come from that:

- **A secret is a reference, never a value.** The list is served to whoever holds the API key, and a key holder must not read your
  GitHub token. An env value or header whose name looks secret has to be written `${NAME}`: the client takes the value from its own
  environment, and the file never holds it. `expand` does that, and refuses a name that is not set rather than guess.
- **What runs is what you approved.** A client runs a command because a person trusted that exact definition, not because a file
  says so (anyone who can write the list could otherwise start a program on every client). `digest` is what a client records when
  you approve a server, and a changed definition is a new question.

Nothing here starts a server: that is `mcp_client.py`. The standard library only.
"""
import hashlib
import json
import os
import re
import tempfile
import threading

KEYS = ("command", "args", "env", "cwd", "url", "headers", "disabled", "includeTools", "excludeTools", "timeoutSeconds",
        "connectTimeoutSeconds")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
RESERVED = ("builtin",)                       # the extension's own in-process server
REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
SECRET_NAME = re.compile(r"(token|key|secret|password|passwd|credential|auth|bearer|cookie)", re.I)
FILE_LOCK = threading.Lock()


class RegistryError(ValueError):
    """A list or a server that cannot be used, with the reason a person can act on."""


def path():
    explicit = os.environ.get("COLI_MCP_SERVERS")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    try:
        import setup_flow
        return os.path.join(setup_flow.setup_home(), "mcp-servers.json")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".colibri-mcp-servers.json")


def _text_map(value, what):
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise RegistryError(f"`{what}` must be an object of text values")
    return dict(value)


def _text_list(value, what):
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RegistryError(f"`{what}` must be a list of text")
    return list(value)


def _number(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 86400:
        raise RegistryError(f"`{what}` must be a number of seconds above 0 (and at most a day)")
    return value


def secret_literals(cfg):
    """The env values, headers, URL parts and arguments whose name looks secret and that hold a value, not a `${NAME}` reference:
    [(where, name)]."""
    found = []
    url = cfg.get("url") or ""
    parts = re.match(r"^https?://([^/?#]*)([^#]*)", url)
    if parts:
        if "@" in parts.group(1) and not REFERENCE.search(parts.group(1).split("@")[0]):
            found.append(("url", "user:password@"))
        for pair in parts.group(2).partition("?")[2].split("&"):
            field, _, value = pair.partition("=")
            if value and SECRET_NAME.search(field) and not REFERENCE.fullmatch(value):
                found.append(("url", field))
    args = cfg.get("args") or []
    for i, arg in enumerate(args):
        flag, equals, value = arg.partition("=")
        if flag.startswith("-") and SECRET_NAME.search(flag):
            if not equals:
                value = args[i + 1] if i + 1 < len(args) else ""
            if value and not value.startswith("-") and not REFERENCE.fullmatch(value):
                found.append(("args", flag))
    for where in ("env", "headers"):
        for name, value in (cfg.get(where) or {}).items():
            if SECRET_NAME.search(name) and not REFERENCE.fullmatch(value.strip()) and not (
                    where == "headers" and REFERENCE.search(value)):
                found.append((where, name))
    return found


def check(name, cfg):
    """The server `cfg` as it will be kept (known keys only, types checked), or a RegistryError that says what is wrong."""
    if not isinstance(name, str) or not NAME.match(name):
        raise RegistryError("a server's name is letters, digits, `-` and `_` (40 at most), starting with a letter or digit")
    if name.lower() in RESERVED:
        raise RegistryError(f"`{name}` is reserved for the extension's own tools")
    if not isinstance(cfg, dict):
        raise RegistryError(f"`{name}` must be an object")
    unknown = sorted(set(cfg) - set(KEYS))
    if unknown:
        raise RegistryError(f"`{name}`: unknown setting {unknown[0]!r} (known: {', '.join(KEYS)})")
    out = {}
    if cfg.get("command") and cfg.get("url"):
        raise RegistryError(f"`{name}` has both `command` and `url`: a server is one or the other")
    if not cfg.get("command") and not cfg.get("url"):
        raise RegistryError(f"`{name}` needs a `command` (a program that speaks MCP on its standard input) or a `url`")
    if cfg.get("command") is not None:
        if not isinstance(cfg["command"], str) or not cfg["command"].strip():
            raise RegistryError("`command` must be text")
        out["command"] = cfg["command"]
    if cfg.get("url") is not None:
        if not isinstance(cfg["url"], str) or not re.match(r"^https?://[^\s/]+", cfg["url"]):
            raise RegistryError("`url` must be an http:// or https:// address")
        out["url"] = cfg["url"]
    if "args" in cfg:
        out["args"] = _text_list(cfg["args"], "args")
    if "env" in cfg:
        out["env"] = _text_map(cfg["env"], "env")
    if "headers" in cfg:
        out["headers"] = _text_map(cfg["headers"], "headers")
    if "cwd" in cfg:
        if not isinstance(cfg["cwd"], str):
            raise RegistryError("`cwd` must be text")
        out["cwd"] = cfg["cwd"]
    if "disabled" in cfg:
        if not isinstance(cfg["disabled"], bool):
            raise RegistryError("`disabled` must be true or false")
        out["disabled"] = cfg["disabled"]
    for key in ("includeTools", "excludeTools"):
        if key in cfg:
            out[key] = _text_list(cfg[key], key)
    for key in ("timeoutSeconds", "connectTimeoutSeconds"):
        if key in cfg:
            out[key] = _number(cfg[key], key)
    if "url" in out and ("args" in out or "env" in out or "cwd" in out):
        raise RegistryError(f"`{name}` is a url server: `args`, `env` and `cwd` belong to a command")
    if "command" in out and "headers" in out:
        raise RegistryError(f"`{name}` is a command server: `headers` belong to a url")
    bad = secret_literals(out)
    if bad:
        where, field = bad[0]
        suggestion = re.sub(r"[^A-Za-z0-9]+", "_", field).upper().strip("_") or "SECRET"
        raise RegistryError(f"`{name}`: {where} `{field}` looks like a secret and holds a value. The list is served to everyone with the API key, "
                            f"so write a reference and keep the value in the environment of the machine that runs the server, as ${{{suggestion}}}")
    return out


def parse(text):
    """{name: server} from the text of a list file. An empty file is an empty list."""
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except ValueError as error:
        raise RegistryError(f"not JSON: {error}")
    if not isinstance(data, dict):
        raise RegistryError("the file must hold a JSON object")
    servers = data.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise RegistryError("`mcpServers` must be an object")
    return {name: check(name, cfg) for name, cfg in servers.items()}


def load(file=None):
    """The servers in the file (none when it is not there). A file that is not valid is an error naming what to fix."""
    file = file or path()
    try:
        with open(file, encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        return {}
    except OSError as error:
        raise RegistryError(f"{file} cannot be read: {error}")
    try:
        return parse(text)
    except RegistryError as error:
        raise RegistryError(f"{file}: {error}")


def save(servers, file=None):
    """Write the list, all or nothing (a temporary file, then a rename): a reader never sees half of it."""
    file = file or path()
    checked = {name: check(name, cfg) for name, cfg in servers.items()}
    folder = os.path.dirname(file) or "."
    os.makedirs(folder, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".mcp-servers-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump({"mcpServers": checked}, out, indent=2)
            out.write("\n")
        os.replace(temporary, file)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def put(name, cfg, file=None):
    """Add or replace one server. Returns the server as kept and whether it replaced one."""
    with FILE_LOCK:
        servers = load(file)
        kept = check(name, cfg)
        replaced = name in servers
        servers[name] = kept
        save(servers, file)
    return kept, replaced


def remove(name, file=None):
    """Take one server out. Returns whether it was there."""
    with FILE_LOCK:
        servers = load(file)
        if name not in servers:
            return False
        del servers[name]
        save(servers, file)
    return True


def set_disabled(name, disabled, file=None):
    with FILE_LOCK:
        servers = load(file)
        if name not in servers:
            raise RegistryError(f"no server is called `{name}`")
        servers[name]["disabled"] = bool(disabled)
        save(servers, file)


def expand(cfg, environ=None):
    """The server with every `${NAME}` taken from `environ` (the process's, by default): in `args`, `env` values, `headers` values, `url`
    and `cwd`. A name that is not set is an error that says which, never an empty string."""
    environ = os.environ if environ is None else environ
    missing = []

    def fill(text):
        def one(match):
            if match.group(1) not in environ:
                missing.append(match.group(1))
                return ""
            return environ[match.group(1)]
        return REFERENCE.sub(one, text)

    out = dict(cfg)
    for key in ("url", "cwd"):
        if key in out:
            out[key] = fill(out[key])
    if "args" in out:
        out["args"] = [fill(a) for a in out["args"]]
    for key in ("env", "headers"):
        if key in out:
            out[key] = {k: fill(v) for k, v in out[key].items()}
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise RegistryError(f"{names} {'is' if len(set(missing)) == 1 else 'are'} not set in the environment that runs this server")
    return out


def digest(name, cfg):
    """What a client records when a person approves a server: a hash of everything that decides what runs (the command or the address,
    the arguments, the environment and header *names with their references as written*, the folder) and nothing that does not (the tool
    filters, the timeouts, `disabled`). Change what runs and the hash changes, so the approval no longer holds."""
    core = {k: cfg[k] for k in ("command", "args", "env", "cwd", "url", "headers") if k in cfg}
    blob = json.dumps({"name": name, "server": core}, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def describe(cfg):
    """One line for a person: what would run."""
    if "url" in cfg:
        return cfg["url"]
    parts = [cfg["command"]] + list(cfg.get("args", []))
    return " ".join(p if re.fullmatch(r"[\w./:=@%+,\-]+", p) else '"' + p.replace('"', '\\"') + '"' for p in parts)


# ---- the approvals a client keeps (never in the shared list: a trust record is the client's own)

def trust_path():
    explicit = os.environ.get("COLI_MCP_TRUST")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    try:
        import setup_flow
        return os.path.join(setup_flow.setup_home(), "mcp-trust.json")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".colibri-mcp-trust.json")


def trusted(file=None):
    """{name: digest} of the servers this client has approved."""
    try:
        with open(file or trust_path(), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(data, dict) else {}


def is_trusted(name, cfg, file=None):
    return trusted(file).get(name) == digest(name, cfg)


def trust(name, cfg, file=None):
    """Record that this exact definition was approved."""
    file = file or trust_path()
    with FILE_LOCK:
        records = trusted(file)
        records[name] = digest(name, cfg)
        os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
        with open(file, "w", encoding="utf-8") as out:
            json.dump(records, out, indent=2)
            out.write("\n")


def forget(name, file=None):
    file = file or trust_path()
    with FILE_LOCK:
        records = trusted(file)
        if records.pop(name, None) is not None:
            with open(file, "w", encoding="utf-8") as out:
                json.dump(records, out, indent=2)
                out.write("\n")
