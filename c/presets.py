"""Named presets: a model with settings baked in, offered under a name of its own (docs/presets.md).

A preset is "photo-1024" for the image model at 1024x768 and 20 steps, or "terse" for the coder model with a system
prompt and a low temperature. Clients ask for it by name in `model`, like any model; the router loads the base model
and fills in what the request left out. A value the request carries always wins.

The file is JSON, `presets.json` in the setup folder (or COLI_PRESETS, or `coli router --presets FILE`):

    {"presets": {
      "terse":      {"model": "qwen3-coder-30b-a3b", "system": "Answer in one sentence.",
                     "defaults": {"temperature": 0.2, "max_tokens": 300}, "description": "short answers"},
      "photo-1024": {"model": "qwen-image-2.1", "defaults": {"size": "1024x768", "steps": 20}}}}

The router reads it again whenever the file changes, so `coli preset add` takes effect without a restart. A preset that
cannot be used (its base model is not installed, a bad name) is reported and left out; the others still work.
"""
import json
import os
import re
import sys
import threading

NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
# What a request is *about*: a preset supplies how it is done, never what is said, shown or sent.
FORBIDDEN_DEFAULTS = frozenset({
    "model", "messages", "prompt", "input", "image", "images", "mask", "file", "src_audio", "reference_audio",
    "tools", "functions"})
KEYS = frozenset({"model", "defaults", "system", "description", "keep_alive"})


class PresetError(ValueError):
    pass


class Preset:
    def __init__(self, name, model, defaults=None, system=None, description="", keep_alive=None):
        self.name, self.model = name, model
        self.defaults = dict(defaults or {})
        self.system = system
        self.description = description
        self.keep_alive = keep_alive          # as written (a duration, a number, -1), or None: the router reads it

    def public(self):
        """What GET /v1/presets shows."""
        out = {"name": self.name, "model": self.model, "defaults": self.defaults}
        if self.system:
            out["system"] = self.system
        if self.description:
            out["description"] = self.description
        if self.keep_alive is not None:
            out["keep_alive"] = self.keep_alive
        return out


def default_path():
    explicit = os.environ.get("COLI_PRESETS")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    try:
        import setup_flow
        return os.path.join(setup_flow.setup_home(), "presets.json")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".colibri-presets.json")


def check(name, spec):
    """A Preset from one entry of the file, or PresetError saying what is wrong with it."""
    if not isinstance(name, str) or not NAME.match(name):
        raise PresetError("the name must be 1 to 64 characters: lower-case letters, digits, '.', '_' or '-', "
                          "starting with a letter or digit")
    if not isinstance(spec, dict):
        raise PresetError("a preset is an object")
    unknown = sorted(set(spec) - KEYS)
    if unknown:
        raise PresetError("unknown key(s): " + ", ".join(unknown) + " (allowed: " + ", ".join(sorted(KEYS)) + ")")
    model = spec.get("model")
    if not isinstance(model, str) or not model.strip():
        raise PresetError("`model` must name an installed model")
    defaults = spec.get("defaults", {})
    if not isinstance(defaults, dict):
        raise PresetError("`defaults` must be an object")
    for key in defaults:
        if key in FORBIDDEN_DEFAULTS:
            raise PresetError(f"`{key}` cannot be a default: a request supplies what it is about "
                              f"({', '.join(sorted(FORBIDDEN_DEFAULTS))})")
    system = spec.get("system")
    if system is not None and not isinstance(system, str):
        raise PresetError("`system` must be text")
    description = spec.get("description", "")
    if not isinstance(description, str):
        raise PresetError("`description` must be text")
    keep = spec.get("keep_alive")
    if keep is not None and (isinstance(keep, bool) or not isinstance(keep, (int, float, str))):
        raise PresetError("`keep_alive` must be a duration such as 10m, a number of seconds, 0 or -1")
    return Preset(name, model.strip(), defaults, system or None, description, keep)


def parse(text):
    """({name: Preset}, [(name, why)]) from the text of a presets file."""
    try:
        data = json.loads(text) if text.strip() else {}
    except ValueError as error:
        raise PresetError(f"not valid JSON ({error})") from None
    entries = data.get("presets", {}) if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        raise PresetError('the file is {"presets": {name: {...}}}')
    found, problems = {}, []
    for name, spec in entries.items():
        try:
            found[name] = check(name, spec)
        except PresetError as error:
            problems.append((name, str(error)))
    return found, problems


def read(path):
    """({name: Preset}, problems) of a file; an absent file is an empty set. Raises PresetError for a file that cannot be read."""
    try:
        with open(path, encoding="utf-8") as handle:
            return parse(handle.read())
    except FileNotFoundError:
        return {}, []
    except OSError as error:
        raise PresetError(f"cannot read {path}: {error}") from None


def write(path, presets):
    """Save {name: Preset} atomically."""
    document = {"presets": {name: {k: v for k, v in (
        ("model", p.model), ("description", p.description or None), ("system", p.system),
        ("keep_alive", p.keep_alive), ("defaults", p.defaults or None)) if v is not None}
        for name, p in sorted(presets.items())}}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


class Presets:
    """The presets of a file, read again when it changes. Never raises from get() or all(): a file that breaks keeps the
    last good set and the problem is in `problems`."""

    def __init__(self, path, log=None):
        self.path = path
        self.log = log or (lambda message: sys.stderr.write("[presets] %s\n" % message))
        self.lock = threading.Lock()
        self._stamp = "unread"
        self._found, self.problems = {}, []
        self.error = None

    def _stat(self):
        try:
            st = os.stat(self.path)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _refresh(self):
        with self.lock:
            stamp = self._stat()
            if stamp == self._stamp:
                return
            self._stamp = stamp
            try:
                found, problems = read(self.path)
            except PresetError as error:
                if str(error) != self.error:
                    self.log(f"{self.path}: {error}; keeping the presets read before")
                self.error = str(error)
                return
            self.error = None
            for name, why in problems:
                if (name, why) not in self.problems:
                    self.log(f"preset `{name}` ignored: {why}")
            self._found, self.problems = found, problems

    def get(self, name):
        if not isinstance(name, str):
            return None
        self._refresh()
        return self._found.get(name.lower())

    def all(self):
        self._refresh()
        return [self._found[name] for name in sorted(self._found)]


# ---- editing the file (`coli preset add|rm`): the raw document is changed, so an entry this version cannot read is kept
def load_raw(path):
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        return {"presets": {}}
    except OSError as error:
        raise PresetError(f"cannot read {path}: {error}") from None
    try:
        data = json.loads(text) if text.strip() else {"presets": {}}
    except ValueError as error:
        raise PresetError(f"{path} is not valid JSON ({error}); fix or remove it first") from None
    if not isinstance(data, dict) or not isinstance(data.setdefault("presets", {}), dict):
        raise PresetError(f'{path} is not {{"presets": {{...}}}}')
    return data


def save_raw(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def add(path, name, spec):
    """Add or replace a preset in the file after checking it. Returns True when it replaced one."""
    name = name.lower()
    check(name, spec)
    data = load_raw(path)
    replaced = name in data["presets"]
    data["presets"][name] = spec
    save_raw(path, data)
    return replaced


def remove(path, name):
    """Remove a preset; False when there was none by that name."""
    data = load_raw(path)
    if name.lower() not in data["presets"]:
        return False
    del data["presets"][name.lower()]
    save_raw(path, data)
    return True
