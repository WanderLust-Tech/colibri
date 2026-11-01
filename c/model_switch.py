"""The models on this machine, and the in-place relaunch behind the dashboard's
model switcher.

One `coli serve` process serves one model: the family's engine, chat template and
tool parser are fixed when it starts. Switching therefore means a restart, and this
module keeps that small and safe:

  * the browser names an installed model by an opaque key, never by a path; the
    server looks the key up in a fresh scan;
  * the server drains its engine (which saves the expert-usage history), returns
    from serve(), and the launcher re-executes itself with --model replaced;
  * if the new model does not load, the launcher goes back to the previous one
    (FALLBACK_ENV), so a model that does not fit never leaves the dashboard dead.

Nothing here writes setup.json: a switch lasts for this run, and `coli start`
still starts the model the setup installed.
"""
import hashlib
import os
import subprocess
import sys

MAX_ENTRIES = 256
ROOT_ENV = "COLI_MODELS_ROOT"            # extra folders to scan, os.pathsep-separated
FALLBACK_ENV = "COLI_SWITCH_FALLBACK"    # the model to return to if the new one fails
MARKERS = ("config.json", "model_index.json", "rl_agent_config.json")


def model_key(path):
    """A short, stable id for an installed model: what the browser sends back."""
    return hashlib.sha1(os.path.realpath(path).encode("utf-8")).hexdigest()[:10]


def setup_models_root():
    """The folder `coli setup` put its models in, or None when there is no setup."""
    try:
        import setup_flow
        cfg = setup_flow.load_config() or {}
    except Exception:
        return None
    return cfg.get("models_root")


def candidate_roots(current_dir, environ=None, setup_root=None):
    """Folders whose children may be models: COLI_MODELS_ROOT, the setup's models
    folder, the folder holding the model being served, and ~/colibri-models."""
    environ = os.environ if environ is None else environ
    roots = []

    def add(path):
        if path and os.path.isdir(os.path.expanduser(path)):
            real = os.path.realpath(os.path.expanduser(path))
            if real not in roots:
                roots.append(real)

    for part in (environ.get(ROOT_ENV) or "").split(os.pathsep):
        add(part.strip())
    add(setup_root)
    add(os.path.dirname(os.path.realpath(current_dir)))
    add(os.path.join(os.path.expanduser("~"), "colibri-models"))
    return roots


def _manifest(model_dir):
    """(manifest, complete): what `coli setup`'s downloader recorded. A model
    fetched some other way has no manifest and is listed as unverified."""
    try:
        import setup_download
    except Exception:
        return None, False
    try:
        manifest = setup_download.read_manifest(model_dir)
        if not manifest:
            return None, False
        return manifest, bool(setup_download.is_complete(model_dir))
    except Exception:
        return None, False


def _children(root):
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return
    for name in names[:MAX_ENTRIES]:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            yield path


class SwitchContext:
    """What the running server knows about itself that decides whether, and to
    which models, it can switch. Built by the launcher; the gateway only reads it."""

    def __init__(self, current_dir, *, kv_slots=1, cluster=False, environ=None, roots=None):
        environ = os.environ if environ is None else environ
        self.current_dir = os.path.realpath(current_dir)
        self.kv_slots = kv_slots
        self.cluster = cluster
        self._roots = roots
        self._environ = environ
        self.reason = None
        if environ.get("COLI_ENGINE"):
            self.reason = ("COLI_ENGINE pins one engine for every model, so the model "
                           "cannot be switched from here.")

    @property
    def enabled(self):
        return self.reason is None

    def roots(self):
        if self._roots is not None:
            return self._roots
        return candidate_roots(self.current_dir, self._environ, setup_models_root())


def why_not(family, ctx, engine_ready):
    """None if a model of this family can be started with the current flags, else
    a short reason the dashboard can show."""
    if not engine_ready(family):
        return "the engine for this model is not built"
    if not family.has_gateway_adapter:
        return "the server has no adapter for this model"
    if family.modality == "audio":
        # the engine and the route exist (coli music, POST /v1/audio/music); the dashboard has no player for it yet
        return "the dashboard has no music player yet: coli music, or coli serve (POST /v1/audio/music)"
    if ctx.kv_slots > family.limits.max_kv_slots:
        return f"it supports at most {family.limits.max_kv_slots} KV slot(s); the server runs {ctx.kv_slots}"
    if ctx.cluster and family.id != "glm":
        return "cluster mode is for GLM only"
    return None


def scan_installed(ctx, engine_ready):
    """Every model in the roots (plus the one being served), current first. Each
    entry has `path` for the server's own use; use public() before sending it out."""
    from family_registry import default_model_id, display_for, resolve_model

    seen, entries = set(), []
    dirs = [ctx.current_dir]
    for root in ctx.roots():
        dirs.extend(_children(root))
    for path in dirs:
        real = os.path.realpath(path)
        if real in seen or len(entries) >= MAX_ENTRIES:
            continue
        seen.add(real)
        if not any(os.path.isfile(os.path.join(real, name)) for name in MARKERS):
            continue
        manifest, complete = _manifest(real)
        if manifest and not complete:
            continue                      # a download still in progress, or damaged
        try:
            resolved = resolve_model(real)
        except Exception:
            continue                      # a folder with a config we do not serve
        family = resolved.descriptor
        name, scale = display_for(resolved)
        label = name if not scale or scale in name else f"{name} {scale}"
        size = None
        if manifest:
            size = sum(int(spec.get("size") or 0) for spec in manifest.get("files", []))
        current = real == ctx.current_dir
        reason = None if current else why_not(family, ctx, engine_ready)
        if not ctx.enabled:
            reason = ctx.reason
        entries.append({
            "key": model_key(real), "path": real, "id": default_model_id(resolved),
            "name": label, "family": family.id, "modality": family.modality,
            "size_bytes": size or None, "verified": bool(manifest and complete),
            "current": current, "switchable": ctx.enabled and not current and reason is None,
            "reason": reason,
        })
    entries.sort(key=lambda entry: (not entry["current"], entry["name"].lower()))
    return entries


def public(entry):
    return {key: value for key, value in entry.items() if key != "path"}


def replace_model(args, new_model):
    """The launcher's arguments with --model pointing at `new_model`, and the flags
    that named the old model (--model-id) dropped: the new model announces its own."""
    out, replaced, i = [], False, 0
    while i < len(args):
        arg = args[i]
        if arg == "--model":
            out += ["--model", new_model]
            replaced, i = True, i + 2
        elif arg.startswith("--model="):
            out += ["--model", new_model]
            replaced, i = True, i + 1
        elif arg == "--model-id":
            i += 2
        elif arg.startswith("--model-id="):
            i += 1
        else:
            out.append(arg)
            i += 1
    if not replaced:
        out[1:1] = ["--model", new_model]       # after the sub-command
    if out and out[0] == "web" and "--no-browser" not in out:
        out.append("--no-browser")              # the dashboard is already open
    return out


def relaunch_command(launcher, args, new_model):
    return [sys.executable, launcher] + replace_model(list(args), new_model)


def relaunch_env(new_model, previous_model=None, environ=None):
    env = dict(os.environ if environ is None else environ)
    env.pop("COLI_MODEL_ID", None)
    env.pop(FALLBACK_ENV, None)
    if "COLI_MODEL" in env:
        env["COLI_MODEL"] = new_model
    if previous_model:
        env[FALLBACK_ENV] = previous_model
    return env


def relaunch(cmd, env):
    """Replace this process with `cmd`. Where exec is not available (Windows) wait
    for the child, so the console, and Ctrl+C, stay with the server."""
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name != "nt":
        os.execve(cmd[0], cmd, env)
    sys.exit(subprocess.call(cmd, env=env))
