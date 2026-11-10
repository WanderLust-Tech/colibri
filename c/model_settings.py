"""Settings of single models that outlive a restart: switched off, or pinned to the card or the CPU (docs/model-settings.md).

`model-settings.json` in the setup folder (or COLI_MODEL_SETTINGS, or `coli router --model-settings FILE`):

    {"models": {
      "qwen3-coder-30b-a3b-colibri": {"gpu": "cpu"},
      "gemma-4-31b-colibri":         {"enabled": false}}}

  enabled   false: the router does not load the model and answers a request for it with a 403 that says how to turn it on again.
            One that is loaded is unloaded (when it has finished what it is answering). Anything else, or no entry: enabled.
  gpu       "vulkan" or "cpu" pins where the model runs; "auto" (or no entry) follows the router's --gpu policy (gpu.py).

`coli model enable|disable|gpu` writes the file, and the router reads it again whenever it changes, so nothing needs restarting.
A model is named as it is everywhere else (its id, its key, its folder).
"""
import json
import os
import threading

GPU_CHOICES = ("auto", "vulkan", "cpu")
KEYS = frozenset({"enabled", "gpu"})
FILE_LOCK = threading.Lock()


class SettingsError(ValueError):
    pass


def default_path():
    explicit = os.environ.get("COLI_MODEL_SETTINGS")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    try:
        import setup_flow
        return os.path.join(setup_flow.setup_home(), "model-settings.json")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".colibri-model-settings.json")


def _check(model, record):
    if not isinstance(record, dict):
        raise SettingsError(f"{model!r}: expected an object")
    for key in record:
        if key not in KEYS:
            raise SettingsError(f"{model!r}: unknown setting {key!r} (known: {', '.join(sorted(KEYS))})")
    if "enabled" in record and not isinstance(record["enabled"], bool):
        raise SettingsError(f"{model!r}: `enabled` must be true or false")
    if "gpu" in record and record["gpu"] not in GPU_CHOICES:
        raise SettingsError(f"{model!r}: `gpu` must be one of {', '.join(GPU_CHOICES)}")
    return record


def read(path):
    """The file's models as {id: record}. A missing file is no settings; a broken one is an error that says what."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise SettingsError(f"{path} cannot be read: {error}")
    if not isinstance(data, dict) or not isinstance(data.get("models", {}), dict):
        raise SettingsError(f"{path} must hold an object with a `models` object")
    return {str(model): dict(_check(str(model), record)) for model, record in data.get("models", {}).items()}


def update(path, model, enabled=None, gpu=None):
    """Change one model's settings in the file (atomically). `enabled` True removes the switch (enabled is the default);
    `gpu` "auto" removes the pin. A record left empty is removed. Returns the model's record as now stored."""
    if gpu is not None and gpu not in GPU_CHOICES:
        raise SettingsError(f"gpu must be one of {', '.join(GPU_CHOICES)}")
    with FILE_LOCK:
        models = read(path)
        record = dict(models.get(model, {}))
        if enabled is not None:
            if enabled:
                record.pop("enabled", None)
            else:
                record["enabled"] = False
        if gpu is not None:
            if gpu == "auto":
                record.pop("gpu", None)
            else:
                record["gpu"] = gpu
        if record:
            models[model] = record
        else:
            models.pop(model, None)
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        temp = f"{path}.{os.getpid()}.tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump({"models": models}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp, path)
        return record


class ModelSettings:
    """The file, read again when it changes. A file that cannot be read keeps the last good settings and says so once."""

    def __init__(self, path=None, log=None):
        self.path = path or default_path()
        self._log = log or (lambda message: None)
        self._models, self._stamp, self._warned = {}, None, None
        self._lock = threading.Lock()

    def _stat(self):
        try:
            info = os.stat(self.path)
            return (info.st_mtime_ns, info.st_size)
        except OSError:
            return None

    def reload(self):
        with self._lock:
            stamp = self._stat()
            if stamp == self._stamp:                  # unchanged (or still no file)
                return
            try:
                self._models, self._warned = read(self.path), None
            except SettingsError as error:
                if self._warned != str(error):
                    self._log(f"model settings: {error}; keeping the last good ones")
                    self._warned = str(error)
            self._stamp = stamp

    def record(self, entry):
        self.reload()
        for name in (entry.get("id"), entry.get("key")):
            if name in self._models:
                return self._models[name]
        return {}

    def enabled(self, entry):
        return self.record(entry).get("enabled", True)

    def gpu(self, entry):
        """vulkan, cpu, or None (follow the policy)."""
        choice = self.record(entry).get("gpu")
        return None if choice in (None, "auto") else choice

    def all(self):
        self.reload()
        return {model: dict(record) for model, record in self._models.items()}
