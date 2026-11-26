"""One door in front of several models.

`coli serve` loads one model per process: the family's engine, chat template and tool
parser are fixed when it starts (openai_server.py keeps them in module globals). The
router keeps that and adds a front door instead. It is a separate process that

  * listens where clients expect the server (port 8000), checks the API key once and
    serves the dashboard;
  * reads the `model` of each request and sends it to a worker, an ordinary
    `coli serve` on a loopback port that holds that one model;
  * starts a worker the first time its model is asked for, and stops it after it has
    been idle for the keep-alive time;
  * before loading a model, compares its size with the free memory and stops idle
    workers (least recently used first) until it fits, never one that is answering;
  * lets only one model compute at a time. colibri's models share the CPU, so two
    renders at once each run several times slower; keeping models loaded is cheap,
    running them together is not (`parallel_models` turns this off).

The design follows Ollama's scheduler (server/sched.go: load on demand, a memory check
with headroom, idle eviction, keep-alive), with a router in front of per-model processes
in place of Ollama's runners.

The dashboard's model switcher keeps working unchanged: /v1/models lists the models that
are loaded, /v1/models/installed lists every model on the machine with its state, and
POST /v1/models/switch loads one (it answers when the model is ready, or says why not).
"""
import collections
import hashlib
import hmac
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

import features
import gpu as gpu_mod
import model_settings as settings_mod
import mcp_registry as mcp_registry_mod
import presets as presets_mod
import voices as voices_mod
import web_headers
import log_tail
import request_log
import client_tags
import library
import jobs
import model_installer

DEFAULT_KEEP_ALIVE = 600.0          # seconds a model stays loaded after its last request
DEFAULT_MAX_LOADED = 3
FIT_HEADROOM = 0.8                  # a model must fit in this share of the free memory (Ollama's rule)
# Memory pressure between requests (Pool.relieve): below the floor, an idle model is stopped, then the next look waits.
PRESSURE_FLOOR_GB = 1.5             # free memory below this (or below PRESSURE_FLOOR_SHARE of the total, if more) is pressure
PRESSURE_FLOOR_SHARE = 0.03
PRESSURE_TARGET = 1.5               # stop idle models until about this many times the floor is free: a little more than the shortfall
PRESSURE_MIN_IDLE = 20.0            # seconds a model must have been idle to be stopped for pressure (unless free memory is under half the floor)
PRESSURE_COOLDOWN = 15.0            # seconds to wait after a stop for the memory to show up as free before stopping another
PEAK_SAMPLE_S = 5.0                 # how often a busy worker's memory is read, to learn the most it grows to
CARD_MIN_IDLE = 60.0                # seconds a model must have been idle before it is stopped to give the card to another
DEMOTE_TIMEOUT_S = 300.0            # how long a worker has to give the card back (it reads its dense weights back into RAM first)
CARD_SETTLE_S = 10.0                # how long to wait for the card to show its memory free again after a stop
# A model that started on the CPU only because the card was full is stopped once the card has room, to load there next time (Pool.promote).
PROMOTE_MIN_IDLE = 20.0             # seconds it must have been idle (a model in a burst of requests is left alone)
PROMOTE_MIN_AGE = 60.0              # seconds it must have been running, so a model is not stopped soon after it loaded
PROMOTE_MARGIN = 1.25               # the card must have this much of the model's size free, beyond the reserve
PROMOTE_MODALITIES = ("embedding", "rerank")   # models that load in a few seconds: a reload costs less than the card saves
# A chat model costs more to put back on the card (it is stopped, loads again on the next request and loses its conversation cache),
# so it waits longer, and only when its reload is not slow. It is the one that gains most from the card (Qwen3-Coder: 3 to 18 tokens a second).
PROMOTE_CHAT_MODALITIES = ("text",)
PROMOTE_CHAT_MIN_IDLE = 300.0       # seconds a chat model must have been idle (a conversation that pauses to think is left alone)
PROMOTE_CHAT_MAX_RELOAD_S = 120.0   # its measured (or guessed) load time must not be longer than this
PROMOTE_PROMISE_TTL = 900.0         # seconds a card is held for a model stopped to move onto it (it loads on its next request)
RELOAD_BYTES_PER_S = 150e6          # what a model never timed is assumed to load at (a slow disk): only used to compare
STOP_GRACE = 25.0                   # seconds a worker gets to save its state before it is killed
CARD_FALLBACK_S = 1800.0            # after a start on the graphics card failed, the model starts on the CPU for this long
LOOPBACK = {"127.0.0.1", "localhost", "::1", ""}

# What a worker is asked to compute, and the biggest body each may carry. The workers enforce
# their own, tighter limits; these only bound what the router reads into memory.
PRESET_FILE_LOCK = threading.Lock()      # one change to the presets file at a time (it is read, changed and written back)
COMPUTE_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/systemone",
                 "/v1/images/generations", "/v1/images/edits", "/v1/audio/music", "/v1/audio/transcriptions",
                 "/v1/audio/speech", "/v1/audio/alignments", "/v1/embeddings", "/v1/rerank", "/v2/rerank")

# Who goes first in the queue. A render (a picture, a song) takes minutes to an hour and a chat takes seconds, so a chat
# that arrives while renders are waiting goes ahead of them. Nothing is interrupted: a request that is already
# running always finishes, and requests of the same priority keep their order. A client can say otherwise with the
# header `X-Colibri-Priority: low | normal | high`.
PRIORITY_HEADER = "X-Colibri-Priority"
PRIORITIES = {"low": -1, "normal": 0, "high": 1}
LOW_PRIORITY_PATHS = ("/v1/images/generations", "/v1/images/edits", "/v1/audio/music")


def request_priority(path, headers):
    """The queue priority of a request: what its header says, else low for a render and normal for the rest.
    A header that says something else is ignored, not an error: priority is a hint, never a reason to refuse."""
    said = (headers.get(PRIORITY_HEADER) or "").strip().lower() if headers is not None else ""
    if said in PRIORITIES:
        return PRIORITIES[said]
    return PRIORITIES["low"] if path in LOW_PRIORITY_PATHS else PRIORITIES["normal"]


HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "content-length", "host", "expect"}
DROP_REQUEST = HOP_BY_HOP | {"authorization", "x-api-key", "origin", "referer", "accept-encoding"}
DROP_RESPONSE = HOP_BY_HOP | {"server", "date", "vary"}


def log(message):
    sys.stderr.write("[router] %s\n" % message)
    sys.stderr.flush()
    log_tail.TAIL.add("router", message)


class ApiError(Exception):
    """An error the router answers with, in the shape the workers use."""

    def __init__(self, status, message, code=None, param=None, error_type="invalid_request_error", headers=None):
        super().__init__(message)
        self.status, self.message, self.code, self.param = status, message, code, param
        self.error_type, self.headers = error_type, headers or {}

    def body(self):
        return {"error": {"message": self.message, "type": self.error_type, "param": self.param, "code": self.code}}


def disabled_error(entry):
    return ApiError(403, f"The model `{entry['id']}` is turned off. Turn it on with `coli model enable {entry['id']}`.",
                    "model_disabled", "model")


def parse_duration(text):
    """'30s', '5m', '2h', a bare number of seconds, '0' (unload as soon as idle) or '-1' / 'forever'
    (never). Returns seconds, or None for never."""
    text = str(text).strip().lower()
    if text in ("-1", "forever", "never", "inf"):
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smh]?)", text)
    if not match:
        raise ValueError(f"not a duration: {text!r} (try 30s, 10m, 2h, 0 or -1)")
    return float(match.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]


# ------------------------------------------------------------------------------ memory
def available_memory():
    """Bytes the system could give a new process, or None when unknown."""
    try:
        import setup_hw
        return setup_hw.detect_memory().get("available")
    except Exception:
        return None


def total_memory():
    try:
        import setup_hw
        return setup_hw.detect_memory().get("total")
    except Exception:
        return None


def _job_pids(job):
    """The process ids inside a Windows job object (the worker, its engine, the engine's re-exec)."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    count = 256

    class IdList(ctypes.Structure):
        _fields_ = [("assigned", wintypes.DWORD), ("listed", wintypes.DWORD),
                    ("ids", ctypes.c_size_t * count)]

    info = IdList()
    k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                              ctypes.c_void_p]
    if not k32.QueryInformationJobObject(job, 3, ctypes.byref(info), ctypes.sizeof(info), None):
        return []
    return [info.ids[i] for i in range(min(info.listed, count))]


def _windows_working_set(pid):
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    handle = k32.OpenProcess(0x1000, False, int(pid))           # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return 0
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if k32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return 0
    finally:
        k32.CloseHandle(handle)


def resident_bytes(pid, job=None):
    """Memory held by a worker and everything it started, in bytes; 0 when it cannot be read."""
    try:
        if sys.platform == "win32":
            pids = _job_pids(job) if job else [pid]
            return sum(_windows_working_set(p) for p in pids)
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=10).stdout
        rows = [tuple(int(x) for x in line.split()) for line in out.splitlines() if len(line.split()) == 3]
        children = collections.defaultdict(list)
        rss = {}
        for p, parent, kb in rows:
            children[parent].append(p)
            rss[p] = kb * 1024
        total, todo = 0, [pid]
        while todo:
            p = todo.pop()
            total += rss.get(p, 0)
            todo.extend(children.get(p, ()))
        return total
    except Exception:
        return 0


def directory_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


# -------------------------------------------------------------------------------- state
class State:
    """What the router learns and keeps between runs: how much memory each model really took."""

    def __init__(self, path=None):
        self.path = path
        self.data = {}
        self.lock = threading.Lock()
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    self.data = loaded
            except (OSError, ValueError):
                pass

    def resident(self, key):
        value = (self.data.get(key) or {}).get("resident_bytes")
        return int(value) if isinstance(value, (int, float)) and value > 0 else None

    def load_seconds(self, key):
        """How long this model takes to load, as its slowest of the last three starts: a cold read from a slow drive is what a
        wrong eviction would cost again, and the next start is faster only by luck (the disk cache). None when never timed."""
        loads = [v for v in ((self.data.get(key) or {}).get("loads") or []) if isinstance(v, (int, float)) and v > 0]
        return max(loads) if loads else None

    def peak(self, key):
        value = (self.data.get(key) or {}).get("peak_bytes")
        return int(value) if isinstance(value, (int, float)) and value > 0 else None

    def record_load(self, key, model_id, seconds):
        with self.lock:
            saved = dict(self.data.get(key) or {})
            saved["id"] = model_id
            saved["loads"] = (list(saved.get("loads") or []) + [round(float(seconds), 1)])[-3:]
            self.data[key] = saved
            self._save()

    def record_peak(self, key, model_id, peak):
        """The most memory this model was seen to hold while working. Only ever raised, and written only for a rise of 3% or more."""
        with self.lock:
            saved = dict(self.data.get(key) or {})
            before = saved.get("peak_bytes") or 0
            if peak <= before * 1.03:
                return False
            saved["id"], saved["peak_bytes"] = model_id, int(peak)
            self.data[key] = saved
            self._save()
            return True

    def _save(self):
        """Write the file (call with the lock held)."""
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, indent=1)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def cached_entry(self, key):
        """The /v1/models object and card the model announced the last time it ran, or (None, None)."""
        saved = self.data.get(key) or {}
        entry, card = saved.get("entry"), saved.get("card")
        return (entry if isinstance(entry, dict) else None), (card if isinstance(card, dict) else None)

    def record(self, key, model_id, resident, entry=None, card=None):
        if not resident and not entry:
            return
        with self.lock:
            saved = dict(self.data.get(key) or {})
            saved["id"] = model_id
            if resident:
                saved["resident_bytes"] = int(resident)
            if entry:
                saved["entry"], saved["card"] = entry, card
            self.data[key] = saved
            if not self.path:
                return
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump(self.data, handle, indent=1)
                os.replace(tmp, self.path)
            except OSError:
                pass


KNOWN_FIELDS = ("path", "id", "name", "family", "modality", "size_bytes")


def why_missing(path, exists=os.path.exists, empty=lambda folder: not os.listdir(folder)):
    """(reason, forget) for a model that was found on an earlier scan and is not now. A drive or share that is not there is the
    reason to keep it listed; a folder that is there but is no longer a model says its files are not what they were; a folder gone
    from a place that is there was deleted, and is forgotten. `exists` and `empty` (is this folder empty?) are parameters for tests."""
    path = path.rstrip("/\\")
    if exists(path):
        return "its files are no longer recognised as a model (something was deleted or changed): try `coli model verify`", False
    drive = os.path.splitdrive(path)[0]
    if drive and not exists(drive + os.sep):
        return f"the drive {drive} is not connected", False
    probe = os.path.dirname(path)                   # the nearest folder above it that is there
    while probe and not exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    if not drive and probe and not exists(os.path.dirname(path)) and len([p for p in probe.replace("\\", "/").split("/") if p]) <= 3:
        try:
            if empty(probe):                                # a drive that is not plugged in leaves its mount point an empty folder
                return f"the drive or share at {probe} is not mounted", False
        except OSError:
            pass
    return None, True


def default_state_path():
    try:
        import setup_flow
        return os.path.join(setup_flow.setup_home(), "router.json")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".colibri-router.json")


def pressure_floor():
    """Free memory, in bytes, under which an idle model is stopped: PRESSURE_FLOOR_GB or PRESSURE_FLOOR_SHARE of the total, whichever
    is more. COLI_MEMORY_FLOOR_GB overrides it (0 turns the watch off). None when the machine's memory cannot be read."""
    explicit = os.environ.get("COLI_MEMORY_FLOOR_GB")
    if explicit:
        try:
            return int(float(explicit) * 2**30) or None
        except ValueError:
            log(f"COLI_MEMORY_FLOOR_GB={explicit!r} is not a number of GB; ignored")
    total = total_memory()
    return int(max(PRESSURE_FLOOR_GB * 2**30, (total or 0) * PRESSURE_FLOOR_SHARE)) if total else None


def default_jobs_path():
    """Where the job records live: COLI_JOBS, else jobs.jsonl beside the router's other state."""
    explicit = os.environ.get("COLI_JOBS")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    return os.path.join(os.path.dirname(default_state_path()), "jobs.jsonl")


def job_logger(logger):
    """What to do when a job ends: a line in the request log (if there is one), so Requests shows the render that the request
    asking for it only accepted."""
    def finished(snapshot):
        if logger is None:
            return
        started = snapshot.get("started") or snapshot["created"]
        failed = snapshot["status"] == "failed"
        record = {"source": "router", "method": "POST", "path": snapshot.get("endpoint") or "", "job": snapshot["id"],
                  "model": snapshot.get("model"), "status": {"succeeded": 200, "cancelled": 499}.get(snapshot["status"], 500),
                  "seconds": max(0, (snapshot.get("finished") or started) - started)}
        if failed and snapshot.get("error"):
            record["error"] = snapshot["error"].get("code") or snapshot["error"].get("message", "")[:200]
        elif snapshot["status"] == "cancelled":
            record["error"] = "cancelled"
        logger.write({k: v for k, v in record.items() if v not in (None, "")})
    return finished


# ------------------------------------------------------------------------------ catalog
class Catalog:
    """The models on this machine, by the names clients use for them."""

    def __init__(self, roots=None, models=(), engine_ready=None, worker_kv_slots=1, ttl=10.0, entries=None, known_path=None):
        self.roots, self.models = roots, tuple(models)
        self.known_path = known_path          # where the models seen before are remembered, or None: a model that goes is simply gone
        self.engine_ready = engine_ready or (lambda family: True)
        self.kv_slots, self.ttl = worker_kv_slots, ttl
        self._fixed = entries                  # tests give the entries directly
        self._entries, self._at = [], 0.0
        self._details = {}                     # key -> (mtime of its folder, what model_info found)
        self.lock = threading.Lock()

    def _scan(self):
        if self._fixed is not None:
            return [dict(entry) for entry in self._fixed]
        import model_switch
        from family_registry import family_by_id
        roots = self.roots
        if roots is None:
            roots = model_switch.candidate_roots(os.path.join(os.path.expanduser("~"), "colibri-models", "_"),
                                                 os.environ, model_switch.setup_models_root())
        ctx = model_switch.SwitchContext(os.path.join(os.path.expanduser("~"), "colibri-models", "_none"),
                                         kv_slots=self.kv_slots, roots=list(roots), environ={})
        entries = model_switch.scan_installed(ctx, self.engine_ready)
        for path in self.models:                # models named one by one, wherever they are
            one = model_switch.SwitchContext(path, kv_slots=self.kv_slots, roots=[], environ={})
            for entry in model_switch.scan_installed(one, self.engine_ready):
                if entry["path"] == os.path.realpath(path) and all(e["path"] != entry["path"] for e in entries):
                    entry["reason"] = model_switch.why_not(family_by_id(entry["family"]), one, self.engine_ready)
                    entries.append(entry)
        for entry in entries:
            entry["current"] = False
            entry["switchable"] = entry["reason"] is None
        return self._with_known(entries)

    # ---- models that were here and are not
    def _load_known(self):
        try:
            with open(self.known_path, encoding="utf-8") as handle:
                found = json.load(handle).get("models")
            return found if isinstance(found, dict) else {}
        except (OSError, ValueError, AttributeError):
            return {}

    def _with_known(self, entries):
        """Add the models seen on an earlier scan that are not found now, listed and switched off with the reason: a drive that is
        not plugged in, files that are no longer a model. A model whose folder was deleted (its parent is there, it is not) is
        forgotten, so the list does not keep what somebody removed on purpose."""
        if not self.known_path:
            return entries
        known, changed = self._load_known(), False
        present = {entry["key"] for entry in entries}
        for entry in entries:
            if entry.get("unfinished"):
                continue                       # not a model yet: nothing to come back to
            record = {name: entry.get(name) for name in KNOWN_FIELDS}
            if known.get(entry["key"]) != record:
                known[entry["key"]], changed = record, True
        for key, record in list(known.items()):
            if key in present:
                continue
            reason, forget = why_missing(record.get("path") or "")
            if forget:
                del known[key]
                changed = True
                continue
            entries.append({**record, "key": key, "reason": reason, "missing": True, "current": False, "switchable": False,
                            "verified": False, "unfinished": False})
        if changed:
            try:
                os.makedirs(os.path.dirname(self.known_path), exist_ok=True)
                tmp = self.known_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump({"version": 1, "models": known}, handle, indent=1)
                os.replace(tmp, self.known_path)
            except OSError:
                pass
        return entries

    def entries(self, refresh=False):
        with self.lock:
            age = time.monotonic() - self._at
            if not self._entries or age > self.ttl or (refresh and age > 2.0):
                self._entries, self._at = self._scan(), time.monotonic()
            return [dict(entry) for entry in self._entries]

    def details(self, entry):
        """What the model folder says about itself, from its small files only (model_info.inspect): architecture, weights, the
        model's own context limit, where it came from. Read once and kept until the folder changes; None for a model that is not there."""
        if entry.get("missing") or not entry.get("path"):
            return None
        try:
            stamp = os.stat(entry["path"]).st_mtime_ns
        except OSError:
            return None
        with self.lock:
            cached = self._details.get(entry["key"])
        if cached and cached[0] == stamp:
            return cached[1]
        import model_info
        found = model_info.inspect(entry["path"])
        out = {name: found[name] for name in ("architecture", "quantisation", "context_length", "shards", "bytes") if found.get(name)}
        if (found.get("source") or {}).get("repo"):
            out["source"] = found["source"]["repo"]
        if found["problems"]:
            out["problems"] = found["problems"][:3]
        with self.lock:
            self._details[entry["key"]] = (stamp, out)
        return out

    def aliases(self, entry):
        names = {entry["id"], entry["key"], os.path.basename(entry["path"].rstrip("/\\")), entry.get("name", "")}
        if entry["id"].endswith("-colibri"):
            names.add(entry["id"][:-len("-colibri")])
        return {name.lower() for name in names if name}

    def find(self, name):
        """The entry a request's `model` names: its id, its key, its folder name, or the id without
        the `-colibri` suffix. Looks again on a miss, so a model downloaded since is found."""
        if not isinstance(name, str) or not name:
            return None
        wanted = name.lower()
        for refresh in (False, True):
            matches = [entry for entry in self.entries(refresh) if wanted in self.aliases(entry)]
            if matches:
                # a copy that cannot be used (half downloaded, a drive that is unplugged) must not hide one that can
                return next((entry for entry in matches if entry["reason"] is None), matches[0])
        return None

    def by_key(self, key):
        for entry in self.entries():
            if entry["key"] == key:
                return entry
        return None

    def scan_roots(self):
        """The folders whose children are models: the ones given to the router, else the default ones."""
        if self._fixed is not None or self.roots is not None:
            return list(self.roots or [])
        import model_switch
        return model_switch.candidate_roots(os.path.join(os.path.expanduser("~"), "colibri-models", "_"), os.environ,
                                            model_switch.setup_models_root())

    def install_roots(self):
        """Where a model fetched through the router may go: a folder this catalog scans, so it is listed when it arrives.
        A router started with no folder of its own still has ~/colibri-models, made on the first install."""
        roots = self.scan_roots()
        if not roots and self.roots is None and self._fixed is None:
            roots = [os.path.join(os.path.expanduser("~"), "colibri-models")]
        return roots

    def invalidate(self):
        """Forget the last scan: a model was fetched or removed, and the next listing must see it."""
        with self.lock:
            self._at = 0.0


# ------------------------------------------------------------------------------- worker
def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post_json(port, path, body, timeout=5):
    payload = json.dumps(body).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("POST", path, body=payload, headers={"Host": f"127.0.0.1:{port}", "Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        return response.status, json.loads(data) if data else None
    finally:
        connection.close()


def fetch_json(port, path, timeout=5):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
        response = connection.getresponse()
        body = response.read()
        return response.status, json.loads(body) if body else None
    finally:
        connection.close()


class Worker:
    """One model in its own `coli serve` process on a loopback port."""

    def __init__(self, entry, keep_alive):
        self.entry = entry
        self.key = entry["key"]
        self.state = "starting"                 # starting -> ready -> stopping -> (gone); or failed
        self.port = None
        self.proc = None
        self.job = None
        self.keep_alive = keep_alive
        self.inflight = 0
        self.started = time.monotonic()
        self.last_used = self.started
        self.resident = 0
        self.model_entry, self.jev = None, None
        self.gpu = None                         # the gpu.Decision it was started with
        self.card = False                       # on the graphics card now (started with COLI_VULKAN=1, and not given back since)
        self.demoted = False                    # it gave the card back and runs on the CPU
        self.error = None
        self.log = collections.deque(maxlen=40)
        self.settled = threading.Event()        # set when starting or stopping has finished

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _pump(self):
        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self.log.append(line)
                sys.stderr.write("[%s] %s\n" % (self.entry["id"], line))
                log_tail.TAIL.add(self.entry["id"], line)
        sys.stderr.flush()
        try:
            self.proc.stdout.close()
        except OSError:
            pass

    def spawn(self, command, env, make_job):
        self.port = int(command[command.index("--port") + 1]) if "--port" in command else free_port()
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP         # so CTRL_BREAK reaches this worker only
        self.proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, creationflags=flags)
        if make_job:
            self.job = make_job(self.proc.pid)
        threading.Thread(target=self._pump, daemon=True).start()

    def wait_ready(self, poll=0.2):
        """Until the worker answers /health. No deadline: a 500 GB model takes a long while to load."""
        while True:
            if not self.alive():
                tail = " | ".join(list(self.log)[-6:]) or f"exit code {self.proc.returncode}"
                raise ApiError(503, f"`{self.entry['id']}` did not start: {tail}", "model_load_failed",
                               None, "server_error")
            try:
                status, _ = fetch_json(self.port, "/health", timeout=2)
                if status == 200:
                    break
            except (OSError, ValueError, http.client.HTTPException):
                pass
            time.sleep(poll)
        try:
            status, listing = fetch_json(self.port, "/v1/models")
            if status == 200 and listing and listing.get("data"):
                self.model_entry = listing["data"][0]
                models = listing.get("models") or []
                self.jev = models[0] if models else None
        except (OSError, ValueError, http.client.HTTPException):
            pass

    def terminate(self, grace):
        """Ask it to finish (its engine saves what it learned), then make sure."""
        if not self.alive():
            return
        asked = False
        try:
            if sys.platform == "win32":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.terminate()
            asked = True
        except (OSError, ValueError):
            pass
        if asked:
            try:
                self.proc.wait(grace)
                return
            except subprocess.TimeoutExpired:
                pass
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
        else:
            self.proc.kill()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            pass

    def close_job(self):
        if self.job and sys.platform == "win32":
            try:
                import ctypes
                ctypes.WinDLL("kernel32").CloseHandle(self.job)
            except Exception:
                pass
            self.job = None


# --------------------------------------------------------------------------------- pool
class Pool:
    """The workers: start on demand, stop when idle, make room before loading."""

    def __init__(self, catalog, *, keep_alive=DEFAULT_KEEP_ALIVE, max_loaded=DEFAULT_MAX_LOADED,
                 worker_command=None, worker_env=None, state=None, memory=None, measure=None,
                 grace=STOP_GRACE, make_job=None, queue_timeout=300.0, gpu=None, settings=None,
                 fallback_memory=CARD_FALLBACK_S, pressure_floor=None, pressure_min_idle=PRESSURE_MIN_IDLE,
                 pressure_cooldown=PRESSURE_COOLDOWN, peak_sample=PEAK_SAMPLE_S, watch_pressure=True,
                 card_evict=True, card_min_idle=CARD_MIN_IDLE, card_settle=CARD_SETTLE_S, card_demote=True,
                 card_promote=True, promote_min_idle=PROMOTE_MIN_IDLE, promote_min_age=PROMOTE_MIN_AGE,
                 promote_chat=True, promote_chat_min_idle=PROMOTE_CHAT_MIN_IDLE, promote_chat_max_reload=PROMOTE_CHAT_MAX_RELOAD_S):
        self.catalog = catalog
        self.gpu = gpu                                   # gpu.GpuPolicy, or None: the environment is left as it is
        self.settings = settings                         # model_settings.ModelSettings, or None
        self.decisions = {}                              # key -> the gpu.Decision of its last start
        self.fallbacks = {}                              # key -> (until, why): a start on the card failed, so it runs on the CPU
        self.fallback_memory = fallback_memory
        self.keep_alive, self.max_loaded = keep_alive, max_loaded
        self.worker_command = worker_command
        self.worker_env = worker_env
        self.state = state or State()
        self.memory = memory or available_memory         # bytes free for a new model, or None
        self.measure = measure or resident_bytes
        self.grace, self.make_job, self.queue_timeout = grace, make_job, queue_timeout
        # Memory pressure between requests: free memory under `pressure_floor` bytes stops an idle model (None: never, as before).
        self.pressure_floor, self.pressure_min_idle, self.pressure_cooldown = pressure_floor, pressure_min_idle, pressure_cooldown
        self.card_evict, self.card_min_idle, self.card_settle = card_evict, card_min_idle, card_settle
        self.card_demote = card_demote                # give the card back and keep the model running, where its engine can
        self.card_promote, self.promote_min_idle, self.promote_min_age = card_promote, promote_min_idle, promote_min_age
        self.promote_chat, self.promote_chat_min_idle = promote_chat, promote_chat_min_idle
        self.promote_chat_max_reload = promote_chat_max_reload
        self.promotions = {"stops": 0, "chat_stops": 0, "last": None}
        self._promised = {}                              # key -> (GB, until): a model stopped to load on the card, which holds its room
        self.watch_pressure = watch_pressure              # the reaper's loop calls relieve(); a test that calls it itself turns this off
        self.pressure = {"stops": 0, "last": None}
        self._relief_until = 0.0
        self.peak_sample = peak_sample
        self._sampled = 0.0
        self.workers = {}
        self.cv = threading.Condition(threading.Lock())
        self.closed = False
        self.last_error = {}                             # key -> why the last load failed
        self._wake = threading.Event()                   # a model that should go as soon as it is idle
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()

    # ---- sizes
    def estimate(self, entry):
        """What loading this model will take: the most it was seen to hold (it grows as its caches fill, so more than it held right
        after loading), else what it took after loading last time, else its size on disk."""
        known = self.state.resident(entry["key"])
        peak = self.state.peak(entry["key"])
        if known or peak:
            return max(known or 0, peak or 0)
        size = entry.get("size_bytes") or directory_bytes(entry["path"])
        total = total_memory()
        return min(size, int(total * 0.9)) if total else size

    def fits(self, entry, free):
        return free is None or self.estimate(entry) <= free * FIT_HEADROOM

    # ---- views
    def snapshot(self):
        with self.cv:
            return list(self.workers.values())

    def ready_workers(self):
        with self.cv:
            return [w for w in self.workers.values() if w.state == "ready" and w.alive()]

    def most_recent(self, modality=None):
        ready = [w for w in self.ready_workers() if modality is None or w.entry.get("modality") == modality]
        return max(ready, key=lambda w: w.last_used, default=None)

    def get(self, key):
        with self.cv:
            return self.workers.get(key)

    # ---- the planner (called with the lock held): one step, then the caller acts
    def _plan(self, entry, free):
        key = entry["key"]
        worker = self.workers.get(key)
        if worker is not None:
            if worker.state == "ready" and worker.alive():
                worker.inflight += 1
                worker.last_used = time.monotonic()
                return "use", worker
            if worker.state in ("starting", "stopping", "demoting"):
                return "wait", worker
            del self.workers[key]                        # failed, or died while ready
        others = list(self.workers.values())
        counted = [w for w in others if w.state in ("starting", "ready", "stopping", "demoting")]
        room = self.max_loaded <= 0 or len(counted) < self.max_loaded
        if room and self.fits(entry, free):
            return "load", self._register(entry)
        victims = [w for w in others if w.state == "ready" and w.inflight == 0]
        if victims:
            need = self.estimate(entry) / FIT_HEADROOM - free if (free is not None and not self.fits(entry, free)) else 0
            victim = self.pick_victim(victims, need)
            victim.state = "stopping"
            victim.settled.clear()
            return "evict", victim
        if not others:                                   # nothing left to free: try anyway
            log(f"`{entry['id']}` may not fit in memory (needs about "
                f"{self.estimate(entry) / 2**30:.1f} GB), loading it anyway")
            return "load", self._register(entry)
        return "block", None                             # others are busy or loading: wait for them

    # ---- who goes first
    def reload_seconds(self, worker_or_entry):
        """What stopping this model would cost to undo: its measured load time, else a guess from its size (a slow disk), which
        is only good for comparing models nobody has timed."""
        entry = getattr(worker_or_entry, "entry", worker_or_entry)
        measured = self.state.load_seconds(entry["key"])
        if measured:
            return measured
        size = self.state.resident(entry["key"]) or entry.get("size_bytes") or 0
        return max(5.0, size / RELOAD_BYTES_PER_S)

    def victim_score(self, worker, now=None):
        """Idle time times the memory stopping it frees, over what it costs to load again: the higher, the better a victim. A model
        that has sat idle a long time, is big, and loads quickly goes first; one on a slow drive stays."""
        now = time.monotonic() if now is None else now
        idle = max(now - worker.last_used, 1.0)
        return idle * max(worker.resident or 1, 1) / max(self.reload_seconds(worker), 1.0)

    def pick_victim(self, victims, need=0):
        """The idle worker to stop. When memory is what is short (`need` bytes), one that frees enough by itself is preferred,
        so a single stop settles it instead of a string of small ones; among those, the best score."""
        enough = [w for w in victims if w.resident >= need] if need > 0 else victims
        now = time.monotonic()
        return max(enough or victims, key=lambda w: (self.victim_score(w, now), now - w.last_used))      # a tie goes to the longest idle

    def _register(self, entry):
        worker = Worker(entry, self.keep_alive)
        self.workers[entry["key"]] = worker
        return worker

    def acquire(self, entry):
        """A ready worker for `entry` with one request counted against it (release it when done).
        Loads the model if needed, after making room. Waiting for room is bounded by the queue
        timeout; the load itself is not."""
        if self.settings is not None and not self.settings.enabled(entry):
            raise disabled_error(entry)
        deadline = time.monotonic() + self.queue_timeout
        while True:
            free = self.memory()                         # read outside the lock: it may be slow
            with self.cv:
                if self.closed:
                    raise ApiError(503, "The server is shutting down.", "shutting_down", None, "server_error")
                action, worker = self._plan(entry, free)
                if action == "block":
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise ApiError(429, "Timed out waiting for memory to load the model.", "queue_timeout",
                                       None, "rate_limit_error", {"Retry-After": "1"})
                    self.cv.wait(min(left, 1.0))
                    continue
            if action == "use":
                return worker
            if action == "evict":
                self._stop(worker, "making room for `%s`" % entry["id"])
                continue
            if action == "wait":
                worker.settled.wait(1.0)
                if worker.state == "failed":
                    raise worker.error
                continue
            return self._load(worker)                    # action == "load"

    def card_required(self, entry):
        """True when someone chose the card for this model (a pin, or --gpu on): a failure there is reported, not worked around."""
        if self.settings is not None and self.settings.gpu(entry) == "vulkan":
            return True
        return self.gpu is not None and self.gpu.mode == "on"

    def _load(self, worker):
        entry = worker.entry
        log(f"loading `{entry['id']}`")
        began = time.monotonic()
        card_failure = None                              # why the first try, on the card, failed; set when the CPU is being tried
        while True:
            worker.card = False
            try:
                command, env = self._command(entry, card_failure)
                worker.gpu = self.decisions.get(entry["key"])
                worker.card = env.get("COLI_VULKAN") == "1"
                worker.spawn(command, env, self.make_job)
                worker.wait_ready()
                break
            except BaseException as error:
                api = error if isinstance(error, ApiError) else ApiError(
                    503, f"`{entry['id']}` did not start: {error}", "model_load_failed", None, "server_error")
                if (card_failure is None and worker.card and isinstance(error, Exception) and not self.closed
                        and not self.card_required(entry)):
                    # One more try, on the CPU. If that works, the next starts stay there for a while: a card that fails
                    # to load a model (a driver, a device that was lost, memory another program took) is not worth a
                    # wait every time. If it fails too, the model is the problem and the card is not blamed.
                    card_failure = api.message
                    log(f"`{entry['id']}` did not start on the graphics card; trying the CPU")
                    worker.terminate(2.0)
                    worker.close_job()
                    worker.log.clear()
                    continue
                self._load_failed(worker, entry, api)
                raise api
        if card_failure is not None:
            self.fallbacks[worker.key] = (time.monotonic() + self.fallback_memory, card_failure)
            log(f"`{entry['id']}` runs on the CPU; the next starts stay there for {self.fallback_memory / 60:.0f} min")
        self._loaded(worker, entry, began)
        return worker

    def _load_failed(self, worker, entry, api):
        worker.error, worker.state = api, "failed"
        worker.terminate(2.0)
        worker.close_job()
        with self.cv:
            if self.workers.get(worker.key) is worker:
                del self.workers[worker.key]
            self.last_error[worker.key] = api.message
            worker.settled.set()
            self.cv.notify_all()
        log(api.message)

    def _loaded(self, worker, entry, began):
        worker.resident = self.measure(worker.proc.pid, worker.job)
        self.state.record(worker.key, entry["id"], worker.resident, worker.model_entry, worker.jev)
        self.state.record_load(worker.key, entry["id"], time.monotonic() - began)
        with self.cv:
            worker.state = "ready"
            worker.inflight += 1
            worker.last_used = time.monotonic()
            self.last_error.pop(worker.key, None)
            worker.settled.set()
            self.cv.notify_all()
        log(f"`{entry['id']}` ready in {time.monotonic() - began:.0f} s, "
            f"{worker.resident / 2**30:.1f} GB resident, port {worker.port}")

    def _command(self, entry, card_failure=None):
        """The command and environment for a worker. `card_failure` is why a start on the card just failed: this one is on the CPU."""
        port = free_port()
        if self.worker_command is not None:
            command = list(self.worker_command(entry, port))
        else:
            launcher = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coli")
            command = [sys.executable, launcher, "serve", "--model", entry["path"], "--host", "127.0.0.1",
                       "--port", str(port)]
        if "--port" not in command:
            command += ["--port", str(port)]
        env = dict(os.environ)
        for name in ("COLI_API_KEY", "COLI_MODEL_ID", "COLI_MODEL", "COLI_SWITCH_FALLBACK"):
            env.pop(name, None)                           # a worker is loopback only, and the key is the router's
        env["PYTHONUTF8"] = "1"
        if self.worker_env is not None:
            env.update(self.worker_env(entry) or {})
        window = context_window(self.settings, entry)
        if window:
            env[window[0]] = str(window[1])
            log(f"`{entry['id']}` will have a context window of {window[1]} tokens (its own setting)")
        if self.gpu is not None or (self.settings is not None and self.settings.gpu(entry)):
            policy = self.gpu or gpu_mod.GpuPolicy()
            pinned = self.settings.gpu(entry) if self.settings is not None else None
            if card_failure is None:
                self._free_the_card(policy, entry, pinned)
            decision = policy.decide(entry, pinned)
            env.update(decision.env)
            self.decisions[entry["key"]] = decision
            log(f"`{entry['id']}` will run {decision.summary()}")
        reason = card_failure
        if reason is None:
            remembered = self.fallbacks.get(entry["key"])
            if remembered is not None:
                if time.monotonic() >= remembered[0] or self.card_required(entry):
                    del self.fallbacks[entry["key"]]           # the time is up, or the card was chosen since
                else:
                    reason = remembered[1]
        if reason is not None and env.get("COLI_VULKAN") == "1":
            decision = gpu_mod.Decision("cpu", "the card failed to start it: " + reason[-160:], env={"COLI_VULKAN": "0"})
            env.update(decision.env)
            self.decisions[entry["key"]] = decision
            log(f"`{entry['id']}` will run {decision.summary()}")
        return command, env

    def _free_the_card(self, policy, entry, pinned):
        """A model about to start would go to the CPU only because the card's memory is taken. If an idle model holds the card, stop
        it first (the best victim by the usual score, and only one that has been idle a while, so two models that both want the card
        do not trade it back and forth), and wait for the card to show its memory free. The model stopped loads again, on the CPU or
        the card, when it is next asked for. Off with `card_evict=False`; never for a model pinned to either side."""
        if not self.card_evict or pinned in ("cpu", "vulkan") or policy.mode != "auto":
            return
        for _ in range(4):
            if policy.decide(entry, pinned).mode != "cpu":
                return
            now = time.monotonic()
            with self.cv:
                holders = [w for w in self.workers.values() if w.state == "ready" and w.card and w.inflight == 0 and w.alive()
                           and w.key != entry["key"] and now - w.last_used >= self.card_min_idle]
                if not holders:
                    return
                victim = self.pick_victim(holders)
                victim.state = "demoting" if self.card_demote else "stopping"
                victim.settled.clear()
            why = f"it holds the graphics card, which `{entry['id']}` needs, and has been idle for {now - victim.last_used:.0f} s"
            if self.card_demote:
                outcome = self.demote(victim, why)
                if outcome == "busy":
                    return                                         # it is in use again: the new model goes to the CPU this time
                if outcome != "done":
                    with self.cv:
                        victim.state = "stopping"
                    self._stop(victim, why)
            else:
                self._stop(victim, why)
            end = time.monotonic() + self.card_settle
            while time.monotonic() < end and policy.decide(entry, pinned).mode == "cpu":
                time.sleep(0.25)                                   # the driver gives the memory back when the process has gone

    def demote(self, worker, why):
        """Take the graphics card from an idle worker and leave it running on the CPU (POST /v1/demote to its gateway, which asks
        the engine; the engine tears the card down and reads back what only the card held). Called with the worker in the
        "demoting" state, which keeps requests off it, and leaves it "ready" (done, and it counts as a CPU model now) or in
        whatever the caller makes it. Returns "done", "busy" (a request got there first: nothing changed, and the worker is ready
        again) or "failed" (it could not, an older worker or an engine without the means: nothing changed, the caller stops it)."""
        began = time.monotonic()
        outcome, reason = "failed", None
        try:
            status, answer = post_json(worker.port, "/v1/demote", {}, timeout=DEMOTE_TIMEOUT_S)
            if status == 200 and isinstance(answer, dict) and answer.get("demoted") is True:
                outcome = "done"
            elif status == 200 and isinstance(answer, dict):
                reason = answer.get("reason")
                if reason and "request is running" in reason:
                    outcome = "busy"
            else:
                reason = f"HTTP {status}"
        except (OSError, ValueError, http.client.HTTPException) as error:
            reason = str(error) or error.__class__.__name__
        if outcome == "done":
            seen = self.measure(worker.proc.pid, worker.job) if worker.alive() else None
            with self.cv:
                worker.card = False
                worker.demoted = True
                worker.gpu = gpu_mod.Decision("cpu", "it gave the graphics card back: " + why, env={"COLI_VULKAN": "0"})
                if seen:
                    worker.resident = seen
                worker.state = "ready"
                worker.settled.set()
                self.cv.notify_all()
            log(f"`{worker.entry['id']}` gave the graphics card back in {time.monotonic() - began:.1f} s and runs on the CPU "
                f"({worker.resident / 2**30:.1f} GB resident); {why}")
        elif outcome == "busy":
            with self.cv:
                worker.state = "ready"
                worker.settled.set()
                self.cv.notify_all()
        else:
            log(f"`{worker.entry['id']}` could not give the graphics card back ({reason}); it will be stopped instead")
        return outcome

    def release(self, worker):
        with self.cv:
            worker.inflight = max(0, worker.inflight - 1)
            worker.last_used = time.monotonic()
            self.cv.notify_all()
            if worker.keep_alive == 0 and worker.inflight == 0:
                self._wake.set()                         # keep_alive 0: the reaper need not wait for its next look

    def release_wake(self):
        """Have the reaper look now (a model was told to go when idle)."""
        self._wake.set()

    def _stop(self, worker, why):
        log(f"unloading `{worker.entry['id']}` ({why})")
        worker.terminate(self.grace)
        worker.close_job()
        with self.cv:
            worker.state = "gone"
            if self.workers.get(worker.key) is worker:
                del self.workers[worker.key]
            worker.settled.set()
            self.cv.notify_all()

    def unload(self, key, why="asked to"):
        """Stop a model if it is loaded and idle. Returns 'unloaded', 'busy' or 'not loaded'."""
        with self.cv:
            worker = self.workers.get(key)
            if worker is None or worker.state not in ("ready", "starting"):
                return "not loaded"
            if worker.state == "starting" or worker.inflight:
                return "busy"
            worker.state = "stopping"
            worker.settled.clear()
        self._stop(worker, why)
        return "unloaded"

    # ---- idle workers and dead ones
    def _reap_loop(self):
        while not self.closed:
            self._wake.wait(1.0)
            self._wake.clear()
            try:
                self.reap()
                self.sample_peaks()
                if self.watch_pressure:
                    self.relieve()
                    self.promote()
            except Exception as error:                    # the reaper must not die
                log(f"reaper: {error}")

    def reap(self):
        if self.closed:
            return
        now = time.monotonic()
        due = []
        with self.cv:
            for worker in list(self.workers.values()):
                if worker.state == "ready" and not worker.alive():
                    log(f"`{worker.entry['id']}` exited on its own")
                    if worker.card and not self.card_required(worker.entry):
                        self.fallbacks[worker.key] = (now + self.fallback_memory,
                                                      f"it exited on its own (code {worker.proc.returncode}) while on the card")
                        log(f"`{worker.entry['id']}` will start on the CPU for the next {self.fallback_memory / 60:.0f} min")
                    del self.workers[worker.key]
                    worker.state = "gone"
                    worker.settled.set()
                    self.cv.notify_all()
                elif (worker.state == "ready" and worker.inflight == 0 and worker.keep_alive is not None
                      and now - worker.last_used >= worker.keep_alive):
                    worker.state = "stopping"
                    worker.settled.clear()
                    due.append(worker)
        for worker in due:
            self._stop(worker, "idle for %d s" % worker.keep_alive if worker.keep_alive else "keep-alive is 0")

    def sample_peaks(self, now=None):
        """Read the memory of each worker that is answering, every few seconds, and remember the most any of them reached: the
        router measures a model only right after it loads, and it grows as its caches fill."""
        now = time.monotonic() if now is None else now
        if now - self._sampled < self.peak_sample:
            return
        self._sampled = now
        with self.cv:
            busy = [w for w in self.workers.values() if w.state == "ready" and w.inflight > 0 and w.alive()]
        for worker in busy:
            seen = self.measure(worker.proc.pid, worker.job)
            if seen and seen > worker.resident:
                worker.resident = seen                   # what the planner counts for it from now on
            if seen and self.state.record_peak(worker.key, worker.entry["id"], seen):
                log(f"`{worker.entry['id']}` now holds {seen / 2**30:.1f} GB while working")

    def relieve(self, now=None):
        """Memory pressure between requests: when the memory free falls under the floor, stop the best idle model (the one
        whose stopping frees enough, longest idle, cheapest to load again), then leave a while for the memory to come back before
        looking again, so one spike does not empty the router. A model that is answering is never touched, nor one used just
        now unless memory is nearly gone. Returns the model stopped, or None."""
        floor = self.pressure_floor
        if not floor or self.closed:
            return None
        now = time.monotonic() if now is None else now
        if now < self._relief_until:
            return None
        free = self.memory()
        if free is None or free >= floor:
            return None
        with self.cv:
            urgent = free < floor / 2
            victims = [w for w in self.workers.values() if w.state == "ready" and w.inflight == 0 and w.alive()
                       and (urgent or now - w.last_used >= self.pressure_min_idle)]
            if not victims:
                return None
            victim = self.pick_victim(victims, floor * PRESSURE_TARGET - free)
            victim.state = "stopping"
            victim.settled.clear()
        self._relief_until = now + self.pressure_cooldown
        self.pressure["stops"] += 1
        self.pressure["last"] = {"model": victim.entry["id"], "free_bytes": free, "at": int(time.time())}
        self._stop(victim, f"memory is low: {free / 2**30:.1f} GB free, under {floor / 2**30:.1f} GB; it was idle for "
                           f"{now - victim.last_used:.0f} s and held {victim.resident / 2**30:.1f} GB")
        return victim.entry["id"]

    def promote(self, now=None):
        """Give a model back the card it could not have. A model that started on the CPU only because the card was full (its
        decision says so: not pinned, not a failed start), or that gave the card back to another (demoted), stays there until it
        unloads, which is half an hour by default. When the card has room for it now (what the policy asks of a new model, and its
        size with some margin) and it is idle, it is stopped, and the next request loads it on the card. Only when `--gpu auto`,
        and never one that has not run for promote_min_age seconds or has been used within its idle wait, so a model does not bounce.
        The models in PROMOTE_MODALITIES (they load in seconds) wait promote_min_idle. A chat model (PROMOTE_CHAT_MODALITIES) costs a
        slow reload and its conversation cache, so it waits promote_chat_min_idle and is moved only when its reload takes no longer
        than promote_chat_max_reload (`promote_chat=False` leaves chat models alone). A model stopped for the card holds its room
        until it loads (or PROMOTE_PROMISE_TTL passes), so two models are not stopped for one free card. Returns the model stopped,
        or None."""
        policy = self.gpu
        if not self.card_promote or self.closed or policy is None or policy.mode != "auto":
            return None
        now = time.monotonic() if now is None else now
        seen = policy.memory()
        if seen is None:
            return None
        with self.cv:
            for key in [k for k, (_, until) in self._promised.items() if until <= now or k in self.workers]:
                del self._promised[key]
            held = sum(gb for gb, _ in self._promised.values())
            usable = seen[0] - policy.reserve_gb - held
            chosen, chat = None, False
            for w in self.workers.values():
                if w.state != "ready" or w.inflight or not w.alive() or w.card or now - w.started < self.promote_min_age:
                    continue
                modality = w.entry.get("modality")
                remembered = self.fallbacks.get(w.key)
                if remembered is not None and now < remembered[0]:
                    continue                                 # a start on the card failed lately: the next start is on the CPU anyway
                if modality in PROMOTE_MODALITIES:
                    if w.demoted or w.gpu is None or not w.gpu.room or now - w.last_used < self.promote_min_idle:
                        continue
                    this_chat = False
                elif modality in PROMOTE_CHAT_MODALITIES and self.promote_chat:
                    if not w.demoted and (w.gpu is None or not w.gpu.room):
                        continue                             # on the CPU by choice (pinned, or the card failed): not for the card to say
                    if now - w.last_used < self.promote_chat_min_idle or self.reload_seconds(w) > self.promote_chat_max_reload:
                        continue
                    this_chat = True
                else:
                    continue
                size = max(w.resident, w.entry.get("size_bytes") or 0) / 2**30
                if usable < max(policy.min_gb, size * PROMOTE_MARGIN):
                    continue
                # the quick models go first (a reload costs little), then the one idle longest
                if chosen is None or (this_chat, w.last_used) < (chat, chosen.last_used):
                    chosen, chat = w, this_chat
            if chosen is None:
                return None
            chosen.state = "stopping"
            chosen.settled.clear()
            self._promised[chosen.key] = (max(chosen.resident, chosen.entry.get("size_bytes") or 0) / 2**30 * PROMOTE_MARGIN,
                                          now + PROMOTE_PROMISE_TTL)
        self.promotions["stops"] += 1
        self.promotions["chat_stops"] += 1 if chat else 0
        self.promotions["last"] = {"model": chosen.entry["id"], "free_gb": round(seen[0], 1), "at": int(time.time()), "chat": chat}
        was = "it gave the card back to another model" if chosen.demoted else "it started on the CPU because the card was full"
        self._stop(chosen, f"the graphics card has room for it now ({seen[0]:.1f} GB free) and {was}; "
                           f"it loads on the card when it is next asked for")
        return chosen.entry["id"]

    def close(self):
        self.closed = True
        with self.cv:
            workers = list(self.workers.values())
            self.cv.notify_all()
        threads = [threading.Thread(target=w.terminate, args=(self.grace,)) for w in workers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for worker in workers:
            worker.close_job()


# ---------------------------------------------------------------------------------- gate
class Gate:
    """Lets requests for one model run together, but never two models at once.

    A request for model M may start when no other model has requests running and no request
    for another model is waiting ahead of it. Waiters are served in order, so a model that is
    asked for continuously cannot starve the others. Bounded like a worker's own queue.

    A request with a higher priority goes ahead of the waiting requests with a lower one (it never interrupts a
    running one), and keeps its order among those of the same priority. A low-priority request that keeps being
    overtaken still gives up with the usual timeout."""

    def __init__(self, max_queue=8, timeout=300.0, parallel=False):
        self.max_queue, self.timeout, self.parallel = max_queue, timeout, parallel
        self.cv = threading.Condition(threading.Lock())
        self.active = collections.Counter()
        self.waiting = collections.deque()

    def _may_run(self, ticket, key):
        if any(k != key and n > 0 for k, n in self.active.items()):
            return False
        for other, other_key, _ in self.waiting:
            if other is ticket:
                return True
            if other_key != key:
                return False
        return True

    def _enqueue(self, ticket, key, priority):
        position = len(self.waiting)
        while position > 0 and self.waiting[position - 1][2] < priority:
            position -= 1
        self.waiting.insert(position, (ticket, key, priority))

    def acquire(self, key, priority=0, cancelled=None, patient=False):
        """Wait for a turn. A background job passes `cancelled` (polled while it waits; raises jobs.Cancelled) and `patient`
        (no timeout: a job may wait behind an hour of rendering, which a chat's five minutes would not allow)."""
        if self.parallel:
            with self.cv:
                self.active[key] += 1
            return
        ticket = object()
        deadline = float("inf") if patient else time.monotonic() + self.timeout
        with self.cv:
            if self._may_run(ticket, key):            # nobody else computing, nobody ahead of us for another model
                self.active[key] += 1
                return
            if len(self.waiting) >= self.max_queue:
                raise ApiError(429, "The inference queue is full.", "queue_full", None, "rate_limit_error",
                               {"Retry-After": "1"})
            self._enqueue(ticket, key, priority)
            try:
                while not self._may_run(ticket, key):
                    if cancelled is not None and cancelled():
                        raise jobs.Cancelled()
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise ApiError(429, "Timed out waiting for the inference engine.", "queue_timeout", None,
                                       "rate_limit_error", {"Retry-After": "1"})
                    self.cv.wait(min(left, 0.25) if cancelled is not None else left)
                self.active[key] += 1
            finally:
                self.waiting.remove((ticket, key, priority))
                self.cv.notify_all()

    def release(self, key):
        with self.cv:
            self.active[key] -= 1
            if self.active[key] <= 0:
                del self.active[key]
            self.cv.notify_all()

    def stats(self):
        with self.cv:
            return {"active": sum(self.active.values()), "queued": len(self.waiting)}


# ------------------------------------------------------------------------------- server
UNSET = object()                    # a request that says nothing about keep_alive


def keep_alive_value(value):
    """A request's `keep_alive` as seconds, None for never unload, or UNSET when absent. Like Ollama's: a duration
    string ('30s', '10m', '2h'), a number of seconds, 0 (unload as soon as the model is idle) or a negative number
    (never)."""
    if value is None or value is UNSET:
        return UNSET
    if isinstance(value, bool):
        value = object()
    if isinstance(value, (int, float)):
        return None if value < 0 else float(value)
    if isinstance(value, str):
        try:
            return parse_duration(value)
        except ValueError as error:
            raise ApiError(400, f"`keep_alive`: {error}", "invalid_request_error", "keep_alive")
    raise ApiError(400, "`keep_alive` must be a duration such as 30s or 10m, a number of seconds, 0 or -1.",
                   "invalid_request_error", "keep_alive")


def _multipart_value(raw, name):
    at = raw.find(b'name="%s"' % name.encode())
    if at < 0:
        return None
    start = raw.find(b"\r\n\r\n", at)
    if start < 0:
        return None
    end = raw.find(b"\r\n", start + 4)
    return raw[start + 4:end if end >= 0 else None].decode("utf-8", "replace").strip()


def _boundary(content_type):
    for piece in (content_type or "").split(";")[1:]:
        key, _, value = piece.strip().partition("=")
        if key.lower() == "boundary":
            return value.strip('"')
    return ""


def _multipart_with(raw, content_type, defaults):
    """The multipart body with a text field for each of `defaults` it does not already have."""
    boundary = _boundary(content_type)
    closing = raw.rfind(b"--" + boundary.encode() + b"--") if boundary else -1
    if closing < 0:
        return raw
    parts = b""
    for key, value in defaults.items():
        if value is None or _multipart_value(raw, key) is not None:
            continue
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        parts += (b'--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n' % (boundary.encode(), key.encode())
                  + text.encode("utf-8") + b"\r\n")
    return raw[:closing] + parts + raw[closing:]


def _multipart_without(raw, name, content_type):
    """The multipart body without the field `name` (its whole part), or the body itself when it has none."""
    at = raw.find(b'name="%s"' % name.encode())
    if at < 0:
        return raw
    boundary = _boundary(content_type)
    if not boundary:
        return raw
    delimiter = b"--" + boundary.encode()
    start = raw.rfind(delimiter, 0, at)
    end = raw.find(b"\r\n" + delimiter, at)
    if start < 0 or end < 0:
        return raw
    return raw[:start] + raw[end + 2:]


class Body:
    """A request body read once for what the router needs from it: the `model` and the `keep_alive`. A JSON body is
    parsed one time (a song's audio may make it many megabytes); a multipart body is searched, not parsed."""

    def __init__(self, content_type, raw):
        self.content_type, self.raw = content_type, raw
        self.multipart = (content_type or "").split(";", 1)[0].strip().lower() == "multipart/form-data"
        self.json = None
        if not self.multipart:
            try:
                parsed = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                parsed = None
            self.json = parsed if isinstance(parsed, dict) else None

    @property
    def model(self):
        if self.multipart:
            return _multipart_value(self.raw, "model") or None
        model = self.json.get("model") if self.json is not None else None
        return model if isinstance(model, str) else None

    @property
    def keep_alive(self):
        """The request's keep_alive as keep_alive_value() reads it: seconds, None for never, or UNSET."""
        if self.multipart:
            text = _multipart_value(self.raw, "keep_alive")
            return keep_alive_value(text if text else None)
        return keep_alive_value(self.json.get("keep_alive") if self.json is not None else None)

    def for_worker(self, model_id, preset=None, path=""):
        """The body to send on: `model` set to the id the worker knows it by (a client may name it by its key, its
        folder or the id without `-colibri`; a worker accepts only its own id) and without `keep_alive`, which is
        the router's. The same bytes when there is nothing to change.

        With a preset, what the request left out is filled in from it: each default the body has no value for, and the
        preset's system prompt when the conversation has none (see apply_preset)."""
        if self.multipart:
            raw = self.raw
            at = raw.find(b'name="model"')
            start = raw.find(b"\r\n\r\n", at) if at >= 0 else -1
            if start >= 0:
                end = raw.find(b"\r\n", start + 4)
                end = len(raw) if end < 0 else end
                if raw[start + 4:end] != model_id.encode():
                    raw = raw[:start + 4] + model_id.encode() + raw[end:]
            raw = _multipart_without(raw, "keep_alive", self.content_type)
            if preset is not None and preset.defaults:
                raw = _multipart_with(raw, self.content_type, preset.defaults)
            return raw
        if self.json is None:
            return self.raw
        body = {k: v for k, v in self.json.items() if k != "keep_alive"}
        body["model"] = model_id
        if preset is not None:
            apply_preset(body, preset, path)
        if body == self.json:
            return self.raw
        return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


def apply_preset(body, preset, path):
    """Fill a JSON request body in from a preset, in place. A value the request carries (anything but null) wins. The
    system prompt is added only when the conversation has none of its own: /v1/chat/completions gets a system message
    in front, /v1/messages its `system` field; other endpoints have no system prompt to give."""
    for key, value in preset.defaults.items():
        if body.get(key) is None:
            body[key] = value
    if not preset.system:
        return
    if path == "/v1/chat/completions":
        messages = body.get("messages")
        if isinstance(messages, list) and not any(
                isinstance(m, dict) and m.get("role") in ("system", "developer") for m in messages):
            body["messages"] = [{"role": "system", "content": preset.system}] + messages
    elif path == "/v1/messages":
        if body.get("system") in (None, "", []):
            body["system"] = preset.system


def extract_model(content_type, raw):
    """The `model` of a JSON or multipart/form-data request body, or None."""
    return Body(content_type, raw).model


def with_model(content_type, raw, model_id):
    """The body to send to the worker that holds `model_id` (see Body.for_worker)."""
    return Body(content_type, raw).for_worker(model_id)


# ------------------------------------------------------------------------------- background jobs
BACKGROUND_PATHS = {"/v1/images/generations": "image", "/v1/images/edits": "image", "/v1/audio/music": "music"}
LIBRARY_KIND = {"image": "images", "music": "music"}
SUMMARY_KEYS = ("prompt", "size", "width", "height", "steps", "seed", "duration", "language", "task")
PROGRESS_KEYS = ("stage", "step", "steps", "elapsed")


def wants_async(headers):
    """Does the request say `Prefer: respond-async` (RFC 7240)?"""
    for value in headers.get_all("Prefer") or ():
        if "respond-async" in [token.split("=")[0].strip().lower() for token in value.split(",")]:
            return True
    return False


def request_summary(body):
    """What a job list shows of a request: the prompt (cut) and the few numbers that say what was asked for."""
    found = {}
    for key in SUMMARY_KEYS + ("caption",):
        if body.multipart:
            value = _multipart_value(body.raw, key)
        else:
            value = body.json.get(key) if body.json is not None else None
        if isinstance(value, (str, int, float)) and not isinstance(value, bool) and value != "":
            found[key] = value
    caption = found.pop("caption", None)
    if "prompt" not in found and caption is not None:
        found["prompt"] = caption
    if isinstance(found.get("prompt"), str) and len(found["prompt"]) > 300:
        found["prompt"] = found["prompt"][:300] + "..."
    return found


def streaming_body(path, body, raw):
    """`raw` (what the worker would have been sent) changed to ask for server-sent events, so the job can follow progress: no
    `response_format` (a song as a WAV file cannot be streamed, and the result is a kept file anyway) and no partial pictures
    (nobody is watching them; an image engine skips them when told zero)."""
    changes = {"stream": True}
    if path != "/v1/audio/music":
        changes["partial_images"] = 0
    if body.multipart:
        for key in ("stream", "partial_images", "response_format"):
            raw = _multipart_without(raw, key, body.content_type)
        return _multipart_with(raw, body.content_type,
                               {key: ("true" if value is True else value) for key, value in changes.items()})
    if body.json is None:
        return raw
    changed = {k: v for k, v in json.loads(raw).items() if k != "response_format"}
    changed.update(changes)
    return json.dumps(changed, ensure_ascii=False, separators=(",", ":")).encode()


def sse_events(response):
    """(event, data) for each server-sent event in a response, as it arrives. Comments (keep-alives) are skipped."""
    buffer = b""
    while True:
        chunk = response.read1(65536)
        if not chunk:
            return
        buffer += chunk.replace(b"\r\n", b"\n")
        while b"\n\n" in buffer:
            block, buffer = buffer.split(b"\n\n", 1)
            name, data = None, []
            for line in block.decode("utf-8", "replace").split("\n"):
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
            if data:
                yield name, "\n".join(data)


def error_of(status, raw):
    """(message, code) of an error body in the workers' shape."""
    try:
        error = json.loads(raw).get("error") or {}
        return error.get("message") or f"The model answered {status}.", error.get("code")
    except (ValueError, AttributeError):
        return f"The model answered {status}.", None


def kept_result(kind, payload, summary):
    """The job's result: the file the library lists. The worker says which it kept; one that kept nothing
    (COLI_IMAGE_SAVE=0) leaves the picture only in the answer, and a background job has nobody holding the connection to
    take it from there, so it is kept here."""
    import base64
    info = payload.get("colibri") or {}
    library_kind = LIBRARY_KIND[kind]
    name = info.get("file")
    if not name:
        try:
            data = base64.b64decode(payload["data"][0]["b64_json"])
        except (KeyError, IndexError, TypeError, ValueError):
            raise jobs.JobError(502, "The model finished but sent no result.", "no_result") from None
        seed, prompt = info.get("seed", 0), summary.get("prompt", "")
        try:
            if kind == "image":
                import image_engine
                name = os.path.basename(image_engine.save_png(data, prompt, seed, library.folder(library_kind)))
            else:
                import music_engine
                name = os.path.basename(music_engine.save_wav(data, prompt, seed, library.folder(library_kind)))
        except OSError as error:
            raise jobs.JobError(500, f"The result could not be kept: {error}", "save_failed") from None
    result = {"kind": library_kind, "file": name, "url": f"/v1/library/{library_kind}/{name}"}
    for key in ("width", "height", "seed", "steps", "duration", "sample_rate", "timings"):
        if info.get(key) is not None:
            result[key] = info[key]
    return result


def run_job_request(server, work, ctx):
    """Do one background request: wait for the model's turn, make it ready, send the request and follow the answer to the end.
    Raises jobs.Cancelled when cancelled and jobs.JobError for what went wrong; returns the job's result."""
    entry, path = work["entry"], work["path"]
    gate, pool = server.gate, server.pool
    priority = request_priority(path, None)
    while True:                          # a full queue is waited out, not failed: this job is already accepted
        try:
            gate.acquire(entry["key"], priority, cancelled=ctx.cancelled.is_set, patient=True)
            break
        except ApiError as error:
            if error.code != "queue_full":
                raise jobs.JobError(error.status, error.message, error.code) from None
            if ctx.cancelled.wait(1.0):
                raise jobs.Cancelled()
    try:
        try:
            worker = pool.acquire(entry)
        except ApiError as error:
            raise jobs.JobError(error.status, error.message, error.code) from None
        try:
            if ctx.cancelled.is_set():
                raise jobs.Cancelled()
            if work["keep"] is not UNSET:
                worker.keep_alive = work["keep"]
            return follow_worker(worker, work, ctx)
        finally:
            pool.release(worker)
    finally:
        gate.release(entry["key"])


def follow_worker(worker, work, ctx):
    kind = work["kind"]
    connection = http.client.HTTPConnection("127.0.0.1", worker.port, timeout=None)

    held = {}                             # http.client lets go of its socket once an answer says it will close: keep our own

    def abort():                          # the worker notices a client that left and stops computing
        sock = held.get("sock")
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
    ctx.on_abort(abort)
    try:
        headers = {"Content-Type": work["content_type"], "Content-Length": str(len(work["raw"])),
                   "Host": f"127.0.0.1:{worker.port}"}
        if work.get("trace"):
            headers["x-colibri-trace"] = work["trace"]
        try:
            connection.connect()
            held["sock"] = connection.sock
            if ctx.cancelled.is_set():
                raise jobs.Cancelled()
            connection.request("POST", work["path"], body=work["raw"], headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                message, code = error_of(response.status, response.read())
                raise jobs.JobError(response.status, message, code)
            payload = None
            if "text/event-stream" in (response.getheader("Content-Type") or ""):
                for name, data in sse_events(response):
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except ValueError:
                        continue
                    if name and name.endswith(".progress"):
                        ctx.progress({k: event.get(k) for k in PROGRESS_KEYS if event.get(k) is not None})
                    elif name and name.endswith(".completed"):
                        payload = event
                    elif name == "error":
                        error = (event.get("error") or {}) if isinstance(event, dict) else {}
                        raise jobs.JobError(500, error.get("message") or "The model failed.", error.get("code"))
            else:
                payload = json.loads(response.read())
        except (OSError, http.client.HTTPException, ValueError) as error:
            if ctx.cancelled.is_set():
                raise jobs.Cancelled() from None
            raise jobs.JobError(502, f"`{worker.entry['id']}` did not answer: {error}", "worker_unavailable") from None
        if ctx.cancelled.is_set():
            raise jobs.Cancelled()
        if payload is None:
            raise jobs.JobError(502, f"`{worker.entry['id']}` closed the connection before it finished.", "worker_closed")
        return kept_result(kind, payload, work["summary"])
    finally:
        connection.close()


class RouterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, pool, catalog, gate, api_key=None, cors_origins=(), allowed_hosts=(),
                 queue_timeout=300.0, presets=None, request_log=None, settings=None, jobs=None, installer=None):
        super().__init__(address, RouterHandler)
        self.pool, self.catalog, self.gate = pool, catalog, gate
        self.installer = installer             # model_installer.Installer: models fetched and removed through the API, or None
        self.jobs = jobs                       # jobs.JobManager: pictures and songs that go on without the connection
        self.presets = presets                 # presets.Presets, or None for a router without any
        self.request_log = request_log         # request_log.RequestLog, or None
        self.settings = settings               # model_settings.ModelSettings, or None
        self.api_key = api_key
        self.cors_origins = tuple(cors_origins)
        self.allowed_hosts = tuple(h.strip().lower() for h in allowed_hosts if h and h.strip())
        self.queue_timeout = queue_timeout
        self.started = time.time()


class RouterHandler(BaseHTTPRequestHandler):
    # HTTP/1.0: each response ends by closing the connection, which is how a streamed body
    # (server-sent events) of unknown length is delimited without re-chunking it.
    protocol_version = "HTTP/1.0"
    server_version = "colibri-router"
    timeout = 60

    def log_message(self, fmt, *args):
        sys.stderr.write("[router] %s - %s\n" % (self.address_string(), fmt % args))

    # ---- the guards the workers have, once, here
    def _check_host(self):
        host = self.headers.get("Host", "")
        if host.startswith("["):
            name = host[1:].split("]", 1)[0]
        elif host.count(":") == 1:
            name = host.rsplit(":", 1)[0]
        else:
            name = host
        name = name.strip().lower()
        allowed = set(LOOPBACK) | set(self.server.allowed_hosts)
        if "*" in allowed:
            return
        allowed.add(str(self.server.server_address[0]).strip("[]").lower())
        if name not in allowed:
            raise ApiError(403, "Host header %r not allowed. Add it with --allowed-host %s, or "
                                "--allowed-host '*' to accept any host when the bind is already public."
                           % (name or "(empty)", name or "<host>"), "forbidden")

    def _authed(self):
        key = self.server.api_key
        if not key:
            return True
        return (hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {key}")
                or hmac.compare_digest(self.headers.get("x-api-key", ""), key))

    def _require_auth(self):
        if not self._authed():
            raise ApiError(401, "Invalid or missing API key.", "invalid_api_key", None, "authentication_error")

    def end_headers(self):
        for name, value in web_headers.BASELINE:
            self.send_header(name, value)
        super().end_headers()

    def _check_fetch_site(self):
        """A second check behind the origin allow-list: a page on another site is refused before anything runs."""
        if web_headers.cross_site_refused(urlsplit(self.path).path, self.headers.get("Sec-Fetch-Site"),
                                          self.headers.get("Origin"), self.server.cors_origins):
            raise ApiError(403, "Cross-site requests are not allowed. Add the page's origin with --cors-origin.", "forbidden")

    def _cors(self):
        origin = self.headers.get("Origin")
        origins = self.server.cors_origins
        if not origin or ("*" not in origins and origin not in origins):
            return
        self.send_header("Access-Control-Allow-Origin", "*" if "*" in origins else origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key, anthropic-version, "
                         "X-TypeSafe-SDK, X-TypeSafe-Runtime, X-TypeSafe-Retry-Count, X-Colibri-Tags")
        self.send_header("Access-Control-Expose-Headers", "x-request-id, x-typesafe-request-id, "
                         "x-colibri-queue-wait-ms, x-colibri-elapsed-ms, x-colibri-engine-ms, Retry-After")
        self.send_header("Access-Control-Max-Age", "600")
        if "*" not in origins:
            self.send_header("Vary", "Origin")

    # ---- replies
    def send_json(self, status, body, headers=None):
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    _body_read = False

    def drain(self, limit=4 << 20):
        """Swallow an unread request body: closing a socket with received data still in it resets the
        connection on some systems, and the client then never sees the answer."""
        if self._body_read:
            return
        self._body_read = True
        try:
            remaining = int(self.headers.get("Content-Length", "0").strip())
        except ValueError:
            return
        if not 0 < remaining <= limit:
            return
        try:
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 1 << 16))
                if not chunk:
                    return
                remaining -= len(chunk)
        except OSError:
            pass

    _status = None
    _error = None

    def send_response(self, code, message=None):
        self._status = code                    # what the request log says happened
        super().send_response(code, message)

    def fail(self, error):
        self._error = error.code or error.message[:200]
        self.drain()
        try:
            self.send_json(error.status, error.body(), error.headers)
        except OSError:
            pass

    def serve_static(self, path):
        import openai_server
        return openai_server.APIHandler.serve_static(self, path)

    @property
    def WEB_DIST(self):
        import openai_server
        return openai_server.APIHandler.WEB_DIST

    def send_cors_headers(self):               # serve_static's name for it
        self._cors()

    # ---- GET
    def do_OPTIONS(self):
        try:
            self._check_host()
        except ApiError:
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self):
        try:
            self._check_host()
            self._check_fetch_site()
            split = urlsplit(self.path)
            path, query = split.path, parse_qs(split.query)
            if path == "/health":
                return self.send_json(200, self.health(query))
            if path in ("/experts", "/profile", "/metrics"):
                # the loaded model's own telemetry; without a key, or with no model loaded, the empty shape
                empty = ({"rows": 0, "cols": 0, "map": "", "hits": "", "seq": 0} if path == "/experts"
                         else {"seq": 0, "turns": []})
                if path == "/metrics":
                    self._require_auth()
                elif not self._authed():
                    return self.send_json(200, empty)
                worker = self.server.pool.most_recent()
                if worker is None:
                    if path == "/metrics":
                        raise ApiError(404, "No model is loaded.", "no_model_loaded")
                    return self.send_json(200, empty)
                return self.relay(worker, "GET", self.path, {}, b"")
            if self.serve_static(path):
                return
            self._require_auth()
            if path == "/v1/models":
                return self.send_json(200, self.models_list())
            if path == "/v1/models/settings":
                return self.send_json(200, self.settings_list())
            if path == "/v1/models/installed":
                return self.send_json(200, self.installed())
            if path == "/v1/models/catalog":
                return self.send_json(200, self.install_catalog())
            if path == "/v1/models/installs":
                return self.send_json(200, {"object": "list", "data": self.need_installer().job_list()})
            if path.startswith("/v1/models/installs/"):
                return self.send_json(200, self.install_job(unquote(path[len("/v1/models/installs/"):])))
            if path == "/v1/presets":
                return self.send_json(200, self.presets_list())
            if path == "/v1/mcp/servers":
                return self.send_json(200, self.mcp_servers_list())
            if path == "/v1/audio/voices":
                return self.send_json(200, self.voices_list())
            if path == "/v1/requests":
                return self.send_json(200, self.requests_list(query))
            if path == "/v1/logs":
                return self.send_json(200, self.logs_list(query))
            if path == "/v1/jobs":
                return self.send_json(200, self.jobs_list(query))
            if path.startswith("/v1/jobs/"):
                return self.send_json(200, self.job_get(unquote(path[len("/v1/jobs/"):])))
            if path == "/v1/library":
                return self.send_json(200, self.library_listing(query))
            if path.startswith("/v1/library/"):
                return self.library_file(unquote(path[len("/v1/library/"):]))
            if path.startswith("/v1/models/"):
                return self.send_json(200, self.model_object(unquote(path[len("/v1/models/"):])))
            raise ApiError(404, "Not found.", "not_found")
        except ApiError as error:
            self.fail(error)
        except ConnectionError:
            pass

    # ---- POST
    def do_POST(self):
        started = time.monotonic()
        seen = {}                                   # what the request log says about it
        try:
            self._check_host()
            self._check_fetch_site()
            self._require_auth()
            path = urlsplit(self.path).path
            if path == "/v1/models/switch":
                return self.switch()
            if path == "/v1/models/unload":
                return self.unload()
            if path == "/v1/models/settings":
                return self.settings_set()
            if path == "/v1/models/install":
                return self.install_start()
            if path == "/v1/models/install/cancel":
                return self.install_cancel()
            if path == "/v1/models/uninstall":
                return self.uninstall()
            if path == "/v1/presets":
                return self.preset_save()
            if path == "/v1/presets/delete":
                return self.preset_delete()
            if path == "/v1/mcp/servers":
                return self.mcp_server_save()
            if path == "/v1/mcp/servers/delete":
                return self.mcp_server_delete()
            if path == "/v1/audio/voices":
                return self.voice_save()
            if path == "/v1/audio/voices/delete":
                return self.voice_delete()
            if path == "/v1/jobs/cancel":
                return self.job_cancel()
            if path == "/v1/responses":
                return self.responses_api()
            if path not in COMPUTE_PATHS:
                raise ApiError(404, "Not found.", "not_found")
            raw = self.read_body(path)
            body = Body(self.headers.get("Content-Type"), raw)
            seen.update(path=path, bytes_in=len(raw), model=body.model, stream=bool(body.json and body.json.get("stream")))
            if self.server.request_log is not None and self.server.request_log.bodies and body.json is not None:
                seen["request"] = request_log.shorten(body.json)
            keep = body.keep_alive                  # read before anything loads: a bad value must not cost a load
            tags = self.tags_header()               # likewise
            if tags:
                seen["tags"] = tags
            preset = self.preset_named(body.model)
            if preset is not None:
                seen["preset"] = preset.name
            entry = self.resolve(path, preset.model if preset else body.model, preset)
            seen["resolved"] = entry["id"]
            if preset is not None and keep is UNSET:
                keep = self.preset_keep_alive(preset)
            if self.server.request_log is not None:
                self._trace = uuid.uuid4().hex[:16]    # the worker writes it too, so the two lines can be joined
            if wants_async(self.headers):
                return self.job_submit(path, entry, body, preset, keep, seen)
            self.serve_model(entry, path, body.for_worker(entry["id"], preset, path), keep)
        except ApiError as error:
            self.fail(error)
        except (ConnectionError, TimeoutError):
            pass
        except Exception as error:
            self.log_error("request failed: %s", error)
            self.fail(ApiError(500, "The colibri router failed to process the request.", "router_error",
                               None, "server_error"))
        finally:
            log = self.server.request_log
            if log is not None and seen:
                record = {"source": "router", "trace": getattr(self, "_trace", None), "method": "POST",
                          "status": self._status, "seconds": round(time.monotonic() - started, 3),
                          "client": self.client_address[0] if self.client_address else None, **seen}
                if self._error:
                    record["error"] = self._error
                log.write({k: v for k, v in record.items() if v is not None})
            self._status = self._error = None

    def responses_api(self):
        """POST /v1/responses: the Responses API, which is all Codex speaks, over this router's own chat endpoint.

        responses_shim.py (from upstream colibri) translates the request into a chat request, sends it to /v1/chat/completions here
        (so it is routed, loaded, queued and logged like any chat request, with the caller's key and tags) and translates the
        chat stream back into Responses events."""
        import responses_shim
        responses_shim.serve(self, urlsplit(f"http://127.0.0.1:{self.server.server_address[1]}"))

    def tags_header(self):
        """The tags the request carries (client_tags.py), None for none. A header that does not fit is a 400 that says what to change."""
        try:
            return client_tags.parse(self.headers.get(client_tags.HEADER))
        except client_tags.TagError as error:
            raise ApiError(400, str(error), "invalid_tags", client_tags.HEADER) from None

    def preset_named(self, name):
        """The preset a request's `model` names, or None. A real model's name is never taken from it."""
        presets = self.server.presets
        if presets is None or not isinstance(name, str):
            return None
        preset = presets.get(name)
        if preset is None:
            return None
        catalog = self.server.catalog
        if any(name.lower() in catalog.aliases(e) for e in catalog.entries()):
            return None
        return preset

    @staticmethod
    def preset_keep_alive(preset):
        try:
            return keep_alive_value(preset.keep_alive)
        except ApiError:
            log(f"preset `{preset.name}`: bad keep_alive {preset.keep_alive!r} ignored")
            return UNSET

    def read_body(self, path):
        import image_engine
        import openai_server
        if self.headers.get_all("Transfer-Encoding"):
            raise ApiError(400, "Transfer-Encoding is not supported; send one Content-Length.")
        try:
            length = int(self.headers.get("Content-Length", "0").strip())
        except ValueError:
            raise ApiError(400, "Invalid Content-Length header.")
        multipart = (self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() == "multipart/form-data")
        if path == "/v1/audio/music":
            limit = openai_server.MUSIC_MAX_BODY_MULTIPART if multipart else openai_server.MUSIC_MAX_BODY_JSON
        elif path == "/v1/images/edits":
            limit = image_engine.MAX_EDIT_BODY
        elif path in (openai_server.TRANSCRIBE_ENDPOINT, openai_server.ALIGN_ENDPOINT):
            limit = openai_server.TRANSCRIBE_MAX_BODY if multipart else openai_server.TRANSCRIBE_MAX_BODY * 4 // 3 + (1 << 20)
        else:
            limit = openai_server.MAX_BODY
        if length < 1 or length > limit:
            raise ApiError(400, f"Request body must be between 1 and {limit} bytes.")
        raw = b""
        chunks = []
        self._body_read = True
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1 << 20))
            if not chunk:
                raise ConnectionError("client went away mid-body")
            chunks.append(chunk)
            remaining -= len(chunk)
        return raw.join(chunks)

    def resolve(self, path, model, preset=None):
        try:
            return self._resolve(path, model)
        except ApiError as error:
            if preset is not None and error.code in ("model_not_found", "not_switchable"):
                raise ApiError(error.status, f"The preset `{preset.name}` names the model `{preset.model}`, which "
                                             f"{'does not exist here' if error.code == 'model_not_found' else 'cannot start here'}"
                                             f" ({error.message})", error.code, "model") from None
            raise

    def _resolve(self, path, model):
        catalog = self.server.catalog
        if path == "/v1/systemone" and not catalog.find(model):
            # a client written for another server names whatever model it likes: the text model answers
            worker = self.server.pool.most_recent("text")
            if worker is not None:
                return worker.entry
            texts = [e for e in catalog.entries() if e.get("modality") == "text" and e["reason"] is None]
            if len(texts) == 1:
                return texts[0]
        if model is None:
            entries = [e for e in catalog.entries() if e["reason"] is None]
            if len(entries) == 1:
                return entries[0]
            raise ApiError(400, "`model` is required.", "missing_required_parameter", "model")
        entry = catalog.find(model)
        if entry is None:
            raise ApiError(404, f"The model `{model}` does not exist.", "model_not_found", "model")
        if entry["reason"] is not None:
            raise ApiError(409, f"`{entry['id']}` cannot be started here: {entry['reason']}.", "not_switchable", "model")
        if self.disabled(entry):
            raise disabled_error(entry)
        return entry

    def serve_model(self, entry, path, raw, keep=UNSET):
        gate, pool = self.server.gate, self.server.pool
        gate.acquire(entry["key"], request_priority(path, self.headers))
        try:
            worker = pool.acquire(entry)
            if keep is not UNSET:
                worker.keep_alive = keep            # sticks until a request says otherwise, as Ollama's does
            try:
                self.relay(worker, "POST", self.path, self.forwardable_headers(), raw)
            finally:
                pool.release(worker)
        finally:
            gate.release(entry["key"])

    def forwardable_headers(self):
        headers = {name: value for name, value in self.headers.items() if name.lower() not in DROP_REQUEST}
        trace = getattr(self, "_trace", None)
        if trace:
            headers["x-colibri-trace"] = trace
        return headers

    def relay(self, worker, method, target, headers, body):
        """Send the request to the worker and stream its answer back as it arrives."""
        connection = http.client.HTTPConnection("127.0.0.1", worker.port, timeout=None)
        try:
            try:
                connection.putrequest(method, target, skip_host=True, skip_accept_encoding=True)
                connection.putheader("Host", f"127.0.0.1:{worker.port}")
                for name, value in headers.items():
                    connection.putheader(name, value)
                if body or method == "POST":
                    connection.putheader("Content-Length", str(len(body)))
                connection.endheaders(body)
                response = connection.getresponse()
            except (OSError, http.client.HTTPException) as error:
                raise ApiError(502, f"`{worker.entry['id']}` did not answer: {error}", "worker_unavailable",
                               None, "server_error")
            self.send_response(response.status)
            for name, value in response.getheaders():
                lower = name.lower()
                if lower in DROP_RESPONSE or lower.startswith("access-control-"):
                    continue
                if lower in web_headers.BASELINE_NAMES:
                    continue                      # this router sends its own, once
                if lower == "content-length" or lower not in HOP_BY_HOP:
                    self.send_header(name, value)
            self.send_header("x-colibri-model", worker.entry["id"])
            self._cors()
            self.end_headers()
            try:
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (OSError, http.client.HTTPException):
                pass                          # a client that left closes the worker's connection below, which cancels it
        finally:
            connection.close()

    # ---- what the router answers itself
    def health(self, query):
        payload = {"status": "ok"}
        if not self._authed():
            return payload
        pool, catalog = self.server.pool, self.server.catalog
        wanted = (query.get("model") or [None])[0]
        entry = self.entry_for(wanted) if wanted else None
        worker = pool.get(entry["key"]) if entry else pool.most_recent()
        if worker is not None and worker.state == "ready":
            try:
                status, body = fetch_json(worker.port, "/health", timeout=3)
                if status == 200 and isinstance(body, dict):
                    payload.update(body)
            except (OSError, ValueError, http.client.HTTPException):
                pass
        gate = self.server.gate
        payload.setdefault("scheduler", {"active": 0, "queued": 0, "capacity": 1, "max_queue": gate.max_queue,
                                         "admitted": 0, "completed": 0, "failed": 0, "rejected": 0,
                                         "timed_out": 0, "cancelled": 0})
        # the router's own list, after the loaded model's health was merged in: that one is a single server's
        payload["features"] = features.describe(router=True, presets=self.server.presets is not None,
                                                request_log=self.server.request_log is not None,
                                                model_settings=self.server.settings is not None,
                                                jobs=self.server.jobs is not None,
                                                model_install=self.server.installer is not None)
        payload["router"] = {
            "loaded": [self.worker_view(w) for w in pool.snapshot()],
            "max_loaded": pool.max_loaded,
            "keep_alive": pool.keep_alive,
            "parallel_models": self.server.gate.parallel,
            "gate": self.server.gate.stats(),
            "free_memory": pool.memory(),
            "memory_pressure": {"floor_bytes": pool.pressure_floor, **pool.pressure},
            "card_promote": {"on": pool.card_promote, **pool.promotions},
        }
        if pool.gpu is not None:
            payload["router"]["gpu"] = pool.gpu.public()
        return payload

    # ---- background jobs
    def job_submit(self, path, entry, body, preset, keep, seen):
        """`Prefer: respond-async`: accept the request as a job, answer 202 with it at once, and go on without the connection."""
        manager, server = self.server.jobs, self.server
        if path not in BACKGROUND_PATHS:
            raise ApiError(400, "`Prefer: respond-async` is for pictures and songs (/v1/images/generations, /v1/images/edits, "
                                "/v1/audio/music); a chat is answered while the connection is open.", "invalid_value")
        if manager is None:
            raise ApiError(501, "This router was started without background jobs.", "jobs_unavailable", None, "server_error")
        kind = BACKGROUND_PATHS[path]
        raw = streaming_body(path, body, body.for_worker(entry["id"], preset, path))
        work = {"entry": entry, "path": path, "kind": kind, "raw": raw, "content_type": body.content_type or "application/json",
                "keep": keep, "trace": getattr(self, "_trace", None), "summary": request_summary(body)}
        try:
            snapshot, created = manager.submit(kind, entry["id"], path, work["summary"],
                                               lambda ctx: run_job_request(server, work, ctx),
                                               (self.headers.get(jobs.ID_HEADER) or "").strip() or None, tags=seen.get("tags"))
        except jobs.JobError as error:
            raise ApiError(error.status, error.message, error.code, error.param,
                           "rate_limit_error" if error.status == 429 else "invalid_request_error",
                           {"Retry-After": "30"} if error.status == 429 else None) from None
        seen["job"] = snapshot["id"]
        self.send_json(202, snapshot, {"Location": f"/v1/jobs/{snapshot['id']}", "Preference-Applied": "respond-async"})

    def jobs_list(self, query):
        if self.server.jobs is None:
            raise ApiError(404, "Not found.", "not_found")
        try:
            status, limit, offset = jobs.page_arguments(query)
            return self.server.jobs.listing(status, limit, offset, client_tags.wanted(query.get("tag")))
        except jobs.JobError as error:
            raise ApiError(error.status, error.message, error.code, error.param) from None
        except client_tags.TagError as error:
            raise ApiError(400, str(error), "invalid_value", "tag") from None

    def job_get(self, job_id):
        found = self.server.jobs.get(job_id) if self.server.jobs is not None else None
        if found is None:
            raise ApiError(404, f"No job `{job_id}`.", "job_not_found", "id")
        return found

    def job_cancel(self):
        """Stop a job: a waiting one is removed, a running one is interrupted. Finished or unknown is `cancelled: false`."""
        if self.server.jobs is None:
            raise ApiError(404, "Not found.", "not_found")
        raw = self.read_body("/v1/jobs/cancel")
        try:
            wanted = json.loads(raw).get("id")
        except (ValueError, AttributeError):
            wanted = None
        if not isinstance(wanted, str) or not wanted:
            raise ApiError(400, "Send a JSON body with the job `id`.", "missing_required_parameter", "id")
        cancelled, status = self.server.jobs.cancel(wanted)
        self.send_json(200, {"id": wanted, "object": "job.cancel", "cancelled": cancelled, "status": status})

    def library_listing(self, query):
        try:
            return library.listing(*library.page_arguments(query))
        except library.LibraryError as error:
            raise ApiError(error.status, error.message, error.code, error.param) from None

    def library_file(self, rest):
        kind, _, name = rest.partition("/")
        try:
            library.serve(self, kind, name, self._cors)
        except library.LibraryError as error:
            raise ApiError(error.status, error.message, error.code, error.param) from None

    def worker_view(self, worker):
        left = None
        if worker.state == "ready" and worker.keep_alive is not None and worker.inflight == 0:
            left = max(0.0, worker.keep_alive - (time.monotonic() - worker.last_used))
        return {"id": worker.entry["id"], "key": worker.key, "state": worker.state, "port": worker.port,
                "resident_bytes": worker.resident, "inflight": worker.inflight, "expires_in": left,
                "keep_alive": worker.keep_alive, "gpu": worker.gpu.public() if worker.gpu is not None else None,
                "demoted": worker.demoted, "reload_seconds": round(self.server.pool.reload_seconds(worker), 1)}

    def models_list(self):
        """Every model that can be asked for, so a client can find them. A loaded one says what its engine
        announced; one that is not loaded says what it announced last time, or what its family can do."""
        pool, state = self.server.pool, self.server.pool.state
        data, cards = [], []
        for entry in sorted(self.server.catalog.entries(), key=lambda e: e["id"]):
            if entry["reason"] is not None:
                continue
            worker = pool.get(entry["key"])
            ready = worker is not None and worker.state == "ready" and worker.model_entry
            if ready:
                item, card = dict(worker.model_entry), worker.jev
            else:
                item, card = state.cached_entry(entry["key"])
                item = dict(item) if item else self.stub(entry)
            item["state"] = "loaded" if ready else "idle"
            if self.disabled(entry):
                item["state"] = "disabled"
            data.append(item)
            if card:
                cards.append(card)
        by_id = {item["id"]: item for item in data}
        for preset in (self.server.presets.all() if self.server.presets is not None else []):
            entry = self.server.catalog.find(preset.model)
            base = by_id.get(entry["id"]) if entry is not None else None      # None: not installed, or cannot start here
            if base is not None and base["state"] != "disabled" and self.preset_named(preset.name) is not None:
                data.append(self.as_preset(base, preset))
        return {"object": "list", "data": data, "models": cards}

    @staticmethod
    def stub(entry):
        """What is known of a model that has never run here: its modality says what it does."""
        item = {"id": entry["id"], "object": "model", "owned_by": "colibri", "created": 0}
        modality = entry.get("modality")
        if modality == "image":
            item["capabilities"] = ["image_generation"]
        elif modality == "audio":
            item["capabilities"] = ["music_generation"]
            item["audio"] = {}
        elif modality == "decision":
            item["capabilities"] = ["systemone", "decision"]
        elif modality == "embedding":
            item["capabilities"] = ["embeddings"]
            item["embedding"] = {}
        elif modality == "rerank":
            item["capabilities"] = ["rerank"]
            item["rerank"] = {}
        elif modality == "transcription" and "aligner" in str(entry.get("id", "")):
            item["capabilities"] = ["alignment"]
            item["alignment"] = {}
        elif modality == "transcription":
            item["capabilities"] = ["transcription"]
            item["transcription"] = {}
        elif modality == "speech":
            item["capabilities"] = ["speech"]
            item["speech"] = {}
        else:
            item["capabilities"] = ["chat", "systemone"]
        item["input_modalities"] = ["audio"] if modality == "transcription" else ["text"]
        return item

    def entry_for(self, name):
        """The catalog entry a name stands for: a model's own, or the base model of the preset it names."""
        preset = self.preset_named(name)
        return self.server.catalog.find(preset.model if preset else name)

    def model_object(self, name):
        entry = self.entry_for(name)
        if entry is None:
            raise ApiError(404, f"The model `{name}` does not exist.", "model_not_found", "model")
        worker = self.server.pool.get(entry["key"])
        if worker is not None and worker.state == "ready" and worker.model_entry:
            item = dict(worker.model_entry, state="loaded")
        else:
            cached, _card = self.server.pool.state.cached_entry(entry["key"])
            item = dict(cached or self.stub(entry), state="idle")
        preset = self.preset_named(name)
        return self.as_preset(item, preset) if preset is not None else item

    @staticmethod
    def as_preset(item, preset):
        """A /v1/models object for a preset: the base model's own, under the preset's name."""
        item = dict(item)
        item.update({"id": preset.name, "preset_of": item["id"], "preset": preset.public()})
        return item

    def presets_list(self):
        """GET /v1/presets: every preset in the file, whether it can be used, and why not."""
        presets = self.server.presets
        out = []
        for preset in (presets.all() if presets is not None else []):
            base = self.server.catalog.find(preset.model)
            if base is None:
                reason = f"the model `{preset.model}` is not installed"
            elif base["reason"] is not None:
                reason = f"the model cannot start here: {base['reason']}"
            elif any(preset.name in self.server.catalog.aliases(e) for e in self.server.catalog.entries()):
                reason = "the name is a model's own"
            else:
                reason = None
            out.append(dict(preset.public(), available=reason is None, reason=reason,
                            resolved=base["id"] if base else None))
        problems = [{"name": name, "reason": why} for name, why in (presets.problems if presets is not None else [])]
        return {"object": "list", "path": presets.path if presets is not None else None, "data": out,
                "problems": problems, "error": presets.error if presets is not None else None}

    def requests_list(self, query):
        """GET /v1/requests?limit=N: the last requests the request log holds, newest first (the dashboard's Requests tab)."""
        log = self.server.request_log
        if log is None:
            return {"object": "list", "enabled": False, "data": []}
        try:
            limit = max(1, min(500, int((query.get("limit") or ["50"])[0])))
        except ValueError:
            raise ApiError(400, "`limit` must be a number.", "invalid_request_error", "limit")
        try:
            tags = client_tags.wanted(query.get("tag"))
        except client_tags.TagError as error:
            raise ApiError(400, str(error), "invalid_value", "tag") from None
        return {"object": "list", "enabled": True, "path": log.path, "bodies": log.bodies,
                "data": request_log.recent(log.path, limit, tags)}

    def logs_list(self, query):
        """GET /v1/logs?since=N&limit=N&source=ID: the lines the router and its workers printed after number N, oldest
        first (the dashboard's Log view and the editor's output channel). `next` is the number to ask for next time."""
        def number(name, default, low, high):
            try:
                return max(low, min(high, int((query.get(name) or [str(default)])[0])))
            except ValueError:
                raise ApiError(400, f"`{name}` must be a number.", "invalid_request_error", name)
        since = number("since", 0, 0, 1 << 62)
        limit = number("limit", 500, 1, log_tail.DEFAULT_LINES)
        source = (query.get("source") or [""])[0] or None
        out = log_tail.TAIL.read(since, limit, source)
        out["object"] = "list"
        return out

    def installed(self):
        pool = self.server.pool
        out = []
        for entry in self.server.catalog.entries():
            worker = pool.get(entry["key"])
            state = worker.state if worker is not None else "idle"
            item = {k: v for k, v in entry.items() if k != "path"}
            details = self.server.catalog.details(entry)
            if details:
                item["details"] = details
            item["current"] = state == "ready"
            item["state"] = state
            item["enabled"] = not self.disabled(entry)
            item["switchable"] = entry["reason"] is None and item["enabled"] and state not in ("starting", "stopping")
            if self.server.settings is not None and self.server.settings.gpu(entry):
                item["gpu_setting"] = self.server.settings.gpu(entry)
            if worker is not None:
                item.update(self.worker_view(worker))
            elif entry["key"] in pool.last_error:
                item["last_error"] = pool.last_error[entry["key"]]
            out.append(item)
        out.sort(key=lambda item: (not item["current"], item["name"].lower()))
        return {"object": "list", "switch": True, "reason": None, "router": True, "data": out}

    def management_json(self, path):
        """The JSON object a management request (switch, unload, presets) carries. Same cross-origin rule as the dashboard's."""
        origin = self.headers.get("Origin")
        if origin and "*" not in self.server.cors_origins and origin not in self.server.cors_origins \
                and urlsplit(origin).netloc != self.headers.get("Host", ""):
            raise ApiError(403, "Cross-origin model switches are not allowed.", "forbidden")
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            raise ApiError(415, "Send Content-Type: application/json.", "unsupported_media_type")
        try:
            body = json.loads(self.read_body(path))
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "Request body must be valid JSON.")
        if not isinstance(body, dict):
            raise ApiError(400, "Request body must be a JSON object.")
        return body

    def management_request(self, path, body=None):
        """The JSON body and the model of a request that manages models (switch, unload): from `key` (what
        /v1/models/installed gives) or `model` (what a client names it)."""
        if body is None:
            body = self.management_json(path)
        key, name = body.get("key"), body.get("model")
        if isinstance(key, str) and key:
            entry, param = self.server.catalog.by_key(key), "key"
        elif isinstance(name, str) and name:
            entry, param = self.entry_for(name), "model"
        else:
            raise ApiError(400, "`key` (or `model`) must name an installed model.", "invalid_request_error", "key")
        if entry is None:
            raise ApiError(404, "No installed model has that " + param + ".", "model_not_found", param)
        return body, entry, param

    def switch(self):
        """Load a model (the dashboard's Switch button, or a preload with a `keep_alive`). Answers when it is ready,
        or says why not."""
        body, entry, param = self.management_request("/v1/models/switch")
        keep = keep_alive_value(body.get("keep_alive"))
        if entry["reason"] is not None:
            raise ApiError(409, f"That model cannot be started here: {entry['reason']}.", "not_switchable", param)
        pool, key = self.server.pool, entry["key"]
        current = pool.get(key)
        if current is not None and current.state == "ready":
            if keep is UNSET:
                raise ApiError(409, "That model is already loaded.", "already_current", param)
            current.keep_alive = keep                  # already there: only the keep-alive changes
            return self.send_json(200, {"status": "loaded", "key": key, "id": entry["id"], "name": entry["name"]})
        self.server.gate.acquire(key)
        try:
            worker = pool.acquire(entry)
            if keep is not UNSET:
                worker.keep_alive = keep
            pool.release(worker)
        finally:
            self.server.gate.release(key)
        self.send_json(200, {"status": "loaded", "key": key, "id": entry["id"], "name": entry["name"]})

    def unload(self):
        """Stop a model (`coli stop MODEL`, the dashboard's Unload). One that is answering is told to go as soon as
        it has finished (202), so a request is never cut off."""
        _body, entry, _param = self.management_request("/v1/models/unload")
        pool, key = self.server.pool, entry["key"]
        outcome = pool.unload(key, "asked to")
        if outcome == "busy":
            worker = pool.get(key)
            if worker is not None:
                worker.keep_alive = 0
                pool.release_wake()
            return self.send_json(202, {"status": "unloading when idle", "key": key, "id": entry["id"]})
        self.send_json(200, {"status": outcome, "key": key, "id": entry["id"]})

    def disabled(self, entry):
        settings = self.server.settings
        return settings is not None and not settings.enabled(entry)

    def settings_list(self):
        """Every installed model with what is set for it (GET /v1/models/settings): on or off, and where it runs."""
        pool, settings = self.server.pool, self.server.settings
        out = []
        for entry in self.server.catalog.entries():
            worker = pool.get(entry["key"])
            limits = context_limits(entry)
            row = {"id": entry["id"], "key": entry["key"], "name": entry["name"], "enabled": not self.disabled(entry),
                   "gpu": (settings.gpu(entry) if settings is not None else None) or "auto",
                   "context": (settings.context(entry) if settings is not None else None),
                   "context_max": limits[1] if limits else 0,
                   "state": worker.state if worker is not None else "idle"}
            if worker is not None and worker.gpu is not None:
                row["running"] = worker.gpu.public()
            out.append(row)
        out.sort(key=lambda row: row["name"].lower())
        policy = pool.gpu.public() if pool.gpu is not None else None
        return {"object": "list", "data": out, "gpu_policy": policy, "file": settings.path if settings is not None else None}

    def settings_set(self):
        """Turn a model on or off, or pin where it runs (`coli model enable|disable|gpu`). Body: `model` (or `key`), and
        `enabled` (true/false) and/or `gpu` ("auto", "vulkan", "cpu"). Turning a model off unloads it, at once, or when it
        has finished what it is answering."""
        settings = self.server.settings
        if settings is None:
            raise ApiError(404, "This router has no model settings file.", "not_found")
        body, entry, param = self.management_request("/v1/models/settings")
        enabled, gpu, context = body.get("enabled"), body.get("gpu"), body.get("context")
        if enabled is None and gpu is None and context is None:
            raise ApiError(400, "Send `enabled`, `gpu` and/or `context`.", "invalid_request_error", "enabled")
        if context is not None:
            if context in ("auto", "default", ""):
                context = 0
            limits = context_limits(entry)
            if isinstance(context, bool) or not isinstance(context, int):
                raise ApiError(400, "`context` must be a number of tokens, or \"auto\" for the default.", "invalid_request_error", "context")
            if context and limits is None:
                raise ApiError(400, f"`{entry['id']}` has no context window to set.", "invalid_request_error", "context")
            if context and context < settings_mod.CONTEXT_MIN:
                raise ApiError(400, f"`context` must be at least {settings_mod.CONTEXT_MIN} tokens.", "invalid_request_error", "context")
            if context and context > limits[1]:
                raise ApiError(400, f"`{entry['id']}` supports a context window of at most {limits[1]} tokens.", "invalid_request_error", "context")
        if enabled is not None and not isinstance(enabled, bool):
            raise ApiError(400, "`enabled` must be true or false.", "invalid_request_error", "enabled")
        if gpu is not None and gpu not in settings_mod.GPU_CHOICES:
            raise ApiError(400, "`gpu` must be one of " + ", ".join(settings_mod.GPU_CHOICES) + ".",
                           "invalid_request_error", "gpu")
        try:
            before = settings.context(entry)
            record = settings_mod.update(settings.path, entry["id"], enabled=enabled, gpu=gpu, context=context)
        except (settings_mod.SettingsError, OSError) as error:
            raise ApiError(500, f"The model settings file could not be changed: {error}", "settings_file_error")
        settings.reload()
        outcome = None
        changed = context is not None and (context or None) != before          # a loaded worker keeps the window it started with: unload it
        if enabled is False or changed:
            pool, key = self.server.pool, entry["key"]
            outcome = pool.unload(key, "turned off" if enabled is False else "context window changed")
            if outcome == "busy":
                worker = pool.get(key)
                if worker is not None:
                    worker.keep_alive = 0
                    pool.release_wake()
                outcome = "unloading when idle"
        self.send_json(200, {"id": entry["id"], "key": entry["key"], "enabled": record.get("enabled", True),
                             "gpu": record.get("gpu", "auto"), "context": record.get("context"), **({"unload": outcome} if outcome else {})})

    # ---- models fetched and removed through the API (model_installer.py, docs/model-install.md)
    def need_installer(self):
        if self.server.installer is None:
            raise ApiError(404, "This router cannot install or remove models.", "not_found")
        return self.server.installer

    def install_catalog(self):
        """GET /v1/models/catalog: the models that can be fetched, what this machine makes of each, which are here."""
        installer = self.need_installer()
        roots = installer.roots()
        free = None
        if roots:
            import setup_hw
            free = setup_hw.disk_free(installer.install_root() or roots[0])
        folders = [{"path": f["path"], "free_gb": round(f["free_bytes"] / 1e9, 1) if f["free_bytes"] is not None else None,
                    "preferred": f["preferred"]} for f in installer.folders()]
        return {"object": "list", "data": installer.catalog_rows(), "roots": roots, "folders": folders,
                "free_gb": round(free / 1e9, 1) if free is not None else None}

    def install_job(self, model_id):
        job = self.need_installer().job(model_id)
        if job is None:
            raise ApiError(404, "That model has no install on record.", "not_found", "id")
        return job

    def install_error(self, error):
        return ApiError(error.status, error.message, error.code, error.param)

    def install_start(self):
        """Fetch a catalog model: body `id` and `confirm: true` (the caller has shown the size and the licence), and `root`,
        one of the `folders` of GET /v1/models/catalog, to say where (else the first with room). Answers 202 with the job;
        GET /v1/models/installs/{id} follows it."""
        installer = self.need_installer()
        body = self.management_json("/v1/models/install")
        model_id = body.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise ApiError(400, "`id` must name a model of GET /v1/models/catalog.", "invalid_request_error", "id")
        root = body.get("root")
        if root is not None and (not isinstance(root, str) or not root.strip()):
            raise ApiError(400, "`root` must be one of the `folders` of GET /v1/models/catalog.", "invalid_request_error", "root")
        try:
            job = installer.start(model_id, confirm=body.get("confirm"), root=root.strip() if root else None)
        except model_installer.InstallError as error:
            raise self.install_error(error)
        self.send_json(202, job)

    def install_cancel(self):
        installer = self.need_installer()
        body = self.management_json("/v1/models/install/cancel")
        try:
            job = installer.cancel_job(body.get("id") if isinstance(body.get("id"), str) else "")
        except model_installer.InstallError as error:
            raise self.install_error(error)
        self.send_json(202, job)

    def uninstall(self):
        """Remove an installed model's files: body `model` (or `key`) and `confirm: true`. A model that is loaded is refused
        (unload it first); so is anything outside the models folders. Presets that name the model are reported, and stay."""
        installer = self.need_installer()
        body = self.management_json("/v1/models/uninstall")
        if isinstance(body.get("id"), str) and not body.get("key") and not body.get("model"):
            return self.uninstall_addon(installer, body)   # an `id` is a catalog add-on's; a model is named by `key` or `model`
        body, entry, param = self.management_request("/v1/models/uninstall", body)
        worker = self.server.pool.get(entry["key"])
        if worker is not None and worker.state in ("starting", "ready", "stopping"):
            raise ApiError(409, f"`{entry['id']}` is loaded; unload it first (POST /v1/models/unload).", "model_loaded", param)
        catalog = self.server.catalog
        names = catalog.aliases(entry)
        named = [p.name for p in (self.server.presets.all() if self.server.presets is not None else ())
                 if (p.model or "").lower() in names]
        item = {"path": entry.get("path"), "key": entry["key"], "name": entry.get("name"),
                "unfinished": entry.get("unfinished"), "missing": entry.get("missing")}
        try:
            result = installer.uninstall(item, confirm=body.get("confirm"))
        except model_installer.InstallError as error:
            raise self.install_error(error)
        self.send_json(200, {**result, "id": entry["id"], "key": entry["key"], "presets": named})

    def uninstall_addon(self, installer, body):
        """Remove an add-on (the planner, the base DiT): body `id` of its catalog row and `confirm: true`. Its parent model must
        not be loaded, since the engine keeps the files open."""
        try:
            plan = installer.addon_plan(body["id"])
        except model_installer.InstallError as error:
            raise self.install_error(error)
        worker = self.server.pool.get(plan["parent"]["key"])
        if worker is not None and worker.state in ("starting", "ready", "stopping"):
            raise ApiError(409, f"`{plan['parent']['id']}` is loaded; unload it first (POST /v1/models/unload).", "model_loaded", "id")
        try:
            result = installer.uninstall_addon(body["id"], confirm=body.get("confirm"))
        except model_installer.InstallError as error:
            raise self.install_error(error)
        self.send_json(200, {**result, "id": body["id"], "key": None, "presets": []})

    def preset_save(self):
        """Create or replace a preset (the dashboard's Presets tab; `coli preset add` does the same to the file). The body is
        the preset as the file holds it, plus its `name`. Takes effect at once, like an edit of the file."""
        if self.server.presets is None:
            raise ApiError(404, "This router has no presets file.", "not_found")
        body = self.management_json("/v1/presets")
        name = body.pop("name", None)
        if not isinstance(name, str) or not name.strip():
            raise ApiError(400, "`name` must be text.", "invalid_request_error", "name")
        name = name.strip().lower()
        try:
            preset = presets_mod.check(name, body)
        except presets_mod.PresetError as error:
            raise ApiError(400, str(error), "invalid_request_error")
        catalog = self.server.catalog
        if any(name in catalog.aliases(e) for e in catalog.entries()):
            raise ApiError(409, f"`{name}` is the name of an installed model; a preset needs a name of its own.",
                           "name_taken", "name")
        if catalog.find(preset.model) is None:
            raise ApiError(404, f"No installed model is called `{preset.model}`.", "model_not_found", "model")
        try:
            with PRESET_FILE_LOCK:
                replaced = presets_mod.add(self.server.presets.path, name, body)
        except (presets_mod.PresetError, OSError) as error:
            raise ApiError(500, f"The presets file could not be changed: {error}", "presets_file_error")
        self.server.presets.reload()
        self.send_json(200, {"status": "replaced" if replaced else "created", "name": name})

    def mcp_servers_list(self):
        """GET /v1/mcp/servers: the MCP servers the clients run (mcp_registry.py, docs/mcp-client.md). The router only keeps and serves the list; a client
        starts the servers itself, after a person has approved what each one runs."""
        file = mcp_registry_mod.path()
        try:
            servers, problem = mcp_registry_mod.load(file), None
        except mcp_registry_mod.RegistryError as error:
            servers, problem = {}, str(error)
        return {"object": "list", "path": file, "problem": problem,
                "data": [dict(cfg, name=name, summary=mcp_registry_mod.describe(cfg)) for name, cfg in sorted(servers.items())]}

    def mcp_write_allowed(self):
        """The list is edited from this PC, as the owner: anyone who holds the API key can read it, but a change could start a program on every client
        that approves it, so a request from another machine is refused unless COLI_MCP_REMOTE_EDIT=1."""
        peer = self.client_address[0] if self.client_address else ""
        if peer in LOOPBACK or os.environ.get("COLI_MCP_REMOTE_EDIT") == "1":
            return
        raise ApiError(403, "The list of MCP servers is changed from the machine the router runs on (or set COLI_MCP_REMOTE_EDIT=1 there).", "forbidden")

    def mcp_server_save(self):
        """POST /v1/mcp/servers: add or replace one server: {"name", "command" + "args" + "env" ... | "url" + "headers" ...}."""
        self.mcp_write_allowed()
        body = self.management_json("/v1/mcp/servers")
        name = body.pop("name", None)
        try:
            kept, replaced = mcp_registry_mod.put(name, body)
        except mcp_registry_mod.RegistryError as error:
            raise ApiError(400, str(error), "invalid_request_error")
        except OSError as error:
            raise ApiError(500, f"The list of MCP servers could not be changed: {error}", "mcp_servers_error")
        self.send_json(200, {"status": "replaced" if replaced else "created", "name": name, "summary": mcp_registry_mod.describe(kept)})

    def mcp_server_delete(self):
        """POST /v1/mcp/servers/delete: {"name"}."""
        self.mcp_write_allowed()
        body = self.management_json("/v1/mcp/servers/delete")
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ApiError(400, "`name` must be text.", "invalid_request_error", "name")
        try:
            removed = mcp_registry_mod.remove(name.strip())
        except mcp_registry_mod.RegistryError as error:
            raise ApiError(409, str(error), "mcp_servers_error")
        except OSError as error:
            raise ApiError(500, f"The list of MCP servers could not be changed: {error}", "mcp_servers_error")
        if not removed:
            raise ApiError(404, f"No MCP server is called `{name.strip()}`.", "mcp_server_not_found", "name")
        self.send_json(200, {"status": "deleted", "name": name.strip()})

    def voices_list(self):
        """GET /v1/audio/voices: the saved voices of the cloning speech models (voices.py), by name."""
        return {"object": "list", "path": str(voices_mod.directory()), "data": [voices_mod.payload(v) for v in voices_mod.list_voices()]}

    def voice_save(self):
        """POST /v1/audio/voices: save a voice from a reference clip: {"name", "audio" (base64 of an audio file, or a data: URL), "text" (what
        the clip says; without it the voice is the speaker's embedding alone)}. `coli voice add` makes the same files."""
        body = self.management_json("/v1/audio/voices")
        audio, text = body.get("audio"), body.get("text", "")
        if not isinstance(audio, str) or not audio:
            raise ApiError(400, "`audio` must be the base64 of an audio file (or a data: URL).", "invalid_request_error", "audio")
        if not isinstance(text, str):
            raise ApiError(400, "`text` must be text: what the clip says.", "invalid_request_error", "text")
        import base64
        try:
            raw = base64.b64decode(audio.split(",", 1)[1] if audio.startswith("data:") and "," in audio else audio, validate=True)
        except Exception:
            raise ApiError(400, "`audio` is not valid base64.", "invalid_request_error", "audio")
        try:
            info = voices_mod.add(body.get("name"), raw, text)
        except voices_mod.VoiceError as error:
            raise ApiError(400, str(error), "invalid_request_error", error.param)
        except OSError as error:
            raise ApiError(500, f"The voice could not be saved: {error}", "voices_error")
        self.send_json(200, {"status": "replaced" if info.pop("replaced") else "created", **voices_mod.payload(info)})

    def voice_delete(self):
        """POST /v1/audio/voices/delete: forget a saved voice."""
        body = self.management_json("/v1/audio/voices/delete")
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ApiError(400, "`name` must be text.", "invalid_request_error", "name")
        if not voices_mod.delete(name):
            raise ApiError(404, f"No voice is saved as `{name.strip()}`.", "voice_not_found", "name")
        self.send_json(200, {"status": "deleted", "name": name.strip().lower()})

    def preset_delete(self):
        """Remove a preset from the file."""
        if self.server.presets is None:
            raise ApiError(404, "This router has no presets file.", "not_found")
        body = self.management_json("/v1/presets/delete")
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ApiError(400, "`name` must be text.", "invalid_request_error", "name")
        try:
            with PRESET_FILE_LOCK:
                removed = presets_mod.remove(self.server.presets.path, name.strip())
        except (presets_mod.PresetError, OSError) as error:
            raise ApiError(500, f"The presets file could not be changed: {error}", "presets_file_error")
        if not removed:
            raise ApiError(404, f"No preset is called `{name.strip()}`.", "preset_not_found", "name")
        self.server.presets.reload()
        self.send_json(200, {"status": "deleted", "name": name.strip().lower()})


def context_limits(entry):
    """(the variable the model's engine reads its context window from, the most tokens its family supports), or None for a model with no context
    window to set (a picture, music or speech model) or a family this registry does not know."""
    try:
        from family_registry import family_by_id
        limits = family_by_id(entry["family"]).limits
    except Exception:
        return None
    return (limits.context_env, limits.max_context) if limits.max_context > 1 and limits.context_env else None


def context_window(settings, entry):
    """(environment variable, tokens) for a model that has its own context window in the settings, else None. A window over the family's limit is
    cut to it."""
    wanted = settings.context(entry) if settings is not None else None
    limits = context_limits(entry) if wanted else None
    if not limits:
        return None
    return limits[0], min(wanted, limits[1])


# ----------------------------------------------------------------------------- entry point
def worker_kv_slots(entry, wanted):
    """The KV slots to start a model's worker with: what was asked for, but no more than its family supports (a
    picture or music model has one), and at least one."""
    try:
        from family_registry import family_by_id
        return max(1, min(wanted, family_by_id(entry["family"]).limits.max_kv_slots))
    except Exception:
        return 1


def serve(*, host="127.0.0.1", port=8000, api_key=None, roots=None, models=(), keep_alive=DEFAULT_KEEP_ALIVE,
          max_loaded=DEFAULT_MAX_LOADED, max_queue=8, queue_timeout=300.0, parallel_models=False,
          cors_origins=None, allowed_hosts=(), worker_args=(), preload=(), engine_ready=None, kv_slots=1,
          presets_path=None, request_log_path=None, jobs_path=None, gpu=None, gpu_reserve_gb=gpu_mod.DEFAULT_RESERVE_GB,
          gpu_min_gb=gpu_mod.DEFAULT_MIN_GB, settings_path=None):
    """Run the router until it is shut down."""
    import openai_server
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        if os.environ.get("COLI_ALLOW_INSECURE_BIND") == "1":
            print("WARNING: binding %s beyond localhost with NO auth (COLI_ALLOW_INSECURE_BIND=1)" % host,
                  file=sys.stderr)
        else:
            print("refusing to bind %s beyond localhost without COLI_API_KEY set "
                  "(set COLI_ALLOW_INSECURE_BIND=1 to override)" % host, file=sys.stderr)
            sys.exit(1)
    if allowed_hosts and "*" in allowed_hosts:
        print("WARNING: --allowed-host '*' accepts ANY Host header (DNS-rebinding guard disabled)", file=sys.stderr)
    catalog = Catalog(roots=roots, models=models, known_path=os.path.join(os.path.dirname(default_state_path()), "known-models.json"),
                      engine_ready=engine_ready or (lambda family: openai_server.default_engine(family).exists()),
                      worker_kv_slots=1)          # every family can run one slot; kv_slots is clamped per model below
    extra = list(worker_args)

    def command(entry, free):
        launcher = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coli")
        return [sys.executable, launcher, "serve", "--model", entry["path"], "--host", "127.0.0.1",
                "--port", str(free), "--kv-slots", str(worker_kv_slots(entry, kv_slots))] + extra

    policy = gpu_mod.GpuPolicy(gpu, gpu_reserve_gb, gpu_min_gb) if gpu else None
    settings = settings_mod.ModelSettings(settings_path or settings_mod.default_path(), log=log)
    pool = Pool(catalog, keep_alive=keep_alive, max_loaded=max_loaded, worker_command=command,
                state=State(default_state_path()), queue_timeout=queue_timeout,
                make_job=openai_server._win_kill_on_close_job, gpu=policy, settings=settings,
                pressure_floor=pressure_floor(), card_evict=os.environ.get("COLI_CARD_EVICT", "1").strip().lower() not in ("0", "false", "no", "off"),
                card_demote=os.environ.get("COLI_CARD_DEMOTE", "1").strip().lower() not in ("0", "false", "no", "off"),
                card_promote=os.environ.get("COLI_CARD_PROMOTE", "1").strip().lower() not in ("0", "false", "no", "off"),
                promote_chat=os.environ.get("COLI_CARD_PROMOTE_CHAT", "1").strip().lower() not in ("0", "false", "no", "off"))
    gate = Gate(max_queue=max_queue, timeout=queue_timeout, parallel=parallel_models)
    presets = presets_mod.Presets(presets_path or presets_mod.default_path(), log=log)
    if request_log_path:
        os.environ["COLI_REQUEST_LOG"] = request_log_path          # the workers inherit it and write their own lines
    logger = request_log.from_env()
    job_manager = jobs.JobManager(jobs.JobStore(jobs_path or default_jobs_path()), on_finish=job_logger(logger))
    def installed_models():
        """What is installed, each with the state of its worker: what the installer matches the catalog against."""
        out = []
        for found in catalog.entries(refresh=True):
            worker = pool.get(found["key"])
            found["state"] = worker.state if worker is not None else "idle"
            out.append(found)
        return out

    installer = model_installer.Installer(
        roots=catalog.install_roots, installed=installed_models, refresh=catalog.invalidate,
        # an engine reads what an add-on adds (the music planner, a second DiT) when it starts: an idle one is stopped so that the next request starts it again
        on_addon=lambda parent: parent and pool.unload(parent["key"], "an add-on was installed"))
    installer.sweep()                                    # a deletion that was cut short by a restart
    server = RouterServer((host, port), pool, catalog, gate, api_key,
                          openai_server.cors_origin_list(cors_origins), allowed_hosts, queue_timeout,
                          presets=presets, request_log=logger, settings=settings, jobs=job_manager, installer=installer)
    entries = catalog.entries(refresh=True)
    usable = [e for e in entries if e["reason"] is None]
    print(f"[router] {len(usable)} model(s) available: " + ", ".join(e["id"] for e in usable), file=sys.stderr)
    for e in entries:
        if e["reason"] is not None:
            print(f"[router]   {e['id']}: not available ({e['reason']})", file=sys.stderr)
    print(f"[router] keep-alive {'forever' if keep_alive is None else '%d s' % keep_alive}, up to "
          f"{max_loaded or 'any number of'} loaded, "
          f"{'models may compute together' if parallel_models else 'one model computes at a time'}",
          file=sys.stderr)
    if policy is not None:
        seen = policy.memory()
        print(f"[router] graphics card: --gpu {policy.mode}, {policy.reserve_gb:g} GB kept back, {policy.min_gb:g} GB needed"
              + (f"; {seen[0]:.1f} of {seen[1]:.1f} GB free now" if seen else "; no reading of its free memory (nvidia-smi)"),
              file=sys.stderr)
    switched = settings.all()
    if switched:
        def words(record):
            return ", ".join(w for w in ("off" if record.get("enabled") is False else "", record.get("gpu", "")) if w)
        print(f"[router] model settings from {settings.path}: "
              + "; ".join(f"{model}: {words(record)}" for model, record in switched.items()), file=sys.stderr)
    named = presets.all()
    if named or os.path.exists(presets.path):
        print(f"[router] {len(named)} preset(s) from {presets.path}"
              + (": " + ", ".join(p.name for p in named) if named else ""), file=sys.stderr)
    if logger is not None:
        print(f"[router] logging requests to {logger.path}" + (" (with their bodies)" if logger.bodies else ""),
              file=sys.stderr)
    print(f"OpenAI-compatible API listening on http://{host}:{port}/v1", file=sys.stderr)

    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, stop)

    def warm():
        for name in preload:
            entry = catalog.find(name)
            if entry is None or entry["reason"] is not None:
                log(f"cannot preload `{name}`")
                continue
            try:
                gate.acquire(entry["key"])
                try:
                    pool.release(pool.acquire(entry))
                finally:
                    gate.release(entry["key"])
            except ApiError as error:
                log(f"preload failed: {error.message}")

    if preload:
        threading.Thread(target=warm, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous)
        server.server_close()
        job_manager.close()
        pool.close()
    return None
