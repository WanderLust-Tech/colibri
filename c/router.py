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

DEFAULT_KEEP_ALIVE = 600.0          # seconds a model stays loaded after its last request
DEFAULT_MAX_LOADED = 3
FIT_HEADROOM = 0.8                  # a model must fit in this share of the free memory (Ollama's rule)
STOP_GRACE = 25.0                   # seconds a worker gets to save its state before it is killed
LOOPBACK = {"127.0.0.1", "localhost", "::1", ""}

# What a worker is asked to compute, and the biggest body each may carry. The workers enforce
# their own, tighter limits; these only bound what the router reads into memory.
COMPUTE_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/systemone",
                 "/v1/images/generations", "/v1/images/edits", "/v1/audio/music", "/v1/audio/transcriptions")

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "content-length", "host", "expect"}
DROP_REQUEST = HOP_BY_HOP | {"authorization", "x-api-key", "origin", "referer", "accept-encoding"}
DROP_RESPONSE = HOP_BY_HOP | {"server", "date", "vary"}


def log(message):
    sys.stderr.write("[router] %s\n" % message)
    sys.stderr.flush()


class ApiError(Exception):
    """An error the router answers with, in the shape the workers use."""

    def __init__(self, status, message, code=None, param=None, error_type="invalid_request_error", headers=None):
        super().__init__(message)
        self.status, self.message, self.code, self.param = status, message, code, param
        self.error_type, self.headers = error_type, headers or {}

    def body(self):
        return {"error": {"message": self.message, "type": self.error_type, "param": self.param, "code": self.code}}


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


def default_state_path():
    try:
        import setup_flow
        return os.path.join(setup_flow.setup_home(), "router.json")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".colibri-router.json")


# ------------------------------------------------------------------------------ catalog
class Catalog:
    """The models on this machine, by the names clients use for them."""

    def __init__(self, roots=None, models=(), engine_ready=None, worker_kv_slots=1, ttl=10.0, entries=None):
        self.roots, self.models = roots, tuple(models)
        self.engine_ready = engine_ready or (lambda family: True)
        self.kv_slots, self.ttl = worker_kv_slots, ttl
        self._fixed = entries                  # tests give the entries directly
        self._entries, self._at = [], 0.0
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
        return entries

    def entries(self, refresh=False):
        with self.lock:
            age = time.monotonic() - self._at
            if not self._entries or age > self.ttl or (refresh and age > 2.0):
                self._entries, self._at = self._scan(), time.monotonic()
            return [dict(entry) for entry in self._entries]

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
            for entry in self.entries(refresh):
                if wanted in self.aliases(entry):
                    return entry
        return None

    def by_key(self, key):
        for entry in self.entries():
            if entry["key"] == key:
                return entry
        return None


# ------------------------------------------------------------------------------- worker
def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


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
                 grace=STOP_GRACE, make_job=None, queue_timeout=300.0):
        self.catalog = catalog
        self.keep_alive, self.max_loaded = keep_alive, max_loaded
        self.worker_command = worker_command
        self.worker_env = worker_env
        self.state = state or State()
        self.memory = memory or available_memory         # bytes free for a new model, or None
        self.measure = measure or resident_bytes
        self.grace, self.make_job, self.queue_timeout = grace, make_job, queue_timeout
        self.workers = {}
        self.cv = threading.Condition(threading.Lock())
        self.closed = False
        self.last_error = {}                             # key -> why the last load failed
        self._wake = threading.Event()                   # a model that should go as soon as it is idle
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()

    # ---- sizes
    def estimate(self, entry):
        """What loading this model will take: what it took last time, else its size on disk."""
        known = self.state.resident(entry["key"])
        if known:
            return known
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
            if worker.state in ("starting", "stopping"):
                return "wait", worker
            del self.workers[key]                        # failed, or died while ready
        others = list(self.workers.values())
        counted = [w for w in others if w.state in ("starting", "ready", "stopping")]
        room = self.max_loaded <= 0 or len(counted) < self.max_loaded
        if room and self.fits(entry, free):
            return "load", self._register(entry)
        victims = [w for w in others if w.state == "ready" and w.inflight == 0]
        if victims:
            victim = min(victims, key=lambda w: w.last_used)
            victim.state = "stopping"
            victim.settled.clear()
            return "evict", victim
        if not others:                                   # nothing left to free: try anyway
            log(f"`{entry['id']}` may not fit in memory (needs about "
                f"{self.estimate(entry) / 2**30:.1f} GB), loading it anyway")
            return "load", self._register(entry)
        return "block", None                             # others are busy or loading: wait for them

    def _register(self, entry):
        worker = Worker(entry, self.keep_alive)
        self.workers[entry["key"]] = worker
        return worker

    def acquire(self, entry):
        """A ready worker for `entry` with one request counted against it (release it when done).
        Loads the model if needed, after making room. Waiting for room is bounded by the queue
        timeout; the load itself is not."""
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

    def _load(self, worker):
        entry = worker.entry
        log(f"loading `{entry['id']}`")
        began = time.monotonic()
        try:
            command, env = self._command(entry)
            worker.spawn(command, env, self.make_job)
            worker.wait_ready()
        except BaseException as error:
            api = error if isinstance(error, ApiError) else ApiError(
                503, f"`{entry['id']}` did not start: {error}", "model_load_failed", None, "server_error")
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
            raise api
        worker.resident = self.measure(worker.proc.pid, worker.job)
        self.state.record(worker.key, entry["id"], worker.resident, worker.model_entry, worker.jev)
        with self.cv:
            worker.state = "ready"
            worker.inflight += 1
            worker.last_used = time.monotonic()
            self.last_error.pop(worker.key, None)
            worker.settled.set()
            self.cv.notify_all()
        log(f"`{entry['id']}` ready in {time.monotonic() - began:.0f} s, "
            f"{worker.resident / 2**30:.1f} GB resident, port {worker.port}")
        return worker

    def _command(self, entry):
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
        return command, env

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
    asked for continuously cannot starve the others. Bounded like a worker's own queue."""

    def __init__(self, max_queue=8, timeout=300.0, parallel=False):
        self.max_queue, self.timeout, self.parallel = max_queue, timeout, parallel
        self.cv = threading.Condition(threading.Lock())
        self.active = collections.Counter()
        self.waiting = collections.deque()

    def _may_run(self, ticket, key):
        if any(k != key and n > 0 for k, n in self.active.items()):
            return False
        for other, other_key in self.waiting:
            if other is ticket:
                return True
            if other_key != key:
                return False
        return True

    def acquire(self, key):
        if self.parallel:
            with self.cv:
                self.active[key] += 1
            return
        ticket = object()
        deadline = time.monotonic() + self.timeout
        with self.cv:
            if self._may_run(ticket, key):            # nobody else computing, nobody ahead of us for another model
                self.active[key] += 1
                return
            if len(self.waiting) >= self.max_queue:
                raise ApiError(429, "The inference queue is full.", "queue_full", None, "rate_limit_error",
                               {"Retry-After": "1"})
            self.waiting.append((ticket, key))
            try:
                while not self._may_run(ticket, key):
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise ApiError(429, "Timed out waiting for the inference engine.", "queue_timeout", None,
                                       "rate_limit_error", {"Retry-After": "1"})
                    self.cv.wait(left)
                self.active[key] += 1
            finally:
                self.waiting.remove((ticket, key))
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


def _multipart_without(raw, name, content_type):
    """The multipart body without the field `name` (its whole part), or the body itself when it has none."""
    at = raw.find(b'name="%s"' % name.encode())
    if at < 0:
        return raw
    boundary = ""
    for piece in (content_type or "").split(";")[1:]:
        key, _, value = piece.strip().partition("=")
        if key.lower() == "boundary":
            boundary = value.strip('"')
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

    def for_worker(self, model_id):
        """The body to send on: `model` set to the id the worker knows it by (a client may name it by its key, its
        folder or the id without `-colibri`; a worker accepts only its own id) and without `keep_alive`, which is
        the router's. The same bytes when there is nothing to change."""
        if self.multipart:
            raw = self.raw
            at = raw.find(b'name="model"')
            start = raw.find(b"\r\n\r\n", at) if at >= 0 else -1
            if start >= 0:
                end = raw.find(b"\r\n", start + 4)
                end = len(raw) if end < 0 else end
                if raw[start + 4:end] != model_id.encode():
                    raw = raw[:start + 4] + model_id.encode() + raw[end:]
            return _multipart_without(raw, "keep_alive", self.content_type)
        if self.json is None or (self.json.get("model") == model_id and "keep_alive" not in self.json):
            return self.raw
        body = {k: v for k, v in self.json.items() if k != "keep_alive"}
        body["model"] = model_id
        return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


def extract_model(content_type, raw):
    """The `model` of a JSON or multipart/form-data request body, or None."""
    return Body(content_type, raw).model


def with_model(content_type, raw, model_id):
    """The body to send to the worker that holds `model_id` (see Body.for_worker)."""
    return Body(content_type, raw).for_worker(model_id)


class RouterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, pool, catalog, gate, api_key=None, cors_origins=(), allowed_hosts=(),
                 queue_timeout=300.0):
        super().__init__(address, RouterHandler)
        self.pool, self.catalog, self.gate = pool, catalog, gate
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

    def _cors(self):
        origin = self.headers.get("Origin")
        origins = self.server.cors_origins
        if not origin or ("*" not in origins and origin not in origins):
            return
        self.send_header("Access-Control-Allow-Origin", "*" if "*" in origins else origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key, anthropic-version, "
                         "X-TypeSafe-SDK, X-TypeSafe-Runtime, X-TypeSafe-Retry-Count")
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

    def fail(self, error):
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
            if path == "/v1/models/installed":
                return self.send_json(200, self.installed())
            if path.startswith("/v1/models/"):
                return self.send_json(200, self.model_object(unquote(path[len("/v1/models/"):])))
            raise ApiError(404, "Not found.", "not_found")
        except ApiError as error:
            self.fail(error)
        except ConnectionError:
            pass

    # ---- POST
    def do_POST(self):
        try:
            self._check_host()
            self._require_auth()
            path = urlsplit(self.path).path
            if path == "/v1/models/switch":
                return self.switch()
            if path == "/v1/models/unload":
                return self.unload()
            if path not in COMPUTE_PATHS:
                raise ApiError(404, "Not found.", "not_found")
            body = Body(self.headers.get("Content-Type"), self.read_body(path))
            keep = body.keep_alive                  # read before anything loads: a bad value must not cost a load
            entry = self.resolve(path, body.model)
            self.serve_model(entry, path, body.for_worker(entry["id"]), keep)
        except ApiError as error:
            self.fail(error)
        except (ConnectionError, TimeoutError):
            pass
        except Exception as error:
            self.log_error("request failed: %s", error)
            self.fail(ApiError(500, "The colibri router failed to process the request.", "router_error",
                               None, "server_error"))

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
        elif path == openai_server.TRANSCRIBE_ENDPOINT:
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

    def resolve(self, path, model):
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
        return entry

    def serve_model(self, entry, path, raw, keep=UNSET):
        gate, pool = self.server.gate, self.server.pool
        gate.acquire(entry["key"])
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
        return {name: value for name, value in self.headers.items() if name.lower() not in DROP_REQUEST}

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
        entry = catalog.find(wanted) if wanted else None
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
        payload["router"] = {
            "loaded": [self.worker_view(w) for w in pool.snapshot()],
            "max_loaded": pool.max_loaded,
            "keep_alive": pool.keep_alive,
            "parallel_models": self.server.gate.parallel,
            "gate": self.server.gate.stats(),
            "free_memory": pool.memory(),
        }
        return payload

    def worker_view(self, worker):
        left = None
        if worker.state == "ready" and worker.keep_alive is not None and worker.inflight == 0:
            left = max(0.0, worker.keep_alive - (time.monotonic() - worker.last_used))
        return {"id": worker.entry["id"], "key": worker.key, "state": worker.state, "port": worker.port,
                "resident_bytes": worker.resident, "inflight": worker.inflight, "expires_in": left,
                "keep_alive": worker.keep_alive}

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
            data.append(item)
            if card:
                cards.append(card)
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
        else:
            item["capabilities"] = ["chat", "systemone"]
        item["input_modalities"] = ["text"]
        return item

    def model_object(self, name):
        entry = self.server.catalog.find(name)
        if entry is None:
            raise ApiError(404, f"The model `{name}` does not exist.", "model_not_found", "model")
        worker = self.server.pool.get(entry["key"])
        if worker is not None and worker.state == "ready" and worker.model_entry:
            return dict(worker.model_entry, state="loaded")
        cached, _card = self.server.pool.state.cached_entry(entry["key"])
        return dict(cached or self.stub(entry), state="idle")

    def installed(self):
        pool = self.server.pool
        out = []
        for entry in self.server.catalog.entries():
            worker = pool.get(entry["key"])
            state = worker.state if worker is not None else "idle"
            item = {k: v for k, v in entry.items() if k != "path"}
            item["current"] = state == "ready"
            item["state"] = state
            item["switchable"] = entry["reason"] is None and state not in ("starting", "stopping")
            if worker is not None:
                item.update(self.worker_view(worker))
            elif entry["key"] in pool.last_error:
                item["last_error"] = pool.last_error[entry["key"]]
            out.append(item)
        out.sort(key=lambda item: (not item["current"], item["name"].lower()))
        return {"object": "list", "switch": True, "reason": None, "router": True, "data": out}

    def management_request(self, path):
        """The JSON body and the model of a request that manages models (switch, unload): from `key` (what
        /v1/models/installed gives) or `model` (what a client names it). Same cross-origin rule as the dashboard's."""
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
        key, name = body.get("key"), body.get("model")
        if isinstance(key, str) and key:
            entry, param = self.server.catalog.by_key(key), "key"
        elif isinstance(name, str) and name:
            entry, param = self.server.catalog.find(name), "model"
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
          cors_origins=None, allowed_hosts=(), worker_args=(), preload=(), engine_ready=None, kv_slots=1):
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
    catalog = Catalog(roots=roots, models=models,
                      engine_ready=engine_ready or (lambda family: openai_server.default_engine(family).exists()),
                      worker_kv_slots=1)          # every family can run one slot; kv_slots is clamped per model below
    extra = list(worker_args)

    def command(entry, free):
        launcher = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coli")
        return [sys.executable, launcher, "serve", "--model", entry["path"], "--host", "127.0.0.1",
                "--port", str(free), "--kv-slots", str(worker_kv_slots(entry, kv_slots))] + extra

    pool = Pool(catalog, keep_alive=keep_alive, max_loaded=max_loaded, worker_command=command,
                state=State(default_state_path()), queue_timeout=queue_timeout,
                make_job=openai_server._win_kill_on_close_job)
    gate = Gate(max_queue=max_queue, timeout=queue_timeout, parallel=parallel_models)
    server = RouterServer((host, port), pool, catalog, gate, api_key,
                          openai_server.cors_origin_list(cors_origins), allowed_hosts, queue_timeout)
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
        pool.close()
    return None
