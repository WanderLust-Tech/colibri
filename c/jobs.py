"""Jobs: a picture or a song that keeps being made after the connection that asked for it has gone.

A picture can take an hour. Asked for the usual way it is one request, so closing the tab, a browser that put the page to sleep or
an editor that reloaded cancels it, and the hour is lost. Asked for with `Prefer: respond-async` the router answers at once with a
job id and runs the request apart from the connection; any client can come back, by id, and ask where it is, and cancel it.

This module is only the lane: the queue, the states, the records on disk and the cancel. What a job does is a function the
router hands in (`run`), so none of it needs a model or a socket to be tested.

  queued -> running -> succeeded | failed | cancelled          (queued -> cancelled too)

One job runs at a time: a picture or a song takes the whole machine, and waiting jobs cost nothing but a record. Every change of
state is appended to a JSON-lines file (like the request log), so the list survives a restart. A job that was waiting or running
when the router stopped is not coming back (its model went with it): it is read back as failed and says why.
"""
import collections
import json
import os
import re
import threading
import time
import uuid

FINAL = ("succeeded", "failed", "cancelled")
STATUSES = ("queued", "running") + FINAL
KEEP_JOBS = 200                      # records kept in memory and, after a restart, on disk
MAX_PENDING = 32                     # queued and running jobs at once; one more is refused
ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ID_HEADER = "X-Colibri-Job-Id"         # a request may name its own job; the same name again is the same job


class JobError(Exception):
    """A job that cannot be made, or one that failed; `code` is the machine-readable word."""

    def __init__(self, status, message, code=None, param=None):
        super().__init__(message)
        self.status, self.message, self.code, self.param = status, message, code, param


class Cancelled(Exception):
    """Raised by a run function that noticed it was cancelled."""


def page_arguments(query):
    """(status, limit, offset) from a parsed query string; JobError 400 for anything that is not a known status or a number in range."""
    def number(name, default, high):
        raw = (query.get(name) or [None])[0]
        if raw is None or raw == "":
            return default
        if not re.fullmatch(r"\d{1,9}", raw) or not (1 if name == "limit" else 0) <= int(raw) <= high:
            raise JobError(400, f"`{name}` must be a whole number from {1 if name == 'limit' else 0} to {high}.", "invalid_value", name)
        return int(raw)
    status = (query.get("status") or [None])[0] or None
    if status is not None and status not in STATUSES:
        raise JobError(400, f"`status` must be one of: {', '.join(STATUSES)}.", "invalid_value", "status")
    return status, number("limit", 50, 200), number("offset", 0, 10 ** 9)


class Context:
    """What a run function sees of its job: whether it was cancelled, a place to report progress, and a way to be interrupted."""

    def __init__(self, manager, job_id):
        self._manager, self.id = manager, job_id
        self.cancelled = threading.Event()
        self._abort = None
        self._lock = threading.Lock()

    def progress(self, data):
        self._manager._set_progress(self.id, data)

    def on_abort(self, function):
        """Register what stops the work right now (closing the connection to the model). It is called at once when the job is
        cancelled, or straight away if it already was."""
        with self._lock:
            self._abort = function
            already = self.cancelled.is_set()
        if already:
            self._call(function)

    def cancel(self):
        self.cancelled.set()
        with self._lock:
            function = self._abort
        if function is not None:
            self._call(function)

    @staticmethod
    def _call(function):
        try:
            function()
        except Exception:
            pass


class JobStore:
    """The records on disk: one JSON object per line, a job appearing once for each change of state; the last line wins."""

    def __init__(self, path, keep=KEEP_JOBS):
        self.path, self.keep = path, keep
        self.lock = threading.Lock()
        self.failed = False

    def write(self, record):
        """Append one record. Never raises: a disk that cannot be written must not stop a job that is running."""
        line = {k: v for k, v in record.items() if k != "progress"}
        try:
            text = json.dumps(line, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
        except (TypeError, ValueError):
            return
        with self.lock:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(text)
                self.failed = False
            except OSError as error:
                if not self.failed:
                    self.failed = True
                    import sys
                    sys.stderr.write(f"[jobs] cannot write {self.path}: {error}\n")

    def load(self):
        """The newest `keep` jobs, oldest first, each as its last record says. The file is rewritten to just those when it
        has grown well past that, so it does not grow for ever."""
        latest, lines = {}, 0
        try:
            with open(self.path, encoding="utf-8") as handle:
                for text in handle:
                    lines += 1
                    try:
                        record = json.loads(text)
                    except ValueError:
                        continue
                    if isinstance(record, dict) and isinstance(record.get("id"), str) and record.get("status") in STATUSES:
                        latest[record["id"]] = record
        except OSError:
            return []
        records = sorted(latest.values(), key=lambda r: r.get("created") or 0)[-self.keep:]
        if lines > self.keep * 4:
            self._rewrite(records)
        return records

    def _rewrite(self, records):
        temporary = self.path + f".{os.getpid()}.tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
            os.replace(temporary, self.path)
        except OSError:
            pass


class JobManager:
    def __init__(self, store=None, on_finish=None, max_pending=MAX_PENDING, keep=KEEP_JOBS, clock=time.time):
        self.store, self.on_finish, self.max_pending, self.keep, self.clock = store, on_finish, max_pending, keep, clock
        self.cv = threading.Condition()
        self.jobs = collections.OrderedDict()           # id -> record, oldest first
        self.queue = collections.deque()                # ids waiting, in order
        self.work = {}                                  # id -> {"run": callable, "context": Context} for queued and running jobs
        self.running = None
        self.closed = False
        for record in (store.load() if store else []):
            if record["status"] in ("queued", "running"):
                was = record["status"]
                record.update(status="failed", finished=self.clock(), error={
                    "code": "router_restarted",
                    "message": f"The router stopped while this job was {'waiting' if was == 'queued' else 'running'}; its model went with it."})
                store.write(record)
            self.jobs[record["id"]] = record
        self.thread = threading.Thread(target=self._loop, name="jobs", daemon=True)
        self.thread.start()

    # ---- asking
    def submit(self, kind, model, path, request, run, job_id=None):
        """Queue a job. Returns (snapshot, created): a job id already held answers with that job and created False, so a
        client that sent the same request twice (a retry after a dropped answer) does not make two renders."""
        if job_id is not None and not ID.match(job_id):
            raise JobError(400, "The job id must be 1 to 64 letters, digits, `_` or `-`.", "invalid_value", "job_id")
        with self.cv:
            if job_id is not None and job_id in self.jobs:
                return self._snapshot(self.jobs[job_id]), False
            if self.closed:
                raise JobError(503, "The router is shutting down.", "shutting_down")
            if len(self.queue) + (1 if self.running else 0) >= self.max_pending:
                raise JobError(429, f"{self.max_pending} jobs are already waiting or running; ask again when one has finished.",
                               "queue_full")
            job_id = job_id or "job_" + uuid.uuid4().hex[:20]
            record = {"id": job_id, "kind": kind, "model": model, "path": path, "status": "queued", "created": self.clock(),
                      "started": None, "finished": None, "request": request, "progress": None, "result": None, "error": None}
            self.jobs[job_id] = record
            self.work[job_id] = {"run": run, "context": Context(self, job_id)}
            self.queue.append(job_id)
            self._trim()
            self._save(record)
            self.cv.notify_all()
            return self._snapshot(record), True

    def get(self, job_id):
        with self.cv:
            record = self.jobs.get(job_id)
            return self._snapshot(record) if record else None

    def listing(self, status=None, limit=50, offset=0):
        with self.cv:
            rows = [r for r in reversed(self.jobs.values()) if status is None or r["status"] == status]
            page = rows[offset:offset + limit]
            return {"object": "list", "total": len(rows), "has_more": offset + len(page) < len(rows),
                    "data": [self._snapshot(r) for r in page]}

    def cancel(self, job_id):
        """Stop a job. The same call whatever state the job is in: waiting is removed, running is interrupted, and a job that
        is finished, or that nobody has heard of, answers `cancelled` False and is not an error. Returns (cancelled, status)."""
        with self.cv:
            record = self.jobs.get(job_id)
            if record is None:
                return False, None
            status = record["status"]
            if status in FINAL:
                return False, status
            if status == "queued":
                self.queue.remove(job_id)
                self.work.pop(job_id, None)
                record.update(status="cancelled", finished=self.clock())
                self._save(record)
                self.cv.notify_all()
                callback = self.on_finish
                snapshot = self._snapshot(record)
            else:
                context = self.work[job_id]["context"]
                callback = snapshot = None
        if status == "queued":
            self._finished(callback, snapshot)
            return True, "cancelled"
        context.cancel()                  # outside the lock: it closes a socket
        return True, "cancelling"

    # ---- the lane
    def _loop(self):
        while True:
            with self.cv:
                while not self.queue and not self.closed:
                    self.cv.wait()
                if self.closed:
                    return
                job_id = self.queue.popleft()
                record, work = self.jobs[job_id], self.work[job_id]
                record.update(status="running", started=self.clock())
                self.running = job_id
                self._save(record)
            context = work["context"]
            result = error = None
            try:
                result = work["run"](context)
            except Cancelled:
                pass
            except JobError as failure:
                error = {"code": failure.code, "message": failure.message}
            except Exception as failure:                  # a bug in the run function must not end the lane
                error = {"code": "job_error", "message": f"The job failed: {failure}"}
            with self.cv:
                if context.cancelled.is_set():
                    status = "cancelled"
                elif error:
                    status = "failed"
                else:
                    status = "succeeded"
                record.update(status=status, finished=self.clock(), progress=None, result=result if status != "failed" else None,
                              error=error if status == "failed" else None)
                self.work.pop(job_id, None)               # the request body held for the run goes with it
                self.running = None
                self._save(record)
                snapshot = self._snapshot(record)
                self.cv.notify_all()
            self._finished(self.on_finish, snapshot)

    def _finished(self, callback, snapshot):
        if callback is None or snapshot is None:
            return
        try:
            callback(snapshot)
        except Exception:
            pass

    def _set_progress(self, job_id, data):
        with self.cv:
            record = self.jobs.get(job_id)
            if record is not None and record["status"] == "running":
                record["progress"] = data

    def close(self, timeout=5.0):
        with self.cv:
            self.closed = True
            contexts = [w["context"] for w in self.work.values()]
            self.cv.notify_all()
        for context in contexts:
            context.cancel()
        self.thread.join(timeout)

    # ---- records
    def _save(self, record):
        if self.store is not None:
            self.store.write(record)

    def _trim(self):
        """Forget the oldest finished jobs past `keep`; one that is not finished is never forgotten."""
        extra = len(self.jobs) - self.keep
        for job_id in list(self.jobs):
            if extra <= 0:
                break
            if self.jobs[job_id]["status"] in FINAL:
                del self.jobs[job_id]
                extra -= 1

    def _snapshot(self, record):
        """The record as a client sees it. Call with the lock held."""
        started, finished = record.get("started"), record.get("finished")
        out = {"id": record["id"], "object": "job", "kind": record.get("kind"), "model": record.get("model"),
               "status": record["status"], "created": int(record.get("created") or 0),
               "started": int(started) if started else None, "finished": int(finished) if finished else None,
               "endpoint": record.get("path"), "request": record.get("request") or {}}
        if record["status"] == "queued":
            out["position"] = list(self.queue).index(record["id"]) + 1 if record["id"] in self.queue else None
        if record["status"] == "running" and record.get("progress"):
            out["progress"] = dict(record["progress"])
        if record.get("result"):
            out["result"] = record["result"]
        if record.get("error"):
            out["error"] = record["error"]
        return out
