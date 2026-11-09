"""A log of every request the server answers: one JSON object per line (docs/request-log.md).

Off unless asked for: `COLI_REQUEST_LOG=FILE` (or `coli serve --request-log FILE`, `coli router --request-log FILE`).
The router and the gateway behind it write the same file; a line says which wrote it (`source`) and, for the worker's, the
router's `trace` id, so a request can be followed from the door to the model. A line holds what happened, not what was
said: the endpoint, the model, the status, the time taken and the tokens. The text of the request goes in only with
`COLI_REQUEST_LOG_BODIES=1`, long strings cut short, because a log of prompts is a privacy decision.

The file is appended to and, past COLI_REQUEST_LOG_MAX bytes (default 50 MB), moved to FILE.1 (one old file kept).
"""
import json
import os
import threading
import time

DEFAULT_MAX_BYTES = 50 << 20
STRING_LIMIT = 500


def shorten(value, limit=STRING_LIMIT, depth=0):
    """`value` with every string over `limit` characters cut (a picture or a song sent as base64 is not a log line)."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"...[{len(value) - limit} more characters]"
    if depth > 12:
        return "..."
    if isinstance(value, dict):
        return {str(k): shorten(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [shorten(v, limit, depth + 1) for v in value[:200]] + (["..."] if len(value) > 200 else [])
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    return value


class RequestLog:
    def __init__(self, path, max_bytes=DEFAULT_MAX_BYTES, bodies=False):
        self.path, self.max_bytes, self.bodies = path, max_bytes, bodies
        self.lock = threading.Lock()
        self.failed = False

    def write(self, record):
        """Append one line. Never raises: a log that cannot be written must not fail the request it describes."""
        line = dict(record)
        line.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + ".%03dZ" % (time.time() % 1 * 1000))
        try:
            text = json.dumps(line, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
        except (TypeError, ValueError):
            return
        with self.lock:
            try:
                if self.max_bytes and os.path.exists(self.path) and os.path.getsize(self.path) + len(text) > self.max_bytes:
                    os.replace(self.path, self.path + ".1")
                directory = os.path.dirname(os.path.abspath(self.path))
                os.makedirs(directory, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(text)
                self.failed = False
            except OSError as error:
                if not self.failed:                      # say it once, not once per request
                    self.failed = True
                    import sys
                    sys.stderr.write("[request-log] cannot write %s: %s\n" % (self.path, error))


def tail(path, count, window=1 << 20):
    """The last `count` records of a log file, oldest first. Reads only the last `window` bytes, so a log of any size is
    cheap to look at; a line cut by the window, or one that is not JSON, is skipped."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - window)
            handle.seek(start)
            data = handle.read()
    except OSError:
        return []
    lines = data.split(b"\n")
    if start > 0:
        lines = lines[1:]                           # the first one is probably cut
    records = []
    for line in reversed(lines):
        if len(records) >= count:
            break
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    records.reverse()
    return records


def recent(path, limit=50):
    """The last `limit` requests of a log the router wrote, newest first, one object each: the router's line with what the
    worker's line (same `trace`) adds, the tokens and why it finished. Request text is never included."""
    records = tail(path, max(1, limit) * 2 + 20)
    workers = {r["trace"]: r for r in records if r.get("source") == "gateway" and r.get("trace")}
    fronted = {r["trace"] for r in records if r.get("source") == "router" and r.get("trace")}
    out = []
    for record in records:
        if record.get("source") == "gateway" and record.get("trace") in fronted:
            continue                                # shown with its router line
        item = {k: v for k, v in record.items() if k not in ("request", "response", "source")}
        worker = workers.get(record.get("trace")) if record.get("source") == "router" else None
        if worker is not None:
            for key in ("prompt_tokens", "completion_tokens", "finish"):
                if key in worker:
                    item[key] = worker[key]
            if "seconds" in worker:
                item["engine_seconds"] = worker["seconds"]
        out.append(item)
    out.reverse()
    return out[:limit]


def from_env(environ=None):
    """The log the environment asks for, or None."""
    environ = os.environ if environ is None else environ
    path = (environ.get("COLI_REQUEST_LOG") or "").strip()
    if not path:
        return None
    try:
        limit = int(environ.get("COLI_REQUEST_LOG_MAX") or DEFAULT_MAX_BYTES)
    except ValueError:
        limit = DEFAULT_MAX_BYTES
    return RequestLog(os.path.abspath(os.path.expanduser(path)), limit,
                      (environ.get("COLI_REQUEST_LOG_BODIES") or "").strip() == "1")
