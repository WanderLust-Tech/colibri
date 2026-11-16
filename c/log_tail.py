"""The last lines the router and its workers wrote, kept in memory so a client can read them (`GET /v1/logs`).

The router prints its own lines and every line a worker prints (a model loading, a failed start, a GPU falling back to the
CPU) to its console. A person using the dashboard or an editor cannot see that console, so when a load is slow or fails
the only answer is "go and look". This keeps the most recent lines, each with a number that only grows, so a client asks
"everything after the last number I saw" and gets just the new lines.

Nothing is written to disk. A line is cut at LINE_LIMIT characters. Worker output can contain whatever an engine prints,
which is why the endpoint needs the API key like every other management route.
"""
import collections
import threading
import time

DEFAULT_LINES = 2000
LINE_LIMIT = 2000


class LogTail:
    def __init__(self, maxlen=DEFAULT_LINES):
        self.lines = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.last = 0                       # the number of the newest line (0: nothing yet)

    def add(self, source, text, now=None):
        text = str(text)
        if len(text) > LINE_LIMIT:
            text = text[:LINE_LIMIT] + "...[cut]"
        with self.lock:
            self.last += 1
            self.lines.append({"seq": self.last, "time": time.time() if now is None else now,
                               "source": str(source), "text": text})

    def read(self, since=0, limit=500, source=None):
        """The lines numbered after `since`, oldest first, at most `limit` (the newest ones when more are waiting).
        `next` is the number to send as `since` next time. `first` is the number of the oldest line still held: when it
        is more than `since` + 1, lines were dropped between two reads and the client should say so."""
        limit = max(1, min(DEFAULT_LINES, int(limit)))
        with self.lock:
            held = [line for line in self.lines if line["seq"] > since and (source is None or line["source"] == source)]
            first = self.lines[0]["seq"] if self.lines else self.last + 1
            last = self.last
        if len(held) > limit:
            held = held[-limit:]
        return {"data": held, "next": last, "first": first}


TAIL = LogTail()
