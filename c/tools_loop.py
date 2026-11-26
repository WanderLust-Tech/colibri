"""Chat completions where the router runs the tools (docs/builtin-tools.md).

`POST /v1/chat/completions` with `"colibri_tools": true` (or a list of tool and group names) makes the router run the loop itself: it asks the
model, runs the tools the model asks for (the built-in ones of `builtin_tools.py`, inside `tools_policy.py`), sends the results back, and
repeats until the model answers in words. The client gets one answer, as from any chat request, plus what happened:

- not streamed: an ordinary chat completion; `colibri.tool_calls` lists every call (name, arguments, whether it worked, the start of the
  result) and `usage` adds up every round;
- streamed: ordinary chat chunks (the text of each round, as it is written, with the model's own tool_calls left out), and chunks with an
  empty `choices` and a `colibri` object between them: `{"event": "tool_call" | "tool_result" | "approval" | "approval_answered", ...}`.

A tool that only reads runs when the model asks. One that changes something waits for a person: the stream says `approval` with an `id`, and
`POST /v1/tools/pending {"id", "allow", "always"}` answers it (the dashboard's button, or `curl`); no answer in `approvalSeconds` is a no. A
refusal goes back to the model as the result, so it can say what it will do instead.

The loop is the router talking to itself over loopback with the caller's own key and tags, like `responses_shim`: each round is a routed,
queued, logged request. The rules of `mcp_agent.run_turn` apply (results cut and the cut said, bad calls answered with the reason, a limit on
rounds). The standard library only.
"""
import http.client
import json
import threading
import time
import urllib.parse
import uuid

import builtin_tools
import mcp_agent
import responses_shim

MAX_ROUNDS = 15
HEARTBEAT = 5.0
RESULT_SHOWN = 2000
NESTED_SECONDS = 3600
FORWARDED = ("Authorization", "x-api-key", "X-Colibri-Tags")
OWN_KEYS = ("colibri_tools",)


class LoopError(Exception):
    """A request the loop cannot take, as (status, code, message, param)."""

    def __init__(self, status, code, message, param=None):
        super().__init__(message)
        self.status, self.code, self.message, self.param = status, code, message, param


class ClientGone(Exception):
    """The caller closed the connection: nobody is left to answer."""


class Upstream(Exception):
    def __init__(self, status, text):
        super().__init__(text)
        self.status, self.text = status, text


class Pending:
    """The calls that wait for a person's yes (GET and POST /v1/tools/pending)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.items = {}

    def add(self, tool, arguments, summary, seconds, client=None):
        item = {"id": uuid.uuid4().hex[:12], "tool": tool, "arguments": arguments, "summary": summary, "created": time.time(),
                "expires": time.time() + seconds, "client": client, "event": threading.Event(), "answer": None}
        with self.lock:
            self.items[item["id"]] = item
        return item

    def listing(self):
        now = time.time()
        with self.lock:
            return [{"id": i["id"], "tool": i["tool"], "arguments": i["arguments"], "summary": i["summary"],
                     "expires_in": max(0, round(i["expires"] - now)), "client": i["client"]}
                    for i in self.items.values() if i["answer"] is None and i["expires"] > now]

    def answer(self, id_, allow, always=False):
        """Record a person's answer. False when there is no such call waiting (answered already, timed out, or never was)."""
        with self.lock:
            item = self.items.get(id_)
            if item is None or item["answer"] is not None or item["expires"] <= time.time():
                return False
            item["answer"] = "always" if (allow and always) else "yes" if allow else "no"
        item["event"].set()
        return True

    def wait(self, item, tick=None):
        """The answer ("yes", "no", "always") once there is one; "no" when time runs out. `tick()` is called every few seconds while waiting
        (it may raise ClientGone: the caller left, and the call is dropped)."""
        try:
            while True:
                left = item["expires"] - time.time()
                if left <= 0:
                    return "no"
                if item["event"].wait(min(HEARTBEAT, left)):
                    return item["answer"] or "no"
                if tick:
                    tick()
        finally:
            with self.lock:
                self.items.pop(item["id"], None)
            item["event"].set()


def _clip(text, limit=RESULT_SHOWN):
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more characters]"


def backend_connection(backend):
    return responses_shim.gateway_connection(backend, timeout=NESTED_SECONDS)


class Run:
    """One chat request being served: the box of tools, the nested requests, and how the answer goes back."""

    def __init__(self, h, backend, body, box, pending, policy, seen=None):
        self.h, self.backend, self.box, self.pending, self.policy = h, backend, box, pending, policy
        self.seen = seen if seen is not None else {}
        self.body = {k: v for k, v in body.items() if k not in OWN_KEYS}
        self.stream = bool(body.get("stream"))
        self.model = body.get("model") or "colibri"
        self.id = "chatcmpl-" + uuid.uuid4().hex[:24]
        self.created = int(time.time())
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.trace = []
        self.started = False                    # the stream's headers are out
        self.sent_role = False
        self.finish = "stop"
        self.headers = {"Content-Type": "application/json"}
        for name in FORWARDED:
            if h.headers.get(name):
                self.headers[name] = h.headers[name]

    # ---- the way back to the caller
    def _write(self, data):
        try:
            self.h.wfile.write(data)
            self.h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            raise ClientGone()

    def start_stream(self):
        if self.started:
            return
        self.started = True
        self.h.send_response(200)
        self.h.send_header("Content-Type", "text/event-stream")
        self.h.send_header("Cache-Control", "no-store")
        self.h.send_header("Connection", "close")
        self.h.close_connection = True
        self.h.end_headers()

    def chunk(self, choices, extra=None):
        out = {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model, "choices": choices}
        if extra:
            out.update(extra)
        self.start_stream()
        self._write(b"data: " + json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode() + b"\n\n")

    def event(self, **fields):
        if self.stream:
            self.chunk([], {"colibri": fields})

    def heartbeat(self):
        if self.stream:
            self.start_stream()
            self._write(b": waiting\n\n")

    # ---- one request to the model, through the router's own door
    def complete(self, messages, tools):
        body = dict(self.body, messages=messages, stream=self.stream)
        body.pop("tools", None)
        if tools:
            body["tools"] = tools
        if self.stream:
            body["stream_options"] = {"include_usage": True}
        conn = backend_connection(self.backend)
        try:
            try:
                conn.request("POST", "/v1/chat/completions", body=json.dumps(body), headers=self.headers)
                response = conn.getresponse()
            except OSError as error:
                raise Upstream(502, f"the router could not reach its own chat endpoint: {error}")
            if response.status != 200:
                raise Upstream(response.status, response.read().decode("utf-8", "replace")[:4000])
            return self._read_stream(response) if self.stream else self._read_json(response)
        finally:
            conn.close()

    def _add_usage(self, usage):
        if isinstance(usage, dict):
            for key in self.usage:
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    self.usage[key] += value

    def _read_json(self, response):
        try:
            data = json.loads(response.read())
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise Upstream(502, "the model's answer could not be read")
        self._add_usage(data.get("usage"))
        self.finish = choice.get("finish_reason") or "stop"
        self.model = data.get("model") or self.model
        calls = [{"id": c.get("id") or f"call_{i}", "name": (c.get("function") or {}).get("name", ""),
                  "arguments": (c.get("function") or {}).get("arguments") or ""} for i, c in enumerate(message.get("tool_calls") or [])]
        for call in calls:
            if not isinstance(call["arguments"], str):
                call["arguments"] = json.dumps(call["arguments"])
        return message.get("content") or "", calls

    def _read_stream(self, response):
        accumulator, text = mcp_agent.Accumulator(), []
        try:
            for data in responses_shim._chat_events(response):
                if data.strip() == b"[DONE]":
                    break
                try:
                    piece = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(piece, dict):
                    continue
                if "error" in piece:
                    error = piece["error"]
                    raise Upstream(502, error.get("message") if isinstance(error, dict) else str(error))
                self._add_usage(piece.get("usage"))
                self.model = piece.get("model") or self.model
                for choice in piece.get("choices") or []:
                    delta = dict(choice.get("delta") or {})
                    accumulator.feed(delta.pop("tool_calls", None))
                    delta.pop("function_call", None)
                    if choice.get("finish_reason"):
                        self.finish = choice["finish_reason"]
                    if delta.get("role") and self.sent_role:
                        delta.pop("role")
                    if isinstance(delta.get("content"), str):
                        text.append(delta["content"])
                    if delta:
                        self.sent_role = self.sent_role or bool(delta.get("role"))
                        self.chunk([{"index": 0, "delta": delta, "finish_reason": None}])
        except (OSError, http.client.HTTPException) as error:
            raise Upstream(502, f"the model's stream was cut ({error})")
        return "".join(text), accumulator.result()

    # ---- what the loop shows
    def approve(self, info, arguments):
        item = self.pending.add(info.name, arguments, self.box.describe(info.name, arguments), self.policy.approval_seconds,
                                client=self.h.client_address[0] if self.h.client_address else None)
        self.event(event="approval", id=item["id"], tool=info.name, arguments=arguments, summary=item["summary"],
                   expires_in=round(self.policy.approval_seconds))
        answer = self.pending.wait(item, tick=self.heartbeat)
        self.event(event="approval_answered", id=item["id"], tool=info.name, allowed=answer in ("yes", "always"))
        return answer

    def on_call(self, info, arguments, how):
        self.trace.append({"name": info.name, "arguments": arguments, "how": how, "ok": None, "result": None})
        self.event(event="tool_call", name=info.name, arguments=arguments, read_only=info.read_only, how=how)

    def on_result(self, info, text):
        ok = not text.startswith("Error: ")
        if self.trace:
            self.trace[-1].update(ok=ok, result=_clip(text), target=self.box.target(info.name, self.trace[-1]["arguments"]))
        self.event(event="tool_result", name=info.name, ok=ok, chars=len(text), text=_clip(text))

    def used(self):
        """What the request log keeps: the tools used and what they were about (never the content)."""
        return [{"tool": t["name"], "target": t.get("target"), "ok": t["ok"], "how": t["how"]} for t in self.trace]

    # ---- the whole request
    def run(self, messages):
        try:
            how = mcp_agent.run_turn(self.complete, messages, self.box, approve=self.approve, on_call=self.on_call, on_result=self.on_result,
                                     max_rounds=MAX_ROUNDS)
        except ClientGone:
            self.seen["tool_calls"] = self.used()
            self.seen["note"] = "client left"
            return
        except Upstream as error:
            self.seen["tool_calls"] = self.used()
            return self.fail(error)
        self.seen["tool_calls"] = self.used()
        reason = "length" if self.finish == "length" else "stop"
        final = messages[-1].get("content") or ""
        if self.stream:
            try:
                self.chunk([{"index": 0, "delta": {}, "finish_reason": reason}])
                if (self.body.get("stream_options") or {}).get("include_usage"):
                    self.chunk([], {"usage": self._totals()})
                self._write(b"data: [DONE]\n\n")
            except ClientGone:
                pass
            return
        self.usage["total_tokens"] = self.usage["prompt_tokens"] + self.usage["completion_tokens"]
        out = {"id": self.id, "object": "chat.completion", "created": self.created, "model": self.model,
               "choices": [{"index": 0, "message": {"role": "assistant", "content": final}, "finish_reason": reason}],
               "usage": self._totals(), "colibri": {"tool_calls": self.trace, "rounds_stopped": how == "rounds"}}
        self.h.send_json(200, out)

    def _totals(self):
        return dict(self.usage, total_tokens=self.usage["prompt_tokens"] + self.usage["completion_tokens"])

    def fail(self, error):
        """The model could not answer: its own error, passed on (as an `error` chunk when the stream has begun)."""
        try:
            parsed = json.loads(error.text)
            body = parsed if isinstance(parsed, dict) and "error" in parsed else None
        except ValueError:
            body = None
        body = body or {"error": {"message": error.text or "the model did not answer", "type": "server_error", "code": "upstream_error"}}
        if self.started:
            try:
                self._write(b"data: " + json.dumps(body, ensure_ascii=False).encode() + b"\n\ndata: [DONE]\n\n")
            except ClientGone:
                pass
            return
        self.h.send_json(error.status if 400 <= error.status < 600 else 502, body)


def check_request(body, policy):
    """The tools a chat request asks for, as a BuiltinBox, or a LoopError saying what to change."""
    if any(k in body for k in ("tools", "functions", "tool_choice", "function_call")):
        raise LoopError(400, "invalid_request_error", "Send either `tools` of your own (then you run the loop) or `colibri_tools` (then the router does), not both.",
                        "colibri_tools")
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise LoopError(400, "invalid_request_error", "`messages` must be a list with at least one message.", "messages")
    if body.get("n") not in (None, 1):
        raise LoopError(400, "invalid_request_error", "`n` must be 1 with `colibri_tools`.", "n")
    try:
        box = builtin_tools.BuiltinBox(policy, body.get("colibri_tools"))
    except builtin_tools.ToolError as error:
        raise LoopError(400, "invalid_request_error", str(error), "colibri_tools")
    if not box.tools:
        raise LoopError(409, "tools_not_enabled", "No built-in tool is on for what was asked. On the router's PC: `coli tools allow FOLDER` (files and git), "
                                                  "`coli tools web on`, or `coli tools allow FOLDER --write`.", "colibri_tools")
    return box


def serve(h, backend, body, policy, pending, seen=None):
    """Answer a chat request that carries `colibri_tools` on handler `h`, with the router at `backend` (a parsed URL) for the model."""
    box = check_request(body, policy)
    run = Run(h, backend, body, box, pending, policy, seen)
    run.run([dict(m) if isinstance(m, dict) else m for m in body["messages"]])


def loopback(port):
    return urllib.parse.urlsplit(f"http://127.0.0.1:{port}")
