"""A small MCP client: start a server (a program on its standard input, or an http address), list its tools, call them (docs/mcp-client.md).

Only the standard library, so `coli chat --mcp` needs nothing installed. It speaks the 2025 revisions of the protocol that matter for
tools: `initialize` and `notifications/initialized`, `tools/list` (with its paging), `tools/call`, `ping`, and `notifications/cancelled`
when a call times out. Two transports:

- **stdio**: the server is a child process; messages are JSON lines on its stdin and stdout; its stderr is kept (the last few KB) so a
  server that cannot start says why.
- **Streamable HTTP**: each message is a POST; the answer is JSON, or an event stream that carries it; the session id the server gives is
  sent back on every later request. (The older two-endpoint SSE transport is not spoken.)

What a server asks of its client (sampling, roots, elicitation) is answered "not supported": this client offers none of them.
`mcp_registry.py` holds the list of servers and the rules for what a person has approved; this file only runs what it is given.
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

PROTOCOL = "2025-06-18"
CLIENT_NAME = "colibri"
STDERR_KEPT = 4000


class McpError(RuntimeError):
    """A server that could not be started or that answered with an error."""


class McpTimeout(McpError):
    """No answer in time."""


class ToolResult:
    def __init__(self, text, is_error=False, content=None):
        self.text, self.is_error, self.content = text, is_error, content or []

    def __repr__(self):
        return f"ToolResult({self.text[:40]!r}, is_error={self.is_error})"


def content_text(result):
    """The text a model is given for a `tools/call` result: the text blocks, a note for the kinds a text model cannot read, and the
    structured result when there is nothing else."""
    parts = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind in ("image", "audio"):
            size = len(block.get("data") or "") * 3 // 4
            parts.append(f"[{kind}: {block.get('mimeType', 'unknown type')}, about {size} bytes, not shown]")
        elif kind == "resource":
            resource = block.get("resource") or {}
            parts.append(str(resource["text"]) if "text" in resource else f"[resource {resource.get('uri', '')}: {resource.get('mimeType', 'binary')}]")
        elif kind == "resource_link":
            parts.append(f"[link: {block.get('uri', '')}" + (f" ({block['name']})" if block.get("name") else "") + "]")
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(parts)


def _hello(request_id):
    return {"jsonrpc": "2.0", "id": request_id, "method": "initialize",
            "params": {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": CLIENT_NAME, "version": "1"}}}


# ---------------------------------------------------------------- stdio

class StdioTransport:
    def __init__(self, command, args=(), env=None, cwd=None):
        found = shutil.which(command, path=(env or os.environ).get("PATH")) if not os.path.dirname(command) else command
        if not found:
            raise McpError(f"the command `{command}` was not found on the PATH")
        full_env = dict(os.environ)
        full_env.update(env or {})
        flags = 0x08000000 if sys.platform == "win32" else 0            # no console window of its own on Windows
        try:
            self.process = subprocess.Popen([found, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            env=full_env, cwd=cwd or None, creationflags=flags)
        except OSError as error:
            raise McpError(f"`{command}` could not be started: {error}")
        self.pending = {}
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.stderr_tail = b""
        self.closed = False
        self.on_request = None
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self):
        try:
            for chunk in iter(lambda: self.process.stderr.read1(4096) if hasattr(self.process.stderr, "read1") else self.process.stderr.read(4096), b""):
                self.stderr_tail = (self.stderr_tail + chunk)[-STDERR_KEPT:]
        except (OSError, ValueError):
            pass

    def stderr_text(self):
        return self.stderr_tail.decode("utf-8", "replace").strip()

    def _read(self):
        try:
            for raw in self.process.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    continue                                          # a server that prints other things on stdout: not a message
                if isinstance(message, dict):
                    self._dispatch(message)
        except (OSError, ValueError):
            pass
        finally:
            with self.lock:
                self.closed = True
                for slot in self.pending.values():
                    slot.put({"error": {"code": -32000, "message": "the server ended"}})

    def _dispatch(self, message):
        if "method" in message:
            if "id" in message and self.on_request:
                self.send(self.on_request(message))
            return
        slot = self.pending.get(message.get("id"))
        if slot is not None:
            slot.put(message)

    def send(self, message):
        if message is None:
            return
        data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        with self.write_lock:
            try:
                self.process.stdin.write(data)
                self.process.stdin.flush()
            except (OSError, ValueError):
                raise McpError("the server is not running" + self._why())

    def _why(self):
        try:
            code = self.process.wait(timeout=1)                       # a server that just closed its pipe is still on its way out
        except subprocess.TimeoutExpired:
            code = None
        tail = self.stderr_text()
        return (f" (it exited with code {code})" if code is not None else "") + (f": {tail[-600:]}" if tail else "")

    def request(self, message, timeout):
        slot = queue.Queue()
        with self.lock:
            if self.closed:
                raise McpError("the server is not running" + self._why())
            self.pending[message["id"]] = slot
        try:
            self.send(message)
            try:
                reply = slot.get(timeout=timeout)
            except queue.Empty:
                raise McpTimeout(f"no answer in {timeout:g} s")
        finally:
            self.pending.pop(message["id"], None)
        if reply.get("error", {}).get("message") == "the server ended":
            raise McpError("the server ended before it answered" + self._why())
        return reply

    def notify(self, message):
        self.send(message)

    def close(self):
        try:
            self.process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        for pipe in (self.process.stdout, self.process.stderr):
            try:
                pipe.close()
            except (OSError, ValueError):
                pass


# ---------------------------------------------------------------- Streamable HTTP

class HttpTransport:
    def __init__(self, url, headers=None):
        self.url, self.headers, self.session = url, dict(headers or {}), None
        self.on_request = None

    def _post(self, message, timeout):
        data = json.dumps(message, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json, text/event-stream")
        request.add_header("MCP-Protocol-Version", PROTOCOL)
        if self.session:
            request.add_header("Mcp-Session-Id", self.session)
        for name, value in self.headers.items():
            request.add_header(name, value)
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            body = error.read(600).decode("utf-8", "replace")
            error.close()
            raise McpError(f"{self.url} answered HTTP {error.code}" + (f": {body.strip()}" if body.strip() else ""))
        except (urllib.error.URLError, OSError) as error:
            if "timed out" in str(error):
                raise McpTimeout(f"no answer in {timeout:g} s")
            raise McpError(f"{self.url} cannot be reached: {getattr(error, 'reason', error)}")

    @staticmethod
    def _events(response):
        """The JSON messages of an event stream, one at a time."""
        data = []
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line and data:
                try:
                    yield json.loads("\n".join(data))
                except ValueError:
                    pass
                data = []
        if data:
            try:
                yield json.loads("\n".join(data))
            except ValueError:
                pass

    def request(self, message, timeout):
        deadline = time.monotonic() + timeout
        response = self._post(message, timeout)
        try:
            if message.get("method") == "initialize":
                self.session = response.headers.get("Mcp-Session-Id") or self.session
            kind = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if kind == "text/event-stream":
                for item in self._events(response):
                    if item.get("id") == message["id"] and "method" not in item:
                        return item
                    if "method" in item and "id" in item and self.on_request:
                        self._post(self.on_request(item), max(1.0, deadline - time.monotonic())).close()
                    if time.monotonic() > deadline:
                        raise McpTimeout(f"no answer in {timeout:g} s")
                raise McpError("the event stream ended before the answer")
            body = response.read().decode("utf-8", "replace")
            try:
                reply = json.loads(body)
            except ValueError:
                raise McpError("the server did not answer with JSON")
            if isinstance(reply, list):                                # a batch: the answer to our request is one of them
                reply = next((r for r in reply if isinstance(r, dict) and r.get("id") == message["id"]), None) or {}
            return reply
        finally:
            response.close()

    def notify(self, message):
        self._post(message, 20).close()

    def close(self):
        if self.session:
            try:
                request = urllib.request.Request(self.url, method="DELETE")
                request.add_header("Mcp-Session-Id", self.session)
                for name, value in self.headers.items():
                    request.add_header(name, value)
                urllib.request.urlopen(request, timeout=5).close()
            except Exception:
                pass                                                   # a server may refuse it: the session ends anyway


# ---------------------------------------------------------------- the client

def _answer_server_request(message):
    """What a server asked of us: a `ping` is answered, anything else is a method this client does not have."""
    if message.get("method") == "ping":
        return {"jsonrpc": "2.0", "id": message["id"], "result": {}}
    return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "this client does not offer " + str(message.get("method"))}}


class McpClient:
    """One connected server. Build it with `connect(cfg)` (a server of mcp_registry, its `${NAME}`s already expanded)."""

    def __init__(self, transport, name="server", call_timeout=120.0):
        self.transport, self.name, self.call_timeout = transport, name, call_timeout
        self.next_id = 1
        self.info = {}
        self.instructions = ""
        transport.on_request = _answer_server_request

    def _id(self):
        self.next_id += 1
        return self.next_id

    def _call(self, method, params, timeout):
        message = {"jsonrpc": "2.0", "id": self._id(), "method": method}
        if params is not None:
            message["params"] = params
        try:
            reply = self.transport.request(message, timeout)
        except McpTimeout:
            try:
                self.transport.notify({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": message["id"], "reason": "timed out"}})
            except McpError:
                pass
            raise
        if "error" in reply:
            error = reply["error"] if isinstance(reply["error"], dict) else {}
            raise McpError(f"{method}: {error.get('message', reply['error'])}")
        return reply.get("result") or {}

    def initialize(self, timeout=30.0):
        message = _hello(1)
        reply = self.transport.request(message, timeout)
        if "error" in reply:
            raise McpError("initialize: " + str((reply["error"] or {}).get("message", reply["error"])))
        result = reply.get("result") or {}
        self.info = result.get("serverInfo") or {}
        self.instructions = result.get("instructions") or ""
        self.transport.notify({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result

    def list_tools(self, timeout=30.0):
        tools, cursor = [], None
        for _ in range(50):                                            # a server that never stops paging is not followed for ever
            result = self._call("tools/list", {"cursor": cursor} if cursor else None, timeout)
            tools += [t for t in result.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("name"), str)]
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(self, tool, arguments, timeout=None):
        result = self._call("tools/call", {"name": tool, "arguments": arguments or {}}, timeout or self.call_timeout)
        return ToolResult(content_text(result), bool(result.get("isError")), result.get("content"))

    def close(self):
        self.transport.close()


def connect(name, cfg, connect_timeout=None):
    """A connected, initialised client for the server `cfg` (a registry entry with its references expanded), or an McpError that says
    why not. The caller closes it."""
    timeout = float(cfg.get("connectTimeoutSeconds") or connect_timeout or 30.0)
    if cfg.get("url"):
        transport = HttpTransport(cfg["url"], cfg.get("headers"))
    else:
        transport = StdioTransport(cfg["command"], cfg.get("args") or [], cfg.get("env"), cfg.get("cwd"))
    client = McpClient(transport, name, float(cfg.get("timeoutSeconds") or 120.0))
    try:
        client.initialize(timeout)
    except McpTimeout:
        tail = getattr(transport, "stderr_text", lambda: "")()
        client.close()
        raise McpError(f"`{name}` did not start in {timeout:g} s" + (f": {tail[-400:]}" if tail else ""))
    except McpError:
        client.close()
        raise
    return client
