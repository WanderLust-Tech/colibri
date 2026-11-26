"""The tools of a chat session: open the MCP servers a person has approved, offer their tools to the model, run what it asks for (docs/mcp-client.md).

No terminal here, so it can be tested with a scripted model: `ToolBox` opens servers and turns their tools into the OpenAI `tools` list and back,
`run_turn` is the loop (the model answers, asks for tools, the tools run, their results go back, until the model answers in words), and
`Accumulator` puts a streamed answer's `tool_calls` fragments together. `coli chat --mcp` supplies the screen: what to show, and how to ask.

The rules the loop keeps, because the model is not the one who answers for what it runs:

- a tool that is not read-only is run only when `approve` says so, and a refusal goes back to the model as the result (it can say what it will do instead);
- a tool the model invents, or arguments that are not JSON, go back as errors it can correct, not as a crash;
- a result is cut to a length the context can hold, in the middle, with the cut said (a model told nothing about a cut reasons from half a file);
- a call that raises, times out or finds a dead server is a result with the reason, and the session goes on;
- the loop stops after `max_rounds` rounds of tools, so a model that keeps asking does not run for ever.
"""
import concurrent.futures
import difflib
import fnmatch
import json
import re
import threading

import mcp_client
import mcp_registry

MAX_RESULT_CHARS = 20000
MAX_NAME = 64
SEPARATOR = "__"


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "x"


def clip_middle(text, limit):
    """`text` cut to about `limit` characters, the middle taken out and the cut said."""
    if len(text) <= limit:
        return text
    keep = max(1, limit // 2)
    return text[:keep] + f"\n[... {len(text) - 2 * keep} characters left out of the middle ...]\n" + text[-keep:]


def allowed(tool, include, exclude):
    """Whether `tool` passes a server's `includeTools` and `excludeTools` (`*` matches anything)."""
    if include and not any(fnmatch.fnmatchcase(tool, pattern) for pattern in include):
        return False
    return not any(fnmatch.fnmatchcase(tool, pattern) for pattern in exclude or [])


class ToolInfo:
    def __init__(self, name, server, tool, description, schema, read_only):
        self.name, self.server, self.tool, self.description, self.schema, self.read_only = name, server, tool, description, schema, read_only

    def spec(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.schema}}


class ToolBox:
    """The servers of one session, connected, and the tools they offer under names that say which server they are from."""

    def __init__(self, servers, approve_server=None, environ=None, trust_file=None, connect=None):
        """`servers`: {name: registry entry}. `approve_server(name, entry)` is asked about a server this client has not approved (None: such a
        server is not started). `connect` is mcp_client.connect, replaceable for a test."""
        self.servers, self.approve_server, self.environ = servers, approve_server, environ
        self.trust_file, self._connect = trust_file, connect or mcp_client.connect
        self.clients, self.tools, self.problems = {}, {}, {}
        self.always = set()                      # tools the person has said to run without asking again, this session
        self.lock = threading.Lock()

    # ---- opening
    def _open_one(self, name, entry):
        if entry.get("disabled"):
            return name, None, "disabled"
        try:
            running = mcp_registry.expand(entry, self.environ)
            client = self._connect(name, running)
            listed = client.list_tools()
        except (mcp_registry.RegistryError, mcp_client.McpError) as error:
            return name, None, str(error)
        return name, (client, listed), None

    def open(self, names=None):
        """Connect the servers (all of them, or `names`), the ones that need no approval together. Returns {name: None | the reason it is not here}."""
        chosen = {n: e for n, e in self.servers.items() if names is None or n in names}
        unknown = [n for n in (names or []) if n not in self.servers]
        for name in unknown:
            self.problems[name] = "no such server in the list"
        asked = []                                # approvals are asked one at a time, in order, before anything connects
        for name, entry in chosen.items():
            if not entry.get("disabled") and not mcp_registry.is_trusted(name, entry, self.trust_file):
                asked.append(name)
        for name in asked:
            if not self.approve_server or not self.approve_server(name, chosen[name]):
                self.problems[name] = "not approved on this machine"
            else:
                mcp_registry.trust(name, chosen[name], self.trust_file)
        todo = {n: e for n, e in chosen.items() if n not in self.problems}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(8, len(todo)))) as pool:
            for name, got, why in pool.map(lambda item: self._open_one(*item), todo.items()):
                if why:
                    self.problems[name] = why
                    continue
                client, listed = got
                self.clients[name] = client
                entry = self.servers[name]
                for tool in listed:
                    if allowed(tool["name"], entry.get("includeTools"), entry.get("excludeTools")):
                        self._add(name, tool)
        return {n: self.problems.get(n) for n in list(chosen) + unknown}

    def _add(self, server, tool):
        base = (safe_name(server) + SEPARATOR + safe_name(tool["name"]))[:MAX_NAME]
        name, n = base, 2
        while name in self.tools:
            suffix = f"_{n}"
            name, n = base[:MAX_NAME - len(suffix)] + suffix, n + 1
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        description = (tool.get("description") or tool.get("title") or "").strip() or f"{tool['name']} (from {server})"
        read_only = bool((tool.get("annotations") or {}).get("readOnlyHint"))     # the server's own word: it is a program the person approved to run
        self.tools[name] = ToolInfo(name, server, tool["name"], f"[{server}] {description}", schema, read_only)

    # ---- using
    def specs(self):
        return [t.spec() for t in self.tools.values()]

    def instructions(self):
        """What the servers said about themselves, for the system prompt (their own words, labelled as theirs)."""
        return {n: c.instructions for n, c in self.clients.items() if c.instructions}

    def call(self, name, arguments):
        """The text a model is given for calling `name`: the result, or the reason there is none. Never raises for a server's trouble."""
        info = self.tools[name]
        try:
            result = self.clients[info.server].call_tool(info.tool, arguments)
        except mcp_client.McpTimeout as error:
            return f"Error: the tool `{info.tool}` did not answer ({error})."
        except mcp_client.McpError as error:
            return f"Error: {error}"
        return ("Error: " if result.is_error else "") + (result.text or "(the tool returned nothing)")

    def close(self):
        for client in self.clients.values():
            try:
                client.close()
            except Exception:
                pass
        self.clients.clear()


class Accumulator:
    """Puts the `tool_calls` of a streamed answer together: each chunk has an `index`, and the id, the name and pieces of the arguments
    arrive over several chunks."""

    def __init__(self):
        self.calls = {}

    def feed(self, fragments):
        for position, piece in enumerate(fragments or []):
            if not isinstance(piece, dict):
                continue
            index = piece.get("index", position)
            call = self.calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if piece.get("id"):
                call["id"] = piece["id"]
            function = piece.get("function") or {}
            if function.get("name"):
                call["name"] += function["name"]
            if function.get("arguments"):
                call["arguments"] += function["arguments"] if isinstance(function["arguments"], str) else json.dumps(function["arguments"])

    def result(self):
        out = []
        for index in sorted(self.calls):
            call = dict(self.calls[index])
            if not call["name"]:
                continue
            call["id"] = call["id"] or f"call_{index}"
            out.append(call)
        return out


def assistant_message(text, calls):
    message = {"role": "assistant", "content": text or ""}
    if calls:
        message["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for c in calls]
    return message


def run_turn(complete, messages, box, approve=None, on_call=None, on_result=None, max_rounds=25, max_chars=MAX_RESULT_CHARS):
    """Let the model answer, and run the tools it asks for, until it answers without asking. `messages` is extended in place.

    `complete(messages, tools)` makes one request and returns (text, calls): calls are `{"id", "name", "arguments"}` (the arguments as the model
    wrote them, text). `approve(info, arguments)` returns "yes", "no" or "always" for a tool that is not read-only (None: refuse them all).
    `on_call(info, arguments, how)` and `on_result(info, text)` let a screen show what happens. Returns "done", "interrupted" (Ctrl-C during a tool), or "rounds" when the model was
    still asking after `max_rounds` rounds of tools: it is then asked once more with no tools to offer, to say what it has."""
    for rounds in range(max_rounds + 1):
        last = rounds == max_rounds
        text, calls = complete(messages, [] if last else box.specs())
        messages.append(assistant_message(text, [] if last else calls))
        if last:
            return "rounds"
        if not calls:
            return "done"
        for done, call in enumerate(calls):
            try:
                reply = _one_call(call, box, approve, on_call, on_result, max_chars)
            except KeyboardInterrupt:                                  # every call keeps its answer, or the transcript is not valid
                for rest in calls[done:]:
                    messages.append({"role": "tool", "tool_call_id": rest["id"], "content": "The person interrupted before this ran or finished."})
                return "interrupted"
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": reply})
        if rounds + 1 == max_rounds:
            messages.append({"role": "user", "content": f"[The limit of {max_rounds} rounds of tools for one question is reached. Answer now with what you have.]"})
    return "rounds"


def _one_call(call, box, approve, on_call, on_result, max_chars):
    info = box.tools.get(call["name"])
    if info is None:
        near = difflib.get_close_matches(call["name"], list(box.tools), n=3, cutoff=0.6)
        return f"Error: there is no tool called `{call['name']}`." + (f" Did you mean: {', '.join(near)}?" if near else
                                                                       " The tools are: " + ", ".join(box.tools) + ".")
    try:
        arguments = json.loads(call["arguments"]) if (call["arguments"] or "").strip() else {}
    except ValueError as error:
        return f"Error: the arguments of `{call['name']}` are not valid JSON ({error}). Send one JSON object."
    if not isinstance(arguments, dict):
        return f"Error: the arguments of `{call['name']}` must be a JSON object."
    if info.read_only or info.name in box.always:
        how = "read-only" if info.read_only else "approved for this session"
    else:
        answer = approve(info, arguments) if approve else "no"
        if answer == "always":
            box.always.add(info.name)
        elif answer != "yes":
            if on_call:
                on_call(info, arguments, "declined")
            return "The person declined to run this tool call. Say what you will do instead, or ask them."
        how = "approved"
    if on_call:
        on_call(info, arguments, how)
    text = box.call(info.name, arguments)
    if on_result:
        on_result(info, text)
    return clip_middle(text, max_chars)
