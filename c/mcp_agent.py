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
import math
import re
import threading

import mcp_client
import mcp_registry

MAX_RESULT_CHARS = 20000
MAX_NAME = 64
SEPARATOR = "__"
SEARCH_TOOL, CALL_TOOL = "tools_search", "tools_call"      # what the model sees for a server whose `toolMode` is "search"
SEARCH_LIMIT, SEARCH_SCHEMA_CHARS, SEARCH_DESCRIPTION_CHARS = 5, 2500, 220


def signature(schema):
    """A tool's arguments in one line: `path*: string, detail: summary|complete` (`*` = required)."""
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict) or not properties:
        return "(none)"
    required = set(schema.get("required") or [])
    bits = []
    for name, spec in list(properties.items())[:14]:
        kind = spec.get("type") if isinstance(spec, dict) else None
        if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]:
            kind = "|".join(str(v) for v in spec["enum"][:6]) + ("|..." if len(spec["enum"]) > 6 else "")
        elif isinstance(kind, list):
            kind = "/".join(str(k) for k in kind)
        bits.append(f"{name}{'*' if name in required else ''}: {kind or 'any'}")
    if len(properties) > 14:
        bits.append(f"... +{len(properties) - 14} more")
    return ", ".join(bits)


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
        self.deferred = {}                       # the tools of "search" servers: not offered, found with tools_search and run with tools_call
        self.searched = set()                    # searches already made, so an exact repeat can be told it is one
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
                        self._add(name, tool, deferred=entry.get("toolMode") == "search")
        return {n: self.problems.get(n) for n in list(chosen) + unknown}

    def _add(self, server, tool, deferred=False):
        base = (safe_name(server) + SEPARATOR + safe_name(tool["name"]))[:MAX_NAME]
        name, n = base, 2
        while name in self.tools or name in self.deferred:
            suffix = f"_{n}"
            name, n = base[:MAX_NAME - len(suffix)] + suffix, n + 1
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        description = (tool.get("description") or tool.get("title") or "").strip() or f"{tool['name']} (from {server})"
        read_only = bool((tool.get("annotations") or {}).get("readOnlyHint"))     # the server's own word: it is a program the person approved to run
        (self.deferred if deferred else self.tools)[name] = ToolInfo(name, server, tool["name"], f"[{server}] {description}", schema, read_only)

    # ---- using
    def specs(self):
        return [t.spec() for t in self.tools.values()] + self.meta_specs()

    # ---- tools found on demand (a server with hundreds of tools costs the whole window if they are all listed)
    def meta_specs(self):
        """`tools_search` and `tools_call`, when a server's tools are on demand: two small fixed definitions instead of all of theirs
        (and a list that never changes, so the model's cache of the conversation stays good)."""
        if not self.deferred:
            return []
        servers = {}
        for info in self.deferred.values():
            servers[info.server] = servers.get(info.server, 0) + 1
        where = ", ".join(f"{s} ({n} tools)" for s, n in servers.items())
        return [
            {"type": "function", "function": {
                "name": SEARCH_TOOL,
                "description": f"Find tools. The tools of these servers are not all listed here: {where}. Say in a few words what you want to do and the answer lists the "
                               "matching tools with their arguments; an empty query lists every tool's name. Then run one with tools_call.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what you want to do, e.g. 'decompile a function'"},
                                                                "limit": {"type": "integer", "description": f"how many to list (default {SEARCH_LIMIT}, at most 10)"},
                                                                "full": {"type": "boolean", "description": "give the first match's complete argument schema"}}}}},
            {"type": "function", "function": {
                "name": CALL_TOOL,
                "description": "Run a tool found with tools_search. `name` is the tool's name exactly as the search showed it; `arguments` is an object with the arguments it lists.",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name"]}}}]

    def meta_info(self, which):
        """A ToolInfo for the screen to show a search or a call-through with."""
        return ToolInfo(which, "", which, which, {"type": "object", "properties": {}}, which == SEARCH_TOOL)

    def find(self, name):
        """The on-demand tool `name` stands for: its full name, or its own name when only one server has it. None when there is none."""
        if name in self.deferred:
            return self.deferred[name]
        same = [i for i in self.deferred.values() if i.tool == name]
        return same[0] if len(same) == 1 else None

    def search(self, query, limit=SEARCH_LIMIT, full=False):
        """The text a model gets for tools_search: the best matches, each with its arguments in short (`full` gives the first one's whole schema).
        Words that few tools use count for more than words many use (`address` is in most of a binary tool's descriptions), and a word in a tool's
        name counts for more than one in its description."""
        query = (query or "").strip() if isinstance(query, str) else ""
        if not query:
            servers = {}
            for info in self.deferred.values():
                servers.setdefault(info.server, []).append(info.name)
            return "\n".join(f"{server} ({len(names)} tools): " + ", ".join(names) for server, names in servers.items()) \
                + "\nSearch for words about what you want to do to see a tool's description and arguments."
        words = list(dict.fromkeys(re.findall(r"[a-z0-9]+", query.lower())))
        rows = [(info, info.tool.lower(), set(re.findall(r"[a-z0-9]+", info.tool.lower())), info.description.lower()) for info in self.deferred.values()]
        weight = {w: math.log(1 + len(rows) / max(1, sum(1 for _, tool, _, text in rows if w in tool or w in text))) for w in words}
        scored = []
        for info, tool, parts, text in rows:
            score = sum(weight[w] * ((4 if w in parts else 2 if w in tool else 0) + (1 if w in text else 0)) for w in words)
            if score:
                scored.append((-score, info.name, info))
        scored.sort(key=lambda s: s[:2])
        if not scored:
            near = difflib.get_close_matches(query.lower(), [i.tool.lower() for i in self.deferred.values()], n=5, cutoff=0.5)
            return f"No tool matches {query!r}." + (f" Names that are close: {', '.join(near)}." if near else "") + " An empty query lists every tool's name."
        limit = max(1, min(10, limit if isinstance(limit, int) else SEARCH_LIMIT))
        out = []
        for position, (_, _, info) in enumerate(scored[:limit]):
            if full and position == 0:
                shown = json.dumps(info.schema, separators=(",", ":"))
                if len(shown) > SEARCH_SCHEMA_CHARS:
                    shown = shown[:SEARCH_SCHEMA_CHARS] + " ...[the rest of the schema left out]"
            else:
                shown = signature(info.schema)
            first = info.description.split("] ", 1)[-1]
            first = first.split(". ")[0][:SEARCH_DESCRIPTION_CHARS]
            out.append(f"{info.name}  ({'read-only' if info.read_only else 'asks the person first'})\n  {first}\n  arguments: {shown}")
        more = len(scored) - limit
        text = "\n\n".join(out) + ("\n\n(* = required)" if not full else "") \
            + (f"\n{more} more match{'es' if more != 1 else ''}: a more specific query, or a larger limit, shows them." if more > 0 else "")
        key = (query.lower(), limit, bool(full))
        if key in self.searched:
            text = "[You searched for exactly this before and the answer is the same. Pick a tool from it and run it with tools_call, or search with different words.]\n" + text
        self.searched.add(key)
        return text

    def instructions(self):
        """What the servers said about themselves, for the system prompt (their own words, labelled as theirs)."""
        return {n: c.instructions for n, c in self.clients.items() if c.instructions}

    def call(self, name, arguments):
        """The text a model is given for calling `name`: the result, or the reason there is none. Never raises for a server's trouble."""
        info = self.tools.get(name) or self.deferred[name]
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


def _through(call, box, approve, on_call, on_result, max_chars):
    """tools_search and tools_call: the search answers from the list the box kept; a call-through is the real tool's own call (its name, its approval,
    its read-only mark), so the person is asked about `procedure_pseudo_code` with its arguments, not about `tools_call`."""
    try:
        arguments = json.loads(call["arguments"]) if (call["arguments"] or "").strip() else {}
    except ValueError as error:
        return f"Error: the arguments of `{call['name']}` are not valid JSON ({error}). Send one JSON object."
    if not isinstance(arguments, dict):
        return f"Error: the arguments of `{call['name']}` must be a JSON object."
    if call["name"] == SEARCH_TOOL:
        info = box.meta_info(SEARCH_TOOL)
        if on_call:
            on_call(info, arguments, "read-only")
        text = box.search(arguments.get("query"), arguments.get("limit", SEARCH_LIMIT), arguments.get("full") is True)
        if on_result:
            on_result(info, text)
        return clip_middle(text, max_chars)
    target = box.find(arguments.get("name")) if isinstance(arguments.get("name"), str) else None
    if target is None:
        asked = arguments.get("name")
        near = difflib.get_close_matches(asked, list(box.deferred), n=3, cutoff=0.6) if isinstance(asked, str) else []
        return (f"Error: there is no tool called `{asked}` to run." if asked else "Error: tools_call needs the `name` of a tool.") \
            + (f" Did you mean: {', '.join(near)}?" if near else " Find it first with tools_search.")
    inner = arguments.get("arguments", {})
    if isinstance(inner, str):                                      # some models send the arguments as text
        try:
            inner = json.loads(inner) if inner.strip() else {}
        except ValueError as error:
            return f"Error: the `arguments` of `{target.name}` are not valid JSON ({error}). Send an object."
    if not isinstance(inner, dict):
        return f"Error: the `arguments` of `{target.name}` must be an object."
    return _one_call({"id": call["id"], "name": target.name, "arguments": json.dumps(inner)}, box, approve, on_call, on_result, max_chars, deferred_ok=True)


def _one_call(call, box, approve, on_call, on_result, max_chars, deferred_ok=False):
    if box.deferred and call["name"] in (SEARCH_TOOL, CALL_TOOL) and not deferred_ok:
        return _through(call, box, approve, on_call, on_result, max_chars)
    info = box.tools.get(call["name"]) or (box.deferred.get(call["name"]) if deferred_ok else None)
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
