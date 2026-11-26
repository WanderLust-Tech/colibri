"""A small MCP server over stdio for the client's tests (not a test itself).

    python mcp_fake_server.py [noise] [crash-on-call] [page] [slow] [bad-init] [asks-ping] [stderr-on-start] [exit-on-start]

Tools: `echo` (text), `add` (structured), `fail` (isError), `slow` (sleeps), `crash` (exits), `image` (an image block).
"""
import json
import sys
import time

FLAGS = set(sys.argv[1:])


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


TOOLS = [
    {"name": "echo", "description": "Say the text back", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
     "annotations": {"readOnlyHint": True}},
    {"name": "add", "description": "Add two numbers", "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}},
    {"name": "fail", "description": "Always an error", "inputSchema": {"type": "object"}},
    {"name": "slow", "description": "Sleeps", "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}}},
    {"name": "crash", "description": "Exits", "inputSchema": {"type": "object"}},
    {"name": "image", "description": "An image", "inputSchema": {"type": "object"}},
]

if "stderr-on-start" in FLAGS or "exit-on-start" in FLAGS:
    sys.stderr.write("fake server: the database file is missing\n")
    sys.stderr.flush()
if "exit-on-start" in FLAGS:
    sys.exit(3)
if "noise" in FLAGS:
    sys.stdout.write("starting up, not a message\n")
    sys.stdout.flush()

for line in sys.stdin:
    if not line.strip():
        continue
    message = json.loads(line)
    method, ident = message.get("method"), message.get("id")
    if method == "notifications/cancelled":
        sys.stderr.write(f"fake server: cancelled {message['params']['requestId']}\n")
        sys.stderr.flush()
        continue
    if method is None and ident == 900:
        sys.stderr.write("fake server: ping answered\n" if "result" in message else "fake server: ping refused\n")
        sys.stderr.flush()
        continue
    if method is None and ident == 901:
        sys.stderr.write("fake server: sampling refused: " + str((message.get("error") or {}).get("code")) + "\n")
        sys.stderr.flush()
        continue
    if method == "initialize":
        if "bad-init" in FLAGS:
            send({"jsonrpc": "2.0", "id": ident, "error": {"code": -32602, "message": "unsupported protocol version"}})
        else:
            send({"jsonrpc": "2.0", "id": ident, "result": {"protocolVersion": message["params"]["protocolVersion"], "capabilities": {"tools": {}},
                                                             "serverInfo": {"name": "fake", "version": "0.1"}, "instructions": "be careful"}})
    elif method == "tools/list":
        cursor = (message.get("params") or {}).get("cursor")
        if "page" in FLAGS:
            if not cursor:
                send({"jsonrpc": "2.0", "id": ident, "result": {"tools": TOOLS[:2], "nextCursor": "p2"}})
            else:
                send({"jsonrpc": "2.0", "id": ident, "result": {"tools": TOOLS[2:]}})
        else:
            send({"jsonrpc": "2.0", "id": ident, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        name, args = message["params"]["name"], message["params"].get("arguments") or {}
        if "asks-ping" in FLAGS:
            send({"jsonrpc": "2.0", "id": 900, "method": "ping"})
            send({"jsonrpc": "2.0", "id": 901, "method": "sampling/createMessage", "params": {}})
        if name == "echo":
            send({"jsonrpc": "2.0", "id": ident, "result": {"content": [{"type": "text", "text": args.get("text", "")}]}})
        elif name == "add":
            send({"jsonrpc": "2.0", "id": ident, "result": {"content": [], "structuredContent": {"sum": args.get("a", 0) + args.get("b", 0)}}})
        elif name == "fail":
            send({"jsonrpc": "2.0", "id": ident, "result": {"content": [{"type": "text", "text": "it did not work"}], "isError": True}})
        elif name == "slow":
            time.sleep(float(args.get("seconds", 5)))
            send({"jsonrpc": "2.0", "id": ident, "result": {"content": [{"type": "text", "text": "done"}]}})
        elif name == "crash":
            sys.stderr.write("fake server: segmentation fault\n")
            sys.stderr.flush()
            sys.exit(7)
        elif name == "image":
            send({"jsonrpc": "2.0", "id": ident, "result": {"content": [{"type": "image", "mimeType": "image/png", "data": "AAAA" * 100},
                                                                         {"type": "text", "text": "a picture"}]}})
        else:
            send({"jsonrpc": "2.0", "id": ident, "error": {"code": -32602, "message": f"unknown tool {name}"}})
    elif method == "ping":
        send({"jsonrpc": "2.0", "id": ident, "result": {}})
    elif ident is not None:
        send({"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": "no such method"}})
