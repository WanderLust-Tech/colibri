"""Progress of a tool call being written (openai_server.ToolProgress, header `X-Colibri-Progress: 1`).

The gateway holds a tool call back and parses it when generation ends, so a long call showed a client nothing for minutes.
A client that asks now gets an SSE comment about once a second: `: colibri-tool {"chars": N, "name": "write_file"}`. Nothing
else on the wire changes, for it or for a client that does not ask.

The first half tests the helper with a clock of its own; the second runs the gateway as a real subprocess against a mock engine
(the SERVE wire protocol) and reads the stream over HTTP. The mock engine is a python script: on POSIX it is run directly, on
Windows through a .cmd shim, so this runs on both.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server  # noqa: E402

SERVER = Path(__file__).resolve().parent.parent / "openai_server.py"
MODEL_ID = "glm-5.2-colibri"


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class ToolProgressUnit(unittest.TestCase):
    def make(self, names=lambda text: ["write_file"], gap=1.0):
        clock, wrote = FakeClock(), []
        return openai_server.ToolProgress(wrote.append, names, clock, gap), clock, wrote

    @staticmethod
    def parsed(wrote):
        out = []
        for line in wrote:
            self_prefix, _, body = line.partition(" ")
            assert self_prefix == "colibri-tool", line
            out.append(json.loads(body))
        return out

    def test_the_first_report_goes_out_at_once_and_later_ones_wait_for_the_gap(self):
        progress, clock, wrote = self.make()
        progress.feed(10, lambda: "reply")
        self.assertEqual(self.parsed(wrote), [{"chars": 10, "name": "write_file"}])
        clock.now += 0.4
        progress.feed(5, lambda: "reply")
        progress.feed(5, lambda: "reply")
        self.assertEqual(len(wrote), 1)                                   # inside the gap: counted, not said
        clock.now += 0.7
        progress.feed(1, lambda: "reply")
        self.assertEqual(self.parsed(wrote)[-1], {"chars": 21, "name": "write_file"})   # all the characters since the start
        self.assertEqual(len(wrote), 2)

    def test_a_call_that_does_not_parse_yet_is_reported_without_a_name(self):
        def refuses(text):
            raise ValueError("half a call")
        progress, clock, wrote = self.make(names=refuses)
        progress.feed(7, lambda: "<tool_call>wri")
        self.assertEqual(self.parsed(wrote), [{"chars": 7, "name": None}])
        progress2, _clock2, wrote2 = self.make(names=lambda text: [])
        progress2.feed(3, lambda: "x")
        self.assertEqual(self.parsed(wrote2), [{"chars": 3, "name": None}])

    def test_the_last_call_named_is_the_one_reported_and_the_reply_is_read_only_when_a_report_goes_out(self):
        reads = []

        def reply():
            reads.append(1)
            return "reply"
        progress, clock, wrote = self.make(names=lambda text: ["a", "b"])
        progress.feed(1, reply)
        progress.feed(1, reply)                                           # inside the gap: the reply is not even joined
        self.assertEqual(self.parsed(wrote), [{"chars": 1, "name": "b"}])
        self.assertEqual(len(reads), 1)

    def test_the_default_gap_is_one_second(self):
        self.assertEqual(openai_server.TOOL_PROGRESS_GAP, float(os.environ.get("COLI_TOOL_PROGRESS_GAP", "1.0")))


MOCK_ENGINE = r'''
import sys, os, time
out, inp = sys.stdout.buffer, sys.stdin.buffer
out.write(b"\x01\x01READY\x01\x01\n" + b"STAT 0 0 0 0 0\n"); out.flush()

def send(rid, text):
    data = text.encode("utf-8")
    out.write(("DATA %s %d\n" % (rid, len(data))).encode() + data + b"\n"); out.flush()

while True:
    line = inp.readline()
    if not line: break
    f = line.decode().strip().split()
    if not f or f[0] != "SUBMIT": continue
    rid, plen = f[1], int(f[3])
    prompt = inp.read(plen).decode("utf-8", "replace"); inp.read(1)
    if "write the file" in prompt:
        send(rid, "Sure. <tool_call>write_file<arg_key>path</arg_key><arg_value>a.txt</arg_value><arg_key>content</arg_key><arg_value>")
        for i in range(8):
            time.sleep(0.12)
            send(rid, "line %d of the file\n" % i)
        send(rid, "</arg_value></tool_call>")
    else:
        send(rid, "Just an answer.")
    out.write(("DONE %s STAT 5 1.0 50.0 10.0 42 0\n" % rid).encode()); out.flush()
'''

TOOLS = [{"type": "function", "function": {
    "name": "write_file", "description": "Write a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["path", "content"]}}}]


class ToolProgressE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}), encoding="utf-8")
        script = root / "mock_engine.py"
        script.write_text(MOCK_ENGINE)
        if os.name == "posix":
            script.write_text("#!" + sys.executable + "\n" + MOCK_ENGINE)
            script.chmod(0o755)
            engine = script
        else:
            engine = root / "mock_engine.cmd"
            engine.write_text(f'@"{sys.executable}" "{script}"\r\n')
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        env = dict(os.environ, COLI_TOOL_PROGRESS_GAP="0.05")
        env.pop("COLI_API_KEY", None)
        cls.server = subprocess.Popen([sys.executable, str(SERVER), "--model", str(root), "--engine", str(engine), "--port", str(cls.port)],
                                      env=env, stderr=subprocess.DEVNULL)
        cls.base = f"http://127.0.0.1:{cls.port}/v1"
        for _ in range(150):
            try:
                with urllib.request.urlopen(cls.base + "/models", timeout=2):
                    return
            except OSError:
                if cls.server.poll() is not None:
                    raise RuntimeError("gateway exited during startup")
                time.sleep(0.1)
        raise RuntimeError("gateway did not come up")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=10)
        cls.tmp.cleanup()

    def stream(self, text, headers=None, tools=True):
        body = {"model": MODEL_ID, "stream": True, "messages": [{"role": "user", "content": text}]}
        if tools:
            body["tools"] = TOOLS
        req = urllib.request.Request(self.base + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
        lines = []
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw in resp:
                lines.append(raw.decode().rstrip("\r\n"))
        events = [json.loads(line[6:]) for line in lines if line.startswith("data: ") and line != "data: [DONE]"]
        comments = [json.loads(line[len(": colibri-tool "):]) for line in lines if line.startswith(": colibri-tool ")]
        return lines, events, comments

    @staticmethod
    def tool_calls(events):
        return [d["tool_calls"] for e in events if e["choices"] for d in [e["choices"][0]["delta"]] if d.get("tool_calls")]

    def test_a_client_that_asks_hears_how_much_of_the_call_is_written_and_its_name(self):
        lines, events, comments = self.stream("please write the file", {"X-Colibri-Progress": "1"})
        self.assertGreaterEqual(len(comments), 3, comments)
        chars = [c["chars"] for c in comments]
        self.assertEqual(chars, sorted(chars), "never going back")
        self.assertGreater(chars[-1], chars[0])
        self.assertIn("write_file", [c["name"] for c in comments], comments)
        # the stream itself is as ever: one call, whole, with its arguments; the call's text is not in the content
        calls = self.tool_calls(events)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["function"]["name"], "write_file")
        arguments = json.loads(calls[0][0]["function"]["arguments"])
        self.assertEqual(arguments["path"], "a.txt")
        self.assertIn("line 7 of the file", arguments["content"])
        content = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e["choices"])
        self.assertNotIn("<tool_call>", content)
        self.assertIn("Sure.", content)
        self.assertTrue(all(not line.startswith(": ") or line.startswith(": colibri-tool ") for line in lines), "no other comments")

    def test_a_client_that_does_not_ask_gets_the_same_stream_without_a_single_comment(self):
        lines, events, comments = self.stream("please write the file")
        self.assertEqual(comments, [])
        self.assertFalse([line for line in lines if line.startswith(":")])
        self.assertEqual(len(self.tool_calls(events)), 1)

    def test_a_client_that_asks_for_something_else_than_one_is_not_asking(self):
        for value in ("0", "yes", ""):
            _lines, _events, comments = self.stream("please write the file", {"X-Colibri-Progress": value})
            self.assertEqual(comments, [], value)

    def test_nothing_is_said_when_no_tool_call_is_written(self):
        lines, events, comments = self.stream("say something", {"X-Colibri-Progress": "1"})
        self.assertEqual(comments, [])
        self.assertEqual(self.tool_calls(events), [])
        self.assertIn("Just an answer.", "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e["choices"]))

    def test_nothing_is_said_when_the_request_has_no_tools(self):
        _lines, _events, comments = self.stream("please write the file", {"X-Colibri-Progress": "1"}, tools=False)
        self.assertEqual(comments, [])


if __name__ == "__main__":
    unittest.main()
