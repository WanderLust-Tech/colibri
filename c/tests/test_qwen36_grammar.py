"""A request's grammar on Qwen3.6 (and so Qwen3-Coder): the engine limits its answer to it (c/constrain.h).

    python -m unittest tests.test_qwen36_grammar

Runs the real engine against the tiny random-init fixture the oracle job builds (tools/make_qwen36_tiny.py and
tools/convert_qwen36.py), with the byte-level tokenizer the prefix tests give it. A random model left alone writes noise, so
what comes back being JSON, judged by Python's parser and not by the engine's own grammar walker, is the engine's doing.
"""
import json
import os
import re
import subprocess
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import READY, ensure_byte_tokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
FIXTURE = Path(os.environ.get("QWEN36_TINY", HERE / "qwen36_tiny_c"))
EOS = "300"        # an id inside the fixture's 320-token vocabulary that no byte token uses: the end of the turn


def max_whitespace_run(text):
    """The longest run of JSON whitespace outside the strings of `text` (a string's own spaces are the model's content)."""
    longest = run = 0
    inside = escaped = False
    for ch in text:
        if inside:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                inside = False
            continue
        if ch == '"':
            inside = True
            run = 0
        elif ch in " \t\n\r":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


class Engine:
    """A serving engine; every request's frames are read until its DONE or its ERROR."""

    def __init__(self, env=None, slots=1):
        full = dict(os.environ, SNAP=str(FIXTURE), SERVE="1", COLI_DENSE_I8="0", KV_SLOTS=str(slots), **(env or {}))
        self.p = subprocess.Popen([str(ENGINE), "8", "8"], env=full, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=0)
        self.err = []
        threading.Thread(target=lambda: [self.err.append(line) for line in iter(self.p.stderr.readline, b"")],
                         daemon=True).start()
        self.caps = {}
        self.next_id = 1
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited before READY:\n" + b"".join(self.err).decode("utf-8", "replace")[-2000:])
            if line.startswith(b"CAPS "):
                self.caps = dict(f.split("=") for f in line.decode().split()[1:])
            if line.startswith(b"STAT"):
                break

    def submit(self, prompt=b"hello", grammar=None, max_tok=40, temp=1.0, top_p=1.0, slot=0):
        rid = str(self.next_id)
        self.next_id += 1
        g = grammar.encode("utf-8") if grammar else b""
        head = f"SUBMIT {rid} {slot} {len(prompt)} {max_tok} {temp} {top_p}" + (f" {len(g)}" if g else "")
        self.p.stdin.write(head.encode() + b"\n" + prompt + g + b"\n")
        self.p.stdin.flush()
        return rid

    def read_frame(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("engine closed:\n" + b"".join(self.err).decode("utf-8", "replace")[-2000:])
        parts = line.decode("latin-1").rstrip("\n").split(" ")
        body = b""
        if parts[0] == "DATA":
            body, want = b"", int(parts[2])
            while len(body) < want:                 # an unbuffered pipe may hand over part of a payload: read until it is whole
                piece = self.p.stdout.read(want - len(body))
                if not piece:
                    raise RuntimeError("engine closed in the middle of a DATA frame")
                body += piece
            self.p.stdout.readline()
        return parts[0], parts[1:], body

    def collect(self, *rids):
        """{id: {"data", "done", "error"}} once every id has its DONE or ERROR."""
        out = {r: {"data": b"", "done": None, "error": None} for r in rids}
        pending = set(rids)
        while pending:
            kind, args, body = self.read_frame()
            if not args or args[0] not in out:
                continue
            entry = out[args[0]]
            if kind == "DATA":
                entry["data"] += body
            elif kind == "DONE":
                entry["done"] = args
                pending.discard(args[0])
            elif kind == "ERROR":
                entry["error"] = " ".join(args[1:])
                pending.discard(args[0])
        return out

    def ask(self, **kw):
        rid = self.submit(**kw)
        return self.collect(rid)[rid]

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        for stream in (self.p.stdout, self.p.stderr):
            try:
                stream.close()
            except Exception:
                pass


# The start of what the schema allows, with whitespace anywhere between its tokens (a random model spends tokens on it)
PREFIX = re.compile(r'\A\s*(\{\s*("name"\s*(:.*)?)?)?\Z', re.DOTALL)


def limited(done):
    return done[-1] == "1"          # DONE <id> STAT gen tps hit rss np limited


@unittest.skipUnless(ENGINE.exists(), "qwen36 is not built")
@unittest.skipUnless((FIXTURE / "model-globals.safetensors").exists(),
                     "tiny qwen36 container is absent (tools/make_qwen36_tiny.py, tools/convert_qwen36.py)")
class Qwen36GrammarTest(unittest.TestCase):
    SCHEMA = json.dumps({"type": "object", "properties": {"name": {"type": "string"}, "n": {"type": "integer"},
                                                          "ok": {"type": "boolean"}},
                         "required": ["name", "n", "ok"]})

    @classmethod
    def setUpClass(cls):
        ensure_byte_tokenizer(FIXTURE)
        sys.path.insert(0, str(HERE))
        import openai_server
        cls.generic = openai_server.GENERIC_JSON_GBNF

    def engine(self, **kw):
        e = Engine(**kw)
        self.addCleanup(e.close)
        return e

    def answers(self, e, grammar, count, max_tok=160):
        out = []
        for _ in range(count):
            r = e.ask(grammar=grammar, max_tok=max_tok)
            self.assertIsNone(r["error"])
            if not limited(r["done"]):
                # strict: the grammar keeps a string's bytes well-formed UTF-8 (an answer cut by max_tokens may stop inside a character)
                out.append(r["data"].decode("utf-8"))
        return out

    def test_the_handshake_says_the_engine_takes_grammars(self):
        self.assertEqual(self.engine().caps["grammar"], "1")

    def test_a_random_model_under_the_json_grammar_writes_json(self):
        e = self.engine(env={"Q36_EOS": EOS})
        done = self.answers(e, self.generic, 40)
        self.assertGreater(len(done), 5, "too few answers ended by themselves for the check to mean anything")
        for text in done:
            self.assertIsInstance(json.loads(text), dict)     # json_object: an object at the top
            self.assertLessEqual(max_whitespace_run(text), 20)
        self.assertGreater(len(set(done)), len(done) // 2, "temperature 1 should make different answers")

    def test_a_random_model_under_a_schema_writes_that_shape(self):
        e = self.engine(env={"Q36_EOS": EOS})
        done = self.answers(e, self.SCHEMA, 40)
        self.assertGreater(len(done), 5)
        for text in done:
            value = json.loads(text)
            self.assertLessEqual(max_whitespace_run(text), 20)
            self.assertEqual(list(value), ["name", "n", "ok"])
            self.assertIsInstance(value["name"], str)
            self.assertTrue(isinstance(value["n"], int) and not isinstance(value["n"], bool))
            self.assertIsInstance(value["ok"], bool)

    def test_greedy_under_a_grammar_is_repeatable_and_obeys_it(self):
        e = self.engine(env={"Q36_EOS": EOS})
        first = e.ask(grammar=self.SCHEMA, max_tok=120, temp=0.0)
        again = e.ask(grammar=self.SCHEMA, max_tok=120, temp=0.0)
        self.assertEqual(first["data"], again["data"])
        text = first["data"].decode("utf-8", "surrogateescape")
        # a greedy random model that prefers whitespace may not live on it: a run is bounded, so the object opens
        self.assertRegex(text, PREFIX)
        self.assertLessEqual(max_whitespace_run(text), 20)
        self.assertIn("{", text)
        if not limited(first["done"]):
            json.loads(text)

    def test_gbnf_text_and_a_word_that_must_end(self):
        e = self.engine(env={"Q36_EOS": EOS})
        r = e.ask(grammar='root ::= "yes" | "no"', max_tok=20)
        self.assertIn(r["data"], (b"yes", b"no"))
        self.assertFalse(limited(r["done"]))                  # it ended itself: the end token is allowed once the word is whole
        self.assertEqual(e.ask(grammar='root ::= "ab"', max_tok=20)["data"], b"ab")

    def test_a_vocabulary_with_no_end_token_cannot_finish_an_answer_and_says_so(self):
        e = self.engine()                                     # the fixture's end-of-turn id is outside its 320 tokens
        r = e.ask(grammar='root ::= "ab"', max_tok=20)
        self.assertEqual(r["data"], b"ab")
        self.assertTrue((r["error"] or "").startswith("GRAMMAR_STUCK"), r)
        self.assertIsNotNone(e.ask(max_tok=3)["done"])        # and the engine goes on

    def test_a_request_without_a_grammar_is_not_touched(self):
        e = self.engine(env={"Q36_EOS": EOS})
        plain = e.ask(max_tok=8, temp=0.0)
        e.ask(grammar=self.generic, max_tok=8)
        self.assertEqual(e.ask(max_tok=8, temp=0.0)["data"], plain["data"])

    def test_a_grammar_that_does_not_compile_is_refused_by_id_and_the_engine_goes_on(self):
        e = self.engine(env={"Q36_EOS": EOS})
        for bad in ("this is not a grammar", '{"type": "object", "properties": {"a": {"type": "mystery"}}, "required": ["a"]}'):
            r = e.ask(grammar=bad, max_tok=4)
            self.assertTrue((r["error"] or "").startswith("BAD_GRAMMAR"), (bad, r))
        self.assertIsNotNone(e.ask(max_tok=3)["done"])

    def test_a_grammar_beside_a_plain_request_in_the_other_slot(self):
        e = self.engine(env={"Q36_EOS": EOS}, slots=2)
        alone = e.ask(max_tok=10, temp=0.0, slot=1)["data"]
        g = e.submit(grammar=self.SCHEMA, max_tok=60, slot=0)
        p = e.submit(max_tok=10, temp=0.0, slot=1)
        got = e.collect(g, p)
        self.assertEqual(got[p]["data"], alone)               # the other conversation says what it says alone
        self.assertIsNone(got[g]["error"])
        self.assertRegex(got[g]["data"].decode("utf-8", "surrogateescape"), PREFIX)

    def test_a_stuck_grammar_in_one_slot_leaves_the_other_alone(self):
        e = self.engine(slots=2)                              # no end token: "ab" cannot be finished
        alone = e.ask(max_tok=10, temp=0.0, slot=1)["data"]
        g = e.submit(grammar='root ::= "ab"', max_tok=20, slot=0)
        p = e.submit(max_tok=10, temp=0.0, slot=1)
        got = e.collect(g, p)
        self.assertTrue((got[g]["error"] or "").startswith("GRAMMAR_STUCK"))
        self.assertEqual(got[p]["data"], alone)
        self.assertIsNotNone(got[p]["done"])
        self.assertIsNotNone(e.ask(max_tok=3, slot=0)["done"])   # the slot is free again

    def test_a_grammar_with_logprobs_is_refused(self):
        e = self.engine(env={"Q36_EOS": EOS})
        g = b'root ::= "ab"'
        prompt = b"hello"
        e.p.stdin.write(f"SUBMIT 9 0 {len(prompt)} 4 0 1 {len(g)} logprobs=3\n".encode() + prompt + g + b"\n")
        e.p.stdin.flush()
        r = e.collect("9")["9"]
        self.assertTrue((r["error"] or "").startswith("BAD_GRAMMAR"), r)


if __name__ == "__main__":
    unittest.main()
