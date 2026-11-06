"""The Gemma 4 engine (gemma4.c) against transformers, and its serve protocol.

    python -m unittest tests.test_gemma4

Needs the built engine (make gemma4); skipped without it. The model is c/gemma4_tiny, a random
Gemma 4 built by tools/make_gemma4_tiny.py: sliding and full layers, the keys-equal-values layers,
proportional rope, the dense MLP beside the experts, the router scale, the layer scalar, the soft
cap. Its ref.json holds what transformers computes for it in float32, with the weights already on
the container's quantization grid, so the numbers must agree to float rounding.

What is held:
  - logits: every prompt, one row at a time, in chunks, with a ring that wraps (window 6) and an
    expert cache too small for the experts a layer wants;
  - the container writer: int8 rows and the planar int4 layout round-trip, and the nibbles sit where
    expert_ffn.h reads them;
  - the serve protocol: the handshake, a greedy reply equal to the offline greedy continuation, the
    prompt counted by the tokenizer, KV reuse when a turn extends the last one (and the same text
    with reuse off), two slots, a busy slot, a bad slot, CANCEL, a prompt that does not fit.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(C_DIR, "tools"))
EXE = os.environ.get("GEMMA4_EXE") or next(
    (p for p in (os.path.join(C_DIR, "gemma4.exe"), os.path.join(C_DIR, "gemma4")) if os.path.isfile(p)), None)
TINY = os.environ.get("GEMMA4_TINY", os.path.join(C_DIR, "gemma4_tiny"))
REF = os.path.join(TINY, "ref.json")


def offline(ids, greedy=0, extra=None):
    """The engine's test mode: logits (and a greedy continuation) of a prompt of token ids."""
    env = dict(os.environ, SNAP=TINY, G4_TEST="1", G4_IDS=",".join(map(str, ids)), G4_GREEDY=str(greedy),
               OMP_NUM_THREADS="2")
    env.update(extra or {})
    cap = env.pop("CAP", "128")
    done = subprocess.run([EXE, cap], env=env, capture_output=True, text=True, timeout=120)
    if done.returncode:
        raise AssertionError(done.stdout[-400:] + done.stderr[-1500:])
    return json.loads(done.stdout.strip().splitlines()[-1])


@unittest.skipUnless(EXE and os.path.isfile(REF), "needs the built gemma4 and c/gemma4_tiny")
class LogitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REF, encoding="utf-8") as handle:
            cls.ref = json.load(handle)

    def check(self, **extra):
        for case in self.ref["cases"]:
            out = offline(case["ids"], len(case["greedy"]), extra)
            worst = max(abs(a - b) for a, b in zip(out["last"], case["last_logits"]))
            self.assertLess(worst, 2e-3, f"last-row logits differ by {worst} ({len(case['ids'])} tokens, {extra})")
            rows = max(abs(a - b) for ra, rb in zip(out["rows"], case["logit_sample"]) for a, b in zip(ra, rb))
            self.assertLess(rows, 2e-3, f"row logits differ by {rows} ({extra})")
            self.assertEqual(out["top"], case["all_top"], extra)
            self.assertEqual(out["greedy"], case["greedy"], extra)

    def test_whole_prompt_at_once(self):
        self.check()

    def test_a_row_at_a_time(self):
        self.check(G4_CHUNK="1")

    def test_in_chunks_with_a_ring_that_wraps(self):
        self.check(G4_CHUNK="7")           # ring = window 6 + 7 rows: the 41-token prompt wraps it three times

    def test_with_an_expert_cache_smaller_than_what_a_layer_wants(self):
        self.check(CAP="3", G4_CHUNK="5")  # 8 experts, 3 slots: groups, eviction, reloads

    def test_the_logits_are_not_trivial(self):
        case = self.ref["cases"][0]
        spread = max(case["last_logits"]) - min(case["last_logits"])
        self.assertGreater(spread, 2.0)
        self.assertGreater(len(set(case["all_top"])), 10)


class ContainerTest(unittest.TestCase):
    def test_int8_rows_round_trip(self):
        import numpy as np
        import convert_gemma4 as cv
        rng = np.random.default_rng(1)
        w = rng.normal(0, 1, (5, 40)).astype(np.float32)
        q, s = cv.quantize_int8_rows(w)
        back = cv.dequantize_int8_rows(q, s)
        self.assertLess(float(np.abs(back - w).max()), float(np.abs(w).max()) / 127)
        again, s2 = cv.quantize_int8_rows(back)
        self.assertTrue((again == q).all())

    def test_int4_planar_layout_is_what_expert_ffn_reads(self):
        import numpy as np
        import convert_gemma4 as cv
        w = np.zeros((1, 128), dtype=np.float32)
        w[0, 3] = 7.0          # element 3 of block 0: low nibble of byte 3
        w[0, 35] = -7.0        # element 35 of block 0: high nibble of byte 3
        w[0, 64 + 0] = 3.0     # block 1, element 0
        packed, scale = cv.quantize_int4_planar(w)
        self.assertEqual(packed.shape, (1, 64))
        self.assertEqual(packed[0, 3] & 0x0F, 7 + 8)
        self.assertEqual(packed[0, 3] >> 4, -7 + 8)
        self.assertEqual(packed[0, 32 + 0] & 0x0F, 7 + 8)     # block 1 starts at byte 32; its element 0 is 3/(3/7) = 7
        self.assertAlmostEqual(float(scale[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(scale[0, 1]), 3.0 / 7.0, places=6)
        self.assertTrue((cv.dequantize_int4_planar(packed, scale) == w).all() or
                        abs(cv.dequantize_int4_planar(packed, scale) - w).max() < 1e-6)

    def test_int4_is_exact_on_its_own_output(self):
        import numpy as np
        import convert_gemma4 as cv
        rng = np.random.default_rng(2)
        w = rng.normal(0, 1, (6, 192)).astype(np.float32)
        p, s = cv.quantize_int4_planar(w)
        back = cv.dequantize_int4_planar(p, s)
        p2, s2 = cv.quantize_int4_planar(back)
        self.assertTrue((p == p2).all())
        self.assertTrue(np.allclose(s, s2, rtol=1e-6))

    def test_an_unknown_tensor_stops_the_conversion(self):
        import convert_gemma4 as cv
        self.assertEqual(cv.classify("model.language_model.layers.3.self_attn.q_proj.weight"), "matrix")
        self.assertEqual(cv.classify("model.language_model.layers.3.experts.gate_up_proj"), "expert")
        self.assertEqual(cv.classify("model.vision_tower.encoder.layers.0.input_layernorm.weight"), "skip")
        self.assertIsNone(cv.classify("model.language_model.layers.3.self_attn.mystery.weight"))


class Engine:
    """One serving engine: frames come off stdout on a thread, so a test can write while it runs."""

    def __init__(self, slots=1, extra=None, cap="128", ctx="4096"):
        env = dict(os.environ, SNAP=TINY, SERVE="1", SERVE_BATCH="1", KV_SLOTS=str(slots), CTX=ctx,
                   OMP_NUM_THREADS="2", COLI_PREFIX_LOG="1")
        env.update(extra or {})
        self.proc = subprocess.Popen([EXE, cap], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        self.frames, self.log = queue.Queue(), []
        self.startup = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        self.next_id = 100
        while True:                                  # the handshake: READY, CAPS, STAT, HWINFO, TIERS, EMAP
            line = self.get()
            self.startup.append(line)
            if line[0] == "EMAP":
                break

    def _read_err(self):
        for line in iter(self.proc.stderr.readline, b""):
            self.log.append(line.decode("utf-8", "replace").rstrip())

    def _read(self):
        out = self.proc.stdout
        buf = b""

        def more():
            chunk = out.read(1 << 16)
            return chunk

        while True:
            while b"\n" not in buf:
                chunk = more()
                if not chunk:
                    self.frames.put(("EOF", [], b""))
                    return
                buf += chunk
            line, buf = buf.split(b"\n", 1)
            text = line.decode("utf-8", "replace")
            parts = text.split(" ")
            kind = parts[0].replace("\x01", "")
            body = b""
            if kind == "DATA":
                need = int(parts[2]) + 1
                while len(buf) < need:
                    chunk = more()
                    if not chunk:
                        self.frames.put(("EOF", [], b""))
                        return
                    buf += chunk
                body, buf = buf[:int(parts[2])], buf[need:]
            self.frames.put((kind, parts[1:], body))

    def get(self, timeout=60):
        return self.frames.get(timeout=timeout)

    def submit(self, text, slot=0, max_tokens=6, temperature=0.0, top_p=1.0, rid=None):
        rid = rid or str(self.next_id)
        self.next_id += 1
        payload = text.encode("utf-8")
        self.proc.stdin.write(f"SUBMIT {rid} {slot} {len(payload)} {max_tokens} {temperature} {top_p}\n".encode() + payload + b"\n")
        self.proc.stdin.flush()
        return rid

    def cancel(self, rid):
        self.proc.stdin.write(f"CANCEL {rid}\n".encode())
        self.proc.stdin.flush()

    def collect(self, rid, timeout=60):
        """Frames of one request until its DONE or ERROR."""
        data, accept, done, error = b"", None, None, None
        others = []
        while True:
            kind, args, body = self.get(timeout)
            if kind in ("DATA", "ACCEPT", "DONE", "ERROR") and args and args[0] == rid:
                if kind == "DATA":
                    data += body
                elif kind == "ACCEPT":
                    accept = int(args[1])
                elif kind == "DONE":
                    done = args
                    break
                else:
                    error = " ".join(args[1:])
                    break
            else:
                others.append((kind, args))
        return {"data": data, "accept": accept, "done": done, "error": error, "others": others}

    def ask(self, text, **kw):
        return self.collect(self.submit(text, **kw))

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
        for stream in (self.proc.stdout, self.proc.stderr):
            try:
                stream.close()
            except Exception:
                pass


@unittest.skipUnless(EXE and os.path.isfile(REF), "needs the built gemma4 and c/gemma4_tiny")
class ServeTest(unittest.TestCase):
    TURN = "<|turn>user\nthe quick brown fox<turn|>\n<|turn>model\n"

    @classmethod
    def setUpClass(cls):
        with open(REF, encoding="utf-8") as handle:
            cls.ref = json.load(handle)
        cls.ids = {e["text"]: e["ids"] for e in cls.ref["encodings"]}

    def engine(self, **kw):
        e = Engine(**kw)
        self.addCleanup(e.close)
        return e

    def test_handshake(self):
        e = self.engine()
        kinds = [line[0] for line in e.startup]
        self.assertEqual(kinds[0], "READY")
        self.assertIn(("CAPS", ["vision=0"]), [(k, a) for k, a, _ in e.startup])
        for kind in ("STAT", "HWINFO", "TIERS", "EMAP"):
            self.assertIn(kind, kinds)

    def test_the_prompt_is_counted_by_the_tokenizer(self):
        e = self.engine()
        for item in self.ref["encodings"]:
            out = e.ask(item["text"], max_tokens=1)
            self.assertEqual(out["accept"], len(item["ids"]), item["text"])
            self.assertEqual(int(out["done"][6]), len(item["ids"]))

    def test_a_greedy_reply_is_the_offline_greedy_continuation(self):
        ids = self.ids["the quick brown fox"]
        want = offline(ids, 6)
        e = self.engine()
        out = e.ask("the quick brown fox", max_tokens=6)
        self.assertEqual(out["data"], bytes.fromhex(want["greedy_hex"]))
        self.assertEqual(int(out["done"][2]), 6)            # tokens generated
        self.assertEqual(out["done"][7], "1")              # stopped by the length limit
        self.assertEqual(e.ask("the quick brown fox", max_tokens=6)["data"], out["data"])

    def test_a_turn_that_extends_the_last_reuses_its_cache_and_says_the_same(self):
        first = "<|turn>user\nhello there<turn|>"
        second = first + "\n<|turn>user\nthe quick brown fox<turn|>\n<|turn>model\n"
        with_reuse = self.engine()
        with_reuse.ask(first, max_tokens=1)
        a = with_reuse.ask(second, max_tokens=6)
        self.assertTrue(any("[PREFIX] reusing" in line for line in with_reuse.log), with_reuse.log)
        without = self.engine(extra={"COLI_KV_PREFIX": "0"})
        without.ask(first, max_tokens=1)
        b = without.ask(second, max_tokens=6)
        self.assertFalse(any("reusing" in line for line in without.log))
        self.assertEqual(a["data"], b["data"])
        self.assertEqual(a["data"], with_reuse.ask(second, max_tokens=6)["data"])

    def test_two_conversations_keep_their_own_cache_and_may_overlap(self):
        e = self.engine(slots=2)
        alone = {t: self.engine().ask(t, max_tokens=5)["data"] for t in ("the quick brown fox", "hello there")}
        r1 = e.submit("the quick brown fox", slot=0, max_tokens=5)
        r2 = e.submit("hello there", slot=1, max_tokens=5)
        seen, got = {}, {}
        while len(got) < 2:
            kind, args, body = e.get()
            if kind == "DATA":
                seen[args[0]] = seen.get(args[0], b"") + body
            elif kind == "DONE":
                got[args[0]] = args
        self.assertEqual(seen[r1], alone["the quick brown fox"])
        self.assertEqual(seen[r2], alone["hello there"])

    def test_a_busy_slot_and_a_bad_slot_are_refused_by_id(self):
        e = self.engine(slots=2)
        first = e.submit("the quick brown fox", slot=0, max_tokens=40)
        second = e.submit("hello there", slot=0, max_tokens=3)
        bad = e.submit("hello there", slot=7, max_tokens=3)
        errors = {}
        while len(errors) < 2:
            kind, args, _ = e.get()
            if kind == "ERROR":
                errors[args[0]] = " ".join(args[1:])
        self.assertEqual(errors[second], "SLOT_BUSY")
        self.assertIn("invalid cache slot", errors[bad])
        e.cancel(first)

    def test_cancel_ends_a_request_early_and_the_engine_goes_on(self):
        e = self.engine()
        rid = e.submit("the quick brown fox", max_tokens=400)
        kind = None
        while kind != "DATA":
            kind, args, _ = e.get()
        e.cancel(rid)
        out = e.collect(rid)
        self.assertLess(int(out["done"][2]), 400)
        self.assertEqual(out["done"][7], "0")              # not "stopped by the length limit"
        self.assertEqual(e.ask("hello there", max_tokens=2)["done"][2], "2")

    def test_a_prompt_that_does_not_fit_is_refused_with_the_frame_the_gateway_reads(self):
        e = self.engine(ctx="8")
        out = e.ask("the quick brown fox jumps over the lazy dog again and again", max_tokens=4)
        self.assertIsNone(out["done"])
        self.assertTrue(out["error"].startswith("CONTEXT_EXCEEDED"), out["error"])
        self.assertEqual(e.ask("hi", max_tokens=1)["done"][2], "1")

    def test_a_reply_may_not_pass_the_context(self):
        e = self.engine(ctx="12")
        out = e.ask("hello there", max_tokens=100)
        self.assertEqual(out["done"][7], "1")
        self.assertLessEqual(int(out["done"][6]) + int(out["done"][2]), 12)

    def test_end_of_input_is_a_clean_exit(self):
        e = self.engine()
        e.ask("hello there", max_tokens=1)
        e.proc.stdin.close()
        self.assertEqual(e.proc.wait(timeout=30), 0)


if __name__ == "__main__":
    unittest.main()
