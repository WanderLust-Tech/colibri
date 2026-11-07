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


def ensure_container():
    """The weights are not committed (no repo weights are): rebuild them from the fixed recipe, numpy only."""
    import glob
    if glob.glob(os.path.join(TINY, "model-*.safetensors")) or not os.path.isfile(REF):
        return True
    try:
        import make_gemma4_tiny
        make_gemma4_tiny.build_container(TINY)
        return True
    except ImportError:
        return False


HAVE_FIXTURE = bool(EXE) and os.path.isfile(REF) and ensure_container()


def offline(ids, greedy=0, extra=None, images=None):
    """The engine's test mode: logits (and a greedy continuation) of a prompt of token ids. images: (patch bytes,
    grid height, grid width) for each run of picture placeholders in the prompt."""
    import tempfile
    env = dict(os.environ, SNAP=TINY, G4_TEST="1", G4_IDS=",".join(map(str, ids)), G4_GREEDY=str(greedy),
               OMP_NUM_THREADS="2", G4_SLACK="0",          # no slack: the ring is window + chunk, so it wraps
               G4_IMG_TOKENS="1")                          # ... and no room kept for pictures, unless a test says so
    env.update(extra or {})
    cap = env.pop("CAP", "128")
    with tempfile.TemporaryDirectory() as tmp:
        specs = []
        for i, (patches, gh, gw) in enumerate(images or ()):
            path = os.path.join(tmp, f"picture{i}.bin")
            with open(path, "wb") as handle:
                handle.write(patches)
            specs.append(f"{path}|{gh}|{gw}")
        if specs:
            env["G4_IMAGES"] = ";".join(specs)
        done = subprocess.run([EXE, cap], env=env, capture_output=True, text=True, timeout=120)
    if done.returncode:
        raise AssertionError(done.stdout[-400:] + done.stderr[-1500:])
    return json.loads(done.stdout.strip().splitlines()[-1])


@unittest.skipUnless(HAVE_FIXTURE, "needs the built gemma4, c/gemma4_tiny and numpy")
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


@unittest.skipUnless(HAVE_FIXTURE, "needs the built gemma4, c/gemma4_tiny and numpy")
class BatchTest(unittest.TestCase):
    """A decode step of several conversations in one pass over the weights (the serve loop's way of decoding): each
    conversation must get, bit for bit, the logits it gets stepping alone. Not a tolerance: a reply must not depend
    on who else is asking."""

    @classmethod
    def setUpClass(cls):
        with open(REF, encoding="utf-8") as handle:
            cls.ref = json.load(handle)
        a, b, c = (case["ids"] for case in cls.ref["cases"])
        cls.prompts = [a, b, c, c[:17], a[:9] + [123, 45], b + [7], c[:30]]       # 23, 4, 41, 17, 11, 5, 30 tokens

    def batch(self, count, threads="2", **extra):
        prompts = self.prompts[:count]
        env = dict(os.environ, SNAP=TINY, G4_TEST="1", G4_IDS=",".join(map(str, prompts[0])),
                   G4_IDS2=";".join(",".join(map(str, p)) for p in prompts[1:]), G4_GREEDY="8",
                   OMP_NUM_THREADS=threads, G4_SLACK="0", G4_IMG_TOKENS="1")
        env.update(extra)
        cap = env.pop("CAP", "128")
        done = subprocess.run([EXE, cap], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout[-300:] + done.stderr[-1000:])
        return json.loads(done.stdout.strip().splitlines()[-1])["batch"]

    def check(self, count, **extra):
        out = self.batch(count, **extra)
        self.assertEqual(out["conversations"], count)
        self.assertEqual(out["max_diff"], 0.0, f"a batch of {count} changed a logit by {out['max_diff']} ({extra})")
        self.assertEqual(out["identical"], 1)

    def test_two_conversations(self):
        self.check(2)

    def test_seven_conversations_of_different_lengths(self):
        self.check(7)          # more than the four rows the prefill kernel takes at once: the single-row kernel for each

    def test_with_one_thread_and_many(self):
        self.check(5, threads="1")
        self.check(5, threads="8")

    def test_with_an_expert_cache_smaller_than_the_experts_the_batch_wants(self):
        self.check(5, CAP="3")
        self.check(7, CAP="5")


def picture(ref, index):
    """(patch bytes, grid height, grid width) of picture `index` of the reference."""
    p = ref["vision"]["images"][index]
    return bytes.fromhex(p["hex"]), p["gh"], p["gw"]


@unittest.skipUnless(HAVE_FIXTURE, "needs the built gemma4, c/gemma4_tiny and numpy")
class VisionTest(unittest.TestCase):
    """The vision tower and the pictures inside a prompt, against transformers' Gemma4ForConditionalGeneration: the
    soft tokens the tower makes, and the logits of whole prompts with one or two pictures in them. The reference pads
    the pictures to the longest and masks the padding; the engine pads nothing. The sliding layers (window 6) let a
    picture's rows see each other, and the 12-token picture is longer than the window."""

    @classmethod
    def setUpClass(cls):
        with open(REF, encoding="utf-8") as handle:
            cls.ref = json.load(handle)
        cls.cases = cls.ref["vision"]["cases"]

    def check(self, features=True, **extra):
        extra = dict({"G4_IMG_TOKENS": "12"}, **extra)
        for case in self.cases:
            out = offline(case["ids"], len(case["greedy"]), extra, [picture(self.ref, i) for i in case["images"]])
            if features:
                worst = max(abs(a - b) for fa, fb in zip(out["features"], case["features"]) for a, b in zip(fa, fb))
                self.assertLess(worst, 1e-3, f"soft tokens differ by {worst}")
            worst = max(abs(a - b) for a, b in zip(out["last"], case["last_logits"]))
            self.assertLess(worst, 2e-3, f"last-row logits differ by {worst} ({case['spans']}, {extra})")
            rows = max(abs(a - b) for ra, rb in zip(out["rows"], case["logit_sample"]) for a, b in zip(ra, rb))
            self.assertLess(rows, 2e-3, f"row logits differ by {rows} ({extra})")
            self.assertEqual(out["top"], case["all_top"], extra)
            self.assertEqual(out["greedy"], case["greedy"], extra)

    def test_whole_prompt_at_once(self):
        self.check()

    def test_chunks_never_split_a_picture(self):
        self.check(features=False, G4_CHUNK="5")        # a picture of 6 or 12 rows does not fit: it goes alone, whole

    def test_a_row_at_a_time_takes_each_picture_whole(self):
        self.check(features=False, G4_CHUNK="1")

    def test_with_a_ring_that_wraps_and_a_small_expert_cache(self):
        self.check(features=False, G4_CHUNK="3", CAP="3")

    def test_the_picture_changes_the_answer(self):
        """Not trivial: the same ids with the picture's colours inverted give different logits."""
        case = self.cases[0]
        patches, gh, gw = picture(self.ref, 0)
        extra = {"G4_IMG_TOKENS": "12"}
        a = offline(case["ids"], 0, extra, [(patches, gh, gw)])
        b = offline(case["ids"], 0, extra, [(bytes(255 - v for v in patches), gh, gw)])
        self.assertGreater(max(abs(x - y) for x, y in zip(a["last"], b["last"])), 0.05)
        self.assertGreater(max(abs(v) for v in case["features"][0]), 0.5)

    def test_a_prompt_with_pictures_needs_them(self):
        case = self.cases[0]
        with self.assertRaises(AssertionError) as caught:
            offline(case["ids"], 0, {"G4_IMG_TOKENS": "12"}, None)
        self.assertIn("G4_IMAGES", str(caught.exception))
        with self.assertRaises(AssertionError) as caught:
            offline(case["ids"], 0, {"G4_IMG_TOKENS": "12", "G4_VISION": "0"}, [picture(self.ref, 0)])
        self.assertIn("no vision tower", str(caught.exception))


try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
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
        self.assertEqual(cv.classify("model.vision_tower.encoder.layers.0.input_layernorm.weight"), "vexact")
        self.assertEqual(cv.classify("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"), "vmatrix")
        self.assertEqual(cv.classify("model.vision_tower.patch_embedder.position_embedding_table"), "vexact")
        self.assertEqual(cv.classify("model.embed_vision.embedding_projection.weight"), "vmatrix")
        self.assertEqual(cv.classify("model.audio_tower.layers.0.self_attn.q_proj.weight"), "skip")
        self.assertIsNone(cv.classify("model.vision_tower.encoder.layers.0.mystery.weight"))
        self.assertIsNone(cv.classify("model.language_model.layers.3.self_attn.mystery.weight"))

    def source(self, directory):
        """A float checkpoint of the tiny model, text and vision, in `directory`."""
        import shutil
        import make_gemma4_tiny as tiny
        state = tiny.random_weights()
        state.update(tiny.random_vision_weights())
        for name in ("config.json", "generation_config.json", "tokenizer.json"):
            shutil.copyfile(os.path.join(TINY, name), os.path.join(directory, name))
        tiny.save_f32_safetensors(os.path.join(directory, "model.safetensors"), state)
        return state

    def names(self, directory):
        with open(os.path.join(directory, "model.safetensors.index.json"), encoding="utf-8") as handle:
            return json.load(handle)["weight_map"]

    def test_the_vision_tower_is_carried_as_int8_rows_and_exact_vectors(self):
        import tempfile
        import numpy as np
        import convert_gemma4 as cv
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            state = self.source(src)
            meta = cv.convert(src, out, log=lambda *_: None)
            self.assertEqual(meta["vision"]["hidden"], 32)
            self.assertEqual(meta["vision"]["layers"], 2)
            self.assertEqual(meta["skipped_tensors"], 0)
            names = self.names(out)
            ck = cv.Checkpoint(out)
            try:
                for name in ("model.vision_tower.encoder.layers.1.mlp.down_proj.linear.weight",
                             "model.vision_tower.patch_embedder.input_proj.weight",
                             "model.embed_vision.embedding_projection.weight"):
                    self.assertIn(name + ".scale", names)
                    q, s = cv.quantize_int8_rows(state[name])
                    self.assertTrue((np.frombuffer(ck.file_of(name).raw(name), dtype=np.int8).reshape(q.shape) == q).all(), name)
                for name in ("model.vision_tower.std_bias", "model.vision_tower.patch_embedder.position_embedding_table",
                             "model.vision_tower.encoder.layers.0.self_attn.q_norm.weight"):
                    self.assertNotIn(name + ".scale", names)
                    self.assertTrue((ck.array(name) == state[name]).all(), name)
            finally:
                ck.close()

    def test_no_vision_leaves_the_tower_out_and_add_vision_makes_the_same_container(self):
        import tempfile
        import numpy as np
        import convert_gemma4 as cv
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as full, tempfile.TemporaryDirectory() as late:
            self.source(src)
            cv.convert(src, full, log=lambda *_: None)
            meta = cv.convert(src, late, log=lambda *_: None, with_vision=False)
            tower = [n for n in self.names(full) if n.startswith(("model.vision_tower.", "model.embed_vision."))]
            self.assertTrue(tower)
            self.assertFalse([n for n in self.names(late) if n in tower])
            self.assertNotIn("vision", meta)
            import make_gemma4_tiny as tiny
            self.assertEqual(meta["skipped_tensors"], len(tiny.vision_shapes()))      # left out, and counted
            cv.add_vision(src, late, log=lambda *_: None)
            self.assertEqual(sorted(self.names(late)), sorted(self.names(full)))
            def vision_meta(directory):
                with open(os.path.join(directory, "gemma4_meta.json"), encoding="utf-8") as handle:
                    return json.load(handle)["vision"]
            self.assertEqual(vision_meta(late), vision_meta(full))
            a, b = cv.Checkpoint(full), cv.Checkpoint(late)
            try:
                for name in tower:
                    self.assertTrue((np.asarray(a.file_of(name).raw(name)) == np.asarray(b.file_of(name).raw(name))).all(), name)
            finally:
                a.close()
                b.close()
            with self.assertRaises(SystemExit):                      # a second time: it is there already
                cv.add_vision(src, late, log=lambda *_: None)


class Engine:
    """One serving engine: frames come off stdout on a thread, so a test can write while it runs."""

    def __init__(self, slots=1, extra=None, cap="128", ctx="4096"):
        env = dict(os.environ, SNAP=TINY, SERVE="1", SERVE_BATCH="1", KV_SLOTS=str(slots), CTX=ctx,
                   OMP_NUM_THREADS="2", COLI_PREFIX_LOG="1", G4_IMG_TOKENS="1")   # 1: the ring is sized for text, as the tests count it
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

    def submit(self, text, slot=0, max_tokens=6, temperature=0.0, top_p=1.0, rid=None, images=None):
        """images: (patch bytes, grid height, grid width) for each picture, sent as IMAGE frames just before the SUBMIT."""
        rid = rid or str(self.next_id)
        self.next_id += 1
        payload = text.encode("utf-8")
        frames = b""
        for patches, gh, gw in images or ():
            frames += f"IMAGE {rid} {len(patches)} {gh} {gw}\n".encode() + patches + b"\n"
        self.proc.stdin.write(frames + f"SUBMIT {rid} {slot} {len(payload)} {max_tokens} {temperature} {top_p}\n".encode() + payload + b"\n")
        self.proc.stdin.flush()
        return rid

    def cancel(self, rid, verb="CANCEL"):
        self.proc.stdin.write(f"{verb} {rid}\n".encode())
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


@unittest.skipUnless(HAVE_FIXTURE, "needs the built gemma4, c/gemma4_tiny and numpy")
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
        self.assertIn(("CAPS", ["vision=1", "img_tokens=1"]), [(k, a) for k, a, _ in e.startup])
        text_only = self.engine(extra={"G4_VISION": "0"})
        self.assertIn(("CAPS", ["vision=0", "img_tokens=0"]), [(k, a) for k, a, _ in text_only.startup])
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

    def reuse_lines(self, engine):
        return [line for line in engine.log if line.startswith("[PREFIX]")]

    def test_a_rewound_conversation_keeps_what_it_shares_with_the_last(self):
        """The template drops the empty thought channel from past turns, so a resent conversation differs from
        what was fed right after "<|turn>model": the shared prefix is kept, the rest read again."""
        shared_text = "<|turn>user\nthe quick brown fox<turn|>\n<|turn>model\n"
        first = shared_text + "<|channel>thought\n<channel|>"
        second = (shared_text + "hello there<turn|>\n<|turn>user\nmore<turn|>\n<|turn>model\n"
                  "<|channel>thought\n<channel|>")
        with_reuse = self.engine()
        with_reuse.ask(first, max_tokens=6)
        a = with_reuse.ask(second, max_tokens=6)
        line = self.reuse_lines(with_reuse)[-1]
        shared = self.engine().ask(shared_text, max_tokens=1)["accept"]
        self.assertIn(f"reusing {shared} of", line)        # exactly the shared tokens, not the whole first prompt
        without = self.engine(extra={"COLI_KV_PREFIX": "0"})
        without.ask(first, max_tokens=6)
        self.assertEqual(a["data"], without.ask(second, max_tokens=6)["data"])

    def test_a_conversation_rewound_past_the_rings_slack_is_read_again(self):
        """Window 6, chunk 4, no slack: the ring holds 10 positions, so a prefix may end at most 5 before the end
        of what was fed."""
        small = {"G4_SLACK": "0", "G4_CHUNK": "4"}
        long_a = "a " * 40
        # "a " * 40 is 41 tokens; two generated tokens leave 42 fed. A prompt sharing the first k tokens rewinds 42 - k:
        # 5 is the most the ring allows.
        for text, kept in (("a " * 37 + "s s s", True), ("a " * 36 + "s s s", False), ("a " * 10 + "s " * 30, False)):
            e = self.engine(extra=small)
            e.ask(long_a, max_tokens=2)
            got = e.ask(text, max_tokens=5)
            line = self.reuse_lines(e)[-1]
            self.assertEqual("reusing" in line, kept, line)
            plain = self.engine(extra=dict(small, COLI_KV_PREFIX="0"))
            plain.ask(long_a, max_tokens=2)
            self.assertEqual(got["data"], plain.ask(text, max_tokens=5)["data"], text)

    def test_what_was_forgotten_by_a_rewind_is_not_matched_later(self):
        """A is rewound to B (shorter), then C is B followed by exactly the tokens A had after B's end. If the engine
        still believed it held A's tail, it would keep keys and values computed under A's middle for C."""
        t = "<turn|>"
        a = "<|turn>user\nthe quick" + t + " a" * 8 + t + " the fox"
        b = "<|turn>user\nthe quick" + t + " a" * 3 + t
        c = b + " a" * 5 + t + " the fox"
        e = self.engine()
        e.ask(a, max_tokens=1)
        in_b = e.ask(b, max_tokens=1)["accept"]
        got = e.ask(c, max_tokens=8)
        self.assertIn(f"reusing {in_b} of", self.reuse_lines(e)[-1])     # B, not A's longer prefix
        self.assertEqual(got["data"], self.engine().ask(c, max_tokens=8)["data"])

    def test_the_same_prompt_again_feeds_only_its_last_token(self):
        e = self.engine()
        e.ask(self.TURN, max_tokens=6)
        again = e.ask(self.TURN, max_tokens=6)
        self.assertIn("reusing", self.reuse_lines(e)[-1])
        self.assertEqual(again["data"], self.engine().ask(self.TURN, max_tokens=6)["data"])

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

    def drain(self, engine, rids):
        """The DATA of each request until every one has its DONE."""
        seen, done = {r: b"" for r in rids}, {}
        while len(done) < len(rids):
            kind, args, body = engine.get()
            if kind == "DATA" and args[0] in seen:
                seen[args[0]] += body
            elif kind in ("DONE", "ERROR") and args and args[0] in seen:
                done[args[0]] = (kind, args)
        return seen, done

    def test_conversations_decoding_together_say_what_each_says_alone(self):
        texts = ["the quick brown fox", "hello there", "  two  spaces\n\nand lines", "café 日本 \U0001f600"]
        lengths = [3, 9, 6, 12]                                            # they finish on different steps
        alone = [self.engine().ask(t, max_tokens=n)["data"] for t, n in zip(texts, lengths)]
        e = self.engine(slots=4)
        rids = [e.submit(t, slot=i, max_tokens=n) for i, (t, n) in enumerate(zip(texts, lengths))]
        seen, done = self.drain(e, rids)
        for rid, want in zip(rids, alone):
            self.assertEqual(done[rid][0], "DONE")
            self.assertEqual(seen[rid], want)

    def test_one_conversation_leaving_a_batch_does_not_disturb_the_others(self):
        texts = ["the quick brown fox", "hello there", "  two  spaces\n\nand lines"]
        alone = [self.engine().ask(t, max_tokens=8)["data"] for t in texts]
        e = self.engine(slots=3)
        rids = [e.submit(t, slot=i, max_tokens=8 if i != 1 else 300) for i, t in enumerate(texts)]
        seen = {r: b"" for r in rids}
        stopped, done = False, {}
        while len(done) < 3:
            kind, args, body = e.get()
            if kind == "DATA" and args[0] in seen:
                seen[args[0]] += body
                if args[0] == rids[1] and not stopped:
                    e.cancel(rids[1])                                      # the long one is cancelled in the middle
                    stopped = True
            elif kind == "DONE" and args[0] in seen:
                done[args[0]] = args
        self.assertEqual(seen[rids[0]], alone[0])
        self.assertEqual(seen[rids[2]], alone[2])
        self.assertLess(int(done[rids[1]][2]), 300)
        self.assertEqual(e.ask("hello there", slot=1, max_tokens=2)["done"][2], "2")

    def test_a_conversation_reading_its_prompt_and_others_decoding_take_turns(self):
        """A long prompt is read a chunk at a time between the decode steps of the others, and nobody's reply changes."""
        long_text = "a " * 200
        want_long = self.engine().ask(long_text, max_tokens=4)["data"]
        want_short = self.engine().ask("hello there", max_tokens=20)["data"]
        e = self.engine(slots=2, extra={"G4_CHUNK": "16"})
        r1 = e.submit("hello there", slot=0, max_tokens=20)
        r2 = e.submit(long_text, slot=1, max_tokens=4)
        seen, done = self.drain(e, [r1, r2])
        self.assertEqual(seen[r1], want_short)
        self.assertEqual(seen[r2], want_long)

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

    def test_stop_ends_a_request_early_the_same_way(self):
        """The gateway sends STOP when a client's stop sequence has matched: the engine must not run on to max_tokens."""
        e = self.engine()
        rid = e.submit("the quick brown fox", max_tokens=400)
        kind = None
        while kind != "DATA":
            kind, args, _ = e.get()
        e.cancel(rid, verb="STOP")
        out = e.collect(rid)
        self.assertLess(int(out["done"][2]), 400)
        self.assertEqual(out["done"][7], "0")
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


@unittest.skipUnless(HAVE_FIXTURE, "needs the built gemma4, c/gemma4_tiny and numpy")
class ServeVisionTest(unittest.TestCase):
    """Pictures over the serve protocol: IMAGE frames before the SUBMIT, soft tokens computed when the prompt reaches
    them, a cache that remembers which picture it was built on."""
    VISION = {"G4_IMG_TOKENS": "12"}

    @classmethod
    def setUpClass(cls):
        with open(REF, encoding="utf-8") as handle:
            cls.ref = json.load(handle)
        cls.ids = {e["text"]: e["ids"] for e in cls.ref["encodings"]}

    def engine(self, **kw):
        kw["extra"] = dict(self.VISION, **kw.get("extra", {}))
        e = Engine(**kw)
        self.addCleanup(e.close)
        return e

    def prompt(self, *pictures, before="hello there", after="the quick brown fox"):
        """Text and the ids it makes: `before`, each picture (as the gateway expands it: begin, soft tokens, end),
        `after`."""
        text, ids = before, list(self.ids[before]) if before else []
        for index in pictures:
            tokens = (self.ref["vision"]["images"][index]["gh"] // 3) * (self.ref["vision"]["images"][index]["gw"] // 3)
            text += "<|image>" + "<|image|>" * tokens + "<image|>"
            ids += [501] + [500] * tokens + [502]
        text += after
        ids += self.ids[after]
        return text, ids

    def pictures(self, *indexes):
        return [picture(self.ref, i) for i in indexes]

    def pictures_made(self, e):
        return sum("picture:" in line for line in e.log)

    def test_the_reply_is_the_offline_greedy_continuation(self):
        text, ids = self.prompt(0)
        want = offline(ids, 6, self.VISION, self.pictures(0))
        e = self.engine()
        out = e.ask(text, max_tokens=6, images=self.pictures(0))
        self.assertIsNone(out["error"])
        self.assertEqual(out["accept"], len(ids))
        self.assertEqual(out["data"], bytes.fromhex(want["greedy_hex"]))

    def test_two_pictures_and_small_chunks(self):
        text, ids = self.prompt(2, 1)
        want = offline(ids, 5, self.VISION, self.pictures(2, 1))
        e = self.engine(extra={"G4_CHUNK": "4"})        # the 12-token picture is three chunks wide, and goes whole
        out = e.ask(text, max_tokens=5, images=self.pictures(2, 1))
        self.assertEqual(out["data"], bytes.fromhex(want["greedy_hex"]))
        self.assertEqual(self.pictures_made(e), 2)

    def test_a_turn_that_extends_a_picture_conversation_reads_the_picture_once(self):
        first, _ = self.prompt(0)
        second = first + "<turn|>\n<|turn>user\nhello there"
        e = self.engine()
        e.ask(first, max_tokens=2, images=self.pictures(0))
        got = e.ask(second, max_tokens=5, images=self.pictures(0))
        self.assertIn("reusing", [l for l in e.log if l.startswith("[PREFIX]")][-1])
        self.assertEqual(self.pictures_made(e), 1)       # the second turn's picture rows were in the cache
        plain = self.engine(extra={"COLI_KV_PREFIX": "0"})
        plain.ask(first, max_tokens=2, images=self.pictures(0))
        self.assertEqual(got["data"], plain.ask(second, max_tokens=5, images=self.pictures(0))["data"])

    def test_another_picture_is_not_served_from_the_cache_of_this_one(self):
        """The same ids, a different picture: the placeholders match and the pixels do not."""
        text, ids = self.prompt(0)
        patches, gh, gw = self.pictures(0)[0]
        other = bytes(255 - v for v in patches)
        e = self.engine()
        first = e.ask(text, max_tokens=4, images=[(patches, gh, gw)])
        second = e.ask(text, max_tokens=4, images=[(other, gh, gw)])
        line = [l for l in e.log if l.startswith("[PREFIX]")][-1]
        before = len(self.ids["hello there"]) + 1                    # the text and the begin-of-image token
        self.assertIn(f"reusing {before} of", line)                  # up to the picture, not into it
        self.assertEqual(self.pictures_made(e), 2)
        fresh = self.engine().ask(text, max_tokens=4, images=[(other, gh, gw)])
        self.assertEqual(second["data"], fresh["data"])
        self.assertNotEqual(first["data"], second["data"])
        again = e.ask(text, max_tokens=4, images=[(patches, gh, gw)])      # and back: the first picture's rows are gone
        self.assertEqual(again["data"], first["data"])

    def test_the_picture_count_must_match_the_placeholders(self):
        text, _ = self.prompt(0)
        e = self.engine()
        out = e.ask(text, max_tokens=2)                                    # placeholders, no picture
        self.assertTrue((out["error"] or "").startswith("BAD_IMAGE"), out)
        out = e.ask("hello there", max_tokens=2, images=self.pictures(0))  # a picture, no placeholders
        self.assertTrue((out["error"] or "").startswith("BAD_IMAGE"), out)
        out = e.ask(self.prompt(1)[0], max_tokens=2, images=self.pictures(0))      # the wrong size
        self.assertIn("soft tokens", out["error"] or "")
        self.assertEqual(e.ask("hello there", max_tokens=1)["done"][2], "1")       # and the engine goes on

    def test_a_malformed_picture_is_refused_by_id_and_its_submit_is_silent(self):
        text, _ = self.prompt(0)
        patches, gh, gw = self.pictures(0)[0]
        e = self.engine()
        for bad in ((patches[:-3], gh, gw), (patches, gh + 1, gw), (patches, gh, gw + 3)):
            out = e.ask(text, max_tokens=2, images=[bad])
            self.assertTrue((out["error"] or "").startswith("BAD_IMAGE"), out)
        self.assertEqual(e.ask("hello there", max_tokens=1)["done"][2], "1")

    def test_a_picture_beyond_what_the_engine_takes_is_refused(self):
        text, _ = self.prompt(2)
        e = self.engine(extra={"G4_IMG_TOKENS": "6"})                       # the 12-token picture does not fit
        out = e.ask(text, max_tokens=2, images=self.pictures(2))
        self.assertIn("G4_IMG_TOKENS", out["error"] or "")

    def test_an_engine_without_the_tower_refuses_pictures(self):
        text, _ = self.prompt(0)
        e = self.engine(extra={"G4_VISION": "0"})
        out = e.ask(text, max_tokens=2, images=self.pictures(0))
        self.assertIn("no vision tower", out["error"] or "")
        self.assertEqual(e.ask("hello there", max_tokens=1)["done"][2], "1")

    def test_cancel_during_a_picture_prompt(self):
        text, _ = self.prompt(2, 0)
        e = self.engine()
        rid = e.submit(text, max_tokens=300, images=self.pictures(2, 0))
        e.cancel(rid)
        out = e.collect(rid)
        self.assertIsNotNone(out["done"])
        self.assertEqual(e.ask("hello there", max_tokens=1)["done"][2], "1")


if __name__ == "__main__":
    unittest.main()
