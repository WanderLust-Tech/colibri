"""The ACE-Step engine's line protocol (acestep_serve.h, `acestep --model DIR --serve`), on the tiny fixture.

    ACESTEP_TINY=/path/to/fixture python -m unittest tests.test_acestep_serve

Needs the built engine (make acestep) and the tiny fixture (tools/make_acestep_tiny.py); skipped without them. Holds the
engine to its own documented protocol: a READY line with the limits, GEN answered by PROGRESS frames and one AUDIO frame
whose byte count is its own, the same seed giving the same bytes, every refused request leaving the engine ready for the
next, a CANCEL written behind its GEN answered "cancelled", requests queued behind a running one answered in order, a
line past the limit refused by id, and end of input a clean exit.
"""
import json
import os
import queue
import subprocess
import tempfile
import threading
import unittest
import wave

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXE = next((p for p in (os.path.join(C_DIR, "acestep.exe"), os.path.join(C_DIR, "acestep")) if os.path.isfile(p)), None)
TINY = os.environ.get("ACESTEP_TINY")


class Engine:
    """One engine process: frames come off stdout on a thread, so a test can write while the engine runs."""

    def __init__(self, model):
        self.proc = subprocess.Popen([EXE, "--model", model, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=0)
        self.frames = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        out = self.proc.stdout
        buf = b""
        while True:
            chunk = out.read(1 << 16)
            if not chunk:
                break
            buf += chunk
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = buf[:nl].decode("utf-8", "replace")
                kind, _, body = line.partition(" ")
                frame = json.loads(body) if body.startswith("{") else {}
                rest = buf[nl + 1:]
                if kind == "AUDIO":
                    need = frame["bytes"] + 1
                    while len(rest) < need:
                        more = out.read(1 << 20)
                        if not more:
                            self.frames.put(("EOF", {}, b""))
                            return
                        rest += more
                    frame["_pcm"], buf = rest[:frame["bytes"]], rest[need:]
                    frame["_nl"] = rest[frame["bytes"]:need]
                else:
                    buf = rest
                self.frames.put((kind, frame, frame.get("_pcm", b"")))
        self.frames.put(("EOF", {}, b""))

    def send(self, kind, obj=None, raw=None):
        data = raw if raw is not None else (kind + " " + json.dumps(obj) + "\n").encode()
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def next(self, timeout=120):
        return self.frames.get(timeout=timeout)

    def until_final(self, rid, timeout=300):
        """Everything up to the AUDIO or ERROR for rid: (the progress frames, the final kind, the final frame)."""
        progress = []
        while True:
            kind, frame, _ = self.next(timeout)
            if kind == "PROGRESS" and frame.get("id") == rid:
                progress.append(frame)
            elif kind in ("AUDIO", "ERROR") and frame.get("id") == rid:
                return progress, kind, frame
            elif kind == "EOF":
                raise AssertionError("the engine ended while waiting for " + rid)

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.reader.join(timeout=10)
        self.proc.stdout.close()


@unittest.skipUnless(EXE and TINY and os.path.isdir(TINY), "needs the built engine and ACESTEP_TINY")
class Serve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(TINY)
        kind, cls.ready, _ = cls.engine.next(300)
        assert kind == "READY", (kind, cls.ready)
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def gen(self, **fields):
        type(self).n += 1
        rid = "t%d" % self.n
        body = {"id": rid, "caption": "calm piano with soft strings", "duration": 3}
        body.update(fields)
        self.engine.send("GEN", body)
        return self.engine.until_final(rid)

    def test_ready_says_what_the_engine_is(self):
        r = self.ready
        self.assertEqual((r["sample_rate"], r["channels"], r["format"]), (48000, 2, "s16le"))
        self.assertEqual((r["min_duration"], r["max_duration"], r["max_steps"], r["protocol"]), (1, 600, 20, 1))
        for key in ("model", "default_duration", "default_steps", "default_shift", "max_caption_bytes", "max_lyric_bytes", "max_seed"):
            self.assertIn(key, r)

    def test_a_request_is_answered_with_progress_and_audio_of_the_size_it_says(self):
        progress, kind, a = self.gen(seed=5, steps=4)
        self.assertEqual(kind, "AUDIO", a)
        self.assertEqual((a["sample_rate"], a["channels"]), (48000, 2))
        self.assertEqual(a["bytes"], a["samples"] * 4)
        self.assertEqual((a["seed"], a["steps"]), (5, 4))
        self.assertEqual(a["frames"], 128)                       # three seconds is under the 128-frame floor
        self.assertEqual(a["samples"], 144000)                   # and is cut to the three seconds asked
        self.assertEqual(len(a["_pcm"]), a["bytes"])
        self.assertEqual(a["_nl"], b"\n")
        stages = [p["stage"] for p in progress]
        self.assertEqual(stages[:2], ["text", "condition"])
        self.assertEqual([p["step"] for p in progress if p["stage"] == "denoise"], [0, 1, 2, 3, 4])
        self.assertEqual(stages[-1], "decode")
        self.assertLessEqual(a["peak"], 100.0)
        self.assertEqual(set(a["timings"]), {"text", "condition", "denoise", "decode"})

    def test_a_clip_over_the_floor_has_the_frames_of_its_duration(self):
        _, kind, a = self.gen(duration=7.7, steps=2, seed=1)
        self.assertEqual(kind, "AUDIO", a)
        self.assertEqual(a["frames"], 192)                       # int(7.7 * 48000) // 1920
        self.assertEqual(a["samples"], 192 * 1920)

    def test_the_same_seed_gives_the_same_bytes_and_another_does_not(self):
        _, _, a = self.gen(seed=11, steps=2)
        _, _, b = self.gen(seed=11, steps=2)
        _, _, c = self.gen(seed=12, steps=2)
        self.assertEqual(a["_pcm"], b["_pcm"])
        self.assertNotEqual(a["_pcm"], c["_pcm"])

    def test_an_empty_caption_is_allowed_but_a_missing_one_is_not(self):
        _, kind, a = self.gen(caption="", lyrics="[Verse]\nla la la", seed=2, steps=1)
        self.assertEqual(kind, "AUDIO", a)
        self.engine.send("GEN", {"id": "nocap", "duration": 3})
        _, kind, frame = self.engine.until_final("nocap")
        self.assertEqual(kind, "ERROR")
        self.assertIn("caption", frame["message"])

    def test_what_is_left_out_means_what_the_documentation_says(self):
        """No lyrics is instrumental and no language is "unknown"; an empty lyric string is used as it is."""
        _, _, none = self.gen(seed=4, steps=1)
        _, _, explicit = self.gen(seed=4, steps=1, lyrics="[Instrumental]", language="unknown")
        _, _, empty = self.gen(seed=4, steps=1, lyrics="")
        self.assertEqual(none["_pcm"], explicit["_pcm"])
        self.assertNotEqual(none["_pcm"], empty["_pcm"])

    def test_the_prompt_changes_the_music(self):
        _, _, a = self.gen(seed=3, steps=2)
        _, _, b = self.gen(seed=3, steps=2, caption="heavy metal with screaming guitars")
        _, _, c = self.gen(seed=3, steps=2, lyrics="[Verse]\nla la la")
        _, _, d = self.gen(seed=3, steps=2, bpm=120, key="C Major", timesig="4")
        self.assertEqual(len({a["_pcm"], b["_pcm"], c["_pcm"], d["_pcm"]}), 4)

    def test_dcw_and_shift_and_steps_are_honoured(self):
        _, _, a = self.gen(seed=3, steps=3)
        _, _, nodcw = self.gen(seed=3, steps=3, dcw=False)
        _, _, shifted = self.gen(seed=3, steps=3, shift=3)
        self.assertNotEqual(a["_pcm"], nodcw["_pcm"])
        self.assertNotEqual(a["_pcm"], shifted["_pcm"])
        self.assertEqual(shifted["shift"], 3)

    def test_lyrics_in_other_scripts_and_a_cut_caption(self):
        _, kind, a = self.gen(lyrics="こんにちは世界\n😀", language="ja", steps=1)
        self.assertEqual(kind, "AUDIO", a)
        self.assertFalse(a["truncated"])
        _, kind, a = self.gen(caption="piano " * 600, steps=1)
        self.assertEqual(kind, "AUDIO", a)
        self.assertTrue(a["truncated"])

    def test_every_refusal_names_its_reason_and_keeps_the_engine(self):
        cases = (
            ({"caption": None}, "caption"), ({"caption": 5}, "caption"),
            ({"duration": 0.5}, "duration"), ({"duration": 601}, "duration"), ({"duration": "10"}, "duration"),
            ({"steps": 0}, "steps"), ({"steps": 21}, "steps"), ({"steps": 2.5}, "steps"),
            ({"shift": 0}, "shift"), ({"shift": 11}, "shift"),
            ({"seed": -1}, "seed"), ({"seed": 2 ** 32}, "seed"), ({"seed": 1.5}, "seed"),
            ({"bpm": -3}, "bpm"), ({"bpm": 1.5}, "bpm"), ({"dcw": "yes"}, "dcw"),
            ({"lyrics": 7}, "lyrics"), ({"lyrics": "x" * 40000}, "lyrics"), ({"caption": "x" * 5000}, "caption"),
            ({"key": "k" * 40}, "key"), ({"language": "l" * 40}, "language"),
        )
        for fields, word in cases:
            with self.subTest(fields={k: (v if not isinstance(v, str) or len(v) < 20 else v[:8] + "...") for k, v in fields.items()}):
                _, kind, frame = self.gen(**fields)
                self.assertEqual(kind, "ERROR", frame)
                self.assertIn(word, frame["message"])
        _, kind, a = self.gen(seed=1, steps=1)
        self.assertEqual(kind, "AUDIO", a)

    def test_bad_lines_are_answered_and_noise_is_ignored(self):
        self.engine.send("GEN", raw=b"GEN this is not json\n")
        kind, frame, _ = self.engine.next()
        self.assertEqual((kind, frame["id"]), ("ERROR", "?"))
        self.assertIn("JSON", frame["message"])
        self.engine.send("GEN", raw=b"GEN [1, 2]\n")
        kind, frame, _ = self.engine.next()
        self.assertEqual(kind, "ERROR")
        self.engine.send("NOISE", raw=b"hello there\n\nCANCEL {\"id\":\"nothing-running\"}\n")
        _, kind, a = self.gen(seed=1, steps=1)
        self.assertEqual(kind, "AUDIO", a)

    def test_a_line_over_the_limit_is_refused_by_its_id(self):
        body =('GEN {"id":"huge","caption":"' + "a" * ((1 << 20) + 5000) + '"}\n').encode()
        self.engine.send("GEN", raw=body)
        kind, frame, _ = self.engine.next(60)
        self.assertEqual((kind, frame["id"]), ("ERROR", "huge"))
        self.assertIn("too long", frame["message"])
        _, kind, a = self.gen(seed=1, steps=1)
        self.assertEqual(kind, "AUDIO", a)

    def test_cancel_written_behind_its_gen_is_answered_cancelled(self):
        self.engine.send("GEN", {"id": "c1", "caption": "calm piano", "duration": 120, "steps": 8})
        self.engine.send("CANCEL", {"id": "c1"})
        _, kind, frame = self.engine.until_final("c1")
        self.assertEqual((kind, frame["message"]), ("ERROR", "cancelled"))
        _, kind, a = self.gen(seed=1, steps=1)
        self.assertEqual(kind, "AUDIO", a)

    def test_cancel_on_the_first_denoise_frame_stops_before_the_last_step(self):
        self.engine.send("GEN", {"id": "c2", "caption": "calm piano", "duration": 120, "steps": 8})
        seen = []
        while True:
            kind, frame, _ = self.engine.next(300)
            if kind == "PROGRESS" and frame["id"] == "c2":
                seen.append(frame)
                if frame["stage"] == "denoise":
                    self.engine.send("CANCEL", {"id": "c2"})
                    break
            elif kind in ("AUDIO", "ERROR"):
                self.fail("finished before it could be cancelled: %s %s" % (kind, frame))
        _, kind, frame = self.engine.until_final("c2")
        self.assertEqual((kind, frame["message"]), ("ERROR", "cancelled"))

    def test_a_cancel_for_another_id_does_not_stop_this_one(self):
        self.engine.send("GEN", {"id": "k1", "caption": "calm piano", "duration": 3, "steps": 2})
        self.engine.send("CANCEL", {"id": "someone-else"})
        _, kind, a = self.engine.until_final("k1")
        self.assertEqual(kind, "AUDIO", a)

    def test_requests_queued_behind_a_running_one_are_answered_in_order(self):
        for rid in ("q1", "q2", "q3"):
            self.engine.send("GEN", {"id": rid, "caption": "calm piano", "duration": 3, "steps": 1, "seed": 9})
        finals = []
        while len(finals) < 3:
            kind, frame, _ = self.engine.next(300)
            if kind in ("AUDIO", "ERROR"):
                finals.append((kind, frame["id"]))
        self.assertEqual(finals, [("AUDIO", "q1"), ("AUDIO", "q2"), ("AUDIO", "q3")])

    def test_the_audio_is_what_the_command_line_makes(self):
        """The one-shot mode and the protocol are one pipeline: the same request gives the same samples."""
        _, _, a = self.gen(seed=21, steps=2, duration=3)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "clip.wav")
            r = subprocess.run([EXE, "--model", TINY, "--caption", "calm piano with soft strings", "--duration", "3", "--steps", "2",
                                "--seed", "21", "--out", out], capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-300:])
            with wave.open(out, "rb") as w:
                self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()), (48000, 2, 2))
                self.assertEqual(w.readframes(w.getnframes()), a["_pcm"])


@unittest.skipUnless(EXE and TINY and os.path.isdir(TINY), "needs the built engine and ACESTEP_TINY")
class ServeEnds(unittest.TestCase):
    def test_end_of_input_is_a_clean_exit(self):
        e = Engine(TINY)
        kind, _, _ = e.next(300)
        self.assertEqual(kind, "READY")
        e.proc.stdin.close()
        self.assertEqual(e.proc.wait(timeout=60), 0)
        e.close()

    def test_a_missing_model_is_a_named_failure_not_a_hang(self):
        r = subprocess.run([EXE, "--model", os.path.join(TINY, "nope"), "--serve"], capture_output=True, text=True, timeout=60, input="")
        self.assertNotEqual(r.returncode, 0)
        self.assertLess(r.returncode, 128)
        self.assertNotIn("READY", r.stdout)


if __name__ == "__main__":
    unittest.main()
