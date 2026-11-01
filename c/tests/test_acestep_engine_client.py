"""music_engine.py: the client of the acestep engine's line protocol, run against tools/acestep_stub.py.

The stub speaks the protocol byte for byte, so the client's handling of progress, a refusal, a CANCEL, an abandoned
request, a crash and a stray line of output is held to it without the 6 GB of weights. (The real engine is held to the
same protocol by tests/test_acestep_serve.py.) Also the request rules, WAV, saving and the plan.

    python -m unittest tests.test_acestep_engine_client
"""
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import music_engine  # noqa: E402

STUB = HERE / "tools" / "acestep_stub.py"
RATE = 8000


def stub_env(**extra):
    env = dict(os.environ)
    env.update({"ACESTEP_STUB_RATE": str(RATE), "ACESTEP_STUB_DELAY": "0.02"})
    env.update({k: str(v) for k, v in extra.items()})
    return env


class WithStub(unittest.TestCase):
    def open(self, **env):
        engine = music_engine.MusicEngine(STUB, "model-dir", env=stub_env(**env), load_timeout=60,
                                          on_log=self.logs.append)
        self.addCleanup(engine.close)
        return engine

    def setUp(self):
        self.logs = []


class Client(WithStub):
    def test_ready_fills_the_info_and_the_engine_is_alive(self):
        engine = self.open()
        self.assertEqual((engine.info["sample_rate"], engine.info["channels"]), (RATE, 2))
        self.assertEqual(engine.info["max_duration"], 600)
        self.assertEqual(engine.modality, "audio")
        self.assertTrue(engine.alive)

    def test_a_request_returns_pcm_of_the_size_it_says_with_progress_in_order(self):
        engine = self.open()
        seen = []
        result = engine.generate("calm piano", duration=2, steps=3, seed=5, on_progress=seen.append)
        self.assertEqual(result["samples"], 2 * RATE)
        self.assertEqual(len(result["pcm"]), 2 * RATE * 2 * 2)
        self.assertEqual((result["seed"], result["steps"], result["channels"]), (5, 3, 2))
        stages = [p["stage"] for p in seen]
        self.assertEqual(stages[:2], ["text", "condition"])
        self.assertEqual([p["step"] for p in seen if p["stage"] == "denoise"], [0, 1, 2, 3])
        self.assertEqual(stages[-1], "decode")
        self.assertNotIn("id", result)

    def test_the_same_request_gives_the_same_bytes_and_each_field_changes_them(self):
        engine = self.open()
        base = dict(duration=1, steps=1, seed=3)
        first = engine.generate("calm piano", **base)["pcm"]
        self.assertEqual(engine.generate("calm piano", **base)["pcm"], first)
        for change in ({"seed": 4}, {"lyrics": "la la"}, {"bpm": 90}, {"key": "C Major"}, {"language": "en"}):
            with self.subTest(change=change):
                self.assertNotEqual(engine.generate("calm piano", **{**base, **change})["pcm"], first)
        self.assertNotEqual(engine.generate("heavy metal", **base)["pcm"], first)

    def test_a_field_left_as_none_is_left_out_of_the_request(self):
        engine = self.open()
        sent = []
        original = engine._send
        engine._send = lambda keyword, body: (sent.append((keyword, dict(body))), original(keyword, body))[1]
        engine.generate("calm piano", duration=1, steps=1)
        keyword, body = sent[0]
        self.assertEqual(keyword, "GEN")
        self.assertEqual(sorted(body), ["caption", "duration", "id", "steps"])

    def test_a_refusal_is_an_error_and_keeps_the_engine(self):
        engine = self.open()
        for kwargs, word in (({"duration": 0.5}, "duration"), ({"duration": 700}, "duration"), ({"steps": 0}, "steps"),
                             ({"steps": 21}, "steps"), ({"seed": -1}, "seed"), ({"shift": 0}, "shift")):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(music_engine.MusicEngineError) as raised:
                    engine.generate("calm piano", **kwargs)
                self.assertIn(word, str(raised.exception))
        self.assertEqual(engine.generate("calm piano", duration=1, steps=1)["samples"], RATE)

    def test_the_engine_may_be_told_to_fail_and_stays_ready(self):
        engine = self.open()
        with self.assertRaises(music_engine.MusicEngineError) as raised:
            engine.generate("__stub_error__", duration=1)
        self.assertIn("asked to fail", str(raised.exception))
        self.assertTrue(engine.alive)
        self.assertEqual(engine.generate("calm piano", duration=1, steps=1)["samples"], RATE)

    def test_cancel_through_the_callback_is_an_outcome_and_keeps_the_engine(self):
        engine = self.open(ACESTEP_STUB_DELAY=0.1)
        flag = threading.Event()
        with self.assertRaises(music_engine.MusicCancelled):
            engine.generate("calm piano", duration=1, steps=20, cancelled=flag.is_set,
                            on_progress=lambda p: flag.set() if p["stage"] == "denoise" else None)
        self.assertEqual(engine.generate("calm piano", duration=1, steps=1)["samples"], RATE)

    def test_a_callback_that_raises_does_not_swallow_the_next_request(self):
        engine = self.open(ACESTEP_STUB_DELAY=0.1)

        def boom(frame):
            if frame["stage"] == "denoise":
                raise ConnectionResetError("client went away")
        with self.assertRaises(ConnectionResetError):
            engine.generate("calm piano", duration=1, steps=20, on_progress=boom)
        again = engine.generate("calm piano again", duration=1, steps=1, seed=9)
        self.assertEqual((again["samples"], again["seed"]), (RATE, 9))

    def test_a_stray_line_of_output_is_logged_and_ignored(self):
        engine = self.open()
        self.assertEqual(engine.generate("__stub_garbage__ piano", duration=1, steps=1)["samples"], RATE)
        self.assertTrue(any("stray" in message for message in self.logs), self.logs)

    def test_a_crash_in_the_middle_is_an_error_naming_the_exit(self):
        engine = self.open()
        with self.assertRaises(music_engine.MusicEngineError) as raised:
            engine.generate("__stub_crash__", duration=1, steps=4)
        self.assertIn("exit code 3", str(raised.exception))
        self.assertFalse(engine.alive)
        with self.assertRaises(music_engine.MusicEngineError):
            engine.generate("calm piano", duration=1)

    def test_close_is_a_clean_exit_by_end_of_input(self):
        engine = self.open()
        engine.close()
        self.assertEqual(engine.process.poll(), 0)
        self.assertFalse(engine.alive)


class Startup(unittest.TestCase):
    def test_a_missing_executable_is_named(self):
        with self.assertRaises(music_engine.MusicEngineError) as raised:
            music_engine.MusicEngine(HERE / "no-such-engine", "model-dir")
        self.assertIn("cannot start", str(raised.exception))

    def test_an_engine_that_exits_while_loading_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "dies.py"
            script.write_text("import sys\nsys.stderr.write('no weights\\n')\nsys.exit(2)\n")
            with self.assertRaises(music_engine.MusicEngineError) as raised:
                music_engine.MusicEngine(script, "model-dir", stderr=subprocess_devnull())
            self.assertIn("exit code 2", str(raised.exception))

    def test_an_engine_that_never_says_ready_is_given_up_on(self):
        with self.assertRaises(music_engine.MusicEngineError) as raised:
            music_engine.MusicEngine(STUB, "model-dir", env=stub_env(ACESTEP_STUB_LOAD_DELAY=30), load_timeout=1)
        self.assertIn("did not become ready", str(raised.exception))

    def test_an_error_before_ready_is_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "refuses.py"
            script.write_text('import sys, time\nprint(\'ERROR {"message":"no model here"}\', flush=True)\ntime.sleep(30)\n')
            with self.assertRaises(music_engine.MusicEngineError) as raised:
                music_engine.MusicEngine(script, "model-dir", load_timeout=20)
            self.assertIn("no model here", str(raised.exception))


def subprocess_devnull():
    import subprocess
    return subprocess.DEVNULL


class Requests(unittest.TestCase):
    def ok(self, **body):
        return music_engine.music_request(body)

    def refused(self, param, **body):
        with self.assertRaises(music_engine.MusicRequestError) as raised:
            music_engine.music_request(body)
        self.assertEqual(raised.exception.param, param, str(raised.exception))
        return str(raised.exception)

    def test_the_defaults(self):
        r = self.ok(prompt="calm piano")
        self.assertEqual((r["caption"], r["lyrics"], r["language"], r["duration"], r["steps"], r["shift"], r["dcw"]),
                         ("calm piano", None, None, 30, 8, 1.0, True))
        self.assertFalse(r["seed_given"])
        self.assertTrue(0 <= r["seed"] <= 2 ** 32 - 1)

    def test_a_seed_is_kept_and_a_missing_one_is_made(self):
        self.assertEqual(self.ok(prompt="x", seed=7)["seed"], 7)
        self.assertTrue(self.ok(prompt="x", seed=7)["seed_given"])
        self.assertGreater(len({self.ok(prompt="x")["seed"] for _ in range(20)}), 15)       # drawn at random, not constant

    def test_caption_is_an_alias_for_prompt_and_the_two_may_not_disagree(self):
        self.assertEqual(self.ok(caption="calm piano")["caption"], "calm piano")
        self.assertEqual(self.ok(prompt="a", caption="a")["caption"], "a")
        self.refused("prompt", prompt="a", caption="b")

    def test_something_must_be_said(self):
        self.refused("prompt")
        self.refused("prompt", prompt="   ")
        self.refused("prompt", prompt="", lyrics="  \n")
        self.assertEqual(self.ok(lyrics="[Verse]\nla la")["caption"], "")      # lyrics alone are enough
        self.assertEqual(self.ok(prompt="x", lyrics="")["lyrics"], None)       # an empty lyric means instrumental

    def test_types_and_ranges_name_their_parameter(self):
        cases = (("prompt", {"prompt": 5}), ("lyrics", {"prompt": "x", "lyrics": 5}), ("language", {"prompt": "x", "language": 5}),
                 ("language", {"prompt": "x", "language": "l" * 33}), ("duration", {"prompt": "x", "duration": "10"}),
                 ("duration", {"prompt": "x", "duration": 0.5}), ("duration", {"prompt": "x", "duration": 601}),
                 ("duration", {"prompt": "x", "duration": True}), ("duration", {"prompt": "x", "duration": float("nan")}),
                 ("steps", {"prompt": "x", "steps": 0}), ("steps", {"prompt": "x", "steps": 21}), ("steps", {"prompt": "x", "steps": 2.5}),
                 ("steps", {"prompt": "x", "steps": True}), ("shift", {"prompt": "x", "shift": 0}), ("shift", {"prompt": "x", "shift": 11}),
                 ("shift", {"prompt": "x", "shift": "1"}), ("seed", {"prompt": "x", "seed": -1}), ("seed", {"prompt": "x", "seed": 2 ** 32}),
                 ("seed", {"prompt": "x", "seed": 1.5}), ("seed", {"prompt": "x", "seed": True}), ("bpm", {"prompt": "x", "bpm": 0}),
                 ("bpm", {"prompt": "x", "bpm": 1001}), ("bpm", {"prompt": "x", "bpm": 90.5}), ("key", {"prompt": "x", "key": 5}),
                 ("key", {"prompt": "x", "key": "k" * 33}), ("time_signature", {"prompt": "x", "time_signature": "123456789"}),
                 ("time_signature", {"prompt": "x", "time_signature": [4]}), ("dcw", {"prompt": "x", "dcw": "yes"}),
                 ("n", {"prompt": "x", "n": 2}), ("response_format", {"prompt": "x", "response_format": "mp3"}))
        for param, body in cases:
            with self.subTest(body=body):
                self.refused(param, **body)

    def test_the_size_limits_come_from_the_engine(self):
        info = {"max_duration": 60, "min_duration": 5, "max_steps": 10, "max_caption_bytes": 10, "max_lyric_bytes": 12, "max_seed": 99}
        for body, param in (({"prompt": "x", "duration": 61}, "duration"), ({"prompt": "x", "duration": 4}, "duration"),
                            ({"prompt": "x", "steps": 11}, "steps"), ({"prompt": "x" * 11}, "prompt"),
                            ({"prompt": "x", "lyrics": "l" * 13}, "lyrics"), ({"prompt": "x", "seed": 100}, "seed")):
            with self.subTest(body=body):
                with self.assertRaises(music_engine.MusicRequestError) as raised:
                    music_engine.music_request(body, info)
                self.assertEqual(raised.exception.param, param)
        self.assertEqual(music_engine.music_request({"prompt": "x", "duration": 60}, info)["duration"], 60)

    def test_a_time_signature_given_as_a_number_is_made_a_string(self):
        self.assertEqual(self.ok(prompt="x", time_signature=4)["time_signature"], "4")
        self.assertEqual(self.ok(prompt="x", time_signature="3/4")["time_signature"], "3/4")

    def test_the_body_must_be_an_object(self):
        with self.assertRaises(music_engine.MusicRequestError):
            music_engine.music_request([1, 2])

    def test_bytes_not_characters_are_counted(self):
        self.refused("prompt", prompt="é" * 2100)             # 4200 bytes over the 4096 limit, 2100 characters
        self.assertEqual(self.ok(prompt="é" * 2000)["caption"], "é" * 2000)


class Wav(unittest.TestCase):
    def test_encode_and_decode_round_trip(self):
        pcm = struct.pack("<hhhh", 1, -2, 300, -400)
        wav = music_engine.encode_wav(pcm, 48000, 2)
        self.assertEqual(wav[:4], b"RIFF")
        self.assertEqual(len(wav), 44 + len(pcm))
        self.assertEqual(struct.unpack("<I", wav[4:8])[0], 36 + len(pcm))
        self.assertEqual(music_engine.decode_wav(wav), (pcm, 48000, 2))

    def test_a_partial_frame_is_refused(self):
        with self.assertRaises(ValueError):
            music_engine.encode_wav(b"\x00\x00\x00", 48000, 2)

    def test_decode_refuses_what_it_does_not_read(self):
        for blob in (b"", b"RIFF" + b"\0" * 60, b"x" * 100):
            with self.subTest(size=len(blob)):
                with self.assertRaises(ValueError):
                    music_engine.decode_wav(blob)
        float_wav = bytearray(music_engine.encode_wav(b"\0" * 8, 48000, 2))
        float_wav[20:22] = struct.pack("<H", 3)
        with self.assertRaises(ValueError):
            music_engine.decode_wav(bytes(float_wav))

    def test_the_standard_library_reads_what_is_written(self):
        import wave
        import io
        pcm = struct.pack("<hh", 100, -100) * 480
        with wave.open(io.BytesIO(music_engine.encode_wav(pcm, 48000, 2)), "rb") as w:
            self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()), (48000, 2, 2, 480))
            self.assertEqual(w.readframes(480), pcm)


class Saving(unittest.TestCase):
    def test_a_name_that_says_what_it_is_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = music_engine.save_wav(b"one", "Calm piano, with soft strings!", 42, tmp, now=1_700_000_000)
            b = music_engine.save_wav(b"two", "Calm piano, with soft strings!", 42, tmp, now=1_700_000_000)
            self.assertNotEqual(a, b)
            self.assertTrue(os.path.basename(a).endswith("-calm-piano-with-soft-strings-s42.wav"), a)
            self.assertTrue(b.endswith("-2.wav"), b)
            self.assertEqual(Path(a).read_bytes(), b"one")
            self.assertEqual(Path(b).read_bytes(), b"two")
            self.assertEqual([p for p in os.listdir(tmp) if p.endswith(".tmp")], [])

    def test_an_empty_prompt_is_music_and_a_long_one_is_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("-music-s1.wav", music_engine.save_wav(b"x", "", 1, tmp))
            self.assertIn("-music-s2.wav", music_engine.save_wav(b"x", "!!!", 2, tmp))
            long_name = os.path.basename(music_engine.save_wav(b"x", "word " * 100, 3, tmp))
            self.assertLess(len(long_name), 100)

    def test_the_directory_follows_the_environment(self):
        self.assertEqual(music_engine.music_dir({"COLI_MUSIC_DIR": "/tmp/songs"}), os.path.expanduser("/tmp/songs"))
        self.assertTrue(music_engine.music_dir({}).endswith("colibri-music"))


def safetensors(path, tensors):
    """Write a safetensors file whose tensors are zeros: {name: (dtype, shape)}."""
    sizes = {"BF16": 2, "F32": 4, "F16": 2}
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = sizes[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + bytes(offset))


class Model(unittest.TestCase):
    def make(self, tmp, planner=True, config=None):
        root = Path(tmp)
        (root / "config.json").write_text(json.dumps(config if config is not None else {"model_type": "acestep"}))
        for name in music_engine.COMPONENTS:
            (root / name).mkdir()
        safetensors(root / music_engine.DIT_DIR / "model.safetensors", {"decoder.w": ("BF16", (100, 100))})
        (root / music_engine.DIT_DIR / "silence_latent.pt").write_bytes(b"x")
        safetensors(root / music_engine.TEXT_DIR / "model.safetensors", {"embed": ("BF16", (50, 20)), "norm": ("BF16", (20,))})
        (root / music_engine.TEXT_DIR / "tokenizer.json").write_text("{}")
        safetensors(root / music_engine.VAE_DIR / "diffusion_pytorch_model.safetensors",
                    {"decoder.conv": ("BF16", (10, 10)), "encoder.conv": ("BF16", (1000, 10)), "decoder.b": ("F32", (4,))})
        if planner:
            (root / "acestep-5Hz-lm-1.7B").mkdir()
            (root / "acestep-5Hz-lm-1.7B" / "model.safetensors").write_bytes(bytes(5000))
        return root

    def test_a_checkpoint_is_recognised_by_its_config_and_its_dit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make(tmp)
            self.assertTrue(music_engine.is_music_model_dir(root))
            (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
            self.assertFalse(music_engine.is_music_model_dir(root))
            (root / "config.json").write_text("not json")
            self.assertFalse(music_engine.is_music_model_dir(root))
            (root / "config.json").write_text(json.dumps({"model_type": "acestep"}))
            (root / music_engine.DIT_DIR / "model.safetensors").unlink()
            self.assertFalse(music_engine.is_music_model_dir(root))
        self.assertFalse(music_engine.is_music_model_dir(""))
        self.assertFalse(music_engine.is_music_model_dir("/no/such/dir"))

    def test_the_inventory_counts_what_the_engine_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            inv = music_engine.inspect_music_model(self.make(tmp))
            comps = inv["components"]
            self.assertEqual(comps[music_engine.DIT_DIR]["resident_bytes"], 100 * 100 * 2)
            self.assertEqual(comps[music_engine.TEXT_DIR]["resident_bytes"], (50 * 20 + 20) * 2)
            vae = comps[music_engine.VAE_DIR]
            self.assertEqual(vae["resident_bytes"], (10 * 10 + 4) * 4)       # the decoder, widened to f32
            self.assertEqual(vae["unloaded_params"], 10000)                  # the encoder stays on disk
            self.assertEqual(inv["planner"], ["acestep-5Hz-lm-1.7B"])
            self.assertEqual(inv["planner_bytes"], 5000)
            self.assertTrue(inv["tokenizer"] and inv["silence_latent"])

    def test_the_plan_says_whether_it_fits_and_warns_of_what_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.make(tmp)
            resident = sum(c["resident_bytes"] for c in music_engine.inspect_music_model(root)["components"].values())
            fits = music_engine.plan_music_model(root, available_memory=resident + 1000, available_disk=10 ** 9)
            self.assertTrue(fits["modes"]["resident"]["fits"])
            self.assertEqual(fits["modes"]["resident"]["peak_bytes"], resident)
            self.assertEqual(fits["warnings"], [])
            tight = music_engine.plan_music_model(root, available_memory=resident - 1)
            self.assertFalse(tight["modes"]["resident"]["fits"])
            self.assertTrue(any("do not fit" in w for w in tight["warnings"]))
            (root / music_engine.TEXT_DIR / "tokenizer.json").unlink()
            (root / music_engine.DIT_DIR / "silence_latent.pt").unlink()
            warnings = music_engine.plan_music_model(root)["warnings"]
            self.assertTrue(any("tokenizer" in w for w in warnings) and any("silence" in w for w in warnings), warnings)

    def test_the_plan_prints(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = music_engine.format_music_plan(music_engine.plan_music_model(self.make(tmp), 10 ** 10, 10 ** 12))
            for word in ("text-to-music", "DiT", "VAE decoder", "text encoder", "all resident", "planner", "colibri-music"):
                self.assertIn(word, text)


if __name__ == "__main__":
    unittest.main()
