"""The music family from the registry up to the commands a user types.

Detection (model_type acestep at the root), dispatch (engine, tokenizer), and every entry point the project rule
names: `coli info`, `coli plan`, `coli doctor`, `coli music`, `coli run`, and the refusals (`coli chat`, `coli tune`,
--ctx). The engine is tools/acestep_stub.py, reached through COLI_ENGINE exactly as a packaged engine would be.
"""
import importlib.machinery
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acestep_fixture import (RATE, STUB, TENSORS, expected_resident, make_fake_checkpoint,  # noqa: E402
                             stub_env)

import music_engine  # noqa: E402
from family_registry import (FamilyConfigError, UnknownFamilyError, family_by_id,  # noqa: E402
                             family_for_config, resolve_model)

C_DIR = Path(__file__).resolve().parent.parent
CLI = C_DIR / "coli"


def load_cli():
    loader = importlib.machinery.SourceFileLoader("coli_music_test", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def run_coli(args, env, stdin=None, timeout=120):
    return subprocess.run([sys.executable, str(CLI), *args], env=env, input=stdin,
                          capture_output=True, text=True, timeout=timeout, cwd=str(C_DIR))


class FamilyDetectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_the_checkpoint_root_is_the_music_family(self):
        model = make_fake_checkpoint(self.root / "m")
        resolved = resolve_model(model)
        family = resolved.descriptor
        self.assertEqual(family.id, "ace_step")
        self.assertEqual(family.modality, "audio")
        self.assertEqual(family.engine_artifact, "acestep")
        self.assertEqual(family.default_model_id, "ace-step-1.5-colibri")
        self.assertEqual(family.display_name, "ACE-Step 1.5")
        self.assertEqual(family.tokenizer_file, "Qwen3-Embedding-0.6B/tokenizer.json")
        self.assertEqual(resolved.model_type, "acestep")
        self.assertEqual(family_for_config({"model_type": "acestep"}).id, "ace_step")

    def test_it_has_none_of_what_a_text_family_has(self):
        family = family_by_id("ace_step")
        self.assertFalse(family.capabilities.image)
        self.assertFalse(family.capabilities.decision)
        self.assertEqual(family.limits.kv_env if hasattr(family.limits, "kv_env") else "", "")
        self.assertFalse(family.supports_accelerator)

    def test_a_text_checkpoint_is_not_taken_for_it(self):
        model = self.root / "m"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "olmoe"}))
        self.assertEqual(resolve_model(model).descriptor.id, "olmoe")
        self.assertFalse(music_engine.is_music_model_dir(model))

    def test_a_directory_with_the_config_but_no_dit_is_not_a_music_model(self):
        model = make_fake_checkpoint(self.root / "m")
        (Path(model) / "acestep-v15-turbo" / "model.safetensors").unlink()
        self.assertFalse(music_engine.is_music_model_dir(model))


class LauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = make_fake_checkpoint(Path(self.tmp.name) / "m")

    def test_dispatch(self):
        self.assertEqual(self.cli.model_arch(self.model), "ace_step")
        self.assertEqual(os.path.basename(self.cli.engine_for(self.model)), "acestep" + self.cli._EXE)
        self.assertTrue(self.cli.is_music_model(self.model))
        self.assertFalse(self.cli.is_image_model(self.model))
        self.assertFalse(self.cli.is_decision_model(self.model))

    def test_a_text_model_is_not_a_music_model(self):
        other = Path(self.tmp.name) / "text"
        other.mkdir()
        (other / "config.json").write_text(json.dumps({"model_type": "olmoe"}))
        self.assertFalse(self.cli.is_music_model(other))
        self.assertFalse(self.cli.is_music_model(Path(self.tmp.name) / "nowhere"))

    def test_need_model_wants_the_text_encoders_tokenizer(self):
        os.remove(Path(self.model) / "Qwen3-Embedding-0.6B" / "tokenizer.json")
        with self.assertRaises(SystemExit) as caught:
            self.cli.need_model(self.model, engine=str(STUB))
        self.assertIn("Qwen3-Embedding-0.6B/tokenizer.json", str(caught.exception))
        (Path(self.model) / "Qwen3-Embedding-0.6B" / "tokenizer.json").write_text("{}")
        self.cli.need_model(self.model, engine=str(STUB))      # no exit
        self.assertFalse((Path(self.model) / "tokenizer.json").exists())

    def test_ctx_is_refused_for_a_music_model(self):
        class Args:
            ctx = 4096
            ram = 0
            temp = None
            model = self.model

            def __getattr__(self, name):
                return None
        with self.assertRaises(SystemExit) as caught:
            self.cli.env_for_engine(Args(), "ace_step")
        self.assertIn("--ctx", str(caught.exception))

    def test_lyrics_may_come_from_a_file(self):
        path = Path(self.tmp.name) / "lyrics.txt"
        path.write_text("[Verse]\nla la la\n", encoding="utf-8")
        self.assertEqual(self.cli.music_lyrics("@" + str(path)), "[Verse]\nla la la\n")
        self.assertEqual(self.cli.music_lyrics("plain words"), "plain words")
        self.assertIsNone(self.cli.music_lyrics(None))
        with self.assertRaises(SystemExit) as caught:
            self.cli.music_lyrics("@" + str(Path(self.tmp.name) / "no-such-file.txt"))
        self.assertIn("--lyrics", str(caught.exception))


class CommandsTest(unittest.TestCase):
    """The commands, as a user runs them, against the stub engine."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = make_fake_checkpoint(Path(self.tmp.name) / "ace-step")
        self.songs = Path(self.tmp.name) / "songs"
        self.env = stub_env(COLI_ENGINE=str(STUB), COLI_MUSIC_DIR=str(self.songs))
        for key in ("COLI_MODEL", "COLI_COLOR", "COLI_API_KEY"):
            self.env.pop(key, None)

    def wav(self, path):
        return music_engine.decode_wav(Path(path).read_bytes())

    def test_info_describes_the_checkpoint(self):
        result = run_coli(["info", "--model", self.model], self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        for needle in ("ACE-Step 1.5", "text-to-music", "Qwen3-Embedding-0.6B", "acestep-v15-turbo", "vae",
                       "Qwen3-Embedding-0.6B/tokenizer.json", "silence_latent.pt", str(self.songs), "acestep-5Hz-lm-1.7B",
                       "coli music", "/v1/audio/music"):
            self.assertIn(needle, result.stdout)
        self.assertNotIn("expert/layer", result.stdout)

    def test_plan_counts_what_the_engine_holds(self):
        result = run_coli(["plan", "--model", self.model, "--json"], self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["kind"], "audio")
        parts = {c["name"]: c for c in plan["components"]}
        for name in TENSORS:
            self.assertEqual(parts[name]["resident_bytes"], expected_resident(name), name)
        self.assertEqual(plan["modes"]["resident"]["peak_bytes"], sum(expected_resident(n) for n in TENSORS))
        self.assertEqual(plan["model"]["planner_unused_bytes"], 2000)
        text = run_coli(["plan", "--model", self.model], self.env)
        self.assertEqual(text.returncode, 0, text.stderr)
        for needle in ("all resident", "working buffers", "VAE decoder", "planner"):
            self.assertIn(needle, text.stdout)
        self.assertNotIn("experts", text.stdout)
        refused = run_coli(["plan", "--model", self.model, "--ctx", "4096"], self.env)
        self.assertNotEqual(refused.returncode, 0)

    def test_doctor_runs_the_music_checks(self):
        result = run_coli(["doctor", "--model", self.model, "--json"], self.env)
        report = json.loads(result.stdout)
        ids = {check["id"]: check["status"] for check in report["checks"]}
        self.assertEqual(ids["model.family"], "pass")
        self.assertEqual(ids["model.tokenizer"], "pass")
        self.assertEqual(ids["model.silence_latent"], "pass")
        self.assertEqual(ids["model.components"], "pass")
        self.assertNotIn("model.config", ids)
        self.assertIsNotNone(report["music_plan"])

    def test_doctor_names_what_is_missing(self):
        os.remove(Path(self.model) / "acestep-v15-turbo" / "silence_latent.pt")
        os.remove(Path(self.model) / "Qwen3-Embedding-0.6B" / "tokenizer.json")
        report = json.loads(run_coli(["doctor", "--model", self.model, "--json"], self.env).stdout)
        ids = {check["id"]: check["status"] for check in report["checks"]}
        self.assertEqual(ids["model.silence_latent"], "fail")
        self.assertEqual(ids["model.tokenizer"], "fail")
        self.assertEqual(report["status"], "error")

    def test_music_writes_a_wav_and_prints_its_path(self):
        out = Path(self.tmp.name) / "clip.wav"
        result = run_coli(["music", "calm piano with soft strings", "--model", self.model, "--duration", "2",
                           "--steps", "2", "--seed", "5", "--out", str(out)], self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], str(out))      # the banner comes first, the path last
        self.assertIn("text-to-music", result.stdout)
        pcm, rate, channels = self.wav(out)
        self.assertEqual((rate, channels, len(pcm)), (RATE, 2, 2 * RATE * 4))
        self.assertIn("seed 5", result.stderr)
        self.assertIn("saved", result.stderr)

    def test_music_without_out_goes_to_the_music_folder_under_a_name_that_says_what_it_is(self):
        result = run_coli(["music", "calm piano", "--model", self.model, "--duration", "1", "--steps", "1", "--seed", "6"], self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        files = list(self.songs.glob("*.wav"))
        self.assertEqual(len(files), 1)
        self.assertRegex(files[0].name, r"^\d{8}-\d{6}-calm-piano-s6\.wav$")
        self.assertEqual(result.stdout.strip().splitlines()[-1], str(files[0]))

    def test_the_same_seed_gives_the_same_file_and_the_flags_change_it(self):
        def make(*extra):
            out = Path(self.tmp.name) / ("c%d.wav" % len(list(Path(self.tmp.name).glob("c*.wav"))))
            r = run_coli(["music", "calm piano", "--model", self.model, "--duration", "1", "--steps", "1", "--seed", "3",
                          *extra, "--out", str(out)], self.env)
            self.assertEqual(r.returncode, 0, r.stderr)
            return out.read_bytes()
        base = make()
        self.assertEqual(make(), base)
        for extra in (["--lyrics", "la la la"], ["--bpm", "90"], ["--key", "C Major"], ["--language", "en"], ["--seed", "4"]):
            with self.subTest(extra=extra):
                self.assertNotEqual(make(*extra), base)

    def test_lyrics_from_a_file_and_lyrics_alone(self):
        lyrics = Path(self.tmp.name) / "lyrics.txt"
        lyrics.write_text("[Verse]\nla la la\n", encoding="utf-8")
        a, b, c = (Path(self.tmp.name) / n for n in ("a.wav", "b.wav", "c.wav"))
        common = ["--model", self.model, "--duration", "1", "--steps", "1", "--seed", "2"]
        self.assertEqual(run_coli(["music", "calm piano", *common, "--lyrics", "@" + str(lyrics), "--out", str(a)], self.env).returncode, 0)
        self.assertEqual(run_coli(["music", "calm piano", *common, "--lyrics", "[Verse]\nla la la\n", "--out", str(b)], self.env).returncode, 0)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertEqual(run_coli(["music", *common, "--lyrics", "la la la", "--out", str(c)], self.env).returncode, 0)   # no prompt
        self.assertTrue(c.exists())

    def test_run_on_a_music_model_makes_a_song_too(self):
        out = Path(self.tmp.name) / "run.wav"
        result = run_coli(["run", "calm piano", "--model", self.model, "--steps", "1", "--seed", "8", "--out", str(out)], self.env, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.wav(out)[1:], (RATE, 2))

    def test_a_flag_out_of_range_names_the_flag_and_makes_nothing(self):
        out = Path(self.tmp.name) / "no.wav"
        for flags, word in ((["--duration", "0"], "duration"), (["--duration", "9999"], "duration"), (["--steps", "0"], "steps"),
                            (["--steps", "99"], "steps"), (["--seed", "-1"], "seed"), (["--bpm", "0"], "bpm"),
                            (["--shift", "0"], "shift"), (["--lyrics", "@" + str(Path(self.tmp.name) / "none.txt")], "lyrics")):
            with self.subTest(flags=flags):
                r = run_coli(["music", "calm piano", "--model", self.model, *flags, "--out", str(out)], self.env)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(word, r.stderr + r.stdout)
        self.assertFalse(out.exists())

    def test_music_needs_something_to_say(self):
        r = run_coli(["music", "--model", self.model], self.env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("usage: coli music", r.stderr + r.stdout)

    def test_an_engine_failure_is_said_and_is_a_failure(self):
        out = Path(self.tmp.name) / "fail.wav"
        r = run_coli(["music", "__stub_error__", "--model", self.model, "--out", str(out)], self.env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("asked to fail", r.stderr)
        self.assertFalse(out.exists())

    def test_music_on_a_model_that_is_not_music_is_refused_by_name(self):
        other = Path(self.tmp.name) / "text"
        other.mkdir()
        (other / "config.json").write_text(json.dumps({"model_type": "olmoe"}))
        r = run_coli(["music", "calm piano", "--model", str(other)], self.env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("text-to-music", r.stderr + r.stdout)

    def test_chat_and_tune_refuse_a_music_model_with_a_pointer(self):
        chat = run_coli(["chat", "--model", self.model], self.env, stdin="")
        self.assertNotEqual(chat.returncode, 0)
        self.assertIn("coli music", chat.stderr + chat.stdout)
        tune = run_coli(["tune", "--model", self.model], self.env)
        self.assertNotEqual(tune.returncode, 0)
        self.assertIn("text engines", tune.stderr + tune.stdout)

    def test_web_says_the_dashboard_has_no_player_yet(self):
        r = run_coli(["web", "--model", self.model, "--no-browser"], self.env, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("coli music", r.stderr + r.stdout)

    def test_help_lists_the_music_command_and_its_flags(self):
        r = run_coli(["music", "--help"], self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        for flag in ("--lyrics", "--language", "--duration", "--steps", "--seed", "--shift", "--bpm", "--key", "--time-signature",
                     "--no-dcw", "--out"):
            self.assertIn(flag, r.stdout)


if __name__ == "__main__":
    unittest.main()
