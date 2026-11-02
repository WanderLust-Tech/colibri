"""`coli music` with audio in: --cover, --repaint --from --to, --ref and their settings, as a user types them, against
the stub engine (COLI_ENGINE, as a packaged engine would be reached). Every flag reaches the engine, every rule that
needs no engine is applied before one is started, and a clip that cannot be read is named by its flag."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acestep_fixture import RATE, STUB, make_fake_checkpoint, make_mp3, make_wav, stub_env  # noqa: E402
from test_acestep_cli import load_cli, run_coli  # noqa: E402

import music_engine  # noqa: E402


class AudioInCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = make_fake_checkpoint(self.root / "ace-step")
        self.env = stub_env(COLI_ENGINE=str(STUB), COLI_MUSIC_DIR=str(self.root / "songs"))
        for key in ("COLI_MODEL", "COLI_COLOR", "COLI_API_KEY"):
            self.env.pop(key, None)
        self.song = self.root / "song.wav"
        self.song.write_bytes(make_wav(3))
        self.voice = self.root / "voice.wav"
        self.voice.write_bytes(make_wav(2, frequency=700))
        self.out = self.root / "out.wav"

    def music(self, *args, env=None):
        return run_coli(["music", *args, "--model", self.model, "--steps", "1", "--seed", "4"], env or self.env)

    def made(self, *args):
        self.out.unlink(missing_ok=True)
        result = self.music(*args, "--out", str(self.out))
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        pcm, rate, channels = music_engine.decode_wav(self.out.read_bytes())
        return pcm, len(pcm) / (2 * channels) / rate, result

    def refused(self, *args, word=None, env=None):
        result = self.music(*args, "--out", str(self.out), env=env)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        text = result.stderr + result.stdout
        if word:
            self.assertIn(word, text)
        self.assertFalse(self.out.exists())
        return text

    # -- what each task makes -----------------------------------------------------------------------------------
    def test_a_cover_needs_no_prompt_and_is_as_long_as_the_clip(self):
        _pcm, seconds, result = self.made("--cover", str(self.song))
        self.assertAlmostEqual(seconds, 3.0, places=3)
        self.assertIn("cover", result.stderr)

    def test_a_cover_with_a_prompt_a_reference_and_its_settings(self):
        plain, _, _ = self.made("warm jazz", "--cover", str(self.song))
        songs = {plain}
        for extra in (["--cover-strength", "0.5"], ["--cover-noise", "0.4"], ["--ref", str(self.voice)]):
            pcm, seconds, _ = self.made("warm jazz", "--cover", str(self.song), *extra)
            self.assertAlmostEqual(seconds, 3.0, places=3)
            songs.add(pcm)
        self.assertEqual(len(songs), 4)

    def test_a_repaint_takes_a_span_and_keeps_the_length_unless_it_goes_past_the_ends(self):
        _, seconds, result = self.made("drums", "--repaint", str(self.song), "--from", "1", "--to", "2")
        self.assertAlmostEqual(seconds, 3.0, places=3)
        self.assertIn("repaint", result.stderr)
        _, seconds, _ = self.made("drums", "--repaint", str(self.song), "--from", "1")              # to the end
        self.assertAlmostEqual(seconds, 3.0, places=3)
        _, seconds, _ = self.made("drums", "--repaint", str(self.song), "--from", "1", "--to", "5")  # 2 s past it
        self.assertAlmostEqual(seconds, 5.0, places=3)
        _, seconds, _ = self.made("drums", "--repaint", str(self.song), "--from", "-1.5", "--to", "2")
        self.assertAlmostEqual(seconds, 4.5, places=3)

    def test_a_repaint_s_mode_and_strength_reach_the_engine(self):
        base, _, _ = self.made("drums", "--repaint", str(self.song), "--from", "1", "--to", "2")
        songs = {base}
        for extra in (["--repaint-mode", "conservative"], ["--repaint-mode", "aggressive"], ["--repaint-strength", "0.8"]):
            songs.add(self.made("drums", "--repaint", str(self.song), "--from", "1", "--to", "2", *extra)[0])
        self.assertEqual(len(songs), 4)

    def test_a_reference_clip_alone_takes_the_style_and_keeps_the_duration(self):
        plain, _, _ = self.made("warm jazz", "--duration", "2")
        referenced, seconds, _ = self.made("warm jazz", "--duration", "2", "--ref", str(self.voice))
        self.assertAlmostEqual(seconds, 2.0, places=3)
        self.assertNotEqual(plain, referenced)

    def test_an_mp3_works_like_a_wav(self):
        mp3 = self.root / "song.mp3"
        mp3.write_bytes(make_mp3(80, id3=True))
        _, seconds, _ = self.made("--cover", str(mp3))
        self.assertAlmostEqual(seconds, 80 * 1152 / 44100, places=3)

    def test_run_takes_the_same_flags_on_a_music_model(self):
        out = self.root / "run.wav"
        result = run_coli(["run", "--cover", str(self.song), "--model", self.model, "--steps", "1", "--seed", "8", "--out", str(out)], self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertAlmostEqual(len(music_engine.decode_wav(out.read_bytes())[0]) / 4 / RATE, 3.0, places=3)

    def test_the_progress_line_says_it_is_reading_the_clip(self):
        self.assertEqual(self.load().MusicProgress.STAGES["encode"], "reading the clip")

    def load(self):
        return load_cli()

    # -- what is refused, and by which flag ---------------------------------------------------------------------
    def test_cover_and_repaint_are_two_tasks(self):
        text = self.refused("--cover", str(self.song), "--repaint", str(self.song), word="--cover and --repaint")
        self.assertIn("pick one", text)

    def test_a_flag_without_its_task_is_refused(self):
        for flags, word in ((["--from", "1"], "--from is for --repaint"), (["--to", "2"], "--to is for --repaint"),
                            (["--repaint-mode", "conservative"], "--repaint-mode is for --repaint"),
                            (["--repaint-strength", "0.5"], "--repaint-strength is for --repaint"),
                            (["--cover-strength", "0.5"], "--cover-strength is for --cover"),
                            (["--cover-noise", "0.5"], "--cover-noise is for --cover"),
                            (["--cover-strength", "0.5", "--repaint", str(self.song)], "--cover-strength is for --cover"),
                            (["--from", "1", "--cover", str(self.song)], "--from is for --repaint")):
            with self.subTest(flags=flags):
                self.refused("calm piano", *flags, word=word)

    def test_a_duration_is_refused_when_the_clip_decides_the_length(self):
        self.refused("--cover", str(self.song), "--duration", "5", word="--duration does not apply to a cover")
        self.refused("--repaint", str(self.song), "--duration", "5", word="--duration does not apply to a repaint")

    def test_a_clip_that_cannot_be_read_is_named_by_its_flag(self):
        missing = str(self.root / "none.wav")
        self.refused("--cover", missing, word="--cover: cannot read")
        self.refused("--repaint", missing, word="--repaint: cannot read")
        self.refused("calm piano", "--ref", missing, word="--ref: cannot read")
        self.refused("calm piano", "--ref", str(self.root), word="--ref: cannot read")          # a directory

    def test_a_format_that_is_not_wav_or_mp3_is_refused_by_name(self):
        for name, data, word in (("a.flac", b"fLaC" + bytes(64), "FLAC"), ("b.ogg", b"OggS" + bytes(64), "Ogg"),
                                 ("c.m4a", bytes(4) + b"ftypM4A " + bytes(60), "M4A"),
                                 ("d.wav", b"this is a text file named like a wav", "not a WAV or MP3")):
            path = self.root / name
            path.write_bytes(data)
            with self.subTest(name=name):
                text = self.refused("--cover", str(path), word="--cover")
                self.assertIn(word, text)
                self.assertIn("send a WAV or an MP3", text)
        self.refused("calm piano", "--ref", str(self.root / "a.flac"), word="--ref")

    def test_a_clip_that_is_too_short_or_empty_is_refused(self):
        short = self.root / "short.wav"
        short.write_bytes(make_wav(0.5))
        self.refused("--cover", str(short), word="shorter than 1 second")
        empty = self.root / "empty.wav"
        empty.write_bytes(b"")
        self.refused("--cover", str(empty), word="is empty")

    def test_a_file_over_the_cap_is_refused_before_it_is_read(self):
        cli = self.load()
        old = music_engine.MAX_CLIP_BYTES
        music_engine.MAX_CLIP_BYTES = 1000
        self.addCleanup(setattr, music_engine, "MAX_CLIP_BYTES", old)
        with self.assertRaises(SystemExit) as caught:
            cli.music_clip(None, "--cover", str(self.song))
        self.assertIn("--cover", str(caught.exception))
        self.assertIn("at most", str(caught.exception))

    def test_the_settings_have_the_ranges_the_request_says(self):
        for flags, word in ((["--cover", "SONG", "--cover-strength", "0"], "--cover-strength"),
                            (["--cover", "SONG", "--cover-strength", "1.5"], "--cover-strength"),
                            (["--cover", "SONG", "--cover-noise", "2"], "--cover-noise"),
                            (["--repaint", "SONG", "--repaint-strength", "0"], "--repaint-strength"),
                            (["--repaint", "SONG", "--repaint-strength", "3"], "--repaint-strength"),
                            (["--repaint", "SONG", "--from", "3", "--to", "3"], "--to"),
                            (["--repaint", "SONG", "--from", "5000"], "--from")):
            with self.subTest(flags=flags):
                flags = [str(self.song) if f == "SONG" else f for f in flags]
                self.refused("calm piano", *flags, word=word)

    def test_a_mode_that_does_not_exist_is_refused_by_the_parser(self):
        result = self.music("calm piano", "--repaint", str(self.song), "--repaint-mode", "wild")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("balanced", result.stderr)

    def test_an_engine_without_audio_in_says_so(self):
        env = dict(self.env, ACESTEP_STUB_NO_AUDIO_IN="1")
        self.refused("--cover", str(self.song), word="does not take audio in", env=env)
        self.refused("calm piano", "--ref", str(self.voice), word="does not take audio in", env=env)
        self.assertEqual(self.music("calm piano", "--duration", "1", "--out", str(self.out), env=env).returncode, 0)

    def test_text_to_music_still_needs_something_to_say(self):
        text = self.refused()
        self.assertIn("usage: coli music", text)
        self.assertIn("--cover", text)
        self.refused("--ref", str(self.voice), word="usage: coli music")                   # a reference alone is not a song

    # -- the help and the info ----------------------------------------------------------------------------------
    def test_help_lists_the_audio_flags(self):
        for command in ("music", "run"):
            r = run_coli([command, "--help"], self.env)
            self.assertEqual(r.returncode, 0, r.stderr)
            for flag in ("--cover", "--repaint", "--from", "--to", "--ref", "--cover-strength", "--cover-noise",
                         "--repaint-mode", "--repaint-strength"):
                with self.subTest(command=command, flag=flag):
                    self.assertIn(flag, r.stdout)

    def test_info_and_plan_say_what_the_model_can_do(self):
        info = run_coli(["info", "--model", self.model], self.env)
        self.assertEqual(info.returncode, 0, info.stderr)
        self.assertIn("covers and repaints", info.stdout)
        self.assertIn("--cover FILE", info.stdout)
        plan = run_coli(["plan", "--model", self.model], self.env)
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertIn("covers, repaints", plan.stdout)
        self.assertIn("with audio in", plan.stdout)
        self.assertIn("first use of audio in", plan.stdout)


if __name__ == "__main__":
    unittest.main()
