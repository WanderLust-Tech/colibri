"""ACE-Step audio in, from the request rules to the HTTP route and `coli music`, against the stub engine.

What is under test is everything the Python side adds to a clip: how a request names one (base64, a data: URL, a
multipart file), what is refused before the engine hears of it (a wrong format, a wrong task, a setting that belongs to
another task, a clip over the caps), where the clip is written for the engine and that the directory is gone however
the request ends, and what the stub engine (tools/acestep_stub.py, the protocol of c/acestep_serve.h) does with it.
The stub reads a clip's header or frames for its length and nothing else, so the tests pin lengths, task and
settings reaching the engine, stages and refusals, not what a cover sounds like.
"""
import base64
import http.client
import json
import os
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acestep_fixture import (RATE, STUB, make_fake_checkpoint, make_mp3, make_wav, multipart_body,  # noqa: E402
                             stub_env)
from test_acestep_gateway import GatewayHarness, KEY, MODEL_ID, parse_sse, wav_of  # noqa: E402

import music_engine  # noqa: E402
import openai_server  # noqa: E402

INFO = {"tasks": ["text2music", "cover", "repaint"],
        "audio_in": {"formats": ["wav", "mp3"], "max_seconds": 600, "ref_seconds": 30}}
B64 = lambda data: base64.b64encode(data).decode()  # noqa: E731
HUNDRED_SAMPLES = 2 * RATE * 2                      # a clip of N seconds at the stub's rate is N * RATE samples a channel


def make_request(**body):
    body.setdefault("prompt", "calm piano")
    return music_engine.music_request(body, INFO)


class Sniffing(unittest.TestCase):
    def test_the_formats_are_told_by_their_bytes(self):
        self.assertEqual(music_engine.audio_format(make_wav(1)), "wav")
        self.assertEqual(music_engine.audio_format(make_mp3(3)), "mp3")
        self.assertEqual(music_engine.audio_format(make_mp3(3, id3=True)), "mp3")
        self.assertEqual(music_engine.audio_format(make_mp3(3, junk=300)), "mp3")        # a few leading bytes before a frame
        self.assertEqual(music_engine.audio_format(b"fLaC" + bytes(40)), "flac")
        self.assertEqual(music_engine.audio_format(b"OggS" + bytes(40)), "ogg")
        self.assertEqual(music_engine.audio_format(bytes(4) + b"ftypM4A " + bytes(40)), "m4a")
        self.assertEqual(music_engine.audio_format(b"FORM\0\0\0\0AIFF" + bytes(40)), "aiff")
        for junk in (b"", b"hello world, this is text", os.urandom(0) + bytes(64), b"\xff\xff\xff\xff" * 8):
            with self.subTest(junk=junk[:10]):
                self.assertIsNone(music_engine.audio_format(junk))

    def test_a_frame_header_must_be_real(self):
        self.assertIsNone(music_engine.audio_format(bytes([0xFF, 0xE8, 0x90, 0x00]) + bytes(100)))   # version 1 is reserved
        self.assertIsNone(music_engine.audio_format(bytes([0xFF, 0xFD, 0x90, 0x00]) + bytes(100)))   # layer II
        self.assertIsNone(music_engine.audio_format(bytes([0xFF, 0xFB, 0xF0, 0x00]) + bytes(100)))   # bitrate 15
        self.assertIsNone(music_engine.audio_format(bytes([0xFF, 0xFB, 0x9C, 0x00]) + bytes(100)))   # sample rate 3

    def test_a_wav_says_how_long_it_is(self):
        self.assertAlmostEqual(music_engine.wav_seconds(make_wav(2.5)), 2.5, places=4)
        self.assertAlmostEqual(music_engine.wav_seconds(make_wav(1.0, rate=44100, channels=1)), 1.0, places=4)
        self.assertIsNone(music_engine.wav_seconds(b"RIFF\0\0\0\0WAVE"))
        self.assertIsNone(music_engine.wav_seconds(b"not a wav at all"))

    def test_a_data_chunk_longer_than_the_file_counts_what_is_there(self):
        wav = bytearray(make_wav(2.0))
        wav[40:44] = (0x7FFFFFF0).to_bytes(4, "little")                 # the data chunk claims 2 GB
        self.assertAlmostEqual(music_engine.wav_seconds(bytes(wav)), 2.0, places=4)
        self.assertAlmostEqual(music_engine.wav_seconds(bytes(wav[:len(wav) // 2])), 1.0, places=2)


class Requests(unittest.TestCase):
    def refused(self, param, info=INFO, **body):
        body.setdefault("prompt", "calm piano")
        with self.assertRaises(music_engine.MusicRequestError) as caught:
            music_engine.music_request(body, info)
        self.assertEqual(caught.exception.param, param, str(caught.exception))
        return str(caught.exception)

    def test_text_to_music_is_unchanged(self):
        request = make_request()
        self.assertEqual((request["task"], request["duration"], request["src_audio"], request["reference_audio"]),
                         ("text2music", 30, None, None))
        for name in ("repainting_start", "repainting_end", "repaint_mode", "repaint_strength", "audio_cover_strength",
                     "cover_noise_strength"):
            self.assertIsNone(request[name])

    def test_a_cover_takes_the_clip_as_base64_a_data_url_or_bytes_and_ignores_the_duration(self):
        wav = make_wav(1.5)
        for form in (B64(wav), "data:audio/wav;base64," + B64(wav), wav, bytearray(wav), B64(wav)[:40] + "\n" + B64(wav)[40:]):
            with self.subTest(form=type(form).__name__ + str(len(form))):
                request = make_request(task="cover", src_audio=form, duration="not a number")
                self.assertEqual(request["src_audio"], wav)
                self.assertEqual(request["task"], "cover")
                self.assertIsNone(request["duration"])            # the clip's length is the song's

    def test_a_cover_with_all_its_settings(self):
        request = make_request(task="cover", src_audio=B64(make_wav(1.5)), audio_cover_strength=0.5, cover_noise_strength=0,
                               reference_audio=B64(make_mp3(40)))
        self.assertEqual((request["audio_cover_strength"], request["cover_noise_strength"]), (0.5, 0.0))
        self.assertEqual(music_engine.audio_format(request["reference_audio"]), "mp3")

    def test_a_repaint_with_all_its_settings(self):
        request = make_request(task="repaint", src_audio=B64(make_wav(3)), repainting_start=1, repainting_end=2.5,
                               repaint_mode="conservative", repaint_strength=0.3)
        self.assertEqual((request["repainting_start"], request["repainting_end"], request["repaint_mode"],
                          request["repaint_strength"]), (1.0, 2.5, "conservative", 0.3))
        open_ended = make_request(task="repaint", src_audio=B64(make_wav(3)), repainting_start=-2, repainting_end=-1)
        self.assertEqual((open_ended["repainting_start"], open_ended["repainting_end"]), (-2.0, -1.0))   # an end of 0 or less is the clip's end

    def test_a_reference_clip_goes_with_text_to_music_and_keeps_its_duration(self):
        request = make_request(reference_audio=B64(make_wav(2)), duration=12)
        self.assertEqual((request["task"], request["duration"]), ("text2music", 12))
        self.assertIsNotNone(request["reference_audio"])
        self.refused("duration", reference_audio=B64(make_wav(2)), duration=0)

    def test_a_cover_or_a_repaint_need_not_describe_the_music_but_text_to_music_must(self):
        make_request(prompt="", task="cover", src_audio=B64(make_wav(1.5)))
        make_request(prompt="", task="repaint", src_audio=B64(make_wav(1.5)))
        self.refused("prompt", prompt="")
        self.refused("prompt", prompt="", reference_audio=B64(make_wav(1.5)))

    def test_the_task_rules(self):
        wav = B64(make_wav(1.5))
        cases = [
            ("task", dict(task="remix", src_audio=wav)),
            ("task", dict(task=3)),
            ("src_audio", dict(task="cover")),
            ("src_audio", dict(task="repaint")),
            ("src_audio", dict(src_audio=wav)),                                   # text to music takes a reference, not a source
            ("src_audio", dict(task="text2music", src_audio=wav)),
            ("audio_cover_strength", dict(audio_cover_strength=0.5)),             # a cover's setting without a cover
            ("cover_noise_strength", dict(cover_noise_strength=0.5)),
            ("repainting_start", dict(repainting_start=1)),
            ("repaint_mode", dict(repaint_mode="balanced")),
            ("audio_cover_strength", dict(task="repaint", src_audio=wav, audio_cover_strength=0.5)),
            ("cover_noise_strength", dict(task="repaint", src_audio=wav, cover_noise_strength=0.5)),
            ("repainting_end", dict(task="cover", src_audio=wav, repainting_end=3)),
            ("repaint_mode", dict(task="cover", src_audio=wav, repaint_mode="aggressive")),
            ("repaint_strength", dict(task="cover", src_audio=wav, repaint_strength=0.5)),
            ("repainting_start", dict(task="cover", src_audio=wav, repainting_start=0)),
        ]
        for param, body in cases:
            with self.subTest(param=param, body=sorted(body)):
                self.refused(param, **body)
        self.assertIn("reference clip", self.refused("src_audio", src_audio=wav))

    def test_the_settings_have_ranges(self):
        wav = B64(make_wav(1.5))
        cover = dict(task="cover", src_audio=wav)
        repaint = dict(task="repaint", src_audio=wav)
        cases = [
            ("audio_cover_strength", {**cover, "audio_cover_strength": 0}),       # 0 would be read as "not given" by the engine
            ("audio_cover_strength", {**cover, "audio_cover_strength": 1.01}),
            ("audio_cover_strength", {**cover, "audio_cover_strength": -0.5}),
            ("audio_cover_strength", {**cover, "audio_cover_strength": "0.5"}),
            ("audio_cover_strength", {**cover, "audio_cover_strength": True}),
            ("cover_noise_strength", {**cover, "cover_noise_strength": -0.1}),
            ("cover_noise_strength", {**cover, "cover_noise_strength": 1.1}),
            ("cover_noise_strength", {**cover, "cover_noise_strength": float("nan")}),
            ("repaint_strength", {**repaint, "repaint_strength": 0}),
            ("repaint_strength", {**repaint, "repaint_strength": 2}),
            ("repaint_strength", {**repaint, "repaint_strength": "x"}),
            ("repaint_mode", {**repaint, "repaint_mode": "wild"}),
            ("repaint_mode", {**repaint, "repaint_mode": 1}),
            ("repainting_start", {**repaint, "repainting_start": 3601}),
            ("repainting_start", {**repaint, "repainting_start": "soon"}),
            ("repainting_start", {**repaint, "repainting_start": True}),
            ("repainting_end", {**repaint, "repainting_end": -3601}),
            ("repainting_end", {**repaint, "repainting_end": 2, "repainting_start": 2}),
            ("repainting_end", {**repaint, "repainting_end": 2, "repainting_start": 5}),
        ]
        for param, body in cases:
            with self.subTest(param=param, body={k: v for k, v in body.items() if k != "src_audio"}):
                self.refused(param, **body)
        # and what is on the edge is taken
        make_request(**cover, audio_cover_strength=1, cover_noise_strength=1)
        make_request(**repaint, repaint_strength=1, repainting_start=-3600, repainting_end=3600)

    def test_a_clip_that_is_not_a_clip_is_refused_by_name(self):
        for param in ("src_audio", "reference_audio"):
            task = dict(task="cover") if param == "reference_audio" else dict(task="cover")
            good = {"src_audio": B64(make_wav(1.5))}
            cases = [
                (123, "must be a WAV or MP3"), ([1, 2], "must be a WAV or MP3"), ({"url": "x"}, "must be a WAV or MP3"),
                ("", "is empty"), ("   ", "is empty"), ("!!! not base64 !!!", "not valid base64"),
                ("http://example.com/song.wav", "does not fetch audio from URLs"),
                ("file:///etc/passwd", "does not fetch audio from URLs"),
                ("data:audio/wav,plain", "must be base64-encoded"),
                (B64(b"fLaC" + bytes(100)), "FLAC"), (B64(b"OggS" + bytes(100)), "Ogg"),
                (B64(bytes(4) + b"ftypM4A " + bytes(60)), "M4A"), (B64(b"FORM\0\0\0\0AIFF" + bytes(60)), "AIFF"),
                (B64(b"hello, this is text and not music at all"), "not a WAV or MP3"),
                (b"", "is empty"),
            ]
            for value, word in cases:
                with self.subTest(param=param, value=str(value)[:24]):
                    body = dict(task, **good)
                    body[param] = value
                    message = self.refused(param, **body)
                    self.assertIn(word, message)

    def test_a_wav_too_short_or_too_long_is_refused_before_the_engine(self):
        message = self.refused("src_audio", task="cover", src_audio=B64(make_wav(0.5)))
        self.assertIn("shorter than 1 second", message)
        short_limit = {**INFO, "audio_in": {**INFO["audio_in"], "max_seconds": 3}}
        message = self.refused("src_audio", info=short_limit, task="cover", src_audio=B64(make_wav(5)))
        self.assertIn("5 seconds long; at most 3", message)
        music_engine.music_request({"prompt": "x", "task": "cover", "src_audio": B64(make_wav(3))}, short_limit)

    def test_the_byte_caps_are_the_clip_and_the_pair(self):
        old = music_engine.MAX_CLIP_BYTES, music_engine.MAX_CLIPS_BYTES
        self.addCleanup(lambda: (setattr(music_engine, "MAX_CLIP_BYTES", old[0]), setattr(music_engine, "MAX_CLIPS_BYTES", old[1])))
        wav = make_wav(2)                                                    # 64 044 bytes
        music_engine.MAX_CLIP_BYTES, music_engine.MAX_CLIPS_BYTES = len(wav) - 1, 10 * len(wav)
        self.assertIn("MiB", self.refused("src_audio", task="cover", src_audio=wav))
        self.assertIn("MiB", self.refused("src_audio", task="cover", src_audio=B64(wav) * 2))     # refused by size before it is decoded
        music_engine.MAX_CLIP_BYTES, music_engine.MAX_CLIPS_BYTES = len(wav), len(wav) + len(wav) // 2
        make_request(task="cover", src_audio=wav)                            # one fits, exactly
        message = self.refused("reference_audio", task="cover", src_audio=wav, reference_audio=wav)
        self.assertIn("together", message)

    def test_the_default_caps_are_what_the_gateway_says_they_are(self):
        self.assertEqual((music_engine.MAX_CLIP_BYTES, music_engine.MAX_CLIPS_BYTES), (128 << 20, 192 << 20))
        self.assertEqual(openai_server.MUSIC_MAX_BODY_MULTIPART, (192 << 20) + (1 << 20))
        self.assertGreater(openai_server.MUSIC_MAX_BODY_JSON, (192 << 20) * 4 // 3)      # base64 of the pair, and the fields

    def test_an_engine_that_announced_no_tasks_takes_text_only(self):
        old = {"sample_rate": RATE}
        wav = B64(make_wav(1.5))
        self.assertIsNone(music_engine.audio_in_info(old))
        self.refused("task", info=old, task="cover", src_audio=wav)
        self.refused("reference_audio", info=old, reference_audio=wav)
        self.refused("audio_cover_strength", info=old, audio_cover_strength=0.5)
        music_engine.music_request({"prompt": "x"}, old)
        self.assertEqual(music_engine.audio_in_info(INFO)["max_seconds"], 600)
        self.assertEqual(music_engine.audio_in_info({"tasks": ["text2music", "cover"]})["ref_seconds"], 30)


# ---------------------------------------------------------------- the client against the stub
def open_engine(tmp, **stub):
    model = make_fake_checkpoint(Path(tmp) / "ace-step")
    return music_engine.MusicEngine(STUB, model, env=stub_env(0.0, **stub), on_log=lambda _m: None)


class Recorder:
    """The directories `_stage_clips` makes, remembered so a test can ask whether they are gone."""

    def __init__(self, test):
        self.dirs = []
        original = music_engine._stage_clips

        def staged(src, ref):
            staging = original(src, ref)
            self.dirs.append(staging["dir"])
            return staging
        music_engine._stage_clips = staged
        test.addCleanup(setattr, music_engine, "_stage_clips", original)

    def gone(self):
        return all(not os.path.exists(d) for d in self.dirs)

    def wait_gone(self, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.dirs and self.gone():
                return True
            time.sleep(0.05)
        return self.gone()


class Client(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = open_engine(self.tmp.name)
        self.addCleanup(self.engine.close, 5)
        self.recorder = Recorder(self)

    def song(self, **kw):
        kw.setdefault("caption", "calm piano")
        kw.setdefault("steps", 2)
        kw.setdefault("seed", 7)
        return self.engine.generate(**kw)

    def seconds(self, result):
        return result["samples"] / result["sample_rate"]

    def test_ready_announces_audio_in(self):
        self.assertEqual(self.engine.tasks, ("text2music", "cover", "repaint"))
        self.assertEqual(self.engine.info["audio_in"]["max_seconds"], 600)

    def test_a_cover_is_as_long_as_its_source_and_says_so(self):
        result = self.song(task="cover", src_audio=make_wav(2.5))
        self.assertAlmostEqual(self.seconds(result), 2.5, places=3)
        self.assertAlmostEqual(result["src_seconds"], 2.5, places=3)
        self.assertEqual(len(result["pcm"]), result["samples"] * 4)
        long = self.song(task="cover", src_audio=make_wav(10))              # 10 s is 250 frames, over the floor of 128
        self.assertAlmostEqual(self.seconds(long), 10.0, places=3)

    def test_a_text_song_reports_no_source(self):
        self.assertEqual(self.song(duration=2)["src_seconds"], 0)

    def test_the_encode_stage_comes_first_and_only_with_a_clip(self):
        with_clip, without = [], []
        self.song(task="cover", src_audio=make_wav(1.5), on_progress=lambda f: with_clip.append(f["stage"]))
        self.song(duration=1, on_progress=lambda f: without.append(f["stage"]))
        self.assertEqual(with_clip[0], "encode")
        self.assertEqual([s for i, s in enumerate(with_clip) if i == 0 or with_clip[i - 1] != s],
                         ["encode", "text", "condition", "denoise", "decode"])
        self.assertNotIn("encode", without)

    def test_a_repaint_adds_silence_before_and_after_the_clip(self):
        plain = self.song(task="repaint", src_audio=make_wav(3), repaint_start=1, repaint_end=2)
        self.assertAlmostEqual(self.seconds(plain), 3.0, places=3)
        after = self.song(task="repaint", src_audio=make_wav(3), repaint_start=1, repaint_end=5)        # 2 s past the end
        self.assertAlmostEqual(self.seconds(after), 5.0, places=3)
        before = self.song(task="repaint", src_audio=make_wav(3), repaint_start=-1.5, repaint_end=2)
        self.assertAlmostEqual(self.seconds(before), 4.5, places=3)
        self.assertAlmostEqual(before["src_seconds"], 3.0, places=3)
        to_the_end = self.song(task="repaint", src_audio=make_wav(3), repaint_start=1, repaint_end=-1)
        self.assertAlmostEqual(self.seconds(to_the_end), 3.0, places=3)

    def test_a_reference_clip_changes_the_song_and_not_its_length(self):
        plain = self.song(duration=2)
        referenced = self.song(duration=2, ref_audio=make_wav(3, frequency=300))
        other = self.song(duration=2, ref_audio=make_wav(3, frequency=900))
        self.assertAlmostEqual(self.seconds(referenced), 2.0, places=3)
        self.assertEqual(len({plain["pcm"], referenced["pcm"], other["pcm"]}), 3)
        self.assertEqual(self.song(duration=2, ref_audio=make_wav(3, frequency=300))["pcm"], referenced["pcm"])

    def test_the_task_and_every_setting_reach_the_engine(self):
        base = dict(task="cover", src_audio=make_wav(2))
        songs = {self.song(**base)["pcm"]}
        for extra in (dict(cover_strength=0.5), dict(cover_noise=0.4), dict(ref_audio=make_wav(2, frequency=500))):
            songs.add(self.song(**base, **extra)["pcm"])
        repaint = dict(task="repaint", src_audio=make_wav(4), repaint_start=1, repaint_end=3)
        songs.add(self.song(**repaint)["pcm"])
        for extra in (dict(repaint_mode="aggressive"), dict(repaint_strength=0.8), dict(repaint_start=0.5), dict(repaint_end=3.5)):
            songs.add(self.song(**{**repaint, **extra})["pcm"])
        self.assertEqual(len(songs), 9)                                       # every one of them made a different song

    def test_an_mp3_source_is_measured_by_its_frames(self):
        result = self.song(task="cover", src_audio=make_mp3(40, id3=True))
        self.assertAlmostEqual(result["src_seconds"], 40 * 1152 / 44100, places=3)

    def test_the_clip_is_in_a_private_directory_while_the_engine_reads_it(self):
        seen = []

        def look(_frame):
            for d in self.recorder.dirs:
                files = sorted(os.listdir(d)) if os.path.isdir(d) else []
                mode = stat.S_IMODE(os.stat(d).st_mode) if os.path.isdir(d) else None
                seen.append((files, mode, [stat.S_IMODE(os.stat(os.path.join(d, f)).st_mode) for f in files]))
        self.song(task="cover", src_audio=make_wav(1.5), ref_audio=make_wav(1.5), on_progress=look)
        self.assertTrue(seen)
        files, mode, file_modes = seen[0]
        self.assertEqual(len(files), 2)
        self.assertTrue(files[0].startswith("reference") and files[1].startswith("source"))
        if os.name == "posix":
            self.assertEqual(mode, 0o700)
            self.assertEqual(file_modes, [0o600, 0o600])

    def test_the_directory_is_gone_when_the_song_is_made(self):
        self.song(task="cover", src_audio=make_wav(1.5))
        self.assertEqual(len(self.recorder.dirs), 1)
        self.assertTrue(self.recorder.gone())

    def test_the_directory_is_gone_when_the_engine_refuses_the_clip(self):
        with self.assertRaises(music_engine.MusicEngineError) as caught:
            self.song(task="cover", src_audio=make_wav(0.5))
        self.assertIn("the source clip is shorter than one second", str(caught.exception))
        self.assertTrue(self.recorder.gone())
        self.song(duration=1)                                                 # the engine is still there

    def test_the_directory_is_gone_when_the_request_is_cancelled(self):
        with self.assertRaises(music_engine.MusicCancelled):
            self.song(task="cover", src_audio=make_wav(2), steps=8, cancelled=lambda: True)
        self.assertEqual(len(self.recorder.dirs), 1)
        self.assertTrue(self.recorder.gone())
        self.song(duration=1)

    def test_a_request_left_without_its_final_frame_keeps_its_directory_until_the_next_one(self):
        def boom(_frame):
            raise RuntimeError("the caller lost interest")
        with self.assertRaises(RuntimeError):
            self.song(task="cover", src_audio=make_wav(2), steps=8, on_progress=boom)
        self.assertFalse(self.recorder.gone())                                # the engine may still be reading it
        self.song(duration=1)                                                 # the next request drains the abandoned one first
        self.assertTrue(self.recorder.gone())

    def test_closing_the_engine_removes_what_an_abandoned_request_left(self):
        def boom(_frame):
            raise RuntimeError("gone")
        with self.assertRaises(RuntimeError):
            self.song(task="cover", src_audio=make_wav(2), steps=8, on_progress=boom)
        self.engine.close(5)
        self.assertTrue(self.recorder.gone())

    def test_nothing_is_written_when_the_engine_cannot_take_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = open_engine(tmp, ACESTEP_STUB_NO_AUDIO_IN="1")
            self.addCleanup(engine.close, 5)
            self.assertEqual(engine.tasks, ("text2music",))
            for kw in (dict(task="cover", src_audio=make_wav(1.5)), dict(ref_audio=make_wav(1.5))):
                with self.assertRaises(music_engine.MusicEngineError) as caught:
                    engine.generate("x", **kw)
                self.assertIn("does not take audio in", str(caught.exception))
            self.assertEqual(self.recorder.dirs, [])
            self.assertEqual(engine.generate("x", duration=1, steps=1, seed=1)["samples"], RATE)

    def test_the_stub_refuses_what_the_engine_refuses_in_its_words(self):
        wav = make_wav(1.5)
        cases = [
            (dict(src_audio=wav), "a source clip is for a cover or a repaint; for text to music, give a reference clip (ref) to take its style"),
            (dict(task="cover"), "a cover needs a source clip (src)"),
            (dict(task="repaint"), "a repaint needs a source clip (src)"),
            (dict(task="repaint", src_audio=wav, cover_strength=0.5), "cover_strength and cover_noise are for a cover"),
            (dict(task="cover", src_audio=wav, repaint_start=1), "the repaint settings are for a repaint"),
            (dict(task="cover", src_audio=wav, cover_strength=1.5), "cover_strength and cover_noise must be 0 to 1"),
            (dict(task="repaint", src_audio=wav, repaint_strength=2), "repaint_strength must be 0 to 1"),
            (dict(task="repaint", src_audio=wav, repaint_mode="wild"), "repaint_mode must be balanced, conservative or aggressive"),
            (dict(task="repaint", src_audio=wav, repaint_start=1.4, repaint_end=1.2), "the repaint must end after it starts (1.4 to 1.2 s)"),
            (dict(task="repaint", src_audio=wav, repaint_start=2, repaint_end=5), "the repaint starts at 2 s, past the end of the 1.50 s clip"),
            (dict(task="repaint", src_audio=wav, repaint_start=-4000), "the repaint span is out of range"),
            (dict(task="cover", src_audio=b"not a clip, just text"), "the source clip: not a WAV or an MP3"),
            (dict(task="cover", src_audio=make_wav(1.5), ref_audio=make_wav(1.5, silent=True)), "the reference clip is silent"),
            (dict(ref_audio=b"not a clip, just text"), "the reference clip: not a WAV or an MP3"),
        ]
        for kw, message in cases:
            with self.subTest(message=message[:50]):
                with self.assertRaises(music_engine.MusicEngineError) as caught:
                    self.song(**kw)
                self.assertIn(message, str(caught.exception))
        self.assertTrue(self.recorder.gone())
        self.song(duration=1)                                                 # all of that left the engine alive

    def test_a_repaint_is_refused_when_it_would_pass_ten_minutes(self):
        with self.assertRaises(music_engine.MusicEngineError) as caught:
            self.song(task="repaint", src_audio=make_wav(3), repaint_start=-598, repaint_end=2)
        self.assertIn("longer than 600 seconds", str(caught.exception))


# ---------------------------------------------------------------- the gateway
def post_raw(gw, content_type, body, key=KEY):
    connection = http.client.HTTPConnection("127.0.0.1", gw.port, timeout=60)
    try:
        connection.request("POST", "/v1/audio/music", body=body,
                           headers={"Content-Type": content_type, "Authorization": f"Bearer {key}"})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def declare_length(gw, content_type, length, wait=30):
    """(status, body) of a POST that says it carries `length` bytes and sends none of them; (None, b"") when the
    server answers nothing within `wait` seconds (it is waiting for the body it was promised)."""
    sock = socket.create_connection(("127.0.0.1", gw.port), timeout=wait)
    try:
        sock.sendall((f"POST /v1/audio/music HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {KEY}\r\n"
                      f"Content-Type: {content_type}\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n").encode())
        data = b""
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
        except socket.timeout:
            return None, b""
        head, _, body = data.partition(b"\r\n\r\n")
        return int(head.split()[1]), body
    finally:
        sock.close()


class Gateway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def setUp(self):
        self.recorder = Recorder(self)

    def music(self, **body):
        status, headers, raw = self.gw.music(**body)
        return status, headers, raw

    def ok(self, **body):
        status, _, raw = self.music(**body)
        self.assertEqual(status, 200, raw[:300])
        return json.loads(raw)

    def refused(self, param, **body):
        status, _, raw = self.music(**body)
        self.assertEqual(status, 400, raw[:300])
        error = json.loads(raw)["error"]
        self.assertEqual(error["param"], param, error)
        return error["message"]

    def seconds_of(self, payload):
        pcm, rate, channels = wav_of(payload["data"][0]["b64_json"])
        return len(pcm) / (2 * channels) / rate

    def test_a_cover_is_as_long_as_the_clip_and_the_answer_says_what_was_done(self):
        payload = self.ok(prompt="warm jazz", task="cover", src_audio=B64(make_wav(2.5)), steps=2, seed=3)
        self.assertAlmostEqual(self.seconds_of(payload), 2.5, places=3)
        info = payload["colibri"]
        self.assertEqual(info["task"], "cover")
        self.assertAlmostEqual(info["src_seconds"], 2.5, places=3)
        self.assertTrue(self.recorder.gone())

    def test_a_repaint_and_its_span(self):
        payload = self.ok(prompt="drums", task="repaint", src_audio=B64(make_wav(3)), repainting_start=1, repainting_end=2,
                          repaint_mode="conservative", repaint_strength=0.4, steps=2, seed=3)
        self.assertAlmostEqual(self.seconds_of(payload), 3.0, places=3)
        self.assertEqual(payload["colibri"]["task"], "repaint")
        extended = self.ok(prompt="drums", task="repaint", src_audio=B64(make_wav(3)), repainting_start=1, repainting_end=5,
                           steps=2, seed=3)
        self.assertAlmostEqual(self.seconds_of(extended), 5.0, places=3)

    def test_a_reference_clip_with_text_to_music(self):
        plain = self.ok(prompt="warm jazz", duration=2, steps=2, seed=3)
        referenced = self.ok(prompt="warm jazz", duration=2, steps=2, seed=3, reference_audio=B64(make_wav(3)))
        self.assertAlmostEqual(self.seconds_of(referenced), 2.0, places=3)
        self.assertEqual((plain["colibri"]["task"], referenced["colibri"]["task"]), ("text2music", "text2music"))
        self.assertNotEqual(plain["data"][0]["b64_json"], referenced["data"][0]["b64_json"])

    def test_the_clip_may_be_sent_as_a_data_url_and_as_an_mp3(self):
        a = self.ok(prompt="x", task="cover", src_audio="data:audio/wav;base64," + B64(make_wav(2)), steps=1, seed=1)
        b = self.ok(prompt="x", task="cover", src_audio=B64(make_wav(2)), steps=1, seed=1)
        self.assertEqual(a["data"], b["data"])
        mp3 = self.ok(prompt="x", task="cover", src_audio=B64(make_mp3(40, id3=True)), steps=1, seed=1)
        self.assertAlmostEqual(mp3["colibri"]["src_seconds"], 40 * 1152 / 44100, places=3)

    def test_multipart_gives_the_same_song_as_json(self):
        wav, ref = make_wav(2), make_wav(2, frequency=700)
        via_json = self.ok(prompt="warm jazz", task="cover", src_audio=B64(wav), reference_audio=B64(ref), steps=2, seed=5,
                           audio_cover_strength=0.5, cover_noise_strength=0.25)
        content_type, body = multipart_body(
            {"model": MODEL_ID, "prompt": "warm jazz", "task": "cover", "steps": 2, "seed": 5, "audio_cover_strength": 0.5,
             "cover_noise_strength": 0.25},
            [("src_audio", "song.wav", wav), ("reference_audio", "voice.wav", ref)])
        status, _, raw = post_raw(self.gw, content_type, body)
        self.assertEqual(status, 200, raw[:300])
        self.assertEqual(json.loads(raw)["data"], via_json["data"])
        self.assertTrue(self.recorder.gone())

    def test_multipart_numbers_and_flags_are_checked_like_json(self):
        wav = make_wav(2)

        def send(fields, files=None):
            content_type, body = multipart_body({"model": MODEL_ID, "prompt": "x", **fields},
                                                [("src_audio", "s.wav", wav)] if files is None else files)
            status, _, raw = post_raw(self.gw, content_type, body)
            return status, json.loads(raw)
        status, answer = send({"task": "cover", "steps": "lots"})
        self.assertEqual((status, answer["error"]["param"]), (400, "steps"))
        status, answer = send({"task": "cover", "audio_cover_strength": "high"})
        self.assertEqual((status, answer["error"]["param"]), (400, "audio_cover_strength"))
        status, answer = send({"task": "cover", "dcw": "maybe"})
        self.assertEqual((status, answer["error"]["param"]), (400, "dcw"))
        status, answer = send({"task": "cover", "audio_cover_strength": 0})
        self.assertEqual((status, answer["error"]["param"]), (400, "audio_cover_strength"))
        status, answer = send({"task": "cover"}, files=[("src_audio", "a.wav", wav), ("src_audio", "b.wav", wav)])
        self.assertEqual((status, answer["error"]["param"]), (400, "src_audio"))
        self.assertIn("2 times", answer["error"]["message"])
        status, answer = send({"task": "cover", "src_audio": B64(wav)})
        self.assertEqual((status, answer["error"]["param"]), (400, "src_audio"))
        status, answer = send({"task": "cover", "steps": 1, "seed": 1, "dcw": "false"})
        self.assertEqual(status, 200)
        status, answer = send({"task": "text2music"}, files=[])                    # no clip, a plain song, by multipart
        self.assertEqual(status, 200)

    def test_streaming_says_it_is_reading_the_clip_first(self):
        status, _, events = self.gw.stream(prompt="x", task="cover", src_audio=B64(make_wav(1.5)), steps=2, seed=2)
        self.assertEqual(status, 200)
        stages = [d["stage"] for name, d in events if name == "music_generation.progress"]
        self.assertEqual(stages[0], "encode")
        self.assertEqual([s for i, s in enumerate(stages) if i == 0 or stages[i - 1] != s],
                         ["encode", "text", "condition", "denoise", "decode"])
        done = [d for name, d in events if name == "music_generation.completed"][0]
        self.assertEqual((done["colibri"]["task"], round(done["colibri"]["src_seconds"], 2)), ("cover", 1.5))
        self.assertEqual(events[-1], (None, "[DONE]"))
        self.assertTrue(self.recorder.gone())

    def test_a_wav_can_come_back_as_the_file_and_names_nothing_of_the_clip(self):
        status, headers, raw = self.music(prompt="x", task="cover", src_audio=B64(make_wav(2)), steps=1, seed=4,
                                          response_format="wav")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "audio/wav")
        self.assertEqual(music_engine.decode_wav(raw)[1], RATE)

    def test_what_is_refused_before_the_engine_is_asked(self):
        wav = B64(make_wav(2))
        self.assertIn("reference clip", self.refused("src_audio", prompt="x", src_audio=wav))
        self.assertIn("required", self.refused("src_audio", prompt="x", task="cover"))
        self.refused("task", prompt="x", task="remix", src_audio=wav)
        self.refused("audio_cover_strength", prompt="x", task="repaint", src_audio=wav, audio_cover_strength=0.5)
        self.refused("repaint_mode", prompt="x", task="repaint", src_audio=wav, repaint_mode="wild")
        self.refused("src_audio", prompt="x", task="cover", src_audio="http://example.com/a.wav")
        self.assertIn("FLAC", self.refused("src_audio", prompt="x", task="cover", src_audio=B64(b"fLaC" + bytes(64))))
        self.assertIn("not valid base64", self.refused("reference_audio", prompt="x", reference_audio="@@@"))
        self.assertIn("shorter than 1 second", self.refused("src_audio", prompt="x", task="cover", src_audio=B64(make_wav(0.5))))
        self.assertEqual(self.recorder.dirs, [])                                  # none of those wrote a file

    def test_the_engines_own_refusal_is_a_500_with_its_words_and_leaves_no_file(self):
        status, _, raw = self.music(prompt="x", task="repaint", src_audio=B64(make_wav(3)), repainting_start=-598,
                                    repainting_end=2, steps=1)
        self.assertEqual(status, 500, raw[:300])
        self.assertIn("longer than 600 seconds", json.loads(raw)["error"]["message"])
        self.assertTrue(self.recorder.gone())
        self.ok(prompt="x", duration=1, steps=1, seed=1)                          # and the engine is fine

    def test_the_model_card_and_health_say_what_audio_in_takes(self):
        status, _, raw = self.gw.request("/v1/models", method="GET")
        self.assertEqual(status, 200)
        entry = json.loads(raw)["data"][0]
        self.assertEqual(entry["capabilities"], ["music_generation", "audio_input"])
        self.assertEqual(entry["audio"]["tasks"], ["text2music", "cover", "repaint"])
        audio_in = entry["audio"]["audio_in"]
        self.assertEqual((audio_in["formats"], audio_in["max_seconds"], audio_in["ref_seconds"]), (["wav", "mp3"], 600, 30))
        self.assertEqual((audio_in["max_clip_bytes"], audio_in["max_total_bytes"]), (128 << 20, 192 << 20))
        self.assertEqual(audio_in["multipart_fields"], ["src_audio", "reference_audio"])
        status, _, raw = self.gw.request("/health", method="GET")
        health = json.loads(raw)
        self.assertEqual(health["audio"]["tasks"], ["text2music", "cover", "repaint"])
        self.assertEqual(health["audio"]["audio_in"]["max_seconds"], 600)

    def test_a_client_that_hangs_up_mid_stream_leaves_no_clip_behind(self):
        gw = GatewayHarness(delay=0.15)
        self.addCleanup(gw.close)
        recorder = Recorder(self)
        content_type, body = multipart_body(
            {"model": MODEL_ID, "prompt": "x", "task": "cover", "steps": 8, "seed": 1, "stream": "true"},
            [("src_audio", "song.wav", make_wav(2))])
        sock = socket.create_connection(("127.0.0.1", gw.port), timeout=30)
        head = (f"POST /v1/audio/music HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {KEY}\r\nContent-Type: {content_type}\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
        sock.sendall(head + body)
        seen = b""
        while b"denoise" not in seen:                                              # the stream is going: the engine has the clip
            chunk = sock.recv(4096)
            self.assertTrue(chunk, seen)
            seen += chunk
        self.assertTrue(recorder.dirs)
        self.assertFalse(recorder.gone())
        sock.close()
        self.assertTrue(recorder.wait_gone(20), "the clip directory was left behind after the client hung up")
        status, _, raw = gw.music(prompt="x", duration=1, steps=1, seed=1)         # the engine was cancelled, not wedged
        self.assertEqual(status, 200, raw[:200])


class GatewayWithoutAudioIn(unittest.TestCase):
    """An engine that announced no tasks (an older build): music from text only, and the cheap body limit."""

    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness(ACESTEP_STUB_NO_AUDIO_IN="1")

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def test_the_card_says_text_only(self):
        entry = json.loads(self.gw.request("/v1/models", method="GET")[2])["data"][0]
        self.assertEqual(entry["capabilities"], ["music_generation"])
        self.assertEqual(entry["audio"]["tasks"], ["text2music"])
        self.assertIsNone(entry["audio"]["audio_in"])

    def test_a_clip_is_refused_by_the_gateway(self):
        status, _, raw = self.gw.music(prompt="x", task="cover", src_audio=B64(make_wav(2)))
        self.assertEqual(status, 400, raw[:200])
        self.assertIn("does not take audio in", json.loads(raw)["error"]["message"])
        status, _, raw = self.gw.music(prompt="x", reference_audio=B64(make_wav(2)))
        self.assertEqual(status, 400)
        self.assertEqual(self.gw.music(prompt="x", duration=1, steps=1, seed=1)[0], 200)

    def test_the_big_body_limit_is_only_for_an_engine_that_takes_audio(self):
        # a length over the limit is refused from the header alone, before a byte of the body is read
        for content_type in ("application/json", "multipart/form-data; boundary=zzz"):
            with self.subTest(content_type=content_type):
                status, raw = declare_length(self.gw, content_type, 5 << 20)
                self.assertEqual(status, 400, raw[:200])
                self.assertIn("between 1 and 4194304", json.loads(raw)["error"]["message"])


class GatewayBodyLimits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def test_the_audio_engine_allows_what_the_other_one_does_not(self):
        for content_type in ("application/json", "multipart/form-data; boundary=zzz"):
            with self.subTest(content_type=content_type):
                status, _ = declare_length(self.gw, content_type, 5 << 20, wait=1.0)
                self.assertIsNone(status)             # no refusal: it is waiting for the 5 MiB it was promised

    def test_a_body_over_the_limit_is_refused_unread_and_one_under_it_is_taken(self):
        old = openai_server.MUSIC_MAX_BODY_MULTIPART, openai_server.MUSIC_MAX_BODY_JSON
        self.addCleanup(lambda: (setattr(openai_server, "MUSIC_MAX_BODY_MULTIPART", old[0]),
                                 setattr(openai_server, "MUSIC_MAX_BODY_JSON", old[1])))
        wav = make_wav(2)
        content_type, body = multipart_body({"model": MODEL_ID, "prompt": "x", "task": "cover", "steps": 1, "seed": 1},
                                            [("src_audio", "s.wav", wav)])
        openai_server.MUSIC_MAX_BODY_MULTIPART = len(body) - 1
        status, _, raw = post_raw(self.gw, content_type, body)
        self.assertEqual(status, 400, raw[:200])
        self.assertIn("between 1 and", json.loads(raw)["error"]["message"])
        openai_server.MUSIC_MAX_BODY_MULTIPART = len(body)
        self.assertEqual(post_raw(self.gw, content_type, body)[0], 200)
        as_json = json.dumps({"model": MODEL_ID, "prompt": "x", "task": "cover", "steps": 1, "seed": 1,
                              "src_audio": B64(wav)}).encode()
        openai_server.MUSIC_MAX_BODY_JSON = len(as_json) - 1
        self.assertEqual(post_raw(self.gw, "application/json", as_json)[0], 400)
        openai_server.MUSIC_MAX_BODY_JSON = len(as_json)
        self.assertEqual(post_raw(self.gw, "application/json", as_json)[0], 200)

    def test_a_multipart_body_without_parts_or_a_name_is_refused(self):
        status, _, raw = post_raw(self.gw, "multipart/form-data; boundary=zzz", b"nothing like a form")
        self.assertEqual(status, 400, raw[:200])
        body = b'--zzz\r\nContent-Disposition: form-data\r\n\r\nvalue\r\n--zzz--\r\n'
        status, _, raw = post_raw(self.gw, "multipart/form-data; boundary=zzz", body)
        self.assertEqual(status, 400)
        self.assertIn("no name", json.loads(raw)["error"]["message"])


if __name__ == "__main__":
    unittest.main()
