"""Gemma 4's other audio formats and its video (tools/gemma4_media.py, and the gateway and renderer around it).

    python -m unittest tests.test_gemma4_media

What needs nothing: the sniffing of audio files, the frame-sampling rule, the times, the PNG splitter, how a video becomes a
prompt, and every refusal. What needs something is skipped without it: the MP3 reader needs the built `acestep_audio_tool` (or
ffmpeg), the video and FLAC cases need an ffmpeg (COLI_FFMPEG, the PATH, or the one in the imageio-ffmpeg package), and the
comparisons with transformers need it and torchvision.
"""
import base64
import io
import math
import os
import struct
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE.parent))

import gemma4_chat as g  # noqa: E402
import gemma4_media as media  # noqa: E402
import openai_server  # noqa: E402

try:
    import numpy as np
    from PIL import Image
    import gemma4_audio  # noqa: E402
    import gemma4_image  # noqa: E402
    HAVE_IMAGING = True
except ImportError:                                   # pragma: no cover
    HAVE_IMAGING = False

TONE = HERE / "fixtures" / "tone_440hz_16k.mp3"


def find_ffmpeg():
    """An ffmpeg for the tests that need one: the environment's, or the imageio-ffmpeg package's."""
    exe = media.ffmpeg_path()
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = find_ffmpeg()


def png(width, height, colour):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, "PNG")
    return buf.getvalue()


def data_url(raw, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


class SniffTest(unittest.TestCase):
    def test_a_file_is_what_its_first_bytes_say(self):
        cases = {"wav": b"RIFF\x24\x00\x00\x00WAVEfmt ", "mp3": b"ID3\x04\x00\x00\x00\x00\x00\x00", "flac": b"fLaC\x00\x00\x00\x22",
                 "ogg": b"OggS\x00\x02\x00\x00", "m4a": b"\x00\x00\x00\x20ftypM4A ", "aiff": b"FORM\x00\x00\x00\x00AIFFCOMM",
                 "webm": b"\x1a\x45\xdf\xa3\x01\x00\x00\x00", "unknown": b"hello, this is text"}
        for kind, head in cases.items():
            self.assertEqual(media.audio_kind(head), kind, kind)
        self.assertEqual(media.audio_kind(b"\xff\xfb\x90\x00" + bytes(12)), "mp3")        # a frame header with no ID3 tag in front
        self.assertEqual(media.audio_kind(b""), "unknown")
        self.assertEqual(media.audio_kind(bytes([0xFF, 0x00, 0x90, 0x00]) + bytes(12)), "unknown")        # 0xFF alone is not a frame sync

    def test_a_clip_nobody_can_read_is_refused_by_name(self):
        with patch.dict(os.environ, {"COLI_FFMPEG": os.devnull + ".none"}):
            for head, word in ((b"fLaC" + bytes(20), "FLAC"), (b"OggS" + bytes(20), "OGG")):
                with self.assertRaises(media.MediaError) as caught:
                    media.decode_audio(head)
                self.assertIn(word, str(caught.exception))
                self.assertIn("ffmpeg", str(caught.exception))
            with self.assertRaises(media.MediaError) as caught:
                media.decode_audio(b"just some words that are not audio at all")
            self.assertIn("not audio", str(caught.exception))


class SamplingTest(unittest.TestCase):
    def test_indices_are_transformers_uniform_rule(self):
        for total, n in ((100, 32), (32, 32), (33, 32), (50, 32), (3600, 32), (1, 32), (10, 4)):
            k = min(n, total)
            want = [i * total // k for i in range(k)]
            self.assertEqual(media.sample_indices(total, n), want, (total, n))
        self.assertEqual(media.sample_indices(50, 32)[:4], [0, 1, 3, 4])
        self.assertEqual(media.sample_indices(10), list(range(10)))                       # a short video: every frame
        with self.assertRaises(media.MediaError):
            media.sample_indices(0)

    def test_the_rule_is_the_one_in_transformers(self):
        try:
            from transformers.video_utils import default_sample_indices_fn
            from transformers.video_utils import VideoMetadata
        except Exception:
            self.skipTest("transformers is not installed")
        for total in (32, 33, 50, 97, 1000):
            meta = VideoMetadata(total_num_frames=total, fps=24.0)
            self.assertEqual(media.sample_indices(total, 32), [int(i) for i in default_sample_indices_fn(meta, num_frames=32)])

    def test_times_are_minutes_and_seconds_not_reduced(self):
        self.assertEqual([media.timestamp_text(s) for s in (0, 0.99, 1, 59.9, 60, 75.2, 3725)], ["00:00", "00:00", "00:01", "00:59", "01:00", "01:15", "62:05"])

    def test_png_files_in_a_stream_are_told_apart(self):
        if not HAVE_IMAGING:
            self.skipTest("needs Pillow")
        one, two, three = png(8, 8, "red"), png(16, 4, "blue"), png(2, 2, "green")
        self.assertEqual(media._split_png(one + two + three), [one, two, three])
        self.assertEqual(media._split_png(b""), [])
        self.assertEqual(media._split_png(b"not a picture"), [])


@unittest.skipUnless(HAVE_IMAGING, "needs numpy and Pillow")
class VideoPromptTest(unittest.TestCase):
    """A video becomes frames, one IMAGE frame each, and the processor's text: 'mm:ss <|image>soft tokens<image|>' per frame, joined by spaces."""

    def expand(self, part, limit=None):
        messages = [{"role": "user", "content": [{"type": "text", "text": "what happens?"}, part]}]
        return openai_server.expand_gemma4_images(messages, None, limit)

    def test_frames_a_client_took_become_the_processors_text(self):
        frames = [data_url(png(320, 240, c)) for c in ("red", "green", "blue")]
        messages, pictures = self.expand({"type": "video", "video": {"frames": frames, "fps": 2}})
        part = messages[0]["content"][1]
        self.assertEqual(part["type"], "video")
        self.assertEqual(len(pictures), 3)
        self.assertEqual(part["_times"], ["00:00", "00:00", "00:01"])
        sizes = []
        for picture, entry in zip(pictures, part["_frames"]):
            patches, grid_h, grid_w = picture
            self.assertEqual(entry["_soft_tokens"], (grid_h // 3) * (grid_w // 3))
            self.assertLessEqual(entry["_soft_tokens"], 70)
            sizes.append(entry["_soft_tokens"])
        self.assertEqual([e["_image_index"] for e in part["_frames"]], [0, 1, 2])
        order = []
        prompt = g.render(messages, image_order=order)
        want = " ".join(f"{t} <|image>{'<|image|>' * n}<image|>" for t, n in zip(["00:00", "00:00", "00:01"], sizes))
        self.assertIn("<|turn>user\nwhat happens?" + want + "<turn|>", prompt)
        self.assertEqual(order, [0, 1, 2])

    def test_a_frame_is_at_most_70_soft_tokens_and_never_more_than_the_engine_takes(self):
        big = data_url(png(1280, 720, "white"))
        messages, _ = self.expand({"type": "video", "video": {"frames": [big]}})
        self.assertLessEqual(messages[0]["content"][1]["_frames"][0]["_soft_tokens"], 70)
        messages, _ = self.expand({"type": "video", "video": {"frames": [big]}}, limit=60)
        self.assertLessEqual(messages[0]["content"][1]["_frames"][0]["_soft_tokens"], 60)

    def test_more_than_32_frames_are_thinned_evenly_and_keep_their_times(self):
        frames = [data_url(png(96, 96, (i * 5, 0, 0))) for i in range(50)]
        messages, pictures = self.expand({"type": "video", "video": {"frames": frames, "fps": 10}})
        keep = media.sample_indices(50, 32)
        self.assertEqual(len(pictures), 32)
        self.assertEqual(messages[0]["content"][1]["_times"], [media.timestamp_text(i / 10) for i in keep])

    def test_two_videos_and_a_picture_keep_their_order(self):
        clip = {"type": "video", "video": {"frames": [data_url(png(96, 96, "red"))] * 2}}
        picture = {"type": "image_url", "image_url": {"url": data_url(png(96, 96, "blue"))}}
        messages = [{"role": "user", "content": [clip, picture, clip]}]
        rewritten, pictures = openai_server.expand_gemma4_images(messages, None)
        self.assertEqual(len(pictures), 5)
        order = []
        g.render(rewritten, image_order=order)
        self.assertEqual(order, [0, 1, 2, 3, 4])

    def test_a_video_in_an_earlier_turn_stays_in_the_prompt(self):
        clip = {"type": "video", "video": {"frames": [data_url(png(96, 96, "red"))]}}
        messages = [{"role": "user", "content": [{"type": "text", "text": "watch"}, clip]},
                    {"role": "assistant", "content": "I watched it."}, {"role": "user", "content": "and now?"}]
        rewritten, pictures = openai_server.expand_gemma4_images(messages, None)
        prompt = g.render(rewritten)
        self.assertEqual(prompt.count("<|image>"), 1)
        self.assertEqual(len(pictures), 1)

    def refused(self, part, *words):
        with self.assertRaises(openai_server.APIError) as caught:
            self.expand(part)
        self.assertEqual(caught.exception.status, 400)
        for word in words:
            self.assertIn(word, str(caught.exception))

    def test_bad_videos_are_a_400_that_says_what_is_wrong(self):
        good = data_url(png(96, 96, "red"))
        self.refused({"type": "video", "video": 7}, "must be an object")
        self.refused({"type": "video", "video": {"frames": []}}, "frames")
        self.refused({"type": "video", "video": {"frames": "nope"}}, "frames")
        self.refused({"type": "video", "video": {"frames": [good], "fps": 0}}, "fps")
        self.refused({"type": "video", "video": {"frames": [good], "fps": True}}, "fps")
        self.refused({"type": "video", "video": {"frames": [good], "fps": "fast"}}, "fps")
        self.refused({"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}, "not fetched")
        self.refused({"type": "video_url", "video_url": {}}, "url")
        self.refused({"type": "video", "video": {"frames": ["data:image/png;base64,@@@"]}}, "base64")
        self.refused({"type": "video", "video": {"frames": [data_url(b"not a picture")]}})

    def test_a_video_file_without_ffmpeg_says_so(self):
        with patch.dict(os.environ, {"COLI_FFMPEG": os.devnull + ".none"}):
            self.refused({"type": "video_url", "video_url": {"url": data_url(b"\x00\x00\x00\x20ftypmp42", "video/mp4")}}, "ffmpeg")

    def test_an_unexpanded_video_is_still_refused_by_the_renderer(self):
        for part in ({"type": "video"}, {"type": "video", "video": {"url": "u"}}, {"type": "video_url", "video_url": {"url": "u"}}):
            with self.assertRaises(g.ChatError) as caught:
                g.render([{"role": "user", "content": [{"type": "text", "text": "what is this"}, part]}])
            self.assertEqual(caught.exception.code, "unsupported_content_type")
            self.assertEqual(caught.exception.param, "messages.0.content.1")
            self.assertIn("not turned into frames", str(caught.exception))


@unittest.skipUnless(HAVE_IMAGING, "needs numpy and Pillow")
class ModalitiesTest(unittest.TestCase):
    def test_video_is_offered_with_pictures_and_not_without(self):
        class Engine:
            vision, audio = True, False

        engine = Engine()
        server = type("S", (), {"engine": engine, "input_modalities": openai_server.APIServer.input_modalities})()
        with patch("openai_server.ARCH", "gemma4"):
            self.assertEqual(server.input_modalities(), ["text", "image", "video"])
            engine.audio = True
            self.assertEqual(server.input_modalities(), ["text", "image", "video", "audio"])
            engine.vision = engine.audio = False
            self.assertEqual(server.input_modalities(), ["text"])


@unittest.skipUnless(HAVE_IMAGING, "needs numpy and Pillow")
class FrameParityTest(unittest.TestCase):
    """A video's frames go through the image path with a budget of 70. transformers' video processor resizes with torchvision
    where the image processor uses PIL, so the patches agree to within a level or two of 255 on a few values (the grid, and so the
    soft tokens, are identical), not byte for byte."""

    def test_the_grid_and_the_pixels_match_the_video_processor(self):
        try:
            import torch
            from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor
        except Exception:
            self.skipTest("transformers, torch and torchvision are not all installed")
        try:
            import torchvision  # noqa: F401
        except Exception:
            self.skipTest("torchvision is not installed")
        processor = Gemma4VideoProcessor(max_soft_tokens=70, num_frames=32, do_sample_frames=False)
        rng = np.random.default_rng(1)
        for width, height in ((320, 240), (640, 360), (100, 300), (96, 96)):
            base = rng.integers(0, 255, (height // 8 + 1, width // 8 + 1, 3), dtype=np.uint8)
            image = Image.fromarray(base).resize((width, height), Image.BICUBIC)
            buf = io.BytesIO()
            image.save(buf, "PNG")
            patches, grid_h, grid_w = gemma4_image.preprocess(buf.getvalue(), 70, 16, 3)
            frame = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None]
            out = processor.preprocess([frame], do_sample_frames=False, return_tensors="pt")
            self.assertEqual(int(out["num_soft_tokens_per_video"][0]), (grid_h // 3) * (grid_w // 3), (width, height))
            reference = (out["pixel_values_videos"][0][:grid_h * grid_w].numpy() * 255.0).round()
            self.assertLessEqual(float(np.abs(reference - patches.astype(np.float32)).max()), 2.0, (width, height))


@unittest.skipUnless(TONE.is_file(), "the MP3 fixture is missing")
@unittest.skipUnless(HAVE_IMAGING, "needs numpy")
class Mp3Test(unittest.TestCase):
    def test_an_mp3_is_decoded_to_the_tone_it_holds(self):
        if media._mp3_tool() is None and not FFMPEG:
            self.skipTest("no acestep_audio_tool built (make acestep_audio_tool) and no ffmpeg")
        with patch.dict(os.environ, {"COLI_FFMPEG": FFMPEG or ""}):
            samples, rate = media.decode_audio(TONE.read_bytes())
        self.assertEqual(rate, 16000 if media._mp3_tool() else 48000)
        seconds = len(samples) / rate
        self.assertTrue(0.95 < seconds < 1.15, seconds)
        spectrum = np.abs(np.fft.rfft(samples[: rate]))
        peak = int(np.argmax(spectrum)) * rate / len(samples[: rate])
        self.assertAlmostEqual(peak, 440, delta=8)

    def test_an_mp3_goes_through_the_audio_front_end_like_a_wav(self):
        if media._mp3_tool() is None and not FFMPEG:
            self.skipTest("no acestep_audio_tool built and no ffmpeg")
        with patch.dict(os.environ, {"COLI_FFMPEG": FFMPEG or ""}):
            frames, tokens = gemma4_audio.preprocess(TONE.read_bytes())
        self.assertEqual(frames.shape[1], 128)
        self.assertEqual(tokens, gemma4_audio.soft_tokens(frames.shape[0]))
        self.assertTrue(20 < tokens < 40, tokens)

    def test_the_gateway_reads_an_mp3_whatever_the_request_calls_it(self):
        if media._mp3_tool() is None and not FFMPEG:
            self.skipTest("no acestep_audio_tool built and no ffmpeg")
        raw = TONE.read_bytes()
        with patch.dict(os.environ, {"COLI_FFMPEG": FFMPEG or ""}):
            for label in ("mp3", "wav", "flac", None):
                part = {"type": "input_audio", "input_audio": {"data": base64.b64encode(raw).decode(), **({"format": label} if label else {})}}
                clip, tokens = openai_server._gemma4_audio_part(part, 750)
                self.assertEqual(clip[3], "audio")
                self.assertTrue(20 < tokens < 40, (label, tokens))

    def test_a_wav_is_read_as_before(self):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 7))) for i in range(16000)))
        frames, tokens = gemma4_audio.preprocess(buf.getvalue())
        self.assertTrue(20 < tokens < 40)


@unittest.skipUnless(FFMPEG and HAVE_IMAGING, "needs ffmpeg, numpy and Pillow")
class FfmpegTest(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"COLI_FFMPEG": FFMPEG})
        patcher.start()
        self.addCleanup(patcher.stop)

    def encode(self, args, suffix):
        import subprocess
        import tempfile
        path = tempfile.mktemp(suffix=suffix)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        done = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args, path], capture_output=True)
        if done.returncode:
            self.skipTest("this ffmpeg cannot make that file: " + done.stderr.decode(errors="replace")[-200:])
        return Path(path).read_bytes()

    def test_a_video_file_gives_its_evenly_spaced_frames_and_times(self):
        video = self.encode(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=5", "-c:v", "mpeg4"], ".mp4")
        frames, seconds = media.video_frames(video)
        keep = media.sample_indices(50, 32)
        self.assertEqual(len(frames), 32)
        self.assertEqual(seconds, [i / 10 for i in keep])
        for raw in frames[:3]:
            self.assertEqual(Image.open(io.BytesIO(raw)).size, (160, 120))

    def test_a_short_video_gives_every_frame(self):
        video = self.encode(["-f", "lavfi", "-i", "testsrc=size=96x96:rate=5:duration=2", "-c:v", "mpeg4"], ".mp4")
        frames, seconds = media.video_frames(video)
        self.assertEqual(len(frames), 10)
        self.assertEqual(seconds, [i / 5 for i in range(10)])

    def test_the_frames_are_the_right_ones(self):
        video = self.encode(["-f", "lavfi", "-i", "color=c=black:size=64x64:rate=4:duration=2,geq=r='255*N/8':g=0:b=0", "-c:v", "mpeg4",
                             "-qscale:v", "1"], ".mp4")
        frames, _ = media.video_frames(video)
        reds = [int(np.array(Image.open(io.BytesIO(f)).convert("RGB"))[..., 0].mean()) for f in frames]
        self.assertEqual(len(reds), 8)
        self.assertEqual(reds, sorted(reds))                       # frame N is redder than frame N-1: they come in order, once each
        self.assertGreater(reds[-1] - reds[0], 150)

    def test_a_video_goes_through_the_gateway_as_video_url(self):
        video = self.encode(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=8:duration=2", "-c:v", "mpeg4"], ".mp4")
        part = {"type": "video_url", "video_url": {"url": data_url(video, "video/mp4")}}
        messages, pictures = openai_server.expand_gemma4_images([{"role": "user", "content": [part]}], None)
        entry = messages[0]["content"][0]
        self.assertEqual(len(pictures), 16)
        self.assertEqual(entry["_times"][:3], ["00:00", "00:00", "00:00"])
        self.assertEqual(entry["_times"][-1], "00:01")

    def test_a_file_with_no_video_is_said_so(self):
        audio = self.encode(["-f", "lavfi", "-i", "sine=frequency=440:duration=1"], ".wav")
        with self.assertRaises(media.MediaError) as caught:
            media.video_frames(audio)
        self.assertIn("no video", str(caught.exception))

    def test_garbage_is_not_a_video(self):
        with self.assertRaises(media.MediaError):
            media.video_frames(b"this is not a video file at all " * 20)

    def test_audio_formats_ffmpeg_makes_are_read(self):
        wav = self.encode(["-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=16000"], ".wav")
        self.assertEqual(media.audio_kind(wav), "wav")
        for suffix, extra in ((".flac", []), (".ogg", ["-c:a", "libvorbis"]), (".m4a", ["-c:a", "aac"]), (".mp3", ["-c:a", "libmp3lame"])):
            raw = self.encode(["-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=16000", *extra], suffix)
            samples, rate = media.decode_audio(raw)
            self.assertTrue(0.9 < len(samples) / rate < 1.2, (suffix, len(samples) / rate))
            spectrum = np.abs(np.fft.rfft(samples[:rate]))
            self.assertAlmostEqual(int(np.argmax(spectrum)) * rate / len(samples[:rate]), 440, delta=10, msg=suffix)

    def test_garbage_audio_is_a_message_not_a_crash(self):
        with self.assertRaises(media.MediaError):
            media.decode_audio(b"fLaC" + os.urandom(64), "flac")


if __name__ == "__main__":
    unittest.main()
