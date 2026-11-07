"""Gemma 4's audio front end (tools/gemma4_audio.py) against transformers' own feature extractor.

    python -m unittest tests.test_gemma4_audio

Needs numpy; the comparison with Gemma4AudioFeatureExtractor needs transformers >= 5.5 and is skipped without it. The
log-mel frames must agree exactly (same filters, same window, same arithmetic order): they are what the audio tower reads.
"""
import io
import math
import struct
import sys
import unittest
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))

try:
    import numpy as np
    import gemma4_audio as ga
    HAVE_NUMPY = True
except ImportError:                                   # pragma: no cover
    HAVE_NUMPY = False

try:
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
    HAVE_TRANSFORMERS = HAVE_NUMPY
except Exception:                                     # pragma: no cover
    HAVE_TRANSFORMERS = False


def tone(seconds, rate=16000, hertz=440.0, noise=0.0, seed=0):
    n = int(seconds * rate)
    x = 0.3 * np.sin(2 * np.pi * hertz * np.arange(n) / rate)
    if noise:
        x = x + noise * np.random.default_rng(seed).standard_normal(n)
    return x.astype(np.float32)


def wav_bytes(samples, rate=16000, channels=1, bits=16, float_format=False, extensible=False):
    """A WAV file of float samples in [-1, 1] (interleaved across channels if more than one)."""
    if float_format:
        data = np.asarray(samples, dtype="<f4").tobytes() if bits == 32 else np.asarray(samples, dtype="<f8").tobytes()
        code = 3
    elif bits == 8:
        data = (np.clip(np.asarray(samples) * 128 + 128, 0, 255)).astype(np.uint8).tobytes()
        code = 1
    elif bits == 16:
        data = np.clip(np.asarray(samples) * 32768, -32768, 32767).astype("<i2").tobytes()
        code = 1
    elif bits == 24:
        v = np.clip(np.asarray(samples) * 8388608, -8388608, 8388607).astype(np.int32)
        data = b"".join(struct.pack("<i", int(i))[:3] for i in v)
        code = 1
    else:
        data = np.clip(np.asarray(samples) * 2147483648, -2147483648, 2147483647).astype("<i4").tobytes()
        code = 1
    fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else code, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
    if extensible:
        fmt += struct.pack("<HHIH", 22, bits, 0, code) + b"\x00" * 14
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


@unittest.skipUnless(HAVE_NUMPY, "needs numpy")
class FrontEndTest(unittest.TestCase):
    def test_a_clip_makes_the_frames_and_tokens_the_arithmetic_says(self):
        frames, count = ga.log_mel(tone(1.0))
        self.assertEqual(frames.shape, (count, 128))
        self.assertEqual(frames.dtype, np.float32)
        self.assertEqual(ga.soft_tokens(count), 25)                 # 99 frames -> 50 -> 25
        self.assertEqual([ga.soft_tokens(m) for m in (1, 2, 3, 4, 5, 8, 9, 99, 1099, 3000)], [1, 1, 1, 1, 2, 2, 3, 25, 275, 750])

    def test_a_tone_shows_in_the_right_mel_bin(self):
        frames, _ = ga.log_mel(tone(1.0, hertz=1000.0))
        quiet, _ = ga.log_mel(tone(1.0, hertz=100.0))
        loud_at = int(np.argmax(frames[40]))
        self.assertNotEqual(loud_at, int(np.argmax(quiet[40])))
        self.assertGreater(loud_at, int(np.argmax(quiet[40])))       # a higher tone, a higher bin

    def test_nothing_longer_than_thirty_seconds_is_kept(self):
        frames, count = ga.log_mel(np.zeros(16000 * 40, dtype=np.float32))
        self.assertEqual(ga.soft_tokens(count), 750)
        frames, tokens = ga.preprocess(wav_bytes(tone(40.0, noise=0.01)), max_tokens=750)
        self.assertEqual(tokens, 750)

    def test_the_engines_token_limit_cuts_the_clip(self):
        _, tokens = ga.preprocess(wav_bytes(tone(10.0, noise=0.01)), max_tokens=100)
        self.assertLessEqual(tokens, 100)
        self.assertGreater(tokens, 90)

    def test_a_clip_too_short_to_make_a_frame_is_an_error(self):
        with self.assertRaises(ValueError):
            ga.preprocess(wav_bytes(np.zeros(20, dtype=np.float32)))

    def test_wav_formats(self):
        x = tone(0.5, noise=0.02)
        for kwargs in ({"bits": 8}, {"bits": 16}, {"bits": 24}, {"bits": 32}, {"bits": 32, "float_format": True},
                       {"bits": 64, "float_format": True}, {"bits": 16, "extensible": True}):
            with self.subTest(**kwargs):
                got, rate = ga.read_wav(wav_bytes(x, **kwargs))
                self.assertEqual(rate, 16000)
                self.assertEqual(len(got), len(x))
                self.assertLess(float(np.abs(got - x).max()), 0.02 if kwargs.get("bits") == 8 else 1e-3)

    def test_stereo_is_mixed_down_by_the_mean(self):
        left, right = tone(0.2, hertz=300.0), tone(0.2, hertz=900.0)
        interleaved = np.stack([left, right], axis=1).reshape(-1)
        got, _ = ga.read_wav(wav_bytes(interleaved, channels=2))
        self.assertLess(float(np.abs(got - (left + right) / 2).max()), 1e-3)

    def test_not_a_wav_file_is_a_value_error(self):
        for bad in (b"", b"RIFF", b"hello world, this is not audio at all", wav_bytes(tone(0.1))[:30]):
            with self.assertRaises(ValueError):
                ga.read_wav(bad)

    def test_a_file_from_the_standard_library_reads_back(self):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((tone(0.3, noise=0.01) * 32767).astype("<i2").tobytes())
        got, rate = ga.read_wav(buffer.getvalue())
        self.assertEqual((rate, len(got)), (16000, 4800))

    def test_resampling_keeps_a_tone_where_it_was(self):
        for rate in (8000, 22050, 44100, 48000):
            with self.subTest(rate=rate):
                out = ga.resample(tone(1.0, rate=rate, hertz=1000.0), rate)
                self.assertAlmostEqual(len(out) / 16000, 1.0, places=3)
                spectrum = np.abs(np.fft.rfft(out[2000:14000]))
                peak = float(np.argmax(spectrum)) * 16000 / 12000
                self.assertAlmostEqual(peak, 1000.0, delta=5.0)
                self.assertGreater(float(np.abs(out[2000:14000]).max()), 0.27)       # not attenuated

    def test_resampling_removes_what_the_lower_rate_cannot_hold(self):
        out = ga.resample(tone(1.0, rate=48000, hertz=20000.0), 48000)          # above 8 kHz: gone, not folded back down
        self.assertLess(float(np.abs(out[2000:14000]).max()), 0.02)

    def test_a_clip_at_another_rate_has_the_duration_it_had(self):
        _, tokens16 = ga.preprocess(wav_bytes(tone(2.0, rate=16000, noise=0.01)))
        _, tokens44 = ga.preprocess(wav_bytes(tone(2.0, rate=44100, noise=0.01), rate=44100))
        self.assertLessEqual(abs(tokens16 - tokens44), 1)


@unittest.skipUnless(HAVE_TRANSFORMERS, "needs numpy and transformers with Gemma 4")
class AgainstTransformersTest(unittest.TestCase):
    def test_the_filter_bank_and_the_window(self):
        extractor = Gemma4AudioFeatureExtractor()
        self.assertEqual(float(np.abs(extractor.mel_filters - ga.mel_filters()).max()), 0.0)
        self.assertEqual(float(np.abs(extractor.window - ga._window()).max()), 0.0)

    def test_the_frames_and_the_soft_tokens(self):
        extractor = Gemma4AudioFeatureExtractor()
        for seconds, noise in ((0.05, 0.1), (0.37, 0.1), (1.0, 0.05), (2.5, 0.1), (5.31, 0.1), (12.0, 0.2), (30.0, 0.1)):
            with self.subTest(seconds=seconds):
                x = tone(seconds, noise=noise, seed=int(seconds * 100))
                out = extractor([x], return_tensors="np")
                features, mask = out["input_features"][0], out["input_features_mask"][0]
                ours, count = ga.log_mel(x)
                self.assertEqual(count, int(mask.sum()))
                self.assertEqual(float(np.abs(features[mask] - ours).max()), 0.0)
                # the processor's own count of soft tokens: two convolutions of stride 2 laid over the mask
                length, shrunk = len(mask), mask
                for _ in range(2):
                    out_len = (length + 2 - 3) // 2 + 1
                    shrunk = shrunk[::2][:out_len]
                    length = len(shrunk)
                self.assertEqual(ga.soft_tokens(count), int(shrunk.sum()))


if __name__ == "__main__":
    unittest.main()
