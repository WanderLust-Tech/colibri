#!/usr/bin/env python3
"""Gemma 4's audio front end: a clip in, the log-mel frames its audio tower reads out.

A port of transformers' Gemma4AudioFeatureExtractor (feature_extraction_gemma4.py), which tests/test_gemma4_audio.py
compares with the real one:

  1. the clip is mixed down to one channel and resampled to 16 kHz (the extractor expects 16 kHz; this does the
     resampling it leaves to the caller), and cut to 30 seconds (the model's limit);
  2. it is padded with zeros to a multiple of 128 samples;
  3. frames of 20 ms every 10 ms, a Hann window, a 512-point FFT, 128 HTK mel filters over 0..8000 Hz, the natural log of the
     magnitude plus 0.001.

The frames whose end lies past the clip itself are padding and are dropped (the model masks them; the engine does not read
them). What is left is M frames of 128 float32 values, and the clip makes ceil(ceil(M / 2) / 2) soft tokens (two convolutions
of stride 2), at most 750 of them.

Only WAV files are read: PCM of 8, 16, 24 or 32 bits and 32- or 64-bit float, any number of channels, any rate.
Needs numpy.
"""
import math
import struct

import numpy as np

SAMPLE_RATE = 16000
FRAME = 320                    # 20 ms
HOP = 160                      # 10 ms
FFT = 512
BINS = 128
MEL_FLOOR = 1e-3
MAX_SAMPLES = 480000           # 30 seconds
PAD_MULTIPLE = 128


def read_wav(data):
    """(samples float32 in [-1, 1] mixed down to mono, sample rate) from the bytes of a WAV file."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file (audio is read as WAV only)")
    pos, fmt, samples = 12, None, None
    while pos + 8 <= len(data):
        tag, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if tag == b"fmt ":
            if len(body) < 16:
                raise ValueError("WAV: a fmt chunk that is too short")
            code, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if code == 0xFFFE and len(body) >= 26:                  # WAVE_FORMAT_EXTENSIBLE: the real code is in the sub-format
                code = struct.unpack("<H", body[24:26])[0]
            fmt = (code, channels, rate, bits)
        elif tag == b"data":
            samples = body
        pos += 8 + size + (size & 1)
    if fmt is None or samples is None:
        raise ValueError("WAV: no fmt or data chunk")
    code, channels, rate, bits = fmt
    if channels < 1 or rate < 1:
        raise ValueError("WAV: a header without channels or a rate")
    if code == 1 and bits == 8:
        x = (np.frombuffer(samples, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif code == 1 and bits == 16:
        x = np.frombuffer(samples[:len(samples) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    elif code == 1 and bits == 24:
        raw = np.frombuffer(samples[:len(samples) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        x = (np.where(value >= 1 << 23, value - (1 << 24), value)).astype(np.float32) / 8388608.0
    elif code == 1 and bits == 32:
        x = np.frombuffer(samples[:len(samples) // 4 * 4], dtype="<i4").astype(np.float32) / 2147483648.0
    elif code == 3 and bits == 32:
        x = np.frombuffer(samples[:len(samples) // 4 * 4], dtype="<f4").astype(np.float32)
    elif code == 3 and bits == 64:
        x = np.frombuffer(samples[:len(samples) // 8 * 8], dtype="<f8").astype(np.float32)
    else:
        raise ValueError(f"WAV: format {code} with {bits} bits is not read (PCM of 8, 16, 24 or 32 bits, or float)")
    x = x[:len(x) // channels * channels].reshape(-1, channels).mean(axis=1) if channels > 1 else x
    return np.ascontiguousarray(x, dtype=np.float32), rate


def resample(x, rate, target=SAMPLE_RATE):
    """x at `rate` Hz as float32 at `target` Hz: windowed-sinc interpolation, low-passed below the new Nyquist frequency when
    the rate goes down."""
    if rate == target or len(x) == 0:
        return x
    cutoff = min(1.0, target / rate)                        # in units of the input's Nyquist frequency
    half = 16                                               # taps on each side, in units of the cutoff's period
    reach = int(math.ceil(half / cutoff))
    n_out = int(len(x) * target / rate)
    out = np.empty(n_out, dtype=np.float32)
    padded = np.concatenate([np.zeros(reach, np.float32), x, np.zeros(reach + 1, np.float32)])
    taps = np.arange(-reach, reach + 1)
    for start in range(0, n_out, 8192):
        stop = min(n_out, start + 8192)
        position = np.arange(start, stop) * (rate / target)           # where each output sample falls, in input samples
        base = np.floor(position).astype(np.int64)
        index = base[:, None] + taps[None, :]
        distance = (position[:, None] - index) * cutoff
        weight = np.sinc(distance) * np.kaiser(2 * reach + 1, 8.6)[None, :] * cutoff
        out[start:stop] = (padded[index + reach] * weight).sum(axis=1)
    return out


def _window():
    return np.hanning(FRAME + 1)[:-1].astype(np.float32)        # a periodic Hann window, as transformers' window_function


def _hertz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hertz(m):
    return 700.0 * (np.power(10, m / 2595.0) - 1.0)


def mel_filters():
    """[FFT // 2 + 1, BINS]: triangular HTK mel filters over 0..8000 Hz, unnormalised (transformers' mel_filter_bank)."""
    bins = FFT // 2 + 1
    mel = np.linspace(_hertz_to_mel(0.0), _hertz_to_mel(8000.0), BINS + 2)
    centres = _mel_to_hertz(mel)
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, bins)
    diff = np.diff(centres)
    slopes = centres[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down, up))


def log_mel(samples):
    """(frames [M, BINS] float32, M): the log-mel frames of a 16 kHz clip, padding dropped, as the extractor makes them."""
    samples = np.asarray(samples, dtype=np.float32)[:MAX_SAMPLES]
    real = len(samples)
    padded_len = -(-real // PAD_MULTIPLE) * PAD_MULTIPLE
    wave = np.zeros(padded_len, dtype=np.float32)
    wave[:real] = samples
    mask = np.zeros(padded_len, dtype=np.int64)
    mask[:real] = 1
    pad_left = FRAME // 2
    wave = np.pad(wave, (pad_left, 0))
    mask = np.pad(mask, (pad_left, 0))
    size = FRAME + 1
    count = (len(wave) - size) // HOP + 1
    if count <= 0:
        return np.zeros((0, BINS), dtype=np.float32), 0
    frames = np.lib.stride_tricks.as_strided(wave, shape=(count, size), strides=(wave.strides[0] * HOP, wave.strides[0]))[:, :-1]
    spectrum = np.abs(np.fft.rfft(frames * _window(), n=FFT, axis=-1))
    mel = np.matmul(spectrum, mel_filters())
    features = np.log(mel + MEL_FLOOR).astype(np.float32)
    ends = np.arange(count) * HOP + size - 1
    valid = mask[ends].astype(bool)
    return np.ascontiguousarray(features[valid]), int(valid.sum())


def soft_tokens(frames):
    """The soft tokens M mel frames make: two convolutions of stride 2."""
    half = (frames - 1) // 2 + 1
    return (half - 1) // 2 + 1 if frames > 0 else 0


def preprocess(data, max_tokens=750):
    """(frames, tokens) for the bytes of a WAV file: frames is float32 [M, 128], tokens the soft tokens it makes. A clip
    longer than the model takes (30 s, or what max_tokens allows at 40 ms a token) is cut."""
    samples, rate = read_wav(data)
    samples = resample(samples, rate)
    if max_tokens:
        samples = samples[:max(1, max_tokens) * SAMPLE_RATE // 25]
    frames, count = log_mel(samples)
    if count == 0:
        raise ValueError("the clip is too short: it makes no audio frame")
    return frames, soft_tokens(count)
