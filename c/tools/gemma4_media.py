#!/usr/bin/env python3
"""What Gemma 4's gateway reads besides WAV files and pictures: other audio files, and video.

Audio. WAV is read by gemma4_audio.read_wav. MP3 is decoded by minimp3 (c/third_party/minimp3.h, through the small
`acestep_audio_tool` that ACE-Step's audio input already uses; `make acestep_audio_tool`). Anything else (FLAC, Ogg, M4A, AAC,
Opus, AIFF, the sound of a WebM) is decoded by ffmpeg when there is one (`COLI_FFMPEG`, or `ffmpeg` on the PATH), and refused by
name when there is not. Either way the result is mono float32 samples and a sample rate, like read_wav.

Video. The model reads a video as pictures: transformers' Gemma4VideoProcessor takes up to 32 frames, evenly spaced
(`indices = arange(n) * total // n`), runs each through the image processor with a budget of 70 soft tokens, and the processor
writes the prompt as one "mm:ss <boi>soft tokens<eoi>" group per frame. This module does the two things before that: it finds
the frames of a video file (ffmpeg; nothing here decodes video itself) and names their times. The sound of a video is not read.

Needs numpy for audio; video needs only ffmpeg and the image path (Pillow, numpy).
"""
import os
import re
import shutil
import struct
import subprocess
import tempfile

NUM_FRAMES = 32                 # the video processor's num_frames
VIDEO_SOFT_TOKENS = 70          # its max_soft_tokens a frame
MAX_VIDEO_BYTES = 512 * 1024 * 1024
TIMEOUT = 120                   # seconds an ffmpeg or the MP3 tool may take

C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MediaError(ValueError):
    """A clip or a video that cannot be read, with a message that says what to do about it."""


# ---------------------------------------------------------------------------------------------------------- ffmpeg

def ffmpeg_path():
    """The ffmpeg to use: COLI_FFMPEG (a path), else `ffmpeg` on the PATH, else None."""
    named = os.environ.get("COLI_FFMPEG")
    if named:
        return named if os.path.isfile(named) else None
    return shutil.which("ffmpeg")


def _need_ffmpeg(what):
    exe = ffmpeg_path()
    if not exe:
        raise MediaError(f"{what} needs ffmpeg, and there is none on this server (put ffmpeg on the PATH, or name it with "
                         "COLI_FFMPEG)")
    return exe


def _run(cmd, data=None, timeout=TIMEOUT):
    try:
        return subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise MediaError(f"{os.path.basename(cmd[0])} took more than {timeout} s and was stopped")
    except OSError as problem:
        raise MediaError(f"cannot run {cmd[0]}: {problem}")


def _spool(data, suffix=""):
    """The bytes in a temporary file (ffmpeg needs to seek in an MP4); the caller removes it."""
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(data)
    finally:
        handle.close()
    return handle.name


# ----------------------------------------------------------------------------------------------------------- audio

def audio_kind(data):
    """What a file of audio is, from its first bytes (not from what the request calls it): wav, mp3, flac, ogg, m4a, aiff,
    webm (Matroska) or unknown."""
    head = bytes(data[:16])
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return "mp3"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] == b"ftyp":
        return "m4a"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    return "unknown"


def _mp3_tool():
    exe = "acestep_audio_tool.exe" if os.name == "nt" else "acestep_audio_tool"
    for folder in (C_DIR, os.path.join(C_DIR, "build")):
        path = os.path.join(folder, exe)
        if os.path.isfile(path):
            return path
    return None


def _mixdown(pcm, channels):
    import numpy as np
    x = np.frombuffer(pcm, dtype="<f4")
    if channels > 1:
        x = x[:len(x) - len(x) % channels].reshape(-1, channels).mean(axis=1)
    return x.astype(np.float32)


def _decode_mp3(data):
    tool = _mp3_tool()
    if tool is None:
        return None
    src = _spool(data, ".mp3")
    out = src + ".f32"
    try:
        done = _run([tool, "mem", src, out])
        line = done.stdout.decode("utf-8", "replace").strip()
        match = re.match(r"OK rate=(\d+) channels=(\d+) frames=(\d+)", line)
        if not match:
            raise MediaError("this MP3 cannot be read" + (f" ({line[4:]})" if line.startswith("ERR ") else ""))
        with open(out, "rb") as f:
            pcm = f.read()
        return _mixdown(pcm, int(match.group(2))), int(match.group(1))
    finally:
        for path in (src, out):
            try:
                os.remove(path)
            except OSError:
                pass


def _decode_with_ffmpeg(data, kind):
    exe = _need_ffmpeg(f"{kind.upper() if kind != 'unknown' else 'this'} audio")
    rate = 48000                              # ffmpeg keeps the rate; gemma4_audio resamples to 16 kHz
    src = _spool(data)                        # a file, not a pipe: an M4A or MP4 has its index at the end and ffmpeg must seek
    try:
        done = _run([exe, "-hide_banner", "-loglevel", "error", "-i", src, "-vn", "-f", "f32le", "-ac", "1",
                     "-ar", str(rate), "pipe:1"])
    finally:
        try:
            os.remove(src)
        except OSError:
            pass
    if done.returncode != 0 or not done.stdout:
        why = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise MediaError("ffmpeg cannot read this audio" + (f": {why[-1]}" if why else ""))
    import numpy as np
    return np.frombuffer(done.stdout, dtype="<f4").astype(np.float32), rate


def decode_audio(data, kind=None):
    """(mono float32 samples, sample rate) of the bytes of an audio file that is not a WAV: an MP3 by minimp3, anything else by
    ffmpeg. Raises MediaError, which says what is missing."""
    kind = kind or audio_kind(data)
    if kind == "mp3":
        decoded = _decode_mp3(data)
        if decoded is not None:
            return decoded
        if not ffmpeg_path():
            raise MediaError("MP3 needs the MP3 reader (`make acestep_audio_tool` in c/, or ffmpeg on the PATH); send a WAV "
                             "instead")
    if kind == "unknown":
        raise MediaError("this is not audio the server can read (WAV and MP3, and with ffmpeg FLAC, Ogg, M4A and more)")
    return _decode_with_ffmpeg(data, kind)


# ----------------------------------------------------------------------------------------------------------- video

def sample_indices(total, n=NUM_FRAMES):
    """transformers' default_sample_indices_fn: arange(n) * total // n. It raises when n is more than total; this takes
    every frame of a short video instead."""
    if total < 1:
        raise MediaError("the video has no frames")
    n = min(n, total)
    return [i * total // n for i in range(n)]


def timestamp_text(seconds):
    """Gemma4Processor's mm:ss, the minutes not reduced (a long video reads 75:03)."""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _parse_probe(stderr):
    text = stderr.decode("utf-8", "replace")
    match = re.search(r"Video:.*?,\s*(\d+)x(\d+)", text)
    fps = re.search(r"(\d+(?:\.\d+)?)\s*fps", text) or re.search(r"(\d+(?:\.\d+)?)\s*tbr", text)
    count = re.findall(r"frame=\s*(\d+)", text)
    return {"width": int(match.group(1)) if match else None, "height": int(match.group(2)) if match else None,
            "fps": float(fps.group(1)) if fps else None, "frames": int(count[-1]) if count else None,
            "has_video": "Video:" in text}


def _split_png(blob):
    """The PNG files of a byte stream of concatenated PNGs."""
    out, pos = [], 0
    sig = b"\x89PNG\r\n\x1a\n"
    while True:
        start = blob.find(sig, pos)
        if start < 0:
            return out
        p = start + 8
        while p + 8 <= len(blob):
            size = struct.unpack(">I", blob[p:p + 4])[0]
            tag = blob[p + 4:p + 8]
            p += 12 + size
            if tag == b"IEND":
                break
        out.append(blob[start:p])
        pos = p


def video_frames(data, num_frames=NUM_FRAMES):
    """(frames, seconds): up to num_frames PNG files, evenly spaced through the video in `data` (the bytes of a video file),
    and the time of each in seconds (frame number / the video's frame rate). Needs ffmpeg."""
    if len(data) > MAX_VIDEO_BYTES:
        raise MediaError(f"the video is {len(data) >> 20} MB; the limit is {MAX_VIDEO_BYTES >> 20} MB")
    exe = _need_ffmpeg("a video file")
    path = _spool(data)
    try:
        # how many frames, and how fast: a pass over the packets, no decoding
        info = _parse_probe(_run([exe, "-hide_banner", "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"]).stderr)
        if not info["has_video"]:
            raise MediaError("ffmpeg finds no video in this file")
        if not info["frames"]:
            raise MediaError("ffmpeg cannot count the frames of this video")
        fps = info["fps"] or 24.0
        indices = sample_indices(info["frames"], num_frames)
        select = "select='" + "+".join(f"eq(n\\,{i})" for i in indices) + "'"
        out = None
        for sync in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
            done = _run([exe, "-hide_banner", "-loglevel", "error", "-i", path, "-map", "0:v:0", "-vf", select, *sync,
                         "-f", "image2pipe", "-c:v", "png", "pipe:1"])
            if done.returncode == 0:
                out = done
                break
        if out is None:
            why = done.stderr.decode("utf-8", "replace").strip().splitlines()
            raise MediaError("ffmpeg cannot take the frames out of this video" + (f": {why[-1]}" if why else ""))
        frames = _split_png(out.stdout)
        if len(frames) != len(indices):
            indices = indices[:len(frames)]
        if not frames:
            raise MediaError("ffmpeg found no frames in this video")
        return frames, [i / fps for i in indices]
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
