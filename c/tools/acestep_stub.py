#!/usr/bin/env python3
"""A stand-in for the acestep engine: same command line, same serve protocol.

The front ends (the gateway's POST /v1/audio/music, `coli music`) are tested without the
real engine, which needs 6 GB of weights. This script speaks the protocol byte for byte and
makes a synthetic clip that depends on the caption, the lyrics, the seed and the duration, so
a rendered song is recognisable and two requests that differ in one of them audibly differ.
For audio in it reads the clips it is given (a WAV's header, an MP3's frames: for their length),
applies the engine's own rules to them, and makes a clip that also depends on their bytes and on
the task's settings. It is not a model.

It is deliberately self-contained (stdlib only): it plays the part of a separate binary, and
sharing code with the client it is used to test would let a bug on one side hide the same bug
on the other.

    acestep_stub.py --model DIR --serve

Serve protocol (one request at a time, logs on stderr), the header of c/acestep_serve.h:
    engine -> READY {...}                  with "tasks" and "audio_in"
    client -> GEN {"id","caption","lyrics","language","duration","steps","shift","seed","bpm",
                   "key","timesig","dcw",
                   "task","src_path","ref_path","repaint_start","repaint_end","repaint_mode",
                   "repaint_strength","cover_strength","cover_noise"}
    engine -> PROGRESS {"id","stage","step","steps"}    stages encode (only with a clip), text,
                                                         condition, denoise, decode
    engine -> AUDIO {"id","sample_rate","channels","samples","bytes","src_seconds",...} + N bytes + "\\n"
    engine -> ERROR {"id","message"}
    client -> CANCEL {"id"}        answered with ERROR "cancelled"
    stdin EOF: clean exit

What it models of audio in: the rules of ace_check (acestep_gen.h) in its words; a clip's length from
its header; the output length (the source's samples with a repaint's silence added, in whole 1920-sample
frames, never fewer than 128 frames cut back to the source's length); a clip that is not a WAV or an MP3,
is shorter than a second or longer than 600 s is refused as "the source clip: ..." / "the reference clip:
..."; a silent reference (WAV) is refused. What it does not model: decoding (an MP3's samples are never
read), resampling, the VAE, the tokenizer: it never looks at a clip's audio beyond that.

Knobs (environment), all optional:
    ACESTEP_STUB_DELAY       seconds per denoising step (default 0.05)
    ACESTEP_STUB_LOAD_DELAY  seconds before READY (default 0)
    ACESTEP_STUB_RATE        sample rate of the clips (default 48000; the tests use a low one to stay small)
    ACESTEP_STUB_NO_AUDIO_IN set to make it an older engine: READY without tasks, GEN fields ignored
Captions containing __stub_error__ get an ERROR, __stub_crash__ kills the process in the middle of
the denoising, __stub_garbage__ writes a stray stdout line first: the three failure shapes a front
end has to survive.
"""
import argparse
import hashlib
import json
import math
import os
import queue
import struct
import sys
import threading
import time

READY = {"model": "ace-step-1.5-turbo", "sample_rate": 48000, "channels": 2, "format": "s16le",
         "default_duration": 30, "min_duration": 1, "max_duration": 600, "default_steps": 8, "max_steps": 20,
         "default_shift": 1, "max_caption_bytes": 4096, "max_lyric_bytes": 32768, "max_seed": 4294967295,
         "protocol": 1}
AUDIO_IN = {"tasks": ["text2music", "cover", "repaint"],
            "audio_in": {"formats": ["wav", "mp3"], "max_seconds": 600, "ref_seconds": 30}}
MAX_LINE = 1 << 20
HOP = 1920
MIN_FRAMES = 128
MAX_SECONDS = 600
RATE48 = 48000


def log(message):
    sys.stderr.write(f"[acestep-stub] {message}\n")
    sys.stderr.flush()


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check(request, audio_in=True):
    """The engine's own refusals, in its own words (acestep_gen.h ace_check)."""
    caption = request.get("caption")
    if not isinstance(caption, str):
        return "caption is required (it may be empty)"
    if len(caption.encode()) > READY["max_caption_bytes"]:
        return f"caption longer than {READY['max_caption_bytes']} bytes"
    lyrics = request.get("lyrics")
    if lyrics is not None and (not isinstance(lyrics, str) or len(lyrics.encode()) > READY["max_lyric_bytes"]):
        return "lyrics must be a string of at most %d bytes" % READY["max_lyric_bytes"]
    task = request.get("task", "text2music") if audio_in else "text2music"
    if task not in AUDIO_IN["tasks"]:
        return "task must be text2music, cover or repaint"
    duration = request.get("duration", READY["default_duration"])
    if task == "text2music" and (not number(duration) or not 1 <= duration <= 600):
        return "duration must be 1 to 600 seconds"
    steps = request.get("steps", 8)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 20:
        return "steps must be 1 to 20"
    shift = request.get("shift", 1)
    if not number(shift) or not 0 < shift <= 10:
        return "shift must be above 0 and at most 10"
    seed = request.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        return "seed must be a whole number from 0 to 4294967295"
    bpm = request.get("bpm", 0)
    if isinstance(bpm, bool) or not isinstance(bpm, int) or not 0 <= bpm <= 1000:
        return "bpm must be a whole number"
    key, signature = request.get("key"), request.get("timesig")
    if (isinstance(key, str) and len(key) > 32) or (isinstance(signature, str) and len(signature) > 8):
        return "key or time signature too long"
    if not audio_in:
        return None
    src, ref = request.get("src_path"), request.get("ref_path")
    if task != "text2music" and not (isinstance(src, str) and src):
        return "a %s needs a source clip (src)" % ("cover" if task == "cover" else "repaint")
    if task == "text2music" and isinstance(src, str) and src:
        return ("a source clip is for a cover or a repaint; for text to music, give a reference clip (ref) "
                "to take its style")
    if (src is not None and not isinstance(src, str)) or (isinstance(src, str) and len(src) > 4000):
        return "src path too long"
    if (ref is not None and not isinstance(ref, str)) or (isinstance(ref, str) and len(ref) > 4000):
        return "ref path too long"
    cover_strength, cover_noise = request.get("cover_strength", 0), request.get("cover_noise", 0)
    if task != "cover" and (cover_strength != 0 or cover_noise != 0):
        return "cover_strength and cover_noise are for a cover"
    if not (number(cover_strength) and 0 <= cover_strength <= 1) or not (number(cover_noise) and 0 <= cover_noise <= 1):
        return "cover_strength and cover_noise must be 0 to 1"
    mode = request.get("repaint_mode")
    mode_code = {None: 0, "balanced": 0, "conservative": 1, "aggressive": 2}.get(mode, -1) if not isinstance(mode, list) else -1
    start, end, strength = request.get("repaint_start", 0), request.get("repaint_end", 0), request.get("repaint_strength", 0)
    if task != "repaint" and (start != 0 or end != 0 or mode_code != 0 or strength != 0):
        return "the repaint settings are for a repaint"
    if mode_code < 0:
        return "repaint_mode must be balanced, conservative or aggressive"
    if not (number(strength) and 0 <= strength <= 1):
        return "repaint_strength must be 0 to 1"
    if not number(start) or not number(end) or abs(start) > 3600 or abs(end) > 3600:
        return "the repaint span is out of range"
    return None


# ----------------------------------------------------------------- clips
_MPEG1 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
_MPEG2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


def mpeg_frame(data, at):
    """(frame length, samples, rate) of the layer III frame header at `at`, or None."""
    if at + 4 > len(data) or data[at] != 0xFF or data[at + 1] & 0xE0 != 0xE0:
        return None
    version, layer = (data[at + 1] >> 3) & 3, (data[at + 1] >> 1) & 3
    bitrate_index, rate_index, padding = data[at + 2] >> 4, (data[at + 2] >> 2) & 3, (data[at + 2] >> 1) & 1
    if version == 1 or layer != 1 or bitrate_index in (0, 15) or rate_index == 3:
        return None
    rate = _RATES[version][rate_index]
    if version == 3:
        return 144 * _MPEG1[bitrate_index] * 1000 // rate + padding, 1152, rate
    return 72 * _MPEG2[bitrate_index] * 1000 // rate + padding, 576, rate


def mp3_seconds(data):
    """The length of an MP3 by its frames: an ID3v2 tag skipped, the frames counted until the sync is lost."""
    at = 0
    if data[:3] == b"ID3" and len(data) >= 10:
        at = 10 + ((data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9])
    limit = min(len(data), at + 4096)
    while at < limit and not mpeg_frame(data, at):
        at += 1
    seconds, found = 0.0, 0
    while True:
        frame = mpeg_frame(data, at)
        if not frame:
            break
        length, samples, rate = frame
        seconds += samples / rate
        found += 1
        at += max(length, 4)
    return seconds if found else None


def wav_info(data):
    """(seconds, silent) of a WAV, or None when it cannot be read. A data chunk longer than the file counts what is there."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    position, rate, frame, bits = 12, None, None, None
    while position + 8 <= len(data):
        kind, size = data[position:position + 4], struct.unpack("<I", data[position + 4:position + 8])[0]
        if kind == b"fmt " and position + 24 <= len(data):
            _tag, channels, rate, _bps, _align, bits = struct.unpack("<HHIIHH", data[position + 8:position + 24])
            frame = channels * ((bits + 7) // 8)
        elif kind == b"data":
            if not rate or not frame:
                return None
            body = data[position + 8:position + 8 + size]
            silent = bits != 8 and not any(body)
            return len(body) / frame / rate, silent
        position += 8 + size + (size & 1)
    return None


def read_clip(path, what):
    """(seconds, silent, digest) of the clip at `path`, or (None, None, message) in the engine's words."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as error:
        return None, None, f"the {what} clip: cannot open {path}: {error.strerror or error}"
    digest = hashlib.sha256(data).hexdigest()
    seconds, silent = None, False
    if data[:4] == b"RIFF":
        info = wav_info(data)
        if info is None:
            return None, None, f"the {what} clip: not a WAV this engine reads"
        seconds, silent = info
    else:
        seconds = mp3_seconds(data)
        if seconds is None:
            return None, None, f"the {what} clip: not a WAV or an MP3"
    if seconds > MAX_SECONDS:
        return None, None, f"the {what} clip: longer than {MAX_SECONDS} seconds"
    if seconds < 1:
        return None, None, f"the {what} clip is shorter than one second"
    return seconds, silent, digest


def plan(request, src_seconds):
    """(samples at 48 kHz asked for, latent frames run) as acg does it: text to music from the duration, audio from the
    source with a repaint's silence added; whole 1920-sample frames, never fewer than 128. Returns (want48, frames) or
    an error message."""
    task = request.get("task", "text2music")
    if task == "text2music":
        duration = round(float(request.get("duration", 30)), 1)
        want = int(duration * RATE48)
    else:
        n = int(src_seconds * RATE48)
        lead = tail = 0
        if task == "repaint":
            sd = n / RATE48
            start, end = float(request.get("repaint_start", 0)), float(request.get("repaint_end", 0))
            end = end if end > 0 else sd
            if not end > start:
                return "the repaint must end after it starts (%g to %g s)" % (start, end)
            if start >= sd:
                return ("the repaint starts at %g s, past the end of the %.2f s clip: to add music after it, end the "
                        "repaint later and start it inside the clip" % (start, sd))
            lead = int((-start if start < 0 else 0) * RATE48)
            tail = int((end - sd if end - sd > 0 else 0) * RATE48)
            if (n + lead + tail) / RATE48 > MAX_SECONDS:
                return "the repainted clip would be longer than %g seconds" % MAX_SECONDS
        want = n + lead + tail
    return want, max(MIN_FRAMES, want // HOP)


def clip(request, rate, want48, frames, digests):
    """A recognisable clip: a few sine voices whose pitches come from a hash of the request (and the clips' bytes)."""
    total48 = frames * HOP
    samples = min(total48, want48) * rate // RATE48
    digest = hashlib.sha256(json.dumps([request.get("caption"), request.get("lyrics"), request.get("language"),
                                        request.get("seed", 0), request.get("bpm"), request.get("key"),
                                        request.get("task", "text2music"), digests,
                                        request.get("repaint_start"), request.get("repaint_end"),
                                        request.get("repaint_mode"), request.get("repaint_strength"),
                                        request.get("cover_strength"), request.get("cover_noise")]).encode()).digest()
    voices = [(220.0 * 2 ** (digest[i] / 96.0), 0.12 + digest[i + 4] / 2550.0) for i in range(4)]
    pan = digest[8] / 255.0
    out = bytearray()
    for n in range(samples):
        t = n / rate
        v = sum(a * math.sin(2 * math.pi * f * t) for f, a in voices)
        left, right = v * (1 - 0.5 * pan), v * (0.5 + 0.5 * pan)
        out += struct.pack("<hh", int(max(-1, min(1, left)) * 20000), int(max(-1, min(1, right)) * 20000))
    return bytes(out), samples


class Out:
    """stdout, one frame at a time."""

    def __init__(self):
        self.lock = threading.Lock()

    def line(self, keyword, obj, payload=None):
        data = (keyword + " " + json.dumps(obj, separators=(",", ":")) + "\n").encode()
        with self.lock:
            sys.stdout.buffer.write(data)
            if payload is not None:
                sys.stdout.buffer.write(payload + b"\n")
            sys.stdout.buffer.flush()


def serve():
    delay = env_float("ACESTEP_STUB_DELAY", 0.05)
    time.sleep(env_float("ACESTEP_STUB_LOAD_DELAY", 0))
    rate = int(env_float("ACESTEP_STUB_RATE", 48000))
    audio_in = not os.environ.get("ACESTEP_STUB_NO_AUDIO_IN")
    out = Out()
    out.line("READY", {**READY, "sample_rate": rate, **(AUDIO_IN if audio_in else {})})
    lines = queue.Queue()

    def reader():
        for raw in sys.stdin.buffer:
            lines.put(raw)
        lines.put(None)
    threading.Thread(target=reader, daemon=True).start()
    pending = []

    def next_line(block=True):
        if pending:
            return pending.pop(0)
        try:
            return lines.get(block=block, timeout=None if block else 0)
        except queue.Empty:
            return b""

    while True:
        raw = next_line()
        if raw is None:
            return 0
        if not raw.strip():
            continue
        text = raw.decode("utf-8", "replace").rstrip("\r\n")
        keyword, _, body = text.partition(" ")
        if len(raw) > MAX_LINE:
            rid = "?"
            marker = text.find('"id":"')
            if marker >= 0:
                rid = text[marker + 6:].split('"', 1)[0]
            out.line("ERROR", {"id": rid, "message": "request line too long"})
            continue
        if keyword == "CANCEL":
            continue
        if keyword != "GEN":
            continue
        try:
            request = json.loads(body)
        except ValueError:
            out.line("ERROR", {"id": "?", "message": "bad GEN line (not JSON)"})
            continue
        if not isinstance(request, dict):
            out.line("ERROR", {"id": "?", "message": "bad GEN line (not a JSON object)"})
            continue
        rid = request.get("id") if isinstance(request.get("id"), str) else "?"
        problem = check(request, audio_in)
        if problem:
            out.line("ERROR", {"id": rid, "message": problem})
            continue
        caption = request["caption"]
        if "__stub_garbage__" in caption:
            sys.stdout.buffer.write(b"this is not a frame\n")
            sys.stdout.buffer.flush()
        if "__stub_error__" in caption:
            out.line("ERROR", {"id": rid, "message": "the stub was asked to fail"})
            continue
        steps = request.get("steps", 8)
        stop = False

        def poll():
            """CANCEL is honoured between stages and steps, like the real engine; any other line waits."""
            nonlocal stop
            while True:
                got = next_line(block=False)
                if got is None:                       # end of input: finish this request, then the loop exits
                    pending.append(None)
                    return stop
                if not got:
                    return stop
                t = got.decode("utf-8", "replace").rstrip("\r\n")
                kw, _, b = t.partition(" ")
                if kw == "CANCEL":
                    try:
                        if json.loads(b).get("id") == rid:
                            stop = True
                    except ValueError:
                        pass
                else:
                    pending.append(got)

        started = time.monotonic()
        # audio in: read the clips (the engine's first stage), in its order: the source, then the reference
        src_seconds, digests = 0.0, []
        clips = [(role, request.get(key)) for role, key in (("source", "src_path"), ("reference", "ref_path")) if request.get(key)]
        failed = False
        if clips and audio_in:
            out.line("PROGRESS", {"id": rid, "stage": "encode", "step": 0, "steps": 1})
            for role, path in clips:
                seconds, silent, extra = read_clip(path, role)
                if seconds is None:
                    out.line("ERROR", {"id": rid, "message": extra})
                    failed = True
                    break
                if role == "reference" and silent:
                    out.line("ERROR", {"id": rid, "message": "the reference clip is silent"})
                    failed = True
                    break
                digests.append(extra)
                if role == "source":
                    src_seconds = seconds
            if failed:
                continue
        planned = plan(request, src_seconds)
        if isinstance(planned, str):
            out.line("ERROR", {"id": rid, "message": planned})
            continue
        want48, frames = planned
        if clips and audio_in:
            out.line("PROGRESS", {"id": rid, "stage": "encode", "step": 1, "steps": 1})
            if poll():
                out.line("ERROR", {"id": rid, "message": "cancelled"})
                continue
        out.line("PROGRESS", {"id": rid, "stage": "text", "step": 0, "steps": 1})
        out.line("PROGRESS", {"id": rid, "stage": "condition", "step": 0, "steps": 1})
        for i in range(steps):
            out.line("PROGRESS", {"id": rid, "stage": "denoise", "step": i, "steps": steps})
            if poll():
                break
            if "__stub_crash__" in caption and i == steps // 2:
                os._exit(3)
            time.sleep(delay)
        if stop or poll():
            out.line("ERROR", {"id": rid, "message": "cancelled"})
            continue
        out.line("PROGRESS", {"id": rid, "stage": "denoise", "step": steps, "steps": steps})
        out.line("PROGRESS", {"id": rid, "stage": "decode", "step": 0, "steps": 1})
        pcm, samples = clip(request, rate, want48, frames, digests)
        lyrics_given = bool(request.get("lyrics"))
        out.line("AUDIO", {"id": rid, "sample_rate": rate, "channels": 2, "samples": samples, "bytes": len(pcm),
                           "seed": request.get("seed", 0), "steps": steps, "shift": request.get("shift", 1),
                           "duration": round(samples / rate, 3), "frames": frames,
                           "src_seconds": round(src_seconds, 3),
                           "caption_tokens": len(caption.split()) + 20, "lyric_tokens": 15 if not lyrics_given else 40,
                           "truncated": False, "peak": 0.5,
                           "timings": {"text": 0.0, "condition": 0.0, "denoise": round(time.monotonic() - started, 3), "decode": 0.0}},
                 pcm)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if not args.serve:
        parser.error("only --serve is implemented")
    sys.exit(serve())


if __name__ == "__main__":
    main()
