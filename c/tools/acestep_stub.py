#!/usr/bin/env python3
"""A stand-in for the acestep engine: same command line, same serve protocol.

The front ends (the gateway's POST /v1/audio/music, `coli music`) are tested without the
real engine, which needs 6 GB of weights. This script speaks the protocol byte for byte and
makes a synthetic clip that depends on the caption, the lyrics, the seed and the duration, so
a rendered song is recognisable and two requests that differ in one of them audibly differ.
It is not a model.

It is deliberately self-contained (stdlib only): it plays the part of a separate binary, and
sharing code with the client it is used to test would let a bug on one side hide the same bug
on the other.

    acestep_stub.py --model DIR --serve

Serve protocol (one request at a time, logs on stderr), the header of c/acestep_serve.h:
    engine -> READY {...}
    client -> GEN {"id","caption","lyrics","language","duration","steps","shift","seed","bpm",
                   "key","timesig","dcw"}
    engine -> PROGRESS {"id","stage","step","steps"}    stages text, condition, denoise, decode
    engine -> AUDIO {"id","sample_rate","channels","samples","bytes",...} + N bytes + "\\n"
    engine -> ERROR {"id","message"}
    client -> CANCEL {"id"}        answered with ERROR "cancelled"
    stdin EOF: clean exit

Knobs (environment), all optional:
    ACESTEP_STUB_DELAY       seconds per denoising step (default 0.05)
    ACESTEP_STUB_LOAD_DELAY  seconds before READY (default 0)
    ACESTEP_STUB_RATE        sample rate of the clips (default 48000; the tests use a low one to stay small)
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
MAX_LINE = 1 << 20


def log(message):
    sys.stderr.write(f"[acestep-stub] {message}\n")
    sys.stderr.flush()


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def check(request):
    """The engine's own refusals, in its own words (acestep_gen.h ace_check)."""
    caption = request.get("caption")
    if not isinstance(caption, str):
        return "caption is required (it may be empty)"
    if len(caption.encode()) > READY["max_caption_bytes"]:
        return f"caption longer than {READY['max_caption_bytes']} bytes"
    lyrics = request.get("lyrics")
    if lyrics is not None and (not isinstance(lyrics, str) or len(lyrics.encode()) > READY["max_lyric_bytes"]):
        return "lyrics must be a string of at most %d bytes" % READY["max_lyric_bytes"]
    duration = request.get("duration", READY["default_duration"])
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not 1 <= duration <= 600:
        return "duration must be 1 to 600 seconds"
    steps = request.get("steps", 8)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 20:
        return "steps must be 1 to 20"
    shift = request.get("shift", 1)
    if not isinstance(shift, (int, float)) or isinstance(shift, bool) or not 0 < shift <= 10:
        return "shift must be above 0 and at most 10"
    seed = request.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        return "seed must be a whole number from 0 to 4294967295"
    bpm = request.get("bpm", 0)
    if isinstance(bpm, bool) or not isinstance(bpm, int) or not 0 <= bpm <= 1000:
        return "bpm must be a whole number"
    return None


def clip(request, rate):
    """A recognisable clip: a few sine voices whose pitches come from a hash of the request."""
    duration = round(float(request.get("duration", 30)), 1)
    samples = int(duration * rate)
    digest = hashlib.sha256(json.dumps([request.get("caption"), request.get("lyrics"), request.get("language"),
                                        request.get("seed", 0), request.get("bpm"), request.get("key")]).encode()).digest()
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
    out = Out()
    out.line("READY", {**READY, "sample_rate": rate})
    lines = queue.Queue()

    def reader():
        for raw in sys.stdin.buffer:
            lines.put(raw)
        lines.put(None)
    threading.Thread(target=reader, daemon=True).start()
    cancelled = set()
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
        problem = check(request)
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
        pcm, samples = clip(request, rate)
        lyrics_given = bool(request.get("lyrics"))
        out.line("AUDIO", {"id": rid, "sample_rate": rate, "channels": 2, "samples": samples, "bytes": len(pcm),
                           "seed": request.get("seed", 0), "steps": steps, "shift": request.get("shift", 1),
                           "duration": round(samples / rate, 3), "frames": max(128, samples * 25 // rate),
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
