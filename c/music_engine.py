#!/usr/bin/env python3
"""The music engine from Python: its serve protocol, WAV, request rules, and the plan.

Shared by the gateway (openai_server.py, POST /v1/audio/music) and `coli music`, so
the two cannot disagree about what a valid request is or how audio is framed on the
wire. Stdlib only. The engine is c/acestep (docs/acestep.md); the protocol is the
header of c/acestep_serve.h, the same shape as the image engine's (image_engine.py):

    READY {...}                      once, after the weights are loaded: the limits and defaults;
                                     "tasks" and "audio_in" when the engine takes audio in
    GEN {...}                        client, one request at a time:
                                     caption, lyrics, language, duration, steps, shift, seed,
                                     bpm, key, timesig, dcw; and for audio in: task, src_path, ref_path,
                                     repaint_start, repaint_end, repaint_mode, repaint_strength,
                                     cover_strength, cover_noise (paths are files this process wrote)
    PROGRESS {...}                   any number: stage encode|text|condition|denoise|decode, step, steps
    AUDIO {..."bytes":N} + N + \\n    16-bit little-endian PCM, channels interleaved, 48 kHz
    ERROR {...}                      the engine stays alive for the next GEN
    CANCEL {...}                     client; answered with ERROR "cancelled"

Audio in (a cover of a clip, a repaint of a span of it, a reference clip to take the style from): a clip arrives
as bytes (a WAV or an MP3), is checked here by what it says it is, and is written to a private temporary directory
for the engine to read; the directory is removed when the request ends, however it ends. A client never names a
path. The request fields use upstream's own names: `task`, `src_audio`, `reference_audio`, `repainting_start`,
`repainting_end`, `repaint_mode`, `repaint_strength`, `audio_cover_strength`, `cover_noise_strength`.
"""
import base64
import json
import os
import queue
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from image_engine import (ABANDONED_TIMEOUT, _dtype_bytes, _parse_line, _tensors, describe_exit,
                          engine_command)

SAMPLE_RATE = 48000
CHANNELS = 2
# What an engine that says less than the contract is assumed to mean. The engine's
# READY line wins field by field; these only fill gaps.
DEFAULT_INFO = {"model": "ace-step-1.5-turbo", "sample_rate": SAMPLE_RATE, "channels": CHANNELS,
                "format": "s16le", "default_duration": 30, "min_duration": 1, "max_duration": 600,
                "default_steps": 8, "max_steps": 20, "default_shift": 1.0,
                "max_caption_bytes": 4096, "max_lyric_bytes": 32768, "max_seed": 2**32 - 1}
MAX_SEED = 2**32 - 1
MIN_STEPS = 1
MAX_STEPS = 20
MIN_SHIFT, MAX_SHIFT = 0.0, 10.0          # exclusive, inclusive
MAX_BPM = 1000
MAX_LANGUAGE = 32
MAX_KEY = 32
MAX_TIME_SIGNATURE = 8
# 600 s of 48 kHz 16-bit stereo is 115.2 MB; a frame past 160 MB is a desynchronised
# stream, not a song, and is refused before a byte of it is allocated.
MAX_FRAME_BYTES = 160 << 20
_FRAMES = ("READY", "PROGRESS", "AUDIO", "ERROR")
STAGES = ("encode", "text", "condition", "denoise", "decode")
RESPONSE_FORMATS = (None, "b64_json", "wav")
# Audio in. What the engine announces in READY wins (`tasks`, `audio_in`); these are the contract's own numbers.
TASKS = ("text2music", "cover", "repaint")
REPAINT_MODES = ("balanced", "conservative", "aggressive")
DEFAULT_AUDIO_IN = {"formats": ["wav", "mp3"], "max_seconds": 600, "ref_seconds": 30}
# A clip is held in memory as bytes, then as the temporary file: 128 MiB is 12 minutes of 48 kHz 16-bit stereo WAV
# (the engine takes 10) or 7 hours of 320 kbps MP3; the pair together may not pass 192 MiB.
MAX_CLIP_BYTES = 128 << 20
MAX_CLIPS_BYTES = 192 << 20
MIN_CLIP_SECONDS = 1.0                    # the engine refuses a clip shorter than one second
MAX_REPAINT_SPAN = 3600.0                 # seconds, on either side of zero: the engine's own bound
CLIP_FIELDS = ("src_audio", "reference_audio")


class MusicEngineError(RuntimeError):
    """The engine refused or failed a request, or is not running."""


class MusicCancelled(Exception):
    """The request was cancelled and the engine confirmed it."""


class MusicRequestError(ValueError):
    """A request the engine would refuse, caught before it is sent."""

    def __init__(self, message, param=None):
        super().__init__(message)
        self.param = param


# ---------------------------------------------------------------- the process
class MusicEngine:
    """One engine process and the one request it can work on at a time.

    A reader thread turns stdout into frames on a queue, so the caller can wait for the
    next frame with a timeout and notice, in between, that its client went away: that is
    when CANCEL goes out. Frames carry the request id; a frame for any other id (a request
    abandoned without its final frame) is dropped rather than attributed to the current one."""

    modality = "audio"

    def __init__(self, executable, model, env=None, stderr=None, load_timeout=None,
                 on_log=None, on_stderr=None):
        """`stderr` is where the engine's log goes (inherited by default). With `on_stderr` it
        is read line by line from the start and handed to that callable: a reader that began
        only after READY would let a chatty load fill the pipe and stall the engine."""
        self.model_dir = str(model)
        self.executable = str(executable)
        self.on_log = on_log or (lambda message: sys.stderr.write(f"[music] {message}\n"))
        if on_stderr is not None:
            stderr = subprocess.PIPE
        child_env = dict(os.environ if env is None else env)
        # SNAP and SERVE=1 are how `coli stop` recognises an engine left running for a port.
        child_env["SNAP"] = str(model)
        child_env["SERVE"] = "1"
        # Its own process group (a new session on POSIX): Ctrl-C in the terminal reaches coli,
        # which turns it into CANCEL, instead of killing the engine in the middle of a song.
        # Losing coli still ends the engine: its stdin closes, and EOF is the clean exit.
        spawn = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                 if sys.platform == "win32" else {"start_new_session": True})
        try:
            self.process = subprocess.Popen(
                engine_command(executable, model), env=child_env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=stderr, bufsize=0, **spawn)
        except OSError as error:
            raise MusicEngineError(f"cannot start the music engine {executable}: {error}") from error
        if on_stderr is not None:
            def drain(stream=self.process.stderr):
                try:
                    for raw in iter(stream.readline, b""):
                        on_stderr(raw.decode("utf-8", "replace").rstrip("\r\n"))
                except (OSError, ValueError):
                    pass
            threading.Thread(target=drain, name="acestep-stderr", daemon=True).start()
        self.lock = threading.Lock()          # one request at a time
        self.write_lock = threading.Lock()
        self.events = queue.Queue()
        self.dead = None                      # why the engine is gone, once it is
        self._seq = 0
        self._abandoned = None                # id of a request left without its final frame
        self._leftovers = []                  # clip directories of abandoned requests, removed once the engine is done
        try:
            self.info = self._await_ready(load_timeout)
        except BaseException:
            # A failed start must not leave a process or three pipes behind.
            self._kill_quietly()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass
            raise
        self.dispatcher = threading.Thread(target=self._dispatch, name="acestep-stdout", daemon=True)
        self.dispatcher.start()

    # -- startup -------------------------------------------------------------
    def _await_ready(self, timeout):
        timer = None
        if timeout:
            timer = threading.Timer(timeout, self._kill_quietly)
            timer.daemon = True
            timer.start()
        try:
            while True:
                raw = self.process.stdout.readline()
                if not raw:
                    try:
                        code = self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        code = None
                    if timer is not None and not timer.is_alive() and timeout:
                        raise MusicEngineError(f"the music engine did not become ready within {timeout:g}s")
                    raise MusicEngineError(f"the music engine exited while loading ({describe_exit(code)})")
                keyword, body = _parse_line(raw)
                if keyword == "READY" and body is not None:
                    info = dict(DEFAULT_INFO)
                    info.update({k: v for k, v in body.items() if v is not None})
                    return info
                if keyword == "ERROR" and body is not None:
                    self._kill_quietly()
                    raise MusicEngineError(f"the music engine failed to load: {body.get('message') or 'unknown error'}")
                self.on_log("ignoring engine output before READY: " + raw.decode("utf-8", "replace").strip()[:200])
        finally:
            if timer is not None:
                timer.cancel()

    def _kill_quietly(self):
        try:
            self.process.kill()
        except OSError:
            pass

    # -- reading -------------------------------------------------------------
    def _read_exact(self, size):
        chunks, remaining = [], size
        while remaining:
            chunk = self.process.stdout.read(min(remaining, 1 << 20))
            if not chunk:
                raise MusicEngineError("the music engine closed its output in the middle of a frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _dispatch(self):
        try:
            while True:
                raw = self.process.stdout.readline()
                if not raw:
                    break
                keyword, body = _parse_line(raw)
                if keyword not in _FRAMES or body is None:
                    # A stray printf on stdout must not stop the reader: every later frame is
                    # still well formed, and the audio is read by count, never by scanning.
                    self.on_log("ignoring stray engine output: " + raw.decode("utf-8", "replace").strip()[:200])
                    continue
                payload = None
                if keyword == "AUDIO":
                    size = body.get("bytes")
                    if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= MAX_FRAME_BYTES:
                        raise MusicEngineError(f"invalid AUDIO frame size {size!r}")
                    payload = self._read_exact(size)
                    if self.process.stdout.read(1) != b"\n":
                        raise MusicEngineError("AUDIO payload is not followed by a newline")
                self.events.put((keyword, body, payload))
        except Exception as error:            # the stream is unusable from here on
            self.events.put(("_error", {"message": str(error)}, None))
            self._kill_quietly()
        finally:
            self.events.put(("_eof", {}, None))

    # -- writing -------------------------------------------------------------
    def _send(self, keyword, body):
        data = f"{keyword} {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n".encode("utf-8")
        try:
            with self.write_lock:
                view = memoryview(data)
                while view:
                    sent = self.process.stdin.write(view)
                    if not sent:
                        raise OSError("the engine's stdin accepted no bytes")
                    view = view[sent:]
                self.process.stdin.flush()
        except (OSError, ValueError) as error:
            raise MusicEngineError(f"cannot send {keyword} to the music engine ({error})") from error

    @property
    def alive(self):
        return self.dead is None and self.process.poll() is None

    @property
    def tasks(self):
        """The tasks the engine announced: just text2music for an engine that says nothing."""
        return tuple(self.info.get("tasks") or ("text2music",))

    def generate(self, caption, duration=None, lyrics=None, language=None, steps=None, shift=None, seed=None,
                 bpm=None, key=None, time_signature=None, dcw=None, on_progress=None, cancelled=None,
                 on_idle=None, interruptible=False, on_interrupt=None, task=None, src_audio=None,
                 ref_audio=None, repaint_start=None, repaint_end=None, repaint_mode=None,
                 repaint_strength=None, cover_strength=None, cover_noise=None):
        """Run one GEN to its final frame and return the audio.

        Fields left as None are left out of the request, so the engine's own defaults apply (no lyrics
        is instrumental, no language is "unknown"). Returns {"pcm", "sample_rate", "channels", "samples",
        "seed", "steps", "shift", "duration", "frames", "truncated", "peak", "src_seconds", "timings", ...}
        with `pcm` the 16-bit little-endian interleaved bytes. Raises MusicCancelled once the engine
        confirms a CANCEL and MusicEngineError for anything the engine refuses or a process that dies.
        `cancelled` is polled between frames and on every idle tick; with `interruptible` the first Ctrl-C
        sends CANCEL and keeps waiting (the engine is busy until it answers), the second one propagates.

        Audio in: `task` ("cover" or "repaint"; a cover or a repaint needs `src_audio`), `src_audio` and
        `ref_audio` (a reference clip, any task) as the bytes of a WAV or MP3 file, `repaint_start` and
        `repaint_end` in seconds of the source (an end of 0 or less is its end), `repaint_mode`,
        `repaint_strength`, `cover_strength`, `cover_noise` (the engine's own names, see the header of
        acestep_serve.h). The clips are written to a private temporary directory for the engine, and it
        is removed when the request ends: found, failed, cancelled or abandoned."""
        request = {"caption": caption}
        for name, value in (("lyrics", lyrics), ("language", language), ("duration", duration), ("steps", steps),
                            ("shift", shift), ("seed", seed), ("bpm", bpm), ("key", key),
                            ("timesig", time_signature), ("dcw", dcw)):
            if value is not None:
                request[name] = value
        audio_in = (task not in (None, "text2music") or src_audio is not None or ref_audio is not None)
        if audio_in and "cover" not in self.tasks:
            raise MusicEngineError("this music engine does not take audio in (it announced no tasks): "
                                   "update the engine")
        if task not in (None, "text2music"):
            request["task"] = task
        for name, value in (("repaint_start", repaint_start), ("repaint_mode", repaint_mode),
                            ("repaint_strength", repaint_strength), ("cover_strength", cover_strength),
                            ("cover_noise", cover_noise)):
            if value is not None:
                request[name] = value
        if repaint_end is not None and repaint_end > 0:
            request["repaint_end"] = repaint_end
        staging = None
        try:
            if src_audio is not None or ref_audio is not None:
                staging = _stage_clips(src_audio, ref_audio)
                request.update(staging["paths"])
            return self._generate(request, seed, on_progress, cancelled, on_idle, interruptible, on_interrupt, staging)
        finally:
            if staging is not None:
                # An abandoned request (a callback raised, the caller was interrupted) is still being read by the
                # engine: its directory goes at the next request, when the engine has said it is done.
                if staging.get("abandoned"):
                    self._leftovers.append(staging["dir"])
                else:
                    _remove_dir(staging["dir"])

    def _generate(self, request, seed, on_progress, cancelled, on_idle, interruptible, on_interrupt, staging):
        with self.lock:
            if self.dead:
                raise MusicEngineError(self.dead)
            self._finish_abandoned()
            self._sweep()
            self._seq += 1
            rid = f"r{self._seq}"
            self._send("GEN", {"id": rid, **request})
            cancel_sent = False

            def cancel():
                nonlocal cancel_sent
                if not cancel_sent:
                    cancel_sent = True
                    self._send("CANCEL", {"id": rid})

            finished = False
            try:
                while True:
                    try:
                        try:
                            kind, body, payload = self.events.get(timeout=0.2)
                        except queue.Empty:
                            if cancelled is not None and cancelled():
                                cancel()
                            if on_idle is not None:
                                on_idle()
                            continue
                    except KeyboardInterrupt:
                        if not interruptible or cancel_sent:
                            raise
                        cancel()
                        if on_interrupt is not None:
                            on_interrupt()
                        continue
                    if kind == "_error":
                        self.dead = f"the music engine protocol broke: {body.get('message')}"
                        continue                  # the _eof that follows raises
                    if kind == "_eof":
                        try:
                            code = self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            code = None
                        self.dead = self.dead or f"the music engine exited ({describe_exit(code)})"
                        raise MusicEngineError(self.dead)
                    if body.get("id") not in (rid, None, ""):
                        continue                  # a frame for an abandoned request
                    if kind == "PROGRESS":
                        if on_progress is not None:
                            on_progress(body)
                    elif kind == "AUDIO":
                        finished = True           # the engine is done with this request
                        rate, channels, samples = body.get("sample_rate"), body.get("channels"), body.get("samples")
                        if (not all(isinstance(v, int) and not isinstance(v, bool) for v in (rate, channels, samples)) or
                                rate < 1 or channels < 1 or samples < 0 or len(payload) != samples * channels * 2):
                            raise MusicEngineError(
                                f"AUDIO frame does not describe its payload ({samples} samples of {channels} channels "
                                f"at {rate} Hz, {len(payload)} bytes)")
                        result = {key_: body[key_] for key_ in body if key_ != "id"}
                        result["pcm"] = payload
                        result.setdefault("seed", seed)
                        result.setdefault("timings", {})
                        return result
                    elif kind == "ERROR":
                        finished = True
                        message = str(body.get("message") or "unknown engine error")
                        if message == "cancelled":
                            # A cancel is an outcome, not a failure, whoever asked for it.
                            raise MusicCancelled()
                        raise MusicEngineError(message)
                    if cancelled is not None and cancelled():
                        cancel()
            finally:
                if not finished and not self.dead:
                    # Left without the final frame (a callback raised, the caller was interrupted):
                    # the engine is still working. Ask it to stop, and make the next request drain
                    # this one's frames first, or they would be taken for its own.
                    try:
                        if not cancel_sent:
                            self._send("CANCEL", {"id": rid})
                    except MusicEngineError:
                        pass
                    self._abandoned = rid
                    if staging is not None:
                        staging["abandoned"] = True

    def _sweep(self):
        """Remove the clip directories of requests that were abandoned: the engine has finished with them."""
        left, self._leftovers = self._leftovers, []
        for path in left:
            if not _remove_dir(path):
                self._leftovers.append(path)

    def _finish_abandoned(self):
        """Drain the frames of an abandoned request up to its final one."""
        rid, self._abandoned = self._abandoned, None
        if rid is None:
            return
        deadline = time.monotonic() + ABANDONED_TIMEOUT
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self.dead = "the music engine did not finish an abandoned request"
                self._kill_quietly()
                raise MusicEngineError(self.dead)
            try:
                kind, body, _payload = self.events.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if kind in ("_error", "_eof"):
                self.events.put((kind, body, None))   # the request that follows sees it
                return
            if kind in ("AUDIO", "ERROR") and body.get("id") == rid:
                return

    def close(self, timeout=10):
        """stdin EOF is the engine's clean exit; escalate only if it ignores it."""
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
        self.dead = self.dead or "the music engine was closed"
        self._sweep()
        dispatcher = getattr(self, "dispatcher", None)
        if dispatcher is not None and dispatcher is not threading.current_thread():
            dispatcher.join(timeout=5)
        if self.process.poll() is not None:
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass


# ---------------------------------------------------------------- requests
def random_seed():
    return random.SystemRandom().randrange(0, MAX_SEED + 1)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


# ---------------------------------------------------------------- audio in
def audio_in_info(info):
    """The engine's audio-in limits (READY's `audio_in`), the contract's own for what it leaves out, or None
    when the engine announced no tasks beyond text to music."""
    info = info or {}
    if "cover" not in (info.get("tasks") or ()):
        return None
    return {**DEFAULT_AUDIO_IN, **(info.get("audio_in") or {})}


def _mpeg_header(data, at):
    """True when data[at:at+4] is an MPEG audio layer III frame header: sync, a real version, layer III, a bitrate
    and a sample rate that exist."""
    if at + 4 > len(data) or data[at] != 0xFF or data[at + 1] & 0xE0 != 0xE0:
        return False
    version, layer = (data[at + 1] >> 3) & 3, (data[at + 1] >> 1) & 3
    return version != 1 and layer == 1 and data[at + 2] >> 4 not in (0, 15) and (data[at + 2] >> 2) & 3 != 3


def audio_format(data):
    """"wav" for a RIFF/WAVE file, "mp3" for one that starts with an ID3v2 tag or an MPEG layer III frame (leading
    bytes up to 4 KiB are skipped when a frame header follows), else the name of what it is when that is a format
    the engine does not read (flac, ogg, m4a, aiff), else None."""
    head = bytes(data[:16])
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:3] == b"ID3" or _mpeg_header(data, 0):
        return "mp3"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] == b"ftyp":
        return "m4a"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    for at in range(1, min(len(data), 4096)):
        if _mpeg_header(data, at):
            return "mp3"
    return None


def wav_seconds(data):
    """The length in seconds a WAV file's header and data chunk say, or None when they cannot be read. A data chunk
    that claims more than the file holds counts what is there, as the engine's reader does."""
    return _wav_seconds(bytes(data[:1 << 16]), len(data))


def _wav_seconds(head, total):
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return None
    position, rate, frame = 12, None, None
    while position + 8 <= len(head):
        kind, size = head[position:position + 4], struct.unpack("<I", head[position + 4:position + 8])[0]
        if kind == b"fmt " and position + 8 + 16 <= len(head):
            _tag, channels, rate, _bps, _align, bits = struct.unpack("<HHIIHH", head[position + 8:position + 24])
            frame = channels * ((bits + 7) // 8)
        elif kind == b"data":
            if not rate or not frame:
                return None
            return min(size, max(0, total - position - 8)) / frame / rate
        position += 8 + size + (size & 1)
    return None


def _remove_dir(path):
    """True once `path` is gone (or never was)."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return True
    except OSError:
        return not os.path.exists(path)
    return True


def _stage_clips(src, ref):
    """Write the clips to a private temporary directory (0700, files 0600): {"dir", "paths": {"src_path", "ref_path"}}.
    Whatever goes wrong half way, the directory does not stay."""
    directory = tempfile.mkdtemp(prefix="coli-music-")
    try:
        paths = {}
        for key, data in (("src_path", src), ("ref_path", ref)):
            if data is None:
                continue
            name = os.path.join(directory, ("source" if key == "src_path" else "reference") +
                                {"mp3": ".mp3"}.get(audio_format(data), ".wav"))
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            paths[key] = name
        return {"dir": directory, "paths": paths}
    except BaseException:
        _remove_dir(directory)
        raise


def _clip(value, where, limit_audio):
    """A clip of a request -> bytes. A string is base64 (a `data:audio/...;base64,` URL too), bytes are taken as they
    are (a multipart upload). Nothing is fetched from a URL or read from a path the client names."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    elif isinstance(value, str):
        text = value.strip()
        if text[:5].lower() == "data:":
            head, comma, text = text.partition(",")
            if not comma or not head.lower().endswith(";base64"):
                raise MusicRequestError(f"`{where}`: a data: URL must be base64-encoded.", where)
        elif re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
            raise MusicRequestError(f"`{where}`: this server does not fetch audio from URLs; send the file "
                                    "(multipart/form-data), a data: URL or base64.", where)
        if len(text) * 3 // 4 > MAX_CLIP_BYTES + 3:
            raise MusicRequestError(f"`{where}` is longer than {MAX_CLIP_BYTES >> 20} MiB.", where)
        try:
            data = base64.b64decode(re.sub(r"\s+", "", text), validate=True)
        except (ValueError, TypeError):
            raise MusicRequestError(f"`{where}` is not valid base64.", where) from None
    else:
        raise MusicRequestError(f"`{where}` must be a WAV or MP3 file: base64 text, or an uploaded file.", where)
    if not data:
        raise MusicRequestError(f"`{where}` is empty.", where)
    if len(data) > MAX_CLIP_BYTES:
        raise MusicRequestError(f"`{where}` is longer than {MAX_CLIP_BYTES >> 20} MiB.", where)
    kind = audio_format(data)
    if kind not in ("wav", "mp3"):
        what = {"flac": "a FLAC file", "ogg": "an Ogg file", "m4a": "an M4A/MP4 file", "aiff": "an AIFF file"}.get(kind)
        raise MusicRequestError(f"`{where}` is {what or 'not a WAV or MP3 file'}: send a WAV or an MP3.", where)
    if kind == "wav":
        seconds = wav_seconds(data)
        if seconds is not None:
            if seconds < MIN_CLIP_SECONDS:
                raise MusicRequestError(f"`{where}` is shorter than {MIN_CLIP_SECONDS:g} second.", where)
            if seconds > limit_audio["max_seconds"]:
                raise MusicRequestError(f"`{where}` is {seconds:.0f} seconds long; at most "
                                        f"{limit_audio['max_seconds']:g} are read.", where)
    return data


def _unit_number(body, name, low_open=False, default=None):
    """`name` as a number in [0, 1] ((0, 1] with low_open), or None when the body does not give it."""
    value = body.get(name)
    if value is None:
        return default
    if not _number(value) or not (0 < value <= 1 if low_open else 0 <= value <= 1):
        raise MusicRequestError(f"`{name}` must be a number above 0 and at most 1." if low_open
                                else f"`{name}` must be a number between 0 and 1.", name)
    return float(value)


def _audio_request(body, info):
    """The audio-in half of a music request: task, clips and the task's settings, validated."""
    task = body.get("task")
    if task is None:
        task = "text2music"
    if task not in TASKS:
        raise MusicRequestError("`task` must be \"text2music\", \"cover\" or \"repaint\".", "task")
    limits = audio_in_info(info)
    clips = {name: body.get(name) for name in CLIP_FIELDS if body.get(name) is not None}
    given = [name for name in ("audio_cover_strength", "cover_noise_strength", "repainting_start", "repainting_end",
                               "repaint_mode", "repaint_strength") if body.get(name) is not None]
    if limits is None and (task != "text2music" or clips):
        raise MusicRequestError("This music engine does not take audio in (it announced no tasks): update the "
                                "engine.", "task" if task != "text2music" else next(iter(clips)))
    if task == "text2music":
        if "src_audio" in clips:
            raise MusicRequestError("`src_audio` is for a cover or a repaint; for text to music, give a reference "
                                    "clip (`reference_audio`) to take its style.", "src_audio")
        for name in given:
            raise MusicRequestError(f"`{name}` is for a {'cover' if 'cover' in name else 'repaint'}: set `task`.", name)
    elif "src_audio" not in clips:
        raise MusicRequestError(f"`src_audio` is required for task `{task}`: the clip to {task}.", "src_audio")
    cover_only, repaint_only = ("audio_cover_strength", "cover_noise_strength"), \
        ("repainting_start", "repainting_end", "repaint_mode", "repaint_strength")
    if task == "cover":
        for name in repaint_only:
            if body.get(name) is not None:
                raise MusicRequestError(f"`{name}` is for a repaint; this is a cover.", name)
    if task == "repaint":
        for name in cover_only:
            if body.get(name) is not None:
                raise MusicRequestError(f"`{name}` is for a cover; this is a repaint.", name)
    out = {"task": task, "src_audio": None, "reference_audio": None, "repainting_start": None, "repainting_end": None,
           "repaint_mode": None, "repaint_strength": None, "audio_cover_strength": None, "cover_noise_strength": None}
    total = 0
    for name in CLIP_FIELDS:
        if name in clips:
            out[name] = _clip(clips[name], name, limits)
            total += len(out[name])
            if total > MAX_CLIPS_BYTES:
                raise MusicRequestError(f"`src_audio` and `reference_audio` together are longer than "
                                        f"{MAX_CLIPS_BYTES >> 20} MiB.", name)
    if task == "cover":
        # 0 would be read by the engine as "not given" (the default, 1), so it is refused rather than misread
        out["audio_cover_strength"] = _unit_number(body, "audio_cover_strength", low_open=True)
        out["cover_noise_strength"] = _unit_number(body, "cover_noise_strength")
    if task == "repaint":
        for name in ("repainting_start", "repainting_end"):
            value = body.get(name)
            if value is not None and (not _number(value) or abs(value) > MAX_REPAINT_SPAN):
                raise MusicRequestError(f"`{name}` must be a number of seconds within {MAX_REPAINT_SPAN:g} of the "
                                        "start of the clip.", name)
            out[name] = None if value is None else float(value)
        start, end = out["repainting_start"], out["repainting_end"]
        if end is not None and end > 0 and end <= (start or 0.0):
            raise MusicRequestError("`repainting_end` must be after `repainting_start` (an end of 0 or less means "
                                    "the end of the clip).", "repainting_end")
        mode = body.get("repaint_mode")
        if mode is not None and mode not in REPAINT_MODES:
            raise MusicRequestError("`repaint_mode` must be \"balanced\", \"conservative\" or \"aggressive\".", "repaint_mode")
        out["repaint_mode"] = mode
        # 0 would be read by the engine as "not given" (the default, 0.5); in balanced mode it is the conservative
        # mode, which is asked for by name
        out["repaint_strength"] = _unit_number(body, "repaint_strength", low_open=True)
    return out


def music_request(body, info=None):
    """Validate a music request against the engine's READY (`info`, or the contract's defaults).

    The body is what POST /v1/audio/music takes: `prompt` (the caption; `caption` is accepted too),
    `lyrics`, `language`, `duration` in seconds, `steps`, `shift`, `seed`, `bpm`, `key`,
    `time_signature`, `dcw`, `response_format`; and for audio in, with upstream's own names, `task`
    ("text2music", "cover" or "repaint"), `src_audio` (the clip to cover or repaint) and `reference_audio`
    (a clip to take the style from) as base64, a data: URL or bytes of a WAV or MP3 file, `repainting_start`
    and `repainting_end` in seconds, `repaint_mode`, `repaint_strength`, `audio_cover_strength`,
    `cover_noise_strength`. Returns {"caption", "lyrics", "language", "duration", "steps", "shift", "seed",
    "seed_given", "bpm", "key", "time_signature", "dcw", "response_format"} with the defaults filled in, plus the
    audio-in fields (task, src_audio and reference_audio as decoded bytes or None, the settings or None);
    raises MusicRequestError naming the offending parameter. For a cover or a repaint the clip's own length is
    the song's, and `duration` is not read. An empty `lyrics` means none (instrumental); one of `prompt` and
    `lyrics` must say something."""
    info = {**DEFAULT_INFO, **(info or {})}
    if not isinstance(body, dict):
        raise MusicRequestError("The request must be a JSON object.")
    caption = body.get("prompt", body.get("caption"))
    if body.get("prompt") is not None and body.get("caption") is not None and body["prompt"] != body["caption"]:
        raise MusicRequestError("Give `prompt` or `caption`, not two different ones.", "prompt")
    if caption is not None and not isinstance(caption, str):
        raise MusicRequestError("`prompt` must be a string.", "prompt")
    caption = caption or ""
    lyrics = body.get("lyrics")
    if lyrics is not None and not isinstance(lyrics, str):
        raise MusicRequestError("`lyrics` must be a string.", "lyrics")
    lyrics = lyrics if lyrics and lyrics.strip() else None
    if not caption.strip() and lyrics is None and body.get("task") not in ("cover", "repaint"):
        # a cover or a repaint may say nothing: the clip is what it works from
        raise MusicRequestError("Say what the music is like in `prompt`, or give `lyrics`.", "prompt")
    if len(caption.encode("utf-8", "surrogatepass")) > int(info["max_caption_bytes"]):
        raise MusicRequestError(f"`prompt` is longer than {info['max_caption_bytes']} bytes.", "prompt")
    if lyrics is not None and len(lyrics.encode("utf-8", "surrogatepass")) > int(info["max_lyric_bytes"]):
        raise MusicRequestError(f"`lyrics` is longer than {info['max_lyric_bytes']} bytes.", "lyrics")
    n = body.get("n", 1)
    if n is not None and (isinstance(n, bool) or n != 1):
        raise MusicRequestError("Only `n`: 1 is supported; the engine makes one song at a time.", "n")
    response_format = body.get("response_format")
    if response_format not in RESPONSE_FORMATS:
        raise MusicRequestError("`response_format` must be \"b64_json\" or \"wav\".", "response_format")
    language = body.get("language")
    if language is not None and (not isinstance(language, str) or len(language) > MAX_LANGUAGE):
        raise MusicRequestError(f"`language` must be a language code of at most {MAX_LANGUAGE} characters.", "language")
    language = language or None
    audio = _audio_request(body, info)
    low, high = float(info["min_duration"]), float(info["max_duration"])
    duration = body.get("duration")
    duration = info["default_duration"] if duration is None else duration
    if audio["task"] != "text2music":
        duration = None                       # the clip's own length is the song's
    elif not _number(duration) or not low <= duration <= high:
        raise MusicRequestError(f"`duration` must be a number of seconds between {low:g} and {high:g}.", "duration")
    steps = body.get("steps")
    steps = info["default_steps"] if steps is None else steps
    max_steps = int(info["max_steps"])
    if isinstance(steps, bool) or not isinstance(steps, int) or not MIN_STEPS <= steps <= max_steps:
        raise MusicRequestError(f"`steps` must be an integer between {MIN_STEPS} and {max_steps}.", "steps")
    shift = body.get("shift")
    shift = info["default_shift"] if shift is None else shift
    if not _number(shift) or not MIN_SHIFT < shift <= MAX_SHIFT:
        raise MusicRequestError(f"`shift` must be a number above {MIN_SHIFT:g} and at most {MAX_SHIFT:g}.", "shift")
    seed = body.get("seed")
    seed_given = seed is not None
    max_seed = int(info["max_seed"])
    if seed_given:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= max_seed:
            raise MusicRequestError(f"`seed` must be an integer between 0 and {max_seed}.", "seed")
    else:
        seed = random_seed()
    bpm = body.get("bpm")
    if bpm is not None and (isinstance(bpm, bool) or not isinstance(bpm, int) or not 1 <= bpm <= MAX_BPM):
        raise MusicRequestError(f"`bpm` must be an integer between 1 and {MAX_BPM}.", "bpm")
    key = body.get("key")
    if key is not None and (not isinstance(key, str) or len(key) > MAX_KEY):
        raise MusicRequestError(f"`key` must be a string of at most {MAX_KEY} characters, like \"C Major\".", "key")
    signature = body.get("time_signature")
    if signature is not None:
        if isinstance(signature, int) and not isinstance(signature, bool):
            signature = str(signature)
        if not isinstance(signature, str) or len(signature) > MAX_TIME_SIGNATURE:
            raise MusicRequestError(f"`time_signature` must be a string of at most {MAX_TIME_SIGNATURE} characters, "
                                    "like \"4\".", "time_signature")
    dcw = body.get("dcw")
    if dcw is not None and not isinstance(dcw, bool):
        raise MusicRequestError("`dcw` must be true or false.", "dcw")
    return {"caption": caption, "lyrics": lyrics, "language": language, "duration": duration, "steps": steps,
            "shift": shift, "seed": seed, "seed_given": seed_given, "bpm": bpm, "key": key or None,
            "time_signature": signature or None, "dcw": True if dcw is None else dcw,
            "response_format": response_format, **audio}


# ---------------------------------------------------------------- WAV
def encode_wav(pcm, sample_rate=SAMPLE_RATE, channels=CHANNELS):
    """A canonical 44-byte-header PCM WAV around 16-bit little-endian interleaved samples."""
    if len(pcm) % (2 * channels):
        raise ValueError("the PCM is not a whole number of frames")
    if len(pcm) > 0xFFFFFFFF - 36:
        raise ValueError("the PCM is too long for a WAV file")
    header = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
              struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, sample_rate * channels * 2, channels * 2, 16) +
              b"data" + struct.pack("<I", len(pcm)))
    return header + pcm


def decode_wav(data):
    """(pcm bytes, sample_rate, channels) from a 16-bit PCM WAV, for tests and for reading a clip back. Refuses
    anything else by name rather than guessing."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    position, fmt, pcm = 12, None, None
    while position + 8 <= len(data):
        kind, size = data[position:position + 4], struct.unpack("<I", data[position + 4:position + 8])[0]
        body = data[position + 8:position + 8 + size]
        if kind == b"fmt ":
            if len(body) < 16:
                raise ValueError("short fmt chunk")
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif kind == b"data":
            pcm = body
        position += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("no fmt or data chunk")
    tag, channels, rate, _, _, bits = fmt
    if tag != 1 or bits != 16:
        raise ValueError(f"only 16-bit PCM is read (format {tag}, {bits} bits)")
    return pcm, rate, channels


# ---------------------------------------------------------------- saving
def music_dir(env=None):
    env = os.environ if env is None else env
    configured = env.get("COLI_MUSIC_DIR")
    return os.path.expanduser(configured) if configured else os.path.join(os.path.expanduser("~"), "colibri-music")


def _slug_of(prompt, limit=48):
    words = re.sub(r"[^a-z0-9]+", "-", (prompt or "").lower()).strip("-")
    return words[:limit].rstrip("-") or "music"


def save_wav(wav, prompt, seed, directory=None, now=None):
    """Write the WAV under a name that says what it is and never overwrites.

    <dir>/20260927-153012-calm-piano-with-soft-strings-s42.wav; a second song in the same
    second gets -2, -3, ... The file appears complete or not at all."""
    directory = directory or music_dir()
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    stem = f"{stamp}-{_slug_of(prompt)}-s{seed}"
    path = os.path.join(directory, stem + ".wav")
    counter = 2
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}-{counter}.wav")
        counter += 1
    temporary = path + f".{os.getpid()}.tmp"
    with open(temporary, "wb") as handle:
        handle.write(wav)
    os.replace(temporary, path)
    return path


def wav_b64(pcm, sample_rate=SAMPLE_RATE, channels=CHANNELS):
    return base64.b64encode(encode_wav(pcm, sample_rate, channels)).decode("ascii")


# ---------------------------------------------------------------- the model
DIT_DIR = "acestep-v15-turbo"
TEXT_DIR = "Qwen3-Embedding-0.6B"
VAE_DIR = "vae"
COMPONENTS = (TEXT_DIR, DIT_DIR, VAE_DIR)
TOKENIZER_FILE = TEXT_DIR + "/tokenizer.json"
SILENCE_FILE = DIT_DIR + "/silence_latent.pt"


def is_music_model_dir(model_dir):
    """An ACE-Step checkpoint: its root config says model_type acestep and the turbo DiT is in its directory."""
    if not model_dir:
        return False
    root = Path(model_dir).expanduser()
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(config, dict) and config.get("model_type") == "acestep" and
            (root / DIT_DIR / "model.safetensors").is_file())


def inspect_music_model(model_dir):
    """What is on disk, per component, and what each will weigh in RAM.

    The engine keeps the DiT and the text encoder as the checkpoint ships them (bf16, widened inside each
    product) and widens the VAE decoder to f32. What only audio in needs is loaded the first time a request
    brings a clip, and kept after: the VAE's encoder (widened to f32) and the DiT's audio tokenizer and
    detokenizer (a cover's). Those are counted as `on_demand`, not `resident`. Nothing under acestep-5Hz-lm-*
    is loaded (the language-model planner, not used by this engine)."""
    root = Path(model_dir).expanduser()
    components = {}
    for name in COMPONENTS:
        directory = root / name
        try:
            files = sorted(p for p in directory.iterdir() if p.is_file())
        except OSError:
            files = []
        weights = [p for p in files if p.suffix == ".safetensors"]
        entry = {"name": name, "present": directory.is_dir(), "disk_bytes": sum(p.stat().st_size for p in files),
                 "weight_files": len(weights), "params": 0, "dtypes": [], "resident_bytes": 0,
                 "on_demand_params": 0, "on_demand_bytes": 0}
        dtypes, params, resident, deferred, deferred_bytes = set(), 0, 0, 0, 0
        for shard in weights:
            for tensor, dtype, shape, nbytes in _tensors(shard):
                dtypes.add(dtype)
                elements = 1
                for dim in shape:
                    elements *= dim
                params += elements
                if name == VAE_DIR:
                    widened = elements * 4 if _dtype_bytes(dtype) else nbytes
                    if tensor.startswith("encoder."):
                        deferred += elements
                        deferred_bytes += widened
                        continue
                    resident += widened
                elif name == DIT_DIR and tensor.startswith(("tokenizer.", "detokenizer.")):
                    deferred += elements
                    deferred_bytes += nbytes
                else:
                    resident += nbytes
        entry.update(params=params, dtypes=sorted(dtypes), resident_bytes=resident, on_demand_params=deferred,
                     on_demand_bytes=deferred_bytes)
        components[name] = entry
    planner = sorted(p.name for p in root.glob("acestep-5Hz-lm-*") if p.is_dir()) if root.is_dir() else []
    planner_bytes = sum(f.stat().st_size for d in planner for f in (root / d).rglob("*") if f.is_file())
    return {"path": str(root), "components": components, "tokenizer": (root / TOKENIZER_FILE).is_file(),
            "silence_latent": (root / SILENCE_FILE).is_file(), "planner": planner, "planner_bytes": planner_bytes,
            "disk_bytes": sum(c["disk_bytes"] for c in components.values())}


def plan_music_model(model_dir, available_memory=None, available_disk=None):
    """RAM plan for the music engine: its three weight sets, all resident, as the engine holds them.

    Working buffers grow with the length of the clip (the DiT's attention over the clip's tokens, the VAE's
    full-resolution feature maps in windows of 256 frames) and are not in these numbers."""
    inventory = inspect_music_model(model_dir)
    comps = inventory["components"]
    resident = sum(comps[name]["resident_bytes"] for name in COMPONENTS)
    warnings = []
    for name in COMPONENTS:
        if not comps[name]["weight_files"]:
            warnings.append(f"{name}/ has no .safetensors weights")
    if not inventory["tokenizer"]:
        warnings.append(f"{TOKENIZER_FILE} is missing")
    if not inventory["silence_latent"]:
        warnings.append(f"{SILENCE_FILE} is missing")
    on_demand = sum(comps[name]["on_demand_bytes"] for name in COMPONENTS)
    mode = {"peak_bytes": resident, "idle_bytes": resident, "on_demand_bytes": on_demand,
            "note": "every weight stays loaded between songs; audio in adds the on-demand part the first time it is used"}
    if available_memory:
        mode["headroom_bytes"] = available_memory - resident
        mode["fits"] = available_memory > resident
        if not mode["fits"]:
            warnings.append("the weights do not fit in the available RAM")
    return {"version": 1, "kind": "audio", "family_id": "ace_step",
            "model": {"path": inventory["path"], "disk_bytes": inventory["disk_bytes"],
                      "planner_unused_bytes": inventory["planner_bytes"]},
            "components": [comps[name] for name in COMPONENTS],
            "modes": {"resident": mode},
            "memory": {"available_bytes": available_memory},
            "disk": {"available_bytes": available_disk},
            "working_buffers": "not included (they grow with the length of the clip)",
            "warnings": warnings}


def _gb(value):
    return f"{value / 1e9:.1f} GB"


def format_music_plan(plan):
    names = {TEXT_DIR: "text encoder", DIT_DIR: "DiT", VAE_DIR: "VAE decoder"}
    lines = [f"model  ACE-Step 1.5 turbo · text-to-music, covers, repaints · {_gb(plan['model']['disk_bytes'])} on disk"]
    lines.append(f"{'':15}{'on disk':>18}{'in RAM':>18}")
    for comp in plan["components"]:
        disk = f"{_gb(comp['disk_bytes'])} {'/'.join(d.lower() for d in comp['dtypes']) or '?'}"
        lines.append(f"{names[comp['name']]:15}{disk:>18}{_gb(comp['resident_bytes']):>18}")
        if comp.get("on_demand_params"):
            what = "the encoder" if comp["name"] == VAE_DIR else "the audio tokenizer"
            lines.append(f"{'':15}{names[comp['name']]}: {comp['on_demand_params'] / 1e6:.0f}M parameters ({what}) "
                         f"load on first use of audio in, {_gb(comp['on_demand_bytes'])} in RAM")
    if plan["model"]["planner_unused_bytes"]:
        lines.append(f"{'':15}the 5 Hz language-model planner ({_gb(plan['model']['planner_unused_bytes'])}) is on disk and not used")
    lines.append("")
    entry = plan["modes"]["resident"]
    row = f"{'all resident':24}{_gb(entry['peak_bytes']):>9} peak"
    available = plan["memory"]["available_bytes"]
    if available:
        left = entry["headroom_bytes"]
        row += (f" · leaves {_gb(left)} of {_gb(available)} available" if left > 0
                else f" · {_gb(-left)} MORE than the {_gb(available)} available")
    lines.append(row)
    lines.append(f"{'':24}{entry['note']}")
    if entry.get("on_demand_bytes"):
        lines.append(f"{'with audio in':24}{_gb(entry['peak_bytes'] + entry['on_demand_bytes']):>9} once a clip has been used")
    lines.append(f"working buffers: {plan['working_buffers']}")
    disk = plan["disk"]["available_bytes"]
    if disk is not None:
        lines.append(f"disk   {_gb(disk)} free · generated songs go to {music_dir()}")
    lines.extend(f"warn   {warning}" for warning in plan["warnings"])
    return "\n".join(lines)
