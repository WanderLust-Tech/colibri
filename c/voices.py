#!/usr/bin/env python3
"""Saved voices for the voice-cloning speech models (Qwen3-TTS Base): a name, a few seconds of reference speech, and what it says.

    ~/colibri-voices/<name>/ref.wav        the clip, 24 kHz mono 16-bit (what the engine reads)
    ~/colibri-voices/<name>/voice.json     {"name", "text", "seconds", "created"}

`COLI_VOICES` names another folder. A request says `"voice": "<name>"` and the gateway hands the engine that clip and text; `POST /v1/audio/voices`,
`coli voice add` and the dashboard make them. With the text the voice is cloned in context (the model reads the clip's transcript and carries on
from it: the closer likeness); without it only the speaker embedding is used. Stdlib, plus numpy for the audio conversion (gemma4_audio).
"""
import io
import json
import os
import re
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path

SAMPLE_RATE = 24000
NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
MIN_SECONDS = 1.0                       # the engine's floor is half a second; under a second is not a voice
MAX_SECONDS = 60.0                      # the reference's frames ride in every request's prompt: 12.5 frames a second
MAX_TEXT_CHARS = 2000


class VoiceError(ValueError):
    """A voice that cannot be made or found; `param` names the request field to blame."""

    def __init__(self, message, param=None, status=400):
        super().__init__(message)
        self.param = param
        self.status = status


def directory():
    return Path(os.environ.get("COLI_VOICES") or Path.home() / "colibri-voices")


def check_name(name):
    if not isinstance(name, str) or not NAME_RE.match(name.strip().lower()):
        raise VoiceError("`name` must be 1 to 64 characters: lower-case letters, digits, '-' and '_', starting with a letter or digit.", "name")
    return name.strip().lower()


def prepare_reference(audio):
    """The bytes of a 24 kHz mono 16-bit WAV (the engine's reference) and its length in seconds, from the bytes of any audio file the server
    can read (WAV, MP3, and with ffmpeg more). VoiceError for audio that is unreadable, too short or too long."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
        import gemma4_audio
        import numpy as np
    except ImportError as problem:
        raise VoiceError(f"audio support needs numpy ({problem}).", "audio")
    try:
        samples, rate = gemma4_audio.read_audio(bytes(audio))
        samples = gemma4_audio.resample(samples, rate, SAMPLE_RATE)
    except ValueError as problem:
        raise VoiceError(str(problem), "audio")
    seconds = len(samples) / SAMPLE_RATE
    if seconds < MIN_SECONDS:
        raise VoiceError(f"the reference is {seconds:.1f} s long; it needs at least {MIN_SECONDS:.0f} s of speech (3 to 30 s is best).", "audio")
    if seconds > MAX_SECONDS:
        raise VoiceError(f"the reference is {seconds:.0f} s long; the limit is {MAX_SECONDS:.0f} s (3 to 30 s is best).", "audio")
    x = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return out.getvalue(), seconds


def _info(folder):
    try:
        meta = json.loads((folder / "voice.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not (folder / "ref.wav").is_file():
        return None
    return {"name": folder.name, "text": str(meta.get("text") or ""), "seconds": float(meta.get("seconds") or 0.0),
            "created": int(meta.get("created") or 0), "path": str(folder / "ref.wav")}


def list_voices():
    """Every saved voice, by name."""
    root = directory()
    if not root.is_dir():
        return []
    found = [_info(folder) for folder in sorted(root.iterdir()) if folder.is_dir() and NAME_RE.match(folder.name)]
    return [voice for voice in found if voice]


def get(name):
    """The saved voice called `name` ({"name", "text", "seconds", "created", "path"}), or None."""
    if not isinstance(name, str) or not NAME_RE.match(name.strip().lower()):
        return None
    return _info(directory() / name.strip().lower())


def add(name, audio, text=""):
    """Save a voice (replacing one of the same name) from the bytes of an audio file and what it says (may be empty: the speaker embedding
    alone is used). Returns the voice's info, with `replaced`."""
    name = check_name(name)
    text = (text or "").strip()
    if len(text) > MAX_TEXT_CHARS:
        raise VoiceError(f"`text` is {len(text)} characters; the limit is {MAX_TEXT_CHARS}.", "text")
    wav, seconds = prepare_reference(audio)
    folder = directory() / name
    replaced = folder.is_dir()
    folder.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=folder, suffix=".part")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(wav)
        os.replace(tmp, folder / "ref.wav")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    meta = {"name": name, "text": text, "seconds": round(seconds, 3), "created": int(time.time())}
    (folder / "voice.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    info = _info(folder)
    info["replaced"] = replaced
    return info


def delete(name):
    """Remove a saved voice; whether there was one."""
    voice = get(name)
    if voice is None:
        return False
    folder = Path(voice["path"]).parent
    for item in folder.iterdir():
        try:
            item.unlink()
        except OSError:
            pass
    try:
        folder.rmdir()
    except OSError:
        return False
    return True


def payload(voice):
    """The JSON shape of a voice for the API (no filesystem path)."""
    return {"name": voice["name"], "text": voice["text"], "seconds": voice["seconds"], "created": voice["created"],
            "mode": "in-context" if voice["text"] else "x-vector"}
