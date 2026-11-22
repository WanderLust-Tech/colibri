"""`coli talk` (docs/voice-loop.md): a spoken conversation with a running server's models, from the terminal.

It records from the microphone, stops when you stop speaking, has the speech recognition model write it down, gives the words to a chat model,
and speaks the answer with the speech model as the text is written (voice_loop.py). The microphone and the loudspeaker are reached through ffmpeg
and ffplay (no Python audio library): `ffmpeg` reads the capture device, `ffplay` plays the PCM it is fed. numpy does the arithmetic of the
voice-activity detector. `--file` runs the same turns from audio files, and `--text` from typed words, for a machine with no microphone.
"""
import array
import io
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import wave

import retrieval_cli
import speech_cli
import voice_loop
from retrieval_cli import Refused, choose_model

RATE = 16000                    # what the microphone is read at (the speech recognition model's rate)
FRAME = RATE // 50              # 20 ms
OUT_RATE = 24000                # the speech model's


# ------------------------------------------------------------------------------------------------ voice activity

class Vad:
    """Cuts a stream of 16 kHz mono 16-bit audio into utterances by loudness.

    The noise floor is learned from the first `calibrate` seconds (and follows the quiet stretches after); speech is a run of `attack` frames
    above 3 times the floor (and above `threshold`, an absolute RMS in 0..1); an utterance ends after `silence` seconds below it. `feed(frame)`
    returns the utterance (the bytes, `preroll` seconds of lead-in included) when one has just ended, else None; `reset()` forgets what is held."""

    def __init__(self, threshold=0.012, silence=0.8, attack=3, preroll=0.3, calibrate=0.4, max_seconds=30.0, min_seconds=0.4):
        self.threshold, self.silence_frames, self.attack = threshold, int(silence * 50), attack
        self.preroll_frames, self.calibrate_frames = int(preroll * 50), int(calibrate * 50)
        self.max_frames, self.min_frames = int(max_seconds * 50), int(min_seconds * 50)
        self.floor = None
        self.seen = 0
        self.reset()

    def reset(self):
        self.recent = []                 # the last frames before speech, kept as lead-in
        self.frames = []                 # the utterance so far
        self.loud = 0
        self.quiet = 0
        self.speaking = False

    @staticmethod
    def level(frame):
        samples = array.array("h")
        samples.frombytes(frame[:len(frame) // 2 * 2])
        if not samples:
            return 0.0
        return (sum(s * s for s in samples) / len(samples)) ** 0.5 / 32768.0

    def feed(self, frame):
        level = self.level(frame)
        self.seen += 1
        if self.floor is None:
            self.floor = level
        if not self.speaking and self.seen <= self.calibrate_frames:
            self.floor = min(self.floor, level) if self.seen > 1 else level
            self.recent = (self.recent + [frame])[-self.preroll_frames:]
            return None
        gate = max(self.threshold, 3.0 * self.floor)
        if not self.speaking:
            self.recent = (self.recent + [frame])[-self.preroll_frames - self.attack:]
            self.loud = self.loud + 1 if level > gate else 0
            if level <= gate:
                self.floor = 0.95 * self.floor + 0.05 * level                       # the floor follows the room
            if self.loud >= self.attack:
                self.speaking = True
                self.frames = list(self.recent)
                self.quiet = 0
            return None
        self.frames.append(frame)
        self.quiet = self.quiet + 1 if level <= gate else 0
        if self.quiet >= self.silence_frames or len(self.frames) >= self.max_frames:
            body = self.frames[:len(self.frames) - self.quiet + min(self.quiet, 10)]       # keep a short tail of the silence
            voiced = len(body) - min(self.quiet, 10)
            self.reset()
            return b"".join(body) if voiced >= self.min_frames else None
        return None


def wav_bytes(pcm, rate=RATE):
    out = io.BytesIO()
    with wave.open(out, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(pcm)
    return out.getvalue()


# ------------------------------------------------------------------------------------------------ the microphone and the loudspeaker

def capture_args(device=None):
    """ffmpeg's arguments to read a capture device as 16 kHz mono 16-bit PCM on stdout, for this platform."""
    system = platform.system()
    if system == "Windows":
        source = ["-f", "dshow", "-i", f"audio={device or first_dshow_microphone()}"]
    elif system == "Darwin":
        source = ["-f", "avfoundation", "-i", f":{device or '0'}"]
    else:
        source = ["-f", "pulse", "-i", device or "default"]
    return source + ["-ac", "1", "-ar", str(RATE), "-f", "s16le", "-loglevel", "error", "pipe:1"]


def dshow_microphones(ffmpeg):
    run = subprocess.run([ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"], capture_output=True, text=True, errors="replace")
    names = []
    for line in run.stderr.splitlines():
        if "(audio)" in line and '"' in line:
            names.append(line.split('"')[1])
    return names


def first_dshow_microphone():
    ffmpeg = shutil.which("ffmpeg")
    names = dshow_microphones(ffmpeg) if ffmpeg else []
    if not names:
        raise Refused("no microphone found (ffmpeg lists no audio capture device): plug one in, or use --file / --text")
    return names[0]


class Microphone:
    """ffmpeg reading the capture device, a thread putting each 20 ms frame, stamped with when it was read, on a queue."""

    def __init__(self, device=None):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise Refused("ffmpeg is needed to reach the microphone (winget install Gyan.FFmpeg); or use --file / --text")
        self.process = subprocess.Popen([ffmpeg, "-hide_banner", *capture_args(device)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = queue.Queue()
        self.thread = threading.Thread(target=self.pump, daemon=True)
        self.thread.start()

    def pump(self):
        size = FRAME * 2
        while True:
            data = self.process.stdout.read(size)
            if len(data) < size:
                self.frames.put((time.monotonic(), None))
                return
            self.frames.put((time.monotonic(), data))

    def read(self, not_before=0.0, timeout=1.0):
        """The next frame read at or after `not_before` (older ones are dropped), None for none yet; raises when the microphone is gone."""
        while True:
            try:
                stamp, data = self.frames.get(timeout=timeout)
            except queue.Empty:
                return None
            if data is None:
                err = self.process.stderr.read().decode("utf-8", "replace").strip()[-300:]
                raise Refused(f"the microphone stopped ({err or 'ffmpeg exited'})")
            if stamp >= not_before:
                return data

    def close(self):
        self.process.kill()


class Player:
    """ffplay playing the PCM it is fed, as it is fed."""

    def __init__(self):
        ffplay = shutil.which("ffplay")
        if not ffplay:
            raise Refused("ffplay is needed to play the answer (it comes with ffmpeg); or use --out DIR to save it")
        self.process = subprocess.Popen([ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", "-f", "s16le", "-ar", str(OUT_RATE), "-ac", "1", "-i", "pipe:0"],
                                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, pcm):
        try:
            self.process.stdin.write(pcm)
            self.process.stdin.flush()
        except (OSError, ValueError):
            pass

    def finish(self):
        """Wait until everything fed has been played."""
        try:
            self.process.stdin.close()
        except (OSError, ValueError):
            pass
        self.process.wait()


# ------------------------------------------------------------------------------------------------ the command

def pick_models(a, api, C):
    """(chat, asr, tts) model ids: those named, else what the server has (a fast chat model first)."""
    chat = a.model
    if not chat:
        status, body = api("GET", "/v1/models")
        if status is None:
            raise Refused("nothing answers on that host and port: start the server (coli service start, or coli router)")
        if status == 401:
            raise Refused("the server wants an API key: --api-key, or COLI_API_KEY")
        chats = [item["id"] for item in (body or {}).get("data") or [] if "chat" in (item.get("capabilities") or [])]
        if not chats:
            raise Refused("the server has no chat model: coli model install gemma-4-e2b")
        chat = next((m for m in chats if "e2b" in m), next((m for m in chats if "e4b" in m), chats[0]))
    asr = choose_model(api, "transcription", a.asr, "turns speech into text")
    saved = False
    if a.voice and not a.tts:
        status, body = api("GET", "/v1/audio/voices")
        saved = status == 200 and any(v.get("name") == a.voice.strip().lower() for v in (body or {}).get("data") or [])
    from types import SimpleNamespace
    tts = a.tts or (speech_cli.clone_model(api, SimpleNamespace(model=None)) if saved else choose_model(api, "speech", None, "speaks text"))
    return chat, asr, tts


def save_wav(path, pcm):
    with wave.open(path, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(OUT_RATE)
        clip.writeframes(pcm)


def talk(a, api, C):
    """coli talk [--file CLIP ...] [--text WORDS] [--model CHAT] [--asr ASR] [--tts TTS] [--voice NAME] [--system PROMPT] [--out DIR] [--no-play]"""
    try:
        chat, asr, tts = pick_models(a, api, C)
    except Refused as why:
        print(f"  {C.yel}{why}{C.r}")
        return 1
    server = voice_loop.Server(f"http://{a.host}:{a.port or 8000}", a.api_key)
    loop = voice_loop.VoiceLoop(server, chat, asr, tts, voice=a.voice, language=a.language, system=a.system or voice_loop.DEFAULT_SYSTEM,
                                chat_options={"max_tokens": a.max_tokens})
    print(f"  {C.dim}chat {chat} · ears {asr} · voice {tts}{' (' + a.voice + ')' if a.voice else ''}{C.r}")
    files = list(a.files or [])
    scripted = bool(files or a.text)
    play = not a.no_play and not a.out
    try:
        player_holder = []
        pcm = bytearray()
        said = []

        def emit(kind, value):
            if kind == "heard":
                print(f"  {C.dim}you:{C.r} {value or '(nothing heard)'}")
                print(f"  {C.dim}colibri:{C.r} ", end="", flush=True)
            elif kind == "text":
                print(value, end="", flush=True)
            elif kind == "audio":
                pcm.extend(value)
                if play:
                    if not player_holder:
                        player_holder.append(Player())
                    player_holder[0].write(value)
            elif kind == "reply":
                print()
            elif kind == "timing":
                parts = [f"{name.replace('_', ' ')} {value[name]:.2f}s" for name in ("heard", "first_text", "first_audio", "done") if name in value]
                print(f"  {C.dim}{' · '.join(parts)}{C.r}")

        def one(audio=None, text=None):
            pcm.clear()
            player_holder.clear()
            try:
                result = loop.turn(audio=audio, text=text, emit=emit)
            except voice_loop.VoiceError as error:
                print(f"\n  {C.yel}{error.step} failed: {error.message}{C.r}")
                return None
            finally:
                if player_holder:
                    player_holder[0].finish()
            return result

        if scripted:
            if a.out:
                os.makedirs(a.out, exist_ok=True)
            turns = [("file", f) for f in files] + ([("text", a.text)] if a.text else [])
            for number, (kind, item) in enumerate(turns, 1):
                if kind == "file":
                    try:
                        result = one(audio=speech_cli.read_audio(item))
                    except Refused as why:
                        print(f"  {C.yel}{why}{C.r}")
                        return 1
                else:
                    result = one(text=item)
                if result is None:
                    return 1
                if a.out and pcm:
                    path = os.path.join(a.out, f"reply-{number}.wav")
                    save_wav(path, bytes(pcm))
                    print(f"  {C.dim}saved {path}{C.r}")
            return 0
        return listen_loop(a, loop, one, C)
    except KeyboardInterrupt:
        print()
        return 0


def listen_loop(a, loop, one, C):
    try:
        mic = Microphone(a.mic)
    except Refused as why:
        print(f"  {C.yel}{why}{C.r}")
        return 1
    vad = Vad(threshold=a.threshold, silence=a.silence)
    print(f"  {C.dim}listening: speak, then pause; Ctrl-C to stop{C.r}")
    not_before = 0.0
    try:
        while True:
            try:
                frame = mic.read(not_before)
            except Refused as why:
                print(f"  {C.yel}{why}{C.r}")
                return 1
            if frame is None:
                continue
            utterance = vad.feed(frame)
            if utterance is None:
                continue
            one(audio=wav_bytes(utterance))
            vad.reset()
            not_before = time.monotonic() + 0.3                  # what the microphone heard while the answer played is dropped
            print(f"  {C.dim}listening…{C.r}")
    finally:
        mic.close()


def list_mics(C):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(f"  {C.yel}ffmpeg is not installed (winget install Gyan.FFmpeg){C.r}")
        return 1
    if platform.system() != "Windows":
        print("  " + {"Darwin": "ffmpeg -f avfoundation -list_devices true -i ''"}.get(platform.system(), "pactl list short sources") + "  lists them; pass the name or index with --mic")
        return 0
    names = dshow_microphones(ffmpeg)
    if not names:
        print(f"  {C.yel}no audio capture device found{C.r}")
        return 1
    for name in names:
        print(f"  {name}")
    return 0
