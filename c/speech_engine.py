#!/usr/bin/env python3
"""The speech-synthesis engine from Python: its serve protocol, and the request rules of POST /v1/audio/speech for it.

Used by the gateway (openai_server.py). Stdlib only. The engine is c/qwen3tts (docs/qwen3-tts.md; Qwen3-TTS-12Hz CustomVoice); the
protocol is the header of c/qwen3tts.c, the other engines' shape (lines of a keyword and a JSON object):

    READY {...}                      once, after the weights are loaded: model, task "speech", kind, sample_rate, speakers, languages
    SPEAK {...}                      client, one request at a time: id, text, speaker, language, out (the path of the WAV file to write),
                                     seed, greedy, max_frames, temperature; instruct (a style, or VoiceDesign's description of the voice);
                                     a Base (cloning) model: ref_wav (a 24 kHz mono WAV on this machine), ref_text, x_vector_only
    AUDIO {...}                      id, path, samples, sample_rate, frames, audio_seconds, seconds, truncated
    ERROR {...}                      the engine stays alive for the next SPEAK

The sound reaches the gateway as a file on this machine (the engine runs on loopback beside it); the gateway turns the WAV into the
format the client asked for.
"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from embedding_engine import EmbeddingEngine, EmbeddingEngineError, EmbeddingRequestError

DEFAULT_INFO = {"model": "speech", "task": "speech", "sample_rate": 24000, "speakers": [], "languages": [], "clone": False}
MAX_REFERENCE_B64 = 6 << 20                 # an inline reference_audio, as base64 (the body limit is 4 MB anyway)
MAX_INPUT_CHARS = 4096                       # OpenAI's own limit on `input`
FORMATS = ("wav", "pcm", "mp3", "opus", "aac", "flac")
NEEDS_FFMPEG = ("mp3", "opus", "aac", "flac")
CONTENT_TYPES = {"wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac"}
DEFAULT_FORMAT = "wav"                       # OpenAI's default is mp3; a WAV needs nothing on the server to play
FFMPEG_ARGS = {"mp3": ["-f", "mp3", "-b:a", "128k"], "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"],
               "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"], "flac": ["-f", "flac"]}

# OpenAI's voice names, for the clients written against it: each stands for one of the model's speakers (docs/qwen3-tts.md). The
# model's own speaker names work as they are.
OPENAI_VOICES = {"alloy": "ryan", "echo": "aiden", "fable": "dylan", "onyx": "uncle_fu", "nova": "vivian", "shimmer": "serena",
                 "ash": "eric", "coral": "ono_anna", "sage": "sohee"}
DEFAULT_SPEAKER = "ryan"

# The names the model's prompt uses, by ISO-639 code (`language` also takes the name itself: "english").
ISO_LANGUAGES = {"zh": "chinese", "en": "english", "de": "german", "it": "italian", "pt": "portuguese", "es": "spanish",
                 "ja": "japanese", "ko": "korean", "fr": "french", "ru": "russian"}


class SpeechEngine(EmbeddingEngine):
    """The retrieval engines' process wrapper over a Qwen3-TTS checkpoint: SPEAK in, AUDIO out."""

    modality = "speech"
    label = "speech synthesis engine"
    log_tag = "tts"
    default_info = DEFAULT_INFO

    def speak(self, text, speaker="", language="auto", seed=None, greedy=False, max_frames=None, temperature=None, instruct=None, reference=None):
        """Run one SPEAK and return {"wav": the WAV file's bytes, "sample_rate", "frames", "audio_seconds", "seconds", "truncated"}.
        `reference` (a Base model): {"wav": the bytes of a 24 kHz mono WAV, or "path": one on disk, "text": what it says, "x_vector_only": bool}."""
        handle, path = tempfile.mkstemp(prefix="coli-tts-", suffix=".wav")
        os.close(handle)
        ref_path = None
        try:
            request = {"text": text, "speaker": speaker, "language": language, "out": path.replace("\\", "/")}
            if instruct:
                request["instruct"] = instruct
            if reference:
                if reference.get("path"):
                    ref_path = None
                    request["ref_wav"] = str(reference["path"]).replace("\\", "/")
                else:
                    rhandle, ref_path = tempfile.mkstemp(prefix="coli-ref-", suffix=".wav")
                    with os.fdopen(rhandle, "wb") as out:
                        out.write(reference["wav"])
                    request["ref_wav"] = ref_path.replace("\\", "/")
                if reference.get("text") and not reference.get("x_vector_only"):
                    request["ref_text"] = reference["text"]
                if reference.get("x_vector_only"):
                    request["x_vector_only"] = True
            if seed:
                request["seed"] = int(seed)
            if greedy:
                request["greedy"] = True
            if max_frames:
                request["max_frames"] = int(max_frames)
            if temperature:
                request["temperature"] = float(temperature)
            body = self._exchange("SPEAK", "AUDIO", request)
            try:
                wav = Path(path).read_bytes()
            except OSError as error:
                raise EmbeddingEngineError(f"the {self.label} said it wrote the audio, but {error}") from error
            if len(wav) < 44:
                raise EmbeddingEngineError(f"the {self.label} wrote an empty audio file")
            return {"wav": wav, "sample_rate": int(body.get("sample_rate") or 24000), "frames": int(body.get("frames") or 0),
                    "audio_seconds": float(body.get("audio_seconds") or 0.0), "seconds": float(body.get("seconds") or 0.0),
                    "truncated": bool(body.get("truncated"))}
        finally:
            for leftover in (path, ref_path):
                if leftover:
                    try:
                        os.unlink(leftover)
                    except OSError:
                        pass


def speech_request(body, info=None):
    """The checked fields of POST /v1/audio/speech: {"input", "speaker", "language", "format", "seed", "greedy", "instruct"}, or
    EmbeddingRequestError (the gateway answers 400 with its `param`).

    `input` is the text (up to 4096 characters). `voice` is a speaker of the model (the READY line lists them) or one of OpenAI's names
    (alloy, echo, ...), which stand for speakers; left out it is Ryan. `language` is an extension: a name or ISO code, "auto" (the
    default) lets the model decide. `response_format` is wav (the default), pcm (raw 16-bit little-endian mono, 24 kHz), and mp3, opus,
    aac and flac when the server has ffmpeg. `speed` other than 1 is refused (the model has no rate control), and so is `instructions`
    for a model that does not take them (the 1.7B CustomVoice takes a style, VoiceDesign the description of the voice: both go to the engine
    as `instruct`; VoiceDesign has no speakers, so `voice` names no one there). `seed` is an extension (a given seed repeats the same speech).

    A Base model (the READY line says `clone`) speaks in a cloned voice: `voice` names a saved voice (voices.py; `POST /v1/audio/voices`), or the
    request carries its own reference: `reference_audio` (base64 of an audio file, or a data: URL) with `reference_text`, what the clip says
    (without it only the speaker's embedding is used: a looser likeness); `x_vector_only` asks for that even with the text. The checked
    fields come back as `voice` (the saved name, or None), `reference` (the audio's bytes, or None), `reference_text`, `x_vector_only`."""
    info = info or {}
    if not isinstance(body, dict):
        raise EmbeddingRequestError("The request body must be a JSON object.")
    text = body.get("input")
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingRequestError("`input` is required: the text to speak.", "input")
    if len(text) > MAX_INPUT_CHARS:
        raise EmbeddingRequestError(f"`input` is {len(text)} characters; the limit is {MAX_INPUT_CHARS}.", "input")
    fmt = body.get("response_format", DEFAULT_FORMAT)
    if fmt not in FORMATS:
        raise EmbeddingRequestError(f"`response_format` must be one of {', '.join(FORMATS)}.", "response_format")
    speed = body.get("speed", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0.25 <= speed <= 4.0:
        raise EmbeddingRequestError("`speed` must be a number from 0.25 to 4.0.", "speed")
    if speed != 1:
        raise EmbeddingRequestError("`speed` other than 1 is not supported: the model has no control over the pace.", "speed")
    instructions = body.get("instructions")
    if instructions not in (None, ""):
        if not isinstance(instructions, str):
            raise EmbeddingRequestError("`instructions` must be text.", "instructions")
        if not info.get("instruct"):
            raise EmbeddingRequestError("`instructions` is not supported by this model: it speaks in the voice of a named speaker "
                                        "and takes no style description.", "instructions")
    clone = bool(info.get("clone"))
    ref_audio = body.get("reference_audio")
    ref_text = body.get("reference_text")
    x_vector_only = body.get("x_vector_only", False)
    reference = None
    if ref_audio not in (None, ""):
        if not clone:
            raise EmbeddingRequestError("`reference_audio` is for a voice-cloning model (a Qwen3-TTS Base checkpoint): this one speaks in its own voices.",
                                        "reference_audio")
        if not isinstance(ref_audio, str) or len(ref_audio) > MAX_REFERENCE_B64:
            raise EmbeddingRequestError("`reference_audio` must be the base64 of an audio file (or a data: URL), at most a few megabytes.", "reference_audio")
        import base64
        data = ref_audio.split(",", 1)[1] if ref_audio.startswith("data:") and "," in ref_audio else ref_audio
        try:
            reference = base64.b64decode(data, validate=True)
        except Exception:
            raise EmbeddingRequestError("`reference_audio` is not valid base64.", "reference_audio")
        if not reference:
            raise EmbeddingRequestError("`reference_audio` is empty.", "reference_audio")
    if ref_text is not None and (not isinstance(ref_text, str) or len(ref_text) > 2000):
        raise EmbeddingRequestError("`reference_text` must be text of at most 2000 characters: what the reference audio says.", "reference_text")
    if not isinstance(x_vector_only, bool):
        raise EmbeddingRequestError("`x_vector_only` must be true or false.", "x_vector_only")
    stream_format = body.get("stream_format")
    if stream_format not in (None, "", "audio"):
        raise EmbeddingRequestError("`stream_format` \"sse\" is not supported: the audio comes in one piece.", "stream_format")
    speakers = [str(name).lower() for name in info.get("speakers") or ()]
    voice = body.get("voice", None if clone else DEFAULT_SPEAKER)
    if isinstance(voice, dict):
        voice = voice.get("id")
    if clone and voice in (None, ""):
        voice = ""
    elif not isinstance(voice, str) or not voice.strip():
        raise EmbeddingRequestError("`voice` must be the name of a " + ("saved voice." if clone else "speaker."), "voice")
    saved_name = voice.strip().lower()
    speaker = voice.strip().lower().replace(" ", "_").replace("-", "_")
    speaker = OPENAI_VOICES.get(speaker, speaker) if speaker not in speakers else speaker
    if speakers and speaker not in speakers:
        raise EmbeddingRequestError(f"`voice` {voice!r} is not a speaker of this model; it has {', '.join(speakers)} "
                                    f"(and OpenAI's {', '.join(sorted(OPENAI_VOICES))} stand for some of them).", "voice")
    language = body.get("language", "auto")
    if not isinstance(language, str) or not language.strip():
        raise EmbeddingRequestError("`language` must be a language name or an ISO-639-1 code, or \"auto\".", "language")
    language = language.strip().lower().replace(" ", "_")
    language = ISO_LANGUAGES.get(language.split("-")[0], language) if len(language.split("-")[0]) == 2 else language
    languages = [str(name).lower() for name in info.get("languages") or ()]
    if language != "auto" and languages and language not in languages:
        raise EmbeddingRequestError(f"`language` {body.get('language')!r} is not one this model speaks; it has {', '.join(languages)} "
                                    "(or \"auto\").", "language")
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise EmbeddingRequestError("`seed` must be a non-negative integer.", "seed")
    return {"input": text, "speaker": "" if clone else speaker, "language": language, "format": fmt, "seed": seed or None, "greedy": False,
            "instruct": instructions if isinstance(instructions, str) and instructions.strip() else None,
            "voice": saved_name if clone and saved_name else None, "reference": reference,
            "reference_text": ref_text.strip() if isinstance(ref_text, str) and ref_text.strip() else None, "x_vector_only": x_vector_only}


def wav_pcm(wav):
    """The raw samples (16-bit little-endian mono) inside a RIFF/WAVE file: the bytes of its `data` chunk."""
    if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    at = 12
    while at + 8 <= len(wav):
        name, size = wav[at:at + 4], struct.unpack_from("<I", wav, at + 4)[0]
        if name == b"data":
            return wav[at + 8:at + 8 + size]
        at += 8 + size + (size & 1)
    raise ValueError("the WAV file has no data chunk")


FFMPEG_MISSING = ("`response_format` {fmt!r} needs ffmpeg, and there is none on this server: ask for wav or pcm "
                  "(or put ffmpeg on the PATH, or name it with COLI_FFMPEG).")


def ffmpeg_path():
    """The ffmpeg this server would use (COLI_FFMPEG, else the PATH), or None."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    import gemma4_media
    return gemma4_media.ffmpeg_path()


def encode_audio(wav, fmt, ffmpeg=None):
    """The WAV file's bytes in the format asked for: wav as it is, pcm without the header, the rest through ffmpeg (`ffmpeg`, the path
    of one, or the one ffmpeg_path finds). Raises EmbeddingRequestError (param response_format) when ffmpeg is needed and missing."""
    if fmt == "wav":
        return wav
    if fmt == "pcm":
        return wav_pcm(wav)
    ffmpeg = ffmpeg or ffmpeg_path()
    if not ffmpeg:
        raise EmbeddingRequestError(FFMPEG_MISSING.format(fmt=fmt), "response_format")
    run = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0", *FFMPEG_ARGS[fmt], "pipe:1"],
                         input=wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if run.returncode != 0 or not run.stdout:
        raise EmbeddingEngineError(f"ffmpeg could not make {fmt}: {run.stderr.decode('utf-8', 'replace').strip()[-200:]}")
    return run.stdout
