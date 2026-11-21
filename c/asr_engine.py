#!/usr/bin/env python3
"""The speech-recognition engine from Python: its serve protocol, and the request rules of POST /v1/audio/transcriptions for it.

Used by the gateway (openai_server.py). Stdlib only. The engine is c/qwen3asr (docs/qwen3-asr.md; Qwen3-ASR-0.6B and -1.7B); the
protocol is the header of c/qwen3asr.c, the retrieval engines' shape (lines of a keyword and a JSON object):

    READY {...}                      once, after the weights are loaded: model, task "transcribe", sample_rate, piece_seconds, layers
    TRANSCRIBE {...}                 client, one request at a time: id, wav (the path of a 16 kHz WAV file), language, context, max_new
    TEXT {...}                       id, language (what the model heard, "" when it did not say), text, tokens, pieces, audio_seconds, seconds, truncated
    ERROR {...}                      the engine stays alive for the next TRANSCRIBE

The audio reaches the engine as a file on this machine: the gateway has already decoded the upload to 16 kHz mono (WAV, MP3, and
with ffmpeg more), and the engine runs on loopback beside it.
"""
import os
import re
import tempfile

from embedding_engine import EmbeddingEngine, EmbeddingEngineError, EmbeddingRequestError

DEFAULT_INFO = {"model": "asr", "task": "transcribe", "sample_rate": 16000, "piece_seconds": 1200}
FORMATS = ("json", "text", "verbose_json")
MAX_CONTEXT_CHARS = 1000

# The 30 languages Qwen3-ASR names (its config's support_languages), by the name the prompt uses.
LANGUAGES = ("Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese", "Indonesian", "Italian",
             "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish",
             "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian")

# ISO-639 codes (what OpenAI's `language` field takes) to those names. Codes the model has no name for are refused.
ISO_NAMES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
    "pt": "Portuguese", "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
    "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "pl": "Polish", "cs": "Czech", "tl": "Filipino", "fil": "Filipino", "fa": "Persian", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian"}


class TranscriptionEngine(EmbeddingEngine):
    """The retrieval engines' process wrapper over a Qwen3-ASR checkpoint: TRANSCRIBE in, TEXT out."""

    modality = "transcription"
    label = "speech recognition engine"
    log_tag = "asr"
    default_info = DEFAULT_INFO

    def transcribe(self, wav_path, language=None, context="", max_new=None):
        """Run one TRANSCRIBE on the WAV file at `wav_path` and return {"text", "language", "tokens", "pieces", "audio_seconds",
        "seconds", "truncated"}. `language` is a name from LANGUAGES (forcing it) or None (the model says what it hears)."""
        request = {"wav": str(wav_path)}
        if language:
            request["language"] = language
        if context:
            request["context"] = context
        if max_new:
            request["max_new"] = int(max_new)
        body = self._exchange("TRANSCRIBE", "TEXT", request)
        if not isinstance(body.get("text"), str):
            raise EmbeddingEngineError(f"the {self.label} sent a malformed TEXT line")
        return {"text": body["text"].strip(), "language": str(body.get("language") or ""), "tokens": int(body.get("tokens") or 0),
                "pieces": int(body.get("pieces") or 1), "audio_seconds": float(body.get("audio_seconds") or 0.0),
                "seconds": float(body.get("seconds") or 0.0), "truncated": bool(body.get("truncated"))}

    def transcribe_samples(self, wav_bytes, **fields):
        """Writes `wav_bytes` (a 16 kHz mono WAV file) to a temporary file for the engine and transcribes it."""
        handle, path = tempfile.mkstemp(prefix="coli-asr-", suffix=".wav")
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(wav_bytes)
            return self.transcribe(path, **fields)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def transcription_request(body):
    """The checked fields of POST /v1/audio/transcriptions for a speech-recognition model: {"format", "language" (a name or None),
    "language_code", "context"}, or EmbeddingRequestError (the gateway answers 400 with its `param`).

    `language` is an ISO-639 code ("en"): it forces that language (the model then writes only the transcription); without it the
    model detects the language and says which. `prompt` is context for the model (names, terms), at most 1000 characters; it goes
    into the system turn. `temperature` is accepted and ignored: the model is decoded greedily. srt, vtt and word timestamps are
    refused: Qwen3-ASR gives none (its forced aligner, a separate model, does)."""
    if not isinstance(body, dict):
        raise EmbeddingRequestError("The request body must be a form or a JSON object.")
    fmt = body.get("response_format", "json")
    if fmt in ("srt", "vtt"):
        raise EmbeddingRequestError(f"`response_format` {fmt!r} needs timestamps, which this model does not give: use json, text or verbose_json.",
                                    "response_format")
    if fmt not in FORMATS:
        raise EmbeddingRequestError(f"`response_format` must be one of {', '.join(FORMATS)}.", "response_format")
    grains = body.get("timestamp_granularities", body.get("timestamp_granularities[]"))
    if "word" in (grains if isinstance(grains, list) else [grains]):
        raise EmbeddingRequestError("Word timestamps are not available: the model does not give them.", "timestamp_granularities")
    if str(body.get("stream", False)).lower() in ("true", "1"):
        raise EmbeddingRequestError("`stream` is not supported on transcriptions.", "stream")
    code = body.get("language")
    name = None
    if code not in (None, ""):
        if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z]{2,3}(-[A-Za-z]{2,4})?", code):
            raise EmbeddingRequestError("`language` must be an ISO-639-1 code such as \"en\".", "language")
        name = ISO_NAMES.get(code.lower().split("-")[0])
        if name is None:
            raise EmbeddingRequestError(f"`language` {code!r} is not one this model transcribes; it takes "
                                        f"{', '.join(sorted(ISO_NAMES))} (or leave it out and the model detects the language).", "language")
    context = body.get("prompt")
    if context is not None and (not isinstance(context, str) or len(context) > MAX_CONTEXT_CHARS):
        raise EmbeddingRequestError(f"`prompt` must be text of at most {MAX_CONTEXT_CHARS} characters.", "prompt")
    temperature = body.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 1:
        raise EmbeddingRequestError("`temperature` must be a number from 0 to 1.", "temperature")
    return {"format": fmt, "language": name, "language_code": code or None, "context": (context or "").strip()}


def transcription_payload(request, result, duration, model_id):
    """OpenAI's answer for the requested format: json {"text", "usage"}, verbose_json (task, language, duration, text, one segment),
    or None for `text` (the caller sends the plain text). The language is lower case, as OpenAI writes it ("english")."""
    usage = {"type": "duration", "seconds": int(round(duration))}
    text = result["text"]
    if request["format"] == "verbose_json":
        payload = {"task": "transcribe", "duration": round(duration, 3), "text": text, "usage": usage}
        language = (request["language"] or result.get("language") or "").lower()
        if language:
            payload["language"] = language
        payload["segments"] = [{"id": 0, "seek": 0, "start": 0.0, "end": round(duration, 3), "text": text}] if text else []
        return payload
    return {"text": text, "usage": usage}
