#!/usr/bin/env python3
"""Forced alignment from Python: the words of a text, the engine's ALIGN exchange, and the request rules of POST /v1/audio/alignments.

Used by the gateway (openai_server.py). Stdlib only. The engine is c/qwen3asr started on a Qwen3-ForcedAligner checkpoint (docs/qwen3-aligner.md); its
protocol adds to the transcription engine's (asr_engine.py):

    READY {...}                      model, task "align", sample_rate, piece_seconds (the longest audio), layers
    ALIGN {...}                      client, one request at a time: id, wav (the path of a 16 kHz WAV file), words (the text, cut into units)
    ALIGNED {...}                    id, starts and ends (milliseconds, one for each word), audio_seconds, seconds
    ERROR {...}                      the engine stays alive for the next request

The text is cut into words here, as the reference package's `Qwen3ForceAlignProcessor` does it: a unit is a run of letters and digits (and apostrophes) between
anything else, and each CJK character is a unit of its own. The engine does the rest.
"""
import unicodedata

from asr_engine import ISO_NAMES, TranscriptionEngine
from embedding_engine import EmbeddingEngineError, EmbeddingRequestError

MAX_TEXT_CHARS = 20000
MAX_SECONDS = 300                       # the reference's limit on the audio of one alignment
# The languages the aligner names (its config's support_languages). The text is cut the same way for all but Japanese and Korean.
LANGUAGES = ("Chinese", "Cantonese", "English", "German", "Spanish", "French", "Italian", "Portuguese", "Russian", "Korean", "Japanese")
FORMATS = ("json", "verbose_json", "srt", "vtt", "text")


def is_kept_char(ch):
    if ch == "'":
        return True
    category = unicodedata.category(ch)
    return category.startswith("L") or category.startswith("N")


def clean_token(token):
    return "".join(ch for ch in token if is_kept_char(ch))


def is_cjk_char(ch):
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F or 0x2B820 <= code <= 0x2CEAF or 0xF900 <= code <= 0xFAFF)


def split_segment_with_chinese(segment):
    tokens, buffer = [], []
    for ch in segment:
        if is_cjk_char(ch):
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(ch)
        else:
            buffer.append(ch)
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def tokenize_space_language(text):
    tokens = []
    for segment in text.split():
        cleaned = clean_token(segment)
        if cleaned:
            tokens.extend(split_segment_with_chinese(cleaned))
    return tokens


def tokenize_japanese(text):
    """Words by nagisa when it is installed (what the reference does); without it, a unit for each character, which the model still places."""
    try:
        import nagisa
    except ImportError:
        return [clean for clean in (clean_token(ch) for ch in text if not ch.isspace()) if clean]
    return [clean for clean in (clean_token(word) for word in nagisa.tagging(text).words) if clean]


def split_words(text, language=None):
    """The units of `text` the aligner places, in order. Japanese goes through nagisa (a character each without it); Korean is cut at spaces (the
    reference uses a dictionary tokenizer that is not here); everything else as in the reference's space-language rule."""
    name = (language or "").lower()
    if name == "japanese":
        return tokenize_japanese(text)
    return tokenize_space_language(text)


HARD_END = ".!?\u2026\u3002\uff01\uff1f"          # . ! ? ... and their CJK forms: a sentence ends
SOFT_END = ",;:\u3001\uff0c\uff1b\uff1a"            # a comma, semicolon, colon: a clause ends
CLOSERS = "\"')]\u201d\u2019\u300d\u300f\uff09"


def split_words_marked(text, language=None):
    """(words, marks): `split_words`, and for each word 2 when a sentence ends after it, 1 when a clause does, else 0 (what the subtitles break on)."""
    name = (language or "").lower()
    words, marks = [], []
    if name == "japanese":
        words = tokenize_japanese(text)
        return words, [0] * len(words)
    for segment in text.split():
        cleaned = clean_token(segment)
        if not cleaned:
            if marks and (segment[-1:] in HARD_END or segment.rstrip(CLOSERS)[-1:] in HARD_END):
                marks[-1] = 2                         # a mark standing alone (" ... ") closes the word before it
            continue
        units = split_segment_with_chinese(cleaned)
        tail = segment.rstrip(CLOSERS)[-1:]
        words.extend(units)
        marks.extend([0] * (len(units) - 1) + [2 if tail in HARD_END else 1 if tail in SOFT_END else 0])
    return words, marks


def guess_language(text):
    """A language name for a text that did not say: Japanese when it has kana, Korean when it has hangul, else none (the generic rule)."""
    for ch in text:
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF:
            return "Japanese"
        if 0xAC00 <= code <= 0xD7AF:
            return "Korean"
    return None


class AlignmentEngine(TranscriptionEngine):
    """The process wrapper over a Qwen3-ForcedAligner checkpoint: ALIGN in, ALIGNED out. (It is the speech recognition engine's executable: the
    gateway tells it by the READY line's task.)"""

    label = "forced alignment engine"

    def align(self, wav_path, words):
        """Run one ALIGN of `words` against the WAV file at `wav_path`: {"starts", "ends" (seconds, one for each word), "audio_seconds", "seconds"}."""
        body = self._exchange("ALIGN", "ALIGNED", {"wav": str(wav_path), "words": list(words)})
        starts, ends = body.get("starts"), body.get("ends")
        if (not isinstance(starts, list) or not isinstance(ends, list) or len(starts) != len(words) or len(ends) != len(words)):
            raise EmbeddingEngineError(f"the {self.label} sent a malformed ALIGNED line")
        return {"starts": [round(int(ms) / 1000.0, 3) for ms in starts], "ends": [round(int(ms) / 1000.0, 3) for ms in ends],
                "audio_seconds": float(body.get("audio_seconds") or 0.0), "seconds": float(body.get("seconds") or 0.0)}

    def align_samples(self, wav_bytes, words):
        """Writes `wav_bytes` (a 16 kHz mono WAV file) to a temporary file for the engine and aligns `words` with it."""
        import os
        import tempfile
        handle, path = tempfile.mkstemp(prefix="coli-align-", suffix=".wav")
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(wav_bytes)
            return self.align(path, words)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def adopt(runtime):
    """A speech engine the gateway started as a recogniser, made an AlignmentEngine when its READY line says it is a forced aligner (both are the same
    executable and the same process wrapper: the launcher cannot tell until the engine has said so). Returns the runtime."""
    if (getattr(runtime, "info", None) or {}).get("task") == "align" and not isinstance(runtime, AlignmentEngine):
        runtime.__class__ = AlignmentEngine
    return runtime


def alignment_request(body):
    """The checked fields of POST /v1/audio/alignments: {"text", "language" (a name or None), "words", "format"}, or EmbeddingRequestError (the gateway
    answers 400 with its `param`).

    `text` is what is said in the audio (required; the aligner places its words, it does not listen for them). `language` is a name or an ISO-639 code of the
    aligner's languages; it only decides how the text is cut into words (Japanese and Korean differ). `response_format` is `json` (the default: the words with
    their times), `verbose_json` (adds the language and the duration), `text` (a line for each word: start, end, word), and `srt` or `vtt` (subtitles: the words
    gathered into lines; `max_chars` and `max_seconds` set how long a line may be)."""
    if not isinstance(body, dict):
        raise EmbeddingRequestError("The request body must be a form or a JSON object.")
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingRequestError("`text` is required: what is said in the audio.", "text")
    if len(text) > MAX_TEXT_CHARS:
        raise EmbeddingRequestError(f"`text` is {len(text)} characters; the limit is {MAX_TEXT_CHARS}.", "text")
    fmt = body.get("response_format", "json")
    if fmt not in FORMATS:
        raise EmbeddingRequestError(f"`response_format` must be one of {', '.join(FORMATS)}.", "response_format")
    language = body.get("language")
    name = None
    if language not in (None, ""):
        if not isinstance(language, str):
            raise EmbeddingRequestError("`language` must be a language name or an ISO-639-1 code.", "language")
        trimmed = language.strip()
        name = ISO_NAMES.get(trimmed.lower().split("-")[0]) if len(trimmed.split("-")[0]) <= 3 else trimmed.title()
        if name not in LANGUAGES:
            raise EmbeddingRequestError(f"`language` {language!r} is not one the aligner takes; it has {', '.join(LANGUAGES)}.", "language")
    else:
        name = guess_language(text)
    words, marks = split_words_marked(text, name)
    if not words:
        raise EmbeddingRequestError("`text` has no words in it to align (only punctuation).", "text")
    limits = {}
    for key, default, low, high in (("max_chars", 42, 8, 200), ("max_seconds", 7.0, 1.0, 30.0)):
        value = body.get(key, default)
        try:
            if isinstance(value, bool):
                raise ValueError
            value = float(value)
        except (TypeError, ValueError):
            raise EmbeddingRequestError(f"`{key}` must be a number.", key) from None
        if not low <= value <= high:
            raise EmbeddingRequestError(f"`{key}` must be from {low:g} to {high:g}.", key)
        limits[key] = value
    return {"text": text, "language": name, "words": words, "marks": marks, "format": fmt, "max_chars": int(limits["max_chars"]),
            "max_seconds": limits["max_seconds"]}


def alignment_items(request, result):
    """[{"word", "start", "end"}] for the words of the request, in seconds."""
    return [{"word": word, "start": start, "end": end} for word, start, end in zip(request["words"], result["starts"], result["ends"])]


def render_alignment(request, result):
    """(content type, bytes) of the answer for the formats that are not JSON: `srt`, `vtt` (the words gathered into lines) and `text` (start, end, word)."""
    import subtitles
    items = alignment_items(request, result)
    if request["format"] == "text":
        return "text/plain; charset=utf-8", "".join(f"{it['start']:.2f} {it['end']:.2f} {it['word']}\n" for it in items).encode("utf-8")
    cues = subtitles.build_cues(items, request["marks"], request["max_chars"], request["max_seconds"])
    if request["format"] == "vtt":
        return "text/vtt; charset=utf-8", subtitles.vtt(cues).encode("utf-8")
    return "application/x-subrip; charset=utf-8", subtitles.srt(cues).encode("utf-8")


def alignment_payload(request, result, duration, model_id):
    """The JSON answer: the words with their times; `verbose_json` adds what the request was about. (`srt`, `vtt` and `text` are made by the caller.)"""
    items = alignment_items(request, result)
    payload = {"words": items}
    if request["format"] == "verbose_json":
        payload = {"task": "align", "model": model_id, "language": (request["language"] or "").lower(), "duration": round(duration, 3), "text": request["text"],
                   "words": items}
    return payload
