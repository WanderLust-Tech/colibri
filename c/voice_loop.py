#!/usr/bin/env python3
"""The voice loop: speech in, a chat model's answer, speech out, through a running server's three models (docs/voice-loop.md).

    heard = transcribe(speech)        POST /v1/audio/transcriptions   (Qwen3-ASR)
    reply = chat(heard)               POST /v1/chat/completions       (any chat model, streamed)
    spoken = speak(reply)             POST /v1/audio/speech           (Qwen3-TTS, streamed as server-sent events)

The reply is cut into sentences as the chat model writes them and each goes to the speech model at once, so the voice starts after the first sentence
rather than after the whole answer. A router that computes one model at a time runs them in turn (the speech starts when the text is done); one
started with parallel models overlaps them. Stdlib only; `VoiceLoop` is what `coli talk` and the tests drive, and any client can do the same three calls.
"""
import base64
import http.client
import json
import queue
import re
import threading
import time
import uuid
from urllib.parse import urlsplit

DEFAULT_SYSTEM = ("You are a voice assistant: your answer is spoken aloud. Answer in one to three short sentences of plain speech, as you would say "
                  "it. No lists, no markdown, no emoji, no code unless asked. Say numbers and symbols in words.")
MAX_HISTORY = 16                         # messages kept (not counting the system prompt): a spoken conversation is short
MAX_SPEECH_CHARS = 4096                  # the speech route's limit on one request


class VoiceError(Exception):
    """A step that failed, with the step's name in `step` (transcribe, chat, speak)."""

    def __init__(self, step, message):
        super().__init__(f"{step}: {message}")
        self.step = step
        self.message = message


# ------------------------------------------------------------------------------------------------ text for the voice

_FENCE = re.compile(r"```.*?(```|$)", re.S)
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_INLINE = re.compile(r"[*_`~]{1,3}")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.M)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_EMOJI = re.compile("[\U0001F000-\U0001FFFF☀-➿️‍]")


def spoken_text(text):
    """What a voice should say of a chat reply: markdown, code blocks, links' addresses and emoji dropped, white space made plain."""
    text = _FENCE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _INLINE.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceSplitter:
    """Cuts streamed text into sentences for the voice. `feed(chunk)` returns the sentences completed so far; `flush()` the rest.

    A sentence ends at . ! ? … or a line break followed by white space or the end of a chunk's text; one shorter than `minimum` characters (the first
    one: `first`, so the voice starts early) is held and joined to the next, which keeps "Dr. Smith" and "3.5" in one piece."""

    END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+|\n+")

    def __init__(self, minimum=24, first=14):
        self.minimum = minimum
        self.first = first
        self.buffer = ""
        self.sent = 0

    def _take(self, final):
        out = []
        while True:
            need = self.first if self.sent == 0 and not out else self.minimum
            cut = None
            for match in self.END.finditer(self.buffer):
                piece = self.buffer[:match.end()]
                if len(spoken_text(piece)) >= need:
                    cut = match.end()
                    break
            if cut is None:
                break
            out.append(self.buffer[:cut])
            self.buffer = self.buffer[cut:]
            self.sent += 1
        if final and self.buffer.strip():
            out.append(self.buffer)
            self.buffer = ""
            self.sent += 1
        return [s for s in (spoken_text(piece) for piece in out) if s]

    def feed(self, chunk):
        self.buffer += chunk
        return self._take(False)

    def flush(self):
        return self._take(True)


# ------------------------------------------------------------------------------------------------ the three calls

class Server:
    """One colibri server (a router or `coli serve`): its three calls, each a plain HTTP request on a fresh connection."""

    def __init__(self, base_url, api_key=None, timeout=600):
        parts = urlsplit(base_url if "//" in base_url else "http://" + base_url)
        self.host = parts.hostname or "127.0.0.1"
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.secure = parts.scheme == "https"
        self.prefix = parts.path.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(self, path, body, content_type="application/json"):
        """The open response of a POST (the caller reads it and closes the connection through `response.conn`)."""
        cls = http.client.HTTPSConnection if self.secure else http.client.HTTPConnection
        conn = cls(self.host, self.port, timeout=self.timeout)
        headers = {"Content-Type": content_type}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            conn.request("POST", self.prefix + path, body, headers)
            response = conn.getresponse()
        except (OSError, http.client.HTTPException) as error:
            conn.close()
            raise ConnectionError(f"nothing answers at {self.host}:{self.port} ({error})") from error
        response.conn = conn
        return response

    def error_of(self, response):
        try:
            raw = response.read()
            error = json.loads(raw).get("error")
            return error.get("message") if isinstance(error, dict) else str(error)
        except (ValueError, AttributeError):
            return f"HTTP {response.status}"
        finally:
            response.conn.close()


def multipart(fields, filename, data):
    boundary = "----coli" + uuid.uuid4().hex
    out = []
    for name, value in fields.items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
    out.append(data)
    out.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def transcribe(server, model, audio, language=None):
    """The words in `audio` (the bytes of an audio file), by the speech recognition model `model`."""
    fields = {"model": model, "response_format": "json"}
    if language:
        fields["language"] = language
    body, content_type = multipart(fields, "speech.wav", audio)
    try:
        response = server.request("/v1/audio/transcriptions", body, content_type)
    except ConnectionError as error:
        raise VoiceError("transcribe", str(error)) from None
    if response.status != 200:
        raise VoiceError("transcribe", server.error_of(response))
    try:
        return str(json.loads(response.read()).get("text", "")).strip()
    finally:
        response.conn.close()


def sse_events(response):
    """The JSON payloads of a server-sent-event response, as they arrive; the connection is closed at the end ([DONE] ends it too)."""
    buffer = b""
    try:
        while True:
            chunk = response.read1(65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n\n" in buffer:
                block, buffer = buffer.split(b"\n\n", 1)
                for line in block.split(b"\n"):
                    if line.startswith(b"data:"):
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            return
                        try:
                            yield json.loads(data)
                        except ValueError:
                            continue
    finally:
        response.conn.close()


def chat_stream(server, model, messages, **options):
    """The text of a chat model's answer, in the pieces it is written in."""
    body = json.dumps({"model": model, "messages": messages, "stream": True, **options}).encode()
    try:
        response = server.request("/v1/chat/completions", body)
    except ConnectionError as error:
        raise VoiceError("chat", str(error)) from None
    if response.status != 200:
        raise VoiceError("chat", server.error_of(response))
    for event in sse_events(response):
        if isinstance(event.get("error"), dict):
            raise VoiceError("chat", str(event["error"].get("message") or "the model failed"))
        for choice in event.get("choices") or ():
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                yield piece


def speak_stream(server, model, text, voice=None, language=None, **extra):
    """The speech for `text` as PCM (16-bit mono, 24 kHz) in the pieces the speech model makes it in."""
    request = {"model": model, "input": text[:MAX_SPEECH_CHARS], "stream_format": "sse", **extra}
    if voice:
        request["voice"] = voice
    if language:
        request["language"] = language
    try:
        response = server.request("/v1/audio/speech", json.dumps(request).encode())
    except ConnectionError as error:
        raise VoiceError("speak", str(error)) from None
    if response.status != 200:
        raise VoiceError("speak", server.error_of(response))
    for event in sse_events(response):
        kind = event.get("type")
        if kind == "speech.audio.delta":
            yield base64.b64decode(event.get("audio") or "")
        elif kind == "error":
            raise VoiceError("speak", str((event.get("error") or {}).get("message") or "the speech model failed"))


# ------------------------------------------------------------------------------------------------ the loop

class VoiceLoop:
    """A conversation: what was said and answered so far, and the three models that carry it."""

    def __init__(self, server, chat_model, asr_model, tts_model, voice=None, language=None, system=DEFAULT_SYSTEM, chat_options=None, speech_options=None):
        self.server = server
        self.chat_model, self.asr_model, self.tts_model = chat_model, asr_model, tts_model
        self.voice, self.language = voice, language
        self.system = system
        self.chat_options = dict(chat_options or {})
        self.speech_options = dict(speech_options or {})
        self.history = []

    def messages(self, heard):
        kept = self.history[-MAX_HISTORY:]
        return ([{"role": "system", "content": self.system}] if self.system else []) + kept + [{"role": "user", "content": heard}]

    def hear(self, audio):
        return transcribe(self.server, self.asr_model, audio, self.language)

    def turn(self, audio=None, text=None, emit=lambda kind, value: None, speak=True):
        """One exchange. The user's words are `text`, or the transcription of `audio`. `emit(kind, value)` is told as things happen:
        "heard" (the words), "text" (a piece of the reply), "reply" (all of it), "audio" (PCM bytes), "timing" (seconds, per step).
        Returns {"heard", "reply", "timing"}. A step that fails raises VoiceError; what was said so far stays in the history."""
        started = time.monotonic()
        timing = {}
        heard = text if text is not None else self.hear(audio)
        timing["heard"] = time.monotonic() - started
        emit("heard", heard)
        if not heard.strip():
            return {"heard": "", "reply": "", "timing": timing}
        sentences = queue.Queue()
        failure = []

        def mouth():
            first = True
            while True:
                sentence = sentences.get()
                if sentence is None:
                    return
                if failure:
                    continue                                      # the voice failed: drain the rest, the text still goes on
                try:
                    for pcm in speak_stream(self.server, self.tts_model, sentence, self.voice, self.language, **self.speech_options):
                        if first:
                            timing["first_audio"] = time.monotonic() - started
                            first = False
                        emit("audio", pcm)
                except VoiceError as error:
                    failure.append(error)

        thread = threading.Thread(target=mouth, daemon=True) if speak else None
        if thread:
            thread.start()
        splitter = SentenceSplitter()
        reply = []
        try:
            for piece in chat_stream(self.server, self.chat_model, self.messages(heard), **self.chat_options):
                if "first_text" not in timing:
                    timing["first_text"] = time.monotonic() - started
                reply.append(piece)
                emit("text", piece)
                if thread:
                    for sentence in splitter.feed(piece):
                        sentences.put(sentence)
            timing["text_done"] = time.monotonic() - started
            if thread:
                for sentence in splitter.flush():
                    sentences.put(sentence)
        finally:
            if thread:
                sentences.put(None)
                thread.join()
        answer = "".join(reply).strip()
        if answer:
            self.history += [{"role": "user", "content": heard}, {"role": "assistant", "content": answer}]
            del self.history[:-MAX_HISTORY]
        timing["done"] = time.monotonic() - started
        emit("reply", answer)
        emit("timing", timing)
        if failure:
            raise failure[0]
        return {"heard": heard, "reply": answer, "timing": timing}
