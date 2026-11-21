"""`coli transcribe` (docs/qwen3-asr.md): speech to text from the terminal, through a running server's speech recognition model.

It talks to a running server (`coli router` or `coli serve`) through its API, the same one an application uses. `api(method, path, body)`
and `upload(path, fields, filename, data)` are coli's helpers: each returns (status, parsed JSON or text) or (None, None) when nothing
answers. The functions here print what they find and return the exit code; nothing here starts an engine."""
import json
import os
import sys

import retrieval_cli
from retrieval_cli import Refused, choose_model, excerpt

MAX_BYTES = 25 << 20           # the gateway's own limit on an upload


def read_audio(path):
    """The bytes of an audio file, or Refused with a reason."""
    try:
        size = os.path.getsize(path)
    except OSError as error:
        raise Refused(f"cannot read {path}: {error.strerror or error}")
    if size > MAX_BYTES:
        raise Refused(f"{path} is {size / 1e6:.0f} MB; the server takes up to {MAX_BYTES >> 20} MB a request (cut it, or convert it to a smaller format)")
    if size == 0:
        raise Refused(f"{path} is empty")
    with open(path, "rb") as handle:
        return handle.read()


def one(a, api, upload, model, path):
    """Transcribes one file; returns (answer, text). Refused for what the person can act on."""
    fields = {"model": model, "response_format": "verbose_json" if a.verbose or a.json else "json"}
    if a.language:
        fields["language"] = a.language
    if a.context:
        fields["prompt"] = a.context
    status, answer = upload("/v1/audio/transcriptions", fields, os.path.basename(path), read_audio(path))
    if status is None:
        raise Refused("nothing answers on that host and port: start the server (coli service start, or coli router)")
    if status == 401:
        raise Refused("the server wants an API key: --api-key, or COLI_API_KEY")
    if status != 200:
        error = (answer or {}).get("error") if isinstance(answer, dict) else None
        message = error.get("message") if isinstance(error, dict) else error
        raise Refused(f"the server said {status}: {message or 'no reason given'}")
    return answer, str((answer or {}).get("text", ""))


def transcribe(a, api, upload, C):
    """coli transcribe FILE... [--language CODE] [--context TEXT] [--model ID] [--verbose] [--json]"""
    if not a.files:
        print(f"  {C.yel}usage: coli transcribe FILE [FILE ...] [--language en] [--context \"names, terms\"] [--model ID] [--verbose] [--json]{C.r}")
        return 1
    status = 0
    try:
        model = choose_model(api, "transcription", a.model, "turns speech into text")
    except Refused as why:
        print(f"  {C.yel}{why}{C.r}")
        return 1
    for path in a.files:
        try:
            answer, text = one(a, api, upload, model, path)
        except Refused as why:
            print(f"  {C.yel}{path}: {why}{C.r}", file=sys.stderr if a.json else sys.stdout)
            status = 1
            continue
        if a.json:
            print(json.dumps(answer))
            continue
        if len(a.files) > 1 or a.verbose:
            seconds = answer.get("duration")
            language = answer.get("language") or "language not said"
            heard = f"{seconds:.1f} s, {language}" if isinstance(seconds, (int, float)) else language
            print(f"  {C.dim}{os.path.basename(path)}  ({heard}){C.r}")
        print(text if text else f"  {C.dim}(nothing was heard in {excerpt(os.path.basename(path), 40)}){C.r}")
    return status


FORMAT_OF_SUFFIX = {".wav": "wav", ".mp3": "mp3", ".flac": "flac", ".opus": "opus", ".ogg": "opus", ".aac": "aac", ".pcm": "pcm"}
SUFFIX_OF_FORMAT = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "opus": ".opus", "aac": ".aac", "pcm": ".pcm"}


def speak(a, api, fetch, C):
    """coli speak TEXT... [--voice NAME] [--instruct STYLE] [--language CODE] [--out FILE] [--format F] [--seed N] [--model ID]

    `fetch(path, body)` is coli's POST that hands back (status, the raw body bytes, the headers as a dict), or (None, None, None) when
    nothing answers. The audio goes to --out (speech.wav by default; the file's suffix picks the format when --format is not given);
    the words are the arguments, or standard input when the only argument is -."""
    words = " ".join(a.text or []).strip()
    if words == "-":
        words = sys.stdin.read().strip()
    if not words:
        print(f"  {C.yel}usage: coli speak TEXT [--voice ryan] [--instruct \"speak slowly\"] [--language en] [--out speech.wav] [--format wav|mp3|flac|opus|aac|pcm] [--seed N] [--model ID]{C.r}")
        return 1
    out = a.out or "speech" + SUFFIX_OF_FORMAT.get(a.format or "wav", ".wav")
    fmt = a.format or FORMAT_OF_SUFFIX.get(os.path.splitext(out)[1].lower(), "wav")
    try:
        model = choose_model(api, "speech", a.model, "speaks text")
    except Refused as why:
        print(f"  {C.yel}{why}{C.r}")
        return 1
    body = {"model": model, "input": words, "response_format": fmt}
    if a.voice:
        body["voice"] = a.voice
    if getattr(a, "instruct", None):
        body["instructions"] = a.instruct
    if a.language:
        body["language"] = a.language
    if a.seed is not None:
        body["seed"] = a.seed
    status, data, headers = fetch("/v1/audio/speech", body)
    if status is None:
        print(f"  {C.yel}nothing answers on that host and port: start the server (coli service start, or coli router){C.r}")
        return 1
    if status == 401:
        print(f"  {C.yel}the server wants an API key: --api-key, or COLI_API_KEY{C.r}")
        return 1
    if status != 200:
        try:
            error = json.loads(data or b"null")
            error = error.get("error") if isinstance(error, dict) else None
            message = error.get("message") if isinstance(error, dict) else error
        except ValueError:
            message = None
        print(f"  {C.yel}the server said {status}: {message or 'no reason given'}{C.r}")
        return 1
    try:
        with open(out, "wb") as handle:
            handle.write(data)
    except OSError as error:
        print(f"  {C.yel}cannot write {out}: {error.strerror or error}{C.r}")
        return 1
    seconds = (headers or {}).get("x-colibri-audio-seconds")
    said = f"{float(seconds):.1f} s of speech, " if seconds else ""
    print(f"  {out}  ({said}{len(data) / 1e3:.0f} kB, {fmt}){C.dim}{'  [cut short: the text was longer than one piece]' if (headers or {}).get('x-colibri-truncated') else ''}{C.r}")
    return 0
