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
