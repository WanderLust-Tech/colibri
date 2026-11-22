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
    timed = timed_format(a)
    try:
        model = choose_model(api, "transcription", a.model, "turns speech into text")
        aligner = choose_model(api, "alignment", getattr(a, "aligner", None), "places words in speech") if timed else None
    except Refused as why:
        print(f"  {C.yel}{why}{C.r}")
        return 1
    for path in a.files:
        try:
            answer, text = one(a, api, upload, model, path)
            if timed and text:
                body = align_file(a, api, upload, aligner, path, text, timed[0])
        except Refused as why:
            print(f"  {C.yel}{path}: {why}{C.r}", file=sys.stderr if a.json else sys.stdout)
            status = 1
            continue
        if timed:
            if not text:
                print(f"  {C.dim}(nothing was heard in {excerpt(os.path.basename(path), 40)}){C.r}")
            elif deliver_timed(a, path, body, timed[1], C):
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


TIMED_FORMATS = (("srt", "srt", ".srt"), ("vtt", "vtt", ".vtt"), ("words", "text", ".txt"))


def timed_format(a):
    """(response_format, suffix) the person asked for with --srt, --vtt or --words, or None."""
    for flag, fmt, suffix in TIMED_FORMATS:
        if getattr(a, flag, False):
            return fmt, suffix
    return None


def align_file(a, api, upload, aligner, path, text, fmt):
    """The aligner's answer for `text` said in the file at `path`, in the response format `fmt`: the body (text, or the JSON). Refused for what the person can act on."""
    fields = {"model": aligner, "text": text, "response_format": fmt}
    if getattr(a, "language", None):
        fields["language"] = a.language
    status, answer = upload("/v1/audio/alignments", fields, os.path.basename(path), read_audio(path))
    if status is None:
        raise Refused("nothing answers on that host and port: start the server (coli service start, or coli router)")
    if status == 401:
        raise Refused("the server wants an API key: --api-key, or COLI_API_KEY")
    if status != 200:
        error = (answer or {}).get("error") if isinstance(answer, dict) else None
        message = error.get("message") if isinstance(error, dict) else error
        raise Refused(f"the server said {status}: {message or 'no reason given'}")
    return answer


def deliver_timed(a, path, body, suffix, C):
    """Print the aligned words, or write the subtitles next to the audio file (or to --out). Returns 0, or 1 for a file that cannot be written."""
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    if suffix == ".txt" or getattr(a, "json", False):
        print(text.rstrip("\n"))
        return 0
    target = getattr(a, "out", None) or os.path.splitext(path)[0] + suffix
    try:
        with open(target, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except OSError as error:
        print(f"  {C.yel}cannot write {target}: {error.strerror or error}{C.r}")
        return 1
    print(f"  {C.dim}wrote {target}{C.r}")
    return 0


def align(a, api, upload, C):
    """coli align FILE (--text WORDS | --text-file FILE) [--srt | --vtt | --words | --json] [--language CODE] [--aligner ID] [--out FILE]

    Where each word of a transcript you have is said in the audio, by the server's forced aligner (Qwen3-ForcedAligner): the words with their times by
    default, `--srt` or `--vtt` for subtitles (written next to the audio, or to --out)."""
    text = a.text
    if getattr(a, "text_file", None):
        try:
            with open(a.text_file, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as error:
            print(f"  {C.yel}cannot read {a.text_file}: {error.strerror or error}{C.r}")
            return 1
    if not a.files or len(a.files) != 1 or not (text or "").strip():
        print(f"  {C.yel}usage: coli align FILE (--text \"what is said\" | --text-file FILE) [--srt | --vtt | --words | --json] [--language en] [--aligner ID] [--out FILE]{C.r}")
        return 1
    chosen = timed_format(a) or ("verbose_json" if a.json else "text", ".txt")
    try:
        aligner = choose_model(api, "alignment", a.aligner, "places words in speech")
        body = align_file(a, api, upload, aligner, a.files[0], text, chosen[0])
    except Refused as why:
        print(f"  {C.yel}{why}{C.r}")
        return 1
    return deliver_timed(a, a.files[0], body, chosen[1] if not a.json else ".json", C)


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
    reference = getattr(a, "reference", None)
    cloning = bool(reference)
    if not cloning and a.voice and not a.model:                      # a saved voice is a cloning model's: ask it
        status, saved = api("GET", "/v1/audio/voices")
        cloning = status == 200 and any(item.get("name") == a.voice.strip().lower() for item in (saved or {}).get("data") or [])
    try:
        model = clone_model(api, a) if cloning else choose_model(api, "speech", a.model, "speaks text")
        body = {"model": model, "input": words, "response_format": fmt}
        if reference:
            body["reference_audio"] = encode_reference(reference)
            if getattr(a, "reference_text", None):
                body["reference_text"] = a.reference_text
            if getattr(a, "x_vector", False):
                body["x_vector_only"] = True
    except Refused as why:
        print(f"  {C.yel}{why}{C.r}")
        return 1
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


MAX_REFERENCE_BYTES = 3 << 20      # an inline reference rides in the request as base64: the server takes 4 MB a request


def clone_model(api, a):
    """The model to ask for a cloned voice: --model, else the first speech model that is a Base (cloning) checkpoint, else any speech model."""
    if a.model:
        return a.model
    status, body = api("GET", "/v1/models")
    if status is not None and status != 401:
        for item in (body or {}).get("data") or []:
            if "speech" in (item.get("capabilities") or []) and "base" in str(item.get("id", "")).lower():
                return item["id"]
    return choose_model(api, "speech", None, "speaks text")


def encode_reference(path):
    """The base64 of a reference clip, or Refused with a reason."""
    import base64
    try:
        size = os.path.getsize(path)
    except OSError as error:
        raise Refused(f"cannot read {path}: {error.strerror or error}")
    if size > MAX_REFERENCE_BYTES:
        raise Refused(f"{path} is {size / 1e6:.1f} MB; send at most {MAX_REFERENCE_BYTES >> 20} MB (a clip of 3 to 30 s: trim it, or make it an mp3)")
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


def voice(a, api, upload, C):
    """coli voice list | add NAME FILE [--text WORDS] [--no-text] | remove NAME: the saved voices of the cloning speech models.

    A voice is a few seconds of reference speech and what it says; `coli speak --voice NAME` then speaks in it. Without --text the server's
    speech recognition model writes the words down (--no-text skips that: the voice is then the speaker's embedding alone, a looser likeness)."""
    action = a.action
    if action == "list":
        status, body = api("GET", "/v1/audio/voices")
        if status is None:
            print(f"  {C.yel}nothing answers on that host and port: start the server (coli service start, or coli router){C.r}")
            return 1
        if status == 401:
            print(f"  {C.yel}the server wants an API key: --api-key, or COLI_API_KEY{C.r}")
            return 1
        if status != 200:
            print(f"  {C.yel}the server said {status}: it has no saved voices (a router keeps them){C.r}")
            return 1
        found = (body or {}).get("data") or []
        if getattr(a, "json", False):
            print(json.dumps(body))
            return 0
        if not found:
            print(f"  {C.dim}no saved voices: coli voice add NAME clip.wav --text \"what the clip says\"{C.r}")
            return 0
        for item in found:
            print(f"  {item['name']:<20} {item.get('seconds', 0):>5.1f} s  {item.get('mode', ''):<10} {C.dim}{excerpt(item.get('text') or '', 60)}{C.r}")
        return 0
    if action == "remove":
        if not a.name:
            print(f"  {C.yel}usage: coli voice remove NAME{C.r}")
            return 1
        status, body = api("POST", "/v1/audio/voices/delete", {"name": a.name})
        if status == 200:
            print(f"  removed {a.name}")
            return 0
        error = (body or {}).get("error") if isinstance(body, dict) else None
        print(f"  {C.yel}{(error or {}).get('message') if isinstance(error, dict) else 'the server said ' + str(status)}{C.r}")
        return 1
    if action == "add":
        if not a.name or not a.file:
            print(f"  {C.yel}usage: coli voice add NAME FILE [--text \"what the clip says\"] [--no-text]{C.r}")
            return 1
        try:
            data = encode_reference(a.file)
            words = (a.text or "").strip()
            if not words and not a.no_text:
                fields = {"model": "", "response_format": "json"}
                try:
                    model = choose_model(api, "transcription", None, "turns speech into text")
                    fields["model"] = model
                    status, answer = upload("/v1/audio/transcriptions", fields, os.path.basename(a.file), read_audio(a.file))
                    if status == 200:
                        words = str((answer or {}).get("text", "")).strip()
                        print(f"  {C.dim}heard: {excerpt(words, 100)}{C.r}")
                    else:
                        print(f"  {C.yel}the words could not be written down (the server said {status}); saving the voice without them{C.r}")
                except Refused as why:
                    print(f"  {C.yel}{why}; saving the voice without the words (--text gives them){C.r}")
        except Refused as why:
            print(f"  {C.yel}{why}{C.r}")
            return 1
        status, body = api("POST", "/v1/audio/voices", {"name": a.name, "audio": data, "text": words})
        if status is None:
            print(f"  {C.yel}nothing answers on that host and port: start the server (coli service start, or coli router){C.r}")
            return 1
        if status != 200:
            error = (body or {}).get("error") if isinstance(body, dict) else None
            print(f"  {C.yel}{(error or {}).get('message') if isinstance(error, dict) else 'the server said ' + str(status)}{C.r}")
            return 1
        print(f"  {body.get('status', 'saved')} {body.get('name')}: {body.get('seconds', 0):.1f} s, {body.get('mode')}")
        return 0
    print(f"  {C.yel}usage: coli voice list | add NAME FILE [--text WORDS] | remove NAME{C.r}")
    return 1
