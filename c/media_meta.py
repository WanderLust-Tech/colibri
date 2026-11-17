"""Settings inside a finished picture or song.

A render can take an hour and the file it leaves says nothing about how it was made. This writes one small JSON
record into the file itself, where a viewer or player ignores it: a PNG gets an iTXt chunk named "colibri" (before the
pixels, so a reader that stops early still finds it), a WAV gets a LIST/INFO chunk after the samples whose comment
(ICMT) is the record and whose title (INAM) is the prompt, for a player to show. `read_png` and `read_wav` take it
back out, and the dashboard does the same in the browser to refill its form from a file.

The record is what a request needs to make the same thing again: prompt, size, steps, seed, model. It never holds the
pictures or clips an edit, cover or repaint started from. Setting COLI_MEDIA_META=0 (or false, no, off) leaves the files
plain, for someone who shares them and does not want the prompt to travel with them."""
import json
import os
import struct
import zlib

APP = "colibri"
VERSION = 1
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_KEYWORD = b"colibri"


def enabled(env=None):
    env = os.environ if env is None else env
    return env.get("COLI_MEDIA_META", "1").strip().lower() not in ("0", "false", "no", "off")


def record(kind, model, **fields):
    """The record: the settings that are not None, in the order given, under the fixed header."""
    out = {"app": APP, "version": VERSION, "kind": kind}
    if model:
        out["model"] = model
    out.update({key: value for key, value in fields.items() if value is not None})
    return out


def image_record(request, result, model):
    return record("image", model, prompt=request["prompt"], width=result["width"], height=result["height"],
                  steps=result["steps"], seed=result["seed"],
                  edit=len(request["images"]) if request.get("images") else None,
                  ref_resolution=request.get("ref_resolution") if request.get("images") else None)


def music_record(request, result, model):
    return record("music", model, prompt=request["caption"], lyrics=request.get("lyrics"),
                  language=request.get("language"), duration=result.get("duration", request.get("duration")),
                  steps=result.get("steps", request.get("steps")), shift=result.get("shift", request.get("shift")),
                  seed=result["seed"], bpm=request.get("bpm"), key=request.get("key"),
                  time_signature=request.get("time_signature"), task=request.get("task"))


def _dump(fields):
    return json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _load(raw):
    try:
        fields = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return fields if isinstance(fields, dict) and fields.get("app") == APP else None


# ---------------------------------------------------------------- PNG
def _chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def _png_chunks(png):
    """(type, start, end) of each chunk, end past its CRC; stops at the first thing that is not a chunk."""
    position = len(PNG_MAGIC)
    while position + 12 <= len(png):
        size = struct.unpack(">I", png[position:position + 4])[0]
        end = position + 12 + size
        if end > len(png):
            return
        yield png[position + 4:position + 8], position, end
        position = end


def png_with(png, fields):
    """The PNG with the record added as an iTXt chunk right after IHDR (a record already there is replaced).
    Bytes that are not a PNG come back unchanged."""
    if not png.startswith(PNG_MAGIC):
        return png
    body = PNG_KEYWORD + b"\x00" + b"\x00\x00" + b"\x00" + b"\x00" + _dump(fields)
    kept, header_end = [], None
    for kind, start, end in _png_chunks(png):
        if kind == b"iTXt" and png[start + 8:start + 8 + len(PNG_KEYWORD) + 1] == PNG_KEYWORD + b"\x00":
            continue
        kept.append(png[start:end])
        if kind == b"IHDR":
            header_end = len(kept)
    if header_end is None:
        return png
    kept.insert(header_end, _chunk(b"iTXt", body))
    return PNG_MAGIC + b"".join(kept)


def read_png(png):
    """The record from a PNG, or None when there is none (or it is not ours)."""
    if not png.startswith(PNG_MAGIC):
        return None
    for kind, start, end in _png_chunks(png):
        if kind == b"IDAT":
            break
        if kind != b"iTXt":
            continue
        body = png[start + 8:end - 4]
        keyword, _, rest = body.partition(b"\x00")
        if keyword != PNG_KEYWORD or len(rest) < 2 or rest[0] != 0:      # an uncompressed one only
            continue
        _language, _, rest = rest[2:].partition(b"\x00")
        _translated, _, text = rest.partition(b"\x00")
        return _load(text)
    return None


# ---------------------------------------------------------------- WAV
def _riff(kind, body):
    return kind + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) & 1 else b"")


def _text(value):
    return value.encode("utf-8") + b"\x00"


def wav_with(wav, fields):
    """The WAV with a LIST/INFO chunk added after its last chunk (RIFF size fixed). Bytes that are not a RIFF/WAVE
    file come back unchanged."""
    if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return wav
    info = b"INFO" + _riff(b"ISFT", _text("Colibri"))
    if fields.get("prompt"):
        info += _riff(b"INAM", _text(str(fields["prompt"])[:200]))
    info += _riff(b"ICMT", _text(_dump(fields).decode("utf-8")))
    out = wav + _riff(b"LIST", info)
    return out[:4] + struct.pack("<I", len(out) - 8) + out[8:]


def read_wav(wav):
    """The record from a WAV, or None when there is none (or it is not ours)."""
    if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return None
    position = 12
    while position + 8 <= len(wav):
        kind, size = wav[position:position + 4], struct.unpack("<I", wav[position + 4:position + 8])[0]
        body = wav[position + 8:position + 8 + size]
        if kind == b"LIST" and body[:4] == b"INFO":
            inner = 4
            while inner + 8 <= len(body):
                tag, length = body[inner:inner + 4], struct.unpack("<I", body[inner + 4:inner + 8])[0]
                if tag == b"ICMT":
                    found = _load(body[inner + 8:inner + 8 + length].rstrip(b"\x00"))
                    if found:
                        return found
                inner += 8 + length + (length & 1)
        position += 8 + size + (size & 1)
    return None
