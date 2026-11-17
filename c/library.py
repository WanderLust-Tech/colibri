"""The pictures and songs a server has kept, listed and served (GET /v1/library).

Every finished picture goes to ~/colibri-images and every song to ~/colibri-music (COLI_IMAGE_DIR, COLI_MUSIC_DIR), but
nothing listed or served them: a page that was closed during a render had no way back to the result. This reads those two
folders and nothing else. A name is one path component that exists as a plain file in its folder, with the extension the
kind uses; `..`, a separator, a drive letter or a link out of the folder is "not found".

Each entry says what the file is (name, size, time) and, read from the file's own chunk headers and never its pixels or
samples, how big it is and the settings media_meta wrote into it (prompt, seed, model...), so a listing can show a card and
offer "make this again". Files are never overwritten (the server picks a new name), so a served file may be cached by the
browser for a day.
"""
import os
import re
import threading

import image_engine
import media_meta
import music_engine

KINDS = {
    "images": {"dir": image_engine.image_dir, "ext": ".png", "type": "image/png", "inspect": media_meta.inspect_png},
    "music": {"dir": music_engine.music_dir, "ext": ".wav", "type": "audio/wav", "inspect": media_meta.inspect_wav},
}
MAX_LIMIT = 200
DEFAULT_LIMIT = 50
CACHE_SECONDS = 86400
_NAME = re.compile(r"^[^\\/:\x00-\x1f]+$")
_cache = {}                       # (path, mtime_ns, size) -> what inspect found; bounded, and only ever a speed-up
_cache_lock = threading.Lock()
_CACHE_MAX = 4096


class LibraryError(Exception):
    def __init__(self, status, message, code, param=None):
        super().__init__(message)
        self.status, self.message, self.code, self.param = status, message, code, param


def kind_of(value):
    if value not in KINDS:
        raise LibraryError(404, "Not found.", "not_found")
    return value


def folder(kind, env=None):
    return os.path.realpath(os.path.expanduser(KINDS[kind]["dir"](env)))


def resolve(kind, name, env=None):
    """The real path of a kept file, or None for anything that is not a plain file of this kind directly inside its folder."""
    spec = KINDS[kind]
    if not isinstance(name, str) or name in ("", ".", "..") or not _NAME.match(name) or not name.lower().endswith(spec["ext"]):
        return None
    root = folder(kind, env)
    path = os.path.realpath(os.path.join(root, name))
    if os.path.dirname(path) != root or not os.path.isfile(path):          # a link to somewhere else resolves out of the folder
        return None
    return path


def _inspected(kind, path, stat):
    key = (path, stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        found = _cache.get(key)
    if found is None:
        found = KINDS[kind]["inspect"](path)
        with _cache_lock:
            if len(_cache) >= _CACHE_MAX:
                _cache.clear()
            _cache[key] = found
    return found


def entry(kind, path, stat=None):
    stat = stat or os.stat(path)
    name = os.path.basename(path)
    out = {"kind": kind, "name": name, "url": f"/v1/library/{kind}/{name}", "bytes": stat.st_size,
           "modified": int(stat.st_mtime)}
    out.update(_inspected(kind, path, stat))
    return out


def listing(kind=None, limit=DEFAULT_LIMIT, offset=0, env=None):
    """{"object": "list", "data": [...], "total", "has_more"}: the newest first. A folder that does not exist yet is an empty
    library, not an error."""
    kinds = [kind_of(kind)] if kind else list(KINDS)
    found = []
    for each in kinds:
        ext, root = KINDS[each]["ext"], folder(each, env)
        try:
            with os.scandir(root) as scan:
                for item in scan:
                    if item.name.lower().endswith(ext) and item.is_file(follow_symlinks=False):
                        found.append((item.stat().st_mtime_ns, item.name, each, item.path))
        except OSError:
            continue
    found.sort(reverse=True)
    page = found[offset:offset + limit]
    data = []
    for _mtime, _name, each, path in page:
        try:
            data.append(entry(each, path))
        except OSError:
            continue                                # deleted between the scan and now
    return {"object": "list", "data": data, "total": len(found), "has_more": offset + len(page) < len(found)}


def page_arguments(query):
    """(kind, limit, offset) from a parsed query string, 400 on anything that is not a plain number in range."""
    def number(name, default, low, high):
        raw = (query.get(name) or [None])[0]
        if raw is None or raw == "":
            return default
        if not re.fullmatch(r"\d{1,9}", raw) or not low <= int(raw) <= high:
            raise LibraryError(400, f"`{name}` must be a whole number from {low} to {high}.", "invalid_value", name)
        return int(raw)
    kind = (query.get("kind") or [None])[0] or None
    if kind is not None and kind not in KINDS:
        raise LibraryError(400, f"`kind` must be one of: {', '.join(KINDS)}.", "invalid_value", "kind")
    return kind, number("limit", DEFAULT_LIMIT, 1, MAX_LIMIT), number("offset", 0, 0, 10 ** 9)


def byte_range(header, size):
    """(start, end) inclusive for a single `Range: bytes=a-b` header, None when there is none or it cannot be read as one,
    "unsatisfiable" when it lies past the end. Several ranges are not served: the whole file is."""
    if not header:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match or (match.group(1) == "" and match.group(2) == ""):
        return None
    first, last = match.groups()
    if first == "":                                 # the last N bytes
        count = int(last)
        if count == 0:
            return "unsatisfiable"
        return max(0, size - count), size - 1
    start = int(first)
    end = size - 1 if last == "" else min(int(last), size - 1)
    if start >= size or end < start:
        return "unsatisfiable"
    return start, end


def serve(handler, kind, name, cors, env=None):
    """Send one kept file through a request handler (either server's): 200 with the bytes, 206 for a byte range (a player
    seeks with them), 404 for a name that is not a kept file, 416 for a range past the end. `cors` adds the server's
    CORS headers before the headers end."""
    kind = kind_of(kind)
    path = resolve(kind, name, env)
    if path is None:
        raise LibraryError(404, "Not found.", "not_found")
    size = os.path.getsize(path)
    span = byte_range(handler.headers.get("Range"), size)
    if span == "unsatisfiable":
        handler.send_response(416)
        handler.send_header("Content-Range", f"bytes */{size}")
        handler.send_header("Content-Length", "0")
        cors()
        handler.end_headers()
        return
    start, end = span if span else (0, size - 1)
    handler.send_response(206 if span else 200)
    handler.send_header("Content-Type", KINDS[kind]["type"])
    handler.send_header("Content-Length", str(end - start + 1))
    handler.send_header("Accept-Ranges", "bytes")
    if span:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.send_header("Cache-Control", f"private, max-age={CACHE_SECONDS}")
    handler.send_header("Content-Disposition", "inline")
    cors()
    handler.end_headers()
    if handler.command == "HEAD":
        return
    with open(path, "rb") as source:
        source.seek(start)
        left = end - start + 1
        while left > 0:
            chunk = source.read(min(left, 1 << 20))
            if not chunk:
                break
            handler.wfile.write(chunk)
            left -= len(chunk)
