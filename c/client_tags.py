"""Tags a client puts on a request, so what it did can be found again by chat or by app (docs/request-log.md, docs/api.md).

    X-Colibri-Tags: {"app": "dashboard", "chat": "c-7f3a"}

A small JSON object in one header: at most 8 names, each a letter then letters, digits, `_`, `.` or `-` (up to 32), each value a string
(up to 120 characters, no control characters), a number or true/false, the whole header up to 1024 bytes. The router writes the tags
on the request's line in the request log, keeps them on a background job and shows them there, and lists only what carries some tags
with `?tag=name:value` on `GET /v1/requests` and `GET /v1/jobs` (more than one `tag` must all match). They change nothing else: the
model never sees them. A header that does not fit is refused with a 400 that says what to change, before anything loads, rather than
cut down in silence.
"""
import json
import math
import re

HEADER = "X-Colibri-Tags"
MAX_BYTES = 1024
MAX_NAMES = 8
MAX_VALUE = 120
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,31}$")
CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class TagError(ValueError):
    """The header or the filter cannot be used; the message says what to change."""


def parse(text):
    """The tags a header holds as a dict, or None when there is none. Raises TagError."""
    if text is None or not str(text).strip():
        return None
    text = str(text)
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise TagError(f"{HEADER} is over {MAX_BYTES} bytes; keep to a few short names.")
    try:
        value = json.loads(text)
    except ValueError:
        raise TagError(f'{HEADER} must be a JSON object, for example {{"app": "dashboard", "chat": "c-7f3a"}}.') from None
    if not isinstance(value, dict):
        raise TagError(f"{HEADER} must be a JSON object, not {type(value).__name__ if value is not None else 'null'}.")
    if len(value) > MAX_NAMES:
        raise TagError(f"{HEADER} has {len(value)} names; at most {MAX_NAMES}.")
    for name, item in value.items():
        if not NAME.match(name):
            raise TagError(f"The tag name {name!r} must start with a letter and use only letters, digits, `_`, `.` and `-` (up to 32).")
        if isinstance(item, bool) or (isinstance(item, (int, float)) and not (isinstance(item, float) and not math.isfinite(item))):
            continue
        if isinstance(item, str):
            if len(item) > MAX_VALUE or CONTROL.search(item):
                raise TagError(f"The value of tag `{name}` must be text of up to {MAX_VALUE} characters with no control characters.")
            continue
        raise TagError(f"The value of tag `{name}` must be a string, a number or true or false.")
    return dict(value) or None


def text(value):
    """A tag value as the filter compares it: what a client would type after the colon."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def wanted(filters):
    """The `tag=name:value` query values as a dict. A value may hold colons; only the first one splits. Raises TagError."""
    out = {}
    for item in filters or ():
        name, colon, value = str(item).partition(":")
        if not colon or not NAME.match(name):
            raise TagError("`tag` must be name:value, for example tag=chat:c-7f3a.")
        out[name] = value
    return out


def matches(tags, filters):
    """Whether `tags` (a dict or None) has every name:value in `filters` (a dict). No filter matches everything."""
    if not filters:
        return True
    tags = tags or {}
    return all(name in tags and text(tags[name]) == value for name, value in filters.items())
