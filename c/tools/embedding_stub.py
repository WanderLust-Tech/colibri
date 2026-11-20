#!/usr/bin/env python3
"""A stand-in for c/qwen3embed that speaks its serve protocol (the header of c/qwen3embed.c), for tests.

    embedding_stub.py --model DIR --serve

The vector of a text is a pure function of the text (and the instruction), so a test can tell two texts apart, see that a
batch keeps its order, and see `dimensions` cut and normalise. A "token" is a word; EMBED_STUB_MAX_TOKENS (default 8) is the
longest input before truncation. Environment: EMBED_STUB_DIM (default 16), EMBED_STUB_DELAY (seconds per request),
EMBED_STUB_DIE_ON (a text that makes the stub exit without answering), EMBED_STUB_FAIL_ON (a text it answers with ERROR).
"""
import hashlib
import json
import math
import os
import sys
import time

DIM = int(os.environ.get("EMBED_STUB_DIM", "16"))
MAX_TOKENS = int(os.environ.get("EMBED_STUB_MAX_TOKENS", "8"))
DELAY = float(os.environ.get("EMBED_STUB_DELAY", "0"))
DIE_ON = os.environ.get("EMBED_STUB_DIE_ON")
FAIL_ON = os.environ.get("EMBED_STUB_FAIL_ON")


def send(keyword, body):
    sys.stdout.write(f"{keyword} {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n")
    sys.stdout.flush()


def vector(text, dims):
    values = []
    block = 0
    while len(values) < DIM:
        digest = hashlib.sha256(f"{block}:{text}".encode("utf-8")).digest()
        values.extend((byte - 127.5) / 127.5 for byte in digest)
        block += 1
    values = values[:DIM][:dims]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def main():
    if "--serve" not in sys.argv:
        sys.exit("embedding_stub: --serve only")
    send("READY", {"model": "embedding-stub", "dimensions": DIM, "max_tokens": MAX_TOKENS, "layers": 1, "min_dimensions": 1})
    for line in sys.stdin:
        keyword, _, rest = line.rstrip("\r\n").partition(" ")
        if keyword != "EMBED":
            send("ERROR", {"id": "", "message": "unknown message"})
            continue
        try:
            request = json.loads(rest)
        except ValueError:
            send("ERROR", {"id": "", "message": "EMBED needs a JSON object"})
            continue
        rid = request.get("id", "")
        inputs = request.get("inputs")
        if not isinstance(inputs, list) or not inputs or not all(isinstance(t, str) for t in inputs):
            send("ERROR", {"id": rid, "message": "inputs must be an array of 1 to 2048 strings"})
            continue
        if DELAY:
            time.sleep(DELAY)
        if DIE_ON is not None and DIE_ON in inputs:
            sys.exit(3)
        if FAIL_ON is not None and FAIL_ON in inputs:
            send("ERROR", {"id": rid, "message": "the text could not be embedded"})
            continue
        dims = request.get("dimensions") or DIM
        if dims > DIM:
            send("ERROR", {"id": rid, "message": f"dimensions is {dims}; this model's vectors are {DIM} wide"})
            continue
        instruction = request.get("instruction") or ""
        vectors, tokens, cuts = [], [], []
        refused = None
        for index, text in enumerate(inputs):
            words = text.split()
            count = len(words) + 1                         # the appended end-of-text token
            cut = count > MAX_TOKENS
            if cut and request.get("truncate") is False:
                refused = f'input {index} is longer than {MAX_TOKENS} tokens (send "truncate": true to cut it)'
                break
            if cut:
                words, count = words[:MAX_TOKENS - 1], MAX_TOKENS
            vectors.append(vector(instruction + "\n" + " ".join(words), dims))
            tokens.append(count)
            cuts.append(1 if cut else 0)
        if refused:
            send("ERROR", {"id": rid, "message": refused})
            continue
        send("VECTORS", {"id": rid, "dimensions": min(dims, DIM), "tokens": tokens, "truncated": cuts, "vectors": vectors})


if __name__ == "__main__":
    main()
