#!/usr/bin/env python3
"""A stand-in for c/qwen3embed on a reranker checkpoint, speaking its serve protocol (the header of c/qwen3embed.c), for tests.

    rerank_stub.py --model DIR --serve

The score of a document is a pure function of the query and the document, so a test can see that the answer follows the
order of the documents and that sorting works: the share of the query's words the document contains (0 to 1, with 0.001 and
0.999 as the ends so the logit is finite). A "token" is a word of the query and the document plus 2 for the fixed ends;
RERANK_STUB_MAX_TOKENS (default 12) is the longest pair before the document is cut. Environment: RERANK_STUB_DELAY (seconds
per request), RERANK_STUB_DIE_ON (a document that makes the stub exit without answering), RERANK_STUB_FAIL_ON (a document it
answers with ERROR).
"""
import json
import math
import os
import sys
import time

MAX_TOKENS = int(os.environ.get("RERANK_STUB_MAX_TOKENS", "12"))
DELAY = float(os.environ.get("RERANK_STUB_DELAY", "0"))
DIE_ON = os.environ.get("RERANK_STUB_DIE_ON")
FAIL_ON = os.environ.get("RERANK_STUB_FAIL_ON")


def send(keyword, body):
    sys.stdout.write(f"{keyword} {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n")
    sys.stdout.flush()


def main():
    if "--serve" not in sys.argv:
        sys.exit("rerank_stub: --serve only")
    send("READY", {"model": "rerank-stub", "task": "rerank", "max_tokens": MAX_TOKENS, "layers": 1, "max_documents": 2048})
    for line in sys.stdin:
        keyword, _, rest = line.rstrip("\r\n").partition(" ")
        if keyword != "RERANK":
            send("ERROR", {"id": "", "message": "unknown message"})
            continue
        try:
            request = json.loads(rest)
        except ValueError:
            send("ERROR", {"id": "", "message": "RERANK needs a JSON object"})
            continue
        rid = request.get("id", "")
        query, documents = request.get("query"), request.get("documents")
        if not isinstance(query, str):
            send("ERROR", {"id": rid, "message": "query must be a string"})
            continue
        if not isinstance(documents, list) or not documents or not all(isinstance(t, str) for t in documents):
            send("ERROR", {"id": rid, "message": "documents must be an array of 1 to 2048 strings"})
            continue
        if DELAY:
            time.sleep(DELAY)
        if DIE_ON is not None and DIE_ON in documents:
            sys.exit(3)
        if FAIL_ON is not None and FAIL_ON in documents:
            send("ERROR", {"id": rid, "message": "the pair could not be scored"})
            continue
        words = set(query.lower().split())
        scores, logits, tokens, cuts = [], [], [], []
        refused = None
        for index, document in enumerate(documents):
            text = document.split()
            count = len(query.split()) + len(text) + 2
            cut = count > MAX_TOKENS
            if cut and request.get("truncate") is False:
                refused = f'document {index} with the query is longer than {MAX_TOKENS} tokens (send "truncate": true to cut it)'
                break
            if cut:
                text, count = text[:max(0, MAX_TOKENS - 2 - len(query.split()))], MAX_TOKENS
            share = len(words & {w.lower() for w in text}) / len(words) if words else 0.0
            share = min(0.999, max(0.001, share))
            scores.append(share)
            logits.append(math.log(share / (1 - share)))
            tokens.append(count)
            cuts.append(1 if cut else 0)
        if refused:
            send("ERROR", {"id": rid, "message": refused})
            continue
        send("SCORES", {"id": rid, "scores": scores, "logits": logits, "tokens": tokens, "truncated": cuts})


if __name__ == "__main__":
    main()
