"""`coli embed` and `coli rerank` (docs/embeddings.md, docs/rerank.md): try the retrieval models from the terminal.

Both talk to a running server (`coli router` or `coli serve`) through its API, the same one an application uses. `api(method, path,
body)` is coli's _api_call: it returns (status, parsed JSON) or (None, None) when nothing answers. The functions here print what
they find and return the exit code; nothing here starts an engine."""
import json
import math
import sys

MODEL_WAIT = 900               # seconds: the first request loads the model


class Refused(Exception):
    """A reason to stop that the person can act on (printed as one line)."""


def read_lines(path):
    """The non-empty lines of a file, or of standard input for `-`."""
    try:
        handle = sys.stdin if path == "-" else open(path, encoding="utf-8")
    except OSError as error:
        raise Refused(f"cannot read {path}: {error.strerror or error}")
    with handle:
        return [line.rstrip("\r\n") for line in handle if line.strip()]


def gather(words, file):
    """The words on the command line plus the lines of --file; a lone `-` word means the lines of standard input."""
    texts = [w for w in (words or []) if w != "-"]
    if "-" in (words or []):
        texts += read_lines("-")
    if file:
        texts += read_lines(file)
    return texts


def choose_model(api, capability, wanted, hint):
    """The model id to ask: --model as given, else the first the server lists with that capability."""
    if wanted:
        return wanted
    status, body = api("GET", "/v1/models")
    if status is None:
        raise Refused("nothing answers on that host and port: start the server (coli service start, or coli router)")
    if status == 401:
        raise Refused("the server wants an API key: --api-key, or COLI_API_KEY")
    for item in (body or {}).get("data") or []:
        if capability in (item.get("capabilities") or []):
            return item["id"]
    installs = {"embeddings": "qwen3-embedding-0.6b", "rerank": "qwen3-reranker-0.6b", "transcription": "qwen3-asr-0.6b"}
    raise Refused(f"the server has no model that {hint}: coli model install {installs.get(capability, 'MODEL')}")


def post(api, path, body):
    """The answer of a POST, or Refused with the server's own words."""
    status, answer = api("POST", path, body)
    if status is None:
        raise Refused("nothing answers on that host and port: start the server (coli service start, or coli router)")
    if status == 401:
        raise Refused("the server wants an API key: --api-key, or COLI_API_KEY")
    if status != 200:
        error = (answer or {}).get("error") if isinstance(answer, dict) else None
        message = error.get("message") if isinstance(error, dict) else error
        raise Refused(f"the server said {status}: {message or 'no reason given'}")
    return answer


def excerpt(text, width=70):
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 1] + "…"


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def embed(a, api, C):
    """coli embed TEXT... [--file F] [--model ID] [--dimensions N] [--instruction TASK] [--json]"""
    try:
        texts = gather(a.texts, a.file)
        if not texts:
            raise Refused('usage: coli embed "text" ["another text" ...] [--file FILE] [--model ID] [--dimensions N] [--instruction TASK] [--json]')
        model = choose_model(api, "embeddings", a.model, "turns text into vectors")
        body = {"model": model, "input": texts}
        if a.dimensions:
            body["dimensions"] = a.dimensions
        if a.instruction:
            body["instruction"] = a.instruction
        answer = post(api, "/v1/embeddings", body)
    except Refused as why:
        print(f"  {C.yel}{why}{C.r}")
        return 1
    if a.json:
        print(json.dumps(answer))
        return 0
    rows = sorted(answer.get("data") or [], key=lambda row: row.get("index", 0))
    width = len(rows[0]["embedding"]) if rows else 0
    usage = answer.get("usage") or {}
    kind = "queries" if a.instruction else "documents"
    print(f"  {answer.get('model', model)}: {len(rows)} {kind}, {width} numbers each, {usage.get('prompt_tokens', '?')} tokens")
    for i, row in enumerate(rows):
        head = " ".join(f"{x:+.3f}" for x in row["embedding"][:4])
        print(f"  [{i}] {head} ...   {excerpt(texts[i], 50)}")
    if 2 <= len(rows) <= 10:                      # the vectors are unit length, so a dot product is the cosine
        print(f"\n  {C.dim}how alike they are (1.00 = the same direction):{C.r}")
        print("       " + "".join(f"[{j}]".rjust(7) for j in range(len(rows))))
        for i, row in enumerate(rows):
            cells = "".join(f"{cosine(row['embedding'], other['embedding']):7.2f}" for other in rows)
            print(f"  [{i}] {cells}")
    return 0


def rerank(a, api, C):
    """coli rerank QUERY DOCUMENT... [--file F] [--model ID] [--top N] [--instruction TASK] [--json]"""
    try:
        if not a.query:
            raise Refused('usage: coli rerank "the question" "a document" ["another" ...] [--file FILE] [--model ID] [--top N] [--instruction TASK] [--json]')
        docs = gather(a.documents, a.file)
        if not docs:
            raise Refused("no documents: put them after the query, or give --file FILE (one document a line; - for standard input)")
        model = choose_model(api, "rerank", a.model, "ranks documents")
        body = {"model": model, "query": a.query, "documents": docs}
        if a.top:
            body["top_n"] = a.top
        if a.instruction:
            body["instruction"] = a.instruction
        if a.truncate:
            body["truncate"] = True
        answer = post(api, "/v1/rerank", body)
    except Refused as why:
        print(f"  {C.yel}{why}{C.r}")
        return 1
    if a.json:
        print(json.dumps(answer))
        return 0
    results = answer.get("results") or []
    usage = answer.get("usage") or {}
    print(f"  {answer.get('model', model)}: {len(docs)} documents ranked, {usage.get('total_tokens', '?')} tokens, query: {excerpt(a.query, 50)}")
    for rank, row in enumerate(results, 1):
        score = row.get("relevance_score")
        bar = "#" * int(round(10 * score)) if isinstance(score, (int, float)) and math.isfinite(score) else ""
        print(f"  {rank:>2}  {score:6.3f}  {bar:<10}  [{row['index']}] {excerpt(docs[row['index']])}")
    cut = (answer.get("colibri") or {}).get("truncated") or []          # the positions of the documents that were cut
    if cut:
        print(f"  {C.dim}cut to the model's limit: {', '.join(str(i) for i in cut)}{C.r}")
    return 0
