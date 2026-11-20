#!/usr/bin/env python3
"""The embedding engine from Python: its serve protocol, and the request rules of POST /v1/embeddings and POST /v1/rerank.

Used by the gateway (openai_server.py). Stdlib only. The engine is c/qwen3embed (docs/embeddings.md, docs/rerank.md); the
protocol is the header of c/qwen3embed.c, the same shape as the image and music engines' (lines of a keyword and a JSON
object):

    READY {...}                      once, after the weights are loaded: model, dimensions, max_tokens, layers
    EMBED {...}                      client, one request at a time: id, inputs (strings), instruction, dimensions, truncate
    VECTORS {...}                    id, dimensions, tokens, truncated, vectors (one unit vector per input)
    ERROR {...}                      the engine stays alive for the next EMBED

and, when the checkpoint is a Qwen3-Reranker (RerankEngine):

    READY {...}                      model, task "rerank", max_tokens, layers, max_documents
    RERANK {...}                     id, query, documents (strings), instruction, truncate
    SCORES {...}                     id, scores (0..1, one per document), logits, tokens, truncated

A request is answered in OpenAI's shape ({"object": "list", "data": [{"object": "embedding", ...}], "usage": ...}) so
the clients written for it (the openai libraries, LangChain, LlamaIndex, Open WebUI, Continue) work unchanged.
"""
import base64
import json
import os
import struct
import subprocess
import sys
import threading

from image_engine import _parse_line, describe_exit, engine_command

MAX_INPUTS = 2048                 # the engine's own limit
MAX_INSTRUCTION_BYTES = 4096
ENCODING_FORMATS = (None, "float", "base64")
DEFAULT_INFO = {"model": "embedding", "dimensions": 1024, "min_dimensions": 1, "max_tokens": 8192}
DEFAULT_RERANK_INFO = {"model": "reranker", "max_tokens": 8192, "max_documents": MAX_INPUTS}
MAX_QUERY_BYTES = 65536            # a query is a sentence or a paragraph; a document may be much longer


class EmbeddingEngineError(RuntimeError):
    """The engine refused a request, or the process is gone."""


class EmbeddingRequestError(ValueError):
    """A request the gateway refuses before the engine sees it (400). `param` names the field."""

    def __init__(self, message, param=None):
        super().__init__(message)
        self.param = param


class EmbeddingEngine:
    """One engine process and the one request it can work on at a time.

    The engine answers each EMBED with exactly one line (VECTORS or ERROR), so a request is a write and a read under a
    lock: no reader thread. A request cannot be interrupted once it is running (a few hundred tokens a second on the
    CPU, so a long document is minutes at most); a caller that finds its client gone before the request started skips it."""

    modality = "embedding"
    chats = False                  # no generating endpoint
    decides = False
    label = "embedding engine"     # how the engine is named in messages
    log_tag = "embeddings"
    default_info = DEFAULT_INFO

    def __init__(self, executable, model, env=None, stderr=None, load_timeout=None, on_log=None):
        self.model_dir = str(model)
        self.executable = str(executable)
        self.on_log = on_log or (lambda message: sys.stderr.write(f"[{self.log_tag}] {message}\n"))
        child_env = dict(os.environ if env is None else env)
        # SNAP and SERVE=1 are how `coli stop` recognises an engine left running for a port.
        child_env["SNAP"] = str(model)
        child_env["SERVE"] = "1"
        # Its own process group: Ctrl-C in the terminal reaches the gateway, not the engine in the middle of a request.
        # Losing the gateway still ends the engine: its stdin closes, and EOF is the clean exit.
        spawn = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                 if sys.platform == "win32" else {"start_new_session": True})
        try:
            self.process = subprocess.Popen(
                engine_command(executable, model), env=child_env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=stderr, bufsize=0, **spawn)
        except OSError as error:
            raise EmbeddingEngineError(f"cannot start the {self.label} {executable}: {error}") from error
        self.lock = threading.Lock()
        self.dead = None                      # why the engine is gone, once it is
        self._seq = 0
        try:
            self.info = self._await_ready(load_timeout)
        except BaseException:
            self._kill_quietly()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass
            raise

    def _await_ready(self, timeout):
        timer = None
        if timeout:
            timer = threading.Timer(timeout, self._kill_quietly)
            timer.daemon = True
            timer.start()
        try:
            while True:
                raw = self.process.stdout.readline()
                if not raw:
                    try:
                        code = self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        code = None
                    if timer is not None and not timer.is_alive() and timeout:
                        raise EmbeddingEngineError(f"the {self.label} did not become ready within {timeout:g}s")
                    raise EmbeddingEngineError(f"the {self.label} exited while loading ({describe_exit(code)})")
                keyword, body = _parse_line(raw)
                if keyword == "READY" and body is not None:
                    info = dict(self.default_info)
                    info.update({k: v for k, v in body.items() if v is not None})
                    return info
                if keyword == "ERROR" and body is not None:
                    self._kill_quietly()
                    raise EmbeddingEngineError(f"the {self.label} failed to load: {body.get('message') or 'unknown error'}")
                self.on_log("ignoring engine output before READY: " + raw.decode("utf-8", "replace").strip()[:200])
        finally:
            if timer is not None:
                timer.cancel()

    def _kill_quietly(self):
        try:
            self.process.kill()
        except OSError:
            pass

    @property
    def alive(self):
        return self.dead is None and self.process.poll() is None

    @property
    def dimensions(self):
        return int(self.info["dimensions"])

    def embed(self, inputs, instruction=None, dimensions=None, truncate=True):
        """Run one EMBED and return {"vectors": [[float...]...], "tokens": [int...], "truncated": [bool...],
        "dimensions": int}. Raises EmbeddingEngineError for anything the engine refuses or a process that dies."""
        request = {"inputs": list(inputs)}
        if instruction:
            request["instruction"] = instruction
        if dimensions:
            request["dimensions"] = int(dimensions)
        if truncate is False:
            request["truncate"] = False
        body = self._exchange("EMBED", "VECTORS", request)
        vectors = body.get("vectors")
        tokens = body.get("tokens")
        if (not isinstance(vectors, list) or len(vectors) != len(request["inputs"])
                or not isinstance(tokens, list) or len(tokens) != len(vectors)):
            raise EmbeddingEngineError(f"the {self.label} sent a malformed VECTORS line")
        cut = body.get("truncated") or [0] * len(vectors)
        return {"vectors": vectors, "tokens": [int(n) for n in tokens],
                "truncated": [bool(c) for c in cut], "dimensions": int(body.get("dimensions") or self.dimensions)}

    def _exchange(self, verb, answer, request):
        """One request line to the engine and its one answer line back (the keyword `answer`, or ERROR). Returns the answer's
        JSON object. Raises EmbeddingEngineError for an ERROR line and for a process that is gone."""
        with self.lock:
            if self.dead:
                raise EmbeddingEngineError(self.dead)
            self._seq += 1
            rid = f"e{self._seq}"
            data = f"{verb} {json.dumps({'id': rid, **request}, ensure_ascii=False, separators=(',', ':'))}\n".encode("utf-8")
            try:
                view = memoryview(data)
                while view:
                    sent = self.process.stdin.write(view)
                    if not sent:
                        raise OSError("the engine's stdin accepted no bytes")
                    view = view[sent:]
                self.process.stdin.flush()
            except (OSError, ValueError) as error:
                self.dead = f"cannot send {verb} to the {self.label} ({error})"
                raise EmbeddingEngineError(self.dead) from error
            while True:
                raw = self.process.stdout.readline()
                if not raw:
                    try:
                        code = self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        code = None
                    self.dead = f"the {self.label} exited ({describe_exit(code)})"
                    raise EmbeddingEngineError(self.dead)
                keyword, body = _parse_line(raw)
                # An ERROR with no id is the engine's answer to a line it could not read (too long, not JSON, a message it
                # does not know): requests go one at a time, so it can only be about this one.
                if (body is None or keyword not in (answer, "ERROR")
                        or body.get("id") != rid and not (keyword == "ERROR" and body.get("id") in ("", None))):
                    self.on_log("ignoring stray engine output: " + raw.decode("utf-8", "replace").strip()[:200])
                    continue
                if keyword == "ERROR":
                    raise EmbeddingEngineError(str(body.get("message") or "the engine refused the request"))
                return body

    def close(self, timeout=10):
        """Ends the engine: EOF on its stdin is its clean exit (after the request it is working on); one that
        has not gone after `timeout` seconds is killed."""
        try:
            self.process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_quietly()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for stream in (self.process.stdout, self.process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass


class RerankEngine(EmbeddingEngine):
    """The same process wrapper over a Qwen3-Reranker checkpoint: RERANK in, SCORES out (c/qwen3embed decides by the
    checkpoint which of the two it is)."""

    modality = "rerank"
    label = "reranking engine"
    log_tag = "rerank"
    default_info = DEFAULT_RERANK_INFO

    def rerank(self, query, documents, instruction=None, truncate=True):
        """Run one RERANK and return {"scores": [float...], "logits": [float...], "tokens": [int...], "truncated": [bool...]},
        one entry per document in the order given. Raises EmbeddingEngineError for anything the engine refuses."""
        request = {"query": query, "documents": list(documents)}
        if instruction:
            request["instruction"] = instruction
        if truncate is False:
            request["truncate"] = False
        body = self._exchange("RERANK", "SCORES", request)
        scores, tokens = body.get("scores"), body.get("tokens")
        if (not isinstance(scores, list) or len(scores) != len(request["documents"])
                or not isinstance(tokens, list) or len(tokens) != len(scores)):
            raise EmbeddingEngineError(f"the {self.label} sent a malformed SCORES line")
        cut = body.get("truncated") or [0] * len(scores)
        logits = body.get("logits")
        return {"scores": [float(x) for x in scores], "tokens": [int(n) for n in tokens],
                "logits": [float(x) for x in logits] if isinstance(logits, list) and len(logits) == len(scores) else None,
                "truncated": [bool(c) for c in cut]}


# ---------------------------------------------------------------- requests
def embedding_request(body, info=None):
    """The checked request of POST /v1/embeddings (OpenAI's body, plus `instruction` and `truncate`), or EmbeddingRequestError.

    `input` is a string or a list of strings (token arrays are not taken: there is no tokenizer in the gateway, and the
    ids of another model's tokenizer would mean nothing here). `dimensions` keeps the first N components of each vector
    and normalises again, 1 to the model's width. `instruction` turns each input into a query for that task
    ("Instruct: {instruction}\\nQuery:{input}"); leave it out for documents. `truncate` (default true) cuts an input
    longer than the engine's limit; false refuses it."""
    info = {**DEFAULT_INFO, **(info or {})}
    if not isinstance(body, dict):
        raise EmbeddingRequestError("The request body must be a JSON object.")
    value = body.get("input")
    if value is None:
        raise EmbeddingRequestError("`input` is required: a string, or a list of strings.", "input")
    inputs = [value] if isinstance(value, str) else value
    if not isinstance(inputs, list) or not inputs:
        raise EmbeddingRequestError("`input` must be a string or a non-empty list of strings.", "input")
    if any(isinstance(item, (int, list)) and not isinstance(item, bool) for item in inputs):
        raise EmbeddingRequestError("`input` takes text, not token arrays: this model has its own tokenizer.", "input")
    if len(inputs) > MAX_INPUTS:
        raise EmbeddingRequestError(f"`input` holds {len(inputs)} items; at most {MAX_INPUTS} per request.", "input")
    if not all(isinstance(item, str) for item in inputs):
        raise EmbeddingRequestError("Every item of `input` must be a string.", "input")
    fmt = body.get("encoding_format")
    if fmt not in ENCODING_FORMATS:
        raise EmbeddingRequestError('`encoding_format` must be "float" or "base64".', "encoding_format")
    dimensions = body.get("dimensions")
    width = int(info["dimensions"])
    if dimensions is not None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise EmbeddingRequestError("`dimensions` must be an integer.", "dimensions")
        if not int(info.get("min_dimensions") or 1) <= dimensions <= width:
            raise EmbeddingRequestError(f"`dimensions` must be between {info.get('min_dimensions') or 1} and {width} "
                                        "(this model's width).", "dimensions")
    instruction = body.get("instruction")
    if instruction is not None:
        if not isinstance(instruction, str):
            raise EmbeddingRequestError("`instruction` must be a string.", "instruction")
        if len(instruction.encode("utf-8")) > MAX_INSTRUCTION_BYTES:
            raise EmbeddingRequestError(f"`instruction` is over {MAX_INSTRUCTION_BYTES} bytes.", "instruction")
    truncate = body.get("truncate", True)
    if not isinstance(truncate, bool):
        raise EmbeddingRequestError("`truncate` must be a boolean.", "truncate")
    user = body.get("user")
    if user is not None and not isinstance(user, str):
        raise EmbeddingRequestError("`user` must be a string.", "user")
    return {"inputs": inputs, "encoding_format": fmt or "float", "dimensions": dimensions,
            "instruction": instruction or None, "truncate": truncate}


def pack_vector(vector):
    """base64 of the vector as little-endian float32: OpenAI's encoding_format "base64"."""
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def embedding_payload(request, result, model_id):
    """The response body for a finished request."""
    base64_out = request["encoding_format"] == "base64"
    data = [{"object": "embedding", "index": index,
             "embedding": pack_vector(vector) if base64_out else vector}
            for index, vector in enumerate(result["vectors"])]
    total = sum(result["tokens"])
    payload = {"object": "list", "data": data, "model": model_id,
               "usage": {"prompt_tokens": total, "total_tokens": total}}
    cut = [index for index, flag in enumerate(result["truncated"]) if flag]
    if cut:
        payload["colibri"] = {"truncated": cut}       # which inputs were longer than the engine takes and were shortened
    return payload


def rerank_request(body, info=None):
    """The checked request of POST /v1/rerank (the body Cohere's and Jina's rerank endpoints take, plus `instruction` and
    `truncate`), or EmbeddingRequestError. `documents` is a list of strings, or of objects with a `text` field (Cohere's
    long form); `top_n` keeps only the best N in the answer; `return_documents` echoes each text back."""
    info = {**DEFAULT_RERANK_INFO, **(info or {})}
    if not isinstance(body, dict):
        raise EmbeddingRequestError("The request body must be a JSON object.")
    query = body.get("query")
    if isinstance(query, dict):
        query = query.get("text")                          # Jina's long form
    if not isinstance(query, str) or not query.strip():
        raise EmbeddingRequestError("`query` is required: the text to judge the documents against.", "query")
    if len(query.encode("utf-8")) > MAX_QUERY_BYTES:
        raise EmbeddingRequestError(f"`query` is over {MAX_QUERY_BYTES} bytes.", "query")
    value = body.get("documents")
    if not isinstance(value, list) or not value:
        raise EmbeddingRequestError("`documents` is required: a non-empty list of strings.", "documents")
    limit = int(info.get("max_documents") or MAX_INPUTS)
    if len(value) > limit:
        raise EmbeddingRequestError(f"`documents` holds {len(value)} items; at most {limit} per request.", "documents")
    documents = []
    for item in value:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            item = item["text"]
        if not isinstance(item, str):
            raise EmbeddingRequestError("Every item of `documents` must be a string (or an object with a `text` string).", "documents")
        documents.append(item)
    top_n = body.get("top_n")
    if top_n is not None:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise EmbeddingRequestError("`top_n` must be a positive integer.", "top_n")
    return_documents = body.get("return_documents", False)
    if not isinstance(return_documents, bool):
        raise EmbeddingRequestError("`return_documents` must be a boolean.", "return_documents")
    instruction = body.get("instruction")
    if instruction is not None:
        if not isinstance(instruction, str):
            raise EmbeddingRequestError("`instruction` must be a string.", "instruction")
        if len(instruction.encode("utf-8")) > MAX_INSTRUCTION_BYTES:
            raise EmbeddingRequestError(f"`instruction` is over {MAX_INSTRUCTION_BYTES} bytes.", "instruction")
    truncate = body.get("truncate", True)
    if not isinstance(truncate, bool):
        raise EmbeddingRequestError("`truncate` must be a boolean.", "truncate")
    return {"query": query, "documents": documents, "top_n": top_n, "return_documents": return_documents,
            "instruction": instruction or None, "truncate": truncate}


def rerank_payload(request, result, model_id, request_id=None):
    """The response body for a finished request: results best first, each with its position in `documents` and a score from
    0 to 1 (the model's chance of "yes" to "does this document answer the query"). Cohere's and Jina's clients read
    `results[].index` and `results[].relevance_score`; `usage.total_tokens` is Jina's."""
    order = sorted(range(len(result["scores"])), key=lambda i: (-result["scores"][i], i))
    if request["top_n"]:
        order = order[:request["top_n"]]
    results = []
    for index in order:
        item = {"index": index, "relevance_score": result["scores"][index]}
        if request["return_documents"]:
            item["document"] = {"text": request["documents"][index]}
        results.append(item)
    total = sum(result["tokens"])
    payload = {"id": request_id or "", "object": "rerank", "model": model_id, "results": results,
               "usage": {"prompt_tokens": total, "total_tokens": total}}
    cut = [index for index, flag in enumerate(result["truncated"]) if flag]
    if cut:
        payload["colibri"] = {"truncated": cut}
    return payload
