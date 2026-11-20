# Embeddings

`POST /v1/embeddings` turns text into a vector of numbers, so that texts that mean the same thing end up near each other. It is what
search over your own notes, code or documents is built from: embed every chunk once, embed the question, and take the chunks whose
vectors are closest. Colibri answers it with **Qwen3-Embedding** (0.6B, 4B and 8B; Apache-2.0), in OpenAI's request and response shape,
so the clients written for OpenAI's route (the `openai` libraries, LangChain, LlamaIndex, Open WebUI, Continue) work unchanged.

The engine is `c/qwen3embed`. Its layers are the ones ACE-Step's text encoder already runs (that encoder is Qwen3-Embedding-0.6B), so
the new code is the pooling, the query prefix, the batching and the line protocol.

## Get one

```
coli model install qwen3-embedding-0.6b      1.2 GB; the 4B (8.1 GB) and the 8B (15.2 GB) are in the catalog too
```

or the **Models** tab of the dashboard, or the models menu of the editor extension ([model-install.md](model-install.md)). With
`coli router` the model is listed next to the others and starts the first time something asks it for vectors; with `coli serve --model
DIR` it is the one model served. The engine needs `make qwen3embed` (`coli setup` and the release archives build it).

## Use it

```bash
curl http://127.0.0.1:8000/v1/embeddings -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": ["How do I reset my password?", "Our refund policy"]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=KEY)
vectors = client.embeddings.create(model="qwen3-embedding-0.6b", input=["a first text", "a second text"]).data
```

The answer is `{"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [...]}, ...], "model": ..., "usage":
{"prompt_tokens": N, "total_tokens": N}}`. Every vector has length 1, so the dot product of two of them is their cosine similarity.

| Field | |
|---|---|
| `input` | a string, or a list of up to 2048 strings. Text only: a token array is refused (the model has its own tokenizer). An empty string is a text like any other. |
| `encoding_format` | `float` (default), or `base64` (the vector as little-endian float32, a third of the size on the wire) |
| `dimensions` | keep the first N numbers of each vector and normalise again, 1 up to the model's width (1024, 2560, 4096). The models are trained for this: a shorter vector is cheaper to store and search, and a little less exact. |
| `instruction` | *Colibri's addition.* Makes each input a **query** for that task; leave it out for **documents**. See below. |
| `truncate` | *Colibri's addition.* Default `true`: an input longer than the model takes is cut, and `colibri.truncated` in the answer lists which. `false` refuses it with a 400. |
| `user` | accepted, ignored |

`GET /v1/models` says what a loaded model takes: `capabilities: ["embeddings"]` and `embedding: {dimensions, min_dimensions,
max_tokens}`. An embedding model answers 400 on the chat, images and music routes, and every other model answers 400 here, each with a
pointer to the right route. The dashboard leaves embedding models out of the conversation's model list.

## Queries and documents

The model was trained with a task description in front of the **query** and nothing in front of the **document**:

```
query     Instruct: Given a web search query, retrieve relevant passages that answer the query
          Query:How do I reset my password?
document  To reset your password, open Settings and ...
```

Send `"instruction": "Given a web search query, retrieve relevant passages that answer the query"` with the questions and leave it out
for the chunks you index; the engine writes the two lines above. The model's card says the prefix is worth 1 to 5 % on retrieval, and
that the instruction should name your task (`"Given a question, find the code that implements it"`). Use the same instruction for every
query of one index.

## Limits

* **8192 tokens an input** (the model's context is 32k; the engine stops at 8192 because the cost grows with the square of the length).
  A long document should be cut into chunks anyway: a vector holds one meaning.
* **2048 inputs a request**, and the body the gateway reads is at most 4 MiB.
* One request at a time per model, in order; several waiting are queued like chats are (the usual 429 when the queue is full).
* Float32 on the CPU. No GPU path yet.

## How fast, and how big

Measured on the 14900KF (24 cores, 32 threads), CPU, the whole request including its tokenizing:

| Model | Vector | Weights in RAM | one text, 24 tokens | 64 texts, 24 tokens each | a 880-token text | 8192 tokens |
|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 1024 | 1.2 GB | 0.10 s | 2.6 s | 3.0 s | 99 s |
| Qwen3-Embedding-4B | 2560 | 8.0 GB | 0.51 s | 17.6 s | 13.2 s | 218 s |
| Qwen3-Embedding-8B | 4096 | 15.1 GB | 0.84 s | 31.2 s | 20.9 s | 290 s |

The 8192-token times of the 4B and 8B were taken while other work ran on the machine: read them as upper bounds.

A batch is much cheaper per text than a single text (the weights are read once for the whole batch, and the matrix products work on
all the rows together), so **send a list** when indexing. The batches go through the layers up to 4096 tokens at a time.

## Checked

Against transformers (the model card's recipe: last-token pooling, left padding, L2 normalisation) on the real checkpoints. The 0.6B
and the 4B in fp32 agree with it to about 1e-7 on every component; the 8B, whose fp32 reference needs 32 GB of RAM, against a bf16
reference, with a cosine of 0.9998 or better (the rounding of the reference). The cases: plain text, a query with its instruction,
the empty text, accented, Chinese, Russian and emoji text, code with tabs and newlines, `dimensions` 256, a 300-token text, a batch of
seven texts of very different lengths in one request (each against its own reference) and, on the 0.6B, a text cut at 8192 tokens.
The token counts are equal in every case, so the tokenizer, the end-of-text token and the truncation agree with the reference.

`tests/test_embeddings_gateway.py` (36 tests) runs a real gateway against `tools/embedding_stub.py`, a stand-in that speaks the
engine's protocol: the request rules, order and accounting of a batch, `dimensions`, base64, truncation both ways, a failing engine,
a dead one, requests at the same time, and the refusals on the other routes. `tests/test_qwen3embed_real.py` compares the real engine
with stored transformers vectors; it needs the model, so it is skipped without `QWEN3EMBED_MODEL` pointing at a folder with the 0.6B.

## The engine's protocol

`qwen3embed --model DIR --serve` speaks lines of a keyword and a JSON object on stdin and stdout (the header of `c/qwen3embed.c` is
the reference): `READY` once the weights are loaded, then for each `EMBED {id, inputs, instruction, dimensions, truncate}` exactly one
`VECTORS {id, dimensions, tokens, truncated, vectors}` or `ERROR {id, message}`. `qwen3embed --model DIR --text "..."` prints the
vector of one text, for trying a checkpoint.

## Not done

* A GPU path (the Vulkan chain that Z-Image and Gemma 4 use would make the 4B and 8B several times faster).
* Other embedding models (BGE, e5, nomic), token arrays as `input`, and a vector store: the point of this route is that the stores and
  the frameworks already exist. The reranker that goes with these models is built: [rerank.md](rerank.md).
* `coli embed` for the terminal.
