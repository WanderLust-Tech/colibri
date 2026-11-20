# Rerank

`POST /v1/rerank` sorts documents by how well they answer a query. It is the second half of search over your own notes, code or
documents: [embeddings](embeddings.md) find a few dozen candidates quickly, and a reranker reads each candidate together with the
question and orders them properly. Colibri answers it with **Qwen3-Reranker** (0.6B, 4B and 8B; Apache-2.0), in the request and
response shape Jina's and Cohere's rerank clients use (`/v1/rerank` and Cohere's `/v2/rerank` are the same endpoint), so the tools
written for those (LangChain, LlamaIndex, Open WebUI, Continue) work unchanged.

There is no new binary. `c/qwen3embed` runs a reranker when its checkpoint is one (the folder has `1_LogitScore/config.json`): the
decoder, the batching and the attention are the embedding model's, and the new code is the pair prompt, the two-row output head and
the protocol.

## Get one

```
coli model install qwen3-reranker-0.6b      1.2 GB; the 4B (8.1 GB) and the 8B (16.4 GB) are in the catalog too
```

or the **Models** tab of the dashboard, or the models menu of the editor extension ([model-install.md](model-install.md)). With
`coli router` the model is listed next to the others and starts the first time something asks it to rank; with `coli serve --model DIR`
it is the one model served.

## Use it

```bash
curl http://127.0.0.1:8000/v1/rerank -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{
  "model": "qwen3-reranker-0.6b",
  "query": "What is the capital of China?",
  "documents": ["Gravity is a force that attracts two bodies towards each other.", "The capital of China is Beijing."],
  "top_n": 1, "return_documents": true }'
```

```json
{"id": "req_...", "object": "rerank", "model": "qwen3-reranker-0.6b",
 "results": [{"index": 1, "relevance_score": 0.9995, "document": {"text": "The capital of China is Beijing."}}],
 "usage": {"prompt_tokens": 169, "total_tokens": 169}}
```

| Field | |
|---|---|
| `query` | the text the documents are judged against (Jina's `{"text": ...}` form is taken too) |
| `documents` | a list of up to 2048 strings, or of objects with a `text` string (Cohere's long form). An empty string is a document like any other. |
| `top_n` | keep only the best N results. Every document is still read, so `usage` counts them all. |
| `return_documents` | echo each result's text as `document.text` (default `false`) |
| `instruction` | *Colibri's addition.* The task the model judges by. The default is web search: "Given a web search query, retrieve relevant passages that answer the query". See below. |
| `truncate` | *Colibri's addition.* Default `true`: a document that makes its pair longer than the model takes is cut, and `colibri.truncated` lists which. `false` refuses it with a 400. |

`results` come best first; two documents with the same score keep the order they came in. `index` is the document's position in
`documents`. `relevance_score` is from 0 to 1: the model is asked whether the document meets the requirements of the query and told to
answer "yes" or "no", and the score is its chance of "yes" against "no" (`sigmoid(logit yes - logit no)`, what the model card
computes). It is a probability the model gives, not a calibrated one: use it to **order** documents, and treat a threshold on it as
something to measure on your own data.

`GET /v1/models` says what a loaded model takes: `capabilities: ["rerank"]` and `rerank: {max_tokens, max_documents}`. A reranking model
answers 400 on the chat, completions, images, music and embeddings routes, and every other model answers 400 here, each with a pointer to
the right route. The dashboard leaves reranking models out of the conversation's model list.

## The task instruction

The model is instruction-aware: the pair it reads is

```
<Instruct>: Given a web search query, retrieve relevant passages that answer the query
<Query>: What is the capital of China?
<Document>: The capital of China is Beijing.
```

inside a fixed system prompt and assistant turn (`c/qwen3embed.c` has the exact text, which is the model card's). Send `instruction` to
say what "relevant" means for your search (`"Given a question about the codebase, find the function that implements it"`); the model
card says a task that fits is worth 1 to 5 %, and that English instructions work best even for other languages. Use the same
instruction for every request of one search.

## Limits

* **8192 tokens a pair**: the fixed prompt (about 60 tokens), the instruction, the query and the document together. A long document is
  cut at its end, the way the model card does, unless `truncate` is `false`. Rerank passages, not whole files.
* **2048 documents a request**; the body the gateway reads is at most 4 MiB. Rerank what the embedding search returned (tens, not
  thousands): every document costs a full run of the model.
* One request at a time per model, in order; several waiting are queued like chats are (the usual 429 when the queue is full).
* Float32 on the CPU. No GPU path yet.

## How fast, and how big

Measured on the 14900KF (24 cores, 32 threads), CPU, the whole request including its tokenizing. The pairs are a short query and a
passage, so each is about 107 tokens (the fixed prompt is a good half of that) or about 290 for the longer passages:

| Model | Weights in RAM | one pair | 64 pairs, ~107 tokens each | 32 pairs, ~290 tokens each |
|---|---|---|---|---|
| Qwen3-Reranker-0.6B | 1.2 GB | 0.2 s | 10.6 s (165 ms a pair) | 15.5 s (484 ms a pair) |
| Qwen3-Reranker-4B | 8.0 GB | 0.8 s | 43.5 s (680 ms a pair) | 63.8 s (2.0 s a pair) |
| Qwen3-Reranker-8B | 15.1 GB | 1.4 s | 72.2 s (1.1 s a pair) | 101 s (3.2 s a pair) |

About 650 tokens a second on the 0.6B, 160 on the 4B and 95 on the 8B.

A request is a batch: the pairs go through the layers together, up to 4096 tokens at a time, so the weights are read once for the batch
and **a list is much cheaper per document than one document at a time**. What would help most is not recomputing the part every pair
shares (the system prompt, the instruction and the query are the same for all documents of a request and a good half of a short
passage's tokens): not done, see below.

## Checked

Against transformers (the model card's recipe: the pair prompt, the yes and no logits at the last position) on the real checkpoints.
The 0.6B in fp32 agrees with it to 1.6e-5 on the logit difference; the 4B and the 8B, whose fp32 references need 16 and 32 GB, against
a bf16 reference, to the rounding of bf16 (logits are quantised to 1/16 or 1/8 there). The token counts are equal in every case, so
the prompt, the tokenizer and the truncation agree with the reference. The nine cases: the model card's two documents against two
queries, a custom task, French with Chinese and an emoji, code with tabs and newlines, the empty document and a 450-token pair. The 8B
is the one whose output matrix is not tied to its embedding table (it has its own `lm_head.weight`, in the last of five shards): the
engine reads the two rows it needs from whichever it finds, and that path is checked on the 8B.

`tests/test_rerank_gateway.py` (33 tests) runs a real gateway against `tools/rerank_stub.py`, a stand-in that speaks the engine's
protocol: the request rules, sorting and ties, `top_n`, `return_documents`, the Cohere path, truncation both ways, a failing engine,
a dead one, requests at the same time and the refusals on the other routes. `tests/test_qwen3rerank_real.py` (6) compares the real
engine with stored transformers values (`tools/qwen3_reranker_0.6b_reference.json`, made by `tools/rerank_ref.py capture`); it needs
the model, so it is skipped without `QWEN3RERANK_MODEL` pointing at a folder with the 0.6B reranker.

## The engine's protocol

`qwen3embed --model DIR --serve` on a reranker checkpoint speaks lines of a keyword and a JSON object on stdin and stdout (the header
of `c/qwen3embed.c` is the reference): `READY {model, task: "rerank", max_tokens, layers, max_documents}` once the weights are loaded,
then for each `RERANK {id, query, documents, instruction, truncate}` exactly one `SCORES {id, scores, logits, tokens, truncated}` or
`ERROR {id, message}`. `qwen3embed --model DIR --query "..." --text "..."` prints the line for one pair, for trying a checkpoint.

## Not done

* **Sharing the prompt.** All the pairs of a request start with the same system prompt, instruction and query. Computing their keys
  and values once and letting every document attend to them would cut the work of a short-passage request roughly in half.
* A GPU path (the Vulkan chain that Z-Image and Gemma 4 use).
* Other rerankers (BGE, Jina), and `coli rerank` for the terminal.
