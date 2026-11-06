# Gemma 4 (26B-A4B) on colibri

`c/gemma4.c` runs [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
(Apache 2.0): a mixture of experts with 25.2B parameters of which 3.8B are used for each
token, as plain C with no dependencies. Text in, text out: the vision tower in the checkpoint is
not loaded.

## What the model is

| | |
|---|---|
| layers | 30: 25 sliding-window attention (window 1024, 8 KV heads of 256), 5 full attention (2 KV heads of 512, keys equal values, rotary on a quarter of the head), every sixth layer full |
| feed-forward | a dense MLP (GELU) and 128 experts side by side, 8 chosen per token by a router that has its own norm, scale and per-expert scale; both branches are normalised, summed and normalised again |
| other | attention scores are not scaled (queries and keys are RMS-normalised per head), a scalar on every layer's output, embeddings times sqrt(hidden), tied output head, logits capped at 30 |
| context | 262144 tokens (the engine's default is 8192; `--ctx` raises it) |
| chat format | `<|turn>` / `<turn|>` turns, a thought channel, tool calls in `<|tool_call>call:f{key:<|"|>value<|"|>}<tool_call|>` |

The model's reference implementation is transformers' `modeling_gemma4.py`; `c/gemma4.c` follows it
line by line, and the tests below hold it to that.

## Quickstart

Convert the checkpoint (a 52 GB download of bf16 weights, 14 GB afterwards; both must fit on the disk
while it runs). The converter needs only numpy, and `huggingface_hub` for `--repo`:

```sh
pip install numpy huggingface_hub
python3 c/tools/convert_gemma4.py --repo google/gemma-4-26B-A4B-it --out ~/Models/gemma4-26b-a4b
# or, from a checkpoint already on disk:
python3 c/tools/convert_gemma4.py --src /path/to/gemma-4-26B-A4B-it --out ~/Models/gemma4-26b-a4b
```

Build and chat:

```sh
make -C c gemma4
./c/coli chat --model ~/Models/gemma4-26b-a4b
./c/coli serve --model ~/Models/gemma4-26b-a4b          # OpenAI- and Anthropic-compatible API
```

`coli router` serves it beside the other installed models like any other chat model.

## What is in the container

Dense matrices (attention, the dense MLP, the embedding table that is also the output head) are
per-row int8, 2.6 GB and always resident. Each layer's 128 experts are one tensor of equal
records in the planar int4 group-of-64 layout `c/expert_ffn.h` reads (the kernel Qwen3.6 and OLMoE
use; it learned GELU for this): 3.35 MB an expert, 12.9 GB for all 3840. Norms, the router and the
scalars stay in float32.

## Memory

The engine caches experts per layer, `--cap` of them. With no `--cap` it sizes the cache from the
free memory when it starts, and says what it chose; all 128 per layer need about 13 GB.
Fewer streams the rest from the disk with `pread`, which is what a small machine does at a few
tokens a second.

The keys and values of one conversation are float32. The sliding layers keep a ring of
window + 256 + 2048 positions (1.4 GB, however long the conversation, resident as it is used); only the
five full layers keep every position, at 41 KB a token:

| context | one conversation's cache |
|---|---|
| 8192 | 1.7 GB |
| 32768 | 2.7 GB |
| 262144 | 12.1 GB |

That is a fifth of what a model with a KV head in every layer needs, and why a long window is
affordable here. `--kv-slots N` keeps N conversations apart; the gateway puts each on the slot that
holds its earlier turns (docs/api.md).

## Chat, thinking and tools

The gateway renders the model's own chat template (`c/gemma4_chat.py`, tested byte for byte
against the release's `chat_template.jinja` over 21 conversations). Thinking is off by default, as
in the template, which then adds an empty thought channel to the cue so the model answers at once;
`enable_thinking` or a `reasoning_effort` turns it on, and the reasoning comes back as
`reasoning_content`. Tools work through the OpenAI `tools` field and the Anthropic `tool_use` blocks;
tool-call arguments are read from Gemma's own syntax and returned as JSON. `tool_choice` `none` is
honoured; `required` and a named tool are instructions the model may ignore, as on the other engines.

## A conversation sent again

The template drops the empty thought channel (and a past turn's reasoning) from the history, so a
resent conversation differs from what the engine fed, just after `<|turn>model`. The other engines keep
their cache only when the new prompt begins with everything they fed, which here would read the whole
conversation again on every turn. This engine keeps the part they share and forgets the rest, as long as
the sliding layers still hold the window before the point where they differ (`G4_SLACK`, 2048
positions beyond the window and a prompt chunk). A long reply, or a long block of reasoning, beyond that is
read again.

## How it was checked

- **Against transformers.** `c/tools/make_gemma4_tiny.py` builds a random Gemma 4 with every part of
  the real one (both layer kinds with their head sizes, the shared keys and values, proportional rope,
  the router, the layer scalar, the cap) and a window of 6, runs it in transformers in float32, and
  rounds its weights to what the container holds. `tests/test_gemma4.py` requires the engine's logits to
  agree to float rounding (measured: 1.5e-5), the argmax of every row and a greedy continuation to be
  identical, with the whole prompt at once, one row at a time, in chunks that wrap the ring three
  times, and with an expert cache too small for a layer's routing.
- **Mutations.** `c/tools/mutate_gemma4.py` breaks the engine 27 ways (attention scaled, the window one
  position off, a missing norm, SiLU for GELU, the wrong expert served, a cache reused one position
  too far, ...); the tests catch 26. The twenty-seventh is equivalent: renormalising the top-k router
  weights is undone by the RMS norm that follows.
- **Tokenizer.** `c/gemma4_tok.h` (SentencePiece-style BPE with byte fallback, which `tok.h` cannot do)
  gives the ids of the `tokenizers` library on 323 inputs, including emoji, runs of spaces and
  newlines, bytes with no vocabulary entry and the special tokens, and 18288 tokens of source code.
- **The real checkpoint.** @@REAL@@

```sh
python3 c/tools/make_gemma4_tiny.py --reference   # recompute the reference (torch + transformers)
python3 -m unittest tests.test_gemma4 tests.test_gemma4_chat_template tests.test_gemma4_gateway
python3 c/tools/mutate_gemma4.py
```

## Settings

| Variable | Default | |
|---|---|---|
| `CTX` | 8192 | context window in tokens (`--ctx`) |
| `KV_SLOTS` | 1 | conversations kept apart (`--kv-slots`, up to 16) |
| `G4_SLACK` | 2048 | positions beyond window + chunk the sliding layers keep, for rewinding a conversation |
| `G4_TOP_K` | 64 | sampling keeps this many candidates when the temperature is above zero (the model's recommendation) |
| `SEED` | the clock | sampling seed |
| `COLI_KV_PREFIX` | on | `0` never reuses a cache |
| `COLI_PREFIX_LOG` | off | `1` says what each request reused |

## What is not there

- **Pictures and audio.** The checkpoint's vision tower is skipped by the converter (and counted) and a
  request with an image is answered 400.
- **The other Gemma 4 sizes.** E2B, E4B, 12B and the 31B are dense models, which the streaming design has
  nothing to do for.
- **A GPU path.** CPU only.
- **A layer-segment adapter** (`docs/segment-runtime.md`): the family is exempt from that gate until it has one.
- **Decoding several conversations in one step.** Each active request takes its turn, one token at a time;
  the weights are read once per token, not once per step.
