# Gemma 4 on colibri

`c/gemma4.c` runs Google's [Gemma 4](https://huggingface.co/collections/google/gemma-4) (Apache 2.0) as plain C with no
dependencies: text, pictures and, on the two small models, audio in; text out. One engine serves all four checkpoints.

| model | what it is | parameters | container | verified |
|---|---|---|---|---|
| [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | mixture of experts, 128 experts, 8 a token | 25.2B, 3.8B a token | 14.8 GB | real weights: text, pictures, tools, thinking |
| [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | dense | 31B | about 33 GB | a tiny model against transformers; not run on the real weights (it needs the machine to itself) |
| [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | dense, per-layer embeddings, audio | 8B with embeddings, 4B in use | about 8 GB | a tiny model against transformers; same code as the E2B |
| [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | dense, per-layer embeddings, audio | 5B with embeddings, 2B in use | 4.8 GB | real weights: text, pictures, audio |

## What the models are

| | |
|---|---|
| layers | 26B-A4B: 30; 31B: 60; E4B: 42; E2B: 35. Every sixth is full attention, the rest are sliding-window attention (window 1024, or 512 on the E models) |
| attention | scores are not scaled (queries and keys are RMS-normalised per head, values without a weight); on the 26B-A4B and the 31B the full layers have keys equal to values and a quarter of the head rotated; sliding and full layers have different head sizes and rope bases |
| feed-forward | 26B-A4B: a dense MLP and 128 experts side by side, normalised, summed and normalised again; the dense models: the MLP alone, twice as wide on the E2B's last layers |
| the E models | per-layer embeddings (a table by token and a projection of the embedding give every layer its own input, gated into the residual); the last 18 layers (E4B) or 20 (E2B) read the keys and values of an earlier layer of the same kind and have none of their own; clipped linears in the towers |
| other | embeddings times sqrt(hidden), a scalar on every layer's output, tied output head, logits capped at 30 |
| pictures | a 27-layer (26B-A4B, 31B) or 16-layer (E models) vision tower, 2D rotary positions, 3 x 3 pooling: a picture is 70 to 1120 soft tokens, 280 by default; inside one picture the sliding layers let rows see each other |
| audio | E2B and E4B only: a conformer (twelve layers, chunked local attention with a relative-position bias), 40 ms a soft token, 30 seconds at most |
| context | 262144 tokens on the large models, 131072 on the E models (the engine's default is 8192; `--ctx` raises it) |
| chat format | `<|turn>` / `<turn|>` turns, a thought channel, tool calls in `<|tool_call>call:f{key:<|"|>value<|"|>}<tool_call|>` |

The reference implementation is transformers' `modeling_gemma4.py`; `c/gemma4.c` follows it line by line, and the tests
below hold it to that.

## Quickstart

Convert a checkpoint (the 26B-A4B is a 52 GB download of bf16 weights; the container is 15 GB; both must fit on the disk
while it runs). The converter needs only numpy, and `huggingface_hub` for `--repo`:

```sh
pip install numpy huggingface_hub
python3 c/tools/convert_gemma4.py --repo google/gemma-4-26B-A4B-it --out ~/Models/gemma4-26b-a4b
python3 c/tools/convert_gemma4.py --repo google/gemma-4-E2B-it --out ~/Models/gemma4-e2b
# or, from a checkpoint already on disk:
python3 c/tools/convert_gemma4.py --src /path/to/gemma-4-26B-A4B-it --out ~/Models/gemma4-26b-a4b
```

The towers come with it (`--no-vision` and `--no-audio` leave them out). A container made before the towers were carried
gets them in a minute without converting the rest: `--add-vision`, `--add-audio`.

Build and chat:

```sh
make -C c gemma4
./c/coli chat --model ~/Models/gemma4-26b-a4b
./c/coli serve --model ~/Models/gemma4-26b-a4b          # OpenAI- and Anthropic-compatible API
```

`coli router` serves it beside the other installed models like any other chat model, and the model card says `image` (and
`audio`, on the E models) when the engine loaded the tower.

## Pictures and audio

Send them as OpenAI content parts, or Anthropic image blocks:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What is written here?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "input_audio", "input_audio": {"data": "<base64 of a WAV file>", "format": "wav"}}]}
```

Pictures are read by the gateway the way Google's image processor reads them (`c/tools/gemma4_image.py`: resized to the
largest size in multiples of 48 pixels that fits the budget, transparency laid on white, cut into 16-pixel patches); a clip
is read as WAV (8 to 32 bits or float, any channels, any rate; mixed down and resampled to 16 kHz) and turned into the model's
128-bin log-mel frames (`c/tools/gemma4_audio.py`). The prompt gets the model's own begin, one placeholder for every soft token and end
tokens where the chat template writes its marker, the engine gets an `IMAGE` or `AUDIO` frame for each, in the order of the
conversation, and answers an unreadable or mismatched one by name with a 400. `GET /v1/models` says what the loaded engine takes.

A picture costs its soft tokens of context (280 by default) and a conversation can carry several. `G4_IMG_TOKENS` (the
engine, default 280, up to 4096) is the biggest picture it accepts and also the room it keeps for one; the gateway picks the
largest budget of the processor's 70, 140, 280, 560 and 1120 that fits it. More tokens read small print better and take
longer. `G4_AUDIO_TOKENS` (default 750, thirty seconds) is the longest clip; a longer one is cut.

The cache knows pictures and clips apart: the same words after a different picture are not served from the first picture's
cache, the same picture again is not read twice, and a conversation extended with a new turn keeps the picture it had.

Not supported: video, audio other than WAV, and audio on the 26B-A4B and the 31B (their checkpoints have no audio tower).

## What is in the container

Dense matrices (attention, the dense MLP, the embedding table that is also the output head, the vision and audio towers'
matrices) are per-row int8, always resident. On the 26B-A4B each layer's 128 experts are one tensor of equal records in the
planar int4 group-of-64 layout `c/expert_ffn.h` reads (the kernel Qwen3.6 and OLMoE use; it learned GELU for this): 3.35 MB an
expert, 12.9 GB for all 3840. Norms, the router, the scalars, the towers' convolutions and position tables and a clipped
linear's four limits stay in float32. Keep the container on a fast disk: on a hard disk the experts that are not cached wait
for it.

## Memory

The 26B-A4B caches experts per layer, `--cap` of them. With no `--cap` it sizes the cache from the free memory when it
starts, and says what it chose; all 128 per layer need about 13 GB. Fewer streams the rest from the disk with `pread`, which
is what a small machine does at a few tokens a second. The dense models have nothing to cache: every weight is read for every
token, so they run at the speed of the machine's memory.

The keys and values of one conversation are float32. The sliding layers keep a ring of window + the biggest chunk (256 rows,
or a picture) + 2048 positions, however long the conversation, resident as it is used; only the full layers keep every
position, and the layers that share keys and values keep none:

| model, context | one conversation's cache |
|---|---|
| 26B-A4B, 8192 | 1.7 GB |
| 26B-A4B, 32768 | 2.7 GB |
| 26B-A4B, 262144 | 12.1 GB |
| 31B, 8192 | 6.8 GB |
| E4B, 8192 | 0.5 GB |
| E2B, 8192 | 0.17 GB |

`--kv-slots N` keeps N conversations apart; the gateway puts each on the slot that holds its earlier turns (docs/api.md).

## Chat, thinking and tools

The gateway renders the model's own chat template (`c/gemma4_chat.py`, tested byte for byte against the releases'
templates: the 26B-A4B's and 31B's are one, the E2B's and E4B's differ only in having no empty thought channel on the cue
when thinking is off, which the gateway reads from the container's own `chat_template.jinja`). Thinking is off by default, as in
the template; `enable_thinking` or a `reasoning_effort` turns it on, and the reasoning comes back as `reasoning_content`.
Tools work through the OpenAI `tools` field and the Anthropic `tool_use` blocks; tool-call arguments are read from Gemma's own
syntax and returned as JSON. `tool_choice` `none` is honoured; `required` and a named tool are instructions the model may
ignore, as on the other engines.

## A conversation sent again

The template drops the empty thought channel (and a past turn's reasoning) from the history, so a resent conversation differs
from what the engine fed, just after `<|turn>model`. The other engines keep their cache only when the new prompt begins with
everything they fed, which here would read the whole conversation again on every turn. This engine keeps the part they share
and forgets the rest, as long as the sliding layers still hold the window before the point where they differ (`G4_SLACK`,
2048 positions beyond the window and a prompt chunk). A long reply, or a long block of reasoning, beyond that is read again.

## Several conversations at once

Every conversation that is decoding takes its step in one pass over the weights, so the weights are read once for all of
them, not once for each. A conversation gets exactly the logits it would alone, bit for bit (a test compares every logit of
seven conversations of different lengths stepping alone and together, with one thread and eight, and with expert caches too
small for the batch): a reply does not depend on who else is asking. On the real E2B, on a machine busy with something else,
one conversation decoded at about 9 tokens a second and four together at about 19 in all.

## How it was checked

- **Against transformers.** `c/tools/make_gemma4_tiny.py` builds three tiny random Gemma 4 models, each with every part of its
  real counterpart: the mixture of experts (26B-A4B), a dense one with keys equal to values (31B), and one with per-layer
  embeddings, shared keys and values, a double-wide MLP, clipped linears and an audio tower (E2B). Each runs in transformers in
  float32 with its weights already rounded to what the container holds, so the engine and the reference multiply the same
  numbers. `tests/test_gemma4.py` requires the engine's logits to agree to float rounding (text 1.5e-5 to 4e-5; the vision
  tower's soft tokens 7e-6; the audio tower's 2e-6; whole prompts with pictures and clips 1e-5 to 1e-4), the argmax of every row
  and a greedy continuation to be identical, with the whole prompt at once, one row at a time, in chunks that wrap the ring,
  with pictures and clips split across chunks, and with an expert cache too small for a layer's routing. The same tests run
  on all three. The reference pads pictures and clips and masks the padding; the engine pads nothing.
- **Mutations.** `c/tools/mutate_gemma4.py` breaks the engine 83 ways (attention scaled, the window one position off, a missing
  norm, SiLU for GELU, the wrong expert served, a cache reused one position too far, a picture's rows read causally, the
  tower's positions swapped, a clip's attention window one key too long, a batched step using the wrong kernel, ...); the tests
  catch all of them. Three are equivalent in exact arithmetic: renormalising the router's top-k weights, the 1/sqrt(2) on a
  per-layer input, and the cancel check inside a batched step each have a norm or the next turn of the loop that undoes them.
  The first is caught only by the norm's epsilon, by a hair (2.02e-3 against the tests' 2e-3).
- **Tokenizer.** `c/gemma4_tok.h` (SentencePiece-style BPE with byte fallback, which `tok.h` cannot do) gives the ids of the
  `tokenizers` library on 323 inputs, including emoji, runs of spaces and newlines, bytes with no vocabulary entry and the
  special tokens, and 18288 tokens of source code.
- **The image processor and the audio front end.** `tests/test_gemma4_image.py` compares the patches with transformers'
  `Gemma4ImageProcessorPil` byte for byte on nine sizes and modes (RGBA and palette pictures, thin strips, small pictures
  scaled up); `tests/test_gemma4_audio.py` compares the log-mel frames with `Gemma4AudioFeatureExtractor` exactly on seven
  durations from 50 ms to 30 s and the soft-token counts with the processor's.
- **The real 26B-A4B**, converted (15 GB from 52 GB, 20 minutes, mostly the reading of a USB disk) and run on an i9-14900KF with
  8 of its threads, every expert of every layer cached: a one-sentence answer correct and stopped by the model's own end of
  turn; a paragraph at **14.2 tokens a second** over 300 tokens; a 3509-token prompt read in 90.6 s (**39 tokens a second**, cold
  experts included) and the same prompt again in 0.1 s; a code word placed at the start of a 3449-token prompt, beyond the
  1024-token window, recalled exactly; a tool call parsed into `tool_calls`; thinking on, the reasoning in its channel; a second
  turn that kept the 22 tokens it shared with the first. And a picture of a blue square, a red circle and the words "HELLO 42":
  described correctly, the text read back exactly, both colours named, the shapes counted, with the follow-up questions about
  the same picture answered in 1 to 3 seconds from the cache.
- **The real E2B**, 4.8 GB: the same picture described and its text read; a sentence answered at about 21 tokens a second on 6
  threads; and eleven seconds of speech ("And so my fellow Americans, ask not what your country can do for you, ask what you can
  do for your country") transcribed word for word, the tower taking 0.9 s.

```sh
python3 c/tools/make_gemma4_tiny.py --reference   # recompute the references (torch + transformers)
python3 -m unittest tests.test_gemma4 tests.test_gemma4_chat_template tests.test_gemma4_gateway tests.test_gemma4_image tests.test_gemma4_audio
python3 c/tools/mutate_gemma4.py [part of a name ...]
```

## Settings

| Variable | Default | |
|---|---|---|
| `CTX` | 8192 | context window in tokens (`--ctx`) |
| `KV_SLOTS` | 1 | conversations kept apart (`--kv-slots`, up to 16) |
| `G4_SLACK` | 2048 | positions beyond window + chunk the sliding layers keep, for rewinding a conversation |
| `G4_TOP_K` | 64 | sampling keeps this many candidates when the temperature is above zero (the model's recommendation) |
| `G4_IMG_TOKENS` | 280 | the biggest picture the engine takes, in soft tokens, and the room it keeps for one |
| `G4_AUDIO_TOKENS` | 750 | the longest clip the engine takes, in soft tokens (40 ms each) |
| `G4_VISION`, `G4_AUDIO` | on | `0` does not load that tower: less memory, and a picture or a clip gets a 400 |
| `G4_MAX_IMAGE_TOKENS` | the engine's | the gateway uses a smaller budget than the engine allows, to read pictures faster |
| `SEED` | the clock | sampling seed |
| `COLI_KV_PREFIX` | on | `0` never reuses a cache |
| `COLI_PREFIX_LOG` | off | `1` says what each request reused, and how long each tower took |

## What is not there

- **Video, and audio that is not WAV.** The models read video as frames; mp3, flac and ogg are not decoded here.
- **A GPU path.** CPU only. The Vulkan backend's `coli_vk_matmul` computes exactly what this container's dense matrices need
  (int8 rows, one scale per row), so offloading them to a graphics card is a short step and the experts would stay on the CPU.
  It is not done: the engine would have to be built against that backend, and how much a launch for each of about 200 matrices
  a token gains over the CPU has not been measured. The on-device chains the larger families have are a project of their own.
- **A layer-segment adapter** (`docs/segment-runtime.md`): the family is exempt from that gate until it has one.
- **The 31B and E4B on real weights.** Their structure is checked against transformers on tiny models and shares every line
  with the models that were run, but neither was run here (the 31B in int8 needs 33 GB beside whatever else is loaded; the E4B
  was not downloaded). The first run on one of them should be a short question and a look at the answer.
- **Speed of the vision tower on a big picture.** A 280-token picture is 2400 patches through 27 layers: tens of seconds on the
  26B-A4B's tower on a busy machine, about eight seconds on the E models' smaller one. It runs once per picture and the cache
  keeps it.
