# Gemma 4 on colibri

`c/gemma4.c` runs Google's [Gemma 4](https://huggingface.co/collections/google/gemma-4) (Apache 2.0) as plain C with no
dependencies: text, pictures and, on the two small models, audio in; text out. One engine serves all four checkpoints.

| model | what it is | parameters | container | verified |
|---|---|---|---|---|
| [gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | mixture of experts, 128 experts, 8 a token | 25.2B, 3.8B a token | 14.8 GB | real weights: text, pictures, tools, thinking |
| [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | dense | 31B | 31.4 GB | real weights (2026-11-10): text and a picture; a tiny model against transformers |
| [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | dense, per-layer embeddings, audio | 8B with embeddings, 4B in use | 8.06 GB | real weights (2026-11-10): text, a picture, audio; a tiny model against transformers |
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
# the same through the CLI (--repo takes a Hugging Face repo or a directory; a directory is read in place and kept):
coli convert --repo google/gemma-4-26B-A4B-it --model ~/Models/gemma4-26b-a4b
```

`coli setup` does not list Gemma: its catalogue holds only downloads that run as they are (a ready container on
Hugging Face, or an official checkpoint the engine reads directly), and Gemma needs this conversion first.

The towers come with it (`--no-vision` and `--no-audio` leave them out). A container made before the towers were carried
gets them in a minute without converting the rest: `--add-vision`, `--add-audio`.

Build and chat:

```sh
make -C c gemma4
./c/coli chat --model ~/Models/gemma4-26b-a4b
./c/coli serve --model ~/Models/gemma4-26b-a4b          # OpenAI- and Anthropic-compatible API
```

`coli router` serves it beside the other installed models like any other chat model, and the model card says `image` (and
`audio`, on the E models) when the engine loaded the tower, and `video` with `image`.

## Pictures, audio and video

The gateway reads pictures and audio before the engine sees them, so the Python that runs `coli serve` (or `coli router`) needs
`numpy` for both and `Pillow` for pictures (`pip install numpy pillow`). Without them the request is refused with a 400 that
names the missing package; text chat is not affected.

Send them as OpenAI content parts, or Anthropic image blocks:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What is written here?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "input_audio", "input_audio": {"data": "<base64 of a WAV, MP3, FLAC ... file>", "format": "wav"}},
  {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}]}
```

Pictures are read by the gateway the way Google's image processor reads them (`c/tools/gemma4_image.py`: resized to the
largest size in multiples of 48 pixels that fits the budget, transparency laid on white, cut into 16-pixel patches); a clip
is read (a WAV: 8 to 32 bits or float, any channels, any rate; see below for the other formats), mixed down, resampled to 16 kHz and turned into the model's
128-bin log-mel frames (`c/tools/gemma4_audio.py`). The prompt gets the model's own begin, one placeholder for every soft token and end
tokens where the chat template writes its marker, the engine gets an `IMAGE` or `AUDIO` frame for each, in the order of the
conversation, and answers an unreadable or mismatched one by name with a 400. `GET /v1/models` says what the loaded engine takes.

A picture costs its soft tokens of context (280 by default) and a conversation can carry several. `G4_IMG_TOKENS` (the
engine, default 280, up to 4096) is the biggest picture it accepts and also the room it keeps for one; the gateway picks the
largest budget of the processor's 70, 140, 280, 560 and 1120 that fits it. More tokens read small print better and take
longer. `G4_AUDIO_TOKENS` (default 750, thirty seconds) is the longest clip; a longer one is cut.

The cache knows pictures and clips apart: the same words after a different picture are not served from the first picture's
cache, the same picture again is not read twice, and a conversation extended with a new turn keeps the picture it had.

Not supported: audio on the 26B-A4B and the 31B (their checkpoints have no audio tower), and the sound of a video.

### Other audio formats

The file says what it is (its first bytes; `format` in the request is only a hint). A WAV is read directly. An **MP3** is decoded
by minimp3 through `acestep_audio_tool` (`make acestep_audio_tool` in `c/`, built beside the engines and installed with them; the
same reader ACE-Step's audio input uses). **FLAC, Ogg, M4A and anything else** needs ffmpeg (`ffmpeg` on the PATH, or `COLI_FFMPEG`
naming it); without one the request is a 400 that says so. The same reader serves `/v1/audio/transcriptions`. Checked on the E2B:
the JFK clip as WAV, MP3, FLAC, Ogg Vorbis and M4A came back as the same words.

### Video

The model reads a video as pictures. The gateway takes up to 32 frames, evenly spaced (transformers' rule, `i * total // n`;
every frame of a shorter video), runs each through the image processor with a budget of 70 soft tokens, and writes the prompt the way
the processor does: for each frame its time as `mm:ss`, the begin-of-image token, one placeholder for each soft token, and the
end-of-image token, the frames joined by spaces. Each frame is an `IMAGE` frame for the engine, so a video is up to 32 pictures
of at most 70 tokens: about 2,300 tokens of context (the default is 8,192) and, on the CPU, a few seconds of vision tower per
frame. Two ways to send one:

```text
{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}              // a file; needs ffmpeg
{"type": "video", "video": {"frames": ["data:image/png;base64,...", ...], "fps": 1}}   // frames you took yourself; no ffmpeg
```

A file is taken apart by ffmpeg (any container and codec it reads; up to 512 MB). Frames are your own: `fps` says how far apart they
are in time (default 1 a second; more than 32 are thinned evenly, the times kept). The URL rules are those of pictures: a `data:`
URI, never a web address, and a path only under `COLI_IMAGE_ROOT`. The engine takes up to 80 pictures and clips in a request
(it took 16 before video).

Checked on the E2B: a 3-second WebM that shows the words ONE, TWO and THREE answered "ONE, TWO, THREE" as a file (32 frames, 62 s
on the CPU with a busy desktop), and the same three frames sent as `frames` were each named with their time (00:00, 00:01,
00:02). The frames are the image processor's work with one difference from transformers' video processor, which resizes with
torchvision where the image processor uses PIL: the patches agree to within 2 of 255 on a few values, and the grid and so the
soft tokens are identical (`tests/test_gemma4_media.py` compares them where torchvision is installed). Not done: the sound of
a video, frames chosen by a frame rate (`fps` sampling) instead of a count, and remote URLs.

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

## Vulkan for the dense matrices

Built with `make gemma4 VK=1` (the Vulkan headers and `glslc`; nothing links a Vulkan loader, so the binary still starts on a
machine without one), and switched on with `COLI_VULKAN=1`, the dense matrices (attention, the dense MLP, the towers, and the
embedding table that is also the output head) run on the graphics card: the same int8 rows and scales, uploaded once on first
use. The routed experts stay on the CPU, and a matrix the device will not take (no room, too many rows) falls back to the CPU.
`G4_VK_DENSE=0` brings the device up and keeps everything on the CPU.

Measured on an RTX 5090 beside a running desktop (renders and a browser open, the engine at low priority, so the numbers are
noisy), the same questions at temperature 0, the same words back on the card as on the CPU:

| | CPU | card | |
|---|---|---|---|
| E2B, 425-token prompt | 8.6 s | 2.8 s | 3 x |
| E2B, 150 tokens out | 13.1 s | 6.9 s | 1.9 x |
| E2B, a picture and 40 tokens | 18.1 s | 12.7 s | 1.4 x |
| 26B-A4B (32 experts cached a layer), 429-token prompt | 19.0 s | 15.1 s | 1.3 x |
| 26B-A4B, 150 tokens out | 19.2 s | 13.6 s | 1.4 x |
| 26B-A4B, a picture and 20 tokens | 34.4 s | 23.0 s | 1.5 x |

Decode gains little on the 26B-A4B because the experts, which are most of the work, still run on the CPU, and each of about
200 dense matrices a token is a synchronous call. The dense 31B (30 GB of int8) would not fit beside anything else on the card.

It is not bit-identical: the card adds in its own order, so the logits are close to the CPU's, not equal, and a row alone
and the same row among several conversations may differ in the last bits. The 179 engine tests pass on the CPU path; with
`COLI_VULKAN=1` all pass but the twelve that require batched decode to equal single-row decode to the bit
(`BatchTest`, `DenseBatchTest`, `PleBatchTest`: two, seven, one-thread and small-cache cases). The CPU stays the default for that
reason.

## What is not there

- **Video with sound, and video without ffmpeg.** The sound of a video is not read, and a video file needs an ffmpeg on the server (frames you send yourself do not).
- **A GPU path for more than the dense matrices.** The experts stay on the CPU, and a graphics card runs the dense matrices only
  (see [Vulkan](#vulkan-for-the-dense-matrices) below). The on-device chains the larger families have (a layer in one
  submission) are a project of their own, and are why the decode gain below is modest.
- **A layer-segment adapter** (`docs/segment-runtime.md`): the family is exempt from that gate until it has one.
- **The 31B** ran on 2026-11-10 (converted from the download in 279 s to a 31.4 GB container; loaded in 17 s, 29.2 GB resident,
  served on its own port at low priority). "What is the capital of France?" answered "The capital of France is Paris." in
  7.5 s; the same 448 by 192 picture as the E4B's was read right, text and both shapes and colours (61 s with the tower, on a
  machine with renders running). It leaves about 5 GB of a 64 GB machine free, so run it alone. Speed was not measured on a
  quiet machine and it was not run against transformers on the real weights.
- **The E4B** ran on 2026-11-10 (converted from the 16 GB download in 55 s to an 8.06 GB container, served on its own port at
  low priority): "What is the capital of France?" answered "The capital of France is Paris." in 1.0 s; a 448 by 192 picture
  with "COLIBRI 4827", a red rectangle and a blue circle was read back right, text, shapes and colours (14.8 s with the tower);
  the JFK clip came back word for word (7.7 s). Not timed for speed on a quiet machine, and not run against transformers on
  the real weights.
- **Speed of the vision tower on a big picture.** A 280-token picture is 2400 patches through 27 layers: tens of seconds on the
  26B-A4B's tower on a busy machine, about eight seconds on the E models' smaller one. It runs once per picture and the cache
  keeps it.
