# Every model colibri can run

Written 2026-11-20 for the owner of this fork. One page that lists each model the engines here can serve, what it is, how big it is,
what it can do, and where it stands on this machine. The details of each family are in its own page (linked in the tables); this page
is the map.

Where the facts come from, so they can be checked and refreshed:

- `c/setup_catalog.py`: the downloads `coli setup` and `coli model install` know (sizes, RAM, licence, `--model` id).
- `c/family_registry.py`: every engine, the context limits, and what the gateway lets each family do (tools, thinking, pictures).
- `GET /v1/models` on the running router: what is installed and what each model takes, live.
- The family pages in [docs/](.), and the measurements in them. A blank means nobody has measured it.

**This machine** (the one the fork is used on): an i9-14900KF (24 cores, 32 threads), 64 GB of RAM, an RTX 5090 with 32 GB, Windows
11. The router (`coli router`, port 8000) loads a model when it is asked for and puts it on the card when it fits
([router.md](router.md), [gpu-offload.md](gpu-offload.md)).

## 1. What is installed here

These are in `C:\Users\kp-mi\colibri-models` and appear in `GET /v1/models` (checked 2026-11-20). "Runs on" is where the router puts it
when it loads (card when it fits, otherwise the CPU); the embedding model is the one known to be on the CPU right now.

| Model id (as the API names it) | Kind | Takes in | Window | Runs on | Measured here |
|---|---|---|---|---|---|
| `gemma-4-e2b-colibri` | chat | text, pictures, video, audio | 32768 | card when it fits | about 109 tokens/s through the router |
| `gemma-4-e4b-colibri` | chat | text, pictures, video, audio | 32768 | card when it fits | about 80 tokens/s through the router |
| `gemma-4-26b-a4b-colibri` | chat | text, pictures, video | 32768 | card when it fits (experts too) | 25 to 28 ms a token (about 36 to 40 tokens/s); 51 ms before the experts went on the card |
| `qwen3-coder-30b-a3b-colibri` | chat (code) | text | 65536 | card when it fits | 1.4 to 1.9 times the CPU, 17 GB of the card |
| `qwen-image-2.1-colibri` | pictures, editing | text (and up to 4 pictures to edit) | | card when it fits (about 13 GB, 18 GB for an edit) | 768x512 in 6.1 s warm (94 s of denoising on the CPU); an edit 20 s (137 s on the CPU) |
| `z-image-turbo-colibri` | pictures | text | | card | 13 s of denoising for 768x768 (171 s on the CPU) |
| `ace-step-1.5-colibri` | music | text, a reference clip (WAV, MP3) | | CPU | 10 s of music in about 16 s |
| `qwen3-embedding-0.6b` | embeddings | text | 8192 a text | CPU (the card was full when it loaded) | 64 texts of 230 tokens: 23.5 s on the CPU, 1.13 s on the card |
| `qwen3-reranker-0.6b` | reranking | query and documents | 8192 a pair | card | 20 passages in 0.37 s (5.9 s on the CPU) |
| `qwen3-reranker-4b` | reranking | query and documents | 8192 a pair | card | 20 passages in 1.2 s (24.8 s on the CPU) |
| `qwen3-reranker-8b` | reranking | query and documents | 8192 a pair | card | 20 passages in 1.96 s (40.8 s on the CPU) |

Through the router, 20 passages of about 400 tokens ranked in 0.9 s (0.6B), 3.0 s (4B) and 4.6 s (8B) with the model already loaded.

Not installed but supported and sized for this machine: the Gemma 4 31B (31 GB container; leaves about 5 GB of RAM free), Qwen3.6-35B-A3B
(23 GB), the 4B and 8B embedding models, and the two decision models (Laya, GLiNER2.5-Decide, each under 2 GB). Install with
`coli model install <id>` ([model-install.md](model-install.md)).

## 2. What each one can do

A quick matrix; the next sections give the sizes. "Tools" is function calling, "Thinking" is a reasoning channel the client can show
or hide, "JSON mode" is `response_format` held to a grammar token by token ([json-mode.md](json-mode.md)).

| Model | Tools | Thinking | Pictures in | Audio in | Video in | JSON mode | Notes |
|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | yes | yes | yes | no | no | yes | |
| Qwen3-Coder-30B-A3B | yes | no | no | no | no | yes | built for code |
| Gemma 4 26B-A4B and 31B | yes | yes | yes | no | yes | yes | |
| Gemma 4 E4B and E2B | yes | yes | yes | yes | yes | yes | also `/v1/audio/transcriptions` (speech to text) |
| Qwen3.8-Flash-Next | yes | yes | yes | no | no | no | multi-token prediction drafts (`Q38_MTP`) |
| Qwen3.8-27B (dense) | yes | yes | yes | no | no | not tried | runs on the qwen36 engine |
| DeepSeek V4 Flash and its REAP | yes | yes | no | no | no | no | |
| DeepSeek V4.1 Flash | yes | yes | yes | no | no | no | |
| MiMo-V2.6 Flash and Pro | yes | yes | yes | no | no | no | |
| GLM-5.2 / 5.3 | yes | yes | no | no | no | no | 5.2 has the MTP head for speculative decoding |
| GLM-5.3-Flash | yes | yes | yes | no | no | no | |
| Inkling | no | yes | no | yes | no | no | |
| Kimi K3 | yes | yes | no | no | no | no | text only |
| OLMoE | no | no | no | no | no | no | small, to learn the tools on |

Qwen-Image-2.1, Z-Image-Turbo, ACE-Step, Qwen3-ASR, Qwen3-TTS, the embedding and reranking models and the decision models do not chat; section 4 and
after.

## 3. Chat models

### 3.1 Small: they run from RAM

| Model | `--model` / install id | Parameters | Download | RAM (least / good) | Window the engine allows | GPU | Licence | Page |
|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | `qwen36-35b` | 35B, about 3B a token; 25% gated attention, 75% Gated DeltaNet; 256 experts, 8 and 1 shared | 23 GB | 10 / 20 GB | 262144 (default 8192) | CUDA, Vulkan | Apache-2.0 | [qwen36.md](qwen36.md) |
| Qwen3-Coder-30B-A3B | `qwen3-coder-30b` | 30B, about 3B a token | 19 GB | 8 / 18 GB | 262144 (65536 on this router) | CUDA, Vulkan | Apache-2.0 | [qwen36.md](qwen36.md#qwen3-coder-30b-a3b) |
| Gemma 4 26B-A4B | no catalogue entry: convert | 25.2B, 3.8B a token; 128 experts, 8 a token; 30 layers | 52 GB, 15 GB after conversion | 18 GB with every expert cached | 262144 (default 8192, 32768 here) | Vulkan (this fork) | Apache-2.0 | [gemma4.md](gemma4.md) |
| Gemma 4 31B | same | 31B dense; 60 layers | the bf16 checkpoint; 31.4 GB after conversion | 29.2 GB resident when it ran (leaves about 5 GB free here) | 262144 | Vulkan | Apache-2.0 | [gemma4.md](gemma4.md) |
| Gemma 4 E4B | same | 8B with per-layer embeddings, 4B in use; 42 layers | 8.06 GB container | small | 131072 | Vulkan | Apache-2.0 | [gemma4.md](gemma4.md) |
| Gemma 4 E2B | same | 5B with per-layer embeddings, 2B in use; 35 layers | 4.8 GB container | small | 131072 | Vulkan | Apache-2.0 | [gemma4.md](gemma4.md) |

Speeds from the page of each: Qwen3.6 6.0 tokens/s on a CPU, 9.9 on an integrated GPU, 30.0 with CUDA on an RTX 3070; Qwen3-Coder 8.5 to
9.6 with every expert in RAM; Gemma 26B-A4B 14 tokens/s on the CPU alone (i9-14900KF, 8 threads), and the figures in section 1 on the card.
Gemma is the only family with all of pictures, video, audio (E models), tools, thinking and JSON mode, and it is the family
the chain and expert-tier work on the card was done for ([vulkan.md](vulkan.md)).

### 3.2 Large: the experts stream from the disk

The disk sets the speed (a fast NVMe helps most); RAM holds the dense part and a cache of experts. "RAM" is the least it starts with / where
it runs as measured. Download sizes are the repository listings.

| Model | `--model` / install id | Parameters | Download | RAM | GPU | Measured | Licence | Page |
|---|---|---|---|---|---|---|---|---|
| DeepSeek V4 Flash REAP 150B | `deepseek-v4-flash-reap` | 150B: 132 of the 256 experts kept | 85 GB | 16 / 32 GB | CUDA, Vulkan | | MIT | [deepseek-v4.md](deepseek-v4.md) |
| DeepSeek V4 Flash | `deepseek-v4-flash` | 284B; 43 layers, 256 experts, top 6 and 1 shared; MLA and sparse attention; fp4 experts | 167 GB | 16 / 32 GB | CUDA, Vulkan | 0.93 to 1.24 tokens/s on the CPU; 1.5 to 1.6 with CUDA | MIT | [deepseek-v4.md](deepseek-v4.md) |
| Qwen3.8-Flash-Next | `qwen38-flash-next` | 125B with 6B a token, plus a 51B hashed n-gram table; 48 layers; 512 experts, top 10 | 186 GB | 24 / 32 GB | CUDA, Vulkan | 1.9 to 2.6 tokens/s; 4.0 with the int4 experts | Qwen community licence | [qwen38.md](qwen38.md) |
| MiMo-V2.6 Flash | `mimo-v2.6-flash` | 309B, 15B a token; 48 layers; 256 experts, top 8 | 172 GB | 32 / 52 GB | Vulkan | 2.3 to 3.4 tokens/s | MIT | [mimo.md](mimo.md) |
| MiMo-V2.6 Pro | `mimo-v2.6-pro` | 1.02T, 42B a token; 70 layers; 384 experts, top 8 | 564 GB | 54 / 64 GB | Vulkan | 0.66 to 0.79 tokens/s | MIT | [mimo.md](mimo.md#pro) |
| GLM-5.2 | `glm-5.2` | 744B, the reference model, with the int8 MTP head | 429 GB | 16 / 24 GB | CUDA, Vulkan | 1.83 tokens/s on a 128 GB machine; 9.0 on six RTX 5090 | MIT | [README](../README.md#2-get-the-model) |
| GLM-5.3 | `glm-5.3` | 744B, the same engine, no MTP head | 419 GB | 16 / 24 GB | CUDA, Vulkan | | GLM-5.3 licence | [README](../README.md#2-get-the-model) |
| Inkling | `inkling` | 975B, 41B a token; int4 experts, bf16 dense weights | 514 GB | 120 / 128 GB as downloaded, 25 GB after a dense conversion | CUDA, Vulkan | 0.25 tokens/s | Apache-2.0 | [inkling.md](inkling.md) |
| Kimi K3 | `kimi-k3` | 2.8T, 104B a token; 93 layers; native MXFP4 experts | 1.56 TB | 32 / 64 GB | CUDA, Vulkan | about 9.4 s a token | Kimi K3 licence | [kimi_k3.md](kimi_k3.md) |

For this machine, by the RAM column alone: the REAP, V4 Flash, Qwen3.8-Flash-Next, MiMo Flash and the two GLMs can start; MiMo Pro is at
the edge of 64 GB; Inkling needs its dense conversion first; Kimi K3 starts but its 1.56 TB has to live on a disk. The window the engines
allow for all of the big families is 1,048,576 (Qwen3.8: 262144) with a default of 4096 to 8192.

### 3.3 By hand: a conversion or a preparation after the download

`coli setup` cannot download these ready to run.

| Model | Download, then on disk | RAM | GPU | Measured | Page |
|---|---|---|---|---|---|
| OLMoE 7B | 14 GB, 7 GB as int8 | 8 GB | Vulkan | 22 to 23 tokens/s (CPU) | [README](../README.md#other-supported-models) |
| Gemma 4 (all four) | see 3.1; `coli convert --repo google/gemma-4-...` | | | | [gemma4.md](gemma4.md) |
| Qwen3.8-27B, dense; text and pictures | 56 GB, 51 GB after conversion | 20 GB int4, 30 GB int8 | Vulkan | 3.45 tokens/s int4, 2.1 int8 | [qwen36.md](qwen36.md#the-dense-27b) |
| GLM-5.3-Flash 321B; pictures and tools | 328 GB, 195 GB after conversion | 25 GB | CUDA, Vulkan | about 20 s a token warm | [glm53-flash.md](glm53-flash.md) |
| DeepSeek V4.1 Flash 552B; pictures and tools | 510 GB, one-off preparation | about 18 GB plus the expert cache | Vulkan | 0.21 to 0.24 tokens/s | [deepseek-v41.md](deepseek-v41.md) |

## 4. Pictures

| Model | Does | Sizes | Steps | Download | RAM | GPU | Licence | Page |
|---|---|---|---|---|---|---|---|---|
| Qwen-Image-2.1 | text to picture; edit up to 4 pictures with an instruction | each side a multiple of 32, 256 to 2048; default 768x512; edit references 384 to 1024 (default 768), at most 4 megapixels | default 8 | 33 GB | 12 / 18 GB | Vulkan (15.7 times the CPU's denoise here) | Qwen Research Licence, **non-commercial** | [qwen-image.md](qwen-image.md) |
| Z-Image-Turbo | text to picture only (no editing) | each side a multiple of 16, 256 to 2048; default 768x768 | 1 to 200, default 8 (distilled for 8) | 33 GB | 10 / 14 GB | Vulkan | Apache-2.0 | [zimage.md](zimage.md) |

Both read the official diffusers checkpoint and quantize to int8 while loading (no conversion). Z-Image is the lighter one and the one with an
open licence. Reached by `POST /v1/images/generations` (and `/v1/images/edits` for Qwen-Image), `coli run`, `coli chat` and the dashboard.
Every finished picture is kept in `~/colibri-images`.

## 5. Music

| Model | Does | Output | Download | RAM | Licence | Page |
|---|---|---|---|---|---|---|
| ACE-Step 1.5 | text (a caption, optional lyrics) to a song; a reference clip, a cover of a clip, a repaint of a clip | 48 kHz stereo, 1 to 600 s (default 30), 8 steps by default (up to 20); caption up to 4096 bytes, lyrics up to 32768; input clips WAV or MP3 up to 600 s | 6.4 GB | 9 / 12 GB | MIT | [acestep.md](acestep.md) |

`POST /v1/audio/music`, `coli music`, and a player in the dashboard. Runs on the CPU (no Vulkan path). Finished songs are kept in
`~/colibri-music`.

## 6. Embeddings and reranking

One engine, `c/qwen3embed`, runs both families; it uses the card when there is room.

| Model | `--model` / install id | Output | Download | RAM | Page |
|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | `qwen3-embedding-0.6b` | vectors of 1024 numbers | 1.2 GB | 2 / 4 GB | [embeddings.md](embeddings.md) |
| Qwen3-Embedding-4B | `qwen3-embedding-4b` | 2560 numbers; about 7 times slower than the 0.6B | 8.1 GB | 10 / 12 GB | [embeddings.md](embeddings.md) |
| Qwen3-Embedding-8B | `qwen3-embedding-8b` | 4096 numbers; about 13 times slower | 15.2 GB | 18 / 20 GB | [embeddings.md](embeddings.md) |
| Qwen3-Reranker-0.6B | `qwen3-reranker-0.6b` | a score a document | 1.2 GB | 2 / 4 GB | [rerank.md](rerank.md) |
| Qwen3-Reranker-4B | `qwen3-reranker-4b` | about 4 times slower | 8.1 GB | 10 / 12 GB | [rerank.md](rerank.md) |
| Qwen3-Reranker-8B | `qwen3-reranker-8b` | about 7 times slower | 16.4 GB | 18 / 20 GB | [rerank.md](rerank.md) |

All Apache-2.0. Embeddings: `POST /v1/embeddings` in OpenAI's shape, up to 8192 tokens an input (the model's own window is 32k), with
`dimensions` to keep the first N numbers (down to 1) and Colibri's `instruction` to mark an input as a query. Reranking: `POST /v1/rerank`
and `/v2/rerank` (Cohere's shape), up to 8192 tokens a pair and 2048 documents a request. `coli embed` and `coli rerank` call both
from the terminal.

## 7. Speech to text

| Model | `--model` / install id | Takes | Download | RAM | Speed here (CPU) | Page |
|---|---|---|---|---|---|---|
| Qwen3-ASR-0.6B | `qwen3-asr-0.6b` | 16 kHz speech; 30 languages and 22 Chinese dialects, detected or forced | 1.9 GB | 3 / 5 GB | an 11 s clip in 4.3 s | [qwen3-asr.md](qwen3-asr.md) |
| Qwen3-ASR-1.7B | `qwen3-asr-1.7b` | the same, more accurate (about 5.4% word error on English, against 7.6% for Whisper large-v3-turbo, by Qwen's report and the Open ASR leaderboard: not measured here) | 4.7 GB | 7 / 10 GB | an 11 s clip in 7.6 s | [qwen3-asr.md](qwen3-asr.md) |

Both Apache-2.0. `POST /v1/audio/transcriptions` in the shape Whisper clients send (`language`, `prompt` as context, `json`, `text` and
`verbose_json`), up to 15 minutes and 25 MB a request (the engine takes 20 minutes in one piece); `coli transcribe FILE` from the terminal.
Gemma 4's E2B and E4B also transcribe on the same route, as a chat model asked to write down what it hears. Not installed here yet.

## 8. Text to speech

| Model | `--model` / install id | Gives | Download | RAM | Speed here (CPU) | Page |
|---|---|---|---|---|---|---|
| Qwen3-TTS-0.6B-CustomVoice | `qwen3-tts-0.6b-customvoice` | 24 kHz speech in 10 languages, nine named speakers (a dialect for two of them) | 2.5 GB | 4 / 6 GB | 5 s of speech in about 17 s (about 3.5 times slower than real time) | [qwen3-tts.md](qwen3-tts.md) |

Apache-2.0. `POST /v1/audio/speech` in OpenAI's shape (`input`, `voice`, `response_format`; OpenAI's voice names stand for speakers; `language`
and `seed` are extensions), WAV by default, PCM, and MP3, Opus, AAC and FLAC with ffmpeg; `coli speak "text"` from the terminal. No style
instructions, voice design or cloning yet (the 1.7B and the Base models are not built). Not installed here yet.

## 9. Decision models

They answer typed questions about a state (`choice`, `score`, `noul`) with calibrated probabilities in one forward pass, and never
write text. Reached through `POST /v1/systemone` ([systemone.md](systemone.md)). The chat models above also answer it.

| Model | What it is | Download | RAM | GPU | Measured | Page |
|---|---|---|---|---|---|---|
| Laya (Convai Innovations) | 421M, English | 0.85 GB | 1.7 GB | CPU | 219 ms a question | [laya.md](laya.md) |
| GLiNER2.5-Decide (fastino) | 340M, a DeBERTa-v3-large encoder; intent, routing, sentiment, urgency, moderation | 1.95 GB | 1.9 GB | CPU | 294 ms a question | [gliner_decide.md](gliner_decide.md) |
| Clef (Cloudflare) | Qwen3.8-27B with a decision head; it also chats | 55 GB, 52 GB converted | 19 GB int4 to 55 GB f16 | CPU | 20.4 s a request (int8) | [clef.md](clef.md) |

## 10. Which route takes which model

| Route | Models |
|---|---|
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/messages` | every chat model (section 3) |
| `POST /v1/audio/transcriptions` | Qwen3-ASR 0.6B and 1.7B (a dedicated recogniser); Gemma 4 E2B and E4B (a chat model asked to transcribe) |
| `POST /v1/audio/speech` | Qwen3-TTS-0.6B-CustomVoice |
| `POST /v1/images/generations` | Qwen-Image-2.1, Z-Image-Turbo |
| `POST /v1/images/edits` | Qwen-Image-2.1 |
| `POST /v1/audio/music` | ACE-Step 1.5 |
| `POST /v1/embeddings` | Qwen3-Embedding 0.6B, 4B, 8B |
| `POST /v1/rerank`, `/v2/rerank` | Qwen3-Reranker 0.6B, 4B, 8B |
| `POST /v1/systemone` | Laya, GLiNER2.5-Decide, Clef; also every chat model |

A model sent to a route it does not serve answers 400 with a one-line reason. The full request shapes are in [api.md](api.md).

## 11. Keeping this page right

When a model is added or changes, these places move with it: its `CatalogModel` in `c/setup_catalog.py` and its family in
`c/family_registry.py` (the tests in `c/tests/test_setup_catalog.py` hold the two together), the family's own page, the README's
tables, and this page. The installed list (section 1) comes from `GET /v1/models`; refresh it after `coli model install`.
