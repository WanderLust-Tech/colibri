# What this fork changes, and what is still planned

Written 2026-11-07 for the owner of this fork (WanderLust-Tech/colibri, used on one Windows machine). It lists
everything that differs from the checkout we started from, where each piece stands (built, merged, pushed, running),
and what is still open, including the items in [colibri-ideas.md](colibri-ideas.md).

## 1. The baseline and how far we have moved

| | |
|---|---|
| Original checkout | `e0db083c`, upstream PR #2139 merged, 2026-10-17 (upstream v2.0.0 plus what followed) |
| Local `main` now | the tip of `main` (`git log origin/main..main` lists what the fork lacks: 21 commits on the evening of 2026-11-07) |
| Size of the difference | 47 commits (43 of ours, 4 merges), 162 files, +41,473 / -266 lines; 108 of those files are outside `c/tests` and the fixtures |
| Remotes | `origin` = the fork (WanderLust-Tech), `upstream` = JustVugg. Nothing here is meant for upstream; no PR has been opened there |
| Upstream since then | `upstream/main` was 1,412 commits ahead of our baseline at the last fetch (2026-10-31). We have not merged any of it |
| Fork's `main` | `e0414e87`. Local `main` is ahead of it by song saving, KV slots, Gemma 4, the Gemma convert fix, `coli launch`, and the tooling-matrix fix |
| Live server | clone `C:\Users\kp-mi\colibri` at `2266b695` (brought there by a local fetch, not through GitHub; deployed 2026-11-07), running `coli router` on port 8000 through `C:\Users\kp-mi\restart-colibri.ps1 -Router` |

Where each piece stands (**local** = in local `main` only; **fork** = pushed to the fork's `main`; **live** = the code is in
the clone that serves port 8000):

| Piece | Branch(es) | Local | Fork | Live |
|---|---|---|---|---|
| Qwen-Image editing | `feat/qwenimage-vae-encoder` | yes | yes | yes |
| Dashboard: model switcher, size presets, image edit UI | same branch | yes | yes | yes |
| ACE-Step 1.5 text to music, `coli music`, music player | `feat/acestep` | yes | yes | yes |
| ACE-Step audio in (reference, cover, repaint) | `feat/acestep-audio-in` | yes | yes | yes (`acestep.exe` there was built after the merge) |
| Gateway keeps every finished picture | `fix/gateway-saves-images` | yes | yes | yes |
| Dashboard takes its key from `#key=` | on `main` | yes | yes | yes |
| `coli router`, keep-alive, `coli ps`, `coli stop MODEL` | `feat/model-router`, `feat/router-keepalive` | yes | yes | yes |
| Gateway keeps every finished song | `feat/free-pipeline-stages` | yes | **no** | yes |
| Chat turn goes to the KV slot that holds it | `feat/smart-kv-slots` | yes | **no** | yes (the restart script runs 1 slot, so it matters only with `-KvSlots 2` or more) |
| Gemma 4 (26B-A4B, 31B, E4B, E2B; pictures, audio, batched decode) | `feat/gemma4`, `feat/gemma4-setup` | yes | **no** | yes: `gemma4.exe` built, 26B-A4B and E2B containers in `colibri-models` |
| `coli launch` | `feat/coli-launch` | yes | **no** | yes |
| JSON mode (constrained `response_format` on Gemma 4 and Qwen3.6/Qwen3-Coder) | `feat/json-mode` | **no** | **no** | **no** |

Deployed 2026-11-07: the live clone was fast-forwarded to `main` by a local fetch, `gemma4.exe` built there, both Gemma
containers moved from `colibri-staging` to `colibri-models`, and the router restarted with `restart-colibri.ps1 -Router` (the
server was idle). A chat to the E2B through the live router answered in 4.2 s. The 31B and E4B are not installed, and no
picture or audio request has gone through the live router yet.

Not in `main` at all: `docs/repo-deep-dive` (the repository deep-dive, `fdb25925`, pushed to the fork on its own branch),
and `pr-2045`, upstream's image-editing PR, kept only for reference (see 2.1).

## 2. What changed

### 2.1 Qwen-Image editing (`/v1/images/edits`)

Upstream PR #2045 added strength-based img2img, but its C engine never receives the picture, and Qwen-Image-2.1 edits by
reference conditioning, not by adding noise. We did not take it. We built the real thing in four steps, each held to
diffusers or transformers by an oracle:

- VAE encoder (`c/qwenimage_vae_enc.h`), the DiT's reference-image path (interleaved sequence, bidirectional image blocks,
  three-axis RoPE, t=0 modulation for reference rows), the Qwen3-VL vision tower with deepstack and M-RoPE
  (`c/qwenimage_vit.h`), and PIL-compatible Lanczos resizing (`c/qwenimage_resize.h`).
- The whole edit matches diffusers' pipeline to 1e-6 on the tiny fixture; 18 mutations of the tower are caught; the real
  tower matches transformers to 1e-5 up to 640x1024.
- Gateway `POST /v1/images/edits` (multipart or JSON; no URL fetch, file ids or masks), `coli run --image`, `/image` in the
  REPL, `READY "edit":true` in the engine protocol, the stub engine, `coli info`.
- Dashboard: an attach button, thumbnails, an "Auto" size, "edited from N", Again/Retry keep the files.
- A real edit works (sunset sky, rest unchanged): 91 s for 8 steps at 576x448 on the CPU build, 88 s on the Vulkan build.
- Docs: `docs/qwen-image.md` ("Editing pictures", "POST /v1/images/edits").

Limits: the engine takes pictures already resized to a multiple of 32 (the gateway and CLI do the resize); a prompt
containing a literal `<|image_pad|>` is refused; no real-weights DiT edit check at the pipeline's 1024 reference.

### 2.2 Dashboard

- **Model switcher** (`a0cc9ebb`, cherry-picked from `docs/repo-deep-dive`): choose the model from the page. With the router it
  shows loaded or not, a Load button, memory use, time to unload, and an Unload button, and nothing restarts. Backed by
  `c/model_switch.py` and `POST /v1/models/switch`.
- **Image sizes**: more presets, grouped by ratio, in the dashboard and `coli`.
- **Music mode**: lyrics box, length, steps, language, seed; progress with Cancel; audio player with Download WAV and New
  seed; Compose / Cover / Repaint switch.
- **`#key=` link**: the dashboard reads its API key once from the address, strips it, and keeps it in memory only.
- Translations: German, Italian, Indonesian, Chinese added for the new strings. `web/README.md` documents the lot.
- Tests: vitest suites (`api`, `images`, `music`, `storage`), Playwright scripts `web/tests/image_edit_ui.py`,
  `music_ui.py`, `router_ui.py`.

### 2.3 ACE-Step 1.5 (music)

A new engine family, `acestep`, written in C from nothing, with every stage held to upstream's own handler by an oracle
that compares tensors, and mutation tools that prove each oracle can fail:

- VAE decoder, DiT with condition encoders and the turbo sampler, Qwen3 text encoder and tokenizer, and the pipeline (the
  seed is torch's, so a seed gives the clip upstream makes). 48 kHz stereo WAV, 1 to 600 s. No planner.
- Audio in: VAE encoder, audio tokenizer, WAV and MP3 loader (bundled `c/third_party/minimp3.h`). A reference clip, a cover,
  a repaint. Twenty whole generations match upstream.
- `coli music`, `POST /v1/audio/music` (JSON, WAV, SSE progress, multipart), `coli info`, `coli plan`, `coli doctor`,
  `coli setup` (6.4 GB, 9 GB RAM), `make install` and the release archives carry the engine.
- A mutation tool for every stage (decoder, DiT, text path, pipeline, audio in); every mutant is caught except the few
  documented as equivalent in the tools.
- Docs: `docs/acestep.md`. Licence: ACE-Step is MIT (`THIRD_PARTY_NOTICES.md`).

### 2.4 The gateway keeps what it makes

- **Pictures** (`4cbf90d1`, live): every finished picture is saved to `~/colibri-images` (`COLI_IMAGE_DIR`) before it is sent;
  `COLI_IMAGE_SAVE=0` turns it off. A sleeping browser tab could otherwise lose a render.
- **Songs** (`17c7381d`, local only): the same for `~/colibri-music` (`COLI_MUSIC_DIR`, `COLI_MUSIC_SAVE=0`).

### 2.5 `coli router`: several models behind one port

Follows Ollama's scheduler. One front door checks the key once, reads each request's `model`, and sends it to that model's own
`coli serve` worker on a loopback port, started on first use and stopped after `--keep-alive` (default 10 minutes). Before a
load it compares the model's size with free memory and stops the least recently used idle model until it fits; a model that
is answering is never stopped. Only one model computes at a time (`--parallel-models` turns that off), and there is a
loaded-model limit and a queue limit. Then: `keep_alive` per request (`"10m"`, seconds, `0`, `-1`), `coli ps`,
`coli stop MODEL`, `POST /v1/models/unload`, `GET /health` with a `router` block. 62 tests with real worker processes, more
with real gateways and stub engines, 43 mutants all caught. Docs: `docs/router.md`. Live on port 8000.

### 2.6 KV slot choice (`842fc3cc`, local only)

With `--kv-slots N` above 1 a chat request goes to the slot that already holds its earlier turns; a new conversation goes to an
idle never-used slot, else the least recently used. Before, the slot came from a hash of the opening, so unrelated
conversations shared slots and a side request (a title, a summary) could throw away the cache of the conversation in progress,
which on a CPU means re-reading a long prompt for minutes. A `cache_slot` named by the client is still honoured; `GET /health`
shows what each slot holds; `coli router --kv-slots N` gives each worker as many slots as its family allows.

### 2.7 Gemma 4 (`feat/gemma4`, local only)

One engine, `c/gemma4.c`, runs the 26B-A4B (mixture of experts) and the dense 31B, E4B and E2B.

- Converter `c/tools/convert_gemma4.py` (numpy only; dense matrices int8 per row, experts int4; `--add-vision` and `--add-audio`
  give an existing container its towers), own tokenizer header, chat templates reproduced byte for byte, tool calls in Gemma's
  syntax, the thought channel as `reasoning_content`; the E models' template differs and is detected from the container.
- Pictures on all four (vision tower in C; the image processor ported byte for byte); audio on E2B and E4B (conformer in C, mel
  front end identical to the official extractor, WAV only).
- Sliding layers keep a ring, full layers keep every position: 41 KB per token on the 26B-A4B. The cache knows one picture or clip
  from another; a conversation sent again keeps what it shares with the last turn.
- Batched decode: every decoding conversation takes its step in one pass over the weights, with logits bit-identical to
  decoding alone.
- Checked against transformers on three tiny random models (text 4e-5, picture soft tokens 7e-6, clip soft tokens 2e-6), 83
  mutants (three equivalent, the rest caught), and on real weights: 26B-A4B (14.2 tok/s, picture and tools) and E2B (about
  21 tok/s, picture, JFK clip transcribed word for word, HTTP end to end). The 31B and E4B ran only on the tiny models.
- Registry names the four sizes; gateway knows `gemma4_no_cue`; protocol gained an `AUDIO` frame and `BAD_AUDIO`.
- CPU only. No video, no audio other than WAV. Docs: `docs/gemma4.md`.
- The 62 GB of downloaded source checkpoints were deleted 2026-11-07; the converted containers are in
  `C:\Users\kp-mi\colibri-staging` (`gemma4-26b-a4b`, 14.8 GiB; `gemma4-e2b`, 4.8 GiB).

### 2.8 Build, CI, docs and licences

- `c/Makefile`: targets `acestep` and `gemma4`, both in `install`; `install` also ships `router.py`, `model_switch.py`,
  `music_engine.py`, `gemma4_chat.py`.
- `.github/workflows/ci.yml`: ACE-Step tiny-fixture job, Qwen-Image edit and VAE steps, Gemma 4. They have never run on GitHub
  (Actions is off on the fork, and CI does not matter here); every check was run locally.
- `family_registry.py`, `setup_catalog.py`, `doctor.py`, `coli` (about 690 lines added), `openai_server.py` (about 880),
  `image_engine.py`, `music_engine.py` (new, about 1,000), `router.py` (new, about 1,430).
- Docs: new `docs/acestep.md`, `docs/gemma4.md`, `docs/router.md`; changed `docs/api.md`, `docs/qwen-image.md`,
  `docs/api-reference/openapi.json`, `CHANGELOG.md` (an `[Unreleased]` section), `README.md` and its translations (partly),
  `site/index.html`.
- `docs/api.md` now has a second per-engine table (JSON mode, pictures in, audio in) beside the tool-calling one, and
  `c/tests/test_api_matrix_docs.py` holds both to `family_registry.py`. The registry's `tools` flag for `qwen36` was wrong
  (False) and is fixed; the test that should have caught it now probes the gateway's real dispatcher.
- Method for every new engine: a tiny random model in the real layout, an oracle against the reference library, then mutation
  testing so the oracle is known to fail when the engine is wrong.

### 2.9 Outside the repository

| What | Where |
|---|---|
| Restart script (LAN, persistent random key, `-NewKey`, `-Router`) | `C:\Users\kp-mi\restart-colibri.ps1` |
| Live clone | `C:\Users\kp-mi\colibri` (rollback for the image engine: `c/qwenimage.exe.old`) |
| Installed models | `C:\Users\kp-mi\colibri-models`: `qwen-image-2.1`, `ace-step-1.5`, `qwen3-coder-30b` |
| Gemma containers (not installed) | `C:\Users\kp-mi\colibri-staging` |
| Saved pictures and songs | `~/colibri-images`, `~/colibri-music` |
| Worktrees | `D:\code\colibri-{acestep,gemma4,imgsave,router,router2,slots,stages,vae-encoder}` |
| Reference environments | `D:\code\gemma4-venv`, `D:\code\gemma4-ref` (logs, configs, `jfk.wav`); ACE-Step and Qwen-Image reference venvs lived in a session scratchpad and may be gone (rebuild from `c/tools/requirements-*.txt`) |
| Build on this machine | PowerShell with `C:\msys64\ucrt64\bin;C:\msys64\usr\bin` first on PATH, then `make <engine> ARCH=native` (add `VK=1` for Vulkan). `make` fails from Git Bash; for Gemma use the printed gcc command |

Uncommitted in `D:\code\colibri`: 31 web files that differ only in line endings (CRLF), and `docs/colibri-ideas.md`. Left alone.

### 2.10 JSON mode (`feat/json-mode`, not merged, not deployed)

Real constrained decoding for `response_format` on Gemma 4 and Qwen3.6/Qwen3-Coder: [json-mode.md](json-mode.md).

- `c/constrain.h` masks the logits by the grammar's next byte (`c/grammar.h`'s byte-level automaton), lets the engine's sampler
  choose, checks the token byte by byte, re-samples one that fails; the end token only where the grammar can end; an engine
  that cannot go on says so, never unconstrained.
- Engines say `grammar=1` in `CAPS`, take the grammar as the seventh `SUBMIT` field, answer `BAD_GRAMMAR`, `GRAMMAR_STUCK`,
  `GRAMMAR_LOST`. The gateway refuses it with thinking, with tools, and for schemas outside `schema_gbnf.h`'s subset; registry
  flag `constrained_output`, held to the docs and the engine sources by `test_api_matrix_docs.py`.
- Checked with random-weight models (noise left alone) against Python's `json.loads` and the schema's shape; mutation-tested.
  Before this, no engine promised valid JSON, GLM-5.2's grammar being only a draft source.
- Not promised: finishing inside `max_tokens`; thrifty whitespace; UTF-8 inside strings. Not run yet on the real 26B or the
  coder model.

## 3. Planned and open

### 3.1 Waiting for your decision

1. ~~Deploy Gemma 4~~ done 2026-11-07 (see section 1).
2. **Push local `main` to the fork** (song saving, KV slots, Gemma 4, the Gemma convert fix, `coli launch`, the tooling-matrix
   fix). Nothing goes to upstream.
3. **GPU offload for Gemma 4** (ask first; shares the 5090 with your renders). Move the large matrix multiplications, the
   towers first, to the GPU through the existing Vulkan `coli_vk_matmul` (format 1, int8 rows with a per-row scale, which the
   Gemma matrices already use). Expected to help towers and long-prompt prefill most, decode less (the MoE experts stay on
   the CPU). The gain is unmeasured. Needs a Vulkan build of `gemma4.exe` and a CPU-versus-GPU equality test.

### 3.2 Gemma 4 follow-ups

- Real-weights runs of the 31B (needs 33 GB for the container) and the E4B (never downloaded).
- `coli setup` cannot list Gemma: its catalogue holds only ready-to-run downloads, and Gemma needs conversion. That would take a
  container hosted on Hugging Face (publishing, so your call). Instead, `feat/gemma4-setup` (`fdebaae5`, one commit off
  `main`, not merged) lets `coli convert --repo` take a local directory, tests the routing, documents the conversion and
  fixes the Makefile's "text only" comment.
- Video, and audio other than WAV.

### 3.3 Items from colibri-ideas.md

Status as of today (the file itself has been brought into line with this):

| Item | Status |
|---|---|
| A1-A5 router, load on demand, memory fit, one render at a time, limits | **done**, live |
| A0 one server per model on separate ports | set aside, the router replaced it |
| B1-B3 `keep_alive`, `coli ps` / `coli stop`, Unload button | **done**, live |
| C1 `coli launch` | **done**, branch `feat/coli-launch`, not merged: Claude Code and opencode via the environment; Codex needs a Responses API colibri lacks; settings-file tools left out; not run against the real tools |
| C2 Named presets (saved sizes, steps, durations that show up as model names) | open, S-M |
| C3 Windows tray or autostart (would replace the restart script) | open, M |
| C4 One place for settings, request logging | open, S-M |
| C5 Embeddings, structured JSON output, speech to text and back: check what the chat engines have | **surveyed** (no embeddings, no speech route; STT only through Gemma E-model chat); structured JSON **built**, branch `feat/json-mode`, not merged (2.10) |
| D1 Delete the original after convert | mostly there (`coli convert --repo` does it); not built for `--indir`, on purpose |
| D2 Free pipeline stages | already in the engine (`COLI_IMG_TE=stage`, about 14 s per new prompt); not needed on 64 GB |
| D3 Try `TRUNK_RESIDENT_LAYERS` on the dense Qwen3.8-27B | open, S |
| E1 Save finished songs | **done**, local only (see 2.4) |
| E2 Audio-in cases in the small CI fixture | open, S |
| E3 ACE-Step's 5 Hz planner (1.7B, also 0.6B and 4B) | open, L |
| E4 Lego, extract and complete tasks (need other checkpoints) | open, L |
| E5 Push `main` to the fork | open; see 3.1 |
| F (streaming the DiT from disk, LoRA training on streamed layers, AirLLM-style loading for MoE) | decided against |

### 3.4 Other open points

- **Catching up with upstream**: 1,412 commits behind. Merging them is a large job with conflicts in `openai_server.py`, `coli`,
  `family_registry.py` and the Makefile; decide whether to do it, or to cherry-pick what is wanted.
- **Qwen-Image editing**: Linux, ASan and UBSan runs; a Vulkan check of the chain with an edit prefix; a real edit at the 1024
  reference; translated READMEs (it, ja, zh, id) for editing.
- **ACE-Step**: rebuild `acestep.exe` in the live clone if audio in is wanted (see section 1); the upstream handler once rewrote
  files in `ace-step-1.5` (weights untouched); the 1.7B planner weights sit in that folder unused.
- **CI**: never run on GitHub. Turning it on is only worth it if you want Linux and sanitizer coverage.
- `docs/repo-deep-dive` holds a repository deep-dive that is not in `main`.
