# What this fork changes, and what is still planned

Written 2026-11-07 for the owner of this fork (WanderLust-Tech/colibri, used on one Windows machine). It lists
everything that differs from the checkout we started from, where each piece stands (built, merged, pushed, running),
and what is still open, including the items in [colibri-ideas.md](colibri-ideas.md). Brought up to date on 2026-11-10
(Gemma on the graphics card, video, the card-or-CPU choice, model settings, context windows, the VS Code extension), and
again on 2026-11-20 (Z-Image, the model installer, embeddings, reranking and their graphics-card path, the models page).
[models.md](models.md) lists every model and what it can do.

## 1. The baseline and how far we have moved

| | |
|---|---|
| Original checkout | `e0db083c`, upstream PR #2139 merged, 2026-10-17 (upstream v2.0.0 plus what followed) |
| Local `main` now | the tip of `main`; on 2026-11-20 the fork has all of it (`git log origin/main..main` is empty) |
| Size of the difference | 129 commits, 322 files, +81,836 / -5,995 lines (2026-11-20, before the models page) |
| Remotes | `origin` = the fork (WanderLust-Tech), `upstream` = JustVugg. Nothing here is meant for upstream; no PR has been opened there |
| Upstream since then | `upstream/main` was 1,412 commits ahead of our baseline at the last fetch (2026-10-31). We have not merged any of it |
| Fork's `main` | `98abb119` (pushed 2026-11-20); the models page lands after it |
| Live server | clone `C:\Users\kp-mi\colibri` at `98abb119` (brought there by `git pull` from the fork), running `coli router` on port 8000 through `C:\Users\kp-mi\restart-colibri.ps1 -Router` (2.9 lists what that script sets); `gemma4.exe`, `qwen36.exe` and `qwen3embed.exe` are the `VK=1` builds |

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
| Gateway keeps every finished song | `feat/free-pipeline-stages` | yes | yes | yes |
| Chat turn goes to the KV slot that holds it | `feat/smart-kv-slots` | yes | yes | yes (the restart script runs 1 slot, so it matters only with `-KvSlots 2` or more) |
| Gemma 4 (26B-A4B, 31B, E4B, E2B; pictures, audio, batched decode) | `feat/gemma4`, `feat/gemma4-setup` | yes | yes | yes: `gemma4.exe` built, 26B-A4B and E2B containers in `colibri-models` |
| `coli launch` | `feat/coli-launch` | yes | yes | yes |
| JSON mode (constrained `response_format` on Gemma 4 and Qwen3.6/Qwen3-Coder), with the whitespace and UTF-8 fix | `feat/json-mode`, `fix/json-whitespace` | yes (`d31c2dd6`) | yes | yes (2026-11-09): live clone at `d31c2dd6`, `gemma4.exe` and `qwen36.exe` rebuilt; checked on the real E2B, 26B and Qwen3-Coder |
| `/v1/audio/transcriptions` (speech to text on the Gemma E models) | `feat/audio-transcriptions` | yes | yes | yes (2026-11-09); the JFK clip came back word for word on the E2B |
| Named presets, request log, `coli settings` (C2, C4) | `feat/presets-settings` | yes (`05932e22`) | yes | yes (2026-11-09): live clone and router at `05932e22`, request log on at `~/colibri-requests.log` |
| Dashboard: Presets and Requests tabs, presets in the model list | `feat/web-presets` | yes (`5b000777`) | yes | yes (2026-11-09): `npm run build` in the live clone, router restarted; the new routes answer on 8000 |
| Tray icon and background service (`coli tray`, `coli service`, C3) | `feat/tray` | yes (`ecfd9219`) | yes | yes (2026-11-09): live clone at `ecfd9219`, tray started from it; router on 8000 not restarted (the tray is not part of it); the Windows sign-in entry is installed (the `Colibri tray` value under `HKCU\...\Run`, seen 2026-11-18) |
| Gemma 4 dense matrices on the Vulkan device (G2) | `396a774f` | yes | yes | yes (2026-11-10): `gemma4.exe` is the `VK=1` build; the router decides per model whether it runs on the card (2.14) |
| Gemma 4 decode chain (`gemma4_chain.h`): a decode row's layers in one submission, the head fused | `feat/gemma4-chain` | yes (`877fa6bd`) | yes | yes (2026-11-18): live clone at `877fa6bd`, `gemma4.exe` rebuilt with `VK=1`, router restarted; E2B ~109 tok/s and E4B ~80 through it (2.19) |
| Gemma 4 experts on the card (the shared expert tier, GeGLU in the device shaders) | `feat/gemma4-experts-gpu` | yes (`c54ed0e9`) | yes | yes (2026-11-18): the 26B-A4B decodes in 25 to 28 ms a token (2.20) |
| Gemma 4 video, and audio that is not WAV (G5) | `b6fae35c` | yes | yes | yes (2026-11-10): ffmpeg 9.0.2 installed with winget, `acestep_audio_tool.exe` built |
| Card or CPU by free memory (`coli router --gpu auto`), models turned off and on, `service.json` `env` | `3b4bd293` | yes | yes | yes (2026-11-10): `COLI_GPU=auto` in the restart script and in `service.json` |
| Dashboard: Runs on and Turn off/on per model | `7ae38131` | yes | yes | yes (2026-11-10): `web/dist` rebuilt in the live clone |
| `context_length` on the model card | `26af52c9` | yes | yes | yes (2026-11-10): `gemma4.exe` and `qwen36.exe` relinked, router restarted |
| Z-Image-Turbo (text to picture) | `feat/zimage` | yes (`6a506de9`) | yes | yes (2026-11-18), 2.21 |
| Installing and removing models while it runs (`coli model install`, router API, dashboard) | `feat/model-installer` | yes (`6b0e6432`) | yes | yes (2026-11-19), 2.22 |
| Embeddings, `/v1/embeddings` (Qwen3-Embedding) | `feat/embeddings` | yes (`a6a2407a`) | yes | yes (2026-11-20), 2.23 |
| Reranking, `/v1/rerank` (Qwen3-Reranker); shared start; `coli embed` and `coli rerank` | `feat/reranker`, `feat/rerank-prefix` | yes (`ea3f4bda`, `f29d170c`) | yes | yes (2026-11-20), 2.24, 2.25 |
| Embeddings and reranking on the card (one Vulkan frame a forward) | `feat/embed-gpu` | yes (`d94a2a1b`, LF fix `98abb119`) | yes | yes (2026-11-20): `qwen3embed.exe` is the `VK=1` build; rerankers report `device: vulkan`, 2.26 |
| VS Code extension 0.14.0 (outside this repository, 2.17) | none: its own folder, no commits | n/a | n/a | the package is built and installed on this machine (0.14.0 installed 2026-11-20) |

Deployed 2026-11-07: the live clone was fast-forwarded to `main` by a local fetch, `gemma4.exe` built there, both Gemma
containers moved from `colibri-staging` to `colibri-models`, and the router restarted with `restart-colibri.ps1 -Router` (the
server was idle). A chat to the E2B through the live router answered in 4.2 s. The 31B and E4B are not installed, and no
picture or audio request has gone through the live router yet.

Deployed 2026-11-10 (the server was idle each time, and you said to): Gemma on the card and video and audio formats (VK build of
`gemma4.exe`, ffmpeg), then the card-or-CPU policy (VK build of `qwen36.exe` with the expert tier, `COLI_GPU=auto`), then
`context_length` (relink of both engines, router restart). Not installed: the 31B (its container is in `colibri-staging`).

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
  21 tok/s, picture, JFK clip transcribed word for word, HTTP end to end). The E4B and 31B were run on real weights on 2026-11-10 (text, picture; audio on the E4B).
- Registry names the four sizes; gateway knows `gemma4_no_cue`; protocol gained an `AUDIO` frame and `BAD_AUDIO`.
- Video and audio other than WAV came later (2.14), as did the graphics card. Docs: `docs/gemma4.md`.
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
| Restart script (LAN, persistent random key, `-NewKey`, `-Router`). Defaults since 2026-11-10: `-KeepAlive 30m`, `-Ctx 65536` (Qwen3.6 family, `Q36_MAXT`), `-GemmaCtx 32768` (`CTX`), `-Cap 128`, one KV slot; it also sets `COLI_GPU=auto` and `COLI_FFMPEG` | `C:\Users\kp-mi\restart-colibri.ps1` |
| Service settings (what the tray and `coli service` start with): `env` has `COLI_GPU`, `Q36_MAXT`, `CTX`, `COLI_FFMPEG` | `%LOCALAPPDATA%\colibri\service.json` |
| ffmpeg 9.0.2 (winget, user scope) | `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_*` |
| VS Code extension (source) | `C:\code\colibri-vscode-extension\colibri-agent-connector` |
| Per-model settings (turned off, where it runs) | `%LOCALAPPDATA%\colibri\model-settings.json` (docs/model-settings.md) |
| Live clone | `C:\Users\kp-mi\colibri` (rollback for the image engine: `c/qwenimage.exe.old`) |
| Installed models | `C:\Users\kp-mi\colibri-models`: `qwen-image-2.1`, `ace-step-1.5`, `qwen3-coder-30b`, `gemma4-26b-a4b`, `gemma4-e2b`, `gemma4-e4b` |
| Gemma containers (the 31B is only here, not installed) | `C:\Users\kp-mi\colibri-staging` |
| Saved pictures and songs | `~/colibri-images`, `~/colibri-music` |
| Worktrees | `D:\code\colibri-{acestep,gemma4,imgsave,json,router,router2,slots,stages,stt,vae-encoder}` |
| Reference environments | `D:\code\gemma4-venv`, `D:\code\gemma4-ref` (logs, configs, `jfk.wav`); ACE-Step and Qwen-Image reference venvs lived in a session scratchpad and may be gone (rebuild from `c/tools/requirements-*.txt`) |
| Build on this machine | PowerShell with `C:\msys64\ucrt64\bin;C:\msys64\usr\bin` first on PATH, then `make <engine> ARCH=native` (add `VK=1` for Vulkan). `make` fails from Git Bash; for Gemma use the printed gcc command |

Uncommitted in `D:\code\colibri`: 31 web files that differ only in line endings (CRLF), and `docs/colibri-ideas.md`. Left alone.

### 2.10 JSON mode (`feat/json-mode`, `fix/json-whitespace`; merged, live 2026-11-09, not pushed)

Real constrained decoding for `response_format` on Gemma 4 and Qwen3.6/Qwen3-Coder: [json-mode.md](json-mode.md).

- `c/constrain.h` masks the logits by the grammar's next byte (`c/grammar.h`'s byte-level automaton), lets the engine's sampler
  choose, checks the token byte by byte, re-samples one that fails; the end token only where the grammar can end; an engine
  that cannot go on says so, never unconstrained.
- Engines say `grammar=1` in `CAPS`, take the grammar as the seventh `SUBMIT` field, answer `BAD_GRAMMAR`, `GRAMMAR_STUCK`,
  `GRAMMAR_LOST`. The gateway refuses it with thinking, with tools, and for schemas outside `schema_gbnf.h`'s subset; registry
  flag `constrained_output`, held to the docs and the engine sources by `test_api_matrix_docs.py`.
- Checked with random-weight models (noise left alone) against Python's `json.loads` and the schema's shape; mutation-tested.
  Before this, no engine promised valid JSON, GLM-5.2's grammar being only a draft source.
- Not promised: finishing inside `max_tokens`; thrifty whitespace (a run is bounded at 20 bytes, since the first real run, on
  a Gemma E2B, showed an unbounded one eat the whole budget). Strings are valid UTF-8; `json_object` is an object. Real runs
  (2026-11-09) on the E2B, 26B-A4B and Qwen3-Coder: schema, `json_object` and `gbnf` all valid; an answer can still carry up to
  20 whitespace bytes before the object or after it (the E2B and Qwen did before, the 26B after).

### 2.11 Speech to text (`feat/audio-transcriptions`, merged and live 2026-11-09)

`POST /v1/audio/transcriptions` on a Gemma 4 E2B or E4B, in OpenAI's multipart shape ([api.md](api.md#speech-to-text-v1audiotranscriptions)).

- A WAV file of any rate is mixed to mono and resampled; one over 30 s is cut at its quietest 100 ms within the last 4 s of
  each piece (`gemma4_audio.split_clip`), each piece goes through the same chat path as an `input_audio` part, and the texts are
  joined. Silent pieces are skipped. `json`, `text` and `verbose_json` (segments are the pieces); `srt`, `vtt` and word
  timestamps are refused. 25 MB, 15 minutes. `coli router` forwards the route (its body limit follows).
- Checked with the scripted engine of `tests/test_gemma4_gateway.py` (prompt, frames, cutting, refusals), a router test, and
  9 deliberate breakages that the tests caught. On the real E2B: the 11 s JFK clip word for word in 9 s; a 44 s file in two
  pieces, cut at 26.85 s (mid-sentence: the looped clip has no pause). Quality beyond that is unmeasured.

### 2.12 Presets, request log, settings (`feat/presets-settings`, merged, pushed and live 2026-11-09)

- **Presets (C2):** `c/presets.py`, applied in `c/router.py`; `coli preset add|rm|show|list`; [presets.md](presets.md).
- **Request log (C4):** `c/request_log.py`; router and gateway lines joined by a trace id; [request-log.md](request-log.md).
- **Settings (C4):** `coli settings` over the tables of `ENVIRONMENT.md` plus a scan of the source for what is read; typo report.
- 43 new tests, 12 deliberate breakages caught. Run on a router with the real Gemma E2B (port 8010) before the deploy.
- **Dashboard (`feat/web-presets`):** Settings gets Presets (list, new, edit, delete, use) and Requests (the log, newest first);
  presets are a group in the chat's model list and choosing one starts the temperature, length, size and steps from it. New
  router routes `POST /v1/presets`, `POST /v1/presets/delete`, `GET /v1/requests`. English, German, Italian. Checked in a browser
  against a router with stub engines (`web/tests/presets_ui.py`); not yet against the live router.

### 2.13 Tray icon and background service (`feat/tray`, merged, live)

- **`c/service.py` / `coli service`:** start, stop, restart, status, link, config for `coli router` in the background: a key kept
  in `~/colibri-api-key.txt`, the router started without a window with the key in its environment, a stop that refuses while a
  request runs. The file names and defaults are those of `restart-colibri.ps1`. Settings in `service.json` (typos are errors).
- **`c/tray.py` / `coli tray`:** the notification-area icon (ctypes, nothing to install) with a menu: state, unload a loaded model,
  dashboard, copy key or link, start/stop/restart (with a "will be lost" wording while busy), logs, start with Windows (the `Run`
  key). [tray.md](tray.md)
- 44 tests (`tests/test_service.py`): config, the command line, a fake `/health` server, the stop rules, a real router started
  and stopped on a free port, the menu and icon logic, and on Windows the window, its menu and the `Run` key. The real window was
  also run: icon added, changed and removed, a message shown. Not done: a person clicking the menu.

### 2.14 Gemma 4 on the graphics card, video and more audio (G2, G5; merged, pushed, live 2026-11-10)

- **Dense matrices on Vulkan** (`QMat.vk`, `vk_mm` in `c/gemma4.c`, built with `make gemma4 VK=1`): attention, dense MLP, towers and
  the output head go through the existing `coli_vk_matmul`; the experts of the 26B-A4B stay on the CPU. The same text as the CPU,
  1.3-3x faster on prompts and pictures, 1.4-1.9x on decoding; not bit-identical (12 batch tests that compare to the bit fail with
  it on, by design). `G4_VK_RESERVE_GB` keeps device memory free, `G4_VK_DENSE=0` keeps the matrices on the CPU. One binary runs
  either way: `COLI_VULKAN` decides. [gemma4.md](gemma4.md#vulkan-for-the-dense-matrices), [gpu-offload.md](gpu-offload.md)
- **Video:** a `video_url` file (taken apart by ffmpeg) or a `video` part with the client's own frames; up to 32 frames of 70 tokens,
  each with its `mm:ss` as Gemma4Processor writes it. The sound of a video is not read. Engine picture limit 8 to 40.
- **Audio:** MP3 through minimp3 (`make acestep_audio_tool`), FLAC, Ogg and M4A through ffmpeg (`COLI_FFMPEG`, else the PATH); the
  declared format is only a hint, the bytes are sniffed. [gemma4.md](gemma4.md#video)
- Real runs of the E4B and the 31B (G3) came first, and the Stop test's Windows fixture bug was fixed on the way.

### 2.15 Card or CPU by free memory, and models turned off (`3b4bd293`, dashboard `7ae38131`; merged, pushed, live 2026-11-10)

- **`coli router --gpu auto|on|off`** (`c/gpu.py`): when a model loads, the router decides whether its worker runs on the card, from
  `nvidia-smi`'s free memory: `COLI_GPU_RESERVE_GB` (6) is kept back and `COLI_GPU_MIN_GB` (3) more is needed. The restart script and
  `service.json` set `COLI_GPU=auto`. `coli ps` says what each model runs on; that is the router's decision, not proof (ACE-Step
  says GPU but has no Vulkan path).
- **Measured** on the 5090 with the Qwen3-Coder expert tier: a 600-token prompt 24.2 s to 12.8 s, decoding 10-13 to about 17 tok/s,
  17.3 GB of the card (Gemma E2B/E4B/26B cost 2.4/4.4/2.5 GB; the desktop alone holds about 5.5 GB). With Coder on the card the
  26B goes to the CPU. [gpu-offload.md](gpu-offload.md)
- **Models can be turned off and pinned** (`c/model_settings.py`, `model-settings.json`): `coli model enable|disable|gpu NAME`,
  `GET` and `POST /v1/models/settings`, a request for a turned-off model is a `403 model_disabled`. The dashboard's Settings > Model
  has Runs on (automatic, card, CPU) and Turn off/on. [model-settings.md](model-settings.md)
- **`env` in `service.json`** gives the tray and `coli service` the same environment the restart script sets.

### 2.16 Context windows (`26af52c9`; merged, pushed, live 2026-11-10)

- The engines (`gemma4`, `qwen36`) announce `ctx=N` in their `CAPS` line and the gateway puts it on the model card as
  `context_length`; the router keeps the card of a model that has run, so it survives an unload. A model that has not run since the
  build shows none.
- **What is served is not what the model can do.** Gemma 4 serves 8192 unless `CTX` is set (the router has no per-model `--ctx`, so
  `CTX` in its environment reaches every Gemma); the Qwen3.6 family reads `Q36_MAXT`. On this machine since 2026-11-10: Gemma
  32768 and Qwen3-Coder 65536 (checked on the cards; no 65,000-token prompt has been run through Coder). What a larger window
  costs: [gemma4.md](gemma4.md#context-window).

### 2.17 The VS Code extension, `colibri-agent-connector` 0.4.0 (outside this repository)

Its own folder (`C:\code\colibri-vscode-extension\colibri-agent-connector`, no commits). Updated for the router:

- **Start, Stop and Restart Server go through `coli service`**, as the tray does, so Restart no longer replaces a router with a
  one-model `coli serve` (`colibri.serverControl`: `auto`, `service`, `serve`); a stop that would cut a running request asks first.
- **The API key is read from the key file** `coli service` keeps, for a server on this machine; health checks now send it (without it
  a keyed server looked like "not Colibri").
- **Colibri: Models...** and a status bar item: each model's state, memory, where it runs and when it unloads; Use for chat, Load,
  Unload, Turn off/on, Run on automatic/card/CPU. Models that are turned off are left out of the chat's picker, and a loading model
  says so instead of "Thinking...".
- **`context_length` from the card** replaces learning the window from an overflow (the `colibri.contextWindow` setting still wins, so
  `0` is the right value with a router).
- 36 unit tests for the parts that do not need VS Code (`npm run test:unit`). The menu and status bar were checked against the live
  router's API, not clicked through in VS Code.

### 2.18 Router lanes, the log, the API spec and the feature list (2026-11-16; ideas from reading ComfyUI, nothing copied)

- **A chat goes ahead of waiting pictures and songs** (`X-Colibri-Priority`; [router.md](router.md#priority)). A running request is never interrupted.
- **`GET /v1/logs`**, the dashboard's Settings → Log and the extension's "Colibri: Show Router Log" (0.10.0): the last 2000 lines the
  router and its models printed, in memory only.
- **The OpenAPI spec lists every route** (`docs/api-reference/openapi.json`; it had 11 paths and the servers answer 22) and
  `c/tests/test_api_spec.py` fails when a route in `openai_server.py` or `router.py` is missing from it, when the spec keeps a route that is gone, or
  when no page under `docs/` mentions a route.
- **`features` in the authenticated `/health`** (`c/features.py`, [api.md](api.md#which-features-this-server-has)): what this server can do,
  so the dashboard hides Presets, Requests and Log against a plain `coli serve`. A server without the block keeps every page.
  The extension still asks and reads a 404 as "not here"; new features should go in the list.
- **Security headers** (`c/web_headers.py`): `nosniff` and `Referrer-Policy` on every response, a content policy and `X-Frame-Options`
  on the dashboard's files, and a 403 for a browser request to `/v1/...` that is `Sec-Fetch-Site: cross-site` from an origin that is
  not allowed. [THREAT_MODEL.md](THREAT_MODEL.md).
- **Settings inside saved media** (`c/media_meta.py`): every picture's PNG and song's WAV carries a JSON record (prompt, size, steps, seed,
  model, a song's lyrics and language) that the dashboard's "Open the settings…" button reads back into the controls
  ([qwen-image.md](qwen-image.md#where-the-images-go), [acestep.md](acestep.md)). `COLI_MEDIA_META=0` leaves the files plain.
- **Model catalogue** ([router.md](router.md#models-that-are-not-there), [model-settings.md](model-settings.md#looking-inside-a-model-and-checking-its-files)): `coli model info` and
  `coli model verify [--quick]` (`c/model_info.py`: headers only; the manifest check plus each safetensors file against its own header, which
  finds a cut-short download without a manifest); the router remembers the models it has seen, so a drive that is unplugged or a folder
  that was damaged stays on the list with the reason, and a half-downloaded model is listed with how far it got instead of silently
  skipped; `GET /v1/models/installed` carries `details` and `reload_seconds`, and the Model page shows them.
- **Who the router stops, and memory pressure** ([router.md](router.md#which-model-goes-first-and-when-memory-runs-low)): eviction scores idle time times memory
  freed over the measured time to load again (kept in `router.json`, shown as `reload_seconds`), so a model off the slow drive stays; the
  reaper reads free memory each second and stops one idle model under a floor, then waits; the router remembers the peak memory of a model
  while it answers and plans with that; and with `--gpu auto` an idle model that holds the card is stopped when the next one would
  otherwise go to the CPU (`COLI_CARD_EVICT=0` for off). From 2026-11-17 it asks that model to give the card back and keep running
  instead of stopping it ([gpu-offload.md](gpu-offload.md#giving-the-card-back): `POST /v1/demote`, a `DEMOTE` frame, qwen36 only;
  `COLI_CARD_DEMOTE=0` for off).
- **Client tags, a picture quality check** (2026-11-17): `X-Colibri-Tags` on a request, kept on its request log line and its job and
  listed by with `?tag=name:value` ([api.md](api.md#client-tags)); the dashboard tags each chat. `c/image_quality.py` scores two pictures by
  SSIM ([qwen-image.md](qwen-image.md#checking-a-picture-against-a-baseline)); it settled that Qwen-Image on the card reproduces the CPU's picture.
- **Background jobs and the Library** (`c/jobs.py`, `c/library.py`; [api.md](api.md#background-jobs), [router.md](router.md#background-jobs)):
  `Prefer: respond-async` on a picture or song request makes the router answer 202 with a job and make the render apart from the connection,
  so a closed tab no longer cancels an hour of work. `GET /v1/jobs`, `GET /v1/jobs/{id}`, `POST /v1/jobs/cancel` (one call in every state);
  one job at a time in the low-priority lane; records in `jobs.jsonl` (a job in flight at a restart comes back failed, saying so); a client
  id (`X-Colibri-Job-Id`) makes a resend the same job. `GET /v1/library` and `/v1/library/{kind}/{name}` (both servers) list and serve the
  kept files, with the size and settings read from each file's header, byte ranges and a day of caching. Dashboard: a "Keep going if I
  leave" switch in the picture and music composers, and Settings → Library with a Jobs part. Not done: thumbnails made on the server, a
  URL in the response instead of base64, delete, the extension.
- **A failed start on the card is retried on the CPU** and remembered for 30 minutes ([router.md](router.md#a-model-that-will-not-start-on-the-card)),
  unless the card was chosen (a pin, `--gpu on`).

### 2.19 Gemma 4 on the chain (`feat/gemma4-chain`, 2026-11-18; merged, pushed (`877fa6bd`) and live the same day)

- **A decode row's layers as one command buffer** ([gemma4.md](gemma4.md#the-decode-chain-gemma4_chainh), [vulkan.md](vulkan.md#gemma-4-on-the-chain-gemma4_chainh)):
  the dense models run entirely on the card from the embedding to the logits (E2B 46 ms a token on the CPU, 28 to 34 with each matrix
  a call, **5.4** on the chain; E4B 97, 40, **3.8**); the 26B-A4B crosses to the host once a layer for its routed experts (64 to 80, **51**).
  The same words as the CPU, checked on tiny models of all three kinds (`tests/test_gemma4_chain.py`), on the real E2B, E4B and 26B-A4B
  (every step compared, residual within ~7e-5 of the CPU's) and through the gateway (two conversations interleaved on two slots, one of
  2,400 tokens). A device lost mid-step hands the same step to the CPU (injected and tested).
- **Two shaders' worth of new ops in the shared chain**, which the other engines do not use: `chain_attn512.comp` (heads up to 512; a
  copy of `chain_attn.comp`, which is untouched) and `GEGLU` and `SOFTCAP` in `chain_ew.comp`; tests in `tests/test_vk_chain.c`, which
  now builds on Windows (a `setenv` shim).
- **Not done:** a prompt chunk and several conversations decoding together on the chain (the per-matrix path as before), the 26B-A4B's
  experts on the card (done in 2.20), an integrated GPU (not measured: off there), partial chains and a KV split past the
  device's budget (the mirrors are sized to the context, or the chain declines).
- Switches: `COLI_VK_CHAIN=0` off, `COLI_VK_CHAIN_CHECK=1` compares every step with the CPU's.

### 2.20 Gemma 4's routed experts on the card (`feat/gemma4-experts-gpu`, 2026-11-18; merged, pushed (`c54ed0e9`) and live the same day)

- **The 26B-A4B's experts through the shared tier** ([gemma4.md](gemma4.md#the-expert-tier-the-26b-a4bs-routed-experts-on-the-card),
  [vulkan.md](vulkan.md#gemma-4-experts-on-the-tier)): all 3,840 (12 GiB) on an RTX 5090 after a 5 to 7 s warm start, a decoded token
  **25 to 28 ms** against 51 with the chain and the experts on the CPU (110 on the CPU), a 1,676-token prompt 112 tokens a second against
  55. The same words as the CPU: logits within 4e-5 on the real model, and two conversations interleaved on two slots (one of 2,467
  tokens) through a router, all seven turns word for word. The card takes 15.3 GiB with the tier (3.2 without), which leaves about 12 GiB
  for a second model on a 32 GB card.
- **Changes outside Gemma's engine** (all small): GeGLU (`gelu_pytorch_tanh(gate) * up`) as act 3 of `qmatmul_gate_up.comp` and
  `expert_act.comp`, `VKT_ACT_GEGLU` / `COLI_VK_ACT_GEGLU`, a case in the self-test; `COLI_VK_TIER_FAULT=n` (tests: fail the n-th batch);
  "gemma4" in route_trace.h's table of history writers; the tier's line says how many grouped batches ran. The grouped GEMM and GEMV
  shaders (AMD) were not changed: they fuse silu and are not taken with GeGLU.
- **In the engine:** `g4_vk_tier_start` (the budget leaves the dense matrices and the chain's mirrors; the history is `.coli_usage`
  beside the model, every expert floored at 1 so a card with room holds them all), `moe_routed_vk` (the card's batch beside the CPU's
  for the rest, every rank added in order), the history saved after each turn, the EMAP and TIERS lines say which experts the card
  holds. `COLI_VK_TIER=0` turns it off; `COLI_VK_TIER_GB` caps it.
- **Tests:** `tests/test_gemma4_tier.py` (5: every expert on the card and the CPU's continuation, a budget for a few experts, off,
  a batch that fails in a step, the history kept and read back). The 179 engine tests pass on the CPU; with the card on, the same
  12 batch bit-equality tests fail as before (they need batched decode equal to single-row decode to the bit).
- **Not done:** the router on the card (a routed layer is two round trips and costs 0.57 ms of the 0.8 ms a layer; the batch's own
  device time is 0.4 of that), the grouped routes with GeGLU (AMD), streaming for big prompt steps (a prompt goes in blocks of 64 rows), an integrated
  GPU (the tier is off there), a second device.

### 2.21 Z-Image-Turbo (`feat/zimage`, 2026-11-18; merged, pushed (`6a506de9`) and live the same day)

- **A second picture model**, `c/zimage` ([zimage.md](zimage.md)): Tongyi-MAI/Z-Image-Turbo (Apache-2.0, where Qwen-Image-2.1's weights
  are non-commercial) read straight from the diffusers checkpoint: Qwen3-4B text encoder, the 30+2+2 block S3-DiT, the 16-channel
  VAE decoder, 8 flow-matching steps with a fixed shift. Same line protocol and front ends as `qwenimage`, so `coli run`, `coli chat`,
  the dashboard and `/v1/images/generations` work for it. No editing, no guidance.
- **Checked** against fp32 diffusers stage by stage (bf16 weights match to about 1e-6; the int8 default keeps the step-0
  prediction within 4%), on the real model and on a random tiny pipeline that CI builds (`zimage-tiny-check`).
- **Measured** on the 14900KF and the 5090, 768x768, 8 steps: CPU 21.4 s a step, the 30 blocks as one Vulkan chain 1.4 s a step (13 s
  for the steps, 6 s for the CPU's VAE decode); 8.6 GB of RAM with the text encoder staged. Through the live router: 49 s for the first
  picture (it loads), 18 s after.

### 2.22 Installing and removing models while it runs (`feat/model-installer`, 2026-11-19; merged, pushed (`6b0e6432`) and live the same day)

- **`c/model_installer.py`**: fetch a catalog model as a background job (progress, cancel, resume, one at a time, a lock file between
  processes) and remove an installed one (only a direct child of a models folder, never a link, never one a download is writing;
  renamed out of the scan before the files go). Nothing happens without `confirm: true`, which the caller sends after showing the size
  and the licence. [model-install.md](model-install.md)
- **Router routes** `GET /v1/models/catalog`, `POST /v1/models/install`, `/install/cancel`, `/uninstall`, `GET /v1/models/installs[/{id}]`,
  and `features.model_install`; a loaded model is refused. **`coli model available | install | uninstall`** (`c/model_cli.py`) goes
  through the router when one answers and works on the local folders when none does; an empty answer (a null stdin) is not a yes.
- **The catalog's licence lines** are what each repository declares on Hugging Face (read 2026-11-19).
- **Dashboard** (Settings > Model): an *Install a model* list with size, licence and fit, a confirmation, a progress bar with Stop,
  and Remove on each model. **VS Code extension 0.14.0** (outside the repository): the same two entries in the models menu.
- **Tests:** `tests/test_model_installer.py` (25), `test_router_install.py` (12), `test_model_cli.py` (19), `web/src/lib/installs.test.ts`,
  `web/tests/install_ui.py` (a real browser against a router and a fake hub), and a run of the extension's API client against a live
  router. The install list of the Makefile needed the two new modules (`test_install_closure` caught it).
- **Not done:** building the engine of a model that has none (that is `coli setup`), a download queue, several downloads at once, moving
  a model, translations of the dashboard strings (English only; the others fall back).

### 2.23 Embeddings: Qwen3-Embedding on `/v1/embeddings` (`feat/embeddings`, 2026-11-19; merged, pushed (`a6a2407a`) and live 2026-11-20)

- **`c/qwen3embed`** ([embeddings.md](embeddings.md)): Qwen/Qwen3-Embedding 0.6B, 4B and 8B (Apache-2.0) read straight from the
  checkpoint: the tokenizer, the Qwen3 decoder, last-token pooling, L2 normalisation, an optional `Instruct: ... Query:` prefix for
  queries, and `dimensions` (the first N components, normalised again). The layers are ACE-Step's text encoder's (that encoder is
  Qwen3-Embedding-0.6B), so `acestep_text.h` only gained a loader that takes the encoder's own folder. Texts go through the layers
  together (up to 4096 tokens at a time, the matrix products over all the rows; RoPE and a causal attention kernel per text), which is
  what makes a list of short texts cost a tenth of what the same texts one by one do.
- **Gateway and router**: `POST /v1/embeddings` in OpenAI's shape (`input` a string or up to 2048 strings, `encoding_format` float or
  base64, `dimensions`, `usage`), plus `instruction` and `truncate`; `embedding_engine.py` drives the engine over its line protocol;
  the model card says `capabilities: ["embeddings"]` and an `embedding` block; the router forwards the path, and every other route
  answers an embedding model with a 400 that names the right one. A new family `qwen3_embedding` (told apart from a chat Qwen3 by
  `modules.json` and `1_Pooling/config.json`), three catalog entries (1.2, 8.1 and 15.2 GB, Apache-2.0), `make qwen3embed` and the CI,
  release and clean lists. OpenAPI and `api.md` have the route.
- **Checked** against transformers (the model card's recipe) on the real models: the 0.6B and the 4B in fp32 match to 1e-7 on every
  component (plain, query, empty, non-ASCII, code, `dimensions`, a 300-token text, a batch of seven different lengths in one request,
  a text cut at 8192 tokens), with the same token counts; the 8B against a bf16 reference (its fp32 one needs 32 GB), cosine 0.9998 or
  better. The real-engine test keeps stored transformers vectors (`tools/qwen3_embedding_0.6b_reference.json`), so it needs neither torch
  nor transformers.
- **Measured** on the 14900KF, CPU: 64 texts of 24 tokens in 2.6 s (0.6B), 17.6 s (4B), 31.2 s (8B); one text 0.10, 0.51, 0.84 s; a text
  of 880 tokens 3.0, 13.2, 20.9 s. A first version took 27 s for the 64 texts: the DiT's attention routine forks and joins per head, which
  costs more than the attention of a short text, so the engine has its own for texts up to 2048 tokens.
- **`coli`**: `coli info` describes an embedding checkpoint; `coli run`, `chat`, `tune` and `--ctx` refuse one and point at the route.
  The dashboard keeps embedding models out of the conversation's model list; its installer list and the extension's menu say
  "embeddings" for the kind.
- **Tests:** `tests/test_embeddings_gateway.py` (36, a real gateway over `tools/embedding_stub.py`; five mutations of the code each
  broke at least one), `tests/test_qwen3embed_real.py` (13, the real engine, skipped without `QWEN3EMBED_MODEL`; three mutations of the C
  each broke it), a router test, the registry, catalog and API-spec tests updated, `web/src/lib/api.test.ts`, the extension's unit tests.
- **Not done:** a GPU path (the Vulkan chain would make the 4B and 8B several times faster), token arrays as `input`, other embedding
  models, a reranker (built next: 2.24), `coli embed`, translations of the one new dashboard string.

### 2.24 Reranking: Qwen3-Reranker on `/v1/rerank` (`feat/reranker`, 2026-11-20; merged, pushed (`ea3f4bda`) and live the same day)

- **The same engine**: `c/qwen3embed` runs a Qwen3-Reranker checkpoint (0.6B, 4B, 8B; Apache-2.0) when the folder has
  `1_LogitScore/config.json` ([rerank.md](rerank.md)). A pair is written the way the model card does (a system prompt, then
  `<Instruct>`, `<Query>` and `<Document>`, then the assistant turn with an empty think block), run through the decoder together with
  the other pairs of the request, and the score is the chance of "yes" against "no" at the last position: `sigmoid(logit(yes) -
  logit(no))`. Only those two rows of the output matrix are read (`lm_head.weight`, or the embedding table when the weights are tied, as
  they are on the 0.6B and the 4B). `acestep_text.h` gained a loader that finds the `model.` prefix of a language-model checkpoint and
  returns chosen rows of the head; the layers, the batching and the attention are the embedding model's.
- **Gateway and router**: `POST /v1/rerank` and `/v2/rerank` in the shape Jina's and Cohere's clients use (`query`, `documents` as strings
  or `{text}`, `top_n`, `return_documents`; `results[]` best first with `index` and `relevance_score`), plus `instruction` and
  `truncate`; `embedding_engine.py` has a `RerankEngine` beside the embedding one; the model card says `capabilities: ["rerank"]` and a
  `rerank` block; every other route answers a reranking model with a 400 that names the right one, and the new refusals share one helper
  (`engine_purpose`). A new family `qwen3_reranker` (told apart by `1_LogitScore/config.json`), three catalog entries (1.2, 8.1 and 16.4
  GB, Apache-2.0), a new modality `rerank` for the router, `coli`, the dashboard and the extension. No new binary: the CI, release and
  clean lists are unchanged.
- **A bug in the embedding wrapper, found by a test of this work**: an `ERROR` line with no id (what the engine sends for a request line
  it cannot read, such as one over 64 MiB) was ignored as stray output, so the gateway would have waited for it for ever. It is now the
  answer to the request in hand (requests go one at a time).
- **Checked** against transformers (the model card's recipe, fp32) on the real 0.6B: the logit differences agree to 1.6e-5 and the
  token counts exactly, on nine pairs (default task, a custom task, two languages, code with tabs, the empty document, a 450-token pair).
  `tools/rerank_ref.py` captures and compares; the golden values are in `tools/qwen3_reranker_0.6b_reference.json`.
- **Checked on the other sizes**: the 4B (tied head) and the 8B (its own `lm_head.weight` in the fifth of five shards) against bf16
  references: every logit within 0.11 (bf16 quantises them to 1/16 or 1/8), identical token counts.
- **Measured** on the 14900KF, CPU: 64 pairs of about 107 tokens in 10.6 s (0.6B), 43.5 s (4B), 72.2 s (8B); one pair 0.2, 0.8, 1.4 s;
  32 pairs of about 290 tokens 15.5, 63.8, 101 s. Half of a short pair is the fixed prompt, instruction and query, which every pair of
  a request repeats: sharing them is built in 2.25.
- **Tests:** `tests/test_rerank_gateway.py` (33, a real gateway over `tools/rerank_stub.py`; eight mutations of the Python each broke at
  least one), `tests/test_qwen3rerank_real.py` (6, the real engine, skipped without `QWEN3RERANK_MODEL`; four mutations of the C each
  broke it), a router test, the registry and API-spec tests updated, `web/src/lib/api.test.ts`.

### 2.25 Rerank: a shared start, and `coli embed` / `coli rerank` (`feat/rerank-prefix`, 2026-11-20; merged, pushed (`f29d170c`) and live the same day)

- **The shared start**: the pairs of one rerank request begin with the same system prompt, instruction and query. `qwen3embed` finds the
  longest run of token ids all the pairs share (cut so that each keeps its last row, and at least 8 tokens, and only when no pair is
  over 2048 tokens), runs it through the layers once, keeps the keys and values of every layer (`QePast`), and the pairs then run only
  their own tokens, with positions that start after the shared part and attention that reads the shared keys first. The scores are
  identical to the last digit with it on and off (checked on twelve request shapes: distinct documents, shared first words,
  identical documents, one document the start of another, empty ones, unicode, a custom instruction, long and very long ones,
  truncation). `COLI_RERANK_SHARE=0` turns it off. 0.6B on the 14900KF: 64 short pairs 11.3 s to 4.6 s, 32 pairs of 290 tokens 14.2 s
  to 11.0 s, 8 short pairs 1.18 s to 0.40 s. The 4B and 8B were not measured again.
- **`coli embed` and `coli rerank`** (`c/retrieval_cli.py`): the retrieval routes from the terminal, through the running server's API.
  `coli embed TEXT...` prints the width, the first numbers of each vector and, for two to ten texts, a cosine table; `coli rerank QUERY
  DOCUMENT...` prints the documents best first with score and position; both take `--file` (or `-` for standard input), `--model`,
  `--instruction`, `--json`; `--dimensions` for embed, `--top` and `--truncate` for rerank. Plain one-line refusals for no model
  installed, a server that does not answer, a wrong key and a server error.
- **Tests:** `tests/test_retrieval_cli.py` (11, a stub server; three mutations each broke a test), a new real-engine test that
  compares the shared start with the one-pair-at-a-time path (two of three mutations of the C broke it; the third, dropping the
  one-token cap, is already enforced by the loop beside it).

### 2.26 Embeddings and reranking on the graphics card (`feat/embed-gpu`, 2026-11-20; merged, pushed (`98abb119`) and live the same day)

- **One frame a forward**: with `make qwen3embed VK=1` and `COLI_VULKAN=1`, `qwen3embed` runs the embedding and reranking models'
  whole stack of layers on a Vulkan device in a single submission (`qe_chain_run`, on the dense chain of `vk_chain.h`: norms, q/k/v,
  head norms, RoPE, `vkc_attn_w` per text, o, SwiGLU, down, residuals). The activations stay on the device; the last row of each text
  comes back, and for a rerank request's shared start its keys and values (kept on the device and on the host). Matrices go up as
  bf16 through `coli_vk_tensor_ensure` and are shared with the per-matrix path (`qe_lin`, `COLI_VK_CHAIN=0`). `READY` says
  `"device":"vulkan"` or `"cpu"`. A refused allocation runs that forward per matrix; a failed frame turns the chain off and the CPU
  redoes the forward. The Makefile rule links `$(VK_OBJ)`.
- **Measured** (RTX 5090 against the 14900KF): 20 passages, 0.6B 5.9 s to 0.37 s, 4B 24.8 s to 1.2 s, 8B 40.8 s to 1.96 s (all three
  agree with the CPU's scores); 64 embeddings of 230 tokens on the 0.6B 23.5 s to 1.13 s; 20 passages of 520 tokens on the
  4B 61.6 s to 3.05 s; a 7000-token document 40 s to 4.1 s; 120 passages over several batches 39.8 s to 2.2 s. The per-matrix path first
  built (activations over PCIe for every product) got only 3.3x on the 4B: a product's 98 MB crossing at 9 GB/s was most of its time.
- **Checked** against the CPU on twelve request shapes (largest logit difference 3.9e-5, same token counts, truncation, empty and
  unicode documents, 300 tiny documents, a request over one batch, a 7000-token text) and on embeddings (cosine 1.0000000); the
  real-engine tests gained a card-against-CPU test each (skipped without a Vulkan build and device); seven mutations of the device
  code (RoPE pairs, attention positions and cache rows, the q norm weight, SwiGLU, the last-row pick, a residual) each broke them.
- **Through the router** (2026-11-20, after the deploy): 20 passages of about 400 tokens ranked in 0.9 s (0.6B), 3.0 s (4B) and 4.6 s (8B)
  with the model loaded; the model cards say `device: vulkan`. The 0.6B embedding model loaded last, with the three rerankers holding
  about 25 GB of the card, and the router put it on the CPU.
- **A slip, fixed**: the first commits of this work wrote six files with CRLF line endings (a Python text-mode write on Windows), so
  every line of the Makefile, `openai_server.py`, `rerank_stub.py`, two tests and `docs/vulkan.md` showed as changed. `98abb119` put them
  back to LF; the content did not change.

### 2.27 The models page (`docs/models-catalogue`, 2026-11-20)

- [models.md](models.md): every model the engines here can serve, with size, RAM, window, what it takes in and can do (tools, thinking,
  pictures, audio, video, JSON mode), the route it answers, the measurements, and a table of what is installed on this machine, built from
  `setup_catalog.py`, `family_registry.py`, the family pages and the live `GET /v1/models`.
- `docs.json` lists the pages the fork added since the navigation was written (models, router, model-install, zimage, embeddings, rerank).

### 2.28 A small model gets the card when it frees up (`feat/promote-to-card`, 2026-11-20)

- **The gap**: the router chooses card or CPU when a model loads and never looks again. The 0.6B embedding model, loaded while the
  rerankers held the card, ran on the CPU (23.5 s for a batch the card does in 1.13 s) for as long as it stayed loaded.
- **`Pool.promote`** (`c/router.py`, called by the reaper beside `relieve`): an idle embedding or reranking model whose start on the CPU was
  because the card was full (`gpu.Decision.room`, set by the `auto` policy only) is stopped when the card has room for it, so the next
  request loads it on the card. It must have been idle 20 s and running 60 s, and the card must have the policy's reserve plus its size
  with a quarter more. Chat models, pinned models, failed starts, models on the card and demoted ones are never touched. `/health`:
  `router.card_promote`; `COLI_CARD_PROMOTE=0` turns it off.
- **Tests:** `tests/test_router_promote.py` (12: the move, a full card, a model too big for what is free, a model answering, too soon after
  use and after start, pinned or failed starts, a model on the card or demoted, chat models, the switches, `/health`, and the reaper doing
  it by itself); eleven mutations of `promote` and its hook each broke at least one.

## 3. Planned and open

### 3.1 Waiting for your decision

1. ~~Deploy Gemma 4~~ done 2026-11-07 (see section 1).
2. ~~Push local `main` to the fork~~ done; the fork has everything except the latest docs commits (push when you say).
3. ~~GPU offload for Gemma 4~~ done 2026-11-10 (2.14, 2.15).
4. ~~`coli tray --install`~~ done: the `Colibri tray` entry is in the sign-in list (seen 2026-11-18).
5. **Try the 65536 window on Qwen3-Coder with a real long conversation** and watch memory; go back with `-Ctx 32768` if it is too much.

### 3.2 Gemma 4 follow-ups

- The E4B and the 31B ran on real weights on 2026-11-10 (text, pictures, and audio on the E4B); neither is installed in `colibri-models`. The 31B leaves about 5 GB of a 64 GB machine free.
- `coli setup` cannot list Gemma: its catalogue holds only ready-to-run downloads, and Gemma needs conversion. That would take a
  container hosted on Hugging Face (publishing, so your call). Instead, `feat/gemma4-setup` (`fdebaae5`, merged) lets `coli convert --repo` take a local directory, tests the routing, documents the conversion and
  fixes the Makefile's "text only" comment.
- ~~Video, and audio other than WAV~~ done 2026-11-10 (2.14).
- Gemma's window is 32768 here by `CTX`; the automatic expert-cache sizing counts a conversation's full layers for at most 8192
  positions, so with a much larger window set `--cap` yourself (gemma4.md, Context window).

### 3.3 Items from colibri-ideas.md

Status as of today (the file itself has been brought into line with this):

| Item | Status |
|---|---|
| A1-A5 router, load on demand, memory fit, one render at a time, limits | **done**, live |
| A0 one server per model on separate ports | set aside, the router replaced it |
| B1-B3 `keep_alive`, `coli ps` / `coli stop`, Unload button | **done**, live |
| C1 `coli launch` | **done**, merged and live: Claude Code and opencode via the environment; Codex needs a Responses API colibri lacks; settings-file tools left out; not run against the real tools |
| C2 Named presets (saved sizes, steps, durations that show up as model names) | **done**, live (2.12); dashboard live (`feat/web-presets`) |
| C3 Windows tray or autostart (would replace the restart script) | **done**, live (2.13), and it starts at sign-in |
| C4 One place for settings, request logging | **done**, live (2.12); the log is in the dashboard on `feat/web-presets` |
| C5 Embeddings, structured JSON output, speech to text and back: check what the chat engines have | **surveyed** (no embeddings, no speech route; STT only through Gemma E-model chat); structured JSON **built**, live (2.10); embeddings **built** 2026-11-19 (2.23) |
| D1 Delete the original after convert | mostly there (`coli convert --repo` does it); not built for `--indir`, on purpose |
| D2 Free pipeline stages | already in the engine (`COLI_IMG_TE=stage`, about 14 s per new prompt); not needed on 64 GB |
| D3 Try `TRUNK_RESIDENT_LAYERS` on the dense Qwen3.8-27B | open, S |
| E1 Save finished songs | **done**, local only (see 2.4) |
| E2 Audio-in cases in the small CI fixture | open, S |
| E3 ACE-Step's 5 Hz planner (1.7B, also 0.6B and 4B) | open, L |
| E4 Lego, extract and complete tasks (need other checkpoints) | open, L |
| E5 Push `main` to the fork | **done** (the latest docs commits wait for your go-ahead) |
| F (streaming the DiT from disk, LoRA training on streamed layers, AirLLM-style loading for MoE) | decided against |

### 3.4 Other open points

- **Catching up with upstream**: 1,412 commits behind. Merging them is a large job with conflicts in `openai_server.py`, `coli`,
  `family_registry.py` and the Makefile; decide whether to do it, or to cherry-pick what is wanted.
- **Qwen-Image editing**: Linux, ASan and UBSan runs; a Vulkan check of the chain with an edit prefix; a real edit at the 1024
  reference; translated READMEs (it, ja, zh, id) for editing.
- **Graphics card**: Qwen-Image on Vulkan (`make qwenimage VK=1`) is unmeasured; ACE-Step has no Vulkan path. Making room on the card **is
  built** (2.15, [router.md](router.md)): a model that would start on the CPU only because the
  card is full makes the router demote (qwen36 engines) or stop (the others) an idle holder of the card, but only one idle for 60 s.
  Checked live 2026-11-20: with the 4B and 8B rerankers idle, the embedding model loaded on the card and the 4B was stopped. A small
  model that started on the CPU gets the card when it frees up (2.28). What is not built: putting a demoted model or a chat model back on
  the card when it is freed (they stay where they are until they unload, 30 minutes by default), and demoting engines other than `qwen36`.
- **Context**: E4B and the 26B show no `context_length` until they have run once; a per-model window would need a per-model setting
  in the router (today `CTX` and `Q36_MAXT` are per family, in the router's environment).
- **VS Code extension**: not done yet: the server's KV slot (`cache_slot`) so a long chat is not read again, picture, audio and video
  attachments, presets from the settings, the reasoning switch per model, the key in SecretStorage; the roadmap in its README has them.
- **ACE-Step**: rebuild `acestep.exe` in the live clone if audio in is wanted (see section 1); the upstream handler once rewrote
  files in `ace-step-1.5` (weights untouched); the 1.7B planner weights sit in that folder unused.
- **CI**: never run on GitHub. Turning it on is only worth it if you want Linux and sanitizer coverage.
- `docs/repo-deep-dive` holds a repository deep-dive that is not in `main`.
