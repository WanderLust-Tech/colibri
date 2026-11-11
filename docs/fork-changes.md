# What this fork changes, and what is still planned

Written 2026-11-07 for the owner of this fork (WanderLust-Tech/colibri, used on one Windows machine). It lists
everything that differs from the checkout we started from, where each piece stands (built, merged, pushed, running),
and what is still open, including the items in [colibri-ideas.md](colibri-ideas.md). Brought up to date on 2026-11-10
(Gemma on the graphics card, video, the card-or-CPU choice, model settings, context windows, the VS Code extension).

## 1. The baseline and how far we have moved

| | |
|---|---|
| Original checkout | `e0db083c`, upstream PR #2139 merged, 2026-10-17 (upstream v2.0.0 plus what followed) |
| Local `main` now | the tip of `main` (`git log origin/main..main` lists what the fork lacks: on 2026-11-10 one docs commit, Gemma's context window) |
| Size of the difference | 76 commits (72 of ours, 4 merges), 206 files, +51,505 / -314 lines (2026-11-10) |
| Remotes | `origin` = the fork (WanderLust-Tech), `upstream` = JustVugg. Nothing here is meant for upstream; no PR has been opened there |
| Upstream since then | `upstream/main` was 1,412 commits ahead of our baseline at the last fetch (2026-10-31). We have not merged any of it |
| Fork's `main` | `26af52c9` (pushed 2026-11-10); local `main` is ahead by the docs commits that follow it |
| Live server | clone `C:\Users\kp-mi\colibri` at `26af52c9` (brought there by `git pull` from the fork), running `coli router` on port 8000 through `C:\Users\kp-mi\restart-colibri.ps1 -Router` (2.9 lists what that script sets) |

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
| Tray icon and background service (`coli tray`, `coli service`, C3) | `feat/tray` | yes (`ecfd9219`) | yes | yes (2026-11-09): live clone at `ecfd9219`, tray started from it; router on 8000 not restarted (the tray is not part of it); the Windows sign-in entry is not installed |
| Gemma 4 dense matrices on the Vulkan device (G2) | `396a774f` | yes | yes | yes (2026-11-10): `gemma4.exe` is the `VK=1` build; the router decides per model whether it runs on the card (2.14) |
| Gemma 4 video, and audio that is not WAV (G5) | `b6fae35c` | yes | yes | yes (2026-11-10): ffmpeg 9.0.2 installed with winget, `acestep_audio_tool.exe` built |
| Card or CPU by free memory (`coli router --gpu auto`), models turned off and on, `service.json` `env` | `3b4bd293` | yes | yes | yes (2026-11-10): `COLI_GPU=auto` in the restart script and in `service.json` |
| Dashboard: Runs on and Turn off/on per model | `7ae38131` | yes | yes | yes (2026-11-10): `web/dist` rebuilt in the live clone |
| `context_length` on the model card | `26af52c9` | yes | yes | yes (2026-11-10): `gemma4.exe` and `qwen36.exe` relinked, router restarted |
| VS Code extension 0.4.0 (outside this repository, 2.17) | none: its own folder, no commits | n/a | n/a | the package is built and installed on this machine |

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

## 3. Planned and open

### 3.1 Waiting for your decision

1. ~~Deploy Gemma 4~~ done 2026-11-07 (see section 1).
2. ~~Push local `main` to the fork~~ done; the fork has everything except the latest docs commits (push when you say).
3. ~~GPU offload for Gemma 4~~ done 2026-11-10 (2.14, 2.15).
4. **`coli tray --install`** (the Windows sign-in entry for the tray) is yours to run; it was blocked when I tried, so I did not work around it.
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
| C3 Windows tray or autostart (would replace the restart script) | **done**, live (2.13); started by hand, not yet installed to start at sign-in |
| C4 One place for settings, request logging | **done**, live (2.12); the log is in the dashboard on `feat/web-presets` |
| C5 Embeddings, structured JSON output, speech to text and back: check what the chat engines have | **surveyed** (no embeddings, no speech route; STT only through Gemma E-model chat); structured JSON **built**, live (2.10) |
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
- **Graphics card**: Qwen-Image on Vulkan (`make qwenimage VK=1`) is unmeasured; ACE-Step has no Vulkan path; unloading an idle model
  that holds the card to make room for another is not built (the router only picks card or CPU at load time).
- **Context**: E4B and the 26B show no `context_length` until they have run once; a per-model window would need a per-model setting
  in the router (today `CTX` and `Q36_MAXT` are per family, in the router's environment).
- **VS Code extension**: not done yet: the server's KV slot (`cache_slot`) so a long chat is not read again, picture, audio and video
  attachments, presets from the settings, the reasoning switch per model, the key in SecretStorage; the roadmap in its README has them.
- **ACE-Step**: rebuild `acestep.exe` in the live clone if audio in is wanted (see section 1); the upstream handler once rewrote
  files in `ace-step-1.5` (weights untouched); the 1.7B planner weights sit in that folder unused.
- **CI**: never run on GitHub. Turning it on is only worth it if you want Linux and sanitizer coverage.
- `docs/repo-deep-dive` holds a repository deep-dive that is not in `main`.
