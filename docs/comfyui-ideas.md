# Ideas from ComfyUI

What we can learn from [ComfyUI](https://github.com/Comfy-Org/ComfyUI), the node-based image and video generation
program. Read on 2026-11-16 from a local clone (`D:\Code\ComfyUI`, branch `master`, version 0.45.0).

**Reviewed up to this ComfyUI commit** (so the next look only needs what came after):

| | |
|---|---|
| Repository | https://github.com/Comfy-Org/ComfyUI (local clone `D:\Code\ComfyUI`, branch `master`) |
| Commit | `e9afbccc2669677076feb08e52658f129cd8c12b` — "Make the cuda version warning yellow. (#17769)", 2026-11-15 |
| Reviewed | 2026-11-16 |
| Next review | `git -C D:\Code\ComfyUI fetch` then `git -C D:\Code\ComfyUI log --oneline e9afbccc..origin/master`; look at changes under `server.py`, `execution.py`, `comfy_execution/jobs.py`, `comfy/model_management.py`, `comfy/memory_management.py`, `app/assets/`, `comfy_api/feature_flags.py`, `middleware/` (the files this page cites). If that fails with "unknown revision", the history was rewritten: find the commit by its subject with `git log --grep`. |

When a later review is done, change the commit above to the new one.

**Licence: read this first.** ComfyUI is GPL-3.0; Colibri is Apache-2.0. Copying its code into Colibri would put the
copied part under the GPL, so **nothing here is copied**. The file names below say where an idea lives so it can be read;
every item is described in our own words and would be written from scratch.

**How this was done.** Three readers each took one area (memory and model management; execution, queue and API; model
files, settings, agents and security) and were told what Colibri already has, so nothing here is something we have. A
sample of their claims about Colibri was then checked against the live clone (listed under "Checked"). The rest of their
claims about ComfyUI are theirs and were not re-read.

## What ComfyUI is, in five lines

- A graph of nodes (load model, encode text, sample, decode, save) that a web page builds and a Python server runs.
- `POST /prompt` checks the whole graph, queues it and returns an id at once; one worker runs jobs one at a time, and
  everything the page learns afterwards comes over one websocket (progress, previews, results, errors).
- Memory is managed by one module that tracks every loaded model, evicts the least useful when room is needed, and can
  load a model partly onto the card and keep the rest in RAM.
- Every saved picture carries the graph that made it inside the file, so dropping it on the page rebuilds the work.
- It is built for a person running it locally, so it has a small, explicit idea of what a local server must refuse
  (foreign origins, files that browsers would run).

## Status

| Item | State |
|---|---|
| 1 Settings inside saved media | done 2026-11-16 (PNG iTXt, WAV LIST/INFO; the dashboard reads them back through a button; `coli run` and `coli music`, which write their own files, do not tag them) |
| 2 Jobs that survive the page | open |
| 3 A Library tab | open |
| 4 Smarter memory (4a cost-aware eviction, 4b pressure watchdog, 4c demote instead of stop) | open |
| 5 Failure ladder for a worker that will not load | done 2026-11-16 |
| 6 Gateway feature list | done 2026-11-16 (server and dashboard; the extension still reads a 404) |
| 7 Keep the API spec honest | done 2026-11-16 |
| 8 Security headers on the dashboard | done 2026-11-16 |
| 9 Model catalogue (offline entries, verify, header info) | open |
| 10 Smaller ideas | priority lane and log tail done 2026-11-16 (the router needs a restart for both); SSIM baselines and echoed metadata open |

Suggested order: a batch of the small ones that stand alone (1, 5, 6, 7, 8), then 2 and 3 as one project, then memory
(4a, 4b) once there is a measurement of what the reaper gets wrong today.

## Checked against Colibri (2026-11-16, live clone)

- `save_png` in `c/image_engine.py` writes the picture bytes with no text chunks: nothing about the prompt, seed or
  settings is inside the file (the file name has a short slug of the prompt and the seed).
- `docs/api-reference/openapi.json` lists 11 paths: `/experts`, `/health`, `/profile`, `/v1/audio/music`,
  `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/models`, `/v1/models/installed`, `/v1/models/switch`,
  `/v1/models/{model}`. It does not list the image routes, transcriptions, presets, requests, model settings, unload,
  metrics or systemone.
- `c/openai_server.py` and `c/router.py` set no `Content-Security-Policy`, `X-Content-Type-Options`, `X-Frame-Options`,
  `frame-ancestors` or `Referrer-Policy`.
- The router picks the model to stop with `min(victims, key=lambda w: w.last_used)`: least recently used only.
- `c/router.py` has no retry for a worker that fails to load; the answer is a 503.

## A. Saved media and jobs (the biggest gap)

- [x] **1. Settings inside saved media (S to write, M for the dashboard).** ComfyUI writes the prompt and the graph into
  the PNG's text chunks, so a picture can be reloaded into the page. Ours keeps every render in `~/colibri-images` but the
  file knows nothing about how it was made, and a render can take an hour. Idea: one text chunk with prompt, negative
  prompt, seed, steps, size, model id and preset; tags in the music files; in the dashboard, dropping a file refills the
  form. *Where in ComfyUI:* `nodes.py` (SaveImage), `comfy_extras/nodes_images.py`. *Nearest here:* `save_png` in
  `c/image_engine.py`.
- [ ] **2. Jobs that survive the page (L; the list/cancel part is S to M).** Image and music generation is one streaming
  request, so closing the tab, a sleeping browser or an extension reload cancels it. ComfyUI answers a submit with a job id
  at once, runs the job apart from the connection, and lets any client re-attach: the first thing a re-attached client
  gets is a snapshot of where the job is. Idea: a `background: true` request flag returning an id; job records kept on
  disk like the request log (ComfyUI keeps its history only in memory and loses it on restart, which is its weak point);
  `GET /v1/jobs` (status filter, limit, offset), `GET /v1/jobs/{id}`, `POST /v1/jobs/{id}/cancel`. Cancel is the same call
  in every state: running is interrupted, queued is removed, finished or unknown answers "not cancelled" and not an error.
  Two details worth keeping: the interrupt is done only if that job is still the running one, and the interrupt flag is
  cleared at the start of each job, so a cancel cannot leak onto the next. A client-supplied id makes a resubmit
  idempotent. *Where:* `server.py` (`post_prompt`, `websocket_handler`, the `/api/jobs*` routes), `execution.py`
  (`PromptQueue`), `comfy_execution/jobs.py` (`classify_job_for_cancel`, `JobStatus`). *Nearest here:* `Gate.waiting` in
  `c/router.py` (invisible to clients), `X-Request-Id`, `GET /v1/requests`. Gives a Stop button for a queued request, a
  visible queue position, cancel from another tab.
- [ ] **3. A Library tab (M).** Nothing serves or lists the saved pictures. Idea: `GET /v1/images/files` and a view route
  with a resize parameter for thumbnails, path-traversal guards, `nosniff`, and a day-long cache for images; width, height
  and duration recorded per output. Also lets a response carry a URL instead of base64 for large pictures and songs.
  *Where:* `server.py` (`/view`, `/view_metadata`), `middleware/cache_middleware.py`,
  `comfy_execution/media_enrichment.py`. *Nearest here:* `_finish_image` saves the file; the dashboard has no gallery.

## B. Memory and the router

- [ ] **4a. Cost-aware eviction (S).** Instead of "least recently used", score each idle worker by idle time times bytes
  freed, divided by what it costs to load again. Record each model's measured load time and where its files live (USB or
  spinning disk versus NVMe), so a model on the slow D: drive stays loaded longer than one on fast storage. *Where:*
  `comfy/model_management.py` (`free_memory`'s sort key), `comfy_execution/caching.py` (`RAMPressureCache`),
  `comfy/storage.py`. *Nearest here:* `_plan` in `c/router.py`.
- [ ] **4b. Pressure watchdog in the reaper (S to M).** The reaper only reacts to the 30-minute timer, and memory is read
  only when a model loads, so a browser or game that spikes RAM pages the model that is answering. Idea: react to RAM or
  card pressure between requests, with hysteresis (free a little more than the shortfall, so it does not thrash) and a
  ladder: caches and KV slots first, then demote, then stop idle workers. On Windows, judge RAM pressure by swap use or
  very low available memory, not by "available" alone. *Where:* `comfy/system_memory.py`, `comfy/pinned_memory.py`,
  the post-node loop in `execution.py`. *Nearest here:* `reap()` in `c/router.py`.
- [ ] **4c. Demote an idle model off the card instead of stopping it (S in the router, M in the engine).** Ask the idle
  worker to drop its card memory and keep running on the CPU; count that memory as reclaimable when the router decides card
  or CPU. A reload from the USB disk costs minutes, and this would also fix the case where Qwen3-Coder on the card sends
  the next model to the CPU. *Where:* `model_management.free_memory`, `ModelPatcher.partially_unload`. *Nearest here:*
  open item H1; `GpuPolicy.decide` reads only `nvidia-smi`.
- [ ] **4d. Learn each job's peak memory (M).** Store the peak working set per model and size class, make room before a
  render, and size KV slots or batches from free memory. `c/image_engine.py` says its working buffers are "not included
  (being measured)". *Where:* `comfy/model_base.py` (`memory_required`), `comfy/samplers.py`.
- [ ] **4e. Working-set trim for idle workers (S, measure first).** After a short idle time, ask Windows to move a worker's
  read-only mapped weight pages to the standby list, so RAM looks free but pages return quickly. The expert caches and KV
  are ordinary memory and would go to the pagefile, so measure before trusting it.
- [ ] **Not recommended: GPU prefill with streamed weights (L).** Stream layers through the card for long prompts and keep
  decode on the CPU. It only pays where the arithmetic per byte is high (prefill); for one token at a time the bus is no
  faster than reading RAM. Related to the open "experts next step" in G2.
- [ ] **Not recommended yet: pick a quantisation by free memory (M).** Needs several files of one model on a tight disk.

## C. Robustness and the surface

- [x] **5. Failure ladder (S). Done 2026-11-16:** a failed start on the card is tried once on the CPU and remembered for 30 minutes ("A model that will not start on the card" in `docs/router.md`). Checked the side question: Qwen-Image's Vulkan VAE decode already works in bands and falls back to the CPU. If a Vulkan worker fails to load or dies, relaunch once on the CPU and remember that for a
  while; skip an attempt the estimate says cannot fit. Also worth checking whether Qwen-Image's decode has a tiled
  fallback for an out-of-memory. *Where:* `comfy/sd.py` (`VAE.decode`), the out-of-memory branch in `execution.py`.
  *Nearest here:* a 503 `model_load_failed`.
- [x] **6. Gateway feature list (S). Done 2026-11-16:** `features` in the authenticated `/health` (`c/features.py`; see "Which features this server has" in `docs/api.md`); the dashboard hides Presets, Requests and Log against a `coli serve`. The extension keeps asking and reading a 404. An authenticated `features` block (presets, request log, cached tokens, JSON mode,
  router, background jobs once they exist, upload limit) so the extension and the dashboard hide controls an older server
  does not have. The vsix and the server are deployed separately, so a new extension can meet an old server. ComfyUI sends
  flags both ways and drops a mistyped value with a warning. *Where:* `comfy_api/feature_flags.py`, `server.py`
  (`/features`). *Nearest here:* `context_length` on a model card, `/health`.
- [x] **7. Keep the API spec honest (S). Done 2026-11-16:** `c/tests/test_api_spec.py`; the spec went from 11 to 22 paths, and `Health` lists `features`. A test that pulls the routes out of `c/openai_server.py` and `c/router.py` and
  fails when `docs/api-reference/openapi.json` or `docs/api.md` is missing one; then fill in the missing paths. The
  extension and dashboard depend on exactly those routes. ComfyUI lints its spec in CI and has a test that fails when a
  route has no smoke test. *Where:* `tests-unit/assets_test/test_api_surface.py`, `openapi.yaml`, `.spectral.yaml`.
  *Nearest here:* `c/tests/test_api_matrix_docs.py` (family tables only).
- [x] **8. Security headers on the dashboard (S). Done 2026-11-16:** `c/web_headers.py`, "Static headers" in `docs/THREAT_MODEL.md`. `Content-Security-Policy`, `X-Content-Type-Options: nosniff`,
  `frame-ancestors`, `Referrer-Policy`; and refuse `Sec-Fetch-Site: cross-site` on `/v1` as a second check behind the
  origin allow-list. Defence in depth only: `web/src/components/Markdown.tsx` does not use `innerHTML`. Add a "static
  headers" line to `docs/THREAT_MODEL.md`. *Where:* `server.py` (`create_block_external_middleware`,
  `create_origin_only_middleware`).
- [ ] **9. Model catalogue (S to M).** (a) Keep an unplugged drive's models or a half-downloaded one visible, disabled,
  with a reason ("drive not connected", "download incomplete"), from a small last-known list; today `scan_installed`
  silently skips them. (b) `coli model verify` to re-check files against the manifest (the downloader has the check, the
  CLI does not expose it). (c) A header-only look at a container for architecture, quantisation, context length and
  source, shown by `coli model info` and on the dashboard card (this is open item H3: `context_length` only appears after
  the first load); name the missing piece when a container is not recognised. *Where:* `app/assets/`,
  `comfy/model_detection.py`, `app/model_manager.py`.
- [ ] **10. Smaller ideas.**
  - **Done 2026-11-16:** a priority lane, so a short chat goes ahead of queued music and image jobs (ComfyUI never
    interrupts a running job either; neither does this). Pictures and songs are low, the rest normal, and the header
    `X-Colibri-Priority: low|normal|high` overrides it; see "Priority" in `docs/router.md`. *Where:* `number` / `front` in `post_prompt`. *Nearest here:* a first-in-first-out gate.
  - An image-quality check by structural similarity against saved baselines (ComfyUI uses 0.95), which would settle open
    item H4 (Qwen-Image on Vulkan, unmeasured). *Where:* `tests/compare/test_quality.py`.
  - **Done 2026-11-16:** a live log tail for load failures and slow loads: `GET /v1/logs` (the last 2000 router and worker lines, in memory, needs the key), the dashboard's Settings → Log, and the extension's "Colibri: Show Router Log" (0.10.0). *Where:*
    `api_server/services/terminal_service.py`.
  - A small, size-capped client object echoed on every event and written to the request log, so the Requests tab and the
    extension can group by chat. Weak without item 2.
  - Per-user dashboard preferences kept on the server. Modest; skip if one browser is used.
  - Validation that reports every problem at once with stable type codes. The OpenAI error shape limits the gain.

## Looked at and not useful

- **Node-output caching and the graph** (`comfy_execution/caching.py`, `cache_provider.py`, `graph.py`): keyed by the
  signature of a node's inputs; Colibri has no graph, and the engine already keeps the prompt prefix.
- **Latent previews** (`latent_preview.py`, `preview_tonemap.py`): Colibri already has a fitted latent-to-picture preview.
- **Binary websocket frames:** server-sent events with base64 previews are enough here.
- **The assets database** (SQLite, migrations, content hashing, tags) and server-side workflow and user-data storage: heavy
  for a flat model folder, and it conflicts with the server keeping no transcript.
- **`comfy_api` versioning:** protects third-party Python nodes; the gateway has none. Only the feature list (item 6) carries
  over.
- **Memory code tied to PyTorch and CUDA:** the allocator tuning, CUDA graphs, pinned-memory registry, LoRA patching.
  Linux cgroup limits.
- **Quantisation formats** (`QUANTIZATION.md`): the packing and rotation tricks need CUDA kernels; a research project.
- **Enterprise policy and cloud deployment** (`app/governance.py`, `app/deploy_*.py`), **`app/agent_host.py`** (a plug-in
  contract for ComfyUI's own agent), **blueprints and subgraphs** (our presets are the equivalent), **`extra_model_paths`**
  (we have `COLI_MODELS_ROOT` and `--models-root`), **`SECURITY.md`** and **`AGENTS.md`** (ours are covered by
  `docs/THREAT_MODEL.md`; two house rules match what we do: no outbound requests, fail loudly on unsupported formats).
- **ComfyUI's own limits, not to copy:** history lost on restart, one global interrupt flag, one client at a time
  receiving progress.

## Where to look in ComfyUI (read only)

| Topic | File |
|---|---|
| Queue, submit, websocket, cancel | `server.py`, `execution.py`, `comfy_execution/jobs.py` |
| Job states, progress | `comfy_execution/jobs.py`, `comfy_execution/progress.py` |
| Memory, eviction, offload | `comfy/model_management.py`, `comfy/memory_management.py`, `comfy/model_patcher.py` |
| RAM pressure, pinned memory | `comfy/system_memory.py`, `comfy/pinned_memory.py`, `comfy_execution/caching.py` |
| Disk speed classes | `comfy/storage.py` |
| Settings in saved files | `nodes.py` (SaveImage), `comfy_extras/nodes_images.py` |
| Serving outputs, caching | `server.py` (`/view`), `middleware/cache_middleware.py` |
| Feature flags | `comfy_api/feature_flags.py`, `server.py` (`/features`) |
| Local-server safety | `server.py` (middleware), `middleware/`, `tests-unit/security_test/` |
| API spec and drift tests | `openapi.yaml`, `.spectral.yaml`, `tests-unit/assets_test/test_api_surface.py` |
| Model detection and metadata | `comfy/model_detection.py`, `app/assets/`, `app/model_manager.py` |
