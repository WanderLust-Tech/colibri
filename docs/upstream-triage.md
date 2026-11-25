# Catching up with upstream: a triage (2026-11-25)

The fork's `main` and `upstream/main` (JustVugg/colibri) share a base, `e0db083c` (2026-10-17). Since then the fork has 200 commits of its own
and upstream has **1,576 commits** (2,396 counting merges), up to 2026-11-22: about 480,000 added lines, 862 files under `c/`, 803 under `docs/`.
Nothing of the fork is meant for upstream ([fork-changes.md](fork-changes.md)); this is only about what to take from it.

## What was measured

- **Files changed on both sides: 81** (of 1,771 upstream files). Upstream-only files: 1,690; fork-only: 330.
- **Commits that touch only files the fork never changed: 690 of 1,576.** Those apply by file with no conflict. The other 886 mostly touch a few shared
  files: `docs/ENVIRONMENT.md` (227 commits), `c/Makefile` (199), `c/qwen36.c` (143), `CHANGELOG.md`, `docs/vulkan.md`, `ci.yml`,
  `c/vk_chain.c` (86), `c/openai_server.py` (84), `c/backend_vulkan.c`, `c/coli`, `c/vk_tier.c`.
- **A trial merge of all of upstream** (in a throwaway worktree, discarded): 57 conflicted files and 110 conflict hunks. The heavy ones are
  `c/openai_server.py` (27 hunks), `c/qwen36.c` (11), `c/setup_catalog.py` (9), `c/coli` (5), `c/Makefile` (4), `c/image_engine.py` (3); the rest are docs,
  README translations and the web dashboard (`web/src`, which both sides changed). A whole merge is possible, but it brings in 480,000 lines of engines
  nobody here can test (see CUDA below), so it is not the plan.
- **What upstream does not have, and the fork does: nothing to reconcile.** Zero upstream commits on Gemma 4, ACE-Step, speech (ASR, TTS, alignment),
  embeddings and reranking; the router, the model installer, the tray, the extension. All of that is the fork's own.

## What upstream did (commits by topic, overlapping)

CUDA 328, GGUF 203, Vulkan 149, speculative decoding / MTP 60, Windows 52, the Qwen engine (`qwen36`) about 90, images 22, LoRA 10, Responses API 9, TUI 6,
desktop app 6, MCP server fixes 4, integrated GPU 2. Also new model families (Inkling-2, DeepSeek V4.1, Kimi K3.1, Mistral, LFM3 vision), a `ui/` chat page
and a `desktop/` shell.

## Candidate batches, in the order worth taking them

| # | Batch | Why | Effort and risk |
|---|---|---|---|
| 1 | **The Responses API** (`ui/responses_shim.py`, 447 lines of standard library, and five tests in `c/tests/test_responses_*.py`) | Codex only speaks `POST /v1/responses`; our `coli launch` could not offer it (fork-changes, C1). It translates Responses requests to chat completions and the stream back, with text, reasoning, function calls and Codex's custom tools. | Small. The files are new to the fork. The work is hooking it into our router and gateway (upstream hooks it into its own `ui/server.py` proxy), then a real Codex run. |
| 2 | **Input validation and robustness fixes** in code the fork did not change: WAV parsing (`fix(audio)`), image requests (`fix(image)`: boolean sizes, PNG inflation, surrogate prompts, checkpoint tensor layouts), MCP server (`fix(mcp)`) | Small, independent, and each has its own test upstream. | Small. Pick by file; review the list first, since a few touch `openai_server.py`. |
| 3 | **Windows fixes** (52 commits, 23 touching only unchanged files; for example `posix_memalign` buffers freed the Windows way, GGUF overlapped reads on Windows) | The fork runs on Windows only. | Small to medium; read each. |
| 4 | **The Qwen engine's speed work** (`qwen36`: MTP and speculative decoding, prefill streaming, cache policy, several-conversation cancel, a long step's sliding window) | Qwen3-Coder is the everyday chat model. | Medium to large. `qwen36.c` is the second-most conflicted file (11 hunks) and the fork changed it too (demotion). **Measure first**: take one change, time Qwen3-Coder on the 5090 before and after. MTP needs a model that ships an MTP head, which Qwen3-Coder may not. |
| 5 | **GGUF files read as they are** (`gguf.h`, `ggml_quants.h`, `st_gguf.h`, 66 independent and 71 shared commits; upstream's README: "the language models also run from the GGUF files already published") | The biggest feature for users: the published GGUFs (Q4_K_M, UD quants) would run without conversion. | Large. New files, but they plug into the family registry, the setup catalog and `coli convert`, which both sides changed. Needs its own plan. |
| 6 | **CUDA** (328 commits) | Possibly faster than Vulkan on the RTX 5090. | Large, and blocked: this machine has no CUDA toolkit (no `nvcc`) and the Windows build needs `nvcc` and MSVC (`CUDA_DLL=1`). Install them and measure one model before deciding. |
| - | **Skip**: README and translations, CHANGELOG, `docs/*` that describe upstream's own setup, `ci.yml`, the upstream `web/` page (the fork's dashboard is its own; 14 files conflict), the `desktop/` shell and `ui/` page unless wanted for themselves | They conflict and carry nothing the fork needs. | None. |

## Progress

- **Batch 1, the Responses API: done and live 2026-11-25** (branch `upstream-picks`, merged into `main` as `37dfdb44`; fork-changes 2.41). Not yet run with a real Codex.

## Batch 2: validation and robustness fixes, the review list (2026-11-25)

How it was found: the 60 scoped upstream fixes (`fix(audio|image|images|mcp|json|schema|api|tools|serve|cli|usage)`) were cherry-picked one at a
time, in date order, into a throwaway worktree off `main`; after each, the test modules that commit touches were run; a commit that did not apply, or whose
tests failed, was dropped. (The two DeepSeek V4.1 tests `test_case1_*` and `test_case2_*` already fail on `main` and were not counted.) Result: **35 apply
and pass, 5 apply but break a test (they rest on an unpicked earlier commit), 20 conflict.** Applying every fix "that applies" blindly would have broken
the MiMo, MCP and history-JSON tests: a fix can apply cleanly and still depend on one that was not taken.

**Recommended to take (26, all apply and pass)**

- Gateway request validation (10): `dedda914` falsey non-array `tools`; `5e6e5635` out-of-range integers; `d57f0d54` non-integer `n`/`best_of`; `66355b09`
  JSON limits are client errors; `22999fca` non-finite strict tool arguments; `eb82d7f9` misplaced Anthropic tool blocks; `7c0146fa` NaN and Infinity in
  request bodies; `b3574515` non-string text part in a continued assistant turn; `dc08a7f7` a static path with NUL is a 404; `4156a055` an Optional string
  tool argument stays a string.
- JSON and the grammar behind JSON mode (7): `52dfdc4f`, `c3858b30`, `d54b7003`, `164ed3c1`, `e64185c8` (`schema_gbnf.h`), `ae0654d6`, `934b5f51` (`json.h`).
- Images (5): `872dde1e` automatic file names reserved before they are published; `4c74554c` VAE weights counted as resident float32;
  `9dd0c766` unpaired surrogate prompts; `16ed632c` boolean width or height; `49f53e0a` complete quoted image paths in `coli`.
- MCP server (2): `bef3fabd`, `4628ab7e`. CLI (2): `6630672b`, `e45d1eb4`.

**Apply and pass, but for engines or tools the fork does not run (9): optional**: `d148eb16` OLMoE, `58a28b51` Kimi, `cbbad148` and `87933353` DeepSeek V4.1,
`690bb5ef` Inkling; `f085485d`, `5ca2b801`, `1b72c4d2`, `f55a9c58` (usage and placement analysis tools).

**Conflict, worth a hand port (relevant to what the fork serves)**

- **WAV hardening (5)**: `8d667a77` truncated chunks and incomplete frames (applies but its test needs the rest), `6b4a4187` extensible WAV by subformat GUID,
  `5901bedb` non-finite samples, `59cb4b1d` chunks past the declared RIFF size, `6cd33e65` a trailing odd chunk. They all edit the gateway's WAV reader, which
  the fork's audio work (ASR, Gemma audio) also changed. Port as one piece, with upstream's `test_wav_audio.py`.
- **Images (2)**: `a097db37` bound PNG inflation, `cafc1ba8` validate checkpoint tensor layouts (`image_engine.py`, which the fork changed).
- **Qwen function calls (1)**: `bf84d729` recover valid Qwen function calls (`openai_server.py`): the fork's everyday chat model is Qwen3-Coder; look at what it
  recovers before deciding.
- **MCP**: `21bb2079` keeps the stdio session alive after JSON nesting errors, but its test fails here, so it rests on an earlier upstream change; look at it with the
  gateway's `d7c9d18f` (history JSON error policy) and `d114b9bb` (CORS on metrics), which fail the same way.

**Leave**: everything for Muse, GLM-5.3 and MiMo (`4b4f9520`, `6e2fc7bf`, `1261e334`, `30385c22`, `ebc1efac`, `4d848a9c`, `df227c89`), the logprobs channel work
(`0d9f2972`, `97a656b5`), `282b789c` (touches the web dashboard too), `84a208ab` (line endings of a file both sides rewrote), `d1dea849` and `9de40423`
(`coli`; small, conflict, not needed), `20951347` (the installer's state; the fork's installer is its own).

- **Batch 2, the 26 recommended validation fixes and the WAV and image hand ports: built 2026-11-25** (fork-changes 2.42), full suite compared with `main`: no regression. The Qwen function-call recovery (`bf84d729`) is held back, with the reasons in 2.42.
- **Batch 3, the Windows fixes: built 2026-11-25** (fork-changes 2.43): 22 of the 142 commits that mention Windows were taken and tested one by one (the hybrid-CPU team size, measured on Qwen3-Coder: +29% decode on the CPU, neutral on the card); the rest are held back or do not apply, with the reasons in 2.43. The two I/O-path commits were measured neutral (cold start of Qwen3-Coder unchanged).

- **Vulkan and the Qwen engine, measured (2026-11-25): a large gain, taken (fork-changes 2.44).** 182 upstream commits touch the Vulkan files and the fork changed those files very little (`backend_vulkan.c` +3/-1, `vk_chain.c` +43/-3, `vk_tier.c` +11/-1; `qwen36.c` +181/-16 against upstream's +6,095). Upstream's own build of `qwen36` (Vulkan, `e479104a`) against the fork's, Qwen3-Coder-30B on the RTX 5090, `OMP_NUM_THREADS=8`, builds alternated twice, real source code as the prompt, temperature 0, the same first words of the answer: a 367-token prompt, decode **182-191 tok/s against 20** and 0.05-0.18 s to the first token against 1.1 s; a 5,382-token prompt, 1.2 s against 18 s to the first token and decode **172 against 10.5 tok/s**; a 7,055-token prompt of filler text, 1.55 s against 21.8 s and 170 against 9.2 tok/s. Upstream with `COLI_LOOKUP=0 Q36_MTP=0` (no speculative drafts) still decodes at 187 tok/s, so it is the chain and not drafting. Why, from the chain profile (7k prompt): the fork's prompt reading spends 13.9 of 16.4 s of device time in attention, because `chain_attn_flash` (the matrix units) loads only for subgroups of 64 and this card's are 32 (`vk_chain.c` line 405); upstream added the 32 case and moved the routed experts into the device frame (`chain_moe`), so a token is one frame and not a frame plus a host step. A trial merge of all of upstream into `main`: 77 conflicted files, the Vulkan core only 4 hunks (the weight is `openai_server.py` 27, `qwen36.c` 11, `setup_catalog.py` 9, `coli` 9; many are our own cherry-picks of the same changes). (An earlier version of this note said no gain: the benchmark script ignored its build override, so both sides ran the fork's engine. Corrected the same day.)

## How to take things

1. A branch `upstream-picks` off `main`, one batch at a time, each ending in the test suite and a merge to `main` only when it is green.
2. **Take files, not a chain of commits**: for a file the fork never changed, take upstream's current version whole (all its commits at once, so no
   half-applied history); for the 81 shared files, hand-merge the hunks that belong to the batch. Cherry-picking commit by commit fails where a commit
   rests on an earlier one that was not taken.
3. Every batch gets: the upstream tests it brings, a build on this machine, the fork's own suites for what it touches, and a measurement when it claims speed.
4. Record what was taken and from which upstream commit (`e479104a` is the tip at the time of this note) in fork-changes.md, so the next catch-up starts from
   a known point.

## Decisions for the owner

- Which batch first. The recommendation is 1, then 2 and 3 together (small), then measure one Qwen change (4) before committing to more.
- Whether GGUF (5) is wanted at all: it is the largest piece and changes what the fork can run.
- Whether to install the CUDA toolkit to find out about 6.
