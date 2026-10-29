# colibri — repository deep dive

> Snapshot: local `main` at `e0db083c` (2026-10-17), version **2.0.0** (`c/version.py`).
> Method: static reading of the source, Makefiles, CI and docs. **Nothing was built or run**, and no benchmark
> number below was reproduced; figures are quoted from the repo's own docs. Where a claim rests on a header
> comment or doc rather than traced code, or where docs and code disagree, it is flagged.

---

## 1. What this project is

**colibri** ("tiny engine, immense model") is a pure-C inference engine for very large open mixture-of-experts
(MoE) language models on ordinary hardware. A 744B-parameter model uses ~40B parameters per token, and only
~11 GB of that changes token to token (the routed experts). colibri therefore:

- keeps the **dense part** (attention, shared experts, embeddings) in RAM,
- keeps the **routed experts on disk** and reads them when the router asks, through a cache that learns which
  experts your work uses,
- optionally puts the hottest experts and dense layers on a **GPU** (CUDA, Vulkan, Metal, or an AMD NPU).

Design rule stated in the README: **placement only decides speed.** Router decisions and weight precision must
be identical whether an expert was served from VRAM, RAM or disk. Less fast memory may cost speed; it must not
quietly change the model.

Other stated properties: no BLAS and no Python at runtime (Python runs only setup, the launcher, converters and
the HTTP gateway); Apache-2.0 licence (`NOTICE`: Vincenzo Fornaro, 2026); the default CPU path is dependency-free.

### Supported engines (13)

| Kind | Models |
|---|---|
| Chat / text (10) | GLM-5.2 / 5.3, GLM-5.3-Flash, Inkling, Kimi K3, DeepSeek V4 Flash, DeepSeek V4.1 Flash, MiMo-V2.6 Flash (and Pro), Qwen3.8-Flash-Next, Qwen3.6 (also runs Qwen3-Coder and dense Qwen3.8-27B), OLMoE |
| Image (1) | Qwen-Image-2.1 |
| Decision (2) | Laya, GLiNER2.5-Decide (a third, Clef, runs inside the Qwen3.6 engine) |

Scale range: OLMoE 7B up to Kimi K3 2.8T (1.56 TB download). Measured speeds range from 0.05–0.1 tok/s (GLM-5.2
cold on a 25 GB laptop) to 9.0–9.2 tok/s (GLM-5.2 on 6x RTX 5090 with every expert resident); small models such
as Qwen3.6-35B reach 6–30 tok/s depending on CPU/GPU (README tables).

---

## 2. Repository at a glance

1,191 tracked files, ~187k lines in `c/*.c|h|py|cu|mm` alone.

| Path | Files | Role |
|---|---|---|
| `c/` | 966 | Everything that matters: engines, shared headers, GPU backends, Python launcher/gateway/setup, tests, tools |
| `docs/` | 101 | Reference docs (Mintlify site), experiments, baselines, media |
| `web/` | 60 | React dashboard (pure API client) |
| `desktop/` | 15 | Minimal Tauri shell around the dashboard |
| `.github/` | 9 | 4 workflows + issue/PR templates |
| `site/` | 5 | Static landing page |
| `docker/` | 5 | Two Dockerfiles, compose, READMEs |
| `colibri/` | 3 | pip package shim (`coli` console script) |
| root | — | `README*.md` (6 languages), `CHANGELOG.md`, `GPU_BACKENDS.md`, `flake.nix`, `START-HERE.bat`/`start-here.sh`, thin `Makefile` forwarding to `c/` |

Inside `c/`: `tests/` (520 files), `tools/` (~158), `shaders/` (73), and the tiny fixtures
(`*_tiny/` directories).

Largest sources (lines): `deepseek_v4.c` 21,152 · `qwen38_nfc_tables.h` 18,585 (generated) · `colibri.c` 14,149 ·
`openai_server.py` 8,325 · `qwen36.c` 7,380 · `backend_vulkan.c` 6,590 · `deepseek_v41.c` 5,339 · `glm53.c` 5,244 ·
`kimi_k3.c` 5,044 · `inkling.c` 4,480 · `qwen38_core.h` 4,153 · `olmoe.c` 3,529 · `resource_plan.py` 3,419 ·
`c/coli` 3,700.

---

## 3. Architecture overview

```
                       ┌─────────────────────────────────────────────┐
 browser / desktop ──► │ web/ dashboard (React) — pure API client    │
 OpenAI/Anthropic SDKs │                                             │
 coding CLIs, MCP      └───────────────┬─────────────────────────────┘
                                       │ HTTP (SSE)
                       ┌───────────────▼─────────────────────────────┐
                       │ c/openai_server.py  (gateway, Python)       │
                       │  chat templates, tool-call parsing,         │
                       │  scheduler/queue, /v1/* endpoints           │
                       └───────────────┬─────────────────────────────┘
                                       │ stdin/stdout "mux" protocol
                       ┌───────────────▼─────────────────────────────┐
 c/coli (launcher) ──► │ one C engine process per model family       │
 setup_*.py, MCP       │  colibri.c, qwen36.c, deepseek_v4.c, …      │
                       │  over shared single-file headers            │
                       └───┬───────────────┬──────────────┬──────────┘
                           │               │              │
                    expert cache      GPU backends     disk (safetensors,
                    (pin + LRU)    CUDA/Vulkan/Metal    optional mirrors)
```

Key structural decisions:

1. **One `.c` file per model family over shared single headers.** An engine owns its architecture; anything two
   engines need goes in a header both include (README "Repo layout"). In practice a lot is still duplicated per
   engine (see §5).
2. **Chat templates and tool-call parsing live in the Python gateway**, not the engine: the engine receives a
   fully rendered prompt (`render_chat_for_arch`, `c/openai_server.py:3460`).
3. **The engine is a child process** speaking a byte-counted line protocol (§8), so the gateway can multiplex
   many conversations and the engine can be swapped per family.
4. **Backends are compile-time opt-ins** (`CUDA=1`, `VK=1`, `METAL=1`, `XDNA=1`), so the default build stays pure C.

---

## 4. The core engine (`c/colibri.c`, GLM-5.2/5.3)

`colibri.c` is the reference engine (renamed from `glm.c`; `docs/serve_protocol.md` still says `glm.c`).

### 4.1 Startup and token flow

`main` (`c/colibri.c:12492`): OpenMP tuning (one-time Linux `execv` re-exec, thread count sized to physical
cores via `c/omp_tune.h`) → env parsing → `model_init` (`:13107`; loads `Cfg` via `load_cfg` `:1788`, resident
dense weights via `qt_from_disk` `:2227`) → MTP/speculation depth → mirror setup → `.coli_usage` load and
autopin → `cap_for_ram` (sizes the per-layer LRU) → dispatch to scoring, serve (mux or legacy), `run_text`, or
the oracle comparison.

Core structs: `Cfg` (:182), `QT` quantized tensor (:265), `Layer` (:407), `ESlot` cached expert (:435), `KVState`
(:476), `DecodeRow` (:488), `Model` (:493).

Per token (prefill and decode share a path): `step`/`step_all`/`step_decode_batch` (`:7744/:7783/:7799`) →
`layers_forward_rows_range` (`:7529`, may delegate to the CUDA pipe or the Vulkan chain) → per layer
`attention_rows` (`:4508`, MLA with absorption for S≤4; DSA sparse selection once `pos_base+S > index_topk`) →
`moe` (`:5739`) → final rmsnorm and `lm_head`.

`moe` phases: (A) batched router matmul, sigmoid + `router_bias`, top-K; (B) union of the experts the batch
touches; (C/D) in blocks of 64 unique experts, resolve each as pin hit / LRU hit / miss, load misses (OMP,
`PIPE`, or `URING`), compute, then promote loaded slots into the LRU; (E) shared expert.

### 4.2 Expert streaming, caching and tiers

- **On-disk layout:** per expert `model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight` plus `.qs` f32 scale
  sidecars. When gate/up/down are contiguous, `expert_load_impl` (`:3039`) issues **one coalesced read** into a
  4K-aligned slab.
- **Read paths:** buffered `pread`; `DIRECT=1` (`O_DIRECT`); `COLI_MMAP=1` (page cache is the cache; not with
  `URING`); async `PIPE` loader pool (default 8 workers, max 16); `URING=1` (Linux, raw-syscall io_uring wrapper
  `uring.h`).
- **Tiers:** *pin* (never evicted) → per-layer *LRU* (`eslot_lru_victim` `:458`) → optional *VRAM tier*
  (CUDA / Vulkan `vk_tier` / Metal) → disk.
- **Prefetch:** `PILOT=1` applies layer L+1's router to layer L's post-attention state on a dedicated I/O thread
  (README: GLM-5.2 routing is 71.6% predictable one layer ahead); `PILOT_REAL`, `COUPLE`, next-block `WILLNEED`.
- **Multi-disk:** `COLI_MODEL_MIRROR` points at byte-identical read-only copies; `expert_route` (`:2840`)
  hashes (layer, expert) to a replica, and under `O_DIRECT` `mir_pread_striped` splits one read across replicas
  by measured bandwidth. `COLI_MODEL_DIRS` instead spreads distinct shards. `coli mirror plan|stage|verify`
  builds usage-ranked partial mirrors. README: two independent NVMe drives measured +37.5% decode.
- **NUMA:** `COLI_NUMA=1` `mbind`s slabs across memory controllers.

### 4.3 What "the cache learns" means

Every routed (layer, expert) selection bumps `eusage` (persistent) and `eheat` (session). `eusage` is saved to
`<model>/.coli_usage` each turn (`telemetry.h`, format in `route_trace.h`). At startup, once history holds ≥5,000
selections, **autopin** pins hot experts (`:13254`; budget `0.5 × expert-available RAM × min(1, hist/200000)`;
logged as `[PIN] auto:`). `cap_for_ram` sizes the LRU. Optional `REPIN=n` swaps the coldest pinned expert for a
hotter one at turn boundaries using LFRU + hysteresis (`c/tier.h`: promote when heat > 1.25 × cold + 4, ≤4 swaps
per pass). So: pin set follows long-term history, LRU follows the session, REPIN follows the live workload.
Two seed histories ship in `c/profiles/` (`qwen36-35b`, `qwen38-flash-next`).

Opt-in, **lossy** routing levers (not default): `CACHE_ROUTE=1` (prefer resident experts within a top-M window;
`docs/CACHE_ROUTE.md`), `EXPERT_BUDGET`, `TOPP`, `DEGRADE_ZERO`.

### 4.4 Storage and quantization formats

Plain safetensors (`c/st.h`; `pread` + `fadvise(DONTNEED)` rather than mmap, hash index over ~120k tensors). A
quantized weight is a U8 tensor with an f32 `name.qs` scale tensor. There is **no format ordinal on disk**:
`qt_resolve_fmt` (`:1964`) infers the format from byte counts against `[O,I]`, with an optional verified
`colibri.fmt` metadata stamp (`docs/FORMATS.md`).

| fmt | Name | Notes |
|---|---|---|
| 0 | f32 | norms, router |
| 1 / 2 / 3 | int8 / int4 / int2, per-row f32 scale | |
| 4 | int4 grouped (g64) | the recommended GLM container ("older per-row int4 mirrors measure ~9 points worse") |
| 5 | int3-g64 | |
| 6 | E8/IQ3 lattice | 98 B per 256 weights; activations FWHT-rotated first |
| 7 | MXFP4 | Vulkan / Kimi-K3 only, never a CPU `QT` |
| 8 | fp8-e4m3, 128x128 block scales | |

ISA is chosen **at compile time** (`-march`; no runtime CPU dispatch). Integer-dot kernels (`idot.h`: AVX2,
AVX-VNNI/512-VNNI, NEON sdot/smmla, VSX, AMX) quantize activations to int8 per row; integer sums are exact so
ISAs are bit-identical. The GLM MTP head must stay int8 (int4 heads collapse to 0–4% acceptance, `docs/tuning.md`).

**Entropy-coded containers are offline-only:** `rans.h` (int4-rans256-g0, ~0.76 ratio on GLM-5.2), `fse_coli.h`
and `qpack.*` are *not* on `colibri.c`'s decode path (qpack is used by `qwen36_qpack.c`; rANS is tools/tests
only). Don't point the GLM engine at a rANS-repacked directory.

### 4.5 KV cache, speculation, grammar

- **KV:** stores only the normed MLA latent + rope part per token per layer (README: 576 floats/token vs 32,768,
  57x smaller). `kv_persist.h` writes `.coli_kv` incrementally each turn (row count written last and `fsync`ed;
  magics `COLIKV1/2/3` = f32 / KV8 / TQ), so conversations reopen warm. `KV8` = e4m3 rows (~3.9x smaller, CPU
  attention only); `KV_TQ` = randomized-Hadamard + 4-bit Lloyd-Max codec ("TurboQuant-ish", phase 1 only,
  mutually exclusive with KV8). Prefix reuse is truncate-and-extend in `mux_submit` (`:10192`).
- **Speculation** (`spec_decode`, `:8130`): draft priority grammar → corpus → MTP head → prompt lookup; verify in
  one `step_all` over 1+g rows. Auto MTP depth is 1 ("fastest in every measured configuration"). A soft guard
  pauses MTP for 256 tokens when acceptance over a 24-proposal window falls below 70%. Multi-slot mux forces
  drafting off (not ragged-safe).
- **Grammar:** `grammar.h` is a byte-level GBNF-subset PDA; `schema_gbnf.h` compiles JSON Schema (unsupported →
  fails closed). Grammar **never constrains sampling**; it only proposes forced draft spans that verification
  accepts or rejects, so output is unchanged and only speed varies (greedy requests only).

---

## 5. The other model-family engines

All follow the same shape (dense resident, experts streamed, LRU, serve loop); each has its own `Model` struct,
serve loop and linear-attention kernels.

| Family | Source (lines) | Traits |
|---|---|---|
| GLM-5.3-Flash 321B | `glm53.c` (5,244) | 34 KDA + 11 NoPE-MLA layers, DSA indexer, mHC residual; 288 routed (top-8); int4 experts; 24-block ViT. Only engine using `delta_attention.h`, `sparse_index.h`, `vision_tower.h`. Built from the Kimi K3 skeleton |
| DeepSeek V4 Flash 284B | `deepseek_v4.c` (21,152) + 6 headers | MLA + DSA + sliding window + compressed KV + mHC; 256 routed (top-6); **native fp4 experts / fp8 dense, no conversion**; opt-in "DSpark" MTP drafter; CUDA tier, optional DeepGEMM sm_120a |
| DeepSeek V4.1 Flash 552B | `deepseek_v41.c` (5,339) | MQA + window ring + two-level DSA; 384 routed; **Engram**: two n-gram tables (203 GB, read by hash, ~13 KB/token); vision; needs a `prepare_dsv41.py` sidecar. A separate engine, not a V4 unit |
| Qwen3.6 family | `qwen36.c` (7,380) + tier/qpack/chain | Gated attention every 4th layer, Gated DeltaNet otherwise; 256 routed (top-8). One engine, many checkpoints via `display_variants` (Qwen3.6-35B, Qwen3.8-2.4T, Qwen3.8-27B, Qwen3-Coder-30B, REAP 25B). Also hosts Clef |
| Qwen3.8-Flash-Next 176B | `qwen38.c` + `qwen38_core.h` (4,153) | 12 × (3 DeltaNet + 1 sparse attention); 4 gated residual streams; 512 routed (top-10); native block-FP8; **PLE**: 51B hashed n-gram table on disk (16 rows × 160 B/token); MTP **on by default** |
| Kimi K3 2.8T | `kimi_k3.c` (5,044) | 69 KDA + 24 gated NoPE-MLA; AttnRes residual; LatentMoE 896 experts top-16 (7168→3584), SiTU-GLU; native MXFP4 streamed from HF shards |
| Inkling 975B | `inkling.c` (4,480) | Sliding-window/global GQA 5:1, no RoPE, learned relative bias, depthwise short convs; DMel audio input; own tiny CUDA backend (`backend_cuda_ink.cu`) |
| MiMo-V2.6 Flash/Pro | `mimo.c` (2,668) | Hybrid sliding/full attention, sinks, MXFP4 experts, FP8 dense, no conversion; ViT images only |
| OLMoE 7B | `olmoe.c` (3,529) | Conventional GQA; the original reference engine (its `PILOT/HOT/WIDE` prefetch vars were inherited) |
| Qwen-Image-2.1 | `qwenimage.c` (1,546) | Text-to-image; see §5.2 |
| Laya, GLiNER2.5-Decide | `laya.c` (1,390), `gliner_decide.c` (1,644) | Encoder decision models; see §5.1 |

### 5.1 System One and decision models

`POST /v1/systemone` (Jev-API-compatible) takes a `state` plus typed questions (`noul` yes/no, `choice` with
1–255 labels, `score`) and returns a probability per option and `confidence = (n·peak−1)/(n−1)`. **Nothing is
generated**, so an answer can't fall outside your list.

- **On any chat model:** the server snapshots the state's KV (`pin=1`), sends each option as a prompt with
  `logprobs=k max_tokens=0`, sums option log-probs (default `normalize=sum`; mean favours longer options) and
  softmaxes. README claims 5.7x faster than generating on Qwen3.6 with four questions per document.
- **Native decision engines** answer in one forward pass over a record (`DECIDE` frame, engine announces
  `CAPS decide=1 chat=0`; `decide_serve.h` shares the parsing/serve loop): **Laya** (ModernBERT-large, one
  `[MASK]` marker per option, per-type temperature clamped [0.5, 5], act/escalate head), **GLiNER2.5-Decide**
  (DeBERTa-v3-large, `[P]/[L]/[SEP_*]` markers, extraction heads never loaded), **Clef** (Qwen3.8-27B + a joint
  schema head in `clef_head.h`, handled in `qwen36.c`; detected via `joint_head_config.json`).

### 5.2 Image engine pipeline

No conversion: reads the diffusers layout and quantizes to int8 per row at load. (1) Qwen3-VL text encode,
36 layers, can be unloaded before denoising (9 GB peak). (2) Prefix pass: prompt K/V computed once.
(3) Seeded latents with dynamic-shift sigmas. (4) 32 single-stream DiT blocks per step with a plain Euler update
(no CFG/negative prompt found) and per-step progress/preview. (5) Wan-style causal VAE reduced to 2D
(im2col+GEMM; Vulkan path available). (6) PNG output. `--serve` speaks `READY/GEN/PROGRESS/PREVIEW/IMAGE/ERROR`
(+`CANCEL`), client side in `c/image_engine.py`, exposed as `POST /v1/images/generations`.

### 5.3 Why `deepseek_v4.c` is 21k lines

It is an **amalgamation of ~29 formerly separate units**, each wrapped in `#ifdef COLI_V4_UNIT_<X>`. The build
compiles the same file once per unit (`-DCOLI_V4_UNIT_X`) and links the objects (`c/Makefile.deepseek-v4`,
`.units` file "generated by `_amalgamate_v4.py`" — that script is **not in the repo**). Objects are named after
the unit, not the flags, so the Makefile records flags in a stamp file (stale-object bug #1702). `route_trace.h`
statics need a single owning unit. Rough map: ST adapter (15–588) · resident/plan/head cache (592–1683) ·
attention (1939–3395) · compressor+indexer (3398–4451) · block_hybrid (4555–6801) · KV rollback snapshots for
speculation (6804–8147) · hot-rows16 expert store (8150–10147) · GPU tier (10851–12637) ·
GENERATE_STATS + `v4_serve_main` + real `main` (12705–17046; a legacy first-token `main` is renamed away at
`:12719`) · test-only units · native quant (18442–20170) · segment/edge adapters (20173+). It was 8,717 lines at
its first commit; *why* a single file was chosen is not documented.

### 5.4 Shared vs duplicated

Shared headers: `st.h`, `json.h`, `tok.h`, `quant.h`, `kv_prefix.h`, `pin_pool.h`, `route_trace.h`,
`serve_codec.h`, `vk_tier.h`/`vk_chain.h`, `hyper_connections.h` (mHC: V4, V4.1, GLM-5.3), `hybrid_split.h`,
`spec_draft.h`, `expert_ffn.h`, `qwen36_tier.h`, `qwen38_vision.h`, `qwen38_nfc.h` (NFC normaliser, also used by
image/decision engines). Duplicated by design: each engine's `Model`, serve loop, RMSNorm and
linear-attention kernel (KDA ×2, DeltaNet ×2).

Doc/code mismatches worth knowing: `delta_attention.h`'s comment says Kimi/Qwen3.6/GLM-5.3 use it (only
`glm53.c` includes it); `dsv4_mhc.h`/`dsv4_quant.h` are test-only; `c/qwen38.c:9` says MTP is opt-in but the code
defaults it on.

### 5.5 Adding a family

Add a `FamilyDescriptor` to `FAMILIES` in `c/family_registry.py:1241` (id, `model_types`, `engine_artifact`,
`build_target`, planner geometry, expert inventory, limits, capabilities; optional modality/variants/converter).
`_build_registry` (`:1794`) enforces uniqueness; `resolve_model` (`:1983`) dispatches on `config.json`
(`model_index.json` → image, `rl_agent_config.json` → Laya, `model_type: extractor` → GLiNER). You also need a
Makefile target, the engine `.c`, a tiny oracle fixture, tests, and a row in the segment-conformance manifest.
13 families are registered.

### 5.6 Edge and segment runtimes

Two engine-neutral **local C ABIs** (not network protocols) for an external distributed-inference consumer
named "Lumabri" in the docs, which owns transport/placement; colibri owns model math.
`edge_runtime` (ABI v2): tokenize/detokenize/`embed`/`select`/`logits`. `segment_runtime` (ABI v1): run a
half-open layer range `[begin,end)` with session create/run/`snapshot`/`restore`. The `.c` files are only
registries + validation (16 slots); logic lives in `#ifdef COLI_SEGMENT_ADAPTER`/`COLI_EDGE_ADAPTER` blocks in
engines. 8 engines have adapters (glm, glm53, inkling, kimi, olmoe, qwen36, qwen38, deepseek_v4; docs say seven —
stale). Built via `make -C c segment-edge-library`; all adapters currently advertise CPU only.

---

## 6. GPU and accelerator backends

There is **no vtable/plugin interface**: each backend is its own C API, and engines call it behind
`#ifdef COLI_<BACKEND>` hooks. `QT` carries one optional opaque handle per backend.

| Backend | Used by | Loading | Notes |
|---|---|---|---|
| **CUDA / HIP** (`backend_cuda.cu`, 3,283 lines) | colibri, qwen36/38, inkling, kimi_k3 | Linux: linked. Windows: separate `coli_cuda.dll` resolved by `backend_loader.c` (absolute path beside exe, `LOAD_WITH_ALTERED_SEARCH_PATH`, never CWD; missing DLL → CPU) | One source for both vendors via `backend_gpu_compat.h`. Custom kernels, tensor-core W4A16 (NVIDIA only). Optional `COLI_ANS` compressed VRAM tier |
| **DeepSeek V4 CUDA** (`backend_cuda_dsv4.cu`) | deepseek_v4 | Own ABI/DLLs: generic (sm_80+) and DeepGEMM `_dg` (sm_120a, RTX 50 only) | DeepGEMM headers not vendored: `fetch_deepgemm.sh` pins a community sm120 port and applies `patches-deepgemm-sm120-msvc.patch` (guards inline asm, pads TMA descriptors for MSVC). CI tests fetch+patch only, not the compile |
| **Vulkan** (`backend_vulkan.c` 6,590, `vk_tier.c`, `vk_chain.c`) | **all** engines | `dlopen("libvulkan.so.1")` / `vulkan-1.dll` at runtime; ~60-function table in `vk_load.h`. A `VK=1` binary runs on CPU where Vulkan is absent (CI asserts no `libvulkan` in `ldd`) | Any vendor, Vulkan 1.2; optional cooperative-matrix, memory-budget/priority, external-memory-host. Ships prebuilt SPIR-V |
| **Metal** (`backend_metal.mm`, 2,367) | colibri, glm53, inkling, kimi_k3, qwen36 | Linked; shader source embedded and compiled at run time (no Xcode needed) | No tier: unified memory, zero-copy reads of RAM slabs |
| **XDNA** (AMD Ryzen AI NPU, Windows) | GLM only | Optional `coli_xdna.dll` helper (links XRT); host never links XRT | Experimental, opt-in (`--xdna`); accelerates only GLM's shared-expert gate/up (K=6144,N=2048, g64 int4); BF16 compute so output can diverge (5 of 8 prompts differed in one experiment); SHA256-registered `.xclbin`s |

### 6.1 The two big GPU ideas

- **Expert tier.** Hot experts cached in VRAM; the CPU computes the misses *concurrently* and outputs are added
  in CPU rank order. Shared admission policy in `tier.h`. CUDA/GLM: placed by `CUDA_EXPERT_GB` + `PIN` heat;
  Qwen3.6/3.8: `qwen36_tier.c` homes expert `eid` on GPU `eid % n_gpus` and *prices the dense trunk against the
  experts it would displace* (auto-placement withdrawn if the GPU loses a probe — it lost on four Tesla M10s).
  Vulkan: one tier for every MoE engine, warm-started from `.coli_usage`, resident experts run as one async
  submit on a second queue.
- **Dense chain (Vulkan only).** A whole layer is recorded as one command buffer, residual stream stays on-device
  across layers; only router logits, CPU-expert input rows, the routed sum and new K/V rows cross. Each engine
  has a `*_chain.h` adapter. Default decision (`coli_vk_chain_decide`, `backend_vulkan.c:2365`): explicit
  `COLI_VK_CHAIN` wins; on an integrated GPU use the per-engine *measured* default (Qwen3.6 yes, Qwen3.8 no);
  off on software devices; on for discrete. `vk_kvsplit.h` handles KV larger than VRAM (window + CPU attends the
  rest, merged via softmax stats). CUDA's analogue is `COLI_CUDA_PIPE`.

Other notable: **no-ReBAR staging** (`place_decide`, `backend_vulkan.c:554`; uploads through staging buffers when
the host-visible device-local heap < ¼ of the largest — *tested by forcing/emulation, never measured on such a
card*); **`Q36_DN_GPU=1`** puts Qwen3.6's DeltaNet layers (conv ring + 32×128×128 f32 state) on the card (README:
25.4→30.0 tok/s on an RTX 3070; opt-in; not byte-identical to CPU).

### 6.2 Selection env vars

CUDA/HIP: `COLI_CUDA=1`, `COLI_GPU`/`COLI_GPUS` (not both), `CUDA_EXPERT_GB`, `COLI_CUDA_PIPE`. Vulkan:
`COLI_VULKAN=1`, `COLI_VK_SHADERS`, `COLI_VK_DEV`/`DEV2`, `COLI_VK_TIER(_GB)`, `COLI_VK_DENSE`, `COLI_VK_CHAIN`,
`COLI_VK_STAGED`. Metal: `COLI_METAL=1` (`K3_METAL`). XDNA: `COLI_XDNA` (non-zero integer). Failure modes differ:
colibri/GLM **exits** on a missing device; other engines log `[VK] … no usable Vulkan device` and continue on CPU.

### 6.3 Shaders

73 files = 38 GLSL `.comp` (`#version 450`, `glslc --target-env=vulkan1.2`) + 35 `.spv`; **all 35 `.spv` are
tracked** as prebuilt fallbacks, loaded at runtime (not embedded in C). Release archives ship `shaders/*.spv`
beside the binaries. Categories: per-matrix primitives (`qmatmul*`, `expert_act*`, `attention_absorb`,
`rmsnorm`) and 24 `chain_*` shaders (gemv/gemm, norm/rope, attention variants, DeltaNet/KDA/mHC, plus
engine-specific ones for V4, Kimi AttnRes, Qwen3.8 PLE, Inkling convs, encoders, VAE). Six use cooperative matrix.

### 6.4 Correctness vs CPU

- **Vulkan:** CI runs every engine on **Lavapipe** (software driver), CPU vs Vulkan on tiny fixtures, expecting
  the same tokens; also ASan/UBSan, device-lost, staged-upload fault injection, no-loader tests. Where both
  sides use f32 activations logits agree ~1e-4 (≈2e-7 on fixtures); a long answer can still drift by a word.
- **CUDA/HIP:** **CI only compiles** (nvcc syntax check, Windows MSVC DLL build, hipcc syntax check). Kernel
  execution is tested only by `make cuda-test`/`hip-test` on real hardware. GPU greedy output is not
  token-identical to CPU; MTP acceptance dropped ~40%→~31% on one GPU-heavy config.
- **Metal:** `make metal-test` runs on `macos-latest`; greedy decode token-exact vs CPU, prefill can flip near ties.
- **XDNA:** hardware-free unit tests in CI; physical device tests are local-only.

---

## 7. Python layer

### 7.1 The `coli` launcher (`c/coli`, 3,700 lines)

Single-file Python script (`coli.cmd` is the Windows shim; `colibri/cli.py` is the pip entry that `runpy`s it).
Subcommands: `setup`/`start`/`status`/`logs`/`stop`, `chat`, `serve`, `web`, `run`, `plan`, `doctor [--deep]`,
`tune`, `info`, `build`, `convert`, `bench` (GLM only), `mirror`, `cluster coordinator|worker`, `mcp`, `--version`.

- **Engine resolution:** `COLI_ENGINE` → `c/colibri` → `c/glm` → `$PREFIX/libexec/colibri/colibri`;
  `engine_for(model)` (`c/coli:433`) reads `config.json` via `family_registry.resolve_model`.
- **`chat`** first probes `127.0.0.1:8000` for a running server and attaches; otherwise GLM runs its engine
  directly over the legacy protocol while other families spawn `openai_server.py` as a child.
- **`stop`** SIGTERMs launchers first (30 s drain so the usage heat file is saved), then stray engines, then
  force-kills; Windows liveness uses `OpenProcess`.
- **`web`** = `serve` + open browser once `/health` answers, with *no iteration cap* (Kimi K3 loads 40+ min).
- **`tune`** (`autotune.py`) sweeps **only scheduling knobs** (never TOPK/TOPP/DRAFT/quantization), requires a
  min gain (default 3%), persists profiles per machine fingerprint.
- **`plan`/`doctor`:** `resource_plan.py` (3.4k lines) reads safetensors headers, memory (cgroup-aware), GPUs,
  SSD speed, and outputs a Disk/RAM/VRAM plan with per-layer cache slots and a projected hit rate.

### 7.2 The gateway (`c/openai_server.py`, 8,325 lines)

`ThreadingHTTPServer` (64 connections total, 8 per IP). One engine child per server; per-request queues keyed by
request id; Windows engines get a KILL_ON_JOB_CLOSE job object. A `GenerationScheduler` does bounded FIFO
admission (capacity = `kv_slots`; 429 `queue_full`/`queue_timeout`; Prometheus histograms). Decode is
continuously batched *in the engine*. `conversation_cache_slot` hashes system + first user message to a stable
KV slot so stateless HTTP turns reuse prefix caches.

Endpoints: `GET /health, /metrics, /experts, /profile, /v1/models[/{id}]` + static `web/dist` with SPA
fallback; `POST /v1/chat/completions, /v1/completions, /v1/messages` (Anthropic, translated to OpenAI shape),
`/v1/systemone`, `/v1/images/generations`. No `/v1/embeddings` or `count_tokens` found.

Security: non-loopback bind without `--api-key` refuses to start unless `COLI_ALLOW_INSECURE_BIND=1`;
Host-header DNS-rebinding guard (`--allowed-host`); CORS list; 4 MiB body cap; `/health`, `/experts`, `/profile`
withhold hardware/scheduler detail from unauthenticated callers (SEC-8).

Per-family chat templates and tool parsers (GLM, DSML for V4/V4.1 in `v4_dsml.py`/`v41_dsml.py`, K3 XTML, Qwen,
MiMo) live here.

### 7.3 One-step setup (`coli setup`, `c/setup_flow.py`)

`START-HERE.bat`/`start-here.sh` only locate Python ≥3.10 (the `.bat` can `winget`-install it) and run
`coli setup`. Flow: **detect hardware** (`setup_hw.py`: RAM, CPU, disk, NVIDIA via `nvidia-smi` + driver probe,
Vulkan via ctypes in a child process) → **recommend** from the 12-entry `CATALOG` (`setup_catalog.py`; static
`ram_min/ram_good/disk` thresholds; status good/tight/needs-ram/needs-disk/unsupported) → **choose backend**
(CUDA → Vulkan → CPU; integrated GPUs stay on CPU unless measured faster) → **build** with `make` if a compiler
exists, else **fetch the release archive** (SHA256-verified) → **resumable stdlib-only HF download**
(`.part` files, `Range`, `.colibri-download.json` manifest; can hand off to Windows `curl.exe` on WSL) →
write `setup.json` and **start the server**. Failed builds fall back to the next backend. Note `resource_plan` is
advisory *after* download, not the recommender.

### 7.4 MCP server and AI-assisted setup

`c/mcp_server.py`: JSON-RPC 2.0 over stdio (protocol versions 2024-11-05 … 2025-06-18); tools `detect_hardware`,
`recommend_models`, `install` (detached job with progress notifications), `start`, `stop`, `status`, `logs`.
`docs/AI_SETUP.md` lets an assistant drive setup with machine-readable output and instructs it to ask before
downloading a model or installing system packages.

### 7.5 Cluster mode

GLM engine only. The coordinator keeps generation, routing and KV local; remote workers run routed expert FFNs
(one persistent TCP request per layer's batch-union; wire magic `COLIEX01`, `colibri.c:3335`). `cluster.py` is
an HTTP control plane (default port 8765; register/heartbeat/topology; nodes dropped after 30 s). Workers run
the GLM engine with `EXPERT_WORKER=1`. CI has a `cluster-parity` job.

---

## 8. Serve protocol (engine ⇄ gateway)

Two line-oriented, byte-counted protocols over stdin/stdout (`docs/serve_protocol.md`):

- **Mux** (`SERVE=1 SERVE_BATCH=1`, used by the gateway): engine sends `READY` (+ `CAPS`, `HWINFO`, `TIERS`,
  `EMAP`) or `LOAD_FAIL kind=nomem|io|format|unsupported`. Gateway sends
  `SUBMIT <id> <slot> <bytes> <max_tokens> <temp> <top_p> [gbytes] [logprobs=k ids=1 pin=1]`, `STOP`, `CANCEL`,
  (`IMAGE`, `DECIDE` for other engines). Engine replies `ACCEPT`, `DATA <id> <n>`+bytes, `TOPK`, `HITS`/`REPIN`,
  telemetry, `DONE <id> STAT …`, or `ERROR <id> CODE`. Unknown line kinds must be ignored (forward-compat).
  Prefill is serial; decode contributes one row per active slot per step.
- **Legacy** (`SERVE=1`, used by `coli chat` on GLM): `\x02RESET/MORE/PROMPT`, answers end `\x01\x01END\x01\x01`.

Docs lag the code: the doc omits `ACCEPT`, `CONTEXT_EXCEEDED`, the `SUBMIT` extensions, and says 1–16 slots
while code allows up to 512 under `SERVE_BATCH=1`.

---

## 9. Web dashboard, desktop, containers

### Web (`web/`)
React 18, Vite 8, TypeScript, Tailwind 4, vitest — about 3,700 lines of TS/TSX; no router, state library,
markdown library or chart library (markdown and SVG charts are hand-written). Four shadcn primitives only.
Views: **chat** (streaming, tok/s, TTFT, reasoning blocks, regenerate/continue, image attach, image
generation mode, KV "sessions"), **Brio / System One** (document + multi-question scoring with probability bars
and entropy), **Brain** (live routing: canvas grid polling `GET /experts` every 1.5 s, one cell per expert
coloured by tier with flashing hits; plus an "Explore" 3D cortex built from the static 3.5 MB `experts.json`
atlas of 13,260 characterised GLM-5.2 experts), **Profiling** (polls `/profile`, last 120 turns), and settings.
The Brain tooltip hardcodes GLM-5.2 layer mapping. 6 locales (en/it complete; de/zh-CN/zh-TW roughly 100
keys each, so partial). Conversations live in memory only; the API key is never persisted. The server serves
`web/dist` itself. Playwright scripts in `web/tests/` are **not wired into CI** (CI runs build + vitest on Node 22).

### Desktop (`desktop/`)
A "first increment": Tauri v2 shell, a 5-line `lib.rs`, no plugins, `core:default` capabilities only, CSP
restricting `connect-src` to localhost. It does **not** bundle or manage the engine, and no workflow builds it.

### Docker (`docker/`)
`Dockerfile` clones upstream and builds (not your local tree). `Dockerfile.slim` is a two-stage portable
(`x86-64-v3`) build, GLM engine only, non-root, model mounted at `/model`, port 5000; its Python module list is
hand-maintained and can drift. Compose requires `MODEL_DIR`. CI builds the slim image; **no image is published**.

---

## 10. Testing and quality discipline

- `make check` (= `clean` + `portable` + `test`): one portable CPU build, all C unit tests (`tools/run_tests.py`),
  and Python `unittest` discovery. No downloads, no CUDA, no torch. ~14 min on Linux (a comment).
- `c/tests/`: ~213 C tests, ~193 Python test files (~2,534 `def test_` by grep, not a collected count), 20 `.cu`,
  2 `.mm`, 14 `.sh` Vulkan harnesses. Python uses stdlib `unittest` only. `fake_hub.py` fakes the Hugging Face
  hub (tree API, Range, fault injection) so setup tests never touch the network.
- **Oracles on tiny fixtures:** `tools/make_*_oracle.py` / `make_*_tiny.py` build random-weight real-architecture
  miniatures plus a reference token stream from the reference implementation (e.g. `make_glm_oracle.py`
  requires transformers ≥ 5.11 for correct RoPE). The engine replays it **token-exact**
  (`ORACLE_STRICT=1`; teacher-forcing allows ≤2 mismatches for GLM near-ties). This is why LTO is off by default
  and integer kernels are ISA-bit-identical. Some fixtures are committed (`clef_tiny`, `deepseek_v4_tiny`,
  `kimi_k3_tiny`, `laya_tiny`, `mimo_tiny`, `gliner_decide_tiny`); the rest are gitignored and regenerated.
- **CI** (`ci.yml`, 1,937 lines, ~28 jobs; `check.yml`): Linux/macOS/Windows engine builds, sanitizers (ASan+UBSan
  + rANS fuzz), musl, Vulkan-on-Lavapipe matrix, CUDA/HIP *syntax* builds, tiny-oracle jobs per family, cluster
  parity, TypeSafe SDK check, web, Python, ARM64.
- **Benchmark discipline** (`CONTRIBUTING.md`, `docs/benchmarking.md`): perf PRs must attach an experiment
  manifest validated by `c/experiment_manifest.py` — full 40-hex commit, ≥3 raw samples per arm, medians derived
  from samples, hashed evidence, passing quality gate, and **exactly one changed variable**. Negative results are
  first-class. `docs/baselines/` is a manifest protocol for Colibri-vs-SGLang-vs-vLLM (explicitly *not* a ranking).

---

## 11. Build and release

- **Build:** `make -C c <engine>`; ISA from `-march` (`ARCH=native` default on Linux x86; `make portable` →
  `x86-64-v3`/`armv8-a`/`power8`); backends via `CUDA=1`, `HIP=1`, `METAL=1`, `VK=1`, `XDNA=1`; Windows uses MinGW
  engines + a separate nvcc/MSVC DLL for CUDA. `.build-config` stamp forces relink on flag change. A Nix flake
  is provided.
- **Release** (`release.yml`, on `v*` tags): linux-x86_64 (`x86-64-v3 VK=1`), macos-arm64 (`METAL=1` for
  colibri/inkling/kimi_k3), windows-x86_64 (`XDNA=1 VK=1`), plus a separate `windows-x86_64-cuda` zip
  (`coli_cuda.dll`, sm_80+). 12–13 engines per archive plus `coli`, Python modules, `tools/`, `shaders/`,
  `web/dist`. A verify step unpacks into a clean directory and checks dispatch for every family, shaders, and
  that Vulkan engines run on Lavapipe from the archive. Not released: desktop app, Docker images, pip wheel,
  Linux ARM. `site.yml` deploys `site/` to GitHub Pages.
- **Docs site:** Mintlify (`docs/docs.json`) with an OpenAPI-driven API reference.

---

## 12. History and community

- First commit 2026-07-01; HEAD 2026-10-17; **3,738 commits** on `main` (~1,000/month Jul–Sep), 1,315 merges, 217
  authors (maintainer **JustVugg / Vincenzo Fornaro**, ~1,159 + ~976 commits under two identities, then
  ZacharyZcR 217, monotophic 156, kreuzzelg 77, …). 20 tags, v1.0.0 (2026-07-19) → v2.0.0 (2026-10-06).
- **Workflow:** `main` stays known-good; PRs target **`dev`**, which the maintainer fast-forwards into `main` in
  tested batches. PRs need a clean 0-warning build, the GLM oracle, and targeted validation.
- **Timeline:** 1.0 GLM-5.2 on ~25 GB RAM (07-19) → 1.3 Kimi K3 + Inkling → 1.5 DeepSeek V4 → 1.6.2 security
  release (six memory-safety advisories) → 1.7 Qwen3.6 → 1.10 Qwen3.8-Flash-Next → 1.11 DeepSeek V4.1 →
  1.12 dashboard redesign + System One on all engines → **2.0.0** (10-06; 169 PRs; every engine on any Vulkan
  GPU with expert tier + dense chain, up to 16 KV slots, MiMo/Qwen3-Coder/Qwen-Image/Qwen3.8-27B).
- Third-party: ds4/ggml (MIT, DSv4 CUDA router), a vLLM oracle (no vendored code), Swiftlet qpack.

---

## 13. Things to be aware of (observations)

Verified locally unless marked otherwise:

1. **The 37 "modified" files under `web/` are line-ending churn only.** `git diff` is empty with CR-at-EOL
   ignored (3,775 insertions = 3,775 deletions); the worktree is CRLF, the index LF, `core.autocrlf` false.
   There is no real in-progress change there; staging them would produce a noisy diff.
2. **Stale docs:** `docs/index.mdx` lists 6 engine families (there are 13); `docs/MAINTAINING-DOCS.md` says "four
   binaries"; `openapi.json` omits `/v1/systemone`, `/v1/images/generations` and `/metrics`; `serve_protocol.md`
   refers to `glm.c`; `c/tools/README.md` lists only a few tools; `PIPE` default is documented inconsistently
   (code: off except on Windows); the segment/edge docs say seven adapters (code: eight).
3. **`pyproject.toml` description still says "run GLM-5.2"**, though the project now runs 13 engines.
4. **`deepseek_v4.c`'s amalgamation generator (`_amalgamate_v4.py`) is not in the repo**, so the file is edited
   by hand despite being "generated" per the units file.
5. **CUDA kernels are never executed in CI** (compile-only); correctness relies on manual hardware runs.
6. **Discrete-GPU Vulkan is largely unmeasured by the maintainers** (README: "We have not measured a discrete GPU
   ourselves yet"; first number is a user report); no-ReBAR staging is likewise untested on real hardware.
7. **Desktop app, Docker images and the pip wheel are not part of the release pipeline.**
8. **Dockerfile.slim's hand-listed Python modules** can drift from `c/`.
9. Some Brio source comments are in Italian; the Brain tooltip hardcodes GLM-5.2 layer mapping.

*Caveat:* this document was assembled from five parallel read-only explorations plus my own spot checks (README,
Makefile/CONTRIBUTING, the `web/` diff, version). Line numbers refer to HEAD `e0db083c` and will drift. The
explorers did not read every line of the large backends or `docs/windows.md` / `docs/metal_implementation.md` in
full, and none of the quoted benchmark figures were reproduced.
