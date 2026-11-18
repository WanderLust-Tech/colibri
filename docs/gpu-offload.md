# Running models on the graphics card

What was measured on this machine (2026-11-10), what each engine does with a card, how the router chooses between the card and the
CPU, and the traps met on the way. Written to be looked up later; the reference for the Vulkan backend itself is
[vulkan.md](vulkan.md), and for the variables [ENVIRONMENT.md](ENVIRONMENT.md).

**The machine:** Windows 11, i9-14900KF (32 threads), 64 GB RAM, RTX 5090 with 32 GB, the desktop and the user's renders in use,
the engines at below-normal priority. Every comparison is the same prompt at temperature 0, and every one gave the same words
on the card as on the CPU (the sums are added in a different order, so the logits differ in the last bits; an answer that sits
on a knife edge could differ). The timings are noisy for that reason: read a ratio of 1.3 as "faster", not as a number to quote.

## What it bought

| Model | Run | CPU | Card | Faster |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | 615-token prompt | 24.2 s | 12.8 s | 1.9 x |
| | 150 tokens out (code) | 14.8 s and 11.7 s (10 to 13 tok/s) | 9.0 s and 8.5 s (17 tok/s) | 1.4 to 1.6 x |
| Gemma 4 E2B | 425-token prompt | 8.6 s | 2.8 s | 3 x |
| | 150 tokens out | 13.1 s | 6.9 s | 1.9 x |
| | a picture and 40 tokens | 18.1 s | 12.7 s | 1.4 x |
| | a 32-frame video (WebM) | 62 s | 34.7 s | 1.8 x |
| Gemma 4 26B-A4B (32 experts cached a layer) | 429-token prompt | 19.0 s | 15.1 s | 1.3 x |
| | 150 tokens out | 19.2 s | 13.6 s | 1.4 x |
| | a picture and 20 tokens | 34.4 s | 23.0 s | 1.5 x |

**Through the live router** (all shaders present, `--gpu auto`, the card at 26 GB free when it loaded): the same 615-token prompt took
**1.5 s** and the 150-token answers 7.3 s (20 tok/s), 23.5 GB of the card used afterwards. The tree used for the table above was missing
`expert_act.spv`, which is why its prompt number is so much worse: with the full set of shaders the prompt is about 16 times faster than the
CPU's 24.2 s, not 1.9 times. Gemma 26B-A4B, asked for while Qwen3-Coder held the card (8.4 GB free, 6 kept back, 3 needed), was started on
the CPU by the policy, as designed.

**Qwen3-Coder had three states**, and the middle one is a trap: with the card for the dense layers only (the expert tier off because
a shader was missing, see [below](#traps)) the prompt took 19.6 s and the answers 10.8 s and 7.7 s. With the **expert tier** as
well, all 6,144 experts (48 layers of 128) sit on the card: 15.2 GiB, uploaded in about 7 s at start (`warm start, 6144 experts`),
and the CPU's routed-expert work disappears. Most of the gain is the tier.

**What it costs in card memory** (the card's used memory from `nvidia-smi` with the model loaded and answered, less the same with nothing
of ours loaded; the desktop and the user's programs held 5.5 GB at the time, and every figure went back to the baseline on unload):

| Model | On the card | Notes |
|---|---|---|
| Qwen3-Coder-30B-A3B | 17.3 GB | 15.2 GiB of experts, 1.15 GiB of dense matrices, chain buffers. The tier sizes itself to the free memory less `COLI_VK_TIER_RESERVE_GB`, so on a busy card it holds fewer experts |
| Gemma 4 E2B | 2.4 GB | dense matrices only; the host copy stays in RAM |
| Gemma 4 E4B | 4.4 GB | |
| Gemma 4 26B-A4B | 15.3 GiB (3.2 GiB with `COLI_VK_TIER=0`) | the dense matrices, the KV mirrors and, with the expert tier, all 3,840 experts (12 GiB, read from the driver while it decoded); `COLI_VK_TIER_GB` caps the experts, and the CPU computes the ones the card does not hold |
| Gemma 4 31B | about 30 GB (expected, not run on the card) | dense: all of it would have to fit beside the desktop's 5.5 GB, which a 32 GB card cannot do; with `G4_VK_RESERVE_GB` the matrices that do not fit stay on the CPU |

With nothing of ours loaded the card showed 5.5 GB used (the desktop and a browser); Qwen3-Coder on the card took it to 22.9 GB, and
the three Gemma models together would add 9.3 GB. A render that wants more than what is left is where the trouble starts, which is what
[`--gpu auto`](#choosing-by-free-memory) is for.

## What each engine puts on the card

| Engine | Models | On the card | State here |
|---|---|---|---|
| `qwen36` | Qwen3-Coder-30B, Qwen3.6, Qwen3.8-27B | the dense trunk (whole layers in one submission, the "chain"), and the routed-expert tier | built `VK=1` and measured: above |
| `gemma4` | Gemma 4 E2B, E4B, 26B-A4B, 31B | the dense matrices: attention, dense MLP, the vision and audio towers, the output head. Not the experts | built `VK=1`, live ([gemma4.md](gemma4.md#vulkan-for-the-dense-matrices)) |
| `qwenimage` | Qwen-Image-2.1 | the transformer: every block of a step on the chain. Not the text encoder or the VAE | **not built or measured here**; [qwen-image.md](qwen-image.md) |
| `acestep` | ACE-Step 1.5 | nothing: no Vulkan path | CPU only |
| others | GLM, Kimi, DeepSeek, MiMo, OLMoE, Inkling, Qwen3.8 Flash | see [vulkan.md](vulkan.md#the-other-engines) | not installed here |

One binary does both: a `VK=1` build runs on the CPU when `COLI_VULKAN` is not `1`, and on the card when it is. So the choice can be made
per worker, at the moment it starts.

## Choosing by free memory

`coli router --gpu auto` decides each time a model loads. It reads the card's free memory (`nvidia-smi`) and starts the worker with
`COLI_VULKAN=1` when `free - reserve >= min`, and with `COLI_VULKAN=0` otherwise.

```
coli router --gpu auto --gpu-reserve-gb 6 --gpu-min-gb 3
```

| Flag | Variable | Default | |
|---|---|---|---|
| `--gpu off\|auto\|on` | `COLI_GPU` | unset | unset: the router does not touch `COLI_VULKAN`; `on`: always the card; `off`: never |
| `--gpu-reserve-gb` | `COLI_GPU_RESERVE_GB` | 6 | free card memory to leave for everything else (renders, games, the browser) |
| `--gpu-min-gb` | `COLI_GPU_MIN_GB` | 3 | the least free memory beyond the reserve that is worth starting a model on the card for |

The reserve is also handed to the engines (`COLI_VK_TIER_RESERVE_GB` for the expert tier, `G4_VK_RESERVE_GB` for Gemma), so what they
place on the card leaves it free: a Qwen3-Coder started with 14 GB free and a reserve of 6 holds about 6 GB of experts, not 15.

What it does **not** do: it does not move a model that is loaded to the card (a model keeps its choice until it unloads, and the next
load looks again), and it cannot see a card without `nvidia-smi` (an AMD or Intel card: `auto` then leaves the environment alone and
says so in the router's log). What it does about a model that holds the card is [below](#giving-the-card-back).

`RUNS ON` and `running` report the router's decision. An engine that is not a `VK=1` build ignores `COLI_VULKAN`, so a model can say `GPU`
and still run on the CPU (ACE-Step does: it has no Vulkan path; the live `qwenimage.exe` is a CPU build). The engine's own log line
(`[VK] <engine>: device ready, dense matrices on the device`) is the proof.

A loaded model says where it runs: `coli ps` has a RUNS ON column (`GPU`, `CPU`), `coli model` shows it with the policy and the free
memory now, and `/health` carries it (`router.loaded[].gpu`, `router.gpu`).

A model can be pinned whatever the policy says: `coli model gpu qwen3-coder-30b cpu` (or `vulkan`, or `auto` to follow the policy
again). See [model-settings.md](model-settings.md), which also turns a model off. `COLI_GPU_FAKE_FREE_GB=8` stands in for the reading, to try a
policy out.

**Where to set it.** The restart script (`restart-colibri.ps1`) and the `env` setting of `service.json` (what the tray starts with) both
carry it: `COLI_GPU=auto` in place of a fixed `COLI_VULKAN=1`, which would put every model on the card whatever the card holds.

## Giving the card back

With `--gpu auto`, when a model about to start would go to the CPU only because an idle one holds the card, the router first asks that
one to **give the card back and keep running on the CPU** (`POST /v1/demote` to its gateway, a `DEMOTE` frame to the engine; see
[router.md](router.md)). Only the `qwen36` engine can (the Qwen3, Qwen3.5 and Qwen3.6 models, Qwen3-Coder among them). In order: the
dense chain brings its state down and goes off, the expert tier is torn down, the dense matrices that only the card held (1.15 GB for
Qwen3-Coder) are read back into RAM, and the device is destroyed, which is what hands the memory back to the driver. The engine is then the
CPU build for good: it does not go back to the card until it is stopped and started again. Any other engine, or an engine that says no, is
stopped as before.

Measured on this machine (2026-11-17), Qwen3-Coder with the tier on the card:

| | Card used (`nvidia-smi`) | Answer |
|---|---|---|
| before | 20.7 GB (the desktop alone holds 3.5) | 54 tokens in 3.0 s, prompt included |
| `POST /v1/demote` | back to 3.5 GB, answered in 2.3 s | |
| after | 3.5 GB | the same 54 tokens, word for word, in 23.4 s (cold expert cache) |

The CPU speed after the demotion is the CPU's: a fresh start of the same engine without the card answered a 68-token reply in 22.6 s
(three runs of the same prompt, 30.9 s then 22.6 s twice), and the demoted one answered the same prompt in 25.3 s both times it was asked. So
the gain is not speed, it is that the model is still loaded: a restart reads 19 GB again (15 s here, with the files in the system cache; minutes
from a USB drive) and loses the conversation cache. Through the router, with 12 GB kept back so that the card looked full once the Coder was on it: the Coder
idle for 76 s, a request for Gemma E2B made the router demote it in 2.2 s and start Gemma on the card (6.0 GB used, 26.2 free); the Coder
then answered a request on the CPU in 8.6 s. A short greeting came out differently on the card ("Hello there, nice to meet you!") and
after the demotion ("Hello there everyone!"), as a card and a CPU can at temperature 0 (the sums are added in a different order).

`COLI_CARD_DEMOTE=0` makes the router stop such a model instead; `COLI_CARD_EVICT=0` turns off making room altogether. `/health` shows
`demoted` and a `gpu` that says why the model is on the CPU. A model asked for while the router demotes it waits the few seconds and then
runs on the CPU. Not done: putting a demoted model back on the card when the card is free again (it needs the engine to take its tier and
chain up a second time, which it does only at start).

## Building it

`VK=1` builds link the backend, which opens the Vulkan library at run time (nothing links `vulkan-1.dll`), so such a binary still runs
on a machine without a card. On Windows with MSYS2 (`C:\msys64\ucrt64\bin` and `usr\bin` on the PATH):

```
make -n qwen36 VK=1        # prints the gcc lines (and glslc lines for the shaders)
```

and run the `gcc` lines. Running `make` itself from the Claude Code Bash tool fails ("Cannot create temporary file"), and so does
`bash script.sh`; `eval` each printed line in the current shell. The engine looks for its shaders in `shaders/` beside the binary
(`COLI_VK_SHADERS` to move them).

### Traps

* **A missing shader turns a feature off, quietly.** A fresh checkout has 35 of the `.spv` files; the others are ignored by git and
  must be compiled (`glslc --target-env=vulkan1.2 shaders/X.comp -o shaders/X.spv`). Without `qmatmul_gate_up.spv`, `rmsnorm.spv` and
  `attention_absorb.spv` the Qwen3-Coder log said `[VK] tier qwen36: no expert batch on this device (shaders?), the experts stay on
  the CPU`, and the run was only the dense half of the gain. Look for `[VK] cannot open ...spv` lines in the worker's log
  (`~/colibri-serve.err`, lines are prefixed with the model name). `expert_act.spv` and `expert_act_v4.spv` were also missing in the
  test tree and the tier still started; the live clone has all of them.
* **The shaders that git tracks are binaries too.** Running the shader step of `make` in the live clone rewrote the tracked `.spv` files
  (a different `glslc` makes different bytes); `git checkout -- c/shaders` puts them back. Compile only the missing ones.
* **Windows does not show per-process card memory** (`nvidia-smi` prints `[N/A]`), so only the total and the free figures are usable.
  That is what the router reads.
* **The card is shared with whatever is drawing.** The driver pages card memory between programs, so a render that finds the card full does
  not necessarily fail, it slows down. The reserve is the protection; a model on the card should not be left loaded while a long render runs
  (the keep-alive unloads it, or `coli stop NAME`).
* **Same words, not the same bits.** The tests that require a batched decode to equal a single-row decode to the last bit fail with the
  card on (12 of Gemma's 179); everything checked against transformers passes.

## Not done, and what it would take

* **Qwen-Image on the card.** The transformer is where the time goes; `make qwenimage VK=1` and `COLI_VULKAN=1`. Not built here. Its
  weights (7 GB for the transformer) and the 6.95 GB text encoder together are about what the card has to spare while a render runs.
* **Gemma 4 experts on the card** (the 26B-A4B's 12.9 GB of experts): the dense half is on the card, the experts are not. The Gemma
  engine has no expert tier; the shared one in `vk_tier.c` is what Qwen uses and is the way to add it.
* **A prompt chunk of Gemma on the chain.** A decode row's layers run in one submission ([gemma4.md](gemma4.md#the-decode-chain-gemma4_chainh):
  E2B 46 to 5.4 ms a token, E4B 97 to 3.8); a prompt chunk, several conversations decoding together and a row inside a picture
  still make a synchronous call for each of the ~200 dense matrices, which is why a prompt gains less than decode now does.
* **Putting a demoted model back on the card** ([above](#giving-the-card-back)), and demoting an engine other than `qwen36` (Gemma 4's
  dense path is on the card for 2.4 to 4.4 GB; the same teardown would give it back, and has not been written).

## Repeating the measurements

```
# the card, with the tier: the router's worker, or by hand
set COLI_VULKAN=1
python coli serve --model C:\Users\kp-mi\colibri-models\qwen3-coder-30b --port 8140 --cap 128 --api-key test
# a prompt of about 600 tokens, then 150 tokens out, twice, at temperature 0; compare with COLI_VULKAN unset
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

`COLI_VULKAN=1` alone starts the tier at the whole of the free card memory less its reserve (1 GiB), which is what took 17 GB;
`COLI_VK_TIER_GB=8` caps it.
