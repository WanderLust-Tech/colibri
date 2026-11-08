# Colibri ideas

Collected 2026-11-03 from a review of Ollama (`D:\code\ollama`, commit 355eaee3) and AirLLM
(README only), plus loose ends from the ACE-Step work. Effort is a rough guess (S = under a day,
M = a few days, L = a week or more). "Checked" means I read code or docs; "unchecked" means it
still needs looking at before we commit to it.

Status key: `[ ]` not started, `[~]` in progress, `[x]` done, `[-]` decided against.

## A. Several models at once (from Ollama)

How Ollama does it: the server only schedules, and each loaded model runs in its own runner
process. A request that names a model uses it if loaded, otherwise the scheduler loads it while the
request waits. Before loading it estimates the model's memory use and compares it with free memory
(20% headroom). If it won't fit it unloads an idle model, shortest keep-alive first, and waits for
the memory to come back. A model with a request in flight is never evicted. Models stay loaded for
5 minutes by default; `keep_alive` per request overrides that. Source: `server/sched.go`,
`envconfig/config.go`, `docs/faq.mdx`.

> **Update 2026-11-07:** A1-A5 and B1-B3 are all built, merged into local `main`, pushed to the fork and live on port 8000
> (`coli router`, [docs/router.md](router.md)). The whole fork's state is in [fork-changes.md](fork-changes.md).

- [x] **A1. Router in front of per-model workers (M-L).** One front door on port 8000 checks the
  API key once, merges `/v1/models`, and forwards each request to a worker by its `model` field.
  Each worker is today's `coli serve` on a loopback port with no key. Replaces the model switcher's
  restart. The dashboard and API clients see one server.
- [x] **A2. Load on demand, unload when idle (M).** Needs A1. The first request for a model starts
  its worker and waits; a worker with no requests for `keep_alive` is stopped.
- [x] **A3. Memory-fit check and eviction (M).** Needs A2. Compare the model's resident size with
  free RAM (64 GB machine, about 26 GB free with Qwen-Image loaded) with headroom, evict an idle
  worker if needed, never one that is rendering. Also Ollama's crash recovery: if a load crashes
  while others are resident, unload the others and retry once.
- [x] **A4. One render at a time across models (S).** Unlike Ollama's GPUs, colibri's models share
  the CPU: two renders at once each run several times slower. Keep several models loaded, but let
  only one compute at a time (a global lock, optional).
- [x] **A5. Limits (S).** A maximum number of loaded models and a queue limit, both with settings.
- [-] **A0. Interim: one server per model on separate ports (S).** Works today with the restart
  script and the same key. The dashboard's endpoint field switches between them. Only worth doing
  if A1 is far off.

## B. Keep-alive, ps, stop (from Ollama)

Needs A1/A2 to mean anything.

- [x] **B1. `keep_alive` (S).** A server default and a per-request override: a duration, `-1` to
  keep forever, `0` to unload now. An empty request with a longer keep-alive preloads a model.
- [x] **B2. `coli ps` and `coli stop MODEL` (S).** A command and an endpoint that list loaded
  models with memory use and expiry time, and unload one.
- [x] **B3. Dashboard: loaded models and an Unload button (S).** The model switcher becomes a
  load/unload control, not a restart.

## C. Integrations and configuration (from Ollama)

- [x] **C1. `coli launch` (S-M).** *Built 2026-11-07 on `feat/coli-launch`, merged and live: Claude Code and opencode, both through
  the environment (nothing written to their settings); Codex needs the Responses API, which colibri does not serve; settings-file
  tools (Zed, Cline, VS Code) left out. Neither tool is installed here, so they were checked against a stand-in program, not
  run for real. [launch.md](launch.md).*  Ollama's `ollama launch` (`cmd/launch`) configures coding tools
  (Claude Code, Codex, opencode, Zed and others) to use the local server in one command. Colibri
  already serves `/v1/chat/completions` and `/v1/messages`, so most of the work is writing each
  tool's configuration. Unchecked: which tools are worth supporting.
- [ ] **C2. Named presets (S-M).** Ollama's Modelfile bakes a system prompt and parameters into a
  named variant. For colibri: saved presets such as "photo 1024x768, 20 steps" or "lo-fi 60 s" that
  appear as selectable model names.
- [ ] **C3. Windows tray or autostart (M).** Ollama's `app/` has a tray app and a start at login.
  Would replace `restart-colibri.ps1` for everyday use.
- [ ] **C4. One place for settings, and request logging (S-M).** Ollama keeps every server setting
  in `envconfig` and prints them with `ollama serve -h`, and can log each inference request.
  Colibri's settings are spread across `docs/ENVIRONMENT.md` and the code.
- [x] **C5. Chat model features.** *Surveyed 2026-11-07 (read-only, on `main` at a75c06e4).* Ollama has an embeddings endpoint,
  structured JSON output, tool calling, and speech-to-text and text-to-speech. Colibri today:
  - **Tool calling: yes**, on GLM-5.2, GLM-5.3-Flash, DeepSeek V4 and V4.1, Kimi K3, MiMo, Qwen3.6 (so Qwen3-Coder), Qwen3.8 and
    Gemma 4, with the OpenAI and Anthropic forms (the matrix in `docs/api.md`). No on Inkling and OLMoE (HTTP 400). *Fixed
    the same day (branch `fix/tooling-matrix`):* `family_registry.py` said `tools: False` for `qwen36` although its tool calls
    work, and the existing consistency test never noticed because it looked for a renderer named after the family and found none.
    It now asks the gateway's real dispatcher, `docs/api.md` has the missing DeepSeek V4.1 row and a second table (JSON mode,
    pictures, audio), and `c/tests/test_api_matrix_docs.py` fails when either table drifts from the registry.
  - **Structured JSON: accepted but only works on GLM-5.2.** The gateway reads `response_format` (`json_object`, `json_schema`,
    `gbnf`) and turns it into a grammar, but only the `glm` family takes a grammar payload; every other model, including
    Qwen3-Coder and Gemma 4, answers HTTP 400 "not supported by the X engine yet". Even on GLM it does not constrain anything:
    "grammar-forced drafts" (`docs/grammar-draft.md`) only draft the bytes the grammar forces and the model verifies them, so valid
    JSON is not guaranteed (checked in the doc, 2026-11-07). **No engine can promise valid JSON; this is the real gap.**
    Reusable parts exist: `c/grammar.h` (a byte-level GBNF pushdown automaton with `gr_accept` and `gr_admissible`) and
    `c/schema_gbnf.h` (JSON schema to GBNF).
  - **Embeddings: no.** No `/v1/embeddings` route, and no engine produces embeddings.
  - **Speech to text: only through chat, on Gemma E2B/E4B.** Send `input_audio` (WAV) in a chat message (JFK clip transcribed word
    for word); there is no `/v1/audio/transcriptions` route. Inkling also takes audio (DMel frames, command line only).
  - **Text to speech: no.** ACE-Step makes music, not speech.
  - **Not served at all:** `/v1/responses` (what Codex needs), and `/v1/completions` exists as the legacy route.
  - Pictures in chat: GLM-5.3, Qwen3.6, Qwen3.8, DeepSeek V4.1, MiMo and Gemma 4.
  - What would be worth building, if anything: (a) JSON mode for the models in daily use (Qwen3-Coder, Gemma 4) as a real
    sampling constraint, which is also what makes `tool_choice: required` honest; (b) an `/v1/audio/transcriptions` route that
    wraps Gemma E-model audio, so Whisper-style clients work. (The registry flag fix is done.)

## D. Memory and disk (from AirLLM)

AirLLM streams every layer for every token. Colibri already streams only the experts a
mixture-of-experts model picks, with prefetch, an expert cache, multi-SSD striping, int4/int8
conversion and `TRUNK_RESIDENT_LAYERS`, so AirLLM adds little for those models.

- [-] **D1. Delete the original after a successful convert (S).** *Mostly already there (2026-11-03): `coli convert --repo` deletes each downloaded shard once it is converted, and `convert_gguf_to_olmoe.py` has `--remove-source-file`. Only a hand-supplied `--indir` keeps its source; deleting someone's originals is destructive, so not built unless wanted.* AirLLM's `delete_original=True`.
  `convert_gguf_to_olmoe.py` has `--remove-source-file`; I did not find an equivalent in
  `coli convert` or the setup flow. Delete only after the whole conversion has succeeded.
- [-] **D2. Free pipeline stages in the image and music engines (M).** *Already in the engine (2026-11-03): `COLI_IMG_TE=stage` frees the 6.95 GB text encoder after each prompt (the router passes its environment to workers). It costs about 14 s per new prompt, and on a 64 GB machine the three models fit loaded together, so nothing to gain here; worth turning on only where memory is tight.* The Qwen-Image text encoder
  (6.95 GB) is needed only at the start of a request, then the transformer (7.09 GB) runs the
  steps. Unloading the encoder after use and reloading it lazily helps two models share memory.
  Unchecked: whether the engine already releases it.
- [ ] **D3. Check `TRUNK_RESIDENT_LAYERS` on a dense model (S).** It is the closest thing to
  AirLLM's layer streaming. Try it on the dense Qwen3.8-27B and see whether it is useful.

## E. ACE-Step and image loose ends

- [x] **E1. Save finished songs to disk (S).** *Built 2026-11-03 on `feat/free-pipeline-stages` (`COLI_MUSIC_DIR`, `COLI_MUSIC_SAVE=0`).* The image route now keeps a copy of every picture in
  `~/colibri-images`; the music route doesn't, so a stalled browser tab could lose a song the same
  way.
- [ ] **E2. Audio-in cases in the small CI fixture (S).** The tiny fixture covers text to music
  only.
- [ ] **E3. The 5 Hz language-model planner (L).** ACE-Step's optional planner
  (`acestep-5Hz-lm-1.7B`, also 0.6B and 4B), which writes metadata and audio codes first.
- [ ] **E4. Lego, extract and complete tasks (L).** Need other ACE-Step checkpoints.
- [ ] **E5. Push `main` to the fork (S).** *Partly done:* the fork's `main` has everything through the router
  (`e0414e87`). Local `main` is 17 commits ahead (song saving, KV slots, Gemma 4), not pushed. Ask before pushing.

## G. Gemma 4 follow-ups (added 2026-11-07)

Gemma 4 itself (26B-A4B, 31B, E4B, E2B, pictures, audio, batched decode) is built and merged into local `main`, not deployed.

- [x] **G1. Deploy Gemma 4 (S).** *Done 2026-11-07: `gemma4.exe` built in the live clone, the 26B-A4B and E2B containers in
  `colibri-models`, router restarted; a chat to the E2B through port 8000 worked.* Build `gemma4.exe` in the live clone, move a container into `colibri-models`, restart the
  router. Interrupts work in flight, so only when the server is idle.
- [ ] **G2. GPU offload (M-L).** Send the big matrix multiplications, the vision and audio towers first, through the
  existing Vulkan `coli_vk_matmul`. Ask first: it shares the GPU with renders. Gain unmeasured.
- [ ] **G3. Real runs of the 31B and the E4B (S).** Only the tiny random models have run them.
- [-] **G4. Offer Gemma in `coli setup` (S).** *Not possible as the catalogue works (2026-11-07): it lists only downloads that
  run as they are, and Gemma needs `convert_gemma4.py` first. It would need a ready container hosted on Hugging Face, which
  is publishing, so your call. Done instead on `feat/gemma4-setup` (fdebaae5, not merged): `coli convert --repo` takes a
  local directory, tests for the routing, the docs say how to convert, and the stale Makefile comment is fixed.*
- [ ] **G5. Video, and audio other than WAV (M).**

## F. Considered and set aside

- [-] **Streaming the denoising transformer from disk.** It runs every layer on every one of 8-35
  steps, so each step would re-read the whole model.
- [-] **AirLLM-style LoRA training on streamed layers.** Novel, but colibri is an inference engine;
  a large separate project.
- [-] **AirLLM's layer-by-layer loading for mixture-of-experts models.** Colibri's expert-level
  streaming reads far less per token.

## Suggested order

1. A1 + A2 + A3 + A4 together, then B1-B3 (the main improvement; replaces restart and switch).
2. D2, which makes A3 cheaper.
3. D1, E1 and C1: small and independent.
4. C2-C5 and the rest as wanted.
