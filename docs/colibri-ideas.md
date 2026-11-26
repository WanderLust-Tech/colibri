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
- [x] **C2. Named presets (S-M).** *Built 2026-11-09 on `feat/presets-settings`: a presets file, applied by the router, `coli preset`, [presets.md](presets.md).* Ollama's Modelfile bakes a system prompt and parameters into a
  named variant. For colibri: saved presets such as "photo 1024x768, 20 steps" or "lo-fi 60 s" that
  appear as selectable model names.
- [x] **C3. Windows tray or autostart (M).** *Built 2026-11-09 on `feat/tray`: `coli tray` (the icon, its menu, start with Windows) over `coli service` (start, stop, restart, status, link); [tray.md](tray.md).* Ollama's `app/` has a tray app and a start at login.
  Would replace `restart-colibri.ps1` for everyday use.
- [x] **C4. One place for settings, and request logging (S-M).** *Built 2026-11-09 on `feat/presets-settings`: `coli settings` over `ENVIRONMENT.md` with a typo check, and a JSON-lines request log ([request-log.md](request-log.md)); both unmerged.* Ollama keeps every server setting
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
  - **Structured JSON: built 2026-11-07 on `feat/json-mode`, merged into local `main` 2026-11-08 (not deployed).** Real constrained decoding on Gemma 4 and
    Qwen3.6/Qwen3-Coder, [json-mode.md](json-mode.md); the survey text below is what it replaced. *Before:* accepted but only
    worked on GLM-5.2. The gateway reads `response_format` (`json_object`, `json_schema`,
    `gbnf`) and turns it into a grammar, but only the `glm` family takes a grammar payload; every other model, including
    Qwen3-Coder and Gemma 4, answers HTTP 400 "not supported by the X engine yet". Even on GLM it does not constrain anything:
    "grammar-forced drafts" (`docs/grammar-draft.md`) only draft the bytes the grammar forces and the model verifies them, so valid
    JSON is not guaranteed (checked in the doc, 2026-11-07). **No engine can promise valid JSON; this is the real gap.**
    Reusable parts exist: `c/grammar.h` (a byte-level GBNF pushdown automaton with `gr_accept` and `gr_admissible`) and
    `c/schema_gbnf.h` (JSON schema to GBNF).
  - **Embeddings: no** when this was surveyed; **built 2026-11-19**: `/v1/embeddings` on Qwen3-Embedding 0.6B, 4B and 8B (`c/qwen3embed`, [embeddings.md](embeddings.md));
    the reranker that pairs with them, `/v1/rerank` on Qwen3-Reranker, **built 2026-11-20** ([rerank.md](rerank.md)).
  - **Speech to text: on Gemma E2B/E4B.** Send `input_audio` (WAV) in a chat message (JFK clip transcribed word
    for word). *Built 2026-11-08 on `feat/audio-transcriptions` (not merged):* a `/v1/audio/transcriptions` route over the same path,
    with cutting of long files, [api.md](api.md#speech-to-text-v1audiotranscriptions). Inkling also takes audio (DMel frames, command line only).
  - **Text to speech: no** from ACE-Step, which makes music, not speech. Both directions got dedicated engines on 2026-11-21: speech to text,
    Qwen3-ASR ([qwen3-asr.md](qwen3-asr.md)), and text to speech, Qwen3-TTS ([qwen3-tts.md](qwen3-tts.md)); the plan is [qwen3-speech.md](qwen3-speech.md).
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
- [-] **D3. Check `TRUNK_RESIDENT_LAYERS` on a dense model (S).** *Not testable on this PC (2026-11-10): it works on Linux, macOS and FreeBSD only; on Windows the trunk stays resident (docs/ENVIRONMENT.md).* It is the closest thing to
  AirLLM's layer streaming. Try it on the dense Qwen3.8-27B and see whether it is useful.

## E. ACE-Step and image loose ends

- [x] **E1. Save finished songs to disk (S).** *Built 2026-11-03 on `feat/free-pipeline-stages` (`COLI_MUSIC_DIR`, `COLI_MUSIC_SAVE=0`).* The image route now keeps a copy of every picture in
  `~/colibri-images`; the music route doesn't, so a stalled browser tab could lose a song the same
  way.
- [x] **E2. Audio-in cases in the small CI fixture (S). Done 2026-11-22** (`make_acestep_tiny.py --audio-in`, [acestep.md](acestep.md)): the VAE encoder, the
  tokenizer, a cover, a reference clip and a repaint are held to upstream's handler in CI, 8 + 5 + 19 cases at about 1e-6, with their mutants (28 of 28, 23 of 23
  and 45 of 45 once the mutants a fixture cannot show are left out). Building it found that the 3.4 MB fixture's encoder could never hold 64 latents (an encoder's
  last layer is the mean and the scale, so 64 needs 128 channels): the audio-in fixture is a second, 7.8 MB one with a flat 128-wide encoder. It also found one
  real difference from upstream, unreachable with Qwen's tokenizer: when the text-to-music instruction tokenizes SHORTER than the cover one, upstream pads the
  non-cover prompt with pad tokens that its cross-attention does not mask (it builds that mask from the audio's, not the text's); the engine now does the same.
- [x] **E3. The 5 Hz language-model planner (L). Done 2026-11-22** ([acestep.md](acestep.md#the-5-hz-planner-a-language-model-writes-the-song-first)): both phases, held to
  upstream's `LLMHandler`. Phase 1 (the model writing the metadata under upstream's constrained decoder, so that a bare caption is enough: tempo, key, time signature,
  duration, language and a rewritten caption) is identical to upstream's in every step of 7 cases on the real 1.7B (state, tokens let through, logits at 1e-6, the token), in its
  token tables, its parse and its PyYAML-exact YAML (6229 strings, 1546 texts); phase 2 (the codes) and the codes into the music as before; through the engine, the protocol, the gateway
  (`thinking`, `use_cot_caption`, `use_cot_language`, `lm_*`; the `meta` stage and `colibri.meta`) and `coli music --plan`. Open: the `genres` field (refused, not guessed), the planner's
  other jobs (understanding a clip, formatting), the 4B and 0.6B on real weights (done 2026-11-23). Done after: the planner on the Vulkan card (phase 1 22 s to 2 s, same tokens as upstream), `coli model install` of the planner and the base DiT, the dashboard's controls. The DiT's denoising is on the card too (a step 1.5 s to 0.1 s: a 10 s song is 4 s, 13 s with the planner); what is left of the time is the VAE decode and the condition encoders on the CPU (a 2-minute clip's decode is 30 s).
- [x] **E4. Lego, extract and complete tasks (L). Done 2026-11-22** ([acestep.md](acestep.md#the-base-and-sft-dits-guidance-extract-lego-complete)): the base and SFT
  DiTs (guidance by APG or ADG against the null condition, up to 200 steps) and the three tasks, held to upstream's handler on 19 cases of the real base checkpoint
  and the tiny fixture (29 of 30 mutants caught; the 30th needs a tokenizer whose cover and text instructions differ in length), through the engine, the protocol, the gateway (`dit`, `guidance_scale`, `track_name`, `complete_track_classes`)
  and `coli music --extract/--lego/--complete`. Not done: `coli model install` of the base DiT (it goes in by hand), the dashboard's controls, the Heun and SDE samplers.
- [x] **E5. Push `main` to the fork (S).** *Done 2026-11-10: the fork's `main` is at `0b51fd62`, level with local `main` and the live clone.* Ask before each push.

## G. Gemma 4 follow-ups (added 2026-11-07)

Gemma 4 itself (26B-A4B, 31B, E4B, E2B, pictures, audio, batched decode) is built and merged into local `main`, not deployed.

- [x] **G1. Deploy Gemma 4 (S).** *Done 2026-11-07: `gemma4.exe` built in the live clone, the 26B-A4B and E2B containers in
  `colibri-models`, router restarted; a chat to the E2B through port 8000 worked.* Build `gemma4.exe` in the live clone, move a container into `colibri-models`, restart the
  router. Interrupts work in flight, so only when the server is idle.
- [x] **G2. GPU offload (M-L).** *Dense matrices done 2026-11-10 on `feat/gemma4-vulkan` (`COLI_VULKAN=1`, opt-in): the same text, 1.3-3x on prompts and pictures, 1.4-1.9x on decoding; deployed 2026-11-10 (the live `gemma4.exe` is the `VK=1` build; the router decides per model). An on-device chain for a decode row done 2026-11-18 on `feat/gemma4-chain` (docs/gemma4.md: E2B 5.4 ms a token against 28 to 34, E4B 3.8 against 40, the 26B-A4B 51 against 64 to 80). The experts of the 26B-A4B on the card done 2026-11-18 on `feat/gemma4-experts-gpu` (the shared expert tier with GeGLU: 25 to 28 ms a token against 51, prompts 112 tokens a second against 55). Next step if wanted: the router on the card (a routed layer is two round trips today), and the chain for a prompt.* Original note: Send the big matrix multiplications, the vision and audio towers first, through the
  existing Vulkan `coli_vk_matmul`. Ask first: it shares the GPU with renders. Gain unmeasured.
- [x] **G3. Real runs of the 31B and the E4B (S).** *Done 2026-11-10: the E4B read text, a picture and the JFK clip; the 31B read text and a picture (docs/gemma4.md). Neither is installed in `colibri-models`.*
- [-] **G4. Offer Gemma in `coli setup` (S).** *Not possible as the catalogue works (2026-11-07): it lists only downloads that
  run as they are, and Gemma needs `convert_gemma4.py` first. It would need a ready container hosted on Hugging Face, which
  is publishing, so your call. Done instead on `feat/gemma4-setup` (fdebaae5, merged): `coli convert --repo` takes a
  local directory, tests for the routing, the docs say how to convert, and the stale Makefile comment is fixed.*
- [x] **G5. Video, and audio other than WAV (M).** *Done 2026-11-10 on `feat/gemma4-video-audio`: video as up to 32 frames (a file through ffmpeg, or frames the client sends), MP3 by minimp3, FLAC/Ogg/M4A through ffmpeg; the sound of a video is not read (docs/gemma4.md).*

## H. Router and extension follow-ups (added 2026-11-10)

- [x] **H1. Unload an idle model to make room on the card (M). Done 2026-11-16** (router.md, "An idle model that holds the graphics card"; `COLI_CARD_EVICT=0` turns it off). The router picks card or CPU when a model loads, from the free
  memory; a model that sits idle on the card for its keep-alive makes the next one go to the CPU. It could unload the idle one first
  (the policy would have to know which models hold the card; Windows has no per-process figure, so it would track its own workers).
- [x] **H2. A context window per model (S-M). Done 2026-11-22** ([model-settings.md](model-settings.md)): a `context` field in `model-settings.json`
  (`coli model context NAME 32k`, `POST /v1/models/settings`, a selector on the Model page); the router starts that model's worker with it, in the variable its
  family reads, over the router's own `CTX`; a window the family cannot hold is refused, a changed window unloads the model. Not done: the model card saying the
  served window before the first load (H3's remainder: the window it will have is now known from the settings, the family's default is not shown).
- [x] **H3. Keep `context_length` for models that have not run since the engines announced it (S). Partly done 2026-11-16:** `details.context_length` in `/v1/models/installed` is the folder's own limit from config.json, shown on the Model page before a first load; the card's `context_length` (what the server serves) is still only known after a load. Today the card appears
  after the model's first load.
- [x] **H4. Qwen-Image on Vulkan (M). Measured 2026-11-17** (qwen-image.md): on the RTX 5090 a 512x512, 8-step picture takes 10 s against the
  CPU's 95 s, and is the CPU's picture (SSIM 0.999999, no pixel off by more than 1 of 255). Not measured: 1024x1024, and a card with less room.
  ACE-Step has no Vulkan path.
- [x] **H6. Report reused tokens in `usage` (S).** Done on `feat/cached-tokens`: the engine puts the reused count on the end of its
  `DONE ... STAT` line (qwen36, gemma4 and deepseek_v4 do; the others leave it off), and the gateway returns it as
  `usage.prompt_tokens_details.cached_tokens` (also on the streamed usage chunk) and as `cached_tokens` in the request log. A field
  the engine did not send is left out, never shown as 0. Measured on Qwen3-Coder and Gemma E2B: cold 0, a continuation nearly all
  of the last prompt; after an edit to the system message Qwen reports 0 (all or nothing) and Gemma the part before the edit.
  Still to do: the VS Code extension showing it (the `cache:` log line could say "kept, N of M tokens").
- [x] **H5. VS Code extension (its own roadmap lists them). Done, or decided against, by 2026-11-20** (0.13.0 and 0.14.0; the extension's README roadmap has the detail):
  picture, audio and video attachments for the models whose card lists them, presets in the Models menu, the key in SecretStorage, and a model installer are built;
  `cache_slot` per chat was decided against (Colibri places each turn by its message prefix and the engine already reuses what it has read; what broke that was
  the extension changing the start of the prompt, fixed in 0.4.1); a per-model reasoning switch is not needed (the gateway turns `reasoning_effort` into each
  model's own switch). What is left is the extension's own list: showing the reused tokens (H6), an editable diff, a preview for `run_command`.

## I. Ideas of 2026-11-25 (listed, not started)

Collected in one go; what each rests on is stated, so the next look does not start from nothing. Sizes as above.

**Already known, now in one list**

- [~] **I1. The chain for prompt chunks and for several conversations (M-L).** Prompt chunks are done (fork-changes 2.46: a chunk of up to 256 rows is one frame, 26B-A4B 2k tokens
  31 s to 1.7 s); what is left is a batch of conversations' decode rows (needs a device KV mirror a conversation; live use is one slot, so it waits for a user of `--kv-slots`),
  and the attention at 4k+ tokens (done in a local commit: fork-changes 2.47, `chain_attnf`, a register-tiled float form, 8x; the matrix-unit flash is not accurate enough for Gemma). Original text: Gemma 4's decode chain (2.19, [gemma4.md](gemma4.md)) is one
  submission a token for one conversation. A prompt chunk (64 rows at a time) and a batch of conversations still go through the per-matrix
  path, which is why a 26B prompt and a batch are slower than a single token's chain. Listed as "Not done" in fork-changes 2.19.
- [-] **I2. Gemma gives the card back instead of being stopped (M).** Set aside 2026-11-25 after reading the pool code, not built: the router already stops an idle holder (60 s) when another model needs the card, and `Pool.promote` already returns a CPU chat model to the card when there is room; the Coder landing on the CPU beside a just-used 26B is that holder not being idle yet, which a Gemma `DEMOTE` would not change. A demoted 26B would only keep running on the CPU (a 7k prompt takes minutes there, against 12 s to reload onto the card), and the teardown would live in the shared Vulkan layer. If the annoyance matters, the lever is `card_min_idle` (60 s, a constant in `router.py`). Original text: `Pool.demote` exists for the `qwen36` engines only; a Gemma holding the
  card is stopped, and the 26B holds 15 GiB of it now. Needs a `DEMOTE` in the Gemma engine's own tier. Named in fork-changes 3.4 ("demoting
  engines other than `qwen36`"). It touches the card code the Gemma 26B shares with other engines (`vk_tier.c`), which is the same care as I-set-aside
  below: the user declined work there once, so settle that first.
- [ ] **I3. Gemma's chain and tier on an integrated GPU (S, a measurement).** Both are off by default there because they were never measured on a card
  that shares the CPU's RAM ([gemma4.md](gemma4.md), the chain and the tier). Needs a machine with one; the question is whether either wins.
- [x] **I4. Qwen-Image at 1024 x 1024 (S, a check).** Done 2026-11-25 on the 5090, through the router ([qwen-image.md](qwen-image.md)): a
  1024x1024 render takes 14.3 s at 8 steps (27.4 s at 16, which looks visibly better at that size); an edit with a 1024 reference takes 93.5 s
  against 51 s with a 768 one and gives the same result to the eye (SSIM 0.967), so the default stays 768.

**New**

- [x] **I5. The VS Code extension works on what is open in the editor (M).** Built as extension 0.25.0 (2026-11-25; the vsix is built, not yet installed): an open file is read from its buffer and edited through it (one undo step, the diff on the card against the buffer; saved after the edit when it had no unsaved changes, left unsaved when it had; a file that is not open stays on disk and is not opened), plus `list_editors`, `diagnostics` and `open_file`; `colibri.editorTools` switches it off. The open questions below were decided that way. Original text: Today the model's file tools read and write files on disk and the
  extension attaches the selection; it cannot see which files are open in tabs, what is unsaved, or the Problems list, and its edits go to disk
  rather than through the editor (no undo step, an unsaved buffer is bypassed). A set of tools: list the open editors (with a dirty flag), read a
  buffer (unsaved text, the selection, a range), edit a buffer with a `WorkspaceEdit` (one undo step, through the approval and editable diff that
  `write_file` already has), read the diagnostics (errors and warnings of a file), open a file at a line. Open questions: what to do when a buffer
  is dirty and the model edits the file on disk, and whether an edit to a file that is not open should open it.
- [~] **I6. Music: mixes (M, to be pinned down).** **The transition reading was tried 2026-11-26** ([acestep.md](acestep.md#a-transition-between-two-songs-by-repaint)): it works with
  what the engine has, but needs a caption, `balanced` or `conservative` (never `aggressive`), and a level match the engine does not do; whether it *sounds* good is for a
  listener (the files are in `~/colibri-music/transition-test/`). A `coli music blend A B` that does the steps below is the build, if wanted. The remix reading still needs I7.
  The original text: Two readings: a transition between two songs, or a remix. The first fits what the engine does
  today: join the tail of clip A and the head of clip B with a gap and repaint the seam with a prompt (the repaint task, 2.33), a cheap thing to try
  before building anything; the second is stems recombined (see I7). Ask which is meant before choosing.
- [ ] **I7. Music: vocals out of a song (M-L).** Three routes. (a) The extract task pulls a named track (`vocals`, `drums` ...) out of a clip
  ([acestep.md](acestep.md), tasks table), but it regenerates the stem with the DiT, so it is not the original sound and subtracting it from the mix
  does not give a clean instrumental. (b) A cover at low strength with an instrumental caption keeps the tune and drops the voice (no new code; try
  it). (c) A proper stem separator (an HT-Demucs or MDX-style model, tens of MB) as a new engine like the ASR and TTS ports: exact stems from the
  original audio, which is what "strip the vocals" and any later remix need. (c) is the real answer and the larger job.
- [x] **I8. Video understanding (S to measure, L to extend).** Measured 2026-11-26: Gemma 4 reads video on the card (the 26B-A4B gets the
  direction of motion right both ways, the E4B not; 38 to 52 s for a 6-second clip, 61% of it the vision tower; the numbers and one hallucination
  caution are in [gemma4.md](gemma4.md#video)). Kimi K3 is **not** the answer: its card says 2.8T parameters (104B active, MXFP4), a 401M
  MoonViT-V2 vision encoder, and "text, images, and video" in its prose but "Text, Image" in its spec table, with no video example in the
  vLLM or SGLang usage; the weights are about 1.5 TB, so a vision port would be work for a model this machine cannot run. What is left is to
  name a better video model than Gemma 4, or to take the time down (fewer frames, and finding what the 14 s after the tower is). The original text: Gemma 4 already takes video (frames and, on the E models, the soundtrack, through
  `video_url`; fork-changes 2.14, G5), so a video question works today. The Kimi K3 engine in this fork (`c/kimi_k3.c`) is text-only and nothing in
  the repository says the model takes pictures or video; check the model card before planning a vision tower port (L). If what is wanted is
  "a better video model than Gemma 4", name it and see whether its weights and architecture are a port worth making.
- [x] **I9. The tray as a real program, not a Python script (M).** Built 2026-11-26: `colibri-tray.exe` ([tray.md](tray.md#the-tray-as-a-program-of-its-own),
  fork-changes 2.13): a C program over the same Win32 calls, held to `tray.py` by tests that run both; it leaves starting and stopping the
  server to `coli service`, so Python is needed for that and not for the icon. Not done: an installer, signing, release archives. The original text: `coli tray` is `c/tray.py` (ctypes over the Win32 API, run with `pythonw`,
  started at sign-in by a Run key; 2.13). A native `colibri-tray.exe` (a small C program over the same Win32 calls, or the script frozen) would
  need no Python on the path to show the icon and start the router, give the process a proper name and icon in Task Manager, and could go in an
  installer. The router and `coli` stay Python, so the tray would start them the way it does now.
- [x] **I10. Colibri as an MCP client: the models use MCP tools from the CLI, the dashboard and the extension (L).** The CLI half is built 2026-11-26
  ([mcp-client.md](mcp-client.md), fork-changes 2.48). The question the idea left open was settled with the user: **each client runs its own loop**, the router
  keeps only the shared list (a router-side loop would let any API-key holder make a model run commands here). The extension and the dashboard are
  I12. The original text: `coli mcp` is the other
  direction (Colibri as an MCP server for an assistant, [MCP_SERVER.md](MCP_SERVER.md)). The extension has its own MCP client and its own server
  list. The idea: one list of MCP servers kept by the router (a file in the setup folder, `coli mcp add`), tools offered to a chat request and run
  by the router in a loop (`coli chat --mcp`, a Tools switch and approval cards in the dashboard chat), with the extension able to use the same list.
  The weight is in the agent loop in the gateway (one generation at a time per model, approvals from three clients, a tool result going back into
  the next request), in where an MCP server's process runs and with what rights (they run local commands), and in what happens to the
  extension's own servers. Settle who owns the loop and the approvals before building.
- [x] **I12. The extension uses the router's MCP list (M).** Built 2026-11-26 as extension 0.28.0 ([mcp-client.md](mcp-client.md#the-vs-code-extension)): its own approval of the exact
  definition on that machine, references from its environment, the setting wins a name, a down router breaks nothing; vsix built, not installed. The dashboard's cards
  for HTTP servers stay open. The original text: The extension has its own `colibri.mcpServers` setting, its own MCP client and its own approvals.
  Reading `GET /v1/mcp/servers` as a second source (the setting wins on a name clash) needs the same rules the CLI keeps: the secret references expanded from
  the extension host's environment, and an approval on that machine for the exact definition (a hash like `mcp_registry.digest`, kept by the extension, so a
  server added to the list from elsewhere is not started unasked). A new vsix; the dashboard's cards for HTTP servers after that.
- [x] **I11. Choose where a model is installed (S-M).** Built 2026-11-25 (local commits, not pushed or deployed): `root` in `POST /v1/models/install` (one of the new `folders` of the catalog, each with its free space), `coli model install ID --dest FOLDER` and the folders listed by `coli model available`, a folder list in the dashboard's install confirmation and a folder pick in the extension (0.27.0). A folder that is not a models folder is refused with the list and how to add one (`COLI_MODELS_ROOT`); adding a folder from the API was left out. The list now judges a model's room by the folder with the most. docs/model-install.md. Original text: `coli model install` (and the dashboard and the extension's installer) put the model in the
  first models folder with enough free disk (`Installer.install_root`); there is no way to say which. A `--dest FOLDER` (a folder already scanned by
  the catalog, or one to add), a list of the folders with their free space in the dashboard and the extension's menu, and `root` in the install
  request; the model must end up in a folder the router scans, or be added to them.

## F. Considered and set aside

- [-] **Streaming the denoising transformer from disk.** It runs every layer on every one of 8-35
  steps, so each step would re-read the whole model.
- [-] **AirLLM-style LoRA training on streamed layers.** Novel, but colibri is an inference engine;
  a large separate project.
- [-] **AirLLM's layer-by-layer loading for mixture-of-experts models.** Colibri's expert-level
  streaming reads far less per token.

- [-] **Gemma 4's router on the card (declined 2026-11-23).** Moving the 26B's router onto the card would cut a share of the remaining
  per-layer round trips (a part of the 25 ms, not all of it) but needs changes in `vk_tier.c`, which other models share; judged too risky for one
  family. Do not start it again without a new reason.

## Suggested order

1. A1 + A2 + A3 + A4 together, then B1-B3 (the main improvement; replaces restart and switch).
2. D2, which makes A3 cheaper.
3. D1, E1 and C1: small and independent.
4. C2-C5 and the rest as wanted.
