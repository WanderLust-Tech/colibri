# ACE-Step 1.5 in C

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) turns a caption and lyrics into a song: 48 kHz stereo, 10
seconds to 10 minutes, from a diffusion transformer over 25 Hz latents and an audio VAE. This directory of work ports
it to the same shape as [Qwen-Image](qwen-image.md): one C file, weights read as they are, every stage held to the
reference by an oracle that compares tensors, not just the final audio.

**Using it:** `coli music "calm piano with soft strings"` writes a song, `coli serve` answers `POST /v1/audio/music`
(see [In colibri](#in-colibri)). The engine itself is `acestep --model DIR --caption "calm piano" --out clip.wav`, and
every part of it is held to upstream's handler by an oracle. The dashboard has no player yet.

| stage | what | state |
|---|---|---|
| 0 | the reference: upstream's own handler, observed at every stage; a tiny random checkpoint of the real structure | done |
| 1 | the VAE decoder, `c/acestep_vae.h`: latents to 48 kHz stereo, tiled as upstream tiles | done, oracle-checked |
| 2 | the DiT, its condition encoders and the turbo sampler (DCW included), `c/acestep_dit.h` | done, oracle-checked |
| 3 | the text encoder (Qwen3-Embedding-0.6B), its tokenizer and the prompt templates, `c/acestep_text.h` | done, oracle-checked |
| 4a | the engine proper in C: a request in, audio out; one-shot and `--serve`; `c/acestep_gen.h`, `c/acestep_serve.h` | done, oracle-checked |
| 4b | around it: the registry's audio modality, `music_engine.py`, `POST /v1/audio/music`, `coli music`, `coli info` / `plan` / `doctor`, `coli setup`, `make install` | done, tested against a stub engine and smoke-tested on the real one |
| 4c | the dashboard's music player | |
| 5, 6 | optional: the 5 Hz language-model planner; cover, repaint, reference audio, the base and XL models | |

## The model, as measured from the checkpoint

All weights are bf16 safetensors (`tools/acestep_manifest.json` lists every tensor of the turbo checkpoint).

- **Text encoder**, Qwen3-Embedding-0.6B: 28 layers, 1024 wide. The caption uses its last hidden state; the lyrics use
  only its embedding table.
- **DiT** (turbo, 4.8 GB): 24 layers, 2048 wide, 16 query and 8 key/value heads of 128, sliding-window attention
  (128 frames) alternating with full attention, cross-attention to the condition tokens, adaLN modulation from two
  timestep embeddings, patch size 2 over latents of 64 channels at 25 Hz. 8 steps, no guidance pass.
- **VAE** (0.34 GB): Oobleck, 48 kHz stereo, 1920 samples per latent frame. Convolutions are weight-normalised,
  activations are Snake, upsampling is a transposed convolution.
- Upstream's sampler applies **DCW**, a Haar-wavelet correction at every step, by default for turbo. Without
  `pytorch_wavelets` (which needs `setuptools<72`) upstream silently skips it.

## Rebuilding the references

The reference is upstream's code, run on the CPU in float32 (the bf16 weights widen exactly, as the engine widens
them). `tools/requirements-acestep-ref.txt` pins the environment and names the upstream commit.

```sh
pip install -r c/tools/requirements-acestep-ref.txt
git clone https://github.com/ace-step/ACE-Step-1.5 up && git -C up checkout ca1e85fe9430179831e6bc6be790c332190a3866
export ACESTEP_UPSTREAM=$PWD/up PYTHONUTF8=1

cd c
python3 tools/make_acestep_tiny.py acestep_tiny          # 3.4 MB, the real tensor names, dtypes and ranks
python3 tools/acestep_ref.py capture acestep_tiny acestep_tiny_ref --check-determinism
python3 tools/acestep_ref.py capture ~/models/ace-step-1.5 ace_ref --cases inst10,long30      # the real checkpoint
```

`acestep_ref.py` never downloads and never rewrites the checkpoint directory (upstream's handler would do both: it
counts the 1.7B planner as part of the "main model"); it fingerprints the directory and fails if a file changed.
Sixteen cases are captured, each in its own directory (`ref.safetensors`, `ref.json`, `audio.wav`): DCW on and off,
with lyrics, an odd frame count, shift 3, a 5-step schedule, and a 30-second clip (eleven cases, which the DiT and VAE
oracles use), then set metas, other scripts and accents, a caption cut at 256 tokens, nothing at all, and a fractional
duration (five more, which only the text oracle needs; it runs on all sixteen). Read the DCW-off cases with care:
on the real checkpoint turbo without DCW often collapses to the silence latent (0.001 rms), which is upstream's
behaviour and so what the port reproduces, but it makes the final waveform a weak check there.

## The VAE oracle

```sh
make acestep
./acestep --model acestep_tiny --vae-ref acestep_tiny_ref/long30
./acestep --model ~/models/ace-step-1.5 --vae-ref ace_ref/long30 --wav out.wav
```

It decodes every VAE input the handler saw and compares with what the handler's VAE returned, then decodes the whole
clip the way the handler does (overlapping windows of 256 frames with 64 of context each side, the overlap discarded,
then the peak limit) and compares with the audio the handler returned. It fails on either the ratio of the error and
reference norms (above 2e-4) or the largest single error (above 5e-5, in a waveform that lives in [-1, 1]).

Measured: the real checkpoint matches to a relative 3.5e-6 to 4.8e-5 and an absolute 1.2e-5 at worst (the 30-second
clip, six windows); the tiny fixture to 1.3e-6 and 3.6e-6. Those are float32 rounding: against a float64 decode, the
C decoder is closer to the answer than torch's own float32 is. The tiny fixture is built to be well conditioned on
purpose (its residual branches are weak): left at full strength a random 15-unit residual stack with Snake
activations is chaotic and torch's float32 is itself 2e-4 from float64, so no oracle could be sharper than that.

`tools/mutate_acestep_vae.py` shows the oracle can fail: 20 one-line breakages of the decoder (a dilation, a padding,
a tap order, a dropped weight-norm gain, a trimmed window, the peak limit) are compiled and run, and all 20 are
caught. Two mutants are absent on purpose because they are equivalent: starting each window a frame later, or
halving the overlap, gives the same audio. The decoder's receptive field is 6 to 8 latent frames each side
(tiny fixture: overlap 6 is wrong by 8e-2, 8 right to 4e-5; real checkpoint: see below) and upstream's overlap is 64.

Cost: 61 GMAC (122 GFLOP) per second of audio, counted from the layer shapes. On a 24-core i9 a 10-second clip decodes
in 1.4 s (0.87 TFLOP/s) and a 30-second one in 8 s. The 30-second clip looks slower per FLOP only because upstream's
windows overlap: six windows of 192 to 256 frames decode 1390 frames to deliver 750, 85% extra. On the real
checkpoint the 30-second clip decoded with an overlap of 12, 16 or 32 has exactly the error of upstream's 64 (1.2e-5
at worst), 8 is still inside the limits (2.7e-5), and 6 is not (5e-3): the receptive field is 6 to 8 latent frames
each side. So an engine that does not need to reproduce upstream's windows could decode 1.5 times faster with an
overlap of 16; the default stays upstream's (256 and 64) until that is chosen on purpose. Peak memory on the real checkpoint is 1.2 GB: the weights are 334 MB as f32 (the weight-norm fold
must be done in f32; folding to bf16 would move every weight by up to 0.4%) and a 256-frame window's activations are
about 0.75 GB.

## The DiT oracle

```sh
./acestep --model acestep_tiny --dit-ref acestep_tiny_ref/inst10
./acestep --model ~/models/ace-step-1.5 --dit-ref ace_ref/long30 --dump out/long30     # f32 velocities and hidden states
```

Its input is what the handler fed the DiT (the captured text and lyric hidden states, the reference timbre latents, the
source latents and masks, the seed's noise), so it checks the DiT, its condition encoders and the sampler, not the text
encoder. In order, each against the handler's own tensor: the packed condition tokens (lyric, timbre, text); the
context; the timestep embedding and its six modulation vectors at every call; the RoPE table; every DiT call fed the
latents and timesteps the handler fed it; each layer **alone**, fed the handler's hidden state, modulation and RoPE
table (a case captured with `--taps`); the schedule; the sampler's update **alone** (`x - v dt`, the straight jump to
x0 on the last step, DCW) fed the handler's latents and velocity; and finally the chained sampler from the captured
noise. `acestep_ref.py capture` records what is needed; it does not rebuild upstream's tensors by hand.

The limits are not one number, and the reason is the measurement. Chained, a deep network turns independent float32
rounding into a larger difference, and a sampler loop turns that into a larger one again, so the pieces are held to
what each can honestly reach and the whole to what is left:

| compared | measured on the real checkpoint (11 cases) | held at (norm ratio, largest error / peak) |
|---|---|---|
| a layer alone | 1e-7 typical, 3.3e-7 worst, no growth with depth | 2e-6, 2e-5 |
| a DiT call | 1e-6 typical, 8.2e-6 worst (at t = 1, where the network is most sensitive) | 5e-5, 1e-4 |
| the sampler's update alone | 7e-8 (float32 arithmetic; DCW included) | 1e-6, 2e-6 |
| the chained sampler | 4e-7 to 2.2e-5 final; in `steps5_nodcw`, 7.8e-5 final and 1.8e-4 mid-run | 5e-4, 2e-3 |

The chained run needs the loose limit because its steps feed their rounding into the next: in `steps5_nodcw`, whose
steps are the longest, the difference grows 8e-6, 2.6e-5, 3.8e-5, 1.8e-4 over five steps, though every call and every
update in it is at the tight level. That is the sampler's dynamics, not an error in any operation. What was ruled out on
the way, by measurement: the timestep embedding (matches torch at 3e-7 even at t = 1), the RoPE tables (6e-8, one ulp),
`qi_gemm` (identical to torch's float32 matmul against float64, ratio 1.00 at every DiT shape), and the layers
themselves (above). I could not fully explain why torch's own chain accumulates less than ours (torch fp32 is 1e-6 from
float64 at the last layer, ours 5e-6) when every layer alone agrees to 1e-7; the numbers are what they are, and the
limits above are set from them.

End to end, from the handler's conditioning to a waveform in C (this DiT's latents through the C VAE), against the audio
the handler returned: the largest sample error is 8e-6 for `inst10`, 4e-5 for the case with the noisiest latents, and
9.7e-5 for the 30-second clip (9.7e-5 of a waveform normalised to peak 1.0 is -80 dBFS, about three steps of 16-bit
audio).

`tools/mutate_acestep_dit.py` shows the oracle can fail: 46 one-line breakages (the timbre frame, the window, RoPE, the
norms, grouped-query head mapping, AdaLN, the gates, SwiGLU, the timestep embedding, the patch order, the schedule, the
sampler, each part of DCW) are compiled and run against five cases, and all 46 are caught. Which case catches which is
the point of passing several: the odd-tail DCW mutant survives every even-length clip and is caught by `odd9`;
`sampler-r-zero` (the network is told `r = 0`) is invisible to every teacher-forced call, which use the captured `r`, and
is caught only by the chained run.

Cost on the real checkpoint (24-core i9): a DiT call is 1.0 to 1.5 s for a 10-second clip (125 tokens), 4 s for 30
seconds (375), so 8 steps take 8 to 12 s and 32 s, against 5.9 s for torch's float32 on the same CPU for the 10-second
clip: about 1.5 to 2 times slower, which the small M of these products (125 rows) makes plausible but which has not
been tuned. The weights stay bf16 (4.4 GB resident, 4.2 GB peak for the whole oracle); `qi_gemm` widens them inside the
product, which is exact. The condition (both encoders, 76 tokens) takes 1.1 s.

## The text encoder oracle

```sh
python3 tools/make_acestep_tok_ref.py acestep_tiny tok_ref.json          # 118 strings through upstream's tokenizer
./acestep --model acestep_tiny --tok-ref tok_ref.json
./acestep --model acestep_tiny --text-ref acestep_tiny_ref/cut10
./acestep --model ~/models/ace-step-1.5 --text-ref ace_ref/meta10
```

`--tok-ref` holds the tokenizer to `AutoTokenizer(text, truncation=True, max_length=N)` on strings chosen to break a
port: other scripts and emoji, accents in both Unicode forms, digits, whitespace runs, contractions, text that looks like
a special token, and texts cut at the 256 and 2048 limits. `--text-ref` builds the caption prompt (instruction, caption,
metas) and the lyric prompt (language, lyrics) from the case's parameters and compares, in order: the prompt token ids
(exactly), the embedding rows, each of the 28 layers **alone** fed torch's own residual stream and RoPE table, and the
whole encoder chained, caption and lyric, against the handler's hidden states.

What upstream does that is easy to miss, each of which a mutant below breaks:

- the tokenizer NFC-normalizes first, splits digits one at a time, and treats a literal `<|endoftext|>` in the text as one
  token;
- its post-processor appends one more `<|endoftext|>` to every sequence, so a prompt that already ends with one ends with
  two, and truncation (256 for the caption, 2048 for the lyrics) keeps the first N-1 tokens to leave room for it;
- the caption's last hidden state is taken after the final norm, with causal attention; the lyrics skip the transformer
  and use only the embedding rows;
- the duration in the metas is `int(seconds)`, not rounded (`frac96` is 9.6 s), and every missing meta reads `N/A`.

On the real checkpoint all 16 cases have identical ids and exactly identical embedding rows. The limits again follow the
measurements:

| compared | measured on the real checkpoint (16 cases) | held at (norm ratio, largest error / peak) |
|---|---|---|
| a layer alone | 5.8e-9 to 9.2e-7 (the largest is layer 27), no growth with depth | 5e-6, 5e-6 of the larger peak going in or out |
| the whole encoder, 28 layers chained | 1.7e-6 to 2.1e-6 typical; 1.1e-5 worst (`cut10`, 256 tokens) | 5e-5, 2e-4 |

The absolute limit of a layer is a fraction of the larger of its input's and its output's peak, not of its output's alone,
because Qwen3-Embedding has **massive activations**: layers 7 to 26 carry a few channels of about 6000 (against a typical
value near 1), and layer 27 collapses them to 247. Its output is small, but its arithmetic is done on 6000-sized values,
so its rounding is 3e-3 absolute: 4.8e-7 of its input's scale and 1.2e-5 of its own output's. Holding it to its output
would have asked for more than float32 gives; the relative norm ratio (8.8e-7) never needed the correction.

`tools/mutate_acestep_text.py` shows both oracles can fail: 26 one-line breakages (a blank line, the trailing newline or
the `<|endoftext|>` of the caption or the lyric prompt, the order and the defaults of the metas, a rounded duration,
dropped bpm or key, the appended token, truncation one short, no NFC, a wrong special token id, a wrong embedding row,
bidirectional attention, RoPE theta, the final norm, the query or key norm, unrotated keys, a swapped SwiGLU, an epsilon,
the wrong norm weight, a dropped residual) are compiled and run against the tokenizer reference and seven cases. All 26
are caught, on the tiny fixture and on the real checkpoint. The duration mutant needs `frac96`, the bpm and key
mutants need `meta10`, the NFC mutant needs strings with decomposed accents, and the truncation mutant needs a string
cut at its limit, which is why several cases are passed.

Cost: the caption's encoder is 0.6 s for 60 tokens and 1.5 s for 256 on the real checkpoint (24-core i9); the lyric
path is a table lookup.

## The engine

```sh
./acestep --model ~/models/ace-step-1.5 --caption "calm piano with soft strings" --duration 20 --out clip.wav
./acestep --model ~/models/ace-step-1.5 --caption "upbeat pop" --lyrics "[Verse]
walking down the street" --language en --seed 7 --out song.wav
./acestep --model ~/models/ace-step-1.5 --serve          # the gateway's way in
./acestep --model acestep_tiny --gen-ref acestep_tiny_ref/inst10    # the oracle
```

`acestep_gen.h` joins the three networks as upstream's handler does for text-to-music with the turbo DiT and no planner:
prompts, token ids, the caption's hidden states and the lyrics' embedding rows, the condition encoders (lyric, timbre, text),
`steps` Euler steps with DCW, the tiled VAE decode, the peak limit. Defaults: instrumental, 30 s, 8 steps, shift 1, DCW on.
Every weight stays in memory for the life of the process (6.4 GB: the DiT 4.8, the text encoder 1.2, the VAE 0.3).

What upstream does that a port would get wrong, each held by the oracle:

- **The seed is torch's.** The noise comes from `acg_randn`, a Mersenne Twister and Box-Muller on blocks of 16 exactly as
  ATen's CPU kernel makes `torch.randn(..., generator=Generator().manual_seed(seed))`, so a seed gives the clip upstream's
  own handler gives for it. Measured against the handler's noise: 1.8e-7 relative (a few float32 ulps; ATen's AVX2 build
  uses Sleef's `log`, `cos` and `sin`, this uses libm's). A seed is 0 to 2^32 - 1.
- **The clip's length** goes through three rules: the duration is rounded to one decimal first (9.07 s is 9.1 s, 227
  frames, not 226), the frames are the whole 1920-sample hops in it (9.1 s is 227.5, so 227), and a clip under 128 frames
  is run on 128 (5.0 s is 125). Each has its own captured case (`round907`, `floor91`, `short5`).
- **The source latents and the timbre reference are the silence latent** (`silence_latent.pt`, a torch file: a stored zip
  holding a float32 `[1][64][N]` tensor, read here without torch and transposed to `[N][64]`). The first `frames` of it are
  the source, the first 750 the timbre reference, and the mask is all true.
- **Lyrics are used as given.** No lyrics at all means `[Instrumental]`; an empty string is an empty lyric, which upstream
  also does and which makes different music (`empty10`).
- One difference on purpose: a clip asked for under 5.12 s is made on its 128 frames and then cut to the length asked, where
  the handler returns all 128.

The protocol (`acestep_serve.h`, the same shape as the image engine's): `READY {...}` once with the limits; `GEN {id, caption,
lyrics, language, duration, steps, shift, seed, bpm, key, timesig, dcw}`; `PROGRESS {id, stage, step, steps}` for the stages
`text`, `condition`, `denoise` (one per step) and `decode` (one per VAE window); `AUDIO {id, ..., bytes}` followed by that
many bytes of 16-bit little-endian interleaved PCM at 48 kHz and a newline; `ERROR {id, message}`, after which the engine is
ready for the next request; `CANCEL {id}`, honoured between stages and between sampler steps and VAE windows, answered
`ERROR "cancelled"`. A `GEN` that arrives while one runs is kept and run after it. Limits: duration 1 to 600 s, 1 to 20 steps,
caption 4096 bytes, lyrics 32768 bytes (the prompts are then cut at the handler's 256 and 2048 tokens, and the result says
`truncated`).

`--gen-ref` runs a captured case's own parameters through all of it and compares: the silence latent (exactly), the source
latents and timbre reference cut from it (exactly), the noise, the length asked, and the audio.

| compared | measured on the real checkpoint (19 cases) | held at |
|---|---|---|
| the silence latent and its slices | identical | 1e-9 |
| the noise from the seed | 1.8e-7 relative, 1.7e-6 absolute | 1e-6, 1e-6 |
| the audio, caption to waveform | norm ratio 8.2e-6 to 1.6e-4 (the worst is `steps5_nodcw`); largest sample 2e-7 to 4.8e-4 (the 30 s clip) | 5e-4, 2e-3 |

The audio is what the whole chain leaves: the text encoder's 1e-5, eight sampler steps and the decoder each pass on what they
were given, so it is looser than any piece (the DiT with the handler's own conditioning gave 9.7e-5 on the same clip). 2e-3 is
-54 dBFS at the single worst sample of a waveform in [-1, 1]; the norm ratio is the bound on the whole. A wrong prompt, clip
length, seed or schedule is wrong by 0.1 to 1.

`tools/mutate_acestep_gen.py` shows the oracle can fail: 34 one-line breakages (each part of the generator, the
transpose and offset of the silence latent, the timbre length, each clip-length rule, the mask, the lyric default, the
metas, the shift, DCW, the arguments the network is called with, the peak limit, the VAE overlap) are compiled and run against ten
cases and all are caught. Two are deliberately absent because they are equivalent: the generator's tail branch (a clip's noise
is a multiple of 16 values, so it never runs) and a VAE overlap of 12 or more.

`tests/test_acestep_serve.py` holds the protocol itself (the frame sizes, the same seed giving the same bytes and another
not, every refusal naming its reason and leaving the engine alive, a CANCEL behind its GEN and on the first denoise frame,
queued requests answered in order, a line over 1 MiB refused by its id, end of input a clean exit, and the one-shot command
line making the same samples as the protocol); `tests/test_acestep_gen_load.py` the refusals (every way the silence latent
can be wrong, a missing component, a latent width the VAE does not share, a reference without its parameters, the command line).

Cost on the real checkpoint (24-core i9, 10 s of music): text 0.7 s, condition 1.1 s, eight steps 12 s, decode 2 s, about 16 s
in all after a 5 s load; 30 s of music is about 50 s.

## In colibri

The checkpoint is Hugging Face `ACE-Step/Ace-Step1.5`; `coli setup` offers it (6.4 GB: the turbo DiT, the text
encoder and the VAE; the repository's 5 Hz planners are not downloaded, since this engine does not use them). The
family is `ace_step`, modality `audio`, found by `"model_type": "acestep"` in the checkpoint's root `config.json`.

```sh
coli music "calm piano with soft strings" --duration 20            # ~/colibri-music/<date>-<caption>-s<seed>.wav
coli music "upbeat pop" --lyrics @words.txt --language en --bpm 120 --key "C Major" --seed 7 --out song.wav
coli plan --model ~/models/ace-step-1.5       # the three weight sets, all resident, against the RAM free now
coli info --model ~/models/ace-step-1.5       # what is on disk and what the engine will hold
coli doctor --model ~/models/ace-step-1.5
coli serve --model ~/models/ace-step-1.5      # POST /v1/audio/music
```

`coli run "prompt"` on a music model does the same as `coli music`; `coli chat` and `coli tune` refuse it with a
pointer (there is no conversation and no decode loop to tune); `--ctx` does not apply. `coli music` prints the song's
path as its last line of stdout, so a script can read it; progress goes to the terminal's line (stderr when piped).
Ctrl-C asks the engine to stop at its next step and waits for it; a second one quits.

**The route.** `POST /v1/audio/music` takes `model`, `prompt` (the caption; `caption` is the same), `lyrics` (none or empty
is instrumental), `language`, `duration` (1 to 600 s, default 30), `steps` (1 to 20, default 8), `shift` (default 1),
`seed` (default random, 0 to 4294967295), `bpm`, `key`, `time_signature`, `dcw`, `stream`, `response_format`. The answer has
the images route's shape: `{"created", "data": [{"b64_json": <a WAV>, "format": "wav"}], "colibri": {sample_rate, channels,
samples, duration, seed, steps, shift, frames, truncated, peak, caption_tokens, lyric_tokens, timings}}`; with
`response_format: "wav"` the body is the WAV itself (`audio/wav`, the seed in `x-colibri-seed`); with `stream: true` it is
SSE: `music_generation.progress` events (`stage`: `text`, `condition`, `denoise`, `decode`; `step`, `steps`), then
`music_generation.completed` with the same payload, then `[DONE]`. A bad field is a 400 naming it (`param`); a full queue
is a 503 (a song takes minutes, so 429's invitation to retry at once would only refill the queue); a client that hangs up
cancels the song at the engine's next step. `GET /v1/models` lists `"capabilities": ["music_generation"]` and an `audio`
block with the limits above, so a client can build a valid request. The text endpoints and the images routes answer a music
model with a 400 that points at this one.

Measured through the real gateway on the real checkpoint: a 6 s clip as JSON and as a raw WAV for the same seed are the
same bytes, and the same as `coli music` and `acestep --caption` make for it.

**Memory.** Every weight stays loaded between songs: 6.9 GB peak resident set for a 10 s clip, 6.95 GB for 60 s (the
weights dominate; the working buffers are a few hundred MB at most). Measured on Windows with `PeakWorkingSet64`.

**Speed.** The engine uses every core, so another busy process slows it a lot: the same 10 s clip took 16 s on an idle
24-core i9 and 68 to 75 s while a Qwen-Image server was drawing on the same machine. Quote the first.

**What is tested without the weights.** `tools/acestep_stub.py` speaks the protocol and makes a synthetic clip that
depends on the request, the way `qwenimage_stub.py` does for pictures. `tests/test_acestep_engine_client.py` holds
`music_engine.py` (the client, the request rules, WAV, saving, the plan) to it, `tests/test_acestep_gateway.py` holds
the route (real `APIServer`, real `MusicEngine`, the stub behind them: the framing, determinism, every refusal, SSE, the
raw WAV, hang-up cancellation, queueing, the model card), and `tests/test_acestep_cli.py` the commands as a user types
them. The real engine is held to the same protocol by `tests/test_acestep_serve.py`.

CI (`acestep-tiny-check`) rebuilds the fixture and the references from upstream at the pinned commit, runs the oracles,
the refusal tests (`tests/test_acestep_{vae,dit,text,gen}_load.py`, `tests/test_acestep_serve.py`), the four mutation tools, and the oracles again under ASan
and UBSan.
