# ACE-Step 1.5 in C

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) turns a caption and lyrics into a song: 48 kHz stereo, 10
seconds to 10 minutes, from a diffusion transformer over 25 Hz latents and an audio VAE. This directory of work ports
it to the same shape as [Qwen-Image](qwen-image.md): one C file, weights read as they are, every stage held to the
reference by an oracle that compares tensors, not just the final audio.

**Using it:** `coli music "calm piano with soft strings"` writes a song, `coli serve` answers `POST /v1/audio/music`
(see [In colibri](#in-colibri)). The engine itself is `acestep --model DIR --caption "calm piano" --out clip.wav`, and
every part of it is held to upstream's handler by an oracle. `coli web` has a player for it.

| stage | what | state |
|---|---|---|
| 0 | the reference: upstream's own handler, observed at every stage; a tiny random checkpoint of the real structure | done |
| 1 | the VAE decoder, `c/acestep_vae.h`: latents to 48 kHz stereo, tiled as upstream tiles | done, oracle-checked |
| 2 | the DiT, its condition encoders and the turbo sampler (DCW included), `c/acestep_dit.h` | done, oracle-checked |
| 3 | the text encoder (Qwen3-Embedding-0.6B), its tokenizer and the prompt templates, `c/acestep_text.h` | done, oracle-checked |
| 4a | the engine proper in C: a request in, audio out; one-shot and `--serve`; `c/acestep_gen.h`, `c/acestep_serve.h` | done, oracle-checked |
| 4b | around it: the registry's audio modality, `music_engine.py`, `POST /v1/audio/music`, `coli music`, `coli info` / `plan` / `doctor`, `coli setup`, `make install` | done, tested against a stub engine and smoke-tested on the real one |
| 4c | the dashboard's music player: `web/src/components/MusicTurn.tsx`, `web/src/lib/useMusic.ts` | done, tested in a real browser |
| 5 | audio in: WAV and MP3 loading, the VAE encoder, the audio tokenizer and detokenizer; a reference clip, a cover, a repaint | done, oracle-checked ([below](#audio-in-a-reference-clip-a-cover-a-repaint)) |
| 6 | optional, not built: the 5 Hz language-model planner and the audio codes it makes; lego, extract and complete (they need the base or SFT checkpoints); the XL models | |

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

## Audio in: a reference clip, a cover, a repaint

Three things the checkpoint does with a clip you give it. Upstream calls them reference audio, the `cover` task and the `repaint`
task; the turbo model does all three. WAV and MP3 are read (WAV: PCM 8, 16, 24 and 32 bit, float 32 and 64, any rate and
channel count; MP3: MPEG-1, 2 and 2.5 layer III, with or without tags), up to 10 minutes, decoded from the bytes (never from the
extension) by `c/acestep_audio.h` and made 48 kHz stereo the way upstream's `_normalize_audio_to_stereo_48k` does it, with
torchaudio's own resampler ported (sinc interpolation, a Hann window, width 6, roll-off 0.99).

- **Reference.** The timbre encoder is given the clip's latents where text to music gives it the silence latent's first
  750 frames (30 s). A clip shorter than 30 s is repeated to reach it; a longer one is cut in thirds and 10 s are taken from
  each, joined. Works with every task. The clip must not be silent.
- **Cover.** The source is encoded to latents, and what conditions the DiT is not those latents but what survives a trip through
  a 5 Hz bottleneck: the audio tokenizer (`tokenizer.*` in the DiT's checkpoint: an attention pooler over windows of 5 frames, then
  a six-dimensional finite scalar quantizer with levels 8, 8, 8, 5, 5, 5) and the detokenizer back to 25 Hz (`acestep_tok.h`).
  So a cover keeps a song's musical structure, not its exact sound, and the description decides the rest. The clip's length is
  the result's, the prompt's instruction is upstream's cover one. `cover_strength` (0 to 1, default 1) is the share of the steps that
  follow the source; the remaining steps run as text to music, on the text-to-music instruction (upstream builds this prompt
  at the cover prompt's token count, which cuts its last token, and so does the engine). `cover_noise` (0 to 1, default 0)
  starts the sampler from the source renoised to that level instead of from noise, which keeps it nearer the original.
- **Repaint.** A span of the source, `start` to `end` seconds, is generated again and the rest is kept: the DiT is conditioned on
  the source with the span's frames replaced by the silence latent, and during the first part of the steps the source,
  noised to the level of the step, is put back outside the span. The span's edges are crossfaded in latent space, and after
  the decode the source's own samples are spliced back outside it with a short crossfade (so the unchanged part is the original,
  not its VAE reconstruction). `mode` balances all this: `balanced` (default, with a `strength` of 0 to 1: 0.5 is the default and
  higher lets the span change more), `conservative` (the source is put back for every step, longer crossfades) and `aggressive`
  (nothing is put back and nothing spliced: the span is free and the rest is only reconstructed). A start before 0 adds that much
  silence before the clip and an end past its length adds silence after it, and music is made there: that is how a clip is extended.

What each is held to. Every number below is the engine against upstream's own handler on the real checkpoint, the reference
run on the CPU in float32 (`tools/acestep_ref_audio.py`), and each oracle has mutants (`tools/mutate_acestep_*.py`) that prove it can fail.

| oracle | what it compares | measured | held at | mutants |
|---|---|---|---|---|
| `--vae-enc-ref` | the VAE encoder's parameters and the tiled, sampled latents of 7 clips (one pass, an odd length, a clip of exactly one chunk, windows with a tail, the 30 s window) | 6e-6 relative one pass, 3e-5 tiled | 2e-5, 1e-4 | 29 of 29 |
| `--cover-ref` | the pooler, the quantizer's codes, the detokenizer, each fed upstream's input, and the whole path (5 clips) | 1.4e-6, codes identical in 261 windows, 2.3e-7, 2.2e-7 | 5e-6, exact, 1e-6, 1e-6 | 24 of 24 |
| `--gen-ref` | 20 whole generations: every intermediate (noise, timbre, source and clean latents, the cover and non-cover contexts, the final latents) and the audio | audio 4e-6 to 1.2e-4 of the norm, 4.0e-4 at worst (a repaint that adds silence) | 1.5e-3 | 45 of 45 |
| `tests/test_acestep_audio.py` | WAV and MP3 loading and the resampler, 26 fixtures against upstream's own code | bit-exact WAV; 2.4e-7 resampled, 1.4e-6 MP3 | 0, 6e-7, 2e-5 | 70 of 70 |

The VAE encoder's float32 is closer to the float64 network than torch's float32 is (1.7e-6 against 5.2e-6 on a 10 s clip), so what
the oracle sees is torch's rounding, not the engine's.

What the engine does differently from upstream, each on purpose:

- **The posterior draw.** Upstream's `latent_dist.sample()` takes torch's global generator, which nothing seeds: two runs on one
  clip differ by the posterior's width (about 1e-2 of a latent). The engine draws from torch's generator seeded with the request's seed
  xor `0x5EED0001`, source first, then reference, so a seed means one song; the oracle gives upstream the same draw.
- **The reference's 10 s segments** start in the middle of their third's range, where upstream picks a random start. A clip of exactly
  30 s, or one that leaves no range, has no choice to make.
- **The encoder's window** is 30 s with 2 s of context (what upstream picks on a GPU with more than 8 GB; with no GPU it picks 15 s, a choice
  about memory), so a 30 s reference is one pass.
- **A clip under 128 frames** (5.12 s) is made on 128 and cut to its length, as in text to music.
- **A repaint that ends before it starts** is refused; upstream quietly regenerates the whole clip.

The tasks that need the base or SFT checkpoints (lego, extract, complete) and the 5 Hz planner (the metadata, then the codes) are in the next sections.

Cost, 24-core i9, a 10 s clip: reading and encoding the source 1.9 s, the tokenizer and detokenizer 2 s on top of the condition
for a cover, so a cover is about 16 s against 11 s for text to music; a 30 s reference adds 5 s. The encoder (0.4 GB) and the
tokenizer (0.4 GB) are loaded the first time a request needs them, and stay.

Reproduce:

```sh
cd c && make acestep
python3 tools/acestep_ref_audio.py enc ~/models/ace-step-1.5 enc_ref wavs                # wavs/: a 10 s and a 30 s stereo 48 kHz wav
python3 tools/acestep_ref_audio.py tasks ~/models/ace-step-1.5 task_ref wavs             # reference, cover and repaint cases
./acestep --model ~/models/ace-step-1.5 --vae-enc-ref enc_ref/tile_small
./acestep --model ~/models/ace-step-1.5 --cover-ref task_ref/cover10
./acestep --model ~/models/ace-step-1.5 --gen-ref task_ref/repaint_mid
python3 tools/mutate_acestep_venc.py ~/models/ace-step-1.5 enc_ref/single10 enc_ref/tile_small ...
python3 tools/mutate_acestep_tok.py ~/models/ace-step-1.5 task_ref/cover10 task_ref/cover_odd task_ref/cover_pad4
python3 tools/mutate_acestep_audioin.py ~/models/ace-step-1.5 task_ref/cover10 task_ref/repaint_mid ...
```

## The base and SFT DiTs: guidance, extract, lego, complete

The turbo DiT has classifier-free guidance distilled into it: eight steps, one call each. The checkpoint's other DiTs, `acestep-v15-base`
and `acestep-v15-sft` (4.8 GB each, the same network with the same tensor names and shapes, `is_turbo` false in the config), are
guided when they sample: every step evaluates the model twice, on the condition and on the null condition, and combines the two velocities.
They take 30 to 100 steps and do three tasks the turbo one was not trained for:

| task | what it makes | the instruction it is given |
|---|---|---|
| `extract` | the one track (`drums`, `bass`, `vocals` ...) out of the source clip | `Extract the DRUMS track from the audio:` (without a track: `Extract the track from the audio:`) |
| `lego` | that track, added to the source inside a span (`repainting_start` to `repainting_end`), the rest of the source kept | `Generate the BASS track based on the audio context:` |
| `complete` | a partial track completed with other tracks (`complete_track_classes`); the clip is optional, without one the song is made from the description | `Complete the input track with DRUMS \| BASS:` |

The tracks are `woodwinds`, `brass`, `fx`, `synth`, `strings`, `percussion`, `keyboard`, `guitar`, `bass`, `drums`, `backing_vocals`
and `vocals`. Extract and complete hand the DiT the source's latents whole as its context; a lego is a repaint that **shows** the span
instead of hiding it (a repaint puts the silence latent there), with the same injection, crossfade and waveform splice. The prompt's
duration is always the source's, whatever `duration` says (upstream's `prepare_batch_data`).

**Installing.** The engine opens `acestep-v15-turbo` and loads any other DiT the first time a request names it, so the base DiT goes beside the
turbo one in the checkpoint: `acestep-v15-base/` with `config.json`, `model.safetensors` and `silence_latent.pt` from
`ACE-Step/acestep-v15-base`: `coli model install ace-step-1.5-base-dit` (an add-on of the catalog, [model install](model-install.md#from-the-terminal); the dashboard's
install list has it too) or by hand with `huggingface-cli download ACE-Step/acestep-v15-base config.json model.safetensors silence_latent.pt --local-dir
<model>/acestep-v15-base`. The engine never reads the Python files in that repository. The SFT DiT (`ACE-Step/acestep-v15-sft`, the same three files) is not in the catalog, and goes beside the others the same way.
**The SFT DiT on real weights** (2026-11-23): all 19 guided cases of `tools/acestep_ref_audio.py tasks --dit acestep-v15-sft` (the base's list: guidance variants, extract, lego, complete, cover, repaint) captured from
upstream's handler (the code beside the weights has to be upstream's at the pinned commit: copy `acestep/models/sft/*.py` there first) and held by `acestep --gen-ref DIR --dit sft`: all 19 within their limits on the CPU
(the audio 5e-6 to 5.6e-5 of the norm from upstream's, the repaint 1.05e-3 against its limit of 2e-3) and on the card (the same, the `dcw` case 1.5e-4 against 5e-4).

**Choosing.** A request names `dit` (`turbo`, `base` or `sft`); extract, lego and complete run on the base DiT (the SFT if there is no
base) when it does not, and everything else on the turbo one. A guided DiT's defaults are upstream's: 32 steps (up to 200; the turbo takes
at most 20), guidance 7, no DCW, `shift` 1. `READY` says what is there (`dits`, `guided`, and the three tasks under `tasks`), and a
gateway or `coli music` refuses the tasks, by name and with the remedy, when the checkpoint has no guided DiT.

**Guidance** is upstream's `apg_guidance.py`. APG (the default): the difference of the two velocities is smoothed over the steps by a
momentum of -0.75, cut to a norm of 2.5 along the frames, and only its part orthogonal to the conditional velocity is added, scaled by
guidance - 1. `use_adg` selects ADG, the angle-based form. `cfg_interval_start` and `cfg_interval_end` limit it to a part of the schedule
(t from 1 down to 0; outside it the velocity is the conditional one alone). A guidance of 1 or less is off, and then the null branch is not
evaluated. The schedule is `linspace(1, 0, steps + 1)` bent by `shift`, so its last step goes from t to 0. Not built: the Heun sampler,
the SDE sampler, velocity-norm clamping and EMA smoothing (upstream's options, off by default), and flow-edit.

**What is held to upstream's handler, and how.** `tools/acestep_ref_audio.py tasks MODEL_DIR OUT WAVDIR --dit acestep-v15-base` captures 19
cases from upstream's own handler (CPU, float32): guidance on and off, the interval, ADG, DCW, shift 3, 40 steps, a cover (with and without
the last steps on the text-to-music prompt, which then has its own null condition), a repaint, extract (a track, none, vocals of a longer
clip), lego (a span, to the end, no track) and complete (tracks, none, and a clip shorter than the duration). `acestep --dit base
--gen-ref CASE` runs each through the engine and compares the noise, the contexts and the final latents and the audio.

| | the real base checkpoint | the tiny fixture (`make_acestep_tiny.py --audio-in --base`) |
|---|---|---|
| text to music (7 cases) | latents 2.5e-6 to 5.7e-5 of the norm, audio 7e-6 to 4.4e-5 | 4e-7 to 1.4e-6, audio 7e-7 to 1.6e-6 |
| with audio in (12 cases) | latents 1e-5 to 1.9e-3, audio 1.4e-5 to 1.2e-3 of the norm | 1.4e-6 to 1.8e-6, audio 8e-7 to 1.9e-6 |

The audio-in cases on the real checkpoint are worse for one reason that is shown, not assumed: the source's latents come from our VAE encoder, whose float32 differs
from torch's by 4e-5, and guidance carries that through twelve steps. Fed the handler's own source latents (`ACESTEP_FEED_LATENTS=1`) the same
cases are within 1e-5 of the norm and 3e-4 at one value; on the tiny fixture, whose encoder is exact, the whole chain is at 1.5e-6. The guided audio-in limit
(latents 4e-3, 1e-1 at one value) is therefore the encoder's noise, amplified, with the margin the other oracles keep, and a wrong context, mask, guidance
or schedule is wrong by 1e-1 to 1.

`tools/mutate_acestep_guided.py` breaks the code one line at a time (APG's momentum, its norm cut and its projection, ADG's angle and clip, the schedule's
shift and length, the null condition, the interval, the guidance default, a lego hiding its span, an extract with a silent context, each instruction and the
track's case) and 29 of its 30 mutants are caught on the tiny fixture (the 30th, keeping the first prompt's null condition after the cover switch, needs a cover and a text instruction that tokenize to different lengths, as the real tokenizer's do and the tiny one's do not). The hub's `modeling_acestep_v15_base.py` is older than upstream's and spells the
guidance argument `diffusion_guidance_sale`, so a capture made with it ignores `guidance_scale` and runs every case at 7: the capture tool refuses a checkpoint
whose modeling code is not upstream's at the pinned commit.

Cost on the real checkpoint (24-core i9), a 10 s clip: 2.5 s a step (two DiT calls), so 32 steps are 80 s of denoising and about 90 s in all, against 12 s for
the turbo's eight; the base DiT adds 4.8 GB to the resident set once a request names it. On a Vulkan card (below) the base DiT's 32 steps are 5.7 s and the clip 8.5 s in all.

```sh
cd c && make acestep
./acestep --model ~/models/ace-step-1.5 --dit base --caption "bright ukulele" --duration 10 --out base.wav               # 32 steps, guidance 7
./acestep --model ~/models/ace-step-1.5 --extract song.wav --track drums --out drums.wav                                 # the base DiT is chosen
./acestep --model ~/models/ace-step-1.5 --lego song.wav --track bass --from 8 --to 20 --out song-bass.wav
./acestep --model ~/models/ace-step-1.5 --complete song.wav --tracks drums,bass --out full.wav
coli music --extract song.wav --track drums                       # the same through the gateway's rules
python3 tools/acestep_ref_audio.py tasks ~/models/ace-step-1.5 base_ref wavs --dit acestep-v15-base
./acestep --model ~/models/ace-step-1.5 --dit base --gen-ref base_ref/b_lego
python3 tools/mutate_acestep_guided.py ~/models/ace-step-1.5 base_ref/b_t2m base_ref/b_lego ...
```

## The DiT on the card

Built with `VK=1` and started with `COLI_VULKAN=1`, the DiT's denoising steps run on the graphics card (`c/acestep_dit_vk.h`; `acd_forward` tries it first). The 24 layers of a
step are one submission: the matrices (bf16, f32 arithmetic) go up once, when the first step runs (4.4 GB for the turbo DiT; the base one when a request names it), and a step
uploads the patch embedding and the step's modulation, and brings back the hidden rows. The patch embedding, the output norm and the output projection are small and stay on the CPU, as do
the text encoder and the condition encoders and the VAE (each has its own card path, in the sections below). The ops are the chain's, shared with the planner and the image models: the adaptive norm is `vkc_norm` with the weight times
`(1 + scale)` made on the host each step and the shift added onto a buffer of ones, the gates are `GATE_ADD`'s per-column factor, the self-attention's K and V projections go up with their rows
repeated for each group of query heads (16 over 8: the attention shader reads full-width K and V) and the cross-attention's cached K and V are repeated the same way when a condition goes up (the condition
and the null condition of a guided DiT are two slots), and the attention is the encoder shader's (a range of rows a query: the sliding window is `[i - 128, i + 128]`, the full layers and the cross-attention
`[0, n - 1]`).

On the 5090 a step of the 10 s clip is 0.1 s (1.5 s on the CPU) and the turbo's eight steps 0.8 s (11.8 s); a whole 10 s song is 4.3 s (text 0.5, condition 0.9, denoise 1.6, decode 1.4), a 10 s
song with the planner 13 s (it was 48), the base DiT's 32 guided steps 5.7 s, and a 2-minute clip's 8 steps 6.2 s, of which the decode on the CPU is then 30 s. Held to upstream by the same oracles as the CPU's
with COLI_VULKAN=1 on the real checkpoint: `--dit-ref` (a call 1e-6 of the norm, as the CPU; the chain of 8 steps 1.1e-4 at one value against the CPU's 9.8e-5), `--gen-ref` on all 11 turbo cases
(the 30-second clip exercises the sliding window, the 9-second one an odd length) and all 19 base cases (the audio-in ones within the limits the CPU's are). One case is wider on the card: the base DiT's 40 guided
steps end 1.9e-4 of the norm from upstream's latents (the CPU's 5.7e-5; the audio 1.7e-4, limit 5e-4), a different rounding carried through forty steps of seven-fold guidance, so the oracle's latents limit
is 5 times wider when the card ran the DiT. `COLI_VK_DIT=0` keeps the DiT on the CPU. A card that is missing, a matrix that will not go up or a step that fails leaves that step to the CPU (a step has no state,
so nothing is repeated: `ACESTEP_DIT_VK_FAULT=N` makes step N fail, once, to test it). `tests/test_acestep_dit_vk.py` holds the card to the CPU on the tiny guided fixture (turbo and base, a clip past
the window, an odd length, a failing step, the switch), skipped without a VK build and a device.

## The VAE on the card

The last long step of a song was the VAE decode on the CPU (1.5 s for 10 s of music, 31 s for two minutes). Built with `VK=1` and started with `COLI_VULKAN=1`, each window of the
tiled decode (up to 256 frames, upstream's tile) is one submission on the card (`c/acestep_vae_vk.h`; `acv_decode_ws` tries it first). The weights stay f32 (the fold of the weight
norm is the CPU's, done at load: 0.33 GB on the card) and the ops are the chain's, the Qwen3-TTS codec's among them (`chain_codec.comp`): a convolution is an im2col tile (mode 6, new: the taps of a
symmetric, dilated convolution, zero outside the window) into the chain's GEMM and the bias (mode 3); Snake is mode 1 (the argument reduced to [-pi, pi] first, as the codec's); a transposed
convolution is one GEMM to `[T][2s * cout]` and mode 7 (new), a gather-add in which each output row takes the two taps the CPU's takes, in the same order; a residual unit's sum is the chain's `ADD`. A window of 256
frames is 1.1 GB of activations; a window that would not fit in a 4 GB binding is the CPU's.

On the 5090 a window of 256 frames decodes in 0.1 s (1.5 s on the CPU), a 10 s clip in 0.2 s and a 2-minute one in 1.7 s (30.8 s). A whole 10 s song is 3.3 s (text 0.5, condition 0.9, denoise 1.7, decode 0.2)
against 5.1 s with the decode on the CPU, and a 2-minute song 10.4 s against 39.4 s. Held to upstream by the oracle the CPU is, `acestep --vae-ref` with COLI_VULKAN=1 on the real checkpoint, all 11 captured cases
(every window of the 30-second clip, and its tiled whole): 3 to 6e-6 of the norm from upstream's audio on the 10 s clips (the CPU's 3e-6 to 6e-6; the limit is 2e-4), 3.4e-6 on the tiled 30 s one (the CPU's 2.6e-6),
and a whole song differs from the CPU's by at most one step of the 16-bit sample. `COLI_VK_VAE=0` keeps the decode on the CPU. A card that is missing, a matrix that will not go up or a window that
fails leaves that window to the CPU (a window has no state: `ACESTEP_VAE_VK_FAULT=N` makes window N fail, once, to test it), and a failure that loses the device takes the card off for the rest of the process.
`tests/test_acestep_vae_vk.py` holds the card to the CPU on the tiny fixture with the DiT and the planner left on the CPU in both (a song, a 30 s clip of six windows, an odd length, a failing window,
the switch), skipped without a VK build and a device. The condition encoders and the text encoder, the rest of a song's time, have their own path (the next section).

## The text and condition encoders on the card

What was left of a song on the CPU was the encoders: Qwen3-Embedding's text encoder (0.4 s for a caption of 60 tokens) and the DiT's condition encoders, the **timbre** encoder (0.65 s: 4 layers over 750 frames of the
reference clip, which for plain text to music is the fixed silence latent) and the **lyric** encoder (0.13 s: 8 layers). Built with `VK=1` and started with `COLI_VULKAN=1`, each of the three stacks of layers
is one submission on the card (`c/acestep_enc_vk.h`; `acd_encoder` and `act_encode` try it first). They are one plain layer in three sizes: RMSNorm, grouped-query attention with a per-head norm on q and k and
RoPE, RMSNorm, SwiGLU, each with its residual, no modulation. The ops are the DiT's: the matrices (bf16, f32 arithmetic) go up once, when a stack first runs; K and V go up with their rows repeated for each group
of query heads; the attention is the encoder shader's, a range of keys a query row (`[0, i]` for the text encoder's causal attention, `[i - 128, i + 128]` for a sliding layer of the condition encoders,
`[0, n - 1]` for a full one). The embeddings (the token table, the lyric and timbre input projections) and the final norm stay on the CPU. A stack is stateless: a missing device, a matrix that will not go up or a call
that fails leaves that call to the CPU. The text encoder takes 1.0 GB of the card, the lyric and timbre encoders of each DiT 1.3 GB (the base DiT has its own pair).

On the 5090, with the engine already warm (the first request loads the models and uploads the encoders' weights: 8.2 s against 7.8 s with the encoders on the CPU), a 10 s song is **0.73 s** from the request to the audio, against 1.83 s with the encoders on the CPU (the same song through
the command line, which pays the uploads each time, is 2.5 s: text 0.4, condition 0.4, denoise 1.4, decode 0.2). Held to upstream by the oracles the CPU's are, with COLI_VULKAN=1 on the real checkpoint:
`--text-ref` (the caption's hidden states 2.1e-6 of the norm from upstream's, the CPU's 1.7e-6), and `--gen-ref` on every captured generation, 49 of 49 (the encoders' joint output, `encoder_hidden`, is 2.0 to 2.6e-6 of the
norm from upstream's against the CPU's 1.8e-6; the turbo, the base DiT's own encoders, a reference clip's timbre, a repaint, a lego, a cover). `COLI_VK_ENC=0` keeps the encoders on the CPU; `ACESTEP_ENC_VK_FAULT=N`
makes the Nth call fail, once, to test it. `tests/test_acestep_enc_vk.py` holds the card to the CPU on the tiny fixture with the DiT, the VAE and the planner left on the CPU in both (a song, lyrics, the base
DiT's pair of stacks, a failing call, the switch); three mutants (the causal range, the end and the start of the sliding window) each fail it.

## The 5 Hz planner: a language model writes the song first

ACE-Step's optional planner (`acestep-5Hz-lm-0.6B`, `-1.7B`, `-4B`; the 1.7B is in the main repository, the others are separate) is a Qwen3 decoder with the
audio codes added to its vocabulary: ids from `<|audio_code_0|>` up, the 64000 codes of the audio tokenizer's quantizer, 5 a second. Given the caption and the lyrics
it plans in two phases, as upstream's `generate_with_stop_condition` does. **Phase 1** writes the song's metadata as a YAML block inside `<think>`: the tempo, a
rewritten caption, the duration, the key and scale, the language and the time signature. **Phase 2** writes the codes of the song's coarse structure under a prompt
that carries that block, and the DiT takes their hints (the detokenizer's output, as for a cover's source) as its context where a clip would have been: upstream's
handler turns text to music with `audio_code_string` into a cover of those codes. Songs made this way follow a structure; plain text to music makes it up as it denoises.

**Using it.** The checkpoint needs `acestep-5Hz-lm-1.7B/` beside the DiT: `coli model install ace-step-1.5-planner-1.7b` (3.7 GB, an add-on of the catalog) or
`huggingface-cli download ACE-Step/Ace-Step1.5 --include "acestep-5Hz-lm-1.7B/*" --local-dir <model>`. The 0.6B and 4B planners are not in the catalog (`huggingface-cli download ACE-Step/acestep-5Hz-lm-0.6B --local-dir <model>/acestep-5Hz-lm-0.6B`, the 4B the same, two shards; ask for one with `--lm 0.6B`, `lm_model`).
They are held to upstream on their real weights (2026-11-23): the same four `--lm-ref` cases and the seven `--meta-ref` cases of the 1.7B, captured from upstream's handler with `--lm acestep-5Hz-lm-0.6B` / `-4B`, and `--meta-trees`
(every table upstream's for both tokenizers): the codes identical in every case on the CPU (0.12 s a token for the 0.6B, 0.28 s for the 4B; the metadata phase's logits within their limits in all seven), the logits 1e-6 of the norm from upstream's;
on the card two of the four code cases each (0.02 and 0.06 s a token), the codes identical too. These two are saved as a Qwen3ForCausalLM, their tensors named `model.layers.N...` where the 1.7B's are `layers.N...`: the loader takes either (`tests/test_acestep_lm_prefix.py`). **A bare caption is enough:** `acestep --caption "bright ukulele" --plan
--out song.wav`, `coli music "bright ukulele" --plan`, or `POST /v1/audio/music` with `thinking: true`. Whatever of `bpm`, `key`, `time_signature` and `duration` you
give the model is told as written; it writes the rest, picks the length (10 to 600 s) when there is no `duration`, rewrites the caption the DiT follows (`--no-lm-caption`,
`use_cot_caption: false` keeps yours) and writes the language (`--no-lm-language`, `use_cot_language: false`). With all four of the tempo, key, time signature and duration
given, phase 1 is skipped, as it is upstream. `lm_model`, `lm_temperature`, `lm_cfg_scale`, `lm_top_p`, `lm_top_k` choose the model and its sampling (upstream's defaults are
0.85, 2, 0.9, off; phase 1 runs without guidance, as upstream's does, so the scale is the codes' only). The request streams a `meta` stage (phase 1: how many tokens it
takes is not known, so one frame at the start and one at the end), then a `plan` stage of `duration * 5 + 1` steps, then the usual ones; `colibri` in the answer says the
planner that ran, the codes it wrote and, after phase 1, what it chose (`colibri.meta`: tempo, key, time signature, language, duration and the rewritten caption). READY's `planner`
says what is installed.

**What is built, and what is not.** Both phases, held to upstream (below). Two things of upstream's are replicated as they are, strange as they look: the language the model
writes goes only into phase 2's prompt, not to the DiT (upstream looks for a `vocal_language` key that the parse never makes), and the DiT's key is the model's lowercase
`C major`. **Not built:** the `genres` field (upstream skips it in this phase; a model that writes `genres:` after the caption is refused, not guessed at), the planner's other
jobs (describing a clip from its codes, formatting a caption or lyrics, writing a song from a one-line query).
Upstream limits the duration the model may write to what the GPU can hold (240 s on a CPU, 480 s, up to 600 s); this engine allows 600 s (`ame_max_duration`).

**How phase 1 runs.** The prompt (the chat template with the caption and the lyrics, the assistant turn left open) goes in in one pass, and one sequence advances a token at
a time under upstream's decoder, a state machine (`constrained_logits_processor.py`, 2300 lines there, `acestep_meta.h` here) that forces the field names (`bpm:`, `caption:`,
`duration:` ... as the tokens the tokenizer spells them in), lets through only the digits that keep a value in range (a table of every valid value's tokens: 30 to 300 for the tempo,
10 to 600 for the duration, 70 key and scales, 51 languages, 2, 3, 4 or 6), takes the language as the model's top choice among those that start one, writes a value the caller gave
token by token, and leaves the caption free but for the audio codes and the backtick, until the model writes a line that is not indented and the name of the next field after it
(or 512 tokens). The model's head is read only at the tokens the machine lets through: a few for a value, 153203 for the caption. After the time signature the machine forces the end
token (upstream never writes `</think>`). Then the text is parsed as upstream parses it (a field a non-indented line with a colon, a value that line's rest and the indented lines
after it; the caption's lines flattened to one) and written back as PyYAML writes it for phase 2: 80 columns, plain, single-quoted or double-quoted as PyYAML chooses for each
string, which needs its analysis of the string, its implicit resolvers (a string that would read back as a bool, a number, a timestamp is quoted) and its three scalar writers. Cost
on the 24-core CPU for the 1.7B: 0.13 s a token, 22 s for a 172-token block with a 120-token caption; then phase 2 as before (0.2 s a token, two rows).
On a Vulkan card (below) the same block is 2 s and phase 2 is 0.03 s a token.

**On the card.** Built with `VK=1` and started with `COLI_VULKAN=1` (the router's `--gpu auto` sets it when the card has room, as for any model), the planner runs on the
graphics card: its weights go up once, when the first sequence starts (3.7 GB for the 1.7B, a couple of seconds), each sequence (phase 1's, and the conditional and
unconditional of phase 2) keeps its KV cache on the device, and a step is one submission of the dense chain (norms, q/k/v, head norms, RoPE, attention, o, SwiGLU, down), then
the tied embedding table as the output matrix; only the hidden row and the logits come back (`c/acestep_lm_vk.h`, the speech models' decode chain on the planner's tensors). The text encoder and the VAE stay on the CPU, and the DiT has its own card path (a section below). On the 5090: phase 1 is 0.01 s a token (172 tokens in 2 s, from 22), phase 2 0.03 s a token (50 codes in 1.6 s, from 9), a 10-second
planned song 28 s from a cold start where the CPU took 48 (the DiT's 8.5 s of denoising is now most of what is left). It is the same planner: all 7 phase-1 cases and all 4 codes cases of the
real 1.7B run through `--meta-ref` and `--lm-ref` with COLI_VULKAN=1 and match upstream token for token with the logits at 1e-5 or better (worst 1.9e-5, against limits of 3e-5 and 1e-5;
the matrices are bf16 and the arithmetic f32, so the last bits differ from the CPU's). `COLI_VK_LM=0` keeps the planner on the CPU. A card that is missing, a matrix that will not go up, or a
step that fails (`ACESTEP_LM_VK_FAULT=N` makes step N fail, to test it) puts the whole run back on the CPU: the run is made again from the start and the rest of the process stays there.
`tests/test_acestep_planner_vk.py` holds the card to the CPU on the tiny fixture (same metadata, same codes, so the same audio byte for byte; a failing step; the switch), skipped without a
VK build and a device.

**What is held to upstream, and how.** Phase 2 and the sampler as before (`acestep_ref_lm.py`, `--lm-ref`, `--lm-selftest`). For phase 1, `tools/acestep_ref_meta.py`:

- `capture` runs upstream's own `LLMHandler` on the CPU in float32 at temperature 0 (torch's `multinomial` has no reproduction) and records, for every step of phase 1, the
  decoder's state, the tokens it let through with the model's raw logit at each (the highest 64 when it lets nearly everything through), and the token chosen; the text the model
  wrote, the metadata parsed from it, the YAML of phase 2's prompt and, for the cases that go on, phase 2's prompts and codes; and the token tables the processor builds from the tokenizer.
  `acestep --meta-ref CASE` runs the engine from the same inputs and compares every one; `--meta-trees` compares the tables (every prefix of every value and what may follow it,
  the tokens of every fixed string).
- `yaml` writes a corpus of 6229 strings (what a caption can look like, and every kind of string a plain scalar must not be mistaken for: `1.5e+3`, `.Inf`, timestamps with zones,
  `--- x`, a byte order mark ...) with what `yaml.dump({"caption": s}, allow_unicode=True)` writes for each; `acestep --yaml-ref` compares byte for byte.
- `parse` writes 1546 texts a model might write (crafted, and random lines) with what `parse_lm_output` makes of each and what `_format_metadata_as_cot` writes for that;
  `acestep --parse-ref` compares both.

| | the real 1.7B (7 cases: bare, with lyrics, in Japanese, and four with some of the metadata given, or the caption or language turned off) | the tiny fixture |
|---|---|---|
| the token tables (tempo 302 prefixes, duration 242 at upstream's CPU limit, key 106, language 53, time signature 6, the fixed strings) | identical | identical |
| phase 1's prompt | token ids identical | identical |
| every step: state, how many tokens let through, which, the token chosen | identical in all 172, 119, 169, 157, 127, 29 and 156 steps | identical; and the 3500 steps of a planner that never ends its caption |
| the logits at those tokens | 1e-6 of their norm overall; 1.2e-5 at the worst step (a caption token, logits in the tens) | 6e-7 |
| the metadata parsed, phase 2's YAML (the captions wrap over lines), the prompts, the codes | identical (40, 30, 40 and 50 codes) | identical |
| the YAML emitter on 6229 strings; the parse on 1546 texts | identical | identical |

The sampler (temperature, top-k, top-p) is what the oracle cannot reach, as before: the cases run at temperature 0. The tiny fixture's planner (`make_acestep_tiny.py --lm --lm-text`)
has learned a few metadata blocks (a caption over several lines, a tab-indented one, a trailing space before a break, a capitalised field name, a field the decoder does not know), so that
its greedy decoding walks every state of the machine; random weights never end a caption.

`tools/mutate_acestep_meta.py` breaks acestep_meta.h one line at a time (the YAML width, quoting, escapes and every resolver; the parse; the tables' ranges and tokenization rules;
what ends a caption, which field follows, what is injected, what is forced; the loop) and 67 of 67 mutants are caught; three more need tokens the tiny tokenizer never forms (a line break or a
colon inside a longer token, an audio code as the top token after a line break) are left to the real planner's cases.

```sh
cd c && make acestep                                                                                  # make VK=1 acestep, then COLI_VULKAN=1: the planner on the card
./acestep --model ~/models/ace-step-1.5 --caption "bright ukulele" --plan --out planned.wav          # the model picks tempo, key, length and the rest
./acestep --model ~/models/ace-step-1.5 --caption "bright ukulele" --plan --duration 30 --bpm 120 --out planned.wav
python3 tools/acestep_ref_meta.py capture ~/models/ace-step-1.5 meta_ref                            # upstream's LLMHandler, 7 cases
./acestep --model ~/models/ace-step-1.5 --meta-trees meta_ref/trees.json
./acestep --model ~/models/ace-step-1.5 --meta-ref meta_ref/meta_dur8
python3 tools/acestep_ref_meta.py yaml yaml.json && ./acestep --yaml-ref yaml.json
python3 tools/acestep_ref_meta.py parse parse.json && ./acestep --parse-ref parse.json
python3 tools/acestep_ref_lm.py capture ~/models/ace-step-1.5 lm_ref                                 # phase 2 given the metadata, as before
./acestep --model ~/models/ace-step-1.5 --lm-ref lm_ref/lm10
./acestep --lm-selftest 1
```

## In colibri

The checkpoint is Hugging Face `ACE-Step/Ace-Step1.5`; `coli setup` offers it (6.4 GB: the turbo DiT, the text
encoder and the VAE; the repository's 5 Hz planners are not downloaded, since this engine does not use them). The
family is `ace_step`, modality `audio`, found by `"model_type": "acestep"` in the checkpoint's root `config.json`.

```sh
coli music "calm piano with soft strings" --duration 20            # ~/colibri-music/<date>-<caption>-s<seed>.wav
coli music "upbeat pop" --lyrics @words.txt --language en --bpm 120 --key "C Major" --seed 7 --out song.wav
coli music "with strings" --cover song.mp3 --cover-strength 0.6    # a cover: the song's structure, the description's style
coli music "a bright chorus" --repaint song.wav --from 41 --to 52 --out fixed.wav   # make 41 s to 52 s again, keep the rest
coli music "warm lo-fi" --ref voice.wav --duration 40                # a reference clip lends its sound and style
coli plan --model ~/models/ace-step-1.5       # the three weight sets, all resident, against the RAM free now
coli info --model ~/models/ace-step-1.5       # what is on disk and what the engine will hold
coli doctor --model ~/models/ace-step-1.5
coli serve --model ~/models/ace-step-1.5      # POST /v1/audio/music
```

`coli serve` and `coli web` also write each finished song to that folder before they send it (`COLI_MUSIC_DIR` moves it;
the server's log says `[music] saved <path>`), so a song cannot be lost to a browser tab that was put to sleep, or closed,
while it was made. A failed save is logged and the song is still sent. `COLI_MUSIC_SAVE=0` turns this off.

The WAV also says how it was made: a `LIST`/`INFO` chunk after the samples, whose title (`INAM`) is the prompt, so a player
shows something, and whose comment (`ICMT`) is a JSON record with the prompt, lyrics, language, length, steps, shift, seed,
tempo, key, time signature, task and model. It never holds the clips a cover or repaint started from. The saved file and the
copy sent are the same bytes, and the dashboard's "Open the settings of a saved picture or song" button reads it back into the
controls. `COLI_MEDIA_META=0` leaves the files plain.

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
SSE: `music_generation.progress` events (`stage`: `encode` (reading a clip; only when one is given), `text`, `condition`, `denoise`, `decode`; `step`, `steps`), then
`music_generation.completed` with the same payload, then `[DONE]`. A bad field is a 400 naming it (`param`); a full queue
is a 503 (a song takes minutes, so 429's invitation to retry at once would only refill the queue); a client that hangs up
cancels the song at the engine's next step. `GET /v1/models` lists `"capabilities": ["music_generation"]` and an `audio`
block with the limits above, so a client can build a valid request. The text endpoints and the images routes answer a music
model with a 400 that points at this one.

**Audio in, on the route.** The same endpoint takes `task` (`text2music`, the default, `cover` or `repaint`), `src_audio` (the clip to
cover or repaint) and `reference_audio` (a style reference, any task), each a base64 string (a `data:audio/...;base64,` URL is fine) in a
JSON body, or a file part in a `multipart/form-data` body whose other fields have the same names; WAV or MP3, the format read from the bytes,
at most 128 MiB each and 192 MiB together. A cover or a repaint takes its length from the clip (`duration` is ignored) and its caption
may be empty. The task's own fields, with upstream's names: `repainting_start`, `repainting_end` (seconds; no end is the clip's end; a
start below 0 or an end past the clip adds silence there), `repaint_mode` (`balanced`, `conservative`, `aggressive`), `repaint_strength`
(above 0 and at most 1: the default is 0.5, and `conservative` is the mode for the small end), `audio_cover_strength` (the same range,
default 1) and `cover_noise_strength` (0 to 1, default 0). Each is refused with a 400 naming it when it does not apply to the task. The
answer is the same, with `task` and `src_seconds` added to its `colibri` block; `/v1/models` lists `audio.tasks` and `audio.audio_in`
(`formats`, `max_seconds`, `ref_seconds`, `max_clip_bytes`, `max_total_bytes`) and the capability `audio_input`, so a client can offer the
clips only where they work. The clips are written to a private temporary directory for the engine, which reads them from there, and the
directory is removed when the request ends, however it ends. The examples are in [api.md](api.md).

Measured through the real gateway on the real checkpoint: a 6 s clip as JSON and as a raw WAV for the same seed are the
same bytes, and the same as `coli music` and `acestep --caption` make for it.

**Memory.** Every weight stays loaded between songs: 6.9 GB peak resident set for a 10 s clip, 6.95 GB for 60 s (the
weights dominate; the working buffers are a few hundred MB at most). Measured on Windows with `PeakWorkingSet64`.

**Speed.** The engine uses every core, so another busy process slows it a lot: the same 10 s clip took 16 s on an idle
24-core i9 and 68 to 75 s while a Qwen-Image server was drawing on the same machine. Quote the first.

**The dashboard.** `coli web` (or the page `coli serve` serves) shows the music composer when the selected model lists
`music_generation`: a lyrics box, length, steps, language, seed, a progress bar with Cancel, then the browser's audio player
with Download WAV, Copy seed and New seed, and, when the engine takes audio in, a Compose / Cover / Repaint switch with clip
cards and a Reference chip (docs: `web/README.md`). The model switcher offers a music model like any other. What the checkpoint has shows
as more controls (`/v1/models`' `audio.planner`, `audio.guided` and `audio.dits` say so, and a checkpoint without them shows none): a **Plan**
chip (the 5 Hz planner writes the song's outline first) with an **Auto length** chip that leaves the length to it, a **DiT** choice (Turbo, or
the base DiT with its guidance and its own 32-step default), and Extract / Add a track / Complete beside Cover and Repaint (a track, the span of a
lego, the tracks to complete with). The finished song says what the planner chose (tempo, key, time signature, language, the caption it
rewrote) under **What the planner chose**, and the DiT that ran. `web/tests/music_guided_ui.py` holds those in a browser against a stub told
that the checkpoint has the planner and the base DiT.
`web/tests/music_ui.py` runs it in a real browser (Playwright) against the real gateway and the stub engine: a song whose
length the browser reads back from the audio element, the downloaded file, a new seed, lyrics and a language on the wire,
a failure, an invalid length held back, a cancel; and `web/src/lib/music.test.ts` holds the client (`generateMusic`'s
request and its stream, the fallback to a plain request, every error) and the page's strings.

**What is tested without the weights.** `tools/acestep_stub.py` speaks the protocol and makes a synthetic clip that
depends on the request, the way `qwenimage_stub.py` does for pictures. `tests/test_acestep_engine_client.py` holds
`music_engine.py` (the client, the request rules, WAV, saving, the plan) to it, `tests/test_acestep_gateway.py` holds
the route (real `APIServer`, real `MusicEngine`, the stub behind them: the framing, determinism, every refusal, SSE, the
raw WAV, hang-up cancellation, queueing, the model card), and `tests/test_acestep_cli.py` the commands as a user types
them. Audio in has its own files: `tests/test_acestep_audio_in.py` (every request rule and limit, base64, data URLs and multipart,
format sniffing, the temporary files removed on success, error, cancel and hang-up, SSE with the encode stage, a cover, a repaint and
a reference through the real gateway) and `tests/test_acestep_audio_in_cli.py` (each flag and each refusal). The real engine is held
to the same protocol by `tests/test_acestep_serve.py`.

CI (`acestep-tiny-check`) rebuilds the fixture and the references from upstream at the pinned commit, runs the oracles,
the refusal tests (`tests/test_acestep_{vae,dit,text,gen}_load.py`, `tests/test_acestep_serve.py`), the four mutation tools, and the oracles again under ASan
and UBSan.

Audio in is held in CI too, on a second fixture: `python3 tools/make_acestep_tiny.py acestep_tiny_enc --audio-in` (7.8 MB). An encoder's last layer holds the mean
and the scale of the latents, so 64 latent channels need a 128-wide encoder, and that width is also every stage's (times `channel_multiples`); the first fixture's
8-wide one makes 4 latents, which the DiT, the tokenizer and the engine all refuse. The audio-in fixture has the same DiT and text encoder and a flat 128-wide
encoder (`channel_multiples` all 1), calibrated like the real one: posterior means near 1, a scale near 1e-2. The text-to-music captures' own audio is what gets encoded.
Against upstream's handler: `--vae-enc-ref` on 8 clips (parameters 1e-5 of the signal, tiled 1e-5), `--cover-ref` on 5 (2e-6), `--gen-ref` on 19 whole generations
(audio 1e-6 of the peak, against 4e-6 to 1.2e-4 on the real checkpoint, where the float32 noise is larger), and the mutation tools: `mutate_acestep_venc.py`
(28 of 28), `mutate_acestep_tok.py` (23 of 23) and `mutate_acestep_audioin.py` (45 of 45). A mutant a fixture cannot show is left out with `--skip` and the tool says
which: the flat encoder has no widths to get wrong (`block-widths`), the tiny tokenizer never reaches the clamp (`fsq-no-hard-clamp`), and the non-cover prompt is cut
only with Qwen's tokenizer (`noncover-prompt-not-cut`) and padded only with the tiny one (`noncover-prompt-not-padded`). Running them for the first time on this
fixture found `cover_half` wrong: see the next paragraph.

The non-cover prompt (a cover with a strength under 1 runs its last steps on the text-to-music prompt): upstream pads its ids to the cover prompt's length. With Qwen's
tokenizer the text-to-music instruction is a token longer, so the pad is a cut of the final token, which the engine always did. With the tiny tokenizer it is shorter,
and the pad token goes after the text; upstream does not mask those rows, because the DiT builds its cross-attention mask from the audio's mask and never the text's,
so they take part in attention. The engine now pads with the tokenizer's `<|endoftext|>` and does the same (`cover_half`: latents 2e-2 off before, 6e-7 after).
