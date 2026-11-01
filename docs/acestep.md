# ACE-Step 1.5 in C (work in progress)

[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) turns a caption and lyrics into a song: 48 kHz stereo, 10
seconds to 10 minutes, from a diffusion transformer over 25 Hz latents and an audio VAE. This directory of work ports
it to the same shape as [Qwen-Image](qwen-image.md): one C file, weights read as they are, every stage held to the
reference by an oracle that compares tensors, not just the final audio.

**Nothing here is user-facing yet.** The engine does not generate music. What exists is the first stage, the VAE
decoder, and the tooling every later stage is checked with. `coli` does not know about it and `make install` does
not ship it.

| stage | what | state |
|---|---|---|
| 0 | the reference: upstream's own handler, observed at every stage; a tiny random checkpoint of the real structure | done |
| 1 | the VAE decoder, `c/acestep_vae.h`: latents to 48 kHz stereo, tiled as upstream tiles | done, oracle-checked |
| 2 | the DiT, its condition encoders and the turbo sampler (DCW included) | next |
| 3 | the text encoder (Qwen3-Embedding-0.6B) and the prompt template | |
| 4 | the engine proper: serve protocol, gateway route, `coli music`, the web player | |
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
Eleven cases are captured, each in its own directory (`ref.safetensors`, `ref.json`, `audio.wav`): DCW on and off,
with lyrics, an odd frame count, shift 3, a 5-step schedule, and a 30-second clip. Read the DCW-off cases with care:
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

CI (`acestep-tiny-check`) rebuilds the fixture and the references from upstream at the pinned commit, runs the oracle,
the refusal tests (`tests/test_acestep_vae_load.py`), the mutations, and the oracle again under ASan and UBSan.
