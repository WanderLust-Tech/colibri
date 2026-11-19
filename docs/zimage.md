# Z-Image on colibri

`c/zimage` turns a text prompt into a picture with
[`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), read straight from the official
diffusers checkpoint. There is no conversion step: the engine quantizes the text encoder and the diffusion
transformer to int8 while it loads them. It is written beside [`qwenimage`](qwen-image.md) and shares its matrix
kernels, its line protocol and its front ends: `coli run "prompt"`, `coli chat`, `coli serve` and
`POST /v1/images/generations` work for it as they do for Qwen-Image.

Z-Image is the lighter of the two picture models, and the one with an open licence: **Apache-2.0**, where
Qwen-Image-2.1's weights are under the Qwen Research License (non-commercial). It draws from text only; it does not
edit pictures (the engine answers an edit request with an error; the gateway lists only `image_generation` for it and
answers `/v1/images/edits` with a 400).

## Download

```
hf download Tongyi-MAI/Z-Image-Turbo --local-dir ~/Models/Z-Image-Turbo --exclude "assets/*"
```

The checkpoint is 32.9 GB: the diffusion transformer (24.6 GB, stored in float32), a Qwen3-4B text encoder
(8.0 GB bf16), the VAE (0.2 GB) and the tokenizer. The engine reads the transformer's float32 values and keeps them as
int8 rows (6.0 GB in memory); a pre-converted int8 container would cut the download and the 14 s the first load spends
quantizing, and is not made yet. `coli setup` lists it as `z-image-turbo`.

```
coli info --model ~/Models/Z-Image-Turbo       # its parts and their weight
coli plan --model ~/Models/Z-Image-Turbo       # what fits in RAM
coli run --model ~/Models/Z-Image-Turbo --size 768x768 "a small red lighthouse on a cliff at dusk"
```

## RAM

The weights in memory, int8 rows except the VAE (f32):

| part | on disk | in RAM |
|---|---|---|
| text encoder (Qwen3-4B, 35 of its 36 layers) | 8.0 GB bf16 | 3.5 GB int8 |
| transformer (S3-DiT, 6.15B parameters) | 24.6 GB f32 | 6.0 GB int8 |
| VAE decoder | 0.2 GB bf16 | 0.3 GB f32 |

A one-shot run (`coli run`, `zimage --prompt`) loads the text encoder for the prompt and frees it before the
transformer loads; a server (`coli serve`) keeps both loaded (`COLI_IMG_TE=stage` frees the text encoder after each
prompt). Measured peak working set on the CPU: **8.6 GB** for a 768x768 picture with the text encoder staged,
**10.1 GB** at 1024x1024, 11.9 GB at 768x768 with everything resident. For comparison Qwen-Image-2.1 holds 15.5 GB
when everything is resident (text encoder 7.0 GB, transformer 7.1 GB, VAE 1.4 GB).

## Sizes, steps, seed

- **Size**: each side a multiple of **16** between 256 and 2048 (Qwen-Image snaps to 32). Default 768x768.
- **Steps**: 1 to 200, default **8**. Z-Image-Turbo is distilled for 8 steps; more steps are slower and do not make
  it better. The schedule is flow matching with Euler steps and a fixed shift of 3 (the checkpoint's
  `scheduler_config.json`).
- **Seed**: the same seed gives the same picture. The noise is colibri's own generator, not PyTorch's, so a seed does
  not reproduce a picture made by the diffusers pipeline; the engine is held to the pipeline by feeding it the
  pipeline's own noise (below).
- **Prompt**: a user turn of Qwen3's chat template, at most 512 tokens (longer ones are cut, as the pipeline cuts them).
  There is no negative prompt and no guidance: the Turbo model is run without classifier-free guidance, as its
  authors do.

## Timings

768x768 and 1024x1024, 8 steps, seed 42, int8 weights, on a 24-core i9-14900KF and an RTX 5090. "Encode" includes
loading the weights the first time (the text encoder 5 s, the transformer 14 s); a server pays it once.

| | per step | denoise | VAE decode |
|---|---|---|---|
| 768x768, CPU | 21.4 s | 171 s | 6.0 s |
| 768x768, matrices on the device one by one (`COLI_VK_CHAIN=0`) | 8.7 s | 72 s | 6.0 s |
| 768x768, **device chain** (`VK=1`, `COLI_VULKAN=1`) | **1.4 s** | **13 s** | 6.0 s |
| 1024x1024, CPU | 41 s | 328 s | 11.1 s |
| 1024x1024, **device chain** | **3.2 s** | **27 s** | 11.1 s |

Qwen-Image-2.1 on the same machine, 768x768: CPU 18.3 s a step, device chain 1.04 s a step, VAE 1.8 s (its VAE runs
on the device; Z-Image's does not yet). On the CPU the two are within a fifth of each other; Z-Image's advantage is
the memory it holds, and its licence.

The CPU profile of a 768x768 step: 74% in the matrix products (about 1.6 TFLOPS of int8), 23% in the attention (f32,
2336 rows), 3% in everything else.

With `COLI_VULKAN=1` and a build that has Vulkan (`make VK=1 zimage`), the transformer's matrices go to the device,
and when the device's free memory holds all 30 blocks (6 GB of weights and a step's buffers) the blocks run as one
chain (`zimage_chain.h`): the sequence goes up once, every block (norms, projections, rotary, attention, gated
residuals) runs on the device, and the image rows come back. The two refiner blocks on each side and the final layer
stay on the CPU. `COLI_VK_CHAIN=0` keeps the blocks off the chain; a device that is lost mid-run hands the step back
to the CPU.

## How it is checked

`tools/zimage_ref.py` runs the real pipeline in diffusers and saves what the engine has to reproduce: the prompt's
tokens, the text encoder's output in float32, the noise and the latent after every step, one float32 evaluation of the
transformer with taps after the noise refiner, the context refiner and the first, middle and last main blocks, and the
VAE decode in float32. `zimage --model DIR --ref REFDIR` starts every stage from the reference's input for that stage,
so an error shows up where it is made:

```
COLI_IMG_BITS=16 COLI_IMG_ACT8=0 ./zimage --model ~/Models/Z-Image-Turbo --ref refdir
```

With `COLI_IMG_BITS=16` the engine computes on the checkpoint's own bf16 values, so each stage must match to summation
order. Measured on the real model (384x384 and 448x448 pictures; 448x448 has 784 image tokens, which the transformer
pads to 800): the text encoder 2e-7 relative, every transformer tap 1e-7 to 3e-6, the noise prediction 1e-6, the VAE
decode 8e-7, and the tokenizer produces the reference's ids exactly. On the device the chain's prediction agrees
to 1.4e-6. With the default int8 weights the step-0 prediction is within 4% of the float32 reference, which is the
quantization, and the pictures are as good as the pipeline's (the first run, 384x384, is a red lighthouse on a rocky
cliff; the sampled latents drift from a bf16 pipeline's as the steps go, as two chaotic samplers do).

`tools/make_zimage_tiny.py` builds a random tiny pipeline in the checkpoint's layout and its float32 reference, in
seconds, for CI (`zimage-tiny-check`): padded caption and image rows, a timestep embedding that is 64 wide instead of
256, grouped-query attention in the text encoder, a float32 transformer and bf16 text encoder and VAE.
`tests/test_zimage_engine_serve.py` (with `ZIMAGE_TINY=dir`) holds the engine to the serve protocol.

## What is not there

- **Editing and image-conditioned drawing.** Text to picture only.
- **The base Z-Image** (`Tongyi-MAI/Z-Image`, 20.5 GB, bf16): the same network, but made for about 50 steps with
  classifier-free guidance (a second pass with a negative prompt each step). The engine loads it but runs it
  without guidance, which is not how it is meant to run (not tried); guidance is not implemented.
- **The VAE on the device.** The decode is 6 s at 768x768 and 11 s at 1024x1024 on the CPU, now the biggest piece of a
  picture on a card; Qwen-Image's decoder runs on the device and takes 1.8 s.
- **Streaming blocks for a small card.** The chain runs only when the device holds all 30 blocks (about 7 GB with a
  1024x1024 step's buffers); on a card that cannot, the matrices still go to the device one by one.
- **Previews** (`"preview": true` in a request) are not sent.
- **A pre-converted container.** See Download.

## License

The weights are Apache-2.0 (the model card). The engine is colibri's, Apache-2.0.
