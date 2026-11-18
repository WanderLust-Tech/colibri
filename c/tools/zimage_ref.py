#!/usr/bin/env python3
"""A reference run of Z-Image through diffusers on the CPU, saving what the C engine (zimage.c) must reproduce.

    python tools/zimage_ref.py MODEL_DIR OUT_DIR [SIZE|WxH] [STEPS] [PROMPT] [--float32-loop] [--no-fp32-dit]

Needs torch (CPU is enough), diffusers >= 0.36 with ZImagePipeline, transformers, safetensors, numpy and Pillow,
and the checkpoint (Tongyi-MAI/Z-Image-Turbo). It writes OUT_DIR/ref.safetensors, which `zimage --model MODEL_DIR
--ref OUT_DIR` reads, and OUT_DIR/image.png (what the pipeline made).

Stages, each saved on its own so the engine's check can start every stage from the reference input:
  1. the prompt through the chat template and Qwen3-4B in float32: the second-to-last hidden state of the real tokens
  2. the denoising loop in bfloat16 from fixed noise: the latent after every step
  3. one transformer evaluation in float32 at step 0, with taps after the noise refiner, the context refiner and the
     first, middle and last main layers (the real model needs about 25 GB of RAM for it; --no-fp32-dit skips it)
  (--float32-loop runs stage 2 in float32 as well: for a tiny model, where it costs nothing)
  4. the VAE decode of the final latent in float32
The sizes are reduced to what a CPU can do in minutes: 384 x 384 and 8 steps take about six minutes."""
import gc
import os
import sys
import time

import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SKIP_FP32 = "--no-fp32-dit" in sys.argv
if len(args) < 2:
    sys.exit(__doc__)
MODEL, OUT = args[0], args[1]
SIZE_TEXT = args[2] if len(args) > 2 else "384"
WIDTH, HEIGHT = (int(v) for v in (SIZE_TEXT.split("x") if "x" in SIZE_TEXT else (SIZE_TEXT, SIZE_TEXT)))
LOOP_DTYPE_FP32 = "--float32-loop" in sys.argv
STEPS = int(args[3]) if len(args) > 3 else 8
PROMPT = args[4] if len(args) > 4 else "A small red lighthouse on a rocky cliff at dusk, calm sea, soft orange sky, photograph"
SEED = 42
os.makedirs(OUT, exist_ok=True)

import torch
from PIL import Image
from safetensors.numpy import save_file

torch.set_num_threads(os.cpu_count() or 8)
t0 = time.time()
tensors = {}


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def keep(name, array):
    a = np.ascontiguousarray(array.astype(np.float32) if array.dtype == np.float64 else array)
    tensors[name] = a


# ---- 1. the text encoder, float32 ----
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.path.join(MODEL, "tokenizer"))
text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
with open(os.path.join(OUT, "prompt_text.txt"), "w", encoding="utf-8", newline="") as f:
    f.write(text)
enc = tok([text], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
mask = enc.attention_mask.bool()
ids = enc.input_ids[mask]
keep("ids", ids.numpy().astype(np.int64))
log(f"prompt: {len(ids)} tokens")
te = AutoModel.from_pretrained(os.path.join(MODEL, "text_encoder"), dtype=torch.float32)
te.eval()
with torch.no_grad():
    out = te(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True)
embeds = out.hidden_states[-2][0][mask[0]].float()
keep("embeds", embeds.numpy())
log(f"embeds {tuple(embeds.shape)}")
del te, out
gc.collect()

# ---- 2. the denoising loop, bfloat16 ----
from diffusers import ZImagePipeline

loop_dtype = torch.float32 if LOOP_DTYPE_FP32 else torch.bfloat16
pipe = ZImagePipeline.from_pretrained(MODEL, text_encoder=None, tokenizer=None, vae=None, torch_dtype=loop_dtype)
steps = {}


def on_step(p, i, t, kw):
    steps[i] = kw["latents"].detach().clone().float().numpy()
    log(f"step {i} t={float(t):.2f}")
    return kw


gen = torch.Generator().manual_seed(SEED)
lat = pipe(prompt=None, prompt_embeds=[embeds.to(loop_dtype)], height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
           guidance_scale=0.0, generator=gen, output_type="latent", callback_on_step_end=on_step,
           callback_on_step_end_tensor_inputs=["latents"]).images
final = lat.float().numpy()
keep("final_latent", final)
for i, a in steps.items():
    keep(f"latent_step{i}", a)
shape = (1, 16, 2 * (HEIGHT // 16), 2 * (WIDTH // 16))
keep("init_latent", torch.randn(shape, generator=torch.Generator().manual_seed(SEED), dtype=torch.float32).numpy())
keep("sigmas", pipe.scheduler.sigmas.float().numpy())
keep("timesteps", pipe.scheduler.timesteps.float().numpy())
del pipe
gc.collect()

# ---- 3. one float32 evaluation of the transformer at step 0, with taps ----
if not SKIP_FP32:
    from diffusers import ZImageTransformer2DModel

    tr = ZImageTransformer2DModel.from_pretrained(os.path.join(MODEL, "transformer"), torch_dtype=torch.float32)
    tr.eval()
    taps = {}

    def tap(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            o = o.detach().float()
            taps[name] = (o[0] if o.dim() == 3 else o).numpy().copy()
        return hook

    tr.noise_refiner[-1].register_forward_hook(tap("dit_after_noise_refiner"))
    tr.context_refiner[-1].register_forward_hook(tap("dit_after_context_refiner"))
    n_layers = len(tr.layers)
    for name, i in (("first", 0), ("mid", (n_layers - 1) // 2), ("last", n_layers - 1)):
        tr.layers[i].register_forward_hook(tap(f"dit_after_layer_{name}"))
    ts0 = float(tensors["timesteps"][0])
    with torch.no_grad():
        o = tr([torch.from_numpy(tensors["init_latent"][0]).unsqueeze(1)], torch.tensor([(1000.0 - ts0) / 1000.0]), [embeds], return_dict=False)[0]
    keep("dit_pred0", o[0].squeeze(1).float().numpy())
    ntok = (HEIGHT // 16) * (WIDTH // 16)
    for k, v in taps.items():
        keep(k, v[:ntok] if "layer" in k else v)   # the image rows come first in the unified sequence
    del tr
    gc.collect()
    log("float32 transformer evaluation done")

# ---- 4. the VAE, float32 ----
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(os.path.join(MODEL, "vae"), torch_dtype=torch.float32)
z = torch.from_numpy(final) / vae.config.scaling_factor + vae.config.shift_factor
with torch.no_grad():
    img = vae.decode(z, return_dict=False)[0]
keep("decoded", img.float().numpy())
arr = ((img[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
Image.fromarray(arr).save(os.path.join(OUT, "image.png"))

save_file(tensors, os.path.join(OUT, "ref.safetensors"))
log(f"saved {OUT}/ref.safetensors ({len(tensors)} tensors) and image.png")
