#!/usr/bin/env python3
"""End-to-end oracle for an edit: c/qwenimage.c's engine_edit against diffusers' whole QwenImage21Pipeline.

    make_qwenimage_edit_e2e_ref.py MODEL_DIR OUTDIR

runs the pipeline (float32) on MODEL_DIR (the tiny pipeline of make_qwenimage_tiny.py) with RGBA source
pictures of arbitrary sizes, a prompt and given initial noise, and writes OUTDIR/ref.json and
OUTDIR/ref.safetensors. The engine is handed the same pictures at their own size and has to do everything the
pipeline does: resize them (calculate_dimensions, Lanczos), encode them with the vision tower, the language
model and the VAE, build the DiT's prefix, denoise and decode. Each case k stores

    c{k}.src{j}   u8   [h][w][4]   picture j as given
    c{k}.lat      f32  [N][C]      the initial noise (tokens of the target's latent grid)
    c{k}.final    f32  [N][C]      the latents after the last step
    c{k}.rgba     u8   [H][W][4]   the decoded picture

and ref.json lists, per case, the prompt, the source sizes, the target size, the steps and output_resolution.
The tiny pipeline's output_resolution is 384 (the engine's smallest): a reference picture is about 147k
pixels, and the sizes below keep every resized picture above the processor's 65536-pixel floor.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwenimage_ref as qref  # noqa: E402

SEED = 20261101
RES = 384
STEPS = 3
# (name, prompt, source sizes (h, w), target (h, w) or None for the pipeline's default)
CASES = [
    ("one_picture_default_size", "make the sky purple", [(200, 300)], None),
    ("two_pictures_given_size", "put the cat from the first picture next to the dog", [(384, 256), (300, 400)], (256, 320)),
    ("small_source_upscaled", "turn it into a watercolor", [(120, 90)], (288, 352)),
]


def picture(rng, h, w):
    """Smooth colour fields plus a little noise; an opaque top, a clear patch and a blended remainder."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([127 + 100 * np.sin(xx / (11 + 5 * c) + 2 * c) * np.cos(yy / (13 + 7 * c)) for c in range(3)], axis=-1)
    rgb = np.clip(base + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8)
    alpha = np.clip(200 + 55 * np.sin(xx / 17) * np.cos(yy / 23) + rng.normal(0, 4, (h, w)), 0, 255).astype(np.uint8)
    alpha[: h // 3] = 255
    alpha[h // 3: h // 2, : w // 3] = 0
    return np.dstack([rgb, alpha])


def main():
    from diffusers import QwenImage21Pipeline
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    model, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    pipe = QwenImage21Pipeline.from_pretrained(str(model), torch_dtype=torch.float32,
                                               processor=qref.load_processor(model))
    pipe.set_progress_bar_config(disable=True)
    C = pipe.transformer.config.in_channels
    rng = np.random.default_rng(SEED)
    tensors, meta = {}, []
    for k, (name, prompt, sizes, target) in enumerate(CASES):
        arrays = [picture(rng, h, w) for h, w in sizes]
        pils = [Image.fromarray(a) for a in arrays]
        if target is None:
            w, h, _ = calculate_dimensions(RES * RES, pils[-1].size[0] / pils[-1].size[1])
        else:
            h, w = target
        h, w = h // 32 * 32, w // 32 * 32
        lat = qref.pack_initial_latents(SEED + k, C, w, h)
        final = {}

        def cb(p, i, t, kw):
            final["latents"] = kw["latents"].detach().clone()
            return {}

        with torch.no_grad():
            image = pipe(prompt=prompt, image=pils, latents=lat[None], width=w, height=h, num_inference_steps=STEPS,
                         true_cfg_scale=1.0, output_resolution=RES, callback_on_step_end=cb).images[0]
        rgba = np.array(image)
        assert rgba.shape == (h, w, 4), rgba.shape
        for j, a in enumerate(arrays):
            tensors[f"c{k}.src{j}"] = torch.from_numpy(a.reshape(-1)).contiguous()
        tensors[f"c{k}.lat"] = lat
        tensors[f"c{k}.final"] = final["latents"][0].float().contiguous()
        tensors[f"c{k}.rgba"] = torch.from_numpy(rgba.reshape(-1).copy())
        meta.append({"name": name, "prompt": prompt, "sources": [{"h": a.shape[0], "w": a.shape[1]} for a in arrays],
                     "width": w, "height": h, "steps": STEPS, "ref_resolution": RES, "seed": SEED + k})
        print(f"case{k} {name}: {len(pils)} picture(s) -> {w}x{h}, final latents std {final['latents'].std().item():.3f}, "
              f"rgba mean {rgba.mean():.1f}")
    save_file(tensors, str(out / "ref.safetensors"))
    (out / "ref.json").write_text(json.dumps({"cases": meta}))


if __name__ == "__main__":
    main()
