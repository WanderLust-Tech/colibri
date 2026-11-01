#!/usr/bin/env python3
"""Oracle for the Qwen-Image-2.1 VAE decoder (c/qwenimage_vae.h).

    make_qwenimage_vae_tiny.py OUTDIR
        builds a tiny random AutoencoderKLQwenImage21 with the real depth
        (five up blocks, three resnets each, the mid attention) and narrow
        channels, saves it to OUTDIR/vae, and writes the diffusers decode of
        random latents to OUTDIR/vae_ref/ref.safetensors.

    make_qwenimage_vae_tiny.py OUTDIR --vae VAE_DIR --shapes 32x32,27x48
        decodes random latents with an existing VAE (the real checkpoint) and
        writes only OUTDIR/vae_ref/ref.safetensors.

Every case k stores three tensors:
    case{k}.z     f32  [h][w][z_dim]  normalized latents, token-major: exactly
                                      what the DiT hands to the decoder
    case{k}.out   f32  [4][16h][16w]  vae.decode() after the clamp to [-1, 1]
    case{k}.rgba  u8   [16h][16w][4]  VaeImageProcessor.postprocess(..., "pil")
The decode path is the pipeline's own: unpack to [1, C, 1, h, w], multiply by
latents_std and add latents_mean from the VAE config, decode, take frame 0,
postprocess. Needs torch and diffusers (the reference venv), not the engine.

The encoder (c/qwenimage_vae_enc.h) has its own cases, --enc-shapes (latent HxW):
    enc{k}.img    u8   [16h][16w][4]  an RGBA picture: smooth colour, noise, varying alpha
    enc{k}.z      f32  [h][w][z_dim]  what QwenImage21Pipeline._encode_vae_image makes of it:
                                      vae.encode(...) in argmax mode (the posterior's mean),
                                      then (latent - latents_mean) / latents_std; token-major,
                                      so it is what the decoder takes
The picture goes through VaeImageProcessor.preprocess as the pipeline does.
"""
import argparse
import os
import time

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from safetensors.torch import save_file


def tiny_vae(seed):
    torch.manual_seed(seed)
    z_dim = 16  # not 64: the engine must take z_dim from config.json
    g = torch.Generator().manual_seed(seed + 1)
    vae = AutoencoderKLQwenImage21(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=z_dim,
        latents_mean=(torch.randn(z_dim, generator=g) * 1.5).tolist(),
        latents_std=(2.5 + torch.rand(z_dim, generator=g) * 2).tolist(),
    )
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if name.endswith("gamma"):
                # the default is all ones, which would hide a gamma indexed per the wrong axis
                p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=g))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
        # left at its default init, conv_out spans the whole [-1, 1] range and
        # saturates about 3% of the values: the clamp is exercised, yet the
        # uint8 comparison still sees almost every pixel unclamped
    return vae.eval()


def decode_case(vae, z_tok, h, w):
    """z_tok [h*w][z_dim] normalized latents -> (float [4][H][W], uint8 [H][W][4], seconds)."""
    cfg = vae.config
    z_dim = cfg.z_dim
    # pipeline._unpack_latents, without its rounding of height and width to 32 pixels
    lat = z_tok.unsqueeze(0).transpose(1, 2).reshape(1, z_dim, 1, h, w)
    mean = torch.tensor(cfg.latents_mean).view(1, z_dim, 1, 1, 1).to(lat.dtype)
    std = torch.tensor(cfg.latents_std).view(1, z_dim, 1, 1, 1).to(lat.dtype)
    lat = lat * std + mean
    t0 = time.time()
    with torch.no_grad():
        image = vae.decode(lat, return_dict=False)[0][:, :, 0]
    dt = time.time() - t0
    proc = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=z_dim)
    pil = proc.postprocess(image, output_type="pil")[0]
    rgba = np.array(pil)
    return image[0].contiguous(), torch.from_numpy(rgba.copy()), dt


def test_picture(h, w, seed):
    """A deterministic RGBA picture [16h][16w][4] uint8. Smooth colour so the encoder sees
    structure at every scale, noise so it is not only low-pass, alpha that varies so the
    fourth input channel is not a constant."""
    H, W = 16 * h, 16 * w
    g = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    chans = []
    for c in range(3):
        fy, fx, ph = g.uniform(1.0, 4.0), g.uniform(1.0, 4.0), g.uniform(0, 6.28)
        chans.append(0.5 + 0.4 * np.sin(2 * np.pi * (fy * yy * (c + 1) / 2 + fx * xx) + ph))
    alpha = 0.75 + 0.25 * np.sin(2 * np.pi * (xx + yy))
    img = np.stack(chans + [alpha], axis=-1) + 0.12 * g.standard_normal((H, W, 4))
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)


def encode_case(vae, rgba):
    """rgba [H][W][4] uint8 -> (normalized latents [h][w][z_dim], seconds)."""
    cfg = vae.config
    H, W = rgba.shape[:2]
    # the pipeline: preprocess to [-1, 1], add the frame axis, encode in argmax mode, normalize
    proc = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=cfg.z_dim)
    x = proc.preprocess(Image.fromarray(rgba, "RGBA"), height=H, width=W).unsqueeze(2)
    assert x.shape == (1, cfg.in_channels, 1, H, W), x.shape
    t0 = time.time()
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.mode()
    dt = time.time() - t0
    mean = torch.tensor(cfg.latents_mean).view(1, cfg.z_dim, 1, 1, 1).to(lat.dtype)
    std = torch.tensor(cfg.latents_std).view(1, cfg.z_dim, 1, 1, 1).to(lat.dtype)
    lat = (lat - mean) / std
    return lat[0, :, 0].permute(1, 2, 0).contiguous(), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--vae", help="existing VAE dir (default: build a tiny random one)")
    ap.add_argument("--shapes", default="5x7,4x6", help="latent HxW list, comma separated")
    ap.add_argument("--enc-shapes", default="3x5,4x4", help="latent HxW of the encoder cases (the picture is 16x larger)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    if a.vae:
        vae = AutoencoderKLQwenImage21.from_pretrained(a.vae, torch_dtype=torch.float32).eval()
    else:
        vae = tiny_vae(a.seed)
        vae.save_pretrained(os.path.join(a.outdir, "vae"))
    g = torch.Generator().manual_seed(a.seed + 7)
    tensors = {}
    for k, s in enumerate(a.shapes.split(",")):
        h, w = (int(v) for v in s.lower().split("x"))
        z_tok = torch.randn(h * w, vae.config.z_dim, generator=g)
        out, rgba, dt = decode_case(vae, z_tok, h, w)
        sat = (out.abs() >= 1.0).float().mean().item()
        print(f"case{k}: latent {h}x{w} -> image {16 * h}x{16 * w}, diffusers decode {dt:.2f} s, "
              f"out range [{out.min().item():.3f}, {out.max().item():.3f}], saturated {100 * sat:.1f}%")
        tensors[f"case{k}.z"] = z_tok.reshape(h, w, -1).contiguous()
        tensors[f"case{k}.out"] = out
        tensors[f"case{k}.rgba"] = rgba
    for k, s in enumerate(a.enc_shapes.split(",") if a.enc_shapes else []):
        h, w = (int(v) for v in s.lower().split("x"))
        pic = test_picture(h, w, a.seed + 100 + k)
        z, dt = encode_case(vae, pic)
        print(f"enc{k}: image {16 * h}x{16 * w} -> latent {h}x{w}, diffusers encode {dt:.2f} s, "
              f"z mean {z.mean().item():.3f} std {z.std().item():.3f}")
        tensors[f"enc{k}.img"] = torch.from_numpy(pic.copy())
        tensors[f"enc{k}.z"] = z
    os.makedirs(os.path.join(a.outdir, "vae_ref"), exist_ok=True)
    save_file(tensors, os.path.join(a.outdir, "vae_ref", "ref.safetensors"))


if __name__ == "__main__":
    main()
