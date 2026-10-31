#!/usr/bin/env python3
"""Oracle for the reference-image path of the Qwen-Image-2.1 DiT (c/qwenimage.c, dit_prefix_edit).

    make_qwenimage_edit_ref.py MODEL_DIR OUTDIR

loads MODEL_DIR/transformer with diffusers' QwenImage21Transformer2DModel (float32), runs its forward on
synthetic edit inputs, and writes OUTDIR/ref.json and OUTDIR/ref.safetensors. MODEL_DIR is the tiny
pipeline of make_qwenimage_tiny.py (or the real checkpoint). The text encoder and the VAE are not
involved: the inputs are what the pipeline hands the transformer, so the DiT's half of an edit is held
to diffusers on its own.

An edit is one joint sequence. The vision-language sequence has `<|image_pad|>` slots, each standing
for 2x2 latent tokens; the reference latents fill them, the target's tokens are appended. Each case k
stores

    c{k}.emb        f32  [Lv][ctx]      the VLM hidden states (random: only the DiT's reading of them matters)
    c{k}.img_mask   u8   [Lv]           1 at the image_pad slots
    c{k}.ref        f32  [Nref][C]      the reference latents, every image's tokens in raster order, back to back
    c{k}.tgt        f32  [Nt][C]        the target's noisy latents
    c{k}.cos/.sin   f32  [Ltot][hd/2]   the RoPE the transformer built for the whole joint sequence
    c{k}.pred       f32  [Nt][C]        the noise prediction for the target's tokens
and ref.json lists, per case, the reference and target shapes (in latent tokens) and the timestep.

Three layouts: one reference; two references with text between them and after; two references with
no text between them (adjacent runs of image slots must stay separate, bidirectional blocks).
The script also checks that the dense forward equals the pipeline's prefix-cache path (kv_cache
"extract" then "cached"), which is the shape the C engine computes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from safetensors.torch import save_file

SEED = 20261030
# (name, reference shapes (h, w) in latent tokens, target (h, w), VLM layout)
# layout: 't<n>' text tokens, 'i<k>' the image slots of reference k
CASES = [
    ("one_ref", [(4, 6)], (6, 4), ["t5", "i0", "t7"]),
    ("two_refs_text_between", [(2, 4), (4, 2)], (4, 4), ["t3", "i0", "t4", "i1", "t6"]),
    ("two_refs_adjacent", [(4, 4), (2, 6)], (4, 6), ["t4", "i0", "i1", "t5"]),
]


def build_case(tr, g, refs, tgt, layout, t):
    cfg = tr.config
    C, ctx = cfg.in_channels, cfg.context_in_dim
    mask = []
    for seg in layout:
        n = int(seg[1:])
        if seg[0] == "t":
            mask += [False] * n
        else:
            h, w = refs[n]
            mask += [True] * ((h * w) // 4)
    Lv = len(mask)
    nt = tgt[0] * tgt[1]
    emb = torch.randn(1, Lv, ctx, generator=g) * 3.0
    ref = torch.cat([torch.randn(1, h * w, C, generator=g) for h, w in refs], dim=1)
    tg = torch.randn(1, nt, C, generator=g)
    # the pipeline appends one slot per 2x2 group of target latents to the mask it hands over
    img_mask = torch.tensor(mask + [True] * (nt // 4)).unsqueeze(0)
    img_shapes = [[*[(1, h, w) for h, w in refs], (1, tgt[0], tgt[1])]]
    kw = dict(
        hidden_states=torch.cat([ref, tg], dim=1),
        timestep=torch.tensor([t]),
        encoder_hidden_states=emb,
        encoder_hidden_states_mask=torch.ones(1, Lv, dtype=torch.long),
        img_shapes=img_shapes,
        img_mask=img_mask,
        return_dict=False,
    )
    with torch.no_grad():
        dense = tr(**kw)[0][:, -nt:]
        cache = QwenImage21KVCache(len(tr.transformer_blocks))
        tr(**kw, kv_cache=cache, kv_cache_mode="extract")
        cached = tr(**kw, kv_cache=cache, kv_cache_mode="cached")[0][:, -nt:]
        # the RoPE of the whole joint sequence, as the forward builds it
        repeats = torch.where(img_mask, 4, 1)[0]
        pad = torch.repeat_interleave(img_mask[0], repeats)
        rope = torch.view_as_real(tr.pos_embed(img_shapes[0], pad, device=emb.device))
    err = (dense - cached).abs().max().item()
    assert err < 1e-4, f"dense and prefix-cache forward differ by {err}"
    return dict(emb=emb[0], img_mask=torch.tensor(mask, dtype=torch.uint8), ref=ref[0], tgt=tg[0],
                cos=rope[..., 0].contiguous(), sin=rope[..., 1].contiguous(), pred=dense[0].contiguous()), err


def main():
    model, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    tr = QwenImage21Transformer2DModel.from_pretrained(str(model / "transformer"), torch_dtype=torch.float32).eval()
    g = torch.Generator().manual_seed(SEED)
    tensors, meta = {}, []
    for k, (name, refs, tgt, layout) in enumerate(CASES):
        t = 0.55 + 0.1 * k
        tn, err = build_case(tr, g, refs, tgt, layout, t)
        for key, val in tn.items():
            tensors[f"c{k}.{key}"] = val.float() if val.dtype != torch.uint8 else val
        meta.append({"name": name, "refs": refs, "target": list(tgt), "t": t, "lv": int(tn["emb"].shape[0])})
        print(f"case{k} {name}: Lv {tn['emb'].shape[0]}, joint {tn['cos'].shape[0]} tokens, target {tgt[0] * tgt[1]}, "
              f"pred std {tn['pred'].std().item():.3f}, dense vs cache {err:.1e}")
    save_file(tensors, str(out / "ref.safetensors"))
    (out / "ref.json").write_text(json.dumps({"cases": meta}))


if __name__ == "__main__":
    main()
