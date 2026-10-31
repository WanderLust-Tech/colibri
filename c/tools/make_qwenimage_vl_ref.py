#!/usr/bin/env python3
"""Oracle for the image side of a Qwen-Image-2.1 edit's text encoder (c/qwenimage_vit.h and the
vision-language path of c/qwenimage.c).

    make_qwenimage_vl_ref.py MODEL_DIR OUTDIR [--tower-only]

loads MODEL_DIR as QwenImage21Pipeline (float32; the tiny pipeline of make_qwenimage_tiny.py or the real
checkpoint), runs the pipeline's own condition-image encoding on synthetic RGBA pictures, and writes
OUTDIR/ref.json and OUTDIR/ref.safetensors. Each case k (a prompt and one or more pictures) stores

    c{k}.ids           i64 [T]          the processor's token ids for the whole prompt (system turn included)
    c{k}.pos           i64 [3][T]       the M-RoPE position ids get_rope_index builds for them
    c{k}.emb           f32 [Lv][hidden] the prompt embeddings the DiT reads (last layer, before the final norm,
                                        the system turn dropped), Lv = T - drop_idx
    c{k}.img_mask      u8  [Lv]         1 at the <|image_pad|> slots
and, for picture j of the case,
    c{k}.i{j}.rgba     u8  [h][w][4]    the picture as the pipeline is handed it (already at its final size)
    c{k}.i{j}.rgb      u8  [h][w][3]    that picture flattened over white, as the vision copy is
    c{k}.i{j}.pix      f32 [n][C*P*P]   the processor's patches (merge-block order; the two temporal frames
                                        of a still image are identical, so one is stored)
    c{k}.i{j}.vis      f32 [n/4][out]   the vision tower's merged tokens for this picture alone
    c{k}.i{j}.ds{m}    f32 [n/4][out]   deepstack feature m

With --tower-only the language model and the pipeline are not loaded: only the vision tower (and the
processor) of MODEL_DIR, in float32, on pictures up to 1024x1024. That is what fits next to the real
checkpoint's 8B text encoder on a 64 GB machine, and it holds the tower to real shapes (head_dim 72,
27 blocks, the 48x48 position grid, deepstack taps at 8/16/24). Such a case has no ids/pos/emb/img_mask
and says "tower_only" in ref.json.

Every picture is at least 256x256 and a multiple of 32 on both sides: inside the processor's pixel bounds,
so its resize step leaves the picture alone (the pipeline resizes before it gets here, and so must a caller).
The alpha channel is random, with solid and fully transparent regions, so the flattening's integer
rounding is exercised.
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

SEED = 20261031
# (name, prompt, picture sizes (h, w))
CASES = [
    ("one_picture", "make the sky purple", [(256, 256)]),
    ("two_pictures_wide_and_tall", "put the cat from the first picture next to the dog", [(256, 320), (320, 256)]),
    ("empty_prompt", "", [(288, 256)]),
]


def picture(rng, h, w):
    """Smooth colour fields plus noise, with an alpha that is opaque, clear or in between by region."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        127 + 100 * np.sin(xx / (11 + 5 * c) + 2 * c) * np.cos(yy / (13 + 7 * c)) for c in range(3)], axis=-1)
    rgb = np.clip(base + rng.normal(0, 12, (h, w, 3)), 0, 255).astype(np.uint8)
    alpha = rng.integers(0, 256, (h, w), dtype=np.uint8)
    alpha[: h // 3] = 255
    alpha[h // 3: h // 2, : w // 2] = 0
    return np.dstack([rgb, alpha])


def flatten(im):
    white = Image.new("RGB", im.size, (255, 255, 255))
    white.paste(im, mask=im.getchannel("A"))
    return white


TOWER_CASES = [("tower_256", [(256, 256)]), ("tower_wide_512x384", [(384, 512)]), ("tower_tall_1024x640", [(1024, 640)])]


def tower_only(model, out):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    proc = qref.load_processor(model)
    cfg = AutoConfig.from_pretrained(str(model / "text_encoder"))
    vit = Qwen3VLVisionModel(cfg.vision_config)
    sd = {}
    for f in sorted((model / "text_encoder").glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            for k in st.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual."):]] = st.get_tensor(k).float()
    vit.load_state_dict(sd)
    vit = vit.float().eval()
    P = cfg.vision_config.patch_size
    rng = np.random.default_rng(SEED)
    tensors, meta = {}, []
    for k, (name, sizes) in enumerate(TOWER_CASES):
        info = []
        for j, (h, w) in enumerate(sizes):
            arr = picture(rng, h, w)
            fl = flatten(Image.fromarray(arr))
            inp = proc.image_processor(images=[fl], return_tensors="pt")
            pix, grid = inp["pixel_values"], inp["image_grid_thw"]
            t_, gh, gw = [int(x) for x in grid[0]]
            n = gh * gw
            frames = pix.view(n, 3, 2, P, P)
            assert t_ == 1 and torch.equal(frames[:, :, 0], frames[:, :, 1])
            with torch.no_grad():
                res = vit(pix, grid_thw=grid, return_dict=True)
            tensors[f"c{k}.i{j}.rgba"] = torch.from_numpy(arr.reshape(-1)).contiguous()
            tensors[f"c{k}.i{j}.rgb"] = torch.from_numpy(np.asarray(fl).reshape(-1).copy())
            tensors[f"c{k}.i{j}.pix"] = frames[:, :, 0].reshape(n, -1).contiguous()
            tensors[f"c{k}.i{j}.vis"] = res.pooler_output.contiguous()
            for m, d in enumerate(res.deepstack_features):
                tensors[f"c{k}.i{j}.ds{m}"] = d.contiguous()
            info.append({"h": h, "w": w, "grid": [gh, gw]})
            print(f"case{k} {name}: {w}x{h}, {n} patches, {len(res.deepstack_features)} deepstack taps, "
                  f"vis std {res.pooler_output.std().item():.3f}")
        meta.append({"name": name, "images": info, "nds": len(res.deepstack_features), "tower_only": True})
    save_file(tensors, str(out / "ref.safetensors"))
    (out / "ref.json").write_text(json.dumps({"cases": meta}))


def main():
    from diffusers import QwenImage21Pipeline

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model, out = Path(args[0]), Path(args[1])
    out.mkdir(parents=True, exist_ok=True)
    if "--tower-only" in sys.argv:
        return tower_only(model, out)
    pipe = QwenImage21Pipeline.from_pretrained(str(model), torch_dtype=torch.float32,
                                               processor=qref.load_processor(model))
    pipe.set_progress_bar_config(disable=True)
    tm = pipe.text_encoder.model
    P = tm.visual.config.patch_size
    rng = np.random.default_rng(SEED)
    tensors, meta = {}, []
    for k, (name, prompt, sizes) in enumerate(CASES):
        arrays = [picture(rng, h, w) for h, w in sizes]
        pils = [Image.fromarray(a) for a in arrays]
        assert all(p.mode == "RGBA" for p in pils)
        flat = [flatten(p) for p in pils]

        # the same text the pipeline builds, to read the processor's intermediates from
        replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
        for i in range(2, len(pils) + 1):
            replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
        text = pipe.prompt_template_ti2i.replace("<image1><|vision_start|><|image_pad|><|vision_end|>", replace)
        text = text.format(prompt if prompt else " ")
        inputs = pipe.processor(text=[text], images=flat, padding=True, padding_side="left", return_tensors="pt")
        ids, grid = inputs.input_ids, inputs.image_grid_thw
        assert bool(inputs.attention_mask.all()), "a single prompt is never padded"
        pos, _ = tm.get_rope_index(ids, inputs.mm_token_type_ids, grid, attention_mask=inputs.attention_mask)

        with torch.no_grad():
            emb, _, pad = pipe._get_qwen_prompt_embeds([prompt], pils, torch.device("cpu"))
        drop = pipe._drop_idx
        Lv = ids.shape[1] - drop
        assert emb.shape[1] == Lv and pad.shape[1] == Lv

        T = ids.shape[1]
        tensors[f"c{k}.ids"] = ids[0].contiguous()
        tensors[f"c{k}.pos"] = pos[:, 0].contiguous()
        tensors[f"c{k}.emb"] = emb[0].contiguous()
        tensors[f"c{k}.img_mask"] = pad[0].to(torch.uint8).contiguous()

        off, info = 0, []
        for j, (arr, fl) in enumerate(zip(arrays, flat)):
            t_, gh, gw = [int(x) for x in grid[j]]
            assert t_ == 1
            n = gh * gw
            pix = inputs.pixel_values[off: off + n]
            off += n
            frames = pix.view(n, 3, 2, P, P)
            assert torch.equal(frames[:, :, 0], frames[:, :, 1]), "a still image's two frames must match"
            with torch.no_grad():
                res = tm.get_image_features(pix, grid[j: j + 1])
            vis = res.pooler_output[0]
            tensors[f"c{k}.i{j}.rgba"] = torch.from_numpy(arr.reshape(-1)).contiguous()
            tensors[f"c{k}.i{j}.rgb"] = torch.from_numpy(np.asarray(fl).reshape(-1).copy())
            tensors[f"c{k}.i{j}.pix"] = frames[:, :, 0].reshape(n, -1).contiguous()
            tensors[f"c{k}.i{j}.vis"] = vis.contiguous()
            for m, d in enumerate(res.deepstack_features):
                tensors[f"c{k}.i{j}.ds{m}"] = d.contiguous()
            info.append({"h": arr.shape[0], "w": arr.shape[1], "grid": [gh, gw]})
        nds = len(res.deepstack_features)
        meta.append({"name": name, "prompt": prompt, "images": info, "drop_idx": drop, "t": T, "lv": Lv, "nds": nds})
        print(f"case{k} {name}: {len(pils)} picture(s), {T} tokens (drop {drop}), {nds} deepstack taps, "
              f"emb std {emb.std().item():.3f}, vis std {vis.std().item():.3f}")
    save_file(tensors, str(out / "ref.safetensors"))
    (out / "ref.json").write_text(json.dumps({"cases": meta}))


if __name__ == "__main__":
    main()
