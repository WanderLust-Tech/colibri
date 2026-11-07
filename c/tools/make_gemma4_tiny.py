#!/usr/bin/env python3
"""A tiny random Gemma 4 (mixture of experts) for testing c/gemma4.c, and the numbers a correct
engine must reproduce.

    python c/tools/make_gemma4_tiny.py                  rebuild the container (numpy only)
    python c/tools/make_gemma4_tiny.py --reference      also recompute ref.json (needs torch and
                                                        transformers >= 5.5: the real model
                                                        implementation is the oracle)

Everything lives in c/gemma4_tiny. Committed: config.json, generation_config.json, tokenizer.json
(a 512-entry tokenizer with Gemma's shape), ref.json. Rebuilt on demand and ignored by git, like every
other weight file here: the container (model-*.safetensors, index, gemma4_meta.json).

The weights come from a fixed recipe over numpy's generator, in sorted tensor-name order, so they are the
same on every machine and ref.json stays valid without torch. The reference model is given the weights
ALREADY ROUNDED to what the container holds (int8 rows, int4 groups), so the engine and transformers
multiply the same numbers and differ only in float rounding; a mismatch bigger than that is a bug in the
engine, not quantization noise.

The model is small but has every part of the 26B-A4B that matters: sliding and full attention layers with
different head sizes, the keys-equal-values full layers, proportional rope, the dense MLP next to the
experts, the router's own scale, the layer scalar, the logit softcap. The sliding window (6) is shorter than
the prompts, so the window is exercised.
"""
import argparse
import copy
import json
import os
import shutil
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert_gemma4 as cv  # noqa: E402

FIXTURE = os.path.join(HERE, "..", "gemma4_tiny")
SEED = 7

TINY = dict(
    vocab_size=512, hidden_size=64, intermediate_size=64, moe_intermediate_size=64,
    num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    global_head_dim=64, num_global_key_value_heads=1, num_experts=8, top_k_experts=3,
    sliding_window=6, max_position_embeddings=512, rms_norm_eps=1e-6,
    final_logit_softcapping=30.0, hidden_activation="gelu_pytorch_tanh",
    attention_k_eq_v=True, enable_moe_block=True, hidden_size_per_layer_input=0, use_bidirectional_attention="vision",
    num_kv_shared_layers=0, tie_word_embeddings=True, attention_bias=False,
    layer_types=["sliding_attention", "sliding_attention", "full_attention",
                 "sliding_attention", "sliding_attention", "full_attention"],
    rope_parameters={
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    bos_token_id=2, eos_token_id=1, pad_token_id=0,
)

# A tiny vision tower: every part of the real one (patch embedding with two position tables, axial rope over
# two halves of the head, per-head norms, sandwich norms, gelu MLP, 3 x 3 pooling, standardisation, the
# projector) at a size a test can run in transformers.
TINY_VISION = dict(
    hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    head_dim=8, hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6, pooling_kernel_size=3, patch_size=4,
    position_embedding_size=64, use_clipped_linears=False, standardize=True,
    rope_parameters={"rope_theta": 100.0, "rope_type": "default"}, max_position_embeddings=1024,
)
IMAGE_IDS = dict(image_token_id=500, boi_token_id=501, eoi_token_id=502)

# The other sizes of the family, as tiny as the first: the 31B (dense: the MLP alone, keys are values on the full layers) and the
# E2B/E4B (per-layer embeddings, the last two layers reading the keys and values of earlier ones, a double-wide MLP there, and
# clipped vision linears). Each variant has its own directory, its own seeds, and its own reference.
_DENSE = dict(enable_moe_block=False, num_experts=None, top_k_experts=None, moe_intermediate_size=None)
TINY_DENSE = dict(TINY, intermediate_size=128, **_DENSE)
TINY_PLE = dict(TINY, intermediate_size=96, attention_k_eq_v=False, hidden_size_per_layer_input=16,
                vocab_size_per_layer_input=512, num_kv_shared_layers=2, use_double_wide_mlp=True, **_DENSE)
TINY_VISION_CLIPPED = dict(TINY_VISION, use_clipped_linears=True, standardize=False)
VARIANTS = {
    "moe": {"dir": FIXTURE, "text": TINY, "vision": TINY_VISION, "seed": SEED},
    "dense": {"dir": FIXTURE + "_dense", "text": TINY_DENSE, "vision": TINY_VISION, "seed": 11},
    "ple": {"dir": FIXTURE + "_ple", "text": TINY_PLE, "vision": TINY_VISION_CLIPPED, "seed": 13},
}

SPECIAL = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<|tool>": 46, "<tool|>": 47, "<|tool_call>": 48,
           "<tool_call|>": 49, "<|tool_response>": 50, "<tool_response|>": 51, '<|"|>': 52, "<|think|>": 98,
           "<|channel>": 100, "<channel|>": 101, "<|turn>": 105, "<turn|>": 106,
           "<|image|>": 500, "<|image>": 501, "<image|>": 502}

TEXTS = ["hello there", "the quick brown fox", "  two  spaces\n\nand lines", "café 日本 \U0001f600",
         "<|turn>user\nhi<turn|>\n<|turn>model\n", "a<|channel>b<channel|>c"]

P = "model.language_model."


def shapes(cfg=TINY):
    """name -> shape, in the real checkpoint's naming."""
    H, V, Fd, nh = cfg["hidden_size"], cfg["vocab_size"], cfg["intermediate_size"], cfg["num_attention_heads"]
    moe = cfg.get("enable_moe_block", False)
    E, Fm = (cfg["num_experts"], cfg["moe_intermediate_size"]) if moe else (0, 0)
    L, Pd = cfg["num_hidden_layers"], cfg.get("hidden_size_per_layer_input", 0)
    shared = cfg.get("num_kv_shared_layers", 0)
    first_shared = L - shared if shared else L
    k_eq_v = cfg.get("attention_k_eq_v", False)
    out = {P + "embed_tokens.weight": (V, H), P + "norm.weight": (H,)}
    if Pd:
        out[P + "embed_tokens_per_layer.weight"] = (cfg["vocab_size_per_layer_input"], L * Pd)
        out[P + "per_layer_model_projection.weight"] = (L * Pd, H)
        out[P + "per_layer_projection_norm.weight"] = (Pd,)
    for l, kind in enumerate(cfg["layer_types"]):
        full = kind == "full_attention"
        own = l < first_shared                                   # a layer past first_shared reads another layer's keys and values
        ff = Fd * (2 if cfg.get("use_double_wide_mlp") and not own else 1)
        hd = cfg["global_head_dim"] if full else cfg["head_dim"]
        nkv = cfg["num_global_key_value_heads"] if (full and k_eq_v) else cfg["num_key_value_heads"]
        p = f"{P}layers.{l}."
        norms = ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]
        if moe:
            norms += ["pre_feedforward_layernorm_2", "post_feedforward_layernorm_1", "post_feedforward_layernorm_2"]
        for n in norms:
            out[p + n + ".weight"] = (H,)
        out[p + "layer_scalar"] = (1,)
        out[p + "self_attn.q_proj.weight"] = (nh * hd, H)
        out[p + "self_attn.o_proj.weight"] = (H, nh * hd)
        out[p + "self_attn.q_norm.weight"] = (hd,)
        if own:
            out[p + "self_attn.k_proj.weight"] = (nkv * hd, H)
            out[p + "self_attn.k_norm.weight"] = (hd,)
            if not (full and k_eq_v):
                out[p + "self_attn.v_proj.weight"] = (nkv * hd, H)
        out[p + "mlp.gate_proj.weight"] = (ff, H)
        out[p + "mlp.up_proj.weight"] = (ff, H)
        out[p + "mlp.down_proj.weight"] = (H, ff)
        if moe:
            out[p + "router.proj.weight"] = (E, H)
            out[p + "router.scale"] = (H,)
            out[p + "router.per_expert_scale"] = (E,)
            out[p + "experts.gate_up_proj"] = (E, 2 * Fm, H)
            out[p + "experts.down_proj"] = (E, H, Fm)
        if Pd:
            out[p + "per_layer_input_gate.weight"] = (Pd, H)
            out[p + "per_layer_projection.weight"] = (H, Pd)
            out[p + "post_per_layer_input_norm.weight"] = (H,)
    return out


def random_weights(cfg=TINY, seed=SEED):
    """Random weights with enough spread that every norm, scale and gate matters."""
    rng = np.random.default_rng(seed)
    state = {}
    for name in sorted(shapes(cfg)):
        shape = shapes(cfg)[name]
        if name.endswith("layer_scalar"):
            v = rng.uniform(0.6, 1.4, shape)
        elif name.endswith("per_expert_scale") or name.endswith("router.scale"):
            v = rng.uniform(0.5, 1.5, shape)
        elif name.endswith("norm.weight") or "layernorm" in name or name.endswith("_norm.weight"):
            v = rng.uniform(0.6, 1.4, shape)
        elif name.endswith("embed_tokens.weight"):
            v = rng.normal(0, 0.08, shape)      # small: the layers, not the token itself, must decide the output
        elif name.endswith("embed_tokens_per_layer.weight"):
            v = rng.normal(0, 0.05, shape)
        elif name.endswith("router.proj.weight"):
            v = rng.normal(0, 0.3, shape)
        else:
            v = rng.normal(0, 0.12, shape)
        state[name] = v.astype(np.float32)
    return state


V = "model.vision_tower."


LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def vision_shapes(vcfg=TINY_VISION, text_hidden=TINY["hidden_size"]):
    """The vision tower's tensors, in the real checkpoint's naming."""
    D, F, nh, hd = vcfg["hidden_size"], vcfg["intermediate_size"], vcfg["num_attention_heads"], vcfg["head_dim"]
    nkv, patch = vcfg["num_key_value_heads"], vcfg["patch_size"]
    out = {V + "patch_embedder.input_proj.weight": (D, 3 * patch * patch),
           V + "patch_embedder.position_embedding_table": (2, vcfg["position_embedding_size"], D),
           "model.embed_vision.embedding_projection.weight": (text_hidden, D)}
    if vcfg["standardize"]:
        out[V + "std_bias"] = (D,)
        out[V + "std_scale"] = (D,)
    for l in range(vcfg["num_hidden_layers"]):
        p = f"{V}encoder.layers.{l}."
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            out[p + n + ".weight"] = (D,)
        out[p + "self_attn.q_proj.linear.weight"] = (nh * hd, D)
        out[p + "self_attn.k_proj.linear.weight"] = (nkv * hd, D)
        out[p + "self_attn.v_proj.linear.weight"] = (nkv * hd, D)
        out[p + "self_attn.o_proj.linear.weight"] = (D, nh * hd)
        out[p + "self_attn.q_norm.weight"] = (hd,)
        out[p + "self_attn.k_norm.weight"] = (hd,)
        out[p + "mlp.gate_proj.linear.weight"] = (F, D)
        out[p + "mlp.up_proj.linear.weight"] = (F, D)
        out[p + "mlp.down_proj.linear.weight"] = (D, F)
        if vcfg["use_clipped_linears"]:
            for module in LINEARS:
                for limit in ("input_min", "input_max", "output_min", "output_max"):
                    out[p + module + "." + limit] = ()
    return out


def random_vision_weights(seed=SEED + 100, vcfg=TINY_VISION, text_hidden=TINY["hidden_size"]):
    """Vision weights from their own generator, so adding them did not move a single text weight (and
    ref.json, computed before they existed, is still the text model's reference)."""
    rng = np.random.default_rng(seed)
    state = {}
    shapes_ = vision_shapes(vcfg, text_hidden)
    for name in sorted(shapes_):
        shape = shapes_[name]
        if name.endswith(("input_min", "output_min")):          # a clipped linear's limits: tight enough to clip
            v = -rng.uniform(0.8, 1.6) if name.endswith("input_min") else -rng.uniform(1.0, 2.0)
        elif name.endswith(("input_max", "output_max")):
            v = rng.uniform(0.8, 1.6) if name.endswith("input_max") else rng.uniform(1.0, 2.0)
        elif name.endswith("std_scale"):
            v = rng.uniform(0.5, 1.5, shape)
        elif name.endswith("std_bias"):
            v = rng.normal(0, 0.5, shape)
        elif "norm" in name:
            v = rng.uniform(0.6, 1.4, shape)
        elif name.endswith("position_embedding_table"):
            v = rng.normal(0, 0.5, shape)
        elif name.endswith("input_proj.weight"):
            v = rng.normal(0, 0.3, shape)
        else:
            v = rng.normal(0, 0.2, shape)
        state[name] = np.asarray(v, dtype=np.float32)
    return state


def snap(name, array):
    """What the container will hold for this tensor, dequantized."""
    if name.startswith(("model.vision_tower.", "model.embed_vision.")):
        if name.endswith(".linear.weight") or name.endswith("input_proj.weight") or name.endswith("embedding_projection.weight"):
            q, s = cv.quantize_int8_rows(array)
            return cv.dequantize_int8_rows(q, s)
        return array
    if name.endswith(("embed_tokens.weight", "embed_tokens_per_layer.weight", "per_layer_model_projection.weight")) or             name.rsplit(".", 2)[-2] in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                        "per_layer_input_gate", "per_layer_projection"):
        q, s = cv.quantize_int8_rows(array)
        return cv.dequantize_int8_rows(q, s)
    if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
        out = np.empty_like(array)
        for e in range(array.shape[0]):
            p, s = cv.quantize_int4_planar(array[e])
            out[e] = cv.dequantize_int4_planar(p, s)
        return out
    return array


def save_f32_safetensors(path, tensors):
    header, offset = {}, 0
    for name, a in tensors.items():
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [offset, offset + a.nbytes]}
        offset += a.nbytes
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for a in tensors.values():
            handle.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())


def tiny_tokenizer(path):
    """A 512-entry tokenizer with Gemma's shape: the mark for spaces, byte fallback at the real
    ids (238 + byte), the chat control tokens at their real ids, a few merges. Needs `tokenizers`; the
    result is committed, so only whoever changes it needs the library."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders
    vocab = dict(SPECIAL)
    for b in range(256):
        vocab["<0x%02X>" % b] = 238 + b
    free = iter(i for i in range(512) if i not in set(vocab.values()))
    mark = "▁"                                   # the mark Gemma writes for a space
    chars = [mark, "\n"] + [chr(c) for c in range(33, 127) if chr(c) not in "<"]
    for ch in chars:
        vocab[ch] = next(free)
    merges = []
    for pair in [(mark, "t"), ("h", "e"), (mark + "t", "he"), ("i", "n"), ("e", "r"), (mark, "a"), (mark, "s"),
                 ("o", "n"), (mark + "a", "n"), ("t", "h"), ("e", "n"), (mark, "w"), ("a", "t"), (mark, "o"),
                 (mark, "i"), ("o", "u"), ("r", "e"), (mark + "i", "s"), ("\n", "\n")]:
        merged = pair[0] + pair[1]
        if pair[0] in vocab and pair[1] in vocab and merged not in vocab:
            vocab[merged] = next(free)
            merges.append(pair)
    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>", fuse_unk=True, byte_fallback=True))
    tk.normalizer = normalizers.Replace(" ", mark)
    tk.pre_tokenizer = pre_tokenizers.Split(" ", "merged_with_previous")
    tk.decoder = decoders.Sequence([decoders.Replace(mark, " "), decoders.ByteFallback(), decoders.Fuse()])
    tk.add_special_tokens(list(SPECIAL))
    tk.save(path)
    return tk


def config_json(variant="moe"):
    text = {k: v for k, v in VARIANTS[variant]["text"].items()}
    text["model_type"] = "gemma4_text"
    vision = dict(VARIANTS[variant]["vision"], model_type="gemma4_vision")
    return {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
            "tie_word_embeddings": True, "text_config": text, "vision_config": vision,
            "vision_soft_tokens_per_image": 12, **IMAGE_IDS}


def write_committed_files(fixture=FIXTURE, variant="moe"):
    os.makedirs(fixture, exist_ok=True)
    with open(os.path.join(fixture, "config.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(config_json(variant), handle, indent=1)
    with open(os.path.join(fixture, "generation_config.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106, 50], "pad_token_id": 0, "temperature": 1.0,
                   "top_k": 64, "top_p": 0.95}, handle)


def build_container(fixture=FIXTURE, variant="moe"):
    """The container, from the committed config and tokenizer (a variant borrows the first one's tokenizer). numpy only."""
    import tempfile
    var = VARIANTS[variant]
    state = random_weights(var["text"], var["seed"])
    state.update(random_vision_weights(var["seed"] + 100, var["vision"], var["text"]["hidden_size"]))
    with tempfile.TemporaryDirectory() as src:
        for name in ("config.json", "generation_config.json", "tokenizer.json"):
            have = os.path.join(fixture, name)
            shutil.copyfile(have if os.path.exists(have) else os.path.join(FIXTURE, name), os.path.join(src, name))
        save_f32_safetensors(os.path.join(src, "model.safetensors"), state)
        cv.convert(src, fixture, log=lambda *_: None)


def reference(fixture=FIXTURE, variant="moe"):
    """ref.json: what transformers computes for the (snapped) weights."""
    import torch
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from tokenizers import Tokenizer

    var = VARIANTS[variant]
    seed = var["seed"]
    torch.manual_seed(seed)
    model = Gemma4ForCausalLM(Gemma4TextConfig(**copy.deepcopy(var["text"]))).to(torch.float32).eval()     # a copy: the library edits what it is given
    snapped = {}
    for name, a in random_weights(var["text"], seed).items():
        snapped[name.replace("model.language_model.", "model.", 1)] = torch.from_numpy(snap(name, a))
    missing = model.load_state_dict(snapped, strict=False)
    assert not [k for k in missing.missing_keys if not k.startswith("lm_head")], missing
    assert not missing.unexpected_keys, missing
    model.tie_weights()
    rng = np.random.default_rng(seed + 1)
    prompts = [
        [2] + [int(x) for x in rng.integers(5, 500, 23)],
        [2] + [int(x) for x in rng.integers(5, 500, 4)],
        [2] + [int(x) for x in rng.integers(5, 500, 40)],
    ]
    tk = Tokenizer.from_file(os.path.join(FIXTURE, "tokenizer.json"))
    out = {"config": var["text"], "cases": [],
           "encodings": [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in TEXTS]}
    with torch.no_grad():
        for ids in prompts:
            logits = model(input_ids=torch.tensor([ids])).logits[0].float()      # [T, V], softcapped
            seq = list(ids)
            for _ in range(8):                                                    # a greedy continuation
                seq.append(int(model(input_ids=torch.tensor([seq])).logits[0, -1].argmax()))
            out["cases"].append({
                "ids": ids,
                "last_logits": [float(v) for v in logits[-1]],
                "all_top": [int(v) for v in logits.argmax(-1)],
                "logit_sample": [[float(v) for v in row[:16]] for row in logits],
                "greedy": seq[len(ids):],
            })
    out["vision"] = vision_reference(tk, variant)
    with open(os.path.join(fixture, "ref.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)


def image_patches(pixels, patch=TINY_VISION["patch_size"]):
    """(H, W, 3) uint8 -> [(H/patch) * (W/patch), patch * patch * 3] in the order the real processor makes them:
    row-major over the patch grid, each patch flattened as (row, column, channel)."""
    h, w, _ = pixels.shape
    gh, gw = h // patch, w // patch
    return (pixels.reshape(gh, patch, gw, patch, 3).transpose(0, 2, 1, 3, 4).reshape(gh * gw, patch * patch * 3), gh, gw)


# (height, width) in pixels of the images the vision cases use: 6 x 9 patches (6 soft tokens), 3 x 3 (1 token)
# and 9 x 12 (12 tokens, longer than the sliding window of 6)
VISION_IMAGES = [(24, 36), (12, 12), (36, 48)]
VISION_PROMPTS = [
    # (image indices, text ids around them): the ids list holds None where an image goes
    ([0], [2, 11, 12, None, 13, 14, 15]),
    ([0, 1], [2, 11, None, 12, 13, None, 14, 15, 16]),
    ([2], [2] + [20 + i for i in range(9)] + [None] + [40 + i for i in range(5)]),
    ([2, 0], [2, None, 30, 31, 32, None]),
]


def vision_cases(variant="moe"):
    """The images (random pixels) and the prompts that use them, as ids with the placeholders expanded."""
    rng = np.random.default_rng(VARIANTS[variant]["seed"] + 2)
    images = []
    for h, w in VISION_IMAGES:
        base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        images.append(base)
    prompts = []
    k = TINY_VISION["pooling_kernel_size"]
    for which, layout in VISION_PROMPTS:
        ids, spans, it = [], [], iter(which)
        for item in layout:
            if item is None:
                img = images[next(it)]
                tokens = (img.shape[0] // TINY_VISION["patch_size"] // k) * (img.shape[1] // TINY_VISION["patch_size"] // k)
                ids.append(IMAGE_IDS["boi_token_id"])
                spans.append((len(ids), tokens))
                ids.extend([IMAGE_IDS["image_token_id"]] * tokens)
                ids.append(IMAGE_IDS["eoi_token_id"])
            else:
                ids.append(item)
        prompts.append({"ids": ids, "images": which, "spans": spans})
    return images, prompts


def vision_reference(tk, variant="moe"):
    """What transformers' Gemma4ForConditionalGeneration computes for prompts with pictures in them: the soft
    tokens the tower produces, and the logits of the whole model, with the bidirectional attention inside an
    image that the sliding layers apply. Images are padded to the longest, as the real processor does, so the
    padding path is the oracle's and the engine (which pads nothing) must agree with it."""
    import torch
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig, Gemma4VisionConfig

    var = VARIANTS[variant]
    torch.manual_seed(var["seed"])
    cfg = Gemma4Config(text_config=Gemma4TextConfig(**copy.deepcopy(var["text"])),
                       vision_config=Gemma4VisionConfig(**copy.deepcopy(var["vision"])), **IMAGE_IDS)
    model = Gemma4ForConditionalGeneration(cfg).to(torch.float32).eval()
    state = {}
    for name, a in random_weights(var["text"], var["seed"]).items():
        state[name] = torch.from_numpy(snap(name, a))
    for name, a in random_vision_weights(var["seed"] + 100, var["vision"], var["text"]["hidden_size"]).items():
        state[name] = torch.from_numpy(snap(name, a))
    missing = model.load_state_dict(state, strict=False)
    assert not [k for k in missing.missing_keys if not k.startswith("lm_head")], missing.missing_keys
    assert not missing.unexpected_keys, missing.unexpected_keys
    model.tie_weights()
    images, prompts = vision_cases(variant)
    patches = [image_patches(img) for img in images]
    # hex: the patches as the engine takes them (the IMAGE frame's payload); pixels: the picture they came from
    out = {"images": [{"gh": gh, "gw": gw, "hex": np.ascontiguousarray(p).tobytes().hex(),
                       "height": img.shape[0], "width": img.shape[1], "pixels": img.reshape(-1).tobytes().hex()}
                      for img, (p, gh, gw) in zip(images, patches)], "cases": []}
    with torch.no_grad():
        for case in prompts:
            use = [patches[i] for i in case["images"]]
            longest = max(p.shape[0] for p, _, _ in use)
            pix = torch.zeros(len(use), longest, use[0][0].shape[1])
            pos = torch.full((len(use), longest, 2), -1, dtype=torch.long)
            for i, (p, gh, gw) in enumerate(use):
                pix[i, :p.shape[0]] = torch.from_numpy(p.astype(np.float32) * (1 / 255))
                ys, xs = np.divmod(np.arange(gh * gw), gw)
                pos[i, :p.shape[0], 0] = torch.from_numpy(xs)
                pos[i, :p.shape[0], 1] = torch.from_numpy(ys)
            ids = case["ids"]
            kinds = [1 if t == IMAGE_IDS["image_token_id"] else 0 for t in ids]

            def run(seq):
                tt = torch.tensor([kinds + [0] * (len(seq) - len(kinds))])
                return model(input_ids=torch.tensor([seq]), pixel_values=pix, image_position_ids=pos,
                             mm_token_type_ids=tt).logits[0].float()

            features = model.get_image_features(pixel_values=pix, image_position_ids=pos).pooler_output
            logits = run(ids)
            seq = list(ids)
            for _ in range(6):
                seq.append(int(run(seq)[-1].argmax()))
            out["cases"].append({
                "ids": ids, "images": case["images"], "spans": case["spans"],
                "features": [[float(v) for v in f.reshape(-1)] for f in features],
                "last_logits": [float(v) for v in logits[-1]],
                "all_top": [int(v) for v in logits.argmax(-1)],
                "logit_sample": [[float(v) for v in row[:16]] for row in logits],
                "greedy": seq[len(ids):],
            })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS) + ["all"], default="all",
                    help="moe (26B-A4B), dense (31B) or ple (E2B/E4B)")
    ap.add_argument("--reference", action="store_true", help="recompute ref.json with transformers")
    ap.add_argument("--tokenizer", action="store_true", help="rewrite tokenizer.json (needs tokenizers)")
    args = ap.parse_args()
    for name in (sorted(VARIANTS) if args.variant == "all" else [args.variant]):
        fixture = VARIANTS[name]["dir"]
        write_committed_files(fixture, name)
        if name == "moe" and (args.tokenizer or not os.path.exists(os.path.join(fixture, "tokenizer.json"))):
            tiny_tokenizer(os.path.join(fixture, "tokenizer.json"))
        build_container(fixture, name)
        if args.reference:
            reference(fixture, name)
        print(f"tiny gemma4 ({name}) in {fixture}")


if __name__ == "__main__":
    main()
