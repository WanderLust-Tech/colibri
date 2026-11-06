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
    attention_k_eq_v=True, enable_moe_block=True, hidden_size_per_layer_input=0,
    num_kv_shared_layers=0, tie_word_embeddings=True, attention_bias=False,
    layer_types=["sliding_attention", "sliding_attention", "full_attention",
                 "sliding_attention", "sliding_attention", "full_attention"],
    rope_parameters={
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    bos_token_id=2, eos_token_id=1, pad_token_id=0,
)

SPECIAL = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<|tool>": 46, "<tool|>": 47, "<|tool_call>": 48,
           "<tool_call|>": 49, "<|tool_response>": 50, "<tool_response|>": 51, '<|"|>': 52, "<|think|>": 98,
           "<|channel>": 100, "<channel|>": 101, "<|turn>": 105, "<turn|>": 106}

TEXTS = ["hello there", "the quick brown fox", "  two  spaces\n\nand lines", "café 日本 \U0001f600",
         "<|turn>user\nhi<turn|>\n<|turn>model\n", "a<|channel>b<channel|>c"]

P = "model.language_model."


def shapes(cfg=TINY):
    """name -> shape, in the real checkpoint's naming."""
    H, V, E = cfg["hidden_size"], cfg["vocab_size"], cfg["num_experts"]
    Fd, Fm, nh = cfg["intermediate_size"], cfg["moe_intermediate_size"], cfg["num_attention_heads"]
    out = {P + "embed_tokens.weight": (V, H), P + "norm.weight": (H,)}
    for l, kind in enumerate(cfg["layer_types"]):
        full = kind == "full_attention"
        hd = cfg["global_head_dim"] if full else cfg["head_dim"]
        nkv = cfg["num_global_key_value_heads"] if full else cfg["num_key_value_heads"]
        p = f"{P}layers.{l}."
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm", "pre_feedforward_layernorm_2",
                  "post_feedforward_layernorm_1", "post_feedforward_layernorm_2"):
            out[p + n + ".weight"] = (H,)
        out[p + "layer_scalar"] = (1,)
        out[p + "self_attn.q_proj.weight"] = (nh * hd, H)
        out[p + "self_attn.k_proj.weight"] = (nkv * hd, H)
        if not full:
            out[p + "self_attn.v_proj.weight"] = (nkv * hd, H)
        out[p + "self_attn.o_proj.weight"] = (H, nh * hd)
        out[p + "self_attn.q_norm.weight"] = (hd,)
        out[p + "self_attn.k_norm.weight"] = (hd,)
        out[p + "mlp.gate_proj.weight"] = (Fd, H)
        out[p + "mlp.up_proj.weight"] = (Fd, H)
        out[p + "mlp.down_proj.weight"] = (H, Fd)
        out[p + "router.proj.weight"] = (E, H)
        out[p + "router.scale"] = (H,)
        out[p + "router.per_expert_scale"] = (E,)
        out[p + "experts.gate_up_proj"] = (E, 2 * Fm, H)
        out[p + "experts.down_proj"] = (E, H, Fm)
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
        elif name.endswith("router.proj.weight"):
            v = rng.normal(0, 0.3, shape)
        else:
            v = rng.normal(0, 0.12, shape)
        state[name] = v.astype(np.float32)
    return state


def snap(name, array):
    """What the container will hold for this tensor, dequantized."""
    if name.endswith("embed_tokens.weight") or name.rsplit(".", 2)[-2] in (
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
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


def config_json():
    text = {k: v for k, v in TINY.items()}
    text["model_type"] = "gemma4_text"
    return {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
            "tie_word_embeddings": True, "text_config": text}


def write_committed_files(fixture=FIXTURE):
    os.makedirs(fixture, exist_ok=True)
    with open(os.path.join(fixture, "config.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(config_json(), handle, indent=1)
    with open(os.path.join(fixture, "generation_config.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106, 50], "pad_token_id": 0, "temperature": 1.0,
                   "top_k": 64, "top_p": 0.95}, handle)


def build_container(fixture=FIXTURE):
    """The container, from the committed config and tokenizer. numpy only."""
    import tempfile
    state = random_weights()
    with tempfile.TemporaryDirectory() as src:
        for name in ("config.json", "generation_config.json", "tokenizer.json"):
            shutil.copyfile(os.path.join(fixture, name), os.path.join(src, name))
        save_f32_safetensors(os.path.join(src, "model.safetensors"), state)
        cv.convert(src, fixture, log=lambda *_: None)


def reference(fixture=FIXTURE):
    """ref.json: what transformers computes for the (snapped) weights."""
    import torch
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from tokenizers import Tokenizer

    torch.manual_seed(SEED)
    model = Gemma4ForCausalLM(Gemma4TextConfig(**TINY)).to(torch.float32).eval()
    snapped = {}
    for name, a in random_weights().items():
        snapped[name.replace("model.language_model.", "model.", 1)] = torch.from_numpy(snap(name, a))
    missing = model.load_state_dict(snapped, strict=False)
    assert not [k for k in missing.missing_keys if not k.startswith("lm_head")], missing
    assert not missing.unexpected_keys, missing
    model.tie_weights()
    rng = np.random.default_rng(SEED + 1)
    prompts = [
        [2] + [int(x) for x in rng.integers(5, 500, 23)],
        [2] + [int(x) for x in rng.integers(5, 500, 4)],
        [2] + [int(x) for x in rng.integers(5, 500, 40)],
    ]
    tk = Tokenizer.from_file(os.path.join(fixture, "tokenizer.json"))
    out = {"config": TINY, "cases": [],
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
    with open(os.path.join(fixture, "ref.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default=FIXTURE)
    ap.add_argument("--reference", action="store_true", help="recompute ref.json with transformers")
    ap.add_argument("--tokenizer", action="store_true", help="rewrite tokenizer.json (needs tokenizers)")
    args = ap.parse_args()
    write_committed_files(args.fixture)
    if args.tokenizer or not os.path.exists(os.path.join(args.fixture, "tokenizer.json")):
        tiny_tokenizer(os.path.join(args.fixture, "tokenizer.json"))
    build_container(args.fixture)
    if args.reference:
        reference(args.fixture)
    print(f"tiny gemma4 in {args.fixture}")


if __name__ == "__main__":
    main()
