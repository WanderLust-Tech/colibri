#!/usr/bin/env python3
"""Make a tiny random Gemma 4 (mixture of experts) for testing c/gemma4.c, and the
numbers a correct engine must reproduce.

Needs torch and transformers >= 5.5 (the real model implementation is the oracle):

    python c/tools/make_gemma4_tiny.py --out /tmp/gemma4_tiny

writes

  OUT/hf/          the checkpoint in the real layout (model.language_model.* names,
                   float32 safetensors, config.json with a text_config), weights ROUNDED to
                   what the container can hold
  OUT/container/   the same model after c/tools/convert_gemma4.py
  OUT/ref.json     prompts, the logits transformers produces for them, and a greedy
                   continuation, all in float32

The weights are snapped to the quantization grid before transformers sees them, so the
reference and the engine multiply the same numbers and differ only in float rounding; a
mismatch bigger than that is a bug in the engine, not quantization noise.

The model is small but has every part of the 26B-A4B that matters: sliding and full
attention layers with different head sizes, the keys-equal-values full layers, proportional
rope, the dense MLP next to the experts, the router's own scale, the layer scalar, the
logit softcap. The sliding window is shorter than the prompts, so the window is exercised.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_gemma4 as cv  # noqa: E402

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


def tiny_tokenizer(path):
    """A 512-entry tokenizer with Gemma's shape: the mark for spaces, byte fallback at the real
    ids (238 + byte), the chat control tokens at their real ids, a few merges."""
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


def randomize(model, rng):
    """Random weights with enough spread that every norm, scale and gate matters."""
    state = {}
    for name, p in model.state_dict().items():
        shape = tuple(p.shape)
        if name.endswith("layer_scalar"):
            v = rng.uniform(0.6, 1.4, shape)
        elif name.endswith("per_expert_scale"):
            v = rng.uniform(0.5, 1.5, shape)
        elif name.endswith("router.scale"):
            v = rng.uniform(0.5, 1.5, shape)
        elif name.endswith("norm.weight") or "layernorm" in name or name.endswith("_norm.weight"):
            v = rng.uniform(0.6, 1.4, shape)
        elif name.endswith("embed_tokens.weight"):
            v = rng.normal(0, 0.08, shape)     # small: the layers, not the token itself, must decide the output
        elif name.endswith("router.proj.weight"):
            v = rng.normal(0, 0.3, shape)
        else:
            v = rng.normal(0, 0.12, shape)
        state[name] = v.astype(np.float32)
    return state


def snap(name, array):
    """What the container will hold for this tensor, dequantized."""
    if name.endswith("embed_tokens.weight") or name.rsplit(".", 2)[-2:] in (["q_proj", "weight"], ["k_proj", "weight"],
            ["v_proj", "weight"], ["o_proj", "weight"], ["gate_proj", "weight"], ["up_proj", "weight"], ["down_proj", "weight"]):
        q, s = cv.quantize_int8_rows(array)
        return cv.dequantize_int8_rows(q, s)
    if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
        out = np.empty_like(array)
        for e in range(array.shape[0]):
            p, s = cv.quantize_int4_planar(array[e])
            out[e] = cv.dequantize_int4_planar(p, s)
        return out
    return array


def save_safetensors(path, tensors):
    header, offset = {}, 0
    for name, a in tensors.items():
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [offset, offset + a.nbytes]}
        offset += a.nbytes
    import struct
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for a in tensors.values():
            handle.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    rng = np.random.default_rng(args.seed)
    config = Gemma4TextConfig(**TINY)
    torch.manual_seed(args.seed)
    model = Gemma4ForCausalLM(config).to(torch.float32).eval()
    raw = randomize(model, rng)

    # The checkpoint, in the real layout: ForCausalLM's "model.X" is "model.language_model.X" there.
    ckpt = {}
    for name, a in raw.items():
        if name.startswith("lm_head"):
            continue
        ckpt[name.replace("model.", "model.language_model.", 1)] = a
    hf_dir, container = os.path.join(args.out, "hf"), os.path.join(args.out, "container")
    os.makedirs(hf_dir, exist_ok=True)
    config_json = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
                   "tie_word_embeddings": True, "text_config": {**config.to_dict(), "model_type": "gemma4_text"}}
    # keep the file small and the real names: only what the checkpoint's config carries
    config_json["text_config"] = {k: v for k, v in config_json["text_config"].items()
                                  if k in set(TINY) | {"model_type", "dtype", "use_cache", "initializer_range"}}
    config_json["text_config"]["layer_types"] = TINY["layer_types"]
    # transformers folds these two into its per-layer config; the real config.json states them
    config_json["text_config"]["global_head_dim"] = TINY["global_head_dim"]
    config_json["text_config"]["num_global_key_value_heads"] = TINY["num_global_key_value_heads"]
    with open(os.path.join(hf_dir, "config.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(config_json, handle, indent=1)
    save_safetensors(os.path.join(hf_dir, "model.safetensors"), ckpt)
    tk = tiny_tokenizer(os.path.join(hf_dir, "tokenizer.json"))
    with open(os.path.join(hf_dir, "generation_config.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106, 50], "pad_token_id": 0, "temperature": 1.0,
                   "top_k": 64, "top_p": 0.95}, handle)
    cv.convert(hf_dir, container, log=lambda *_: None)

    # The reference model gets the snapped weights.
    snapped = {name: torch.from_numpy(snap(name, a)) for name, a in raw.items() if not name.startswith("lm_head")}
    missing = model.load_state_dict(snapped, strict=False)
    assert not [k for k in missing.missing_keys if not k.startswith("lm_head")], missing
    model.tie_weights()

    # Prompts long enough to pass the sliding window (6) several times over.
    prompts = [
        [2] + [int(x) for x in rng.integers(5, 500, 23)],
        [2] + [int(x) for x in rng.integers(5, 500, 4)],
        [2] + [int(x) for x in rng.integers(5, 500, 40)],
    ]
    texts = ["hello there", "the quick brown fox", "  two  spaces\n\nand lines", "café 日本 \U0001f600",
             "<|turn>user\nhi<turn|>\n<|turn>model\n", "a<|channel>b<channel|>c"]
    out = {"config": TINY, "cases": [], "encodings": [{"text": t, "ids": tk.encode(t, add_special_tokens=False).ids} for t in texts]}
    with torch.no_grad():
        for ids in prompts:
            x = torch.tensor([ids])
            logits = model(input_ids=x).logits[0].float()                      # [T, V], softcapped
            seq = list(ids)
            for _ in range(8):                                               # a greedy continuation
                nxt = int(model(input_ids=torch.tensor([seq])).logits[0, -1].argmax())
                seq.append(nxt)
            out["cases"].append({
                "ids": ids,
                "last_logits": [float(v) for v in logits[-1]],
                "all_top": [int(v) for v in logits.argmax(-1)],
                "logit_sample": [[float(v) for v in row[:16]] for row in logits],
                "greedy": seq[len(ids):],
            })
    with open(os.path.join(args.out, "ref.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
    print(f"tiny gemma4 written to {args.out}")


if __name__ == "__main__":
    main()
