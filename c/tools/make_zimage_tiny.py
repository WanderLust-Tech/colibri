#!/usr/bin/env python3
"""A tiny random Z-Image pipeline and its float32 reference.

    python3 tools/make_zimage_tiny.py OUTDIR [--size 64x96] [--steps 3]

Writes OUTDIR in the diffusers layout of the real checkpoint (model_index.json, tokenizer/, text_encoder/,
transformer/, vae/, scheduler/) with the real classes at toy widths, then runs tools/zimage_ref.py on it and leaves
OUTDIR/ref/ref.safetensors and OUTDIR/ref/prompt_text.txt. The C engine loads OUTDIR exactly like the real model:

    COLI_IMG_BITS=16 COLI_IMG_ACT8=0 ./zimage --model OUTDIR --ref OUTDIR/ref

What is tiny and what is not:
  text_encoder  Qwen3ForCausalLM (loaded as Qwen3Model by the pipeline), hidden 64, 3 layers, GQA 4/2, head_dim 16,
                MLP 128, rope_theta 1e6, tied embeddings, bf16 in several shards plus an index
  transformer   ZImageTransformer2DModel, 3 blocks, 2 + 2 refiner blocks of one layer each (n_refiner_layers 1),
                2 heads x 32, RoPE axes [8, 12, 12], caption width 64; float32 in several shards plus an index (as the
                real checkpoint is). The timestep embedding is min(dim, 256) = 64 wide, not 256: a hard-coded 256
                anywhere shows up
  vae           AutoencoderKL, four blocks of 8/16/32/32 channels, 4 groups, 16 latent channels, bf16
  tokenizer     byte-level BPE over the 256 bytes with a few merges, the Qwen pre-tokenizer and specials, and the
                Qwen3 chat template reduced to the branches the pipeline reaches. No downloads.
  scheduler     the real FlowMatchEulerDiscreteScheduler config (fixed shift 3).

Every weight is random (seeded), norm scales and pad tokens included, so a port that forgets a norm weight or a pad
token fails. The default 64x96 picture has 4 x 6 = 24 image tokens, which the transformer pads to 32, and the prompt's
tokens are padded to 32 as well: both padding paths are exercised.

Reference stack: diffusers >= 0.43 (ZImagePipeline), transformers 5, torch 2.14+ (CPU), tokenizers, safetensors, numpy,
Pillow."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

SEED = 20261118
PROMPT = "a red fox sitting in the snow, photograph"

# ---------------------------------------------------------------- tokenizer (the Qwen2 BPE, tiny)
MERGES = [("Ġ", "t"), ("h", "e"), ("i", "n"), ("Ġt", "he"), ("o", "w"), ("Ġ", "s"), ("Ġs", "n"),
          ("Ġsn", "ow"), ("e", "d"), ("Ġ", "a"), ("r", "e")]
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
            "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"]
VOCAB_SIZE = 288
QWEN_SPLIT_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+|\s+(?!\S)|\s+")
# Qwen3's chat template reduced to what the pipeline renders: user turns, the generation prompt, and the empty think
# block only when thinking is switched off (the pipeline leaves it on).
CHAT_TEMPLATE = r"""{%- for message in messages %}
    {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
"""


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [(b, chr(c)) for b, c in zip(bs, cs)]


def write_tokenizer(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    vocab = {ch: i for i, (_, ch) in enumerate(bytes_to_unicode())}
    for a, b in MERGES:
        vocab[a + b] = len(vocab)
    base = 256 + len(MERGES)
    specials = {tok: base + i for i, tok in enumerate(SPECIALS)}
    assert max(specials.values()) < VOCAB_SIZE
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for t, i in specials.items()]
    byte_level = {"add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN_SPLIT_REGEX}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", **byte_level}]},
        "post_processor": {"type": "ByteLevel", **byte_level},
        "decoder": {"type": "ByteLevel", **byte_level},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [[a, b] for a, b in MERGES]},
    }
    (out / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("#version: 0.2\n" + "".join(f"{a} {b}\n" for a, b in MERGES), encoding="utf-8")
    entry = {"lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
    (out / "tokenizer_config.json").write_text(json.dumps({
        "add_bos_token": False, "add_prefix_space": False,
        "added_tokens_decoder": {str(i): {"content": t, **entry, "special": True} for t, i in specials.items()},
        "additional_special_tokens": SPECIALS[1:], "bos_token": None, "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>", "errors": "replace", "extra_special_tokens": {}, "model_max_length": 131072,
        "pad_token": "<|endoftext|>", "split_special_tokens": False, "tokenizer_class": "Qwen2Tokenizer",
        "unk_token": None, "chat_template": CHAT_TEMPLATE}, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- random weights
def randomize(module, seed):
    """Seeded weights that keep activations around one: matrices 1/sqrt(fan_in), norm scales 1 + noise, biases small."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "embed_tokens" in name:
                v = torch.randn(p.shape, generator=g)
            elif name.endswith("_pad_token"):
                v = 0.5 * torch.randn(p.shape, generator=g)
            elif p.ndim >= 2:
                fan_in = p[0].numel()
                v = torch.randn(p.shape, generator=g) / math.sqrt(fan_in)
            elif "norm" in name and name.endswith("weight"):
                v = 1.0 + 0.2 * torch.randn(p.shape, generator=g)
            else:
                v = 0.1 * torch.randn(p.shape, generator=g)
            p.copy_(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--size", default="64x96", help="WxH of the reference picture (multiples of 16)")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    from diffusers import AutoencoderKL, ZImageTransformer2DModel
    from transformers import Qwen3Config, Qwen3ForCausalLM

    write_tokenizer(out / "tokenizer")

    te_cfg = Qwen3Config(vocab_size=VOCAB_SIZE, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
                         rope_theta=1000000.0, max_position_embeddings=512, tie_word_embeddings=True,
                         attention_bias=False, sliding_window=None, use_sliding_window=False)
    te = Qwen3ForCausalLM(te_cfg)
    randomize(te, SEED)
    te.to(torch.bfloat16).save_pretrained(out / "text_encoder", safe_serialization=True, max_shard_size="150KB")

    tr = ZImageTransformer2DModel(dim=64, n_layers=3, n_refiner_layers=1, n_heads=2, n_kv_heads=2, norm_eps=1e-5,
                                  qk_norm=True, cap_feat_dim=64, rope_theta=256.0, t_scale=1000.0,
                                  axes_dims=[8, 12, 12], axes_lens=[256, 64, 64], in_channels=16)
    randomize(tr, SEED + 1)
    # stored as float32 like the real checkpoint, whose values are bf16 ones (it was trained in bf16): rounding through
    # bf16 first keeps a bf16-weight run of the engine exact
    tr.to(torch.bfloat16).to(torch.float32)
    tr.save_pretrained(out / "transformer", safe_serialization=True, max_shard_size="200KB")

    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=["DownEncoderBlock2D"] * 4,
                        up_block_types=["UpDecoderBlock2D"] * 4, block_out_channels=[8, 16, 32, 32],
                        layers_per_block=2, latent_channels=16, norm_num_groups=4, act_fn="silu",
                        sample_size=64, scaling_factor=0.3611, shift_factor=0.1159,
                        use_quant_conv=False, use_post_quant_conv=False, mid_block_add_attention=True,
                        force_upcast=True)
    randomize(vae, SEED + 2)
    vae.to(torch.bfloat16).save_pretrained(out / "vae", safe_serialization=True)

    (out / "scheduler").mkdir(exist_ok=True)
    (out / "scheduler" / "scheduler_config.json").write_text(json.dumps({
        "_class_name": "FlowMatchEulerDiscreteScheduler", "_diffusers_version": "0.36.0.dev0",
        "num_train_timesteps": 1000, "use_dynamic_shifting": False, "shift": 3.0}, indent=2), encoding="utf-8")
    (out / "model_index.json").write_text(json.dumps({
        "_class_name": "ZImagePipeline", "_diffusers_version": "0.36.0.dev0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3Model"], "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "transformer": ["diffusers", "ZImageTransformer2DModel"], "vae": ["diffusers", "AutoencoderKL"]}, indent=4),
        encoding="utf-8")

    # the reference: the same staged code as for the real model, in float32 end to end, on the files just written
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "zimage_ref.py"), str(out), str(out / "ref"), args.size, str(args.steps),
                    PROMPT, "--float32-loop"], check=True)
    print(f"tiny Z-Image pipeline and reference in {out}")


if __name__ == "__main__":
    main()
