#!/usr/bin/env python3
"""A tiny random ACE-Step 1.5 turbo checkpoint with the real structure, for the oracles and CI.

    make_acestep_tiny.py OUTDIR [--seed N]

OUTDIR gets the layout of the Hugging Face repository ACE-Step/Ace-Step1.5, so upstream's own handler
(and the C engine) read it the way they read the real one:

    acestep-v15-turbo/     config.json, model.safetensors, silence_latent.pt, and the two modeling files
    vae/                   config.json, diffusion_pytorch_model.safetensors
    Qwen3-Embedding-0.6B/  config.json, model.safetensors, tokenizer files

What is the real checkpoint's and what is not:
  * the tensor names, dtypes (bf16) and ranks are the real ones, in the same stacks: tests/test_acestep_fixture.py
    holds the fixture to tools/acestep_manifest.json;
  * the widths are tiny (DiT 64 wide with 4 layers, VAE 8 channels, text encoder 32 wide with 2 layers) so a
    whole generation takes seconds, but every one is read from config.json by the engine, none is assumed;
  * what the real checkpoint fixes stays fixed: 64 latent channels at 25 Hz, the VAE's 1920 samples per frame
    (ratios 2, 4, 4, 6, 10), 48 kHz stereo, patch size 2, the FSQ levels [8, 8, 8, 5, 5, 5], pool window 5;
  * the sliding window is 8 frames (the real one is 128), so a 10 second clip (125 DiT tokens) is long enough
    for the window to cut attention off, which a window wider than the clip would never show;
  * the weights are random but not small: every projection is drawn at 1/sqrt(fan_in), norms are 1 + noise,
    biases and modulation tables are not zero, so no stage is an identity that a wrong index could hide behind;
  * the tokenizer is a 600-entry byte-level BPE trained on the prompt template, not Qwen's 151k: the real
    tokenizer is exercised against the real tokenizer.json in its own test.

Needs torch, transformers 4.57, diffusers and tokenizers (the reference venv, tools/requirements-acestep-ref.txt)
and a clone of https://github.com/ace-step/ACE-Step-1.5, named by ACESTEP_UPSTREAM (the modeling code is upstream's:
it is imported here and copied next to the weights, exactly as upstream's own loader syncs it).
"""
import argparse
import json
import os
import shutil
import sys

import torch


def upstream_root():
    root = os.environ.get("ACESTEP_UPSTREAM")
    if not root or not os.path.isfile(os.path.join(root, "acestep", "models", "turbo", "modeling_acestep_v15_turbo.py")):
        sys.exit("set ACESTEP_UPSTREAM to a clone of https://github.com/ace-step/ACE-Step-1.5")
    return root


# The real turbo config.json, for every key the tiny one does not override.
REAL_DIT_CONFIG = {
    "architectures": ["AceStepConditionGenerationModel"],
    "attention_bias": False, "attention_dropout": 0.0, "audio_acoustic_hidden_dim": 64,
    "auto_map": {"AutoConfig": "configuration_acestep_v15.AceStepConfig",
                 "AutoModel": "modeling_acestep_v15_turbo.AceStepConditionGenerationModel"},
    "data_proportion": 0.5, "dtype": "bfloat16", "fsq_dim": 2048, "fsq_input_levels": [8, 8, 8, 5, 5, 5],
    "fsq_input_num_quantizers": 1, "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048, "in_channels": 192,
    "initializer_range": 0.02, "intermediate_size": 6144, "is_turbo": True,
    "max_position_embeddings": 32768, "model_type": "acestep", "model_version": "turbo", "num_attention_heads": 16,
    "num_attention_pooler_hidden_layers": 2, "num_audio_decoder_hidden_layers": 24, "num_hidden_layers": 24,
    "num_key_value_heads": 8, "num_lyric_encoder_hidden_layers": 8, "num_timbre_encoder_hidden_layers": 4,
    "patch_size": 2, "pool_window_size": 5, "rms_norm_eps": 1e-06, "rope_scaling": None, "rope_theta": 1000000,
    "sliding_window": 128, "text_hidden_dim": 1024, "timbre_fix_frame": 750, "timbre_hidden_dim": 64,
    "timestep_mu": -0.4, "timestep_sigma": 1.0, "use_cache": True, "use_sliding_window": True, "vocab_size": 64003,
}

TINY_DIT = {
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4, "num_audio_decoder_hidden_layers": 4,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "fsq_dim": 64, "text_hidden_dim": 32,
    "num_lyric_encoder_hidden_layers": 2, "num_timbre_encoder_hidden_layers": 2,
    "num_attention_pooler_hidden_layers": 2, "sliding_window": 8,
}

VAE_CONFIG = {
    "_class_name": "AutoencoderOobleck", "audio_channels": 2, "channel_multiples": [1, 2, 4, 8, 16],
    "decoder_channels": 8, "decoder_input_channels": 64, "downsampling_ratios": [2, 4, 4, 6, 10],
    "encoder_hidden_size": 8, "sampling_rate": 48000,
}

TINY_TEXT = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 8}

SILENCE_FRAMES = 1500   # 60 seconds at 25 Hz; the real file holds 15000 (10 minutes)

# Strength of each VAE residual unit's output convolution. A random stack of 15 residual units with Snake
# activations at full strength is chaotic: activations reach 200 and float32 itself is 2e-4 (relative) from the
# float64 answer, so an oracle could only be as sharp as that and two correct decoders disagree by about as much.
# At 0.25 the activations stay near 4 and float32 is within 1e-6 of float64 (measured, tools/../tests), which lets
# the oracle's tolerance be 20 times that and still catch a wrong tap, dilation or padding.
RES_BRANCH = 0.25

# Text the tokenizer is trained on: the prompt template, the lyric template, and some words.
CORPUS = [
    "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n# Caption\n\n# Metas\n- bpm: 120\n"
    "- timesignature: 4\n- keyscale: C major\n- duration: 10 seconds\n<|endoftext|>",
    "# Languages\nen\n\n# Lyric\n[Verse 1]\nwalking down the empty street\nthe city sleeps beneath my feet\n"
    "[Chorus]\nsing it out loud tonight\n<|endoftext|>",
    "calm piano with soft strings, upbeat electronic dance music, a slow acoustic guitar ballad, heavy rock drums",
    "unknown zh ja ko fr de es it pt ru instrumental NA bpm timesignature keyscale duration minor major",
] * 4


# Qwen2/Qwen3's pre-tokenizer regex, copied from the real Qwen3-Embedding-0.6B tokenizer.json: contractions in either case,
# letters with at most one leading non-letter, ONE digit at a time (cl100k takes up to three), punctuation runs, newline and
# whitespace runs. The tiny tokenizer has to split text the way the real one does, or a port tested on it proves nothing.
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def make_tokenizer(path):
    """A 600-entry byte-level BPE with the real tokenizer's pipeline: an NFC normalizer, the Qwen split regex, and a
    post-processor that appends <|endoftext|> to every sequence (so a prompt ending in the literal token ends in two)."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok.train_from_iterator(CORPUS, trainers.BpeTrainer(
        vocab_size=600, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.post_processor = processors.Sequence([
        processors.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
        processors.TemplateProcessing(single="$A <|endoftext|>", pair="$A <|endoftext|> $B:1 <|endoftext|>:1",
                                      special_tokens=[("<|endoftext|>", tok.token_to_id("<|endoftext|>"))])])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<|endoftext|>", eos_token="<|im_end|>",
                                   clean_up_tokenization_spaces=False)
    fast.save_pretrained(path)
    return len(fast)


def spread(generator, shape, scale, fp32=True):
    return torch.randn(shape, generator=generator) * scale


def randomize(module, seed, snake_scale=0.3):
    """Draw every parameter so that no stage is an identity (see the module docstring)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("weight_g",):
                continue   # set from weight_v below
            if leaf == "weight_v":
                p.copy_(spread(g, p.shape, 1.0 / (p[0].numel() ** 0.5)))
                gname = name[:-1] + "g"
                dict(module.named_parameters())[gname].copy_(
                    p.flatten(1).norm(dim=1).reshape(-1, 1, 1) * (0.8 + 0.4 * torch.rand(p.shape[0], 1, 1, generator=g)))
            elif leaf in ("alpha", "beta"):
                p.copy_(spread(g, p.shape, snake_scale))
            elif leaf == "bias":
                p.copy_(spread(g, p.shape, 0.1))
            elif leaf == "weight" and p.dim() == 1:                       # RMSNorm and q/k norms
                p.copy_(1.0 + spread(g, p.shape, 0.2))
            elif leaf == "weight":                                        # Linear [out, in], Conv1d [out, in, k],
                fan_in = p[0].numel() if p.dim() != 3 or "proj_out" not in name else p.shape[0] * p.shape[2]
                p.copy_(spread(g, p.shape, 1.0 / (fan_in ** 0.5)))        # ConvTranspose1d [in, out, k]: fan is in*k
            elif leaf in ("scale_shift_table", "special_tokens"):
                p.copy_(spread(g, p.shape, 0.3))
            else:                                                         # null_condition_emb and the like
                p.copy_(spread(g, p.shape, 1.0))


def calibrate_decoder(vae, seed, target_rms=0.3):
    """Scale the decoder's last convolution so a typical latent decodes to a waveform of rms ~0.3.

    Random residual stacks with Snake activations multiply the signal at every block: left alone the tiny decoder
    answers with peaks near 170 where the real one stays near 1, and its oracle would be a test of the gain.
    The scale is one number per seed (the output convolution is linear), so the structure is untouched.
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(1, 64, 40, generator=g) * 1.2          # the DiT's latents have a standard deviation near 1
    with torch.no_grad():
        rms = vae.decode(z).sample.pow(2).mean().sqrt().item()
        vae.decoder.conv2.weight_g.mul_(target_rms / rms)


def bf16_state(module):
    """The module's tensors as the checkpoint stores them: bf16, contiguous, parametrization names undone."""
    out = {}
    for name, t in module.state_dict().items():
        name = name.replace(".parametrizations.weight.original0", ".weight_g")
        name = name.replace(".parametrizations.weight.original1", ".weight_v")
        out[name] = t.detach().to(torch.bfloat16).contiguous().clone()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    root = upstream_root()
    sys.path.insert(0, root)
    from safetensors.torch import save_file

    out = os.path.abspath(a.outdir)
    dit_dir, vae_dir, text_dir = (os.path.join(out, d) for d in ("acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B"))
    for d in (dit_dir, vae_dir, text_dir):
        os.makedirs(d, exist_ok=True)

    # --- text encoder and tokenizer -------------------------------------------------------------------
    from transformers import Qwen3Config, Qwen3Model
    vocab = make_tokenizer(text_dir)
    text_cfg = Qwen3Config(vocab_size=vocab, max_position_embeddings=4096, rope_theta=1000000.0, rms_norm_eps=1e-6,
                           tie_word_embeddings=False, **TINY_TEXT)
    torch.manual_seed(a.seed)
    text = Qwen3Model(text_cfg)
    randomize(text, a.seed + 1)
    text_cfg.dtype = "bfloat16"
    text_cfg.save_pretrained(text_dir)
    save_file(bf16_state(text), os.path.join(text_dir, "model.safetensors"), metadata={"format": "pt"})

    # --- VAE ---------------------------------------------------------------------------------------------
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck(**{k: v for k, v in VAE_CONFIG.items() if not k.startswith("_")})
    randomize(vae, a.seed + 2)
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if "res_unit" in name and ".conv2." in name and name.endswith("weight_g"):
                p.mul_(RES_BRANCH)
    calibrate_decoder(vae, a.seed + 5)
    with open(os.path.join(vae_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**VAE_CONFIG, "_diffusers_version": "0.34.0"}, f, indent=2)
    save_file(bf16_state(vae), os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})

    # --- DiT, condition encoders, FSQ tokenizer ------------------------------------------------------------
    cfg = {**REAL_DIT_CONFIG, **TINY_DIT}
    cfg["layer_types"] = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(cfg["num_hidden_layers"])]
    from acestep.models.common.configuration_acestep_v15 import AceStepConfig
    from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepConditionGenerationModel
    dit = AceStepConditionGenerationModel(AceStepConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "auto_map")}))
    randomize(dit, a.seed + 3)
    with open(os.path.join(dit_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    save_file(bf16_state(dit), os.path.join(dit_dir, "model.safetensors"), metadata={"format": "pt"})
    for name in ("modeling_acestep_v15_turbo.py", "configuration_acestep_v15.py"):
        src = os.path.join(root, "acestep", "models", "turbo", name)
        shutil.copyfile(src, os.path.join(dit_dir, name))

    # A smooth, nonzero "silence": the engine slices and tiles this, so a constant would hide a wrong offset.
    g = torch.Generator().manual_seed(a.seed + 4)
    t = torch.arange(SILENCE_FRAMES, dtype=torch.float32)
    base = torch.randn(64, 1, generator=g) * 0.5 + 0.3 * torch.sin(t[None] / 50.0 + torch.randn(64, 1, generator=g))
    torch.save((base + 0.02 * torch.randn(64, SILENCE_FRAMES, generator=g))[None].contiguous(),
               os.path.join(dit_dir, "silence_latent.pt"))

    with open(os.path.join(out, "fixture.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": a.seed, "dit": TINY_DIT, "vae": VAE_CONFIG, "text": TINY_TEXT, "tokenizer_vocab": vocab,
                   "silence_frames": SILENCE_FRAMES, "torch": torch.__version__}, f, indent=1)
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    print(f"wrote {out}: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
