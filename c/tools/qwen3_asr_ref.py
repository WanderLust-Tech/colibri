#!/usr/bin/env python3
"""Reference tensors of Qwen3-ASR for the C engine to be held to (docs/qwen3-speech.md, stage A0).

    <venv with qwen-asr>\\python tools/qwen3_asr_ref.py MODEL_DIR CLIP.wav OUT_DIR [--language English] [--max-new 256]

Runs the official `qwen-asr` code in fp32 on the CPU and writes, in OUT_DIR, raw little-endian float32 / int32 files and a
`meta.json` naming them:

    features.f32    [128][T]   the log-mel features the processor makes (WhisperFeatureExtractor), T = valid frames
    conv.f32        [frames][d_model]   after the three stride-2 convolutions, conv_out and the chunk positions (no layers yet)
    encoder.f32     [frames][output_dim] the audio tower's output (what replaces the audio-pad embeddings)
    prompt_ids.i32  the prompt's token ids with the audio pads expanded
    generated.i32   the greedy continuation (token ids, up to the end token)
    text            the decoded answer, raw and parsed (in meta.json)

The C side loads the same clip and compares each stage. Needs torch, transformers and qwen-asr (the venv in docs/qwen3-speech.md).
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import parse_asr_output


def windowed_attention_forward(self, hidden_states, cu_seqlens=None, attention_mask=None, **kwargs):
    """Qwen3ASRAudioAttention.forward with the windows honoured: each [cu_seqlens[i], cu_seqlens[i+1]) block attends only to itself."""
    seq = hidden_states.shape[0]
    q = self.q_proj(hidden_states).reshape(seq, self.num_heads, -1)
    k = self.k_proj(hidden_states).reshape(seq, self.num_heads, -1)
    v = self.v_proj(hidden_states).reshape(seq, self.num_heads, -1)
    out = torch.empty_like(q)
    bounds = cu_seqlens.tolist()
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        o = torch.nn.functional.scaled_dot_product_attention(
            q[lo:hi].transpose(0, 1).unsqueeze(0), k[lo:hi].transpose(0, 1).unsqueeze(0), v[lo:hi].transpose(0, 1).unsqueeze(0))
        out[lo:hi] = o.squeeze(0).transpose(0, 1)
    return self.out_proj(out.reshape(seq, -1).contiguous())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--language", default=None)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--context", default="")
    ap.add_argument("--windowed", action="store_true",
                    help="attend inside the 800-frame windows (cu_seqlens), as the vLLM backend does; the stock transformers path (sdpa) ignores "
                         "them and attends over the whole audio, which is the same up to 8 s of audio")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    wav, sr = sf.read(a.clip, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    wav = wav.astype(np.float32)

    if a.windowed:
        from qwen_asr.core.transformers_backend import modeling_qwen3_asr as modeling
        modeling.Qwen3ASRAudioAttention.forward = windowed_attention_forward
    asr = Qwen3ASRModel.from_pretrained(a.model, dtype=torch.float32, device_map="cpu", max_inference_batch_size=1, max_new_tokens=a.max_new)
    model, processor = asr.model, asr.processor
    prompt = asr._build_text_prompt(context=a.context, force_language=a.language)
    inputs = processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)
    feats = inputs["input_features"][0]                       # [128][T padded]
    n = int(inputs["feature_attention_mask"][0].sum())
    feats = feats[:, :n].contiguous()
    feats.numpy().astype("<f4").tofile(os.path.join(a.out, "features.f32"))

    tower = model.thinker.audio_tower
    captured = {}

    def hook_layers(module, args, kwargs):
        captured["pre_layers"] = args[0].detach().clone() if args else kwargs["hidden_states"].detach().clone()

    handle = tower.layers[0].register_forward_pre_hook(hook_layers, with_kwargs=True)
    with torch.no_grad():
        enc = tower(feats, feature_lens=torch.tensor([n])).last_hidden_state
    handle.remove()
    captured["pre_layers"].numpy().astype("<f4").tofile(os.path.join(a.out, "conv.f32"))
    enc.numpy().astype("<f4").tofile(os.path.join(a.out, "encoder.f32"))

    ids = inputs["input_ids"][0].to(torch.int32)
    ids.numpy().astype("<i4").tofile(os.path.join(a.out, "prompt_ids.i32"))
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=a.max_new)
    gen = out.sequences[0, inputs["input_ids"].shape[1]:].to(torch.int32)
    gen.numpy().astype("<i4").tofile(os.path.join(a.out, "generated.i32"))
    raw = processor.batch_decode(gen.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    lang, text = parse_asr_output(raw, user_language=a.language)
    meta = {
        "model": os.path.basename(os.path.normpath(a.model)), "clip": os.path.basename(a.clip), "samples": int(wav.shape[0]),
        "mel_frames": n, "mel_bins": int(feats.shape[0]), "conv_shape": list(captured["pre_layers"].shape), "encoder_shape": list(enc.shape),
        "prompt_tokens": int(ids.shape[0]), "generated_tokens": int(gen.shape[0]), "generated": gen.tolist(),
        "raw": raw, "language": lang, "text": text, "forced_language": a.language, "prompt": prompt, "windowed": bool(a.windowed),
    }
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=1)
    print(json.dumps({k: meta[k] for k in ("mel_frames", "conv_shape", "encoder_shape", "prompt_tokens", "generated_tokens", "language", "text")}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
