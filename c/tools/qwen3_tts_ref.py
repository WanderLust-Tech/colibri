#!/usr/bin/env python3
"""Reference tensors of Qwen3-TTS for the C engine to be held to (docs/qwen3-speech.md, stage T0).

    <venv with qwen-tts>\\python tools/qwen3_tts_ref.py MODEL_DIR OUT_DIR [--text "..."] [--speaker Ryan] [--language English]

Runs the official `qwen-tts` code in fp32 on the CPU, **greedily** (no sampling, so the codes are deterministic; the repetition penalty and the
two-step minimum length of the reference still apply), and writes, in OUT_DIR, raw little-endian files and a `meta.json` naming them:

    input_ids.i32      the text prompt's token ids (`<|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n`)
    prefill.f32        [L][hidden]  the talker's prefill input (what the prompt becomes)
    logits0.f32        [3072]       the talker's logits at the last prefill position, before any processor
    codes.i32          [frames][16] the generated codes
    wav.f32            [samples]    the decoded waveform, 24 kHz

The C side loads the same text and compares each stage. Needs torch, transformers and qwen-tts (the venv in docs/qwen3-speech.md).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--text", default="Hello, this is a test of the speech engine.")
    ap.add_argument("--speaker", default="Ryan")
    ap.add_argument("--language", default="English")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--instruct", default=None, help="a style (CustomVoice 1.7B) or the voice's description (VoiceDesign)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(a.model, dtype=torch.float32, device_map="cpu")
    inner = tts.model
    talker = inner.talker
    seen = {}

    original_generate = talker.generate

    def recording_generate(*args, **kwargs):
        seen["prefill"] = kwargs["inputs_embeds"].detach().clone()
        return original_generate(*args, **kwargs)

    talker.generate = recording_generate

    def head_hook(module, args, output):
        if "logits0" not in seen and output.shape[1] > 1:                    # the prefill: more than one position
            seen["logits0"] = output[:, -1, :].detach().clone()

    handle = talker.codec_head.register_forward_hook(head_hook)
    text = tts._build_assistant_text(a.text)
    ids = tts._tokenize_texts([text])[0]
    design = inner.tts_model_type == "voice_design"

    def synth():
        if design:
            return tts.generate_voice_design(text=a.text, instruct=a.instruct or "", language=a.language, do_sample=False, subtalker_dosample=False,
                                             max_new_tokens=a.max_new)
        return tts.generate_custom_voice(text=a.text, speaker=a.speaker, language=a.language, instruct=a.instruct, do_sample=False,
                                         subtalker_dosample=False, max_new_tokens=a.max_new)

    wavs, sr = synth()
    handle.remove()
    # the codes: run the talker part again through the same entry point, to get them (generate_custom_voice returns audio only)
    captured = {}
    original = inner.generate

    def keep(*args, **kwargs):
        codes, hidden = original(*args, **kwargs)
        captured["codes"] = codes[0].detach().clone()
        return codes, hidden

    inner.generate = keep
    wavs2, _ = synth()
    codes = captured["codes"].to(torch.int32)
    ids.reshape(-1).to(torch.int32).numpy().astype("<i4").tofile(os.path.join(a.out, "input_ids.i32"))
    seen["prefill"][0].numpy().astype("<f4").tofile(os.path.join(a.out, "prefill.f32"))
    seen["logits0"][0].numpy().astype("<f4").tofile(os.path.join(a.out, "logits0.f32"))
    codes.numpy().astype("<i4").tofile(os.path.join(a.out, "codes.i32"))
    wav = np.asarray(wavs[0], dtype=np.float32)
    wav.astype("<f4").tofile(os.path.join(a.out, "wav.f32"))
    meta = {"model": os.path.basename(os.path.normpath(a.model)), "text": a.text, "speaker": "" if design else a.speaker, "instruct": a.instruct, "language": a.language,
            "prompt_text": text, "prompt_ids": int(ids.numel()), "prefill_len": int(seen["prefill"].shape[1]), "frames": int(codes.shape[0]),
            "code_groups": int(codes.shape[1]), "sample_rate": int(sr), "samples": int(wav.shape[0]),
            "same_waveform_twice": bool(np.array_equal(wav, np.asarray(wavs2[0], dtype=np.float32))), "codes": codes.tolist()}
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as handle_out:
        json.dump(meta, handle_out, ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("prompt_ids", "prefill_len", "frames", "code_groups", "sample_rate", "samples", "same_waveform_twice")}))


if __name__ == "__main__":
    sys.exit(main())
