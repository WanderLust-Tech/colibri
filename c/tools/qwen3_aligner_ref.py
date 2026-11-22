#!/usr/bin/env python3
"""Reference tensors of Qwen3-ForcedAligner for the C engine to be held to (docs/qwen3-aligner.md).

    <venv with qwen-asr>\\python tools/qwen3_aligner_ref.py MODEL_DIR OUT_DIR CLIP.wav --text "..." [--language English]

Runs the official `qwen-asr` aligner in fp32 on the CPU and writes, in OUT_DIR, raw little-endian files and a `meta.json`:

    words.json         the words the text was cut into
    input_ids.i32      the prompt's token ids (the audio slots expanded)
    classes.i32        [2 * words] the argmax class at each <timestamp> position (times 80 ms)
    raw_ms.i32         the same times in milliseconds, before the fix
    fixed_ms.i32       the times after fix_timestamp (what start and end are made of)
    logits.f32         [2 * words][5000] the head's outputs at those positions

The C side loads the same clip and text and compares each stage. Needs torch, transformers, qwen-asr (and nagisa, which qwen-asr imports).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from qwen_asr import Qwen3ForcedAligner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("clip")
    ap.add_argument("--text", required=True)
    ap.add_argument("--language", default="English")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    aligner = Qwen3ForcedAligner.from_pretrained(a.model, dtype=torch.float32, device_map="cpu")
    words, text = aligner.aligner_processor.encode_timestamp(a.text, a.language)
    seen = {}
    thinker = aligner.model.thinker

    def keep(module, args, kwargs, output):
        seen["input_ids"] = kwargs["input_ids"].detach().clone()
        seen["logits"] = output.logits.detach().clone()

    handle = thinker.register_forward_hook(keep, with_kwargs=True)
    result = aligner.align(a.clip, a.text, a.language)[0]
    handle.remove()
    ids = seen["input_ids"][0]
    positions = (ids == aligner.timestamp_token_id).nonzero().flatten()
    logits = seen["logits"][0][positions]
    classes = logits.argmax(dim=-1)
    raw_ms = (classes * aligner.timestamp_segment_time).numpy()
    fixed = aligner.aligner_processor.fix_timestamp(raw_ms)
    ids.numpy().astype("<i4").tofile(os.path.join(a.out, "input_ids.i32"))
    classes.numpy().astype("<i4").tofile(os.path.join(a.out, "classes.i32"))
    np.asarray(raw_ms, dtype="<i4").tofile(os.path.join(a.out, "raw_ms.i32"))
    np.asarray(fixed, dtype="<i4").tofile(os.path.join(a.out, "fixed_ms.i32"))
    logits.numpy().astype("<f4").tofile(os.path.join(a.out, "logits.f32"))
    items = [{"text": it.text, "start": it.start_time, "end": it.end_time} for it in result]
    meta = {"model": os.path.basename(os.path.normpath(a.model)), "text": a.text, "language": a.language, "words": words, "prompt_ids": int(ids.numel()),
            "timestamps": int(positions.numel()), "items": items, "raw_ms": raw_ms.astype(int).tolist(), "fixed_ms": fixed}
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as handle_out:
        json.dump(meta, handle_out, ensure_ascii=False)
    print(json.dumps({"words": len(words), "timestamps": meta["timestamps"], "prompt_ids": meta["prompt_ids"]}))
    for it in items:
        print(it)


if __name__ == "__main__":
    sys.exit(main())
