#!/usr/bin/env python3
"""Reference outputs of ACE-Step 1.5's 5 Hz language model (the planner) captured from upstream's own `LLMHandler`, for the C engine's oracle.

    acestep_ref_lm.py capture MODEL_DIR OUTDIR [--lm acestep-5Hz-lm-1.7B] [--cases a,b]

MODEL_DIR is the checkpoint (it holds the LM's directory), OUTDIR gets one directory a case. The handler is upstream's `LLMHandler`
(`generate_with_stop_condition`, the codes phase of the two-phase planner with the metadata given, so the first phase is skipped) on the CPU in
float32, with the temperature at 0 (an argmax: torch's multinomial cannot be reproduced) and the guidance scale of the real flow.

    OUTDIR/<case>/ref.safetensors
        prompt.cond, prompt.uncond   int64 ids of the two prompts, as the handler tokenizes them (before the batch pads them)
        step{i}.cond, step{i}.uncond f32 [1, N] raw logits of the model at step i, at the audio-code ids and the end token (see codes.ids)
        codes.ids                    int64 [N] the token ids those columns are: the audio codes 0..63999 then <|im_end|>
        codes                        int64 the audio codes the handler returned (0..63999)
    OUTDIR/<case>/ref.json           the case's parameters, the two prompts as text, the handler's text output

The model's own batch has the shorter prompt left-padded; RoPE only sees relative positions and the pad is masked, so the engine runs each prompt from
position 0 and the logits are the same (the oracle checks that they are).
"""
import argparse
import json
import os
import sys
import tempfile
import shutil
import atexit

import torch

AUDIO_CODES = 64000

# name: (caption, lyrics, the metadata the user gives, seconds of music asked for)
CASES = {
    "lm10": ("bright ukulele and light percussion, summery", "[Instrumental]",
             {"bpm": 120, "keyscale": "C Major", "timesignature": "4", "duration": 10, "language": "unknown"}, 10),
    "lm_vocal10": ("warm indie folk with soft vocals", "[Verse 1]\nmorning light on the kitchen floor\n[Chorus]\nsing it back to me once more",
                   {"bpm": 96, "keyscale": "G Major", "timesignature": "3", "duration": 10, "language": "en"}, 10),
    "lm_cjk6": ("ミニマルなテクノ、深いベース", "[Instrumental]",
                {"bpm": 128, "keyscale": "A minor", "timesignature": "4", "duration": 6, "language": "ja"}, 6),
    "lm_nometa8": ("slow cinematic strings", "[Instrumental]",
                   {"bpm": 70, "keyscale": "D minor", "timesignature": "4", "duration": 8}, 8),
}


def open_handler(model_dir, lm):
    root = os.environ.get("ACESTEP_UPSTREAM")
    if not root or not os.path.isdir(os.path.join(root, "acestep")):
        sys.exit("set ACESTEP_UPSTREAM to a clone of https://github.com/ace-step/ACE-Step-1.5")
    root, model_dir = os.path.abspath(root), os.path.abspath(model_dir)
    caller_cwd, scratch = os.getcwd(), tempfile.mkdtemp(prefix="acestep-ref-lm-")
    os.chdir(scratch)

    def leave_scratch():
        os.chdir(caller_cwd)
        shutil.rmtree(scratch, ignore_errors=True)
    atexit.register(leave_scratch)
    sys.path.insert(0, root)
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    from acestep.llm_inference import LLMHandler
    torch.set_num_threads(os.cpu_count() or 8)
    h = LLMHandler()
    h.disable_tqdm = True
    msg, ok = h.initialize(checkpoint_dir=model_dir, lm_model_path=lm, backend="pt", device="cpu", dtype=torch.float32)
    if not ok:
        sys.exit(str(msg))
    return h


def capture(a):
    from safetensors.torch import save_file
    h = open_handler(a.model_dir, a.lm)
    tok = h.llm_tokenizer
    valid = [tok.convert_tokens_to_ids(f"<|audio_code_{i}|>") for i in range(AUDIO_CODES)] + [tok.eos_token_id]
    valid_t = torch.tensor(valid, dtype=torch.long)
    names = list(CASES) if not a.cases else a.cases.split(",")
    unknown = [n for n in names if n not in CASES]
    if unknown:
        sys.exit(f"unknown case(s) {unknown}; known: {list(CASES)}")
    os.makedirs(a.outdir, exist_ok=True)
    state = {}

    def forward_hook(module, args, kwargs, output):
        i = state["step"]
        state["step"] += 1
        last = output.logits[:, -1, :].detach()
        if i == 0:
            state["input_ids"] = kwargs["input_ids"].detach().clone()
            state["mask"] = kwargs.get("attention_mask")
        n = last.shape[0] // 2
        state["t"][f"step{i}.cond"] = last[:n][:, valid_t].float().clone().contiguous()
        state["t"][f"step{i}.uncond"] = last[n:][:, valid_t].float().clone().contiguous()
    h.llm.register_forward_hook(forward_hook, with_kwargs=True)
    original = h.build_formatted_prompt_with_cot

    def build(caption, lyrics, cot_text, is_negative_prompt=False, **kw):
        text = original(caption, lyrics, cot_text, is_negative_prompt=is_negative_prompt, **kw)
        state["prompts"]["uncond" if is_negative_prompt else "cond"] = text
        return text
    h.build_formatted_prompt_with_cot = build

    for name in names:
        caption, lyrics, meta, seconds = CASES[name]
        state.update(step=0, t={}, prompts={})
        res = h.generate_with_stop_condition(
            caption=caption, lyrics=lyrics, infer_type="llm_dit", temperature=0.0, cfg_scale=2.0, top_k=None, top_p=None,
            repetition_penalty=1.0, use_constrained_decoding=True, target_duration=float(seconds), user_metadata=dict(meta),
            use_cot_metas=False, use_cot_caption=False, use_cot_language=False)
        if not res.get("success"):
            sys.exit(f"{name}: {res.get('error')}")
        codes_text = res["audio_codes"]
        import re
        codes = [int(c) for c in re.findall(r"<\|audio_code_(\d+)\|>", codes_text)]
        tensors = dict(state["t"])
        for k in ("cond", "uncond"):
            prompt = state["prompts"].get(k)
            if prompt is None:
                sys.exit(f"{name}: the handler built no {k} prompt")
            tensors[f"prompt.{k}"] = torch.tensor(tok(prompt, add_special_tokens=False)["input_ids"], dtype=torch.long)
        tensors["codes.ids"] = valid_t.clone()
        tensors["codes"] = torch.tensor(codes, dtype=torch.long)
        d = os.path.join(a.outdir, name)
        os.makedirs(d, exist_ok=True)
        save_file(tensors, os.path.join(d, "ref.safetensors"), metadata={"case": name})
        with open(os.path.join(d, "ref.json"), "w", encoding="utf-8") as f:
            json.dump({"case": name, "caption": caption, "lyrics": lyrics, "metadata": meta, "seconds": seconds, "temperature": 0.0, "cfg_scale": 2.0,
                       "prompts": state["prompts"], "steps": state["step"], "codes": len(codes), "cot": res.get("metadata")}, f, indent=1, default=str, ensure_ascii=False)
        print(f"{name}: {len(codes)} codes in {state['step']} steps, prompts {tuple(tensors['prompt.cond'].shape)} and {tuple(tensors['prompt.uncond'].shape)} tokens")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("model_dir")
    c.add_argument("outdir")
    c.add_argument("--lm", default="acestep-5Hz-lm-1.7B")
    c.add_argument("--cases", help="comma-separated; default all")
    a = ap.parse_args()
    capture(a)


if __name__ == "__main__":
    main()
