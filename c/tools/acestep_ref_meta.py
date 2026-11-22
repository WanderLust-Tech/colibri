#!/usr/bin/env python3
"""Reference outputs of the planner's FIRST phase (the language model writing the song's metadata under upstream's constrained decoding), captured
from upstream's own `LLMHandler`, for the C engine's oracle.

    acestep_ref_meta.py capture MODEL_DIR OUTDIR [--lm acestep-5Hz-lm-1.7B] [--cases a,b]
    acestep_ref_meta.py yaml OUTFILE [--count N]
    acestep_ref_meta.py parse OUTFILE [--count N]

`capture`: MODEL_DIR is the checkpoint (it holds the LM's directory), OUTDIR gets one directory a case. The handler is upstream's `LLMHandler`
(`generate_with_stop_condition`: phase 1, the metadata under the constrained decoder, then, in the cases that ask for it, phase 2, the codes) on the CPU
in float32 with the temperature at 0 (an argmax: torch's multinomial cannot be reproduced). Phase 1 runs with no guidance (upstream turns it off there).

    OUTDIR/<case>/ref.json       the case's parameters, the prompt, one record a step of phase 1 (the decoder's state, how many tokens it let through,
                                 the token chosen), the text the model wrote, the metadata parsed from it, the YAML of phase 2's prompt
    OUTDIR/<case>/ref.safetensors
        prompt1                  int64 ids of phase 1's prompt
        gen1                     int64 ids the model wrote in phase 1
        s{i}.idx, s{i}.val       the tokens the decoder let through at step i and the model's raw logit at each; when it lets through nearly the whole
                                 vocabulary (the free caption), the 64 highest instead (idx by descending value)
        prompt.cond, prompt.uncond, step{i}.cond, step{i}.uncond, codes.ids, codes     as acestep_ref_lm.py writes them, for the cases with phase 2
    OUTDIR/trees.json            (--cases none writes only this) the decoder's token tables as upstream builds them from this tokenizer: for each field, every prefix of the value's
                                 token ids and the tokens that may follow it, and the token ids of every fixed string

`parse`: a corpus of texts a model might write, with what upstream's `parse_lm_output` makes of each (the metadata dict) and what its
`_format_metadata_as_cot` writes for that dict (the YAML of phase 2's prompt), for the engine's parser and YAML writer.

`yaml`: a corpus of strings with what PyYAML's `yaml.dump({"caption": s}, allow_unicode=True, sort_keys=True)` writes for each, for the engine's emitter.
"""
import argparse
import json
import os
import random
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acestep_ref_lm as lmref  # noqa: E402

AUDIO_CODES = 64000

# name: (caption, lyrics, the metadata the user gives, flags, whether phase 2 runs too)
# Phase 2 is kept to a few seconds of music: the handler on the CPU in float32 is slow.
CASES = {
    "meta_bare": ("bright ukulele and light percussion, summery", "[Instrumental]", {}, {}, False),
    "meta_vocal": ("warm indie folk with soft vocals", "[Verse 1]\nmorning light on the kitchen floor\n[Chorus]\nsing it back to me once more", {}, {}, False),
    "meta_ja": ("ミニマルなテクノ、深いベース", "[Instrumental]", {}, {}, False),
    "meta_dur8": ("slow cinematic strings", "[Instrumental]", {"duration": 8}, {}, True),
    "meta_bpm_key6": ("upbeat electronic dance music", "[Instrumental]", {"bpm": 128, "keyscale": "A minor", "duration": 6}, {}, True),
    "meta_nocap8": ("calm piano", "[Instrumental]", {"duration": 8}, {"use_cot_caption": False, "use_cot_language": False}, True),
    "meta_nolang10": ("heavy rock drums and a distorted guitar", "[Instrumental]", {"timesignature": "4", "duration": 10}, {"use_cot_language": False}, True),
}


def tree_dump(tree):
    return [[list(k), sorted(v)] for k, v in sorted(tree.items(), key=lambda kv: kv[0])]


def dump_trees(proc, path):
    from acestep.constrained_logits_processor import FSMState
    fixed = {}
    for st, text in proc.fixed_strings.items():
        proc.position_in_state = 0
        seq, pos, guard = [], 0, 0
        while guard < 40:
            guard += 1
            proc.position_in_state = pos
            allowed = proc._get_allowed_tokens_for_fixed_string(text)
            if not allowed:
                break
            seq.append(list(allowed))
            pos += len(proc.tokenizer.decode([allowed[0]]))
        fixed[st.name] = {"text": text, "allowed_by_step": seq}
    proc.position_in_state = 0
    out = {
        "bpm": tree_dump(proc.bpm_prefix_tree), "duration": tree_dump(proc.duration_prefix_tree), "keyscale": tree_dump(proc.keyscale_prefix_tree),
        "language": tree_dump(proc.language_prefix_tree), "timesignature": tree_dump(proc.timesig_prefix_tree), "fixed": fixed,
        "max_duration": proc.max_duration, "newline": proc.newline_token, "eos": proc.eos_token_id, "backtick": proc.backtick_token,
        "audio_codes": [min(proc.audio_code_token_ids), max(proc.audio_code_token_ids), len(proc.audio_code_token_ids)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)


def capture(a):
    from safetensors.torch import save_file
    h = lmref.open_handler(a.model_dir, a.lm)
    from acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor as P
    tok = h.llm_tokenizer
    valid = [tok.convert_tokens_to_ids(f"<|audio_code_{i}|>") for i in range(AUDIO_CODES)] + [tok.eos_token_id]
    valid_t = torch.tensor(valid, dtype=torch.long)
    names = list(CASES) if not a.cases else ([] if a.cases == "none" else a.cases.split(","))
    unknown = [n for n in names if n not in CASES]
    if unknown:
        sys.exit(f"unknown case(s) {unknown}; known: {list(CASES)}")
    os.makedirs(a.outdir, exist_ok=True)
    state = {"phase": 0, "calls": 0}

    original_call = P.__call__
    original_update = P.update_state

    def call(self, input_ids, scores):
        if state["phase"] != 1:
            return original_call(self, input_ids, scores)
        before = scores.detach().clone()
        st = self.state.name
        flags = (bool(self.caption_after_newline), bool(self.caption_ending), len(self.user_field_token_queue))
        out = original_call(self, input_ids, scores)
        fin = torch.isfinite(out[0])
        idx = fin.nonzero()[:, 0]
        i = state["calls"]
        state["calls"] += 1
        n = int(idx.numel())
        if n <= 2048:
            sel = idx
        else:
            top = torch.topk(out[0].masked_fill(~fin, float("-inf")), 64)
            sel = top.indices
        state["t"][f"s{i}.idx"] = sel.to(torch.int64).contiguous()
        state["t"][f"s{i}.val"] = before[0][sel].float().contiguous()
        state["steps"].append({"state": st, "n": n, "flags": list(flags), "tok": None, "after": None})
        return out

    def update(self, tok_id):
        if state["phase"] == 1 and state["steps"] and state["steps"][-1]["tok"] is None:
            state["steps"][-1]["tok"] = int(tok_id)
        original_update(self, tok_id)
        if state["phase"] == 1 and state["steps"]:
            state["steps"][-1]["after"] = self.state.name

    P.__call__ = call
    P.update_state = update

    def forward_hook(module, args, kwargs, output):
        last = output.logits[:, -1, :].detach()
        if last.shape[0] != 2:
            return
        i = state["step2"]
        state["step2"] += 1
        state["t"][f"step{i}.cond"] = last[:1][:, valid_t].float().clone().contiguous()
        state["t"][f"step{i}.uncond"] = last[1:][:, valid_t].float().clone().contiguous()
    h.llm.register_forward_hook(forward_hook, with_kwargs=True)

    original_gen = h.generate_from_formatted_prompt

    def gen(formatted_prompt, cfg=None, **kw):
        phase = (cfg or {}).get("generation_phase", "cot")
        if phase == "cot":
            state["phase"] = 1
            state["prompt1"] = formatted_prompt
        else:
            state["phase"] = 2
        try:
            text, status = original_gen(formatted_prompt, cfg=cfg, **kw)
        finally:
            state["phase"] = 0
        if phase == "cot":
            state["text1"] = text
        return text, status
    h.generate_from_formatted_prompt = gen

    original_p2 = h.build_formatted_prompt_with_cot

    def build2(caption, lyrics, cot_text, is_negative_prompt=False, **kw):
        text = original_p2(caption, lyrics, cot_text, is_negative_prompt=is_negative_prompt, **kw)
        state["prompts"]["uncond" if is_negative_prompt else "cond"] = text
        state["cot"] = cot_text if not is_negative_prompt else state.get("cot")
        return text
    h.build_formatted_prompt_with_cot = build2

    dump_trees(h.constrained_processor, os.path.join(a.outdir, "trees.json"))
    if a.cases == "none":
        return

    for name in names:
        caption, lyrics, meta, flags, phase2 = CASES[name]
        state.update(calls=0, steps=[], t={}, prompts={}, step2=0, text1=None, prompt1=None, cot=None)
        dur = float(meta["duration"]) if "duration" in meta else None
        res = h.generate_with_stop_condition(
            caption=caption, lyrics=lyrics, infer_type="llm_dit" if phase2 else "dit", temperature=0.0, cfg_scale=2.0, top_k=None, top_p=None,
            repetition_penalty=1.0, use_constrained_decoding=True, target_duration=dur, user_metadata=dict(meta) if meta else None,
            use_cot_metas=True, use_cot_caption=flags.get("use_cot_caption", True), use_cot_language=flags.get("use_cot_language", True))
        if not res.get("success"):
            sys.exit(f"{name}: {res.get('error')}")
        tensors = dict(state["t"])
        tensors["prompt1"] = torch.tensor(tok(state["prompt1"], add_special_tokens=False)["input_ids"], dtype=torch.long)
        gen_ids = [s["tok"] for s in state["steps"] if s["tok"] is not None]
        tensors["gen1"] = torch.tensor(gen_ids, dtype=torch.long)
        codes = []
        if phase2:
            for k in ("cond", "uncond"):
                tensors[f"prompt.{k}"] = torch.tensor(tok(state["prompts"][k], add_special_tokens=False)["input_ids"], dtype=torch.long)
            tensors["codes.ids"] = valid_t.clone()
            codes = [int(c) for c in re.findall(r"<\|audio_code_(\d+)\|>", res["audio_codes"])]
            tensors["codes"] = torch.tensor(codes, dtype=torch.long)
        d = os.path.join(a.outdir, name)
        os.makedirs(d, exist_ok=True)
        save_file(tensors, os.path.join(d, "ref.safetensors"), metadata={"case": name})
        with open(os.path.join(d, "ref.json"), "w", encoding="utf-8") as f:
            json.dump({"case": name, "caption": caption, "lyrics": lyrics, "user_metadata": meta, "flags": flags, "phase2": phase2,
                       "prompt1": state["prompt1"], "text1": state["text1"], "steps": state["steps"], "metadata": res.get("metadata"),
                       "cot": state["cot"], "max_duration": h.constrained_processor.max_duration, "prompts": state["prompts"], "codes": len(codes), "steps2": state["step2"],
                       "temperature": 0.0, "cfg_scale": 2.0}, f, indent=1, default=str, ensure_ascii=False)
        print(f"{name}: phase 1 in {len(state['steps'])} steps -> {res.get('metadata')}; {len(codes)} codes", flush=True)


# Strings for the YAML emitter: what a rewritten caption can look like, and what a plain scalar must not be mistaken for.
YAML_WORDS = ["a", "calm", "piano", "with", "soft", "strings", "and", "a", "warm", "pad", "-", "#", "no", "yes", "on", "off", "true", "null", "~", "1", "12", "0x1F",
              "1.5", "1e3", ".inf", "2001-01-01", "10:30", "'", '"', ":", "a:b", "a: b", "a #b", "!x", "&y", "*z", "%p", "@q", "`r", "[s]", "{t}", ",", "?", "|",
              ">", "<<", "=", "é", "日本語", "🎵", "\t", "  ", "\u00a0", "\u2028", "\x7f", "\x01", "\\", "a\\b", "end.", "(x)", "50%", "'quoted'", '"dq"',
              # what the resolvers and the analysis must tell from a plain word: floats with signed exponents, infinities, timestamps with zones, document markers,
              # characters a plain scalar cannot hold (a byte order mark, non-characters, the last code point)
              "1.5e+3", "1.5e-3", "-1.5e+10", ".5e+3", "1.e+3", "1.5E+3", ".Inf", ".INF", "-.inf", "+.Inf", ".nan", ".NaN", ".NAN", "190:20:30.15", "1:20", "-1:20.5",
              "2001-12-14t21:59:43.10-05:00", "2001-12-14 21:59:43.10 -5", "2001-12-15 2:59:43.10 Z", "2001-12-14T21:59:43Z", "2001-12-14 21:59:43 +05:30",
              "2001-1-1 1:00:00", "2001-12-14", "2001-12-14x", "--- text", "... text", "---", "...", "--", "..", "\ufeff", "a\ufeffb", "\ufffe", "\uffff",
              "\U0010ffff", "\U0010fffe", "0b101", "0o17", "017", "0_7", "1_000", "+12", "-0", "0x_1", "0b2", "yes ", " yes", "~ ", "null#", "a\u0085b", "a\u2029b"]


def yaml_corpus(count, seed):
    rng = random.Random(seed)
    out = [""] * 1
    for w in YAML_WORDS:
        out.append(w)
        out.append(w + " " + w)
    alphabet = "abcdefgxyzEeTZ0123456789 .:-+_#'\"\\,;?!&*%@`[]{}|<>=~\t"
    for _ in range(count):
        k = rng.random()
        if k < 0.45:
            n = rng.randint(1, 12)
            out.append("".join(rng.choice(alphabet) for _ in range(n)))
        elif k < 0.85:
            n = rng.randint(1, 60)
            out.append(" ".join(rng.choice(YAML_WORDS) for _ in range(n)))
        else:
            n = rng.randint(30, 400)
            words = [rng.choice(["bright", "ukulele", "and", "light", "percussion,", "summery", "mood.", "a", "slow", "build", "with", "warm", "pads;", "intro", "-",
                                 "lush", "vocal", "harmonies", "(chorus)", "key:", "tempo", "120bpm", "#1"]) for _ in range(n)]
            out.append(" ".join(words))
    return out


def yaml_cmd(a):
    import yaml
    items = []
    for s in yaml_corpus(a.count, a.seed):
        try:
            text = yaml.dump({"caption": s}, allow_unicode=True, sort_keys=True)
        except Exception:
            continue
        items.append({"in": s, "out": text})
    with open(a.outfile, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"{len(items)} strings -> {a.outfile}")


PARSE_CASES = [
    "<think>\nbpm: 120\ncaption: A calm piano.\nduration: 30\nkeyscale: C major\nlanguage: en\ntimesignature: 4\n</think>",
    "<think>\nbpm: 100\ncaption: A bright and cheerful track.\n  The arrangement begins with a clean melody.\nduration: 150\nkeyscale: C major\nlanguage: unknown\ntimesignature: 4\n<|im_end|>",
    "<think>\n<foo>: x\nbpm: 5\n</think>",
    "caption: a\n<tag>: z\n  more\nduration: 5",
    "caption: a\n  <tag>: z\n  more\nduration: 5",
    "bpm: 100\n  duration: 200\nkeyscale: D minor",
    "<think>\nbpm: 100\n  duration: 200\n</think>",
    "caption:\n  A long\n  caption here\nduration: 5",
    "caption:\n\ttabbed\n\tcaption\nduration: 5",
    "bpm: 1\nbpm: 2\nbpm: 3",
    "bpm: fast\nduration: ten",
    "bpm: +120\nduration: 0042",
    "bpm: -5\nduration: 1_0",
    "bpm:  99 \nduration:   7  \nkeyscale:   G major   ",
    "genres: pop\nlyrics: [Verse]\nmood: calm\nbpm: 90",
    "caption: A: b: c\nduration: 4",
    "caption: line one\n\n  line two\n   \n  line three\nduration: 4",
    "caption: a\u3000b\n\u3000c\nduration: 4",
    "caption: a\u00a0b\n  c\u00a0\nduration: 4",
    "bpm: 5\n<|audio_code_1|><|audio_code_2|>",
    "no tags here\nbpm: 77\n<|audio_code_9|>bpm: 1",
    "<think>\r\nbpm: 120\r\ncaption: crlf text\r\nduration: 30\r\n</think>",
    "BPM: 5\nDuration : 6\nKeyScale: E minor\nLanguage: EN",
    "<think>\nbpm: 1\n</think>\n<think>\nbpm: 2\n</think>",
    "<think>\nbpm: 3\n",
    "<think>",
    "",
    "   \n\n  ",
    "timesignature: 3/4\nkeyscale: C#  major",
    "timesignature: 4/4\nlanguage: no",
    "timesignature: 12/8\nlanguage: yes",
    "timesignature: 4\nlanguage: 12",
    "keyscale: 123\nlanguage: true",
    "caption: 12345\nlanguage: null",
    "caption: ~\nkeyscale: 1.5",
    "caption: it's a 'quoted' caption\nlanguage: 2001-12-14",
    "caption: # not a comment\nduration: 8",
    "caption: - starts with a dash\nduration: 8",
    "caption: \u65e5\u672c\u8a9e\u306e\u8aac\u660e\nduration: 8",
    "caption: emoji \U0001f3b5 inside\nduration: 8",
    "bpm: 120 \n\n\nduration: 8",
    ":bpm 5\nbpm 6\n: 7",
    "bpm:\nduration:\nkeyscale:",
    "caption:\nbpm: 3",
    "  indented first line: 5\nbpm: 3",
    "<think>\nbpm: 120\ncaption: " + "word " * 40 + "end.\nduration: 30\n</think>",
]
PARSE_FRAGMENTS = ["bpm: 90", "bpm: x", "caption: a calm tune", "caption:", "  continued text", "\tmore", "duration: 30", "duration: 12.5", "keyscale: A minor", "language: en",
                   "language: no", "timesignature: 4", "timesignature: 3/4", "genres: jazz", "<think>", "</think>", "<|im_end|>", "<|audio_code_5|>", "noise: line",
                   "", "   ", "a: b", ": c", "bpm : 7", "caption: 7", "keyscale: 8"]


def parse_corpus(count, seed):
    rng = random.Random(seed)
    out = list(PARSE_CASES)
    for _ in range(count):
        n = rng.randint(1, 9)
        out.append("\n".join(rng.choice(PARSE_FRAGMENTS) for _ in range(n)))
    return out


def parse_cmd(a):
    from loguru import logger
    logger.remove()
    root = os.environ.get("ACESTEP_UPSTREAM")
    if not root or not os.path.isdir(os.path.join(root, "acestep")):
        sys.exit("set ACESTEP_UPSTREAM to a clone of https://github.com/ace-step/ACE-Step-1.5")
    sys.path.insert(0, os.path.abspath(root))
    from acestep.llm_inference import LLMHandler
    h = LLMHandler()
    items = []
    for text in parse_corpus(a.count, a.seed):
        metadata, _ = h.parse_lm_output(text)
        try:
            cot = h._format_metadata_as_cot(metadata)
        except Exception:
            cot = None
        items.append({"text": text, "metadata": metadata, "cot": cot})
    with open(a.outfile, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"{len(items)} texts -> {a.outfile}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("model_dir")
    c.add_argument("outdir")
    c.add_argument("--lm", default="acestep-5Hz-lm-1.7B")
    c.add_argument("--cases", help="comma-separated; default all")
    y = sub.add_parser("yaml")
    y.add_argument("outfile")
    y.add_argument("--count", type=int, default=4000)
    y.add_argument("--seed", type=int, default=1)
    pr = sub.add_parser("parse")
    pr.add_argument("outfile")
    pr.add_argument("--count", type=int, default=1500)
    pr.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "capture":
        capture(a)
    elif a.cmd == "parse":
        parse_cmd(a)
    else:
        yaml_cmd(a)


if __name__ == "__main__":
    main()
