#!/usr/bin/env python3
"""Reference token ids for the ACE-Step text encoder's tokenizer (c/acestep_text.h), from the library upstream uses.

    make_acestep_tok_ref.py MODEL_DIR OUT.json

MODEL_DIR holds Qwen3-Embedding-0.6B/ (the real one, or the tiny fixture's). The ids are what upstream's handler gets
from `AutoTokenizer.from_pretrained(...)(text, padding="longest", truncation=True, max_length=N)`: that call runs the
tokenizer's NFC normalizer, splits the added tokens out of the text (a literal "<|endoftext|>" is one token), applies
the pre-tokenizer and BPE, and then the post-processor appends one <|endoftext|> to the sequence, with truncation
leaving room for it. OUT.json is a list of {"text", "max_length", "ids"}.

The cases are the prompts as upstream builds them (the caption template with its metas, the lyric template with its
language), and strings chosen to break a tokenizer port: other scripts and emoji, accents in both Unicode forms, runs of
digits (Qwen splits them one at a time), whitespace and newline runs, contractions in both cases, tokens that look like
special ones, empty and one-character strings, and texts long enough to be cut at the 256 and 2048 limits the handler
uses and at tiny ones.

Needs the reference venv (tools/requirements-acestep-ref.txt); upstream's own code is not needed, only the tokenizer.
"""
import json
import os
import sys

# upstream's templates (acestep/constants.py SFT_GEN_PROMPT and handler/prompt_utils.py _format_lyrics)
INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
CAPTION_TEMPLATE = "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n"
LYRIC_TEMPLATE = "# Languages\n{}\n\n# Lyric\n{}<|endoftext|>"


def metas(bpm="N/A", ts="N/A", key="N/A", seconds=10):
    return f"- bpm: {bpm}\n- timesignature: {ts}\n- keyscale: {key}\n- duration: {seconds} seconds\n"


CAPTIONS = [
    "calm piano with soft strings",
    "upbeat pop with bright synths",
    "",
    "lo-fi hip hop, 85 bpm, vinyl crackle, mellow keys & dusty drums",
    "ミニマルなテクノ、深いベース、暗い雰囲気",
    "épique orchestral — cordes, cuivres & timbales 🎻🎺",
    "x" * 4000,
]
LYRICS = [
    "[Instrumental]",
    "[Verse 1]\nwalking down the empty street\nthe city sleeps beneath my feet\n[Chorus]\nsing it out loud tonight",
    "",
    "[Verse]\n你好，世界\n我们一起唱歌\n[Chorus]\nla la la 1 2 3 4",
    "[Verse]\ncafé crème — naïve façade\ncafé crème",
    "don't stop, I'LL be there; we've got 1,000.50 reasons\n\n\n[Bridge]\n\t tabbed   line  ",
    "word " * 3000,
]
STRESS = [
    "", " ", "\n", "\n\n\n", "   ", "a", "a b", "1", "12345 6789", "1,000.50", "3.14159", "2026-11-01 12:30",
    "don't", "DON'T", "I'LL", "we've", "it's", "rock'n'roll",
    "café", "café", "Å", "Å", "ﬁ ligature", "é́",
    "你好，世界", "こんにちは世界", "안녕하세요", "שלום עולם", "مرحبا بالعالم", "Привет, мир", "สวัสดีชาวโลก",
    "🎵", "🎻🎺🥁", "a 🎵 b", "👨‍👩‍👧", "​", "﻿", " ",
    "<|endoftext|>", "x<|endoftext|>y", "<|endoftext|><|endoftext|>", "<|im_start|>user\nhi<|im_end|>", "<|not_a_token|>", "<|", "|>",
    "#", "# ", "# Instruction\n", "##", "- bpm: 120\n", "  leading", "trailing  ", "tab\there", "line1\r\nline2", "a\n\nb\n",
    "[Verse 1]", "[Chorus]\n", "(whisper)", "{json: [1, 2.5, null]}", "https://example.com/a?b=c&d=e",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "the quick brown fox jumps over the lazy dog " * 40,
]


def build_cases():
    cases = []
    for cap in CAPTIONS:
        for seconds, bpm, key in ((10, "N/A", "N/A"), (225, 120, "C major"), (9, "N/A", "A minor")):
            cases.append((CAPTION_TEMPLATE.format(INSTRUCTION, cap, metas(bpm, "N/A", key, seconds)), 256))
    for lyr in LYRICS:
        for lang in ("en", "unknown", "zh"):
            cases.append((LYRIC_TEMPLATE.format(lang, lyr), 2048))
    for s in STRESS:
        cases.append((s, 256))
    # truncation at other lengths, so that the rule is tested where a prompt is only just longer than the limit
    for n in (1, 2, 3, 8, 16, 60, 61):
        cases.append(("the quick brown fox jumps over the lazy dog " * 5, n))
        cases.append((CAPTION_TEMPLATE.format(INSTRUCTION, CAPTIONS[0], metas()), n))
    return cases


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_dir, out = sys.argv[1:3]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "Qwen3-Embedding-0.6B"))
    rows = []
    for text, max_length in build_cases():
        enc = tok(text, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        rows.append({"text": text, "max_length": max_length, "ids": enc.input_ids[0].tolist()})
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, ensure_ascii=False)
    longest = max(len(r["ids"]) for r in rows)
    print(f"{len(rows)} cases, longest {longest} ids, {sum(len(r['ids']) == r['max_length'] for r in rows)} cut at their limit -> {out}")


if __name__ == "__main__":
    main()
