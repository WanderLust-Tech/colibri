#!/usr/bin/env python3
"""Writes tokenizer.json for a Qwen3 speech checkpoint (Qwen3-ASR, Qwen3-TTS) from the files it ships.

    python tools/qwen3_tokenizer.py MODEL_DIR [--out FILE]

These checkpoints carry a Qwen2 byte-level BPE as vocab.json + merges.txt + tokenizer_config.json (the added tokens), and no
tokenizer.json, which is what colibri's engines read. This builds the same file transformers' Qwen2TokenizerFast does:
NFC normalisation, the Qwen2 split pattern, byte-level pre-tokenising, the byte-level post-processor and decoder, the vocabulary and the
merges in order, and the added tokens with their ids and `special` flags. Stdlib only. `coli model install` runs it after the download
(the catalogue's post_install), and it is safe to run again: an existing tokenizer.json is left alone unless --force.
"""
import argparse
import json
import os
import sys

SPLIT = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")


def build(model_dir):
    """The tokenizer.json object for MODEL_DIR; raises FileNotFoundError or ValueError with a message that says what is wrong."""
    with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as handle:
        vocab = json.load(handle)
    merges = []
    with open(os.path.join(model_dir, "merges.txt"), encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#version"):
                continue
            pair = line.split(" ")
            if len(pair) != 2:
                raise ValueError(f"merges.txt: a line that is not a pair of tokens: {line[:40]!r}")
            merges.append(pair)
    with open(os.path.join(model_dir, "tokenizer_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    added = []
    for key, value in (config.get("added_tokens_decoder") or {}).items():
        added.append({"id": int(key), "content": value["content"], "single_word": bool(value.get("single_word", False)),
                      "lstrip": bool(value.get("lstrip", False)), "rstrip": bool(value.get("rstrip", False)),
                      "normalized": bool(value.get("normalized", False)), "special": bool(value.get("special", False))})
    added.sort(key=lambda item: item["id"])
    if not vocab or not merges or not added:
        raise ValueError("vocab.json, merges.txt and the added tokens of tokenizer_config.json must all be present")
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": SPLIT}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": False, "vocab": vocab, "merges": merges},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=None, help="where to write (default: MODEL_DIR/tokenizer.json)")
    ap.add_argument("--force", action="store_true", help="replace an existing file")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(args.model_dir, "tokenizer.json")
    if os.path.exists(out) and not args.force:
        print(f"{out} exists; left as it is")
        return 0
    try:
        data = build(args.model_dir)
    except (OSError, ValueError, KeyError) as error:
        print(f"qwen3_tokenizer: {error}", file=sys.stderr)
        return 1
    temporary = out + ".part"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, out)
    print(f"wrote {out}: {len(data['model']['vocab'])} tokens, {len(data['model']['merges'])} merges, {len(data['added_tokens'])} added tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
