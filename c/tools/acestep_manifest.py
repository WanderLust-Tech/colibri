#!/usr/bin/env python3
"""Tensor manifest of an ACE-Step 1.5 checkpoint, read from the safetensors headers only.

    acestep_manifest.py MODEL_DIR [-o tools/acestep_manifest.json]
    acestep_manifest.py --check MODEL_DIR            compare a directory with the stored manifest

MODEL_DIR holds the three components the turbo model is made of, in the layout of the
Hugging Face repository ACE-Step/Ace-Step1.5:

    acestep-v15-turbo/model.safetensors        DiT, condition encoders, FSQ tokenizer
    vae/diffusion_pytorch_model.safetensors    Oobleck VAE (weight-normed convolutions)
    Qwen3-Embedding-0.6B/model.safetensors     text encoder

Each tensor name is stored with its layer indices collapsed to N ("decoder.layers.N.mlp.up_proj.weight")
together with its dtype, its shape and how many times the pattern occurs, so the manifest says "24 DiT
layers, each with these tensors" in a few hundred lines. Where the layers of one stack differ in shape
(the VAE's blocks) the names are kept as they are. The tiny random fixture
(make_acestep_tiny.py) must have exactly the same patterns, dtypes and tensor ranks as the real
checkpoint: tests/test_acestep_fixture.py holds it to that, and the C loader asserts the same names.
Only the Python standard library is needed: no torch, no safetensors package.
"""
import argparse
import collections
import json
import os
import re
import struct
import sys

COMPONENTS = {
    "dit": "acestep-v15-turbo/model.safetensors",
    "vae": "vae/diffusion_pytorch_model.safetensors",
    "text": "Qwen3-Embedding-0.6B/model.safetensors",
}
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, "acestep_manifest.json")
# A header is a JSON object: refuse one that claims to be absurd instead of reading it all.
MAX_HEADER = 64 << 20


def read_header(path):
    """The tensor table of a safetensors file: {name: {"dtype", "shape", ...}}, without __metadata__."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: too short to be a safetensors file")
        (size,) = struct.unpack("<Q", raw)
        if size > MAX_HEADER:
            raise ValueError(f"{path}: header of {size} bytes")
        table = json.loads(f.read(size))
    table.pop("__metadata__", None)
    return table


def collapse(table):
    """{pattern: {"dtype", "shape", "count"}} with every ".<digits>." in a name written ".N.".

    Tensors collapse only when they are the same layer of the same stack, which means the same dtype
    and the same shape. A stack whose layers differ in width is kept name by name instead: the VAE
    decoder's five blocks are 1024, 512, 256, 128 and 128 channels wide, and one pattern saying
    "decoder.block.N.conv_t1.bias" with a single shape would be wrong for four of them. A pattern that
    occurs once keeps count 1.
    """
    groups = collections.OrderedDict()
    for name in sorted(table):
        pattern = re.sub(r"\.\d+(?=\.)", ".N", name)
        groups.setdefault(pattern, []).append(name)
    out = collections.OrderedDict()
    for pattern, names in groups.items():
        shapes = {(table[n]["dtype"], tuple(table[n]["shape"])) for n in names}
        if len(shapes) == 1:
            dtype, shape = next(iter(shapes))
            out[pattern] = {"dtype": dtype, "shape": list(shape), "count": len(names)}
        else:
            for n in names:
                out[n] = {"dtype": table[n]["dtype"], "shape": list(table[n]["shape"]), "count": 1}
    return out


def build(model_dir):
    manifest = {}
    for component, rel in COMPONENTS.items():
        path = os.path.join(model_dir, *rel.split("/"))
        manifest[component] = collapse(read_header(path))
    return manifest


def structure(patterns):
    """What a tiny fixture must share with the real checkpoint: the pattern names, dtypes and ranks."""
    return {p: (e["dtype"], len(e["shape"])) for p, e in patterns.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("-o", "--out", default=DEFAULT_MANIFEST)
    ap.add_argument("--check", action="store_true", help="compare MODEL_DIR with the stored manifest, write nothing")
    a = ap.parse_args()
    manifest = build(a.model_dir)
    if a.check:
        with open(a.out, encoding="utf-8") as f:
            stored = json.load(f)
        bad = 0
        for component in COMPONENTS:
            if manifest[component] != stored[component]:
                bad += 1
                have, want = set(manifest[component]), set(stored[component])
                print(f"{component}: differs; only here {sorted(have - want)[:5]}, only stored {sorted(want - have)[:5]}")
        print("manifest matches" if not bad else f"{bad} component(s) differ")
        sys.exit(1 if bad else 0)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    for component, patterns in manifest.items():
        total = sum(e["count"] for e in patterns.values())
        print(f"{component}: {len(patterns)} patterns, {total} tensors")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
