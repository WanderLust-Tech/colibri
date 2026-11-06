#!/usr/bin/env python3
"""Convert a Gemma 4 mixture-of-experts checkpoint (google/gemma-4-26B-A4B-it) into a
colibri container that c/gemma4.c reads.

Only numpy is needed: the checkpoint's safetensors files are read through a memory
map and the container is written by hand, so no PyTorch install is required.

What is written (everything under the directory --out):

  config.json, generation_config.json, tokenizer.json, tokenizer_config.json,
  chat_template.jinja     copied from the checkpoint (the engine reads config.json;
                          the gateway reads the template files)
  gemma4_meta.json        the container's own description (below)
  model-NNNNN-of-MMMMM.safetensors + model.safetensors.index.json

Tensors, with the checkpoint's own names (prefix P = model.language_model.):

  P embed_tokens.weight           I8  [V, H]    per-row int8, tied with the output head
  P embed_tokens.weight.scale     F32 [V]
  P norm.weight                   F32 [H]
  P layers.N.<attention and dense MLP matrix>.weight        I8  [O, I]  per-row int8
  P layers.N.<same>.weight.scale                            F32 [O]
  P layers.N.router.proj.weight   F32 [E, H]                (tiny: kept exact)
  P layers.N.router.scale / .per_expert_scale / the norms / layer_scalar   F32
  P layers.N.experts.blob         U8  [E, B]    one contiguous record per expert:
        gate rows (F x H/2 bytes) | up rows | down rows (H x F/2 bytes)
        | gate scales (F x H/64 f32) | up scales | down scales (H x F/64 f32)
      The rows are "planar unsigned int4, gs64" exactly as c/expert_ffn.h reads them:
      a row is I/64 blocks of 32 bytes, byte k of a block holds element k in its low
      nibble and element k+32 in its high nibble, both stored as v+8; one f32 scale per
      (row, block); element = (nibble - 8) * scale. Values are confined to -7..7, so a
      quantize / dequantize / quantize round trip is exact.

The vision tower (model.vision_tower.*, model.embed_vision.*) is skipped on purpose and
counted in the report: this engine is text-only. A tensor name this script does not
know stops the conversion rather than being dropped silently.

    python c/tools/convert_gemma4.py --src /path/to/gemma-4-26B-A4B-it --out ~/colibri-models/gemma4-26b-a4b
    python c/tools/convert_gemma4.py --repo google/gemma-4-26B-A4B-it --out ~/colibri-models/gemma4-26b-a4b
"""
import argparse
import json
import mmap
import os
import re
import shutil
import struct
import sys
import time

import numpy as np

META_FORMAT = "colibri-gemma4-v1"
BLOCK = 64                    # int4 group size: fixed by c/expert_ffn.h
SHARD_BYTES = 4 << 30         # target size of an output shard

COPY_FILES = ("config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "chat_template.jinja", "processor_config.json")

DTYPE_NAMES = {"F32": np.float32, "F16": np.float16, "BF16": None, "I8": np.int8, "U8": np.uint8}


# ---------------------------------------------------------------------------- reading

class SafeFile:
    """One .safetensors file, memory mapped. Tensors are read lazily as float32."""

    def __init__(self, path):
        self.path = path
        self.fh = open(path, "rb")
        self.map = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)
        n = struct.unpack("<Q", self.map[:8])[0]
        header = json.loads(self.map[8:8 + n])
        header.pop("__metadata__", None)
        self.base = 8 + n
        self.tensors = header

    def close(self):
        self.map.close()
        self.fh.close()

    def shape(self, name):
        return tuple(self.tensors[name]["shape"])

    def dtype(self, name):
        return self.tensors[name]["dtype"]

    def raw(self, name):
        t = self.tensors[name]
        lo, hi = t["data_offsets"]
        return self.map[self.base + lo:self.base + hi]

    def array(self, name):
        """The whole tensor as float32 (a copy)."""
        t = self.tensors[name]
        buf = self.raw(name)
        kind = t["dtype"]
        if kind == "BF16":
            u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(t["shape"])
        if kind == "F16":
            return np.frombuffer(buf, dtype=np.float16).astype(np.float32).reshape(t["shape"])
        if kind == "F32":
            return np.frombuffer(buf, dtype=np.float32).reshape(t["shape"]).copy()
        raise SystemExit(f"{name}: unsupported checkpoint dtype {kind}")

    def slab(self, name, index):
        """Row `index` of the leading axis, as float32 (the experts are stacked [E, ...])."""
        t = self.tensors[name]
        shape = t["shape"]
        per = int(np.prod(shape[1:]))
        kind = t["dtype"]
        esz = {"BF16": 2, "F16": 2, "F32": 4}[kind]
        lo = self.base + t["data_offsets"][0] + index * per * esz
        buf = self.map[lo:lo + per * esz]
        if kind == "BF16":
            u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape[1:])
        if kind == "F16":
            return np.frombuffer(buf, dtype=np.float16).astype(np.float32).reshape(shape[1:])
        return np.frombuffer(buf, dtype=np.float32).reshape(shape[1:]).copy()


class Checkpoint:
    def __init__(self, directory):
        self.dir = directory
        index_path = os.path.join(directory, "model.safetensors.index.json")
        self.files = {}
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as handle:
                self.weight_map = json.load(handle)["weight_map"]
            for name in sorted(set(self.weight_map.values())):
                self.files[name] = SafeFile(os.path.join(directory, name))
        else:
            single = os.path.join(directory, "model.safetensors")
            if not os.path.exists(single):
                raise SystemExit(f"{directory}: neither model.safetensors.index.json nor model.safetensors")
            self.files["model.safetensors"] = SafeFile(single)
            self.weight_map = {n: "model.safetensors" for n in self.files["model.safetensors"].tensors}

    def names(self):
        return list(self.weight_map)

    def file_of(self, name):
        return self.files[self.weight_map[name]]

    def array(self, name):
        return self.file_of(name).array(name)

    def close(self):
        for f in self.files.values():
            f.close()


# ------------------------------------------------------------------------ quantizing

def quantize_int8_rows(w):
    """Per-row symmetric int8. Returns (q int8 [O, I], scale f32 [O])."""
    w = np.asarray(w, dtype=np.float32)
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8_rows(q, scale):
    return q.astype(np.float32) * scale[:, None]


def quantize_int4_planar(w):
    """Group-of-64 symmetric int4 in the planar layout. Returns
    (packed uint8 [O, I/2], scales f32 [O, I/64])."""
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    if cols % BLOCK:
        raise SystemExit(f"int4 rows need a multiple of {BLOCK} columns, got {cols}")
    blocks = w.reshape(rows, cols // BLOCK, BLOCK)
    amax = np.abs(blocks).max(axis=2)
    scale = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(blocks / scale[:, :, None]), -7, 7).astype(np.int8)
    u = (q + 8).astype(np.uint8).reshape(rows, cols // BLOCK, 2, BLOCK // 2)
    packed = u[:, :, 0, :] | (u[:, :, 1, :] << 4)                   # byte k: element k | element k+32 << 4
    return np.ascontiguousarray(packed.reshape(rows, cols // 2)), scale


def dequantize_int4_planar(packed, scale):
    rows = packed.shape[0]
    nb = scale.shape[1]
    p = packed.reshape(rows, nb, BLOCK // 2)
    lo = (p & 0x0F).astype(np.int32) - 8
    hi = (p >> 4).astype(np.int32) - 8
    q = np.concatenate([lo, hi], axis=2).astype(np.float32)          # elements 0..31 then 32..63
    return (q * scale[:, :, None]).reshape(rows, nb * BLOCK)


def expert_blob(gate, up, down):
    """One expert's record: the packed rows then the scales (see the module docstring)."""
    g4, gs = quantize_int4_planar(gate)
    u4, us = quantize_int4_planar(up)
    d4, ds = quantize_int4_planar(down)
    parts = [g4, u4, d4, gs, us, ds]
    return np.concatenate([np.ascontiguousarray(p).view(np.uint8).reshape(-1) for p in parts])


def blob_bytes(hidden, inter):
    return 2 * inter * (hidden // 2) + hidden * (inter // 2) + 4 * (2 * inter * (hidden // BLOCK) + hidden * (inter // BLOCK))


# ------------------------------------------------------------------------- classifying

LANG = "model.language_model."
LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")

MATRIX_TAILS = {
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
}
EXACT_TAILS = {
    "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight", "pre_feedforward_layernorm_2.weight",
    "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight", "layer_scalar",
    "router.proj.weight", "router.scale", "router.per_expert_scale",
}
EXPERT_TAILS = {"experts.gate_up_proj", "experts.down_proj"}
SKIPPED_PREFIXES = ("model.vision_tower.", "model.embed_vision.", "model.audio_tower.", "model.embed_audio.")


def classify(name):
    if name.startswith(SKIPPED_PREFIXES):
        return "skip"
    if name == LANG + "embed_tokens.weight":
        return "embed"
    if name == LANG + "norm.weight":
        return "exact"
    m = LAYER.match(name)
    if m:
        tail = m.group(2)
        if tail in MATRIX_TAILS:
            return "matrix"
        if tail in EXACT_TAILS:
            return "exact"
        if tail in EXPERT_TAILS:
            return "expert"
    return None


# --------------------------------------------------------------------------- writing

class ShardWriter:
    """Writes safetensors shards from (name, dtype, shape, producer) records.

    The header needs every tensor's size before any data is written, and every size here
    follows from its shape, so each record carries a producer that makes the bytes when
    its turn comes. Nothing is held beyond the tensor being written."""

    def __init__(self, out_dir):
        self.out = out_dir
        self.pending = []          # (name, dtype, shape, nbytes, producer)
        self.bytes = 0
        self.shards = []           # (filename, [names])

    ITEM = {"F32": 4, "I8": 1, "U8": 1}

    def add(self, name, dtype, shape, producer):
        nbytes = self.ITEM[dtype] * int(np.prod(shape))
        if self.pending and self.bytes + nbytes > SHARD_BYTES:
            self.flush()
        self.pending.append((name, dtype, list(shape), nbytes, producer))
        self.bytes += nbytes

    def flush(self):
        if not self.pending:
            return
        number = len(self.shards) + 1
        filename = f"model-{number:05d}.safetensors.part"
        header, offset = {}, 0
        for name, dtype, shape, nbytes, _ in self.pending:
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
        header["__metadata__"] = {"format": "pt"}
        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        blob += b" " * ((8 - len(blob) % 8) % 8)
        with open(os.path.join(self.out, filename), "wb") as handle:
            handle.write(struct.pack("<Q", len(blob)))
            handle.write(blob)
            for name, dtype, shape, nbytes, producer in self.pending:
                data = np.ascontiguousarray(producer())
                if data.nbytes != nbytes:
                    raise SystemExit(f"{name}: produced {data.nbytes} bytes, header says {nbytes}")
                handle.write(data.tobytes())
        self.shards.append((filename, [p[0] for p in self.pending]))
        self.pending, self.bytes = [], 0

    def finish(self):
        self.flush()
        total = len(self.shards)
        weight_map, size = {}, 0
        for i, (partial, names) in enumerate(self.shards, 1):
            final = f"model-{i:05d}-of-{total:05d}.safetensors"
            os.replace(os.path.join(self.out, partial), os.path.join(self.out, final))
            for n in names:
                weight_map[n] = final
            size += os.path.getsize(os.path.join(self.out, final))
        with open(os.path.join(self.out, "model.safetensors.index.json"), "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"metadata": {"total_size": size}, "weight_map": weight_map}, handle, indent=1)
        return size


# ----------------------------------------------------------------------- the converter

def text_config(config):
    return config.get("text_config", config)


def convert(src, out, log=print):
    with open(os.path.join(src, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    tc = text_config(config)
    hidden, inter = tc["hidden_size"], tc["moe_intermediate_size"]
    experts, layers = tc["num_experts"], tc["num_hidden_layers"]
    if tc.get("num_kv_shared_layers", 0):
        raise SystemExit("num_kv_shared_layers > 0 is not supported (the 26B-A4B has none)")
    if tc.get("hidden_size_per_layer_input", 0):
        raise SystemExit("per-layer embeddings are not supported (the 26B-A4B has none)")
    if not tc.get("enable_moe_block", False):
        raise SystemExit("this converter is for the mixture-of-experts Gemma 4 (enable_moe_block)")
    if hidden % BLOCK or inter % BLOCK:
        raise SystemExit(f"hidden_size {hidden} and moe_intermediate_size {inter} must be multiples of {BLOCK}")

    ck = Checkpoint(src)
    names = ck.names()
    kinds, unknown, skipped = {}, [], 0
    for n in names:
        k = classify(n)
        if k is None:
            unknown.append(n)
        elif k == "skip":
            skipped += 1
        else:
            kinds[n] = k
    if unknown:
        raise SystemExit("tensors this converter does not know (refusing to drop them silently):\n  "
                         + "\n  ".join(unknown[:20]) + (f"\n  ... and {len(unknown) - 20} more" if len(unknown) > 20 else ""))

    os.makedirs(out, exist_ok=True)
    w = ShardWriter(out)
    t0 = time.time()

    def add_f32(name):
        w.add(name, "F32", ck.file_of(name).shape(name), lambda n=name: ck.array(n))

    def add_int8(name):
        shape = ck.file_of(name).shape(name)
        cache = {}

        def both(n=name):
            if "q" not in cache:
                cache["q"], cache["s"] = quantize_int8_rows(ck.array(n))
            return cache

        w.add(name, "I8", shape, lambda: both()["q"])
        w.add(name + ".scale", "F32", (shape[0],), lambda: both()["s"])

    add_int8(LANG + "embed_tokens.weight")
    add_f32(LANG + "norm.weight")
    for layer in range(layers):
        prefix = f"{LANG}layers.{layer}."
        for tail in sorted(MATRIX_TAILS):
            if prefix + tail in kinds:
                add_int8(prefix + tail)
        for tail in sorted(EXACT_TAILS):
            if prefix + tail in kinds:
                add_f32(prefix + tail)
        gate_up, down = prefix + "experts.gate_up_proj", prefix + "experts.down_proj"
        if gate_up not in kinds or down not in kinds:
            raise SystemExit(f"layer {layer}: expert tensors missing")
        fg, fd = ck.file_of(gate_up), ck.file_of(down)
        if fg.shape(gate_up) != (experts, 2 * inter, hidden) or fd.shape(down) != (experts, hidden, inter):
            raise SystemExit(f"layer {layer}: unexpected expert shapes {fg.shape(gate_up)} {fd.shape(down)}")

        def make(g=gate_up, d=down, fg=fg, fd=fd, layer=layer):
            out_blob = np.empty((experts, blob_bytes(hidden, inter)), dtype=np.uint8)
            for e in range(experts):
                gu = fg.slab(g, e)
                out_blob[e] = expert_blob(gu[:inter], gu[inter:], fd.slab(d, e))
            log(f"  layer {layer + 1}/{layers} experts quantized ({time.time() - t0:.0f}s)")
            return out_blob

        w.add(prefix + "experts.blob", "U8", (experts, blob_bytes(hidden, inter)), make)
    for n, k in kinds.items():
        if k == "matrix" and not n.startswith(LANG + "layers."):
            raise SystemExit(f"unexpected matrix outside the layers: {n}")

    size = w.finish()
    ck.close()
    for f in COPY_FILES:
        if os.path.exists(os.path.join(src, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(out, f))
    meta = {
        "format": META_FORMAT,
        "dense_bits": 8, "expert_bits": 4, "expert_group": BLOCK,
        "expert_blob_bytes": blob_bytes(hidden, inter),
        "layers": layers, "experts": experts,
        "skipped_tensors": skipped,
        "skipped_note": "vision tower and projector: this engine is text-only",
    }
    with open(os.path.join(out, "gemma4_meta.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, indent=1)
    log(f"wrote {size / 1e9:.2f} GB in {time.time() - t0:.0f}s; skipped {skipped} vision tensors")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--src", help="a local checkpoint directory (config.json + safetensors)")
    where.add_argument("--repo", help="a Hugging Face repo to download first, e.g. google/gemma-4-26B-A4B-it")
    ap.add_argument("--out", required=True, help="the container directory to write")
    ap.add_argument("--keep-source", action="store_true", help="with --repo: keep the downloaded checkpoint")
    args = ap.parse_args()
    src = args.src
    downloaded = None
    if args.repo:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("--repo needs huggingface_hub (pip install huggingface_hub)")
        downloaded = args.out.rstrip("/\\") + ".source"
        print(f"downloading {args.repo} to {downloaded} (about 52 GB)")
        src = snapshot_download(args.repo, local_dir=downloaded,
                                allow_patterns=["*.json", "*.safetensors", "*.jinja"])
    convert(src, args.out)
    if downloaded and not args.keep_source:
        shutil.rmtree(downloaded, ignore_errors=True)
        print(f"removed {downloaded}")


if __name__ == "__main__":
    sys.exit(main())
