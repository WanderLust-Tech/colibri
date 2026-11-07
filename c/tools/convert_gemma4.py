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

The vision tower and its projector (model.vision_tower.*, model.embed_vision.*) and the audio tower
(model.audio_tower.*, model.embed_audio.*: the E2B and E4B have one, the 26B-A4B and the 31B do not) are
carried when the checkpoint has them (--no-vision and --no-audio leave them out): the matrices as
per-row int8 with the same names, the norms, convolutions, position tables, clip limits and the like as
F32. A tensor name this script does not know stops the conversion rather than being dropped silently.

    python c/tools/convert_gemma4.py --src /path/to/gemma-4-26B-A4B-it --out ~/colibri-models/gemma4-26b-a4b
    python c/tools/convert_gemma4.py --repo google/gemma-4-26B-A4B-it --out ~/colibri-models/gemma4-26b-a4b
    # a container converted before the vision tower was carried: add just the tower (a minute)
    python c/tools/convert_gemma4.py --src /path/to/gemma-4-26B-A4B-it --out ~/colibri-models/gemma4-26b-a4b --add-vision
    python c/tools/convert_gemma4.py --src /path/to/gemma-4-E2B-it --out ~/colibri-models/gemma4-e2b --add-audio
"""
import argparse
import json
import os
import re
import shutil
import struct
import sys
import time

try:
    import numpy as np
except ImportError:
    sys.exit("convert_gemma4.py needs numpy: pip install numpy (and huggingface_hub for --repo)")

META_FORMAT = "colibri-gemma4-v1"
BLOCK = 64                    # int4 group size: fixed by c/expert_ffn.h
SHARD_BYTES = 4 << 30         # target size of an output shard

COPY_FILES = ("config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "chat_template.jinja", "processor_config.json")

DTYPE_NAMES = {"F32": np.float32, "F16": np.float16, "BF16": None, "I8": np.int8, "U8": np.uint8}


# ---------------------------------------------------------------------------- reading

class SafeFile:
    """One .safetensors file. Tensors are read with plain sequential reads, never through a memory map:
    a checkpoint is often on a spinning disk (or a USB one), where the page faults of a mapped file arrive in
    random order and run at a few MB/s, and one read of a whole tensor runs at the disk's speed."""

    def __init__(self, path):
        self.path = path
        self.fh = open(path, "rb", buffering=0)
        n = struct.unpack("<Q", self.fh.read(8))[0]
        header = json.loads(self.fh.read(n))
        header.pop("__metadata__", None)
        self.base = 8 + n
        self.tensors = header

    def close(self):
        self.fh.close()

    def shape(self, name):
        return tuple(self.tensors[name]["shape"])

    def dtype(self, name):
        return self.tensors[name]["dtype"]

    def _read(self, offset, nbytes):
        buf = np.empty(nbytes, dtype=np.uint8)
        view = memoryview(buf)
        self.fh.seek(offset)
        got = 0
        while got < nbytes:
            n = self.fh.readinto(view[got:got + (64 << 20)])
            if not n:
                raise SystemExit(f"{self.path}: ended {nbytes - got} bytes early")
            got += n
        return buf

    def raw(self, name):
        lo, hi = self.tensors[name]["data_offsets"]
        return self._read(self.base + lo, hi - lo)

    @staticmethod
    def _decode(buf, kind, shape):
        if kind == "BF16":
            u = buf.view(np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        if kind == "F16":
            return buf.view(np.float16).astype(np.float32).reshape(shape)
        if kind == "F32":
            return buf.view(np.float32).reshape(shape).copy()
        raise SystemExit(f"unsupported checkpoint dtype {kind}")

    def array(self, name):
        """The whole tensor as float32."""
        t = self.tensors[name]
        return self._decode(self.raw(name), t["dtype"], t["shape"])

    def stack(self, name):
        """A stacked tensor ([E, ...]) read once, whole, and handed out one slab at a time as float32."""
        t = self.tensors[name]
        buf = self.raw(name)
        shape = t["shape"]
        esz = {"BF16": 2, "F16": 2, "F32": 4}[t["dtype"]]
        per = int(np.prod(shape[1:])) * esz

        def slab(index):
            return self._decode(buf[index * per:(index + 1) * per], t["dtype"], shape[1:])
        return slab


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


def quantize_tensor_rows(ck, name, rows_per_block=2048):
    """quantize_int8_rows of a checkpoint tensor, a block of rows at a time: the per-layer embedding table of the
    E4B is 5.6 GB in bf16 and 11 GB as float32, and a machine that converts it should not need both."""
    f = ck.file_of(name)
    rows, cols = f.shape(name)
    kind = f.dtype(name)
    item = {"BF16": 2, "F16": 2, "F32": 4}[kind]
    raw = f.raw(name)
    q = np.empty((rows, cols), dtype=np.int8)
    scale = np.empty(rows, dtype=np.float32)
    for r0 in range(0, rows, rows_per_block):
        r1 = min(rows, r0 + rows_per_block)
        block = f._decode(raw[r0 * cols * item:r1 * cols * item], kind, (r1 - r0, cols))
        q[r0:r1], scale[r0:r1] = quantize_int8_rows(block)
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
    "per_layer_input_gate.weight", "per_layer_projection.weight",           # the per-layer-embedding block (E2B, E4B)
}
EXACT_TAILS = {
    "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight", "pre_feedforward_layernorm_2.weight",
    "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight", "layer_scalar", "post_per_layer_input_norm.weight",
    "router.proj.weight", "router.scale", "router.per_expert_scale",
}
EXPERT_TAILS = {"experts.gate_up_proj", "experts.down_proj"}
AUDIO_PREFIXES = ("model.audio_tower.", "model.embed_audio.")
# the audio tower's matrices (clipped linears, the relative-position projection, the subsampling and output projections and the
# projector into the text model): per-row int8. Everything else of it (norms, the two convolutions, the depthwise kernels, the
# per-dimension scales, the output bias, the clip limits) stays exact.
AUDIO_MATRIX = re.compile(r"(\.linear\.weight|self_attn\.relative_k_proj\.weight|input_proj_linear\.weight|"
                          r"^model\.audio_tower\.output_proj\.weight|^model\.embed_audio\.embedding_projection\.weight)$")
_LIN = r"(?:linear\.weight|(?:input|output)_(?:min|max))"       # a clipped linear: its matrix, or one of its four limits
AUDIO_KNOWN = re.compile(
    r"^model\.audio_tower\.(?:"
    r"subsample_conv_projection\.(?:layer[01]\.(?:conv|norm)\.weight|input_proj_linear\.weight)"
    r"|output_proj\.(?:weight|bias)"
    r"|layers\.\d+\.(?:"
    r"feed_forward[12]\.(?:ffw_layer_[12]\." + _LIN + r"|pre_layer_norm\.weight|post_layer_norm\.weight)"
    r"|norm_pre_attn\.weight|norm_post_attn\.weight|norm_out\.weight"
    r"|self_attn\.(?:(?:q_proj|k_proj|v_proj|post)\." + _LIN + r"|relative_k_proj\.weight|per_dim_scale)"
    r"|lconv1d\.(?:(?:linear_start|linear_end)\." + _LIN + r"|depthwise_conv1d\.weight|pre_layer_norm\.weight|conv_norm\.weight)"
    r"))$|^model\.embed_audio\.embedding_projection\.weight$")
SKIPPED_PREFIXES = ()
VISION_PREFIXES = ("model.vision_tower.", "model.embed_vision.")
VISION_LAYER = re.compile(r"^model\.vision_tower\.encoder\.layers\.(\d+)\.(.+)$")
VISION_MATRIX_TAILS = {
    "self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight", "self_attn.v_proj.linear.weight",
    "self_attn.o_proj.linear.weight", "mlp.gate_proj.linear.weight", "mlp.up_proj.linear.weight",
    "mlp.down_proj.linear.weight",
}
VISION_EXACT_TAILS = {
    "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
}
VISION_MATRICES = {"model.vision_tower.patch_embedder.input_proj.weight",
                   "model.embed_vision.embedding_projection.weight"}
VISION_EXACT = {"model.vision_tower.patch_embedder.position_embedding_table",
                "model.vision_tower.std_bias", "model.vision_tower.std_scale"}
VISION_CLIP = re.compile(r"\.(input|output)_(min|max)$")      # a clipped linear's four limits (E2B, E4B): scalars, kept exact


def classify(name):
    if name.startswith(AUDIO_PREFIXES):
        if not AUDIO_KNOWN.match(name):
            return None
        return "amatrix" if AUDIO_MATRIX.search(name) else "aexact"
    if name.startswith(VISION_PREFIXES):
        if name in VISION_MATRICES:
            return "vmatrix"
        if name in VISION_EXACT:
            return "vexact"
        m = VISION_LAYER.match(name)
        if m and m.group(2) in VISION_MATRIX_TAILS:
            return "vmatrix"
        if m and (m.group(2) in VISION_EXACT_TAILS or VISION_CLIP.search(m.group(2))):
            return "vexact"
        return None
    if name in (LANG + "embed_tokens.weight", LANG + "embed_tokens_per_layer.weight"):
        return "embed"
    if name == LANG + "per_layer_model_projection.weight":
        return "topmatrix"
    if name in (LANG + "norm.weight", LANG + "per_layer_projection_norm.weight"):
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

    def __init__(self, out_dir, prefix="model"):
        self.out = out_dir
        self.prefix = prefix
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
        filename = f"{self.prefix}-{number:05d}.safetensors.part"
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
            final = f"{self.prefix}-{i:05d}-of-{total:05d}.safetensors"
            os.replace(os.path.join(self.out, partial), os.path.join(self.out, final))
            for n in names:
                weight_map[n] = final
            size += os.path.getsize(os.path.join(self.out, final))
        with open(os.path.join(self.out, "model.safetensors.index.json"), "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"metadata": {"total_size": size}, "weight_map": weight_map}, handle, indent=1)
        return size

    def finish_into(self, index):
        """Finish, merging this writer's shards into an existing index (a dict, as the index file holds it)."""
        self.flush()
        total = len(self.shards)
        for i, (partial, names) in enumerate(self.shards, 1):
            final = f"{self.prefix}-{i:05d}-of-{total:05d}.safetensors"
            os.replace(os.path.join(self.out, partial), os.path.join(self.out, final))
            for n in names:
                index["weight_map"][n] = final
            index["metadata"]["total_size"] += os.path.getsize(os.path.join(self.out, final))
        with open(os.path.join(self.out, "model.safetensors.index.json"), "w", encoding="utf-8", newline="\n") as handle:
            json.dump(index, handle, indent=1)


# ----------------------------------------------------------------------- the converter

def text_config(config):
    return config.get("text_config", config)


def vision_summary(config):
    """The tower's shape, for gemma4_meta.json (the gateway reads it), or None."""
    vc = config.get("vision_config")
    if not isinstance(vc, dict):
        return None
    rope = vc.get("rope_parameters") or {}
    return {"hidden": vc["hidden_size"], "layers": vc["num_hidden_layers"], "heads": vc["num_attention_heads"],
            "head_dim": vc.get("head_dim") or vc["hidden_size"] // vc["num_attention_heads"],
            "intermediate": vc["intermediate_size"], "patch": vc["patch_size"],
            "pool": vc["pooling_kernel_size"], "positions": vc["position_embedding_size"],
            "eps": vc.get("rms_norm_eps", 1e-6), "rope_theta": rope.get("rope_theta", 100.0),
            "standardize": bool(vc.get("standardize", True)),
            "max_soft_tokens": config.get("vision_soft_tokens_per_image", 280)}


def audio_summary(config):
    """The audio tower's shape, for gemma4_meta.json, or None."""
    ac = config.get("audio_config")
    if not isinstance(ac, dict):
        return None
    return {"hidden": ac["hidden_size"], "layers": ac["num_hidden_layers"], "heads": ac["num_attention_heads"],
            "chunk": ac.get("attention_chunk_size", 12), "left": ac.get("attention_context_left", 13),
            "right": ac.get("attention_context_right", 0), "kernel": ac.get("conv_kernel_size", 5),
            "channels": list(ac.get("subsampling_conv_channels", (128, 32))), "out_dims": ac.get("output_proj_dims", ac["hidden_size"])}


def add_tensor_producers(w, ck, names_wanted):
    """Queue the I8 (+ .scale) / F32 tensors of `names_wanted` (name -> kind) on writer w."""
    for name in sorted(names_wanted):
        shape = ck.file_of(name).shape(name)
        if names_wanted[name] in ("vmatrix", "amatrix"):
            if len(shape) != 2:
                raise SystemExit(f"{name}: expected a matrix, got shape {shape}")
            cache = {}

            def both(n=name, cache=cache):
                if "q" not in cache:
                    cache["q"], cache["s"] = quantize_tensor_rows(ck, n)
                return cache

            w.add(name, "I8", shape, lambda b=both: b()["q"])
            w.add(name + ".scale", "F32", (shape[0],), lambda b=both: b()["s"])
        else:
            w.add(name, "F32", shape, lambda n=name: ck.array(n))


def add_tower(src, out, which, log=print):
    """Add the vision or the audio tower ("vision" / "audio") to a container that was converted without it: one more
    shard, the index and the meta file updated. The text weights are not touched."""
    prefixes = VISION_PREFIXES if which == "vision" else AUDIO_PREFIXES
    with open(os.path.join(src, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    summary = vision_summary(config) if which == "vision" else audio_summary(config)
    if summary is None:
        raise SystemExit(f"{src}: config.json has no {which}_config")
    meta_path = os.path.join(out, "gemma4_meta.json")
    index_path = os.path.join(out, "model.safetensors.index.json")
    if not (os.path.exists(meta_path) and os.path.exists(index_path)):
        raise SystemExit(f"{out}: not a Gemma 4 container (gemma4_meta.json or the index is missing)")
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)
    if any(n.startswith(prefixes) for n in index["weight_map"]):
        raise SystemExit(f"{out}: already has the {which} tower")
    ck = Checkpoint(src)
    wanted, unknown = {}, []
    for n in ck.names():
        if n.startswith(prefixes):
            k = classify(n)
            if k is None:
                unknown.append(n)
            else:
                wanted[n] = k
    if unknown:
        raise SystemExit(f"{which} tensors this converter does not know:\n  " + "\n  ".join(unknown[:20]))
    if not wanted:
        raise SystemExit(f"{src}: no {which} tensors in the checkpoint")
    t0 = time.time()
    w = ShardWriter(out, prefix=which)
    add_tensor_producers(w, ck, wanted)
    w.finish_into(index)
    ck.close()
    for f in COPY_FILES:                    # the config now carries the tower's section, the processor config its sizes
        if f in ("config.json", "processor_config.json") and os.path.exists(os.path.join(src, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(out, f))
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    meta[which] = summary
    meta.pop("skipped_note", None)
    meta["skipped_tensors"] = max(0, meta.get("skipped_tensors", 0) - len(wanted))
    with open(meta_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, indent=1)
    log(f"added {len(wanted)} {which} tensors in {time.time() - t0:.0f}s")
    return meta


def add_vision(src, out, log=print):
    return add_tower(src, out, "vision", log)


def convert(src, out, log=print, with_vision=True, with_audio=True):
    with open(os.path.join(src, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    tc = text_config(config)
    moe = bool(tc.get("enable_moe_block", False))          # the 26B-A4B; the others (31B, E2B, E4B) are dense
    hidden, layers = tc["hidden_size"], tc["num_hidden_layers"]
    inter, experts = (tc["moe_intermediate_size"], tc["num_experts"]) if moe else (0, 0)
    if moe and (hidden % BLOCK or inter % BLOCK):
        raise SystemExit(f"hidden_size {hidden} and moe_intermediate_size {inter} must be multiples of {BLOCK}")

    ck = Checkpoint(src)
    names = ck.names()
    kinds, unknown, skipped = {}, [], 0
    vision_names = {}
    for n in names:
        k = classify(n)
        if k is None:
            unknown.append(n)
        elif k == "skip" or (k in ("vmatrix", "vexact") and not with_vision) or (k in ("amatrix", "aexact") and not with_audio):
            skipped += 1
        elif k in ("vmatrix", "vexact", "amatrix", "aexact"):
            vision_names[n] = k                                  # (the towers' tensors, vision and audio alike)
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
                cache["q"], cache["s"] = quantize_tensor_rows(ck, n)
            return cache

        w.add(name, "I8", shape, lambda: both()["q"])
        w.add(name + ".scale", "F32", (shape[0],), lambda: both()["s"])

    add_int8(LANG + "embed_tokens.weight")
    add_f32(LANG + "norm.weight")
    for top in sorted(n for n, k in kinds.items() if n.startswith(LANG) and not n.startswith(LANG + "layers.")):
        if top in (LANG + "embed_tokens.weight", LANG + "norm.weight"):
            continue
        (add_f32 if kinds[top] == "exact" else add_int8)(top)      # the per-layer embedding table, its projection and norm
    for layer in range(layers):
        prefix = f"{LANG}layers.{layer}."
        for tail in sorted(MATRIX_TAILS):
            if prefix + tail in kinds:
                add_int8(prefix + tail)
        for tail in sorted(EXACT_TAILS):
            if prefix + tail in kinds:
                add_f32(prefix + tail)
        if not moe:
            continue
        gate_up, down = prefix + "experts.gate_up_proj", prefix + "experts.down_proj"
        if gate_up not in kinds or down not in kinds:
            raise SystemExit(f"layer {layer}: expert tensors missing")
        fg, fd = ck.file_of(gate_up), ck.file_of(down)
        if fg.shape(gate_up) != (experts, 2 * inter, hidden) or fd.shape(down) != (experts, hidden, inter):
            raise SystemExit(f"layer {layer}: unexpected expert shapes {fg.shape(gate_up)} {fd.shape(down)}")

        def make(g=gate_up, d=down, fg=fg, fd=fd, layer=layer):
            out_blob = np.empty((experts, blob_bytes(hidden, inter)), dtype=np.uint8)
            gate_up_slab, down_slab = fg.stack(g), fd.stack(d)           # each read once, front to back
            for e in range(experts):
                gu = gate_up_slab(e)
                out_blob[e] = expert_blob(gu[:inter], gu[inter:], down_slab(e))
            log(f"  layer {layer + 1}/{layers} experts quantized ({time.time() - t0:.0f}s)")
            return out_blob

        w.add(prefix + "experts.blob", "U8", (experts, blob_bytes(hidden, inter)), make)
    for n, k in kinds.items():
        if k == "matrix" and not n.startswith(LANG + "layers."):
            raise SystemExit(f"unexpected matrix outside the layers: {n}")
        if k == "expert" and not moe:
            raise SystemExit(f"{n}: expert tensors in a checkpoint whose config says enable_moe_block is false")

    add_tensor_producers(w, ck, vision_names)
    size = w.finish()
    ck.close()
    for f in COPY_FILES:
        if os.path.exists(os.path.join(src, f)):
            shutil.copyfile(os.path.join(src, f), os.path.join(out, f))
    meta = {
        "format": META_FORMAT,
        "dense_bits": 8, "expert_bits": 4, "expert_group": BLOCK,
        "expert_blob_bytes": blob_bytes(hidden, inter) if moe else 0,
        "layers": layers, "experts": experts, "moe": moe,
        "skipped_tensors": skipped,
    }
    if any(k in ("vmatrix", "vexact") for k in vision_names.values()):
        meta["vision"] = vision_summary(config)
    if any(k in ("amatrix", "aexact") for k in vision_names.values()):
        meta["audio"] = audio_summary(config)
    if not vision_names and skipped:
        meta["skipped_note"] = "vision tower and projector left out (--no-vision, or the checkpoint has none)"
    with open(os.path.join(out, "gemma4_meta.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, indent=1)
    log(f"wrote {size / 1e9:.2f} GB in {time.time() - t0:.0f}s; {len(vision_names)} vision tensors carried, {skipped} skipped")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--src", help="a local checkpoint directory (config.json + safetensors)")
    where.add_argument("--repo", help="a Hugging Face repo to download first, e.g. google/gemma-4-26B-A4B-it")
    ap.add_argument("--out", "--outdir", dest="out", required=True,
                    help="the container directory to write (coli convert passes it as --outdir)")
    ap.add_argument("--keep-source", action="store_true", help="with --repo: keep the downloaded checkpoint")
    ap.add_argument("--no-vision", action="store_true", help="leave the vision tower out of the container")
    ap.add_argument("--no-audio", action="store_true", help="leave the audio tower (E2B, E4B) out of the container")
    ap.add_argument("--add-audio", action="store_true",
                    help="add the audio tower to an existing container (--out) that was converted without it")
    ap.add_argument("--add-vision", action="store_true",
                    help="add the vision tower to an existing container (--out) that was converted without it")
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
    if args.add_vision:
        add_vision(src, args.out)
    elif args.add_audio:
        add_tower(src, args.out, "audio")
    else:
        convert(src, args.out, with_vision=not args.no_vision, with_audio=not args.no_audio)
    if downloaded and not args.keep_source:
        shutil.rmtree(downloaded, ignore_errors=True)
        print(f"removed {downloaded}")


if __name__ == "__main__":
    sys.exit(main())
