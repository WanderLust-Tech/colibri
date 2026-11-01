"""A fake ACE-Step checkpoint directory and the stub engine, for the gateway and CLI tests.

The directory has the real checkpoint's layout (the DiT's config.json at the root with model_type acestep, one folder
per component, the tokenizer in Qwen3-Embedding-0.6B/, silence_latent.pt) and safetensors files whose headers name
real tensor names with tiny shapes. The data is zeros: nothing here reads it, the planner and `coli info` only parse
headers. The engine is tools/acestep_stub.py.
"""
import json
import os
import struct
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
STUB = C_DIR / "tools" / "acestep_stub.py"
RATE = 8000                                    # the stub's clips: small, so the tests stay quick

TENSORS = {
    "Qwen3-Embedding-0.6B": [("embed_tokens.weight", "BF16", (1000, 64)),
                             ("layers.0.self_attn.q_proj.weight", "BF16", (64, 64)),
                             ("norm.weight", "BF16", (64,))],
    "acestep-v15-turbo": [("decoder.proj_in.1.weight", "BF16", (128, 192, 2)),
                          ("decoder.layers.0.self_attn.q_proj.weight", "BF16", (128, 128)),
                          ("decoder.norm_out.weight", "BF16", (128,))],
    "vae": [("decoder.conv1.weight_g", "BF16", (64, 1, 1)),
            ("decoder.conv1.weight_v", "BF16", (64, 64, 7)),
            ("encoder.conv1.weight_v", "BF16", (64, 2, 7))],
}
_SIZES = {"BF16": 2, "F32": 4}


def elements(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, dtype, shape in tensors:
        size = _SIZES[dtype] * elements(shape)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)) + raw + bytes(offset))


def expected_resident(component):
    """What music_engine says the engine holds: the checkpoint's bytes, the VAE's decoder widened to f32."""
    total = 0
    for name, dtype, shape in TENSORS[component]:
        if component == "vae":
            if not name.startswith("encoder."):
                total += 4 * elements(shape)
        else:
            total += _SIZES[dtype] * elements(shape)
    return total


def make_fake_checkpoint(root, planner=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "acestep", "architectures": ["AceStepConditionGenerationModel"],
                                                   "hidden_size": 2048, "num_hidden_layers": 24}))
    for component, tensors in TENSORS.items():
        directory = root / component
        directory.mkdir(exist_ok=True)
        name = "diffusion_pytorch_model.safetensors" if component == "vae" else "model.safetensors"
        write_safetensors(directory / name, tensors)
    (root / "Qwen3-Embedding-0.6B" / "tokenizer.json").write_text("{}")
    (root / "acestep-v15-turbo" / "silence_latent.pt").write_bytes(b"x")
    if planner:
        (root / "acestep-5Hz-lm-1.7B").mkdir(exist_ok=True)
        (root / "acestep-5Hz-lm-1.7B" / "model.safetensors").write_bytes(bytes(2000))
    return str(root)


def stub_env(delay=0.0, **extra):
    env = dict(os.environ)
    env.update({"ACESTEP_STUB_RATE": str(RATE), "ACESTEP_STUB_DELAY": str(delay)})
    env.update({k: str(v) for k, v in extra.items()})
    return env
