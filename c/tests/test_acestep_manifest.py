"""tools/acestep_manifest.py: the tensor manifest of an ACE-Step checkpoint, from safetensors headers alone.

Standard library only, no model needed: the checkpoints here are headers followed by zero bytes.
"""
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import acestep_manifest as am  # noqa: E402

ITEM = {"BF16": 2, "F32": 4}


def write_safetensors(path, tensors, metadata=None):
    """tensors: {name: (dtype, shape)}; the data is zeros, which is all a header test needs."""
    table, offset = {}, 0
    if metadata is not None:
        table["__metadata__"] = metadata
    for name, (dtype, shape) in tensors.items():
        size = ITEM[dtype]
        for d in shape:
            size *= d
        table[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(table).encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + bytes(offset))


class Collapse(unittest.TestCase):
    def test_layers_of_one_shape_become_one_pattern_with_a_count(self):
        table = {f"decoder.layers.{i}.mlp.up_proj.weight": {"dtype": "BF16", "shape": [6144, 2048]} for i in range(24)}
        table["decoder.norm_out.weight"] = {"dtype": "BF16", "shape": [2048]}
        out = am.collapse(table)
        self.assertEqual(out["decoder.layers.N.mlp.up_proj.weight"], {"dtype": "BF16", "shape": [6144, 2048], "count": 24})
        self.assertEqual(out["decoder.norm_out.weight"]["count"], 1)
        self.assertEqual(len(out), 2)

    def test_two_digit_and_nested_indices_collapse_too(self):
        table = {f"a.{i}.b.{j}.w": {"dtype": "F32", "shape": [3]} for i in (9, 10) for j in (0, 1)}
        self.assertEqual(list(am.collapse(table)), ["a.N.b.N.w"])
        self.assertEqual(am.collapse(table)["a.N.b.N.w"]["count"], 4)

    def test_a_stack_that_narrows_keeps_its_names(self):
        """The VAE decoder's blocks are 1024, 512, 256, ... wide: one shared pattern would be wrong."""
        widths = [1024, 512, 256]
        table = {f"decoder.block.{i}.conv_t1.bias": {"dtype": "BF16", "shape": [w]} for i, w in enumerate(widths)}
        out = am.collapse(table)
        self.assertEqual(sorted(out), sorted(table))
        for i, w in enumerate(widths):
            self.assertEqual(out[f"decoder.block.{i}.conv_t1.bias"], {"dtype": "BF16", "shape": [w], "count": 1})

    def test_a_dtype_change_alone_also_keeps_the_names(self):
        table = {"x.0.w": {"dtype": "BF16", "shape": [4]}, "x.1.w": {"dtype": "F32", "shape": [4]}}
        self.assertEqual(sorted(am.collapse(table)), ["x.0.w", "x.1.w"])

    def test_an_index_not_between_dots_is_left_alone(self):
        """Only a whole ".<digits>." path component is a layer index: names that end in or contain digits are not."""
        table = {"decoder.proj_in.1.weight": {"dtype": "BF16", "shape": [2]}, "layer2.weight": {"dtype": "BF16", "shape": [2]}}
        self.assertEqual(sorted(am.collapse(table)), ["decoder.proj_in.N.weight", "layer2.weight"])


class Headers(unittest.TestCase):
    def test_reads_a_header_and_drops_the_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.safetensors")
            write_safetensors(path, {"a": ("BF16", (2, 3))}, metadata={"format": "pt"})
            table = am.read_header(path)
        self.assertEqual(list(table), ["a"])
        self.assertEqual(table["a"]["shape"], [2, 3])

    def test_refuses_what_is_not_a_safetensors_file(self):
        with tempfile.TemporaryDirectory() as d:
            short = os.path.join(d, "short")
            with open(short, "wb") as f:
                f.write(b"abc")
            with self.assertRaises(ValueError):
                am.read_header(short)
            huge = os.path.join(d, "huge")
            with open(huge, "wb") as f:
                f.write(struct.pack("<Q", 1 << 40))
            with self.assertRaisesRegex(ValueError, "header of"):
                am.read_header(huge)


class Directory(unittest.TestCase):
    def make_dir(self, d, extra=None):
        write_safetensors(os.path.join(d, "acestep-v15-turbo", "model.safetensors"),
                          {f"decoder.layers.{i}.w": ("BF16", (4, 4)) for i in range(3)})
        write_safetensors(os.path.join(d, "vae", "diffusion_pytorch_model.safetensors"),
                          {"decoder.conv1.bias": ("BF16", (8,)), **(extra or {})})
        write_safetensors(os.path.join(d, "Qwen3-Embedding-0.6B", "model.safetensors"),
                          {"embed_tokens.weight": ("BF16", (10, 4))})

    def test_build_and_check_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_dir(d)
            built = am.build(d)
            stored = os.path.join(d, "manifest.json")
            with open(stored, "w", encoding="utf-8") as f:
                json.dump(built, f)
            self.assertEqual(built["dit"]["decoder.layers.N.w"]["count"], 3)
            self.assertEqual(sorted(built), ["dit", "text", "vae"])
            with open(stored, encoding="utf-8") as f:
                self.assertEqual(json.load(f), built)

    def test_structure_is_what_a_fixture_must_share(self):
        patterns = am.collapse({"a.0.w": {"dtype": "BF16", "shape": [4, 4]}, "a.1.w": {"dtype": "BF16", "shape": [4, 4]}})
        self.assertEqual(am.structure(patterns), {"a.N.w": ("BF16", 2)})

    def test_structure_ignores_widths_and_layer_counts_but_not_dtype_or_rank(self):
        wide = {f"l.{i}.w": {"dtype": "BF16", "shape": [64, 64]} for i in range(24)}
        narrow = {f"l.{i}.w": {"dtype": "BF16", "shape": [8, 8]} for i in range(2)}
        self.assertEqual(am.structure(wide), am.structure(narrow))
        self.assertNotEqual(am.structure(wide), am.structure({"l.0.w": {"dtype": "F32", "shape": [8, 8]}}))
        self.assertNotEqual(am.structure(wide), am.structure({"l.0.w": {"dtype": "BF16", "shape": [8]}}))
        with self.assertRaises(ValueError):
            am.structure({"l.0.w": {"dtype": "BF16", "shape": [8]}, "l.1.w": {"dtype": "BF16", "shape": [8, 8]}})

    def test_a_missing_component_is_an_error_not_an_empty_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_dir(d)
            os.remove(os.path.join(d, "vae", "diffusion_pytorch_model.safetensors"))
            with self.assertRaises(OSError):
                am.build(d)


class StoredManifest(unittest.TestCase):
    """The manifest in the repository describes the real turbo checkpoint: its headline numbers are the ones
    the C loader is written against, so a regeneration that changes them must be noticed."""

    @classmethod
    def setUpClass(cls):
        with open(am.DEFAULT_MANIFEST, encoding="utf-8") as f:
            cls.stored = json.load(f)

    def test_the_dit_has_24_layers_with_the_expected_projections(self):
        dit = self.stored["dit"]
        self.assertEqual(dit["decoder.layers.N.self_attn.q_proj.weight"]["count"], 24)
        self.assertEqual(dit["decoder.layers.N.self_attn.q_proj.weight"]["shape"], [2048, 2048])
        self.assertEqual(dit["decoder.layers.N.self_attn.k_proj.weight"]["shape"], [1024, 2048])  # 8 KV heads of 128
        self.assertEqual(dit["decoder.layers.N.mlp.up_proj.weight"]["shape"], [6144, 2048])
        self.assertEqual(dit["decoder.layers.N.scale_shift_table"]["shape"], [1, 6, 2048])
        self.assertEqual(dit["encoder.lyric_encoder.layers.N.mlp.up_proj.weight"]["count"], 8)
        self.assertEqual(dit["encoder.timbre_encoder.layers.N.mlp.up_proj.weight"]["count"], 4)

    def test_everything_is_bf16_and_the_tensor_counts_are_the_real_ones(self):
        for component, want in (("dit", 677), ("vae", 365), ("text", 310)):
            patterns = self.stored[component]
            self.assertEqual(sum(e["count"] for e in patterns.values()), want, component)
            self.assertEqual({e["dtype"] for e in patterns.values()}, {"BF16"}, component)

    def test_the_vae_is_weight_normed_snake_with_the_48k_stereo_ends(self):
        vae = self.stored["vae"]
        self.assertEqual(vae["decoder.conv1.weight_v"]["shape"], [2048, 64, 7])   # 64 latent channels in
        self.assertEqual(vae["decoder.conv2.weight_v"]["shape"], [2, 128, 7])     # stereo out
        self.assertIn("decoder.snake1.alpha", vae)
        self.assertIn("decoder.block.0.conv_t1.weight_g", vae)


if __name__ == "__main__":
    unittest.main()
