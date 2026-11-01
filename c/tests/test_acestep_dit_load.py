"""The ACE-Step DiT (acestep_dit.h, run through `acestep --dit-ref`) refuses a bad checkpoint or case cleanly.

    ACESTEP_TINY=/path/to/fixture ACESTEP_TINY_REF=/path/to/case python -m unittest tests.test_acestep_dit_load

Needs the built engine (make acestep), the tiny fixture (tools/make_acestep_tiny.py) and one captured case of it
(tools/acestep_ref.py capture ... --cases inst10); skipped without them. A config that disagrees with itself, a tensor that
is missing or the wrong shape, a reference that lacks what the oracle needs: each must end in a named error and a non-zero
exit, never a crash, an out-of-bounds read, or a run on garbage.
"""
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXE = next((p for p in (os.path.join(C_DIR, "acestep.exe"), os.path.join(C_DIR, "acestep")) if os.path.isfile(p)), None)
TINY = os.environ.get("ACESTEP_TINY")
REF = os.environ.get("ACESTEP_TINY_REF")
DIT = "acestep-v15-turbo"


def rewrite_header(path, edit):
    """Apply edit(table) to a safetensors header and write the file back with the data untouched."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        table = json.loads(f.read(n))
        data = f.read()
    edit(table)
    raw = json.dumps(table).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + data)


@unittest.skipUnless(EXE and TINY and REF and os.path.isdir(TINY) and os.path.isdir(REF),
                     "needs the built engine, ACESTEP_TINY and ACESTEP_TINY_REF")
class LoaderRefusals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="acestep-dit-load-")
        self.addCleanup(self.tmp.cleanup)
        self.model = os.path.join(self.tmp.name, "model")
        shutil.copytree(os.path.join(TINY, DIT), os.path.join(self.model, DIT))
        self.dit = os.path.join(self.model, DIT)

    def run_oracle(self, model=None, ref=None):
        return subprocess.run([EXE, "--model", model or self.model, "--dit-ref", ref or REF],
                              capture_output=True, text=True, timeout=120)

    def edit_config(self, **changes):
        path = os.path.join(self.dit, "config.json")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.update(changes)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

    def refused(self, result, *words):
        self.assertNotEqual(result.returncode, 0, result.stderr[-300:])
        self.assertLess(result.returncode, 128, f"crashed (signal): {result.stderr[-300:]}")
        for w in words:
            self.assertIn(w, result.stderr)

    def test_the_untouched_copy_passes(self):
        r = self.run_oracle()
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertIn("comparison(s) within their limits", r.stderr)

    def test_a_missing_config(self):
        os.remove(os.path.join(self.dit, "config.json"))
        self.refused(self.run_oracle(), "cannot open")

    def test_a_config_that_is_not_json(self):
        with open(os.path.join(self.dit, "config.json"), "w") as f:
            f.write("not json at all")
        self.refused(self.run_oracle())

    def test_sizes_out_of_range(self):
        for key, value in (("hidden_size", 0), ("hidden_size", 10 ** 9), ("num_hidden_layers", 0), ("num_hidden_layers", 10 ** 6),
                           ("num_attention_heads", 0), ("head_dim", 0), ("head_dim", 7), ("patch_size", 0), ("patch_size", 99),
                           ("intermediate_size", 0), ("text_hidden_dim", 0), ("num_lyric_encoder_hidden_layers", 0),
                           ("audio_acoustic_hidden_dim", 0)):
            with self.subTest(key=key, value=value):
                self.edit_config(**{key: value})
                self.refused(self.run_oracle(), "out of range")
                shutil.copyfile(os.path.join(TINY, DIT, "config.json"), os.path.join(self.dit, "config.json"))

    def test_heads_that_do_not_divide_into_key_value_heads(self):
        self.edit_config(num_key_value_heads=3)
        self.refused(self.run_oracle(), "out of range")

    def test_input_channels_that_are_not_latents_source_and_mask(self):
        self.edit_config(in_channels=100)
        self.refused(self.run_oracle(), "in_channels")

    def test_a_timbre_width_that_is_not_the_latent_width(self):
        self.edit_config(timbre_hidden_dim=32)
        self.refused(self.run_oracle(), "timbre_hidden_dim")

    def test_unsupported_variants_are_refused_by_name(self):
        self.edit_config(rope_scaling={"rope_type": "yarn", "factor": 2.0})
        self.refused(self.run_oracle(), "rope_scaling")
        shutil.copyfile(os.path.join(TINY, DIT, "config.json"), os.path.join(self.dit, "config.json"))
        self.edit_config(attention_bias=True)
        self.refused(self.run_oracle(), "attention_bias")

    def test_layer_types_too_short_or_unknown(self):
        self.edit_config(layer_types=["sliding_attention"])
        self.refused(self.run_oracle(), "layer_types")
        self.edit_config(layer_types=["sliding_attention", "full_attention", "bogus", "full_attention"])
        self.refused(self.run_oracle(), "layer_types[2]")

    def test_a_config_that_disagrees_with_the_tensors(self):
        """Right in itself, wrong for this checkpoint: the first tensor that does not fit is named, with both shapes."""
        self.edit_config(hidden_size=128)
        self.refused(self.run_oracle(), "expected")

    def test_a_missing_tensor_is_named(self):
        rewrite_header(os.path.join(self.dit, "model.safetensors"), lambda t: t.pop("decoder.layers.2.mlp.up_proj.weight"))
        self.refused(self.run_oracle(), "decoder.layers.2.mlp.up_proj.weight")

    def test_a_missing_encoder_tensor_is_named(self):
        rewrite_header(os.path.join(self.dit, "model.safetensors"), lambda t: t.pop("encoder.timbre_encoder.norm.weight"))
        self.refused(self.run_oracle(), "encoder.timbre_encoder.norm.weight")

    def test_a_wrong_shape_is_named(self):
        def reshape(t):
            t["decoder.layers.1.self_attn.q_proj.weight"]["shape"] = [2, 3]
        rewrite_header(os.path.join(self.dit, "model.safetensors"), reshape)
        self.refused(self.run_oracle(), "decoder.layers.1.self_attn.q_proj.weight")

    def test_a_non_float_tensor_is_refused(self):
        def retype(t):
            t["decoder.norm_out.weight"]["dtype"] = "I64"
        rewrite_header(os.path.join(self.dit, "model.safetensors"), retype)
        self.refused(self.run_oracle(), "decoder.norm_out.weight")

    def test_no_weights_at_all(self):
        os.remove(os.path.join(self.dit, "model.safetensors"))
        self.refused(self.run_oracle(), "no safetensors")

    def test_no_dit_directory(self):
        shutil.rmtree(self.dit)
        self.refused(self.run_oracle())

    def test_an_empty_reference_directory(self):
        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty)
        self.refused(self.run_oracle(ref=empty), "no safetensors")

    def test_a_reference_without_the_inputs_the_oracle_needs(self):
        """A case captured from something else (here: only a VAE decode's tensors) is a clear refusal, not a run on nothing."""
        ref = os.path.join(self.tmp.name, "thin")
        os.makedirs(ref)
        data = bytes(4 * 64 * 4)                                               # one f32 tensor, 1 x 64 x 4, zeros
        header = json.dumps({"vae0.in": {"dtype": "F32", "shape": [1, 64, 4], "data_offsets": [0, len(data)]}}).encode()
        with open(os.path.join(ref, "ref.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(header)) + header + data)
        self.refused(self.run_oracle(ref=ref), "lacks")

    def test_nonsense_command_lines(self):
        for argv in ([], ["--model"], ["--dit-ref", REF], ["--model", self.model], ["--bogus"]):
            with self.subTest(argv=argv):
                r = subprocess.run([EXE, *argv], capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 2, r.stderr[-200:])
                self.assertIn("usage", r.stderr)


if __name__ == "__main__":
    unittest.main()
