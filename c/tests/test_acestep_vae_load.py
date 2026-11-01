"""The ACE-Step VAE decoder (acestep_vae.h, run through `acestep --vae-ref`) refuses a bad checkpoint cleanly.

    ACESTEP_TINY=/path/to/fixture ACESTEP_TINY_REF=/path/to/case python -m unittest tests.test_acestep_vae_load

Needs the built engine (make acestep), the tiny fixture (tools/make_acestep_tiny.py) and one captured case of it
(tools/acestep_ref.py capture ... --cases inst10); skipped without them. A checkpoint is outside data: a config that
disagrees with itself, a tensor that is missing or the wrong shape, a reference that is not what it says must end in
a named error and a non-zero exit, never a crash, an out-of-bounds read, or a decode of garbage.
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXE = next((p for p in (os.path.join(C_DIR, "acestep.exe"), os.path.join(C_DIR, "acestep")) if os.path.isfile(p)), None)
TINY = os.environ.get("ACESTEP_TINY")
REF = os.environ.get("ACESTEP_TINY_REF")
WEIGHTS = "diffusion_pytorch_model.safetensors"


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
        self.tmp = tempfile.TemporaryDirectory(prefix="acestep-load-")
        self.addCleanup(self.tmp.cleanup)
        self.model = os.path.join(self.tmp.name, "model")
        shutil.copytree(os.path.join(TINY, "vae"), os.path.join(self.model, "vae"))
        self.vae = os.path.join(self.model, "vae")

    def run_oracle(self, model=None, ref=None, *extra):
        return subprocess.run([EXE, "--model", model or self.model, "--vae-ref", ref or REF, *extra],
                              capture_output=True, text=True, timeout=120)

    def edit_config(self, **changes):
        path = os.path.join(self.vae, "config.json")
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
        self.assertIn("within rel", r.stderr)

    def test_a_missing_config(self):
        os.remove(os.path.join(self.vae, "config.json"))
        self.refused(self.run_oracle(), "cannot open")

    def test_a_config_that_is_not_json(self):
        with open(os.path.join(self.vae, "config.json"), "w") as f:
            f.write("not json at all")
        self.refused(self.run_oracle())

    def test_ratios_and_multiples_of_different_lengths(self):
        self.edit_config(channel_multiples=[1, 2, 4, 8])
        self.refused(self.run_oracle(), "same length")

    def test_no_blocks_and_too_many_blocks(self):
        for ratios in ([], [2] * 9):
            with self.subTest(blocks=len(ratios)):
                self.edit_config(downsampling_ratios=ratios, channel_multiples=ratios)
                self.refused(self.run_oracle(), "same length")

    def test_sizes_out_of_range(self):
        for key, value in (("decoder_channels", 0), ("decoder_channels", 10 ** 9), ("decoder_input_channels", 0),
                           ("audio_channels", 0), ("audio_channels", 99)):
            with self.subTest(key=key, value=value):
                self.edit_config(**{key: value})
                self.refused(self.run_oracle(), "out of range")

    def test_a_stride_or_width_too_large(self):
        self.edit_config(downsampling_ratios=[2, 4, 4, 6, 1000])
        self.refused(self.run_oracle(), "out of range")

    def test_a_non_integer_ratio(self):
        self.edit_config(downsampling_ratios=[2, 4, 4, 6, 10.5])
        self.refused(self.run_oracle(), "same length")

    def test_a_config_that_disagrees_with_the_tensors(self):
        """Right in itself, wrong for this checkpoint: the first tensor that does not fit is named."""
        self.edit_config(decoder_channels=16)
        self.refused(self.run_oracle(), "decoder.")

    def test_a_missing_tensor_is_named(self):
        rewrite_header(os.path.join(self.vae, WEIGHTS), lambda t: t.pop("decoder.block.2.snake1.beta"))
        self.refused(self.run_oracle(), "decoder.block.2.snake1.beta")

    def test_a_wrong_shape_is_named(self):
        def reshape(t):
            t["decoder.block.1.conv_t1.weight_v"]["shape"] = [2, 3, 4]
        rewrite_header(os.path.join(self.vae, WEIGHTS), reshape)
        self.refused(self.run_oracle(), "decoder.block.1.conv_t1.weight_v")

    def test_a_non_float_tensor_is_refused(self):
        def retype(t):
            t["decoder.conv1.bias"]["dtype"] = "I64"
        rewrite_header(os.path.join(self.vae, WEIGHTS), retype)
        self.refused(self.run_oracle(), "decoder.conv1.bias")

    def test_no_weights_at_all(self):
        os.remove(os.path.join(self.vae, WEIGHTS))
        self.refused(self.run_oracle(), "no safetensors")

    def test_no_vae_directory(self):
        shutil.rmtree(self.vae)
        self.refused(self.run_oracle())

    def test_an_empty_reference_directory(self):
        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty)
        self.refused(self.run_oracle(ref=empty), "no safetensors")

    def test_a_reference_whose_latents_do_not_fit_the_decoder(self):
        """A case captured from a model with other latent channels must not be decoded as if it fitted."""
        self.edit_config(decoder_input_channels=32)
        # the config now disagrees with the weights too; either way the result is a refusal, not a decode
        self.refused(self.run_oracle())

    def test_nonsense_command_lines(self):
        for argv in ([], ["--model"], ["--vae-ref", REF], ["--model", self.model], ["--bogus"]):
            with self.subTest(argv=argv):
                r = subprocess.run([EXE, *argv], capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 2, r.stderr[-200:])
                self.assertIn("usage", r.stderr)

    def test_chunk_and_overlap_that_cannot_tile_are_refused_or_repaired(self):
        """chunk <= 2*overlap halves the overlap as upstream does (never below 4); a chunk of 1 cannot tile a clip."""
        r = self.run_oracle(None, None, "--chunk", "100", "--overlap", "64")      # halves to 32: still a valid tiling
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        r = self.run_oracle(None, None, "--chunk", "1", "--overlap", "0")
        self.assertLess(r.returncode, 128, "crashed")


if __name__ == "__main__":
    unittest.main()
