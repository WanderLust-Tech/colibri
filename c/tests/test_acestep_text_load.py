"""The ACE-Step text encoder and tokenizer (acestep_text.h, run through `acestep --text-ref` and `--tok-ref`) refuse bad input cleanly.

    ACESTEP_TINY=/path/to/fixture ACESTEP_TINY_REF=/path/to/case ACESTEP_TINY_TOKREF=/path/to/tok_ref.json \
        python -m unittest tests.test_acestep_text_load

Needs the built engine (make acestep), the tiny fixture (tools/make_acestep_tiny.py), one captured case of it
(tools/acestep_ref.py capture ... --taps --cases inst10) and the tokenizer reference made from it
(tools/make_acestep_tok_ref.py); skipped without them. A config that disagrees with itself, a tensor that is missing or the
wrong shape, a tokenizer that is gone, a reference that lacks what the oracle needs: each must end in a named error and a
non-zero exit, never a crash, an out-of-bounds read, or a run on garbage.
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
TOKREF = os.environ.get("ACESTEP_TINY_TOKREF")
TEXT = "Qwen3-Embedding-0.6B"


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


@unittest.skipUnless(EXE and TINY and REF and TOKREF and os.path.isdir(TINY) and os.path.isdir(REF) and os.path.isfile(TOKREF),
                     "needs the built engine, ACESTEP_TINY, ACESTEP_TINY_REF and ACESTEP_TINY_TOKREF")
class TextRefusals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="acestep-text-load-")
        self.addCleanup(self.tmp.cleanup)
        self.model = os.path.join(self.tmp.name, "model")
        shutil.copytree(os.path.join(TINY, TEXT), os.path.join(self.model, TEXT))
        self.text = os.path.join(self.model, TEXT)

    def run_text(self, model=None, ref=None):
        return subprocess.run([EXE, "--model", model or self.model, "--text-ref", ref or REF],
                              capture_output=True, text=True, timeout=120)

    def run_tok(self, ref=None):
        return subprocess.run([EXE, "--model", self.model, "--tok-ref", ref or TOKREF],
                              capture_output=True, text=True, timeout=120)

    def edit_config(self, **changes):
        path = os.path.join(self.text, "config.json")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.update(changes)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)

    def restore_config(self):
        shutil.copyfile(os.path.join(TINY, TEXT, "config.json"), os.path.join(self.text, "config.json"))

    def refused(self, result, *words):
        self.assertNotEqual(result.returncode, 0, result.stderr[-300:])
        self.assertLess(result.returncode, 128, f"crashed (signal): {result.stderr[-300:]}")
        for w in words:
            self.assertIn(w, result.stderr)

    def write_tok_ref(self, payload, name="tok_ref.json"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload if isinstance(payload, str) else json.dumps(payload))
        return path

    # --- the encoder -----------------------------------------------------------------------------------------------
    def test_the_untouched_copy_passes(self):
        r = self.run_text()
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertIn("comparison(s) within their limits", r.stderr)
        r = self.run_tok()
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertIn("cases identical", r.stderr)

    def test_a_missing_config(self):
        os.remove(os.path.join(self.text, "config.json"))
        self.refused(self.run_text(), "cannot open")

    def test_a_config_that_is_not_json(self):
        with open(os.path.join(self.text, "config.json"), "w") as f:
            f.write("not json at all")
        self.refused(self.run_text())

    def test_sizes_out_of_range(self):
        for key, value in (("hidden_size", 0), ("hidden_size", 10 ** 9), ("num_hidden_layers", 0), ("num_hidden_layers", 10 ** 6),
                           ("num_attention_heads", 0), ("num_key_value_heads", 0), ("head_dim", 0), ("head_dim", 7),
                           ("intermediate_size", 0), ("vocab_size", 0), ("vocab_size", 10 ** 9)):
            with self.subTest(key=key, value=value):
                self.edit_config(**{key: value})
                self.refused(self.run_text(), "out of range")
                self.restore_config()

    def test_heads_that_do_not_divide_into_key_value_heads(self):
        self.edit_config(num_key_value_heads=3)
        self.refused(self.run_text(), "out of range")

    def test_unsupported_variants_are_refused_by_name(self):
        for changes, word in (({"model_type": "llama"}, "model_type"),
                              ({"rope_scaling": {"rope_type": "yarn", "factor": 2.0}}, "rope_scaling"),
                              ({"attention_bias": True}, "attention_bias"),
                              ({"use_sliding_window": True}, "use_sliding_window")):
            with self.subTest(changes=changes):
                self.edit_config(**changes)
                self.refused(self.run_text(), word)
                self.restore_config()

    def test_a_layer_that_is_not_full_attention(self):
        self.edit_config(layer_types=["full_attention", "sliding_attention"])
        self.refused(self.run_text(), "layer_types[1]")

    def test_a_config_that_disagrees_with_the_tensors(self):
        """Right in itself, wrong for this checkpoint: the first tensor that does not fit is named, with both shapes."""
        self.edit_config(hidden_size=128)
        self.refused(self.run_text(), "expected")

    def test_a_vocabulary_that_disagrees_with_the_embedding_table(self):
        self.edit_config(vocab_size=100)
        self.refused(self.run_text(), "embed_tokens.weight", "expected")

    def test_a_missing_tensor_is_named(self):
        rewrite_header(os.path.join(self.text, "model.safetensors"), lambda t: t.pop("layers.1.mlp.up_proj.weight"))
        self.refused(self.run_text(), "layers.1.mlp.up_proj.weight")

    def test_a_missing_embedding_table_is_named(self):
        rewrite_header(os.path.join(self.text, "model.safetensors"), lambda t: t.pop("embed_tokens.weight"))
        self.refused(self.run_text(), "embed_tokens.weight")

    def test_a_missing_final_norm_is_named(self):
        rewrite_header(os.path.join(self.text, "model.safetensors"), lambda t: t.pop("norm.weight"))
        self.refused(self.run_text(), "norm.weight")

    def test_a_wrong_shape_is_named(self):
        def reshape(t):
            t["layers.0.self_attn.q_proj.weight"]["shape"] = [2, 3]
        rewrite_header(os.path.join(self.text, "model.safetensors"), reshape)
        self.refused(self.run_text(), "layers.0.self_attn.q_proj.weight")

    def test_a_wrong_embedding_shape_is_named(self):
        def reshape(t):
            t["embed_tokens.weight"]["shape"] = [32, 504]
        rewrite_header(os.path.join(self.text, "model.safetensors"), reshape)
        self.refused(self.run_text(), "embed_tokens.weight", "expected")

    def test_a_non_float_tensor_is_refused(self):
        def retype(t):
            t["norm.weight"]["dtype"] = "I64"
        rewrite_header(os.path.join(self.text, "model.safetensors"), retype)
        self.refused(self.run_text(), "norm.weight")

    def test_a_non_float_embedding_table_is_refused(self):
        def retype(t):
            t["embed_tokens.weight"]["dtype"] = "I64"
        rewrite_header(os.path.join(self.text, "model.safetensors"), retype)
        self.refused(self.run_text(), "embed_tokens.weight")        # the reader's span check says so before the loader's dtype check

    def test_no_weights_at_all(self):
        os.remove(os.path.join(self.text, "model.safetensors"))
        self.refused(self.run_text(), "no safetensors")

    def test_no_text_encoder_directory(self):
        shutil.rmtree(self.text)
        self.refused(self.run_text())

    # --- the tokenizer ---------------------------------------------------------------------------------------------
    def test_a_missing_tokenizer_is_named(self):
        os.remove(os.path.join(self.text, "tokenizer.json"))
        self.refused(self.run_text(), "tokenizer.json")
        self.refused(self.run_tok(), "tokenizer.json")

    def test_a_tokenizer_that_is_not_json(self):
        with open(os.path.join(self.text, "tokenizer.json"), "w") as f:
            f.write("not json at all")
        self.refused(self.run_text())
        self.refused(self.run_tok())

    # --- the references --------------------------------------------------------------------------------------------
    def test_an_empty_reference_directory(self):
        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty)
        self.refused(self.run_text(ref=empty), "no safetensors")

    def test_a_reference_without_the_inputs_the_oracle_needs(self):
        """A case captured from something else (here: only a VAE decode's tensors, no params) is a clear refusal."""
        ref = os.path.join(self.tmp.name, "thin")
        os.makedirs(ref)
        data = bytes(4 * 64 * 4)                                               # one f32 tensor, 1 x 64 x 4, zeros
        header = json.dumps({"vae0.in": {"dtype": "F32", "shape": [1, 64, 4], "data_offsets": [0, len(data)]}}).encode()
        with open(os.path.join(ref, "ref.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(header)) + header + data)
        self.refused(self.run_text(ref=ref))

    def test_a_reference_whose_ref_json_has_no_prompts(self):
        ref = os.path.join(self.tmp.name, "noprompts")
        shutil.copytree(REF, ref)
        with open(os.path.join(ref, "ref.json"), "w") as f:
            f.write("{}")
        self.refused(self.run_text(ref=ref))

    def test_a_tokenizer_reference_that_is_missing(self):
        self.refused(self.run_tok(ref=os.path.join(self.tmp.name, "nope.json")), "cannot open")

    def test_a_tokenizer_reference_that_is_not_an_array(self):
        self.refused(self.run_tok(ref=self.write_tok_ref({"text": "a"})), "not a JSON array")

    def test_a_tokenizer_reference_with_nothing_in_it(self):
        self.refused(self.run_tok(ref=self.write_tok_ref([])), "nothing to compare")

    def test_a_malformed_tokenizer_reference_case(self):
        for case in ({"text": "a"}, {"text": "a", "max_length": 0, "ids": []}, {"text": 5, "max_length": 8, "ids": []},
                     {"text": "a", "max_length": 8, "ids": "x"}):
            with self.subTest(case=case):
                self.refused(self.run_tok(ref=self.write_tok_ref([case])), "malformed")

    def test_a_tokenizer_reference_the_engine_disagrees_with(self):
        r = self.run_tok(ref=self.write_tok_ref([{"text": "hello", "max_length": 16, "ids": [1, 2, 3]}]))
        self.refused(r, "DIFFERS")

    def test_nonsense_command_lines(self):
        for argv in ([], ["--model"], ["--text-ref", REF], ["--tok-ref", TOKREF], ["--model", self.model], ["--bogus"]):
            with self.subTest(argv=argv):
                r = subprocess.run([EXE, *argv], capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 2, r.stderr[-200:])
                self.assertIn("usage", r.stderr)


if __name__ == "__main__":
    unittest.main()
