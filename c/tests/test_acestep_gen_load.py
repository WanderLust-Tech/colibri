"""The ACE-Step end-to-end path (acestep_gen.h: `acestep --gen-ref`, and the one-shot `--caption ... --out`) refuses bad input cleanly.

    ACESTEP_TINY=/path/to/fixture ACESTEP_TINY_REF=/path/to/case python -m unittest tests.test_acestep_gen_load

Needs the built engine (make acestep), the tiny fixture (tools/make_acestep_tiny.py) and one captured case of it
(tools/acestep_ref.py capture ... --cases inst10); skipped without them (the silence latent files are written with torch, so
those tests also need it). The silence latent is a torch file read without torch: anything that is not exactly a stored zip
holding a float32 [1][64][N] tensor with contiguous strides and N of at least 128 frames must end in a named error and a
non-zero exit, never a crash, an out-of-bounds read or a run on garbage.
"""
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import zipfile

try:
    import torch
except ImportError:                                             # the refusal tests of the file need it to write one
    torch = None

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXE = next((p for p in (os.path.join(C_DIR, "acestep.exe"), os.path.join(C_DIR, "acestep")) if os.path.isfile(p)), None)
TINY = os.environ.get("ACESTEP_TINY")
REF = os.environ.get("ACESTEP_TINY_REF")
DIT = "acestep-v15-turbo"


@unittest.skipUnless(EXE and TINY and REF and os.path.isdir(TINY) and os.path.isdir(REF),
                     "needs the built engine, ACESTEP_TINY and ACESTEP_TINY_REF")
class GenRefusals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="acestep-gen-load-")
        self.addCleanup(self.tmp.cleanup)
        self.model = os.path.join(self.tmp.name, "model")
        for part in (DIT, "vae", "Qwen3-Embedding-0.6B"):
            shutil.copytree(os.path.join(TINY, part), os.path.join(self.model, part))
        self.silence = os.path.join(self.model, DIT, "silence_latent.pt")

    def gen_ref(self, ref=None):
        return subprocess.run([EXE, "--model", self.model, "--gen-ref", ref or REF], capture_output=True, text=True, timeout=300)

    def make(self, *args):
        return subprocess.run([EXE, "--model", self.model, *args], capture_output=True, text=True, timeout=300)

    def refused(self, result, *words, code=None):
        self.assertNotEqual(result.returncode, 0, result.stderr[-300:])
        self.assertLess(result.returncode, 128, f"crashed (signal): {result.stderr[-300:]}")
        if code is not None:
            self.assertEqual(result.returncode, code, result.stderr[-300:])
        for w in words:
            self.assertIn(w, result.stderr)

    def save_silence(self, tensor):
        torch.save(tensor, self.silence)

    def test_the_untouched_copy_passes(self):
        r = self.gen_ref()
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertIn("comparison(s) within their limits", r.stderr)

    # --- the silence latent -----------------------------------------------------------------------------------------
    def test_a_missing_silence_latent(self):
        os.remove(self.silence)
        self.refused(self.gen_ref(), "silence_latent.pt")

    def test_a_file_that_is_not_a_zip(self):
        with open(self.silence, "wb") as f:
            f.write(b"this is not a torch file at all" * 10)
        self.refused(self.gen_ref(), "not a torch file")

    def test_an_empty_or_tiny_file(self):
        for blob in (b"", b"PK", b"x" * 50):
            with self.subTest(size=len(blob)):
                with open(self.silence, "wb") as f:
                    f.write(blob)
                self.refused(self.gen_ref(), "silence latent")

    def _rewrite_zip(self, edit, compress=False):
        with zipfile.ZipFile(self.silence) as z:
            entries = {i.filename: z.read(i.filename) for i in z.infolist()}
        edit(entries)
        with zipfile.ZipFile(self.silence, "w", zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED) as z:
            for name, data in entries.items():
                z.writestr(name, data)

    @unittest.skipUnless(torch, "needs torch to write the file")
    def test_a_compressed_archive(self):
        self.save_silence(torch.zeros(1, 64, 300))
        self._rewrite_zip(lambda e: None, compress=True)
        self.refused(self.gen_ref(), "not a torch file")

    @unittest.skipUnless(torch, "needs torch to write the file")
    def test_an_archive_without_the_data(self):
        self.save_silence(torch.zeros(1, 64, 300))
        self._rewrite_zip(lambda e: [e.pop(k) for k in [k for k in e if k.endswith("/data/0")]])
        self.refused(self.gen_ref(), "not a torch file")

    @unittest.skipUnless(torch, "needs torch to write the file")
    def test_layouts_that_are_not_float32_1_64_n(self):
        for label, t in (("float64", torch.zeros(1, 64, 300, dtype=torch.float64)), ("2-d", torch.zeros(64, 300)),
                         ("frames first", torch.zeros(1, 300, 64)), ("128 channels", torch.zeros(1, 128, 300)),
                         ("batch of 2", torch.zeros(2, 64, 300)), ("int64", torch.zeros(1, 64, 300, dtype=torch.int64)),
                         ("not contiguous", torch.zeros(1, 300, 64).transpose(1, 2))):
            with self.subTest(layout=label):
                self.save_silence(t)
                self.refused(self.gen_ref(), "silence_latent.pt")

    @unittest.skipUnless(torch, "needs torch to write the file")
    def test_fewer_frames_than_the_floor(self):
        self.save_silence(torch.zeros(1, 64, 100))
        self.refused(self.gen_ref(), "fewer than 128")

    @unittest.skipUnless(torch, "needs torch to write the file")
    def test_data_that_does_not_match_its_shape(self):
        self.save_silence(torch.zeros(1, 64, 300))
        self._rewrite_zip(lambda e: e.update({k: v[:-8] for k, v in e.items() if k.endswith("/data/0")}))
        self.refused(self.gen_ref(), "bytes of data")

    @unittest.skipUnless(torch, "needs torch to write the file")
    def test_a_value_that_is_not_finite(self):
        t = torch.zeros(1, 64, 300)
        t[0, 5, 7] = float("nan")
        self.save_silence(t)
        self.refused(self.gen_ref(), "non-finite")

    @unittest.skipUnless(torch, "needs torch to write the file")
    def test_a_silence_latent_of_a_different_length_is_a_mismatch_with_the_handler(self):
        """A good file of 300 frames is read fine, and the oracle then says it is not the handler's file."""
        self.save_silence(torch.zeros(1, 64, 1500))
        r = self.gen_ref()
        self.refused(r)
        self.assertIn("FAILED", r.stderr)

    # --- the checkpoint around it -----------------------------------------------------------------------------------
    def test_a_missing_component_is_named(self):
        for part, word in ((DIT, "DiT"), ("vae", "VAE"), ("Qwen3-Embedding-0.6B", "tokenizer")):
            with self.subTest(part=part):
                saved = os.path.join(self.tmp.name, "saved-" + part)
                shutil.move(os.path.join(self.model, part), saved)
                self.refused(self.gen_ref(), word)
                shutil.move(saved, os.path.join(self.model, part))

    def test_a_latent_width_the_vae_does_not_share(self):
        path = os.path.join(self.model, "vae", "config.json")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["decoder_input_channels"] = 32
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        self.refused(self.gen_ref())

    # --- the references ---------------------------------------------------------------------------------------------
    def test_a_reference_without_its_parameters(self):
        ref = os.path.join(self.tmp.name, "noparams")
        shutil.copytree(REF, ref)
        with open(os.path.join(ref, "ref.json"), "w") as f:
            f.write("{}")
        self.refused(self.gen_ref(ref), "params.captions")

    def test_a_reference_with_a_duration_the_engine_refuses(self):
        ref = os.path.join(self.tmp.name, "baddur")
        shutil.copytree(REF, ref)
        path = os.path.join(ref, "ref.json")
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["params"]["audio_duration"] = 0.2
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        self.refused(self.gen_ref(ref), "duration")

    def test_an_empty_reference_directory(self):
        empty = os.path.join(self.tmp.name, "empty")
        os.makedirs(empty)
        self.refused(self.gen_ref(empty), "no safetensors")

    # --- the one-shot command line ----------------------------------------------------------------------------------
    def test_a_caption_needs_somewhere_to_write(self):
        self.refused(self.make("--caption", "calm piano"), "--out", code=2)

    def test_the_modes_do_not_mix(self):
        self.refused(self.make("--caption", "x", "--out", "x.wav", "--serve"), "separate modes", code=2)
        self.refused(self.make("--gen-ref", REF, "--serve"), "separate modes", code=2)

    def test_a_request_the_engine_refuses_is_named_before_anything_is_loaded(self):
        out = os.path.join(self.tmp.name, "x.wav")
        for args, word in ((["--duration", "0.5"], "duration"), (["--duration", "601"], "duration"), (["--steps", "0"], "steps"),
                           (["--steps", "21"], "steps"), (["--shift", "0"], "shift"), (["--shift", "11"], "shift"),
                           (["--seed", "4294967296"], "seed"), (["--bpm", "-1"], "bpm"), (["--duration", "abc"], "duration")):
            with self.subTest(args=args):
                self.refused(self.make("--caption", "calm piano", "--out", out, *args), word, code=2)
        self.assertFalse(os.path.exists(out))

    def test_an_output_path_that_cannot_be_written(self):
        r = self.make("--caption", "calm piano", "--duration", "2", "--steps", "1", "--seed", "1",
                      "--out", os.path.join(self.tmp.name, "no", "such", "dir", "x.wav"))
        self.refused(r, "cannot write")

    def test_a_clip_is_written_as_a_wav_and_the_same_seed_gives_the_same_file(self):
        a, b = os.path.join(self.tmp.name, "a.wav"), os.path.join(self.tmp.name, "b.wav")
        for out in (a, b):
            r = self.make("--caption", "calm piano", "--duration", "2", "--steps", "1", "--seed", "3", "--out", out)
            self.assertEqual(r.returncode, 0, r.stderr[-300:])
        with open(a, "rb") as fa, open(b, "rb") as fb:
            da, db = fa.read(), fb.read()
        self.assertEqual(da, db)
        self.assertEqual(da[:4], b"RIFF")
        self.assertEqual(struct.unpack("<IHH", da[24:32])[0:1], (48000,))
        self.assertEqual(len(da), 44 + 96000 * 4)                   # two seconds, cut from the 128-frame floor

    def test_nonsense_command_lines(self):
        for argv in ([], ["--model"], ["--gen-ref", REF], ["--serve"], ["--caption", "x"], ["--model", self.model], ["--bogus"]):
            with self.subTest(argv=argv):
                r = subprocess.run([EXE, *argv], capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 2, r.stderr[-200:])
                self.assertIn("usage", r.stderr)


if __name__ == "__main__":
    unittest.main()
