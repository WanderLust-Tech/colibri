"""model_info: a header-only look at a model folder, and a check that its files are whole."""
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_info  # noqa: E402
import setup_download  # noqa: E402
from qwen_image_fixture import make_fake_pipeline  # noqa: E402


def safetensors(tensors, extra=0, cut=0):
    """The bytes of a safetensors file holding {name: (dtype, shape)}; `extra` bytes of junk after, or `cut` bytes missing."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        size = count * model_info.DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    body = json.dumps(header).encode()
    data = struct.pack("<Q", len(body)) + body + bytes(offset) + bytes(extra)
    return data[:len(data) - cut] if cut else data


def manifest_for(folder, files, complete=True, repo="acme/model", revision="abc123def4567"):
    specs = []
    for relative in files:
        data = (Path(folder) / relative).read_bytes()
        specs.append({"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    Path(folder, setup_download.MANIFEST).write_text(json.dumps({"repo": repo, "revision": revision, "complete": complete, "files": specs}))


class Folder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def model(self, name="m", config=None, shards=None):
        folder = self.root / name
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(config if config is not None else {"model_type": "toy", "max_position_embeddings": 32768}))
        for shard, tensors in (shards or {"model-00001.safetensors": {"a": ("BF16", (4, 4)), "b": ("F32", (2,))}}).items():
            (folder / shard).write_bytes(safetensors(tensors))
        return folder


class HeaderTest(Folder):
    def test_a_header_is_read_without_the_weights(self):
        path = self.model() / "model-00001.safetensors"
        header, offset, size = model_info.read_header(path)
        self.assertEqual(set(header), {"a", "b"})
        self.assertEqual(offset + 40, size)

    def test_what_is_not_a_safetensors_file_is_named_not_raised(self):
        folder = self.model()
        for name, data, words in (("short", b"abc", "shorter"), ("huge", struct.pack("<Q", 1 << 50) + b"x" * 20, "does not fit"),
                                  ("notjson", struct.pack("<Q", 5) + b"{oops", "not JSON"), ("list", struct.pack("<Q", 2) + b"[]", "not a table")):
            (folder / f"{name}.safetensors").write_bytes(data)
            self.assertIn(words, model_info.check_safetensors(folder / f"{name}.safetensors"), name)

    def test_a_file_cut_short_or_with_junk_after_is_found_by_its_own_header(self):
        folder = self.model()
        whole = folder / "model-00001.safetensors"
        self.assertIsNone(model_info.check_safetensors(whole))
        whole.write_bytes(safetensors({"a": ("BF16", (4, 4))}, cut=7))
        self.assertIn("cut short", model_info.check_safetensors(whole))
        whole.write_bytes(safetensors({"a": ("BF16", (4, 4))}, extra=9))
        self.assertIn("accounts for", model_info.check_safetensors(whole))
        self.assertIn("cannot be read", model_info.check_safetensors(folder / "nope.safetensors"))


class InspectTest(Folder):
    def test_it_says_architecture_context_weights_and_files(self):
        folder = self.model(config={"architectures": ["ToyForCausalLM"], "text_config": {"max_position_embeddings": 262144}},
                            shards={"a.safetensors": {"w": ("BF16", (100, 100))}, "b.safetensors": {"v": ("BF16", (50, 50)), "s": ("F32", (4,))}})
        found = model_info.inspect(folder)
        self.assertEqual((found["architecture"], found["context_length"], found["shards"], found["tensors"]), ("ToyForCausalLM", 262144, 2, 3))
        self.assertEqual(found["quantisation"], "bf16")
        self.assertEqual(found["bytes"], sum((folder / n).stat().st_size for n in ("a.safetensors", "b.safetensors")))

    def test_the_dominant_dtype_names_the_quantisation_and_a_config_can_name_a_method(self):
        folder = self.model(shards={"m.safetensors": {"q": ("I8", (1000, 1000)), "s": ("F16", (1000,))}})
        self.assertEqual(model_info.inspect(folder)["quantisation"], "int8")
        mixed = self.model("mix", shards={"m.safetensors": {"q": ("F8_E4M3", (500, 500)), "s": ("BF16", (300, 300))}})
        self.assertEqual(model_info.inspect(mixed)["quantisation"], "fp8 with bf16")
        gptq = self.model("gptq", config={"model_type": "toy", "quantization_config": {"quant_method": "gptq", "bits": 4}},
                          shards={"m.safetensors": {"q": ("I32", (100, 100))}})
        self.assertTrue(model_info.inspect(gptq)["quantisation"].startswith("gptq 4-bit"))

    def test_packed_bytes_are_not_called_8_bit(self):
        folder = self.model(shards={"m.safetensors": {"q": ("U8", (1000, 1000)), "s": ("F16", (100,))}})
        self.assertEqual(model_info.inspect(folder)["quantisation"], "u8 packed")

    def test_a_pipeline_folder_is_read_one_level_down(self):
        folder = make_fake_pipeline(self.root / "qwen-image")
        found = model_info.inspect(folder)
        self.assertEqual(found["family"], "qwen_image")
        self.assertEqual(found["modality"], "image")
        self.assertGreater(found["shards"], 0)
        self.assertEqual(found["problems"], [])

    def test_a_folder_nothing_claims_says_which_piece_is_missing(self):
        found = model_info.inspect(self.model(config={"model_type": "unheard-of"}))
        self.assertEqual(found["architecture"], "unheard-of")
        self.assertTrue(any("no engine" in problem for problem in found["problems"]), found["problems"])
        empty = self.root / "empty"
        empty.mkdir()
        problems = model_info.inspect(empty)["problems"]
        self.assertTrue(any("config.json" in p for p in problems), problems)
        self.assertTrue(any("no safetensors" in p for p in problems), problems)
        self.assertEqual(model_info.inspect(self.root / "gone")["problems"], ["the folder is not there"])

    def test_an_unreadable_shard_is_a_problem_and_the_rest_is_still_counted(self):
        folder = self.model(shards={"ok.safetensors": {"a": ("F32", (4,))}})
        (folder / "bad.safetensors").write_bytes(b"junk")
        found = model_info.inspect(folder)
        self.assertEqual(found["shards"], 1)
        self.assertTrue(any("bad.safetensors" in p for p in found["problems"]))

    def test_the_source_comes_from_the_manifest_and_an_unfinished_download_is_a_problem(self):
        folder = self.model()
        manifest_for(folder, ["model-00001.safetensors"])
        found = model_info.inspect(folder)
        self.assertEqual(found["source"], {"repo": "acme/model", "revision": "abc123def4567"})
        self.assertTrue(found["complete"])
        manifest_for(folder, ["model-00001.safetensors"], complete=False)
        self.assertTrue(any("did not finish" in p for p in model_info.inspect(folder)["problems"]))

    def test_the_description_reads_like_a_page_of_facts(self):
        folder = self.model(config={"architectures": ["ToyForCausalLM"], "max_position_embeddings": 8192})
        manifest_for(folder, ["model-00001.safetensors"])
        text = "\n".join(model_info.describe(model_info.inspect(folder)))
        for fragment in ("ToyForCausalLM", "8,192 tokens", "bf16", "1 safetensors, 2 tensors", "acme/model @ abc123def456"):
            self.assertIn(fragment, text)
        self.assertIn("not recorded", "\n".join(model_info.describe(model_info.inspect(self.model("plain")))))


class VerifyTest(Folder):
    def test_a_whole_model_with_a_manifest_is_ok(self):
        folder = self.model()
        manifest_for(folder, ["config.json", "model-00001.safetensors"])
        found = model_info.verify_model(folder)
        self.assertEqual((found["status"], found["checked"], found["bad"], found["hashed"]), ("ok", 2, [], 2))
        self.assertEqual(found["how"], "sizes, checksums and safetensors headers")

    def test_a_manifest_without_checksums_says_so_and_still_catches_a_cut_file_by_its_header(self):
        folder = self.model()
        Path(folder, setup_download.MANIFEST).write_text(json.dumps({"complete": True, "files": [
            {"path": "config.json", "size": (folder / "config.json").stat().st_size},
            {"path": "model-00001.safetensors", "size": (folder / "model-00001.safetensors").stat().st_size}]}))
        found = model_info.verify_model(folder)
        self.assertEqual((found["status"], found["hashed"]), ("ok", 0))
        self.assertIn("holds no checksums", found["how"])
        self.assertIn("quick", model_info.verify_model(folder, quick=True)["how"])
        cut = (folder / "model-00001.safetensors").read_bytes()[:-3]
        (folder / "model-00001.safetensors").write_bytes(cut)
        Path(folder, setup_download.MANIFEST).write_text(json.dumps({"complete": True, "files": [
            {"path": "model-00001.safetensors", "size": len(cut)}]}))                       # a manifest that was written for the short file
        self.assertIn("cut short", model_info.verify_model(folder)["bad"][0]["problem"])

    def test_a_missing_short_and_changed_file_are_each_named(self):
        folder = self.model(shards={"a.safetensors": {"x": ("F32", (8,))}, "b.safetensors": {"y": ("F32", (8,))}, "c.safetensors": {"z": ("F32", (8,))}})
        manifest_for(folder, ["a.safetensors", "b.safetensors", "c.safetensors", "config.json"])
        (folder / "a.safetensors").unlink()
        (folder / "b.safetensors").write_bytes((folder / "b.safetensors").read_bytes()[:-4])
        data = bytearray((folder / "c.safetensors").read_bytes())
        data[-1] ^= 0xFF                                                                     # the size is right, a byte is not
        (folder / "c.safetensors").write_bytes(bytes(data))
        found = model_info.verify_model(folder)
        problems = {item["path"]: item["problem"] for item in found["bad"]}
        self.assertEqual(found["status"], "damaged")
        self.assertEqual(problems["a.safetensors"], "missing")
        self.assertIn("should be", problems["b.safetensors"])
        self.assertEqual(problems["c.safetensors"], "checksum mismatch")
        self.assertEqual(found["checked"], 1)

    def test_quick_checks_sizes_only_and_so_misses_a_flipped_byte(self):
        folder = self.model()
        manifest_for(folder, ["model-00001.safetensors"])
        data = bytearray((folder / "model-00001.safetensors").read_bytes())
        data[-1] ^= 0xFF
        (folder / "model-00001.safetensors").write_bytes(bytes(data))
        self.assertEqual(model_info.verify_model(folder, quick=True)["status"], "ok")
        self.assertEqual(model_info.verify_model(folder)["status"], "damaged")

    def test_an_unfinished_download_is_damaged_even_if_every_listed_file_is_there(self):
        folder = self.model()
        manifest_for(folder, ["model-00001.safetensors"], complete=False)
        found = model_info.verify_model(folder)
        self.assertEqual(found["status"], "damaged")
        self.assertIn("never finished", found["bad"][-1]["problem"])

    def test_a_name_in_the_manifest_that_leaves_the_folder_is_refused(self):
        folder = self.model()
        Path(folder, setup_download.MANIFEST).write_text(json.dumps({"complete": True, "files": [{"path": "../../etc/passwd", "size": 1}]}))
        found = model_info.verify_model(folder)
        self.assertEqual(found["bad"][0]["problem"], "a name that leaves the folder")

    def test_without_a_manifest_each_safetensors_file_is_checked_against_its_own_header(self):
        folder = self.model(shards={"a.safetensors": {"x": ("F32", (8,))}, "b.safetensors": {"y": ("F32", (8,))}})
        self.assertEqual(model_info.verify_model(folder)["status"], "ok")
        (folder / "b.safetensors").write_bytes((folder / "b.safetensors").read_bytes()[:-1])
        found = model_info.verify_model(folder)
        self.assertEqual((found["status"], [b["path"] for b in found["bad"]]), ("damaged", ["b.safetensors"]))
        self.assertIn("no manifest", found["how"])

    def test_nothing_to_check_is_said_so(self):
        folder = self.root / "bare"
        folder.mkdir()
        found = model_info.verify_model(folder)
        self.assertEqual(found["status"], "unverifiable")

    def test_progress_is_reported_before_each_file(self):
        folder = self.model()
        manifest_for(folder, ["config.json", "model-00001.safetensors"])
        seen = []
        model_info.verify_model(folder, progress=lambda done, total, path: seen.append((done, total, path)))
        self.assertEqual(seen, [(0, 2, "config.json"), (1, 2, "model-00001.safetensors")])


class CliTest(Folder):
    """`coli model info` and `coli model verify`, run as the real command on a folder."""

    def coli(self, *args):
        env = {**os.environ, "NO_COLOR": "1", "PYTHONUTF8": "1", "COLI_SETUP_HOME": self.tmp.name}
        done = subprocess.run([sys.executable, str(Path(model_info.__file__).parent / "coli"), "model", *args], capture_output=True, text=True,
                              env=env, timeout=120)
        return done.returncode, done.stdout + done.stderr

    def test_info_describes_a_folder_without_a_server(self):
        folder = self.model(config={"architectures": ["ToyForCausalLM"], "max_position_embeddings": 4096})
        code, out = self.coli("info", str(folder))
        self.assertEqual(code, 0, out)
        for fragment in ("ToyForCausalLM", "4,096 tokens", "bf16", "no engine"):
            self.assertIn(fragment, out)

    def test_verify_exits_0_for_a_whole_model_and_1_with_the_file_named_for_a_damaged_one(self):
        folder = self.model()
        manifest_for(folder, ["config.json", "model-00001.safetensors"])
        code, out = self.coli("verify", str(folder))
        self.assertEqual(code, 0, out)
        self.assertIn("2 files whole", out)
        (folder / "model-00001.safetensors").write_bytes(b"short")
        code, out = self.coli("verify", str(folder), "--quick")
        self.assertEqual(code, 1, out)
        self.assertIn("model-00001.safetensors: is 5 bytes", out)

    def test_a_name_nobody_has_is_said_so(self):
        code, out = self.coli("info", "no-such-model-anywhere")
        self.assertEqual(code, 1, out)
        self.assertIn("no installed model is called no-such-model-anywhere", out)


if __name__ == "__main__":
    unittest.main()
