"""The real engine (c/qwen3embed) against stored transformers vectors, and its protocol on bad input.

    QWEN3EMBED_MODEL=/path/to/Qwen3-Embedding-0.6B python -m unittest tests.test_qwen3embed_real

Needs the built engine (make qwen3embed) and the 0.6B checkpoint (coli model install qwen3-embedding-0.6b); skipped without
them. The reference is tools/qwen3_embedding_0.6b_reference.json, made by `tools/embedding_ref.py capture` from transformers in
fp32 (the model card's recipe), so these tests need neither torch nor transformers.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXE = next((p for p in (os.path.join(C_DIR, "qwen3embed.exe"), os.path.join(C_DIR, "qwen3embed")) if os.path.isfile(p)), None)
MODEL = os.environ.get("QWEN3EMBED_MODEL")
REFERENCE = os.path.join(C_DIR, "tools", "qwen3_embedding_0.6b_reference.json")


class Engine:
    def __init__(self, model=MODEL, extra=()):
        self.process = subprocess.Popen([EXE, "--model", model, "--serve", *extra], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1)
        self.ready = self.process.stdout.readline()

    def send(self, line):
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        keyword, _, body = self.process.stdout.readline().partition(" ")
        return keyword, json.loads(body)

    def embed(self, inputs, **fields):
        return self.send("EMBED " + json.dumps({"id": "t", "inputs": inputs, **fields}))

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=60)


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


@unittest.skipUnless(EXE and MODEL and os.path.isdir(MODEL), "needs the built qwen3embed and QWEN3EMBED_MODEL (the 0.6B checkpoint)")
class RealEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_ready_says_what_was_loaded(self):
        keyword, _, body = self.engine.ready.partition(" ")
        info = json.loads(body)
        self.assertEqual(keyword, "READY")
        self.assertEqual((info["dimensions"], info["layers"], info["max_tokens"]), (1024, 28, 8192))

    def test_the_vectors_match_transformers(self):
        with open(REFERENCE, encoding="utf-8") as handle:
            reference = json.load(handle)
        for case in reference["cases"]:
            with self.subTest(text=case["text"][:30], instruction=bool(case["instruction"]), dimensions=case["dimensions"]):
                fields = {}
                if case["instruction"]:
                    fields["instruction"] = case["instruction"]
                if case["dimensions"]:
                    fields["dimensions"] = case["dimensions"]
                keyword, body = self.engine.embed([case["text"]], **fields)
                self.assertEqual(keyword, "VECTORS", body)
                self.assertEqual(body["tokens"], [case["tokens"]])
                got = body["vectors"][0]
                self.assertEqual(len(got), len(case["vector"]))
                self.assertGreater(cosine(got, case["vector"]), 0.99999)
                self.assertLess(max(abs(a - b) for a, b in zip(got, case["vector"])), 1e-4)

    def test_a_text_gives_the_same_vector_alone_and_in_a_batch(self):
        long_text = "The quick brown fox jumps over the lazy dog. " * 40
        _, alone = self.engine.embed(["a short question about embeddings"])
        _, batch = self.engine.embed([long_text, "a short question about embeddings", "x", long_text[:300]])
        self.assertGreater(cosine(alone["vectors"][0], batch["vectors"][1]), 0.99999)
        _, again = self.engine.embed([long_text])
        self.assertGreater(cosine(again["vectors"][0], batch["vectors"][0]), 0.99999)

    def test_every_vector_is_a_unit_vector_and_a_cut_one_is_normalised_again(self):
        _, body = self.engine.embed(["alpha", "beta"], dimensions=32)
        for vector in body["vectors"]:
            self.assertEqual(len(vector), 32)
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in vector)), 1.0, places=5)

    def test_a_query_is_not_the_same_vector_as_the_document(self):
        _, document = self.engine.embed(["how do I reset my password"])
        _, query = self.engine.embed(["how do I reset my password"], instruction="Retrieve relevant passages")
        self.assertLess(cosine(document["vectors"][0], query["vectors"][0]), 0.999)

    def test_the_meaning_decides_what_is_near(self):
        _, body = self.engine.embed(["a cat sat on the mat", "a kitten rests on a rug", "quarterly revenue grew by nine percent"])
        cat, kitten, revenue = body["vectors"]
        self.assertGreater(cosine(cat, kitten), cosine(cat, revenue) + 0.2)

    def test_bad_requests_get_an_error_and_the_engine_goes_on(self):
        for line in ('EMBED {"id":"x"}', 'EMBED {"id":"x","inputs":[]}', 'EMBED {"id":"x","inputs":["a",3]}',
                     'EMBED {"id":"x","inputs":["a"],"dimensions":99999}', 'EMBED not json', 'HELLO {}', 'EMBED [1]'):
            with self.subTest(line=line):
                keyword, body = self.engine.send(line)
                self.assertEqual(keyword, "ERROR", body)
                self.assertTrue(body["message"])
        keyword, body = self.engine.embed(["still alive"])
        self.assertEqual(keyword, "VECTORS")

    def test_an_input_over_the_limit_is_cut_or_refused(self):
        engine = Engine(extra=("--max-tokens", "16"))
        try:
            text = "word " * 100
            _, cut = engine.embed([text])
            self.assertEqual((cut["tokens"], cut["truncated"]), ([16], [1]))
            keyword, refused = engine.embed([text], truncate=False)
            self.assertEqual(keyword, "ERROR")
            self.assertIn("longer than 16 tokens", refused["message"])
            _, fits = engine.embed(["a few words"])
            self.assertEqual(fits["truncated"], [0])
        finally:
            engine.close()


@unittest.skipUnless(EXE and MODEL and os.path.isdir(MODEL), "needs the built qwen3embed and QWEN3EMBED_MODEL (the 0.6B checkpoint)")
class BadCheckpointTest(unittest.TestCase):
    """A checkpoint that is not whole ends in a named error and a non-zero exit, never a crash or a run on garbage."""

    def run_engine(self, model):
        return subprocess.run([EXE, "--model", model, "--serve"], input="", capture_output=True, text=True, timeout=60)

    def test_no_such_folder(self):
        result = self.run_engine(os.path.join(tempfile.gettempdir(), "no-such-embedding-folder"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("qwen3embed:", result.stderr)

    def test_without_the_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("config.json", "model.safetensors"):
                try:
                    os.link(os.path.join(MODEL, name), os.path.join(tmp, name))
                except OSError:
                    shutil.copy(os.path.join(MODEL, name), tmp)
            result = self.run_engine(tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("tokenizer", result.stderr)

    def test_without_the_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("config.json", "tokenizer.json"):
                shutil.copy(os.path.join(MODEL, name), tmp)
            result = self.run_engine(tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("qwen3embed:", result.stderr)

    def test_a_config_that_is_not_qwen3(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("tokenizer.json", "model.safetensors"):
                shutil.copy(os.path.join(MODEL, name), tmp)
            with open(os.path.join(MODEL, "config.json"), encoding="utf-8") as handle:
                config = json.load(handle)
            config["model_type"] = "llama"
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            result = self.run_engine(tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("model_type", result.stderr)

    def test_the_usage_line_when_asked_for_nothing_or_both(self):
        for args in ([], ["--model", MODEL], ["--model", MODEL, "--serve", "--text", "x"]):
            result = subprocess.run([EXE, *args], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 2, args)
            self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
