"""The real engine (c/qwen3embed on a reranker checkpoint) against stored transformers scores, and its protocol on bad input.

    QWEN3RERANK_MODEL=/path/to/Qwen3-Reranker-0.6B python -m unittest tests.test_qwen3rerank_real

Needs the built engine (make qwen3embed) and the 0.6B reranker checkpoint (coli model install qwen3-reranker-0.6b); skipped
without them. The reference is tools/qwen3_reranker_0.6b_reference.json, made by `tools/rerank_ref.py capture` from
transformers in fp32 (the model card's recipe), so these tests need neither torch nor transformers.
"""
import json
import math
import os
import subprocess
import unittest

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXE = next((p for p in (os.path.join(C_DIR, "qwen3embed.exe"), os.path.join(C_DIR, "qwen3embed")) if os.path.isfile(p)), None)
MODEL = os.environ.get("QWEN3RERANK_MODEL")
REFERENCE = os.path.join(C_DIR, "tools", "qwen3_reranker_0.6b_reference.json")


class Engine:
    def __init__(self, model=MODEL, extra=(), env=None):
        self.process = subprocess.Popen([EXE, "--model", model, "--serve", *extra], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1, env=dict(os.environ, **env) if env else None)
        self.ready = self.process.stdout.readline()

    def send(self, line):
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        keyword, _, body = self.process.stdout.readline().partition(" ")
        return keyword, json.loads(body)

    def rerank(self, query, documents, **fields):
        return self.send("RERANK " + json.dumps({"id": "t", "query": query, "documents": documents, **fields}))

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=60)
        self.process.stdout.close()


@unittest.skipUnless(EXE and MODEL and os.path.isdir(MODEL), "needs the built qwen3embed and QWEN3RERANK_MODEL (the 0.6B reranker)")
class RealEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_ready_says_it_is_a_reranker(self):
        keyword, _, body = self.engine.ready.partition(" ")
        info = json.loads(body)
        self.assertEqual(keyword, "READY")
        self.assertEqual((info["task"], info["layers"], info["max_tokens"]), ("rerank", 28, 8192))
        self.assertNotIn("dimensions", info)

    def test_the_scores_match_transformers(self):
        with open(REFERENCE, encoding="utf-8") as handle:
            reference = json.load(handle)
        for case in reference:
            with self.subTest(query=case["query"][:30], document=case["document"][:20]):
                fields = {"instruction": case["instruction"]} if case["instruction"] else {}
                keyword, body = self.engine.rerank(case["query"], [case["document"]], **fields)
                self.assertEqual(keyword, "SCORES", body)
                self.assertEqual(body["tokens"], [case["tokens"]])
                self.assertLess(abs(body["logits"][0] - case["logit"]), 1e-3)
                self.assertLess(abs(body["scores"][0] - case["score"]), 1e-4)

    def test_a_batch_scores_each_pair_as_it_would_alone_and_keeps_the_order(self):
        with open(REFERENCE, encoding="utf-8") as handle:
            reference = [c for c in json.load(handle) if not c["instruction"] and c["query"] == "Explain gravity"]
        docs = [c["document"] for c in reference]
        keyword, body = self.engine.rerank("Explain gravity", docs + docs[::-1])
        self.assertEqual(keyword, "SCORES", body)
        n = len(docs)
        for i, case in enumerate(reference):
            self.assertLess(abs(body["logits"][i] - case["logit"]), 1e-3)
            self.assertLess(abs(body["logits"][2 * n - 1 - i] - case["logit"]), 1e-3)

    def test_pairs_that_share_a_start_score_as_if_each_ran_alone(self):
        # One request: the pairs share the fixed prefix, the instruction, the query (and here also a first word); the engine runs
        # that start through the layers once. COLI_RERANK_SHARE=0 turns that off, so the two engines must agree to the last digit.
        docs = ["The capital of France is Paris.", "The capital of Spain is Madrid.", "The weather is mild today.", "The", "",
                "The capital of France is Paris."]
        alone = Engine(env={"COLI_RERANK_SHARE": "0"})
        try:
            keyword, expected = alone.rerank("What is the capital of France?", docs)
        finally:
            alone.close()
        self.assertEqual(keyword, "SCORES", expected)
        keyword, body = self.engine.rerank("What is the capital of France?", docs)
        self.assertEqual(keyword, "SCORES", body)
        self.assertEqual(body["tokens"], expected["tokens"])
        for got, want in zip(body["logits"], expected["logits"]):
            self.assertAlmostEqual(got, want, places=5)
        self.assertEqual(body["logits"][0], body["logits"][5])                  # the same pair twice is the same score
        self.assertGreater(body["scores"][0], body["scores"][1])
        # every pair the same: the whole text is shared, yet each pair keeps its last row (the score is read from it)
        keyword, body = self.engine.rerank("What is the capital of France?", ["Paris."] * 3)
        self.assertEqual(keyword, "SCORES", body)
        self.assertEqual(len(set(body["logits"])), 1)
        self.assertEqual(body["logits"][0], self.engine.rerank("What is the capital of France?", ["Paris."])[1]["logits"][0])

    def test_the_right_document_wins_and_a_score_is_a_probability(self):
        keyword, body = self.engine.rerank("What is the capital of China?", [
            "Gravity is a force that attracts two bodies towards each other.", "The capital of China is Beijing."])
        self.assertEqual(keyword, "SCORES", body)
        self.assertGreater(body["scores"][1], 0.99)
        self.assertLess(body["scores"][0], 0.01)
        for score, logit in zip(body["scores"], body["logits"]):
            self.assertAlmostEqual(score, 1 / (1 + math.exp(-logit)), places=6)

    def test_a_long_document_is_cut_to_the_limit_and_says_so(self):
        small = Engine(extra=("--max-tokens", "200"))               # the limit is a flag, so the test needs no 8192-token pair
        try:
            long = "word " * 3000
            keyword, body = small.rerank("anything", [long, "short"])
            self.assertEqual(keyword, "SCORES", body)
            self.assertEqual((body["truncated"], body["tokens"][0]), ([1, 0], 200))
            self.assertTrue(0 < body["scores"][0] < 1)
            keyword, body = small.rerank("anything", [long], truncate=False)
            self.assertEqual(keyword, "ERROR")
            self.assertIn("longer than 200 tokens", body["message"])
            self.assertEqual(small.rerank("anything", ["short"])[0], "SCORES")      # a refusal is not a crash
        finally:
            small.close()

    def test_a_bad_request_is_an_error_and_the_engine_goes_on(self):
        for fields in ({"query": 5, "documents": ["a"]}, {"query": "q", "documents": []}, {"query": "q", "documents": ["a", 3]},
                       {"query": "q", "documents": "a"}):
            with self.subTest(fields=fields):
                keyword, body = self.engine.send("RERANK " + json.dumps({"id": "bad", **fields}))
                self.assertEqual((keyword, body["id"]), ("ERROR", "bad"))
        self.assertEqual(self.engine.send("EMBED " + json.dumps({"id": "e", "inputs": ["a"]}))[0], "ERROR")     # it is not an embedder
        self.assertEqual(self.engine.send("RERANK not json")[0], "ERROR")
        self.assertEqual(self.engine.rerank("q", ["still alive"])[0], "SCORES")


if __name__ == "__main__":
    unittest.main()
