"""The real engine (c/qwen3embed on a reranker checkpoint) against stored transformers scores, and its protocol on bad input.

    QWEN3RERANK_MODEL=/path/to/Qwen3-Reranker-0.6B python -m unittest tests.test_qwen3rerank_real

Needs the built engine (make qwen3embed) and the 0.6B reranker checkpoint (coli model install qwen3-reranker-0.6b); skipped
without them. The reference is tools/qwen3_reranker_0.6b_reference.json, made by `tools/rerank_ref.py capture` from
transformers in fp32 (the model card's recipe), so these tests need neither torch nor transformers.
"""
import json
import math
import os
import re
import subprocess
import unittest

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXE = next((p for p in (os.path.join(C_DIR, "qwen3embed.exe"), os.path.join(C_DIR, "qwen3embed")) if os.path.isfile(p)), None)
MODEL = os.environ.get("QWEN3RERANK_MODEL")
REFERENCE = os.path.join(C_DIR, "tools", "qwen3_reranker_0.6b_reference.json")


class Engine:
    def __init__(self, model=MODEL, extra=(), env=None, capture=False):
        self.process = subprocess.Popen([EXE, "--model", model, "--serve", *extra], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE if capture else None, text=True, encoding="utf-8", bufsize=1,
                                        env=dict(os.environ, **env) if env else None)
        self.ready = self.process.stdout.readline()
        self.stderr = ""                                   # what the engine said, when capture is on: set by close()

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
        if self.process.stderr is not None:
            self.stderr = self.process.stderr.read()
            self.process.stderr.close()
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
            self.assertAlmostEqual(got, want, places=4)             # on a graphics card the two paths cut the sums differently
        self.assertEqual(body["logits"][0], body["logits"][5])                  # the same pair twice is the same score
        self.assertGreater(body["scores"][0], body["scores"][1])
        # every pair the same: the whole text is shared, yet each pair keeps its last row (the score is read from it)
        keyword, body = self.engine.rerank("What is the capital of France?", ["Paris."] * 3)
        self.assertEqual(keyword, "SCORES", body)
        self.assertEqual(len(set(body["logits"])), 1)
        self.assertAlmostEqual(body["logits"][0], self.engine.rerank("What is the capital of France?", ["Paris."])[1]["logits"][0], places=4)

    def test_the_graphics_card_scores_like_the_cpu(self):
        # The same engine with COLI_VULKAN=1 and =0 (a build without Vulkan, or no device, says "cpu" and the test is skipped).
        card = Engine(env={"COLI_VULKAN": "1"})
        try:
            if json.loads(card.ready.partition(" ")[2]).get("device") != "vulkan":
                self.skipTest("this build or machine has no Vulkan device")
            cpu = Engine(env={"COLI_VULKAN": "0"})
            try:
                docs = ["The capital of France is Paris.", "The capital of Spain is Madrid.", "", "Bananas are rich in potassium. " * 40,
                        "日本の首都は東京です。", "The capital of France is Paris."] + [f"Passage {i} about cities." for i in range(30)]
                keyword, want = cpu.rerank("What is the capital of France?", docs)
                self.assertEqual(keyword, "SCORES", want)
                keyword, got = card.rerank("What is the capital of France?", docs)
                self.assertEqual(keyword, "SCORES", got)
                self.assertEqual(got["tokens"], want["tokens"])
                for a, b in zip(got["logits"], want["logits"]):
                    self.assertAlmostEqual(a, b, places=3)
                self.assertEqual(sorted(range(len(docs)), key=lambda i: -round(got["logits"][i], 2)),
                                 sorted(range(len(docs)), key=lambda i: -round(want["logits"][i], 2)))
                keyword, body = card.rerank("anything", ["word " * 3000, "short"], truncate=True)      # a long text on the card
                self.assertEqual(keyword, "SCORES", body)
                self.assertEqual(card.rerank("q", ["alive"])[0], "SCORES")
            finally:
                cpu.close()
        finally:
            card.close()

    SEQUENCE = [                      # requests one after another to one engine: the second reuses the first's start, the third part of it
        ("What is the capital of France?", ["Paris is the capital of France.", "Berlin is in Germany.", "Madrid", "Paris"], {}),
        ("What is the capital of France?", ["The capital of France is Paris, a city on the Seine.", "Lyon is a city in France."], {}),
        ("Who wrote Hamlet?", ["Shakespeare wrote Hamlet around 1600.", "Marlowe wrote Doctor Faustus.", "William Shakespeare"], {}),
        ("Who wrote Hamlet?", ["Shakespeare wrote Hamlet around 1600."], {}),                                  # one pair: its start is kept too
        ("What is the capital of France?", ["Paris."], {}),
        ("What is the capital of France?", ["Paris is the capital of France.", "Berlin is in Germany."], {"instruction": "Judge by tone"}),
        ("Who wrote Hamlet?", ["x " * 400, "short"], {"truncate": True}),
        ("What is the capital of France?", ["", " ", "Paris"], {}),
    ]

    def run_sequence(self, env, capture=True):
        engine = Engine(env=env, capture=capture)
        try:
            answers = []
            for query, docs, fields in self.SEQUENCE:
                keyword, body = engine.rerank(query, docs, **fields)
                self.assertEqual(keyword, "SCORES", body)
                answers.append(body)
        finally:
            engine.close()
        return answers, engine

    @staticmethod
    def kept_counts(engine):
        found = re.search(r"kept start: (\d+) requests used it whole, (\d+) in part, (\d+) made a new one; (\d+) rows", engine.stderr)
        return tuple(int(x) for x in found.groups()) if found else None

    def test_a_start_kept_between_requests_changes_no_score(self):
        # The host keeps the start of a request (the fixed prompt, the instruction, the query) for the next one: the same query
        # again, a new query under the same instruction, one pair at a time. A fresh engine without the cache is the reference,
        # and the scores must be the same to the last digit.
        kept, engine = self.run_sequence({"COLI_VULKAN": "0"})
        fresh, quiet = self.run_sequence({"COLI_VULKAN": "0", "COLI_RERANK_CACHE": "0"})
        for (query, docs, _), a, b in zip(self.SEQUENCE, kept, fresh):
            self.assertEqual(a["tokens"], b["tokens"], query)
            self.assertEqual(a["logits"], b["logits"], query)
        whole, part, new, rows = self.kept_counts(engine)
        self.assertGreaterEqual(whole, 1)                               # the same start again
        self.assertGreaterEqual(part, 1)                                # only the fixed prompt and instruction in common
        self.assertGreaterEqual(new, 1)
        self.assertGreater(rows, 100)                                   # rows that did not go through the layers again
        self.assertIsNone(self.kept_counts(quiet))                      # COLI_RERANK_CACHE=0 keeps nothing

    def test_the_graphics_card_makes_the_start_for_each_request(self):
        # On the device chain a kept start gave scores that were now and then wrong (cause not found), so there the start is made
        # again for each request: the scores match the CPU's and the engine reports nothing kept.
        card = Engine(env={"COLI_VULKAN": "1"})
        try:
            if json.loads(card.ready.partition(" ")[2]).get("device") != "vulkan":
                self.skipTest("this build or machine has no Vulkan device")
        finally:
            card.close()
        kept, engine = self.run_sequence({"COLI_VULKAN": "1"})
        cpu, _ = self.run_sequence({"COLI_VULKAN": "0", "COLI_RERANK_CACHE": "0"})
        for (query, docs, _), a, b in zip(self.SEQUENCE, kept, cpu):
            self.assertEqual(a["tokens"], b["tokens"], query)
            for x, y in zip(a["logits"], b["logits"]):
                self.assertAlmostEqual(x, y, places=3, msg=query)
        self.assertIsNone(self.kept_counts(engine))
        # many requests of one pair each, which is where the kept start used to go wrong
        card = Engine(env={"COLI_VULKAN": "1"})
        cpu = Engine(env={"COLI_VULKAN": "0", "COLI_RERANK_CACHE": "0"})
        try:
            query = "which planet is known as the red planet and why does it look red from the earth?"
            for i in range(40):
                doc = f"Passage {i}: Mars is the fourth planet from the Sun; iron oxide on its surface gives it a reddish colour ({i})."
                got, want = card.rerank(query, [doc])[1]["logits"][0], cpu.rerank(query, [doc])[1]["logits"][0]
                self.assertAlmostEqual(got, want, places=3, msg=f"request {i}")
        finally:
            card.close()
            cpu.close()

    def test_a_frame_that_fails_while_the_start_is_made_falls_back_to_the_cpu(self):
        # COLI_VK_CHAIN_FAULT=n makes the n-th device submission fail as a lost device would. A request's first forward makes the shared
        # start and keeps its keys and values for the pairs; when the device is lost while its buffers are made the CPU must redo it with the
        # host copies (the engine once freed them there and died with an access violation).
        card = Engine(env={"COLI_VULKAN": "1"})
        try:
            if json.loads(card.ready.partition(" ")[2]).get("device") != "vulkan":
                self.skipTest("this build or machine has no Vulkan device")
        finally:
            card.close()
        docs = ["Paris is the capital of France.", "Paris is the capital of Spain, said nobody.", "Berlin is in Germany.", "Madrid", "Paris",
                "Bananas are yellow fruit."]
        cpu = Engine(env={"COLI_VULKAN": "0"})
        try:
            want = cpu.rerank("what is the capital of France?", docs)[1]["logits"]
        finally:
            cpu.close()
        lost = 0
        for n in (24, 26, 30):
            engine = Engine(env={"COLI_VULKAN": "1", "COLI_VK_CHAIN_FAULT": str(n)}, capture=True)
            try:
                keyword, body = engine.rerank("what is the capital of France?", docs)
            finally:
                engine.close()
            self.assertEqual(engine.process.returncode, 0, f"fault {n}")
            self.assertEqual(keyword, "SCORES", body)
            for x, y in zip(body["logits"], want):
                self.assertAlmostEqual(x, y, places=3, msg=f"fault {n}")
            lost += "device is lost" in engine.stderr
        self.assertGreaterEqual(lost, 1)                                # at least one of them did reach a frame of this request

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
