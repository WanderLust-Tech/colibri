"""POST /v1/rerank, end to end against the stub engine, and the rules around it.

A real RerankEngine drives tools/rerank_stub.py over its serve protocol, and a real APIServer answers HTTP on a free port:
nothing between the socket and the engine's stdin is mocked. The stub's score is the share of the query's words a document
contains, which is what lets these tests pin the sorting, `top_n`, the order of a batch and the accounting.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embedding_engine  # noqa: E402
from embedding_engine import (EmbeddingEngine, EmbeddingEngineError, EmbeddingRequestError, RerankEngine,  # noqa: E402
                              rerank_payload, rerank_request)
from family_registry import (UnknownFamilyError, default_model_id, display_for, family_by_id,  # noqa: E402
                             resolve_model)
from openai_server import APIServer  # noqa: E402

TOOLS = Path(__file__).resolve().parent.parent / "tools"
STUB = str(TOOLS / "rerank_stub.py")
EMBED_STUB = str(TOOLS / "embedding_stub.py")
MODEL_ID = "qwen3-reranker-0.6b"
KEY = "secret"
QUERY = "capital of france"
DOCS = ["gravity pulls things down", "paris is the capital of france", "france makes cheese", "the capital of"]


def stub_env(**extra):
    env = dict(os.environ)
    env.update({key: str(value) for key, value in extra.items()})
    return env


class RequestRulesTest(unittest.TestCase):
    INFO = {"max_tokens": 12, "max_documents": 5}

    def check(self, body):
        return rerank_request(body, self.INFO)

    def refused(self, body, param, fragment=""):
        with self.assertRaises(EmbeddingRequestError) as caught:
            self.check(body)
        self.assertEqual(caught.exception.param, param, str(caught.exception))
        self.assertIn(fragment, str(caught.exception))

    def test_the_defaults(self):
        request = self.check({"query": "q", "documents": ["a", "b"]})
        self.assertEqual(request, {"query": "q", "documents": ["a", "b"], "top_n": None, "return_documents": False,
                                   "instruction": None, "truncate": True})

    def test_a_document_may_be_a_string_or_an_object_with_text(self):
        request = self.check({"query": "q", "documents": ["a", {"text": "b"}]})
        self.assertEqual(request["documents"], ["a", "b"])

    def test_jinas_query_object_is_taken_for_its_text(self):
        self.assertEqual(self.check({"query": {"text": "find"}, "documents": ["a"]})["query"], "find")

    def test_the_query_and_the_documents_are_required(self):
        self.refused({"documents": ["a"]}, "query", "required")
        self.refused({"query": "", "documents": ["a"]}, "query")
        self.refused({"query": "   ", "documents": ["a"]}, "query")
        self.refused({"query": 5, "documents": ["a"]}, "query")
        self.refused({"query": "q"}, "documents", "required")
        self.refused({"query": "q", "documents": []}, "documents")
        self.refused({"query": "q", "documents": "a"}, "documents")
        self.refused({"query": "q", "documents": ["a", 3]}, "documents", "string")
        self.refused({"query": "q", "documents": ["a", {"body": "b"}]}, "documents", "string")
        self.refused({"query": "q", "documents": ["a"] * 6}, "documents", "at most 5")

    def test_an_empty_document_is_a_document_like_any_other(self):
        self.assertEqual(self.check({"query": "q", "documents": [""]})["documents"], [""])

    def test_the_other_fields(self):
        self.refused({"query": "q", "documents": ["a"], "top_n": 0}, "top_n", "positive")
        self.refused({"query": "q", "documents": ["a"], "top_n": "2"}, "top_n")
        self.refused({"query": "q", "documents": ["a"], "top_n": True}, "top_n")
        self.refused({"query": "q", "documents": ["a"], "return_documents": "yes"}, "return_documents")
        self.refused({"query": "q", "documents": ["a"], "instruction": 3}, "instruction")
        self.refused({"query": "q", "documents": ["a"], "instruction": "x" * (embedding_engine.MAX_INSTRUCTION_BYTES + 1)},
                     "instruction", "over")
        self.refused({"query": "q", "documents": ["a"], "truncate": 1}, "truncate")
        with self.assertRaises(EmbeddingRequestError):
            rerank_request([], self.INFO)

    def test_the_whole_request_is_checked(self):
        full = self.check({"query": "q", "documents": ["a"], "top_n": 3, "return_documents": True, "instruction": "find it",
                           "truncate": False, "model": "x"})
        self.assertEqual((full["top_n"], full["return_documents"], full["instruction"], full["truncate"]), (3, True, "find it", False))

    def test_results_come_best_first_with_the_position_of_each_document(self):
        request = {"documents": ["a", "b", "c"], "top_n": None, "return_documents": False}
        result = {"scores": [0.2, 0.9, 0.5], "tokens": [3, 4, 5], "truncated": [False, False, False]}
        payload = rerank_payload(request, result, "m", "req_1")
        self.assertEqual([(r["index"], r["relevance_score"]) for r in payload["results"]], [(1, 0.9), (2, 0.5), (0, 0.2)])
        self.assertEqual((payload["model"], payload["id"], payload["usage"]), ("m", "req_1", {"prompt_tokens": 12, "total_tokens": 12}))
        self.assertNotIn("colibri", payload)
        self.assertNotIn("document", payload["results"][0])

    def test_equal_scores_keep_the_order_the_documents_came_in(self):
        request = {"documents": ["a", "b", "c"], "top_n": None, "return_documents": False}
        result = {"scores": [0.5, 0.5, 0.5], "tokens": [1, 1, 1], "truncated": [False] * 3}
        self.assertEqual([r["index"] for r in rerank_payload(request, result, "m")["results"]], [0, 1, 2])

    def test_top_n_and_return_documents_and_the_truncated_list(self):
        request = {"documents": ["a", "b", "c"], "top_n": 2, "return_documents": True}
        result = {"scores": [0.2, 0.9, 0.5], "tokens": [3, 4, 5], "truncated": [False, True, False]}
        payload = rerank_payload(request, result, "m")
        self.assertEqual([(r["index"], r["document"]) for r in payload["results"]], [(1, {"text": "b"}), (2, {"text": "c"})])
        self.assertEqual(payload["colibri"], {"truncated": [1]})
        self.assertEqual(payload["usage"]["total_tokens"], 12)           # every pair was read, the answer keeps two


class EngineClientTest(unittest.TestCase):
    def engine(self, **env):
        engine = RerankEngine(STUB, "model-dir", env=stub_env(**env), on_log=lambda _m: None)
        self.addCleanup(engine.close)
        return engine

    def test_ready_tells_what_the_engine_does(self):
        engine = self.engine(RERANK_STUB_MAX_TOKENS=40)
        self.assertEqual((engine.info["max_tokens"], engine.info["max_documents"]), (40, 2048))
        self.assertTrue(engine.alive)
        self.assertEqual((engine.modality, engine.chats, engine.decides), ("rerank", False, False))
        self.assertNotIn("dimensions", engine.info)

    def test_a_batch_keeps_its_order_and_each_score_is_a_probability(self):
        engine = self.engine(RERANK_STUB_MAX_TOKENS=100)
        result = engine.rerank(QUERY, DOCS)
        self.assertEqual(len(result["scores"]), 4)
        self.assertTrue(all(0 < score < 1 for score in result["scores"]))
        self.assertGreater(result["scores"][1], result["scores"][0])
        self.assertEqual(result["tokens"], [3 + 4 + 2, 3 + 6 + 2, 3 + 3 + 2, 3 + 3 + 2])
        alone = engine.rerank(QUERY, [DOCS[2]])
        self.assertEqual(alone["scores"][0], result["scores"][2])
        self.assertEqual(len(result["logits"]), 4)

    def test_a_long_pair_is_cut_and_says_so_unless_told_to_refuse(self):
        engine = self.engine(RERANK_STUB_MAX_TOKENS=9)
        result = engine.rerank(QUERY, ["one two three four five six seven", "short"])
        self.assertEqual((result["truncated"], result["tokens"]), ([True, False], [9, 3 + 1 + 2]))
        with self.assertRaises(EmbeddingEngineError) as caught:
            engine.rerank(QUERY, ["one two three four five six seven"], truncate=False)
        self.assertIn("longer than 9 tokens", str(caught.exception))
        self.assertTrue(engine.alive)
        self.assertEqual(len(engine.rerank(QUERY, ["fine"])["scores"]), 1)

    def test_an_error_from_the_engine_leaves_it_alive(self):
        engine = self.engine(RERANK_STUB_FAIL_ON="bad")
        with self.assertRaises(EmbeddingEngineError) as caught:
            engine.rerank(QUERY, ["good", "bad"])
        self.assertIn("could not be scored", str(caught.exception))
        self.assertTrue(engine.alive)
        self.assertEqual(len(engine.rerank(QUERY, ["good"])["scores"]), 1)

    def test_a_dead_engine_says_why_and_names_itself_properly(self):
        engine = self.engine(RERANK_STUB_DIE_ON="boom")
        with self.assertRaises(EmbeddingEngineError) as caught:
            engine.rerank(QUERY, ["boom"])
        self.assertIn("reranking engine exited", str(caught.exception))
        self.assertIn("exit code 3", str(caught.exception))
        self.assertFalse(engine.alive)

    def test_an_embedding_engine_does_not_answer_a_rerank_and_the_other_way_round(self):
        embed = EmbeddingEngine(EMBED_STUB, "model-dir", env=stub_env(), on_log=lambda _m: None)
        self.addCleanup(embed.close)
        with self.assertRaises(EmbeddingEngineError):
            embed._exchange("RERANK", "SCORES", {"query": "q", "documents": ["a"]})      # the stub says "unknown message"
        rerank = self.engine()
        with self.assertRaises(EmbeddingEngineError):
            rerank._exchange("EMBED", "VECTORS", {"inputs": ["a"]})


class GatewayHarness:
    def __init__(self, max_queue=8, **env):
        self.engine = RerankEngine(STUB, "model-dir", env=stub_env(**env), on_log=lambda _m: None)
        self.server = APIServer(("127.0.0.1", 0), self.engine, MODEL_ID, KEY, 16, max_queue=max_queue, queue_timeout=30)
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def close(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.engine.close()

    def request(self, path, body=None, key=KEY, method=None):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = None if body is None else json.dumps(body).encode()
        req = Request(f"http://127.0.0.1:{self.port}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=30) as response:
                return response.status, dict(response.headers), json.loads(response.read() or b"{}")
        except HTTPError as error:
            return error.code, dict(error.headers), json.loads(error.read() or b"{}")

    def rerank(self, path="/v1/rerank", **body):
        body.setdefault("model", MODEL_ID)
        return self.request(path, body)


class EndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness(RERANK_STUB_MAX_TOKENS=14)

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def test_documents_come_back_best_first_in_the_shape_clients_expect(self):
        status, headers, body = self.gw.rerank(query=QUERY, documents=DOCS)
        self.assertEqual(status, 200, body)
        self.assertEqual((body["object"], body["model"]), ("rerank", MODEL_ID))
        self.assertTrue(body["id"].startswith("req_"))
        self.assertEqual([r["index"] for r in body["results"]], [1, 3, 2, 0])
        scores = [r["relevance_score"] for r in body["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(0 <= score <= 1 for score in scores))
        self.assertEqual(body["usage"]["total_tokens"], 9 + 11 + 8 + 8)
        self.assertEqual(body["usage"]["prompt_tokens"], body["usage"]["total_tokens"])
        self.assertIn("x-colibri-queue-wait-ms", {k.lower() for k in headers})
        self.assertNotIn("document", body["results"][0])

    def test_cohere_has_its_own_path_and_it_is_the_same_endpoint(self):
        status, _, body = self.gw.rerank(path="/v2/rerank", query=QUERY, documents=DOCS)
        self.assertEqual(status, 200, body)
        self.assertEqual([r["index"] for r in body["results"]], [1, 3, 2, 0])

    def test_top_n_keeps_the_best_and_return_documents_echoes_the_text(self):
        status, _, body = self.gw.rerank(query=QUERY, documents=DOCS, top_n=2, return_documents=True)
        self.assertEqual(status, 200, body)
        self.assertEqual([(r["index"], r["document"]["text"]) for r in body["results"]],
                         [(1, DOCS[1]), (3, DOCS[3])])
        self.assertEqual(body["usage"]["total_tokens"], 9 + 11 + 8 + 8)      # all four were scored

    def test_documents_given_as_objects_are_scored_as_their_text(self):
        status, _, body = self.gw.rerank(query=QUERY, documents=[{"text": d} for d in DOCS])
        self.assertEqual(status, 200, body)
        self.assertEqual([r["index"] for r in body["results"]], [1, 3, 2, 0])

    def test_a_long_pair_is_cut_and_reported(self):
        status, _, body = self.gw.rerank(query=QUERY, documents=["a b c d e f g h i j k l", "ok"])
        self.assertEqual(status, 200, body)
        self.assertEqual(body["colibri"], {"truncated": [0]})

    def test_truncate_false_refuses_a_long_pair_with_a_400(self):
        status, _, body = self.gw.rerank(query=QUERY, documents=["a b c d e f g h i j k l"], truncate=False)
        self.assertEqual(status, 400, body)
        self.assertEqual(body["error"]["param"], "documents")
        self.assertIn("longer than 14 tokens", body["error"]["message"])

    def test_bad_requests_are_a_400_naming_the_field(self):
        cases = (({}, "query"), ({"query": "q"}, "documents"), ({"query": "q", "documents": []}, "documents"),
                 ({"query": "q", "documents": [1]}, "documents"), ({"query": "q", "documents": ["a"], "top_n": 0}, "top_n"),
                 ({"query": "q", "documents": ["a"], "truncate": 1}, "truncate"))
        for body, param in cases:
            with self.subTest(body=body):
                status, _, answer = self.gw.rerank(**body)
                self.assertEqual((status, answer["error"]["param"]), (400, param), answer)

    def test_the_key_and_the_model_are_checked(self):
        self.assertEqual(self.gw.request("/v1/rerank", {"model": MODEL_ID, "query": "q", "documents": ["a"]}, key=None)[0], 401)
        status, _, body = self.gw.rerank(model="some-other-model", query="q", documents=["a"])
        self.assertEqual((status, body["error"]["code"]), (404, "model_not_found"))

    def test_the_model_card_and_health_say_what_it_does(self):
        status, _, models = self.gw.request("/v1/models")
        self.assertEqual(status, 200)
        card = models["data"][0]
        self.assertEqual((card["id"], card["capabilities"]), (MODEL_ID, ["rerank"]))
        self.assertEqual(card["rerank"], {"max_tokens": 14, "max_documents": 2048, "device": "cpu"})       # where the layers run, from READY
        status, _, health = self.gw.request("/health")
        self.assertEqual((status, health["capabilities"]), (200, ["rerank"]))
        self.assertEqual(health["rerank"]["max_tokens"], 14)

    def test_the_other_endpoints_say_this_model_only_ranks(self):
        for path, body in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
                           ("/v1/completions", {"prompt": "hi"})):
            with self.subTest(path=path):
                status, _, answer = self.gw.request(path, {"model": MODEL_ID, **body})
                self.assertEqual(status, 400, answer)
                self.assertIn("POST /v1/rerank", answer["error"]["message"])
                self.assertEqual(answer["error"]["code"], "unsupported_endpoint")
        for path, body in (("/v1/images/generations", {"prompt": "x"}), ("/v1/audio/music", {"prompt": "x"}),
                           ("/v1/embeddings", {"input": "x"})):
            with self.subTest(path=path):
                status, _, answer = self.gw.request(path, {"model": MODEL_ID, **body})
                self.assertEqual(status, 400, answer)
                self.assertIn("it ranks documents (POST /v1/rerank)", answer["error"]["message"])

    def test_requests_at_the_same_time_each_get_their_own_answer(self):
        results = {}

        def work(i):
            results[i] = self.gw.rerank(query=f"word{i}", documents=[f"word{i} here", "nothing"], top_n=1)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual({status for status, _, _ in results.values()}, {200})
        self.assertEqual({body["results"][0]["index"] for _, _, body in results.values()}, {0})


class DeadEngineTest(unittest.TestCase):
    def test_a_dead_engine_is_a_503_not_a_hang(self):
        gw = GatewayHarness(RERANK_STUB_DIE_ON="boom")
        try:
            status, _, body = gw.rerank(query="q", documents=["boom"])
            self.assertEqual(status, 500, body)
            self.assertIn("reranking engine failed", body["error"]["message"])
            status, _, body = gw.rerank(query="q", documents=["after"])
            self.assertEqual((status, body["error"]["code"]), (503, "engine_unavailable"))
        finally:
            gw.close()


class OtherModelsRefuseTest(unittest.TestCase):
    """A server that fronts an embedding engine answers /v1/rerank with a pointer, and the reverse."""

    def test_an_embedding_engine_does_not_rank(self):
        engine = EmbeddingEngine(EMBED_STUB, "model-dir", env=stub_env(), on_log=lambda _m: None)
        server = APIServer(("127.0.0.1", 0), engine, "qwen3-embedding-0.6b", KEY, 16, max_queue=4, queue_timeout=10)
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        try:
            req = Request(f"http://127.0.0.1:{server.server_port}/v1/rerank",
                          data=json.dumps({"model": "qwen3-embedding-0.6b", "query": "q", "documents": ["a"]}).encode(),
                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(req, timeout=30)
            self.assertEqual(caught.exception.code, 400)
            self.assertIn("does not rank documents: it makes embeddings", json.loads(caught.exception.read())["error"]["message"])
        finally:
            server.scheduler.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            engine.close(timeout=5)


class FamilyTest(unittest.TestCase):
    def checkpoint(self, root, hidden, layers, logit_score=True, pooling=False):
        root = Path(root)
        root.mkdir(parents=True)
        (root / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                                                      "hidden_size": hidden, "num_hidden_layers": layers}))
        (root / "modules.json").write_text("[]")
        if logit_score:
            (root / "1_LogitScore").mkdir()
            (root / "1_LogitScore" / "config.json").write_text(json.dumps({"true_token_id": 9693, "false_token_id": 2152}))
        if pooling:
            (root / "1_Pooling").mkdir()
            (root / "1_Pooling" / "config.json").write_text("{}")
        return root

    def test_the_reranker_checkpoints_are_told_apart_by_their_logit_score_file(self):
        for hidden, layers, model_id, scale in ((1024, 28, "qwen3-reranker-0.6b", "0.6B"),
                                                (2560, 36, "qwen3-reranker-4b", "4B"),
                                                (4096, 36, "qwen3-reranker-8b", "8B")):
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as tmp:
                resolved = resolve_model(self.checkpoint(Path(tmp) / "m", hidden, layers))
                family = resolved.descriptor
                self.assertEqual((family.id, family.modality, family.engine_artifact), ("qwen3_reranker", "rerank", "qwen3embed"))
                self.assertEqual(default_model_id(resolved), model_id)
                self.assertEqual(display_for(resolved), ("Qwen3-Reranker", scale))

    def test_a_pooling_checkpoint_is_still_an_embedding_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_model(self.checkpoint(Path(tmp) / "m", 1024, 28, logit_score=False, pooling=True))
            self.assertEqual(resolved.descriptor.id, "qwen3_embedding")

    def test_a_plain_qwen3_config_is_not_a_reranker(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnknownFamilyError):
                resolve_model(self.checkpoint(Path(tmp) / "m", 1024, 28, logit_score=False))

    def test_the_family_is_a_registered_one_request_at_a_time_engine(self):
        family = family_by_id("qwen3_reranker")
        self.assertEqual((family.limits.max_kv_slots, family.has_gateway_adapter, family.supports_accelerator), (1, True, False))
        self.assertFalse(family.capabilities.decision)


if __name__ == "__main__":
    unittest.main()
