"""POST /v1/embeddings, end to end against the stub engine, and the rules around it.

A real EmbeddingEngine drives tools/embedding_stub.py over its serve protocol, and a real APIServer answers HTTP on a free
port: nothing between the socket and the engine's stdin is mocked. The stub's vector is a pure function of the text (and the
instruction), which is what lets these tests pin the order of a batch, the cut to fewer dimensions and the accounting.
"""
import base64
import json
import math
import os
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embedding_engine  # noqa: E402
from embedding_engine import (EmbeddingEngine, EmbeddingEngineError, EmbeddingRequestError,  # noqa: E402
                              embedding_payload, embedding_request, pack_vector)
from family_registry import (UnknownFamilyError, default_model_id, display_for, family_by_id,  # noqa: E402
                             resolve_model)
from openai_server import APIServer  # noqa: E402

STUB = str(Path(__file__).resolve().parent.parent / "tools" / "embedding_stub.py")
MODEL_ID = "qwen3-embedding-0.6b"
KEY = "secret"


def stub_env(**extra):
    env = dict(os.environ)
    env.update({key: str(value) for key, value in extra.items()})
    return env


def norm(vector):
    return math.sqrt(sum(v * v for v in vector))


class RequestRulesTest(unittest.TestCase):
    INFO = {"dimensions": 16, "min_dimensions": 1, "max_tokens": 8}

    def check(self, body):
        return embedding_request(body, self.INFO)

    def refused(self, body, param, fragment=""):
        with self.assertRaises(EmbeddingRequestError) as caught:
            self.check(body)
        self.assertEqual(caught.exception.param, param, str(caught.exception))
        self.assertIn(fragment, str(caught.exception))

    def test_a_string_is_one_input_and_a_list_is_many(self):
        self.assertEqual(self.check({"input": "a"})["inputs"], ["a"])
        self.assertEqual(self.check({"input": ["a", "b"]})["inputs"], ["a", "b"])

    def test_the_defaults(self):
        request = self.check({"input": "a"})
        self.assertEqual((request["encoding_format"], request["dimensions"], request["instruction"], request["truncate"]),
                         ("float", None, None, True))

    def test_an_empty_string_is_an_input_like_any_other(self):
        self.assertEqual(self.check({"input": ""})["inputs"], [""])

    def test_the_whole_request_is_checked(self):
        full = self.check({"input": ["a"], "encoding_format": "base64", "dimensions": 4, "instruction": "find it",
                           "truncate": False, "user": "me", "model": "x"})
        self.assertEqual((full["encoding_format"], full["dimensions"], full["instruction"], full["truncate"]),
                         ("base64", 4, "find it", False))

    def test_input_is_required_and_must_be_text(self):
        self.refused({}, "input", "required")
        self.refused({"input": []}, "input", "non-empty")
        self.refused({"input": 5}, "input")
        self.refused({"input": [1, 2, 3]}, "input", "token arrays")
        self.refused({"input": [[1, 2]]}, "input", "token arrays")
        self.refused({"input": ["a", None]}, "input", "string")
        self.refused({"input": ["a", True]}, "input", "string")
        self.refused({"input": ["a"] * (embedding_engine.MAX_INPUTS + 1)}, "input", "at most")

    def test_dimensions_are_bounded_by_the_model(self):
        self.check({"input": "a", "dimensions": 1})
        self.check({"input": "a", "dimensions": 16})
        self.refused({"input": "a", "dimensions": 0}, "dimensions", "between 1 and 16")
        self.refused({"input": "a", "dimensions": 17}, "dimensions", "between 1 and 16")
        self.refused({"input": "a", "dimensions": "8"}, "dimensions", "integer")
        self.refused({"input": "a", "dimensions": True}, "dimensions", "integer")
        self.refused({"input": "a", "dimensions": 1.5}, "dimensions", "integer")

    def test_the_other_fields(self):
        self.refused({"input": "a", "encoding_format": "hex"}, "encoding_format")
        self.refused({"input": "a", "instruction": 3}, "instruction")
        self.refused({"input": "a", "instruction": "x" * (embedding_engine.MAX_INSTRUCTION_BYTES + 1)}, "instruction", "over")
        self.refused({"input": "a", "truncate": "yes"}, "truncate")
        self.refused({"input": "a", "user": 7}, "user")
        with self.assertRaises(EmbeddingRequestError):
            embedding_request([], self.INFO)

    def test_base64_is_little_endian_float32(self):
        vector = [0.5, -1.25, 3.0]
        raw = base64.b64decode(pack_vector(vector))
        self.assertEqual(struct.unpack("<3f", raw), tuple(vector))

    def test_the_payload_has_openais_shape(self):
        request = {"encoding_format": "float"}
        result = {"vectors": [[1.0, 0.0], [0.0, 1.0]], "tokens": [3, 4], "truncated": [False, True], "dimensions": 2}
        payload = embedding_payload(request, result, "m")
        self.assertEqual(payload["object"], "list")
        self.assertEqual([(d["object"], d["index"], d["embedding"]) for d in payload["data"]],
                         [("embedding", 0, [1.0, 0.0]), ("embedding", 1, [0.0, 1.0])])
        self.assertEqual(payload["usage"], {"prompt_tokens": 7, "total_tokens": 7})
        self.assertEqual(payload["colibri"], {"truncated": [1]})
        result["truncated"] = [False, False]
        self.assertNotIn("colibri", embedding_payload(request, result, "m"))


class EngineClientTest(unittest.TestCase):
    def engine(self, **env):
        engine = EmbeddingEngine(STUB, "model-dir", env=stub_env(**env), on_log=lambda _m: None)
        self.addCleanup(engine.close)
        return engine

    def test_ready_tells_what_the_engine_makes(self):
        engine = self.engine(EMBED_STUB_DIM=32, EMBED_STUB_MAX_TOKENS=100)
        self.assertEqual((engine.info["dimensions"], engine.info["max_tokens"], engine.dimensions), (32, 100, 32))
        self.assertTrue(engine.alive)
        self.assertEqual((engine.modality, engine.chats, engine.decides), ("embedding", False, False))

    def test_a_batch_keeps_its_order_and_each_vector_is_a_unit_vector(self):
        engine = self.engine()
        first = engine.embed(["alpha", "beta", "gamma"])
        again = engine.embed(["gamma", "alpha"])
        self.assertEqual(len(first["vectors"]), 3)
        for vector in first["vectors"]:
            self.assertAlmostEqual(norm(vector), 1.0, places=6)
        self.assertEqual(again["vectors"][0], first["vectors"][2])
        self.assertEqual(again["vectors"][1], first["vectors"][0])
        self.assertNotEqual(first["vectors"][0], first["vectors"][1])
        self.assertEqual(first["tokens"], [2, 2, 2])

    def test_dimensions_keep_the_first_components_and_normalise_again(self):
        engine = self.engine()
        whole = engine.embed(["alpha"])["vectors"][0]
        cut = engine.embed(["alpha"], dimensions=4)
        self.assertEqual(cut["dimensions"], 4)
        self.assertEqual(len(cut["vectors"][0]), 4)
        self.assertAlmostEqual(norm(cut["vectors"][0]), 1.0, places=6)
        scale = norm(whole[:4])
        for got, want in zip(cut["vectors"][0], whole[:4]):
            self.assertAlmostEqual(got, want / scale, places=6)

    def test_an_instruction_makes_another_vector(self):
        engine = self.engine()
        plain = engine.embed(["alpha"])["vectors"][0]
        asked = engine.embed(["alpha"], instruction="find the answer")["vectors"][0]
        self.assertNotEqual(plain, asked)

    def test_a_long_input_is_cut_and_says_so_unless_told_to_refuse(self):
        engine = self.engine(EMBED_STUB_MAX_TOKENS=4)
        result = engine.embed(["one two three four five six", "short"])
        self.assertEqual((result["truncated"], result["tokens"]), ([True, False], [4, 2]))
        with self.assertRaises(EmbeddingEngineError) as caught:
            engine.embed(["one two three four five six"], truncate=False)
        self.assertIn("longer than 4 tokens", str(caught.exception))
        self.assertTrue(engine.alive)                       # a refusal is not a crash
        self.assertEqual(len(engine.embed(["fine"])["vectors"]), 1)

    def test_an_error_from_the_engine_leaves_it_alive(self):
        engine = self.engine(EMBED_STUB_FAIL_ON="bad")
        with self.assertRaises(EmbeddingEngineError) as caught:
            engine.embed(["good", "bad"])
        self.assertIn("could not be embedded", str(caught.exception))
        self.assertTrue(engine.alive)
        self.assertEqual(len(engine.embed(["good"])["vectors"]), 1)

    def test_a_dead_engine_says_why_and_stays_dead(self):
        engine = self.engine(EMBED_STUB_DIE_ON="boom")
        with self.assertRaises(EmbeddingEngineError) as caught:
            engine.embed(["boom"])
        self.assertIn("exit code 3", str(caught.exception))
        self.assertFalse(engine.alive)
        with self.assertRaises(EmbeddingEngineError):
            engine.embed(["again"])

    def test_requests_from_several_threads_are_answered_one_by_one(self):
        engine = self.engine(EMBED_STUB_DELAY=0.02)
        answers = {}

        def work(name):
            answers[name] = engine.embed([name])["vectors"][0]

        threads = [threading.Thread(target=work, args=(f"text{i}",)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual(len(answers), 8)
        alone = self.engine()
        for name, vector in answers.items():
            self.assertEqual(vector, alone.embed([name])["vectors"][0], name)

    def test_close_ends_an_engine_and_kills_one_that_will_not_go(self):
        quick = EmbeddingEngine(STUB, "model-dir", env=stub_env(), on_log=lambda _m: None)
        quick.close()
        self.assertFalse(quick.alive)
        quick.close()                                       # and again: nothing to do
        stuck = EmbeddingEngine(STUB, "model-dir", env=stub_env(EMBED_STUB_DELAY=30), on_log=lambda _m: None)
        worker = threading.Thread(target=lambda: self.assertRaises(EmbeddingEngineError, stuck.embed, ["slow"]), daemon=True)
        worker.start()
        started = threading.Event()
        threading.Timer(0.3, started.set).start()
        started.wait()
        stuck.close(timeout=0.5)                            # it is in the middle of a request: it does not read EOF in time
        worker.join(timeout=20)
        self.assertFalse(worker.is_alive())
        self.assertFalse(stuck.alive)

    def test_an_engine_that_cannot_start_is_an_error_not_a_hang(self):
        with self.assertRaises(EmbeddingEngineError):
            EmbeddingEngine(str(Path(STUB).with_name("no_such_engine.py")), "model-dir", on_log=lambda _m: None, load_timeout=20)


class GatewayHarness:
    def __init__(self, max_queue=8, **env):
        self.engine = EmbeddingEngine(STUB, "model-dir", env=stub_env(**env), on_log=lambda _m: None)
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

    def embed(self, **body):
        body.setdefault("model", MODEL_ID)
        return self.request("/v1/embeddings", body)


class EndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness(EMBED_STUB_MAX_TOKENS=6)

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def test_one_text_comes_back_as_one_unit_vector_in_openais_shape(self):
        status, headers, body = self.gw.embed(input="hello world")
        self.assertEqual(status, 200, body)
        self.assertEqual((body["object"], body["model"]), ("list", MODEL_ID))
        self.assertEqual(len(body["data"]), 1)
        item = body["data"][0]
        self.assertEqual((item["object"], item["index"], len(item["embedding"])), ("embedding", 0, 16))
        self.assertAlmostEqual(norm(item["embedding"]), 1.0, places=6)
        self.assertEqual(body["usage"], {"prompt_tokens": 3, "total_tokens": 3})
        self.assertIn("x-colibri-queue-wait-ms", {k.lower() for k in headers})

    def test_a_batch_is_answered_in_order_with_an_index_each(self):
        status, _, body = self.gw.embed(input=["alpha", "beta gamma", "alpha"])
        self.assertEqual(status, 200, body)
        self.assertEqual([d["index"] for d in body["data"]], [0, 1, 2])
        self.assertEqual(body["data"][0]["embedding"], body["data"][2]["embedding"])
        self.assertNotEqual(body["data"][0]["embedding"], body["data"][1]["embedding"])
        self.assertEqual(body["usage"]["total_tokens"], 2 + 3 + 2)

    def test_dimensions_cut_the_vector(self):
        status, _, body = self.gw.embed(input="alpha", dimensions=5)
        self.assertEqual(status, 200, body)
        self.assertEqual(len(body["data"][0]["embedding"]), 5)
        self.assertAlmostEqual(norm(body["data"][0]["embedding"]), 1.0, places=6)

    def test_base64_decodes_to_the_float_answer(self):
        _, _, plain = self.gw.embed(input="alpha")
        status, _, packed = self.gw.embed(input="alpha", encoding_format="base64")
        self.assertEqual(status, 200, packed)
        raw = base64.b64decode(packed["data"][0]["embedding"])
        decoded = struct.unpack(f"<{len(raw) // 4}f", raw)
        for got, want in zip(decoded, plain["data"][0]["embedding"]):
            self.assertAlmostEqual(got, want, places=6)

    def test_an_instruction_makes_a_query(self):
        _, _, document = self.gw.embed(input="alpha")
        _, _, query = self.gw.embed(input="alpha", instruction="Retrieve passages")
        self.assertNotEqual(document["data"][0]["embedding"], query["data"][0]["embedding"])

    def test_a_long_input_is_cut_and_reported(self):
        status, _, body = self.gw.embed(input=["one two three four five six seven", "ok"])
        self.assertEqual(status, 200, body)
        self.assertEqual(body["colibri"], {"truncated": [0]})
        self.assertEqual(body["usage"]["prompt_tokens"], 6 + 2)

    def test_truncate_false_refuses_a_long_input_with_a_400(self):
        status, _, body = self.gw.embed(input="one two three four five six seven", truncate=False)
        self.assertEqual(status, 400, body)
        self.assertEqual(body["error"]["param"], "input")
        self.assertIn("longer than 6 tokens", body["error"]["message"])

    def test_bad_requests_are_a_400_naming_the_field(self):
        cases = (({}, "input"), ({"input": []}, "input"), ({"input": [1, 2]}, "input"), ({"input": "a", "dimensions": 99}, "dimensions"),
                 ({"input": "a", "encoding_format": "hex"}, "encoding_format"), ({"input": "a", "truncate": 1}, "truncate"))
        for body, param in cases:
            with self.subTest(body=body):
                status, _, answer = self.gw.embed(**body)
                self.assertEqual((status, answer["error"]["param"]), (400, param), answer)

    def test_the_key_and_the_model_are_checked(self):
        self.assertEqual(self.gw.request("/v1/embeddings", {"model": MODEL_ID, "input": "a"}, key=None)[0], 401)
        self.assertEqual(self.gw.request("/v1/embeddings", {"model": MODEL_ID, "input": "a"}, key="wrong")[0], 401)
        status, _, body = self.gw.embed(model="some-other-model", input="a")
        self.assertEqual((status, body["error"]["code"]), (404, "model_not_found"))

    def test_the_model_card_and_health_say_what_it_does(self):
        status, _, models = self.gw.request("/v1/models")
        self.assertEqual(status, 200)
        card = models["data"][0]
        self.assertEqual(card["id"], MODEL_ID)
        self.assertEqual(card["capabilities"], ["embeddings"])
        self.assertEqual(card["embedding"], {"dimensions": 16, "min_dimensions": 1, "max_tokens": 6})
        status, _, health = self.gw.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["capabilities"], ["embeddings"])
        self.assertEqual(health["embedding"]["dimensions"], 16)

    def test_the_other_endpoints_say_this_model_only_embeds(self):
        for path, body in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
                           ("/v1/completions", {"prompt": "hi"}), ("/v1/images/generations", {"prompt": "x"}),
                           ("/v1/audio/music", {"prompt": "x"})):
            with self.subTest(path=path):
                status, _, answer = self.gw.request(path, {"model": MODEL_ID, **body})
                self.assertEqual(status, 400, answer)
                self.assertIn("POST /v1/embeddings", answer["error"]["message"])
                self.assertEqual(answer["error"]["code"], "unsupported_endpoint")

    def test_requests_at_the_same_time_each_get_their_own_answer(self):
        results = {}

        def work(i):
            results[i] = self.gw.embed(input=f"text number {i}")

        threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        self.assertEqual({status for status, _, _ in results.values()}, {200})
        vectors = {tuple(body["data"][0]["embedding"]) for _, _, body in results.values()}
        self.assertEqual(len(vectors), 6)


class DeadEngineTest(unittest.TestCase):
    def test_a_dead_engine_is_a_503_not_a_hang(self):
        gw = GatewayHarness(EMBED_STUB_DIE_ON="boom")
        try:
            status, _, body = gw.embed(input="boom")
            self.assertEqual(status, 500, body)
            self.assertIn("embedding engine failed", body["error"]["message"])
            status, _, body = gw.embed(input="after")
            self.assertEqual((status, body["error"]["code"]), (503, "engine_unavailable"))
        finally:
            gw.close()


class OtherModelsRefuseTest(unittest.TestCase):
    """A server that fronts a chat or a music engine answers /v1/embeddings with a pointer, not a crash."""

    def test_a_music_engine_does_not_embed(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from acestep_fixture import STUB as MUSIC_STUB, make_fake_checkpoint  # noqa: E402
        from acestep_fixture import stub_env as music_env  # noqa: E402
        import music_engine
        with tempfile.TemporaryDirectory() as tmp:
            model = make_fake_checkpoint(Path(tmp) / "ace-step")
            engine = music_engine.MusicEngine(MUSIC_STUB, model, env=music_env(0.0), on_log=lambda _m: None)
            server = APIServer(("127.0.0.1", 0), engine, "ace-step-1.5-colibri", KEY, 16, max_queue=4, queue_timeout=10)
            thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
            thread.start()
            try:
                req = Request(f"http://127.0.0.1:{server.server_port}/v1/embeddings",
                              data=json.dumps({"model": "ace-step-1.5-colibri", "input": "a"}).encode(),
                              headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
                with self.assertRaises(HTTPError) as caught:
                    urlopen(req, timeout=30)
                self.assertEqual(caught.exception.code, 400)
                self.assertIn("makes music", json.loads(caught.exception.read())["error"]["message"])
            finally:
                server.scheduler.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                engine.close(timeout=5)


class FamilyTest(unittest.TestCase):
    def checkpoint(self, root, hidden, layers, sentence_transformers=True):
        root = Path(root)
        (root / "1_Pooling").mkdir(parents=True)
        (root / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                                                      "hidden_size": hidden, "num_hidden_layers": layers}))
        if sentence_transformers:
            (root / "modules.json").write_text("[]")
            (root / "1_Pooling" / "config.json").write_text(json.dumps({"pooling_mode_lasttoken": True}))
        return root

    def test_the_embedding_checkpoints_are_told_apart_by_their_sentence_transformers_files(self):
        for hidden, layers, model_id, scale in ((1024, 28, "qwen3-embedding-0.6b", "0.6B"),
                                                (2560, 36, "qwen3-embedding-4b", "4B"),
                                                (4096, 36, "qwen3-embedding-8b", "8B")):
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as tmp:
                resolved = resolve_model(self.checkpoint(Path(tmp) / "m", hidden, layers))
                family = resolved.descriptor
                self.assertEqual((family.id, family.modality, family.engine_artifact), ("qwen3_embedding", "embedding", "qwen3embed"))
                self.assertEqual(default_model_id(resolved), model_id)
                self.assertEqual(display_for(resolved), ("Qwen3-Embedding", scale))

    def test_a_plain_qwen3_config_is_not_an_embedding_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UnknownFamilyError):
                resolve_model(self.checkpoint(Path(tmp) / "m", 1024, 28, sentence_transformers=False))

    def test_the_family_is_a_registered_one_request_at_a_time_engine(self):
        family = family_by_id("qwen3_embedding")
        self.assertEqual((family.limits.max_kv_slots, family.has_gateway_adapter, family.supports_accelerator), (1, True, False))
        self.assertFalse(family.capabilities.decision)


if __name__ == "__main__":
    unittest.main()
