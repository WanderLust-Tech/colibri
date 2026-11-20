"""`coli embed` and `coli rerank` (retrieval_cli.py): what a person sees, against a stub of the server's API."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import retrieval_cli  # noqa: E402

NO_COLOR = SimpleNamespace(dim="", r="", grn="", yel="")


def embed_args(texts=(), **kw):
    base = dict(texts=list(texts), file=None, model=None, dimensions=None, instruction=None, json=False)
    base.update(kw)
    return SimpleNamespace(**base)


def rerank_args(query="q", documents=(), **kw):
    base = dict(query=query, documents=list(documents), file=None, model=None, top=None, instruction=None, truncate=False, json=False)
    base.update(kw)
    return SimpleNamespace(**base)


class Server:
    """The routes the commands use: /v1/models, /v1/embeddings, /v1/rerank."""

    def __init__(self, models=None, status=200, answer=None, backwards=False):
        self.backwards = backwards
        self.models = models if models is not None else [
            {"id": "chat-model", "capabilities": ["chat"]},
            {"id": "emb-model", "capabilities": ["embeddings"]},
            {"id": "rank-model", "capabilities": ["rerank"]}]
        self.status, self.answer, self.calls = status, answer, []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if self.status in (None, 401) and path == "/v1/models":
            return self.status, None
        if path == "/v1/models":
            return 200, {"data": self.models}
        if self.status != 200:
            return self.status, self.answer
        if path == "/v1/embeddings":
            vectors = {"cat": [1.0, 0.0], "kitten": [0.6, 0.8], "car": [0.0, 1.0]}
            rows = [{"index": i, "embedding": vectors.get(text, [0.0, 0.0])} for i, text in enumerate(body["input"])]
            return 200, {"object": "list", "model": body["model"], "data": rows[::-1] if self.backwards else rows, "usage": {"prompt_tokens": 9, "total_tokens": 9}}
        if path == "/v1/rerank":
            scores = {"a": 0.2, "b": 0.9, "c": 0.5}
            order = sorted(range(len(body["documents"])), key=lambda i: -scores.get(body["documents"][i], 0.0))
            if body.get("top_n"):
                order = order[:body["top_n"]]
            result = {"model": body["model"], "results": [{"index": i, "relevance_score": scores.get(body["documents"][i], 0.0)} for i in order],
                      "usage": {"prompt_tokens": 30, "total_tokens": 30}}
            if body.get("truncate"):
                result["colibri"] = {"truncated": [0]}
            return 200, result
        raise AssertionError(path)


def run(function, args, api):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = function(args, api, NO_COLOR)
    return code, out.getvalue()


class EmbedTest(unittest.TestCase):
    def test_it_picks_the_embedding_model_and_shows_how_alike_the_texts_are(self):
        server = Server()
        code, text = run(retrieval_cli.embed, embed_args(["cat", "kitten", "car"]), server)
        self.assertEqual(code, 0)
        post = [c for c in server.calls if c[0] == "POST"][0]
        self.assertEqual(post[2]["model"], "emb-model")
        self.assertEqual(post[2]["input"], ["cat", "kitten", "car"])
        self.assertIn("3 documents, 2 numbers each, 9 tokens", text)
        self.assertIn("[0]    1.00   0.60   0.00", text)                      # cat against kitten is 0.6, against car 0
        self.assertIn("[2]    0.00   0.80   1.00", text)

    def test_vectors_are_matched_to_their_texts_by_index_not_by_order_of_arrival(self):
        code, text = run(retrieval_cli.embed, embed_args(["cat", "car"]), Server(backwards=True))
        self.assertEqual(code, 0)
        self.assertIn("[0] +1.000 +0.000 ...   cat", text)
        self.assertIn("[1] +0.000 +1.000 ...   car", text)

    def test_options_reach_the_request_and_instruction_means_queries(self):
        server = Server()
        code, text = run(retrieval_cli.embed, embed_args(["cat"], model="mine", dimensions=64, instruction="find pets"), server)
        self.assertEqual(code, 0)
        body = server.calls[-1][2]
        self.assertEqual((body["model"], body["dimensions"], body["instruction"]), ("mine", 64, "find pets"))
        self.assertEqual([c[1] for c in server.calls], ["/v1/embeddings"])      # a named model is not looked up
        self.assertIn("1 queries", text)
        self.assertNotIn("how alike", text)                                      # one text has nothing to compare with

    def test_json_prints_the_answer_only(self):
        code, text = run(retrieval_cli.embed, embed_args(["cat"], json=True), Server())
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(text)["data"][0]["embedding"], [1.0, 0.0])

    def test_texts_come_from_a_file_and_standard_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "t.txt")
            Path(path).write_text("cat\n\nkitten\n", encoding="utf-8")
            server = Server()
            self.assertEqual(run(retrieval_cli.embed, embed_args(["car"], file=path), server)[0], 0)
            self.assertEqual(server.calls[-1][2]["input"], ["car", "cat", "kitten"])
            with mock.patch("sys.stdin", io.StringIO("cat\ncar\n")):
                server = Server()
                self.assertEqual(run(retrieval_cli.embed, embed_args(["-"]), server)[0], 0)
                self.assertEqual(server.calls[-1][2]["input"], ["cat", "car"])

    def test_what_goes_wrong_is_one_plain_line_and_a_failing_code(self):
        for server, words in ((Server(models=[{"id": "chat-model", "capabilities": ["chat"]}]), "coli model install qwen3-embedding-0.6b"),
                              (Server(status=None), "nothing answers"), (Server(status=401), "API key")):
            with self.subTest(words=words):
                code, text = run(retrieval_cli.embed, embed_args(["cat"]), server)
                self.assertEqual(code, 1)
                self.assertIn(words, text)
                self.assertFalse([c for c in server.calls if c[0] == "POST"])
        code, text = run(retrieval_cli.embed, embed_args(["cat"], model="x"), Server(status=400, answer={"error": {"message": "dimensions is too big"}}))
        self.assertEqual(code, 1)
        self.assertIn("the server said 400: dimensions is too big", text)
        self.assertEqual(run(retrieval_cli.embed, embed_args([]), Server())[0], 1)          # no text: the usage line


class RerankTest(unittest.TestCase):
    def test_it_ranks_best_first_and_shows_the_position_in_the_input(self):
        server = Server()
        code, text = run(retrieval_cli.rerank, rerank_args("q", ["a", "b", "c"]), server)
        self.assertEqual(code, 0)
        post = [c for c in server.calls if c[0] == "POST"][0]
        self.assertEqual((post[2]["model"], post[2]["query"], post[2]["documents"]), ("rank-model", "q", ["a", "b", "c"]))
        lines = [line for line in text.splitlines() if "[" in line and "0." in line]
        self.assertEqual([line.split("[")[1][0] for line in lines], ["1", "2", "0"])      # b, c, a
        self.assertIn("0.900", lines[0])
        self.assertIn("3 documents ranked, 30 tokens", text)

    def test_top_instruction_and_truncate_reach_the_request(self):
        server = Server()
        code, text = run(retrieval_cli.rerank, rerank_args("q", ["a", "b", "c"], top=2, instruction="be strict", truncate=True), server)
        self.assertEqual(code, 0)
        body = server.calls[-1][2]
        self.assertEqual((body["top_n"], body["instruction"], body["truncate"]), (2, "be strict", True))
        self.assertEqual(len([line for line in text.splitlines() if "0." in line and "[" in line]), 2)
        self.assertIn("cut to the model's limit: 0", text)

    def test_documents_from_standard_input_and_a_long_one_is_shortened_on_screen_only(self):
        long = "word " * 100
        with mock.patch("sys.stdin", io.StringIO(long + "\nb\n")):
            server = Server()
            code, text = run(retrieval_cli.rerank, rerank_args("q", ["-"]), server)
        self.assertEqual(code, 0)
        self.assertEqual(server.calls[-1][2]["documents"], [long.rstrip("\n"), "b"])
        self.assertIn("…", text)
        self.assertNotIn("word " * 30, text)

    def test_json_and_the_refusals(self):
        code, text = run(retrieval_cli.rerank, rerank_args("q", ["b"], json=True), Server())
        self.assertEqual((code, json.loads(text)["results"][0]["index"]), (0, 0))
        self.assertEqual(run(retrieval_cli.rerank, rerank_args(None, ["b"]), Server())[0], 1)      # no query
        code, text = run(retrieval_cli.rerank, rerank_args("q", []), Server())
        self.assertEqual(code, 1)
        self.assertIn("no documents", text)
        code, text = run(retrieval_cli.rerank, rerank_args("q", ["a"]), Server(models=[{"id": "m", "capabilities": ["chat"]}]))
        self.assertEqual(code, 1)
        self.assertIn("coli model install qwen3-reranker-0.6b", text)
        code, text = run(retrieval_cli.rerank, rerank_args("q", ["a"], model="m"), Server(status=503, answer={"error": {"message": "engine is down"}}))
        self.assertEqual(code, 1)
        self.assertIn("the server said 503: engine is down", text)

    def test_a_missing_file_is_a_plain_refusal(self):
        code, text = run(retrieval_cli.rerank, rerank_args("q", [], file="no-such-file.txt"), Server())
        self.assertEqual(code, 1)
        self.assertIn("cannot read no-such-file.txt", text)


if __name__ == "__main__":
    unittest.main()
