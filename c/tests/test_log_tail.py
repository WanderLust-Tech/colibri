"""The in-memory tail of the router's and its workers' output (`c/log_tail.py`)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import log_tail  # noqa: E402


class LogTailTest(unittest.TestCase):
    def test_lines_are_numbered_and_read_from_a_number_on(self):
        tail = log_tail.LogTail()
        for i in range(5):
            tail.add("router", f"line {i}", now=100.0 + i)
        everything = tail.read()
        self.assertEqual([l["text"] for l in everything["data"]], [f"line {i}" for i in range(5)])
        self.assertEqual([l["seq"] for l in everything["data"]], [1, 2, 3, 4, 5])
        self.assertEqual((everything["next"], everything["first"]), (5, 1))
        after = tail.read(since=3)
        self.assertEqual([l["text"] for l in after["data"]], ["line 3", "line 4"])
        self.assertEqual(tail.read(since=5)["data"], [], "nothing new is an empty list, and next stays")
        self.assertEqual(tail.read(since=5)["next"], 5)

    def test_an_empty_tail_says_where_it_starts(self):
        out = log_tail.LogTail().read()
        self.assertEqual((out["data"], out["next"], out["first"]), ([], 0, 1))

    def test_only_the_newest_lines_are_kept_and_first_shows_the_gap(self):
        tail = log_tail.LogTail(maxlen=3)
        for i in range(10):
            tail.add("router", f"line {i}")
        out = tail.read(since=2)
        self.assertEqual([l["text"] for l in out["data"]], ["line 7", "line 8", "line 9"])
        self.assertEqual((out["first"], out["next"]), (8, 10))
        self.assertGreater(out["first"], 2 + 1, "a client that last saw 2 can tell that lines were lost")

    def test_limit_keeps_the_newest(self):
        tail = log_tail.LogTail()
        for i in range(10):
            tail.add("router", f"line {i}")
        self.assertEqual([l["text"] for l in tail.read(limit=2)["data"]], ["line 8", "line 9"])

    def test_a_source_can_be_asked_for_alone(self):
        tail = log_tail.LogTail()
        tail.add("router", "a")
        tail.add("qwen-colibri", "b")
        tail.add("router", "c")
        self.assertEqual([l["text"] for l in tail.read(source="qwen-colibri")["data"]], ["b"])
        self.assertEqual(tail.read(source="qwen-colibri")["next"], 3)

    def test_a_very_long_line_is_cut(self):
        tail = log_tail.LogTail()
        tail.add("router", "x" * (log_tail.LINE_LIMIT + 500))
        text = tail.read()["data"][0]["text"]
        self.assertEqual(len(text), log_tail.LINE_LIMIT + len("...[cut]"))
        self.assertTrue(text.endswith("...[cut]"))


if __name__ == "__main__":
    unittest.main()
