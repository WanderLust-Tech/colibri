"""`coli settings`: the settings index (c/settings_index.py) and the command.

    python -m unittest tests.test_settings_index
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import settings_index  # noqa: E402

TABLE = """# Environment

## Common

| Variable | Default | Effect |
|---|---|---|
| `CTX` | `4096` | Maximum context length. Second sentence. |
| `COLI_TEMP` / `TEMP` | `-1` (auto) | Sampling temperature. A choice: `quality` \\| `balanced`. |
| `COLI_API_KEY` | unset | Required bearer token. |
| `COLI_ONE` | `1` | One. |
| `COLI_TWO` | `2` | Two. |

## Engine

Prose, not a table row: | `NOT_A_ROW` | x | y |

| `Q36_MAXT` | engine default | Lower the context. |
| `Q36_EOS` | unset | End id. |
| `Q36_THIRD` | unset | Third. |
"""


class IndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.doc = Path(self.tmp.name) / "ENVIRONMENT.md"
        self.doc.write_text(TABLE, encoding="utf-8")
        self.listed = settings_index.settings(self.doc)

    def test_the_rows_of_the_tables_are_the_settings(self):
        names = [s.name for s in self.listed]
        self.assertEqual(names, ["CTX", "COLI_TEMP", "TEMP", "COLI_API_KEY", "COLI_ONE", "COLI_TWO",
                                 "Q36_MAXT", "Q36_EOS", "Q36_THIRD"])                       # a row of two names is two settings
        by = {s.name: s for s in self.listed}
        self.assertEqual((by["CTX"].default, by["CTX"].section), ("4096", "Common"))
        self.assertEqual(by["Q36_MAXT"].section, "Engine")
        self.assertIn("`quality` | `balanced`", by["COLI_TEMP"].effect)                        # an escaped pipe is a pipe
        self.assertEqual(by["CTX"].short(), "Maximum context length")
        self.assertNotIn("NOT_A_ROW", names)

    def test_the_real_documentation_has_a_long_list_and_the_settings_this_work_added(self):
        real = {s.name: s for s in settings_index.settings()}
        self.assertGreater(len(real), 300)
        for name in ("COLI_PRESETS", "COLI_REQUEST_LOG", "COLI_REQUEST_LOG_BODIES", "COLI_REQUEST_LOG_MAX", "COLI_KEEP_ALIVE",
                     "COLI_API_KEY", "KV_SLOTS", "CTX"):
            self.assertIn(name, real)
            self.assertTrue(real[name].default and real[name].effect, name)

    def test_what_is_set_is_told_from_what_nothing_reads(self):
        used, unknown = settings_index.in_effect(
            {"CTX": "8192", "COLI_API_KEY": "k", "COLI_TWOO": "1", "PATH": "x", "HOME": "y", "NOT_COLI_AT_ALL": "1",
             "Q36_MAXT": "1"}, self.listed)
        self.assertEqual([s.name for s, _ in used], ["COLI_API_KEY", "CTX", "Q36_MAXT"])
        self.assertEqual([name for name, _value, _close in unknown], ["COLI_TWOO"])        # only a known family is judged
        self.assertEqual(unknown[0][2][:1], ["COLI_TWO"])                                   # and the close name is offered

    def test_a_setting_the_source_reads_but_the_tables_lack_is_not_a_typo(self):
        # COLI_REQUEST_LOG is in the source (request_log.py) and, in this tiny table, not in the tables
        used, unknown = settings_index.in_effect({"COLI_REQUEST_LOG": "x.log"}, self.listed)
        self.assertEqual((used, unknown), ([], []))

    def test_search_wants_every_word_and_puts_name_matches_first(self):
        hits = settings_index.search(self.listed, ["context"])
        self.assertEqual([s.name for s in hits], ["CTX", "Q36_MAXT"])
        self.assertEqual([s.name for s in settings_index.search(self.listed, ["coli", "temp"])], ["COLI_TEMP"])
        self.assertEqual(settings_index.search(self.listed, ["no", "such", "thing"]), [])
        self.assertEqual([s.name for s in settings_index.search(self.listed, ["q36"])][:1], ["Q36_EOS"])

    def test_a_missing_document_is_no_settings(self):
        self.assertEqual(settings_index.settings(Path(self.tmp.name) / "nope.md"), [])


class CommandTest(unittest.TestCase):
    def run_coli(self, *args, env=None):
        full = dict(os.environ, PYTHONUTF8="1", COLI_COLOR="0")
        for name in [n for n in full if n.startswith(("COLI_", "Q36_", "G4_"))]:
            del full[name]
        full.update(env or {})
        return subprocess.run([sys.executable, str(HERE.parent / "coli"), "settings", *args], capture_output=True, text=True,
                              env=full, errors="replace", timeout=120)

    def test_it_lists_what_is_set_and_hides_keys(self):
        out = self.run_coli(env={"COLI_KV_SLOTS": "3", "COLI_API_KEY": "hunter2hunter2"})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("COLI_KV_SLOTS = 3", out.stdout)
        self.assertIn("COLI_API_KEY = (hidden)", out.stdout)
        self.assertNotIn("hunter2", out.stdout)

    def test_a_typo_is_named_with_its_neighbour_and_the_exit_status_says_so(self):
        out = self.run_coli(env={"COLI_KEEP_ALIV": "5m"})
        self.assertEqual(out.returncode, 1)
        self.assertIn("COLI_KEEP_ALIV is set, but nothing reads it.  did you mean COLI_KEEP_ALIVE", out.stdout)

    def test_search_and_json(self):
        out = self.run_coli("request", "log")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("COLI_REQUEST_LOG", out.stdout)
        data = json.loads(self.run_coli("--json", "--all").stdout)
        self.assertGreater(len(data["settings"]), 300)
        self.assertEqual(data["unknown"], [])
        self.assertEqual(self.run_coli("zzzqqq").returncode, 1)


if __name__ == "__main__":
    unittest.main()
