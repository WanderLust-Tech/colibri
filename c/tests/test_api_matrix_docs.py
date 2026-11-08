"""The per-engine tables in docs/api.md must say what the registry says.

The registry's `tools` flag once said False for Qwen3.6 while its tool calls worked, and the doc had no row for DeepSeek V4.1;
nothing compared them. These tests read the two tables ("Tool-calling support" and "What else each engine takes") and fail
when a row is missing, unknown, or disagrees with family_registry.py. (That the registry's `tools` flag matches what the
gateway's renderer really does is held by test_family_registry.py.)
"""
import sys
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))

from family_registry import FAMILIES  # noqa: E402

API_MD = C_DIR.parent / "docs" / "api.md"

# What the first cell of a row begins with -> the family it describes. Longest first, so "DeepSeek V4.1" is not read as V4.
ROWS = (("DeepSeek V4.1", "deepseek_v41"), ("DeepSeek V4", "deepseek_v4"), ("GLM-5.2", "glm"), ("GLM-5.3-Flash", "glm53"),
        ("Inkling", "inkling"), ("Kimi K3", "kimi"), ("MiMo-V2.6", "mimo"), ("Qwen3.6", "qwen36"),
        ("Qwen3.8-Flash-Next", "qwen38"), ("Gemma 4", "gemma4"), ("OLMoE", "olmoe"))


def table_after(heading):
    """The rows (lists of cell texts) of the first table below `heading` in docs/api.md."""
    lines = API_MD.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("|"):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        elif rows:
            break
    return rows[2:]                                   # past the header and the |---| line


def by_family(rows):
    found = {}
    for cells in rows:
        match = [family for prefix, family in ROWS if cells[0].startswith(prefix)]
        if not match:
            raise AssertionError(f"docs/api.md has a row for an engine this test does not know: {cells[0]!r}")
        if match[0] in found:
            raise AssertionError(f"docs/api.md lists {cells[0]!r} twice")
        found[match[0]] = cells
    return found


def yes(cell):
    return cell.lower().startswith("yes")


def json_mode(cell):
    """What a `response_format` cell promises: the answer is "constrained" to the grammar, the grammar only feeds "drafts" the
    model verifies (a speed-up), or it is "no" (refused)."""
    text = cell.lower()
    if text.startswith("yes, constrained"):
        return "constrained"
    if text.startswith("yes, as a speed-up only"):
        return "drafts"
    if text.startswith("no"):
        return "no"
    raise AssertionError(f"a JSON mode cell this test cannot read: {cell!r}")


def chat_families():
    return {f.id: f for f in FAMILIES if f.modality == "text" and not f.capabilities.decision}


class ApiMatrixDocsTest(unittest.TestCase):
    def test_every_chat_family_has_a_row_in_each_table(self):
        for heading in ("### Tool-calling support", "### What else each engine takes"):
            self.assertEqual(set(by_family(table_after(heading))), set(chat_families()), heading)

    def test_tool_calling_table_matches_the_registry(self):
        for family_id, cells in by_family(table_after("### Tool-calling support")).items():
            flag = chat_families()[family_id].capabilities.tools
            self.assertEqual(yes(cells[1]), flag, f"{family_id}: OpenAI `tools` column says {cells[1]!r}, registry says {flag}")
            self.assertEqual(yes(cells[2]), flag, f"{family_id}: Anthropic `tool_use` column says {cells[2]!r}, registry says {flag}")

    def test_the_constrained_families_are_the_engines_that_say_grammar_1(self):
        """The gateway sends a grammar to an engine whose CAPS line says grammar=1; the registry and docs say which families
        should. They must name the same engines (a string literal with grammar=1 in an engine's source)."""
        import re
        said = {path.stem for path in C_DIR.glob("*.c")
                if re.search(r'"[^"\n]*\bgrammar=1', path.read_text(encoding="utf-8", errors="replace"))}
        registered = {family_id for family_id, family in chat_families().items() if family.capabilities.constrained_output}
        self.assertEqual(said, registered)

    def test_no_family_is_both_draft_only_and_constrained(self):
        for family_id, family in chat_families().items():
            self.assertFalse(family.capabilities.grammar_payload and family.capabilities.constrained_output, family_id)

    def test_json_mode_and_pictures_match_the_registry(self):
        for family_id, cells in by_family(table_after("### What else each engine takes")).items():
            caps = chat_families()[family_id].capabilities
            registry = "constrained" if caps.constrained_output else "drafts" if caps.grammar_payload else "no"
            self.assertEqual(json_mode(cells[1]), registry,
                             f"{family_id}: JSON mode column says {cells[1]!r}, the registry says {registry}")
            self.assertEqual(yes(cells[2]), caps.image,
                             f"{family_id}: pictures column says {cells[2]!r}, registry image is {caps.image}")


if __name__ == "__main__":
    unittest.main()
