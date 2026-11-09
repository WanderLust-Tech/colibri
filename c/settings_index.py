"""The settings colibri reads, in one place: `coli settings` (docs/settings.md).

The list is not kept here. docs/ENVIRONMENT.md already describes every variable the engines and the gateway read, with its
default and what it does; this module reads those tables, so there is one text to maintain, and adds two things the page
cannot do: which of the variables are set in *this* environment, and which that are set are read by nothing (a typo).

A variable counts as read when it is in the tables or appears anywhere in the program's own source (c/*.c, *.h, *.py, the
`coli` script): a setting that is real but undocumented is not called a typo, and one that is neither is.
"""
import difflib
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parent / "docs" / "ENVIRONMENT.md"
TOKEN = re.compile(r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]*[A-Z0-9])\b")
NAME = re.compile(r"`([A-Z][A-Z0-9_]*)`")
SOURCE_SUFFIXES = (".c", ".h", ".py", ".cu", ".mm", ".m", ".cpp", ".sh", ".ps1")


class Setting:
    def __init__(self, name, default, effect, section):
        self.name, self.default, self.effect, self.section = name, default, effect, section

    def short(self, width=100):
        """The first sentence of the effect, cut to `width`."""
        text = re.sub(r"\s+", " ", self.effect.replace("**", "").replace("`", ""))
        first = re.split(r"(?<=[a-z0-9)])\.\s", text, 1)[0]
        return first if len(first) <= width else first[:width - 1].rstrip() + "…"

    def public(self):
        return {"name": self.name, "default": self.default, "effect": self.effect, "section": self.section}


def split_row(line):
    """The cells of a markdown table row; `\\|` inside a cell is a pipe, not a separator."""
    cells, cell, i = [], "", 0
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            cell += "|"
            i += 2
            continue
        if ch == "|":
            cells.append(cell.strip())
            cell = ""
        else:
            cell += ch
        i += 1
    if cell.strip():
        cells.append(cell.strip())
    return cells


def settings(path=None):
    """[Setting] from the tables of ENVIRONMENT.md: every row whose first cell names variables in backticks."""
    path = Path(path) if path else DOC
    out, seen, section = [], set(), ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            continue
        if not line.startswith("| `"):
            continue
        cells = split_row(line)
        if len(cells) < 3:
            continue
        names = NAME.findall(cells[0])
        default = re.sub(r"\s+", " ", cells[1].replace("`", "")).strip()
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(Setting(name, default, cells[2], section))
    return out


def source_names(root=None):
    """Every upper-case name with an underscore that the program's own source mentions."""
    root = Path(root) if root else HERE
    names = set()
    for path in root.rglob("*"):
        if path.is_dir() or "tests" in path.parts or "node_modules" in path.parts:
            continue
        if path.suffix not in SOURCE_SUFFIXES and path.name != "coli":
            continue
        try:
            if path.stat().st_size > 6 << 20:
                continue
            names.update(TOKEN.findall(path.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return names


def prefixes(documented, minimum=3):
    """The families of settings ("COLI", "Q36", "G4"...): a prefix shared by at least `minimum` documented names."""
    counts = {}
    for name in documented:
        if "_" in name:
            head = name.split("_", 1)[0]
            counts[head] = counts.get(head, 0) + 1
    return {head for head, n in counts.items() if n >= minimum}


def in_effect(environ, listed):
    """([(Setting, value)] of the documented settings that are set, [(name, value, [close names])] of the set variables of a
    known family that nothing reads)."""
    by_name = {s.name: s for s in listed}
    families = prefixes(by_name)
    known = None
    used, unknown = [], []
    for name in sorted(environ):
        if name in by_name:
            used.append((by_name[name], environ[name]))
            continue
        if "_" not in name or name.split("_", 1)[0] not in families:
            continue
        if known is None:
            known = source_names()
        if name in known:
            continue
        pool = sorted(set(by_name) | {n for n in known if n.split("_", 1)[0] == name.split("_", 1)[0]})
        unknown.append((name, environ[name], difflib.get_close_matches(name, pool, n=3, cutoff=0.75)))
    return used, unknown


def search(listed, words):
    """The settings whose name or effect holds every word (case does not matter), names first."""
    wanted = [w.lower() for w in words]
    hits = [s for s in listed if all(w in (s.name + " " + s.effect + " " + s.section).lower() for w in wanted)]
    return sorted(hits, key=lambda s: (not all(w in s.name.lower() for w in wanted), s.name))
