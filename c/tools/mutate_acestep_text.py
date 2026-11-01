#!/usr/bin/env python3
"""Proof that the text oracles can fail: break acestep_text.h one line at a time and see that an oracle says so.

    mutate_acestep_text.py MODEL_DIR TOKREF.json REFDIR [REFDIR ...] [--cc gcc] [--only NAME,NAME]

MODEL_DIR is a checkpoint (the tiny fixture of make_acestep_tiny.py is enough), TOKREF.json the tokenizer reference made
from it (tools/make_acestep_tok_ref.py) and each REFDIR a captured case of it (tools/acestep_ref.py capture ... --taps).
A mutant is a small, plausible mistake in a prompt template, the metas, the tokenizer's rules, or the encoder. It is
compiled into a temporary directory and run against the tokenizer reference (--tok-ref) and each case (--text-ref): it is
CAUGHT when any run fails. A mistake that only shows on a fractional duration, on set metas, on text past the 256 or 2048
limit or on a decomposed accent needs the case or the strings that have them, which is why several cases are passed.
The unmodified source is run first and has to pass. Exits non-zero if any mutant survives.

Pass these cases: inst10, vocal10, meta10, cjk10, cut10, empty10 and frac96.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate_acestep_vae as base  # noqa: E402

HEADER = "acestep_text.h"
FLAG = "--text-ref"

# name: (text in acestep_text.h, its replacement[, how many times it occurs]). Raw strings: the C source has a backslash and an n.
MUTANTS = {
    # --- the prompts ---------------------------------------------------------------------------------------------
    "caption-blank-line-after-instruction": (r"%s%s\n\n# Caption", r"%s%s\n# Caption"),
    "caption-no-trailing-newline": (r'seconds\n<|endoftext|>\n",', r'seconds\n<|endoftext|>",'),
    "caption-no-endoftext": (r'seconds\n<|endoftext|>\n",', r'seconds\n",'),
    "metas-order": (r"- timesignature: %s\n- keyscale: %s", r"- keyscale: %s\n- timesignature: %s"),
    "metas-duration-rounded": ("(int)duration);", "(int)(duration + 0.5));"),
    "metas-bpm-ignored": ('*b = bpm && *bpm ? bpm : "N/A"', '*b = "N/A"'),
    "metas-default-spelling": ('*b = bpm && *bpm ? bpm : "N/A"', '*b = bpm && *bpm ? bpm : "n/a"'),
    "metas-key-ignored": ('*k = act_blank(key) ? "N/A" : key', '*k = "N/A"'),
    "lyric-language-dropped": (r"# Languages\n%s\n\n# Lyric", r"# Languages\n\n\n# Lyric"),
    "lyric-no-endoftext": (r'# Lyric\n%s<|endoftext|>"', r'# Lyric\n%s"'),
    "lyric-header": (r"# Languages\n%s\n\n# Lyric\n%s<|endoftext|>", r"# Language\n%s\n\n# Lyric\n%s<|endoftext|>"),
    # --- the tokenizer's rules ---------------------------------------------------------------------------------
    "tokenize-no-appended-endoftext": ("    out[keep++] = t->eot;\n", ""),
    "tokenize-truncation-one-short": ("int keep = n < max_length - 1 ? n : max_length - 1;", "int keep = n < max_length - 2 ? n : max_length - 2;"),
    "tokenize-no-nfc": ("if (q38_nfc_normalize(s, strlen(s), &norm, &norm_len) != 0)", "if (1)"),
    "tokenize-wrong-special-id": ("t->eot = tok_id_of(&t->tok, \"<|endoftext|>\");", "t->eot = tok_id_of(&t->tok, \"<|im_end|>\");"),
    # --- the lyric path and the encoder ------------------------------------------------------------------------
    "embedding-row-index": ("(const uint16_t *)m->embed + (size_t)ids[i] * m->hidden;", "(const uint16_t *)m->embed + (size_t)(ids[i] ^ 1) * m->hidden;"),
    "attention-bidirectional": ("Hk * hd, T, H, Hk, hd, -1, 1);", "Hk * hd, T, H, Hk, hd, -1, 0);"),
    "rope-theta": ("acd_rope_table(cs, sn, n, m->hd, m->theta);", "acd_rope_table(cs, sn, n, m->hd, m->theta * 0.5f);"),
    "final-norm-skipped": ("    acd_rmsnorm(x, x, n, m->hidden, m->norm, m->eps);\n", ""),
    "query-norm-skipped": ("    acd_head_norm(q, T, H, hd, l->sa.qn, m->eps);\n", ""),
    "key-norm-skipped": ("    acd_head_norm(k, T, Hk, hd, l->sa.kn, m->eps);\n", ""),
    "keys-not-rotated": ("    acd_rope(k, T, Hk, hd, cs, sn);\n", ""),
    "swiglu-up-gate-swapped": ("g[i] = acd_silu(g[i]) * u[i];", "g[i] = acd_silu(u[i]) * g[i];"),
    "attention-norm-epsilon": ("acd_rmsnorm(n, x, T, D, l->ln1, m->eps);", "acd_rmsnorm(n, x, T, D, l->ln1, 1e-3f);"),
    "mlp-norm-weight": ("acd_rmsnorm(n, x, T, D, l->ln2, m->eps);", "acd_rmsnorm(n, x, T, D, l->ln1, m->eps);"),
    "attention-residual-dropped": ("    acd_lin(o, cx, T, &l->sa.o);\n    #pragma omp parallel for schedule(static)\n    for (int64_t i = 0; i < (int64_t)T * D; i++) x[i] += o[i];",
                                   "    acd_lin(o, cx, T, &l->sa.o);\n    #pragma omp parallel for schedule(static)\n    for (int64_t i = 0; i < (int64_t)T * D; i++) x[i] = o[i];"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("tokref")
    ap.add_argument("refdirs", nargs="+")
    ap.add_argument("--cc", default="gcc")
    ap.add_argument("--only", help="comma-separated mutant names")
    a = ap.parse_args()
    names = list(MUTANTS) if not a.only else a.only.split(",")
    unknown = [n for n in names if n not in MUTANTS]
    if unknown:
        sys.exit(f"unknown mutant(s) {unknown}; known: {list(MUTANTS)}")
    refs = [("--tok-ref", a.tokref)] + [(FLAG, r) for r in a.refdirs]
    with tempfile.TemporaryDirectory(prefix="acestep-mut-") as work:
        verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, None, HEADER, FLAG)
        print(f"{'unmodified':36s} {verdict}  {note}")
        if verdict != "pass":
            sys.exit("the unmodified source does not pass the oracles: fix that first")
        survivors = []
        for name in names:
            verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, MUTANTS[name], HEADER, FLAG)
            caught = verdict == "fail"
            print(f"{name:36s} {'CAUGHT' if caught else 'SURVIVED (' + verdict + ')'}  {note[:100]}", flush=True)
            if not caught:
                survivors.append(name)
    print(f"\n{len(names) - len(survivors)} of {len(names)} mutants caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
