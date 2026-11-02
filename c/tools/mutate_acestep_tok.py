#!/usr/bin/env python3
"""Proof that the cover tokenizer's oracle can fail: break acestep_tok.h one line at a time and see that `--cover-ref` says so.

    mutate_acestep_tok.py MODEL_DIR REFDIR [REFDIR ...] [--cc gcc] [--only NAME,NAME]

MODEL_DIR is the real checkpoint and each REFDIR a case captured by `acestep_ref_audio.py tasks` (a cover case: it holds
the tokenizer's own tensors). A mutant is a small, plausible mistake in the path a cover's source takes: the padding of a
length that is not a multiple of 5, the pooler's special token and its position, the quantizer's soft clamp and rounding, the
detokenizer's position tokens and window layout. It is compiled into a temporary directory and run against `--cover-ref` on
every case: CAUGHT when any run fails. The unmodified source is run first and has to pass. Exits non-zero if a mutant survives.

Pass these cases: cover10, cover_odd (234 frames: one frame of padding), cover_pad4 (231: the last window is one frame and four of
padding, which the quantizer cannot absorb), cover_vocal.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate_acestep_vae as base  # noqa: E402

HEADER = "acestep_tok.h"
FLAG = "--cover-ref"

# name: (text in acestep_tok.h, its replacement[, how many times it occurs])
MUTANTS = {
    # --- the padding of a length that is not a multiple of the window -------------------------------------------------------
    "pad-with-zeros": ("if (pad) memcpy(xp + (size_t)T * A, silence, (size_t)pad * A * sizeof(float));", "if (pad) memset(xp + (size_t)T * A, 0, (size_t)pad * A * sizeof(float));"),
    "pad-from-the-clip": ("if (pad) memcpy(xp + (size_t)T * A, silence, (size_t)pad * A * sizeof(float));", "if (pad) memcpy(xp + (size_t)T * A, x + (size_t)(T - pad) * A, (size_t)pad * A * sizeof(float));"),
    # --- the pooler --------------------------------------------------------------------------------------------------
    "pool-special-token-last": ("memcpy(h + (size_t)w * L * D, t->pool_special, (size_t)D * sizeof(float));\n            memcpy(h + (size_t)(w * L + 1) * D, e + (size_t)w * P * D, (size_t)P * D * sizeof(float));",
                                "memcpy(h + (size_t)(w * L + P) * D, t->pool_special, (size_t)D * sizeof(float));\n            memcpy(h + (size_t)(w * L) * D, e + (size_t)w * P * D, (size_t)P * D * sizeof(float));"),
    "pool-reads-last-token": ("memcpy(out + (size_t)(w0 + w) * D, h + (size_t)w * L * D, (size_t)D * sizeof(float));", "memcpy(out + (size_t)(w0 + w) * D, h + (size_t)(w * L + P) * D, (size_t)D * sizeof(float));"),
    "pool-embed-skipped": ("acd_lin(e, p, nw * P, &t->pool_embed);", "memcpy(e, p, (size_t)nw * P * D * sizeof(float));"),
    "pool-norm-skipped": ("acd_rmsnorm(h, h, nw * L, D, t->pool_norm, d->eps);", ""),
    "pool-one-layer": ("for (int i = 0; i < t->npool; i++) atk_enc_layer_win(d, &t->pool[i], h, nw, L, cs, sn);", "for (int i = 0; i < 1; i++) atk_enc_layer_win(d, &t->pool[i], h, nw, L, cs, sn);"),
    "pool-rope-length": ("atk_rope_tiled(d, L, bw, &cs, &sn);\n    float *p =", "atk_rope_tiled(d, L + 1, bw, &cs, &sn);\n    float *p ="),
    "pool-windows-attend-each-other": ("acd_attention(ctx + (size_t)w * L * H * hd, H * hd, q + (size_t)w * L * H * hd, H * hd, L,\n                      k + (size_t)w * L * Hk * hd, v + (size_t)w * L * Hk * hd, Hk * hd, L, H, Hk, hd, l->sliding ? d->window : -1);",
                                       "acd_attention(ctx + (size_t)w * L * H * hd, H * hd, q + (size_t)w * L * H * hd, H * hd, L,\n                      k, v, Hk * hd, rows, H, Hk, hd, l->sliding ? d->window : -1);"),
    # --- the quantizer ---------------------------------------------------------------------------------------------------
    "fsq-soft-clamp-factor": ("const float clampv = 1.f + 1.f / (float)(t->levels[i] - 1);", "const float clampv = 1.f + 1.f / (float)t->levels[i];"),
    "fsq-no-soft-clamp": ("float v = tanhf(z[(size_t)w * n + i] / clampv) * clampv;", "float v = z[(size_t)w * n + i];"),
    "fsq-clamp-scale": ("float v = tanhf(z[(size_t)w * n + i] / clampv) * clampv;", "float v = tanhf(z[(size_t)w * n + i]) * clampv;"),
    "fsq-no-hard-clamp": ("const float zc = z < -1.f ? -1.f : z > 1.f ? 1.f : z;", "const float zc = z;"),
    "fsq-rounds-down-only": ("const float b = lm1 * (zc + 1.f) / 2.f + 0.5f;", "const float b = lm1 * (zc + 1.f) / 2.f;"),
    "fsq-levels-off-by-one": ("const float lm1 = (float)(L - 1);", "const float lm1 = (float)L;"),
    "fsq-code-scale": ("return (2.f / lm1) * floorf(b) - 1.f;", "return (1.f / lm1) * floorf(b) - 1.f;"),
    "fsq-round-not-floor": ("return (2.f / lm1) * floorf(b) - 1.f;", "return (2.f / lm1) * roundf(b) - 1.f;"),
    "fsq-level-order": ("const float clampv = 1.f + 1.f / (float)(t->levels[i] - 1);\n            float v = tanhf(z[(size_t)w * n + i] / clampv) * clampv;       /* the soft clamp ahead of the residual quantizer */\n            c[(size_t)w * n + i] = atk_code(v, t->levels[i],",
                        "const float clampv = 1.f + 1.f / (float)(t->levels[n - 1 - i] - 1);\n            float v = tanhf(z[(size_t)w * n + i] / clampv) * clampv;       /* the soft clamp ahead of the residual quantizer */\n            c[(size_t)w * n + i] = atk_code(v, t->levels[i],"),
    # --- the detokenizer -----------------------------------------------------------------------------------------------
    "detok-no-position-tokens": ("for (int i = 0; i < D; i++) o[i] = r[i] + s[i];", "for (int i = 0; i < D; i++) o[i] = r[i];"),
    "detok-positions-reversed": ("const float *s = t->d_special + (size_t)p * D, *r = e + (size_t)w * D;", "const float *s = t->d_special + (size_t)(P - 1 - p) * D, *r = e + (size_t)w * D;"),
    "detok-embed-skipped": ("acd_lin(e, q + (size_t)w0 * D, nw, &t->d_embed);", "memcpy(e, q + (size_t)w0 * D, (size_t)nw * D * sizeof(float));"),
    "detok-norm-skipped": ("acd_rmsnorm(h, h, nw * P, D, t->d_norm, d->eps);", ""),
    "detok-window-rows": ("acd_lin(out + (size_t)w0 * P * d->acdim, h, nw * P, &t->d_proj);", "acd_lin(out + (size_t)w0 * P * d->acdim, h, nw * P, &t->d_proj); if (nw > 1) memcpy(out + (size_t)w0 * P * d->acdim, out + (size_t)(w0 + 1) * P * d->acdim, (size_t)P * d->acdim * sizeof(float));"),
    "hints-not-cut": ("memcpy(hints, hp, (size_t)T * A * sizeof(float));", "memcpy(hints, hp + (size_t)pad * A, (size_t)T * A * sizeof(float));"),
}
# Deliberately NOT here, because no output can tell it from the original:
#  - padding a whole extra window when T is already a multiple of 5 (`pad = P - T % P`): windows never see each other and the
#    result is cut back to T frames, so an extra window changes nothing but the time.
# (a mutant whose text is not in the file is a mistake in this list: bad-mutant is reported by the runner)



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("refdirs", nargs="+")
    ap.add_argument("--cc", default="gcc")
    ap.add_argument("--only", help="comma-separated mutant names")
    a = ap.parse_args()
    names = list(MUTANTS) if not a.only else a.only.split(",")
    unknown = [n for n in names if n not in MUTANTS]
    if unknown:
        sys.exit(f"unknown mutant(s) {unknown}; known: {list(MUTANTS)}")
    refs = [(FLAG, r) for r in a.refdirs]
    with tempfile.TemporaryDirectory(prefix="acestep-mut-") as work:
        verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, None, HEADER, FLAG)
        print(f"{'unmodified':34s} {verdict}  {note}")
        if verdict != "pass":
            sys.exit("the unmodified source does not pass the oracle: fix that first")
        survivors = []
        for name in names:
            verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, MUTANTS[name], HEADER, FLAG)
            caught = verdict == "fail"
            print(f"{name:34s} {'CAUGHT' if caught else 'SURVIVED (' + verdict + ')'}  {note[:100]}", flush=True)
            if not caught:
                survivors.append(name)
    print(f"\n{len(names) - len(survivors)} of {len(names)} mutants caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
