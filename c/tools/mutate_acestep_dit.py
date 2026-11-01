#!/usr/bin/env python3
"""Proof that the DiT oracle can fail: break acestep_dit.h one line at a time and see that the oracle says so.

    mutate_acestep_dit.py MODEL_DIR REFDIR [REFDIR ...] [--cc gcc] [--only NAME,NAME]

MODEL_DIR is a checkpoint (the tiny fixture of make_acestep_tiny.py is enough) and each REFDIR a captured case of it
(tools/acestep_ref.py capture ... --taps). A mutant is a small, plausible mistake in the condition encoders, the DiT
layer, the timestep embedding, the sampler or DCW. It is compiled into a temporary directory and run against the cases:
it is CAUGHT when the oracle fails on any of them (a mistake that only shows on an odd clip, or a shift-3 schedule, or
DCW, needs a case that has it), and SURVIVES when every case still passes. The unmodified source is run first and has to
pass. Exits non-zero if any mutant survives.

Pass these cases: one with DCW (inst10), an odd length (odd9), shift 3 (shift3), 5 steps (steps5) and the 30-second clip
(long30), so that the sliding window, the schedule transforms and the odd DCW tail are all exercised.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate_acestep_vae as base  # noqa: E402

HEADER = "acestep_dit.h"
FLAG = "--dit-ref"

# name: (text in acestep_dit.h, its replacement[, how many times the text occurs and is replaced]).
MUTANTS = {
    # --- the condition: lyric and timbre encoders, packing -------------------------------------------------------
    "timbre-last-frame": ("memcpy(raw + (size_t)Ll * D, tb, (size_t)D * sizeof(float));",
                          "memcpy(raw + (size_t)Ll * D, tb + (size_t)(Tr - 1) * D, (size_t)D * sizeof(float));"),
    "encoder-final-norm-skipped": ("    acd_rmsnorm(x, x, T, d->D, norm, d->eps);\n    free(cs); free(sn);\n    return x;",
                                   "    (void)norm;\n    free(cs); free(sn);\n    return x;"),
    "encoder-layers-all-full": ("L[i].sliding = d->sliding[i];", "L[i].sliding = 0;"),
    "text-projection-skipped": ("acd_lin(raw + (size_t)(Ll + 1) * D, text, Lt, &d->text_proj);",
                                "for (int i = 0; i < Lt * D; i++) raw[(size_t)(Ll + 1) * D + i] = text[(size_t)(i / D) * d->textdim + i % D];"),
    "condition-embedder-skipped": ("acd_lin(emb, raw, c->len, &d->cond_emb);", "memcpy(emb, raw, (size_t)c->len * D * sizeof(float));"),
    # --- attention ---------------------------------------------------------------------------------------------
    "window-one-too-small": ("sliding ? d->window : -1);", "sliding ? d->window - 1 : -1);"),
    "window-one-too-large": ("sliding ? d->window : -1);", "sliding ? d->window + 1 : -1);"),
    "rope-keys-not-rotated": ("    acd_rope(k, T, Hk, hd, cs, sn);\n", ""),
    "rope-sign": ("v[i] = x1 * c[i] - x2 * s[i];", "v[i] = x1 * c[i] + x2 * s[i];"),
    "rope-theta": ("inv[i] = 1.0f / powf(theta, (float)(2 * i) / (float)hd);", "inv[i] = 1.0f / powf(theta, (float)i / (float)hd);"),
    "query-norm-skipped": ("    acd_head_norm(q, T, H, hd, a->qn, d->eps);\n    acd_head_norm(k, T, Hk, hd, a->kn, d->eps);",
                           "    acd_head_norm(k, T, Hk, hd, a->kn, d->eps);"),
    "key-norm-skipped": ("    acd_head_norm(k, T, Hk, hd, a->kn, d->eps);\n    acd_rope(q, T, H, hd, cs, sn);",
                         "    acd_rope(q, T, H, hd, cs, sn);"),
    "kv-head-mapping": ("const int h = kh * group + g;", "const int h = g * kvheads + kh;"),
    "attention-scale": ("const float scale = 1.f / sqrtf((float)hd);\n    const int group", "const float scale = 1.f / (float)hd;\n    const int group"),
    "cross-keys-not-normed": ("        acd_head_norm(c->ck[i], c->len, Hk, hd, d->L[i].ca.kn, d->eps);\n", ""),
    "cross-queries-not-normed": ("    acd_head_norm(q, Tp, H, hd, l->ca.qn, d->eps);\n", ""),
    # --- the DiT layer ------------------------------------------------------------------------------------------
    "adaln-one-plus-scale": ("* (1.f + scale_a[i]) + shift_a[i];", "* scale_a[i] + shift_a[i];"),
    "adaln-shift-scale-swapped": ("const float *shift_a = ss[0], *scale_a = ss[1],", "const float *shift_a = ss[1], *scale_a = ss[0],"),
    "attention-gate-index": ("*gate_a = ss[2],", "*gate_a = ss[3],"),
    "mlp-gate-dropped": ("h[(size_t)p * D + i] += o[(size_t)p * D + i] * gate_m[i];", "h[(size_t)p * D + i] += o[(size_t)p * D + i];"),
    "mlp-modulation-swapped": ("*shift_m = ss[3], *scale_m = ss[4],", "*shift_m = ss[4], *scale_m = ss[3],"),
    "cross-attention-gated": ("    for (int64_t i = 0; i < (int64_t)Tp * D; i++) h[i] += o[i];\n    if (acd_tap_fn) acd_tap_fn(1001",
                              "    for (int64_t i = 0; i < (int64_t)Tp * D; i++) h[i] += o[i] * gate_a[i % D];\n    if (acd_tap_fn) acd_tap_fn(1001"),
    "swiglu-up-gate-swapped": ("g[i] = acd_silu(g[i]) * u[i];", "g[i] = acd_silu(u[i]) * g[i];", 2),
    "silu-sign": ("return x / (1.f + expf(-x));", "return x / (1.f + expf(x));"),
    "rmsnorm-epsilon": ("float inv = 1.f / sqrtf((float)(s / n) + eps);", "float inv = 1.f / sqrtf((float)(s / n) + 1e-3f);"),
    "rmsnorm-weight": ("y[i] = w[i] * (x[i] * inv);", "y[i] = x[i] * inv;"),
    # --- the timestep embedding and the patches ------------------------------------------------------------------
    "second-embedding-of-t": ("acd_time_embed(d, &d->ter, t - r, tr, pr);", "acd_time_embed(d, &d->ter, t, tr, pr);"),
    "second-embedding-dropped": ("    for (int i = 0; i < D; i++) temb[i] += tr[i];\n    for (int i = 0; i < 6 * D; i++) proj[i] += pr[i];\n", ""),
    "embedding-sin-cos-swapped": ("e[i] = (float)cos((double)a); e[half + i] = (float)sin((double)a);",
                                  "e[i] = (float)sin((double)a); e[half + i] = (float)cos((double)a);"),
    "embedding-scale": ("float ts = t * 1000.0f;", "float ts = t * 100.0f;"),
    "embedding-max-period": ("(float)(-9.210340371976184)", "(float)(-6.907755278982137)"),
    "patch-context-xt-order": ("memcpy(dst, ctx + (size_t)f * 2 * A, (size_t)2 * A * sizeof(float));\n            memcpy(dst + 2 * A, xt + (size_t)f * A, (size_t)A * sizeof(float));",
                               "memcpy(dst, xt + (size_t)f * A, (size_t)A * sizeof(float));\n            memcpy(dst + A, ctx + (size_t)f * 2 * A, (size_t)2 * A * sizeof(float));"),
    "output-shift-scale-swapped": ("shift[i] = d->sst_out[i] + temb[i]; scale[i] = d->sst_out[D + i] + temb[i];",
                                   "shift[i] = d->sst_out[D + i] + temb[i]; scale[i] = d->sst_out[i] + temb[i];"),
    "output-norm-dropped": ("    acd_rmsnorm(n, h, Tp, D, d->norm_out, d->eps);\n    #pragma omp parallel for schedule(static)\n    for (int p = 0; p < Tp; p++) for (int i = 0; i < D; i++) n[(size_t)p * D + i] = n[(size_t)p * D + i] * (1.f + scale[i]) + shift[i];",
                            "    memcpy(n, h, (size_t)Tp * D * sizeof(float));\n    #pragma omp parallel for schedule(static)\n    for (int p = 0; p < Tp; p++) for (int i = 0; i < D; i++) n[(size_t)p * D + i] = n[(size_t)p * D + i] * (1.f + scale[i]) + shift[i];"),
    # --- the sampler and DCW ------------------------------------------------------------------------------------
    "euler-step-uses-t": ("out[k] = x[k] - v[k] * dt;", "out[k] = x[k] - v[k] * t;"),
    "last-step-not-to-x0": ("out[k] = x[k] - v[k] * t;", "out[k] = x[k] - v[k] * 1.0f;"),
    "sampler-r-zero": ("acd_forward(d, c, x, ctx, T, sched[i], sched[i], v);", "acd_forward(d, c, x, ctx, T, sched[i], 0.0f, v);"),   # only the chained run sees it
    "schedule-shift-formula": ("raw = (double)shift * raw / (1.0 + ((double)shift - 1.0) * raw);", "raw = (double)shift * raw / (1.0 + (double)shift * raw);"),
    "schedule-linspace": ("double raw = 1.0 - (double)i / n;", "double raw = 1.0 - (double)i / (n - 1 > 0 ? n - 1 : 1);"),
    "dcw-never-applied": ("acd_dcw(out, den, T, A, (float)((double)t * o->dcw_low), (float)((1.0 - (double)t) * o->dcw_high));",
                          "(void)den;"),
    "dcw-bands-swapped": ("(float)((double)t * o->dcw_low), (float)((1.0 - (double)t) * o->dcw_high));",
                          "(float)((1.0 - (double)t) * o->dcw_high), (float)((double)t * o->dcw_low));"),
    "dcw-low-not-scaled-by-t": ("(float)((double)t * o->dcw_low),", "(float)(o->dcw_low),"),
    "dcw-high-not-complementary": ("(float)((1.0 - (double)t) * o->dcw_high)", "(float)((double)t * o->dcw_high)"),
    "dcw-second-frame-sign": ("if (x1) x1[c] += low - high;", "if (x1) x1[c] += low + high;"),
    "dcw-odd-tail": ("e1 = x1 ? x1[c] - y1[c] : 0.f;", "e1 = x1 ? x1[c] - y1[c] : e0;"),     # shows only on an odd number of frames
    "dcw-uses-new-latents": ("for (size_t k = 0; k < N; k++) den[k] = x[k] - v[k] * t;", "for (size_t k = 0; k < N; k++) den[k] = out[k] - v[k] * t;"),
}


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
    with tempfile.TemporaryDirectory(prefix="acestep-mut-") as work:
        verdict, note = base.build_and_run(a.cc, work, a.model_dir, a.refdirs, None, HEADER, FLAG)
        print(f"{'unmodified':30s} {verdict}  {note}")
        if verdict != "pass":
            sys.exit("the unmodified source does not pass the oracle: fix that first")
        survivors = []
        for name in names:
            verdict, note = base.build_and_run(a.cc, work, a.model_dir, a.refdirs, MUTANTS[name], HEADER, FLAG)
            caught = verdict == "fail"
            print(f"{name:30s} {'CAUGHT' if caught else 'SURVIVED (' + verdict + ')'}  {note[:100]}", flush=True)
            if not caught:
                survivors.append(name)
    print(f"\n{len(names) - len(survivors)} of {len(names)} mutants caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
