#!/usr/bin/env python3
"""Proof that the VAE encoder oracle can fail: break acestep_vae_enc.h one line at a time and see that `--vae-enc-ref` says so.

    mutate_acestep_venc.py MODEL_DIR REFDIR [REFDIR ...] [--cc gcc] [--only NAME,NAME]

MODEL_DIR is a checkpoint (the real one: the tiny fixture has no encoder) and each REFDIR a case captured by
`acestep_ref_audio.py enc`. A mutant is a small, plausible mistake in what the encoder adds to the decoder's kernels: the
strided convolution, the order of a block, the split of the output into mean and scale, the std, the posterior draw's layout,
and the windows of the tiled encode. It is compiled into a temporary directory and run against `--vae-enc-ref` on every case:
CAUGHT when any run fails. The unmodified source is run first and has to pass. Exits non-zero if any mutant survives.

Pass these cases: single10 and single28 (the parameters), tile_small, tile_odd and tile_short_tail (windows, an odd length and a
tail), tile_exact (a clip of exactly one chunk: one pass, not two windows), tile_default (the 30 s chunk).
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate_acestep_vae as base  # noqa: E402

HEADER = "acestep_vae_enc.h"
FLAG = "--vae-enc-ref"

# name: (text in acestep_vae_enc.h, its replacement[, how many times it occurs])
MUTANTS = {
    # --- the network ---------------------------------------------------------------------------------------------------
    "down-padding": ("const int k = c->k, pad = (s + 1) / 2, kc = k * c->cin;", "const int k = c->k, pad = s / 2 + 1, kc = k * c->cin;"),
    "down-taps-reversed": ("int64_t src = (r0 + i) * s + j - pad;", "int64_t src = (r0 + i) * s + (k - 1 - j) - pad;"),
    "down-stride-one": ("int64_t src = (r0 + i) * s + j - pad;", "int64_t src = (r0 + i) * (s - 1) + j - pad;"),
    "down-bias-dropped": ("qi_gemm(y + (size_t)r0 * c->cout, col, (int)n, &c->m, c->b);", "qi_gemm(y + (size_t)r0 * c->cout, col, (int)n, &c->m, NULL);"),
    "down-length": ("const int64_t Tout = (T + 2 * pad - k) / s + 1;", "const int64_t Tout = (T + 2 * pad - k) / s;"),
    "dilations": ("static const int dil[3] = {1, 3, 9};", "static const int dil[3] = {1, 3, 3};"),
    "res-unit-order": ("for (int j = 0; j < 3; j++) acv_res(&b->r[j], x, (int)cur, nx, tmp, w->col);", "for (int j = 2; j >= 0; j--) acv_res(&b->r[j], x, (int)cur, nx, tmp, w->col);"),
    "block-snake-skipped": ("        acv_snake(x, x, (int)cur, &b->sn);\n        cur = acve_down", "        cur = acve_down"),
    "block-snake-before-units": ("        for (int j = 0; j < 3; j++) acv_res(&b->r[j], x, (int)cur, nx, tmp, w->col);\n        acv_snake(x, x, (int)cur, &b->sn);",
                                 "        acv_snake(x, x, (int)cur, &b->sn);\n        for (int j = 0; j < 3; j++) acv_res(&b->r[j], x, (int)cur, nx, tmp, w->col);"),
    "final-snake-skipped": ("    acv_snake(x, x, (int)cur, &e->sn);\n    acv_conv(&e->conv2", "    acv_conv(&e->conv2"),
    "block-widths": ("b->cin = ch * m[i]; b->cout = ch * m[i + 1];", "b->cin = ch * m[i]; b->cout = ch * m[i + 1 < nr ? i + 1 : i];"),
    "block-strides-reversed": ("b->s = ratios[i];", "b->s = ratios[nr - 1 - i];"),
    "first-conv-kernel": ("acv_load_conv(&S, \"encoder.conv1\", e->ach, ch * m[0], 7, 1, 1, &e->conv1, err, sizeof err)", "acv_load_conv(&S, \"encoder.conv1\", e->ach, ch * m[0], 7, 2, 1, &e->conv1, err, sizeof err)"),
    # --- the posterior --------------------------------------------------------------------------------------------------
    "mean-scale-swapped": ("float mean = raw[(size_t)t * 2 * zdim + c], scale = raw[(size_t)t * 2 * zdim + zdim + c];",
                           "float mean = raw[(size_t)t * 2 * zdim + zdim + c], scale = raw[(size_t)t * 2 * zdim + c];"),
    "std-epsilon": ("(acve_softplus(scale) + 1e-4f)", "(acve_softplus(scale) + 1e-3f)"),
    "std-no-softplus": ("(acve_softplus(scale) + 1e-4f)", "(expf(scale) + 1e-4f)"),
    "eps-not-transposed": ("eps[(size_t)c * T + t]", "eps[(size_t)t * zdim + c]"),
    "eps-sign": ("mean + (acve_softplus(scale) + 1e-4f) * eps[(size_t)c * T + t]", "mean - (acve_softplus(scale) + 1e-4f) * eps[(size_t)c * T + t]"),
    "eps-ignored": ("eps ? mean + (acve_softplus(scale) + 1e-4f) * eps[(size_t)c * T + t] : mean", "mean"),
    # --- the windows ------------------------------------------------------------------------------------------------------
    "tile-single-pass-boundary": ("    if (N <= chunk) {", "    if (N < chunk) {"),
    "tile-stride": ("const int64_t stride = chunk - 2 * overlap;", "const int64_t stride = chunk - overlap;"),
    "tile-context-start": ("int64_t ws = cs - overlap > 0 ? cs - overlap : 0,", "int64_t ws = cs - overlap / 2 > 0 ? cs - overlap / 2 : 0,"),
    "tile-context-end": ("we = ce + overlap < N ? ce + overlap : N;", "we = ce < N ? ce : N;"),
    "tile-trim-start": ("int64_t trim0 = acve_round((double)(cs - ws) / down),", "int64_t trim0 = acve_round((double)(cs - ws) / down) + 1,"),
    "tile-trim-end-dropped": ("int64_t end = trim1 > 0 ? T - trim1 : T;", "int64_t end = T;"),
    "tile-trim-end-extra": ("int64_t end = trim1 > 0 ? T - trim1 : T;", "int64_t end = trim1 > 0 ? T - trim1 - 1 : T;"),
    "tile-eps-once": ("        if (eps) eps(dr, (size_t)T * z, ud);\n        acve_sample(raw, T, z, dr, zz);",
                      "        if (eps && i == 0) eps(dr, (size_t)T * z, ud);\n        acve_sample(raw, T, z, dr, zz);"),
    "tile-window-offset": ("wav + (size_t)ws * e->ach, we - ws, raw);", "wav + (size_t)(ws + 1) * e->ach, we - ws, raw);"),
    "tile-downsample-factor": ("if (down == 0) down = (double)(we - ws) / (double)T;", "down = 3840.0;"),
}

# Deliberately NOT here, because no output can tell them from the original:
#  - softplus's threshold (20): the scale half of the posterior is negative in every capture (the std is 1e-4 to 1e-2), so the
#    branch `x > threshold` is never taken, for 20 or for 2; it is torch's own guard against overflow, kept for fidelity;
#  - acve_round's tie-breaking (to even, as Python's round): a window's context is trimmed by (context samples / 1920) frames,
#    and the context is always a whole number of frames in upstream's defaults (96000 = 50 x 1920), so a tie (x.5) cannot occur;
#  - the sample's eps being NULL (the posterior's mean): no case asks for it, only a caller that wants a deterministic latent.


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
