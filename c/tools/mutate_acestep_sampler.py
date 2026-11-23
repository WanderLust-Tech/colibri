#!/usr/bin/env python3
"""Proof that the sampler options' oracle can fail: break the Heun, SDE, velocity clamp and smoothing code one line at a time and see that `--gen-ref` says so.

    mutate_acestep_sampler.py MODEL_DIR REFDIR [REFDIR ...] [--dit acestep-v15-base] [--cc gcc] [--only NAME,NAME] [--skip NAME,NAME]

MODEL_DIR is a checkpoint and each REFDIR a case captured by `acestep_ref_audio.py tasks` with the sampler options (the `s_*` cases for the turbo DiT; the
`b_*` sampler cases with `--dit acestep-v15-base`, which this tool then needs too). A mutant is a small, plausible mistake in what the options add: the SDE's
noise and the level it noises to, Heun's predictor, corrector and average, the guidance the corrector uses, the clamp and the smoothing. It is compiled into a
temporary directory and run against `--gen-ref` on every case: CAUGHT when any run fails. The unmodified source is run first and has to pass. A few
mutants only a guided DiT can show (the corrector's guidance, the SDE's unshifted level): run the tool once on the turbo cases and once on the base ones, and
a mutant has to be caught by one of the two. Exits non-zero if a mutant survives its run.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate_acestep_vae as base  # noqa: E402

FLAG = "--gen-ref"

# name: (header, text, its replacement[, how many times it occurs])
MUTANTS = {
    # --- the SDE step ------------------------------------------------------------------------------------------------------------
    "sde-seed": ("acestep_gen.h", "acg_mt_seed(&sde_mt, (uint32_t)rq->seed ^ ACG_SDE_SEED_XOR);", "acg_mt_seed(&sde_mt, (uint32_t)rq->seed);"),
    "sde-weights-swapped": ("acestep_gen.h", "nx[k] = a * vdraw[k] + b * (x[k] - v[k] * t);", "nx[k] = b * vdraw[k] + a * (x[k] - v[k] * t);"),
    "sde-predicts-from-the-next-level": ("acestep_gen.h", "nx[k] = a * vdraw[k] + b * (x[k] - v[k] * t);", "nx[k] = a * vdraw[k] + b * (x[k] - v[k] * (float)tn);"),
    "sde-level-is-the-shifted-one": ("acestep_gen.h", "const double tn = d->turbo ? (double)sched[i + 1] : 1.0 - (double)(i + 1) / (double)n;",
                                     "const double tn = (double)sched[i + 1];"),
    "sde-level-ignores-the-turbo-schedule": ("acestep_gen.h", "const double tn = d->turbo ? (double)sched[i + 1] : 1.0 - (double)(i + 1) / (double)n;",
                                             "const double tn = 1.0 - (double)(i + 1) / (double)n;"),
    "sde-dcw-dropped": ("acestep_gen.h", "acd_dcw_after(nx, x, v, t, T, A, &so);\n            t_after = (float)tn;", "t_after = (float)tn;"),
    "sde-repaint-level": ("acestep_gen.h", "const float tn = t_after, a = tn,", "const float tn = sched[i + 1], a = tn,"),
    "turbo-last-step-is-noised": ("acestep_gen.h", "const int to_x0 = d->turbo && i == n - 1;", "const int to_x0 = d->turbo && i == n - 2;"),
    # --- Heun ------------------------------------------------------------------------------------------------------------------
    "heun-average": ("acestep_gen.h", "v[k] = 0.5f * (v[k] + v2[k]);", "v[k] = v2[k];"),
    "heun-weights": ("acestep_gen.h", "v[k] = 0.5f * (v[k] + v2[k]);", "v[k] = 0.25f * v[k] + 0.75f * v2[k];"),
    "heun-predictor-step": ("acestep_gen.h", "xp[k] = x[k] - v[k] * dt;", "xp[k] = x[k] - v[k] * t;"),
    "heun-corrector-level": ("acestep_gen.h", "acd_forward(d, cur, xp, cur_ctx, T, tn, tn, v2);", "acd_forward(d, cur, xp, cur_ctx, T, t, t, v2);"),
    "heun-corrector-input": ("acestep_gen.h", "acd_forward(d, cur, xp, cur_ctx, T, tn, tn, v2);", "acd_forward(d, cur, x, cur_ctx, T, tn, tn, v2);"),
    "heun-dcw-from-the-average": ("acestep_gen.h", "acd_dcw_after(nx, x, vd, t, T, A, &so);", "acd_dcw_after(nx, x, v, t, T, A, &so);"),
    "heun-last-step-of-the-turbo": ("acestep_gen.h", "} else if (heun && !to_x0) {", "} else if (heun) {"),
    "heun-skips-the-guided-last-step": ("acestep_gen.h", "} else if (heun && !to_x0) {", "} else if (heun && i < n - 1) {"),
    "heun-corrector-guidance-is-apg": ("acestep_gen.h", "else acd_cfg(vg, v2, vu, NL, guidance);", "else acd_apg(vg, v2, vu, T, A, guidance, mom);"),
    "heun-corrector-adg-at-zero": ("acestep_gen.h", "if (rq->adg && tn > 0) acd_adg(", "if (rq->adg) acd_adg("),
    "heun-corrector-interval-at-the-current-level": ("acestep_gen.h", "if (use_cfg && tn >= cfg_s && tn <= cfg_e) {", "if (use_cfg && t >= cfg_s && t <= cfg_e) {"),
    "heun-corrector-unguided": ("acestep_gen.h", "if (use_cfg && tn >= cfg_s && tn <= cfg_e) {", "if (0) {"),
    # --- the clamp and the smoothing ---------------------------------------------------------------------------------------------
    "vclamp-threshold": ("acestep_gen.h", "const double s = threshold * (sqrt(xn) + 1e-10) / (sqrt(vn) + 1e-10);", "const double s = 0.9 * threshold * (sqrt(xn) + 1e-10) / (sqrt(vn) + 1e-10);"),
    "vclamp-never-grows": ("acestep_gen.h", "if (s < 1.0) { const float f = (float)s;", "{ const float f = (float)s;"),
    "vclamp-norm-of-the-wrong-latents": ("acestep_gen.h", "xn += (double)x[k] * x[k]; }", "xn += (double)v[k] * v[k]; }"),
    "vclamp-skipped-in-the-corrector": ("acestep_gen.h", "if (clamp) acg_vclamp(v2, xp, NL, rq->vnorm);", ""),
    "vclamp-corrector-norm-of-the-current-latents": ("acestep_gen.h", "if (clamp) acg_vclamp(v2, xp, NL, rq->vnorm);", "if (clamp) acg_vclamp(v2, x, NL, rq->vnorm);"),
    "vema-weights-swapped": ("acestep_gen.h", "const float a = (float)(1.0 - f), b = (float)f;", "const float a = (float)f, b = (float)(1.0 - f);"),
    "vema-first-step": ("acestep_gen.h", "if (ema && have_prev) acg_vema(v, vprev, NL, rq->vema);", "if (ema) acg_vema(v, vprev, NL, rq->vema);"),
    "vema-state-is-not-kept": ("acestep_gen.h", "if (ema) { memcpy(vprev, v, NL * sizeof(float)); have_prev = 1; }", "if (ema) { have_prev = 1; }"),
    "vema-skipped-in-the-corrector": ("acestep_gen.h", "if (ema) acg_vema(v2, v, NL, rq->vema);", ""),
    "vema-before-the-clamp": ("acestep_gen.h", "if (clamp) acg_vclamp(v, x, NL, rq->vnorm);\n        if (ema && have_prev) acg_vema(v, vprev, NL, rq->vema);",
                              "if (ema && have_prev) acg_vema(v, vprev, NL, rq->vema);\n        if (clamp) acg_vclamp(v, x, NL, rq->vnorm);"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("refdirs", nargs="+")
    ap.add_argument("--dit", help="the DiT the cases were captured with (acestep-v15-base): passed to --gen-ref")
    ap.add_argument("--cc", default="gcc")
    ap.add_argument("--only", help="comma-separated mutant names")
    ap.add_argument("--skip", help="comma-separated mutant names to leave out")
    a = ap.parse_args()
    names = list(MUTANTS) if not a.only else a.only.split(",")
    names = [n for n in names if n not in (a.skip.split(",") if a.skip else [])]
    unknown = [n for n in names if n not in MUTANTS]
    if unknown:
        sys.exit(f"unknown mutant(s) {unknown}; known: {list(MUTANTS)}")
    refs = [(FLAG, r) for r in a.refdirs]
    extra = ("--dit", a.dit) if a.dit else ()
    with tempfile.TemporaryDirectory(prefix="acestep-mut-") as work:
        verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, None, "acestep_gen.h", FLAG, extra)
        print(f"{'unmodified':52s} {verdict}  {note}")
        if verdict != "pass":
            sys.exit("the unmodified source does not pass the oracle: fix that first")
        survivors = []
        for name in names:
            header, old, new, *count = MUTANTS[name]
            verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, (old, new, *count), header, FLAG, extra)
            caught = verdict == "fail"
            print(f"{name:52s} {'CAUGHT' if caught else 'SURVIVED (' + verdict + ')'}  {note[:100]}", flush=True)
            if not caught:
                survivors.append(name)
    print(f"\n{len(names) - len(survivors)} of {len(names)} mutants caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
