#!/usr/bin/env python3
"""Proof that the flow-edit oracle can fail: break the paired source and target loop one line at a time and see that `--gen-ref` says so.

    mutate_acestep_flowedit.py MODEL_DIR REFDIR [REFDIR ...] [--dit acestep-v15-base] [--cc gcc] [--only NAME,NAME] [--skip NAME,NAME]

MODEL_DIR is a checkpoint and each REFDIR a case captured by `acestep_ref_audio.py tasks` with flow-edit (the `s_fe*` cases for the turbo DiT; the `b_fe*`
cases with `--dit acestep-v15-base`, which this tool then needs too). A mutant is a small, plausible mistake in what flow-edit adds: the noise's seed and
weights, which prompt goes with which branch, the sign of the delta, the guidance momentum of each branch and when it moves, the averaging of the draws, the
window's edges, the switch to the target alone, the clamp and the smoothing, the source prompt. It is compiled into a temporary directory and run against
`--gen-ref` on every case: CAUGHT when any run fails. The unmodified source is run first and has to pass. A few mutants only a guided DiT can show (the
guidance momentum, the null conditions): run the tool once on the turbo cases and once on the base ones, and a mutant has to be caught by one of the two.
Exits non-zero if a mutant survives its run.

One mutant is not in the list because it is equivalent: swapping the two branches' null conditions. A null condition is the null embedding repeated to the
length of the condition it stands for, every row the same vector, and cross-attention over identical rows gives the same answer whatever their number.
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
    # --- the noise and the two trajectories ------------------------------------------------------------------------------------------
    "noise-seed": ("acestep_gen.h", "acg_mt_seed(&mt, (uint32_t)rq->seed);", "acg_mt_seed(&mt, (uint32_t)rq->seed ^ 1u);"),
    "noise-weights-swapped": ("acestep_gen.h", "b[ZS][k] = a1 * lat[k] + a2 * b[FN][k];", "b[ZS][k] = a2 * lat[k] + a1 * b[FN][k];"),
    "target-without-the-edit": ("acestep_gen.h", "b[ZG][k] = b[ZT][k] + b[ZS][k] - lat[k];", "b[ZG][k] = b[ZS][k];"),
    "target-adds-the-clip": ("acestep_gen.h", "b[ZG][k] = b[ZT][k] + b[ZS][k] - lat[k];", "b[ZG][k] = b[ZT][k] + b[ZS][k] + lat[k];"),
    "start-from-nothing": ("acestep_gen.h", "memcpy(b[ZT], lat, NL * sizeof(float));", "memset(b[ZT], 0, NL * sizeof(float));"),
    "prompts-swapped": ("acestep_gen.h", "acd_forward(d, br ? cond_t : cond_s, z, ctx, T, tc, tc, v);", "acd_forward(d, br ? cond_s : cond_t, z, ctx, T, tc, tc, v);"),
    "delta-sign": ("acestep_gen.h", "b[ZT][k] += dt * (b[VTS][k] - b[VSS][k]);", "b[ZT][k] += dt * (b[VSS][k] - b[VTS][k]);"),
    "delta-is-the-target-only": ("acestep_gen.h", "b[ZT][k] += dt * (b[VTS][k] - b[VSS][k]);", "b[ZT][k] += dt * b[VTS][k];"),
    "step-from-the-wrong-level": ("acestep_gen.h", "const float tc = sched[i], tp = i < n - 1 ? sched[i + 1] : 0.0f;", "const float tc = sched[i], tp = i < n - 1 ? sched[i + 1] : sched[i];"),
    # --- guidance ------------------------------------------------------------------------------------------------------------------
    "momentum-shared-by-the-branches": ("acestep_gen.h", "*mom = br ? b[MT] : b[MS],", "*mom = b[MS],"),
    "momentum-not-reset-per-draw": ("acestep_gen.h", "memcpy(b[RUN], mom, NL * sizeof(float));          /* every draw starts from the momentum the step began with */", "(void)0;"),
    "momentum-does-not-move": ("acestep_gen.h", "b[MS][k] = b[DSS][k] / fn + (-0.75f) * b[MS][k]; b[MT][k] = b[DTS][k] / fn + (-0.75f) * b[MT][k];", "(void)k;"),
    "momentum-moves-by-the-sum": ("acestep_gen.h", "b[MS][k] = b[DSS][k] / fn + (-0.75f) * b[MS][k]; b[MT][k] = b[DTS][k] / fn + (-0.75f) * b[MT][k];",
                                  "b[MS][k] = b[DSS][k] + (-0.75f) * b[MS][k]; b[MT][k] = b[DTS][k] + (-0.75f) * b[MT][k];"),
    "interval-at-the-next-level": ("acestep_gen.h", "const int now_cfg = use_cfg && tcf >= cfg_s && tcf <= cfg_e;", "const int now_cfg = use_cfg && (double)tp >= cfg_s && (double)tp <= cfg_e;"),
    "guidance-off": ("acestep_gen.h", "const int now_cfg = use_cfg && tcf >= cfg_s && tcf <= cfg_e;", "const int now_cfg = 0;"),
    # --- the window and the averaging --------------------------------------------------------------------------------------------
    "window-starts-at-0": ("acestep_gen.h", "const int i_min = (int)((double)n * rq->edit_nmin), i_max", "const int i_min = 0, i_max"),
    "window-ends-at-the-end": ("acestep_gen.h", "i_max = (int)((double)n * nmax_d);", "i_max = n;"),
    "draws-not-averaged": ("acestep_gen.h", "for (size_t k = 0; k < NL; k++) { b[VSS][k] /= fn; b[VTS][k] /= fn; }", "(void)fn;"),
    "one-draw": ("acestep_gen.h", "for (int r = 0; r < navg; r++) {", "for (int r = 0; r < 1; r++) {"),
    # --- the clamp and the smoothing ---------------------------------------------------------------------------------------------
    "clamp-norm-of-the-edit": ("acestep_gen.h", "if (clamp) acg_vclamp(v, z, NL, rq->vnorm);", "if (clamp) acg_vclamp(v, b[ZT], NL, rq->vnorm);"),
    "clamp-skipped-after-the-window": ("acestep_gen.h", "if (clamp) acg_vclamp(b[VT], b[XT], NL, rq->vnorm);", ""),
    "smoothing-skipped-for-the-source": ("acestep_gen.h", "if (ema && have_ps) acg_vema(b[VSS], b[PS], NL, rq->vema);", ""),
    "smoothing-skipped-for-the-target": ("acestep_gen.h", "if (ema && have_pt) acg_vema(b[VTS], b[PT], NL, rq->vema);", ""),
    "smoothing-skipped-after-the-window": ("acestep_gen.h", "if (ema && have_pt) acg_vema(b[VT], b[PT], NL, rq->vema);", ""),
    "smoothing-state-of-the-target-not-kept": ("acestep_gen.h", "memcpy(b[PT], b[VTS], NL * sizeof(float)); have_ps = have_pt = 1;", "have_ps = have_pt = 1;"),
    # --- the switch to the target alone ------------------------------------------------------------------------------------------
    "switch-without-the-noised-clip": ("acestep_gen.h", "b[XT][k] = b[ZT][k] + (a1 * lat[k] + a2 * b[FN][k]) - lat[k];", "b[XT][k] = b[ZT][k];"),
    "switch-draws-no-noise": ("acestep_gen.h", "if (!have_xt) {\n                    acg_randn_mt(&mt, NL, b[FN]);", "if (!have_xt) {"),
    "after-window-momentum-of-the-source": ("acestep_gen.h", "acd_apg(b[VG], b[VT], b[VU], T, A, guidance, b[MT]);", "acd_apg(b[VG], b[VT], b[VU], T, A, guidance, b[MS]);"),
    # --- the source's prompt -------------------------------------------------------------------------------------------------------
    "source-lyrics-default-is-instrumental": ("acestep_gen.h", 'rq->edit_lyrics ? rq->edit_lyrics : ""', 'rq->edit_lyrics ? rq->edit_lyrics : "[Instrumental]"'),
    "source-caption-is-the-targets": ("acestep_gen.h", 'rq->edit_caption ? rq->edit_caption : ""', "rq->caption"),
    "context-is-the-clip": ("acestep_gen.h", "if (text_like) acg_silence_slice(e, T, src);", "if (text_like && !morph) acg_silence_slice(e, T, src);"),
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
