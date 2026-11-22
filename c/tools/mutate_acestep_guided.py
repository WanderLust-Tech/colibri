#!/usr/bin/env python3
"""Proof that the guided DiT's oracle can fail: break what the base and SFT DiTs add one line at a time and see that `--gen-ref` says so.

    mutate_acestep_guided.py MODEL_DIR REFDIR [REFDIR ...] [--cc gcc] [--only NAME,NAME] [--skip NAME,NAME]

MODEL_DIR is a checkpoint with acestep-v15-base/ (the real one, or the tiny one built with `make_acestep_tiny.py --audio-in --base`) and each
REFDIR a case captured by `acestep_ref_audio.py tasks --dit acestep-v15-base`. A mutant is a small, plausible mistake in what is not turbo:
the guidance (APG's momentum, its norm cut and its projection, ADG's angle, the interval), the schedule, the null condition, and the three
tasks (the instruction each builds, what the context is, whether a lego's span is shown or hidden). It is compiled into a temporary
directory and run against `--gen-ref --dit acestep-v15-base` on every case: CAUGHT when any run fails. The unmodified source is run first
and has to pass. Exits non-zero if a mutant survives.

Pass these cases (quickest to fail first): b_t2m, b_t2m_shift3, b_t2m_nocfg, b_t2m_interval, b_t2m_adg, b_t2m_steps40, b_cover_half,
b_extract, b_extract_default, b_lego, b_lego_tail, b_complete, b_complete_default, b_complete_longer.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate_acestep_vae as base  # noqa: E402

FLAG = "--gen-ref"
EXTRA = ("--dit", "acestep-v15-base")

# name: (header, text, its replacement[, how many times it occurs])
MUTANTS = {
    # --- APG, the default guidance ---------------------------------------------------------------------------------------------
    "apg-momentum": ("acestep_dit.h", "(-0.75f) * run[k];", "(-0.5f) * run[k];"),
    "apg-no-momentum": ("acestep_dit.h", "run[k] = (cond[k] - unc[k]) + (-0.75f) * run[k];", "run[k] = (cond[k] - unc[k]);"),
    "apg-norm-threshold": ("acestep_dit.h", "q = 2.5f / nrm;", "q = 5.0f / nrm;"),
    "apg-no-norm-cut": ("acestep_dit.h", "sf[c] = q < 1.0f ? q : 1.0f;", "sf[c] = 1.0f;"),
    "apg-keeps-the-parallel-part": ("acestep_dit.h", "(orth + 0.0f * (float)(par[c] * v1))", "(orth + 1.0f * (float)(par[c] * v1))"),
    "apg-scale": ("acestep_dit.h", "(float)(scale - 1.0) * (orth", "(float)(scale) * (orth"),
    "apg-over-channels": ("acestep_dit.h", "dn[c] += r * r; cn[c] += v * v;", "dn[c] += r * r; cn[c] += r * r;"),
    # --- ADG ---------------------------------------------------------------------------------------------------------------------
    "adg-angle-clip": ("acestep_dit.h", "const double clip = 3.14 / 6;", "const double clip = 3.14 / 3;"),
    "adg-no-clip": ("acestep_dit.h", "if (tn > clip) tn = clip;", ""),
    "adg-weight-offset": ("acestep_dit.h", "weight = weight * (weight > 0) + 1e-3;", "weight = weight * (weight > 0) + 0.5;"),
    "adg-uncond-prediction": ("acestep_dit.h", "lu[c] = x[o + c] - sigma * unc[o + c];", "lu[c] = x[o + c] - unc[o + c];"),
    "adg-perpendicular": ("acestep_dit.h", "perp[c] = diff[c] - k * lu[c];", "perp[c] = diff[c] - 0.5f * k * lu[c];"),
    # --- the schedule and the null condition -----------------------------------------------------------------------------------
    "schedule-shift": ("acestep_dit.h", "t[i] = shift * t[i] / (1.0f + (shift - 1.0f) * t[i]);", "t[i] = shift * t[i] / (1.0f + shift * t[i]);"),
    "schedule-length": ("acestep_dit.h", "const int n = steps + 1;", "const int n = steps;"),
    "null-condition-row-dropped": ("acestep_dit.h", "for (int i = 0; i < like->len; i++) memcpy(raw", "for (int i = 0; i < like->len - 1; i++) memcpy(raw"),
    # --- the sampler's use of guidance -----------------------------------------------------------------------------------------
    "guidance-default": ("acestep_gen.h", "rq->guidance > 0 ? rq->guidance : 7.0;", "rq->guidance > 0 ? rq->guidance : 5.0;"),
    "interval-strict": ("acestep_gen.h", "sched[i] >= cfg_s && sched[i] <= cfg_e", "sched[i] > cfg_s && sched[i] < cfg_e"),
    "interval-default-end": ("acestep_gen.h", "rq->cfg_end > 0 ? rq->cfg_end : 1.0;", "rq->cfg_end > 0 ? rq->cfg_end : 0.9;"),
    "adg-and-apg-swapped": ("acestep_gen.h", "if (rq->adg) acd_adg(", "if (!rq->adg) acd_adg("),
    "null-branch-context": ("acestep_gen.h", "acd_forward(d, ncur, x, cur_ctx,", "acd_forward(d, ncur, x, ctx,"),
    "cover-switch-keeps-the-null-condition": ("acestep_gen.h", "cur = cond2; ncur = ncond2; cur_ctx = ctx2;", "cur = cond2; cur_ctx = ctx2;"),
    # --- extract, lego, complete -----------------------------------------------------------------------------------------------
    "lego-span-hidden": ("acestep_gen.h", "if (task == ACG_TASK_REPAINT) memcpy(src + (size_t)f * A,", "if (task == ACG_TASK_REPAINT || task == ACG_TASK_LEGO) memcpy(src + (size_t)f * A,"),
    "extract-context-silent": ("acestep_gen.h", "if (text_like) acg_silence_slice(e, T, src);", "if (text_like || task == ACG_TASK_EXTRACT) acg_silence_slice(e, T, src);"),
    "complete-without-source-duration": ("acestep_gen.h", "meta_duration = info->src_seconds;", "meta_duration = rq->duration > 0 ? rq->duration : info->src_seconds;"),
    "extract-instruction-is-lego": ("acestep_gen.h", 'task == ACG_TASK_EXTRACT ? "Extract the %s track from the audio:" : "Generate the %s track based on the audio context:", name);',
                                    'task == ACG_TASK_EXTRACT ? "Generate the %s track based on the audio context:" : "Extract the %s track from the audio:", name);'),
    "track-lower-cased": ("acestep_gen.h", "name[m++] = (char)toupper((unsigned char)*p);", "name[m++] = (char)tolower((unsigned char)*p);"),
    "complete-join": ("acestep_gen.h", 'm += (size_t)snprintf(list + m, sizeof list - m, " | ");', 'm += (size_t)snprintf(list + m, sizeof list - m, ", ");'),
    "complete-default-instruction": ("acestep_gen.h", 'if (!m) return "Complete the input track:";', 'if (!m) return "Complete the input track with :";'),
    "extract-default-instruction": ("acestep_gen.h", 'return task == ACG_TASK_EXTRACT ? "Extract the track from the audio:" : "Generate the track based on the audio context:";',
                                    'return task == ACG_TASK_EXTRACT ? "Extract the track from the audio:" : "Extract the track from the audio:";'),
    "steps-cap-for-guided": ("acestep_gen.h", "if (d->turbo && rq->steps > ACG_MAX_STEPS_TURBO)", "if (rq->steps > ACG_MAX_STEPS_TURBO)"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("refdirs", nargs="+")
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
    with tempfile.TemporaryDirectory(prefix="acestep-mut-") as work:
        verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, None, "acestep_gen.h", FLAG, EXTRA)
        print(f"{'unmodified':44s} {verdict}  {note}")
        if verdict != "pass":
            sys.exit("the unmodified source does not pass the oracle: fix that first")
        survivors = []
        for name in names:
            header, old, new, *count = MUTANTS[name]
            verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, (old, new, *count), header, FLAG, EXTRA)
            caught = verdict == "fail"
            print(f"{name:44s} {'CAUGHT' if caught else 'SURVIVED (' + verdict + ')'}  {note[:100]}", flush=True)
            if not caught:
                survivors.append(name)
    print(f"\n{len(names) - len(survivors)} of {len(names)} mutants caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
