#!/usr/bin/env python3
"""Proof that the end-to-end oracle can fail: break acestep_gen.h one line at a time and see that `--gen-ref` says so.

    mutate_acestep_gen.py MODEL_DIR REFDIR [REFDIR ...] [--cc gcc] [--only NAME,NAME]

MODEL_DIR is a checkpoint (the tiny fixture of make_acestep_tiny.py is enough) and each REFDIR a captured case of it
(tools/acestep_ref.py capture ... --cases ...). A mutant is a small, plausible mistake in the pieces acestep_gen.h adds to
the three networks: torch's generator, the silence latent and its slices, the clip's length, the context, the sampler's
arguments, the peak limit. It is compiled into a temporary directory and run against `--gen-ref` on every case: it is
CAUGHT when any run fails. A mistake in a clip's length shows only on a duration with a fraction (round907, floor91) or
under the 128-frame floor (short5), which is why several cases are passed. The unmodified source is run first and has to
pass. Exits non-zero if any mutant survives.

Pass these cases: inst10, vocal10, odd9_nodcw, shift3, round907, floor91, short5, meta10, empty10 and long30 (the one with
several VAE windows).
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate_acestep_vae as base  # noqa: E402

HEADER = "acestep_gen.h"
FLAG = "--gen-ref"

# name: (text in acestep_gen.h, its replacement[, how many times it occurs])
MUTANTS = {
    # --- torch's generator -----------------------------------------------------------------------------------------
    "noise-seed-offset": ("acg_mt_seed(&m, seed);\n    if (n < 16)", "acg_mt_seed(&m, seed + 1);\n    if (n < 16)"),
    "noise-seed-recurrence": ("1812433253u * (m->s[j - 1]", "1812433252u * (m->s[j - 1]"),
    "noise-tempering": ("y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680u;", "y ^= y >> 10; y ^= (y << 7) & 0x9d2c5680u;"),
    "noise-twist-constant": ("0x9908b0dfu", "0x9908b0deu"),
    "noise-24-bits": ("(float)(acg_mt_next(&m) & 0xffffff) * (1.0f / 16777216.0f)", "(float)(acg_mt_next(&m) & 0x7fffff) * (1.0f / 8388608.0f)", 2),
    "noise-sin-cos-swapped": ("d[j] = radius * cosf(theta);\n        d[j + 8] = radius * sinf(theta);", "d[j] = radius * sinf(theta);\n        d[j + 8] = radius * cosf(theta);"),
    "noise-uniforms-swapped": ("const float u1 = 1.0f - d[j], u2 = d[j + 8];", "const float u1 = 1.0f - d[j + 8], u2 = d[j];"),
    "noise-radius-factor": ("sqrtf(-2.0f * logf(u1))", "sqrtf(-1.0f * logf(u1))"),
    "noise-theta-factor": ("2.0 * 3.14159265358979323846 * (double)u2", "3.14159265358979323846 * (double)u2"),
    # --- the silence latent ------------------------------------------------------------------------------------------
    "silence-not-transposed": ("out[(size_t)t * 64 + c] = v;", "out[(size_t)c * N + t] = v;"),
    "silence-slice-offset": ("e->silence + (size_t)(i % e->silence_frames) * 64", "e->silence + (size_t)((i + 1) % e->silence_frames) * 64"),
    "timbre-frames": ("#define ACG_TIMBRE_FRAMES 750", "#define ACG_TIMBRE_FRAMES 749"),
    "timbre-from-the-clip": ("acg_silence_slice(e, ACG_TIMBRE_FRAMES, timbre);", "acg_silence_slice(e, ACG_TIMBRE_FRAMES, timbre); acg_randn(1, (size_t)ACG_TIMBRE_FRAMES * A, timbre);"),
    # --- the length of the clip ----------------------------------------------------------------------------------------
    "clip-duration-not-rounded": ("char b[64]; snprintf(b, sizeof b, \"%.1f\", x);\n    return strtod(b, NULL);", "return x;"),
    "clip-frames-rounded-up": ("int64_t f = s / ACG_HOP;", "int64_t f = (s + ACG_HOP - 1) / ACG_HOP;"),
    "clip-frames-rounded-nearest": ("int64_t f = s / ACG_HOP;", "int64_t f = (s + ACG_HOP / 2) / ACG_HOP;"),
    "clip-floor": ("#define ACG_MIN_FRAMES 128 ", "#define ACG_MIN_FRAMES 127 "),
    "clip-no-floor": ("*frames = (int)(f < ACG_MIN_FRAMES ? ACG_MIN_FRAMES : f);", "*frames = (int)f;"),
    "clip-not-cut-to-the-length-asked": ("info->samples = (int)total < want_samples ? (int)total : want_samples;", "info->samples = (int)total;"),
    # --- the context and the prompts --------------------------------------------------------------------------------
    "mask-not-true": ("ctx[(size_t)f * 2 * A + A + c] = 1.0f;", "ctx[(size_t)f * 2 * A + A + c] = 0.0f;"),
    "source-not-silence": ("memcpy(ctx + (size_t)f * 2 * A, src + (size_t)f * A, (size_t)A * sizeof(float));", "memcpy(ctx + (size_t)f * 2 * A, noise + (size_t)f * A, (size_t)A * sizeof(float));"),
    "lyrics-default": ("rq->lyrics ? rq->lyrics : \"[Instrumental]\"", "rq->lyrics ? rq->lyrics : \"\""),
    "language-default": ("rq->language && *rq->language ? rq->language : \"unknown\"", "rq->language && *rq->language ? rq->language : \"en\""),
    "empty-lyrics-replaced": ("rq->lyrics ? rq->lyrics : \"[Instrumental]\"", "rq->lyrics && *rq->lyrics ? rq->lyrics : \"[Instrumental]\""),
    "duration-in-metas": ("cp = act_caption_prompt(NULL, rq->caption, bpm, rq->key, rq->timesig, rq->duration);", "cp = act_caption_prompt(NULL, rq->caption, bpm, rq->key, rq->timesig, rq->duration + 1.0);"),
    "bpm-dropped": ("if (rq->bpm > 0) snprintf(bpm, sizeof bpm, \"%d\", rq->bpm);", ""),
    # --- the sampler's arguments and the decode ------------------------------------------------------------------
    "shift-ignored": ("acd_schedule(rq->steps, (float)rq->shift, sched)", "acd_schedule(rq->steps, 1.0f, sched)"),
    "dcw-always-on": ("AcdSampleOpts so = { rq->dcw ? 1 : 0, 0.05, 0.02 };", "AcdSampleOpts so = { 1, 0.05, 0.02 };"),
    "dcw-strengths": ("AcdSampleOpts so = { rq->dcw ? 1 : 0, 0.05, 0.02 };", "AcdSampleOpts so = { rq->dcw ? 1 : 0, 0.02, 0.05 };"),
    "network-told-r-zero": ("acd_forward(d, cond, x, ctx, T, sched[i], sched[i], v);", "acd_forward(d, cond, x, ctx, T, sched[i], 0.0f, v);"),
    "network-told-the-wrong-time": ("acd_forward(d, cond, x, ctx, T, sched[i], sched[i], v);", "acd_forward(d, cond, x, ctx, T, sched[n - 1 - i], sched[n - 1 - i], v);"),
    "step-index": ("acd_sample_step(x, v, i, n, sched, T, A, &so, nx);", "acd_sample_step(x, v, i, n + 1, sched, T, A, &so, nx);"),
    "peak-limit-skipped": ("info->peak = acv_normalize(wav, total * e->vae->ach);", "info->peak = 0;"),
    "decode-overlap-too-small": ("#define ACG_VAE_OVERLAP 64", "#define ACG_VAE_OVERLAP 1"),
}

# Deliberately NOT here, because no output can tell them from the original:
#  - dropping the tail of acg_randn (the last 16 values drawn again): a clip's noise has frames * 64 values, always a
#    multiple of 16, so the branch never runs for a clip;
#  - the VAE overlap 64 against any value of 12 or more, the same equivalence as the decoder's own mutants (see
#    mutate_acestep_vae.py); "decode-overlap-too-small" is the same family made observable.


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
