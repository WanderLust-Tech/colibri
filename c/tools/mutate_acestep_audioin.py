#!/usr/bin/env python3
"""Proof that the audio-in oracle can fail: break what acestep_gen.h adds for a reference clip, a cover and a repaint one line at a
time and see that `--gen-ref` says so.

    mutate_acestep_audioin.py MODEL_DIR REFDIR [REFDIR ...] [--cc gcc] [--only NAME,NAME]

MODEL_DIR is the real checkpoint (the tiny fixture has no encoder or tokenizer) and each REFDIR a case captured by
`acestep_ref_audio.py tasks`. A mutant is a small, plausible mistake in the pieces that join the VAE encoder, the tokenizer and the
sampler into the three tasks: how a reference clip is cut and encoded, the posterior's seed and its order, the instruction of each
task, what a cover conditions on, where a cover-noise start and a cover-strength switch fall, how a repaint's span, padding,
injection, crossfade and splice work. It is compiled into a temporary directory and run against `--gen-ref` on every case: CAUGHT
when any run fails. The unmodified source is run first and has to pass. Exits non-zero if a mutant survives.

Pass these cases (the order is the order they run in, so the quickest to fail come first): cover10, cover_odd (a source of 234.25
frames: the only one whose length is not a whole number of frames), cover_half, cover_noise, cover_ref, cover_short, cover_pad4,
ref_short, ref_long, repaint_mid, repaint_str, repaint_aggr, repaint_cons, repaint_tail, repaint_extend, repaint_lead, repaint_short.
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
    # --- the reference clip ---------------------------------------------------------------------------------------------
    "ref-segments-at-the-thirds-starts": ("const int64_t start[3] = { r0 / 2, seg + r0 / 2, 2 * seg + r2 / 2 };", "const int64_t start[3] = { 0, seg, 2 * seg };"),
    "ref-not-repeated": ("if (n < target) reps = (target + n - 1) / n;", "if (n < target) reps = 1;"),
    "ref-segment-length": ("#define ACG_REF_SEGMENT 10", "#define ACG_REF_SEGMENT 9"),
    "ref-thirds": ("const int64_t total = n * reps, seg = total / 3;", "const int64_t total = n * reps, seg = total / 2;"),
    # --- the posterior draws of the VAE encoder ----------------------------------------------------------------------------
    "vae-seed-xor": ("#define ACG_VAE_SEED_XOR 0x5EED0001u", "#define ACG_VAE_SEED_XOR 0x5EED0002u"),
    "vae-seed-ignores-the-request": ("acg_mt_seed(&vae_mt, (uint32_t)rq->seed ^ ACG_VAE_SEED_XOR);", "acg_mt_seed(&vae_mt, ACG_VAE_SEED_XOR);"),
    "vae-reference-draws-from-a-fresh-generator": ("F = acg_encode(e, refpcm, (int64_t)ACG_REF_SECONDS * ACG_RATE, &vae_mt, h, rl, err, errn);",
                                                   "AcgMt m2; acg_mt_seed(&m2, (uint32_t)rq->seed ^ ACG_VAE_SEED_XOR); F = acg_encode(e, refpcm, (int64_t)ACG_REF_SECONDS * ACG_RATE, &m2, h, rl, err, errn);"),
    "vae-window": ("#define ACG_VAE_ENC_CHUNK ((int64_t)30 * ACG_RATE)", "#define ACG_VAE_ENC_CHUNK ((int64_t)15 * ACG_RATE)"),
    # --- the prompts ---------------------------------------------------------------------------------------------------
    "cover-uses-the-default-instruction": ("instructions[task == ACG_TASK_REPAINT ? 1 : task == ACG_TASK_COVER ? 2 : 0]", "instructions[task == ACG_TASK_REPAINT ? 1 : 0]"),
    "repaint-uses-the-default-instruction": ("instructions[task == ACG_TASK_REPAINT ? 1 : task == ACG_TASK_COVER ? 2 : 0]", "instructions[task == ACG_TASK_COVER ? 2 : 0]"),
    "duration-in-the-metas-is-the-request's": ("cp = act_caption_prompt(instruction, rq->caption, bpm, rq->key, rq->timesig, meta_duration);", "cp = act_caption_prompt(instruction, rq->caption, bpm, rq->key, rq->timesig, rq->duration);"),
    "noncover-prompt-instruction": ("cp2 = act_caption_prompt(instructions[0], rq->caption, bpm, rq->key, rq->timesig, meta_duration);", "cp2 = act_caption_prompt(instructions[2], rq->caption, bpm, rq->key, rq->timesig, meta_duration);"),
    "noncover-prompt-not-cut": ("        if (cn2 > cn) cn2 = cn;", ""),
    # --- the source's latents ------------------------------------------------------------------------------------------
    "source-padded-with-zeros": ("if (lat_frames < T) acg_silence_slice(e, T - lat_frames, lat + (size_t)lat_frames * A);", "if (lat_frames < T) memset(lat + (size_t)lat_frames * A, 0, (size_t)(T - lat_frames) * A * sizeof(float));"),
    "source-length-rounded-up": ("lat_frames = (int)(Ns / ACG_HOP);", "lat_frames = (int)((Ns + ACG_HOP - 1) / ACG_HOP);"),
    # --- a cover ---------------------------------------------------------------------------------------------------------
    "cover-conditions-on-the-source-itself": ("        dit_src = hints;", "        dit_src = src;"),
    "cover-noise-level": ("const double level = 1.0 - rq->cover_noise;", "const double level = rq->cover_noise;"),
    "cover-noise-roles-swapped": ("x[k] = a * noise[k] + b * lat[k];", "x[k] = b * noise[k] + a * lat[k];"),
    "cover-noise-schedule-not-cut": ("sched = sched_all + best; n = n_all - best;", "sched = sched_all; n = n_all - best;"),
    "cover-strength-switch-late": ("const int cover_steps = cond2 ? (int)(n * cover_strength) : n;", "const int cover_steps = cond2 ? (int)(n * cover_strength) + 1 : n;"),
    "cover-strength-keeps-the-source-context": ("if (cond2 && i >= cover_steps) { cur = cond2; cur_ctx = ctx2; }", "if (cond2 && i >= cover_steps) { cur = cond2; }"),
    "noncover-source-is-the-clip": ("memcpy(ctx2 + (size_t)f * 2 * A, sil + (size_t)f * A, (size_t)A * sizeof(float));", "memcpy(ctx2 + (size_t)f * 2 * A, src + (size_t)f * A, (size_t)A * sizeof(float));"),
    # --- a repaint: the span and the padding ---------------------------------------------------------------------------------
    "repaint-no-padding-before": ("lead = (int64_t)(left * ACG_RATE); tail = (int64_t)(right * ACG_RATE);", "lead = 0; tail = (int64_t)(right * ACG_RATE);"),
    "repaint-no-padding-after": ("lead = (int64_t)(left * ACG_RATE); tail = (int64_t)(right * ACG_RATE);", "lead = (int64_t)(left * ACG_RATE); tail = 0;"),
    "repaint-default-end": ("const double end = rq->repaint_end > 0 ? rq->repaint_end : sd;", "const double end = rq->repaint_end > 0 ? rq->repaint_end : sd - 1.0;"),
    "repaint-span-start-rounded-up": ("span_a = (int)acg_floordiv(sp_start * ACG_RATE, ACG_HOP);", "span_a = (int)ceil(sp_start * ACG_RATE / ACG_HOP);"),
    "repaint-span-end-short": ("span_b = (int)acg_floordiv(sp_end * ACG_RATE, ACG_HOP);", "span_b = (int)acg_floordiv(sp_end * ACG_RATE, ACG_HOP) - 1;"),
    "repaint-span-not-silenced": ("for (int f = span_a; f < span_b; f++) { memcpy(src + (size_t)f * A, e->silence + (size_t)(f % e->silence_frames) * 64, 64 * sizeof(float)); rmask[f] = 1; }",
                                  "for (int f = span_a; f < span_b; f++) { rmask[f] = 1; }"),
    "repaint-mask-in-the-context": ("ctx[(size_t)f * 2 * A + A + c] = 1.0f;", "ctx[(size_t)f * 2 * A + A + c] = rmask[f] ? 1.0f : 0.0f;"),
    # --- a repaint: the sampler ------------------------------------------------------------------------------------------------
    "injection-skipped": ("if (task == ACG_TASK_REPAINT && i < n - 1 && i < cutoff) {", "if (task == ACG_TASK_REPAINT && i < n - 1 && i < 0) {"),
    "injection-after-the-last-step": ("if (task == ACG_TASK_REPAINT && i < n - 1 && i < cutoff) {", "if (task == ACG_TASK_REPAINT && i < cutoff) {"),
    "injection-cutoff-truncated": ("const int cutoff = (int)acg_pyround(inject * n);", "const int cutoff = (int)(inject * n);"),
    "injection-time": ("const float tn = sched[i + 1], a = tn, b = (float)(1.0 - (double)tn);", "const float tn = sched[i], a = tn, b = (float)(1.0 - (double)tn);"),
    "injection-inside-the-span": ("                if (!rmask[f])\n                    for (int c = 0; c < A; c++) x[(size_t)f * A + c] = a * noise", "                if (rmask[f])\n                    for (int c = 0; c < A; c++) x[(size_t)f * A + c] = a * noise"),
    "boundary-blend-skipped": ("if (task == ACG_TASK_REPAINT && cf_frames > 0) acg_blend_boundary(x, lat, rmask, T, A, cf_frames);", "if (task == ACG_TASK_REPAINT && cf_frames > 100) acg_blend_boundary(x, lat, rmask, T, A, cf_frames);"),
    "boundary-ramp-direction": ("for (int t = fs; t < left; t++) m[t] = acg_ramp(0.f, 1.f, left - fs, t - fs);", "for (int t = fs; t < left; t++) m[t] = acg_ramp(1.f, 0.f, left - fs, t - fs);"),
    "boundary-frames-truncated": ("*inject = inv; *cf_frames = (int)acg_pyround(25 * inv); *wav_cf = 0.05 * inv;", "*inject = inv; *cf_frames = (int)(25 * inv); *wav_cf = 0.05 * inv;"),
    # --- a repaint: the modes and the waveform splice --------------------------------------------------------------------------
    "conservative-injection": ("if (mode == ACG_REPAINT_CONSERVATIVE) { *inject = 1.0; *cf_frames = 25; *wav_cf = 0.05; return; }", "if (mode == ACG_REPAINT_CONSERVATIVE) { *inject = 0.5; *cf_frames = 25; *wav_cf = 0.05; return; }"),
    "aggressive-injection": ("if (mode == ACG_REPAINT_AGGRESSIVE) { *inject = 0.0; *cf_frames = 0; *wav_cf = 0.0; return; }", "if (mode == ACG_REPAINT_AGGRESSIVE) { *inject = 0.5; *cf_frames = 0; *wav_cf = 0.0; return; }"),
    "balanced-strength-default": ("if (strength <= 0) strength = 0.5;", "if (strength <= 0) strength = 0.3;"),
    "splice-skipped": ("if (task == ACG_TASK_REPAINT && rq->repaint_mode != ACG_REPAINT_AGGRESSIVE) acg_splice_waveform(", "if (task == ACG_TASK_REPAINT && rq->repaint_mode == 99) acg_splice_waveform("),
    "splice-in-aggressive-mode": ("if (task == ACG_TASK_REPAINT && rq->repaint_mode != ACG_REPAINT_AGGRESSIVE) acg_splice_waveform(", "if (task == ACG_TASK_REPAINT) acg_splice_waveform("),
    "splice-span-shifted": ("int64_t a = (int64_t)(start_s * (double)ACG_RATE), b = (int64_t)(end_s * (double)ACG_RATE);", "int64_t a = (int64_t)(start_s * (double)ACG_RATE) + 4800, b = (int64_t)(end_s * (double)ACG_RATE);"),
    "splice-crossfade-width": ("*inject = inv; *cf_frames = (int)acg_pyround(25 * inv); *wav_cf = 0.05 * inv;", "*inject = inv; *cf_frames = (int)acg_pyround(25 * inv); *wav_cf = 0.05;"),
    "splice-ramp-direction": ("if (i >= fs && i < a) m = acg_ramp(0.f, 1.f, (int)(a - fs), (int)(i - fs));", "if (i >= fs && i < a) m = acg_ramp(1.f, 0.f, (int)(a - fs), (int)(i - fs));"),
}

# Deliberately NOT here, because no output can tell them from the original:
#  - the half-way split of torch's linspace (acg_ramp's `idx < half`): both branches give the same value to float rounding;
#  - the injection's source being the clean latents against the patched ones (src): outside the span, the only frames it writes,
#    they are the same numbers;
#  - a silent source (the silence latent stands in for its encoding, as upstream's is_silence branch does): no capture is silent,
#    and the engine's own tests (c/tests/test_acestep_cli.py) drive it.


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
        print(f"{'unmodified':44s} {verdict}  {note}")
        if verdict != "pass":
            sys.exit("the unmodified source does not pass the oracle: fix that first")
        survivors = []
        for name in names:
            verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, MUTANTS[name], HEADER, FLAG)
            caught = verdict == "fail"
            print(f"{name:44s} {'CAUGHT' if caught else 'SURVIVED (' + verdict + ')'}  {note[:100]}", flush=True)
            if not caught:
                survivors.append(name)
    print(f"\n{len(names) - len(survivors)} of {len(names)} mutants caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
