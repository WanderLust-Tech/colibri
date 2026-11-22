#!/usr/bin/env python3
"""Proof that the planner's oracles can fail: break acestep_lm.h, the codes' road into the DiT and the planner's wiring one line at a time and see
that the oracles say so.

    mutate_acestep_lm.py MODEL_DIR LMREF [LMREF ...] --codes CODEREF [CODEREF ...] [--cc gcc] [--only NAME,NAME] [--skip NAME,NAME]

MODEL_DIR is a checkpoint with a planner (the real one, or the tiny one built with `make_acestep_tiny.py --audio-in --lm`), each LMREF a case
captured by `acestep_ref_lm.py capture` and each CODEREF a case with `audio_code_string` captured by `acestep_ref_audio.py tasks` (codes10,
codes_short, codes_odd, codes_half, codes_noise). A mutant is a small, plausible mistake: in the model (a head norm, RoPE, the attention's scale,
its window, its grouping, the final norm, the head's rows), in the guidance and the end-token rule, in the prompts and the metadata's YAML, in the
sampler's filters and draw, and in how codes become the DiT's context (the FSQ's digits, the padding). It is compiled into a temporary directory and
run against `--lm-selftest` (the sampler, which torch's multinomial leaves no oracle for), then every `--lm-ref`, then every `--gen-ref`: CAUGHT when
any run fails. The unmodified source is run first and has to pass. Exits non-zero if a mutant survives.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate_acestep_vae as base  # noqa: E402

# name: (header, text, its replacement[, how many times it occurs])
MUTANTS = {
    # --- the model -----------------------------------------------------------------------------------------------------------------
    "decode-q-norm-skipped": ("acestep_lm.h", "        acd_head_norm(q, nr, H, hd, l->sa.qn, m->eps);\n", ""),
    "decode-k-norm-skipped": ("acestep_lm.h", "        acd_head_norm(kk, nr, Hk, hd, l->sa.kn, m->eps);\n", ""),
    "prefill-q-norm-skipped": ("acestep_lm.h", "        acd_head_norm(q, n, H, hd, l->sa.qn, m->eps);\n", ""),
    "prefill-k-norm-skipped": ("acestep_lm.h", "        acd_head_norm(kk, n, Hk, hd, l->sa.kn, m->eps);\n", ""),
    "decode-position-off-by-one": ("acestep_lm.h", "m->cs + (size_t)pos[r] * hd, m->sn + (size_t)pos[r] * hd);", "m->cs + (size_t)(pos[r] + 1) * hd, m->sn + (size_t)(pos[r] + 1) * hd);", 2),
    "attention-scale": ("acestep_lm.h", "const float scale = 1.0f / sqrtf((float)hd);\n    for (int h = 0; h < H; h++) {", "const float scale = 1.1f / sqrtf((float)hd);\n    for (int h = 0; h < H; h++) {"),
    "attention-misses-the-newest-key": ("acestep_lm.h", "pos[r] + 1, sc + (size_t)r", "pos[r], sc + (size_t)r"),
    "attention-kv-grouping": ("acestep_lm.h", "(size_t)(h / group) * hd", "(size_t)(h % Hk) * hd", 2),
    "prefill-not-causal": ("acestep_lm.h", "n, H, Hk, hd, -1, 1);\n        acd_lin(o, cx, n, &l->sa.o);", "n, H, Hk, hd, -1, 0);\n        acd_lin(o, cx, n, &l->sa.o);"),
    "final-norm-skipped": ("acestep_lm.h", "acd_rmsnorm(hn, h, nr, D, m->norm, m->eps);", "memcpy(hn, h, (size_t)nr * D * sizeof(float));"),
    "head-row-off-by-one": ("acestep_lm.h", "m->embed + (size_t)ids[j] * D;", "m->embed + (size_t)(ids[j] + 1) * D;"),
    "embedding-of-the-last-token": ("acestep_lm.h", "alm_embed_row(m, m->audio0 + tok, x + (size_t)r * D);", "alm_embed_row(m, m->audio0 + tok + 1, x + (size_t)r * D);"),
    # --- guidance and the end token ------------------------------------------------------------------------------------------------
    "guidance-from-the-conditional": ("acestep_lm.h", "cfg[j] = ul[j] + (float)s->cfg_scale * (cl[j] - ul[j]);", "cfg[j] = cl[j] + (float)s->cfg_scale * (cl[j] - ul[j]);"),
    "guidance-scale-minus-one": ("acestep_lm.h", "cfg[j] = ul[j] + (float)s->cfg_scale * (cl[j] - ul[j]);", "cfg[j] = ul[j] + (float)(s->cfg_scale - 1.0) * (cl[j] - ul[j]);"),
    "end-token-one-code-early": ("acestep_lm.h", "if (count < target) cfg[ALM_CODES] = -INFINITY;", "if (count < target - 1) cfg[ALM_CODES] = -INFINITY;"),
    "codes-not-blocked-after-the-target": ("acestep_lm.h", "else { for (int j = 0; j < ALM_CODES; j++) cfg[j] = -INFINITY; }", "else { }"),
    "uncond-prompt-runs-from-the-conditional-position": ("acestep_lm.h", "int pos[2] = { nc, nu };", "int pos[2] = { nc, nc };"),
    # --- the prompts and the metadata ----------------------------------------------------------------------------------------------
    "no-user-input-text": ("acestep_lm.h", '#define ALM_NO_INPUT "NO USER INPUT"', '#define ALM_NO_INPUT "NO INPUT"'),
    "empty-thinking": ("acestep_lm.h", 'uncond ? "<think>\\n\\n</think>" : cot;', 'uncond ? "<think>\\n</think>" : cot;'),
    "metadata-key-order": ("acestep_meta.h", "    for (int f = 0; f < AMF_N; f++) {\n        if (!m->have[f] || !m->s[f]) continue;", "    for (int f = AMF_N - 1; f >= 0; f--) {\n        if (!m->have[f] || !m->s[f]) continue;"),
    "metadata-duration-rounded": ("acestep_meta.h", 'snprintf(nb, sizeof nb, "%d", (int)duration); alm_meta_set(&m, AMF_DURATION, nb);', 'snprintf(nb, sizeof nb, "%d", (int)(duration + 0.5)); alm_meta_set(&m, AMF_DURATION, nb);'),
    "metadata-slash-four-kept": ("acestep_meta.h", 'if (l >= 2 && !strcmp(v + l - 2, "/4"))', "if (0)"),
    "metadata-language-line-dropped": ("acestep_meta.h", "if (language && *language) alm_meta_set(&m, AMF_LANGUAGE, language);", ""),
    "yaml-no-is-a-boolean": ("acestep_meta.h", '"yes", "Yes", "YES", "no", "No", "NO", ', '"yes", "Yes", "YES", '),
    "instruction-text": ("acestep_lm.h", '#define ALM_INSTRUCTION "Generate audio semantic tokens based on the given conditions:"', '#define ALM_INSTRUCTION "Generate audio semantic tokens based on the conditions:"'),
    "prompt-without-the-separator-after-thinking": ("acestep_lm.h", 'snprintf(out + n, cap - n, "<|im_start|>assistant\\n%s\\n\\n", think);', 'snprintf(out + n, cap - n, "<|im_start|>assistant\\n%s\\n", think);'),
    # --- the sampler ---------------------------------------------------------------------------------------------------------------
    "top-k-keeps-one-more": ("acestep_lm.h", "const int target = nid - s->top_k;", "const int target = nid - s->top_k - 1;"),
    "top-k-drops-ties": ("acestep_lm.h", "if (lg[i] < kth) lg[i] = -INFINITY;", "if (lg[i] <= kth) lg[i] = -INFINITY;"),
    "top-p-counts-the-token-itself": ("acestep_lm.h", "if (i > 0 && cum - p > s->top_p)", "if (i > 0 && cum > s->top_p)"),
    "temperature-ignored": ("acestep_lm.h", "exp(((double)lg[i] - mx) / s->temperature)", "exp(((double)lg[i] - mx))", 2),
    "argmax-picks-the-last-tie": ("acestep_lm.h", "for (int i = 1; i < nid; i++) if (lg[i] > lg[best]) best = i;", "for (int i = 1; i < nid; i++) if (lg[i] >= lg[best]) best = i;"),
    # --- codes into the DiT --------------------------------------------------------------------------------------------------------
    "fsq-digit-reversed": ("acestep_tok.h", "(float)(idx % L) - 1.f;", "(float)(L - 1 - idx % L) - 1.f;"),
    "fsq-level-scale": ("acestep_tok.h", "c[(size_t)w * n + i] = (2.f / (float)(L - 1)) * (float)(idx % L) - 1.f;", "c[(size_t)w * n + i] = (2.f / (float)L) * (float)(idx % L) - 1.f;"),
    "hints-padded-with-zeros": ("acestep_tok.h", "for (int f = keep; f < T; f++) memcpy(hints + (size_t)f * A, silence + (size_t)(f - keep) * A, (size_t)A * sizeof(float));", "for (int f = keep; f < T; f++) memset(hints + (size_t)f * A, 0, (size_t)A * sizeof(float));"),
    "codes-frames-per-code": ("acestep_gen.h", "#define ACG_CODE_FRAMES 5 ", "#define ACG_CODE_FRAMES 4 "),
    "codes-song-keeps-the-text-to-music-instruction": ("acestep_gen.h", "const int from_codes = rq->ncodes > 0 && rq->task == ACG_TASK_TEXT2MUSIC;\n    const int A = d->acdim, task = from_codes ? ACG_TASK_COVER : rq->task;",
                                                       "const int from_codes = rq->ncodes > 0 && rq->task == ACG_TASK_TEXT2MUSIC;\n    const int A = d->acdim, task = rq->task;"),
    "codes-hints-not-the-cover-noise-start": ("acestep_gen.h", "memcpy(lat, hints, (size_t)T * A * sizeof(float));", ""),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("lmrefs", nargs="+")
    ap.add_argument("--codes", nargs="+", required=True, help="cases with audio_code_string, for --gen-ref")
    ap.add_argument("--cc", default="gcc")
    ap.add_argument("--only", help="comma-separated mutant names")
    ap.add_argument("--skip", help="comma-separated mutant names to leave out")
    a = ap.parse_args()
    names = list(MUTANTS) if not a.only else a.only.split(",")
    names = [n for n in names if n not in (a.skip.split(",") if a.skip else [])]
    unknown = [n for n in names if n not in MUTANTS]
    if unknown:
        sys.exit(f"unknown mutant(s) {unknown}; known: {list(MUTANTS)}")
    refs = [("--lm-selftest", "1")] + [("--lm-ref", r) for r in a.lmrefs] + [("--gen-ref", r) for r in a.codes]
    with tempfile.TemporaryDirectory(prefix="acestep-mut-") as work:
        verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, None, "acestep_lm.h", "--lm-ref")
        print(f"{'unmodified':52s} {verdict}  {note}")
        if verdict != "pass":
            sys.exit("the unmodified source does not pass the oracles: fix that first")
        survivors = []
        for name in names:
            header, old, new, *count = MUTANTS[name]
            verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, (old, new, *count), header, "--lm-ref")
            caught = verdict == "fail"
            print(f"{name:52s} {'CAUGHT' if caught else 'SURVIVED (' + verdict + ')'}  {note[:100]}", flush=True)
            if not caught:
                survivors.append(name)
    print(f"\n{len(names) - len(survivors)} of {len(names)} mutants caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
