#!/usr/bin/env python3
"""Proof that the planner's first-phase oracles can fail: break acestep_meta.h one line at a time and see that the oracles say so.

    mutate_acestep_meta.py MODEL_DIR --yaml CORPUS --parse PARSE --trees TREES --cases REF [REF ...] [--random-model DIR --random-case REF]
                           [--cc gcc] [--only NAME,NAME] [--skip NAME,NAME]

MODEL_DIR is a checkpoint whose planner has learned the metadata blocks (`make_acestep_tiny.py --audio-in --base --lm --lm-text`); CORPUS is
`acestep_ref_meta.py yaml`'s, TREES its trees.json and each REF one of its cases, all captured from that checkpoint. The random model (`--lm` without
`--lm-text`) and its case are the degenerate flow: a planner that never ends its caption, which walks the decoder's free states to the 3500-token limit.

A mutant is a small, plausible mistake: in the YAML emitter (the wrap, the quoting, the resolvers), in the token tables (a range, a tokenization rule), in
the decoder's machine (what ends a caption, which field follows, what is injected), in the parse of the model's text, or in the loop. It is compiled
into a temporary directory and run against `--lm-selftest`, `--yaml-ref`, `--parse-ref`, `--meta-trees` and every `--meta-ref`: CAUGHT when any run fails. The
unmodified source is run first and has to pass. Exits non-zero if a mutant survives.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate_acestep_vae as base  # noqa: E402

H = "acestep_meta.h"
# name: (header, text, its replacement[, how many times it occurs])
MUTANTS = {
    # --- the YAML emitter ------------------------------------------------------------------------------------------------------------
    "yaml-width-79": (H, "#define AME_WIDTH 80", "#define AME_WIDTH 79"),
    "yaml-plain-wraps-at-the-width": (H, "start + 1 == end && y->col > AME_WIDTH && split) { ame_yindent(y); y->ws = 0; y->ind = 0; }", "start + 1 == end && y->col >= AME_WIDTH && split) { ame_yindent(y); y->ws = 0; y->ind = 0; }"),
    "yaml-single-wraps-at-the-width": (H, "y->col > AME_WIDTH && split && start != 0 && end != L)", "y->col >= AME_WIDTH && split && start != 0 && end != L)"),
    "yaml-single-quote-not-doubled": (H, "ame_yputs(y, \"''\"); start = end + 1;", "ame_yputs(y, \"'\"); start = end + 1;"),
    "yaml-single-quote-wrap-never": (H, "if (start + 1 == end && y->col > AME_WIDTH && split && start != 0 && end != L) ame_yindent(y);", "if (0) ame_yindent(y);"),
    "yaml-double-wrap-never": (H, "&& y->col + (end - start) > AME_WIDTH && split) {", "&& y->col + (end - start) > 100000 && split) {"),
    "yaml-double-hex-lowercase": (H, "snprintf(buf, sizeof buf, \"\\\\x%02X\", ch);", "snprintf(buf, sizeof buf, \"\\\\x%02x\", ch);"),
    "yaml-double-astral-kept": (H, "|| (ch >= 0xE000 && ch <= 0xFFFD);", "|| (ch >= 0xE000);"),
    "yaml-resolver-no": (H, "\"yes\", \"Yes\", \"YES\", \"no\", \"No\", \"NO\", ", "\"yes\", \"Yes\", \"YES\", "),
    "yaml-resolver-null-tilde": (H, "static const char *const nulls[] = { \"~\", \"null\",", "static const char *const nulls[] = { \"null\","),
    "yaml-resolver-merge": (H, "\"<<\", \"=\", NULL };", "\"=\", NULL };"),
    "yaml-resolver-empty-is-null": (H, "\"NULL\", \"\", NULL };", "\"NULL\", NULL };"),
    "yaml-int-sexagesimal": (H, "        int s = ame_sexa(t, n, e);\n        if (s == n) return 1;\n", ""),
    "yaml-int-hex": (H, "if (t[i] == '0' && i + 1 < n && t[i + 1] == 'x') {", "if (0) {"),
    "yaml-int-octal": (H, "if (t[i] == '0' && i + 1 < n) { int e = ame_run(t, n, i + 1, ame_oct_us); if (e == n) return 1; }", ""),
    "yaml-float-exponent-sign": (H, "if (i + 1 < n && (t[i + 1] == '-' || t[i + 1] == '+') && i + 2 < n) {", "if (i + 1 < n && t[i + 1] == '-' && i + 2 < n) {"),
    "yaml-float-inf": (H, "static const char *const inf[] = { \"inf\", \"Inf\", \"INF\", NULL };", "static const char *const inf[] = { \"inf\", NULL };"),
    "yaml-timestamp-date-only": (H, "if (n == 10 && ame_dd(t, n, 0, 4)", "if (n == 11 && ame_dd(t, n, 0, 4)"),
    "yaml-timestamp-zone": (H, "if (j + 1 == n && t[j] == 'Z') return 1;", ""),
    "yaml-colon-before-space": (H, "            if (ch == ':' && followed_ws) block_ind = 1;\n            if (ch == '#' && prev_ws) block_ind = 1;", "            if (ch == '#' && prev_ws) block_ind = 1;"),
    "yaml-hash-without-space": (H, "if (ch == '#' && prev_ws) block_ind = 1;", "if (ch == '#') block_ind = 1;"),
    "yaml-leading-indicator-percent": (H, "strchr(\"#,[]{}&*!|>'\\\"%@`\", ch)", "strchr(\"#,[]{}&*!|>'\\\"@`\", ch)"),
    "yaml-leading-dash-alone": (H, "if (ch == '-' && followed_ws) block_ind = 1;", "if (ch == '-') block_ind = 1;"),
    "yaml-document-marker": (H, "if (L >= 3 && ((t[0] == '-' && t[1] == '-' && t[2] == '-') || (t[0] == '.' && t[1] == '.' && t[2] == '.'))) block_ind = 1;", ""),
    "yaml-trailing-space-plain": (H, "if (leading_space || leading_break || trailing_space || trailing_break) a.allow_block_plain = 0;", "if (leading_space || leading_break || trailing_break) a.allow_block_plain = 0;"),
    "yaml-special-characters": (H, "if (space_break || special) { a.allow_block_plain = 0; a.allow_single = 0; }", "if (space_break) { a.allow_block_plain = 0; a.allow_single = 0; }"),
    "yaml-unicode-allowed": (H, "(ch >= 0xE000 && ch <= 0xFFFD) || (ch >= 0x10000 && ch < 0x10FFFF)) && ch != 0xFEFF)) special = 1;", "(ch >= 0xE000 && ch <= 0xFFFD) || (ch >= 0x10000 && ch < 0x10FFFF)) && 1)) special = 1;"),
    "yaml-cot-timesignature-int": (H, "if (digits && ame_pyint(v, nums[n], sizeof nums[n])) {", "if (0 && digits && ame_pyint(v, nums[n], sizeof nums[n])) {"),
    "yaml-cot-slash-four": (H, "if (l >= 2 && !strcmp(v + l - 2, \"/4\"))", "if (0)"),
    # --- parsing what the model wrote --------------------------------------------------------------------------------------------------
    "parse-caption-joined-with-newlines": (H, "            if (!first) out[o++] = ' ';\n", "            if (!first) out[o++] = '\\n';\n"),
    "parse-caption-lines-not-stripped": (H, "        while (a < b && ame_isspace(cp[a])) a++;\n        while (b > a && ame_isspace(cp[b - 1])) b--;\n        if (b > a) {\n            if (!first)", "        if (b > a) {\n            if (!first)"),
    "parse-text-after-audio-code": (H, "        const size_t keep = code ? (size_t)(code - text) : strlen(text);", "        const size_t keep = strlen(text);"),
    "parse-tag-lines-kept": (H, "        if (starts_tag) { /* skipped */ }\n        else if", "        if (0) { }\n        else if"),
    "parse-field-needs-no-indent": (H, "else if (ll > 0 && !ame_isspace(line[0]) && colon >= 0) {", "else if (ll > 0 && colon >= 0) {"),
    "parse-first-value-line-kept-empty": (H, "            if (nonblank) {\n", "            if (1) {\n"),
    # --- the token tables --------------------------------------------------------------------------------------------------------------
    "tree-bpm-range": (H, "nv = 271; vals", "nv = 270; vals"),
    "tree-duration-range": (H, "nv = ame_max_duration - 9;", "nv = ame_max_duration - 10;"),
    "tree-no-newline-after-a-complete-value": (H, "const int tok = t->len[i] == pl ? nl : t->seq[i][pl];", "const int tok = t->len[i] == pl ? -1 : t->seq[i][pl];"),
    "tree-keyscale-flat-missing": (H, "*const acc[] = { \"\", \"#\", \"b\", ", "*const acc[] = { \"\", \"#\", \"#\", "),
    "tree-keyscale-note-check": (H, "if (*q && !(c >= 'A' && c <= 'G')) ok = 0;", "if (!(c >= 'A' && c <= 'G')) ok = 0;"),
    "tree-language-missing": (H, "\"vi\", \"yue\", \"zh\", \"unknown\"", "\"vi\", \"zh\", \"unknown\""),
    "tree-timesig-six": (H, "static const char *const ts[] = { \"2\", \"3\", \"4\", \"6\" };", "static const char *const ts[] = { \"2\", \"3\", \"4\" };"),
    "tab-fixed-shortest-prefix": (H, "for (int end = L - pos; end >= 1 && best < 0; end--) {", "for (int end = 1; end <= L - pos && best < 0; end++) {"),
    "tab-caption-keeps-the-backtick": (H, "&& i != t->backtick) t->cand_cap", ") t->cand_cap"),
    "tab-caption-keeps-the-audio-codes": (H, "if ((i < m->audio0 || i >= m->audio0 + ALM_CODES) && i != t->backtick)", "if (i != t->backtick)"),
    "tab-context-not-stripped": (H, "int ok = n >= cn && !memcmp(ids, cids, (size_t)cn * sizeof(int));\n        const int vl = n - cn;", "int ok = n >= cn && !memcmp(ids, cids, (size_t)cn * sizeof(int));\n        const int vl = n - cn + 1;"),
    # --- the decoder's machine ----------------------------------------------------------------------------------------------------------
    "fsm-language-lowest-not-top": (H, "int best = 0; for (int i = 1; i < nf; i++) if (lg[i] > lg[best]) best = i;", "int best = 0; for (int i = 1; i < nf; i++) if (lg[i] < lg[best]) best = i;"),
    "fsm-language-first-token-forced": (H, "if (f->state == AMS_LANG_VAL && f->nacc == 0) { a->mode = 3; return 0; }", ""),
    "fsm-caption-line-indent-inverted": (H, "if (bl > 0 && b[0] != ' ' && b[0] != '\\t') { f.ending = 1; f.pendn = 0; mode = 2; }", "if (bl > 0 && (b[0] == ' ' || b[0] == '\\t')) { f.ending = 1; f.pendn = 0; mode = 2; }"),
    "fsm-caption-line-tab-is-not-indent": (H, "if (bl > 0 && b[0] != ' ' && b[0] != '\\t') {", "if (bl > 0 && b[0] != ' ') {"),
    "fsm-caption-argmax-over-text-only": (H, "alm_logits(m, h, 1, t->cand_all, t->vocab, lg); have_all = 1;\n                int best = 0; for (int i = 1; i < t->vocab; i++) if (lg[i] > lg[best]) best = i;",
                                          "alm_logits(m, h, 1, t->cand_cap, t->ncap, lg);\n                int best = 0; for (int i = 1; i < t->ncap; i++) if (lg[i] > lg[best]) best = i; best = t->cand_cap[best];"),
    "fsm-caption-limit-512": (H, "#define ALM_META_CAPTION_MAX 512", "#define ALM_META_CAPTION_MAX 511"),
    "fsm-caption-newline-detection": (H, "f->after_nl = strchr(ts, '\\n') != NULL;", "f->after_nl = ts[0] == '\\n';"),
    "fsm-field-name-colon": (H, "                if (strchr(ts, ':')) {\n                    char nm[256];", "                if (ts[0] == ':') {\n                    char nm[256];"),
    "fsm-field-name-case": (H, "nm[k++] = f->pend[i] >= 'A' && f->pend[i] <= 'Z' ? (char)(f->pend[i] + 32) : f->pend[i];", "nm[k++] = f->pend[i];"),
    "fsm-unknown-field-name": (H, "else { f->ending = 0; f->pendn = 0; ame_transition(f); }", "else { f->ending = 0; f->pendn = 0; }"),
    "fsm-skipped-caption": (H, "if (i == AMF_CAPTION && f->skip_caption) continue;", ""),
    "fsm-skipped-language": (H, "if (i == AMF_LANGUAGE && f->skip_language) continue;", ""),
    "fsm-injected-value-has-no-space": (H, "snprintf(text, sizeof text, \" %s\\n\", value);", "snprintf(text, sizeof text, \"%s\\n\", value);"),
    "fsm-injection-ends-in-name-state": (H, "ame_enter(f, ame_next_field(f, field));\n        }\n        return 0;", "ame_enter(f, AMS_END);\n        }\n        return 0;"),
    "fsm-user-bpm-ignored": (H, "if ((bpm ? f->user_bpm > 0 : f->user_dur > 0) && f->nacc == 0) {", "if ((bpm ? 0 : f->user_dur > 0) && f->nacc == 0) {"),
    "fsm-user-key-ignored": (H, "if (user && *user && f->nacc == 0) {", "if (0 && user && *user && f->nacc == 0) {"),
    "fsm-end-token-not-forced": (H, "case AMS_END: a->ids[a->n++] = t->eos; return 0;", "case AMS_END: a->ids[a->n++] = t->nl; return 0;"),
    "fsm-fixed-string-position": (H, "a->ids[a->n++] = t->fixed[f->state][f->pos];", "a->ids[a->n++] = t->fixed[f->state][f->pos ? f->pos - 1 : 0];"),
    "fsm-timesig-ends-in-name": (H, "case AMS_TS_VAL: return AMS_END;", "case AMS_TS_VAL: return AMS_TS_NAME;"),
    "fsm-keyscale-before-duration": (H, "case AMS_DUR_VAL: return ame_next_field(f, AMF_DURATION);", "case AMS_DUR_VAL: return AMS_LANG_NAME;"),
    # --- the loop ----------------------------------------------------------------------------------------------------------------------
    "meta-prompt-text": (H, "# Caption\\n%s\\n\\n# Lyric\\n%s\\n<|im_end|>\\n<|im_start|>assistant\\n\", ALM_INSTRUCTION", "# Captions\\n%s\\n\\n# Lyric\\n%s\\n<|im_end|>\\n<|im_start|>assistant\\n\", ALM_INSTRUCTION"),
    "meta-stops-at-the-end-token": (H, "if (tok == t->eos || (t->pad >= 0 && tok == t->pad)) break;", "if (0) break;"),
    "meta-position-skips": (H, "alm_decode_rows(m, caches, &pos, x, 1);\n        pos++;", "alm_decode_rows(m, caches, &pos, x, 1);\n        pos += 2;"),
    "meta-limit-3499": (H, "#define ALM_META_MAX_NEW 3500 ", "#define ALM_META_MAX_NEW 3499 "),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--parse", required=True, help="acestep_ref_meta.py parse's corpus")
    ap.add_argument("--trees", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--random-model")
    ap.add_argument("--random-case")
    ap.add_argument("--cc", default="gcc")
    ap.add_argument("--only", help="comma-separated mutant names")
    ap.add_argument("--skip", help="comma-separated mutant names to leave out")
    a = ap.parse_args()
    names = list(MUTANTS) if not a.only else a.only.split(",")
    names = [n for n in names if n not in (a.skip.split(",") if a.skip else [])]
    unknown = [n for n in names if n not in MUTANTS]
    if unknown:
        sys.exit(f"unknown mutant(s) {unknown}; known: {list(MUTANTS)}")
    refs = [("--lm-selftest", "1"), ("--yaml-ref", a.yaml), ("--parse-ref", a.parse), ("--meta-trees", a.trees)] + [("--meta-ref", r) for r in a.cases]
    if a.random_model and a.random_case:
        refs.append(("--meta-ref", a.random_case, a.random_model))
    with tempfile.TemporaryDirectory(prefix="acestep-mut-") as work:
        verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, None, H, "--meta-ref")
        print(f"{'unmodified':52s} {verdict}  {note}")
        if verdict != "pass":
            sys.exit("the unmodified source does not pass the oracles: fix that first")
        survivors = []
        for name in names:
            header, old, new, *count = MUTANTS[name]
            verdict, note = base.build_and_run(a.cc, work, a.model_dir, refs, (old, new, *count), header, "--meta-ref")
            caught = verdict == "fail"
            print(f"{name:52s} {'CAUGHT' if caught else 'SURVIVED (' + verdict + ')'}  {note[:100]}", flush=True)
            if not caught:
                survivors.append(name)
    print(f"\n{len(names) - len(survivors)} of {len(names)} mutants caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
