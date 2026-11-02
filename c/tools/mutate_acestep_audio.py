#!/usr/bin/env python3
"""Proof that the audio-input oracle can fail: break acestep_audio.h one edit at a time and see that the oracle says so.

    PYTHONUTF8=1 python mutate_acestep_audio.py [--cc gcc] [--only NAME,NAME] [--jobs 4]

Needs the reference venv and ACESTEP_UPSTREAM (see acestep_audio_ref.py). Each mutant is a small, plausible mistake in the
loader or the resampler: a byte order, a scale, a missing chunk pad, a length that rounds the wrong way, the encoder delay
off by one, a check that was removed. It is compiled into a temporary directory and run against the oracle (upstream's code
on the same bytes, then the robustness suite); it is CAUGHT when anything fails. The unmodified source is run first and has
to pass. Exits non-zero if any mutant survives.

A mutant is (old, new) or (old, new, count), or a list of those when one mistake takes several edits: `old` must occur exactly
`count` times (default 1) in acestep_audio.h. CASES names the fixtures a mutant is run on when it needs the slow one (a 6-minute
clip, built only then); every other mutant runs on all the others.
"""
import argparse
import concurrent.futures
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
HEADER = "acestep_audio.h"
SLOW_CASE = "wav44k_mono_8bit_long"          # the fixture of acestep_audio_ref.build_fixtures(slow=True)

MUTANTS = {
    # --- the resampler: torchaudio's kernel -----------------------------------------------------------------------------
    "rolloff-0.98": ("(double)(o < nf ? o : nf) * 0.99;", "(double)(o < nf ? o : nf) * 0.98;"),
    "lowpass-width-5": ("(int)ceil((double)(6 * o) / base)", "(int)ceil((double)(5 * o) / base)"),
    "window-rectangular": ("double w = cos(t * pi / 6.0 / 2.0);", "double w = cos(t * pi / 6.0 / 2.0) * 0.0 + 1.0;"),
    "window-not-squared": ("w = w * w;", ""),
    "window-angle": ("cos(t * pi / 6.0 / 2.0)", "cos(t * pi / 6.0)"),
    "sinc-without-pi": ("t *= pi;", ""),
    "kernel-scale-dropped": ("row[k] = (float)(s * (w * scale));", "row[k] = (float)(s * w);"),
    "kernel-scale-inverted": ("const double scale = base / (double)o;", "const double scale = (double)o / base;"),
    "phase-sign": ("(double)((float)(-p) / (float)nf)", "(double)((float)(p) / (float)nf)"),
    "tap-index-offset": ("const double idx = (double)(-width + k) / (double)o;", "const double idx = (double)(-width + k + 1) / (double)o;"),
    "pad-shifted": ("memcpy(xp + width, x, len * sizeof(float));", "memcpy(xp + width + 1, x, (len - 1) * sizeof(float));"),
    "block-stride": ("const float *xs = xp + bl * (size_t)o;", "const float *xs = xp + bl * (size_t)nf;"),
    "phases-not-interleaved": ("y[bl * (size_t)nf + (size_t)p] =", "y[(size_t)p * nb + bl] ="),
    "unrolled-tap-misaligned": ("a4 += row[k + 4] * xs[k + 4];", "a4 += row[k + 4] * xs[k + 3];"),
    "length-floor": ("size_t target = (size_t)ceilf(want);", "size_t target = (size_t)floorf(want);"),
    "length-plus-one": ("size_t target = (size_t)ceilf(want);", "size_t target = (size_t)ceilf(want) + 1;"),
    "length-not-float32": ("size_t target = (size_t)ceilf(want);",
                           "size_t target = (size_t)ceil((double)((uint64_t)nf * (uint64_t)len) / (double)o);"),
    # --- _normalize_audio_to_stereo_48k ---------------------------------------------------------------------------------
    "resample-skipped": ("if (rate != ACA_TARGET_RATE) {", "if (rate != ACA_TARGET_RATE && 0) {"),
    "clamp-left-skipped": ("a = a < -1.0f ? -1.0f : (a > 1.0f ? 1.0f : a);", ""),
    "clamp-right-skipped": ("b = b < -1.0f ? -1.0f : (b > 1.0f ? 1.0f : b);", ""),
    "clamp-upper-only": ("a = a < -1.0f ? -1.0f : (a > 1.0f ? 1.0f : a);", "a = a > 1.0f ? 1.0f : a;"),
    "mono-second-channel-silent": ("res[n + i] = b;", "res[n + i] = two ? b : 0.0f;"),
    "third-channel-taken": ("c1[i] = pcm[i * (size_t)channels + 1];", "c1[i] = pcm[i * (size_t)channels + (channels > 2 ? 2 : 1)];"),
    "first-channel-twice": ("c1[i] = pcm[i * (size_t)channels + 1];", "c1[i] = pcm[i * (size_t)channels];"),
    "output-not-planar": ("res[n + i] = b;", "res[2 * i + 1] = b;"),
    "second-channel-from-first": ("aca_resample(c1, frames, rate", "aca_resample(c0, frames, rate"),
    # --- WAV ----------------------------------------------------------------------------------------------------------
    "wav8-signed": ("(float)((int)d[i] - 128) * (1.0f / 128.0f)", "(float)(int8_t)d[i] * (1.0f / 128.0f)"),
    "wav8-scale": ("(1.0f / 128.0f)", "(1.0f / 127.0f)"),
    "wav16-scale": ("(1.0f / 32768.0f)", "(1.0f / 32767.0f)"),
    "wav16-unsigned": ("(float)(int16_t)aca_rd16(d + i * 2)", "(float)(int)aca_rd16(d + i * 2)"),
    "wav16-big-endian": ("(float)(int16_t)aca_rd16(d + i * 2)", "(float)(int16_t)(d[i * 2] << 8 | d[i * 2 + 1])"),
    "wav24-no-sign-extension": ("int32_t v = (int32_t)((aca_rd16(d + i * 3) | ((uint32_t)d[i * 3 + 2] << 16)) << 8) >> 8;",
                                "int32_t v = (int32_t)(aca_rd16(d + i * 3) | ((uint32_t)d[i * 3 + 2] << 16));"),
    "wav24-scale": ("(float)v * (1.0f / 8388608.0f)", "(float)v * (1.0f / 8388607.0f)"),
    "wav32-scale": ("(1.0f / 2147483648.0f)", "(1.0f / 2147483000.0f)"),
    "wav-float64-stride": ("memcpy(&v, d + i * 8, 8);", "memcpy(&v, d + i * 4, 8);"),
    "wav-chunk-pad-ignored": ("pos += sz + (sz & 1);", "pos += sz;"),
    "wav-truncated-data-not-cut": ("const size_t dlen = sz > avail ? avail : sz;", "const size_t dlen = sz;"),
    "wav-partial-frame-kept": ("const size_t nfr = dlen / blk;", "const size_t nfr = (dlen + blk - 1) / blk;"),
    "wavex-subformat-ignored": ("tag = aca_rd16(p + 24);", ""),
    "wav-zero-length-accepted": ("if (nfr == 0) {", "if (0) {"),
    "wav-cap-removed": ("if (nfr > aca_cap_frames(max_seconds, (int)rt)) {", "if (0) {"),
    "wav-block-align-unchecked": ("if (blk != ch * bytes) {", "if (0) {"),
    "wav-sample-size-unchecked": ("if (!okbits) {", "if (0) {"),
    "wav-data-before-fmt-allowed": ("if (!have_fmt) {", "if (0) {"),
    "wav-fmt-size-unchecked": ("if (sz < 16 || avail < 16) {", "if (0) {"),
    "wav-rate-limit-removed": ("rt < 1 || rt > ACA_MAX_RATE", "rt < 1"),
    "wav-channel-limit-removed": ("ch < 1 || ch > ACA_MAX_CHANNELS", "ch < 1"),
    "resample-kernel-cap-removed": ("if (K > ACA_MAX_KERNEL) {", "if (0) {"),
    # --- MP3 --------------------------------------------------------------------------------------------------------
    "mp3-id3v2-not-skipped": ("while (n - pos >= 10 && !memcmp(b + pos, \"ID3\", 3)) {", "while (0 && n - pos >= 10 && !memcmp(b + pos, \"ID3\", 3)) {"),
    "mp3-id3-overrun-unchecked": ("if (total > n - pos) {", "if (0) {"),
    "mp3-id3v1-not-stripped": ("if (n - pos >= 128 && !memcmp(b + n - 128, \"TAG\", 3)) {", "if (0) {"),
    "mp3-gapless-begin-off-by-one": ("begin = delay + 529;", "begin = delay + 528;"),
    "mp3-gapless-end-off-by-one": ("const size_t total = tag_frames * spf + 529;", "const size_t total = tag_frames * spf + 528;"),
    "mp3-gapless-off": ("if (have_tag) {", "if (0) {"),
    "mp3-info-frame-played": ("if (is_info) {", "if (0) {"),
    "mp3-tag-found-behind-crc": [("if (crc) return 0;", ""), ("size_t o = 4 + side;", "size_t o = 4 + (crc ? 2 : 0) + side;")],
    "mp3-samples-per-frame-fixed": ("spf = (size_t)s;", "spf = 1152;"),
    "mp3-side-info-mono-swapped": ("const size_t side = mpeg1 ? (mono ? 17 : 32) : (mono ? 9 : 17);",
                                   "const size_t side = mpeg1 ? (mono ? 32 : 17) : (mono ? 17 : 9);"),
    "mp3-delay-shift": ("*delay = ((size_t)t[0] << 4) | (t[1] >> 4);", "*delay = ((size_t)t[0] << 3) | (t[1] >> 4);"),
    "mp3-padding-shift": ("*padding = (((size_t)t[1] & 15) << 8) | t[2];", "*padding = (((size_t)t[1] & 15) << 4) | t[2];"),
    "mp3-lame-offset": ("const unsigned char *t = f + o + 21;", "const unsigned char *t = f + o + 20;"),
    "mp3-xing-toc-size": ("if (flags & 4) o += 100;", "if (flags & 4) o += 99;"),
    "mp3-frames-field-offset": ("frames_field = aca_be32(f + o); o += 4;", "frames_field = aca_be32(f + o + 4); o += 4;"),
    "mp3-claimed-length-unchecked": ("if (have_tag && tag_frames * spf > limit + 4 * spf) {", "if (0) {"),
    "mp3-cap-removed": ("if (got + (size_t)s > limit ||", "if (0 ||"),
    "mp3-format-change-allowed": ("info.hz != hz || info.channels != ch", "0"),
    # --- telling the formats apart --------------------------------------------------------------------------------------
    "flac-not-named": ("if (n >= 4 && !memcmp(b, \"fLaC\", 4)) {", "if (0) {"),
    "garbage-tried-as-mp3": ("if ((n >= 3 && !memcmp(b, \"ID3\", 3)) || (n >= 2 && b[0] == 0xFF && (b[1] & 0xE0) == 0xE0)) return aca_mp3(",
                             "if (1) return aca_mp3("),
    "empty-file-not-refused": ("if (!b || n == 0) {", "if (0) {"),
    "riff-form-type-ignored": ("!memcmp(b + 8, \"WAVE\", 4)", "1"),
}

# Tried, and not mistakes any output comparison can catch: the edit changes the result by less than float32 noise (measured: the
# worst error over every WAV case is 2.38e-7 with it, as without it). Run and reported, not counted.
EQUIVALENT = {
    "width-floor": (("(int)ceil((double)(6 * o) / base)", "(int)floor((double)(6 * o) / base)"),
                    "drops the tap on each side that sits at |t| = 5.94..6, where the Hann window is ~0: the worst error is 2.38e-7 with it, as without"),
}

# what a mutant needs beyond the default fixtures
CASES = {"length-not-float32": {SLOW_CASE}}


def apply_edits(text, entry):
    edits = entry if isinstance(entry, list) else [entry]
    for e in edits:
        old, new, count = (*e, 1)[:3] if len(e) == 2 else e
        if text.count(old) != count:
            return None, f"{old!r} occurs {text.count(old)} times in {HEADER}, expected {count}"
        text = text.replace(old, new)
    return text, ""


def judge(cc, cases, name, entry, source):
    """("caught" | "survived" | "build-failed" | "bad-mutant" | "pass", note) for one mutant (entry None: the unmodified source)."""
    import acestep_audio_ref as ref     # numpy and soundfile: only a run needs them, not the list of mutants
    with tempfile.TemporaryDirectory(prefix="acestep-audio-mut-") as work:
        text = source
        if entry is not None:
            text, why = apply_edits(source, entry)
            if text is None:
                return "bad-mutant", why
        exe, err = ref.build_tool(cc, work, header_text=text)
        if not exe:
            return "build-failed", err.strip().splitlines()[-1] if err.strip() else "?"
        only = CASES.get(name)
        run = [c for c in cases if (c.name in only if only else c.name != SLOW_CASE)]
        problems = ref.run_oracle(exe, run, work, verbose=False, stop_early=True)
        if not problems and not only:
            problems = ref.run_robustness(exe, work, verbose=False, stop_early=True, fuzz_runs=40)
        if entry is None:
            return ("pass", "") if not problems else ("fail", problems[0][:140])
        return ("caught", problems[0][:110]) if problems else ("survived", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cc", default="gcc")
    ap.add_argument("--only", help="comma-separated mutant names")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    import acestep_audio_ref as ref
    names = list(MUTANTS) if not a.only else a.only.split(",")
    unknown = [n for n in names if n not in MUTANTS]
    if unknown:
        sys.exit(f"unknown mutant(s) {unknown}; known: {list(MUTANTS)}")
    with open(os.path.join(C_DIR, HEADER), encoding="utf-8", newline="") as f:
        source = f.read()
    with tempfile.TemporaryDirectory(prefix="acestep-audio-cases-") as d:
        cases = ref.prepare_cases(d, slow=any(n in CASES for n in names) or not a.only)
        verdict, note = judge(a.cc, cases, "unmodified", None, source)
        print(f"{'unmodified':32s} {verdict}  {note}", flush=True)
        if verdict != "pass":
            sys.exit("the unmodified source does not pass the oracle: fix that first")
        survivors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futs = {n: pool.submit(judge, a.cc, cases, n, MUTANTS[n], source) for n in names}
            for n in names:
                verdict, note = futs[n].result()
                print(f"{n:32s} {'CAUGHT' if verdict == 'caught' else 'NOT CAUGHT (' + verdict + ')'}  {note}", flush=True)
                if verdict != "caught":
                    survivors.append(n)
        if not a.only:
            for n, (entry, why) in EQUIVALENT.items():
                verdict, note = judge(a.cc, cases, n, entry, source)
                print(f"{n:32s} EQUIVALENT, not counted ({'survived, as documented' if verdict == 'survived' else verdict}): {why}", flush=True)
    print(f"\n{len(names) - len(survivors)} of {len(names)} mutants caught" + (f"; not caught: {', '.join(survivors)}" if survivors else ""))
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
