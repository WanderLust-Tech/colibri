#!/usr/bin/env python3
"""Proof that the VAE oracle can fail: break acestep_vae.h one line at a time and see that the oracle says so.

    mutate_acestep_vae.py MODEL_DIR REFDIR [--cc gcc] [--only NAME,NAME]

MODEL_DIR is a checkpoint (the tiny fixture of make_acestep_tiny.py is enough) and REFDIR a captured case with
several VAE windows and a peak above 1 (tools/acestep_ref.py capture ... --cases long30). Each mutant is a small,
plausible mistake: a dilation, a padding, a tap order, a trimmed window, a dropped scale. It is compiled into a
temporary directory (acestep.c and acestep_vae.h are copied there; every other header is read from c/) and run
against the oracle, which has to exit non-zero. The unmodified source is run first and has to pass, so a failure
means the mutation, not the setup. Exits non-zero if any mutant survives.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# name: (text in acestep_vae.h, its replacement). The text must occur exactly once.
MUTANTS = {
    "dilation-3-to-1": ("static const int dil[3] = {1, 3, 9};", "static const int dil[3] = {1, 1, 9};"),
    "dilation-9-to-8": ("static const int dil[3] = {1, 3, 9};", "static const int dil[3] = {1, 3, 8};"),
    "convt-padding": ("c->pad = (s + 1) / 2;", "c->pad = s / 2 - 1;"),
    "convt-second-tap": ("(size_t)(kk + s) * cout", "(size_t)kk * cout"),
    "convt-bias-dropped": ("for (int i = 0; i < cout; i++) y[(size_t)j * cout + i] = c->b[i];",
                           "for (int i = 0; i < cout; i++) y[(size_t)j * cout + i] = 0;"),
    "im2col-taps-reversed": ("int s = r0 + i + j * c->dil - pad;", "int s = r0 + i + (c->k - 1 - j) * c->dil - pad;"),
    "conv-bias-dropped": ("qi_gemm(y + (size_t)r0 * c->cout, col, n, &c->m, c->b);",
                          "qi_gemm(y + (size_t)r0 * c->cout, col, n, &c->m, NULL);"),
    "conv-padding": ("const int kc = c->k * c->cin, pad = (c->k - 1) * c->dil / 2;",
                     "const int kc = c->k * c->cin, pad = (c->k - 3) * c->dil / 2;"),
    "weight-norm-gain-ignored": ("scale[i] = g[i] / (float)sqrt(s);", "scale[i] = 1.0f / (float)sqrt(s);"),
    "weight-norm-not-normalised": ("scale[i] = g[i] / (float)sqrt(s);", "scale[i] = g[i];"),
    "snake-alpha-ignored": ("float sn = sinf(s->ea[i] * x[i]);", "float sn = sinf(x[i]);"),
    "snake-not-squared": ("y[i] = x[i] + s->ib[i] * (sn * sn);", "y[i] = x[i] + s->ib[i] * sn;"),
    "snake-beta-ignored": ("y[i] = x[i] + s->ib[i] * (sn * sn);", "y[i] = x[i] + (sn * sn);"),
    "snake-epsilon": ("b[i] = 1.0f / (expf(b[i]) + 1e-9f);", "b[i] = 1.0f / (expf(b[i]) + 1e-3f);"),
    "residual-not-added": ("x[i] += a[i];", "x[i] = a[i];"),
    "tile-trim-end": ("size_t trim0 = (size_t)(cs - ws) * v->hop, trim1 = (size_t)(we - ce) * v->hop;",
                      "size_t trim0 = (size_t)(cs - ws) * v->hop, trim1 = (size_t)(we - ce) * v->hop + 1;"),
    "tile-trim-start-always-overlap": ("size_t trim0 = (size_t)(cs - ws) * v->hop,", "size_t trim0 = (size_t)ov * v->hop,"),
    "tile-core-too-long": ("int cs = i * stride, ce = cs + stride < F ? cs + stride : F;",
                           "int cs = i * stride, ce = cs + stride + 1 < F ? cs + stride + 1 : F;"),
    "tile-overlap-too-small": ("const int stride = chunk - 2 * ov;", "ov = 1; const int stride = chunk - 2 * ov;"),
    "peak-limit-threshold": ("if (peak > 1.0f) for", "if (peak >= 2.0f) for"),
}

# Mutants that are deliberately NOT here, because they are equivalent: no output can tell them from the original.
# Starting each window one frame later (so with one frame less context), and halving the overlap (64 to 32), give
# the same audio to float rounding, since the decoder's receptive field is about 6 to 8 latent frames on each side
# (measured with the tiny fixture: chunk 40 with overlap 6 is wrong by 8e-2, with overlap 8 right to 4e-5, with 16
# to noise) and upstream's overlap is 64. "tile-overlap-too-small" is the same family made observable: with 1 frame
# of context the output is wrong, so the oracle does see the overlap when it matters.


def build_and_run(cc, work, model, ref, source_edit=None):
    shutil.copyfile(os.path.join(C_DIR, "acestep.c"), os.path.join(work, "acestep.c"))
    with open(os.path.join(C_DIR, "acestep_vae.h"), encoding="utf-8", newline="") as f:
        text = f.read()
    if source_edit:
        old, new = source_edit
        if text.count(old) != 1:
            return "bad-mutant", f"{old!r} occurs {text.count(old)} times in acestep_vae.h"
        text = text.replace(old, new)
    with open(os.path.join(work, "acestep_vae.h"), "w", encoding="utf-8", newline="") as f:
        f.write(text)
    exe = os.path.join(work, "acestep_mut.exe" if os.name == "nt" else "acestep_mut")
    cmd = [cc, "-O2", "-march=native", "-fopenmp", "-D_GNU_SOURCE", "-D_FILE_OFFSET_BITS=64", "-I", C_DIR,
           os.path.join(work, "acestep.c"), "-o", exe, "-lm"]
    built = subprocess.run(cmd, capture_output=True, text=True)
    if built.returncode:
        return "build-failed", built.stderr[-400:]
    run = subprocess.run([exe, "--model", model, "--vae-ref", ref], capture_output=True, text=True)
    return ("pass" if run.returncode == 0 else "fail"), run.stderr.strip().splitlines()[-1] if run.stderr.strip() else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model_dir")
    ap.add_argument("refdir")
    ap.add_argument("--cc", default="gcc")
    ap.add_argument("--only", help="comma-separated mutant names")
    a = ap.parse_args()
    names = list(MUTANTS) if not a.only else a.only.split(",")
    unknown = [n for n in names if n not in MUTANTS]
    if unknown:
        sys.exit(f"unknown mutant(s) {unknown}; known: {list(MUTANTS)}")
    with tempfile.TemporaryDirectory(prefix="acestep-mut-") as work:
        verdict, note = build_and_run(a.cc, work, a.model_dir, a.refdir)
        print(f"{'unmodified':26s} {verdict}  {note}")
        if verdict != "pass":
            sys.exit("the unmodified source does not pass the oracle: fix that first")
        survivors = []
        for name in names:
            verdict, note = build_and_run(a.cc, work, a.model_dir, a.refdir, MUTANTS[name])
            caught = verdict == "fail"
            print(f"{name:26s} {'CAUGHT' if caught else 'SURVIVED (' + verdict + ')'}  {note[:110]}")
            if not caught:
                survivors.append(name)
    print(f"\n{len(names) - len(survivors)} of {len(names)} mutants caught")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
