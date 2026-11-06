#!/usr/bin/env python3
"""Mutation check for c/gemma4.c: change the engine in ways a careless edit might, build each
variant, and require tests/test_gemma4.py's LogitsTest to fail on it.

    python c/tools/mutate_gemma4.py            (needs gcc and the tests' requirements)

The source file is never modified: each mutant is built from a copy in a temporary directory.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.join(HERE, "..")
SRC = os.path.join(C_DIR, "gemma4.c")
CFLAGS = ["-D_FILE_OFFSET_BITS=64", "-O2", "-march=x86-64-v3", "-fopenmp", "-w"]
LIBS = ["-lm", "-fopenmp"] + (["-static", "-lpsapi"] if sys.platform == "win32" else [])

# The router renormalises its top-k weights to sum to 1, which scales the whole expert output of a row by
# one constant, and post_feedforward_layernorm_2 (an RMS norm) divides that constant out again. Dropping the
# renormalisation changes nothing but the norm's epsilon, so no test can or should tell the two apart.
EQUIVALENT = {"top-k weights not renormalised"}

MUTANTS = [
    ("attention scaled by 1/4", "float v = dot_f32(qh, q->K[l] + (slot * nkv + kvh) * hd, hd);",
     "float v = 0.25f * dot_f32(qh, q->K[l] + (slot * nkv + kvh) * hd, hd);"),
    ("window one too long", "int lo = y->full ? 0 : (p - c->window + 1 > 0 ? p - c->window + 1 : 0);",
     "int lo = y->full ? 0 : (p - c->window > 0 ? p - c->window : 0);"),
    ("window one too short", "int lo = y->full ? 0 : (p - c->window + 1 > 0 ? p - c->window + 1 : 0);",
     "int lo = y->full ? 0 : (p - c->window + 2 > 0 ? p - c->window + 2 : 0);"),
    ("sliding layers use the full-layer rope base", "1.f / powf(c->theta_s, (float)(2 * i) / (float)c->hd_s)",
     "1.f / powf(c->theta_g, (float)(2 * i) / (float)c->hd_s)"),
    ("full layers rotate the whole head", "int angles = (int)(c->prf * c->hd_g / 2);", "int angles = c->hd_g / 2;"),
    ("values are not normalised", "                rms_nw(vh, vh, hd, c->eps);\n", ""),
    ("keys-equal-values layer has zero values", "else memcpy(m->vb, m->kb, (size_t)S * nkv * hd * sizeof(float));",
     "else memset(m->vb, 0, (size_t)S * nkv * hd * sizeof(float));"),
    ("dense MLP gated with silu", "m->g1[i] = gelu_tanh(m->g1[i]) * m->u1[i];",
     "m->g1[i] = (m->g1[i] / (1.f + expf(-m->g1[i]))) * m->u1[i];"),
    ("experts gated with silu", "#define XF_ACT_GELU_TANH 1\n", ""),
    ("no layer scalar", "xr[i] = (xr[i] + a[i]) * y->layer_scalar;", "xr[i] = (xr[i] + a[i]);"),
    ("embeddings not scaled", "float sc = m->embed.s[id] * emb_scale;", "float sc = m->embed.s[id];"),
    ("no logit soft cap", "out[v] = c->softcap > 0 ? c->softcap * tanhf(l / c->softcap) : l;", "out[v] = l;"),
    ("per-expert scale ignored", "val[s * K + k] = val[s * K + k] / tot * y->router_pes[idx[s * K + k]];",
     "val[s * K + k] = val[s * K + k] / tot;"),
    ("top-k weights not renormalised", "val[s * K + k] = val[s * K + k] / tot * y->router_pes", "val[s * K + k] = val[s * K + k] * y->router_pes"),   # EQUIVALENT, see below
    ("final sandwich norm skipped", "            rms_w(a, a, y->post_ff_ln, H, c->eps);\n", ""),
    ("router scale ignored", "rn[i] = rn[i] * y->router_scale[i] * scale;", "rn[i] = rn[i] * scale;"),
    ("ring too small", "m->ring = c->window + m->chunk;", "m->ring = c->window;"),
    ("queries not normalised", "                rms_w(qh, qh, y->q_norm, hd, c->eps);\n", ""),
    ("keys not normalised", "                rms_w(kh, kh, y->k_norm, hd, c->eps);\n", ""),
    ("attention post-norm skipped", "            rms_w(m->m1 + (size_t)s * H, m->m1 + (size_t)s * H, y->post_attn_ln, H, c->eps);\n", ""),
    ("dense branch not normalised", "        for (int s = 0; s < S; s++) rms_w(m->m1 + (size_t)s * H, m->m1 + (size_t)s * H, y->post_ff_ln1, H, c->eps);\n", ""),
    ("expert input not pre-normalised", "rms_w(m->e_in + (size_t)s * H, xres + (size_t)s * H, y->pre_ff_ln2, H, c->eps);",
     "memcpy(m->e_in + (size_t)s * H, xres + (size_t)s * H, (size_t)H * 4);"),
    ("expert cache serves the wrong expert", "int sl = lc->slot_of[want[i]];", "int sl = lc->slot_of[(want[i] + 1) % c->n_experts];"),
    # the serve loop's reuse of a rewound conversation (ServeTest)
    ("reuse ignores what the ring still holds", "if (p->len > m->ring && p->len - c > m->ring - m->c.window + 1) return 0;", "", "ServeTest"),
    ("reuse allows one position too many", "p->len - c > m->ring - m->c.window + 1) return 0;", "p->len - c > m->ring - m->c.window + 2) return 0;", "ServeTest"),
    ("reuse allows one position too few", "p->len - c > m->ring - m->c.window + 1) return 0;", "p->len - c > m->ring - m->c.window) return 0;", "ServeTest"),
    ("reuse forgets to cut the record", "    p->len = c;\n    return c;", "    return c;", "ServeTest"),
    ("reuse feeds nothing when the prompt repeats", "if (c > np - 1) c = np - 1;", "if (c > np) c = np;", "ServeTest"),
]


def main():
    with open(SRC, encoding="utf-8", newline="") as handle:
        original = handle.read()
    survived = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, old, new, *which in MUTANTS:
            which = which[0] if which else "LogitsTest"
            if original.count(old) != 1:
                print(f"?? {name}: the pattern matches {original.count(old)} times, fix the mutant")
                survived.append(name)
                continue
            mutant = os.path.join(tmp, "m.c")
            exe = os.path.join(tmp, "m.exe" if sys.platform == "win32" else "m")
            with open(mutant, "w", encoding="utf-8", newline="") as handle:
                handle.write(original.replace(old, new))
            build = subprocess.run(["gcc", *CFLAGS, "-I", C_DIR, mutant, "-o", exe, *LIBS], capture_output=True, text=True)
            if build.returncode:
                print(f"?? {name}: does not build\n{build.stderr[-400:]}")
                survived.append(name)
                continue
            run = subprocess.run([sys.executable, os.path.join(C_DIR, "tests", "test_gemma4.py"), which],
                                 env=dict(os.environ, GEMMA4_EXE=exe, PYTHONUTF8="1"), capture_output=True, text=True)
            killed = run.returncode != 0
            print(("killed   " if killed else ("equivalent " if name in EQUIVALENT else "SURVIVED ")) + name)
            if not killed and name not in EQUIVALENT:
                survived.append(name)
    print(f"{len(MUTANTS) - len(survived) - len(EQUIVALENT)}/{len(MUTANTS) - len(EQUIVALENT)} mutants killed "
          f"({len(EQUIVALENT)} equivalent)")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
