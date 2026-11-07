#!/usr/bin/env python3
"""Mutation check for c/gemma4.c: change the engine in ways a careless edit might, build each
variant, and require tests/test_gemma4.py to fail on it (LogitsTest for the text model, ServeTest for
the serve loop's reuse of a conversation, VisionTest and ServeVisionTest for pictures).

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
# renormalisation changes nothing in exact arithmetic but the norm's epsilon. On the real model's activations that
# is nothing; on the tiny model's small ones it shows, just: the logits move by 2.02e-3 against the tests' 2e-3, so
# the mutant is caught, but by the epsilon and by a hair, not by the router's arithmetic being checked.
# It is listed here so that a future change of tolerance that lets it live is read for what it is.
EQUIVALENT = {"top-k weights not renormalised",
              # the main loop ends a cancelled request on its own turn anyway; the check inside the batch loop only
              # spares it one more token, and no test can tell one token from none without a race
              "a cancelled conversation goes on decoding"}

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
    ("ring too small", "m->ring = c->window + (m->img_tokens > m->chunk ? m->img_tokens : m->chunk) + m->slack;",
     "m->ring = c->window;"),
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
    # pictures: the decoder's side (VisionTest, offline against transformers; ServeVisionTest, over the protocol)
    ("picture rows are causal on the sliding layers", "int hi = y->full ? p : reach[s];", "int hi = p;", "VisionTest"),
    ("picture rows are bidirectional on the full layers too", "int hi = y->full ? p : reach[s];", "int hi = reach[s];", "VisionTest"),
    ("picture rows read as plain embeddings", "        if (g) {\n            if (!g->feat)", "        if (g && 0) {\n            if (!g->feat)", "VisionTest"),
    ("ring too small for a picture", "m->ring = c->window + (m->img_tokens > m->chunk ? m->img_tokens : m->chunk) + m->slack;",
     "m->ring = c->window + m->chunk + m->slack;", "VisionTest"),
    ("a chunk may end inside a picture", "if (a > pos && a < pos + n && b > pos + n) n = a - pos;", "", "VisionTest"),
    ("a chunk that starts in a picture takes only a chunk", "if (pos >= a && pos < b) return b - pos < left ? b - pos : left;",
     "if (pos >= a && pos < b) return n;", "VisionTest"),
    # pictures: the tower (VisionTest compares its soft tokens with transformers')
    ("pixels not mapped to [-1, 1]", "in[i] = 2.f * ((float)pix[i] * rescale - 0.5f);", "in[i] = (float)pix[i] * rescale;", "VisionTest"),
    ("column and row position tables swapped",
     "const float *a = v->pos + (size_t)col * D, *b = v->pos + ((size_t)c->vc.npos + row) * D;",
     "const float *a = v->pos + (size_t)row * D, *b = v->pos + ((size_t)c->vc.npos + col) * D;", "VisionTest"),
    ("rope: rows turn the first half too", "x[i] = a * cx[i] - b * sx[i]; x[i + h4] = b * cx[i] + a * sx[i];",
     "x[i] = a * cy[i] - b * sy[i]; x[i + h4] = b * cy[i] + a * sy[i];", "VisionTest"),
    ("rope: second half not rotated", "x[2 * h4 + i] = e * cy[i] - f * sy[i]; x[3 * h4 + i] = f * cy[i] + e * sy[i];", "", "VisionTest"),
    ("vision attention scaled by 1/sqrt(d)", "float s = dot_f32(qh, kk + ((size_t)j * nh + hh) * hd, hd); sc[j] = s;",
     "float s = 0.35f * dot_f32(qh, kk + ((size_t)j * nh + hh) * hd, hd); sc[j] = s;", "VisionTest"),
    ("vision values not normalised", "            rms_nw(vv + o, vv + o, hd, eps);\n", "", "VisionTest"),
    ("vision queries not normalised", "            rms_w(q + o, q + o, y->q_norm, hd, eps);\n", "", "VisionTest"),
    ("vision MLP gated with silu", "g[i] = gelu_tanh(g[i]) * u[i];", "g[i] = (g[i] / (1.f + expf(-g[i]))) * u[i];", "VisionTest"),
    ("pooling sums instead of averaging", "float inv_kk = 1.f / (float)(k * k), root", "float inv_kk = 1.f, root", "VisionTest"),
    ("pooling by the wrong tokens", "int tok = (p / gw / k) * tw + (p % gw) / k;", "int tok = (p / gw / k) + (p % gw) / k;", "VisionTest"),
    ("no sqrt(hidden) scale after pooling", "            r[i] *= root;\n", "", "VisionTest"),
    ("no standardisation", "if (c->vc.standardize) r[i] = (r[i] - v->std_bias[i]) * v->std_scale[i];", "", "VisionTest"),
    ("projector input not normalised", "        rms_nw(r, r, D, eps);\n    }\n    float *feat", "    }\n    float *feat", "VisionTest"),
    # pictures: the serve loop (ServeVisionTest)
    ("the cache does not tell pictures apart", "for (int t = 0; t < g->ntok; t++) r->keys[g->start + t] = image_row_key(g->key, t);", "", "ServeVisionTest"),
    ("the tower runs only for a picture that starts the chunk",
     "if (!r->sp[k].feat && r->sp[k].start < r->done_np + n && r->sp[k].start + r->sp[k].ntok > r->done_np) { pic = &r->sp[k]; break; }",
     "if (!r->sp[k].feat && r->sp[k].start == r->done_np) { pic = &r->sp[k]; break; }", "ServeVisionTest"),
    ("placeholder runs not counted against pictures", "if (nf != g_npend) snprintf(msg", "if (0) snprintf(msg", "ServeVisionTest"),
    ("picture size not checked against its placeholders", "if (found[i].ntok != want) snprintf(msg", "if (0) snprintf(msg", "ServeVisionTest"),
    ("picture bytes not checked", "else if ((size_t)cmd.payload_bytes != want) {", "else if (0) {", "ServeVisionTest"),
    # batched decode (BatchTest: bit for bit against stepping alone; ServeTest: whole conversations)
    ("a decode batch uses the 4-row kernel", "if (!g_rows1)", "if (1)", "BatchTest"),
    ("batch rows all read the first conversation's cache", "const Seq *cq = qs ? qs[s] : q;", "const Seq *cq = qs ? qs[0] : q;", "BatchTest"),
    ("batch rows all rotate by the first position", "float p = (float)(ps ? ps[s] : pos0 + s);", "float p = (float)(ps ? ps[0] : pos0 + s);", "BatchTest"),
    ("experts summed in the order the cache groups come", "float *ctb = nu > m->cap ? xmalloc((size_t)S * K * H * sizeof(float)) : NULL;",
     "float *ctb = NULL;", "BatchTest"),
    ("the batched head reads one row for all", "for (int s = 0; s < S; s++) rms_w(hn + (size_t)s * H, xr + (size_t)s * H, m->final_norm, H, c->eps);",
     "for (int s = 0; s < S; s++) rms_w(hn + (size_t)s * H, xr, m->final_norm, H, c->eps);", "BatchTest"),
    ("a cancelled conversation goes on decoding", "if (w->cancel) { w->limited = 0; req_finish(m, w); serve_tiers_emap(m); continue; }", "", "ServeTest"),
    ("a batch step feeds the wrong token", "bi[nb] = j; bt[nb] = t; bp[nb] = w->np + w->gen - 1;", "bi[nb] = j; bt[nb] = 0; bp[nb] = w->np + w->gen - 1;", "ServeTest"),
]


def main():
    with open(SRC, encoding="utf-8", newline="") as handle:
        original = handle.read()
    survived = []
    with tempfile.TemporaryDirectory() as tmp:
        only = sys.argv[1:]            # names (or parts of names) to run; none: all of them
        for name, old, new, *which in MUTANTS:
            if only and not any(part in name for part in only):
                continue
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
            try:
                run = subprocess.run([sys.executable, os.path.join(C_DIR, "tests", "test_gemma4.py"), which],
                                     env=dict(os.environ, GEMMA4_EXE=exe, PYTHONUTF8="1"), capture_output=True, text=True,
                                     timeout=900)
                killed = run.returncode != 0
            except subprocess.TimeoutExpired:                  # a mutant that hangs the engine is as dead as one that fails
                killed = True
            print(("killed   " if killed else ("equivalent " if name in EQUIVALENT else "SURVIVED ")) + name)
            if not killed and name not in EQUIVALENT:
                survived.append(name)
    print(f"{len(MUTANTS) - len(survived) - len(EQUIVALENT)}/{len(MUTANTS) - len(EQUIVALENT)} mutants killed (all, unless you named some) "
          f"({len(EQUIVALENT)} equivalent)")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
