/* acestep.c -- ACE-Step 1.5 text-to-music in pure C.
 *
 * The checkpoint is the Hugging Face repository ACE-Step/Ace-Step1.5: three networks used one after the other,
 *
 *   Qwen3-Embedding-0.6B   the text encoder (caption and lyric tokens)
 *   acestep-v15-turbo      the DiT with its condition encoders: 8 flow-matching steps over 25 Hz latents
 *   vae                    the Oobleck decoder, latents to 48 kHz stereo (acestep_vae.h)
 *
 * This file is being built stage by stage, each one held to the reference by an oracle that compares every
 * tensor with what upstream's own handler produced (tools/acestep_ref.py):
 *
 *   1. the VAE decoder        acestep --model DIR --vae-ref REFDIR           (acestep_vae.h)
 *   2. the DiT and its sampler  acestep --model DIR --dit-ref REFDIR         (acestep_dit.h)
 *   3. the text encoder and its tokenizer  acestep --model DIR --tok-ref FILE / --text-ref REFDIR   (acestep_text.h)
 *   4. the engine proper                   acestep --model DIR --gen-ref REFDIR / --serve / --caption ... --out FILE.wav
 *                                          (acestep_gen.h joins the three networks, acestep_serve.h speaks the line protocol)
 *
 * Usage:
 *   acestep --model DIR --caption TEXT --out FILE.wav [--lyrics TEXT] [--language CODE] [--duration S] [--steps N] [--shift X]
 *           [--seed N] [--bpm N] [--key K] [--timesig T] [--no-dcw]
 *       Makes a clip and writes it as a 16-bit stereo WAV at 48 kHz. Defaults: instrumental, 30 s, 8 steps, shift 1, DCW on,
 *       a seed taken from the clock (printed). The same seed gives the same clip as upstream's handler (acg_randn is torch's generator).
 *       With audio in (a WAV or an MP3; docs/acestep.md): --ref CLIP takes the timbre and style from a clip, in any task;
 *       --cover CLIP makes a cover of it (--cover-strength 0..1 and --cover-noise 0..1 steer how near it stays; --caption is
 *       then optional); --repaint CLIP --from S --to S makes that stretch of it again and keeps the rest (--to 0 or absent: its
 *       end; a negative --from or a --to past the end adds silence there; --repaint-mode balanced|conservative|aggressive,
 *       --repaint-strength 0..1 for balanced). The result is the clip's length.
 *   acestep --model DIR --serve
 *       The line protocol of acestep_serve.h on stdin and stdout, for the gateway and `coli music`.
 *   acestep --model DIR --gen-ref REFDIR [--wav OUT.wav]
 *       The whole pipeline from a captured case's own parameters (caption, lyrics, metas, duration, seed, steps, shift, DCW)
 *       against what the handler returned: the noise, the silence latent slices, and the audio.
 *   acestep --model DIR --dit-ref REFDIR [--dump PREFIX] [--threads N]
 *       DIR is the checkpoint (it holds acestep-v15-turbo/); REFDIR one captured case. Compares, in order: the condition
 *       encoders' packed tokens with the handler's, the context, the timestep embedding and modulation of every call,
 *       the DiT's RoPE table, every DiT call fed the latents and timesteps the handler fed it, then (with a case captured
 *       with --taps) each layer ALONE fed the handler's own hidden state, and finally the whole sampler from the captured
 *       noise: the latents each step receives, each velocity, the schedule, and the final latents. The oracle checks the
 *       DiT, not the text encoder: its input is the captured text and lyric hidden states.
 *       --dump PREFIX writes the C velocities of each call (PREFIX.dit{i}.f32) and, with taps, its hidden states.
 *       ACESTEP_DEBUG_LAYER=N also taps the attention stages (q, k, v, output) of layer N, for comparing with float64.
 *   acestep --model DIR --vae-ref REFDIR [--chunk N] [--overlap N] [--wav OUT.wav] [--dump PREFIX] [--threads N]
 *       DIR is the checkpoint (it holds vae/); REFDIR one captured case (tools/acestep_ref.py capture: REFDIR/ref.safetensors).
 *       Decodes every captured vae{k}.in and compares with vae{k}.out, then decodes the case's latents the way
 *       upstream's handler does (overlapping windows of --chunk frames, default 256, with --overlap, default 64, on
 *       each side), applies its peak limit, and compares with the audio it returned. --wav writes that audio;
 *       --dump PREFIX writes each single decode's planar f32 to PREFIX.decode{k}.f32, for comparing with another
 *       reference than the captured one (a float64 decode shows the model's own rounding sensitivity). */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "st.h"
#include "json.h"
#include "qi_gemm.h"
#include "acestep_vae.h"
#include "acestep_vae_enc.h"
#include "acestep_dit.h"
#include "acestep_tok.h"
#include "acestep_text.h"
#include "acestep_lm.h"           /* the 5 Hz planner: caption and lyrics to audio codes */
#include "acestep_meta.h"         /* its first phase: the metadata, written under upstream's constrained decoder */
#include "acestep_gen.h"          /* the three networks joined: a request in, audio out */
#include "acestep_serve.h"        /* and its line protocol */

static void usage(void){
    fprintf(stderr,
        "usage: acestep --model DIR --dit-ref REFDIR [--dump PREFIX] [--threads N]\n"
        "       acestep --model DIR --vae-ref REFDIR [--chunk N] [--overlap N] [--wav OUT.wav] [--dump PREFIX] [--threads N]\n"
        "       acestep --model DIR --vae-enc-ref REFDIR | --cover-ref REFDIR\n"
        "       acestep --model DIR --tok-ref FILE.json | --text-ref REFDIR\n"
        "       acestep --model DIR --gen-ref REFDIR [--wav OUT.wav]\n"
        "       acestep --model DIR --lm-ref REFDIR [--lm 1.7B]      the planner (tools/acestep_ref_lm.py)\n"
        "       acestep --model DIR --meta-ref REFDIR [--lm 1.7B]    the planner's first phase (tools/acestep_ref_meta.py)\n"
        "       acestep --model DIR --meta-trees FILE [--lm 1.7B]    its token tables against upstream's (OUTDIR/trees.json)\n"
        "       acestep --yaml-ref FILE                              the YAML emitter against PyYAML (tools/acestep_ref_meta.py yaml)\n"
        "       acestep --parse-ref FILE                             the parse of the model's text and its YAML against upstream's (tools/acestep_ref_meta.py parse)\n"
        "       acestep --model DIR --caption TEXT --plan --bpm N --key K --timesig T --duration S --out FILE.wav [--lm 0.6B|1.7B|4B] [--lm-temperature X]\n"
        "               [--lm-cfg X] [--lm-top-p X] [--lm-top-k N]    the 5 Hz planner writes the song's structure first (needs bpm, key, timesig and duration)\n"
        "       acestep --model DIR --caption TEXT --out FILE.wav [--lyrics TEXT] [--language CODE] [--duration S] [--steps N]\n"
        "               [--shift X] [--seed N] [--bpm N] [--key K] [--timesig T] [--no-dcw]\n"
        "               [--ref CLIP] [--cover CLIP [--cover-strength X] [--cover-noise X]]\n"
        "               [--repaint CLIP [--from S] [--to S] [--repaint-mode M] [--repaint-strength X]]\n"
        "       acestep --model DIR --serve\n"
        "  --dit NAME  the DiT to run: acestep-v15-turbo (the default), acestep-v15-base or acestep-v15-sft. The base and SFT ones are guided\n"
        "              (--guidance X [--cfg-start T --cfg-end T] [--adg]), take up to 200 steps (default 32) and do three more tasks:\n"
        "               [--extract CLIP --track NAME] [--lego CLIP --track NAME [--from S] [--to S]] [--complete [CLIP] --tracks A,B]\n"
        "  the sampler, any DiT (upstream's options, off by default): [--sampler heun] [--sde] [--vnorm X] [--vema X]\n"
        "              Heun's second-order step (two DiT calls a step), the stochastic step (not with Heun), a cap on a step's velocity as X times the\n"
        "              latents' norm (above 0) and a smoothing of it with the last step's (0 to 1)\n"
        "  flow-edit (upstream's overlay on text to music): [--edit CLIP --edit-caption TEXT [--edit-lyrics TEXT] [--n-min X] [--n-max X] [--n-avg N]]\n"
        "              edits CLIP toward --caption and --lyrics, taking --edit-caption and --edit-lyrics as what the clip is; the edit runs over the window\n"
        "              n-min to n-max (0 to 1, default 0 to 1) of the schedule, n-avg draws averaged a step. Euler only; the clip's length is the song's\n"
        "  DIR     the ACE-Step checkpoint (holds acestep-v15-turbo/ and vae/, and maybe acestep-v15-base/ and -sft/)\n"
        "  REFDIR  one case captured by tools/acestep_ref.py (REFDIR/ref.safetensors)\n");
}

static double now(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* The two limits a decode has to meet. The ratio of the norms of the error and the reference says the waveform is
 * right as a whole; the largest single error says no sample is wrong on its own, which the ratio cannot: one click of
 * 0.01 in thirty seconds of audio moves it by 1e-6. A waveform lives in [-1, 1], so 5e-5 is -86 dBFS, far below
 * hearing and 4x above the float32 noise measured on the real checkpoint (1.2e-5). The relative limit is 4x above
 * the noisiest real case (4.8e-5, a near-silent clip where the same absolute noise is a larger share). */
#define TOL_REL 2e-4
#define TOL_ABS 5e-5

static const char *g_dit = "acestep-v15-turbo";      /* --dit: the DiT the generation oracle, --make and --serve run on */
static const char *g_tag = "vae-oracle";      /* the oracle that is running: the prefix of its lines */
static double g_tol_rel = TOL_REL, g_tol_abs = TOL_ABS;      /* the limits of the oracle that is running */
static double g_cmp_scale = 0;      /* when set: a scale (a peak) the absolute limit is also a fraction of, besides the reference's own */

/* Compare a computed tensor with the reference against the given limits; returns 1 when it breaks either. The absolute
 * limit scales with the reference's peak where that exceeds 1 (a waveform is within [-1, 1]; a hidden state is not). */
static int cmp_tol(const char *what, const float *got, const float *want, size_t n, double tol_rel, double tol_abs){
    double emax = 0, en = 0, wn = 0, wpeak = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - want[i];
        if (fabs(d) > emax) emax = fabs(d);
        if (fabs((double)want[i]) > wpeak) wpeak = fabs((double)want[i]);
        en += d * d; wn += (double)want[i] * want[i];
    }
    double rel = wn > 0 ? sqrt(en / wn) : sqrt(en);
    if (g_cmp_scale > wpeak) wpeak = g_cmp_scale;
    int bad = rel > tol_rel || emax > tol_abs * (wpeak > 1.0 ? wpeak : 1.0);
    fprintf(stderr, "[%s] %-30s n=%-9zu max|err| %.3e  rel %.3e%s\n", g_tag, what, n, emax, rel, bad ? "  <-- FAIL" : "");
    return bad;
}
static int cmp(const char *what, const float *got, const float *want, size_t n){
    return cmp_tol(what, got, want, n, g_tol_rel, g_tol_abs);
}

/* A float tensor of the reference file, widened to f32; its element count in *n. */
static float *ref_tensor(shards *S, const char *name, int64_t *n, int64_t *shape, int *rank){
    st_tensor *t = st_find(S, name);
    if (!t) return NULL;
    if (t->dtype > 2 && t->dtype != 6 && t->dtype != 7) { fprintf(stderr, "[%s] %s is not a float, int32 or int64 tensor\n", g_tag, name); return NULL; }
    float *v = (float *)malloc((size_t)(t->numel > 0 ? t->numel : 1) * sizeof(float));
    if (!v) { fprintf(stderr, "[%s] out of memory\n", g_tag); exit(1); }
    if (t->dtype == 7) {            /* int32 (the quantizer's codes): exact in float below 2^24, and a code is below 2^18 */
        if (t->nbytes != t->numel * 4) { fprintf(stderr, "[%s] %s: shape and bytes disagree\n", g_tag, name); free(v); return NULL; }
        int32_t *raw = (int32_t *)malloc((size_t)(t->nbytes > 0 ? t->nbytes : 1));
        if (!raw) { fprintf(stderr, "[%s] out of memory\n", g_tag); exit(1); }
        st_read_raw(S, name, raw, 0);
        for (int64_t i = 0; i < t->numel; i++) v[i] = (float)raw[i];
        free(raw);
    } else if (t->dtype == 6) {     /* int64 (token ids): exact in float below 2^24, far above any vocabulary here */
        if (t->nbytes != t->numel * 8) { fprintf(stderr, "[%s] %s: shape and bytes disagree\n", g_tag, name); free(v); return NULL; }
        int64_t *raw = (int64_t *)malloc((size_t)(t->nbytes > 0 ? t->nbytes : 1));
        if (!raw) { fprintf(stderr, "[%s] out of memory\n", g_tag); exit(1); }
        st_read_raw(S, name, raw, 0);
        for (int64_t i = 0; i < t->numel; i++) v[i] = (float)raw[i];
        free(raw);
    } else st_read_f32(S, name, v, 0);
    if (n) *n = t->numel;
    if (rank) *rank = t->rank;
    if (shape) for (int i = 0; i < t->rank && i < ST_MAX_RANK; i++) shape[i] = t->shape[i];
    return v;
}

/* [n][c] interleaved -> [c][n] planar */
static float *planar(const float *x, size_t n, int c){
    float *p = (float *)malloc(n * c * sizeof(float));
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int ch = 0; ch < c; ch++)
        for (size_t i = 0; i < n; i++) p[(size_t)ch * n + i] = x[i * c + ch];
    return p;
}

/* 16-bit PCM WAV of interleaved f32 in [-1, 1]. */
static int write_wav16(const char *path, const float *x, size_t frames, int channels, int rate){
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t data = (uint32_t)(frames * channels * 2), riff = 36 + data, fmt = 16, br = (uint32_t)rate * channels * 2, sr = (uint32_t)rate;
    uint16_t pcm = 1, ch = (uint16_t)channels, align = (uint16_t)(channels * 2), bits = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmt, 4, 1, f);
    fwrite(&pcm, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&sr, 4, 1, f); fwrite(&br, 4, 1, f);
    fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f); fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
    for (size_t i = 0; i < frames * channels; i++) {
        float v = x[i] < -1.f ? -1.f : x[i] > 1.f ? 1.f : x[i];
        int16_t s = (int16_t)lrintf(v * 32767.f);
        fwrite(&s, 2, 1, f);
    }
    return fclose(f);
}

static int tile_progress(int done, int total, void *ud){
    (void)ud;
    if (total > 1 && done < total) fprintf(stderr, "\r[vae-oracle] tiled decode: window %d/%d", done + 1, total);
    else if (total > 1) fprintf(stderr, "\r%60s\r", "");
    return 0;
}

static int run_vae_oracle(const char *model, const char *refdir, int chunk, int overlap, const char *wav, const char *dump){
    char vdir[4096];
    snprintf(vdir, sizeof vdir, "%s/vae", model);
    AcVae *v = acv_load(vdir);
    if (!v) return 2;
    fprintf(stderr, "[vae-oracle] decoder: %d blocks, %d latent channels -> %d x %d Hz, hop %d, %.0f MB of f32 weights\n",
            v->nblk, v->zdim, v->ach, v->rate, v->hop, v->wbytes / 1e6);
    shards R; memset(&R, 0, sizeof R); st_init(&R, refdir);
    if (R.n == 0) { fprintf(stderr, "[vae-oracle] no safetensors in %s\n", refdir); acv_free(v); return 2; }
    int fails = 0, cases = 0;

    /* 1. every decode the handler made: its input latents [1][zdim][T] against its output [1][ach][N] */
    for (int k = 0;; k++) {
        char nin[32], nout[32]; int64_t n1, n2, sh[ST_MAX_RANK]; int rk;
        snprintf(nin, sizeof nin, "vae%d.in", k); snprintf(nout, sizeof nout, "vae%d.out", k);
        float *in = ref_tensor(&R, nin, &n1, sh, &rk);
        if (!in) break;
        float *want = ref_tensor(&R, nout, &n2, NULL, NULL);
        if (!want || rk != 3 || sh[0] != 1 || sh[1] != v->zdim) { fprintf(stderr, "[vae-oracle] %s: unexpected shape\n", nin); free(in); free(want); fails++; break; }
        int T = (int)sh[2];
        float *z = (float *)malloc((size_t)T * v->zdim * sizeof(float));
        for (int c = 0; c < v->zdim; c++)
            for (int t = 0; t < T; t++) z[(size_t)t * v->zdim + c] = in[(size_t)c * T + t];     /* [C][T] -> [T][C] */
        size_t N = (size_t)T * v->hop;
        if ((int64_t)(N * v->ach) != n2) { fprintf(stderr, "[vae-oracle] %s: %lld values, expected %zu\n", nout, (long long)n2, N * v->ach); fails++; free(in); free(want); free(z); continue; }
        float *out = (float *)malloc(N * v->ach * sizeof(float));
        double t0 = now();
        int rc = acv_decode(v, z, T, out);
        double dt = now() - t0;
        if (rc) { fprintf(stderr, "[vae-oracle] decode %d failed\n", k); fails++; }
        else {
            float *p = planar(out, N, v->ach);
            char label[64]; snprintf(label, sizeof label, "decode %d (%d frames, %.1fs)", k, T, dt);
            fails += cmp(label, p, want, N * v->ach);
            if (dump) {          /* planar f32, to compare with another reference than the captured one */
                char path[4096]; snprintf(path, sizeof path, "%s.decode%d.f32", dump, k);
                FILE *df = fopen(path, "wb");
                if (df) { fwrite(p, sizeof(float), N * v->ach, df); fclose(df); }
                else fprintf(stderr, "[vae-oracle] cannot write %s\n", path);
            }
            free(p);
        }
        cases++;
        free(in); free(want); free(z); free(out);
    }

    /* 2. the tiled decode of the whole case, the handler's peak limit, and what it returned */
    int64_t nl, na, shl[ST_MAX_RANK]; int rl;
    float *lat = ref_tensor(&R, "latents", &nl, shl, &rl);
    float *aud = ref_tensor(&R, "audio", &na, NULL, NULL);
    if (lat && aud && rl == 3 && shl[0] == 1 && shl[2] == v->zdim) {
        int F = (int)shl[1];
        size_t N = (size_t)F * v->hop;
        if ((int64_t)(N * v->ach) != na) { fprintf(stderr, "[vae-oracle] audio has %lld values, expected %zu\n", (long long)na, N * v->ach); fails++; }
        else {
            float *out = (float *)malloc(N * v->ach * sizeof(float));
            double t0 = now();
            int rc = acv_decode_tiled(v, lat, F, chunk, overlap, out, tile_progress, NULL);
            double dt = now() - t0;
            if (rc) { fprintf(stderr, "[vae-oracle] tiled decode failed\n"); fails++; }
            else {
                float peak = acv_normalize(out, N * v->ach);
                float *p = planar(out, N, v->ach);
                char label[64]; snprintf(label, sizeof label, "tiled %d frames (%.1fs)", F, dt);
                fails += cmp(label, p, aud, N * v->ach);
                fprintf(stderr, "[vae-oracle] raw peak %.4f%s\n", peak, peak > 1.0f ? " (divided by it)" : "");
                free(p);
                if (wav) {
                    if (write_wav16(wav, out, N, v->ach, v->rate)) { fprintf(stderr, "[vae-oracle] cannot write %s\n", wav); fails++; }
                    else fprintf(stderr, "[vae-oracle] wrote %s (%.2f s)\n", wav, (double)N / v->rate);
                }
                cases++;
            }
            free(out);
        }
    }
    free(lat); free(aud);
    st_destroy(&R);
    acv_free(v);
    if (!cases) { fprintf(stderr, "[vae-oracle] nothing to compare in %s\n", refdir); return 2; }
    if (fails) { fprintf(stderr, "[vae-oracle] %d of %d comparison(s) FAILED\n", fails, cases); return 1; }
    fprintf(stderr, "[vae-oracle] all %d comparison(s) within rel %.0e and abs %.0e\n", cases, g_tol_rel, g_tol_abs);
    return 0;
}

/* ---------------------------------------------------------- the VAE encoder oracle -- */

/* The encoder's limits. The network alone (its parameters against upstream's, one pass) and the whole tiled encode (windows,
 * trims, and the posterior draw from torch's own generator) are held separately, so a wrong layer and a wrong seam each show
 * where they are. Measured on the real checkpoint (single10, single_odd, single28, tile_*), as the ratio of the norms of the
 * error and the reference and as a fraction of the tensor's peak (the values peak near 6, the std below 1):
 *
 *   the network, one pass (parameters)  measured <= 6.0e-6 rel, 8.2e-5 of the peak (4.7e-4 abs)   held at 2e-5, 2e-4
 *   the std the sampler uses            measured <= 6.6e-5 rel, 1.0e-6 abs (std is 1e-4 to 1e-2: its relative error is
 *                                       the scale's absolute one)                                   held at 3e-4, 1e-5
 *   the tiled encode, sampled           measured <= 3.0e-5 rel, 8.2e-5 of the peak                  held at 1e-4, 2e-4
 *
 * The error is torch's, not ours: the same network in float64 (`acestep_ref_audio.py enc ... --f64`) is 1.7e-6 rel from this
 * encoder and 5.2e-6 from torch's float32 on a 10 s clip, 6.6e-7 and 2.1e-6 on a 28 s one. */
#define VENC_REL 2e-5
#define VENC_ABS 2e-4
#define VENC_STD_REL 3e-4
#define VENC_STD_ABS 1e-5
#define VENC_TILE_REL 1e-4
#define VENC_TILE_ABS 2e-4

static void venc_eps(float *dst, size_t n, void *ud){ acg_randn_mt((AcgMt *)ud, n, dst); }

/* wav [1][2][N] planar -> [N][2] interleaved */
static float *interleave2(const float *p, size_t n){
    float *x = (float *)malloc(n * 2 * sizeof(float));
    if (!x) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (size_t i = 0; i < n; i++) { x[i * 2] = p[i]; x[i * 2 + 1] = p[n + i]; }
    return x;
}

static int run_vae_enc_oracle(const char *model, const char *refdir){
    char vdir[4096];
    snprintf(vdir, sizeof vdir, "%s/vae", model);
    AcvEnc *e = acve_load(vdir);
    if (!e) return 2;
    g_tag = "vae-enc-oracle";
    fprintf(stderr, "[vae-enc-oracle] encoder: %d blocks, %d x %d Hz -> %d latent channels, hop %d\n", e->nblk, e->ach, e->rate, e->zdim, e->hop);
    shards R; memset(&R, 0, sizeof R); st_init(&R, refdir);
    if (R.n == 0) { fprintf(stderr, "[vae-enc-oracle] no safetensors in %s\n", refdir); acve_free(e); return 2; }
    int fails = 0, cases = 0;
    const int z = e->zdim;

    /* one pass: the posterior's parameters */
    for (int k = 0;; k++) {
        char nw[32], np[32]; int64_t nn, npar, sh[ST_MAX_RANK]; int rk;
        snprintf(nw, sizeof nw, "enc%d.wav", k); snprintf(np, sizeof np, "enc%d.params", k);
        float *wav = ref_tensor(&R, nw, &nn, sh, &rk);
        if (!wav) break;
        int64_t shp[ST_MAX_RANK]; int rp;
        float *want = ref_tensor(&R, np, &npar, shp, &rp);
        if (!want || rk != 3 || sh[1] != e->ach || rp != 3 || shp[1] != 2 * z) { fprintf(stderr, "[vae-enc-oracle] %s: unexpected shape\n", nw); fails++; free(wav); free(want); break; }
        const int64_t N = sh[2], T = shp[2];
        float *x = interleave2(wav, (size_t)N);
        float *got = (float *)malloc((size_t)(T > 0 ? T : 1) * 2 * z * sizeof(float));
        double t0 = now();
        int64_t Tc = acve_encode_raw(e, x, N, got);
        double dt = now() - t0;
        if (Tc != T) { fprintf(stderr, "[vae-enc-oracle] %lld samples gave %lld frames, upstream %lld\n", (long long)N, (long long)Tc, (long long)T); fails++; }
        else {
            float *p = planar(got, (size_t)T, 2 * z);
            char label[64];
            snprintf(label, sizeof label, "params %d (%lld samples, %.1fs)", k, (long long)N, dt);
            fails += cmp_tol(label, p, want, (size_t)T * 2 * z, VENC_REL, VENC_ABS);
            /* the std the sampler uses, through the sampler itself: z(eps = 1) - z(eps = mean only) is std, against upstream's
             * softplus of its own scale plus 1e-4 */
            float *ones = (float *)malloc((size_t)T * z * sizeof(float)), *z1 = (float *)malloc((size_t)T * z * sizeof(float)), *z0 = (float *)malloc((size_t)T * z * sizeof(float));
            float *gs = (float *)malloc((size_t)T * z * sizeof(float)), *ws = (float *)malloc((size_t)T * z * sizeof(float));
            for (int64_t i = 0; i < (int64_t)T * z; i++) ones[i] = 1.0f;
            acve_sample(got, T, z, ones, z1);                       /* z1[t][c]; eps is [z][T], here all ones */
            acve_sample(got, T, z, NULL, z0);
            for (int64_t t = 0; t < T; t++)
                for (int c = 0; c < z; c++) {
                    double b = want[(size_t)(z + c) * T + t];
                    gs[(size_t)c * T + t] = z1[(size_t)t * z + c] - z0[(size_t)t * z + c];
                    ws[(size_t)c * T + t] = (float)((b > 20.0 ? b : log1p(exp(b))) + 1e-4);
                }
            fails += cmp_tol("  std", gs, ws, (size_t)T * z, VENC_STD_REL, VENC_STD_ABS);
            free(ones); free(z1); free(z0);
            char n64[32]; snprintf(n64, sizeof n64, "enc%d.params64", k);
            float *w64 = ref_tensor(&R, n64, NULL, NULL, NULL);
            if (w64) {            /* captured with --f64: which of the two float32 results is nearer the float64 network */
                cmp_tol("  C vs float64", p, w64, (size_t)T * 2 * z, 1.0, 1e9);
                cmp_tol("  torch f32 vs float64", want, w64, (size_t)T * 2 * z, 1.0, 1e9);
                free(w64);
            }
            free(gs); free(ws); free(p);
        }
        cases++;
        free(wav); free(want); free(x); free(got);
    }

    /* tiled, sampled */
    for (int k = 0;; k++) {
        char nw[32], no[32], nc[32]; int64_t nn, nl, sh[ST_MAX_RANK], shl[ST_MAX_RANK]; int rk, rl;
        snprintf(nw, sizeof nw, "tile%d.wav", k); snprintf(no, sizeof no, "tile%d.out", k); snprintf(nc, sizeof nc, "tile%d.cfg", k);
        float *wav = ref_tensor(&R, nw, &nn, sh, &rk);
        if (!wav) break;
        float *want = ref_tensor(&R, no, &nl, shl, &rl);
        float *cfg = ref_tensor(&R, nc, NULL, NULL, NULL);
        if (!want || !cfg || rk != 3 || sh[1] != e->ach || rl != 3 || shl[1] != z) { fprintf(stderr, "[vae-enc-oracle] %s: unexpected shape\n", nw); fails++; free(wav); free(want); free(cfg); break; }
        const int64_t N = sh[2], F = shl[2], chunk_s = (int64_t)cfg[0], overlap_s = (int64_t)cfg[1];
        const uint32_t seed = (uint32_t)cfg[2];
        float *x = interleave2(wav, (size_t)N);
        float *got = (float *)malloc((size_t)(N / e->hop + 1) * z * sizeof(float));
        AcgMt m; acg_mt_seed(&m, seed);
        double t0 = now();
        int64_t Fc = acve_encode_tiled(e, x, N, chunk_s, overlap_s, venc_eps, &m, got, NULL, NULL);
        double dt = now() - t0;
        if (Fc != F) { fprintf(stderr, "[vae-enc-oracle] tile %d: %lld frames, upstream %lld\n", k, (long long)Fc, (long long)F); fails++; }
        else {
            float *p = planar(got, (size_t)F, z);
            char label[96];
            snprintf(label, sizeof label, "tiled %d (%lld samples, chunk %lld, %.1fs)", k, (long long)N, (long long)chunk_s, dt);
            fails += cmp_tol(label, p, want, (size_t)F * z, VENC_TILE_REL, VENC_TILE_ABS);
            if (getenv("ACESTEP_DEBUG_ENC")) {           /* where the error is: the worst frames, and the error per tenth of the clip */
                double worst[3] = {0, 0, 0}; int64_t wt[3] = {-1, -1, -1};
                for (int64_t t = 0; t < F; t++) {
                    double m = 0;
                    for (int c = 0; c < z; c++) { double d = fabs((double)p[(size_t)c * F + t] - want[(size_t)c * F + t]); if (d > m) m = d; }
                    for (int s = 0; s < 3; s++) if (m > worst[s]) { for (int u = 2; u > s; u--) { worst[u] = worst[u - 1]; wt[u] = wt[u - 1]; } worst[s] = m; wt[s] = t; break; }
                }
                fprintf(stderr, "[vae-enc-oracle]   worst frames: %lld (%.2e)  %lld (%.2e)  %lld (%.2e)\n", (long long)wt[0], worst[0], (long long)wt[1], worst[1], (long long)wt[2], worst[2]);
                for (int s = 0; s < 10; s++) {
                    int64_t a = F * s / 10, b = F * (s + 1) / 10; double m = 0, pk = 0, rel = 0;
                    for (int64_t t = a; t < b; t++) for (int c = 0; c < z; c++) {
                        double w = fabs((double)want[(size_t)c * F + t]), d = fabs((double)p[(size_t)c * F + t] - want[(size_t)c * F + t]);
                        if (d > m) m = d;
                        if (w > pk) pk = w;
                        if (d / (w + 1.0) > rel) rel = d / (w + 1.0);
                    }
                    fprintf(stderr, "[vae-enc-oracle]   frames %4lld..%4lld  max|err| %.2e  peak|ref| %.2f  max err/(|ref|+1) %.2e\n", (long long)a, (long long)b, m, pk, rel);
                }
            }
            free(p);
        }
        cases++;
        free(wav); free(want); free(cfg); free(x); free(got);
    }
    st_destroy(&R);
    acve_free(e);
    if (!cases) { fprintf(stderr, "[vae-enc-oracle] nothing to compare in %s\n", refdir); return 2; }
    if (fails) { fprintf(stderr, "[vae-enc-oracle] %d comparison(s) FAILED in %d case(s)\n", fails, cases); return 1; }
    fprintf(stderr, "[vae-enc-oracle] all %d case(s) within the limits\n", cases);
    return 0;
}

/* ------------------------------------------------- the cover tokenizer oracle -- */

/* The limits of the tokenizer and detokenizer, each stage fed upstream's own input (so a stage's error is its own), then the
 * whole path. Measured on the real checkpoint (cover10, cover_odd, cover_vocal, cover_half, cover_noise), as the ratio of the
 * norms of the error and the reference and as a fraction of the tensor's peak:
 *
 *   the pooler (2 encoder layers over 6 tokens)   measured <= 1.4e-6 rel, 3.5e-5 abs   held at 5e-6, 1e-4
 *   the quantizer, fed upstream's pooled vectors  codes identical in every window; output 6.1e-8 rel  (float rounding only)
 *   the detokenizer                                measured <= 2.3e-7 rel, 2.4e-6 abs   held at 1e-6, 1e-5
 *   the whole path, source latents -> context     measured <= 2.2e-7 rel, 2.4e-6 abs   held at 1e-6, 1e-5
 *
 * (The quantizer's floor makes one code of one window differ when a value is within float rounding of a boundary; none did in
 * the 5 cases, 261 windows. Such a window is accepted when a value was within 1e-3 of a boundary, and only then.) */
#define TOK_POOL_REL 5e-6
#define TOK_POOL_ABS 1e-4
#define TOK_DETOK_REL 1e-6
#define TOK_DETOK_ABS 1e-5
#define TOK_CHAIN_REL 1e-6
#define TOK_CHAIN_ABS 1e-5

/* The quantizer's integer codes (the FSQ index of a window) from the brackets atk_fsq returns: sum of floor(bracket_i) * basis_i,
 * the basis the cumulative product of the levels before it (vector_quantize_pytorch's codes_to_indices). */
static int64_t tok_index(const AcdTok *t, const float *bracket){
    int64_t idx = 0, basis = 1;
    for (int i = 0; i < t->nlev; i++) { idx += (int64_t)floorf(bracket[i]) * basis; basis *= t->levels[i]; }
    return idx;
}

static int run_cover_oracle(const char *model, const char *refdir){
    g_tag = "cover-oracle";
    AceDit *d = acd_load(model);
    if (!d) return 2;
    AcdTok *t = atk_load(d, model);
    if (!t) { acd_free(d); return 2; }
    shards R; memset(&R, 0, sizeof R); st_init(&R, refdir);
    if (R.n == 0) { fprintf(stderr, "[cover-oracle] no safetensors in %s\n", refdir); atk_free(t); acd_free(d); return 2; }
    int fails = 0, cases = 0;
    const int D = d->D, A = d->acdim, P = t->P;
    int64_t nx, sx[ST_MAX_RANK]; int rx;
    float *x = ref_tensor(&R, "tok.x", &nx, sx, &rx);
    if (!x) { fprintf(stderr, "[cover-oracle] %s has no tokenizer capture (tok.x)\n", refdir); st_destroy(&R); atk_free(t); acd_free(d); return 2; }
    if (rx != 4 || sx[2] != P || sx[3] != A) { fprintf(stderr, "[cover-oracle] tok.x has an unexpected shape\n"); fails++; }
    else {
        const int W = (int)sx[1];
        int64_t n;
        float *pooled = ref_tensor(&R, "tok.pooled", &n, NULL, NULL);
        float *q = ref_tensor(&R, "tok.q", NULL, NULL, NULL);
        float *idx = ref_tensor(&R, "tok.idx", NULL, NULL, NULL);
        float *dout = ref_tensor(&R, "detok.out", NULL, NULL, NULL);
        if (!pooled || !q || !idx || !dout || n != (int64_t)W * D) { fprintf(stderr, "[cover-oracle] the tokenizer capture is incomplete\n"); fails++; }
        else {
            /* 1. the pooler, fed the padded source frames */
            float *got = (float *)malloc((size_t)W * D * sizeof(float));
            double t0 = now();
            atk_pool(d, t, x, W, got);
            char label[96]; snprintf(label, sizeof label, "pooler (%d windows, %.1fs)", W, now() - t0);
            fails += cmp_tol(label, got, pooled, (size_t)W * D, TOK_POOL_REL, TOK_POOL_ABS); cases++;
            free(got);
            /* 2. the quantizer, fed upstream's pooled vectors: the codes must be upstream's except where a value sat on a boundary */
            float *qq = (float *)malloc((size_t)W * D * sizeof(float)), *br = (float *)malloc((size_t)W * t->nlev * sizeof(float));
            atk_fsq(d, t, pooled, W, qq, br);
            int flips = 0, unexplained = 0;
            for (int w = 0; w < W; w++) {
                if (tok_index(t, br + (size_t)w * t->nlev) == (int64_t)idx[w]) continue;
                flips++;
                double nearest = 1e9;
                for (int i = 0; i < t->nlev; i++) { double b = br[(size_t)w * t->nlev + i], f = fabs(b - floor(b + 0.5)); if (f < nearest) nearest = f; }
                if (nearest > 1e-3) unexplained++;        /* a code that differs while no value was within 1e-3 of a boundary is a wrong quantizer */
            }
            fprintf(stderr, "[cover-oracle] %-30s %d of %d windows differ in code, %d of them with no value near a boundary%s\n", "quantizer codes", flips, W, unexplained, unexplained ? "  <-- FAIL" : "");
            fails += unexplained ? 1 : 0; cases++;
            if (!flips) { fails += cmp_tol("quantizer output", qq, q, (size_t)W * D, TOK_POOL_REL, TOK_POOL_ABS); cases++; }
            free(qq); free(br);
            /* 3. the detokenizer, fed upstream's quantized vectors */
            float *dd = (float *)malloc((size_t)W * P * A * sizeof(float));
            t0 = now();
            atk_detok(d, t, q, W, dd);
            snprintf(label, sizeof label, "detokenizer (%d windows, %.1fs)", W, now() - t0);
            fails += cmp_tol(label, dd, dout, (size_t)W * P * A, TOK_DETOK_REL, TOK_DETOK_ABS); cases++;
            free(dd);
            /* 4. the whole path from the source latents the handler had, cut to its length, against what went into the DiT */
            int64_t ns, ss[ST_MAX_RANK]; int rs;
            float *src = ref_tensor(&R, "in.src_latents", &ns, ss, &rs);
            float *sil = ref_tensor(&R, "in.silence_latent", NULL, NULL, NULL);
            float *ctx = ref_tensor(&R, "context", NULL, NULL, NULL);
            if (src && sil && ctx && rs == 3 && ss[2] == A) {
                const int T = (int)ss[1];
                float *h = (float *)malloc((size_t)T * A * sizeof(float)), *want = (float *)malloc((size_t)T * A * sizeof(float));
                t0 = now();
                atk_hints(d, t, src, T, sil, h);
                for (int f = 0; f < T; f++) memcpy(want + (size_t)f * A, ctx + (size_t)f * 2 * A, (size_t)A * sizeof(float));   /* the context's first half is the hints */
                snprintf(label, sizeof label, "whole path -> context (%d frames, %.1fs)", T, now() - t0);
                fails += cmp_tol(label, h, want, (size_t)T * A, TOK_CHAIN_REL, TOK_CHAIN_ABS); cases++;
                free(h); free(want);
            }
            free(src); free(sil); free(ctx);
        }
        free(pooled); free(q); free(idx); free(dout);
    }
    free(x);
    st_destroy(&R); atk_free(t); acd_free(d);
    if (!cases) { fprintf(stderr, "[cover-oracle] nothing to compare in %s\n", refdir); return 2; }
    if (fails) { fprintf(stderr, "[cover-oracle] %d comparison(s) FAILED\n", fails); return 1; }
    fprintf(stderr, "[cover-oracle] all %d comparison(s) within the limits\n", cases);
    return 0;
}

/* ------------------------------------------------------------ the DiT oracle -- */

/* A mask the reference stores as U8 (1 or 0) -> f32. */
static float *ref_mask(shards *S, const char *name, int64_t *n){
    st_tensor *t = st_find(S, name);
    if (!t) return NULL;
    if (t->dtype != 3) return ref_tensor(S, name, n, NULL, NULL);
    uint8_t *raw = (uint8_t *)malloc((size_t)(t->nbytes > 0 ? t->nbytes : 1));
    float *v = (float *)malloc((size_t)(t->numel > 0 ? t->numel : 1) * sizeof(float));
    if (!raw || !v) { fprintf(stderr, "[%s] out of memory\n", g_tag); exit(1); }
    st_read_raw(S, name, raw, 0);
    for (int64_t i = 0; i < t->numel; i++) v[i] = raw[i] ? 1.f : 0.f;
    free(raw);
    if (n) *n = t->numel;
    return v;
}

static float *ref_named(shards *S, const char *fmt, int i, int64_t *n){
    char name[64]; snprintf(name, sizeof name, fmt, i);
    return ref_tensor(S, name, n, NULL, NULL);
}

/* The DiT's limits, from what was measured on the real checkpoint (tools/acestep_ref.py capture, all 11 cases), as the
 * ratio of the norms of the error and the reference, and as a fraction of the tensor's peak for the largest single error.
 *
 *   a layer ALONE, fed torch's own input     measured <= 3.3e-7 rel (typically 1e-7: float32 rounding)   held at 2e-6, 2e-5
 *   a DiT call, fed the handler's own input   measured <= 8.2e-6 rel, 2e-5 of the peak (at t = 1, where the
 *                                              network is most sensitive; 24 layers chain the layers' rounding)   5e-5, 1e-4
 *   the sampler's UPDATE alone (x - v dt, DCW)  measured 7e-8 rel, 4.8e-7 abs: plain float32 arithmetic on the
 *                                              handler's own x and v, no network in it                               1e-6, 2e-6
 *   the CHAINED sampler (steps feeding steps)   measured <= 1.8e-4 rel, 7e-4 of the peak: the loop amplifies the
 *                                              calls' rounding (8e-6, then 2.6e-5, 3.8e-5, 1.8e-4 over the five
 *                                              steps of steps5_nodcw, whose steps are the longest); each call and
 *                                              each update is checked tight above, so this one only has to show the
 *                                              pieces compose, and a logic mistake is wrong by 1e-2 or more         5e-4, 2e-3 */
#define DIT_REL 5e-5
#define DIT_ABS 1e-4
#define DIT_LAYER_REL 2e-6
#define DIT_LAYER_ABS 2e-5
#define DIT_UPDATE_REL 1e-6
#define DIT_UPDATE_ABS 2e-6
#define DIT_CHAIN_REL 5e-4
#define DIT_CHAIN_ABS 2e-3

typedef struct { shards *R; int fails; int cmp_chain; } DitCheck;

static const char *g_dump = NULL;       /* --dump PREFIX: write what is compared, for another reference */

/* --taps captures: the hidden state after the patch embedding and each layer of the first call, against ours. */
static void dit_tap_check(int layer, const float *h, int Tp, int D, void *ud){
    DitCheck *c = (DitCheck *)ud;
    char name[64], label[64]; int64_t n;
    if (layer < 0) snprintf(name, sizeof name, "dit0.tap.proj_in");
    else if (layer >= 2000) snprintf(name, sizeof name, "dit0.tap.attn%d", layer - 2000);
    else if (layer >= 1000) snprintf(name, sizeof name, "dit0.tap.%s%d", (layer & 1) ? "ca" : "sa", (layer - 1000) / 2);
    else snprintf(name, sizeof name, "dit0.tap.layer%d", layer);
    if (g_dump) {
        char path[4096]; snprintf(path, sizeof path, "%s.%s.f32", g_dump, name);
        FILE *df = fopen(path, "wb");
        if (df) { fwrite(h, sizeof(float), (size_t)Tp * D, df); fclose(df); }
    }
    float *want = ref_tensor(c->R, name, &n, NULL, NULL);
    if (!want) return;
    if (n != (int64_t)Tp * D) { fprintf(stderr, "[dit-oracle] %s has %lld values, expected %d\n", name, (long long)n, Tp * D); c->fails++; free(want); return; }
    if (layer < 0) snprintf(label, sizeof label, "tap: patch embedding"); else snprintf(label, sizeof label, "tap: after layer %d", layer);
    c->cmp_chain++;
    c->fails += cmp(label, h, want, (size_t)n);
    free(want);
}

/* The sampler's view of each step: the latents it fed the DiT and the velocity, against the captured ones. */
static void dit_step_check(int step, int steps, const float *xt, const float *v, void *ud){
    DitCheck *c = (DitCheck *)ud;
    int64_t n;
    char label[64];
    float *rx = ref_named(c->R, "dit%d.xt", step, &n);
    if (rx) { snprintf(label, sizeof label, "chain step %d/%d latents in", step, steps); c->fails += cmp_tol(label, xt, rx, (size_t)n, DIT_CHAIN_REL, DIT_CHAIN_ABS); free(rx); }
    float *rv = ref_named(c->R, "dit%d.v", step, &n);
    if (rv) { snprintf(label, sizeof label, "chain step %d/%d velocity", step, steps); c->fails += cmp_tol(label, v, rv, (size_t)n, DIT_CHAIN_REL, DIT_CHAIN_ABS); free(rv); }
}

static int run_dit_oracle(const char *model, const char *refdir, const char *dump){
    g_tag = "dit-oracle"; g_dump = dump; g_tol_rel = DIT_REL; g_tol_abs = DIT_ABS;
    AceDit *d = acd_load(model);
    if (!d) return 2;
    fprintf(stderr, "[dit-oracle] %d layers, %d wide, %d heads (%d key/value), window %d, lyric %d + timbre %d encoder layers, %.0f MB of weights\n",
            d->nl, d->D, d->heads, d->kv, d->window, d->nlyric, d->ntimbre, d->wbytes / 1e6);
    shards R; memset(&R, 0, sizeof R); st_init(&R, refdir);
    if (R.n == 0) { fprintf(stderr, "[dit-oracle] no safetensors in %s\n", refdir); acd_free(d); return 2; }
    int fails = 0, checks = 0;
    int64_t n, sh[ST_MAX_RANK]; int rk;

    /* settings of the case: steps, shift, DCW, from ref.json */
    int steps = 8; float shift = 1.0f; AcdSampleOpts so = {1, 0.05, 0.02};
    {
        char path[4096]; snprintf(path, sizeof path, "%s/ref.json", refdir);
        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
            char *text = (char *)malloc(len > 0 ? len + 1 : 1);
            if (text && len > 0 && len < (8 << 20) && fread(text, 1, len, f) == (size_t)len) {
                text[len] = 0;
                jval *root = json_parse_checked(text);
                jval *p = root ? json_get(root, "params") : NULL;
                if (p && p->t == J_OBJ) {
                    steps = acd_json_int(p, "inference_steps", steps);
                    shift = (float)acd_json_num(p, "shift", shift);
                    jval *dc = json_get(p, "dcw_enabled");
                    so.dcw = dc && dc->t == J_BOOL ? dc->boolean : so.dcw;
                    so.dcw_low = acd_json_num(p, "dcw_scaler", so.dcw_low);
                    so.dcw_high = acd_json_num(p, "dcw_high_scaler", so.dcw_high);
                }
                json_free(root);
            }
            free(text); fclose(f);
        }
    }
    fprintf(stderr, "[dit-oracle] case: %d steps, shift %g, DCW %s\n", steps, shift, so.dcw ? "on" : "off");

    float *text = ref_tensor(&R, "in.text_hidden_states", &n, sh, &rk);
    int Lt = text && rk == 3 ? (int)sh[1] : 0;
    float *lyric = ref_tensor(&R, "in.lyric_hidden_states", &n, sh, &rk);
    int Ll = lyric && rk == 3 ? (int)sh[1] : 0;
    float *timbre = ref_tensor(&R, "in.refer_audio_acoustic_hidden_states_packed", &n, sh, &rk);
    int Tr = timbre && rk == 3 ? (int)sh[1] : 0;
    float *src = ref_tensor(&R, "in.src_latents", &n, sh, &rk);
    int T = src && rk == 3 ? (int)sh[1] : 0;
    float *mask = ref_mask(&R, "in.chunk_masks", &n);
    float *noise = ref_tensor(&R, "noise", NULL, NULL, NULL);
    if (!text || !lyric || !timbre || !src || !mask || !noise || Lt < 1 || Ll < 1 || Tr < 1 || T < 1) {
        fprintf(stderr, "[dit-oracle] the case lacks in.text_hidden_states, in.lyric_hidden_states, in.refer_audio..., in.src_latents, in.chunk_masks or noise\n");
        acd_free(d); return 2;
    }

    /* 1. the condition: lyric, timbre, text packed */
    double t0 = now();
    AcdCond *c = acd_cond_new(d, text, Lt, lyric, Ll, timbre, Tr);
    if (!c) { fprintf(stderr, "[dit-oracle] condition failed\n"); acd_free(d); return 1; }
    double dt = now() - t0;
    float *renc = ref_tensor(&R, "encoder.hidden", &n, sh, &rk);
    if (renc) {
        char label[64]; snprintf(label, sizeof label, "condition (%d tokens, %.1fs)", c->len, dt);
        if (rk != 3 || sh[1] != c->len || n != (int64_t)c->len * d->D) { fprintf(stderr, "[dit-oracle] encoder.hidden has the wrong shape\n"); fails++; }
        else fails += cmp(label, c->enc, renc, (size_t)n);
        checks++;
    }
    free(renc);

    /* 2. the context: source latents ++ mask */
    const int A = d->acdim;
    float *ctx = (float *)malloc((size_t)T * 2 * A * sizeof(float));
    if (!ctx) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int f = 0; f < T; f++) {
        memcpy(ctx + (size_t)f * 2 * A, src + (size_t)f * A, (size_t)A * sizeof(float));
        memcpy(ctx + (size_t)f * 2 * A + A, mask + (size_t)f * A, (size_t)A * sizeof(float));
    }
    float *rctx = ref_tensor(&R, "context", &n, NULL, NULL);
    if (rctx) { fails += cmp("context", ctx, rctx, (size_t)n); checks++; }
    free(rctx);

    /* 2b. the DiT's RoPE table: positions 0..tokens-1, the angles and their cos and sin, float32 as torch builds them */
    {
        int64_t nc, ns2; int rk2; int64_t sh2[ST_MAX_RANK];
        float *rc = ref_tensor(&R, "dit0.rope_cos", &nc, sh2, &rk2), *rs = ref_tensor(&R, "dit0.rope_sin", &ns2, NULL, NULL);
        if (rc && rs && rk2 == 3 && sh2[2] == d->hd) {
            int Tp = (int)sh2[1];
            float *mc = (float *)malloc((size_t)Tp * d->hd * sizeof(float)), *ms = (float *)malloc((size_t)Tp * d->hd * sizeof(float));
            acd_rope_table(mc, ms, Tp, d->hd, d->theta);
            fails += cmp("rope cos", mc, rc, (size_t)nc); fails += cmp("rope sin", ms, rs, (size_t)ns2); checks += 2;
            free(mc); free(ms);
        }
        free(rc); free(rs);
    }

    /* 3. each DiT call the handler made, fed the latents and timesteps it was fed (no error carried from step to step) */
    float *v = (float *)malloc((size_t)T * A * sizeof(float));
    for (int i = 0;; i++) {
        float *xt = ref_named(&R, "dit%d.xt", i, NULL), *tt = ref_named(&R, "dit%d.t", i, NULL);
        float *rr = ref_named(&R, "dit%d.r", i, NULL), *want = ref_named(&R, "dit%d.v", i, &n);
        if (!xt || !tt || !rr || !want) { free(xt); free(tt); free(rr); free(want); break; }
        /* the timestep embeddings of this call, before anything else: a wrong phase here would colour every layer */
        for (int which = 0; which < 2; which++) {
            char nt[32], np[32]; int64_t n1, n2;
            snprintf(nt, sizeof nt, "dit%d.temb_%c", i, which ? 'r' : 't'); snprintf(np, sizeof np, "dit%d.proj_%c", i, which ? 'r' : 't');
            float *wt = ref_tensor(&R, nt, &n1, NULL, NULL), *wp = ref_tensor(&R, np, &n2, NULL, NULL);
            if (wt && wp && n1 == d->D && n2 == 6 * (int64_t)d->D) {
                float *gt = (float *)malloc((size_t)d->D * sizeof(float)), *gp = (float *)malloc((size_t)6 * d->D * sizeof(float));
                acd_time_embed(d, which ? &d->ter : &d->te, which ? tt[0] - rr[0] : tt[0], gt, gp);
                char lab[64];
                snprintf(lab, sizeof lab, "time embed %c, step %d", which ? 'r' : 't', i); fails += cmp(lab, gt, wt, (size_t)d->D);
                snprintf(lab, sizeof lab, "modulation %c, step %d", which ? 'r' : 't', i); fails += cmp(lab, gp, wp, (size_t)6 * d->D);
                checks += 2; free(gt); free(gp);
            }
            free(wt); free(wp);
        }
        t0 = now();
        acd_forward(d, c, xt, ctx, T, tt[0], rr[0], v);
        dt = now() - t0;
        char label[64]; snprintf(label, sizeof label, "DiT %d, t=%.4f (%.1fs)", i, tt[0], dt);
        fails += cmp(label, v, want, (size_t)n); checks++;
        if (dump) {          /* f32 [T][A], to compare with another reference than the captured one */
            char path[4096]; snprintf(path, sizeof path, "%s.dit%d.f32", dump, i);
            FILE *df = fopen(path, "wb");
            if (df) { fwrite(v, sizeof(float), (size_t)n, df); fclose(df); } else fprintf(stderr, "[dit-oracle] cannot write %s\n", path);
        }
        free(xt); free(tt); free(rr); free(want);
    }

    /* 3b. with --taps captures: the first call again, hidden state by hidden state */
    if (st_find(&R, "dit0.tap.proj_in")) {
        float *xt = ref_named(&R, "dit0.xt", 0, NULL), *tt = ref_named(&R, "dit0.t", 0, NULL), *rr = ref_named(&R, "dit0.r", 0, NULL);
        if (xt && tt && rr) {
            DitCheck tc = {&R, 0, 0};
            const char *dbg = getenv("ACESTEP_DEBUG_LAYER");        /* tap the attention stages of this layer too */
            acd_dbg_layer = dbg ? atoi(dbg) : -1;
            acd_tap_fn = dit_tap_check; acd_tap_ud = &tc;
            acd_forward(d, c, xt, ctx, T, tt[0], rr[0], v);
            acd_tap_fn = NULL; acd_tap_ud = NULL; acd_dbg_layer = -1;
            fails += tc.fails; checks += tc.cmp_chain;
        }
        free(xt); free(tt); free(rr);

        /* 3c. each layer ALONE, fed the reference's own input, modulation and RoPE table: the error of one layer's arithmetic,
         * with nothing carried in from the layers before it (3b's error is that, plus what 24 layers make of it) */
        int64_t np_, nr_; int rkc; int64_t shc[ST_MAX_RANK];
        float *pt = ref_tensor(&R, "dit0.proj_t", &np_, NULL, NULL), *pr = ref_tensor(&R, "dit0.proj_r", &nr_, NULL, NULL);
        float *rcos = ref_tensor(&R, "dit0.rope_cos", &nr_, shc, &rkc), *rsin = ref_tensor(&R, "dit0.rope_sin", &nr_, NULL, NULL);
        float *pin = ref_tensor(&R, "dit0.tap.proj_in", &np_, NULL, NULL);
        if (pt && pr && rcos && rsin && pin && rkc == 3) {
            const int Tp = (int)shc[1], D = d->D;
            float *proj = (float *)malloc((size_t)6 * D * sizeof(float)), *h = (float *)malloc((size_t)Tp * D * sizeof(float));
            for (int i = 0; i < 6 * D; i++) proj[i] = pt[i] + pr[i];
            for (int k = 0; k < d->nl; k++) {
                char in_name[48], out_name[48]; int64_t n1, n2;
                if (k == 0) snprintf(in_name, sizeof in_name, "dit0.tap.proj_in"); else snprintf(in_name, sizeof in_name, "dit0.tap.layer%d", k - 1);
                snprintf(out_name, sizeof out_name, "dit0.tap.layer%d", k);
                float *hin = ref_tensor(&R, in_name, &n1, NULL, NULL), *hout = ref_tensor(&R, out_name, &n2, NULL, NULL);
                if (!hin || !hout || n1 != (int64_t)Tp * D || n2 != n1) { free(hin); free(hout); break; }
                memcpy(h, hin, (size_t)Tp * D * sizeof(float));
                acd_layer(d, c, k, h, Tp, proj, rcos, rsin);
                char lab[64]; snprintf(lab, sizeof lab, "layer %d alone", k);
                fails += cmp_tol(lab, h, hout, (size_t)n1, DIT_LAYER_REL, DIT_LAYER_ABS); checks++;
                free(hin); free(hout);
            }
            free(proj); free(h);
        }
        free(pt); free(pr); free(rcos); free(rsin); free(pin);
    }

    /* 3d. the sampler's update ALONE, fed the handler's own latents and velocity at each step, against the handler's next
     * latents (the clean ones on the last step): the Euler step, the straight jump to x0 and DCW checked without the
     * network's noise being amplified through the loop (the chained run in 4 shows that amplification) */
    int nsched = steps > 20 ? 20 : steps;
    float sched[20]; acd_schedule(steps, shift, sched);
    {
        float *got = (float *)malloc((size_t)T * A * sizeof(float));
        for (int i = 0; i < nsched && got; i++) {
            int64_t nn;
            float *x = ref_named(&R, "dit%d.xt", i, &nn), *vv = ref_named(&R, "dit%d.v", i, NULL);
            float *want = i < nsched - 1 ? ref_named(&R, "dit%d.xt", i + 1, NULL) : ref_tensor(&R, "latents", NULL, NULL, NULL);
            if (x && vv && want) {
                acd_sample_step(x, vv, i, nsched, sched, T, A, &so, got);
                char label[64]; snprintf(label, sizeof label, "sampler update %d/%d alone", i, nsched);
                fails += cmp_tol(label, got, want, (size_t)nn, DIT_UPDATE_REL, DIT_UPDATE_ABS); checks++;
            }
            free(x); free(vv); free(want);
        }
        free(got);
    }

    /* 4. the sampler from the captured noise: every step's input and velocity, then the latents */
    int sched_ok = 1;
    for (int i = 0; i < nsched; i++) {
        float *tt = ref_named(&R, "dit%d.t", i, NULL);
        if (!tt || tt[0] != sched[i]) { sched_ok = 0; fprintf(stderr, "[dit-oracle] schedule step %d: %.9g, the handler used %.9g\n", i, sched[i], tt ? tt[0] : -1.f); }
        free(tt);
    }
    fprintf(stderr, "[dit-oracle] schedule of %d steps: %s\n", nsched, sched_ok ? "IDENTICAL to the handler's" : "DIFFERENT"); fails += !sched_ok; checks++;
    float *lat = (float *)malloc((size_t)T * A * sizeof(float));
    DitCheck dc = {&R, 0, 0};
    t0 = now();
    int ns = acd_sample(d, c, noise, ctx, T, steps, shift, &so, dit_step_check, &dc, lat);
    dt = now() - t0;
    fails += dc.fails; checks += 2 * ns;
    if (dump && ns) {          /* the chain's final latents: decode them with the VAE to compare the audio end to end */
        char path[4096]; snprintf(path, sizeof path, "%s.latents.f32", dump);
        FILE *df = fopen(path, "wb");
        if (df) { fwrite(lat, sizeof(float), (size_t)T * A, df); fclose(df); } else fprintf(stderr, "[dit-oracle] cannot write %s\n", path);
    }
    float *rlat = ref_tensor(&R, "latents", &n, NULL, NULL);
    if (rlat && ns) { char label[64]; snprintf(label, sizeof label, "chain latents (%.1fs)", dt); fails += cmp_tol(label, lat, rlat, (size_t)n, DIT_CHAIN_REL, DIT_CHAIN_ABS); checks++; }
    free(rlat); free(lat); free(v); free(ctx);
    free(text); free(lyric); free(timbre); free(src); free(mask); free(noise);
    acd_cond_free(c); acd_free(d); st_destroy(&R);
    if (!checks) { fprintf(stderr, "[dit-oracle] nothing to compare in %s\n", refdir); return 2; }
    if (fails) { fprintf(stderr, "[dit-oracle] %d of %d comparison(s) FAILED\n", fails, checks); return 1; }
    fprintf(stderr, "[dit-oracle] all %d comparison(s) within their limits (a call: rel %.0e, abs %.0e; a layer alone: %.0e, %.0e; an update alone: %.0e, %.0e; the chain: %.0e, %.0e)\n",
            checks, g_tol_rel, g_tol_abs, DIT_LAYER_REL, DIT_LAYER_ABS, DIT_UPDATE_REL, DIT_UPDATE_ABS, DIT_CHAIN_REL, DIT_CHAIN_ABS);
    return 0;
}

/* ---------------------------------------------------------- the tokenizer oracle -- */

/* A short, printable view of a string for a report: control bytes and non-ASCII shown as escapes. */
static void show_text(const char *s, char *out, size_t cap){
    size_t o = 0;
    for (; *s && o + 8 < cap; s++) {
        unsigned char c = (unsigned char)*s;
        if (o > 70) { o += (size_t)snprintf(out + o, cap - o, "..."); break; }
        if (c == '\n') o += (size_t)snprintf(out + o, cap - o, "\\n");
        else if (c < 32 || c >= 127) o += (size_t)snprintf(out + o, cap - o, "\\x%02x", c);
        else out[o++] = (char)c;
    }
    out[o < cap ? o : cap - 1] = 0;
}

static int run_tok_oracle(const char *model, const char *file){
    g_tag = "tok-oracle";
    AceTok T;
    if (act_tok_open(&T, model)) return 2;
    FILE *f = fopen(file, "rb");
    if (!f) { fprintf(stderr, "[tok-oracle] cannot open %s\n", file); act_tok_close(&T); return 2; }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = (char *)malloc(len > 0 ? len + 1 : 1);
    if (!text || len <= 0 || len > (256L << 20) || fread(text, 1, len, f) != (size_t)len) { fclose(f); free(text); fprintf(stderr, "[tok-oracle] cannot read %s\n", file); act_tok_close(&T); return 2; }
    fclose(f); text[len] = 0;
    jval *root = json_parse_checked(text);
    if (!root || root->t != J_ARR) { fprintf(stderr, "[tok-oracle] %s is not a JSON array\n", file); free(text); act_tok_close(&T); return 2; }
    int fails = 0, cases = 0, cut = 0;
    for (int i = 0; i < root->len; i++) {
        jval *c = root->kids[i], *s = json_get(c, "text"), *m = json_get(c, "max_length"), *ids = json_get(c, "ids");
        if (!s || s->t != J_STR || !m || m->t != J_NUM || !ids || ids->t != J_ARR || m->num < 1 || m->num > 1e6) { fprintf(stderr, "[tok-oracle] case %d is malformed\n", i); fails++; continue; }
        int max = (int)m->num;
        int *got = (int *)malloc((size_t)max * sizeof(int));
        if (!got) { fprintf(stderr, "[tok-oracle] out of memory\n"); exit(1); }
        int n = act_tokenize(&T, s->str, max, got);
        int same = n == ids->len;
        int first = -1;
        for (int k = 0; k < n && k < ids->len; k++)
            if (got[k] != (int)ids->kids[k]->num) { same = 0; first = k; break; }
        cases++; if (n == max) cut++;
        if (!same) {
            char view[160]; show_text(s->str, view, sizeof view);
            fprintf(stderr, "[tok-oracle] case %d DIFFERS (max %d): got %d ids, want %d; first difference at %d (got %d, want %d); text \"%s\"\n",
                    i, max, n, ids->len, first, first >= 0 ? got[first] : -1, first >= 0 ? (int)ids->kids[first]->num : -1, view);
            fails++;
        }
        free(got);
    }
    json_free(root); free(text); act_tok_close(&T);
    if (!cases) { fprintf(stderr, "[tok-oracle] nothing to compare in %s\n", file); return 2; }
    if (fails) { fprintf(stderr, "[tok-oracle] %d of %d case(s) DIFFER\n", fails, cases); return 1; }
    fprintf(stderr, "[tok-oracle] all %d cases identical (%d of them cut at their limit)\n", cases, cut);
    return 0;
}

/* ------------------------------------------------------------ the text oracle -- */

/* The text encoder's limits, from what was measured on the real checkpoint (all 16 cases), as in the DiT's table.
 *
 *   a layer ALONE, fed torch's own input  measured 5.8e-9 to 9.2e-7 rel (float32 rounding, no growth with depth) and, as a
 *                                          fraction of the larger of its input's and output's peak, <= 5e-7: layers 7 to
 *                                          26 carry "massive activations" (peaks near 6000) and layer 27 collapses them
 *                                          (peak 247), so its error is the 6000-sized values' rounding, not 1e-5 of 247     5e-6, 5e-6
 *   the whole encoder (28 layers chained)  measured <= 1.1e-5 rel, <= 4e-5 of the final peak (256 tokens, the longest)       5e-5, 2e-4
 *   the embedding rows                     exact (a table lookup)                                                            1e-7, 1e-7 */
#define TEXT_REL 5e-5
#define TEXT_ABS 2e-4
#define TEXT_LAYER_REL 5e-6
#define TEXT_LAYER_ABS 5e-6

/* A string field of a JSON object, or NULL. */
static const char *json_str(jval *o, const char *key){
    jval *v = o ? json_get(o, key) : NULL;
    return v && v->t == J_STR ? v->str : NULL;
}

static int same_ids(const char *what, const int *got, int n, const float *want, int64_t m){
    int same = n == m;
    int first = -1;
    for (int i = 0; i < n && i < m; i++) if (got[i] != (int)want[i]) { same = 0; first = i; break; }
    fprintf(stderr, "[%s] %s: %d tokens (ref %lld): %s", g_tag, what, n, (long long)m, same ? "IDENTICAL\n" : "DIFFERENT");
    if (!same) fprintf(stderr, " (first difference at %d: got %d, want %d)\n", first, first >= 0 ? got[first] : -1, first >= 0 ? (int)want[first] : -1);
    return !same;
}

static int run_text_oracle(const char *model, const char *refdir){
    g_tag = "text-oracle"; g_tol_rel = TEXT_REL; g_tol_abs = TEXT_ABS;
    AceTok T; AceText *m = NULL;
    if (act_tok_open(&T, model)) return 2;
    if (!(m = act_load(model))) { act_tok_close(&T); return 2; }
    fprintf(stderr, "[text-oracle] %d layers, %d wide, %d heads (%d key/value), vocabulary %d, %.0f MB of weights\n", m->nl, m->hidden, m->heads, m->kv, m->vocab, m->wbytes / 1e6);
    shards R; memset(&R, 0, sizeof R); st_init(&R, refdir);
    if (R.n == 0) { fprintf(stderr, "[text-oracle] no safetensors in %s\n", refdir); act_free(m); act_tok_close(&T); return 2; }
    int fails = 0, checks = 0;

    /* the case's settings: what the handler was asked for */
    char *meta_text = NULL; jval *root = NULL, *p = NULL;
    {
        char path[4096]; snprintf(path, sizeof path, "%s/ref.json", refdir);
        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
            meta_text = (char *)malloc(len > 0 ? len + 1 : 1);
            if (meta_text && len > 0 && len < (16 << 20) && fread(meta_text, 1, len, f) == (size_t)len) { meta_text[len] = 0; root = json_parse_checked(meta_text); p = root ? json_get(root, "params") : NULL; }
            fclose(f);
        }
    }
    const char *caption = json_str(p, "captions"), *lyrics = json_str(p, "lyrics"), *lang = json_str(p, "vocal_language");
    if (!p || !caption || !lyrics) { fprintf(stderr, "[text-oracle] ref.json needs params.captions and params.lyrics\n"); json_free(root); free(meta_text); act_free(m); act_tok_close(&T); return 2; }
    double seconds = acd_json_num(p, "audio_duration", 30);
    char bpm[32] = ""; jval *jb = json_get(p, "bpm");
    if (jb && jb->t == J_NUM && jb->num > 0) snprintf(bpm, sizeof bpm, "%d", (int)jb->num);
    else if (jb && jb->t == J_STR) snprintf(bpm, sizeof bpm, "%s", jb->str);

    /* 1. the prompts, as strings in C and then as ids: against the ids the handler's tokenizer produced */
    char *cp = act_caption_prompt(NULL, caption, bpm, json_str(p, "key_scale"), json_str(p, "time_signature"), seconds);
    char *lp = act_lyric_prompt(lyrics, lang ? lang : "en");
    int cids[ACT_CAPTION_MAX], *lids = (int *)malloc(ACT_LYRIC_MAX * sizeof(int));
    int cn = act_tokenize(&T, cp, ACT_CAPTION_MAX, cids), ln = act_tokenize(&T, lp, ACT_LYRIC_MAX, lids);
    int64_t nrc, nrl;
    float *rcids = ref_tensor(&R, "text.ids", &nrc, NULL, NULL), *rlids = ref_tensor(&R, "lyric.ids", &nrl, NULL, NULL);
    if (!rcids || !rlids) { fprintf(stderr, "[text-oracle] the case lacks text.ids or lyric.ids\n"); fails++; }
    else { fails += same_ids("caption prompt", cids, cn, rcids, nrc); fails += same_ids("lyric prompt", lids, ln, rlids, nrl); checks += 2; }

    /* 2. the lyric path: the table's rows for the handler's ids */
    if (rlids) {
        int *ri = (int *)malloc((size_t)nrl * sizeof(int));
        for (int64_t i = 0; i < nrl; i++) ri[i] = (int)rlids[i];
        float *e = act_embed(m, ri, (int)nrl), *want = ref_tensor(&R, "lyric.hidden", NULL, NULL, NULL);
        if (e && want) { fails += cmp_tol("lyric embedding rows", e, want, (size_t)nrl * m->hidden, 1e-7, 1e-7); checks++; }
        free(e); free(want); free(ri);
    }

    /* 3. the caption pass, fed the handler's own ids: the first hidden state, each layer alone, then the whole encoder */
    if (rcids) {
        int *ri = (int *)malloc((size_t)nrc * sizeof(int));
        for (int64_t i = 0; i < nrc; i++) ri[i] = (int)rcids[i];
        const int n = (int)nrc, D = m->hidden;
        float *e0 = act_embed(m, ri, n);
        float *tap_e = ref_tensor(&R, "text.tap.embed", NULL, NULL, NULL);
        if (e0 && tap_e) { fails += cmp_tol("embedding rows (caption)", e0, tap_e, (size_t)n * D, 1e-7, 1e-7); checks++; }
        float *rc = ref_tensor(&R, "text.rope_cos", NULL, NULL, NULL), *rs = ref_tensor(&R, "text.rope_sin", NULL, NULL, NULL);
        if (tap_e && rc && rs) {
            float *mc = (float *)malloc((size_t)n * m->hd * sizeof(float)), *ms = (float *)malloc((size_t)n * m->hd * sizeof(float));
            acd_rope_table(mc, ms, n, m->hd, m->theta);
            fails += cmp_tol("text rope cos", mc, rc, (size_t)n * m->hd, 1e-6, 1e-6); fails += cmp_tol("text rope sin", ms, rs, (size_t)n * m->hd, 1e-6, 1e-6); checks += 2;
            float *h = (float *)malloc((size_t)n * D * sizeof(float));
            for (int k = 0; k < m->nl; k++) {
                char in_name[48], out_name[48];
                if (k == 0) snprintf(in_name, sizeof in_name, "text.tap.embed"); else snprintf(in_name, sizeof in_name, "text.tap.layer%d", k - 1);
                snprintf(out_name, sizeof out_name, "text.tap.layer%d", k);
                float *hin = ref_tensor(&R, in_name, NULL, NULL, NULL), *hout = ref_tensor(&R, out_name, NULL, NULL, NULL);
                if (!hin || !hout) { free(hin); free(hout); break; }
                memcpy(h, hin, (size_t)n * D * sizeof(float));
                act_layer(m, &m->L[k], h, n, rc, rs);
                char lab[64]; snprintf(lab, sizeof lab, "text layer %d alone", k);
                double in_peak = 0;                                   /* the residual stream's scale going in (see TEXT_LAYER_ABS) */
                for (size_t q = 0; q < (size_t)n * D; q++) if (fabs((double)hin[q]) > in_peak) in_peak = fabs((double)hin[q]);
                g_cmp_scale = in_peak;
                fails += cmp_tol(lab, h, hout, (size_t)n * D, TEXT_LAYER_REL, TEXT_LAYER_ABS); checks++;
                g_cmp_scale = 0;
                free(hin); free(hout);
            }
            free(h); free(mc); free(ms);
        }
        free(rc); free(rs); free(tap_e); free(e0);
        double t0 = now();
        float *hid = act_encode(m, ri, n), *want = ref_tensor(&R, "text.hidden", NULL, NULL, NULL);
        if (hid && want) { char lab[64]; snprintf(lab, sizeof lab, "caption hidden (%d tokens, %.1fs)", n, now() - t0); fails += cmp(lab, hid, want, (size_t)n * D); checks++; }
        /* and what the pipeline would do: the C ids, not the handler's (the same when the ids are identical) */
        if (hid && want && (cn != n || memcmp(cids, ri, (size_t)n * sizeof(int)))) {
            float *own = act_encode(m, cids, cn);
            fprintf(stderr, "[text-oracle] the ids differ, so the hidden state of the C ids is not compared with the handler's\n");
            free(own);
        }
        free(hid); free(want); free(ri);
    }
    free(rcids); free(rlids); free(cp); free(lp); free(lids);
    json_free(root); free(meta_text); act_free(m); act_tok_close(&T); st_destroy(&R);
    if (!checks) { fprintf(stderr, "[text-oracle] nothing to compare in %s\n", refdir); return 2; }
    if (fails) { fprintf(stderr, "[text-oracle] %d of %d comparison(s) FAILED\n", fails, checks); return 1; }
    fprintf(stderr, "[text-oracle] all %d comparison(s) within their limits (the encoder: rel %.0e, abs %.0e; a layer alone: %.0e, %.0e)\n",
            checks, TEXT_REL, TEXT_ABS, TEXT_LAYER_REL, TEXT_LAYER_ABS);
    return 0;
}

/* The whole pipeline against the handler. The case's own parameters go through acestep_gen.h and the audio comes out; what
 * is compared is what the handler's tensors can say: the noise (torch's generator, reproduced), the silence latent that
 * the source latents and the timbre reference are cut from, and the audio. The pieces in between are the other oracles'. */
/* Limits from the real checkpoint (all 19 captured cases through here): the audio's norm ratio is 8.2e-6 to 1.6e-4 (the
 * worst is the 5-step case without DCW) and its largest sample error 2e-7 to 4.8e-4 (the 30 s clip, the longest chain: the text encoder's own 1e-5, then 8
 * sampler steps and the decoder, each passing on what it was given). A waveform lives in [-1, 1], so 2e-3 is -54 dBFS at the
 * very worst sample and the norm ratio, 5e-4, is the real bound on the whole; a wrong prompt, a wrong clip length, a wrong
 * seed or a wrong schedule is wrong by 0.1 to 1 (the mutants of tools/mutate_acestep_gen.py are). */
#define GEN_REL 5e-4
#define GEN_ABS 2e-3
/* With audio in (20 captured cases: cover, reference, repaint, and a few edge cases of each) the audio carries the VAE encoder's own
 * float32 error besides, and the sampler amplifies it: the worst case measured is 4.0e-4 of the norm and 6.7e-4 at one sample (a repaint
 * that adds silence before the clip, whose span is the longest chain); the cover cases are as exact as text to music (5e-6), the
 * reference ones 3.5e-5 to 1.2e-4. Held at 1.5e-3 and 4e-3: 3.8x and 6x above the worst; a wrong cover context, span, injection or
 * crossfade is wrong by 1e-2 to 1 (tools/mutate_acestep_audioin.py). */
#define GEN_AUDIO_REL 1.5e-3
#define GEN_AUDIO_ABS 4e-3

/* What each tap of ace_generate is compared with, and how closely. The stages that only copy (the silence latent's slices, the
 * noise: torch's generator reproduced) are held to rounding; a stage fed by the VAE encoder carries the encoder's own error
 * (VENC_TILE_*: torch's float32 against ours, 5e-5 to 3e-5 relative, the posterior draw included); the final latents carry the
 * DiT's chain on top (DIT_CHAIN_*). */
/* A guided DiT (base, SFT) with the source clip's latents from our VAE encoder: the encoder's float32 difference from torch's (4e-5) goes in
 * at the first step and the two branches of the guidance, seven-fold, over the steps carry it to the final latents. Measured on the real base
 * checkpoint (18 cases): the final latents are within 2e-3 of their norm and 6e-2 at one value (extract with no track name), the audio within
 * 1.2e-3 and 1.2e-3 of full scale; fed the handler's own source latents (ACESTEP_FEED_LATENTS) the same cases are within 1e-5 of their norm and
 * 3e-4 at one value, and on the tiny fixture, whose encoder is exact, 1.5e-6. So this is the encoder's noise, amplified, and a wrong
 * context, mask, guidance or schedule is wrong by 1e-1 to 1. */
#define GEN_GUIDED_REL 4e-3
#define GEN_GUIDED_ABS 1e-1
/* A guided DiT run with Heun or the SDE step (the sampler options): the loop amplifies a rounding-sized difference, and upstream does so to itself. Measured
 * on the real base checkpoint with upstream (the captures of the b_heun, b_sde and b_sde_shift3 cases, run again with the input noise scaled by 1 + 1e-6): the
 * final latents move by 8.6e-4, 1.5e-4 and 1.3e-3 of their norm (3.1e-2, 2.2e-3 and 4.6e-2 at one value) and the audio by 1.3e-4, 8e-5 and 6.5e-3 (8.8e-3
 * at one sample); this engine's differ from upstream's by less (the CPU's latents: 5.0e-4, 3.0e-4 and 1.0e-4 of the norm; the card's 1.5e-4, 1.9e-4 and
 * 1.7e-4). The SDE at a shift other than 1 is the worst: upstream noises to the level 1 - i/steps and asks the model about the shifted one, which is
 * ill-conditioned. Held at 1.5 to 3 times upstream's own response, 4e-3 and 1e-1 for the latents, 1e-2 and 2e-2 for the audio; a wrong step is wrong by
 * 1e-1 to 1 (tools/mutate_acestep_sampler.py). The turbo DiT's sampler cases (9 captured) are held to the tight limits: they agree to 1e-4 and better. */
#define GEN_AMP_REL 4e-3
#define GEN_AMP_ABS 1e-1
#define GEN_AMP_AUDIO_REL 1e-2
#define GEN_AMP_AUDIO_ABS 2e-2
/* The DiT's steps on the card (acestep_dit_vk.h) are as exact per call as the CPU's (1e-6 of the norm against upstream's), but the sampler's 40 guided steps
 * carry a different rounding further: on the real base checkpoint's 40-step case the final latents are 1.9e-4 of their norm and 3.2e-3 at one value
 * from upstream's (the CPU's 5.7e-5 and 1.1e-3), the audio 1.7e-4 (limit 5e-4). The latents' limits are 5x wider when the card ran the DiT. */
static int dit_on_card(void){
#ifdef COLI_VULKAN
    for (int i = 0; i < adv_nstates; i++) if (adv_states[i].ok && adv_states[i].steps) return 1;
#endif
    return 0;
}
typedef struct { shards *R; int fails, checks, audio_in; float *fed; int guided; int amplified; } GenTap;

/* ACESTEP_FEED_LATENTS=1: give the pipeline the handler's own source latents (in.clean_src_latents, or in.src_latents where there is
 * no span) instead of the ones our VAE encoder makes. The encoder's float32 error is then out of the comparison, and what is left is
 * the sampler's and the guidance's own: on the real base checkpoint the encoder's 4e-5 is amplified by seven-fold guidance over the
 * steps to 1e-3 of the latents, which this shows by making it 5e-6. */
static const float *gen_feed(void *ud, const char *name, size_t n){
    GenTap *g = (GenTap *)ud;
    if (strcmp(name, "source_latents") || !getenv("ACESTEP_FEED_LATENTS")) return NULL;
    int64_t m = 0;
    free(g->fed);
    g->fed = ref_tensor(g->R, "in.clean_src_latents", &m, NULL, NULL);
    if (!g->fed) g->fed = ref_tensor(g->R, "in.src_latents", &m, NULL, NULL);
    if (!g->fed || m != (int64_t)n) { free(g->fed); g->fed = NULL; return NULL; }
    fprintf(stderr, "[gen-oracle] fed the handler's own source latents (ACESTEP_FEED_LATENTS)\n");
    return g->fed;
}

static void gen_tap(void *ud, const char *name, const float *x, size_t n){
    GenTap *g = (GenTap *)ud;
    static const struct { const char *tap, *ref; int encoded; } map[] = {
        { "noise", "noise", 0 },
        { "timbre", "in.refer_audio_acoustic_hidden_states_packed", 1 },
        { "src_latents", "in.src_latents", 1 },
        { "ctx_src_latents", "in.ctx_src_latents", 4 },
        { "edit_src_hidden", "fe.src.hidden", 3 },
        { "clean_src_latents", "in.clean_src_latents", 1 },
        { "context", "context", 1 },
        { "encoder_hidden", "encoder.hidden", 3 },
        { "noncover_hidden", "noncover.hidden", 3 },
        { "noncover_context", "noncover.context", 4 },
        { "latents", "latents", 2 },
    };
    for (size_t i = 0; i < sizeof map / sizeof map[0]; i++) {
        if (strcmp(name, map[i].tap)) continue;
        int64_t m = 0;
        float *w = ref_tensor(g->R, map[i].ref, &m, NULL, NULL);
        if (!w) return;
        char label[96]; snprintf(label, sizeof label, "%s", name);
        if (m != (int64_t)n) { fprintf(stderr, "[gen-oracle] %-30s has %zu values, the handler's %lld\n", label, n, (long long)m); g->fails++; }
        else if (map[i].encoded == 2 && g->amplified) g->fails += cmp_tol(label, x, w, n, GEN_AMP_REL, GEN_AMP_ABS);
        else if (map[i].encoded == 2) g->fails += cmp_tol(label, x, w, n, g->audio_in ? (g->guided ? GEN_GUIDED_REL : DIT_CHAIN_REL) : (dit_on_card() ? 5e-4 : 1e-4), g->audio_in ? (g->guided ? GEN_GUIDED_ABS : DIT_CHAIN_ABS) : (dit_on_card() ? 1e-2 : 2e-3));
        else if (map[i].encoded == 3) g->fails += cmp_tol(label, x, w, n, g->audio_in ? 1e-4 : 1e-5, g->audio_in ? 1e-3 : 1e-4);      /* the condition encoders, from the text and the timbre */
        else if (map[i].encoded == 4) g->fails += cmp_tol(label, x, w, n, 1e-6, 1e-6);       /* the silence latent and ones: copies */
        else if (map[i].encoded == 1) g->fails += g->audio_in ? cmp_tol(label, x, w, n, VENC_TILE_REL, VENC_TILE_ABS) : cmp_tol(label, x, w, n, 1e-6, 1e-6);
        else g->fails += cmp_tol(label, x, w, n, 1e-6, 1e-6);
        g->checks++;
        free(w);
        return;
    }
}

static int run_gen_oracle(const char *model, const char *refdir, const char *wav){
    g_tag = "gen-oracle";
    char err[512] = "";
    int fails = 0, checks = 0;
    shards R; memset(&R, 0, sizeof R); st_init(&R, refdir);
    if (R.n == 0) { fprintf(stderr, "[gen-oracle] no safetensors in %s\n", refdir); return 2; }
    char path[4300]; snprintf(path, sizeof path, "%s/ref.json", refdir);
    char *meta_text = NULL; jval *root = NULL, *p = NULL;
    FILE *f = fopen(path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
        meta_text = (char *)malloc(len > 0 ? (size_t)len + 1 : 1);
        if (meta_text && len > 0 && len < (16 << 20) && fread(meta_text, 1, len, f) == (size_t)len) { meta_text[len] = 0; root = json_parse_checked(meta_text); p = root ? json_get(root, "params") : NULL; }
        fclose(f);
    }
    const char *caption = json_str(p, "captions"), *lyrics = json_str(p, "lyrics");
    if (!p || !caption || !lyrics) { fprintf(stderr, "[gen-oracle] ref.json needs params.captions and params.lyrics\n"); json_free(root); free(meta_text); st_destroy(&R); return 2; }
    AceRequest rq; memset(&rq, 0, sizeof rq);
    /* An instrumental case in an unnamed language is asked for the way a user asks: with no lyrics and no language, so that
     * the engine's defaults are what the oracle checks ("[Instrumental]" and "unknown" are the handler's own words for them). */
    const char *lang = json_str(p, "vocal_language");
    rq.caption = caption; rq.lyrics = strcmp(lyrics, "[Instrumental]") ? lyrics : NULL; rq.language = lang && strcmp(lang, "unknown") ? lang : NULL;
    rq.key = json_str(p, "key_scale"); rq.timesig = json_str(p, "time_signature");
    jval *jb = json_get(p, "bpm"); rq.bpm = jb && jb->t == J_NUM && jb->num > 0 ? (int)jb->num : 0;
    rq.duration = acd_json_num(p, "audio_duration", 30);
    rq.steps = acd_json_int(p, "inference_steps", 8);
    rq.shift = acd_json_num(p, "shift", 1.0);
    jval *dc = json_get(p, "dcw_enabled"); rq.dcw = dc && dc->t == J_BOOL ? dc->boolean : 1;
    rq.seed = (uint64_t)acd_json_num(p, "seed", 42);
    /* audio in: the clips the capture wrote beside the tensors (src.wav, ref.wav), asked for the way a user asks (a default is left unset) */
    const char *task = json_str(p, "task_type");
    rq.task = !task ? ACG_TASK_TEXT2MUSIC : !strcmp(task, "cover") ? ACG_TASK_COVER : !strcmp(task, "repaint") ? ACG_TASK_REPAINT : !strcmp(task, "extract") ? ACG_TASK_EXTRACT :
              !strcmp(task, "lego") ? ACG_TASK_LEGO : !strcmp(task, "complete") ? ACG_TASK_COMPLETE : ACG_TASK_TEXT2MUSIC;
    char src_path[4300], ref_path[4300], tracks_buf[512] = "";
    snprintf(src_path, sizeof src_path, "%s/src.wav", refdir); snprintf(ref_path, sizeof ref_path, "%s/ref.wav", refdir);
    { FILE *t = fopen(ref_path, "rb"); if (t) { fclose(t); rq.ref_path = ref_path; } }
    if (rq.task != ACG_TASK_TEXT2MUSIC || (json_get(p, "flow_edit_morph") && json_get(p, "flow_edit_morph")->t == J_BOOL && json_get(p, "flow_edit_morph")->boolean)) {
        FILE *t = fopen(src_path, "rb"); if (t) { fclose(t); rq.src_path = src_path; } }      /* a complete may have none; a flow-edit is text to music on a clip */
    /* the base DiT: guidance and the tracks (the capture keeps the caller's `_track` and `_classes` in its parameters) */
    jval *gs = json_get(p, "guidance_scale"); rq.guidance = gs && gs->t == J_NUM ? gs->num : 0;
    { const char *cn = root ? json_str(root, "case") : NULL; size_t cl = cn ? strlen(cn) : 0;
      if (cl >= 8 && !strcmp(cn + cl - 8, "_default")) rq.guidance = 0; }            /* a case named for the default leaves the scale unset (the engine's is 7, the handler's too) */
    rq.cfg_start = acd_json_num(p, "cfg_interval_start", 0); rq.cfg_end = acd_json_num(p, "cfg_interval_end", 1);
    if (rq.cfg_end >= 1) rq.cfg_end = 0;
    { jval *ad = json_get(p, "use_adg"); rq.adg = ad && ad->t == J_BOOL && ad->boolean; }
    { const char *sm = json_str(p, "sampler_mode"), *im = json_str(p, "infer_method"); rq.heun = sm && !strcmp(sm, "heun"); rq.sde = im && !strcmp(im, "sde"); }
    rq.vnorm = acd_json_num(p, "velocity_norm_threshold", 0); rq.vema = acd_json_num(p, "velocity_ema_factor", 0);
    { jval *fm = json_get(p, "flow_edit_morph"); rq.morph = fm && fm->t == J_BOOL && fm->boolean; }
    if (rq.morph) {
        rq.edit_caption = json_str(p, "flow_edit_source_caption"); rq.edit_lyrics = json_str(p, "flow_edit_source_lyrics");
        rq.edit_nmin = acd_json_num(p, "flow_edit_n_min", 0.0); rq.edit_nmax = acd_json_num(p, "flow_edit_n_max", 1.0); rq.edit_navg = acd_json_int(p, "flow_edit_n_avg", 1);
    }
    rq.track = json_str(p, "_track");
    { jval *cl = json_get(p, "_classes");
      if (cl && cl->t == J_ARR) for (int k = 0; k < cl->len; k++) if (cl->kids[k]->t == J_STR) { size_t m = strlen(tracks_buf); snprintf(tracks_buf + m, sizeof tracks_buf - m, "%s%s", m ? "," : "", cl->kids[k]->str); }
      if (tracks_buf[0]) rq.tracks = tracks_buf; }
    if (rq.task == ACG_TASK_REPAINT || rq.task == ACG_TASK_LEGO) {
        rq.repaint_start = acd_json_num(p, "repainting_start", 0);
        const double re = acd_json_num(p, "repainting_end", -1);
        rq.repaint_end = re > 0 ? re : 0;
        const char *mode = json_str(p, "repaint_mode");
        rq.repaint_mode = mode && !strcmp(mode, "conservative") ? ACG_REPAINT_CONSERVATIVE : mode && !strcmp(mode, "aggressive") ? ACG_REPAINT_AGGRESSIVE : ACG_REPAINT_BALANCED;
        const double rs = acd_json_num(p, "repaint_strength", 0.5);
        rq.repaint_strength = rs == 0.5 ? 0 : rs;
    }
    if (rq.task == ACG_TASK_COVER || (json_str(p, "audio_code_string") && *json_str(p, "audio_code_string"))) {        /* a cover, or text to music from codes: the same */
        const double cs = acd_json_num(p, "audio_cover_strength", 1.0);
        rq.cover_strength = cs == 1.0 ? 0 : cs;
        rq.cover_zero = cs == 0.0;
        rq.cover_noise = acd_json_num(p, "cover_noise_strength", 0.0);
    }
    /* the planner's codes, as the handler was given them (audio_code_string): text to music from them */
    int *code_buf = NULL;
    { const char *cs = json_str(p, "audio_code_string");
      if (cs && *cs) {
          int n = 0, cap = 64;
          code_buf = (int *)malloc((size_t)cap * sizeof(int));
          for (const char *q = cs; (q = strstr(q, "<|audio_code_")); ) {
              q += 13;
              if (n == cap) { cap *= 2; code_buf = (int *)realloc(code_buf, (size_t)cap * sizeof(int)); }
              code_buf[n++] = atoi(q);
          }
          rq.codes = code_buf; rq.ncodes = n;
      } }
    const int audio_in = rq.src_path || rq.ref_path || rq.ncodes > 0;
    fprintf(stderr, "[gen-oracle] case: %s%s, %.1f s, %d steps, shift %g, DCW %s, seed %llu\n", task ? task : "text2music", rq.ref_path ? " + reference" : "",
            rq.duration, rq.steps, rq.shift, rq.dcw ? "on" : "off", (unsigned long long)rq.seed);

    double t0 = now();
    AceEngine *e = ace_open_dit(model, g_dit, err, sizeof err);
    if (!e) { fprintf(stderr, "[gen-oracle] %s\n", err); json_free(root); free(meta_text); st_destroy(&R); return 2; }
    fprintf(stderr, "[gen-oracle] loaded in %.1fs (silence latent: %d frames)\n", now() - t0, e->silence_frames);

    int T, want_samples; acg_clip(rq.duration, &T, &want_samples);
    const int A = e->dit->acdim;
    int64_t n; int rk; int64_t sh[ST_MAX_RANK];
    float *rsil = ref_tensor(&R, "in.silence_latent", &n, sh, &rk);
    if (rsil && rk == 3 && sh[2] == A) {                                  /* the file the handler loaded, against ours, exactly */
        int fr = (int)sh[1];
        if (fr != e->silence_frames) { fprintf(stderr, "[gen-oracle] silence latent: %d frames, the handler's has %d\n", e->silence_frames, fr); fails++; }
        else fails += cmp_tol("silence latent", e->silence, rsil, (size_t)fr * A, 1e-9, 1e-9);
        checks++;
    }
    free(rsil);
    if (!audio_in) {          /* plain text to music: the source latents and the timbre are slices of the silence latent, and the clip's length is acg_clip's */
        float *slice = (float *)malloc((size_t)(T > ACG_TIMBRE_FRAMES ? T : ACG_TIMBRE_FRAMES) * A * sizeof(float));
        float *rsrc = ref_tensor(&R, "in.src_latents", &n, sh, &rk);
        if (rsrc && rk == 3 && sh[1] == T && sh[2] == A) { acg_silence_slice(e, T, slice); fails += cmp_tol("source latents", slice, rsrc, (size_t)T * A, 1e-9, 1e-9); checks++; }
        else if (rsrc) { fprintf(stderr, "[gen-oracle] the handler ran %lld frames, this engine %d\n", (long long)(rk == 3 ? sh[1] : -1), T); fails++; checks++; }
        free(rsrc);
        float *rtim = ref_tensor(&R, "in.refer_audio_acoustic_hidden_states_packed", &n, sh, &rk);
        if (rtim && rk == 3 && sh[1] == ACG_TIMBRE_FRAMES && sh[2] == A) { acg_silence_slice(e, ACG_TIMBRE_FRAMES, slice); fails += cmp_tol("timbre reference", slice, rtim, (size_t)ACG_TIMBRE_FRAMES * A, 1e-9, 1e-9); checks++; }
        else if (rtim) { fprintf(stderr, "[gen-oracle] the timbre reference is not the silence latent's first %d frames\n", ACG_TIMBRE_FRAMES); fails++; checks++; }
        free(rtim);
        float *rnoise = ref_tensor(&R, "noise", &n, sh, &rk);
        if (rnoise && n == (int64_t)T * A) {
            acg_randn((uint32_t)rq.seed, (size_t)T * A, slice);
            fails += cmp_tol("noise (torch's generator)", slice, rnoise, (size_t)T * A, 1e-6, 1e-6); checks++;
        } else if (rnoise) { fprintf(stderr, "[gen-oracle] noise has %lld values, expected %d\n", (long long)n, T * A); fails++; checks++; }
        free(rnoise); free(slice);
    }

    float *pcm = NULL; AceInfo info;
    GenTap gt = { &R, 0, 0, audio_in, NULL, !e->dit->turbo, !e->dit->turbo && (rq.heun || rq.sde) };
    AceHooks hk = { NULL, NULL, NULL, gen_tap, gen_feed };
    hk.ud = &gt;
    t0 = now();
    int rc = ace_generate(e, &rq, &hk, &pcm, &info, err, sizeof err);
    double dt = now() - t0;
    fails += gt.fails; checks += gt.checks;
    if (rc) { fprintf(stderr, "[gen-oracle] generation failed: %s\n", err); fails++; checks++; }
    else {
        fprintf(stderr, "[gen-oracle] %d frames, %d samples; encode %.1fs, text %.1fs (%d + %d tokens), condition %.1fs, denoise %.1fs, decode %.1fs, total %.1fs, peak %.4f\n",
                info.frames, info.samples, info.t_encode, info.t_text, info.caption_tokens, info.lyric_tokens, info.t_cond, info.t_denoise, info.t_decode, dt, info.peak);
        float *aud = ref_tensor(&R, "audio", &n, sh, &rk);
        const int ach = e->vae->ach;
        /* The handler returns every frame it ran (at least 128); this engine cuts a short clip to the length asked, so what
         * it must return is the shorter of the two, and that prefix of each channel is what is compared. For audio in, the
         * length asked is the source's (with a repaint's silence added). */
        const int64_t per = aud && ach > 0 && n % ach == 0 ? n / ach : -1;
        const int clip_len = rq.task != ACG_TASK_TEXT2MUSIC || rq.ncodes > 0 || rq.morph;         /* the song's length is the clip's or the codes' */
        const int ask = audio_in && clip_len ? (int)((int64_t)info.frames * ACG_HOP) : want_samples;
        const int64_t expect = per < 0 ? -1 : per < ask ? per : ask;
        if (aud && per > 0 && (info.samples == expect || (clip_len && info.samples <= per))) {
            const int64_t cmpn = info.samples < per ? info.samples : per;
            float *pl = planar(pcm, (size_t)info.samples, ach), *want = (float *)malloc((size_t)cmpn * ach * sizeof(float)), *got = (float *)malloc((size_t)cmpn * ach * sizeof(float));
            if (!want || !got) { fprintf(stderr, "out of memory\n"); exit(1); }
            for (int ch = 0; ch < ach; ch++) { memcpy(want + (size_t)ch * cmpn, aud + (size_t)ch * per, (size_t)cmpn * sizeof(float)); memcpy(got + (size_t)ch * cmpn, pl + (size_t)ch * info.samples, (size_t)cmpn * sizeof(float)); }
            const double a_rel = gt.amplified ? GEN_AMP_AUDIO_REL : audio_in ? GEN_AUDIO_REL : GEN_REL, a_abs = gt.amplified ? GEN_AMP_AUDIO_ABS : audio_in ? GEN_AUDIO_ABS : GEN_ABS;
            fails += cmp_tol("audio, end to end", got, want, (size_t)cmpn * ach, a_rel, a_abs); checks++;
            free(pl); free(want); free(got);
        } else if (aud) {
            fprintf(stderr, "[gen-oracle] the handler's audio is %lld samples a channel and %d were asked for: expected %lld back, this engine made %d\n",
                    (long long)per, ask, (long long)expect, info.samples);
            fails++; checks++;
        }
        free(aud);
        if (wav && write_wav16(wav, pcm, (size_t)info.samples, ach, e->vae->rate)) { fprintf(stderr, "[gen-oracle] cannot write %s\n", wav); fails++; }
        else if (wav) fprintf(stderr, "[gen-oracle] wrote %s (%.2f s)\n", wav, (double)info.samples / e->vae->rate);
    }
    free(pcm); free(gt.fed); free(code_buf);
    ace_close(e); json_free(root); free(meta_text); st_destroy(&R);
    if (!checks) { fprintf(stderr, "[gen-oracle] nothing to compare in %s\n", refdir); return 2; }
    if (fails) { fprintf(stderr, "[gen-oracle] %d of %d comparison(s) FAILED\n", fails, checks); return 1; }
    fprintf(stderr, "[gen-oracle] all %d comparison(s) within their limits (the audio: rel %.0e, abs %.0e)\n", checks, gt.amplified ? GEN_AMP_AUDIO_REL : audio_in ? GEN_AUDIO_REL : GEN_REL, gt.amplified ? GEN_AMP_AUDIO_ABS : audio_in ? GEN_AUDIO_ABS : GEN_ABS);
    return 0;
}

/* ------------------------------------------------------------------------------- the planner's oracle -- */

/* The language model against upstream's LLMHandler (tools/acestep_ref_lm.py): the two prompts' ids (exactly), then the codes phase run free from
 * those prompts at temperature 0, its logits over the audio codes and the end token compared at every step with the handler's (as long as the
 * tokens agree, which they do unless a near tie flips), and the codes themselves (exactly). */
#define LM_REL 1e-5
#define LM_ABS 2e-3
typedef struct { shards *R; const int *codes; int ncodes, fails, checks, diverged, worst_step; double worst_rel; } LmTap;

static int lm_step(int step, const float *cond, const float *unc, int token, void *ud){
    LmTap *t = (LmTap *)ud;
    const int nid = ALM_CODES + 1;
    if (t->diverged) return 0;
    char n1[32], n2[32]; int64_t m1 = 0, m2 = 0;
    snprintf(n1, sizeof n1, "step%d.cond", step); snprintf(n2, sizeof n2, "step%d.uncond", step);
    float *wc = ref_tensor(t->R, n1, &m1, NULL, NULL), *wu = ref_tensor(t->R, n2, &m2, NULL, NULL);
    if (!wc || !wu || m1 != nid || m2 != nid) { fprintf(stderr, "[lm-oracle] step %d: the handler recorded %lld and %lld logits, expected %d\n", step, (long long)m1, (long long)m2, nid); t->fails++; t->checks++; t->diverged = 1; free(wc); free(wu); return 0; }
    char lab[64];
    snprintf(lab, sizeof lab, "step %d conditional logits", step); int f1 = cmp_tol(lab, cond, wc, (size_t)nid, LM_REL, LM_ABS);
    snprintf(lab, sizeof lab, "step %d unconditional logits", step); int f2 = unc ? cmp_tol(lab, unc, wu, (size_t)nid, LM_REL, LM_ABS) : 0;
    t->fails += f1 + f2; t->checks += 2;
    free(wc); free(wu);
    if (step < t->ncodes ? token != t->codes[step] : token != -1) {
        fprintf(stderr, "[lm-oracle] step %d: chose %d, the handler %d\n", step, token, step < t->ncodes ? t->codes[step] : -1);
        t->fails++; t->checks++; t->diverged = 1;
    }
    return 0;
}

/* The codes half of the planner's oracle: the two prompts that carry `cot` (their ids, exactly), then the codes phase run free from them at temperature 0, its logits over the
 * audio codes and the end token compared at every step, and the codes themselves. */
static void lm_codes_check(AceLm *m, shards *Rp, const char *caption, const char *lyrics, const char *cot, double seconds, double cfg_scale, int *fails_out, int *checks_out){
    char err[512] = "";
    int fails = 0, checks = 0;
    char *pc = alm_prompt(caption, lyrics, cot, 0), *pu = alm_prompt(caption, lyrics, cot, 1);
    int nc = 0, nu = 0;
    int *ic = alm_tokenize(m, pc, &nc), *iu = alm_tokenize(m, pu, &nu);
    for (int k = 0; k < 2; k++) {
        const int *ids = k ? iu : ic; const int n = k ? nu : nc;
        int64_t rn = 0; float *want = ref_tensor(Rp, k ? "prompt.uncond" : "prompt.cond", &rn, NULL, NULL);
        int bad = !want || rn != n;
        for (int i = 0; !bad && i < n; i++) if ((int)want[i] != ids[i]) { fprintf(stderr, "[lm-oracle] %s prompt: token %d is %d, the handler's %d\n", k ? "unconditional" : "conditional", i, ids[i], (int)want[i]); bad = 1; }
        if (!want || rn != n) fprintf(stderr, "[lm-oracle] %s prompt: %d tokens, the handler's %lld\n", k ? "unconditional" : "conditional", n, (long long)rn);
        fprintf(stderr, "[lm-oracle] %-28s %d tokens %s\n", k ? "unconditional prompt" : "conditional prompt", n, bad ? "<-- FAIL" : "identical");
        fails += bad; checks++;
        free(want);
    }
    int64_t ncodes = 0; float *rc = ref_tensor(Rp, "codes", &ncodes, NULL, NULL);
    int *rcodes = (int *)malloc((size_t)(ncodes > 0 ? ncodes : 1) * sizeof(int));
    for (int64_t i = 0; i < ncodes; i++) rcodes[i] = (int)rc[i];
    free(rc);
    const int target = (int)(seconds * 5);
    int *codes = (int *)malloc((size_t)(target > 0 ? target : 1) * sizeof(int));
    LmTap tap = { Rp, rcodes, (int)ncodes, 0, 0, 0, -1, 0 };
    AlmSample sm = { 0.0, cfg_scale, 0, 0.0, 1 };
    const double t0 = now();
    int got = alm_generate_codes(m, ic, nc, iu, nu, target, &sm, codes, lm_step, &tap, err, sizeof err);
    double dt = now() - t0;
    fails += tap.fails; checks += tap.checks;
    if (got < 0) { fprintf(stderr, "[lm-oracle] generation failed: %s\n", err); fails++; checks++; }
    else {
        int same = got == (int)ncodes;
        for (int i = 0; same && i < got; i++) if (codes[i] != rcodes[i]) same = 0;
        fprintf(stderr, "[lm-oracle] %-28s %d codes, the handler's %lld: %s (%.1fs, %.2f s a token)\n", "the codes", got, (long long)ncodes, same ? "identical" : "<-- FAIL", dt, dt / (got + 1));
        fails += !same; checks++;
    }
    free(pc); free(pu); free(ic); free(iu); free(rcodes); free(codes);
    *fails_out += fails; *checks_out += checks;
}

static int run_lm_oracle(const char *model, const char *refdir, const char *lm_name){
    g_tag = "lm-oracle";
    char path[4300];
    snprintf(path, sizeof path, "%s/%s", model, lm_name);
    shards R; memset(&R, 0, sizeof R); st_init(&R, refdir);
    if (R.n == 0) { fprintf(stderr, "[lm-oracle] no safetensors in %s\n", refdir); return 2; }
    char jpath[4300]; snprintf(jpath, sizeof jpath, "%s/ref.json", refdir);
    char *meta_text = NULL; jval *root = NULL;
    FILE *f = fopen(jpath, "rb");
    if (f) {
        fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
        meta_text = (char *)malloc(len > 0 ? (size_t)len + 1 : 1);
        if (meta_text && len > 0 && len < (16 << 20) && fread(meta_text, 1, len, f) == (size_t)len) { meta_text[len] = 0; root = json_parse_checked(meta_text); }
        fclose(f);
    }
    const char *caption = json_str(root, "caption"), *lyrics = json_str(root, "lyrics");
    jval *md = root ? json_get(root, "metadata") : NULL;
    if (!root || !caption || !lyrics || !md) { fprintf(stderr, "[lm-oracle] ref.json needs caption, lyrics and metadata\n"); json_free(root); free(meta_text); st_destroy(&R); return 2; }
    const double seconds = acd_json_num(root, "seconds", 10), cfg_scale = acd_json_num(root, "cfg_scale", 2.0);
    double t0 = now();
    AceLm *m = alm_load(path);
    if (!m) { json_free(root); free(meta_text); st_destroy(&R); return 2; }
    fprintf(stderr, "[lm-oracle] %s: %d layers, %d wide, %d tokens in the vocabulary, loaded in %.1fs (%.2f GB)\n", m->dir_name, m->nl, m->D, m->vocab, now() - t0, m->wbytes / 1e9);
    int fails = 0, checks = 0;
    char *cot = alm_cot((int)acd_json_num(md, "bpm", 0), acd_json_num(md, "duration", seconds), json_str(md, "keyscale"), json_str(md, "language"), json_str(md, "timesignature"));
    lm_codes_check(m, &R, caption, lyrics, cot, seconds, cfg_scale, &fails, &checks);
    free(cot);
    alm_free(m); json_free(root); free(meta_text); st_destroy(&R);
    if (fails) { fprintf(stderr, "[lm-oracle] %d of %d comparison(s) FAILED\n", fails, checks); return 1; }
    fprintf(stderr, "[lm-oracle] all %d comparison(s) within their limits (logits: rel %.0e, abs %.0e)\n", checks, LM_REL, LM_ABS);
    return 0;
}

/* The sampler's filters and draw, on logits made up for the purpose. torch's multinomial cannot be reproduced, so the planner's oracle runs at
 * temperature 0; what a temperature, a top-k and a top-p do is held here to what upstream's _apply_top_k_filter, _apply_top_p_filter and
 * _sample_tokens say they do, on cases small enough to know the answer. */
static int lm_check(int ok, const char *what){
    if (!ok) fprintf(stderr, "[lm-selftest] %s <-- FAIL\n", what);
    return !ok;
}

static int run_lm_selftest(void){
    int fails = 0;
    uint64_t rng = 12345;
    {   /* top-k keeps every logit not below the k-th largest, ties included */
        float lg[5] = {5, 4, 3, 2, 1}; AlmSample s = { 0.0, 1.0, 2, 0.0, 1 };
        int t = alm_choose(lg, 5, &s, &rng);
        fails += lm_check(t == 0 && lg[0] == 5 && lg[1] == 4 && isinf(lg[2]) && isinf(lg[3]) && isinf(lg[4]), "top-k 2 of five keeps the two largest");
        float tie[4] = {3, 3, 3, 1}; AlmSample s2 = { 0.0, 1.0, 2, 0.0, 1 };
        alm_choose(tie, 4, &s2, &rng);
        fails += lm_check(tie[0] == 3 && tie[1] == 3 && tie[2] == 3 && isinf(tie[3]), "top-k keeps ties with the k-th largest");
        float all[3] = {1, 2, 3}; AlmSample s3 = { 0.0, 1.0, 5, 0.0, 1 };
        alm_choose(all, 3, &s3, &rng);
        fails += lm_check(!isinf(all[0]) && !isinf(all[1]) && !isinf(all[2]), "top-k larger than the row changes nothing");
    }
    {   /* top-p keeps the smallest prefix, by descending probability, whose cumulative probability before the next token has not passed p; the first is always kept */
        const double pr[5] = {0.5, 0.3, 0.1, 0.06, 0.04};
        for (int c = 0; c < 4; c++) {
            const double p = c == 0 ? 0.45 : c == 1 ? 0.55 : c == 2 ? 0.85 : 0.99;
            const int keep = c == 0 ? 1 : c == 1 ? 2 : c == 2 ? 3 : 5;
            float lg[5]; for (int i = 0; i < 5; i++) lg[4 - i] = (float)log(pr[i]);      /* ascending in memory: the sort has work to do */
            AlmSample s = { 0.0, 1.0, 0, p, 1 };
            alm_choose(lg, 5, &s, &rng);
            int kept = 0; for (int i = 0; i < 5; i++) kept += !isinf(lg[i]);
            char what[96]; snprintf(what, sizeof what, "top-p %.2f keeps %d of five (kept %d)", p, keep, kept);
            fails += lm_check(kept == keep && !isinf(lg[4]), what);
        }
        float one[3] = {10, 0, 0}; AlmSample s = { 0.0, 1.0, 0, 0.01, 1 };
        alm_choose(one, 3, &s, &rng);
        fails += lm_check(!isinf(one[0]) && isinf(one[1]) && isinf(one[2]), "top-p 0.01 still keeps the first token");
    }
    {   /* temperature 0 is an argmax, a small one nearly always; 1 draws from the softmax */
        float lg[3] = {1, 3, 2}; AlmSample s = { 0.0, 1.0, 0, 0.0, 1 };
        fails += lm_check(alm_choose(lg, 3, &s, &rng) == 1, "temperature 0 is the argmax");
        float tie[4] = {3, 5, 5, 1};
        fails += lm_check(alm_choose(tie, 4, &s, &rng) == 1, "an argmax over a tie is the first of them, as torch.argmax gives");
        int hits = 0;
        for (int i = 0; i < 200; i++) { float l2[3] = {1, 3, 2}; AlmSample sm = { 0.05, 1.0, 0, 0.0, 1 }; hits += alm_choose(l2, 3, &sm, &rng) == 1; }
        fails += lm_check(hits >= 195, "a temperature of 0.05 is nearly an argmax");
        int count[3] = {0, 0, 0};
        const int N = 30000;
        for (int i = 0; i < N; i++) { float l3[3] = {2, 1, 0}; AlmSample sm = { 1.0, 1.0, 0, 0.0, 1 }; count[alm_choose(l3, 3, &sm, &rng)]++; }
        const double z = exp(2.0) + exp(1.0) + 1.0, want[3] = {exp(2.0) / z, exp(1.0) / z, 1.0 / z};
        int ok = 1; for (int i = 0; i < 3; i++) if (fabs((double)count[i] / N - want[i]) > 0.012) ok = 0;
        fails += lm_check(ok, "temperature 1 draws from the softmax");
        int flat[3] = {0, 0, 0};
        for (int i = 0; i < N; i++) { float l4[3] = {2, 1, 0}; AlmSample sm = { 100.0, 1.0, 0, 0.0, 1 }; flat[alm_choose(l4, 3, &sm, &rng)]++; }
        ok = 1; for (int i = 0; i < 3; i++) if (fabs((double)flat[i] / N - 1.0 / 3) > 0.02) ok = 0;
        fails += lm_check(ok, "a high temperature is nearly uniform");
        int masked = 0;
        for (int i = 0; i < 2000; i++) { float l5[3] = {0, -INFINITY, 0}; AlmSample sm = { 1.0, 1.0, 0, 0.0, 1 }; masked += alm_choose(l5, 3, &sm, &rng) == 1; }
        fails += lm_check(masked == 0, "a token at minus infinity is never drawn");
    }
    {   /* the same seed is the same draws, another seed is not */
        uint64_t a = 7, b = 7, c = 8; int same = 1, differ = 0;
        for (int i = 0; i < 50; i++) {
            float l1[4] = {1, 1, 1, 1}, l2[4] = {1, 1, 1, 1}, l3[4] = {1, 1, 1, 1}; AlmSample sm = { 1.0, 1.0, 0, 0.0, 1 };
            int x = alm_choose(l1, 4, &sm, &a), y = alm_choose(l2, 4, &sm, &b), z = alm_choose(l3, 4, &sm, &c);
            if (x != y) same = 0;
            if (x != z) differ = 1;
        }
        fails += lm_check(same && differ, "a seed repeats its draws and another seed makes others");
    }
    {   /* the prompts and the metadata, as upstream's template and YAML write them */
        char *cot = alm_cot(120, 10, "C Major", "unknown", "4");
        fails += lm_check(cot && !strcmp(cot, "<think>\nbpm: 120\nduration: 10\nkeyscale: C Major\nlanguage: unknown\ntimesignature: 4\n</think>"), "the metadata as YAML");
        free(cot);
        cot = alm_cot(96, 9.6, "G minor", "no", "3/4");
        fails += lm_check(cot && !strcmp(cot, "<think>\nbpm: 96\nduration: 9\nkeyscale: G minor\nlanguage: 'no'\ntimesignature: 3\n</think>"), "a duration cut to whole seconds, Norwegian quoted as YAML does, 3/4 as 3");
        free(cot);
        cot = alm_cot(70, 8, "D minor", NULL, "4");
        fails += lm_check(cot && !strstr(cot, "language"), "no language, no line");
        free(cot);
        char *p = alm_prompt("a song", "[Instrumental]", "<think>\nbpm: 1\n</think>", 0), *u = alm_prompt("a song", "[Instrumental]", "<think>\nbpm: 1\n</think>", 1);
        fails += lm_check(p && !strcmp(p, "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n<|im_end|>\n<|im_start|>user\n"
                                          "# Caption\na song\n\n# Lyric\n[Instrumental]\n<|im_end|>\n<|im_start|>assistant\n<think>\nbpm: 1\n</think>\n\n"), "the conditional prompt");
        fails += lm_check(u && !strcmp(u, "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n<|im_end|>\n<|im_start|>user\n"
                                          "NO USER INPUT<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"), "the unconditional prompt");
        free(p); free(u);
    }
    if (fails) { fprintf(stderr, "[lm-selftest] %d check(s) FAILED\n", fails); return 1; }
    fprintf(stderr, "[lm-selftest] all checks passed\n");
    return 0;
}

/* acestep --model DIR --caption TEXT --out FILE.wav ...: one clip, no protocol. */
/* ----------------------------------------------------------------------- the planner's first phase: its oracles -- */

static char *slurp_file(const char *path){
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = (char *)malloc(len > 0 ? (size_t)len + 1 : 1);
    if (!t || len <= 0 || len > (256L << 20) || fread(t, 1, (size_t)len, f) != (size_t)len) { fclose(f); free(t); return NULL; }
    fclose(f);
    t[len] = 0;
    return t;
}

/* The YAML emitter against PyYAML (tools/acestep_ref_meta.py yaml): for every string of the corpus, yaml.dump({"caption": s}, allow_unicode=True) byte for byte. */
static int run_yaml_oracle(const char *file){
    g_tag = "yaml-oracle";
    char *text = slurp_file(file);
    jval *root = text ? json_parse_checked(text) : NULL;
    if (!root || root->t != J_ARR) { fprintf(stderr, "[yaml-oracle] %s is not a corpus (tools/acestep_ref_meta.py yaml)\n", file); json_free(root); free(text); return 2; }
    int fails = 0, quoted = 0, wrapped = 0, doubled = 0;
    for (int i = 0; i < root->len; i++) {
        jval *it = root->kids[i];
        const char *in = json_str(it, "in"), *want = json_str(it, "out");
        if (!in || !want) { fails++; continue; }
        AmeItem item = { "caption", in, 0, 0 };
        char *got = ame_yaml_dump(&item, 1);
        if (!got || strcmp(got, want)) {
            if (fails < 8) fprintf(stderr, "[yaml-oracle] <-- FAIL for %.80s\n    want %.200s\n    got  %.200s\n", in, want, got ? got : "(none)");
            fails++;
        }
        quoted += strstr(want, "caption: '") != NULL; doubled += strstr(want, "caption: \"") != NULL; wrapped += strchr(strchr(want, '\n') + 1, '\n') != NULL;
        free(got);
    }
    fprintf(stderr, "[yaml-oracle] %d strings: %d plain, %d single-quoted, %d double-quoted, %d wrapped over lines; %d differ from PyYAML%s\n", root->len,
            root->len - quoted - doubled, quoted, doubled, wrapped, fails, fails ? "  <-- FAIL" : "");
    json_free(root); free(text);
    return fails ? 1 : 0;
}

/* A tokenizer's worth of an AceLm: the token tables only need that. */
static AceLm *meta_stub(const char *lm_dir, char *err, size_t errn){
    AceLm *m = (AceLm *)calloc(1, sizeof *m);
    char path[4400];
    snprintf(path, sizeof path, "%s/tokenizer.json", lm_dir);
    FILE *f = fopen(path, "rb");
    if (!m || !f) { snprintf(err, errn, "no tokenizer in %.300s", lm_dir); free(m); if (f) fclose(f); return NULL; }
    fclose(f);
    tok_load(&m->tok, path);
    m->audio0 = tok_id_of(&m->tok, "<|audio_code_0|>");
    m->eos = tok_id_of(&m->tok, "<|im_end|>");
    snprintf(path, sizeof path, "%s/config.json", lm_dir);
    char *ct = slurp_file(path); jval *cj = ct ? json_parse_checked(ct) : NULL;
    m->vocab = cj ? acd_json_int(cj, "vocab_size", 0) : 0;
    json_free(cj); free(ct);
    if (m->vocab < 1000 || m->audio0 < 0 || m->eos < 0) { snprintf(err, errn, "the tokenizer or the config of %.300s does not fit", lm_dir); tok_free(&m->tok); free(m); return NULL; }
    return m;
}
static void meta_stub_free(AceLm *m){ if (m->meta_tab && m->meta_free) m->meta_free(m->meta_tab); tok_free(&m->tok); free(m); }

/* The token tables against the ones upstream's processor builds from the same tokenizer (trees.json): every prefix of every value and the tokens that may follow it,
 * the tokens that spell the fixed strings, and the newline, end and backtick tokens. */
static int meta_check_tree(const char *name, jval *want, const AmeTree *t, int nl){
    int fails = 0, rows = 0;
    for (int i = 0; want && i < want->len; i++) {
        jval *row = want->kids[i], *pre = row->kids[0], *nx = row->kids[1];
        int prefix[AME_MAXSEQ + 4], pl = pre->len, got[AME_MAXSEQ * 8 + 8], has;
        for (int k = 0; k < pl && k < AME_MAXSEQ + 4; k++) prefix[k] = (int)pre->kids[k]->num;
        const int gn = ame_tree_next(t, nl, prefix, pl, got, (int)(sizeof got / sizeof *got), &has);
        int ok = has && gn == nx->len;
        for (int k = 0; ok && k < gn; k++) if (got[k] != (int)nx->kids[k]->num) ok = 0;
        if (!ok) { if (fails < 4) fprintf(stderr, "[meta-trees] %s: after a prefix of %d tokens: %d tokens may follow here, upstream's %d\n", name, pl, gn, nx->len); fails++; }
        rows++;
    }
    int mine = 0;                                          /* the prefixes mine has: each distinct one once */
    for (int i = 0; i < t->n; i++)
        for (int l = 0; l <= t->len[i]; l++) {
            int seen = 0;
            for (int j = 0; j < i && !seen; j++) if (t->len[j] >= l && !memcmp(t->seq[j], t->seq[i], (size_t)l * sizeof(int))) seen = 1;
            if (!seen) mine++;
        }
    if (mine != rows) { fprintf(stderr, "[meta-trees] %s: %d prefixes, upstream's %d\n", name, mine, rows); fails++; }
    fprintf(stderr, "[meta-trees] %-14s %5d prefixes %s\n", name, rows, fails ? "<-- FAIL" : "identical");
    return fails;
}

static int run_meta_trees(const char *model, const char *file, const char *lm_name){
    g_tag = "meta-trees";
    char path[4300], err[512] = "";
    snprintf(path, sizeof path, "%s/%s", model, lm_name);
    AceLm *m = meta_stub(path, err, sizeof err);
    if (!m) { fprintf(stderr, "[meta-trees] %s\n", err); return 2; }
    char *text = slurp_file(file);
    jval *root = text ? json_parse_checked(text) : NULL;
    if (!root) { fprintf(stderr, "[meta-trees] cannot read %s\n", file); meta_stub_free(m); free(text); return 2; }
    ame_max_duration = (int)acd_json_num(root, "max_duration", 600);
    double t0 = now();
    AmeTab *t = ame_tab(m, err, sizeof err);
    if (!t) { fprintf(stderr, "[meta-trees] %s\n", err); json_free(root); free(text); meta_stub_free(m); return 1; }
    fprintf(stderr, "[meta-trees] the tables took %.2fs\n", now() - t0);
    int fails = 0;
    fails += meta_check_tree("bpm", json_get(root, "bpm"), &t->bpm, t->nl);
    fails += meta_check_tree("duration", json_get(root, "duration"), &t->dur, t->nl);
    fails += meta_check_tree("keyscale", json_get(root, "keyscale"), &t->key, t->nl);
    fails += meta_check_tree("language", json_get(root, "language"), &t->lang, t->nl);
    fails += meta_check_tree("timesignature", json_get(root, "timesignature"), &t->ts, t->nl);
    jval *fx = json_get(root, "fixed");
    for (int s = 0; s < AMS_N; s++) {
        if (!AMS_FIXED[s]) continue;
        jval *w = fx ? json_get(fx, AMS_NAMES[s]) : NULL, *steps = w ? json_get(w, "allowed_by_step") : NULL;
        int bad = !steps || steps->len != t->nfixed[s];
        for (int k = 0; !bad && k < steps->len; k++) if (steps->kids[k]->len != 1 || (int)steps->kids[k]->kids[0]->num != t->fixed[s][k]) bad = 1;
        fprintf(stderr, "[meta-trees] %-14s %d tokens %s\n", AMS_NAMES[s], t->nfixed[s], bad ? "<-- FAIL" : "identical");
        fails += bad;
    }
    const int nl = (int)acd_json_num(root, "newline", -1), eos = (int)acd_json_num(root, "eos", -1), bt = (int)acd_json_num(root, "backtick", -1);
    jval *ac = json_get(root, "audio_codes");
    int bad = nl != t->nl || eos != t->eos || bt != t->backtick || !ac || ac->len != 3 || (int)ac->kids[0]->num != m->audio0 || (int)ac->kids[1]->num != m->audio0 + ALM_CODES - 1 || (int)ac->kids[2]->num != ALM_CODES;
    fprintf(stderr, "[meta-trees] %-14s newline %d, end %d, backtick %d, audio codes %d..%d %s\n", "special tokens", t->nl, t->eos, t->backtick, m->audio0, m->audio0 + ALM_CODES - 1, bad ? "<-- FAIL" : "identical");
    fails += bad;
    json_free(root); free(text); meta_stub_free(m);
    if (fails) { fprintf(stderr, "[meta-trees] %d table(s) differ from upstream's\n", fails); return 1; }
    fprintf(stderr, "[meta-trees] every table is upstream's\n");
    return 0;
}

/* The first phase against upstream's handler (tools/acestep_ref_meta.py capture): phase 1's prompt (exactly), then phase 1 run from it at temperature 0, every step
 * compared with the handler's -- the decoder's state, how many tokens it let through (and which, with the model's logit at each, or the highest 64 when it lets
 * nearly everything through), the token chosen -- and the metadata parsed from what the model wrote. For a case that goes on to phase 2: the YAML of its prompt, the
 * prompts' ids, and the codes under them (lm_codes_check). */
#define META_REL 3e-5          /* the first phase reads a 217204-wide head through 28 layers: float noise of 1e-5 a step is what two summation orders do */
typedef struct { shards *R; jval *steps; int nsteps, fails, checks, diverged, worst_step; double worst_rel, sse, ssw; } MetaTap;

static void meta_trace(void *ud, int step, int state, int n, const int *ids, const float *raw, int tok){
    MetaTap *t = (MetaTap *)ud;
    if (t->diverged) return;
    t->checks++;
    if (step >= t->nsteps) { fprintf(stderr, "[meta-oracle] step %d: the handler stopped after %d\n", step, t->nsteps); t->fails++; t->diverged = 1; return; }
    jval *js = t->steps->kids[step];
    const char *rs = json_str(js, "state");
    const int rn = (int)acd_json_num(js, "n", -1), rtok = (int)acd_json_num(js, "tok", -2);
    if (!rs || strcmp(rs, AMS_NAMES[state])) { fprintf(stderr, "[meta-oracle] step %d: this engine is in %s, the handler in %s\n", step, AMS_NAMES[state], rs ? rs : "?"); t->fails++; t->diverged = 1; return; }
    if (rn != n) { fprintf(stderr, "[meta-oracle] step %d (%s): %d tokens let through, the handler's %d\n", step, rs, n, rn); t->fails++; t->diverged = 1; return; }
    char ni[32], nv[32]; int64_t m1 = 0, m2 = 0;
    snprintf(ni, sizeof ni, "s%d.idx", step); snprintf(nv, sizeof nv, "s%d.val", step);
    float *ri = ref_tensor(t->R, ni, &m1, NULL, NULL), *rv = ref_tensor(t->R, nv, &m2, NULL, NULL);
    if (!ri || !rv || m1 != m2 || m1 < 1) { fprintf(stderr, "[meta-oracle] step %d: no logits recorded\n", step); t->fails++; t->diverged = 1; free(ri); free(rv); return; }
    double se = 0, sw = 0, emax = 0, peak = 0;
    int bad = 0;
    for (int64_t k = 0; k < m1 && !bad; k++) {
        const int id = (int)ri[k];
        int at = -1;
        if (n <= 2048) { at = (int)k < n && ids[k] == id ? (int)k : -1; }
        else { int lo = 0, hi = n - 1; while (lo <= hi) { int mid = (lo + hi) / 2; if (ids[mid] == id) { at = mid; break; } if (ids[mid] < id) lo = mid + 1; else hi = mid - 1; } }
        if (at < 0) { fprintf(stderr, "[meta-oracle] step %d (%s): token %d is not among those this engine lets through\n", step, rs, id); bad = 1; break; }
        const double d = (double)raw[at] - rv[k];
        se += d * d; sw += (double)rv[k] * rv[k];
        if (fabs(d) > emax) emax = fabs(d);
        if (fabs((double)rv[k]) > peak) peak = fabs((double)rv[k]);
    }
    const double rel = sw > 0 ? sqrt(se / sw) : sqrt(se);
    t->sse += se; t->ssw += sw;
    if (rel > t->worst_rel) { t->worst_rel = rel; t->worst_step = step; }
    if (!bad && ((m1 >= 8 && rel > META_REL) || emax > LM_ABS * (peak > 1.0 ? peak : 1.0))) { fprintf(stderr, "[meta-oracle] step %d (%s): the logits differ: max|err| %.3e, rel %.3e <-- FAIL\n", step, rs, emax, rel); bad = 1; }
    if (!bad && tok != rtok) { fprintf(stderr, "[meta-oracle] step %d (%s): chose %d, the handler %d <-- FAIL\n", step, rs, tok, rtok); bad = 1; }
    if (bad) { t->fails++; t->diverged = 1; }
    free(ri); free(rv);
}

static int meta_field_check(jval *md, const AlmMeta *got, int *checks){
    int fails = 0;
    for (int f = 0; f < AMF_N; f++) {
        jval *w = md ? json_get(md, AMF_KEYS[f]) : NULL;
        char want[16384] = "";
        const int have = w && w->t != J_NULL;
        if (have) { if (w->t == J_NUM) snprintf(want, sizeof want, "%lld", (long long)w->num); else snprintf(want, sizeof want, "%s", w->str ? w->str : ""); }
        const int same = have == got->have[f] && (!have || !strcmp(want, got->s[f]));
        if (!same) {
            fprintf(stderr, "[meta-oracle] %s: this engine has %s%.120s, the handler %s%.120s <-- FAIL\n", AMF_KEYS[f], got->have[f] ? "" : "(none) ", got->have[f] ? got->s[f] : "", have ? "" : "(none) ", want);
            if (have && got->have[f]) {
                size_t k = 0; while (got->s[f][k] && want[k] && got->s[f][k] == want[k]) k++;
                fprintf(stderr, "    first difference at byte %zu (%zu and %zu bytes long): this engine \"%.40s\", the handler \"%.40s\"\n", k, strlen(got->s[f]), strlen(want), got->s[f] + k, want + k);
            }
        }
        fails += !same; (*checks)++;
    }
    return fails;
}

static int run_meta_oracle(const char *model, const char *refdir, const char *lm_name){
    g_tag = "meta-oracle";
    char path[4300], err[512] = "";
    snprintf(path, sizeof path, "%s/%s", model, lm_name);
    shards R; memset(&R, 0, sizeof R); st_init(&R, refdir);
    if (R.n == 0) { fprintf(stderr, "[meta-oracle] no safetensors in %s\n", refdir); return 2; }
    char jpath[4300]; snprintf(jpath, sizeof jpath, "%s/ref.json", refdir);
    char *jt = slurp_file(jpath); jval *root = jt ? json_parse_checked(jt) : NULL;
    const char *caption = json_str(root, "caption"), *lyrics = json_str(root, "lyrics");
    jval *um = root ? json_get(root, "user_metadata") : NULL, *fl = root ? json_get(root, "flags") : NULL, *steps = root ? json_get(root, "steps") : NULL, *md = root ? json_get(root, "metadata") : NULL;
    if (!root || !caption || !lyrics || !steps || steps->t != J_ARR || !md) { fprintf(stderr, "[meta-oracle] ref.json is not a case of tools/acestep_ref_meta.py\n"); json_free(root); free(jt); st_destroy(&R); return 2; }
    ame_max_duration = (int)acd_json_num(root, "max_duration", 600);
    double t0 = now();
    AceLm *m = alm_load(path);
    if (!m) { json_free(root); free(jt); st_destroy(&R); return 2; }
    fprintf(stderr, "[meta-oracle] %s: loaded in %.1fs\n", m->dir_name, now() - t0);
    int fails = 0, checks = 0;
    AlmAsk ask; memset(&ask, 0, sizeof ask);
    char ukey[64] = "", uts[16] = "";
    if (um) {
        ask.bpm = (int)acd_json_num(um, "bpm", 0); ask.duration = (int)acd_json_num(um, "duration", 0);
        if (json_str(um, "keyscale")) { snprintf(ukey, sizeof ukey, "%s", json_str(um, "keyscale")); ask.key = ukey; }
        if (json_str(um, "timesignature")) { snprintf(uts, sizeof uts, "%s", json_str(um, "timesignature")); ask.timesig = uts; }
    }
    ask.caption = fl && json_get(fl, "use_cot_caption") ? json_get(fl, "use_cot_caption")->boolean : 1;
    ask.language = fl && json_get(fl, "use_cot_language") ? json_get(fl, "use_cot_language")->boolean : 1;
    char *p1 = alm_meta_prompt(caption, lyrics);
    int np = 0; int *ip = alm_tokenize(m, p1, &np);
    {
        int64_t rn = 0; float *want = ref_tensor(&R, "prompt1", &rn, NULL, NULL);
        int bad = !want || rn != np;
        for (int i = 0; !bad && i < np; i++) if ((int)want[i] != ip[i]) { fprintf(stderr, "[meta-oracle] phase 1 prompt: token %d is %d, the handler's %d\n", i, ip[i], (int)want[i]); bad = 1; }
        if (want && rn != np) fprintf(stderr, "[meta-oracle] phase 1 prompt: %d tokens, the handler's %lld\n", np, (long long)rn);
        fprintf(stderr, "[meta-oracle] %-28s %d tokens %s\n", "phase 1 prompt", np, bad ? "<-- FAIL" : "identical");
        fails += bad; checks++;
        free(want);
    }
    MetaTap tap = { &R, steps, steps->len, 0, 0, 0, -1, 0, 0, 0 };
    AlmSample sm = { 0.0, 1.0, 0, 0.0, 1 };
    AlmMeta meta; memset(&meta, 0, sizeof meta);
    t0 = now();
    int got = alm_generate_meta(m, ip, np, &ask, &sm, &meta, meta_trace, &tap, NULL, NULL, NULL, err, sizeof err);
    const double dt = now() - t0;
    fails += tap.fails; checks += tap.checks;
    if (got < 0) { fprintf(stderr, "[meta-oracle] phase 1 failed: %s\n", err); fails++; checks++; }
    else {
        const int same = got == steps->len;
        fprintf(stderr, "[meta-oracle] %-28s %d tokens in %.1fs (%.2f s a token), the handler's %d: %s; logits worst rel %.2e at step %d, overall %.2e\n", "phase 1", got, dt, dt / (got > 0 ? got : 1),
                steps->len, same ? "same length" : "<-- FAIL", tap.worst_rel, tap.worst_step, tap.ssw > 0 ? sqrt(tap.sse / tap.ssw) : 0.0);
        fails += !same; checks++;
        fails += meta_field_check(md, &meta, &checks);
        char *cot = ame_cot(&meta);
        const char *want_cot = json_str(root, "cot");
        if (json_get(root, "phase2") && json_get(root, "phase2")->boolean) {
            const int ok = cot && want_cot && !strcmp(cot, want_cot);
            fprintf(stderr, "[meta-oracle] %-28s %s\n", "phase 2's YAML", ok ? "identical" : "<-- FAIL");
            if (!ok) fprintf(stderr, "    want %.400s\n    got  %.400s\n", want_cot ? want_cot : "(none)", cot ? cot : "(none)");
            fails += !ok; checks++;
            const double seconds = ask.duration > 0 ? (double)ask.duration : (meta.have[AMF_DURATION] ? atof(meta.s[AMF_DURATION]) : 0.0);
            if (cot && seconds > 0) lm_codes_check(m, &R, caption, lyrics, cot, seconds, acd_json_num(root, "cfg_scale", 2.0), &fails, &checks);
        }
        free(cot);
    }
    alm_meta_free(&meta);
    free(p1); free(ip);
    alm_free(m); json_free(root); free(jt); st_destroy(&R);
    if (fails) { fprintf(stderr, "[meta-oracle] %d of %d comparison(s) FAILED\n", fails, checks); return 1; }
    fprintf(stderr, "[meta-oracle] all %d comparison(s) within their limits (logits: rel %.0e, abs %.0e)\n", checks, META_REL, LM_ABS);
    return 0;
}

/* The parse of the model's text and the YAML of phase 2's prompt against upstream's own parse_lm_output and _format_metadata_as_cot (tools/acestep_ref_meta.py parse): for every
 * text of the corpus (what a model might write, crafted and random), the metadata dict it makes and the YAML it writes for that dict. */
static int meta_fields_equal(jval *md, const AlmMeta *got){
    for (int f = 0; f < AMF_N; f++) {
        jval *w = md ? json_get(md, AMF_KEYS[f]) : NULL;
        const int have = w && w->t != J_NULL;
        if (have != got->have[f]) return 0;
        if (!have) continue;
        char want[8192] = "";
        if (w->t == J_NUM) snprintf(want, sizeof want, "%lld", (long long)w->num); else snprintf(want, sizeof want, "%s", w->str ? w->str : "");
        if (strcmp(want, got->s[f])) return 0;
    }
    return 1;
}

static int run_parse_oracle(const char *file){
    g_tag = "parse-oracle";
    char *text = slurp_file(file);
    jval *root = text ? json_parse_checked(text) : NULL;
    if (!root || root->t != J_ARR) { fprintf(stderr, "[parse-oracle] %s is not a corpus (tools/acestep_ref_meta.py parse)\n", file); json_free(root); free(text); return 2; }
    int bad_parse = 0, bad_cot = 0, withmeta = 0;
    for (int i = 0; i < root->len; i++) {
        jval *it = root->kids[i];
        const char *in = json_str(it, "text"), *want_cot = json_str(it, "cot");
        jval *md = json_get(it, "metadata");
        AlmMeta got; memset(&got, 0, sizeof got);
        ame_parse(in ? in : "", &got);
        if (md && md->len) withmeta++;
        if (!meta_fields_equal(md, &got)) {
            if (bad_parse < 5) { fprintf(stderr, "[parse-oracle] <-- FAIL: the metadata of %.100s differs\n", in ? in : ""); for (int f = 0; f < AMF_N; f++) if (got.have[f]) fprintf(stderr, "    %s: %.100s\n", AMF_KEYS[f], got.s[f]); }
            bad_parse++;
        }
        if (want_cot) {
            char *cot = ame_cot(&got);
            if (!cot || strcmp(cot, want_cot)) {
                if (bad_cot < 5) fprintf(stderr, "[parse-oracle] <-- FAIL: the YAML for %.100s\n    want %.240s\n    got  %.240s\n", in ? in : "", want_cot, cot ? cot : "(none)");
                bad_cot++;
            }
            free(cot);
        }
        alm_meta_free(&got);
    }
    fprintf(stderr, "[parse-oracle] %d texts (%d with metadata): %d parse differently, %d write other YAML%s\n", root->len, withmeta, bad_parse, bad_cot, bad_parse || bad_cot ? "  <-- FAIL" : "");
    json_free(root); free(text);
    return bad_parse || bad_cot ? 1 : 0;
}

static int run_make(const char *model, AceRequest *rq, const char *out){
    char err[512] = "";
    if (ace_check(rq, err, sizeof err)) { fprintf(stderr, "acestep: %s\n", err); return 2; }
    /* the turbo DiT's own limit, named before anything is loaded (a request that picks a guided DiT is checked once it is known) */
    if (!strcmp(g_dit, "acestep-v15-turbo") && rq->task < ACG_TASK_EXTRACT && !(rq->dit && *rq->dit) && rq->steps > ACG_MAX_STEPS_TURBO) {
        fprintf(stderr, "acestep: steps must be 1 to %d for the turbo DiT (a base or SFT one, --dit base, takes up to %d)\n", ACG_MAX_STEPS_TURBO, ACG_MAX_STEPS);
        return 2;
    }
    AceEngine *e = ace_open_dit(model, g_dit, err, sizeof err);
    if (!e) { fprintf(stderr, "acestep: %s\n", err); return 2; }
    float *pcm = NULL; AceInfo info;
    fprintf(stderr, "[acestep] %s%s, %d steps, shift %g, DCW %s, seed %llu\n",
            (const char *const[]){ "text to music", "cover", "repaint", "extract", "lego", "complete" }[rq->task], rq->ref_path ? " with a reference" : "",
            rq->steps, rq->shift, rq->dcw ? "on" : "off", (unsigned long long)rq->seed);
    double t0 = now();
    int rc = ace_generate(e, rq, NULL, &pcm, &info, err, sizeof err);
    if (rc) { fprintf(stderr, "acestep: %s\n", err); ace_close(e); return 1; }
    if (write_wav16(out, pcm, (size_t)info.samples, e->vae->ach, e->vae->rate)) { fprintf(stderr, "acestep: cannot write %s\n", out); free(pcm); ace_close(e); return 1; }
    if (info.meta_run) fprintf(stderr, "[acestep] the planner chose: tempo %d, key %s, time signature %s, language %s, %g s; caption: %s\n", info.meta_bpm, info.meta_key, info.meta_timesig,
                               info.meta_language[0] ? info.meta_language : "(not written)", info.meta_duration, info.meta_caption[0] ? info.meta_caption : "(not rewritten)");
    if (info.plan_codes) fprintf(stderr, "[acestep] the planner (%s) wrote %d codes in %.1fs\n", info.lm, info.plan_codes, info.t_plan);
    fprintf(stderr, "[acestep] wrote %s: %.2f s of %d Hz stereo in %.1fs (text %.1f, condition %.1f, denoise %.1f, decode %.1f)%s\n", out,
            (double)info.samples / e->vae->rate, e->vae->rate, now() - t0, info.t_text, info.t_cond, info.t_denoise, info.t_decode,
            info.caption_cut || info.lyric_cut ? "; the caption or the lyrics were cut at their token limit" : "");
    free(pcm); ace_close(e);
    return 0;
}

int main(int argc, char **argv){
    const char *model = NULL, *vae_ref = NULL, *dit_ref = NULL, *tok_ref = NULL, *text_ref = NULL, *gen_ref = NULL, *venc_ref = NULL, *cover_ref = NULL, *wav = NULL, *dump = NULL, *out = NULL;
    int chunk = 256, overlap = 64, serve = 0, make = 0, steps_given = 0, dcw_given = 0, duration_given = 0;
    const char *lm_ref = NULL, *lm_name = "acestep-5Hz-lm-1.7B";
    int lm_selftest = 0;
    const char *meta_ref = NULL, *meta_trees = NULL, *yaml_ref = NULL, *parse_ref = NULL;
    AceRequest rq; memset(&rq, 0, sizeof rq);
#ifdef COLI_VULKAN
    adv_attach();                      /* the DiT's steps go to the card when COLI_VULKAN says so and there is one */
    avv_attach();                      /* and so do the VAE's windows */
    aev_attach();                      /* and the encoders' layers */
#endif
    rq.duration = ACS_DEFAULT_DURATION; rq.steps = ACS_DEFAULT_STEPS; rq.shift = ACS_DEFAULT_SHIFT; rq.dcw = 1;
    rq.lm_temperature = ACS_DEFAULT_LM_TEMPERATURE; rq.lm_cfg_scale = ACS_DEFAULT_LM_CFG; rq.lm_top_p = ACS_DEFAULT_LM_TOP_P;
    rq.seed = (uint64_t)((uint32_t)time(NULL) ^ ((uint32_t)clock() << 7));
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--model") && v) { model = v; i++; }
        else if (!strcmp(a, "--gen-ref") && v) { gen_ref = v; i++; }
        else if (!strcmp(a, "--lm-ref") && v) { lm_ref = v; i++; }
        else if (!strcmp(a, "--lm-selftest") && v) { lm_selftest = 1; i++; }
        else if (!strcmp(a, "--meta-ref") && v) { meta_ref = v; i++; }
        else if (!strcmp(a, "--meta-trees") && v) { meta_trees = v; i++; }
        else if (!strcmp(a, "--yaml-ref") && v) { yaml_ref = v; i++; }
        else if (!strcmp(a, "--parse-ref") && v) { parse_ref = v; i++; }
        else if (!strcmp(a, "--lm") && v) {
            static char lm_dir[80];
            snprintf(lm_dir, sizeof lm_dir, "%s%s", strncmp(v, "acestep-", 8) ? "acestep-5Hz-lm-" : "", v);
            lm_name = lm_dir; rq.lm = v; i++;
        }
        else if (!strcmp(a, "--plan")) { rq.plan = 1; make = 1; }
        else if (!strcmp(a, "--lm-temperature") && v) { rq.lm_temperature = atof(v); i++; }
        else if (!strcmp(a, "--lm-cfg") && v) { rq.lm_cfg_scale = atof(v); i++; }
        else if (!strcmp(a, "--lm-top-p") && v) { rq.lm_top_p = atof(v); i++; }
        else if (!strcmp(a, "--lm-top-k") && v) { rq.lm_top_k = atoi(v); i++; }
        else if (!strcmp(a, "--serve")) serve = 1;
        else if (!strcmp(a, "--out") && v) { out = v; i++; }
        else if (!strcmp(a, "--caption") && v) { rq.caption = v; make = 1; i++; }
        else if (!strcmp(a, "--lyrics") && v) { rq.lyrics = v; i++; }
        else if (!strcmp(a, "--language") && v) { rq.language = v; i++; }
        else if (!strcmp(a, "--key") && v) { rq.key = v; i++; }
        else if (!strcmp(a, "--timesig") && v) { rq.timesig = v; i++; }
        else if (!strcmp(a, "--bpm") && v) { rq.bpm = atoi(v); i++; }
        else if (!strcmp(a, "--duration") && v) { rq.duration = atof(v); duration_given = 1; i++; }
        else if (!strcmp(a, "--no-lm-caption")) rq.lm_caption_off = 1;
        else if (!strcmp(a, "--no-lm-language")) rq.lm_language_off = 1;
        else if (!strcmp(a, "--steps") && v) { rq.steps = atoi(v); steps_given = 1; i++; }
        else if (!strcmp(a, "--shift") && v) { rq.shift = atof(v); i++; }
        else if (!strcmp(a, "--seed") && v) { rq.seed = (uint64_t)strtoull(v, NULL, 10); i++; }
        else if (!strcmp(a, "--no-dcw")) { rq.dcw = 0; dcw_given = 1; }
        else if (!strcmp(a, "--dcw")) { rq.dcw = 1; dcw_given = 1; }
        else if (!strcmp(a, "--ref") && v) { rq.ref_path = v; make = 1; i++; }
        else if (!strcmp(a, "--cover") && v) { rq.task = ACG_TASK_COVER; rq.src_path = v; make = 1; i++; }
        else if (!strcmp(a, "--repaint") && v) { rq.task = ACG_TASK_REPAINT; rq.src_path = v; make = 1; i++; }
        else if (!strcmp(a, "--extract") && v) { rq.task = ACG_TASK_EXTRACT; rq.src_path = v; make = 1; i++; }
        else if (!strcmp(a, "--lego") && v) { rq.task = ACG_TASK_LEGO; rq.src_path = v; make = 1; i++; }
        else if (!strcmp(a, "--complete")) { rq.task = ACG_TASK_COMPLETE; make = 1; if (v && v[0] != '-') { rq.src_path = v; i++; } }
        else if (!strcmp(a, "--edit") && v) { rq.morph = 1; rq.src_path = v; make = 1; i++; }
        else if (!strcmp(a, "--edit-cover")) { rq.task = ACG_TASK_COVER; }
        else if (!strcmp(a, "--edit-caption") && v) { rq.edit_caption = v; i++; }
        else if (!strcmp(a, "--edit-lyrics") && v) { rq.edit_lyrics = v; i++; }
        else if (!strcmp(a, "--n-min") && v) { rq.edit_nmin = atof(v); i++; }
        else if (!strcmp(a, "--n-max") && v) { rq.edit_nmax = atof(v); i++; }
        else if (!strcmp(a, "--n-avg") && v) { rq.edit_navg = atoi(v); i++; }
        else if (!strcmp(a, "--track") && v) { rq.track = v; i++; }
        else if (!strcmp(a, "--tracks") && v) { rq.tracks = v; i++; }
        else if (!strcmp(a, "--dit") && v) {
            static char dit_dir[64];
            if (ace_dit_dir(v, dit_dir, sizeof dit_dir)) { fprintf(stderr, "acestep: --dit is turbo, base or sft\n"); return 2; }
            g_dit = dit_dir; i++;
        }
        else if (!strcmp(a, "--guidance") && v) { rq.guidance = atof(v); i++; }
        else if (!strcmp(a, "--cfg-start") && v) { rq.cfg_start = atof(v); i++; }
        else if (!strcmp(a, "--cfg-end") && v) { rq.cfg_end = atof(v); i++; }
        else if (!strcmp(a, "--adg")) rq.adg = 1;
        else if (!strcmp(a, "--sampler") && v) { if (!strcmp(v, "heun")) rq.heun = 1; else if (!strcmp(v, "euler")) rq.heun = 0; else { fprintf(stderr, "acestep: --sampler is euler or heun\n"); return 2; } i++; }
        else if (!strcmp(a, "--sde")) rq.sde = 1;
        else if (!strcmp(a, "--vnorm") && v) { rq.vnorm = atof(v); i++; }
        else if (!strcmp(a, "--vema") && v) { rq.vema = atof(v); i++; }
        else if (!strcmp(a, "--from") && v) { rq.repaint_start = atof(v); i++; }
        else if (!strcmp(a, "--to") && v) { rq.repaint_end = atof(v); i++; }
        else if (!strcmp(a, "--repaint-mode") && v) {
            if (!strcmp(v, "balanced")) rq.repaint_mode = ACG_REPAINT_BALANCED;
            else if (!strcmp(v, "conservative")) rq.repaint_mode = ACG_REPAINT_CONSERVATIVE;
            else if (!strcmp(v, "aggressive")) rq.repaint_mode = ACG_REPAINT_AGGRESSIVE;
            else { fprintf(stderr, "acestep: --repaint-mode is balanced, conservative or aggressive\n"); return 2; }
            i++;
        }
        else if (!strcmp(a, "--repaint-strength") && v) { rq.repaint_strength = atof(v); i++; }
        else if (!strcmp(a, "--cover-strength") && v) { rq.cover_strength = atof(v); rq.cover_zero = rq.cover_strength == 0.0; i++; }
        else if (!strcmp(a, "--cover-noise") && v) { rq.cover_noise = atof(v); i++; }
        else if (!strcmp(a, "--vae-ref") && v) { vae_ref = v; i++; }
        else if (!strcmp(a, "--vae-enc-ref") && v) { venc_ref = v; i++; }
        else if (!strcmp(a, "--cover-ref") && v) { cover_ref = v; i++; }
        else if (!strcmp(a, "--dit-ref") && v) { dit_ref = v; i++; }
        else if (!strcmp(a, "--tok-ref") && v) { tok_ref = v; i++; }
        else if (!strcmp(a, "--text-ref") && v) { text_ref = v; i++; }
        else if (!strcmp(a, "--chunk") && v) { chunk = atoi(v); i++; }
        else if (!strcmp(a, "--overlap") && v) { overlap = atoi(v); i++; }
        else if (!strcmp(a, "--wav") && v) { wav = v; i++; }
        else if (!strcmp(a, "--dump") && v) { dump = v; i++; }
        else if (!strcmp(a, "--threads") && v) {
#ifdef _OPENMP
            omp_set_num_threads(atoi(v));
#endif
            i++;
        }
        else { usage(); return 2; }
    }
    if (strcmp(g_dit, "acestep-v15-turbo") || rq.task >= ACG_TASK_EXTRACT) {     /* a guided DiT: its own defaults, as upstream's (32 steps, no DCW) */
        if (!steps_given) rq.steps = 32;
        if (!dcw_given) rq.dcw = 0;
    }
    if (rq.plan && !duration_given) rq.duration = 0;                      /* the planner chooses the length when it is not asked for one */
    if (serve + make + (gen_ref != NULL) > 1) { fprintf(stderr, "acestep: --serve, --caption and --gen-ref are separate modes\n"); return 2; }
    if (model && make && !rq.caption) rq.caption = "";                  /* a cover or a repaint needs no description: its style is the clip's */
    if (model && make && !out) { fprintf(stderr, "acestep: --caption, --cover, --repaint and --ref need --out FILE.wav\n"); return 2; }
    if (model && serve) {
        char err[512] = "";
        AceEngine *e = ace_open_dit(model, g_dit, err, sizeof err);
        if (!e) { fprintf(stderr, "acestep: %s\n", err); return 2; }
        int r = acs_serve(e);
        ace_close(e);
        return r;
    }
    if (model && make) return run_make(model, &rq, out);
    if (model && gen_ref) return run_gen_oracle(model, gen_ref, wav);
    if (lm_selftest) return run_lm_selftest();
    if (yaml_ref) return run_yaml_oracle(yaml_ref);
    if (parse_ref) return run_parse_oracle(parse_ref);
    if (model && lm_ref) return run_lm_oracle(model, lm_ref, lm_name);
    if (model && meta_trees) return run_meta_trees(model, meta_trees, lm_name);
    if (model && meta_ref) return run_meta_oracle(model, meta_ref, lm_name);
    if (!model || (!vae_ref && !venc_ref && !cover_ref && !dit_ref && !tok_ref && !text_ref)) { usage(); return 2; }
    int rc = 0;
    if (tok_ref) rc = run_tok_oracle(model, tok_ref);
    if (text_ref) { int r0 = run_text_oracle(model, text_ref); if (r0 > rc) rc = r0; }
    if (dit_ref) { int r1 = run_dit_oracle(model, dit_ref, dump); if (r1 > rc) rc = r1; }
    if (cover_ref) { int r4 = run_cover_oracle(model, cover_ref); if (r4 > rc) rc = r4; }
    if (venc_ref) { int r3 = run_vae_enc_oracle(model, venc_ref); if (r3 > rc) rc = r3; }
    if (vae_ref) { g_tag = "vae-oracle"; g_tol_rel = TOL_REL; g_tol_abs = TOL_ABS; int r2 = run_vae_oracle(model, vae_ref, chunk, overlap, wav, dump); if (r2 > rc) rc = r2; }
    return rc;
}
