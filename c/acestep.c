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
#include "acestep_gen.h"          /* the three networks joined: a request in, audio out */
#include "acestep_serve.h"        /* and its line protocol */

static void usage(void){
    fprintf(stderr,
        "usage: acestep --model DIR --dit-ref REFDIR [--dump PREFIX] [--threads N]\n"
        "       acestep --model DIR --vae-ref REFDIR [--chunk N] [--overlap N] [--wav OUT.wav] [--dump PREFIX] [--threads N]\n"
        "       acestep --model DIR --vae-enc-ref REFDIR | --cover-ref REFDIR\n"
        "       acestep --model DIR --tok-ref FILE.json | --text-ref REFDIR\n"
        "       acestep --model DIR --gen-ref REFDIR [--wav OUT.wav]\n"
        "       acestep --model DIR --caption TEXT --out FILE.wav [--lyrics TEXT] [--language CODE] [--duration S] [--steps N]\n"
        "               [--shift X] [--seed N] [--bpm N] [--key K] [--timesig T] [--no-dcw]\n"
        "               [--ref CLIP] [--cover CLIP [--cover-strength X] [--cover-noise X]]\n"
        "               [--repaint CLIP [--from S] [--to S] [--repaint-mode M] [--repaint-strength X]]\n"
        "       acestep --model DIR --serve\n"
        "  DIR     the ACE-Step checkpoint (holds acestep-v15-turbo/ and vae/)\n"
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
typedef struct { shards *R; int fails, checks, audio_in; } GenTap;

static void gen_tap(void *ud, const char *name, const float *x, size_t n){
    GenTap *g = (GenTap *)ud;
    static const struct { const char *tap, *ref; int encoded; } map[] = {
        { "noise", "noise", 0 },
        { "timbre", "in.refer_audio_acoustic_hidden_states_packed", 1 },
        { "src_latents", "in.src_latents", 1 },
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
        else if (map[i].encoded == 2) g->fails += cmp_tol(label, x, w, n, g->audio_in ? DIT_CHAIN_REL : 1e-4, g->audio_in ? DIT_CHAIN_ABS : 2e-3);
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
    rq.task = task && !strcmp(task, "cover") ? ACG_TASK_COVER : task && !strcmp(task, "repaint") ? ACG_TASK_REPAINT : ACG_TASK_TEXT2MUSIC;
    char src_path[4300], ref_path[4300];
    snprintf(src_path, sizeof src_path, "%s/src.wav", refdir); snprintf(ref_path, sizeof ref_path, "%s/ref.wav", refdir);
    { FILE *t = fopen(ref_path, "rb"); if (t) { fclose(t); rq.ref_path = ref_path; } }
    if (rq.task != ACG_TASK_TEXT2MUSIC) rq.src_path = src_path;
    if (rq.task == ACG_TASK_REPAINT) {
        rq.repaint_start = acd_json_num(p, "repainting_start", 0);
        const double re = acd_json_num(p, "repainting_end", -1);
        rq.repaint_end = re > 0 ? re : 0;
        const char *mode = json_str(p, "repaint_mode");
        rq.repaint_mode = mode && !strcmp(mode, "conservative") ? ACG_REPAINT_CONSERVATIVE : mode && !strcmp(mode, "aggressive") ? ACG_REPAINT_AGGRESSIVE : ACG_REPAINT_BALANCED;
        const double rs = acd_json_num(p, "repaint_strength", 0.5);
        rq.repaint_strength = rs == 0.5 ? 0 : rs;
    }
    if (rq.task == ACG_TASK_COVER) {
        const double cs = acd_json_num(p, "audio_cover_strength", 1.0);
        rq.cover_strength = cs == 1.0 ? 0 : cs;
        rq.cover_noise = acd_json_num(p, "cover_noise_strength", 0.0);
    }
    const int audio_in = rq.task != ACG_TASK_TEXT2MUSIC || rq.ref_path;
    fprintf(stderr, "[gen-oracle] case: %s%s, %.1f s, %d steps, shift %g, DCW %s, seed %llu\n", task ? task : "text2music", rq.ref_path ? " + reference" : "",
            rq.duration, rq.steps, rq.shift, rq.dcw ? "on" : "off", (unsigned long long)rq.seed);

    double t0 = now();
    AceEngine *e = ace_open(model, err, sizeof err);
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
    GenTap gt = { &R, 0, 0, audio_in };
    AceHooks hk = { NULL, NULL, NULL, gen_tap };
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
        const int ask = audio_in && rq.task != ACG_TASK_TEXT2MUSIC ? (int)((int64_t)info.frames * ACG_HOP) : want_samples;
        const int64_t expect = per < 0 ? -1 : per < ask ? per : ask;
        if (aud && per > 0 && (info.samples == expect || (rq.task != ACG_TASK_TEXT2MUSIC && info.samples <= per))) {
            const int64_t cmpn = info.samples < per ? info.samples : per;
            float *pl = planar(pcm, (size_t)info.samples, ach), *want = (float *)malloc((size_t)cmpn * ach * sizeof(float)), *got = (float *)malloc((size_t)cmpn * ach * sizeof(float));
            if (!want || !got) { fprintf(stderr, "out of memory\n"); exit(1); }
            for (int ch = 0; ch < ach; ch++) { memcpy(want + (size_t)ch * cmpn, aud + (size_t)ch * per, (size_t)cmpn * sizeof(float)); memcpy(got + (size_t)ch * cmpn, pl + (size_t)ch * info.samples, (size_t)cmpn * sizeof(float)); }
            fails += cmp_tol("audio, end to end", got, want, (size_t)cmpn * ach, audio_in ? GEN_AUDIO_REL : GEN_REL, audio_in ? GEN_AUDIO_ABS : GEN_ABS); checks++;
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
    free(pcm);
    ace_close(e); json_free(root); free(meta_text); st_destroy(&R);
    if (!checks) { fprintf(stderr, "[gen-oracle] nothing to compare in %s\n", refdir); return 2; }
    if (fails) { fprintf(stderr, "[gen-oracle] %d of %d comparison(s) FAILED\n", fails, checks); return 1; }
    fprintf(stderr, "[gen-oracle] all %d comparison(s) within their limits (the audio: rel %.0e, abs %.0e)\n", checks, audio_in ? GEN_AUDIO_REL : GEN_REL, audio_in ? GEN_AUDIO_ABS : GEN_ABS);
    return 0;
}

/* acestep --model DIR --caption TEXT --out FILE.wav ...: one clip, no protocol. */
static int run_make(const char *model, AceRequest *rq, const char *out){
    char err[512] = "";
    if (ace_check(rq, err, sizeof err)) { fprintf(stderr, "acestep: %s\n", err); return 2; }
    AceEngine *e = ace_open(model, err, sizeof err);
    if (!e) { fprintf(stderr, "acestep: %s\n", err); return 2; }
    float *pcm = NULL; AceInfo info;
    fprintf(stderr, "[acestep] %s%s, %d steps, shift %g, DCW %s, seed %llu\n",
            rq->task == ACG_TASK_COVER ? "cover" : rq->task == ACG_TASK_REPAINT ? "repaint" : "text to music", rq->ref_path ? " with a reference" : "",
            rq->steps, rq->shift, rq->dcw ? "on" : "off", (unsigned long long)rq->seed);
    double t0 = now();
    int rc = ace_generate(e, rq, NULL, &pcm, &info, err, sizeof err);
    if (rc) { fprintf(stderr, "acestep: %s\n", err); ace_close(e); return 1; }
    if (write_wav16(out, pcm, (size_t)info.samples, e->vae->ach, e->vae->rate)) { fprintf(stderr, "acestep: cannot write %s\n", out); free(pcm); ace_close(e); return 1; }
    fprintf(stderr, "[acestep] wrote %s: %.2f s of %d Hz stereo in %.1fs (text %.1f, condition %.1f, denoise %.1f, decode %.1f)%s\n", out,
            (double)info.samples / e->vae->rate, e->vae->rate, now() - t0, info.t_text, info.t_cond, info.t_denoise, info.t_decode,
            info.caption_cut || info.lyric_cut ? "; the caption or the lyrics were cut at their token limit" : "");
    free(pcm); ace_close(e);
    return 0;
}

int main(int argc, char **argv){
    const char *model = NULL, *vae_ref = NULL, *dit_ref = NULL, *tok_ref = NULL, *text_ref = NULL, *gen_ref = NULL, *venc_ref = NULL, *cover_ref = NULL, *wav = NULL, *dump = NULL, *out = NULL;
    int chunk = 256, overlap = 64, serve = 0, make = 0;
    AceRequest rq; memset(&rq, 0, sizeof rq);
    rq.duration = ACS_DEFAULT_DURATION; rq.steps = ACS_DEFAULT_STEPS; rq.shift = ACS_DEFAULT_SHIFT; rq.dcw = 1;
    rq.seed = (uint64_t)((uint32_t)time(NULL) ^ ((uint32_t)clock() << 7));
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--model") && v) { model = v; i++; }
        else if (!strcmp(a, "--gen-ref") && v) { gen_ref = v; i++; }
        else if (!strcmp(a, "--serve")) serve = 1;
        else if (!strcmp(a, "--out") && v) { out = v; i++; }
        else if (!strcmp(a, "--caption") && v) { rq.caption = v; make = 1; i++; }
        else if (!strcmp(a, "--lyrics") && v) { rq.lyrics = v; i++; }
        else if (!strcmp(a, "--language") && v) { rq.language = v; i++; }
        else if (!strcmp(a, "--key") && v) { rq.key = v; i++; }
        else if (!strcmp(a, "--timesig") && v) { rq.timesig = v; i++; }
        else if (!strcmp(a, "--bpm") && v) { rq.bpm = atoi(v); i++; }
        else if (!strcmp(a, "--duration") && v) { rq.duration = atof(v); i++; }
        else if (!strcmp(a, "--steps") && v) { rq.steps = atoi(v); i++; }
        else if (!strcmp(a, "--shift") && v) { rq.shift = atof(v); i++; }
        else if (!strcmp(a, "--seed") && v) { rq.seed = (uint64_t)strtoull(v, NULL, 10); i++; }
        else if (!strcmp(a, "--no-dcw")) rq.dcw = 0;
        else if (!strcmp(a, "--ref") && v) { rq.ref_path = v; make = 1; i++; }
        else if (!strcmp(a, "--cover") && v) { rq.task = ACG_TASK_COVER; rq.src_path = v; make = 1; i++; }
        else if (!strcmp(a, "--repaint") && v) { rq.task = ACG_TASK_REPAINT; rq.src_path = v; make = 1; i++; }
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
        else if (!strcmp(a, "--cover-strength") && v) { rq.cover_strength = atof(v); i++; }
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
    if (serve + make + (gen_ref != NULL) > 1) { fprintf(stderr, "acestep: --serve, --caption and --gen-ref are separate modes\n"); return 2; }
    if (model && make && !rq.caption) rq.caption = "";                  /* a cover or a repaint needs no description: its style is the clip's */
    if (model && make && !out) { fprintf(stderr, "acestep: --caption, --cover, --repaint and --ref need --out FILE.wav\n"); return 2; }
    if (model && serve) {
        char err[512] = "";
        AceEngine *e = ace_open(model, err, sizeof err);
        if (!e) { fprintf(stderr, "acestep: %s\n", err); return 2; }
        int r = acs_serve(e);
        ace_close(e);
        return r;
    }
    if (model && make) return run_make(model, &rq, out);
    if (model && gen_ref) return run_gen_oracle(model, gen_ref, wav);
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
