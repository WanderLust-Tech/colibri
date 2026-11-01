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
 *   3. the text encoder
 *   4. the engine proper: a serve protocol, WAV output
 *
 * Usage:
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
#include "acestep_dit.h"

static void usage(void){
    fprintf(stderr,
        "usage: acestep --model DIR --dit-ref REFDIR [--dump PREFIX] [--threads N]\n"
        "       acestep --model DIR --vae-ref REFDIR [--chunk N] [--overlap N] [--wav OUT.wav] [--dump PREFIX] [--threads N]\n"
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
    if (t->dtype > 2) { fprintf(stderr, "[%s] %s is not a float tensor\n", g_tag, name); return NULL; }
    float *v = (float *)malloc((size_t)(t->numel > 0 ? t->numel : 1) * sizeof(float));
    if (!v) { fprintf(stderr, "[%s] out of memory\n", g_tag); exit(1); }
    st_read_f32(S, name, v, 0);
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

static void tile_progress(int done, int total, void *ud){
    (void)ud;
    if (total > 1 && done < total) fprintf(stderr, "\r[vae-oracle] tiled decode: window %d/%d", done + 1, total);
    else if (total > 1) fprintf(stderr, "\r%60s\r", "");
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

int main(int argc, char **argv){
    const char *model = NULL, *vae_ref = NULL, *dit_ref = NULL, *wav = NULL, *dump = NULL;
    int chunk = 256, overlap = 64;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--model") && v) { model = v; i++; }
        else if (!strcmp(a, "--vae-ref") && v) { vae_ref = v; i++; }
        else if (!strcmp(a, "--dit-ref") && v) { dit_ref = v; i++; }
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
    if (!model || (!vae_ref && !dit_ref)) { usage(); return 2; }
    int rc = 0;
    if (dit_ref) rc = run_dit_oracle(model, dit_ref, dump);
    if (vae_ref) { g_tag = "vae-oracle"; g_tol_rel = TOL_REL; g_tol_abs = TOL_ABS; int r2 = run_vae_oracle(model, vae_ref, chunk, overlap, wav, dump); if (r2 > rc) rc = r2; }
    return rc;
}
