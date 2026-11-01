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
 *   1. the VAE decoder        acestep --model DIR --vae-ref REFDIR           (this file today)
 *   2. the DiT and its sampler
 *   3. the text encoder
 *   4. the engine proper: a serve protocol, WAV output
 *
 * Usage:
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

static void usage(void){
    fprintf(stderr,
        "usage: acestep --model DIR --vae-ref REFDIR [--chunk N] [--overlap N] [--wav OUT.wav] [--dump PREFIX] [--threads N]\n"
        "  DIR     the ACE-Step checkpoint (holds vae/)\n"
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

/* Compare a computed tensor with the reference; returns 1 when it breaks either limit. */
static int cmp(const char *what, const float *got, const float *want, size_t n){
    double emax = 0, en = 0, wn = 0, wpeak = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - want[i];
        if (fabs(d) > emax) emax = fabs(d);
        if (fabs((double)want[i]) > wpeak) wpeak = fabs((double)want[i]);
        en += d * d; wn += (double)want[i] * want[i];
    }
    double rel = wn > 0 ? sqrt(en / wn) : sqrt(en);
    int bad = rel > TOL_REL || emax > TOL_ABS * (wpeak > 1.0 ? wpeak : 1.0);
    fprintf(stderr, "[vae-oracle] %-30s n=%-9zu max|err| %.3e  rel %.3e%s\n", what, n, emax, rel, bad ? "  <-- FAIL" : "");
    return bad;
}

/* A float tensor of the reference file, widened to f32; its element count in *n. */
static float *ref_tensor(shards *S, const char *name, int64_t *n, int64_t *shape, int *rank){
    st_tensor *t = st_find(S, name);
    if (!t) return NULL;
    if (t->dtype > 2) { fprintf(stderr, "[vae-oracle] %s is not a float tensor\n", name); return NULL; }
    float *v = (float *)malloc((size_t)(t->numel > 0 ? t->numel : 1) * sizeof(float));
    if (!v) { fprintf(stderr, "[vae-oracle] out of memory\n"); exit(1); }
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
    fprintf(stderr, "[vae-oracle] all %d comparison(s) within rel %.0e and abs %.0e\n", cases, TOL_REL, TOL_ABS);
    return 0;
}

int main(int argc, char **argv){
    const char *model = NULL, *vae_ref = NULL, *wav = NULL, *dump = NULL;
    int chunk = 256, overlap = 64;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--model") && v) { model = v; i++; }
        else if (!strcmp(a, "--vae-ref") && v) { vae_ref = v; i++; }
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
    if (!model || !vae_ref) { usage(); return 2; }
    return run_vae_oracle(model, vae_ref, chunk, overlap, wav, dump);
}
