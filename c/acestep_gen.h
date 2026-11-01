/* acestep_gen.h -- ACE-Step 1.5 text to music, end to end: a request in, 48 kHz stereo out.
 *
 * Joins the three networks of the earlier stages (acestep_text.h, acestep_dit.h, acestep_vae.h) the way upstream's handler
 * does for its text2music task with the turbo DiT and no planner:
 *
 *   prompts -> token ids -> caption: Qwen3-Embedding's last hidden state, lyrics: its embedding rows
 *   -> condition encoders (lyric, timbre = the silence latent's first 750 frames, text) packed into the DiT's cross-attention
 *   -> noise (torch's own generator, so a seed means what it means upstream) -> `steps` Euler steps with DCW
 *   -> the VAE, tiled -> the peak limit -> 16-bit PCM
 *
 * What the handler does for the length of a clip, reproduced here: the duration is rounded to one decimal, the clip is
 * int(duration * 48000) // 1920 latent frames of 25 Hz, never fewer than 128 (the handler pads shorter clips with the
 * silence latent), and the source latents the DiT is conditioned on are the silence latent's first frames (tiled when the
 * clip is longer than the file, which it cannot be within 600 s). The one deliberate difference: a clip asked for under
 * 5.12 s is generated with its 128 frames and then cut to the length asked, where the handler returns all 128.
 *
 * The oracle for all of this is `acestep --gen-ref CASEDIR` (acestep.c): the case's own parameters through this file, against
 * the audio the handler returned. */
#ifndef ACESTEP_GEN_H
#define ACESTEP_GEN_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "acestep_vae.h"
#include "acestep_dit.h"
#include "acestep_text.h"

#define ACG_RATE 48000
#define ACG_HOP 1920                   /* samples per latent frame: 25 latent frames a second */
#define ACG_MIN_FRAMES 128             /* the handler's floor on a clip, 5.12 s */
#define ACG_MIN_SECONDS 1.0
#define ACG_MAX_SECONDS 600.0          /* the handler's DURATION_MAX */
#define ACG_MAX_STEPS 20               /* the schedule has no more entries than this (acd_schedule) */
#define ACG_TIMBRE_FRAMES 750          /* the reference timbre is the silence latent's first 30 s */
#define ACG_MAX_CAPTION_BYTES 4096
#define ACG_MAX_LYRIC_BYTES 32768
#define ACG_MAX_SEED 4294967295.0      /* torch's CPU generator takes a 32-bit seed here */
#define ACG_VAE_CHUNK 256              /* upstream's tiled decode */
#define ACG_VAE_OVERLAP 64

/* ----------------------------------------------------------------------------------------- the noise -- */

/* torch.randn(shape, generator=torch.Generator().manual_seed(seed)) for a float32 tensor of n >= 16 values, as ATen's CPU
 * kernel makes it (aten/src/ATen/native/cpu/DistributionTemplates.h normal_fill): a Mersenne Twister, every value first a
 * uniform of 24 random bits, then Box-Muller on blocks of 16 (the first 8 as radius * cos, the other 8 as radius * sin),
 * and for a size that is not a multiple of 16 the last 16 values are drawn again. ATen's AVX2 build computes the log, cos and
 * sin with Sleef (1 ulp), here libm's: the two agree to a few float32 ulps, which the oracle measures. */
typedef struct { uint32_t s[624]; int i; } AcgMt;

static void acg_mt_seed(AcgMt *m, uint32_t seed){
    m->s[0] = seed;
    for (int j = 1; j < 624; j++) m->s[j] = 1812433253u * (m->s[j - 1] ^ (m->s[j - 1] >> 30)) + (uint32_t)j;
    m->i = 624;
}

static uint32_t acg_mt_next(AcgMt *m){
    if (m->i >= 624) {
        for (int k = 0; k < 624; k++) {
            uint32_t y = (m->s[k] & 0x80000000u) | (m->s[(k + 1) % 624] & 0x7fffffffu);
            m->s[k] = m->s[(k + 397) % 624] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        m->i = 0;
    }
    uint32_t y = m->s[m->i++];
    y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680u; y ^= (y << 15) & 0xefc60000u; y ^= y >> 18;
    return y;
}

static void acg_box_muller16(float *d){
    for (int j = 0; j < 8; j++) {
        const float u1 = 1.0f - d[j], u2 = d[j + 8];
        const float radius = sqrtf(-2.0f * logf(u1));
        const float theta = (float)(2.0 * 3.14159265358979323846 * (double)u2);   /* ATen: 2.0f * c10::pi<double> * u2 */
        d[j] = radius * cosf(theta);
        d[j + 8] = radius * sinf(theta);
    }
}

static void acg_randn(uint32_t seed, size_t n, float *out){
    AcgMt m; acg_mt_seed(&m, seed);
    if (n < 16) {                                   /* ATen's scalar path: one Box-Muller (double) per value; never used for a clip */
        for (size_t i = 0; i < n; i++) {
            double u1 = 1.0 - (double)(acg_mt_next(&m) & 0xffffff) / 16777216.0, u2 = (double)(acg_mt_next(&m) & 0xffffff) / 16777216.0;
            out[i] = (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
        }
        return;
    }
    for (size_t i = 0; i < n; i++) out[i] = (float)(acg_mt_next(&m) & 0xffffff) * (1.0f / 16777216.0f);
    for (size_t i = 0; i + 15 < n; i += 16) acg_box_muller16(out + i);
    if (n % 16) {
        float *tail = out + n - 16;
        for (int i = 0; i < 16; i++) tail[i] = (float)(acg_mt_next(&m) & 0xffffff) * (1.0f / 16777216.0f);
        acg_box_muller16(tail);
    }
}

/* ------------------------------------------------------------------------------------ the silence latent -- */

static uint32_t acg_u16(const uint8_t *p){ return (uint32_t)p[0] | (uint32_t)p[1] << 8; }
static uint32_t acg_u32(const uint8_t *p){ return acg_u16(p) | acg_u16(p + 2) << 16; }

/* A zip entry whose name ends with `suffix`, stored uncompressed: its bytes and size inside the file image, or NULL. */
static const uint8_t *acg_zip_entry(const uint8_t *z, size_t zn, const char *suffix, size_t *size){
    if (zn < 22) return NULL;
    size_t eocd = 0; int found = 0;
    for (size_t i = zn - 22, lim = zn > 22 + 65535 ? zn - 22 - 65535 : 0;; i--) {
        if (acg_u32(z + i) == 0x06054b50u) { eocd = i; found = 1; break; }
        if (i == lim) break;
    }
    if (!found) return NULL;
    const size_t count = acg_u16(z + eocd + 10), cd = acg_u32(z + eocd + 16);
    size_t p = cd, sl = strlen(suffix);
    for (size_t e = 0; e < count; e++) {
        if (p + 46 > zn || acg_u32(z + p) != 0x02014b50u) return NULL;
        const size_t method = acg_u16(z + p + 10), csize = acg_u32(z + p + 20), usize = acg_u32(z + p + 24);
        const size_t nl = acg_u16(z + p + 28), xl = acg_u16(z + p + 30), cl = acg_u16(z + p + 32), lo = acg_u32(z + p + 42);
        if (p + 46 + nl > zn) return NULL;
        if (nl >= sl && !memcmp(z + p + 46 + nl - sl, suffix, sl)) {
            if (method != 0 || csize != usize || lo + 30 > zn || acg_u32(z + lo) != 0x04034b50u) return NULL;
            const size_t start = lo + 30 + acg_u16(z + lo + 26) + acg_u16(z + lo + 28);
            if (start > zn || usize > zn - start) return NULL;
            *size = usize;
            return z + start;
        }
        p += 46 + nl + xl + cl;
    }
    return NULL;
}

/* One integer of a pickle (BININT1 K, BININT2 M, BININT J), or -1. */
static int64_t acg_pickle_int(const uint8_t *p, const uint8_t *end, const uint8_t **next){
    if (p >= end) return -1;
    if (*p == 'K' && p + 2 <= end) { *next = p + 2; return p[1]; }
    if (*p == 'M' && p + 3 <= end) { *next = p + 3; return acg_u16(p + 1); }
    if (*p == 'J' && p + 5 <= end) { *next = p + 5; return (int64_t)(int32_t)acg_u32(p + 1); }
    return -1;
}

/* silence_latent.pt is torch.save of a float32 tensor [1][64][N]: a zip holding a pickle and the raw floats. Read it
 * without torch, accepting only exactly that layout (the pickle's shape and strides are checked), and return it transposed
 * as upstream does, [N][64]. */
static float *acg_load_silence(const char *path, int *frames, char *err, size_t errn){
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errn, "cannot open %.400s", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 100 || sz > (64L << 20)) { fclose(f); snprintf(err, errn, "%.400s is not a silence latent (size)", path); return NULL; }
    uint8_t *z = (uint8_t *)malloc((size_t)sz);
    if (!z || fread(z, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(z); snprintf(err, errn, "cannot read %.400s", path); return NULL; }
    fclose(f);
    size_t pn = 0, dn = 0;
    const uint8_t *pkl = acg_zip_entry(z, (size_t)sz, "/data.pkl", &pn), *data = acg_zip_entry(z, (size_t)sz, "/data/0", &dn);
    float *out = NULL;
    if (!pkl || !data) { snprintf(err, errn, "%.400s is not a torch file this engine reads (no stored data.pkl and data/0)", path); goto done; }
    {
        /* ... FloatStorage ... K 1 K 64 <N> TUPLE3 [memo] <stride0> <stride1> <stride2> TUPLE3: the shape and the strides */
        const uint8_t *e = pkl + pn, *q = NULL;
        int ok = 0; int64_t N = 0;
        for (const uint8_t *s = pkl; s + 4 < e && !ok; s++) {
            if (!(s[0] == 'K' && s[1] == 1 && s[2] == 'K' && s[3] == 64)) continue;
            const uint8_t *nx = NULL;
            N = acg_pickle_int(s + 4, e, &nx);
            if (N < 1 || !nx || nx >= e || *nx != 0x87) continue;
            q = nx + 1;
            if (q < e && *q == 'q') q += 2;                          /* the memo put */
            int64_t st[3]; int good = 1;
            for (int k = 0; k < 3 && good; k++) { st[k] = acg_pickle_int(q, e, &q); if (st[k] < 0) good = 0; }
            if (good && q < e && *q == 0x87 && st[0] == 64 * N && st[1] == N && st[2] == 1) ok = 1;
        }
        int fs = 0; for (const uint8_t *s = pkl; s + 12 <= pkl + pn; s++) if (!memcmp(s, "FloatStorage", 12)) { fs = 1; break; }
        if (!ok || !fs) { snprintf(err, errn, "%.400s: the tensor is not a contiguous float32 [1][64][N]", path); goto done; }
        if ((size_t)N * 64 * 4 != dn) { snprintf(err, errn, "%.400s: %zu bytes of data for a [1][64][%lld] tensor", path, dn, (long long)N); goto done; }
        if (N < ACG_MIN_FRAMES) { snprintf(err, errn, "%.400s holds %lld frames, fewer than %d", path, (long long)N, ACG_MIN_FRAMES); goto done; }
        out = (float *)malloc((size_t)N * 64 * sizeof(float));
        if (!out) { snprintf(err, errn, "out of memory"); goto done; }
        for (int64_t c = 0; c < 64; c++)
            for (int64_t t = 0; t < N; t++) {
                const uint8_t *b = data + ((size_t)c * N + t) * 4; uint32_t u = acg_u32(b); float v; memcpy(&v, &u, 4);
                if (!isfinite(v)) { free(out); out = NULL; snprintf(err, errn, "%.400s holds a non-finite value", path); goto done; }
                out[(size_t)t * 64 + c] = v;
            }
        *frames = (int)N;
    }
done:
    free(z);
    return out;
}

/* ------------------------------------------------------------------------------------------ the engine -- */

typedef struct {
    AceDit *dit; AceText *text; AceTok tok; int tok_open; AcVae *vae;
    float *silence; int silence_frames;
} AceEngine;

static void ace_close(AceEngine *e){
    if (!e) return;
    acd_free(e->dit); act_free(e->text); if (e->tok_open) act_tok_close(&e->tok); acv_free(e->vae); free(e->silence); free(e);
}

/* Load the three networks and the silence latent from the checkpoint directory. NULL (with err) when one is missing or does
 * not fit; the components' own loaders have printed their reasons on stderr. */
static AceEngine *ace_open(const char *model_dir, char *err, size_t errn){
    AceEngine *e = (AceEngine *)calloc(1, sizeof *e);
    char path[4300];
    if (!e) { snprintf(err, errn, "out of memory"); return NULL; }
    if (act_tok_open(&e->tok, model_dir)) { snprintf(err, errn, "cannot open the tokenizer in %.400s", model_dir); ace_close(e); return NULL; }
    e->tok_open = 1;
    if (!(e->text = act_load(model_dir))) { snprintf(err, errn, "cannot load the text encoder from %.400s", model_dir); ace_close(e); return NULL; }
    if (!(e->dit = acd_load(model_dir))) { snprintf(err, errn, "cannot load the DiT from %.400s", model_dir); ace_close(e); return NULL; }
    snprintf(path, sizeof path, "%s/vae", model_dir);
    if (!(e->vae = acv_load(path))) { snprintf(err, errn, "cannot load the VAE from %.400s", path); ace_close(e); return NULL; }
    snprintf(path, sizeof path, "%s/acestep-v15-turbo/silence_latent.pt", model_dir);
    if (!(e->silence = acg_load_silence(path, &e->silence_frames, err, errn))) { ace_close(e); return NULL; }
    if (e->dit->acdim != 64 || e->vae->zdim != e->dit->acdim || e->vae->hop != ACG_HOP || e->vae->rate != ACG_RATE) {
        snprintf(err, errn, "the DiT's latents (%d) and the VAE's (%d, hop %d, %d Hz) do not fit each other or this engine", e->dit->acdim, e->vae->zdim, e->vae->hop, e->vae->rate);
        ace_close(e); return NULL;
    }
    if (e->text->hidden != e->dit->textdim) { snprintf(err, errn, "the text encoder is %d wide, the DiT wants %d", e->text->hidden, e->dit->textdim); ace_close(e); return NULL; }
    return e;
}

typedef struct {
    const char *caption, *lyrics, *language;       /* lyrics: NULL means "[Instrumental]", anything else is used as it is; language: NULL means "unknown" */
    int bpm;                                       /* 0: not given */
    const char *key, *timesig;                     /* NULL or "": not given */
    double duration;                               /* seconds */
    int steps; double shift; int dcw;
    uint64_t seed;
} AceRequest;

typedef struct {
    void (*progress)(void *ud, const char *stage, int step, int steps);
    int (*cancelled)(void *ud);                    /* nonzero stops the request at the next opportunity */
    void *ud;
} AceHooks;

typedef struct {
    int frames, samples;                           /* latent frames the DiT ran on; samples per channel returned */
    int caption_tokens, lyric_tokens, caption_cut, lyric_cut;
    double t_text, t_cond, t_denoise, t_decode;
    float peak;                                    /* the raw waveform's peak before the limit */
} AceInfo;

/* Python's round(x, 1): the nearest decimal to the binary value, which printf does and x * 10 does not at a tie. */
static double acg_round1(double x){
    char b[64]; snprintf(b, sizeof b, "%.1f", x);
    return strtod(b, NULL);
}

/* The latent frames of a clip of `duration` seconds as the handler computes them, and the samples it was asked for. */
static void acg_clip(double duration, int *frames, int *samples){
    const double d = acg_round1(duration);
    const int64_t s = (int64_t)(d * (double)ACG_RATE);
    int64_t f = s / ACG_HOP;
    *samples = (int)s;
    *frames = (int)(f < ACG_MIN_FRAMES ? ACG_MIN_FRAMES : f);
}

/* The reason a request cannot be run, in err, or 0. */
static int ace_check(const AceRequest *r, char *err, size_t errn){
    if (!r->caption) { snprintf(err, errn, "caption is required"); return 1; }          /* an empty one is fine: the handler makes music from lyrics alone, or from nothing */
    if (strlen(r->caption) > ACG_MAX_CAPTION_BYTES) { snprintf(err, errn, "caption longer than %d bytes", ACG_MAX_CAPTION_BYTES); return 1; }
    if (r->lyrics && strlen(r->lyrics) > ACG_MAX_LYRIC_BYTES) { snprintf(err, errn, "lyrics longer than %d bytes", ACG_MAX_LYRIC_BYTES); return 1; }
    if (r->language && strlen(r->language) > 32) { snprintf(err, errn, "language is not a language code"); return 1; }
    if (!(r->duration >= ACG_MIN_SECONDS && r->duration <= ACG_MAX_SECONDS)) { snprintf(err, errn, "duration must be %g to %g seconds", ACG_MIN_SECONDS, ACG_MAX_SECONDS); return 1; }
    if (r->steps < 1 || r->steps > ACG_MAX_STEPS) { snprintf(err, errn, "steps must be 1 to %d", ACG_MAX_STEPS); return 1; }
    if (!(r->shift > 0.0 && r->shift <= 10.0)) { snprintf(err, errn, "shift must be above 0 and at most 10"); return 1; }
    if (r->bpm < 0 || r->bpm > 1000) { snprintf(err, errn, "bpm must be 0 (not given) to 1000"); return 1; }
    if ((r->key && strlen(r->key) > 32) || (r->timesig && strlen(r->timesig) > 8)) { snprintf(err, errn, "key or time signature too long"); return 1; }
    if (r->seed > (uint64_t)ACG_MAX_SEED) { snprintf(err, errn, "seed must be 0 to 4294967295"); return 1; }
    return 0;
}

static double acg_now(void){
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct { const AceHooks *h; int steps; int stopped; } AcgTileCtx;
static int acg_tile_hook(int done, int total, void *ud){
    AcgTileCtx *c = (AcgTileCtx *)ud;
    if (c->h && c->h->progress && done < total) c->h->progress(c->h->ud, "decode", done, total);
    if (c->h && c->h->cancelled && c->h->cancelled(c->h->ud)) { c->stopped = 1; return 1; }
    return 0;
}

/* The silence latent's first n frames, tiled when n is more than the file holds (the handler's _get_silence_latent_slice). */
static void acg_silence_slice(const AceEngine *e, int n, float *out){
    for (int i = 0; i < n; i++) memcpy(out + (size_t)i * 64, e->silence + (size_t)(i % e->silence_frames) * 64, 64 * sizeof(float));
}

/* Make the music. Returns 0 and the interleaved stereo waveform in [-1, 1] (malloc'd, *pcm, info->samples frames), 1 with a
 * reason in err, or 2 when the hooks asked to stop. */
static int ace_generate(AceEngine *e, const AceRequest *rq, const AceHooks *h, float **pcm, AceInfo *info, char *err, size_t errn){
    memset(info, 0, sizeof *info);
    *pcm = NULL;
    if (ace_check(rq, err, errn)) return 1;
    const AceDit *d = e->dit;
    const int A = d->acdim;
    int rc = 1;
    char *cp = NULL, *lp = NULL;
    float *text = NULL, *lyric = NULL, *timbre = NULL, *src = NULL, *ctx = NULL, *noise = NULL, *x = NULL, *nx = NULL, *v = NULL, *wav = NULL;
    AcdCond *cond = NULL;
    int *lids = NULL;
    #define ACG_STOP() do { if (h && h->cancelled && h->cancelled(h->ud)) { rc = 2; snprintf(err, errn, "cancelled"); goto out; } } while (0)
    #define ACG_PROGRESS(stage, a, b) do { if (h && h->progress) h->progress(h->ud, stage, a, b); } while (0)

    /* 1. the prompts, ids and the text encoder */
    double t0 = acg_now();
    ACG_PROGRESS("text", 0, 1);
    const char *lyrics = rq->lyrics ? rq->lyrics : "[Instrumental]";      /* as given, even if empty (the handler does); only none at all is instrumental */
    char bpm[16] = ""; if (rq->bpm > 0) snprintf(bpm, sizeof bpm, "%d", rq->bpm);
    cp = act_caption_prompt(NULL, rq->caption, bpm, rq->key, rq->timesig, rq->duration);
    lp = act_lyric_prompt(lyrics, rq->language && *rq->language ? rq->language : "unknown");
    int cids[ACT_CAPTION_MAX];
    lids = (int *)malloc(ACT_LYRIC_MAX * sizeof(int));
    if (!lids) { snprintf(err, errn, "out of memory"); goto out; }
    const int cn = act_tokenize(&e->tok, cp, ACT_CAPTION_MAX, cids), ln = act_tokenize(&e->tok, lp, ACT_LYRIC_MAX, lids);
    info->caption_tokens = cn; info->lyric_tokens = ln;
    info->caption_cut = cn == ACT_CAPTION_MAX; info->lyric_cut = ln == ACT_LYRIC_MAX;      /* an exact fit counts as cut: it cannot be told apart cheaply */
    text = act_encode(e->text, cids, cn);
    lyric = act_embed(e->text, lids, ln);
    if (!text || !lyric) { snprintf(err, errn, "the text encoder failed"); goto out; }
    info->t_text = acg_now() - t0;
    ACG_STOP();

    /* 2. the condition: lyric, timbre (the silence latent's first frames) and text */
    t0 = acg_now();
    ACG_PROGRESS("condition", 0, 1);
    timbre = (float *)malloc((size_t)ACG_TIMBRE_FRAMES * A * sizeof(float));
    if (!timbre) { snprintf(err, errn, "out of memory"); goto out; }
    acg_silence_slice(e, ACG_TIMBRE_FRAMES, timbre);
    cond = acd_cond_new(d, text, cn, lyric, ln, timbre, ACG_TIMBRE_FRAMES);
    if (!cond) { snprintf(err, errn, "the condition encoders failed"); goto out; }
    info->t_cond = acg_now() - t0;
    ACG_STOP();

    /* 3. the clip: source latents (silence), the all-true mask, noise */
    int T, want_samples;
    acg_clip(rq->duration, &T, &want_samples);
    info->frames = T;
    src = (float *)malloc((size_t)T * A * sizeof(float));
    ctx = (float *)malloc((size_t)T * 2 * A * sizeof(float));
    noise = (float *)malloc((size_t)T * A * sizeof(float));
    x = (float *)malloc((size_t)T * A * sizeof(float)); nx = (float *)malloc((size_t)T * A * sizeof(float)); v = (float *)malloc((size_t)T * A * sizeof(float));
    if (!src || !ctx || !noise || !x || !nx || !v) { snprintf(err, errn, "out of memory"); goto out; }
    acg_silence_slice(e, T, src);
    for (int f = 0; f < T; f++) {
        memcpy(ctx + (size_t)f * 2 * A, src + (size_t)f * A, (size_t)A * sizeof(float));
        for (int c = 0; c < A; c++) ctx[(size_t)f * 2 * A + A + c] = 1.0f;
    }
    acg_randn((uint32_t)rq->seed, (size_t)T * A, noise);

    /* 4. the Euler steps. acd_sample is this loop; it is spelled out here to report each step and to stop between them. */
    float sched[ACG_MAX_STEPS];
    const int n = acd_schedule(rq->steps, (float)rq->shift, sched);
    if (n < 1) { snprintf(err, errn, "no schedule for %d steps", rq->steps); goto out; }
    AcdSampleOpts so = { rq->dcw ? 1 : 0, 0.05, 0.02 };
    memcpy(x, noise, (size_t)T * A * sizeof(float));
    t0 = acg_now();
    for (int i = 0; i < n; i++) {
        ACG_PROGRESS("denoise", i, n);
        ACG_STOP();
        acd_forward(d, cond, x, ctx, T, sched[i], sched[i], v);
        acd_sample_step(x, v, i, n, sched, T, A, &so, nx);
        float *s = x; x = nx; nx = s;
    }
    ACG_PROGRESS("denoise", n, n);
    for (size_t k = 0; k < (size_t)T * A; k++) if (!isfinite(x[k])) { snprintf(err, errn, "the sampler produced a non-finite latent"); goto out; }
    info->t_denoise = acg_now() - t0;
    ACG_STOP();

    /* 5. the VAE, tiled as upstream tiles it, then the peak limit */
    t0 = acg_now();
    const size_t total = (size_t)T * ACG_HOP;
    wav = (float *)malloc(total * e->vae->ach * sizeof(float));
    if (!wav) { snprintf(err, errn, "out of memory"); goto out; }
    AcgTileCtx tc = { h, 0, 0 };
    if (acv_decode_tiled(e->vae, x, T, ACG_VAE_CHUNK, ACG_VAE_OVERLAP, wav, acg_tile_hook, &tc)) {
        if (tc.stopped) { rc = 2; snprintf(err, errn, "cancelled"); } else snprintf(err, errn, "the VAE decode failed");
        goto out;
    }
    info->peak = acv_normalize(wav, total * e->vae->ach);
    info->t_decode = acg_now() - t0;
    info->samples = (int)total < want_samples ? (int)total : want_samples;
    *pcm = wav; wav = NULL;
    rc = 0;
out:
    free(cp); free(lp); free(lids); free(text); free(lyric); free(timbre); free(src); free(ctx); free(noise); free(x); free(nx); free(v); free(wav);
    acd_cond_free(cond);
    #undef ACG_STOP
    #undef ACG_PROGRESS
    return rc;
}

#endif
