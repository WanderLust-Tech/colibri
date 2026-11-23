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
#include "acestep_vae_enc.h"
#include "acestep_dit.h"
#include "acestep_tok.h"
#include "acestep_text.h"
#include "acestep_audio.h"
#include "acestep_lm.h"
#include "acestep_meta.h"
#ifdef COLI_VULKAN
#include "acestep_dit_vk.h"           /* the DiT's steps on the card (acd_forward calls it first) */
#include "acestep_vae_vk.h"           /* the VAE's windows on the card (acv_decode_ws calls it first) */
#endif

#define ACG_RATE 48000
#define ACG_HOP 1920                   /* samples per latent frame: 25 latent frames a second */
#define ACG_MIN_FRAMES 128             /* the handler's floor on a clip, 5.12 s */
#define ACG_MIN_SECONDS 1.0
#define ACG_MAX_SECONDS 600.0          /* the handler's DURATION_MAX */
#define ACG_MAX_STEPS 200              /* the most a base or SFT DiT takes: its schedule is a linspace of any length */
#define ACG_MAX_STEPS_TURBO 20         /* the turbo's schedule has no more entries than this (acd_schedule) */
#define ACG_TIMBRE_FRAMES 750          /* the reference timbre is the silence latent's first 30 s */
#define ACG_MAX_CAPTION_BYTES 4096
#define ACG_MAX_LYRIC_BYTES 32768
#define ACG_MAX_SEED 4294967295.0      /* torch's CPU generator takes a 32-bit seed here */
#define ACG_VAE_CHUNK 256              /* upstream's tiled decode */
#define ACG_VAE_OVERLAP 64
#define ACG_VAE_ENC_CHUNK ((int64_t)30 * ACG_RATE)     /* the encoder's window: what a GPU with memory to spare gets from upstream (a CPU gets 15 s) */
#define ACG_VAE_ENC_OVERLAP ((int64_t)2 * ACG_RATE)
#define ACG_VAE_SEED_XOR 0x5EED0001u   /* the posterior draws of the encoder come from a generator seeded with the request's seed xor this */
#define ACG_REF_SECONDS 30             /* a reference clip is made 30 s: three segments of 10 s */
#define ACG_REF_SEGMENT 10

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

/* One torch.randn call's worth, drawn from a generator that goes on afterwards: several calls on one generator are what
 * `torch.randn(shape, generator=g)` called several times does (the posterior draws of a tiled VAE encode). */
static void acg_randn_mt(AcgMt *m, size_t n, float *out){
    if (n < 16) {                                   /* ATen's scalar path: one Box-Muller (double) per value; never used for a clip */
        for (size_t i = 0; i < n; i++) {
            double u1 = 1.0 - (double)(acg_mt_next(m) & 0xffffff) / 16777216.0, u2 = (double)(acg_mt_next(m) & 0xffffff) / 16777216.0;
            out[i] = (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
        }
        return;
    }
    for (size_t i = 0; i < n; i++) out[i] = (float)(acg_mt_next(m) & 0xffffff) * (1.0f / 16777216.0f);
    for (size_t i = 0; i + 15 < n; i += 16) acg_box_muller16(out + i);
    if (n % 16) {
        float *tail = out + n - 16;
        for (int i = 0; i < 16; i++) tail[i] = (float)(acg_mt_next(m) & 0xffffff) * (1.0f / 16777216.0f);
        acg_box_muller16(tail);
    }
}

static void acg_randn(uint32_t seed, size_t n, float *out){
    AcgMt m; acg_mt_seed(&m, seed);
    acg_randn_mt(&m, n, out);
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

/* One DiT of the checkpoint: acestep-v15-turbo, -base or -sft. The turbo is guided inside (8 steps); the base and SFT ones are guided at
 * sampling (up to 200 steps) and do the extract, lego and complete tasks. Each has its own tokenizer for covers and its own silence latent
 * (the file beside its weights); the text encoder, the VAE and its encoder are the checkpoint's own and shared. */
typedef struct { char name[64]; AceDit *dit; AcdTok *cover; float *silence; int silence_frames; } AceSlot;

typedef struct {
    AceDit *dit; AcdTok *cover; float *silence; int silence_frames;   /* the active slot's: ace_select_for points these at slot[cur] */
    AceText *text; AceTok tok; int tok_open; AcVae *vae;
    char model_dir[4096];
    char dit_name[64];                             /* the DiT the engine was opened on: what a request that names none runs on */
    AceSlot slot[3]; int nslot, cur;               /* the DiTs loaded so far: the others come in the first time a request names them */
    AcvEnc *venc;                                  /* the VAE encoder: loaded the first time a request brings audio in */
    AceLm *lm;                                     /* the 5 Hz planner: loaded the first time a request asks to plan (ace_planner) */
} AceEngine;

static void ace_close(AceEngine *e){
    if (!e) return;
    for (int i = 0; i < e->nslot; i++) { acd_free(e->slot[i].dit); atk_free(e->slot[i].cover); free(e->slot[i].silence); }
    act_free(e->text); if (e->tok_open) act_tok_close(&e->tok); acv_free(e->vae);
    acve_free(e->venc); alm_free(e->lm);
    free(e);
}

static void ace_alias(AceEngine *e){
    const AceSlot *s = &e->slot[e->cur];
    e->dit = s->dit; e->cover = s->cover; e->silence = s->silence; e->silence_frames = s->silence_frames;
}

/* Load a DiT and its silence latent into the next slot, after the VAE and the text encoder. 0, or -1 with err. */
static int ace_load_slot(AceEngine *e, const char *dit_name, char *err, size_t errn){
    char path[4300];
    if (e->nslot >= 3) { snprintf(err, errn, "too many DiTs loaded"); return -1; }
    AceSlot *s = &e->slot[e->nslot];
    memset(s, 0, sizeof *s);
    snprintf(s->name, sizeof s->name, "%s", dit_name);
    if (!(s->dit = acd_load_variant(e->model_dir, dit_name))) { snprintf(err, errn, "cannot load the DiT %.60s from %.400s", dit_name, e->model_dir); return -1; }
#ifdef COLI_VULKAN
    adv_attach();
#endif
    snprintf(path, sizeof path, "%s/%s/silence_latent.pt", e->model_dir, dit_name);
    if (!(s->silence = acg_load_silence(path, &s->silence_frames, err, errn))) { acd_free(s->dit); memset(s, 0, sizeof *s); return -1; }
    if (s->dit->acdim != 64 || e->vae->zdim != s->dit->acdim || e->vae->hop != ACG_HOP || e->vae->rate != ACG_RATE) {
        snprintf(err, errn, "the DiT's latents (%d) and the VAE's (%d, hop %d, %d Hz) do not fit each other or this engine", s->dit->acdim, e->vae->zdim, e->vae->hop, e->vae->rate);
        acd_free(s->dit); free(s->silence); memset(s, 0, sizeof *s); return -1;
    }
    if (e->text->hidden != s->dit->textdim) {
        snprintf(err, errn, "the text encoder is %d wide, the DiT wants %d", e->text->hidden, s->dit->textdim);
        acd_free(s->dit); free(s->silence); memset(s, 0, sizeof *s); return -1;
    }
    e->nslot++;
    return 0;
}

/* Load the networks and the silence latent from the checkpoint directory, on the DiT `dit_name`. NULL (with err) when one is missing
 * or does not fit; the components' own loaders have printed their reasons on stderr. */
static AceEngine *ace_open_dit(const char *model_dir, const char *dit_name, char *err, size_t errn){
    AceEngine *e = (AceEngine *)calloc(1, sizeof *e);
    char path[4300];
    if (!e) { snprintf(err, errn, "out of memory"); return NULL; }
    if (snprintf(e->model_dir, sizeof e->model_dir, "%s", model_dir) >= (int)sizeof e->model_dir) { snprintf(err, errn, "the model path is too long"); free(e); return NULL; }
    if (act_tok_open(&e->tok, model_dir)) { snprintf(err, errn, "cannot open the tokenizer in %.400s", model_dir); ace_close(e); return NULL; }
    e->tok_open = 1;
    if (!(e->text = act_load(model_dir))) { snprintf(err, errn, "cannot load the text encoder from %.400s", model_dir); ace_close(e); return NULL; }
    snprintf(path, sizeof path, "%s/vae", model_dir);
    if (!(e->vae = acv_load(path))) { snprintf(err, errn, "cannot load the VAE from %.400s", path); ace_close(e); return NULL; }
#ifdef COLI_VULKAN
    avv_attach();
#endif
    snprintf(e->dit_name, sizeof e->dit_name, "%s", dit_name);
    if (ace_load_slot(e, dit_name, err, errn)) { ace_close(e); return NULL; }
    e->cur = 0; ace_alias(e);
    return e;
}

static AceEngine *ace_open(const char *model_dir, char *err, size_t errn){ return ace_open_dit(model_dir, "acestep-v15-turbo", err, errn); }

/* The planners in a checkpoint, smallest last: acestep-5Hz-lm-0.6B, -1.7B and -4B (the 1.7B ships with the checkpoint). */
static const char *const ACE_LM_DIRS[3] = { "acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-4B", "acestep-5Hz-lm-0.6B" };

/* "0.6B", "1.7B" or "4B", or a whole directory name, to the directory name; NULL or "" is the first one that is there. 0, or -1 with err. */
static int ace_lm_dir(const AceEngine *e, const char *given, char *out, size_t n, char *err, size_t errn){
    char path[4400];
    if (given && *given) {
        for (int i = 0; i < 3; i++)
            if (!strcmp(given, ACE_LM_DIRS[i]) || !strcmp(given, ACE_LM_DIRS[i] + 15)) { snprintf(out, n, "%s", ACE_LM_DIRS[i]); goto check; }
        snprintf(err, errn, "lm must be 0.6B, 1.7B or 4B"); return -1;
    }
    for (int i = 0; i < 3; i++) {
        snprintf(path, sizeof path, "%s/%s/config.json", e->model_dir, ACE_LM_DIRS[i]);
        FILE *f = fopen(path, "rb");
        if (f) { fclose(f); snprintf(out, n, "%s", ACE_LM_DIRS[i]); return 0; }
    }
    snprintf(err, errn, "this checkpoint has no 5 Hz planner (acestep-5Hz-lm-1.7B is not in it)"); return -1;
check:
    snprintf(path, sizeof path, "%s/%s/config.json", e->model_dir, out);
    { FILE *f = fopen(path, "rb"); if (!f) { snprintf(err, errn, "the planner %s is not installed in this checkpoint", out); return -1; } fclose(f); }
    return 0;
}

/* The planner, loaded on first use and kept; asking for another size replaces it. */
static AceLm *ace_planner(AceEngine *e, const char *given, char *err, size_t errn){
    char dir[64], path[4400];
    if (ace_lm_dir(e, given, dir, sizeof dir, err, errn)) return NULL;
    if (e->lm && !strcmp(e->lm->dir_name, dir)) return e->lm;
    alm_free(e->lm); e->lm = NULL;
    snprintf(path, sizeof path, "%s/%s", e->model_dir, dir);
    if (!(e->lm = alm_load(path))) { snprintf(err, errn, "cannot load the planner from %.400s", path); return NULL; }
    return e->lm;
}

/* Is the DiT's directory in the checkpoint? */
static int ace_have_dit(const AceEngine *e, const char *dir_name){
    char path[4400];
    snprintf(path, sizeof path, "%s/%s/config.json", e->model_dir, dir_name);
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

/* "turbo", "base" or "sft", or the whole directory name, to the directory name. 0, or -1 for anything else. */
static int ace_dit_dir(const char *given, char *out, size_t n){
    if (!strcmp(given, "turbo") || !strcmp(given, "base") || !strcmp(given, "sft")) { snprintf(out, n, "acestep-v15-%s", given); return 0; }
    if (!strcmp(given, "acestep-v15-turbo") || !strcmp(given, "acestep-v15-base") || !strcmp(given, "acestep-v15-sft")) { snprintf(out, n, "%s", given); return 0; }
    return -1;
}

/* The VAE encoder, loaded on first use. */
static AcvEnc *ace_encoder(AceEngine *e, char *err, size_t errn){
    if (!e->venc) {
        char path[4300];
        snprintf(path, sizeof path, "%s/vae", e->model_dir);
        if (!(e->venc = acve_load(path))) { snprintf(err, errn, "cannot load the VAE encoder from %.400s", path); return NULL; }
        if (e->venc->zdim != e->dit->acdim || e->venc->hop != ACG_HOP || e->venc->rate != ACG_RATE || e->venc->ach != 2) {
            snprintf(err, errn, "the VAE encoder does not fit the DiT's latents or this engine");
            acve_free(e->venc); e->venc = NULL; return NULL;
        }
    }
    return e->venc;
}

/* The audio tokenizer and detokenizer, loaded on first use. */
static AcdTok *ace_cover_net(AceEngine *e, char *err, size_t errn){
    if (!e->cover && !(e->cover = atk_load(e->dit, e->model_dir))) { snprintf(err, errn, "cannot load the audio tokenizer from %.400s", e->model_dir); return NULL; }
    e->slot[e->cur].cover = e->cover;
    return e->cover;
}

#define ACG_TASK_TEXT2MUSIC 0
#define ACG_TASK_COVER 1
#define ACG_TASK_REPAINT 2
#define ACG_TASK_EXTRACT 3                         /* the base and SFT DiTs only: pull one track out of the source */
#define ACG_TASK_LEGO 4                            /* ... add one track to the source inside a span (a repaint of one instrument) */
#define ACG_TASK_COMPLETE 5                        /* ... complete a partial track with others; the source is optional */
#define ACG_MAX_TRACK 64
#define ACG_CODE_FRAMES 5                        /* latent frames a planner code stands for: 25 Hz frames pooled by 5 to 5 codes a second */
#define ACG_MAX_CODES 3000                         /* 600 seconds of them */
#define ACG_REPAINT_BALANCED 0
#define ACG_REPAINT_CONSERVATIVE 1
#define ACG_REPAINT_AGGRESSIVE 2

/* A request. The duration, steps and shift have no zero default (the callers set them: ACS_DEFAULT_*); the audio fields below do,
 * so a request that never heard of audio is unchanged. */
typedef struct {
    const char *caption, *lyrics, *language;       /* lyrics: NULL means "[Instrumental]", anything else is used as it is; language: NULL means "unknown" */
    int bpm;                                       /* 0: not given */
    const char *key, *timesig;                     /* NULL or "": not given */
    double duration;                               /* seconds; a cover or a repaint takes its source's */
    int steps; double shift; int dcw;
    uint64_t seed;
    /* audio in (0 and NULL are the defaults): */
    int task;                                      /* ACG_TASK_*: text to music, a cover, a repaint, an extract, a lego or a complete of src */
    const char *src_path;                          /* the clip a cover or a repaint works on: a WAV or an MP3 */
    const char *ref_path;                          /* a clip to take the timbre and style from, in any task */
    double repaint_start, repaint_end;             /* seconds in the clip; an end of 0 or less is its end, a start before 0 adds silence before it and an end past it silence after it */
    int repaint_mode;                              /* ACG_REPAINT_*: how much of the source the span's edges keep */
    double repaint_strength;                       /* balanced mode, 0 to 1, higher = freer: 0 is the default, 0.5 */
    double cover_strength;                         /* a cover: 0 to 1, the share of the steps that follow the source's structure; 0 is the default, 1 */
    double cover_noise;                            /* a cover: 0 to 1, how near the source the first step starts; 0 starts from noise */
    /* the base and SFT DiTs (0, NULL and 0 are the defaults; the turbo DiT has guidance baked in and ignores the first three): */
    double guidance;                               /* classifier-free guidance scale; 0 is the default, 7; 1 or less turns it off */
    double cfg_start, cfg_end;                     /* the part of the schedule (t from 1 down to 0) guidance applies to; an end of 0 is 1 */
    int adg;                                       /* angle-based guidance instead of APG */
    const char *track;                             /* extract and lego: the track (drums, bass, vocals ...); NULL or "" is "the track" */
    const char *tracks;                            /* complete: the tracks to complete the input with, separated by commas */
    const char *dit;                               /* turbo, base or sft; NULL or "": see ace_select_for */
    const int *codes; int ncodes;                  /* the planner's audio codes (0..63999): text to music from them, as upstream's cover of codes; ncodes 0 is none */
    /* the planner (text to music only): the 5 Hz language model first writes the song's metadata (the tempo, key, time signature, duration, language and a
     * rewritten caption; whatever of bpm, key, timesig and duration is not given) and then the codes of its structure, which the DiT takes as its context.
     * With all four given the first phase is skipped, as upstream skips it. A duration of 0 or less lets the model choose (10 to 600 seconds). The callers set
     * the sampling defaults upstream has: temperature 0.85, guidance 2 (the codes only), top-p 0.9, top-k off. */
    int plan;
    const char *lm;                                /* 0.6B, 1.7B or 4B; NULL or "": the first the checkpoint has */
    double lm_temperature, lm_cfg_scale, lm_top_p; int lm_top_k;
    int lm_caption_off, lm_language_off;           /* 1: do not have the model rewrite the caption (the DiT keeps yours), or write the language */
} AceRequest;

/* Taps: the oracle's view of what the stages made, by name ("noise", "timbre", "src_latents", "clean_src_latents", "context",
 * "latents"), each a float array of n values. NULL in production. */
typedef struct {
    void (*progress)(void *ud, const char *stage, int step, int steps);
    int (*cancelled)(void *ud);                    /* nonzero stops the request at the next opportunity */
    void *ud;
    void (*tap)(void *ud, const char *name, const float *x, size_t n);
    /* an oracle's way to feed a stage what the reference had instead of what this engine made ("source_latents": the clean latents of the
     * source clip, [T][64]): returns n floats to use, or NULL to keep our own. NULL in production. */
    const float *(*feed)(void *ud, const char *name, size_t n);
} AceHooks;

typedef struct {
    int frames, samples;                           /* latent frames the DiT ran on; samples per channel returned */
    int caption_tokens, lyric_tokens, caption_cut, lyric_cut;
    double t_text, t_encode, t_cond, t_denoise, t_decode;
    float peak;                                    /* the raw waveform's peak before the limit */
    double src_seconds;                            /* the source clip's length, 0 for text to music */
    char dit[64];                                  /* the DiT that ran: acestep-v15-turbo, -base or -sft */
    char lm[64];                                   /* the planner that ran, "" without one */
    int plan_codes; double t_plan;                 /* the codes it wrote, and the seconds it took (the model's load included) */
    int meta_run;                                  /* the planner's first phase ran: what it wrote follows */
    int meta_bpm; double meta_duration;
    char meta_key[48], meta_timesig[16], meta_language[48], meta_caption[2600];
    double guidance;                               /* the guidance scale applied (0 for the turbo DiT, or when it is 1 or less) */
} AceInfo;

/* Python's round(x, 1): the nearest decimal to the binary value, which printf does and x * 10 does not at a tie. */
static double acg_round1(double x){
    char b[64]; snprintf(b, sizeof b, "%.1f", x);
    return strtod(b, NULL);
}

/* Python's round(x): to nearest, ties to even (the repaint's step and frame counts). */
static int64_t acg_pyround(double x){
    double r = floor(x + 0.5);
    if (r - x == 0.5 && fmod(r, 2.0) != 0.0) r -= 1.0;
    return (int64_t)r;
}

/* Python's a // b for floats (CPython's float_floor_div), which is not floor(a / b) when the quotient rounds up to a whole number. */
static double acg_floordiv(double a, double b){
    double mod = fmod(a, b), div = (a - mod) / b;
    if (mod != 0.0 && ((b < 0) != (mod < 0))) { mod += b; div -= 1.0; }
    double fl = floor(div);
    if (div - fl > 0.5) fl += 1.0;
    return fl;
}

/* The latent frames of a clip of `duration` seconds as the handler computes them, and the samples it was asked for. */
static void acg_clip(double duration, int *frames, int *samples){
    const double d = acg_round1(duration);
    const int64_t s = (int64_t)(d * (double)ACG_RATE);
    int64_t f = s / ACG_HOP;
    *samples = (int)s;
    *frames = (int)(f < ACG_MIN_FRAMES ? ACG_MIN_FRAMES : f);
}

/* What a repaint's mode and strength mean (the handler's _resolve_repaint_config): the share of the steps that keep the source
 * outside the span, the frames of crossfade at the span's edges in latent space, and the seconds of it in the waveform. */
static void acg_repaint_config(int mode, double strength, double *inject, int *cf_frames, double *wav_cf){
    if (strength <= 0) strength = 0.5;
    if (strength > 1) strength = 1;
    if (mode == ACG_REPAINT_AGGRESSIVE) { *inject = 0.0; *cf_frames = 0; *wav_cf = 0.0; return; }
    if (mode == ACG_REPAINT_CONSERVATIVE) { *inject = 1.0; *cf_frames = 25; *wav_cf = 0.05; return; }
    const double inv = 1.0 - strength;
    *inject = inv; *cf_frames = (int)acg_pyround(25 * inv); *wav_cf = 0.05 * inv;
}

/* The reason a request cannot be run, in err, or 0. */
static int ace_check(const AceRequest *r, char *err, size_t errn){
    if (!r->caption) { snprintf(err, errn, "caption is required"); return 1; }          /* an empty one is fine: the handler makes music from lyrics alone, or from nothing */
    if (strlen(r->caption) > ACG_MAX_CAPTION_BYTES) { snprintf(err, errn, "caption longer than %d bytes", ACG_MAX_CAPTION_BYTES); return 1; }
    if (r->lyrics && strlen(r->lyrics) > ACG_MAX_LYRIC_BYTES) { snprintf(err, errn, "lyrics longer than %d bytes", ACG_MAX_LYRIC_BYTES); return 1; }
    if (r->language && strlen(r->language) > 32) { snprintf(err, errn, "language is not a language code"); return 1; }
    if (r->task < ACG_TASK_TEXT2MUSIC || r->task > ACG_TASK_COMPLETE) { snprintf(err, errn, "task must be text2music, cover, repaint, extract, lego or complete"); return 1; }
    if ((r->task == ACG_TASK_TEXT2MUSIC || (r->task == ACG_TASK_COMPLETE && !(r->src_path && *r->src_path))) && !(r->duration >= ACG_MIN_SECONDS && r->duration <= ACG_MAX_SECONDS) &&
        !(r->plan && r->duration <= 0)) { snprintf(err, errn, "duration must be %g to %g seconds", ACG_MIN_SECONDS, ACG_MAX_SECONDS); return 1; }
    if (r->steps < 1 || r->steps > ACG_MAX_STEPS) { snprintf(err, errn, "steps must be 1 to %d", ACG_MAX_STEPS); return 1; }
    if (!(r->shift > 0.0 && r->shift <= 10.0)) { snprintf(err, errn, "shift must be above 0 and at most 10"); return 1; }
    if (r->bpm < 0 || r->bpm > 1000) { snprintf(err, errn, "bpm must be 0 (not given) to 1000"); return 1; }
    if ((r->key && strlen(r->key) > 32) || (r->timesig && strlen(r->timesig) > 8)) { snprintf(err, errn, "key or time signature too long"); return 1; }
    if (r->seed > (uint64_t)ACG_MAX_SEED) { snprintf(err, errn, "seed must be 0 to 4294967295"); return 1; }
    static const char *const task_names[6] = { "text to music", "a cover", "a repaint", "an extract", "a lego", "a complete" };
    const int needs_src = r->task != ACG_TASK_TEXT2MUSIC && r->task != ACG_TASK_COMPLETE;
    if (needs_src && !(r->src_path && *r->src_path)) { snprintf(err, errn, "%s needs a source clip (src)", task_names[r->task]); return 1; }
    if (r->task == ACG_TASK_TEXT2MUSIC && r->src_path && *r->src_path) { snprintf(err, errn, "a source clip is for a cover, a repaint, an extract, a lego or a complete; for text to music, give a reference clip (ref) to take its style"); return 1; }
    if (r->src_path && strlen(r->src_path) > 4000) { snprintf(err, errn, "src path too long"); return 1; }
    if (r->ref_path && strlen(r->ref_path) > 4000) { snprintf(err, errn, "ref path too long"); return 1; }
    if (r->plan) {
        if (r->task != ACG_TASK_TEXT2MUSIC || r->ncodes > 0) { snprintf(err, errn, "the planner is for text to music, with no codes of your own"); return 1; }
        if (!(r->lm_cfg_scale >= 0 && r->lm_cfg_scale <= 20) || !(r->lm_temperature >= 0 && r->lm_temperature <= 2) || !(r->lm_top_p >= 0 && r->lm_top_p <= 1) || r->lm_top_k < 0 || r->lm_top_k > 100000) {
            snprintf(err, errn, "lm_temperature must be 0 to 2, lm_cfg_scale 0 to 20, lm_top_p 0 to 1 and lm_top_k a whole number"); return 1; }
    } else if (r->lm && *r->lm) { snprintf(err, errn, "lm is for the planner (plan)"); return 1; }
    if (r->ncodes < 0 || r->ncodes > ACG_MAX_CODES || (r->ncodes > 0 && !r->codes)) { snprintf(err, errn, "codes must be 1 to %d audio codes", ACG_MAX_CODES); return 1; }
    if (r->ncodes > 0 && r->task != ACG_TASK_TEXT2MUSIC) { snprintf(err, errn, "audio codes are for text to music (a cover has its clip's own)"); return 1; }
    if (r->task != ACG_TASK_COVER && !(r->ncodes > 0) && (r->cover_strength != 0 || r->cover_noise != 0)) { snprintf(err, errn, "cover_strength and cover_noise are for a cover"); return 1; }
    if (!(r->cover_strength >= 0 && r->cover_strength <= 1) || !(r->cover_noise >= 0 && r->cover_noise <= 1)) { snprintf(err, errn, "cover_strength and cover_noise must be 0 to 1"); return 1; }
    if (r->task != ACG_TASK_REPAINT && r->task != ACG_TASK_LEGO && (r->repaint_start != 0 || r->repaint_end != 0 || r->repaint_mode != 0 || r->repaint_strength != 0)) { snprintf(err, errn, "the repaint settings are for a repaint or a lego"); return 1; }
    if (!(r->guidance >= 0 && r->guidance <= 100)) { snprintf(err, errn, "guidance must be 0 (the default) to 100"); return 1; }
    if (!(r->cfg_start >= 0 && r->cfg_start <= 1) || !(r->cfg_end >= 0 && r->cfg_end <= 1) || (r->cfg_end > 0 && r->cfg_end < r->cfg_start)) { snprintf(err, errn, "the guidance interval must be 0 to 1, from before to after"); return 1; }
    if (r->track && *r->track && r->task != ACG_TASK_EXTRACT && r->task != ACG_TASK_LEGO) { snprintf(err, errn, "a track is for an extract or a lego; a complete takes tracks"); return 1; }
    if (r->tracks && *r->tracks && r->task != ACG_TASK_COMPLETE) { snprintf(err, errn, "tracks are for a complete"); return 1; }
    for (int k = 0; k < 2; k++) {
        const char *s = k ? r->tracks : r->track;
        if (!s) continue;
        if (strlen(s) > 512) { snprintf(err, errn, "the track names are too long"); return 1; }
        for (; *s; s++) if (!(isalnum((unsigned char)*s) || *s == '_' || *s == '-' || *s == ' ' || (k && *s == ','))) { snprintf(err, errn, "a track name has a letter, digit, space, - or _ in it (got '%c')", *s); return 1; }
    }
    if (r->repaint_mode < 0 || r->repaint_mode > ACG_REPAINT_AGGRESSIVE) { snprintf(err, errn, "repaint_mode must be balanced, conservative or aggressive"); return 1; }
    if (!(r->repaint_strength >= 0 && r->repaint_strength <= 1)) { snprintf(err, errn, "repaint_strength must be 0 to 1"); return 1; }
    if (!isfinite(r->repaint_start) || !isfinite(r->repaint_end) || fabs(r->repaint_start) > 3600 || fabs(r->repaint_end) > 3600) { snprintf(err, errn, "the repaint span is out of range"); return 1; }
    return 0;
}

/* What a caller writes as a task's instruction (upstream's generate_instruction). A track is upper-cased; complete joins its tracks with " | ". */
static const char *acg_instruction(int task, const char *track, const char *tracks, char *buf, size_t n){
    static const char *const fixed[3] = { "Fill the audio semantic mask based on the given conditions:",
                                           "Generate audio semantic tokens based on the given conditions:",
                                           "Repaint the mask area based on the given conditions:" };
    if (task <= ACG_TASK_REPAINT) return fixed[task];
    if (task == ACG_TASK_COMPLETE) {
        char list[640]; size_t m = 0; list[0] = 0;
        for (const char *p = tracks ? tracks : ""; *p; ) {
            while (*p == ',' || *p == ' ') p++;
            const char *q = p;
            while (*q && *q != ',') q++;
            const char *r = q;
            while (r > p && r[-1] == ' ') r--;
            if (r > p) {
                if (m) m += (size_t)snprintf(list + m, sizeof list - m, " | ");
                for (const char *c = p; c < r && m + 1 < sizeof list; c++) list[m++] = (char)toupper((unsigned char)*c);
                list[m] = 0;
            }
            p = q;
        }
        if (!m) return "Complete the input track:";
        snprintf(buf, n, "Complete the input track with %s:", list);
        return buf;
    }
    char name[ACG_MAX_TRACK * 2]; size_t m = 0;
    for (const char *p = track ? track : ""; *p && m + 1 < sizeof name; p++) name[m++] = (char)toupper((unsigned char)*p);
    name[m] = 0;
    if (!m) return task == ACG_TASK_EXTRACT ? "Extract the track from the audio:" : "Generate the track based on the audio context:";
    snprintf(buf, n, task == ACG_TASK_EXTRACT ? "Extract the %s track from the audio:" : "Generate the %s track based on the audio context:", name);
    return buf;
}

static double acg_now(void){
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct { const AceHooks *h; const char *stage; int stopped; } AcgTileCtx;
static int acg_tile_hook(int done, int total, void *ud){
    AcgTileCtx *c = (AcgTileCtx *)ud;
    if (c->h && c->h->progress && done < total) c->h->progress(c->h->ud, c->stage, done, total);
    if (c->h && c->h->cancelled && c->h->cancelled(c->h->ud)) { c->stopped = 1; return 1; }
    return 0;
}

/* The silence latent's first n frames, tiled when n is more than the file holds (the handler's _get_silence_latent_slice). */
static void acg_silence_slice(const AceEngine *e, int n, float *out){
    for (int i = 0; i < n; i++) memcpy(out + (size_t)i * 64, e->silence + (size_t)(i % e->silence_frames) * 64, 64 * sizeof(float));
}

/* ------------------------------------------------------------------------------------------- audio in -- */

/* A clip file as upstream's handler reads it: planar stereo at 48 kHz in [-1, 1] ([2][n]), n samples a channel. */
static int acg_load_audio(const char *path, const char *what, float **planar, int64_t *n, int *silent, char *err, size_t errn){
    float *pcm = NULL, *out = NULL; size_t frames = 0, nf = 0; int rate = 0, ch = 0;
    char m[300] = "";
    if (aca_load_ex(path, ACG_MAX_SECONDS, &pcm, &frames, &rate, &ch, m, sizeof m)) { snprintf(err, errn, "the %s clip: %s", what, m); return 1; }
    if (aca_stereo48k_err(pcm, frames, rate, ch, &out, &nf, m, sizeof m)) { free(pcm); snprintf(err, errn, "the %s clip: %s", what, m); return 1; }
    free(pcm);
    int sil = 1;
    for (size_t i = 0; i < 2 * nf && sil; i++) if (fabsf(out[i]) >= 1e-6f) sil = 0;      /* the handler's is_silence */
    if (nf < (size_t)ACG_RATE) { free(out); snprintf(err, errn, "the %s clip is shorter than one second", what); return 1; }
    *planar = out; *n = (int64_t)nf; *silent = sil;
    return 0;
}

/* planar [2][n] -> interleaved [n][2], with `lead` and `tail` samples of silence added */
static float *acg_interleave_padded(const float *planar, int64_t n, int64_t lead, int64_t tail){
    float *x = (float *)calloc((size_t)(n + lead + tail) * 2, sizeof(float));
    if (!x) return NULL;
    for (int64_t i = 0; i < n; i++) { x[(size_t)(lead + i) * 2] = planar[i]; x[(size_t)(lead + i) * 2 + 1] = planar[n + i]; }
    return x;
}

/* The reference clip as upstream builds it (process_reference_audio): a clip shorter than 30 s is repeated to reach it, the whole
 * is cut in thirds and 10 s are taken from each third, joined: [2][30 s]. Upstream picks the start inside each third at random
 * (unseeded, so no two runs agree); this takes the middle of the range, so a seed means one result. A clip of exactly 30 s, or
 * one that is a third-multiple of 10 s, has no range at all and no choice is made. */
static float *acg_reference_30s(const float *planar, int64_t n){
    const int64_t target = (int64_t)ACG_REF_SECONDS * ACG_RATE, segf = (int64_t)ACG_REF_SEGMENT * ACG_RATE;
    int64_t reps = 1;
    if (n < target) reps = (target + n - 1) / n;
    const int64_t total = n * reps, seg = total / 3;
    float *out = (float *)malloc((size_t)target * 2 * sizeof(float));
    if (!out) return NULL;
    const int64_t r0 = seg - segf > 0 ? seg - segf : 0, r2 = (total - 2 * seg) - segf > 0 ? (total - 2 * seg) - segf : 0;
    const int64_t start[3] = { r0 / 2, seg + r0 / 2, 2 * seg + r2 / 2 };
    for (int k = 0; k < 3; k++)
        for (int c = 0; c < 2; c++)
            for (int64_t i = 0; i < segf; i++) out[(size_t)c * target + (size_t)k * segf + i] = planar[(size_t)c * n + (start[k] + i) % n];
    return out;
}

/* One posterior draw of the VAE encoder: torch's randn on one generator, call after call (acestep_vae_enc.h). */
static void acg_vae_eps(float *dst, size_t n, void *ud){ acg_randn_mt((AcgMt *)ud, n, dst); }

/* Encode interleaved [N][2] to latents [F][64] with upstream's 30 s windows and 2 s of context, sampling the posterior from
 * the generator; F frames are written to out (capacity N/1920 + 1). Returns F, -1, or -2 when stopped. */
static int64_t acg_encode(AceEngine *e, const float *wav, int64_t N, AcgMt *mt, const AceHooks *h, float *out, char *err, size_t errn){
    AcvEnc *enc = ace_encoder(e, err, errn);
    if (!enc) return -1;
    AcgTileCtx tc = { h, "encode", 0 };
    int64_t F = acve_encode_tiled(enc, wav, N, ACG_VAE_ENC_CHUNK, ACG_VAE_ENC_OVERLAP, acg_vae_eps, mt, out, acg_tile_hook, &tc);
    if (F == -2 || tc.stopped) { snprintf(err, errn, "cancelled"); return -2; }
    if (F < 1) snprintf(err, errn, "the VAE encoder failed");
    return F;
}

/* The repaint's linspace(a, b, count + 2)[1:-1], element i: torch's own arithmetic (float32, the halfway split). */
static float acg_ramp(float a, float b, int count, int i){
    const int steps = count + 2, idx = i + 1, half = steps / 2;
    const float step = (b - a) / (float)(steps - 1);
    return idx < half ? a + step * (float)idx : b - step * (float)(steps - idx - 1);
}

/* upstream's _repaint_boundary_blend for one clip: x_gen [T][A] <- m * x_gen + (1 - m) * clean, m the span's mask with `cf`
 * frames of ramp on each side. */
static void acg_blend_boundary(float *x, const float *clean, const unsigned char *mask, int T, int A, int cf){
    float *m = (float *)malloc((size_t)T * sizeof(float));
    if (!m) return;
    int left = -1, right = -1;
    for (int t = 0; t < T; t++) { m[t] = mask[t] ? 1.f : 0.f; if (mask[t]) { if (left < 0) left = t; right = t + 1; } }
    if (left >= 0 && !(left == 0 && right == T)) {                      /* a row that is all true, or all false, is left alone */
        const int fs = left - cf > 0 ? left - cf : 0;
        for (int t = fs; t < left; t++) m[t] = acg_ramp(0.f, 1.f, left - fs, t - fs);
        const int fe = right + cf < T ? right + cf : T;
        for (int t = right; t < fe; t++) m[t] = acg_ramp(1.f, 0.f, fe - right, t - right);
    }
    for (int t = 0; t < T; t++)
        for (int c = 0; c < A; c++) x[(size_t)t * A + c] = m[t] * x[(size_t)t * A + c] + (1.0f - m[t]) * clean[(size_t)t * A + c];
    free(m);
}

/* upstream's apply_repaint_waveform_splice for one clip: the source's own samples outside the span (with `cf` samples of
 * crossfade at its edges), the decoded ones inside. wav, src interleaved stereo. */
static void acg_splice_waveform(float *wav, int64_t total, const float *src, int64_t Ns, double start_s, double end_s, double crossfade_s){
    const int64_t mn = total < Ns ? total : Ns;
    int64_t a = (int64_t)(start_s * (double)ACG_RATE), b = (int64_t)(end_s * (double)ACG_RATE);
    a = a < 0 ? 0 : a > mn ? mn : a;
    b = b < a ? a : b > mn ? mn : b;
    if (a == 0 && b >= mn) return;
    const int64_t cf = (int64_t)(crossfade_s * (double)ACG_RATE);
    for (int64_t i = 0; i < mn; i++) {
        float m = (i >= a && i < b) ? 1.f : 0.f;
        if (cf > 0) {
            const int64_t fs = a - cf > 0 ? a - cf : 0, fe = b + cf < mn ? b + cf : mn;
            if (i >= fs && i < a) m = acg_ramp(0.f, 1.f, (int)(a - fs), (int)(i - fs));
            else if (i >= b && i < fe) m = acg_ramp(1.f, 0.f, (int)(fe - b), (int)(i - b));
        }
        for (int c = 0; c < 2; c++) wav[(size_t)i * 2 + c] = m * wav[(size_t)i * 2 + c] + (1.0f - m) * src[(size_t)i * 2 + c];
    }
}

/* The DiT a request runs on: the one it names; else, for extract, lego and complete, the engine's own if that is guided, or the base
 * (the SFT if there is no base) of the checkpoint; else the engine's own. Makes it the active one, loading it the first time. */
static int ace_pick_dit(const AceEngine *e, const AceRequest *rq, char *want, size_t n, char *err, size_t errn){
    if (rq->dit && *rq->dit) {
        if (ace_dit_dir(rq->dit, want, n)) { snprintf(err, errn, "dit must be turbo, base or sft"); return 1; }
    } else if (rq->task >= ACG_TASK_EXTRACT) {
        if (strcmp(e->dit_name, "acestep-v15-turbo")) snprintf(want, n, "%s", e->dit_name);
        else if (ace_have_dit(e, "acestep-v15-base")) snprintf(want, n, "acestep-v15-base");
        else if (ace_have_dit(e, "acestep-v15-sft")) snprintf(want, n, "acestep-v15-sft");
        else { snprintf(err, errn, "extract, lego and complete need the base or SFT DiT, and this checkpoint has only the turbo one (acestep-v15-base is not installed)"); return 1; }
    } else snprintf(want, n, "%s", e->dit_name);
    return 0;
}

static int ace_select_for(AceEngine *e, const AceRequest *rq, char *err, size_t errn){
    char want[64];
    if (ace_pick_dit(e, rq, want, sizeof want, err, errn)) return 1;
    int at = -1;
    for (int i = 0; i < e->nslot; i++) if (!strcmp(e->slot[i].name, want)) at = i;
    if (at < 0) {
        if (!ace_have_dit(e, want)) { snprintf(err, errn, "the DiT %s is not installed in this checkpoint", want); return 1; }
        if (ace_load_slot(e, want, err, errn)) return 1;
        at = e->nslot - 1;
    }
    e->cur = at; ace_alias(e);
    return 0;
}

typedef struct { const AceHooks *h; int target; int stopped; } AcgPlanCtx;
static int acg_plan_step(int step, const float *cond, const float *unc, int token, void *ud){
    AcgPlanCtx *c = (AcgPlanCtx *)ud;
    if (c->h && c->h->progress) c->h->progress(c->h->ud, "plan", step, c->target + 1);
    if (c->h && c->h->cancelled && c->h->cancelled(c->h->ud)) { c->stopped = 1; return 1; }
    return 0;
}

/* The first phase writes an unknown number of tokens (the caption is free): one frame at the start, one at the end, and the stop request is heard at every token. */
typedef struct { const AceHooks *h; int stopped; } AcgMetaCtx;
static int acg_meta_step(void *ud, int step){
    AcgMetaCtx *c = (AcgMetaCtx *)ud;
    if (step == 0 && c->h && c->h->progress) c->h->progress(c->h->ud, "meta", 0, 1);
    if (c->h && c->h->cancelled && c->h->cancelled(c->h->ud)) { c->stopped = 1; return 1; }
    return 0;
}

static int ace_generate(AceEngine *e, const AceRequest *rq, const AceHooks *h, float **pcm, AceInfo *info, char *err, size_t errn);

/* The planner's half of a request: upstream's generate_with_stop_condition. Phase 1, unless all of bpm, key, time signature and duration are given: the 5 Hz
 * model writes the metadata under the constrained decoder, with no guidance. Phase 2: the model writes the codes under a prompt that carries that metadata as
 * YAML. Then the request runs again with them, as a cover of codes, with the metadata the user did not give taken from phase 1 and the caption it rewrote. */
static int ace_plan_and_generate(AceEngine *e, const AceRequest *rq, const AceHooks *h, float **pcm, AceInfo *info, char *err, size_t errn){
    double t0 = acg_now();
    AceLm *lm = ace_planner(e, rq->lm, err, errn);
    if (!lm) return 1;
    const char *lyrics = rq->lyrics;                                        /* what the planner is told, with the handler's words for nothing */
    if (!lyrics) lyrics = "[Instrumental]";
    const int user_bpm = rq->bpm > 0 ? rq->bpm : 0, user_dur = rq->duration > 0 ? (int)rq->duration : 0;
    const int has_key = !act_blank(rq->key), has_ts = !act_blank(rq->timesig);
    const int has_all = user_bpm && has_key && has_ts && user_dur > 0;
    AlmMeta meta; memset(&meta, 0, sizeof meta);
    AceRequest r2 = *rq;
    char *cot = NULL, *pc = NULL, *pu = NULL;
    int *ic = NULL, *iu = NULL, *codes = NULL;
    int rc = 1;
    if (!has_all) {
        char *p1 = alm_meta_prompt(rq->caption, lyrics);
        int np = 0; int *ip = p1 ? alm_tokenize(lm, p1, &np) : NULL;
        if (!ip) { free(p1); snprintf(err, errn, "out of memory"); goto out; }
        AlmAsk ask = { user_bpm, has_key ? rq->key : NULL, has_ts ? rq->timesig : NULL, user_dur, !rq->lm_caption_off, !rq->lm_language_off };
        AlmSample sm = { rq->lm_temperature, 1.0, rq->lm_top_k, rq->lm_top_p, rq->seed ? rq->seed : 1 };   /* no guidance in this phase: upstream turns it off */
        AcgMetaCtx mc = { h, 0 };
        const int got = alm_generate_meta(lm, ip, np, &ask, &sm, &meta, NULL, NULL, acg_meta_step, &mc, NULL, err, errn);
        free(p1); free(ip);
        if (got == -2 || mc.stopped) { rc = 2; snprintf(err, errn, "cancelled"); goto out; }
        if (got < 0) goto out;
        if (h && h->progress) h->progress(h->ud, "meta", 1, 1);
        static const char *const what[AMF_N] = { "tempo", "caption", "duration", "key", "language", "time signature" };
        const int needed[] = { AMF_BPM, AMF_DURATION, AMF_KEYSCALE, AMF_TIMESIG };
        for (int i = 0; i < 4; i++) {
            const int f = needed[i];
            const int given = f == AMF_BPM ? user_bpm > 0 : f == AMF_DURATION ? user_dur > 0 : f == AMF_KEYSCALE ? has_key : has_ts;
            if (!given && !(meta.have[f] && meta.s[f][0])) { snprintf(err, errn, "the planner did not write the song's %s (it wrote %d tokens that do not hold it)", what[f], got); goto out; }
        }
        if (!user_bpm) r2.bpm = atoi(meta.s[AMF_BPM]);
        if (!has_key) r2.key = meta.s[AMF_KEYSCALE];
        if (!has_ts) r2.timesig = meta.s[AMF_TIMESIG];
        if (!user_dur) r2.duration = (double)atoi(meta.s[AMF_DURATION]);
        if (!rq->lm_caption_off && meta.have[AMF_CAPTION]) r2.caption = meta.s[AMF_CAPTION];             /* upstream: use_cot_caption puts the model's caption in the DiT's prompt */
        cot = ame_cot(&meta);
        info->meta_run = 1;
        info->meta_bpm = r2.bpm; info->meta_duration = r2.duration;
        snprintf(info->meta_key, sizeof info->meta_key, "%s", r2.key ? r2.key : "");
        snprintf(info->meta_timesig, sizeof info->meta_timesig, "%s", r2.timesig ? r2.timesig : "");
        snprintf(info->meta_language, sizeof info->meta_language, "%s", meta.have[AMF_LANGUAGE] ? meta.s[AMF_LANGUAGE] : "");
        snprintf(info->meta_caption, sizeof info->meta_caption, "%s", meta.have[AMF_CAPTION] ? meta.s[AMF_CAPTION] : "");
    } else {
        cot = alm_cot(rq->bpm, rq->duration, rq->key, NULL, rq->timesig);                                /* the metadata as given: no language line, as upstream's has none */
    }
    const int target = (int)(r2.duration * 5);                                          /* upstream: int(duration * 5) codes, a second being 5 */
    if (target < 5 || target > ACG_MAX_CODES) { snprintf(err, errn, "the duration asks for %d codes: 1 to %d seconds", target, ACG_MAX_CODES / 5); goto out; }
    pc = cot ? alm_prompt(rq->caption, lyrics, cot, 0) : NULL; pu = cot ? alm_prompt(rq->caption, lyrics, cot, 1) : NULL;
    int nc = 0, nu = 0;
    ic = pc ? alm_tokenize(lm, pc, &nc) : NULL; iu = pu ? alm_tokenize(lm, pu, &nu) : NULL;
    codes = (int *)malloc((size_t)target * sizeof(int));
    if (!ic || !iu || !codes) { snprintf(err, errn, "out of memory"); goto out; }
    {
        AlmSample sm = { rq->lm_temperature, rq->lm_cfg_scale, rq->lm_top_k, rq->lm_top_p, rq->seed ? rq->seed : 1 };
        AcgPlanCtx ctx = { h, target, 0 };
        int got = alm_generate_codes(lm, ic, nc, iu, nu, target, &sm, codes, acg_plan_step, &ctx, err, errn);
        if (got == -2 || ctx.stopped) { rc = 2; snprintf(err, errn, "cancelled"); goto out; }
        if (got < 1) { if (got == 0) snprintf(err, errn, "the planner wrote no codes"); goto out; }
        r2.plan = 0; r2.lm = NULL; r2.codes = codes; r2.ncodes = got;
        const double t_plan = acg_now() - t0;
        AceInfo keep = *info;
        rc = ace_generate(e, &r2, h, pcm, info, err, errn);
        snprintf(info->lm, sizeof info->lm, "%s", lm->dir_name);
        info->plan_codes = got; info->t_plan = t_plan;
        info->meta_run = keep.meta_run; info->meta_bpm = keep.meta_bpm; info->meta_duration = keep.meta_duration;
        memcpy(info->meta_key, keep.meta_key, sizeof info->meta_key); memcpy(info->meta_timesig, keep.meta_timesig, sizeof info->meta_timesig);
        memcpy(info->meta_language, keep.meta_language, sizeof info->meta_language); memcpy(info->meta_caption, keep.meta_caption, sizeof info->meta_caption);
    }
out:
    alm_meta_free(&meta);
    free(cot); free(pc); free(pu); free(ic); free(iu); free(codes);
    return rc;
}

/* Make the music. Returns 0 and the interleaved stereo waveform in [-1, 1] (malloc'd, *pcm, info->samples frames), 1 with a
 * reason in err, or 2 when the hooks asked to stop. */
static int ace_generate(AceEngine *e, const AceRequest *rq, const AceHooks *h, float **pcm, AceInfo *info, char *err, size_t errn){
    memset(info, 0, sizeof *info);
    *pcm = NULL;
    if (ace_check(rq, err, errn)) return 1;
    if (rq->plan) return ace_plan_and_generate(e, rq, h, pcm, info, err, errn);
    if (ace_select_for(e, rq, err, errn)) return 1;
    const AceDit *d = e->dit;
    snprintf(info->dit, sizeof info->dit, "%s", e->slot[e->cur].name);
    /* text to music with the planner's codes is what the handler turns into a cover (_resolve_generate_music_task): the cover instruction, and the
     * codes' own hints as the context where a cover's source would be heard */
    const int from_codes = rq->ncodes > 0 && rq->task == ACG_TASK_TEXT2MUSIC;
    const int A = d->acdim, task = from_codes ? ACG_TASK_COVER : rq->task;
    const int repaint_like = task == ACG_TASK_REPAINT || task == ACG_TASK_LEGO;       /* a span is generated, the rest is the source put back */
    const int text_like = task == ACG_TASK_TEXT2MUSIC || (task == ACG_TASK_COMPLETE && !(rq->src_path && *rq->src_path));   /* no source: silence is the context */
    if (d->turbo && task >= ACG_TASK_EXTRACT) { snprintf(err, errn, "extract, lego and complete need the base or SFT DiT (acestep-v15-base): this engine runs the turbo one"); return 1; }
    if (d->turbo && rq->steps > ACG_MAX_STEPS_TURBO) { snprintf(err, errn, "the turbo DiT takes at most %d steps (a base or SFT one takes up to %d)", ACG_MAX_STEPS_TURBO, ACG_MAX_STEPS); return 1; }
    const double guidance = rq->guidance > 0 ? rq->guidance : 7.0;
    const int use_cfg = !d->turbo && guidance > 1.0;                                   /* the turbo has guidance baked in and upstream forces it to 1 */
    info->guidance = use_cfg ? guidance : 0;
    const double cfg_s = rq->cfg_start, cfg_e = rq->cfg_end > 0 ? rq->cfg_end : 1.0;
    int rc = 1;
    char *cp = NULL, *cp2 = NULL, *lp = NULL;
    float *text = NULL, *text2 = NULL, *lyric = NULL, *timbre = NULL, *lat = NULL, *src = NULL, *ctx = NULL, *ctx2 = NULL, *noise = NULL;
    float *x = NULL, *nx = NULL, *v = NULL, *wav = NULL, *srcpcm = NULL, *planar = NULL, *refp = NULL, *refpcm = NULL;
    float *vu = NULL, *vg = NULL, *mom = NULL;
    unsigned char *rmask = NULL;
    AcdCond *cond = NULL, *cond2 = NULL, *ncond = NULL, *ncond2 = NULL;
    int *lids = NULL;
    #define ACG_STOP() do { if (h && h->cancelled && h->cancelled(h->ud)) { rc = 2; snprintf(err, errn, "cancelled"); goto out; } } while (0)
    #define ACG_PROGRESS(stage, a, b) do { if (h && h->progress) h->progress(h->ud, stage, a, b); } while (0)
    #define ACG_TAP(name, p, n) do { if (h && h->tap) h->tap(h->ud, name, p, n); } while (0)

    /* 1. the clip's length, and the audio that comes in: the source (cover, repaint) and the reference */
    double t0 = acg_now();
    int T, want_samples = 0, lat_frames = 0, span_a = 0, span_b = 0;
    int64_t Ns = 0;                                    /* samples of the source clip as it is run on: with a repaint's silence added */
    double meta_duration = rq->duration, sp_start = 0, sp_end = 0;
    AcgMt vae_mt; acg_mt_seed(&vae_mt, (uint32_t)rq->seed ^ ACG_VAE_SEED_XOR);
    int src_silent = 0;
    if (text_like) acg_clip(rq->duration, &T, &want_samples);
    else if (from_codes) {                                                     /* the codes' length is the song's: 5 frames a code, 128 at least */
        lat_frames = rq->ncodes * ACG_CODE_FRAMES;
        T = lat_frames < ACG_MIN_FRAMES ? ACG_MIN_FRAMES : lat_frames;
        want_samples = lat_frames * ACG_HOP;
    } else {
        int64_t n = 0;
        ACG_PROGRESS("encode", 0, 1);
        if (acg_load_audio(rq->src_path, "source", &planar, &n, &src_silent, err, errn)) goto out;
        info->src_seconds = (double)n / ACG_RATE;
        meta_duration = info->src_seconds;             /* the prompt's duration is the source's, whatever duration was asked for (batch_prep: a source wins) */
        int64_t lead = 0, tail = 0;
        if (repaint_like) {
            const double sd = (double)n / ACG_RATE;
            const double end = rq->repaint_end > 0 ? rq->repaint_end : sd;
            const double left = rq->repaint_start < 0 ? -rq->repaint_start : 0, right = end - sd > 0 ? end - sd : 0;
            lead = (int64_t)(left * ACG_RATE); tail = (int64_t)(right * ACG_RATE);
            sp_start = rq->repaint_start + left; sp_end = end + left;
            if (!(end > rq->repaint_start)) { snprintf(err, errn, "the repaint must end after it starts (%g to %g s)", rq->repaint_start, end); goto out; }
            if (rq->repaint_start >= sd) { snprintf(err, errn, "the repaint starts at %g s, past the end of the %.2f s clip: to add music after it, end the repaint later and start it inside the clip", rq->repaint_start, sd); goto out; }
            if ((double)(n + lead + tail) / ACG_RATE > ACG_MAX_SECONDS) { snprintf(err, errn, "the repainted clip would be longer than %g seconds", ACG_MAX_SECONDS); goto out; }
        }
        Ns = n + lead + tail;
        lat_frames = (int)(Ns / ACG_HOP);
        T = lat_frames < ACG_MIN_FRAMES ? ACG_MIN_FRAMES : lat_frames;
        want_samples = (int)Ns;
        if (repaint_like) {
            span_a = (int)acg_floordiv(sp_start * ACG_RATE, ACG_HOP); span_b = (int)acg_floordiv(sp_end * ACG_RATE, ACG_HOP);
            span_a = span_a < 0 ? 0 : span_a > T - 1 ? T - 1 : span_a;
            span_b = span_b < span_a + 1 ? span_a + 1 : span_b > T ? T : span_b;
        }
        srcpcm = acg_interleave_padded(planar, n, lead, tail);
        free(planar); planar = NULL;
        if (!srcpcm) { snprintf(err, errn, "out of memory"); goto out; }
    }
    info->frames = T;

    lat = (float *)malloc((size_t)T * A * sizeof(float));
    if (!lat) { snprintf(err, errn, "out of memory"); goto out; }
    if (from_codes) acg_silence_slice(e, T, lat);                              /* replaced by the codes' hints below, once the detokenizer is loaded */
    else if (!text_like) {
        /* the source's latents: the VAE encoder's posterior sample (a silent clip is the silence latent, as upstream treats it),
         * then silence to the floor of 128 frames */
        if (src_silent) acg_silence_slice(e, lat_frames, lat);
        else {
            int64_t F = acg_encode(e, srcpcm, Ns, &vae_mt, h, lat, err, errn);
            if (F == -2) { rc = 2; goto out; }
            if (F != lat_frames) { if (F > 0) snprintf(err, errn, "the VAE encoder returned %lld frames for %d", (long long)F, lat_frames); goto out; }
        }
        if (lat_frames < T) acg_silence_slice(e, T - lat_frames, lat + (size_t)lat_frames * A);      /* the handler pads with the silence latent's first frames */
        if (h && h->feed) { const float *o = h->feed(h->ud, "source_latents", (size_t)T * A); if (o) memcpy(lat, o, (size_t)T * A * sizeof(float)); }
    }
    /* the timbre reference: the first 750 frames of the silence latent, or the reference clip's latents */
    timbre = (float *)malloc((size_t)ACG_TIMBRE_FRAMES * A * sizeof(float));
    if (!timbre) { snprintf(err, errn, "out of memory"); goto out; }
    if (rq->ref_path && *rq->ref_path) {
        int64_t n = 0, F; int sil = 0;
        ACG_PROGRESS("encode", 0, 1);
        if (acg_load_audio(rq->ref_path, "reference", &planar, &n, &sil, err, errn)) goto out;
        if (sil) { snprintf(err, errn, "the reference clip is silent"); goto out; }
        refp = acg_reference_30s(planar, n);
        refpcm = refp ? acg_interleave_padded(refp, (int64_t)ACG_REF_SECONDS * ACG_RATE, 0, 0) : NULL;
        free(planar); planar = NULL;
        if (!refpcm) { snprintf(err, errn, "out of memory"); goto out; }
        float *rl = (float *)malloc(((size_t)ACG_REF_SECONDS * ACG_RATE / ACG_HOP + 2) * A * sizeof(float));
        if (!rl) { snprintf(err, errn, "out of memory"); goto out; }
        F = acg_encode(e, refpcm, (int64_t)ACG_REF_SECONDS * ACG_RATE, &vae_mt, h, rl, err, errn);
        if (F == -2) { rc = 2; free(rl); goto out; }
        if (F != ACG_TIMBRE_FRAMES) { if (F > 0) snprintf(err, errn, "the reference encoded to %lld frames, not %d", (long long)F, ACG_TIMBRE_FRAMES); free(rl); goto out; }
        memcpy(timbre, rl, (size_t)ACG_TIMBRE_FRAMES * A * sizeof(float));
        free(rl);
    } else acg_silence_slice(e, ACG_TIMBRE_FRAMES, timbre);
    ACG_PROGRESS("encode", 1, 1);
    info->t_encode = acg_now() - t0;
    ACG_STOP();

    /* 2. the prompts, ids and the text encoder */
    t0 = acg_now();
    ACG_PROGRESS("text", 0, 1);
    static const char *const instructions[3] = { "Fill the audio semantic mask based on the given conditions:",
                                                  "Repaint the mask area based on the given conditions:",
                                                  "Generate audio semantic tokens based on the given conditions:" };
    char instr_buf[1024];
    const char *instruction = acg_instruction(task, rq->track, rq->tracks, instr_buf, sizeof instr_buf);
    const char *lyrics = rq->lyrics ? rq->lyrics : "[Instrumental]";      /* as given, even if empty (the handler does); only none at all is instrumental */
    char bpm[16] = ""; if (rq->bpm > 0) snprintf(bpm, sizeof bpm, "%d", rq->bpm);
    cp = act_caption_prompt(instruction, rq->caption, bpm, rq->key, rq->timesig, meta_duration);
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
    const double cover_strength = rq->cover_strength > 0 ? rq->cover_strength : 1.0;
    int cn2 = 0;
    if (task == ACG_TASK_COVER && cover_strength < 1.0) {
        /* the steps after the switch run on the text-to-music prompt (the default instruction) and the silence as the source */
        int cids2[ACT_CAPTION_MAX];
        cp2 = act_caption_prompt(instructions[0], rq->caption, bpm, rq->key, rq->timesig, meta_duration);
        cn2 = act_tokenize(&e->tok, cp2, ACT_CAPTION_MAX, cids2);
        /* Upstream pads this prompt's ids to the length of the cover prompt's (_pad_sequences). With Qwen's tokenizer the default
         * instruction is a token longer than the cover one, and torch's pad with a negative amount CUTS: the last token (the end
         * of text the tokenizer appends) is dropped. The encoder is causal, so the first rows of the whole prompt's states are
         * the cut one's. A tokenizer that splits the two instructions the other way round (the tiny test fixture's does) gets
         * the pad token after the text instead, and those rows are not masked: the DiT builds its cross-attention mask from the
         * AUDIO's mask, never the text's, so the padding takes part in attention like any other row of the condition. */
        for (int i = cn2; i < cn; i++) cids2[i] = e->tok.eot;
        const int enc2 = cn2 < cn ? cn : cn2;
        text2 = act_encode(e->text, cids2, enc2);
        if (!text2) { snprintf(err, errn, "the text encoder failed"); goto out; }
        cn2 = enc2 > cn ? cn : enc2;
    }
    info->t_text = acg_now() - t0;
    ACG_STOP();

    /* 3. the condition: lyric, timbre and text */
    t0 = acg_now();
    ACG_PROGRESS("condition", 0, 1);
    cond = acd_cond_new(d, text, cn, lyric, ln, timbre, ACG_TIMBRE_FRAMES);
    if (!cond) { snprintf(err, errn, "the condition encoders failed"); goto out; }
    if (text2 && !(cond2 = acd_cond_new(d, text2, cn2, lyric, ln, timbre, ACG_TIMBRE_FRAMES))) { snprintf(err, errn, "the condition encoders failed"); goto out; }
    if (use_cfg) {                                                                     /* the other branch of guidance: the null condition, as long as each one */
        ncond = acd_cond_null(d, cond);
        if (cond2) ncond2 = acd_cond_null(d, cond2);
        if (!ncond || (cond2 && !ncond2)) { snprintf(err, errn, "the null condition failed"); goto out; }
    }
    ACG_TAP("timbre", timbre, (size_t)ACG_TIMBRE_FRAMES * A);
    ACG_TAP("encoder_hidden", cond->enc, (size_t)cond->len * d->D);
    if (cond2) ACG_TAP("noncover_hidden", cond2->enc, (size_t)cond2->len * d->D);

    /* 4. the context the DiT sees: the source it is conditioned on, and the mask (all true in every task). Text to music: the
     * silence latent. A cover: the source through the tokenizer and detokenizer (what survives of it). A repaint: the source with
     * the span's frames replaced by the silence latent. */
    src = (float *)malloc((size_t)T * A * sizeof(float));
    ctx = (float *)malloc((size_t)T * 2 * A * sizeof(float));
    noise = (float *)malloc((size_t)T * A * sizeof(float));
    x = (float *)malloc((size_t)T * A * sizeof(float)); nx = (float *)malloc((size_t)T * A * sizeof(float)); v = (float *)malloc((size_t)T * A * sizeof(float));
    rmask = (unsigned char *)calloc((size_t)T, 1);
    if (!src || !ctx || !noise || !x || !nx || !v || !rmask) { snprintf(err, errn, "out of memory"); goto out; }
    if (text_like) acg_silence_slice(e, T, src);
    else memcpy(src, lat, (size_t)T * A * sizeof(float));
    float *dit_src = src;                                              /* what goes in the context as the source half */
    float *hints = NULL;
    if (task == ACG_TASK_COVER) {
        AcdTok *net = ace_cover_net(e, err, errn);
        if (!net) goto out;
        hints = (float *)malloc((size_t)T * A * sizeof(float));
        if (!hints) { snprintf(err, errn, "out of memory"); goto out; }
        if (from_codes) {
            if (net->P != ACG_CODE_FRAMES) { free(hints); snprintf(err, errn, "the audio tokenizer pools %d frames, a code is %d", net->P, ACG_CODE_FRAMES); goto out; }
            if (atk_hints_from_codes(d, net, rq->codes, rq->ncodes, T, e->silence, hints)) { free(hints); snprintf(err, errn, "the audio detokenizer failed"); goto out; }
            memcpy(lat, hints, (size_t)T * A * sizeof(float));                 /* the handler's target latents are the decoded codes (what a cover-noise start renoises) */
            memcpy(src, hints, (size_t)T * A * sizeof(float));
        } else if (atk_hints(d, net, src, T, e->silence, hints)) { free(hints); snprintf(err, errn, "the audio tokenizer failed"); goto out; }
        dit_src = hints;
    } else if (repaint_like) {
        /* a repaint hides the span from the DiT (silence in its place); a lego shows it: the track is added to what is there */
        for (int f = span_a; f < span_b; f++) {
            if (task == ACG_TASK_REPAINT) memcpy(src + (size_t)f * A, e->silence + (size_t)(f % e->silence_frames) * 64, 64 * sizeof(float));
            rmask[f] = 1;
        }
    } else memset(rmask, 1, (size_t)T);
    ACG_TAP("src_latents", src, (size_t)T * A);
    if (repaint_like) ACG_TAP("clean_src_latents", lat, (size_t)T * A);
    for (int f = 0; f < T; f++) {
        memcpy(ctx + (size_t)f * 2 * A, dit_src + (size_t)f * A, (size_t)A * sizeof(float));
        for (int c = 0; c < A; c++) ctx[(size_t)f * 2 * A + A + c] = 1.0f;      /* all true, repaint too: chunk_mask_mode "auto" sets every entry (the span is told only by the source's silence and the injection) */
    }
    free(hints);
    ACG_TAP("context", ctx, (size_t)T * 2 * A);
    if (cond2) {
        ctx2 = (float *)malloc((size_t)T * 2 * A * sizeof(float));
        if (!ctx2) { snprintf(err, errn, "out of memory"); goto out; }
        float *sil = (float *)malloc((size_t)T * A * sizeof(float));
        if (!sil) { snprintf(err, errn, "out of memory"); goto out; }
        acg_silence_slice(e, T, sil);
        for (int f = 0; f < T; f++) {
            memcpy(ctx2 + (size_t)f * 2 * A, sil + (size_t)f * A, (size_t)A * sizeof(float));
            for (int c = 0; c < A; c++) ctx2[(size_t)f * 2 * A + A + c] = 1.0f;
        }
        free(sil);
        ACG_TAP("noncover_context", ctx2, (size_t)T * 2 * A);
    }
    info->t_cond = acg_now() - t0;
    acg_randn((uint32_t)rq->seed, (size_t)T * A, noise);
    ACG_TAP("noise", noise, (size_t)T * A);
    ACG_STOP();

    /* 5. the Euler steps. acd_sample is this loop; it is spelled out here to report each step, to stop between them, and for what
     * the audio tasks add: a cover that starts from the renoised source, a cover whose last steps drop the source's context,
     * and a repaint that puts the (renoised) source back outside its span while the first steps run. */
    float sched_all[ACG_MAX_STEPS + 1];
    const int n_all = d->turbo ? acd_schedule(rq->steps, (float)rq->shift, sched_all) : acd_schedule_linspace(rq->steps, (float)rq->shift, sched_all);
    if (n_all < 1) { snprintf(err, errn, "no schedule for %d steps", rq->steps); goto out; }
    const float *sched = sched_all;
    int n = n_all;
    AcdSampleOpts so = { rq->dcw ? 1 : 0, 0.05, 0.02 };
    memcpy(x, noise, (size_t)T * A * sizeof(float));
    if (task == ACG_TASK_COVER && rq->cover_noise > 0) {
        /* start where the schedule is nearest 1 - cover_noise, from the source renoised to that level (the first nearest wins a tie) */
        const double level = 1.0 - rq->cover_noise;
        int best = 0;
        for (int i = 1; i < n_all; i++) if (fabs((double)sched_all[i] - level) < fabs((double)sched_all[best] - level)) best = i;
        const double tn = sched_all[best];
        const float a = (float)tn, b = (float)(1.0 - tn);
        for (size_t k = 0; k < (size_t)T * A; k++) x[k] = a * noise[k] + b * lat[k];
        sched = sched_all + best; n = n_all - best;
    }
    const int cover_steps = cond2 ? (int)(n * cover_strength) : n;                    /* int(num_steps * audio_cover_strength) */
    double inject = 0, wav_cf = 0; int cf_frames = 0;
    if (repaint_like) acg_repaint_config(rq->repaint_mode, rq->repaint_strength, &inject, &cf_frames, &wav_cf);
    const int cutoff = (int)acg_pyround(inject * n);
    AcdCond *cur = cond, *ncur = ncond; const float *cur_ctx = ctx;
    if (use_cfg) {
        vu = (float *)malloc((size_t)T * A * sizeof(float)); vg = (float *)malloc((size_t)T * A * sizeof(float)); mom = (float *)calloc((size_t)T * A, sizeof(float));
        if (!vu || !vg || !mom) { snprintf(err, errn, "out of memory"); goto out; }
    }
    t0 = acg_now();
    for (int i = 0; i < n; i++) {
        ACG_PROGRESS("denoise", i, n);
        ACG_STOP();
        if (cond2 && i >= cover_steps) { cur = cond2; ncur = ncond2; cur_ctx = ctx2; }
        acd_forward(d, cur, x, cur_ctx, T, sched[i], sched[i], v);
        if (use_cfg && sched[i] >= cfg_s && sched[i] <= cfg_e) {                       /* upstream: apply_cfg_guidance = t_curr >= start and t_curr <= end */
            acd_forward(d, ncur, x, cur_ctx, T, sched[i], sched[i], vu);
            if (rq->adg) acd_adg(vg, x, v, vu, T, A, sched[i], guidance);
            else acd_apg(vg, v, vu, T, A, guidance, mom);
            memcpy(v, vg, (size_t)T * A * sizeof(float));
        }
        acd_sample_step(x, v, i, n, sched, T, A, &so, nx);
        float *s = x; x = nx; nx = s;
        if (repaint_like && i < n - 1 && i < cutoff) {                    /* outside the span the source, noised to the level the next step expects */
            const float tn = sched[i + 1], a = tn, b = (float)(1.0 - (double)tn);
            for (int f = 0; f < T; f++)
                if (!rmask[f])
                    for (int c = 0; c < A; c++) x[(size_t)f * A + c] = a * noise[(size_t)f * A + c] + b * lat[(size_t)f * A + c];
        }
    }
    ACG_PROGRESS("denoise", n, n);
    if (repaint_like && cf_frames > 0) acg_blend_boundary(x, lat, rmask, T, A, cf_frames);
    for (size_t k = 0; k < (size_t)T * A; k++) if (!isfinite(x[k])) { snprintf(err, errn, "the sampler produced a non-finite latent"); goto out; }
    ACG_TAP("latents", x, (size_t)T * A);
    info->t_denoise = acg_now() - t0;
    ACG_STOP();

    /* 6. the VAE, tiled as upstream tiles it, then the peak limit, then (a repaint) the source's own samples put back outside the span */
    t0 = acg_now();
    const size_t total = (size_t)T * ACG_HOP;
    wav = (float *)malloc(total * e->vae->ach * sizeof(float));
    if (!wav) { snprintf(err, errn, "out of memory"); goto out; }
    AcgTileCtx tc = { h, "decode", 0 };
    if (acv_decode_tiled(e->vae, x, T, ACG_VAE_CHUNK, ACG_VAE_OVERLAP, wav, acg_tile_hook, &tc)) {
        if (tc.stopped) { rc = 2; snprintf(err, errn, "cancelled"); } else snprintf(err, errn, "the VAE decode failed");
        goto out;
    }
    info->peak = acv_normalize(wav, total * e->vae->ach);
    if (repaint_like && rq->repaint_mode != ACG_REPAINT_AGGRESSIVE) acg_splice_waveform(wav, (int64_t)total, srcpcm, Ns, sp_start, sp_end, wav_cf);
    info->t_decode = acg_now() - t0;
    info->samples = (int)total < want_samples ? (int)total : want_samples;
    *pcm = wav; wav = NULL;
    rc = 0;
out:
    free(cp); free(cp2); free(lp); free(lids); free(text); free(text2); free(lyric); free(timbre); free(lat); free(src); free(ctx); free(ctx2);
    free(noise); free(x); free(nx); free(v); free(wav); free(srcpcm); free(planar); free(refp); free(refpcm); free(rmask);
    free(vu); free(vg); free(mom);
    acd_cond_free(cond); acd_cond_free(cond2); acd_cond_free(ncond); acd_cond_free(ncond2);
    #undef ACG_STOP
    #undef ACG_PROGRESS
    #undef ACG_TAP
    return rc;
}

#endif
