/* acestep_vae.h -- the ACE-Step 1.5 audio VAE decoder (diffusers AutoencoderOobleck), pure C on top of qi_gemm.
 *
 * Latents [T][64] at 25 Hz in, 48 kHz stereo [T*1920][2] out. The decoder is the Stable Audio "Oobleck" one:
 *   conv1 (64 -> 16*128 channels, k=7)
 *   five blocks, strides 10, 6, 4, 4, 2 (the encoder's downsampling ratios reversed), each
 *       Snake -> ConvTranspose1d (k = 2*stride, padding ceil(stride/2): the length is exactly stride x)
 *       -> three residual units (Snake -> Conv1d k=7 dilation 1/3/9 -> Snake -> Conv1d k=1, plus the input)
 *     whose widths go 2048 -> 1024 -> 512 -> 256 -> 128 -> 128
 *   Snake -> conv2 (128 -> 2, k=7, no bias)
 * Snake is x + sin^2(exp(alpha) x) / (exp(beta) + 1e-9) per channel (diffusers' logscale=True).
 *
 * Every convolution is weight-normalised in the checkpoint (weight_g, weight_v). The fold, w = g * v / ||v||
 * with the norm over everything but dimension 0, is done at load in f32 and the folded weights stay f32:
 * folding to bf16 would move every weight by up to 0.4%, which is not the model any more. Dimension 0 is the
 * output channel of a Conv1d and the INPUT channel of a ConvTranspose1d (weight_v is [in][out][k] there).
 * Every size comes from config.json and the tensor shapes (checked), none is assumed.
 *
 * Activations are channel-last [T][C]. A convolution is one qi_gemm per band of output rows over an im2col of
 * k*Cin columns, the weight permuted at load to [Cout][k][Cin]; a transposed convolution is one qi_gemm to
 * [T][2s*Cout] and a gather-add over the output rows (each output row takes at most two taps, so no two threads
 * ever write one row). The im2col and the product band are bounded (acv_col_floats), so a long clip's cost is
 * time, not memory: the activations of the last block (T*1920 rows x 128) are what the working set is.
 *
 * Long clips are decoded the way upstream's handler does it, in overlapping windows with the overlap discarded
 * (acv_decode_tiled): the windows, and so the numbers, are upstream's.
 *
 * Interface: acv_load(vae_dir) / acv_decode(v, z, T, out) / acv_decode_tiled(...) / acv_free(v).
 */
#ifndef COLIBRI_ACESTEP_VAE_H
#define COLIBRI_ACESTEP_VAE_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "st.h"
#include "json.h"
#include "qi_gemm.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define ACV_MAX_BLOCKS 8
#define ACV_MAX_CHANNELS 8192
#define ACV_MAX_STRIDE 64
/* Floats for the im2col band and for the transposed convolution's product band (64 MB each). Variables so a
 * test can shrink them to a few rows and cover the band edges on a tiny model. */
static size_t acv_col_floats = (size_t)16 << 20;
static size_t acv_z_floats = (size_t)16 << 20;

typedef struct { int cin, cout, k, dil; float *w, *b; QiMat m; } AcvConv;          /* w [cout][k][cin] */
typedef struct { int cin, cout, s, pad; float *w, *b; QiMat m; } AcvConvT;         /* w [2s][cout][cin] */
typedef struct { int c; float *ea, *ib; } AcvSnake;                                /* exp(alpha), 1/(exp(beta)+1e-9) */
typedef struct { AcvSnake s1, s2; AcvConv c1, c2; } AcvRes;
typedef struct { int cin, cout, s; AcvSnake sn; AcvConvT up; AcvRes r[3]; } AcvBlock;

typedef struct {
    int zdim, ach, nblk, hop, rate;
    AcvConv conv1, conv2;
    AcvBlock blk[ACV_MAX_BLOCKS];
    AcvSnake sn;
    size_t wbytes;
} AcVae;

/* Scratch for one decode of up to maxT frames: three activation buffers, the im2col band, the product band. */
typedef struct { float *buf[3], *col, *zb; size_t nbuf, ncol, nzb; int maxT; } AcvWork;

/* ----------------------------------------------------------------- loading -- */

static float *acv_alloc(size_t n){
    float *p = (float *)malloc(n ? n * sizeof(float) : 1);
    if (!p) fprintf(stderr, "acestep vae: out of memory (%.1f MB)\n", n * 4.0 / 1e6);
    return p;
}

/* A float tensor of the given shape, widened to f32. */
static float *acv_read(shards *S, const char *name, int rank, const int64_t *shape, char *err, size_t errn){
    st_tensor *t = st_find(S, name);
    if (!t) { snprintf(err, errn, "missing tensor %s", name); return NULL; }
    if (t->dtype > 2) { snprintf(err, errn, "tensor %s is %s, not a float tensor", name, st_dtype_name(t->dtype)); return NULL; }
    int64_t numel = 1;
    for (int i = 0; i < rank; i++) numel *= shape[i];
    int ok = t->rank == rank && t->numel == numel;
    for (int i = 0; ok && i < rank; i++) ok = t->shape[i] == shape[i];
    if (!ok) {
        char got[96] = "", want[96] = "";
        for (int i = 0; i < t->rank && i < ST_MAX_RANK; i++) snprintf(got + strlen(got), sizeof got - strlen(got), "%s%lld", i ? "x" : "", (long long)t->shape[i]);
        for (int i = 0; i < rank; i++) snprintf(want + strlen(want), sizeof want - strlen(want), "%s%lld", i ? "x" : "", (long long)shape[i]);
        snprintf(err, errn, "tensor %s is %s, expected %s", name, got, want);
        return NULL;
    }
    float *p = acv_alloc((size_t)numel);
    if (!p) { snprintf(err, errn, "out of memory reading %s", name); return NULL; }
    st_read_f32_cap(S, name, p, numel, 1);
    return p;
}

static void acv_free_conv(AcvConv *c){ free(c->w); free(c->b); memset(c, 0, sizeof *c); }
static void acv_free_convt(AcvConvT *c){ free(c->w); free(c->b); memset(c, 0, sizeof *c); }
static void acv_free_snake(AcvSnake *s){ free(s->ea); free(s->ib); memset(s, 0, sizeof *s); }

static int acv_load_snake(shards *S, const char *pfx, int c, AcvSnake *s, char *err, size_t errn){
    char n[192]; int64_t sh[3] = {1, c, 1};
    snprintf(n, sizeof n, "%s.alpha", pfx); float *a = acv_read(S, n, 3, sh, err, errn);
    snprintf(n, sizeof n, "%s.beta", pfx);  float *b = a ? acv_read(S, n, 3, sh, err, errn) : NULL;
    if (!a || !b) { free(a); free(b); return -1; }
    for (int i = 0; i < c; i++) { a[i] = expf(a[i]); b[i] = 1.0f / (expf(b[i]) + 1e-9f); }
    s->c = c; s->ea = a; s->ib = b;
    return 0;
}

/* w = g * v / ||v||: the scale of slice i of dimension 0 is g[i] over the norm of everything else in the slice. */
static void acv_fold_scales(const float *v, const float *g, int64_t n0, int64_t rest, float *scale){
    for (int64_t i = 0; i < n0; i++) {
        double s = 0;
        for (int64_t j = 0; j < rest; j++) s += (double)v[i * rest + j] * v[i * rest + j];
        scale[i] = g[i] / (float)sqrt(s);
    }
}

/* Conv1d: weight_v [cout][cin][k], weight_g [cout][1][1], optional bias [cout]. */
static int acv_load_conv(shards *S, const char *pfx, int cin, int cout, int k, int dil, int bias, AcvConv *c, char *err, size_t errn){
    char n[192];
    int64_t shv[3] = {cout, cin, k}, shg[3] = {cout, 1, 1}, shb[1] = {cout};
    snprintf(n, sizeof n, "%s.weight_v", pfx); float *v = acv_read(S, n, 3, shv, err, errn);
    snprintf(n, sizeof n, "%s.weight_g", pfx); float *g = v ? acv_read(S, n, 3, shg, err, errn) : NULL;
    float *b = NULL, *w = NULL, *scale = NULL;
    if (!v || !g) goto fail;
    if (bias) { snprintf(n, sizeof n, "%s.bias", pfx); if (!(b = acv_read(S, n, 1, shb, err, errn))) goto fail; }
    w = acv_alloc((size_t)cout * k * cin); scale = acv_alloc((size_t)cout);
    if (!w || !scale) { snprintf(err, errn, "out of memory"); goto fail; }
    acv_fold_scales(v, g, cout, (int64_t)cin * k, scale);
    for (int co = 0; co < cout; co++)
        for (int ci = 0; ci < cin; ci++)
            for (int j = 0; j < k; j++)
                w[((size_t)co * k + j) * cin + ci] = v[((size_t)co * cin + ci) * k + j] * scale[co];
    free(v); free(g); free(scale);
    c->cin = cin; c->cout = cout; c->k = k; c->dil = dil; c->w = w; c->b = b;
    c->m = (QiMat){QI_F32, cout, k * cin, w, NULL, 0};
    return 0;
fail:
    free(v); free(g); free(b); free(w); free(scale);
    return -1;
}

/* ConvTranspose1d: weight_v [cin][cout][k=2s], weight_g [cin][1][1] (dimension 0 is the INPUT channel), bias [cout]. */
static int acv_load_convt(shards *S, const char *pfx, int cin, int cout, int s, AcvConvT *c, char *err, size_t errn){
    char n[192]; int k = 2 * s;
    int64_t shv[3] = {cin, cout, k}, shg[3] = {cin, 1, 1}, shb[1] = {cout};
    snprintf(n, sizeof n, "%s.weight_v", pfx); float *v = acv_read(S, n, 3, shv, err, errn);
    snprintf(n, sizeof n, "%s.weight_g", pfx); float *g = v ? acv_read(S, n, 3, shg, err, errn) : NULL;
    float *b = NULL, *w = NULL, *scale = NULL;
    if (!v || !g) goto fail;
    snprintf(n, sizeof n, "%s.bias", pfx); if (!(b = acv_read(S, n, 1, shb, err, errn))) goto fail;
    w = acv_alloc((size_t)k * cout * cin); scale = acv_alloc((size_t)cin);
    if (!w || !scale) { snprintf(err, errn, "out of memory"); goto fail; }
    acv_fold_scales(v, g, cin, (int64_t)cout * k, scale);
    for (int ci = 0; ci < cin; ci++)
        for (int co = 0; co < cout; co++)
            for (int j = 0; j < k; j++)
                w[((size_t)j * cout + co) * cin + ci] = v[((size_t)ci * cout + co) * k + j] * scale[ci];
    free(v); free(g); free(scale);
    c->cin = cin; c->cout = cout; c->s = s; c->pad = (s + 1) / 2; c->w = w; c->b = b;
    c->m = (QiMat){QI_F32, k * cout, cin, w, NULL, 0};
    return 0;
fail:
    free(v); free(g); free(b); free(w); free(scale);
    return -1;
}

static void acv_free(AcVae *v){
    if (!v) return;
    acv_free_conv(&v->conv1); acv_free_conv(&v->conv2); acv_free_snake(&v->sn);
    for (int i = 0; i < v->nblk; i++) {
        AcvBlock *b = &v->blk[i];
        acv_free_snake(&b->sn); acv_free_convt(&b->up);
        for (int j = 0; j < 3; j++) { acv_free_snake(&b->r[j].s1); acv_free_snake(&b->r[j].s2); acv_free_conv(&b->r[j].c1); acv_free_conv(&b->r[j].c2); }
    }
    free(v);
}

static int acv_json_int(jval *cfg, const char *key, int def){
    jval *j = json_get(cfg, key);
    return (j && j->t == J_NUM) ? (int)j->num : def;
}

/* A JSON array of positive integers into out[]; returns its length, or -1. */
static int acv_json_ints(jval *cfg, const char *key, int *out, int cap){
    jval *j = json_get(cfg, key);
    if (!j || j->t != J_ARR || j->len < 1 || j->len > cap) return -1;
    for (int i = 0; i < j->len; i++) {
        if (j->kids[i]->t != J_NUM || j->kids[i]->num < 1 || j->kids[i]->num > ACV_MAX_CHANNELS || j->kids[i]->num != (int)j->kids[i]->num) return -1;
        out[i] = (int)j->kids[i]->num;
    }
    return j->len;
}

/* vae_dir holds config.json and diffusion_pytorch_model.safetensors (the Hugging Face "vae" directory). */
static AcVae *acv_load(const char *vae_dir){
    char err[512] = "", path[4096];
    AcVae *v = (AcVae *)calloc(1, sizeof(AcVae));
    shards S; int have_s = 0;
    jval *cfg = NULL; char *text = NULL;
    if (!v) { fprintf(stderr, "acestep vae: out of memory\n"); return NULL; }

    if (snprintf(path, sizeof path, "%s/config.json", vae_dir) >= (int)sizeof path) { snprintf(err, sizeof err, "path too long"); goto fail; }
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, sizeof err, "cannot open %.400s", path); goto fail; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    text = (char *)malloc(n > 0 ? n + 1 : 1);
    if (!text || n <= 0 || n > (8 << 20) || fread(text, 1, n, f) != (size_t)n) { fclose(f); snprintf(err, sizeof err, "cannot read %.400s", path); goto fail; }
    fclose(f); text[n] = 0;
    cfg = json_parse_checked(text);
    if (!cfg || cfg->t != J_OBJ) { snprintf(err, sizeof err, "%.400s is not a JSON object", path); goto fail; }

    int ratios[ACV_MAX_BLOCKS], mults[ACV_MAX_BLOCKS];
    int nr = acv_json_ints(cfg, "downsampling_ratios", ratios, ACV_MAX_BLOCKS);
    int nm = acv_json_ints(cfg, "channel_multiples", mults, ACV_MAX_BLOCKS);
    int ch = acv_json_int(cfg, "decoder_channels", 0);
    v->zdim = acv_json_int(cfg, "decoder_input_channels", 0);
    v->ach = acv_json_int(cfg, "audio_channels", 0);
    v->rate = acv_json_int(cfg, "sampling_rate", 0);
    if (nr < 1 || nm != nr) { snprintf(err, sizeof err, "config: downsampling_ratios and channel_multiples must be arrays of the same length (1..%d)", ACV_MAX_BLOCKS); goto fail; }
    if (ch < 1 || ch > ACV_MAX_CHANNELS || v->zdim < 1 || v->zdim > ACV_MAX_CHANNELS || v->ach < 1 || v->ach > 8) { snprintf(err, sizeof err, "config: decoder_channels, decoder_input_channels or audio_channels out of range"); goto fail; }
    v->nblk = nr; v->hop = 1;
    for (int i = 0; i < nr; i++) {
        if (ratios[i] > ACV_MAX_STRIDE || (int64_t)mults[i] * ch > ACV_MAX_CHANNELS) { snprintf(err, sizeof err, "config: ratio %d or width %d out of range", ratios[i], mults[i] * ch); goto fail; }
        v->hop *= ratios[i];
    }
    /* widths: [1] + channel_multiples, block i goes from mult[nblk-i] to mult[nblk-i-1] with the reversed ratios */
    int m[ACV_MAX_BLOCKS + 1]; m[0] = 1;
    for (int i = 0; i < nm; i++) m[i + 1] = mults[i];

    st_init(&S, vae_dir); have_s = 1;
    if (S.n == 0) { snprintf(err, sizeof err, "no safetensors in %.400s", vae_dir); goto fail; }

    if (acv_load_conv(&S, "decoder.conv1", v->zdim, ch * m[nr], 7, 1, 1, &v->conv1, err, sizeof err)) goto fail;
    for (int i = 0; i < nr; i++) {
        AcvBlock *b = &v->blk[i];
        char p[160];
        b->cin = ch * m[nr - i]; b->cout = ch * m[nr - i - 1]; b->s = ratios[nr - 1 - i];
        snprintf(p, sizeof p, "decoder.block.%d.snake1", i);
        if (acv_load_snake(&S, p, b->cin, &b->sn, err, sizeof err)) goto fail;
        snprintf(p, sizeof p, "decoder.block.%d.conv_t1", i);
        if (acv_load_convt(&S, p, b->cin, b->cout, b->s, &b->up, err, sizeof err)) goto fail;
        static const int dil[3] = {1, 3, 9};
        for (int j = 0; j < 3; j++) {
            AcvRes *r = &b->r[j];
            snprintf(p, sizeof p, "decoder.block.%d.res_unit%d.snake1", i, j + 1);
            if (acv_load_snake(&S, p, b->cout, &r->s1, err, sizeof err)) goto fail;
            snprintf(p, sizeof p, "decoder.block.%d.res_unit%d.conv1", i, j + 1);
            if (acv_load_conv(&S, p, b->cout, b->cout, 7, dil[j], 1, &r->c1, err, sizeof err)) goto fail;
            snprintf(p, sizeof p, "decoder.block.%d.res_unit%d.snake2", i, j + 1);
            if (acv_load_snake(&S, p, b->cout, &r->s2, err, sizeof err)) goto fail;
            snprintf(p, sizeof p, "decoder.block.%d.res_unit%d.conv2", i, j + 1);
            if (acv_load_conv(&S, p, b->cout, b->cout, 1, 1, 1, &r->c2, err, sizeof err)) goto fail;
        }
        v->wbytes += sizeof(float) * ((size_t)b->up.cin * b->up.cout * 2 * b->s + 3 * ((size_t)b->cout * b->cout * 8));
    }
    if (acv_load_snake(&S, "decoder.snake1", ch, &v->sn, err, sizeof err)) goto fail;
    if (acv_load_conv(&S, "decoder.conv2", ch, v->ach, 7, 1, 0, &v->conv2, err, sizeof err)) goto fail;
    st_destroy(&S);
    json_free(cfg); free(text);
    return v;
fail:
    fprintf(stderr, "acestep vae: %s\n", err[0] ? err : "load failed");
    if (have_s) st_destroy(&S);
    json_free(cfg); free(text);
    acv_free(v);
    return NULL;
}

/* -------------------------------------------------------------------- math -- */

/* y = x + sin^2(ea x) * ib per channel, rows of c channels. dst may equal src. */
static void acv_snake(float *dst, const float *src, int T, const AcvSnake *s){
    const int c = s->c;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float *x = src + (size_t)t * c; float *y = dst + (size_t)t * c;
        for (int i = 0; i < c; i++) {
            float sn = sinf(s->ea[i] * x[i]);
            y[i] = x[i] + s->ib[i] * (sn * sn);
        }
    }
}

/* y[T][cout] = conv1d(x[T][cin]), zero padding (k-1)*dil/2 each side: the output has the input's length. */
static void acv_conv(const AcvConv *c, const float *x, int T, float *y, float *col){
    if (c->k == 1) { qi_gemm(y, x, T, &c->m, c->b); return; }
    const int kc = c->k * c->cin, pad = (c->k - 1) * c->dil / 2;
    int rows = (int)(acv_col_floats / (size_t)kc);
    if (rows < 1) rows = 1;
    for (int r0 = 0; r0 < T; r0 += rows) {
        int n = T - r0 < rows ? T - r0 : rows;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < c->k; j++) {
                int s = r0 + i + j * c->dil - pad;
                float *d = col + ((size_t)i * c->k + j) * c->cin;
                if (s >= 0 && s < T) memcpy(d, x + (size_t)s * c->cin, (size_t)c->cin * sizeof(float));
                else memset(d, 0, (size_t)c->cin * sizeof(float));
            }
        qi_gemm(y + (size_t)r0 * c->cout, col, n, &c->m, c->b);
    }
}

/* y[s*T][cout] = conv_transpose1d(x[T][cin]) with kernel 2s, stride s, padding ceil(s/2). Output row j takes tap
 * k = j+pad-t*s of input frame t, for the two t with k in [0, 2s). */
static void acv_convt(const AcvConvT *c, const float *x, int T, float *y, float *zb){
    const int s = c->s, cout = c->cout, nz = 2 * s * cout, pad = c->pad;
    const int64_t nout = (int64_t)s * T;
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < nout; j++)
        for (int i = 0; i < cout; i++) y[(size_t)j * cout + i] = c->b[i];
    int rows = (int)(acv_z_floats / (size_t)nz);
    if (rows < 1) rows = 1;
    for (int t0 = 0; t0 < T; t0 += rows) {
        int tn = T - t0 < rows ? T - t0 : rows;
        qi_gemm(zb, x + (size_t)t0 * c->cin, tn, &c->m, NULL);
        int64_t jlo = (int64_t)t0 * s - pad, jhi = (int64_t)(t0 + tn - 1) * s + 2 * s - pad;
        if (jlo < 0) jlo = 0;
        if (jhi > nout) jhi = nout;
        #pragma omp parallel for schedule(static)
        for (int64_t j = jlo; j < jhi; j++) {
            int64_t jp = j + pad, t1 = jp / s;
            int kk = (int)(jp - t1 * s);
            float *o = y + (size_t)j * cout;
            if (t1 >= t0 && t1 < t0 + tn) {
                const float *z = zb + (size_t)(t1 - t0) * nz + (size_t)kk * cout;
                for (int i = 0; i < cout; i++) o[i] += z[i];
            }
            if (t1 - 1 >= t0 && t1 - 1 < t0 + tn && t1 >= 1) {
                const float *z = zb + (size_t)(t1 - 1 - t0) * nz + (size_t)(kk + s) * cout;
                for (int i = 0; i < cout; i++) o[i] += z[i];
            }
        }
    }
}

/* x += conv2(snake2(conv1(snake1(x)))): a and b are scratch of at least T*C floats. */
static void acv_res(const AcvRes *r, float *x, int T, float *a, float *b, float *col){
    acv_snake(a, x, T, &r->s1);
    acv_conv(&r->c1, a, T, b, col);
    acv_snake(b, b, T, &r->s2);
    acv_conv(&r->c2, b, T, a, col);
    const size_t n = (size_t)T * r->c1.cout;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; i++) x[i] += a[i];
}

/* ----------------------------------------------------------------- driver -- */

static void acv_work_free(AcvWork *w){
    if (!w) return;
    for (int i = 0; i < 3; i++) free(w->buf[i]);
    free(w->col); free(w->zb); free(w);
}

/* Scratch big enough for any decode of up to maxT frames. */
static AcvWork *acv_work_new(const AcVae *v, int maxT){
    AcvWork *w = (AcvWork *)calloc(1, sizeof *w);
    if (!w) return NULL;
    size_t nbuf = (size_t)maxT * v->conv1.cout, ncol = 0, nzb = 0;
    size_t kc = (size_t)v->conv1.k * v->conv1.cin;
    ncol = kc;
    int64_t T = maxT;
    for (int i = 0; i < v->nblk; i++) {
        const AcvBlock *b = &v->blk[i];
        T *= b->s;
        if ((size_t)T * b->cout > nbuf) nbuf = (size_t)T * b->cout;
        if ((size_t)T / b->s * b->cin > nbuf) nbuf = (size_t)(T / b->s) * b->cin;
        size_t zr = (size_t)2 * b->s * b->cout;
        if (zr > nzb) nzb = zr;
        if ((size_t)7 * b->cout > ncol) ncol = (size_t)7 * b->cout;
    }
    if ((size_t)T * v->ach > nbuf) nbuf = (size_t)T * v->ach;
    if ((size_t)7 * v->conv2.cin > ncol) ncol = (size_t)7 * v->conv2.cin;
    /* a band holds acv_col_floats floats, but never fewer than one row, and never more than the clip needs */
    size_t colcap = acv_col_floats > ncol ? acv_col_floats : ncol;
    size_t zcap = acv_z_floats > nzb ? acv_z_floats : nzb;
    (void)T;
    w->nbuf = nbuf; w->ncol = colcap; w->nzb = zcap; w->maxT = maxT;
    for (int i = 0; i < 3; i++) w->buf[i] = acv_alloc(nbuf);
    w->col = acv_alloc(colcap); w->zb = acv_alloc(zcap);
    if (!w->buf[0] || !w->buf[1] || !w->buf[2] || !w->col || !w->zb) { acv_work_free(w); return NULL; }
    return w;
}

/* z [T][zdim] -> out [T*hop][ach], T <= w->maxT. */
static void acv_decode_ws(const AcVae *v, AcvWork *w, const float *z, int T, float *out){
    float *x = w->buf[0], *nx = w->buf[1], *tmp = w->buf[2];
    acv_conv(&v->conv1, z, T, x, w->col);
    int64_t cur = T;
    for (int i = 0; i < v->nblk; i++) {
        const AcvBlock *b = &v->blk[i];
        acv_snake(x, x, (int)cur, &b->sn);
        acv_convt(&b->up, x, (int)cur, nx, w->zb);
        cur *= b->s;
        float *t = x; x = nx; nx = t;
        for (int j = 0; j < 3; j++) acv_res(&b->r[j], x, (int)cur, nx, tmp, w->col);
    }
    acv_snake(x, x, (int)cur, &v->sn);
    acv_conv(&v->conv2, x, (int)cur, out, w->col);
}

/* One decode of the whole clip: z [T][zdim] -> out [T*hop][ach] (interleaved channels). 0 on success. */
static int acv_decode(const AcVae *v, const float *z, int T, float *out){
    if (T < 1) return -1;
    AcvWork *w = acv_work_new(v, T);
    if (!w) return -1;
    acv_decode_ws(v, w, z, T, out);
    acv_work_free(w);
    return 0;
}

/* Upstream's tiled decode (acestep/core/generation/handler/vae_decode_chunks.py): windows of `chunk` frames with
 * `overlap` frames of context on each side, only each window's core of chunk-2*overlap frames kept. The overlap is
 * halved while the stride would be empty, but never below 4. out [F*hop][ach]. 0 on success. `progress` is called before each
 * window (done, total) and once at the end; a nonzero return from a call before a window stops the decode with -2. */
static int acv_decode_tiled(const AcVae *v, const float *z, int F, int chunk, int overlap, float *out,
                            int (*progress)(int done, int total, void *ud), void *ud){
    if (F < 1 || chunk < 1 || overlap < 0) return -1;
    if (F <= chunk) {
        if (progress && progress(0, 1, ud)) return -2;
        int r = acv_decode(v, z, F, out);
        if (progress) progress(1, 1, ud);
        return r;
    }
    const int min_overlap = 4;
    int ov = overlap;
    while (chunk - 2 * ov <= 0 && ov > min_overlap) ov /= 2;
    if (ov < min_overlap && overlap >= min_overlap) ov = min_overlap;
    const int stride = chunk - 2 * ov;
    if (stride <= 0) { fprintf(stderr, "acestep vae: chunk %d must exceed twice the overlap %d\n", chunk, ov); return -1; }
    const int steps = (F + stride - 1) / stride;
    int maxwin = stride + 2 * ov < F ? stride + 2 * ov : F;
    AcvWork *w = acv_work_new(v, maxwin);
    float *win = acv_alloc((size_t)maxwin * v->hop * v->ach);
    if (!w || !win) { acv_work_free(w); free(win); return -1; }
    size_t pos = 0;
    for (int i = 0; i < steps; i++) {
        if (progress && progress(i, steps, ud)) { acv_work_free(w); free(win); return -2; }       /* asked to stop */
        int cs = i * stride, ce = cs + stride < F ? cs + stride : F;
        int ws = cs - ov > 0 ? cs - ov : 0, we = ce + ov < F ? ce + ov : F;
        acv_decode_ws(v, w, z + (size_t)ws * v->zdim, we - ws, win);
        size_t trim0 = (size_t)(cs - ws) * v->hop, trim1 = (size_t)(we - ce) * v->hop;
        size_t len = (size_t)(we - ws) * v->hop;
        memcpy(out + pos * v->ach, win + trim0 * v->ach, (len - trim0 - trim1) * v->ach * sizeof(float));
        pos += len - trim0 - trim1;
    }
    if (progress) progress(steps, steps, ud);
    acv_work_free(w); free(win);
    return pos == (size_t)F * v->hop ? 0 : -1;
}

/* The handler's last step: a waveform whose peak exceeds 1 is divided by its peak (never amplified). */
static float acv_normalize(float *wav, size_t n){
    float peak = 0;
    for (size_t i = 0; i < n; i++) { float a = fabsf(wav[i]); if (a > peak) peak = a; }
    if (peak > 1.0f) for (size_t i = 0; i < n; i++) wav[i] /= peak;
    return peak;
}

#endif
