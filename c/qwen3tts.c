/* qwen3tts.c -- text to speech with Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice and its siblings), docs/qwen3-speech.md.
 *
 * What the reference (the qwen-tts package) does, and so what this does, in the order it was built:
 *
 *   T1  the speech codec's decoder: 16 codes a frame (12.5 frames a second) -> 24 kHz audio, 1920 samples a frame.
 *         the codes are looked up (embedding_sum / max(cluster_usage, 1e-5)), the first codebook and the sum of the other 15 each
 *         projected 256 -> 512 and added; a causal conv 512 -> 1024; a transformer (8 layers, sliding window 72, layer scales)
 *         over 512; two stages of causal transposed conv (x2) + ConvNeXt block; a causal conv to 1536, four blocks (SnakeBeta,
 *         transposed conv by 8, 5, 4, 3, three residual units of dilation 1, 3, 9), SnakeBeta, a causal conv to 1 channel, clamp.
 *         Long streams are decoded in chunks of 300 frames with 25 frames of left context, as the reference does.
 *
 * The kernels and the Qwen3 layer helpers are the ones the ACE-Step and Qwen3-ASR engines use (qi_gemm.h, acestep_dit.h).
 *
 * Usage:
 *   qwen3tts --codec DIR --decode-codes FILE.i32 [--out FILE.wav] [--ref REFDIR]
 *       FILE.i32 is [frames][16] little-endian int32 codes; DIR is the codec's folder (speech_tokenizer/ of a TTS checkpoint, or
 *       Qwen3-TTS-Tokenizer-12Hz); --ref holds the reference's wav.f32 to compare with.
 */
#define _GNU_SOURCE
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
#include "acestep_text.h"

#define QT_SR 24000
#define QT_GROUPS 16
#define QT_TILE 1024                /* positions through one im2col GEMM */

static double qt_now(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void qt_die(const char *m){ fprintf(stderr, "qwen3tts: %s\n", m); exit(1); }

/* ------------------------------------------------------------------------------------------------ weights */

typedef struct { AcdLin w; float *b; int K, dil, cin, cout; } QtConv;          /* a causal conv, stride 1; w is [cout][K*cin] (taps outer) */
typedef struct { AcdLin w; float *b; int r, cin, cout; } QtConvT;              /* a causal transposed conv; w is [cout*2r][cin] */
typedef struct { float *ea, *ib; int c; } QtSnake;                             /* exp(alpha) and 1 / (exp(beta) + 1e-9) */
typedef struct { QtSnake a1, a2; QtConv c1, c2; } QtRes;
typedef struct { QtSnake sn; QtConvT up; QtRes r[3]; } QtBlock;
typedef struct { float *dw, *dwb, *lnw, *lnb, *g; AcdLin pw1, pw2; float *pw1b, *pw2b; } QtNeXt;
typedef struct { AcdAttn a; float *ln1, *ln2, *s1, *s2; AcdLin gate, up, down; } QtTLayer;

typedef struct {
    int latent, cbdim, cbsize, ddim, hid, inter, hd, heads, nlayers, nq, window;
    float eps, theta;
    float *cb[QT_GROUPS];                 /* [cbsize][cbdim/2] each */
    AcdLin proj_first, proj_rest;         /* 256 -> 512, no bias */
    QtConv pre_conv;
    AcdLin in_proj, out_proj; float *in_b, *out_b;
    QtTLayer *L;
    float *tnorm;
    QtConvT up[2];
    QtNeXt nx[2];
    QtConv d0;
    QtBlock blk[4];
    int rates[4];
    QtSnake sn_out;
    QtConv d_out;
    size_t bytes;
} QtCodec;

static float *qt_read(shards *S, const char *name, int64_t numel, char *err, size_t errn){
    st_tensor *t = st_find(S, name);
    if (!t) { snprintf(err, errn, "missing tensor %s", name); return NULL; }
    if (t->dtype > 2) { snprintf(err, errn, "tensor %s is not a float tensor", name); return NULL; }
    if (t->numel != numel) { snprintf(err, errn, "tensor %s has %lld values, expected %lld", name, (long long)t->numel, (long long)numel); return NULL; }
    float *p = (float *)malloc((size_t)(numel ? numel : 1) * sizeof(float));
    if (!p) { snprintf(err, errn, "out of memory reading %s", name); return NULL; }
    st_read_f32_cap(S, name, p, numel, 1);
    return p;
}

static int qt_lin(shards *S, const char *name, int n, int k, AcdLin *l, size_t *bytes, char *err, size_t errn){
    float *w = qt_read(S, name, (int64_t)n * k, err, errn);
    if (!w) return -1;
    l->m = (QiMat){QI_F32, n, k, w, NULL, 0}; l->own = w; l->b = NULL;
    *bytes += (size_t)n * k * 4;
    return 0;
}

/* Conv1d weight [cout][cin][K] -> [cout][K][cin] so that an im2col row is K taps of cin channels */
static int qt_conv(shards *S, const char *wn, const char *bn, int cout, int cin, int K, int dil, QtConv *c, size_t *bytes, char *err, size_t errn){
    float *raw = qt_read(S, wn, (int64_t)cout * cin * K, err, errn);
    if (!raw) return -1;
    float *w = (float *)malloc((size_t)cout * K * cin * sizeof(float));
    for (int o = 0; o < cout; o++)
        for (int i = 0; i < cin; i++)
            for (int k = 0; k < K; k++) w[((size_t)o * K + k) * cin + i] = raw[((size_t)o * cin + i) * K + k];
    free(raw);
    c->w.m = (QiMat){QI_F32, cout, K * cin, w, NULL, 0}; c->w.own = w; c->w.b = NULL;
    c->b = bn ? qt_read(S, bn, cout, err, errn) : NULL;
    if (bn && !c->b) return -1;
    c->K = K; c->dil = dil; c->cin = cin; c->cout = cout;
    *bytes += (size_t)cout * K * cin * 4;
    return 0;
}

/* ConvTranspose1d weight [cin][cout][2r] -> rows (co * 2r + j), columns ci */
static int qt_convt(shards *S, const char *wn, const char *bn, int cin, int cout, int r, QtConvT *c, size_t *bytes, char *err, size_t errn){
    const int K = 2 * r;
    float *raw = qt_read(S, wn, (int64_t)cin * cout * K, err, errn);
    if (!raw) return -1;
    float *w = (float *)malloc((size_t)cout * K * cin * sizeof(float));
    for (int i = 0; i < cin; i++)
        for (int o = 0; o < cout; o++)
            for (int j = 0; j < K; j++) w[((size_t)o * K + j) * cin + i] = raw[((size_t)i * cout + o) * K + j];
    free(raw);
    c->w.m = (QiMat){QI_F32, cout * K, cin, w, NULL, 0}; c->w.own = w; c->w.b = NULL;
    c->b = qt_read(S, bn, cout, err, errn);
    if (!c->b) return -1;
    c->r = r; c->cin = cin; c->cout = cout;
    *bytes += (size_t)cout * K * cin * 4;
    return 0;
}

static int qt_snake(shards *S, const char *an, const char *bn, int c, QtSnake *s, char *err, size_t errn){
    float *a = qt_read(S, an, c, err, errn), *b = qt_read(S, bn, c, err, errn);
    if (!a || !b) { free(a); free(b); return -1; }
    for (int i = 0; i < c; i++) { a[i] = expf(a[i]); b[i] = 1.0f / (expf(b[i]) + 0.000000001f); }
    s->ea = a; s->ib = b; s->c = c;
    return 0;
}

static int qt_codec_load(QtCodec *m, const char *dir, char *err, size_t errn){
    memset(m, 0, sizeof *m);
    char path[4300];
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errn, "cannot open %s", path); return -1; }
    fseek(f, 0, SEEK_END); long fn = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)fn + 1);
    if (!buf || fread(buf, 1, (size_t)fn, f) != (size_t)fn) { fclose(f); free(buf); snprintf(err, errn, "cannot read %s", path); return -1; }
    fclose(f); buf[fn] = 0;
    jval *root = json_parse_checked(buf);
    free(buf);
    jval *dc = root ? json_get(root, "decoder_config") : NULL;
    if (!dc) { snprintf(err, errn, "config.json has no decoder_config: not a Qwen3-TTS speech codec"); return -1; }
    m->latent = acd_json_int(dc, "latent_dim", 0); m->cbdim = acd_json_int(dc, "codebook_dim", 0); m->cbsize = acd_json_int(dc, "codebook_size", 0);
    m->ddim = acd_json_int(dc, "decoder_dim", 0); m->hid = acd_json_int(dc, "hidden_size", 0); m->inter = acd_json_int(dc, "intermediate_size", 0);
    m->hd = acd_json_int(dc, "head_dim", 0); m->heads = acd_json_int(dc, "num_attention_heads", 0); m->nlayers = acd_json_int(dc, "num_hidden_layers", 0);
    m->nq = acd_json_int(dc, "num_quantizers", 0); m->window = acd_json_int(dc, "sliding_window", 0);
    m->eps = (float)acd_json_num(dc, "rms_norm_eps", 1e-5); m->theta = (float)acd_json_num(dc, "rope_theta", 10000.0);
    jval *ur = json_get(dc, "upsample_rates"), *ug = json_get(dc, "upsampling_ratios");
    if (!ur || ur->t != J_ARR || ur->len != 4 || !ug || ug->t != J_ARR || ug->len != 2 || m->nq != QT_GROUPS || m->cbdim % 2 || m->heads * m->hd <= 0) {
        snprintf(err, errn, "unsupported codec geometry in config.json"); return -1;
    }
    for (int i = 0; i < 4; i++) m->rates[i] = (int)ur->kids[i]->num;
    for (int i = 0; i < 2; i++) if ((int)ug->kids[i]->num != 2) { snprintf(err, errn, "only x2 upsampling stages are supported"); return -1; }
    shards S;
    st_init(&S, dir);
    char n[300], b[300];
    const int CB = m->cbdim / 2;
    for (int q = 0; q < QT_GROUPS; q++) {
        const char *grp = q == 0 ? "rvq_first" : "rvq_rest";
        const int layer = q == 0 ? 0 : q - 1;
        snprintf(n, sizeof n, "decoder.quantizer.%s.vq.layers.%d._codebook.embedding_sum", grp, layer);
        snprintf(b, sizeof b, "decoder.quantizer.%s.vq.layers.%d._codebook.cluster_usage", grp, layer);
        float *sum = qt_read(&S, n, (int64_t)m->cbsize * CB, err, errn), *use = qt_read(&S, b, m->cbsize, err, errn);
        if (!sum || !use) return -1;
        for (int i = 0; i < m->cbsize; i++) {
            const float u = use[i] < 1e-5f ? 1e-5f : use[i];
            for (int j = 0; j < CB; j++) sum[(size_t)i * CB + j] /= u;
        }
        free(use);
        m->cb[q] = sum;
        m->bytes += (size_t)m->cbsize * CB * 4;
    }
    if (qt_lin(&S, "decoder.quantizer.rvq_first.output_proj.weight", m->cbdim, CB, &m->proj_first, &m->bytes, err, errn)) return -1;
    if (qt_lin(&S, "decoder.quantizer.rvq_rest.output_proj.weight", m->cbdim, CB, &m->proj_rest, &m->bytes, err, errn)) return -1;
    if (qt_conv(&S, "decoder.pre_conv.conv.weight", "decoder.pre_conv.conv.bias", m->latent, m->cbdim, 3, 1, &m->pre_conv, &m->bytes, err, errn)) return -1;
    if (qt_lin(&S, "decoder.pre_transformer.input_proj.weight", m->hid, m->latent, &m->in_proj, &m->bytes, err, errn)) return -1;
    if (!(m->in_b = qt_read(&S, "decoder.pre_transformer.input_proj.bias", m->hid, err, errn))) return -1;
    if (qt_lin(&S, "decoder.pre_transformer.output_proj.weight", m->latent, m->hid, &m->out_proj, &m->bytes, err, errn)) return -1;
    if (!(m->out_b = qt_read(&S, "decoder.pre_transformer.output_proj.bias", m->latent, err, errn))) return -1;
    if (!(m->tnorm = qt_read(&S, "decoder.pre_transformer.norm.weight", m->hid, err, errn))) return -1;
    const int qd = m->heads * m->hd;
    m->L = (QtTLayer *)calloc((size_t)m->nlayers, sizeof *m->L);
    for (int i = 0; i < m->nlayers; i++) {
        QtTLayer *l = &m->L[i];
        char p[300];
        struct { const char *nm; AcdLin *dst; int N, K; } mats[4] = {{"self_attn.q_proj", &l->a.q, qd, m->hid}, {"self_attn.k_proj", &l->a.k, qd, m->hid},
                                                                      {"self_attn.v_proj", &l->a.v, qd, m->hid}, {"self_attn.o_proj", &l->a.o, m->hid, qd}};
        for (int j = 0; j < 4; j++) {
            snprintf(p, sizeof p, "decoder.pre_transformer.layers.%d.%s.weight", i, mats[j].nm);
            if (qt_lin(&S, p, mats[j].N, mats[j].K, mats[j].dst, &m->bytes, err, errn)) return -1;
        }
        struct { const char *nm; AcdLin *dst; int N, K; } mlp[3] = {{"mlp.gate_proj", &l->gate, m->inter, m->hid}, {"mlp.up_proj", &l->up, m->inter, m->hid},
                                                                    {"mlp.down_proj", &l->down, m->hid, m->inter}};
        for (int j = 0; j < 3; j++) {
            snprintf(p, sizeof p, "decoder.pre_transformer.layers.%d.%s.weight", i, mlp[j].nm);
            if (qt_lin(&S, p, mlp[j].N, mlp[j].K, mlp[j].dst, &m->bytes, err, errn)) return -1;
        }
        struct { const char *nm; float **dst; } vecs[4] = {{"input_layernorm.weight", &l->ln1}, {"post_attention_layernorm.weight", &l->ln2},
                                                           {"self_attn_layer_scale.scale", &l->s1}, {"mlp_layer_scale.scale", &l->s2}};
        for (int j = 0; j < 4; j++) {
            snprintf(p, sizeof p, "decoder.pre_transformer.layers.%d.%s", i, vecs[j].nm);
            if (!(*vecs[j].dst = qt_read(&S, p, m->hid, err, errn))) return -1;
        }
    }
    for (int i = 0; i < 2; i++) {
        snprintf(n, sizeof n, "decoder.upsample.%d.0.conv.weight", i); snprintf(b, sizeof b, "decoder.upsample.%d.0.conv.bias", i);
        if (qt_convt(&S, n, b, m->latent, m->latent, 1, &m->up[i], &m->bytes, err, errn)) return -1;     /* kernel 2, stride 2: r = 1 here means k = 2r = 2 */
        m->up[i].r = 2;                                                                                    /* the stride; its kernel is 2, not 4 */
        QtNeXt *x = &m->nx[i];
        char p[300];
        snprintf(p, sizeof p, "decoder.upsample.%d.1.dwconv.conv.weight", i); if (!(x->dw = qt_read(&S, p, (int64_t)m->latent * 7, err, errn))) return -1;
        snprintf(p, sizeof p, "decoder.upsample.%d.1.dwconv.conv.bias", i); if (!(x->dwb = qt_read(&S, p, m->latent, err, errn))) return -1;
        snprintf(p, sizeof p, "decoder.upsample.%d.1.norm.weight", i); if (!(x->lnw = qt_read(&S, p, m->latent, err, errn))) return -1;
        snprintf(p, sizeof p, "decoder.upsample.%d.1.norm.bias", i); if (!(x->lnb = qt_read(&S, p, m->latent, err, errn))) return -1;
        snprintf(p, sizeof p, "decoder.upsample.%d.1.gamma", i); if (!(x->g = qt_read(&S, p, m->latent, err, errn))) return -1;
        snprintf(p, sizeof p, "decoder.upsample.%d.1.pwconv1.weight", i); if (qt_lin(&S, p, 4 * m->latent, m->latent, &x->pw1, &m->bytes, err, errn)) return -1;
        snprintf(p, sizeof p, "decoder.upsample.%d.1.pwconv1.bias", i); if (!(x->pw1b = qt_read(&S, p, 4 * m->latent, err, errn))) return -1;
        snprintf(p, sizeof p, "decoder.upsample.%d.1.pwconv2.weight", i); if (qt_lin(&S, p, m->latent, 4 * m->latent, &x->pw2, &m->bytes, err, errn)) return -1;
        snprintf(p, sizeof p, "decoder.upsample.%d.1.pwconv2.bias", i); if (!(x->pw2b = qt_read(&S, p, m->latent, err, errn))) return -1;
    }
    if (qt_conv(&S, "decoder.decoder.0.conv.weight", "decoder.decoder.0.conv.bias", m->ddim, m->latent, 7, 1, &m->d0, &m->bytes, err, errn)) return -1;
    int cin = m->ddim;
    for (int i = 0; i < 4; i++) {
        const int cout = m->ddim >> (i + 1), idx = i + 1;
        QtBlock *bk = &m->blk[i];
        snprintf(n, sizeof n, "decoder.decoder.%d.block.0.alpha", idx); snprintf(b, sizeof b, "decoder.decoder.%d.block.0.beta", idx);
        if (qt_snake(&S, n, b, cin, &bk->sn, err, errn)) return -1;
        snprintf(n, sizeof n, "decoder.decoder.%d.block.1.conv.weight", idx); snprintf(b, sizeof b, "decoder.decoder.%d.block.1.conv.bias", idx);
        if (qt_convt(&S, n, b, cin, cout, m->rates[i], &bk->up, &m->bytes, err, errn)) return -1;
        static const int dils[3] = {1, 3, 9};
        for (int u = 0; u < 3; u++) {
            QtRes *r = &bk->r[u];
            const int bi = u + 2;
            snprintf(n, sizeof n, "decoder.decoder.%d.block.%d.act1.alpha", idx, bi); snprintf(b, sizeof b, "decoder.decoder.%d.block.%d.act1.beta", idx, bi);
            if (qt_snake(&S, n, b, cout, &r->a1, err, errn)) return -1;
            snprintf(n, sizeof n, "decoder.decoder.%d.block.%d.act2.alpha", idx, bi); snprintf(b, sizeof b, "decoder.decoder.%d.block.%d.act2.beta", idx, bi);
            if (qt_snake(&S, n, b, cout, &r->a2, err, errn)) return -1;
            snprintf(n, sizeof n, "decoder.decoder.%d.block.%d.conv1.conv.weight", idx, bi); snprintf(b, sizeof b, "decoder.decoder.%d.block.%d.conv1.conv.bias", idx, bi);
            if (qt_conv(&S, n, b, cout, cout, 7, dils[u], &r->c1, &m->bytes, err, errn)) return -1;
            snprintf(n, sizeof n, "decoder.decoder.%d.block.%d.conv2.conv.weight", idx, bi); snprintf(b, sizeof b, "decoder.decoder.%d.block.%d.conv2.conv.bias", idx, bi);
            if (qt_conv(&S, n, b, cout, cout, 1, 1, &r->c2, &m->bytes, err, errn)) return -1;
        }
        cin = cout;
    }
    if (qt_snake(&S, "decoder.decoder.5.alpha", "decoder.decoder.5.beta", cin, &m->sn_out, err, errn)) return -1;
    if (qt_conv(&S, "decoder.decoder.6.conv.weight", "decoder.decoder.6.conv.bias", 1, cin, 7, 1, &m->d_out, &m->bytes, err, errn)) return -1;
    st_destroy(&S);
    return 0;
}

/* ------------------------------------------------------------------------------------------------ the codec's layers (time-major: x [T][C]) */

/* y [T][cout] = causal conv of x [T][cin] (stride 1, dilation c->dil): tiles of QT_TILE positions through an im2col GEMM */
static void qt_conv_run(float *y, const float *x, int T, const QtConv *c){
    const int K = c->K, cin = c->cin, kc = K * cin;
    const int span = (K - 1) * c->dil;
    float *col = (float *)malloc((size_t)QT_TILE * kc * sizeof(float));
    if (!col) qt_die("out of memory");
    for (int t0 = 0; t0 < T; t0 += QT_TILE) {
        const int tn = T - t0 < QT_TILE ? T - t0 : QT_TILE;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < tn; t++) {
            float *row = col + (size_t)t * kc;
            for (int k = 0; k < K; k++) {
                const int s = t0 + t - span + k * c->dil;
                if (s < 0) memset(row + (size_t)k * cin, 0, (size_t)cin * sizeof(float));
                else memcpy(row + (size_t)k * cin, x + (size_t)s * cin, (size_t)cin * sizeof(float));
            }
        }
        qi_gemm(y + (size_t)t0 * c->cout, col, tn, &c->w.m, c->b);
    }
    free(col);
}

/* y [T*r][cout] = causal transposed conv (kernel k, stride r; the reference trims the last k - r samples): input tiles through a GEMM,
 * overlap-added; positions past T*r are dropped */
static void qt_convt_run(float *y, const float *x, int T, const QtConvT *c, int kernel){
    const int r = c->r, cout = c->cout, K = kernel, rows = cout * K;
    const int TT = 1024;
    float *tmp = (float *)malloc((size_t)TT * rows * sizeof(float));
    if (!tmp) qt_die("out of memory");
    const size_t out_n = (size_t)T * r * cout;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)out_n; i++) y[i] = 0.f;
    for (int t0 = 0; t0 < T; t0 += TT) {
        const int tn = T - t0 < TT ? T - t0 : TT;
        qi_gemm(tmp, x + (size_t)t0 * c->cin, tn, &c->w.m, NULL);
        #pragma omp parallel for schedule(static)
        for (int co = 0; co < cout; co++)
            for (int t = 0; t < tn; t++)
                for (int j = 0; j < K; j++) {
                    const int pos = (t0 + t) * r + j;
                    if (pos < T * r) y[(size_t)pos * cout + co] += tmp[(size_t)t * rows + (size_t)co * K + j];
                }
    }
    #pragma omp parallel for schedule(static)
    for (int pos = 0; pos < T * r; pos++)
        for (int co = 0; co < cout; co++) y[(size_t)pos * cout + co] += c->b[co];
    free(tmp);
}

static void qt_snake_run(float *x, int T, const QtSnake *s){
    const int C = s->c;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        float *v = x + (size_t)t * C;
        for (int c = 0; c < C; c++) {
            const float sn = sinf(v[c] * s->ea[c]);
            v[c] = v[c] + s->ib[c] * sn * sn;
        }
    }
}

static inline float qt_gelu(float x){ return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

/* a ConvNeXt block over x [T][C] in place */
static void qt_convnext(float *x, int T, int C, const QtNeXt *nx){
    float *h = (float *)malloc((size_t)T * C * sizeof(float)), *z = (float *)malloc((size_t)T * 4 * C * sizeof(float)), *o = (float *)malloc((size_t)T * C * sizeof(float));
    if (!h || !z || !o) qt_die("out of memory");
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)                                       /* depthwise k7, causal */
        for (int c = 0; c < C; c++) {
            float s = nx->dwb[c];
            for (int k = 0; k < 7; k++) {
                const int tt = t - 6 + k;
                if (tt >= 0) s += nx->dw[(size_t)c * 7 + k] * x[(size_t)tt * C + c];
            }
            h[(size_t)t * C + c] = s;
        }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {                                     /* LayerNorm eps 1e-6 */
        float *v = h + (size_t)t * C;
        double mu = 0, var = 0;
        for (int c = 0; c < C; c++) mu += v[c];
        mu /= C;
        for (int c = 0; c < C; c++) { const double d = v[c] - mu; var += d * d; }
        var /= C;
        const float inv = 1.0f / sqrtf((float)var + 1e-6f);
        for (int c = 0; c < C; c++) v[c] = (float)((v[c] - mu) * inv) * nx->lnw[c] + nx->lnb[c];
    }
    qi_gemm(z, h, T, &nx->pw1.m, nx->pw1b);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * 4 * C; i++) z[i] = qt_gelu(z[i]);
    qi_gemm(o, z, T, &nx->pw2.m, nx->pw2b);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * C; i++) x[i] += nx->g[i % C] * o[i];
    free(h); free(z); free(o);
}

/* the codec's transformer on x [T][latent] -> [T][latent] (in place) */
static void qt_transformer(const QtCodec *m, float *x, int T){
    const int D = m->hid, H = m->heads, hd = m->hd, qd = H * hd, I = m->inter;
    float *h = (float *)malloc((size_t)T * D * sizeof(float));
    qi_gemm(h, x, T, &m->in_proj.m, m->in_b);
    float *n = (float *)malloc((size_t)T * D * sizeof(float)), *q = (float *)malloc((size_t)T * qd * sizeof(float)), *k = (float *)malloc((size_t)T * qd * sizeof(float));
    float *v = (float *)malloc((size_t)T * qd * sizeof(float)), *cx = (float *)malloc((size_t)T * qd * sizeof(float)), *o = (float *)malloc((size_t)T * D * sizeof(float));
    float *g = (float *)malloc((size_t)T * I * sizeof(float)), *u = (float *)malloc((size_t)T * I * sizeof(float));
    float *cs = (float *)malloc((size_t)T * hd * sizeof(float)), *sn = (float *)malloc((size_t)T * hd * sizeof(float));
    if (!n || !q || !k || !v || !cx || !o || !g || !u || !cs || !sn) qt_die("out of memory");
    acd_rope_table(cs, sn, T, hd, m->theta);
    for (int li = 0; li < m->nlayers; li++) {
        const QtTLayer *l = &m->L[li];
        acd_rmsnorm(n, h, T, D, l->ln1, m->eps);
        acd_lin(q, n, T, &l->a.q); acd_lin(k, n, T, &l->a.k); acd_lin(v, n, T, &l->a.v);
        acd_rope(q, T, H, hd, cs, sn);
        acd_rope(k, T, H, hd, cs, sn);
        acd_attention_ex(cx, qd, q, qd, T, k, v, qd, T, H, H, hd, m->window - 1, 1);      /* the last `window` positions: j > i - window */
        acd_lin(o, cx, T, &l->a.o);
        for (int64_t i = 0; i < (int64_t)T * D; i++) h[i] += l->s1[i % D] * o[i];
        acd_rmsnorm(n, h, T, D, l->ln2, m->eps);
        acd_lin(g, n, T, &l->gate); acd_lin(u, n, T, &l->up);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)T * I; i++) g[i] = acd_silu(g[i]) * u[i];
        acd_lin(o, g, T, &l->down);
        for (int64_t i = 0; i < (int64_t)T * D; i++) h[i] += l->s2[i % D] * o[i];
    }
    acd_rmsnorm(n, h, T, D, m->tnorm, m->eps);
    qi_gemm(x, n, T, &m->out_proj.m, m->out_b);
    free(h); free(n); free(q); free(k); free(v); free(cx); free(o); free(g); free(u); free(cs); free(sn);
}

/* codes [T][16] -> wav [T * 1920] (one chunk; the caller keeps chunks within memory) */
static float *qt_codec_chunk(const QtCodec *m, const int *codes, int T, int *nsamples){
    const int CB = m->cbdim / 2, C512 = m->cbdim;
    float *a = (float *)malloc((size_t)T * CB * sizeof(float)), *b = (float *)malloc((size_t)T * CB * sizeof(float));
    float *e = (float *)malloc((size_t)T * C512 * sizeof(float)), *e2 = (float *)malloc((size_t)T * C512 * sizeof(float));
    if (!a || !b || !e || !e2) qt_die("out of memory");
    for (int t = 0; t < T; t++) {
        memcpy(a + (size_t)t * CB, m->cb[0] + (size_t)codes[t * QT_GROUPS] * CB, (size_t)CB * sizeof(float));
        float *dst = b + (size_t)t * CB;
        memset(dst, 0, (size_t)CB * sizeof(float));
        for (int q = 1; q < QT_GROUPS; q++) {
            const float *src = m->cb[q] + (size_t)codes[t * QT_GROUPS + q] * CB;
            for (int j = 0; j < CB; j++) dst[j] += src[j];
        }
    }
    qi_gemm(e, a, T, &m->proj_first.m, NULL);
    qi_gemm(e2, b, T, &m->proj_rest.m, NULL);
    for (int64_t i = 0; i < (int64_t)T * C512; i++) e[i] += e2[i];
    free(a); free(b); free(e2);
    float *x = (float *)malloc((size_t)T * m->latent * sizeof(float));
    qt_conv_run(x, e, T, &m->pre_conv);
    free(e);
    qt_transformer(m, x, T);
    int len = T;
    for (int i = 0; i < 2; i++) {                                     /* x2 transposed conv (kernel 2) + ConvNeXt */
        float *y = (float *)malloc((size_t)len * 2 * m->latent * sizeof(float));
        qt_convt_run(y, x, len, &m->up[i], 2);
        free(x);
        x = y; len *= 2;
        qt_convnext(x, len, m->latent, &m->nx[i]);
    }
    float *w = (float *)malloc((size_t)len * m->ddim * sizeof(float));
    qt_conv_run(w, x, len, &m->d0);
    free(x);
    x = w;
    int ch = m->ddim;
    for (int i = 0; i < 4; i++) {
        const QtBlock *bk = &m->blk[i];
        const int r = m->rates[i], cout = bk->up.cout;
        qt_snake_run(x, len, &bk->sn);
        float *y = (float *)malloc((size_t)len * r * cout * sizeof(float));
        qt_convt_run(y, x, len, &bk->up, 2 * r);
        free(x);
        x = y; len *= r; ch = cout;
        float *t1 = (float *)malloc((size_t)len * ch * sizeof(float)), *t2 = (float *)malloc((size_t)len * ch * sizeof(float));
        if (!t1 || !t2) qt_die("out of memory");
        for (int u = 0; u < 3; u++) {                                 /* residual unit: snake, conv k7 (dilated), snake, conv k1, + input */
            const QtRes *rs = &bk->r[u];
            memcpy(t1, x, (size_t)len * ch * sizeof(float));
            qt_snake_run(t1, len, &rs->a1);
            qt_conv_run(t2, t1, len, &rs->c1);
            qt_snake_run(t2, len, &rs->a2);
            qt_conv_run(t1, t2, len, &rs->c2);
            #pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < (int64_t)len * ch; j++) x[j] += t1[j];
        }
        free(t1); free(t2);
    }
    qt_snake_run(x, len, &m->sn_out);
    float *wav = (float *)malloc((size_t)len * sizeof(float));
    qt_conv_run(wav, x, len, &m->d_out);
    free(x);
    for (int i = 0; i < len; i++) wav[i] = wav[i] < -1.f ? -1.f : wav[i] > 1.f ? 1.f : wav[i];
    *nsamples = len;
    return wav;
}

/* codes [T][16] -> wav, in chunks of 300 frames with 25 frames of left context (qwen-tts's chunked_decode); malloc'd, n samples */
static float *qt_codec_decode(const QtCodec *m, const int *codes, int T, int *nsamples){
    const int CHUNK = 300, CTX = 25, UP = 1920;
    float *out = (float *)malloc((size_t)(T > 0 ? T : 1) * UP * sizeof(float));
    int at = 0;
    for (int start = 0; start < T;) {
        const int end = start + CHUNK < T ? start + CHUNK : T;
        const int ctx = start - CTX > 0 ? CTX : start;
        int n = 0;
        float *w = qt_codec_chunk(m, codes + (size_t)(start - ctx) * QT_GROUPS, end - (start - ctx), &n);
        const int skip = ctx * UP, keep = n - skip;
        memcpy(out + at, w + skip, (size_t)keep * sizeof(float));
        at += keep;
        free(w);
        start = end;
    }
    *nsamples = at;
    return out;
}

/* ------------------------------------------------------------------------------------------------ files, the command line */

static float *qt_load_f32(const char *path, size_t *got){
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    float *b = (float *)malloc((size_t)sz);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f);
    *got = (size_t)sz / 4;
    return b;
}

static int qt_write_wav(const char *path, const float *w, int n, int rate){
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const uint32_t data = (uint32_t)n * 2, riff = 36 + data, sr = (uint32_t)rate, br = sr * 2;
    const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    const uint32_t sixteen = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&sixteen, 4, 1, f);
    fwrite(&fmt, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&sr, 4, 1, f); fwrite(&br, 4, 1, f); fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
    for (int i = 0; i < n; i++) {
        float v = w[i] * 32767.0f;
        int16_t s = (int16_t)(v < -32768.f ? -32768.f : v > 32767.f ? 32767.f : lrintf(v));
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv){
    const char *codec_dir = NULL, *codes_path = NULL, *out = NULL, *ref = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--codec") && i + 1 < argc) codec_dir = argv[++i];
        else if (!strcmp(argv[i], "--decode-codes") && i + 1 < argc) codes_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
#ifdef _OPENMP
            omp_set_num_threads(atoi(argv[++i]));
#else
            ++i;
#endif
        } else { fprintf(stderr, "usage: qwen3tts --codec DIR --decode-codes FILE.i32 [--out FILE.wav] [--ref REFDIR]\n"); return 2; }
    }
    if (!codec_dir || !codes_path) { fprintf(stderr, "usage: qwen3tts --codec DIR --decode-codes FILE.i32 [--out FILE.wav] [--ref REFDIR]\n"); return 2; }
    char err[512] = "";
    QtCodec *m = (QtCodec *)calloc(1, sizeof *m);
    double t0 = qt_now();
    if (qt_codec_load(m, codec_dir, err, sizeof err)) { fprintf(stderr, "qwen3tts: %s\n", err); return 1; }
    fprintf(stderr, "qwen3tts: codec decoder loaded: %.2f GB, %.1f s\n", (double)m->bytes / 1e9, qt_now() - t0);
    size_t nvals = 0;
    int *codes = (int *)qt_load_f32(codes_path, &nvals);
    if (!codes || nvals % QT_GROUPS) { fprintf(stderr, "qwen3tts: cannot read codes from %s\n", codes_path); return 1; }
    const int T = (int)(nvals / QT_GROUPS);
    for (size_t i = 0; i < nvals; i++) if (codes[i] < 0 || codes[i] >= m->cbsize) { fprintf(stderr, "qwen3tts: code %d out of range\n", codes[i]); return 1; }
    t0 = qt_now();
    int n = 0;
    float *wav = qt_codec_decode(m, codes, T, &n);
    fprintf(stderr, "qwen3tts: %d frames -> %d samples (%.2f s of audio) in %.2f s\n", T, n, (double)n / QT_SR, qt_now() - t0);
    if (out && qt_write_wav(out, wav, n, QT_SR)) { fprintf(stderr, "qwen3tts: cannot write %s\n", out); return 1; }
    if (ref) {
        char path[4300];
        snprintf(path, sizeof path, "%s/wav.f32", ref);
        size_t got = 0;
        float *r = qt_load_f32(path, &got);
        if (!r) { printf("no reference waveform in %s\n", ref); return 0; }
        if (got != (size_t)n) printf("waveform: reference has %zu samples, here %d\n", got, n);
        const size_t cmp = got < (size_t)n ? got : (size_t)n;
        double mx = 0, sum = 0, ref_abs = 0, err2 = 0, sig2 = 0;
        for (size_t i = 0; i < cmp; i++) {
            const double d = fabs((double)wav[i] - r[i]);
            if (d > mx) mx = d;
            sum += d; ref_abs += fabs(r[i]); err2 += d * d; sig2 += (double)r[i] * r[i];
        }
        printf("waveform   n=%zu  max abs diff %.3e  mean abs diff %.3e  (mean |ref| %.3e)  SNR %.1f dB\n", cmp, mx, sum / (double)(cmp ? cmp : 1), ref_abs / (double)(cmp ? cmp : 1),
               err2 > 0 ? 10.0 * log10(sig2 / err2) : 999.0);
    }
    return 0;
}
