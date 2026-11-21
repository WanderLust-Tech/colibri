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
 *   T2  the talker (a Qwen3 decoder over 1024) and the code predictor (5 layers): the prompt becomes embeddings, the talker writes
 *         the first code of each frame, the predictor the other 15; greedy or sampled (top-k 50, temperature 0.9, repetition penalty 1.05).
 *
 * The kernels and the Qwen3 layer helpers are the ones the ACE-Step and Qwen3-ASR engines use (qi_gemm.h, acestep_dit.h).
 *
 * Usage:
 *   qwen3tts --model DIR --text TEXT [--speaker Ryan] [--language English] [--out FILE.wav] [--greedy] [--ref REFDIR]
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
#include "omp_tune.h"
#ifdef COLI_VULKAN
#include "backend_vulkan.h"
#include "vk_chain.h"
#endif

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
    if (!f) { snprintf(err, errn, "cannot open %.400s", path); return -1; }
    fseek(f, 0, SEEK_END); long fn = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)fn + 1);
    if (!buf || fread(buf, 1, (size_t)fn, f) != (size_t)fn) { fclose(f); free(buf); snprintf(err, errn, "cannot read %.400s", path); return -1; }
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

#ifdef COLI_VULKAN
static float *qt_codec_vk(const QtCodec *m, const float *x, int len0, int *nsamples);
#endif

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
#ifdef COLI_VULKAN
    { float *wv = qt_codec_vk(m, x, len, nsamples); if (wv) { free(x); return wv; } }       /* the vocoder on the device */
#endif
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

/* ------------------------------------------------------------------------------------------------ one row at a time */

/* y [N] = W x (+ bias) for one input row: the decode steps of the talker and the code predictor read every weight once per token, so this is
 * memory-bound; the batch GEMM's panel packing costs more than it saves for M = 1. bf16 weights are widened a vector at a time. */
static void qt_mv(float *y, const float *x, const AcdLin *l){
    const QiMat *W = &l->m;
    const int N = W->N, K = W->K;
    if (W->fmt != QI_BF16) { qi_gemm(y, x, 1, W, l->b); return; }
    const int ld = W->ld ? W->ld : K;
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const uint16_t *w = (const uint16_t *)W->w + (size_t)n * ld;
        float s = 0.f;
        int k = 0;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        for (; k + 16 <= K; k += 16) {
            const __m256i r0 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(w + k)));
            const __m256i r1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(w + k + 8)));
            a0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(r0, 16)), _mm256_loadu_ps(x + k), a0);
            a1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(r1, 16)), _mm256_loadu_ps(x + k + 8), a1);
        }
        a0 = _mm256_add_ps(a0, a1);
        __m128 lo = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        s = _mm_cvtss_f32(lo);
#endif
        for (; k < K; k++) s += qi_bf16(w[k]) * x[k];
        y[n] = s + (l->b ? l->b[n] : 0.f);
    }
}

/* y [M][N] = x W^T: one row through qt_mv, more through the GEMM */
static void qt_lin_any(float *y, const float *x, int M, const AcdLin *l){
    if (M == 1) qt_mv(y, x, l); else acd_lin(y, x, M, l);
}

/* ------------------------------------------------------------------------------------------------ the language model side (T2) */

/* A plain Qwen3 stack (the talker, the code predictor): layers, final norm and the sizes. */
typedef struct { int nl, D, H, Hk, hd, inter; float eps, theta; AcdEncLayer *L; float *norm; } QtDec;

typedef struct {
    QtDec talker, pred;
    uint16_t *codec_emb, *text_emb, *pred_emb[QT_GROUPS - 1];     /* bf16 [3072][D], [151936][2048], 15 x [2048][D] */
    AcdLin fc1, fc2, codec_head, pred_head[QT_GROUPS - 1];
    AcdLin mtp;                                                   /* small_to_mtp_projection: talker width -> predictor width (the 1.7B) */
    int has_mtp, design, instruct_ok;                              /* the projection is there; VoiceDesign checkpoint; the model takes a style instruction */
    int vocab, text_vocab, text_hidden, cb_vocab;
    int tts_bos, tts_eos, tts_pad, im_start, im_end;
    int codec_bos, codec_eos, codec_pad, codec_think, codec_nothink, codec_think_bos, codec_think_eos;
    jval *cfg;                                                     /* kept for the speaker and language tables */
    Tok tok;
    QtCodec codec;
    size_t bytes;
} QtModel;

static int qt_dec_load(shards *S, const char *pfx, jval *c, int exp_nl, QtDec *d, size_t *bytes, char *err, size_t errn){
    d->nl = acd_json_int(c, "num_hidden_layers", 0); d->D = acd_json_int(c, "hidden_size", 0); d->H = acd_json_int(c, "num_attention_heads", 0);
    d->Hk = acd_json_int(c, "num_key_value_heads", 0); d->hd = acd_json_int(c, "head_dim", 128); d->inter = acd_json_int(c, "intermediate_size", 0);
    d->eps = (float)acd_json_num(c, "rms_norm_eps", 1e-6); d->theta = (float)acd_json_num(c, "rope_theta", 1e6);
    if (!d->nl || !d->D || !d->H || !d->Hk || !d->inter || (exp_nl && d->nl != exp_nl)) { snprintf(err, errn, "unsupported %s geometry in config.json", pfx); return -1; }
    char n[300];
    int64_t shd[1] = {d->D};
    d->L = (AcdEncLayer *)calloc((size_t)d->nl, sizeof *d->L);
    for (int i = 0; i < d->nl; i++) {
        char p[300];
        snprintf(p, sizeof p, "%s.layers.%d.self_attn", pfx, i);
        if (acd_load_attn(S, p, d->D, d->H, d->Hk, d->hd, &d->L[i].sa, bytes, err, errn)) return -1;
        snprintf(n, sizeof n, "%s.layers.%d.input_layernorm.weight", pfx, i);          if (!(d->L[i].ln1 = acd_read_f32(S, n, 1, shd, err, errn))) return -1;
        snprintf(n, sizeof n, "%s.layers.%d.post_attention_layernorm.weight", pfx, i); if (!(d->L[i].ln2 = acd_read_f32(S, n, 1, shd, err, errn))) return -1;
        snprintf(p, sizeof p, "%s.layers.%d.mlp", pfx, i);
        if (acd_load_mlp(S, p, d->D, d->inter, &d->L[i].gate, &d->L[i].up, &d->L[i].down, bytes, err, errn)) return -1;
    }
    snprintf(n, sizeof n, "%s.norm.weight", pfx);
    return (d->norm = acd_read_f32(S, n, 1, shd, err, errn)) ? 0 : -1;
}

static uint16_t *qt_bf16_table(shards *S, const char *name, int64_t numel, size_t *bytes, char *err, size_t errn){
    st_tensor *t = st_find(S, name);
    if (!t || t->dtype != 0 || t->numel != numel) { snprintf(err, errn, "tensor %s is missing or not bf16 of %lld values", name, (long long)numel); return NULL; }
    uint16_t *p = (uint16_t *)malloc((size_t)numel * 2);
    if (!p) { snprintf(err, errn, "out of memory reading %s", name); return NULL; }
    st_read_raw(S, name, p, 1);
    *bytes += (size_t)numel * 2;
    return p;
}

static int qt_json_id(jval *c, const char *key){ return acd_json_int(c, key, -1); }

static int qt_model_load(QtModel *m, const char *dir, char *err, size_t errn){
    memset(m, 0, sizeof *m);
    char path[4300];
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errn, "cannot open %.400s", path); return -1; }
    fseek(f, 0, SEEK_END); long fn = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)fn + 1);
    if (!buf || fread(buf, 1, (size_t)fn, f) != (size_t)fn) { fclose(f); free(buf); snprintf(err, errn, "cannot read %.400s", path); return -1; }
    fclose(f); buf[fn] = 0;
    m->cfg = json_parse_checked(buf);
    free(buf);
    jval *tc = m->cfg ? json_get(m->cfg, "talker_config") : NULL, *pc = tc ? json_get(tc, "code_predictor_config") : NULL;
    if (!tc || !pc) { snprintf(err, errn, "config.json has no talker_config.code_predictor_config: not a Qwen3-TTS checkpoint"); return -1; }
    jval *type = json_get(m->cfg, "tts_model_type");
    if (!type || type->t != J_STR || (strcmp(type->str, "custom_voice") && strcmp(type->str, "voice_design"))) {
        snprintf(err, errn, "only the CustomVoice and VoiceDesign checkpoints are supported so far (this one is %s)", type && type->t == J_STR ? type->str : "unknown"); return -1;
    }
    m->design = !strcmp(type->str, "voice_design");
    jval *msize = json_get(m->cfg, "tts_model_size");
    m->instruct_ok = m->design || (msize && msize->t == J_STR && strcmp(msize->str, "0b6"));                  /* the 0.6B ignores `instruct` */
    m->tts_bos = qt_json_id(m->cfg, "tts_bos_token_id"); m->tts_eos = qt_json_id(m->cfg, "tts_eos_token_id"); m->tts_pad = qt_json_id(m->cfg, "tts_pad_token_id");
    m->codec_bos = qt_json_id(tc, "codec_bos_id"); m->codec_eos = qt_json_id(tc, "codec_eos_token_id"); m->codec_pad = qt_json_id(tc, "codec_pad_id");
    m->codec_think = qt_json_id(tc, "codec_think_id"); m->codec_nothink = qt_json_id(tc, "codec_nothink_id");
    m->codec_think_bos = qt_json_id(tc, "codec_think_bos_id"); m->codec_think_eos = qt_json_id(tc, "codec_think_eos_id");
    m->vocab = acd_json_int(tc, "vocab_size", 0); m->text_vocab = acd_json_int(tc, "text_vocab_size", 0); m->text_hidden = acd_json_int(tc, "text_hidden_size", 0);
    if (m->tts_bos < 0 || m->tts_eos < 0 || m->tts_pad < 0 || m->codec_bos < 0 || m->codec_eos < 0 || m->codec_pad < 0 || !m->vocab || !m->text_vocab || !m->text_hidden ||
        acd_json_int(tc, "num_code_groups", 0) != QT_GROUPS) { snprintf(err, errn, "unsupported Qwen3-TTS geometry in config.json"); return -1; }
    shards S;
    st_init(&S, dir);
    if (qt_dec_load(&S, "talker.model", tc, 0, &m->talker, &m->bytes, err, errn)) return -1;
    if (qt_dec_load(&S, "talker.code_predictor.model", pc, 0, &m->pred, &m->bytes, err, errn)) return -1;
    const int D = m->talker.D;
    m->has_mtp = m->pred.D != D;
    if (m->has_mtp) {                                                                 /* the predictor is narrower: its inputs go through a projection with a bias */
        int64_t sm[2] = {m->pred.D, D};
        if (acd_load_lin(&S, "talker.code_predictor.small_to_mtp_projection.weight", "talker.code_predictor.small_to_mtp_projection.bias", m->pred.D, D, 2, sm, NULL, 0, 0, &m->mtp, &m->bytes, err, errn)) return -1;
    }
    if (!(m->codec_emb = qt_bf16_table(&S, "talker.model.codec_embedding.weight", (int64_t)m->vocab * D, &m->bytes, err, errn))) return -1;
    if (!(m->text_emb = qt_bf16_table(&S, "talker.model.text_embedding.weight", (int64_t)m->text_vocab * m->text_hidden, &m->bytes, err, errn))) return -1;
    char n[300], b[300];
    int64_t s1[2] = {m->text_hidden, m->text_hidden}, s2[2] = {D, m->text_hidden};
    if (acd_load_lin(&S, "talker.text_projection.linear_fc1.weight", "talker.text_projection.linear_fc1.bias", m->text_hidden, m->text_hidden, 2, s1, NULL, 0, 0, &m->fc1, &m->bytes, err, errn)) return -1;
    if (acd_load_lin(&S, "talker.text_projection.linear_fc2.weight", "talker.text_projection.linear_fc2.bias", D, m->text_hidden, 2, s2, NULL, 0, 0, &m->fc2, &m->bytes, err, errn)) return -1;
    int64_t sh[2] = {m->vocab, D};
    if (acd_load_lin(&S, "talker.codec_head.weight", NULL, m->vocab, D, 2, sh, NULL, 0, 0, &m->codec_head, &m->bytes, err, errn)) return -1;
    const int pv = acd_json_int(pc, "vocab_size", 2048);
    m->cb_vocab = pv;
    int64_t sp[2] = {pv, m->pred.D};
    for (int g = 0; g < QT_GROUPS - 1; g++) {
        snprintf(n, sizeof n, "talker.code_predictor.model.codec_embedding.%d.weight", g);
        if (!(m->pred_emb[g] = qt_bf16_table(&S, n, (int64_t)pv * D, &m->bytes, err, errn))) return -1;
        snprintf(n, sizeof n, "talker.code_predictor.lm_head.%d.weight", g); (void)b;
        if (acd_load_lin(&S, n, NULL, pv, m->pred.D, 2, sp, NULL, 0, 0, &m->pred_head[g], &m->bytes, err, errn)) return -1;
    }
    st_destroy(&S);
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    FILE *tf = fopen(path, "rb");
    if (!tf) { snprintf(err, errn, "no tokenizer.json in the model folder (python tools/qwen3_tokenizer.py DIR makes it from vocab.json and merges.txt)"); return -1; }
    fclose(tf);
    tok_load(&m->tok, path);
    m->im_start = tok_id_of(&m->tok, "<|im_start|>"); m->im_end = tok_id_of(&m->tok, "<|im_end|>");
    if (m->im_start < 0 || m->im_end < 0) { snprintf(err, errn, "the tokenizer lacks the chat tokens"); return -1; }
    snprintf(path, sizeof path, "%s/speech_tokenizer", dir);
    if (qt_codec_load(&m->codec, path, err, errn)) return -1;
    m->bytes += m->codec.bytes;
    return 0;
}

static inline void qt_row(const uint16_t *tab, int id, int n, float *dst){
    const uint16_t *r = tab + (size_t)id * n;
    for (int i = 0; i < n; i++) { uint32_t u = (uint32_t)r[i] << 16; memcpy(&dst[i], &u, 4); }
}

/* text rows through text_projection: x [n][text_hidden] -> out [n][D] (fc1, SiLU, fc2) */
static void qt_text_proj(const QtModel *m, const float *x, int n, float *out){
    float *h = (float *)malloc((size_t)n * m->text_hidden * sizeof(float));
    acd_lin(h, x, n, &m->fc1);
    for (int64_t i = 0; i < (int64_t)n * m->text_hidden; i++) h[i] = acd_silu(h[i]);
    acd_lin(out, h, n, &m->fc2);
    free(h);
}

/* the projected embeddings of text token ids: [n][D] */
static float *qt_text_embed(const QtModel *m, const int *ids, int n){
    float *x = (float *)malloc((size_t)(n ? n : 1) * m->text_hidden * sizeof(float)), *out = (float *)malloc((size_t)(n ? n : 1) * m->talker.D * sizeof(float));
    for (int i = 0; i < n; i++) qt_row(m->text_emb, ids[i], m->text_hidden, x + (size_t)i * m->text_hidden);
    qt_text_proj(m, x, n, out);
    free(x);
    return out;
}

static int qt_lookup(jval *obj, const char *key, int fallback){
    jval *v = obj ? json_get(obj, key) : NULL;
    return v && v->t == J_NUM ? (int)v->num : fallback;
}

static void qt_lower(char *s){ for (; *s; s++) if (*s >= 'A' && *s <= 'Z') *s += 32; }

/* The token ids of a chat-framed text: fmt is a printf format with one %s, malloc'd, count in *n. */
static int *qt_frame_ids(const QtModel *m_, const char *fmt, const char *text, int *n){
    QtModel *m = (QtModel *)m_;
    const size_t len = strlen(text) + strlen(fmt) + 16;
    char *s = (char *)malloc(len);
    snprintf(s, len, fmt, text);
    char *nrm = NULL; size_t nl = 0;
    if (q38_nfc_normalize(s, strlen(s), &nrm, &nl) != 0) { nrm = strdup(s); nl = strlen(s); }
    free(s);
    int cap = (int)nl + 16;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    *n = tok_encode(&m->tok, nrm, (int)nl, ids, cap);
    free(nrm);
    return ids;
}

/* The text prompt's token ids: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n, malloc'd, count in *n. */
static int *qt_text_ids(const QtModel *m, const char *text, int *n){
    return qt_frame_ids(m, "<|im_start|>assistant\n%s<|im_end|>\n<|im_start|>assistant\n", text, n);
}

/* The talker's prefill input for a request in the reference's non-streaming mode: [L][D] malloc'd, L in *rows. A style instruction (or, for
 * VoiceDesign, the description of the voice) goes first as <|im_start|>user\n{instruct}<|im_end|>\n through the text embedding; no speaker (an
 * empty name) leaves the speaker's row out. Returns NULL with err set for an unknown speaker or language or an instruction the model cannot take. */
static float *qt_prefill(const QtModel *m, const int *ids, int nids, const char *speaker, const char *instruct, const char *language, int *rows, char *err, size_t errn){
    const int D = m->talker.D;
    int ninst = 0, *inst_ids = NULL;
    if (instruct && *instruct && !m->instruct_ok) { snprintf(err, errn, "this model takes no style instruction (the 1.7B CustomVoice and VoiceDesign do)"); return NULL; }
    jval *tc = json_get(m->cfg, "talker_config");
    int spk = -1, lang = -1;
    char name[64] = "", lg[64] = "";
    snprintf(name, sizeof name, "%s", speaker ? speaker : "");
    snprintf(lg, sizeof lg, "%s", language && *language ? language : "auto");
    qt_lower(name); qt_lower(lg);
    if (m->design) name[0] = 0;                                                       /* VoiceDesign has no speakers: the instruction makes the voice */
    if (*name) {
        spk = qt_lookup(json_get(tc, "spk_id"), name, -1);
        if (spk < 0) { snprintf(err, errn, "the speaker `%s` is not one this model has", speaker); return NULL; }
    }
    if (strcmp(lg, "auto")) {
        lang = qt_lookup(json_get(tc, "codec_language_id"), lg, -1);
        if (lang < 0) { snprintf(err, errn, "the language `%s` is not one this model speaks", language); return NULL; }
    }
    if (*name && (!strcmp(lg, "chinese") || !strcmp(lg, "auto"))) {                  /* a dialect speaker (Dylan, Eric) swaps in the dialect's language id */
        jval *dia = json_get(json_get(tc, "spk_is_dialect"), name);
        if (dia && dia->t == J_STR) { const int d = qt_lookup(json_get(tc, "codec_language_id"), dia->str, -1); if (d >= 0) lang = d; }
    }
    if (nids < 9) { snprintf(err, errn, "the prompt is too short"); return NULL; }
    int prefix[8], np = 0;
    if (lang < 0) { prefix[np++] = m->codec_nothink; prefix[np++] = m->codec_think_bos; prefix[np++] = m->codec_think_eos; }
    else { prefix[np++] = m->codec_think; prefix[np++] = m->codec_think_bos; prefix[np++] = lang; prefix[np++] = m->codec_think_eos; }
    if (instruct && *instruct) inst_ids = qt_frame_ids(m, "<|im_start|>user\n%s<|im_end|>\n", instruct, &ninst);
    const int ntext = nids - 8;                                                       /* ids[3 .. nids - 5) */
    const int total = ninst + 3 + (np + (spk >= 0) + 2 - 1) + (ntext + 1) + 1;
    float *out = (float *)calloc((size_t)total * D, sizeof(float));
    float *tmp = (float *)malloc((size_t)D * sizeof(float));
    const int sp[3] = {m->tts_bos, m->tts_eos, m->tts_pad};
    float *sv = qt_text_embed(m, sp, 3);                                              /* tts_bos, tts_eos, tts_pad, projected */
    const float *bos = sv, *eos = sv + D, *pad = sv + 2 * D;
    int at = 0;
    if (ninst) {                                                                      /* the instruction first */
        float *ie = qt_text_embed(m, inst_ids, ninst);
        memcpy(out, ie, (size_t)ninst * D * sizeof(float)); at = ninst;
        free(ie); free(inst_ids);
    }
    float *role = qt_text_embed(m, ids, 3);
    memcpy(out + (size_t)at * D, role, (size_t)3 * D * sizeof(float)); at += 3;
    free(role);
    int codec[12], nc = 0;                                                            /* the codec ids of the prefix, the speaker and [pad, bos] */
    for (int i = 0; i < np; i++) codec[nc++] = prefix[i];
    if (spk >= 0) codec[nc++] = spk;
    codec[nc++] = m->codec_pad; codec[nc++] = m->codec_bos;
    for (int i = 0; i < nc - 1; i++) {                                                /* (tts_pad x (nc - 2), tts_bos) + codec[:-1] */
        const float *t = i < nc - 2 ? pad : bos;
        qt_row(m->codec_emb, codec[i], D, tmp);
        float *dst = out + (size_t)at * D;
        for (int j = 0; j < D; j++) dst[j] = t[j] + tmp[j];
        at++;
    }
    float *txt = qt_text_embed(m, ids + 3, ntext);                                    /* the text, then tts_eos, each + codec_pad */
    qt_row(m->codec_emb, m->codec_pad, D, tmp);
    for (int i = 0; i < ntext + 1; i++) {
        const float *t = i < ntext ? txt + (size_t)i * D : eos;
        float *dst = out + (size_t)at * D;
        for (int j = 0; j < D; j++) dst[j] = t[j] + tmp[j];
        at++;
    }
    qt_row(m->codec_emb, m->codec_bos, D, tmp);                                       /* tts_pad + codec_bos */
    {
        float *dst = out + (size_t)at * D;
        for (int j = 0; j < D; j++) dst[j] = pad[j] + tmp[j];
        at++;
    }
    free(txt); free(sv); free(tmp);
    if (at != total) { snprintf(err, errn, "internal: prefill length %d, expected %d", at, total); free(out); return NULL; }
    *rows = total;
    return out;
}

typedef struct { float **k, **v; int cap, len; } QtCache;

static QtCache *qt_cache_new(const QtDec *d, int cap){
    QtCache *c = (QtCache *)calloc(1, sizeof *c);
    c->cap = cap; c->len = 0;
    c->k = (float **)calloc((size_t)d->nl, sizeof(float *)); c->v = (float **)calloc((size_t)d->nl, sizeof(float *));
    for (int i = 0; i < d->nl; i++) { c->k[i] = (float *)malloc((size_t)cap * d->Hk * d->hd * sizeof(float)); c->v[i] = (float *)malloc((size_t)cap * d->Hk * d->hd * sizeof(float)); }
    return c;
}
static void qt_cache_free(const QtDec *d, QtCache *c){
    for (int i = 0; i < d->nl; i++) { free(c->k[i]); free(c->v[i]); }
    free(c->k); free(c->v); free(c);
}

/* n rows x [n][D] at positions pos0.. through every layer (no final norm), keys and values appended to the cache; causal, seeing the cache first */
static void qt_dec_rows(const QtDec *d, QtCache *C, float *x, int n, int pos0, const float *cs, const float *sn){
    const int D = d->D, H = d->H, Hk = d->Hk, hd = d->hd, I = d->inter, group = H / Hk;
    float *nb = (float *)malloc((size_t)n * D * sizeof(float)), *o = (float *)malloc((size_t)n * D * sizeof(float));
    float *q = (float *)malloc((size_t)n * H * hd * sizeof(float)), *cx = (float *)malloc((size_t)n * H * hd * sizeof(float));
    float *kk = (float *)malloc((size_t)n * Hk * hd * sizeof(float)), *vv = (float *)malloc((size_t)n * Hk * hd * sizeof(float));
    float *g = (float *)malloc((size_t)n * I * sizeof(float)), *u = (float *)malloc((size_t)n * I * sizeof(float));
    const size_t scn = (size_t)(C->cap > 1 ? C->cap : 1);
    float *sc_all = (float *)malloc((size_t)H * scn * sizeof(float));                  /* one score row per head, so the heads run side by side */
    for (int li = 0; li < d->nl; li++) {
        const AcdEncLayer *l = &d->L[li];
        acd_rmsnorm(nb, x, n, D, l->ln1, d->eps);
        qt_lin_any(q, nb, n, &l->sa.q); qt_lin_any(kk, nb, n, &l->sa.k); qt_lin_any(vv, nb, n, &l->sa.v);
        acd_head_norm(q, n, H, hd, l->sa.qn, d->eps);
        acd_head_norm(kk, n, Hk, hd, l->sa.kn, d->eps);
        acd_rope(q, n, H, hd, cs + (size_t)pos0 * hd, sn + (size_t)pos0 * hd);
        acd_rope(kk, n, Hk, hd, cs + (size_t)pos0 * hd, sn + (size_t)pos0 * hd);
        memcpy(C->k[li] + (size_t)pos0 * Hk * hd, kk, (size_t)n * Hk * hd * sizeof(float));
        memcpy(C->v[li] + (size_t)pos0 * Hk * hd, vv, (size_t)n * Hk * hd * sizeof(float));
        if (pos0 == 0 && n > 1) {
            acd_attention_ex(cx, H * hd, q, H * hd, n, C->k[li], C->v[li], Hk * hd, n, H, Hk, hd, -1, 1);
        } else {
            const float scale = 1.0f / sqrtf((float)hd);
            for (int r = 0; r < n; r++) {
                const int upto = pos0 + r + 1;
                #pragma omp parallel for schedule(static)
                for (int h = 0; h < H; h++) {
                    float *sc = sc_all + (size_t)h * scn;
                    const float *qh = q + (size_t)r * H * hd + (size_t)h * hd;
                    const float *K = C->k[li] + (size_t)(h / group) * hd, *V = C->v[li] + (size_t)(h / group) * hd;
                    float mx = -1e30f;
                    for (int j = 0; j < upto; j++) {
                        const float *kj = K + (size_t)j * Hk * hd;
                        float s = 0;
                        for (int e = 0; e < hd; e++) s += qh[e] * kj[e];
                        sc[j] = s * scale;
                        if (sc[j] > mx) mx = sc[j];
                    }
                    float den = 0;
                    /* a weight under 1e-30 changes nothing, and as a denormal it slows every multiply after it by a hundredfold */
                    for (int j = 0; j < upto; j++) { const float e = expf(sc[j] - mx); sc[j] = e < 1e-30f ? 0.f : e; den += sc[j]; }
                    float *oh = cx + (size_t)r * H * hd + (size_t)h * hd;
                    for (int e = 0; e < hd; e++) oh[e] = 0;
                    for (int j = 0; j < upto; j++) {
                        if (sc[j] == 0.f) continue;
                        const float w = sc[j] / den;
                        const float *vj = V + (size_t)j * Hk * hd;
                        for (int e = 0; e < hd; e++) oh[e] += w * vj[e];
                    }
                }
            }
        }
        qt_lin_any(o, cx, n, &l->sa.o);
        for (int64_t i = 0; i < (int64_t)n * D; i++) x[i] += o[i];
        acd_rmsnorm(nb, x, n, D, l->ln2, d->eps);
        qt_lin_any(g, nb, n, &l->gate); qt_lin_any(u, nb, n, &l->up);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n * I; i++) g[i] = acd_silu(g[i]) * u[i];
        qt_lin_any(o, g, n, &l->down);
        for (int64_t i = 0; i < (int64_t)n * D; i++) x[i] += o[i];
    }
    C->len = pos0 + n;
    free(nb); free(o); free(q); free(cx); free(kk); free(vv); free(g); free(u); free(sc_all);
}

#ifdef COLI_VULKAN
/* ------------------------------------------------------------------------------------------------ the decode steps on a Vulkan device
 * A frame is one talker step and sixteen code predictor steps, each a row against every weight. On the CPU that is memory-bound and
 * spends its time at the barriers between small matrices; on the card each step is one submission of the dense chain (norms, q/k/v, head
 * norms, RoPE, attention over a KV cache that lives on the device, o, SwiGLU, down) and only the logits and the hidden row come back.
 * The prompt rows go through the same chain. The device holds the matrices as bf16 and computes in f32, so a logit differs from the
 * CPU's in the last bits; a device that is missing, or a frame that fails, leaves the CPU's path in charge. */
static struct { int ready, chain; } qt_vk;

static void qt_vk_open(void){
    memset(&qt_vk, 0, sizeof qt_vk);
    if (!coli_vk_init_env("qwen3tts")) return;               /* COLI_VULKAN unset, or no device: the CPU */
    qt_vk.ready = 1;
    atexit(coli_vk_shutdown);
    const char *c = getenv("COLI_VK_CHAIN");
    if (!(c && !strcmp(c, "0")) && vkc_init() && vkc_ready()) { qt_vk.chain = 1; atexit(vkc_shutdown); }   /* registered after the backend's: runs before it */
}

typedef struct {
    int tried, ok, nl, D, H, Hk, hd, I, nheads;
    float eps;
    ColiVkTensor **t;                       /* [nl][7]: q k v o gate up down */
    ColiVkTensor **head;                    /* the output matrices (the talker's codec head; the predictor's fifteen) */
    VkcBuf *prm;                            /* per layer: ln1 [D], ln2 [D], q norm [hd], k norm [hd]; then the final norm [D] */
    VkcBuf *cs, *kc, *vc, *xu, *x, *nrm, *q, *k, *v, *ctx, *g, *u, *tmp, *lg, *out;
    size_t cap, rows, nlog;
    unsigned long long steps;
} QtVk;

static ColiVkTensor *qt_vk_tensor(const AcdLin *l){
    const QiMat *W = &l->m;
    if (l->b || W->ld || !W->w || (W->fmt != QI_F32 && W->fmt != QI_BF16) || W->K % 2) return NULL;
    ColiVkTensor *t = NULL;
    return coli_vk_tensor_ensure(&t, W->w, NULL, W->fmt == QI_BF16 ? 11 : 10, W->K, W->N, 0) ? t : NULL;
}

/* The matrices and norm weights of a stack go up once (the first call; 0 when anything would not). */
static int qt_vk_setup(QtVk *v, const QtDec *d, const AcdLin *heads, int nheads){
    if (v->tried) return v->ok;
    v->tried = 1;
    if (!qt_vk.chain || d->hd > 256 || d->H % d->Hk) return 0;
    v->nl = d->nl; v->D = d->D; v->H = d->H; v->Hk = d->Hk; v->hd = d->hd; v->I = d->inter; v->eps = d->eps; v->nheads = nheads;
    v->t = (ColiVkTensor **)calloc((size_t)d->nl * 7, sizeof(ColiVkTensor *));
    v->head = (ColiVkTensor **)calloc((size_t)nheads, sizeof(ColiVkTensor *));
    const size_t per = (size_t)2 * d->D + 2 * d->hd, total = (size_t)d->nl * per + d->D;
    float *arena = (float *)calloc(total, sizeof(float));
    if (!v->t || !v->head || !arena) { free(arena); return 0; }
    for (int l = 0; l < d->nl; l++) {
        const AcdEncLayer *L = &d->L[l];
        const AcdLin *mats[7] = { &L->sa.q, &L->sa.k, &L->sa.v, &L->sa.o, &L->gate, &L->up, &L->down };
        for (int i = 0; i < 7; i++) {
            v->t[l * 7 + i] = qt_vk_tensor(mats[i]);
            if (!v->t[l * 7 + i]) { fprintf(stderr, "[VK] qwen3tts: a matrix of layer %d did not reach the device; the CPU runs the steps\n", l); free(arena); return 0; }
        }
        memcpy(arena + l * per, L->ln1, (size_t)d->D * sizeof(float));
        memcpy(arena + l * per + d->D, L->ln2, (size_t)d->D * sizeof(float));
        memcpy(arena + l * per + 2 * d->D, L->sa.qn, (size_t)d->hd * sizeof(float));
        memcpy(arena + l * per + 2 * d->D + d->hd, L->sa.kn, (size_t)d->hd * sizeof(float));
    }
    memcpy(arena + (size_t)d->nl * per, d->norm, (size_t)d->D * sizeof(float));
    for (int i = 0; i < nheads; i++) {
        v->head[i] = qt_vk_tensor(&heads[i]);
        if (!v->head[i]) { fprintf(stderr, "[VK] qwen3tts: an output matrix did not reach the device; the CPU runs the steps\n"); free(arena); return 0; }
        if ((size_t)heads[i].m.N > v->nlog) v->nlog = (size_t)heads[i].m.N;
    }
    v->prm = vkc_buf(total * sizeof(float), VKC_DEV);
    int ok = v->prm && vkc_begin() && vkc_write(v->prm, 0, arena, total * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) return 0;
    v->ok = 1;
    return 1;
}

/* The buffers for `rows` rows a step and `cap` positions of cache; the rope table (cos, sin pairs) goes up with them. */
static int qt_vk_reserve(QtVk *v, size_t rows, size_t cap, const float *cs, const float *sn){
    const size_t D = v->D, rq = (size_t)v->H * v->hd, rk = (size_t)v->Hk * v->hd, hd = v->hd;
    if (!vkc_reserve(&v->xu, rows * D * 4, VKC_UP) || !vkc_reserve(&v->x, rows * D * 4, VKC_DEV) || !vkc_reserve(&v->nrm, rows * D * 4, VKC_DEV) ||
        !vkc_reserve(&v->tmp, rows * D * 4, VKC_DEV) || !vkc_reserve(&v->q, rows * rq * 4, VKC_DEV) || !vkc_reserve(&v->ctx, rows * rq * 4, VKC_DEV) ||
        !vkc_reserve(&v->k, rows * rk * 4, VKC_DEV) || !vkc_reserve(&v->v, rows * rk * 4, VKC_DEV) || !vkc_reserve(&v->g, rows * v->I * 4, VKC_DEV) ||
        !vkc_reserve(&v->u, rows * v->I * 4, VKC_DEV) || !vkc_reserve(&v->lg, v->nlog * 4, VKC_DEV) || !vkc_reserve(&v->out, (D + v->nlog) * 4, VKC_DOWN) ||
        !vkc_reserve(&v->kc, (size_t)v->nl * cap * rk * 4, VKC_DEV) || !vkc_reserve(&v->vc, (size_t)v->nl * cap * rk * 4, VKC_DEV) ||
        !vkc_reserve(&v->cs, cap * hd * 4, VKC_DEV))
        return 0;
    float *tab = (float *)malloc(cap * hd * sizeof(float));
    if (!tab) return 0;
    for (size_t r = 0; r < cap; r++)
        for (size_t j = 0; j < hd / 2; j++) { tab[r * hd + 2 * j] = cs[r * hd + j]; tab[r * hd + 2 * j + 1] = sn[r * hd + j]; }
    const int ok = vkc_begin() && vkc_write(v->cs, 0, tab, cap * hd * sizeof(float)) && vkc_submit(1);
    free(tab);
    v->cap = cap; v->rows = rows;
    return ok;
}

/* n rows (host, [n][D]) at positions pos0.. through every layer, their keys and values appended to the device cache; then the last row
 * through the final norm (hn, [D], when asked) and the output matrix `head` (logits [N], when asked). 1 on success; 0 leaves the CPU to run it. */
static int qt_vk_run(QtVk *v, const float *xrows, int n, int pos0, int head, float *hn, float *logits, int nlogits){
    if (!v->ok || (size_t)n > v->rows || (size_t)(pos0 + n) > v->cap) return 0;
    const int D = v->D, H = v->H, Hk = v->Hk, hd = v->hd, I = v->I;
    const size_t rq = (size_t)H * hd, rk = (size_t)Hk * hd, per = (size_t)2 * D + 2 * hd;
    const float scale = 1.f / sqrtf((float)hd);
    memcpy(vkc_ptr(v->xu), xrows, (size_t)n * D * sizeof(float));
    int ok = vkc_begin() && vkc_copy(v->x, 0, v->xu, 0, (size_t)n * D);
    for (int l = 0; l < v->nl && ok; l++) {
        ColiVkTensor **t = v->t + (size_t)l * 7;
        const int wb = (int)(l * per);
        const size_t kbase = (size_t)l * v->cap * rk;
        VkcNorm n1 = {n, D, 1, 0, D, D, 0, D, D, wb, 0, 0, v->eps, 1.f};
        VkcNorm qn = {n * H, hd, H, 0, H * hd, hd, 0, H * hd, hd, wb + 2 * D, 0, 0, v->eps, 1.f};
        VkcNorm kn = {n * Hk, hd, Hk, 0, Hk * hd, hd, 0, Hk * hd, hd, wb + 2 * D + hd, 0, 0, v->eps, 1.f};
        VkcRope rq_ = {n * H, H, 0, H * hd, hd, hd / 2, pos0 * hd, hd}, rk_ = {n * Hk, Hk, 0, Hk * hd, hd, hd / 2, pos0 * hd, hd};
        ok = vkc_norm(v->x, v->prm, v->nrm, &n1) &&
             vkc_matmul(t[0], v->nrm, 0, v->q, 0, n) && vkc_matmul(t[1], v->nrm, 0, v->k, 0, n) && vkc_matmul(t[2], v->nrm, 0, v->v, 0, n) &&
             vkc_norm(v->q, v->prm, v->q, &qn) && vkc_norm(v->k, v->prm, v->k, &kn) && vkc_rope(v->q, v->cs, &rq_) && vkc_rope(v->k, v->cs, &rk_) &&
             vkc_copy(v->kc, kbase + (size_t)pos0 * rk, v->k, 0, (size_t)n * rk) && vkc_copy(v->vc, kbase + (size_t)pos0 * rk, v->v, 0, (size_t)n * rk);
        if (ok) {
            VkcAttnW a;
            memset(&a, 0, sizeof a);
            a.a = (VkcAttn){.S = n, .H = H, .KVH = Hk, .hd = hd, .pos_base = pos0, .cap = pos0 + n, .q_off = 0, .q_row = (int)rq, .q_seg = hd,
                            .o_off = 0, .o_row = (int)rq, .scale = scale, .k_off = (int)kbase, .v_off = (int)kbase};
            a.vd = hd; a.kv_pm = 1;
            ok = vkc_attn_w(v->q, v->kc, v->vc, v->ctx, NULL, NULL, NULL, &a);
        }
        VkcNorm n2 = {n, D, 1, 0, D, D, 0, D, D, wb + D, 0, 0, v->eps, 1.f};
        VkcEw add = {VKC_EW_ADD, n * D, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f}, act = {VKC_EW_SWIGLU, n * I, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        ok = ok && vkc_matmul(t[3], v->ctx, 0, v->tmp, 0, n) && vkc_ew(v->x, v->x, v->tmp, NULL, NULL, &add) &&
             vkc_norm(v->x, v->prm, v->nrm, &n2) && vkc_matmul(t[4], v->nrm, 0, v->g, 0, n) && vkc_matmul(t[5], v->nrm, 0, v->u, 0, n) &&
             vkc_ew(v->g, v->g, v->u, NULL, NULL, &act) && vkc_matmul(t[6], v->g, 0, v->tmp, 0, n) && vkc_ew(v->x, v->x, v->tmp, NULL, NULL, &add);
    }
    if (ok) {                                                           /* the last row: final norm, then the output matrix */
        VkcNorm fn = {1, D, 1, (n - 1) * D, D, D, 0, D, D, (int)(v->nl * per), 0, 0, v->eps, 1.f};
        ok = vkc_norm(v->x, v->prm, v->nrm, &fn) && vkc_copy(v->out, 0, v->nrm, 0, (size_t)D);
        if (ok && logits) ok = vkc_matmul(v->head[head], v->nrm, 0, v->lg, 0, 1) && vkc_copy(v->out, (size_t)D, v->lg, 0, (size_t)nlogits);
    }
    ok = ok && vkc_submit(1);
    if (!ok) {
        fprintf(stderr, "[VK] qwen3tts: a frame failed; the CPU runs the steps from here on\n");
        v->ok = 0;
        if (vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); qt_vk.ready = 0; qt_vk.chain = 0; }
        return 0;
    }
    const float *o = (const float *)vkc_ptr(v->out);
    if (hn) memcpy(hn, o, (size_t)D * sizeof(float));
    if (logits) memcpy(logits, o + D, (size_t)nlogits * sizeof(float));
    v->steps++;
    return 1;
}

static QtVk qt_vk_talker, qt_vk_pred;

/* ------------------------------------------------------------------------------------------------ the codec's vocoder on the device
 * The codec is two parts: the quantizer, the pre-conv, the transformer and the first two upsamplings (about a tenth of the time, on the
 * CPU) and the vocoder, a convolutional stack that turns 4 rows a frame into 1920 samples: that is where the time goes (0.9 s of 1.05 s
 * for 5 s of speech). On the card each causal convolution is an im2col (chain_codec, mode 0) and a GEMM over the resident weights,
 * the SnakeBeta, the bias, the transposed convolution's overlap-add and the clamp are chain_codec modes, and a chunk is one submission. */
typedef struct { ColiVkTensor *t; int off; } QcvW;               /* a convolution's resident weights and its bias (or parameters) in tab, -1: none */
typedef struct {
    int tried, ok;
    QcvW d0, d_out, up[4], c1[4][3], c2[4][3];
    int sn[4], a1[4][3], a2[4][3], sn_out, d_out_w;
    VkcBuf *tab, *buf[3], *tmp, *col, *xu, *out;
    size_t cap_act, cap_tmp, cap_in, cap_out;
} QtCodecVk;
static QtCodecVk qt_cvk;

typedef struct { float *p; size_t n, cap; } QcvArena;
static int qcv_add(QcvArena *a, const float *v, size_t n){
    if (a->n + n > a->cap) {
        size_t c = a->cap ? a->cap * 2 : 1 << 16;
        while (c < a->n + n) c *= 2;
        float *g = (float *)realloc(a->p, c * sizeof(float));
        if (!g) return -1;
        a->p = g; a->cap = c;
    }
    memcpy(a->p + a->n, v, n * sizeof(float));
    a->n += n;
    return (int)(a->n - n);
}

static int qcv_conv_w(QcvW *w, QcvArena *a, const QtConv *c){
    w->t = qt_vk_tensor(&c->w);
    w->off = c->b ? qcv_add(a, c->b, (size_t)c->cout) : -1;
    return w->t != NULL && (!c->b || w->off >= 0);
}
static int qcv_snake_w(int *off, QcvArena *a, const QtSnake *s){
    *off = qcv_add(a, s->ea, (size_t)s->c);
    return *off >= 0 && qcv_add(a, s->ib, (size_t)s->c) >= 0;
}

static int qt_codec_vk_setup(const QtCodec *m){
    QtCodecVk *v = &qt_cvk;
    if (v->tried) return v->ok;
    v->tried = 1;
    if (!qt_vk.chain || !vkc_codec_ready()) return 0;
    QcvArena a = {0};
    int ok = qcv_conv_w(&v->d0, &a, &m->d0) && m->d_out.cout == 1 && m->d_out.b && qcv_snake_w(&v->sn_out, &a, &m->sn_out);
    if (ok) {                                                      /* the last layer reads its taps straight from tab */
        v->d_out.off = qcv_add(&a, m->d_out.b, 1);
        v->d_out_w = qcv_add(&a, (const float *)m->d_out.w.m.w, (size_t)m->d_out.K * m->d_out.cin);
        ok = v->d_out.off >= 0 && v->d_out_w >= 0;
    }
    for (int i = 0; ok && i < 4; i++) {
        const QtBlock *b = &m->blk[i];
        v->up[i].t = qt_vk_tensor(&b->up.w);
        v->up[i].off = qcv_add(&a, b->up.b, (size_t)b->up.cout);
        ok = v->up[i].t && v->up[i].off >= 0 && qcv_snake_w(&v->sn[i], &a, &b->sn);
        for (int u = 0; ok && u < 3; u++)
            ok = qcv_conv_w(&v->c1[i][u], &a, &b->r[u].c1) && qcv_conv_w(&v->c2[i][u], &a, &b->r[u].c2) &&
                 qcv_snake_w(&v->a1[i][u], &a, &b->r[u].a1) && qcv_snake_w(&v->a2[i][u], &a, &b->r[u].a2);
    }
    if (ok) {
        v->tab = vkc_buf(a.n * sizeof(float), VKC_DEV);
        ok = v->tab && vkc_begin() && vkc_write(v->tab, 0, a.p, a.n * sizeof(float)) && vkc_submit(1);
    }
    free(a.p);
    if (!ok) fprintf(stderr, "[VK] qwen3tts: the codec's weights did not reach the device; the CPU runs the vocoder\n");
    v->ok = ok;
    return ok;
}

/* y [T][cout] = causal conv of x [T][cin] (+ bias): im2col in tiles of at most 16M floats, each through the GEMM */
static int qcv_conv(const QtCodecVk *v, const QcvW *w, const QtConv *c, VkcBuf *x, VkcBuf *y, int T){
    const int K = c->K, cin = c->cin, cout = c->cout, kc = K * cin;
    int ok = 1;
    if (K == 1) {                                                   /* no taps to gather: the rows go straight in, in tiles of 32768 */
        for (int t0 = 0; ok && t0 < T; t0 += 32768) {
            const int tn = T - t0 < 32768 ? T - t0 : 32768;
            ok = vkc_matmul(w->t, x, (size_t)t0 * cin, y, (size_t)t0 * cout, tn);
        }
    } else {
        int tile = (int)((size_t)(16u << 20) / kc);
        if (tile < 256) tile = 256;
        if (tile > T) tile = T;
        for (int t0 = 0; ok && t0 < T; t0 += tile) {
            const int tn = T - t0 < tile ? T - t0 : tile;
            VkcCodec im = {0, cin, K, c->dil, T, t0, 0, 0, tn * kc, 0, 0, 0};
            ok = vkc_codec(x, v->col, v->tab, &im) && vkc_matmul(w->t, v->col, 0, y, (size_t)t0 * cout, tn);
        }
    }
    if (ok && w->off >= 0) { VkcCodec bi = {3, cout, 0, 0, T, 0, 0, 0, T * cout, w->off, 0, 0}; ok = vkc_codec(v->tab, y, v->tab, &bi); }
    return ok;
}

static int qcv_snake(const QtCodecVk *v, int off, VkcBuf *x, VkcBuf *y, int T, int C){
    VkcCodec sn = {1, C, 0, 0, T, 0, 0, 0, T * C, off, 0, 0};
    return vkc_codec(x, y, v->tab, &sn);
}

/* x [len][latent] (host, after the second ConvNeXt) -> the waveform [len * 480] (malloc'd), or NULL: the caller runs the CPU */
static float *qt_codec_vk(const QtCodec *m, const float *x, int len0, int *nsamples){
    if (!qt_codec_vk_setup(m)) return NULL;
    QtCodecVk *v = &qt_cvk;
    size_t act = (size_t)len0 * m->latent, tmp = 0;
    int len = len0, ch = m->ddim;
    if ((size_t)len * ch > act) act = (size_t)len * ch;
    for (int i = 0; i < 4; i++) {
        const int r = m->rates[i], cout = m->blk[i].up.cout;
        if ((size_t)len * cout * 2 * r > tmp) tmp = (size_t)len * cout * 2 * r;
        len *= r; ch = cout;
        if ((size_t)len * ch > act) act = (size_t)len * ch;
    }
    const int total = len;
    if (!vkc_reserve(&v->buf[0], act * 4, VKC_DEV) || !vkc_reserve(&v->buf[1], act * 4, VKC_DEV) || !vkc_reserve(&v->buf[2], act * 4, VKC_DEV) ||
        !vkc_reserve(&v->tmp, tmp * 4, VKC_DEV) || !vkc_reserve(&v->col, (size_t)(16u << 20) * 4, VKC_DEV) ||
        !vkc_reserve(&v->xu, (size_t)len0 * m->latent * 4, VKC_UP) || !vkc_reserve(&v->out, (size_t)total * 4, VKC_DOWN)) {
        fprintf(stderr, "[VK] qwen3tts: device memory for the vocoder refused; the CPU runs it\n");
        return NULL;
    }
    memcpy(vkc_ptr(v->xu), x, (size_t)len0 * m->latent * sizeof(float));
    VkcBuf *A = v->buf[0], *B = v->buf[1], *C = v->buf[2], *cur, *s1, *s2;
    len = len0;
    int ok = vkc_begin() && vkc_copy(A, 0, v->xu, 0, (size_t)len0 * m->latent) && qcv_conv(v, &v->d0, &m->d0, A, B, len);
    cur = B; s1 = A; s2 = C;
    ch = m->ddim;
    for (int i = 0; ok && i < 4; i++) {
        const QtBlock *bk = &m->blk[i];
        const int r = m->rates[i], cout = bk->up.cout, K = 2 * r;
        ok = qcv_snake(v, v->sn[i], cur, s1, len, ch);
        for (int t0 = 0; ok && t0 < len; t0 += 32768) {              /* the GEMM takes at most 32768 rows at a time */
            const int tn = len - t0 < 32768 ? len - t0 : 32768;
            ok = vkc_matmul(v->up[i].t, s1, (size_t)t0 * ch, v->tmp, (size_t)t0 * cout * K, tn);
        }
        VkcCodec ct = {2, len, K, 0, 0, 0, 0, 0, len * r * cout, v->up[i].off, r, cout};
        ok = ok && vkc_codec(v->tmp, cur, v->tab, &ct);
        len *= r; ch = cout;
        VkcEw add = {VKC_EW_ADD, len * ch, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        for (int u = 0; ok && u < 3; u++)
            ok = qcv_snake(v, v->a1[i][u], cur, s1, len, ch) && qcv_conv(v, &v->c1[i][u], &bk->r[u].c1, s1, s2, len) &&
                 qcv_snake(v, v->a2[i][u], s2, s1, len, ch) && qcv_conv(v, &v->c2[i][u], &bk->r[u].c2, s1, s2, len) &&
                 vkc_ew(cur, cur, s2, NULL, NULL, &add);
    }
    if (ok) {
        VkcCodec cl = {5, m->d_out.cin, m->d_out.K, m->d_out.dil, len, 0, 0, 0, len, v->d_out.off, v->d_out_w, 1};
        ok = qcv_snake(v, v->sn_out, cur, s1, len, ch) && vkc_codec(s1, v->out, v->tab, &cl);
    }
    ok = ok && vkc_submit(1);
    if (!ok) {
        fprintf(stderr, "[VK] qwen3tts: the vocoder's frame failed; the CPU runs it from here on\n");
        v->ok = 0;
        if (vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); qt_vk.ready = 0; qt_vk.chain = 0; }
        return NULL;
    }
    float *wav = (float *)malloc((size_t)len * sizeof(float));
    if (!wav) return NULL;
    memcpy(wav, vkc_ptr(v->out), (size_t)len * sizeof(float));
    *nsamples = len;
    return wav;
}
#endif

typedef struct { uint64_t s; } QtRng;
static double qt_rand(QtRng *r){                                   /* xorshift64*: uniform in [0, 1) */
    r->s ^= r->s >> 12; r->s ^= r->s << 25; r->s ^= r->s >> 27;
    return (double)((r->s * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}

/* the next id from logits [n]: greedy, or temperature then top-k then a draw */
static int qt_pick(const float *logits, int n, int greedy, float temp, int topk, QtRng *rng){
    int best = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    if (greedy) return best;
    const int k = topk > 0 && topk < n ? topk : n;
    int *idx = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int a = 0; a < k; a++) {                                  /* the k largest, by partial selection */
        int b = a;
        for (int j = a + 1; j < n; j++) if (logits[idx[j]] > logits[idx[b]]) b = j;
        const int t = idx[a]; idx[a] = idx[b]; idx[b] = t;
    }
    double mx = logits[idx[0]], sum = 0;
    double *p = (double *)malloc((size_t)k * sizeof(double));
    for (int a = 0; a < k; a++) { p[a] = exp(((double)logits[idx[a]] - mx) / (double)temp); sum += p[a]; }
    double r = qt_rand(rng) * sum;
    int pick = idx[k - 1];
    for (int a = 0; a < k; a++) { r -= p[a]; if (r <= 0) { pick = idx[a]; break; } }
    free(idx); free(p);
    return pick;
}

/* The decode steps are one row against every weight: memory-bound, with a barrier after each small matrix. A team of every logical CPU
 * spends its time at the barriers (32 threads: 346 ms a frame; 6: 115 ms on the same machine), so the frame loop runs on a smaller team:
 * --decode-threads N, or QT_DECODE_THREADS, or by default the physical cores capped at 6. The prefill and the codec keep the full team. */
static int qt_decode_threads(void){
#ifdef _OPENMP
    const char *e = getenv("QT_DECODE_THREADS");
    if (e && atoi(e) > 0) return atoi(e);
    int phys = coli_physical_cores();
    int n = phys > 0 ? phys : omp_get_max_threads();
    return n > 6 ? 6 : n;
#else
    return 1;
#endif
}

typedef struct { int greedy; float temperature; int top_k; float rep_penalty; int sub_greedy; float sub_temperature; int sub_top_k; int max_frames; uint64_t seed; } QtGen;

/* Teacher forcing (a test of the whole stack against reference codes that does not trip on ties): the given codes drive the generation, and
 * at each choice the engine says whether the given code is its own argmax, and if not by how much (in logits) the argmax beat it. */
typedef struct { const int *codes; int frames; int choices, flips; double worst_gap; int eos_ok; int *first_flip; } QtTeacher;

static double qt_argmax_gap(const float *logits, int n, int taught, int *best_out){
    int best = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    *best_out = best;
    return (double)logits[best] - (double)logits[taught];
}

static void qt_teach_note(QtTeacher *t, const float *logits, int n, int taught, int frame){
    int best = 0;
    const double gap = qt_argmax_gap(logits, n, taught, &best);
    t->choices++;
    if (best != taught) {
        t->flips++;
        if (gap > t->worst_gap) t->worst_gap = gap;
        if (t->first_flip && *t->first_flip < 0) *t->first_flip = frame;
    }
}

/* The frames of codes [frames][16] (malloc'd) for a prefill [L][D]. logits0, when not NULL, gets the first step's raw logits [vocab]. */
static int *qt_generate_on(const QtModel *m, const float *prefill, int L, const QtGen *g, int *frames, float *logits0, QtTeacher *teach, int allow_dev, int *dev_failed){
    const QtDec *T = &m->talker, *P = &m->pred;
    const int D = T->D, Dp = P->D, maxf = g->max_frames;
    const int cap = L + maxf + 1;
    QtCache *tc = qt_cache_new(T, cap), *pcache = qt_cache_new(P, QT_GROUPS + 1);
    float *tcs = (float *)malloc((size_t)cap * T->hd * sizeof(float)), *tsn = (float *)malloc((size_t)cap * T->hd * sizeof(float));
    float *pcs = (float *)malloc((size_t)(QT_GROUPS + 1) * P->hd * sizeof(float)), *psn = (float *)malloc((size_t)(QT_GROUPS + 1) * P->hd * sizeof(float));
    acd_rope_table(tcs, tsn, cap, T->hd, T->theta);
    acd_rope_table(pcs, psn, QT_GROUPS + 1, P->hd, P->theta);
    float *x = (float *)malloc((size_t)L * D * sizeof(float));
    memcpy(x, prefill, (size_t)L * D * sizeof(float));
    float *hn = (float *)malloc((size_t)D * sizeof(float)), *logits = (float *)malloc((size_t)m->vocab * sizeof(float)), *raw = (float *)malloc((size_t)m->vocab * sizeof(float));
    int dev = 0;
#ifdef COLI_VULKAN
    if (allow_dev && qt_vk.chain && qt_vk_setup(&qt_vk_talker, T, &m->codec_head, 1) && qt_vk_setup(&qt_vk_pred, P, m->pred_head, QT_GROUPS - 1))
        dev = qt_vk_reserve(&qt_vk_talker, (size_t)L, (size_t)cap, tcs, tsn) && qt_vk_reserve(&qt_vk_pred, 2, QT_GROUPS + 1, pcs, psn) &&
              qt_vk_run(&qt_vk_talker, x, L, 0, 0, hn, logits, m->vocab);
#endif
    if (!dev) {
        qt_dec_rows(T, tc, x, L, 0, tcs, tsn);
        acd_rmsnorm(hn, x + (size_t)(L - 1) * D, 1, D, T->norm, T->eps);
        qt_mv(logits, hn, &m->codec_head);
    }
#ifdef _OPENMP
    const int team_before = omp_get_max_threads();
    omp_set_num_threads(qt_decode_threads());
#endif
    int *codes = (int *)malloc((size_t)(maxf > 0 ? maxf : 1) * QT_GROUPS * sizeof(int));
    int *seen = (int *)calloc((size_t)m->vocab, sizeof(int));
    QtRng rng = {g->seed ? g->seed : 0x9E3779B97F4A7C15ULL};
    float *pad = qt_text_embed(m, &m->tts_pad, 1);
    float *prow = (float *)malloc((size_t)2 * D * sizeof(float)), *ph = (float *)malloc((size_t)Dp * sizeof(float)), *pl = (float *)malloc((size_t)m->cb_vocab * sizeof(float));
    float *prj = (float *)malloc((size_t)2 * Dp * sizeof(float));                  /* the predictor's input rows: prow, projected when its width is not the talker's */
    float *emb = (float *)malloc((size_t)D * sizeof(float)), *sum = (float *)malloc((size_t)D * sizeof(float)), *next = (float *)malloc((size_t)D * sizeof(float));
    int f = 0;
    for (; f < maxf; f++) {
        if (f == 0 && logits0) memcpy(logits0, logits, (size_t)m->vocab * sizeof(float));
        memcpy(raw, logits, (size_t)m->vocab * sizeof(float));
        for (int i = m->vocab - 1024; i < m->vocab; i++) if (i != m->codec_eos) raw[i] = -INFINITY;     /* suppress_tokens */
        if (f < 2) raw[m->codec_eos] = -INFINITY;                                                       /* min_new_tokens */
        if (g->rep_penalty != 1.0f)
            for (int i = 0; i < m->vocab; i++)
                if (seen[i] && raw[i] != -INFINITY) raw[i] = raw[i] < 0 ? raw[i] * g->rep_penalty : raw[i] / g->rep_penalty;
        int c0;
        if (teach) {
            if (f >= teach->frames) { int b = 0; qt_argmax_gap(raw, m->vocab, m->codec_eos, &b); teach->eos_ok = b == m->codec_eos; break; }
            c0 = teach->codes[(size_t)f * QT_GROUPS];
            qt_teach_note(teach, raw, m->vocab, c0, f);
        } else c0 = qt_pick(raw, m->vocab, g->greedy, g->temperature, g->top_k, &rng);
        if (c0 == m->codec_eos) break;
        seen[c0] = 1;
        int *code = codes + (size_t)f * QT_GROUPS;
        code[0] = c0;
        /* the code predictor: [the talker's hidden state, embedding of c0], then one code at a time */
        memcpy(prow, hn, (size_t)D * sizeof(float));
        qt_row(m->codec_emb, c0, D, prow + D);
        if (m->has_mtp) qt_lin_any(prj, prow, 2, &m->mtp); else memcpy(prj, prow, (size_t)2 * D * sizeof(float));
        pcache->len = 0;
#ifdef COLI_VULKAN
        if (dev && !qt_vk_run(&qt_vk_pred, prj, 2, 0, 0, NULL, pl, m->cb_vocab)) { *dev_failed = 1; break; }
#endif
        if (!dev) {
            qt_dec_rows(P, pcache, prj, 2, 0, pcs, psn);
            acd_rmsnorm(ph, prj + Dp, 1, Dp, P->norm, P->eps);
            qt_mv(pl, ph, &m->pred_head[0]);
        }
        if (teach) { code[1] = teach->codes[(size_t)f * QT_GROUPS + 1]; qt_teach_note(teach, pl, m->cb_vocab, code[1], f); }
        else code[1] = qt_pick(pl, m->cb_vocab, g->sub_greedy, g->sub_temperature, g->sub_top_k, &rng);
        for (int i = 1; i < QT_GROUPS - 1; i++) {
            qt_row(m->pred_emb[i - 1], code[i], D, emb);
            if (m->has_mtp) qt_lin_any(prj, emb, 1, &m->mtp); else memcpy(prj, emb, (size_t)D * sizeof(float));
#ifdef COLI_VULKAN
            if (dev && !qt_vk_run(&qt_vk_pred, prj, 1, i + 1, i, NULL, pl, m->cb_vocab)) { *dev_failed = 1; break; }
#endif
            if (!dev) {
                qt_dec_rows(P, pcache, prj, 1, i + 1, pcs, psn);
                acd_rmsnorm(ph, prj, 1, Dp, P->norm, P->eps);
                qt_mv(pl, ph, &m->pred_head[i]);
            }
            if (teach) { code[i + 1] = teach->codes[(size_t)f * QT_GROUPS + i + 1]; qt_teach_note(teach, pl, m->cb_vocab, code[i + 1], f); }
            else code[i + 1] = qt_pick(pl, m->cb_vocab, g->sub_greedy, g->sub_temperature, g->sub_top_k, &rng);
        }
        if (*dev_failed) break;
        /* the next talker input: the sum of the 16 embeddings + tts_pad (the text is all in the prefill) */
        qt_row(m->codec_emb, c0, D, sum);
        for (int gi = 1; gi < QT_GROUPS; gi++) {
            qt_row(m->pred_emb[gi - 1], code[gi], D, emb);
            for (int j = 0; j < D; j++) sum[j] += emb[j];
        }
        for (int j = 0; j < D; j++) next[j] = sum[j] + pad[j];
#ifdef COLI_VULKAN
        if (dev && !qt_vk_run(&qt_vk_talker, next, 1, L + f, 0, hn, logits, m->vocab)) { *dev_failed = 1; break; }
#endif
        if (!dev) qt_dec_rows(T, tc, next, 1, L + f, tcs, tsn);
        if (!dev) { acd_rmsnorm(hn, next, 1, D, T->norm, T->eps); qt_mv(logits, hn, &m->codec_head); }

    }
    *frames = f;
#ifdef _OPENMP
    omp_set_num_threads(team_before);
#endif
    qt_cache_free(T, tc); qt_cache_free(P, pcache);
    free(tcs); free(tsn); free(pcs); free(psn); free(x); free(hn); free(logits); free(raw); free(seen); free(pad); free(prow); free(prj); free(ph); free(pl); free(emb); free(sum); free(next);
    if (*dev_failed) { free(codes); return NULL; }
    return codes;
}

/* On the device when there is one; if it fails part way, the whole generation again on the CPU (its caches were never filled). */
static int *qt_generate(const QtModel *m, const float *prefill, int L, const QtGen *g, int *frames, float *logits0, QtTeacher *teach){
    int failed = 0;
    QtTeacher saved;
    if (teach) saved = *teach;
    int *codes = qt_generate_on(m, prefill, L, g, frames, logits0, teach, 1, &failed);
    if (!failed) return codes;
    if (teach) *teach = saved;
    failed = 0;
    return qt_generate_on(m, prefill, L, g, frames, logits0, teach, 0, &failed);
}

static void qt_compare(const char *label, const float *a, const float *b, size_t n){
    double mx = 0, sum = 0, ref = 0;
    for (size_t i = 0; i < n; i++) { const double d = fabs((double)a[i] - b[i]); if (d > mx) mx = d; sum += d; ref += fabs(b[i]); }
    printf("%-10s n=%zu  max abs diff %.3e  mean abs diff %.3e  (mean |ref| %.3e)\n", label, n, mx, sum / (double)(n ? n : 1), ref / (double)(n ? n : 1));
}

/* ------------------------------------------------------------------------------------------------ serving (T3) */

#define QT_MAX_LINE ((size_t)1 << 20)

static void qt_escape(const char *s){
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { putchar('\\'); putchar(c); }
        else if (c == '\n') fputs("\\n", stdout);
        else if (c == '\r') fputs("\\r", stdout);
        else if (c == '\t') fputs("\\t", stdout);
        else if (c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
}
static void qt_error(const char *id, const char *message){
    printf("ERROR {\"id\":\""); qt_escape(id ? id : "");
    printf("\",\"message\":\""); qt_escape(message); printf("\"}\n");
    fflush(stdout);
}
static char *qt_read_line(int *too_long){
    size_t cap = 1 << 12, n = 0;
    char *buf = (char *)malloc(cap);
    int c, any = 0;
    *too_long = 0;
    if (!buf) exit(1);
    while ((c = fgetc(stdin)) != EOF) {
        any = 1;
        if (c == '\n') break;
        if (*too_long) continue;
        if (n + 2 > cap) {
            if (cap >= QT_MAX_LINE) { *too_long = 1; continue; }
            cap *= 2; buf = (char *)realloc(buf, cap);
            if (!buf) exit(1);
        }
        buf[n++] = (char)c;
    }
    if (!any) { free(buf); return NULL; }
    if (n && buf[n - 1] == '\r') n--;
    buf[n] = 0;
    return buf;
}

typedef struct { int frames, samples, truncated; double gen_s, codec_s; } QtSpoken;

/* One request end to end: text, speaker and language to a WAV file at `path`. 0 on success, else err set. */
static int qt_speak(const QtModel *m, const char *text, const char *speaker, const char *instruct, const char *language, const QtGen *gen, const char *path, QtSpoken *res, char *err, size_t errn){
    int nids = 0, rows = 0;
    int *ids = qt_text_ids(m, text, &nids);
    float *prefill = qt_prefill(m, ids, nids, speaker, instruct, language, &rows, err, errn);
    free(ids);
    if (!prefill) return -1;
    double t0 = qt_now();
    int frames = 0;
    int *codes = qt_generate(m, prefill, rows, gen, &frames, NULL, NULL);
    free(prefill);
    res->gen_s = qt_now() - t0;
    res->frames = frames;
    res->truncated = frames >= gen->max_frames;
    if (frames <= 0) { free(codes); snprintf(err, errn, "the model produced no audio for this text"); return -1; }
    t0 = qt_now();
    int n = 0;
    float *wav = qt_codec_decode(&m->codec, codes, frames, &n);
    free(codes);
    res->codec_s = qt_now() - t0;
    res->samples = n;
    const int bad = !wav || qt_write_wav(path, wav, n, QT_SR);
    free(wav);
    if (bad) { snprintf(err, errn, "cannot write %s", path); return -1; }
    return 0;
}

static const char *qt_name_of(const char *dir){
    const char *s = strrchr(dir, '/'), *b = strrchr(dir, '\\');
    if (b && (!s || b > s)) s = b;
    return s ? s + 1 : dir;
}

static void qt_names(jval *obj){
    if (!obj || obj->t != J_OBJ) return;
    for (int i = 0; i < obj->len; i++) { if (i) putchar(','); putchar('"'); qt_escape(obj->keys[i]); putchar('"'); }
}

static const char *qt_device(void){
#ifdef COLI_VULKAN
    if (qt_vk.ready && qt_vk.chain) return "vulkan";
#endif
    return "cpu";
}

/* READY, then SPEAK lines until stdin closes (docs/qwen3-tts.md has the protocol). */
static int qt_serve(const QtModel *m, const char *name, const QtGen *base){
    jval *tc = json_get(m->cfg, "talker_config");
    jval *kind = json_get(m->cfg, "tts_model_type");
    printf("READY {\"model\":\""); qt_escape(name);
    printf("\",\"task\":\"speech\",\"kind\":\"%s\",\"instruct\":%s,\"sample_rate\":%d,\"layers\":%d,\"speakers\":[", kind && kind->t == J_STR ? kind->str : "custom_voice", m->instruct_ok ? "true" : "false", QT_SR, m->talker.nl);
    qt_names(json_get(tc, "spk_id"));
    printf("],\"languages\":[");
    qt_names(json_get(tc, "codec_language_id"));
    printf("],\"device\":\"%s\"}\n", qt_device());
    fflush(stdout);
    for (;;) {
        int too_long;
        char *line = qt_read_line(&too_long);
        if (!line) break;
        if (too_long) { qt_error("", "the request line is too long"); free(line); continue; }
        if (!*line) { free(line); continue; }
        if (strncmp(line, "SPEAK ", 6)) { qt_error("", "unknown message"); free(line); continue; }
        jval *j = json_parse_checked(line + 6);
        if (!j || j->t != J_OBJ) { qt_error("", "SPEAK needs a JSON object"); if (j) json_free(j); free(line); continue; }
        jval *jid = json_get(j, "id"), *jt = json_get(j, "text"), *js = json_get(j, "speaker"), *jl = json_get(j, "language"), *jo = json_get(j, "out");
        jval *jin = json_get(j, "instruct"), *jseed = json_get(j, "seed"), *jg = json_get(j, "greedy"), *jmf = json_get(j, "max_frames"), *jtemp = json_get(j, "temperature");
        const char *id = jid && jid->t == J_STR ? jid->str : "";
        char err[512] = "";
        if (!jt || jt->t != J_STR || !*jt->str) qt_error(id, "text must be a non-empty string");
        else if (!jo || jo->t != J_STR || !*jo->str) qt_error(id, "out must be the path of the WAV file to write");
        else {
            QtGen g = *base;
            if (jseed && jseed->t == J_NUM && jseed->num > 0) g.seed = (uint64_t)jseed->num;
            if (jg && jg->t == J_BOOL && jg->boolean) { g.greedy = 1; g.sub_greedy = 1; }
            if (jmf && jmf->t == J_NUM && jmf->num >= 1 && jmf->num < g.max_frames) g.max_frames = (int)jmf->num;
            if (jtemp && jtemp->t == J_NUM && jtemp->num > 0) { g.temperature = (float)jtemp->num; g.sub_temperature = (float)jtemp->num; }
            QtSpoken r;
            memset(&r, 0, sizeof r);
            const double t0 = qt_now();
            if (qt_speak(m, jt->str, js && js->t == J_STR ? js->str : "", jin && jin->t == J_STR ? jin->str : "", jl && jl->t == J_STR ? jl->str : "auto", &g, jo->str, &r, err, sizeof err)) qt_error(id, err);
            else {
                printf("AUDIO {\"id\":\""); qt_escape(id);
                printf("\",\"path\":\""); qt_escape(jo->str);
                printf("\",\"samples\":%d,\"sample_rate\":%d,\"frames\":%d,\"audio_seconds\":%.3f,\"seconds\":%.3f,\"truncated\":%s}\n",
                       r.samples, QT_SR, r.frames, (double)r.samples / QT_SR, qt_now() - t0, r.truncated ? "true" : "false");
                fflush(stdout);
                fprintf(stderr, "qwen3tts: %.2f s of audio: talker and predictor %.2f s (%.0f ms a frame), codec %.2f s\n", (double)r.samples / QT_SR, r.gen_s, 1000.0 * r.gen_s / (r.frames ? r.frames : 1), r.codec_s);
            }
        }
        json_free(j);
        free(line);
    }
    return 0;
}

static const char *QT_USAGE =
    "usage: qwen3tts --model DIR --serve | --text TEXT [--speaker NAME] [--instruct STYLE] [--language NAME] [--out FILE.wav] [--greedy] [--seed N] [--max-frames N] [--ref REFDIR]\n"
    "       qwen3tts --codec DIR --decode-codes FILE.i32 [--out FILE.wav] [--ref REFDIR]\n";

int main(int argc, char **argv){
    const char *model_dir = NULL, *codec_dir = NULL, *codes_path = NULL, *out = NULL, *ref = NULL, *text = NULL, *speaker = "Ryan", *instruct = NULL, *language = "Auto";
    int teacher_forcing = 0, serve = 0;
    QtGen gen = {0, 0.9f, 50, 1.05f, 0, 0.9f, 50, 4096, 0};
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--codec") && i + 1 < argc) codec_dir = argv[++i];
        else if (!strcmp(argv[i], "--decode-codes") && i + 1 < argc) codes_path = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) text = argv[++i];
        else if (!strcmp(argv[i], "--speaker") && i + 1 < argc) speaker = argv[++i];
        else if (!strcmp(argv[i], "--instruct") && i + 1 < argc) instruct = argv[++i];
        else if (!strcmp(argv[i], "--language") && i + 1 < argc) language = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--greedy")) { gen.greedy = 1; gen.sub_greedy = 1; }
        else if (!strcmp(argv[i], "--teacher")) teacher_forcing = 1;
        else if (!strcmp(argv[i], "--serve")) serve = 1;
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) gen.seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--max-frames") && i + 1 < argc) gen.max_frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--decode-threads") && i + 1 < argc) {
#ifdef _OPENMP
            static char env[32];
            snprintf(env, sizeof env, "QT_DECODE_THREADS=%d", atoi(argv[++i]));
            putenv(env);
#else
            ++i;
#endif
        }
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
#ifdef _OPENMP
            omp_set_num_threads(atoi(argv[++i]));
#else
            ++i;
#endif
        } else { fputs(QT_USAGE, stderr); return 2; }
    }
    char err[512] = "";
    char path[4300];
    if (codec_dir && codes_path) {                                   /* the codec alone: codes in, audio out */
        QtCodec *c = (QtCodec *)calloc(1, sizeof *c);
        double t0 = qt_now();
        if (qt_codec_load(c, codec_dir, err, sizeof err)) { fprintf(stderr, "qwen3tts: %s\n", err); return 1; }
#ifdef COLI_VULKAN
        qt_vk_open();
#endif
        fprintf(stderr, "qwen3tts: codec decoder loaded: %.2f GB, %.1f s\n", (double)c->bytes / 1e9, qt_now() - t0);
        size_t nvals = 0;
        int *codes = (int *)qt_load_f32(codes_path, &nvals);
        if (!codes || nvals % QT_GROUPS) { fprintf(stderr, "qwen3tts: cannot read codes from %s\n", codes_path); return 1; }
        for (size_t i = 0; i < nvals; i++) if (codes[i] < 0 || codes[i] >= c->cbsize) { fprintf(stderr, "qwen3tts: code %d out of range\n", codes[i]); return 1; }
        t0 = qt_now();
        int n = 0;
        float *wav = qt_codec_decode(c, codes, (int)(nvals / QT_GROUPS), &n);
        fprintf(stderr, "qwen3tts: %d frames -> %d samples (%.2f s of audio) in %.2f s\n", (int)(nvals / QT_GROUPS), n, (double)n / QT_SR, qt_now() - t0);
        if (out && qt_write_wav(out, wav, n, QT_SR)) { fprintf(stderr, "qwen3tts: cannot write %s\n", out); return 1; }
        if (ref) {
            snprintf(path, sizeof path, "%s/wav.f32", ref);
            size_t got = 0;
            float *r = qt_load_f32(path, &got);
            if (r) qt_compare("waveform", wav, r, got < (size_t)n ? got : (size_t)n);
        }
        return 0;
    }
    if (!model_dir || (!text && !serve)) { fputs(QT_USAGE, stderr); return 2; }
    QtModel *m = (QtModel *)calloc(1, sizeof *m);
    double t0 = qt_now();
    if (qt_model_load(m, model_dir, err, sizeof err)) { fprintf(stderr, "qwen3tts: %s\n", err); return 1; }
    fprintf(stderr, "qwen3tts: %d talker layers (d %d), %d predictor layers, %.2f GB resident, loaded in %.1f s\n", m->talker.nl, m->talker.D, m->pred.nl, (double)m->bytes / 1e9, qt_now() - t0);
#ifdef COLI_VULKAN
    qt_vk_open();                      /* once the weights are loaded: the device opens, and each matrix goes up on the first request */
#endif
    if (serve) return qt_serve(m, qt_name_of(model_dir), &gen);
    int nids = 0, rows = 0;
    int *ids = qt_text_ids(m, text, &nids);
    float *prefill = qt_prefill(m, ids, nids, speaker, instruct, language, &rows, err, sizeof err);
    if (!prefill) { fprintf(stderr, "qwen3tts: %s\n", err); return 1; }
    float *logits0 = (float *)malloc((size_t)m->vocab * sizeof(float));
    t0 = qt_now();
    int frames = 0;
    QtTeacher teach;
    int first_flip = -1;
    memset(&teach, 0, sizeof teach);
    teach.first_flip = &first_flip;
    if (teacher_forcing) {                                           /* the reference's codes drive the run (needs --ref) */
        size_t got = 0;
        char tp[4300];
        snprintf(tp, sizeof tp, "%s/codes.i32", ref ? ref : ".");
        teach.codes = (const int *)qt_load_f32(tp, &got);
        if (!ref || !teach.codes || got % QT_GROUPS) { fprintf(stderr, "qwen3tts: --teacher needs --ref with codes.i32\n"); return 1; }
        teach.frames = (int)(got / QT_GROUPS);
    }
    int *codes = qt_generate(m, prefill, rows, &gen, &frames, logits0, teacher_forcing ? &teach : NULL);
    const double t_gen = qt_now() - t0;
    t0 = qt_now();
    int n = 0;
    float *wav = frames > 0 ? qt_codec_decode(&m->codec, codes, frames, &n) : (float *)calloc(1, sizeof(float));
    fprintf(stderr, "qwen3tts: %d frames (%.2f s of audio): talker and predictor %.2f s (%.0f ms a frame), codec %.2f s\n", frames, (double)n / QT_SR, t_gen,
            frames ? 1000.0 * t_gen / frames : 0.0, qt_now() - t0);
    if (out && qt_write_wav(out, wav, n, QT_SR)) { fprintf(stderr, "qwen3tts: cannot write %s\n", out); return 1; }
    if (teacher_forcing)
        printf("teacher: %d choices over %d frames; %d of the reference's codes were not my argmax (largest margin %.2e logits, first at frame %d); ends with the end token: %s\n",
               teach.choices, teach.frames, teach.flips, teach.worst_gap, first_flip, teach.eos_ok ? "yes" : "NO");
    if (ref) {
        size_t got = 0;
        snprintf(path, sizeof path, "%s/input_ids.i32", ref);
        int *r = (int *)qt_load_f32(path, &got);
        if (r) { printf("input ids: %s (%d here, %zu in the reference)\n", got == (size_t)nids && !memcmp(r, ids, got * 4) ? "identical" : "DIFFERENT", nids, got); free(r); }
        snprintf(path, sizeof path, "%s/prefill.f32", ref);
        float *f = qt_load_f32(path, &got);
        if (f) { if (got == (size_t)rows * m->talker.D) qt_compare("prefill", prefill, f, got); else printf("prefill: reference has %zu values, here %d\n", got, rows * m->talker.D); free(f); }
        snprintf(path, sizeof path, "%s/logits0.f32", ref);
        f = qt_load_f32(path, &got);
        if (f) { if (got == (size_t)m->vocab) qt_compare("logits0", logits0, f, got); else printf("logits0: reference has %zu values, here %d\n", got, m->vocab); free(f); }
        snprintf(path, sizeof path, "%s/codes.i32", ref);
        r = (int *)qt_load_f32(path, &got);
        if (r) {
            const int same = got == (size_t)frames * QT_GROUPS && !memcmp(r, codes, got * 4);
            printf("codes: %s (%d frames here, %zu in the reference)\n", same ? "identical" : "DIFFERENT", frames, got / QT_GROUPS);
            if (!same) {
                size_t first = 0;
                const size_t cmp = got < (size_t)frames * QT_GROUPS ? got : (size_t)frames * QT_GROUPS;
                while (first < cmp && r[first] == codes[first]) first++;
                printf("  first difference at frame %zu, group %zu\n", first / QT_GROUPS, first % QT_GROUPS);
            }
            free(r);
        }
        snprintf(path, sizeof path, "%s/wav.f32", ref);
        f = qt_load_f32(path, &got);
        if (f) { if (got == (size_t)n) qt_compare("waveform", wav, f, got); else printf("waveform: reference has %zu samples, here %d\n", got, n); free(f); }
    }
    return 0;
}
