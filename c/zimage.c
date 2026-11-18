/* zimage.c -- Z-Image (Turbo) text-to-image in pure C.
 *
 * The checkpoint is a diffusers directory (Tongyi-MAI/Z-Image-Turbo) with three networks that are never
 * needed at the same time:
 *
 *   text_encoder  Qwen3-4B (36 layers, hidden 2560). Only a prefill, and only the first 35 layers: the
 *                 prompt's embedding is the hidden state BEFORE the last layer (diffusers'
 *                 hidden_states[-2]), taken at the prompt's real tokens. lm_head is not read.
 *   transformer   the S3-DiT: 30 single-stream blocks over [image tokens ; caption tokens], with two
 *                 refiner blocks on each side first (zimage_dit.h).
 *   vae           a 16-channel 2D autoencoder, decode only (zimage_vae.h).
 *
 * Weights are read straight from the checkpoint's safetensors and kept in int8 with one scale per row by
 * default (COLI_IMG_BITS=16 keeps bf16, 32 f32). Activations stay f32. Every matrix product runs through
 * qi_gemm.h, the same kernels as qwenimage.c, which this file was written beside.
 *
 * Usage:
 *   zimage --model DIR --ref REFDIR   compare every stage with a reference dump (tools/zimage_ref.py) */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "st.h"
#include "json.h"
#include "tok.h"
#include "qwen38_nfc.h"
#include "qi_gemm.h"
#include "serve_poll.h"
#ifdef COLI_VULKAN
/* VK=1 and COLI_VULKAN=1: the DiT's matrices run on the GPU, all the image tokens of a product in one call. One command
 * buffer: main thread only. */
#include "backend_vulkan.h"
static int g_vk_ready, g_vk_tried;
#endif

static double now_s(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void *xmalloc(size_t n){
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "[zimage] out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}
static float *fmalloc(size_t n){ return (float *)xmalloc(n * sizeof(float)); }

static char *read_file(const char *path){
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = xmalloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0; fclose(f);
    return b;
}
static jval *read_json(const char *dir, const char *file, char **arena){
    char path[2048]; snprintf(path, sizeof path, "%s/%s", dir, file);
    char *text = read_file(path);
    if (!text) { fprintf(stderr, "[zimage] cannot read %s\n", path); exit(1); }
    jval *v = json_parse(text, arena);
    free(text);
    if (!v) { fprintf(stderr, "[zimage] bad JSON in %s\n", path); exit(1); }
    return v;
}
static double jnum(jval *o, const char *k, double def){
    jval *v = o ? json_get(o, k) : NULL;
    return v && v->t == J_NUM ? v->num : def;
}
static int jbool(jval *o, const char *k, int def){
    jval *v = o ? json_get(o, k) : NULL;
    return v && v->t == J_BOOL ? v->boolean : def;
}

/* ---- weights ------------------------------------------------------------ */

static int g_bits = 8;   /* COLI_IMG_BITS: storage of the big matrices */
static int g_act8 = 1;   /* COLI_IMG_ACT8=0 turns off the int8 activations of the DiT blocks (VNNI) */

/* act8: this matrix may take int8 activations (COLI_IMG_ACT8). Only the DiT's
 * block matrices do. Measured on the real model: the text encoder's hidden
 * states carry a few huge channels, and one int8 scale per token wipes out the
 * rest (last layer 124% off, against 7% with int8 weights alone), while the
 * DiT blocks keep the picture (30.1 dB against 35.6 for int8 weights alone). */
typedef struct {
    QiMat m; void *own; float *own_sc; int act8;
    float *bias;    /* added to every output row, when the layer has one */
    int gpu;        /* may run on the GPU (COLI_VULKAN): the DiT's matrices */
    void *vk;       /* its device copy, made on first use */
    int vk_off;     /* the upload failed: this one stays on the CPU */
} Lin;

static st_tensor *need_tensor(shards *S, const char *name, int64_t n0, int64_t n1){
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "[zimage] missing tensor %s\n", name); exit(1); }
    int64_t numel = 1; for (int i = 0; i < t->rank; i++) numel *= t->shape[i];
    if (numel != n0 * n1) {
        fprintf(stderr, "[zimage] %s has %lld values, expected %lld x %lld\n",
                name, (long long)numel, (long long)n0, (long long)n1);
        exit(1);
    }
    return t;
}

/* How many layers load at once (COLI_IMG_LOAD_THREADS, default 4): each thread holds
 * one matrix's bf16 bytes while it quantizes them (up to ~140 MB for the text encoder's
 * MLP), so the count bounds the load's transient memory as well as its speed. */
static int qi_load_threads(void){
    static int n = 0;
    if (!n) {
        const char *e = getenv("COLI_IMG_LOAD_THREADS");
        n = e && atoi(e) > 0 ? atoi(e) : 4;
#ifdef _OPENMP
        if (n > omp_get_max_threads()) n = omp_get_max_threads();
#endif
        if (n < 1) n = 1;
    }
    return n;
}

/* bits: 8 = int8 rows, 16 = bf16, 32 = f32. Small matrices ask for 32. */
static void lin_load(shards *S, const char *name, int N, int K, int bits, Lin *out){
    st_tensor *t = need_tensor(S, name, N, K);
    if (bits == 8 && t->dtype == 0 && t->nbytes == (int64_t)N * K * 2) {
        /* bf16 rows straight to int8 rows: the same values as through an f32 copy */
        uint16_t *raw = xmalloc((size_t)N * K * 2);
        st_read_raw_cap(S, name, raw, (int64_t)N * K * 2, 1);
        memset(out, 0, sizeof *out);
        out->m.N = N; out->m.K = K;
        int8_t *q = xmalloc((size_t)N * K);
        float *sc = fmalloc(N);
        qi_quantize_i8_bf16(raw, N, K, q, sc);
        free(raw);
        out->m.fmt = QI_I8; out->m.w = out->own = q; out->m.sc = out->own_sc = sc;
        return;
    }
    float *tmp = fmalloc((size_t)N * K);
    st_read_f32_cap(S, name, tmp, (int64_t)N * K, 1);
    memset(out, 0, sizeof *out);
    out->m.N = N; out->m.K = K;
    if (bits == 32) {
        out->m.fmt = QI_F32; out->m.w = out->own = tmp;
    } else if (bits == 16) {
        uint16_t *b = xmalloc((size_t)N * K * 2);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)N * K; i++) {
            union { float f; uint32_t u; } v; v.f = tmp[i];
            b[i] = (uint16_t)((v.u + 0x7fff + ((v.u >> 16) & 1)) >> 16);
        }
        free(tmp);
        out->m.fmt = QI_BF16; out->m.w = out->own = b;
    } else {
        int8_t *q = xmalloc((size_t)N * K);
        float *sc = fmalloc(N);
        qi_quantize_i8(tmp, N, K, q, sc);
        free(tmp);
        out->m.fmt = QI_I8; out->m.w = out->own = q; out->m.sc = out->own_sc = sc;
    }
}
static void lin_free(Lin *l){
#ifdef COLI_VULKAN
    if (l->vk) coli_vk_tensor_free((ColiVkTensor *)l->vk);
#endif
    free(l->own); free(l->own_sc); free(l->bias); memset(l, 0, sizeof *l);
}
static size_t lin_bytes(const Lin *l){
    size_t e = l->m.fmt == QI_F32 ? 4 : l->m.fmt == QI_BF16 ? 2 : 1;
    return (size_t)l->m.N * l->m.K * e + (l->own_sc ? (size_t)l->m.N * 4 : 0);
}
static float *vec_load(shards *S, const char *name, int n){
    need_tensor(S, name, n, 1);
    float *v = fmalloc(n);
    st_read_f32_cap(S, name, v, n, 1);
    return v;
}
/* Y[M][N] = X . W^T on the GPU, for every row of X in one call. Each storage format is one of the shader's as it is
 * stored: QI_I8 is fmt 1 (int8 [N][K], one f32 scale per row), QI_BF16 fmt 11, QI_F32 fmt 10. The activations stay f32,
 * so the result is the CPU's f32-activation product up to summation order (COLI_IMG_ACT8 is a CPU kernel). The bias is
 * added here. 0 when the product stays on the CPU. */
static int qi_vk_linear(float *y, const float *x, int M, const Lin *l){
#ifdef COLI_VULKAN
    if (!g_vk_ready || !coli_vk_dense() || !l->gpu || l->vk_off || l->m.ld || M < 1) return 0;
#ifdef _OPENMP
    if (omp_in_parallel()) return 0;
#endif
    int fmt = l->m.fmt == QI_I8 ? 1 : l->m.fmt == QI_BF16 ? 11 : l->m.fmt == QI_F32 ? 10 : -1;
    if (fmt < 0) return 0;
    Lin *dev = (Lin *)l;    /* the device copy is a cache inside a read-only matrix */
    if (coli_vk_matmul((ColiVkTensor **)&dev->vk, y, x, l->m.w, fmt == 1 ? l->m.sc : NULL, fmt,
                       M, l->m.K, l->m.N, 0)) {
        if (l->bias) {
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < M; r++) for (int i = 0; i < l->m.N; i++) y[(int64_t)r * l->m.N + i] += l->bias[i];
        }
        return 1;
    }
    if (!dev->vk) dev->vk_off = 1;
#endif
    (void)y; (void)x; (void)M; (void)l;
    return 0;
}

/* ZIMAGE_PROF=1: where a step's time goes (linears, attention, the rest) */
static int g_prof; static double g_t_lin, g_t_att;
static inline void linear(float *y, const float *x, int M, const Lin *l){
    double t0 = g_prof ? now_s() : 0;
    if (!qi_vk_linear(y, x, M, l)) {
        if (g_act8 && l->act8 && l->m.fmt == QI_I8) qi_gemm_act8(y, l->m.N, x, l->m.K, M, &l->m, l->bias);
        else qi_gemm(y, x, M, &l->m, l->bias);
    }
    if (g_prof) g_t_lin += now_s() - t0;
}

/* ---- small kernels -------------------------------------------------------- */

static void rmsnorm_rows(float *y, const float *x, const float *w, int T, int D, float eps, int plus_one){
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float *r = x + (int64_t)t * D; float *o = y + (int64_t)t * D;
        double ss = 0; for (int i = 0; i < D; i++) ss += (double)r[i] * r[i];
        float s = 1.f / sqrtf((float)(ss / D) + eps);
        for (int i = 0; i < D; i++) o[i] = r[i] * s * (plus_one ? w[i] + 1.f : w[i]);
    }
}
/* LayerNorm without affine, then * (1 + scale). */
static void layernorm_mod(float *y, const float *x, const float *scale, int T, int D, float eps){
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float *r = x + (int64_t)t * D; float *o = y + (int64_t)t * D;
        double m = 0; for (int i = 0; i < D; i++) m += r[i];
        m /= D;
        double v = 0; for (int i = 0; i < D; i++) { double d = r[i] - m; v += d * d; }
        v /= D;
        float inv = 1.f / sqrtf((float)v + eps), mf = (float)m;
        for (int i = 0; i < D; i++) o[i] = (r[i] - mf) * inv * (1.f + scale[i]);
    }
}
static inline float silu(float x){ return x / (1.f + expf(-x)); }
static inline float gelu_tanh(float x){
    return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

/* Attention for every head: out[Tq][heads*hd] = softmax(Q K^T / sqrt(hd)) V.
 * Q rows have stride ldq, K/V rows stride ldk; query head h reads kv head
 * h / (heads/kvheads). causal >= 0: query i sees keys j <= i + causal.
 * qlim, when not NULL, gives each query its own count of visible keys (the first
 * qlim[i]) and replaces causal: how a bidirectional image block sits in a causal sequence. */
static void attention(float *out, int ldo, const float *Q, int ldq, int Tq,
                      const float *K, const float *V, int ldk, int Tk,
                      int heads, int kvheads, int hd, int causal, const int *qlim){
    double tprof = g_prof ? now_s() : 0;
    float *S = fmalloc((size_t)Tq * Tk);
    float *vt = fmalloc((size_t)hd * Tk);
    const float scale = 1.f / sqrtf((float)hd);
    int group = heads / kvheads, vt_head = -1;
    for (int h = 0; h < heads; h++) {
        int kh = h / group;
        QiMat Km = { QI_F32, Tk, hd, K + (int64_t)kh * hd, NULL, ldk };
        qi_gemm_ld(S, Tk, Q + (int64_t)h * hd, ldq, Tq, &Km, NULL);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Tq; i++) {
            float *r = S + (int64_t)i * Tk;
            int lim = qlim ? qlim[i] : causal >= 0 ? i + causal + 1 : Tk;
            if (lim > Tk) lim = Tk;
            float mx = -INFINITY;
            for (int j = 0; j < lim; j++) { r[j] *= scale; if (r[j] > mx) mx = r[j]; }
            double sum = 0;
            for (int j = 0; j < lim; j++) { r[j] = expf(r[j] - mx); sum += r[j]; }
            float inv = (float)(1.0 / sum);
            for (int j = 0; j < lim; j++) r[j] *= inv;
            for (int j = lim; j < Tk; j++) r[j] = 0.f;
        }
        if (kh != vt_head) {
            #pragma omp parallel for schedule(static)
            for (int d = 0; d < hd; d++)
                for (int j = 0; j < Tk; j++) vt[(int64_t)d * Tk + j] = V[(int64_t)j * ldk + (int64_t)kh * hd + d];
            vt_head = kh;
        }
        QiMat Vm = { QI_F32, hd, Tk, vt, NULL, 0 };
        qi_gemm_ld(out + (int64_t)h * hd, ldo, S, Tk, Tq, &Vm, NULL);
    }
    free(S); free(vt);
    if (g_prof) g_t_att += now_s() - tprof;
}

/* ---- text encoder (Qwen3-4B, prefill only) --------------------------------- */

typedef struct { Lin q, k, v, o, gate, up, down; float *ln1, *ln2, *qn, *kn; } TeLayer;
typedef struct {
    int hidden, layers, heads, kv, hd, inter, vocab;
    float eps, theta;
    shards S; int open;
    st_tensor *embed;
    TeLayer *L; int loaded;
    Tok tok; int tok_loaded;
    int im_start, im_end;
} Te;

static void te_config(Te *te, const char *model){
    char dir[2048]; snprintf(dir, sizeof dir, "%s/text_encoder", model);
    char *arena = NULL;
    jval *c = read_json(dir, "config.json", &arena);
    te->hidden = (int)jnum(c, "hidden_size", 0);
    te->layers = (int)jnum(c, "num_hidden_layers", 0);
    te->heads = (int)jnum(c, "num_attention_heads", 0);
    te->kv = (int)jnum(c, "num_key_value_heads", te->heads);
    te->hd = (int)jnum(c, "head_dim", te->heads ? te->hidden / te->heads : 0);
    te->inter = (int)jnum(c, "intermediate_size", 0);
    te->vocab = (int)jnum(c, "vocab_size", 0);
    te->eps = (float)jnum(c, "rms_norm_eps", 1e-6);
    te->theta = (float)jnum(c, "rope_theta", 1000000.0);
    json_free(c); free(arena);
    /* Model files come from mirrors nobody vouches for: every dimension that sizes a buffer is bounded,
     * and products are formed in 64 bits, so a config cannot wrap an int into a small allocation. */
    if (te->hidden <= 0 || te->hidden > 65536 || te->layers <= 1 || te->layers > 1024 ||
        te->heads <= 0 || te->heads > 1024 || te->kv <= 0 || te->hd <= 0 || te->hd > 1024 ||
        te->heads % te->kv || te->inter <= 0 || te->inter > (1 << 18) || te->vocab <= 0 ||
        te->vocab > (1 << 22) || te->hd % 2 || (int64_t)te->heads * te->hd > 65536) {
        fprintf(stderr, "[zimage] text_encoder/config.json: unsupported shape\n"); exit(1);
    }
}

static void te_open(Te *te, const char *model){
    if (te->open) return;
    char dir[2048]; snprintf(dir, sizeof dir, "%s/text_encoder", model);
    memset(&te->S, 0, sizeof te->S);
    st_init(&te->S, dir);
    te->open = 1;
    te->embed = need_tensor(&te->S, "model.embed_tokens.weight", te->vocab, te->hidden);
    if (te->embed->dtype != 0 && te->embed->dtype != 2) {
        fprintf(stderr, "[zimage] embed_tokens must be bf16 or f32\n"); exit(1);
    }
}

/* The last layer's output is never read (the embedding is the state before it), so it is not loaded. */
static void te_load(Te *te, const char *model){
    if (te->loaded) return;
    te_open(te, model);
    double t0 = now_s();
    int nl = te->layers - 1;
    te->L = calloc(nl, sizeof(TeLayer));
    int H = te->hidden, qd = te->heads * te->hd, kd = te->kv * te->hd;
    size_t bytes = 0;
    /* a few layers at once: each reads and quantizes its own matrices (COLI_IMG_LOAD_THREADS) */
    #pragma omp parallel for schedule(dynamic, 1) num_threads(qi_load_threads()) reduction(+:bytes)
    for (int l = 0; l < nl; l++) {
        TeLayer *L = &te->L[l];
        char n[256];
#define TN(s) (snprintf(n, sizeof n, "model.layers.%d.%s", l, s), n)
        lin_load(&te->S, TN("self_attn.q_proj.weight"), qd, H, g_bits, &L->q);
        lin_load(&te->S, TN("self_attn.k_proj.weight"), kd, H, g_bits, &L->k);
        lin_load(&te->S, TN("self_attn.v_proj.weight"), kd, H, g_bits, &L->v);
        lin_load(&te->S, TN("self_attn.o_proj.weight"), H, qd, g_bits, &L->o);
        lin_load(&te->S, TN("mlp.gate_proj.weight"), te->inter, H, g_bits, &L->gate);
        lin_load(&te->S, TN("mlp.up_proj.weight"), te->inter, H, g_bits, &L->up);
        lin_load(&te->S, TN("mlp.down_proj.weight"), H, te->inter, g_bits, &L->down);
        L->ln1 = vec_load(&te->S, TN("input_layernorm.weight"), H);
        L->ln2 = vec_load(&te->S, TN("post_attention_layernorm.weight"), H);
        L->qn = vec_load(&te->S, TN("self_attn.q_norm.weight"), te->hd);
        L->kn = vec_load(&te->S, TN("self_attn.k_norm.weight"), te->hd);
#undef TN
        bytes += lin_bytes(&L->q) + lin_bytes(&L->k) + lin_bytes(&L->v) + lin_bytes(&L->o) +
                 lin_bytes(&L->gate) + lin_bytes(&L->up) + lin_bytes(&L->down);
    }
    te->loaded = 1;
    fprintf(stderr, "[zimage] text encoder: %d of %d layers, %.2f GB resident (%s), %.1f s\n",
            nl, te->layers, bytes / 1e9, g_bits == 8 ? "int8" : g_bits == 16 ? "bf16" : "f32", now_s() - t0);
}

static void te_unload(Te *te){
    if (!te->loaded) return;
    for (int l = 0; l < te->layers - 1; l++) {
        TeLayer *L = &te->L[l];
        lin_free(&L->q); lin_free(&L->k); lin_free(&L->v); lin_free(&L->o);
        lin_free(&L->gate); lin_free(&L->up); lin_free(&L->down);
        free(L->ln1); free(L->ln2); free(L->qn); free(L->kn);
    }
    free(te->L); te->L = NULL; te->loaded = 0;
}

static void te_tokenizer(Te *te, const char *model){
    if (te->tok_loaded) return;
    char path[2048]; snprintf(path, sizeof path, "%s/tokenizer/tokenizer.json", model);
    memset(&te->tok, 0, sizeof te->tok);
    tok_load(&te->tok, path);
    te->tok_loaded = 1;
    te->im_start = tok_id_of(&te->tok, "<|im_start|>");
    te->im_end = tok_id_of(&te->tok, "<|im_end|>");
    if (te->im_start < 0 || te->im_end < 0) {
        fprintf(stderr, "[zimage] tokenizer has no <|im_start|>/<|im_end|>\n"); exit(1);
    }
}

/* The pipeline's template: the prompt as a user turn of Qwen3's chat template, with the assistant's turn
 * opened and thinking left on (enable_thinking=True adds nothing after it). No system turn, nothing dropped. */
static int *te_encode_prompt(Te *te, const char *prompt, int *n_out){
    char *norm = NULL; size_t norm_len = 0;
    const char *p = prompt ? prompt : "";
    if (q38_nfc_normalize(p, strlen(p), &norm, &norm_len) != 0) { norm = strdup(p); norm_len = strlen(p); }
    size_t cap = norm_len + 128;
    char *text = xmalloc(cap);
    snprintf(text, cap, "<|im_start|>user\n%.*s<|im_end|>\n<|im_start|>assistant\n", (int)norm_len, norm);
    int max = (int)strlen(text) + 16;
    int *ids = xmalloc(sizeof(int) * max);
    *n_out = tok_encode(&te->tok, text, (int)strlen(text), ids, max);
    free(text); free(norm);
    return ids;
}

/* rotate-half RoPE over the full head (the plain 1-D rotary: position = row index) */
static void te_rope(float *x, int T, int nh, int hd, int ld, float theta){
    int half = hd / 2;
    float *inv = fmalloc(half);
    for (int i = 0; i < half; i++) inv[i] = 1.f / powf(theta, (float)(2 * i) / (float)hd);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int h = 0; h < nh; h++) {
            float *r = x + (int64_t)t * ld + (int64_t)h * hd;
            for (int i = 0; i < half; i++) {
                float a = (float)t * inv[i], c = cosf(a), s = sinf(a);
                float x0 = r[i], x1 = r[i + half];
                r[i] = x0 * c - x1 * s;
                r[i + half] = x1 * c + x0 * s;
            }
        }
    free(inv);
}

/* ids[T] -> hidden[T][hidden]: the output of the layer before the last, with no final norm. */
static float *te_forward(Te *te, const int *ids, int T){
    int H = te->hidden, qd = te->heads * te->hd, kd = te->kv * te->hd, I = te->inter;
    float *x = fmalloc((size_t)T * H), *h = fmalloc((size_t)T * H);
    float *q = fmalloc((size_t)T * qd), *k = fmalloc((size_t)T * kd), *v = fmalloc((size_t)T * kd);
    float *a = fmalloc((size_t)T * qd), *g = fmalloc((size_t)T * I), *u = fmalloc((size_t)T * I);
    int esz = te->embed->dtype == 0 ? 2 : 4;
    void *row = xmalloc((size_t)H * esz);
    for (int t = 0; t < T; t++) {
        float *dst = x + (int64_t)t * H;
        if (ids[t] < 0 || ids[t] >= te->vocab) { fprintf(stderr, "[zimage] token id %d out of range\n", ids[t]); exit(1); }
        st_pread_full(te->embed->fd, row, (int64_t)H * esz, te->embed->off + (int64_t)ids[t] * H * esz, "embed row");
        if (esz == 2) for (int i = 0; i < H; i++) dst[i] = qi_bf16(((uint16_t *)row)[i]);
        else memcpy(dst, row, (size_t)H * 4);
    }
    free(row);
    for (int l = 0; l < te->layers - 1; l++) {
        TeLayer *L = &te->L[l];
        rmsnorm_rows(h, x, L->ln1, T, H, te->eps, 0);
        linear(q, h, T, &L->q); linear(k, h, T, &L->k); linear(v, h, T, &L->v);
        rmsnorm_rows(q, q, L->qn, T * te->heads, te->hd, te->eps, 0);
        rmsnorm_rows(k, k, L->kn, T * te->kv, te->hd, te->eps, 0);
        te_rope(q, T, te->heads, te->hd, qd, te->theta);
        te_rope(k, T, te->kv, te->hd, kd, te->theta);
        attention(a, qd, q, qd, T, k, v, kd, T, te->heads, te->kv, te->hd, 0, NULL);
        linear(h, a, T, &L->o);
        for (int64_t i = 0; i < (int64_t)T * H; i++) x[i] += h[i];
        rmsnorm_rows(h, x, L->ln2, T, H, te->eps, 0);
        linear(g, h, T, &L->gate); linear(u, h, T, &L->up);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)T * I; i++) g[i] = silu(g[i]) * u[i];
        linear(h, g, T, &L->down);
        for (int64_t i = 0; i < (int64_t)T * H; i++) x[i] += h[i];
    }
    free(h); free(q); free(k); free(v); free(a); free(g); free(u);
    return x;
}

/* ---- DiT (S3-DiT) ------------------------------------------------------------- */

typedef struct { Lin q, k, v, o, w1, w2, w3, mod; float *nq, *nk, *an1, *an2, *fn1, *fn2; int has_mod; } ZBlock;
typedef struct {
    int layers, nref, heads, hd, dim, mlp, in_ch, cap_dim, patch, axes[3];
    int adaln;                  /* min(dim, 256): the timestep embedding's width, which the modulations read */
    float eps, theta, t_scale;
    Lin x_emb, cap_emb, t1, t2, fin_mod, fin_out;
    float *cap_norm, *x_pad, *cap_pad;
    ZBlock *B, *NR, *CR;        /* main layers, noise refiner, context refiner */
    int loaded;
} Dit;

static void dit_config(Dit *d, const char *model){
    char dir[2048]; snprintf(dir, sizeof dir, "%s/transformer", model);
    char *arena = NULL;
    jval *c = read_json(dir, "config.json", &arena);
    d->layers = (int)jnum(c, "n_layers", 0);
    d->nref = (int)jnum(c, "n_refiner_layers", 0);
    d->heads = (int)jnum(c, "n_heads", 0);
    int kvh = (int)jnum(c, "n_kv_heads", d->heads);
    d->dim = (int)jnum(c, "dim", 0);
    d->in_ch = (int)jnum(c, "in_channels", 16);
    d->cap_dim = (int)jnum(c, "cap_feat_dim", 0);
    d->eps = (float)jnum(c, "norm_eps", 1e-5);
    d->theta = (float)jnum(c, "rope_theta", 256.0);
    d->t_scale = (float)jnum(c, "t_scale", 1000.0);
    int qk_norm = jbool(c, "qk_norm", 1);
    int psize = 0, fsize = 0;
    jval *ap = json_get(c, "all_patch_size"), *af = json_get(c, "all_f_patch_size");
    if (ap && ap->t == J_ARR && ap->len == 1 && ap->kids[0]->t == J_NUM) psize = (int)ap->kids[0]->num;
    if (af && af->t == J_ARR && af->len == 1 && af->kids[0]->t == J_NUM) fsize = (int)af->kids[0]->num;
    d->patch = psize;
    jval *ax = json_get(c, "axes_dims");
    int ok = ax && ax->t == J_ARR && ax->len == 3;
    for (int i = 0; ok && i < 3; i++) {
        d->axes[i] = ax->kids[i]->t == J_NUM ? (int)ax->kids[i]->num : -1;
        ok = d->axes[i] > 0 && d->axes[i] % 2 == 0;       /* each axis fills whole complex pairs */
    }
    json_free(c); free(arena);
    d->hd = d->heads > 0 ? d->dim / d->heads : 0;
    d->mlp = (int)((int64_t)d->dim * 8 / 3);                /* int(dim / 3 * 8) */
    d->adaln = d->dim < 256 ? d->dim : 256;
    if (!ok || d->layers <= 0 || d->layers > 1024 || d->nref < 0 || d->nref > 64 || d->heads <= 0 || d->heads > 1024 ||
        kvh != d->heads || d->dim <= 0 || d->dim > 65536 || d->dim % d->heads || d->hd % 2 ||
        d->axes[0] + d->axes[1] + d->axes[2] != d->hd || d->in_ch <= 0 || d->in_ch > 1024 ||
        d->cap_dim <= 0 || d->cap_dim > 65536 || psize != 2 || fsize != 1 || !qk_norm) {
        fprintf(stderr, "[zimage] transformer/config.json: unsupported shape "
                        "(needs patch 2, frame patch 1, qk_norm, as many kv heads as heads, axes summing to the head dim)\n");
        exit(1);
    }
}

static float *vec_load_n(shards *S, const char *name, int n){ return vec_load(S, name, n); }

static void zblock_load(shards *S, const char *pre, int D, int mlp, int hd, int adaln, int mod, ZBlock *B){
    char n[256];
#define BN(s) (snprintf(n, sizeof n, "%s.%s", pre, s), n)
    lin_load(S, BN("attention.to_q.weight"), D, D, g_bits, &B->q);
    lin_load(S, BN("attention.to_k.weight"), D, D, g_bits, &B->k);
    lin_load(S, BN("attention.to_v.weight"), D, D, g_bits, &B->v);
    lin_load(S, BN("attention.to_out.0.weight"), D, D, g_bits, &B->o);
    lin_load(S, BN("feed_forward.w1.weight"), mlp, D, g_bits, &B->w1);
    lin_load(S, BN("feed_forward.w2.weight"), D, mlp, g_bits, &B->w2);
    lin_load(S, BN("feed_forward.w3.weight"), mlp, D, g_bits, &B->w3);
    B->q.act8 = B->k.act8 = B->v.act8 = B->o.act8 = B->w1.act8 = B->w2.act8 = B->w3.act8 = 1;
    B->q.gpu = B->k.gpu = B->v.gpu = B->o.gpu = B->w1.gpu = B->w2.gpu = B->w3.gpu = 1;
    B->nq = vec_load_n(S, BN("attention.norm_q.weight"), hd);
    B->nk = vec_load_n(S, BN("attention.norm_k.weight"), hd);
    B->an1 = vec_load_n(S, BN("attention_norm1.weight"), D);
    B->an2 = vec_load_n(S, BN("attention_norm2.weight"), D);
    B->fn1 = vec_load_n(S, BN("ffn_norm1.weight"), D);
    B->fn2 = vec_load_n(S, BN("ffn_norm2.weight"), D);
    B->has_mod = mod;
    if (mod) {
        lin_load(S, BN("adaLN_modulation.0.weight"), 4 * D, adaln, 32, &B->mod);
        B->mod.bias = vec_load_n(S, BN("adaLN_modulation.0.bias"), 4 * D);
    }
#undef BN
}
static size_t zblock_bytes(const ZBlock *B){
    return lin_bytes(&B->q) + lin_bytes(&B->k) + lin_bytes(&B->v) + lin_bytes(&B->o) +
           lin_bytes(&B->w1) + lin_bytes(&B->w2) + lin_bytes(&B->w3);
}

static void dit_load(Dit *d, const char *model){
    if (d->loaded) return;
    char dir[2048]; snprintf(dir, sizeof dir, "%s/transformer", model);
    shards S; memset(&S, 0, sizeof S);
    st_init(&S, dir);
    double t0 = now_s();
    int D = d->dim, pd = d->patch * d->patch * d->in_ch;
    lin_load(&S, "all_x_embedder.2-1.weight", D, pd, 32, &d->x_emb);
    d->x_emb.bias = vec_load_n(&S, "all_x_embedder.2-1.bias", D);
    lin_load(&S, "cap_embedder.1.weight", D, d->cap_dim, g_bits, &d->cap_emb);
    d->cap_emb.bias = vec_load_n(&S, "cap_embedder.1.bias", D);
    d->cap_norm = vec_load_n(&S, "cap_embedder.0.weight", d->cap_dim);
    lin_load(&S, "t_embedder.mlp.0.weight", 1024, 256, 32, &d->t1);
    d->t1.bias = vec_load_n(&S, "t_embedder.mlp.0.bias", 1024);
    lin_load(&S, "t_embedder.mlp.2.weight", d->adaln, 1024, 32, &d->t2);
    d->t2.bias = vec_load_n(&S, "t_embedder.mlp.2.bias", d->adaln);
    lin_load(&S, "all_final_layer.2-1.adaLN_modulation.1.weight", D, d->adaln, 32, &d->fin_mod);
    d->fin_mod.bias = vec_load_n(&S, "all_final_layer.2-1.adaLN_modulation.1.bias", D);
    lin_load(&S, "all_final_layer.2-1.linear.weight", pd, D, 32, &d->fin_out);
    d->fin_out.bias = vec_load_n(&S, "all_final_layer.2-1.linear.bias", pd);
    d->x_pad = vec_load_n(&S, "x_pad_token", D);
    d->cap_pad = vec_load_n(&S, "cap_pad_token", D);
    d->B = calloc(d->layers, sizeof(ZBlock));
    d->NR = calloc(d->nref ? d->nref : 1, sizeof(ZBlock));
    d->CR = calloc(d->nref ? d->nref : 1, sizeof(ZBlock));
    size_t bytes = 0;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(qi_load_threads()) reduction(+:bytes)
    for (int l = 0; l < d->layers; l++) {
        char pre[64]; snprintf(pre, sizeof pre, "layers.%d", l);
        zblock_load(&S, pre, D, d->mlp, d->hd, d->adaln, 1, &d->B[l]);
        bytes += zblock_bytes(&d->B[l]);
    }
    for (int l = 0; l < d->nref; l++) {
        char pre[64];
        snprintf(pre, sizeof pre, "noise_refiner.%d", l); zblock_load(&S, pre, D, d->mlp, d->hd, d->adaln, 1, &d->NR[l]);
        snprintf(pre, sizeof pre, "context_refiner.%d", l); zblock_load(&S, pre, D, d->mlp, d->hd, d->adaln, 0, &d->CR[l]);
        bytes += zblock_bytes(&d->NR[l]) + zblock_bytes(&d->CR[l]);
    }
    st_destroy(&S);
    d->x_emb.gpu = d->cap_emb.gpu = d->t1.gpu = d->t2.gpu = d->fin_mod.gpu = d->fin_out.gpu = 1;
    for (int l = 0; l < d->layers; l++) d->B[l].mod.gpu = 1;
    for (int l = 0; l < d->nref; l++) d->NR[l].mod.gpu = 1;
    d->loaded = 1;
    fprintf(stderr, "[zimage] transformer: %d blocks + 2x%d refiner blocks, %.2f GB resident, %.1f s\n",
            d->layers, d->nref, bytes / 1e9, now_s() - t0);
#ifdef COLI_VULKAN
    /* after the DiT's weights, once: a missing device costs one line */
    if (!g_vk_tried) { g_vk_tried = 1; g_vk_ready = coli_vk_init_env("zimage"); }
#endif
}

/* The timestep embedding. `t` is the model's input, (1000 - scheduler timestep) as the pipeline passes it
 * before the transformer's own t_scale (the pipeline hands (1000 - t) / 1000 and the transformer multiplies by
 * t_scale = 1000 again): sinusoids [cos | sin] of 256, then Linear, SiLU, Linear -> 256 values. */
static void dit_temb(Dit *d, float t, float *adaln){
    float e[256];
    const int half = 128;
    float tt = (t / 1000.f) * d->t_scale;
    for (int i = 0; i < half; i++) {
        float f = expf((float)(-9.210340371976184) * (float)i / (float)half);  /* -ln(10000) */
        float a = tt * f;
        e[i] = cosf(a); e[half + i] = sinf(a);
    }
    float *h = fmalloc(1024);
    linear(h, e, 1, &d->t1);
    for (int i = 0; i < 1024; i++) h[i] = silu(h[i]);
    linear(adaln, h, 1, &d->t2);
    free(h);
}

/* cos/sin for the three-axis RoPE: pos3[T][3] -> tokens x hd/2 complex frequencies. */
static void dit_rope_table(const Dit *d, const int *pos3, int T, float *cs, float *sn){
    int half = d->hd / 2;
    float *freq = fmalloc(half); int *axis = xmalloc(sizeof(int) * half);
    int j = 0;
    for (int a = 0; a < 3; a++)
        for (int i = 0; i < d->axes[a] / 2; i++, j++) {
            freq[j] = (float)(1.0 / pow((double)d->theta, (double)(2 * i) / (double)d->axes[a]));
            axis[j] = a;
        }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int i = 0; i < half; i++) {
            float ang = (float)((double)pos3[t * 3 + axis[i]] * (double)freq[i]);
            cs[(int64_t)t * half + i] = cosf(ang);
            sn[(int64_t)t * half + i] = sinf(ang);
        }
    free(freq); free(axis);
}
/* interleaved pairs (2i, 2i+1) rotated as complex numbers */
static void dit_rope_apply(float *x, int T, int heads, int hd, const float *cs, const float *sn){
    int half = hd / 2, D = heads * hd;
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++)
        for (int h = 0; h < heads; h++) {
            float *r = x + (int64_t)t * D + (int64_t)h * hd;
            const float *c = cs + (int64_t)t * half, *s = sn + (int64_t)t * half;
            for (int i = 0; i < half; i++) {
                float a = r[2 * i], b = r[2 * i + 1];
                r[2 * i] = a * c[i] - b * s[i];
                r[2 * i + 1] = a * s[i] + b * c[i];
            }
        }
}

typedef struct { float *h, *q, *k, *v, *a, *g, *u; } DitScratch;
static void scratch_alloc(DitScratch *w, const Dit *d, int T){
    size_t D = d->dim;
    w->h = fmalloc(T * D); w->q = fmalloc(T * D); w->k = fmalloc(T * D); w->v = fmalloc(T * D); w->a = fmalloc(T * D);
    w->g = fmalloc((size_t)T * d->mlp); w->u = fmalloc((size_t)T * d->mlp);
}
static void scratch_free(DitScratch *w){ free(w->h); free(w->q); free(w->k); free(w->v); free(w->a); free(w->g); free(w->u); }

/* the modulation of a block from the timestep embedding: scale_msa, gate_msa, scale_mlp, gate_mlp (4 * dim) */
static void zblock_mod(const ZBlock *B, const float *adaln, float *mod){ linear(mod, adaln, 1, &B->mod); }

/* x[T][dim] <- x + gate_msa * norm2(attn(norm1(x) * (1 + scale_msa))); then the same for the MLP. mod NULL: no
 * modulation (the context refiner), the norms alone. Attention is full: nothing is masked. */
static void zblock(Dit *d, const ZBlock *B, float *x, int T, const float *mod, const float *cs, const float *sn, DitScratch *w){
    int D = d->dim;
    rmsnorm_rows(w->h, x, B->an1, T, D, d->eps, 0);
    if (mod) {
        const float *scale = mod;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) for (int i = 0; i < D; i++) w->h[(int64_t)t * D + i] *= 1.f + scale[i];
    }
    linear(w->q, w->h, T, &B->q); linear(w->k, w->h, T, &B->k); linear(w->v, w->h, T, &B->v);
    rmsnorm_rows(w->q, w->q, B->nq, T * d->heads, d->hd, d->eps, 0);
    rmsnorm_rows(w->k, w->k, B->nk, T * d->heads, d->hd, d->eps, 0);
    dit_rope_apply(w->q, T, d->heads, d->hd, cs, sn);
    dit_rope_apply(w->k, T, d->heads, d->hd, cs, sn);
    attention(w->a, D, w->q, D, T, w->k, w->v, D, T, d->heads, d->heads, d->hd, -1, NULL);
    linear(w->h, w->a, T, &B->o);
    rmsnorm_rows(w->h, w->h, B->an2, T, D, d->eps, 0);
    {
        const float *gate = mod ? mod + D : NULL;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++)
            for (int i = 0; i < D; i++) x[(int64_t)t * D + i] += (gate ? tanhf(gate[i]) : 1.f) * w->h[(int64_t)t * D + i];
    }
    rmsnorm_rows(w->h, x, B->fn1, T, D, d->eps, 0);
    if (mod) {
        const float *scale = mod + 2 * D;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) for (int i = 0; i < D; i++) w->h[(int64_t)t * D + i] *= 1.f + scale[i];
    }
    linear(w->g, w->h, T, &B->w1); linear(w->u, w->h, T, &B->w3);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * d->mlp; i++) w->g[i] = silu(w->g[i]) * w->u[i];
    linear(w->h, w->g, T, &B->w2);
    rmsnorm_rows(w->h, w->h, B->fn2, T, D, d->eps, 0);
    {
        const float *gate = mod ? mod + 3 * D : NULL;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++)
            for (int i = 0; i < D; i++) x[(int64_t)t * D + i] += (gate ? tanhf(gate[i]) : 1.f) * w->h[(int64_t)t * D + i];
    }
}

#define SEQ_MULTI_OF 32
static int pad32(int n){ return (n + SEQ_MULTI_OF - 1) / SEQ_MULTI_OF * SEQ_MULTI_OF; }

/* What a prompt contributes, which does not depend on the step: the caption tokens through cap_embedder, the pad
 * token and the two context-refiner blocks. cap_out[Nc][dim], Nc = L rounded up to 32. The pad rows take
 * positions on the frame axis like any other row (1 + i), as diffusers' truncation of its position list makes them. */
typedef struct { int L, Nc; float *x; float *cs, *sn; } Caption;

static void dit_caption(Dit *d, const float *emb, int L, Caption *c){
    int D = d->dim, Nc = pad32(L);
    c->L = L; c->Nc = Nc;
    float *n = fmalloc((size_t)L * d->cap_dim);
    rmsnorm_rows(n, emb, d->cap_norm, L, d->cap_dim, d->eps, 0);
    c->x = fmalloc((size_t)Nc * D);
    linear(c->x, n, L, &d->cap_emb);
    free(n);
    for (int t = L; t < Nc; t++) memcpy(c->x + (size_t)t * D, d->cap_pad, (size_t)D * 4);
    int *pos = xmalloc(sizeof(int) * 3 * (size_t)Nc);
    for (int t = 0; t < Nc; t++) { pos[3 * t] = 1 + t; pos[3 * t + 1] = 0; pos[3 * t + 2] = 0; }
    c->cs = fmalloc((size_t)Nc * d->hd / 2); c->sn = fmalloc((size_t)Nc * d->hd / 2);
    dit_rope_table(d, pos, Nc, c->cs, c->sn);
    free(pos);
    DitScratch w; scratch_alloc(&w, d, Nc);
    for (int l = 0; l < d->nref; l++) zblock(d, &d->CR[l], c->x, Nc, NULL, c->cs, c->sn, &w);
    scratch_free(&w);
}
static void caption_free(Caption *c){ free(c->x); free(c->cs); free(c->sn); memset(c, 0, sizeof *c); }

/* The latent grid: gh x gw latent pixels (both even); tokens are 2x2 patches. */
typedef struct { int gh, gw, th, tw, N, Nx; float *cs, *sn; } ZGrid;

static void zgrid_init(ZGrid *g, const Dit *d, int gh, int gw, int Nc){
    g->gh = gh; g->gw = gw; g->th = gh / d->patch; g->tw = gw / d->patch;
    g->N = g->th * g->tw; g->Nx = pad32(g->N);
    int *pos = xmalloc(sizeof(int) * 3 * (size_t)g->Nx);
    for (int y = 0; y < g->th; y++)
        for (int x = 0; x < g->tw; x++) {
            int t = y * g->tw + x;
            pos[3 * t] = Nc + 1; pos[3 * t + 1] = y; pos[3 * t + 2] = x;
        }
    for (int t = g->N; t < g->Nx; t++) pos[3 * t] = pos[3 * t + 1] = pos[3 * t + 2] = 0;
    g->cs = fmalloc((size_t)g->Nx * d->hd / 2); g->sn = fmalloc((size_t)g->Nx * d->hd / 2);
    dit_rope_table(d, pos, g->Nx, g->cs, g->sn);
    free(pos);
}
static void zgrid_free(ZGrid *g){ free(g->cs); free(g->sn); memset(g, 0, sizeof *g); }

/* The latent [16][gh][gw] -> tokens [N][patch*patch*16], token (y, x), value index (ph * patch + pw) * 16 + c. */
static void patchify(const Dit *d, const float *lat, const ZGrid *g, float *tok){
    int P = d->patch, C = d->in_ch;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < g->th; y++)
        for (int x = 0; x < g->tw; x++) {
            float *o = tok + ((int64_t)y * g->tw + x) * P * P * C;
            for (int ph = 0; ph < P; ph++)
                for (int pw = 0; pw < P; pw++)
                    for (int c = 0; c < C; c++)
                        o[(ph * P + pw) * C + c] = lat[((int64_t)c * g->gh + y * P + ph) * g->gw + x * P + pw];
        }
}
static void unpatchify(const Dit *d, const float *tok, const ZGrid *g, float *lat){
    int P = d->patch, C = d->in_ch;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < g->th; y++)
        for (int x = 0; x < g->tw; x++) {
            const float *o = tok + ((int64_t)y * g->tw + x) * P * P * C;
            for (int ph = 0; ph < P; ph++)
                for (int pw = 0; pw < P; pw++)
                    for (int c = 0; c < C; c++)
                        lat[((int64_t)c * g->gh + y * P + ph) * g->gw + x * P + pw] = o[(ph * P + pw) * C + c];
        }
}

/* Debug taps for the oracle: the tensors after the noise refiner, after the context refiner, and after chosen main layers. */
typedef struct { float *after_noise_refiner, *after_context_refiner; float *after_layer[4]; int layer_at[4]; } DitTap;

#ifdef COLI_VULKAN
#include "zimage_chain.h"   /* COLI_VK_CHAIN: the 30 main blocks on the device */
#endif

/* One model evaluation: the noise prediction (the model's output before the pipeline negates it) for the latent
 * lat[16][gh][gw] at model time t_in (= 1000 - scheduler timestep), written to out[16][gh][gw]. */
static void dit_forward(Dit *d, const Caption *cap, const ZGrid *g, const float *lat, float t_in, float *out, DitTap *tap){
    int D = d->dim, Nx = g->Nx, Nc = cap->Nc, T = Nx + Nc;
    float *adaln = fmalloc(256), *mod = fmalloc(4 * (size_t)D), *s = fmalloc(256);
    dit_temb(d, t_in, adaln);

    /* image tokens: embed, pad rows, the noise refiner */
    float *tok = fmalloc((size_t)g->N * d->patch * d->patch * d->in_ch);
    patchify(d, lat, g, tok);
    float *u = fmalloc((size_t)T * D);              /* the unified sequence [x ; caption] */
    linear(u, tok, g->N, &d->x_emb);
    free(tok);
    for (int t = g->N; t < Nx; t++) memcpy(u + (size_t)t * D, d->x_pad, (size_t)D * 4);
    DitScratch w; scratch_alloc(&w, d, T);
    for (int l = 0; l < d->nref; l++) {
        zblock_mod(&d->NR[l], adaln, mod);
        zblock(d, &d->NR[l], u, Nx, mod, g->cs, g->sn, &w);
    }
    if (tap && tap->after_noise_refiner) memcpy(tap->after_noise_refiner, u, (size_t)Nx * D * 4);
    if (tap && tap->after_context_refiner) memcpy(tap->after_context_refiner, cap->x, (size_t)Nc * D * 4);

    /* the unified sequence and the main layers */
    memcpy(u + (size_t)Nx * D, cap->x, (size_t)Nc * D * 4);
    float *cs = fmalloc((size_t)T * d->hd / 2), *sn = fmalloc((size_t)T * d->hd / 2);
    memcpy(cs, g->cs, (size_t)Nx * d->hd / 2 * 4); memcpy(sn, g->sn, (size_t)Nx * d->hd / 2 * 4);
    memcpy(cs + (size_t)Nx * d->hd / 2, cap->cs, (size_t)Nc * d->hd / 2 * 4);
    memcpy(sn + (size_t)Nx * d->hd / 2, cap->sn, (size_t)Nc * d->hd / 2 * 4);
    /* the 30 blocks: on the device as one chain when it takes them (not while an oracle wants the taps between blocks) */
    int dev = 0;
#ifdef COLI_VULKAN
    if (!tap) dev = zic_layers(d, u, T, Nx, cs, sn, adaln);
#endif
    for (int l = 0; !dev && l < d->layers; l++) {
        zblock_mod(&d->B[l], adaln, mod);
        zblock(d, &d->B[l], u, T, mod, cs, sn, &w);
        if (tap)
            for (int j = 0; j < 4; j++)
                if (tap->after_layer[j] && tap->layer_at[j] == l) memcpy(tap->after_layer[j], u, (size_t)Nx * D * 4);
    }

    /* the final layer on the image rows: LayerNorm * (1 + scale), then the projection back to patch values */
    for (int i = 0; i < d->adaln; i++) s[i] = silu(adaln[i]);
    float *scale = fmalloc(D);
    linear(scale, s, 1, &d->fin_mod);
    layernorm_mod(w.h, u, scale, g->N, D, 1e-6f);
    float *tokout = fmalloc((size_t)g->N * d->patch * d->patch * d->in_ch);
    linear(tokout, w.h, g->N, &d->fin_out);
    unpatchify(d, tokout, g, out);
    free(tokout); free(scale); free(cs); free(sn); free(u); scratch_free(&w);
    free(adaln); free(mod); free(s);
}

/* ---- scheduler: flow-matching Euler with a fixed shift ---------------------------- */

typedef struct { double shift; } Sched;
static void sched_config(Sched *s, const char *model){
    char dir[2048]; snprintf(dir, sizeof dir, "%s/scheduler", model);
    char *arena = NULL;
    jval *c = read_json(dir, "scheduler_config.json", &arena);
    s->shift = jnum(c, "shift", 3.0);
    int dyn = jbool(c, "use_dynamic_shifting", 0);
    json_free(c); free(arena);
    if (dyn || s->shift <= 0) {
        fprintf(stderr, "[zimage] scheduler_config.json: only the fixed-shift Euler schedule is implemented\n"); exit(1);
    }
}
/* sig[steps+1] (last = 0) and ts[steps] = sig * 1000. The pipeline's own sigmas are linspace(1, 1/steps, steps)
 * (float32), shifted: s' = shift * s / (1 + (shift - 1) * s). */
static void sched_sigmas(const Sched *s, int steps, float *sig, float *ts){
    for (int i = 0; i < steps; i++) {
        float t = steps == 1 ? 1.f : (float)(1.0 + (1.0 / steps - 1.0) * i / (steps - 1));
        sig[i] = (float)(s->shift * t / (1.0 + (s->shift - 1.0) * t));
    }
    sig[steps] = 0.f;
    for (int i = 0; i < steps; i++) ts[i] = sig[i] * 1000.f;
}

/* ---- VAE decoder (a 16-channel AutoencoderKL, the Flux one) ----------------------------- */

/* Everything is NHWC f32: a pixel's channels are contiguous, so a 3x3 convolution is one matrix product over
 * im2col rows (taps in (ky, kx, channel) order) and its output rows are already NHWC. Convolutions run in bands of
 * rows so that the im2col matrix stays a few hundred MB at any size. */
typedef struct { int cin, cout, k; float *w, *b; } Conv;
typedef struct { float *g, *b; } GNorm;
typedef struct { GNorm n1, n2; Conv c1, c2, sc; int has_sc, cin, cout; } VRes;
typedef struct { GNorm gn; Lin q, k, v, o; } VAttn;
typedef struct {
    Conv in, out; GNorm nout;
    VRes mid[2]; VAttn att;
    VRes up[4][3]; Conv upc[4];
    int ch[4];                   /* channels of the up blocks, in decoding order */
    int groups, latent_ch; float eps, scaling, shift;
    int loaded;
} Vae;

static void conv_load(shards *S, const char *name, int cin, int cout, int k, Conv *c){
    char wn[256], bn[256];
    snprintf(wn, sizeof wn, "%s.weight", name); snprintf(bn, sizeof bn, "%s.bias", name);
    int64_t n = (int64_t)cout * cin * k * k;
    need_tensor(S, wn, cout, (int64_t)cin * k * k);
    float *raw = fmalloc((size_t)n);
    st_read_f32_cap(S, wn, raw, n, 1);
    c->cin = cin; c->cout = cout; c->k = k;
    c->w = fmalloc((size_t)n);
    for (int co = 0; co < cout; co++)
        for (int ci = 0; ci < cin; ci++)
            for (int ky = 0; ky < k; ky++)
                for (int kx = 0; kx < k; kx++)
                    c->w[(int64_t)co * k * k * cin + (ky * k + kx) * cin + ci] = raw[(((int64_t)co * cin + ci) * k + ky) * k + kx];
    free(raw);
    c->b = vec_load(S, bn, cout);
}
static void conv_free(Conv *c){ free(c->w); free(c->b); memset(c, 0, sizeof *c); }
static void gnorm_load(shards *S, const char *name, int C, GNorm *g){
    char n[256];
    snprintf(n, sizeof n, "%s.weight", name); g->g = vec_load(S, n, C);
    snprintf(n, sizeof n, "%s.bias", name); g->b = vec_load(S, n, C);
}
static void gnorm_free(GNorm *g){ free(g->g); free(g->b); memset(g, 0, sizeof *g); }

static void vres_load(shards *S, const char *pre, int cin, int cout, VRes *r){
    char n[256];
    r->cin = cin; r->cout = cout;
    snprintf(n, sizeof n, "%s.norm1", pre); gnorm_load(S, n, cin, &r->n1);
    snprintf(n, sizeof n, "%s.conv1", pre); conv_load(S, n, cin, cout, 3, &r->c1);
    snprintf(n, sizeof n, "%s.norm2", pre); gnorm_load(S, n, cout, &r->n2);
    snprintf(n, sizeof n, "%s.conv2", pre); conv_load(S, n, cout, cout, 3, &r->c2);
    r->has_sc = cin != cout;
    if (r->has_sc) { snprintf(n, sizeof n, "%s.conv_shortcut", pre); conv_load(S, n, cin, cout, 1, &r->sc); }
}
static void vres_free(VRes *r){
    gnorm_free(&r->n1); gnorm_free(&r->n2); conv_free(&r->c1); conv_free(&r->c2);
    if (r->has_sc) conv_free(&r->sc);
    memset(r, 0, sizeof *r);
}

static void vae_config(Vae *v, const char *model){
    char dir[2048]; snprintf(dir, sizeof dir, "%s/vae", model);
    char *arena = NULL;
    jval *c = read_json(dir, "config.json", &arena);
    v->latent_ch = (int)jnum(c, "latent_channels", 16);
    v->groups = (int)jnum(c, "norm_num_groups", 32);
    v->scaling = (float)jnum(c, "scaling_factor", 1.0);
    v->shift = (float)jnum(c, "shift_factor", 0.0);
    int lpb = (int)jnum(c, "layers_per_block", 2);
    int mid_attn = jbool(c, "mid_block_add_attention", 1);
    int pq = jbool(c, "use_post_quant_conv", 0);
    jval *bo = json_get(c, "block_out_channels");
    int ok = bo && bo->t == J_ARR && bo->len == 4;
    for (int i = 0; ok && i < 4; i++) {
        double x = bo->kids[i]->t == J_NUM ? bo->kids[i]->num : -1;
        ok = x > 0 && x <= 4096;
        if (ok) v->ch[3 - i] = (int)x;               /* the decoder walks the list backwards */
    }
    json_free(c); free(arena);
    v->eps = 1e-6f;
    if (!ok || lpb != 2 || !mid_attn || pq || v->latent_ch <= 0 || v->latent_ch > 64 || v->groups <= 0 || v->groups > 256) {
        fprintf(stderr, "[zimage] vae/config.json: unsupported shape (needs four blocks, layers_per_block 2, a mid attention, no post-quant conv)\n");
        exit(1);
    }
    for (int i = 0; i < 4; i++) if (v->ch[i] % v->groups) { fprintf(stderr, "[zimage] vae/config.json: channels not divisible by the groups\n"); exit(1); }
}

static void vae_load(Vae *v, const char *model){
    if (v->loaded) return;
    char dir[2048]; snprintf(dir, sizeof dir, "%s/vae", model);
    shards S; memset(&S, 0, sizeof S);
    st_init(&S, dir);
    double t0 = now_s();
    char n[256];
    conv_load(&S, "decoder.conv_in", v->latent_ch, v->ch[0], 3, &v->in);
    for (int i = 0; i < 2; i++) { snprintf(n, sizeof n, "decoder.mid_block.resnets.%d", i); vres_load(&S, n, v->ch[0], v->ch[0], &v->mid[i]); }
    int C = v->ch[0];
    gnorm_load(&S, "decoder.mid_block.attentions.0.group_norm", C, &v->att.gn);
#define AT(s) (snprintf(n, sizeof n, "decoder.mid_block.attentions.0.%s", s), n)
    lin_load(&S, AT("to_q.weight"), C, C, 32, &v->att.q); v->att.q.bias = vec_load(&S, AT("to_q.bias"), C);
    lin_load(&S, AT("to_k.weight"), C, C, 32, &v->att.k); v->att.k.bias = vec_load(&S, AT("to_k.bias"), C);
    lin_load(&S, AT("to_v.weight"), C, C, 32, &v->att.v); v->att.v.bias = vec_load(&S, AT("to_v.bias"), C);
    lin_load(&S, AT("to_out.0.weight"), C, C, 32, &v->att.o); v->att.o.bias = vec_load(&S, AT("to_out.0.bias"), C);
#undef AT
    int prev = v->ch[0];
    for (int b = 0; b < 4; b++)
        for (int r = 0; r < 3; r++) {
            snprintf(n, sizeof n, "decoder.up_blocks.%d.resnets.%d", b, r);
            vres_load(&S, n, r == 0 ? prev : v->ch[b], v->ch[b], &v->up[b][r]);
            if (r == 2) prev = v->ch[b];
        }
    for (int b = 0; b < 3; b++) { snprintf(n, sizeof n, "decoder.up_blocks.%d.upsamplers.0.conv", b); conv_load(&S, n, v->ch[b], v->ch[b], 3, &v->upc[b]); }
    gnorm_load(&S, "decoder.conv_norm_out", v->ch[3], &v->nout);
    conv_load(&S, "decoder.conv_out", v->ch[3], 3, 3, &v->out);
    st_destroy(&S);
    v->loaded = 1;
    fprintf(stderr, "[zimage] vae decoder loaded, %.1f s\n", now_s() - t0);
}

static void vae_free(Vae *v){
    if (!v->loaded) return;
    conv_free(&v->in); conv_free(&v->out); gnorm_free(&v->nout);
    for (int i = 0; i < 2; i++) vres_free(&v->mid[i]);
    gnorm_free(&v->att.gn); lin_free(&v->att.q); lin_free(&v->att.k); lin_free(&v->att.v); lin_free(&v->att.o);
    for (int b = 0; b < 4; b++) for (int r = 0; r < 3; r++) vres_free(&v->up[b][r]);
    for (int b = 0; b < 3; b++) conv_free(&v->upc[b]);
    memset(v, 0, sizeof *v);
}

/* y[H][W][cout] = conv(x[H][W][cin]), zero padding, k = 3 or 1 */
static void conv_fwd(const Conv *c, const float *x, int H, int W, float *y){
    int K = c->k * c->k * c->cin;
    QiMat m = { QI_F32, c->cout, K, c->w, NULL, 0 };
    if (c->k == 1) { qi_gemm(y, x, H * W, &m, c->b); return; }
    int R = (int)((48u << 20) / ((size_t)W * K));
    if (R < 1) R = 1;
    if (R > H) R = H;
    float *col = fmalloc((size_t)R * W * K);
    for (int y0 = 0; y0 < H; y0 += R) {
        int rows = R < H - y0 ? R : H - y0;
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < rows * W; p++) {
            int yy = y0 + p / W, xx = p % W;
            float *o = col + (size_t)p * K;
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    int sy = yy + ky - 1, sx = xx + kx - 1;
                    if (sy < 0 || sy >= H || sx < 0 || sx >= W) memset(o, 0, (size_t)c->cin * 4);
                    else memcpy(o, x + ((size_t)sy * W + sx) * c->cin, (size_t)c->cin * 4);
                    o += c->cin;
                }
        }
        qi_gemm(y + (size_t)y0 * W * c->cout, col, rows * W, &m, c->b);
    }
    free(col);
}

/* GroupNorm over [HW][C] (the groups are runs of C / groups channels), optionally followed by SiLU. */
static void gnorm_fwd(float *y, const float *x, const GNorm *gn, int HW, int C, int groups, float eps, int act){
    int cg = C / groups;
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < groups; g++) {
        double s = 0, s2 = 0;
        for (int p = 0; p < HW; p++) {
            const float *r = x + (size_t)p * C + (size_t)g * cg;
            for (int i = 0; i < cg; i++) { s += r[i]; s2 += (double)r[i] * r[i]; }
        }
        double n = (double)HW * cg, mean = s / n, var = s2 / n - mean * mean;
        if (var < 0) var = 0;
        float inv = (float)(1.0 / sqrt(var + eps)), mf = (float)mean;
        for (int p = 0; p < HW; p++) {
            const float *r = x + (size_t)p * C + (size_t)g * cg; float *o = y + (size_t)p * C + (size_t)g * cg;
            for (int i = 0; i < cg; i++) {
                float t = (r[i] - mf) * inv * gn->g[g * cg + i] + gn->b[g * cg + i];
                o[i] = act ? silu(t) : t;
            }
        }
    }
}

/* One residual block: out = shortcut(x) + conv2(silu(norm2(conv1(silu(norm1(x)))))). out must not alias x. */
static void vres_fwd(const Vae *v, const VRes *r, const float *x, int H, int W, float *out, float *t1, float *t2){
    int HW = H * W;
    gnorm_fwd(t1, x, &r->n1, HW, r->cin, v->groups, v->eps, 1);
    conv_fwd(&r->c1, t1, H, W, t2);
    gnorm_fwd(t1, t2, &r->n2, HW, r->cout, v->groups, v->eps, 1);
    conv_fwd(&r->c2, t1, H, W, t2);
    if (r->has_sc) conv_fwd(&r->sc, x, H, W, t1);
    const float *s = r->has_sc ? t1 : x;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)HW * r->cout; i++) out[i] = s[i] + t2[i];
}

/* The mid block's attention: one head of the channel width over all pixels, with a residual connection. */
static void vattn_fwd(const Vae *v, const VAttn *a, float *x, int H, int W, float *t1, float *t2, float *t3, float *t4){
    int HW = H * W, C = v->ch[0];
    gnorm_fwd(t1, x, &a->gn, HW, C, v->groups, v->eps, 0);
    linear(t2, t1, HW, &a->q); linear(t3, t1, HW, &a->k); linear(t4, t1, HW, &a->v);
    /* queries in blocks: the score matrix is blocks x HW floats, not HW x HW */
    int blk = (int)((256u << 20) / ((size_t)HW * 4)); if (blk < 64) blk = 64; if (blk > HW) blk = HW;
    for (int q0 = 0; q0 < HW; q0 += blk) {
        int n = blk < HW - q0 ? blk : HW - q0;
        attention(t1 + (size_t)q0 * C, C, t2 + (size_t)q0 * C, C, n, t3, t4, C, HW, 1, 1, C, -1, NULL);
    }
    linear(t2, t1, HW, &a->o);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)HW * C; i++) x[i] += t2[i];
}

/* z[latent_ch][gh][gw] (the sampler's latent, NCHW) -> image. rgb_f, when not NULL, receives the pixels as [3][H][W] in
 * [-1, 1] before clamping (the reference's `decoded`); rgba, when not NULL, the bytes. H = 8 gh, W = 8 gw. */
static int vae_decode(const Vae *v, const float *z, int gh, int gw, uint8_t *rgba, float *rgb_f){
    int H = gh, W = gw;
    size_t big = 0;
    { int h = gh, w = gw; for (int b = 0; b < 4; b++) {
          int cm = v->ch[b] > (b ? v->ch[b - 1] : v->ch[0]) ? v->ch[b] : (b ? v->ch[b - 1] : v->ch[0]);
          size_t n = (size_t)h * w * cm; if (n > big) big = n; if (b < 3) { h *= 2; w *= 2; } } }
    size_t nin = (size_t)gh * gw * v->latent_ch; if (nin > big) big = nin;
    float *A = fmalloc(big), *Bf = fmalloc(big), *T1 = fmalloc(big), *T2 = fmalloc(big), *T3 = NULL, *T4 = NULL;
    size_t mid = (size_t)gh * gw * v->ch[0];
    T3 = fmalloc(mid); T4 = fmalloc(mid);
    /* NCHW -> NHWC with the pipeline's un-scaling: z / scaling + shift */
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < gh * gw; p++)
        for (int c = 0; c < v->latent_ch; c++) A[(size_t)p * v->latent_ch + c] = z[(size_t)c * gh * gw + p] / v->scaling + v->shift;
    conv_fwd(&v->in, A, H, W, Bf);                       /* Bf: [H][W][ch0] */
    vres_fwd(v, &v->mid[0], Bf, H, W, A, T1, T2);
    vattn_fwd(v, &v->att, A, H, W, T1, T2, T3, T4);
    vres_fwd(v, &v->mid[1], A, H, W, Bf, T1, T2);
    float *cur = Bf, *oth = A;
    for (int b = 0; b < 4; b++) {
        for (int r = 0; r < 3; r++) {
            vres_fwd(v, &v->up[b][r], cur, H, W, oth, T1, T2);
            float *t = cur; cur = oth; oth = t;
        }
        if (b < 3) {                                      /* nearest-neighbour 2x, then a 3x3 convolution */
            int C = v->ch[b];
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < 2 * H; y++)
                for (int x = 0; x < 2 * W; x++)
                    memcpy(oth + ((size_t)y * 2 * W + x) * C, cur + ((size_t)(y / 2) * W + x / 2) * C, (size_t)C * 4);
            H *= 2; W *= 2;
            conv_fwd(&v->upc[b], oth, H, W, cur);
        }
    }
    gnorm_fwd(T1, cur, &v->nout, H * W, v->ch[3], v->groups, v->eps, 1);
    conv_fwd(&v->out, T1, H, W, oth);                    /* oth: [H][W][3] */
    if (rgb_f)
        for (int c = 0; c < 3; c++) for (int p = 0; p < H * W; p++) rgb_f[(size_t)c * H * W + p] = oth[(size_t)p * 3 + c];
    if (rgba)
        for (int p = 0; p < H * W; p++) {
            for (int c = 0; c < 3; c++) {
                float t = (oth[(size_t)p * 3 + c] + 1.f) * 127.5f;
                t = t < 0 ? 0 : t > 255 ? 255 : t;
                rgba[(size_t)p * 4 + c] = (uint8_t)(t + 0.5f);
            }
            rgba[(size_t)p * 4 + 3] = 255;
        }
    free(A); free(Bf); free(T1); free(T2); free(T3); free(T4);
    return 0;
}

/* ---- RNG: splitmix64 + Box-Muller -------------------------------------------- */

static uint64_t g_rng;
static uint64_t rng_next(void){
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static void randn(float *x, size_t n, uint64_t seed){
    g_rng = seed * 0x2545F4914F6CDD1Dull + 1;
    for (size_t i = 0; i < n; i += 2) {
        double u1 = ((rng_next() >> 11) + 1.0) / 9007199254740993.0, u2 = (rng_next() >> 11) / 9007199254740992.0;
        double r = sqrt(-2.0 * log(u1)), a = 6.283185307179586 * u2;
        x[i] = (float)(r * cos(a));
        if (i + 1 < n) x[i + 1] = (float)(r * sin(a));
    }
}

/* ---- PNG (stored deflate blocks: valid, uncompressed, no zlib) ---------------- */

static uint32_t crc_tab[256];
static void crc_init(void){
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1;
        crc_tab[n] = c;
    }
}
static uint32_t crc_upd(uint32_t c, const uint8_t *b, size_t n){
    for (size_t i = 0; i < n; i++) c = crc_tab[(c ^ b[i]) & 0xff] ^ (c >> 8);
    return c;
}
static void be32(uint8_t *p, uint32_t v){ p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static void png_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n){
    uint8_t h[8]; be32(h, n); memcpy(h + 4, type, 4);
    fwrite(h, 1, 8, f);
    if (n) fwrite(data, 1, n, f);
    uint32_t c = crc_upd(0xffffffffu, (const uint8_t *)type, 4);
    c = crc_upd(c, data, n) ^ 0xffffffffu;
    uint8_t t[4]; be32(t, c); fwrite(t, 1, 4, f);
}
static int write_png(const char *path, const uint8_t *rgba, int w, int h){
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    crc_init();
    static const uint8_t sig[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
    fwrite(sig, 1, 8, f);
    uint8_t ih[13]; be32(ih, w); be32(ih + 4, h); ih[8] = 8; ih[9] = 6; ih[10] = ih[11] = ih[12] = 0;
    png_chunk(f, "IHDR", ih, 13);
    size_t raw_n = (size_t)h * (1 + (size_t)w * 4);
    uint8_t *raw = xmalloc(raw_n);
    for (int y = 0; y < h; y++) { raw[(size_t)y * (1 + w * 4)] = 0; memcpy(raw + (size_t)y * (1 + w * 4) + 1, rgba + (size_t)y * w * 4, (size_t)w * 4); }
    size_t blocks = (raw_n + 65534) / 65535;
    size_t z_n = 2 + raw_n + blocks * 5 + 4;
    uint8_t *z = xmalloc(z_n), *o = z;
    *o++ = 0x78; *o++ = 0x01;
    uint32_t a = 1, b = 0;
    for (size_t off = 0; off < raw_n; off += 65535) {
        size_t n = raw_n - off < 65535 ? raw_n - off : 65535;
        *o++ = off + n == raw_n ? 1 : 0;
        *o++ = n & 0xff; *o++ = n >> 8; *o++ = ~n & 0xff; *o++ = (~n >> 8) & 0xff;
        memcpy(o, raw + off, n); o += n;
    }
    for (size_t i = 0; i < raw_n; i++) { a = (a + raw[i]) % 65521; b = (b + a) % 65521; }
    be32(o, (b << 16) | a); o += 4;
    png_chunk(f, "IDAT", z, (uint32_t)(o - z));
    png_chunk(f, "IEND", NULL, 0);
    free(raw); free(z);
    return fclose(f);
}

/* ---- serve ------------------------------------------------------------------- */

/* Our own line reader on fd 0: stdio's buffer would hide a CANCEL that arrived
 * together with the GEN from coli_serve_stdin_ready(). */
/* The buffer grows up to QI_MAX_LINE: room for the pictures of an edit, base64 inside the GEN line
 * (QI_MAX_EDIT_SRC pictures of up to 2048x2048 RGBA, the gateway's own limit, is about 90 MB). A longer
 * line is read through to its newline and discarded, and read_line hands back
 * "TOOLONG <id>" with the id found at its start, so the client that sent it
 * gets an ERROR instead of waiting forever for an answer to a request the
 * engine never saw. */
#define QI_MAX_LINE ((size_t)96 << 20)
static char *g_in; static size_t g_in_n, g_in_cap;
static int g_skipping; static char g_skip_id[128];

static void note_line_id(const char *buf, size_t n){
    snprintf(g_skip_id, sizeof g_skip_id, "?");
    const char *k = NULL;                /* memmem is not in the Windows CRT */
    for (size_t i = 0; i + 6 <= (n < 4096 ? n : 4096); i++)
        if (!memcmp(buf + i, "\"id\":\"", 6)) { k = buf + i; break; }
    if (!k) return;
    k += 6;
    size_t i = 0;
    while (i + 1 < sizeof g_skip_id && k + i < buf + n && k[i] != '"' && k[i] != '\\' && (unsigned char)k[i] >= 0x20) {
        g_skip_id[i] = k[i]; i++;
    }
    g_skip_id[i] = 0;
}

static char *read_line(int block){
    for (;;) {
        char *nl = g_in_n ? memchr(g_in, '\n', g_in_n) : NULL;
        if (nl) {
            size_t len = (size_t)(nl - g_in);
            char *line;
            if (g_skipping) {                 /* the tail of an overlong line */
                g_skipping = 0;
                line = xmalloc(sizeof g_skip_id + 16);
                snprintf(line, sizeof g_skip_id + 16, "TOOLONG %s", g_skip_id);
            } else {
                line = xmalloc(len + 1);
                memcpy(line, g_in, len); line[len] = 0;
            }
            memmove(g_in, nl + 1, g_in_n - len - 1); g_in_n -= len + 1;
            return line;
        }
        if (g_in_n == QI_MAX_LINE) {
            if (!g_skipping) { note_line_id(g_in, g_in_n); g_skipping = 1; }
            g_in_n = 0;
        }
        if (!block && !coli_serve_stdin_ready()) return NULL;
        if (g_in_n == g_in_cap) {
            size_t cap = g_in_cap ? g_in_cap * 2 : (size_t)1 << 16;
            if (cap > QI_MAX_LINE) cap = QI_MAX_LINE;
            char *grown = realloc(g_in, cap);
            if (!grown) { fprintf(stderr, "[zimage] out of memory reading stdin\n"); exit(1); }
            g_in = grown; g_in_cap = cap;
        }
        ssize_t r = read(0, g_in + g_in_n, g_in_cap - g_in_n);
        if (r <= 0) return NULL;
        g_in_n += (size_t)r;
    }
}

static void send_error(const char *id, const char *m);
static int g_cancel_pending;
static char g_cancel_id[128];
/* Lines that arrive while an image is being made and are not a CANCEL for it
 * (a GEN from a client that gave up waiting, say) are kept for serve_loop,
 * never dropped: a dropped GEN is a client waiting forever for its answer. */
static char *g_later[64]; static int g_later_n;
static int serve_cancelled(const char *id){
    char *line;
    while ((line = read_line(0))) {
        if (strncmp(line, "CANCEL ", 7)) {
            if (g_later_n < (int)(sizeof g_later / sizeof g_later[0])) { g_later[g_later_n++] = line; continue; }
            if (!strncmp(line, "GEN ", 4)) {        /* the queue is full: answer rather than drop */
                char *arena = NULL; jval *j = json_parse(line + 4, &arena);
                jval *v = j ? json_get(j, "id") : NULL;
                send_error(v && v->t == J_STR ? v->str : "?", "engine busy");
                if (j) json_free(j);
                free(arena);
            }
            free(line);
            continue;
        }
        {
            char *arena = NULL; jval *j = json_parse(line + 7, &arena);
            jval *v = j ? json_get(j, "id") : NULL;
            if (v && v->t == J_STR) { snprintf(g_cancel_id, sizeof g_cancel_id, "%s", v->str); g_cancel_pending = 1; }
            if (j) json_free(j);
            free(arena);
        }
        free(line);
    }
    return g_cancel_pending && !strcmp(g_cancel_id, id);
}

static void json_escape_out(const char *s){
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') printf("\\%c", c);
        else if (c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
}
static void send_error(const char *id, const char *m){
    printf("ERROR {\"id\":\""); json_escape_out(id); printf("\",\"message\":\""); json_escape_out(m); printf("\"}\n");
    fflush(stdout);
}

static void send_image(const char *id, int w, int h, uint64_t seed, int steps, const double tm[3], const uint8_t *rgba){
    printf("IMAGE {\"id\":\""); json_escape_out(id);
    printf("\",\"width\":%d,\"height\":%d,\"channels\":4,\"bytes\":%lld,\"seed\":%llu,\"steps\":%d,"
           "\"timings\":{\"encode\":%.2f,\"denoise\":%.2f,\"decode\":%.2f}}\n",
           w, h, (long long)w * h * 4, (unsigned long long)seed, steps, tm[0], tm[1], tm[2]);
    fwrite(rgba, 1, (size_t)w * h * 4, stdout);
    putchar('\n');
    fflush(stdout);
}


/* ---- the pipeline ------------------------------------------------------------ */

typedef struct {
    char model[2048];
    Te te; Dit dit; Sched sch; Vae vae;
    int te_resident;
    /* the last prompt's caption (its embedding through the transformer's caption path): a new seed or size for the
     * same prompt skips the text encoder entirely */
    char *last_prompt; Caption cap;
} Engine;

typedef struct { const char *id; int (*cancelled)(const char *id); int preview; } Progress;

static void json_escape_out(const char *s);
static void send_error(const char *id, const char *m);

static void emit_progress(const Progress *pg, const char *stage, int step, int steps, double t0){
    if (!pg || !pg->id) return;
    printf("PROGRESS {\"id\":\""); json_escape_out(pg->id);
    printf("\",\"stage\":\"%s\",\"step\":%d,\"steps\":%d,\"elapsed\":%.2f}\n", stage, step, steps, now_s() - t0);
    fflush(stdout);
}

static void engine_init(Engine *e, const char *model){
    memset(e, 0, sizeof *e);
    snprintf(e->model, sizeof e->model, "%s", model);
    te_config(&e->te, model);
    dit_config(&e->dit, model);
    sched_config(&e->sch, model);
    vae_config(&e->vae, model);
    te_tokenizer(&e->te, model);
    if (e->te.hidden != e->dit.cap_dim) {
        fprintf(stderr, "[zimage] text encoder hidden %d != transformer caption width %d\n", e->te.hidden, e->dit.cap_dim);
        exit(1);
    }
    /* checked now, so a VAE that does not fit the transformer fails at start-up and not after a whole denoise */
    if (e->vae.latent_ch != e->dit.in_ch) {
        fprintf(stderr, "[zimage] the VAE has %d latent channels, the transformer %d\n", e->vae.latent_ch, e->dit.in_ch);
        exit(1);
    }
}

/* The pipeline truncates the templated prompt to 512 tokens (max_sequence_length); so do we. */
#define ZI_MAX_PROMPT_TOKENS 512

/* The prompt's caption, cached for the next request with the same prompt. Returns 0, or -1 with msg. */
static int engine_caption(Engine *e, const char *prompt, char *msg, size_t msgn){
    if (e->last_prompt && !strcmp(e->last_prompt, prompt) && e->cap.x) return 0;
    caption_free(&e->cap);
    free(e->last_prompt); e->last_prompt = NULL;
    int n;
    int *ids = te_encode_prompt(&e->te, prompt, &n);
    if (n <= 0) { snprintf(msg, msgn, "empty prompt"); free(ids); return -1; }
    if (n > ZI_MAX_PROMPT_TOKENS) n = ZI_MAX_PROMPT_TOKENS;
    te_load(&e->te, e->model);
    float *h = te_forward(&e->te, ids, n);
    free(ids);
    if (!e->te_resident) te_unload(&e->te);
    dit_load(&e->dit, e->model);
    dit_caption(&e->dit, h, n, &e->cap);
    free(h);
    e->last_prompt = strdup(prompt);
    return 0;
}

static int check_size(int w, int h, int steps, char *msg, size_t n){
    if (w < 256 || h < 256 || w > 2048 || h > 2048 || w % 16 || h % 16) {
        snprintf(msg, n, "width and height must be multiples of 16 between 256 and 2048 (got %dx%d)", w, h);
        return -1;
    }
    if (steps < 1 || steps > 200) { snprintf(msg, n, "steps must be between 1 and 200 (got %d)", steps); return -1; }
    return 0;
}

/* Everything after the caption: noise (or lat_init, the oracle's own), the denoising loop, and the decode.
 * rgba is [height][width][4]. */
static int engine_sample(Engine *e, int width, int height, int steps, uint64_t seed, const float *lat_init,
                         uint8_t *rgba, double timings[3], const Progress *pg, double t0, double t1,
                         char *msg, size_t msgn){
    int gh = height / 8, gw = width / 8, C = e->dit.in_ch;
    size_t nlat = (size_t)C * gh * gw;
    float *lat = fmalloc(nlat), *np = fmalloc(nlat);
    if (lat_init) memcpy(lat, lat_init, nlat * 4); else randn(lat, nlat, seed);
    float sig[256], ts[256];
    sched_sigmas(&e->sch, steps, sig, ts);
    ZGrid grid; zgrid_init(&grid, &e->dit, gh, gw, e->cap.Nc);
    int rc = 0;
    for (int i = 0; i < steps; i++) {
        emit_progress(pg, "denoise", i, steps, t0);
        if (pg && pg->cancelled && pg->cancelled(pg->id)) { snprintf(msg, msgn, "cancelled"); rc = -1; break; }
        double ts0 = now_s();
        g_t_lin = g_t_att = 0;
        dit_forward(&e->dit, &e->cap, &grid, lat, 1000.f - ts[i], np, NULL);
        float dt = sig[i + 1] - sig[i];
        for (size_t k = 0; k < nlat; k++) lat[k] -= dt * np[k];     /* the pipeline negates the output, then Euler */
        double tstep = now_s() - ts0;
        fprintf(stderr, "[zimage] step %d/%d  t=%.1f  %.2f s", i + 1, steps, ts[i], tstep);
        if (g_prof) fprintf(stderr, "  (linears %.2f, attention %.2f, rest %.2f)", g_t_lin, g_t_att, tstep - g_t_lin - g_t_att);
        fputc('\n', stderr);
    }
    zgrid_free(&grid);
    double t2 = now_s();
    if (rc == 0) {
        emit_progress(pg, "decode", steps, steps, t0);
        vae_load(&e->vae, e->model);
        if (vae_decode(&e->vae, lat, gh, gw, rgba, NULL) != 0) { snprintf(msg, msgn, "VAE decode failed"); rc = -1; }
    }
    double t3 = now_s();
    if (timings) { timings[0] = t1 - t0; timings[1] = t2 - t1; timings[2] = t3 - t2; }
    free(lat); free(np);
    return rc;
}

/* Generates one image. rgba is [height][width][4]. Returns 0, or -1 with msg. */
static int engine_generate(Engine *e, const char *prompt, int width, int height, int steps, uint64_t seed,
                           uint8_t *rgba, double timings[3], const Progress *pg, char *msg, size_t msgn){
    double t0 = now_s();
    emit_progress(pg, "encode", 0, steps, t0);
    if (engine_caption(e, prompt, msg, msgn) < 0) return -1;
    return engine_sample(e, width, height, steps, seed, NULL, rgba, timings, pg, t0, now_s(), msg, msgn);
}

static int serve_loop(Engine *e, int dw, int dh, int dsteps){
    printf("READY {\"model\":\"z-image\",\"default_width\":%d,\"default_height\":%d,\"default_steps\":%d,"
           "\"min_side\":256,\"max_side\":2048,\"multiple\":16}\n", dw, dh, dsteps);
    fflush(stdout);
    char *line;
    for (;;) {
        if (g_later_n) {                              /* what came in during the last image */
            line = g_later[0];
            memmove(g_later, g_later + 1, sizeof g_later[0] * (size_t)--g_later_n);
        } else if (!(line = read_line(1))) break;
        if (!strncmp(line, "TOOLONG ", 8)) { send_error(line + 8, "request line too long"); free(line); continue; }
        if (strncmp(line, "GEN ", 4)) { free(line); continue; }       /* CANCEL with nothing running, noise */
        char *arena = NULL;
        jval *j = json_parse(line + 4, &arena);
        char id[128] = "?", msg[512] = "";
        jval *v;
        if (j && (v = json_get(j, "id")) && v->t == J_STR) snprintf(id, sizeof id, "%s", v->str);
        const char *prompt = j && (v = json_get(j, "prompt")) && v->t == J_STR ? v->str : NULL;
        int w = (int)jnum(j, "width", dw), h = (int)jnum(j, "height", dh), steps = (int)jnum(j, "steps", dsteps);
        uint64_t seed = (uint64_t)jnum(j, "seed", (double)(time(NULL) & 0x7fffffff));
        if (!j) send_error(id, "bad GEN line (not JSON)");
        else if (!prompt || !*prompt) send_error(id, "empty prompt");
        else if (json_get(j, "images")) send_error(id, "this model does not edit pictures");
        else if (check_size(w, h, steps, msg, sizeof msg)) send_error(id, msg);
        else {
            g_cancel_pending = 0;
            uint8_t *rgba = xmalloc((size_t)w * h * 4);
            double tm[3];
            Progress pg = { id, serve_cancelled, 0 };
            if (engine_generate(e, prompt, w, h, steps, seed, rgba, tm, &pg, msg, sizeof msg)) send_error(id, msg);
            else send_image(id, w, h, seed, steps, tm, rgba);
            free(rgba);
        }
        if (j) json_free(j);
        free(arena); free(line);
    }
    return 0;
}

/* ---- oracle: every stage against a reference dump (tools/zimage_ref.py) ----------- */

/* Compare a computed tensor with the reference: max abs error and the ratio of norms of the error and the reference. */
static double cmp(const char *what, const float *got, const float *want, size_t n){
    double emax = 0, en = 0, wn = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)got[i] - want[i];
        if (fabs(d) > emax) emax = fabs(d);
        en += d * d; wn += (double)want[i] * want[i];
    }
    double rel = wn > 0 ? sqrt(en / wn) : sqrt(en);
    fprintf(stderr, "[oracle] %-26s n=%-9zu max|err| %.3e  rel %.3e\n", what, n, emax, rel);
    return rel;
}
static float *ref_tensor(shards *S, const char *name, int64_t *n){
    st_tensor *t = st_find(S, name);
    if (!t) return NULL;
    int64_t numel = 1; for (int i = 0; i < t->rank; i++) numel *= t->shape[i];
    float *v = fmalloc((size_t)numel);
    if (t->dtype == 6) {    /* int64 ids */
        int64_t *raw = xmalloc((size_t)numel * 8);
        st_pread_full(t->fd, raw, numel * 8, t->off, "ref");
        for (int64_t i = 0; i < numel; i++) v[i] = (float)raw[i];
        free(raw);
    } else st_read_f32(S, name, v, 0);
    if (n) *n = numel;
    return v;
}

static int run_oracle(const char *model, const char *refdir){
    shards R; memset(&R, 0, sizeof R);
    st_init(&R, refdir);
    static Te te;
    te_config(&te, model);
    te_tokenizer(&te, model);
    int rc = 0;

    /* 1. the tokenizer: the reference's prompt text is in prompt_text.txt next to the dump */
    char path[2048]; snprintf(path, sizeof path, "%s/prompt_text.txt", refdir);
    char *ptext = read_file(path);
    int64_t nref = 0;
    float *ids_ref = ref_tensor(&R, "ids", &nref);
    if (!ids_ref) { fprintf(stderr, "[oracle] the dump has no ids\n"); return 1; }
    /* the reference text is the whole chat-templated string; the user's words sit between the markers */
    const char *u0 = ptext ? strstr(ptext, "<|im_start|>user\n") : NULL;
    const char *u1 = ptext ? strstr(ptext, "<|im_end|>\n<|im_start|>assistant") : NULL;
    if (!u0 || !u1) { fprintf(stderr, "[oracle] prompt_text.txt is not the expected chat text\n"); return 1; }
    u0 += strlen("<|im_start|>user\n");
    char *prompt = xmalloc((size_t)(u1 - u0) + 1);
    memcpy(prompt, u0, (size_t)(u1 - u0)); prompt[u1 - u0] = 0;
    int n; int *ids = te_encode_prompt(&te, prompt, &n);
    int same = n == (int)nref;
    for (int i = 0; same && i < n; i++) same = ids[i] == (int)ids_ref[i];
    fprintf(stderr, "[oracle] %-26s %d ids, %s\n", "tokenizer", n, same ? "identical to the reference" : "DIFFERENT");
    if (!same) rc = 1;

    /* 2. the text encoder, from the reference's own ids */
    int T = (int)nref; int *rid = xmalloc(sizeof(int) * T);
    for (int i = 0; i < T; i++) rid[i] = (int)ids_ref[i];
    te_load(&te, model);
    double t0 = now_s();
    float *emb = te_forward(&te, rid, T);
    fprintf(stderr, "[oracle] text encoder forward %.1f s\n", now_s() - t0);
    int64_t ne = 0; float *emb_ref = ref_tensor(&R, "embeds", &ne);
    if (!emb_ref || ne != (int64_t)T * te.hidden) { fprintf(stderr, "[oracle] bad embeds in the dump\n"); return 1; }
    double rel = cmp("text encoder", emb, emb_ref, (size_t)ne);
    /* int8 weights keep a few percent; f32 weights (COLI_IMG_BITS=32) must match to summation order */
    double limit = g_bits == 32 ? 2e-4 : g_bits == 16 ? 2e-2 : 0.12;
    if (!(rel < limit)) { fprintf(stderr, "[oracle] text encoder: rel %.3e is over %.1e\n", rel, limit); rc = 1; }
    free(emb_ref);
    te_unload(&te);

    /* 3. the DiT, from the reference's embedding (not ours): one evaluation at step 0, then the whole loop */
    static Dit dit; static Sched sch;
    dit_config(&dit, model); sched_config(&sch, model);
    dit_load(&dit, model);
    int64_t nl = 0; float *lat0 = ref_tensor(&R, "init_latent", &nl);
    int gh = 0, gw = 0;
    { st_tensor *t = st_find(&R, "init_latent"); if (t && t->rank == 4) { gh = (int)t->shape[2]; gw = (int)t->shape[3]; } }
    if (!lat0 || gh <= 0 || nl != (int64_t)dit.in_ch * gh * gw) { fprintf(stderr, "[oracle] bad init_latent\n"); return 1; }
    int steps = 0; { st_tensor *t = st_find(&R, "timesteps"); if (t) steps = (int)t->shape[0]; }
    float sig[64], ts[64]; if (steps <= 0 || steps > 60) { fprintf(stderr, "[oracle] bad timesteps\n"); return 1; }
    sched_sigmas(&sch, steps, sig, ts);
    float *sig_ref = ref_tensor(&R, "sigmas", NULL), *ts_ref = ref_tensor(&R, "timesteps", NULL);
    double ds = 0; for (int i = 0; i <= steps; i++) ds = fmax(ds, fabs(sig[i] - sig_ref[i]));
    for (int i = 0; i < steps; i++) ds = fmax(ds, fabs(ts[i] - ts_ref[i]) / 1000.0);
    fprintf(stderr, "[oracle] %-26s max |diff| %.2e %s\n", "scheduler sigmas", ds, ds < 1e-6 ? "" : "  <-- DIFFERENT");
    if (ds >= 1e-6) rc = 1;

    Caption cap; double t1 = now_s();
    dit_caption(&dit, emb, T, &cap);                 /* from OUR embedding for the chained run; see below for the reference's */
    ZGrid grid; zgrid_init(&grid, &dit, gh, gw, cap.Nc);
    fprintf(stderr, "[oracle] caption %d rows (%d padded), image %d tokens (%d padded), %.1f s\n", cap.L, cap.Nc, grid.N, grid.Nx, now_s() - t1);
    size_t nlat = (size_t)nl;
    float *pred = fmalloc(nlat), *lat = fmalloc(nlat);

    /* 3a. step 0 from the reference's own embedding and noise: our prediction against the reference's */
    Caption capr; float *embr = ref_tensor(&R, "embeds", NULL);
    dit_caption(&dit, embr, T, &capr);
    DitTap tap; memset(&tap, 0, sizeof tap);
    size_t nx = (size_t)grid.Nx * dit.dim;
    float *tap_noise = fmalloc(nx), *tap_ctx = fmalloc((size_t)capr.Nc * dit.dim);
    tap.after_noise_refiner = tap_noise; tap.after_context_refiner = tap_ctx;
    const int at[3] = { 0, (dit.layers - 1) / 2, dit.layers - 1 };
    float *tap_l[3];
    for (int j = 0; j < 3; j++) { tap_l[j] = fmalloc(nx); tap.after_layer[j] = tap_l[j]; tap.layer_at[j] = at[j]; }
    t1 = now_s();
    dit_forward(&dit, &capr, &grid, lat0, 1000.f - ts[0], pred, &tap);
    fprintf(stderr, "[oracle] one evaluation %.1f s\n", now_s() - t1);
    if (st_find(&R, "dit_pred0")) {
        /* a float32 reference of this one evaluation (tools/zimage_ref_dit.py): the stages where an error is made */
        double lim = g_bits == 32 ? 2e-4 : g_bits == 16 ? 3e-3 : 0.10;
        struct { const char *name; const float *got; size_t n; } stage[] = {
            { "dit_after_noise_refiner", tap_noise, (size_t)grid.N * dit.dim },
            { "dit_after_context_refiner", tap_ctx, (size_t)capr.Nc * dit.dim },
            { "dit_after_layer_first", tap_l[0], (size_t)grid.N * dit.dim },
            { "dit_after_layer_mid", tap_l[1], (size_t)grid.N * dit.dim },
            { "dit_after_layer_last", tap_l[2], (size_t)grid.N * dit.dim },
            { "dit_pred0", pred, nlat },
        };
        for (size_t k = 0; k < sizeof stage / sizeof stage[0]; k++) {
            float *want = ref_tensor(&R, stage[k].name, NULL);
            double r = cmp(stage[k].name, stage[k].got, want, stage[k].n);
            if (!(r < lim)) { fprintf(stderr, "[oracle] %s: rel %.3e is over %.1e\n", stage[k].name, r, lim); rc = 1; }
            free(want);
        }
    }
    if (st_find(&R, "dit_pred0")) {
        /* the same evaluation without taps: on a device build it is the chain's (the taps keep it off the device chain) */
        float *again = fmalloc(nlat), *want = ref_tensor(&R, "dit_pred0", NULL);
        dit_forward(&dit, &capr, &grid, lat0, 1000.f - ts[0], again, NULL);
        double r = cmp("dit_pred0 (no taps)", again, want, nlat);
        double lim = g_bits == 32 ? 2e-4 : g_bits == 16 ? 3e-3 : 0.10;
        if (!(r < lim)) { fprintf(stderr, "[oracle] dit_pred0 without taps: rel %.3e is over %.1e\n", r, lim); rc = 1; }
        free(again); free(want);
    }
    {
        float *l1 = ref_tensor(&R, "latent_step0", NULL);
        float *pr = fmalloc(nlat);
        for (size_t i = 0; i < nlat; i++) pr[i] = -(l1[i] - lat0[i]) / (sig[1] - sig[0]);   /* the pipeline negates the output, then Euler */
        double r = cmp("noise prediction, step 0", pred, pr, nlat);
        if (!(r < (g_bits == 8 ? 0.15 : 0.05))) { fprintf(stderr, "[oracle] DiT step 0: rel %.3e is too far\n", r); rc = 1; }
        free(l1); free(pr);
    }
    free(tap_noise); free(tap_ctx); for (int j = 0; j < 3; j++) free(tap_l[j]);

    /* 3b. the whole loop, each step from OUR latent: the error after each step against the reference's latent */
    memcpy(lat, lat0, nlat * 4);
    t1 = now_s();
    for (int i = 0; i < steps; i++) {
        dit_forward(&dit, &capr, &grid, lat, 1000.f - ts[i], pred, NULL);
        for (size_t j = 0; j < nlat; j++) lat[j] += (sig[i + 1] - sig[i]) * -pred[j];
        char nm[48]; snprintf(nm, sizeof nm, "latent_step%d", i);
        float *lr = ref_tensor(&R, nm, NULL);
        char what[48]; snprintf(what, sizeof what, "latent after step %d", i);
        double r = cmp(what, lat, lr, nlat);
        /* the reference loop runs in bf16 and the two drift apart as the steps go (more with few steps): this only
         * guards against a gross error; the strict check is the float32 evaluation above */
        if (i == steps - 1 && !(r < 0.25)) { fprintf(stderr, "[oracle] final latent: rel %.3e is too far\n", r); rc = 1; }
        free(lr);
    }
    fprintf(stderr, "[oracle] %d steps %.1f s\n", steps, now_s() - t1);
    free(pred); free(lat); free(lat0); free(embr); free(sig_ref); free(ts_ref);

    /* 4. the VAE: the reference's final latent through our decoder against its float32 decode */
    {
        static Vae vae;
        vae_config(&vae, model); vae_load(&vae, model);
        int64_t nf = 0; float *fin = ref_tensor(&R, "final_latent", &nf), *dec = ref_tensor(&R, "decoded", NULL);
        if (fin && dec) {
            float *got = fmalloc((size_t)3 * (8 * gh) * (8 * gw));
            double t2 = now_s();
            vae_decode(&vae, fin, gh, gw, NULL, got);
            fprintf(stderr, "[oracle] vae decode %.1f s\n", now_s() - t2);
            double r = cmp("vae decode", got, dec, (size_t)3 * (8 * gh) * (8 * gw));
            if (!(r < 1e-3)) { fprintf(stderr, "[oracle] vae decode: rel %.3e is too far\n", r); rc = 1; }
            free(got);
        } else fprintf(stderr, "[oracle] the dump has no final_latent/decoded: the VAE is not checked\n");
        free(fin); free(dec); vae_free(&vae);
    }
    caption_free(&cap); caption_free(&capr); zgrid_free(&grid);

    free(emb); free(rid); free(ids); free(ids_ref); free(prompt); free(ptext);
    fprintf(stderr, "[oracle] %s\n", rc ? "FAILED" : "all stages within tolerance");
    return rc;
}

/* Once, at the end: how many products the GPU actually took, so a run that asked for Vulkan and quietly stayed on
 * the CPU says so. */
static void zi_vk_report(void){
#ifdef COLI_VULKAN
    if (g_vk_ready) fprintf(stderr, "[VK] zimage: %llu matmuls on the GPU\n", coli_vk_matmul_calls());
    zic_report();
#endif
}

static void usage(void){
    fprintf(stderr,
        "usage: zimage --model DIR --prompt TEXT [--width 768] [--height 768] [--steps 8] [--seed N] --out FILE.png\n"
        "       zimage --model DIR --serve        line protocol: READY, then GEN {json} in, PROGRESS/IMAGE/ERROR out\n"
        "       zimage --model DIR --ref REFDIR   compare every stage with a reference dump\n"
        "env:   COLI_IMG_BITS=8|16|32 weight storage (default 8: int8 rows)\n"
        "       COLI_IMG_ACT8=0  f32 activations in the DiT (default where VNNI exists: int8)\n"
        "       COLI_IMG_TE=resident|stage  keep the text encoder loaded between prompts (serve default: resident)\n"
        "       COLI_IMG_LOAD_THREADS=n  layers loaded at once (default 4)\n");
}

int main(int argc, char **argv){
    coli_serve_binary_mode();
    const char *model = NULL, *ref = NULL, *prompt = NULL, *out = NULL;
    int width = 768, height = 768, steps = 8, serve = 0;
    uint64_t seed = 42; int seed_set = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--model") && v) { model = v; i++; }
        else if (!strcmp(a, "--ref") && v) { ref = v; i++; }
        else if (!strcmp(a, "--prompt") && v) { prompt = v; i++; }
        else if (!strcmp(a, "--out") && v) { out = v; i++; }
        else if (!strcmp(a, "--width") && v) { width = atoi(v); i++; }
        else if (!strcmp(a, "--height") && v) { height = atoi(v); i++; }
        else if (!strcmp(a, "--steps") && v) { steps = atoi(v); i++; }
        else if (!strcmp(a, "--seed") && v) { seed = strtoull(v, NULL, 10); seed_set = 1; i++; }
        else if (!strcmp(a, "--serve")) serve = 1;
        else if (!strcmp(a, "--threads") && v) {
#ifdef _OPENMP
            omp_set_num_threads(atoi(v));
#endif
            i++;
        }
        else { usage(); return 2; }
    }
    if (!model || (!serve && !ref && (!prompt || !out))) { usage(); return 2; }
    const char *b = getenv("COLI_IMG_BITS");
    if (b && *b) { g_bits = atoi(b); if (g_bits != 8 && g_bits != 16 && g_bits != 32) { usage(); return 2; } }
    const char *a8 = getenv("COLI_IMG_ACT8");
    g_act8 = !(a8 && *a8 == '0');
#ifndef QI_HAVE_VNNI
    g_act8 = 0;
#endif
    g_prof = getenv("ZIMAGE_PROF") != NULL;
    if (ref) { int rc = run_oracle(model, ref); zi_vk_report(); return rc; }
    if (!seed_set && !serve) seed = (uint64_t)time(NULL);
    static Engine e;
    engine_init(&e, model);
    if (serve) {
        const char *te = getenv("COLI_IMG_TE");
        e.te_resident = !(te && !strcmp(te, "stage"));
        te_load(&e.te, model);
        dit_load(&e.dit, model);
        int rc = serve_loop(&e, width, height, steps);
        zi_vk_report();
        return rc;
    }
    char msg[512];
    if (check_size(width, height, steps, msg, sizeof msg)) { fprintf(stderr, "%s\n", msg); return 2; }
    uint8_t *rgba = xmalloc((size_t)width * height * 4);
    double tm[3];
    if (engine_generate(&e, prompt, width, height, steps, seed, rgba, tm, NULL, msg, sizeof msg)) {
        fprintf(stderr, "[zimage] %s\n", msg); return 1;
    }
    if (write_png(out, rgba, width, height)) { fprintf(stderr, "[zimage] cannot write %s\n", out); return 1; }
    fprintf(stderr, "[zimage] %s  %dx%d  %d steps  seed %llu  encode %.1f s  denoise %.1f s  decode %.1f s\n",
            out, width, height, steps, (unsigned long long)seed, tm[0], tm[1], tm[2]);
    zi_vk_report();
    return 0;
}
