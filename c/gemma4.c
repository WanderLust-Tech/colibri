/* gemma4.c -- Gemma 4 mixture-of-experts (google/gemma-4-26B-A4B-it) in plain C.
 *
 * What the model is (text only; transformers' modeling_gemma4.py is the specification):
 *   - 30 layers, 25 sliding-window attention (window 1024, 8 KV heads of 256) and 5 full
 *     attention (2 KV heads of 512, keys equal values, rope on a quarter of the head),
 *     every 6th layer full. Attention is NOT scaled by 1/sqrt(d): queries and keys are
 *     RMS-normalised per head instead; values are normalised without a weight.
 *   - after attention and again after the feed-forward, a "sandwich" of RMS norms.
 *   - the feed-forward is a dense MLP (GELU) and, in parallel, 128 experts with the top 8 chosen
 *     by a router that has its own norm, scale and per-expert scale; both branches are
 *     normalised, summed and normalised again.
 *   - a scalar on every layer's output, embeddings scaled by sqrt(hidden), tied output head,
 *     logits soft-capped at 30.
 *
 * What it does with the hardware:
 *   - dense matrices: per-row int8, resident (about 2.4 GB with the embedding table).
 *   - experts: int4 group-of-64 records streamed from the container with pread into a
 *     per-layer LRU cache (--cap experts per layer; 128 keeps them all, about 13 GB). The
 *     arithmetic is expert_ffn.h's, the kernel Qwen3.6 and OLMoE already use.
 *   - KV: the sliding layers keep a ring of window + chunk positions, the full layers every
 *     position (2 heads only), so a long context costs about 41 KB a token, a fifth of what
 *     Qwen3-Coder needs.
 *
 * Container: c/tools/convert_gemma4.py. Serving: the mux protocol of docs/serve_protocol.md,
 * one KV slot per conversation (KV_SLOTS), a request's prompt prefilled in chunks so a CANCEL
 * is heard, the next decode steps of all active requests taken in turn.
 *
 * Environment: SNAP=<container>  SERVE=1  KV_SLOTS=n  CTX=<tokens>  COLI_KV_PREFIX=0
 *   G4_TEST=1 with G4_IDS=1,2,3 prints the logits of a prompt as JSON (tests/test_gemma4.py).
 *
 *   SNAP=dir ./gemma4 <experts cached per layer>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#include "st.h"
#include "omp_tune.h"
#include "kv_prefix.h"
#include "serve_codec.h"
#include "serve_budget.h"
#define XF_ACT_GELU_TANH 1
#include "expert_ffn.h"
#include "gemma4_tok.h"

#ifdef _WIN32
#include <windows.h>
#endif

#define CHUNK 256                       /* rows of a prompt processed together */
#define MAX_SLOTS 16

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0 * 1024.0 * 1024.0); }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0 * 1024.0); }
#endif

static void die(const char *fmt, ...) __attribute__((noreturn));
#include <stdarg.h>
static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    char msg[512]; vsnprintf(msg, sizeof msg, fmt, ap); va_end(ap);
    fprintf(stderr, "gemma4: %s\n", msg);
    printf("LOAD_FAIL kind=format %s\n", msg); fflush(stdout);
    exit(1);
}

/* ------------------------------------------------------------------------------ config */

typedef struct {
    int hidden, n_layers, n_heads, vocab, n_experts, topk, moe_inter, dense_inter, window;
    int hd_s, nkv_s, hd_g, nkv_g;       /* head size and KV heads: sliding layers, full layers */
    int k_eq_v;
    float eps, softcap, theta_s, theta_g, prf;
    uint8_t is_full[128];
    int stop[16], n_stop;
} Cfg;

static double jnum(jval *o, const char *key, double dflt) {
    jval *v = o ? json_get(o, key) : NULL;
    return v && v->t == J_NUM ? v->num : dflt;
}

static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof path, "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if (!f) die("cannot open %s", path);
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1); if (fread(buf, 1, (size_t)n, f) != (size_t)n) die("short read of %s", path);
    buf[n] = 0; fclose(f);
    char *arena = NULL; jval *root = json_parse(buf, &arena);
    jval *t = json_get(root, "text_config"); if (!t) t = root;
    memset(c, 0, sizeof *c);
    c->hidden = (int)jnum(t, "hidden_size", 0);
    c->n_layers = (int)jnum(t, "num_hidden_layers", 0);
    c->n_heads = (int)jnum(t, "num_attention_heads", 0);
    c->vocab = (int)jnum(t, "vocab_size", 0);
    c->n_experts = (int)jnum(t, "num_experts", 0);
    c->topk = (int)jnum(t, "top_k_experts", 0);
    c->moe_inter = (int)jnum(t, "moe_intermediate_size", 0);
    c->dense_inter = (int)jnum(t, "intermediate_size", 0);
    c->window = (int)jnum(t, "sliding_window", 0);
    c->hd_s = (int)jnum(t, "head_dim", 0);
    c->nkv_s = (int)jnum(t, "num_key_value_heads", 0);
    c->hd_g = (int)jnum(t, "global_head_dim", c->hd_s);
    c->nkv_g = (int)jnum(t, "num_global_key_value_heads", c->nkv_s);
    jval *kv = json_get(t, "attention_k_eq_v");
    c->k_eq_v = kv && kv->t == J_BOOL ? kv->boolean : 0;
    c->eps = (float)jnum(t, "rms_norm_eps", 1e-6);
    c->softcap = (float)jnum(t, "final_logit_softcapping", 0);
    jval *rp = json_get(t, "rope_parameters");
    jval *rs = rp ? json_get(rp, "sliding_attention") : NULL, *rg = rp ? json_get(rp, "full_attention") : NULL;
    c->theta_s = (float)jnum(rs, "rope_theta", 10000.0);
    c->theta_g = (float)jnum(rg, "rope_theta", 1000000.0);
    c->prf = (float)jnum(rg, "partial_rotary_factor", 1.0);
    jval *lt = json_get(t, "layer_types");
    if (!lt || lt->t != J_ARR || lt->len != c->n_layers) die("config has no layer_types for %d layers", c->n_layers);
    for (int i = 0; i < lt->len && i < 128; i++)
        c->is_full[i] = lt->kids[i]->str && !strcmp(lt->kids[i]->str, "full_attention");
    if (c->n_layers < 1 || c->n_layers > 128 || c->hidden < 1 || c->n_experts < 1 || c->topk < 1 || c->topk > c->n_experts)
        die("config: implausible geometry (%d layers, hidden %d, %d experts, top %d)", c->n_layers, c->hidden, c->n_experts, c->topk);
    if (c->hidden % 64 || c->moe_inter % 64) die("hidden %d and moe_intermediate_size %d must be multiples of 64", c->hidden, c->moe_inter);
    if (c->hd_s % 2 || c->hd_g % 2) die("head sizes must be even");
    if (jnum(t, "num_kv_shared_layers", 0) || jnum(t, "hidden_size_per_layer_input", 0)) die("kv sharing and per-layer embeddings are not supported");
    /* stop ids: generation_config.json lists them (eos, end of turn, start of a tool response) */
    snprintf(path, sizeof path, "%s/generation_config.json", snap);
    jval *eos = NULL; char *garena = NULL; jval *groot = NULL;
    FILE *g = fopen(path, "rb");
    if (g) {
        fseek(g, 0, SEEK_END); long gn = ftell(g); fseek(g, 0, SEEK_SET);
        char *gb = malloc((size_t)gn + 1); if (fread(gb, 1, (size_t)gn, g) != (size_t)gn) gn = 0; gb[gn] = 0; fclose(g);
        groot = json_parse(gb, &garena); eos = groot ? json_get(groot, "eos_token_id") : NULL; free(gb);
    }
    if (!eos) eos = json_get(t, "eos_token_id");
    if (!eos) eos = json_get(root, "eos_token_id");
    if (eos && eos->t == J_NUM) c->stop[c->n_stop++] = (int)eos->num;
    else if (eos && eos->t == J_ARR) for (int i = 0; i < eos->len && c->n_stop < 16; i++) c->stop[c->n_stop++] = (int)eos->kids[i]->num;
    if (!c->n_stop) c->stop[c->n_stop++] = 1;
    free(buf);
}

/* ------------------------------------------------------------------------------ weights */

typedef struct { int8_t *q; float *s; int O, I; } QMat;

typedef struct {
    int full;                                         /* a full-attention layer */
    int hd, nkv;                                      /* this layer's head size and KV heads */
    float *in_ln, *post_attn_ln, *pre_ff_ln, *post_ff_ln, *pre_ff_ln2, *post_ff_ln1, *post_ff_ln2;
    float *q_norm, *k_norm;
    QMat q, k, v, o, gate, up, down;                  /* v.q == NULL on a layer whose values are its keys */
    float *router_proj, *router_scale, *router_pes;
    float layer_scalar;
    const char *blob_name; char blob_buf[160];
} Layer;

typedef struct { int eid; uint64_t used; uint32_t group; uint8_t *buf; } ESlot;
typedef struct { ESlot *s; int n; int *slot_of; } LCache;

typedef struct { float **K, **V; kv_prefix kvp; } Seq;       /* one conversation's keys and values */

typedef struct {
    Cfg c; shards S;
    QMat embed; float *final_norm;
    Layer *L;
    float *inv_s, *inv_g;                             /* rope frequencies: sliding, full */
    int cap; size_t blob_bytes;
    LCache *cache; uint64_t clock; uint32_t group; uint64_t hits, miss; double disk_s;
    uint8_t *ehit;                                    /* [layers*experts] routed this request */
    int max_t, ring, chunk;                           /* chunk: rows per forward_chunk (<= CHUNK); ring: window + chunk */
    /* working memory for one chunk */
    float *x, *h, *qb, *kb, *vb, *ao, *g1, *u1, *m1, *m2, *e_in, *e_out, *rl, *kv_tmp;
    int *ridx; float *rval; const XfExpert **eptr; XfExpert *xe; void *scratch; size_t scratch_bytes;
    float **scores;                                   /* per thread */
    double t_attn, t_moe, t_head; long long forwards;
    int trace;
} Model;

static int g_slots_hint = 1;                           /* KV_SLOTS, for sizing the expert cache */

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory (%zu bytes)", n); return p; }

static void load_f32(Model *m, const char *name, float **out, int64_t want) {
    st_tensor *t = st_find(&m->S, name);
    if (!t) die("container lacks %s", name);
    if (t->numel != want) die("%s has %lld values, expected %lld", name, (long long)t->numel, (long long)want);
    *out = xmalloc((size_t)want * sizeof(float));
    st_read_f32(&m->S, name, *out, 0);
}

static void load_q(Model *m, const char *name, QMat *w, int O, int I) {
    char sname[256]; snprintf(sname, sizeof sname, "%s.scale", name);
    st_tensor *t = st_find(&m->S, name), *s = st_find(&m->S, sname);
    if (!t || !s) die("container lacks %s (or its scale)", name);
    if (t->numel != (int64_t)O * I || t->nbytes != (int64_t)O * I) die("%s: expected int8 [%d,%d]", name, O, I);
    if (s->numel != O) die("%s: expected %d scales", sname, O);
    w->O = O; w->I = I;
    w->q = xmalloc((size_t)O * I);
    st_read_raw(&m->S, name, w->q, 0);
    load_f32(m, sname, &w->s, O);
}

static void model_init(Model *m, const char *snap, int cap, int max_t) {
    memset(m, 0, sizeof *m);
    load_cfg(&m->c, snap);
    Cfg *c = &m->c;
    st_init(&m->S, snap);
    char nm[256];
    const char *P = "model.language_model.";
    snprintf(nm, sizeof nm, "%sembed_tokens.weight", P); load_q(m, nm, &m->embed, c->vocab, c->hidden);
    snprintf(nm, sizeof nm, "%snorm.weight", P); load_f32(m, nm, &m->final_norm, c->hidden);
    m->L = calloc((size_t)c->n_layers, sizeof(Layer));
    int H = c->hidden, E = c->n_experts;
    for (int l = 0; l < c->n_layers; l++) {
        Layer *y = &m->L[l];
        y->full = c->is_full[l];
        y->hd = y->full ? c->hd_g : c->hd_s; y->nkv = y->full ? c->nkv_g : c->nkv_s;
#define NAME(tail) (snprintf(nm, sizeof nm, "%slayers.%d.%s", P, l, tail), nm)
        load_f32(m, NAME("input_layernorm.weight"), &y->in_ln, H);
        load_f32(m, NAME("post_attention_layernorm.weight"), &y->post_attn_ln, H);
        load_f32(m, NAME("pre_feedforward_layernorm.weight"), &y->pre_ff_ln, H);
        load_f32(m, NAME("post_feedforward_layernorm.weight"), &y->post_ff_ln, H);
        load_f32(m, NAME("pre_feedforward_layernorm_2.weight"), &y->pre_ff_ln2, H);
        load_f32(m, NAME("post_feedforward_layernorm_1.weight"), &y->post_ff_ln1, H);
        load_f32(m, NAME("post_feedforward_layernorm_2.weight"), &y->post_ff_ln2, H);
        load_f32(m, NAME("self_attn.q_norm.weight"), &y->q_norm, y->hd);
        load_f32(m, NAME("self_attn.k_norm.weight"), &y->k_norm, y->hd);
        load_q(m, NAME("self_attn.q_proj.weight"), &y->q, c->n_heads * y->hd, H);
        load_q(m, NAME("self_attn.k_proj.weight"), &y->k, y->nkv * y->hd, H);
        if (y->full && c->k_eq_v) y->v.q = NULL;
        else load_q(m, NAME("self_attn.v_proj.weight"), &y->v, y->nkv * y->hd, H);
        load_q(m, NAME("self_attn.o_proj.weight"), &y->o, H, c->n_heads * y->hd);
        load_q(m, NAME("mlp.gate_proj.weight"), &y->gate, c->dense_inter, H);
        load_q(m, NAME("mlp.up_proj.weight"), &y->up, c->dense_inter, H);
        load_q(m, NAME("mlp.down_proj.weight"), &y->down, H, c->dense_inter);
        load_f32(m, NAME("router.proj.weight"), &y->router_proj, (int64_t)E * H);
        load_f32(m, NAME("router.scale"), &y->router_scale, H);
        load_f32(m, NAME("router.per_expert_scale"), &y->router_pes, E);
        float *ls; load_f32(m, NAME("layer_scalar"), &ls, 1); y->layer_scalar = ls[0]; free(ls);
        snprintf(y->blob_buf, sizeof y->blob_buf, "%slayers.%d.experts.blob", P, l);
        y->blob_name = y->blob_buf;
#undef NAME
    }
    m->blob_bytes = (size_t)2 * c->moe_inter * (H / 2) + (size_t)H * (c->moe_inter / 2)
                  + 4 * ((size_t)2 * c->moe_inter * (H / 64) + (size_t)H * (c->moe_inter / 64));
    st_tensor *b = st_find(&m->S, m->L[0].blob_name);
    if (!b || b->numel != (int64_t)E * (int64_t)m->blob_bytes) die("expert blobs have the wrong size: the container does not match its config");
    if (cap <= 0) {                                          /* nobody chose: what the machine can hold */
        double total = 0, avail = 0; compat_meminfo_gb(&total, &avail);
        if (avail <= 0) avail = total > 0 ? total * 0.5 : 8.0;
        double ring_gb = 0, tok_gb = 0;                      /* KV of one conversation: the rings, and a 8192-token reply's full layers */
        for (int l = 0; l < c->n_layers; l++) {
            double per_pos = 2.0 * m->L[l].nkv * m->L[l].hd * 4 / 1e9;
            if (m->L[l].full) tok_gb += per_pos * (max_t < 8192 ? max_t : 8192); else ring_gb += per_pos * (c->window + CHUNK);
        }
        double kv_gb = g_slots_hint * (ring_gb + tok_gb);
        double budget = avail * 0.75 - rss_gb() - kv_gb;
        double per_cap = (double)c->n_layers * (double)m->blob_bytes / 1e9;
        cap = budget > 0 ? (int)(budget / per_cap) : 0;
        if (cap < 8) cap = 8;
        fprintf(stderr, "[gemma4] %.1f GB free: %d of %d experts cached per layer (%.1f GB; KV for %d conversation%s %.1f GB)%s\n",
                avail, cap > E ? E : cap, E, (cap > E ? E : cap) * per_cap, g_slots_hint, g_slots_hint == 1 ? "" : "s", kv_gb,
                cap < 32 ? " -- little memory: reading experts from disk will be slow; --cap or --ctx overrides" : "");
    }
    if (cap < c->topk) cap = c->topk;
    if (cap > E) cap = E;
    m->cap = cap;
    m->cache = calloc((size_t)c->n_layers, sizeof(LCache));
    for (int l = 0; l < c->n_layers; l++) {
        m->cache[l].s = calloc((size_t)cap, sizeof(ESlot));
        m->cache[l].slot_of = xmalloc((size_t)E * sizeof(int));
        for (int e = 0; e < E; e++) m->cache[l].slot_of[e] = -1;
        for (int i = 0; i < cap; i++) m->cache[l].s[i].eid = -1;
    }
    m->ehit = calloc((size_t)c->n_layers * E, 1);
    /* rope: sliding layers rotate the whole head; full layers a quarter of it ("proportional":
     * the exponent still counts over the whole head, and the rest of the angles are zero) */
    m->inv_s = xmalloc((size_t)(c->hd_s / 2) * sizeof(float));
    for (int i = 0; i < c->hd_s / 2; i++) m->inv_s[i] = 1.f / powf(c->theta_s, (float)(2 * i) / (float)c->hd_s);
    m->inv_g = xmalloc((size_t)(c->hd_g / 2) * sizeof(float));
    int angles = (int)(c->prf * c->hd_g / 2);
    for (int i = 0; i < c->hd_g / 2; i++)
        m->inv_g[i] = i < angles ? 1.f / powf(c->theta_g, (float)(2 * i) / (float)c->hd_g) : 0.f;
    m->max_t = max_t;
    m->chunk = getenv("G4_CHUNK") ? atoi(getenv("G4_CHUNK")) : CHUNK;
    if (m->chunk < 1 || m->chunk > CHUNK) m->chunk = CHUNK;
    m->ring = c->window + m->chunk;
    int nh = c->n_heads, hdm = c->hd_g > c->hd_s ? c->hd_g : c->hd_s;
    int kvm = c->nkv_s * c->hd_s > c->nkv_g * c->hd_g ? c->nkv_s * c->hd_s : c->nkv_g * c->hd_g;
    int Fd = c->dense_inter;
    m->x = xmalloc((size_t)CHUNK * H * 4); m->h = xmalloc((size_t)CHUNK * H * 4);
    m->qb = xmalloc((size_t)CHUNK * nh * hdm * 4); m->ao = xmalloc((size_t)CHUNK * nh * hdm * 4);
    m->kb = xmalloc((size_t)CHUNK * kvm * 4); m->vb = xmalloc((size_t)CHUNK * kvm * 4);
    m->g1 = xmalloc((size_t)CHUNK * Fd * 4); m->u1 = xmalloc((size_t)CHUNK * Fd * 4);
    m->m1 = xmalloc((size_t)CHUNK * H * 4); m->m2 = xmalloc((size_t)CHUNK * H * 4);
    m->e_in = xmalloc((size_t)CHUNK * H * 4); m->e_out = xmalloc((size_t)CHUNK * H * 4);
    m->rl = xmalloc((size_t)CHUNK * E * 4);
    m->ridx = xmalloc((size_t)CHUNK * c->topk * sizeof(int)); m->rval = xmalloc((size_t)CHUNK * c->topk * 4);
    m->eptr = xmalloc((size_t)CHUNK * c->topk * sizeof(void *));
    m->xe = xmalloc((size_t)E * sizeof(XfExpert));
    m->scratch_bytes = xf_moe_scratch_bytes(CHUNK, c->topk, H, c->moe_inter);
    m->scratch = xmalloc(m->scratch_bytes);
    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif
    m->scores = xmalloc((size_t)T * sizeof(float *));
    for (int i = 0; i < T; i++) m->scores[i] = xmalloc(((size_t)max_t + 16) * 4);
    m->trace = getenv("G4_TRACE") != NULL;
}

static void seq_init(Model *m, Seq *q) {
    Cfg *c = &m->c;
    memset(q, 0, sizeof *q);
    q->K = calloc((size_t)c->n_layers, sizeof(float *)); q->V = calloc((size_t)c->n_layers, sizeof(float *));
    for (int l = 0; l < c->n_layers; l++) {
        Layer *y = &m->L[l];
        size_t positions = y->full ? (size_t)m->max_t : (size_t)m->ring;
        q->K[l] = xmalloc(positions * y->nkv * y->hd * 4);
        q->V[l] = xmalloc(positions * y->nkv * y->hd * 4);
    }
    kv_prefix_alloc(&q->kvp, m->max_t);
}

/* ------------------------------------------------------------------------------ kernels */

static inline float dot_f32(const float *a, const float *b, int n) {
    int i = 0; float r = 0.f;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
    }
    a0 = _mm256_add_ps(a0, a1);
    float t[8]; _mm256_storeu_ps(t, a0);
    r = ((t[0] + t[4]) + (t[1] + t[5])) + ((t[2] + t[6]) + (t[3] + t[7]));
#endif
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}

/* r[k] = sum_i w[i] * x[k*I + i] for k < rows (rows <= 4); w is int8. */
static inline void dot_q8(const int8_t *w, const float *x, int I, int rows, float *r) {
    int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256 a[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    if (rows == 1) {
        __m256 b[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (; i + 32 <= I; i += 32) {
            for (int u = 0; u < 4; u++) {
                __m256 wv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w + i + 8 * u))));
                b[u] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + i + 8 * u), b[u]);
            }
        }
        a[0] = _mm256_add_ps(_mm256_add_ps(b[0], b[1]), _mm256_add_ps(b[2], b[3]));
    }
    for (; i + 8 <= I; i += 8) {
        __m256 wv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w + i))));
        for (int k = 0; k < rows; k++) a[k] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + (size_t)k * I + i), a[k]);
    }
    for (int k = 0; k < rows; k++) {
        float t[8]; _mm256_storeu_ps(t, a[k]);
        r[k] = ((t[0] + t[4]) + (t[1] + t[5])) + ((t[2] + t[6]) + (t[3] + t[7]));
    }
#else
    for (int k = 0; k < rows; k++) r[k] = 0.f;
#endif
    for (; i < I; i++) for (int k = 0; k < rows; k++) r[k] += (float)w[i] * x[(size_t)k * I + i];
}

/* y[S][O] = x[S][I] . W^T, W per-row int8 */
static void mm(float *y, const float *x, int S, const QMat *W) {
    int O = W->O, I = W->I;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = W->q + (size_t)o * I; float sc = W->s[o], r[4];
        int s = 0;
        for (; s + 4 <= S; s += 4) { dot_q8(w, x + (size_t)s * I, I, 4, r); for (int k = 0; k < 4; k++) y[(size_t)(s + k) * O + o] = r[k] * sc; }
        for (; s < S; s++) { dot_q8(w, x + (size_t)s * I, I, 1, r); y[(size_t)s * O + o] = r[0] * sc; }
    }
}

static void rms_w(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.f; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = powf(ss / (float)n + eps, -0.5f);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}
static void rms_nw(float *out, const float *x, int n, float eps) {          /* no weight */
    float ss = 0.f; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = powf(ss / (float)n + eps, -0.5f);
    for (int i = 0; i < n; i++) out[i] = x[i] * r;
}
static inline float gelu_tanh(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))); }

/* rotate one head by the angles of row `s` (tables cs/sn [hd/2]) */
static void rope_head(float *x, int hd, const float *cs, const float *sn) {
    int h2 = hd / 2;
    for (int i = 0; i < h2; i++) { float a = x[i], b = x[i + h2]; x[i] = a * cs[i] - b * sn[i]; x[i + h2] = b * cs[i] + a * sn[i]; }
}

/* ------------------------------------------------------------------------------ experts */

/* Make the experts of `want` (n distinct ids) resident in layer l's cache; fills out[i] with the
 * slot. n <= cap, and slots of this group are never taken to make room for one another. */
static void experts_ready(Model *m, int l, const int *want, int n, ESlot **out) {
    Cfg *c = &m->c; LCache *lc = &m->cache[l];
    uint32_t grp = ++m->group;
    int miss_n = 0;
    int *todo = xmalloc((size_t)n * sizeof(int)), *todo_slot = xmalloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int sl = lc->slot_of[want[i]];
        if (sl >= 0) { ESlot *s = &lc->s[sl]; s->group = grp; s->used = ++m->clock; out[i] = s; m->hits++; }
        else { todo[miss_n++] = i; out[i] = NULL; }
    }
    for (int j = 0; j < miss_n; j++) {                        /* a slot for each miss: empty first, then least recently used */
        int best = -1; uint64_t bu = ~0ULL;
        for (int k = 0; k < m->cap; k++) {
            ESlot *s = &lc->s[k];
            if (s->group == grp) continue;
            uint64_t u = s->eid < 0 ? 0 : s->used + 1;
            if (u < bu) { bu = u; best = k; if (u == 0) break; }
        }
        if (best < 0) die("expert cache: no slot to give (cap %d)", m->cap);
        ESlot *s = &lc->s[best];
        if (s->eid >= 0) lc->slot_of[s->eid] = -1;
        s->eid = -2; s->group = grp;                          /* reserved */
        todo_slot[j] = best;
        out[todo[j]] = s;
    }
    if (miss_n) {
        double t0 = now_s();
#pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < miss_n; j++) {
            ESlot *s = &lc->s[todo_slot[j]];
            if (!s->buf) s->buf = malloc(m->blob_bytes);
            if (!s->buf) die("out of memory for an expert");
            st_read_slice_raw_cap(&m->S, m->L[l].blob_name, (int64_t)want[todo[j]] * (int64_t)m->blob_bytes,
                                  (int64_t)m->blob_bytes, s->buf, (int64_t)m->blob_bytes, 0);
        }
        m->disk_s += now_s() - t0;
        for (int j = 0; j < miss_n; j++) {
            ESlot *s = &lc->s[todo_slot[j]];
            s->eid = want[todo[j]]; s->used = ++m->clock;
            lc->slot_of[s->eid] = todo_slot[j];
            m->miss++;
        }
    }
    free(todo); free(todo_slot);
    (void)c;
}

static XfExpert expert_view(const Model *m, const uint8_t *buf) {
    int H = m->c.hidden, F = m->c.moe_inter;
    size_t gp = (size_t)F * (H / 2);
    const uint8_t *g4 = buf, *u4 = buf + gp, *d4 = buf + 2 * gp;
    const float *gs = (const float *)(buf + 2 * gp + (size_t)H * (F / 2));
    const float *us = gs + (size_t)F * (H / 64), *ds = us + (size_t)F * (H / 64);
    return (XfExpert){g4, u4, d4, gs, us, ds};
}

/* the routed half of the feed-forward for S rows: e_out = sum_k w_k * expert_k(e_in) */
static void moe_routed(Model *m, int l, const float *xres, int S) {
    Cfg *c = &m->c; Layer *y = &m->L[l];
    int H = c->hidden, E = c->n_experts, K = c->topk, F = c->moe_inter;
    float scale = 1.f / sqrtf((float)H);
    int *idx = m->ridx; float *val = m->rval;
    for (int s = 0; s < S; s++) {                              /* the router */
        float *rn = m->e_out + (size_t)s * H;                  /* scratch: e_out is rewritten below */
        rms_nw(rn, xres + (size_t)s * H, H, c->eps);
        for (int i = 0; i < H; i++) rn[i] = rn[i] * y->router_scale[i] * scale;
        float *lg = m->rl + (size_t)s * E, mx = -INFINITY;
        for (int e = 0; e < E; e++) { lg[e] = dot_f32(y->router_proj + (size_t)e * H, rn, H); if (lg[e] > mx) mx = lg[e]; }
        float sum = 0.f; for (int e = 0; e < E; e++) { lg[e] = expf(lg[e] - mx); sum += lg[e]; }
        for (int e = 0; e < E; e++) lg[e] /= sum;
        float tot = 0.f;
        for (int k = 0; k < K; k++) {
            int b = -1; float bv = -1.f;
            for (int e = 0; e < E; e++) { int taken = 0; for (int j = 0; j < k; j++) if (idx[s * K + j] == e) taken = 1; if (!taken && lg[e] > bv) { bv = lg[e]; b = e; } }
            idx[s * K + k] = b; val[s * K + k] = bv; tot += bv;
        }
        for (int k = 0; k < K; k++) { val[s * K + k] = val[s * K + k] / tot * y->router_pes[idx[s * K + k]]; m->ehit[(size_t)l * E + idx[s * K + k]] = 1; }
        rms_w(m->e_in + (size_t)s * H, xres + (size_t)s * H, y->pre_ff_ln2, H, c->eps);
    }
    memset(m->e_out, 0, (size_t)S * H * sizeof(float));
    /* the distinct experts, taken cap at a time */
    int *uniq = xmalloc((size_t)S * K * sizeof(int)); int nu = 0;
    uint8_t *seen = calloc((size_t)E, 1);
    for (int i = 0; i < S * K; i++) if (!seen[idx[i]]) { seen[idx[i]] = 1; uniq[nu++] = idx[i]; }
    free(seen);
    ESlot **slots = xmalloc((size_t)m->cap * sizeof(ESlot *));
    int *pos_of = xmalloc((size_t)E * sizeof(int));
    for (int g0 = 0; g0 < nu; g0 += m->cap) {
        int n = nu - g0 < m->cap ? nu - g0 : m->cap;
        experts_ready(m, l, uniq + g0, n, slots);
        for (int e = 0; e < E; e++) pos_of[e] = -1;
        for (int i = 0; i < n; i++) { m->xe[i] = expert_view(m, slots[i]->buf); pos_of[uniq[g0 + i]] = i; }
        for (int i = 0; i < S * K; i++) m->eptr[i] = pos_of[idx[i]] >= 0 ? &m->xe[pos_of[idx[i]]] : NULL;
        xf_moe_add(m->e_out, m->e_in, S, K, H, F, idx, val, m->eptr, 0, m->scratch);
    }
    free(uniq); free(slots); free(pos_of);
}

/* ------------------------------------------------------------------------------ the forward pass */

static void trace(Model *m, const char *what, int l, const float *x, int n) {
    if (!m->trace) return;
    double s = 0, a = 0; for (int i = 0; i < n; i++) { s += x[i]; a += fabs(x[i]); }
    fprintf(stderr, "[G4] %-10s L%-2d sum % .6f abs % .6f  %.5f %.5f %.5f\n", what, l, s, a, x[0], x[1], x[2]);
}

/* S rows of one conversation at positions pos0..pos0+S-1, S <= CHUNK. Leaves the residual stream in m->x. */
static void forward_chunk(Model *m, Seq *q, const int *ids, int S, int pos0) {
    Cfg *c = &m->c;
    int H = c->hidden, nh = c->n_heads;
    double t0;
    float emb_scale = sqrtf((float)H);
    for (int s = 0; s < S; s++) {                              /* embedding */
        int id = ids[s]; if (id < 0 || id >= c->vocab) id = 0;
        const int8_t *w = m->embed.q + (size_t)id * H; float sc = m->embed.s[id] * emb_scale;
        for (int i = 0; i < H; i++) m->x[(size_t)s * H + i] = (float)w[i] * sc;
    }
    trace(m, "embed", -1, m->x + (size_t)(S - 1) * H, H);
    /* rope tables for the chunk: sliding and full */
    int hs2 = c->hd_s / 2, hg2 = c->hd_g / 2;
    float *cs_s = xmalloc((size_t)S * hs2 * 4), *sn_s = xmalloc((size_t)S * hs2 * 4);
    float *cs_g = xmalloc((size_t)S * hg2 * 4), *sn_g = xmalloc((size_t)S * hg2 * 4);
    for (int s = 0; s < S; s++) {
        float p = (float)(pos0 + s);
        for (int i = 0; i < hs2; i++) { float a = p * m->inv_s[i]; cs_s[(size_t)s * hs2 + i] = cosf(a); sn_s[(size_t)s * hs2 + i] = sinf(a); }
        for (int i = 0; i < hg2; i++) { float a = p * m->inv_g[i]; cs_g[(size_t)s * hg2 + i] = cosf(a); sn_g[(size_t)s * hg2 + i] = sinf(a); }
    }
    for (int l = 0; l < c->n_layers; l++) {
        Layer *y = &m->L[l];
        int hd = y->hd, nkv = y->nkv, grp = nh / nkv;
        const float *cs = y->full ? cs_g : cs_s, *sn = y->full ? sn_g : sn_s;
        int h2 = hd / 2;
        /* ---- attention ---- */
        t0 = now_s();
        for (int s = 0; s < S; s++) rms_w(m->h + (size_t)s * H, m->x + (size_t)s * H, y->in_ln, H, c->eps);
        mm(m->qb, m->h, S, &y->q);
        mm(m->kb, m->h, S, &y->k);
        if (y->v.q) mm(m->vb, m->h, S, &y->v);
        else memcpy(m->vb, m->kb, (size_t)S * nkv * hd * sizeof(float));      /* values are the raw keys */
        for (int s = 0; s < S; s++) {
            for (int h = 0; h < nh; h++) {
                float *qh = m->qb + ((size_t)s * nh + h) * hd;
                rms_w(qh, qh, y->q_norm, hd, c->eps);
                rope_head(qh, hd, cs + (size_t)s * h2, sn + (size_t)s * h2);
            }
            for (int h = 0; h < nkv; h++) {
                float *kh = m->kb + ((size_t)s * nkv + h) * hd, *vh = m->vb + ((size_t)s * nkv + h) * hd;
                rms_w(kh, kh, y->k_norm, hd, c->eps);
                rope_head(kh, hd, cs + (size_t)s * h2, sn + (size_t)s * h2);
                rms_nw(vh, vh, hd, c->eps);
            }
            /* store this row's keys and values */
            int p = pos0 + s;
            size_t slot = y->full ? (size_t)p : (size_t)(p % m->ring);
            memcpy(q->K[l] + slot * nkv * hd, m->kb + (size_t)s * nkv * hd, (size_t)nkv * hd * sizeof(float));
            memcpy(q->V[l] + slot * nkv * hd, m->vb + (size_t)s * nkv * hd, (size_t)nkv * hd * sizeof(float));
        }
        /* scores, softmax, weighted values; the model does not scale scores */
#pragma omp parallel for schedule(dynamic, 4)
        for (int it = 0; it < S * nh; it++) {
            int s = it / nh, h = it % nh, p = pos0 + s, kvh = h / grp;
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float *sc = m->scores[tid];
            int lo = y->full ? 0 : (p - c->window + 1 > 0 ? p - c->window + 1 : 0);
            const float *qh = m->qb + ((size_t)s * nh + h) * hd;
            float mx = -INFINITY;
            for (int j = lo; j <= p; j++) {
                size_t slot = y->full ? (size_t)j : (size_t)(j % m->ring);
                float v = dot_f32(qh, q->K[l] + (slot * nkv + kvh) * hd, hd);
                sc[j - lo] = v; if (v > mx) mx = v;
            }
            float sum = 0.f;
            for (int j = lo; j <= p; j++) { sc[j - lo] = expf(sc[j - lo] - mx); sum += sc[j - lo]; }
            float inv = 1.f / sum;
            float *out = m->ao + ((size_t)s * nh + h) * hd;
            for (int d = 0; d < hd; d++) out[d] = 0.f;
            for (int j = lo; j <= p; j++) {
                size_t slot = y->full ? (size_t)j : (size_t)(j % m->ring);
                const float *vv = q->V[l] + (slot * nkv + kvh) * hd; float w = sc[j - lo] * inv;
                for (int d = 0; d < hd; d++) out[d] += w * vv[d];
            }
        }
        mm(m->m1, m->ao, S, &y->o);                              /* o_proj */
        for (int s = 0; s < S; s++) {
            rms_w(m->m1 + (size_t)s * H, m->m1 + (size_t)s * H, y->post_attn_ln, H, c->eps);
            for (int i = 0; i < H; i++) m->x[(size_t)s * H + i] += m->m1[(size_t)s * H + i];
        }
        m->t_attn += now_s() - t0;
        trace(m, "attn", l, m->x + (size_t)(S - 1) * H, H);
        /* ---- feed-forward: dense MLP and experts side by side, both off the same residual ---- */
        t0 = now_s();
        int Fd = c->dense_inter;
        for (int s = 0; s < S; s++) rms_w(m->h + (size_t)s * H, m->x + (size_t)s * H, y->pre_ff_ln, H, c->eps);
        mm(m->g1, m->h, S, &y->gate);
        mm(m->u1, m->h, S, &y->up);
        for (size_t i = 0; i < (size_t)S * Fd; i++) m->g1[i] = gelu_tanh(m->g1[i]) * m->u1[i];
        mm(m->m1, m->g1, S, &y->down);
        for (int s = 0; s < S; s++) rms_w(m->m1 + (size_t)s * H, m->m1 + (size_t)s * H, y->post_ff_ln1, H, c->eps);
        moe_routed(m, l, m->x, S);
        for (int s = 0; s < S; s++) {
            rms_w(m->m2 + (size_t)s * H, m->e_out + (size_t)s * H, y->post_ff_ln2, H, c->eps);
            float *a = m->m1 + (size_t)s * H, *b = m->m2 + (size_t)s * H;
            for (int i = 0; i < H; i++) a[i] += b[i];
            rms_w(a, a, y->post_ff_ln, H, c->eps);
            float *xr = m->x + (size_t)s * H;
            for (int i = 0; i < H; i++) xr[i] = (xr[i] + a[i]) * y->layer_scalar;
        }
        m->t_moe += now_s() - t0;
        trace(m, "layer", l, m->x + (size_t)(S - 1) * H, H);
    }
    free(cs_s); free(sn_s); free(cs_g); free(sn_g);
}

/* logits of one residual row: final norm, tied head, soft cap. out has c->vocab floats. */
static void head_row(Model *m, const float *xrow, float *out) {
    Cfg *c = &m->c; int H = c->hidden, V = c->vocab;
    double t0 = now_s();
    float *hn = xmalloc((size_t)H * 4);
    rms_w(hn, xrow, m->final_norm, H, c->eps);
    trace(m, "final", -1, hn, H);
#pragma omp parallel for schedule(static)
    for (int v = 0; v < V; v++) {
        float r[4]; dot_q8(m->embed.q + (size_t)v * H, hn, H, 1, r);
        float l = r[0] * m->embed.s[v];
        out[v] = c->softcap > 0 ? c->softcap * tanhf(l / c->softcap) : l;
    }
    free(hn);
    m->t_head += now_s() - t0;
}

/* Run `n` tokens of a conversation starting at position pos0 (chunked). Returns the logits after the
 * last token (V floats, malloc'd); with all != NULL, every row's logits too (all: n*V floats). */
static float *forward(Model *m, Seq *q, const int *ids, int n, int pos0, float *all) {
    Cfg *c = &m->c; int V = c->vocab;
    float *last = xmalloc((size_t)V * 4);
    for (int s0 = 0; s0 < n; s0 += m->chunk) {
        int S = n - s0 < m->chunk ? n - s0 : m->chunk;
        forward_chunk(m, q, ids + s0, S, pos0 + s0);
        kv_prefix_record(&q->kvp, ids + s0, pos0 + s0, S);
        if (all) for (int s = 0; s < S; s++) head_row(m, m->x + (size_t)s * c->hidden, all + (size_t)(s0 + s) * V);
        if (s0 + m->chunk >= n) head_row(m, m->x + (size_t)(S - 1) * c->hidden, last);
        m->forwards += S;
    }
    return last;
}

/* ------------------------------------------------------------------------------ sampling */

static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;
static double rndu(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (double)(g_rng >> 11) / 9007199254740992.0; }

static int argmax_v(const float *lo, int V) { int b = 0; for (int i = 1; i < V; i++) if (lo[i] > lo[b]) b = i; return b; }

typedef struct { float v; int i; } VI;
static int cmp_vi(const void *a, const void *b) { float x = ((const VI *)a)->v, y = ((const VI *)b)->v; return x < y ? 1 : x > y ? -1 : 0; }

static int pick_token(const float *lo, int V, float temp, float top_p, int top_k) {
    if (temp <= 0.f) return argmax_v(lo, V);
    int k = top_k > 0 && top_k < V ? top_k : V;
    VI *c = xmalloc((size_t)V * sizeof(VI));
    for (int i = 0; i < V; i++) { c[i].v = lo[i]; c[i].i = i; }
    if (k < V) {                                               /* partial selection of the k largest */
        float thr; { VI *t = xmalloc((size_t)V * sizeof(VI)); memcpy(t, c, (size_t)V * sizeof(VI));
            /* nth largest by repeated partition would be ideal; a heap of size k is simple and enough */
            VI *hp = xmalloc((size_t)k * sizeof(VI)); int hn = 0;
            for (int i = 0; i < V; i++) {
                if (hn < k) { hp[hn++] = t[i]; for (int j = hn - 1; j > 0; ) { int p = (j - 1) / 2; if (hp[p].v <= hp[j].v) break; VI x = hp[p]; hp[p] = hp[j]; hp[j] = x; j = p; } }
                else if (t[i].v > hp[0].v) { hp[0] = t[i]; for (int j = 0;;) { int l = 2 * j + 1, r = l + 1, mn = j; if (l < k && hp[l].v < hp[mn].v) mn = l; if (r < k && hp[r].v < hp[mn].v) mn = r; if (mn == j) break; VI x = hp[mn]; hp[mn] = hp[j]; hp[j] = x; j = mn; } }
            }
            thr = hp[0].v; free(hp); free(t); }
        int w = 0; for (int i = 0; i < V; i++) if (c[i].v >= thr) c[w++] = c[i];
        V = w;
    }
    qsort(c, (size_t)V, sizeof(VI), cmp_vi);
    float mx = c[0].v, sum = 0.f;
    for (int i = 0; i < V; i++) { c[i].v = expf((c[i].v - mx) / temp); sum += c[i].v; }
    float cum = 0.f; int keep = V;
    for (int i = 0; i < V; i++) { cum += c[i].v / sum; if (top_p > 0.f && top_p < 1.f && cum >= top_p) { keep = i + 1; break; } }
    float tot = 0.f; for (int i = 0; i < keep; i++) tot += c[i].v;
    double r = rndu() * tot; int pick = c[keep - 1].i;
    for (int i = 0; i < keep; i++) { r -= c[i].v; if (r <= 0) { pick = c[i].i; break; } }
    free(c);
    return pick;
}

static int is_stop(const Cfg *c, int t) { for (int i = 0; i < c->n_stop; i++) if (c->stop[i] == t) return 1; return 0; }

/* ------------------------------------------------------------------------------ test mode */

static int run_test(Model *m) {
    const char *e = getenv("G4_IDS");
    if (!e) die("G4_TEST needs G4_IDS=1,2,3");
    int *ids = xmalloc(strlen(e) * sizeof(int) + 16), n = 0;
    for (const char *p = e; *p;) { char *end; long v = strtol(p, &end, 10); if (end == p) break; ids[n++] = (int)v; p = *end == ',' ? end + 1 : end; }
    int greedy = getenv("G4_GREEDY") ? atoi(getenv("G4_GREEDY")) : 0;
    int V = m->c.vocab;
    Seq q; seq_init(m, &q);
    float *all = xmalloc((size_t)n * V * 4);
    float *last = forward(m, &q, ids, n, 0, all);
    printf("{\"top\":[");
    for (int s = 0; s < n; s++) printf(s ? ",%d" : "%d", argmax_v(all + (size_t)s * V, V));
    printf("],\"rows\":[");
    for (int s = 0; s < n; s++) { printf(s ? ",[" : "["); for (int i = 0; i < 16 && i < V; i++) printf(i ? ",%.9g" : "%.9g", all[(size_t)s * V + i]); printf("]"); }
    printf("],\"last\":[");
    for (int i = 0; i < V; i++) printf(i ? ",%.9g" : "%.9g", last[i]);
    printf("],\"greedy\":[");
    int *gids = xmalloc(((size_t)greedy + 1) * sizeof(int));
    for (int g = 0; g < greedy; g++) {
        int t = argmax_v(last, V); gids[g] = t; printf(g ? ",%d" : "%d", t);
        free(last); last = forward(m, &q, &t, 1, n + g, NULL);
    }
    printf("]");
    /* what those tokens spell, as hex bytes: what a served reply's DATA frames carry */
    static G4Tok T;
    char tp[2048]; snprintf(tp, sizeof tp, "%s/tokenizer.json", getenv("SNAP"));
    FILE *probe = fopen(tp, "rb");
    if (probe) {
        fclose(probe);
        if (g4_tok_load(&T, tp)) {
            printf(",\"greedy_hex\":\"");
            for (int g = 0; g < greedy; g++) {
                char piece[512]; int k = g4_tok_piece(&T, gids[g], piece, sizeof piece);
                for (int i = 0; i < k; i++) printf("%02x", (unsigned char)piece[i]);
            }
            printf("\"");
        }
    }
    printf("}\n");
    return 0;
}

/* ------------------------------------------------------------------------------ serving */

typedef struct {
    int active, phase;                  /* phase 0 = prompt still being read, 1 = decoding */
    char id[COLI_SERVE_ID_CAP];
    int *ids, np, done_np;              /* prompt tokens, how many are in the KV */
    int gen, max_tok, limited, cancel;
    float temp, top_p;
    float *lo;
    double t0; uint64_t h0, m0;
    double attn0, moe0, head0, disk0; long long fwd0;
    int reuse;
} Req;

static const ColiServeWireProfile g4_wire = {
    .max_header_bytes = 511, .max_payload_bytes = 1u << 25, .max_tokens = 1 << 20,
    .require_exact_lf = 1, .require_finite_sampling = 0,
};

static void serve_hits(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts, rows = c->n_layers;
    int nb = (rows * E + 7) / 8;
    uint8_t *bm = calloc((size_t)nb, 1); int bit = 0;
    for (int i = 0; i < rows; i++) for (int e = 0; e < E; e++, bit++)
        if (m->ehit[(size_t)i * E + e]) { bm[bit >> 3] |= (uint8_t)(1 << (bit & 7)); m->ehit[(size_t)i * E + e] = 0; }
    char *hex = malloc((size_t)nb * 2 + 1); int w = 0;
    for (int b = 0; b < nb; b++) { hex[w++] = "0123456789abcdef"[bm[b] >> 4]; hex[w++] = "0123456789abcdef"[bm[b] & 15]; }
    hex[w] = 0;
    printf("HITS %d %d %s\n", rows, E, hex); fflush(stdout);
    free(hex); free(bm);
}

static void serve_hwinfo(void) {
    double rt = 0, ra = 0; compat_meminfo_gb(&rt, &ra);
    int cores = 1;
#ifdef _OPENMP
    cores = omp_get_max_threads();
#endif
    printf("HWINFO %d %.1f %.1f 0 0.0 unknown|\n", cores, rt, ra); fflush(stdout);
}

static void serve_tiers_emap(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts, filled = 0;
    for (int l = 0; l < c->n_layers; l++) for (int i = 0; i < m->cap; i++) filled += m->cache[l].s[i].eid >= 0;
    printf("TIERS 0 %d %d 0.00 %.2f\n", filled, c->n_layers * E - filled, filled * (double)m->blob_bytes / 1e9);
    char *hex = malloc((size_t)c->n_layers * E * 2 + 1); int w = 0;
    for (int l = 0; l < c->n_layers; l++) for (int e = 0; e < E; e++) {
        int b = (m->cache[l].slot_of[e] >= 0 ? 1 : 0) << 6;
        hex[w++] = "0123456789abcdef"[b >> 4]; hex[w++] = "0123456789abcdef"[b & 15];
    }
    hex[w] = 0;
    printf("EMAP %d %d %s\n", c->n_layers, E, hex); fflush(stdout);
    free(hex);
}

static void req_finish(Model *m, Req *r) {
    double dt = now_s() - r->t0;
    double tot = (double)(m->hits - r->h0 + m->miss - r->m0);
    ColiServeDone done = { .completion_tokens = r->gen, .tokens_per_second = dt > 0 ? r->gen / dt : 0.0,
                           .cache_hit_percent = tot ? 100.0 * (double)(m->hits - r->h0) / tot : 100.0,
                           .rss_gb = rss_gb(), .prompt_tokens = r->np, .length_limited = r->cancel ? 0 : r->limited };
    coli_serve_write_done(stdout, r->id, &done);
    double disk = m->disk_s - r->disk0, mm_s = (m->t_moe - r->moe0) - disk; if (mm_s < 0) mm_s = 0;
    printf("PROF %.6f %d %d %.6f 0.0 %.6f %.6f %.6f %lld\n", dt, r->np, r->gen, disk, mm_s,
           m->t_attn - r->attn0, m->t_head - r->head0, m->forwards - r->fwd0);
    fflush(stdout);
    serve_hits(m);
    free(r->ids); free(r->lo); r->ids = NULL; r->lo = NULL; r->active = 0;
}

static int kv_prefix_off(void) { const char *e = getenv("COLI_KV_PREFIX"); return e && *e == '0'; }

/* Begin a request: tokenize, budget, find what the slot already holds. 0 when it ended here. */
static int req_start(Model *m, G4Tok *T, Req *r, Seq *q, const char *payload, int plen, int ctx_cap) {
    int cap = plen + 16;
    r->ids = xmalloc((size_t)cap * sizeof(int));
    r->np = g4_tok_encode(T, payload, plen, r->ids, cap);
    if (r->np <= 0) { coli_serve_write_error(stdout, r->id, "empty prompt"); free(r->ids); r->ids = NULL; return 0; }
    int budget = coli_serve_budget(r->np, r->max_tok, ctx_cap, 0);
    if (budget < 0) {
        char msg[160]; snprintf(msg, sizeof msg, "CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d", r->np, r->max_tok, ctx_cap);
        coli_serve_write_error(stdout, r->id, msg); free(r->ids); r->ids = NULL; return 0;
    }
    if (budget < r->max_tok) r->max_tok = budget;
    int reuse = kv_prefix_off() ? 0 : kv_prefix_reuse(&q->kvp, r->ids, r->np);
    if (!reuse) kv_prefix_clear(&q->kvp);
    if (getenv("COLI_PREFIX_LOG"))
        fprintf(stderr, reuse ? "[PREFIX] reusing %d of %d prompt tokens\n" : "[PREFIX] no reuse of %d held, prompt %d\n", reuse ? reuse : q->kvp.len, r->np);
    r->done_np = reuse; r->reuse = reuse;
    r->t0 = now_s(); r->h0 = m->hits; r->m0 = m->miss;
    r->attn0 = m->t_attn; r->moe0 = m->t_moe; r->head0 = m->t_head; r->disk0 = m->disk_s; r->fwd0 = m->forwards;
    r->phase = 0; r->active = 1; r->limited = 1; r->gen = 0; r->cancel = 0; r->lo = NULL;
    return 1;
}

static void serve_loop(Model *m, G4Tok *T, int ctx_cap, int n_slots) {
    Cfg *c = &m->c; int V = c->vocab;
    Seq *seq = calloc((size_t)n_slots, sizeof(Seq));
    Req *rq = calloc((size_t)n_slots, sizeof(Req));
    for (int i = 0; i < n_slots; i++) seq_init(m, &seq[i]);
    coli_serve_stdio_init();
    coli_serve_write_ready_caps(stdout, rss_gb(), "vision=0");
    serve_hwinfo(); serve_tiers_emap(m);
    fprintf(stderr, "[gemma4] serving %d conversation%s, context %d, %d experts cached per layer\n", n_slots, n_slots == 1 ? "" : "s", ctx_cap, m->cap);
    int input_eof = 0, rr = 0;
    float top_k_env = getenv("G4_TOP_K") ? (float)atoi(getenv("G4_TOP_K")) : 64;
    for (;;) {
        int active = 0; for (int i = 0; i < n_slots; i++) active += rq[i].active;
        if (!input_eof && (!active || coli_stdin_readable())) {
            ColiServeCommand cmd;
            ColiServeReadResult res = coli_serve_read_command(stdin, &g4_wire, &cmd);
            if (res == COLI_SERVE_READ_EOF || res == COLI_SERVE_READ_BAD_FRAME) input_eof = 1;
            else if (res == COLI_SERVE_READ_NOMEM) { coli_serve_write_error(stdout, cmd.id, "out of memory"); input_eof = 1; }
            else if (res == COLI_SERVE_READ_BAD_REQUEST) {
                if (cmd.kind == COLI_SERVE_COMMAND_SUBMIT) coli_serve_write_error(stdout, cmd.id, "bad submit header");
                coli_serve_command_dispose(&cmd);
            } else if (res == COLI_SERVE_READ_OK) {
                if (cmd.kind == COLI_SERVE_COMMAND_CANCEL) {
                    for (int i = 0; i < n_slots; i++) if (rq[i].active && !strcmp(rq[i].id, cmd.id)) rq[i].cancel = 1;
                } else if (cmd.kind == COLI_SERVE_COMMAND_SUBMIT) {
                    if (cmd.slot < 0 || cmd.slot >= n_slots) coli_serve_write_error(stdout, cmd.id, "invalid cache slot");
                    else if (rq[cmd.slot].active) coli_serve_write_error(stdout, cmd.id, "SLOT_BUSY");
                    else {
                        Req *r = &rq[cmd.slot];
                        memset(r, 0, sizeof *r);
                        snprintf(r->id, sizeof r->id, "%s", cmd.id);
                        r->max_tok = cmd.max_tokens; r->temp = cmd.temperature; r->top_p = cmd.top_p;
                        char *payload = (char *)coli_serve_command_take_payload(&cmd);
                        if (req_start(m, T, r, &seq[cmd.slot], payload, (int)cmd.payload_bytes, ctx_cap))
                            coli_serve_write_accept(stdout, r->id, r->np);
                        free(payload);
                    }
                } else if (cmd.kind != COLI_SERVE_COMMAND_STOP) {
                    coli_serve_write_error(stdout, cmd.id, "NOT_SUPPORTED");
                }
                coli_serve_command_dispose(&cmd);
            }
        }
        active = 0; for (int i = 0; i < n_slots; i++) active += rq[i].active;
        if (!active) { if (input_eof) break; continue; }
        /* one unit of work for the next active request in turn: a chunk of prompt, or one token */
        int i = -1;
        for (int k = 0; k < n_slots; k++) { int j = (rr + k) % n_slots; if (rq[j].active) { i = j; break; } }
        rr = i + 1;
        Req *r = &rq[i]; Seq *q = &seq[i];
        if (r->cancel) { r->limited = 0; req_finish(m, r); serve_tiers_emap(m); continue; }
        if (r->phase == 0) {
            int n = r->np - r->done_np; if (n > m->chunk) n = m->chunk;
            float *lo = forward(m, q, r->ids + r->done_np, n, r->done_np, NULL);
            r->done_np += n;
            if (r->done_np >= r->np) { r->lo = lo; r->phase = 1; } else free(lo);
            continue;
        }
        /* decode: pick the next token from the logits, send it, then feed it */
        int t = pick_token(r->lo, V, r->temp, r->top_p, (int)top_k_env);
        if (is_stop(c, t)) { r->limited = 0; req_finish(m, r); serve_tiers_emap(m); continue; }
        char piece[512]; int nb = g4_tok_piece(T, t, piece, sizeof piece);
        coli_serve_write_data(stdout, r->id, piece, (size_t)nb);
        r->gen++;
        if (r->gen >= r->max_tok || r->np + r->gen >= ctx_cap) { req_finish(m, r); serve_tiers_emap(m); continue; }
        free(r->lo);
        r->lo = forward(m, q, &t, 1, r->np + r->gen - 1, NULL);
    }
    free(seq); free(rq);
}

/* ------------------------------------------------------------------------------ main */

int main(int argc, char **argv) {
    coli_omp_tune_threads("gemma4");
    const char *snap = getenv("SNAP");
    if (!snap) { coli_print_launcher_help("Gemma 4", "SNAP=<model directory> SERVE=1 ./gemma4 <experts cached per layer>"); return 1; }
    int cap = argc > 1 ? atoi(argv[1]) : (getenv("G4_CAP") ? atoi(getenv("G4_CAP")) : 0);   /* 0: size it from free memory */
    int test = getenv("G4_TEST") && getenv("G4_TEST")[0] == '1';
    int n_slots = 1;
    if (getenv("KV_SLOTS") && *getenv("KV_SLOTS")) {
        char *end; long v = strtol(getenv("KV_SLOTS"), &end, 10);
        if (*end || v < 1 || v > MAX_SLOTS) { fprintf(stderr, "KV_SLOTS must be between 1 and %d\n", MAX_SLOTS); return 2; }
        n_slots = (int)v;
    }
    g_slots_hint = n_slots;
    int ctx = getenv("CTX") ? atoi(getenv("CTX")) : 8192;
    if (ctx < 1 || ctx > 262144) { fprintf(stderr, "CTX must be 1..262144 (got %d)\n", ctx); return 2; }
    if (getenv("SEED")) g_rng ^= (uint64_t)atoll(getenv("SEED")) * 0xD1B54A32D192ED03ULL; else g_rng ^= (uint64_t)time(NULL) * 0xD1B54A32D192ED03ULL;
    static Model m;
    double t0 = now_s();
    model_init(&m, snap, cap, ctx);
    fprintf(stderr, "[gemma4] weights loaded in %.1fs, RSS %.2f GB\n", now_s() - t0, rss_gb());
    if (test) return run_test(&m);
    static G4Tok T;
    char tp[2048]; snprintf(tp, sizeof tp, "%s/tokenizer.json", snap);
    if (!g4_tok_load(&T, tp)) die("cannot load %s", tp);
    serve_loop(&m, &T, ctx, n_slots);
    return 0;
}
