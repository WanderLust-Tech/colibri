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
#include "kv_image_key.h"
#define XF_ACT_GELU_TANH 1
#include "expert_ffn.h"
#include "gemma4_tok.h"
#include "constrain.h"
#ifdef COLI_VULKAN
#include "backend_vulkan.h"
#endif

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
    int moe;                            /* enable_moe_block: experts beside the dense MLP (the 26B-A4B); the others are dense */
    int ple_dim, ple_vocab;             /* hidden_size_per_layer_input (0: none) and the vocabulary of that table */
    int kv_shared, double_wide, pad_token;   /* layers at the end that read an earlier layer's keys and values; their MLP is twice as wide (E2B) */
    int image_token;                    /* the placeholder id of one soft token (-1: the config names none) */
    int audio_token;                    /* the same for one audio soft token */
    struct { int present, hidden, layers, heads, hd, chunk, left, right, kconv, out_dims, ch0, ch1; float eps, cap, res_w, clip; } ac;
    struct { int present, hidden, layers, heads, hd, inter, patch, pool, npos, standardize; float eps, theta; } vc;
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
    jval *mb = json_get(t, "enable_moe_block");
    c->moe = mb && mb->t == J_BOOL ? mb->boolean : 0;
    c->n_experts = (int)jnum(t, "num_experts", 0);
    c->topk = (int)jnum(t, "top_k_experts", 0);
    c->moe_inter = (int)jnum(t, "moe_intermediate_size", 0);
    c->ple_dim = (int)jnum(t, "hidden_size_per_layer_input", 0);
    c->ple_vocab = (int)jnum(t, "vocab_size_per_layer_input", c->vocab);
    c->kv_shared = (int)jnum(t, "num_kv_shared_layers", 0);
    jval *dw = json_get(t, "use_double_wide_mlp");
    c->double_wide = dw && dw->t == J_BOOL ? dw->boolean : 0;
    c->pad_token = (int)jnum(t, "pad_token_id", 0);
    c->dense_inter = (int)jnum(t, "intermediate_size", 0);
    c->window = (int)jnum(t, "sliding_window", 0);
    c->hd_s = (int)jnum(t, "head_dim", 0);
    c->nkv_s = (int)jnum(t, "num_key_value_heads", 0);
    c->hd_g = (int)jnum(t, "global_head_dim", c->hd_s);
    c->nkv_g = (int)jnum(t, "num_global_key_value_heads", c->nkv_s);
    jval *kv = json_get(t, "attention_k_eq_v");
    c->k_eq_v = kv && kv->t == J_BOOL ? kv->boolean : 0;
    if (!c->k_eq_v) c->nkv_g = c->nkv_s;                   /* the global KV head count applies only to the keys-are-values layers */
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
    if (c->n_layers < 1 || c->n_layers > 128 || c->hidden < 1 || (c->moe && (c->n_experts < 1 || c->topk < 1 || c->topk > c->n_experts)))
        die("config: implausible geometry (%d layers, hidden %d, %d experts, top %d)", c->n_layers, c->hidden, c->n_experts, c->topk);
    if (c->moe && (c->hidden % 64 || c->moe_inter % 64)) die("hidden %d and moe_intermediate_size %d must be multiples of 64", c->hidden, c->moe_inter);
    if (c->kv_shared < 0 || c->kv_shared >= c->n_layers) die("config: num_kv_shared_layers %d of %d layers", c->kv_shared, c->n_layers);
    if (c->ple_dim < 0 || (c->ple_dim && c->ple_vocab < 1)) die("config: implausible per-layer embeddings");
    if (c->hd_s % 2 || c->hd_g % 2) die("head sizes must be even");
    c->image_token = (int)jnum(root, "image_token_id", -1);
    c->audio_token = (int)jnum(root, "audio_token_id", -1);
    jval *acfg = json_get(root, "audio_config");
    if (acfg && acfg->t == J_OBJ) {
        c->ac.present = 1;
        c->ac.hidden = (int)jnum(acfg, "hidden_size", 0); c->ac.layers = (int)jnum(acfg, "num_hidden_layers", 0);
        c->ac.heads = (int)jnum(acfg, "num_attention_heads", 0);
        c->ac.hd = c->ac.heads ? c->ac.hidden / c->ac.heads : 0;
        c->ac.chunk = (int)jnum(acfg, "attention_chunk_size", 12); c->ac.left = (int)jnum(acfg, "attention_context_left", 13);
        c->ac.right = (int)jnum(acfg, "attention_context_right", 0); c->ac.kconv = (int)jnum(acfg, "conv_kernel_size", 5);
        c->ac.out_dims = (int)jnum(acfg, "output_proj_dims", c->ac.hidden);
        c->ac.eps = (float)jnum(acfg, "rms_norm_eps", 1e-6); c->ac.cap = (float)jnum(acfg, "attention_logit_cap", 50.0);
        c->ac.res_w = (float)jnum(acfg, "residual_weight", 0.5); c->ac.clip = (float)jnum(acfg, "gradient_clipping", 1e10);
        jval *ch = json_get(acfg, "subsampling_conv_channels");
        c->ac.ch0 = ch && ch->t == J_ARR && ch->len == 2 ? (int)ch->kids[0]->num : 128;
        c->ac.ch1 = ch && ch->t == J_ARR && ch->len == 2 ? (int)ch->kids[1]->num : 32;
        jval *act = json_get(acfg, "hidden_act");
        if (act && act->str && strcmp(act->str, "silu")) die("an audio tower with the activation '%s' is not supported (silu is)", act->str);
        if (c->ac.hidden < 1 || c->ac.layers < 1 || c->ac.heads < 1 || c->ac.hidden % c->ac.heads || c->ac.kconv < 1 ||
            c->ac.ch0 < 4 || c->ac.ch0 % 4 || c->ac.ch1 < 1 || c->ac.out_dims < 1 || c->ac.chunk < 1)
            die("config: implausible audio_config");
        /* the relative-position bias of the chunked attention is laid out for a past as long as a chunk and no future */
        if (c->ac.left - 1 != c->ac.chunk || c->ac.right != 0)
            die("an audio tower whose attention context is not (chunk %d, past %d, future 0) is not supported", c->ac.chunk, c->ac.left - 1);
    }
    jval *vcfg = json_get(root, "vision_config");
    if (vcfg && vcfg->t == J_OBJ) {
        c->vc.present = 1;
        c->vc.hidden = (int)jnum(vcfg, "hidden_size", 0); c->vc.layers = (int)jnum(vcfg, "num_hidden_layers", 0);
        c->vc.heads = (int)jnum(vcfg, "num_attention_heads", 0); c->vc.inter = (int)jnum(vcfg, "intermediate_size", 0);
        c->vc.hd = (int)jnum(vcfg, "head_dim", c->vc.heads ? c->vc.hidden / c->vc.heads : 0);
        c->vc.patch = (int)jnum(vcfg, "patch_size", 16); c->vc.pool = (int)jnum(vcfg, "pooling_kernel_size", 3);
        c->vc.npos = (int)jnum(vcfg, "position_embedding_size", 10240);
        c->vc.eps = (float)jnum(vcfg, "rms_norm_eps", 1e-6);
        jval *vr = json_get(vcfg, "rope_parameters");
        c->vc.theta = (float)jnum(vr, "rope_theta", 100.0);
        jval *sd = json_get(vcfg, "standardize");
        c->vc.standardize = sd && sd->t == J_BOOL ? sd->boolean : 0;
        if (c->vc.hidden < 1 || c->vc.layers < 1 || c->vc.heads < 1 || c->vc.hd < 4 || c->vc.hd % 4 || c->vc.inter < 1 ||
            c->vc.patch < 1 || c->vc.pool < 1 || c->vc.npos < 1)
            die("config: implausible vision_config");
        if (jnum(vcfg, "num_key_value_heads", c->vc.heads) != c->vc.heads) die("a vision tower with grouped keys and values is not supported");
    }
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

typedef struct { int8_t *q; float *s; int O, I; int clipped; float cl[4]; void *vk; } QMat;      /* vk: the matrix on the Vulkan device (COLI_VULKAN=1), made on its first use */      /* cl: a clipped linear's input min, max, output min, max */

typedef struct {
    int full;                                         /* a full-attention layer */
    int hd, nkv;                                      /* this layer's head size and KV heads */
    int kv_src;                                       /* >= 0: this layer has no keys and values of its own, it reads those of layer kv_src */
    int ff;                                           /* its dense MLP's width */
    QMat ple_gate, ple_proj; float *ple_norm;         /* the per-layer-embedding block (PLE models) */
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
    float *in_ln, *post_attn_ln, *pre_ff_ln, *post_ff_ln, *q_norm, *k_norm;
    QMat q, k, v, o, gate, up, down;
} VLayer;

typedef struct {                                      /* the vision tower and the projector into the text model's space */
    int loaded;
    QMat input_proj, proj;
    float *pos;                                       /* [2][npos][hidden]: column table, then row table */
    float *std_bias, *std_scale;
    float *inv;                                       /* rope frequencies, hd/4 */
    VLayer *L;
} Vision;

typedef struct {                                      /* one conformer layer of the audio tower */
    QMat ff_l1[2], ff_l2[2]; float *ff_pre[2], *ff_post[2];      /* the two feed-forwards, before and after the attention */
    float *norm_pre_attn, *norm_post_attn, *norm_out;
    QMat q, k, v, post, rel; float *per_dim;                     /* attention: rel is the relative-position projection (not clipped) */
    QMat lc_start, lc_end; float *lc_pre, *lc_norm, *lc_dw;      /* the light convolution; lc_dw is the depthwise kernel [hidden][kconv] */
    float *relk;                                                 /* rel applied to the positions 0..past: [past + 1][heads * hd] */
} ALayer;

typedef struct {                                      /* the audio tower and the projector into the text model's space */
    int loaded;
    float *conv0, *ln0, *conv1, *ln1;                 /* conv0 [ch0][9], conv1 [ch1][9][ch0] (taps, then input channels), and their layer norms */
    QMat in_proj, out_proj, proj; float *out_bias;
    ALayer *L;
} Audio;

/* One picture or one clip of audio inside a prompt: its soft tokens occupy positions [start, start + ntok), which a prompt's rows
 * read from `feat` instead of the embedding table, and which the sliding layers attend to in both directions. */
typedef struct {
    int kind;                                         /* 0 a picture (its rows see each other on the sliding layers), 1 audio (plain causal rows) */
    int start, ntok, gh, gw;                          /* a clip: gh frames of gw mel bins */
    unsigned char *pix;                               /* gh*gw patches of patch*patch*3 bytes, as the gateway sent them; a clip's float32 mel frames */
    float *feat;                                      /* ntok x hidden, computed when the prompt reaches the picture */
    uint64_t key;                                     /* identifies the picture: a cache is reused only for the same one */
} ImgSpan;

typedef struct {
    Cfg c; shards S;
    Audio aud; int aud_tokens;                        /* aud_tokens: the soft tokens of the longest clip accepted (0: no audio) */
    Vision vis; int rows, img_tokens;                 /* rows: the most one forward_chunk takes (a prompt chunk, or a whole picture);
                                                       * img_tokens: the soft tokens of the biggest picture accepted (0: no vision) */
    QMat embed; float *final_norm;
    QMat ple_embed, ple_mproj; float *ple_pnorm;      /* per-layer embeddings: the token table [ple_vocab][layers*P], the projection of the embedding, its norm */
    float *ple;                                       /* one chunk's per-layer inputs [rows][layers][P] */
    float *pg;                                        /* one chunk's gate [rows][P] */
    int max_ff;                                       /* the widest dense MLP of any layer */
    Layer *L;
    float *inv_s, *inv_g;                             /* rope frequencies: sliding, full */
    int cap; size_t blob_bytes;
    LCache *cache; uint64_t clock; uint32_t group; uint64_t hits, miss; double disk_s;
    uint8_t *ehit;                                    /* [layers*experts] routed this request */
    int max_t, ring, chunk, slack;                    /* chunk: rows of prompt per step (<= CHUNK); slack: positions the sliding
                                                       * layers keep beyond window + chunk, so a conversation can be rewound; ring: all three */
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
    w->O = O; w->I = I; w->vk = NULL;
    w->q = xmalloc((size_t)O * I);
    st_read_raw(&m->S, name, w->q, 0);
    load_f32(m, sname, &w->s, O);
    /* a vision linear of the E2B and E4B is "clipped": its input and output are clamped to four learned limits, stored
     * beside it as <module>.input_min, .input_max, .output_min, .output_max */
    size_t nl = strlen(name), tl = strlen(".linear.weight");
    if (nl > tl && !strcmp(name + nl - tl, ".linear.weight")) {
        static const char *limit[4] = { ".input_min", ".input_max", ".output_min", ".output_max" };
        char cn[300];
        snprintf(cn, sizeof cn, "%.*s%s", (int)(nl - tl), name, limit[0]);
        if (st_find(&m->S, cn)) {
            w->clipped = 1;
            for (int i = 0; i < 4; i++) {
                snprintf(cn, sizeof cn, "%.*s%s", (int)(nl - tl), name, limit[i]);
                float *v; load_f32(m, cn, &v, 1); w->cl[i] = v[0]; free(v);
            }
        }
    }
}

/* The vision tower, when the config describes one and the container carries it. */
static void load_vision(Model *m) {
    Cfg *c = &m->c; Vision *v = &m->vis;
    if (getenv("G4_VISION") && getenv("G4_VISION")[0] == '0') return;      /* text only, as asked: the tower is not read */
    if (!c->vc.present || !st_find(&m->S, "model.vision_tower.patch_embedder.input_proj.weight")) return;
    int D = c->vc.hidden, F = c->vc.inter, nh = c->vc.heads, hd = c->vc.hd;
    char nm[256];
    load_q(m, "model.vision_tower.patch_embedder.input_proj.weight", &v->input_proj, D, c->vc.patch * c->vc.patch * 3);
    load_f32(m, "model.vision_tower.patch_embedder.position_embedding_table", &v->pos, (int64_t)2 * c->vc.npos * D);
    if (c->vc.standardize) { load_f32(m, "model.vision_tower.std_bias", &v->std_bias, D); load_f32(m, "model.vision_tower.std_scale", &v->std_scale, D); }
    load_q(m, "model.embed_vision.embedding_projection.weight", &v->proj, c->hidden, D);
    v->L = calloc((size_t)c->vc.layers, sizeof(VLayer));
    for (int l = 0; l < c->vc.layers; l++) {
        VLayer *y = &v->L[l];
#define VNAME(tail) (snprintf(nm, sizeof nm, "model.vision_tower.encoder.layers.%d.%s", l, tail), nm)
        load_f32(m, VNAME("input_layernorm.weight"), &y->in_ln, D);
        load_f32(m, VNAME("post_attention_layernorm.weight"), &y->post_attn_ln, D);
        load_f32(m, VNAME("pre_feedforward_layernorm.weight"), &y->pre_ff_ln, D);
        load_f32(m, VNAME("post_feedforward_layernorm.weight"), &y->post_ff_ln, D);
        load_f32(m, VNAME("self_attn.q_norm.weight"), &y->q_norm, hd);
        load_f32(m, VNAME("self_attn.k_norm.weight"), &y->k_norm, hd);
        load_q(m, VNAME("self_attn.q_proj.linear.weight"), &y->q, nh * hd, D);
        load_q(m, VNAME("self_attn.k_proj.linear.weight"), &y->k, nh * hd, D);
        load_q(m, VNAME("self_attn.v_proj.linear.weight"), &y->v, nh * hd, D);
        load_q(m, VNAME("self_attn.o_proj.linear.weight"), &y->o, D, nh * hd);
        load_q(m, VNAME("mlp.gate_proj.linear.weight"), &y->gate, F, D);
        load_q(m, VNAME("mlp.up_proj.linear.weight"), &y->up, F, D);
        load_q(m, VNAME("mlp.down_proj.linear.weight"), &y->down, D, F);
#undef VNAME
    }
    /* axial rope: the first half of a head turns with the patch's column, the second with its row; each half
     * rotates its own two halves by the same hd/4 angles */
    v->inv = xmalloc((size_t)(hd / 4) * sizeof(float));
    for (int j = 0; j < hd / 4; j++) v->inv[j] = 1.f / powf(c->vc.theta, (float)(2 * j) / (float)(hd / 2));
    v->loaded = 1;
}

/* The audio tower, when the config describes one and the container carries it. */
static void load_audio(Model *m) {
    Cfg *c = &m->c; Audio *a = &m->aud;
    if (getenv("G4_AUDIO") && getenv("G4_AUDIO")[0] == '0') return;          /* no audio, as asked: the tower is not read */
    if (!c->ac.present || !st_find(&m->S, "model.audio_tower.output_proj.weight")) return;
    int D = c->ac.hidden, K = c->ac.kconv, C0 = c->ac.ch0, C1 = c->ac.ch1, hd = c->ac.hd;
    char nm[300];
    const char *T = "model.audio_tower.";
    snprintf(nm, sizeof nm, "%ssubsample_conv_projection.layer0.conv.weight", T); load_f32(m, nm, &a->conv0, (int64_t)C0 * 9);
    snprintf(nm, sizeof nm, "%ssubsample_conv_projection.layer0.norm.weight", T); load_f32(m, nm, &a->ln0, C0);
    float *raw;
    snprintf(nm, sizeof nm, "%ssubsample_conv_projection.layer1.conv.weight", T); load_f32(m, nm, &raw, (int64_t)C1 * C0 * 9);
    a->conv1 = xmalloc((size_t)C1 * 9 * C0 * sizeof(float));                  /* [oc][ic][3][3] -> [oc][tap][ic]: the input channels contiguous */
    for (int oc = 0; oc < C1; oc++) for (int ic = 0; ic < C0; ic++) for (int t = 0; t < 9; t++)
        a->conv1[((size_t)oc * 9 + t) * C0 + ic] = raw[((size_t)oc * C0 + ic) * 9 + t];
    free(raw);
    snprintf(nm, sizeof nm, "%ssubsample_conv_projection.layer1.norm.weight", T); load_f32(m, nm, &a->ln1, C1);
    snprintf(nm, sizeof nm, "%ssubsample_conv_projection.input_proj_linear.weight", T); load_q(m, nm, &a->in_proj, D, (C0 / 4) * C1);
    a->L = calloc((size_t)c->ac.layers, sizeof(ALayer));
    for (int l = 0; l < c->ac.layers; l++) {
        ALayer *y = &a->L[l];
#define ANAME(...) (snprintf(nm, sizeof nm, "%slayers.%d.", T, l), snprintf(nm + strlen(nm), sizeof nm - strlen(nm), __VA_ARGS__), nm)
        for (int i = 0; i < 2; i++) {
            load_q(m, ANAME("feed_forward%d.ffw_layer_1.linear.weight", i + 1), &y->ff_l1[i], 4 * D, D);
            load_q(m, ANAME("feed_forward%d.ffw_layer_2.linear.weight", i + 1), &y->ff_l2[i], D, 4 * D);
            load_f32(m, ANAME("feed_forward%d.pre_layer_norm.weight", i + 1), &y->ff_pre[i], D);
            load_f32(m, ANAME("feed_forward%d.post_layer_norm.weight", i + 1), &y->ff_post[i], D);
        }
        load_f32(m, ANAME("norm_pre_attn.weight"), &y->norm_pre_attn, D);
        load_f32(m, ANAME("norm_post_attn.weight"), &y->norm_post_attn, D);
        load_f32(m, ANAME("norm_out.weight"), &y->norm_out, D);
        load_q(m, ANAME("self_attn.q_proj.linear.weight"), &y->q, D, D);
        load_q(m, ANAME("self_attn.k_proj.linear.weight"), &y->k, D, D);
        load_q(m, ANAME("self_attn.v_proj.linear.weight"), &y->v, D, D);
        load_q(m, ANAME("self_attn.post.linear.weight"), &y->post, D, D);
        load_q(m, ANAME("self_attn.relative_k_proj.weight"), &y->rel, D, D);
        load_f32(m, ANAME("self_attn.per_dim_scale"), &y->per_dim, hd);
        load_q(m, ANAME("lconv1d.linear_start.linear.weight"), &y->lc_start, 2 * D, D);
        load_q(m, ANAME("lconv1d.linear_end.linear.weight"), &y->lc_end, D, D);
        load_f32(m, ANAME("lconv1d.depthwise_conv1d.weight"), &y->lc_dw, (int64_t)D * K);
        load_f32(m, ANAME("lconv1d.pre_layer_norm.weight"), &y->lc_pre, D);
        load_f32(m, ANAME("lconv1d.conv_norm.weight"), &y->lc_norm, D);
#undef ANAME
    }
    snprintf(nm, sizeof nm, "%soutput_proj.weight", T); load_q(m, nm, &a->out_proj, c->ac.out_dims, D);
    snprintf(nm, sizeof nm, "%soutput_proj.bias", T); load_f32(m, nm, &a->out_bias, c->ac.out_dims);
    load_q(m, "model.embed_audio.embedding_projection.weight", &a->proj, c->hidden, c->ac.out_dims);
    a->loaded = 1;
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
    int first_shared = c->kv_shared > 0 ? c->n_layers - c->kv_shared : c->n_layers;
    m->max_ff = c->dense_inter;
    if (c->ple_dim) {                                        /* per-layer embeddings: a table by token, and a projection of the embedding */
        int LP = c->n_layers * c->ple_dim;
        snprintf(nm, sizeof nm, "%sembed_tokens_per_layer.weight", P); load_q(m, nm, &m->ple_embed, c->ple_vocab, LP);
        snprintf(nm, sizeof nm, "%sper_layer_model_projection.weight", P); load_q(m, nm, &m->ple_mproj, LP, H);
        snprintf(nm, sizeof nm, "%sper_layer_projection_norm.weight", P); load_f32(m, nm, &m->ple_pnorm, c->ple_dim);
    }
    for (int l = 0; l < c->n_layers; l++) {
        Layer *y = &m->L[l];
        y->full = c->is_full[l];
        y->hd = y->full ? c->hd_g : c->hd_s; y->nkv = y->full ? c->nkv_g : c->nkv_s;
        y->kv_src = -1;
        if (l >= first_shared)                               /* the last layer of its kind before the sharing starts */
            for (int j = first_shared - 1; j >= 0; j--) if (c->is_full[j] == c->is_full[l]) { y->kv_src = j; break; }
        if (l >= first_shared && y->kv_src < 0) die("layer %d shares keys and values but no earlier layer is of its kind", l);
        y->ff = c->dense_inter * (c->double_wide && l >= first_shared && c->kv_shared > 0 ? 2 : 1);
        if (y->ff > m->max_ff) m->max_ff = y->ff;
#define NAME(tail) (snprintf(nm, sizeof nm, "%slayers.%d.%s", P, l, tail), nm)
        load_f32(m, NAME("input_layernorm.weight"), &y->in_ln, H);
        load_f32(m, NAME("post_attention_layernorm.weight"), &y->post_attn_ln, H);
        load_f32(m, NAME("pre_feedforward_layernorm.weight"), &y->pre_ff_ln, H);
        load_f32(m, NAME("post_feedforward_layernorm.weight"), &y->post_ff_ln, H);
        if (c->moe) {
            load_f32(m, NAME("pre_feedforward_layernorm_2.weight"), &y->pre_ff_ln2, H);
            load_f32(m, NAME("post_feedforward_layernorm_1.weight"), &y->post_ff_ln1, H);
            load_f32(m, NAME("post_feedforward_layernorm_2.weight"), &y->post_ff_ln2, H);
        }
        load_f32(m, NAME("self_attn.q_norm.weight"), &y->q_norm, y->hd);
        load_q(m, NAME("self_attn.q_proj.weight"), &y->q, c->n_heads * y->hd, H);
        if (y->kv_src < 0) {                                 /* a layer that shares keys and values has no projections for them */
            load_f32(m, NAME("self_attn.k_norm.weight"), &y->k_norm, y->hd);
            load_q(m, NAME("self_attn.k_proj.weight"), &y->k, y->nkv * y->hd, H);
            if (y->full && c->k_eq_v) y->v.q = NULL;
            else load_q(m, NAME("self_attn.v_proj.weight"), &y->v, y->nkv * y->hd, H);
        }
        load_q(m, NAME("self_attn.o_proj.weight"), &y->o, H, c->n_heads * y->hd);
        load_q(m, NAME("mlp.gate_proj.weight"), &y->gate, y->ff, H);
        load_q(m, NAME("mlp.up_proj.weight"), &y->up, y->ff, H);
        load_q(m, NAME("mlp.down_proj.weight"), &y->down, H, y->ff);
        if (c->moe) {
            load_f32(m, NAME("router.proj.weight"), &y->router_proj, (int64_t)E * H);
            load_f32(m, NAME("router.scale"), &y->router_scale, H);
            load_f32(m, NAME("router.per_expert_scale"), &y->router_pes, E);
            snprintf(y->blob_buf, sizeof y->blob_buf, "%slayers.%d.experts.blob", P, l);
            y->blob_name = y->blob_buf;
        }
        if (c->ple_dim) {
            load_q(m, NAME("per_layer_input_gate.weight"), &y->ple_gate, c->ple_dim, H);
            load_q(m, NAME("per_layer_projection.weight"), &y->ple_proj, H, c->ple_dim);
            load_f32(m, NAME("post_per_layer_input_norm.weight"), &y->ple_norm, H);
        }
        float *ls; load_f32(m, NAME("layer_scalar"), &ls, 1); y->layer_scalar = ls[0]; free(ls);
#undef NAME
    }
    load_vision(m);
    load_audio(m);
    if (m->aud.loaded) {
        int n = getenv("G4_AUDIO_TOKENS") ? atoi(getenv("G4_AUDIO_TOKENS")) : 750;     /* 750: 30 seconds, the model's own limit */
        m->aud_tokens = n < 1 ? 1 : n > 4096 ? 4096 : n;
    }
    m->rows = CHUNK;                                         /* a picture's soft tokens go through one forward_chunk together */
    if (m->vis.loaded) {
        int img = getenv("G4_IMG_TOKENS") ? atoi(getenv("G4_IMG_TOKENS")) : 280;
        if (img < 1) img = 1;
        if (img > 4096) img = 4096;
        m->img_tokens = img;
        if (img > m->rows) m->rows = img;
    }
    if (c->moe) {
        m->blob_bytes = (size_t)2 * c->moe_inter * (H / 2) + (size_t)H * (c->moe_inter / 2)
                      + 4 * ((size_t)2 * c->moe_inter * (H / 64) + (size_t)H * (c->moe_inter / 64));
        st_tensor *b = st_find(&m->S, m->L[0].blob_name);
        if (!b || b->numel != (int64_t)E * (int64_t)m->blob_bytes) die("expert blobs have the wrong size: the container does not match its config");
    }
    if (!c->moe) cap = 1;                                    /* nothing to cache */
    if (cap <= 0) {                                          /* nobody chose: what the machine can hold */
        double total = 0, avail = 0; compat_meminfo_gb(&total, &avail);
        if (avail <= 0) avail = total > 0 ? total * 0.5 : 8.0;
        double ring_gb = 0, tok_gb = 0;                      /* KV of one conversation: the rings, and a 8192-token reply's full layers */
        for (int l = 0; l < c->n_layers; l++) {
            if (m->L[l].kv_src >= 0) continue;                   /* it reads another layer's */
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
    if (c->moe) {
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
    }
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
    m->slack = getenv("G4_SLACK") ? atoi(getenv("G4_SLACK")) : 2048;
    if (m->slack < 0) m->slack = 0;
    /* a sliding row reads the window behind it; the rows of one forward_chunk are all written before any is read, so the
     * ring holds the window and the largest chunk (a prompt chunk, or a whole picture) */
    m->ring = c->window + (m->img_tokens > m->chunk ? m->img_tokens : m->chunk) + m->slack;
    int nh = c->n_heads, hdm = c->hd_g > c->hd_s ? c->hd_g : c->hd_s;
    int kvm = c->nkv_s * c->hd_s > c->nkv_g * c->hd_g ? c->nkv_s * c->hd_s : c->nkv_g * c->hd_g;
    int Fd = m->max_ff, R = m->rows;
    m->x = xmalloc((size_t)R * H * 4); m->h = xmalloc((size_t)R * H * 4);
    m->qb = xmalloc((size_t)R * nh * hdm * 4); m->ao = xmalloc((size_t)R * nh * hdm * 4);
    m->kb = xmalloc((size_t)R * kvm * 4); m->vb = xmalloc((size_t)R * kvm * 4);
    m->g1 = xmalloc((size_t)R * Fd * 4); m->u1 = xmalloc((size_t)R * Fd * 4);
    m->m1 = xmalloc((size_t)R * H * 4); m->m2 = xmalloc((size_t)R * H * 4);
    m->e_in = xmalloc((size_t)R * H * 4); m->e_out = xmalloc((size_t)R * H * 4);
    m->rl = xmalloc((size_t)R * E * 4);
    m->ridx = xmalloc((size_t)R * c->topk * sizeof(int)); m->rval = xmalloc((size_t)R * c->topk * 4);
    m->eptr = xmalloc((size_t)R * c->topk * sizeof(void *));
    m->xe = xmalloc((size_t)E * sizeof(XfExpert));
    if (c->moe) {
        m->scratch_bytes = xf_moe_scratch_bytes(R, c->topk, H, c->moe_inter);
        m->scratch = xmalloc(m->scratch_bytes);
    }
    if (c->ple_dim) {
        m->ple = xmalloc((size_t)R * c->n_layers * c->ple_dim * 4);
        m->pg = xmalloc((size_t)R * c->ple_dim * 4);
    }
    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif
    m->scores = xmalloc((size_t)T * sizeof(float *));
    /* a sliding row reads the window behind it and, inside a picture, the rest of the picture ahead of it */
    for (int i = 0; i < T; i++) m->scores[i] = xmalloc(((size_t)max_t + c->window + R + 16) * 4);
    m->trace = getenv("G4_TRACE") != NULL;
}

static void seq_init(Model *m, Seq *q) {
    Cfg *c = &m->c;
    memset(q, 0, sizeof *q);
    q->K = calloc((size_t)c->n_layers, sizeof(float *)); q->V = calloc((size_t)c->n_layers, sizeof(float *));
    for (int l = 0; l < c->n_layers; l++) {
        Layer *y = &m->L[l];
        if (y->kv_src >= 0) continue;                           /* it reads the keys and values of layer kv_src */
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
    if (i + 8 <= n) {                                   /* a head of 72 (the vision tower's) is 4 x 16 + 8 */
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        i += 8;
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
static int g_rows1 = 0;       /* a decode step: every row goes through the single-row kernel, so a row gets the same bits
                               * whether it is alone or one of several conversations' tokens (the 4-row kernel adds in another order) */

#ifdef COLI_VULKAN
/* COLI_VULKAN=1: the dense matrices (attention, dense MLP, towers, the embedding that is also the head) run on the Vulkan device,
 * uploaded once on first use. It is the same int8 rows and scales, but the sums are added in the shader's order, so the logits
 * are close to the CPU's and not equal to the bit (and a row alone and a row among several may differ). G4_VK_DENSE=0 keeps
 * them on the CPU with the device otherwise up. A matrix the device refuses (no room, too many rows) falls back to the CPU. */
static int g_gvk = 0;
static int vk_mm(float *y, const float *x, int S, const QMat *W) {
    return g_gvk && coli_vk_matmul((ColiVkTensor **)&((QMat *)W)->vk, y, x, W->q, W->s, 1, S, W->I, W->O, 0);
}
#endif

static void mm(float *y, const float *x, int S, const QMat *W) {
    int O = W->O, I = W->I;
#ifdef COLI_VULKAN
    if (vk_mm(y, x, S, W)) return;
#endif
    /* The rows go through in blocks of 32: a block of activations (32 x I floats) stays in the cache while every
     * weight row is read against it, instead of the whole of x being streamed again for each of the O weight rows.
     * That is what lets a picture's 2400 rows (or a chunk's 256) run near the arithmetic's speed. Each dot product
     * is the same sum in the same order, so the results do not change. */
    for (int s0 = 0; s0 < S; s0 += 32) {
        int sn = S - s0 < 32 ? S - s0 : 32;
#pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = W->q + (size_t)o * I; float sc = W->s[o], r[4];
            int s = s0, end = s0 + sn;
            if (!g_rows1)
                for (; s + 4 <= end; s += 4) { dot_q8(w, x + (size_t)s * I, I, 4, r); for (int k = 0; k < 4; k++) y[(size_t)(s + k) * O + o] = r[k] * sc; }
            for (; s < end; s++) { dot_q8(w, x + (size_t)s * I, I, 1, r); y[(size_t)s * O + o] = r[0] * sc; }
        }
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
    /* A cache smaller than the experts the rows want means several groups. Added group by group, a row's experts
     * would be summed in the order the groups come, which depends on what else is in the batch; so then each
     * expert's output is kept and every row is summed in its own rank order at the end, whatever the grouping. */
    float *ctb = nu > m->cap ? xmalloc((size_t)S * K * H * sizeof(float)) : NULL;
    for (int g0 = 0; g0 < nu; g0 += m->cap) {
        int n = nu - g0 < m->cap ? nu - g0 : m->cap;
        experts_ready(m, l, uniq + g0, n, slots);
        for (int e = 0; e < E; e++) pos_of[e] = -1;
        for (int i = 0; i < n; i++) { m->xe[i] = expert_view(m, slots[i]->buf); pos_of[uniq[g0 + i]] = i; }
        for (int i = 0; i < S * K; i++) m->eptr[i] = pos_of[idx[i]] >= 0 ? &m->xe[pos_of[idx[i]]] : NULL;
        if (ctb) xf_moe_compute(ctb, m->e_in, S, K, H, F, idx, m->eptr, 0, m->scratch);
        else xf_moe_add(m->e_out, m->e_in, S, K, H, F, idx, val, m->eptr, 0, m->scratch);
    }
    if (ctb) {
        for (int s = 0; s < S; s++) {
            float *os = m->e_out + (size_t)s * H;
            for (int k = 0; k < K; k++) {
                int i = s * K + k; float w = val[i]; const float *cc = ctb + (size_t)i * H;
                for (int d = 0; d < H; d++) os[d] += w * cc[d];
            }
        }
        free(ctb);
    }
    free(uniq); free(slots); free(pos_of);
}

/* ------------------------------------------------------------------------------ vision */

/* mm for a vision matrix: a clipped one clamps its input and its output to its limits. */
static void vmm(float *y, const float *x, int S, const QMat *W) {
    if (!W->clipped) { mm(y, x, S, W); return; }
    size_t n = (size_t)S * W->I, o = (size_t)S * W->O;
    float *t = xmalloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) t[i] = fminf(fmaxf(x[i], W->cl[0]), W->cl[1]);
    mm(y, t, S, W);
    free(t);
    for (size_t i = 0; i < o; i++) y[i] = fminf(fmaxf(y[i], W->cl[2]), W->cl[3]);
}

/* Turn the first half of a head by the patch's column angles and the second half by its row angles; each half
 * is rotated as two halves (rotate_half) by the same hd/4 angles. cx/sx, cy/sy: [hd/4] for this patch. */
static void rope_axial(float *x, int hd, const float *cx, const float *sx, const float *cy, const float *sy) {
    int h4 = hd / 4;
    for (int i = 0; i < h4; i++) {
        float a = x[i], b = x[i + h4];
        x[i] = a * cx[i] - b * sx[i]; x[i + h4] = b * cx[i] + a * sx[i];
        float e = x[2 * h4 + i], f = x[3 * h4 + i];
        x[2 * h4 + i] = e * cy[i] - f * sy[i]; x[3 * h4 + i] = f * cy[i] + e * sy[i];
    }
}

/* The soft tokens of one picture: (gh/pool) * (gw/pool) rows of the text model's hidden size, malloc'd.
 * pix holds gh*gw patches of patch*patch*3 bytes (row, column, channel), row-major over the grid, the bytes the
 * image processor makes before it divides by 255. transformers' Gemma4VisionModel and Gemma4MultimodalEmbedder
 * are the specification; there is no padding here, so none of the padding's handling is. */
static float *vision_encode(Model *m, const unsigned char *pix, int gh, int gw) {
    Cfg *c = &m->c; Vision *v = &m->vis;
    int D = c->vc.hidden, F = c->vc.inter, nh = c->vc.heads, hd = c->vc.hd, k = c->vc.pool, H = c->hidden;
    int P = gh * gw, pd = c->vc.patch * c->vc.patch * 3, h4 = hd / 4;
    float eps = c->vc.eps;
    double t0 = now_s();
    size_t PD = (size_t)P * D, PQ = (size_t)P * nh * hd;
    float *x = xmalloc(PD * 4), *h = xmalloc(PD * 4), *t = xmalloc(PD * 4);
    float *q = xmalloc(PQ * 4), *kk = xmalloc(PQ * 4), *vv = xmalloc(PQ * 4), *ao = xmalloc(PQ * 4);
    float *g = xmalloc((size_t)P * F * 4), *u = xmalloc((size_t)P * F * 4);
    {
        float *in = xmalloc((size_t)P * pd * 4);
        const float rescale = (float)(1.0 / 255.0);
        for (size_t i = 0; i < (size_t)P * pd; i++) in[i] = 2.f * ((float)pix[i] * rescale - 0.5f);
        mm(x, in, P, &v->input_proj);
        free(in);
    }
    float *cx = xmalloc((size_t)P * h4 * 4), *sx = xmalloc((size_t)P * h4 * 4), *cy = xmalloc((size_t)P * h4 * 4), *sy = xmalloc((size_t)P * h4 * 4);
    for (int p = 0; p < P; p++) {
        int col = p % gw, row = p / gw;
        const float *a = v->pos + (size_t)col * D, *b = v->pos + ((size_t)c->vc.npos + row) * D;
        for (int i = 0; i < D; i++) x[(size_t)p * D + i] += a[i] + b[i];
        for (int j = 0; j < h4; j++) {
            float ax = (float)col * v->inv[j], ay = (float)row * v->inv[j];
            cx[(size_t)p * h4 + j] = cosf(ax); sx[(size_t)p * h4 + j] = sinf(ax);
            cy[(size_t)p * h4 + j] = cosf(ay); sy[(size_t)p * h4 + j] = sinf(ay);
        }
    }
    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif
    float *scb = xmalloc((size_t)T * P * 4);
    double t_attn = 0, t_mm = 0, t1;
    for (int l = 0; l < c->vc.layers; l++) {
        VLayer *y = &v->L[l];
        t1 = now_s();
        for (int p = 0; p < P; p++) rms_w(h + (size_t)p * D, x + (size_t)p * D, y->in_ln, D, eps);
        vmm(q, h, P, &y->q); vmm(kk, h, P, &y->k); vmm(vv, h, P, &y->v);
        t_mm += now_s() - t1;
        for (int p = 0; p < P; p++) for (int hh = 0; hh < nh; hh++) {
            size_t o = ((size_t)p * nh + hh) * hd;
            rms_w(q + o, q + o, y->q_norm, hd, eps);
            rope_axial(q + o, hd, cx + (size_t)p * h4, sx + (size_t)p * h4, cy + (size_t)p * h4, sy + (size_t)p * h4);
            rms_w(kk + o, kk + o, y->k_norm, hd, eps);
            rope_axial(kk + o, hd, cx + (size_t)p * h4, sx + (size_t)p * h4, cy + (size_t)p * h4, sy + (size_t)p * h4);
            rms_nw(vv + o, vv + o, hd, eps);
        }
        /* every patch sees every patch; the scores are not scaled */
        t1 = now_s();
#pragma omp parallel for schedule(dynamic, 4)
        for (int it = 0; it < P * nh; it++) {
            int p = it / nh, hh = it % nh, tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float *sc = scb + (size_t)tid * P;
            const float *qh = q + ((size_t)p * nh + hh) * hd;
            float mx = -INFINITY;
            for (int j = 0; j < P; j++) { float s = dot_f32(qh, kk + ((size_t)j * nh + hh) * hd, hd); sc[j] = s; if (s > mx) mx = s; }
            float sum = 0.f;
            for (int j = 0; j < P; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
            float inv = 1.f / sum;
            float *out = ao + ((size_t)p * nh + hh) * hd;
            for (int d = 0; d < hd; d++) out[d] = 0.f;
            for (int j = 0; j < P; j++) {
                const float *vj = vv + ((size_t)j * nh + hh) * hd; float w = sc[j] * inv;
                for (int d = 0; d < hd; d++) out[d] += w * vj[d];
            }
        }
        t_attn += now_s() - t1;
        t1 = now_s();
        vmm(t, ao, P, &y->o);
        for (int p = 0; p < P; p++) {
            rms_w(t + (size_t)p * D, t + (size_t)p * D, y->post_attn_ln, D, eps);
            for (int i = 0; i < D; i++) x[(size_t)p * D + i] += t[(size_t)p * D + i];
        }
        for (int p = 0; p < P; p++) rms_w(h + (size_t)p * D, x + (size_t)p * D, y->pre_ff_ln, D, eps);
        vmm(g, h, P, &y->gate); vmm(u, h, P, &y->up);
        for (size_t i = 0; i < (size_t)P * F; i++) g[i] = gelu_tanh(g[i]) * u[i];
        vmm(t, g, P, &y->down);
        for (int p = 0; p < P; p++) {
            rms_w(t + (size_t)p * D, t + (size_t)p * D, y->post_ff_ln, D, eps);
            for (int i = 0; i < D; i++) x[(size_t)p * D + i] += t[(size_t)p * D + i];
        }
        t_mm += now_s() - t1;
    }
    /* pool k x k patches into one soft token, scale, standardise, then the projector (a weightless RMS norm and a matrix) */
    int tw = gw / k, ntok = (gh / k) * tw;
    float *pooled = calloc((size_t)ntok * D, sizeof(float));
    if (!pooled) die("out of memory (%zu bytes)", (size_t)ntok * D * sizeof(float));
    float inv_kk = 1.f / (float)(k * k), root = sqrtf((float)D);
    for (int p = 0; p < P; p++) {
        int tok = (p / gw / k) * tw + (p % gw) / k;
        for (int i = 0; i < D; i++) pooled[(size_t)tok * D + i] += x[(size_t)p * D + i] * inv_kk;
    }
    for (int tk = 0; tk < ntok; tk++) {
        float *r = pooled + (size_t)tk * D;
        for (int i = 0; i < D; i++) {
            r[i] *= root;
            if (c->vc.standardize) r[i] = (r[i] - v->std_bias[i]) * v->std_scale[i];
        }
        rms_nw(r, r, D, eps);
    }
    float *feat = xmalloc((size_t)ntok * H * 4);
    mm(feat, pooled, ntok, &v->proj);
    free(pooled); free(x); free(h); free(t); free(q); free(kk); free(vv); free(ao); free(g); free(u);
    free(cx); free(sx); free(cy); free(sy); free(scb);
    if (getenv("COLI_PREFIX_LOG") || m->trace)
        fprintf(stderr, "[gemma4] picture: %d patches -> %d soft tokens in %.2fs (attention %.2fs, matrices %.2fs)\n",
                P, ntok, now_s() - t0, t_attn, t_mm);
    return feat;
}

/* ------------------------------------------------------------------------------ audio */

static void trace(Model *m, const char *what, int l, const float *x, int n);
static inline float silu_f(float x) { return x / (1.f + expf(-x)); }

static void clamp_rows(float *x, size_t n, float limit) {            /* the tower's gradient clipping: a no-op at the checkpoint's 1e10 */
    if (limit >= 1e30f) return;
    for (size_t i = 0; i < n; i++) x[i] = fminf(fmaxf(x[i], -limit), limit);
}

/* layer norm (mean removed, weight, no bias) over n channels, then relu */
static void layernorm_relu(float *x, int n, const float *w, float eps) {
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    float r = 1.f / sqrtf(var / (float)n + eps);
    for (int i = 0; i < n; i++) { float v = (x[i] - mean) * r * w[i]; x[i] = v > 0.f ? v : 0.f; }
}

/* A feed-forward block of the conformer: clamp, norm, up, silu, down, clamp, norm, scaled, added to its input. */
static void audio_ffn(Model *m, ALayer *y, int i, float *x, int T, float *tmp, float *big) {
    Cfg *c = &m->c; int D = c->ac.hidden; float eps = c->ac.eps, lim = c->ac.clip;
    memcpy(tmp, x, (size_t)T * D * sizeof(float));
    clamp_rows(tmp, (size_t)T * D, lim);
    for (int t = 0; t < T; t++) rms_w(tmp + (size_t)t * D, tmp + (size_t)t * D, y->ff_pre[i], D, eps);
    vmm(big, tmp, T, &y->ff_l1[i]);
    for (size_t j = 0; j < (size_t)T * 4 * D; j++) big[j] = silu_f(big[j]);
    vmm(tmp, big, T, &y->ff_l2[i]);
    clamp_rows(tmp, (size_t)T * D, lim);
    for (int t = 0; t < T; t++) {
        rms_w(tmp + (size_t)t * D, tmp + (size_t)t * D, y->ff_post[i], D, eps);
        for (int d = 0; d < D; d++) x[(size_t)t * D + d] += tmp[(size_t)t * D + d] * c->ac.res_w;
    }
}

/* The soft tokens of one clip of audio: mel holds M frames of ch0 log-mel bins (the model needs as many bins as its first
 * convolution has channels: 128 in the E2B and E4B). Returns (ntok, hidden) floats, malloc'd, ntok = ceil(ceil(M/2)/2).
 * transformers' Gemma4AudioModel and Gemma4MultimodalEmbedder are the specification; the clip is not padded here, which the
 * model's masking makes equivalent (everything after the first two convolutions looks left only). */
static float *audio_encode(Model *m, const float *mel, int M, int *ntok_out) {
    Cfg *c = &m->c; Audio *a = &m->aud;
    int D = c->ac.hidden, nh = c->ac.heads, hd = c->ac.hd, K = c->ac.kconv, C0 = c->ac.ch0, C1 = c->ac.ch1, bins = C0, H = c->hidden;
    int past = c->ac.left - 1, P = past + 1; float eps = c->ac.eps, lim = c->ac.clip;
    double t0 = now_s();
    int T0 = (M - 1) / 2 + 1, F0 = (bins - 1) / 2 + 1, T1 = (T0 - 1) / 2 + 1, F1 = (F0 - 1) / 2 + 1;
    float *o0 = xmalloc((size_t)T0 * F0 * C0 * sizeof(float));
#pragma omp parallel for schedule(static)
    for (int tf = 0; tf < T0 * F0; tf++) {                                  /* conv 3x3 stride 2 pad 1 over (time, mel), one channel in */
        int t = tf / F0, f = tf % F0; float *out = o0 + (size_t)tf * C0;
        for (int oc = 0; oc < C0; oc++) {
            float acc = 0.f;
            for (int kh = 0; kh < 3; kh++) {
                int ti = 2 * t + kh - 1; if (ti < 0 || ti >= M) continue;
                for (int kw = 0; kw < 3; kw++) {
                    int fi = 2 * f + kw - 1; if (fi < 0 || fi >= bins) continue;
                    acc += a->conv0[oc * 9 + kh * 3 + kw] * mel[(size_t)ti * bins + fi];
                }
            }
            out[oc] = acc;
        }
        layernorm_relu(out, C0, a->ln0, eps);
    }
    float *o1 = xmalloc((size_t)T1 * F1 * C1 * sizeof(float));
#pragma omp parallel for schedule(static)
    for (int tf = 0; tf < T1 * F1; tf++) {
        int t = tf / F1, f = tf % F1; float *out = o1 + (size_t)tf * C1;
        for (int oc = 0; oc < C1; oc++) {
            float acc = 0.f;
            for (int kh = 0; kh < 3; kh++) {
                int ti = 2 * t + kh - 1; if (ti < 0 || ti >= T0) continue;
                for (int kw = 0; kw < 3; kw++) {
                    int fi = 2 * f + kw - 1; if (fi < 0 || fi >= F0) continue;
                    acc += dot_f32(a->conv1 + ((size_t)oc * 9 + kh * 3 + kw) * C0, o0 + ((size_t)ti * F0 + fi) * C0, C0);
                }
            }
            out[oc] = acc;
        }
        layernorm_relu(out, C1, a->ln1, eps);
    }
    free(o0);
    int T = T1;
    float *x = xmalloc((size_t)T * D * 4), *tmp = xmalloc((size_t)T * D * 4), *big = xmalloc((size_t)T * 4 * D * 4);
    float *q = xmalloc((size_t)T * D * 4), *k = xmalloc((size_t)T * D * 4), *v = xmalloc((size_t)T * D * 4), *ao = xmalloc((size_t)T * D * 4);
    mm(x, o1, T, &a->in_proj);                                              /* [T][F1 * C1] -> [T][D] */
    free(o1);
    trace(m, "asub", -1, x, T * D);
    /* the sinusoids of the relative positions past..0, as keys (once per layer) */
    float *pe = xmalloc((size_t)P * D * 4);
    {
        int half = D / 2; float inc = logf(10000.f) / (float)(half - 1 > 1 ? half - 1 : 1);
        for (int j = 0; j < P; j++) for (int i = 0; i < half; i++) {
            float ang = (float)(past - j) * expf(-(float)i * inc);
            pe[(size_t)j * D + i] = sinf(ang); pe[(size_t)j * D + half + i] = cosf(ang);
        }
    }
    float qs = powf((float)hd, -0.5f) / logf(2.f), ks = logf(1.f + expf(1.f)) / logf(2.f);
    int Tn = omp_get_max_threads();
    float *scb = xmalloc((size_t)Tn * (past + 1) * 4);
    for (int l = 0; l < c->ac.layers; l++) {
        ALayer *y = &a->L[l];
        if (!y->relk) { y->relk = xmalloc((size_t)P * D * 4); mm(y->relk, pe, P, &y->rel); }
        audio_ffn(m, y, 0, x, T, tmp, big);
        if (l == 0) trace(m, "a_ff1", l, x, T * D);
        /* chunked local attention with a relative-position bias: a row sees the `past` rows before it and itself */
        memcpy(tmp, x, (size_t)T * D * 4);
        clamp_rows(tmp, (size_t)T * D, lim);
        for (int t = 0; t < T; t++) rms_w(tmp + (size_t)t * D, tmp + (size_t)t * D, y->norm_pre_attn, D, eps);
        vmm(q, tmp, T, &y->q); vmm(k, tmp, T, &y->k); vmm(v, tmp, T, &y->v);
        for (int t = 0; t < T; t++) for (int h = 0; h < nh; h++) for (int d = 0; d < hd; d++) {
            float pd = y->per_dim[d], sp = pd > 20.f ? pd : log1pf(expf(pd));          /* softplus */
            q[(size_t)t * D + h * hd + d] = q[(size_t)t * D + h * hd + d] * qs * sp;
            k[(size_t)t * D + h * hd + d] *= ks;
        }
#pragma omp parallel for schedule(dynamic, 8)
        for (int it = 0; it < T * nh; it++) {
            int t = it / nh, h = it % nh, tid = omp_get_thread_num();
            float *sc = scb + (size_t)tid * (past + 1);
            const float *qh = q + (size_t)t * D + h * hd;
            int lo = t - past + 1 > 0 ? t - past + 1 : 0, n = t - lo + 1;     /* the reference's mask is dist < context_left - 1: `past` keys, itself included */
            float mx = -INFINITY;
            for (int s2 = lo; s2 <= t; s2++) {
                float ac = dot_f32(qh, k + (size_t)s2 * D + h * hd, hd);
                float bd = dot_f32(qh, y->relk + (size_t)(past - (t - s2)) * D + h * hd, hd);
                float z = tanhf((ac + bd) / c->ac.cap) * c->ac.cap;
                sc[s2 - lo] = z; if (z > mx) mx = z;
            }
            float sum = 0.f;
            for (int j = 0; j < n; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
            float inv = 1.f / sum, *out = ao + (size_t)t * D + h * hd;
            for (int d = 0; d < hd; d++) out[d] = 0.f;
            for (int s2 = lo; s2 <= t; s2++) {
                const float *vv = v + (size_t)s2 * D + h * hd; float w = sc[s2 - lo] * inv;
                for (int d = 0; d < hd; d++) out[d] += w * vv[d];
            }
        }
        vmm(tmp, ao, T, &y->post);
        clamp_rows(tmp, (size_t)T * D, lim);
        for (int t = 0; t < T; t++) {
            rms_w(tmp + (size_t)t * D, tmp + (size_t)t * D, y->norm_post_attn, D, eps);
            for (int d = 0; d < D; d++) x[(size_t)t * D + d] += tmp[(size_t)t * D + d];
        }
        if (l == 0) trace(m, "a_att", l, x, T * D);
        /* the light convolution: norm, up to twice the width, gated, depthwise causal convolution, norm, silu, down */
        memcpy(tmp, x, (size_t)T * D * 4);
        for (int t = 0; t < T; t++) rms_w(tmp + (size_t)t * D, tmp + (size_t)t * D, y->lc_pre, D, eps);
        vmm(big, tmp, T, &y->lc_start);
        for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) {
            float av = big[(size_t)t * 2 * D + d], gv = big[(size_t)t * 2 * D + D + d];
            tmp[(size_t)t * D + d] = av / (1.f + expf(-gv));                              /* glu */
        }
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) {
            float acc = 0.f;
            for (int kk = 0; kk < K; kk++) { int ti = t - (K - 1) + kk; if (ti >= 0) acc += y->lc_dw[(size_t)d * K + kk] * tmp[(size_t)ti * D + d]; }
            ao[(size_t)t * D + d] = acc;
        }
        clamp_rows(ao, (size_t)T * D, lim);
        for (int t = 0; t < T; t++) {
            rms_w(ao + (size_t)t * D, ao + (size_t)t * D, y->lc_norm, D, eps);
            for (int d = 0; d < D; d++) ao[(size_t)t * D + d] = silu_f(ao[(size_t)t * D + d]);
        }
        vmm(tmp, ao, T, &y->lc_end);
        for (size_t j = 0; j < (size_t)T * D; j++) x[j] += tmp[j];
        if (l == 0) trace(m, "a_lc", l, x, T * D);
        audio_ffn(m, y, 1, x, T, tmp, big);
        clamp_rows(x, (size_t)T * D, lim);
        for (int t = 0; t < T; t++) rms_w(x + (size_t)t * D, x + (size_t)t * D, y->norm_out, D, eps);
        trace(m, "alayer", l, x, T * D);
    }
    int Od = c->ac.out_dims;
    float *out = xmalloc((size_t)T * Od * 4);
    mm(out, x, T, &a->out_proj);
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < Od; d++) out[(size_t)t * Od + d] += a->out_bias[d];
        rms_nw(out + (size_t)t * Od, out + (size_t)t * Od, Od, eps);                        /* the projector: a weightless norm, then a matrix */
    }
    float *feat = xmalloc((size_t)T * H * 4);
    mm(feat, out, T, &a->proj);
    free(out); free(x); free(tmp); free(big); free(q); free(k); free(v); free(ao); free(pe); free(scb);
    if (getenv("COLI_PREFIX_LOG") || m->trace)
        fprintf(stderr, "[gemma4] audio: %d mel frames -> %d soft tokens in %.2fs\n", M, T, now_s() - t0);
    *ntok_out = T;
    return feat;
}

/* ------------------------------------------------------------------------------ the forward pass */

static void trace(Model *m, const char *what, int l, const float *x, int n) {
    if (!m->trace) return;
    double s = 0, a = 0; for (int i = 0; i < n; i++) { s += x[i]; a += fabs(x[i]); }
    fprintf(stderr, "[G4] %-10s L%-2d sum % .6f abs % .6f  %.5f %.5f %.5f\n", what, l, s, a, x[0], x[1], x[2]);
}

static const ImgSpan *span_at(const ImgSpan *sp, int nsp, int p) {
    for (int i = 0; i < nsp; i++) if (p >= sp[i].start && p < sp[i].start + sp[i].ntok) return &sp[i];
    return NULL;
}

/* How many of the `left` rows from position pos go through forward_chunk together: up to a chunk, but never part
 * of a picture. The sliding layers let a picture's rows see each other, so all of its keys must be there when
 * its first row is read: a chunk that would end inside a picture stops before it, and one that starts at a
 * picture takes the whole of it (more rows than a chunk, up to the picture's size). */
static int next_rows(const Model *m, const ImgSpan *sp, int nsp, int pos, int left) {
    int n = left < m->chunk ? left : m->chunk;
    for (int i = 0; i < nsp; i++) {
        if (sp[i].kind) continue;                          /* a clip's rows are plain causal rows: a chunk may end anywhere in one */
        int a = sp[i].start, b = a + sp[i].ntok;
        if (pos >= a && pos < b) return b - pos < left ? b - pos : left;
        if (a > pos && a < pos + n && b > pos + n) n = a - pos;
    }
    return n;
}

/* S rows at positions pos0..pos0+S-1 of one conversation q, S <= m->rows, or, with qs and ps given, S rows that
 * are each their own conversation's next token (qs[s] at position ps[s]: a decode step of several conversations,
 * q, pos0 and the pictures then unused). Leaves the residual stream in m->x.
 * sp: the pictures of the prompt; their rows are the tower's soft tokens, not embeddings, and a picture lies
 * wholly inside one chunk. */
static void forward_chunk(Model *m, Seq *q, const int *ids, int S, int pos0, const ImgSpan *sp, int nsp,
                          Seq *const *qs, const int *ps) {
    Cfg *c = &m->c;
    int H = c->hidden, nh = c->n_heads;
    double t0;
    float emb_scale = sqrtf((float)H);
    int *reach = xmalloc((size_t)S * sizeof(int));              /* the last position a sliding row reads: itself, or the end of its picture */
    for (int s = 0; s < S; s++) {                              /* embedding */
        int p = ps ? ps[s] : pos0 + s;
        const ImgSpan *g = nsp ? span_at(sp, nsp, p) : NULL;
        reach[s] = p;
        if (g) {
            if (!g->feat) die("internal: a picture's rows reached the model without its soft tokens");
            if (!g->kind) {
                reach[s] = g->start + g->ntok - 1;
                if (ps || reach[s] >= pos0 + S) die("internal: a chunk ends inside a picture (%d rows from %d)", S, pos0);
            }
            memcpy(m->x + (size_t)s * H, g->feat + (size_t)(p - g->start) * H, (size_t)H * sizeof(float));
            continue;
        }
        int id = ids[s]; if (id < 0 || id >= c->vocab) id = 0;
        const int8_t *w = m->embed.q + (size_t)id * H; float sc = m->embed.s[id] * emb_scale;
        for (int i = 0; i < H; i++) m->x[(size_t)s * H + i] = (float)w[i] * sc;
    }
    trace(m, "embed", -1, m->x + (size_t)(S - 1) * H, H);
    if (c->ple_dim) {
        /* per-layer inputs: for each row and layer a vector of ple_dim, made of what the embedding says (its projection,
         * scaled and normalised) and what the token says (its row of the per-layer table, scaled), averaged. A picture's
         * rows have the pad token's row for the second part and the picture's soft token for the first. */
        int P = c->ple_dim, LP = c->n_layers * P;
        float pscale = 1.f / sqrtf((float)H), tscale = sqrtf((float)P), comb = 0.70710678118654752f;
        mm(m->ple, m->x, S, &m->ple_mproj);
        for (int s = 0; s < S; s++) {
            int p = ps ? ps[s] : pos0 + s;
            const ImgSpan *g = nsp ? span_at(sp, nsp, p) : NULL;
            int id = g ? c->pad_token : ids[s]; if (id < 0 || id >= c->ple_vocab) id = 0;
            const int8_t *w = m->ple_embed.q + (size_t)id * LP; float sc = m->ple_embed.s[id] * tscale;
            float *pv = m->ple + (size_t)s * LP;
            for (int l = 0; l < c->n_layers; l++) {
                float *v = pv + (size_t)l * P;
                for (int i = 0; i < P; i++) v[i] *= pscale;
                rms_w(v, v, m->ple_pnorm, P, c->eps);
                for (int i = 0; i < P; i++) v[i] = (v[i] + (float)w[(size_t)l * P + i] * sc) * comb;
            }
        }
    }
    /* rope tables for the chunk: sliding and full */
    int hs2 = c->hd_s / 2, hg2 = c->hd_g / 2;
    float *cs_s = xmalloc((size_t)S * hs2 * 4), *sn_s = xmalloc((size_t)S * hs2 * 4);
    float *cs_g = xmalloc((size_t)S * hg2 * 4), *sn_g = xmalloc((size_t)S * hg2 * 4);
    for (int s = 0; s < S; s++) {
        float p = (float)(ps ? ps[s] : pos0 + s);
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
        int own = y->kv_src < 0, ksrc = own ? l : y->kv_src;         /* a layer that shares reads the keys and values of layer ksrc */
        if (own) {
            mm(m->kb, m->h, S, &y->k);
            if (y->v.q) mm(m->vb, m->h, S, &y->v);
            else memcpy(m->vb, m->kb, (size_t)S * nkv * hd * sizeof(float));      /* values are the raw keys */
        }
        for (int s = 0; s < S; s++) {
            for (int h = 0; h < nh; h++) {
                float *qh = m->qb + ((size_t)s * nh + h) * hd;
                rms_w(qh, qh, y->q_norm, hd, c->eps);
                rope_head(qh, hd, cs + (size_t)s * h2, sn + (size_t)s * h2);
            }
            if (!own) continue;
            for (int h = 0; h < nkv; h++) {
                float *kh = m->kb + ((size_t)s * nkv + h) * hd, *vh = m->vb + ((size_t)s * nkv + h) * hd;
                rms_w(kh, kh, y->k_norm, hd, c->eps);
                rope_head(kh, hd, cs + (size_t)s * h2, sn + (size_t)s * h2);
                rms_nw(vh, vh, hd, c->eps);
            }
            /* store this row's keys and values */
            int p = ps ? ps[s] : pos0 + s;
            Seq *cq = qs ? qs[s] : q;
            size_t slot = y->full ? (size_t)p : (size_t)(p % m->ring);
            memcpy(cq->K[l] + slot * nkv * hd, m->kb + (size_t)s * nkv * hd, (size_t)nkv * hd * sizeof(float));
            memcpy(cq->V[l] + slot * nkv * hd, m->vb + (size_t)s * nkv * hd, (size_t)nkv * hd * sizeof(float));
        }
        /* scores, softmax, weighted values; the model does not scale scores */
#pragma omp parallel for schedule(dynamic, 4)
        for (int it = 0; it < S * nh; it++) {
            int s = it / nh, h = it % nh, p = ps ? ps[s] : pos0 + s, kvh = h / grp;
            const Seq *cq = qs ? qs[s] : q;
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float *sc = m->scores[tid];
            int lo = y->full ? 0 : (p - c->window + 1 > 0 ? p - c->window + 1 : 0);
            int hi = y->full ? p : reach[s];                    /* full layers are causal; sliding ones read the whole picture a row is in */
            const float *qh = m->qb + ((size_t)s * nh + h) * hd;
            float mx = -INFINITY;
            for (int j = lo; j <= hi; j++) {
                size_t slot = y->full ? (size_t)j : (size_t)(j % m->ring);
                float v = dot_f32(qh, cq->K[ksrc] + (slot * nkv + kvh) * hd, hd);
                sc[j - lo] = v; if (v > mx) mx = v;
            }
            float sum = 0.f;
            for (int j = lo; j <= hi; j++) { sc[j - lo] = expf(sc[j - lo] - mx); sum += sc[j - lo]; }
            float inv = 1.f / sum;
            float *out = m->ao + ((size_t)s * nh + h) * hd;
            for (int d = 0; d < hd; d++) out[d] = 0.f;
            for (int j = lo; j <= hi; j++) {
                size_t slot = y->full ? (size_t)j : (size_t)(j % m->ring);
                const float *vv = cq->V[ksrc] + (slot * nkv + kvh) * hd; float w = sc[j - lo] * inv;
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
        /* ---- feed-forward: the dense MLP, and on a mixture-of-experts layer the experts beside it, both off the same residual ---- */
        t0 = now_s();
        int Fd = y->ff;
        for (int s = 0; s < S; s++) rms_w(m->h + (size_t)s * H, m->x + (size_t)s * H, y->pre_ff_ln, H, c->eps);
        mm(m->g1, m->h, S, &y->gate);
        mm(m->u1, m->h, S, &y->up);
        for (size_t i = 0; i < (size_t)S * Fd; i++) m->g1[i] = gelu_tanh(m->g1[i]) * m->u1[i];
        mm(m->m1, m->g1, S, &y->down);
        if (c->moe) {
            for (int s = 0; s < S; s++) rms_w(m->m1 + (size_t)s * H, m->m1 + (size_t)s * H, y->post_ff_ln1, H, c->eps);
            moe_routed(m, l, m->x, S);
            for (int s = 0; s < S; s++) {
                rms_w(m->m2 + (size_t)s * H, m->e_out + (size_t)s * H, y->post_ff_ln2, H, c->eps);
                float *a = m->m1 + (size_t)s * H, *b = m->m2 + (size_t)s * H;
                for (int i = 0; i < H; i++) a[i] += b[i];
            }
        }
        for (int s = 0; s < S; s++) {
            float *a = m->m1 + (size_t)s * H, *xr = m->x + (size_t)s * H;
            rms_w(a, a, y->post_ff_ln, H, c->eps);
            for (int i = 0; i < H; i++) xr[i] += a[i];
        }
        if (c->ple_dim) {                                        /* the per-layer-embedding block: gate, mix in this layer's input, project back */
            int P = c->ple_dim, LP = c->n_layers * P;
            mm(m->pg, m->x, S, &y->ple_gate);
            for (int s = 0; s < S; s++) {
                float *g = m->pg + (size_t)s * P; const float *pv = m->ple + (size_t)s * LP + (size_t)l * P;
                for (int i = 0; i < P; i++) g[i] = gelu_tanh(g[i]) * pv[i];
            }
            mm(m->m1, m->pg, S, &y->ple_proj);
            for (int s = 0; s < S; s++) {
                float *a = m->m1 + (size_t)s * H, *xr = m->x + (size_t)s * H;
                rms_w(a, a, y->ple_norm, H, c->eps);
                for (int i = 0; i < H; i++) xr[i] += a[i];
            }
        }
        for (int s = 0; s < S; s++) { float *xr = m->x + (size_t)s * H; for (int i = 0; i < H; i++) xr[i] *= y->layer_scalar; }
        m->t_moe += now_s() - t0;
        trace(m, "layer", l, m->x + (size_t)(S - 1) * H, H);
    }
    free(cs_s); free(sn_s); free(cs_g); free(sn_g); free(reach);
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

/* Logits of S residual rows at once (rows of xr), one pass over the tied head: out[s] gets V floats. Each row's
 * logits are what head_row gives it alone, bit for bit (the single-row kernel for every row). */
static void head_rows(Model *m, const float *xr, int S, float **out) {
    Cfg *c = &m->c; int H = c->hidden, V = c->vocab;
    double t0 = now_s();
    float *hn = xmalloc((size_t)S * H * 4);
    for (int s = 0; s < S; s++) rms_w(hn + (size_t)s * H, xr + (size_t)s * H, m->final_norm, H, c->eps);
#ifdef COLI_VULKAN
    if (g_gvk) {                                            /* the head on the device: one product for the S rows, then the soft cap */
        float *lg = xmalloc((size_t)S * V * 4);
        if (vk_mm(lg, hn, S, &m->embed)) {
            for (int s = 0; s < S; s++)
                for (int v = 0; v < V; v++) { float l = lg[(size_t)s * V + v]; out[s][v] = c->softcap > 0 ? c->softcap * tanhf(l / c->softcap) : l; }
            free(lg); free(hn);
            m->t_head += now_s() - t0;
            return;
        }
        free(lg);
    }
#endif
#pragma omp parallel for schedule(static)
    for (int v = 0; v < V; v++) {
        const int8_t *w = m->embed.q + (size_t)v * H; float sc = m->embed.s[v];
        for (int s = 0; s < S; s++) {
            float r[4]; dot_q8(w, hn + (size_t)s * H, H, 1, r);
            float l = r[0] * sc;
            out[s][v] = c->softcap > 0 ? c->softcap * tanhf(l / c->softcap) : l;
        }
    }
    free(hn);
    m->t_head += now_s() - t0;
}

/* One decode step of several conversations: row s is token ids[s] at position ps[s] of conversation qs[s]. The
 * projections, the experts and the head run once over the S rows (the weights are read once, not S times); the
 * attention reads and writes each row's own conversation. out[s] gets that conversation's logits (V floats,
 * malloc'd), the same bits it would get stepping alone. */
static void forward_rows(Model *m, Seq *const *qs, const int *ids, const int *ps, int S, float **out) {
    int V = m->c.vocab;
    g_rows1 = 1;
    forward_chunk(m, NULL, ids, S, 0, NULL, 0, qs, ps);
    for (int s = 0; s < S; s++) { kv_prefix_record(&qs[s]->kvp, ids + s, ps[s], 1); out[s] = xmalloc((size_t)V * 4); }
    head_rows(m, m->x, S, out);
    g_rows1 = 0;
    m->forwards += S;
}

/* Run `n` tokens of a conversation starting at position pos0 (chunked). Returns the logits after the
 * last token (V floats, malloc'd); with all != NULL, every row's logits too (all: n*V floats).
 * keys: what the cache records for each row (NULL: the ids), which is how a picture's rows are told from another's. */
static float *forward(Model *m, Seq *q, const int *ids, int n, int pos0, float *all,
                      const int *keys, const ImgSpan *sp, int nsp) {
    Cfg *c = &m->c; int V = c->vocab;
    float *last = xmalloc((size_t)V * 4);
    for (int s0 = 0; s0 < n;) {
        int S = next_rows(m, sp, nsp, pos0 + s0, n - s0);
        forward_chunk(m, q, ids + s0, S, pos0 + s0, sp, nsp, NULL, NULL);
        kv_prefix_record(&q->kvp, keys ? keys + s0 : ids + s0, pos0 + s0, S);
        if (all) for (int s = 0; s < S; s++) head_row(m, m->x + (size_t)s * c->hidden, all + (size_t)(s0 + s) * V);
        if (s0 + S >= n) head_row(m, m->x + (size_t)(S - 1) * c->hidden, last);
        m->forwards += S;
        s0 += S;
    }
    return last;
}

/* The cache's key for row `t` of a picture: negative, so it can never equal a token id, and different for each
 * row and each picture. */
static int image_row_key(uint64_t picture, int t) {
    uint64_t h = (picture ^ ((uint64_t)(t + 1) * 0x9E3779B97F4A7C15ULL)) * 0xD1B54A32D192ED03ULL;
    return -1 - (int)((h >> 33) & 0x3fffffff);
}

/* Find the pictures and clips a prompt's placeholders make: runs of the image token and runs of the audio token. Fills
 * sp[i].kind, .start and .ntok in the order they come, returns how many (at most cap are filled). */
static int find_pictures(const Cfg *c, const int *ids, int n, ImgSpan *sp, int cap) {
    int k = 0;
    for (int i = 0; i < n;) {
        int kind = c->image_token >= 0 && ids[i] == c->image_token ? 0 : c->audio_token >= 0 && ids[i] == c->audio_token ? 1 : -1;
        if (kind < 0) { i++; continue; }
        int j = i; while (j < n && ids[j] == ids[i]) j++;
        if (k < cap) { sp[k].kind = kind; sp[k].start = i; sp[k].ntok = j - i; }
        k++; i = j;
    }
    return k;
}

/* soft tokens of a clip of M mel frames: two convolutions of stride 2 */
static int audio_tokens_for(int frames) { int t = (frames - 1) / 2 + 1; return (t - 1) / 2 + 1; }

/* The soft tokens of a picture or a clip, computed (the tower is the work of one turn of the serve loop). */
static float *span_features(Model *m, const ImgSpan *g) {
    if (g->kind == 0) return vision_encode(m, g->pix, g->gh, g->gw);
    int n = 0;
    float *f = audio_encode(m, (const float *)g->pix, g->gh, &n);
    if (n != g->ntok) die("internal: a clip of %d frames made %d soft tokens, the prompt has %d", g->gh, n, g->ntok);
    return f;
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

static int parse_ids(const char *e, int *ids) {
    int n = 0;
    for (const char *p = e; *p;) { char *end; long v = strtol(p, &end, 10); if (end == p) break; ids[n++] = (int)v; p = *end == ',' ? end + 1 : end; }
    return n;
}

/* G4_TEST with G4_IDS and G4_IDS2: two conversations step G4_GREEDY times, each alone and then together in one batched
 * pass, and every logit of every step is compared bit for bit: a conversation must get from a batch exactly what it
 * gets alone (the serve loop's promise, and why a reply does not depend on who else is asking). The steps are fed
 * the tokens of the solo run, so the inputs are the same even if the bits were not. */
static int run_batch_test(Model *m) {
    int V = m->c.vocab, steps = getenv("G4_GREEDY") ? atoi(getenv("G4_GREEDY")) : 8;
    /* G4_IDS is the first conversation; G4_IDS2 holds the others, separated by ';' (at most MAX_SLOTS - 1) */
    const char *src[MAX_SLOTS]; int N = 0;
    src[N++] = getenv("G4_IDS");
    char *rest = strdup(getenv("G4_IDS2"));
    for (char *tokp = strtok(rest, ";"); tokp && N < MAX_SLOTS; tokp = strtok(NULL, ";")) src[N++] = tokp;
    int *ids[MAX_SLOTS], n[MAX_SLOTS];
    for (int a = 0; a < N; a++) { ids[a] = xmalloc(strlen(src[a]) * sizeof(int) + 16); n[a] = parse_ids(src[a], ids[a]); }
    float *alone[MAX_SLOTS]; int *tok[MAX_SLOTS];
    for (int a = 0; a < N; a++) {
        Seq q; seq_init(m, &q);
        alone[a] = xmalloc((size_t)steps * V * 4); tok[a] = xmalloc((size_t)steps * sizeof(int));
        float *last = forward(m, &q, ids[a], n[a], 0, NULL, NULL, NULL, 0);
        for (int st = 0; st < steps; st++) {
            tok[a][st] = argmax_v(last, V);
            free(last); last = forward(m, &q, &tok[a][st], 1, n[a] + st, NULL, NULL, NULL, 0);
            memcpy(alone[a] + (size_t)st * V, last, (size_t)V * 4);
        }
        free(last);
    }
    Seq qs[MAX_SLOTS]; Seq *qp[MAX_SLOTS];
    for (int a = 0; a < N; a++) { qp[a] = &qs[a]; seq_init(m, &qs[a]); free(forward(m, &qs[a], ids[a], n[a], 0, NULL, NULL, NULL, 0)); }
    int identical = 1; double worst = 0;
    for (int st = 0; st < steps; st++) {
        int bt[MAX_SLOTS], bp[MAX_SLOTS];
        for (int a = 0; a < N; a++) { bt[a] = tok[a][st]; bp[a] = n[a] + st; }
        float *out[MAX_SLOTS];
        forward_rows(m, qp, bt, bp, N, out);
        for (int a = 0; a < N; a++) {
            const float *want = alone[a] + (size_t)st * V;
            if (memcmp(out[a], want, (size_t)V * 4)) identical = 0;
            for (int i = 0; i < V; i++) { double d = fabs((double)out[a][i] - (double)want[i]); if (d > worst) worst = d; }
            free(out[a]);
        }
    }
    printf("{\"batch\":{\"conversations\":%d,\"steps\":%d,\"identical\":%d,\"max_diff\":%.9g}}\n", N, steps, identical, worst);
    return 0;
}

static int run_test(Model *m) {
    if (getenv("G4_IDS2")) return run_batch_test(m);
    const char *e = getenv("G4_IDS");
    if (!e) die("G4_TEST needs G4_IDS=1,2,3");
    int *ids = xmalloc(strlen(e) * sizeof(int) + 16), n = 0;
    for (const char *p = e; *p;) { char *end; long v = strtol(p, &end, 10); if (end == p) break; ids[n++] = (int)v; p = *end == ',' ? end + 1 : end; }
    int greedy = getenv("G4_GREEDY") ? atoi(getenv("G4_GREEDY")) : 0;
    int V = m->c.vocab;
    Seq q; seq_init(m, &q);
    /* pictures and clips: G4_IMAGES=file|gh|gw;... (one per run of image placeholders, each file the raw patch bytes) and
     * G4_AUDIOS=file|frames|bins;... (one per run of audio placeholders, each file float32 log-mel frames) */
    ImgSpan sp[16]; memset(sp, 0, sizeof sp);
    int nsp = find_pictures(&m->c, ids, n, sp, 16);
    int *keys = NULL;
    if (nsp > 16) die("G4_TEST: more than 16 pictures and clips");
    if (nsp) {
        const char *cur[2] = { getenv("G4_IMAGES"), getenv("G4_AUDIOS") };
        keys = xmalloc((size_t)n * sizeof(int)); memcpy(keys, ids, (size_t)n * sizeof(int));
        for (int i = 0; i < nsp; i++) {
            int kind = sp[i].kind;
            if (kind == 0 && !m->vis.loaded) die("G4_TEST: the prompt has pictures and the container has no vision tower");
            if (kind == 1 && !m->aud.loaded) die("G4_TEST: the prompt has audio and the container has no audio tower");
            if (!cur[kind]) die("G4_TEST: the prompt has %s: %s=file|a|b;...", kind ? "audio" : "pictures", kind ? "G4_AUDIOS" : "G4_IMAGES");
            char path[1024]; int gh, gw; int used = 0;
            if (sscanf(cur[kind], "%1023[^|]|%d|%d%n", path, &gh, &gw, &used) != 3) die("G4_TEST: bad entry near '%s'", cur[kind]);
            cur[kind] += used; if (*cur[kind] == ';') cur[kind]++;
            size_t bytes;
            if (kind == 0) {
                int k = m->c.vc.pool;
                if (gh % k || gw % k || (gh / k) * (gw / k) != sp[i].ntok) die("G4_IMAGES: %dx%d patches give %d soft tokens, the prompt has %d", gh, gw, (gh / k) * (gw / k), sp[i].ntok);
                bytes = (size_t)gh * gw * m->c.vc.patch * m->c.vc.patch * 3;
            } else {
                if (gw != m->c.ac.ch0 || audio_tokens_for(gh) != sp[i].ntok) die("G4_AUDIOS: %d frames of %d bins give %d soft tokens, the prompt has %d", gh, gw, audio_tokens_for(gh), sp[i].ntok);
                bytes = (size_t)gh * gw * sizeof(float);
            }
            sp[i].pix = xmalloc(bytes);
            FILE *f = fopen(path, "rb");
            if (!f || fread(sp[i].pix, 1, bytes, f) != bytes) die("G4_TEST: cannot read %zu bytes from %s", bytes, path);
            fclose(f);
            sp[i].gh = gh; sp[i].gw = gw;
            sp[i].key = kv_image_hash(sp[i].pix, bytes, gh, gw) ^ (kind ? 0x5A5A5A5A5A5A5A5AULL : 0);
            sp[i].feat = span_features(m, &sp[i]);
            for (int t = 0; t < sp[i].ntok; t++) keys[sp[i].start + t] = image_row_key(sp[i].key, t);
        }
    }
    float *all = xmalloc((size_t)n * V * 4);
    float *last = forward(m, &q, ids, n, 0, all, keys, sp, nsp);
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
        free(last); last = forward(m, &q, &t, 1, n + g, NULL, NULL, NULL, 0);
    }
    printf("]");
    if (nsp) {                                                 /* the soft tokens the tower made, for the tower's own test */
        printf(",\"features\":[");
        for (int i = 0; i < nsp; i++) {
            printf(i ? ",[" : "[");
            for (size_t j = 0; j < (size_t)sp[i].ntok * m->c.hidden; j++) printf(j ? ",%.9g" : "%.9g", sp[i].feat[j]);
            printf("]");
        }
        printf("]");
    }
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
    int *keys;                          /* what the cache records per prompt row: the ids, a picture's rows keyed by the picture */
    ImgSpan *sp; int nsp;               /* the prompt's pictures */
    Cn *cn;                             /* the request's grammar (response_format), NULL when it has none: constrain.h */
} Req;

#define MAX_PICS 40                   /* pictures or clips in one request: 2 * MAX_PICS, enough for a video's 32 frames and some more */
/* Pictures that arrived (IMAGE frames) for the SUBMIT that follows; the gateway writes them together. */
static struct { int kind; unsigned char *pix; int gh, gw; } g_pend[2 * MAX_PICS];      /* kind 0 a picture, 1 a clip of audio */
static int g_npend, g_pend_bad;
static void pending_clear(void) { for (int i = 0; i < g_npend; i++) free(g_pend[i].pix); g_npend = 0; }

/* The optional 7th numeric field of SUBMIT is the byte length of a grammar that follows the prompt (what the gateway sends
 * GLM for its grammar-forced drafts); here it constrains the answer (constrain.h). The gateway sends it only after the
 * CAPS line said grammar=1. */
static const ColiServeWireProfile g4_wire = {
    .max_header_bytes = 511, .max_payload_bytes = 1u << 25, .max_tokens = 1 << 20,
    .require_exact_lf = 1, .require_finite_sampling = 0,
    .max_extension_bytes = 1u << 20, .allow_extension_bytes = 1,
};

/* What each token adds to the answer, for constrain.h: built from the tokenizer the first time a request brings a grammar. */
static CnVocab *g4_cn_vocab(const Cfg *c, const G4Tok *T) {
    CnVocab *v = cn_vocab_new(c->vocab);
    if (!v) return NULL;
    static char piece[4096];
    for (int t = 0; t < c->vocab; t++) {
        if (is_stop(c, t)) { cn_vocab_set(v, t, NULL, 0, CN_END); continue; }
        if (t >= T->n_ids || !T->piece[t] || T->is_special[t]) continue;            /* stays CN_SPECIAL: never chosen */
        int n = g4_tok_piece(T, t, piece, sizeof piece);
        if (n > 0 && n < (int)sizeof piece) cn_vocab_set(v, t, (const unsigned char *)piece, n, CN_TEXT);
    }
    return v;
}

typedef struct { int V, top_k; float temp, top_p; } G4Sampler;
static int g4_sample_cb(const float *lo, void *ctx) {
    const G4Sampler *s = (const G4Sampler *)ctx;
    return pick_token(lo, s->V, s->temp, s->top_p, s->top_k);
}

static void serve_hits(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts, rows = c->n_layers;
    if (!c->moe) return;                                       /* a dense model routes nothing */
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
    if (!c->moe) return;
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

static void req_release(Req *r) {
    free(r->ids); free(r->lo); r->ids = NULL; r->lo = NULL; r->active = 0;
    free(r->keys); r->keys = NULL;
    for (int i = 0; i < r->nsp; i++) { free(r->sp[i].pix); free(r->sp[i].feat); }
    free(r->sp); r->sp = NULL; r->nsp = 0;
    cn_free(r->cn); r->cn = NULL;
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
    req_release(r);
}

/* A request that cannot go on: the gateway gets the ERROR (its answer to the client) and no DONE. */
static void req_fail(Model *m, Req *r, const char *msg) {
    coli_serve_write_error(stdout, r->id, msg);
    serve_hits(m);
    req_release(r);
}

static int kv_prefix_off(void) { const char *e = getenv("COLI_KV_PREFIX"); return e && *e == '0'; }

/* How much of what a slot holds this prompt may keep. Colibri's other engines keep a cache only when the new prompt
 * begins with everything fed. That would waste Gemma 4: its template drops the empty thought channel (and a past
 * turn's reasoning) from the history, so a resent conversation differs from what was fed just after "<|turn>model",
 * and the whole conversation would be read again every turn. Here the common prefix is kept, and the positions
 * after it forgotten, as long as the sliding layers still hold the window before it: their ring keeps the last
 * `ring` positions, so the prefix may end at most ring - window + 1 before the end of what was fed. The full layers
 * keep every position, so they never limit it. At least one token is always left to feed. */
static int reuse_len(Model *m, Seq *q, const int *keys, int np, const ImgSpan *sp, int nsp) {
    kv_prefix *p = &q->kvp;
    if (!p->fed || p->tainted || p->len <= 0) return 0;
    int c = 0, top = p->len < np ? p->len : np;
    while (c < top && p->fed[c] == keys[c]) c++;
    if (c > np - 1) c = np - 1;
    for (int i = 0; i < nsp; i++)                       /* a picture is kept whole or not at all: its rows were made together */
        if (c > sp[i].start && c < sp[i].start + sp[i].ntok) c = sp[i].start;
    if (c <= 0) return 0;
    if (p->len > m->ring && p->len - c > m->ring - m->c.window + 1) return 0;      /* the window before c is gone */
    p->len = c;
    return c;
}

/* Begin a request: tokenize, budget, find what the slot already holds. 0 when it ended here. */
static int req_start(Model *m, G4Tok *T, Req *r, Seq *q, const char *payload, int plen, int ctx_cap) {
    int cap = plen + 16;
    r->ids = xmalloc((size_t)cap * sizeof(int));
    r->np = g4_tok_encode(T, payload, plen, r->ids, cap);
    if (r->np <= 0) { coli_serve_write_error(stdout, r->id, "empty prompt"); free(r->ids); r->ids = NULL; pending_clear(); return 0; }
    int budget = coli_serve_budget(r->np, r->max_tok, ctx_cap, 0);
    if (budget < 0) {
        char msg[160]; snprintf(msg, sizeof msg, "CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d", r->np, r->max_tok, ctx_cap);
        coli_serve_write_error(stdout, r->id, msg); free(r->ids); r->ids = NULL; pending_clear(); return 0;
    }
    if (budget < r->max_tok) r->max_tok = budget;
    /* pictures and clips: one run of placeholders for each that arrived, as many soft tokens as it makes */
    r->keys = xmalloc((size_t)r->np * sizeof(int)); memcpy(r->keys, r->ids, (size_t)r->np * sizeof(int));
    r->sp = NULL; r->nsp = 0;
    if (m->vis.loaded || m->aud.loaded) {
        ImgSpan found[2 * MAX_PICS + 1]; memset(found, 0, sizeof found);
        int nf = find_pictures(&m->c, r->ids, r->np, found, 2 * MAX_PICS + 1), k = m->c.vc.pool;
        int queue[2][2 * MAX_PICS], have[2] = { 0, 0 }, want_n[2] = { 0, 0 }, used[2] = { 0, 0 };
        for (int i = 0; i < g_npend; i++) queue[g_pend[i].kind][have[g_pend[i].kind]++] = i;
        for (int i = 0; i < nf && i < 2 * MAX_PICS + 1; i++) want_n[found[i].kind]++;
        char msg[200]; msg[0] = 0;
        if (nf > 2 * MAX_PICS) snprintf(msg, sizeof msg, "BAD_IMAGE the prompt has more than %d pictures and clips", 2 * MAX_PICS);
        else if (want_n[0] != have[0]) snprintf(msg, sizeof msg, "BAD_IMAGE the prompt has %d image placeholder run%s and %d picture%s arrived", want_n[0], want_n[0] == 1 ? "" : "s", have[0], have[0] == 1 ? "" : "s");
        else if (want_n[1] != have[1]) snprintf(msg, sizeof msg, "BAD_AUDIO the prompt has %d audio placeholder run%s and %d clip%s arrived", want_n[1], want_n[1] == 1 ? "" : "s", have[1], have[1] == 1 ? "" : "s");
        for (int i = 0; !msg[0] && i < nf; i++) {
            int kind = found[i].kind, idx = queue[kind][used[kind]++];
            int want = kind ? audio_tokens_for(g_pend[idx].gh) : (g_pend[idx].gh / k) * (g_pend[idx].gw / k);
            if (found[i].ntok != want)
                snprintf(msg, sizeof msg, kind ? "BAD_AUDIO clip %d has %d soft tokens and the prompt %d placeholders" : "BAD_IMAGE picture %d has %d soft tokens and the prompt %d placeholders", used[kind], want, found[i].ntok);
        }
        if (msg[0]) {
            coli_serve_write_error(stdout, r->id, msg);
            free(r->ids); r->ids = NULL; free(r->keys); r->keys = NULL; pending_clear(); return 0;
        }
        if (nf) {
            r->sp = calloc((size_t)nf, sizeof(ImgSpan));
            used[0] = used[1] = 0;
            for (int i = 0; i < nf; i++) {
                ImgSpan *g = &r->sp[i];
                int kind = found[i].kind, idx = queue[kind][used[kind]++];
                g->kind = kind; g->start = found[i].start; g->ntok = found[i].ntok; g->gh = g_pend[idx].gh; g->gw = g_pend[idx].gw;
                g->pix = g_pend[idx].pix; g_pend[idx].pix = NULL;
                size_t bytes = kind ? (size_t)g->gh * g->gw * sizeof(float) : (size_t)g->gh * g->gw * m->c.vc.patch * m->c.vc.patch * 3;
                g->key = kv_image_hash(g->pix, bytes, g->gh, g->gw) ^ (kind ? 0x5A5A5A5A5A5A5A5AULL : 0);
                for (int t = 0; t < g->ntok; t++) r->keys[g->start + t] = image_row_key(g->key, t);
            }
            r->nsp = nf;
        }
        g_npend = 0;
    }
    int reuse = kv_prefix_off() ? 0 : reuse_len(m, q, r->keys, r->np, r->sp, r->nsp);
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
    {   char caps[128]; snprintf(caps, sizeof caps, "vision=%d img_tokens=%d audio=%d audio_tokens=%d grammar=1", m->vis.loaded, m->img_tokens, m->aud.loaded, m->aud_tokens);
        coli_serve_write_ready_caps(stdout, rss_gb(), caps); }
    serve_hwinfo(); serve_tiers_emap(m);
    if (c->moe) fprintf(stderr, "[gemma4] serving %d conversation%s, context %d, %d experts cached per layer\n", n_slots, n_slots == 1 ? "" : "s", ctx_cap, m->cap);
    else fprintf(stderr, "[gemma4] serving %d conversation%s, context %d (a dense model)\n", n_slots, n_slots == 1 ? "" : "s", ctx_cap);
    int input_eof = 0, rr = 0;
    CnVocab *cn_vocab_g = NULL;                                  /* made when the first grammar arrives */
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
                if (cmd.kind == COLI_SERVE_COMMAND_CANCEL || cmd.kind == COLI_SERVE_COMMAND_STOP) {   /* both end the request with its DONE */
                    for (int i = 0; i < n_slots; i++) if (rq[i].active && !strcmp(rq[i].id, cmd.id)) rq[i].cancel = 1;
                } else if (cmd.kind == COLI_SERVE_COMMAND_IMAGE) {
                    /* IMAGE <id> <bytes> <gh> <gw>: the patch grid of one picture, for the SUBMIT that follows. Its payload
                     * is already read; a refusal here makes the SUBMIT after it a no-op (the gateway has the ERROR). */
                    Cfg *cc = &m->c; int k = cc->vc.pool;
                    size_t want = (size_t)cmd.grid_h * cmd.grid_w * cc->vc.patch * cc->vc.patch * 3;
                    const char *why = NULL; char msg[160];
                    if (!m->vis.loaded) why = "this engine has no vision tower; images are not supported";
                    else if (g_npend >= 2 * MAX_PICS) { snprintf(msg, sizeof msg, "BAD_IMAGE at most %d pictures and clips in a request", 2 * MAX_PICS); why = msg; }
                    else if (cmd.grid_h % k || cmd.grid_w % k || cmd.grid_h > cc->vc.npos || cmd.grid_w > cc->vc.npos) {
                        snprintf(msg, sizeof msg, "BAD_IMAGE the patch grid %dx%d is not a multiple of %d or exceeds %d", cmd.grid_h, cmd.grid_w, k, cc->vc.npos); why = msg; }
                    else if ((size_t)cmd.payload_bytes != want) {
                        snprintf(msg, sizeof msg, "BAD_IMAGE bytes=%llu expected=%zu", (unsigned long long)cmd.payload_bytes, want); why = msg; }
                    else if ((cmd.grid_h / k) * (cmd.grid_w / k) > m->img_tokens) {
                        snprintf(msg, sizeof msg, "BAD_IMAGE the picture makes %d soft tokens and this engine takes %d (G4_IMG_TOKENS)", (cmd.grid_h / k) * (cmd.grid_w / k), m->img_tokens); why = msg; }
                    if (why) { coli_serve_write_error(stdout, cmd.id, why); g_pend_bad = 1; }
                    else {
                        g_pend[g_npend].kind = 0; g_pend[g_npend].pix = (unsigned char *)coli_serve_command_take_payload(&cmd);
                        g_pend[g_npend].gh = cmd.grid_h; g_pend[g_npend].gw = cmd.grid_w; g_npend++;
                    }
                } else if (cmd.kind == COLI_SERVE_COMMAND_AUDIO) {
                    /* AUDIO <id> <bytes> <frames> <bins>: the log-mel frames of one clip, float32, for the SUBMIT that follows */
                    Cfg *cc = &m->c;
                    size_t want = (size_t)cmd.grid_h * cmd.grid_w * sizeof(float);
                    const char *why = NULL; char msg[160];
                    if (!m->aud.loaded) why = "this engine has no audio tower; audio is not supported";
                    else if (g_npend >= 2 * MAX_PICS) { snprintf(msg, sizeof msg, "BAD_AUDIO at most %d pictures and clips in a request", 2 * MAX_PICS); why = msg; }
                    else if (cmd.grid_w != cc->ac.ch0) { snprintf(msg, sizeof msg, "BAD_AUDIO %d mel bins and this model reads %d", cmd.grid_w, cc->ac.ch0); why = msg; }
                    else if ((size_t)cmd.payload_bytes != want) { snprintf(msg, sizeof msg, "BAD_AUDIO bytes=%llu expected=%zu", (unsigned long long)cmd.payload_bytes, want); why = msg; }
                    else if (audio_tokens_for(cmd.grid_h) > m->aud_tokens) {
                        snprintf(msg, sizeof msg, "BAD_AUDIO the clip makes %d soft tokens and this engine takes %d (G4_AUDIO_TOKENS)", audio_tokens_for(cmd.grid_h), m->aud_tokens); why = msg; }
                    if (why) { coli_serve_write_error(stdout, cmd.id, why); g_pend_bad = 1; }
                    else {
                        g_pend[g_npend].kind = 1; g_pend[g_npend].pix = (unsigned char *)coli_serve_command_take_payload(&cmd);
                        g_pend[g_npend].gh = cmd.grid_h; g_pend[g_npend].gw = cmd.grid_w; g_npend++;
                    }
                } else if (cmd.kind == COLI_SERVE_COMMAND_SUBMIT && g_pend_bad) {
                    g_pend_bad = 0; pending_clear();                 /* its picture was refused and answered: nothing more to say */
                    free(coli_serve_command_take_payload(&cmd));
                } else if (cmd.kind == COLI_SERVE_COMMAND_SUBMIT) {
                    if (cmd.slot < 0 || cmd.slot >= n_slots) { coli_serve_write_error(stdout, cmd.id, "invalid cache slot"); pending_clear(); }
                    else if (rq[cmd.slot].active) { coli_serve_write_error(stdout, cmd.id, "SLOT_BUSY"); pending_clear(); }
                    else {
                        Req *r = &rq[cmd.slot];
                        memset(r, 0, sizeof *r);
                        snprintf(r->id, sizeof r->id, "%s", cmd.id);
                        r->max_tok = cmd.max_tokens; r->temp = cmd.temperature; r->top_p = cmd.top_p;
                        char *payload = (char *)coli_serve_command_take_payload(&cmd);
                        int ok = 1;
                        if (cmd.extension_bytes) {                       /* a grammar: compile it before anything else is touched */
                            char *gtext = xmalloc((size_t)cmd.extension_bytes + 1), gerr[200];
                            memcpy(gtext, payload + cmd.payload_bytes + 1, (size_t)cmd.extension_bytes);
                            gtext[cmd.extension_bytes] = 0;
                            if (!cn_vocab_g && !(cn_vocab_g = g4_cn_vocab(c, T))) { coli_serve_write_error(stdout, r->id, "out of memory"); ok = 0; }
                            else if (!(r->cn = cn_new(gtext, gerr, sizeof gerr))) {
                                char msg[260]; snprintf(msg, sizeof msg, "BAD_GRAMMAR %s", gerr);
                                coli_serve_write_error(stdout, r->id, msg); ok = 0;
                            }
                            free(gtext);
                            if (!ok) pending_clear();
                        }
                        if (ok && req_start(m, T, r, &seq[cmd.slot], payload, (int)cmd.payload_bytes, ctx_cap))
                            coli_serve_write_accept(stdout, r->id, r->np);
                        else cn_free(r->cn), r->cn = NULL;               /* r is inactive here: a request that ended before it began */
                        free(payload);
                    }
                } else {
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
            int n = next_rows(m, r->sp, r->nsp, r->done_np, r->np - r->done_np);
            ImgSpan *pic = NULL;                               /* a picture in the rows about to run, not yet seen by the tower: that is this turn's work */
            for (int k = 0; k < r->nsp; k++)
                if (!r->sp[k].feat && r->sp[k].start < r->done_np + n && r->sp[k].start + r->sp[k].ntok > r->done_np) { pic = &r->sp[k]; break; }
            if (pic) { pic->feat = span_features(m, pic); continue; }
            float *lo = forward(m, q, r->ids + r->done_np, n, r->done_np, NULL, r->keys + r->done_np, r->sp, r->nsp);
            r->done_np += n;
            if (r->done_np >= r->np) { r->lo = lo; r->phase = 1; } else free(lo);
            continue;
        }
        /* decode: every conversation that is decoding takes its step together. Each picks its next token from its
         * logits and sends it; the ones that go on are fed in one pass, so the weights are read once for all of them
         * (what makes two conversations at once cost far less than twice one). A conversation gets the bits it would
         * alone. */
        int bi[MAX_SLOTS], bt[MAX_SLOTS], bp[MAX_SLOTS], nb = 0; Seq *bq[MAX_SLOTS];
        for (int j = 0; j < n_slots; j++) {
            Req *w = &rq[j];
            if (!w->active || w->phase != 1) continue;
            if (w->cancel) { w->limited = 0; req_finish(m, w); serve_tiers_emap(m); continue; }
            int t;
            if (w->cn) {                                         /* a grammar: only tokens that keep the answer inside it */
                G4Sampler sm = { V, (int)top_k_env, w->temp, w->top_p };
                t = cn_pick(w->cn, cn_vocab_g, w->lo, g4_sample_cb, &sm);
                if (t < 0) {
                    req_fail(m, w, t == -1 ? "GRAMMAR_STUCK no token of this vocabulary can continue the grammar"
                                           : "GRAMMAR_LOST the grammar is too ambiguous to follow");
                    serve_tiers_emap(m); continue;
                }
            } else t = pick_token(w->lo, V, w->temp, w->top_p, (int)top_k_env);
            if (is_stop(c, t)) { w->limited = 0; req_finish(m, w); serve_tiers_emap(m); continue; }
            char piece[512]; int np_ = g4_tok_piece(T, t, piece, sizeof piece);
            coli_serve_write_data(stdout, w->id, piece, (size_t)np_);
            w->gen++;
            if (w->gen >= w->max_tok || w->np + w->gen >= ctx_cap) { req_finish(m, w); serve_tiers_emap(m); continue; }
            free(w->lo); w->lo = NULL;
            bi[nb] = j; bt[nb] = t; bp[nb] = w->np + w->gen - 1; bq[nb] = &seq[j]; nb++;
        }
        if (nb) {
            float *outs[MAX_SLOTS];
            forward_rows(m, bq, bt, bp, nb, outs);
            for (int b = 0; b < nb; b++) rq[bi[b]].lo = outs[b];
        }
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
#ifdef COLI_VULKAN
    if (coli_vk_init_env("gemma4")) {                  /* COLI_VULKAN=1 and a usable device */
        const char *d = getenv("G4_VK_DENSE");
        g_gvk = d && *d ? atoi(d) != 0 : 1;
        atexit(coli_vk_shutdown);
        fprintf(stderr, "[gemma4] dense matrices on %s\n", g_gvk ? coli_vk_device_name() : "the CPU (G4_VK_DENSE=0)");
    }
#endif
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
