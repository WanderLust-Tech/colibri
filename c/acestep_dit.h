/* acestep_dit.h -- the ACE-Step 1.5 diffusion transformer, its condition encoders and the turbo sampler, pure C on
 * top of qi_gemm. Text-to-music (the planner and the reference-audio tasks are not in here).
 *
 * What one clip costs, in four pieces:
 *
 *   condition  (once per request)  text hidden states (Qwen3-Embedding, 1024 wide) -> a linear layer; lyric
 *              embedding-table rows -> a linear layer, 8 encoder layers, a norm; the timbre reference (750 frames of
 *              the silence latent) -> a linear layer, 4 encoder layers, a norm, and only its FIRST frame's output
 *              (the special token the class once prepended is commented out upstream). The three are packed in the
 *              order lyric, timbre, text, then go through the DiT's condition_embedder.
 *   cross K/V  (once per request)  the condition tokens' keys and values in every DiT layer: they do not depend on
 *              the step, so they are computed once and every step only runs its queries against them.
 *   forward    (once per step)  latents [T][64] ++ context [T][128] -> patches of 2 frames (a stride-2 convolution)
 *              -> 24 layers -> adaptive norm -> a stride-2 transposed convolution back to [T][64]. A layer is
 *              AdaLN-modulated self-attention (gated), plain cross-attention, AdaLN-modulated SwiGLU (gated); the
 *              modulation comes from two timestep embeddings, one of t and one of t - r (which is 0 in turbo).
 *   sampler    Euler on the flow, x <- x - v dt, the last step x - v t; then DCW, a Haar-wavelet correction (below).
 *
 * Attention is Qwen3's: grouped-query, RMS-normalised queries and keys (per head), RoPE on self-attention only,
 * bidirectional. Layers alternate sliding (|i - j| <= window, in tokens) and full, by config.layer_types. Upstream
 * builds 4-D masks, and in the DiT it drops the padding mask altogether (attention_mask is reassigned to None);
 * for one clip without padding that is the plain band or the full matrix, which is what is computed here: a
 * sliding layer only ever touches the keys inside the band, so a 10-minute clip costs the band, not T^2.
 *
 * Weights stay bf16 in memory and are widened inside the GEMM (exactly, so the result is the float32 product of the
 * checkpoint's weights, as the reference computes it). Everything the checkpoint fixes is read from config.json
 * and shape-checked; nothing is assumed.
 *
 * DCW (Differential Correction in Wavelet domain, upstream's default for turbo): after each step, with
 * e = x_next - denoised, denoised = x_before - v t, and a Haar transform along time, the low band is boosted by
 * t * 0.05 and the high band by (1 - t) * 0.02. In the time domain, per pair of frames (e0, e1):
 *     x'[2k]   = x[2k]   + lo (e0 + e1)/2 + hi (e0 - e1)/2
 *     x'[2k+1] = x[2k+1] + lo (e0 + e1)/2 - hi (e0 - e1)/2
 * and an odd last frame pairs with a zero. (Checked against upstream's pytorch_wavelets implementation, odd and
 * even lengths, to float rounding.)
 *
 * Interface: acd_load(model_dir) / acd_cond_new(...) / acd_forward(...) / acd_sample(...) / acd_free.
 */
#ifndef COLIBRI_ACESTEP_DIT_H
#define COLIBRI_ACESTEP_DIT_H
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

#define ACD_MAX_LAYERS 128
#define ACD_MAX_DIM 16384
#define ACD_MAX_HEADS 256
#define ACD_MAX_HD 512
#define ACD_MAX_PATCH 8
#define ACD_QB 128                  /* query rows per attention block */

typedef struct { QiMat m; float *b; void *own; } AcdLin;     /* y = x W^T + b; W [N][K], bf16 or f32; b may be NULL */
typedef struct { AcdLin q, k, v, o; float *qn, *kn; } AcdAttn;
typedef struct { AcdAttn sa; float *ln1, *ln2; AcdLin gate, up, down; int sliding; } AcdEncLayer;
typedef struct {
    AcdAttn sa, ca; float *n_sa, *n_ca, *n_mlp, *sst;          /* sst [6][D] */
    AcdLin gate, up, down; int sliding;
} AcdLayer;
typedef struct { AcdLin l1, l2, proj; } AcdTimeEmb;

typedef struct {
    int D, inter, nl, heads, kv, hd, window, patch, inch, acdim, textdim, timbredim, nlyric, ntimbre, tfix;
    float eps, theta;
    int *sliding;                                              /* per layer index: 1 = sliding, 0 = full */
    AcdTimeEmb te, ter;
    AcdLin cond_emb, text_proj, lyric_embed, timbre_embed, proj_in, proj_out;
    float *lyric_norm, *timbre_norm, *norm_out, *sst_out;      /* sst_out [2][D] */
    AcdEncLayer *lyric, *timbre;
    AcdLayer *L;
    size_t wbytes;
} AceDit;

/* What one request's condition tokens make of the network: the packed tokens and every layer's cross K/V. */
typedef struct { int len; float *enc; float **ck, **cv; int nl; } AcdCond;

/* ----------------------------------------------------------------- loading -- */

static float *acd_read_f32(shards *S, const char *name, int rank, const int64_t *shape, char *err, size_t errn){
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
    float *p = (float *)malloc((size_t)(numel ? numel : 1) * sizeof(float));
    if (!p) { snprintf(err, errn, "out of memory reading %s", name); return NULL; }
    st_read_f32_cap(S, name, p, numel, 1);
    return p;
}

/* A weight matrix [n][k], kept bf16 (or f32 when the file is f32; f16 is widened): the raw bytes of tensor `name`, whose
 * file shape is `rank` dims with the leading n*k elements in row-major order. `perm`, when not NULL, is called to build
 * the [n][k] layout from the raw one (the convolutions' taps move to the columns). */
typedef void (*AcdPerm)(const uint16_t *raw, uint16_t *out, int n, int k, int aux);

static int acd_load_lin(shards *S, const char *wname, const char *bname, int n, int k, int rank, const int64_t *shape,
                        AcdPerm perm, int aux, int bias_n, AcdLin *l, size_t *bytes, char *err, size_t errn){
    st_tensor *t = st_find(S, wname);
    if (!t) { snprintf(err, errn, "missing tensor %s", wname); return -1; }
    int ok = t->rank == rank;
    int64_t numel = 1;
    for (int i = 0; i < rank; i++) numel *= shape[i];
    for (int i = 0; ok && i < rank; i++) ok = t->shape[i] == shape[i];
    if (!ok || t->numel != numel || (int64_t)n * k != numel) {
        char got[96] = "", want[96] = "";
        for (int i = 0; i < t->rank && i < ST_MAX_RANK; i++) snprintf(got + strlen(got), sizeof got - strlen(got), "%s%lld", i ? "x" : "", (long long)t->shape[i]);
        for (int i = 0; i < rank; i++) snprintf(want + strlen(want), sizeof want - strlen(want), "%s%lld", i ? "x" : "", (long long)shape[i]);
        snprintf(err, errn, "tensor %s is %s, expected %s", wname, got, want);
        return -1;
    }
    if (t->dtype > 2) { snprintf(err, errn, "tensor %s is %s, not a float tensor", wname, st_dtype_name(t->dtype)); return -1; }
    void *own = NULL;
    if (t->dtype == 0) {                                   /* bf16: kept as it is */
        uint16_t *raw = (uint16_t *)malloc((size_t)numel * 2);
        if (!raw) { snprintf(err, errn, "out of memory reading %s", wname); return -1; }
        st_read_raw(S, wname, raw, 1);
        if (perm) {
            uint16_t *out = (uint16_t *)malloc((size_t)numel * 2);
            if (!out) { free(raw); snprintf(err, errn, "out of memory reading %s", wname); return -1; }
            perm(raw, out, n, k, aux);
            free(raw); raw = out;
        }
        own = raw; l->m = (QiMat){QI_BF16, n, k, raw, NULL, 0}; *bytes += (size_t)numel * 2;
    } else {                                               /* f32 or f16: widened to f32 */
        float *w = (float *)malloc((size_t)numel * sizeof(float));
        if (!w) { snprintf(err, errn, "out of memory reading %s", wname); return -1; }
        st_read_f32_cap(S, wname, w, numel, 1);
        if (perm) {                                        /* the tap permutations are written for 16-bit words */
            snprintf(err, errn, "tensor %s: only a bf16 checkpoint is supported for the convolution weights", wname);
            free(w); return -1;
        }
        own = w; l->m = (QiMat){QI_F32, n, k, w, NULL, 0}; *bytes += (size_t)numel * 4;
    }
    l->own = own;
    l->b = NULL;
    if (bname) {
        int64_t shb[1] = {bias_n ? bias_n : n};
        float *b = acd_read_f32(S, bname, 1, shb, err, errn);
        if (!b) return -1;
        l->b = b;
        *bytes += (size_t)shb[0] * 4;
    }
    return 0;
}

static void acd_free_lin(AcdLin *l){ free(l->own); free(l->b); memset(l, 0, sizeof *l); }

static void acd_free_attn(AcdAttn *a){ acd_free_lin(&a->q); acd_free_lin(&a->k); acd_free_lin(&a->v); acd_free_lin(&a->o); free(a->qn); free(a->kn); memset(a, 0, sizeof *a); }

static void acd_free(AceDit *d){
    if (!d) return;
    acd_free_lin(&d->te.l1); acd_free_lin(&d->te.l2); acd_free_lin(&d->te.proj);
    acd_free_lin(&d->ter.l1); acd_free_lin(&d->ter.l2); acd_free_lin(&d->ter.proj);
    acd_free_lin(&d->cond_emb); acd_free_lin(&d->text_proj); acd_free_lin(&d->lyric_embed); acd_free_lin(&d->timbre_embed);
    acd_free_lin(&d->proj_in); acd_free_lin(&d->proj_out);
    free(d->lyric_norm); free(d->timbre_norm); free(d->norm_out); free(d->sst_out); free(d->sliding);
    for (int i = 0; d->lyric && i < d->nlyric; i++) { AcdEncLayer *e = &d->lyric[i]; acd_free_attn(&e->sa); free(e->ln1); free(e->ln2); acd_free_lin(&e->gate); acd_free_lin(&e->up); acd_free_lin(&e->down); }
    for (int i = 0; d->timbre && i < d->ntimbre; i++) { AcdEncLayer *e = &d->timbre[i]; acd_free_attn(&e->sa); free(e->ln1); free(e->ln2); acd_free_lin(&e->gate); acd_free_lin(&e->up); acd_free_lin(&e->down); }
    for (int i = 0; d->L && i < d->nl; i++) {
        AcdLayer *l = &d->L[i];
        acd_free_attn(&l->sa); acd_free_attn(&l->ca); free(l->n_sa); free(l->n_ca); free(l->n_mlp); free(l->sst);
        acd_free_lin(&l->gate); acd_free_lin(&l->up); acd_free_lin(&l->down);
    }
    free(d->lyric); free(d->timbre); free(d->L);
    free(d);
}

/* convolution taps to columns: raw Conv1d weight [o][c][j] (stride = kernel = P) becomes W[o][j*C + c] */
static void acd_perm_conv(const uint16_t *raw, uint16_t *out, int n, int k, int aux){
    const int P = aux, C = k / P;
    for (int o = 0; o < n; o++)
        for (int c = 0; c < C; c++)
            for (int j = 0; j < P; j++) out[(size_t)o * k + (size_t)j * C + c] = raw[((size_t)o * C + c) * P + j];
}
/* transposed convolution: raw ConvTranspose1d weight [i][c][j] becomes W[j*C + c][i] (n = P*C rows, k = I columns) */
static void acd_perm_convt(const uint16_t *raw, uint16_t *out, int n, int k, int aux){
    const int P = aux, C = n / P;
    for (int i = 0; i < k; i++)
        for (int c = 0; c < C; c++)
            for (int j = 0; j < P; j++) out[((size_t)j * C + c) * k + i] = raw[((size_t)i * C + c) * P + j];
}

static int acd_json_int(jval *cfg, const char *key, int def){
    jval *j = json_get(cfg, key);
    return (j && j->t == J_NUM) ? (int)j->num : def;
}
static double acd_json_num(jval *cfg, const char *key, double def){
    jval *j = json_get(cfg, key);
    return (j && j->t == J_NUM) ? j->num : def;
}

static int acd_load_attn(shards *S, const char *pfx, int D, int heads, int kv, int hd, AcdAttn *a, size_t *bytes, char *err, size_t errn){
    char n[256];
    int64_t shq[2] = {(int64_t)heads * hd, D}, shk[2] = {(int64_t)kv * hd, D}, sho[2] = {D, (int64_t)heads * hd}, shn[1] = {hd};
    snprintf(n, sizeof n, "%s.q_proj.weight", pfx); if (acd_load_lin(S, n, NULL, heads * hd, D, 2, shq, NULL, 0, 0, &a->q, bytes, err, errn)) return -1;
    snprintf(n, sizeof n, "%s.k_proj.weight", pfx); if (acd_load_lin(S, n, NULL, kv * hd, D, 2, shk, NULL, 0, 0, &a->k, bytes, err, errn)) return -1;
    snprintf(n, sizeof n, "%s.v_proj.weight", pfx); if (acd_load_lin(S, n, NULL, kv * hd, D, 2, shk, NULL, 0, 0, &a->v, bytes, err, errn)) return -1;
    snprintf(n, sizeof n, "%s.o_proj.weight", pfx); if (acd_load_lin(S, n, NULL, D, heads * hd, 2, sho, NULL, 0, 0, &a->o, bytes, err, errn)) return -1;
    snprintf(n, sizeof n, "%s.q_norm.weight", pfx); if (!(a->qn = acd_read_f32(S, n, 1, shn, err, errn))) return -1;
    snprintf(n, sizeof n, "%s.k_norm.weight", pfx); if (!(a->kn = acd_read_f32(S, n, 1, shn, err, errn))) return -1;
    return 0;
}

static int acd_load_mlp(shards *S, const char *pfx, int D, int inter, AcdLin *gate, AcdLin *up, AcdLin *down, size_t *bytes, char *err, size_t errn){
    char n[256];
    int64_t shu[2] = {inter, D}, shd[2] = {D, inter};
    snprintf(n, sizeof n, "%s.gate_proj.weight", pfx); if (acd_load_lin(S, n, NULL, inter, D, 2, shu, NULL, 0, 0, gate, bytes, err, errn)) return -1;
    snprintf(n, sizeof n, "%s.up_proj.weight", pfx);   if (acd_load_lin(S, n, NULL, inter, D, 2, shu, NULL, 0, 0, up, bytes, err, errn)) return -1;
    snprintf(n, sizeof n, "%s.down_proj.weight", pfx); if (acd_load_lin(S, n, NULL, D, inter, 2, shd, NULL, 0, 0, down, bytes, err, errn)) return -1;
    return 0;
}

static int acd_load_enc_layers(shards *S, const char *pfx, int count, const AceDit *d, AcdEncLayer **out, size_t *bytes, char *err, size_t errn){
    AcdEncLayer *L = (AcdEncLayer *)calloc((size_t)count, sizeof *L);
    if (!L) { snprintf(err, errn, "out of memory"); return -1; }
    *out = L;                                        /* freed by acd_free even on a partial load */
    for (int i = 0; i < count; i++) {
        char p[256], n[256]; int64_t shd[1] = {d->D};
        L[i].sliding = d->sliding[i];
        snprintf(p, sizeof p, "%s.layers.%d.self_attn", pfx, i);
        if (acd_load_attn(S, p, d->D, d->heads, d->kv, d->hd, &L[i].sa, bytes, err, errn)) return -1;
        snprintf(n, sizeof n, "%s.layers.%d.input_layernorm.weight", pfx, i);          if (!(L[i].ln1 = acd_read_f32(S, n, 1, shd, err, errn))) return -1;
        snprintf(n, sizeof n, "%s.layers.%d.post_attention_layernorm.weight", pfx, i); if (!(L[i].ln2 = acd_read_f32(S, n, 1, shd, err, errn))) return -1;
        snprintf(p, sizeof p, "%s.layers.%d.mlp", pfx, i);
        if (acd_load_mlp(S, p, d->D, d->inter, &L[i].gate, &L[i].up, &L[i].down, bytes, err, errn)) return -1;
    }
    return 0;
}

static int acd_load_timeemb(shards *S, const char *pfx, int D, AcdTimeEmb *t, size_t *bytes, char *err, size_t errn){
    char w[256], b[256];
    int64_t s1[2] = {D, 256}, s2[2] = {D, D}, s3[2] = {6 * (int64_t)D, D};
    snprintf(w, sizeof w, "%s.linear_1.weight", pfx); snprintf(b, sizeof b, "%s.linear_1.bias", pfx);
    if (acd_load_lin(S, w, b, D, 256, 2, s1, NULL, 0, 0, &t->l1, bytes, err, errn)) return -1;
    snprintf(w, sizeof w, "%s.linear_2.weight", pfx); snprintf(b, sizeof b, "%s.linear_2.bias", pfx);
    if (acd_load_lin(S, w, b, D, D, 2, s2, NULL, 0, 0, &t->l2, bytes, err, errn)) return -1;
    snprintf(w, sizeof w, "%s.time_proj.weight", pfx); snprintf(b, sizeof b, "%s.time_proj.bias", pfx);
    if (acd_load_lin(S, w, b, 6 * D, D, 2, s3, NULL, 0, 0, &t->proj, bytes, err, errn)) return -1;
    return 0;
}

/* model_dir holds acestep-v15-turbo/ (config.json and model.safetensors). */
static AceDit *acd_load(const char *model_dir){
    char err[512] = "", path[4200], dir[4096];
    AceDit *d = (AceDit *)calloc(1, sizeof(AceDit));
    shards S; int have_s = 0;
    jval *cfg = NULL; char *text = NULL;
    if (!d) { fprintf(stderr, "acestep dit: out of memory\n"); return NULL; }
    if (snprintf(dir, sizeof dir, "%s/acestep-v15-turbo", model_dir) >= (int)sizeof dir) { snprintf(err, sizeof err, "path too long"); goto fail; }
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, sizeof err, "cannot open %.400s", path); goto fail; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    text = (char *)malloc(n > 0 ? n + 1 : 1);
    if (!text || n <= 0 || n > (8 << 20) || fread(text, 1, n, f) != (size_t)n) { fclose(f); snprintf(err, sizeof err, "cannot read %.400s", path); goto fail; }
    fclose(f); text[n] = 0;
    cfg = json_parse_checked(text);
    if (!cfg || cfg->t != J_OBJ) { snprintf(err, sizeof err, "%.400s is not a JSON object", path); goto fail; }

    d->D = acd_json_int(cfg, "hidden_size", 0);        d->inter = acd_json_int(cfg, "intermediate_size", 0);
    d->nl = acd_json_int(cfg, "num_hidden_layers", 0); d->heads = acd_json_int(cfg, "num_attention_heads", 0);
    d->kv = acd_json_int(cfg, "num_key_value_heads", 0); d->hd = acd_json_int(cfg, "head_dim", 0);
    d->patch = acd_json_int(cfg, "patch_size", 0);     d->inch = acd_json_int(cfg, "in_channels", 0);
    d->acdim = acd_json_int(cfg, "audio_acoustic_hidden_dim", 0);
    d->textdim = acd_json_int(cfg, "text_hidden_dim", 0); d->timbredim = acd_json_int(cfg, "timbre_hidden_dim", 0);
    d->nlyric = acd_json_int(cfg, "num_lyric_encoder_hidden_layers", 0);
    d->ntimbre = acd_json_int(cfg, "num_timbre_encoder_hidden_layers", 0);
    d->tfix = acd_json_int(cfg, "timbre_fix_frame", 0);
    d->eps = (float)acd_json_num(cfg, "rms_norm_eps", 1e-6);
    d->theta = (float)acd_json_num(cfg, "rope_theta", 10000.0);
    jval *sw = json_get(cfg, "use_sliding_window");
    d->window = (sw && sw->t == J_BOOL && !sw->boolean) ? -1 : acd_json_int(cfg, "sliding_window", -1);
    jval *rs = json_get(cfg, "rope_scaling");
    if (rs && rs->t != J_NULL) { snprintf(err, sizeof err, "config: rope_scaling is not supported"); goto fail; }
    jval *ab = json_get(cfg, "attention_bias");
    if (ab && ab->t == J_BOOL && ab->boolean) { snprintf(err, sizeof err, "config: attention_bias is not supported"); goto fail; }
    if (d->D < 1 || d->D > ACD_MAX_DIM || d->inter < 1 || d->inter > 4 * ACD_MAX_DIM || d->nl < 1 || d->nl > ACD_MAX_LAYERS ||
        d->heads < 1 || d->heads > ACD_MAX_HEADS || d->kv < 1 || d->heads % d->kv || d->hd < 2 || d->hd > ACD_MAX_HD || (d->hd & 1) ||
        d->patch < 1 || d->patch > ACD_MAX_PATCH || d->acdim < 1 || d->acdim > ACD_MAX_DIM || d->textdim < 1 || d->textdim > ACD_MAX_DIM ||
        d->timbredim < 1 || d->timbredim > ACD_MAX_DIM || d->nlyric < 1 || d->nlyric > ACD_MAX_LAYERS || d->ntimbre < 1 || d->ntimbre > ACD_MAX_LAYERS ||
        (int64_t)d->heads * d->hd > ACD_MAX_DIM * 2) {
        snprintf(err, sizeof err, "config: a size is missing or out of range"); goto fail; }
    if (d->inch != 3 * d->acdim) { snprintf(err, sizeof err, "config: in_channels %d is not 3 x audio_acoustic_hidden_dim %d (latents, source latents, mask)", d->inch, d->acdim); goto fail; }
    if (d->timbredim != d->acdim) { snprintf(err, sizeof err, "config: timbre_hidden_dim %d is not audio_acoustic_hidden_dim %d", d->timbredim, d->acdim); goto fail; }
    int need = d->nl;
    if (d->nlyric > need) need = d->nlyric;
    if (d->ntimbre > need) need = d->ntimbre;
    jval *lt = json_get(cfg, "layer_types");
    d->sliding = (int *)calloc((size_t)need, sizeof(int));
    if (!d->sliding) { snprintf(err, sizeof err, "out of memory"); goto fail; }
    for (int i = 0; i < need; i++) {
        if (lt && lt->t == J_ARR && i < lt->len && lt->kids[i]->t == J_STR) {
            if (!strcmp(lt->kids[i]->str, "sliding_attention")) d->sliding[i] = d->window > 0;
            else if (strcmp(lt->kids[i]->str, "full_attention")) { snprintf(err, sizeof err, "config: layer_types[%d] is %.40s", i, lt->kids[i]->str); goto fail; }
        } else { snprintf(err, sizeof err, "config: layer_types has %d entries, %d layers need one", lt && lt->t == J_ARR ? lt->len : 0, need); goto fail; }
    }

    st_init(&S, dir); have_s = 1;
    if (S.n == 0) { snprintf(err, sizeof err, "no safetensors in %.400s", dir); goto fail; }
    const int D = d->D;
    int64_t shD[1] = {D}, shsst2[3] = {1, 2, D}, shsst6[3] = {1, 6, D};

    if (acd_load_timeemb(&S, "decoder.time_embed", D, &d->te, &d->wbytes, err, sizeof err)) goto fail;
    if (acd_load_timeemb(&S, "decoder.time_embed_r", D, &d->ter, &d->wbytes, err, sizeof err)) goto fail;
    { int64_t s[2] = {D, D};
      if (acd_load_lin(&S, "decoder.condition_embedder.weight", "decoder.condition_embedder.bias", D, D, 2, s, NULL, 0, 0, &d->cond_emb, &d->wbytes, err, sizeof err)) goto fail; }
    { int64_t s[3] = {D, d->inch, d->patch};
      if (acd_load_lin(&S, "decoder.proj_in.1.weight", "decoder.proj_in.1.bias", D, d->patch * d->inch, 3, s, acd_perm_conv, d->patch, 0, &d->proj_in, &d->wbytes, err, sizeof err)) goto fail; }
    { int64_t s[3] = {D, d->acdim, d->patch};
      if (acd_load_lin(&S, "decoder.proj_out.1.weight", NULL, d->patch * d->acdim, D, 3, s, acd_perm_convt, d->patch, 0, &d->proj_out, &d->wbytes, err, sizeof err)) goto fail;
      int64_t sb[1] = {d->acdim};
      float *b = acd_read_f32(&S, "decoder.proj_out.1.bias", 1, sb, err, sizeof err);
      if (!b) goto fail;
      d->proj_out.b = (float *)malloc((size_t)d->patch * d->acdim * sizeof(float));      /* the bias repeats for every tap */
      if (!d->proj_out.b) { free(b); snprintf(err, sizeof err, "out of memory"); goto fail; }
      for (int j = 0; j < d->patch; j++) memcpy(d->proj_out.b + (size_t)j * d->acdim, b, (size_t)d->acdim * sizeof(float));
      free(b); }
    if (!(d->norm_out = acd_read_f32(&S, "decoder.norm_out.weight", 1, shD, err, sizeof err))) goto fail;
    if (!(d->sst_out = acd_read_f32(&S, "decoder.scale_shift_table", 3, shsst2, err, sizeof err))) goto fail;
    { int64_t s[2] = {D, d->textdim};
      if (acd_load_lin(&S, "encoder.text_projector.weight", NULL, D, d->textdim, 2, s, NULL, 0, 0, &d->text_proj, &d->wbytes, err, sizeof err)) goto fail; }
    { int64_t s[2] = {D, d->textdim};
      if (acd_load_lin(&S, "encoder.lyric_encoder.embed_tokens.weight", "encoder.lyric_encoder.embed_tokens.bias", D, d->textdim, 2, s, NULL, 0, 0, &d->lyric_embed, &d->wbytes, err, sizeof err)) goto fail; }
    { int64_t s[2] = {D, d->timbredim};
      if (acd_load_lin(&S, "encoder.timbre_encoder.embed_tokens.weight", "encoder.timbre_encoder.embed_tokens.bias", D, d->timbredim, 2, s, NULL, 0, 0, &d->timbre_embed, &d->wbytes, err, sizeof err)) goto fail; }
    if (!(d->lyric_norm = acd_read_f32(&S, "encoder.lyric_encoder.norm.weight", 1, shD, err, sizeof err))) goto fail;
    if (!(d->timbre_norm = acd_read_f32(&S, "encoder.timbre_encoder.norm.weight", 1, shD, err, sizeof err))) goto fail;
    if (acd_load_enc_layers(&S, "encoder.lyric_encoder", d->nlyric, d, &d->lyric, &d->wbytes, err, sizeof err)) goto fail;
    if (acd_load_enc_layers(&S, "encoder.timbre_encoder", d->ntimbre, d, &d->timbre, &d->wbytes, err, sizeof err)) goto fail;

    d->L = (AcdLayer *)calloc((size_t)d->nl, sizeof(AcdLayer));
    if (!d->L) { snprintf(err, sizeof err, "out of memory"); goto fail; }
    for (int i = 0; i < d->nl; i++) {
        AcdLayer *l = &d->L[i]; char p[256], nm[256];
        l->sliding = d->sliding[i];
        snprintf(p, sizeof p, "decoder.layers.%d.self_attn", i);  if (acd_load_attn(&S, p, D, d->heads, d->kv, d->hd, &l->sa, &d->wbytes, err, sizeof err)) goto fail;
        snprintf(p, sizeof p, "decoder.layers.%d.cross_attn", i); if (acd_load_attn(&S, p, D, d->heads, d->kv, d->hd, &l->ca, &d->wbytes, err, sizeof err)) goto fail;
        snprintf(nm, sizeof nm, "decoder.layers.%d.self_attn_norm.weight", i);  if (!(l->n_sa = acd_read_f32(&S, nm, 1, shD, err, sizeof err))) goto fail;
        snprintf(nm, sizeof nm, "decoder.layers.%d.cross_attn_norm.weight", i); if (!(l->n_ca = acd_read_f32(&S, nm, 1, shD, err, sizeof err))) goto fail;
        snprintf(nm, sizeof nm, "decoder.layers.%d.mlp_norm.weight", i);        if (!(l->n_mlp = acd_read_f32(&S, nm, 1, shD, err, sizeof err))) goto fail;
        snprintf(nm, sizeof nm, "decoder.layers.%d.scale_shift_table", i);      if (!(l->sst = acd_read_f32(&S, nm, 3, shsst6, err, sizeof err))) goto fail;
        snprintf(p, sizeof p, "decoder.layers.%d.mlp", i);
        if (acd_load_mlp(&S, p, D, d->inter, &l->gate, &l->up, &l->down, &d->wbytes, err, sizeof err)) goto fail;
    }
    st_destroy(&S);
    json_free(cfg); free(text);
    return d;
fail:
    fprintf(stderr, "acestep dit: %s\n", err[0] ? err : "load failed");
    if (have_s) st_destroy(&S);
    json_free(cfg); free(text);
    acd_free(d);
    return NULL;
}

/* -------------------------------------------------------------------- math -- */

static inline float acd_silu(float x){ return x / (1.f + expf(-x)); }

static void acd_lin(float *Y, const float *X, int M, const AcdLin *l){ qi_gemm(Y, X, M, &l->m, l->b); }

/* y = x * rsqrt(mean(x^2) + eps) * w, rows of n; dst may equal src. */
static void acd_rmsnorm(float *dst, const float *src, int rows, int n, const float *w, float eps){
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *x = src + (size_t)r * n; float *y = dst + (size_t)r * n;
        double s = 0;
        for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
        float inv = 1.f / sqrtf((float)(s / n) + eps);
        for (int i = 0; i < n; i++) y[i] = w[i] * (x[i] * inv);
    }
}

/* the same per head, in place: x [rows][heads*hd], one weight [hd] shared by every head */
static void acd_head_norm(float *x, int rows, int heads, int hd, const float *w, float eps){
    acd_rmsnorm(x, x, rows * heads, hd, w, eps);
}

/* RoPE tables for positions 0..n-1: cos/sin [n][hd], the half-width angles repeated as Qwen3 does. float32 throughout,
 * as the reference computes them (an angle p * inv_freq rounded to float, then cos and sin). */
static void acd_rope_table(float *cs, float *sn, int n, int hd, float theta){
    const int h2 = hd / 2;
    float *inv = (float *)malloc((size_t)h2 * sizeof(float));
    for (int i = 0; i < h2; i++) inv[i] = 1.0f / powf(theta, (float)(2 * i) / (float)hd);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < n; p++)
        for (int i = 0; i < h2; i++) {
            float a = (float)p * inv[i];
            float c = cosf(a), s = sinf(a);
            cs[(size_t)p * hd + i] = c; cs[(size_t)p * hd + i + h2] = c;
            sn[(size_t)p * hd + i] = s; sn[(size_t)p * hd + i + h2] = s;
        }
    free(inv);
}

/* x [rows][heads*hd] in place: x*cos + rotate_half(x)*sin, rotate_half(x) = [-x2, x1] */
static void acd_rope(float *x, int rows, int heads, int hd, const float *cs, const float *sn){
    const int h2 = hd / 2;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *c = cs + (size_t)r * hd, *s = sn + (size_t)r * hd;
        for (int h = 0; h < heads; h++) {
            float *v = x + (size_t)r * heads * hd + (size_t)h * hd;
            for (int i = 0; i < h2; i++) {
                float x1 = v[i], x2 = v[i + h2];
                v[i] = x1 * c[i] - x2 * s[i];
                v[i + h2] = x2 * c[i + h2] + x1 * s[i + h2];
            }
        }
    }
}

/* out[Tq][heads*hd] = softmax(Q K^T / sqrt(hd)) V. Q rows have stride ldq, K and V stride ldk; query head h reads kv head
 * h / (heads/kvheads). window < 0: every key; else key j is visible to query i when |i-j| <= window. causal: only keys
 * j <= i (the text encoder; the DiT and its condition encoders are bidirectional). Done in blocks of ACD_QB queries: a
 * block only multiplies against the keys its band can reach. */
static void acd_attention_ex(float *out, int ldo, const float *Q, int ldq, int Tq, const float *K, const float *V, int ldk, int Tk,
                             int heads, int kvheads, int hd, int window, int causal){
    const float scale = 1.f / sqrtf((float)hd);
    const int group = heads / kvheads;
    int maxnk = Tk;
    if (window >= 0 && ACD_QB + 2 * window + 1 < maxnk) maxnk = ACD_QB + 2 * window + 1;
    float *S = (float *)malloc((size_t)ACD_QB * maxnk * sizeof(float));
    float *vt = (float *)malloc((size_t)hd * maxnk * sizeof(float));
    if (!S || !vt) { fprintf(stderr, "acestep dit: out of memory in attention\n"); exit(1); }
    for (int q0 = 0; q0 < Tq; q0 += ACD_QB) {
        const int rows = Tq - q0 < ACD_QB ? Tq - q0 : ACD_QB;
        int lo = 0, hi = Tk;
        if (window >= 0) { lo = q0 - window > 0 ? q0 - window : 0; hi = q0 + rows - 1 + window + 1 < Tk ? q0 + rows + window : Tk; }
        if (causal && hi > q0 + rows) hi = q0 + rows;                          /* no query of this block sees past its own row */
        const int nk = hi - lo;
        if (nk <= 0) { for (int i = 0; i < rows; i++) memset(out + (size_t)(q0 + i) * ldo, 0, (size_t)heads * hd * sizeof(float)); continue; }
        for (int kh = 0; kh < kvheads; kh++) {
            #pragma omp parallel for schedule(static)
            for (int dd = 0; dd < hd; dd++)
                for (int j = 0; j < nk; j++) vt[(size_t)dd * nk + j] = V[(size_t)(lo + j) * ldk + (size_t)kh * hd + dd];
            for (int g = 0; g < group; g++) {
                const int h = kh * group + g;
                QiMat Km = { QI_F32, nk, hd, K + (size_t)lo * ldk + (size_t)kh * hd, NULL, ldk };
                qi_gemm_ld(S, nk, Q + (size_t)q0 * ldq + (size_t)h * hd, ldq, rows, &Km, NULL);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < rows; i++) {
                    float *r = S + (size_t)i * nk;
                    int gi = q0 + i, a = 0, b = nk;
                    if (window >= 0) { a = gi - window - lo > 0 ? gi - window - lo : 0; b = gi + window + 1 - lo < nk ? gi + window + 1 - lo : nk; }
                    if (causal && b > gi + 1 - lo) b = gi + 1 - lo;
                    float mx = -INFINITY;
                    for (int j = a; j < b; j++) { r[j] *= scale; if (r[j] > mx) mx = r[j]; }
                    double sum = 0;
                    for (int j = a; j < b; j++) { r[j] = expf(r[j] - mx); sum += r[j]; }
                    float inv = (float)(1.0 / sum);
                    for (int j = a; j < b; j++) r[j] *= inv;
                    for (int j = 0; j < a; j++) r[j] = 0.f;
                    for (int j = b; j < nk; j++) r[j] = 0.f;
                }
                QiMat Vm = { QI_F32, hd, nk, vt, NULL, 0 };
                qi_gemm_ld(out + (size_t)q0 * ldo + (size_t)h * hd, ldo, S, nk, rows, &Vm, NULL);
            }
        }
    }
    free(S); free(vt);
}

/* bidirectional: the DiT, its condition encoders, and its cross-attention */
static void acd_attention(float *out, int ldo, const float *Q, int ldq, int Tq, const float *K, const float *V, int ldk, int Tk,
                          int heads, int kvheads, int hd, int window){
    acd_attention_ex(out, ldo, Q, ldq, Tq, K, V, ldk, Tk, heads, kvheads, hd, window, 0);
}

/* A debugging tap: when set, acd_forward calls it with the hidden state [Tp][D] after the patch embedding (layer -1),
 * after each layer (layer = its index), and inside each layer after the self-attention residual (1000 + 2 layer) and after
 * the cross-attention residual (1001 + 2 layer), so an oracle can say which layer, and which operation, a discrepancy
 * first appears in. For the layer in acd_dbg_layer the attention's own stages are tapped too (2000 + stage: queries and
 * keys after the norm and RoPE, values, the attention output before o_proj, and o_proj's output). Not used in production. */
typedef void (*AcdTapFn)(int layer, const float *h, int Tp, int D, void *ud);
static AcdTapFn acd_tap_fn = NULL;
static void *acd_tap_ud = NULL;
static int acd_dbg_layer = -1, acd_dbg_active = 0;

/* One self-attention sub-layer on x [T][D] (already normalised and modulated): projections, per-head norms, RoPE,
 * attention, output projection into o [T][D]. */
static void acd_self_attn(const AceDit *d, const AcdAttn *a, const float *x, int T, int sliding, const float *cs, const float *sn, float *o){
    const int H = d->heads, Hk = d->kv, hd = d->hd;
    float *q = (float *)malloc((size_t)T * H * hd * sizeof(float));
    float *k = (float *)malloc((size_t)T * Hk * hd * sizeof(float));
    float *v = (float *)malloc((size_t)T * Hk * hd * sizeof(float));
    float *ctx = (float *)malloc((size_t)T * H * hd * sizeof(float));
    if (!q || !k || !v || !ctx) { fprintf(stderr, "acestep dit: out of memory in attention\n"); exit(1); }
    acd_lin(q, x, T, &a->q); acd_lin(k, x, T, &a->k); acd_lin(v, x, T, &a->v);
    acd_head_norm(q, T, H, hd, a->qn, d->eps);
    acd_head_norm(k, T, Hk, hd, a->kn, d->eps);
    acd_rope(q, T, H, hd, cs, sn);
    acd_rope(k, T, Hk, hd, cs, sn);
    if (acd_dbg_active && acd_tap_fn) { acd_tap_fn(2000, q, T, H * hd, acd_tap_ud); acd_tap_fn(2001, k, T, Hk * hd, acd_tap_ud); acd_tap_fn(2002, v, T, Hk * hd, acd_tap_ud); }
    acd_attention(ctx, H * hd, q, H * hd, T, k, v, Hk * hd, T, H, Hk, hd, sliding ? d->window : -1);
    if (acd_dbg_active && acd_tap_fn) acd_tap_fn(2003, ctx, T, H * hd, acd_tap_ud);
    acd_lin(o, ctx, T, &a->o);
    if (acd_dbg_active && acd_tap_fn) acd_tap_fn(2004, o, T, d->D, acd_tap_ud);
    free(q); free(k); free(v); free(ctx);
}

/* x += attn(norm1(x)); x += mlp(norm2(x)) for the encoders' layers (no modulation) */
static void acd_enc_layer(const AceDit *d, const AcdEncLayer *l, float *x, int T, const float *cs, const float *sn){
    const int D = d->D;
    float *n = (float *)malloc((size_t)T * D * sizeof(float)), *o = (float *)malloc((size_t)T * D * sizeof(float));
    float *g = (float *)malloc((size_t)T * d->inter * sizeof(float)), *u = (float *)malloc((size_t)T * d->inter * sizeof(float));
    if (!n || !o || !g || !u) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    acd_rmsnorm(n, x, T, D, l->ln1, d->eps);
    acd_self_attn(d, &l->sa, n, T, l->sliding, cs, sn, o);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * D; i++) x[i] += o[i];
    acd_rmsnorm(n, x, T, D, l->ln2, d->eps);
    acd_lin(g, n, T, &l->gate); acd_lin(u, n, T, &l->up);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * d->inter; i++) g[i] = acd_silu(g[i]) * u[i];
    acd_lin(o, g, T, &l->down);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * D; i++) x[i] += o[i];
    free(n); free(o); free(g); free(u);
}

/* tokens [T][in] -> embed -> layers -> norm; returns [T][D] (malloc) */
static float *acd_encoder(const AceDit *d, const float *tokens, int T, const AcdLin *embed, const AcdEncLayer *layers, int nlayers, const float *norm){
    float *x = (float *)malloc((size_t)T * d->D * sizeof(float));
    float *cs = (float *)malloc((size_t)T * d->hd * sizeof(float)), *sn = (float *)malloc((size_t)T * d->hd * sizeof(float));
    if (!x || !cs || !sn) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    acd_lin(x, tokens, T, embed);
    acd_rope_table(cs, sn, T, d->hd, d->theta);
    for (int i = 0; i < nlayers; i++) acd_enc_layer(d, &layers[i], x, T, cs, sn);
    acd_rmsnorm(x, x, T, d->D, norm, d->eps);
    free(cs); free(sn);
    return x;
}

/* The condition of one request. text [Lt][textdim] (Qwen3-Embedding's last hidden state), lyric [Ll][textdim] (its
 * embedding-table rows), timbre [Tr][timbredim] (the reference audio's latents; the silence latent for plain text-to-music).
 * Packs lyric, timbre, text (one timbre token) and builds every layer's cross K/V. Returns NULL on failure. */
static void acd_cond_free(AcdCond *c){
    if (!c) return;
    for (int i = 0; c->ck && i < c->nl; i++) { free(c->ck[i]); free(c->cv[i]); }
    free(c->ck); free(c->cv); free(c->enc); free(c);
}

static AcdCond *acd_cond_new(const AceDit *d, const float *text, int Lt, const float *lyric, int Ll, const float *timbre, int Tr){
    if (Lt < 1 || Ll < 1 || Tr < 1) return NULL;
    const int D = d->D, Hk = d->kv, hd = d->hd;
    AcdCond *c = (AcdCond *)calloc(1, sizeof *c);
    if (!c) return NULL;
    c->nl = d->nl; c->len = Ll + 1 + Lt;
    float *ly = acd_encoder(d, lyric, Ll, &d->lyric_embed, d->lyric, d->nlyric, d->lyric_norm);
    float *tb = acd_encoder(d, timbre, Tr, &d->timbre_embed, d->timbre, d->ntimbre, d->timbre_norm);
    float *raw = (float *)malloc((size_t)c->len * D * sizeof(float));
    if (!raw) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    memcpy(raw, ly, (size_t)Ll * D * sizeof(float));
    memcpy(raw + (size_t)Ll * D, tb, (size_t)D * sizeof(float));                     /* the first frame's output is the timbre */
    acd_lin(raw + (size_t)(Ll + 1) * D, text, Lt, &d->text_proj);
    free(ly); free(tb);
    c->enc = raw;                                                                     /* the packed tokens, before condition_embedder */
    float *emb = (float *)malloc((size_t)c->len * D * sizeof(float));
    if (!emb) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    acd_lin(emb, raw, c->len, &d->cond_emb);
    c->ck = (float **)calloc((size_t)d->nl, sizeof(float *)); c->cv = (float **)calloc((size_t)d->nl, sizeof(float *));
    if (!c->ck || !c->cv) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    for (int i = 0; i < d->nl; i++) {
        c->ck[i] = (float *)malloc((size_t)c->len * Hk * hd * sizeof(float));
        c->cv[i] = (float *)malloc((size_t)c->len * Hk * hd * sizeof(float));
        if (!c->ck[i] || !c->cv[i]) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
        acd_lin(c->ck[i], emb, c->len, &d->L[i].ca.k);
        acd_head_norm(c->ck[i], c->len, Hk, hd, d->L[i].ca.kn, d->eps);
        acd_lin(c->cv[i], emb, c->len, &d->L[i].ca.v);
    }
    free(emb);
    return c;
}

/* sinusoidal embedding of t scaled by 1000 (cos first, then sin), in float32 as the reference does it, then the MLP:
 * temb = linear_2(silu(linear_1(e))), proj = time_proj(silu(temb)) as [6][D]. */
static void acd_time_embed(const AceDit *d, const AcdTimeEmb *te, float t, float *temb, float *proj){
    float e[256];
    const int half = 128;
    float ts = t * 1000.0f;
    for (int i = 0; i < half; i++) {
        float x = (float)(-9.210340371976184) * (float)i / (float)half;      /* float32 arithmetic, as the reference's */
        float f = (float)exp((double)x);
        float a = ts * f;
        e[i] = (float)cos((double)a); e[half + i] = (float)sin((double)a);
    }
    float *h = (float *)malloc((size_t)d->D * sizeof(float));
    if (!h) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    acd_lin(h, e, 1, &te->l1);
    for (int i = 0; i < d->D; i++) h[i] = acd_silu(h[i]);
    acd_lin(temb, h, 1, &te->l2);
    for (int i = 0; i < d->D; i++) h[i] = acd_silu(temb[i]);
    acd_lin(proj, h, 1, &te->proj);
    free(h);
}

/* One DiT layer on h [Tp][D], in place: modulated gated self-attention, cross-attention, modulated gated SwiGLU. proj is the
 * step's timestep modulation [6][D], cs/sn its RoPE table for Tp tokens. Split out of acd_forward so that an oracle can
 * feed a layer the reference's own input and compare its output alone. */
static void acd_layer(const AceDit *d, const AcdCond *c, int li, float *h, int Tp, const float *proj, const float *cs, const float *sn){
    const int D = d->D, Hk = d->kv, hd = d->hd, H = d->heads;
    const AcdLayer *l = &d->L[li];
    float *n = (float *)malloc((size_t)Tp * D * sizeof(float)), *o = (float *)malloc((size_t)Tp * D * sizeof(float));
    float *g = (float *)malloc((size_t)Tp * d->inter * sizeof(float)), *u = (float *)malloc((size_t)Tp * d->inter * sizeof(float));
    float *q = (float *)malloc((size_t)Tp * H * hd * sizeof(float)), *cx = (float *)malloc((size_t)Tp * H * hd * sizeof(float));
    float *sbuf = (float *)malloc((size_t)6 * D * sizeof(float));
    if (!n || !o || !g || !u || !q || !cx || !sbuf) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    const float *ss[6];
    for (int i = 0; i < 6 * D; i++) sbuf[i] = l->sst[i] + proj[i];
    for (int i = 0; i < 6; i++) ss[i] = sbuf + (size_t)i * D;
    const float *shift_a = ss[0], *scale_a = ss[1], *gate_a = ss[2], *shift_m = ss[3], *scale_m = ss[4], *gate_m = ss[5];
    /* 1. self-attention, modulated, gated */
    acd_rmsnorm(n, h, Tp, D, l->n_sa, d->eps);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < Tp; p++) for (int i = 0; i < D; i++) n[(size_t)p * D + i] = n[(size_t)p * D + i] * (1.f + scale_a[i]) + shift_a[i];
    acd_dbg_active = li == acd_dbg_layer;
    acd_self_attn(d, &l->sa, n, Tp, l->sliding, cs, sn, o);
    acd_dbg_active = 0;
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < Tp; p++) for (int i = 0; i < D; i++) h[(size_t)p * D + i] += o[(size_t)p * D + i] * gate_a[i];
    if (acd_tap_fn) acd_tap_fn(1000 + 2 * li, h, Tp, D, acd_tap_ud);               /* after the self-attention residual */
    /* 2. cross-attention to the condition: queries only, keys and values cached; no RoPE, no mask, no gate */
    acd_rmsnorm(n, h, Tp, D, l->n_ca, d->eps);
    acd_lin(q, n, Tp, &l->ca.q);
    acd_head_norm(q, Tp, H, hd, l->ca.qn, d->eps);
    acd_attention(cx, H * hd, q, H * hd, Tp, c->ck[li], c->cv[li], Hk * hd, c->len, H, Hk, hd, -1);
    acd_lin(o, cx, Tp, &l->ca.o);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)Tp * D; i++) h[i] += o[i];
    if (acd_tap_fn) acd_tap_fn(1001 + 2 * li, h, Tp, D, acd_tap_ud);               /* after the cross-attention residual */
    /* 3. SwiGLU, modulated, gated */
    acd_rmsnorm(n, h, Tp, D, l->n_mlp, d->eps);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < Tp; p++) for (int i = 0; i < D; i++) n[(size_t)p * D + i] = n[(size_t)p * D + i] * (1.f + scale_m[i]) + shift_m[i];
    acd_lin(g, n, Tp, &l->gate); acd_lin(u, n, Tp, &l->up);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)Tp * d->inter; i++) g[i] = acd_silu(g[i]) * u[i];
    acd_lin(o, g, Tp, &l->down);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < Tp; p++) for (int i = 0; i < D; i++) h[(size_t)p * D + i] += o[(size_t)p * D + i] * gate_m[i];
    free(n); free(o); free(g); free(u); free(q); free(cx); free(sbuf);
}

/* One DiT evaluation: xt [T][acdim], ctx [T][2*acdim] (source latents ++ mask), timesteps t and r -> v [T][acdim]. */
static void acd_forward(const AceDit *d, const AcdCond *c, const float *xt, const float *ctx, int T, float t, float r, float *v){
    const int D = d->D, P = d->patch, A = d->acdim, Tp = (T + P - 1) / P, hd = d->hd;
    float *temb = (float *)malloc((size_t)D * sizeof(float)), *tr = (float *)malloc((size_t)D * sizeof(float));
    float *proj = (float *)malloc((size_t)6 * D * sizeof(float)), *pr = (float *)malloc((size_t)6 * D * sizeof(float));
    if (!temb || !tr || !proj || !pr) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    acd_time_embed(d, &d->te, t, temb, proj);
    acd_time_embed(d, &d->ter, t - r, tr, pr);
    for (int i = 0; i < D; i++) temb[i] += tr[i];
    for (int i = 0; i < 6 * D; i++) proj[i] += pr[i];

    /* patches: for token p, tap j (frame P*p + j): [context ++ xt] channels, zero past the end of the clip */
    const int K = P * d->inch;
    float *in = (float *)calloc((size_t)Tp * K, sizeof(float));
    float *h = (float *)malloc((size_t)Tp * D * sizeof(float));
    float *n = (float *)malloc((size_t)Tp * D * sizeof(float));
    float *cs = (float *)malloc((size_t)Tp * hd * sizeof(float)), *sn = (float *)malloc((size_t)Tp * hd * sizeof(float));
    if (!in || !h || !n || !cs || !sn) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < Tp; p++)
        for (int j = 0; j < P; j++) {
            int f = P * p + j;
            if (f >= T) continue;
            float *dst = in + (size_t)p * K + (size_t)j * d->inch;
            memcpy(dst, ctx + (size_t)f * 2 * A, (size_t)2 * A * sizeof(float));
            memcpy(dst + 2 * A, xt + (size_t)f * A, (size_t)A * sizeof(float));
        }
    acd_lin(h, in, Tp, &d->proj_in);
    free(in);
    if (acd_tap_fn) acd_tap_fn(-1, h, Tp, D, acd_tap_ud);
    acd_rope_table(cs, sn, Tp, hd, d->theta);

    for (int li = 0; li < d->nl; li++) {
        acd_layer(d, c, li, h, Tp, proj, cs, sn);
        if (acd_tap_fn) acd_tap_fn(li, h, Tp, D, acd_tap_ud);
    }
    /* adaptive output norm: shift and scale are the two rows of the table plus temb */
    float *shift = (float *)malloc((size_t)D * sizeof(float)), *scale = (float *)malloc((size_t)D * sizeof(float));
    if (!shift || !scale) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    for (int i = 0; i < D; i++) { shift[i] = d->sst_out[i] + temb[i]; scale[i] = d->sst_out[D + i] + temb[i]; }
    acd_rmsnorm(n, h, Tp, D, d->norm_out, d->eps);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < Tp; p++) for (int i = 0; i < D; i++) n[(size_t)p * D + i] = n[(size_t)p * D + i] * (1.f + scale[i]) + shift[i];
    /* de-patchify: row p of the product is the P frames P*p .. P*p+P-1, [tap][channel], i.e. rows of [P*Tp][A] */
    float *outp = (float *)malloc((size_t)Tp * P * A * sizeof(float));
    if (!outp) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    acd_lin(outp, n, Tp, &d->proj_out);
    memcpy(v, outp, (size_t)T * A * sizeof(float));
    free(outp); free(shift); free(scale);
    free(temb); free(tr); free(proj); free(pr); free(h); free(n); free(cs); free(sn);
}

/* ----------------------------------------------------------------- sampler -- */

/* The schedule upstream builds when it is given a step count: n evenly spaced from 1, bent by the shift. Computed in
 * double and rounded to float32 once, as the reference's float32 schedule tensor is. n is clamped to 1..20 as there. */
static int acd_schedule(int steps, float shift, float *t){
    int n = steps > 20 ? 20 : steps;
    if (n < 1) return 0;
    for (int i = 0; i < n; i++) {
        double raw = 1.0 - (double)i / n;
        if (shift != 1.0f) raw = (double)shift * raw / (1.0 + ((double)shift - 1.0) * raw);
        t[i] = (float)raw;
    }
    return n;
}

/* DCW, double mode, Haar: see the header comment. x [T][A] is corrected in place from denoised y. */
static void acd_dcw(float *x, const float *y, int T, int A, float lo, float hi){
    if (lo == 0.f && hi == 0.f) return;
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < T; k += 2) {
        float *x0 = x + (size_t)k * A, *x1 = k + 1 < T ? x + (size_t)(k + 1) * A : NULL;
        const float *y0 = y + (size_t)k * A, *y1 = k + 1 < T ? y + (size_t)(k + 1) * A : NULL;
        for (int c = 0; c < A; c++) {
            float e0 = x0[c] - y0[c], e1 = x1 ? x1[c] - y1[c] : 0.f;
            float low = lo * (e0 + e1) / 2.f, high = hi * (e0 - e1) / 2.f;
            x0[c] += low + high;
            if (x1) x1[c] += low - high;
        }
    }
}

typedef struct { int dcw; double dcw_low, dcw_high; } AcdSampleOpts;     /* upstream's defaults: 0.05 and 0.02 */
/* Called after each DiT evaluation with the step, the latents it was given and the velocity: for oracles and progress. */
typedef void (*AcdStepFn)(int step, int steps, const float *xt, const float *v, void *ud);

/* One sampler update. x are the latents the DiT was given at schedule position i of n, v its velocity; out receives the
 * next latents, or on the last position the clean ones (x - v t, straight to x0). With DCW, the result is then corrected
 * from the predicted clean sample x - v t. Split out so that an oracle can give it the handler's own x and v and compare
 * with the handler's next latents: the sampler's logic checked alone, with no network noise amplified through the loop. */
static void acd_sample_step(const float *x, const float *v, int i, int n, const float *sched, int T, int A,
                            const AcdSampleOpts *o, float *out){
    const size_t N = (size_t)T * A;
    const float t = sched[i];
    if (i == n - 1) {
        for (size_t k = 0; k < N; k++) out[k] = x[k] - v[k] * t;                /* straight to x0 */
    } else {
        const float dt = (float)((double)t - (double)sched[i + 1]);
        for (size_t k = 0; k < N; k++) out[k] = x[k] - v[k] * dt;
    }
    if (o && o->dcw) {
        float *den = (float *)malloc(N * sizeof(float));
        if (!den) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
        for (size_t k = 0; k < N; k++) den[k] = x[k] - v[k] * t;                /* the predicted clean sample */
        /* the per-band strengths are Python doubles, t * scaler and (1 - t) * high_scaler, rounded to float32 once */
        acd_dcw(out, den, T, A, (float)((double)t * o->dcw_low), (float)((1.0 - (double)t) * o->dcw_high));
        free(den);
    }
}

/* Turbo sampling from noise [T][A]: Euler steps of the flow with DCW after each, the last step straight to x0.
 * out [T][A]. Returns the number of steps, or 0 on a bad argument. */
static int acd_sample(const AceDit *d, const AcdCond *c, const float *noise, const float *ctx, int T, int steps, float shift,
                      const AcdSampleOpts *o, AcdStepFn on_step, void *ud, float *out){
    const int A = d->acdim;
    float sched[20];
    int n = acd_schedule(steps, shift, sched);
    if (!n || T < 1) return 0;
    const size_t N = (size_t)T * A;
    float *x = (float *)malloc(N * sizeof(float)), *nx = (float *)malloc(N * sizeof(float)), *v = (float *)malloc(N * sizeof(float));
    if (!x || !nx || !v) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    memcpy(x, noise, N * sizeof(float));
    for (int i = 0; i < n; i++) {
        acd_forward(d, c, x, ctx, T, sched[i], sched[i], v);
        if (on_step) on_step(i, n, x, v, ud);
        acd_sample_step(x, v, i, n, sched, T, A, o, nx);
        float *s = x; x = nx; nx = s;
    }
    memcpy(out, x, N * sizeof(float));
    free(x); free(nx); free(v);
    return n;
}

#endif
