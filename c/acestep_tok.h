/* acestep_tok.h -- the ACE-Step 1.5 audio tokenizer and detokenizer: what a COVER does to the source it is given.
 *
 * Upstream's `prepare_condition` does not hand the DiT the source clip's latents for a cover. It squeezes them through a
 * 5 Hz bottleneck and back, and the DiT is conditioned on what comes out (`lm_hints_25Hz`): the clip's musical structure
 * without its exact sound, which is what lets the new version differ from the original. The path, on the source latents
 * x [T][64] at 25 Hz (acestep-v15-turbo's `tokenizer.*` and `detokenizer.*`, bf16 like the DiT):
 *
 *   pad T up to a multiple of P = 5 with the silence latent's first frames
 *   audio_acoustic_proj (64 -> 2048)                      then per window of 5 frames, a sequence of its own:
 *   attention pooler: embed_tokens (2048 -> 2048), a learned special token put FIRST, 2 encoder layers, a norm, and the
 *       special token's output is the window's one 2048-wide vector                                -> [T/5][2048]
 *   residual FSQ, one quantizer: project_in (2048 -> 6), soft clamp tanh(x/c)*c with c = 1 + 1/(L-1) per level,
 *       then per dimension (levels 8,8,8,5,5,5) the symmetric rounding   code = 2/(L-1) floor((L-1)(clamp(z,-1,1)+1)/2 + 1/2) - 1,
 *       project_out (6 -> 2048)                                                                   -> [T/5][2048]
 *   detokenizer: embed_tokens (2048 -> 2048), each vector repeated 5 times plus 5 learned position tokens, 2 encoder
 *       layers (a window of 5 is a sequence of its own), a norm, proj_out (2048 -> 64)             -> [T][64]
 *
 * The encoder layers are the lyric and timbre encoders' (acestep_dit.h's acd_enc_layer, RoPE and RMSNorm and SwiGLU all
 * the same), run here on many short sequences at once: the projections and the MLP on every row, the attention window by
 * window. The 4-D masks upstream builds are, for one clip with nothing masked, the plain full attention (a window of 6
 * tokens is inside the sliding window of 128), which is what is computed.
 *
 * The quantizer's rounding is a step function, so a value within float rounding of a boundary can land on the other side of
 * it in C and in torch: one code of one window then differs by a level. The oracle (acestep.c --cover-ref) separates that
 * from a wrong quantizer by looking at how close to a boundary the value was.
 *
 * Interface: atk_load(d, model_dir) / atk_hints(d, t, x, T, silence, out) / atk_free(t); the stages atk_pool, atk_fsq and
 * atk_detok are exposed so an oracle can hold each to upstream's own input.
 */
#ifndef COLIBRI_ACESTEP_TOK_H
#define COLIBRI_ACESTEP_TOK_H
#include "acestep_dit.h"

#define ATK_MAX_LEVELS 16
#define ATK_MAX_P 32
#define ATK_WB 256                     /* windows per block: bounds the activations of a long clip */

typedef struct {
    int P, npool, nlev, levels[ATK_MAX_LEVELS];
    AcdLin proj, pool_embed, q_in, q_out, d_embed, d_proj;
    float *pool_special;               /* [D] */
    float *d_special;                  /* [P][D] */
    AcdEncLayer *pool, *detok;
    float *pool_norm, *d_norm;
    size_t wbytes;
} AcdTok;

static void atk_free_layers(AcdEncLayer *L, int n){
    for (int i = 0; L && i < n; i++) { AcdEncLayer *e = &L[i]; acd_free_attn(&e->sa); free(e->ln1); free(e->ln2); acd_free_lin(&e->gate); acd_free_lin(&e->up); acd_free_lin(&e->down); }
    free(L);
}

static void atk_free(AcdTok *t){
    if (!t) return;
    acd_free_lin(&t->proj); acd_free_lin(&t->pool_embed); acd_free_lin(&t->q_in); acd_free_lin(&t->q_out);
    acd_free_lin(&t->d_embed); acd_free_lin(&t->d_proj);
    free(t->pool_special); free(t->d_special); free(t->pool_norm); free(t->d_norm);
    atk_free_layers(t->pool, t->npool); atk_free_layers(t->detok, t->npool);
    free(t);
}

/* model_dir holds acestep-v15-turbo/ (config.json and model.safetensors). `d` is that directory's loaded DiT. */
static AcdTok *atk_load(const AceDit *d, const char *model_dir){
    char err[512] = "", path[4200], dir[4096];
    AcdTok *t = (AcdTok *)calloc(1, sizeof(AcdTok));
    shards S; int have_s = 0;
    jval *cfg = NULL; char *text = NULL;
    if (!t) { fprintf(stderr, "acestep tokenizer: out of memory\n"); return NULL; }
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

    t->P = acd_json_int(cfg, "pool_window_size", 0);
    t->npool = acd_json_int(cfg, "num_attention_pooler_hidden_layers", 0);
    int fsq_dim = acd_json_int(cfg, "fsq_dim", 0), nq = acd_json_int(cfg, "fsq_input_num_quantizers", 0);
    jval *lv = json_get(cfg, "fsq_input_levels");
    if (t->P < 1 || t->P > ATK_MAX_P || t->npool < 1 || t->npool > ACD_MAX_LAYERS || !lv || lv->t != J_ARR || lv->len < 1 || lv->len > ATK_MAX_LEVELS) {
        snprintf(err, sizeof err, "config: pool_window_size, num_attention_pooler_hidden_layers or fsq_input_levels is missing or out of range"); goto fail; }
    t->nlev = lv->len;
    for (int i = 0; i < t->nlev; i++) {
        if (lv->kids[i]->t != J_NUM || lv->kids[i]->num < 2 || lv->kids[i]->num > 1024 || lv->kids[i]->num != (int)lv->kids[i]->num) { snprintf(err, sizeof err, "config: fsq_input_levels[%d] must be an integer of 2 or more", i); goto fail; }
        t->levels[i] = (int)lv->kids[i]->num;
    }
    if (nq != 1) { snprintf(err, sizeof err, "config: fsq_input_num_quantizers is %d; only a single quantizer is supported", nq); goto fail; }
    if (fsq_dim != d->D) { snprintf(err, sizeof err, "config: fsq_dim %d is not the hidden size %d", fsq_dim, d->D); goto fail; }
    if (t->npool > d->nl) { snprintf(err, sizeof err, "config: %d pooler layers but the DiT has %d layer types", t->npool, d->nl); goto fail; }

    st_init(&S, dir); have_s = 1;
    if (S.n == 0) { snprintf(err, sizeof err, "no safetensors in %.400s", dir); goto fail; }
    const int D = d->D, A = d->acdim, P = t->P;
    int64_t shD[1] = {D};
    { int64_t s[2] = {D, A};
      if (acd_load_lin(&S, "tokenizer.audio_acoustic_proj.weight", "tokenizer.audio_acoustic_proj.bias", D, A, 2, s, NULL, 0, 0, &t->proj, &t->wbytes, err, sizeof err)) goto fail; }
    { int64_t s[2] = {D, D};
      if (acd_load_lin(&S, "tokenizer.attention_pooler.embed_tokens.weight", "tokenizer.attention_pooler.embed_tokens.bias", D, D, 2, s, NULL, 0, 0, &t->pool_embed, &t->wbytes, err, sizeof err)) goto fail; }
    { int64_t s[3] = {1, 1, D};
      if (!(t->pool_special = acd_read_f32(&S, "tokenizer.attention_pooler.special_token", 3, s, err, sizeof err))) goto fail; }
    if (!(t->pool_norm = acd_read_f32(&S, "tokenizer.attention_pooler.norm.weight", 1, shD, err, sizeof err))) goto fail;
    if (acd_load_enc_layers(&S, "tokenizer.attention_pooler", t->npool, d, &t->pool, &t->wbytes, err, sizeof err)) goto fail;
    { int64_t s1[2] = {t->nlev, D}, s2[2] = {D, t->nlev};
      if (acd_load_lin(&S, "tokenizer.quantizer.project_in.weight", "tokenizer.quantizer.project_in.bias", t->nlev, D, 2, s1, NULL, 0, 0, &t->q_in, &t->wbytes, err, sizeof err)) goto fail;
      if (acd_load_lin(&S, "tokenizer.quantizer.project_out.weight", "tokenizer.quantizer.project_out.bias", D, t->nlev, 2, s2, NULL, 0, 0, &t->q_out, &t->wbytes, err, sizeof err)) goto fail; }
    { int64_t s[2] = {D, D};
      if (acd_load_lin(&S, "detokenizer.embed_tokens.weight", "detokenizer.embed_tokens.bias", D, D, 2, s, NULL, 0, 0, &t->d_embed, &t->wbytes, err, sizeof err)) goto fail; }
    { int64_t s[3] = {1, P, D};
      if (!(t->d_special = acd_read_f32(&S, "detokenizer.special_tokens", 3, s, err, sizeof err))) goto fail; }
    if (!(t->d_norm = acd_read_f32(&S, "detokenizer.norm.weight", 1, shD, err, sizeof err))) goto fail;
    if (acd_load_enc_layers(&S, "detokenizer", t->npool, d, &t->detok, &t->wbytes, err, sizeof err)) goto fail;
    { int64_t s[2] = {A, D};
      if (acd_load_lin(&S, "detokenizer.proj_out.weight", "detokenizer.proj_out.bias", A, D, 2, s, NULL, 0, 0, &t->d_proj, &t->wbytes, err, sizeof err)) goto fail; }
    st_destroy(&S);
    json_free(cfg); free(text);
    return t;
fail:
    fprintf(stderr, "acestep tokenizer: %s\n", err[0] ? err : "load failed");
    if (have_s) st_destroy(&S);
    json_free(cfg); free(text);
    atk_free(t);
    return NULL;
}

/* One encoder layer (acd_enc_layer) over `nwin` sequences of `L` rows each, stored one after the other: the projections,
 * norms and MLP see every row at once; the attention is each window's own. cs/sn are the RoPE tables of L positions repeated
 * for every window, [nwin*L][hd]. */
static void atk_enc_layer_win(const AceDit *d, const AcdEncLayer *l, float *x, int nwin, int L, const float *cs, const float *sn){
    const int D = d->D, H = d->heads, Hk = d->kv, hd = d->hd, rows = nwin * L;
    float *n = (float *)malloc((size_t)rows * D * sizeof(float)), *o = (float *)malloc((size_t)rows * D * sizeof(float));
    float *g = (float *)malloc((size_t)rows * d->inter * sizeof(float)), *u = (float *)malloc((size_t)rows * d->inter * sizeof(float));
    float *q = (float *)malloc((size_t)rows * H * hd * sizeof(float)), *k = (float *)malloc((size_t)rows * Hk * hd * sizeof(float));
    float *v = (float *)malloc((size_t)rows * Hk * hd * sizeof(float)), *ctx = (float *)malloc((size_t)rows * H * hd * sizeof(float));
    if (!n || !o || !g || !u || !q || !k || !v || !ctx) { fprintf(stderr, "acestep tokenizer: out of memory\n"); exit(1); }
    acd_rmsnorm(n, x, rows, D, l->ln1, d->eps);
    acd_lin(q, n, rows, &l->sa.q); acd_lin(k, n, rows, &l->sa.k); acd_lin(v, n, rows, &l->sa.v);
    acd_head_norm(q, rows, H, hd, l->sa.qn, d->eps);
    acd_head_norm(k, rows, Hk, hd, l->sa.kn, d->eps);
    acd_rope(q, rows, H, hd, cs, sn);
    acd_rope(k, rows, Hk, hd, cs, sn);
    for (int w = 0; w < nwin; w++)
        acd_attention(ctx + (size_t)w * L * H * hd, H * hd, q + (size_t)w * L * H * hd, H * hd, L,
                      k + (size_t)w * L * Hk * hd, v + (size_t)w * L * Hk * hd, Hk * hd, L, H, Hk, hd, l->sliding ? d->window : -1);
    acd_lin(o, ctx, rows, &l->sa.o);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)rows * D; i++) x[i] += o[i];
    acd_rmsnorm(n, x, rows, D, l->ln2, d->eps);
    acd_lin(g, n, rows, &l->gate); acd_lin(u, n, rows, &l->up);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)rows * d->inter; i++) g[i] = acd_silu(g[i]) * u[i];
    acd_lin(o, g, rows, &l->down);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)rows * D; i++) x[i] += o[i];
    free(n); free(o); free(g); free(u); free(q); free(k); free(v); free(ctx);
}

/* RoPE tables of L positions tiled for nwin windows: [nwin*L][hd] */
static void atk_rope_tiled(const AceDit *d, int L, int nwin, float **cs, float **sn){
    float *c = (float *)malloc((size_t)L * d->hd * sizeof(float)), *s = (float *)malloc((size_t)L * d->hd * sizeof(float));
    *cs = (float *)malloc((size_t)nwin * L * d->hd * sizeof(float)); *sn = (float *)malloc((size_t)nwin * L * d->hd * sizeof(float));
    if (!c || !s || !*cs || !*sn) { fprintf(stderr, "acestep tokenizer: out of memory\n"); exit(1); }
    acd_rope_table(c, s, L, d->hd, d->theta);
    for (int w = 0; w < nwin; w++) { memcpy(*cs + (size_t)w * L * d->hd, c, (size_t)L * d->hd * sizeof(float)); memcpy(*sn + (size_t)w * L * d->hd, s, (size_t)L * d->hd * sizeof(float)); }
    free(c); free(s);
}

/* x [W*P][acdim] (the windows' frames in order) -> out [W][D]: the pooler, upstream's AttentionPooler after audio_acoustic_proj. */
static void atk_pool(const AceDit *d, const AcdTok *t, const float *x, int W, float *out){
    const int D = d->D, P = t->P, L = P + 1;
    float *cs, *sn;
    const int bw = W < ATK_WB ? W : ATK_WB;
    atk_rope_tiled(d, L, bw, &cs, &sn);
    float *p = (float *)malloc((size_t)bw * P * D * sizeof(float)), *e = (float *)malloc((size_t)bw * P * D * sizeof(float));
    float *h = (float *)malloc((size_t)bw * L * D * sizeof(float));
    if (!p || !e || !h) { fprintf(stderr, "acestep tokenizer: out of memory\n"); exit(1); }
    for (int w0 = 0; w0 < W; w0 += bw) {
        const int nw = W - w0 < bw ? W - w0 : bw;
        acd_lin(p, x + (size_t)w0 * P * d->acdim, nw * P, &t->proj);
        acd_lin(e, p, nw * P, &t->pool_embed);
        for (int w = 0; w < nw; w++) {
            memcpy(h + (size_t)w * L * D, t->pool_special, (size_t)D * sizeof(float));
            memcpy(h + (size_t)(w * L + 1) * D, e + (size_t)w * P * D, (size_t)P * D * sizeof(float));
        }
        for (int i = 0; i < t->npool; i++) atk_enc_layer_win(d, &t->pool[i], h, nw, L, cs, sn);
        acd_rmsnorm(h, h, nw * L, D, t->pool_norm, d->eps);
        for (int w = 0; w < nw; w++) memcpy(out + (size_t)(w0 + w) * D, h + (size_t)w * L * D, (size_t)D * sizeof(float));      /* the special token's output */
    }
    free(cs); free(sn); free(p); free(e); free(h);
}

/* The FSQ's rounding of one value z of level count L, as vector_quantize_pytorch's symmetry_preserving_bound with a hard
 * clamp: the code in [-1, 1]. `bracket` (before the floor) is returned for the oracle's near-a-boundary test. */
static float atk_code(float z, int L, float *bracket){
    const float zc = z < -1.f ? -1.f : z > 1.f ? 1.f : z;
    const float lm1 = (float)(L - 1);
    const float b = lm1 * (zc + 1.f) / 2.f + 0.5f;
    if (bracket) *bracket = b;
    return (2.f / lm1) * floorf(b) - 1.f;
}

/* pooled [W][D] -> quantized [W][D]: ResidualFSQ with one quantizer. `bracket`, when not NULL, receives [W][nlev]: the value
 * each code was rounded from. */
static void atk_fsq(const AceDit *d, const AcdTok *t, const float *pooled, int W, float *out, float *bracket){
    const int D = d->D, n = t->nlev;
    float *z = (float *)malloc((size_t)W * n * sizeof(float)), *c = (float *)malloc((size_t)W * n * sizeof(float));
    if (!z || !c) { fprintf(stderr, "acestep tokenizer: out of memory\n"); exit(1); }
    acd_lin(z, pooled, W, &t->q_in);
    for (int w = 0; w < W; w++)
        for (int i = 0; i < n; i++) {
            const float clampv = 1.f + 1.f / (float)(t->levels[i] - 1);
            float v = tanhf(z[(size_t)w * n + i] / clampv) * clampv;       /* the soft clamp ahead of the residual quantizer */
            c[(size_t)w * n + i] = atk_code(v, t->levels[i], bracket ? bracket + (size_t)w * n + i : NULL);
        }
    acd_lin(out, c, W, &t->q_out);
    (void)D;
    free(z); free(c);
}

/* quantized [W][D] -> out [W*P][acdim]: the detokenizer. */
static void atk_detok(const AceDit *d, const AcdTok *t, const float *q, int W, float *out){
    const int D = d->D, P = t->P;
    float *cs, *sn;
    const int bw = W < ATK_WB ? W : ATK_WB;
    atk_rope_tiled(d, P, bw, &cs, &sn);
    float *e = (float *)malloc((size_t)bw * D * sizeof(float)), *h = (float *)malloc((size_t)bw * P * D * sizeof(float));
    if (!e || !h) { fprintf(stderr, "acestep tokenizer: out of memory\n"); exit(1); }
    for (int w0 = 0; w0 < W; w0 += bw) {
        const int nw = W - w0 < bw ? W - w0 : bw;
        acd_lin(e, q + (size_t)w0 * D, nw, &t->d_embed);
        for (int w = 0; w < nw; w++)
            for (int p = 0; p < P; p++) {
                const float *s = t->d_special + (size_t)p * D, *r = e + (size_t)w * D;
                float *o = h + (size_t)(w * P + p) * D;
                for (int i = 0; i < D; i++) o[i] = r[i] + s[i];
            }
        for (int i = 0; i < t->npool; i++) atk_enc_layer_win(d, &t->detok[i], h, nw, P, cs, sn);
        acd_rmsnorm(h, h, nw * P, D, t->d_norm, d->eps);
        acd_lin(out + (size_t)w0 * P * d->acdim, h, nw * P, &t->d_proj);
    }
    free(cs); free(sn); free(e); free(h);
}

/* The whole path, upstream's `tokenize` then `detokenize`: x [T][acdim] (source latents) -> hints [T][acdim]. `silence` is the
 * silence latent [>= P][acdim], whose first frames pad a T that is not a multiple of P; the hints are cut back to T. */
static int atk_hints(const AceDit *d, const AcdTok *t, const float *x, int T, const float *silence, float *hints){
    const int A = d->acdim, P = t->P, D = d->D;
    if (T < 1) return -1;
    const int pad = (P - T % P) % P, Tp = T + pad, W = Tp / P;
    float *xp = (float *)malloc((size_t)Tp * A * sizeof(float));
    float *pooled = (float *)malloc((size_t)W * D * sizeof(float)), *q = (float *)malloc((size_t)W * D * sizeof(float));
    float *hp = (float *)malloc((size_t)Tp * A * sizeof(float));
    if (!xp || !pooled || !q || !hp) { free(xp); free(pooled); free(q); free(hp); return -1; }
    memcpy(xp, x, (size_t)T * A * sizeof(float));
    if (pad) memcpy(xp + (size_t)T * A, silence, (size_t)pad * A * sizeof(float));
    atk_pool(d, t, xp, W, pooled);
    atk_fsq(d, t, pooled, W, q, NULL);
    atk_detok(d, t, q, W, hp);
    memcpy(hints, hp, (size_t)T * A * sizeof(float));
    free(xp); free(pooled); free(q); free(hp);
    return 0;
}

#endif
