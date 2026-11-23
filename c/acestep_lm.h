/* acestep_lm.h: ACE-Step 1.5's 5 Hz language model, the planner (acestep-5Hz-lm-0.6B, -1.7B, -4B).
 *
 * A Qwen3 decoder (RMSNorm, grouped-query attention with RoPE and per-head q/k norms, SwiGLU) with the audio codes added to its vocabulary:
 * the ids from <|audio_code_0|> up are the 64000 codes of the audio tokenizer's quantizer (acestep_tok.h, 25 Hz frames pooled by 5 to 5 codes a
 * second), and the head is the embedding table (tied). Given the caption and the lyrics, and the metadata of the song, it writes the codes of the
 * song's coarse structure, which the DiT takes as its context in place of a source clip's (the cover path, from codes instead of from audio).
 *
 * What this file does, held to upstream's LLMHandler by tools/acestep_ref_lm.py and `acestep --lm-ref` (docs/acestep.md):
 *   - the two prompts of the codes phase (the conditional one and the unconditional one that classifier-free guidance subtracts),
 *   - the model on the CPU: a prompt in one pass, then both sequences a token at a time in one batch of two rows (each weight is read once),
 *   - the guided, constrained sampling of upstream's _generate_with_cfg_custom: logits uncond + scale * (cond - uncond) over the audio codes and
 *     the end token only, the end token blocked until the number of codes the duration asks for (5 a second) and forced then, top-k, top-p, and a
 *     temperature (0 is an argmax: torch's multinomial cannot be reproduced, so the engine's own seeded draw is statistically, not bit-for-bit, upstream's).
 * Not here: the first phase (the model writing the metadata itself under a grammar), which needs bpm, key, time signature and duration from the
 * caller instead, and the understanding and formatting prompts. Needs acestep_dit.h and tok.h. */
#ifndef ACESTEP_LM_H
#define ACESTEP_LM_H

#include <strings.h>
#include "acestep_dit.h"
#include "tok.h"
#include "qwen38_nfc.h"

#define ALM_CODES 64000                /* the codes the quantizer has: ids 0..63999 of <|audio_code_N|> */
#define ALM_INSTRUCTION "Generate audio semantic tokens based on the given conditions:"
#define ALM_NO_INPUT "NO USER INPUT"

typedef struct {
    int D, H, Hk, hd, inter, nl, vocab;
    float eps, theta;
    AcdEncLayer *L;                    /* a Qwen3 layer is the DiT's encoder layer: q, k, v, o with their head norms, two norms, gate, up, down */
    float *norm;
    uint16_t *embed;                   /* bf16 [vocab][D]: the input embeddings and, tied, the head */
    size_t wbytes;
    Tok tok;
    int audio0;                        /* the id of <|audio_code_0|> */
    int eos;                           /* <|im_end|> */
    int rope_n; float *cs, *sn;        /* RoPE tables for positions 0 .. rope_n - 1 */
    char dir_name[64];
    void *meta_tab; void (*meta_free)(void *);     /* the first phase's token tables (acestep_meta.h), built the first time */
    void *vk;                          /* the planner's weights on a Vulkan device (acestep_lm_vk.h) */
    int vk_tried, vk_on, vk_fail;      /* the device was asked for; the steps run on it; a step failed on it (the run is started again on the CPU) */
} AceLm;

static void alm_vk_free(AceLm *m);
static void alm_free(AceLm *m){
    if (!m) return;
    alm_vk_free(m);
    for (int i = 0; m->L && i < m->nl; i++) {
        AcdEncLayer *l = &m->L[i];
        acd_free_attn(&l->sa); free(l->ln1); free(l->ln2); acd_free_lin(&l->gate); acd_free_lin(&l->up); acd_free_lin(&l->down);
    }
    free(m->L); free(m->norm); free(m->embed); free(m->cs); free(m->sn);
    tok_free(&m->tok);
    if (m->meta_tab && m->meta_free) m->meta_free(m->meta_tab);
    free(m);
}

/* `dir` is the LM's directory (config.json, model.safetensors, tokenizer.json). NULL (with the reason on stderr) when it is missing or does not fit. */
static AceLm *alm_load(const char *dir){
    char err[512] = "", path[4300];
    AceLm *m = (AceLm *)calloc(1, sizeof *m);
    shards S; int have_s = 0;
    jval *cfg = NULL; char *text = NULL;
    if (!m) { fprintf(stderr, "acestep lm: out of memory\n"); return NULL; }
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, sizeof err, "cannot open %.400s", path); goto fail; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    text = (char *)malloc(n > 0 ? n + 1 : 1);
    if (!text || n <= 0 || n > (8 << 20) || fread(text, 1, n, f) != (size_t)n) { fclose(f); snprintf(err, sizeof err, "cannot read %.400s", path); goto fail; }
    fclose(f); text[n] = 0;
    cfg = json_parse_checked(text);
    if (!cfg || cfg->t != J_OBJ) { snprintf(err, sizeof err, "%.400s is not a JSON object", path); goto fail; }
    m->D = acd_json_int(cfg, "hidden_size", 0); m->inter = acd_json_int(cfg, "intermediate_size", 0);
    m->nl = acd_json_int(cfg, "num_hidden_layers", 0); m->H = acd_json_int(cfg, "num_attention_heads", 0);
    m->Hk = acd_json_int(cfg, "num_key_value_heads", 0); m->hd = acd_json_int(cfg, "head_dim", 0);
    m->vocab = acd_json_int(cfg, "vocab_size", 0);
    m->eps = (float)acd_json_num(cfg, "rms_norm_eps", 1e-6); m->theta = (float)acd_json_num(cfg, "rope_theta", 10000.0);
    jval *rs = json_get(cfg, "rope_scaling");
    if (rs && rs->t != J_NULL) { snprintf(err, sizeof err, "config: rope_scaling is not supported"); goto fail; }
    jval *ab = json_get(cfg, "attention_bias");
    if (ab && ab->t == J_BOOL && ab->boolean) { snprintf(err, sizeof err, "config: attention_bias is not supported"); goto fail; }
    jval *tie = json_get(cfg, "tie_word_embeddings");
    if (!(tie && tie->t == J_BOOL && tie->boolean)) { snprintf(err, sizeof err, "config: only tied embeddings are supported (the head is the embedding table)"); goto fail; }
    if (m->D < 2 || m->D > ACD_MAX_DIM || m->inter < 1 || m->nl < 1 || m->nl > ACD_MAX_LAYERS || m->H < 1 || m->H > ACD_MAX_HEADS || m->Hk < 1 || m->H % m->Hk ||
        m->hd < 2 || (m->hd & 1) || m->hd > ACD_MAX_HD || m->vocab < 1000 || m->vocab > (1 << 20)) { snprintf(err, sizeof err, "config: a size is missing or out of range"); goto fail; }
    st_init(&S, dir); have_s = 1;
    if (S.n == 0) { snprintf(err, sizeof err, "no safetensors in %.400s", dir); goto fail; }
    m->L = (AcdEncLayer *)calloc((size_t)m->nl, sizeof *m->L);
    if (!m->L) { snprintf(err, sizeof err, "out of memory"); goto fail; }
    int64_t shd[1] = {m->D};
    /* the 1.7B's tensors are named `layers.N...`, the 0.6B's and the 4B's (saved as a Qwen3ForCausalLM) `model.layers.N...` */
    const char *pf = st_find(&S, "model.embed_tokens.weight") ? "model." : "";
    for (int i = 0; i < m->nl; i++) {
        char p[300], nm[300];
        snprintf(p, sizeof p, "%slayers.%d.self_attn", pf, i);
        if (acd_load_attn(&S, p, m->D, m->H, m->Hk, m->hd, &m->L[i].sa, &m->wbytes, err, sizeof err)) goto fail;
        snprintf(nm, sizeof nm, "%slayers.%d.input_layernorm.weight", pf, i);    if (!(m->L[i].ln1 = acd_read_f32(&S, nm, 1, shd, err, sizeof err))) goto fail;
        snprintf(nm, sizeof nm, "%slayers.%d.post_attention_layernorm.weight", pf, i); if (!(m->L[i].ln2 = acd_read_f32(&S, nm, 1, shd, err, sizeof err))) goto fail;
        snprintf(p, sizeof p, "%slayers.%d.mlp", pf, i);
        if (acd_load_mlp(&S, p, m->D, m->inter, &m->L[i].gate, &m->L[i].up, &m->L[i].down, &m->wbytes, err, sizeof err)) goto fail;
    }
    char nmn[64], nme[64];
    snprintf(nmn, sizeof nmn, "%snorm.weight", pf); snprintf(nme, sizeof nme, "%sembed_tokens.weight", pf);
    if (!(m->norm = acd_read_f32(&S, nmn, 1, shd, err, sizeof err))) goto fail;
    st_tensor *te = st_find(&S, nme);
    if (!te || te->dtype != 0 || te->numel != (int64_t)m->vocab * m->D) { snprintf(err, sizeof err, "the embedding table is missing or not bf16"); goto fail; }
    m->embed = (uint16_t *)malloc((size_t)te->numel * 2);
    if (!m->embed) { snprintf(err, sizeof err, "out of memory"); goto fail; }
    st_read_raw(&S, nme, m->embed, 1);
    m->wbytes += (size_t)te->numel * 2;
    st_destroy(&S); have_s = 0;
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    f = fopen(path, "rb");
    if (!f) { snprintf(err, sizeof err, "no tokenizer.json in %.400s", dir); goto fail; }
    fclose(f);
    tok_load(&m->tok, path);
    m->audio0 = tok_id_of(&m->tok, "<|audio_code_0|>");
    m->eos = tok_id_of(&m->tok, "<|im_end|>");
    if (m->audio0 < 0 || m->eos < 0 || m->audio0 + ALM_CODES > m->vocab) { snprintf(err, sizeof err, "the tokenizer has no audio codes or no <|im_end|>"); goto fail; }
    { const char *b = strrchr(dir, '/'); const char *b2 = strrchr(dir, '\\'); if (b2 > b) b = b2; snprintf(m->dir_name, sizeof m->dir_name, "%s", b ? b + 1 : dir); }
    json_free(cfg); free(text);
    return m;
fail:
    fprintf(stderr, "acestep lm: %s\n", err[0] ? err : "load failed");
    if (have_s) st_destroy(&S);
    json_free(cfg); free(text);
    alm_free(m);
    return NULL;
}

/* ------------------------------------------------------------------------------------------------------ the model */

/* k, v: the keys and values on the host; NULL when the sequence lives on the device (vk), which holds them instead. */
typedef struct { float **k, **v; int cap, len; void *vk; } AlmCache;

static void alm_rope(AceLm *m, int n);
static void alm_embed_row(const AceLm *m, int id, float *dst);
#ifdef COLI_VULKAN
#include "acestep_lm_vk.h"
#else
static int alm_vk_open(AceLm *m){ (void)m; return 0; }
static void alm_vk_cache_free(AlmCache *c){ (void)c; }
static void alm_vk_free(AceLm *m){ (void)m; }
static int alm_vk_forward(AceLm *m, AlmCache *C, const float *xrows, int n, int pos0, float *last){ (void)m; (void)C; (void)xrows; (void)n; (void)pos0; (void)last; return 0; }
static int alm_vk_logits(AceLm *m, const float *h, int nr, float *full){ (void)m; (void)h; (void)nr; (void)full; return 0; }
#endif

static int alm_cache_new(AceLm *m, AlmCache *c, int cap){
    memset(c, 0, sizeof *c);
    if (alm_vk_open(m)) { c->cap = cap; return 0; }                           /* on the card: the cache is made there, by the first step */
    c->k = (float **)calloc((size_t)m->nl, sizeof(float *)); c->v = (float **)calloc((size_t)m->nl, sizeof(float *));
    if (!c->k || !c->v) return -1;
    c->cap = cap;
    for (int i = 0; i < m->nl; i++) {
        c->k[i] = (float *)malloc((size_t)cap * m->Hk * m->hd * sizeof(float)); c->v[i] = (float *)malloc((size_t)cap * m->Hk * m->hd * sizeof(float));
        if (!c->k[i] || !c->v[i]) return -1;
    }
    return 0;
}
static void alm_cache_free(const AceLm *m, AlmCache *c){
    alm_vk_cache_free(c);
    for (int i = 0; c->k && i < m->nl; i++) { free(c->k[i]); free(c->v[i]); }
    free(c->k); free(c->v); memset(c, 0, sizeof *c);
}

/* RoPE tables out to n positions (grown when a longer sequence comes). */
static void alm_rope(AceLm *m, int n){
    if (n <= m->rope_n) return;
    free(m->cs); free(m->sn);
    m->cs = (float *)malloc((size_t)n * m->hd * sizeof(float)); m->sn = (float *)malloc((size_t)n * m->hd * sizeof(float));
    if (!m->cs || !m->sn) { fprintf(stderr, "acestep lm: out of memory\n"); exit(1); }
    acd_rope_table(m->cs, m->sn, n, m->hd, m->theta);
    m->rope_n = n;
}

static void alm_embed_row(const AceLm *m, int id, float *dst){
    const uint16_t *r = m->embed + (size_t)id * m->D;
    for (int i = 0; i < m->D; i++) { uint32_t u = (uint32_t)r[i] << 16; memcpy(&dst[i], &u, 4); }
}

/* One attention head of one row against cache rows 0 .. upto - 1. */
static void alm_attend_row(const AceLm *m, const AlmCache *C, int li, const float *q, int upto, float *sc, float *out){
    const int H = m->H, Hk = m->Hk, hd = m->hd, group = H / Hk;
    const float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < H; h++) {
        const float *qh = q + (size_t)h * hd;
        const float *K = C->k[li] + (size_t)(h / group) * hd, *V = C->v[li] + (size_t)(h / group) * hd;
        float mx = -1e30f;
        for (int j = 0; j < upto; j++) {
            const float *kj = K + (size_t)j * Hk * hd;
            float s = 0;
            for (int d = 0; d < hd; d++) s += qh[d] * kj[d];
            sc[j] = s * scale;
            if (sc[j] > mx) mx = sc[j];
        }
        float den = 0;
        for (int j = 0; j < upto; j++) { sc[j] = expf(sc[j] - mx); den += sc[j]; }
        float *oh = out + (size_t)h * hd;
        for (int d = 0; d < hd; d++) oh[d] = 0;
        for (int j = 0; j < upto; j++) {
            const float w = sc[j] / den;
            const float *vj = V + (size_t)j * Hk * hd;
            for (int d = 0; d < hd; d++) oh[d] += w * vj[d];
        }
    }
}

/* `nr` rows through every layer, row r being position pos[r] of sequence cache[r] (a decode step of nr sequences: each weight is read once);
 * x [nr][D] is the rows' embeddings and is updated in place. The keys and values go into the caches. */
static void alm_decode_rows(AceLm *m, AlmCache **cache, const int *pos, float *x, int nr){
    const int D = m->D, H = m->H, Hk = m->Hk, hd = m->hd, I = m->inter;
    if (!cache[0]->k) {                                                        /* on the card: a submission a row, each sequence with its own cache */
        for (int r = 0; r < nr && !m->vk_fail; r++) {
            if (alm_vk_forward(m, cache[r], x + (size_t)r * D, 1, pos[r], x + (size_t)r * D)) cache[r]->len = pos[r] + 1;
        }
        return;
    }
    int maxpos = 0;
    for (int r = 0; r < nr; r++) if (pos[r] + 1 > maxpos) maxpos = pos[r] + 1;
    alm_rope(m, maxpos);
    float *nb = (float *)malloc((size_t)nr * D * sizeof(float)), *o = (float *)malloc((size_t)nr * D * sizeof(float));
    float *q = (float *)malloc((size_t)nr * H * hd * sizeof(float)), *cx = (float *)malloc((size_t)nr * H * hd * sizeof(float));
    float *kk = (float *)malloc((size_t)nr * Hk * hd * sizeof(float)), *vv = (float *)malloc((size_t)nr * Hk * hd * sizeof(float));
    float *g = (float *)malloc((size_t)nr * I * sizeof(float)), *u = (float *)malloc((size_t)nr * I * sizeof(float));
    float *sc = (float *)malloc((size_t)(maxpos > 1 ? maxpos : 1) * sizeof(float) * (size_t)nr);
    if (!nb || !o || !q || !cx || !kk || !vv || !g || !u || !sc) { fprintf(stderr, "acestep lm: out of memory\n"); exit(1); }
    for (int li = 0; li < m->nl; li++) {
        const AcdEncLayer *l = &m->L[li];
        acd_rmsnorm(nb, x, nr, D, l->ln1, m->eps);
        acd_lin(q, nb, nr, &l->sa.q); acd_lin(kk, nb, nr, &l->sa.k); acd_lin(vv, nb, nr, &l->sa.v);
        acd_head_norm(q, nr, H, hd, l->sa.qn, m->eps);
        acd_head_norm(kk, nr, Hk, hd, l->sa.kn, m->eps);
        for (int r = 0; r < nr; r++) {                                         /* each row has its own position */
            acd_rope(q + (size_t)r * H * hd, 1, H, hd, m->cs + (size_t)pos[r] * hd, m->sn + (size_t)pos[r] * hd);
            acd_rope(kk + (size_t)r * Hk * hd, 1, Hk, hd, m->cs + (size_t)pos[r] * hd, m->sn + (size_t)pos[r] * hd);
            memcpy(cache[r]->k[li] + (size_t)pos[r] * Hk * hd, kk + (size_t)r * Hk * hd, (size_t)Hk * hd * sizeof(float));
            memcpy(cache[r]->v[li] + (size_t)pos[r] * Hk * hd, vv + (size_t)r * Hk * hd, (size_t)Hk * hd * sizeof(float));
        }
        #pragma omp parallel for schedule(static) if (nr > 1)
        for (int r = 0; r < nr; r++)
            alm_attend_row(m, cache[r], li, q + (size_t)r * H * hd, pos[r] + 1, sc + (size_t)r * (maxpos > 1 ? maxpos : 1), cx + (size_t)r * H * hd);
        acd_lin(o, cx, nr, &l->sa.o);
        for (int64_t i = 0; i < (int64_t)nr * D; i++) x[i] += o[i];
        acd_rmsnorm(nb, x, nr, D, l->ln2, m->eps);
        acd_lin(g, nb, nr, &l->gate); acd_lin(u, nb, nr, &l->up);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)nr * I; i++) g[i] = acd_silu(g[i]) * u[i];
        acd_lin(o, g, nr, &l->down);
        for (int64_t i = 0; i < (int64_t)nr * D; i++) x[i] += o[i];
    }
    for (int r = 0; r < nr; r++) cache[r]->len = pos[r] + 1;
    free(nb); free(o); free(q); free(cx); free(kk); free(vv); free(g); free(u); free(sc);
}

/* A prompt of n tokens, from position 0, into an empty cache; the hidden state of the last token (before the final norm) in last[D]. The attention is
 * causal within the prompt; it runs as one block per layer. */
static void alm_prefill(AceLm *m, AlmCache *C, const int *ids, int n, float *last){
    const int D = m->D, H = m->H, Hk = m->Hk, hd = m->hd, I = m->inter;
    if (!C->k) {                                                               /* on the card */
        float *e = (float *)malloc((size_t)n * D * sizeof(float));
        if (!e) { fprintf(stderr, "acestep lm: out of memory\n"); exit(1); }
        for (int i = 0; i < n; i++) alm_embed_row(m, ids[i], e + (size_t)i * D);
        if (!m->vk_fail && alm_vk_forward(m, C, e, n, 0, last)) C->len = n;
        free(e);
        return;
    }
    alm_rope(m, n);
    float *x = (float *)malloc((size_t)n * D * sizeof(float));
    float *nb = (float *)malloc((size_t)n * D * sizeof(float)), *o = (float *)malloc((size_t)n * D * sizeof(float));
    float *q = (float *)malloc((size_t)n * H * hd * sizeof(float)), *cx = (float *)malloc((size_t)n * H * hd * sizeof(float));
    float *kk = (float *)malloc((size_t)n * Hk * hd * sizeof(float)), *vv = (float *)malloc((size_t)n * Hk * hd * sizeof(float));
    float *g = (float *)malloc((size_t)n * I * sizeof(float)), *u = (float *)malloc((size_t)n * I * sizeof(float));
    if (!x || !nb || !o || !q || !cx || !kk || !vv || !g || !u) { fprintf(stderr, "acestep lm: out of memory\n"); exit(1); }
    for (int i = 0; i < n; i++) alm_embed_row(m, ids[i], x + (size_t)i * D);
    for (int li = 0; li < m->nl; li++) {
        const AcdEncLayer *l = &m->L[li];
        acd_rmsnorm(nb, x, n, D, l->ln1, m->eps);
        acd_lin(q, nb, n, &l->sa.q); acd_lin(kk, nb, n, &l->sa.k); acd_lin(vv, nb, n, &l->sa.v);
        acd_head_norm(q, n, H, hd, l->sa.qn, m->eps);
        acd_head_norm(kk, n, Hk, hd, l->sa.kn, m->eps);
        acd_rope(q, n, H, hd, m->cs, m->sn);
        acd_rope(kk, n, Hk, hd, m->cs, m->sn);
        memcpy(C->k[li], kk, (size_t)n * Hk * hd * sizeof(float));
        memcpy(C->v[li], vv, (size_t)n * Hk * hd * sizeof(float));
        acd_attention_ex(cx, H * hd, q, H * hd, n, C->k[li], C->v[li], Hk * hd, n, H, Hk, hd, -1, 1);
        acd_lin(o, cx, n, &l->sa.o);
        for (int64_t i = 0; i < (int64_t)n * D; i++) x[i] += o[i];
        acd_rmsnorm(nb, x, n, D, l->ln2, m->eps);
        acd_lin(g, nb, n, &l->gate); acd_lin(u, nb, n, &l->up);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n * I; i++) g[i] = acd_silu(g[i]) * u[i];
        acd_lin(o, g, n, &l->down);
        for (int64_t i = 0; i < (int64_t)n * D; i++) x[i] += o[i];
    }
    C->len = n;
    memcpy(last, x + (size_t)(n - 1) * D, (size_t)D * sizeof(float));
    free(x); free(nb); free(o); free(q); free(cx); free(kk); free(vv); free(g); free(u);
}

/* The head, on the columns asked for: logits[r][j] = <final_norm(h[r]), embed[ids[j]]>, j < nid. The head is the embedding table, and a codes
 * phase only needs the 64000 audio codes and the end token of its 217204. */
static void alm_logits(AceLm *m, const float *h, int nr, const int *ids, int nid, float *logits){
    const int D = m->D;
    if (m->vk_on && nr <= 2) {                                                 /* the head on the card, then the columns asked for */
        float *full = (float *)malloc((size_t)nr * m->vocab * sizeof(float));
        if (!full) { fprintf(stderr, "acestep lm: out of memory\n"); exit(1); }
        if (alm_vk_logits(m, h, nr, full)) {
            for (int r = 0; r < nr; r++) for (int j = 0; j < nid; j++) logits[(size_t)r * nid + j] = full[(size_t)r * m->vocab + ids[j]];
            free(full);
            return;
        }
        free(full);
        if (m->vk_fail) { memset(logits, 0, (size_t)nr * nid * sizeof(float)); return; }      /* the run is started again on the CPU */
    }
    float *hn = (float *)malloc((size_t)nr * D * sizeof(float));
    if (!hn) { fprintf(stderr, "acestep lm: out of memory\n"); exit(1); }
    acd_rmsnorm(hn, h, nr, D, m->norm, m->eps);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nid; j++) {
        const uint16_t *w = m->embed + (size_t)ids[j] * D;
        for (int r = 0; r < nr; r++) {
            const float *x = hn + (size_t)r * D;
            float s = 0;
            for (int i = 0; i < D; i++) { uint32_t u = (uint32_t)w[i] << 16; float wf; memcpy(&wf, &u, 4); s += x[i] * wf; }
            logits[(size_t)r * nid + j] = s;
        }
    }
    free(hn);
}

/* -------------------------------------------------------------------------------------------------------- prompts */

static char *alm_dup(const char *s){ size_t n = strlen(s); char *p = (char *)malloc(n + 1); if (p) memcpy(p, s, n + 1); return p; }

/* The codes phase's two prompts (upstream's build_formatted_prompt_with_cot, the training-aligned path): the chat template with the instruction as
 * the system turn, then the user's caption and lyrics (or, for the unconditional one, "NO USER INPUT") and the assistant turn left open after its
 * thinking. malloc'd. */
static char *alm_prompt(const char *caption, const char *lyrics, const char *cot, int uncond){
    const char *cp = caption ? caption : "", *ly = lyrics ? lyrics : "";
    const char *think = uncond ? "<think>\n\n</think>" : cot;
    size_t cap = strlen(cp) + strlen(ly) + strlen(think) + 512;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    int n = snprintf(out, cap, "<|im_start|>system\n# Instruction\n%s\n\n<|im_end|>\n<|im_start|>user\n", ALM_INSTRUCTION);
    if (uncond) n += snprintf(out + n, cap - n, "%s<|im_end|>\n", ALM_NO_INPUT);
    else n += snprintf(out + n, cap - n, "# Caption\n%s\n\n# Lyric\n%s\n<|im_end|>\n", cp, ly);
    snprintf(out + n, cap - n, "<|im_start|>assistant\n%s\n\n", think);
    return out;
}

/* The ids of a prompt: the tokenizer's, NFC-normalised first as the other ACE-Step prompts are, with the chat tokens atomic. n ids into a malloc'd array. */
static int *alm_tokenize(AceLm *m, const char *text, int *n){
    char *norm = NULL; size_t nl = 0;
    if (q38_nfc_normalize(text, strlen(text), &norm, &nl) != 0) { norm = alm_dup(text); nl = strlen(text); }
    if (!norm) return NULL;
    int cap = (int)nl + 64;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) { free(norm); return NULL; }
    *n = tok_encode(&m->tok, norm, (int)nl, ids, cap);
    free(norm);
    return ids;
}

/* ------------------------------------------------------------------------------------------------------ the codes */

typedef struct {
    double temperature;                /* 0 is an argmax */
    double cfg_scale;                  /* guidance on the logits; 1 or less runs the conditional sequence alone */
    int top_k;                         /* 0: off */
    double top_p;                      /* 0 or 1: off */
    uint64_t seed;
} AlmSample;

/* Called after each model evaluation with the step, the two sequences' logits over the audio codes and the end token ([ALM_CODES + 1] each; unc is
 * NULL without guidance) and the token chosen: for oracles and progress. Returns nonzero to stop. */
typedef int (*AlmStepFn)(int step, const float *cond, const float *unc, int token, void *ud);

static int alm_cmp_float(const void *a, const void *b){ float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }

static uint64_t alm_rng(uint64_t *s){                  /* splitmix64 */
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* Upstream's filters and sampler on a [nid] row of guided logits, in place: top-k keeps every logit not below the k-th largest; top-p sorts, keeps
 * the smallest prefix whose probability passes p (always the first), -inf for the rest. Then the token: argmax, or a draw from softmax(logits / T). */
static int alm_choose(float *lg, int nid, const AlmSample *s, uint64_t *rng){
    if (s->top_k > 0 && s->top_k < nid) {
        float *tmp = (float *)malloc((size_t)nid * sizeof(float));
        memcpy(tmp, lg, (size_t)nid * sizeof(float));
        qsort(tmp, (size_t)nid, sizeof(float), alm_cmp_float);                 /* ascending: the k-th largest is nid - k */
        const int target = nid - s->top_k;
        const float kth = tmp[target];
        free(tmp);
        for (int i = 0; i < nid; i++) if (lg[i] < kth) lg[i] = -INFINITY;
    }
    if (s->top_p > 0.0 && s->top_p < 1.0) {
        int *ord = (int *)malloc((size_t)nid * sizeof(int));
        for (int i = 0; i < nid; i++) ord[i] = i;
        /* a stable sort by descending logit, by merge sort on indices */
        int *tmp = (int *)malloc((size_t)nid * sizeof(int));
        for (int w = 1; w < nid; w *= 2) {
            for (int a = 0; a < nid; a += 2 * w) {
                int mid = a + w < nid ? a + w : nid, end = a + 2 * w < nid ? a + 2 * w : nid, i = a, j = mid, k = a;
                while (i < mid && j < end) tmp[k++] = lg[ord[j]] > lg[ord[i]] ? ord[j++] : ord[i++];
                while (i < mid) tmp[k++] = ord[i++];
                while (j < end) tmp[k++] = ord[j++];
            }
            int *t = ord; ord = tmp; tmp = t;
        }
        float mx = lg[ord[0]];
        double den = 0;
        for (int i = 0; i < nid; i++) if (lg[ord[i]] > -INFINITY) den += exp((double)lg[ord[i]] - mx);
        double cum = 0;
        for (int i = 0; i < nid; i++) {
            const double p = lg[ord[i]] > -INFINITY ? exp((double)lg[ord[i]] - mx) / den : 0.0;
            cum += p;
            if (i > 0 && cum - p > s->top_p) lg[ord[i]] = -INFINITY;          /* removed when the cumulative probability BEFORE it already passed p */
        }
        free(ord); free(tmp);
    }
    if (s->temperature <= 0) {
        int best = 0;
        for (int i = 1; i < nid; i++) if (lg[i] > lg[best]) best = i;
        return best;
    }
    float mx = -INFINITY;
    for (int i = 0; i < nid; i++) if (lg[i] > mx) mx = lg[i];
    double den = 0;
    for (int i = 0; i < nid; i++) den += lg[i] > -INFINITY ? exp(((double)lg[i] - mx) / s->temperature) : 0.0;
    double u = (double)(alm_rng(rng) >> 11) * (1.0 / 9007199254740992.0) * den, acc = 0;
    int pick = 0;
    for (int i = 0; i < nid; i++) {
        if (!(lg[i] > -INFINITY)) continue;
        acc += exp(((double)lg[i] - mx) / s->temperature);
        pick = i;
        if (u < acc) break;
    }
    return pick;
}

/* Upstream's codes phase: both prompts in, then one token a step until the end token, which is blocked until `target` codes are out and forced after.
 * codes[target] receives the codes (ids 0..63999). Returns how many codes were written, or -1 (reason in err) / -2 when `step` asked to stop. */
static int alm_generate_codes_run(AceLm *m, const int *cond, int nc, const int *unc, int nu, int target, const AlmSample *s, int *codes,
                                  AlmStepFn step, void *ud, char *err, size_t errn){
    const int D = m->D, nid = ALM_CODES + 1;
    const int guided = s->cfg_scale > 1.0;
    int *ids = (int *)malloc((size_t)nid * sizeof(int));
    if (!ids) { snprintf(err, errn, "out of memory"); return -1; }
    for (int i = 0; i < ALM_CODES; i++) ids[i] = m->audio0 + i;
    ids[ALM_CODES] = m->eos;
    const int rows = guided ? 2 : 1;
    AlmCache c0, c1;
    memset(&c0, 0, sizeof c0); memset(&c1, 0, sizeof c1);
    const int cap = (nc > nu ? nc : nu) + target + 8;
    float *h = (float *)malloc((size_t)2 * D * sizeof(float)), *x = (float *)malloc((size_t)2 * D * sizeof(float));
    float *lg = (float *)malloc((size_t)2 * nid * sizeof(float)), *cfg = (float *)malloc((size_t)nid * sizeof(float));
    int produced = -1;
    if (!h || !x || !lg || !cfg || alm_cache_new(m, &c0, cap) || (guided && alm_cache_new(m, &c1, cap))) { snprintf(err, errn, "out of memory"); goto done; }
    alm_prefill(m, &c0, cond, nc, h);
    if (guided) alm_prefill(m, &c1, unc, nu, h + D);
    uint64_t rng = s->seed ? s->seed : 0x2545F4914F6CDD1Dull;
    AlmCache *caches[2] = { &c0, &c1 };
    int pos[2] = { nc, nu };
    int count = 0;
    produced = 0;
    for (int i = 0; i <= target; i++) {                                         /* target codes, then the end token */
        alm_logits(m, h, rows, ids, nid, lg);
        const float *cl = lg, *ul = guided ? lg + nid : NULL;
        if (guided) for (int j = 0; j < nid; j++) cfg[j] = ul[j] + (float)s->cfg_scale * (cl[j] - ul[j]);
        else memcpy(cfg, cl, (size_t)nid * sizeof(float));
        if (count < target) cfg[ALM_CODES] = -INFINITY;                         /* blocked until the duration's worth of codes */
        else { for (int j = 0; j < ALM_CODES; j++) cfg[j] = -INFINITY; }        /* then only the end token */
        int tok = alm_choose(cfg, nid, s, &rng);
        if (step && step(i, cl, ul, tok < ALM_CODES ? tok : -1, ud)) { produced = -2; goto done; }
        if (tok == ALM_CODES) break;
        codes[count++] = tok;
        produced = count;
        for (int r = 0; r < rows; r++) alm_embed_row(m, m->audio0 + tok, x + (size_t)r * D);
        alm_decode_rows(m, caches, pos, x, rows);
        for (int r = 0; r < rows; r++) pos[r]++;
        memcpy(h, x, (size_t)rows * D * sizeof(float));
    }
done:
    free(ids); free(h); free(x); free(lg); free(cfg);
    alm_cache_free(m, &c0); alm_cache_free(m, &c1);
    return produced;
}

/* A run that failed on the card is made again on the CPU (the device is off for the rest of the process by then). */
static int alm_generate_codes(AceLm *m, const int *cond, int nc, const int *unc, int nu, int target, const AlmSample *s, int *codes,
                              AlmStepFn step, void *ud, char *err, size_t errn){
    m->vk_fail = 0;
    int r = alm_generate_codes_run(m, cond, nc, unc, nu, target, s, codes, step, ud, err, errn);
    if (m->vk_fail) { m->vk_fail = 0; r = alm_generate_codes_run(m, cond, nc, unc, nu, target, s, codes, step, ud, err, errn); }
    return r;
}

#endif
