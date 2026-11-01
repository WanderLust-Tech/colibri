/* acestep_text.h -- the text side of ACE-Step 1.5: the prompts the DiT's conditioning is built from, their tokenizer, and
 * the encoder (Qwen3-Embedding-0.6B) that turns the caption prompt into hidden states.
 *
 * What upstream's handler does for text-to-music, and so what this does:
 *
 *   caption prompt  "# Instruction\n{instruction}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n"
 *                   with the instruction "Fill the audio semantic mask based on the given conditions:" and the metas
 *                   "- bpm: {bpm}\n- timesignature: {ts}\n- keyscale: {key}\n- duration: {n} seconds\n" (N/A where a
 *                   field is not given, the duration cut to whole seconds). Tokenized, cut to 256 tokens, run through
 *                   the whole encoder (28 causal layers): its last hidden state, one 1024-wide vector per token.
 *   lyric prompt    "# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>". Tokenized, cut to 2048 tokens; the
 *                   lyric path uses NO layers of this encoder: only the embedding table's rows for those ids.
 *
 * The tokenizer is the one in the checkpoint (Qwen's byte-level BPE, tok.h): an NFC normalizer; a literal
 * "<|endoftext|>" in the text is one token; digits are split one at a time; and the post-processor appends one more
 * <|endoftext|> to every sequence, so a prompt that ends with the literal token ends with two. Truncation leaves room
 * for that appended token: a sequence cut to N tokens is the first N-1 of the text and then <|endoftext|>.
 *
 * Interface: act_tok_open / act_tokenize / act_caption_prompt / act_lyric_prompt; the encoder (act_load, act_encode,
 * act_embed) is below.
 */
#ifndef COLIBRI_ACESTEP_TEXT_H
#define COLIBRI_ACESTEP_TEXT_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tok.h"
#include "qwen38_nfc.h"
#include "acestep_dit.h"                  /* the encoder's layers are computed with the DiT's pieces */

#define ACT_CAPTION_MAX 256       /* the handler's truncation length for the caption prompt */
#define ACT_LYRIC_MAX 2048         /* and for the lyric prompt */
#define ACT_DEFAULT_INSTRUCTION "Fill the audio semantic mask based on the given conditions:"

typedef struct { Tok tok; int eot; } AceTok;

/* model_dir holds Qwen3-Embedding-0.6B/tokenizer.json. 0 on success. A malformed tokenizer.json ends the process, as in
 * the other engines (tok_load). */
static int act_tok_open(AceTok *t, const char *model_dir){
    char path[4300];
    if (snprintf(path, sizeof path, "%s/Qwen3-Embedding-0.6B/tokenizer.json", model_dir) >= (int)sizeof path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "acestep text: cannot open %.400s\n", path); return -1; }
    fclose(f);
    memset(t, 0, sizeof *t);
    tok_load(&t->tok, path);
    t->eot = tok_id_of(&t->tok, "<|endoftext|>");
    if (t->eot < 0) { fprintf(stderr, "acestep text: the tokenizer has no <|endoftext|>\n"); tok_free(&t->tok); return -1; }
    return 0;
}

static void act_tok_close(AceTok *t){ tok_free(&t->tok); memset(t, 0, sizeof *t); }

/* Token ids of `text` as upstream's tokenizer call returns them with truncation at max_length: NFC, BPE with the added
 * tokens atomic, cut to max_length - 1, then the post-processor's <|endoftext|>. out must hold max_length ids; returns
 * how many were written (at least 1: the appended token). */
static int act_tokenize(AceTok *t, const char *text, int max_length, int *out){
    if (max_length < 1) return 0;
    const char *s = text ? text : "";
    char *norm = NULL; size_t norm_len = 0;
    if (q38_nfc_normalize(s, strlen(s), &norm, &norm_len) != 0) { norm = strdup(s); norm_len = strlen(s); }
    if (!norm) { fprintf(stderr, "acestep text: out of memory\n"); exit(1); }
    int cap = (int)norm_len + 16;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) { fprintf(stderr, "acestep text: out of memory\n"); exit(1); }
    int n = tok_encode(&t->tok, norm, (int)norm_len, ids, cap);
    int keep = n < max_length - 1 ? n : max_length - 1;
    memcpy(out, ids, (size_t)keep * sizeof(int));
    out[keep++] = t->eot;
    free(ids); free(norm);
    return keep;
}

/* NULL, empty, or only whitespace (Python's str.strip() leaves nothing) */
static int act_blank(const char *s){
    if (!s) return 1;
    for (; *s; s++) if (!(*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r' || *s == '\v' || *s == '\f')) return 0;
    return 1;
}

/* The caption prompt. bpm, key and ts may be NULL or empty (N/A); duration in seconds, cut to a whole number. malloc'd. */
static char *act_caption_prompt(const char *instruction, const char *caption, const char *bpm, const char *key, const char *ts,
                                double duration){
    const char *ins = instruction && *instruction ? instruction : ACT_DEFAULT_INSTRUCTION;
    size_t il = strlen(ins);
    int colon = il > 0 && ins[il - 1] == ':';                     /* upstream makes the instruction end with a colon */
    const char *cap = caption ? caption : "";
    /* upstream: bpm if bpm else N/A; the key and the time signature count as missing when blank after .strip() */
    const char *b = bpm && *bpm ? bpm : "N/A", *k = act_blank(key) ? "N/A" : key, *s = act_blank(ts) || !strcmp(ts, "N/A") ? "N/A" : ts;
    size_t cap_n = strlen(ins) + strlen(cap) + strlen(b) + strlen(k) + strlen(s) + 256;
    char *out = (char *)malloc(cap_n);
    if (!out) { fprintf(stderr, "acestep text: out of memory\n"); exit(1); }
    snprintf(out, cap_n, "# Instruction\n%s%s\n\n# Caption\n%s\n\n# Metas\n- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n<|endoftext|>\n",
             ins, colon ? "" : ":", cap, b, s, k, (int)duration);
    return out;
}

/* The lyric prompt. malloc'd. */
static char *act_lyric_prompt(const char *lyrics, const char *language){
    const char *ly = lyrics ? lyrics : "", *lang = language ? language : "";
    size_t n = strlen(ly) + strlen(lang) + 64;
    char *out = (char *)malloc(n);
    if (!out) { fprintf(stderr, "acestep text: out of memory\n"); exit(1); }
    snprintf(out, n, "# Languages\n%s\n\n# Lyric\n%s<|endoftext|>", lang, ly);
    return out;
}

/* ------------------------------------------------------------------ the encoder -- */

/* Qwen3-Embedding-0.6B as upstream runs it: AutoModel -> Qwen3Model, the last hidden state. Plain Qwen3: an embedding
 * table, then layers of (RMSNorm, grouped-query attention with per-head RMS-normed q and k, RoPE, causal; RMSNorm, SwiGLU),
 * each with its residual, then a final RMSNorm. The layers have the shape of the DiT's condition-encoder layers, so they
 * are loaded and computed with the same pieces (acestep_dit.h), the one difference being that the attention is causal. */
typedef struct {
    int hidden, inter, nl, heads, kv, hd, vocab, f32;
    float eps, theta;
    void *embed;                    /* [vocab][hidden], bf16 (or f32 when the file is) */
    float *norm;
    AcdEncLayer *L;
    size_t wbytes;
} AceText;

static void act_free(AceText *m){
    if (!m) return;
    free(m->embed); free(m->norm);
    for (int i = 0; m->L && i < m->nl; i++) {
        AcdEncLayer *e = &m->L[i];
        acd_free_attn(&e->sa); free(e->ln1); free(e->ln2); acd_free_lin(&e->gate); acd_free_lin(&e->up); acd_free_lin(&e->down);
    }
    free(m->L); free(m);
}

/* model_dir holds Qwen3-Embedding-0.6B/ (config.json, model.safetensors). */
static AceText *act_load(const char *model_dir){
    char err[512] = "", path[4200], dir[4096];
    AceText *m = (AceText *)calloc(1, sizeof *m);
    shards S; int have_s = 0;
    jval *cfg = NULL; char *text = NULL;
    if (!m) { fprintf(stderr, "acestep text: out of memory\n"); return NULL; }
    if (snprintf(dir, sizeof dir, "%s/Qwen3-Embedding-0.6B", model_dir) >= (int)sizeof dir) { snprintf(err, sizeof err, "path too long"); goto fail; }
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, sizeof err, "cannot open %.400s", path); goto fail; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    text = (char *)malloc(n > 0 ? n + 1 : 1);
    if (!text || n <= 0 || n > (8 << 20) || fread(text, 1, n, f) != (size_t)n) { fclose(f); snprintf(err, sizeof err, "cannot read %.400s", path); goto fail; }
    fclose(f); text[n] = 0;
    cfg = json_parse_checked(text);
    if (!cfg || cfg->t != J_OBJ) { snprintf(err, sizeof err, "%.400s is not a JSON object", path); goto fail; }

    jval *mt = json_get(cfg, "model_type");
    if (mt && mt->t == J_STR && strcmp(mt->str, "qwen3")) { snprintf(err, sizeof err, "config: model_type is %.40s, this is the Qwen3 encoder", mt->str); goto fail; }
    m->hidden = acd_json_int(cfg, "hidden_size", 0);  m->inter = acd_json_int(cfg, "intermediate_size", 0);
    m->nl = acd_json_int(cfg, "num_hidden_layers", 0); m->heads = acd_json_int(cfg, "num_attention_heads", 0);
    m->kv = acd_json_int(cfg, "num_key_value_heads", 0); m->hd = acd_json_int(cfg, "head_dim", 0);
    m->vocab = acd_json_int(cfg, "vocab_size", 0);
    m->eps = (float)acd_json_num(cfg, "rms_norm_eps", 1e-6);
    m->theta = (float)acd_json_num(cfg, "rope_theta", 10000.0);
    jval *rs = json_get(cfg, "rope_scaling"), *ab = json_get(cfg, "attention_bias"), *sw = json_get(cfg, "use_sliding_window");
    if (rs && rs->t != J_NULL) { snprintf(err, sizeof err, "config: rope_scaling is not supported"); goto fail; }
    if (ab && ab->t == J_BOOL && ab->boolean) { snprintf(err, sizeof err, "config: attention_bias is not supported"); goto fail; }
    if (sw && sw->t == J_BOOL && sw->boolean) { snprintf(err, sizeof err, "config: use_sliding_window is not supported"); goto fail; }
    jval *lt = json_get(cfg, "layer_types");
    for (int i = 0; lt && lt->t == J_ARR && i < lt->len; i++)
        if (lt->kids[i]->t == J_STR && strcmp(lt->kids[i]->str, "full_attention")) { snprintf(err, sizeof err, "config: layer_types[%d] is %.40s, only full attention is supported", i, lt->kids[i]->str); goto fail; }
    if (m->hidden < 1 || m->hidden > ACD_MAX_DIM || m->inter < 1 || m->inter > 4 * ACD_MAX_DIM || m->nl < 1 || m->nl > ACD_MAX_LAYERS ||
        m->heads < 1 || m->heads > ACD_MAX_HEADS || m->kv < 1 || m->heads % m->kv || m->hd < 2 || m->hd > ACD_MAX_HD || (m->hd & 1) ||
        m->vocab < 1 || m->vocab > (1 << 22) || (int64_t)m->heads * m->hd > ACD_MAX_DIM * 2) {
        snprintf(err, sizeof err, "config: a size is missing or out of range"); goto fail; }

    st_init(&S, dir); have_s = 1;
    if (S.n == 0) { snprintf(err, sizeof err, "no safetensors in %.400s", dir); goto fail; }
    const int D = m->hidden;
    int64_t shD[1] = {D};
    st_tensor *te = st_find(&S, "embed_tokens.weight");
    if (!te) { snprintf(err, sizeof err, "missing tensor embed_tokens.weight"); goto fail; }
    if (te->rank != 2 || te->shape[0] != m->vocab || te->shape[1] != D) {
        snprintf(err, sizeof err, "tensor embed_tokens.weight is %lldx%lld, expected %dx%d", (long long)(te->rank > 0 ? te->shape[0] : 0), (long long)(te->rank > 1 ? te->shape[1] : 0), m->vocab, D);
        goto fail; }
    if (te->dtype == 0) {
        m->embed = malloc((size_t)m->vocab * D * 2);
        if (!m->embed) { snprintf(err, sizeof err, "out of memory"); goto fail; }
        st_read_raw(&S, "embed_tokens.weight", m->embed, 1); m->wbytes += (size_t)m->vocab * D * 2;
    } else if (te->dtype == 2 || te->dtype == 1) {
        m->embed = malloc((size_t)m->vocab * D * sizeof(float)); m->f32 = 1;
        if (!m->embed) { snprintf(err, sizeof err, "out of memory"); goto fail; }
        st_read_f32_cap(&S, "embed_tokens.weight", (float *)m->embed, (int64_t)m->vocab * D, 1); m->wbytes += (size_t)m->vocab * D * 4;
    } else { snprintf(err, sizeof err, "tensor embed_tokens.weight is %s, not a float tensor", st_dtype_name(te->dtype)); goto fail; }
    if (!(m->norm = acd_read_f32(&S, "norm.weight", 1, shD, err, sizeof err))) goto fail;
    m->L = (AcdEncLayer *)calloc((size_t)m->nl, sizeof(AcdEncLayer));
    if (!m->L) { snprintf(err, sizeof err, "out of memory"); goto fail; }
    for (int i = 0; i < m->nl; i++) {
        char p[256], nm[256];
        AcdEncLayer *e = &m->L[i];
        snprintf(p, sizeof p, "layers.%d.self_attn", i);
        if (acd_load_attn(&S, p, D, m->heads, m->kv, m->hd, &e->sa, &m->wbytes, err, sizeof err)) goto fail;
        snprintf(nm, sizeof nm, "layers.%d.input_layernorm.weight", i);          if (!(e->ln1 = acd_read_f32(&S, nm, 1, shD, err, sizeof err))) goto fail;
        snprintf(nm, sizeof nm, "layers.%d.post_attention_layernorm.weight", i); if (!(e->ln2 = acd_read_f32(&S, nm, 1, shD, err, sizeof err))) goto fail;
        snprintf(p, sizeof p, "layers.%d.mlp", i);
        if (acd_load_mlp(&S, p, D, m->inter, &e->gate, &e->up, &e->down, &m->wbytes, err, sizeof err)) goto fail;
    }
    st_destroy(&S);
    json_free(cfg); free(text);
    return m;
fail:
    fprintf(stderr, "acestep text: %s\n", err[0] ? err : "load failed");
    if (have_s) st_destroy(&S);
    json_free(cfg); free(text);
    act_free(m);
    return NULL;
}

/* The embedding table's rows for ids [n]: [n][hidden] (malloc). An id outside the table is refused (NULL). The lyric
 * path of the condition is exactly this and nothing else. */
static float *act_embed(const AceText *m, const int *ids, int n){
    float *x = (float *)malloc((size_t)(n > 0 ? n : 1) * m->hidden * sizeof(float));
    if (!x) { fprintf(stderr, "acestep text: out of memory\n"); exit(1); }
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= m->vocab) { fprintf(stderr, "acestep text: token id %d is outside the vocabulary (%d)\n", ids[i], m->vocab); free(x); return NULL; }
        float *row = x + (size_t)i * m->hidden;
        if (m->f32) memcpy(row, (const float *)m->embed + (size_t)ids[i] * m->hidden, (size_t)m->hidden * sizeof(float));
        else { const uint16_t *src = (const uint16_t *)m->embed + (size_t)ids[i] * m->hidden; for (int j = 0; j < m->hidden; j++) row[j] = bf16_to_f32(src[j]); }
    }
    return x;
}

/* One layer on x [T][hidden], in place; cs/sn the RoPE table for T positions. */
static void act_layer(const AceText *m, const AcdEncLayer *l, float *x, int T, const float *cs, const float *sn){
    const int D = m->hidden, H = m->heads, Hk = m->kv, hd = m->hd;
    float *n = (float *)malloc((size_t)T * D * sizeof(float)), *o = (float *)malloc((size_t)T * D * sizeof(float));
    float *q = (float *)malloc((size_t)T * H * hd * sizeof(float)), *k = (float *)malloc((size_t)T * Hk * hd * sizeof(float));
    float *v = (float *)malloc((size_t)T * Hk * hd * sizeof(float)), *cx = (float *)malloc((size_t)T * H * hd * sizeof(float));
    float *g = (float *)malloc((size_t)T * m->inter * sizeof(float)), *u = (float *)malloc((size_t)T * m->inter * sizeof(float));
    if (!n || !o || !q || !k || !v || !cx || !g || !u) { fprintf(stderr, "acestep text: out of memory\n"); exit(1); }
    acd_rmsnorm(n, x, T, D, l->ln1, m->eps);
    acd_lin(q, n, T, &l->sa.q); acd_lin(k, n, T, &l->sa.k); acd_lin(v, n, T, &l->sa.v);
    acd_head_norm(q, T, H, hd, l->sa.qn, m->eps);
    acd_head_norm(k, T, Hk, hd, l->sa.kn, m->eps);
    acd_rope(q, T, H, hd, cs, sn);
    acd_rope(k, T, Hk, hd, cs, sn);
    acd_attention_ex(cx, H * hd, q, H * hd, T, k, v, Hk * hd, T, H, Hk, hd, -1, 1);          /* causal */
    acd_lin(o, cx, T, &l->sa.o);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * D; i++) x[i] += o[i];
    acd_rmsnorm(n, x, T, D, l->ln2, m->eps);
    acd_lin(g, n, T, &l->gate); acd_lin(u, n, T, &l->up);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * m->inter; i++) g[i] = acd_silu(g[i]) * u[i];
    acd_lin(o, g, T, &l->down);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * D; i++) x[i] += o[i];
    free(n); free(o); free(q); free(k); free(v); free(cx); free(g); free(u);
}

/* The last hidden state of the encoder for ids [n]: [n][hidden] (malloc), or NULL for a bad id. */
static float *act_encode(const AceText *m, const int *ids, int n){
    float *x = act_embed(m, ids, n);
    if (!x) return NULL;
    float *cs = (float *)malloc((size_t)n * m->hd * sizeof(float)), *sn = (float *)malloc((size_t)n * m->hd * sizeof(float));
    if (!cs || !sn) { fprintf(stderr, "acestep text: out of memory\n"); exit(1); }
    acd_rope_table(cs, sn, n, m->hd, m->theta);
    for (int i = 0; i < m->nl; i++) act_layer(m, &m->L[i], x, n, cs, sn);
    acd_rmsnorm(x, x, n, m->hidden, m->norm, m->eps);
    free(cs); free(sn);
    return x;
}

#endif
