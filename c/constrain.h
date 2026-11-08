/* constrain.h -- constrained decoding: a grammar turned into a limit on which token may come next.
 *
 * The other grammar user in this tree, the grammar-forced draft source (grammar.h, docs/grammar-draft.md), never limits what
 * the model says: it drafts the bytes a grammar forces and the model verifies them, so the output is whatever the model
 * would have said anyway. This is the other thing: a request that carries a grammar (response_format json_object,
 * json_schema or gbnf) gets only tokens whose bytes keep the output inside it, so the answer is valid by construction.
 *
 * One Cn per request. Per step the engine, holding the logits of the row it is about to sample:
 *
 *     int tok = cn_pick(cn, vocab, lo, sampler, ctx);   // masks lo, samples, checks, advances
 *
 * cn_pick masks every token whose first byte the grammar cannot take (one pass over the vocabulary), then lets the engine's own
 * sampler (greedy, temperature, top-p, whatever it uses) choose among what is left. A token of several bytes can still run off
 * the grammar in its second byte, so the choice is checked byte by byte on a copy of the walker, and a token that fails is
 * removed and the sampler asked again: sampling with rejection is sampling from the model's distribution restricted to the
 * tokens that are valid. If the sampler keeps failing, the best valid token by logit is found by scanning. EOS (and any token
 * that ends the answer) is allowed only where the grammar can end, and once nothing but the end can follow, it is all that is
 * left.
 *
 * Return values of cn_pick: the token; -1 if no token of this vocabulary can continue the grammar (it asks for a byte no token
 * carries); -2 if the walker was lost (grammar.h switches itself off when a grammar is too ambiguous or too deep). Either way
 * the engine ends the request with an error: continuing without the constraint would hand back the thing it promised not to.
 *
 * What this does not promise: that the answer FINISHES (max_tokens can cut it short; the engine reports that as it always
 * does), and that bytes the grammar leaves free (a JSON string may hold any byte of 0x20 and up) form UTF-8 -- the model's
 * tokens do. Thinking is not constrained: the gateway refuses a grammar on a request that thinks.
 */
#ifndef COLI_CONSTRAIN_H
#define COLI_CONSTRAIN_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "grammar.h"
#include "schema_gbnf.h"

#define CN_TRIES 64                       /* sampler attempts before falling back to a scan for the best valid token */

enum { CN_TEXT = 0, CN_END = 1, CN_SPECIAL = 2 };   /* a token: carries bytes / ends the answer / may never be chosen */

/* What each token adds to the output, built once per engine. */
typedef struct {
    int V;
    unsigned char *pool;                  /* every token's bytes, back to back */
    size_t pool_n, pool_cap;
    int *off, *len;                       /* per token */
    unsigned char *kind;                  /* CN_TEXT, CN_END or CN_SPECIAL */
} CnVocab;

static CnVocab *cn_vocab_new(int V){
    CnVocab *v = (CnVocab *)calloc(1, sizeof *v);
    if (!v) return NULL;
    v->V = V;
    v->off = (int *)calloc((size_t)V, sizeof(int));
    v->len = (int *)calloc((size_t)V, sizeof(int));
    v->kind = (unsigned char *)calloc((size_t)V, 1);
    if (!v->off || !v->len || !v->kind){ free(v->off); free(v->len); free(v->kind); free(v); return NULL; }
    for (int t = 0; t < V; t++) v->kind[t] = CN_SPECIAL;     /* until the engine says what a token is */
    return v;
}

static void cn_vocab_set(CnVocab *v, int t, const unsigned char *b, int n, int kind){
    if (t < 0 || t >= v->V) return;
    v->kind[t] = (unsigned char)kind;
    v->len[t] = 0;
    if (kind != CN_TEXT || n <= 0) return;
    if (v->pool_n + (size_t)n > v->pool_cap){
        size_t cap = v->pool_cap ? v->pool_cap * 2 : 1 << 20;
        while (cap < v->pool_n + (size_t)n) cap *= 2;
        unsigned char *p = (unsigned char *)realloc(v->pool, cap);
        if (!p){ v->kind[t] = CN_SPECIAL; return; }
        v->pool = p; v->pool_cap = cap;
    }
    memcpy(v->pool + v->pool_n, b, (size_t)n);
    v->off[t] = (int)v->pool_n; v->len[t] = n; v->pool_n += (size_t)n;
}

static void cn_vocab_free(CnVocab *v){
    if (!v) return;
    free(v->pool); free(v->off); free(v->len); free(v->kind); free(v);
}

/* One request's walker. */
typedef struct { Grammar *g; GrState st; int done; } Cn;

/* payload: a JSON schema when it starts with '{' (the convention of the grammar draft source), else GBNF text with a `root`
 * rule. NULL, with a message in err, when it does not compile, when the walker cannot start, or when a schema uses something
 * schema_gbnf.h does not support (there the draft source runs without a grammar; a constraint cannot, so it is refused). */
static Cn *cn_new(const char *payload, char *err, int errsz){
    const char *p = payload;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    char *owned = NULL;
    const char *text = payload;
    if (*p == '{'){
        owned = schema_to_gbnf(payload, err, errsz);
        if (!owned) return NULL;
        text = owned;
    }
    Cn *c = (Cn *)calloc(1, sizeof *c);
    if (c) c->g = (Grammar *)calloc(1, sizeof *c->g);
    if (!c || !c->g){ free(c); free(owned); if (err) snprintf(err, errsz, "out of memory"); return NULL; }
    if (gr_parse(c->g, text) != 0){
        if (err) snprintf(err, errsz, "grammar: %.120s", c->g->err[0] ? c->g->err : "does not parse");
        free(owned); gr_free(c->g); free(c->g); free(c); return NULL;
    }
    free(owned);
    gr_state_init(&c->st, c->g);
    if (!c->st.alive){
        if (err) snprintf(err, errsz, "grammar: too ambiguous to follow");
        gr_free(c->g); free(c->g); free(c); return NULL;
    }
    return c;
}

static void cn_free(Cn *c){
    if (!c) return;
    gr_free(c->g); free(c->g); free(c);
}

/* Set to -inf the logit of every token that cannot be next. */
static void cn_mask(const Cn *c, const CnVocab *v, float *lo){
    unsigned char m[32]; int end = 0;
    gr_admissible(&c->st, m, &end);
    for (int t = 0; t < v->V; t++){
        unsigned char k = v->kind[t];
        if (k == CN_END){ if (!end) lo[t] = -INFINITY; continue; }
        if (k != CN_TEXT || v->len[t] == 0){ lo[t] = -INFINITY; continue; }
        unsigned char b = v->pool[v->off[t]];
        if (!(m[b >> 3] & (1u << (b & 7)))) lo[t] = -INFINITY;
    }
}

/* Would the token keep the output inside the grammar? Does not move the walker. */
static int cn_valid(const Cn *c, const CnVocab *v, int t){
    if (t < 0 || t >= v->V || !c->st.alive) return 0;
    unsigned char k = v->kind[t];
    if (k == CN_END){ unsigned char m[32]; int end = 0; gr_admissible(&c->st, m, &end); return end; }
    if (k != CN_TEXT || v->len[t] == 0) return 0;
    GrState cp = c->st;
    for (int i = 0; i < v->len[t]; i++)
        if (gr_accept(&cp, v->pool[v->off[t] + i]) != 1) return 0;
    return 1;
}

/* Take the token: the walker moves past its bytes, and an ending token closes the answer. */
static int cn_advance(Cn *c, const CnVocab *v, int t){
    if (v->kind[t] == CN_END){ c->done = 1; return 1; }
    for (int i = 0; i < v->len[t]; i++)
        if (gr_accept(&c->st, v->pool[v->off[t] + i]) != 1) return 0;
    return 1;
}

typedef int (*CnSampler)(const float *lo, void *ctx);

static int cn_pick(Cn *c, const CnVocab *v, float *lo, CnSampler sample, void *ctx){
    if (!c->st.alive) return -2;
    cn_mask(c, v, lo);
    for (int tries = 0; tries < CN_TRIES; tries++){
        int t = sample(lo, ctx);
        if (t < 0 || t >= v->V || !(lo[t] > -INFINITY)) break;      /* nothing finite is left for the sampler */
        if (cn_valid(c, v, t)){
            if (!cn_advance(c, v, t)) return -2;
            return t;
        }
        lo[t] = -INFINITY;
    }
    for (;;){                                                       /* the best valid token by logit, by scanning */
        int best = -1;
        for (int t = 0; t < v->V; t++) if (lo[t] > -INFINITY && (best < 0 || lo[t] > lo[best])) best = t;
        if (best < 0) return -1;
        if (cn_valid(c, v, best)){
            if (!cn_advance(c, v, best)) return -2;
            return best;
        }
        lo[best] = -INFINITY;
    }
}

#endif /* COLI_CONSTRAIN_H */
