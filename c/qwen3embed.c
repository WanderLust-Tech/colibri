/* qwen3embed.c -- text embeddings from a Qwen3-Embedding checkpoint (Qwen/Qwen3-Embedding-0.6B, -4B, -8B), and relevance
 * scores from a Qwen3-Reranker checkpoint (Qwen/Qwen3-Reranker-0.6B, -4B, -8B; see "Reranking" below). The checkpoint
 * decides which of the two this process is: a folder with 1_LogitScore/config.json is a reranker.
 *
 * What the model card does, and so what this does:
 *
 *   text  ->  Qwen's byte-level BPE (the checkpoint's tokenizer.json; the post-processor appends <|endoftext|>)
 *         ->  the whole decoder (plain Qwen3: RMSNorm, grouped-query attention with per-head RMS-normed q and k, RoPE,
 *             causal, SwiGLU; a final RMSNorm), no language-model head
 *         ->  the last token's hidden state (last-token pooling: that is the appended <|endoftext|>)
 *         ->  L2 normalised.
 *
 *   A query is written "Instruct: {task}\nQuery:{text}" and a document is the text as it is; --instruction (or "instruction"
 *   in a request) makes the first form. The model was trained with that prefix on queries (the card: 1 to 5 % better).
 *   "dimensions" keeps the first N components and normalises again (the models are trained for that, 32 up to the width).
 *
 * The layers are the ones ACE-Step's text encoder runs (acestep_text.h loads them; that encoder IS Qwen3-Embedding-0.6B).
 * Here several texts go through the layers together: the matrix products run over all their rows at once (a weight is read
 * once for the whole batch, which is most of what a short text costs), RoPE and the attention run inside each text.
 *
 * Usage:
 *   qwen3embed --model DIR --text TEXT [--instruction TASK] [--dimensions N] [--threads N]
 *       prints one VECTORS line for the text.
 *   qwen3embed --model DIR --serve
 *       the line protocol, on stdin and stdout (docs/embeddings.md; a reranker's is below):
 *         READY {...}               once, after the weights are loaded: model, dimensions, max_tokens, layers
 *         EMBED {...}               client: id, inputs (array of strings), instruction, dimensions, truncate (default true)
 *         VECTORS {...}             id, dimensions, tokens [n per input], truncated [0|1 per input], vectors [[...] per input]
 *         ERROR {"id","message"}    the engine stays alive for the next EMBED
 *       Requests run one at a time, in the order they arrive. EOF on stdin ends the process. Nothing is written to stdout
 *       but these lines. A line may be up to 64 MiB.
 *
 * Reranking (docs/rerank.md). The same decoder, with its language-model head read for two tokens: the pair is written as
 *   <|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note
 *   that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n<Instruct>: {task}\n<Query>: {query}\n
 *   <Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 * and the score is the chance of "yes" against "no" at the last position: sigmoid(logit(yes) - logit(no)), which is what the
 * model card computes. Only the two rows of the head that matter are read (lm_head.weight, or the embedding table when the
 * weights are tied).
 *   qwen3embed --model DIR --query Q --text DOC [--instruction TASK]     prints one SCORES line
 *   protocol:  READY {..., "task":"rerank"}   RERANK {id, query, documents, instruction, truncate}
 *              SCORES {id, scores [0..1 per document], logits [yes minus no], tokens, truncated}   or ERROR */
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

#define QE_MAX_LINE ((size_t)64 << 20)
#define QE_MAX_INPUTS 2048
#define QE_DEFAULT_MAX_TOKENS 8192
#define QE_SMALL 2048               /* texts up to this many tokens use the one-region attention; longer ones the blocked routine */
#define QE_BATCH_ROWS 4096          /* tokens through the layers at once: bounds the working memory (a few times rows x the MLP width) */

#define QE_DEFAULT_TASK "Given a web search query, retrieve relevant passages that answer the query"
#define QE_RERANK_PREFIX "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
#define QE_RERANK_SUFFIX "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

typedef struct {
    AceText *m;
    AceTok tok;
    int max_tokens;
    char name[160];
    int rerank;                     /* a Qwen3-Reranker checkpoint */
    float *head;                    /* rerank: the output rows of "no" [0] and "yes" [1], [2][hidden] */
    int *prefix, nprefix, *suffix, nsuffix;     /* rerank: the fixed ends of every pair, as token ids */
} QeEngine;

static void qe_escape(const char *s){
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

static void qe_error(const char *id, const char *message){
    printf("ERROR {\"id\":\""); qe_escape(id ? id : "");
    printf("\",\"message\":\""); qe_escape(message); printf("\"}\n");
    fflush(stdout);
}

static void qe_close(QeEngine *e){
    if (!e) return;
    act_free(e->m);
    act_tok_close(&e->tok);
    free(e->head); free(e->prefix); free(e->suffix);
    free(e);
}

/* Token ids of text with no sentence-end token added (the reranker's pieces; act_tokenize appends <|endoftext|>), malloc'd. */
static int *qe_bpe(QeEngine *e, const char *text, int *n){
    char *norm = NULL; size_t norm_len = 0;
    if (q38_nfc_normalize(text, strlen(text), &norm, &norm_len) != 0) { norm = strdup(text); norm_len = strlen(text); }
    if (!norm) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    int cap = (int)norm_len + 16;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    *n = tok_encode(&e->tok.tok, norm, (int)norm_len, ids, cap);
    free(norm);
    return ids;
}

/* the two token ids a reranker answers with: 1_LogitScore/config.json (true_token_id, false_token_id), else the tokens
 * "yes" and "no" */
static void qe_answer_ids(QeEngine *e, const char *dir, int *yes, int *no){
    char path[4300];
    *yes = *no = -1;
    snprintf(path, sizeof path, "%s/1_LogitScore/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (f) {
        char buf[2048]; size_t n = fread(buf, 1, sizeof buf - 1, f); fclose(f); buf[n] = 0;
        jval *j = json_parse_checked(buf);
        if (j && j->t == J_OBJ) {
            jval *t = json_get(j, "true_token_id"), *u = json_get(j, "false_token_id");
            if (t && t->t == J_NUM) *yes = (int)t->num;
            if (u && u->t == J_NUM) *no = (int)u->num;
        }
        if (j) json_free(j);
    }
    if (*yes < 0) *yes = tok_id_of(&e->tok.tok, "yes");
    if (*no < 0) *no = tok_id_of(&e->tok.tok, "no");
}

static int qe_is_rerank_dir(const char *dir){
    char path[4300];
    snprintf(path, sizeof path, "%s/1_LogitScore/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

static QeEngine *qe_open(const char *dir, char *err, size_t errn){
    QeEngine *e = (QeEngine *)calloc(1, sizeof *e);
    char path[4300];
    if (!e) { snprintf(err, errn, "out of memory"); return NULL; }
    if (snprintf(path, sizeof path, "%s/tokenizer.json", dir) >= (int)sizeof path) { snprintf(err, errn, "path too long"); free(e); return NULL; }
    if (act_tok_open_file(&e->tok, path)) { snprintf(err, errn, "cannot open the tokenizer in %.400s", dir); free(e); return NULL; }
    e->rerank = qe_is_rerank_dir(dir);
    if (e->rerank) {
        int yes, no;
        qe_answer_ids(e, dir, &yes, &no);
        if (yes < 0 || no < 0) { snprintf(err, errn, "the tokenizer has no \"yes\" and \"no\" tokens"); act_tok_close(&e->tok); free(e); return NULL; }
        int rows[2] = {no, yes};
        e->m = act_load_impl(dir, rows, 2, &e->head);
        if (e->m) {
            e->prefix = qe_bpe(e, QE_RERANK_PREFIX, &e->nprefix);
            e->suffix = qe_bpe(e, QE_RERANK_SUFFIX, &e->nsuffix);
        }
    } else e->m = act_load_dir(dir);
    if (!e->m) { snprintf(err, errn, "cannot load the weights in %.400s", dir); act_tok_close(&e->tok); free(e->head); free(e); return NULL; }
    e->max_tokens = QE_DEFAULT_MAX_TOKENS;
    const char *base = strrchr(dir, '/');
    const char *bs = strrchr(dir, '\\');
    if (bs && (!base || bs > base)) base = bs;
    snprintf(e->name, sizeof e->name, "%s", base ? base + 1 : dir);
    return e;
}

/* The token ids of one text (malloc'd, *ntok of them), with the instruction prefix when there is one. *cut is 1 when the text
 * was longer than the limit and was shortened. act_tokenize keeps max_length - 1 text tokens and then the appended
 * <|endoftext|>, so asking for limit + 1 says whether the whole sequence would not fit. */
static int *qe_tokens(QeEngine *e, const char *text, const char *instruction, int limit, int *ntok, int *cut){
    char *full = NULL;
    const char *src = text;
    if (instruction && *instruction) {
        size_t n = strlen(instruction) + strlen(text) + 24;
        full = (char *)malloc(n);
        if (!full) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
        snprintf(full, n, "Instruct: %s\nQuery:%s", instruction, text);
        src = full;
    }
    int *ids = (int *)malloc(((size_t)limit + 2) * sizeof(int));
    if (!ids) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    int n = act_tokenize(&e->tok, src, limit + 1, ids);
    *cut = n > limit;
    if (*cut) { n = limit; ids[n - 1] = e->tok.eot; }                 /* keep the shape of an uncut text: its last token is <|endoftext|> */
    free(full);
    *ntok = n;
    return ids;
}

/* Causal attention for the texts of a batch that are at most QE_SMALL tokens (which[0..nwhich) are their numbers): every
 * (text, head, block of 16 queries) is one unit of work, all of them in one parallel region. The blocked routine of the DiT
 * (acd_attention_ex) forks and joins for every head, which costs more than the attention of a short text; but it streams the
 * keys through matrix products, which wins once a text is a few thousand tokens, so those still go to it (qe_layer).
 * q is [total][H*hd], k and v [total][Hk*hd]; offs[s] is where text s starts. */
#define QE_QBLOCK 16
static void qe_attention(float *cx, const float *q, const float *k, const float *v, const int *lens, const int *offs,
                         const int *which, int nwhich, int H, int Hk, int hd){
    const int group = H / Hk;
    const float scale = 1.f / sqrtf((float)hd);
    int64_t units = 0;
    for (int a = 0; a < nwhich; a++) units += (int64_t)H * ((lens[which[a]] + QE_QBLOCK - 1) / QE_QBLOCK);
    #pragma omp parallel
    {
        float *sc = (float *)malloc((size_t)QE_SMALL * sizeof(float));
        if (!sc) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
        #pragma omp for schedule(dynamic, 1)
        for (int64_t u = 0; u < units; u++) {
            int64_t rest = u;                                       /* decode the unit: text, then head, then block */
            int s = 0;
            for (int a = 0;; a++) {
                s = which[a];
                const int64_t n = (int64_t)H * ((lens[s] + QE_QBLOCK - 1) / QE_QBLOCK);
                if (rest < n) break;
                rest -= n;
            }
            const int blocks = (lens[s] + QE_QBLOCK - 1) / QE_QBLOCK;
            const int h = (int)(rest / blocks), i0 = (int)(rest % blocks) * QE_QBLOCK;
            const int T = lens[s], off = offs[s], kh = h / group;
            const int i1 = i0 + QE_QBLOCK < T ? i0 + QE_QBLOCK : T;
            for (int i = i0; i < i1; i++) {
                const float *qi = q + (size_t)(off + i) * H * hd + (size_t)h * hd;
                float mx = -INFINITY;
                for (int j = 0; j <= i; j++) {
                    const float *kj = k + (size_t)(off + j) * Hk * hd + (size_t)kh * hd;
                    float d = 0.f;
                    #pragma omp simd reduction(+:d)
                    for (int t = 0; t < hd; t++) d += qi[t] * kj[t];
                    d *= scale; sc[j] = d;
                    if (d > mx) mx = d;
                }
                double sum = 0;
                for (int j = 0; j <= i; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                const float inv = (float)(1.0 / sum);
                float *o = cx + (size_t)(off + i) * H * hd + (size_t)h * hd;
                for (int t = 0; t < hd; t++) o[t] = 0.f;
                for (int j = 0; j <= i; j++) {
                    const float pj = sc[j] * inv;
                    const float *vj = v + (size_t)(off + j) * Hk * hd + (size_t)kh * hd;
                    #pragma omp simd
                    for (int t = 0; t < hd; t++) o[t] += pj * vj[t];
                }
            }
        }
        free(sc);
    }
}

/* One layer on x [total][hidden] where the rows are several sequences one after another (lens[nseq], starting at offs[]): the
 * matrix products run over all the rows at once; RoPE runs over all of them too, with a table whose positions start again
 * at each sequence (cs and sn have `total` rows); the causal attention runs inside each sequence. act_layer is the
 * one-sequence version of this. */
static void qe_layer(const AceText *m, const AcdEncLayer *l, float *x, int total, const int *lens, const int *offs, int nseq, const float *cs, const float *sn){
    const int D = m->hidden, H = m->heads, Hk = m->kv, hd = m->hd;
    float *n = (float *)malloc((size_t)total * D * sizeof(float)), *o = (float *)malloc((size_t)total * D * sizeof(float));
    float *q = (float *)malloc((size_t)total * H * hd * sizeof(float)), *k = (float *)malloc((size_t)total * Hk * hd * sizeof(float));
    float *v = (float *)malloc((size_t)total * Hk * hd * sizeof(float)), *cx = (float *)malloc((size_t)total * H * hd * sizeof(float));
    float *g = (float *)malloc((size_t)total * m->inter * sizeof(float)), *u = (float *)malloc((size_t)total * m->inter * sizeof(float));
    if (!n || !o || !q || !k || !v || !cx || !g || !u) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    acd_rmsnorm(n, x, total, D, l->ln1, m->eps);
    acd_lin(q, n, total, &l->sa.q); acd_lin(k, n, total, &l->sa.k); acd_lin(v, n, total, &l->sa.v);
    acd_head_norm(q, total, H, hd, l->sa.qn, m->eps);
    acd_head_norm(k, total, Hk, hd, l->sa.kn, m->eps);
    acd_rope(q, total, H, hd, cs, sn);
    acd_rope(k, total, Hk, hd, cs, sn);
    int *shortseq = (int *)malloc((size_t)nseq * sizeof(int)), nshort = 0;
    if (!shortseq) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    for (int s = 0; s < nseq; s++) {
        if (lens[s] <= QE_SMALL) { shortseq[nshort++] = s; continue; }
        acd_attention_ex(cx + (size_t)offs[s] * H * hd, H * hd, q + (size_t)offs[s] * H * hd, H * hd, lens[s],
                         k + (size_t)offs[s] * Hk * hd, v + (size_t)offs[s] * Hk * hd, Hk * hd, lens[s], H, Hk, hd, -1, 1);
    }
    if (nshort) qe_attention(cx, q, k, v, lens, offs, shortseq, nshort, H, Hk, hd);
    free(shortseq);
    acd_lin(o, cx, total, &l->sa.o);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)total * D; i++) x[i] += o[i];
    acd_rmsnorm(n, x, total, D, l->ln2, m->eps);
    acd_lin(g, n, total, &l->gate); acd_lin(u, n, total, &l->up);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)total * m->inter; i++) g[i] = acd_silu(g[i]) * u[i];
    acd_lin(o, g, total, &l->down);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)total * D; i++) x[i] += o[i];
    free(n); free(o); free(q); free(k); free(v); free(cx); free(g); free(u);
}

/* nseq sequences (ids concatenated, lens each) through every layer: the hidden states [total][hidden] before the final norm
 * (malloc'd; NULL for a bad id). */
static float *qe_hidden(const QeEngine *e, const int *ids, const int *lens, int nseq){
    const AceText *m = e->m;
    int total = 0, longest = 0;
    for (int s = 0; s < nseq; s++) { total += lens[s]; if (lens[s] > longest) longest = lens[s]; }
    float *x = act_embed(m, ids, total);
    if (!x) return NULL;
    float *bcs = (float *)malloc((size_t)longest * m->hd * sizeof(float)), *bsn = (float *)malloc((size_t)longest * m->hd * sizeof(float));
    float *cs = (float *)malloc((size_t)total * m->hd * sizeof(float)), *sn = (float *)malloc((size_t)total * m->hd * sizeof(float));
    int *offs = (int *)malloc((size_t)nseq * sizeof(int));
    if (!bcs || !bsn || !cs || !sn || !offs) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    acd_rope_table(bcs, bsn, longest, m->hd, m->theta);
    for (int s = 0, at = 0; s < nseq; at += lens[s], s++) {          /* the position of a row is its place in its own text */
        offs[s] = at;
        memcpy(cs + (size_t)at * m->hd, bcs, (size_t)lens[s] * m->hd * sizeof(float));
        memcpy(sn + (size_t)at * m->hd, bsn, (size_t)lens[s] * m->hd * sizeof(float));
    }
    free(bcs); free(bsn);
    for (int i = 0; i < m->nl; i++) qe_layer(m, &m->L[i], x, total, lens, offs, nseq, cs, sn);
    free(cs); free(sn); free(offs);
    return x;
}

/* The unit vectors of nseq sequences (ids concatenated, lens each): out[nseq][hidden]; 0 on failure (a bad id, a vector that
 * is not finite). A vector is the last token's hidden state after the final norm, normalised. */
static int qe_encode(const QeEngine *e, const int *ids, const int *lens, int nseq, float *out){
    const AceText *m = e->m;
    const int D = m->hidden;
    float *x = qe_hidden(e, ids, lens, nseq);
    if (!x) return 0;
    int ok = 1;
    for (int s = 0, off = 0; s < nseq; off += lens[s], s++) {
        float *v = out + (size_t)s * D;
        acd_rmsnorm(v, x + (size_t)(off + lens[s] - 1) * D, 1, D, m->norm, m->eps);       /* the final norm is per row: only the last needs it */
        double sum = 0;
        for (int i = 0; i < D; i++) sum += (double)v[i] * v[i];
        if (!(sum > 0) || !isfinite(sum)) { ok = 0; break; }
        float inv = (float)(1.0 / sqrt(sum));
        for (int i = 0; i < D; i++) v[i] *= inv;
    }
    free(x);
    return ok;
}

/* the first `dims` components, normalised again (dims == D leaves the vector as it is) */
static void qe_cut(float *v, int D, int dims){
    if (dims >= D) return;
    double s = 0;
    for (int i = 0; i < dims; i++) s += (double)v[i] * v[i];
    float inv = s > 0 ? (float)(1.0 / sqrt(s)) : 0.f;
    for (int i = 0; i < dims; i++) v[i] *= inv;
}

/* Embeds `count` texts and prints the VECTORS line. 0 on success; on failure prints ERROR and returns 1. The texts go through
 * the model in batches of up to QE_BATCH_ROWS tokens, in the order given. */
static int qe_run(QeEngine *e, const char *id, const char **texts, int count, const char *instruction, int dims, int truncate){
    const int D = e->m->hidden;
    if (dims < 1) dims = D;
    if (dims > D) { char m[160]; snprintf(m, sizeof m, "dimensions is %d; this model's vectors are %d wide", dims, D); qe_error(id, m); return 1; }
    int **ids = (int **)calloc((size_t)count, sizeof(int *));
    int *toks = (int *)calloc((size_t)count, sizeof(int)), *cuts = (int *)calloc((size_t)count, sizeof(int));
    float *vecs = (float *)malloc((size_t)count * D * sizeof(float));
    if (!ids || !toks || !cuts || !vecs) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    int rc = 0;
    for (int i = 0; i < count && !rc; i++) {
        ids[i] = qe_tokens(e, texts[i], instruction, e->max_tokens, &toks[i], &cuts[i]);
        if (cuts[i] && !truncate) {
            char m[200];
            snprintf(m, sizeof m, "input %d is longer than %d tokens (send \"truncate\": true to cut it)", i, e->max_tokens);
            qe_error(id, m); rc = 1;
        }
    }
    for (int from = 0; from < count && !rc; ) {
        int to = from, rows = 0;
        while (to < count && (to == from || rows + toks[to] <= QE_BATCH_ROWS)) rows += toks[to++];
        int *flat = (int *)malloc((size_t)rows * sizeof(int));
        if (!flat) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
        for (int i = from, at = 0; i < to; i++) { memcpy(flat + at, ids[i], (size_t)toks[i] * sizeof(int)); at += toks[i]; }
        if (!qe_encode(e, flat, toks + from, to - from, vecs + (size_t)from * D)) { qe_error(id, "the text could not be embedded"); rc = 1; }
        free(flat);
        from = to;
    }
    if (!rc) {
        printf("VECTORS {\"id\":\""); qe_escape(id);
        printf("\",\"dimensions\":%d,\"tokens\":[", dims);
        for (int i = 0; i < count; i++) printf("%s%d", i ? "," : "", toks[i]);
        printf("],\"truncated\":[");
        for (int i = 0; i < count; i++) printf("%s%d", i ? "," : "", cuts[i]);
        printf("],\"vectors\":[");
        for (int i = 0; i < count; i++) {
            float *v = vecs + (size_t)i * D;
            qe_cut(v, D, dims);
            printf("%s[", i ? "," : "");
            for (int j = 0; j < dims; j++) printf("%s%.9g", j ? "," : "", (double)v[j]);
            putchar(']');
        }
        printf("]}\n");
        fflush(stdout);
    }
    for (int i = 0; i < count; i++) free(ids[i]);
    free(ids); free(toks); free(cuts); free(vecs);
    return rc;
}

/* The logit difference (yes minus no) of nseq pairs (ids concatenated, lens each): out[nseq]; 0 on failure. The head is two
 * rows of the output matrix, so a pair costs the layers and then two dot products over the last row. */
static int qe_score(const QeEngine *e, const int *ids, const int *lens, int nseq, double *out){
    const AceText *m = e->m;
    const int D = m->hidden;
    float *x = qe_hidden(e, ids, lens, nseq);
    if (!x) return 0;
    float *h = (float *)malloc((size_t)D * sizeof(float));
    if (!h) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    int ok = 1;
    for (int s = 0, off = 0; s < nseq; off += lens[s], s++) {
        acd_rmsnorm(h, x + (size_t)(off + lens[s] - 1) * D, 1, D, m->norm, m->eps);
        double no = 0, yes = 0;
        for (int i = 0; i < D; i++) { no += (double)h[i] * e->head[i]; yes += (double)h[i] * e->head[(size_t)D + i]; }
        out[s] = yes - no;
        if (!isfinite(out[s])) { ok = 0; break; }
    }
    free(h); free(x);
    return ok;
}

/* Scores `count` documents against one query and prints the SCORES line. 0 on success; on failure prints ERROR and returns 1.
 * Each pair is prefix + the text of the pair cut to what is left of the token limit + suffix (the model card cuts the
 * text, never the two ends); the pairs go through the model in batches, in the order given. */
static int qe_rerank(QeEngine *e, const char *id, const char *query, const char **docs, int count, const char *instruction, int truncate){
    const int room = e->max_tokens - e->nprefix - e->nsuffix;
    if (!instruction || !*instruction) instruction = QE_DEFAULT_TASK;
    int **ids = (int **)calloc((size_t)count, sizeof(int *));
    int *toks = (int *)calloc((size_t)count, sizeof(int)), *cuts = (int *)calloc((size_t)count, sizeof(int));
    double *logits = (double *)malloc((size_t)count * sizeof(double));
    if (!ids || !toks || !cuts || !logits) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    int rc = 0;
    for (int i = 0; i < count && !rc; i++) {
        size_t n = strlen(instruction) + strlen(query) + strlen(docs[i]) + 64;
        char *text = (char *)malloc(n);
        if (!text) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
        snprintf(text, n, "<Instruct>: %s\n<Query>: %s\n<Document>: %s", instruction, query, docs[i]);
        int k;
        int *body = qe_bpe(e, text, &k);
        free(text);
        cuts[i] = k > room;
        if (cuts[i] && !truncate) {
            char m[200];
            snprintf(m, sizeof m, "document %d with the query is longer than %d tokens (send \"truncate\": true to cut it)", i, e->max_tokens);
            qe_error(id, m); rc = 1;
        }
        if (cuts[i]) k = room > 0 ? room : 0;
        ids[i] = (int *)malloc((size_t)(e->nprefix + k + e->nsuffix) * sizeof(int));
        if (!ids[i]) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
        memcpy(ids[i], e->prefix, (size_t)e->nprefix * sizeof(int));
        memcpy(ids[i] + e->nprefix, body, (size_t)k * sizeof(int));
        memcpy(ids[i] + e->nprefix + k, e->suffix, (size_t)e->nsuffix * sizeof(int));
        toks[i] = e->nprefix + k + e->nsuffix;
        free(body);
    }
    for (int from = 0; from < count && !rc; ) {
        int to = from, rows = 0;
        while (to < count && (to == from || rows + toks[to] <= QE_BATCH_ROWS)) rows += toks[to++];
        int *flat = (int *)malloc((size_t)rows * sizeof(int));
        if (!flat) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
        for (int i = from, at = 0; i < to; i++) { memcpy(flat + at, ids[i], (size_t)toks[i] * sizeof(int)); at += toks[i]; }
        if (!qe_score(e, flat, toks + from, to - from, logits + from)) { qe_error(id, "the pair could not be scored"); rc = 1; }
        free(flat);
        from = to;
    }
    if (!rc) {
        printf("SCORES {\"id\":\""); qe_escape(id);
        printf("\",\"scores\":[");
        for (int i = 0; i < count; i++) printf("%s%.9g", i ? "," : "", 1.0 / (1.0 + exp(-logits[i])));
        printf("],\"logits\":[");
        for (int i = 0; i < count; i++) printf("%s%.9g", i ? "," : "", logits[i]);
        printf("],\"tokens\":[");
        for (int i = 0; i < count; i++) printf("%s%d", i ? "," : "", toks[i]);
        printf("],\"truncated\":[");
        for (int i = 0; i < count; i++) printf("%s%d", i ? "," : "", cuts[i]);
        printf("]}\n");
        fflush(stdout);
    }
    for (int i = 0; i < count; i++) free(ids[i]);
    free(ids); free(toks); free(cuts); free(logits);
    return rc;
}

/* the next line of stdin, malloc'd without its newline; NULL at end of input. A line over the limit is read to its end and
 * reported as *too_long (the text is dropped). */
static char *qe_read_line(int *too_long){
    size_t cap = 1 << 16, n = 0;
    char *buf = (char *)malloc(cap);
    int c, any = 0;
    *too_long = 0;
    if (!buf) exit(1);
    while ((c = fgetc(stdin)) != EOF) {
        any = 1;
        if (c == '\n') break;
        if (*too_long) continue;
        if (n + 2 > cap) {
            if (cap >= QE_MAX_LINE) { *too_long = 1; continue; }
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

static int qe_serve(QeEngine *e){
    printf("READY {\"model\":\""); qe_escape(e->name);
    if (e->rerank) printf("\",\"task\":\"rerank\",\"max_tokens\":%d,\"layers\":%d,\"max_documents\":%d}\n", e->max_tokens, e->m->nl, QE_MAX_INPUTS);
    else printf("\",\"dimensions\":%d,\"max_tokens\":%d,\"layers\":%d,\"min_dimensions\":1}\n", e->m->hidden, e->max_tokens, e->m->nl);
    fflush(stdout);
    for (;;) {
        int too_long;
        char *line = qe_read_line(&too_long);
        if (!line) break;
        if (too_long) { qe_error("", "the request line is too long"); free(line); continue; }
        if (!*line) { free(line); continue; }
        const char *verb = e->rerank ? "RERANK " : "EMBED ";
        const size_t vlen = strlen(verb);
        if (strncmp(line, verb, vlen)) { qe_error("", "unknown message"); free(line); continue; }
        jval *j = json_parse_checked(line + vlen);
        if (!j || j->t != J_OBJ) { qe_error("", e->rerank ? "RERANK needs a JSON object" : "EMBED needs a JSON object"); if (j) json_free(j); free(line); continue; }
        if (e->rerank) {
            jval *jid = json_get(j, "id"), *q = json_get(j, "query"), *docs = json_get(j, "documents"), *ins = json_get(j, "instruction"), *tr = json_get(j, "truncate");
            const char *id = jid && jid->t == J_STR ? jid->str : "";
            if (!q || q->t != J_STR) qe_error(id, "query must be a string");
            else if (!docs || docs->t != J_ARR || docs->len < 1 || docs->len > QE_MAX_INPUTS) qe_error(id, "documents must be an array of 1 to 2048 strings");
            else {
                int ok = 1;
                const char **texts = (const char **)calloc((size_t)docs->len, sizeof(char *));
                if (!texts) exit(1);
                for (int i = 0; i < docs->len; i++) {
                    if (docs->kids[i]->t != J_STR) { ok = 0; break; }
                    texts[i] = docs->kids[i]->str;
                }
                if (!ok) qe_error(id, "every document must be a string");
                else qe_rerank(e, id, q->str, texts, docs->len, ins && ins->t == J_STR ? ins->str : NULL, tr && tr->t == J_BOOL ? tr->boolean : 1);
                free(texts);
            }
            json_free(j);
            free(line);
            continue;
        }
        jval *jid = json_get(j, "id"), *in = json_get(j, "inputs"), *ins = json_get(j, "instruction"), *dm = json_get(j, "dimensions"), *tr = json_get(j, "truncate");
        const char *id = jid && jid->t == J_STR ? jid->str : "";
        if (!in || in->t != J_ARR || in->len < 1 || in->len > QE_MAX_INPUTS) {
            qe_error(id, "inputs must be an array of 1 to 2048 strings");
        } else {
            int ok = 1;
            const char **texts = (const char **)calloc((size_t)in->len, sizeof(char *));
            if (!texts) exit(1);
            for (int i = 0; i < in->len; i++) {
                if (in->kids[i]->t != J_STR) { ok = 0; break; }
                texts[i] = in->kids[i]->str;
            }
            if (!ok) qe_error(id, "every input must be a string");
            else {
                int dims = dm && dm->t == J_NUM ? (int)dm->num : 0;
                int truncate = tr && tr->t == J_BOOL ? tr->boolean : 1;
                qe_run(e, id, texts, in->len, ins && ins->t == J_STR ? ins->str : NULL, dims, truncate);
            }
            free(texts);
        }
        json_free(j);
        free(line);
    }
    return 0;
}

static void usage(void){
    fprintf(stderr,
        "usage: qwen3embed --model DIR --text TEXT [--instruction TASK] [--dimensions N] [--threads N]\n"
        "       qwen3embed --model DIR --query QUERY --text DOCUMENT [--instruction TASK]     (a Qwen3-Reranker checkpoint)\n"
        "       qwen3embed --model DIR --serve [--threads N]\n"
        "DIR is a Qwen3-Embedding or Qwen3-Reranker checkpoint (config.json, tokenizer.json, model.safetensors)\n");
}

int main(int argc, char **argv){
    const char *model = NULL, *text = NULL, *instruction = NULL, *query = NULL;
    int serve = 0, dims = 0, max_tokens = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--model") && v) { model = v; i++; }
        else if (!strcmp(a, "--serve")) serve = 1;
        else if (!strcmp(a, "--text") && v) { text = v; i++; }
        else if (!strcmp(a, "--instruction") && v) { instruction = v; i++; }
        else if (!strcmp(a, "--query") && v) { query = v; i++; }
        else if (!strcmp(a, "--dimensions") && v) { dims = atoi(v); i++; }
        else if (!strcmp(a, "--max-tokens") && v) { max_tokens = atoi(v); i++; }
        else if (!strcmp(a, "--threads") && v) {
#ifdef _OPENMP
            omp_set_num_threads(atoi(v));
#endif
            i++;
        }
        else { usage(); return 2; }
    }
    if (!model || serve == (text != NULL)) { usage(); return 2; }
    char err[512] = "";
    QeEngine *e = qe_open(model, err, sizeof err);
    if (!e) { fprintf(stderr, "qwen3embed: %s\n", err); return 2; }
    if (max_tokens >= 2 && max_tokens <= QE_DEFAULT_MAX_TOKENS) e->max_tokens = max_tokens;
    int rc;
    if (serve) rc = qe_serve(e);
    else if (e->rerank) {
        if (!query) { fprintf(stderr, "qwen3embed: a reranker needs --query as well as --text\n"); qe_close(e); return 2; }
        rc = qe_rerank(e, "1", query, &text, 1, instruction, 1);
    }
    else rc = qe_run(e, "1", &text, 1, instruction, dims, 1);
    qe_close(e);
    return rc;
}
