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
 *              SCORES {id, scores [0..1 per document], logits [yes minus no], tokens, truncated}   or ERROR
 *   The pairs of a request begin with the same system prompt, instruction and query: that shared start goes through the layers
 *   once and every pair attends to its keys and values (QePast; COLI_RERANK_SHARE=0 turns it off).
 *
 * On a Vulkan device (make qwen3embed VK=1, COLI_VULKAN=1): the whole stack of layers runs as one frame of the dense chain
 * (qe_chain_run), the activations never leaving the device; without a device, or when a frame fails, the CPU runs the same
 * forward. READY says which: "device":"vulkan" or "cpu". docs/vulkan.md, "Embeddings and reranking on the chain". */
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

#define QE_SHARE_MIN 8              /* a shared start shorter than this is not worth keeping */

#define QE_DEFAULT_TASK "Given a web search query, retrieve relevant passages that answer the query"
#define QE_RERANK_PREFIX "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
#define QE_RERANK_SUFFIX "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

static int qe_share = 1;            /* COLI_RERANK_SHARE=0 turns the shared start off (to compare the two paths) */

typedef struct {
    AceText *m;
    AceTok tok;
    int max_tokens;
    char name[160];
    int rerank;                     /* a Qwen3-Reranker checkpoint */
    float *head;                    /* rerank: the output rows of "no" [0] and "yes" [1], [2][hidden] */
    int *prefix, nprefix, *suffix, nsuffix;     /* rerank: the fixed ends of every pair, as token ids */
} QeEngine;

/* The dense matrices on a Vulkan device (COLI_VULKAN=1 in a VK=1 build): every matrix product of the layers (q, k, v, o, gate, up,
 * down) goes through coli_vk_matmul, which uploads a matrix on its first use and keeps it. The norms, RoPE, the attention and the
 * SiLU stay on the CPU, so the activations cross to the device and back once for each product; at a few thousand rows that is
 * a small part of the time the CPU spent on the product. A matrix the device cannot take (an odd width, no memory left) and a
 * device that fails run on the CPU, which gives the same numbers to rounding. docs/vulkan.md. */
#ifdef COLI_VULKAN
#include "backend_vulkan.h"
#include "vk_chain.h"
#define QE_VK_SLOTS 4096
static struct {
    int ready;                                  /* the device is open */
    int chain;                                  /* the layers run as one frame on the device (COLI_VK_CHAIN, on by default) */
    struct { const void *w; ColiVkTensor *t; int off; } slot[QE_VK_SLOTS];
    unsigned long long dev, cpu;                /* products run on the device / on the CPU */
    double dev_s, att_s, hid_s;                 /* seconds: in device products, in the attention, in all the layers */
} qe_vk;

static void qe_vk_open(void){
    memset(&qe_vk, 0, sizeof qe_vk);
    if (!coli_vk_init_env("qwen3embed")) return;      /* COLI_VULKAN unset, or no device: the CPU */
    qe_vk.ready = 1;
    atexit(coli_vk_shutdown);
    const char *c = getenv("COLI_VK_CHAIN");
    if (!(c && !strcmp(c, "0")) && vkc_init() && vkc_ready()) { qe_vk.chain = 1; atexit(vkc_shutdown); }   /* registered after the backend's: runs before it */
}

/* The device copy of a matrix (uploaded on first use, shared with qe_vk_gemm), or NULL. */
static ColiVkTensor *qe_vk_tensor(const QiMat *W){
    if (!qe_vk.ready || W->ld || !W->w || (W->fmt != QI_F32 && W->fmt != QI_BF16) || W->K % 2) return NULL;
    uintptr_t key = (uintptr_t)W->w;
    unsigned h = (unsigned)((key >> 4) * 2654435761u) % QE_VK_SLOTS;
    for (int n = 0; n < QE_VK_SLOTS; n++, h = (h + 1) % QE_VK_SLOTS) {
        if (qe_vk.slot[h].w == W->w) {
            if (qe_vk.slot[h].off) return NULL;
            if (!qe_vk.slot[h].t && !coli_vk_tensor_ensure(&qe_vk.slot[h].t, W->w, NULL, W->fmt == QI_BF16 ? 11 : 10, W->K, W->N, 0)) { qe_vk.slot[h].off = 1; return NULL; }
            return qe_vk.slot[h].t;
        }
        if (!qe_vk.slot[h].w) {
            qe_vk.slot[h].w = W->w;
            if (!coli_vk_tensor_ensure(&qe_vk.slot[h].t, W->w, NULL, W->fmt == QI_BF16 ? 11 : 10, W->K, W->N, 0)) { qe_vk.slot[h].off = 1; return NULL; }
            return qe_vk.slot[h].t;
        }
    }
    return NULL;
}

/* y = x W^T (+ bias) on the device; 0 = the caller runs it */
static int qe_vk_gemm(float *Y, const float *X, int M, const QiMat *W, const float *bias){
    if (!qe_vk.ready || W->ld || !W->w || M < 1 || M > 65535 || (W->fmt != QI_F32 && W->fmt != QI_BF16) || W->K % 2) return 0;
    uintptr_t key = (uintptr_t)W->w;
    unsigned h = (unsigned)((key >> 4) * 2654435761u) % QE_VK_SLOTS;
    int at = -1;
    for (int n = 0; n < QE_VK_SLOTS; n++, h = (h + 1) % QE_VK_SLOTS) {
        if (qe_vk.slot[h].w == W->w) { at = (int)h; break; }
        if (!qe_vk.slot[h].w) { qe_vk.slot[h].w = W->w; at = (int)h; break; }
    }
    if (at < 0 || qe_vk.slot[at].off) return 0;
    const int fmt = W->fmt == QI_BF16 ? 11 : 10;
    if (!coli_vk_matmul(&qe_vk.slot[at].t, Y, X, W->w, NULL, fmt, M, W->K, W->N, 0)) {
        if (!qe_vk.slot[at].t) qe_vk.slot[at].off = 1;       /* it could not be uploaded: this matrix stays on the CPU */
        return 0;
    }
    if (bias) {
        const int N = W->N;
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < M; r++) for (int j = 0; j < N; j++) Y[(int64_t)r * N + j] += bias[j];
    }
    return 1;
}
#endif

/* One product Y[M][N] = X[M][K] W^T (+ bias): the device when there is one, else the CPU. */
static void qe_lin(float *Y, const float *X, int M, const AcdLin *l){
#ifdef COLI_VULKAN
    const double t0 = omp_get_wtime();
    if (qe_vk_gemm(Y, X, M, &l->m, l->b)) { qe_vk.dev++; qe_vk.dev_s += omp_get_wtime() - t0; return; }
    qe_vk.cpu++;
#endif
    acd_lin(Y, X, M, l);
}

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
                         const int *which, int nwhich, int H, int Hk, int hd, const float *kp, const float *vp, int P){
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
                for (int j = 0; j < P + i + 1; j++) {                /* the shared past first, then this text's own rows */
                    const float *kj = j < P ? kp + (size_t)j * Hk * hd + (size_t)kh * hd : k + (size_t)(off + j - P) * Hk * hd + (size_t)kh * hd;
                    float d = 0.f;
                    #pragma omp simd reduction(+:d)
                    for (int t = 0; t < hd; t++) d += qi[t] * kj[t];
                    d *= scale; sc[j] = d;
                    if (d > mx) mx = d;
                }
                double sum = 0;
                for (int j = 0; j < P + i + 1; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                const float inv = (float)(1.0 / sum);
                float *o = cx + (size_t)(off + i) * H * hd + (size_t)h * hd;
                for (int t = 0; t < hd; t++) o[t] = 0.f;
                for (int j = 0; j < P + i + 1; j++) {
                    const float pj = sc[j] * inv;
                    const float *vj = j < P ? vp + (size_t)j * Hk * hd + (size_t)kh * hd : v + (size_t)(off + j - P) * Hk * hd + (size_t)kh * hd;
                    #pragma omp simd
                    for (int t = 0; t < hd; t++) o[t] += pj * vj[t];
                }
            }
        }
        free(sc);
    }
}

/* A shared start: the keys and values (after the norm and RoPE) of the first P tokens of every text of a request, one pair of
 * [P][Hk*hd] arrays for each layer. The rerank pairs of one request begin with the same instruction and query, so those
 * tokens go through the layers once (qe_past_make) and every pair attends to them (the texts then hold only what differs). */
typedef struct { int P; float **k, **v;
#ifdef COLI_VULKAN
    void **dk, **dv;                     /* the same keys and values on the device (VkcBuf), made when a device forward needs them */
#endif
} QePast;

static void qe_past_free(QePast *p, int layers){
    if (!p) return;
#ifdef COLI_VULKAN
    for (int i = 0; p->dk && i < layers; i++) vkc_free((VkcBuf *)p->dk[i]);
    for (int i = 0; p->dv && i < layers; i++) vkc_free((VkcBuf *)p->dv[i]);
    free(p->dk); free(p->dv);
#endif
    for (int i = 0; p->k && i < layers; i++) free(p->k[i]);
    for (int i = 0; p->v && i < layers; i++) free(p->v[i]);
    free(p->k); free(p->v);
    memset(p, 0, sizeof *p);
}

/* One layer on x [total][hidden] where the rows are several sequences one after another (lens[nseq], starting at offs[]): the
 * matrix products run over all the rows at once; RoPE runs over all of them too, with a table whose positions start again
 * at each sequence (cs and sn have `total` rows); the causal attention runs inside each sequence. act_layer is the
 * one-sequence version of this. */
static void qe_layer(const AceText *m, const AcdEncLayer *l, float *x, int total, const int *lens, const int *offs, int nseq, const float *cs, const float *sn,
                     const float *kp, const float *vp, int P, float *kcap, float *vcap){
    const int D = m->hidden, H = m->heads, Hk = m->kv, hd = m->hd;
    float *n = (float *)malloc((size_t)total * D * sizeof(float)), *o = (float *)malloc((size_t)total * D * sizeof(float));
    float *q = (float *)malloc((size_t)total * H * hd * sizeof(float)), *k = (float *)malloc((size_t)total * Hk * hd * sizeof(float));
    float *v = (float *)malloc((size_t)total * Hk * hd * sizeof(float)), *cx = (float *)malloc((size_t)total * H * hd * sizeof(float));
    float *g = (float *)malloc((size_t)total * m->inter * sizeof(float)), *u = (float *)malloc((size_t)total * m->inter * sizeof(float));
    if (!n || !o || !q || !k || !v || !cx || !g || !u) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    acd_rmsnorm(n, x, total, D, l->ln1, m->eps);
    qe_lin(q, n, total, &l->sa.q); qe_lin(k, n, total, &l->sa.k); qe_lin(v, n, total, &l->sa.v);
    acd_head_norm(q, total, H, hd, l->sa.qn, m->eps);
    acd_head_norm(k, total, Hk, hd, l->sa.kn, m->eps);
    acd_rope(q, total, H, hd, cs, sn);
    acd_rope(k, total, Hk, hd, cs, sn);
    if (kcap) { memcpy(kcap, k, (size_t)total * Hk * hd * sizeof(float)); memcpy(vcap, v, (size_t)total * Hk * hd * sizeof(float)); }
    int *shortseq = (int *)malloc((size_t)nseq * sizeof(int)), nshort = 0;
    if (!shortseq) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    for (int s = 0; s < nseq; s++) {
        if (P > 0 || lens[s] <= QE_SMALL) { shortseq[nshort++] = s; continue; }       /* with a past every text uses qe_attention (the caller keeps P + len within QE_SMALL) */
        acd_attention_ex(cx + (size_t)offs[s] * H * hd, H * hd, q + (size_t)offs[s] * H * hd, H * hd, lens[s],
                         k + (size_t)offs[s] * Hk * hd, v + (size_t)offs[s] * Hk * hd, Hk * hd, lens[s], H, Hk, hd, -1, 1);
    }
#ifdef COLI_VULKAN
    const double ta = omp_get_wtime();
#endif
    if (nshort) qe_attention(cx, q, k, v, lens, offs, shortseq, nshort, H, Hk, hd, kp, vp, P);
#ifdef COLI_VULKAN
    qe_vk.att_s += omp_get_wtime() - ta;
#endif
    free(shortseq);
    qe_lin(o, cx, total, &l->sa.o);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)total * D; i++) x[i] += o[i];
    acd_rmsnorm(n, x, total, D, l->ln2, m->eps);
    qe_lin(g, n, total, &l->gate); qe_lin(u, n, total, &l->up);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)total * m->inter; i++) g[i] = acd_silu(g[i]) * u[i];
    qe_lin(o, g, total, &l->down);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)total * D; i++) x[i] += o[i];
    free(n); free(o); free(q); free(k); free(v); free(cx); free(g); free(u);
}

#ifdef COLI_VULKAN
/* The whole stack of layers on the device, in one frame: the activations never leave it (the per-matrix path moves every
 * product's input and output across PCIe, which cost more than the product). The embedding rows go up once; what comes back is
 * the last row of each text (the only rows the callers read) and, when the keys and values of a shared start are wanted, those.
 * Per layer: RMSNorm, q/k/v, the q and k head norms, RoPE, attention (one dispatch per text, over the shared start's keys then
 * its own, as one position-major cache), o, the residual, RMSNorm, gate and up, SwiGLU, down, the residual. */
typedef struct {
    int tried, ok, nl;
    ColiVkTensor **t;                    /* [nl][7]: q k v o gate up down */
    VkcBuf *prm;                         /* per layer: ln1 [D], ln2 [D], q norm [hd], k norm [hd] */
    VkcBuf *xu, *csu, *x, *nrm, *q, *k, *v, *ctx, *g, *u, *tmp, *cs, *kc, *vc, *xd, *kvd;
    unsigned long long forwards;
    double ms;
} QeChain;
static QeChain qe_ch;

static int qe_chain_setup(const AceText *m){
    QeChain *c = &qe_ch;
    if (c->tried) return c->ok;
    c->tried = 1;
    if (!qe_vk.chain || m->hd > 256 || m->heads % m->kv) return 0;
    const int D = m->hidden, hd = m->hd, nl = m->nl;
    c->nl = nl;
    c->t = (ColiVkTensor **)calloc((size_t)nl * 7, sizeof(ColiVkTensor *));
    const size_t per = (size_t)2 * D + 2 * hd;
    float *arena = (float *)calloc((size_t)nl * per, sizeof(float));
    if (!c->t || !arena) { free(arena); return 0; }
    for (int l = 0; l < nl; l++) {
        const AcdEncLayer *L = &m->L[l];
        const AcdLin *mats[7] = { &L->sa.q, &L->sa.k, &L->sa.v, &L->sa.o, &L->gate, &L->up, &L->down };
        for (int i = 0; i < 7; i++) {
            c->t[l * 7 + i] = mats[i]->b ? NULL : qe_vk_tensor(&mats[i]->m);          /* these models have no biases */
            if (!c->t[l * 7 + i]) { fprintf(stderr, "[VK] qwen3embed: a matrix of layer %d did not reach the device; per-matrix path\n", l); free(arena); return 0; }
        }
        memcpy(arena + l * per, L->ln1, (size_t)D * sizeof(float));
        memcpy(arena + l * per + D, L->ln2, (size_t)D * sizeof(float));
        memcpy(arena + l * per + 2 * D, L->sa.qn, (size_t)hd * sizeof(float));
        memcpy(arena + l * per + 2 * D + hd, L->sa.kn, (size_t)hd * sizeof(float));
    }
    c->prm = vkc_buf((size_t)nl * per * sizeof(float), VKC_DEV);
    int ok = c->prm && vkc_begin() && vkc_write(c->prm, 0, arena, (size_t)nl * per * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) return 0;
    c->ok = 1;
    fprintf(stderr, "[VK] qwen3embed: %d layers on the device, one frame a forward\n", nl);
    return 1;
}

static void qe_chain_off(const char *why){
    qe_ch.ok = 0;
    fprintf(stderr, "[VK] qwen3embed: the chain stopped (%s); per-matrix path from here on\n", why);
    if (vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); qe_vk.ready = 0; }
}

/* x [total][hidden]: the embedding rows (host). Returns the hidden states in a calloc'd [total][hidden] of which only the last
 * row of each text is filled, or NULL (the caller runs the CPU). past: the shared start (device copies made here when it has none);
 * capture: asks for the keys and values of this one text to be kept (host copies in capture->k/v, device copies in dk/dv). */
static float *qe_chain_run(const AceText *m, const float *xh, int total, const int *lens, const int *offs, int nseq,
                           const float *cs, const float *sn, QePast *past, QePast *capture){
    if (!qe_chain_setup(m)) return NULL;
    QeChain *c = &qe_ch;
    const int D = m->hidden, H = m->heads, Hk = m->kv, hd = m->hd, I = m->inter, nl = m->nl, P = past ? past->P : 0;
    const size_t rq = (size_t)H * hd, rk = (size_t)Hk * hd, per = (size_t)2 * D + 2 * hd, T = (size_t)total;
    const size_t cache_rows = P ? (size_t)nseq * P + T : 0;
    const double t0 = omp_get_wtime();
    if (!vkc_reserve(&c->xu, T * D * 4, VKC_UP) || !vkc_reserve(&c->csu, T * hd * 4, VKC_UP) ||
        !vkc_reserve(&c->x, T * D * 4, VKC_DEV) || !vkc_reserve(&c->nrm, T * D * 4, VKC_DEV) || !vkc_reserve(&c->tmp, T * D * 4, VKC_DEV) ||
        !vkc_reserve(&c->q, T * rq * 4, VKC_DEV) || !vkc_reserve(&c->ctx, T * rq * 4, VKC_DEV) ||
        !vkc_reserve(&c->k, T * rk * 4, VKC_DEV) || !vkc_reserve(&c->v, T * rk * 4, VKC_DEV) ||
        !vkc_reserve(&c->g, T * I * 4, VKC_DEV) || !vkc_reserve(&c->u, T * I * 4, VKC_DEV) || !vkc_reserve(&c->cs, T * hd * 4, VKC_DEV) ||
        !vkc_reserve(&c->xd, (size_t)nseq * D * 4, VKC_DOWN) ||
        (P && (!vkc_reserve(&c->kc, cache_rows * rk * 4, VKC_DEV) || !vkc_reserve(&c->vc, cache_rows * rk * 4, VKC_DEV))) ||
        (capture && !vkc_reserve(&c->kvd, (size_t)nl * 2 * T * rk * 4, VKC_DOWN))) {
        fprintf(stderr, "[VK] qwen3embed: device memory for %d rows refused; this forward runs per matrix\n", total);
        return NULL;
    }
    if (P && !past->dk) {                                       /* a shared start made on the host: its device copy */
        past->dk = (void **)calloc((size_t)nl, sizeof(void *)); past->dv = (void **)calloc((size_t)nl, sizeof(void *));
        if (!past->dk || !past->dv) return NULL;
        int ok = vkc_begin();
        for (int l = 0; l < nl && ok; l++) {
            past->dk[l] = vkc_buf((size_t)P * rk * 4, VKC_DEV); past->dv[l] = vkc_buf((size_t)P * rk * 4, VKC_DEV);
            ok = past->dk[l] && past->dv[l] && vkc_write((VkcBuf *)past->dk[l], 0, past->k[l], (size_t)P * rk * 4) &&
                 vkc_write((VkcBuf *)past->dv[l], 0, past->v[l], (size_t)P * rk * 4);
        }
        ok = ok && vkc_submit(1);
        if (!ok) { qe_chain_off("the shared start would not go up"); return NULL; }
    }
    if (capture) {
        capture->dk = (void **)calloc((size_t)nl, sizeof(void *)); capture->dv = (void **)calloc((size_t)nl, sizeof(void *));
        if (!capture->dk || !capture->dv) return NULL;
        for (int l = 0; l < nl; l++) {
            capture->dk[l] = vkc_buf(T * rk * 4, VKC_DEV); capture->dv[l] = vkc_buf(T * rk * 4, VKC_DEV);
            if (!capture->dk[l] || !capture->dv[l]) { qe_past_free(capture, nl); capture->dk = capture->dv = NULL; return NULL; }
        }
    }
    memcpy(vkc_ptr(c->xu), xh, T * D * sizeof(float));
    {
        float *u = (float *)vkc_ptr(c->csu);                    /* the table the shader reads: cos, sin pairs, one row a position */
        for (size_t r = 0; r < T; r++)
            for (int j = 0; j < hd / 2; j++) { u[r * hd + 2 * j] = cs[r * hd + j]; u[r * hd + 2 * j + 1] = sn[r * hd + j]; }
    }
    VkcRegion *pre = NULL, *own = NULL;                         /* the per-text cache: the shared start, then the text's rows */
    if (P) {
        pre = (VkcRegion *)malloc((size_t)nseq * sizeof *pre); own = (VkcRegion *)malloc((size_t)nseq * sizeof *own);
        if (!pre || !own) { free(pre); free(own); return NULL; }
        for (int s = 0; s < nseq; s++) {
            const size_t at = ((size_t)s * P + offs[s]) * rk;
            pre[s] = (VkcRegion){at, 0, (size_t)P * rk};
            own[s] = (VkcRegion){at + (size_t)P * rk, (size_t)offs[s] * rk, (size_t)lens[s] * rk};
        }
    }
    VkcRegion *last = (VkcRegion *)malloc((size_t)nseq * sizeof *last);
    if (!last) { free(pre); free(own); return NULL; }
    for (int s = 0; s < nseq; s++) last[s] = (VkcRegion){(size_t)s * D, ((size_t)offs[s] + lens[s] - 1) * D, (size_t)D};
    const float scale = 1.f / sqrtf((float)hd);
    int ok = vkc_begin() && vkc_copy(c->x, 0, c->xu, 0, T * D) && vkc_copy(c->cs, 0, c->csu, 0, T * hd);
    for (int l = 0; l < nl && ok; l++) {
        ColiVkTensor **t = c->t + (size_t)l * 7;
        const int wb = (int)(l * per);
        VkcNorm n1 = {total, D, 1, 0, D, D, 0, D, D, wb, 0, 0, m->eps, 1.f};
        VkcNorm qn = {total * H, hd, H, 0, H * hd, hd, 0, H * hd, hd, wb + 2 * D, 0, 0, m->eps, 1.f};
        VkcNorm kn = {total * Hk, hd, Hk, 0, Hk * hd, hd, 0, Hk * hd, hd, wb + 2 * D + hd, 0, 0, m->eps, 1.f};
        VkcRope rq_ = {total * H, H, 0, H * hd, hd, hd / 2, 0, hd}, rk_ = {total * Hk, Hk, 0, Hk * hd, hd, hd / 2, 0, hd};
        ok = vkc_norm(c->x, c->prm, c->nrm, &n1) &&
             vkc_matmul(t[0], c->nrm, 0, c->q, 0, total) && vkc_matmul(t[1], c->nrm, 0, c->k, 0, total) && vkc_matmul(t[2], c->nrm, 0, c->v, 0, total) &&
             vkc_norm(c->q, c->prm, c->q, &qn) && vkc_norm(c->k, c->prm, c->k, &kn) && vkc_rope(c->q, c->cs, &rq_) && vkc_rope(c->k, c->cs, &rk_);
        if (ok && capture)
            ok = vkc_copy((VkcBuf *)capture->dk[l], 0, c->k, 0, T * rk) && vkc_copy((VkcBuf *)capture->dv[l], 0, c->v, 0, T * rk) &&
                 vkc_copy(c->kvd, (size_t)l * 2 * T * rk, c->k, 0, T * rk) && vkc_copy(c->kvd, (size_t)l * 2 * T * rk + T * rk, c->v, 0, T * rk);
        if (ok && P)
            ok = vkc_copy_regions(c->kc, (VkcBuf *)past->dk[l], pre, nseq) && vkc_copy_regions(c->kc, c->k, own, nseq) &&
                 vkc_copy_regions(c->vc, (VkcBuf *)past->dv[l], pre, nseq) && vkc_copy_regions(c->vc, c->v, own, nseq);
        for (int s = 0; s < nseq && ok; s++) {
            VkcAttnW a;
            memset(&a, 0, sizeof a);
            const size_t kb = P ? ((size_t)s * P + offs[s]) * rk : (size_t)offs[s] * rk;
            a.a = (VkcAttn){.S = lens[s], .H = H, .KVH = Hk, .hd = hd, .pos_base = P, .cap = P + lens[s], .q_off = (int)((size_t)offs[s] * rq),
                            .q_row = (int)rq, .q_seg = hd, .o_off = (int)((size_t)offs[s] * rq), .o_row = (int)rq, .scale = scale,
                            .k_off = (int)kb, .v_off = (int)kb};
            a.vd = hd; a.kv_pm = 1;
            ok = vkc_attn_w(c->q, P ? c->kc : c->k, P ? c->vc : c->v, c->ctx, NULL, NULL, NULL, &a);
        }
        VkcNorm n2 = {total, D, 1, 0, D, D, 0, D, D, wb + D, 0, 0, m->eps, 1.f};
        VkcEw add = {VKC_EW_ADD, total * D, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f}, act = {VKC_EW_SWIGLU, (int)(T * I), 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        ok = ok && vkc_matmul(t[3], c->ctx, 0, c->tmp, 0, total) && vkc_ew(c->x, c->x, c->tmp, NULL, NULL, &add) &&
             vkc_norm(c->x, c->prm, c->nrm, &n2) && vkc_matmul(t[4], c->nrm, 0, c->g, 0, total) && vkc_matmul(t[5], c->nrm, 0, c->u, 0, total) &&
             vkc_ew(c->g, c->g, c->u, NULL, NULL, &act) && vkc_matmul(t[6], c->g, 0, c->tmp, 0, total) && vkc_ew(c->x, c->x, c->tmp, NULL, NULL, &add);
    }
    ok = ok && vkc_copy_regions(c->xd, c->x, last, nseq) && vkc_submit(1);
    free(pre); free(own); free(last);
    if (!ok) {
        if (capture) { qe_past_free(capture, nl); capture->dk = capture->dv = NULL; }
        qe_chain_off("a frame failed");
        return NULL;
    }
    float *out = (float *)calloc(T * D, sizeof(float));
    if (!out) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    for (int s = 0; s < nseq; s++) memcpy(out + ((size_t)offs[s] + lens[s] - 1) * D, (const float *)vkc_ptr(c->xd) + (size_t)s * D, (size_t)D * sizeof(float));
    if (capture) {
        const float *kv = (const float *)vkc_ptr(c->kvd);
        for (int l = 0; l < nl; l++) {
            memcpy(capture->k[l], kv + (size_t)l * 2 * T * rk, T * rk * sizeof(float));
            memcpy(capture->v[l], kv + (size_t)l * 2 * T * rk + T * rk, T * rk * sizeof(float));
        }
    }
    c->forwards++;
    c->ms += (omp_get_wtime() - t0) * 1e3;
    return out;
}
#endif

/* nseq sequences (ids concatenated, lens each) through every layer: the hidden states [total][hidden] before the final norm
 * (malloc'd; NULL for a bad id). */
static float *qe_hidden(const QeEngine *e, const int *ids, const int *lens, int nseq, QePast *past, QePast *capture){
    const AceText *m = e->m;
    const int P = past ? past->P : 0;
    int total = 0, longest = 0;
    for (int s = 0; s < nseq; s++) { total += lens[s]; if (lens[s] > longest) longest = lens[s]; }
    float *x = act_embed(m, ids, total);
    if (!x) return NULL;
    float *bcs = (float *)malloc((size_t)(P + longest) * m->hd * sizeof(float)), *bsn = (float *)malloc((size_t)(P + longest) * m->hd * sizeof(float));
    float *cs = (float *)malloc((size_t)total * m->hd * sizeof(float)), *sn = (float *)malloc((size_t)total * m->hd * sizeof(float));
    int *offs = (int *)malloc((size_t)nseq * sizeof(int));
    if (!bcs || !bsn || !cs || !sn || !offs) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
    acd_rope_table(bcs, bsn, P + longest, m->hd, m->theta);
    for (int s = 0, at = 0; s < nseq; at += lens[s], s++) {          /* the position of a row is its place in its own text (after the past) */
        offs[s] = at;
        memcpy(cs + (size_t)at * m->hd, bcs + (size_t)P * m->hd, (size_t)lens[s] * m->hd * sizeof(float));
        memcpy(sn + (size_t)at * m->hd, bsn + (size_t)P * m->hd, (size_t)lens[s] * m->hd * sizeof(float));
    }
    free(bcs); free(bsn);
    if (capture) {                                                   /* the keys and values of this one text become a past */
        const size_t row = (size_t)m->kv * m->hd;
        capture->P = total;
        capture->k = (float **)calloc((size_t)m->nl, sizeof(float *)); capture->v = (float **)calloc((size_t)m->nl, sizeof(float *));
        if (!capture->k || !capture->v) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
        for (int i = 0; i < m->nl; i++) {
            capture->k[i] = (float *)malloc((size_t)total * row * sizeof(float)); capture->v[i] = (float *)malloc((size_t)total * row * sizeof(float));
            if (!capture->k[i] || !capture->v[i]) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
        }
    }
#ifdef COLI_VULKAN
    const double th = omp_get_wtime();
    if (qe_vk.chain) {                                               /* the whole stack on the device; NULL: the host runs it below */
        float *r = qe_chain_run(m, x, total, lens, offs, nseq, cs, sn, past, capture);
        if (r) { free(x); free(cs); free(sn); free(offs); qe_vk.hid_s += omp_get_wtime() - th; return r; }
    }
#endif
    for (int i = 0; i < m->nl; i++)
        qe_layer(m, &m->L[i], x, total, lens, offs, nseq, cs, sn, past ? past->k[i] : NULL, past ? past->v[i] : NULL, P,
                 capture ? capture->k[i] : NULL, capture ? capture->v[i] : NULL);
#ifdef COLI_VULKAN
    qe_vk.hid_s += omp_get_wtime() - th;
#endif
    free(cs); free(sn); free(offs);
    return x;
}

/* The unit vectors of nseq sequences (ids concatenated, lens each): out[nseq][hidden]; 0 on failure (a bad id, a vector that
 * is not finite). A vector is the last token's hidden state after the final norm, normalised. */
static int qe_encode(const QeEngine *e, const int *ids, const int *lens, int nseq, float *out){
    const AceText *m = e->m;
    const int D = m->hidden;
    float *x = qe_hidden(e, ids, lens, nseq, NULL, NULL);
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

/* Runs one text (the shared start of a request) through the layers to keep its keys and values as a past; 0 on failure. */
static int qe_hidden_free(const QeEngine *e, const int *ids, const int *len, QePast *past){
    float *x = qe_hidden(e, ids, len, 1, NULL, past);
    if (!x) return 0;
    free(x);
    return 1;
}

/* The logit difference (yes minus no) of nseq pairs (ids concatenated, lens each): out[nseq]; 0 on failure. The head is two
 * rows of the output matrix, so a pair costs the layers and then two dot products over the last row. */
static int qe_score(const QeEngine *e, const int *ids, const int *lens, int nseq, double *out, QePast *past){
    const AceText *m = e->m;
    const int D = m->hidden;
    float *x = qe_hidden(e, ids, lens, nseq, past, NULL);
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
    /* The longest start every pair has in common (the fixed prefix, the instruction, the query, often a first word of the
     * document): it goes through the layers once. One token at least is left to each pair, since the score is read from
     * its last row; the pairs must still fit the one-region attention with the start in front of them. */
    int P = 0;
    if (!rc && count >= 2 && qe_share) {
        int small = 1, lim = toks[0] - 1;
        for (int i = 0; i < count; i++) { if (toks[i] > QE_SMALL) small = 0; if (toks[i] - 1 < lim) lim = toks[i] - 1; }
        if (small) {
            P = lim;
            for (int i = 1; i < count && P > 0; i++) { int j = 0; while (j < P && ids[i][j] == ids[0][j]) j++; P = j; }
        }
        if (P < QE_SHARE_MIN) P = 0;
    }
    QePast past; memset(&past, 0, sizeof past);
    if (P > 0) {
        int *pi = ids[0], pl = P;
        if (!qe_hidden_free(e, pi, &pl, &past)) { qe_error(id, "the pair could not be scored"); rc = 1; }
    }
    for (int from = 0; from < count && !rc; ) {
        int to = from, rows = 0;
        while (to < count && (to == from || rows + toks[to] - P <= QE_BATCH_ROWS)) rows += toks[to++] - P;
        int *flat = (int *)malloc((size_t)rows * sizeof(int)), *lens = (int *)malloc((size_t)(to - from) * sizeof(int));
        if (!flat || !lens) { fprintf(stderr, "qwen3embed: out of memory\n"); exit(1); }
        for (int i = from, at = 0; i < to; i++) {
            lens[i - from] = toks[i] - P;
            memcpy(flat + at, ids[i] + P, (size_t)(toks[i] - P) * sizeof(int)); at += toks[i] - P;
        }
        if (!qe_score(e, flat, lens, to - from, logits + from, P > 0 ? &past : NULL)) { qe_error(id, "the pair could not be scored"); rc = 1; }
        free(flat); free(lens);
        from = to;
    }
    qe_past_free(&past, e->m->nl);
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

static const char *qe_device(void){
#ifdef COLI_VULKAN
    if (qe_vk.ready) return "vulkan";
#endif
    return "cpu";
}

static int qe_serve(QeEngine *e){
    printf("READY {\"model\":\""); qe_escape(e->name);
    if (e->rerank) printf("\",\"task\":\"rerank\",\"max_tokens\":%d,\"layers\":%d,\"max_documents\":%d,\"device\":\"%s\"}\n", e->max_tokens, e->m->nl, QE_MAX_INPUTS, qe_device());
    else printf("\",\"dimensions\":%d,\"max_tokens\":%d,\"layers\":%d,\"min_dimensions\":1,\"device\":\"%s\"}\n", e->m->hidden, e->max_tokens, e->m->nl, qe_device());
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
    { const char *sh = getenv("COLI_RERANK_SHARE"); if (sh && !strcmp(sh, "0")) qe_share = 0; }
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
#ifdef COLI_VULKAN
    qe_vk_open();                      /* once the weights are loaded: the device opens, and each matrix goes up on its first use */
#endif
    int rc;
    if (serve) rc = qe_serve(e);
    else if (e->rerank) {
        if (!query) { fprintf(stderr, "qwen3embed: a reranker needs --query as well as --text\n"); qe_close(e); return 2; }
        rc = qe_rerank(e, "1", query, &text, 1, instruction, 1);
    }
    else rc = qe_run(e, "1", &text, 1, instruction, dims, 1);
#ifdef COLI_VULKAN
    if (qe_vk.ready) fprintf(stderr, "[VK] qwen3embed: %llu forwards as one frame (%.2f s); %llu products on the device, %llu on the CPU; layers %.2f s, of it products %.2f s, attention %.2f s\n",
                             qe_ch.forwards, qe_ch.ms / 1e3, qe_vk.dev, qe_vk.cpu, qe_vk.hid_s, qe_vk.dev_s, qe_vk.att_s);
#endif
    qe_close(e);
    return rc;
}
