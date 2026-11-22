/* acestep_lm_vk.h: the planner's decode steps on a Vulkan device (acestep_lm.h includes it when built with COLI_VULKAN).
 *
 * The planner writes a token at a time, and on the CPU each token is every weight read once: 0.13 s for a token of the 1.7B, of which the output matrix
 * (the tied embedding table, 217204 rows) is a third. On the card a step is one submission of the dense chain (norms, q/k/v, head norms, RoPE, attention
 * over a KV cache that lives on the device, o, SwiGLU, down), and the head is one more matrix product; only the hidden row and the logits come back. The
 * layers are the speech models' (qwen3tts.c does the same for its talker), so this is that code on the planner's tensors.
 *
 * The weights go up once, when the first sequence starts. Each sequence (the conditional and the unconditional of the codes phase) has a cache of its own on
 * the device and none on the host. The matrices are held as bf16 and computed in f32, so a logit differs from the CPU's in the last bits.
 * COLI_VK_LM=0 keeps the planner on the CPU. A card that is missing, a matrix that will not go up, or a step that fails leaves the CPU in charge: a run
 * that fails on the card is started again on the CPU (alm_generate_codes, alm_generate_meta), and the rest of the process stays there. */
#ifndef ACESTEP_LM_VK_H
#define ACESTEP_LM_VK_H
#ifdef COLI_VULKAN

#include "backend_vulkan.h"
#include "vk_chain.h"

typedef struct {
    int ok, nl;
    ColiVkTensor **t;                       /* [nl][7]: q k v o gate up down */
    ColiVkTensor *head;                     /* the embedding table as the output matrix */
    VkcBuf *prm;                            /* per layer: ln1 [D], ln2 [D], q norm [hd], k norm [hd]; then the final norm [D] */
    VkcBuf *cs, *xu, *x, *nrm, *q, *k, *v, *ctx, *g, *u, *tmp, *out, *lg, *lgd;
    size_t cs_cap;                          /* the positions the rope table on the device covers */
    unsigned long long steps, fault;        /* the steps made; ACESTEP_LM_VK_FAULT=N makes step N fail (the test of the way back to the CPU) */
} AlmVkW;

typedef struct { VkcBuf *kc, *vc; size_t cap; } AlmVk;      /* one sequence's keys and values, [nl][cap][Hk * hd] */

static struct { int tried, chain; } alm_vk_dev;

static ColiVkTensor *alm_vk_tensor(const AcdLin *l){
    const QiMat *W = &l->m;
    if (l->b || W->ld || !W->w || (W->fmt != QI_F32 && W->fmt != QI_BF16) || W->K % 2) return NULL;
    ColiVkTensor *t = NULL;
    return coli_vk_tensor_ensure(&t, W->w, NULL, W->fmt == QI_BF16 ? 11 : 10, W->K, W->N, 0) ? t : NULL;
}

static void alm_vk_lost(AceLm *m){
    if (vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); alm_vk_dev.chain = 0; }
    m->vk_on = 0; m->vk_fail = 1;
}

/* The device and the weights, the first time a sequence is made. 1 when the planner will run on the card. */
static int alm_vk_open(AceLm *m){
    if (m->vk_tried) return m->vk_on;
    m->vk_tried = 1;
    const char *off = getenv("COLI_VK_LM");
    if (off && !strcmp(off, "0")) return 0;
    if (!alm_vk_dev.tried) {
        alm_vk_dev.tried = 1;
        if (coli_vk_init_env("acestep")) {                    /* COLI_VULKAN unset, or no device: the CPU */
            atexit(coli_vk_shutdown);
            const char *c = getenv("COLI_VK_CHAIN");
            if (!(c && !strcmp(c, "0")) && vkc_init() && vkc_ready()) { alm_vk_dev.chain = 1; atexit(vkc_shutdown); }   /* registered after the backend's: runs before it */
        }
    }
    if (!alm_vk_dev.chain || m->hd > 256 || m->H % m->Hk) return 0;
    AlmVkW *w = (AlmVkW *)calloc(1, sizeof *w);
    if (!w) return 0;
    const int nl = m->nl, D = m->D, hd = m->hd;
    w->nl = nl;
    { const char *f = getenv("ACESTEP_LM_VK_FAULT"); if (f) w->fault = strtoull(f, NULL, 10); }
    w->t = (ColiVkTensor **)calloc((size_t)nl * 7, sizeof(ColiVkTensor *));
    const size_t per = (size_t)2 * D + 2 * hd, total = (size_t)nl * per + D;
    float *arena = (float *)calloc(total, sizeof(float));
    if (!w->t || !arena) { free(arena); free(w->t); free(w); return 0; }
    for (int l = 0; l < nl; l++) {
        const AcdEncLayer *L = &m->L[l];
        const AcdLin *mats[7] = { &L->sa.q, &L->sa.k, &L->sa.v, &L->sa.o, &L->gate, &L->up, &L->down };
        for (int i = 0; i < 7; i++) {
            w->t[l * 7 + i] = alm_vk_tensor(mats[i]);
            if (!w->t[l * 7 + i]) { fprintf(stderr, "[VK] acestep lm: a matrix of layer %d did not reach the device; the CPU runs the planner\n", l); free(arena); free(w->t); free(w); return 0; }
        }
        memcpy(arena + l * per, L->ln1, (size_t)D * sizeof(float));
        memcpy(arena + l * per + D, L->ln2, (size_t)D * sizeof(float));
        memcpy(arena + l * per + 2 * D, L->sa.qn, (size_t)hd * sizeof(float));
        memcpy(arena + l * per + 2 * D + hd, L->sa.kn, (size_t)hd * sizeof(float));
    }
    memcpy(arena + (size_t)nl * per, m->norm, (size_t)D * sizeof(float));
    ColiVkTensor *head = NULL;
    if (!coli_vk_tensor_ensure(&head, m->embed, NULL, 11, D, m->vocab, 0)) {
        fprintf(stderr, "[VK] acestep lm: the output matrix did not reach the device; the CPU runs the planner\n");
        free(arena); free(w->t); free(w); return 0;
    }
    w->head = head;
    w->prm = vkc_buf(total * sizeof(float), VKC_DEV);
    const int ok = w->prm && vkc_begin() && vkc_write(w->prm, 0, arena, total * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) { free(w->t); free(w); return 0; }
    w->ok = 1;
    m->vk = w;
    m->vk_on = 1;
    fprintf(stderr, "[VK] acestep lm: %s on the card (%d layers, %.1f GB of weights)\n", m->dir_name, nl, (double)m->wbytes / 1e9);
    return 1;
}

static void alm_vk_free(AceLm *m){
    AlmVkW *w = (AlmVkW *)m->vk;
    if (!w) return;
    VkcBuf **all[] = { &w->prm, &w->cs, &w->xu, &w->x, &w->nrm, &w->q, &w->k, &w->v, &w->ctx, &w->g, &w->u, &w->tmp, &w->out, &w->lg, &w->lgd };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) if (*all[i]) vkc_free(*all[i]);
    free(w->t); free(w);
    m->vk = NULL; m->vk_on = 0;
}

static void alm_vk_cache_free(AlmCache *c){
    AlmVk *v = (AlmVk *)c->vk;
    if (!v) return;
    if (v->kc) vkc_free(v->kc);
    if (v->vc) vkc_free(v->vc);
    free(v);
    c->vk = NULL;
}

/* The buffers for `rows` rows a step (a prompt's, or one), and the rope table out to `cap` positions (cos, sin pairs). */
static int alm_vk_reserve(AceLm *m, AlmVkW *w, size_t rows, size_t cap){
    const size_t D = m->D, rq = (size_t)m->H * m->hd, rk = (size_t)m->Hk * m->hd, hd = m->hd, I = m->inter;
    if (!vkc_reserve(&w->xu, rows * D * 4, VKC_UP) || !vkc_reserve(&w->x, rows * D * 4, VKC_DEV) || !vkc_reserve(&w->nrm, rows * D * 4, VKC_DEV) ||
        !vkc_reserve(&w->tmp, rows * D * 4, VKC_DEV) || !vkc_reserve(&w->q, rows * rq * 4, VKC_DEV) || !vkc_reserve(&w->ctx, rows * rq * 4, VKC_DEV) ||
        !vkc_reserve(&w->k, rows * rk * 4, VKC_DEV) || !vkc_reserve(&w->v, rows * rk * 4, VKC_DEV) || !vkc_reserve(&w->g, rows * I * 4, VKC_DEV) ||
        !vkc_reserve(&w->u, rows * I * 4, VKC_DEV) || !vkc_reserve(&w->out, D * 4, VKC_DOWN))
        return 0;
    if (cap > w->cs_cap) {
        alm_rope(m, (int)cap);
        float *tab = (float *)malloc(cap * hd * sizeof(float));
        if (!tab || !vkc_reserve(&w->cs, cap * hd * 4, VKC_DEV)) { free(tab); return 0; }
        for (size_t r = 0; r < cap; r++)
            for (size_t j = 0; j < hd / 2; j++) { tab[r * hd + 2 * j] = m->cs[r * hd + j]; tab[r * hd + 2 * j + 1] = m->sn[r * hd + j]; }
        const int ok = vkc_begin() && vkc_write(w->cs, 0, tab, cap * hd * sizeof(float)) && vkc_submit(1);
        free(tab);
        if (!ok) return 0;
        w->cs_cap = cap;
    }
    return 1;
}

/* n rows (host, [n][D]: embeddings) at positions pos0.. through every layer of one sequence, their keys and values appended to its cache on the device;
 * the hidden state of the last row (before the final norm) in last[D]. 1 on success. */
static int alm_vk_forward(AceLm *m, AlmCache *C, const float *xrows, int n, int pos0, float *last){
    AlmVkW *w = (AlmVkW *)m->vk;
    if (!w || !w->ok || pos0 + n > C->cap) return 0;
    const int D = m->D, H = m->H, Hk = m->Hk, hd = m->hd, I = m->inter;
    const size_t rq = (size_t)H * hd, rk = (size_t)Hk * hd, per = (size_t)2 * D + 2 * hd, cap = (size_t)C->cap;
    AlmVk *c = (AlmVk *)C->vk;
    if (!c) { c = (AlmVk *)calloc(1, sizeof *c); if (!c) return 0; C->vk = c; }
    if (!alm_vk_reserve(m, w, (size_t)n, cap)) return 0;
    if (c->cap < cap) {
        if (!vkc_reserve(&c->kc, (size_t)m->nl * cap * rk * 4, VKC_DEV) || !vkc_reserve(&c->vc, (size_t)m->nl * cap * rk * 4, VKC_DEV)) return 0;
        c->cap = cap;
    }
    const float scale = 1.f / sqrtf((float)hd);
    memcpy(vkc_ptr(w->xu), xrows, (size_t)n * D * sizeof(float));
    int ok = vkc_begin() && vkc_copy(w->x, 0, w->xu, 0, (size_t)n * D);
    for (int l = 0; l < m->nl && ok; l++) {
        ColiVkTensor **t = w->t + (size_t)l * 7;
        const int wb = (int)(l * per);
        const size_t kbase = (size_t)l * cap * rk;
        VkcNorm n1 = {n, D, 1, 0, D, D, 0, D, D, wb, 0, 0, m->eps, 1.f};
        VkcNorm qn = {n * H, hd, H, 0, H * hd, hd, 0, H * hd, hd, wb + 2 * D, 0, 0, m->eps, 1.f};
        VkcNorm kn = {n * Hk, hd, Hk, 0, Hk * hd, hd, 0, Hk * hd, hd, wb + 2 * D + hd, 0, 0, m->eps, 1.f};
        VkcRope rq_ = {n * H, H, 0, H * hd, hd, hd / 2, pos0 * hd, hd}, rk_ = {n * Hk, Hk, 0, Hk * hd, hd, hd / 2, pos0 * hd, hd};
        ok = vkc_norm(w->x, w->prm, w->nrm, &n1) &&
             vkc_matmul(t[0], w->nrm, 0, w->q, 0, n) && vkc_matmul(t[1], w->nrm, 0, w->k, 0, n) && vkc_matmul(t[2], w->nrm, 0, w->v, 0, n) &&
             vkc_norm(w->q, w->prm, w->q, &qn) && vkc_norm(w->k, w->prm, w->k, &kn) && vkc_rope(w->q, w->cs, &rq_) && vkc_rope(w->k, w->cs, &rk_) &&
             vkc_copy(c->kc, kbase + (size_t)pos0 * rk, w->k, 0, (size_t)n * rk) && vkc_copy(c->vc, kbase + (size_t)pos0 * rk, w->v, 0, (size_t)n * rk);
        if (ok) {
            VkcAttnW a;
            memset(&a, 0, sizeof a);
            a.a = (VkcAttn){.S = n, .H = H, .KVH = Hk, .hd = hd, .pos_base = pos0, .cap = pos0 + n, .q_off = 0, .q_row = (int)rq, .q_seg = hd,
                            .o_off = 0, .o_row = (int)rq, .scale = scale, .k_off = (int)kbase, .v_off = (int)kbase};
            a.vd = hd; a.kv_pm = 1;
            ok = vkc_attn_w(w->q, c->kc, c->vc, w->ctx, NULL, NULL, NULL, &a);
        }
        VkcNorm n2 = {n, D, 1, 0, D, D, 0, D, D, wb + D, 0, 0, m->eps, 1.f};
        VkcEw add = {VKC_EW_ADD, n * D, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f}, act = {VKC_EW_SWIGLU, n * I, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        ok = ok && vkc_matmul(t[3], w->ctx, 0, w->tmp, 0, n) && vkc_ew(w->x, w->x, w->tmp, NULL, NULL, &add) &&
             vkc_norm(w->x, w->prm, w->nrm, &n2) && vkc_matmul(t[4], w->nrm, 0, w->g, 0, n) && vkc_matmul(t[5], w->nrm, 0, w->u, 0, n) &&
             vkc_ew(w->g, w->g, w->u, NULL, NULL, &act) && vkc_matmul(t[6], w->g, 0, w->tmp, 0, n) && vkc_ew(w->x, w->x, w->tmp, NULL, NULL, &add);
    }
    ok = ok && vkc_copy(w->out, 0, w->x, (size_t)(n - 1) * D, (size_t)D) && vkc_submit(1);
    if (ok && w->fault && w->steps + 1 == w->fault) ok = 0;
    if (!ok) { fprintf(stderr, "[VK] acestep lm: a step failed; the CPU runs the planner from here on\n"); alm_vk_lost(m); return 0; }
    memcpy(last, vkc_ptr(w->out), (size_t)D * sizeof(float));
    w->steps++;
    return 1;
}

/* The full head on the device: full[r][vocab] = <final_norm(h[r]), embed[j]> for nr rows of hidden states (before the final norm). */
static int alm_vk_logits(AceLm *m, const float *h, int nr, float *full){
    AlmVkW *w = (AlmVkW *)m->vk;
    if (!w || !w->ok) return 0;
    const int D = m->D;
    const size_t per = (size_t)2 * D + 2 * m->hd, bytes = (size_t)nr * m->vocab * 4;
    if (!alm_vk_reserve(m, w, (size_t)nr, 1) || !vkc_reserve(&w->lg, bytes, VKC_DEV) || !vkc_reserve(&w->lgd, bytes, VKC_DOWN)) return 0;
    memcpy(vkc_ptr(w->xu), h, (size_t)nr * D * sizeof(float));
    VkcNorm fn = {nr, D, 1, 0, D, D, 0, D, D, (int)(m->nl * per), 0, 0, m->eps, 1.f};
    const int ok = vkc_begin() && vkc_copy(w->x, 0, w->xu, 0, (size_t)nr * D) && vkc_norm(w->x, w->prm, w->nrm, &fn) &&
                   vkc_matmul(w->head, w->nrm, 0, w->lg, 0, nr) && vkc_copy(w->lgd, 0, w->lg, 0, (size_t)nr * m->vocab) && vkc_submit(1);
    if (!ok) { fprintf(stderr, "[VK] acestep lm: the output matrix failed; the CPU runs the planner from here on\n"); alm_vk_lost(m); return 0; }
    memcpy(full, vkc_ptr(w->lgd), bytes);
    return 1;
}

#endif
#endif
