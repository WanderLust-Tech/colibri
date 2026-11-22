/* acestep_dit_vk.h: the DiT's denoising steps on a Vulkan device (acestep_gen.h includes it when built with COLI_VULKAN; acd_forward calls it first).
 *
 * A step is 24 layers over Tp = T/2 tokens: modulated gated self-attention (sliding window on half the layers), cross-attention to the condition, modulated
 * gated SwiGLU. On the CPU that is a second a step for a 10 s clip and 8 to 32 of them a song. On the card the layers are one submission: the matrices
 * (bf16, f32 arithmetic) go up once, a step uploads the patch embedding and the step's modulation vectors, and only the hidden rows come back; the
 * patch embedding, the output norm and the output projection are small and stay on the CPU.
 *
 *   - The adaptive norm's `norm(x) * w * (1 + scale) + shift` is vkc_norm with the weight w * (1 + scale) (made on the host each step) and a GATE_ADD
 *     of the shift onto a buffer of ones; the gates are GATE_ADD's per-column factor.
 *   - Grouped-query attention (16 query heads, 8 key/value heads) reads full-width K and V: the self-attention's K and V projections go up with their
 *     rows repeated per group, and the cross-attention's cached K and V are repeated per group when a condition goes up (keyed by AcdCond.id; the
 *     condition and the null condition of a guided DiT are two slots).
 *   - The attention is the encoder shader's (vkc_enc_attn mode 5, a range of rows a query): the sliding window is a range [i - w, i + w], the full
 *     layers and the cross-attention's [0, n - 1].
 * A device that is missing, a matrix that will not go up or a step that fails leaves that step to the CPU (a step is stateless, so there is nothing to
 * redo). COLI_VK_DIT=0 keeps the DiT on the CPU; ACESTEP_DIT_VK_FAULT=N makes the Nth device step fail (the test of that way back). */
#ifndef ACESTEP_DIT_VK_H
#define ACESTEP_DIT_VK_H
#ifdef COLI_VULKAN

#include "acestep_vk.h"

enum { ADV_Q, ADV_K, ADV_V, ADV_O, ADV_CQ, ADV_CO, ADV_GATE, ADV_UP, ADV_DOWN, ADV_N };

typedef struct {
    unsigned long long id;
    int len;
    VkcBuf **k, **v;                                   /* per layer: [len][H * hd], the cross-attention's keys (normed) and values, repeated per group */
} AdvSlot;

typedef struct {
    const AceDit *d;
    int tried, ok;
    ColiVkTensor **t;                                  /* [nl][ADV_N] */
    void **own;                                        /* the repeated K and V matrices (the host copies the device's upload came from) */
    int nown;
    VkcBuf *cst;                                       /* per layer: cross norm [D], self q norm [hd], self k norm [hd], cross q norm [hd] */
    VkcBuf *mod;                                       /* VKC_UP, per layer: [norm weight, shift, gate] twice (attention, mlp), 6 * D */
    VkcBuf *hu, *h, *nrm, *tmp, *ctx, *qkv, *g, *u, *ones, *out, *cs, *rgS, *rgF, *rgC;
    size_t rows, kvrows;                               /* the buffers' sizes in rows */
    int cs_tp, rg_tp, rgc_len;
    AdvSlot slot[2];
    int next_slot;
    unsigned long long steps, fault;
} AdvState;

static AdvState adv_states[4];
static int adv_nstates;

static AdvState *adv_find(const AceDit *d){
    for (int i = 0; i < adv_nstates; i++) if (adv_states[i].d == d) return &adv_states[i];
    return NULL;
}

static void adv_buf_free(VkcBuf **b){ if (*b) { vkc_free(*b); *b = NULL; } }

static void adv_release(AdvState *s){
    VkcBuf **all[] = { &s->cst, &s->mod, &s->hu, &s->h, &s->nrm, &s->tmp, &s->ctx, &s->qkv, &s->g, &s->u, &s->ones, &s->out, &s->cs, &s->rgS, &s->rgF, &s->rgC };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) adv_buf_free(all[i]);
    for (int j = 0; j < 2; j++) {
        for (int l = 0; s->slot[j].k && l < s->d->nl; l++) { adv_buf_free(&s->slot[j].k[l]); adv_buf_free(&s->slot[j].v[l]); }
        free(s->slot[j].k); free(s->slot[j].v);
    }
    for (int i = 0; i < s->nown; i++) free(s->own[i]);
    free(s->own); free(s->t);
    memset(s, 0, sizeof *s);
}

static void adv_free_hook(const AceDit *d){
    for (int i = 0; i < adv_nstates; i++) {
        if (adv_states[i].d != d) continue;
        adv_release(&adv_states[i]);
        adv_states[i] = adv_states[--adv_nstates];
        return;
    }
}

/* A projection's matrix with each key/value head's rows repeated for its group of query heads: [H * hd][K] from [Hk * hd][K]. */
static void *adv_repeat(const AcdLin *l, int H, int Hk, int hd){
    const QiMat *W = &l->m;
    const size_t es = W->fmt == QI_BF16 ? 2 : 4, row = (size_t)W->K * es;
    char *out = (char *)malloc((size_t)H * hd * row);
    if (!out) return NULL;
    const int group = H / Hk;
    for (int h = 0; h < H; h++)
        for (int j = 0; j < hd; j++)
            memcpy(out + ((size_t)h * hd + j) * row, (const char *)W->w + ((size_t)(h / group) * hd + j) * row, row);
    return out;
}

static ColiVkTensor *adv_tensor(const void *w, int fmt, int K, int N){
    ColiVkTensor *t = NULL;
    return coli_vk_tensor_ensure(&t, w, NULL, fmt == QI_BF16 ? 11 : 10, K, N, 0) ? t : NULL;
}

static int adv_plain(const AcdLin *l){
    return !l->b && !l->m.ld && l->m.w && (l->m.fmt == QI_F32 || l->m.fmt == QI_BF16) && l->m.K % 2 == 0;
}

/* The matrices and constants go up once (the first step). 1 when the whole DiT is there. */
static int adv_setup(AdvState *s){
    if (s->tried) return s->ok;
    s->tried = 1;
    const AceDit *d = s->d;
    const int nl = d->nl, D = d->D, H = d->heads, Hk = d->kv, hd = d->hd;
    if (hd > 128 || (hd & 1) || H % Hk || D % 2 || H * hd != D) { fprintf(stderr, "[VK] acestep dit: a size its ops do not take; the CPU runs the DiT\n"); return 0; }
    { const char *f = getenv("ACESTEP_DIT_VK_FAULT"); if (f) s->fault = strtoull(f, NULL, 10); }
    s->t = (ColiVkTensor **)calloc((size_t)nl * ADV_N, sizeof(ColiVkTensor *));
    s->own = (void **)calloc((size_t)nl * 2, sizeof(void *));
    const size_t per = (size_t)D + 3 * hd;
    float *cst = (float *)calloc((size_t)nl * per, sizeof(float));
    if (!s->t || !s->own || !cst) { free(cst); return 0; }
    for (int l = 0; l < nl; l++) {
        const AcdLayer *L = &d->L[l];
        const AcdLin *plain[ADV_N] = { &L->sa.q, NULL, NULL, &L->sa.o, &L->ca.q, &L->ca.o, &L->gate, &L->up, &L->down };
        for (int i = 0; i < ADV_N; i++) {
            if (i == ADV_K || i == ADV_V) {
                const AcdLin *src = i == ADV_K ? &L->sa.k : &L->sa.v;
                if (!adv_plain(src)) { fprintf(stderr, "[VK] acestep dit: a matrix of layer %d does not go up; the CPU runs the DiT\n", l); free(cst); return 0; }
                void *rep = adv_repeat(src, H, Hk, hd);
                if (!rep) { free(cst); return 0; }
                s->own[s->nown++] = rep;
                s->t[l * ADV_N + i] = adv_tensor(rep, src->m.fmt, src->m.K, H * hd);
            } else {
                if (!adv_plain(plain[i])) { fprintf(stderr, "[VK] acestep dit: a matrix of layer %d does not go up; the CPU runs the DiT\n", l); free(cst); return 0; }
                s->t[l * ADV_N + i] = adv_tensor(plain[i]->m.w, plain[i]->m.fmt, plain[i]->m.K, plain[i]->m.N);
            }
            if (!s->t[l * ADV_N + i]) { fprintf(stderr, "[VK] acestep dit: a matrix of layer %d did not reach the device; the CPU runs the DiT\n", l); free(cst); return 0; }
        }
        memcpy(cst + l * per, L->n_ca, (size_t)D * sizeof(float));
        memcpy(cst + l * per + D, L->sa.qn, (size_t)hd * sizeof(float));
        memcpy(cst + l * per + D + hd, L->sa.kn, (size_t)hd * sizeof(float));
        memcpy(cst + l * per + D + 2 * hd, L->ca.qn, (size_t)hd * sizeof(float));
    }
    s->cst = vkc_buf((size_t)nl * per * sizeof(float), VKC_DEV);
    s->mod = vkc_buf((size_t)nl * 6 * D * sizeof(float), VKC_UP);
    const int ok = s->cst && s->mod && vkc_begin() && vkc_write(s->cst, 0, cst, (size_t)nl * per * sizeof(float)) && vkc_submit(1);
    free(cst);
    if (!ok) return 0;
    for (int j = 0; j < 2; j++) {
        s->slot[j].k = (VkcBuf **)calloc((size_t)nl, sizeof(VkcBuf *));
        s->slot[j].v = (VkcBuf **)calloc((size_t)nl, sizeof(VkcBuf *));
        if (!s->slot[j].k || !s->slot[j].v) return 0;
    }
    s->ok = 1;
    fprintf(stderr, "[VK] acestep dit: %s on the card (%d layers, %.1f GB of weights)\n", d->variant, nl, (double)d->wbytes / 1e9);
    return 1;
}

/* Buffers for Tp rows and a condition of up to `len` rows, the rotation table and the attention ranges for this Tp. */
static int adv_reserve(AdvState *s, int Tp, int len){
    const AceDit *d = s->d;
    const size_t D = d->D, I = d->inter, rows = (size_t)Tp, kvr = (size_t)(Tp > len ? Tp : len), hd = d->hd;
    const int grew = rows > s->rows || kvr > s->kvrows;
    if (!vkc_reserve(&s->hu, rows * D * 4, VKC_UP) || !vkc_reserve(&s->h, rows * D * 4, VKC_DEV) || !vkc_reserve(&s->nrm, rows * D * 4, VKC_DEV) ||
        !vkc_reserve(&s->tmp, rows * D * 4, VKC_DEV) || !vkc_reserve(&s->ctx, rows * D * 4, VKC_DEV) || !vkc_reserve(&s->g, rows * I * 4, VKC_DEV) ||
        !vkc_reserve(&s->u, rows * I * 4, VKC_DEV) || !vkc_reserve(&s->out, rows * D * 4, VKC_DOWN) ||
        !vkc_reserve(&s->qkv, (rows + 2 * kvr) * D * 4, VKC_DEV) || !vkc_reserve(&s->ones, rows * D * 4, VKC_UP) ||
        !vkc_reserve(&s->rgS, rows * 8, VKC_UP) || !vkc_reserve(&s->rgF, rows * 8, VKC_UP) || !vkc_reserve(&s->rgC, rows * 8, VKC_UP) ||
        !vkc_reserve(&s->cs, rows * hd * 4, VKC_DEV))
        return 0;
    if (grew) { float *o = (float *)vkc_ptr(s->ones); for (size_t i = 0; i < rows * D; i++) o[i] = 1.f; s->cs_tp = 0; s->rg_tp = 0; s->rgc_len = 0; }
    if (rows > s->rows) s->rows = rows;
    if (kvr > s->kvrows) s->kvrows = kvr;
    if (s->cs_tp != Tp) {
        float *cs = (float *)malloc(rows * hd * sizeof(float)), *sn = (float *)malloc(rows * hd * sizeof(float)), *tab = (float *)malloc(rows * hd * sizeof(float));
        if (!cs || !sn || !tab) { free(cs); free(sn); free(tab); return 0; }
        acd_rope_table(cs, sn, Tp, d->hd, d->theta);
        for (size_t r = 0; r < rows; r++)
            for (size_t j = 0; j < hd / 2; j++) { tab[r * hd + 2 * j] = cs[r * hd + j]; tab[r * hd + 2 * j + 1] = sn[r * hd + j]; }
        const int ok = vkc_begin() && vkc_write(s->cs, 0, tab, rows * hd * sizeof(float)) && vkc_submit(1);
        free(cs); free(sn); free(tab);
        if (!ok) return 0;
        s->cs_tp = Tp;
    }
    if (s->rg_tp != Tp) {
        int32_t *S = (int32_t *)vkc_ptr(s->rgS), *F = (int32_t *)vkc_ptr(s->rgF);
        const int w = d->window;
        for (int i = 0; i < Tp; i++) {
            S[2 * i] = w >= 0 && i - w > 0 ? i - w : 0;
            S[2 * i + 1] = w >= 0 && i + w < Tp - 1 ? i + w : Tp - 1;
            F[2 * i] = 0; F[2 * i + 1] = Tp - 1;
        }
        s->rg_tp = Tp;
        s->rgc_len = 0;
    }
    if (s->rgc_len != len) {
        int32_t *C = (int32_t *)vkc_ptr(s->rgC);
        for (int i = 0; i < Tp; i++) { C[2 * i] = 0; C[2 * i + 1] = len - 1; }
        s->rgc_len = len;
    }
    return 1;
}

/* The cross-attention's keys and values of a condition, repeated per group of query heads, on the device (a slot a condition: the guided DiT's pair). */
static AdvSlot *adv_cond(AdvState *s, const AcdCond *c){
    for (int j = 0; j < 2; j++) if (s->slot[j].id == c->id && s->slot[j].len == c->len) return &s->slot[j];
    AdvSlot *sl = &s->slot[s->next_slot];
    s->next_slot ^= 1;
    const AceDit *d = s->d;
    const int H = d->heads, Hk = d->kv, hd = d->hd, group = H / Hk;
    const size_t rowk = (size_t)Hk * hd, rowh = (size_t)H * hd, bytes = (size_t)c->len * rowh * 4;
    float *ex = (float *)malloc(bytes);
    if (!ex) return NULL;
    sl->id = 0;
    for (int l = 0; l < d->nl; l++) {
        for (int pass = 0; pass < 2; pass++) {
            const float *src = pass ? c->cv[l] : c->ck[l];
            VkcBuf **dst = pass ? &sl->v[l] : &sl->k[l];
            for (int r = 0; r < c->len; r++)
                for (int h = 0; h < H; h++) memcpy(ex + (size_t)r * rowh + (size_t)h * hd, src + (size_t)r * rowk + (size_t)(h / group) * hd, (size_t)hd * sizeof(float));
            if (!vkc_reserve(dst, bytes, VKC_DEV) || !vkc_begin() || !vkc_write(*dst, 0, ex, bytes) || !vkc_submit(1)) { free(ex); acv_failed(); return NULL; }
        }
    }
    free(ex);
    sl->len = c->len; sl->id = c->id;
    return sl;
}

/* One DiT evaluation on the card: acd_forward's, with the layers on the device. 0: not taken (the CPU runs the step). */
static int adv_forward(const AceDit *d, const AcdCond *c, const float *xt, const float *ctx, int T, float t, float r, float *v){
    const char *off = getenv("COLI_VK_DIT");
    if (off && !strcmp(off, "0")) return 0;
    AdvState *s = adv_find(d);
    if (!s) {
        if (adv_nstates >= 4 || !acv_open()) return 0;
        s = &adv_states[adv_nstates++];
        memset(s, 0, sizeof *s);
        s->d = d;
    }
    if (!acv_dev.chain || !vkc_enc_ready() || !adv_setup(s)) return 0;
    const int D = d->D, P = d->patch, A = d->acdim, Tp = (T + P - 1) / P, hd = d->hd, H = d->heads, I = d->inter, nl = d->nl;
    const size_t D2 = (size_t)H * hd;
    if (!adv_reserve(s, Tp, c->len)) { acv_failed(); return 0; }
    AdvSlot *sl = adv_cond(s, c);
    if (!sl) return 0;

    /* the host's part of the step, as acd_forward does it: the timestep embeddings and the patch embedding */
    float *temb = (float *)malloc((size_t)D * sizeof(float)), *tr = (float *)malloc((size_t)D * sizeof(float));
    float *proj = (float *)malloc((size_t)6 * D * sizeof(float)), *pr = (float *)malloc((size_t)6 * D * sizeof(float));
    const int K = P * d->inch;
    float *in = (float *)calloc((size_t)Tp * K, sizeof(float));
    float *h0 = (float *)vkc_ptr(s->hu);
    if (!temb || !tr || !proj || !pr || !in) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    acd_time_embed(d, &d->te, t, temb, proj);
    acd_time_embed(d, &d->ter, t - r, tr, pr);
    for (int i = 0; i < D; i++) temb[i] += tr[i];
    for (int i = 0; i < 6 * D; i++) proj[i] += pr[i];
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < Tp; p++)
        for (int j = 0; j < P; j++) {
            int f = P * p + j;
            if (f >= T) continue;
            float *dst = in + (size_t)p * K + (size_t)j * d->inch;
            memcpy(dst, ctx + (size_t)f * 2 * A, (size_t)2 * A * sizeof(float));
            memcpy(dst + 2 * A, xt + (size_t)f * A, (size_t)A * sizeof(float));
        }
    acd_lin(h0, in, Tp, &d->proj_in);
    free(in);
    /* the step's modulation: per layer the attention norm's weight times (1 + scale), its shift and gate, then the same for the mlp */
    float *mod = (float *)vkc_ptr(s->mod);
    for (int l = 0; l < nl; l++) {
        const AcdLayer *L = &d->L[l];
        const float *sst = L->sst;
        float *m = mod + (size_t)l * 6 * D;
        for (int i = 0; i < D; i++) {
            float sh_a = sst[i] + proj[i], sc_a = sst[D + i] + proj[D + i], g_a = sst[2 * D + i] + proj[2 * D + i];
            float sh_m = sst[3 * D + i] + proj[3 * D + i], sc_m = sst[4 * D + i] + proj[4 * D + i], g_m = sst[5 * D + i] + proj[5 * D + i];
            m[i] = L->n_sa[i] * (1.f + sc_a); m[D + i] = sh_a; m[2 * D + i] = g_a;
            m[3 * D + i] = L->n_mlp[i] * (1.f + sc_m); m[4 * D + i] = sh_m; m[5 * D + i] = g_m;
        }
    }
    free(tr); free(proj); free(pr);

    const size_t KO = (size_t)Tp * D2;                                  /* q at 0, k at KO, v after the larger of Tp and the condition's rows */
    const size_t VO = KO + (size_t)(Tp > c->len ? Tp : c->len) * D2;
    const size_t per = (size_t)D + 3 * hd;
    const float scale = 1.f / sqrtf((float)hd);
    const size_t nT = (size_t)Tp * D;
    int ok = vkc_begin() && vkc_copy(s->h, 0, s->hu, 0, nT);
    for (int l = 0; l < nl && ok; l++) {
        ColiVkTensor **tn = s->t + (size_t)l * ADV_N;
        const AcdLayer *L = &d->L[l];
        const int mb = l * 6 * D, cb = (int)(l * per);
        /* 1. self-attention */
        VkcNorm n1 = {Tp, D, 1, 0, D, D, 0, D, D, mb, 0, 0, d->eps, 1.f};
        VkcEw shift1 = {VKC_EW_GATE_ADD, Tp * D, D, 1, 0, 0, 0, 0, 0, 0, mb + D, 1.f};
        VkcNorm qn = {Tp * H, hd, H, 0, (int)D2, hd, 0, (int)D2, hd, cb + D, 0, 0, d->eps, 1.f};
        VkcNorm kn = {Tp * H, hd, H, (int)KO, (int)D2, hd, (int)KO, (int)D2, hd, cb + D + hd, 0, 0, d->eps, 1.f};
        VkcRope rq = {Tp * H, H, 0, (int)D2, hd, hd / 2, 0, hd}, rk = {Tp * H, H, (int)KO, (int)D2, hd, hd / 2, 0, hd};
        VkcEncAttn at = {5, Tp, H, hd, 0, (int)KO, (int)VO, (int)D2, 0, (int)D2, 0, 0, 0, 0, 0, scale, 0};
        VkcEw ga = {VKC_EW_GATE_ADD, Tp * D, D, 1, 0, 0, 0, 0, 0, 0, mb + 2 * D, 1.f};
        ok = vkc_norm(s->h, s->mod, s->nrm, &n1) && vkc_ew(s->nrm, s->nrm, s->ones, NULL, s->mod, &shift1) &&
             vkc_matmul(tn[ADV_Q], s->nrm, 0, s->qkv, 0, Tp) && vkc_matmul(tn[ADV_K], s->nrm, 0, s->qkv, KO, Tp) && vkc_matmul(tn[ADV_V], s->nrm, 0, s->qkv, VO, Tp) &&
             vkc_norm(s->qkv, s->cst, s->qkv, &qn) && vkc_norm(s->qkv, s->cst, s->qkv, &kn) && vkc_rope(s->qkv, s->cs, &rq) && vkc_rope(s->qkv, s->cs, &rk) &&
             vkc_enc_attn(s->qkv, NULL, NULL, s->ctx, L->sliding ? s->rgS : s->rgF, NULL, NULL, &at) &&
             vkc_matmul(tn[ADV_O], s->ctx, 0, s->tmp, 0, Tp) && vkc_ew(s->h, s->h, s->tmp, NULL, s->mod, &ga);
        /* 2. cross-attention: queries from the hidden rows, keys and values of the condition (copied into the attention's buffer) */
        VkcNorm n2 = {Tp, D, 1, 0, D, D, 0, D, D, cb, 0, 0, d->eps, 1.f};
        VkcNorm cq = {Tp * H, hd, H, 0, (int)D2, hd, 0, (int)D2, hd, cb + D + 2 * hd, 0, 0, d->eps, 1.f};
        VkcEw add = {VKC_EW_ADD, Tp * D, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1.f};
        ok = ok && vkc_norm(s->h, s->cst, s->nrm, &n2) && vkc_matmul(tn[ADV_CQ], s->nrm, 0, s->qkv, 0, Tp) && vkc_norm(s->qkv, s->cst, s->qkv, &cq) &&
             vkc_copy(s->qkv, KO, sl->k[l], 0, (size_t)c->len * D2) && vkc_copy(s->qkv, VO, sl->v[l], 0, (size_t)c->len * D2) &&
             vkc_enc_attn(s->qkv, NULL, NULL, s->ctx, s->rgC, NULL, NULL, &at) &&
             vkc_matmul(tn[ADV_CO], s->ctx, 0, s->tmp, 0, Tp) && vkc_ew(s->h, s->h, s->tmp, NULL, NULL, &add);
        /* 3. SwiGLU, modulated, gated */
        VkcNorm n3 = {Tp, D, 1, 0, D, D, 0, D, D, mb + 3 * D, 0, 0, d->eps, 1.f};
        VkcEw shift2 = {VKC_EW_GATE_ADD, Tp * D, D, 1, 0, 0, 0, 0, 0, 0, mb + 4 * D, 1.f};
        VkcEw act = {VKC_EW_SWIGLU, Tp * I, I, 1, 0, 0, 0, 0, 0, 0, 0, 1.f};
        VkcEw gm = {VKC_EW_GATE_ADD, Tp * D, D, 1, 0, 0, 0, 0, 0, 0, mb + 5 * D, 1.f};
        ok = ok && vkc_norm(s->h, s->mod, s->nrm, &n3) && vkc_ew(s->nrm, s->nrm, s->ones, NULL, s->mod, &shift2) &&
             vkc_matmul(tn[ADV_GATE], s->nrm, 0, s->g, 0, Tp) && vkc_matmul(tn[ADV_UP], s->nrm, 0, s->u, 0, Tp) &&
             vkc_ew(s->g, s->g, s->u, NULL, NULL, &act) && vkc_matmul(tn[ADV_DOWN], s->g, 0, s->tmp, 0, Tp) && vkc_ew(s->h, s->h, s->tmp, NULL, s->mod, &gm);
    }
    ok = ok && vkc_copy(s->out, 0, s->h, 0, nT) && vkc_submit(1);
    if (ok && s->fault && s->steps + 1 == s->fault) { ok = 0; s->fault = 0; }        /* once */
    if (!ok) {
        fprintf(stderr, "[VK] acestep dit: a step failed; the CPU runs it\n");
        acv_failed();
        free(temb);
        return 0;
    }
    s->steps++;

    /* the host's tail: adaptive output norm and the output projection */
    const float *h = (const float *)vkc_ptr(s->out);
    float *shift = (float *)malloc((size_t)D * sizeof(float)), *scl = (float *)malloc((size_t)D * sizeof(float));
    float *n = (float *)malloc(nT * sizeof(float)), *outp = (float *)malloc((size_t)Tp * P * A * sizeof(float));
    if (!shift || !scl || !n || !outp) { fprintf(stderr, "acestep dit: out of memory\n"); exit(1); }
    for (int i = 0; i < D; i++) { shift[i] = d->sst_out[i] + temb[i]; scl[i] = d->sst_out[D + i] + temb[i]; }
    acd_rmsnorm(n, h, Tp, D, d->norm_out, d->eps);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < Tp; p++) for (int i = 0; i < D; i++) n[(size_t)p * D + i] = n[(size_t)p * D + i] * (1.f + scl[i]) + shift[i];
    acd_lin(outp, n, Tp, &d->proj_out);
    memcpy(v, outp, (size_t)T * A * sizeof(float));
    free(shift); free(scl); free(n); free(outp); free(temb);
    return 1;
}

/* Called when a DiT is loaded: from now on its steps go to the card when there is one. */
static void adv_attach(void){
    acd_forward_hook = adv_forward;
    acd_free_hook = adv_free_hook;
}

#endif
#endif
