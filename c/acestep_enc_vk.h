/* acestep_enc_vk.h: the text encoder's and the condition encoders' layers on a Vulkan device (acestep_gen.h includes it when built with COLI_VULKAN;
 * acd_encoder, for the lyric and timbre encoders, and act_encode, for Qwen3-Embedding's text encoder, call it first).
 *
 * All three are stacks of the same plain layer: RMSNorm, grouped-query attention with a per-head RMS norm on q and k and RoPE, RMSNorm, SwiGLU, each with its
 * residual, no modulation. The text encoder's attention is causal (28 layers, width 1024, 16 query heads over 8); the lyric (8 layers) and timbre (4 layers)
 * encoders' is bidirectional, a sliding window on the layers config.layer_types says and the whole sequence on the others. On the CPU the timbre encoder alone
 * is 0.65 s of a song (750 frames of the silence latent), the text encoder 0.4 s, the lyric encoder 0.13 s; on the card a stack is one submission each.
 *
 * The ops are the DiT's (acestep_dit_vk.h): the matrices (bf16, f32 arithmetic) go up once, when the stack first runs; K and V go up with their rows repeated for
 * each group of query heads, because the attention shader (vkc_enc_attn mode 5) reads full-width K and V and takes a range of keys for each query row: [0, i] for a
 * causal stack, [i - w, i + w] for a sliding layer, [0, n - 1] for a full one. The embedding (the lyric's and timbre's input projections and the text's token table)
 * and the final norm stay on the CPU. A stack has no state between calls: a device that is missing, a matrix that will not go up or a call that fails leaves that
 * call to the CPU. COLI_VK_ENC=0 keeps the encoders on the CPU; ACESTEP_ENC_VK_FAULT=N makes the Nth call fail, once (the test of that way back). */
#ifndef ACESTEP_ENC_VK_H
#define ACESTEP_ENC_VK_H
#ifdef COLI_VULKAN

#include "acestep_dit_vk.h"

enum { AEV_Q, AEV_K, AEV_V, AEV_O, AEV_GATE, AEV_UP, AEV_DOWN, AEV_N };

typedef struct {
    const void *key;                                   /* the first layer: what a stack is known by */
    int tried, ok, nl, D, D2, H, Hk, hd, I;
    ColiVkTensor **t;                                  /* [nl][AEV_N] */
    void **own;                                        /* the repeated K and V matrices (the host copies the device's upload came from) */
    int nown;
    VkcBuf *prm;                                       /* per layer: ln1 [D], ln2 [D], q norm [hd], k norm [hd] */
    VkcBuf *xu, *x, *nrm, *tmp, *ctx, *qkv, *g, *u, *out, *csu, *cs, *rgF, *rgS;
    size_t rows;
} AevStack;

static AevStack aev_stacks[8];
static int aev_nstacks;
static unsigned long long aev_calls, aev_fault;
static int aev_fault_read;

static void aev_buf_free(VkcBuf **b){ if (*b) { vkc_free(*b); *b = NULL; } }

static void aev_release(AevStack *s){
    VkcBuf **all[] = { &s->prm, &s->xu, &s->x, &s->nrm, &s->tmp, &s->ctx, &s->qkv, &s->g, &s->u, &s->out, &s->csu, &s->cs, &s->rgF, &s->rgS };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) aev_buf_free(all[i]);
    for (int i = 0; s->t && i < s->nl * AEV_N; i++) if (s->t[i]) coli_vk_tensor_free(s->t[i]);
    for (int i = 0; i < s->nown; i++) free(s->own[i]);
    free(s->own); free(s->t);
    memset(s, 0, sizeof *s);
}

static void aev_free_hook(const void *layers){
    for (int i = 0; i < aev_nstacks; i++) {
        if (aev_stacks[i].key != layers) continue;
        aev_release(&aev_stacks[i]);
        aev_stacks[i] = aev_stacks[--aev_nstacks];
        return;
    }
}

static int aev_plain(const AcdLin *l){
    return !l->b && !l->m.ld && l->m.w && (l->m.fmt == QI_F32 || l->m.fmt == QI_BF16) && l->m.K % 2 == 0;
}

/* The matrices and norm weights go up once. 1 when the whole stack is there. */
static int aev_setup(AevStack *s, const AcdEncLayer *L, int nl, const AcdStackCfg *c){
    if (s->tried) return s->ok;
    s->tried = 1;
    const int D = c->D, H = c->H, Hk = c->Hk, hd = c->hd;
    if (hd > 128 || (hd & 1) || H % Hk || D % 2 || (H * hd) % 2) { fprintf(stderr, "[VK] acestep encoder: a size its ops do not take; the CPU runs it\n"); return 0; }
    if (c->causal) for (int l = 0; l < nl; l++) if (L[l].sliding) { fprintf(stderr, "[VK] acestep encoder: a sliding layer in a causal stack; the CPU runs it\n"); return 0; }
    s->nl = nl; s->D = D; s->D2 = H * hd; s->H = H; s->Hk = Hk; s->hd = hd; s->I = c->inter;
    s->t = (ColiVkTensor **)calloc((size_t)nl * AEV_N, sizeof(ColiVkTensor *));
    s->own = (void **)calloc((size_t)nl * 2, sizeof(void *));
    const size_t per = (size_t)2 * D + 2 * hd;
    float *arena = (float *)calloc((size_t)nl * per, sizeof(float));
    if (!s->t || !s->own || !arena) { free(arena); return 0; }
    for (int l = 0; l < nl; l++) {
        const AcdEncLayer *e = &L[l];
        const AcdLin *plain[AEV_N] = { &e->sa.q, NULL, NULL, &e->sa.o, &e->gate, &e->up, &e->down };
        for (int i = 0; i < AEV_N; i++) {
            if (i == AEV_K || i == AEV_V) {
                const AcdLin *src = i == AEV_K ? &e->sa.k : &e->sa.v;
                if (!aev_plain(src)) { fprintf(stderr, "[VK] acestep encoder: a matrix of layer %d does not go up; the CPU runs it\n", l); free(arena); return 0; }
                void *rep = adv_repeat(src, H, Hk, hd);
                if (!rep) { free(arena); return 0; }
                s->own[s->nown++] = rep;
                s->t[l * AEV_N + i] = adv_tensor(rep, src->m.fmt, src->m.K, H * hd);
            } else {
                if (!aev_plain(plain[i])) { fprintf(stderr, "[VK] acestep encoder: a matrix of layer %d does not go up; the CPU runs it\n", l); free(arena); return 0; }
                s->t[l * AEV_N + i] = adv_tensor(plain[i]->m.w, plain[i]->m.fmt, plain[i]->m.K, plain[i]->m.N);
            }
            if (!s->t[l * AEV_N + i]) { fprintf(stderr, "[VK] acestep encoder: a matrix of layer %d did not reach the device; the CPU runs it\n", l); free(arena); return 0; }
        }
        memcpy(arena + l * per, e->ln1, (size_t)D * sizeof(float));
        memcpy(arena + l * per + D, e->ln2, (size_t)D * sizeof(float));
        memcpy(arena + l * per + 2 * D, e->sa.qn, (size_t)hd * sizeof(float));
        memcpy(arena + l * per + 2 * D + hd, e->sa.kn, (size_t)hd * sizeof(float));
    }
    s->prm = vkc_buf((size_t)nl * per * sizeof(float), VKC_DEV);
    const int ok = s->prm && vkc_begin() && vkc_write(s->prm, 0, arena, (size_t)nl * per * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) return 0;
    s->ok = 1;
    fprintf(stderr, "[VK] acestep encoder: %d layers of width %d (%s) on the card\n", nl, D, c->causal ? "causal" : "bidirectional");
    return 1;
}

static int aev_reserve(AevStack *s, int T){
    const size_t D = s->D, D2 = s->D2, I = s->I, rows = (size_t)T;
    if (rows <= s->rows && s->xu) return 1;
    if (!vkc_reserve(&s->xu, rows * D * 4, VKC_UP) || !vkc_reserve(&s->x, rows * D * 4, VKC_DEV) || !vkc_reserve(&s->nrm, rows * D * 4, VKC_DEV) ||
        !vkc_reserve(&s->tmp, rows * D * 4, VKC_DEV) || !vkc_reserve(&s->ctx, rows * D2 * 4, VKC_DEV) || !vkc_reserve(&s->qkv, 3 * rows * D2 * 4, VKC_DEV) ||
        !vkc_reserve(&s->g, rows * I * 4, VKC_DEV) || !vkc_reserve(&s->u, rows * I * 4, VKC_DEV) || !vkc_reserve(&s->out, rows * D * 4, VKC_DOWN) ||
        !vkc_reserve(&s->csu, rows * s->hd * 4, VKC_UP) || !vkc_reserve(&s->cs, rows * s->hd * 4, VKC_DEV) ||
        !vkc_reserve(&s->rgF, rows * 8, VKC_UP) || !vkc_reserve(&s->rgS, rows * 8, VKC_UP))
        return 0;
    s->rows = rows;
    return 1;
}

/* One stack on the card: x [T][D] in place. 0: not taken (the CPU runs it). */
static int aev_run(const void *layers, int nl, const AcdStackCfg *c, float *x, int T){
    const char *off = getenv("COLI_VK_ENC");
    if (off && !strcmp(off, "0")) return 0;
    if (!aev_fault_read) { aev_fault_read = 1; const char *f = getenv("ACESTEP_ENC_VK_FAULT"); if (f) aev_fault = strtoull(f, NULL, 10); }
    AevStack *s = NULL;
    for (int i = 0; i < aev_nstacks; i++) if (aev_stacks[i].key == layers) s = &aev_stacks[i];
    if (!s) {
        if (aev_nstacks >= (int)(sizeof aev_stacks / sizeof aev_stacks[0]) || !acv_open()) return 0;
        s = &aev_stacks[aev_nstacks++];
        memset(s, 0, sizeof *s);
        s->key = layers;
    }
    if (!acv_dev.chain || !vkc_enc_ready() || T < 1 || T > 65535 || !aev_setup(s, (const AcdEncLayer *)layers, nl, c)) return 0;
    if (!aev_reserve(s, T)) { acv_failed(); return 0; }
    const AcdEncLayer *L = (const AcdEncLayer *)layers;
    const int D = s->D, D2 = s->D2, H = s->H, hd = s->hd, I = s->I;
    memcpy(vkc_ptr(s->xu), x, (size_t)T * D * sizeof(float));
    {                                                                                   /* RoPE in the shader's layout (cos, sin pairs, a row a position) */
        float *cs = (float *)malloc((size_t)T * hd * sizeof(float)), *sn = (float *)malloc((size_t)T * hd * sizeof(float));
        if (!cs || !sn) { free(cs); free(sn); return 0; }
        acd_rope_table(cs, sn, T, hd, c->theta);
        float *u = (float *)vkc_ptr(s->csu);
        for (int r = 0; r < T; r++)
            for (int j = 0; j < hd / 2; j++) { u[(size_t)r * hd + 2 * j] = cs[(size_t)r * hd + j]; u[(size_t)r * hd + 2 * j + 1] = sn[(size_t)r * hd + j]; }
        free(cs); free(sn);
    }
    {                                                                                   /* the key range of each row: full (or causal) and sliding */
        int32_t *F = (int32_t *)vkc_ptr(s->rgF), *S = (int32_t *)vkc_ptr(s->rgS);
        const int w = c->window;
        for (int i = 0; i < T; i++) {
            F[2 * i] = 0; F[2 * i + 1] = c->causal ? i : T - 1;
            S[2 * i] = w >= 0 && i - w > 0 ? i - w : 0;
            S[2 * i + 1] = w >= 0 && i + w < T - 1 ? i + w : T - 1;
        }
    }
    const size_t KO = (size_t)T * D2, VO = 2 * KO, per = (size_t)2 * D + 2 * hd;
    const float scale = 1.f / sqrtf((float)hd);
    int ok = vkc_begin() && vkc_copy(s->x, 0, s->xu, 0, (size_t)T * D) && vkc_copy(s->cs, 0, s->csu, 0, (size_t)T * hd);
    for (int l = 0; l < nl && ok; l++) {
        ColiVkTensor **tn = s->t + (size_t)l * AEV_N;
        const int wb = (int)(l * per);
        VkcNorm n1 = {T, D, 1, 0, D, D, 0, D, D, wb, 0, 0, c->eps, 1.f};
        VkcNorm qn = {T * H, hd, H, 0, D2, hd, 0, D2, hd, wb + 2 * D, 0, 0, c->eps, 1.f};
        VkcNorm kn = {T * H, hd, H, (int)KO, D2, hd, (int)KO, D2, hd, wb + 2 * D + hd, 0, 0, c->eps, 1.f};
        VkcRope rq = {T * H, H, 0, D2, hd, hd / 2, 0, hd}, rk = {T * H, H, (int)KO, D2, hd, hd / 2, 0, hd};
        VkcEncAttn at = {5, T, H, hd, 0, (int)KO, (int)VO, D2, 0, D2, 0, 0, 0, 0, 0, scale, 0};
        VkcNorm n2 = {T, D, 1, 0, D, D, 0, D, D, wb + D, 0, 0, c->eps, 1.f};
        VkcEw add = {VKC_EW_ADD, T * D, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1.f}, act = {VKC_EW_SWIGLU, T * I, I, 1, 0, 0, 0, 0, 0, 0, 0, 1.f};
        ok = vkc_norm(s->x, s->prm, s->nrm, &n1) &&
             vkc_matmul(tn[AEV_Q], s->nrm, 0, s->qkv, 0, T) && vkc_matmul(tn[AEV_K], s->nrm, 0, s->qkv, KO, T) && vkc_matmul(tn[AEV_V], s->nrm, 0, s->qkv, VO, T) &&
             vkc_norm(s->qkv, s->prm, s->qkv, &qn) && vkc_norm(s->qkv, s->prm, s->qkv, &kn) && vkc_rope(s->qkv, s->cs, &rq) && vkc_rope(s->qkv, s->cs, &rk) &&
             vkc_enc_attn(s->qkv, NULL, NULL, s->ctx, L[l].sliding ? s->rgS : s->rgF, NULL, NULL, &at) &&
             vkc_matmul(tn[AEV_O], s->ctx, 0, s->tmp, 0, T) && vkc_ew(s->x, s->x, s->tmp, NULL, NULL, &add) &&
             vkc_norm(s->x, s->prm, s->nrm, &n2) && vkc_matmul(tn[AEV_GATE], s->nrm, 0, s->g, 0, T) && vkc_matmul(tn[AEV_UP], s->nrm, 0, s->u, 0, T) &&
             vkc_ew(s->g, s->g, s->u, NULL, NULL, &act) && vkc_matmul(tn[AEV_DOWN], s->g, 0, s->tmp, 0, T) && vkc_ew(s->x, s->x, s->tmp, NULL, NULL, &add);
    }
    ok = ok && vkc_copy(s->out, 0, s->x, 0, (size_t)T * D) && vkc_submit(1);
    aev_calls++;
    if (ok && aev_fault && aev_calls == aev_fault) { ok = 0; aev_fault = 0; }          /* once */
    if (!ok) {
        fprintf(stderr, "[VK] acestep encoder: a call failed; the CPU runs it\n");
        acv_failed();
        return 0;
    }
    memcpy(x, vkc_ptr(s->out), (size_t)T * D * sizeof(float));
    return 1;
}

/* Called when an engine opens: from now on its encoders' layers go to the card when there is one. */
static void aev_attach(void){
    acd_stack_hook = aev_run;
    acd_stack_free_hook = aev_free_hook;
}

#endif
#endif
