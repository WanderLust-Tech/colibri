/* zimage_chain.h -- Z-Image's 30 main blocks as a dense chain on the Vulkan device (vk_chain.h). Included once by
 * zimage.c in a COLI_VULKAN build, after the CPU DiT it stands in for; COLI_VK_CHAIN decides (coli_vk_chain_decide).
 *
 * The refiners (two small blocks a side) and the final layer stay on the CPU, with their matrices on the device one by one;
 * what the chain runs is the unified sequence [image tokens ; caption tokens] through the 30 main blocks, which is almost
 * all of a step. The sequence goes up once, every block runs on the device, and the image rows come back:
 *   per block: h = RMSNorm(x) * an1 * (1 + scale_msa); q, k, v of h; an RMSNorm on every head of q and k; RoPE;
 *   attention over every row; o = RMSNorm(o(attention)) * an2; x += tanh(gate_msa) * o;
 *   h = RMSNorm(x) * fn1 * (1 + scale_mlp); x += tanh(gate_mlp) * RMSNorm(w2(silu(w1(h)) * w3(h))) * fn2
 * The modulation (the timestep's embedding through each block's adaLN linear) stays on the CPU and goes up as four vectors
 * a block and step: the two pre-norms' weights already multiplied by (1 + scale), and the two tanh gates.
 *
 * The rotary pairs: the CPU rotates (2i, 2i+1) in place; the device writes each pair's result into the two halves of the
 * head, (i, i + hd/2) (chain_mla.comp's interleaved style). That permutes q's and k's dimensions alike and leaves every
 * q . k as it was; v is not touched.
 *
 * The device sums in other orders than the CPU (the norms in float trees, the GEMMs by tiles, the softmax online), so a
 * prediction agrees with the CPU's to rounding.
 *
 * Declines (the per-matrix path runs): COLI_VK_CHAIN off, a matrix the device does not take, the device's memory refusing a
 * buffer. A device lost in a step: the step runs again on the CPU, and the CPU runs from there. */
#include "vk_chain.h"
#include "vk_kvsplit.h"   /* vkc_dev_avail */

typedef struct {
    int ok, failed, planned;
    int T;                       /* rows the buffers hold */
    VkcBuf *prm;                 /* per block [nq hd | nk hd | an2 D | fn2 D]: the norms that do not change */
    VkcBuf *mod;                 /* VKC_UP, per block [an1 (1+scale_msa) D | tanh(gate_msa) D | fn1 (1+scale_mlp) D | tanh(gate_mlp) D] */
    VkcBuf *cs;                  /* VKC_UP, the rows' (cos, sin) pairs, [T][hd] */
    VkcBuf *x, *h, *g, *u;       /* [T][D] and [T][mlp] */
    VkcBuf *qkv;                 /* q, k, v: [3][T][D] */
    VkcBuf *a;                   /* attention output, [T][D] */
    VkcBuf *xd;                  /* VKC_DOWN, the image rows after the last block, [Nx][D] */
    unsigned long long steps;
} ZiChain;

static ZiChain g_zic;
static int g_zic_on = -1;        /* -1: not decided yet */

static ColiVkTensor *zic_tensor(Lin *l) {
    if (l->vk) return (ColiVkTensor *)l->vk;
    if (l->m.ld) return NULL;
    int fmt = l->m.fmt == QI_I8 ? 1 : l->m.fmt == QI_BF16 ? 11 : 10;
    return coli_vk_tensor_ensure((ColiVkTensor **)&l->vk, l->m.w, l->m.fmt == QI_I8 ? l->m.sc : NULL, fmt,
                                 l->m.K, l->m.N, 0) ? (ColiVkTensor *)l->vk : NULL;
}

static void zic_off(const char *why) {
    if (!g_zic.failed) fprintf(stderr, "[VK] zimage chain: %s; the matrices run on the device one by one\n", why);
    g_zic.failed = 1;
}

static void zic_block_mats(ZBlock *B, Lin **m) {
    m[0] = &B->q; m[1] = &B->k; m[2] = &B->v; m[3] = &B->o; m[4] = &B->w1; m[5] = &B->w3; m[6] = &B->w2;
}

/* When the weights load: whether the chain runs, and whether the device's memory holds every block (the int8 DiT is 6 GB,
 * and a step's buffers come on top). A device that does not hold them leaves the per-matrix path alone. */
static void zic_plan(Dit *d) {
    g_zic_on = vkc_init() && coli_vk_chain_decide("zimage", -1, COLI_VK_CHAIN_ON) == COLI_VK_CHAIN_ON;
    g_zic.planned = 1;
    if (g_zic_on <= 0 || !coli_vk_dense()) { g_zic_on = 0; return; }
    size_t blocks = 0, D = d->dim, M = d->mlp, T = 4096 + 256;
    for (int l = 0; l < d->layers; l++) {
        Lin *m[7]; zic_block_mats(&d->B[l], m);
        for (int k = 0; k < 7; k++) {
            int fmt = m[k]->m.fmt == QI_I8 ? 1 : m[k]->m.fmt == QI_BF16 ? 11 : 10;
            blocks += coli_vk_tensor_payload(fmt, m[k]->m.K, m[k]->m.N, 0) + coli_vk_buffer_alignment();
        }
    }
    size_t work = 4 * (7 * T * D + 2 * T * M) + (size_t)d->layers * (2 * d->hd + 2 * D) * 4;
    size_t avail = vkc_dev_avail(), need = vkc_fit_reserve() + blocks + work;
    if (avail / 10 * 9 < need) {
        g_zic_on = 0;
        fprintf(stderr, "[VK] zimage: the device (%.1f GiB free) does not hold the 30 blocks and a step (%.1f GiB); "
                        "the matrices run on the device one by one\n", avail / 1073741824.0, need / 1073741824.0);
    }
}

/* The weights and the norms, once. */
static int zic_setup(Dit *d) {
    int hd = d->hd, D = d->dim, Lr = d->layers;
    for (int l = 0; l < Lr; l++) {
        Lin *m[7]; zic_block_mats(&d->B[l], m);
        for (int k = 0; k < 7; k++)
            if (!zic_tensor(m[k])) { zic_off("the device refused a block's matrices"); return 0; }
    }
    size_t per = (size_t)2 * hd + 2 * D;
    float *n = malloc((size_t)Lr * per * sizeof(float));
    if (!n) return 0;
    for (int l = 0; l < Lr; l++) {
        float *p = n + (size_t)l * per;
        memcpy(p, d->B[l].nq, (size_t)hd * 4); memcpy(p + hd, d->B[l].nk, (size_t)hd * 4);
        memcpy(p + 2 * hd, d->B[l].an2, (size_t)D * 4); memcpy(p + 2 * hd + D, d->B[l].fn2, (size_t)D * 4);
    }
    g_zic.prm = vkc_buf((size_t)Lr * per * 4, VKC_DEV);
    int ok = g_zic.prm && vkc_begin() && vkc_write(g_zic.prm, 0, n, (size_t)Lr * per * 4) && vkc_submit(1);
    free(n);
    g_zic.mod = vkc_buf((size_t)Lr * 4 * D * 4, VKC_UP);
    if (!ok || !g_zic.mod) { zic_off("the device refused the parameters' buffer"); return 0; }
    g_zic.ok = 1;
    fprintf(stderr, "[VK] zimage chain: %d blocks on the device\n", Lr);
    return 1;
}

static int zic_buffers(Dit *d, int T, int Nx) {
    size_t D = d->dim, M = d->mlp, t = T;
    int ok = vkc_reserve(&g_zic.x, t * D * 4, VKC_DEV) && vkc_reserve(&g_zic.h, t * D * 4, VKC_DEV) &&
             vkc_reserve(&g_zic.a, t * D * 4, VKC_DEV) && vkc_reserve(&g_zic.g, t * M * 4, VKC_DEV) &&
             vkc_reserve(&g_zic.u, t * M * 4, VKC_DEV) && vkc_reserve(&g_zic.qkv, 3 * t * D * 4, VKC_DEV) &&
             vkc_reserve(&g_zic.cs, t * d->hd * 4, VKC_UP) && vkc_reserve(&g_zic.xd, (size_t)Nx * D * 4, VKC_DOWN);
    if (!ok) { zic_off("the device refused a step's buffers"); return 0; }
    return 1;
}

/* The attention over every row in slices of rows that end their submission (COLI_VK_ATTN_SLICE): a big picture's attention is
 * seconds a block on an integrated GPU, past a driver's job timeout. */
static int zic_attn(const Dit *d, int T, size_t KO, size_t VO) {
    int D = d->dim, H = d->heads, hd = d->hd;
    long long b = vkc_attn_slice_budget(), per = (long long)T * H * hd;
    int rr = b > 0 ? (int)(b / per / 16 * 16) : T;
    if (rr < 16) rr = 16;
    int ok = 1;
    for (int r0 = 0; ok && r0 < T; r0 += rr) {
        int n = T - r0 < rr ? T - r0 : rr;
        VkcAttnFull f = {n, T, H, hd, r0 * D, D, (int)KO, (int)VO, D, r0 * D, D, 1.f / sqrtf((float)hd)};
        ok = vkc_attn_full(g_zic.qkv, g_zic.a, &f);
        ok = ok && (r0 + rr >= T || (vkc_submit(0) && vkc_begin()));
    }
    return ok;
}

/* The 30 main blocks over u[T][D] (T = Nx + Nc rows, the image rows first), on the device. cs/sn are the rows' RoPE tables
 * [T][hd/2], adaln the timestep embedding. On success u's first Nx rows hold the image rows after the last block and 1 is
 * returned; 0: not taken (the CPU runs the blocks). */
static int zic_layers(Dit *d, float *u, int T, int Nx, const float *cs, const float *sn, const float *adaln) {
    if (!g_zic.planned) zic_plan(d);
    if (g_zic_on <= 0 || g_zic.failed || vkc_lost()) return 0;
    if (!g_zic.ok && !zic_setup(d)) return 0;
    int D = d->dim, H = d->heads, hd = d->hd, M = d->mlp;
    if (hd > 128 || hd % 2 || D > 4096) { zic_off("a head or a row its ops do not take"); return 0; }
    if (!vkc_attn_full_ready()) { zic_off("the full attention does not load for this head size"); return 0; }
    if (T != g_zic.T && !zic_buffers(d, T, Nx)) return 0;
    g_zic.T = T;
    {                                      /* the rows' rotations: (cos, sin) pairs, a couple of MB */
        float *c = vkc_ptr(g_zic.cs);
        for (int r = 0; r < T; r++)
            for (int j = 0; j < hd / 2; j++) {
                c[(size_t)r * hd + 2 * j] = cs[(size_t)r * (hd / 2) + j];
                c[(size_t)r * hd + 2 * j + 1] = sn[(size_t)r * (hd / 2) + j];
            }
    }
    /* the step's modulation: every block's four vectors */
    float *mod = fmalloc(4 * (size_t)D), *mv = vkc_ptr(g_zic.mod);
    for (int l = 0; l < d->layers; l++) {
        zblock_mod(&d->B[l], adaln, mod);
        const ZBlock *B = &d->B[l];
        float *o = mv + (size_t)l * 4 * D;
        for (int i = 0; i < D; i++) {
            o[i] = B->an1[i] * (1.f + mod[i]); o[D + i] = tanhf(mod[D + i]);
            o[2 * D + i] = B->fn1[i] * (1.f + mod[2 * D + i]); o[3 * D + i] = tanhf(mod[3 * D + i]);
        }
    }
    free(mod);
    size_t KO = (size_t)T * D, VO = 2 * KO, per = (size_t)2 * hd + 2 * D;
    VkcEw sw = {VKC_EW_SWIGLU, T * M, M, 1, 0, 0, 0, 0, 0, 0, 0, 1.f};
    VkcMlaRow rq = {T * H, H, hd, hd, VKC_ROPE_INTERLEAVED, 0, D, hd, 0, D, hd, 0, hd, 0, 0, 0, 0.f}, rk = rq;
    rk.x_off = rk.y_off = (int)KO;
    int ok = vkc_begin() && vkc_write(g_zic.x, 0, u, (size_t)T * D * 4);
    for (int l = 0; ok && l < d->layers; l++) {
        ZBlock *B = &d->B[l];
        VkcNorm n1 = {T, D, 1, 0, D, D, 0, D, D, l * 4 * D, 0, 0, d->eps, 1.f}, n2 = n1, na = n1, nf = n1;
        n2.w_off = l * 4 * D + 2 * D;
        na.x_off = na.y_off = 0; na.w_off = (int)((size_t)l * per + 2 * hd);        /* the post-attention norm */
        nf.w_off = (int)((size_t)l * per + 2 * hd + D);                              /* the post-MLP norm */
        VkcNorm nq = {T * H, hd, H, 0, D, hd, 0, D, hd, (int)((size_t)l * per), 0, 0, d->eps, 1.f}, nk = nq;
        nk.x_off = nk.y_off = (int)KO; nk.w_off = (int)((size_t)l * per + hd);
        VkcEw g1 = {VKC_EW_GATE_ADD, T * D, D, 1, 0, 0, 0, 0, 0, 0, l * 4 * D + D, 1.f}, g2 = g1;
        g2.e_off = l * 4 * D + 3 * D;
        ok = vkc_norm(g_zic.x, g_zic.mod, g_zic.h, &n1) &&
             vkc_matmul(zic_tensor(&B->q), g_zic.h, 0, g_zic.qkv, 0, T) &&
             vkc_matmul(zic_tensor(&B->k), g_zic.h, 0, g_zic.qkv, KO, T) &&
             vkc_matmul(zic_tensor(&B->v), g_zic.h, 0, g_zic.qkv, VO, T) &&
             vkc_norm(g_zic.qkv, g_zic.prm, g_zic.qkv, &nq) && vkc_norm(g_zic.qkv, g_zic.prm, g_zic.qkv, &nk) &&
             vkc_mla_rope(g_zic.qkv, g_zic.cs, g_zic.qkv, &rq) && vkc_mla_rope(g_zic.qkv, g_zic.cs, g_zic.qkv, &rk) &&
             zic_attn(d, T, KO, VO) &&
             vkc_matmul(zic_tensor(&B->o), g_zic.a, 0, g_zic.h, 0, T) &&
             vkc_norm(g_zic.h, g_zic.prm, g_zic.h, &na) &&
             vkc_ew(g_zic.x, g_zic.x, g_zic.h, NULL, g_zic.mod, &g1) &&
             vkc_norm(g_zic.x, g_zic.mod, g_zic.h, &n2) &&
             vkc_matmul(zic_tensor(&B->w1), g_zic.h, 0, g_zic.g, 0, T) &&
             vkc_matmul(zic_tensor(&B->w3), g_zic.h, 0, g_zic.u, 0, T) &&
             vkc_ew(g_zic.g, g_zic.g, g_zic.u, NULL, NULL, &sw) &&
             vkc_matmul(zic_tensor(&B->w2), g_zic.g, 0, g_zic.h, 0, T) &&
             vkc_norm(g_zic.h, g_zic.prm, g_zic.h, &nf) &&
             vkc_ew(g_zic.x, g_zic.x, g_zic.h, NULL, g_zic.mod, &g2) &&
             vkc_submit(0);   /* a submission a block: no single one runs for seconds */
        ok = ok && vkc_begin();
    }
    ok = ok && vkc_copy(g_zic.xd, 0, g_zic.x, 0, (size_t)Nx * D) && vkc_submit(1);
    if (!ok) {
        if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(0); }
        zic_off("a frame failed (the device was lost); this step runs again on the CPU");
        return 0;
    }
    memcpy(u, vkc_ptr(g_zic.xd), (size_t)Nx * D * 4);
    g_zic.steps++;
    return 1;
}

static void zic_report(void) {
    if (!g_zic.ok || !g_zic.steps) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] zimage chain: %llu steps, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                    "%.1f ms waiting for the device, %.1f MiB on the device\n",
            g_zic.steps, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, st.dev_bytes / 1048576.0);
}
