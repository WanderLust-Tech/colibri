/* gemma4_chain.h -- Gemma 4's decode step as a dense chain on the Vulkan device (vk_chain.h).
 * Included once by gemma4.c in a COLI_VULKAN build, after the CPU pieces it reuses (moe_routed, rms_w)
 * and before forward_chunk, which calls it; COLI_VK_CHAIN decides (coli_vk_chain_decide, in main).
 *
 * Without it every dense matrix of a decode step is a synchronous coli_vk_matmul (about 200 a token on the
 * 26B-A4B: upload the row, run, download the answer) and the norms, RoPE, attention and activations run on the
 * CPU between them. Here one decode row's layers are recorded into one command buffer and the residual stream
 * stays on the device from the embedding to the last layer.
 *
 * What runs where, for ONE row (a decode step; a prompt chunk or several conversations' rows take the per-matrix
 * path as before):
 *   host: the embedding row, and on a PLE model the row's per-layer inputs (forward_chunk's own code), the RoPE
 *     tables of the position; the head after the layers (the per-matrix product over the tied embedding).
 *   device, per layer: input RMSNorm, q/k/v (a layer whose values are its keys copies k before the norms; a layer
 *     that shares another's keys and values has no k/v at all), per-head q/k norms and RoPE, the value norm
 *     (no weight), the new K/V row into the device's cache, attention (scores unscaled, as the model has them;
 *     a sliding layer over its ring, a full layer over everything; head dims up to 512), o_proj, post-attention
 *     norm and the residual add, then the dense MLP (pre-norm, gate, up, GELU-tanh gate, down), post-feed-forward
 *     norm and the add, the per-layer-embedding block, the layer scalar.
 *   a mixture-of-experts layer (26B-A4B): the dense MLP and the experts both read the residual after attention.
 *     The frame ends with that residual coming down; the dense MLP runs on the device while the host does the
 *     router and the routed experts (moe_routed, the CPU's own code and cache), then the experts' normalized sum
 *     goes up and the next frame adds it. Crossing per such layer: H floats down, H floats up.
 *   A dense model (the E2B, E4B and 31B) records the whole forward in one frame.
 *
 * The KV caches, and who owns them. The host's (Seq.K and Seq.V) stay canonical in the layout the CPU keeps: a
 * full layer [position][kv head][head dim], a sliding layer a ring of m->ring rows. Every chain step copies its
 * new rows back, so the CPU path, a prefix reuse and a rewind see what they always saw. The device holds a
 * mirror for the one conversation it last served (owner): a full layer's the same layout, a sliding layer's a ring
 * of exactly `window` rows (all a row reads). Per layer the mirror is valid for the positions [v_lo, v_hi); a step
 * at position p uploads what its window needs outside that, a CPU step (a prompt chunk, another conversation)
 * lowers v_hi to its first position, another conversation takes the mirror over by invalidating it. The new rows
 * reach the host only when the step's last frame has completed, so a device lost in the middle of a step loses
 * nothing: the host's caches describe whole steps, and the same step runs on the CPU.
 *
 * Declines (the per-matrix path runs): more than one row, a row inside a picture (a sliding layer reads the
 * whole picture), the trace on, a matrix that did not reach the device, no room for the caches. COLI_VK_CHAIN=0
 * turns it off. Not the bits of the CPU's (sums in another order), the same as the per-matrix path. */
#include "vk_chain.h"
#include "vk_dmoe.h"   /* the routed experts in the frame: route, experts and sum on the device */

typedef struct {
    int ok, failed;
    int L;
    VkcBuf *prm;                               /* every norm weight */
    size_t *o_in, *o_pa, *o_pf, *o_pof, *o_p1, *o_qn, *o_kn, *o_ple;
    ColiVkTensor **t_q, **t_k, **t_v, **t_o, **t_g, **t_u, **t_d, **t_pg, **t_pp;
    ColiVkTensor *t_head;                      /* the tied embedding as the head (NULL: the head stays with the caller) */
    size_t o_final;                            /* the final norm's weight in prm */
    VkcBuf **kc, **vc;                         /* the device's mirror of each layer that owns keys and values */
    int *rows_d;                               /* its rows: max_t for a full layer, window for a sliding one */
    int *v_lo, *v_hi;                          /* positions [v_lo, v_hi) equal the host's */
    void *owner;                               /* the Seq the mirror follows */
    size_t *o_kvd;                             /* where a layer's new K and V rows sit in kvd */
    VkcBuf *xu, *x, *nrm, *tmp, *q, *k, *v, *ctx, *g, *u, *pg, *plev, *cs, *xd, *kvd, *routed, *fin, *lg, *lgd;
    float *stage;                              /* a sliding layer's rows packed in position order for one upload */
    size_t stage_floats;
    float *host_h;                             /* the experts' output as normalized on the host */
    /* a mixture layer whose experts the tier holds all of runs its router and experts in the frame (g4c_moe_dev): */
    VkcDmoe dm;                                /* vk_dmoe.h: the tier's table, the ids and weights, the hidden rows, the sum */
    ColiVkTensor **t_rt;                       /* the routers' projections (f32 [E][H]) */
    size_t *o_rs, *o_pf2, *o_pof2;             /* in prm: router.scale / sqrt(H), pre_feedforward_layernorm_2, post_feedforward_layernorm_2 */
    VkcBuf *pes, *rn, *rlg, *xe, *rt2;         /* router.per_expert_scale [L][E], the router's input row, its logits, the experts' input row, their normalized sum */
    uint8_t *devl;                             /* the layers that ran their experts in this step's frame */
    int dmoe_ok;                               /* the buffers above exist */
    unsigned long long dev_layers;             /* layer steps whose experts ran in the frame */
    int rowcap;                                /* the most rows a step takes: a prompt chunk's (m->chunk), or 1 when the buffers for it were refused */
    unsigned long long forwards, frames, declined, chunks, chunk_rows;   /* chunks: prompt chunks stepped here, chunk_rows their rows */
    double wait_ms, host_ms, t_sync, t_rec, t_post, t_chunk;   /* t_*: the step's host time before the frame, recording it, after it */
} G4Chain;

static int g_vk_chain = 0;     /* COLI_VK_CHAIN decided on, and the chain's pipelines are up */

static Model *g_g4c_model;     /* for the report at exit */
/* COLI_VK_CHAIN_CHECK=1: every chained step is run again on the CPU path from the same input, and the two residuals compared
 * (the check costs a CPU step each: for tests and a first look at a new model). The worst difference is reported at exit. */
static int g_g4c_check;
static double g_g4c_worst;
static long g_g4c_checked;
static G4Chain *g4c_of(Model *m) { return (G4Chain *)m->vkchain; }

/* the matrix on the device, made the way the per-matrix path makes it (fmt 1: int8 rows and a scale a row) */
static ColiVkTensor *g4c_tensor(QMat *W) {
    if (!W->q) return NULL;
    if (W->vk) return (ColiVkTensor *)W->vk;
    if (W->vk_no) return NULL;
    if (!coli_vk_tensor_ensure((ColiVkTensor **)&W->vk, W->q, W->s, 1, W->I, W->O, 0)) { W->vk_no = 1; return NULL; }
    return (ColiVkTensor *)W->vk;
}

/* The device memory the chain's KV mirrors take (g4c_setup's `need`, from the same rows), for the expert tier to leave room for. */
static size_t g4c_mirror_bytes(const Model *m) {
    const Cfg *c = &m->c; size_t need = 0;
    for (int l = 0; l < c->n_layers; l++) {
        const Layer *y = &m->L[l];
        if (y->kv_src >= 0) continue;
        size_t rows = y->full ? (size_t)m->max_t : (size_t)(c->window + m->chunk < m->ring ? c->window + m->chunk : m->ring);
        need += rows * (size_t)y->nkv * y->hd * 2 * sizeof(float);
    }
    return need;
}

/* ---- setup ------------------------------------------------------------------------ */

/* The step's buffers for up to rc rows (a prompt chunk's; the head's, which only a one-row step uses, are one row's). The new rows
 * of the keys and values come down together in kvd: per layer, rc rows of keys then rc rows of values (o_kvd[l]). */
static int g4c_bufs(G4Chain *ch, Model *m, int rc) {
    Cfg *c = &m->c; int L = c->n_layers, H = c->hidden, P = c->ple_dim;
    VkcBuf **all[] = {&ch->xu, &ch->x, &ch->nrm, &ch->tmp, &ch->q, &ch->k, &ch->v, &ch->ctx, &ch->g, &ch->u, &ch->pg, &ch->plev, &ch->cs,
                      &ch->xd, &ch->kvd, &ch->routed, &ch->fin, &ch->lg, &ch->lgd};
    for (size_t i = 0; i < sizeof all / sizeof *all; i++) if (*all[i]) { vkc_free(*all[i]); *all[i] = NULL; }
    size_t kvd = 0, R = (size_t)rc;
    for (int l = 0; l < L; l++) {
        Layer *y = &m->L[l];
        if (y->kv_src >= 0) continue;
        ch->o_kvd[l] = kvd; kvd += 2 * (size_t)y->nkv * y->hd * R;
    }
    int hdm = c->hd_g > c->hd_s ? c->hd_g : c->hd_s, nh = c->n_heads;
    int kvm = c->nkv_s * c->hd_s > c->nkv_g * c->hd_g ? c->nkv_s * c->hd_s : c->nkv_g * c->hd_g;
    size_t ffm = (size_t)m->max_ff;
    int ok = (ch->xu = vkc_buf(R * H * 4, VKC_UP)) && (ch->x = vkc_buf(R * H * 4, VKC_DEV)) &&
             (ch->nrm = vkc_buf(R * H * 4, VKC_DEV)) && (ch->tmp = vkc_buf(R * H * 4, VKC_DEV)) &&
             (ch->q = vkc_buf(R * nh * hdm * 4, VKC_DEV)) && (ch->ctx = vkc_buf(R * nh * hdm * 4, VKC_DEV)) &&
             (ch->k = vkc_buf(R * kvm * 4, VKC_DEV)) && (ch->v = vkc_buf(R * kvm * 4, VKC_DEV)) &&
             (ch->g = vkc_buf(R * ffm * 4, VKC_DEV)) && (ch->u = vkc_buf(R * ffm * 4, VKC_DEV)) &&
             (ch->pg = vkc_buf(R * (P ? P : 1) * 4, VKC_DEV)) && (ch->plev = vkc_buf(R * (P ? L * P : 1) * 4, VKC_UP)) &&
             (ch->cs = vkc_buf(R * (c->hd_s + c->hd_g) * 4, VKC_UP)) && (ch->xd = vkc_buf(R * H * 4, VKC_DOWN)) &&
             (ch->kvd = vkc_buf((kvd ? kvd : 1) * 4, VKC_DOWN)) && (ch->routed = vkc_buf(R * H * 4, VKC_UP));
    if (ok && ch->t_head) {
        ok = (ch->fin = vkc_buf((size_t)H * 4, VKC_DEV)) && (ch->lg = vkc_buf((size_t)c->vocab * 4, VKC_DEV)) &&
             (ch->lgd = vkc_buf((size_t)c->vocab * 4, VKC_DOWN));
        if (!ok) { ch->t_head = NULL; ok = 1; }          /* no room for the logits: the head stays with the caller */
    }
    return ok;
}

static G4Chain *g4c_setup(Model *m) {
    G4Chain *ch = g4c_of(m);
    if (ch) return ch->ok ? ch : NULL;
    ch = (G4Chain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    m->vkchain = ch;
    g_g4c_model = m;
    Cfg *c = &m->c; int L = c->n_layers, H = c->hidden, P = c->ple_dim;
    ch->L = L;
    if (c->hd_s > 512 || c->hd_g > 512 || ((c->hd_s > 256 || c->hd_g > 256) && !vkc_attn_wide())) {
        fprintf(stderr, "[VK] gemma4 chain: head dims %d/%d need chain_attn512.spv (up to 512); per-matrix path\n", c->hd_s, c->hd_g);
        return NULL;
    }
#define ALLOC(p, n, sz) (p = calloc((size_t)(n), sz))
    if (!ALLOC(ch->o_in, L, sizeof(size_t)) || !ALLOC(ch->o_pa, L, sizeof(size_t)) || !ALLOC(ch->o_pf, L, sizeof(size_t)) ||
        !ALLOC(ch->o_pof, L, sizeof(size_t)) || !ALLOC(ch->o_p1, L, sizeof(size_t)) || !ALLOC(ch->o_qn, L, sizeof(size_t)) ||
        !ALLOC(ch->o_kn, L, sizeof(size_t)) || !ALLOC(ch->o_ple, L, sizeof(size_t)) || !ALLOC(ch->o_kvd, L, sizeof(size_t)) ||
        !ALLOC(ch->t_q, L, sizeof(void *)) || !ALLOC(ch->t_k, L, sizeof(void *)) || !ALLOC(ch->t_v, L, sizeof(void *)) ||
        !ALLOC(ch->t_o, L, sizeof(void *)) || !ALLOC(ch->t_g, L, sizeof(void *)) || !ALLOC(ch->t_u, L, sizeof(void *)) ||
        !ALLOC(ch->t_d, L, sizeof(void *)) || !ALLOC(ch->t_pg, L, sizeof(void *)) || !ALLOC(ch->t_pp, L, sizeof(void *)) ||
        !ALLOC(ch->kc, L, sizeof(void *)) || !ALLOC(ch->vc, L, sizeof(void *)) || !ALLOC(ch->rows_d, L, sizeof(int)) ||
        !ALLOC(ch->v_lo, L, sizeof(int)) || !ALLOC(ch->v_hi, L, sizeof(int)) || !ALLOC(ch->t_rt, L, sizeof(void *)) ||
        !ALLOC(ch->o_rs, L, sizeof(size_t)) || !ALLOC(ch->o_pf2, L, sizeof(size_t)) || !ALLOC(ch->o_pof2, L, sizeof(size_t)) ||
        !ALLOC(ch->devl, L, 1))
        return NULL;
#undef ALLOC
    /* the matrices: the device copies the per-matrix path uploads */
    for (int l = 0; l < L; l++) {
        Layer *y = &m->L[l];
        int own = y->kv_src < 0;
        ch->t_q[l] = g4c_tensor(&y->q); ch->t_o[l] = g4c_tensor(&y->o);
        ch->t_g[l] = g4c_tensor(&y->gate); ch->t_u[l] = g4c_tensor(&y->up); ch->t_d[l] = g4c_tensor(&y->down);
        int ok = ch->t_q[l] && ch->t_o[l] && ch->t_g[l] && ch->t_u[l] && ch->t_d[l];
        if (own) {
            ch->t_k[l] = g4c_tensor(&y->k);
            ok = ok && ch->t_k[l];
            if (y->v.q) { ch->t_v[l] = g4c_tensor(&y->v); ok = ok && ch->t_v[l]; }
        }
        if (P) {
            ch->t_pg[l] = g4c_tensor(&y->ple_gate); ch->t_pp[l] = g4c_tensor(&y->ple_proj);
            ok = ok && ch->t_pg[l] && ch->t_pp[l];
        }
        if (!ok) { fprintf(stderr, "[VK] gemma4 chain: a dense matrix of layer %d did not reach the device; per-matrix path\n", l); return NULL; }
    }
    /* the routers' projections as f32 rows (a layer whose does not reach the device keeps the host's router: dmoe_ok 0) */
    int rt_ok = c->moe && c->n_experts > 0;
    if (rt_ok) for (int l = 0; l < L; l++) {
        Layer *y = &m->L[l];
        if (!y->router_proj || !coli_vk_tensor_ensure_f32(&ch->t_rt[l], y->router_proj, H, c->n_experts, 0)) { rt_ok = 0; break; }
    }
    /* the norm weights, one arena */
    size_t n = 0;
    for (int l = 0; l < L; l++) {
        Layer *y = &m->L[l];
        ch->o_in[l] = n; n += H; ch->o_pa[l] = n; n += H; ch->o_pf[l] = n; n += H; ch->o_pof[l] = n; n += H;
        if (c->moe) { ch->o_p1[l] = n; n += H; ch->o_rs[l] = n; n += H; ch->o_pf2[l] = n; n += H; ch->o_pof2[l] = n; n += H; }
        ch->o_qn[l] = n; n += y->hd;
        if (y->kv_src < 0) { ch->o_kn[l] = n; n += y->hd; }
        if (P) { ch->o_ple[l] = n; n += H; }
    }
    ch->o_final = n; n += H;
    float *arena = calloc(n ? n : 1, sizeof(float));
    if (!arena) return NULL;
    memcpy(arena + ch->o_final, m->final_norm, H * sizeof(float));
    for (int l = 0; l < L; l++) {
        Layer *y = &m->L[l];
        memcpy(arena + ch->o_in[l], y->in_ln, H * sizeof(float));
        memcpy(arena + ch->o_pa[l], y->post_attn_ln, H * sizeof(float));
        memcpy(arena + ch->o_pf[l], y->pre_ff_ln, H * sizeof(float));
        memcpy(arena + ch->o_pof[l], y->post_ff_ln, H * sizeof(float));
        if (c->moe) {
            memcpy(arena + ch->o_p1[l], y->post_ff_ln1, H * sizeof(float));
            const float rscale = 1.f / sqrtf((float)H);      /* moe_routed: rms_nw(x) * router_scale * rscale, one weighted norm here */
            for (int i = 0; i < H; i++) arena[ch->o_rs[l] + i] = y->router_scale[i] * rscale;
            memcpy(arena + ch->o_pf2[l], y->pre_ff_ln2, H * sizeof(float));
            memcpy(arena + ch->o_pof2[l], y->post_ff_ln2, H * sizeof(float));
        }
        memcpy(arena + ch->o_qn[l], y->q_norm, y->hd * sizeof(float));
        if (y->kv_src < 0) memcpy(arena + ch->o_kn[l], y->k_norm, y->hd * sizeof(float));
        if (P) memcpy(arena + ch->o_ple[l], y->ple_norm, H * sizeof(float));
    }
    ch->prm = vkc_buf((n ? n : 1) * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, arena, n * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) return NULL;
    /* the caches: a full layer as long as the context, a sliding layer the window */
    size_t need = 0;
    for (int l = 0; l < L; l++) {
        Layer *y = &m->L[l];
        if (y->kv_src >= 0) continue;
        /* a sliding layer's ring: the window and a prompt chunk (the chunk's rows are all written before any is read) */
        ch->rows_d[l] = y->full ? m->max_t : (c->window + m->chunk < m->ring ? c->window + m->chunk : m->ring);
        need += (size_t)ch->rows_d[l] * y->nkv * y->hd * 2 * sizeof(float);
    }
    if (g_gvk_reserve > 0 && (double)coli_vk_free_bytes() < (double)need + g_gvk_reserve * 1073741824.0) {
        fprintf(stderr, "[VK] gemma4 chain: the KV mirrors need %.1f MiB and the card has less than %.1f GB free beyond them (G4_VK_RESERVE_GB); per-matrix path\n",
                need / 1048576.0, g_gvk_reserve);
        return NULL;
    }
    size_t stage = 0;
    for (int l = 0; l < L; l++) {
        Layer *y = &m->L[l];
        if (y->kv_src >= 0) continue;
        size_t row = (size_t)y->nkv * y->hd;
        ch->kc[l] = vkc_buf((size_t)ch->rows_d[l] * row * sizeof(float), VKC_DEV);
        ch->vc[l] = vkc_buf((size_t)ch->rows_d[l] * row * sizeof(float), VKC_DEV);
        if (!ch->kc[l] || !ch->vc[l]) { fprintf(stderr, "[VK] gemma4 chain: device memory for the KV mirrors refused; per-matrix path\n"); return NULL; }
        if (!y->full && (size_t)ch->rows_d[l] * row > stage) stage = (size_t)ch->rows_d[l] * row;
    }
    ch->t_head = g4c_tensor(&m->embed);        /* the head fused into the step when it is on the device, else the caller's */
    ch->rowcap = m->chunk > 1 && !(getenv("G4_VK_CHUNK") && *getenv("G4_VK_CHUNK") == '0') ? m->chunk : 1;   /* G4_VK_CHUNK=0: decode rows only */
    ok = g4c_bufs(ch, m, ch->rowcap);
    if (!ok && ch->rowcap > 1) { ch->rowcap = 1; ok = g4c_bufs(ch, m, 1); }   /* no room for a chunk's rows: decode only */
    if (!ok) { fprintf(stderr, "[VK] gemma4 chain: device memory for the step's buffers refused; per-matrix path\n"); return NULL; }
    if (rt_ok) {   /* the experts in the frame: the per-expert scales and the router's and the experts' rows */
        int E = c->n_experts;
        float *pes = malloc((size_t)L * E * sizeof(float));
        size_t R = (size_t)ch->rowcap;
        ok = pes && (ch->pes = vkc_buf((size_t)L * E * 4, VKC_DEV)) && (ch->rn = vkc_buf(R * H * 4, VKC_DEV)) &&
             (ch->rlg = vkc_buf(R * E * 4, VKC_DEV)) && (ch->xe = vkc_buf(R * H * 4, VKC_DEV)) && (ch->rt2 = vkc_buf(R * H * 4, VKC_DEV));
        if (ok) {
            for (int l = 0; l < L; l++) memcpy(pes + (size_t)l * E, m->L[l].router_pes, (size_t)E * sizeof(float));
            ok = vkc_begin() && vkc_write(ch->pes, 0, pes, (size_t)L * E * 4) && vkc_submit(1);
        }
        free(pes);
        ch->dmoe_ok = ok;
        ch->dm.big = 1; ch->dm.act = 2;      /* a prompt chunk's experts too: the grouped GEMMs, GeGLU */
        /* in floats: on the matrix units (f16 operands) the 26B-A4B's prompt moved 80 of 679 positions' next-token choices and the last row's logits by 1.5
         * (the CPU's against the floats': 3 of 679 and 0.02); G4_VK_CHUNK_COOP=1 takes the matrix units */
        ch->dm.floats = !(getenv("G4_VK_CHUNK_COOP") && atoi(getenv("G4_VK_CHUNK_COOP")));
    }
    ch->stage = malloc((stage ? stage : 1) * sizeof(float)); ch->stage_floats = stage;
    ch->host_h = malloc((size_t)ch->rowcap * H * sizeof(float));
    if (!ch->stage || !ch->host_h) return NULL;
    ch->ok = 1;
    fprintf(stderr, "[VK] gemma4 chain: %d layers on the device (%d keep keys and values of their own), %.1f MiB of KV mirrors%s\n",
            L, (int)(L - (c->kv_shared > 0 ? c->kv_shared : 0)), need / 1048576.0, c->moe ? (ch->dmoe_ok ? ", the routed experts in the frame when the tier holds them" : ", the routed experts on the host") : "");
    return ch;
}

/* ---- the KV mirrors ---------------------------------------------------------------- */

/* A CPU step (a prompt chunk, a batch row) writes the host's rows from its first position on: the device's no longer match there. */
static void g4c_cpu_step(Model *m, Seq *q, Seq *const *qs, const int *ps, int S, int pos0) {
    G4Chain *ch = g4c_of(m);
    if (!ch || !ch->ok) return;
    for (int s = 0; s < S; s++) {
        Seq *cq = qs ? qs[s] : q;
        if (cq != ch->owner) continue;
        int pos = ps ? ps[s] : pos0 + s;
        for (int l = 0; l < ch->L; l++) {
            if (ch->v_hi[l] > pos) ch->v_hi[l] = pos;
            if (ch->v_lo[l] > ch->v_hi[l]) ch->v_lo[l] = ch->v_hi[l];
        }
    }
}

/* host rows [t0, t1) of layer l's cache (K or V) into the device mirror */
static int g4c_push(G4Chain *ch, Model *m, int l, VkcBuf *dst, const float *host, int t0, int t1) {
    Layer *y = &m->L[l];
    size_t row = (size_t)y->nkv * y->hd;
    if (y->full) {                                        /* position t is row t on both sides: contiguous, in pieces of about 16 MiB */
        size_t piece = (size_t)(16 << 20) / (row * sizeof(float)); if (piece < 1) piece = 1;
        for (size_t a = (size_t)t0; a < (size_t)t1; a += piece) {
            size_t n = (size_t)t1 - a < piece ? (size_t)t1 - a : piece;
            if (!vkc_begin() || !vkc_write(dst, a * row, host + a * row, n * row * sizeof(float)) || !vkc_submit(1)) return 0;
        }
        return 1;
    }
    int R = ch->rows_d[l], n = t1 - t0;
    for (int i = 0; i < n; i++) memcpy(ch->stage + (size_t)i * row, host + (size_t)((t0 + i) % m->ring) * row, row * sizeof(float));
    if (!vkc_begin()) return 0;
    for (int done = 0; done < n;) {                       /* the device ring wraps: at most two runs */
        int slot = (t0 + done) % R, run = R - slot < n - done ? R - slot : n - done;
        if (!vkc_write(dst, (size_t)slot * row, ch->stage + (size_t)done * row, (size_t)run * row * sizeof(float))) return 0;
        done += run;
    }
    return vkc_submit(1);
}

/* make the mirrors hold what a step at position p reads */
static int g4c_sync(G4Chain *ch, Model *m, Seq *cq, int p) {
    if (ch->owner != cq) {
        for (int l = 0; l < ch->L; l++) ch->v_lo[l] = ch->v_hi[l] = 0;
        ch->owner = cq;
    }
    for (int l = 0; l < ch->L; l++) {
        Layer *y = &m->L[l];
        if (y->kv_src >= 0) continue;
        int lo = y->full ? 0 : (p - m->c.window + 1 > 0 ? p - m->c.window + 1 : 0);
        if (ch->v_hi[l] > p) ch->v_hi[l] = p;
        if (ch->v_lo[l] > ch->v_hi[l]) ch->v_lo[l] = ch->v_hi[l];
        int from = (ch->v_lo[l] <= lo && ch->v_hi[l] >= lo) ? (ch->v_hi[l] > lo ? ch->v_hi[l] : lo) : lo;
        if (from < p && (!g4c_push(ch, m, l, ch->kc[l], cq->K[l], from, p) || !g4c_push(ch, m, l, ch->vc[l], cq->V[l], from, p))) return 0;
        ch->v_lo[l] = lo; ch->v_hi[l] = p;
    }
    return 1;
}

/* ---- the layer's pieces ------------------------------------------------------------ */

/* Every piece takes the step's S rows (1 for a decode row, up to a chunk for a prompt): S rows of the residual are x[S][H], the rows of
 * the projections follow them, and the elementwise ones run over S times a row. */

static int g4c_rows_norm(VkcBuf *x, size_t xo, VkcBuf *w, size_t wo, VkcBuf *y, size_t yo, int D, float eps, int S) {
    VkcNorm p = {S, D, 1, (int)xo, D, D, (int)yo, D, D, (int)wo, 0, 0, eps, 1.f};
    return vkc_norm(x, w, y, &p);
}
/* n heads of D floats a row, in place, one weight for all of them (flags: VKC_NORM_NOW for no weight) */
static int g4c_head_norm(VkcBuf *x, VkcBuf *w, size_t wo, int n, int D, int flags, float eps, int S) {
    VkcNorm p = {S * n, D, n, 0, n * D, D, 0, n * D, D, (int)wo, 0, flags, eps, 1.f};
    return vkc_norm(x, w, x, &p);
}
static int g4c_add(VkcBuf *x, VkcBuf *b, int n) {
    VkcEw p = {VKC_EW_ADD, n, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(x, x, b, NULL, NULL, &p);
}
static int g4c_geglu(VkcBuf *g, VkcBuf *u, int n, int u_off) {
    VkcEw p = {VKC_EW_GEGLU, n, 1, 1, 0, 1, 0, 0, u_off, 0, 0, 1.f};
    return vkc_ew(g, g, u, NULL, NULL, &p);
}

/* attention of layer l through the residual add: x += post_norm(o_proj(attention(...))); a layer that owns keys and values puts
 * its new rows (positions p..p+S-1) in the device's cache and in kvd for the host. A sliding layer's ring may wrap inside the rows. */
static int g4c_attn(G4Chain *ch, Model *m, int l, int p, int S) {
    Cfg *c = &m->c; Layer *y = &m->L[l];
    int H = c->hidden, nh = c->n_heads, hd = y->hd, nkv = y->nkv, own = y->kv_src < 0, ks = own ? l : y->kv_src;
    int cs_off = y->full ? c->hd_s : 0, cs_row = c->hd_s + c->hd_g;   /* a row's table: the sliding one first (hd_s floats), then the full one */
    size_t row = (size_t)nkv * hd;
    int ok = g4c_rows_norm(ch->x, 0, ch->prm, ch->o_in[l], ch->nrm, 0, H, c->eps, S) &&
             vkc_matmul(ch->t_q[l], ch->nrm, 0, ch->q, 0, S);
    if (ok && own) {
        ok = vkc_matmul(ch->t_k[l], ch->nrm, 0, ch->k, 0, S);
        if (ok) ok = ch->t_v[l] ? vkc_matmul(ch->t_v[l], ch->nrm, 0, ch->v, 0, S) : vkc_copy(ch->v, 0, ch->k, 0, (size_t)S * row);   /* the values are the raw keys */
    }
    ok = ok && g4c_head_norm(ch->q, ch->prm, ch->o_qn[l], nh, hd, 0, c->eps, S);
    {
        VkcRope r = {S * nh, nh, 0, nh * hd, hd, hd / 2, cs_off, cs_row};
        ok = ok && vkc_rope(ch->q, ch->cs, &r);
    }
    if (ok && own) {
        VkcRope r = {S * nkv, nkv, 0, nkv * hd, hd, hd / 2, cs_off, cs_row};
        ok = g4c_head_norm(ch->k, ch->prm, ch->o_kn[l], nkv, hd, 0, c->eps, S) && vkc_rope(ch->k, ch->cs, &r) &&
             g4c_head_norm(ch->v, ch->prm, 0, nkv, hd, VKC_NORM_NOW, c->eps, S);
        if (ok && y->full) {                                  /* position t is row t: one run */
            ok = vkc_copy(ch->kc[l], (size_t)p * row, ch->k, 0, (size_t)S * row) && vkc_copy(ch->vc[l], (size_t)p * row, ch->v, 0, (size_t)S * row);
        } else if (ok) {                                      /* the ring: up to two runs */
            int R = ch->rows_d[l], s0 = p % R, first = R - s0 < S ? R - s0 : S;
            ok = vkc_copy(ch->kc[l], (size_t)s0 * row, ch->k, 0, (size_t)first * row) && vkc_copy(ch->vc[l], (size_t)s0 * row, ch->v, 0, (size_t)first * row);
            if (ok && first < S)
                ok = vkc_copy(ch->kc[l], 0, ch->k, (size_t)first * row, (size_t)(S - first) * row) && vkc_copy(ch->vc[l], 0, ch->v, (size_t)first * row, (size_t)(S - first) * row);
        }
        ok = ok && vkc_copy(ch->kvd, ch->o_kvd[l], ch->k, 0, (size_t)S * row) && vkc_copy(ch->kvd, ch->o_kvd[l] + (size_t)S * row, ch->v, 0, (size_t)S * row);
    }
    if (!ok) return 0;
    VkcAttnW a;
    memset(&a, 0, sizeof a);
    a.a = (VkcAttn){.S = S, .H = nh, .KVH = nkv, .hd = hd, .pos_base = p, .q_row = nh * hd, .q_seg = hd, .o_row = nh * hd, .scale = 1.0f};
    a.vd = hd; a.kv_pm = 1;
    if (!y->full) { a.win = c->window; a.ring = ch->rows_d[ks]; }
    return vkc_attn_w(ch->q, ch->kc[ks], ch->vc[ks], ch->ctx, NULL, NULL, NULL, &a) &&
           vkc_matmul(ch->t_o[l], ch->ctx, 0, ch->tmp, 0, S) &&
           g4c_rows_norm(ch->tmp, 0, ch->prm, ch->o_pa[l], ch->tmp, 0, H, c->eps, S) && g4c_add(ch->x, ch->tmp, S * H);
}

/* the dense MLP on the residual: tmp = down(gelu(gate(norm(x))) * up(norm(x))); on a mixture layer its output is then
 * normalized by post_feedforward_layernorm_1 */
static int g4c_mlp(G4Chain *ch, Model *m, int l, int S) {
    Cfg *c = &m->c; Layer *y = &m->L[l];
    int H = c->hidden, ff = y->ff;
    int ok = g4c_rows_norm(ch->x, 0, ch->prm, ch->o_pf[l], ch->nrm, 0, H, c->eps, S) &&
             vkc_matmul(ch->t_g[l], ch->nrm, 0, ch->g, 0, S) && vkc_matmul(ch->t_u[l], ch->nrm, 0, ch->u, 0, S) &&
             g4c_geglu(ch->g, ch->u, S * ff, 0) && vkc_matmul(ch->t_d[l], ch->g, 0, ch->tmp, 0, S);
    if (ok && c->moe) ok = g4c_rows_norm(ch->tmp, 0, ch->prm, ch->o_p1[l], ch->tmp, 0, H, c->eps, S);
    return ok;
}

/* the routed experts of a mixture layer in the frame (vk_dmoe.h): the router on the residual after attention (the same
 * weighted norm and projection as moe_routed, then softmax, the top-K renormalized and times the expert's own scale), the
 * experts on pre_feedforward_layernorm_2 of it as GeGLU, their rank-order sum normalized by post_feedforward_layernorm_2
 * and added to tmp (the dense MLP's, already normalized by post_feedforward_layernorm_1). More than a few rows (a prompt
 * chunk) go through the experts' grouped GEMMs (vkc_moe_rows). */
static int g4c_moe_dev(G4Chain *ch, Model *m, int l, int S) {
    Cfg *c = &m->c;
    int H = c->hidden, E = c->n_experts;
    VkcRoute rp = {.kind = 0, .has_bias = 3, .b_off = l * E, .n_group = 1, .topk_group = 1, .norm = 1, .temp = 1.f, .scale = 1.f};
    VkcMoe mp = {.D = H, .I = c->moe_inter, .act = 2};
    return g4c_rows_norm(ch->x, 0, ch->prm, ch->o_rs[l], ch->rn, 0, H, c->eps, S) &&
           vkc_matmul(ch->t_rt[l], ch->rn, 0, ch->rlg, 0, S) &&
           g4c_rows_norm(ch->x, 0, ch->prm, ch->o_pf2[l], ch->xe, 0, H, c->eps, S) &&
           vkc_dmoe_layer(&ch->dm, l, S, ch->rlg, ch->pes, ch->xe, &rp, &mp) &&
           g4c_rows_norm(ch->dm.y, 0, ch->prm, ch->o_pof2[l], ch->rt2, 0, H, c->eps, S) && g4c_add(ch->tmp, ch->rt2, S * H);
}

/* from tmp (the feed-forward's output, the experts' normalized sum already added on a mixture layer) to the end of
 * the layer: post_feedforward_layernorm, the residual add, the per-layer-embedding block, the layer scalar. The per-layer inputs
 * of the S rows are in plev as [layer][row][P]. */
static int g4c_tail(G4Chain *ch, Model *m, int l, int S) {
    Cfg *c = &m->c; Layer *y = &m->L[l];
    int H = c->hidden, P = c->ple_dim;
    int ok = g4c_rows_norm(ch->tmp, 0, ch->prm, ch->o_pof[l], ch->tmp, 0, H, c->eps, S) && g4c_add(ch->x, ch->tmp, S * H);
    if (ok && P) {
        ok = vkc_matmul(ch->t_pg[l], ch->x, 0, ch->pg, 0, S) && g4c_geglu(ch->pg, ch->plev, S * P, l * S * P) &&
             vkc_matmul(ch->t_pp[l], ch->pg, 0, ch->tmp, 0, S) &&
             g4c_rows_norm(ch->tmp, 0, ch->prm, ch->o_ple[l], ch->tmp, 0, H, c->eps, S) && g4c_add(ch->x, ch->tmp, S * H);
    }
    if (ok) {
        VkcEw p = {VKC_EW_SCALE, S * H, 1, 1, 0, 1, 0, 0, 0, 0, 0, y->layer_scalar};
        ok = vkc_ew(ch->x, ch->x, NULL, NULL, NULL, &p);
    }
    return ok;
}

/* ---- the step ---------------------------------------------------------------------- */

/* One step of S rows of one conversation at positions p..p+S-1: a decode row (S 1), or a chunk of prompt (up to the chunk's rows, none of
 * them inside a picture). The embeddings (and on a PLE model their per-layer inputs) are in m->x and m->ple, the rows' RoPE tables are
 * cs_*, sn_*. On success m->x holds the residual after the last layer, the new K/V rows are in the host's caches (conversation cq) and 1
 * is returned. 0: it declined or a frame failed; nothing the host holds has changed. */
static int g4c_forward(Model *m, Seq *cq, int p, int S, const float *cs_s, const float *sn_s, const float *cs_g, const float *sn_g) {
    if (!g_vk_chain || !g_gvk || m->trace) return 0;
    G4Chain *ch = g4c_setup(m);
    if (!ch || ch->failed || S < 1 || S > ch->rowcap) return 0;
    Cfg *c = &m->c; int H = c->hidden, L = c->n_layers, P = c->ple_dim, hs2 = c->hd_s / 2, hg2 = c->hd_g / 2;
    double t0 = now_s();
    double ts0 = now_s();
    if (!g4c_sync(ch, m, cq, p)) goto lost;
    ch->t_sync += now_s() - ts0;
    double tr0 = now_s();
    memcpy(vkc_ptr(ch->xu), m->x, (size_t)S * H * sizeof(float));
    if (P) {                                                 /* m->ple is [row][layer][P]; the frame wants [layer][row][P] */
        float *pl = (float *)vkc_ptr(ch->plev);
        for (int l = 0; l < L; l++)
            for (int s = 0; s < S; s++) memcpy(pl + ((size_t)l * S + s) * P, m->ple + ((size_t)s * L + l) * P, (size_t)P * sizeof(float));
    }
    {
        float *cs = (float *)vkc_ptr(ch->cs);
        size_t cr = (size_t)c->hd_s + c->hd_g;
        for (int s = 0; s < S; s++) {
            float *r = cs + (size_t)s * cr;
            for (int j = 0; j < hs2; j++) { r[2 * j] = cs_s[(size_t)s * hs2 + j]; r[2 * j + 1] = sn_s[(size_t)s * hs2 + j]; }
            for (int j = 0; j < hg2; j++) { r[c->hd_s + 2 * j] = cs_g[(size_t)s * hg2 + j]; r[c->hd_s + 2 * j + 1] = sn_g[(size_t)s * hg2 + j]; }
        }
    }
    int dmoe = c->moe && ch->dmoe_ok && vkc_dmoe_prepare(&ch->dm, 0, L, c->n_experts, c->topk, H, c->moe_inter, S);
    memset(ch->devl, 0, (size_t)L);
    int ok = vkc_begin() && vkc_copy(ch->x, 0, ch->xu, 0, (size_t)S * H);
    if (ok && dmoe) ok = vkc_dmoe_upload(&ch->dm);        /* the tier's residents changed: the table again */
    int sent = 0;                                           /* layers recorded since the frame last went to the device */
    for (int l = 0; l < L && ok; l++) {
        if (l > 0) { ok = vkc_layer_recorded(&sent); if (!ok) break; }   /* every few layers the frame goes off: the device starts while the host records */
        ok = g4c_attn(ch, m, l, p, S);
        if (!ok) break;
        if (!c->moe) {
            ok = g4c_mlp(ch, m, l, S) && g4c_tail(ch, m, l, S);
            continue;
        }
        if (dmoe && vkc_dmoe_on(&ch->dm, l)) {              /* the tier holds this layer's experts: everything stays in the frame */
            ok = g4c_mlp(ch, m, l, S) && g4c_moe_dev(ch, m, l, S) && g4c_tail(ch, m, l, S);
            ch->devl[l] = ok;
            continue;
        }
        /* the residual after attention comes down; the dense MLP runs while the host does the experts */
        ok = vkc_copy(ch->xd, 0, ch->x, 0, (size_t)S * H);
        double w0 = now_s();
        ok = ok && vkc_submit(1);
        ch->frames++; ch->wait_ms += (now_s() - w0) * 1e3;
        if (!ok) break;
        ok = vkc_begin() && g4c_mlp(ch, m, l, S) && vkc_submit(0);
        if (!ok) break;
        double h0 = now_s();
        moe_routed(m, l, (const float *)vkc_ptr(ch->xd), S);
        for (int s = 0; s < S; s++) rms_w(ch->host_h + (size_t)s * H, m->e_out + (size_t)s * H, m->L[l].post_ff_ln2, H, c->eps);
        memcpy(vkc_ptr(ch->routed), ch->host_h, (size_t)S * H * sizeof(float));
        ch->host_ms += (now_s() - h0) * 1e3;
        ok = vkc_begin() && g4c_add(ch->tmp, ch->routed, S * H) && g4c_tail(ch, m, l, S);
    }
    ok = ok && vkc_copy(ch->xd, 0, ch->x, 0, (size_t)S * H);
    int head = ok && S == 1 && ch->t_head && !g_g4c_check;   /* the head in the same frame (a decode row's; the check wants the caller's, from the CPU's residual) */
    if (head) {
        VkcEw sc = {VKC_EW_SOFTCAP, c->vocab, 1, 1, 0, 1, 0, 0, 0, 0, 0, c->softcap};
        ok = g4c_rows_norm(ch->x, 0, ch->prm, ch->o_final, ch->fin, 0, H, c->eps, 1) &&
             vkc_matmul(ch->t_head, ch->fin, 0, ch->lg, 0, 1) && (c->softcap > 0 ? vkc_ew(ch->lg, ch->lg, NULL, NULL, NULL, &sc) : 1) &&
             vkc_copy(ch->lgd, 0, ch->lg, 0, (size_t)c->vocab);
    }
    double w0 = now_s();
    ch->t_rec += w0 - tr0;
    ok = ok && vkc_submit(1);
    ch->frames++; ch->wait_ms += (now_s() - w0) * 1e3;
    if (!ok) goto lost;
    double tp0 = now_s();
    m->g4_lg = head ? (const float *)vkc_ptr(ch->lgd) : NULL;      /* head_row and head_rows take it for this row */
    if (dmoe) {                                                      /* the layers whose experts ran here: moe_routed's books */
        const int K = c->topk, E = c->n_experts;
        for (int l = 0; l < L; l++) {
            if (!ch->devl[l]) continue;
            const int *ids = vkc_dmoe_ids(&ch->dm, l);
            for (int s = 0; s < S; s++) {
                for (int k = 0; k < K; k++) if (ids[(size_t)s * K + k] >= 0) m->ehit[(size_t)l * E + ids[(size_t)s * K + k]] = 1;
                if (m->vk_hist) rt_count(l, ids + (size_t)s * K, K);
            }
            vkt_dev_count(l, S, K, ids);
            ch->dev_layers++;
        }
    }
    /* the step completed: the new rows into the host's caches (the CPU's own placement), the residual back */
    memcpy(m->x, vkc_ptr(ch->xd), (size_t)S * H * sizeof(float));
    {
        const float *kv = (const float *)vkc_ptr(ch->kvd);
        for (int l = 0; l < L; l++) {
            Layer *y = &m->L[l];
            if (y->kv_src >= 0) continue;
            size_t row = (size_t)y->nkv * y->hd;
            for (int s = 0; s < S; s++) {
                size_t slot = y->full ? (size_t)(p + s) : (size_t)((p + s) % m->ring);
                memcpy(cq->K[l] + slot * row, kv + ch->o_kvd[l] + (size_t)s * row, row * sizeof(float));
                memcpy(cq->V[l] + slot * row, kv + ch->o_kvd[l] + ((size_t)S + s) * row, row * sizeof(float));
            }
            ch->v_hi[l] = p + S;
        }
    }
    ch->forwards++;
    if (S > 1) { ch->chunks++; ch->chunk_rows += S; ch->t_chunk += now_s() - t0; }
    ch->t_post += now_s() - tp0;
    m->t_attn += now_s() - t0;
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU runs the step, and every later one */
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); }
    g_vk_chain = 0;
    ch->failed = 1;
    fprintf(stderr, "[VK] gemma4 chain: the device was lost at position %d; the host's caches hold every earlier position, the CPU runs from here on\n", p);
    return 0;
}

/* the residual the chain made against the CPU's after the same step: the largest |a - b| / max(1, |a|, |b|) over the row */
static void g4c_compare(const float *chain, const float *cpu, int n, int pos) {
    double worst = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)chain[i] - cpu[i]) / fmax(1.0, fmax(fabs(chain[i]), fabs(cpu[i])));
        if (d > worst) worst = d;
    }
    if (worst > g_g4c_worst) g_g4c_worst = worst;
    g_g4c_checked++;
    if (worst > 1e-3) fprintf(stderr, "[VK] gemma4 chain check: position %d differs from the CPU's by %.3e\n", pos, worst);
}

static void g4c_report(void) {
    if (g_g4c_check && g_g4c_checked)
        fprintf(stderr, "[VK] gemma4 chain check: %ld steps compared with the CPU's, the worst residual difference %.3e\n", g_g4c_checked, g_g4c_worst);
    G4Chain *ch = g_g4c_model ? g4c_of(g_g4c_model) : NULL;
    if (!ch || !ch->ok || !ch->forwards) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] gemma4 chain: %llu decode steps, %llu prompt chunks (%llu rows, %.0f ms), %llu waits for the device (%.1f ms in all, %.1f ms of routed experts on the host), %llu layer steps with their experts in the frame, %.1f MiB on the device\n",
            ch->forwards - ch->chunks, ch->chunks, ch->chunk_rows, ch->t_chunk * 1e3, ch->frames, ch->wait_ms, ch->host_ms, ch->dev_layers, st.dev_bytes / 1048576.0);
    vkc_prof_print();   /* COLI_VK_CHAIN_PROF=1: the device time by kind of op */
}
