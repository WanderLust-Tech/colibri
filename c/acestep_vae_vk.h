/* acestep_vae_vk.h: the VAE decoder on a Vulkan device (acestep_gen.h includes it when built with COLI_VULKAN; acv_decode_ws calls it first).
 *
 * On the CPU the decode is the last long step of a song (1.4 s for 10 s of music, 30 s for two minutes). On the card a window is one submission:
 * every convolution is an im2col (chain_codec.comp mode 6: taps of a symmetric, dilated convolution, zero outside the window) into the chain's GEMM,
 * Snake is mode 1 (the codec's: v + sin^2(exp(alpha) v) / (exp(beta) + 1e-9) with the argument reduced first), the transposed convolution is one
 * GEMM to [T][2s * cout] and mode 7's gather-add (each output row takes at most two taps, as the CPU's does), the bias is mode 3 and a residual
 * unit's sum is the chain's ADD. The weights stay f32 (the fold of weight norm is the CPU's, done at load): 0.4 GB for the whole decoder.
 *
 * A window of up to 256 frames (upstream's tile) is 1.1 GB of activations. A device that is missing, a matrix that will not go up, a window that does
 * not fit or a submission that fails leaves that window to the CPU (a window is stateless) and a failed submission takes the card off for the rest of
 * the process. COLI_VK_VAE=0 keeps the decode on the CPU; ACESTEP_VAE_VK_FAULT=N makes the Nth window fail (the test of that way back). */
#ifndef ACESTEP_VAE_VK_H
#define ACESTEP_VAE_VK_H
#ifdef COLI_VULKAN

#include "acestep_vk.h"

typedef struct { ColiVkTensor *t; int off; } AvvW;                     /* the weights on the device; the bias's offset in tab (-1: none) */
typedef struct { AvvW up, c1[3], c2[3]; int sn, a1[3], a2[3]; } AvvBlock;

typedef struct {
    const AcVae *v;
    int tried, ok;
    AvvW conv1, conv2;
    AvvBlock blk[ACV_MAX_BLOCKS];
    int sn_out;
    ColiVkTensor *own[256];                                            /* every matrix, to give back */
    int nown;
    VkcBuf *tab, *buf[3], *zb, *col, *xu, *out;
    size_t nbuf, nzb;
    unsigned long long windows, fault;
} AvvState;

static AvvState avv_state;

typedef struct { float *p; size_t n, cap; } AvvArena;

static int avv_add(AvvArena *a, const float *src, size_t n){
    if (a->n + n > a->cap) {
        size_t c = a->cap ? 2 * a->cap : 65536;
        while (c < a->n + n) c *= 2;
        float *p = (float *)realloc(a->p, c * sizeof(float));
        if (!p) return -1;
        a->p = p; a->cap = c;
    }
    memcpy(a->p + a->n, src, n * sizeof(float));
    a->n += n;
    return (int)(a->n - n);
}

static void avv_release(AvvState *s){
    for (int i = 0; i < s->nown; i++) coli_vk_tensor_free(s->own[i]);
    VkcBuf **all[] = { &s->tab, &s->buf[0], &s->buf[1], &s->buf[2], &s->zb, &s->col, &s->xu, &s->out };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) if (*all[i]) { vkc_free(*all[i]); *all[i] = NULL; }
    memset(s, 0, sizeof *s);
}

static void avv_free_hook(const AcVae *v){
    if (avv_state.v == v) avv_release(&avv_state);
}

static ColiVkTensor *avv_tensor(AvvState *s, const QiMat *m){
    ColiVkTensor *t = NULL;
    if (s->nown >= (int)(sizeof s->own / sizeof s->own[0]) || m->ld || !m->w || m->fmt != QI_F32 || m->K % 2) return NULL;
    if (!coli_vk_tensor_ensure(&t, m->w, NULL, 10, m->K, m->N, 0)) return NULL;
    s->own[s->nown++] = t;
    return t;
}

static int avv_conv_w(AvvState *s, AvvArena *a, AvvW *w, const AcvConv *c){
    w->t = avv_tensor(s, &c->m);
    w->off = c->b ? avv_add(a, c->b, (size_t)c->cout) : -1;
    return w->t != NULL && (!c->b || w->off >= 0);
}

static int avv_snake_w(AvvArena *a, int *off, const AcvSnake *sn){
    *off = avv_add(a, sn->ea, (size_t)sn->c);
    return *off >= 0 && avv_add(a, sn->ib, (size_t)sn->c) >= 0;
}

/* The weights and constants go up once. 1 when the whole decoder is there. */
static int avv_setup(AvvState *s, const AcVae *v){
    if (s->tried) return s->ok;
    s->tried = 1; s->v = v;
    { const char *f = getenv("ACESTEP_VAE_VK_FAULT"); if (f) s->fault = strtoull(f, NULL, 10); }
    if (!acv_dev.chain || !vkc_codec_ready()) return 0;
    AvvArena a = {0};
    int ok = avv_conv_w(s, &a, &s->conv1, &v->conv1) && avv_conv_w(s, &a, &s->conv2, &v->conv2) && avv_snake_w(&a, &s->sn_out, &v->sn);
    for (int i = 0; ok && i < v->nblk; i++) {
        const AcvBlock *b = &v->blk[i];
        AvvBlock *d = &s->blk[i];
        d->up.t = avv_tensor(s, &b->up.m);
        d->up.off = avv_add(&a, b->up.b, (size_t)b->up.cout);
        ok = d->up.t && d->up.off >= 0 && avv_snake_w(&a, &d->sn, &b->sn);
        for (int u = 0; ok && u < 3; u++)
            ok = avv_conv_w(s, &a, &d->c1[u], &b->r[u].c1) && avv_conv_w(s, &a, &d->c2[u], &b->r[u].c2) &&
                 avv_snake_w(&a, &d->a1[u], &b->r[u].s1) && avv_snake_w(&a, &d->a2[u], &b->r[u].s2);
    }
    if (ok) {
        s->tab = vkc_buf(a.n * sizeof(float), VKC_DEV);
        ok = s->tab && vkc_begin() && vkc_write(s->tab, 0, a.p, a.n * sizeof(float)) && vkc_submit(1);
    }
    free(a.p);
    if (!ok) {
        fprintf(stderr, "[VK] acestep vae: the decoder's weights did not reach the device; the CPU runs the VAE\n");
        for (int i = 0; i < s->nown; i++) coli_vk_tensor_free(s->own[i]);
        s->nown = 0;
        if (s->tab) { vkc_free(s->tab); s->tab = NULL; }
    } else fprintf(stderr, "[VK] acestep vae: the decoder is on the card (%.2f GB of f32 weights)\n", (double)v->wbytes / 1e9);
    s->ok = ok;
    return ok;
}

/* y [T][cout] = conv1d(x [T][cin]) with zero padding (k-1)*dil/2 a side, + bias: im2col in tiles of at most 16M floats, each through the GEMM. */
static int avv_conv(const AvvState *s, const AvvW *w, const AcvConv *c, VkcBuf *x, VkcBuf *y, int T){
    const int K = c->k, cin = c->cin, cout = c->cout, kc = K * cin;
    int ok = 1;
    if (K == 1) {                                                      /* no taps to gather: the rows go straight in */
        for (int t0 = 0; ok && t0 < T; t0 += 32768) {
            const int tn = T - t0 < 32768 ? T - t0 : 32768;
            ok = vkc_matmul(w->t, x, (size_t)t0 * cin, y, (size_t)t0 * cout, tn);
        }
    } else {
        int tile = (int)((size_t)(16u << 20) / kc);
        if (tile > 32768) tile = 32768;
        if (tile < 256) tile = 256;
        if (tile > T) tile = T;
        const int pad = (K - 1) * c->dil / 2;
        for (int t0 = 0; ok && t0 < T; t0 += tile) {
            const int tn = T - t0 < tile ? T - t0 : tile;
            VkcCodec im = {6, cin, K, c->dil, T, t0, 0, 0, tn * kc, 0, pad, 0};
            ok = vkc_codec(x, s->col, s->tab, &im) && vkc_matmul(w->t, s->col, 0, y, (size_t)t0 * cout, tn);
        }
    }
    if (ok && w->off >= 0) { VkcCodec bi = {3, cout, 0, 0, T, 0, 0, 0, T * cout, w->off, 0, 0}; ok = vkc_codec(s->tab, y, s->tab, &bi); }
    return ok;
}

static int avv_snake(const AvvState *s, int off, VkcBuf *x, VkcBuf *y, int T, int C){
    VkcCodec sn = {1, C, 0, 0, T, 0, 0, 0, T * C, off, 0, 0};
    return vkc_codec(x, y, s->tab, &sn);
}

static int avv_reserve(AvvState *s, const AcVae *v, int T){
    size_t nbuf = (size_t)T * v->conv1.cout, nzb = 0;
    int64_t cur = T;
    for (int i = 0; i < v->nblk; i++) {
        const AcvBlock *b = &v->blk[i];
        cur *= b->s;
        if ((size_t)cur * b->cout > nbuf) nbuf = (size_t)cur * b->cout;
        if ((size_t)(cur / b->s) * b->cin > nbuf) nbuf = (size_t)(cur / b->s) * b->cin;
        if ((size_t)(cur / b->s) * 2 * b->s * b->cout > nzb) nzb = (size_t)(cur / b->s) * 2 * b->s * b->cout;
    }
    if ((size_t)cur * v->ach > nbuf) nbuf = (size_t)cur * v->ach;
    if (nbuf > ((size_t)1 << 30) || nzb > ((size_t)1 << 30)) return 0;       /* more than a 4 GB binding: a window that size is the CPU's */
    return vkc_reserve(&s->buf[0], nbuf * 4, VKC_DEV) && vkc_reserve(&s->buf[1], nbuf * 4, VKC_DEV) && vkc_reserve(&s->buf[2], nbuf * 4, VKC_DEV) &&
           vkc_reserve(&s->zb, nzb * 4, VKC_DEV) && vkc_reserve(&s->col, (size_t)(16u << 20) * 4, VKC_DEV) &&
           vkc_reserve(&s->xu, (size_t)T * v->zdim * 4, VKC_UP) && vkc_reserve(&s->out, (size_t)cur * v->ach * 4, VKC_DOWN);
}

/* acv_decode_ws's decode of one window on the card: z [T][zdim] -> out [T*hop][ach]. 0: not taken (the CPU runs the window). */
static int avv_decode(const AcVae *v, const float *z, int T, float *out){
    const char *off = getenv("COLI_VK_VAE");
    if (off && !strcmp(off, "0")) return 0;
    AvvState *s = &avv_state;
    if (s->tried && s->v != v) return 0;                                  /* another decoder than the one on the card */
    if (!s->tried && !acv_open()) return 0;
    if (!avv_setup(s, v)) return 0;
    if (!acv_dev.chain) return 0;
    if (!avv_reserve(s, v, T)) { acv_failed(); return 0; }
    memcpy(vkc_ptr(s->xu), z, (size_t)T * v->zdim * sizeof(float));
    VkcBuf *x, *nx, *tmp;
    int ok = vkc_begin() && vkc_copy(s->buf[0], 0, s->xu, 0, (size_t)T * v->zdim) && avv_conv(s, &s->conv1, &v->conv1, s->buf[0], s->buf[1], T);
    x = s->buf[1]; nx = s->buf[0]; tmp = s->buf[2];
    int64_t cur = T;
    for (int i = 0; ok && i < v->nblk; i++) {
        const AcvBlock *b = &v->blk[i];
        const AvvBlock *d = &s->blk[i];
        const int sd = b->s, nz = 2 * sd * b->cout;
        ok = avv_snake(s, d->sn, x, x, (int)cur, b->cin);
        for (int t0 = 0; ok && t0 < cur; t0 += 32768) {
            const int tn = cur - t0 < 32768 ? (int)(cur - t0) : 32768;
            ok = vkc_matmul(d->up.t, x, (size_t)t0 * b->cin, s->zb, (size_t)t0 * nz, tn);
        }
        VkcCodec ct = {7, (int)cur, sd, 0, 0, 0, 0, 0, (int)(cur * sd * b->cout), d->up.off, b->up.pad, b->cout};
        ok = ok && vkc_codec(s->zb, nx, s->tab, &ct);
        cur *= sd;
        { VkcBuf *t = x; x = nx; nx = t; }
        VkcEw add = {VKC_EW_ADD, (int)(cur * b->cout), 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        for (int u = 0; ok && u < 3; u++)
            ok = avv_snake(s, d->a1[u], x, nx, (int)cur, b->cout) && avv_conv(s, &d->c1[u], &b->r[u].c1, nx, tmp, (int)cur) &&
                 avv_snake(s, d->a2[u], tmp, tmp, (int)cur, b->cout) && avv_conv(s, &d->c2[u], &b->r[u].c2, tmp, nx, (int)cur) &&
                 vkc_ew(x, x, nx, NULL, NULL, &add);
    }
    if (ok) ok = avv_snake(s, s->sn_out, x, x, (int)cur, v->conv2.cin) && avv_conv(s, &s->conv2, &v->conv2, x, nx, (int)cur) &&
                 vkc_copy(s->out, 0, nx, 0, (size_t)cur * v->ach);
    ok = ok && vkc_submit(1);
    if (ok && s->fault && s->windows + 1 == s->fault) { ok = 0; s->fault = 0; }          /* once */
    if (!ok) {
        fprintf(stderr, "[VK] acestep vae: a window failed; the CPU runs it%s\n", vkc_lost() ? " (the device is lost: the CPU runs the rest)" : "");
        acv_failed();
        return 0;
    }
    s->windows++;
    memcpy(out, vkc_ptr(s->out), (size_t)cur * v->ach * sizeof(float));
    return 1;
}

/* Called when a VAE is loaded: from now on its decodes go to the card when there is one. */
static void avv_attach(void){
    acv_decode_hook = avv_decode;
    acv_free_hook = avv_free_hook;
}

#endif
#endif
