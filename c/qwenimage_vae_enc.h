/* qwenimage_vae_enc.h -- the Qwen-Image-2.1 VAE encoder (diffusers AutoencoderKLQwenImage21.encode),
 * one image at a time, pure C on top of the decoder's blocks (qwenimage_vae.h).
 *
 * The encoder is the decoder's mirror, and for one frame it folds the same way:
 *   - every CausalConv3d is a zero-padded Conv2d, so the weights are plain [Cout][Cin][k][k];
 *   - a downsample3d Resample runs its spatial half only: the time_conv of the first
 *     chunk is skipped (the feature cache stores the frame and returns), so it is
 *     never loaded;
 *   - AvgDown3D with factor_t = 2 pads one zero frame in front of the single frame,
 *     so half of the channel groups it averages are exactly zero (qive_avgdown_add).
 * Path: conv_in (3x3), five residual down blocks (two resnets, then a 3x3 stride-2 conv
 * on the input zero-padded by one column on the right and one row at the bottom, added to
 * the average-pooled block input; the last block does not downsample), the mid block
 * (resnet, single-head attention over all pixels, resnet), RMS norm + SiLU, conv_out (3x3,
 * 2*z_dim channels), quant_conv (1x1). The latent is the first z_dim channels (the mode of
 * the posterior, retrieve_latents(..., "argmax")), then the pipeline's
 * (latent - latents_mean) / latents_std (QwenImage21Pipeline._encode_vae_image).
 *
 * The input is what the pipeline hands the encoder: the picture as RGBA, bytes / 255
 * * 2 - 1 (VaeImageProcessor.preprocess), height and width multiples of 16.
 *
 * Interface: qive_load(vae_dir) / qive_encode(e, rgba, H, W, z) / qive_free(e). z is
 * [H/16][W/16][z_dim], token y*w + x: exactly what qiv_decode takes, so decode(encode(x))
 * round-trips. The encoder weights are loaded apart from the decoder's, so a text-to-image
 * run does not pay for them. */
#ifndef COLIBRI_QWENIMAGE_VAE_ENC_H
#define COLIBRI_QWENIMAGE_VAE_ENC_H
#include "qwenimage_vae.h"

typedef struct {
    int cin, cout;
    int down, ft;           /* down: 3x3 stride-2 conv + 2x2 pooling shortcut; ft: AvgDown3D's factor_t (1 or 2) */
    int nres;
    QivRes res[QIV_MAX_RES];
    QivConv dsc;            /* the stride-2 conv, only when down */
} QiveBlock;

typedef struct QiVaeEnc {
    int z_dim, in_ch, scale;    /* scale: pixels per latent along each side (16) */
    float *mean, *std;          /* [z_dim] from config.json */
    QivConv conv_in;
    int nblk;
    QiveBlock blk[QIV_MAX_BLOCKS];
    QivRes mid[2];
    float *attn_g;
    QivConv qkv, proj;
    float *g_out;
    QivConv conv_out;           /* 3x3, 2*z_dim outputs: mean and log-variance */
    QivConv qc;                 /* quant_conv, 1x1, 2*z_dim -> 2*z_dim */
    size_t wbytes;
} QiVaeEnc;

static void qive_free(QiVaeEnc *e){
    if (!e) return;
    free(e->mean); free(e->std);
    qiv_free_conv(&e->conv_in);
    for (int i = 0; i < e->nblk; i++) {
        for (int j = 0; j < e->blk[i].nres; j++) qiv_free_res(&e->blk[i].res[j]);
        qiv_free_conv(&e->blk[i].dsc);
    }
    qiv_free_res(&e->mid[0]); qiv_free_res(&e->mid[1]);
    free(e->attn_g); qiv_free_conv(&e->qkv); qiv_free_conv(&e->proj);
    free(e->g_out); qiv_free_conv(&e->conv_out); qiv_free_conv(&e->qc);
    free(e);
}

static QiVaeEnc *qive_load(const char *vae_dir){
    char err[512] = "", path[4096];
    QiVaeEnc *e = (QiVaeEnc *)calloc(1, sizeof(QiVaeEnc));
    /* the decoder's loaders count the bytes they keep in a QiVae: a scratch one here */
    QiVae *acct = (QiVae *)calloc(1, sizeof(QiVae));
    shards S; int have_s = 0;
    jval *cfg = NULL;
    char *text = NULL;
    if (!e || !acct) { fprintf(stderr, "qwenimage vae encoder: out of memory\n"); free(e); free(acct); return NULL; }

    if (snprintf(path, sizeof path, "%s/config.json", vae_dir) >= (int)sizeof path) {
        snprintf(err, sizeof err, "path too long"); goto fail; }
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, sizeof err, "cannot open %.400s", path); goto fail; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    text = (char *)malloc(n > 0 ? n + 1 : 1);
    if (!text || n <= 0 || fread(text, 1, n, f) != (size_t)n) {
        fclose(f); snprintf(err, sizeof err, "cannot read %.400s", path); goto fail; }
    fclose(f); text[n] = 0;
    cfg = json_parse_checked(text);
    if (!cfg || cfg->t != J_OBJ) { snprintf(err, sizeof err, "%.400s is not a JSON object", path); goto fail; }

    e->z_dim = qiv_json_int(cfg, "z_dim", 0);
    e->in_ch = qiv_json_int(cfg, "in_channels", 4);
    int base = qiv_json_int(cfg, "base_dim", 0);       /* the encoder's width, not decoder_base_dim */
    int nrb = qiv_json_int(cfg, "num_res_blocks", 2);
    jval *jm = json_get(cfg, "dim_mult"), *jt = json_get(cfg, "temperal_downsample");
    jval *jr = json_get(cfg, "is_residual"), *jp = json_get(cfg, "patch_size");
    jval *jmean = json_get(cfg, "latents_mean"), *jstd = json_get(cfg, "latents_std");
    if (e->z_dim <= 0 || base <= 0 || nrb < 1 || nrb > QIV_MAX_RES || !jm || jm->t != J_ARR
        || jm->len < 1 || jm->len > QIV_MAX_BLOCKS || !jt || jt->t != J_ARR || jt->len != jm->len - 1) {
        snprintf(err, sizeof err, "%.400s: missing or unexpected z_dim/base_dim/num_res_blocks/dim_mult/temperal_downsample", path);
        goto fail;
    }
    if (e->in_ch != 3 && e->in_ch != 4) { snprintf(err, sizeof err, "in_channels=%d, expected 3 or 4", e->in_ch); goto fail; }
    if (!jr || jr->t != J_BOOL || !jr->boolean) { snprintf(err, sizeof err, "only is_residual=true is supported"); goto fail; }
    if (jp && jp->t != J_NULL) { snprintf(err, sizeof err, "patch_size must be null"); goto fail; }
    if (!jmean || jmean->t != J_ARR || jmean->len != e->z_dim || !jstd || jstd->t != J_ARR || jstd->len != e->z_dim) {
        snprintf(err, sizeof err, "latents_mean/latents_std must have z_dim=%d entries", e->z_dim); goto fail; }
    if (e->z_dim > 4096) { snprintf(err, sizeof err, "z_dim %d is out of range", e->z_dim); goto fail; }
    e->mean = qiv_alloc(e->z_dim); e->std = qiv_alloc(e->z_dim);
    if (!e->mean || !e->std) { snprintf(err, sizeof err, "out of memory"); goto fail; }
    for (int i = 0; i < e->z_dim; i++) {
        if (jmean->kids[i]->t != J_NUM || jstd->kids[i]->t != J_NUM) { snprintf(err, sizeof err, "latents_mean/latents_std must be numbers"); goto fail; }
        e->mean[i] = (float)jmean->kids[i]->num; e->std[i] = (float)jstd->kids[i]->num;
    }

    int nm = jm->len, dims[QIV_MAX_BLOCKS + 1], tds[QIV_MAX_BLOCKS];
    for (int i = 0; i < nm; i++)
        if (jm->kids[i]->t != J_NUM || jm->kids[i]->num < 1 || jm->kids[i]->num > 64 || base > 4096) {
            snprintf(err, sizeof err, "dim_mult entries must be whole numbers from 1 to 64 (base %d)", base); goto fail; }
    for (int i = 0; i < nm - 1; i++) {
        if (jt->kids[i]->t != J_BOOL) { snprintf(err, sizeof err, "temperal_downsample must hold booleans"); goto fail; }
        tds[i] = jt->kids[i]->boolean;
    }
    dims[0] = base;                                     /* dims = [base * u for u in [1] + dim_mult] */
    for (int i = 0; i < nm; i++) dims[i + 1] = base * (int)jm->kids[i]->num;

    {
        DIR *d = opendir(vae_dir);
        if (!d) { snprintf(err, sizeof err, "cannot open directory %.400s", vae_dir); goto fail; }
        closedir(d);
    }
    st_init(&S, vae_dir); have_s = 1;
    if (S.n == 0) { snprintf(err, sizeof err, "no safetensors in %.400s", vae_dir); goto fail; }

    if (qiv_load_conv(&S, acct, "encoder.conv_in", e->in_ch, dims[0], 3, &e->conv_in, err, sizeof err)) goto fail;
    e->scale = 1;
    e->nblk = nm;
    for (int i = 0; i < nm; i++) {
        QiveBlock *b = &e->blk[i];
        b->cin = dims[i]; b->cout = dims[i + 1];
        b->down = i != nm - 1;
        b->ft = b->down && tds[i] ? 2 : 1;
        b->nres = nrb;
        /* AvgDown3D needs whole channel groups: in * factor / out */
        const int factor = b->ft * (b->down ? 4 : 1);
        if ((b->cin * factor) % b->cout) {
            snprintf(err, sizeof err, "down block %d: AvgDown3D %d->%d channels is not a whole group", i, b->cin, b->cout);
            goto fail;
        }
        for (int j = 0; j < b->nres; j++) {
            char p[128]; snprintf(p, sizeof p, "encoder.down_blocks.%d.resnets.%d", i, j);
            if (qiv_load_res(&S, acct, p, j ? b->cout : b->cin, b->cout, &b->res[j], err, sizeof err)) goto fail;
        }
        if (b->down) {
            char p[128]; snprintf(p, sizeof p, "encoder.down_blocks.%d.downsampler.resample.1", i);
            if (qiv_load_conv(&S, acct, p, b->cout, b->cout, 3, &b->dsc, err, sizeof err)) goto fail;
            e->scale *= 2;
        }
    }
    for (int i = 0; i < 2; i++) {
        char p[128]; snprintf(p, sizeof p, "encoder.mid_block.resnets.%d", i);
        if (qiv_load_res(&S, acct, p, dims[nm], dims[nm], &e->mid[i], err, sizeof err)) goto fail;
    }
    if (!(e->attn_g = qiv_read(&S, "encoder.mid_block.attentions.0.norm.gamma", dims[nm], err, sizeof err))) goto fail;
    if (qiv_load_conv(&S, acct, "encoder.mid_block.attentions.0.to_qkv", dims[nm], 3 * dims[nm], 1, &e->qkv, err, sizeof err)) goto fail;
    if (qiv_load_conv(&S, acct, "encoder.mid_block.attentions.0.proj", dims[nm], dims[nm], 1, &e->proj, err, sizeof err)) goto fail;
    if (!(e->g_out = qiv_read(&S, "encoder.norm_out.gamma", dims[nm], err, sizeof err))) goto fail;
    if (qiv_load_conv(&S, acct, "encoder.conv_out", dims[nm], 2 * e->z_dim, 3, &e->conv_out, err, sizeof err)) goto fail;
    if (qiv_load_conv(&S, acct, "quant_conv", 2 * e->z_dim, 2 * e->z_dim, 1, &e->qc, err, sizeof err)) goto fail;
    e->wbytes = acct->wbytes + (size_t)(dims[nm] + dims[nm]) * sizeof(float);
    st_destroy(&S);
    json_free(cfg); free(text); free(acct);
    return e;
fail:
    fprintf(stderr, "qwenimage vae encoder: %s\n", err[0] ? err : "load failed");
    if (have_s) st_destroy(&S);
    json_free(cfg); free(text); free(acct);
    qive_free(e);
    return NULL;
}

/* ---------------------------------------------------------------- compute -- */

/* y[Ho][Wo][C] = conv3x3, stride 2, of x[H][W][C] zero-padded by one column on the
 * right and one row at the bottom (nn.ZeroPad2d((0, 1, 0, 1)) then Conv2d(3, stride 2)):
 * output pixel (oy, ox) reads rows 2*oy..2*oy+2 and columns 2*ox..2*ox+2, taps past the
 * edge are zero. H and W are even, Ho = H/2. One GEMM per band of output rows over an
 * im2col whose 9*C floats per pixel are ordered [kx][ky][c], the order the weight was
 * permuted to at load. */
static int qive_down(const QivConv *cv, const float *x, int H, int W, float *y){
    const int C = cv->cin, Ho = H / 2, Wo = W / 2, K = 9 * C;
    int band = (int)(qiv_col_floats / ((size_t)Wo * K));
    if (band < 1) band = 1;
    if (band > Ho) band = Ho;
    float *Q = qiv_alloc((size_t)band * Wo * K);
    if (!Q) return -1;
    for (int y0 = 0; y0 < Ho; y0 += band) {
        const int nb = Ho - y0 < band ? Ho - y0 : band;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < nb; r++)
            for (int u = 0; u < Wo; u++) {
                float *d = Q + ((size_t)r * Wo + u) * K;
                for (int kx = 0; kx < 3; kx++)
                    for (int ky = 0; ky < 3; ky++) {
                        const int iy = 2 * (y0 + r) + ky, ix = 2 * u + kx;
                        float *t = d + (kx * 3 + ky) * C;
                        if (iy >= H || ix >= W) memset(t, 0, sizeof(float) * C);
                        else memcpy(t, x + ((size_t)iy * W + ix) * C, sizeof(float) * C);
                    }
            }
        qi_gemm(y + (size_t)y0 * Wo * C, Q, nb * Wo, &cv->m, cv->b);
    }
    free(Q);
    return 0;
}

/* y[Ho][Wo][cout] += AvgDown3D(x[H][W][cin]) for one frame, Ho = H/fs, Wo = W/fs.
 * AvgDown3D folds [C][ft][fs][fs] into C*factor channels (c*factor + (t*fs + hy)*fs + hx),
 * groups them by group = cin*factor/cout and takes the mean of each group. With
 * ft = 2 the one frame sits behind a zero frame (F.pad on the time axis, at the front),
 * so the taps with t = 0 are zero: they stay in the divisor and out of the sum. */
static int qive_avgdown_add(float *y, const float *x, int H, int W, int cin, int cout, int ft, int fs){
    const int factor = ft * fs * fs, group = cin * factor / cout, Ho = H / fs, Wo = W / fs;
    typedef struct { int c, dy, dx; } Tap;
    Tap *tap = (Tap *)malloc(sizeof(Tap) * (size_t)cout * group);
    int *nt = (int *)calloc((size_t)cout, sizeof(int));
    if (!tap || !nt) { free(tap); free(nt); return -1; }
    for (int o = 0; o < cout; o++)
        for (int g = 0; g < group; g++) {
            const int ch = o * group + g, c = ch / factor, s = ch % factor;
            const int t = s / (fs * fs), r = s % (fs * fs);
            if (t != ft - 1) continue;              /* the zero frame */
            tap[(size_t)o * group + nt[o]++] = (Tap){ c, r / fs, r % fs };
        }
    const float inv = (float)group;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int py = 0; py < Ho; py++)
        for (int px = 0; px < Wo; px++) {
            float *d = y + ((size_t)py * Wo + px) * cout;
            for (int o = 0; o < cout; o++) {
                const Tap *tp = tap + (size_t)o * group;
                float acc = 0.f;
                for (int k = 0; k < nt[o]; k++)
                    acc += x[((size_t)(py * fs + tp[k].dy) * W + (px * fs + tp[k].dx)) * cin + tp[k].c];
                d[o] += acc / inv;
            }
        }
    free(tap); free(nt);
    return 0;
}

/* rgba: [H][W][4] uint8, top row first. z: [H/scale][W/scale][z_dim], normalized.
 * H and W must be multiples of e->scale. Returns 0, or -1 on bad arguments / out of memory. */
static int qive_encode(QiVaeEnc *e, const uint8_t *rgba, int H, int W, float *z){
    if (!e || !rgba || !z || H <= 0 || W <= 0 || H % e->scale || W % e->scale) return -1;
    const int prof = getenv("QIV_PROFILE") != NULL;
    double t0 = qiv_now(), tl = t0;
    const int Z = e->z_dim;
    int h = H, w = W;
#define QIVE_STAGE(name) do { double tn = qiv_now(); \
        if (prof) fprintf(stderr, "[vae-enc] %-10s %4dx%-4d %7.3f s\n", name, h, w, tn - tl); tl = tn; } while (0)
    float *a = qiv_alloc((size_t)H * W * e->in_ch), *x = NULL;
    if (!a) goto fail;
    /* VaeImageProcessor.preprocess: float32 bytes / 255, then 2 * v - 1 */
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < (size_t)H * W; p++)
        for (int c = 0; c < e->in_ch; c++) a[p * e->in_ch + c] = 2.f * ((float)rgba[p * 4 + c] / 255.f) - 1.f;
    x = qiv_alloc((size_t)H * W * e->conv_in.cout);
    if (!x || qiv_conv3(&e->conv_in, a, H, W, NULL, 0, x, NULL)) goto fail;
    free(a); a = NULL;
    QIVE_STAGE("conv_in");

    float *nx;
    for (int i = 0; i < e->nblk; i++) {
        const QiveBlock *bk = &e->blk[i];
        /* xin: the block input, kept for the pooling shortcut, so the first resnet must not write over it */
        float *xin = x, *cur = x;
        const int Hin = h, Win = w;
        x = NULL;
        for (int j = 0; j < bk->nres; j++) {
            if (!(nx = qiv_resnet(&bk->res[j], cur, h, w, j == 0))) {
                if (cur != xin) free(cur);
                free(xin); goto fail;
            }
            cur = nx;
        }
        if (bk->down) {
            float *y = qiv_alloc((size_t)(h / 2) * (w / 2) * bk->cout);
            if (!y || qive_down(&bk->dsc, cur, h, w, y)) { free(y); free(cur); free(xin); goto fail; }
            free(cur); cur = y;
            h /= 2; w /= 2;
        }
        if (qive_avgdown_add(cur, xin, Hin, Win, bk->cin, bk->cout, bk->ft, bk->down ? 2 : 1)) {
            free(cur); free(xin); goto fail;
        }
        free(xin);
        x = cur;
        char nm[16]; snprintf(nm, sizeof nm, "block%d", i);
        QIVE_STAGE(nm);
    }

    if (!(nx = qiv_resnet(&e->mid[0], x, h, w, 0))) goto fail;
    x = nx;
    if (qiv_attention_w(&e->qkv, &e->proj, e->attn_g, x, h * w)) goto fail;
    if (!(nx = qiv_resnet(&e->mid[1], x, h, w, 0))) goto fail;
    x = nx;
    QIVE_STAGE("mid");

    {
        const size_t HW = (size_t)h * w;
        float *o = qiv_alloc(HW * 2 * Z), *q = qiv_alloc(HW * 2 * Z);
        QivEpi ep; memset(&ep, 0, sizeof ep); ep.rel_x = 1;
        if (!o || !q || qiv_conv3(&e->conv_out, x, h, w, e->g_out, 0, o, &ep)) { free(o); free(q); goto fail; }
        free(x); x = NULL;
        qi_gemm(q, o, (int)HW, &e->qc.m, e->qc.b);
        /* the posterior's mode is its first z_dim channels; then (latent - mean) / std */
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < HW; p++)
            for (int c = 0; c < Z; c++) z[p * Z + c] = (q[p * 2 * Z + c] - e->mean[c]) / e->std[c];
        free(o); free(q);
    }
    QIVE_STAGE("head");
    if (prof) fprintf(stderr, "[vae-enc] total %.3f s\n", qiv_now() - t0);
#undef QIVE_STAGE
    return 0;
fail:
    free(a); free(x);
    fprintf(stderr, "qwenimage vae encoder: encode failed (out of memory?)\n");
    return -1;
}

#endif
