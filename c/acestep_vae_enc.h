/* acestep_vae_enc.h -- the ACE-Step 1.5 audio VAE encoder (diffusers AutoencoderOobleck.encode), pure C on top of the decoder's
 * kernels in acestep_vae.h.
 *
 * 48 kHz stereo [N][2] in, latents [N/1920][64] at 25 Hz out. The encoder is the decoder's mirror:
 *   conv1 (2 -> 128 channels, k=7)
 *   five blocks, strides 2, 4, 4, 6, 10 (config.json's downsampling_ratios), each
 *       three residual units (Snake -> Conv1d k=7 dilation 1/3/9 -> Snake -> Conv1d k=1, plus the input)
 *       -> Snake -> Conv1d (k = 2*stride, stride, padding ceil(stride/2): every ratio is even, so the length is
 *       floor(L / stride))
 *     whose widths go 128 -> 128 -> 256 -> 512 -> 1024 -> 2048
 *   Snake -> conv2 (2048 -> 128, k=3)
 * and its 128 output channels are the posterior's parameters: the first 64 the mean, the last 64 the scale, with
 * std = softplus(scale) + 1e-4. The latent upstream conditions on is a SAMPLE of that posterior, mean + std * eps, with eps
 * torch's randn drawn from the global generator, which upstream never seeds: two runs of the handler on one file give two
 * (very slightly) different latents. The engine takes eps from a caller-supplied generator instead, so a seed means one
 * latent; the oracle drives upstream with the same eps (see acestep.c --vae-enc-ref).
 *
 * Long clips are encoded the way upstream's handler does it (acestep/core/generation/handler/vae_encode*.py): up to `chunk`
 * samples in one pass, otherwise overlapping windows with the overlap discarded, the posterior sampled window by window.
 *
 * Interface: acve_load(vae_dir) / acve_encode_raw(...) / acve_encode_tiled(...) / acve_free(e).
 */
#ifndef COLIBRI_ACESTEP_VAE_ENC_H
#define COLIBRI_ACESTEP_VAE_ENC_H
#include "acestep_vae.h"

typedef struct { int cin, cout, s; AcvRes r[3]; AcvSnake sn; AcvConv down; } AcveBlock;

typedef struct {
    int zdim, ach, nblk, hop, rate;
    AcvConv conv1, conv2;
    AcveBlock blk[ACV_MAX_BLOCKS];
    AcvSnake sn;
} AcvEnc;

static void acve_free(AcvEnc *e){
    if (!e) return;
    acv_free_conv(&e->conv1); acv_free_conv(&e->conv2); acv_free_snake(&e->sn);
    for (int i = 0; i < e->nblk; i++) {
        AcveBlock *b = &e->blk[i];
        acv_free_snake(&b->sn); acv_free_conv(&b->down);
        for (int j = 0; j < 3; j++) { acv_free_snake(&b->r[j].s1); acv_free_snake(&b->r[j].s2); acv_free_conv(&b->r[j].c1); acv_free_conv(&b->r[j].c2); }
    }
    free(e);
}

/* vae_dir holds config.json and diffusion_pytorch_model.safetensors (the Hugging Face "vae" directory). */
static AcvEnc *acve_load(const char *vae_dir){
    char err[512] = "", path[4096];
    AcvEnc *e = (AcvEnc *)calloc(1, sizeof(AcvEnc));
    shards S; int have_s = 0;
    jval *cfg = NULL; char *text = NULL;
    if (!e) { fprintf(stderr, "acestep vae encoder: out of memory\n"); return NULL; }

    if (snprintf(path, sizeof path, "%s/config.json", vae_dir) >= (int)sizeof path) { snprintf(err, sizeof err, "path too long"); goto fail; }
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, sizeof err, "cannot open %.400s", path); goto fail; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    text = (char *)malloc(n > 0 ? n + 1 : 1);
    if (!text || n <= 0 || n > (8 << 20) || fread(text, 1, n, f) != (size_t)n) { fclose(f); snprintf(err, sizeof err, "cannot read %.400s", path); goto fail; }
    fclose(f); text[n] = 0;
    cfg = json_parse_checked(text);
    if (!cfg || cfg->t != J_OBJ) { snprintf(err, sizeof err, "%.400s is not a JSON object", path); goto fail; }

    int ratios[ACV_MAX_BLOCKS], mults[ACV_MAX_BLOCKS];
    int nr = acv_json_ints(cfg, "downsampling_ratios", ratios, ACV_MAX_BLOCKS);
    int nm = acv_json_ints(cfg, "channel_multiples", mults, ACV_MAX_BLOCKS);
    int ch = acv_json_int(cfg, "encoder_hidden_size", 0);
    e->zdim = acv_json_int(cfg, "decoder_input_channels", 0);
    e->ach = acv_json_int(cfg, "audio_channels", 0);
    e->rate = acv_json_int(cfg, "sampling_rate", 0);
    if (nr < 1 || nm != nr) { snprintf(err, sizeof err, "config: downsampling_ratios and channel_multiples must be arrays of the same length (1..%d)", ACV_MAX_BLOCKS); goto fail; }
    if (ch < 1 || ch > ACV_MAX_CHANNELS || e->zdim < 1 || e->zdim > ACV_MAX_CHANNELS || e->ach < 1 || e->ach > 8) { snprintf(err, sizeof err, "config: encoder_hidden_size, decoder_input_channels or audio_channels out of range"); goto fail; }
    if (ch != 2 * e->zdim) { snprintf(err, sizeof err, "config: the encoder's last layer (%d channels) must hold the mean and the scale of %d latent channels", ch, e->zdim); goto fail; }
    e->nblk = nr; e->hop = 1;
    for (int i = 0; i < nr; i++) {
        if (ratios[i] < 2 || ratios[i] % 2 || ratios[i] > ACV_MAX_STRIDE || (int64_t)mults[i] * ch > ACV_MAX_CHANNELS) { snprintf(err, sizeof err, "config: ratio %d (it must be even) or width %d out of range", ratios[i], mults[i] * ch); goto fail; }
        e->hop *= ratios[i];
    }
    int m[ACV_MAX_BLOCKS + 1]; m[0] = 1;
    for (int i = 0; i < nm; i++) m[i + 1] = mults[i];

    st_init(&S, vae_dir); have_s = 1;
    if (S.n == 0) { snprintf(err, sizeof err, "no safetensors in %.400s", vae_dir); goto fail; }

    if (acv_load_conv(&S, "encoder.conv1", e->ach, ch * m[0], 7, 1, 1, &e->conv1, err, sizeof err)) goto fail;
    for (int i = 0; i < nr; i++) {
        AcveBlock *b = &e->blk[i];
        char p[160];
        b->cin = ch * m[i]; b->cout = ch * m[i + 1]; b->s = ratios[i];
        static const int dil[3] = {1, 3, 9};
        for (int j = 0; j < 3; j++) {
            AcvRes *r = &b->r[j];
            snprintf(p, sizeof p, "encoder.block.%d.res_unit%d.snake1", i, j + 1);
            if (acv_load_snake(&S, p, b->cin, &r->s1, err, sizeof err)) goto fail;
            snprintf(p, sizeof p, "encoder.block.%d.res_unit%d.conv1", i, j + 1);
            if (acv_load_conv(&S, p, b->cin, b->cin, 7, dil[j], 1, &r->c1, err, sizeof err)) goto fail;
            snprintf(p, sizeof p, "encoder.block.%d.res_unit%d.snake2", i, j + 1);
            if (acv_load_snake(&S, p, b->cin, &r->s2, err, sizeof err)) goto fail;
            snprintf(p, sizeof p, "encoder.block.%d.res_unit%d.conv2", i, j + 1);
            if (acv_load_conv(&S, p, b->cin, b->cin, 1, 1, 1, &r->c2, err, sizeof err)) goto fail;
        }
        snprintf(p, sizeof p, "encoder.block.%d.snake1", i);
        if (acv_load_snake(&S, p, b->cin, &b->sn, err, sizeof err)) goto fail;
        snprintf(p, sizeof p, "encoder.block.%d.conv1", i);
        if (acv_load_conv(&S, p, b->cin, b->cout, 2 * b->s, 1, 1, &b->down, err, sizeof err)) goto fail;
    }
    if (acv_load_snake(&S, "encoder.snake1", ch * m[nr], &e->sn, err, sizeof err)) goto fail;
    if (acv_load_conv(&S, "encoder.conv2", ch * m[nr], ch, 3, 1, 1, &e->conv2, err, sizeof err)) goto fail;
    st_destroy(&S);
    json_free(cfg); free(text);
    return e;
fail:
    fprintf(stderr, "acestep vae encoder: %s\n", err[0] ? err : "load failed");
    if (have_s) st_destroy(&S);
    json_free(cfg); free(text);
    acve_free(e);
    return NULL;
}

/* y[(T + 2 pad - 2s) / s + 1][cout] = Conv1d(k = 2s, stride s, padding pad = ceil(s/2)) of x[T][cin]; returns the new length.
 * Output row t reads input rows s*t - pad + j, j in [0, 2s), zero outside. */
static int64_t acve_down(const AcvConv *c, int s, const float *x, int64_t T, float *y, float *col){
    const int k = c->k, pad = (s + 1) / 2, kc = k * c->cin;
    if (T + 2 * pad - k < 0) return 0;
    const int64_t Tout = (T + 2 * pad - k) / s + 1;
    int64_t rows = (int64_t)(acv_col_floats / (size_t)kc);
    if (rows < 1) rows = 1;
    for (int64_t r0 = 0; r0 < Tout; r0 += rows) {
        int64_t n = Tout - r0 < rows ? Tout - r0 : rows;
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; i++)
            for (int j = 0; j < k; j++) {
                int64_t src = (r0 + i) * s + j - pad;
                float *d = col + ((size_t)i * k + j) * c->cin;
                if (src >= 0 && src < T) memcpy(d, x + (size_t)src * c->cin, (size_t)c->cin * sizeof(float));
                else memset(d, 0, (size_t)c->cin * sizeof(float));
            }
        qi_gemm(y + (size_t)r0 * c->cout, col, (int)n, &c->m, c->b);
    }
    return Tout;
}

/* Scratch for one window of up to maxN samples: the widest activation is the first block's, maxN rows of the first width. */
static AcvWork *acve_work_new(const AcvEnc *e, int64_t maxN){
    AcvWork *w = (AcvWork *)calloc(1, sizeof *w);
    if (!w) return NULL;
    size_t nbuf = (size_t)maxN * e->conv1.cout, ncol = (size_t)e->conv1.k * e->conv1.cin;
    int64_t T = maxN;
    for (int i = 0; i < e->nblk; i++) {
        const AcveBlock *b = &e->blk[i];
        if ((size_t)T * b->cin > nbuf) nbuf = (size_t)T * b->cin;
        T /= b->s;
        if ((size_t)T * b->cout > nbuf) nbuf = (size_t)T * b->cout;
        if ((size_t)7 * b->cin > ncol) ncol = (size_t)7 * b->cin;
        if ((size_t)2 * b->s * b->cin > ncol) ncol = (size_t)2 * b->s * b->cin;
    }
    if ((size_t)3 * e->conv2.cin > ncol) ncol = (size_t)3 * e->conv2.cin;
    size_t colcap = acv_col_floats > ncol ? acv_col_floats : ncol;
    w->nbuf = nbuf; w->ncol = colcap; w->maxT = (int)(maxN > INT32_MAX ? INT32_MAX : maxN);
    for (int i = 0; i < 3; i++) w->buf[i] = acv_alloc(nbuf);
    w->col = acv_alloc(colcap);
    if (!w->buf[0] || !w->buf[1] || !w->buf[2] || !w->col) { acv_work_free(w); return NULL; }
    return w;
}

/* wav [N][ach] -> out [T][2*zdim] (mean | scale, before the softplus), T = floor(N / hop) returned; -1 if N < hop. */
static int64_t acve_run(const AcvEnc *e, AcvWork *w, const float *wav, int64_t N, float *out){
    if (N < e->hop) return -1;
    float *x = w->buf[0], *nx = w->buf[1], *tmp = w->buf[2];
    acv_conv(&e->conv1, wav, (int)N, x, w->col);
    int64_t cur = N;
    for (int i = 0; i < e->nblk; i++) {
        const AcveBlock *b = &e->blk[i];
        for (int j = 0; j < 3; j++) acv_res(&b->r[j], x, (int)cur, nx, tmp, w->col);
        acv_snake(x, x, (int)cur, &b->sn);
        cur = acve_down(&b->down, b->s, x, cur, nx, w->col);
        float *t = x; x = nx; nx = t;
    }
    acv_snake(x, x, (int)cur, &e->sn);
    acv_conv(&e->conv2, x, (int)cur, out, w->col);
    return cur;
}

/* One pass over the whole clip: wav [N][ach] -> out [T][2*zdim]. Returns T, or -1. */
static int64_t acve_encode_raw(const AcvEnc *e, const float *wav, int64_t N, float *out){
    if (N < e->hop || N > INT32_MAX / 2) return -1;
    AcvWork *w = acve_work_new(e, N);
    if (!w) return -1;
    int64_t T = acve_run(e, w, wav, N, out);
    acv_work_free(w);
    return T;
}

/* torch.nn.functional.softplus (beta 1, threshold 20) */
static float acve_softplus(float x){ return x > 20.0f ? x : log1pf(expf(x)); }

/* Python's round(): to nearest, ties to even. */
static int64_t acve_round(double x){
    double r = floor(x + 0.5);
    if (r - x == 0.5 && fmod(r, 2.0) != 0.0) r -= 1.0;
    return (int64_t)r;
}

/* z[t][c] = mean[t][c] + std[t][c] * eps[c * T + t] for one window's raw output [T][2 zdim]; eps is torch's [zdim][T] draw. */
static void acve_sample(const float *raw, int64_t T, int zdim, const float *eps, float *z){
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < T; t++)
        for (int c = 0; c < zdim; c++) {
            float mean = raw[(size_t)t * 2 * zdim + c], scale = raw[(size_t)t * 2 * zdim + zdim + c];
            z[(size_t)t * zdim + c] = eps ? mean + (acve_softplus(scale) + 1e-4f) * eps[(size_t)c * T + t] : mean;
        }
}

/* Upstream's tiled encode (VaeEncodeMixin.tiled_encode and _tiled_encode_offload_cpu): everything in one pass when N <= chunk,
 * otherwise windows of the core `stride = chunk - 2 * overlap` samples plus `overlap` of context on each side, each window's
 * latents trimmed by round(context / 1920) frames and the cores joined. `eps(dst, n, ud)` fills n standard normals and is called
 * once per window with that window's whole [zdim][T] draw, in window order (what torch does with one generator); NULL takes the
 * posterior's mean. out [N/hop + 1][zdim]; returns the frames written, -1 on an error, -2 if `progress` asked to stop. */
static int64_t acve_encode_tiled(const AcvEnc *e, const float *wav, int64_t N, int64_t chunk, int64_t overlap,
                                 void (*eps)(float *dst, size_t n, void *ud), void *ud, float *out,
                                 int (*progress)(int done, int total, void *ud2), void *pud){
    const int z = e->zdim;
    if (N < e->hop || chunk < 1 || overlap < 0 || N > INT32_MAX / 2) return -1;
    if (N <= chunk) {
        if (progress && progress(0, 1, pud)) return -2;
        AcvWork *w = acve_work_new(e, N);
        float *raw = acv_alloc((size_t)(N / e->hop) * 2 * z), *dr = eps ? acv_alloc((size_t)(N / e->hop) * z) : NULL;
        if (!w || !raw || (eps && !dr)) { acv_work_free(w); free(raw); free(dr); return -1; }
        int64_t T = acve_run(e, w, wav, N, raw);
        if (T < 0) { acv_work_free(w); free(raw); free(dr); return -1; }
        if (eps) eps(dr, (size_t)T * z, ud);
        acve_sample(raw, T, z, dr, out);
        acv_work_free(w); free(raw); free(dr);
        if (progress) progress(1, 1, pud);
        return T;
    }
    const int64_t stride = chunk - 2 * overlap;
    if (stride <= 0) { fprintf(stderr, "acestep vae encoder: chunk %lld must exceed twice the overlap %lld\n", (long long)chunk, (long long)overlap); return -1; }
    const int steps = (int)((N + stride - 1) / stride);
    const int64_t maxwin = stride + 2 * overlap < N ? stride + 2 * overlap : N;
    AcvWork *w = acve_work_new(e, maxwin);
    float *raw = acv_alloc((size_t)(maxwin / e->hop + 1) * 2 * z), *zz = acv_alloc((size_t)(maxwin / e->hop + 1) * z);
    float *dr = eps ? acv_alloc((size_t)(maxwin / e->hop + 1) * z) : NULL;
    if (!w || !raw || !zz || (eps && !dr)) { acv_work_free(w); free(raw); free(zz); free(dr); return -1; }
    double down = 0;                                  /* samples per latent frame, from the first window as upstream measures it */
    int64_t pos = 0;
    for (int i = 0; i < steps; i++) {
        if (progress && progress(i, steps, pud)) { acv_work_free(w); free(raw); free(zz); free(dr); return -2; }
        int64_t cs = (int64_t)i * stride, ce = cs + stride < N ? cs + stride : N;
        int64_t ws = cs - overlap > 0 ? cs - overlap : 0, we = ce + overlap < N ? ce + overlap : N;
        int64_t T = acve_run(e, w, wav + (size_t)ws * e->ach, we - ws, raw);
        if (T < 0) { acv_work_free(w); free(raw); free(zz); free(dr); return -1; }
        if (down == 0) down = (double)(we - ws) / (double)T;
        if (eps) eps(dr, (size_t)T * z, ud);
        acve_sample(raw, T, z, dr, zz);
        int64_t trim0 = acve_round((double)(cs - ws) / down), trim1 = acve_round((double)(we - ce) / down);
        int64_t end = trim1 > 0 ? T - trim1 : T;
        if (end > T) end = T;
        if (trim0 < end) {
            memcpy(out + (size_t)pos * z, zz + (size_t)trim0 * z, (size_t)(end - trim0) * z * sizeof(float));
            pos += end - trim0;
        }
    }
    if (progress) progress(steps, steps, pud);
    acv_work_free(w); free(raw); free(zz); free(dr);
    return pos;
}

#endif
