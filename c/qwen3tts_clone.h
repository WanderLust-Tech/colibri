/* qwen3tts_clone.h -- voice cloning for qwen3tts.c: the Base checkpoints (Qwen3-TTS-12Hz-0.6B-Base, -1.7B-Base), docs/qwen3-tts.md.
 *
 * A reference clip (24 kHz mono) becomes two things:
 *   - a speaker embedding: the clip's log-mel spectrogram (n_fft 1024, hop 256, 128 mels up to 12 kHz) through an ECAPA-TDNN (the model's
 *     `speaker_encoder`); the vector takes the place of the speaker's codec row in the prompt.            (x-vector only mode)
 *   - the clip's codes: the codec's *encoder*, which is Kyutai's Mimi (a SEANet convolutional stack with strides 4, 5, 6, 8, an 8-layer
 *     transformer, a stride-2 downsampling to 12.5 Hz and a split residual quantizer; 16 of its codebooks are used).  (in-context mode)
 * In-context mode puts the reference text and the text to speak in front of the reference's codes, so the model carries on from the
 * reference's voice, and cuts the reference's share off the decoded speech afterwards. The C code follows the qwen-tts package
 * (Qwen3TTSModel.generate_voice_clone) and the transformers Mimi it builds on.
 *
 * Included by qwen3tts.c after the codec's convolution code; every function here is static.
 */

/* ------------------------------------------------------------------------------------------------ small helpers */

static inline float qc_elu(float x){ return x > 0.f ? x : expm1f(x); }

static void qc_layernorm(float *dst, const float *src, int rows, int n, const float *w, const float *b, float eps){
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *x = src + (size_t)r * n; float *y = dst + (size_t)r * n;
        double mu = 0, var = 0;
        for (int i = 0; i < n; i++) mu += x[i];
        mu /= n;
        for (int i = 0; i < n; i++) { const double d = x[i] - mu; var += d * d; }
        var /= n;
        const float inv = 1.0f / sqrtf((float)var + eps);
        for (int i = 0; i < n; i++) y[i] = (float)((x[i] - mu) * inv) * w[i] + (b ? b[i] : 0.f);
    }
}

/* Conv1d over x [T][cin] -> y [Ty][cout] with stride, the codec encoder's padding: `left` zeros (or the first row when `replicate`) before, and
 * enough after (the same rule as the reference's _get_extra_padding_for_conv1d) to make Ty = ceil(T / stride); taps (k * dil) apart.
 * Returns Ty. y must hold ceil(T / stride) rows. Tiles of QT_TILE output rows go through an im2col GEMM. */
static int qc_conv(float *y, const float *x, int T, const QtConv *c, int stride, int replicate){
    const int K = c->K, cin = c->cin, kc = K * cin;
    const int ek = (K - 1) * c->dil + 1, pt = ek - stride;
    const int Ty = (T + stride - 1) / stride;
    float *col = (float *)malloc((size_t)QT_TILE * kc * sizeof(float));
    if (!col) qt_die("out of memory");
    for (int t0 = 0; t0 < Ty; t0 += QT_TILE) {
        const int tn = Ty - t0 < QT_TILE ? Ty - t0 : QT_TILE;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < tn; t++) {
            float *row = col + (size_t)t * kc;
            for (int k = 0; k < K; k++) {
                int s = (t0 + t) * stride - pt + k * c->dil;
                if (s < 0 || s >= T) {
                    if (!replicate) { memset(row + (size_t)k * cin, 0, (size_t)cin * sizeof(float)); continue; }
                    s = s < 0 ? 0 : T - 1;
                }
                memcpy(row + (size_t)k * cin, x + (size_t)s * cin, (size_t)cin * sizeof(float));
            }
        }
        qi_gemm(y + (size_t)t0 * c->cout, col, tn, &c->w.m, c->b);
    }
    free(col);
    return Ty;
}

/* A conv that keeps the length with the input mirrored at the ends (padding "same", mode reflect): the speaker encoder's TDNN blocks. */
static void qc_conv_same(float *y, const float *x, int T, const QtConv *c, int relu){
    const int K = c->K, cin = c->cin, kc = K * cin;
    const int left = c->dil * (K - 1) / 2;
    float *col = (float *)malloc((size_t)QT_TILE * kc * sizeof(float));
    if (!col) qt_die("out of memory");
    for (int t0 = 0; t0 < T; t0 += QT_TILE) {
        const int tn = T - t0 < QT_TILE ? T - t0 : QT_TILE;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < tn; t++) {
            float *row = col + (size_t)t * kc;
            for (int k = 0; k < K; k++) {
                int s = t0 + t - left + k * c->dil;
                if (T > 1) { while (s < 0 || s >= T) s = s < 0 ? -s : 2 * (T - 1) - s; } else s = 0;
                memcpy(row + (size_t)k * cin, x + (size_t)s * cin, (size_t)cin * sizeof(float));
            }
        }
        qi_gemm(y + (size_t)t0 * c->cout, col, tn, &c->w.m, c->b);
    }
    free(col);
    if (relu) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)T * c->cout; i++) if (y[i] < 0.f) y[i] = 0.f;
    }
}

/* ------------------------------------------------------------------------------------------------ the log-mel spectrogram */

static void qc_fft(float *re, float *im, int n){                 /* in place, n a power of two */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * 3.14159265358979323846 / len;
        for (int i = 0; i < n; i += len)
            for (int k = 0; k < len / 2; k++) {
                const float wr = (float)cos(ang * k), wi = (float)sin(ang * k);
                const float xr = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi, xi = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
                re[i + k + len / 2] = re[i + k] - xr; im[i + k + len / 2] = im[i + k] - xi;
                re[i + k] += xr; im[i + k] += xi;
            }
    }
}

static double qc_hz_to_mel(double f){                            /* Slaney's scale: linear below 1 kHz, logarithmic above (librosa's default) */
    const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = log(6.4) / 27.0;
    return f >= min_log_hz ? min_log_mel + log(f / min_log_hz) / logstep : f / f_sp;
}
static double qc_mel_to_hz(double m){
    const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = log(6.4) / 27.0;
    return m >= min_log_mel ? min_log_hz * exp(logstep * (m - min_log_mel)) : f_sp * m;
}

/* librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) with Slaney normalisation: [n_mels][n_fft / 2 + 1] */
static float *qc_mel_filters(int sr, int nfft, int nmels, double fmin, double fmax){
    const int nb = nfft / 2 + 1;
    float *fb = (float *)calloc((size_t)nmels * nb, sizeof(float));
    double *mf = (double *)malloc((size_t)(nmels + 2) * sizeof(double));
    const double lo = qc_hz_to_mel(fmin), hi = qc_hz_to_mel(fmax);
    for (int i = 0; i < nmels + 2; i++) mf[i] = qc_mel_to_hz(lo + (hi - lo) * i / (nmels + 1));
    for (int m = 0; m < nmels; m++) {
        const double enorm = 2.0 / (mf[m + 2] - mf[m]);
        for (int b = 0; b < nb; b++) {
            const double f = (double)sr / 2.0 * b / (nb - 1);
            const double lower = (f - mf[m]) / (mf[m + 1] - mf[m]), upper = (mf[m + 2] - f) / (mf[m + 2] - mf[m + 1]);
            const double w = lower < upper ? lower : upper;
            fb[(size_t)m * nb + b] = (float)((w > 0 ? w : 0) * enorm);
        }
    }
    free(mf);
    return fb;
}

/* The speaker encoder's input: wav [n] at 24 kHz -> [frames][128] (time-major), *frames set. malloc'd, NULL when the clip is too short. */
static float *qc_mel(const float *wav, int n, int *frames){
    const int NFFT = 1024, HOP = 256, NM = 128, pad = (NFFT - HOP) / 2, nb = NFFT / 2 + 1;
    if (n < 2) return NULL;
    const int total = n + 2 * pad;
    const int T = total >= NFFT ? 1 + (total - NFFT) / HOP : 0;
    if (T < 1) return NULL;
    float *xp = (float *)malloc((size_t)total * sizeof(float));
    for (int i = 0; i < total; i++) {                              /* reflect padding by `pad` on both sides */
        int s = i - pad;
        while (s < 0 || s >= n) s = s < 0 ? -s : 2 * (n - 1) - s;
        xp[i] = wav[s];
    }
    float *win = (float *)malloc((size_t)NFFT * sizeof(float));
    for (int i = 0; i < NFFT; i++) win[i] = (float)(0.5 - 0.5 * cos(2.0 * 3.14159265358979323846 * i / NFFT));   /* torch.hann_window: periodic */
    float *fb = qc_mel_filters(24000, NFFT, NM, 0.0, 12000.0);
    float *out = (float *)malloc((size_t)T * NM * sizeof(float));
    #pragma omp parallel
    {
        float *re = (float *)malloc((size_t)NFFT * sizeof(float)), *im = (float *)malloc((size_t)NFFT * sizeof(float)), *mag = (float *)malloc((size_t)nb * sizeof(float));
        #pragma omp for schedule(static)
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < NFFT; i++) { re[i] = xp[(size_t)t * HOP + i] * win[i]; im[i] = 0.f; }
            qc_fft(re, im, NFFT);
            for (int b = 0; b < nb; b++) mag[b] = sqrtf(re[b] * re[b] + im[b] * im[b] + 1e-9f);
            for (int m = 0; m < NM; m++) {
                double s = 0;
                const float *w = fb + (size_t)m * nb;
                for (int b = 0; b < nb; b++) s += (double)w[b] * mag[b];
                const float v = (float)s;
                out[(size_t)t * NM + m] = logf(v < 1e-5f ? 1e-5f : v);
            }
        }
        free(re); free(im); free(mag);
    }
    free(xp); free(win); free(fb);
    *frames = T;
    return out;
}

/* ------------------------------------------------------------------------------------------------ the speaker encoder (ECAPA-TDNN) */

typedef struct { QtConv tdnn1, res[7], tdnn2, se1, se2; } QcSeRes;
typedef struct {
    int ok, dim;                                                   /* loaded; the embedding's width */
    QtConv b0, mfa, asp_tdnn, asp_conv, fc;
    QcSeRes blk[3];
} QcSpk;

static int qc_spk_load(QcSpk *s, shards *S, jval *cfg, char *err, size_t errn){
    jval *sc = json_get(cfg, "speaker_encoder_config");
    if (!sc) return 0;                                              /* a checkpoint with no speaker encoder: not a Base model */
    s->dim = acd_json_int(sc, "enc_dim", 0);
    size_t bytes = 0;
    char w[300], b[300];
#define QC_CONV(dst, pfx, cout, cin, K, dil) do { \
        snprintf(w, sizeof w, "%s.weight", pfx); snprintf(b, sizeof b, "%s.bias", pfx); \
        if (qt_conv(S, w, b, cout, cin, K, dil, &(dst), &bytes, err, errn)) return -1; } while (0)
    QC_CONV(s->b0, "speaker_encoder.blocks.0.conv", 512, 128, 5, 1);
    const int dil[3] = {2, 3, 4};
    for (int i = 0; i < 3; i++) {
        char p[200];
        snprintf(p, sizeof p, "speaker_encoder.blocks.%d.tdnn1.conv", i + 1);   QC_CONV(s->blk[i].tdnn1, p, 512, 512, 1, 1);
        for (int j = 0; j < 7; j++) { snprintf(p, sizeof p, "speaker_encoder.blocks.%d.res2net_block.blocks.%d.conv", i + 1, j); QC_CONV(s->blk[i].res[j], p, 64, 64, 3, dil[i]); }
        snprintf(p, sizeof p, "speaker_encoder.blocks.%d.tdnn2.conv", i + 1);   QC_CONV(s->blk[i].tdnn2, p, 512, 512, 1, 1);
        snprintf(p, sizeof p, "speaker_encoder.blocks.%d.se_block.conv1", i + 1); QC_CONV(s->blk[i].se1, p, 128, 512, 1, 1);
        snprintf(p, sizeof p, "speaker_encoder.blocks.%d.se_block.conv2", i + 1); QC_CONV(s->blk[i].se2, p, 512, 128, 1, 1);
    }
    QC_CONV(s->mfa, "speaker_encoder.mfa.conv", 1536, 1536, 1, 1);
    QC_CONV(s->asp_tdnn, "speaker_encoder.asp.tdnn.conv", 128, 4608, 1, 1);
    QC_CONV(s->asp_conv, "speaker_encoder.asp.conv", 1536, 128, 1, 1);
    QC_CONV(s->fc, "speaker_encoder.fc", s->dim, 3072, 1, 1);
#undef QC_CONV
    s->ok = 1;
    return 0;
}

static void qc_gather(float *dst, const float *src, int T, int C, int c0, int n){            /* channels c0 .. c0 + n of a time-major tensor */
    for (int t = 0; t < T; t++) memcpy(dst + (size_t)t * n, src + (size_t)t * C + c0, (size_t)n * sizeof(float));
}
static void qc_scatter(float *dst, const float *src, int T, int C, int c0, int n){
    for (int t = 0; t < T; t++) memcpy(dst + (size_t)t * C + c0, src + (size_t)t * n, (size_t)n * sizeof(float));
}


/* mel [T][128] -> the speaker embedding [dim] (malloc'd) */
static float *qc_spk_embed(const QcSpk *s, const float *mel, int T){
    const int C = 512;
    const size_t tc = (size_t)T * C;
    float *o[4], *t1 = (float *)malloc(tc * sizeof(float)), *t2 = (float *)malloc(tc * sizeof(float));
    float *pa = (float *)malloc((size_t)T * 64 * sizeof(float)), *pb = (float *)malloc((size_t)T * 64 * sizeof(float)), *po = (float *)malloc((size_t)T * 64 * sizeof(float));
    for (int i = 0; i < 4; i++) o[i] = (float *)malloc(tc * sizeof(float));
    qc_conv_same(o[0], mel, T, &s->b0, 1);                                                      /* the first TDNN block */
    for (int b = 0; b < 3; b++) {
        const QcSeRes *r = &s->blk[b];
        const float *in = o[b];
        float *out = o[b + 1];
        qc_conv_same(t1, in, T, &r->tdnn1, 1);                                                  /* tdnn1 */
        for (int j = 0; j < 8; j++) {                                                           /* res2net: eight chunks of 64 channels */
            qc_gather(pa, t1, T, C, j * 64, 64);
            if (j == 0) memcpy(po, pa, (size_t)T * 64 * sizeof(float));
            else {
                if (j >= 2) for (int64_t i = 0; i < (int64_t)T * 64; i++) pa[i] += po[i];       /* + the previous part's output */
                qc_conv_same(pb, pa, T, &r->res[j - 1], 1);
                memcpy(po, pb, (size_t)T * 64 * sizeof(float));
            }
            qc_scatter(t2, po, T, C, j * 64, 64);
        }
        qc_conv_same(out, t2, T, &r->tdnn2, 1);                                                 /* tdnn2 */
        float mean[512], z[128], g[512];                                                        /* squeeze and excitation */
        for (int c = 0; c < C; c++) { double a = 0; for (int t = 0; t < T; t++) a += out[(size_t)t * C + c]; mean[c] = (float)(a / T); }
        qi_gemm(z, mean, 1, &r->se1.w.m, r->se1.b);
        for (int i = 0; i < 128; i++) if (z[i] < 0.f) z[i] = 0.f;
        qi_gemm(g, z, 1, &r->se2.w.m, r->se2.b);
        for (int c = 0; c < C; c++) g[c] = 1.0f / (1.0f + expf(-g[c]));
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) for (int c = 0; c < C; c++) out[(size_t)t * C + c] = out[(size_t)t * C + c] * g[c] + in[(size_t)t * C + c];
    }
    float *cat = (float *)malloc((size_t)T * 3 * C * sizeof(float));                            /* multi-layer feature aggregation: the three blocks' outputs */
    for (int t = 0; t < T; t++) for (int b = 0; b < 3; b++) memcpy(cat + (size_t)t * 3 * C + (size_t)b * C, o[b + 1] + (size_t)t * C, (size_t)C * sizeof(float));
    float *x = (float *)malloc((size_t)T * 1536 * sizeof(float));
    qc_conv_same(x, cat, T, &s->mfa, 1);
    /* attentive statistics pooling */
    double *mean = (double *)calloc(1536, sizeof(double)), *stdv = (double *)calloc(1536, sizeof(double));
    for (int t = 0; t < T; t++) for (int c = 0; c < 1536; c++) mean[c] += x[(size_t)t * 1536 + c];
    for (int c = 0; c < 1536; c++) mean[c] /= T;
    for (int t = 0; t < T; t++) for (int c = 0; c < 1536; c++) { const double d = x[(size_t)t * 1536 + c] - mean[c]; stdv[c] += d * d; }
    for (int c = 0; c < 1536; c++) { stdv[c] /= T; if (stdv[c] < 1e-12) stdv[c] = 1e-12; stdv[c] = sqrt(stdv[c]); }
    float *att = (float *)malloc((size_t)T * 4608 * sizeof(float));
    for (int t = 0; t < T; t++) {
        memcpy(att + (size_t)t * 4608, x + (size_t)t * 1536, 1536 * sizeof(float));
        for (int c = 0; c < 1536; c++) { att[(size_t)t * 4608 + 1536 + c] = (float)mean[c]; att[(size_t)t * 4608 + 3072 + c] = (float)stdv[c]; }
    }
    float *a1 = (float *)malloc((size_t)T * 128 * sizeof(float)), *a2 = (float *)malloc((size_t)T * 1536 * sizeof(float));
    qc_conv_same(a1, att, T, &s->asp_tdnn, 1);
    for (int64_t i = 0; i < (int64_t)T * 128; i++) a1[i] = tanhf(a1[i]);
    qc_conv_same(a2, a1, T, &s->asp_conv, 0);
    float *pooled = (float *)malloc(3072 * sizeof(float));
    for (int c = 0; c < 1536; c++) {                                                            /* softmax over time, then the weighted mean and std */
        double mx = -1e300, den = 0, mu = 0, var = 0;
        for (int t = 0; t < T; t++) if (a2[(size_t)t * 1536 + c] > mx) mx = a2[(size_t)t * 1536 + c];
        for (int t = 0; t < T; t++) { const double e = exp(a2[(size_t)t * 1536 + c] - mx); a2[(size_t)t * 1536 + c] = (float)e; den += e; }
        for (int t = 0; t < T; t++) mu += a2[(size_t)t * 1536 + c] / den * x[(size_t)t * 1536 + c];
        for (int t = 0; t < T; t++) { const double d = x[(size_t)t * 1536 + c] - mu; var += a2[(size_t)t * 1536 + c] / den * d * d; }
        if (var < 1e-12) var = 1e-12;
        pooled[c] = (float)mu; pooled[1536 + c] = (float)sqrt(var);
    }
    float *emb = (float *)malloc((size_t)s->dim * sizeof(float));
    qi_gemm(emb, pooled, 1, &s->fc.w.m, s->fc.b);
    for (int i = 0; i < 4; i++) free(o[i]);
    free(t1); free(t2); free(pa); free(pb); free(po); free(cat); free(x); free(mean); free(stdv); free(att); free(a1); free(a2); free(pooled);
    return emb;
}

/* ------------------------------------------------------------------------------------------------ the codec's encoder (Mimi) */

typedef struct { float *ln1w, *ln1b, *ln2w, *ln2b, *s1, *s2; AcdLin q, k, v, o, fc1, fc2; } QcMimiLayer;
typedef struct {
    int ok, nl, window;
    QtConv c0, resa[4], resb[4], down[4], last, ds;
    int ratio[4];
    QcMimiLayer L[8];
    AcdLin in_sem, in_ac;                                             /* the quantizers' input projections 512 -> 256 */
    float *sem_cb, *ac_cb[QT_GROUPS - 1];                              /* codebooks [2048][256] (embed_sum / usage), one semantic and fifteen acoustic */
} QcMimi;

static int qc_mimi_load(QcMimi *m, const char *dir, char *err, size_t errn){
    char path[4300];
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;                                                  /* no codec weights here: cloning off */
    fclose(f);
    shards S;
    st_init(&S, dir);
    if (!st_find(&S, "encoder.encoder.layers.0.conv.weight")) { st_destroy(&S); return 0; }
    size_t bytes = 0;
    char w[300], b[300];
    m->nl = 8;
    m->window = 0;                  /* the config says sliding_window 250, but the transformers Mimi the package runs applies a plain causal mask: so does this (0 = no window) */
    const char *ws = getenv("QT_MIMI_WINDOW");
    if (ws) m->window = atoi(ws);
#define QC_C(dst, idx, cout, cin, K, dil, hasb) do { \
        snprintf(w, sizeof w, "encoder.encoder.layers.%s.weight", idx); snprintf(b, sizeof b, "encoder.encoder.layers.%s.bias", idx); \
        if (qt_conv(&S, w, (hasb) ? b : NULL, cout, cin, K, dil, &(dst), &bytes, err, errn)) return -1; } while (0)
    QC_C(m->c0, "0.conv", 64, 1, 7, 1, 1);
    const int ratios[4] = {4, 5, 6, 8}, residx[4] = {1, 4, 7, 10}, downidx[4] = {3, 6, 9, 12};
    int ch = 64;
    for (int s = 0; s < 4; s++) {
        char id[64];
        snprintf(id, sizeof id, "%d.block.1.conv", residx[s]); QC_C(m->resa[s], id, ch / 2, ch, 3, 1, 1);
        snprintf(id, sizeof id, "%d.block.3.conv", residx[s]); QC_C(m->resb[s], id, ch, ch / 2, 1, 1, 1);
        snprintf(id, sizeof id, "%d.conv", downidx[s]);        QC_C(m->down[s], id, ch * 2, ch, ratios[s] * 2, 1, 1);
        m->ratio[s] = ratios[s];
        ch *= 2;
    }
    QC_C(m->last, "14.conv", 512, 1024, 3, 1, 1);
#undef QC_C
    if (qt_conv(&S, "encoder.downsample.conv.weight", NULL, 512, 512, 4, 1, &m->ds, &bytes, err, errn)) return -1;
    int64_t sd[2] = {512, 512}, s1[2] = {2048, 512}, s2[2] = {512, 2048}, sh1[1] = {512};
    for (int l = 0; l < m->nl; l++) {
        QcMimiLayer *y = &m->L[l];
        char p[160], n[300];
        snprintf(p, sizeof p, "encoder.encoder_transformer.layers.%d", l);
        const struct { const char *nm; AcdLin *dst; int N, K; int64_t *sh; } mats[6] = {
            {"self_attn.q_proj", &y->q, 512, 512, sd}, {"self_attn.k_proj", &y->k, 512, 512, sd}, {"self_attn.v_proj", &y->v, 512, 512, sd},
            {"self_attn.o_proj", &y->o, 512, 512, sd}, {"mlp.fc1", &y->fc1, 2048, 512, s1}, {"mlp.fc2", &y->fc2, 512, 2048, s2}};
        for (int j = 0; j < 6; j++) {
            snprintf(n, sizeof n, "%s.%s.weight", p, mats[j].nm);
            if (acd_load_lin(&S, n, NULL, mats[j].N, mats[j].K, 2, mats[j].sh, NULL, 0, 0, mats[j].dst, &bytes, err, errn)) return -1;
        }
        const struct { const char *nm; float **dst; } vecs[6] = {{"input_layernorm.weight", &y->ln1w}, {"input_layernorm.bias", &y->ln1b},
            {"post_attention_layernorm.weight", &y->ln2w}, {"post_attention_layernorm.bias", &y->ln2b}, {"self_attn_layer_scale.scale", &y->s1}, {"mlp_layer_scale.scale", &y->s2}};
        for (int j = 0; j < 6; j++) {
            snprintf(n, sizeof n, "%s.%s", p, vecs[j].nm);
            if (!(*vecs[j].dst = acd_read_f32(&S, n, 1, sh1, err, errn))) return -1;
        }
    }
    int64_t sp[2] = {256, 512};
    if (acd_load_lin(&S, "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight", NULL, 256, 512, 3, (int64_t[3]){256, 512, 1}, NULL, 0, 0, &m->in_sem, &bytes, err, errn)) return -1;
    if (acd_load_lin(&S, "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight", NULL, 256, 512, 3, (int64_t[3]){256, 512, 1}, NULL, 0, 0, &m->in_ac, &bytes, err, errn)) return -1;
    (void)sp;
    for (int g = 0; g < QT_GROUPS; g++) {                              /* the codebooks: embed_sum / max(usage, 1e-5) */
        char n[300];
        const char *kind = g == 0 ? "semantic" : "acoustic";
        const int li = g == 0 ? 0 : g - 1;
        snprintf(n, sizeof n, "encoder.quantizer.%s_residual_vector_quantizer.layers.%d.codebook.embed_sum", kind, li);
        float *es = qt_read(&S, n, 2048 * 256, err, errn);
        snprintf(n, sizeof n, "encoder.quantizer.%s_residual_vector_quantizer.layers.%d.codebook.cluster_usage", kind, li);
        float *us = qt_read(&S, n, 2048, err, errn);
        if (!es || !us) { free(es); free(us); return -1; }
        for (int i = 0; i < 2048; i++) { const float u = us[i] < 1e-5f ? 1e-5f : us[i]; for (int d = 0; d < 256; d++) es[(size_t)i * 256 + d] /= u; }
        free(us);
        if (g == 0) m->sem_cb = es; else m->ac_cb[g - 1] = es;
    }
    st_destroy(&S);
    m->ok = 1;
    return 0;
}

/* the 8-layer transformer of the encoder on x [T][512] in place: pre-LayerNorm, rotary attention (causal, over the last `window` positions),
 * layer scales, GELU feed-forward */
static void qc_mimi_transformer(const QcMimi *m, float *x, int T){
    const int D = 512, H = 8, hd = 64, I = 2048;
    float *n = (float *)malloc((size_t)T * D * sizeof(float)), *q = (float *)malloc((size_t)T * D * sizeof(float)), *k = (float *)malloc((size_t)T * D * sizeof(float));
    float *v = (float *)malloc((size_t)T * D * sizeof(float)), *cx = (float *)malloc((size_t)T * D * sizeof(float)), *o = (float *)malloc((size_t)T * D * sizeof(float));
    float *g = (float *)malloc((size_t)T * I * sizeof(float));
    float *cs = (float *)malloc((size_t)T * hd * sizeof(float)), *sn = (float *)malloc((size_t)T * hd * sizeof(float));
    acd_rope_table(cs, sn, T, hd, 10000.0f);
    for (int li = 0; li < m->nl; li++) {
        const QcMimiLayer *l = &m->L[li];
        qc_layernorm(n, x, T, D, l->ln1w, l->ln1b, 1e-5f);
        acd_lin(q, n, T, &l->q); acd_lin(k, n, T, &l->k); acd_lin(v, n, T, &l->v);
        acd_rope(q, T, H, hd, cs, sn);
        acd_rope(k, T, H, hd, cs, sn);
        acd_attention_ex(cx, D, q, D, T, k, v, D, T, H, H, hd, m->window > 0 ? m->window - 1 : -1, 1);
        acd_lin(o, cx, T, &l->o);
        for (int64_t i = 0; i < (int64_t)T * D; i++) x[i] += l->s1[i % D] * o[i];
        qc_layernorm(n, x, T, D, l->ln2w, l->ln2b, 1e-5f);
        acd_lin(g, n, T, &l->fc1);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)T * I; i++) g[i] = qt_gelu(g[i]);
        acd_lin(o, g, T, &l->fc2);
        for (int64_t i = 0; i < (int64_t)T * D; i++) x[i] += l->s2[i % D] * o[i];
    }
    free(n); free(q); free(k); free(v); free(cx); free(o); free(g); free(cs); free(sn);
}

typedef struct { float *conv, *tr, *emb; int t25, t12; } QcMimiDump;

/* wav [n] (24 kHz) -> codes [frames][16] (malloc'd), *frames = ceil(n / 1920) of them. dump, when not NULL, gets the stages' outputs. */
static int *qc_mimi_encode(const QcMimi *m, const float *wav, int n, int *frames, QcMimiDump *dump){
    int T = n;
    float *x = (float *)malloc((size_t)(T > 0 ? T : 1) * sizeof(float));
    memcpy(x, wav, (size_t)T * sizeof(float));
    float *y = (float *)malloc((size_t)T * 64 * sizeof(float));
    T = qc_conv(y, x, T, &m->c0, 1, 0);                                  /* 1 -> 64 channels, k 7 */
    free(x); x = y;
    int ch = 64;
    for (int s = 0; s < 4; s++) {
        float *h = (float *)malloc((size_t)T * ch * sizeof(float)), *h2 = (float *)malloc((size_t)T * ch * sizeof(float));
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)T * ch; i++) h[i] = qc_elu(x[i]);
        float *mid = (float *)malloc((size_t)T * (ch / 2) * sizeof(float));
        qc_conv(mid, h, T, &m->resa[s], 1, 0);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)T * (ch / 2); i++) mid[i] = qc_elu(mid[i]);
        qc_conv(h2, mid, T, &m->resb[s], 1, 0);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)T * ch; i++) x[i] += h2[i];     /* the residual */
        free(h); free(h2); free(mid);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)T * ch; i++) x[i] = qc_elu(x[i]);
        const int Ty = (T + m->ratio[s] - 1) / m->ratio[s];
        y = (float *)malloc((size_t)Ty * ch * 2 * sizeof(float));
        qc_conv(y, x, T, &m->down[s], m->ratio[s], 0);
        free(x); x = y; T = Ty; ch *= 2;
    }
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * ch; i++) x[i] = qc_elu(x[i]);
    y = (float *)malloc((size_t)T * 512 * sizeof(float));
    qc_conv(y, x, T, &m->last, 1, 0);
    free(x); x = y;
    const int T25 = T;
    if (dump) { dump->t25 = T25; dump->conv = (float *)malloc((size_t)T25 * 512 * sizeof(float)); memcpy(dump->conv, x, (size_t)T25 * 512 * sizeof(float)); }
    qc_mimi_transformer(m, x, T25);
    if (dump) { dump->tr = (float *)malloc((size_t)T25 * 512 * sizeof(float)); memcpy(dump->tr, x, (size_t)T25 * 512 * sizeof(float)); }
    const int T12 = (T25 + 1) / 2;
    y = (float *)malloc((size_t)T12 * 512 * sizeof(float));
    qc_conv(y, x, T25, &m->ds, 2, 1);                                    /* stride 2, replicate padding */
    free(x); x = y;
    if (dump) { dump->t12 = T12; dump->emb = (float *)malloc((size_t)T12 * 512 * sizeof(float)); memcpy(dump->emb, x, (size_t)T12 * 512 * sizeof(float)); }
    /* the quantizer: a semantic codebook and fifteen acoustic ones, each residual from its own projection of the embeddings */
    const int want = (n + 1919) / 1920;
    const int F = want < T12 ? want : T12;
    int *codes = (int *)malloc((size_t)(F > 0 ? F : 1) * QT_GROUPS * sizeof(int));
    float *res = (float *)malloc((size_t)T12 * 256 * sizeof(float));
    for (int part = 0; part < 2; part++) {
        acd_lin(res, x, T12, part == 0 ? &m->in_sem : &m->in_ac);
        const int nq = part == 0 ? 1 : QT_GROUPS - 1;
        for (int qi = 0; qi < nq; qi++) {
            const float *cb = part == 0 ? m->sem_cb : m->ac_cb[qi];
            const int col = part == 0 ? 0 : qi + 1;
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < T12; t++) {
                float *r = res + (size_t)t * 256;
                int best = 0; double bd = 1e300;
                for (int e = 0; e < 2048; e++) {
                    const float *c = cb + (size_t)e * 256;
                    double d = 0;
                    for (int j = 0; j < 256; j++) { const double df = (double)r[j] - c[j]; d += df * df; }
                    if (d < bd) { bd = d; best = e; }
                }
                if (t < F) codes[(size_t)t * QT_GROUPS + col] = best;
                const float *c = cb + (size_t)best * 256;
                for (int j = 0; j < 256; j++) r[j] -= c[j];
            }
        }
    }
    free(res); free(x);
    *frames = F;
    return codes;
}
