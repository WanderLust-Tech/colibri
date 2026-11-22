/* qwen3asr.c -- speech to text with Qwen3-ASR (Qwen/Qwen3-ASR-0.6B, -1.7B), docs/qwen3-speech.md.
 *
 * What the reference (the qwen-asr package, transformers' Qwen3ASRForConditionalGeneration) does, and so what this does:
 *
 *   16 kHz mono audio
 *     -> log-mel features, 128 bins, n_fft 400, hop 160 (WhisperFeatureExtractor: 100 frames a second)
 *     -> the audio encoder: the mel is cut into chunks of 100 frames; each chunk goes through three Conv2d(3x3, stride 2, pad 1)
 *        with GELU (100 frames become 13), a bias-free Linear over the 480 channels x 16 mel positions of each frame, and a
 *        sinusoidal position inside the chunk; the chunks' frames are joined; pre-LayerNorm transformer layers attend inside
 *        windows of 800 mel frames (104 encoder frames); ln_post, proj1, GELU, proj2 to the decoder's width
 *     -> a Qwen3 decoder (plain: RMSNorm, GQA with per-head q/k RMSNorm, RoPE, SwiGLU; the checkpoint says interleaved MRoPE,
 *        but all three position streams are the same number for text and audio, which makes it ordinary RoPE) reads
 *          <|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|> [one slot per encoder frame, filled with the
 *          encoder's output] <|audio_end|><|im_end|>\n<|im_start|>assistant\n   [language X<asr_text> when the language is forced]
 *        and writes, greedily,   language English<asr_text>the text<|im_end|>
 *     -> parse_asr_output: the language and the text; repetition loops are cut (detect_and_fix_repetitions).
 *
 * The kernels and the Qwen3 layer code are the ones the ACE-Step engine and qwen3embed use (qi_gemm.h, acestep_dit.h).
 *
 * Usage:
 *   qwen3asr --model DIR --wav FILE [--language NAME] [--context TEXT] [--max-new N] [--threads N]
 *       prints the language and the text (the WAV must be 16 kHz; the gateway resamples).
 *   qwen3asr --model DIR --wav FILE --ref REFDIR
 *       holds each stage to the tensors tools/qwen3_asr_ref.py wrote (features, conv, encoder, prompt ids, generated ids).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <strings.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "st.h"
#include "json.h"
#include "qi_gemm.h"
#include "acestep_text.h"
#ifdef COLI_VULKAN
#include "backend_vulkan.h"
#include "vk_chain.h"
#endif

#define AS_SR 16000
#define AS_NFFT 400
#define AS_HOP 160
#define AS_NMEL 128
#define AS_CHUNK 100                 /* mel frames per encoder chunk (n_window * 2) */

static double as_now(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------------------------------------ the log-mel features */

static double as_hz_to_mel(double f){                           /* Slaney's scale: linear below 1 kHz, logarithmic above */
    const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = log(6.4) / 27.0;
    return f >= min_log_hz ? min_log_mel + log(f / min_log_hz) / logstep : f / f_sp;
}
static double as_mel_to_hz(double m){
    const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = log(6.4) / 27.0;
    return m >= min_log_mel ? min_log_hz * exp(logstep * (m - min_log_mel)) : f_sp * m;
}

/* the mel filter bank of transformers' mel_filter_bank(201, 128, 0, 8000, 16000, norm="slaney", mel_scale="slaney"): [201][128] */
static float *as_mel_filters(void){
    const int NK = AS_NFFT / 2 + 1;
    double edges[AS_NMEL + 2];
    const double lo = as_hz_to_mel(0.0), hi = as_hz_to_mel(8000.0);
    for (int i = 0; i < AS_NMEL + 2; i++) edges[i] = as_mel_to_hz(lo + (hi - lo) * i / (AS_NMEL + 1));
    float *fb = (float *)calloc((size_t)NK * AS_NMEL, sizeof(float));
    if (!fb) { fprintf(stderr, "qwen3asr: out of memory\n"); exit(1); }
    for (int k = 0; k < NK; k++) {
        const double fk = 8000.0 * k / (NK - 1);
        for (int m = 0; m < AS_NMEL; m++) {
            const double l = edges[m], c = edges[m + 1], u = edges[m + 2];
            double down = (fk - l) / (c - l), up = (u - fk) / (u - c), w = down < up ? down : up;
            if (w < 0) w = 0;
            fb[(size_t)k * AS_NMEL + m] = (float)(w * 2.0 / (u - l));
        }
    }
    return fb;
}

/* features [128][T], T = n / 160 frames: reflect-padded STFT with a periodic Hann window, power, mel, log10, floor at max - 8, (x + 4) / 4 */
static float *as_mel(const float *wav, int n, int *T_out){
    const int NK = AS_NFFT / 2 + 1, pad = AS_NFFT / 2;
    const int T = n / AS_HOP;
    *T_out = T;
    if (T < 1) return NULL;
    const int np = n + 2 * pad;
    float *xp = (float *)malloc((size_t)np * sizeof(float));
    for (int i = 0; i < pad; i++) xp[i] = wav[pad - i];
    memcpy(xp + pad, wav, (size_t)n * sizeof(float));
    for (int i = 0; i < pad; i++) xp[pad + n + i] = wav[n - 2 - i];
    double *window = (double *)malloc((size_t)AS_NFFT * sizeof(double));
    for (int i = 0; i < AS_NFFT; i++) window[i] = 0.5 - 0.5 * cos(2.0 * M_PI * i / AS_NFFT);
    double *ct = (double *)malloc((size_t)NK * AS_NFFT * sizeof(double)), *st = (double *)malloc((size_t)NK * AS_NFFT * sizeof(double));
    for (int k = 0; k < NK; k++)
        for (int i = 0; i < AS_NFFT; i++) {
            const double a = 2.0 * M_PI * (double)((long long)k * i % AS_NFFT) / AS_NFFT;
            ct[(size_t)k * AS_NFFT + i] = cos(a) * window[i]; st[(size_t)k * AS_NFFT + i] = sin(a) * window[i];
        }
    float *fb = as_mel_filters();
    float *out = (float *)malloc((size_t)AS_NMEL * T * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float *fr = xp + (size_t)t * AS_HOP;
        double pw[AS_NFFT / 2 + 1];
        for (int k = 0; k < NK; k++) {
            double re = 0, im = 0;
            const double *c = ct + (size_t)k * AS_NFFT, *s = st + (size_t)k * AS_NFFT;
            for (int i = 0; i < AS_NFFT; i++) { re += fr[i] * c[i]; im += fr[i] * s[i]; }
            pw[k] = re * re + im * im;
        }
        for (int m = 0; m < AS_NMEL; m++) {
            double acc = 0;
            for (int k = 0; k < NK; k++) acc += (double)fb[(size_t)k * AS_NMEL + m] * pw[k];
            out[(size_t)m * T + t] = (float)log10(acc > 1e-10 ? acc : 1e-10);
        }
    }
    float mx = -1e30f;
    for (size_t i = 0; i < (size_t)AS_NMEL * T; i++) if (out[i] > mx) mx = out[i];
    for (size_t i = 0; i < (size_t)AS_NMEL * T; i++) { float v = out[i] < mx - 8.0f ? mx - 8.0f : out[i]; out[i] = (v + 4.0f) / 4.0f; }
    free(xp); free(window); free(ct); free(st); free(fb);
    return out;
}

/* ------------------------------------------------------------------------------------------------ the model */

typedef struct { AcdLin q, k, v, o, fc1, fc2; float *ln1w, *ln1b, *ln2w, *ln2b; } AsEncLayer;

typedef struct {
    /* audio encoder */
    int el, ed, eh, eff, eout, dsh;                              /* layers, d_model, heads, ffn, output width, conv channels */
    float *c1w, *c1b, *c2b, *c3b;                                /* conv1 is read as f32 [480][9]; conv2 and conv3 are AcdLin [480][4320] */
    AcdLin c2, c3, conv_out, proj1, proj2;
    AsEncLayer *EL;
    float *lnpw, *lnpb;
    /* decoder */
    int nl, D, H, Hk, hd, inter, vocab;
    float eps, theta;
    AcdEncLayer *L;
    float *norm;
    uint16_t *embed;                                             /* bf16 [vocab][D] */
    AcdLin head;
    int audio_pad, audio_start, audio_end, im_end, eos2, asr_text;
    int aligner, nhead, ts_id, ts_ms;                            /* forced aligner: its head's width, the <timestamp> token, the time of one class in ms */
    Tok tok;
    size_t wbytes;
} Asr;

static void as_die(const char *m){ fprintf(stderr, "qwen3asr: %s\n", m); exit(1); }

static float *as_rd(shards *S, const char *name, int n, char *err, size_t errn){
    int64_t sh[1] = {n};
    return acd_read_f32(S, name, 1, sh, err, errn);
}

static int as_load(Asr *m, const char *dir, char *err, size_t errn){
    memset(m, 0, sizeof *m);
    char path[4300];
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errn, "cannot open %s", path); return -1; }
    fseek(f, 0, SEEK_END); long fn = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)fn + 1);
    if (!buf || fread(buf, 1, (size_t)fn, f) != (size_t)fn) { fclose(f); free(buf); snprintf(err, errn, "cannot read %s", path); return -1; }
    fclose(f); buf[fn] = 0;
    jval *root = json_parse_checked(buf);
    free(buf);
    jval *th = root ? json_get(root, "thinker_config") : NULL;
    jval *ac = th ? json_get(th, "audio_config") : NULL, *tc = th ? json_get(th, "text_config") : NULL;
    if (!ac || !tc) { snprintf(err, errn, "config.json has no thinker_config.audio_config and text_config: not a Qwen3-ASR checkpoint"); return -1; }
    m->el = acd_json_int(ac, "encoder_layers", 0); m->ed = acd_json_int(ac, "d_model", 0); m->eh = acd_json_int(ac, "encoder_attention_heads", 0);
    m->eff = acd_json_int(ac, "encoder_ffn_dim", 0); m->eout = acd_json_int(ac, "output_dim", 0); m->dsh = acd_json_int(ac, "downsample_hidden_size", 480);
    m->nl = acd_json_int(tc, "num_hidden_layers", 0); m->D = acd_json_int(tc, "hidden_size", 0); m->H = acd_json_int(tc, "num_attention_heads", 0);
    m->Hk = acd_json_int(tc, "num_key_value_heads", 0); m->hd = acd_json_int(tc, "head_dim", 128); m->inter = acd_json_int(tc, "intermediate_size", 0);
    jval *tmt = json_get(th, "model_type");
    m->aligner = tmt && tmt->t == J_STR && strstr(tmt->str, "forced_aligner") != NULL;
    m->nhead = acd_json_int(th, "classify_num", 0);
    m->ts_id = acd_json_int(root, "timestamp_token_id", -1);
    m->ts_ms = acd_json_int(root, "timestamp_segment_time", 80);
    m->vocab = acd_json_int(tc, "vocab_size", 0); m->eps = (float)acd_json_num(tc, "rms_norm_eps", 1e-6); m->theta = (float)acd_json_num(tc, "rope_theta", 1e6);
    if (acd_json_int(ac, "num_mel_bins", 128) != AS_NMEL || m->dsh != 480 || m->eout != m->D || !m->el || !m->nl || !m->ed || m->ed % m->eh) {
        snprintf(err, errn, "unsupported Qwen3-ASR geometry in config.json"); return -1;
    }
    shards S;
    st_init(&S, dir);
    const char *E = "thinker.audio_tower";
    char n[300], b[300];
    int64_t sh1[4] = {m->dsh, 1, 3, 3}, sh2[4] = {m->dsh, m->dsh, 3, 3};
    snprintf(n, sizeof n, "%s.conv2d1.weight", E);
    if (!(m->c1w = acd_read_f32(&S, n, 4, sh1, err, errn))) return -1;
    snprintf(n, sizeof n, "%s.conv2d1.bias", E);  if (!(m->c1b = as_rd(&S, n, m->dsh, err, errn))) return -1;
    snprintf(n, sizeof n, "%s.conv2d2.weight", E); snprintf(b, sizeof b, "%s.conv2d2.bias", E);
    if (acd_load_lin(&S, n, b, m->dsh, m->dsh * 9, 4, sh2, NULL, 0, 0, &m->c2, &m->wbytes, err, errn)) return -1;
    snprintf(n, sizeof n, "%s.conv2d3.weight", E); snprintf(b, sizeof b, "%s.conv2d3.bias", E);
    if (acd_load_lin(&S, n, b, m->dsh, m->dsh * 9, 4, sh2, NULL, 0, 0, &m->c3, &m->wbytes, err, errn)) return -1;
    int64_t sco[2] = {m->ed, (int64_t)m->dsh * 16}, sdd[2] = {m->ed, m->ed}, sou[2] = {m->eout, m->ed};
    snprintf(n, sizeof n, "%s.conv_out.weight", E);
    if (acd_load_lin(&S, n, NULL, m->ed, m->dsh * 16, 2, sco, NULL, 0, 0, &m->conv_out, &m->wbytes, err, errn)) return -1;
    snprintf(n, sizeof n, "%s.proj1.weight", E); snprintf(b, sizeof b, "%s.proj1.bias", E);
    if (acd_load_lin(&S, n, b, m->ed, m->ed, 2, sdd, NULL, 0, 0, &m->proj1, &m->wbytes, err, errn)) return -1;
    snprintf(n, sizeof n, "%s.proj2.weight", E); snprintf(b, sizeof b, "%s.proj2.bias", E);
    if (acd_load_lin(&S, n, b, m->eout, m->ed, 2, sou, NULL, 0, 0, &m->proj2, &m->wbytes, err, errn)) return -1;
    snprintf(n, sizeof n, "%s.ln_post.weight", E); if (!(m->lnpw = as_rd(&S, n, m->ed, err, errn))) return -1;
    snprintf(n, sizeof n, "%s.ln_post.bias", E);   if (!(m->lnpb = as_rd(&S, n, m->ed, err, errn))) return -1;
    m->EL = (AsEncLayer *)calloc((size_t)m->el, sizeof *m->EL);
    for (int i = 0; i < m->el; i++) {
        AsEncLayer *l = &m->EL[i];
        int64_t sf1[2] = {m->eff, m->ed}, sf2[2] = {m->ed, m->eff};
        struct { const char *w, *bb; AcdLin *dst; int N, K; int64_t *sh; } mats[6] = {
            {"self_attn.q_proj", NULL, &l->q, m->ed, m->ed, sdd}, {"self_attn.k_proj", NULL, &l->k, m->ed, m->ed, sdd},
            {"self_attn.v_proj", NULL, &l->v, m->ed, m->ed, sdd}, {"self_attn.out_proj", NULL, &l->o, m->ed, m->ed, sdd},
            {"fc1", NULL, &l->fc1, m->eff, m->ed, sf1}, {"fc2", NULL, &l->fc2, m->ed, m->eff, sf2}};
        for (int j = 0; j < 6; j++) {
            snprintf(n, sizeof n, "%s.layers.%d.%s.weight", E, i, mats[j].w); snprintf(b, sizeof b, "%s.layers.%d.%s.bias", E, i, mats[j].w);
            if (acd_load_lin(&S, n, b, mats[j].N, mats[j].K, 2, mats[j].sh, NULL, 0, 0, mats[j].dst, &m->wbytes, err, errn)) return -1;
        }
        struct { const char *nm; float **dst; } lns[4] = {{"self_attn_layer_norm.weight", &l->ln1w}, {"self_attn_layer_norm.bias", &l->ln1b},
                                                          {"final_layer_norm.weight", &l->ln2w}, {"final_layer_norm.bias", &l->ln2b}};
        for (int j = 0; j < 4; j++) {
            snprintf(n, sizeof n, "%s.layers.%d.%s", E, i, lns[j].nm);
            if (!(*lns[j].dst = as_rd(&S, n, m->ed, err, errn))) return -1;
        }
    }
    /* the decoder */
    const char *T = "thinker.model";
    m->L = (AcdEncLayer *)calloc((size_t)m->nl, sizeof *m->L);
    int64_t shd[1] = {m->D};
    for (int i = 0; i < m->nl; i++) {
        char p[300];
        snprintf(p, sizeof p, "%s.layers.%d.self_attn", T, i);
        if (acd_load_attn(&S, p, m->D, m->H, m->Hk, m->hd, &m->L[i].sa, &m->wbytes, err, errn)) return -1;
        snprintf(n, sizeof n, "%s.layers.%d.input_layernorm.weight", T, i);          if (!(m->L[i].ln1 = acd_read_f32(&S, n, 1, shd, err, errn))) return -1;
        snprintf(n, sizeof n, "%s.layers.%d.post_attention_layernorm.weight", T, i); if (!(m->L[i].ln2 = acd_read_f32(&S, n, 1, shd, err, errn))) return -1;
        snprintf(p, sizeof p, "%s.layers.%d.mlp", T, i);
        if (acd_load_mlp(&S, p, m->D, m->inter, &m->L[i].gate, &m->L[i].up, &m->L[i].down, &m->wbytes, err, errn)) return -1;
    }
    snprintf(n, sizeof n, "%s.norm.weight", T); if (!(m->norm = acd_read_f32(&S, n, 1, shd, err, errn))) return -1;
    snprintf(n, sizeof n, "%s.embed_tokens.weight", T);
    st_tensor *te = st_find(&S, n);
    if (!te || te->dtype != 0 || te->numel != (int64_t)m->vocab * m->D) { snprintf(err, errn, "the embedding table is missing or not bf16"); return -1; }
    m->embed = (uint16_t *)malloc((size_t)te->numel * 2);
    st_read_raw(&S, n, m->embed, 1);
    m->wbytes += (size_t)te->numel * 2;
    if (!m->nhead) m->nhead = m->vocab;
    int64_t shh[2] = {m->nhead, m->D};
    const char *hn = st_find(&S, "thinker.lm_head.weight") ? "thinker.lm_head.weight" : n;
    if (acd_load_lin(&S, hn, NULL, m->nhead, m->D, 2, shh, NULL, 0, 0, &m->head, &m->wbytes, err, errn)) return -1;
    st_destroy(&S);
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    FILE *tf = fopen(path, "rb");
    if (!tf) { snprintf(err, errn, "no tokenizer.json in the model folder (python tools/qwen3_tokenizer.py DIR makes it from vocab.json and merges.txt)"); return -1; }
    fclose(tf);
    tok_load(&m->tok, path);
    m->audio_pad = tok_id_of(&m->tok, "<|audio_pad|>"); m->audio_start = tok_id_of(&m->tok, "<|audio_start|>");
    m->audio_end = tok_id_of(&m->tok, "<|audio_end|>"); m->im_end = tok_id_of(&m->tok, "<|im_end|>");
    m->eos2 = tok_id_of(&m->tok, "<|endoftext|>"); m->asr_text = tok_id_of(&m->tok, "<asr_text>");
    if (m->audio_pad < 0 || m->audio_start < 0 || m->audio_end < 0 || m->im_end < 0 || m->asr_text < 0) { snprintf(err, errn, "the tokenizer lacks the audio tokens"); return -1; }
    if (m->aligner && (m->ts_id < 0 || tok_id_of(&m->tok, "<timestamp>") != m->ts_id)) { snprintf(err, errn, "the tokenizer's <timestamp> is not the config's timestamp_token_id"); return -1; }
    return 0;
}

/* ------------------------------------------------------------------------------------------------ the audio encoder */

static void as_lin(float *Y, const float *X, int M, const AcdLin *l);   /* acd_lin; on the device when there is one (defined below) */
#ifdef COLI_VULKAN
static float *as_vk_encode(const Asr *m, const float *x, int F, int win);   /* the layers and the output projections on the device; NULL: the CPU does them */
#endif

static inline float as_gelu(float x){ return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

static inline int as_fdiv(int a, int b){ int q = a / b; return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q; }   /* Python's // */

static int as_conv_len(int L){                                    /* encoder frames of a chunk of L mel frames (_get_feat_extract_output_lengths) */
    const int leave = L % 100, feat = as_fdiv(leave - 1, 2) + 1;
    return as_fdiv(as_fdiv(feat - 1, 2) + 1 - 1, 2) + 1 + (L / 100) * 13;
}

static void as_layernorm(float *dst, const float *src, int rows, int n, const float *w, const float *b){
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *x = src + (size_t)r * n; float *y = dst + (size_t)r * n;
        double mu = 0, var = 0;
        for (int i = 0; i < n; i++) mu += x[i];
        mu /= n;
        for (int i = 0; i < n; i++) { double d = x[i] - mu; var += d * d; }
        var /= n;
        const float inv = 1.0f / sqrtf((float)var + 1e-5f);
        for (int i = 0; i < n; i++) y[i] = (float)((x[i] - mu) * inv) * w[i] + b[i];
    }
}

/* One chunk of 100 mel frames (zero-padded when shorter, as the reference pads it) through the three convolutions and conv_out:
 * mel [128][100] -> out [13][d_model]. */
static void as_conv_chunk(const Asr *m, const float *mel, float *out){
    const int C = m->dsh;
    const int F1 = 64, W1 = 50, F2 = 32, W2 = 25, F3 = 16, W3 = 13;
    float *a1 = (float *)malloc((size_t)F1 * W1 * C * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int pos = 0; pos < F1 * W1; pos++) {
        const int f = pos / W1, w = pos % W1;
        float patch[9];
        for (int kh = 0; kh < 3; kh++)
            for (int kw = 0; kw < 3; kw++) {
                const int fi = 2 * f - 1 + kh, wi = 2 * w - 1 + kw;
                patch[kh * 3 + kw] = (fi < 0 || fi >= AS_NMEL || wi < 0 || wi >= AS_CHUNK) ? 0.f : mel[(size_t)fi * AS_CHUNK + wi];
            }
        for (int co = 0; co < C; co++) {
            float s = m->c1b[co];
            const float *wt = m->c1w + (size_t)co * 9;
            for (int j = 0; j < 9; j++) s += wt[j] * patch[j];
            a1[(size_t)pos * C + co] = as_gelu(s);
        }
    }
    const int K = C * 9;
    float *col = (float *)malloc((size_t)F2 * W2 * K * sizeof(float));
    float *a2 = (float *)malloc((size_t)F2 * W2 * C * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int pos = 0; pos < F2 * W2; pos++) {
        const int f = pos / W2, w = pos % W2;
        float *c = col + (size_t)pos * K;
        for (int ci = 0; ci < C; ci++)
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++) {
                    const int fi = 2 * f - 1 + kh, wi = 2 * w - 1 + kw;
                    c[ci * 9 + kh * 3 + kw] = (fi < 0 || fi >= F1 || wi < 0 || wi >= W1) ? 0.f : a1[((size_t)fi * W1 + wi) * C + ci];
                }
    }
    free(a1);
    as_lin(a2, col, F2 * W2, &m->c2);
    free(col);
    for (size_t i = 0; i < (size_t)F2 * W2 * C; i++) a2[i] = as_gelu(a2[i]);
    col = (float *)malloc((size_t)F3 * W3 * K * sizeof(float));
    float *a3 = (float *)malloc((size_t)F3 * W3 * C * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int pos = 0; pos < F3 * W3; pos++) {
        const int f = pos / W3, w = pos % W3;
        float *c = col + (size_t)pos * K;
        for (int ci = 0; ci < C; ci++)
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++) {
                    const int fi = 2 * f - 1 + kh, wi = 2 * w - 1 + kw;
                    c[ci * 9 + kh * 3 + kw] = (fi < 0 || fi >= F2 || wi < 0 || wi >= W2) ? 0.f : a2[((size_t)fi * W2 + wi) * C + ci];
                }
    }
    free(a2);
    as_lin(a3, col, F3 * W3, &m->c3);
    free(col);
    for (size_t i = 0; i < (size_t)F3 * W3 * C; i++) a3[i] = as_gelu(a3[i]);
    float *vec = (float *)malloc((size_t)W3 * C * F3 * sizeof(float));
    for (int t = 0; t < W3; t++)
        for (int c = 0; c < C; c++)
            for (int f = 0; f < F3; f++) vec[(size_t)t * C * F3 + (size_t)c * F3 + f] = a3[((size_t)f * W3 + t) * C + c];
    free(a3);
    as_lin(out, vec, W3, &m->conv_out);
    free(vec);
}

/* mel [128][T] -> encoder frames [F][eout] (malloc'd), F = *nf; conv_dump, when not NULL, gets the frames before the layers [F][d_model] */
static float *as_encode(const Asr *m, const float *mel, int T, int *nf, float **conv_dump){
    const int D = m->ed, nch = (T + AS_CHUNK - 1) / AS_CHUNK;
    int F = 0;
    for (int c = 0; c < nch; c++) F += as_conv_len(c == nch - 1 ? T - AS_CHUNK * (nch - 1) : AS_CHUNK);
    float *x = (float *)malloc((size_t)F * D * sizeof(float));
    float *pe = (float *)malloc((size_t)13 * D * sizeof(float));
    {
        const int h = D / 2;
        const float lti = (float)(log(10000.0) / (h - 1));
        for (int t = 0; t < 13; t++)
            for (int j = 0; j < h; j++) {
                const float a = (float)t * expf(-lti * (float)j);
                pe[(size_t)t * D + j] = sinf(a); pe[(size_t)t * D + h + j] = cosf(a);
            }
    }
    float *chunk = (float *)malloc((size_t)AS_NMEL * AS_CHUNK * sizeof(float)), *co = (float *)malloc((size_t)13 * D * sizeof(float));
    int at = 0;
    for (int c = 0; c < nch; c++) {
        const int len = c == nch - 1 ? T - AS_CHUNK * (nch - 1) : AS_CHUNK;
        memset(chunk, 0, (size_t)AS_NMEL * AS_CHUNK * sizeof(float));
        for (int f = 0; f < AS_NMEL; f++) memcpy(chunk + (size_t)f * AS_CHUNK, mel + (size_t)f * T + (size_t)c * AS_CHUNK, (size_t)len * sizeof(float));
        as_conv_chunk(m, chunk, co);
        const int v = as_conv_len(len);
        for (int t = 0; t < v; t++)
            for (int j = 0; j < D; j++) x[(size_t)(at + t) * D + j] = co[(size_t)t * D + j] + pe[(size_t)t * D + j];
        at += v;
    }
    free(chunk); free(co); free(pe);
    if (conv_dump) { *conv_dump = (float *)malloc((size_t)F * D * sizeof(float)); memcpy(*conv_dump, x, (size_t)F * D * sizeof(float)); }
    /* the layers: attention inside windows of 104 frames */
    const int WIN = (getenv("AS_FULL_ATTN") || m->aligner) ? (F > 0 ? F : 1) :   /* the aligner: as the reference package runs it, over the whole clip */
        13 * (800 / AS_CHUNK), H = m->eh, hd = D / H;
#ifdef COLI_VULKAN
    {
        float *dout = as_vk_encode(m, x, F, WIN);
        if (dout) { free(x); *nf = F; return dout; }
    }
#endif
    float *n1 = (float *)malloc((size_t)F * D * sizeof(float)), *q = (float *)malloc((size_t)F * D * sizeof(float));
    float *k = (float *)malloc((size_t)F * D * sizeof(float)), *v = (float *)malloc((size_t)F * D * sizeof(float));
    float *cx = (float *)malloc((size_t)F * D * sizeof(float)), *o = (float *)malloc((size_t)F * D * sizeof(float));
    float *g = (float *)malloc((size_t)F * m->eff * sizeof(float));
    for (int li = 0; li < m->el; li++) {
        const AsEncLayer *l = &m->EL[li];
        as_layernorm(n1, x, F, D, l->ln1w, l->ln1b);
        as_lin(q, n1, F, &l->q); as_lin(k, n1, F, &l->k); as_lin(v, n1, F, &l->v);
        for (int w0 = 0; w0 < F; w0 += WIN) {
            const int wl = F - w0 < WIN ? F - w0 : WIN;
            acd_attention_ex(cx + (size_t)w0 * D, D, q + (size_t)w0 * D, D, wl, k + (size_t)w0 * D, v + (size_t)w0 * D, D, wl, H, H, hd, -1, 0);
        }
        as_lin(o, cx, F, &l->o);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)F * D; i++) x[i] += o[i];
        as_layernorm(n1, x, F, D, l->ln2w, l->ln2b);
        as_lin(g, n1, F, &l->fc1);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)F * m->eff; i++) g[i] = as_gelu(g[i]);
        as_lin(o, g, F, &l->fc2);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)F * D; i++) x[i] += o[i];
    }
    as_layernorm(n1, x, F, D, m->lnpw, m->lnpb);
    as_lin(g, n1, F, &m->proj1);                                  /* g is [F][eff] >= [F][d_model]: reuse it as the projection's output */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)F * D; i++) g[i] = as_gelu(g[i]);
    float *out = (float *)malloc((size_t)F * m->eout * sizeof(float));
    as_lin(out, g, F, &m->proj2);
    free(x); free(n1); free(q); free(k); free(v); free(cx); free(o); free(g);
    *nf = F;
    return out;
}

/* ------------------------------------------------------------------------------------------------ the decoder */

typedef struct { float **k, **v; int cap, len; } AsCache;

static void as_embed_row(const Asr *m, int id, float *dst){
    const uint16_t *r = m->embed + (size_t)id * m->D;
    for (int i = 0; i < m->D; i++) { uint32_t u = (uint32_t)r[i] << 16; memcpy(&dst[i], &u, 4); }
}

/* Runs `n` rows x [n][D] (embeddings, positions pos0..pos0+n-1) through every layer, appending their keys and values to the cache;
 * x is updated in place. For n > 1 the attention is causal within the rows and sees the cache before them. */
static void as_decoder_rows(const Asr *m, AsCache *C, float *x, int n, int pos0, const float *cs, const float *sn){
    const int D = m->D, H = m->H, Hk = m->Hk, hd = m->hd, I = m->inter;
    float *nb = (float *)malloc((size_t)n * D * sizeof(float)), *o = (float *)malloc((size_t)n * D * sizeof(float));
    float *q = (float *)malloc((size_t)n * H * hd * sizeof(float)), *cx = (float *)malloc((size_t)n * H * hd * sizeof(float));
    float *kk = (float *)malloc((size_t)n * Hk * hd * sizeof(float)), *vv = (float *)malloc((size_t)n * Hk * hd * sizeof(float));
    float *g = (float *)malloc((size_t)n * I * sizeof(float)), *u = (float *)malloc((size_t)n * I * sizeof(float));
    float *sc = (float *)malloc((size_t)(C->cap > 1 ? C->cap : 1) * sizeof(float));
    const int group = H / Hk;
    for (int li = 0; li < m->nl; li++) {
        const AcdEncLayer *l = &m->L[li];
        acd_rmsnorm(nb, x, n, D, l->ln1, m->eps);
        acd_lin(q, nb, n, &l->sa.q); acd_lin(kk, nb, n, &l->sa.k); acd_lin(vv, nb, n, &l->sa.v);
        acd_head_norm(q, n, H, hd, l->sa.qn, m->eps);
        acd_head_norm(kk, n, Hk, hd, l->sa.kn, m->eps);
        acd_rope(q, n, H, hd, cs + (size_t)pos0 * hd, sn + (size_t)pos0 * hd);
        acd_rope(kk, n, Hk, hd, cs + (size_t)pos0 * hd, sn + (size_t)pos0 * hd);
        memcpy(C->k[li] + (size_t)pos0 * Hk * hd, kk, (size_t)n * Hk * hd * sizeof(float));
        memcpy(C->v[li] + (size_t)pos0 * Hk * hd, vv, (size_t)n * Hk * hd * sizeof(float));
        const int Tk = pos0 + n;
        if (pos0 == 0 && n > 1) {
            acd_attention_ex(cx, H * hd, q, H * hd, n, C->k[li], C->v[li], Hk * hd, n, H, Hk, hd, -1, 1);
        } else {
            const float scale = 1.0f / sqrtf((float)hd);
            for (int r = 0; r < n; r++) {
                const int upto = pos0 + r + 1;
                for (int h = 0; h < H; h++) {
                    const float *qh = q + (size_t)r * H * hd + (size_t)h * hd;
                    const float *K = C->k[li] + (size_t)(h / group) * hd, *V = C->v[li] + (size_t)(h / group) * hd;
                    float mx = -1e30f;
                    for (int j = 0; j < upto; j++) {
                        const float *kj = K + (size_t)j * Hk * hd;
                        float s = 0;
                        for (int d = 0; d < hd; d++) s += qh[d] * kj[d];
                        sc[j] = s * scale;
                        if (sc[j] > mx) mx = sc[j];
                    }
                    float den = 0;
                    for (int j = 0; j < upto; j++) { sc[j] = expf(sc[j] - mx); den += sc[j]; }
                    float *oh = cx + (size_t)r * H * hd + (size_t)h * hd;
                    for (int d = 0; d < hd; d++) oh[d] = 0;
                    for (int j = 0; j < upto; j++) {
                        const float w = sc[j] / den;
                        const float *vj = V + (size_t)j * Hk * hd;
                        for (int d = 0; d < hd; d++) oh[d] += w * vj[d];
                    }
                }
            }
        }
        (void)Tk;
        acd_lin(o, cx, n, &l->sa.o);
        for (int64_t i = 0; i < (int64_t)n * D; i++) x[i] += o[i];
        acd_rmsnorm(nb, x, n, D, l->ln2, m->eps);
        acd_lin(g, nb, n, &l->gate); acd_lin(u, nb, n, &l->up);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n * I; i++) g[i] = acd_silu(g[i]) * u[i];
        acd_lin(o, g, n, &l->down);
        for (int64_t i = 0; i < (int64_t)n * D; i++) x[i] += o[i];
    }
    C->len = pos0 + n;
    free(nb); free(o); free(q); free(cx); free(kk); free(vv); free(g); free(u); free(sc);
}

static int as_argmax_head(const Asr *m, const float *h, float *logits){
    float *hn = (float *)malloc((size_t)m->D * sizeof(float));
    acd_rmsnorm(hn, h, 1, m->D, m->norm, m->eps);
    acd_lin(logits, hn, 1, &m->head);
    free(hn);
    int best = 0;
    for (int i = 1; i < m->vocab; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

#ifdef COLI_VULKAN
/* ------------------------------------------------------------------------------------------------ the decoder on a Vulkan device
 * The decoder is most of the time (3.6 s of the 4.3 s for 11 s of audio on the CPU: the prompt is a hundred and fifty rows, then each token
 * is one row against every weight, which is memory-bound). On the card the prompt and each token are one submission of the dense chain
 * (norms, q/k/v, head norms, RoPE, attention over a KV cache that lives on the device, o, SwiGLU, down); only the logits come back.
 * The device holds the matrices as bf16 and computes in f32, so a logit differs from the CPU's in the last bits; a device that is
 * missing, or a step that fails, leaves the CPU in charge (the whole request again, as qwen3tts does). */
static struct { int ready, chain; } as_vk;

static void as_vk_open(void){
    memset(&as_vk, 0, sizeof as_vk);
    if (!coli_vk_init_env("qwen3asr")) return;               /* COLI_VULKAN unset, or no device: the CPU */
    as_vk.ready = 1;
    atexit(coli_vk_shutdown);
    const char *c = getenv("COLI_VK_CHAIN");
    if (!(c && !strcmp(c, "0")) && vkc_init() && vkc_ready()) { as_vk.chain = 1; atexit(vkc_shutdown); }   /* registered after the backend's: runs before it */
}

typedef struct {
    int tried, ok;
    ColiVkTensor **t;                       /* [nl][7]: q k v o gate up down */
    ColiVkTensor *head;
    VkcBuf *prm;                            /* per layer: ln1 [D], ln2 [D], q norm [hd], k norm [hd]; then the final norm [D] */
    VkcBuf *cs, *kc, *vc, *xu, *x, *nrm, *q, *k, *v, *ctx, *g, *u, *tmp, *lg, *out, *hid;
    size_t cap, rows;
} AsVk;

static AsVk as_vk_dec;

static ColiVkTensor *as_vk_tensor(const AcdLin *l){
    const QiMat *W = &l->m;
    if (l->b || W->ld || !W->w || (W->fmt != QI_F32 && W->fmt != QI_BF16) || W->K % 2) return NULL;
    ColiVkTensor *t = NULL;
    return coli_vk_tensor_ensure(&t, W->w, NULL, W->fmt == QI_BF16 ? 11 : 10, W->K, W->N, 0) ? t : NULL;
}

/* The matrices and norm weights go up once (the first request; 0 when anything would not). */
static int as_vk_setup(AsVk *v, const Asr *m){
    if (v->tried) return v->ok;
    v->tried = 1;
    if (!as_vk.chain || m->hd > 256 || m->H % m->Hk) return 0;
    v->t = (ColiVkTensor **)calloc((size_t)m->nl * 7, sizeof(ColiVkTensor *));
    const size_t per = (size_t)2 * m->D + 2 * m->hd, total = (size_t)m->nl * per + m->D;
    float *arena = (float *)calloc(total, sizeof(float));
    if (!v->t || !arena) { free(arena); return 0; }
    for (int l = 0; l < m->nl; l++) {
        const AcdEncLayer *L = &m->L[l];
        const AcdLin *mats[7] = { &L->sa.q, &L->sa.k, &L->sa.v, &L->sa.o, &L->gate, &L->up, &L->down };
        for (int i = 0; i < 7; i++) {
            v->t[l * 7 + i] = as_vk_tensor(mats[i]);
            if (!v->t[l * 7 + i]) { fprintf(stderr, "[VK] qwen3asr: a matrix of layer %d did not reach the device; the CPU runs the decoder\n", l); free(arena); return 0; }
        }
        memcpy(arena + l * per, L->ln1, (size_t)m->D * sizeof(float));
        memcpy(arena + l * per + m->D, L->ln2, (size_t)m->D * sizeof(float));
        memcpy(arena + l * per + 2 * m->D, L->sa.qn, (size_t)m->hd * sizeof(float));
        memcpy(arena + l * per + 2 * m->D + m->hd, L->sa.kn, (size_t)m->hd * sizeof(float));
    }
    memcpy(arena + (size_t)m->nl * per, m->norm, (size_t)m->D * sizeof(float));
    v->head = as_vk_tensor(&m->head);
    if (!v->head) { fprintf(stderr, "[VK] qwen3asr: the output matrix did not reach the device; the CPU runs the decoder\n"); free(arena); return 0; }
    v->prm = vkc_buf(total * sizeof(float), VKC_DEV);
    int ok = v->prm && vkc_begin() && vkc_write(v->prm, 0, arena, total * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) return 0;
    v->ok = 1;
    return 1;
}

/* The buffers for `rows` rows a step and `cap` positions of cache; the rope table (cos, sin pairs) goes up with them. */
static int as_vk_reserve(AsVk *v, const Asr *m, size_t rows, size_t cap, const float *cs, const float *sn){
    const size_t D = m->D, rq = (size_t)m->H * m->hd, rk = (size_t)m->Hk * m->hd, hd = m->hd, I = m->inter, nlog = (size_t)m->vocab;
    if (!vkc_reserve(&v->xu, rows * D * 4, VKC_UP) || !vkc_reserve(&v->x, rows * D * 4, VKC_DEV) || !vkc_reserve(&v->nrm, rows * D * 4, VKC_DEV) ||
        !vkc_reserve(&v->tmp, rows * D * 4, VKC_DEV) || !vkc_reserve(&v->q, rows * rq * 4, VKC_DEV) || !vkc_reserve(&v->ctx, rows * rq * 4, VKC_DEV) ||
        !vkc_reserve(&v->k, rows * rk * 4, VKC_DEV) || !vkc_reserve(&v->v, rows * rk * 4, VKC_DEV) || !vkc_reserve(&v->g, rows * I * 4, VKC_DEV) ||
        !vkc_reserve(&v->u, rows * I * 4, VKC_DEV) || !vkc_reserve(&v->lg, nlog * 4, VKC_DEV) || !vkc_reserve(&v->out, nlog * 4, VKC_DOWN) ||
        !vkc_reserve(&v->kc, (size_t)m->nl * cap * rk * 4, VKC_DEV) || !vkc_reserve(&v->vc, (size_t)m->nl * cap * rk * 4, VKC_DEV) ||
        !vkc_reserve(&v->cs, cap * hd * 4, VKC_DEV) || !vkc_reserve(&v->hid, rows * D * 4, VKC_DOWN))
        return 0;
    float *tab = (float *)malloc(cap * hd * sizeof(float));
    if (!tab) return 0;
    for (size_t r = 0; r < cap; r++)
        for (size_t j = 0; j < hd / 2; j++) { tab[r * hd + 2 * j] = cs[r * hd + j]; tab[r * hd + 2 * j + 1] = sn[r * hd + j]; }
    const int ok = vkc_begin() && vkc_write(v->cs, 0, tab, cap * hd * sizeof(float)) && vkc_submit(1);
    free(tab);
    v->cap = cap; v->rows = rows;
    return ok;
}

/* n rows (host, [n][D]) at positions pos0.. through every layer, their keys and values appended to the device cache; then, when `logits` is
 * not NULL, the last row through the final norm and the output matrix. 1 on success; 0 leaves the CPU to run the request. */
static int as_vk_run(AsVk *v, const Asr *m, const float *xrows, int n, int pos0, float *logits, float *all){
    if (!v->ok || (size_t)n > v->rows || (size_t)(pos0 + n) > v->cap) return 0;
    const int D = m->D, H = m->H, Hk = m->Hk, hd = m->hd, I = m->inter;
    const size_t rq = (size_t)H * hd, rk = (size_t)Hk * hd, per = (size_t)2 * D + 2 * hd;
    const float scale = 1.f / sqrtf((float)hd);
    memcpy(vkc_ptr(v->xu), xrows, (size_t)n * D * sizeof(float));
    int ok = vkc_begin() && vkc_copy(v->x, 0, v->xu, 0, (size_t)n * D);
    for (int l = 0; l < m->nl && ok; l++) {
        ColiVkTensor **t = v->t + (size_t)l * 7;
        const int wb = (int)(l * per);
        const size_t kbase = (size_t)l * v->cap * rk;
        VkcNorm n1 = {n, D, 1, 0, D, D, 0, D, D, wb, 0, 0, m->eps, 1.f};
        VkcNorm qn = {n * H, hd, H, 0, H * hd, hd, 0, H * hd, hd, wb + 2 * D, 0, 0, m->eps, 1.f};
        VkcNorm kn = {n * Hk, hd, Hk, 0, Hk * hd, hd, 0, Hk * hd, hd, wb + 2 * D + hd, 0, 0, m->eps, 1.f};
        VkcRope rq_ = {n * H, H, 0, H * hd, hd, hd / 2, pos0 * hd, hd}, rk_ = {n * Hk, Hk, 0, Hk * hd, hd, hd / 2, pos0 * hd, hd};
        ok = vkc_norm(v->x, v->prm, v->nrm, &n1) &&
             vkc_matmul(t[0], v->nrm, 0, v->q, 0, n) && vkc_matmul(t[1], v->nrm, 0, v->k, 0, n) && vkc_matmul(t[2], v->nrm, 0, v->v, 0, n) &&
             vkc_norm(v->q, v->prm, v->q, &qn) && vkc_norm(v->k, v->prm, v->k, &kn) && vkc_rope(v->q, v->cs, &rq_) && vkc_rope(v->k, v->cs, &rk_) &&
             vkc_copy(v->kc, kbase + (size_t)pos0 * rk, v->k, 0, (size_t)n * rk) && vkc_copy(v->vc, kbase + (size_t)pos0 * rk, v->v, 0, (size_t)n * rk);
        if (ok) {
            VkcAttnW a;
            memset(&a, 0, sizeof a);
            a.a = (VkcAttn){.S = n, .H = H, .KVH = Hk, .hd = hd, .pos_base = pos0, .cap = pos0 + n, .q_off = 0, .q_row = (int)rq, .q_seg = hd,
                            .o_off = 0, .o_row = (int)rq, .scale = scale, .k_off = (int)kbase, .v_off = (int)kbase};
            a.vd = hd; a.kv_pm = 1;
            ok = vkc_attn_w(v->q, v->kc, v->vc, v->ctx, NULL, NULL, NULL, &a);
        }
        VkcNorm n2 = {n, D, 1, 0, D, D, 0, D, D, wb + D, 0, 0, m->eps, 1.f};
        VkcEw add = {VKC_EW_ADD, n * D, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f}, act = {VKC_EW_SWIGLU, n * I, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        ok = ok && vkc_matmul(t[3], v->ctx, 0, v->tmp, 0, n) && vkc_ew(v->x, v->x, v->tmp, NULL, NULL, &add) &&
             vkc_norm(v->x, v->prm, v->nrm, &n2) && vkc_matmul(t[4], v->nrm, 0, v->g, 0, n) && vkc_matmul(t[5], v->nrm, 0, v->u, 0, n) &&
             vkc_ew(v->g, v->g, v->u, NULL, NULL, &act) && vkc_matmul(t[6], v->g, 0, v->tmp, 0, n) && vkc_ew(v->x, v->x, v->tmp, NULL, NULL, &add);
    }
    if (ok && logits) {                                                 /* the last row: final norm, then the output matrix */
        VkcNorm fn = {1, D, 1, (n - 1) * D, D, D, 0, D, D, (int)(m->nl * per), 0, 0, m->eps, 1.f};
        ok = vkc_norm(v->x, v->prm, v->nrm, &fn) && vkc_matmul(v->head, v->nrm, 0, v->lg, 0, 1) && vkc_copy(v->out, 0, v->lg, 0, (size_t)m->vocab);
    }
    if (ok && all) ok = vkc_copy(v->hid, 0, v->x, 0, (size_t)n * D);                       /* every row's hidden state, for a caller that wants more than the last */
    ok = ok && vkc_submit(1);
    if (!ok) {
        fprintf(stderr, "[VK] qwen3asr: a step failed; the CPU runs the request\n");
        v->ok = 0;
        if (vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); as_vk.ready = 0; as_vk.chain = 0; }
        return 0;
    }
    if (logits) memcpy(logits, vkc_ptr(v->out), (size_t)m->vocab * sizeof(float));
    if (all) memcpy(all, vkc_ptr(v->hid), (size_t)n * D * sizeof(float));
    return 1;
}
#endif

static const char *as_device(void){
#ifdef COLI_VULKAN
    if (as_vk.ready && as_vk.chain) return "vulkan";
#endif
    return "cpu";
}

/* The encoder's matrix products (the convolutions as im2col GEMMs, the attention projections, the feed-forwards) on the device: the rows go up,
 * the product comes back, and the bias, GELU, norms and the windowed attention stay on the CPU (together a third of the encoder). */
#ifdef COLI_VULKAN
typedef struct { const AcdLin *l; ColiVkTensor *t; } AsLinT;
static AsLinT as_lin_tab[512];
static int as_lin_n, as_lin_off;
static VkcBuf *as_lin_up, *as_lin_in, *as_lin_out, *as_lin_down;

static ColiVkTensor *as_lin_tensor(const AcdLin *l){
    for (int i = 0; i < as_lin_n; i++) if (as_lin_tab[i].l == l) return as_lin_tab[i].t;
    if (as_lin_n >= (int)(sizeof as_lin_tab / sizeof as_lin_tab[0])) return NULL;
    const QiMat *W = &l->m;
    ColiVkTensor *t = NULL;
    if (W->ld || !W->w || (W->fmt != QI_F32 && W->fmt != QI_BF16) || W->K % 2 || !coli_vk_tensor_ensure(&t, W->w, NULL, W->fmt == QI_BF16 ? 11 : 10, W->K, W->N, 0)) t = NULL;
    as_lin_tab[as_lin_n].l = l; as_lin_tab[as_lin_n++].t = t;
    return t;
}

static int as_vk_lin(float *Y, const float *X, int M, const AcdLin *l){
    if (as_lin_off || !as_vk.chain || M < 1 || M > 32768) return 0;
    ColiVkTensor *t = as_lin_tensor(l);
    if (!t) return 0;
    const size_t K = l->m.K, N = l->m.N;
    if (!vkc_reserve(&as_lin_up, (size_t)M * K * 4, VKC_UP) || !vkc_reserve(&as_lin_in, (size_t)M * K * 4, VKC_DEV) ||
        !vkc_reserve(&as_lin_out, (size_t)M * N * 4, VKC_DEV) || !vkc_reserve(&as_lin_down, (size_t)M * N * 4, VKC_DOWN))
        return 0;
    memcpy(vkc_ptr(as_lin_up), X, (size_t)M * K * sizeof(float));
    if (!(vkc_begin() && vkc_copy(as_lin_in, 0, as_lin_up, 0, (size_t)M * K) && vkc_matmul(t, as_lin_in, 0, as_lin_out, 0, M) &&
          vkc_copy(as_lin_down, 0, as_lin_out, 0, (size_t)M * N) && vkc_submit(1))) {
        fprintf(stderr, "[VK] qwen3asr: an encoder product failed; the CPU runs the encoder\n");
        as_lin_off = 1;
        if (vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); as_vk.ready = 0; as_vk.chain = 0; }
        return 0;
    }
    memcpy(Y, vkc_ptr(as_lin_down), (size_t)M * N * sizeof(float));
    if (l->b) {
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < M; r++) for (size_t j = 0; j < N; j++) Y[(size_t)r * N + j] += l->b[j];
    }
    return 1;
}
#endif

static void as_lin(float *Y, const float *X, int M, const AcdLin *l){
#ifdef COLI_VULKAN
    if (as_vk_lin(Y, X, M, l)) return;
#endif
    acd_lin(Y, X, M, l);
}

#ifdef COLI_VULKAN
/* The encoder's layers and its output projections as one submission: LayerNorm, the fused q/k/v product, the windowed attention (chain_enc's
 * attention over a row range: each row sees its window of 104 frames), the output product, the residuals, the feed-forward with GELU, then
 * ln_post, proj1 + GELU, proj2. The three attention matrices are joined into one [3 d][d] matrix on the first use (the attention then reads
 * q, k and v out of one row of 3 d floats). */
typedef struct {
    int tried, ok;
    ColiVkTensor **qkv, **o, **fc1, **fc2, *p1, *p2;
    VkcBuf *prm, *xu, *x, *nrm, *big, *att, *tmp, *g, *rng, *ob, *out;
    int *offs;                              /* per layer: ln1 w, ln1 b, ln2 w, ln2 b, bqkv, bo, b1, b2 (8 entries); then lnp w, lnp b, bp1, bp2 */
} AsVkEnc;
static AsVkEnc as_enc;

static int as_enc_setup(AsVkEnc *e, const Asr *m){
    if (e->tried) return e->ok;
    e->tried = 1;
    const int L = m->el, D = m->ed, FF = m->eff, O = m->eout;
    if (!as_vk.chain || !vkc_enc_ready() || D / m->eh > 128 || D % m->eh) return 0;
    e->qkv = (ColiVkTensor **)calloc((size_t)L, sizeof(ColiVkTensor *)); e->o = (ColiVkTensor **)calloc((size_t)L, sizeof(ColiVkTensor *));
    e->fc1 = (ColiVkTensor **)calloc((size_t)L, sizeof(ColiVkTensor *)); e->fc2 = (ColiVkTensor **)calloc((size_t)L, sizeof(ColiVkTensor *));
    e->offs = (int *)calloc((size_t)L * 8 + 4, sizeof(int));
    const size_t per = (size_t)4 * D + 3 * D + D + FF + D, total = (size_t)L * per + 2 * D + D + O;
    float *arena = (float *)calloc(total, sizeof(float));
    if (!e->qkv || !e->o || !e->fc1 || !e->fc2 || !e->offs || !arena) { free(arena); return 0; }
    for (int l = 0; l < L; l++) {
        const AsEncLayer *y = &m->EL[l];
        const AcdLin *q = &y->q, *k = &y->k, *v = &y->v;
        if (q->m.fmt != k->m.fmt || q->m.fmt != v->m.fmt || q->m.ld || k->m.ld || v->m.ld || !q->b || !k->b || !v->b || !y->o.b || !y->fc1.b || !y->fc2.b ||
            (q->m.fmt != QI_F32 && q->m.fmt != QI_BF16)) { free(arena); return 0; }
        const size_t es = q->m.fmt == QI_BF16 ? 2 : 4, blk = (size_t)D * D * es;
        char *cat = (char *)malloc(3 * blk);                       /* kept for good: the device's tensor is made from it */
        if (!cat) { free(arena); return 0; }
        memcpy(cat, q->m.w, blk); memcpy(cat + blk, k->m.w, blk); memcpy(cat + 2 * blk, v->m.w, blk);
        ColiVkTensor *t = NULL;
        if (!coli_vk_tensor_ensure(&t, cat, NULL, q->m.fmt == QI_BF16 ? 11 : 10, D, 3 * D, 0)) { free(arena); return 0; }
        e->qkv[l] = t;
        e->o[l] = as_lin_tensor(&y->o); e->fc1[l] = as_lin_tensor(&y->fc1); e->fc2[l] = as_lin_tensor(&y->fc2);
        if (!e->o[l] || !e->fc1[l] || !e->fc2[l]) { free(arena); return 0; }
        size_t at = l * per;
        int *of = e->offs + (size_t)l * 8;
        of[0] = (int)at; memcpy(arena + at, y->ln1w, D * sizeof(float)); at += D;
        of[1] = (int)at; memcpy(arena + at, y->ln1b, D * sizeof(float)); at += D;
        of[2] = (int)at; memcpy(arena + at, y->ln2w, D * sizeof(float)); at += D;
        of[3] = (int)at; memcpy(arena + at, y->ln2b, D * sizeof(float)); at += D;
        of[4] = (int)at; memcpy(arena + at, q->b, D * sizeof(float)); memcpy(arena + at + D, k->b, D * sizeof(float)); memcpy(arena + at + 2 * D, v->b, D * sizeof(float)); at += 3 * D;
        of[5] = (int)at; memcpy(arena + at, y->o.b, D * sizeof(float)); at += D;
        of[6] = (int)at; memcpy(arena + at, y->fc1.b, FF * sizeof(float)); at += FF;
        of[7] = (int)at; memcpy(arena + at, y->fc2.b, D * sizeof(float));
    }
    size_t at = (size_t)L * per;
    int *tail = e->offs + (size_t)L * 8;
    tail[0] = (int)at; memcpy(arena + at, m->lnpw, D * sizeof(float)); at += D;
    tail[1] = (int)at; memcpy(arena + at, m->lnpb, D * sizeof(float)); at += D;
    tail[2] = (int)at; memcpy(arena + at, m->proj1.b, D * sizeof(float)); at += D;
    tail[3] = (int)at; memcpy(arena + at, m->proj2.b, O * sizeof(float));
    e->p1 = as_lin_tensor(&m->proj1); e->p2 = as_lin_tensor(&m->proj2);
    if (!e->p1 || !e->p2 || !m->proj1.b || !m->proj2.b) { free(arena); return 0; }
    e->prm = vkc_buf(total * sizeof(float), VKC_DEV);
    int ok = e->prm && vkc_begin() && vkc_write(e->prm, 0, arena, total * sizeof(float)) && vkc_submit(1);
    free(arena);
    e->ok = ok;
    return ok;
}

static int as_enc_lin(ColiVkTensor *t, VkcBuf *x, VkcBuf *y, int F, int N, VkcBuf *prm, int boff, int act){
    VkcEncBias b = {0, F * N, N, 0, N, boff, act};
    return vkc_matmul(t, x, 0, y, 0, F) && vkc_enc_bias(y, prm, &b);
}

static float *as_vk_encode(const Asr *m, const float *x, int F, int win){
    AsVkEnc *e = &as_enc;
    if (as_lin_off || !as_vk.chain || F < 1 || F > 32768 || !as_enc_setup(e, m)) return NULL;
    const int D = m->ed, FF = m->eff, O = m->eout, H = m->eh, hd = D / H;
    if (!vkc_reserve(&e->xu, (size_t)F * D * 4, VKC_UP) || !vkc_reserve(&e->x, (size_t)F * D * 4, VKC_DEV) || !vkc_reserve(&e->nrm, (size_t)F * D * 4, VKC_DEV) ||
        !vkc_reserve(&e->big, (size_t)F * 3 * D * 4, VKC_DEV) || !vkc_reserve(&e->att, (size_t)F * D * 4, VKC_DEV) || !vkc_reserve(&e->tmp, (size_t)F * D * 4, VKC_DEV) ||
        !vkc_reserve(&e->g, (size_t)F * FF * 4, VKC_DEV) || !vkc_reserve(&e->rng, (size_t)F * 2 * 4, VKC_DEV) || !vkc_reserve(&e->ob, (size_t)F * O * 4, VKC_DEV) ||
        !vkc_reserve(&e->out, (size_t)F * O * 4, VKC_DOWN))
        return NULL;
    int *rng = (int *)malloc((size_t)F * 2 * sizeof(int));
    if (!rng) return NULL;
    for (int i = 0; i < F; i++) { const int w0 = i / win * win; rng[2 * i] = w0; rng[2 * i + 1] = (w0 + win < F ? w0 + win : F) - 1; }
    memcpy(vkc_ptr(e->xu), x, (size_t)F * D * sizeof(float));
    const float scale = 1.0f / sqrtf((float)hd);
    int ok = vkc_begin() && vkc_write(e->rng, 0, rng, (size_t)F * 2 * sizeof(int)) && vkc_copy(e->x, 0, e->xu, 0, (size_t)F * D);
    free(rng);
    VkcEw add = {VKC_EW_ADD, F * D, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    for (int l = 0; l < m->el && ok; l++) {
        const int *of = e->offs + (size_t)l * 8;
        VkcEncNorm n1 = {0, F, D, 0, D, 0, D, 0, D, of[0], of[1], 0, 1e-5f}, n2 = {0, F, D, 0, D, 0, D, 0, D, of[2], of[3], 0, 1e-5f};
        VkcEncAttn ap = {0, F, H, hd, 0, D, 2 * D, 3 * D, 0, D, 0, 0, 0, 0, 0, scale, 0};
        ok = vkc_enc_norm(e->x, NULL, e->prm, e->nrm, &n1) && as_enc_lin(e->qkv[l], e->nrm, e->big, F, 3 * D, e->prm, of[4], VKC_ENC_ACT_NONE) &&
             vkc_enc_attn(e->big, NULL, NULL, e->att, e->rng, NULL, NULL, &ap) && as_enc_lin(e->o[l], e->att, e->tmp, F, D, e->prm, of[5], VKC_ENC_ACT_NONE) &&
             vkc_ew(e->x, e->x, e->tmp, NULL, NULL, &add) && vkc_enc_norm(e->x, NULL, e->prm, e->nrm, &n2) &&
             as_enc_lin(e->fc1[l], e->nrm, e->g, F, FF, e->prm, of[6], VKC_ENC_ACT_GELU) && as_enc_lin(e->fc2[l], e->g, e->tmp, F, D, e->prm, of[7], VKC_ENC_ACT_NONE) &&
             vkc_ew(e->x, e->x, e->tmp, NULL, NULL, &add);
    }
    if (ok) {
        const int *tl = e->offs + (size_t)m->el * 8;
        VkcEncNorm np = {0, F, D, 0, D, 0, D, 0, D, tl[0], tl[1], 0, 1e-5f};
        ok = vkc_enc_norm(e->x, NULL, e->prm, e->nrm, &np) && as_enc_lin(e->p1, e->nrm, e->tmp, F, D, e->prm, tl[2], VKC_ENC_ACT_GELU) &&
             as_enc_lin(e->p2, e->tmp, e->ob, F, O, e->prm, tl[3], VKC_ENC_ACT_NONE) && vkc_copy(e->out, 0, e->ob, 0, (size_t)F * O);
    }
    ok = ok && vkc_submit(1);
    if (!ok) {
        fprintf(stderr, "[VK] qwen3asr: the encoder failed on the device; the CPU runs it from here on\n");
        e->ok = 0; as_lin_off = 1;
        if (vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); as_vk.ready = 0; as_vk.chain = 0; }
        return NULL;
    }
    float *out = (float *)malloc((size_t)F * O * sizeof(float));
    if (out) memcpy(out, vkc_ptr(e->out), (size_t)F * O * sizeof(float));
    return out;
}
#endif

/* The prompt's token ids with `nf` audio slots; malloc'd, length in *n. */
static int *as_prompt(const Asr *m_, int nf, const char *context, const char *language, int *n){
    Asr *m = (Asr *)m_;
    char head[4096], tail[512];
    snprintf(head, sizeof head, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n<|audio_start|>", context ? context : "");
    snprintf(tail, sizeof tail, "<|audio_end|><|im_end|>\n<|im_start|>assistant\n%s%s%s", language && *language ? "language " : "",
             language && *language ? language : "", language && *language ? "<asr_text>" : "");
    int cap = (int)(strlen(head) + strlen(tail)) + nf + 64;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    char *nrm = NULL; size_t nl = 0;
    int at = 0;
    if (q38_nfc_normalize(head, strlen(head), &nrm, &nl) != 0) { nrm = strdup(head); nl = strlen(head); }
    at += tok_encode(&m->tok, nrm, (int)nl, ids + at, cap - at);
    free(nrm); nrm = NULL;
    for (int i = 0; i < nf; i++) ids[at++] = m->audio_pad;
    if (q38_nfc_normalize(tail, strlen(tail), &nrm, &nl) != 0) { nrm = strdup(tail); nl = strlen(tail); }
    at += tok_encode(&m->tok, nrm, (int)nl, ids + at, cap - at);
    free(nrm);
    *n = at;
    return ids;
}

/* Greedy transcription of encoder frames `enc` [nf][D]: the generated ids (malloc'd), count in *ng. */
static int as_argmax(const float *logits, int n){
    int best = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

static int *as_generate_on(const Asr *m, const float *enc, int nf, const char *context, const char *language, int max_new, int *ng, int *prompt_len,
                           int **prompt_out, int allow_dev, int *dev_failed){
    int n = 0;
    int *ids = as_prompt(m, nf, context, language, &n);
    const int D = m->D, Hk = m->Hk, hd = m->hd;
    const int cap = n + max_new + 1;
    AsCache C;
    C.cap = cap; C.len = 0;
    C.k = (float **)calloc((size_t)m->nl, sizeof(float *)); C.v = (float **)calloc((size_t)m->nl, sizeof(float *));
    for (int i = 0; i < m->nl; i++) { C.k[i] = (float *)malloc((size_t)cap * Hk * hd * sizeof(float)); C.v[i] = (float *)malloc((size_t)cap * Hk * hd * sizeof(float)); }
    float *cs = (float *)malloc((size_t)cap * hd * sizeof(float)), *sn = (float *)malloc((size_t)cap * hd * sizeof(float));
    acd_rope_table(cs, sn, cap, hd, m->theta);
    float *x = (float *)malloc((size_t)n * D * sizeof(float));
    int slot = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] == m->audio_pad && slot < nf) { memcpy(x + (size_t)i * D, enc + (size_t)slot * D, (size_t)D * sizeof(float)); slot++; }
        else as_embed_row(m, ids[i], x + (size_t)i * D);
    }
    float *logits = (float *)malloc((size_t)m->vocab * sizeof(float));
    int dev = 0;
#ifdef COLI_VULKAN
    if (allow_dev && as_vk.chain && as_vk_setup(&as_vk_dec, m))
        dev = as_vk_reserve(&as_vk_dec, m, (size_t)n, (size_t)cap, cs, sn) && as_vk_run(&as_vk_dec, m, x, n, 0, logits, NULL);
#else
    (void)allow_dev;
#endif
    int tok;
    if (dev) tok = as_argmax(logits, m->vocab);
    else {
        as_decoder_rows(m, &C, x, n, 0, cs, sn);
        tok = as_argmax_head(m, x + (size_t)(n - 1) * D, logits);
    }
    int *gen = (int *)malloc((size_t)(max_new + 1) * sizeof(int));
    int ngen = 0, pos = n;
    float *row = (float *)malloc((size_t)D * sizeof(float));
    while (ngen < max_new) {
        gen[ngen++] = tok;
        if (tok == m->im_end || tok == m->eos2) break;
        as_embed_row(m, tok, row);
#ifdef COLI_VULKAN
        if (dev) {
            if (!as_vk_run(&as_vk_dec, m, row, 1, pos, logits, NULL)) { *dev_failed = 1; break; }
            tok = as_argmax(logits, m->vocab);
            pos++;
            continue;
        }
#endif
        as_decoder_rows(m, &C, row, 1, pos, cs, sn);
        tok = as_argmax_head(m, row, logits);
        pos++;
    }
    *ng = ngen;
    if (prompt_len) *prompt_len = n;
    if (*dev_failed) { free(gen); gen = NULL; }
    if (prompt_out && !*dev_failed) *prompt_out = ids; else free(ids);
    for (int i = 0; i < m->nl; i++) { free(C.k[i]); free(C.v[i]); }
    free(C.k); free(C.v); free(cs); free(sn); free(x); free(logits); free(row);
    return gen;
}

/* On the device when there is one; if it fails part way, the whole generation again on the CPU (its caches were never filled). */
static int *as_generate(const Asr *m, const float *enc, int nf, const char *context, const char *language, int max_new, int *ng, int *prompt_len,
                        int **prompt_out){
    int failed = 0;
    int *gen = as_generate_on(m, enc, nf, context, language, max_new, ng, prompt_len, prompt_out, 1, &failed);
    if (!failed) return gen;
    failed = 0;
    return as_generate_on(m, enc, nf, context, language, max_new, ng, prompt_len, prompt_out, 0, &failed);
}

/* ------------------------------------------------------------------------------------------------ forced alignment
 * Qwen3-ForcedAligner-0.6B is the same network as Qwen3-ASR with a 5000-way head instead of the vocabulary: the audio encoder (24 layers), a causal 28-layer
 * decoder, and no generation. The prompt is <|audio_start|> [audio slots] <|audio_end|> then the words, each followed by two <timestamp> tokens
 * (its start and its end); one pass over all of it, and at each <timestamp> position the head's argmax is a class, times 80 ms. The times are then
 * made monotonic (fix_timestamp of the reference: the longest non-decreasing run stays, the rest is interpolated or snapped to its neighbours). */

#define AS_ALIGN_MAX_S 300.0       /* the reference's limit on the audio of one alignment */

static void as_fix_timestamps(const int *data, int n, int *out){
    int *dp = (int *)malloc((size_t)(n ? n : 1) * sizeof(int)), *par = (int *)malloc((size_t)(n ? n : 1) * sizeof(int));
    char *normal = (char *)calloc((size_t)(n ? n : 1), 1);
    double *res = (double *)malloc((size_t)(n ? n : 1) * sizeof(double));
    for (int i = 0; i < n; i++) { dp[i] = 1; par[i] = -1; res[i] = data[i]; }
    for (int i = 1; i < n; i++)
        for (int j = 0; j < i; j++)
            if (data[j] <= data[i] && dp[j] + 1 > dp[i]) { dp[i] = dp[j] + 1; par[i] = j; }
    int best = 0;
    for (int i = 1; i < n; i++) if (dp[i] > dp[best]) best = i;
    for (int idx = n ? best : -1; idx != -1; idx = par[idx]) normal[idx] = 1;
    for (int i = 0; i < n;) {
        if (normal[i]) { i++; continue; }
        int j = i;
        while (j < n && !normal[j]) j++;
        const int count = j - i;
        int has_l = 0, has_r = 0;
        double lv = 0, rv = 0;
        for (int k = i - 1; k >= 0; k--) if (normal[k]) { lv = res[k]; has_l = 1; break; }
        for (int k = j; k < n; k++) if (normal[k]) { rv = res[k]; has_r = 1; break; }
        for (int k = i; k < j; k++) {
            if (count <= 2) res[k] = !has_l ? rv : !has_r ? lv : ((k - (i - 1)) <= (j - k) ? lv : rv);
            else if (has_l && has_r) res[k] = lv + (rv - lv) / (count + 1) * (k - i + 1);
            else if (has_l) res[k] = lv;
            else if (has_r) res[k] = rv;
        }
        i = j;
    }
    for (int i = 0; i < n; i++) out[i] = (int)res[i];
    free(dp); free(par); free(normal); free(res);
}

typedef struct { int *ids, *classes, *raw, *fixed; float *logits; int nids, nts; double enc_s, dec_s; } AsAligned;

static void as_free_aligned(AsAligned *a){ free(a->ids); free(a->classes); free(a->raw); free(a->fixed); free(a->logits); memset(a, 0, sizeof *a); }

/* Aligns `words` (the units the text was cut into, by the caller) with the audio: out->fixed [2 * nwords] is start, end in milliseconds for each. */
static int as_align(const Asr *m, const float *pcm, int n, const char *const *words, int nw, AsAligned *out, char *err, size_t errn){
    memset(out, 0, sizeof *out);
    if (!m->aligner) { snprintf(err, errn, "this model is not a forced aligner"); return -1; }
    if (nw < 1) { snprintf(err, errn, "there are no words to align"); return -1; }
    if ((double)n / AS_SR > AS_ALIGN_MAX_S) { snprintf(err, errn, "the audio is %.0f s long; the aligner takes up to %.0f s", (double)n / AS_SR, AS_ALIGN_MAX_S); return -1; }
    double t0 = as_now();
    int T = 0;
    float *mel = as_mel(pcm, n, &T);
    if (!mel) { snprintf(err, errn, "the audio is shorter than one frame"); return -1; }
    int nf = 0;
    float *enc = as_encode(m, mel, T, &nf, NULL);
    free(mel);
    out->enc_s = as_now() - t0;
    t0 = as_now();
    size_t tl = 32;
    for (int i = 0; i < nw; i++) tl += strlen(words[i]) + 24;
    char *tail = (char *)malloc(tl);
    size_t at = 0;
    at += (size_t)snprintf(tail + at, tl - at, "<|audio_end|>");
    for (int i = 0; i < nw; i++) at += (size_t)snprintf(tail + at, tl - at, "%s<timestamp><timestamp>", words[i]);
    char *nrm = NULL; size_t nl = 0;
    if (q38_nfc_normalize(tail, at, &nrm, &nl) != 0) { nrm = strdup(tail); nl = at; }
    free(tail);
    int cap = (int)nl + nf + 64;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    int n_ids = 0;
    ids[n_ids++] = m->audio_start;
    for (int i = 0; i < nf; i++) ids[n_ids++] = m->audio_pad;
    n_ids += tok_encode((Tok *)&m->tok, nrm, (int)nl, ids + n_ids, cap - n_ids);
    free(nrm);
    const int D = m->D, Hk = m->Hk, hd = m->hd;
    AsCache C;
    C.cap = n_ids; C.len = 0;
    C.k = (float **)calloc((size_t)m->nl, sizeof(float *)); C.v = (float **)calloc((size_t)m->nl, sizeof(float *));
    for (int i = 0; i < m->nl; i++) { C.k[i] = (float *)malloc((size_t)n_ids * Hk * hd * sizeof(float)); C.v[i] = (float *)malloc((size_t)n_ids * Hk * hd * sizeof(float)); }
    float *cs = (float *)malloc((size_t)n_ids * hd * sizeof(float)), *sn = (float *)malloc((size_t)n_ids * hd * sizeof(float));
    acd_rope_table(cs, sn, n_ids, hd, m->theta);
    float *x = (float *)malloc((size_t)n_ids * D * sizeof(float));
    int slot = 0, nts = 0;
    for (int i = 0; i < n_ids; i++) {
        if (ids[i] == m->audio_pad && slot < nf) { memcpy(x + (size_t)i * D, enc + (size_t)slot * D, (size_t)D * sizeof(float)); slot++; }
        else as_embed_row(m, ids[i], x + (size_t)i * D);
        if (ids[i] == m->ts_id) nts++;
    }
    free(enc);
    int on_card = 0;
#ifdef COLI_VULKAN
    if (as_vk.chain && as_vk_setup(&as_vk_dec, m)) on_card = as_vk_reserve(&as_vk_dec, m, (size_t)n_ids, (size_t)n_ids, cs, sn) && as_vk_run(&as_vk_dec, m, x, n_ids, 0, NULL, x);
#endif
    if (!on_card) as_decoder_rows(m, &C, x, n_ids, 0, cs, sn);
    float *rows = (float *)malloc((size_t)(nts ? nts : 1) * D * sizeof(float));
    int r = 0;
    for (int i = 0; i < n_ids; i++) if (ids[i] == m->ts_id) { memcpy(rows + (size_t)r * D, x + (size_t)i * D, (size_t)D * sizeof(float)); r++; }
    acd_rmsnorm(rows, rows, nts, D, m->norm, m->eps);
    float *logits = (float *)malloc((size_t)(nts ? nts : 1) * m->nhead * sizeof(float));
    acd_lin(logits, rows, nts, &m->head);
    out->classes = (int *)malloc((size_t)(nts ? nts : 1) * sizeof(int));
    out->raw = (int *)malloc((size_t)(nts ? nts : 1) * sizeof(int));
    out->fixed = (int *)malloc((size_t)(nts ? nts : 1) * sizeof(int));
    for (int i = 0; i < nts; i++) {
        const float *l = logits + (size_t)i * m->nhead;
        int best = 0;
        for (int j = 1; j < m->nhead; j++) if (l[j] > l[best]) best = j;
        out->classes[i] = best;
        out->raw[i] = best * m->ts_ms;
    }
    as_fix_timestamps(out->raw, nts, out->fixed);
    out->ids = ids; out->nids = n_ids; out->nts = nts; out->logits = logits;
    out->dec_s = as_now() - t0;
    for (int i = 0; i < m->nl; i++) { free(C.k[i]); free(C.v[i]); }
    free(C.k); free(C.v); free(cs); free(sn); free(x); free(rows);
    if (nts != 2 * nw) { snprintf(err, errn, "internal: %d timestamp positions for %d words", nts, nw); as_free_aligned(out); return -1; }
    return 0;
}

static char **as_read_words(const char *path, int *count){
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char **w = NULL;
    int n = 0, cap = 0;
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        if (!len) continue;
        if (n == cap) { cap = cap ? cap * 2 : 64; w = (char **)realloc(w, (size_t)cap * sizeof(char *)); }
        w[n++] = strdup(line);
    }
    fclose(f);
    *count = n;
    return w;
}

/* ------------------------------------------------------------------------------------------------ the answer */

/* detect_and_fix_repetitions: a character repeated more than `thresh` times is cut to one; so is a pattern of up to 20
 * characters repeated that often. Works on UTF-8 by characters. */
static int as_u8len(unsigned char c){ return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4; }

static char *as_fix_repetitions(const char *s, int thresh){
    const int n = (int)strlen(s);
    int *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int nc = 0;
    for (int i = 0; i < n; i += as_u8len((unsigned char)s[i])) off[nc++] = i;
    off[nc] = n;
    char *a = (char *)malloc((size_t)n + 1);                       /* pass 1: runs of one character */
    int ao = 0;
    for (int i = 0; i < nc;) {
        int cnt = 1;
        const int l = off[i + 1] - off[i];
        while (i + cnt < nc && off[i + cnt + 1] - off[i + cnt] == l && !memcmp(s + off[i + cnt], s + off[i], (size_t)l)) cnt++;
        if (cnt > thresh) { memcpy(a + ao, s + off[i], (size_t)l); ao += l; }
        else { memcpy(a + ao, s + off[i], (size_t)(off[i + cnt] - off[i])); ao += off[i + cnt] - off[i]; }
        i += cnt;
    }
    a[ao] = 0;
    free(off);
    /* pass 2: a pattern of k characters repeated at least `thresh` times */
    const int n2 = ao;
    int *o2 = (int *)malloc((size_t)(n2 + 1) * sizeof(int));
    int c2 = 0;
    for (int i = 0; i < n2; i += as_u8len((unsigned char)a[i])) o2[c2++] = i;
    o2[c2] = n2;
    char *res = (char *)malloc((size_t)n2 + 1);
    int ro = 0, i = 0;
    while (i <= c2 - thresh * 2) {
        int found = 0;
        for (int k = 1; k <= 20 && !found; k++) {
            if (i + k * thresh > c2) break;
            const int plen = o2[i + k] - o2[i];
            int valid = 1;
            for (int rep = 1; rep < thresh && valid; rep++) {
                const int st = i + rep * k;
                if (o2[st + k] - o2[st] != plen || memcmp(a + o2[st], a + o2[i], (size_t)plen)) valid = 0;
            }
            if (valid) {
                int end = i + thresh * k;
                while (end + k <= c2 && o2[end + k] - o2[end] == plen && !memcmp(a + o2[end], a + o2[i], (size_t)plen)) end += k;
                memcpy(res + ro, a + o2[i], (size_t)plen); ro += plen;
                i = end;
                found = 1;
            }
        }
        if (!found) { memcpy(res + ro, a + o2[i], (size_t)(o2[i + 1] - o2[i])); ro += o2[i + 1] - o2[i]; i++; }
    }
    memcpy(res + ro, a + o2[i < c2 ? i : c2], (size_t)(n2 - o2[i < c2 ? i : c2])); ro += n2 - o2[i < c2 ? i : c2];
    res[ro] = 0;
    free(a); free(o2);
    return res;
}

/* parse_asr_output: "language English<asr_text>text" -> language and text (both malloc'd). A forced language makes the whole
 * answer the text. */
static void as_parse(const char *raw_in, const char *forced, char **lang, char **text){
    char *tmp = strdup(raw_in);
    char *s = tmp;
    while (*s == ' ' || *s == '\n' || *s == '\t' || *s == '\r') s++;
    int len = (int)strlen(s);
    while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\n' || s[len - 1] == '\t' || s[len - 1] == '\r')) s[--len] = 0;
    if (!*s) { *lang = strdup(""); *text = strdup(""); free(tmp); return; }
    char *fixed = as_fix_repetitions(s, 20);
    free(tmp);
    if (forced && *forced) { *lang = strdup(forced); *text = fixed; return; }
    char *tag = strstr(fixed, "<asr_text>");
    if (!tag) { *lang = strdup(""); *text = fixed; return; }
    *tag = 0;
    char *body = tag + strlen("<asr_text>");
    while (*body == ' ' || *body == '\n') body++;
    int bl = (int)strlen(body);
    while (bl > 0 && (body[bl - 1] == ' ' || body[bl - 1] == '\n')) body[--bl] = 0;
    char *meta = fixed;
    char low[256];
    snprintf(low, sizeof low, "%s", meta);
    for (char *p = low; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;
    if (strstr(low, "language none")) { *lang = strdup(""); *text = strdup(body); free(fixed); return; }
    char name[128] = "";
    for (char *line = strtok(meta, "\n"); line; line = strtok(NULL, "\n")) {
        while (*line == ' ') line++;
        if (!strncasecmp(line, "language ", 9)) {
            snprintf(name, sizeof name, "%s", line + 9);
            for (int i = (int)strlen(name) - 1; i >= 0 && name[i] == ' '; i--) name[i] = 0;
            break;
        }
    }
    *lang = strdup(name);
    *text = strdup(body);
    free(fixed);
}

/* ------------------------------------------------------------------------------------------------ audio in, the command line */

static float *as_read_wav(const char *path, int *n, char *err, size_t errn){
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errn, "cannot open %s", path); return NULL; }
    unsigned char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) { fclose(f); snprintf(err, errn, "%s is not a WAV file", path); return NULL; }
    int fmt = 0, ch = 0, rate = 0, bits = 0;
    unsigned char *data = NULL; uint32_t dsz = 0;
    for (;;) {
        unsigned char ck[8];
        if (fread(ck, 1, 8, f) != 8) break;
        uint32_t sz = ck[4] | (ck[5] << 8) | (ck[6] << 16) | ((uint32_t)ck[7] << 24);
        if (!memcmp(ck, "fmt ", 4)) {
            unsigned char b[16];
            if (sz < 16 || fread(b, 1, 16, f) != 16) break;
            fmt = b[0] | (b[1] << 8); ch = b[2] | (b[3] << 8); rate = b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24); bits = b[14] | (b[15] << 8);
            if (sz > 16) fseek(f, (long)(sz - 16) + (sz & 1), SEEK_CUR);
        } else if (!memcmp(ck, "data", 4)) {
            data = (unsigned char *)malloc(sz ? sz : 1);
            dsz = (uint32_t)fread(data, 1, sz, f);
            break;
        } else fseek(f, (long)sz + (sz & 1), SEEK_CUR);
    }
    fclose(f);
    if (!data || !ch || !rate) { free(data); snprintf(err, errn, "%s has no usable fmt and data chunks", path); return NULL; }
    if (rate != AS_SR) { free(data); snprintf(err, errn, "%s is %d Hz: the engine reads 16000 Hz (resample first)", path, rate); return NULL; }
    if (!((fmt == 1 && (bits == 16 || bits == 8)) || (fmt == 3 && bits == 32))) { free(data); snprintf(err, errn, "%s: only 16-bit PCM and 32-bit float WAV are read here", path); return NULL; }
    const int bps = bits / 8, frames = (int)(dsz / (uint32_t)(bps * ch));
    float *w = (float *)malloc((size_t)(frames ? frames : 1) * sizeof(float));
    for (int i = 0; i < frames; i++) {
        double s = 0;
        for (int c = 0; c < ch; c++) {
            const unsigned char *p = data + ((size_t)i * ch + c) * bps;
            if (fmt == 3) { float v; memcpy(&v, p, 4); s += v; }
            else if (bits == 16) s += (int16_t)(p[0] | (p[1] << 8)) / 32768.0;
            else s += ((int)p[0] - 128) / 128.0;
        }
        w[i] = (float)(s / ch);
    }
    free(data);
    *n = frames;
    return w;
}

static float *as_load_f32(const char *path, size_t want, size_t *got){
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    float *b = (float *)malloc((size_t)sz);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f);
    *got = (size_t)sz / 4; (void)want;
    return b;
}
static int *as_load_i32(const char *path, size_t *got){
    return (int *)as_load_f32(path, 0, got);
}

static void as_compare(const char *label, const float *a, const float *b, size_t n){
    double mx = 0, sum = 0, ref = 0;
    for (size_t i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > mx) mx = d; sum += d; ref += fabs(b[i]); }
    printf("%-10s n=%zu  max abs diff %.3e  mean abs diff %.3e  (mean |ref| %.3e)\n", label, n, mx, sum / (double)(n ? n : 1), ref / (double)(n ? n : 1));
}

/* ------------------------------------------------------------------------------------------------ long audio, one request */

#define AS_MAX_PIECE_S 1200.0       /* the reference takes up to 20 minutes in one piece (QWEN3ASR_MAX_PIECE_S: a smaller limit, for tests) */
static double as_max_piece(void){ const char *e = getenv("QWEN3ASR_MAX_PIECE_S"); const double v = e ? atof(e) : 0; return v >= 2.0 ? v : AS_MAX_PIECE_S; }
#define AS_MIN_PIECE_S 0.5

/* split_audio_into_chunks: pieces close to AS_MAX_PIECE_S, cut at the quietest 100 ms within 5 s either side of the limit. Returns the number of
 * pieces; start[i] and len[i] are in samples (the pieces tile the audio exactly; a short last piece is padded by the caller). */
static int as_split(const float *w, int n, int **start, int **len){
    const double lim = as_max_piece();
    const int maxlen = (int)(lim * AS_SR), expand = (lim / 2 < 5.0 ? lim / 2 : 5.0) * AS_SR, win = AS_SR / 10;
    int cap = n / maxlen + 2, np = 0;
    *start = (int *)malloc((size_t)cap * sizeof(int)); *len = (int *)malloc((size_t)cap * sizeof(int));
    int at = 0;
    while (n - at > maxlen) {
        const int cut = at + maxlen, left = cut - expand > at ? cut - expand : at, right = cut + expand < n ? cut + expand : n;
        int boundary = cut;
        if (right - left > win) {
            double best = 1e300, run = 0;
            int bestpos = 0;
            for (int i = 0; i < win; i++) run += fabsf(w[left + i]);
            best = run;
            for (int p = 1; p + win <= right - left; p++) {
                run += fabsf(w[left + p + win - 1]) - fabsf(w[left + p - 1]);
                if (run < best) { best = run; bestpos = p; }
            }
            int inner = 0;
            float lo = 1e30f;
            for (int i = 0; i < win; i++) { const float a = fabsf(w[left + bestpos + i]); if (a < lo) { lo = a; inner = i; } }
            boundary = left + bestpos + inner;
        }
        if (boundary < at + 1) boundary = at + 1;
        if (boundary > n) boundary = n;
        (*start)[np] = at; (*len)[np] = boundary - at; np++;
        at = boundary;
    }
    (*start)[np] = at; (*len)[np] = n - at; np++;
    return np;
}

typedef struct { char *language, *text; int tokens, pieces, truncated; double audio_s, mel_s, enc_s, gen_s; } AsResult;

static void as_free_result(AsResult *r){ free(r->language); free(r->text); memset(r, 0, sizeof *r); }

/* Transcribes `n` samples at 16 kHz: pieces of up to 20 minutes, each through the encoder and the decoder; the texts are joined and the
 * languages merged (consecutive repeats dropped). Returns 0, or -1 with a message in err. */
static int as_transcribe(const Asr *m, const float *pcm, int n, const char *language, const char *context, int max_new, AsResult *res,
                         char *err, size_t errn){
    memset(res, 0, sizeof *res);
    res->audio_s = (double)n / AS_SR;
    if (n < AS_HOP) { res->language = strdup(""); res->text = strdup(""); return 0; }
    int *st = NULL, *ln = NULL;
    const int np = as_split(pcm, n, &st, &ln);
    res->pieces = np;
    size_t cap = 1024, used = 0;
    char *all = (char *)malloc(cap);
    char langs[512] = "", last[128] = "";
    all[0] = 0;
    for (int pi = 0; pi < np; pi++) {
        const float *w = pcm + st[pi];
        int wn = ln[pi];
        float *padded = NULL;
        if (np > 1 && wn < (int)(AS_MIN_PIECE_S * AS_SR)) {
            const int want = (int)(AS_MIN_PIECE_S * AS_SR);
            padded = (float *)calloc((size_t)want, sizeof(float));
            memcpy(padded, w, (size_t)wn * sizeof(float));
            w = padded; wn = want;
        }
        double t0 = as_now();
        int T = 0;
        float *mel = as_mel(w, wn, &T);
        free(padded);
        if (!mel) continue;
        res->mel_s += as_now() - t0;
        t0 = as_now();
        int nf = 0;
        float *enc = as_encode(m, mel, T, &nf, NULL);
        free(mel);
        res->enc_s += as_now() - t0;
        t0 = as_now();
        int mn = max_new > 0 ? max_new : 256 + (int)((double)wn / AS_SR * 8.0);
        if (mn > 8192) mn = 8192;
        int ng = 0;
        int *gen = as_generate(m, enc, nf, context, language, mn, &ng, NULL, NULL);
        free(enc);
        res->gen_s += as_now() - t0;
        res->tokens += ng;
        if (ng >= mn && gen[ng - 1] != m->im_end && gen[ng - 1] != m->eos2) res->truncated = 1;
        char *raw = (char *)malloc((size_t)ng * 16 + 16);
        int rl = tok_decode((Tok *)&m->tok, gen, ng, raw, ng * 16);
        raw[rl] = 0;
        free(gen);
        char *p;
        while ((p = strstr(raw, "<|im_end|>"))) memmove(p, p + 10, strlen(p + 10) + 1);
        while ((p = strstr(raw, "<|endoftext|>"))) memmove(p, p + 13, strlen(p + 13) + 1);
        char *lang = NULL, *text = NULL;
        as_parse(raw, language, &lang, &text);
        free(raw);
        const size_t need = used + strlen(text) + 1;
        if (need + 1 > cap) { while (need + 1 > cap) cap *= 2; all = (char *)realloc(all, cap); }
        /* the reference joins the pieces with nothing; a word that ends one piece and one that starts the next need a space */
        if (used > 0 && *text && (unsigned char)all[used - 1] < 0x80 && (unsigned char)*text < 0x80 && all[used - 1] != ' ' && *text != ' ' &&
            all[used - 1] != '\n') {
            if (used + 2 > cap) { cap *= 2; all = (char *)realloc(all, cap); }
            all[used++] = ' '; all[used] = 0;
        }
        memcpy(all + used, text, strlen(text)); used += strlen(text); all[used] = 0;
        if (*lang && strcmp(lang, last)) {
            if (*langs && strlen(langs) + strlen(lang) + 2 < sizeof langs) strcat(langs, ",");
            if (strlen(langs) + strlen(lang) + 1 < sizeof langs) strcat(langs, lang);
            snprintf(last, sizeof last, "%s", lang);
        }
        free(lang); free(text);
    }
    free(st); free(ln);
    res->language = strdup(langs);
    res->text = all;
    (void)err; (void)errn;
    return 0;
}

/* ------------------------------------------------------------------------------------------------ the serve loop */

#define AS_MAX_LINE ((size_t)8 << 20)

static void as_escape(const char *s){
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

static void as_error(const char *id, const char *message){
    printf("ERROR {\"id\":\""); as_escape(id ? id : "");
    printf("\",\"message\":\""); as_escape(message); printf("\"}\n");
    fflush(stdout);
}

static char *as_read_line(int *too_long){
    size_t cap = 1 << 12, n = 0;
    char *buf = (char *)malloc(cap);
    int c, any = 0;
    *too_long = 0;
    if (!buf) exit(1);
    while ((c = fgetc(stdin)) != EOF) {
        any = 1;
        if (c == '\n') break;
        if (*too_long) continue;
        if (n + 2 > cap) {
            if (cap >= AS_MAX_LINE) { *too_long = 1; continue; }
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

/*   READY {"model","task":"transcribe","sample_rate":16000,"max_seconds":..,"layers":..}
 *   TRANSCRIBE {"id","wav":PATH,"language":NAME|"","context":TEXT,"max_new":N}
 *   TEXT {"id","language","text","tokens","pieces","audio_seconds","seconds","truncated"}      ERROR {"id","message"} */
static void as_serve_align(const Asr *m, const char *body){
    jval *j = json_parse_checked(body);
    if (!j || j->t != J_OBJ) { as_error("", "ALIGN needs a JSON object"); if (j) json_free(j); return; }
    jval *jid = json_get(j, "id"), *jw = json_get(j, "wav"), *jwords = json_get(j, "words");
    const char *id = jid && jid->t == J_STR ? jid->str : "";
    char err[512] = "";
    if (!jw || jw->t != J_STR) as_error(id, "wav must be the path of a 16 kHz WAV file");
    else if (!jwords || jwords->t != J_ARR || jwords->len < 1) as_error(id, "words must be a list of the words to align");
    else {
        int ok = 1;
        for (int i = 0; i < jwords->len; i++) if (jwords->kids[i]->t != J_STR || !*jwords->kids[i]->str) ok = 0;
        int n = 0;
        float *pcm = NULL;
        if (!ok) as_error(id, "every word must be a non-empty string");
        else if (!(pcm = as_read_wav(jw->str, &n, err, sizeof err))) as_error(id, err);
        else {
            const double t0 = as_now();
            const char **words = (const char **)malloc((size_t)jwords->len * sizeof(char *));
            for (int i = 0; i < jwords->len; i++) words[i] = jwords->kids[i]->str;
            AsAligned al;
            if (as_align(m, pcm, n, words, jwords->len, &al, err, sizeof err)) as_error(id, err);
            else {
                printf("ALIGNED {\"id\":\""); as_escape(id);
                printf("\",\"starts\":[");
                for (int i = 0; i < jwords->len; i++) printf("%s%d", i ? "," : "", al.fixed[2 * i]);
                printf("],\"ends\":[");
                for (int i = 0; i < jwords->len; i++) printf("%s%d", i ? "," : "", al.fixed[2 * i + 1]);
                printf("],\"audio_seconds\":%.3f,\"seconds\":%.3f}\n", (double)n / AS_SR, as_now() - t0);
                fflush(stdout);
                as_free_aligned(&al);
            }
            free(words);
        }
        free(pcm);
    }
    json_free(j);
}

static int as_serve(const Asr *m, const char *name){
    printf("READY {\"model\":\""); as_escape(name);
    printf("\",\"task\":\"%s\",\"sample_rate\":%d,\"piece_seconds\":%d,\"layers\":%d,\"encoder_layers\":%d,\"device\":\"%s\"}\n", m->aligner ? "align" : "transcribe", AS_SR, m->aligner ? (int)AS_ALIGN_MAX_S : (int)as_max_piece(), m->nl, m->el, as_device());
    fflush(stdout);
    for (;;) {
        int too_long;
        char *line = as_read_line(&too_long);
        if (!line) break;
        if (too_long) { as_error("", "the request line is too long"); free(line); continue; }
        if (!*line) { free(line); continue; }
        if (!strncmp(line, "ALIGN ", 6)) {
            if (m->aligner) as_serve_align(m, line + 6); else as_error("", "this model transcribes: it does not align (ALIGN is for a forced aligner)");
            free(line);
            continue;
        }
        if (m->aligner && !strncmp(line, "TRANSCRIBE ", 11)) { as_error("", "this model is a forced aligner: it does not transcribe (it takes ALIGN)"); free(line); continue; }
        if (strncmp(line, "TRANSCRIBE ", 11)) { as_error("", "unknown message"); free(line); continue; }
        jval *j = json_parse_checked(line + 11);
        if (!j || j->t != J_OBJ) { as_error("", "TRANSCRIBE needs a JSON object"); if (j) json_free(j); free(line); continue; }
        jval *jid = json_get(j, "id"), *jw = json_get(j, "wav"), *jl = json_get(j, "language"), *jc = json_get(j, "context"), *jm = json_get(j, "max_new");
        const char *id = jid && jid->t == J_STR ? jid->str : "";
        char err[512] = "";
        int n = 0;
        float *pcm = NULL;
        if (!jw || jw->t != J_STR) as_error(id, "wav must be the path of a 16 kHz WAV file");
        else if (!(pcm = as_read_wav(jw->str, &n, err, sizeof err))) as_error(id, err);
        else {
            AsResult r;
            const double t0 = as_now();
            if (as_transcribe(m, pcm, n, jl && jl->t == J_STR ? jl->str : NULL, jc && jc->t == J_STR ? jc->str : "", jm && jm->t == J_NUM ? (int)jm->num : 0, &r, err, sizeof err)) as_error(id, err);
            else {
                printf("TEXT {\"id\":\""); as_escape(id);
                printf("\",\"language\":\""); as_escape(r.language);
                printf("\",\"text\":\""); as_escape(r.text);
                printf("\",\"tokens\":%d,\"pieces\":%d,\"audio_seconds\":%.3f,\"seconds\":%.3f,\"truncated\":%s}\n", r.tokens, r.pieces, r.audio_s, as_now() - t0, r.truncated ? "true" : "false");
                fflush(stdout);
                fprintf(stderr, "qwen3asr: %.1f s of audio: mel %.2f s, encoder %.2f s, %d tokens in %.2f s\n", r.audio_s, r.mel_s, r.enc_s, r.tokens, r.gen_s);
            }
            as_free_result(&r);
        }
        free(pcm);
        json_free(j);
        free(line);
    }
    return 0;
}

/* ------------------------------------------------------------------------------------------------ the command line */

static const char *as_name_of(const char *dir){
    const char *s = strrchr(dir, '/'), *b = strrchr(dir, '\\');
    if (b && (!s || b > s)) s = b;
    return s ? s + 1 : dir;
}

static const char *USAGE = "usage: qwen3asr --model DIR (--wav FILE [--language NAME] [--context TEXT] [--max-new N] [--ref REFDIR] | --serve) [--threads N]\n";

int main(int argc, char **argv){
    const char *dir = NULL, *wav = NULL, *language = NULL, *context = "", *ref = NULL, *words_file = NULL;
    int max_new = 0, serve = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--wav") && i + 1 < argc) wav = argv[++i];
        else if (!strcmp(argv[i], "--language") && i + 1 < argc) language = argv[++i];
        else if (!strcmp(argv[i], "--context") && i + 1 < argc) context = argv[++i];
        else if (!strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--words-file") && i + 1 < argc) words_file = argv[++i];
        else if (!strcmp(argv[i], "--serve")) serve = 1;
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
#ifdef _OPENMP
            omp_set_num_threads(atoi(argv[++i]));
#else
            ++i;
#endif
        } else { fputs(USAGE, stderr); return 2; }
    }
    if (!dir || (!wav && !serve)) { fputs(USAGE, stderr); return 2; }
    char err[512] = "";
    Asr *m = (Asr *)calloc(1, sizeof *m);
    double t0 = as_now();
    if (as_load(m, dir, err, sizeof err)) {
        if (serve) { as_error("", err); return 1; }
        fprintf(stderr, "qwen3asr: %s\n", err);
        return 1;
    }
    fprintf(stderr, "qwen3asr: %d encoder layers (d %d), %d decoder layers (d %d), %.2f GB resident, loaded in %.1f s\n", m->el, m->ed, m->nl, m->D, (double)m->wbytes / 1e9, as_now() - t0);
#ifdef COLI_VULKAN
    as_vk_open();                      /* once the weights are loaded: the device opens, and each matrix goes up on the first request */
#endif
    if (serve) return as_serve(m, as_name_of(dir));
    int n = 0;
    float *pcm = as_read_wav(wav, &n, err, sizeof err);
    if (!pcm) { fprintf(stderr, "qwen3asr: %s\n", err); return 1; }
    if (m->aligner) {                                              /* a forced aligner: the words come from a file, one a line */
        int nw = 0;
        char **words = words_file ? as_read_words(words_file, &nw) : NULL;
        if (!words || nw < 1) { fprintf(stderr, "qwen3asr: a forced aligner needs --words-file FILE (one word a line)\n"); return 1; }
        AsAligned al;
        if (as_align(m, pcm, n, (const char *const *)words, nw, &al, err, sizeof err)) { fprintf(stderr, "qwen3asr: %s\n", err); return 1; }
        for (int i = 0; i < nw; i++) printf("%.2f %.2f %s\n", al.fixed[2 * i] / 1000.0, al.fixed[2 * i + 1] / 1000.0, words[i]);
        fprintf(stderr, "qwen3asr: %d words, %.1f s of audio: encoder %.2f s, decoder %.2f s\n", nw, (double)n / AS_SR, al.enc_s, al.dec_s);
        if (ref) {
            char rp[4300];
            size_t got = 0;
            snprintf(rp, sizeof rp, "%s/input_ids.i32", ref);
            int *ri = as_load_i32(rp, &got);
            if (ri) { printf("prompt ids: %s (%d here, %zu in the reference)\n", got == (size_t)al.nids && !memcmp(ri, al.ids, got * 4) ? "identical" : "DIFFERENT", al.nids, got); free(ri); }
            snprintf(rp, sizeof rp, "%s/classes.i32", ref);
            ri = as_load_i32(rp, &got);
            if (ri) {
                int diff = 0;
                for (size_t i = 0; i < got && i < (size_t)al.nts; i++) diff += ri[i] != al.classes[i];
                printf("classes: %d of %d differ (%zu in the reference)\n", diff, al.nts, got);
                free(ri);
            }
            snprintf(rp, sizeof rp, "%s/fixed_ms.i32", ref);
            ri = as_load_i32(rp, &got);
            if (ri) {
                int diff = 0;
                for (size_t i = 0; i < got && i < (size_t)al.nts; i++) diff += ri[i] != al.fixed[i];
                printf("fixed times: %d of %d differ (%zu in the reference)\n", diff, al.nts, got);
                free(ri);
            }
            snprintf(rp, sizeof rp, "%s/logits.f32", ref);
            float *rl = as_load_f32(rp, 0, &got);
            if (rl) { if (got == (size_t)al.nts * m->nhead) as_compare("logits", al.logits, rl, got); else printf("logits: reference has %zu values, here %d\n", got, al.nts * m->nhead); free(rl); }
        }
        return 0;
    }
    if (!ref) {
        AsResult r;
        if (as_transcribe(m, pcm, n, language, context, max_new, &r, err, sizeof err)) { fprintf(stderr, "qwen3asr: %s\n", err); return 1; }
        printf("language: %s\ntext: %s\n", r.language, r.text);
        fprintf(stderr, "qwen3asr: %.1f s of audio in %d piece(s): mel %.2f s, encoder %.2f s, %d tokens in %.2f s%s\n", r.audio_s, r.pieces, r.mel_s, r.enc_s, r.tokens, r.gen_s,
                r.truncated ? " (stopped at the token limit)" : "");
        return 0;
    }
    /* --ref: each stage held to the reference tensors */
    char path[4300];
    size_t got = 0;
    int T = 0;
    float *mel = as_mel(pcm, n, &T);
    if (!mel) { fprintf(stderr, "qwen3asr: the audio is shorter than one frame\n"); return 1; }
    snprintf(path, sizeof path, "%s/features.f32", ref);
    float *r = as_load_f32(path, 0, &got);
    if (r) { if (got == (size_t)AS_NMEL * T) as_compare("features", mel, r, got); else printf("features: reference has %zu values, here %d\n", got, AS_NMEL * T); free(r); }
    int nf = 0;
    float *conv = NULL;
    float *enc = as_encode(m, mel, T, &nf, &conv);
    snprintf(path, sizeof path, "%s/conv.f32", ref);
    r = as_load_f32(path, 0, &got);
    if (r) { if (got == (size_t)nf * m->ed) as_compare("conv", conv, r, got); else printf("conv: reference has %zu values, here %d\n", got, nf * m->ed); free(r); }
    snprintf(path, sizeof path, "%s/encoder.f32", ref);
    r = as_load_f32(path, 0, &got);
    if (r) { if (got == (size_t)nf * m->D) as_compare("encoder", enc, r, got); else printf("encoder: reference has %zu values, here %d\n", got, nf * m->D); free(r); }
    int ng = 0, pl = 0, *prompt = NULL;
    int *gen = as_generate(m, enc, nf, context, language, max_new > 0 ? max_new : 256, &ng, &pl, &prompt);
    snprintf(path, sizeof path, "%s/prompt_ids.i32", ref);
    int *ri = as_load_i32(path, &got);
    if (ri) { int same = got == (size_t)pl && !memcmp(ri, prompt, got * 4); printf("prompt ids: %s (%d here, %zu in the reference)\n", same ? "identical" : "DIFFERENT", pl, got); free(ri); }
    snprintf(path, sizeof path, "%s/generated.i32", ref);
    ri = as_load_i32(path, &got);
    if (ri) {
        int same = got == (size_t)ng && !memcmp(ri, gen, got * 4);
        printf("generated ids: %s (%d here, %zu in the reference)\n", same ? "identical" : "DIFFERENT", ng, got);
        if (!same) { printf(" here:"); for (int i = 0; i < ng && i < 40; i++) printf(" %d", gen[i]); printf("\n ref: "); for (size_t i = 0; i < got && i < 40; i++) printf(" %d", ri[i]); printf("\n"); }
        free(ri);
    }
    return 0;
}
