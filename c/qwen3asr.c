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
    int64_t shh[2] = {m->vocab, m->D};
    const char *hn = st_find(&S, "thinker.lm_head.weight") ? "thinker.lm_head.weight" : n;
    if (acd_load_lin(&S, hn, NULL, m->vocab, m->D, 2, shh, NULL, 0, 0, &m->head, &m->wbytes, err, errn)) return -1;
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
    return 0;
}

/* ------------------------------------------------------------------------------------------------ the audio encoder */

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
    acd_lin(a2, col, F2 * W2, &m->c2);
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
    acd_lin(a3, col, F3 * W3, &m->c3);
    free(col);
    for (size_t i = 0; i < (size_t)F3 * W3 * C; i++) a3[i] = as_gelu(a3[i]);
    float *vec = (float *)malloc((size_t)W3 * C * F3 * sizeof(float));
    for (int t = 0; t < W3; t++)
        for (int c = 0; c < C; c++)
            for (int f = 0; f < F3; f++) vec[(size_t)t * C * F3 + (size_t)c * F3 + f] = a3[((size_t)f * W3 + t) * C + c];
    free(a3);
    acd_lin(out, vec, W3, &m->conv_out);
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
    const int WIN = getenv("AS_FULL_ATTN") ? (F > 0 ? F : 1) : 13 * (800 / AS_CHUNK), H = m->eh, hd = D / H;
    float *n1 = (float *)malloc((size_t)F * D * sizeof(float)), *q = (float *)malloc((size_t)F * D * sizeof(float));
    float *k = (float *)malloc((size_t)F * D * sizeof(float)), *v = (float *)malloc((size_t)F * D * sizeof(float));
    float *cx = (float *)malloc((size_t)F * D * sizeof(float)), *o = (float *)malloc((size_t)F * D * sizeof(float));
    float *g = (float *)malloc((size_t)F * m->eff * sizeof(float));
    for (int li = 0; li < m->el; li++) {
        const AsEncLayer *l = &m->EL[li];
        as_layernorm(n1, x, F, D, l->ln1w, l->ln1b);
        acd_lin(q, n1, F, &l->q); acd_lin(k, n1, F, &l->k); acd_lin(v, n1, F, &l->v);
        for (int w0 = 0; w0 < F; w0 += WIN) {
            const int wl = F - w0 < WIN ? F - w0 : WIN;
            acd_attention_ex(cx + (size_t)w0 * D, D, q + (size_t)w0 * D, D, wl, k + (size_t)w0 * D, v + (size_t)w0 * D, D, wl, H, H, hd, -1, 0);
        }
        acd_lin(o, cx, F, &l->o);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)F * D; i++) x[i] += o[i];
        as_layernorm(n1, x, F, D, l->ln2w, l->ln2b);
        acd_lin(g, n1, F, &l->fc1);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)F * m->eff; i++) g[i] = as_gelu(g[i]);
        acd_lin(o, g, F, &l->fc2);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)F * D; i++) x[i] += o[i];
    }
    as_layernorm(n1, x, F, D, m->lnpw, m->lnpb);
    acd_lin(g, n1, F, &m->proj1);                                  /* g is [F][eff] >= [F][d_model]: reuse it as the projection's output */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)F * D; i++) g[i] = as_gelu(g[i]);
    float *out = (float *)malloc((size_t)F * m->eout * sizeof(float));
    acd_lin(out, g, F, &m->proj2);
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
static int *as_generate(const Asr *m, const float *enc, int nf, const char *context, const char *language, int max_new, int *ng, int *prompt_len,
                        int **prompt_out){
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
    as_decoder_rows(m, &C, x, n, 0, cs, sn);
    float *logits = (float *)malloc((size_t)m->vocab * sizeof(float));
    int *gen = (int *)malloc((size_t)(max_new + 1) * sizeof(int));
    int ngen = 0;
    int tok = as_argmax_head(m, x + (size_t)(n - 1) * D, logits);
    float *row = (float *)malloc((size_t)D * sizeof(float));
    while (ngen < max_new) {
        gen[ngen++] = tok;
        if (tok == m->im_end || tok == m->eos2) break;
        as_embed_row(m, tok, row);
        as_decoder_rows(m, &C, row, 1, C.len, cs, sn);
        tok = as_argmax_head(m, row, logits);
    }
    *ng = ngen;
    if (prompt_len) *prompt_len = n;
    if (prompt_out) *prompt_out = ids; else free(ids);
    for (int i = 0; i < m->nl; i++) { free(C.k[i]); free(C.v[i]); }
    free(C.k); free(C.v); free(cs); free(sn); free(x); free(logits); free(row);
    return gen;
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

int main(int argc, char **argv){
    const char *dir = NULL, *wav = NULL, *language = NULL, *context = "", *ref = NULL;
    int max_new = 512;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--wav") && i + 1 < argc) wav = argv[++i];
        else if (!strcmp(argv[i], "--language") && i + 1 < argc) language = argv[++i];
        else if (!strcmp(argv[i], "--context") && i + 1 < argc) context = argv[++i];
        else if (!strcmp(argv[i], "--max-new") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
#ifdef _OPENMP
            omp_set_num_threads(atoi(argv[++i]));
#else
            ++i;
#endif
        } else { fprintf(stderr, "usage: qwen3asr --model DIR --wav FILE [--language NAME] [--context TEXT] [--max-new N] [--ref REFDIR]\n"); return 2; }
    }
    if (!dir || !wav) { fprintf(stderr, "usage: qwen3asr --model DIR --wav FILE [--language NAME] [--context TEXT] [--max-new N] [--ref REFDIR]\n"); return 2; }
    char err[512] = "";
    int n = 0;
    float *pcm = as_read_wav(wav, &n, err, sizeof err);
    if (!pcm) { fprintf(stderr, "qwen3asr: %s\n", err); return 1; }
    Asr *m = (Asr *)calloc(1, sizeof *m);
    double t0 = as_now();
    if (as_load(m, dir, err, sizeof err)) { fprintf(stderr, "qwen3asr: %s\n", err); return 1; }
    fprintf(stderr, "qwen3asr: %d encoder layers (d %d), %d decoder layers (d %d), %.2f GB resident, loaded in %.1f s\n", m->el, m->ed, m->nl, m->D, (double)m->wbytes / 1e9, as_now() - t0);
    char path[4300];
    size_t got = 0;
    t0 = as_now();
    int T = 0;
    float *mel = as_mel(pcm, n, &T);
    if (!mel) { fprintf(stderr, "qwen3asr: the audio is shorter than one frame\n"); return 1; }
    double t_mel = as_now() - t0;
    if (ref) {
        snprintf(path, sizeof path, "%s/features.f32", ref);
        float *r = as_load_f32(path, 0, &got);
        if (r) { if (got == (size_t)AS_NMEL * T) as_compare("features", mel, r, got); else printf("features: reference has %zu values, here %d\n", got, AS_NMEL * T); free(r); }
    }
    t0 = as_now();
    int nf = 0;
    float *conv = NULL;
    float *enc = as_encode(m, mel, T, &nf, ref ? &conv : NULL);
    double t_enc = as_now() - t0;
    if (ref) {
        snprintf(path, sizeof path, "%s/conv.f32", ref);
        float *r = as_load_f32(path, 0, &got);
        if (r) { if (got == (size_t)nf * m->ed) as_compare("conv", conv, r, got); else printf("conv: reference has %zu values, here %d\n", got, nf * m->ed); free(r); }
        snprintf(path, sizeof path, "%s/encoder.f32", ref);
        r = as_load_f32(path, 0, &got);
        if (r) { if (got == (size_t)nf * m->D) as_compare("encoder", enc, r, got); else printf("encoder: reference has %zu values, here %d\n", got, nf * m->D); free(r); }
    }
    t0 = as_now();
    int ng = 0, pl = 0, *prompt = NULL;
    int *gen = as_generate(m, enc, nf, context, language, max_new, &ng, &pl, &prompt);
    double t_gen = as_now() - t0;
    if (ref) {
        snprintf(path, sizeof path, "%s/prompt_ids.i32", ref);
        int *r = as_load_i32(path, &got);
        if (r) { int same = got == (size_t)pl && !memcmp(r, prompt, got * 4); printf("prompt ids: %s (%d here, %zu in the reference)\n", same ? "identical" : "DIFFERENT", pl, got); free(r); }
        snprintf(path, sizeof path, "%s/generated.i32", ref);
        r = as_load_i32(path, &got);
        if (r) {
            int same = got == (size_t)ng && !memcmp(r, gen, got * 4);
            printf("generated ids: %s (%d here, %zu in the reference)\n", same ? "identical" : "DIFFERENT", ng, got);
            if (!same) { printf(" here:"); for (int i = 0; i < ng && i < 40; i++) printf(" %d", gen[i]); printf("\n ref: "); for (size_t i = 0; i < got && i < 40; i++) printf(" %d", r[i]); printf("\n"); }
            free(r);
        }
    }
    char *raw = (char *)malloc((size_t)ng * 16 + 16);
    int rl = tok_decode(&m->tok, gen, ng, raw, ng * 16);
    raw[rl] = 0;
    /* skip_special_tokens: drop <|im_end|> and <|endoftext|> (the added tokens are written literally by tok_decode) */
    char *p;
    while ((p = strstr(raw, "<|im_end|>"))) memmove(p, p + 10, strlen(p + 10) + 1);
    while ((p = strstr(raw, "<|endoftext|>"))) memmove(p, p + 13, strlen(p + 13) + 1);
    char *lang = NULL, *text = NULL;
    as_parse(raw, language, &lang, &text);
    printf("language: %s\ntext: %s\n", lang, text);
    fprintf(stderr, "qwen3asr: %d samples (%.1f s): mel %.2f s, encoder %.2f s (%d frames), %d tokens in %.2f s\n", n, (double)n / AS_SR, t_mel, t_enc, nf, ng, t_gen);
    return 0;
}
