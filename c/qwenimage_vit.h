/* qwenimage_vit.h -- Qwen3-VL's vision tower, for the images of a Qwen-Image-2.1 edit.
 *
 * Included by qwenimage.c after attention(): it uses the engine's Lin, loaders and attention.
 * A reference image reaches the text encoder twice. The VAE sees its pixels (qwenimage_vae_enc.h);
 * this tower turns the same picture into one embedding per 32x32 pixels, which take the place of the
 * prompt's <|image_pad|> tokens, plus one "deepstack" feature per tapped block that the language
 * model adds to those tokens' hidden states in its first layers (te_forward_mm).
 *
 * The picture's path, as transformers' Qwen3VLVisionModel walks it:
 *
 *   flatten    RGBA is composited over white for this copy only (the checkpoint was trained so)
 *   patchify   16x16 patches, channels first, in MERGE-BLOCK order: the tokens of one 2x2 block of
 *              patches are adjacent, which is what the merger's reshape relies on. Row-major order
 *              gives a picture that processes without complaint and reads scrambled.
 *   patch_embed  a Conv3d whose kernel is its stride, over a still image repeated to fill its 2
 *              frames: the same as a 2D conv with the two frames' kernels added, and that is what
 *              is stored (the taps are summed once, at load)
 *   + position the 48x48 learned grid, resampled bilinearly (align_corners) to the image's grid, read
 *              out in the same block order
 *   blocks     pre-norm: LayerNorm, fused qkv (+bias), 2-D rotary (half of the head on the row, half
 *              on the column), full attention over the picture, proj; LayerNorm, MLP with the tanh GELU
 *   mergers    the last block's output goes through `merger` (LayerNorm per patch, THEN four adjacent
 *              patches into one vector); a deepstack tap goes through its own merger, which
 *              normalizes AFTER the grouping (use_postshuffle_norm). Both use the exact (erf) GELU,
 *              the blocks' MLP the tanh one.
 *
 * Weights are kept as the checkpoint stores them (bf16), not quantized: the tower is a small part
 * of an edit, and its output goes straight into the language model's input. */
#ifndef QWENIMAGE_VIT_H
#define QWENIMAGE_VIT_H

#define QIVT_MAX_DS 8

typedef struct { Lin qkv, proj, fc1, fc2; float *qkv_b, *proj_b, *fc1_b, *fc2_b, *n1w, *n1b, *n2w, *n2b; } QivtBlock;
typedef struct { Lin fc1, fc2; float *fc1_b, *fc2_b, *nw, *nb; } QivtMerger;
typedef struct {
    int depth, hidden, heads, hd, inter, patch, temporal, merge, in_ch, out_hidden, npos, side;
    int nds, ds_idx[QIVT_MAX_DS];
    int64_t min_px, max_px;        /* the processor's pixel bounds: inside them it does not resize the picture */
    float eps, theta;
    Lin pe; float *pe_b;           /* patch embedding with its frames' kernels added: [hidden][in_ch*patch*patch] */
    float *pos;                    /* [npos][hidden] */
    QivtBlock *B;
    QivtMerger merger, dsm[QIVT_MAX_DS];
    int loaded;
} QiVit;

static void qivt_config(QiVit *v, const char *model){
    char dir[2048]; snprintf(dir, sizeof dir, "%s/text_encoder", model);
    char *arena = NULL;
    jval *root = read_json(dir, "config.json", &arena);
    jval *c = json_get(root, "vision_config");
    memset(v, 0, sizeof *v);
    v->depth = (int)jnum(c, "depth", 0);
    v->hidden = (int)jnum(c, "hidden_size", 0);
    v->heads = (int)jnum(c, "num_heads", 0);
    v->inter = (int)jnum(c, "intermediate_size", 0);
    v->patch = (int)jnum(c, "patch_size", 0);
    int temporal = v->temporal = (int)jnum(c, "temporal_patch_size", 0);
    v->merge = (int)jnum(c, "spatial_merge_size", 0);
    v->in_ch = (int)jnum(c, "in_channels", 3);
    v->out_hidden = (int)jnum(c, "out_hidden_size", 0);
    v->npos = (int)jnum(c, "num_position_embeddings", 0);
    v->eps = 1e-6f; v->theta = 10000.f;            /* Qwen3VLVisionPatchMerger / VisionRotary defaults */
    jval *act = c ? json_get(c, "hidden_act") : NULL;
    int act_ok = act && act->t == J_STR && !strcmp(act->str, "gelu_pytorch_tanh");
    jval *ds = c ? json_get(c, "deepstack_visual_indexes") : NULL;
    int ds_ok = ds && ds->t == J_ARR && ds->len >= 0 && ds->len <= QIVT_MAX_DS;
    if (ds_ok) {
        v->nds = ds->len;
        for (int i = 0; i < v->nds; i++) {
            ds_ok = ds_ok && ds->kids[i]->t == J_NUM;
            v->ds_idx[i] = ds_ok ? (int)ds->kids[i]->num : -1;
            ds_ok = ds_ok && v->ds_idx[i] >= 0 && v->ds_idx[i] < v->depth && (!i || v->ds_idx[i] > v->ds_idx[i - 1]);
        }
    }
    json_free(root); free(arena);
    v->min_px = 65536; v->max_px = 16777216;       /* the Qwen2-VL image processor's size defaults */
    {
        char path[2048]; snprintf(path, sizeof path, "%s/processor/preprocessor_config.json", model);
        char *text = read_file(path), *pa = NULL;
        jval *pj = text ? json_parse(text, &pa) : NULL;
        jval *sz = pj ? json_get(pj, "size") : NULL;
        double lo = jnum(sz, "shortest_edge", 0), hi = jnum(sz, "longest_edge", 0);
        if (lo > 0) v->min_px = (int64_t)lo;
        if (hi > 0) v->max_px = (int64_t)hi;
        if (pj) json_free(pj);
        free(pa); free(text);
    }
    v->hd = v->heads > 0 ? v->hidden / v->heads : 0;
    v->side = (int)lround(sqrt((double)v->npos));
    /* same discipline as te_config: every dimension that sizes a buffer is bounded */
    if (!c || !act_ok || !ds_ok || v->depth <= 0 || v->depth > 256 || v->hidden <= 0 || v->hidden > 8192 ||
        v->heads <= 0 || v->hidden % v->heads || v->hd % 4 || v->inter <= 0 || v->inter > (1 << 16) ||
        v->patch <= 0 || v->patch > 64 || temporal <= 0 || temporal > 8 || v->merge <= 0 || v->merge > 8 ||
        v->in_ch != 3 || v->out_hidden <= 0 || v->out_hidden > 65536 || v->npos <= 0 ||
        v->side * v->side != v->npos || v->npos > (1 << 16)) {
        fprintf(stderr, "[qwenimage] text_encoder/config.json: unsupported vision_config "
                        "(needs gelu_pytorch_tanh, 3 channels, a square position grid, head_dim divisible by 4)\n");
        exit(1);
    }
}

static void qivt_load_merger(shards *S, const char *pre, const QiVit *v, QivtMerger *g){
    char n[256];
    int W = v->hidden * v->merge * v->merge;
#define MN(s) (snprintf(n, sizeof n, "%s.%s", pre, s), n)
    lin_load(S, MN("linear_fc1.weight"), W, W, 16, &g->fc1);
    lin_load(S, MN("linear_fc2.weight"), v->out_hidden, W, 16, &g->fc2);
    g->fc1_b = vec_load(S, MN("linear_fc1.bias"), W);
    g->fc2_b = vec_load(S, MN("linear_fc2.bias"), v->out_hidden);
#undef MN
}

/* the norm's width depends on which kind of merger it is, so it is loaded by the caller that knows */
static void qivt_load_norm(shards *S, const char *pre, int width, float **w, float **b){
    char n[256];
    snprintf(n, sizeof n, "%s.norm.weight", pre); *w = vec_load(S, n, width);
    snprintf(n, sizeof n, "%s.norm.bias", pre);   *b = vec_load(S, n, width);
}

static void qivt_free(QiVit *v){
    if (!v->loaded) return;
    for (int l = 0; l < v->depth; l++) {
        QivtBlock *B = &v->B[l];
        lin_free(&B->qkv); lin_free(&B->proj); lin_free(&B->fc1); lin_free(&B->fc2);
        free(B->qkv_b); free(B->proj_b); free(B->fc1_b); free(B->fc2_b);
        free(B->n1w); free(B->n1b); free(B->n2w); free(B->n2b);
    }
    free(v->B);
    QivtMerger *ms[1 + QIVT_MAX_DS]; int nm = 0;
    ms[nm++] = &v->merger;
    for (int i = 0; i < v->nds; i++) ms[nm++] = &v->dsm[i];
    for (int i = 0; i < nm; i++) {
        lin_free(&ms[i]->fc1); lin_free(&ms[i]->fc2);
        free(ms[i]->fc1_b); free(ms[i]->fc2_b); free(ms[i]->nw); free(ms[i]->nb);
    }
    lin_free(&v->pe); free(v->pe_b); free(v->pos);
    v->loaded = 0; v->B = NULL;
}

static void qivt_load(QiVit *v, const char *model){
    if (v->loaded) return;
    int temporal = v->temporal;
    char dir[2048]; snprintf(dir, sizeof dir, "%s/text_encoder", model);
    shards S; memset(&S, 0, sizeof S);
    st_init(&S, dir);
    double t0 = now_s();
    int H = v->hidden, P = v->patch, C = v->in_ch, u = v->merge * v->merge;
    /* patch embedding: [H][C][T][P][P], the T frames' kernels added into [H][C][P][P] */
    {
        size_t per = (size_t)C * P * P;
        float *raw = fmalloc((size_t)H * temporal * per);
        need_tensor(&S, "model.visual.patch_embed.proj.weight", H, (int64_t)C * temporal * P * P);
        st_read_f32_cap(&S, "model.visual.patch_embed.proj.weight", raw, (int64_t)H * temporal * per, 1);
        float *w = fmalloc((size_t)H * per);
        for (int o = 0; o < H; o++)
            for (int c = 0; c < C; c++)
                for (size_t k = 0; k < (size_t)P * P; k++) {
                    float s = 0.f;
                    for (int t = 0; t < temporal; t++) s += raw[(((size_t)o * C + c) * temporal + t) * P * P + k];
                    w[(size_t)o * per + (size_t)c * P * P + k] = s;
                }
        free(raw);
        memset(&v->pe, 0, sizeof v->pe);
        v->pe.m.fmt = QI_F32; v->pe.m.N = H; v->pe.m.K = (int)per; v->pe.m.w = v->pe.own = w;
        v->pe_b = vec_load(&S, "model.visual.patch_embed.proj.bias", H);
    }
    v->pos = vec_load(&S, "model.visual.pos_embed.weight", v->npos * H);
    v->B = calloc(v->depth, sizeof(QivtBlock));
    size_t bytes = 0;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(qi_load_threads()) reduction(+:bytes)
    for (int l = 0; l < v->depth; l++) {
        QivtBlock *B = &v->B[l];
        char n[256];
#define BN(s) (snprintf(n, sizeof n, "model.visual.blocks.%d.%s", l, s), n)
        lin_load(&S, BN("attn.qkv.weight"), 3 * H, H, 16, &B->qkv);
        lin_load(&S, BN("attn.proj.weight"), H, H, 16, &B->proj);
        lin_load(&S, BN("mlp.linear_fc1.weight"), v->inter, H, 16, &B->fc1);
        lin_load(&S, BN("mlp.linear_fc2.weight"), H, v->inter, 16, &B->fc2);
        B->qkv_b = vec_load(&S, BN("attn.qkv.bias"), 3 * H);
        B->proj_b = vec_load(&S, BN("attn.proj.bias"), H);
        B->fc1_b = vec_load(&S, BN("mlp.linear_fc1.bias"), v->inter);
        B->fc2_b = vec_load(&S, BN("mlp.linear_fc2.bias"), H);
        B->n1w = vec_load(&S, BN("norm1.weight"), H); B->n1b = vec_load(&S, BN("norm1.bias"), H);
        B->n2w = vec_load(&S, BN("norm2.weight"), H); B->n2b = vec_load(&S, BN("norm2.bias"), H);
#undef BN
        bytes += lin_bytes(&B->qkv) + lin_bytes(&B->proj) + lin_bytes(&B->fc1) + lin_bytes(&B->fc2);
    }
    qivt_load_merger(&S, "model.visual.merger", v, &v->merger);
    qivt_load_norm(&S, "model.visual.merger", H, &v->merger.nw, &v->merger.nb);
    bytes += lin_bytes(&v->merger.fc1) + lin_bytes(&v->merger.fc2);
    for (int i = 0; i < v->nds; i++) {
        char pre[96]; snprintf(pre, sizeof pre, "model.visual.deepstack_merger_list.%d", i);
        qivt_load_merger(&S, pre, v, &v->dsm[i]);
        qivt_load_norm(&S, pre, H * u, &v->dsm[i].nw, &v->dsm[i].nb);
        bytes += lin_bytes(&v->dsm[i].fc1) + lin_bytes(&v->dsm[i].fc2);
    }
    st_destroy(&S);
    v->loaded = 1;
    fprintf(stderr, "[qwenimage] vision tower: %d blocks, %d deepstack taps, %.2f GB resident (bf16), %.1f s\n",
            v->depth, v->nds, bytes / 1e9, now_s() - t0);
}

/* LayerNorm with scale and shift (the block norms and the mergers' norm) */
static void layernorm_affine(float *y, const float *x, const float *w, const float *b, int T, int D, float eps){
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float *r = x + (int64_t)t * D; float *o = y + (int64_t)t * D;
        double m = 0; for (int i = 0; i < D; i++) m += r[i];
        m /= D;
        double var = 0; for (int i = 0; i < D; i++) { double d = r[i] - m; var += d * d; }
        var /= D;
        float inv = 1.f / sqrtf((float)var + eps), mf = (float)m;
        for (int i = 0; i < D; i++) o[i] = (r[i] - mf) * inv * w[i] + b[i];
    }
}
static inline float gelu_exact(float x){ return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }

/* x[n][hidden] in merge-block order -> y[n/u][out_hidden]. post: the deepstack form (norm after grouping) */
static void qivt_merge(const QiVit *v, const QivtMerger *g, int post, const float *x, int n, float *y){
    int u = v->merge * v->merge, T = n / u, W = v->hidden * u;
    float *a = fmalloc((size_t)T * W), *mid = fmalloc((size_t)T * W);
    if (post) layernorm_affine(a, x, g->nw, g->nb, T, W, v->eps);
    else layernorm_affine(a, x, g->nw, g->nb, n, v->hidden, v->eps);
    qi_gemm(mid, a, T, &g->fc1.m, g->fc1_b);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)T * W; i++) mid[i] = gelu_exact(mid[i]);
    qi_gemm(y, mid, T, &g->fc2.m, g->fc2_b);
    free(a); free(mid);
}

/* (row, column) of the i-th patch in merge-block order, for a grid gw patches wide */
static inline void qivt_rc(int i, int gw, int m, int *row, int *col){
    int bw = gw / m, in_col = i % m, in_row = (i / m) % m, bc = (i / (m * m)) % bw, br = i / (m * m * bw);
    *row = br * m + in_row; *col = bc * m + in_col;
}
/* the two bilinear taps and weights of position i on an axis of `size` patches over a `side` grid, align_corners */
static void qivt_taps(int i, int size, int side, int *tap, float *wt){
    float scale = size > 1 ? (float)(side - 1) / (float)(size - 1) : 0.f;
    float src = (float)i * scale, fl = floorf(src);
    for (int t = 0; t < 2; t++) {
        int raw = (int)fl + t;
        tap[t] = raw < 0 ? 0 : raw > side - 1 ? side - 1 : raw;
        float w = 1.f - fabsf(src - fl - (float)t);
        wt[t] = w < 0.f ? 0.f : w;
    }
}

/* patches[n][in_ch*patch*patch] (merge-block order, gh x gw patches) -> out[n/u][out_hidden], and, for each
 * deepstack tap, ds[j][n/u][out_hidden]. ds may be NULL when the caller wants only the final tokens. */
static int qivt_forward(const QiVit *v, const float *patches, int gh, int gw, float *out, float *ds){
    int m = v->merge;
    if (gh <= 0 || gw <= 0 || gh % m || gw % m || (int64_t)gh * gw > (1 << 20)) return -1;
    int n = gh * gw, H = v->hidden, hd = v->hd, half = hd / 2, q4 = hd / 4, nh = v->heads, u = m * m;
    float *x = fmalloc((size_t)n * H), *h = fmalloc((size_t)n * H), *att = fmalloc((size_t)n * H);
    float *qkv = fmalloc((size_t)n * 3 * H), *mlp = fmalloc((size_t)n * v->inter);
    float *cs = fmalloc((size_t)n * half), *sn = fmalloc((size_t)n * half);
    qi_gemm(x, patches, n, &v->pe.m, v->pe_b);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        int row, col, ht[2], wt[2]; float hw[2], ww[2];
        qivt_rc(i, gw, m, &row, &col);
        qivt_taps(row, gh, v->side, ht, hw); qivt_taps(col, gw, v->side, wt, ww);
        float *dst = x + (int64_t)i * H;
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++) {
                float w = hw[a] * ww[b];
                if (w == 0.f) continue;
                const float *src = v->pos + (int64_t)(ht[a] * v->side + wt[b]) * H;
                for (int c = 0; c < H; c++) dst[c] += w * src[c];
            }
        /* rotary angle of rotated pair f: the first head_dim/4 pairs follow the row, the next the column */
        for (int f = 0; f < half; f++) {
            float inv = 1.f / powf(v->theta, (float)(2 * (f % q4)) / (float)(hd / 2));
            float ang = (float)(f < q4 ? row : col) * inv;
            cs[(int64_t)i * half + f] = cosf(ang); sn[(int64_t)i * half + f] = sinf(ang);
        }
    }
    for (int l = 0; l < v->depth; l++) {
        const QivtBlock *B = &v->B[l];
        layernorm_affine(h, x, B->n1w, B->n1b, n, H, v->eps);
        qi_gemm(qkv, h, n, &B->qkv.m, B->qkv_b);
        /* per token [3][heads][hd]; rotate_half on q and k */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            for (int part = 0; part < 2; part++)
                for (int hh = 0; hh < nh; hh++) {
                    float *r = qkv + (int64_t)i * 3 * H + (int64_t)part * H + (int64_t)hh * hd;
                    const float *c = cs + (int64_t)i * half, *s = sn + (int64_t)i * half;
                    for (int f = 0; f < half; f++) {
                        float a = r[f], b = r[f + half];
                        r[f] = a * c[f] - b * s[f];
                        r[f + half] = b * c[f] + a * s[f];
                    }
                }
        attention(att, H, qkv, 3 * H, n, qkv + H, qkv + 2 * H, 3 * H, n, nh, nh, hd, -1, NULL);
        qi_gemm(h, att, n, &B->proj.m, B->proj_b);
        for (int64_t i = 0; i < (int64_t)n * H; i++) x[i] += h[i];
        layernorm_affine(h, x, B->n2w, B->n2b, n, H, v->eps);
        qi_gemm(mlp, h, n, &B->fc1.m, B->fc1_b);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)n * v->inter; i++) mlp[i] = gelu_tanh(mlp[i]);
        qi_gemm(h, mlp, n, &B->fc2.m, B->fc2_b);
        for (int64_t i = 0; i < (int64_t)n * H; i++) x[i] += h[i];
        for (int j = 0; ds && j < v->nds; j++)
            if (v->ds_idx[j] == l) qivt_merge(v, &v->dsm[j], 1, x, n, ds + (size_t)j * (n / u) * v->out_hidden);
    }
    qivt_merge(v, &v->merger, 0, x, n, out);
    free(x); free(h); free(att); free(qkv); free(mlp); free(cs); free(sn);
    return n / u;
}

/* RGBA [h][w][4] -> patches[(h/P/m)*(w/P/m)*m*m][in_ch*P*P], in merge-block order. The alpha is composited over
 * white with PIL's integer arithmetic (Image.paste with the alpha as the mask), so the pixels are the ones the
 * pipeline's processor was given; then (v/255 - 0.5) / 0.5. rgb, when not NULL, receives the flattened bytes
 * [h][w][3] (the oracle compares them exactly). h and w must be multiples of patch*merge. */
static int qivt_patchify(const QiVit *v, const uint8_t *rgba, int h, int w, float *patches, uint8_t *rgb){
    int P = v->patch, m = v->merge;
    if (h <= 0 || w <= 0 || h % (P * m) || w % (P * m)) return -1;
    uint8_t *flat = rgb ? rgb : xmalloc((size_t)h * w * 3);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)h * w; i++) {
        int a = rgba[i * 4 + 3];
        for (int c = 0; c < 3; c++) {
            int fg = rgba[i * 4 + c], t = 255 * (255 - a) + fg * a + 128;     /* DIV255(bg*(255-a) + fg*a), bg = 255 */
            flat[i * 3 + c] = (uint8_t)(((t >> 8) + t) >> 8);
        }
    }
    int gh = h / P, gw = w / P, per = v->in_ch * P * P;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < gh * gw; i++) {
        int row, col; qivt_rc(i, gw, m, &row, &col);
        float *dst = patches + (int64_t)i * per;
        for (int c = 0; c < v->in_ch; c++)
            for (int py = 0; py < P; py++)
                for (int px = 0; px < P; px++) {
                    int b = flat[((int64_t)(row * P + py) * w + col * P + px) * 3 + c];
                    dst[(c * P + py) * P + px] = ((float)b * 0.00392156862745098f - 0.5f) / 0.5f;
                }
    }
    if (!rgb) free(flat);
    return gh * gw;
}

#endif /* QWENIMAGE_VIT_H */
