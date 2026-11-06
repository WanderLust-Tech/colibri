/* gemma4_tok.h -- the Gemma 4 tokenizer (SentencePiece-style BPE with byte fallback).
 *
 * What tokenizer.json says, and what this does about each line:
 *   normalizer     Replace " " -> "▁"            every space becomes the 3-byte mark
 *   pre_tokenizer  Split on " " MergedWithPrevious    nothing left to split after the
 *                                                     normalizer: the text is one word
 *   model          BPE, byte_fallback, no ignore_merges, merges as [left, right] pairs
 *   added_tokens   <|turn> <turn|> <|channel> ... matched literally, never merged
 *   decoder        "▁" -> " ", <0xNN> -> that byte, the rest as written
 *
 * The whole text between two added tokens is one BPE word, which can be long (a prompt is
 * tens of thousands of characters), so the merge loop is the heap one: O(n log n), not the
 * repeated scan for the lowest-ranked pair.
 *
 * A character that is not a vocabulary entry falls back to its UTF-8 bytes, each one the
 * token <0xNN>. tok.h is byte-level (GPT-style) and cannot do this, which is why this is its
 * own header.
 *
 *   G4Tok T; g4_tok_load(&T, "tokenizer.json");
 *   int n = g4_tok_encode(&T, text, len, ids, cap);          // specials in the text are tokens
 *   int k = g4_tok_piece(&T, id, buf, cap);                  // the bytes one token stands for
 */
#ifndef GEMMA4_TOK_H
#define GEMMA4_TOK_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"

typedef struct { const char *k; int klen; int v; } G4Ent;              /* vocabulary hash entry */
typedef struct { uint64_t key; int rank, merged; } G4Merge;            /* (left,right) -> rank, merged id */
typedef struct { const char *s; int len; int id; } G4Special;

typedef struct {
    int n_ids;
    char **piece;                 /* id -> bytes of the token as the vocabulary writes it (NUL-terminated) */
    int *piece_len;
    int *is_special;              /* 1 = an added token: decoded literally, never merged */
    G4Ent *vocab; int vocab_cap;
    G4Merge *merges; int merge_cap;
    int byte_tok[256];            /* id of <0xNN>, -1 when the vocabulary has none */
    G4Special *sp; int nsp;       /* added tokens, longest first */
    int unk, bos, eos;
    jval *root; char *arena;      /* owns the strings the tables point into */
} G4Tok;

static uint64_t g4_fnv(const char *s, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

static void g4_vocab_put(G4Tok *T, const char *k, int klen, int v) {
    uint64_t h = g4_fnv(k, klen) & (uint64_t)(T->vocab_cap - 1);
    while (T->vocab[h].k) {
        if (T->vocab[h].klen == klen && !memcmp(T->vocab[h].k, k, (size_t)klen)) { T->vocab[h].v = v; return; }
        h = (h + 1) & (uint64_t)(T->vocab_cap - 1);
    }
    T->vocab[h].k = k; T->vocab[h].klen = klen; T->vocab[h].v = v;
}

static int g4_vocab_get(const G4Tok *T, const char *k, int klen) {
    uint64_t h = g4_fnv(k, klen) & (uint64_t)(T->vocab_cap - 1);
    while (T->vocab[h].k) {
        if (T->vocab[h].klen == klen && !memcmp(T->vocab[h].k, k, (size_t)klen)) return T->vocab[h].v;
        h = (h + 1) & (uint64_t)(T->vocab_cap - 1);
    }
    return -1;
}

static void g4_merge_put(G4Tok *T, int left, int right, int rank, int merged) {
    uint64_t key = ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    uint64_t h = (key * 0x9E3779B97F4A7C15ULL) >> 20 & (uint64_t)(T->merge_cap - 1);
    while (T->merges[h].rank >= 0) {
        if (T->merges[h].key == key) return;                       /* the first (lowest) rank wins */
        h = (h + 1) & (uint64_t)(T->merge_cap - 1);
    }
    T->merges[h].key = key; T->merges[h].rank = rank; T->merges[h].merged = merged;
}

/* rank and merged id of the pair, or rank -1 */
static int g4_merge_get(const G4Tok *T, int left, int right, int *merged) {
    uint64_t key = ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    uint64_t h = (key * 0x9E3779B97F4A7C15ULL) >> 20 & (uint64_t)(T->merge_cap - 1);
    while (T->merges[h].rank >= 0) {
        if (T->merges[h].key == key) { *merged = T->merges[h].merged; return T->merges[h].rank; }
        h = (h + 1) & (uint64_t)(T->merge_cap - 1);
    }
    return -1;
}

static int g4_cmp_special(const void *a, const void *b) {
    return ((const G4Special *)b)->len - ((const G4Special *)a)->len;
}

static int g4_tok_load(G4Tok *T, const char *path) {
    memset(T, 0, sizeof *T);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gemma4 tokenizer: cannot open %s\n", path); return 0; }
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    char *text = (char *)malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, f) != (size_t)size) { fclose(f); free(text); return 0; }
    text[size] = 0; fclose(f);
    T->root = json_parse(text, &T->arena);
    free(text);                                                       /* strings are copies (j_dup) */
    jval *model = T->root ? json_get(T->root, "model") : NULL;
    jval *vocab = model ? json_get(model, "vocab") : NULL, *merges = model ? json_get(model, "merges") : NULL;
    if (!vocab || vocab->t != J_OBJ || !merges || merges->t != J_ARR) {
        fprintf(stderr, "gemma4 tokenizer: %s has no BPE vocab and merges\n", path); return 0;
    }
    int max_id = 0;
    for (int i = 0; i < vocab->len; i++) if ((int)vocab->kids[i]->num > max_id) max_id = (int)vocab->kids[i]->num;
    jval *added = json_get(T->root, "added_tokens");
    if (added && added->t == J_ARR)
        for (int i = 0; i < added->len; i++) {
            jval *id = json_get(added->kids[i], "id");
            if (id && (int)id->num > max_id) max_id = (int)id->num;
        }
    T->n_ids = max_id + 1;
    T->piece = (char **)calloc((size_t)T->n_ids, sizeof(char *));
    T->piece_len = (int *)calloc((size_t)T->n_ids, sizeof(int));
    T->is_special = (int *)calloc((size_t)T->n_ids, sizeof(int));
    T->vocab_cap = 1; while (T->vocab_cap < 2 * (vocab->len + 64)) T->vocab_cap <<= 1;
    T->vocab = (G4Ent *)calloc((size_t)T->vocab_cap, sizeof(G4Ent));
    for (int i = 0; i < 256; i++) T->byte_tok[i] = -1;
    for (int i = 0; i < vocab->len; i++) {
        const char *k = vocab->keys[i]; int klen = (int)strlen(k), id = (int)vocab->kids[i]->num;
        if (klen == 0 || id < 0 || id >= T->n_ids) continue;
        g4_vocab_put(T, k, klen, id);
        T->piece[id] = (char *)k; T->piece_len[id] = klen;
        unsigned b;
        if (klen == 6 && k[0] == '<' && k[1] == '0' && k[2] == 'x' && k[5] == '>' && sscanf(k + 3, "%2x", &b) == 1)
            T->byte_tok[b] = id;
    }
    if (added && added->t == J_ARR) {
        T->sp = (G4Special *)calloc((size_t)added->len + 1, sizeof(G4Special));
        for (int i = 0; i < added->len; i++) {
            jval *content = json_get(added->kids[i], "content"), *id = json_get(added->kids[i], "id");
            if (!content || !id || content->t != J_STR) continue;
            int idv = (int)id->num;
            if (idv < 0 || idv >= T->n_ids) continue;
            T->piece[idv] = content->str; T->piece_len[idv] = (int)strlen(content->str);
            T->is_special[idv] = 1;
            T->sp[T->nsp++] = (G4Special){content->str, (int)strlen(content->str), idv};
            if (g4_vocab_get(T, content->str, (int)strlen(content->str)) < 0)
                g4_vocab_put(T, content->str, (int)strlen(content->str), idv);
        }
        qsort(T->sp, (size_t)T->nsp, sizeof(G4Special), g4_cmp_special);
    }
    T->merge_cap = 1; while (T->merge_cap < 2 * (merges->len + 64)) T->merge_cap <<= 1;
    T->merges = (G4Merge *)malloc((size_t)T->merge_cap * sizeof(G4Merge));
    for (int i = 0; i < T->merge_cap; i++) T->merges[i].rank = -1;
    char *joined = NULL; size_t jcap = 0;
    for (int i = 0; i < merges->len; i++) {
        jval *pair = merges->kids[i];
        const char *a, *b;
        if (pair->t == J_ARR && pair->len == 2) { a = pair->kids[0]->str; b = pair->kids[1]->str; }
        else if (pair->t == J_STR) {                                   /* the older "left right" form */
            char *sp = strchr(pair->str, ' ');
            if (!sp) continue;
            *sp = 0; a = pair->str; b = sp + 1;
        } else continue;
        int la = (int)strlen(a), lb = (int)strlen(b);
        int ia = g4_vocab_get(T, a, la), ib = g4_vocab_get(T, b, lb);
        if (ia < 0 || ib < 0) continue;
        if ((size_t)(la + lb) + 1 > jcap) { jcap = (size_t)(la + lb) * 2 + 64; joined = (char *)realloc(joined, jcap); }
        memcpy(joined, a, (size_t)la); memcpy(joined + la, b, (size_t)lb);
        int im = g4_vocab_get(T, joined, la + lb);
        if (im < 0) continue;
        g4_merge_put(T, ia, ib, i, im);
    }
    free(joined);
    T->unk = g4_vocab_get(T, "<unk>", 5);
    T->bos = g4_vocab_get(T, "<bos>", 5);
    T->eos = g4_vocab_get(T, "<eos>", 5);
    return 1;
}

static void g4_tok_free(G4Tok *T) {
    free(T->piece); free(T->piece_len); free(T->is_special); free(T->vocab); free(T->merges); free(T->sp);
    if (T->root) json_free(T->root);
    free(T->arena);
    memset(T, 0, sizeof *T);
}

static int g4_tok_id(const G4Tok *T, const char *s) { return g4_vocab_get(T, s, (int)strlen(s)); }

/* ---- the BPE of one word ------------------------------------------------------------- */

typedef struct { int rank, left, right, lid, rid; } G4Cand;            /* a pair that could merge */

static void g4_heap_push(G4Cand *h, int *n, G4Cand c) {
    int i = (*n)++;
    h[i] = c;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].rank < h[i].rank || (h[p].rank == h[i].rank && h[p].left <= h[i].left)) break;
        G4Cand t = h[p]; h[p] = h[i]; h[i] = t; i = p;
    }
}

static G4Cand g4_heap_pop(G4Cand *h, int *n) {
    G4Cand top = h[0];
    h[0] = h[--(*n)];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && (h[l].rank < h[m].rank || (h[l].rank == h[m].rank && h[l].left < h[m].left))) m = l;
        if (r < *n && (h[r].rank < h[m].rank || (h[r].rank == h[m].rank && h[r].left < h[m].left))) m = r;
        if (m == i) break;
        G4Cand t = h[m]; h[m] = h[i]; h[i] = t; i = m;
    }
    return top;
}

/* One word (UTF-8, spaces already turned into the mark) into ids. Returns the count written,
 * stopping at cap. */
static int g4_bpe_word(const G4Tok *T, const char *w, int wlen, int *out, int cap) {
    if (wlen <= 0) return 0;
    int *id = (int *)malloc((size_t)(wlen + 1) * sizeof(int) * 3);
    int *prev = id + (wlen + 1), *next = id + 2 * (wlen + 1);
    int n = 0;
    for (int i = 0; i < wlen;) {
        unsigned char c = (unsigned char)w[i];
        int len = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        if (c >= 0x80 && c < 0xC0) len = 1;                          /* a stray continuation byte */
        if (i + len > wlen) len = wlen - i;
        int t = g4_vocab_get(T, w + i, len);
        if (t >= 0) id[n++] = t;
        else {                                                        /* byte fallback */
            int ok = 1;
            for (int k = 0; k < len; k++) if (T->byte_tok[(unsigned char)w[i + k]] < 0) ok = 0;
            if (ok) for (int k = 0; k < len; k++) id[n++] = T->byte_tok[(unsigned char)w[i + k]];
            else id[n++] = T->unk;
        }
        i += len;
    }
    for (int i = 0; i < n; i++) { prev[i] = i - 1; next[i] = i + 1 < n ? i + 1 : -1; }
    int hcap = 2 * n + 8;
    G4Cand *heap = (G4Cand *)malloc((size_t)hcap * sizeof(G4Cand));
    int hn = 0;
    for (int i = 0; i + 1 < n; i++) {
        int merged, r = g4_merge_get(T, id[i], id[i + 1], &merged);
        if (r >= 0) g4_heap_push(heap, &hn, (G4Cand){r, i, i + 1, id[i], id[i + 1]});
    }
    while (hn > 0) {
        G4Cand c = g4_heap_pop(heap, &hn);
        if (id[c.left] != c.lid || next[c.left] != c.right || id[c.right] != c.rid || id[c.left] < 0) continue;   /* stale */
        int merged = -1; g4_merge_get(T, c.lid, c.rid, &merged);
        id[c.left] = merged;
        int after = next[c.right];
        id[c.right] = -1;
        next[c.left] = after;
        if (after >= 0) prev[after] = c.left;
        if (hn + 3 > hcap) { hcap = 2 * (hn + 3); heap = (G4Cand *)realloc(heap, (size_t)hcap * sizeof(G4Cand)); }
        if (prev[c.left] >= 0) {
            int m, r = g4_merge_get(T, id[prev[c.left]], id[c.left], &m);
            if (r >= 0) g4_heap_push(heap, &hn, (G4Cand){r, prev[c.left], c.left, id[prev[c.left]], id[c.left]});
        }
        if (after >= 0) {
            int m, r = g4_merge_get(T, id[c.left], id[after], &m);
            if (r >= 0) g4_heap_push(heap, &hn, (G4Cand){r, c.left, after, id[c.left], id[after]});
        }
    }
    int w_out = 0;
    for (int i = 0; i >= 0 && i < n; i = next[i] < 0 ? n : next[i]) if (id[i] >= 0 && w_out < cap) out[w_out++] = id[i];
    free(id); free(heap);
    return w_out;
}

/* ---- encode / decode --------------------------------------------------------------------- */

/* Text to ids. Added tokens written out in the text (the chat template's <|turn>, <bos>, ...) become
 * their ids; everything between them is BPE. No beginning-of-sequence token is added: the
 * rendered prompt carries its own. Returns the number of ids written (at most cap). */
static int g4_tok_encode(const G4Tok *T, const char *text, int len, int *out, int cap) {
    int n = 0, start = 0;
    char *buf = NULL; size_t bcap = 0;
    for (int i = 0; i <= len; ) {
        int hit = -1;
        if (i < len && (text[i] == '<' || text[i] == '[')) {
            for (int s = 0; s < T->nsp; s++)
                if (i + T->sp[s].len <= len && !memcmp(text + i, T->sp[s].s, (size_t)T->sp[s].len)) { hit = s; break; }
        }
        if (hit < 0 && i < len) { i++; continue; }
        if (i > start) {                                              /* a stretch of ordinary text */
            int seglen = i - start;
            size_t need = (size_t)seglen * 3 + 1;                     /* every byte may become the 3-byte mark */
            if (need > bcap) { bcap = need; buf = (char *)realloc(buf, bcap); }
            int w = 0;
            for (int k = start; k < i; k++) {
                if (text[k] == ' ') { buf[w++] = (char)0xE2; buf[w++] = (char)0x96; buf[w++] = (char)0x81; }
                else buf[w++] = text[k];
            }
            n += g4_bpe_word(T, buf, w, out + n, cap - n);
        }
        if (hit >= 0) {
            if (n < cap) out[n++] = T->sp[hit].id;
            i += T->sp[hit].len; start = i;
        } else break;
    }
    free(buf);
    return n;
}

/* The bytes one token stands for, as the model's reply spells them. Returns the length. */
static int g4_tok_piece(const G4Tok *T, int id, char *out, int cap) {
    if (id < 0 || id >= T->n_ids || !T->piece[id]) return 0;
    const char *p = T->piece[id]; int len = T->piece_len[id], w = 0;
    if (!T->is_special[id] && len == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x' && p[5] == '>') {
        unsigned b;
        if (sscanf(p + 3, "%2x", &b) == 1) { if (cap > 0) out[w++] = (char)b; return w; }
    }
    for (int i = 0; i < len && w < cap; ) {
        if (!T->is_special[id] && i + 2 < len && (unsigned char)p[i] == 0xE2 && (unsigned char)p[i + 1] == 0x96 && (unsigned char)p[i + 2] == 0x81) {
            out[w++] = ' '; i += 3;
        } else out[w++] = p[i++];
    }
    return w;
}

#endif
