/* acestep_meta.h: the planner's first phase. The 5 Hz language model writes the song's metadata (tempo, a rewritten caption, duration, key, language, time
 * signature) as a YAML block inside <think>, under upstream's constrained decoder (acestep/constrained_logits_processor.py: a finite-state machine that
 * forces the field names, lets the digits through only where a value stays valid, takes the language as the model's top choice, and leaves the caption free
 * but for the audio codes and the backtick), with no guidance. Then phase 2 (acestep_lm.h) writes the codes under a prompt that carries this metadata as
 * the YAML PyYAML writes for it.
 *
 * Everything here is held to upstream by an oracle (tools/acestep_ref_meta.py, acestep --meta-ref, --meta-trees, --yaml-ref): the token tables, every step of the
 * decoder (its state, the tokens it lets through, the model's logits at them, the token chosen), the YAML text, and the metadata parsed from the model's text.
 *
 * Not built: the genres field (upstream skips it in this phase; a model that writes "genres:" after the caption is refused), and the planner's other jobs. */
#ifndef ACESTEP_META_H
#define ACESTEP_META_H

#include "acestep_lm.h"

/* ----------------------------------------------------------------------------------------------------- Python's text */

/* UTF-8 to code points; an invalid sequence (its maximal invalid prefix) is one U+FFFD, as Python's "replace" does. */
static int *ame_cps(const char *s, int len, int *n){
    int *cp = (int *)malloc(((size_t)len + 1) * sizeof(int));
    if (!cp) return NULL;
    int k = 0;
    for (int i = 0; i < len;) {
        const unsigned char c = (unsigned char)s[i];
        if (c < 0x80) { cp[k++] = c; i++; continue; }
        int need, v, lo = 0x80, hi = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) { need = 1; v = c & 0x1F; }
        else if (c >= 0xE0 && c <= 0xEF) { need = 2; v = c & 0x0F; if (c == 0xE0) lo = 0xA0; if (c == 0xED) hi = 0x9F; }
        else if (c >= 0xF0 && c <= 0xF4) { need = 3; v = c & 0x07; if (c == 0xF0) lo = 0x90; if (c == 0xF4) hi = 0x8F; }
        else { cp[k++] = 0xFFFD; i++; continue; }
        int j = 1;
        for (; j <= need; j++) {
            if (i + j >= len) break;
            const unsigned char d = (unsigned char)s[i + j];
            if (d < (j == 1 ? lo : 0x80) || d > (j == 1 ? hi : 0xBF)) break;
            v = (v << 6) | (d & 0x3F);
        }
        if (j <= need) { cp[k++] = 0xFFFD; i += j; continue; }
        cp[k++] = v; i += need + 1;
    }
    *n = k;
    return cp;
}

static int ame_put_utf8(char *o, int c){
    if (c < 0x80) { o[0] = (char)c; return 1; }
    if (c < 0x800) { o[0] = (char)(0xC0 | (c >> 6)); o[1] = (char)(0x80 | (c & 0x3F)); return 2; }
    if (c < 0x10000) { o[0] = (char)(0xE0 | (c >> 12)); o[1] = (char)(0x80 | ((c >> 6) & 0x3F)); o[2] = (char)(0x80 | (c & 0x3F)); return 3; }
    o[0] = (char)(0xF0 | (c >> 18)); o[1] = (char)(0x80 | ((c >> 12) & 0x3F)); o[2] = (char)(0x80 | ((c >> 6) & 0x3F)); o[3] = (char)(0x80 | (c & 0x3F));
    return 4;
}

/* Python's str.isspace on one character. */
static int ame_isspace(int c){
    return (c >= 9 && c <= 13) || (c >= 28 && c <= 32) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

static int ame_isbreak(int c){ return c == '\n' || c == 0x85 || c == 0x2028 || c == 0x2029; }

/* ------------------------------------------------------------------------------------------------------------ YAML */

/* PyYAML's emitter (yaml.dump(d, allow_unicode=True, sort_keys=True): block style, indent 2, width 80), for a mapping of scalars: a key that is a plain
 * word, and a value that is an int or a string. The three scalar writers, the choice of style and the analysis of the string are its own, line by line. */
typedef struct { char *b; size_t n, cap; int col, ws, ind, indent; } AmeY;

static int ame_yadd(AmeY *y, const char *s, size_t n){
    if (y->n + n + 1 > y->cap) {
        size_t c = y->cap ? y->cap * 2 : 256;
        while (c < y->n + n + 1) c *= 2;
        char *nb = (char *)realloc(y->b, c);
        if (!nb) return -1;
        y->b = nb; y->cap = c;
    }
    memcpy(y->b + y->n, s, n); y->n += n; y->b[y->n] = 0;
    return 0;
}
static void ame_yput(AmeY *y, const int *cp, int n){
    char tmp[4];
    for (int i = 0; i < n; i++) { int k = ame_put_utf8(tmp, cp[i]); ame_yadd(y, tmp, (size_t)k); }
    y->col += n;
}
static void ame_yputs(AmeY *y, const char *s){ ame_yadd(y, s, strlen(s)); y->col += (int)strlen(s); }

static void ame_ybreak(AmeY *y, int c){                    /* write_line_break */
    char tmp[4]; int k = ame_put_utf8(tmp, c);
    ame_yadd(y, tmp, (size_t)k);
    y->ws = 1; y->ind = 1; y->col = 0;
}
static void ame_yindent(AmeY *y){                          /* write_indent */
    const int indent = y->indent;
    if (!y->ind || y->col > indent || (y->col == indent && !y->ws)) ame_ybreak(y, '\n');
    if (y->col < indent) {
        y->ws = 1;
        for (int i = y->col; i < indent; i++) ame_yadd(y, " ", 1);
        y->col = indent;
    }
}
static void ame_yindicator(AmeY *y, const char *ind, int need_ws, int ws, int indention){  /* write_indicator */
    if (!(y->ws || !need_ws)) { ame_yadd(y, " ", 1); y->col += 1; }
    ame_yputs(y, ind);
    y->ws = ws;
    y->ind = y->ind && indention;
}

#define AME_WIDTH 80

static void ame_write_plain(AmeY *y, const int *t, int L, int split){
    if (!L) return;
    if (!y->ws) { ame_yadd(y, " ", 1); y->col += 1; }
    y->ws = 0; y->ind = 0;
    int spaces = 0, start = 0, end = 0;
    while (end <= L) {
        const int ch = end < L ? t[end] : -1;
        if (spaces) {
            if (ch != ' ') {
                if (start + 1 == end && y->col > AME_WIDTH && split) { ame_yindent(y); y->ws = 0; y->ind = 0; }
                else ame_yput(y, t + start, end - start);
                start = end;
            }
        } else {                                           /* a plain scalar has no line breaks: the analysis does not allow them */
            if (ch < 0 || ch == ' ') { ame_yput(y, t + start, end - start); start = end; }
        }
        if (ch >= 0) spaces = ch == ' ';
        end++;
    }
}

static void ame_write_single(AmeY *y, const int *t, int L, int split){
    ame_yindicator(y, "'", 1, 0, 0);
    int spaces = 0, breaks = 0, start = 0, end = 0;
    while (end <= L) {
        const int ch = end < L ? t[end] : -1;
        if (spaces) {
            if (ch < 0 || ch != ' ') {
                if (start + 1 == end && y->col > AME_WIDTH && split && start != 0 && end != L) ame_yindent(y);
                else ame_yput(y, t + start, end - start);
                start = end;
            }
        } else if (breaks) {
            if (ch < 0 || !ame_isbreak(ch)) {
                if (t[start] == '\n') ame_ybreak(y, '\n');
                for (int i = start; i < end; i++) ame_ybreak(y, t[i]);
                ame_yindent(y);
                start = end;
            }
        } else {
            if (ch < 0 || ch == ' ' || ame_isbreak(ch) || ch == '\'') {
                if (start < end) { ame_yput(y, t + start, end - start); start = end; }
            }
        }
        if (ch == '\'') { ame_yputs(y, "''"); start = end + 1; }
        if (ch >= 0) { spaces = ch == ' '; breaks = ame_isbreak(ch); }
        end++;
    }
    ame_yindicator(y, "'", 0, 0, 0);
}

static void ame_write_double(AmeY *y, const int *t, int L, int split){
    ame_yindicator(y, "\"", 1, 0, 0);
    int start = 0, end = 0;
    while (end <= L) {
        const int ch = end < L ? t[end] : -1;
        const int plain_range = ch >= 0x20 && ch <= 0x7E, uni = (ch >= 0xA0 && ch <= 0xD7FF) || (ch >= 0xE000 && ch <= 0xFFFD);
        if (ch < 0 || ch == '"' || ch == '\\' || ch == 0x85 || ch == 0x2028 || ch == 0x2029 || ch == 0xFEFF || !(plain_range || uni)) {
            if (start < end) { ame_yput(y, t + start, end - start); start = end; }
            if (ch >= 0) {
                char buf[16];
                const char *rep = NULL;
                switch (ch) {
                    case 0: rep = "0"; break; case 7: rep = "a"; break; case 8: rep = "b"; break; case 9: rep = "t"; break; case 10: rep = "n"; break;
                    case 11: rep = "v"; break; case 12: rep = "f"; break; case 13: rep = "r"; break; case 27: rep = "e"; break; case '"': rep = "\""; break;
                    case '\\': rep = "\\"; break; case 0x85: rep = "N"; break; case 0xA0: rep = "_"; break; case 0x2028: rep = "L"; break; case 0x2029: rep = "P"; break;
                    default: break;
                }
                if (rep) snprintf(buf, sizeof buf, "\\%s", rep);
                else if (ch <= 0xFF) snprintf(buf, sizeof buf, "\\x%02X", ch);
                else if (ch <= 0xFFFF) snprintf(buf, sizeof buf, "\\u%04X", ch);
                else snprintf(buf, sizeof buf, "\\U%08X", ch);
                ame_yputs(y, buf);
                start = end + 1;
            }
        }
        if (end > 0 && end < L - 1 && (ch == ' ' || start >= end) && y->col + (end - start) > AME_WIDTH && split) {
            ame_yput(y, t + start, end - start);
            ame_yputs(y, "\\");
            if (start < end) start = end;
            ame_yindent(y);
            y->ws = 0; y->ind = 0;
            if (t[start] == ' ') ame_yputs(y, "\\");
        }
        end++;
    }
    ame_yindicator(y, "\"", 0, 0, 0);
}

/* The implicit resolvers of PyYAML's default loader (yaml/resolver.py): a plain string that one of them matches would be read back as a bool, number,
 * null, timestamp ..., so it is quoted. Each is the regular expression of that file, by hand; tools/acestep_ref_meta.py's corpus holds them to it. */
static int ame_digit(int c){ return c >= '0' && c <= '9'; }
static int ame_in(const int *t, int n, int i, int c){ return i < n && t[i] == c; }
static int ame_word(const int *t, int n, const char *w){
    int l = (int)strlen(w);
    if (n != l) return 0;
    for (int i = 0; i < l; i++) if (t[i] != (unsigned char)w[i]) return 0;
    return 1;
}
static int ame_any_word(const int *t, int n, const char *const *ws){
    for (int i = 0; ws[i]; i++) if (ame_word(t, n, ws[i])) return 1;
    return 0;
}
/* (?::[0-5]?[0-9])+ from i; the end position, or -1 when there is no group */
static int ame_sexa(const int *t, int n, int i){
    int groups = 0;
    while (i < n && t[i] == ':') {
        int j = i + 1;
        if (j < n && t[j] >= '0' && t[j] <= '5' && j + 1 < n && ame_digit(t[j + 1])) j += 2;
        else if (j < n && ame_digit(t[j])) j += 1;
        else break;
        i = j; groups++;
    }
    return groups ? i : -1;
}
static int ame_run(const int *t, int n, int i, int (*ok)(int)){ while (i < n && ok(t[i])) i++; return i; }
static int ame_dig_us(int c){ return ame_digit(c) || c == '_'; }
static int ame_bin_us(int c){ return c == '0' || c == '1' || c == '_'; }
static int ame_oct_us(int c){ return (c >= '0' && c <= '7') || c == '_'; }
static int ame_hex_us(int c){ return ame_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c == '_'; }

static int ame_is_int(const int *t, int n){
    int i = (n > 0 && (t[0] == '-' || t[0] == '+')) ? 1 : 0;
    if (i >= n) return 0;
    if (t[i] == '0' && i + 1 < n && t[i + 1] == 'b') { int e = ame_run(t, n, i + 2, ame_bin_us); if (e > i + 2 && e == n) return 1; }
    if (t[i] == '0' && i + 1 < n) { int e = ame_run(t, n, i + 1, ame_oct_us); if (e == n) return 1; }
    if (t[i] == '0' && i + 1 == n) return 1;
    if (t[i] >= '1' && t[i] <= '9') {
        int e = ame_run(t, n, i + 1, ame_dig_us);
        if (e == n) return 1;
        int s = ame_sexa(t, n, e);
        if (s == n) return 1;
    }
    if (t[i] == '0' && i + 1 < n && t[i + 1] == 'x') { int e = ame_run(t, n, i + 2, ame_hex_us); if (e > i + 2 && e == n) return 1; }
    return 0;
}

static int ame_exp_tail(const int *t, int n, int i){   /* (?:[eE][-+][0-9]+)? then the end */
    if (i == n) return 1;
    if (t[i] == 'e' || t[i] == 'E') {
        if (i + 1 < n && (t[i + 1] == '-' || t[i + 1] == '+') && i + 2 < n) {
            int j = i + 2;
            while (j < n && ame_digit(t[j])) j++;
            if (j > i + 2 && j == n) return 1;
        }
    }
    return 0;
}

static int ame_is_float(const int *t, int n){
    int i = (n > 0 && (t[0] == '-' || t[0] == '+')) ? 1 : 0;
    if (i < n && ame_digit(t[i])) {
        int e = ame_run(t, n, i + 1, ame_dig_us);
        if (e < n && t[e] == '.') {                                       /* [-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+][0-9]+)? */
            int f = ame_run(t, n, e + 1, ame_dig_us);
            if (ame_exp_tail(t, n, f)) return 1;
        }
        int s = ame_sexa(t, n, e);                                        /* [-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]* */
        if (s > 0 && s < n && t[s] == '.' && ame_run(t, n, s + 1, ame_dig_us) == n) return 1;
    }
    if (i == 0 && n > 1 && t[0] == '.' && ame_digit(t[1])) {              /* \.[0-9][0-9_]*(?:[eE][-+][0-9]+)? */
        int f = ame_run(t, n, 2, ame_dig_us);
        if (ame_exp_tail(t, n, f)) return 1;
    }
    if (i < n && t[i] == '.') {                                           /* [-+]?\.(?:inf|Inf|INF) */
        static const char *const inf[] = { "inf", "Inf", "INF", NULL };
        for (int k = 0; inf[k]; k++) if (ame_word(t + i + 1, n - i - 1, inf[k])) return 1;
    }
    if (i == 0 && n > 1 && t[0] == '.') {                                 /* \.(?:nan|NaN|NAN) */
        static const char *const nan[] = { "nan", "NaN", "NAN", NULL };
        for (int k = 0; nan[k]; k++) if (ame_word(t + 1, n - 1, nan[k])) return 1;
    }
    return 0;
}

static int ame_dd(const int *t, int n, int i, int count){ for (int k = 0; k < count; k++) if (i + k >= n || !ame_digit(t[i + k])) return 0; return 1; }

static int ame_is_timestamp(const int *t, int n){
    if (n == 10 && ame_dd(t, n, 0, 4) && t[4] == '-' && ame_dd(t, n, 5, 2) && t[7] == '-' && ame_dd(t, n, 8, 2)) return 1;
    if (n < 12 || !ame_dd(t, n, 0, 4) || t[4] != '-') return 0;
    int i = 5;
    if (!ame_dd(t, n, i, 1)) return 0;
    i += ame_dd(t, n, i, 2) ? 2 : 1;
    if (!ame_in(t, n, i, '-')) return 0;
    i++;
    if (!ame_dd(t, n, i, 1)) return 0;
    i += ame_dd(t, n, i, 2) ? 2 : 1;
    if (i < n && (t[i] == 'T' || t[i] == 't')) i++;
    else if (i < n && (t[i] == ' ' || t[i] == '\t')) { while (i < n && (t[i] == ' ' || t[i] == '\t')) i++; }
    else return 0;
    if (!ame_dd(t, n, i, 1)) return 0;
    i += ame_dd(t, n, i, 2) ? 2 : 1;
    if (!ame_in(t, n, i, ':') || !ame_dd(t, n, i + 1, 2) || !ame_in(t, n, i + 3, ':') || !ame_dd(t, n, i + 4, 2)) return 0;
    i += 6;
    if (ame_in(t, n, i, '.')) { i++; while (i < n && ame_digit(t[i])) i++; }
    if (i == n) return 1;
    {   /* (?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))? */
        int j = i;
        while (j < n && (t[j] == ' ' || t[j] == '\t')) j++;
        if (j + 1 == n && t[j] == 'Z') return 1;
        if (j < n && (t[j] == '-' || t[j] == '+') && ame_dd(t, n, j + 1, 1)) {
            int k = j + 1 + (ame_dd(t, n, j + 1, 2) ? 2 : 1);
            if (k == n) return 1;
            if (t[k] == ':' && ame_dd(t, n, k + 1, 2) && k + 3 == n) return 1;
        }
    }
    return 0;
}

static int ame_resolves(const int *t, int n){
    static const char *const bools[] = { "yes", "Yes", "YES", "no", "No", "NO", "true", "True", "TRUE", "false", "False", "FALSE", "on", "On", "ON", "off", "Off", "OFF", NULL };
    static const char *const nulls[] = { "~", "null", "Null", "NULL", "", NULL };
    static const char *const marks[] = { "!", "&", "*", "<<", "=", NULL };
    return ame_any_word(t, n, bools) || ame_any_word(t, n, nulls) || ame_any_word(t, n, marks) || ame_is_int(t, n) || ame_is_float(t, n) || ame_is_timestamp(t, n);
}

/* analyze_scalar, for the flags the style choice reads. */
typedef struct { int empty, multiline, allow_block_plain, allow_single; } AmeScalar;

static AmeScalar ame_analyze(const int *t, int L){
    AmeScalar a = { 0, 0, 1, 1 };
    if (!L) { a.empty = 1; return a; }
    int block_ind = 0, line_breaks = 0, special = 0;
    int leading_space = 0, leading_break = 0, trailing_space = 0, trailing_break = 0, break_space = 0, space_break = 0;
    if (L >= 3 && ((t[0] == '-' && t[1] == '-' && t[2] == '-') || (t[0] == '.' && t[1] == '.' && t[2] == '.'))) block_ind = 1;
    int prev_ws = 1;
    int followed_ws = (L == 1 || t[1] == 0 || t[1] == ' ' || t[1] == '\t' || t[1] == '\r' || ame_isbreak(t[1]));
    int prev_space = 0, prev_break = 0;
    for (int index = 0; index < L;) {
        const int ch = t[index];
        if (index == 0) {
            if (strchr("#,[]{}&*!|>'\"%@`", ch) && ch < 128 && ch) block_ind = 1;
            if ((ch == '?' || ch == ':') && followed_ws) block_ind = 1;
            if (ch == '-' && followed_ws) block_ind = 1;
        } else {
            if (ch == ':' && followed_ws) block_ind = 1;
            if (ch == '#' && prev_ws) block_ind = 1;
        }
        if (ame_isbreak(ch)) line_breaks = 1;
        if (!(ch == '\n' || (ch >= 0x20 && ch <= 0x7E))) {
            if (!((ch == 0x85 || (ch >= 0xA0 && ch <= 0xD7FF) || (ch >= 0xE000 && ch <= 0xFFFD) || (ch >= 0x10000 && ch < 0x10FFFF)) && ch != 0xFEFF)) special = 1;
        }
        if (ch == ' ') {
            if (index == 0) leading_space = 1;
            if (index == L - 1) trailing_space = 1;
            if (prev_break) break_space = 1;
            prev_space = 1; prev_break = 0;
        } else if (ame_isbreak(ch)) {
            if (index == 0) leading_break = 1;
            if (index == L - 1) trailing_break = 1;
            if (prev_space) space_break = 1;
            prev_space = 0; prev_break = 1;
        } else { prev_space = 0; prev_break = 0; }
        index++;
        prev_ws = (ch == 0 || ch == ' ' || ch == '\t' || ch == '\r' || ame_isbreak(ch));
        followed_ws = (index + 1 >= L || t[index + 1] == 0 || t[index + 1] == ' ' || t[index + 1] == '\t' || t[index + 1] == '\r' || ame_isbreak(t[index + 1]));
    }
    if (leading_space || leading_break || trailing_space || trailing_break) a.allow_block_plain = 0;
    if (break_space) { a.allow_block_plain = 0; a.allow_single = 0; }
    if (space_break || special) { a.allow_block_plain = 0; a.allow_single = 0; }
    if (line_breaks) a.allow_block_plain = 0;
    if (block_ind) a.allow_block_plain = 0;
    a.multiline = line_breaks;
    return a;
}

/* One value after "key:" of the top-level block mapping. */
static void ame_yvalue_str(AmeY *y, const int *t, int L){
    const AmeScalar a = ame_analyze(t, L);
    const int implicit = !ame_resolves(t, L);
    if (implicit && a.allow_block_plain) ame_write_plain(y, t, L, 1);
    else if (a.allow_single) ame_write_single(y, t, L, 1);
    else ame_write_double(y, t, L, 1);
}

typedef struct { const char *key; const char *str; long long num; int is_int; } AmeItem;

/* The mapping, in the order given (PyYAML sorts: the caller does). Ends with a line break. malloc'd, or NULL. */
static char *ame_yaml_dump(const AmeItem *it, int n){
    AmeY y; memset(&y, 0, sizeof y);
    y.ws = 1; y.ind = 1; y.col = 0; y.indent = 0;
    for (int i = 0; i < n; i++) {
        y.indent = 0;
        ame_yindent(&y);                                   /* the key's line */
        y.indent = 2;                                      /* increase_indent for the scalar, key and value alike */
        {
            int kn = 0; int *kc = ame_cps(it[i].key, (int)strlen(it[i].key), &kn);
            ame_write_plain(&y, kc, kn, 0);
            free(kc);
        }
        y.indent = 0;
        ame_yindicator(&y, ":", 0, 0, 0);
        y.indent = 2;
        if (it[i].is_int) {
            char b[32]; snprintf(b, sizeof b, "%lld", it[i].num);
            int bn = 0; int *bc = ame_cps(b, (int)strlen(b), &bn);
            ame_write_plain(&y, bc, bn, 1);
            free(bc);
        } else {
            int sn = 0; int *sc = ame_cps(it[i].str ? it[i].str : "", it[i].str ? (int)strlen(it[i].str) : 0, &sn);
            ame_yvalue_str(&y, sc, sn);
            free(sc);
        }
        y.indent = 0;
    }
    y.indent = 0;
    ame_yindent(&y);                                       /* the document end */
    return y.b;
}

/* ---------------------------------------------------------------------------------------------------------- metadata */

enum { AMF_BPM, AMF_CAPTION, AMF_DURATION, AMF_KEYSCALE, AMF_LANGUAGE, AMF_TIMESIG, AMF_N };      /* upstream's order, and PyYAML's (sorted) */
static const char *const AMF_KEYS[AMF_N] = { "bpm", "caption", "duration", "keyscale", "language", "timesignature" };

typedef struct { int have[AMF_N]; char *s[AMF_N]; } AlmMeta;

static void alm_meta_free(AlmMeta *m){ for (int i = 0; i < AMF_N; i++) { free(m->s[i]); m->s[i] = NULL; m->have[i] = 0; } }
static void alm_meta_set(AlmMeta *m, int f, const char *v){ free(m->s[f]); m->s[f] = alm_dup(v); m->have[f] = m->s[f] != NULL; }

/* int(value) as Python reads it, for the digits a model writes: an optional sign and digits. The canonical decimal into out, 1; or 0 when it is not one. */
static int ame_pyint(const char *v, char *out, size_t cap){
    const char *p = v;
    int neg = 0;
    if (*p == '+' || *p == '-') { neg = *p == '-'; p++; }
    if (!*p) return 0;
    char digits[64]; size_t nd = 0;                           /* digits, with a single underscore allowed between two of them (1_0 is 10) */
    for (const char *q = p; *q; q++) {
        if (*q >= '0' && *q <= '9') { if (nd + 1 >= sizeof digits) return 0; digits[nd++] = *q; }
        else if (*q == '_' && nd > 0 && q[-1] != '_' && q[1] >= '0' && q[1] <= '9') continue;
        else return 0;
    }
    digits[nd] = 0;
    p = digits;
    while (*p == '0' && p[1]) p++;
    if (strlen(p) + 2 > cap) return 0;
    snprintf(out, cap, "%s%s", neg && strcmp(p, "0") ? "-" : "", p);
    return 1;
}

/* postprocess_caption: the lines of a YAML multi-line value stripped and joined with one space. */
static char *ame_flatten_caption(const char *v){
    int n = 0; int *cp = ame_cps(v, (int)strlen(v), &n);
    if (!cp) return NULL;
    char *out = (char *)malloc((size_t)n * 4 + 1);
    if (!out) { free(cp); return NULL; }
    size_t o = 0; int first = 1;
    for (int i = 0; i <= n;) {
        int j = i;
        while (j < n && cp[j] != '\n') j++;
        int a = i, b = j;
        while (a < b && ame_isspace(cp[a])) a++;
        while (b > a && ame_isspace(cp[b - 1])) b--;
        if (b > a) {
            if (!first) out[o++] = ' ';
            first = 0;
            for (int k = a; k < b; k++) o += (size_t)ame_put_utf8(out + o, cp[k]);
        }
        i = j + 1;
    }
    out[o] = 0;
    free(cp);
    return out;
}

/* parse_lm_output's metadata half, on the text the model wrote (special tokens as text): the lines inside <think>, a field a line that starts with a non-space and
 * has a colon, its value that line's rest and the indented lines after it, joined with newlines. Only the six fields are kept. */
static void ame_parse(const char *text, AlmMeta *out){
    const char *a = strstr(text, "<think>");
    const char *z = a ? strstr(a + 7, "</think>") : NULL;
    char *body;
    int blank = 0;
    if (a && z) {
        body = (char *)malloc((size_t)(z - (a + 7)) + 1); if (!body) return; memcpy(body, a + 7, (size_t)(z - (a + 7))); body[z - (a + 7)] = 0;
        int bn = 0; int *bc = ame_cps(body, (int)strlen(body), &bn);
        blank = 1; for (int i = 0; bc && i < bn; i++) if (!ame_isspace(bc[i])) blank = 0;
        free(bc);
        if (blank) free(body);                                /* an empty block: upstream parses the whole text instead */
    }
    if (!(a && z) || blank) {                                 /* no closed block: the text before the first audio code (`output_text.split('<|audio_code_')[0]`) */
        const char *code = strstr(text, "<|audio_code_");
        const size_t keep = code ? (size_t)(code - text) : strlen(text);
        body = (char *)malloc(keep + 1);
        if (body) { memcpy(body, text, keep); body[keep] = 0; }
    }
    if (!body) return;
    int n = 0; int *cp = ame_cps(body, (int)strlen(body), &n);
    free(body);
    if (!cp) return;
    int s = 0, e = n;                                         /* reasoning_text.strip() */
    while (s < e && ame_isspace(cp[s])) s++;
    while (e > s && ame_isspace(cp[e - 1])) e--;
    char *key = NULL; char *val = NULL; size_t vlen = 0, vcap = 0; int have_val = 0;
    #define AME_SAVE() do { \
        if (key && have_val) { \
            val[vlen] = 0; \
            int f = -1; for (int q = 0; q < AMF_N; q++) if (!strcmp(key, AMF_KEYS[q])) f = q; \
            if (f >= 0) { \
                int vn = 0; int *vc = ame_cps(val, (int)vlen, &vn); \
                int s2 = 0, e2 = vn; while (s2 < e2 && ame_isspace(vc[s2])) s2++; while (e2 > s2 && ame_isspace(vc[e2 - 1])) e2--; \
                char *st = (char *)malloc((size_t)vn * 4 + 1); size_t so = 0; \
                for (int q = s2; q < e2; q++) so += (size_t)ame_put_utf8(st + so, vc[q]); st[so] = 0; free(vc); \
                char nb[40]; \
                if (f == AMF_CAPTION) { char *fl = ame_flatten_caption(val); alm_meta_set(out, f, fl ? fl : ""); free(fl); } \
                else if ((f == AMF_BPM || f == AMF_DURATION) && ame_pyint(st, nb, sizeof nb)) alm_meta_set(out, f, nb); \
                else alm_meta_set(out, f, st); \
                free(st); \
            } \
        } \
        free(key); key = NULL; vlen = 0; have_val = 0; \
    } while (0)
    for (int i = s; i <= e;) {
        int j = i;
        while (j < e && cp[j] != '\n') j++;
        const int ll = j - i;
        const int *line = cp + i;
        int p = 0; while (p < ll && ame_isspace(line[p])) p++;
        const int starts_tag = p < ll && line[p] == '<';
        int colon = -1; for (int q = 0; q < ll; q++) if (line[q] == ':') { colon = q; break; }
        if (starts_tag) { /* skipped */ }
        else if (ll > 0 && !ame_isspace(line[0]) && colon >= 0) {
            AME_SAVE();
            int kl = 0; int ks = 0, ke = colon; while (ks < ke && ame_isspace(line[ks])) ks++; while (ke > ks && ame_isspace(line[ke - 1])) ke--;
            key = (char *)malloc((size_t)(ke - ks) * 4 + 1);
            for (int q = ks; q < ke; q++) { int c = line[q]; if (c >= 'A' && c <= 'Z') c += 32; kl += ame_put_utf8(key + kl, c); }
            key[kl] = 0;
            int fs = colon + 1; int nonblank = 0; for (int q = fs; q < ll; q++) if (!ame_isspace(line[q])) nonblank = 1;
            vlen = 0; have_val = 0;
            if (nonblank) {
                if (vcap < (size_t)ll * 4 + 2) { vcap = (size_t)ll * 4 + 2 + 1024; val = (char *)realloc(val, vcap); }
                for (int q = fs; q < ll; q++) vlen += (size_t)ame_put_utf8(val + vlen, line[q]);
                have_val = 1;
            }
        } else if (ll > 0 && (line[0] == ' ' || line[0] == '\t')) {
            if (key) {
                if (vcap < vlen + (size_t)ll * 4 + 2) { vcap = vlen + (size_t)ll * 4 + 2 + 1024; val = (char *)realloc(val, vcap); }
                if (have_val) val[vlen++] = '\n';
                for (int q = 0; q < ll; q++) vlen += (size_t)ame_put_utf8(val + vlen, line[q]);
                have_val = 1;
            }
        }
        i = j + 1;
    }
    AME_SAVE();
    #undef AME_SAVE
    free(val); free(cp);
}

/* _format_metadata_as_cot: "<think>\n" + the YAML of the fields that are there, in key order, + "\n</think>". A time signature "N/4" is "N"; a string of digits is
 * an int; bpm and duration were ints already. malloc'd. */
static char *ame_cot(const AlmMeta *m){
    AmeItem it[AMF_N]; char nums[AMF_N][48]; char *own[AMF_N]; int n = 0, no = 0;
    for (int f = 0; f < AMF_N; f++) {
        if (!m->have[f] || !m->s[f]) continue;
        const char *v = m->s[f];
        char *tmp = NULL;
        if (f == AMF_TIMESIG) {
            size_t l = strlen(v);
            if (l >= 2 && !strcmp(v + l - 2, "/4")) { const char *sl = strchr(v, '/'); tmp = (char *)malloc((size_t)(sl - v) + 1); memcpy(tmp, v, (size_t)(sl - v)); tmp[sl - v] = 0; v = tmp; }
        }
        int digits = v[0] != 0;
        for (const char *p = v; *p; p++) if (*p < '0' || *p > '9') digits = 0;
        it[n].key = AMF_KEYS[f];
        if (digits && ame_pyint(v, nums[n], sizeof nums[n])) { it[n].is_int = 1; it[n].num = strtoll(nums[n], NULL, 10); it[n].str = NULL; }
        else if (f == AMF_BPM || f == AMF_DURATION) {
            if (ame_pyint(v, nums[n], sizeof nums[n])) { it[n].is_int = 1; it[n].num = strtoll(nums[n], NULL, 10); it[n].str = NULL; }
            else { it[n].is_int = 0; own[no] = alm_dup(v); it[n].str = own[no++]; }
        } else { it[n].is_int = 0; own[no] = alm_dup(v); it[n].str = own[no++]; }
        free(tmp);
        n++;
    }
    char *y = n ? ame_yaml_dump(it, n) : alm_dup("");
    for (int i = 0; i < no; i++) free(own[i]);
    if (!y) return NULL;
    size_t yl = strlen(y);                                   /* .strip(): the dump ends with one line break and starts with none */
    while (yl && (y[yl - 1] == '\n' || y[yl - 1] == ' ')) y[--yl] = 0;
    char *out = (char *)malloc(yl + 32);
    if (!out) { free(y); return NULL; }
    snprintf(out, yl + 32, "<think>\n%s\n</think>", y);
    free(y);
    return out;
}

/* The codes phase's metadata as the caller gave it: bpm, key and time signature, the duration cut to whole seconds, the language when there is one. malloc'd. */
static char *alm_cot(int bpm, double duration, const char *keyscale, const char *language, const char *timesig){
    AlmMeta m; memset(&m, 0, sizeof m);
    char nb[32];
    snprintf(nb, sizeof nb, "%d", bpm); alm_meta_set(&m, AMF_BPM, nb);
    snprintf(nb, sizeof nb, "%d", (int)duration); alm_meta_set(&m, AMF_DURATION, nb);
    alm_meta_set(&m, AMF_KEYSCALE, keyscale ? keyscale : "");
    if (language && *language) alm_meta_set(&m, AMF_LANGUAGE, language);
    alm_meta_set(&m, AMF_TIMESIG, timesig ? timesig : "");
    char *out = ame_cot(&m);
    alm_meta_free(&m);
    return out;
}

/* ------------------------------------------------------------------------------------------------- the token tables */

#define AME_MAXSEQ 12

typedef struct { int n; int (*seq)[AME_MAXSEQ]; int *len; } AmeTree;      /* the value's token ids after its field's context, for every valid value */

enum { AMS_THINK, AMS_NL_THINK, AMS_BPM_NAME, AMS_BPM_VAL, AMS_CAP_NAME, AMS_CAP_VAL, AMS_DUR_NAME, AMS_DUR_VAL, AMS_KEY_NAME, AMS_KEY_VAL, AMS_LANG_NAME, AMS_LANG_VAL,
       AMS_TS_NAME, AMS_TS_VAL, AMS_END, AMS_DONE, AMS_N };
static const char *const AMS_NAMES[AMS_N] = { "THINK_TAG", "NEWLINE_AFTER_THINK", "BPM_NAME", "BPM_VALUE", "CAPTION_NAME", "CAPTION_VALUE", "DURATION_NAME", "DURATION_VALUE",
                                              "KEYSCALE_NAME", "KEYSCALE_VALUE", "LANGUAGE_NAME", "LANGUAGE_VALUE", "TIMESIG_NAME", "TIMESIG_VALUE", "THINK_END_TAG", "COMPLETED" };
static const char *const AMS_FIXED[AMS_N] = { "<think>", "\n", "bpm:", NULL, "caption:", NULL, "duration:", NULL, "keyscale:", NULL, "language:", NULL, "timesignature:", NULL, "</think>", NULL };

typedef struct {
    AmeTree bpm, dur, key, lang, ts;
    int fixed[AMS_N][AME_MAXSEQ], nfixed[AMS_N];       /* the token ids that spell each fixed string, one at a time */
    int nl, backtick, eos, pad, vocab;
    int *cand_cap, ncap;                               /* the caption's free choice: every token but the audio codes and the backtick, ascending */
    int *cand_all;                                     /* every token */
} AmeTab;

static void ame_tree_free(AmeTree *t){ free(t->seq); free(t->len); memset(t, 0, sizeof *t); }
static void ame_tab_free(void *p){
    AmeTab *t = (AmeTab *)p;
    if (!t) return;
    ame_tree_free(&t->bpm); ame_tree_free(&t->dur); ame_tree_free(&t->key); ame_tree_free(&t->lang); ame_tree_free(&t->ts);
    free(t->cand_cap); free(t->cand_all); free(t);
}

static int ame_tree_build(AceLm *m, AmeTree *t, const char *ctx_match, const char *ctx_tok, const char *const *vals, int nv, int note_first){
    int cn = 0; int *cids = alm_tokenize(m, ctx_match, &cn);
    t->seq = (int (*)[AME_MAXSEQ])calloc((size_t)nv, sizeof *t->seq); t->len = (int *)calloc((size_t)nv, sizeof(int));
    if (!cids || !t->seq || !t->len) { free(cids); return -1; }
    char full[256];
    for (int i = 0; i < nv; i++) {
        snprintf(full, sizeof full, "%s%s", ctx_tok, vals[i]);
        int n = 0; int *ids = alm_tokenize(m, full, &n);
        if (!ids) { free(cids); return -1; }
        int ok = n >= cn && !memcmp(ids, cids, (size_t)cn * sizeof(int));
        const int vl = n - cn;
        if (ok && vl > 0 && vl <= AME_MAXSEQ) {
            if (note_first) {
                char b[64]; int bl = tok_decode(&m->tok, ids + cn, 1, b, 63); b[bl] = 0;
                const char *q = b; while (*q == ' ' || *q == '\t' || *q == '\n') q++;
                const int c = *q >= 'a' && *q <= 'z' ? *q - 32 : *q;
                if (*q && !(c >= 'A' && c <= 'G')) ok = 0;      /* upstream: `first_char not in "ABCDEFG"`, which is False for the empty string of a token that is only space */
            }
            if (ok) { memcpy(t->seq[t->n], ids + cn, (size_t)vl * sizeof(int)); t->len[t->n] = vl; t->n++; }
        }
        free(ids);
    }
    free(cids);
    return 0;
}

/* The longest duration the decoder lets the model write: upstream takes it from the GPU tier (240 s, 480 s or 600 s, the most it supports). */
static int ame_max_duration = 600;

/* The tables upstream builds when its processor is created, from this tokenizer. Cached on the model. NULL (reason in err) when the tokenizer cannot spell a
 * fixed string in single tokens (upstream has a fallback for that; this checkpoint family does not need it). */
static AmeTab *ame_tab(AceLm *m, char *err, size_t errn){
    if (m->meta_tab) return (AmeTab *)m->meta_tab;
    AmeTab *t = (AmeTab *)calloc(1, sizeof *t);
    if (!t) { snprintf(err, errn, "out of memory"); return NULL; }
    t->vocab = m->vocab;
    { int n = 0; int *ids = alm_tokenize(m, "\n", &n); t->nl = n ? ids[n - 1] : -1; free(ids); }
    { int n = 0; int *ids = alm_tokenize(m, "`", &n); t->backtick = n ? ids[n - 1] : -1; free(ids); }
    t->eos = m->eos;
    t->pad = tok_id_of(&m->tok, "<|endoftext|>");
    if (t->nl < 0) { snprintf(err, errn, "the tokenizer has no newline"); ame_tab_free(t); return NULL; }
    for (int s = 0; s < AMS_N; s++) {
        const char *fx = AMS_FIXED[s];
        if (!fx) continue;
        const int L = (int)strlen(fx);
        int pos = 0;
        while (pos < L) {
            int best = -1, bestlen = 0;
            for (int end = L - pos; end >= 1 && best < 0; end--) {
                char buf[64]; memcpy(buf, fx + pos, (size_t)end); buf[end] = 0;
                int n = 0; int *ids = alm_tokenize(m, buf, &n);
                if (ids && n == 1) { best = ids[0]; bestlen = end; }
                free(ids);
            }
            if (best < 0 || t->nfixed[s] >= AME_MAXSEQ) { snprintf(err, errn, "the tokenizer cannot spell \"%s\" in single tokens", fx); ame_tab_free(t); return NULL; }
            t->fixed[s][t->nfixed[s]++] = best;
            pos += bestlen;                                    /* upstream adds the decoded token's length; for these ASCII strings, the prefix's */
        }
    }
    { /* the numeric and the keyscale, language and time signature values */
        char **vals = NULL; const char **cv = NULL; int nv;
        nv = 271; vals = (char **)calloc((size_t)nv, sizeof *vals); cv = (const char **)calloc((size_t)nv, sizeof *cv);
        for (int i = 0; i < nv; i++) { vals[i] = (char *)malloc(16); snprintf(vals[i], 16, "%d", 30 + i); cv[i] = vals[i]; }
        int rc = ame_tree_build(m, &t->bpm, "bpm:", "bpm: ", cv, nv, 0);
        for (int i = 0; i < nv; i++) free(vals[i]);
        free(vals); free(cv);
        if (rc) { snprintf(err, errn, "out of memory"); ame_tab_free(t); return NULL; }
        nv = ame_max_duration - 9; vals = (char **)calloc((size_t)nv, sizeof *vals); cv = (const char **)calloc((size_t)nv, sizeof *cv);
        for (int i = 0; i < nv; i++) { vals[i] = (char *)malloc(16); snprintf(vals[i], 16, "%d", 10 + i); cv[i] = vals[i]; }
        rc = ame_tree_build(m, &t->dur, "duration:", "duration: ", cv, nv, 0);
        for (int i = 0; i < nv; i++) free(vals[i]);
        free(vals); free(cv);
        if (rc) { snprintf(err, errn, "out of memory"); ame_tab_free(t); return NULL; }
    }
    {
        static const char *const ts[] = { "2", "3", "4", "6" };
        if (ame_tree_build(m, &t->ts, "timesignature:", "timesignature: ", ts, 4, 0)) { snprintf(err, errn, "out of memory"); ame_tab_free(t); return NULL; }
        static const char *const langs[] = { "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt",
                                             "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown" };
        if (ame_tree_build(m, &t->lang, "language:", "language: ", langs, (int)(sizeof langs / sizeof *langs), 0)) { snprintf(err, errn, "out of memory"); ame_tab_free(t); return NULL; }
        static const char *const notes[] = { "A", "B", "C", "D", "E", "F", "G" }, *const acc[] = { "", "#", "b", "\xE2\x99\xAF", "\xE2\x99\xAD" }, *const modes[] = { "major", "minor" };
        static char ks[70][16]; const char *kv[70]; int k = 0;
        for (int a = 0; a < 7; a++) for (int c = 0; c < 5; c++) for (int d = 0; d < 2; d++) { snprintf(ks[k], 16, "%s%s %s", notes[a], acc[c], modes[d]); kv[k] = ks[k]; k++; }
        if (ame_tree_build(m, &t->key, "keyscale:", "keyscale: ", kv, 70, 1)) { snprintf(err, errn, "out of memory"); ame_tab_free(t); return NULL; }
    }
    t->cand_all = (int *)malloc((size_t)m->vocab * sizeof(int));
    t->cand_cap = (int *)malloc((size_t)m->vocab * sizeof(int));
    if (!t->cand_all || !t->cand_cap) { snprintf(err, errn, "out of memory"); ame_tab_free(t); return NULL; }
    for (int i = 0; i < m->vocab; i++) {
        t->cand_all[i] = i;
        if ((i < m->audio0 || i >= m->audio0 + ALM_CODES) && i != t->backtick) t->cand_cap[t->ncap++] = i;
    }
    m->meta_tab = t; m->meta_free = ame_tab_free;
    return t;
}

/* The tokens that may follow `prefix` in a tree, ascending and unique, with the newline after a complete value; *has says whether the prefix is in the tree at all. */
static int ame_tree_next(const AmeTree *t, int nl, const int *prefix, int pl, int *out, int cap, int *has){
    int n = 0; *has = 0;
    for (int i = 0; i < t->n; i++) {
        if (t->len[i] < pl || memcmp(t->seq[i], prefix, (size_t)pl * sizeof(int))) continue;
        *has = 1;
        const int tok = t->len[i] == pl ? nl : t->seq[i][pl];
        int dup = 0; for (int k = 0; k < n; k++) if (out[k] == tok) { dup = 1; break; }
        if (!dup && n < cap) out[n++] = tok;
    }
    for (int i = 1; i < n; i++) { int v = out[i], j = i - 1; while (j >= 0 && out[j] > v) { out[j + 1] = out[j]; j--; } out[j + 1] = v; }
    return n;
}

/* ----------------------------------------------------------------------------------------------- the decoder's machine */

typedef struct {
    int state, pos;                    /* the state, and how many tokens of its fixed string are out */
    int acc[AME_MAXSEQ + 4], nacc;     /* the value's token ids so far */
    int after_nl, ending, cap_count;
    char pend[256]; int pendn;         /* the field name the model is writing after the caption */
    int queue[64], qn, qi, qfield;     /* a field the caller gave, token by token */
    int skip_caption, skip_language;
    int user_bpm; const char *user_key, *user_ts; int user_dur;
} AmeFsm;

static int ame_next_field(const AmeFsm *f, int field){          /* _get_next_field_state: the NAME state of the next field, THINK_END_TAG at the end */
    static const int names[AMF_N] = { AMS_BPM_NAME, AMS_CAP_NAME, AMS_DUR_NAME, -1, -1, AMS_TS_NAME };
    for (int i = field + 1; i < AMF_N; i++) {
        if (i == AMF_CAPTION && f->skip_caption) continue;
        if (i == AMF_LANGUAGE && f->skip_language) continue;
        if (i == AMF_KEYSCALE) return AMS_KEY_NAME;
        if (i == AMF_LANGUAGE) return AMS_LANG_NAME;
        return names[i];
    }
    return AMS_END;
}

static int ame_state_next(const AmeFsm *f, int s){               /* next_state[s], or -1 */
    switch (s) {
        case AMS_THINK: return AMS_NL_THINK;
        case AMS_NL_THINK: return AMS_BPM_NAME;
        case AMS_BPM_NAME: return AMS_BPM_VAL;
        case AMS_BPM_VAL: return ame_next_field(f, AMF_BPM);
        case AMS_CAP_NAME: return f->skip_caption ? -1 : AMS_CAP_VAL;
        case AMS_CAP_VAL: return f->skip_caption ? -1 : ame_next_field(f, AMF_CAPTION);
        case AMS_DUR_NAME: return AMS_DUR_VAL;
        case AMS_DUR_VAL: return ame_next_field(f, AMF_DURATION);
        case AMS_KEY_NAME: return AMS_KEY_VAL;
        case AMS_KEY_VAL: return ame_next_field(f, AMF_KEYSCALE);
        case AMS_LANG_NAME: return f->skip_language ? -1 : AMS_LANG_VAL;
        case AMS_LANG_VAL: return f->skip_language ? -1 : ame_next_field(f, AMF_LANGUAGE);
        case AMS_TS_NAME: return AMS_TS_VAL;
        case AMS_TS_VAL: return AMS_END;
        default: return -1;
    }
}

static void ame_enter(AmeFsm *f, int s){
    f->state = s; f->pos = 0; f->nacc = 0; f->after_nl = 0; f->ending = 0; f->cap_count = 0; f->pendn = 0;
}
static void ame_transition(AmeFsm *f){ int nx = ame_state_next(f, f->state); if (nx >= 0) ame_enter(f, nx); }

/* A step's decision: tokens that may be chosen. mode 0: the given list (ascending); 1: the caption's free choice; 2: every token. */
typedef struct { int mode, n, ids[AME_MAXSEQ * 8 + 8]; } AmeAllowed;

typedef void (*AmTraceFn)(void *ud, int step, int state, int n, const int *ids, const float *raw, int tok);       /* after each choice: what the machine let through and the raw logits there */

typedef struct {
    int bpm; const char *key, *timesig; int duration;     /* the caller's own values: 0 or NULL when not given (they are written for the model, token by token) */
    int caption, language;                                /* have the model write the caption, the language */
} AlmAsk;

static int ame_queue_field(AceLm *m, AmeFsm *f, int field, const char *value){
    char text[160]; snprintf(text, sizeof text, " %s\n", value);
    int n = 0; int *ids = alm_tokenize(m, text, &n);
    if (!ids || n < 1 || n > 64) { free(ids); return -1; }
    memcpy(f->queue, ids, (size_t)n * sizeof(int)); f->qn = n; f->qi = 0; f->qfield = field;
    free(ids);
    return 0;
}

/* The constraint of the state, with the injection of the caller's values; fills the allowed tokens or says which free mode. Returns -1 when the machine is stuck. */
static int ame_allowed(AceLm *m, const AmeTab *t, AmeFsm *f, AmeAllowed *a){
    a->mode = 0; a->n = 0;
    if (f->qi < f->qn) { a->ids[a->n++] = f->queue[f->qi]; return 0; }
    int has;
    switch (f->state) {
        case AMS_END: a->ids[a->n++] = t->eos; return 0;
        case AMS_THINK: case AMS_NL_THINK: case AMS_BPM_NAME: case AMS_CAP_NAME: case AMS_DUR_NAME: case AMS_KEY_NAME: case AMS_LANG_NAME: case AMS_TS_NAME:
            if (f->pos >= t->nfixed[f->state]) return -1;
            a->ids[a->n++] = t->fixed[f->state][f->pos];
            return 0;
        case AMS_BPM_VAL: case AMS_DUR_VAL: {
            const int bpm = f->state == AMS_BPM_VAL;
            if ((bpm ? f->user_bpm > 0 : f->user_dur > 0) && f->nacc == 0) {
                char v[24]; snprintf(v, sizeof v, "%d", bpm ? f->user_bpm : f->user_dur);
                if (ame_queue_field(m, f, bpm ? AMF_BPM : AMF_DURATION, v)) return -1;
                a->ids[a->n++] = f->queue[0];
                return 0;
            }
            a->n = ame_tree_next(bpm ? &t->bpm : &t->dur, t->nl, f->acc, f->nacc, a->ids, (int)(sizeof a->ids / sizeof *a->ids), &has);
            return 0;
        }
        case AMS_KEY_VAL: case AMS_TS_VAL: case AMS_LANG_VAL: {
            const int key = f->state == AMS_KEY_VAL, ts = f->state == AMS_TS_VAL;
            const char *user = key ? f->user_key : ts ? f->user_ts : NULL;
            if (user && *user && f->nacc == 0) {
                if (ame_queue_field(m, f, key ? AMF_KEYSCALE : AMF_TIMESIG, user)) return -1;
                a->ids[a->n++] = f->queue[0];
                return 0;
            }
            const AmeTree *tr = key ? &t->key : ts ? &t->ts : &t->lang;
            if (f->state == AMS_LANG_VAL && f->nacc == 0) { a->mode = 3; return 0; }       /* the model's top choice among the first tokens: decided on its logits */
            int nx[64], nn = ame_tree_next(tr, t->nl, f->acc, f->nacc, nx, 64, &has);
            int complete = 0; for (int i = 0; i < nn; i++) if (nx[i] == t->nl) complete = 1;
            if (has && complete) { a->ids[a->n++] = t->nl; return 0; }
            if (nn) { memcpy(a->ids, nx, (size_t)nn * sizeof(int)); a->n = nn; return 0; }
            if (!ts) { a->ids[a->n++] = t->nl; return 0; }                                  /* nothing valid to add: end the field (not so for the time signature) */
            a->n = 0;
            return 0;
        }
        case AMS_CAP_VAL:
            a->mode = 1;
            return 0;
        default: return -1;
    }
}

static int ame_tok_str(AceLm *m, int id, char *buf, int cap){ int n = tok_decode(&m->tok, &id, 1, buf, cap - 1); if (n < 0) n = 0; buf[n] = 0; return n; }

static int ame_utf8_chars(const char *s){ int n = 0; for (; *s; s++) if (((unsigned char)*s & 0xC0) != 0x80) n++; return n; }

/* update_state: the machine takes the token that was chosen. Returns 1 when it met something it does not support (a genres field). */
static int ame_update(AceLm *m, const AmeTab *t, AmeFsm *f, int tok){
    if (f->state == AMS_DONE) return 0;
    if (f->qi < f->qn) {
        f->qi++;
        if (f->qi >= f->qn) {
            const int field = f->qfield;
            f->qn = f->qi = 0;
            ame_enter(f, ame_next_field(f, field));
        }
        return 0;
    }
    char ts[64]; ame_tok_str(m, tok, ts, sizeof ts);
    switch (f->state) {
        case AMS_THINK: case AMS_NL_THINK: case AMS_BPM_NAME: case AMS_CAP_NAME: case AMS_DUR_NAME: case AMS_KEY_NAME: case AMS_LANG_NAME: case AMS_TS_NAME: case AMS_END: {
            f->pos++;
            if (f->pos >= t->nfixed[f->state] || f->state == AMS_END) {
                if (f->state == AMS_END) { f->state = AMS_DONE; f->pos = 0; f->nacc = 0; }
                else ame_transition(f);
            }
            break;
        }
        case AMS_BPM_VAL: case AMS_DUR_VAL: case AMS_TS_VAL: case AMS_KEY_VAL: case AMS_LANG_VAL:
            if (tok == t->nl) ame_transition(f);
            else if (f->nacc < (int)(sizeof f->acc / sizeof *f->acc)) f->acc[f->nacc++] = tok;
            break;
        case AMS_CAP_VAL: {
            f->cap_count++;
            f->after_nl = strchr(ts, '\n') != NULL;
            if (f->ending) {
                size_t l = strlen(ts);
                if (f->pendn + l < sizeof f->pend) { memcpy(f->pend + f->pendn, ts, l); f->pendn += (int)l; f->pend[f->pendn] = 0; }
                if (strchr(ts, ':')) {
                    char nm[256]; int k = 0, a = 0, b = f->pendn;
                    while (a < b && (f->pend[a] == ' ' || f->pend[a] == '\t' || f->pend[a] == '\n' || f->pend[a] == '\r')) a++;
                    while (b > a && (f->pend[b - 1] == ' ' || f->pend[b - 1] == '\t' || f->pend[b - 1] == '\n' || f->pend[b - 1] == '\r' || f->pend[b - 1] == ':')) b--;
                    while (a < b && (f->pend[a] == ' ' || f->pend[a] == '\t')) a++;
                    for (int i = a; i < b; i++) nm[k++] = f->pend[i] >= 'A' && f->pend[i] <= 'Z' ? (char)(f->pend[i] + 32) : f->pend[i];
                    nm[k] = 0;
                    if (!strcmp(nm, "duration")) ame_enter(f, AMS_DUR_VAL);
                    else if (!strcmp(nm, "keyscale")) ame_enter(f, AMS_KEY_VAL);
                    else if (!strcmp(nm, "language")) ame_enter(f, AMS_LANG_VAL);
                    else if (!strcmp(nm, "timesignature")) ame_enter(f, AMS_TS_VAL);
                    else if (!strcmp(nm, "genres")) return 1;
                    else { f->ending = 0; f->pendn = 0; ame_transition(f); }
                }
            }
            break;
        }
        default: break;
    }
    return 0;
}

/* ------------------------------------------------------------------------------------------------- phase 1 itself */

#define ALM_META_MAX_NEW 3500             /* upstream: min(4096, DURATION_MAX * 5 + 500) new tokens */
#define ALM_META_CAPTION_MAX 512

/* The prompt of phase 1: the chat template with the instruction as the system turn, the caption and the lyrics, and the assistant turn left open (the machine
 * writes <think> itself). malloc'd. */
static char *alm_meta_prompt(const char *caption, const char *lyrics){
    const char *cp = caption ? caption : "", *ly = lyrics ? lyrics : "";
    size_t cap = strlen(cp) + strlen(ly) + 512;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    snprintf(out, cap, "<|im_start|>system\n# Instruction\n%s\n\n<|im_end|>\n<|im_start|>user\n# Caption\n%s\n\n# Lyric\n%s\n<|im_end|>\n<|im_start|>assistant\n", ALM_INSTRUCTION, cp, ly);
    return out;
}

/* Upstream's phase 1: the prompt in, then a token a step under the machine until the end token, the metadata parsed from what the model wrote. `step` (may be
 * NULL) hears each step with the tokens the machine let through and the model's raw logit at each. Returns the tokens written (and the metadata in out), -1 with
 * a reason in err, or -2 when `prog` asked to stop. */
static int alm_generate_meta_run(AceLm *m, const int *prompt, int np, const AlmAsk *ask, const AlmSample *s, AlmMeta *out, AmTraceFn trace, void *ud,
                                 int (*prog)(void *ud, int step), void *pud, int *wrote_ids, char *err, size_t errn){
    AmeTab *t = ame_tab(m, err, errn);
    if (!t) return -1;
    const int D = m->D;
    AmeFsm f; memset(&f, 0, sizeof f);
    f.state = AMS_THINK;
    f.skip_caption = !ask->caption; f.skip_language = !ask->language;
    f.user_bpm = ask->bpm; f.user_key = ask->key; f.user_ts = ask->timesig; f.user_dur = ask->duration;
    AlmCache c; memset(&c, 0, sizeof c);
    const int cap = np + ALM_META_MAX_NEW + 8;
    float *h = (float *)malloc((size_t)D * sizeof(float)), *x = (float *)malloc((size_t)D * sizeof(float));
    float *lg = (float *)malloc((size_t)m->vocab * sizeof(float)), *wk = (float *)malloc((size_t)m->vocab * sizeof(float));
    int *gen = (int *)malloc((size_t)ALM_META_MAX_NEW * sizeof(int));
    int *ids_buf = (int *)malloc((size_t)m->vocab * sizeof(int));
    int produced = -1;
    uint64_t rng = s->seed ? s->seed : 0x2545F4914F6CDD1Dull;
    if (!h || !x || !lg || !wk || !gen || !ids_buf || alm_cache_new(m, &c, cap)) { snprintf(err, errn, "out of memory"); goto done; }
    alm_prefill(m, &c, prompt, np, h);
    AlmCache *caches[1] = { &c };
    int pos = np, count = 0;
    AlmSample sm = *s; sm.cfg_scale = 1.0;
    for (int step = 0; step < ALM_META_MAX_NEW; step++) {
        AmeAllowed a;
        if (ame_allowed(m, t, &f, &a) < 0) { snprintf(err, errn, "the planner's decoder is stuck in %s", AMS_NAMES[f.state]); goto done; }
        const int *ids = a.ids; int n = a.n;
        const int want = trace != NULL;
        int tok = -1;
        int mode = a.mode, have_all = 0;
        if (mode == 1) {                                            /* the caption: every token but the audio codes and the backtick */
            if (f.after_nl) {                                       /* a line break was written: is the next line indented (the caption goes on) or a new field? */
                alm_logits(m, h, 1, t->cand_all, t->vocab, lg); have_all = 1;
                int best = 0; for (int i = 1; i < t->vocab; i++) if (lg[i] > lg[best]) best = i;
                char b[64]; int bl = ame_tok_str(m, best, b, sizeof b);
                f.after_nl = 0;
                if (bl > 0 && b[0] != ' ' && b[0] != '\t') { f.ending = 1; f.pendn = 0; mode = 2; }
            }
            if (mode == 1 && f.ending) mode = 2;
            if (mode == 1 && f.cap_count >= ALM_META_CAPTION_MAX) { ids_buf[0] = t->nl; ids = ids_buf; n = 1; mode = 0; }
        }
        if (mode == 1) { ids = t->cand_cap; n = t->ncap; }
        else if (mode == 2) { ids = t->cand_all; n = t->vocab; }
        else if (mode == 3) {                                       /* the language's first token: the top choice among those that start a language */
            int has; int first[96]; int nf = ame_tree_next(&t->lang, t->nl, NULL, 0, first, 96, &has);
            alm_logits(m, h, 1, first, nf, lg);
            int best = 0; for (int i = 1; i < nf; i++) if (lg[i] > lg[best]) best = i;
            ids_buf[0] = first[best]; ids = ids_buf; n = 1; mode = 0;
        }
        if (n < 1) { snprintf(err, errn, "the planner's decoder let no token through in %s", AMS_NAMES[f.state]); goto done; }
        if (n == 1 && !want) tok = ids[0];
        else {
            if (!(have_all && ids == t->cand_all)) alm_logits(m, h, 1, ids, n, lg);
            if (n == 1) tok = ids[0];
            else { memcpy(wk, lg, (size_t)n * sizeof(float)); tok = ids[alm_choose(wk, n, &sm, &rng)]; }
        }
        if (trace) trace(ud, step, f.state, n, ids, lg, tok);
        if (prog && prog(pud, step)) { produced = -2; goto done; }
        gen[count++] = tok;
        if (ame_update(m, t, &f, tok)) { snprintf(err, errn, "the planner wrote a genres field, which is not built"); goto done; }
        if (tok == t->eos || (t->pad >= 0 && tok == t->pad)) break;
        alm_embed_row(m, tok, x);
        alm_decode_rows(m, caches, &pos, x, 1);
        pos++;
        memcpy(h, x, (size_t)D * sizeof(float));
    }
    {
        size_t cap_text = (size_t)count * 64 + 64;
        char *text = (char *)malloc(cap_text);
        if (!text) { snprintf(err, errn, "out of memory"); goto done; }
        int tl = tok_decode(&m->tok, gen, count, text, (int)cap_text - 1);
        if (tl < 0) tl = 0;
        text[tl] = 0;
        memset(out, 0, sizeof *out);
        ame_parse(text, out);
        free(text);
        if (wrote_ids) memcpy(wrote_ids, gen, (size_t)count * sizeof(int));
        produced = count;
    }
done:
    free(h); free(x); free(lg); free(wk); free(gen); free(ids_buf);
    alm_cache_free(m, &c);
    return produced;
}

/* A run that failed on the card is made again on the CPU (the device is off for the rest of the process by then). */
static int alm_generate_meta(AceLm *m, const int *prompt, int np, const AlmAsk *ask, const AlmSample *s, AlmMeta *out, AmTraceFn trace, void *ud,
                             int (*prog)(void *ud, int step), void *pud, int *wrote_ids, char *err, size_t errn){
    m->vk_fail = 0;
    int r = alm_generate_meta_run(m, prompt, np, ask, s, out, trace, ud, prog, pud, wrote_ids, err, errn);
    if (m->vk_fail) { m->vk_fail = 0; r = alm_generate_meta_run(m, prompt, np, ask, s, out, trace, ud, prog, pud, wrote_ids, err, errn); }
    return r;
}

#endif
