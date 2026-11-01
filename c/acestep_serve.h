/* acestep_serve.h -- the line protocol of `acestep --model DIR --serve`, the way the gateway and `coli music` drive the engine.
 *
 * One line per control message: a keyword, one space and a JSON object. Two of them are followed by raw bytes.
 *
 *   READY {...}                    once, after the weights are loaded: the model, the audio format, the limits and defaults
 *   GEN {...}                      client, one request at a time:
 *                                    id, caption (required), lyrics, language, duration (s), steps, shift, seed,
 *                                    bpm, key, timesig, dcw
 *   PROGRESS {...}                 any number: {"id", "stage": "text"|"condition"|"denoise"|"decode", "step", "steps"}
 *   AUDIO {..."bytes":N} + N + \n  signed 16-bit little-endian PCM, channels interleaved, 48 kHz
 *   ERROR {"id", "message"}        the engine stays alive for the next GEN
 *   CANCEL {"id"}                  client; answered with ERROR "cancelled" once the running request has stopped
 *
 * A GEN that arrives while another is running is kept and run after it (up to 64), never dropped. A line over 1 MiB
 * is refused with the id found at its start. Nothing is ever written to stdout but these frames. */
#ifndef ACESTEP_SERVE_H
#define ACESTEP_SERVE_H

#include "serve_poll.h"
#include "acestep_gen.h"

#define ACS_MAX_LINE ((size_t)1 << 20)
#define ACS_DEFAULT_DURATION 30.0
#define ACS_DEFAULT_STEPS 8
#define ACS_DEFAULT_SHIFT 1.0
#ifndef ACS_MODEL_NAME
#define ACS_MODEL_NAME "ace-step-1.5-turbo"
#endif

static char *acs_in; static size_t acs_in_n, acs_in_cap;
static int acs_skipping; static char acs_skip_id[128];
static char *acs_later[64]; static int acs_later_n;
static int acs_cancel_pending; static char acs_cancel_id[128];

static void acs_note_id(const char *buf, size_t n){
    snprintf(acs_skip_id, sizeof acs_skip_id, "?");
    const char *k = NULL;
    for (size_t i = 0; i + 6 <= (n < 4096 ? n : 4096); i++) if (!memcmp(buf + i, "\"id\":\"", 6)) { k = buf + i; break; }
    if (!k) return;
    k += 6;
    size_t i = 0;
    while (i + 1 < sizeof acs_skip_id && k + i < buf + n && k[i] != '"' && k[i] != '\\' && (unsigned char)k[i] >= 0x20) { acs_skip_id[i] = k[i]; i++; }
    acs_skip_id[i] = 0;
}

/* The next complete line, malloc'd, or NULL at end of input or (block = 0) when none is there yet. */
static char *acs_read_line(int block){
    for (;;) {
        char *nl = acs_in_n ? (char *)memchr(acs_in, '\n', acs_in_n) : NULL;
        if (nl) {
            size_t len = (size_t)(nl - acs_in);
            char *line;
            if (acs_skipping) {                       /* the tail of an overlong line */
                acs_skipping = 0;
                line = (char *)malloc(sizeof acs_skip_id + 16);
                if (!line) exit(1);
                snprintf(line, sizeof acs_skip_id + 16, "TOOLONG %s", acs_skip_id);
            } else {
                line = (char *)malloc(len + 1);
                if (!line) exit(1);
                memcpy(line, acs_in, len); line[len] = 0;
                if (len && line[len - 1] == '\r') line[len - 1] = 0;
            }
            memmove(acs_in, nl + 1, acs_in_n - len - 1); acs_in_n -= len + 1;
            return line;
        }
        if (acs_in_n == ACS_MAX_LINE) {
            if (!acs_skipping) { acs_note_id(acs_in, acs_in_n); acs_skipping = 1; }
            acs_in_n = 0;
        }
        if (!block && !coli_serve_stdin_ready()) return NULL;
        if (acs_in_n == acs_in_cap) {
            size_t cap = acs_in_cap ? acs_in_cap * 2 : (size_t)1 << 16;
            if (cap > ACS_MAX_LINE) cap = ACS_MAX_LINE;
            char *grown = (char *)realloc(acs_in, cap);
            if (!grown) { fprintf(stderr, "[acestep] out of memory reading stdin\n"); exit(1); }
            acs_in = grown; acs_in_cap = cap;
        }
        ssize_t r = read(0, acs_in + acs_in_n, acs_in_cap - acs_in_n);    /* the descriptor, not stdio: stdio's buffer would hide a CANCEL from the poll */
        if (r <= 0) return NULL;
        acs_in_n += (size_t)r;
    }
}

static void acs_escape(const char *s){
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') printf("\\%c", c);
        else if (c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
}

static void acs_error(const char *id, const char *m){
    printf("ERROR {\"id\":\""); acs_escape(id); printf("\",\"message\":\""); acs_escape(m); printf("\"}\n");
    fflush(stdout);
}

static jval *acs_parse_line(const char *body, char **arena){
    *arena = NULL;
    return json_parse(body, arena);
}

static const char *acs_id_of(jval *j, char *id, size_t n){
    jval *v = j ? json_get(j, "id") : NULL;
    snprintf(id, n, "%s", v && v->t == J_STR ? v->str : "?");
    return id;
}

/* Lines that come in while a request runs: CANCEL for it is noted, a GEN is kept for the loop. */
static int acs_cancelled(void *ud){
    const char *id = (const char *)ud;
    char *line;
    while ((line = acs_read_line(0))) {
        if (strncmp(line, "CANCEL ", 7)) {
            if (acs_later_n < (int)(sizeof acs_later / sizeof acs_later[0])) { acs_later[acs_later_n++] = line; continue; }
            if (!strncmp(line, "GEN ", 4)) {                         /* the queue is full: answer rather than drop */
                char *arena; jval *j = acs_parse_line(line + 4, &arena); char other[128];
                acs_error(acs_id_of(j, other, sizeof other), "engine busy");
                if (j) json_free(j);
                free(arena);
            }
            free(line);
            continue;
        }
        char *arena; jval *j = acs_parse_line(line + 7, &arena); char who[128];
        acs_id_of(j, who, sizeof who);
        if (j && who[0] != '?') { snprintf(acs_cancel_id, sizeof acs_cancel_id, "%s", who); acs_cancel_pending = 1; }
        if (j) json_free(j);
        free(arena); free(line);
    }
    return acs_cancel_pending && !strcmp(acs_cancel_id, id);
}

typedef struct { const char *id; } AcsRun;
static void acs_progress(void *ud, const char *stage, int step, int steps){
    printf("PROGRESS {\"id\":\""); acs_escape(((AcsRun *)ud)->id);
    printf("\",\"stage\":\"%s\",\"step\":%d,\"steps\":%d}\n", stage, step, steps);
    fflush(stdout);
}
static int acs_cancelled_run(void *ud){ return acs_cancelled((void *)((AcsRun *)ud)->id); }

static int acs_number(jval *j, const char *key, double *out, char *err, size_t errn){
    jval *v = json_get(j, key);
    if (!v || v->t == J_NULL) return 0;
    if (v->t != J_NUM || !isfinite(v->num)) { snprintf(err, errn, "%s must be a number", key); return -1; }
    *out = v->num;
    return 1;
}
static int acs_string(jval *j, const char *key, const char **out, char *err, size_t errn){
    jval *v = json_get(j, key);
    if (!v || v->t == J_NULL) return 0;
    if (v->t != J_STR) { snprintf(err, errn, "%s must be a string", key); return -1; }
    *out = v->str;
    return 1;
}

/* GEN's fields into a request, defaults for what is absent. 0 or -1 with err. */
static int acs_request(jval *j, AceRequest *rq, char *err, size_t errn){
    memset(rq, 0, sizeof *rq);
    rq->duration = ACS_DEFAULT_DURATION; rq->steps = ACS_DEFAULT_STEPS; rq->shift = ACS_DEFAULT_SHIFT; rq->dcw = 1;
    rq->seed = (uint64_t)((uint32_t)time(NULL) ^ ((uint32_t)clock() << 7));
    double x;
    int r;
    if (acs_string(j, "caption", &rq->caption, err, errn) < 0) return -1;
    if (!rq->caption) { snprintf(err, errn, "caption is required (it may be empty)"); return -1; }
    if (acs_string(j, "lyrics", &rq->lyrics, err, errn) < 0) return -1;
    if (acs_string(j, "language", &rq->language, err, errn) < 0) return -1;
    if (acs_string(j, "key", &rq->key, err, errn) < 0) return -1;
    if (acs_string(j, "timesig", &rq->timesig, err, errn) < 0) return -1;
    if ((r = acs_number(j, "duration", &rq->duration, err, errn)) < 0) return -1;
    if ((r = acs_number(j, "shift", &rq->shift, err, errn)) < 0) return -1;
    if ((r = acs_number(j, "steps", &x, err, errn)) < 0) return -1;
    if (r) { if (x != floor(x) || fabs(x) > 1e6) { snprintf(err, errn, "steps must be a whole number"); return -1; } rq->steps = (int)x; }
    if ((r = acs_number(j, "bpm", &x, err, errn)) < 0) return -1;
    if (r) { if (x != floor(x) || x < 0 || x > 1e6) { snprintf(err, errn, "bpm must be a whole number"); return -1; } rq->bpm = (int)x; }
    if ((r = acs_number(j, "seed", &x, err, errn)) < 0) return -1;
    if (r) { if (x != floor(x) || x < 0 || x > ACG_MAX_SEED) { snprintf(err, errn, "seed must be a whole number from 0 to 4294967295"); return -1; } rq->seed = (uint64_t)x; }
    jval *d = json_get(j, "dcw");
    if (d && d->t != J_NULL) { if (d->t != J_BOOL) { snprintf(err, errn, "dcw must be true or false"); return -1; } rq->dcw = d->boolean; }
    rq->seed &= 0xffffffffu;
    return 0;
}

/* Signed 16-bit little-endian, as write_wav16 rounds it. */
static int16_t *acs_pcm16(const float *x, size_t n){
    int16_t *o = (int16_t *)malloc((n ? n : 1) * sizeof(int16_t));
    if (!o) return NULL;
    for (size_t i = 0; i < n; i++) {
        float v = x[i] < -1.f ? -1.f : x[i] > 1.f ? 1.f : x[i];
        o[i] = (int16_t)lrintf(v * 32767.f);
    }
    return o;
}

static int acs_serve(AceEngine *e){
    coli_serve_binary_mode();
    printf("READY {\"model\":\"%s\",\"sample_rate\":%d,\"channels\":%d,\"format\":\"s16le\",\"default_duration\":%g,\"min_duration\":%g,"
           "\"max_duration\":%g,\"default_steps\":%d,\"max_steps\":%d,\"default_shift\":%g,\"max_caption_bytes\":%d,\"max_lyric_bytes\":%d,"
           "\"max_seed\":4294967295,\"protocol\":1}\n",
           ACS_MODEL_NAME, e->vae->rate, e->vae->ach, ACS_DEFAULT_DURATION, ACG_MIN_SECONDS, ACG_MAX_SECONDS, ACS_DEFAULT_STEPS,
           ACG_MAX_STEPS, ACS_DEFAULT_SHIFT, ACG_MAX_CAPTION_BYTES, ACG_MAX_LYRIC_BYTES);
    fflush(stdout);
    char *line;
    for (;;) {
        if (acs_later_n) {
            line = acs_later[0];
            memmove(acs_later, acs_later + 1, sizeof acs_later[0] * (size_t)--acs_later_n);
        } else if (!(line = acs_read_line(1))) break;
        if (!strncmp(line, "TOOLONG ", 8)) { acs_error(line + 8, "request line too long"); free(line); continue; }
        if (strncmp(line, "GEN ", 4)) { free(line); continue; }               /* a CANCEL with nothing running, noise */
        char *arena; jval *j = acs_parse_line(line + 4, &arena);
        char id[128], msg[512] = "";
        acs_id_of(j, id, sizeof id);
        AceRequest rq;
        if (!j || j->t != J_OBJ) acs_error(id, "bad GEN line (not a JSON object)");
        else if (acs_request(j, &rq, msg, sizeof msg) || ace_check(&rq, msg, sizeof msg)) acs_error(id, msg);
        else {
            acs_cancel_pending = 0;
            AcsRun run = { id };
            AceHooks hk = { acs_progress, acs_cancelled_run, &run };
            float *pcm = NULL; AceInfo info;
            int rc = ace_generate(e, &rq, &hk, &pcm, &info, msg, sizeof msg);
            int16_t *s16 = rc == 0 ? acs_pcm16(pcm, (size_t)info.samples * e->vae->ach) : NULL;
            if (rc == 0 && !s16) { acs_error(id, "out of memory"); }
            else if (rc) acs_error(id, msg);
            else {
                size_t bytes = (size_t)info.samples * e->vae->ach * sizeof(int16_t);
                printf("AUDIO {\"id\":\""); acs_escape(id);
                printf("\",\"sample_rate\":%d,\"channels\":%d,\"samples\":%d,\"bytes\":%zu,\"seed\":%llu,\"steps\":%d,\"shift\":%g,\"duration\":%.3f,"
                       "\"frames\":%d,\"caption_tokens\":%d,\"lyric_tokens\":%d,\"truncated\":%s,\"peak\":%.4f,"
                       "\"timings\":{\"text\":%.2f,\"condition\":%.2f,\"denoise\":%.2f,\"decode\":%.2f}}\n",
                       e->vae->rate, e->vae->ach, info.samples, bytes, (unsigned long long)rq.seed, rq.steps, rq.shift,
                       (double)info.samples / e->vae->rate, info.frames, info.caption_tokens, info.lyric_tokens,
                       info.caption_cut || info.lyric_cut ? "true" : "false", info.peak, info.t_text, info.t_cond, info.t_denoise, info.t_decode);
                fwrite(s16, 1, bytes, stdout);
                putchar('\n');
                fflush(stdout);
            }
            free(s16); free(pcm);
        }
        if (j) json_free(j);
        free(arena); free(line);
    }
    return 0;
}

#endif
