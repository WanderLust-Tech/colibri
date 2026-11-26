/* colibri-tray.exe: the notification-area icon for colibri, as a program of its own (docs/tray.md).
 *
 * It is c/tray.py again, without Python to draw it: the same icon, the same menu and the same words, so the two can be
 * swapped and a test holds them to each other (tests/test_tray_exe.py runs both on the same states). What it does itself:
 * read service.json, ask the server's /health every few seconds (WinHTTP), draw the icon and the menu, unload a model, copy
 * the key or the link, open the dashboard, the logs, and keep the "start with Windows" entry. What it leaves to
 * `coli service` is what is hard to say twice: starting the server with its key and environment, and stopping it (the
 * refusals, the process tree). Those run `python coli service start|stop|restart`, so Python is needed to start the
 * server, as it always was, and not to show the icon.
 *
 *   colibri-tray.exe                       the tray (one at a time: the mutex is the one tray.py takes)
 *   colibri-tray.exe --python PYTHON.EXE   the interpreter that runs `coli service` (else COLI_PYTHON, else python on PATH)
 *   colibri-tray.exe --config FILE         the service settings (else COLI_SERVICE, else <setup home>\service.json)
 *   colibri-tray.exe --dump-menu FILE      print the menu a state makes, as JSON, and exit   (for the tests)
 *   colibri-tray.exe --dump-icon KIND      print an icon's pixels as hex, and exit           (for the tests)
 *   colibri-tray.exe --status              print what the server is, as `coli service status --json` does, and exit
 *   colibri-tray.exe --print-config        print the settings and the file names in effect, and exit
 *   colibri-tray.exe --do ACTION           do what a menu item does (start, stop, restart, stop_force, restart_force, unload:KEY),
 *                                          with no window, print the message it would pop up, and exit   (for the tests)
 * COLI_TRAY_MUTEX=NAME makes the single-instance mutex another one, so a test can run beside the real tray.
 *
 * Build (MSYS2 UCRT64): make colibri-tray.exe
 */
#define _WIN32_WINNT 0x0601
#define WIN32_LEAN_AND_MEAN
#define UNICODE
#define _UNICODE
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <winhttp.h>
#include <shellapi.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "json.h"

/* ================================================================ small things */

typedef struct { char *p; size_t n, cap; } Str;

static void str_add(Str *s, const char *text, size_t len) {
    if (s->n + len + 1 > s->cap) {
        s->cap = (s->n + len + 1) * 2 + 64;
        s->p = (char *)realloc(s->p, s->cap);
    }
    memcpy(s->p + s->n, text, len);
    s->n += len;
    s->p[s->n] = 0;
}
static void str_cat(Str *s, const char *text) { str_add(s, text, strlen(text)); }
static void str_fmt(Str *s, const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    str_cat(s, buf);
}

static wchar_t *wide(const char *utf8) {                       /* malloc'd; free it */
    int n = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    wchar_t *w = (wchar_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(wchar_t));
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, utf8, -1, w, n); else w[0] = 0;
    return w;
}
static void narrow(const wchar_t *w, char *out, size_t cap) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, out, (int)cap, NULL, NULL);
    if (n <= 0) out[0] = 0;
}

static void out_write(const char *text, size_t len) {          /* stdout, also from a program with no console of its own */
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD done;
    if (h && h != INVALID_HANDLE_VALUE) WriteFile(h, text, (DWORD)len, &done, NULL);
}
static void say(const char *fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0) out_write(buf, (size_t)(n < (int)sizeof buf ? n : (int)sizeof buf - 1));
}

static void json_quote(Str *s, const char *t) {
    str_cat(s, "\"");
    for (; *t; t++) {
        unsigned char c = (unsigned char)*t;
        if (c == '"' || c == '\\') { char e[3] = { '\\', (char)c, 0 }; str_cat(s, e); }
        else if (c == '\n') str_cat(s, "\\n");
        else if (c == '\r') str_cat(s, "\\r");
        else if (c == '\t') str_cat(s, "\\t");
        else if (c < 0x20) str_fmt(s, "\\u%04x", c);
        else { char e[2] = { (char)c, 0 }; str_cat(s, e); }
    }
    str_cat(s, "\"");
}

static char *read_file(const char *path_utf8) {                /* the whole file, NUL-terminated, or NULL */
    wchar_t *w = wide(path_utf8);
    HANDLE h = CreateFileW(w, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
    free(w);
    if (h == INVALID_HANDLE_VALUE) return NULL;
    DWORD size = GetFileSize(h, NULL), got = 0;
    char *buf = NULL;
    if (size != INVALID_FILE_SIZE && size < (64u << 20)) {
        buf = (char *)malloc((size_t)size + 1);
        if (!ReadFile(h, buf, size, &got, NULL)) got = 0;
        buf[got] = 0;
    }
    CloseHandle(h);
    return buf;
}

static int file_exists(const char *path_utf8) {
    wchar_t *w = wide(path_utf8);
    DWORD a = GetFileAttributesW(w);
    free(w);
    return a != INVALID_FILE_ATTRIBUTES;
}

static void getenv_utf8(const wchar_t *name, char *out, size_t cap) {
    wchar_t buf[4096];
    DWORD n = GetEnvironmentVariableW(name, buf, 4096);
    if (n == 0 || n >= 4096) { out[0] = 0; return; }
    narrow(buf, out, cap);
}

/* ~ as Python's expanduser does it on Windows (USERPROFILE), then an absolute path with backslashes */
static void abspath_user(const char *path, char *out, size_t cap) {
    char expanded[2048];
    if (path[0] == '~' && (path[1] == 0 || path[1] == '/' || path[1] == '\\')) {
        char home[1024];
        getenv_utf8(L"USERPROFILE", home, sizeof home);
        snprintf(expanded, sizeof expanded, "%s%s", home, path + 1);
    } else {
        snprintf(expanded, sizeof expanded, "%s", path);
    }
    wchar_t *w = wide(expanded);
    wchar_t full[2048];
    DWORD n = GetFullPathNameW(w, 2048, full, NULL);
    free(w);
    if (n == 0 || n >= 2048) snprintf(out, cap, "%s", expanded); else narrow(full, out, cap);
}

/* ================================================================ what to show (no window needed) */

enum { K_STOPPED, K_STARTING, K_IDLE, K_BUSY, K_ATTENTION, K_COUNT };
static const char *KIND_NAME[K_COUNT] = { "stopped", "starting", "idle", "busy", "attention" };
static const int KIND_RGB[K_COUNT][3] = { {138, 143, 152}, {232, 160, 32}, {46, 170, 90}, {52, 120, 230}, {222, 70, 60} };

typedef struct {
    char id[160], key[160], state[24];
    double resident;                   /* resident_bytes, 0 when not said */
    int inflight, has_expires;
    double expires_in;
} Worker;

#define MAX_WORKERS 32
typedef struct {
    int valid;                         /* 0: not looked at yet (Python's None) */
    char state[16];                    /* stopped | running | unreachable */
    int key_ok, busy, nloaded;
    Worker loaded[MAX_WORKERS];
} Now;

typedef struct {
    int kind;
    char title[256], tip[256];
    int running, stopped, pending, unsure, busy;
    Now now;
} Summary;

static void plural(char *out, size_t cap, long count, const char *word) {
    snprintf(out, cap, "%ld %s%s", count, word, count == 1 ? "" : "s");
}
static void gb_text(char *out, size_t cap, double size) {
    if (size) snprintf(out, cap, "%.1f GB", size / 1073741824.0); else out[0] = 0;
}
static void duration_text(char *out, size_t cap, double seconds) {
    long s = (long)seconds;
    if (s >= 3600) snprintf(out, cap, "%ldh %02ldm", s / 3600, s % 3600 / 60);
    else if (s >= 60) snprintf(out, cap, "%ldm %02lds", s / 60, s % 60);
    else snprintf(out, cap, "%lds", s);
}

static void summarize(Summary *sm, const Now *now, const char *pending, int port) {
    memset(sm, 0, sizeof *sm);
    if (now && now->valid) sm->now = *now;
    const Now *n = now && now->valid ? now : NULL;
    int starting = 0, ready = 0;
    if (n) for (int i = 0; i < n->nloaded; i++) {
        if (!strcmp(n->loaded[i].state, "starting")) starting = 1;
        if (!strcmp(n->loaded[i].state, "ready")) ready++;
    }
    char tmp[64];
    if (pending && pending[0]) { sm->kind = K_STARTING; snprintf(sm->title, sizeof sm->title, "%s...", pending); }
    else if (!n) { sm->kind = K_STARTING; snprintf(sm->title, sizeof sm->title, "Looking..."); }
    else if (!strcmp(n->state, "stopped")) { sm->kind = K_STOPPED; snprintf(sm->title, sizeof sm->title, "Stopped"); }
    else if (!strcmp(n->state, "unreachable")) { sm->kind = K_ATTENTION; snprintf(sm->title, sizeof sm->title, "Port %d is used by another program", port); }
    else if (!n->key_ok) { sm->kind = K_ATTENTION; snprintf(sm->title, sizeof sm->title, "Running, but it does not accept the saved key"); }
    else if (starting) { sm->kind = K_BUSY; snprintf(sm->title, sizeof sm->title, "Loading a model"); }
    else if (n->busy) { sm->kind = K_BUSY; plural(tmp, sizeof tmp, n->busy, "request"); snprintf(sm->title, sizeof sm->title, "Working: %s", tmp); }
    else { sm->kind = K_IDLE; snprintf(sm->title, sizeof sm->title, "Running, idle"); }
    char lower[256];
    snprintf(lower, sizeof lower, "%s", sm->title);
    if (lower[0] >= 'A' && lower[0] <= 'Z') lower[0] = (char)(lower[0] + 32);
    snprintf(sm->tip, sizeof sm->tip, "Colibri: %s", lower);
    if (ready && (sm->kind == K_IDLE || sm->kind == K_BUSY)) {
        plural(tmp, sizeof tmp, ready, "model");
        size_t at = strlen(sm->tip);
        snprintf(sm->tip + at, sizeof sm->tip - at, " (%s loaded)", tmp);
    }
    sm->tip[120] = 0;                                          /* the notification area keeps 127 characters */
    sm->busy = n ? n->busy : 0;
    sm->running = n && !strcmp(n->state, "running");
    sm->stopped = n && !strcmp(n->state, "stopped");
    sm->pending = pending && pending[0];
    sm->unsure = n && (!strcmp(n->state, "unreachable") || (!strcmp(n->state, "running") && !n->key_ok));
}

typedef struct { int sep, has_id, enabled, checked; char id[200], label[320]; } MenuItem;
#define MAX_MENU 64

static MenuItem *menu_push(MenuItem *m, int *n) { MenuItem *it = &m[(*n)++]; memset(it, 0, sizeof *it); return it; }
static void menu_sep(MenuItem *m, int *n) { menu_push(m, n)->sep = 1; }
static void menu_add(MenuItem *m, int *n, const char *id, const char *label, int enabled, int checked) {
    MenuItem *it = menu_push(m, n);
    if (id) { it->has_id = 1; snprintf(it->id, sizeof it->id, "%s", id); }
    snprintf(it->label, sizeof it->label, "%s", label);
    it->enabled = enabled;
    it->checked = checked;
}

static int build_menu(MenuItem *m, const Summary *sm, int lan, int autostart, int log) {
    int n = 0;
    char label[320], tmp[96], part[96], id[200];
    snprintf(label, sizeof label, "Colibri: %s", sm->title);
    menu_add(m, &n, NULL, label, 0, 0);
    menu_sep(m, &n);
    int shown = 0;
    for (int i = 0; i < sm->now.nloaded; i++) {
        const Worker *w = &sm->now.loaded[i];
        int starting = !strcmp(w->state, "starting"), ready = !strcmp(w->state, "ready");
        if (!starting && !ready) continue;
        shown++;
        if (starting) {
            snprintf(label, sizeof label, "%s: loading...", w->id);
            menu_add(m, &n, NULL, label, 0, 0);
            continue;
        }
        char gbs[32], extra[64];
        gb_text(gbs, sizeof gbs, w->resident);
        extra[0] = 0;
        if (w->inflight) snprintf(extra, sizeof extra, "in use");
        else if (w->has_expires) { duration_text(tmp, sizeof tmp, w->expires_in); snprintf(extra, sizeof extra, "unloads in %s", tmp); }
        part[0] = 0;
        if (gbs[0] && extra[0]) snprintf(part, sizeof part, "%s, %s", gbs, extra);
        else if (gbs[0]) snprintf(part, sizeof part, "%s", gbs);
        else snprintf(part, sizeof part, "%s", extra);
        if (part[0]) snprintf(label, sizeof label, "Unload %s (%s)", w->id, part);
        else snprintf(label, sizeof label, "Unload %s", w->id);
        snprintf(id, sizeof id, "unload:%s", w->key);
        menu_add(m, &n, id, label, 1, 0);
    }
    if (!shown) menu_add(m, &n, NULL, "No model loaded", 0, 0);
    menu_sep(m, &n);
    menu_add(m, &n, "open", "Open the dashboard", sm->running, 0);
    menu_add(m, &n, "copy_lan", "Copy the network link", lan, 0);
    menu_add(m, &n, "copy_key", "Copy the API key", 1, 0);
    menu_sep(m, &n);
    int idle = !sm->pending;
    if (sm->stopped) {
        menu_add(m, &n, "start", "Start the server", idle, 0);
    } else if (sm->unsure) {
        menu_add(m, &n, "restart_force", "Restart anyway (it does not say whether it is working)", idle, 0);
        menu_add(m, &n, "stop_force", "Stop anyway (it does not say whether it is working)", idle, 0);
    } else if (sm->busy) {
        plural(tmp, sizeof tmp, sm->busy, "request");
        snprintf(label, sizeof label, "Restart anyway (%s running will be lost)", tmp);
        menu_add(m, &n, "restart_force", label, idle, 0);
        snprintf(label, sizeof label, "Stop anyway (%s running will be lost)", tmp);
        menu_add(m, &n, "stop_force", label, idle, 0);
    } else {
        menu_add(m, &n, "restart", "Restart the server", idle && sm->kind != K_STARTING, 0);
        menu_add(m, &n, "stop", "Stop the server", idle, 0);
    }
    menu_sep(m, &n);
    menu_add(m, &n, "request_log", "Open the request log", log, 0);
    menu_add(m, &n, "logs", "Open the folder with the server's logs", 1, 0);
    menu_add(m, &n, "autostart", "Start with Windows", 1, autostart);
    menu_sep(m, &n);
    menu_add(m, &n, "quit", "Quit the tray (the server keeps running)", 1, 0);
    return n;
}

/* `size` x `size` premultiplied BGRA, top row first: a ring when stopped, a disc otherwise, with a white bar and dot when
 * something is wrong (so the state is not told by colour alone). */
static void icon_pixels(int kind, int size, unsigned char *out) {
    int red = KIND_RGB[kind][0], green = KIND_RGB[kind][1], blue = KIND_RGB[kind][2];
    double center = (size - 1) / 2.0, radius = size / 2.0 - 1.5;
    double thickness = fmax(2.5, size / 9.0);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            double d = hypot(x - center, y - center);
            double cover = fmin(1.0, fmax(0.0, radius - d + 0.5));
            if (kind == K_STOPPED) cover = fmin(cover, fmin(1.0, fmax(0.0, d - (radius - thickness) + 0.5)));
            int r = red, g = green, b = blue;
            if (kind == K_ATTENTION) {
                int bar = fabs(x - center) < size / 14.0 && size * 0.22 <= y && y <= size * 0.62;
                int dot = fabs(x - center) < size / 14.0 && size * 0.72 <= y && y <= size * 0.82;
                if (bar || dot) r = g = b = 255;
            }
            unsigned char *p = out + ((size_t)y * size + x) * 4;
            p[0] = (unsigned char)(int)(b * cover);
            p[1] = (unsigned char)(int)(g * cover);
            p[2] = (unsigned char)(int)(r * cover);
            p[3] = (unsigned char)(int)(255 * cover);
        }
    }
}

/* ================================================================ settings (service.json) */

typedef struct {
    int port, router, lan, start_with_tray;
    char key_path[1100], out_path[1100], err_path[1100], request_log[1100];    /* request_log "" when off */
    char file[1100];                                           /* the settings file asked for (--config), or "" */
} Cfg;

static void config_file_path(const char *explicit_file, char *out, size_t cap) {
    char v[1024], home[1024];
    if (explicit_file && explicit_file[0]) { abspath_user(explicit_file, out, cap); return; }
    getenv_utf8(L"COLI_SERVICE", v, sizeof v);
    if (v[0]) { abspath_user(v, out, cap); return; }
    getenv_utf8(L"COLI_SETUP_HOME", v, sizeof v);
    if (v[0]) { abspath_user(v, home, sizeof home); }
    else {
        getenv_utf8(L"LOCALAPPDATA", v, sizeof v);
        if (!v[0]) { getenv_utf8(L"USERPROFILE", v, sizeof v); snprintf(v + strlen(v), sizeof v - strlen(v), "\\AppData\\Local"); }
        snprintf(home, sizeof home, "%s\\colibri", v);
    }
    snprintf(out, cap, "%s\\service.json", home);
}

static const char *KNOWN_KEYS[] = { "port", "router", "model", "keep_alive", "kv_slots", "cap", "ctx", "lan", "request_log",
                                    "files_dir", "start_with_tray", "extra_args", "env", NULL };

/* 0 and a message when the file cannot be used; the same refusals service.load_config makes for the settings read here */
static int load_cfg(Cfg *c, const char *explicit_file, char *err, size_t errcap) {
    memset(c, 0, sizeof *c);
    c->port = 8000; c->router = 1; c->lan = 1; c->start_with_tray = 1;
    char files_dir[1024] = "~", request_log[1024] = "~/colibri-requests.log";
    char path[1100];
    config_file_path(explicit_file, path, sizeof path);
    if (explicit_file && explicit_file[0]) snprintf(c->file, sizeof c->file, "%s", path);
    char *text = read_file(path);
    if (text) {
        jval *root = json_parse_checked(text);
        free(text);
        if (!root || root->t != J_OBJ) {
            snprintf(err, errcap, "%s cannot be read: it must hold a JSON object", path);
            json_free(root);
            return 0;
        }
        for (int i = 0; i < root->len; i++) {
            int known = 0;
            for (int k = 0; KNOWN_KEYS[k]; k++) if (!strcmp(root->keys[i], KNOWN_KEYS[k])) known = 1;
            if (!known) { snprintf(err, errcap, "%s: unknown setting '%s'", path, root->keys[i]); json_free(root); return 0; }
        }
        jval *v;
        if ((v = json_get(root, "port"))) {
            if (v->t != J_NUM || v->num != floor(v->num)) { snprintf(err, errcap, "setting 'port' must be an int"); json_free(root); return 0; }
            if (v->num < 1 || v->num > 65535) { snprintf(err, errcap, "setting 'port' must be 1-65535, not %.0f", v->num); json_free(root); return 0; }
            c->port = (int)v->num;
        }
        if ((v = json_get(root, "router")) && v->t == J_BOOL) c->router = v->boolean;
        if ((v = json_get(root, "lan")) && v->t == J_BOOL) c->lan = v->boolean;
        if ((v = json_get(root, "start_with_tray")) && v->t == J_BOOL) c->start_with_tray = v->boolean;
        if ((v = json_get(root, "files_dir")) && v->t == J_STR) snprintf(files_dir, sizeof files_dir, "%s", v->str);
        if ((v = json_get(root, "request_log")) && v->t == J_STR) snprintf(request_log, sizeof request_log, "%s", v->str);
        json_free(root);
    }
    char home[1024], suffix[24] = "";
    abspath_user(files_dir, home, sizeof home);
    if (c->port != 8000) snprintf(suffix, sizeof suffix, "-%d", c->port);
    snprintf(c->key_path, sizeof c->key_path, "%s\\colibri-api-key%s.txt", home, suffix);
    snprintf(c->out_path, sizeof c->out_path, "%s\\colibri-serve%s.log", home, suffix);
    snprintf(c->err_path, sizeof c->err_path, "%s\\colibri-serve%s.err", home, suffix);
    if (request_log[0]) abspath_user(request_log, c->request_log, sizeof c->request_log);
    return 1;
}

static void read_key(const Cfg *c, char *key, size_t cap) {
    char *t = read_file(c->key_path);
    key[0] = 0;
    if (!t) return;
    snprintf(key, cap, "%s", t);
    free(t);
    size_t n = strlen(key);
    while (n && (key[n - 1] == '\n' || key[n - 1] == '\r' || key[n - 1] == ' ' || key[n - 1] == '\t')) key[--n] = 0;
}

/* ================================================================ asking the server */

static HINTERNET g_session;                                    /* one for the program, made before any thread runs */

static void http_open(void) {
    g_session = WinHttpOpen(L"colibri-tray", WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (g_session) WinHttpSetTimeouts(g_session, 3000, 3000, 3000, 3000);
}

/* the HTTP status (0 when nothing answered) and the body (malloc'd, NUL-terminated; free it) */
static int http_call(int port, const char *method, const char *path, const char *key, const char *body, char **out) {
    int status = 0;
    *out = NULL;
    if (!g_session) return 0;
    HINTERNET con = WinHttpConnect(g_session, L"127.0.0.1", (INTERNET_PORT)port, 0), req = NULL;
    if (!con) return 0;
    wchar_t *wm = wide(method), *wp = wide(path);
    req = WinHttpOpenRequest(con, wm, wp, NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
    free(wm); free(wp);
    if (req) {
        char head[1024];
        snprintf(head, sizeof head, "%s%s%s", key && key[0] ? "Authorization: Bearer " : "", key && key[0] ? key : "",
                 body ? (key && key[0] ? "\r\nContent-Type: application/json" : "Content-Type: application/json") : "");
        if (head[0]) { wchar_t *wh = wide(head); WinHttpAddRequestHeaders(req, wh, (DWORD)-1, WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE); free(wh); }
        DWORD blen = body ? (DWORD)strlen(body) : 0;
        if (WinHttpSendRequest(req, WINHTTP_NO_ADDITIONAL_HEADERS, 0, (LPVOID)body, blen, blen, 0) && WinHttpReceiveResponse(req, NULL)) {
            DWORD code = 0, size = sizeof code;
            WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &code, &size, WINHTTP_NO_HEADER_INDEX);
            status = (int)code;
            Str s = { 0 };
            char buf[8192];
            DWORD got;
            while (WinHttpReadData(req, buf, sizeof buf, &got) && got) { str_add(&s, buf, got); if (s.n > (32u << 20)) break; }
            if (!s.p) str_cat(&s, "");
            *out = s.p;
        }
        WinHttpCloseHandle(req);
    }
    WinHttpCloseHandle(con);
    return status;
}

static int something_listens(int port) {                       /* a TCP connection to this PC's port is taken */
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) return 0;
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((u_short)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    int ok = connect(s, (struct sockaddr *)&a, sizeof a) == 0;
    closesocket(s);
    return ok;
}

static long num_field(jval *o, const char *key) {              /* int(x or 0) */
    jval *v = json_get(o, key);
    return v && v->t == J_NUM ? (long)v->num : 0;
}

static void copy_str(char *dst, size_t cap, jval *o, const char *key) {
    jval *v = json_get(o, key);
    snprintf(dst, cap, "%s", v && v->t == J_STR ? v->str : "");
}

/* service.status: stopped (nothing listens), running, or unreachable (something listens and does not answer /health) */
static void server_status(const Cfg *c, Now *now) {
    memset(now, 0, sizeof *now);
    now->valid = 1;
    now->key_ok = 1;
    snprintf(now->state, sizeof now->state, "stopped");
    char key[256], *body;
    read_key(c, key, sizeof key);
    int code = http_call(c->port, "GET", "/health", key, NULL, &body);
    if (!code) {
        if (something_listens(c->port)) snprintf(now->state, sizeof now->state, "unreachable");
        return;
    }
    snprintf(now->state, sizeof now->state, "running");
    jval *health = code == 200 && body ? json_parse_checked(body) : NULL;
    if (health && health->t == J_OBJ) {
        jval *router = json_get(health, "router");
        int has_router = router && router->t == J_OBJ;
        now->key_ok = has_router || !c->router;                /* an unauthenticated /health has no router part */
        long busy = 0;
        jval *sched = json_get(health, "scheduler");
        if (sched && sched->t == J_OBJ) busy += num_field(sched, "active") + num_field(sched, "queued");
        if (has_router) {
            jval *gate = json_get(router, "gate");
            if (gate && gate->t == J_OBJ) busy += num_field(gate, "active") + num_field(gate, "queued");
            jval *list = json_get(router, "loaded");
            if (list && list->t == J_ARR) for (int i = 0; i < list->len; i++) {
                jval *w = list->kids[i];
                if (!w || w->t != J_OBJ) continue;
                busy += num_field(w, "inflight");
                if (now->nloaded < MAX_WORKERS) {
                    Worker *o = &now->loaded[now->nloaded++];
                    memset(o, 0, sizeof *o);
                    copy_str(o->id, sizeof o->id, w, "id");
                    copy_str(o->key, sizeof o->key, w, "key");
                    copy_str(o->state, sizeof o->state, w, "state");
                    jval *r = json_get(w, "resident_bytes");
                    o->resident = r && r->t == J_NUM ? r->num : 0;
                    o->inflight = (int)num_field(w, "inflight");
                    jval *e = json_get(w, "expires_in");
                    o->has_expires = e && e->t == J_NUM;
                    o->expires_in = o->has_expires ? e->num : 0;
                }
            }
        }
        now->busy = (int)busy;
    }
    json_free(health);
    free(body);
}

static void print_now(const Now *now) {                        /* the shape of `coli service status --json` (and tray.py's `now`) */
    Str s = { 0 };
    str_cat(&s, "{\"state\": "); json_quote(&s, now->state);
    str_fmt(&s, ", \"key_ok\": %s, \"busy\": %d, \"loaded\": [", now->key_ok ? "true" : "false", now->busy);
    for (int i = 0; i < now->nloaded; i++) {
        const Worker *w = &now->loaded[i];
        if (i) str_cat(&s, ", ");
        str_cat(&s, "{\"id\": "); json_quote(&s, w->id);
        str_cat(&s, ", \"key\": "); json_quote(&s, w->key);
        str_cat(&s, ", \"state\": "); json_quote(&s, w->state);
        str_fmt(&s, ", \"resident_bytes\": %.0f, \"inflight\": %d, \"expires_in\": ", w->resident, w->inflight);
        if (w->has_expires) str_fmt(&s, "%.6g", w->expires_in); else str_cat(&s, "null");
        str_cat(&s, "}");
    }
    str_cat(&s, "]}\n");
    out_write(s.p, s.n);
    free(s.p);
}

/* ================================================================ the Win32 part */

#define WM_ICON (WM_APP + 1)
#define WM_STATE (WM_APP + 2)
#define WM_TOAST (WM_APP + 3)
#define ID_MENU_BASE 1000
#define RUN_KEY L"Software\\Microsoft\\Windows\\CurrentVersion\\Run"
#define RUN_VALUE L"Colibri tray"
#define MAX_TOASTS 8

static struct {
    HWND hwnd;
    Cfg cfg;
    char python[1024], exe[1024], exe_dir[1024];
    Now now;
    Summary summary;
    char pending[16];
    CRITICAL_SECTION lock;
    HICON icons[K_COUNT];
    volatile LONG quitting, expected_stop, busy_action;
    int cli;                           /* --do: messages go to stdout, there is no window */
    char toasts[MAX_TOASTS][256];
    int ntoasts;
    UINT taskbar_created;
} G;

static void toast(const char *text) {                          /* from any thread */
    if (G.cli) { say("%s\n", text); return; }
    EnterCriticalSection(&G.lock);
    if (G.ntoasts < MAX_TOASTS) snprintf(G.toasts[G.ntoasts++], sizeof G.toasts[0], "%s", text);
    LeaveCriticalSection(&G.lock);
    if (G.hwnd) PostMessageW(G.hwnd, WM_TOAST, 0, 0);
}

static HICON icon_for(int kind) {
    if (!G.icons[kind]) {
        unsigned char pixels[32 * 32 * 4], mask[128] = { 0 };
        icon_pixels(kind, 32, pixels);
        HBITMAP color = CreateBitmap(32, 32, 1, 32, pixels), m = CreateBitmap(32, 32, 1, 1, mask);
        ICONINFO info = { TRUE, 0, 0, m, color };
        G.icons[kind] = CreateIconIndirect(&info);
        DeleteObject(color);
        DeleteObject(m);
    }
    return G.icons[kind];
}

static void nid_init(NOTIFYICONDATAW *d, UINT flags) {
    memset(d, 0, sizeof *d);
    d->cbSize = sizeof *d;
    d->hWnd = G.hwnd;
    d->uID = 1;
    d->uFlags = flags;
}

static void tip_set(NOTIFYICONDATAW *d, const char *tip) {
    wchar_t *w = wide(tip);
    wcsncpy(d->szTip, w, 127);
    free(w);
}

static BOOL icon_add(void) {
    NOTIFYICONDATAW d;
    nid_init(&d, NIF_MESSAGE | NIF_ICON | NIF_TIP);
    d.uCallbackMessage = WM_ICON;
    d.hIcon = icon_for(G.summary.kind);
    tip_set(&d, G.summary.tip);
    return Shell_NotifyIconW(NIM_ADD, &d);
}

static void icon_update(void) {
    NOTIFYICONDATAW d;
    nid_init(&d, NIF_ICON | NIF_TIP);
    d.hIcon = icon_for(G.summary.kind);
    tip_set(&d, G.summary.tip);
    Shell_NotifyIconW(NIM_MODIFY, &d);
}

static void balloon(const char *text) {
    NOTIFYICONDATAW d;
    nid_init(&d, NIF_INFO);
    wchar_t *t = wide(text), *title = wide("Colibri");
    wcsncpy(d.szInfo, t, 255);
    wcsncpy(d.szInfoTitle, title, 63);
    d.dwInfoFlags = NIIF_INFO;
    free(t); free(title);
    Shell_NotifyIconW(NIM_MODIFY, &d);
}

/* ---- the Run key */

static void autostart_command(char *out, size_t cap) {
    char extra[2400] = "";
    if (G.python[0]) snprintf(extra + strlen(extra), sizeof extra - strlen(extra), " --python \"%s\"", G.python);
    if (G.cfg.file[0]) snprintf(extra + strlen(extra), sizeof extra - strlen(extra), " --config \"%s\"", G.cfg.file);
    snprintf(out, cap, "\"%s\"%s", G.exe, extra);
}
static int autostart_get(void) {
    HKEY k;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, RUN_KEY, 0, KEY_QUERY_VALUE, &k) != ERROR_SUCCESS) return 0;
    LONG r = RegQueryValueExW(k, RUN_VALUE, NULL, NULL, NULL, NULL);
    RegCloseKey(k);
    return r == ERROR_SUCCESS;
}
static int autostart_set(int on) {
    HKEY k;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, RUN_KEY, 0, KEY_SET_VALUE, &k) != ERROR_SUCCESS) return 0;
    LONG r;
    if (on) {
        char cmd[3000];
        autostart_command(cmd, sizeof cmd);
        wchar_t *w = wide(cmd);
        r = RegSetValueExW(k, RUN_VALUE, 0, REG_SZ, (const BYTE *)w, (DWORD)((wcslen(w) + 1) * sizeof(wchar_t)));
        free(w);
    } else {
        r = RegDeleteValueW(k, RUN_VALUE);
        if (r == ERROR_FILE_NOT_FOUND) r = ERROR_SUCCESS;
    }
    RegCloseKey(k);
    return r == ERROR_SUCCESS;
}

/* ---- running `coli service` */

static int python_found(void) {
    if (G.python[0]) return 1;
    char v[1024];
    getenv_utf8(L"COLI_PYTHON", v, sizeof v);
    if (v[0]) { snprintf(G.python, sizeof G.python, "%s", v); return 1; }
    wchar_t found[1024];
    DWORD n = SearchPathW(NULL, L"python.exe", NULL, 1024, found, NULL);
    if (n && n < 1024 && !wcsstr(found, L"\\WindowsApps\\")) {          /* the Store's stub opens the Store, not Python */
        narrow(found, G.python, sizeof G.python);
        return 1;
    }
    return 0;
}

/* Runs `python coli service ARGS`, waits, and gives the last line it printed. Returns its exit code, or -1 when it could not run. */
static int run_service(const char *args, char *last, size_t cap) {
    last[0] = 0;
    if (!python_found()) { snprintf(last, cap, "Python was not found: give the tray --python, or set COLI_PYTHON."); return -1; }
    char cmd[4096], config[1100] = "";
    if (G.cfg.file[0]) snprintf(config, sizeof config, " --config \"%s\"", G.cfg.file);
    snprintf(cmd, sizeof cmd, "\"%s\" \"%s\\coli\" service %s%s", G.python, G.exe_dir, args, config);
    SECURITY_ATTRIBUTES sa = { sizeof sa, NULL, TRUE };
    HANDLE rd = NULL, wr = NULL;
    if (!CreatePipe(&rd, &wr, &sa, 0)) { snprintf(last, cap, "Could not make a pipe."); return -1; }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si;
    memset(&si, 0, sizeof si);
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = si.hStdError = wr;
    si.hStdInput = INVALID_HANDLE_VALUE;
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, NULL);
    si.hStdInput = nul;
    wchar_t *wc = wide(cmd);
    PROCESS_INFORMATION pi;
    wchar_t dirw[1024];
    wchar_t *d = wide(G.exe_dir);
    wcsncpy(dirw, d, 1023); dirw[1023] = 0; free(d);
    BOOL ok = CreateProcessW(NULL, wc, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, dirw, &si, &pi);
    free(wc);
    CloseHandle(wr);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) { CloseHandle(rd); snprintf(last, cap, "Could not run %s (error %lu).", G.python, GetLastError()); return -1; }
    Str out = { 0 };
    char buf[2048];
    DWORD got;
    while (ReadFile(rd, buf, sizeof buf, &got, NULL) && got) { if (out.n < 65536) str_add(&out, buf, got); }
    CloseHandle(rd);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (out.p) {                                               /* the last line with something in it */
        size_t end = out.n;
        while (end && (out.p[end - 1] == '\n' || out.p[end - 1] == '\r' || out.p[end - 1] == ' ')) end--;
        size_t start = end;
        while (start && out.p[start - 1] != '\n') start--;
        size_t len = end - start;
        const char *line = out.p + start;
        if (len > 14 && !strncmp(line, "coli service: ", 14)) { line += 14; len -= 14; }
        if (len >= cap) len = cap - 1;
        memcpy(last, line, len);
        last[len] = 0;
    }
    free(out.p);
    return (int)code;
}

/* ---- refreshing and acting */

static void set_summary(void) {                                /* from any thread, with G.now and G.pending as they are */
    EnterCriticalSection(&G.lock);
    int was_running = G.summary.running;
    summarize(&G.summary, &G.now, G.pending, G.cfg.port);
    int stopped_now = G.summary.stopped;
    LeaveCriticalSection(&G.lock);
    if (G.hwnd) PostMessageW(G.hwnd, WM_STATE, 0, 0);
    if (was_running && stopped_now && !G.expected_stop && !G.quitting) {
        char dir[1024], text[1200];
        snprintf(dir, sizeof dir, "%s", G.cfg.err_path);
        char *slash = strrchr(dir, '\\');
        if (slash) *slash = 0;
        snprintf(text, sizeof text, "The server stopped. Its log is in %s.", dir);
        toast(text);
    }
}

static void refresh(int poll) {
    if (poll) {
        Now n;
        server_status(&G.cfg, &n);
        EnterCriticalSection(&G.lock);
        G.now = n;
        LeaveCriticalSection(&G.lock);
    }
    set_summary();
}

static void set_pending(const char *word) {
    EnterCriticalSection(&G.lock);
    snprintf(G.pending, sizeof G.pending, "%s", word ? word : "");
    LeaveCriticalSection(&G.lock);
    refresh(0);
}

typedef struct { char action[220]; } ActionArg;

static DWORD WINAPI server_action(LPVOID param) {
    ActionArg *arg = (ActionArg *)param;
    const char *action = arg->action;
    char last[512];
    if (!strncmp(action, "unload:", 7)) {
        char key[256], *body = NULL, req[400];
        read_key(&G.cfg, key, sizeof key);
        Str q = { 0 };
        str_cat(&q, "{\"key\": "); json_quote(&q, action + 7); str_cat(&q, "}");
        snprintf(req, sizeof req, "%s", q.p);
        free(q.p);
        int code = http_call(G.cfg.port, "POST", "/v1/models/unload", key, req, &body);
        char status[64] = "";
        if ((code == 200 || code == 202) && body) {
            jval *j = json_parse_checked(body);
            jval *s = json_get(j, "status");
            if (s && s->t == J_STR) snprintf(status, sizeof status, "%s", s->str);
            json_free(j);
        }
        free(body);
        toast(!strcmp(status, "unloaded") ? "The model is unloaded." :
              !strcmp(status, "unloading when idle") ? "It is answering a request, and will unload when that is done." :
              !strcmp(status, "not loaded") ? "It was not loaded." : "The router did not say.");
        refresh(1);
        free(arg);
        InterlockedExchange(&G.busy_action, 0);
        return 0;
    }
    int force = strstr(action, "_force") != NULL;
    char name[32];
    snprintf(name, sizeof name, "%s", action);
    char *us = strstr(name, "_force");
    if (us) *us = 0;
    const char *word = !strcmp(name, "start") ? "Starting" : !strcmp(name, "stop") ? "Stopping" : !strcmp(name, "restart") ? "Restarting" : NULL;
    set_pending(word);
    char args[128];
    const char *done;
    if (!strcmp(name, "start")) { snprintf(args, sizeof args, "start --wait 300"); done = "The server is up."; }
    else if (!strcmp(name, "stop")) { snprintf(args, sizeof args, "stop%s", force ? " --force" : ""); done = "The server is stopped."; InterlockedExchange(&G.expected_stop, 1); }
    else { snprintf(args, sizeof args, "restart --wait 300%s", force ? " --force" : ""); done = "The server is up again."; InterlockedExchange(&G.expected_stop, 1); }
    int code = run_service(args, last, sizeof last);
    toast(code == 0 ? done : last[0] ? last : "It did not work.");
    set_pending(NULL);
    refresh(1);                                                /* look again before drawing: the last look is from before the action */
    InterlockedExchange(&G.expected_stop, 0);
    free(arg);
    InterlockedExchange(&G.busy_action, 0);
    return 0;
}

static void start_action(const char *action) {
    ActionArg *a = (ActionArg *)calloc(1, sizeof *a);
    snprintf(a->action, sizeof a->action, "%s", action);
    HANDLE h = CreateThread(NULL, 0, server_action, a, 0, NULL);
    if (h) CloseHandle(h); else free(a);
}

static DWORD WINAPI watcher(LPVOID unused) {
    (void)unused;
    while (!G.quitting) {
        refresh(1);
        for (int i = 0; i < 30 && !G.quitting; i++) Sleep(100);        /* every 3 s, and quits within 0.1 s */
    }
    return 0;
}

static DWORD WINAPI start_if_stopped(LPVOID unused) {
    (void)unused;
    refresh(1);
    int stopped;
    EnterCriticalSection(&G.lock);
    stopped = G.summary.stopped;
    LeaveCriticalSection(&G.lock);
    if (stopped) start_action("start");
    return 0;
}

/* ---- links, the clipboard, opening things */

static int lan_address(char *out, size_t cap) {
    SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) return 0;
    struct sockaddr_in a, me;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons(9);
    inet_pton(AF_INET, "192.0.2.1", &a.sin_addr);                  /* a documentation address: nothing is sent */
    int len = sizeof me, ok = 0;
    if (connect(s, (struct sockaddr *)&a, sizeof a) == 0 && getsockname(s, (struct sockaddr *)&me, &len) == 0) {
        inet_ntop(AF_INET, &me.sin_addr, out, (socklen_t)cap);
        ok = strncmp(out, "127.", 4) && strncmp(out, "0.", 2);
    }
    closesocket(s);
    return ok;
}

static void link_text(char *out, size_t cap, int lan) {
    char key[256], host[64] = "127.0.0.1";
    read_key(&G.cfg, key, sizeof key);
    if (lan && !(G.cfg.lan && lan_address(host, sizeof host))) { out[0] = 0; return; }
    if (key[0]) snprintf(out, cap, "http://%s:%d/#key=%s", host, G.cfg.port, key);
    else snprintf(out, cap, "http://127.0.0.1:%d/", G.cfg.port);
}

static void to_clipboard(const char *text) {
    wchar_t *w = wide(text);
    size_t bytes = (wcslen(w) + 1) * sizeof(wchar_t);
    HGLOBAL g = GlobalAlloc(GMEM_MOVEABLE, bytes);
    if (g && OpenClipboard(G.hwnd)) {
        memcpy(GlobalLock(g), w, bytes);
        GlobalUnlock(g);
        EmptyClipboard();
        SetClipboardData(CF_UNICODETEXT, g);
        CloseClipboard();
    } else if (g) GlobalFree(g);
    free(w);
}

static void shell_open(const char *target) {
    wchar_t *w = wide(target);
    HINSTANCE r = ShellExecuteW(NULL, L"open", w, NULL, NULL, SW_SHOWNORMAL);
    free(w);
    if ((INT_PTR)r <= 32) { char msg[1300]; snprintf(msg, sizeof msg, "Could not open %s.", target); toast(msg); }
}

static void act(const char *action) {
    if (!strcmp(action, "quit")) { DestroyWindow(G.hwnd); return; }
    if (!strcmp(action, "autostart")) {
        int on = !autostart_get();
        toast(autostart_set(on) ? (on ? "Colibri's tray will start when you sign in." : "Colibri's tray will not start at sign-in.")
                                : "Could not change it.");
    } else if (!strcmp(action, "open")) {
        char url[600];
        link_text(url, sizeof url, 0);
        shell_open(url);
    } else if (!strcmp(action, "copy_key") || !strcmp(action, "copy_lan")) {
        char key[256], text[600];
        read_key(&G.cfg, key, sizeof key);
        if (!strcmp(action, "copy_key")) snprintf(text, sizeof text, "%s", key); else link_text(text, sizeof text, 1);
        if (!key[0] || !text[0]) { toast("There is no key yet: start the server first."); return; }
        to_clipboard(text);
        toast(!strcmp(action, "copy_key") ? "The API key is on the clipboard." :
              "The link for other PCs is on the clipboard. It holds the key: share it like a password.");
    } else if (!strcmp(action, "request_log")) {
        shell_open(G.cfg.request_log);
    } else if (!strcmp(action, "logs")) {
        char dir[1024];
        snprintf(dir, sizeof dir, "%s", G.cfg.out_path);
        char *slash = strrchr(dir, '\\');
        if (slash) *slash = 0;
        shell_open(dir);
    } else {
        start_action(action);
    }
}

/* ---- the menu and the window */

static void show_menu(void) {
    Summary sm;
    EnterCriticalSection(&G.lock);
    sm = G.summary;
    LeaveCriticalSection(&G.lock);
    char host[64];
    MenuItem items[MAX_MENU];
    int log = G.cfg.request_log[0] && file_exists(G.cfg.request_log);
    int n = build_menu(items, &sm, G.cfg.lan && lan_address(host, sizeof host), autostart_get(), log);
    HMENU menu = CreatePopupMenu();
    for (int i = 0; i < n; i++) {
        if (items[i].sep) { AppendMenuW(menu, MF_SEPARATOR, 0, NULL); continue; }
        wchar_t *w = wide(items[i].label);
        AppendMenuW(menu, (items[i].enabled ? 0 : MF_GRAYED) | (items[i].checked ? MF_CHECKED : 0), (UINT_PTR)(ID_MENU_BASE + i), w);
        free(w);
    }
    POINT pt;
    GetCursorPos(&pt);
    SetForegroundWindow(G.hwnd);                               /* without it the menu stays open when you click elsewhere */
    int chosen = (int)TrackPopupMenu(menu, TPM_RETURNCMD | TPM_RIGHTBUTTON | TPM_BOTTOMALIGN, pt.x, pt.y, 0, G.hwnd, NULL);
    PostMessageW(G.hwnd, WM_NULL, 0, 0);
    DestroyMenu(menu);
    int i = chosen - ID_MENU_BASE;
    if (chosen >= ID_MENU_BASE && i < n && items[i].has_id) act(items[i].id);
}

static LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_ICON && (lp == WM_LBUTTONUP || lp == WM_RBUTTONUP)) { show_menu(); return 0; }
    if (msg == WM_STATE) { icon_update(); return 0; }
    if (msg == WM_TOAST) {
        char list[MAX_TOASTS][256];
        int n;
        EnterCriticalSection(&G.lock);
        n = G.ntoasts;
        memcpy(list, G.toasts, sizeof list);
        G.ntoasts = 0;
        LeaveCriticalSection(&G.lock);
        for (int i = 0; i < n; i++) balloon(list[i]);
        return 0;
    }
    if (G.taskbar_created && msg == G.taskbar_created) { icon_add(); return 0; }   /* Explorer restarted: its icons are gone */
    if (msg == WM_DESTROY) {
        InterlockedExchange(&G.quitting, 1);
        NOTIFYICONDATAW d;
        nid_init(&d, 0);
        Shell_NotifyIconW(NIM_DELETE, &d);
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

static int run_tray(void) {
    wchar_t mutex_name[160] = L"Local\\ColibriTray";           /* the one tray.py takes: only one of the two runs */
    wchar_t other[100];
    if (GetEnvironmentVariableW(L"COLI_TRAY_MUTEX", other, 100) > 0 && other[0]) swprintf(mutex_name, 160, L"Local\\ColibriTray-%ls", other);
    HANDLE mutex = CreateMutexW(NULL, FALSE, mutex_name);
    if (GetLastError() == ERROR_ALREADY_EXISTS) return 2;
    G.taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");
    HINSTANCE inst = GetModuleHandleW(NULL);
    WNDCLASSW cls;
    memset(&cls, 0, sizeof cls);
    cls.lpfnWndProc = wndproc;
    cls.hInstance = inst;
    cls.lpszClassName = L"ColibriTrayWindow";
    if (!RegisterClassW(&cls)) return 1;
    G.hwnd = CreateWindowExW(0, L"ColibriTrayWindow", L"Colibri", 0, 0, 0, 0, 0, NULL, NULL, inst, NULL);
    if (!G.hwnd) return 1;
    summarize(&G.summary, NULL, NULL, G.cfg.port);
    if (!icon_add()) {
        MessageBoxW(NULL, L"Windows would not add the icon to the notification area.", L"Colibri tray", MB_ICONERROR);
        return 1;
    }
    HANDLE t = CreateThread(NULL, 0, watcher, NULL, 0, NULL);
    if (t) CloseHandle(t);
    if (G.cfg.start_with_tray) { t = CreateThread(NULL, 0, start_if_stopped, NULL, 0, NULL); if (t) CloseHandle(t); }
    MSG m;
    while (GetMessageW(&m, NULL, 0, 0) > 0) { TranslateMessage(&m); DispatchMessageW(&m); }
    CloseHandle(mutex);
    return 0;
}

/* ================================================================ the command line */

static int dump_menu(const char *file) {
    char *text = read_file(file);
    jval *j = text ? json_parse_checked(text) : NULL;
    free(text);
    if (!j || j->t != J_OBJ) { say("cannot read %s\n", file); return 1; }
    Now now;
    memset(&now, 0, sizeof now);
    jval *n = json_get(j, "now");
    if (n && n->t == J_OBJ) {
        now.valid = 1;
        copy_str(now.state, sizeof now.state, n, "state");
        jval *k = json_get(n, "key_ok");
        now.key_ok = k && k->t == J_BOOL ? k->boolean : 1;
        now.busy = (int)num_field(n, "busy");
        jval *list = json_get(n, "loaded");
        if (list && list->t == J_ARR) for (int i = 0; i < list->len && now.nloaded < MAX_WORKERS; i++) {
            jval *w = list->kids[i];
            if (!w || w->t != J_OBJ) continue;
            Worker *o = &now.loaded[now.nloaded++];
            memset(o, 0, sizeof *o);
            copy_str(o->id, sizeof o->id, w, "id");
            copy_str(o->key, sizeof o->key, w, "key");
            copy_str(o->state, sizeof o->state, w, "state");
            jval *r = json_get(w, "resident_bytes");
            o->resident = r && r->t == J_NUM ? r->num : 0;
            o->inflight = (int)num_field(w, "inflight");
            jval *e = json_get(w, "expires_in");
            o->has_expires = e && e->t == J_NUM;
            o->expires_in = o->has_expires ? e->num : 0;
        }
    }
    jval *p = json_get(j, "pending"), *port = json_get(j, "port"), *lan = json_get(j, "lan"), *auto_ = json_get(j, "autostart"), *log = json_get(j, "log");
    Summary sm;
    summarize(&sm, &now, p && p->t == J_STR ? p->str : NULL, port && port->t == J_NUM ? (int)port->num : 8000);
    MenuItem items[MAX_MENU];
    int count = build_menu(items, &sm, lan && lan->t == J_BOOL && lan->boolean, auto_ && auto_->t == J_BOOL && auto_->boolean,
                           log && log->t == J_BOOL && log->boolean);
    Str s = { 0 };
    str_fmt(&s, "{\"summary\": {\"kind\": \"%s\", \"title\": ", KIND_NAME[sm.kind]);
    json_quote(&s, sm.title);
    str_cat(&s, ", \"tip\": "); json_quote(&s, sm.tip);
    str_fmt(&s, ", \"running\": %s, \"stopped\": %s, \"pending\": %s, \"unsure\": %s, \"busy\": %d}, \"menu\": [",
            sm.running ? "true" : "false", sm.stopped ? "true" : "false", sm.pending ? "true" : "false", sm.unsure ? "true" : "false", sm.busy);
    for (int i = 0; i < count; i++) {
        if (i) str_cat(&s, ", ");
        if (items[i].sep) { str_cat(&s, "{\"sep\": true}"); continue; }
        str_cat(&s, "{\"id\": ");
        if (items[i].has_id) json_quote(&s, items[i].id); else str_cat(&s, "null");
        str_cat(&s, ", \"label\": "); json_quote(&s, items[i].label);
        str_fmt(&s, ", \"enabled\": %s, \"checked\": %s}", items[i].enabled ? "true" : "false", items[i].checked ? "true" : "false");
    }
    str_cat(&s, "]}\n");
    out_write(s.p, s.n);
    free(s.p);
    json_free(j);
    return 0;
}

static int dump_icon(const char *name) {
    for (int k = 0; k < K_COUNT; k++) if (!strcmp(name, KIND_NAME[k])) {
        unsigned char px[32 * 32 * 4];
        icon_pixels(k, 32, px);
        char *hex = (char *)malloc(sizeof px * 2 + 2);
        for (size_t i = 0; i < sizeof px; i++) sprintf(hex + i * 2, "%02x", px[i]);
        hex[sizeof px * 2] = '\n';
        out_write(hex, sizeof px * 2 + 1);
        free(hex);
        return 0;
    }
    say("unknown icon kind %s\n", name);
    return 1;
}

static int print_config(void) {
    Str s = { 0 };
    str_fmt(&s, "{\"port\": %d, \"router\": %s, \"lan\": %s, \"start_with_tray\": %s, \"key\": ", G.cfg.port,
            G.cfg.router ? "true" : "false", G.cfg.lan ? "true" : "false", G.cfg.start_with_tray ? "true" : "false");
    json_quote(&s, G.cfg.key_path);
    str_cat(&s, ", \"out\": "); json_quote(&s, G.cfg.out_path);
    str_cat(&s, ", \"err\": "); json_quote(&s, G.cfg.err_path);
    str_cat(&s, ", \"request_log\": "); if (G.cfg.request_log[0]) json_quote(&s, G.cfg.request_log); else str_cat(&s, "null");
    str_cat(&s, "}\n");
    out_write(s.p, s.n);
    free(s.p);
    return 0;
}

static void utf8_args(int *argc, char ***argv) {
    int n;
    wchar_t **w = CommandLineToArgvW(GetCommandLineW(), &n);
    char **a = (char **)calloc((size_t)n + 1, sizeof *a);
    for (int i = 0; i < n; i++) { a[i] = (char *)malloc(wcslen(w[i]) * 4 + 1); narrow(w[i], a[i], wcslen(w[i]) * 4 + 1); }
    LocalFree(w);
    *argc = n;
    *argv = a;
}

int main(void) {
    int argc;
    char **argv;
    utf8_args(&argc, &argv);
    InitializeCriticalSection(&G.lock);
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    http_open();
    const char *dump = NULL, *icon = NULL, *config = NULL, *do_action = NULL;
    int want_status = 0, want_config = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--python") && i + 1 < argc) snprintf(G.python, sizeof G.python, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "--dump-menu") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--dump-icon") && i + 1 < argc) icon = argv[++i];
        else if (!strcmp(argv[i], "--do") && i + 1 < argc) do_action = argv[++i];
        else if (!strcmp(argv[i], "--status")) want_status = 1;
        else if (!strcmp(argv[i], "--print-config")) want_config = 1;
        else if (!strcmp(argv[i], "--here")) { /* tray.py's word for "run in this process": this one always does */ }
        else {
            say("colibri-tray.exe [--python PYTHON.EXE] [--config FILE]   (--dump-menu FILE | --dump-icon KIND | --status | --print-config)\n");
            return !strcmp(argv[i], "--help") ? 0 : 1;
        }
    }
    if (dump) return dump_menu(dump);
    if (icon) return dump_icon(icon);
    wchar_t exe[1024];
    GetModuleFileNameW(NULL, exe, 1024);
    narrow(exe, G.exe, sizeof G.exe);
    snprintf(G.exe_dir, sizeof G.exe_dir, "%s", G.exe);
    char *slash = strrchr(G.exe_dir, '\\');
    if (slash) *slash = 0;
    char err[1500];
    if (!load_cfg(&G.cfg, config, err, sizeof err)) {
        if (want_status || want_config) say("colibri-tray: %s\n", err);
        else { wchar_t *w = wide(err); MessageBoxW(NULL, w, L"Colibri tray", MB_ICONERROR); free(w); }
        return 1;
    }
    if (want_config) return print_config();
    if (want_status) { Now now; server_status(&G.cfg, &now); print_now(&now); return 0; }
    if (do_action) {
        G.cli = 1;
        ActionArg *a = (ActionArg *)calloc(1, sizeof *a);
        snprintf(a->action, sizeof a->action, "%s", do_action);
        server_action(a);
        return 0;
    }
    return run_tray();
}
