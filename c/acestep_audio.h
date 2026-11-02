/* acestep_audio.h: reading a WAV or an MP3 and turning it into what ACE-Step 1.5 takes in, stereo at 48 kHz.
 *
 * Upstream's handler reads a file with soundfile (libsndfile; mpg123 for MP3) and `_normalize_audio_to_stereo_48k`
 * (acestep/core/generation/handler/io_audio.py). This file is those two steps, in C, with nothing to link:
 *
 *   aca_load(path, ...) / aca_load_mem(bytes, n, ...)   WAV (PCM 8/16/24/32 int, float 32/64, WAVE_FORMAT_EXTENSIBLE,
 *                                                       any rate, up to 255 channels) and MP3 (MPEG-1/2/2.5 layer III, with
 *                                                       or without an ID3v2 tag) -> interleaved float32 in [-1, 1].
 *                                                       The format is sniffed from the bytes, never from the extension.
 *   aca_stereo48k(pcm, frames, rate, channels, &out, &n)   mono is doubled, a third channel and up are dropped, a rate other
 *                                                       than 48000 is resampled with torchaudio's Resample(rate, 48000)
 *                                                       (sinc_interp_hann, width 6, rolloff 0.99; the same gcd polyphase
 *                                                       kernel, padding and length), then clamped to [-1, 1]. The result is
 *                                                       planar [2][n], channels first, as upstream's tensor is.
 *
 * The oracle is tests/test_acestep_audio.py: upstream's own code on the same bytes, sample by sample, and mutants of this
 * file that the oracle has to catch (tools/mutate_acestep_audio.py).
 *
 * What it refuses, with a message: anything that is not WAV or MP3 (FLAC, Ogg, M4A, AIFF, RF64: say so by name), a WAV that
 * is not one of the sample formats above, a file with no audio in it, a file longer than the cap (20 minutes by default,
 * decided from the header BEFORE anything is allocated; an MP3 without a header is stopped while it decodes), a file over
 * ACA_MAX_FILE_BYTES, and a sample rate whose resampling kernel would be absurd (the reduced ratio to 48 kHz has a
 * denominator above ~1900, e.g. 44056 Hz; torchaudio would build a multi-gigabyte kernel for it too).
 *
 * MP3 is decoded by minimp3 (https://github.com/lieff/minimp3, commit ea99364f61c14656440e8d77e9c233ccf3124633, CC0 1.0,
 * c/third_party/minimp3.h, unmodified; credited in THIRD_PARTY_NOTICES.md). The encoder delay and padding of a LAME/Lavf
 * tag are removed the way mpg123 removes them (begin = delay + 529, end = frames * samples-per-frame - padding + 529), which
 * is what libsndfile gives upstream, so a file written by an encoder that wrote that tag comes back with the length it was
 * encoded with. The Xing/Info frame itself produces no audio. One quirk of mpg123 is kept on purpose, because upstream
 * has it: behind a CRC (protection bit clear) the tag is not looked for, so that frame is played as audio and nothing is
 * trimmed. A stream with no Xing/Info frame is decoded whole, as mpg123 does (libsndfile only cuts it short when its guess at
 * the length from the first frame's bitrate is too small, which no decoder should copy).
 */
#ifndef ACESTEP_AUDIO_H
#define ACESTEP_AUDIO_H

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#define MINIMP3_FLOAT_OUTPUT
#define MINIMP3_IMPLEMENTATION
#include "third_party/minimp3.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#define ACA_DEFAULT_MAX_SECONDS 1200.0           /* the cap on how long a clip may be */
#define ACA_MAX_SAMPLES ((size_t)1 << 28)        /* frames * channels, decoded (1 GiB of float32) */
#define ACA_MAX_FILE_BYTES ((size_t)1 << 30)     /* aca_load will not slurp a bigger file */
#define ACA_MAX_RATE 1000000
#define ACA_MAX_CHANNELS 255
#define ACA_MAX_KERNEL 2048                      /* taps of one polyphase filter: 2 * width + reduced orig rate */
#define ACA_TARGET_RATE 48000

static void aca_seterr(char *err, int errn, const char *fmt, ...) {
    if (!err || errn <= 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, (size_t)errn, fmt, ap);
    va_end(ap);
}

static uint32_t aca_rd16(const unsigned char *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }
static uint32_t aca_rd32(const unsigned char *p) { return aca_rd16(p) | (aca_rd16(p + 2) << 16); }
static uint32_t aca_be32(const unsigned char *p) { return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }

/* How many frames of this rate the cap allows. */
static size_t aca_cap_frames(double max_seconds, int rate) {
    double f = max_seconds * (double)rate;
    if (!(f >= 1.0)) return 0;
    if (f > (double)ACA_MAX_SAMPLES) f = (double)ACA_MAX_SAMPLES;
    return (size_t)f;
}

/* ------------------------------------------------------------------------------------------------ WAV -- */

static int aca_wav(const unsigned char *b, size_t n, double max_seconds, float **pcm, size_t *frames, int *rate, int *channels,
                   char *err, int errn) {
    size_t pos = 12;
    int have_fmt = 0;
    uint32_t tag = 0, ch = 0, rt = 0, blk = 0, bits = 0;
    while (pos + 8 <= n) {
        const unsigned char *id = b + pos;
        const size_t sz = aca_rd32(b + pos + 4);
        pos += 8;
        const size_t avail = n - pos;
        if (!memcmp(id, "fmt ", 4)) {
            if (sz < 16 || avail < 16) { aca_seterr(err, errn, "the WAV's fmt chunk is truncated"); return -1; }
            const unsigned char *p = b + pos;
            tag = aca_rd16(p); ch = aca_rd16(p + 2); rt = aca_rd32(p + 4); blk = aca_rd16(p + 12); bits = aca_rd16(p + 14);
            if (tag == 0xFFFE) {                       /* WAVE_FORMAT_EXTENSIBLE: the real format is the GUID's first two bytes */
                if (sz < 40 || avail < 40) { aca_seterr(err, errn, "the WAV's extensible fmt chunk is truncated"); return -1; }
                tag = aca_rd16(p + 24);
            }
            have_fmt = 1;
        } else if (!memcmp(id, "data", 4)) {
            if (!have_fmt) { aca_seterr(err, errn, "the WAV has audio before its fmt chunk"); return -1; }
            if (tag != 1 && tag != 3) { aca_seterr(err, errn, "unsupported WAV encoding %u (only PCM and IEEE float)", tag); return -1; }
            if (ch < 1 || ch > ACA_MAX_CHANNELS) { aca_seterr(err, errn, "unsupported WAV channel count %u (1 to %d)", ch, ACA_MAX_CHANNELS); return -1; }
            if (rt < 1 || rt > ACA_MAX_RATE) { aca_seterr(err, errn, "unsupported WAV sample rate %u Hz", rt); return -1; }
            const int okbits = tag == 1 ? (bits == 8 || bits == 16 || bits == 24 || bits == 32) : (bits == 32 || bits == 64);
            if (!okbits) { aca_seterr(err, errn, "unsupported WAV sample size: %u-bit %s", bits, tag == 1 ? "PCM" : "float"); return -1; }
            const size_t bytes = bits / 8;
            if (blk != ch * bytes) { aca_seterr(err, errn, "the WAV's block alignment %u does not fit %u channels of %u bits", blk, ch, bits); return -1; }
            const size_t dlen = sz > avail ? avail : sz;   /* a data chunk that claims more than the file holds is cut short */
            const size_t nfr = dlen / blk;
            if (nfr == 0) { aca_seterr(err, errn, "the WAV holds no audio"); return -1; }
            if (nfr > aca_cap_frames(max_seconds, (int)rt)) {
                aca_seterr(err, errn, "the audio is longer than the %g seconds this accepts", max_seconds); return -1;
            }
            if (nfr > ACA_MAX_SAMPLES / ch) { aca_seterr(err, errn, "the audio is too large to hold in memory"); return -1; }
            float *out = (float *)malloc(nfr * ch * sizeof(float));
            if (!out) { aca_seterr(err, errn, "out of memory"); return -1; }
            const unsigned char *d = b + pos;
            const size_t ns = nfr * ch;
            if (tag == 3 && bits == 32) { memcpy(out, d, ns * 4); /* little-endian host: the file's own float32 */ }
            else if (tag == 3) { for (size_t i = 0; i < ns; i++) { double v; memcpy(&v, d + i * 8, 8); out[i] = (float)v; } }
            else if (bits == 8) { for (size_t i = 0; i < ns; i++) out[i] = (float)((int)d[i] - 128) * (1.0f / 128.0f); }
            else if (bits == 16) { for (size_t i = 0; i < ns; i++) out[i] = (float)(int16_t)aca_rd16(d + i * 2) * (1.0f / 32768.0f); }
            else if (bits == 24) {
                for (size_t i = 0; i < ns; i++) {
                    int32_t v = (int32_t)((aca_rd16(d + i * 3) | ((uint32_t)d[i * 3 + 2] << 16)) << 8) >> 8;
                    out[i] = (float)v * (1.0f / 8388608.0f);
                }
            } else { for (size_t i = 0; i < ns; i++) out[i] = (float)(int32_t)aca_rd32(d + i * 4) * (1.0f / 2147483648.0f); }
            *pcm = out; *frames = nfr; *rate = (int)rt; *channels = (int)ch;
            return 0;
        }
        if (sz > avail) break;                         /* a chunk that runs off the end: nothing follows it */
        pos += sz + (sz & 1);                          /* chunks are padded to an even size */
    }
    aca_seterr(err, errn, "the WAV has no data chunk");
    return -1;
}

/* ------------------------------------------------------------------------------------------------ MP3 -- */

/* The LAME/Xing tag of the first frame: how many frames the file holds, the encoder delay and the padding.
 * Returns 1 when gapless information is present (mpg123 only trims when it is). */
static int aca_mp3_tag(const unsigned char *f, size_t avail, size_t *tag_frames, size_t *delay, size_t *padding, int *is_info) {
    *is_info = 0;
    if (avail < 8) return 0;
    const int mpeg1 = ((f[1] >> 3) & 3) == 3, mono = (f[3] >> 6) == 3, crc = !(f[1] & 1);
    if (crc) return 0;          /* mpg123 does not look for the tag behind a CRC (measured, with a valid CRC too): it plays that
                                   frame as audio, with no gapless trim, and libsndfile gives upstream what mpg123 gives */
    const size_t side = mpeg1 ? (mono ? 17 : 32) : (mono ? 9 : 17);
    size_t o = 4 + side;
    if (o + 8 > avail) return 0;
    if (memcmp(f + o, "Xing", 4) && memcmp(f + o, "Info", 4)) return 0;
    *is_info = 1;
    const uint32_t flags = aca_be32(f + o + 4);
    o += 8;
    size_t frames_field = 0;
    if (flags & 1) { if (o + 4 > avail) return 0; frames_field = aca_be32(f + o); o += 4; }
    if (flags & 2) o += 4;
    if (flags & 4) o += 100;
    if (flags & 8) o += 4;
    if (!(flags & 1) || o + 24 > avail) return 0;
    if (memcmp(f + o, "LAME", 4) && memcmp(f + o, "Lavf", 4) && memcmp(f + o, "Lavc", 4)) return 0;
    const unsigned char *t = f + o + 21;
    *tag_frames = frames_field;
    *delay = ((size_t)t[0] << 4) | (t[1] >> 4);
    *padding = (((size_t)t[1] & 15) << 8) | t[2];
    return 1;
}

static int aca_mp3(const unsigned char *b, size_t n, double max_seconds, float **pcm, size_t *frames, int *rate, int *channels,
                   char *err, int errn) {
    size_t pos = 0;
    while (n - pos >= 10 && !memcmp(b + pos, "ID3", 3)) {          /* ID3v2: a syncsafe size, an optional footer */
        const size_t size = ((size_t)(b[pos + 6] & 0x7f) << 21) | ((size_t)(b[pos + 7] & 0x7f) << 14) | ((size_t)(b[pos + 8] & 0x7f) << 7) | (b[pos + 9] & 0x7f);
        const size_t total = 10 + size + ((b[pos + 5] & 0x10) ? 10 : 0);
        if (total > n - pos) { aca_seterr(err, errn, "the MP3's ID3 tag runs past the end of the file"); return -1; }
        pos += total;
    }
    if (n - pos >= 128 && !memcmp(b + n - 128, "TAG", 3)) {            /* an ID3v1 tag (and TAG+) at the end is not audio; left in,
                                                                         it stops minimp3 validating the last frame's successor */
        n -= 128;
        if (n - pos >= 227 && !memcmp(b + n - 227, "TAG+", 4)) n -= 227;
    }
    mp3dec_t *dec = (mp3dec_t *)malloc(sizeof *dec);
    float *tmp = (float *)malloc(MINIMP3_MAX_SAMPLES_PER_FRAME * sizeof(float));
    if (!dec || !tmp) { free(dec); free(tmp); aca_seterr(err, errn, "out of memory"); return -1; }
    mp3dec_init(dec);
    float *out = NULL;
    size_t cap = 0, got = 0;                                         /* frames decoded so far */
    int hz = 0, ch = 0, nframes_seen = 0, rc = -1, have_tag = 0;
    size_t tag_frames = 0, delay = 0, padding = 0, spf = 1152, limit = 0;
    mp3dec_frame_info_t info;
    memset(&info, 0, sizeof info);
    while (pos < n) {
        const int left = (n - pos > (size_t)0x7fffffff) ? 0x7fffffff : (int)(n - pos);
        const int s = mp3dec_decode_frame(dec, b + pos, left, tmp, &info);
        if (info.frame_bytes <= 0) break;                            /* no further frame in what is left */
        const unsigned char *fr = b + pos + info.frame_offset;
        pos += (size_t)info.frame_bytes;
        if (s <= 0) continue;                                        /* skipped bytes (an ID3v1 tag, junk) */
        if (nframes_seen == 0) {
            hz = info.hz; ch = info.channels;
            if (hz < 1 || hz > ACA_MAX_RATE || ch < 1 || ch > 2) { aca_seterr(err, errn, "the MP3 has an unsupported format"); goto done; }
            spf = (size_t)s;
            limit = aca_cap_frames(max_seconds, hz);
            int is_info = 0;
            have_tag = aca_mp3_tag(fr, (size_t)(info.frame_bytes - info.frame_offset), &tag_frames, &delay, &padding, &is_info);
            nframes_seen = 1;
            if (is_info) {                                            /* the Xing/Info frame carries no audio */
                if (have_tag && tag_frames * spf > limit + 4 * spf) { aca_seterr(err, errn, "the audio is longer than the %g seconds this accepts", max_seconds); goto done; }
                continue;
            }
        } else if (info.hz != hz || info.channels != ch) {
            aca_seterr(err, errn, "the MP3 changes its sample rate or channel count part-way");
            goto done;
        }
        if (got + (size_t)s > limit || (got + (size_t)s) * (size_t)ch > ACA_MAX_SAMPLES) {
            aca_seterr(err, errn, "the audio is longer than the %g seconds this accepts", max_seconds); goto done;
        }
        if (got + (size_t)s > cap) {
            size_t ncap = cap ? cap * 2 : (size_t)hz;
            while (ncap < got + (size_t)s) ncap *= 2;
            float *g = (float *)realloc(out, ncap * (size_t)ch * sizeof(float));
            if (!g) { aca_seterr(err, errn, "out of memory"); goto done; }
            out = g; cap = ncap;
        }
        memcpy(out + got * (size_t)ch, tmp, (size_t)s * (size_t)ch * sizeof(float));
        got += (size_t)s;
    }
    if (got == 0) { aca_seterr(err, errn, "no MP3 audio found"); goto done; }
    size_t begin = 0, end = got;
    if (have_tag) {                                                  /* gapless, as mpg123 does it */
        begin = delay + 529;
        const size_t total = tag_frames * spf + 529;
        end = total > padding ? total - padding : 0;
        if (end > got) end = got;
    }
    if (begin >= end) { aca_seterr(err, errn, "the MP3 holds no audio"); goto done; }
    if (begin > 0) memmove(out, out + begin * (size_t)ch, (end - begin) * (size_t)ch * sizeof(float));
    *pcm = out; *frames = end - begin; *rate = hz; *channels = ch;
    out = NULL;
    rc = 0;
done:
    free(out); free(tmp); free(dec);
    return rc;
}

/* ----------------------------------------------------------------------------------- front doors -- */

/* Decode a WAV or MP3 held in memory. max_seconds is the longest clip accepted. 0 on success, -1 with a message in err. */
static int aca_load_mem_ex(const unsigned char *b, size_t n, double max_seconds, float **pcm, size_t *frames, int *rate,
                           int *channels, char *err, int errn) {
    *pcm = NULL; *frames = 0; *rate = 0; *channels = 0;
    if (!b || n == 0) { aca_seterr(err, errn, "the file is empty"); return -1; }
    if (n >= 12 && !memcmp(b, "RIFF", 4) && !memcmp(b + 8, "WAVE", 4)) return aca_wav(b, n, max_seconds, pcm, frames, rate, channels, err, errn);
    if (n >= 4 && (!memcmp(b, "RF64", 4) || !memcmp(b, "BW64", 4))) { aca_seterr(err, errn, "RF64 WAV files are not supported; save it as a plain WAV or an MP3"); return -1; }
    if (n >= 4 && !memcmp(b, "fLaC", 4)) { aca_seterr(err, errn, "FLAC is not supported; save it as a WAV or an MP3"); return -1; }
    if (n >= 4 && !memcmp(b, "OggS", 4)) { aca_seterr(err, errn, "Ogg is not supported; save it as a WAV or an MP3"); return -1; }
    if (n >= 4 && !memcmp(b, "FORM", 4)) { aca_seterr(err, errn, "AIFF is not supported; save it as a WAV or an MP3"); return -1; }
    if (n >= 8 && !memcmp(b + 4, "ftyp", 4)) { aca_seterr(err, errn, "M4A/MP4 is not supported; save it as a WAV or an MP3"); return -1; }
    if ((n >= 3 && !memcmp(b, "ID3", 3)) || (n >= 2 && b[0] == 0xFF && (b[1] & 0xE0) == 0xE0)) return aca_mp3(b, n, max_seconds, pcm, frames, rate, channels, err, errn);
    aca_seterr(err, errn, "not an audio file this can read (WAV and MP3 are supported)");
    return -1;
}

static inline int aca_load_mem(const unsigned char *b, size_t n, float **pcm, size_t *frames, int *rate, int *channels, char *err, int errn) {
    return aca_load_mem_ex(b, n, ACA_DEFAULT_MAX_SECONDS, pcm, frames, rate, channels, err, errn);
}

static int aca_load_ex(const char *path, double max_seconds, float **pcm, size_t *frames, int *rate, int *channels, char *err, int errn) {
    *pcm = NULL; *frames = 0; *rate = 0; *channels = 0;
    FILE *f = fopen(path, "rb");
    if (!f) { aca_seterr(err, errn, "cannot open %s", path); return -1; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); aca_seterr(err, errn, "cannot read %s", path); return -1; }
    const long long sz = (long long)ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); aca_seterr(err, errn, "cannot read %s", path); return -1; }
    if ((unsigned long long)sz > (unsigned long long)ACA_MAX_FILE_BYTES) {
        fclose(f); aca_seterr(err, errn, "the file is larger than the %d MB this accepts", (int)(ACA_MAX_FILE_BYTES >> 20)); return -1;
    }
    unsigned char *buf = (unsigned char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); aca_seterr(err, errn, "out of memory"); return -1; }
    const size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    const int rc = aca_load_mem_ex(buf, got, max_seconds, pcm, frames, rate, channels, err, errn);
    free(buf);
    return rc;
}

static inline int aca_load(const char *path, float **pcm, size_t *frames, int *rate, int *channels, char *err, int errn) {
    return aca_load_ex(path, ACA_DEFAULT_MAX_SECONDS, pcm, frames, rate, channels, err, errn);
}

/* ------------------------------------------------------------------------------------- resampling -- */

static int aca_gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }

/* torchaudio.functional.resample on one channel: _get_sinc_resample_kernel (gcd, sinc_interp_hann, lowpass_filter_width 6,
 * rolloff 0.99, the kernel built in float64 the way transforms.Resample builds it) and _apply_sinc_resample_kernel (zero pad
 * `width` before and `width + orig` after, a conv1d with stride orig, the phases interleaved, cut to ceil(new * len / orig)).
 * Every phase's taps are computed when that phase is used, so memory stays at one row however large the kernel is. */
static int aca_resample(const float *x, size_t len, int orig_freq, int new_freq, float **yout, size_t *ylen, char *err, int errn) {
    const int g = aca_gcd(orig_freq, new_freq);
    const int o = orig_freq / g, nf = new_freq / g;
    const double base = (double)(o < nf ? o : nf) * 0.99;
    const int width = (int)ceil((double)(6 * o) / base);
    const long K = 2L * width + o;
    if (K > ACA_MAX_KERNEL) {
        aca_seterr(err, errn, "a sample rate of %d Hz cannot be resampled to 48000 Hz here (resample the file to 48 kHz first)", orig_freq);
        return -1;
    }
    const size_t nb = len / (size_t)o + 1;                           /* conv1d outputs: floor((len + 2 * width + o - K) / o) + 1 */
    const float want = (float)((double)((uint64_t)nf * (uint64_t)len) / (double)o);
    size_t target = (size_t)ceilf(want);
    if (target > nb * (size_t)nf) target = nb * (size_t)nf;
    float *xp = (float *)calloc(len + 2 * (size_t)width + (size_t)o, sizeof(float));
    float *y = (float *)malloc((nb * (size_t)nf) * sizeof(float));
    if (!xp || !y) { free(xp); free(y); aca_seterr(err, errn, "out of memory"); return -1; }
    memcpy(xp + width, x, len * sizeof(float));
    const double pi = 3.14159265358979323846;
    const double scale = base / (double)o;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        float *row = (float *)malloc((size_t)K * sizeof(float));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int p = 0; p < nf; p++) {
            const double first = (double)((float)(-p) / (float)nf);   /* an int64 arange divided by an int: float32, then widened */
            for (long k = 0; k < K; k++) {
                const double idx = (double)(-width + k) / (double)o;
                double t = (first + idx) * base;
                if (t < -6.0) t = -6.0;
                if (t > 6.0) t = 6.0;
                double w = cos(t * pi / 6.0 / 2.0);
                w = w * w;
                t *= pi;
                const double s = t == 0.0 ? 1.0 : sin(t) / t;
                row[k] = (float)(s * (w * scale));
            }
            for (size_t bl = 0; bl < nb; bl++) {
                const float *xs = xp + bl * (size_t)o;
                float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
                long k = 0;
                for (; k + 8 <= K; k += 8) {
                    a0 += row[k] * xs[k];         a1 += row[k + 1] * xs[k + 1]; a2 += row[k + 2] * xs[k + 2]; a3 += row[k + 3] * xs[k + 3];
                    a4 += row[k + 4] * xs[k + 4]; a5 += row[k + 5] * xs[k + 5]; a6 += row[k + 6] * xs[k + 6]; a7 += row[k + 7] * xs[k + 7];
                }
                for (; k < K; k++) a0 += row[k] * xs[k];
                y[bl * (size_t)nf + (size_t)p] = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
            }
        }
        free(row);
    }
    free(xp);
    *yout = y;
    *ylen = target;
    return 0;
}

/* Upstream's _normalize_audio_to_stereo_48k. pcm is interleaved (as aca_load returns it); *out is planar [2][*out_frames],
 * malloc'ed, caller frees. 0 on success. */
static int aca_stereo48k_err(const float *pcm, size_t frames, int rate, int channels, float **out, size_t *out_frames, char *err, int errn) {
    *out = NULL; *out_frames = 0;
    if (!pcm || frames == 0 || channels < 1 || rate < 1) { aca_seterr(err, errn, "there is no audio to convert"); return -1; }
    const int two = channels >= 2;                                   /* mono is doubled, a third channel on is dropped */
    float *c0 = (float *)malloc(frames * sizeof(float));
    float *c1 = two ? (float *)malloc(frames * sizeof(float)) : NULL;
    if (!c0 || (two && !c1)) { free(c0); free(c1); aca_seterr(err, errn, "out of memory"); return -1; }
    for (size_t i = 0; i < frames; i++) {
        c0[i] = pcm[i * (size_t)channels];
        if (two) c1[i] = pcm[i * (size_t)channels + 1];
    }
    float *r0 = c0, *r1 = c1;
    size_t n = frames;
    if (rate != ACA_TARGET_RATE) {
        size_t n0 = 0, n1 = 0;
        float *a = NULL, *b = NULL;
        if (aca_resample(c0, frames, rate, ACA_TARGET_RATE, &a, &n0, err, errn)) { free(c0); free(c1); return -1; }
        if (two && aca_resample(c1, frames, rate, ACA_TARGET_RATE, &b, &n1, err, errn)) { free(a); free(c0); free(c1); return -1; }
        free(c0); free(c1);
        r0 = a; r1 = b; n = n0;
    }
    float *res = (float *)malloc(2 * n * sizeof(float));
    if (!res) { free(r0); free(r1); aca_seterr(err, errn, "out of memory"); return -1; }
    const float *s1 = two ? r1 : r0;
    for (size_t i = 0; i < n; i++) {
        float a = r0[i], b = s1[i];
        a = a < -1.0f ? -1.0f : (a > 1.0f ? 1.0f : a);
        b = b < -1.0f ? -1.0f : (b > 1.0f ? 1.0f : b);
        res[i] = a;
        res[n + i] = b;
    }
    free(r0); free(r1);
    *out = res; *out_frames = n;
    return 0;
}

static inline int aca_stereo48k(const float *pcm, size_t frames, int rate, int channels, float **out, size_t *out_frames) {
    return aca_stereo48k_err(pcm, frames, rate, channels, out, out_frames, NULL, 0);
}

#endif
