/* qwenimage_resize.h -- the pipeline's picture resize, for the reference images of an edit.
 *
 * QwenImage21Pipeline sizes every condition image itself before anything reads it:
 *
 *   calculate_dimensions(output_resolution^2, w / h)   a side of round(x / 32) * 32 (ties to even), the
 *                                                      area staying near output_resolution^2
 *   VaeImageProcessor.resize(img, w, h)                 PIL Image.resize, LANCZOS, no reducing gap
 *
 * and the same resized picture feeds the vision tower and the VAE. The engine cannot hand that step to the
 * caller (the gateway has no imaging library), so it is here, ported from Pillow's Resample.c:
 *
 *   - RGBA is resampled premultiplied ("RGBa"): colour * alpha / 255 in, colour * 255 / alpha out, so a
 *     transparent pixel's colour does not bleed into its neighbours
 *   - one separable pass per axis, horizontal first, 8-bit in between; an axis whose size is unchanged is
 *     skipped
 *   - the Lanczos window (a = 3) is stretched by the reduction when shrinking, so a downscale averages
 *     over its source footprint; taps are normalized to sum to one and kept as 22-bit fixed point
 *     (round half away from zero), accumulators start at half an LSB, then >> 22 and clamp
 *
 * The caller's result is held to PIL's by the oracle (--vl-ref): bytes may differ by one level where a
 * Pillow release rounds a premultiply differently, never more. */
#ifndef QWENIMAGE_RESIZE_H
#define QWENIMAGE_RESIZE_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QIR_PREC 22

static double qir_sinc(double x){ if (x == 0.0) return 1.0; x *= 3.14159265358979323846; return sin(x) / x; }
static double qir_lanczos(double x){ return x >= -3.0 && x < 3.0 ? qir_sinc(x) * qir_sinc(x / 3.0) : 0.0; }

/* Fixed-point taps of every output position on one axis: k[out][ksize], bounds[out] = (first input, count).
 * NULL on allocation failure. */
static int *qir_coeffs(int in_size, int out_size, int *bounds, int *ksize_out){
    double scale = (double)in_size / out_size, fs = scale < 1.0 ? 1.0 : scale, support = 3.0 * fs;
    int ksize = (int)ceil(support) * 2 + 1;
    int *k = calloc((size_t)out_size * ksize, sizeof(int));
    double *w = malloc(sizeof(double) * ksize);
    if (!k || !w) { free(k); free(w); return NULL; }
    for (int xx = 0; xx < out_size; xx++) {
        double center = (xx + 0.5) * scale, ss = 1.0 / fs, ww = 0.0;
        int xmin = (int)(center - support + 0.5), xmax = (int)(center + support + 0.5);
        if (xmin < 0) xmin = 0;
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        for (int x = 0; x < xmax; x++) { w[x] = qir_lanczos((x + xmin - center + 0.5) * ss); ww += w[x]; }
        for (int x = 0; x < xmax; x++) {
            double v = ww != 0.0 ? w[x] / ww : w[x];
            k[(size_t)xx * ksize + x] = (int)(v < 0 ? -0.5 + v * (1 << QIR_PREC) : 0.5 + v * (1 << QIR_PREC));
        }
        bounds[2 * xx] = xmin; bounds[2 * xx + 1] = xmax;
    }
    free(w);
    *ksize_out = ksize;
    return k;
}

static inline uint8_t qir_clip8(int v){ return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }

/* RGBA -> premultiplied, and back, as Pillow's RGBA <-> RGBa conversions do */
static inline int qir_div255(int v){ v += 128; return ((v >> 8) + v) >> 8; }
static void qir_premultiply(uint8_t *p, size_t n){
    for (size_t i = 0; i < n; i++, p += 4) {
        int a = p[3];                       /* a = 0 zeroes the colour: a clear pixel contributes nothing */
        if (a == 255) continue;
        for (int c = 0; c < 3; c++) p[c] = (uint8_t)qir_div255(p[c] * a);
    }
}
static void qir_unpremultiply(uint8_t *p, size_t n){
    for (size_t i = 0; i < n; i++, p += 4) {
        int a = p[3];
        if (a == 255 || a == 0) continue;
        for (int c = 0; c < 3; c++) p[c] = qir_clip8((int)(p[c] * 255u / (unsigned)a));
    }
}

/* one separable pass over rows of 4-channel pixels: horizontal maps [rows][in][4] to [rows][out][4] */
static void qir_pass_h(const uint8_t *src, int in, int rows, uint8_t *dst, int out, const int *k, int ks, const int *b){
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < out; x++) {
            const int *kk = k + (size_t)x * ks; int x0 = b[2 * x], n = b[2 * x + 1];
            const uint8_t *s = src + ((size_t)y * in + x0) * 4;
            for (int c = 0; c < 4; c++) {
                int acc = 1 << (QIR_PREC - 1);
                for (int i = 0; i < n; i++) acc += s[i * 4 + c] * kk[i];
                dst[((size_t)y * out + x) * 4 + c] = qir_clip8(acc >> QIR_PREC);
            }
        }
}
/* vertical maps [in][w][4] to [out][w][4] */
static void qir_pass_v(const uint8_t *src, int w, uint8_t *dst, int out, const int *k, int ks, const int *b){
    for (int y = 0; y < out; y++) {
        const int *kk = k + (size_t)y * ks; int y0 = b[2 * y], n = b[2 * y + 1];
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 4; c++) {
                int acc = 1 << (QIR_PREC - 1);
                for (int i = 0; i < n; i++) acc += src[((size_t)(y0 + i) * w + x) * 4 + c] * kk[i];
                dst[((size_t)y * w + x) * 4 + c] = qir_clip8(acc >> QIR_PREC);
            }
    }
}

/* src[sh][sw][4] -> dst[dh][dw][4]. 0, or -1 when out of memory or a size is not positive.
 *
 * Which axis goes first changes the 8-bit intermediate and so, by a rounding, the result: a level or two
 * on a photograph, a lot on white noise. Pillow settles it inside its resampler and I found no closed form
 * for it, so this is a fit to its choices (Pillow 13, 700 random shapes with one pass order exactly right):
 * when the height shrinks, vertical first if the width grows or stays; when both shrink, vertical first if
 * its taps x pixels, with 20% added for the strided pass, is below the horizontal order's. It agrees on 96%
 * of shapes and on 83% of the both-shrink ones; the oracle's tolerance is a level, on smooth pictures. */
static int qir_resize_rgba(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh){
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return -1;
    size_t sn = (size_t)sw * sh;
    uint8_t *a = malloc(sn * 4), *tmp = NULL;
    int *bx = NULL, *by = NULL, *kx = NULL, *ky = NULL, ksx = 0, ksy = 0, rc = -1;
    if (!a) return -1;
    memcpy(a, src, sn * 4);
    qir_premultiply(a, sn);
    if (dw != sw) { bx = malloc(sizeof(int) * 2 * dw); kx = bx ? qir_coeffs(sw, dw, bx, &ksx) : NULL; if (!kx) goto done; }
    if (dh != sh) { by = malloc(sizeof(int) * 2 * dh); ky = by ? qir_coeffs(sh, dh, by, &ksy) : NULL; if (!ky) goto done; }
    if (dw != sw && dh != sh) {
        int vfirst = dh < sh && (dw >= sw || 1.2 * ((double)dh * sw * ksy + (double)dh * dw * ksx) <
                                              (double)sh * dw * ksx + (double)dh * dw * ksy);
        tmp = malloc((size_t)(vfirst ? sw : dw) * (vfirst ? dh : sh) * 4);
        if (!tmp) goto done;
        if (vfirst) { qir_pass_v(a, sw, tmp, dh, ky, ksy, by); qir_pass_h(tmp, sw, dh, dst, dw, kx, ksx, bx); }
        else        { qir_pass_h(a, sw, sh, tmp, dw, kx, ksx, bx); qir_pass_v(tmp, dw, dst, dh, ky, ksy, by); }
    } else if (dw != sw) qir_pass_h(a, sw, sh, dst, dw, kx, ksx, bx);
    else if (dh != sh)   qir_pass_v(a, sw, dst, dh, ky, ksy, by);
    else memcpy(dst, a, sn * 4);
    qir_unpremultiply(dst, (size_t)dw * dh);
    rc = 0;
done:
    free(a); free(tmp); free(bx); free(by); free(kx); free(ky);
    return rc;
}

/* the pipeline's calculate_dimensions(area, ratio): sides rounded to a multiple of 32, ties to even
 * (Python's round), for a source of sw x sh. 0 when the sides are at least 32. */
static int qir_calc_dims(double area, int sw, int sh, int *w, int *h){
    double ratio = (double)sw / (double)sh;
    double wd = sqrt(area * ratio), hd = wd / ratio;
    *w = (int)nearbyint(wd / 32.0) * 32;
    *h = (int)nearbyint(hd / 32.0) * 32;
    return *w >= 32 && *h >= 32 ? 0 : -1;
}

#endif /* QWENIMAGE_RESIZE_H */
