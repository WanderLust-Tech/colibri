"""How alike two pictures are: structural similarity (SSIM), for checking a picture against a saved baseline.

    python c/image_quality.py NEW.png BASELINE.png [--min 0.95]

prints the score and exits 1 when it is under `--min`. 1.0 is the same picture; the same seed on the same engine gives 1.0, a
different way of summing the same arithmetic (the CPU's, the card's) gives a little under, and a different picture gives well under
0.5. 0.95 is the line ComfyUI's own image tests use.

The score is the one scikit-image's `structural_similarity` gives a grey picture with its defaults: a 7x7 uniform window, the sample
covariance, K1 0.01 and K2 0.03 over a range of 255, averaged over every window that fits inside the picture. A colour picture is
compared as its luminance (0.299 R + 0.587 G + 0.114 B), which is where the eye looks for structure; it does not see a hue drift, so
`--per-channel` scores red, green and blue each and takes the lowest. Pure Python, no dependencies: a 1024x1024 picture takes about a
second.
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WINDOW = 7
K1, K2 = 0.01, 0.03
RANGE = 255.0


def _planes(channels, pixels, width, height, per_channel=False):
    """The planes to score, each a flat list of floats row by row: one luminance plane, or each colour channel."""
    count = width * height
    if channels in (1, 2):
        return [[float(v) for v in pixels[0::channels]]]
    r, g, b = pixels[0::channels], pixels[1::channels], pixels[2::channels]
    if per_channel:
        return [[float(v) for v in r], [float(v) for v in g], [float(v) for v in b]]
    return [[0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i] for i in range(count)]]


def _table(values, width, height):
    """The summed-area table of a plane: row y, column x holds the sum of everything above and left of it (one row and column of
    zeros first), so any rectangle's sum is four lookups."""
    zero = [0.0] * (width + 1)
    rows = [zero]
    above = zero
    for y in range(height):
        running = list(itertools.accumulate(values[y * width:(y + 1) * width], initial=0.0))
        above = [a + b for a, b in zip(above, running)]
        rows.append(above)
    return rows


def _window_sums(table, height, width, win):
    """For every window that fits, its sum, one list per row of windows."""
    out = []
    for y in range(height - win + 1):
        low, high = table[y], table[y + win]
        out.append([hi_r - lo_r - hi_l + lo_l for hi_r, lo_r, hi_l, lo_l in
                    zip(high[win:], low[win:], high[:-win], low[:-win])])
    return out


def ssim_plane(a, b, width, height, win=WINDOW):
    """The mean SSIM of two planes of the same size (flat lists of floats 0..255)."""
    if width < win or height < win:
        raise ValueError(f"a picture needs at least {win}x{win} pixels to be scored")
    n = win * win
    norm = n / (n - 1)
    c1, c2 = (K1 * RANGE) ** 2, (K2 * RANGE) ** 2
    sums = [_window_sums(_table(plane, width, height), height, width, win)
            for plane in (a, b, [v * v for v in a], [v * v for v in b], [x * y for x, y in zip(a, b)])]
    total, windows = 0.0, 0
    for sx_row, sy_row, sxx_row, syy_row, sxy_row in zip(*sums):
        for sx, sy, sxx, syy, sxy in zip(sx_row, sy_row, sxx_row, syy_row, sxy_row):
            ux, uy = sx / n, sy / n
            vx, vy = norm * (sxx / n - ux * ux), norm * (syy / n - uy * uy)
            vxy = norm * (sxy / n - ux * uy)
            total += ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
            windows += 1
    return min(1.0, total / windows)


def ssim(first, second, per_channel=False):
    """The SSIM of two pictures given as (width, height, channels, pixels), as image_engine.decode_png returns them."""
    (w1, h1, c1, p1), (w2, h2, c2, p2) = first, second
    if (w1, h1) != (w2, h2):
        raise ValueError(f"the pictures are not the same size ({w1}x{h1} and {w2}x{h2})")
    scores = [ssim_plane(a, b, w1, h1) for a, b in zip(_planes(c1, p1, w1, h1, per_channel), _planes(c2, p2, w2, h2, per_channel))]
    return min(scores)


def load(path):
    import image_engine
    with open(path, "rb") as handle:
        return image_engine.decode_png(handle.read())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Structural similarity of two PNG pictures (1.0 = the same).")
    parser.add_argument("picture")
    parser.add_argument("baseline")
    parser.add_argument("--min", type=float, default=None, metavar="SCORE", help="exit 1 when the score is lower (0.95 is a common line)")
    parser.add_argument("--per-channel", action="store_true", help="score red, green and blue each and take the lowest, not the luminance")
    args = parser.parse_args(argv)
    first, second = load(args.picture), load(args.baseline)
    score = ssim(first, second, args.per_channel)
    print(f"ssim {score:.4f} ({first[0]}x{first[1]}, {'lowest channel' if args.per_channel else 'luminance'})")
    if args.min is not None and score < args.min:
        print(f"under {args.min}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
