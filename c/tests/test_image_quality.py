"""Structural similarity of two pictures (image_quality.py)."""
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import image_engine  # noqa: E402
import image_quality  # noqa: E402


def gradient(width, height, flip=False):
    """A picture with structure: a diagonal ramp with a bright square, grey, as (w, h, channels, pixels); `flip` mirrors it."""
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            v = (x * 3 + y * 2) % 256
            if 8 <= x < 20 and 8 <= y < 20:
                v = 250
            pixels.append(v)
    if flip:
        rows = [pixels[y * width:(y + 1) * width][::-1] for y in range(height)]
        pixels = bytearray(b"".join(rows))
    return width, height, 1, bytes(pixels)


def rgb(width, height, color):
    return width, height, 3, bytes(color) * (width * height)


class SsimTest(unittest.TestCase):
    def test_the_same_picture_scores_one(self):
        picture = gradient(40, 40)
        self.assertEqual(image_quality.ssim(picture, picture), 1.0)

    def test_noise_scores_lower_the_more_there_is(self):
        base = gradient(48, 48)
        rng = random.Random(7)

        def noisy(amount):
            return base[0], base[1], 1, bytes(min(255, max(0, v + rng.randint(-amount, amount))) for v in base[3])
        little, lots = image_quality.ssim(base, noisy(4)), image_quality.ssim(base, noisy(60))
        self.assertLess(lots, little)
        self.assertLess(little, 1.0)
        self.assertGreater(little, 0.85)
        self.assertLess(lots, 0.6)

    def test_a_mirrored_picture_scores_far_below_the_line(self):
        self.assertLess(image_quality.ssim(gradient(40, 40), gradient(40, 40, flip=True)), 0.5)

    def test_it_matches_a_hand_worked_window(self):
        # One 7x7 window of a constant 100 against a constant 110: no variance, so the score is the luminance term alone
        # (2*100*110 + C1) / (100^2 + 110^2 + C1), with C1 = (0.01*255)^2 and the contrast term exactly 1.
        a = [100.0] * 49
        b = [110.0] * 49
        c1 = (0.01 * 255) ** 2
        expected = (2 * 100 * 110 + c1) / (100 ** 2 + 110 ** 2 + c1)
        self.assertAlmostEqual(image_quality.ssim_plane(a, b, 7, 7), expected, places=9)

    def test_a_colour_drift_the_luminance_does_not_see_is_seen_per_channel(self):
        red, blue = rgb(16, 16, (200, 40, 40)), rgb(16, 16, (40, 40, 200))
        # equal luminance is not what this is about: the planes are flat, so the score is the brightness term of each channel
        self.assertLess(image_quality.ssim(red, blue, per_channel=True), image_quality.ssim(red, red, per_channel=True))

    def test_pictures_of_different_sizes_are_refused(self):
        with self.assertRaises(ValueError):
            image_quality.ssim(gradient(20, 20), gradient(24, 20))

    def test_a_picture_smaller_than_the_window_is_refused(self):
        with self.assertRaises(ValueError):
            image_quality.ssim(gradient(5, 5), gradient(5, 5))


class CommandTest(unittest.TestCase):
    def write(self, picture, name):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as handle:
            handle.write(image_engine.encode_png(picture[0], picture[1], picture[3], picture[2]))
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_it_prints_the_score_and_exits_zero_over_the_line(self):
        same = self.write(gradient(32, 32), "a.png")
        self.assertEqual(image_quality.main([same, same, "--min", "0.95"]), 0)

    def test_it_exits_one_under_the_line(self):
        a, b = self.write(gradient(32, 32), "a.png"), self.write(gradient(32, 32, flip=True), "b.png")
        self.assertEqual(image_quality.main([a, b, "--min", "0.95"]), 1)


if __name__ == "__main__":
    unittest.main()
