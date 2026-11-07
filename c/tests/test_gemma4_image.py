"""Gemma 4's image preprocessing (tools/gemma4_image.py) against transformers' own processor.

    python -m unittest tests.test_gemma4_image

The pure-arithmetic tests need nothing; the comparison with Gemma4ImageProcessorPil needs Pillow, numpy and
transformers >= 5.5 and is skipped without them. The patches must agree byte for byte: they are what the vision tower
reads, and a one-level difference in a pixel is not "close enough" to anything a test could name.
"""
import io
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))

import gemma4_image as gi  # noqa: E402

try:
    import numpy as np
    from PIL import Image
    HAVE_IMAGING = True
except ImportError:                                   # pragma: no cover
    HAVE_IMAGING = False

try:
    from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
    HAVE_TRANSFORMERS = HAVE_IMAGING
except Exception:                                     # pragma: no cover  (not installed, or too old)
    HAVE_TRANSFORMERS = False


class RuleTest(unittest.TestCase):
    def test_sizes_are_multiples_of_48_and_fit_the_patch_budget(self):
        for height, width in ((301, 517), (1000, 50), (64, 64), (2000, 3000), (7, 5000), (1, 1)):
            for budget in (70, 140, 280, 560, 1120):
                h, w = gi.aspect_ratio_preserving_size(height, width, 16, budget * 9, 3)
                self.assertEqual((h % 48, w % 48), (0, 0), (height, width, budget))
                self.assertLessEqual((h // 16) * (w // 16), budget * 9)
                self.assertGreater(h * w, 0)

    def test_known_sizes(self):
        self.assertEqual(gi.aspect_ratio_preserving_size(301, 517, 16, 2520, 3), (576, 1008))
        self.assertEqual(gi.aspect_ratio_preserving_size(64, 64, 16, 2520, 3), (768, 768))
        self.assertEqual(gi.aspect_ratio_preserving_size(2, 2000, 16, 2520, 3), (48, 13440))     # a strip keeps one row
        self.assertEqual(gi.aspect_ratio_preserving_size(2000, 2, 16, 2520, 3), (13440, 48))     # ... or one column

    def test_a_picture_that_cannot_make_a_patch_is_an_error_not_a_zero(self):
        with self.assertRaises(ValueError):
            gi.aspect_ratio_preserving_size(1, 1, 16, 4, 3)

    def test_the_budget_follows_the_engines_limit(self):
        self.assertEqual(gi.soft_token_budget(280), 280)
        self.assertEqual(gi.soft_token_budget(300), 280)
        self.assertEqual(gi.soft_token_budget(1120), 1120)
        self.assertEqual(gi.soft_token_budget(5000), 1120)
        self.assertEqual(gi.soft_token_budget(70), 70)
        self.assertEqual(gi.soft_token_budget(12), 12)             # below the smallest the processor offers: as asked
        self.assertEqual(gi.soft_token_budget(1120, wanted=140), 140)
        self.assertEqual(gi.soft_token_budget(100, wanted=280), 70)


@unittest.skipUnless(HAVE_IMAGING, "needs Pillow and numpy")
class PreprocessTest(unittest.TestCase):
    def png(self, image):
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()

    def test_not_an_image_is_a_value_error(self):
        with self.assertRaises(ValueError):
            gi.preprocess(b"this is not a picture")

    def test_the_patches_are_cut_row_by_row_each_as_row_column_channel(self):
        # 768 x 768 is a size the processor leaves alone at 280 soft tokens (48 x 48 patches), so the pixels reach the
        # patches untouched (a smaller picture would be scaled up to fill the budget)
        pixels = (np.arange(768 * 768 * 3, dtype=np.int64) % 251).astype(np.uint8).reshape(768, 768, 3)
        patches, gh, gw = gi.preprocess(self.png(Image.fromarray(pixels)), max_soft_tokens=280)
        self.assertEqual((gh, gw), (48, 48))
        self.assertEqual(patches.shape, (2304, 768))
        self.assertTrue((patches[0].reshape(16, 16, 3) == pixels[:16, :16]).all())
        self.assertTrue((patches[1].reshape(16, 16, 3) == pixels[:16, 16:32]).all())      # the next patch to the right
        self.assertTrue((patches[48].reshape(16, 16, 3) == pixels[16:32, :16]).all())     # then the next row
        self.assertTrue((patches[-1].reshape(16, 16, 3) == pixels[-16:, -16:]).all())

    def test_a_small_picture_is_scaled_up_to_fill_the_budget(self):
        patches, gh, gw = gi.preprocess(self.png(Image.new("RGB", (96, 48), (9, 9, 9))), max_soft_tokens=70)
        self.assertEqual((gh, gw), (15, 33))
        self.assertLessEqual(gh * gw, 70 * 9)


@unittest.skipUnless(HAVE_TRANSFORMERS, "needs Pillow, numpy and transformers with Gemma 4")
class AgainstTransformersTest(unittest.TestCase):
    def compare(self, image, tokens):
        processor = Gemma4ImageProcessorPil(max_soft_tokens=tokens)
        out = processor(images=[image], return_tensors="np")
        count = out["num_soft_tokens_per_image"][0]
        count = int(count if np.isscalar(count) else count[0])
        reference = np.rint(out["pixel_values"][0][:count * 9] * 255).astype(np.uint8)
        positions = out["image_position_ids"][0][:count * 9]
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        patches, gh, gw = gi.preprocess(buffer.getvalue(), tokens)
        self.assertEqual((gh // 3) * (gw // 3), count)
        self.assertEqual(patches.shape, reference.shape)
        self.assertTrue((patches == reference).all(), "patch bytes differ from the processor's")
        ys, xs = np.divmod(np.arange(gh * gw), gw)
        self.assertTrue((positions[:, 0] == xs).all() and (positions[:, 1] == ys).all(), "the grid order differs")

    def test_sizes_and_modes(self):
        rng = np.random.default_rng(3)
        for height, width, mode, tokens in ((301, 517, "RGB", 280), (1000, 50, "RGB", 280), (64, 64, "RGB", 280),
                                            (480, 640, "RGBA", 280), (100, 100, "L", 280), (333, 777, "RGB", 70),
                                            (333, 777, "RGB", 560), (2, 2000, "RGB", 280), (47, 48, "P", 140)):
            with self.subTest(size=(height, width), mode=mode, tokens=tokens):
                pixels = rng.integers(0, 256, (height, width, 4 if mode == "RGBA" else 3), dtype=np.uint8)
                image = Image.fromarray(pixels, "RGBA" if mode == "RGBA" else "RGB")
                if mode in ("L", "P"):
                    image = image.convert(mode)
                self.compare(image, tokens)

    def test_transparency_goes_onto_white(self):
        image = Image.new("RGBA", (96, 96), (0, 0, 0, 0))          # fully transparent, black underneath
        patches, _, _ = gi.preprocess(self.png_of(image), 70)
        self.assertTrue((patches == 255).all())
        self.compare(image, 70)

    @staticmethod
    def png_of(image):
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
