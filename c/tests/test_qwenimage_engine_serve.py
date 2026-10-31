"""The real qwenimage engine behind image_engine.ImageEngine, on the tiny fixture.

The gateway and TUI tests run against tools/qwenimage_stub.py; this one holds
the C engine to the same protocol: frames of the right size, the same seed giving
the same bytes, a refused size or a CANCEL leaving the engine ready for the next
request, and EOF as a clean exit.

Needs QWENIMAGE_TINY (a directory written by tools/make_qwenimage_tiny.py) and
the engine built (make qwenimage); skipped otherwise.
  QWENIMAGE_TINY=/path/tiny python3 -m unittest -v tests.test_qwenimage_engine_serve"""
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import image_engine  # noqa: E402

TINY = os.environ.get("QWENIMAGE_TINY")
ENGINE = HERE / ("qwenimage.exe" if sys.platform == "win32" else "qwenimage")


@unittest.skipUnless(TINY and ENGINE.exists(), "needs QWENIMAGE_TINY and a built qwenimage")
class RealEngineServe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = []
        cls.engine = image_engine.ImageEngine(ENGINE, TINY, load_timeout=300,
                                              on_log=cls.logs.append, on_stderr=cls.logs.append)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def generate(self, seed, width=256, height=256, **kw):
        return self.engine.generate("a red fox in the snow", width, height, 2, seed, **kw)

    def test_image_frame_and_determinism(self):
        first = self.generate(7)
        self.assertEqual((first["width"], first["height"], first["channels"]), (256, 256, 4))
        self.assertEqual(len(first["rgba"]), 256 * 256 * 4)
        self.assertEqual(first["seed"], 7)
        self.assertEqual(self.generate(7)["rgba"], first["rgba"], "same seed, different bytes")
        self.assertNotEqual(self.generate(8)["rgba"], first["rgba"], "the seed changed nothing")

    def test_non_square_sizes(self):
        wide = self.generate(3, width=320, height=256)
        self.assertEqual((wide["width"], wide["height"]), (320, 256))
        self.assertEqual(len(wide["rgba"]), 320 * 256 * 4)

    def test_refused_size_keeps_the_engine(self):
        with self.assertRaises(image_engine.ImageEngineError):
            self.generate(1, width=300)            # not a multiple of 32
        self.assertEqual(len(self.generate(1)["rgba"]), 256 * 256 * 4)

    def test_cancel_keeps_the_engine(self):
        # Through the client: a job long enough that the CANCEL, sent at the first
        # frame, lands before the last step (two steps of the tiny model can finish
        # before it arrives, which would be a race, not a result).
        with self.assertRaises(image_engine.ImageCancelled):
            self.engine.generate("a fox", 512, 512, 100, 2, cancelled=lambda: True)
        self.assertEqual(len(self.generate(2)["rgba"]), 256 * 256 * 4)

    def test_cancel_pipelined_with_its_gen(self):
        # The protocol itself: a CANCEL written right behind its GEN is answered
        # "cancelled", whatever the speed of the machine.
        self.engine._send("GEN", {"id": "c1", "prompt": "x", "width": 256, "height": 256, "steps": 2})
        self.engine._send("CANCEL", {"id": "c1"})
        self.assertEqual(self._final("c1"), "cancelled")

    def test_abandoned_request_does_not_swallow_the_next(self):
        # A callback that raises leaves generate() without the final frame while
        # the engine is still drawing: the engine keeps the next GEN instead of
        # dropping it, and the client drains the abandoned frames first.
        def boom(frame):
            if frame.get("stage") == "denoise":
                raise ConnectionResetError("client went away")
        with self.assertRaises(ConnectionResetError):
            self.engine.generate("a fox", 512, 512, 6, 1, on_progress=boom)
        self.assertEqual(len(self.generate(5)["rgba"]), 256 * 256 * 4)

    def test_engine_refuses_what_it_cannot_do(self):
        with self.assertRaisesRegex(image_engine.ImageEngineError, "prompt too long"):
            self.engine.generate("7" * 4000, 256, 256, 2, 1)    # 4 KB, one token per digit
        for steps in (1, 201):
            with self.assertRaises(image_engine.ImageEngineError):
                self.engine._send("GEN", {"id": "raw", "prompt": "x", "width": 256, "height": 256, "steps": steps})
                raise image_engine.ImageEngineError(self._final("raw"))
        self.assertEqual(len(self.generate(6)["rgba"]), 256 * 256 * 4)

    # -- editing ------------------------------------------------------------
    @staticmethod
    def picture(width, height, phase=0):
        """A smooth RGBA gradient with a clear corner, as the {"width","height","rgba"} an edit takes."""
        px = bytearray(width * height * 4)
        for y in range(height):
            for x in range(width):
                i = (y * width + x) * 4
                px[i:i + 4] = bytes(((x * 255 // width + phase) & 255, (y * 255 // height + 2 * phase) & 255,
                                     (x + y + phase) * 255 // (width + height) & 255,
                                     0 if x < width // 4 and y < height // 4 else 255))
        return {"width": width, "height": height, "rgba": bytes(px)}

    def edit(self, seed, pictures, **kw):
        return self.engine.generate("make it purple", None, None, 2, seed, images=pictures, ref_resolution=384, **kw)

    def test_ready_reports_editing(self):
        self.assertTrue(self.engine.can_edit)
        self.assertEqual(self.engine.info["max_edit_images"], image_engine.MAX_EDIT_IMAGES)
        self.assertEqual(self.engine.info["ref_resolution"], {"default": 768, "min": 384, "max": 1024})

    def test_edit_frame_determinism_and_inputs(self):
        pic = self.picture(256, 256)
        first = self.edit(7, [pic])
        # the size follows the picture's aspect at ref_resolution^2 pixels: 384 x 384
        self.assertEqual((first["width"], first["height"], first["channels"]), (384, 384, 4))
        self.assertEqual(len(first["rgba"]), 384 * 384 * 4)
        self.assertEqual(self.edit(7, [pic])["rgba"], first["rgba"], "same seed and picture, different bytes")
        self.assertNotEqual(self.edit(8, [pic])["rgba"], first["rgba"], "the seed changed nothing")
        self.assertNotEqual(self.edit(7, [self.picture(256, 256, phase=90)])["rgba"], first["rgba"],
                            "the picture changed nothing")
        # an edit is not a plain generation, and a plain one still works afterwards
        self.assertNotEqual(self.generate(7, 384, 384)["rgba"], first["rgba"])
        self.assertEqual(len(self.generate(7)["rgba"]), 256 * 256 * 4)

    def test_edit_two_pictures_and_a_given_size(self):
        got = self.engine.generate("put the cat next to the dog", 320, 256, 2, 3, ref_resolution=384,
                                   images=[self.picture(200, 300), self.picture(300, 200, 40)])
        self.assertEqual((got["width"], got["height"]), (320, 256))
        self.assertEqual(len(got["rgba"]), 320 * 256 * 4)

    def test_edit_refusals_keep_the_engine(self):
        import base64
        good = base64.b64encode(self.picture(64, 64)["rgba"]).decode()

        def refused(images, extra=None, pattern=None):
            self.engine._send("GEN", {"id": "bad", "prompt": "x", "steps": 2, "images": images, **(extra or {})})
            message = self._final("bad")
            if pattern:
                self.assertRegex(message, pattern)

        refused([{"width": 64, "height": 64, "data": good[:-4]}], pattern="does not decode")      # short
        refused([{"width": 64, "height": 64, "data": "not base64!"}], pattern="does not decode")
        refused([{"width": 65, "height": 64, "data": good}], pattern="does not decode")           # wrong size
        refused([{"width": 0, "height": 64, "data": good}], pattern="needs width and height")
        refused([{"width": 64, "height": 64}], pattern="needs width and height")
        refused([], pattern="1 to 4")
        refused([{"width": 64, "height": 64, "data": good}] * 5, pattern="1 to 4")
        refused([{"width": 64, "height": 64, "data": good}], {"ref_resolution": 100}, pattern="ref_resolution")
        refused([{"width": 64, "height": 64, "data": good}], {"width": 300, "height": 256}, pattern="multiples of 32")
        refused([{"width": 4096, "height": 8, "data": good}], pattern="needs width and height|does not decode")
        self.assertEqual(len(self.generate(9)["rgba"]), 256 * 256 * 4)

    def test_edit_prompt_with_image_tokens_is_refused(self):
        with self.assertRaisesRegex(image_engine.ImageEngineError, "image_pad"):
            self.engine.generate("add a <|image_pad|> hat", None, None, 2, 1, ref_resolution=384,
                                 images=[self.picture(64, 64)])
        self.assertEqual(len(self.generate(9)["rgba"]), 256 * 256 * 4)

    def _final(self, rid):
        while True:
            kind, body, _payload = self.engine.events.get(timeout=60)
            if body.get("id") == rid and kind in ("IMAGE", "ERROR"):
                self.assertEqual(kind, "ERROR", body)
                return body.get("message")

    def test_progress_counts_completed_steps(self):
        seen = []
        self.generate(4, on_progress=lambda frame: seen.append((frame.get("stage"), frame.get("step"))))
        self.assertIn(("encode", 0), seen)
        self.assertIn(("denoise", 0), seen)
        self.assertIn(("decode", 2), seen)


@unittest.skipUnless(TINY and ENGINE.exists(), "needs QWENIMAGE_TINY and a built qwenimage")
class RealEngineExit(unittest.TestCase):
    def test_eof_is_a_clean_exit(self):
        engine = image_engine.ImageEngine(ENGINE, TINY, load_timeout=300, on_log=lambda m: None,
                                          on_stderr=lambda m: None)
        engine.close()
        self.assertEqual(engine.process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
