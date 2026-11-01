"""POST /v1/images/edits, end to end against the stub engine, and the picture decoder behind it.

Same arrangement as test_qwen_image_gateway.py: a real ImageEngine drives tools/qwenimage_stub.py over its
serve protocol and a real APIServer answers HTTP, nothing mocked in between. The stub's drawing depends on the
pictures it is handed, which is what lets these tests show that an uploaded picture, however it arrived
(multipart file, JSON data URL, bare base64), is the one the engine received.
"""
import base64
import http.client
import json
import struct
import sys
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_qwen_image_gateway import KEY, MODEL_ID, GatewayHarness, parse_sse  # noqa: E402

import image_engine  # noqa: E402


def picture_png(width=64, height=48, shade=0, channels=4):
    """A small PNG of our own making: a gradient, offset by `shade`."""
    px = bytearray()
    for y in range(height):
        for x in range(width):
            px += bytes([(x * 255 // width + shade) & 255, (y * 255 // height) & 255, (x + y + shade) & 255,
                         255][:channels] if channels != 2 else [(x + shade) & 255, 200])
    return image_engine.encode_png(width, height, bytes(px), channels)


def multipart(fields, files, boundary="----colibri-test-boundary"):
    """(content type, body) of a multipart/form-data request: files is [(field, filename, bytes)]."""
    out = bytearray()
    for name, value in fields.items():
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode()
    for name, filename, data in files:
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: image/png\r\n\r\n").encode() + data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", bytes(out)


def data_url(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


class EditCase(unittest.TestCase):
    harness = None

    @classmethod
    def setUpClass(cls):
        cls.harness = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.harness.close()

    def post_multipart(self, fields, files, key=KEY, path="/v1/images/edits"):
        ctype, body = multipart(fields, files)
        conn = http.client.HTTPConnection("127.0.0.1", self.harness.port, timeout=60)
        headers = {"Content-Type": ctype, "Content-Length": str(len(body))}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        conn.request("POST", path, body, headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, dict(response.getheaders()), data

    def edit_json(self, **body):
        body.setdefault("model", MODEL_ID)
        body.setdefault("prompt", "make the sky purple")
        return self.harness.request("/v1/images/edits", body)

    def ok_json(self, **body):
        status, _headers, raw = self.edit_json(**body)
        self.assertEqual(status, 200, raw[:300])
        return json.loads(raw)

    def error(self, status, **body):
        got, _headers, raw = self.edit_json(**body)
        self.assertEqual(got, status, raw[:300])
        return json.loads(raw)["error"]


class EditRequests(EditCase):
    def test_models_advertises_editing(self):
        status, _h, raw = self.harness.request("/v1/models")
        entry = json.loads(raw)["data"][0]
        self.assertEqual(status, 200)
        self.assertIn("image_edit", entry["capabilities"])
        edit = entry["image"]["edit"]
        self.assertEqual(edit["max_images"], image_engine.MAX_EDIT_IMAGES)
        self.assertEqual(edit["ref_resolution"], {"default": 768, "min": 384, "max": 1024})

    def test_json_data_url_edit(self):
        result = self.ok_json(image=data_url(picture_png(64, 48)), size="auto")
        info = result["colibri"]
        # no size given: the engine derives it from the picture, 4:3 at 768^2 pixels -> 896 x 672
        self.assertEqual((info["width"], info["height"]), (896, 672))
        png = base64.b64decode(result["data"][0]["b64_json"])
        w, h, channels, _px = image_engine.decode_png(png)
        self.assertEqual((w, h, channels), (896, 672, 4))

    def test_the_picture_is_what_the_engine_drew_from(self):
        a = self.ok_json(image=data_url(picture_png(64, 48, 0)), seed=5)
        again = self.ok_json(image=data_url(picture_png(64, 48, 0)), seed=5)
        other = self.ok_json(image=data_url(picture_png(64, 48, 90)), seed=5)
        self.assertEqual(a["data"][0]["b64_json"], again["data"][0]["b64_json"])
        self.assertNotEqual(a["data"][0]["b64_json"], other["data"][0]["b64_json"])

    def test_bare_base64_and_image_url_objects(self):
        png = picture_png()
        bare = self.ok_json(image=base64.b64encode(png).decode(), seed=1)
        wrapped = self.ok_json(images=[{"image_url": data_url(png)}], seed=1)
        nested = self.ok_json(images=[{"image_url": {"url": data_url(png)}}], seed=1)
        self.assertEqual(bare["data"][0]["b64_json"], wrapped["data"][0]["b64_json"])
        self.assertEqual(bare["data"][0]["b64_json"], nested["data"][0]["b64_json"])

    def test_multipart_upload_is_the_same_edit(self):
        png = picture_png()
        via_json = self.ok_json(image=data_url(png), seed=2, size="512x512", ref_resolution=384)
        status, _h, raw = self.post_multipart(
            {"model": MODEL_ID, "prompt": "make the sky purple", "seed": "2", "size": "512x512",
             "ref_resolution": "384", "n": "1"}, [("image", "photo.png", png)])
        self.assertEqual(status, 200, raw[:300])
        self.assertEqual(json.loads(raw)["data"][0]["b64_json"], via_json["data"][0]["b64_json"])

    def test_several_pictures(self):
        one, two = picture_png(64, 48, 0), picture_png(48, 64, 70)
        status, _h, raw = self.post_multipart({"model": MODEL_ID, "prompt": "combine them", "seed": "3"},
                                              [("image[]", "a.png", one), ("image[]", "b.png", two)])
        self.assertEqual(status, 200, raw[:300])
        both = json.loads(raw)
        only_second = self.ok_json(prompt="combine them", image=data_url(two), seed=3)
        self.assertNotEqual(both["data"][0]["b64_json"], only_second["data"][0]["b64_json"])
        # the size follows the LAST picture (3:4, portrait)
        self.assertLess(both["colibri"]["width"], both["colibri"]["height"])

    def test_given_size_and_alpha_pictures(self):
        for channels in (1, 3, 4, 2):
            result = self.ok_json(image=data_url(picture_png(40, 40, channels=channels)), size="320x256", seed=4)
            self.assertEqual((result["colibri"]["width"], result["colibri"]["height"]), (320, 256))

    def test_streaming_edit(self):
        status, _h, raw = self.harness.request("/v1/images/edits", {
            "model": MODEL_ID, "prompt": "make it blue", "image": data_url(picture_png()), "stream": True,
            "size": "256x256", "steps": 3})
        self.assertEqual(status, 200)
        events = parse_sse(raw.decode())
        names = [name for name, _d in events]
        self.assertIn("image_generation.progress", names)
        self.assertEqual(names[-2:], ["image_generation.completed", None])
        done = dict(events)["image_generation.completed"]
        self.assertEqual(done["colibri"]["steps"], 3)

    def test_the_generations_route_is_unchanged(self):
        status, _h, raw = self.harness.generate(prompt="a fox", size="256x256", seed=1)
        self.assertEqual(status, 200, raw[:200])
        # and it does not take pictures: an unknown field is simply not read
        self.assertNotIn("image", json.loads(raw)["colibri"])


class EditRefusals(EditCase):
    def test_a_picture_is_required(self):
        self.assertEqual(self.error(400)["param"], "image")
        self.assertEqual(self.error(400, image=[])["param"], "image")

    def test_too_many_pictures(self):
        error = self.error(400, images=[data_url(picture_png())] * (image_engine.MAX_EDIT_IMAGES + 1))
        self.assertIn("At most", error["message"])

    def test_masks_are_not_supported(self):
        self.assertEqual(self.error(400, image=data_url(picture_png()), mask=data_url(picture_png()))["param"], "mask")
        status, _h, raw = self.post_multipart({"model": MODEL_ID, "prompt": "x"},
                                              [("image", "a.png", picture_png()), ("mask", "m.png", picture_png())])
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["error"]["param"], "mask")

    def test_nothing_is_fetched_or_read_on_the_clients_say_so(self):
        for bad in ("http://127.0.0.1:1/x.png", "https://example.com/a.png", "file:///etc/passwd", "ftp://h/x"):
            self.assertIn("does not fetch", self.error(400, image=bad)["message"])
            self.assertIn("does not fetch", self.error(400, images=[{"image_url": bad}])["message"])
        self.assertIn("file_id", self.error(400, images=[{"file_id": "file-123"}])["message"])
        # a local path is not base64, so it is not a picture
        self.assertEqual(self.error(400, image="C:/Users/me/photo.png")["param"], "image")
        self.assertEqual(self.error(400, image="/etc/passwd")["param"], "image")

    def test_bad_pictures(self):
        self.assertIn("not valid base64", self.error(400, image="@@@not base64@@@")["message"])
        self.assertEqual(self.error(400, image=base64.b64encode(b"this is not a picture").decode())["param"], "image")
        self.assertEqual(self.error(400, image=data_url(picture_png()[:60]))["param"], "image")   # truncated
        self.assertIn("base64-encoded", self.error(400, image="data:image/png,abc")["message"])
        self.assertEqual(self.error(400, image=7)["param"], "image")

    def test_bad_parameters(self):
        png = data_url(picture_png())
        self.assertEqual(self.error(400, image=png, ref_resolution=100)["param"], "ref_resolution")
        self.assertEqual(self.error(400, image=png, ref_resolution=2000)["param"], "ref_resolution")
        self.assertEqual(self.error(400, image=png, ref_resolution="big")["param"], "ref_resolution")
        self.assertEqual(self.error(400, image=png, width=512)["param"], "height")        # one side alone
        self.assertEqual(self.error(400, image=png, size="300x256")["param"], "size")
        self.assertEqual(self.error(400, image=png, prompt="  ")["param"], "prompt")
        self.assertEqual(self.error(400, image=png, n=2)["param"], "n")
        self.assertEqual(self.error(400, image=png, steps=1)["param"], "steps")

    def test_the_model_and_the_key_are_still_checked(self):
        self.assertEqual(self.error(404, image=data_url(picture_png()), model="other")["code"], "model_not_found")
        status, _h, _raw = self.harness.request("/v1/images/edits", {"model": MODEL_ID, "prompt": "x"}, key=None)
        self.assertEqual(status, 401)

    def test_other_content_types(self):
        status, _h, raw = self.harness.request("/v1/images/edits", None, headers={"Content-Type": "text/plain"},
                                               method="POST")
        self.assertIn(status, (400, 415))
        conn = http.client.HTTPConnection("127.0.0.1", self.harness.port, timeout=30)
        conn.request("POST", "/v1/images/edits", b"x=1", {"Content-Type": "application/x-www-form-urlencoded",
                                                          "Authorization": f"Bearer {KEY}"})
        self.assertEqual(conn.getresponse().status, 415)
        conn.close()

    def test_a_broken_multipart_body(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.harness.port, timeout=30)
        body = b"no boundary in here"
        conn.request("POST", "/v1/images/edits", body, {"Content-Type": "multipart/form-data; boundary=zzz",
                                                         "Content-Length": str(len(body)),
                                                         "Authorization": f"Bearer {KEY}"})
        self.assertEqual(conn.getresponse().status, 400)
        conn.close()
        status, _h, raw = self.post_multipart({"model": MODEL_ID, "prompt": "x", "steps": "many"},
                                              [("image", "a.png", picture_png())])
        self.assertEqual((status, json.loads(raw)["error"]["param"]), (400, "steps"))

    def test_the_body_limit(self):
        big = b"\0" * (image_engine.MAX_EDIT_BODY + 1)
        conn = http.client.HTTPConnection("127.0.0.1", self.harness.port, timeout=60)
        conn.putrequest("POST", "/v1/images/edits")
        conn.putheader("Content-Type", "application/octet-stream")
        conn.putheader("Content-Length", str(len(big)))
        conn.putheader("Authorization", f"Bearer {KEY}")
        conn.endheaders()
        try:
            conn.send(big[:1 << 16])        # the gateway answers from the headers alone
        except OSError:
            pass
        self.assertEqual(conn.getresponse().status, 400)
        conn.close()

    def test_the_engine_survives_all_of_it(self):
        status, _h, raw = self.harness.generate(prompt="still here", size="256x256", seed=1)
        self.assertEqual(status, 200, raw[:200])


class EditWithoutAnEngineThatEdits(unittest.TestCase):
    def test_refused_up_front_and_not_advertised(self):
        harness = GatewayHarness(QWENIMAGE_STUB_EDIT=0)
        try:
            self.assertFalse(harness.engine.can_edit)
            status, _h, raw = harness.request("/v1/models")
            entry = json.loads(raw)["data"][0]
            self.assertEqual(entry["capabilities"], ["image_generation"])
            self.assertNotIn("edit", entry["image"])
            status, _h, raw = harness.request("/v1/images/edits", {
                "model": MODEL_ID, "prompt": "x", "image": data_url(picture_png())})
            self.assertEqual(status, 400)
            self.assertIn("cannot edit", json.loads(raw)["error"]["message"])
            # the engine is untouched
            self.assertEqual(harness.generate(prompt="a fox", size="256x256", seed=1)[0], 200)
        finally:
            harness.close()


class DecodePicture(unittest.TestCase):
    def test_png_kinds_become_rgba(self):
        for channels, first in ((1, [0, 0, 0, 255]), (3, [0, 0, 0, 255]), (4, [0, 0, 0, 255]), (2, [0, 0, 0, 200])):
            got = image_engine.decode_picture(picture_png(8, 6, channels=channels))
            self.assertEqual((got["width"], got["height"], len(got["rgba"])), (8, 6, 8 * 6 * 4))
            self.assertEqual(list(got["rgba"][:4]), first, channels)

    def test_the_header_is_checked_before_any_pixels(self):
        # 40 000 x 40 000 claimed in a few dozen bytes: refused (without Pillow, or shrunk-and-refused with it)
        header = struct.pack(">IIBBBBB", 40000, 40000, 8, 6, 0, 0, 0)

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
        bomb = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\0" * 100)) + chunk(b"IEND", b"")
        with self.assertRaises(image_engine.ImageRequestError):
            image_engine.decode_picture(bomb)

    def test_png_inflate_is_bounded_by_its_header(self):
        header = struct.pack(">IIBBBBB", 16, 16, 8, 6, 0, 0, 0)

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
        # 16 x 16 RGBA needs 1040 bytes of image data; this stream inflates to 50 MB of zeros
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\0" * (50 << 20), 9)) + chunk(b"IEND", b"")
        with self.assertRaises(ValueError):
            image_engine.decode_png(png)

    def test_garbage_is_refused_with_a_reason(self):
        for blob in (b"", b"GIF89a-not-really", b"\x89PNG\r\n\x1a\n" + b"\0" * 40, bytes(range(256))):
            with self.assertRaises(image_engine.ImageRequestError) as raised:
                image_engine.decode_picture(blob)
            self.assertEqual(raised.exception.param, "image")

    def test_edit_pictures_counts(self):
        with self.assertRaises(image_engine.ImageRequestError):
            image_engine.edit_pictures([])
        with self.assertRaises(image_engine.ImageRequestError):
            image_engine.edit_pictures([picture_png()] * 9)
        self.assertEqual(len(image_engine.edit_pictures([picture_png()] * 2)), 2)

    @unittest.skipUnless(image_engine._pillow(), "needs Pillow")
    def test_with_pillow_other_formats_and_shrinking(self):
        import io
        from PIL import Image
        im = Image.new("RGB", (3000, 1500), (200, 30, 30))
        for fmt in ("JPEG", "WEBP", "BMP", "PNG"):
            buf = io.BytesIO()
            im.save(buf, fmt)
            got = image_engine.decode_picture(buf.getvalue())
            self.assertLessEqual(got["width"] * got["height"], image_engine.MAX_EDIT_PIXELS, fmt)
            self.assertLessEqual(max(got["width"], got["height"]), image_engine.MAX_EDIT_SIDE, fmt)
            self.assertLessEqual(abs(got["width"] - 2 * got["height"]), 2, fmt)           # the 2:1 aspect is kept
            self.assertEqual(len(got["rgba"]), got["width"] * got["height"] * 4)
        pal = Image.new("P", (20, 10))
        pal.putpalette([255, 0, 0] * 256)
        buf = io.BytesIO()
        pal.save(buf, "PNG")                         # palette PNG: decode_png refuses it, Pillow reads it
        self.assertEqual(list(image_engine.decode_picture(buf.getvalue())["rgba"][:4]), [255, 0, 0, 255])


if __name__ == "__main__":
    unittest.main()
