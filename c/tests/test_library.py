"""The kept pictures and songs, listed and served (library.py, GET /v1/library on both servers)."""
import json
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import image_engine  # noqa: E402
import library  # noqa: E402
import media_meta  # noqa: E402
import music_engine  # noqa: E402
from openai_server import APIServer  # noqa: E402
from test_router import Rig  # noqa: E402

KEY = "k"
SETTINGS = {"app": "colibri", "version": 1, "kind": "image", "model": "m", "prompt": "a fox", "seed": 7}


def png(width=4, height=3, settings=None):
    data = image_engine.encode_png(width, height, bytes(width * height * 4), 4)
    return media_meta.png_with(data, settings) if settings else data


def wav(frames=48000, settings=None):
    data = music_engine.encode_wav(bytes(frames * 4), 48000, 2)
    return media_meta.wav_with(data, settings) if settings else data


class Folders:
    """Two temporary folders standing in for ~/colibri-images and ~/colibri-music."""

    def make(self, test):
        self.tmp = tempfile.TemporaryDirectory()
        test.addCleanup(self.tmp.cleanup)
        self.images = Path(self.tmp.name, "images")
        self.music = Path(self.tmp.name, "music")
        self.images.mkdir()
        self.music.mkdir()
        patcher = mock.patch.dict(os.environ, {"COLI_IMAGE_DIR": str(self.images), "COLI_MUSIC_DIR": str(self.music)})
        patcher.start()
        test.addCleanup(patcher.stop)
        return self

    def put(self, folder, name, data, age=0):
        path = folder / name
        path.write_bytes(data)
        moment = time.time() - age
        os.utime(path, (moment, moment))
        return path


class ListingTest(unittest.TestCase):
    def setUp(self):
        self.dirs = Folders().make(self)

    def test_an_empty_or_missing_library_is_an_empty_list(self):
        self.assertEqual(library.listing(), {"object": "list", "data": [], "total": 0, "has_more": False})
        self.dirs.images.rmdir()
        self.assertEqual(library.listing("images")["total"], 0)

    def test_the_newest_come_first_across_both_kinds_and_say_what_they_are(self):
        self.dirs.put(self.dirs.images, "old.png", png(4, 3, SETTINGS), age=300)
        self.dirs.put(self.dirs.music, "song.wav", wav(48000, dict(SETTINGS, kind="music", prompt="calm")), age=200)
        self.dirs.put(self.dirs.images, "new.png", png(2, 2), age=100)
        found = library.listing()
        self.assertEqual([item["name"] for item in found["data"]], ["new.png", "song.wav", "old.png"])
        self.assertEqual(found["total"], 3)
        new, song, old = found["data"]
        self.assertEqual((new["kind"], new["url"], new["width"], new["height"]), ("images", "/v1/library/images/new.png", 2, 2))
        self.assertNotIn("settings", new)                                  # a picture from somewhere else has none
        self.assertEqual((song["kind"], song["duration"], song["sample_rate"], song["settings"]["prompt"]),
                         ("music", 1.0, 48000, "calm"))
        self.assertEqual((old["width"], old["height"], old["settings"]["seed"]), (4, 3, 7))
        self.assertEqual(old["bytes"], len(png(4, 3, SETTINGS)))

    def test_a_kind_can_be_asked_for_and_the_other_is_left_out(self):
        self.dirs.put(self.dirs.images, "a.png", png())
        self.dirs.put(self.dirs.music, "a.wav", wav(100))
        self.assertEqual([i["name"] for i in library.listing("images")["data"]], ["a.png"])
        self.assertEqual([i["name"] for i in library.listing("music")["data"]], ["a.wav"])
        with self.assertRaises(library.LibraryError) as caught:
            library.listing("videos")
        self.assertEqual(caught.exception.status, 404)

    def test_pages(self):
        for n in range(5):
            self.dirs.put(self.dirs.images, f"p{n}.png", png(), age=100 - n * 10)
        first = library.listing("images", limit=2)
        self.assertEqual([i["name"] for i in first["data"]], ["p4.png", "p3.png"])
        self.assertEqual((first["total"], first["has_more"]), (5, True))
        last = library.listing("images", limit=2, offset=4)
        self.assertEqual(([i["name"] for i in last["data"]], last["has_more"]), (["p0.png"], False))

    def test_files_of_other_kinds_and_folders_are_not_listed(self):
        self.dirs.put(self.dirs.images, "notes.txt", b"x")
        self.dirs.put(self.dirs.images, "a.png", png())
        (self.dirs.images / "folder.png").mkdir()
        self.assertEqual([i["name"] for i in library.listing()["data"]], ["a.png"])

    def test_a_file_that_is_not_what_its_name_says_is_listed_without_details_and_does_not_break_the_list(self):
        self.dirs.put(self.dirs.images, "junk.png", b"not a png at all")
        self.dirs.put(self.dirs.music, "junk.wav", b"RIFFxxxx")
        found = library.listing()["data"]
        self.assertEqual(sorted(i["name"] for i in found), ["junk.png", "junk.wav"])
        for item in found:
            self.assertEqual(set(item) - {"kind", "name", "url", "bytes", "modified"}, set())

    def test_the_arguments_are_plain_numbers_in_range(self):
        self.assertEqual(library.page_arguments({}), (None, 50, 0))
        self.assertEqual(library.page_arguments({"kind": ["music"], "limit": ["200"], "offset": ["3"]}), ("music", 200, 3))
        for query in ({"limit": ["0"]}, {"limit": ["201"]}, {"limit": ["x"]}, {"offset": ["-1"]}, {"kind": ["nope"]}, {"limit": ["1.5"]}):
            with self.assertRaises(library.LibraryError, msg=query) as caught:
                library.page_arguments(query)
            self.assertEqual(caught.exception.status, 400)


class NameTest(unittest.TestCase):
    def setUp(self):
        self.dirs = Folders().make(self)
        self.dirs.put(self.dirs.images, "ok.png", png())
        self.dirs.put(self.dirs.music, "ok.wav", wav(100))
        (self.dirs.images / "dir.png").mkdir()
        Path(self.dirs.tmp.name, "secret.png").write_bytes(png())

    def test_a_kept_file_resolves(self):
        self.assertEqual(os.path.basename(library.resolve("images", "ok.png")), "ok.png")
        self.assertEqual(os.path.basename(library.resolve("music", "ok.wav")), "ok.wav")

    def test_nothing_that_is_not_one_plain_file_of_the_kind_in_the_folder_resolves(self):
        for kind, name in (("images", "../secret.png"), ("images", "..\\secret.png"), ("images", "sub/ok.png"),
                           ("images", "C:ok.png"), ("images", "C:\\x\\ok.png"), ("images", "/etc/passwd.png"),
                           ("images", ".."), ("images", "."), ("images", ""), ("images", "ok.txt"), ("images", "ok.wav"),
                           ("images", "dir.png"), ("images", "missing.png"), ("images", "ok.png\x00.txt"), ("music", "ok.png"),
                           ("images", None), ("images", "ok.png/"), ("images", "%2e%2e/secret.png")):
            self.assertIsNone(library.resolve(kind, name), (kind, name))

    def test_a_link_out_of_the_folder_does_not_resolve(self):
        link = self.dirs.images / "link.png"
        try:
            os.symlink(Path(self.dirs.tmp.name, "secret.png"), link)
        except (OSError, NotImplementedError):
            self.skipTest("this system will not make a symbolic link here")
        self.assertIsNone(library.resolve("images", "link.png"))
        self.assertNotIn("link.png", [i["name"] for i in library.listing("images")["data"]])

    def test_byte_ranges(self):
        self.assertIsNone(library.byte_range(None, 100))
        self.assertEqual(library.byte_range("bytes=0-9", 100), (0, 9))
        self.assertEqual(library.byte_range("bytes=90-", 100), (90, 99))
        self.assertEqual(library.byte_range("bytes=90-500", 100), (90, 99))
        self.assertEqual(library.byte_range("bytes=-10", 100), (90, 99))
        self.assertEqual(library.byte_range("bytes=-500", 100), (0, 99))
        for header in ("bytes=100-", "bytes=50-40", "bytes=-0"):
            self.assertEqual(library.byte_range(header, 100), "unsatisfiable", header)
        for header in ("bytes=0-1,5-6", "items=0-1", "bytes=-", "garbage"):
            self.assertIsNone(library.byte_range(header, 100), header)


class ServedMixin:
    """The same requests against either server; subclasses set `self.base` and `self.headers`."""

    def get(self, path, headers=None):
        request = Request(self.base + path, headers={**self.headers, **(headers or {})})
        try:
            return urlopen(request, timeout=5)
        except HTTPError as error:
            self.addCleanup(error.close)
            return error

    def test_the_list_needs_the_key(self):
        request = Request(self.base + "/v1/library")
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 401)

    def test_it_lists_and_serves_a_picture_the_way_a_browser_can_cache(self):
        data = png(4, 3, SETTINGS)
        self.dirs.put(self.dirs.images, "fox.png", data)
        with self.get("/v1/library?kind=images") as response:
            listing = json.loads(response.read())
        self.assertEqual(listing["data"][0]["settings"]["prompt"], "a fox")
        with self.get(listing["data"][0]["url"]) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), data)
            self.assertEqual(response.headers["Content-Type"], "image/png")
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            self.assertEqual(response.headers["Accept-Ranges"], "bytes")
            self.assertIn("max-age=86400", response.headers["Cache-Control"])
            self.assertIn("private", response.headers["Cache-Control"])

    def test_a_player_can_ask_for_a_byte_range(self):
        data = wav(1000)
        self.dirs.put(self.dirs.music, "s.wav", data)
        with self.get("/v1/library/music/s.wav", {"Range": "bytes=44-99"}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), data[44:100])
            self.assertEqual(response.headers["Content-Range"], f"bytes 44-99/{len(data)}")
            self.assertEqual(response.headers["Content-Type"], "audio/wav")
        past = self.get("/v1/library/music/s.wav", {"Range": f"bytes={len(data)}-"})
        self.assertEqual(past.status if hasattr(past, "status") else past.code, 416)

    def test_a_name_that_is_not_a_kept_file_is_not_found_whatever_it_looks_like(self):
        self.dirs.put(self.dirs.images, "ok.png", png())
        Path(self.dirs.tmp.name, "secret.png").write_bytes(b"secret")
        for path in ("/v1/library/images/missing.png", "/v1/library/images/..%2fsecret.png", "/v1/library/images/%2e%2e/secret.png",
                     "/v1/library/images/..%5csecret.png", "/v1/library/images/sub%2fok.png", "/v1/library/videos/ok.png",
                     "/v1/library/images/", "/v1/library/images", "/v1/library/music/ok.png"):
            response = self.get(path)
            code = getattr(response, "status", None) or response.code
            self.assertEqual(code, 404, path)
            self.assertNotIn(b"secret", response.read(), path)

    def test_bad_paging_is_a_400_with_the_usual_error_body(self):
        response = self.get("/v1/library?limit=0")
        self.assertEqual(response.code, 400)
        self.assertEqual(json.loads(response.read())["error"]["param"], "limit")

    def test_a_cross_site_page_cannot_read_it(self):
        self.dirs.put(self.dirs.images, "ok.png", png())
        response = self.get("/v1/library", {"Sec-Fetch-Site": "cross-site", "Origin": "http://evil.example"})
        self.assertEqual(response.code, 403)


class SingleServerTest(ServedMixin, unittest.TestCase):
    def setUp(self):
        self.dirs = Folders().make(self)
        self.server = APIServer(("127.0.0.1", 0), None, "m", KEY)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.headers = {"Authorization": f"Bearer {KEY}"}

    def stop(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class RouterTest(ServedMixin, unittest.TestCase):
    def setUp(self):
        self.dirs = Folders().make(self)
        self.rig = Rig(self, [], api_key=KEY)
        self.base = f"http://127.0.0.1:{self.rig.port}"
        self.headers = {"Authorization": f"Bearer {KEY}"}


if __name__ == "__main__":
    unittest.main()
