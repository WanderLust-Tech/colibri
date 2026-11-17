"""Settings inside a finished picture or song (media_meta)."""
import importlib.util
import json
import os
import struct
import sys
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import image_engine
import media_meta
import music_engine

PNG = image_engine.encode_png(4, 4, bytes(range(64)), 4)
WAV = music_engine.encode_wav(struct.pack("<8h", 1, 2, 3, 4, 5, 6, 7, 8), 44100, 2)
FIELDS = media_meta.record("image", "qwen-image", prompt="日本の猫 — a \"cat\"\nwith a newline", width=4, height=4,
                           steps=8, seed=42)


class RecordTest(unittest.TestCase):
    def test_only_what_is_set_is_kept_under_a_fixed_header(self):
        fields = media_meta.record("music", "ace", prompt="calm", lyrics=None, bpm=90, key=None)
        self.assertEqual(fields, {"app": "colibri", "version": 1, "kind": "music", "model": "ace",
                                  "prompt": "calm", "bpm": 90})

    def test_an_image_record_says_how_many_pictures_an_edit_started_from_and_never_holds_them(self):
        request = {"prompt": "a fox", "images": [b"x" * 1000, b"y" * 1000], "ref_resolution": 1024}
        result = {"width": 64, "height": 32, "steps": 4, "seed": 9}
        fields = media_meta.image_record(request, result, "qwen-image")
        self.assertEqual(fields["edit"], 2)
        self.assertEqual(fields["ref_resolution"], 1024)
        self.assertLess(len(json.dumps(fields)), 300)
        plain = media_meta.image_record({"prompt": "a fox"}, result, "qwen-image")
        self.assertNotIn("edit", plain)
        self.assertNotIn("ref_resolution", plain)

    def test_it_can_be_switched_off(self):
        self.assertTrue(media_meta.enabled({}))
        for off in ("0", "false", "No", " off "):
            self.assertFalse(media_meta.enabled({"COLI_MEDIA_META": off}), off)
        self.assertTrue(media_meta.enabled({"COLI_MEDIA_META": "1"}))


class PngTest(unittest.TestCase):
    def test_the_record_comes_back_exactly(self):
        out = media_meta.png_with(PNG, FIELDS)
        self.assertEqual(media_meta.read_png(out), FIELDS)

    def test_the_pixels_and_the_chunk_checksums_are_untouched(self):
        out = media_meta.png_with(PNG, FIELDS)
        self.assertEqual(image_engine.decode_png(out), image_engine.decode_png(PNG))
        for kind, start, end in media_meta._png_chunks(out):
            body = out[start + 8:end - 4]
            self.assertEqual(struct.unpack(">I", out[end - 4:end])[0], zlib.crc32(kind + body) & 0xFFFFFFFF, kind)

    def test_it_sits_before_the_pixels_so_a_reader_that_stops_early_finds_it(self):
        out = media_meta.png_with(PNG, FIELDS)
        kinds = [kind for kind, _s, _e in media_meta._png_chunks(out)]
        self.assertEqual(kinds, [b"IHDR", b"iTXt", b"IDAT", b"IEND"])

    def test_adding_twice_replaces_and_does_not_pile_up(self):
        twice = media_meta.png_with(media_meta.png_with(PNG, FIELDS), dict(FIELDS, seed=43))
        self.assertEqual(media_meta.read_png(twice)["seed"], 43)
        self.assertEqual([k for k, _s, _e in media_meta._png_chunks(twice)].count(b"iTXt"), 1)

    def test_a_png_without_it_or_somebody_elses_text_reads_as_none(self):
        self.assertIsNone(media_meta.read_png(PNG))
        other = media_meta.png_with(PNG, {"app": "other", "kind": "image"})
        self.assertIsNone(media_meta.read_png(other))

    def test_bytes_that_are_not_a_png_pass_through_and_read_as_none(self):
        self.assertEqual(media_meta.png_with(b"not a png", FIELDS), b"not a png")
        self.assertIsNone(media_meta.read_png(b"not a png"))
        self.assertIsNone(media_meta.read_png(PNG[:20]))                         # cut off inside a chunk

    def test_a_text_chunk_that_is_not_json_reads_as_none(self):
        junk = media_meta.PNG_MAGIC + b"".join(
            media_meta._chunk(kind, body) for kind, body in (
                (b"IHDR", PNG[16:29]), (b"iTXt", b"colibri\x00\x00\x00\x00\x00{oops"), (b"IEND", b"")))
        self.assertIsNone(media_meta.read_png(junk))


class WavTest(unittest.TestCase):
    def test_the_record_comes_back_exactly(self):
        fields = media_meta.record("music", "ace", prompt="calm piano", lyrics="la\nla", duration=30, seed=7)
        self.assertEqual(media_meta.read_wav(media_meta.wav_with(WAV, fields)), fields)

    def test_the_samples_and_the_header_sizes_stay_right(self):
        out = media_meta.wav_with(WAV, FIELDS)
        self.assertEqual(music_engine.decode_wav(out), music_engine.decode_wav(WAV))
        self.assertEqual(struct.unpack("<I", out[4:8])[0], len(out) - 8)
        self.assertEqual(out[8:36], WAV[8:36])                                    # the format part is the same

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "the gateway's audio reader needs numpy")
    def test_the_gateways_own_reader_still_takes_it(self):
        import openai_server
        mono = music_engine.encode_wav(struct.pack("<4h", 1, 2, 3, 4), 16000, 1)
        out = media_meta.wav_with(mono, FIELDS)
        # a list of samples, or an array of them when numpy is installed: compare as lists
        self.assertEqual(list(openai_server.decode_wav_mono16k(out, "x")), list(openai_server.decode_wav_mono16k(mono, "x")))

    def test_a_player_gets_a_title_and_a_software_name(self):
        out = media_meta.wav_with(WAV, FIELDS)
        self.assertIn(b"INAM", out)
        self.assertIn(b"ISFT", out)
        self.assertIn(b"Colibri\x00", out)

    def test_a_long_prompt_is_cut_in_the_title_only(self):
        fields = media_meta.record("music", "ace", prompt="x" * 1000, seed=1)
        out = media_meta.wav_with(WAV, fields)
        self.assertEqual(media_meta.read_wav(out)["prompt"], "x" * 1000)
        self.assertNotIn(b"x" * 201, out.replace(b"x" * 1000, b""))

    def test_a_wav_without_it_or_somebody_elses_comment_reads_as_none(self):
        self.assertIsNone(media_meta.read_wav(WAV))
        self.assertIsNone(media_meta.read_wav(media_meta.wav_with(WAV, {"app": "other"})))

    def test_bytes_that_are_not_a_wav_pass_through_and_read_as_none(self):
        self.assertEqual(media_meta.wav_with(b"nope", FIELDS), b"nope")
        self.assertIsNone(media_meta.read_wav(b"nope"))


if __name__ == "__main__":
    unittest.main()
