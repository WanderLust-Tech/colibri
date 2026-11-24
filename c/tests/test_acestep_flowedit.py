"""ACE-Step's flow-edit (upstream's overlay on text to music: edit a clip toward a new caption and lyrics, `flow_edit_morph` and its source caption,
source lyrics and window), from the request rules to the HTTP route and `coli music --edit`, against the stub engine.

The engine's own numbers (the paired source and target loop) are held to upstream's handler by `acestep --gen-ref` on the `b_fe*` and `s_fe*` cases
captured with `tools/acestep_ref_audio.py tasks` (docs/acestep.md). What is under test here is what the Python side adds: which requests are refused
before the engine hears of them (an engine that does not announce flow-edit would read the clip as something else), what reaches the engine, that the
clip's length is the song's, and what `coli music` makes of its flags.
"""
import base64
import importlib.machinery
import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from acestep_fixture import make_wav  # noqa: E402
from test_acestep_gateway import GatewayHarness, wav_of  # noqa: E402

import music_engine  # noqa: E402

B64 = lambda data: base64.b64encode(data).decode()  # noqa: E731
CLIP = B64(make_wav(2.5))
WITH = {"tasks": ["text2music", "cover", "repaint"], "flow_edit": True, "flow_edit_cover": True, "sampler": True,
        "audio_in": {"formats": ["wav", "mp3"], "max_seconds": 600, "ref_seconds": 30}}
WITHOUT = {k: v for k, v in WITH.items() if k not in ("flow_edit", "flow_edit_cover")}
BEFORE_COVER = {k: v for k, v in WITH.items() if k != "flow_edit_cover"}          # an engine from before the cover variant


def request(info=WITH, **body):
    body.setdefault("prompt", "bright ukulele")
    body.setdefault("flow_edit_morph", True)
    body.setdefault("src_audio", CLIP)
    return music_engine.music_request(body, info)


def refused(param, info=WITH, **body):
    body.setdefault("prompt", "bright ukulele")
    body.setdefault("flow_edit_morph", True)
    body.setdefault("src_audio", CLIP)
    try:
        music_engine.music_request(body, info)
    except music_engine.MusicRequestError as error:
        assert error.param == param, (error.param, str(error))
        return str(error)
    raise AssertionError(f"not refused: { {k: v for k, v in body.items() if k != 'src_audio'} }")


class Rules(unittest.TestCase):
    def test_a_flow_edit_is_text_to_music_on_a_clip_and_the_clips_length_is_the_songs(self):
        r = request()
        self.assertTrue(r["morph"])
        self.assertEqual((r["task"], r["duration"]), ("text2music", None))
        self.assertIsNotNone(r["src_audio"])
        self.assertEqual((r["edit_caption"], r["edit_lyrics"], r["n_min"], r["n_max"], r["n_avg"]), (None, None, None, None, None))

    def test_the_options_are_taken_as_given(self):
        r = request(flow_edit_source_caption="slow piano", flow_edit_source_lyrics="[Verse]\nla", flow_edit_n_min=0.2, flow_edit_n_max=0.7, flow_edit_n_avg=3,
                    velocity_norm_threshold=1.1, velocity_ema_factor=0.4, duration=99)           # a duration is not read: the clip's length wins
        self.assertEqual((r["edit_caption"], r["edit_lyrics"], r["n_min"], r["n_max"], r["n_avg"]), ("slow piano", "[Verse]\nla", 0.2, 0.7, 3))
        self.assertEqual((r["velocity_norm_threshold"], r["velocity_ema_factor"], r["duration"]), (1.1, 0.4, None))
        edges = request(flow_edit_n_min=0, flow_edit_n_max=1, flow_edit_n_avg=16)
        self.assertEqual((edges["n_min"], edges["n_max"], edges["n_avg"]), (0.0, 1.0, 16))

    def test_no_morph_is_what_it_was(self):
        r = music_engine.music_request({"prompt": "calm piano"}, WITH)
        self.assertFalse(r["morph"])
        self.assertEqual((r["edit_caption"], r["n_min"], r["n_avg"]), (None, None, None))

    def test_a_clip_for_text_to_music_is_still_refused_without_the_morph(self):
        refused("src_audio", flow_edit_morph=None)
        refused("src_audio", flow_edit_morph=False)

    def test_what_a_flow_edit_does_not_go_with(self):
        refused("flow_edit_morph", task="repaint")
        refused("src_audio", src_audio=None)
        refused("flow_edit_morph", flow_edit_morph="yes")
        refused("sampler_mode", sampler_mode="heun")
        refused("sampler_mode", infer_method="sde")
        refused("thinking", thinking=True)
        refused("use_adg", use_adg=True)
        request(sampler_mode="euler", infer_method="ode")                                     # naming the plain step is fine

    def test_a_flow_edit_on_a_covers_context(self):
        r = request(task="cover", flow_edit_source_caption="slow piano", flow_edit_n_max=0.7)
        self.assertEqual((r["morph"], r["task"], r["duration"], r["n_max"]), (True, "cover", None, 0.7))
        self.assertEqual((r["audio_cover_strength"], r["cover_noise_strength"]), (None, None))
        self.assertIn("update the engine", refused("task", BEFORE_COVER, task="cover"))      # the older engine would run it on silence: refused in front of it
        request(BEFORE_COVER)                                                                  # and text to music is still fine there
        for name in ("audio_cover_strength", "cover_noise_strength"):
            self.assertIn("not for a flow-edit", refused(name, task="cover", **{name: 0.5}))

    def test_bad_values_are_refused_by_name(self):
        for value in (-0.1, 1.1, "0.5", True):
            refused("flow_edit_n_min", flow_edit_n_min=value)
        for value in (-0.1, 1.5, "1"):
            refused("flow_edit_n_max", flow_edit_n_max=value)
        self.assertIn("above 0", refused("flow_edit_n_max", flow_edit_n_max=0))
        self.assertIn("not be before", refused("flow_edit_n_max", flow_edit_n_min=0.6, flow_edit_n_max=0.4))
        for value in (0, 17, 1.5, "2", True):
            refused("flow_edit_n_avg", flow_edit_n_avg=value)
        refused("flow_edit_source_caption", flow_edit_source_caption=3)
        refused("flow_edit_source_lyrics", flow_edit_source_lyrics=["la"])

    def test_the_fields_are_for_a_flow_edit_only(self):
        for body in (dict(flow_edit_source_caption="x"), dict(flow_edit_source_lyrics="x"), dict(flow_edit_n_min=0.1), dict(flow_edit_n_max=0.9), dict(flow_edit_n_avg=2)):
            name = next(iter(body))
            self.assertIn("set `flow_edit_morph`", refused(name, flow_edit_morph=None, src_audio=None, **body))

    def test_an_engine_that_does_not_announce_it_would_read_the_clip_as_something_else(self):
        self.assertIn("update the engine", refused("flow_edit_morph", WITHOUT))


class Gateway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = GatewayHarness(ACESTEP_STUB_FLOWEDIT=1, ACESTEP_STUB_SAMPLER=1)

    @classmethod
    def tearDownClass(cls):
        cls.rig.close()

    def song(self, **body):
        status, _, raw = self.rig.music(**body)
        return status, json.loads(raw)

    def test_the_model_card_says_the_engine_does_flow_edit(self):
        status, _, raw = self.rig.request("/v1/models", method="GET")
        self.assertEqual(status, 200)
        self.assertIs(json.loads(raw)["data"][0]["audio"]["flow_edit"], True)

    def test_a_flow_edit_is_as_long_as_its_clip_and_the_answer_says_so(self):
        status, out = self.song(prompt="bright ukulele", flow_edit_morph=True, src_audio=CLIP, flow_edit_source_caption="slow piano", steps=2, seed=3)
        self.assertEqual(status, 200, out)
        pcm, rate, channels = wav_of(out["data"][0]["b64_json"])
        self.assertAlmostEqual(len(pcm) / (2 * channels) / rate, 2.5, places=3)
        self.assertIs(out["colibri"]["flow_edit"], True)
        self.assertAlmostEqual(out["colibri"]["src_seconds"], 2.5, places=3)

    def test_a_flow_edit_on_a_covers_context_is_the_clips_length_too(self):
        status, out = self.song(prompt="bright ukulele", task="cover", flow_edit_morph=True, src_audio=CLIP, flow_edit_source_caption="slow piano", steps=2, seed=3)
        self.assertEqual(status, 200, out)
        self.assertIs(out["colibri"]["flow_edit"], True)
        self.assertEqual(out["colibri"]["task"], "cover")
        self.assertAlmostEqual(out["colibri"]["src_seconds"], 2.5, places=3)
        status, out = self.song(prompt="bright ukulele", task="cover", flow_edit_morph=True, src_audio=CLIP, audio_cover_strength=0.5)
        self.assertEqual((status, out["error"]["param"]), (400, "audio_cover_strength"))

    def test_the_model_card_says_the_cover_variant_too(self):
        status, _, raw = self.rig.request("/v1/models", method="GET")
        self.assertIs(json.loads(raw)["data"][0]["audio"]["flow_edit_cover"], True)

    def test_a_plain_song_says_it_was_not_one(self):
        status, out = self.song(prompt="calm piano", duration=2, steps=2, seed=1)
        self.assertEqual(status, 200, out)
        self.assertIs(out["colibri"]["flow_edit"], False)

    def test_the_window_and_the_averages_reach_the_engine(self):
        status, out = self.song(prompt="bright ukulele", flow_edit_morph=True, src_audio=CLIP, flow_edit_n_min=0.2, flow_edit_n_max=0.7, flow_edit_n_avg=2, steps=2, seed=4)
        self.assertEqual(status, 200, out)

    def test_a_bad_request_is_a_400_naming_it(self):
        status, out = self.song(prompt="bright ukulele", flow_edit_morph=True, src_audio=CLIP, flow_edit_n_avg=99)
        self.assertEqual(status, 400)
        self.assertEqual(out["error"]["param"], "flow_edit_n_avg")
        status, out = self.song(prompt="bright ukulele", flow_edit_morph=True)
        self.assertEqual(status, 400)
        self.assertEqual(out["error"]["param"], "src_audio")

    def test_the_form_fields_of_a_multipart_request_are_read_as_numbers(self):
        import openai_server
        for name in ("flow_edit_n_min", "flow_edit_n_max"):
            self.assertIn(name, openai_server._MUSIC_FLOAT_FIELDS)
        self.assertIn("flow_edit_n_avg", openai_server._MUSIC_INT_FIELDS)
        self.assertIn("flow_edit_morph", openai_server._MUSIC_BOOL_FIELDS)


class Plain(unittest.TestCase):
    """An engine without the announcement: the gateway refuses the request in front of it, with the reason."""

    @classmethod
    def setUpClass(cls):
        cls.rig = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.rig.close()

    def test_refused_with_the_reason(self):
        status, _, raw = self.rig.music(prompt="bright ukulele", flow_edit_morph=True, src_audio=CLIP)
        self.assertEqual(status, 400)
        self.assertIn("update the engine", json.loads(raw)["error"]["message"])

    def test_the_model_card_says_it_does_not(self):
        status, _, raw = self.rig.request("/v1/models", method="GET")
        card = json.loads(raw)["data"][0]["audio"]
        self.assertIs(card["flow_edit"], False)
        self.assertIs(card["flow_edit_cover"], False)


class Cli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = importlib.machinery.SourceFileLoader("coli_script_flowedit", str(HERE.parent / "coli"))
        spec = importlib.util.spec_from_loader("coli_script_flowedit", loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def args(self, **kw):
        base = dict(cover=None, repaint=None, extract=None, lego=None, complete=None, ref=None, track=None, tracks=None, dit=None, guidance=None,
                    cfg_start=None, cfg_end=None, adg=False, duration=None, repaint_from=None, repaint_to=None, repaint_mode=None,
                    repaint_strength=None, cover_strength=None, cover_noise=None, sampler=None, sde=False, velocity_norm=None, velocity_ema=None,
                    edit=None, edit_cover=False, edit_caption=None, edit_lyrics=None, n_min=None, n_max=None, n_avg=None, plan=False)
        base.update(kw)
        return type("Args", (), base)()

    def with_clip(self, **kw):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.wav")
            Path(path).write_bytes(make_wav(1.5))
            return self.coli.music_audio_flags(self.args(edit=path, **kw))

    def test_the_flags_become_the_requests_own_fields(self):
        body = self.with_clip(edit_caption="slow piano", edit_lyrics="la", n_min=0.2, n_max=0.7, n_avg=2)
        self.assertIs(body["flow_edit_morph"], True)
        self.assertEqual((body["flow_edit_source_caption"], body["flow_edit_source_lyrics"], body["flow_edit_n_min"], body["flow_edit_n_max"], body["flow_edit_n_avg"]),
                         ("slow piano", "la", 0.2, 0.7, 2))
        self.assertIn("src_audio", body)
        self.assertNotIn("task", body)
        none = self.coli.music_audio_flags(self.args())
        for field in ("flow_edit_morph", "flow_edit_source_caption", "flow_edit_n_min", "src_audio"):
            self.assertNotIn(field, none)

    def test_edit_cover_asks_for_the_cover_context(self):
        body = self.with_clip(edit_cover=True, edit_caption="slow piano")
        self.assertEqual((body["task"], body["flow_edit_morph"]), ("cover", True))
        with self.assertRaises(SystemExit) as caught:
            self.coli.music_audio_flags(self.args(edit_cover=True))
        self.assertIn("--edit FILE", str(caught.exception))

    def exits(self, flag, **kw):
        with self.assertRaises(SystemExit) as caught:
            self.with_clip(**kw)
        self.assertIn(flag, str(caught.exception))

    def test_a_bad_combination_exits_naming_the_flag(self):
        self.exits("--duration", duration=30)
        self.exits("--sampler heun", sampler="heun")
        self.exits("--sde", sde=True)
        self.exits("--plan", plan=True)
        with self.assertRaises(SystemExit) as caught:
            self.coli.music_audio_flags(self.args(n_avg=2))
        self.assertIn("--edit", str(caught.exception))
        with self.assertRaises(SystemExit) as caught:
            self.coli.music_audio_flags(self.args(edit="a.wav", cover="b.wav"))
        self.assertIn("separate tasks", str(caught.exception))

    def test_a_bad_value_exits_naming_its_flag(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.wav")
            Path(path).write_bytes(make_wav(1.5))
            with self.assertRaises(SystemExit) as caught:
                self.coli.music_settings(self.args(edit=path, n_avg=99), "bright ukulele", WITH)
            self.assertIn("--n-avg", str(caught.exception))
            with self.assertRaises(SystemExit) as caught:
                self.coli.music_settings(self.args(edit=path, n_min=0.8, n_max=0.2), "bright ukulele", WITH)
            self.assertIn("--n-max", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
