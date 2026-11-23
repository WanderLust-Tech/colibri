"""ACE-Step's sampler options (Heun, the SDE step, the velocity clamp and smoothing: upstream's `sampler_mode`, `infer_method`,
`velocity_norm_threshold` and `velocity_ema_factor`), from the request rules to the HTTP route and `coli music`, against the stub engine.

The engine's own numbers (the steps themselves) are held to upstream's handler by `acestep --gen-ref` on cases captured with
`tools/acestep_ref_audio.py tasks` (the `s_*` cases for the turbo DiT, the `b_*` sampler cases for the base DiT: docs/acestep.md). What is under test
here is what the Python side adds: which requests are refused before the engine hears of them (an engine that does not announce the options would
ignore them, so those are refused too), what reaches the engine, and what `coli music` makes of its flags.
"""
import importlib.machinery
import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from test_acestep_gateway import GatewayHarness  # noqa: E402

import music_engine  # noqa: E402

WITH = {"tasks": ["text2music"], "sampler": True}
WITHOUT = {"tasks": ["text2music"]}


def request(info=WITH, **body):
    body.setdefault("prompt", "calm piano")
    return music_engine.music_request(body, info)


def refused(param, info=WITH, **body):
    body.setdefault("prompt", "calm piano")
    try:
        music_engine.music_request(body, info)
    except music_engine.MusicRequestError as error:
        assert error.param == param, (error.param, str(error))
        return str(error)
    raise AssertionError(f"not refused: {body}")


class Rules(unittest.TestCase):
    def test_nothing_asked_is_nothing_sent(self):
        r = request()
        self.assertEqual((r["sampler_mode"], r["infer_method"], r["velocity_norm_threshold"], r["velocity_ema_factor"]), (None, None, None, None))

    def test_the_options_are_taken_as_given(self):
        r = request(sampler_mode="heun", velocity_norm_threshold=0.8, velocity_ema_factor=0.3)
        self.assertEqual((r["sampler_mode"], r["infer_method"], r["velocity_norm_threshold"], r["velocity_ema_factor"]), ("heun", None, 0.8, 0.3))
        r = request(infer_method="sde", sampler_mode="euler")
        self.assertEqual((r["sampler_mode"], r["infer_method"]), ("euler", "sde"))
        r = request(velocity_norm_threshold=0, velocity_ema_factor=1)               # the edges of the ranges; 0 is "off"
        self.assertEqual((r["velocity_norm_threshold"], r["velocity_ema_factor"]), (0.0, 1.0))

    def test_every_dit_takes_them(self):
        guided = {**WITH, "tasks": ["text2music", "extract", "lego", "complete"], "dits": ["turbo", "base"],
                  "guided": {"available": True, "default_steps": 32, "max_steps": 200, "default_guidance": 7}}
        self.assertEqual(request(guided, dit="base", sampler_mode="heun")["sampler_mode"], "heun")
        self.assertEqual(request(guided, dit="turbo", infer_method="sde")["infer_method"], "sde")

    def test_bad_values_are_refused_by_name(self):
        refused("sampler_mode", sampler_mode="rk4")
        refused("sampler_mode", sampler_mode=True)
        refused("infer_method", infer_method="ODE")
        refused("infer_method", infer_method=1)
        for value in (-0.1, 100.5, "1", True, float("nan")):
            refused("velocity_norm_threshold", velocity_norm_threshold=value)
        for value in (-0.1, 1.5, "0.3", True):
            refused("velocity_ema_factor", velocity_ema_factor=value)

    def test_heun_and_the_sde_are_not_both(self):
        self.assertIn("pick one", refused("sampler_mode", sampler_mode="heun", infer_method="sde"))
        request(sampler_mode="euler", infer_method="sde")
        request(sampler_mode="heun", infer_method="ode")

    def test_an_engine_that_does_not_announce_them_would_ignore_them(self):
        for body in (dict(sampler_mode="heun"), dict(infer_method="sde"), dict(velocity_norm_threshold=1), dict(velocity_ema_factor=0.2)):
            name = next(iter(body))
            self.assertIn("update the engine", refused(name, WITHOUT, **body))
        self.assertEqual(request(WITHOUT)["sampler_mode"], None)                    # not asking is fine


class Gateway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = GatewayHarness(ACESTEP_STUB_SAMPLER=1)

    @classmethod
    def tearDownClass(cls):
        cls.rig.close()

    def song(self, **body):
        status, _, raw = self.rig.music(**body)
        return status, json.loads(raw)

    def test_a_plain_song_says_euler(self):
        status, out = self.song(prompt="calm piano", duration=2, steps=3, seed=1)
        self.assertEqual(status, 200, out)
        self.assertEqual(out["colibri"]["sampler"], "euler")

    def test_each_option_reaches_the_engine(self):
        for body, said in ((dict(sampler_mode="heun"), "heun"), (dict(infer_method="sde"), "sde"), (dict(sampler_mode="euler", infer_method="ode"), "euler"),
                           (dict(velocity_norm_threshold=0.8, velocity_ema_factor=0.3), "euler")):
            with self.subTest(**body):
                status, out = self.song(prompt="calm piano", duration=2, steps=3, seed=2, **body)
                self.assertEqual(status, 200, out)
                self.assertEqual(out["colibri"]["sampler"], said)

    def test_a_bad_option_is_a_400_naming_it(self):
        status, out = self.song(prompt="calm piano", duration=2, sampler_mode="rk4")
        self.assertEqual(status, 400)
        self.assertEqual(out["error"]["param"], "sampler_mode")
        status, out = self.song(prompt="calm piano", duration=2, sampler_mode="heun", infer_method="sde")
        self.assertEqual(status, 400)

    def test_the_form_fields_of_a_multipart_request_are_read_as_numbers(self):
        import openai_server
        for name in ("velocity_norm_threshold", "velocity_ema_factor"):
            self.assertIn(name, openai_server._MUSIC_FLOAT_FIELDS)


class Plain(unittest.TestCase):
    """An engine without the announcement: the gateway refuses the options in front of it."""

    @classmethod
    def setUpClass(cls):
        cls.rig = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.rig.close()

    def test_refused_with_the_reason(self):
        status, _, raw = self.rig.music(prompt="calm piano", duration=2, sampler_mode="heun")
        self.assertEqual(status, 400)
        self.assertIn("update the engine", json.loads(raw)["error"]["message"])


class Cli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = importlib.machinery.SourceFileLoader("coli_script_sampler", str(HERE.parent / "coli"))
        spec = importlib.util.spec_from_loader("coli_script_sampler", loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def args(self, **kw):
        base = dict(cover=None, repaint=None, extract=None, lego=None, complete=None, ref=None, track=None, tracks=None, dit=None, guidance=None,
                    cfg_start=None, cfg_end=None, adg=False, duration=None, repaint_from=None, repaint_to=None, repaint_mode=None,
                    repaint_strength=None, cover_strength=None, cover_noise=None, sampler=None, sde=False, velocity_norm=None, velocity_ema=None)
        base.update(kw)
        return type("Args", (), base)()

    def test_the_flags_become_the_requests_own_fields(self):
        body = self.coli.music_audio_flags(self.args(sampler="heun", velocity_norm=0.8, velocity_ema=0.3))
        self.assertEqual((body["sampler_mode"], body["velocity_norm_threshold"], body["velocity_ema_factor"]), ("heun", 0.8, 0.3))
        self.assertNotIn("infer_method", body)
        self.assertEqual(self.coli.music_audio_flags(self.args(sde=True))["infer_method"], "sde")
        none = self.coli.music_audio_flags(self.args())
        for field in ("sampler_mode", "infer_method", "velocity_norm_threshold", "velocity_ema_factor"):
            self.assertNotIn(field, none)

    def test_a_bad_combination_exits_naming_the_flag(self):
        with self.assertRaises(SystemExit) as caught:
            self.coli.music_settings(self.args(sampler="heun", sde=True), "calm piano", WITH)
        self.assertIn("--sampler", str(caught.exception))
        with self.assertRaises(SystemExit) as caught:
            self.coli.music_settings(self.args(velocity_ema=2.0), "calm piano", WITH)
        self.assertIn("--velocity-ema", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
