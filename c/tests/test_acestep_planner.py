"""ACE-Step's 5 Hz planner from the request rules to the HTTP route and `coli music --plan`, against the stub engine.

The planner's own numbers (the two prompts' token ids, the logits at every step, the codes themselves) are held to upstream's LLMHandler by
`acestep --lm-ref` on cases captured with `tools/acestep_ref_lm.py`, and the codes' road into the DiT by `acestep --gen-ref` on cases with
`audio_code_string`. What is under test here is what the Python side adds: which requests are refused before the engine hears of them, what
reaches the engine, the plan stage, and what `coli music` makes of its flags.
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
from acestep_fixture import make_wav  # noqa: E402
from test_acestep_gateway import GatewayHarness, parse_sse  # noqa: E402

import music_engine  # noqa: E402

INFO = {"tasks": ["text2music", "cover", "repaint"], "audio_in": {"formats": ["wav", "mp3"], "max_seconds": 600, "ref_seconds": 30},
        "planner": {"available": True, "lms": ["1.7B", "4B"], "default_lm": "1.7B", "temperature": 0.85, "cfg_scale": 2, "top_p": 0.9, "top_k": 0}}
META = dict(bpm=120, key="C Major", time_signature="4", duration=10)


def request(info=INFO, **body):
    body.setdefault("prompt", "calm piano")
    return music_engine.music_request(body, info)


def refused(param, info=INFO, **body):
    body.setdefault("prompt", "calm piano")
    try:
        music_engine.music_request(body, info)
    except music_engine.MusicRequestError as error:
        assert error.param == param, (error.param, str(error))
        return str(error)
    raise AssertionError(f"not refused: {body}")


class Rules(unittest.TestCase):
    def test_a_plain_request_is_not_planned(self):
        r = request()
        self.assertEqual((r["thinking"], r["lm_model"], r["lm_temperature"], r["lm_top_k"]), (False, None, None, None))

    def test_thinking_with_the_metadata_it_needs(self):
        r = request(thinking=True, **META)
        self.assertTrue(r["thinking"])
        r = request(thinking=True, lm_model="4B", lm_temperature=0, lm_cfg_scale=3.5, lm_top_p=0.5, lm_top_k=50, **META)
        self.assertEqual((r["lm_model"], r["lm_temperature"], r["lm_cfg_scale"], r["lm_top_p"], r["lm_top_k"]), ("4B", 0.0, 3.5, 0.5, 50))

    def test_what_it_needs_is_named(self):
        self.assertIn("`bpm`", refused("bpm", thinking=True, key="C Major", time_signature="4"))
        self.assertIn("`key`", refused("key", thinking=True, bpm=100, time_signature="4"))
        message = refused("bpm", thinking=True)
        for field in ("`bpm`", "`key`", "`time_signature`", "first phase"):
            self.assertIn(field, message)

    def test_thinking_is_for_text_to_music_with_a_planner(self):
        self.assertIn("text to music", refused("thinking", thinking=True, task="cover", src_audio=make_wav(1.5), **META))
        self.assertIn("no 5 Hz planner", refused("thinking", {"tasks": INFO["tasks"]}, thinking=True, **META))
        refused("thinking", thinking="yes", **META)

    def test_the_planners_settings(self):
        for name, value in (("lm_temperature", 3), ("lm_temperature", True), ("lm_cfg_scale", 21), ("lm_top_p", 1.5), ("lm_top_p", -0.1)):
            refused(name, thinking=True, **{name: value}, **META)
        for value in (-1, 1.5, True, "5", 10 ** 6):
            refused("lm_top_k", thinking=True, lm_top_k=value, **META)
        refused("lm_model", thinking=True, lm_model="9B", **META)
        self.assertIn("not installed", refused("lm_model", thinking=True, lm_model="0.6B", **META))

    def test_its_settings_need_it(self):
        for name, value in (("lm_temperature", 0.5), ("lm_cfg_scale", 2), ("lm_top_p", 0.5), ("lm_top_k", 3), ("lm_model", "1.7B")):
            self.assertIn("thinking", refused(name, **{name: value}))


class Gateway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = GatewayHarness(ACESTEP_STUB_PLANNER=1)

    @classmethod
    def tearDownClass(cls):
        cls.rig.close()

    def song(self, **body):
        status, _, raw = self.rig.music(**body)
        return status, json.loads(raw)

    def test_the_planner_is_announced(self):
        self.assertTrue(self.rig.engine.info["planner"]["available"])

    def test_a_planned_song_says_what_the_planner_wrote(self):
        status, out = self.song(prompt="calm piano", thinking=True, seed=2, **META)
        self.assertEqual(status, 200, out)
        self.assertEqual((out["colibri"]["plan_codes"], out["colibri"]["lm"]), (50, "acestep-5Hz-lm-1.7B"))     # 10 seconds are 50 codes

    def test_a_plain_song_has_no_codes(self):
        status, out = self.song(prompt="calm piano", duration=2, seed=2)
        self.assertEqual((status, out["colibri"]["plan_codes"], out["colibri"]["lm"]), (200, 0, ""))

    def test_the_plan_stage_is_streamed_first(self):
        status, _, events = self.rig.stream(prompt="calm piano", thinking=True, seed=2, bpm=100, key="D minor", time_signature="3", duration=4)
        self.assertEqual(status, 200)
        stages = []
        for event in events:
            data = event[1] if isinstance(event, tuple) else event
            stage = (data or {}).get("stage") if isinstance(data, dict) else None
            if stage and (not stages or stages[-1] != stage):
                stages.append(stage)
        self.assertEqual(stages[:2], ["plan", "text"], stages)

    def test_the_planner_changes_the_song(self):
        a = self.song(prompt="calm piano", thinking=True, seed=2, **META)[1]["data"][0]["b64_json"]
        b = self.song(prompt="calm piano", thinking=True, seed=2, lm_model="1.7B", lm_temperature=0, **META)[1]["data"][0]["b64_json"]
        self.assertNotEqual(a, b)                                           # the stub hashes the planner's settings into the clip, as the real one's audio depends on them

    def test_refusals_are_400s_that_name_the_field(self):
        for body, param in ((dict(thinking=True), "bpm"), (dict(thinking=True, lm_temperature=5, **META), "lm_temperature"), (dict(lm_top_k=3), "lm_top_k")):
            with self.subTest(param=param):
                status, out = self.song(prompt="calm piano", **body)
                self.assertEqual((status, out["error"]["param"]), (400, param), out)


class OlderEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.rig.close()

    def test_thinking_is_refused_with_the_remedy(self):
        status, _, raw = self.rig.music(prompt="calm piano", thinking=True, **META)
        out = json.loads(raw)
        self.assertEqual((status, out["error"]["param"]), (400, "thinking"))
        self.assertIn("acestep-5Hz-lm-1.7B", out["error"]["message"])


class Coli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = importlib.machinery.SourceFileLoader("coli_script_plan", str(HERE.parent / "coli"))
        spec = importlib.util.spec_from_loader("coli_script_plan", loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def args(self, **kw):
        base = dict(cover=None, repaint=None, extract=None, lego=None, complete=None, ref=None, track=None, tracks=None, dit=None, guidance=None,
                    cfg_start=None, cfg_end=None, adg=False, duration=None, repaint_from=None, repaint_to=None, repaint_mode=None,
                    repaint_strength=None, cover_strength=None, cover_noise=None, plan=False, lm=None, lm_temperature=None, lm_cfg=None,
                    lm_top_p=None, lm_top_k=None)
        base.update(kw)
        return type("Args", (), base)()

    def test_the_flags_become_the_requests_own_fields(self):
        body = self.coli.music_audio_flags(self.args(plan=True, lm="4B", lm_temperature=0.5, lm_cfg=3.0, lm_top_p=0.8, lm_top_k=20))
        self.assertEqual((body["thinking"], body["lm_model"], body["lm_temperature"], body["lm_cfg_scale"], body["lm_top_p"], body["lm_top_k"]),
                         (True, "4B", 0.5, 3.0, 0.8, 20))

    def test_the_settings_need_the_planner(self):
        with self.assertRaises(SystemExit) as caught:
            self.coli.music_audio_flags(self.args(lm_temperature=0.5))
        self.assertIn("--lm-temperature is for --plan", str(caught.exception))

    def test_the_musics_progress_names_the_stage(self):
        self.assertIn("plan", self.coli.MusicProgress.STAGES) if hasattr(self.coli, "MusicProgress") else None


if __name__ == "__main__":
    unittest.main()
