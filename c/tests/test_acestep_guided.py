"""ACE-Step's guided DiTs (acestep-v15-base, -sft) and the three tasks only they do (extract, lego, complete), from the request rules to
the HTTP route and `coli music`, against the stub engine.

The engine's own numbers (APG, ADG, the schedule, the null condition, what each task's context is) are held to upstream's handler by
`acestep --gen-ref` on cases captured with `tools/acestep_ref_audio.py tasks --dit acestep-v15-base`, and their mutants by
`tools/mutate_acestep_guided.py`. What is under test here is what the Python side adds: which requests are refused before the engine
hears of them, the defaults a guided DiT gets (32 steps, no DCW), what reaches the engine, and what `coli music` makes of its flags.
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
from test_acestep_gateway import GatewayHarness  # noqa: E402

import music_engine  # noqa: E402

GUIDED = {"tasks": ["text2music", "cover", "repaint", "extract", "lego", "complete"], "dits": ["turbo", "base"],
          "guided": {"available": True, "default_steps": 32, "max_steps": 200, "default_guidance": 7},
          "audio_in": {"formats": ["wav", "mp3"], "max_seconds": 600, "ref_seconds": 30}}
TURBO_ONLY = {"tasks": ["text2music", "cover", "repaint"], "audio_in": {"formats": ["wav", "mp3"], "max_seconds": 600, "ref_seconds": 30}}
B64 = lambda data: base64.b64encode(data).decode()  # noqa: E731


def request(info=GUIDED, **body):
    body.setdefault("prompt", "calm piano")
    return music_engine.music_request(body, info)


def refused(param, info=GUIDED, **body):
    body.setdefault("prompt", "calm piano")
    try:
        music_engine.music_request(body, info)
    except music_engine.MusicRequestError as error:
        assert error.param == param, (error.param, str(error))
        return str(error)
    raise AssertionError(f"not refused: {body}")


class Rules(unittest.TestCase):
    def test_a_turbo_request_is_what_it_was(self):
        r = request()
        self.assertEqual((r["task"], r["steps"], r["dcw"], r["dit"], r["guided"], r["guidance_scale"]), ("text2music", 8, True, None, False, None))

    def test_the_three_tasks_run_on_a_guided_dit_with_its_defaults(self):
        wav = B64(make_wav(2))
        for body in (dict(task="extract", src_audio=wav, track_name="drums"), dict(task="lego", src_audio=wav, track_name="bass"),
                     dict(task="complete", src_audio=wav, complete_track_classes=["drums", "bass"])):
            with self.subTest(task=body["task"]):
                r = request(**body)
                self.assertEqual((r["steps"], r["dcw"], r["guided"], r["duration"]), (32, None, True, None))     # the clip's length is the song's; DCW is left to the engine

    def test_a_complete_needs_no_clip_and_then_has_a_duration(self):
        r = request(task="complete", complete_track_classes="drums, bass", duration=12)
        self.assertEqual((r["src_audio"], r["duration"], r["complete_track_classes"]), (None, 12, ["drums", "bass"]))
        self.assertEqual(request(task="complete")["duration"], 30)
        self.assertIn("between", refused("duration", task="complete", duration=700))

    def test_an_extract_and_a_lego_need_their_clip(self):
        for task in ("extract", "lego"):
            self.assertIn("src_audio", refused("src_audio", task=task, track_name="drums"))

    def test_the_tracks_are_checked(self):
        wav = B64(make_wav(2))
        refused("track_name", task="extract", src_audio=wav, track_name="kazoo")
        refused("track_name", task="complete", track_name="drums")
        refused("track_name", task="cover", src_audio=wav, track_name="drums")
        refused("complete_track_classes", task="extract", src_audio=wav, complete_track_classes=["drums"])
        refused("complete_track_classes", task="complete", complete_track_classes=["drums", "kazoo"])
        refused("complete_track_classes", task="complete", complete_track_classes=[])
        self.assertEqual(request(task="extract", src_audio=wav, track_name=" Drums ")["track_name"], "drums")

    def test_a_lego_takes_the_repaints_span_and_nothing_else_does(self):
        wav = B64(make_wav(4))
        r = request(task="lego", src_audio=wav, track_name="bass", repainting_start=1, repainting_end=3, repaint_mode="conservative")
        self.assertEqual((r["repainting_start"], r["repainting_end"], r["repaint_mode"]), (1.0, 3.0, "conservative"))
        for task in ("extract", "complete"):
            refused("repainting_start", task=task, src_audio=wav, repainting_start=1)
        refused("repainting_end", task="lego", src_audio=wav, repainting_start=3, repainting_end=2)
        refused("audio_cover_strength", task="lego", src_audio=wav, audio_cover_strength=0.5)

    def test_the_guidance_settings(self):
        r = request(dit="base", guidance_scale=4.5, cfg_interval_start=0.2, cfg_interval_end=0.8, use_adg=True, steps=100)
        self.assertEqual((r["dit"], r["guidance_scale"], r["cfg_interval_start"], r["cfg_interval_end"], r["use_adg"], r["steps"]),
                         ("base", 4.5, 0.2, 0.8, True, 100))
        refused("steps", dit="base", steps=201)
        refused("steps", steps=21)                                          # the turbo's own cap stays
        refused("guidance_scale", dit="base", guidance_scale=101)
        refused("guidance_scale", dit="base", guidance_scale=True)
        refused("cfg_interval_start", dit="base", cfg_interval_start=1.5)
        refused("cfg_interval_end", dit="base", cfg_interval_start=0.8, cfg_interval_end=0.3)
        refused("use_adg", dit="base", use_adg="yes")

    def test_guidance_is_not_for_the_turbo_dit(self):
        for name, value in (("guidance_scale", 3), ("cfg_interval_start", 0.1), ("cfg_interval_end", 0.9), ("use_adg", True)):
            self.assertIn("base and SFT", refused(name, **{name: value}))
        self.assertIn("needs the base or SFT", refused("dit", dit="turbo", task="extract", src_audio=B64(make_wav(2))))

    def test_dit_names(self):
        refused("dit", dit="xl")
        for name in ("turbo", "base", "sft"):
            self.assertEqual(request(dit=name)["dit"], name)

    def test_a_checkpoint_without_the_base_dit_says_so(self):
        wav = B64(make_wav(2))
        self.assertIn("only the turbo DiT", refused("task", TURBO_ONLY, task="extract", src_audio=wav))
        self.assertIn("no base or SFT DiT", refused("dit", TURBO_ONLY, dit="base"))
        self.assertEqual(request(TURBO_ONLY)["steps"], 8)

    def test_an_empty_prompt_is_enough_for_the_clip_tasks(self):
        wav = B64(make_wav(2))
        for body in (dict(task="extract", src_audio=wav), dict(task="lego", src_audio=wav), dict(task="complete")):
            self.assertEqual(request(prompt="", **body)["caption"], "")


class Gateway(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = GatewayHarness(ACESTEP_STUB_GUIDED=1)

    @classmethod
    def tearDownClass(cls):
        cls.rig.close()

    def song(self, **body):
        status, _, raw = self.rig.music(**body)
        return status, json.loads(raw)

    def test_the_model_card_is_not_needed_the_ready_line_is(self):
        self.assertEqual(self.rig.engine.info["guided"]["max_steps"], 200)
        self.assertIn("lego", self.rig.engine.tasks)

    def test_a_turbo_song_is_still_a_turbo_song(self):
        status, body = self.song(prompt="calm piano", duration=2, steps=3, seed=1)
        self.assertEqual(status, 200, body)
        self.assertEqual((body["colibri"]["dit"], body["colibri"]["steps"], body["colibri"]["guidance"]), ("acestep-v15-turbo", 3, 0))

    def test_each_task_reaches_the_engine_on_the_base_dit(self):
        wav = B64(make_wav(2))
        for body in (dict(task="extract", src_audio=wav, track_name="drums"), dict(task="lego", src_audio=wav, track_name="bass"),
                     dict(task="complete", src_audio=wav, complete_track_classes=["drums", "bass"]),
                     dict(task="complete", duration=2, complete_track_classes="synth")):
            with self.subTest(task=body["task"], clip="src_audio" in body):
                status, out = self.song(prompt="", seed=3, steps=4, **body)
                self.assertEqual(status, 200, out)
                self.assertEqual((out["colibri"]["task"], out["colibri"]["dit"], out["colibri"]["steps"], out["colibri"]["guidance"]),
                                 (body["task"], "acestep-v15-base", 4, 7))

    def test_a_guided_text_to_music_song_with_its_guidance(self):
        status, out = self.song(prompt="calm piano", duration=2, dit="base", guidance_scale=3, seed=2, steps=50)
        self.assertEqual(status, 200, out)
        self.assertEqual((out["colibri"]["dit"], out["colibri"]["steps"], out["colibri"]["guidance"]), ("acestep-v15-base", 50, 3))

    def test_the_default_for_a_guided_song_is_32_steps(self):
        status, out = self.song(prompt="calm piano", duration=1, dit="base", seed=2)
        self.assertEqual((status, out["colibri"]["steps"]), (200, 32))

    def test_the_lego_span_changes_the_song(self):
        wav = B64(make_wav(4))
        first = self.song(prompt="", task="lego", src_audio=wav, track_name="bass", repainting_start=0, repainting_end=2, seed=1, steps=2)[1]
        again = self.song(prompt="", task="lego", src_audio=wav, track_name="bass", repainting_start=0, repainting_end=2, seed=1, steps=2)[1]
        other = self.song(prompt="", task="lego", src_audio=wav, track_name="bass", repainting_start=1, repainting_end=3, seed=1, steps=2)[1]
        self.assertEqual(first["data"][0]["b64_json"], again["data"][0]["b64_json"])
        self.assertNotEqual(first["data"][0]["b64_json"], other["data"][0]["b64_json"])

    def test_refusals_are_400s_that_name_the_field(self):
        wav = B64(make_wav(2))
        for body, param in ((dict(task="extract", src_audio=wav, track_name="kazoo"), "track_name"),
                            (dict(task="extract"), "src_audio"),
                            (dict(dit="base", steps=500), "steps"),
                            (dict(guidance_scale=3), "guidance_scale")):
            with self.subTest(param=param):
                status, out = self.song(prompt="x", **body)
                self.assertEqual((status, out["error"]["param"]), (400, param), out)


class OlderEngine(unittest.TestCase):
    """An engine that does not announce guided DiTs: the new tasks are refused by the gateway, in words that say what to install."""

    @classmethod
    def setUpClass(cls):
        cls.rig = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.rig.close()

    def test_the_tasks_are_refused_with_the_remedy(self):
        status, _, raw = self.rig.music(prompt="", task="extract", src_audio=B64(make_wav(2)))
        out = json.loads(raw)
        self.assertEqual((status, out["error"]["param"]), (400, "task"))
        self.assertIn("acestep-v15-base", out["error"]["message"])
        status, _, raw = self.rig.music(prompt="x", dit="base")
        self.assertEqual((status, json.loads(raw)["error"]["param"]), (400, "dit"))


class Coli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = importlib.machinery.SourceFileLoader("coli_script_guided", str(HERE.parent / "coli"))
        spec = importlib.util.spec_from_loader("coli_script_guided", loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def args(self, **kw):
        base = dict(cover=None, repaint=None, extract=None, lego=None, complete=None, ref=None, track=None, tracks=None, dit=None, guidance=None,
                    cfg_start=None, cfg_end=None, adg=False, duration=None, repaint_from=None, repaint_to=None, repaint_mode=None,
                    repaint_strength=None, cover_strength=None, cover_noise=None)
        base.update(kw)
        return type("Args", (), base)()

    def clip(self, tmp):
        path = Path(tmp) / "song.wav"
        path.write_bytes(make_wav(2))
        return str(path)

    def exits(self, **kw):
        with self.assertRaises(SystemExit) as caught:
            self.coli.music_audio_flags(self.args(**kw))
        return str(caught.exception)

    def test_the_flags_become_the_requests_own_fields(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self.clip(tmp)
            body = self.coli.music_audio_flags(self.args(extract=path, track="drums", dit="base", guidance=5.0, adg=True, cfg_start=0.2, cfg_end=0.9))
            self.assertEqual((body["task"], body["track_name"], body["dit"], body["guidance_scale"], body["use_adg"], body["cfg_interval_start"],
                              body["cfg_interval_end"]), ("extract", "drums", "base", 5.0, True, 0.2, 0.9))
            self.assertTrue(body["src_audio"])
            body = self.coli.music_audio_flags(self.args(lego=path, track="bass", repaint_from=1.0, repaint_to=2.0))
            self.assertEqual((body["task"], body["repainting_start"], body["repainting_end"]), ("lego", 1.0, 2.0))
            body = self.coli.music_audio_flags(self.args(complete=path, tracks="drums,bass"))
            self.assertEqual((body["task"], body["complete_track_classes"]), ("complete", "drums,bass"))
            body = self.coli.music_audio_flags(self.args(complete="", tracks="synth"))      # no clip: made from the description
            self.assertEqual((body["task"], "src_audio" in body), ("complete", False))

    def test_the_tasks_exclude_each_other_and_each_flag_has_its_task(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self.clip(tmp)
            self.assertIn("separate tasks", self.exits(extract=path, lego=path))
            self.assertIn("separate tasks", self.exits(cover=path, complete=""))
            self.assertIn("--track is for", self.exits(cover=path, track="drums"))
            self.assertIn("--tracks is for", self.exits(extract=path, tracks="drums"))
            self.assertIn("--from is for", self.exits(extract=path, repaint_from=1.0))
            self.assertIn("--duration does not apply", self.exits(lego=path, duration=10.0))


if __name__ == "__main__":
    unittest.main()
