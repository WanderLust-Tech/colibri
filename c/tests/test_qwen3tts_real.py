"""The real engine (c/qwen3tts on a Qwen3-TTS CustomVoice checkpoint) against the answer the official package gives, and its protocol.

    QWEN3TTS_MODEL=/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice python -m unittest tests.test_qwen3tts_real

Needs the built engine (make qwen3tts) and the 0.6B CustomVoice checkpoint with its speech_tokenizer/ folder and a tokenizer.json
(coli model install qwen3-tts-0.6b-customvoice writes it; or python tools/qwen3_tokenizer.py DIR); skipped without them. The reference is
tools/qwen3_tts_0.6b_reference.json, made by tools/qwen3_tts_ref.py from the official qwen-tts package in fp32 with greedy decoding,
so these tests need neither torch nor transformers: the prompt's token ids and the 63 frames of codes it says.

The engine is checked three ways. Teacher forced (the reference's codes drive the generation) it must find each of them to be its own
argmax: that is a test of the whole stack that a floating-point tie cannot fail. Free-running and greedy it must write the same codes (the
reference has no ties at this text; the test says so if one ever shows). And through its serve protocol: what it answers, and that a
seed repeats a speech.
"""
import json
import os
import re
import struct
import subprocess
import tempfile
import unittest
import wave

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXE = next((p for p in (os.path.join(C_DIR, "qwen3tts.exe"), os.path.join(C_DIR, "qwen3tts")) if os.path.isfile(p)), None)
MODEL = os.environ.get("QWEN3TTS_MODEL")
REFERENCE = os.path.join(C_DIR, "tools", "qwen3_tts_0.6b_reference.json")
READY = bool(EXE and MODEL and os.path.isdir(MODEL) and os.path.isfile(os.path.join(MODEL, "tokenizer.json"))
             and os.path.isdir(os.path.join(MODEL, "speech_tokenizer")))


def read_wav(path):
    with wave.open(path, "rb") as clip:
        frames = clip.readframes(clip.getnframes())
        return clip.getframerate(), clip.getnchannels(), struct.unpack(f"<{len(frames) // 2}h", frames)


class Engine:
    def __init__(self):
        self.process = subprocess.Popen([EXE, "--model", MODEL, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1)
        self.ready = self.process.stdout.readline()
        self.seq = 0

    def send(self, verb, **fields):
        self.seq += 1
        self.process.stdin.write(f"{verb} " + json.dumps({"id": f"t{self.seq}", **fields}) + "\n")
        self.process.stdin.flush()
        keyword, _, body = self.process.stdout.readline().partition(" ")
        return keyword, json.loads(body)

    def speak(self, out, **fields):
        return self.send("SPEAK", out=out.replace("\\", "/"), **fields)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=120)
        self.stderr = self.process.stderr.read()
        self.process.stderr.close()
        self.process.stdout.close()


@unittest.skipUnless(READY, "needs the built qwen3tts and QWEN3TTS_MODEL (the 0.6B CustomVoice with tokenizer.json and speech_tokenizer/)")
class RealEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REFERENCE, encoding="utf-8") as handle:
            cls.ref = json.load(handle)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.refdir = os.path.join(cls.tmp.name, "ref")
        os.makedirs(cls.refdir)
        with open(os.path.join(cls.refdir, "codes.i32"), "wb") as handle:
            for frame in cls.ref["codes"]:
                handle.write(struct.pack("<16i", *frame))
        with open(os.path.join(cls.refdir, "input_ids.i32"), "wb") as handle:
            handle.write(struct.pack(f"<{len(cls.ref['input_ids'])}i", *cls.ref["input_ids"]))
        cls.engine = Engine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def cli(self, *extra):
        run = subprocess.run([EXE, "--model", MODEL, "--text", self.ref["text"], "--speaker", self.ref["speaker"], "--language", self.ref["language"],
                              "--ref", self.refdir, *extra], capture_output=True, text=True, encoding="utf-8", timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-500:])
        return run.stdout

    def test_ready_says_what_the_engine_is(self):
        keyword, _, body = self.engine.ready.partition(" ")
        info = json.loads(body)
        self.assertEqual(keyword, "READY")
        self.assertEqual((info["task"], info["kind"], info["sample_rate"], info["layers"]), ("speech", "custom_voice", 24000, 28))
        self.assertEqual(info["speakers"], ["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"])
        self.assertIn("english", info["languages"])
        self.assertIn("beijing_dialect", info["languages"])

    def test_the_prompt_is_the_references_prompt(self):
        out = self.cli("--teacher", "--max-frames", "1")
        self.assertRegex(out, r"input ids: identical \(19 here, 19 in the reference\)")

    def test_every_code_the_reference_says_is_the_engines_own_argmax(self):
        out = self.cli("--teacher")
        match = re.search(r"teacher: (\d+) choices over (\d+) frames; (\d+) of the reference's codes were not my argmax", out)
        self.assertIsNotNone(match, out)
        self.assertEqual((int(match.group(1)), int(match.group(2)), int(match.group(3))), (63 * 16, 63, 0), out)
        self.assertIn("ends with the end token: yes", out)

    def test_free_running_and_greedy_it_writes_the_references_codes(self):
        out = self.cli("--greedy")
        self.assertIn("codes: identical (63 frames here, 63 in the reference)", out)

    def test_a_speech_has_the_references_length_and_is_a_wav(self):
        keyword, body = self.engine.speak(self.path("a.wav"), text=self.ref["text"], speaker="Ryan", language="English", greedy=True)
        self.assertEqual(keyword, "AUDIO", body)
        self.assertEqual((body["frames"], body["samples"], body["sample_rate"], body["truncated"]), (63, self.ref["samples"], 24000, False))
        rate, channels, samples = read_wav(self.path("a.wav"))
        self.assertEqual((rate, channels, len(samples)), (24000, 1, self.ref["samples"]))
        self.assertGreater(max(abs(s) for s in samples), 3000)                  # it is speech, not silence
        for i, want in enumerate(self.ref["frame_rms"]):                        # the codec's output follows the reference's loudness frame by frame
            chunk = samples[i * 1920:(i + 1) * 1920]
            got = (sum(s * s for s in chunk) / len(chunk)) ** 0.5 / 32768
            self.assertAlmostEqual(got, want, delta=1e-3, msg=f"frame {i}")
        self.assertAlmostEqual(body["audio_seconds"], 5.04, places=2)

    def test_a_seed_repeats_a_speech_and_another_seed_does_not(self):
        text = "The quick brown fox jumps over the lazy dog."
        for name, seed in (("s1", 7), ("s2", 7), ("s3", 8)):
            keyword, body = self.engine.speak(self.path(name + ".wav"), text=text, speaker="vivian", language="english", seed=seed, max_frames=40)
            self.assertEqual(keyword, "AUDIO", body)
        with open(self.path("s1.wav"), "rb") as a, open(self.path("s2.wav"), "rb") as b, open(self.path("s3.wav"), "rb") as c:
            first, again, other = a.read(), b.read(), c.read()
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_the_frame_limit_cuts_the_speech_and_says_so(self):
        keyword, body = self.engine.speak(self.path("cut.wav"), text=self.ref["text"], speaker="ryan", greedy=True, max_frames=5)
        self.assertEqual(keyword, "AUDIO", body)
        self.assertEqual((body["frames"], body["samples"], body["truncated"]), (5, 5 * 1920, True))

    def test_different_speakers_say_it_differently(self):
        for name, speaker in (("ry", "ryan"), ("vi", "vivian")):
            keyword, body = self.engine.speak(self.path(name + ".wav"), text="Good morning.", speaker=speaker, language="english", greedy=True, max_frames=20)
            self.assertEqual(keyword, "AUDIO", body)
        self.assertNotEqual(read_wav(self.path("ry.wav"))[2], read_wav(self.path("vi.wav"))[2])

    def test_a_dialect_speaker_is_taken_in_its_dialect(self):
        keyword, body = self.engine.speak(self.path("dy.wav"), text="你好，今天天气不错。", speaker="dylan", greedy=True, max_frames=20)
        self.assertEqual(keyword, "AUDIO", body)

    def test_what_it_refuses_it_refuses_in_words_and_goes_on(self):
        cases = [(dict(text="", speaker="ryan"), "non-empty"), (dict(text="hi", speaker="nobody"), "nobody"),
                 (dict(text="hi", speaker="ryan", language="klingon"), "klingon"), (dict(speaker="ryan"), "non-empty")]
        for fields, fragment in cases:
            with self.subTest(fields=fields):
                keyword, body = self.engine.speak(self.path("no.wav"), **fields)
                self.assertEqual(keyword, "ERROR", body)
                self.assertIn(fragment, body["message"])
        keyword, body = self.engine.send("SPEAK", text="hi")
        self.assertEqual((keyword, "out" in body["message"]), ("ERROR", True))
        keyword, body = self.engine.speak(self.path("ok.wav"), text="Fine.", speaker="ryan", greedy=True, max_frames=10)
        self.assertEqual(keyword, "AUDIO", body)


if __name__ == "__main__":
    unittest.main()
