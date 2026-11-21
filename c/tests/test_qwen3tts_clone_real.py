"""Voice cloning in c/qwen3tts on the real Base checkpoint, against the official package.

    QWEN3TTS_MODEL_BASE=/path/to/Qwen3-TTS-12Hz-0.6B-Base (with tokenizer.json) QWEN3TTS_CLIP=/path/to/jfk24.wav (the 11 s JFK clip, 24 kHz mono,
    PCM16) [QWEN3TTS_CLONE_REF=/path/to/clone06] python -m unittest tests.test_qwen3tts_clone_real

Skipped without the built engine, the model folder and the clip. tools/qwen3_tts_0.6b_base_reference.json holds the package's output for the
clip (fp32, greedy: the reference's codes, the frames and samples of each mode); with QWEN3TTS_CLONE_REF (the folder tools/qwen3_tts_clone_ref.py
wrote) the engine's stages are held to the package's tensors too: the log-mel features, the speaker embedding, the codec encoder's three
stages, the reference's codes, the prompt, the generated codes and the waveform. With COLI_VULKAN=1 and a VK=1 build the card is held to the same.
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
MODEL = os.environ.get("QWEN3TTS_MODEL_BASE")
CLIP = os.environ.get("QWEN3TTS_CLIP")
REF_DIR = os.environ.get("QWEN3TTS_CLONE_REF")
REFERENCE = os.path.join(C_DIR, "tools", "qwen3_tts_0.6b_base_reference.json")
READY = bool(EXE and MODEL and os.path.isdir(MODEL) and os.path.isfile(os.path.join(MODEL, "tokenizer.json")) and CLIP and os.path.isfile(CLIP))


def read_wav(path):
    with wave.open(path, "rb") as clip:
        frames = clip.readframes(clip.getnframes())
        return struct.unpack(f"<{len(frames) // 2}h", frames)


@unittest.skipUnless(READY, "needs the built qwen3tts, QWEN3TTS_MODEL_BASE (the 0.6B Base with tokenizer.json) and QWEN3TTS_CLIP (the 24 kHz JFK clip)")
class CloneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REFERENCE, encoding="utf-8") as handle:
            cls.ref = json.load(handle)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.process = subprocess.Popen([EXE, "--model", MODEL, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding="utf-8", bufsize=1)
        cls.ready = json.loads(cls.process.stdout.readline().partition(" ")[2])
        cls.seq = 0

    @classmethod
    def tearDownClass(cls):
        cls.process.stdin.close()
        cls.process.wait(timeout=120)
        cls.process.stderr.read()
        cls.process.stderr.close()
        cls.process.stdout.close()
        cls.tmp.cleanup()

    def speak(self, **fields):
        type(self).seq += 1
        out = os.path.join(self.tmp.name, f"s{self.seq}.wav").replace("\\", "/")
        request = {"id": f"t{self.seq}", "out": out, "text": self.ref["text"], "language": self.ref["language"], "greedy": True}
        request.update(fields)
        self.process.stdin.write("SPEAK " + json.dumps(request) + "\n")
        self.process.stdin.flush()
        keyword, _, body = self.process.stdout.readline().partition(" ")
        return keyword, json.loads(body), out

    def cli(self, *extra):
        args = [EXE, "--model", MODEL, "--text", self.ref["text"], "--language", self.ref["language"], "--clone-wav", CLIP, "--greedy"]
        run = subprocess.run(args + list(extra), capture_output=True, text=True, encoding="utf-8", timeout=900)
        self.assertEqual(run.returncode, 0, run.stderr[-500:])
        return run.stdout

    def test_ready_says_the_model_clones_and_has_no_speakers(self):
        self.assertEqual((self.ready["kind"], self.ready["clone"], self.ready["instruct"], self.ready["speakers"]), ("base", True, False, []))

    def test_a_request_without_a_reference_is_an_error_and_the_engine_goes_on(self):
        keyword, body, _ = self.speak()
        self.assertEqual(keyword, "ERROR")
        self.assertIn("reference audio", body["message"])
        self.assertEqual(self.speak(ref_wav=CLIP, x_vector_only=True, max_frames=3)[0], "AUDIO")

    def test_a_reference_that_is_not_a_24khz_wav_is_an_error(self):
        path = os.path.join(self.tmp.name, "bad.wav")
        with wave.open(path, "wb") as clip:
            clip.setnchannels(1)
            clip.setsampwidth(2)
            clip.setframerate(16000)
            clip.writeframes(b"\0\0" * 16000)
        keyword, body, _ = self.speak(ref_wav=path.replace("\\", "/"), x_vector_only=True)
        self.assertEqual(keyword, "ERROR")
        self.assertIn("24 kHz", body["message"])
        keyword, body, _ = self.speak(ref_wav=os.path.join(self.tmp.name, "missing.wav").replace("\\", "/"), x_vector_only=True)
        self.assertEqual(keyword, "ERROR")
        self.assertIn("cannot open", body["message"])

    def test_x_vector_only_speaks_as_many_frames_and_samples_as_the_package(self):
        keyword, body, out = self.speak(ref_wav=CLIP, x_vector_only=True, max_frames=90)
        self.assertEqual(keyword, "AUDIO", body)
        self.assertEqual((body["frames"], body["samples"]), (self.ref["xvec"]["frames"], self.ref["xvec"]["samples"]))
        self.assertGreater(max(abs(s) for s in read_wav(out)), 3000)

    def test_in_context_cloning_continues_after_the_reference_and_cuts_it_off(self):
        keyword, body, out = self.speak(ref_wav=CLIP, ref_text=self.ref["ref_text"], max_frames=self.ref["icl"]["frames"])
        self.assertEqual(keyword, "AUDIO", body)
        self.assertEqual((body["frames"], body["samples"]), (self.ref["icl"]["frames"], self.ref["icl"]["samples"]))     # the reference's share is cut off
        self.assertGreater(max(abs(s) for s in read_wav(out)), 1000)

    def test_the_same_clip_twice_gives_the_same_speech_the_second_time_from_the_cache(self):
        _, first, out1 = self.speak(ref_wav=CLIP, ref_text=self.ref["ref_text"], max_frames=20)
        _, second, out2 = self.speak(ref_wav=CLIP, ref_text=self.ref["ref_text"], max_frames=20)
        self.assertEqual(read_wav(out1), read_wav(out2))
        self.assertLessEqual(second["seconds"], first["seconds"] + 0.05)

    def test_without_the_reference_text_the_context_mode_falls_back_to_the_speaker_embedding(self):
        _, with_text, out1 = self.speak(ref_wav=CLIP, ref_text=self.ref["ref_text"], max_frames=20)
        _, without, out2 = self.speak(ref_wav=CLIP, max_frames=20)
        self.assertNotEqual(read_wav(out1), read_wav(out2))

    @unittest.skipUnless(os.environ.get("COLI_VULKAN") == "1", "the card runs only with COLI_VULKAN=1 and a VK=1 build")
    def test_on_the_card_ready_says_so(self):
        self.assertEqual(self.ready["device"], "vulkan")


@unittest.skipUnless(READY and REF_DIR and os.path.isdir(REF_DIR or ""), "needs QWEN3TTS_CLONE_REF (the folder tools/qwen3_tts_clone_ref.py wrote)")
class StagesAgainstThePackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REFERENCE, encoding="utf-8") as handle:
            cls.ref = json.load(handle)

    def cli(self, *extra):
        args = [EXE, "--model", MODEL, "--text", self.ref["text"], "--language", self.ref["language"], "--clone-wav", CLIP, "--greedy", "--check", REF_DIR]
        run = subprocess.run(args + list(extra), capture_output=True, text=True, encoding="utf-8", timeout=900)
        self.assertEqual(run.returncode, 0, run.stderr[-500:])
        return run.stdout

    def worst(self, out, label):
        match = re.search(re.escape(label) + r"\s+n=\d+\s+max abs diff ([0-9.e+-]+)", out)
        self.assertIsNotNone(match, out)
        return float(match.group(1))

    def test_x_vector_mode_matches_stage_by_stage(self):
        out = self.cli("--xvector", "--max-frames", "90")
        self.assertLess(self.worst(out, "mel"), 5e-3)
        self.assertLess(self.worst(out, "speaker embedding"), 1e-5)
        self.assertRegex(out, r"input ids: identical")
        self.assertLess(self.worst(out, "prefill"), 1e-5)
        self.assertIn(f"codes: identical ({self.ref['xvec']['frames']} frames here, {self.ref['xvec']['frames']} in the reference)", out)
        self.assertLess(self.worst(out, "waveform"), 1e-4)

    def test_in_context_mode_matches_stage_by_stage(self):
        out = self.cli("--clone-text", self.ref["ref_text"], "--max-frames", str(self.ref["icl"]["frames"]))
        self.assertLess(self.worst(out, "codec conv stack"), 5e-3)
        self.assertLess(self.worst(out, "codec transformer"), 5e-3)
        self.assertLess(self.worst(out, "codec embeddings"), 5e-3)
        self.assertRegex(out, rf"reference codes: {self.ref['ref_frames']} frames in the reference, {self.ref['ref_frames']} here; 0 of {self.ref['ref_frames'] * 16} codes differ")
        self.assertIn("reference text ids: identical", out)
        self.assertLess(self.worst(out, "prefill"), 1e-5)
        self.assertIn(f"codes: identical ({self.ref['icl']['frames']} frames here, {self.ref['icl']['frames']} in the reference)", out)
        self.assertLess(self.worst(out, "waveform"), 1e-4)

    def test_every_code_the_package_wrote_is_the_engines_own_argmax(self):
        os.environ["QT_TEACH"] = "1"
        try:
            out = self.cli("--clone-text", self.ref["ref_text"], "--max-frames", str(self.ref["icl"]["frames"]))
        finally:
            os.environ.pop("QT_TEACH", None)
        match = re.search(r"teacher: (\d+) choices over (\d+) frames; (\d+) of the reference's codes were not my argmax", out)
        self.assertIsNotNone(match, out)
        self.assertEqual((int(match.group(1)), int(match.group(2)), int(match.group(3))), (self.ref["icl"]["frames"] * 16, self.ref["icl"]["frames"], 0))


if __name__ == "__main__":
    unittest.main()
