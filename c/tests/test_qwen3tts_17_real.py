"""The 1.7B engines (c/qwen3tts on the 1.7B CustomVoice and the 1.7B VoiceDesign) against the official package, and what an instruction does.

    QWEN3TTS_MODEL_17=/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice QWEN3TTS_MODEL_DESIGN=/path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign \\
        python -m unittest tests.test_qwen3tts_17_real

Each model is skipped without its folder (with tokenizer.json and speech_tokenizer/) and the built engine. The reference is
tools/qwen3_tts_1.7b_reference.json, made by tools/qwen3_tts_ref.py (--instruct) from the official qwen-tts package in fp32, greedy: the
prompt's token ids, the codes and the loudness of each frame. As tests/test_qwen3tts_real.py does for the 0.6B: teacher forced the engine must
find every reference code to be its own argmax, free-running greedy it must write the same codes, and the instruction (the style on the
CustomVoice, the whole voice on VoiceDesign) must be what the speech follows. With COLI_VULKAN=1 and a VK=1 build the card is held to the same.
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
REFERENCE = os.path.join(C_DIR, "tools", "qwen3_tts_1.7b_reference.json")


def usable(path):
    return bool(EXE and path and os.path.isdir(path) and os.path.isfile(os.path.join(path, "tokenizer.json"))
                and os.path.isdir(os.path.join(path, "speech_tokenizer")))


def read_wav(path):
    with wave.open(path, "rb") as clip:
        frames = clip.readframes(clip.getnframes())
        return struct.unpack(f"<{len(frames) // 2}h", frames)


class Engine:
    def __init__(self, model):
        self.process = subprocess.Popen([EXE, "--model", model, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1)
        self.ready = json.loads(self.process.stdout.readline().partition(" ")[2])
        self.seq = 0

    def speak(self, out, **fields):
        self.seq += 1
        self.process.stdin.write("SPEAK " + json.dumps({"id": f"t{self.seq}", "out": out.replace("\\", "/"), **fields}) + "\n")
        self.process.stdin.flush()
        keyword, _, body = self.process.stdout.readline().partition(" ")
        return keyword, json.loads(body)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=120)
        self.process.stderr.read()
        self.process.stderr.close()
        self.process.stdout.close()


class Case:
    """The shared checks; a subclass names the model, the reference and how a request is made."""
    KEY = ""
    MODEL = None

    @classmethod
    def setUpClass(cls):
        with open(REFERENCE, encoding="utf-8") as handle:
            cls.ref = json.load(handle)[cls.KEY]
        cls.tmp = tempfile.TemporaryDirectory()
        cls.refdir = os.path.join(cls.tmp.name, "ref")
        os.makedirs(cls.refdir)
        with open(os.path.join(cls.refdir, "codes.i32"), "wb") as handle:
            for frame in cls.ref["codes"]:
                handle.write(struct.pack("<16i", *frame))
        with open(os.path.join(cls.refdir, "input_ids.i32"), "wb") as handle:
            handle.write(struct.pack(f"<{len(cls.ref['input_ids'])}i", *cls.ref["input_ids"]))
        cls.engine = Engine(cls.MODEL)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def cli(self, *extra):
        args = [EXE, "--model", self.MODEL, "--text", self.ref["text"], "--language", self.ref["language"], "--instruct", self.ref["instruct"],
                "--ref", self.refdir]
        if self.ref["speaker"]:
            args += ["--speaker", self.ref["speaker"]]
        run = subprocess.run(args + list(extra), capture_output=True, text=True, encoding="utf-8", timeout=900)
        self.assertEqual(run.returncode, 0, run.stderr[-500:])
        return run.stdout

    def fields(self, **more):
        out = {"text": self.ref["text"], "language": self.ref["language"], "greedy": True}
        if self.ref["speaker"]:
            out["speaker"] = self.ref["speaker"]
        out.update(more)
        return out

    def test_the_prompt_is_the_references_prompt(self):
        self.assertRegex(self.cli("--teacher", "--max-frames", "1"), r"input ids: identical \(19 here, 19 in the reference\)")

    def test_every_code_the_reference_says_is_the_engines_own_argmax(self):
        out = self.cli("--teacher")
        frames = self.ref["frames"]
        match = re.search(r"teacher: (\d+) choices over (\d+) frames; (\d+) of the reference's codes were not my argmax", out)
        self.assertIsNotNone(match, out)
        self.assertEqual((int(match.group(1)), int(match.group(2)), int(match.group(3))), (frames * 16, frames, 0), out)
        self.assertIn("ends with the end token: yes", out)

    def test_free_running_and_greedy_it_writes_the_references_codes(self):
        self.assertIn(f"codes: identical ({self.ref['frames']} frames here, {self.ref['frames']} in the reference)", self.cli("--greedy"))

    def test_the_speech_follows_the_references_loudness(self):
        keyword, body = self.engine.speak(self.path("a.wav"), instruct=self.ref["instruct"], **self.fields())
        self.assertEqual(keyword, "AUDIO", body)
        self.assertEqual((body["frames"], body["samples"], body["truncated"]), (self.ref["frames"], self.ref["samples"], False))
        samples = read_wav(self.path("a.wav"))
        self.assertGreater(max(abs(s) for s in samples), 3000)
        for i, want in enumerate(self.ref["frame_rms"]):
            chunk = samples[i * 1920:(i + 1) * 1920]
            self.assertAlmostEqual((sum(s * s for s in chunk) / len(chunk)) ** 0.5 / 32768, want, delta=1e-3, msg=f"frame {i}")

    def test_another_instruction_makes_another_speech(self):
        results = []
        for name, style in (("one", self.ref["instruct"]), ("two", "Speak fast, loud and cheerful, with a high voice.")):
            keyword, body = self.engine.speak(self.path(name + ".wav"), instruct=style, **self.fields(max_frames=30))
            self.assertEqual(keyword, "AUDIO", body)
            results.append(read_wav(self.path(name + ".wav")))
        self.assertNotEqual(results[0], results[1])


@unittest.skipUnless(usable(os.environ.get("QWEN3TTS_MODEL_17")), "needs the built qwen3tts and QWEN3TTS_MODEL_17 (the 1.7B CustomVoice with tokenizer.json)")
class CustomVoice17Test(Case, unittest.TestCase):
    KEY = "custom_voice"
    MODEL = os.environ.get("QWEN3TTS_MODEL_17")

    def test_ready_says_the_model_takes_a_style(self):
        self.assertEqual((self.engine.ready["kind"], self.engine.ready["instruct"], self.engine.ready["layers"]), ("custom_voice", True, 28))
        self.assertIn("ryan", self.engine.ready["speakers"])

    def test_no_instruction_is_the_plain_speech_and_an_instruction_changes_it(self):
        spoken = []
        for name, style in (("plain", ""), ("styled", "Whisper, very softly.")):
            keyword, body = self.engine.speak(self.path(name + ".wav"), instruct=style, **self.fields(max_frames=30))
            self.assertEqual(keyword, "AUDIO", body)
            spoken.append(read_wav(self.path(name + ".wav")))
        self.assertNotEqual(spoken[0], spoken[1])


@unittest.skipUnless(usable(os.environ.get("QWEN3TTS_MODEL_DESIGN")), "needs the built qwen3tts and QWEN3TTS_MODEL_DESIGN (the 1.7B VoiceDesign with tokenizer.json)")
class VoiceDesign17Test(Case, unittest.TestCase):
    KEY = "voice_design"
    MODEL = os.environ.get("QWEN3TTS_MODEL_DESIGN")

    def test_ready_says_there_are_no_speakers_and_a_description_is_taken(self):
        self.assertEqual((self.engine.ready["kind"], self.engine.ready["instruct"], self.engine.ready["speakers"]), ("voice_design", True, []))

    def test_a_named_speaker_is_not_a_voice_here(self):
        keyword, body = self.engine.speak(self.path("s.wav"), instruct=self.ref["instruct"], **self.fields(speaker="ryan", max_frames=10))
        self.assertEqual(keyword, "AUDIO", body)                         # ignored: the description makes the voice


if __name__ == "__main__":
    unittest.main()
