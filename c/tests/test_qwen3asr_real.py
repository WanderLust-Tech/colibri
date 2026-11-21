"""The real engine (c/qwen3asr on a Qwen3-ASR checkpoint) against the answer transformers gives, its protocol, and long audio.

    QWEN3ASR_MODEL=/path/to/Qwen3-ASR-0.6B QWEN3ASR_CLIP=/path/to/jfk.wav python -m unittest tests.test_qwen3asr_real

Needs the built engine (make qwen3asr), the 0.6B checkpoint with its tokenizer.json (coli model install qwen3-asr-0.6b writes it; or
python tools/qwen3_tokenizer.py DIR), and the 11 s JFK clip (whisper.cpp's samples/jfk.wav, public domain, 16 kHz mono); skipped
without them. The reference is tools/qwen3_asr_0.6b_reference.json, made by tools/qwen3_asr_ref.py from the official qwen-asr package
in fp32, so these tests need neither torch nor transformers. QWEN3ASR_REF, a folder written by that tool, adds the stage-by-stage
tensor comparison (features, the encoder, the prompt, the generated ids).
"""
import json
import os
import re
import subprocess
import tempfile
import unittest
import wave

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXE = next((p for p in (os.path.join(C_DIR, "qwen3asr.exe"), os.path.join(C_DIR, "qwen3asr")) if os.path.isfile(p)), None)
MODEL = os.environ.get("QWEN3ASR_MODEL")
CLIP = os.environ.get("QWEN3ASR_CLIP")
REF_DIR = os.environ.get("QWEN3ASR_REF")
REFERENCE = os.path.join(C_DIR, "tools", "qwen3_asr_0.6b_reference.json")
READY = bool(EXE and MODEL and os.path.isdir(MODEL) and CLIP and os.path.isfile(CLIP) and os.path.isfile(os.path.join(MODEL, "tokenizer.json")))


class Engine:
    def __init__(self, env=None):
        self.process = subprocess.Popen([EXE, "--model", MODEL, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1, env=dict(os.environ, **env) if env else None)
        self.ready = self.process.stdout.readline()
        self.seq = 0

    def send(self, verb, **fields):
        self.seq += 1
        self.process.stdin.write(f"{verb} " + json.dumps({"id": f"t{self.seq}", **fields}) + "\n")
        self.process.stdin.flush()
        keyword, _, body = self.process.stdout.readline().partition(" ")
        return keyword, json.loads(body)

    def transcribe(self, wav, **fields):
        return self.send("TRANSCRIBE", wav=wav, **fields)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=120)
        self.stderr = self.process.stderr.read()
        self.process.stderr.close()
        self.process.stdout.close()


@unittest.skipUnless(READY, "needs the built qwen3asr, QWEN3ASR_MODEL (the 0.6B with tokenizer.json) and QWEN3ASR_CLIP (the JFK clip)")
class RealEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REFERENCE, encoding="utf-8") as handle:
            cls.ref = json.load(handle)
        cls.engine = Engine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_ready_says_what_the_engine_is(self):
        keyword, _, body = self.engine.ready.partition(" ")
        info = json.loads(body)
        self.assertEqual(keyword, "READY")
        self.assertEqual((info["task"], info["sample_rate"], info["layers"], info["encoder_layers"]), ("transcribe", 16000, 28, 18))
        self.assertEqual(info["piece_seconds"], 1200)

    def test_the_clip_comes_back_as_transformers_hears_it(self):
        keyword, body = self.engine.transcribe(CLIP)
        self.assertEqual(keyword, "TEXT", body)
        self.assertEqual(body["text"], self.ref["text"])
        self.assertEqual(body["language"], self.ref["language"])
        self.assertEqual(body["tokens"], len(self.ref["generated"]))
        self.assertEqual((body["pieces"], body["truncated"]), (1, False))
        self.assertAlmostEqual(body["audio_seconds"], 11.0, places=2)

    def test_a_forced_language_gives_the_text_alone(self):
        keyword, body = self.engine.transcribe(CLIP, language="English")
        self.assertEqual(keyword, "TEXT", body)
        self.assertEqual(body["text"], self.ref["text"])
        self.assertEqual(body["language"], "English")
        self.assertLess(body["tokens"], len(self.ref["generated"]))              # it was not asked to write the language tag

    def test_a_context_is_taken_and_the_text_is_still_right(self):
        keyword, body = self.engine.transcribe(CLIP, context="A speech by John F. Kennedy, 1961.")
        self.assertEqual(keyword, "TEXT", body)
        self.assertIn("ask not what your country can do for you", body["text"])

    def test_a_bad_request_is_an_error_and_the_engine_goes_on(self):
        self.assertEqual(self.engine.send("TRANSCRIBE")[0], "ERROR")                           # no wav
        keyword, body = self.engine.transcribe(os.path.join(tempfile.gettempdir(), "no-such-file-here.wav"))
        self.assertEqual(keyword, "ERROR")
        self.assertIn("cannot open", body["message"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "eight.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(8000)
                w.writeframes(b"\0\0" * 8000)
            keyword, body = self.engine.transcribe(path)
            self.assertEqual(keyword, "ERROR")
            self.assertIn("16000", body["message"])
            junk = os.path.join(tmp, "junk.wav")
            with open(junk, "wb") as handle:
                handle.write(b"not a wave file at all")
            self.assertEqual(self.engine.transcribe(junk)[0], "ERROR")
        self.assertEqual(self.engine.transcribe(CLIP)[0], "TEXT")                              # still alive

    def test_audio_with_nothing_in_it_gives_an_empty_text_or_none_at_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "silence.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\0\0" * 16000 * 2)
            keyword, body = self.engine.transcribe(path)
            self.assertEqual(keyword, "TEXT", body)
            self.assertLess(len(body["text"]), 60)                                              # the model may say nothing or little

    def test_a_clip_cut_into_pieces_is_joined_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "three.wav")
            with wave.open(CLIP, "rb") as source:
                params, frames = source.getparams(), source.readframes(source.getnframes())
            with wave.open(path, "wb") as w:
                w.setparams(params)
                w.writeframes(frames * 3)
            engine = Engine(env={"QWEN3ASR_MAX_PIECE_S": "15"})                                 # a 33 s file in pieces of at most 15 s
            try:
                keyword, body = engine.transcribe(path)
            finally:
                engine.close()
        self.assertEqual(keyword, "TEXT", body)
        self.assertEqual(body["pieces"], 3)
        words = re.sub(r"\s+", " ", body["text"])
        self.assertGreaterEqual(words.count("ask not what your country can do for you"), 3)
        self.assertNotRegex(body["text"], r"country\.And")                                      # the join put a space where the pieces met


@unittest.skipUnless(READY and REF_DIR and os.path.isdir(REF_DIR or ""), "needs QWEN3ASR_REF (a folder written by tools/qwen3_asr_ref.py)")
class StagesAgainstTransformersTest(unittest.TestCase):
    """The engine's own comparison mode: each stage against the reference tensors."""

    @classmethod
    def setUpClass(cls):
        done = subprocess.run([EXE, "--model", MODEL, "--wav", CLIP, "--ref", REF_DIR], capture_output=True, text=True, encoding="utf-8", timeout=600)
        cls.out = done.stdout
        cls.code = done.returncode

    @staticmethod
    def worst(out, label):
        found = re.search(rf"^{label}\s+n=\d+\s+max abs diff ([0-9.e+-]+)", out, re.M)
        return float(found.group(1)) if found else None

    def test_the_stages_agree(self):
        self.assertEqual(self.code, 0, self.out)
        features, conv, encoder = (self.worst(self.out, name) for name in ("features", "conv", "encoder"))
        self.assertIsNotNone(features, self.out)
        self.assertLess(features, 1e-4)                         # the log-mel features
        self.assertLess(conv, 1e-4)                             # after the three convolutions and conv_out
        self.assertLess(encoder, 1e-3)                          # the audio encoder's output
        self.assertIn("prompt ids: identical", self.out)
        self.assertIn("generated ids: identical", self.out)


if __name__ == "__main__":
    unittest.main()
