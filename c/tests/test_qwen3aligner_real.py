"""c/qwen3asr on the real Qwen3-ForcedAligner-0.6B, against the official package.

    QWEN3ALIGNER_MODEL=/path/to/Qwen3-ForcedAligner-0.6B (with tokenizer.json) QWEN3ALIGNER_CLIP=/path/to/jfk.wav (the 11 s JFK clip, 16 kHz mono)
    [QWEN3ALIGNER_CLIP3=/path/to/jfk3.wav (the same clip three times, 33 s)] [QWEN3ALIGNER_REF=/path/to/align_jfk (tools/qwen3_aligner_ref.py's folder)]
    python -m unittest tests.test_qwen3aligner_real

tools/qwen3_aligner_reference.json holds the package's times (fp32, CPU) for the clip: the engine must give the same milliseconds for every word, on the CPU
and, with COLI_VULKAN=1 and a VK=1 build, on the card. With QWEN3ALIGNER_REF the stage tensors are held to the package's too (the prompt's ids, the class at
every <timestamp>, the head's logits).
"""
import json
import os
import re
import subprocess
import tempfile
import unittest

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXE = next((p for p in (os.path.join(C_DIR, "qwen3asr.exe"), os.path.join(C_DIR, "qwen3asr")) if os.path.isfile(p)), None)
MODEL = os.environ.get("QWEN3ALIGNER_MODEL")
CLIP = os.environ.get("QWEN3ALIGNER_CLIP")
CLIP3 = os.environ.get("QWEN3ALIGNER_CLIP3")
REF_DIR = os.environ.get("QWEN3ALIGNER_REF")
REFERENCE = os.path.join(C_DIR, "tools", "qwen3_aligner_reference.json")
READY = bool(EXE and MODEL and os.path.isdir(MODEL) and os.path.isfile(os.path.join(MODEL, "tokenizer.json")) and CLIP and os.path.isfile(CLIP))


@unittest.skipUnless(READY, "needs the built qwen3asr, QWEN3ALIGNER_MODEL (the aligner with tokenizer.json) and QWEN3ALIGNER_CLIP (the JFK clip)")
class AlignerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REFERENCE, encoding="utf-8") as handle:
            cls.ref = json.load(handle)
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

    def send(self, verb, **fields):
        type(self).seq += 1
        self.process.stdin.write(f"{verb} " + json.dumps({"id": f"t{self.seq}", **fields}) + "\n")
        self.process.stdin.flush()
        keyword, _, body = self.process.stdout.readline().partition(" ")
        return keyword, json.loads(body)

    def test_ready_says_it_aligns(self):
        self.assertEqual((self.ready["task"], self.ready["piece_seconds"], self.ready["layers"], self.ready["encoder_layers"]), ("align", 300, 28, 24))

    def test_every_word_is_where_the_package_puts_it(self):
        ref = self.ref["jfk"]
        keyword, body = self.send("ALIGN", wav=CLIP.replace("\\", "/"), words=ref["words"])
        self.assertEqual(keyword, "ALIGNED", body)
        fixed = ref["fixed_ms"]
        self.assertEqual(body["starts"], fixed[0::2])
        self.assertEqual(body["ends"], fixed[1::2])
        self.assertAlmostEqual(body["audio_seconds"], 11.0, places=2)

    @unittest.skipUnless(CLIP3 and os.path.isfile(CLIP3 or ""), "needs QWEN3ALIGNER_CLIP3 (the clip three times)")
    def test_a_longer_clip_with_three_times_the_words_agrees_too(self):
        ref = self.ref["jfk3"]
        keyword, body = self.send("ALIGN", wav=CLIP3.replace("\\", "/"), words=ref["words"])
        self.assertEqual(keyword, "ALIGNED", body)
        self.assertEqual(body["starts"], ref["fixed_ms"][0::2])
        self.assertEqual(body["ends"], ref["fixed_ms"][1::2])

    def test_the_times_never_go_backwards(self):
        keyword, body = self.send("ALIGN", wav=CLIP.replace("\\", "/"), words=["Some", "words", "that", "are", "not", "in", "the", "audio", "at", "all"])
        self.assertEqual(keyword, "ALIGNED", body)
        flat = [t for pair in zip(body["starts"], body["ends"]) for t in pair]
        self.assertEqual(flat, sorted(flat))

    def test_bad_requests_are_errors_and_the_engine_goes_on(self):
        for fields, fragment in (({"wav": CLIP.replace("\\", "/"), "words": []}, "words"), ({"wav": CLIP.replace("\\", "/")}, "words"),
                                 ({"words": ["a"]}, "wav"), ({"wav": "/no/such.wav", "words": ["a"]}, "")):
            keyword, body = self.send("ALIGN", **fields)
            self.assertEqual(keyword, "ERROR", fields)
            self.assertIn(fragment, body["message"])
        keyword, body = self.send("TRANSCRIBE", wav=CLIP.replace("\\", "/"))
        self.assertEqual(keyword, "ERROR")
        self.assertIn("forced aligner", body["message"])
        self.assertEqual(self.send("ALIGN", wav=CLIP.replace("\\", "/"), words=["and"])[0], "ALIGNED")

    def test_a_word_the_tokenizer_splits_into_pieces_and_unusual_letters_are_aligned(self):
        keyword, body = self.send("ALIGN", wav=CLIP.replace("\\", "/"), words=["And", "so", "müsli", "fellow", "Americans"])
        self.assertEqual((keyword, len(body["starts"])), ("ALIGNED", 5))


@unittest.skipUnless(READY and REF_DIR and os.path.isdir(REF_DIR or ""), "needs QWEN3ALIGNER_REF (the folder tools/qwen3_aligner_ref.py wrote)")
class StagesAgainstThePackageTest(unittest.TestCase):
    def test_the_prompt_the_classes_and_the_logits_agree(self):
        words = os.path.join(tempfile.mkdtemp(), "words.txt")
        with open(REFERENCE, encoding="utf-8") as ref_file:
            jfk_words = json.load(ref_file)["jfk"]["words"]
        with open(words, "w", encoding="utf-8") as handle:
            handle.write("\n".join(jfk_words) + "\n")
        run = subprocess.run([EXE, "--model", MODEL, "--wav", CLIP, "--words-file", words, "--ref", REF_DIR], capture_output=True, text=True, encoding="utf-8", timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-500:])
        self.assertIn("prompt ids: identical (212 here, 212 in the reference)", run.stdout)
        self.assertIn("classes: 0 of 44 differ", run.stdout)
        self.assertIn("fixed times: 0 of 44 differ", run.stdout)
        match = re.search(r"logits\s+n=\d+\s+max abs diff ([0-9.e+-]+)", run.stdout)
        self.assertIsNotNone(match, run.stdout)
        self.assertLess(float(match.group(1)), 1e-4)


if __name__ == "__main__":
    unittest.main()
