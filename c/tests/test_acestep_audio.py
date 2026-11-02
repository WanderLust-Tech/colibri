"""acestep_audio.h: reading a WAV or an MP3, and turning it into stereo at 48 kHz, held to upstream's own audio input.

What is checked, and with what:

  * the loader and the normaliser against upstream, sample for sample (tools/acestep_audio_ref.py: soundfile's libsndfile and
    mpg123 for what is read, torchaudio's Resample and the clamp for what is made of it), on WAV files of every sample format,
    a LIST chunk, an odd chunk, a truncated file, an extensible header, and on MP3 files at three sample rates and both MPEG
    versions, with ID3v2/ID3v1 tags, with and without a Xing/Info frame, a CRC-protected one, a cut one; and on a 6-minute
    clip whose length depends on torch's float32 ceil. Needs the reference venv and ACESTEP_UPSTREAM;
  * the robustness suite (garbage, empty, a header that claims 4 GB, a zero-length chunk, an MP3 that claims two billion
    frames, the cap, an absurd resampling kernel, 150 damaged files): needs numpy and soundfile, not upstream;
  * that every mutant tools/mutate_acestep_audio.py applies still applies: needs nothing but a Python. Running the mutants
    (the proof that the oracle can fail) is `PYTHONUTF8=1 python tools/mutate_acestep_audio.py`.

Run it with the reference venv's python: the system python skips what it cannot import.
"""
import os
import shutil
import sys
import tempfile
import unittest

C_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(C_DIR, "tools"))
import mutate_acestep_audio as mut  # noqa: E402  (imports no numpy)

try:
    import numpy  # noqa: F401
    import soundfile  # noqa: F401
    import acestep_audio_ref as ref
    HAVE_REF = True
except ImportError:
    HAVE_REF = False

CC = os.environ.get("CC") or shutil.which("gcc")


class Mutants(unittest.TestCase):
    """What the mutation run needs and this can check without compiling anything."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(C_DIR, mut.HEADER), encoding="utf-8", newline="") as f:
            cls.source = f.read()

    def test_every_mutant_text_occurs_as_often_as_it_says(self):
        for name, entry in list(mut.MUTANTS.items()) + [(n, e[0]) for n, e in mut.EQUIVALENT.items()]:
            with self.subTest(mutant=name):
                text, why = mut.apply_edits(self.source, entry)
                self.assertIsNotNone(text, why)

    def test_every_mutant_changes_something(self):
        for name, entry in mut.MUTANTS.items():
            with self.subTest(mutant=name):
                text, _ = mut.apply_edits(self.source, entry)
                self.assertNotEqual(text, self.source)

    def test_the_mutants_are_distinct_edits(self):
        edits = [repr(e) for e in mut.MUTANTS.values()]
        self.assertEqual(len(edits), len(set(edits)))

    def test_the_families_the_oracle_must_cover_are_all_present(self):
        names = " ".join(mut.MUTANTS)
        for family in ("rolloff", "window", "sinc", "kernel", "phase", "length", "clamp", "mono", "third-channel", "wav8", "wav16",
                       "wav24", "wav32", "wav-chunk-pad", "wav-truncated", "wavex", "mp3-id3v2", "mp3-id3v1", "mp3-gapless",
                       "mp3-info-frame", "mp3-delay", "mp3-padding", "mp3-cap", "flac", "garbage"):
            self.assertIn(family, names)

    def test_the_slow_mutant_has_its_fixture(self):
        self.assertIn(mut.SLOW_CASE, mut.CASES["length-not-float32"])
        for n in mut.CASES:
            self.assertIn(n, mut.MUTANTS)


@unittest.skipUnless(HAVE_REF and CC, "needs numpy, soundfile and a C compiler (the reference venv)")
class Robustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="acestep-audio-test-")
        cls.exe, err = ref.build_tool(CC, cls.work, extra=("-Werror",))        # -Wall -Wextra clean, the vendored header excepted
        assert cls.exe, "the header does not build warning-free: " + err

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_refusals_are_clean_and_say_why(self):
        problems = ref.run_robustness(self.exe, self.work, verbose=False)
        self.assertEqual(problems, [])

    def test_a_refusal_does_not_allocate_what_the_header_claims(self):
        """A header that claims 4 GB of data, in a file of 8 KB: what is there is read, and nothing like 4 GB is touched."""
        path = os.path.join(self.work, "claim.wav")
        ref._wav_raw(path, 1, 2, 44100, 16, bytes(8000), data_size=0xFFFFFFFF, riff_size=0xFFFFFFFF)
        rc, line, mb = ref.run_tool(self.exe, "load", path, os.path.join(self.work, "o.f32"))
        self.assertEqual((rc, line), (0, "OK rate=44100 channels=2 frames=2000"))
        if os.name == "nt":
            self.assertIsNotNone(mb, "the peak working set of the child could not be read")
        if mb is not None:
            self.assertLess(mb, 100.0)


@unittest.skipUnless(HAVE_REF and CC and os.environ.get("ACESTEP_UPSTREAM"), "needs the reference venv and ACESTEP_UPSTREAM")
class OracleAgainstUpstream(unittest.TestCase):
    def test_every_fixture_matches_upstream(self):
        try:
            import torchaudio  # noqa: F401
        except ImportError:
            self.skipTest("torchaudio is not installed")
        with tempfile.TemporaryDirectory(prefix="acestep-audio-oracle-") as work:
            exe, err = ref.build_tool(CC, work)
            self.assertIsNotNone(exe, err)
            cases = ref.prepare_cases(work, slow=True)
            self.assertGreaterEqual(len(cases), 26)
            self.assertEqual(ref.run_oracle(exe, cases, work, verbose=False), [])

    def test_the_tolerances_are_the_measured_ones(self):
        """Documented in the module docstring: they are what the mutants were judged against."""
        self.assertEqual(ref.TOL_LOAD_WAV, 0.0)
        self.assertEqual(ref.TOL_NORM_WAV, 6e-7)
        self.assertEqual(ref.TOL_LOAD_MP3, 2e-5)
        self.assertEqual(ref.TOL_NORM_MP3, 2e-5)


if __name__ == "__main__":
    unittest.main()
