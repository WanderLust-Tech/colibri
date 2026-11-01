"""The tiny ACE-Step fixture (tools/make_acestep_tiny.py) has the real checkpoint's structure.

    ACESTEP_TINY=/path/to/fixture python -m unittest tests.test_acestep_fixture

Skipped without ACESTEP_TINY. Standard library only: the fixture's safetensors headers are read directly and
compared with tools/acestep_manifest.json, the manifest of the real turbo checkpoint. What must match is every
tensor name (layer indices collapsed), its dtype and its rank; what may differ is every width and layer count.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import acestep_manifest as am  # noqa: E402

TINY = os.environ.get("ACESTEP_TINY")


@unittest.skipUnless(TINY and os.path.isdir(TINY), "needs ACESTEP_TINY (tools/make_acestep_tiny.py OUTDIR)")
class TinyFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(am.DEFAULT_MANIFEST, encoding="utf-8") as f:
            cls.real = json.load(f)
        cls.tables = {c: am.read_header(os.path.join(TINY, *rel.split("/"))) for c, rel in am.COMPONENTS.items()}

    def test_every_component_has_the_real_tensor_names_dtypes_and_ranks(self):
        for component, table in self.tables.items():
            with self.subTest(component=component):
                want, have = am.structure(self.real[component]), am.structure(table)
                self.assertEqual(sorted(set(want) - set(have)), [], "in the real checkpoint, not in the fixture")
                self.assertEqual(sorted(set(have) - set(want)), [], "in the fixture, not in the real checkpoint")
                self.assertEqual(have, want)

    def test_the_stacks_keep_their_proportions(self):
        """Layer counts are tiny but every stack is still a stack, with the real per-layer tensor count."""
        dit = am.collapse(self.tables["dit"])
        for pattern, layers in (("decoder.layers.N.self_attn.q_proj.weight", 4),
                                ("encoder.lyric_encoder.layers.N.self_attn.q_proj.weight", 2),
                                ("encoder.timbre_encoder.layers.N.self_attn.q_proj.weight", 2)):
            self.assertEqual(dit[pattern]["count"], layers, pattern)
        real = self.real["dit"]
        kinds = lambda table, prefix: sorted(p for p in table if p.startswith(prefix))   # tensors in one DiT layer
        self.assertEqual(kinds(dit, "decoder.layers.N."), kinds(real, "decoder.layers.N."))
        # 6 self-attention + 6 cross-attention (q, k, v, o, q_norm, k_norm) + 3 MLP + 3 norms + the modulation table
        self.assertEqual(len(kinds(real, "decoder.layers.N.")), 19)

    def test_the_fixed_geometry_is_the_real_one(self):
        """64 latent channels at 25 Hz, patch 2, 192 DiT input channels, 48 kHz stereo: the engine reads the rest."""
        dit, vae = self.tables["dit"], self.tables["vae"]
        self.assertEqual(dit["decoder.proj_in.1.weight"]["shape"][1:], [192, 2])
        self.assertEqual(dit["decoder.proj_out.1.weight"]["shape"][1:], [64, 2])
        self.assertEqual(vae["decoder.conv1.weight_v"]["shape"][1:], [64, 7])
        self.assertEqual(vae["decoder.conv2.weight_v"]["shape"][0], 2)
        with open(os.path.join(TINY, "vae", "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        hop = 1
        for r in cfg["downsampling_ratios"]:
            hop *= r
        self.assertEqual((hop, cfg["sampling_rate"], cfg["audio_channels"]), (1920, 48000, 2))

    def test_everything_is_bf16_and_the_widths_are_tiny(self):
        for component, table in self.tables.items():
            self.assertEqual({t["dtype"] for t in table.values()}, {"BF16"}, component)
        widest = max(max(t["shape"]) for t in self.tables["dit"].values())
        self.assertLessEqual(widest, 750 * 2, "the fixture is meant to be small")
        total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(TINY) for f in fs)
        self.assertLess(total, 20 << 20)


if __name__ == "__main__":
    unittest.main()
