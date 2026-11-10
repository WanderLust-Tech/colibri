"""`coli model` and the GPU column of `coli ps`, run as the real command against a router over fake workers.

    python -m unittest tests.test_coli_model
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(HERE))

import model_settings  # noqa: E402
from test_gpu import CapturingRig, FreeCard, policy  # noqa: E402
from test_router import entry  # noqa: E402


def coli(*args, key=None):
    env = {**os.environ, "NO_COLOR": "1", "PYTHONUTF8": "1", "COLI_SETUP_HOME": tempfile.mkdtemp()}
    env.pop("COLI_API_KEY", None)
    if key:
        env["COLI_API_KEY"] = key
    done = subprocess.run([sys.executable, str(C_DIR / "coli"), *args], capture_output=True, text=True, env=env, timeout=120)
    return done.returncode, done.stdout + done.stderr


class CliTest(unittest.TestCase):
    def rig(self, **kw):
        self.path = os.path.join(tempfile.mkdtemp(), "model-settings.json")
        return CapturingRig(self, [entry("text-a"), entry("text-b")], settings=model_settings.ModelSettings(self.path), **kw)

    def test_disable_enable_and_list(self):
        rig = self.rig()
        port = str(rig.port)
        code, out = coli("model", "disable", "text-a", "--port", port)
        self.assertEqual(code, 0, out)
        self.assertIn("text-a-colibri: off", out)
        self.assertEqual(model_settings.read(self.path), {"text-a-colibri": {"enabled": False}})
        code, out = coli("model", "--port", port)
        self.assertEqual(code, 0, out)
        rows = {line.split()[0]: line.split() for line in out.splitlines() if line.strip().startswith("text-")}
        self.assertEqual(rows["text-a-colibri"][1], "off")
        self.assertEqual(rows["text-b-colibri"][1], "yes")
        code, out = coli("model", "enable", "text-a", "--port", port)
        self.assertEqual(code, 0, out)
        self.assertIn("text-a-colibri: on", out)
        self.assertEqual(model_settings.read(self.path), {})

    def test_gpu_pin(self):
        rig = self.rig()
        port = str(rig.port)
        code, out = coli("model", "gpu", "text-b", "cpu", "--port", port)
        self.assertEqual(code, 0, out)
        self.assertIn("runs on cpu", out)
        self.assertEqual(model_settings.read(self.path), {"text-b-colibri": {"gpu": "cpu"}})
        code, out = coli("model", "gpu", "text-b", "auto", "--port", port)
        self.assertEqual(model_settings.read(self.path), {})

    def test_it_says_what_is_wrong(self):
        rig = self.rig()
        port = str(rig.port)
        code, out = coli("model", "disable", "--port", port)
        self.assertEqual(code, 1)
        self.assertIn("name a model", out)
        code, out = coli("model", "gpu", "text-a", "--port", port)
        self.assertEqual(code, 1)
        self.assertIn("auto|vulkan|cpu", out)
        code, out = coli("model", "gpu", "text-a", "fast", "--port", port)
        self.assertEqual(code, 1)
        code, out = coli("model", "disable", "no-such-model", "--port", port)
        self.assertEqual(code, 1)
        self.assertIn("no installed model is called no-such-model", out)
        self.assertEqual(model_settings.read(self.path), {})

    def test_the_key_is_needed_when_the_router_has_one(self):
        rig = self.rig(api_key="sekrit")
        port = str(rig.port)
        code, out = coli("model", "disable", "text-a", "--port", port)
        self.assertEqual(code, 1)
        self.assertIn("API key", out)
        code, out = coli("model", "disable", "text-a", "--port", port, key="sekrit")
        self.assertEqual(code, 0, out)

    def test_list_shows_where_a_loaded_model_runs_and_the_policy(self):
        rig = self.rig(gpu=policy("auto", FreeCard(20.0)))
        rig.chat("text-a-colibri")
        code, out = coli("model", "--port", str(rig.port))
        self.assertEqual(code, 0, out)
        line = next(l for l in out.splitlines() if l.strip().startswith("text-a-colibri"))
        self.assertIn("on the GPU", line)
        self.assertIn("--gpu auto", out)
        self.assertIn("20.0 of 32.0 GB free", out)

    def test_ps_says_gpu_or_cpu(self):
        card = FreeCard(20.0)
        rig = self.rig(gpu=policy("auto", card))
        rig.chat("text-a-colibri")
        card.free = 3.0
        rig.chat("text-b-colibri")
        code, out = coli("ps", "--port", str(rig.port))
        self.assertEqual(code, 0, out)
        self.assertIn("RUNS ON", out)
        runs = {l.split()[0]: l.split() for l in out.splitlines() if l.strip().startswith("text-")}
        self.assertIn("GPU", runs["text-a-colibri"])
        self.assertIn("CPU", runs["text-b-colibri"])

    def test_with_no_server_a_name_that_is_not_installed_is_said(self):
        code, out = coli("model", "disable", "definitely-not-installed", "--port", "1")
        self.assertEqual(code, 1)
        self.assertIn("no installed model is called", out)

    def test_with_no_server_list_reads_the_file(self):
        path = os.path.join(tempfile.mkdtemp(), "m.json")
        model_settings.update(path, "gemma-4-31b-colibri", enabled=False)
        code, out = coli("model", "--port", "1", "--file", path)
        self.assertEqual(code, 0, out)
        self.assertIn("gemma-4-31b-colibri", out)
        self.assertIn("enabled=no", out)


if __name__ == "__main__":
    unittest.main()
