"""`coli launch`: the environment and configuration it hands a coding tool, and the command run as a user runs it, against a
real router over fake workers and a fake tool that calls the server with exactly what it was given."""
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import launch  # noqa: E402
from test_router import Rig, entry  # noqa: E402
from test_router_cli import coli  # noqa: E402

FAKE_TOOL = r'''
import json, os, sys, urllib.request
base = os.environ.get("ANTHROPIC_BASE_URL")
token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
status = None
if base:
    req = urllib.request.Request(base + "/v1/models", headers={"Authorization": "Bearer " + token})
    try:
        status = urllib.request.urlopen(req, timeout=10).status
    except Exception as error:
        status = getattr(error, "code", str(error))
seen = {k: v for k, v in os.environ.items() if k.startswith(("ANTHROPIC_", "CLAUDE_CODE_", "ENABLE_CLAUDEAI", "OPENCODE_"))}
with open(os.environ["FAKE_TOOL_OUT"], "w") as handle:
    json.dump({"args": sys.argv[1:], "env": seen, "models_status": status}, handle)
sys.exit(int(os.environ.get("FAKE_TOOL_EXIT", "0")))
'''


class FakeTools:
    """A directory holding `claude` and `opencode` programs that record what they were started with."""

    def __init__(self, test):
        self.tmp = tempfile.TemporaryDirectory()
        test.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.out = os.path.join(self.dir, "seen.json")
        script = os.path.join(self.dir, "fake_tool.py")
        Path(script).write_text(FAKE_TOOL, encoding="utf-8")
        for name in ("claude", "opencode"):
            if os.name == "nt":
                Path(self.dir, name + ".bat").write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
            else:
                path = Path(self.dir, name)
                path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
                path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def env(self, **more):
        return {"PATH": self.dir + os.pathsep + os.environ.get("PATH", ""), "FAKE_TOOL_OUT": self.out, **more}

    def seen(self):
        with open(self.out, encoding="utf-8") as handle:
            return json.load(handle)


class EnvironmentTest(unittest.TestCase):
    def test_claude_gets_the_server_the_key_and_the_model_for_every_tier(self):
        env = launch.claude_env("http://127.0.0.1:8000", "qwen-colibri", "sekrit", environ={})
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8000")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "sekrit")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "")
        for key in ("ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                    "CLAUDE_CODE_SUBAGENT_MODEL"):
            self.assertEqual(env[key], "qwen-colibri", key)
        self.assertEqual(env["CLAUDE_CODE_ATTRIBUTION_HEADER"], "0")      # keeps the KV cache's prefix steady

    def test_claude_gets_a_placeholder_when_the_server_has_no_key(self):
        env = launch.claude_env("http://h:1", "m", None, environ={})
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], launch.PLACEHOLDER_KEY)

    def test_the_users_own_auto_mode_choice_is_kept(self):
        self.assertEqual(launch.claude_env("http://h:1", "m", None, environ={})["CLAUDE_CODE_AUTO_MODE_SERVER"], "0")
        self.assertNotIn("CLAUDE_CODE_AUTO_MODE_SERVER",
                         launch.claude_env("http://h:1", "m", None, environ={"CLAUDE_CODE_AUTO_MODE_SERVER": "1"}))

    def test_opencode_config_names_one_provider_with_every_chat_model(self):
        config = json.loads(launch.opencode_config("http://h:8000", "b", ["a", "b"], "sekrit"))
        provider = config["provider"]["colibri"]
        self.assertEqual(provider["npm"], "@ai-sdk/openai-compatible")
        self.assertEqual(provider["options"], {"baseURL": "http://h:8000/v1", "apiKey": "sekrit"})
        self.assertEqual(list(provider["models"]), ["a", "b"])
        self.assertEqual(config["model"], "colibri/b")

    def test_opencode_config_has_no_key_when_there_is_none_and_always_lists_the_chosen_model(self):
        config = json.loads(launch.opencode_config("http://h:8000", "z", ["a"], None))
        self.assertNotIn("apiKey", config["provider"]["colibri"]["options"])
        self.assertIn("z", config["provider"]["colibri"]["models"])

    def test_plan_arguments(self):
        env, args = launch.plan("claude", "http://h:1", "m", ["m"], None, ["-p", "hi"], environ={})
        self.assertEqual(args, ["--model", "m", "-p", "hi"])
        env, args = launch.plan("opencode", "http://h:1", "m", ["m"], None, [])
        self.assertEqual(args, ["--model", "colibri/m"])
        self.assertEqual(list(env), ["OPENCODE_CONFIG_CONTENT"])
        env, args = launch.plan("opencode", "http://h:1", "m", ["m"], None, ["run", "fix it"])
        self.assertEqual(args, ["run", "fix it"])           # the config's own `model` applies; the user's words are not mixed with ours
        with self.assertRaises(KeyError):
            launch.plan("codex", "http://h:1", "m", ["m"], None)


class ModelChoiceTest(unittest.TestCase):
    def test_a_router_lists_chat_models_with_the_loaded_ones_first(self):
        installed = {"router": True, "data": [
            {"id": "pic", "modality": "image", "state": "ready"},
            {"id": "a", "modality": "text", "state": "idle"},
            {"id": "song", "modality": "audio", "state": "idle"},
            {"id": "b", "modality": "text", "state": "ready"},
            {"id": "c", "modality": "text", "state": "idle"}]}
        self.assertEqual(launch.chat_models(installed, None), ["b", "a", "c"])

    def test_a_single_server_is_read_from_its_capabilities(self):
        models = {"data": [{"id": "x", "capabilities": ["chat", "systemone"]}]}
        self.assertEqual(launch.chat_models({"data": []}, models), ["x"])
        pic = {"data": [{"id": "p", "capabilities": ["image_generation"]}]}
        self.assertEqual(launch.chat_models(None, pic), [])
        self.assertEqual(launch.chat_models(None, None), [])


class CommandTest(unittest.TestCase):
    def test_without_a_tool_it_says_which_it_knows(self):
        tools = FakeTools(self)
        code, out = coli("launch", env_extra=tools.env())
        self.assertEqual(code, 0, out)
        self.assertIn("claude", out)
        self.assertIn("opencode", out)
        self.assertIn("found", out)

    def test_claude_is_started_with_an_environment_the_server_accepts(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")], api_key="sekrit")
        tools = FakeTools(self)
        code, out = coli("launch", "claude", "--port", str(rig.port), "--api-key", "sekrit", "--",
                         "-p", "hello", env_extra=tools.env())
        self.assertEqual(code, 0, out)
        seen = tools.seen()
        self.assertEqual(seen["args"], ["--model", "text-a-colibri", "-p", "hello"])
        self.assertEqual(seen["env"]["ANTHROPIC_BASE_URL"], f"http://127.0.0.1:{rig.port}")
        self.assertEqual(seen["env"]["ANTHROPIC_AUTH_TOKEN"], "sekrit")
        self.assertEqual(seen["models_status"], 200, "the tool could not use the server with what it was given")

    def test_a_model_that_is_already_loaded_is_the_default(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")])
        rig.chat("text-b-colibri")
        tools = FakeTools(self)
        code, out = coli("launch", "claude", "--port", str(rig.port), env_extra=tools.env())
        self.assertEqual(code, 0, out)
        self.assertEqual(tools.seen()["args"][:2], ["--model", "text-b-colibri"])

    def test_model_chooses_and_is_checked(self):
        rig = Rig(self, [entry("text-a"), entry("text-b"), entry("pic", modality="image")])
        tools = FakeTools(self)
        code, out = coli("launch", "claude", "--port", str(rig.port), "--model", "text-b-colibri", env_extra=tools.env())
        self.assertEqual(code, 0, out)
        self.assertEqual(tools.seen()["args"][:2], ["--model", "text-b-colibri"])
        os.unlink(tools.out)
        for wrong in ("nope", "pic-colibri"):
            code, out = coli("launch", "claude", "--port", str(rig.port), "--model", wrong, env_extra=tools.env())
            self.assertEqual(code, 1, out)
            self.assertIn("no chat model called " + wrong, out)
            self.assertIn("text-a-colibri", out)
        self.assertFalse(os.path.exists(tools.out), "the tool must not start on a model the server does not have")

    def test_opencode_gets_its_configuration_in_the_environment(self):
        rig = Rig(self, [entry("text-a"), entry("text-b")], api_key="sekrit")
        tools = FakeTools(self)
        code, out = coli("launch", "opencode", "--port", str(rig.port), "--api-key", "sekrit", env_extra=tools.env())
        self.assertEqual(code, 0, out)
        seen = tools.seen()
        self.assertEqual(seen["args"], ["--model", "colibri/text-a-colibri"])
        config = json.loads(seen["env"]["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual(config["provider"]["colibri"]["options"]["baseURL"], f"http://127.0.0.1:{rig.port}/v1")
        self.assertEqual(list(config["provider"]["colibri"]["models"]), ["text-a-colibri", "text-b-colibri"])

    def test_the_tools_exit_code_is_the_commands(self):
        rig = Rig(self, [entry("text-a")])
        tools = FakeTools(self)
        code, out = coli("launch", "claude", "--port", str(rig.port), env_extra=tools.env(FAKE_TOOL_EXIT="3"))
        self.assertEqual(code, 3, out)

    def test_dry_run_shows_the_plan_hides_the_key_and_starts_nothing(self):
        rig = Rig(self, [entry("text-a")], api_key="sekrit")
        tools = FakeTools(self)
        code, out = coli("launch", "claude", "--port", str(rig.port), "--api-key", "sekrit", "--dry-run",
                         env_extra=tools.env())
        self.assertEqual(code, 0, out)
        self.assertIn("ANTHROPIC_BASE_URL=http://127.0.0.1:" + str(rig.port), out)
        self.assertIn("ANTHROPIC_AUTH_TOKEN=***", out)
        self.assertNotIn("sekrit", out)
        self.assertIn("would run:", out)
        self.assertFalse(os.path.exists(tools.out))

    def test_a_server_that_wants_a_key_is_named_as_such(self):
        rig = Rig(self, [entry("text-a")], api_key="sekrit")
        tools = FakeTools(self)
        code, out = coli("launch", "claude", "--port", str(rig.port), env_extra=tools.env())
        self.assertEqual(code, 1)
        self.assertIn("API key", out)
        self.assertFalse(os.path.exists(tools.out))

    def test_no_server_no_chat_model_no_such_tool(self):
        tools = FakeTools(self)
        code, out = coli("launch", "claude", "--port", "1", env_extra=tools.env())
        self.assertEqual((code, "no server answers" in out), (1, True), out)
        rig = Rig(self, [entry("pic", modality="image")])
        code, out = coli("launch", "claude", "--port", str(rig.port), env_extra=tools.env())
        self.assertEqual(code, 1)
        self.assertIn("no chat model", out)
        code, out = coli("launch", "codex", "--port", str(rig.port), env_extra=tools.env())
        self.assertEqual(code, 1)
        self.assertIn("no tool called codex", out)

    def test_a_tool_that_is_not_installed_is_said_so_with_how_to_get_it(self):
        rig = Rig(self, [entry("text-a")])
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        code, out = coli("launch", "opencode", "--port", str(rig.port), env_extra={"PATH": empty.name})
        self.assertEqual(code, 1, out)
        self.assertIn("not installed", out)
        self.assertIn("npm install -g opencode-ai", out)


if __name__ == "__main__":
    unittest.main()
