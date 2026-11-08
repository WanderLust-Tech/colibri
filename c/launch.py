"""`coli launch`: start a coding tool already pointed at a colibri server.

Nothing is written to the tool's own configuration. Each tool takes what it needs from its environment for this one run:

- Claude Code reads ANTHROPIC_BASE_URL and friends, and talks to POST /v1/messages.
- opencode reads its whole configuration from OPENCODE_CONFIG_CONTENT, and talks to POST /v1/chat/completions.

Left out on purpose: Codex (it speaks the Responses API, which colibri does not serve) and tools that keep their server in a
settings file the user owns (Zed, Cline, VS Code), which would mean editing that file.
"""
import json
import os
import shutil

TOOLS = {
    "claude": {"name": "Claude Code", "binary": "claude",
               "install": "https://docs.claude.com/en/docs/claude-code/setup"},
    "opencode": {"name": "opencode", "binary": "opencode", "install": "npm install -g opencode-ai"},
}

# What a client may send as a key when the server has none: the tools insist on a non-empty one.
PLACEHOLDER_KEY = "colibri"


def base_url(host, port):
    return f"http://{host}:{port}"


def chat_models(installed, models):
    """Ids of the models that answer chat, loaded ones first, from the two listings a server gives.

    `installed` is GET /v1/models/installed (a `coli router` answers it: every model on the machine, with its state and
    modality) and `models` is GET /v1/models (every server answers it: the served model, with its capabilities). Either may be
    None. The router's listing wins when it is there."""
    if isinstance(installed, dict) and installed.get("router"):
        rows = [m for m in installed.get("data") or [] if m.get("modality", "text") == "text" and m.get("id")]
        rows.sort(key=lambda m: 0 if m.get("state") == "ready" else 1)      # sort is stable: the rest keep their order
        return [m["id"] for m in rows]
    rows = (models or {}).get("data") or [] if isinstance(models, dict) else []
    return [m["id"] for m in rows if m.get("id") and "chat" in (m.get("capabilities") or ["chat"])]


def claude_env(base, model, api_key, environ=None):
    """The variables that send every model tier of Claude Code to this server.

    CLAUDE_CODE_ATTRIBUTION_HEADER=0 is there because an attribution header that changes between requests, at the start of
    the prompt, would make every turn differ from the last in its first tokens and so throw away the server's KV cache
    (not measured here; Ollama's launcher sets it for the same reason). The auto-mode check is Anthropic's server-side
    classifier; a local server has none, so Claude Code's own is used (unless the user already chose)."""
    environ = os.environ if environ is None else environ
    env = {
        "ANTHROPIC_BASE_URL": base,
        "ANTHROPIC_AUTH_TOKEN": api_key or PLACEHOLDER_KEY,
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    }
    if "CLAUDE_CODE_AUTO_MODE_SERVER" not in environ:
        env["CLAUDE_CODE_AUTO_MODE_SERVER"] = "0"
    return env


def opencode_config(base, model, models, api_key):
    """The JSON for OPENCODE_CONFIG_CONTENT: one provider, `colibri`, with every chat model the server has."""
    ids = list(models)
    if model not in ids:
        ids.insert(0, model)
    options = {"baseURL": base + "/v1"}
    if api_key:
        options["apiKey"] = api_key
    return json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "provider": {"colibri": {"npm": "@ai-sdk/openai-compatible", "name": "colibri", "options": options,
                                 "models": {i: {"name": i} for i in ids}}},
        "model": "colibri/" + model,
    }, separators=(",", ":"))


def find_tool(tool, which=shutil.which, home=None):
    """The path of the tool's program, or None."""
    spec = TOOLS[tool]
    found = which(spec["binary"])
    if found:
        return found
    home = home or os.path.expanduser("~")
    exe = spec["binary"] + (".exe" if os.name == "nt" else "")
    for candidate in (os.path.join(home, ".local", "bin", exe), os.path.join(home, ".claude", "local", exe)):
        if tool == "claude" and os.path.isfile(candidate):
            return candidate
    return None


def plan(tool, base, model, models, api_key, extra=(), environ=None):
    """(variables to add to the environment, arguments to give the tool) for one run."""
    if tool == "claude":
        return claude_env(base, model, api_key, environ), ["--model", model, *extra]
    if tool == "opencode":
        # an explicit --model wins over a model the user's own opencode configuration names for an agent
        return {"OPENCODE_CONFIG_CONTENT": opencode_config(base, model, models, api_key)}, \
            (list(extra) if extra else ["--model", "colibri/" + model])
    raise KeyError(tool)
