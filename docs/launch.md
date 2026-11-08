# Coding tools on a local model: `coli launch`

`coli launch` starts a coding tool already pointed at a running colibri server (`coli serve` or `coli router`). It sets what
the tool needs in that one run's environment and starts it. Nothing is written to the tool's own settings, so the tool
still talks to its usual service the next time you start it by hand.

```
coli launch                                  # the tools it knows, and whether each is installed
coli launch claude                           # Claude Code on the model the server has loaded
coli launch claude --model qwen3-coder-30b-colibri
coli launch claude --port 8000 --api-key "$COLI_API_KEY" -- -p "explain this repo"
coli launch opencode --host 192.168.1.20
coli launch claude --dry-run                 # show the environment and the command, start nothing
```

`--host`, `--port` and `--api-key` (or `COLI_API_KEY`) name the server as for `coli ps`; the port is 8000, or the one
`coli setup` configured. Everything after `--` goes to the tool as it is.

The model is `--model`, else a model the server already has loaded (so nothing has to load first), else the first it lists. Only
chat models are offered: a picture or music model cannot drive a coding tool. Against `coli router` any installed chat model
works, and the router loads it when the first request arrives.

## The tools

| Tool | Talks to | How it is pointed at the server |
|---|---|---|
| `claude` (Claude Code) | `POST /v1/messages` | `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` (the server's key; a placeholder when it has none), and the model for every tier (`ANTHROPIC_DEFAULT_OPUS_MODEL`, `_SONNET_`, `_HAIKU_`, `CLAUDE_CODE_SUBAGENT_MODEL`) |
| `opencode` | `POST /v1/chat/completions` | the whole configuration in `OPENCODE_CONFIG_CONTENT`: one provider, `colibri`, listing every chat model the server has |

For Claude Code it also sets three things you would otherwise find out the hard way:

- `CLAUDE_CODE_ATTRIBUTION_HEADER=0`. Claude Code adds an attribution header to its requests (Ollama's launcher turns it off
  for the same reason). A header that changes from one request to the next at the start of the prompt would make every
  turn differ from the last in its first tokens, and the server's KV cache would be thrown away, which on a CPU means reading
  the whole prompt again. We did not measure whether it does; switching it off costs nothing.
- `CLAUDE_CODE_AUTO_MODE_SERVER=0` (unless you have set it), because the server-side auto-mode check is Anthropic's, and a
  local server has none; Claude Code's own check is used instead.
- `ENABLE_CLAUDEAI_MCP_SERVERS=false`, so claude.ai connectors are not offered to a local model.

## Not supported

- **Codex** speaks the Responses API (`POST /v1/responses`), which colibri does not serve.
- **Zed, Cline, VS Code** keep their server in a settings file you own. Launching them would mean editing that file, and
  `coli launch` edits nothing. Their settings take a base URL (`http://HOST:PORT/v1`), a key and a model name, all from
  `GET /v1/models`.

## What to expect

A coding tool sends a long system prompt and a list of tools with every request. On the CPU a first turn can take minutes
to read; later turns reuse the cache (`--kv-slots` above 1 keeps several conversations apart, see
[api.md](api.md#isolated-kv-contexts)). Claude Code assumes a 200,000-token window; give the server a context it can
serve (`CTX`), and use a model that does tool calls (`qwen3-coder-30b` is the one built for it).

The design follows Ollama's `ollama launch`. Both tools were checked against their documented settings and against this
server's endpoints with a stand-in program; neither was run for real on the machine this was written on.
