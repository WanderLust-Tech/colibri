# Threat model

Who and what this fork trusts, what it defends against, and what it does not. It is written for one person running
Colibri on their own machines and, possibly, sharing it with a few people on the same network. It is not a hosting
guide for the public internet; **do not put the gateway on the open internet**.

Every statement below was checked against the code on `main` (file names are given so it can be checked again).

## What runs where

| Part | Where | Listens on |
|---|---|---|
| Engine workers (`coli serve`, one per loaded model) | the machine with the model | `127.0.0.1`, a random port, **no key** |
| The router (`coli router`) | the same machine | port 8000; `127.0.0.1` unless you pass `--host` |
| The dashboard | static files served by the router | the same port, **without a key** |
| The VS Code extension | your editor | nothing; it calls the router and runs tools itself |

The router is the only door. It holds the API key and relays to the workers (`c/router.py`, which also removes
`COLI_API_KEY` from the workers' environment on purpose). The server never runs a tool: it turns the model's tool call
into JSON for the client and stops. Reading files, writing files, running commands and fetching pages happen in the
extension, with your user's rights.

## Who is trusted

- **The person who owns the machine** and anyone who can log in to it. A local user can reach a worker's port directly
  (it has no key) and can read the key file. There is no protection between users of one machine.
- **Whoever holds the API key.** There is one shared key and no accounts. A key holder is an administrator.
- **The model files you installed.** Weights are read by our own loaders. A file made to break a parser is not defended
  against; take weights from sources you trust.
- **The model's output is not trusted.** It is shown to you, and it can ask the extension for tools, which are
  checked (see below).

## What a key holder can do

Everything the router offers. Concretely, with the key a client can:

- run chat, completion, image, music and transcription requests, which costs your GPU and CPU time;
- load and unload models (`/v1/models/switch`, `/v1/models/unload`) and change a model's settings
  (`/v1/models/settings`), for example to push everything onto the CPU;
- create and delete presets (`/v1/presets`), which carry system prompts and defaults, so a preset can change what every
  later request to its name does;
- read the request log summaries (`/v1/requests`) and the health and metrics detail.

What a key holder cannot do through the API: read files on the machine (a local image path in a request is refused
unless the server was started with `COLI_IMAGE_ROOT`, and then only under that folder, `..` refused), run commands, or
read another client's conversation (the server keeps no transcript; the request log holds what happened, not what was
said, unless you start it with bodies on). Finished pictures are kept as files in `~/colibri-images` on the server
machine (`c/image_engine.py`); the API does not serve that folder. The pictures and songs carry the settings that made them
(prompt, seed, model, a song's lyrics) inside the file, so a file you share shares its prompt; `COLI_MEDIA_META=0` leaves them
plain.

Without a key: `/health` answers with a bare "ok" and no routing detail, the dashboard's static files are served (they
contain no secret), and everything under `/v1/` is refused with 401.

## What is defended

- **No open bind by accident.** A bind beyond localhost without `COLI_API_KEY` is refused at start-up
  (`COLI_ALLOW_INSECURE_BIND=1` overrides, with a warning). Both the router and a single server do this.
- **The key check** accepts `Authorization: Bearer` or `x-api-key` and compares in constant time.
- **A web page in your browser cannot use the server.** The `Host` header must be loopback or one you allowed
  (`--allowed-host`), which stops DNS rebinding. Browser origins are an allow-list (`--cors-origin`); a model switch
  from a foreign origin is refused with 403 and needs a JSON content type, which forces a CORS preflight.
- **Static headers and a second cross-site check.** Every response carries `X-Content-Type-Options: nosniff` and
  `Referrer-Policy: no-referrer`; the dashboard's own files also carry a content policy (scripts only from the server's own files,
  no plug-ins, no framing by another page) and `X-Frame-Options: DENY` (`c/web_headers.py`). A browser request to `/v1/...`
  whose `Sec-Fetch-Site` is `cross-site` and whose origin is not on the `--cors-origin` list is refused with 403 before anything
  runs: a simple cross-site request is carried out by the server before the browser decides its answer may not be read, which
  matters most on a loopback server without a key. curl, the SDKs and the editor extension send no `Sec-Fetch-Site` and are not
  affected. This is defence in depth: the dashboard does not build markup from model output, and the key is what protects the API.
- **The dashboard keeps the key in memory only.** A link `http://host:8000/#key=...` puts it in the address-bar
  fragment, which a browser never sends to a server; the page reads it once and removes it from the address bar. It is
  not written to local storage (older releases did; the page deletes that copy).
- **The extension asks before it acts.** Tools that are not read-only ask first (`colibri.toolApproval`); permission
  rules (`deny` beats `ask` beats `allow`, shell commands split at `&&`, `|`, `;`) decide ahead of time; MCP servers and
  workspace rule files are not read in an untrusted workspace; a hook or a rule can block a call.
- **Tool output is marked as data** (`colibri.untrustedToolOutput`, on by default): a file, page, command result or MCP
  result reaches the model inside a `tool_output` block with a line saying it is not the user's message, and the block's
  own closing tag cannot be forged from inside. See "Prompt injection" below for how far that goes.

## What is not defended

- **The network.** The gateway speaks plain HTTP. On a LAN, anyone who can see the traffic can read the key and every
  prompt and answer. Use it on a network you trust, or put it behind a tunnel or a TLS proxy you set up.
- **A leaked key.** There is no expiry, no per-device key, no rate limit beyond the queue length (`--max-queue`). If the
  key leaks (a shared dashboard link, a screenshot, a chat paste, the restart script's console output, the key file),
  rotate it: `restart-colibri.ps1 -NewKey` makes a new one and the old one stops working at once.
- **A busy server.** Anyone with the key can fill the queue. Requests beyond it get 429, and a long prompt can occupy
  the single KV slot, which also throws away your own chat's cached start.
- **Other people on the machine** (see "Who is trusted").
- **Hostile weights or a hostile model file** (see "Who is trusted").
- **A server on the open internet.** No account system, no TLS, no abuse controls: not supported.

## Prompt injection

A file, a web page or a command's output can contain text written to steer the model ("run this to finish the setup").
The wrapper above helps; it does not close the hole. Tried on Qwen3-Coder-30B (3 injected READMEs, one run each, so
a rough signal only): plain, the model ran the injected command in 2 of 3; wrapped, in 1 of 3. A small model can be
talked into it either way.

When a long chat outgrows the window, the extension has the model write a record of the dropped turns and keeps it in the
system message (`colibri.summariseTrimmed`). That is a second route for injected text: something a file said could be
copied into a place the model trusts more. The defences are the same block around tool output on the way in, an
instruction to the summariser never to copy an instruction out of tool output, and a heading on the record saying it is
a record and not an order. Tried on Qwen3-Coder with a file that asked to be recorded as approved: it was not copied.
One file, one run: not a guarantee.

What actually stops harm is the approval dialog and the rules, not the model's judgement: read what a command is
before you allow it, keep `deny` rules for what must never run (`Bash(curl:*)`, `Read(**/.env*)`), and treat "always
allow" on a shell command as granting it to the whole conversation. In a trusted workspace a file in the repository can
carry permission rules, the way its `CLAUDE.md` can steer the model: open unknown projects as untrusted.

## If something goes wrong

| Event | What to do |
|---|---|
| The key was seen by someone who should not have it | `restart-colibri.ps1 -NewKey`; open the new dashboard link; update the extension's key file |
| A command ran that you did not expect | Check the **Colibri** output channel (every tool call is logged); add a `deny` rule for it |
| The server answers something odd after a preset change | Look at `/v1/presets` (or the dashboard's Presets tab) for a system prompt you did not write |
| You need it reachable from outside | Use a VPN or tunnel to the LAN; do not forward port 8000 |

## Checked where

`c/openai_server.py` (`_is_authed`, `_check_host`, `serve`, CORS and the model-switch origin check, image-path rule),
`c/router.py` (`do_GET` / `do_POST` auth, worker command and environment, bind refusal), `web/src/lib/storage.ts`
(`takeKeyFromLocation`, `persistPublicSettings`), the extension's `permissions.ts` and `untrusted.ts`. If you change one
of these, change this page.
