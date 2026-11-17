# How to use the new dashboard features

What was added to the dashboard (the page at `http://<this-pc>:8000/`) and to the gateway behind it, and how to use each
piece. Each section starts with what you want to do.

The dashboard talks to the router (`coli router`, which `coli service` and the tray icon run). Everything below works
against it. Where a feature needs a model or engine that supports it, the section says so.

- [Find out why a reply was slow](#find-out-why-a-reply-was-slow)
- [Understand an error message](#understand-an-error-message)
- [Edit a message, branch a chat, or rewrite a reply](#edit-a-message-branch-a-chat-or-rewrite-a-reply)
- [Compare models on the same prompt](#compare-models-on-the-same-prompt)
- [Read the cached-token count from the API](#read-the-cached-token-count-from-the-api)
- [If the key leaks, or you want to share the server](#if-the-key-leaks-or-you-want-to-share-the-server)

## Find out why a reply was slow

Under every assistant reply there is a small grey line:

```
qwen3-coder-30b-a3b · 266 of 279 prompt tokens cached · 18 tokens · 18.0 tok/s · first token 1.2 s · 4.1 s · 0.4% of 65,536 context
```

Read it left to right:

| Part | Meaning |
|---|---|
| model name | The model that really answered. The router can answer for a preset, so this is the real one. |
| `via preset NAME` | Appears after the model name when you picked a preset: the preset's name, so you can see why that model answered. |
| `266 of 279 prompt tokens cached` | Of the 279 tokens sent, the engine did not read 266 again. The start of the prompt matched the last request, so only the new tail was read. |
| `204 prompt tokens, none cached` | The engine read the whole prompt. This is the usual cause of a slow reply: a long prompt read from scratch. |
| `279 prompt tokens` | The server did not say how much was cached (an engine that does not report it). This does not mean none was. |
| `18 tokens`, `18.0 tok/s` | How many tokens were written, and the speed after the first one. |
| `first token 1.2 s` | Time until the first token. **If the model had to load first, the load time is in here**, so the first reply after a model unloads shows a long first token. |
| `4.1 s` | Time for the whole reply. |
| `0.4% of 65,536 context` | The prompt plus the reply as a share of the model's window. Left out when the window is not known. |

Hover the line for the same explanation in the page.

**What to do with it.**

- A reply that is slow with `none cached`: the prompt start changed. The usual reasons are an edit to an earlier message,
  a changed system prompt, a different model, or another request that used the same cache slot in between. A chat that
  only grows at the end should show a large cached figure on every reply after the first.
- A reply with `first token` of minutes on the first message after a break: the model was unloaded and loaded again. The
  router keeps a model for 30 minutes after its last use by default (`restart-colibri.ps1 -KeepAlive`).
- The cached figure is reported by the Qwen3-Coder (`qwen36`), Gemma 4 and DeepSeek V4 engines. Others leave it out and
  the line shows only the prompt size.
- The Qwen engine reuses its cache only when the new prompt starts with everything it holds; the Gemma engine reuses the
  part the two prompts share. So a branch or an edit can be cheap on Gemma and a full re-read on Qwen. The line shows
  which happened.

## Understand an error message

When a request fails, the red box above the message field now has two parts: a sentence in plain words, and below it,
smaller, the server's own message (always kept, so nothing is hidden by a wrong guess).

| You see | What happened | What to do |
|---|---|---|
| "This chat is longer than the model's window (40,111 tokens against 32,768)…" | The conversation no longer fits. | Press **New chat** (the button in the box), or restart with a larger window: `restart-colibri.ps1 -Ctx` for the Qwen models, `-GemmaCtx` for Gemma. |
| "The server did not accept the API key…" | Wrong or missing key. | Open the dashboard from the link the restart script prints (it carries the key as `#key=`), or paste the key under Settings. |
| "The dashboard cannot reach the server…" | The server is down or the endpoint is wrong. | Check the service is running (`coli service status`) and the endpoint under Settings → Connection. |
| "The engine is busy with other requests…" | The queue is full (429). | Wait and send again. |
| "The server is shutting down or restarting…" | A restart is in progress. | Try again in a moment. |
| "The server has no model called X…" | The model name is not served. | Pick one from the model list. |
| "This model does not do chat…" | An image, music or decision model was chosen for chat. | Pick a language model. |
| "The engine stopped or failed on this request…" | The engine reported a failure. | Unload the model (Settings → Model) and send again; it loads fresh. |
| "The connection closed before the reply was finished…" | The stream stopped without the server's finish marker: the engine died or the connection dropped. In the chat the part that arrived stays and **Continue** is offered; in Compare the column shows this and cannot be voted for. | Send again; if it repeats, unload the model (Settings → Model) and let it load fresh. |
| "The machine ran out of memory…" | Not enough memory to load or run. | Unload a model you are not using, or close other programs. |
| "The GPU stopped answering…" | The graphics card dropped out; the model's experts now run on the CPU. | Replies are slower until the model is unloaded and loaded again. (This message only reaches the page if a server forwards the engine's log line.) |

An error the page does not recognise is shown exactly as before.

## Edit a message, branch a chat, or rewrite a reply

Three buttons were added to the messages.

**Edit a message you sent.** A pencil sits under each of your messages.

1. Press it. The chat is cut at that message, and its text (and any pictures) go back into the message box.
2. Change the text and send.

The conversation you had goes to **Conversations** (the clock button in the left rail) first, so nothing is lost. Note:
Conversations are kept for this open page only; reloading the page clears them. Use **Export** (the download button at
the top right) to keep one.

*Cost:* everything after the edited message leaves the prompt, so the model reads from there again. Editing the last
message is cheap; editing the first one rereads the lot.

**Branch from a reply.** A branch button sits under every finished reply.

1. Press it. A new chat starts with that reply and everything before it. Your previous chat goes to Conversations.
2. Carry on from there in a different direction.

**Rewrite the last reply.** Under the **last** reply there are two more buttons: **rewrite shorter** and **explain
simpler**. They send a normal follow-up message ("Rewrite your last answer shorter…"). Because they only add to the end,
the conversation before them stays cached, so they are the cheap way to change a reply. You will see the message you
"sent" in the chat.

Deleting a single message is not offered: it would break the order of the conversation and the cache.

These buttons are hidden while a reply is being written, and on picture and music turns.

## Compare models on the same prompt

Use this to see how two to four models answer the same question, and to keep a score of which you prefer.

1. Open the navigation dock: hover or click the small handle at the top centre of the page, then choose **Compare**.
2. Under **Models to compare**, choose two to four models. Presets are listed too (marked "preset"), which lets you
   compare a preset against the plain model. Image and music models are not offered.
3. Type the prompt. Press **Compare** (or Ctrl+Enter).
4. The models answer **one after another**. Each column says "Waiting for its turn…", then "Loading the model and
   thinking… 12 s" until its first token, then fills in. Each column ends with the same stats line as in chat.
5. Press **This one is best** under the answer you prefer.

**Blind mode** (on by default) shows the columns as "Answer A", "Answer B"… in random order, with no model name in the
heading or the stats. Picking a best answer, or pressing **Show the model names**, reveals them. Untick it to see the
names from the start.

**The scoreboard** appears after your first vote: for each model, how many rounds it was in, how often you picked it, and
a percentage. It is kept in this browser (not on the server) and survives a reload; the circular arrow clears it. A round
only counts for the models that answered, so one that failed is not marked down.

Good to know:

- The models run one at a time, so a model that is not loaded is loaded first. Expect the first token of each to take a
  while; the time shown includes that.
- Compare uses the chat's cache slot, so it can cost your open chat its cached start.
- Temperature, reply length and the reasoning switch come from Settings. A preset uses its own temperature and length.
- A model that fails (queue full, out of memory) shows its plain-words reason in its column and the others still run.
- **Stop** ends the round; the model that was answering and those waiting are marked stopped.

## Read the cached-token count from the API

The gateway now reports how much of the prompt the engine did not read again, in the standard OpenAI place:

```json
"usage": {
  "prompt_tokens": 279,
  "completion_tokens": 18,
  "total_tokens": 297,
  "prompt_tokens_details": { "cached_tokens": 266 }
}
```

- `cached_tokens` is left out when the engine does not say (it is never a made-up `0`); `0` means the engine said none.
- It never exceeds `prompt_tokens`.
- The request log (`--request-log`) records the same figure as `cached_tokens`; see [request-log.md](request-log.md).
- The details of the engine protocol are in [serve_protocol.md](serve_protocol.md) and [api.md](api.md).

This is what the reply line and the extension's `cache:` log line are built on.

## Pages that are only there with a router

Settings shows Presets, Requests and Log only when the server is a `coli router`: against a plain `coli serve` they would only say
there is nothing to show. A router started without a request log or a presets file keeps those pages, and they say how to turn
them on. A server that predates the feature list keeps every page.

## See why a model is slow to load, or failed to start

**What it is.** The router prints what it and its models do (a model loading, a start that failed, the GPU giving way
to the CPU), but you cannot see that console from the dashboard or from an editor. **Settings → Log** shows the router's
last lines, and reads again every two seconds.

**What you do.** Open Settings → Log. Each line has the time, who printed it (`router`, or the model's id) and the text; lines
that mention a failure are tinted. **From** limits it to one model. **Pause** stops reading (what is printed meanwhile is
fetched when you resume), **Clear** empties the view and only newer lines come back. The view follows the newest line until
you scroll up.

**What to know.**
- The router keeps the last 2,000 lines, in memory only. A restart empties the log, and a page that is open for hours
  of heavy output can miss some: it then says "Some lines were dropped".
- It needs the API key, like the rest of the page. A plain `coli serve` has no log to read; the page says so.
- A model that failed to start is also named, with the last lines it printed, in the error the dashboard shows
  ("did not start: ...") and in Settings → Model as `last_error`.

## If the key leaks, or you want to share the server

Read [THREAT_MODEL.md](THREAT_MODEL.md). In short:

- The gateway speaks plain HTTP. On a LAN, anyone who can see the traffic can read the key and the conversations. Share it
  only on a network you trust, and never forward port 8000 to the internet.
- If the key leaks, run `restart-colibri.ps1 -NewKey`. The old key stops working at once. Open the new dashboard link and
  update the extension's key file.
- A key holder is an administrator: they can load and unload models, change model settings and edit presets.
