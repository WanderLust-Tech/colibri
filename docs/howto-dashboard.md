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

## Make a saved picture or song again

The server writes the prompt, size, steps, seed and model into every picture (a PNG text chunk) and the prompt, lyrics,
language, length, steps, seed and model into every song (a WAV comment), so a file says how it was made. In a picture or music
model's composer, the file button beside the send button ("Open the settings of a saved picture or song") takes such a file and
puts everything back: it switches to the file's model (or the first model of that kind, if this server has not got it),
fills the prompt and the controls, and leaves the rest as it was. Send it for the same render, or change the seed for a
variation. It is a button because dropping a picture in image mode already attaches it for editing.

## Editing a clip toward a new description

When the music engine does flow-edit (the composer then has an Edit mode next to Cover), choose "Edit", pick the clip, and fill two things: the line above the prompt says **what the clip is now** (its caption, in the
words you would use for a prompt; and, in the box below it, its lyrics if it has any), and the prompt says **what it should become**. The clip keeps its length and much of its character; the new description bends it. Two small
fields beside the chips set the part of the work that is an edit (0 to 1: a narrow window changes the clip lightly and leaves the rest to ordinary denoising) and the noise draws averaged a step (1 to 16: steadier, at that many times the work).
"Keep the structure" runs the edit on a cover's context, so the tune stays and the sound changes. A reference clip works here too. An edit takes the plain Euler step: the Sampler chip is not shown for it, the velocity cap and smoothing are.
Details: `docs/acestep.md`, "Flow-edit".

## The sampler of a song

When the music engine takes upstream's sampler options (the composer then shows them), a Sampler chip picks the step: Euler (the plain one), Heun (a second,
corrected look at every step: twice the time) or SDE (fresh noise at every step). Two small fields beside it cap a step's change at that many times the size of
the music so far (0 to 100) and smooth it with the last step's (0 to 1); empty means off. The song's line under the player says which was used, and "again"
keeps them. Details: `docs/acestep.md`, "The sampler options".

A file from somewhere else, or made by a server started with `COLI_MEDIA_META=0`, says "That file carries no Colibri settings"
and changes nothing. The file is read in the browser; nothing is uploaded. The settings travel with the file, so sharing it
shares the prompt; `COLI_MEDIA_META=0` on the server leaves them out.

## Keep a render going when you close the page

A picture can take an hour, and by default it is one request: close the tab, let the laptop sleep, and it is lost. On a `coli router`
the composer of a picture or music model has a cloud button, "Keep going if I leave". With it on, the router makes the picture or song
apart from your page: the card says "This keeps going if you close the page", shows how far it is (or "Waiting for its turn: number 2
in line" behind another render), and Stop cancels it. Close the page whenever you like; the finished file is kept on the server and
the render is in **Settings → Library**. The switch is remembered, and is offered only where the server says it has jobs.

What you give up: no half-finished previews while it works (the job follows progress, not pictures), and renders run one at a time, in
order, behind any chat that is already waiting. A server that is not a router, or that predates jobs, shows no button and makes it the
ordinary way.

## The Library: everything the server kept

**Settings → Library** (any server that has it) shows the pictures and songs in the server's kept folders, newest first, with the prompt,
the size or length, the seed and the time. Pictures load as they scroll into view; a song loads when you press Play (a song is megabytes).
"All", "Pictures" and "Songs" filter it, and "Show more" reads the next page. Each card has:

- **Make again**: puts that file's model, prompt, size or length, steps and seed back in the composer and takes you to the chat, the same as
  opening a file with the button beside the send button ([above](#make-a-saved-picture-or-song-again)), without the file picking. A file with
  no settings in it (a picture from somewhere else) has no such button.
- **Download**.

Above the files, on a router, the **Jobs** part lists the renders that are waiting or running, each with how far it got, and the ones that
failed with why. The cross on a row cancels it: a waiting job never runs, a running one is interrupted and the engine stops computing. It
asks the router every couple of seconds while the page is in front, so a render you started from another tab, or before you closed the
page, is there with its progress. When one finishes, its picture appears in the grid.

The Library reads the folders the server writes to (`COLI_IMAGE_DIR`, `COLI_MUSIC_DIR`); it cannot show a file a server was told not to
keep. It has no delete button: remove files from the folders. The browser, not the server, scales the pictures down for the grid.

## Pages that are only there with a router

Settings shows Presets, Voices, Requests and Log only when the server is a `coli router`: against a plain `coli serve` they would only say
there is nothing to show. A router started without a request log or a presets file keeps those pages, and they say how to turn
them on. A server that predates the feature list keeps every page.

## Talk to it, and hear it answer

With a speech recognition model and a speech model on the server (a router with `qwen3-asr-0.6b` and `qwen3-tts-0.6b`), the chat composer has
three more buttons: the **microphone** (speak, and it sends what you said when you pause, or when you press it again), **read replies aloud**,
and **hands-free** (listen again after each reply, a conversation with no keys; it stops after 12 seconds of silence or on Stop). Open the dashboard
as `http://localhost:8000`: a browser gives the microphone only to `localhost` and https pages, not to the PC's network address. The page's
**Settings → Voices** keeps the saved voices of the voice-cloning models: **New voice** makes one from an audio clip of 3 to 30 seconds (and, better,
what it says), **Try** speaks a phrase in it and plays it, **Read replies in this voice** makes it the voice of the read-aloud button, **Remove**
forgets it. Details: [voice-loop.md](voice-loop.md), [qwen3-tts.md](qwen3-tts.md).

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
