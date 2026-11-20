# Several models behind one port: `coli router`

`coli serve` loads one model per process. `coli router` puts a front door in front of as many as you
have installed: it listens on one port, reads the `model` of each request, starts that model when it is
first asked for, and stops it again when it has been idle for a while.

```
coli router --host 0.0.0.0 --api-key "$COLI_API_KEY"
```

Clients and the dashboard see one server. `POST /v1/chat/completions` with `"model": "qwen3-coder-30b"`,
`POST /v1/images/generations` with `"model": "qwen-image-2.1"` and `POST /v1/audio/music` with
`"model": "ace-step-1.5"` all go to the same address, and each is answered by its own model without a
restart in between.

Two things build on the router: a model can be offered under a name of its own with settings baked in (`coli preset`,
[presets.md](presets.md)), and every request can be logged to a file, the router's line joined to its worker's
([request-log.md](request-log.md)).

## How it works

The router is a separate process. Each model runs in an ordinary `coli serve` on a loopback port (a
worker), because the gateway keeps one model's chat template and engine in process-wide settings. The
router

- **checks the API key once**, and the Host header and cross-origin rules, then forwards the request.
  Workers listen on `127.0.0.1` only and are not given the key;
- **finds the model**: the request's `model` may be the model's id (`qwen-image-2.1-colibri`), the id
  without `-colibri`, the folder name, or the key `GET /v1/models/installed` gives. The worker is sent
  its own id. A request with no `model` goes to the only model when there is only one;
- **loads on demand**: the first request for a model starts its worker and waits until it answers. A
  second request for the same model while it loads waits for the same load;
- **makes room**: before a load it compares the model's size with the free memory (the size it took last
  time, else its size on disk). If it does not fit in 80% of the free memory, or `--max-loaded` models
  are already loaded, it stops the least recently used idle one, and repeats. A model with a request
  running is never stopped. A model that does not fit even with nothing else loaded is started anyway;
- **unloads when idle**: a model with no request for `--keep-alive` is stopped. Its engine is asked to
  finish first, so it can save what it learned;
- **lets one model compute at a time**. colibri's models share the CPU, so two renders at once each run
  several times slower. Requests for the same model still run together (its own queue decides), but a
  request for another model waits until the running ones finish, in order (a chat goes ahead of waiting pictures and songs, see "Priority"), up to `--max-queue` waiting
  and `--queue-timeout` seconds, with the same `429` the workers use. `--parallel-models` turns this
  off;
- **streams**: server-sent events and progress lines reach the client as they are made. A client that
  hangs up cancels the worker's request.

The design follows Ollama's scheduler (`server/sched.go`: load on demand, a memory check with headroom,
least recently used eviction, keep-alive), with a router in front of per-model processes where Ollama
has runners.

## Options

| Option | Default | What it does |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8000` | Where it listens. A host other than loopback needs `--api-key` (or `COLI_API_KEY`) |
| `--api-key` | `COLI_API_KEY` | The key clients send, as `Authorization: Bearer` or `x-api-key` |
| `--allowed-host`, `--cors-origin` | | As for `coli serve` |
| `--model DIR` | every installed model | Offer only this model directory (repeat for several) |
| `--models-root DIR` | | A folder whose sub-folders are models, besides `coli setup`'s and `~/colibri-models` |
| `--keep-alive TIME` | `10m` (`COLI_KEEP_ALIVE`) | `30s`, `10m`, `2h`; `0` unloads as soon as idle; `-1` never |
| `--max-loaded N` | `3` (`COLI_MAX_LOADED_MODELS`) | Most models loaded at once; `0` for no limit. Memory decides first |
| `--parallel-models` | off | Let two models compute at the same time |
| `--preload MODEL` | | Load this model at start (repeat for several) |
| `--max-queue`, `--queue-timeout` | `8`, `300` | Requests waiting for another model's turn |
| `--kv-slots N` | `1` (`COLI_KV_SLOTS`) | KV slots for each chat model, so several conversations keep their cache (see [api.md](api.md#isolated-kv-contexts)). A model gets no more than its family allows: picture and music models always have one |
| `--worker-arg ARG` | | An extra argument for every `coli serve` the router starts (repeatable) |
| `--gpu off\|auto\|on` | unset (`COLI_GPU`) | Run each model on the graphics card or on the CPU, decided when it loads: `auto` looks at the card's free memory. Unset leaves `COLI_VULKAN` as the environment has it. See [gpu-offload.md](gpu-offload.md) |
| `--gpu-reserve-gb`, `--gpu-min-gb` | `6`, `3` (`COLI_GPU_RESERVE_GB`, `COLI_GPU_MIN_GB`) | With `--gpu`: card memory kept free for everything else, and the least beyond it worth starting a model on the card for |
| `--model-settings FILE` | `model-settings.json` in the setup folder (`COLI_MODEL_SETTINGS`) | Models turned off, and models pinned to the card or the CPU; see [model-settings.md](model-settings.md) |

What the router learns is kept in `router.json` in the setup folder (`%LOCALAPPDATA%\colibri` on
Windows): how much memory each model really took the last time it was loaded.

## Priority

A picture or a song takes minutes, sometimes an hour; a chat takes seconds. While the gate lets only one model compute,
a chat for another model has to wait for the render in front of it, and, in a first-come queue, for every render queued
before the chat. So the queue has a priority: **a chat that arrives while renders are waiting goes ahead of them.**
Nothing is interrupted: a request that is already running always finishes, and requests of the same priority keep their
order.

| Request | Priority |
|---|---|
| `/v1/images/generations`, `/v1/images/edits`, `/v1/audio/music` | low |
| everything else (chat, completions, messages, transcriptions, embeddings, System One) | normal |

A client can choose with the header `X-Colibri-Priority: low`, `normal` or `high`; any other value is ignored (the
default applies), since priority is a hint and never a reason to refuse a request. Two things to know: a low-priority
request that keeps being overtaken still gives up after `--queue-timeout`, so a long run of chats can starve a waiting
render into a `429`; and the priority only orders what is *waiting*, so a chat still waits for the render that is
*running* (stop it with a client disconnect, or see `coli stop`).

## A model that will not start on the card

When the router has a model start on the graphics card (`--gpu auto`, or `on`, or a pin) and the start fails (the process
exits before it answers: a driver that gives up, a device that was lost, memory another program took), the router starts it
**once more on the CPU** instead of answering 503, and logs both. If that works, the next starts of the same model go straight
to the CPU for **30 minutes** (a card that failed once will most likely fail again, and each failed start is a wait), after which
the card is tried again. The model's `gpu` in `GET /health` and `GET /v1/models/settings` then says
`"mode": "cpu"` with `"why": "the card failed to start it: ..."`, and the Log page has the lines. A model that **exits on its own while it
runs on the card** is treated the same way for its next start.

It is not done where someone chose the card: a model pinned to `vulkan` (`coli model gpu NAME vulkan`) or a router started with
`--gpu on` reports the failure as before. And a model that fails on the CPU as well is not blamed on the card: the answer is
the 503 of the second try, and nothing is remembered. The 30 minutes are `CARD_FALLBACK_S` in `c/router.py`.

Qwen-Image's decoder already falls back on its own: its Vulkan VAE decode works in bands and returns to the CPU when the device
is lost or refuses its weights (`c/qwenimage_vae_vk.h`), so a picture that gets as far as decoding is not lost to that.

## Models that are not there

A model the router has seen is remembered (`known-models.json` beside `router.json`). When one is no longer found it stays on the
list, switched off, with the reason: **the drive E: is not connected** (a Windows drive that is not there, or an empty mount point),
or **its files are no longer recognised as a model** (the folder is there and something in it was deleted or changed; `coli model
verify` says what). It is not offered by `/v1/models`, a request for it is the usual 409 "cannot be started here" with that reason, and
it is an ordinary row again when the folder is back. A folder that was deleted from a place that is there is forgotten.

A download that did not finish is listed too, with how many of its files are there (`download incomplete (3 of 5 files here); run
`coli setup` to finish it`), and cannot be loaded. If a complete copy of the same model sits beside an unfinished one, a request for the
model finds the complete one.

## Which model goes first, and when memory runs low

When the router has to stop a model, to make room for another or to keep under `--max-loaded`, it no longer takes simply the one
used least recently. Each idle model gets a score: how long it has sat idle, times the memory stopping it frees, divided by what it
costs to load again, and the highest goes. The cost is measured: the router times every start (`ready in 42 s`), keeps the slowest of
the last three in its state file (`router.json`) and shows it as `reload_seconds` in `/health`, so a model on the slow D: drive, which
takes minutes to come back, stays loaded longer than one on fast storage with the same size and idle time. When memory is what is
short, a model that frees enough by itself is preferred to a string of small stops. A model never timed is guessed from its size.

**Memory that runs low between requests.** The reaper, which looked only at the keep-alive timer, now also reads the free memory each
second. Under a floor (the larger of 1.5 GB and 3% of the total; `COLI_MEMORY_FLOOR_GB` sets it, 0 turns it off) it stops one idle
model, the best by the score above, preferring one whose stop frees past the shortfall by half the floor. Then it waits 15 seconds for the
memory to show up as free before it would stop another, so one spike from a browser or a game does not empty the router. A model that is
answering is never touched, and one used in the last 20 seconds is left alone unless the free memory is under half the floor. Each
stop says why in the Log page, and `/health` shows `router.memory_pressure`: the floor, how many stops, and the last one.

**What a model grows to.** The router measured a model only right after it loaded, but it grows as its caches fill. Every five seconds
it now reads the memory of each model that is answering and remembers the most it saw (`peak_bytes`, only raised); the next "will it
fit" uses that, not the smaller figure from just after the load.

**An idle model that holds the graphics card.** With `--gpu auto`, a model that would go to the CPU only because the card's memory is
taken now makes the router stop an idle model that holds the card first, if it has been idle for a minute (so two models that both want
the card do not trade it back and forth) and is not pinned to either side. It waits for the card to show its memory free, then starts
the new model on it.

It tries to keep the model that holds the card loaded: it asks the idle worker to give the card back (`POST /v1/demote`, `DEMOTE` to the
engine), which makes the engine drop everything it had on the card, read back into RAM the dense weights only the card held, and carry on on
the CPU. That takes a few seconds and the model keeps its conversation cache; a restart would read it all from the drive again, minutes
from a slow one. Only the `qwen36` engine (the Qwen3, Qwen3.5 and Qwen3.6 models, Qwen3-Coder among them) can. For any other, or when it
answers that it cannot, the model is stopped as before and starts again when it is next asked for. A demoted model stays on the CPU until
it is stopped (the idle timer, `coli model`, or a restart): in a short test of Qwen3-Coder here it answered at about 3 tokens a second, the
same as one started on the CPU, against about 18 on the card. `/health` marks it (`demoted`, and `gpu` says why it is on the CPU).
`COLI_CARD_DEMOTE=0` makes the router stop such a model instead; `COLI_CARD_EVICT=0` turns the whole thing off.

## Background jobs

`Prefer: respond-async` on a picture or song request makes it a job: the router answers `202` with an id and makes the render apart from the
connection, so a closed tab no longer cancels an hour of work ([api](api.md#background-jobs)). One job runs at a time, in the same low-priority
lane as any other render, so a chat that arrives goes ahead of the jobs that are waiting (never of the one that is running). The router holds its
own connection to the model for the whole job; cancelling closes it, which is how the model learns to stop. The records are in `jobs.jsonl`
beside the router's other state (`COLI_JOBS`), and a job that was in flight when the router stopped is read back as failed.

## Keep-alive, per request

`--keep-alive` is the default. A request can say how long its model stays loaded after it, as in Ollama:
add `"keep_alive"` to the JSON body (or a `keep_alive` field to a multipart form) of any request that
names a model.

| `keep_alive` | Means |
|---|---|
| `"30s"`, `"10m"`, `"2h"` | that long after the last request |
| a number, `90` | that many seconds |
| `0` | unload as soon as the model is idle |
| `-1`, `"-1"`, `"forever"` | never unload |
| absent, or `null` | leave it as it is |

It sticks: a request that says nothing keeps what an earlier one set, until the model is unloaded (the
next load starts from `--keep-alive` again). The router reads it and does not pass it on to the worker. A
value that is not a duration is a `400` with `param: "keep_alive"`, and nothing is loaded for it.

```
curl localhost:8000/v1/models/switch -d '{"model": "qwen3-coder-30b", "keep_alive": "-1"}' \
     -H "Authorization: Bearer $COLI_API_KEY" -H "Content-Type: application/json"     # load it and keep it
```

## `coli ps` and `coli stop MODEL`

```
$ coli ps
  NAME                          STATE    MEMORY    RUNS ON   UNLOADS IN
  qwen-image-2.1-colibri        loaded   13.3 GB   CPU       9m 41s
  qwen3-coder-30b-a3b-colibri   loaded   2.4 GB    GPU       never

$ coli stop qwen-image-2.1          # unload one; the server and the other models stay
  ✓ unloaded qwen-image-2.1-colibri
```

Both take `--host`, `--port` (default 8000, or the port `coli setup` configured) and `--api-key`
(`COLI_API_KEY`). `UNLOADS IN` says `in use` while the model is answering and `never` for `keep_alive -1`. `RUNS ON` is `GPU` or
`CPU` when the router has a `--gpu` policy (or the model is pinned), and `-` when it does not decide.
Against a plain `coli serve`, `coli ps` names its one model and `coli stop MODEL` says it is not a router.
The dashboard's Settings → Model has the same as an Unload button on each loaded model, and, for each model, where it runs and a
switch that turns it off ([model-settings.md](model-settings.md#in-the-dashboard)).

## What the API shows

- `GET /v1/models` lists every model that can be asked for, each with `"state": "loaded"` or `"idle"`. A
  loaded one carries the limits its engine announced; one that is not loaded carries what it announced the
  last time it ran (kept in `router.json`), or, if it never ran here, what its family can do.
- `GET /v1/models/installed` lists every model on the machine with its `state` (`idle`, `starting`,
  `ready`, `stopping`), its memory use, `keep_alive`, and how long until it is unloaded (`expires_in`). A
  model that failed to load has `last_error`. Each also has `details` (what its folder says about itself: architecture, weights,
  the model's own context limit, where it came from), read from the small files only, so it is there before the model has ever been
  loaded; and `reload_seconds` once it has been loaded.
- `POST /v1/models/switch` with a model's `key` (or its `model` name) loads it, as the dashboard's Load
  button does, and answers when it is ready. It does not stop the other models; the memory check does that
  when it must. Add `keep_alive` to preload with that keep-alive, or to change it on a model that is
  already loaded.
- `POST /v1/models/unload` with `key` or `model` stops a model and leaves the others and the server as
  they are: `200` `{"status": "unloaded"}` (or `"not loaded"`). A model that is answering a request is not
  cut off: it answers `202` `"unloading when idle"` and the model goes when the request has finished.
- `GET /v1/models/catalog`, `POST /v1/models/install`, `POST /v1/models/uninstall` (and `GET /v1/models/installs`, `POST /v1/models/install/cancel`)
  fetch a model from the catalog into a models folder and remove one, while the router runs: [model install](model-install.md). A model that is
  loaded is never removed.
- `GET /v1/logs?since=N&limit=N&source=ID` returns what the router and the models it started printed, oldest first: a
  model loading, a start that failed, the GPU giving way to the CPU. Each line has a number that only grows (`seq`), a
  `time`, a `source` (`router`, or the model's id) and the `text`. Send the `next` of the last answer as `since` and you get only
  the new lines; `first` is the number of the oldest line still held, and when it is more than `since` + 1 lines were dropped
  between two reads. The router keeps the last 2,000 lines in memory only (nothing is written to disk), so a restart empties
  it. It needs the API key, since a worker's output can hold whatever its engine prints. The dashboard's Settings → Log and
  the extension's "Colibri: Show Server Log" read it.
- `GET /health` is the loaded model's own health plus a `router` block: the workers and their states,
  `max_loaded`, `keep_alive`, the queue, and the free memory. `GET /health?model=ID` is that model's.
- Errors have the workers' shape. A model that does not start answers `503` with `model_load_failed` and
  the last lines it printed.

## Limits

- **One context window per engine family, set by the router's environment.** The workers are started with the same
  arguments, so there is no per-model `--ctx`. Gemma 4 serves 8192 unless `CTX` is set in the router's environment (every Gemma
  gets it); the Qwen3.6 family reads `Q36_MAXT` (the restart script's `-Ctx`). Each model's card says what it serves
  (`context_length`); how to set it, and what it costs, is in [gemma4.md](gemma4.md#context-window).
- The memory check is only as good as its estimate. The first time a model loads, its size on disk is
  used; after that, what it measured. Models that size their cache from the free memory when they start
  (the large streamed ones) take most of what is free, so they push the others out.
- A load has no deadline (a 500 GB model takes a long while); waiting for memory or for another model's
  turn does.
- `coli stop` (no model name) stops the router and its workers, found by the pidfile the router writes;
  `coli stop MODEL` unloads one model only.

## Tests

`c/tests/test_router.py` starts real worker processes (`router_fake_worker.py`, which answers like
`coli serve` without a model) and checks routing, loading on demand, eviction and its order, the
one-model-at-a-time rule and the queue, keep-alive, the guards, streaming, a client that hangs up, a
worker that fails or dies, and the dashboard's switch endpoints.
