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
  request for another model waits until the running ones finish, in order, up to `--max-queue` waiting
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
The dashboard's Settings → Model has the same as an Unload button on each loaded model.

## What the API shows

- `GET /v1/models` lists every model that can be asked for, each with `"state": "loaded"` or `"idle"`. A
  loaded one carries the limits its engine announced; one that is not loaded carries what it announced the
  last time it ran (kept in `router.json`), or, if it never ran here, what its family can do.
- `GET /v1/models/installed` lists every model on the machine with its `state` (`idle`, `starting`,
  `ready`, `stopping`), its memory use, `keep_alive`, and how long until it is unloaded (`expires_in`). A
  model that failed to load has `last_error`.
- `POST /v1/models/switch` with a model's `key` (or its `model` name) loads it, as the dashboard's Load
  button does, and answers when it is ready. It does not stop the other models; the memory check does that
  when it must. Add `keep_alive` to preload with that keep-alive, or to change it on a model that is
  already loaded.
- `POST /v1/models/unload` with `key` or `model` stops a model and leaves the others and the server as
  they are: `200` `{"status": "unloaded"}` (or `"not loaded"`). A model that is answering a request is not
  cut off: it answers `202` `"unloading when idle"` and the model goes when the request has finished.
- `GET /health` is the loaded model's own health plus a `router` block: the workers and their states,
  `max_loaded`, `keep_alive`, the queue, and the free memory. `GET /health?model=ID` is that model's.
- Errors have the workers' shape. A model that does not start answers `503` with `model_load_failed` and
  the last lines it printed.

## Limits

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
