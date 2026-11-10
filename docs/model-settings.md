# Turning models off, and pinning them to the card or the CPU

`coli router` offers every model it finds. Two things can be set for a single model, and both survive a restart:

* **off.** The router does not load the model. A request for it is answered with a `403` (`model_disabled`) that says how to turn it
  on again, and the dashboard's and `/v1/models`'s list shows it as `disabled`. A model that is loaded when it is turned off is unloaded:
  at once when it is idle, or when it has finished the request it is answering (nothing is cut off). The presets built on it are not offered
  while it is off. Use it to keep a big model from loading by accident (the 31B takes 30 GB), or to stop a client from loading one.
* **where it runs.** `vulkan` or `cpu` pins the model to the graphics card or to the CPU whatever the router's `--gpu` policy says;
  `auto` (the default) follows the policy. See [gpu-offload.md](gpu-offload.md).

```
coli model                              list every model: on or off, pinned where, where it runs now, and the GPU policy
coli model disable gemma-4-31b          turn it off (and unload it)
coli model enable gemma-4-31b           turn it back on
coli model gpu qwen3-coder-30b cpu      always on the CPU       (vulkan: always the card; auto: follow --gpu)
```

A model is named as everywhere else: its id (`gemma-4-31b-colibri`), the id without `-colibri`, its key or its folder name. Pass
`--host`, `--port` and `--api-key` (or `COLI_API_KEY`) for a router that is not on 8000 or has a key. With no router running the file is
edited directly and the next router reads it.

## The file

`model-settings.json` in the setup folder (`%LOCALAPPDATA%\colibri` on Windows), or `COLI_MODEL_SETTINGS`, or
`coli router --model-settings FILE`:

```json
{
  "models": {
    "gemma-4-31b-colibri":         {"enabled": false},
    "qwen3-coder-30b-a3b-colibri": {"gpu": "cpu"}
  }
}
```

A record has `enabled` (`false`; anything else is on) and `gpu` (`vulkan`, `cpu`, `auto`). The router reads the file again whenever it
changes. A file that cannot be read keeps the last good settings and says so once in the router's log. An unknown key or a wrong value
is an error that names it.

## The API

All need the router's key, and a JSON `Content-Type`; a request from another origin is refused, like the other management calls.

| | |
|---|---|
| `GET /v1/models/settings` | every installed model: `id`, `key`, `enabled`, `gpu` (the pin), `state`, `running` (where a loaded one runs: `{"mode": "vulkan"\|"cpu"\|"unchanged", "why": ..., "free_gb": ...}`), plus `gpu_policy` (the router's `--gpu`, the reserve, the minimum, the card's free memory now) and `file` |
| `POST /v1/models/settings` | `{"model": NAME, "enabled": false, "gpu": "cpu"}`: `model` (or `key`), and `enabled` and/or `gpu`. Answers the model's settings, and `unload` (`unloaded`, `unloading when idle`) when it was turned off |

`GET /v1/models/installed` also carries `enabled` for each model, and `gpu_setting` when one is pinned.

Not done: a switch in the dashboard (the API and `coli model` are what it would call).
