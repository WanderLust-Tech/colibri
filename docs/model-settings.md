# Turning models off, pinning them to the card or the CPU, and giving each its own context window

`coli router` offers every model it finds. Three things can be set for a single model, and all survive a restart:

* **off.** The router does not load the model. A request for it is answered with a `403` (`model_disabled`) that says how to turn it
  on again, and the dashboard's and `/v1/models`'s list shows it as `disabled`. A model that is loaded when it is turned off is unloaded:
  at once when it is idle, or when it has finished the request it is answering (nothing is cut off). The presets built on it are not offered
  while it is off. Use it to keep a big model from loading by accident (the 31B takes 30 GB), or to stop a client from loading one.
* **where it runs.** `vulkan` or `cpu` pins the model to the graphics card or to the CPU whatever the router's `--gpu` policy says;
  `auto` (the default) follows the policy. See [gpu-offload.md](gpu-offload.md).
* **its own context window.** How many tokens of conversation the model keeps in view. Without it a model gets its family's default, or what
  the router was started with for every model (`restart-colibri.ps1 -Ctx` and `-GemmaCtx` set `CTX` for all of them). With it, the router starts
  *that model's* worker with the number, in the variable its family reads (`CTX`, or `Q36_MAXT` for the Qwen3.6 family), whatever the router's own
  environment says: Gemma 4 can have 32768 while Qwen3-Coder has 65536. A window above what the family supports is refused when it is set (and cut to
  it if the file says more); a model with no window at all (a picture, a song, speech, embeddings) refuses one. A longer window costs memory for the
  conversation's keys and values (the launcher script's notes put it at about 1 GB more per conversation for the Gemma 26B-A4B).

```
coli model                              list every model: on or off, pinned where, where it runs now, and the GPU policy
coli model disable gemma-4-31b          turn it off (and unload it)
coli model enable gemma-4-31b           turn it back on
coli model gpu qwen3-coder-30b cpu      always on the CPU       (vulkan: always the card; auto: follow --gpu)
coli model context gemma-4-e4b 32k      a context window of 32768 tokens (also 32768, 128k; auto: the default)
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
    "qwen3-coder-30b-a3b-colibri": {"gpu": "cpu", "context": 65536}
  }
}
```

A record has `enabled` (`false`; anything else is on), `gpu` (`vulkan`, `cpu`, `auto`) and `context` (a whole number of tokens, at least 256). The router reads the file again whenever it
changes. A file that cannot be read keeps the last good settings and says so once in the router's log. An unknown key or a wrong value
is an error that names it.

## The API

All need the router's key, and a JSON `Content-Type`; a request from another origin is refused, like the other management calls.

| | |
|---|---|
| `GET /v1/models/settings` | every installed model: `id`, `key`, `enabled`, `gpu` (the pin), `state`, `running` (where a loaded one runs: `{"mode": "vulkan"\|"cpu"\|"unchanged", "why": ..., "free_gb": ...}`), plus `gpu_policy` (the router's `--gpu`, the reserve, the minimum, the card's free memory now) and `file` |
| `POST /v1/models/settings` | `{"model": NAME, "enabled": false, "gpu": "cpu", "context": 32768}`: `model` (or `key`), and `enabled`, `gpu` and/or `context` (a number of tokens, or `"auto"` for the default; a 400 names the field for a number below 256, above the family's limit, or for a model with no window). A changed window unloads a loaded model, as turning it off does (it starts again with the new window when it is next asked for). Answers the model's settings, and `unload` (`unloaded`, `unloading when idle`) when it was turned off |

`GET /v1/models/settings` rows also carry `context` (the model's own window, or `null`) and `context_max` (what its family supports; `0` for a model with no
window). `GET /v1/models/installed` also carries `enabled` for each model, and `gpu_setting` when one is pinned.

## Looking inside a model, and checking its files

`coli model info [MODEL]` says what a model folder is from its small files only, without loading anything and without a server: the
engine that reads it, the architecture, the weights (`bf16`, `int8`, `u8 packed`: the header cannot tell a packed int4 from other bytes),
the context limit the model was made for (the server may serve less: that is the `context_length` of its card), how many files and tensors,
and where it came from. It names what is missing when a folder is not recognised ("no config.json that can be read", "Colibri has no
engine for it"). `MODEL` is a name, an id or a folder; without one it describes every installed model.

`coli model verify [MODEL] [--quick]` checks that the files are whole. With the manifest `coli setup` wrote it checks every listed file's
size and, where the manifest holds one, its checksum (a manifest from an older download lists sizes only, and the answer says so); it
also checks that each safetensors file is as long as its own header says, which is what a download cut short looks like. A model
fetched another way has no manifest, and gets just the header check. `--quick` skips the checksums, which on a slow drive is the
difference between seconds and minutes. It exits 1 when something is wrong.

## In the dashboard

Settings → Model lists the installed models, and against a router that keeps these settings each row also has:

* **Runs on** (Automatic, Graphics card, CPU): the pin. It takes effect the next time the model loads, and a line above the list gives the
  router's `--gpu` policy and the card's free memory now.
* **Context window** (default, then 4k up to what the family supports, and the model's own setting when it is an odd number): only for models that have
  one. A change unloads a loaded model, which starts again with the new window when it is next asked for.
* **Turn off / Turn on**: the switch. A model that is off says so, cannot be loaded, and is unloaded when it is turned off.
* A loaded model says where the router started it (`on the graphics card`, `on the CPU`) beside its memory and when it unloads.

Both go through `POST /v1/models/settings`. A router without a settings file, or an older one, shows the list as before.
