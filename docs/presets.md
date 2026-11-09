# Named presets

A preset is a model with settings baked in, offered under a name of its own: `photo-1024` for the image model at
1024x768 and 20 steps, `terse` for the coder model with a system prompt and a low temperature, `lofi-60s` for the music
model at a minute. Clients ask for it by name in `model`, like any model. The router loads the base model and fills in
what the request left out.

It follows Ollama's Modelfile (a model plus a system prompt and parameters under a new name), without making a copy of
the model: a preset is a few lines of JSON, the weights stay where they are.

## Making one

```
coli preset add terse --model qwen3-coder-30b-a3b --system "Answer in one sentence." --set temperature=0.2 --set max_tokens=300
coli preset add photo-1024 --model qwen-image-2.1 --set size=1024x768 --set steps=20
coli preset add lofi-60s --model ace-step-1.5 --set duration=60 --set steps=8 --keep-alive 30m
coli preset                  # list
coli preset show terse
coli preset rm terse
```

`--set` values are JSON where they can be (`0.2`, `20`, `true`) and text otherwise (`1024x768`). The file is
`presets.json` in the setup folder (`%LOCALAPPDATA%\colibri` on Windows; `COLI_PRESETS` or `--file` names another):

```json
{"presets": {
  "terse": {"model": "qwen3-coder-30b-a3b", "system": "Answer in one sentence.",
            "defaults": {"temperature": 0.2, "max_tokens": 300}, "description": "short answers"},
  "photo-1024": {"model": "qwen-image-2.1", "defaults": {"size": "1024x768", "steps": 20}}}}
```

A running `coli router` reads the file again whenever it changes, so a new preset needs no restart. A file that
breaks (a bad edit) leaves the router on the presets it had read, and the problem is in the router's log and in
`GET /v1/presets`.

## Using one

```
curl localhost:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -d '{"model": "terse", "messages": [{"role": "user", "content": "What is a mutex?"}]}'
curl localhost:8000/v1/images/generations -H "Authorization: Bearer $KEY" -d '{"model": "photo-1024", "prompt": "a lighthouse at dusk"}'
```

- `GET /v1/models` lists each usable preset as a model of its own, with `preset_of` (the base model's id) and `preset`
  (its settings) added to the base model's card, so a client that picks models from the list shows them.
- `GET /v1/presets` lists every preset in the file with `available` and, when not, `reason` (the model is not
  installed, it cannot start here, the name is a model's own), and the entries of the file that could not be read.
- `POST /v1/models/switch` and `/v1/models/unload` take a preset name and act on its base model.

## What a preset fills in

- **`defaults`:** every key the request has no value for. **A value the request carries wins**, and `null` counts as
  absent. This works for JSON bodies and, as text fields, for multipart ones (picture edits, music with clips).
  Anything the endpoint takes can be a default (`temperature`, `top_p`, `max_tokens`, `size`, `steps`, `duration`,
  `response_format`, `enable_thinking`...), except what a request is *about*, which the file refuses:
  `model`, `messages`, `prompt`, `input`, `image`, `images`, `mask`, `file`, `src_audio`, `reference_audio`, `tools`,
  `functions`.
- **`system`:** a system prompt, added only when the conversation has none of its own (no `system` or `developer`
  message on `/v1/chat/completions`; no `system` field on `/v1/messages`). On other endpoints it does nothing.
- **`keep_alive`:** how long the base model stays loaded, when the request does not say.

The base model sees an ordinary request for itself: the worker is sent the model's own id, and no preset key reaches it.

## Rules

- A name is 1 to 64 characters: lower-case letters, digits, `.`, `_`, `-`; names are not case sensitive.
- **A real model's name is never taken by a preset.** A preset called like an installed model is reported unavailable.
- A preset of a model that is not installed is left out of `/v1/models`; a request for it answers 404 and names the
  missing model.
- Presets do not nest: `model` names an installed model, not another preset.
- Presets are the router's. A single `coli serve` has none.

## Not built

The dashboard does not list presets in its model picker yet (the page is not touched); clients and `coli` use the
API. A preset cannot change what is loaded (quantisation, context window): those belong to the worker, started once for
the base model, so two presets of one model share it.
