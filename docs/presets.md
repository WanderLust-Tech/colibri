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
- `POST /v1/presets` creates or replaces one: the JSON body is the preset as the file holds it (`model`, `system`,
  `description`, `keep_alive`, `defaults`) plus its `name`. It answers `created` or `replaced` and takes effect at once.
  It is refused (400, 404 or 409) for a bad name or key, a model that is not installed, or a name that is a model's own,
  and then the file is not touched. `POST /v1/presets/delete` with `{"name": ...}` removes one (404 when there is none).
  Both need the API key and the dashboard's same-origin rule, like switching models.

## In the dashboard

- **The model list** of the chat shows presets under their own group, with a note saying what each is a preset of. Choosing
  a preset starts the controls from its settings: the temperature and length of a chat, the size and steps of a picture.
  The page always sends those, and a value a request carries beats the preset's, so without this a preset's temperature
  would never apply. Anything the page has no control for (a system prompt, `top_p`, a music `duration`) is left to the
  router, which fills it in because the page does not send it.
- **Settings, Presets** lists the file with each preset's model, settings and system prompt, and why one cannot be used
  (shown, not hidden). *Use* opens the conversation on it, *Edit* and *Delete* (with a question first) change the file, and
  *New preset* opens a form: name, the model to run (from the installed ones), description, a system prompt (for a text
  model), settings as rows of a name and a value (a number, `true`/`false`, or text; the editor suggests the names that fit
  the model) and how long the model stays loaded. The router's refusal (a taken name, a key a preset may not set) is shown
  as it is said.
- A server that is not a router has no presets; the tab says so.

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

The dashboard does not set a preset's music settings (duration, steps) from the preset the way it does a chat's and a
picture's; the router still fills them in. A preset cannot change what is loaded (quantisation, context window): those belong to the worker, started once for
the base model, so two presets of one model share it.
