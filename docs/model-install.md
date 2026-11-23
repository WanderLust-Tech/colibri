# Installing and removing models

`coli setup` fetches one model, builds its engine and starts a server. A machine that keeps several models (a chat model, a
light picture model, a heavy one) needs the other half: fetch another, or remove one, while the router is running. That is
`coli model available`, `install` and `uninstall`, the dashboard's **Models** tab and the editor extension's models menu. All
three do the same thing, through the router's API ([below](#the-api)), or directly when no router answers.

What can be fetched is the same catalog `coli setup` uses (`coli model available` lists it): checkpoints that run straight after
a plain download. A model that needs a conversion first (GLM, OLMoE, Qwen3.8) is not in it; see the README.

## From the terminal

```
coli model available                 what can be fetched: size, kind, whether it fits this machine, licence, and what is here
coli model install z-image-turbo     show the size and the licence, ask, then fetch it with a progress bar
coli model install qwen-image-2.1 --yes
coli model uninstall z-image-turbo   show what goes and how much space it frees, ask, then remove it
```

* `install` shows the download size, the licence note and what this machine makes of it (does it fit in RAM, is there room on the
  disk) and asks. Without a terminal, or with `--yes`, it does not ask, so a script has to say so. Qwen-Image-2.1 for example is
  under the Qwen Research License (non-commercial); Z-Image-Turbo is Apache-2.0.
* The licence line is what the model's repository declares on Hugging Face (read 2026-11-19), shortened; a licence that is not
  a standard one (the Qwen, GLM and Kimi licences) says so and points to the model page for its terms. It is information, not
  legal advice; `install` asks you to accept it, and `--yes` accepts it for you.
* The download is resumable and verified file by file (the same downloader as `coli setup`). Ctrl-C cancels it and keeps what was
  fetched; run the command again to continue. A failed one says why and also keeps its part.
* One install runs at a time. A second says so.
* A model goes into the first models folder with room: `COLI_MODELS_ROOT`, the folder `coli setup` chose, then
  `~/colibri-models`. The router lists it as soon as its files are there; the engine it needs must be built or installed (`coli setup`
  does that, or a release archive has it), and the list says so when it is not.
* `uninstall` refuses a model that is loaded (`coli stop NAME` first), a folder that is not directly inside a models folder, a link,
  and a folder a download is writing. The folder is renamed out of the model list first and then deleted, so a router never lists
  a half-deleted model, and an interruption is finished at the next start.
* Presets that name a removed model stay in the presets file; they report that the model is not installed until it is installed
  again.

**Add-ons.** A few catalog entries are not models of their own but files that go into the folder of one that is installed:
ACE-Step's `ace-step-1.5-planner-1.7b` (the language-model planner, 3.7 GB, from the main repository's `acestep-5Hz-lm-1.7B/`) and
`ace-step-1.5-base-dit` (the guided DiT for extract, lego and complete, 4.8 GB, from `ACE-Step/acestep-v15-base`: only the config, the
weights and the silence latent). They install and uninstall like any other (`coli model install ace-step-1.5-planner-1.7b`) and their rows in
`coli model available` and the dashboard say what they belong to (`into`). The rest differs in four ways:

* the parent has to be installed first (`409 needs_parent`; the row's status is `needs-parent` and says what to run);
* they land in a subfolder of the parent's folder (`<models>/ace-step-1.5/acestep-5Hz-lm-1.7B/`), with a manifest of their own, so
  the parent's stays complete and the engine finds them where it looks; a folder that holds the files `check` names counts as installed
  however it got there (one put together by hand, as docs/acestep.md used to say);
* removing one takes only that subfolder, and is refused while the parent is loaded (the music engine keeps the planner's files open:
  unload it first, `coli stop ace-step-1.5`);
* the music engine reads what is installed when it starts, so the router stops the parent's engine when an add-on has landed (one that
  is answering a request keeps going, and the add-on shows up once it has been stopped and started again: `coli stop ace-step-1.5`).

`coli setup` never offers an add-on. The planner's 0.6B and 4B and the SFT DiT are not in the catalog: the 4B is two shards, and the installer
takes one folder of files. All three have been held to upstream on real weights (2026-11-23, [acestep.md](acestep.md)); they go in by hand.

With `--host`, `--port` and `--api-key` (or `COLI_API_KEY`) the same commands manage the router of another machine: the files go
into that machine's models folder.

## In the dashboard and the editor

The dashboard's **Models** tab has an *Install a model* list below the models that are here: each row shows the size, the kind, the
licence and whether it fits, and an Install button that asks first. A running install shows its progress and a Cancel button.
A model that is here has a Remove button, which asks too and is off while the model is loaded. The editor extension's models menu
has the same two entries (*Install a model...*, *Remove a model...*). Both hide them when the router does not have
`features.model_install`.

## The API

All of these need the API key, as the other model routes do, and only `coli router` has them (a server that serves one model
answers 404). The write routes take `application/json` and refuse a cross-origin page, like `/v1/models/switch`.

| Route | |
|---|---|
| `GET /v1/models/catalog` | every model that can be fetched: `id`, `name`, `kind` (`text`, `image`, `audio`), `download_gb`, `ram_min_gb`, `ram_good_gb`, `summary`, `license_note`, `status` (`good`, `tight`, `needs-ram`, `needs-disk`, `unsupported`, `needs-parent`), `fits`, `reason`, `installed` (`key`, `id`, `state`, or null; an add-on's `key` is null), `unfinished` (a download was started), `job`, and `into` for an add-on (the catalog id of the model it goes into). Also `roots` (the models folders) and `free_gb` |
| `POST /v1/models/install` | `{"id": "z-image-turbo", "confirm": true}`. `confirm` must be `true`: the caller has shown the size and the licence. Answers `202` with the job |
| `GET /v1/models/installs` | the jobs, newest first (the last 8) |
| `GET /v1/models/installs/{id}` | one job: `state` (`starting`, `downloading`, `finishing`, `done`, `failed`, `cancelled`), `done` and `total` bytes, `rate`, `eta_seconds`, `file`, `error`, `license_note` |
| `POST /v1/models/install/cancel` | `{"id": ...}`. The part fetched is kept |
| `POST /v1/models/uninstall` | `{"model": NAME, "confirm": true}` (or `key`, from `GET /v1/models/installed`). Answers `freed_bytes` and `presets`, the names of the presets that use the model. For an add-on: `{"id": "ace-step-1.5-planner-1.7b", "confirm": true}` (`presets` is empty) |

Refusals: `400 confirmation_required`, `404 model_not_found`, `409 already_installed`, `install_busy` (another install is running),
`insufficient_disk`, `folder_exists` (the target folder holds something that is not a download of this model),
`needs_parent` (an add-on whose model is not installed), `model_missing` (nothing of an add-on is here to remove),
`model_busy` (another colibri process, or a download, holds the folder), `model_loaded`, `model_in_use` (the files are open in
another process), `outside_models_folder`, `model_is_link`.

`GET /health` with the key says `features.model_install` when the router has these routes.

## What it does not do

* **Build the engine.** `coli setup` builds or fetches the engine of a model's family; installing a model through the router only
  puts its files in place. A model whose engine is missing is listed as not startable, with the reason.
* **Gated models.** A repository that needs a Hugging Face login uses `HF_TOKEN` of the router's environment, as `coli setup` does.
* **Several downloads at once**, a download queue, or a bandwidth limit.
* **Moving a model** to another folder or drive.
* **Models outside the models folders.** A model the router was given by path (`--model`), or one that is a link, is not removed.

## Tests

`tests/test_model_installer.py` (the installer against a fake hub: confirmation, one at a time, progress, cancel and resume, a
failure that keeps the part, the lock between processes, the uninstall's refusals) and `tests/test_router_install.py` (the
endpoints, with fake workers for the loaded-model cases).
