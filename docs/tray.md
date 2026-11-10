# The tray icon and the background service

A circle in the Windows notification area that shows whether the server is up, and a menu to open the dashboard, start,
stop and restart the server, unload a model, and start with Windows. Behind it is `coli service`, which does the same from
a terminal (and on any system), so the server is started the same way by both.

```
coli tray               start the tray (a separate process; the terminal is free again)
coli tray --install     start the tray whenever you sign in to Windows
coli tray --remove      stop doing that

coli service start      start `coli router` in the background and wait until it answers
coli service stop       stop it (refuses while a request is running; --force to stop anyway)
coli service restart    stop, then start; --new-key makes a new API key
coli service status     running or not, requests in progress, models loaded (exit code 1 when not running)
coli service link       the dashboard link, signed in
coli service config     the settings in effect
```

## The icon

| Circle | Meaning |
|---|---|
| grey ring (hollow) | the server is stopped |
| green | up, nothing to do |
| blue | up and working: a request is running, or a model is loading |
| amber | the tray is starting, stopping or restarting it |
| red, with a `!` | something is wrong: another program uses the port, or the server is up but does not accept the saved key |

The tooltip says the same in words. If the server stops without the tray having done it, a message pops up.

## The menu

Click the icon (either button).

- **The state**, then one line per loaded model: `Unload qwen3-coder-30b (15.2 GB, unloads in 8m 12s)`. A model that is
  answering a request is unloaded as soon as it has finished. The menu is made when you open it, so the times are current.
- **Open the dashboard**, already signed in (the key is in the link and the page removes it from the address bar).
- **Copy the network link** (for other PCs; it holds the key, so share it like a password) and **Copy the API key**.
- **Start / Restart / Stop the server.** While a request is running these are replaced by *Restart anyway* and *Stop anyway*,
  which say how many requests will be lost. The same when the server cannot be asked whether it is working (it does not
  accept the saved key, or something that is not colibri uses the port): it is not ended unless you say so.
- **Open the request log** (when [the log](request-log.md) is on) and **the folder with the server's logs**.
- **Start with Windows**: a tick that adds or removes the tray's entry under `HKCU\...\CurrentVersion\Run`. Nothing needs
  administrator rights, and `coli tray --remove` does the same.
- **Quit the tray.** The server keeps running: it does not belong to the tray.

When the tray starts and the server is not running, it starts it (`"start_with_tray": false` in the settings stops that).

## What the service does

It is what `restart-colibri.ps1` did by hand, and the two can be used one after the other:

- **The key** is made once (256 bits) and kept in `~/colibri-api-key.txt`, readable by you only, and reused on every start;
  `--new-key` replaces it. It reaches the server through its environment, never its command line.
- **The server** is `coli router --port 8000 --keep-alive 10m --kv-slots 1 --worker-arg=--cap --worker-arg=128 --host 0.0.0.0
  --allowed-host '*'`: every installed model on one port, loaded when asked for ([router.md](router.md)), and reachable
  from the network with the key. It runs without a window, in a group of its own, so closing a terminal does not end it. Its
  output is in `~/colibri-serve.log` and `~/colibri-serve.err`.
- **A stop** ends the router and the workers and engines under it. It refuses while a request is running or waiting.
- **Windows' firewall** is not touched. When other PCs cannot reach the server, `coli service start` prints the one command
  (for an Administrator PowerShell) that allows the port, if there is no rule for it yet.

## Settings

All optional, in `service.json` in the setup folder (`%LOCALAPPDATA%\colibri`; `COLI_SERVICE` names another file). A key that
is not in this list is an error, so a typo is not mistaken for a default. `coli service config` shows what is in effect.

| Key | Default | |
|---|---|---|
| `port` | `8000` | |
| `router` | `true` | `false` serves one model with `coli serve`; then `model` must name its directory |
| `model` | none | the model directory, when `router` is false |
| `keep_alive` | `"10m"` | how long a model stays loaded after its last request |
| `kv_slots` | `1` | conversations that keep their own cache |
| `cap` | `128` | experts cached per layer for the chat models (`0`: the engine's default) |
| `ctx` | `32768` | context window for the Qwen3.6-family chat models (`0`: the engine's default) |
| `lan` | `true` | `false`: this PC only (still with a key) |
| `request_log` | `"~/colibri-requests.log"` | `""` turns the [request log](request-log.md) off |
| `files_dir` | `"~"` | where the key and the server's logs are kept |
| `start_with_tray` | `true` | the tray starts the server when it starts |
| `extra_args` | `[]` | more arguments for `coli router` |

A port other than 8000 puts it in the file names (`colibri-api-key-8100.txt`), as the script did.

## Not built

- It is Windows only (`coli service` is not). A tray on macOS or Linux would be another window layer under the same menu.
- The menu does not switch models or edit presets; the dashboard does.
- The tray does not install a Windows service that runs before anyone signs in; it starts with you.
- The tray is checked by tests that make the window and its menu, not by a person clicking it in every Windows setting
  (notification settings or Focus assist can hide the pop-up messages; the icon and its colour still change).
