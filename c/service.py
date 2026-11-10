"""Run `coli router` as a background service: start it, stop it, restart it, say how it is (docs/tray.md).

This is what restart-colibri.ps1 did by hand, so that the tray (tray.py) and `coli service` do it the same way:

- a key made once and kept in a file only you can read, reused on every start;
- the router started hidden, with the key in its environment (never on a command line, where other programs read it);
- a stop that refuses while a request is running, unless told to force it;
- the dashboard link with the key in it, for this PC and for the network.

Nothing here needs more than the standard library. The settings are a JSON file (service.json in the setup folder); every
key is optional and the defaults are the ones the script had. The key, the logs and the request log keep the script's
names in the home folder, so the two can be used one after the other.
"""
import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    "port": 8000,
    "router": True,                    # False: one model, named by "model", behind `coli serve`
    "model": None,
    "keep_alive": "10m",               # how long a router's model stays loaded after its last request
    "kv_slots": 1,
    "cap": 128,                        # experts cached per layer for the chat models (0: the engine's own default)
    "ctx": 32768,                      # context window for the Qwen3.6-family chat models (0: the engine's own default)
    "lan": True,                       # reachable from the network (with the key); False: this PC only
    "request_log": "~/colibri-requests.log",     # one JSON line per request; "" turns it off (docs/request-log.md)
    "files_dir": "~",                  # where the key and the server's logs are kept
    "start_with_tray": True,           # the tray starts the server when it starts, if it is not running
    "extra_args": [],                  # more arguments for `coli router` / `coli serve`
    "env": {},                         # more environment variables for the server and its engines, e.g. {"COLI_VULKAN": "1"} (docs/ENVIRONMENT.md)
}
TYPES = {"port": int, "router": bool, "keep_alive": str, "kv_slots": int, "cap": int, "ctx": int, "lan": bool,
         "files_dir": str, "start_with_tray": bool, "extra_args": list, "env": dict}


class ServiceError(RuntimeError):
    """A stop with a reason the user can act on."""


# ---------------------------------------------------------------- settings


def config_path():
    explicit = os.environ.get("COLI_SERVICE")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    try:
        import setup_flow
        return os.path.join(setup_flow.setup_home(), "service.json")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".colibri-service.json")


def load_config(path=None, overrides=None):
    """The settings: the defaults, then the file, then `overrides`. A file that cannot be read or has a wrong value is an
    error that says which, because a service that quietly ignores a typo starts with settings nobody chose."""
    path = path or config_path()
    cfg = dict(DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                found = json.load(handle)
        except (OSError, ValueError) as error:
            raise ServiceError(f"{path} cannot be read: {error}")
        if not isinstance(found, dict):
            raise ServiceError(f"{path} must hold a JSON object")
        for key in found:
            if key not in DEFAULTS:
                raise ServiceError(f"{path}: unknown setting {key!r} (known: {', '.join(sorted(DEFAULTS))})")
        cfg.update(found)
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key, kind in TYPES.items():
        value = cfg[key]
        if kind is int and isinstance(value, bool) or not isinstance(value, kind):
            raise ServiceError(f"setting {key!r} must be a {kind.__name__}, not {value!r}")
    if not 1 <= cfg["port"] <= 65535:
        raise ServiceError(f"setting 'port' must be 1-65535, not {cfg['port']}")
    for key in ("model", "request_log"):
        if cfg[key] is not None and not isinstance(cfg[key], str):
            raise ServiceError(f"setting {key!r} must be text, not {cfg[key]!r}")
    for name, value in cfg["env"].items():
        if not isinstance(value, str):
            raise ServiceError(f"setting 'env': {name!r} must be text, not {value!r}")
        if not name or "=" in name or name.upper() in ("COLI_API_KEY",):
            raise ServiceError(f"setting 'env': {name!r} cannot be set here" + (" (the service makes the key)" if name.upper() == "COLI_API_KEY" else ""))
    if not cfg["router"] and not cfg["model"]:
        raise ServiceError("setting 'router' is false, so 'model' must name the model directory to serve")
    return cfg


def paths(cfg):
    """Where this service keeps its files; the names are the script's, the port is in them unless it is 8000."""
    home = os.path.abspath(os.path.expanduser(cfg["files_dir"]))
    suffix = "" if cfg["port"] == 8000 else f"-{cfg['port']}"
    return {"key": os.path.join(home, f"colibri-api-key{suffix}.txt"),
            "out": os.path.join(home, f"colibri-serve{suffix}.log"),
            "err": os.path.join(home, f"colibri-serve{suffix}.err"),
            "request_log": os.path.abspath(os.path.expanduser(cfg["request_log"])) if cfg["request_log"] else None}


# ---------------------------------------------------------------- the key and the links


def read_key(cfg):
    try:
        with open(paths(cfg)["key"], encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def make_key():
    return secrets.token_urlsafe(32)                  # 256 bits, URL-safe, so it can sit in a link's #key=


def ensure_key(cfg, new=False):
    """The saved key, or a new one (the first time, or `new`) saved so that only you can read it."""
    key = read_key(cfg)
    if key and not new:
        return key
    key, path = make_key(), paths(cfg)["key"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii") as handle:
        handle.write(key)
    if sys.platform == "win32":
        user = os.environ.get("USERNAME")
        if user:
            subprocess.run(["icacls", path, "/inheritance:r", "/grant:r", f"{user}:(R,W)"],
                           capture_output=True, check=False)
    else:
        os.chmod(path, 0o600)
    return key


def lan_address():
    """This PC's address on the network, or None. Connecting a UDP socket sends nothing; it only makes the system pick the
    interface a packet for the outside would leave by."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))            # a documentation address: nothing is sent
            address = sock.getsockname()[0]
        return None if address.startswith(("127.", "0.")) else address
    except OSError:
        return None


def links(cfg, key, lan=None):
    """The dashboard links, with the key where the page reads it and removes it from the address bar."""
    port = cfg["port"]
    out = {"local": f"http://127.0.0.1:{port}/#key={key}", "lan": None}
    lan = lan_address() if lan is None else lan
    if cfg["lan"] and lan:
        out["lan"] = f"http://{lan}:{port}/#key={key}"
    return out


# ---------------------------------------------------------------- asking the server


def api(cfg, path, key="", method="GET", body=None, timeout=3.0):
    """(status, JSON) from the server on this PC, or (None, None) when nothing answers."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}{path}", data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read() or b"null")
        except ValueError:
            return error.code, None
    except (urllib.error.URLError, OSError, ValueError):
        return None, None


def busy_count(health):
    """Requests running or waiting, in a router (its gate and every worker) or a single server."""
    if not isinstance(health, dict):
        return 0
    scheduler = health.get("scheduler") or {}
    total = int(scheduler.get("active") or 0) + int(scheduler.get("queued") or 0)
    router = health.get("router")
    if isinstance(router, dict):
        gate = router.get("gate") or {}
        total += int(gate.get("active") or 0) + int(gate.get("queued") or 0)
        total += sum(int(w.get("inflight") or 0) for w in router.get("loaded") or [])
    return total


def status(cfg):
    """How the service is. `state` is `stopped` (nothing listens), `running`, or `unreachable` (something listens and does not
    answer /health, so it is not ours or it is stuck). `key_ok` is false when it answers but does not accept our key."""
    key = read_key(cfg)
    code, health = api(cfg, "/health", key)
    out = {"state": "stopped", "port": cfg["port"], "key": key, "key_ok": True, "busy": 0, "loaded": [],
           "router": False, "health": None, "pids": []}
    if code is None:
        pids = listeners(cfg["port"])
        if pids:
            out.update(state="unreachable", pids=pids)
        return out
    out.update(state="running", health=health)
    if code == 200 and isinstance(health, dict):
        router = health.get("router")
        out["router"] = isinstance(router, dict)
        out["key_ok"] = isinstance(router, dict) or not cfg["router"]      # unauthenticated /health has no router part
        out["busy"] = busy_count(health)
        if isinstance(router, dict):
            out["loaded"] = [dict(w) for w in router.get("loaded") or [] if isinstance(w, dict)]
    return out


# ---------------------------------------------------------------- processes


def parse_netstat(text, port):
    """Process ids listening on `port`, from the text of `netstat -ano -p tcp`."""
    pids = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING" \
                and parts[1].rsplit(":", 1)[-1] == str(port):
            if parts[4].isdigit() and int(parts[4]) not in pids:
                pids.append(int(parts[4]))
    return pids


def listeners(port):
    """Ids of the processes that listen on `port` on this PC."""
    try:
        if sys.platform == "win32":
            text = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True, timeout=10).stdout
            return parse_netstat(text, port)
        text = subprocess.run(["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True,
                              timeout=10).stdout
        return [int(p) for p in text.split() if p.isdigit()]
    except (OSError, subprocess.SubprocessError):
        return []


def python_exe():
    """The interpreter for the server: the console one, even when the tray itself runs under pythonw."""
    exe = sys.executable
    if sys.platform == "win32" and os.path.basename(exe).lower() == "pythonw.exe":
        console = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(console):
            return console
    return exe


def command(cfg, python=None):
    """The command line that starts the server (no key: that goes in the environment)."""
    coli = os.path.join(HERE, "coli")
    port = str(cfg["port"])
    if cfg["router"]:
        args = [coli, "router", "--port", port, "--keep-alive", cfg["keep_alive"], "--kv-slots", str(cfg["kv_slots"])]
        if cfg["cap"] > 0:
            args += ["--worker-arg=--cap", f"--worker-arg={cfg['cap']}"]      # only the chat engines read it
    else:
        args = [coli, "serve", "--model", cfg["model"], "--port", port, "--kv-slots", str(cfg["kv_slots"])]
        if cfg["cap"] > 0:
            args += ["--cap", str(cfg["cap"])]
    if cfg["lan"]:
        args += ["--host", "0.0.0.0", "--allowed-host", "*"]
    return [python or python_exe()] + args + [str(a) for a in cfg["extra_args"]]


def environment(cfg, key, base=None):
    """The environment of the server: the key and the settings that are read from it."""
    env = dict(os.environ if base is None else base)
    env["COLI_API_KEY"] = key
    env["PYTHONUTF8"] = "1"
    log = paths(cfg)["request_log"]
    if log:
        env["COLI_REQUEST_LOG"] = log
    if cfg["ctx"] > 0:
        env["Q36_MAXT"] = str(cfg["ctx"])            # read by the chat engine of the Qwen3.6 family; the others ignore it
    env.update(cfg["env"])                           # the user's own, last: they may replace a default (never the key)
    return env


def kill(pid):
    """Stop a process and what it started (the router's workers and their engines)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
        return
    import signal
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def start(cfg, new_key=False, wait=120.0, out=lambda text: None):
    """Start the server and wait until it answers. Returns the status. A server that already answers on the port is left
    alone, unless `new_key` was asked for: the key it has is not the new one, so that needs `restart`."""
    now = status(cfg)
    if now["state"] != "stopped":
        if now["state"] == "unreachable":
            raise ServiceError(f"something is listening on port {cfg['port']} and does not answer /health "
                               f"(process {', '.join(map(str, now['pids']))}); stop it or change the port")
        if new_key:
            raise ServiceError("the server is running with its old key: use restart --new-key")
        out(f"already running on port {cfg['port']}")
        return now
    key = ensure_key(cfg, new_key)
    files = paths(cfg)
    for path in (files["out"],):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs = {"stdin": subprocess.DEVNULL, "env": environment(cfg, key)}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000 | 0x00000200      # no window, and its own group so a Ctrl+C here is not its
    else:
        kwargs["start_new_session"] = True
    with open(files["out"], "ab") as stdout, open(files["err"], "ab") as stderr:
        process = subprocess.Popen(command(cfg), stdout=stdout, stderr=stderr, **kwargs)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        time.sleep(0.5)
        now = status(cfg)
        if now["state"] == "running":
            return now
        if process.poll() is not None:
            raise ServiceError(f"the server stopped at once (exit code {process.returncode}); see {files['err']}")
    raise ServiceError(f"the server did not answer in {int(wait)} seconds; see {files['err']}")


def stop(cfg, force=False, wait=30.0):
    """Stop the server. Refuses while a request is running or waiting, unless `force`. Returns the ids it stopped."""
    now = status(cfg)
    if now["state"] == "stopped":
        return []
    if now["state"] == "running" and now["key_ok"] and now["busy"] and not force:
        raise ServiceError(f"{now['busy']} request(s) are running or waiting; wait for them, or force the stop (they are lost)")
    if (now["state"] == "unreachable" or not now["key_ok"]) and not force:
        raise ServiceError(f"what listens on port {cfg['port']} does not answer /health with our key, so I cannot tell whether "
                           "it is rendering; force the stop to end it anyway")
    pids = listeners(cfg["port"])
    for pid in pids:
        kill(pid)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline and listeners(cfg["port"]):
        time.sleep(0.5)
    if listeners(cfg["port"]):
        raise ServiceError(f"port {cfg['port']} is still in use after {int(wait)} seconds")
    return pids


def restart(cfg, new_key=False, force=False, wait=120.0, out=lambda text: None):
    stop(cfg, force=force)
    return start(cfg, new_key=new_key, wait=wait, out=out)


def unload(cfg, model_key, key=None):
    """Ask the router to unload a model; returns the router's word ('unloaded', 'busy' ...) or None."""
    code, body = api(cfg, "/v1/models/unload", key if key is not None else read_key(cfg), "POST", {"key": model_key})
    if code in (200, 202) and isinstance(body, dict):
        return body.get("status")
    return None


# ---------------------------------------------------------------- command line


def firewall_open(port):
    """Whether Windows has a rule for the port (None where that cannot be told)."""
    if sys.platform != "win32":
        return None
    try:
        found = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", f"name=Colibri {port}"],
                               capture_output=True, text=True, timeout=10)
        return found.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return None


def describe(cfg, now):
    lines = []
    if now["state"] == "stopped":
        lines.append("stopped")
    elif now["state"] == "unreachable":
        lines.append(f"something listens on port {cfg['port']} (process {', '.join(map(str, now['pids']))}) and does not answer")
    else:
        word = f"running on port {cfg['port']}"
        if not now["key_ok"]:
            word += " (it does not accept the saved key: it was started with another)"
        elif now["busy"]:
            word += f", {now['busy']} request(s) in progress"
        lines.append(word)
        for worker in now["loaded"]:
            gb = f"{worker['resident_bytes'] / 2**30:.1f} GB" if worker.get("resident_bytes") else "-"
            lines.append(f"  loaded: {worker.get('id')}  {worker.get('state')}  {gb}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(prog="coli service", description="run `coli router` in the background (docs/tray.md)")
    ap.add_argument("action", choices=("start", "stop", "restart", "status", "config", "link"))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--config", default=None, metavar="FILE", help=f"the settings file (default {config_path()})")
    ap.add_argument("--new-key", action="store_true", help="make a new key; the old one stops working")
    ap.add_argument("--force", action="store_true", help="stop even if a request is running (it is lost)")
    ap.add_argument("--local-only", action="store_true", help="listen on this PC only (still with a key)")
    ap.add_argument("--open", action="store_true", help="open the dashboard, signed in")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.config, {"port": a.port, "lan": False if a.local_only else None})
        if a.action == "config":
            print(json.dumps(cfg, indent=2) if a.json else "\n".join(f"{k:<16}{v}" for k, v in cfg.items()))
            print(f"\nthe file: {a.config or config_path()}" + ("" if os.path.exists(a.config or config_path()) else " (not there; these are the defaults)"),
                  file=sys.stderr)
            return 0
        if a.action == "status":
            now = status(cfg)
            if a.json:
                print(json.dumps({k: v for k, v in now.items() if k not in ("health", "key")}, indent=2))
            else:
                print("\n".join(describe(cfg, now)))
            return 0 if now["state"] == "running" else 1
        if a.action == "link":
            print(links(cfg, ensure_key(cfg))["local"])
            return 0
        if a.action == "stop":
            pids = stop(cfg, force=a.force)
            print(f"stopped (process {', '.join(map(str, pids))})" if pids else "was not running")
            return 0
        say = lambda text: print(text)          # noqa: E731
        now = restart(cfg, a.new_key, a.force, out=say) if a.action == "restart" else start(cfg, a.new_key, out=say)
        key = ensure_key(cfg)
        shown = links(cfg, key)
        print(f"up on port {cfg['port']}. API key in {paths(cfg)['key']}")
        print(f"  this PC:  {shown['local']}")
        if shown["lan"]:
            print(f"  network:  {shown['lan']}")
            if firewall_open(cfg["port"]) is False:
                print(f"\nOther PCs cannot connect until Windows allows port {cfg['port']}. In an Administrator PowerShell:\n"
                      f"  New-NetFirewallRule -DisplayName 'Colibri {cfg['port']}' -Direction Inbound -Protocol TCP "
                      f"-LocalPort {cfg['port']} -Action Allow -RemoteAddress LocalSubnet")
        if a.open:
            import webbrowser
            webbrowser.open(shown["local"])
        return 0
    except ServiceError as error:
        print(f"coli service: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
