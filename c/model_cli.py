"""`coli model available | install | uninstall` (docs/model-install.md).

Two backends behind one flow. With a router answering, the work goes through its API, so the dashboard and the terminal see the
same install and a loaded model is never removed. With none, the same model_installer.Installer runs here against the models
folders on this machine. The flow (show the size and the licence, ask, follow the download, ask again before deleting) is the
same for both."""
import os
import sys
import time

import setup_catalog

GB = 1_000_000_000


def human(value):
    value = float(value or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1000


def eta_text(seconds):
    if seconds is None:
        return "..."
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


class Refused(Exception):
    """Something the person should read; the exit code is 1."""


# ---------------------------------------------------------------- the router
class RemoteBackend:
    name = "router"

    def __init__(self, api):
        self.api = api

    def _call(self, method, path, body=None):
        status, result = self.api(method, path, body)
        if status is None:
            raise Refused("the server stopped answering")
        if status == 401:
            raise Refused("the server wants an API key: --api-key, or COLI_API_KEY")
        if status >= 400:
            error = (result or {}).get("error") or {}
            raise Refused(error.get("message") or f"HTTP {status}")
        return result

    def rows(self):
        return self._call("GET", "/v1/models/catalog")["data"]

    def installed(self):
        return self._call("GET", "/v1/models/installed")["data"]

    def folders(self):
        return self._call("GET", "/v1/models/catalog").get("folders") or []

    def start(self, model_id, root=None):
        return self._call("POST", "/v1/models/install", {"id": model_id, "confirm": True, **({"root": root} if root else {})})

    def job(self, model_id):
        return self._call("GET", f"/v1/models/installs/{model_id}")

    def cancel(self, model_id):
        return self._call("POST", "/v1/models/install/cancel", {"id": model_id})

    def find(self, name):
        wanted = name.lower()
        for item in self.installed():
            names = {item["id"].lower(), item["key"].lower(), (item.get("name") or "").lower()}
            if item["id"].lower().endswith("-colibri"):
                names.add(item["id"].lower()[:-len("-colibri")])
            if wanted in names:
                return item
        for catalog_row in self.rows():                  # the catalog's own id for a model that is here
            held = catalog_row.get("installed")
            if catalog_row["id"].lower() == wanted and held:
                if catalog_row.get("into"):              # an add-on has no entry of its own in the installed list
                    return {"id": catalog_row["id"], "key": None, "addon": True, "size_bytes": held.get("size_bytes")}
                return next((i for i in self.installed() if i["key"] == held["key"]), None)
        return None

    def plan(self, item):
        size = item.get("size_bytes") or (item.get("details") or {}).get("bytes")
        if item.get("state") in ("starting", "ready", "stopping"):
            raise Refused(f"{item['id']} is loaded; stop it first:  coli stop {item['id']}")
        return {"name": item["id"], "bytes": size, "where": None}

    def remove(self, item):
        if item.get("addon"):
            return self._call("POST", "/v1/models/uninstall", {"id": item["id"], "confirm": True})
        result = self._call("POST", "/v1/models/uninstall", {"key": item["key"], "confirm": True})
        return result


# ---------------------------------------------------------------- this machine
class LocalBackend:
    name = "this machine"

    def __init__(self, roots=None):
        import model_installer
        import model_switch
        self.switch, self.mi = model_switch, model_installer
        self._roots = roots
        self.installer = model_installer.Installer(roots=self.roots, installed=self.scan, launcher_dir=os.path.dirname(os.path.abspath(__file__)))

    def roots(self):
        if self._roots is not None:
            return list(self._roots)
        found = self.switch.candidate_roots(os.path.join(os.path.expanduser("~"), "colibri-models", "_"), os.environ,
                                            self.switch.setup_models_root())
        return found or [os.path.join(os.path.expanduser("~"), "colibri-models")]

    def scan(self):
        ctx = self.switch.SwitchContext(os.path.join(os.path.expanduser("~"), "colibri-models", "_none"), kv_slots=1,
                                        roots=self.roots(), environ={})
        return self.switch.scan_installed(ctx, lambda family: True)

    def rows(self):
        return self.installer.catalog_rows()

    def installed(self):
        return self.scan()

    def folders(self):
        return [{"path": f["path"], "free_gb": round(f["free_bytes"] / 1e9, 1) if f["free_bytes"] is not None else None,
                 "preferred": f["preferred"]} for f in self.installer.folders()]

    def start(self, model_id, root=None):
        try:
            return self.installer.start(model_id, confirm=True, root=root)
        except self.mi.InstallError as error:
            raise Refused(error.message)

    def job(self, model_id):
        return self.installer.job(model_id)

    def cancel(self, model_id):
        try:
            return self.installer.cancel_job(model_id)
        except self.mi.InstallError as error:
            raise Refused(error.message)

    def find(self, name):
        wanted = name.lower()
        for item in self.scan():
            names = {item["id"].lower(), item["key"].lower(), (item.get("name") or "").lower(),
                     os.path.basename(item["path"].rstrip("/\\")).lower()}
            if item["id"].lower().endswith("-colibri"):
                names.add(item["id"].lower()[:-len("-colibri")])
            if wanted in names:
                return item
        for row in self.rows():
            if row.get("into") and row.get("installed") and row["id"].lower() == wanted:
                return {"id": row["id"], "key": None, "addon": True, "size_bytes": row["installed"].get("size_bytes")}
        return None

    def plan(self, item):
        if item.get("addon"):
            try:
                plan = self.installer.addon_plan(item["id"])
            except self.mi.InstallError as error:
                raise Refused(error.message)
            return {"name": item["id"], "bytes": plan["bytes"], "where": plan["path"]}
        try:
            plan = self.installer.plan_uninstall(item)
        except self.mi.InstallError as error:
            raise Refused(error.message)
        return {"name": item["id"], "bytes": plan["bytes"], "where": plan["path"]}

    def remove(self, item):
        try:
            if item.get("addon"):
                return self.installer.uninstall_addon(item["id"], confirm=True)
            return self.installer.uninstall(item, confirm=True)
        except self.mi.InstallError as error:
            raise Refused(error.message)


# ---------------------------------------------------------------- the commands
def table(rows):
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  " + "   ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip())


def state_word(row):
    if row.get("job") and row["job"].get("state") in ("starting", "downloading", "finishing"):
        return "installing"
    if row.get("installed"):
        return "installed"
    if row.get("unfinished"):
        return f"partial ({human(row.get('partial_bytes'))})"
    return "-"


def show_available(backend, C):
    rows = backend.rows()
    lines = [("ID", "KIND", "SIZE", "FITS", "LICENCE", "HERE")]
    for row in rows:
        lines.append((row["id"], row["kind"], f"{row['download_gb']:.1f} GB", "yes" if row["fits"] else row["status"],
                      row.get("license_note") or "-", state_word(row)))
    table(lines)
    notes = [r for r in rows if not r["fits"] and not r.get("installed")]
    if notes:
        print()
        for row in notes[:6]:
            print(f"  {C.dim}{row['id']}: {row['reason']}{C.r}")
    folders = backend.folders()
    if len(folders) > 1:
        print(f"\n  models folders (an install goes to the first with room; --dest FOLDER chooses):")
        for f in folders:
            room = f"{f['free_gb']:.0f} GB free" if f.get("free_gb") is not None else "free space unknown"
            print(f"    {f['path']}  ({room}){'  <- default' if f.get('preferred') else ''}")
    print(f"\n  {C.dim}fetch one:  coli model install ID [--dest FOLDER]      remove one:  coli model uninstall NAME{C.r}")
    return 0


def ask(question, default, assume_yes, interactive):
    if assume_yes:
        return True
    if not interactive:
        raise Refused("this needs an answer and there is no terminal: add --yes to go ahead")
    try:
        answer = input(f"  {question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    except EOFError:                       # nobody is there (a pipe, a null stdin that Windows calls a terminal): not a yes
        raise Refused("there is nobody to ask: add --yes to go ahead")
    return default if not answer else answer.startswith("y")


def progress_line(job):
    total = job.get("total") or 0
    done = job.get("done") or 0
    pct = 100.0 * done / total if total else 0.0
    filled = int(24 * pct / 100)
    rate = job.get("rate")
    speed = f"{human(rate)}/s" if rate else "..."
    return (f"  [{'#' * filled}{'.' * (24 - filled)}] {pct:5.1f}%  {human(done)}/{human(total)}  {speed}  "
            f"{eta_text(job.get('eta_seconds'))} left")


def follow(backend, model_id, C, tty, poll=1.0, write=sys.stdout.write):
    """Draw the job until it ends. Ctrl-C cancels it (the part fetched is kept). Returns the final job."""
    last = ""
    try:
        while True:
            job = backend.job(model_id)
            if job is None:
                raise Refused("the install vanished from the list")
            if job["state"] in ("done", "failed", "cancelled"):
                if tty and last:
                    write("\n")
                return job
            line = progress_line(job) if job["state"] == "downloading" else f"  {job['state']}..."
            if tty:
                write("\r" + line + " " * max(0, len(last) - len(line)))
                sys.stdout.flush()
            elif line[:12] != last[:12]:
                print(line, flush=True)
            last = line
            time.sleep(poll)
    except KeyboardInterrupt:
        if tty:
            write("\n")
        print("  cancelling; what was fetched is kept")
        try:
            backend.cancel(model_id)
            end = time.time() + 15
            while time.time() < end:
                job = backend.job(model_id)
                if job and job["state"] in ("done", "failed", "cancelled"):
                    return job
                time.sleep(0.3)
        except Refused:
            pass
        return {"state": "cancelled", "error": "cancelled"}


def install(backend, a, C, interactive, tty=False, poll=1.0):
    if not a.name:
        raise Refused("name a model:  coli model install ID   (coli model available lists them)")
    rows = backend.rows()
    row = next((r for r in rows if r["id"].lower() == a.name.lower()), None)
    if row is None:
        raise Refused(f"no model called {a.name} in the catalog: " + ", ".join(r["id"] for r in rows))
    if row.get("installed"):
        print(f"  {C.grn}✓{C.r} {row['name']} is already installed")
        return 0
    print(f"  {row['name']}  ({row['kind']})")
    print(f"    download  {row['download_gb']:.1f} GB" + (f"   ({human(row.get('partial_bytes'))} already here)" if row.get("unfinished") else ""))
    print(f"    licence   {row.get('license_note') or 'not stated in the catalog; see its page on Hugging Face'}")
    print(f"    this PC   {row['reason']}")
    dest = getattr(a, "dest", None)
    if row["status"] == "unsupported" or (row["status"] == "needs-disk" and not dest):
        raise Refused(row["reason"])           # with --dest the folder named is the one that has to have room: the server says
    if row["status"] == "needs-ram" and not ask("It needs more RAM than this machine has. Install anyway?", False, a.yes, interactive):
        return 1
    if not ask(f"Fetch {row['download_gb']:.1f} GB and accept the licence above?", True, a.yes, interactive):
        print("  not installed")
        return 1
    backend.start(row["id"], dest) if dest else backend.start(row["id"])
    job = follow(backend, row["id"], C, tty, poll=poll)
    if job["state"] == "done":
        print(f"  {C.grn}✓{C.r} {row['name']} installed in {job.get('dir', '')}")
        if backend.name == "router":
            for item in backend.installed():
                if item["id"].lower().startswith(row["id"].lower()) and item.get("reason"):
                    print(f"  {C.yel}!{C.r} it cannot start yet: {item['reason']}")
        return 0
    print(f"  {C.yel}{'cancelled' if job['state'] == 'cancelled' else 'failed'}:{C.r} {job.get('error')}")
    return 1


def uninstall(backend, a, C, interactive):
    if not a.name:
        raise Refused("name a model:  coli model uninstall NAME")
    item = backend.find(a.name)
    if item is None:
        raise Refused(f"no installed model is called {a.name}")
    plan = backend.plan(item)
    size = human(plan["bytes"]) if plan.get("bytes") else "its files"
    print(f"  {plan['name']}" + (f"  ({plan['where']})" if plan.get("where") else ""))
    print(f"    removing it frees {size}; it can be fetched again with coli model install")
    if not ask("Remove it?", False, a.yes, interactive):
        print("  kept")
        return 1
    result = backend.remove(item)
    print(f"  {C.grn}✓{C.r} {plan['name']} removed" + (f", {human(result['freed_bytes'])} freed" if result.get("freed_bytes") else ""))
    if result.get("presets"):
        print(f"  {C.yel}!{C.r} presets that name it: {', '.join(result['presets'])} (they stay; they say the model is not installed)")
    return 0


def run(a, api, C, *, local_ok=True, backend=None, interactive=None, tty=None, poll=1.0):
    """The three actions. `api(method, path, body)` is the router's API (status, json); None status means nothing answers."""
    interactive = sys.stdin.isatty() if interactive is None else interactive
    tty = sys.stdout.isatty() if tty is None else tty
    try:
        if backend is None:
            status, _body = api("GET", "/v1/models/catalog", None)
            if status == 200:
                backend = RemoteBackend(api)
            elif status == 401:
                raise Refused("the server wants an API key: --api-key, or COLI_API_KEY")
            elif local_ok:
                if status == 404:
                    print(f"  {C.dim}the server here is not a router with install routes; working on this machine's folders{C.r}")
                backend = LocalBackend()
            else:
                raise Refused("no router with install routes answers there" if status == 404 else "no server answers there")
        if a.action == "available":
            return show_available(backend, C)
        if a.action == "install":
            return install(backend, a, C, interactive, tty, poll)
        return uninstall(backend, a, C, interactive)
    except Refused as error:
        print(f"  {C.yel}{error}{C.r}")
        return 1
