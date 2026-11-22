"""Install and uninstall models while colibri runs (docs/model-install.md).

`coli setup` fetches one model, builds its engine and starts a server. This is the other half of a machine that
keeps several models: fetch another one, or remove one, with the router (or `coli model`) already up. It reuses the
pieces setup already has: the catalog (setup_catalog.py, what can be fetched and what a machine fits), the resumable
verified downloader (setup_download.py) and the scan that lists what is installed (model_switch.py).

What it adds is the care a request needs that a person at a terminal does not:

- a download is a background job with progress and a cancel; a cancelled or failed one keeps its partial files, and
  asking again resumes them;
- one install at a time, and a lock file in the model's folder, so two colibri processes never write the same files;
- an uninstall only removes a folder that is a direct child of a models folder, never a link, never one a download is
  writing, and renames it out of the scan before the files go, so a router never lists a half-deleted model;
- nothing is fetched or removed without `confirm`: the caller has shown the size and the licence.

The router calls this from its handlers (/v1/models/catalog, /install, /installs, /uninstall); `coli model` calls it
directly when no router answers.
"""
import json
import os
import shutil
import stat
import sys
import threading
import time

import setup_catalog
import setup_download
import setup_hw

GB = 1_000_000_000
LOCK_NAME = ".colibri-install.lock"
DELETING = ".deleting"                  # a folder on its way out ends with this (model_switch skips it)
KEEP_JOBS = 8                           # finished jobs remembered for the list


class InstallError(Exception):
    """A refusal with an HTTP status, a short code and words for the person."""

    def __init__(self, status, message, code, param=None):
        super().__init__(message)
        self.status, self.message, self.code, self.param = status, message, code, param


class Cancelled(Exception):
    """Raised inside the downloader's progress callback to stop a job; the partial files stay."""


def _pid_alive(pid):
    try:
        import setup_flow
        return bool(setup_flow._pid_alive(pid))
    except Exception:
        return False


def _real(path):
    return os.path.normcase(os.path.realpath(path))


# ---------------------------------------------------------------- lock

def take_lock(model_dir, what="install"):
    """Mark `model_dir` as being written by this process. Returns the lock's path; raises InstallError when another live
    process holds it. A lock left by a process that is gone is taken over."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, LOCK_NAME)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = lock_holder(model_dir)
            if holder and holder != os.getpid() and _pid_alive(holder):
                raise InstallError(409, f"another colibri process (pid {holder}) is working in this folder",
                                   "model_busy")
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "what": what, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, handle)
        return path
    raise InstallError(409, "the folder is locked", "model_busy")


def lock_holder(model_dir):
    """The pid in the folder's lock, or None."""
    try:
        with open(os.path.join(model_dir, LOCK_NAME), encoding="utf-8") as handle:
            return int(json.load(handle).get("pid"))
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def release_lock(model_dir):
    if lock_holder(model_dir) == os.getpid():
        try:
            os.remove(os.path.join(model_dir, LOCK_NAME))
        except OSError:
            pass


# ---------------------------------------------------------------- what is installed, against the catalog

def match_installed(entry, installed):
    """The installed folder that is this catalog model, or None. By the folder's name (what `coli setup` makes it)
    or by the repository the downloader recorded in the folder."""
    for item in installed:
        path = item.get("path") or ""
        if os.path.basename(path.rstrip("/\\")) == entry.id:
            return item
    for item in installed:
        manifest = setup_download.read_manifest(item["path"]) if item.get("path") else None
        if manifest and manifest.get("repo") == entry.repo:
            return item
    return None


def addon_parent(entry, installed, catalog):
    """The installed folder an add-on goes into (the catalog model `entry.into`, finished), or None."""
    parent = next((e for e in catalog if e.id == entry.into), None)
    if parent is None:
        return None
    here = match_installed(parent, installed)
    return here if here is not None and not here.get("unfinished") else None


def addon_here(entry, parent_path):
    """How an add-on stands in its parent's folder: (folder, state) with state "installed", "partial" or None. A folder
    that holds the files `check` names is installed however it got there; one with only a manifest or `.part` files is
    a download that stopped."""
    folder = os.path.join(parent_path, entry.subdir)
    if not os.path.isdir(folder):
        return folder, None
    manifest = setup_download.read_manifest(folder)
    if manifest and manifest.get("complete") and setup_download.is_complete(folder):
        return folder, "installed"
    if entry.check and all(os.path.isfile(os.path.join(folder, name)) for name in entry.check) and not manifest:
        return folder, "installed"
    if manifest or any(n.endswith(".part") for n in os.listdir(folder)):
        return folder, "partial"
    return folder, None


def folder_bytes(path):
    """What a folder holds, in bytes (files only, links not followed)."""
    total = 0
    for folder, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.lstat(os.path.join(folder, name)).st_size
            except OSError:
                continue
    return total


def _on_rm_error(func, path, _exc):
    """shutil.rmtree's handler: a read-only file (a git checkout, a Windows download) is made writable and retried."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        raise


def remove_tree(path):
    # Python 3.12 renamed the handler argument; both spellings are the same handler
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_rm_error)
    else:
        shutil.rmtree(path, onerror=_on_rm_error)


class Installer:
    """The installs and uninstalls of one process.

    roots()     the models folders, in the order an install prefers them (the folders the catalog scans)
    installed() what is installed now: entries with `key`, `path`, `id`, `name`, `unfinished`
    refresh()   called after a change, so the next listing sees it
    launcher_dir where tools/ lives, for a model's post-install steps"""

    def __init__(self, roots, installed, refresh=None, launcher_dir=None, catalog=None, memory=None, on_addon=None):
        self.roots, self.installed, self.refresh = roots, installed, refresh or (lambda: None)
        self.on_addon = on_addon                 # called with the parent's installed entry once an add-on has landed
        self.launcher_dir = launcher_dir or os.path.dirname(os.path.abspath(__file__))
        self._catalog = catalog or setup_catalog.catalog
        self._memory = memory or (lambda: setup_hw.detect_memory().get("total"))
        self.lock = threading.Lock()
        self._start_lock = threading.Lock()
        self.jobs = {}                       # catalog id -> job dict
        self.order = []
        self.cancel = {}                     # catalog id -> threading.Event

    # ---- listing
    def install_root(self, need_bytes=0):
        """Where a new model goes: the first models folder with room, else the first one (made if there is none)."""
        roots = [r for r in self.roots() if r]
        if not roots:
            roots = [os.path.join(os.path.expanduser("~"), "colibri-models")]
        for root in roots:
            free = setup_hw.disk_free(root)
            if free is None or free >= need_bytes:
                return root
        return None

    def catalog_rows(self):
        """GET /v1/models/catalog: every model that can be fetched, what a machine like this makes of it, whether it is
        here, and the job that is fetching it."""
        installed = self.installed()
        ram = self._memory()
        rows = []
        catalog = list(self._catalog())
        for entry in catalog:
            if entry.into:
                rows.append(self._addon_row(entry, installed, catalog, ram))
                continue
            here = match_installed(entry, installed)
            have = 0
            if here is not None:
                manifest = setup_download.read_manifest(here["path"])
                have = setup_download.bytes_present(here["path"], manifest.get("files", [])) if manifest else 0
            root = self.install_root() if here is None else os.path.dirname(here["path"])
            status, reason = setup_catalog.assess(entry, ram, setup_hw.disk_free(root) if root else None,
                                                  os_name=sys.platform, machine=setup_hw.platform.machine().lower(),
                                                  already_downloaded=have)
            row = {"id": entry.id, "name": entry.name, "kind": entry.modality, "family": entry.family,
                   "repo": entry.repo, "download_gb": entry.disk_gb, "ram_min_gb": entry.ram_min_gb,
                   "ram_good_gb": entry.ram_good_gb, "summary": entry.summary, "license_note": entry.license_note,
                   "doc": entry.doc, "status": status, "fits": status in ("good", "tight"), "reason": reason,
                   "installed": None, "unfinished": False, "job": self.job(entry.id)}
            if here is not None:
                row["installed"] = {"key": here["key"], "id": here["id"], "state": here.get("state", "idle"),
                                    "size_bytes": here.get("size_bytes")}
                row["unfinished"] = bool(here.get("unfinished"))
                if row["unfinished"]:
                    row["installed"] = None
                    row["partial_bytes"] = have
            rows.append(row)
        rows.sort(key=lambda r: (r["installed"] is None, not r["fits"], r["kind"] != "text", r["download_gb"]))
        return rows

    def _addon_row(self, entry, installed, catalog, ram):
        """A catalog row for an add-on: it needs its parent model here, and is installed when its folder in the parent is."""
        parent = addon_parent(entry, installed, catalog)
        row = {"id": entry.id, "name": entry.name, "kind": entry.modality, "family": entry.family, "repo": entry.repo,
               "download_gb": entry.disk_gb, "ram_min_gb": entry.ram_min_gb, "ram_good_gb": entry.ram_good_gb,
               "summary": entry.summary, "license_note": entry.license_note, "doc": entry.doc, "into": entry.into,
               "installed": None, "unfinished": False, "job": self.job(entry.id)}
        if parent is None:
            row.update(status="needs-parent", fits=False,
                       reason=f"goes into {entry.into}, which is not installed: coli model install {entry.into}")
            return row
        folder, state = addon_here(entry, parent["path"])
        have = 0
        if state is not None:
            manifest = setup_download.read_manifest(folder)
            have = setup_download.bytes_present(folder, manifest.get("files", [])) if manifest else 0
        status, reason = setup_catalog.assess(entry, ram, setup_hw.disk_free(parent["path"]),
                                              os_name=sys.platform, machine=setup_hw.platform.machine().lower(),
                                              already_downloaded=have)
        row.update(status=status, fits=status in ("good", "tight"), reason=reason)
        if state == "installed":
            row["installed"] = {"key": None, "id": entry.id, "state": "idle", "size_bytes": folder_bytes(folder)}
        elif state == "partial":
            row["unfinished"] = True
            row["partial_bytes"] = have
        return row

    # ---- jobs
    def job(self, model_id):
        with self.lock:
            job = self.jobs.get(model_id)
            return dict(job) if job else None

    def job_list(self):
        with self.lock:
            return [dict(self.jobs[i]) for i in reversed(self.order) if i in self.jobs]

    def active(self):
        with self.lock:
            return next((dict(j) for j in self.jobs.values() if j["state"] in ("starting", "downloading", "finishing")), None)

    def _update(self, model_id, **fields):
        with self.lock:
            job = self.jobs.get(model_id)
            if job is not None:
                job.update(fields)

    def start(self, model_id, confirm=False, background=True):
        """Begin fetching a catalog model. Returns the job. Raises InstallError."""
        with self._start_lock:                 # the check for a running install and the registering of this one are one step
            entry, model_dir = self._begin(model_id, confirm)
        if background:
            threading.Thread(target=self._run, args=(entry, model_dir), name=f"install-{entry.id}", daemon=True).start()
        else:
            self._run(entry, model_dir)
        return self.job(entry.id)

    def _begin(self, model_id, confirm):
        """Everything that can refuse an install, then the job and the folder's lock. Returns (entry, folder)."""
        if confirm is not True:
            raise InstallError(400, "Send `confirm`: true once the size and the licence have been shown.",
                               "confirmation_required", "confirm")
        try:
            entry = next(e for e in self._catalog() if e.id == model_id)
        except StopIteration:
            raise InstallError(404, f"No model called `{model_id}` is in the catalog.", "model_not_found", "id")
        installed = self.installed()
        addon = None
        here = None
        if entry.into:
            parent = addon_parent(entry, installed, list(self._catalog()))
            if parent is None:
                raise InstallError(409, f"{entry.name} goes into {entry.into}, which is not installed; install that first.",
                                   "needs_parent", "id")
            addon = addon_here(entry, parent["path"])
            if addon[1] == "installed":
                raise InstallError(409, f"{entry.name} is already installed.", "already_installed", "id")
        else:
            here = match_installed(entry, installed)
            if here is not None and not here.get("unfinished"):
                raise InstallError(409, f"{entry.name} is already installed.", "already_installed", "id")
        running = self.active()
        if running is not None:
            raise InstallError(409, f"{running['name']} is being downloaded; one install at a time.", "install_busy")
        need = int(entry.disk_gb * GB * 1.02 + 2 * GB)
        if addon is not None:
            model_dir = addon[0]
            manifest = setup_download.read_manifest(model_dir)
            if manifest:
                need = max(0, need - setup_download.bytes_present(model_dir, manifest.get("files", [])))
            elif addon[1] is None and os.path.isdir(model_dir) and os.listdir(model_dir):
                raise InstallError(409, f"{model_dir} exists and is not a download of this model; move or remove it.",
                                   "folder_exists")
            free = setup_hw.disk_free(parent["path"])
            if free is not None and free < need:
                raise InstallError(409, f"{entry.name} needs {need / GB:.0f} GB free; {free / GB:.0f} GB is free there.",
                                   "insufficient_disk")
        elif here is not None:
            model_dir = here["path"]
            manifest = setup_download.read_manifest(model_dir) or {}
            need = max(0, need - setup_download.bytes_present(model_dir, manifest.get("files", [])))
        else:
            root = self.install_root(need)
            if root is None:
                free = max((setup_hw.disk_free(r) or 0) for r in (self.roots() or [os.path.expanduser("~")]))
                raise InstallError(409, f"{entry.name} needs {need / GB:.0f} GB free; the most any models folder has is "
                                        f"{free / GB:.0f} GB.", "insufficient_disk")
            model_dir = os.path.join(root, entry.id)
            if os.path.isdir(model_dir) and os.listdir(model_dir) and not setup_download.read_manifest(model_dir):
                raise InstallError(409, f"{model_dir} exists and is not a download of this model; move or remove it.",
                                   "folder_exists")
        take_lock(model_dir)                          # raises when another process is downloading it
        job = {"id": entry.id, "name": entry.name, "state": "starting", "done": 0, "total": int(entry.disk_gb * GB),
               "rate": None, "eta_seconds": None, "file": None, "error": None, "dir": model_dir,
               "started": time.time(), "finished": None, "license_note": entry.license_note}
        with self.lock:
            self.jobs[entry.id] = job
            if entry.id in self.order:
                self.order.remove(entry.id)
            self.order.append(entry.id)
            del self.order[:-KEEP_JOBS]
            self.cancel[entry.id] = threading.Event()
        return entry, model_dir

    def cancel_job(self, model_id):
        with self.lock:
            job = self.jobs.get(model_id)
            event = self.cancel.get(model_id)
            if job is None or job["state"] not in ("starting", "downloading", "finishing") or event is None:
                raise InstallError(404, f"`{model_id}` is not being installed.", "not_installing", "id")
            event.set()
            return dict(job)

    def _run(self, entry, model_dir):
        stop = self.cancel[entry.id]
        started = time.time()
        first = {"done": None}
        last = {"t": 0.0}
        try:
            token = setup_download.hf_token()
            files = setup_download.list_repo_files(entry.repo, entry.revision, include=entry.include,
                                                   exclude=entry.exclude, token=token)
            if entry.strip:                           # the add-on's folder is the repository's subfolder: its paths start there
                files = [dict(f, remote=f["path"], path=f["path"][len(entry.strip):])
                         for f in files if f["path"].startswith(entry.strip)]
            if not files:
                raise setup_download.DownloadError(f"{entry.repo}: the repository lists no files")
            total = sum(f["size"] for f in files)
            have = setup_download.bytes_present(model_dir, files)
            free = setup_hw.disk_free(model_dir)
            if free is not None and free < (total - have) + 1 * GB:
                raise setup_download.DownloadError(
                    f"{(total - have) / GB:.0f} GB still to download, {free / GB:.0f} GB free in {os.path.dirname(model_dir)}")
            self._update(entry.id, state="downloading", total=total, done=have)

            def progress(event):
                if stop.is_set():
                    raise Cancelled()
                now = time.time()
                if first["done"] is None:
                    first["done"] = event["done"]
                if now - last["t"] < 0.5 and event["done"] < event["total"]:
                    return
                last["t"] = now
                rate = (event["done"] - first["done"]) / max(now - started, 1e-3)
                eta = (event["total"] - event["done"]) / rate if rate > 0 else None
                self._update(entry.id, done=event["done"], total=event["total"], file=event["file"],
                             rate=rate, eta_seconds=eta)

            setup_download.download_repo(entry.repo, entry.revision, model_dir, files, token=token, progress=progress)
            self._update(entry.id, state="finishing", done=total, file=None, eta_seconds=0)
            self._post_install(entry, model_dir)
            self._update(entry.id, state="done", finished=time.time(), rate=None, eta_seconds=None)
            if entry.into and self.on_addon is not None:
                try:
                    self.on_addon(addon_parent(entry, self.installed(), list(self._catalog())))
                except Exception:                     # the install is done; a failed courtesy must not undo that
                    pass
        except Cancelled:
            self._update(entry.id, state="cancelled", finished=time.time(), rate=None, eta_seconds=None,
                         error="cancelled; the part that was fetched is kept, install again to continue")
        except Exception as error:                    # the job says why; the router must not die of it
            self._update(entry.id, state="failed", finished=time.time(), rate=None, eta_seconds=None,
                         error=str(error) or error.__class__.__name__)
        finally:
            release_lock(model_dir)
            try:
                self.refresh()
            except Exception:
                pass

    def _post_install(self, entry, model_dir):
        """The catalog's steps after the files are down (a tokenizer file a checkpoint lacks)."""
        if not entry.post_install:
            return
        import setup_flow

        class _Say:
            def say(self, text=""):
                pass

        setup_flow.run_post_install(entry, self.launcher_dir, model_dir, _Say())

    # ---- uninstall
    def plan_uninstall(self, item):
        """What removing an installed entry would do, or the InstallError that says why not. Does not touch anything."""
        path = item.get("path")
        if not path or item.get("missing"):
            raise InstallError(409, "That model is not on this machine (a drive that is unplugged?); there is nothing to remove.",
                               "model_missing")
        real = _real(path)
        parent = os.path.dirname(real)
        roots = [_real(r) for r in self.roots() if r]
        if parent not in roots:
            raise InstallError(409, f"{path} is not inside a models folder ({', '.join(self.roots()) or 'none'}); "
                                    "remove it by hand.", "outside_models_folder")
        if os.path.islink(path) or getattr(os.path, "isjunction", lambda p: False)(path):
            raise InstallError(409, f"{path} is a link; remove the link by hand.", "model_is_link")
        job = self.jobs.get(os.path.basename(path.rstrip("/\\")))
        holder = lock_holder(path)
        if (job and job["state"] in ("starting", "downloading", "finishing")) or \
                (holder and holder != os.getpid() and _pid_alive(holder)):
            raise InstallError(409, "A download is writing that folder; cancel it first.", "model_busy")
        return {"path": path, "bytes": folder_bytes(path), "name": item.get("name"), "unfinished": bool(item.get("unfinished"))}

    def uninstall(self, item, confirm=False):
        """Remove an installed model's folder. The caller has checked that no engine has it loaded."""
        if confirm is not True:
            raise InstallError(400, "Send `confirm`: true to remove the model's files.", "confirmation_required", "confirm")
        plan = self.plan_uninstall(item)
        path = plan["path"]
        gone = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.{os.getpid()}{DELETING}")
        try:
            os.rename(path, gone)            # out of the scan at once; refused when a process still holds files in it
        except OSError as error:
            raise InstallError(409, f"The files are in use ({error.strerror or error}); unload the model and try again.",
                               "model_in_use")
        try:
            remove_tree(gone)
        except OSError as error:
            raise InstallError(500, f"{plan['bytes'] / GB:.1f} GB of {path} were set aside but not all deleted "
                                    f"({error}); the leftovers are in {gone}.", "delete_failed")
        finally:
            self.refresh()
        return {"status": "removed", "path": path, "freed_bytes": plan["bytes"]}

    def is_addon(self, model_id):
        return any(e.id == model_id and e.into for e in self._catalog())

    def addon_plan(self, model_id):
        """What removing an add-on would do: {"id", "name", "path", "bytes", "parent"}, or the InstallError that says why not."""
        catalog = list(self._catalog())
        entry = next((e for e in catalog if e.id == model_id and e.into), None)
        if entry is None:
            raise InstallError(404, f"`{model_id}` is not an add-on of the catalog.", "model_not_found", "id")
        parent = addon_parent(entry, self.installed(), catalog)
        if parent is None:
            raise InstallError(409, f"{entry.name} is not installed.", "model_missing", "id")
        folder, state = addon_here(entry, parent["path"])
        if state is None:
            raise InstallError(409, f"{entry.name} is not installed.", "model_missing", "id")
        real_parent = _real(parent["path"])
        roots = [_real(r) for r in self.roots() if r]
        if os.path.dirname(real_parent) not in roots:
            raise InstallError(409, f"{parent['path']} is not inside a models folder; remove {entry.subdir} by hand.",
                               "outside_models_folder")
        if os.path.islink(folder) or getattr(os.path, "isjunction", lambda p: False)(folder) or os.path.islink(parent["path"]):
            raise InstallError(409, f"{folder} is a link; remove the link by hand.", "model_is_link")
        job = self.jobs.get(entry.id)
        holder = lock_holder(folder)
        if (job and job["state"] in ("starting", "downloading", "finishing")) or \
                (holder and holder != os.getpid() and _pid_alive(holder)):
            raise InstallError(409, "A download is writing that folder; cancel it first.", "model_busy")
        return {"id": entry.id, "name": entry.name, "path": folder, "bytes": folder_bytes(folder), "parent": parent}

    def uninstall_addon(self, model_id, confirm=False):
        """Remove an add-on's folder from its parent's. The caller has checked that no engine has the parent loaded."""
        if confirm is not True:
            raise InstallError(400, "Send `confirm`: true to remove the add-on's files.", "confirmation_required", "confirm")
        plan = self.addon_plan(model_id)
        path = plan["path"]
        gone = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.{os.getpid()}{DELETING}")
        try:
            os.rename(path, gone)
        except OSError as error:
            raise InstallError(409, f"The files are in use ({error.strerror or error}); unload the model and try again.",
                               "model_in_use")
        try:
            remove_tree(gone)
        except OSError as error:
            raise InstallError(500, f"{plan['bytes'] / GB:.1f} GB of {path} were set aside but not all deleted "
                                    f"({error}); the leftovers are in {gone}.", "delete_failed")
        finally:
            self.refresh()
        return {"status": "removed", "path": path, "freed_bytes": plan["bytes"]}

    def sweep(self):
        """Finish deletions that were interrupted (a folder renamed out of the scan and not removed)."""
        done = 0
        for root in self.roots():
            try:
                names = os.listdir(root)
            except OSError:
                continue
            for name in names:
                if name.endswith(DELETING):
                    try:
                        remove_tree(os.path.join(root, name))
                        done += 1
                    except OSError:
                        pass
        return done
