"""The Windows tray icon for colibri: see the server's state at a glance, open the dashboard, start, stop and restart it,
unload a model, and start with Windows (docs/tray.md).

The icon is a circle in the notification area: hollow grey when the server is stopped, green when it is up and idle, blue
when it is working, amber while it starts or stops, red when something is wrong. Click it for the menu.

What it does is `service.py`'s: this file is the menu and the window around it. The logic that decides what to show
(`summarize`, `build_menu`, `icon_pixels`) takes plain data and runs anywhere, so it is tested without a screen; the Win32
part below it (ctypes, nothing to install) runs on Windows only.

    coli tray               start it (detached) and return
    coli tray --install     start it whenever you sign in to Windows
    coli tray --remove      stop doing that
"""
import argparse
import math
import os
import subprocess
import sys
import threading
import time

import service

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- what to show (no window needed)

COLORS = {                       # the icon's colour for each kind of state
    "stopped": (138, 143, 152),
    "starting": (232, 160, 32),
    "idle": (46, 170, 90),
    "busy": (52, 120, 230),
    "attention": (222, 70, 60),
}


def gb(size):
    return f"{size / 2**30:.1f} GB" if size else ""


def duration(seconds):
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def plural(count, word):
    return f"{count} {word}" + ("" if count == 1 else "s")


def summarize(now, pending=None, port=8000):
    """What the tray shows for a `service.status`: the icon's `kind`, a one-line `title` (the menu's first line), the `tip`
    (the tooltip) and the loaded models. `pending` is a word such as "Starting" while this tray is doing that."""
    loaded = list((now or {}).get("loaded") or [])
    if pending:
        kind, title = "starting", f"{pending}..."
    elif now is None:
        kind, title = "starting", "Looking..."
    elif now["state"] == "stopped":
        kind, title = "stopped", "Stopped"
    elif now["state"] == "unreachable":
        kind, title = "attention", f"Port {port} is used by another program"
    elif not now["key_ok"]:
        kind, title = "attention", "Running, but it does not accept the saved key"
    elif any(w.get("state") == "starting" for w in loaded):
        kind, title = "busy", "Loading a model"
    elif now["busy"]:
        kind, title = "busy", f"Working: {plural(now['busy'], 'request')}"
    else:
        kind, title = "idle", "Running, idle"
    tip = "Colibri: " + title[0].lower() + title[1:]
    ready = [w for w in loaded if w.get("state") == "ready"]
    if ready and kind in ("idle", "busy"):
        tip += f" ({plural(len(ready), 'model')} loaded)"
    return {"kind": kind, "title": title, "tip": tip[:120], "loaded": loaded, "busy": (now or {}).get("busy", 0),
            "running": bool(now) and now["state"] == "running", "stopped": bool(now) and now["state"] == "stopped",
            "pending": bool(pending),
            "unsure": bool(now) and (now["state"] == "unreachable" or (now["state"] == "running" and not now["key_ok"]))}


def build_menu(summary, lan=False, autostart=False, log=False):
    """The menu as data: dicts with `id`, `label`, and `enabled` / `checked`, or {"sep": True}. `id` says what to do."""
    items = [{"id": None, "label": "Colibri: " + summary["title"], "enabled": False}, {"sep": True}]
    shown = [w for w in summary["loaded"] if w.get("state") in ("ready", "starting")]
    if shown:
        for worker in shown:
            if worker.get("state") == "starting":
                label = f"{worker.get('id')}: loading..."
            else:
                parts = [gb(worker.get("resident_bytes"))]
                if worker.get("inflight"):
                    parts.append("in use")
                elif worker.get("expires_in") is not None:
                    parts.append(f"unloads in {duration(worker['expires_in'])}")
                label = f"Unload {worker.get('id')}" + (f" ({', '.join(p for p in parts if p)})" if any(parts) else "")
            items.append({"id": f"unload:{worker.get('key')}" if worker.get("state") == "ready" else None,
                          "label": label, "enabled": worker.get("state") == "ready"})
    else:
        items.append({"id": None, "label": "No model loaded", "enabled": False})
    items += [{"sep": True},
              {"id": "open", "label": "Open the dashboard", "enabled": summary["running"]},
              {"id": "copy_lan", "label": "Copy the network link", "enabled": bool(lan)},
              {"id": "copy_key", "label": "Copy the API key", "enabled": True},
              {"sep": True}]
    idle = not summary["pending"]
    if summary["stopped"]:
        items.append({"id": "start", "label": "Start the server", "enabled": idle})
    elif summary["unsure"]:                     # it does not tell us whether it is working, so it is not stopped for us
        items.append({"id": "restart_force", "label": "Restart anyway (it does not say whether it is working)", "enabled": idle})
        items.append({"id": "stop_force", "label": "Stop anyway (it does not say whether it is working)", "enabled": idle})
    elif summary["busy"]:
        n = plural(summary["busy"], "request")
        items.append({"id": "restart_force", "label": f"Restart anyway ({n} running will be lost)", "enabled": idle})
        items.append({"id": "stop_force", "label": f"Stop anyway ({n} running will be lost)", "enabled": idle})
    else:
        items.append({"id": "restart", "label": "Restart the server", "enabled": idle and summary["kind"] != "starting"})
        items.append({"id": "stop", "label": "Stop the server", "enabled": idle})
    items += [{"sep": True},
              {"id": "request_log", "label": "Open the request log", "enabled": bool(log)},
              {"id": "logs", "label": "Open the folder with the server's logs", "enabled": True},
              {"id": "autostart", "label": "Start with Windows", "enabled": True, "checked": bool(autostart)},
              {"sep": True},
              {"id": "quit", "label": "Quit the tray (the server keeps running)", "enabled": True}]
    return items


def icon_pixels(kind, size=32):
    """The icon as `size` x `size` premultiplied BGRA bytes, top row first: a ring when stopped, a disc otherwise, with a
    white bar and dot in it when something is wrong (so the state is not told by colour alone)."""
    red, green, blue = COLORS[kind]
    center, radius = (size - 1) / 2, size / 2 - 1.5
    thickness = max(2.5, size / 9)
    out = bytearray()
    for y in range(size):
        for x in range(size):
            d = math.hypot(x - center, y - center)
            cover = min(1.0, max(0.0, radius - d + 0.5))                 # the outer edge, anti-aliased
            if kind == "stopped":
                cover = min(cover, min(1.0, max(0.0, d - (radius - thickness) + 0.5)))
            r, g, b = red, green, blue
            if kind == "attention":
                bar = abs(x - center) < size / 14 and size * 0.22 <= y <= size * 0.62
                dot = abs(x - center) < size / 14 and size * 0.72 <= y <= size * 0.82
                if bar or dot:
                    r = g = b = 255
            out += bytes((int(b * cover), int(g * cover), int(r * cover), int(255 * cover)))
    return bytes(out)


# ---------------------------------------------------------------- start with Windows

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "Colibri tray"


def windowless_python():
    exe = sys.executable
    if os.path.basename(exe).lower() == "python.exe":
        candidate = os.path.join(os.path.dirname(exe), "pythonw.exe")
        if os.path.exists(candidate):
            return candidate
    return exe


def autostart_command():
    return f'"{windowless_python()}" "{os.path.join(HERE, "tray.py")}" --here'


def autostart_get():
    """The command Windows runs at sign-in for this program, or None."""
    if sys.platform != "win32":
        return None
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            return winreg.QueryValueEx(key, RUN_VALUE)[0]
    except OSError:
        return None


def autostart_set(on):
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if on:
            winreg.SetValueEx(key, RUN_VALUE, 0, winreg.REG_SZ, autostart_command())
        else:
            try:
                winreg.DeleteValue(key, RUN_VALUE)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------- the window (Windows only)

WM_APP = 0x8000
WM_ICON, WM_STATE, WM_TOAST = WM_APP + 1, WM_APP + 2, WM_APP + 3
WM_DESTROY, WM_NULL, WM_LBUTTONUP, WM_RBUTTONUP = 0x0002, 0x0000, 0x0202, 0x0205
NIM_ADD, NIM_MODIFY, NIM_DELETE = 0, 1, 2
NIF_MESSAGE, NIF_ICON, NIF_TIP, NIF_INFO = 1, 2, 4, 0x10
MF_GRAYED, MF_CHECKED, MF_SEPARATOR = 1, 8, 0x800
TPM_RIGHTBUTTON, TPM_RETURNCMD, TPM_BOTTOMALIGN = 0x2, 0x100, 0x20
ERROR_ALREADY_EXISTS = 183


def _win32():
    """The Win32 calls the tray makes, with their types (a 64-bit handle through the default int type is cut in half)."""
    import ctypes
    from ctypes import wintypes as w

    class Api:
        pass
    api = Api()
    api.ctypes, api.w = ctypes, w
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    api.user32, api.shell32, api.kernel32, api.gdi32 = user32, shell32, kernel32, gdi32
    api.LRESULT = ctypes.c_ssize_t
    api.WNDPROC = ctypes.WINFUNCTYPE(api.LRESULT, w.HWND, w.UINT, w.WPARAM, w.LPARAM)

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [("style", w.UINT), ("lpfnWndProc", api.WNDPROC), ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int), ("hInstance", w.HINSTANCE), ("hIcon", w.HICON),
                    ("hCursor", w.HANDLE), ("hbrBackground", w.HBRUSH), ("lpszMenuName", w.LPCWSTR),
                    ("lpszClassName", w.LPCWSTR)]

    class NOTIFYICONDATAW(ctypes.Structure):
        _fields_ = [("cbSize", w.DWORD), ("hWnd", w.HWND), ("uID", w.UINT), ("uFlags", w.UINT),
                    ("uCallbackMessage", w.UINT), ("hIcon", w.HICON), ("szTip", w.WCHAR * 128),
                    ("dwState", w.DWORD), ("dwStateMask", w.DWORD), ("szInfo", w.WCHAR * 256),
                    ("uTimeout", w.UINT), ("szInfoTitle", w.WCHAR * 64), ("dwInfoFlags", w.DWORD),
                    ("guidItem", ctypes.c_ubyte * 16), ("hBalloonIcon", w.HICON)]

    class ICONINFO(ctypes.Structure):
        _fields_ = [("fIcon", w.BOOL), ("xHotspot", w.DWORD), ("yHotspot", w.DWORD),
                    ("hbmMask", w.HBITMAP), ("hbmColor", w.HBITMAP)]
    api.WNDCLASSW, api.NOTIFYICONDATAW, api.ICONINFO = WNDCLASSW, NOTIFYICONDATAW, ICONINFO

    def sig(function, result, *args):
        function.restype, function.argtypes = result, list(args)
    sig(user32.DefWindowProcW, api.LRESULT, w.HWND, w.UINT, w.WPARAM, w.LPARAM)
    sig(user32.RegisterClassW, w.ATOM, ctypes.POINTER(WNDCLASSW))
    sig(user32.CreateWindowExW, w.HWND, w.DWORD, w.LPCWSTR, w.LPCWSTR, w.DWORD, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, w.HWND, w.HMENU, w.HINSTANCE, w.LPVOID)
    sig(user32.DestroyWindow, w.BOOL, w.HWND)
    sig(user32.GetMessageW, w.BOOL, ctypes.POINTER(w.MSG), w.HWND, w.UINT, w.UINT)
    sig(user32.TranslateMessage, w.BOOL, ctypes.POINTER(w.MSG))
    sig(user32.DispatchMessageW, api.LRESULT, ctypes.POINTER(w.MSG))
    sig(user32.PostMessageW, w.BOOL, w.HWND, w.UINT, w.WPARAM, w.LPARAM)
    sig(user32.PostQuitMessage, None, ctypes.c_int)
    sig(user32.RegisterWindowMessageW, w.UINT, w.LPCWSTR)
    sig(user32.CreatePopupMenu, w.HMENU)
    sig(user32.AppendMenuW, w.BOOL, w.HMENU, w.UINT, ctypes.c_size_t, w.LPCWSTR)
    sig(user32.GetMenuItemCount, ctypes.c_int, w.HMENU)
    sig(user32.TrackPopupMenu, w.BOOL, w.HMENU, w.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int, w.HWND, ctypes.c_void_p)
    sig(user32.DestroyMenu, w.BOOL, w.HMENU)
    sig(user32.GetCursorPos, w.BOOL, ctypes.POINTER(w.POINT))
    sig(user32.SetForegroundWindow, w.BOOL, w.HWND)
    sig(user32.CreateIconIndirect, w.HICON, ctypes.POINTER(ICONINFO))
    sig(user32.DestroyIcon, w.BOOL, w.HICON)
    sig(shell32.Shell_NotifyIconW, w.BOOL, w.DWORD, ctypes.POINTER(NOTIFYICONDATAW))
    sig(gdi32.CreateBitmap, w.HBITMAP, ctypes.c_int, ctypes.c_int, w.UINT, w.UINT, ctypes.c_void_p)
    sig(gdi32.DeleteObject, w.BOOL, w.HGDIOBJ)
    sig(kernel32.GetModuleHandleW, w.HMODULE, w.LPCWSTR)
    sig(kernel32.CreateMutexW, w.HANDLE, ctypes.c_void_p, w.BOOL, w.LPCWSTR)
    return api


class Tray:
    """The icon, its menu, and the thread that watches the server. `run()` returns when the user quits."""

    def __init__(self, cfg, interval=3.0):
        if sys.platform != "win32":
            raise service.ServiceError("the tray is for Windows; on other systems use `coli service`")
        self.cfg, self.interval = cfg, interval
        self.api = _win32()
        self.now = None                    # the last service.status
        self.pending = None                # "Starting" while an action of ours runs
        self.summary = summarize(None, None, cfg["port"])
        self.toasts = []
        self.icons = {}
        self.hwnd = None
        self.quitting = threading.Event()
        self.expected_stop = False
        self.lock = threading.Lock()
        self._proc = self.api.WNDPROC(self._wndproc)           # kept: ctypes does not own it, and a freed one crashes

    # -- the notification area
    def _data(self, flags):
        a = self.api
        data = a.NOTIFYICONDATAW()
        data.cbSize, data.hWnd, data.uID, data.uFlags = a.ctypes.sizeof(a.NOTIFYICONDATAW), self.hwnd, 1, flags
        return data

    def _icon(self, kind):
        a = self.api
        if kind not in self.icons:
            pixels = icon_pixels(kind)
            color = a.gdi32.CreateBitmap(32, 32, 1, 32, pixels)
            mask = a.gdi32.CreateBitmap(32, 32, 1, 1, bytes(128))
            info = a.ICONINFO(True, 0, 0, mask, color)
            self.icons[kind] = a.user32.CreateIconIndirect(a.ctypes.byref(info))
            a.gdi32.DeleteObject(color)
            a.gdi32.DeleteObject(mask)
        return self.icons[kind]

    def add_icon(self):
        data = self._data(NIF_MESSAGE | NIF_ICON | NIF_TIP)
        data.uCallbackMessage, data.hIcon, data.szTip = WM_ICON, self._icon(self.summary["kind"]), self.summary["tip"]
        return bool(self.api.shell32.Shell_NotifyIconW(NIM_ADD, self.api.ctypes.byref(data)))

    def update_icon(self):
        data = self._data(NIF_ICON | NIF_TIP)
        data.hIcon, data.szTip = self._icon(self.summary["kind"]), self.summary["tip"]
        return bool(self.api.shell32.Shell_NotifyIconW(NIM_MODIFY, self.api.ctypes.byref(data)))

    def show_toast(self, title, text):
        data = self._data(NIF_INFO)
        data.szInfoTitle, data.szInfo, data.dwInfoFlags = title[:63], text[:255], 1
        return bool(self.api.shell32.Shell_NotifyIconW(NIM_MODIFY, self.api.ctypes.byref(data)))

    def remove_icon(self):
        self.api.shell32.Shell_NotifyIconW(NIM_DELETE, self.api.ctypes.byref(self._data(0)))

    def toast(self, text, title="Colibri"):
        """From any thread: show a balloon."""
        with self.lock:
            self.toasts.append((title, text))
        if self.hwnd:
            self.api.user32.PostMessageW(self.hwnd, WM_TOAST, 0, 0)

    # -- the menu
    def menu(self):
        return build_menu(self.summary, lan=bool(service.lan_address() and self.cfg["lan"]),
                          autostart=autostart_get() is not None,
                          log=bool((service.paths(self.cfg)["request_log"] or "")
                                   and os.path.exists(service.paths(self.cfg)["request_log"])))

    def native_menu(self, items):
        """(menu handle, {numeric id: action}) for `items`."""
        u = self.api.user32
        handle, actions = u.CreatePopupMenu(), {}
        for number, item in enumerate(items, 1000):
            if item.get("sep"):
                u.AppendMenuW(handle, MF_SEPARATOR, 0, None)
                continue
            flags = (0 if item["enabled"] else MF_GRAYED) | (MF_CHECKED if item.get("checked") else 0)
            u.AppendMenuW(handle, flags, number, item["label"])
            if item["id"]:
                actions[number] = item["id"]
        return handle, actions

    def show_menu(self):
        a = self.api
        handle, actions = self.native_menu(self.menu())
        point = a.w.POINT()
        a.user32.GetCursorPos(a.ctypes.byref(point))
        a.user32.SetForegroundWindow(self.hwnd)                  # without it the menu stays open when you click elsewhere
        chosen = a.user32.TrackPopupMenu(handle, TPM_RETURNCMD | TPM_RIGHTBUTTON | TPM_BOTTOMALIGN, point.x, point.y, 0,
                                         self.hwnd, None)
        a.user32.PostMessageW(self.hwnd, WM_NULL, 0, 0)
        a.user32.DestroyMenu(handle)
        if chosen in actions:
            self.act(actions[chosen])

    # -- actions
    def act(self, action):
        """Do what a menu item says. Anything that takes time runs in a thread so the menu never freezes."""
        if action == "quit":
            self.api.user32.DestroyWindow(self.hwnd)
        elif action == "autostart":
            self.set_autostart(autostart_get() is None)
        elif action == "open":
            self.open_dashboard()
        elif action in ("copy_key", "copy_lan"):
            self.copy(action)
        elif action in ("request_log", "logs"):
            self.open_file(action)
        else:
            threading.Thread(target=self._server_action, args=(action,), daemon=True).start()

    def open_dashboard(self):
        import webbrowser
        key = service.read_key(self.cfg)
        webbrowser.open(service.links(self.cfg, key, lan="")["local"] if key else f"http://127.0.0.1:{self.cfg['port']}/")

    def copy(self, action):
        key = service.read_key(self.cfg)
        text = key if action == "copy_key" else service.links(self.cfg, key)["lan"]
        if not key or not text:
            self.toast("There is no key yet: start the server first.")
            return
        subprocess.run(["clip"], input=text.encode("ascii"), creationflags=0x08000000, check=False)
        self.toast("The API key is on the clipboard." if action == "copy_key" else
                   "The link for other PCs is on the clipboard. It holds the key: share it like a password.")

    def open_file(self, which):
        files = service.paths(self.cfg)
        target = files["request_log"] if which == "request_log" else os.path.dirname(files["out"])
        try:
            os.startfile(target)                                  # noqa: the Windows "open"
        except OSError as error:
            self.toast(f"Could not open {target}: {error}")

    def set_autostart(self, on):
        try:
            autostart_set(on)
            self.toast("Colibri's tray will start when you sign in." if on else "Colibri's tray will not start at sign-in.")
        except OSError as error:
            self.toast(f"Could not change it: {error}")

    def _server_action(self, action):
        if action.startswith("unload:"):
            result = service.unload(self.cfg, action.split(":", 1)[1])
            self.toast({"unloaded": "The model is unloaded.", "unloading when idle": "It is answering a request, and "
                        "will unload when that is done.", "not loaded": "It was not loaded."}.get(result, "The router did not say."))
            self.refresh()
            return
        force = action.endswith("_force")
        name = action.replace("_force", "")
        word = {"start": "Starting", "stop": "Stopping", "restart": "Restarting"}.get(name)
        try:
            self.set_pending(word)
            if name == "start":
                service.start(self.cfg, wait=300)
                self.toast("The server is up.")
            elif name == "stop":
                self.expected_stop = True
                service.stop(self.cfg, force=force)
                self.toast("The server is stopped.")
            elif name == "restart":
                self.expected_stop = True
                service.restart(self.cfg, force=force, wait=300)
                self.toast("The server is up again.")
        except service.ServiceError as error:
            self.toast(str(error))
        finally:
            self.pending = None
            self.refresh()                       # look again before redrawing: the last look is from before the action
            self.expected_stop = False

    def set_pending(self, word):
        self.pending = word
        self.refresh(poll=False)

    # -- watching the server
    def refresh(self, poll=True):
        """From any thread: look at the server (if `poll`) and tell the window to redraw."""
        if poll:
            try:
                self.now = service.status(self.cfg)
            except Exception:                                     # the watcher must not die
                self.now = None
        self._set_summary()

    def _set_summary(self):
        previous = self.summary
        self.summary = summarize(self.now, self.pending, self.cfg["port"])
        if self.hwnd:
            self.api.user32.PostMessageW(self.hwnd, WM_STATE, 0, 0)
        if previous["running"] and self.summary["stopped"] and not self.expected_stop and not self.quitting.is_set():
            self.toast("The server stopped. Its log is in " + os.path.dirname(service.paths(self.cfg)["err"]) + ".")

    def _watch(self):
        while not self.quitting.wait(self.interval):
            self.refresh()

    # -- the window
    def _wndproc(self, hwnd, message, wparam, lparam):
        try:
            if message == WM_ICON and lparam in (WM_LBUTTONUP, WM_RBUTTONUP):
                self.show_menu()
                return 0
            if message == WM_STATE:
                self.update_icon()
                return 0
            if message == WM_TOAST:
                with self.lock:
                    pending, self.toasts = self.toasts, []
                for title, text in pending:
                    self.show_toast(title, text)
                return 0
            if message == self.taskbar_created:                   # Explorer restarted: its icons are gone
                self.add_icon()
                return 0
            if message == WM_DESTROY:
                self.quitting.set()
                self.remove_icon()
                self.api.user32.PostQuitMessage(0)
                return 0
        except Exception as error:                                # an exception cannot cross back into Windows
            print(f"tray: {error!r}", file=sys.stderr)
        return self.api.user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def create_window(self):
        a = self.api
        self.taskbar_created = a.user32.RegisterWindowMessageW("TaskbarCreated")
        instance = a.kernel32.GetModuleHandleW(None)
        cls = a.WNDCLASSW()
        cls.lpfnWndProc, cls.hInstance, cls.lpszClassName = self._proc, instance, "ColibriTrayWindow"
        if not a.user32.RegisterClassW(a.ctypes.byref(cls)):
            raise OSError(a.ctypes.get_last_error(), "RegisterClassW failed")
        self.hwnd = a.user32.CreateWindowExW(0, "ColibriTrayWindow", "Colibri", 0, 0, 0, 0, 0, None, None, instance, None)
        if not self.hwnd:
            raise OSError(a.ctypes.get_last_error(), "CreateWindowExW failed")

    def run(self):
        a = self.api
        mutex = a.kernel32.CreateMutexW(None, False, "Local\\ColibriTray")
        if a.ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            return 2
        self.create_window()
        if not self.add_icon():
            raise OSError("Windows would not add the icon to the notification area")
        threading.Thread(target=self._watch, daemon=True).start()
        threading.Thread(target=self.refresh, daemon=True).start()
        if self.cfg["start_with_tray"]:
            threading.Thread(target=self._start_if_stopped, daemon=True).start()
        message = a.w.MSG()
        while a.user32.GetMessageW(a.ctypes.byref(message), None, 0, 0) > 0:
            a.user32.TranslateMessage(a.ctypes.byref(message))
            a.user32.DispatchMessageW(a.ctypes.byref(message))
        del mutex
        return 0

    def _start_if_stopped(self):
        self.refresh()
        if self.now and self.now["state"] == "stopped":
            self._server_action("start")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="coli tray", description="the notification-area icon for colibri (docs/tray.md)")
    ap.add_argument("--install", action="store_true", help="start the tray whenever you sign in to Windows")
    ap.add_argument("--remove", action="store_true", help="stop starting it at sign-in")
    ap.add_argument("--here", action="store_true", help="run in this process (the default starts a separate one and returns)")
    ap.add_argument("--config", default=None, metavar="FILE", help="the service settings (see `coli service config`)")
    a = ap.parse_args(argv)
    if sys.platform != "win32":
        print("coli tray: the tray is for Windows. `coli service start|stop|status` does the same from a terminal.",
              file=sys.stderr)
        return 1
    try:
        if a.install or a.remove:
            autostart_set(a.install)
            print("the tray will start when you sign in:\n  " + autostart_command() if a.install
                  else "the tray will no longer start when you sign in")
            return 0
        cfg = service.load_config(a.config)
        if not a.here:
            command = [windowless_python(), os.path.join(HERE, "tray.py"), "--here"] + (["--config", a.config] if a.config else [])
            subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=0x08000000 | 0x00000200)
            print("the tray is starting: look for the circle in the notification area (the ^ by the clock)")
            return 0
        return Tray(cfg).run()
    except (service.ServiceError, OSError) as error:
        print(f"coli tray: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
