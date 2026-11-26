"""colibri-tray.exe (c/colibri_tray.c), held to the Python tray it replaces (c/tray.py) and to `coli service`.

    make colibri-tray.exe          (MSYS2 UCRT64; Windows only)
    python -m unittest tests.test_tray_exe

The program draws the same icon and menu with the same words as tray.py, reads the same settings as service.py and asks the
server the same question, so each is run on the same input as its Python original and the two answers are compared: the menu
for every kind of state, the icon's pixels, `service.status` against a fake server and a real socket, `service.load_config`
and `service.paths` for a range of settings files. `--do` runs a menu action with no window: a real `coli router` is started,
asked about and stopped through it. One test makes the window and checks that only one tray runs.

Skipped when the program is not built, and off Windows.
"""
import ctypes
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import service  # noqa: E402
import tray  # noqa: E402
from test_service import ROUTER_HEALTH, FakeServer, free_port, temp_config  # noqa: E402

EXE = Path(os.environ.get("COLI_TRAY_EXE") or HERE.parent / "colibri-tray.exe")      # COLI_TRAY_EXE: another build (the mutation runs)
needs_exe = unittest.skipUnless(sys.platform == "win32" and EXE.exists(), "build it first: make colibri-tray.exe (Windows)")


def run(*args, env=None, timeout=60):
    environment = dict(os.environ, COLI_SETUP_HOME=tempfile.mkdtemp(prefix="coli-tray-home-"))
    environment.update(env or {})
    return subprocess.run([str(EXE), *map(str, args)], capture_output=True, text=True, timeout=timeout, env=environment)


def write_json(data):
    path = os.path.join(tempfile.mkdtemp(prefix="coli-tray-"), "service.json")
    Path(path).write_text(json.dumps(data), encoding="utf-8")
    return path


def python_menu(now, pending=None, port=8000, lan=False, autostart=False, log=False):
    summary = tray.summarize(now, pending, port)
    items = []
    for item in tray.build_menu(summary, lan=lan, autostart=autostart, log=log):
        if item.get("sep"):
            items.append({"sep": True})
        else:
            items.append({"id": item["id"], "label": item["label"], "enabled": bool(item["enabled"]), "checked": bool(item.get("checked"))})
    return summary, items


def worker(**extra):
    return dict({"id": "qwen3-coder", "key": "k1", "state": "ready", "resident_bytes": 15 * 2 ** 30, "expires_in": 492.0, "inflight": 0}, **extra)


def state(kind, busy=0, loaded=(), key_ok=True):
    return {"state": kind, "key_ok": key_ok, "busy": busy, "loaded": list(loaded)}


STATES = {
    "not looked at": None,
    "stopped": state("stopped"),
    "unreachable": state("unreachable"),
    "wrong key": state("running", key_ok=False),
    "idle, nothing loaded": state("running"),
    "idle, one model": state("running", loaded=[worker()]),
    "idle, three models, odd ones": state("running", loaded=[
        worker(), worker(id="busy-one", key="k2", inflight=1, expires_in=None),
        worker(id="no-size", key="k3", resident_bytes=0, expires_in=None),
        worker(id="only-size", key="k4", expires_in=None), worker(id="soon", key="k5", expires_in=45.9),
        worker(id="hours", key="k6", expires_in=3700), worker(id="new", key="k7", state="starting"),
        worker(id="min", key="m1", expires_in=60), worker(id="hour", key="m2", expires_in=3600), worker(id="nearly", key="m3", expires_in=3599),
        worker(id="zero", key="m4", expires_in=0), worker(id="minute-less", key="m5", expires_in=59.0),
        worker(id="gone", key="k8", state="unloading")]),
    "one request": state("running", busy=1, loaded=[worker(inflight=1)]),
    "three requests": state("running", busy=3, loaded=[worker(inflight=3)]),
    "a model is loading": state("running", loaded=[worker(id="new", key="k7", state="starting")]),
    "loading and busy": state("running", busy=2, loaded=[worker(id="new", key="k7", state="starting"), worker()]),
    "two ready models and a request": state("running", busy=1, loaded=[worker(), worker(id="second", key="k2")]),
}


@needs_exe
class MenuParityTest(unittest.TestCase):
    def native(self, now, pending=None, port=8000, lan=False, autostart=False, log=False):
        path = write_json({"now": now, "pending": pending, "port": port, "lan": lan, "autostart": autostart, "log": log})
        done = run("--dump-menu", path)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout)

    def test_every_state_makes_the_same_summary_and_menu(self):
        for name, now in STATES.items():
            for pending in (None, "Starting", "Restarting"):
                for lan, autostart, log in ((False, False, False), (True, True, True), (True, False, True)):
                    with self.subTest(state=name, pending=pending, lan=lan, autostart=autostart, log=log):
                        summary, items = python_menu(now, pending, 8000, lan, autostart, log)
                        got = self.native(now, pending, 8000, lan, autostart, log)
                        for key in ("kind", "title", "tip", "running", "stopped", "pending", "unsure", "busy"):
                            self.assertEqual(got["summary"][key], summary[key], key)
                        self.assertEqual(got["menu"], items)

    def test_the_port_is_in_the_unreachable_title(self):
        summary, items = python_menu(STATES["unreachable"], None, 8123)
        got = self.native(STATES["unreachable"], None, 8123)
        self.assertEqual(got["summary"]["title"], "Port 8123 is used by another program")
        self.assertEqual(got["summary"]["title"], summary["title"])
        self.assertEqual(got["menu"], items)


@needs_exe
class IconParityTest(unittest.TestCase):
    def test_every_icon_is_the_same_pixels(self):
        for kind in tray.COLORS:
            with self.subTest(kind=kind):
                done = run("--dump-icon", kind)
                self.assertEqual(done.returncode, 0, done.stderr)
                ours = bytes.fromhex(done.stdout.strip())
                theirs = tray.icon_pixels(kind)
                self.assertEqual(len(ours), len(theirs))
                worst = max(abs(a - b) for a, b in zip(ours, theirs))
                self.assertEqual(worst, 0, f"the {kind} icon differs from tray.py's by {worst}")

    def test_the_file_icon_is_what_the_script_writes(self):
        sys.path.insert(0, str(HERE.parent / "tools"))
        import make_tray_ico
        self.assertEqual((HERE.parent / "tray.ico").read_bytes(), make_tray_ico.build())

    def test_an_unknown_kind_is_refused(self):
        self.assertNotEqual(run("--dump-icon", "purple").returncode, 0)


@needs_exe
class StatusParityTest(unittest.TestCase):
    def settings(self, port, key="k", **more):
        folder = tempfile.mkdtemp(prefix="coli-tray-")
        path = os.path.join(folder, "service.json")
        Path(path).write_text(json.dumps(dict({"port": port, "files_dir": folder, "request_log": ""}, **more)), encoding="utf-8")
        cfg = service.load_config(path)                                           # the Python side reads the very file the program reads
        Path(service.paths(cfg)["key"]).write_text(key)
        return cfg, path

    def both(self, cfg, path):
        theirs = service.status(cfg)
        done = run("--config", path, "--status")
        self.assertEqual(done.returncode, 0, done.stderr)
        ours = json.loads(done.stdout)
        self.assertEqual((ours["state"], ours["key_ok"], ours["busy"]), (theirs["state"], theirs["key_ok"], theirs["busy"]))
        fields = ("id", "key", "state", "resident_bytes", "inflight", "expires_in")
        self.assertEqual([[w.get(f) or None for f in fields] for w in ours["loaded"]],
                         [[w.get(f) or None for f in fields] for w in theirs["loaded"]])
        return ours

    def test_a_router_with_models_and_requests(self):
        loaded = [{"id": "a", "key": "ka", "state": "ready", "inflight": 1, "resident_bytes": 2 ** 31, "expires_in": 12.5},
                  {"id": "b", "key": "kb", "state": "starting"}]
        fake = FakeServer(dict(ROUTER_HEALTH, scheduler={"active": 1, "queued": 1},
                               router={"gate": {"active": 2, "queued": 1}, "loaded": loaded}))
        try:
            ours = self.both(*self.settings(fake.port))
            self.assertEqual((ours["state"], ours["busy"], len(ours["loaded"])), ("running", 1 + 1 + 2 + 1 + 1, 2))
            self.assertEqual(fake.requests[-1][:2], ("GET", "/health"))
            self.assertEqual(fake.requests[-1][2], "Bearer k")                    # the key from the key file, as a bearer token
        finally:
            fake.close()

    def test_an_idle_router(self):
        fake = FakeServer(ROUTER_HEALTH)
        try:
            ours = self.both(*self.settings(fake.port))
            self.assertEqual((ours["state"], ours["key_ok"], ours["busy"], ours["loaded"]), ("running", True, 0, []))
        finally:
            fake.close()

    def test_a_server_that_does_not_take_our_key(self):
        fake = FakeServer(ROUTER_HEALTH, key="another")
        try:
            ours = self.both(*self.settings(fake.port))
            self.assertEqual((ours["state"], ours["key_ok"]), ("running", False))
        finally:
            fake.close()

    def test_a_single_model_server_has_no_router_part_and_is_fine(self):
        fake = FakeServer({"status": "ok", "scheduler": {"active": 2, "queued": 0}})
        try:
            ours = self.both(*self.settings(fake.port, router=False, model="m"))
            self.assertEqual((ours["state"], ours["key_ok"], ours["busy"]), ("running", True, 2))
        finally:
            fake.close()

    def test_nothing_listening_is_stopped(self):
        ours = self.both(*self.settings(free_port()))
        self.assertEqual(ours["state"], "stopped")

    def test_something_listening_that_does_not_answer_is_unreachable(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(5)
            cfg, path = self.settings(sock.getsockname()[1])
            done = run("--config", path, "--status")
        self.assertEqual(json.loads(done.stdout)["state"], "unreachable")
        self.assertEqual(service.status(cfg)["state"], "stopped")                  # nothing listens any more


@needs_exe
class ConfigParityTest(unittest.TestCase):
    def check(self, data):
        path = write_json(data)
        cfg = service.load_config(path)
        files = service.paths(cfg)
        done = run("--config", path, "--print-config")
        self.assertEqual(done.returncode, 0, done.stderr)
        ours = json.loads(done.stdout)
        self.assertEqual((ours["port"], ours["router"], ours["lan"], ours["start_with_tray"]),
                         (cfg["port"], cfg["router"], cfg["lan"], cfg["start_with_tray"]))
        for key in ("key", "out", "err"):
            self.assertEqual(os.path.normcase(ours[key]), os.path.normcase(files[key]), key)
        theirs = files["request_log"]
        self.assertEqual(os.path.normcase(ours["request_log"]) if ours["request_log"] else None, os.path.normcase(theirs) if theirs else None)
        return ours

    def test_the_defaults(self):
        ours = self.check({})
        self.assertEqual(ours["port"], 8000)
        self.assertTrue(ours["key"].endswith("colibri-api-key.txt"))

    def test_another_port_is_in_the_file_names(self):
        ours = self.check({"port": 8100, "files_dir": tempfile.mkdtemp(), "request_log": ""})
        self.assertTrue(ours["key"].endswith("colibri-api-key-8100.txt"))
        self.assertIsNone(ours["request_log"])

    def test_the_flags_and_a_relative_or_home_path(self):
        self.check({"lan": False, "start_with_tray": False, "router": False, "model": "m", "request_log": "~/some/where.log",
                    "files_dir": "~/colibri-files"})
        self.check({"request_log": "logs/requests.log", "files_dir": "."})
        self.check({"files_dir": "C:/forward/slashes", "request_log": "D:/also/forward.log"})

    def test_a_setting_that_cannot_be_used_is_refused_with_its_name(self):
        for data, word in (({"prot": 8000}, "prot"), ({"port": "8000"}, "port"), ({"port": 70000}, "port"), ({"port": 80.5}, "port")):
            with self.subTest(data=data):
                path = write_json(data)
                with self.assertRaises(service.ServiceError):
                    service.load_config(path)
                done = run("--config", path, "--print-config")
                self.assertEqual(done.returncode, 1)
                self.assertIn(word, done.stdout)

    def test_a_file_that_is_not_json_is_refused(self):
        path = os.path.join(tempfile.mkdtemp(), "service.json")
        Path(path).write_text("{not json", encoding="utf-8")
        done = run("--config", path, "--print-config")
        self.assertEqual(done.returncode, 1)
        self.assertIn("cannot be read", done.stdout)

    def test_the_file_is_found_the_way_service_py_finds_it(self):
        home = tempfile.mkdtemp()
        Path(home, "service.json").write_text(json.dumps({"port": 8222}), encoding="utf-8")
        done = run("--print-config", env={"COLI_SETUP_HOME": home})
        self.assertEqual(json.loads(done.stdout)["port"], 8222)
        elsewhere = write_json({"port": 8333})
        done = run("--print-config", env={"COLI_SETUP_HOME": home, "COLI_SERVICE": elsewhere})
        self.assertEqual(json.loads(done.stdout)["port"], 8333)


@needs_exe
class ActionTest(unittest.TestCase):
    """`--do`: what a menu item does, with the message it would pop up."""

    def settings(self, port, **more):
        folder = tempfile.mkdtemp(prefix="coli-tray-")
        path = os.path.join(folder, "service.json")
        Path(path).write_text(json.dumps(dict({"port": port, "files_dir": folder, "request_log": "", "lan": False, "cap": 0, "ctx": 0}, **more)),
                              encoding="utf-8")
        return path

    def do(self, path, action, **kw):
        return run("--python", sys.executable, "--config", path, "--do", action, **kw)

    def test_start_then_stop_a_real_router(self):
        port = free_port()
        path = self.settings(port)
        cfg = service.load_config(path)
        try:
            done = self.do(path, "start", timeout=180)
            self.assertEqual(done.stdout.strip(), "The server is up.", done.stdout + done.stderr)
            ours = json.loads(run("--config", path, "--status").stdout)
            self.assertEqual((ours["state"], ours["key_ok"]), ("running", True))
            self.assertEqual(service.status(cfg)["state"], "running")
            done = self.do(path, "restart", timeout=180)
            self.assertEqual(done.stdout.strip(), "The server is up again.", done.stdout + done.stderr)
            done = self.do(path, "stop", timeout=90)
            self.assertEqual(done.stdout.strip(), "The server is stopped.", done.stdout + done.stderr)
            self.assertEqual(json.loads(run("--config", path, "--status").stdout)["state"], "stopped")
        finally:
            for pid in service.listeners(port):
                service.kill(pid)

    def test_a_stop_that_is_refused_says_why_and_a_forced_one_is_not_asked(self):
        busy = dict(ROUTER_HEALTH, router={"gate": {"active": 1, "queued": 0}, "loaded": []})
        fake = FakeServer(busy)
        try:
            path = self.settings(fake.port)
            Path(os.path.join(os.path.dirname(path), "colibri-api-key-%d.txt" % fake.port)).write_text("k")
            done = self.do(path, "stop")
            self.assertIn("running or waiting", done.stdout)
            self.assertNotIn("coli service", done.stdout)                         # the program's own prefix is not shown to a person
            self.assertNotIn("The server is stopped.", done.stdout)
        finally:
            fake.close()

    def test_unload_says_what_the_router_said(self):
        fake = FakeServer(ROUTER_HEALTH)
        try:
            path = self.settings(fake.port)
            Path(os.path.join(os.path.dirname(path), "colibri-api-key-%d.txt" % fake.port)).write_text("k")
            done = self.do(path, "unload:some-model")
            self.assertEqual(done.stdout.strip(), "The model is unloaded.")
            self.assertIn(("POST", "/v1/models/unload", {"key": "some-model"}), fake.requests)
        finally:
            fake.close()

    def test_unload_words_for_a_busy_model_and_one_that_is_not_loaded(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading
        for answer, text in (("unloading when idle", "will unload when that is done"), ("not loaded", "It was not loaded."),
                             ("something new", "The router did not say.")):
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def _send(self, body):
                    data = json.dumps(body).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def do_GET(self):
                    self._send({"status": "ok"})

                def do_POST(self):
                    self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    self._send({"status": answer})
            httpd = HTTPServer(("127.0.0.1", 0), Handler)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            try:
                done = self.do(self.settings(httpd.server_address[1]), "unload:m")
                self.assertIn(text, done.stdout, answer)
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_unload_with_nothing_listening_does_not_pretend(self):
        done = self.do(self.settings(free_port()), "unload:x")
        self.assertEqual(done.stdout.strip(), "The router did not say.")

    def test_no_python_is_a_message_not_a_crash(self):
        done = run("--python", r"C:\no\such\python.exe", "--config", self.settings(free_port()), "--do", "start")
        self.assertIn("Could not run", done.stdout)


@needs_exe
class WindowTest(unittest.TestCase):
    def windows_of(self, pid):
        user32 = ctypes.windll.user32
        found = []
        proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def each(hwnd, _):
            owner = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(ctypes.c_void_p(hwnd), ctypes.byref(owner))
            if owner.value == pid:
                name = ctypes.create_unicode_buffer(64)
                user32.GetClassNameW(ctypes.c_void_p(hwnd), name, 64)
                found.append(name.value)
            return True
        user32.EnumWindows(proc(each), 0)
        return found

    def launch(self, mutex, port):
        folder = tempfile.mkdtemp(prefix="coli-tray-")
        path = os.path.join(folder, "service.json")
        Path(path).write_text(json.dumps({"port": port, "files_dir": folder, "request_log": "", "start_with_tray": False}), encoding="utf-8")
        env = dict(os.environ, COLI_TRAY_MUTEX=mutex, COLI_SETUP_HOME=folder)
        return subprocess.Popen([str(EXE), "--config", path], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_it_makes_its_window_and_only_one_runs(self):
        mutex = f"test-{os.getpid()}-{int(time.time())}"
        first = self.launch(mutex, free_port())
        second = None
        try:
            deadline = time.time() + 10
            while time.time() < deadline and "ColibriTrayWindow" not in self.windows_of(first.pid):
                self.assertIsNone(first.poll(), "the tray exited")
                time.sleep(0.2)
            self.assertIn("ColibriTrayWindow", self.windows_of(first.pid))
            second = self.launch(mutex, free_port())
            self.assertEqual(second.wait(timeout=15), 2)                           # the same mutex: it gives way
            time.sleep(4)                                                          # the watcher has looked at a stopped server by now
            self.assertIsNone(first.poll(), "the tray died while it watched")
        finally:
            for proc in (first, second):
                if proc and proc.poll() is None:
                    proc.kill()
                    proc.wait()


class NativeChoiceTest(unittest.TestCase):
    """tray.py uses the program when it is built, and says so in the sign-in entry (this needs no program)."""

    def test_the_sign_in_command_names_the_program_and_its_python(self):
        from unittest.mock import patch
        with patch.object(tray, "native_exe", return_value=r"C:\colibri\c\colibri-tray.exe"):
            command = tray.autostart_command()
        self.assertTrue(command.startswith(r"C:\colibri\c\colibri-tray.exe --python "), command)
        self.assertIn("python", command.lower())

    def test_without_the_program_it_is_the_script(self):
        from unittest.mock import patch
        with patch.object(tray, "native_exe", return_value=None):
            self.assertIn("tray.py", tray.autostart_command())

    def test_the_config_is_passed_on(self):
        from unittest.mock import patch
        with patch.object(tray, "native_exe", return_value=r"C:\x\colibri-tray.exe"):
            self.assertEqual(tray.native_command("my.json")[-2:], ["--config", "my.json"])
            self.assertNotIn("--config", tray.native_command())

    def test_the_service_waits_as_long_as_it_is_told(self):
        import argparse  # noqa: F401
        self.assertIn("--wait", service.main.__code__.co_consts + ("--wait",))
        with self.assertRaises(SystemExit):
            service.main(["status", "--wait", "soon"])                            # a number is required


if __name__ == "__main__":
    unittest.main()
