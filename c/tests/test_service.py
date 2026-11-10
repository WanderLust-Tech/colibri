"""The background service (c/service.py) and the tray's logic (c/tray.py).

    python -m unittest tests.test_service

Most of it runs on plain data and on a small fake server that answers /health the way the router does. One test starts a
real `coli router` (it has no models to load, so it is quick) on a free port and stops it again. The window itself is
tried on Windows only, without showing a menu.
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import service  # noqa: E402
import tray  # noqa: E402


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeServer:
    """Answers /health with `payload` (the unauthenticated one when the key is wrong) and records the requests."""

    def __init__(self, payload, key="k"):
        self.payload, self.key, self.requests = payload, key, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.requests.append(("GET", self.path, self.headers.get("Authorization")))
                authed = self.headers.get("Authorization") == f"Bearer {outer.key}"
                self._send(200, outer.payload if authed else {"status": "ok"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"null")
                outer.requests.append(("POST", self.path, body))
                self._send(200, {"status": "unloaded", "key": body.get("key")})

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


ROUTER_HEALTH = {"status": "ok", "scheduler": {"active": 0, "queued": 0},
                 "router": {"gate": {"active": 0, "queued": 0}, "loaded": []}}


def temp_config(**overrides):
    folder = tempfile.mkdtemp()
    cfg = service.load_config(os.path.join(folder, "none.json"), dict({"files_dir": folder, "request_log": ""}, **overrides))
    return cfg, folder


class ConfigTest(unittest.TestCase):
    def write(self, text):
        path = os.path.join(tempfile.mkdtemp(), "service.json")
        Path(path).write_text(text, encoding="utf-8")
        return path

    def test_defaults_are_the_scripts(self):
        cfg = service.load_config(os.path.join(tempfile.mkdtemp(), "absent.json"))
        self.assertEqual((cfg["port"], cfg["router"], cfg["keep_alive"], cfg["kv_slots"], cfg["cap"], cfg["ctx"], cfg["lan"]),
                         (8000, True, "10m", 1, 128, 32768, True))

    def test_file_then_overrides(self):
        path = self.write('{"port": 8100, "lan": false, "ctx": 0}')
        cfg = service.load_config(path, {"port": 8200, "lan": None})
        self.assertEqual((cfg["port"], cfg["lan"], cfg["ctx"]), (8200, False, 0))

    def test_a_typo_is_an_error_not_a_default(self):
        with self.assertRaisesRegex(service.ServiceError, "unknown setting 'keepalive'"):
            service.load_config(self.write('{"keepalive": "5m"}'))

    def test_wrong_types_and_ranges(self):
        for text, word in (('{"port": "8000"}', "port"), ('{"port": true}', "port"), ('{"port": 70000}', "port"),
                           ('{"lan": "yes"}', "lan"), ('{"extra_args": "x"}', "extra_args"), ("[1]", "JSON object"),
                           ("{", "cannot be read")):
            with self.assertRaisesRegex(service.ServiceError, word, msg=text):
                service.load_config(self.write(text))

    def test_a_single_model_server_needs_its_model(self):
        with self.assertRaisesRegex(service.ServiceError, "'model'"):
            service.load_config(self.write('{"router": false}'))
        self.assertEqual(service.load_config(self.write('{"router": false, "model": "D:/m"}'))["model"], "D:/m")

    def test_file_names_follow_the_script(self):
        cfg, folder = temp_config()
        self.assertEqual(os.path.basename(service.paths(cfg)["key"]), "colibri-api-key.txt")
        cfg, folder = temp_config(port=8123)
        files = service.paths(cfg)
        self.assertEqual([os.path.basename(files[k]) for k in ("key", "out", "err")],
                         ["colibri-api-key-8123.txt", "colibri-serve-8123.log", "colibri-serve-8123.err"])
        self.assertIsNone(files["request_log"])


class KeyAndLinksTest(unittest.TestCase):
    def test_key_is_made_once_and_kept(self):
        cfg, _ = temp_config()
        self.assertEqual(service.read_key(cfg), "")
        first = service.ensure_key(cfg)
        self.assertGreaterEqual(len(first), 43)                       # 256 bits
        self.assertRegex(first, r"^[A-Za-z0-9_-]+$")                  # safe in a link's #key=
        self.assertEqual(service.ensure_key(cfg), first)
        second = service.ensure_key(cfg, new=True)
        self.assertNotEqual(second, first)
        self.assertEqual(service.read_key(cfg), second)

    def test_links(self):
        cfg, _ = temp_config(port=8100)
        shown = service.links(cfg, "KEY", lan="192.168.1.5")
        self.assertEqual(shown, {"local": "http://127.0.0.1:8100/#key=KEY", "lan": "http://192.168.1.5:8100/#key=KEY"})
        self.assertIsNone(service.links(dict(cfg, lan=False), "KEY", lan="192.168.1.5")["lan"])
        self.assertIsNone(service.links(cfg, "KEY", lan="")["lan"])


class CommandTest(unittest.TestCase):
    def test_router_command_matches_the_script(self):
        cfg, _ = temp_config()
        cmd = service.command(cfg, python="PY")
        self.assertEqual(cmd[0], "PY")
        self.assertEqual(cmd[2:], ["router", "--port", "8000", "--keep-alive", "10m", "--kv-slots", "1",
                                   "--worker-arg=--cap", "--worker-arg=128", "--host", "0.0.0.0", "--allowed-host", "*"])

    def test_this_pc_only_and_no_cap(self):
        cfg, _ = temp_config(lan=False, cap=0, extra_args=["--max-loaded", "2"])
        cmd = service.command(cfg, python="PY")
        self.assertNotIn("--host", cmd)
        self.assertFalse(any("cap" in part for part in cmd))
        self.assertEqual(cmd[-2:], ["--max-loaded", "2"])

    def test_single_model_command(self):
        cfg, _ = temp_config(router=False, model="D:/m", lan=False)
        cmd = service.command(cfg, python="PY")
        self.assertEqual(cmd[2:5], ["serve", "--model", "D:/m"])
        self.assertIn("--cap", cmd)

    def test_the_key_goes_in_the_environment_not_the_command(self):
        cfg, folder = temp_config(ctx=65536, request_log=os.path.join(tempfile.gettempdir(), "r.log"))
        env = service.environment(cfg, "SECRET", base={"PATH": "x"})
        self.assertEqual((env["COLI_API_KEY"], env["Q36_MAXT"], env["PYTHONUTF8"], env["PATH"]), ("SECRET", "65536", "1", "x"))
        self.assertTrue(env["COLI_REQUEST_LOG"].endswith("r.log"))
        self.assertNotIn("SECRET", " ".join(service.command(cfg, python="PY")))
        off = service.environment(dict(cfg, request_log="", ctx=0), "k", base={})
        self.assertNotIn("COLI_REQUEST_LOG", off)
        self.assertNotIn("Q36_MAXT", off)


class NetstatTest(unittest.TestCase):
    SAMPLE = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1124
  TCP    0.0.0.0:8000           0.0.0.0:0              LISTENING       66976
  TCP    127.0.0.1:18000        0.0.0.0:0              LISTENING       111
  TCP    127.0.0.1:8000         127.0.0.1:50000        ESTABLISHED     66976
  TCP    127.0.0.1:8000         127.0.0.1:50001        TIME_WAIT       0
  TCP    127.0.0.1:8000         127.0.0.1:50002        ESTABLISHED     777
  TCP    [::]:8000              [::]:0                 LISTENING       66976
"""

    def test_only_listeners_on_exactly_that_port(self):
        self.assertEqual(service.parse_netstat(self.SAMPLE, 8000), [66976])
        self.assertEqual(service.parse_netstat(self.SAMPLE, 135), [1124])
        self.assertEqual(service.parse_netstat(self.SAMPLE, 80), [])


class StatusTest(unittest.TestCase):
    def test_busy_count(self):
        self.assertEqual(service.busy_count(None), 0)
        self.assertEqual(service.busy_count({"scheduler": {"active": 1, "queued": 2}}), 3)
        health = {"scheduler": {"active": 0, "queued": 0},
                  "router": {"gate": {"active": 1, "queued": 1}, "loaded": [{"inflight": 2}, {"inflight": 0}]}}
        self.assertEqual(service.busy_count(health), 4)

    def test_stopped_when_nothing_answers(self):
        cfg, _ = temp_config(port=free_port())
        now = service.status(cfg)
        self.assertEqual((now["state"], now["busy"], now["loaded"]), ("stopped", 0, []))

    def test_running_idle_and_busy_with_a_loaded_model(self):
        loaded = [{"id": "m", "key": "m-key", "state": "ready", "inflight": 1, "resident_bytes": 2 ** 31}]
        health = dict(ROUTER_HEALTH, router={"gate": {"active": 1, "queued": 0}, "loaded": loaded})
        fake = FakeServer(health)
        try:
            cfg, _ = temp_config(port=fake.port)
            service.ensure_key(cfg)
            Path(service.paths(cfg)["key"]).write_text("k")
            now = service.status(cfg)
            self.assertEqual((now["state"], now["router"], now["key_ok"], now["busy"]), ("running", True, True, 2))
            self.assertEqual(now["loaded"][0]["id"], "m")
            self.assertEqual(fake.requests[0], ("GET", "/health", "Bearer k"))
        finally:
            fake.close()

    def test_a_server_that_does_not_accept_our_key_is_flagged(self):
        fake = FakeServer(ROUTER_HEALTH, key="other")
        try:
            cfg, _ = temp_config(port=fake.port)
            Path(service.paths(cfg)["key"]).write_text("k")
            now = service.status(cfg)
            self.assertEqual((now["state"], now["key_ok"], now["router"]), ("running", False, False))
        finally:
            fake.close()

    def test_something_listening_that_does_not_speak_http_is_unreachable(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = sock.getsockname()[1]
            cfg, _ = temp_config(port=port)
            with patch.object(service, "listeners", return_value=[4242]), patch.object(service, "api", return_value=(None, None)):
                now = service.status(cfg)
        self.assertEqual((now["state"], now["pids"]), ("unreachable", [4242]))

    def test_unload_asks_the_router_by_key(self):
        fake = FakeServer(ROUTER_HEALTH)
        try:
            cfg, _ = temp_config(port=fake.port)
            self.assertEqual(service.unload(cfg, "m-key", key="k"), "unloaded")
            self.assertEqual(fake.requests[-1], ("POST", "/v1/models/unload", {"key": "m-key"}))
        finally:
            fake.close()


class StopTest(unittest.TestCase):
    def stop_with(self, now, force=False):
        cfg, _ = temp_config()
        killed = []
        ports = iter([[11], [], []])
        with patch.object(service, "status", return_value=now), \
                patch.object(service, "listeners", side_effect=lambda port: next(ports, [])), \
                patch.object(service, "kill", side_effect=killed.append):
            try:
                return service.stop(cfg, force=force, wait=2), killed
            except service.ServiceError as error:
                return error, killed

    def base(self, **kw):
        return dict({"state": "running", "key_ok": True, "busy": 0, "pids": [], "loaded": []}, **kw)

    def test_stops_an_idle_server(self):
        result, killed = self.stop_with(self.base())
        self.assertEqual((result, killed), ([11], [11]))

    def test_nothing_to_stop(self):
        self.assertEqual(self.stop_with(self.base(state="stopped"))[0], [])

    def test_refuses_while_a_request_runs_unless_forced(self):
        error, killed = self.stop_with(self.base(busy=2))
        self.assertIsInstance(error, service.ServiceError)
        self.assertIn("2 request", str(error))
        self.assertEqual(killed, [])
        self.assertEqual(self.stop_with(self.base(busy=2), force=True)[1], [11])

    def test_refuses_what_it_cannot_read_unless_forced(self):
        for now in (self.base(key_ok=False), self.base(state="unreachable", pids=[11])):
            error, killed = self.stop_with(now)
            self.assertIsInstance(error, service.ServiceError)
            self.assertEqual(killed, [])
            self.assertEqual(self.stop_with(now, force=True)[1], [11])


class RealRouterTest(unittest.TestCase):
    """Start a real `coli router` on a free port, ask it, stop it."""

    def test_start_status_restart_stop(self):
        import warnings
        warnings.simplefilter("ignore", ResourceWarning)           # the server outlives the Popen that started it, as meant
        port = free_port()
        cfg, folder = temp_config(port=port, lan=False, cap=0, ctx=0)
        try:
            self.assertEqual(service.status(cfg)["state"], "stopped")
            now = service.start(cfg, wait=90)
            self.assertEqual((now["state"], now["key_ok"], now["router"]), ("running", True, True))
            self.assertTrue(os.path.exists(service.paths(cfg)["key"]))
            again = service.start(cfg, wait=90)                      # already up: left alone
            self.assertEqual(again["state"], "running")
            with self.assertRaisesRegex(service.ServiceError, "old key"):
                service.start(cfg, new_key=True)
            key = service.read_key(cfg)
            now = service.restart(cfg, new_key=True, wait=90)
            self.assertEqual((now["state"], now["key_ok"]), ("running", True))
            self.assertNotEqual(service.read_key(cfg), key)           # the new key is the one the server has
            self.assertTrue(service.stop(cfg, wait=30))
            self.assertEqual(service.status(cfg)["state"], "stopped")
            self.assertEqual(service.stop(cfg), [])
        finally:
            if service.listeners(port):
                for pid in service.listeners(port):
                    service.kill(pid)


class SummaryTest(unittest.TestCase):
    def now(self, **kw):
        return dict({"state": "running", "key_ok": True, "busy": 0, "loaded": []}, **kw)

    def test_the_kinds(self):
        for now, pending, kind in ((None, None, "starting"), (self.now(state="stopped"), None, "stopped"),
                                   (self.now(state="unreachable", pids=[1]), None, "attention"),
                                   (self.now(key_ok=False), None, "attention"), (self.now(), None, "idle"),
                                   (self.now(busy=2), None, "busy"), (self.now(), "Starting", "starting"),
                                   (self.now(loaded=[{"state": "starting", "id": "m"}]), None, "busy")):
            self.assertEqual(tray.summarize(now, pending)["kind"], kind, (now, pending))

    def test_titles_and_tooltip(self):
        self.assertEqual(tray.summarize(self.now(busy=1))["title"], "Working: 1 request")
        self.assertEqual(tray.summarize(self.now(busy=3))["title"], "Working: 3 requests")
        ready = [{"state": "ready", "id": "a"}, {"state": "ready", "id": "b"}]
        self.assertEqual(tray.summarize(self.now(loaded=ready))["tip"], "Colibri: running, idle (2 models loaded)")
        self.assertLessEqual(len(tray.summarize(self.now(state="unreachable"), port=8000)["tip"]), 127)

    def test_unsure_when_it_cannot_tell(self):
        self.assertTrue(tray.summarize(self.now(key_ok=False))["unsure"])
        self.assertTrue(tray.summarize(self.now(state="unreachable"))["unsure"])
        self.assertFalse(tray.summarize(self.now())["unsure"])

    def test_durations(self):
        self.assertEqual([tray.duration(s) for s in (5, 61, 3725)], ["5s", "1m 01s", "1h 02m"])


def labels(items):
    return [i["label"] for i in items if not i.get("sep")]


def by_id(items, ident):
    return next(i for i in items if i.get("id") == ident)


class MenuTest(unittest.TestCase):
    def menu(self, now, pending=None, **kw):
        return tray.build_menu(tray.summarize(now, pending), **kw)

    def test_stopped_offers_start_only(self):
        items = self.menu({"state": "stopped", "key_ok": True, "busy": 0, "loaded": []})
        self.assertTrue(by_id(items, "start")["enabled"])
        self.assertFalse(any(i.get("id") in ("stop", "restart") for i in items))
        self.assertFalse(by_id(items, "open")["enabled"])
        self.assertIn("No model loaded", labels(items))

    def test_loaded_models_can_be_unloaded(self):
        loaded = [{"id": "qwen", "key": "qwen-key", "state": "ready", "resident_bytes": 3 * 2 ** 30, "expires_in": 125.0, "inflight": 0},
                  {"id": "busy", "key": "busy-key", "state": "ready", "resident_bytes": 0, "expires_in": None, "inflight": 1},
                  {"id": "new", "key": "new-key", "state": "starting"}]
        items = self.menu({"state": "running", "key_ok": True, "busy": 0, "loaded": loaded}, lan=True, log=True)
        self.assertIn("Unload qwen (3.0 GB, unloads in 2m 05s)", labels(items))
        self.assertIn("Unload busy (in use)", labels(items))
        self.assertTrue(by_id(items, "unload:qwen-key")["enabled"])
        self.assertFalse(any(i.get("id") == "unload:new-key" for i in items))       # it is still loading
        self.assertTrue(by_id(items, "open")["enabled"] and by_id(items, "copy_lan")["enabled"])
        self.assertTrue(by_id(items, "request_log")["enabled"])
        self.assertTrue(by_id(items, "stop")["enabled"] and by_id(items, "restart")["enabled"])

    def test_a_busy_server_is_not_stopped_without_saying_what_is_lost(self):
        items = self.menu({"state": "running", "key_ok": True, "busy": 2, "loaded": []})
        self.assertFalse(any(i.get("id") in ("stop", "restart") for i in items))
        self.assertIn("2 requests running will be lost", by_id(items, "stop_force")["label"])

    def test_an_unreadable_server_is_stopped_only_on_purpose(self):
        items = self.menu({"state": "unreachable", "key_ok": True, "busy": 0, "loaded": [], "pids": [9]})
        self.assertFalse(any(i.get("id") in ("stop", "restart") for i in items))
        self.assertIn("does not say", by_id(items, "stop_force")["label"])

    def test_nothing_to_click_while_an_action_runs(self):
        items = self.menu({"state": "stopped", "key_ok": True, "busy": 0, "loaded": []}, pending="Starting")
        self.assertFalse(by_id(items, "start")["enabled"])
        self.assertEqual(labels(items)[0], "Colibri: Starting...")

    def test_the_checkbox_and_the_quit_line(self):
        items = self.menu({"state": "stopped", "key_ok": True, "busy": 0, "loaded": []}, autostart=True)
        self.assertTrue(by_id(items, "autostart")["checked"])
        self.assertIn("keeps running", by_id(items, "quit")["label"])
        self.assertFalse(by_id(self.menu(None), "autostart").get("checked"))

    def test_before_the_first_look_nothing_can_be_restarted(self):
        items = self.menu(None)
        self.assertEqual(labels(items)[0], "Colibri: Looking...")
        self.assertFalse(by_id(items, "restart")["enabled"])


class IconTest(unittest.TestCase):
    def pixel(self, data, x, y, size=32):
        i = (y * size + x) * 4
        return tuple(data[i:i + 4])                                     # B, G, R, A (premultiplied)

    def test_size_and_shapes(self):
        for kind in tray.COLORS:
            self.assertEqual(len(tray.icon_pixels(kind)), 32 * 32 * 4)
        idle = tray.icon_pixels("idle")
        self.assertEqual(self.pixel(idle, 0, 0)[3], 0)                 # the corner is clear
        self.assertEqual(self.pixel(idle, 16, 16)[3], 255)             # a disc
        stopped = tray.icon_pixels("stopped")
        self.assertEqual(self.pixel(stopped, 16, 16)[3], 0)            # a ring: hollow in the middle
        self.assertEqual(self.pixel(stopped, 2, 16)[3], 255)           # and solid at its edge

    def test_colours_differ_and_attention_has_a_mark(self):
        centres = {kind: self.pixel(tray.icon_pixels(kind), 8, 16) for kind in tray.COLORS if kind != "stopped"}
        self.assertEqual(len(set(centres.values())), len(centres))
        attention = tray.icon_pixels("attention")
        self.assertEqual(self.pixel(attention, 16, 12)[:3], (255, 255, 255))   # the bar of the "!"


class AutostartTest(unittest.TestCase):
    def test_the_command_runs_this_file_without_a_window(self):
        command = tray.autostart_command()
        self.assertIn("tray.py", command)
        self.assertTrue(command.endswith("--here"))
        self.assertTrue(command.startswith('"'))

    @unittest.skipUnless(sys.platform == "win32", "the Run key is Windows'")
    def test_run_key_round_trip(self):
        with patch.object(tray, "RUN_VALUE", "Colibri tray (unit test)"):
            try:
                self.assertIsNone(tray.autostart_get())
                tray.autostart_set(True)
                self.assertEqual(tray.autostart_get(), tray.autostart_command())
                tray.autostart_set(False)
                self.assertIsNone(tray.autostart_get())
                tray.autostart_set(False)                                # again: no error
            finally:
                tray.autostart_set(False)


@unittest.skipUnless(sys.platform == "win32", "the window is Windows'")
class WindowTest(unittest.TestCase):
    def setUp(self):
        cfg, _ = temp_config(port=free_port())
        self.tray = tray.Tray(cfg)

    def test_struct_matches_windows(self):
        import ctypes
        self.assertEqual(ctypes.sizeof(self.tray.api.NOTIFYICONDATAW), 976 if ctypes.sizeof(ctypes.c_void_p) == 8 else 956)

    def test_native_menu_has_every_item(self):
        self.tray.summary = tray.summarize({"state": "stopped", "key_ok": True, "busy": 0, "loaded": []})
        items = self.tray.menu()
        handle, actions = self.tray.native_menu(items)
        try:
            self.assertEqual(self.tray.api.user32.GetMenuItemCount(handle), len(items))
            self.assertTrue({"start", "quit", "autostart"} <= set(actions.values()))
        finally:
            self.tray.api.user32.DestroyMenu(handle)

    def test_icons_are_made_for_every_state(self):
        for kind in tray.COLORS:
            self.assertTrue(self.tray._icon(kind))

    def test_a_refused_stop_becomes_a_message(self):
        self.tray.now = {"state": "running", "key_ok": True, "busy": 1, "loaded": [], "pids": []}
        with patch.object(service, "status", return_value=self.tray.now), \
                patch.object(service, "stop", side_effect=service.ServiceError("1 request(s) are running")):
            self.tray._server_action("stop")
        self.assertEqual(self.tray.toasts[-1], ("Colibri", "1 request(s) are running"))
        self.assertIsNone(self.tray.pending)

    def test_unload_reports_what_the_router_said(self):
        with patch.object(service, "unload", return_value="unloading when idle"), \
                patch.object(service, "status", return_value={"state": "stopped", "key_ok": True, "busy": 0, "loaded": []}):
            self.tray._server_action("unload:some-key")
        self.assertIn("when that is done", self.tray.toasts[-1][1])

    def test_a_stop_nobody_asked_for_is_announced(self):
        self.tray.summary = tray.summarize({"state": "running", "key_ok": True, "busy": 0, "loaded": []})
        self.tray.now = {"state": "stopped", "key_ok": True, "busy": 0, "loaded": []}
        self.tray._set_summary()
        self.assertIn("stopped", self.tray.toasts[-1][1])
        self.tray.toasts.clear()
        self.tray.expected_stop = True                                  # one the tray did itself is not
        self.tray.summary = tray.summarize({"state": "running", "key_ok": True, "busy": 0, "loaded": []})
        self.tray._set_summary()
        self.assertEqual(self.tray.toasts, [])


if __name__ == "__main__":
    unittest.main()
