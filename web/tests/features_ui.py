"""Browser check of the `features` block of /health: Settings offers Presets, Requests and Log only where there is a router.

Run `npm run build`, then `python3 tests/features_ui.py [dist-dir]` (needs Playwright and its Chromium; set CHROME_PATH to use
another Chromium binary). A local fixture server stands in for the gateway and answers /health three ways: without a
`features` block (an older server), as `coli serve`, and as `coli router`. No model or credentials needed.
"""
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
DIST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"
BASE = {"status": "ok", "kv_slots": 1, "scheduler": {"active": 0, "capacity": 1, "queued": 0, "max_queue": 8, "completed": 0, "rejected": 0, "timed_out": 0, "cancelled": 0}}
ANSWERS = {
    "older server": BASE,
    "coli serve": dict(BASE, features={"api": 1, "router": False, "model_switch": "restart", "server_log": False, "presets": False, "request_log": False, "json_mode": True}),
    "coli router": dict(BASE, features={"api": 1, "router": True, "model_switch": "load", "server_log": True, "presets": False, "request_log": False, "json_mode": True}),
}
state = {"health": "older server"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(DIST), **k)

    def log_message(self, *_):
        pass

    def reply(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/v1/models":
            self.reply({"data": [{"id": "gemma-4-colibri", "capabilities": ["chat"], "context_length": 32768}]})
        elif path == "/health":
            self.reply(ANSWERS[state["health"]])
        elif path in ("/experts", "/profile", "/v1/requests", "/v1/models/installed", "/v1/presets", "/v1/models/settings", "/v1/logs"):
            self.reply({"data": [], "enabled": False})
        else:
            super().do_GET()


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_port}"
errors = []
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def tabs(page):
    return [t.strip() for t in page.locator(".settings-tabs button").all_inner_texts()]


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"], executable_path=os.environ.get("CHROME_PATH") or None)
    page = browser.new_page(viewport={"width": 1280, "height": 900}, locale="en-US", reduced_motion="reduce")
    page.set_default_timeout(15000)
    page.on("pageerror", lambda e: errors.append(str(e)))
    seen = {}
    for name in ("older server", "coli serve", "coli router"):
        state["health"] = name
        page.goto(url)
        expect(page.locator(".model-button")).to_contain_text("gemma-4-colibri")
        page.get_by_role("button", name="Settings", exact=True).click()
        expect(page.locator(".settings-tabs button").first).to_be_visible()
        if name != "older server":
            expect(page.locator(".settings-tabs button")).to_have_count(4 if name == "coli serve" else 7, timeout=10000)
        seen[name] = tabs(page)

    check("a server that sends no feature list keeps every page", len(seen["older server"]) == 7, str(seen["older server"]))
    check("coli serve hides Presets, Requests and Log", seen["coli serve"] == ["General", "Model", "Connection", "System"], str(seen["coli serve"]))
    check("coli router has them all", len(seen["coli router"]) == 7, str(seen["coli router"]))
    check("a router without a request log or presets file keeps those pages (they say how to turn them on)",
          "Requests" in seen["coli router"] and "Presets" in seen["coli router"])

    # the page that was open when the server turned out to have no such page falls back to General
    state["health"] = "coli router"
    page.goto(url)
    expect(page.locator(".model-button")).to_contain_text("gemma-4-colibri")
    page.get_by_role("button", name="Settings", exact=True).click()
    page.locator(".settings-tabs").get_by_role("button", name="Log", exact=True).click()
    state["health"] = "coli serve"
    expect(page.locator(".settings-tabs button")).to_have_count(4, timeout=15000)
    check("a page that is no longer offered gives way to General",
          page.locator(".settings-tabs [aria-current=page]").inner_text().strip() == "General")
    browser.close()

check("no page errors", not errors, "; ".join(errors))
failed = [r for r in results if not r[1]]
print(f"{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
