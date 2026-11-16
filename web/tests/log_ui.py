"""Browser check of the Log page (Settings -> Log): the lines the router and its models printed, read every few seconds.

Run `npm run build`, then `python3 tests/log_ui.py [dist-dir [screenshot-dir]]` (needs Playwright and its Chromium; set
CHROME_PATH to use another Chromium binary). A local fixture server stands in for the router: /v1/logs answers with numbered
lines and adds a new one on every read. No model or credentials needed. Screenshots go to the temp folder.
"""
import json
import os
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
DIST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.gettempdir()) / "colibri-log-ui"
OUT.mkdir(exist_ok=True)
health = {"status": "ok", "kv_slots": 1, "scheduler": {"active": 0, "capacity": 1, "queued": 0, "max_queue": 8, "completed": 0, "rejected": 0, "timed_out": 0, "cancelled": 0}}

state = {"available": True, "lines": [], "reads": [], "first_gap": False}
lock = threading.Lock()


def add(source, text):
    state["lines"].append({"seq": len(state["lines"]) + 1, "time": 1_700_000_000 + len(state["lines"]), "source": source, "text": text})


add("router", "Starting the router")
add("gemma-4-colibri", "loading 12 of 30 layers")
add("gemma-4-colibri", "the device stopped answering, the experts stay on the CPU")


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
        url = urlparse(self.path)
        if url.path == "/v1/models":
            self.reply({"data": [{"id": "gemma-4-colibri", "capabilities": ["chat"], "context_length": 32768}]})
        elif url.path == "/health":
            self.reply(health)
        elif url.path == "/v1/logs":
            with lock:
                if not state["available"]:
                    return self.reply({"error": {"message": "Not found."}}, 404)
                since = int(parse_qs(url.query).get("since", ["0"])[0])
                state["reads"].append(since)
                data = [l for l in state["lines"] if l["seq"] > since]
                first = state["lines"][0]["seq"] if state["lines"] else 1
                if state["first_gap"]:
                    first = since + 40
                self.reply({"object": "list", "data": data, "next": len(state["lines"]), "first": first})
        elif url.path in ("/experts", "/profile", "/v1/requests", "/v1/models/installed", "/v1/presets", "/v1/models/settings"):
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


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"], executable_path=os.environ.get("CHROME_PATH") or None)
    page = browser.new_page(viewport={"width": 1280, "height": 900}, locale="en-US", reduced_motion="reduce")
    page.set_default_timeout(15000)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    expect(page.locator(".model-button")).to_contain_text("gemma-4-colibri")
    page.get_by_role("button", name="Settings", exact=True).click()
    page.locator(".settings-tabs").get_by_role("button", name="Log", exact=True).click()

    expect(page.locator(".log-line")).to_have_count(3)
    check("the lines the router already holds are shown, oldest first",
          page.locator(".log-line .log-text").first.inner_text() == "Starting the router")
    check("each line says which model printed it", page.locator(".log-source-name").nth(1).inner_text() == "gemma-4-colibri")
    check("a line about a failure stands out", page.locator(".log-line.bad").count() == 1)
    page.screenshot(path=str(OUT / "1-log.png"))

    # new lines arrive by themselves, and the page only asks for what came after the last it saw
    with lock:
        add("router", "gemma-4-colibri is ready")
    expect(page.locator(".log-line")).to_have_count(4, timeout=10000)
    check("a new line appears without a reload", page.locator(".log-line .log-text").last.inner_text() == "gemma-4-colibri is ready")
    with lock:
        reads = list(state["reads"])
    check("it asks for what came after the last line, not everything again", reads[0] == 0 and max(reads) == 3, str(reads))

    # filter by source
    page.locator(".log-source select").select_option("gemma-4-colibri")
    check("the source filter shows only that model's lines", page.locator(".log-line").count() == 2)
    page.locator(".log-source select").select_option("")
    check("and Everything brings the rest back", page.locator(".log-line").count() == 4)

    # pause stops the reading
    page.get_by_role("button", name="Pause", exact=True).click()
    page.wait_for_timeout(300)
    with lock:
        before = len(state["reads"])
        add("router", "written while paused")
    page.wait_for_timeout(4500)
    with lock:
        after = len(state["reads"])
    check("Pause stops asking the server", after == before, f"{before} -> {after}")
    check("and the new line is not shown yet", page.locator(".log-line").count() == 4)
    page.get_by_role("button", name="Resume", exact=True).click()
    expect(page.locator(".log-line")).to_have_count(5, timeout=10000)
    check("Resume catches up with what was written meanwhile", page.locator(".log-line .log-text").last.inner_text() == "written while paused")

    # clear empties the view and keeps going from where it was
    page.get_by_role("button", name="Clear", exact=True).click()
    check("Clear empties the view", page.locator(".log-line").count() == 0)
    with lock:
        add("router", "after clearing")
    expect(page.locator(".log-line")).to_have_count(1, timeout=10000)
    check("only newer lines come back after Clear", page.locator(".log-line .log-text").first.inner_text() == "after clearing")

    # lines dropped between two reads are admitted
    with lock:
        state["first_gap"] = True
        add("router", "one more")
    expect(page.get_by_text("Some lines were dropped")).to_be_visible(timeout=10000)
    check("a gap in the log is said out loud", True)
    page.screenshot(path=str(OUT / "2-gap.png"))

    # a server with no such route (a plain coli serve)
    with lock:
        state["available"] = False
    expect(page.get_by_text("This server has no log to read")).to_be_visible(timeout=10000)
    check("a server without the route is explained, not an error", page.locator("[role=alert]").count() == 0)
    browser.close()

check("no page errors", not errors, "; ".join(errors))
failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
