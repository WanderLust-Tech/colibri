"""Browser check of client tags: each conversation puts `X-Colibri-Tags` on what it asks, and the Requests tab lists by a tag.

Run `npm run build`, then `python3 tests/tags_ui.py [dist-dir [screenshot-dir]]` (needs Playwright and its Chromium; set CHROME_PATH
to use another Chromium binary). A local fixture server stands in for the router: it keeps the tags header of every chat it is sent and
answers GET /v1/requests from what it kept, honouring `tag=`. No model or credentials needed.
"""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
DIST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.gettempdir()) / "colibri-tags-ui"
OUT.mkdir(exist_ok=True)
MODEL = "fixture-model-colibri"
health = {"status": "ok", "kv_slots": 1,
          "scheduler": {"active": 0, "capacity": 1, "queued": 0, "max_queue": 8, "completed": 0, "rejected": 0, "timed_out": 0, "cancelled": 0},
          "features": {"api": 1, "router": True, "model_switch": "load", "server_log": True, "presets": False, "request_log": True,
                       "tags": True, "json_mode": True}}
seen = []                   # the tags header of each chat, as the router would log it
queries = []                # the query strings GET /v1/requests was asked
lock = threading.Lock()


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
            self.reply({"data": [{"id": MODEL, "capabilities": ["chat"], "context_length": 32768}]})
        elif url.path == "/health":
            self.reply(health)
        elif url.path == "/v1/requests":
            query = parse_qs(url.query)
            wanted = dict(item.split(":", 1) for item in query.get("tag", []))
            with lock:
                queries.append(url.query)
                rows = [{"ts": "2026-11-17T10:00:0%d.000Z" % i, "path": "/v1/chat/completions", "status": 200, "seconds": 1.5, "model": MODEL,
                         "prompt_tokens": 10, "completion_tokens": 4, **({"tags": tags} if tags else {})}
                        for i, tags in enumerate(seen)
                        if all(str(tags.get(k)) == v for k, v in wanted.items())]
            self.reply({"object": "list", "enabled": True, "path": "requests.jsonl", "bodies": False, "data": list(reversed(rows))})
        elif url.path in ("/experts", "/profile", "/v1/models/installed", "/v1/presets", "/v1/models/settings"):
            self.reply({"data": []})
        else:
            super().do_GET()

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        raw = self.headers.get("X-Colibri-Tags")
        with lock:
            seen.append(json.loads(raw) if raw else None)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ("An ", "answer."):
            self.wfile.write(("data: " + json.dumps({"model": MODEL, "choices": [{"delta": {"content": piece}}]}) + "\n\n").encode())
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(("data: " + json.dumps({"model": MODEL, "choices": [{"delta": {}, "finish_reason": "stop"}]}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_port}"
errors = []
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def send(page, text, count, answers):
    """Send a turn and wait until its answer is on screen (`answers` of them in the conversation) and the page is ready again."""
    page.locator("#draft").fill(text)
    page.locator("#draft").press("Enter")
    expect(page.get_by_text("An answer.")).to_have_count(answers)
    with lock:
        assert len(seen) >= count, seen
    page.wait_for_timeout(400)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"], executable_path=os.environ.get("CHROME_PATH") or None)
    page = browser.new_page(viewport={"width": 1280, "height": 900}, locale="en-US", reduced_motion="reduce")
    page.set_default_timeout(10000)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    expect(page.locator(".model-button")).to_contain_text("fixture-model")

    send(page, "first question", 1, 1)
    send(page, "second question", 2, 2)
    first, second = seen[0], seen[1]
    check("a chat carries the app and a chat tag", bool(first) and first.get("app") == "dashboard" and str(first.get("chat", "")).startswith("c-"), str(first))
    check("the next turn of the same conversation carries the same chat", first == second, f"{first} / {second}")

    page.get_by_role("button", name="New chat", exact=True).first.click()
    send(page, "a new conversation", 3, 1)
    third = seen[2]
    check("a new chat gets a new tag", bool(third) and third["chat"] != first["chat"] and third["app"] == "dashboard", str(third))

    page.get_by_role("button", name="Settings", exact=True).click()
    page.locator(".settings-tabs").get_by_role("button", name="Requests", exact=True).click()
    expect(page.locator(".requests-table tbody tr")).to_have_count(3)
    check("the Requests tab shows each request's tags", page.locator(".request-tag").count() == 6, str(page.locator(".request-tag").count()))
    page.screenshot(path=str(OUT / "1-requests.png"))

    page.locator(".request-tag", has_text=f"chat:{first['chat']}").first.click()
    expect(page.locator(".requests-table tbody tr")).to_have_count(2)
    with lock:
        asked = list(queries)
    check("clicking a tag lists only that chat's requests", True)
    check("the request is for that tag, escaped", any(f"tag=chat%3A{first['chat']}" in q for q in asked), str(asked[-1:]))
    check("the page says what it is filtered by", page.get_by_text(f"tagged chat:{first['chat']}").count() >= 1)
    page.screenshot(path=str(OUT / "2-filtered.png"))

    page.get_by_role("button", name="Show all", exact=True).click()
    expect(page.locator(".requests-table tbody tr")).to_have_count(3)
    check("Show all brings the rest back", True)
    browser.close()

check("no page errors", not errors, "; ".join(errors))
failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
