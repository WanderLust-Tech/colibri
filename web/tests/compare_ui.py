"""Browser check of the compare view (C5): the same prompt to several models, in turn, side by side.

Run `npm run build`, then `python3 tests/compare_ui.py [dist-dir [screenshot-dir]]` (needs Playwright and its Chromium;
set CHROME_PATH to use another Chromium binary). A local fixture server stands in for the gateway with three models:
"alpha" and "beta" answer (alpha slowly, so an overlap would show), "gamma" fails with a full queue (and, for the prompt "Cut off", is cut off after one word). No model or
credentials needed. Screenshots go to the temp folder.
"""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
DIST = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.gettempdir()) / "colibri-compare-ui"
OUT.mkdir(exist_ok=True)
health = {"status": "ok", "kv_slots": 1, "scheduler": {"active": 0, "capacity": 1, "queued": 0, "max_queue": 8, "completed": 0, "rejected": 0, "timed_out": 0, "cancelled": 0}}
ANSWERS = {"alpha-colibri": "Alpha says: forty-two.", "beta-colibri": "Beta says: it depends."}
spans = []   # (model, start, end) of each request, to prove they ran one at a time
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
        if self.path == "/v1/models":
            self.reply({"data": [
                {"id": "alpha-colibri", "capabilities": ["chat"], "context_length": 32768},
                {"id": "beta-colibri", "capabilities": ["chat"], "context_length": 65536},
                {"id": "gamma-colibri", "capabilities": ["chat"]},
                {"id": "picture-colibri", "capabilities": ["image_generation"]},
            ]})
        elif self.path == "/health":
            self.reply(health)
        elif self.path in ("/experts", "/profile"):
            self.reply({})
        else:
            super().do_GET()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body["model"]
        start = time.monotonic()
        cut = model == "gamma-colibri" and body["messages"][0]["content"] == "Cut off"
        if model == "gamma-colibri" and not cut:
            self.reply({"error": {"message": "The inference queue is full."}}, 429)
            with lock:
                spans.append((model, start, time.monotonic()))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.8 if model == "alpha-colibri" else 0.2)   # a model "loading"
        if cut:   # the engine dies after one word: no finish reason, no [DONE]
            self.wfile.write(("data: " + json.dumps({"model": model, "choices": [{"delta": {"content": "Gamma starts "}}]}) + "\n\n").encode())
            self.wfile.flush()
            with lock:
                spans.append((model, start, time.monotonic()))
            return
        for word in ANSWERS[model].split(" "):
            self.wfile.write(("data: " + json.dumps({"model": model, "choices": [{"delta": {"content": word + " "}}]}) + "\n\n").encode())
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(("data: " + json.dumps({"model": model, "choices": [{"delta": {}, "finish_reason": "stop"}]}) + "\n\n").encode())
        usage = {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24, "prompt_tokens_details": {"cached_tokens": 0}}
        self.wfile.write(("data: " + json.dumps({"model": model, "choices": [], "usage": usage}) + "\n\ndata: [DONE]\n\n").encode())
        with lock:
            spans.append((model, start, time.monotonic()))


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
    expect(page.locator(".model-button")).to_contain_text("alpha-colibri")
    page.locator(".dock-handle").click()
    page.locator(".navigation-dock nav").get_by_role("button", name="Compare", exact=True).click()
    expect(page.locator(".compare-workspace")).to_be_visible()

    chips = page.locator(".compare-chip")
    check("only chat models are offered", chips.count() == 3 and page.locator(".compare-chip:has-text('picture')").count() == 0, str(chips.count()))
    expect(page.locator(".compare-chip.active")).to_have_count(2)
    check("the first two are chosen to start", True)
    page.locator(".compare-chip:has-text('gamma')").click()
    check("a third can be added", page.locator(".compare-chip.active").count() == 3)
    page.screenshot(path=str(OUT / "1-setup.png"))
    # clearing the choice must stick: the page re-renders (it polls the server), and used to put the first two back
    for name in ("alpha", "beta", "gamma"):
        chip = page.locator(f".compare-chip:has-text('{name}')")
        if "active" in (chip.get_attribute("class") or ""):
            chip.click()
    page.wait_for_timeout(1500)
    check("clearing every choice stays cleared", page.locator(".compare-chip.active").count() == 0)
    check("and then Compare is off", page.get_by_role("button", name="Compare", exact=True).last.is_disabled())
    for name in ("alpha", "beta", "gamma"):
        page.locator(f".compare-chip:has-text('{name}')").click()

    page.locator("#compare-prompt").fill("What is the answer?")
    page.get_by_role("button", name="Compare", exact=True).last.click()
    expect(page.locator(".compare-column")).to_have_count(3)
    expect(page.get_by_role("button", name="This one is best").first).to_be_visible(timeout=30000)
    expect(page.get_by_role("button", name="Stop")).to_have_count(0)
    page.wait_for_timeout(300)
    text = page.locator(".compare-grid").inner_text()
    headings = [h.inner_text() for h in page.locator(".compare-column h2").all()]
    check("blind: headings are Answer A, B and C", sorted(headings) == ["Answer A", "Answer B", "Answer C"], str(headings))
    check("blind: no model name in the stats lines", "alpha-colibri" not in " ".join(x.inner_text() for x in page.locator(".compare-column .reply-stats").all()))
    check("both good models answered", "forty-two" in text and "it depends" in text)
    check("the failing model shows a plain-words reason, and did not stop the rest",
          "busy with other requests" in text and "The inference queue is full." in text)
    page.screenshot(path=str(OUT / "2-blind.png"))

    ran = sorted(spans, key=lambda s: s[1])
    overlap = any(ran[i][2] > ran[i + 1][1] for i in range(len(ran) - 1))
    check("the models ran one after another, never together", len(ran) == 3 and not overlap, ", ".join(f"{m.split('-')[0]} {s:.1f}-{e:.1f}" for m, s, e in ran))

    # vote for the first answer that finished OK
    page.get_by_role("button", name="This one is best").first.click()
    page.wait_for_timeout(200)
    headings = [h.inner_text() for h in page.locator(".compare-column h2").all()]
    check("voting reveals the names", sorted(headings) == ["alpha-colibri", "beta-colibri", "gamma-colibri"], str(headings))
    rows = page.locator(".compare-scores tbody tr")
    check("the scoreboard counts the round for the two that answered", rows.count() == 2, str(rows.count()))
    wins = sum(int(r.locator("td").nth(1).inner_text()) for r in rows.all())
    check("and one win", wins == 1)
    check("the vote buttons are gone", page.get_by_role("button", name="This one is best").count() == 0)
    check("a trophy marks the pick", page.locator(".compare-column.winner").count() == 1)
    page.screenshot(path=str(OUT / "3-voted.png"))

    # a second round, not blind, two models; scores accumulate and survive a reload
    page.locator(".compare-chip:has-text('gamma')").click()
    page.locator(".compare-blind input").uncheck()
    page.get_by_role("button", name="Compare", exact=True).last.click()
    expect(page.get_by_role("button", name="This one is best").first).to_be_visible(timeout=30000)
    headings = [h.inner_text() for h in page.locator(".compare-column h2").all()]
    check("not blind: the names are shown from the start, in the order chosen", headings == ["alpha-colibri", "beta-colibri"], str(headings))
    page.get_by_role("button", name="This one is best").nth(1).click()
    page.wait_for_timeout(200)
    rounds = [int(r.locator("td").nth(2).inner_text()) for r in page.locator(".compare-scores tbody tr").all()]
    check("two rounds are on the board", rounds == [2, 2], str(rounds))
    page.reload()
    page.locator(".dock-handle").click()
    page.locator(".navigation-dock nav").get_by_role("button", name="Compare", exact=True).click()
    check("the scoreboard survives a reload", page.locator(".compare-scores tbody tr").count() == 2)
    page.get_by_role("button", name="Clear the scoreboard").click()
    check("and can be cleared", page.locator(".compare-scores").count() == 0)

    # a reply that is cut off is shown as such and cannot be voted for as if it were whole
    page.locator(".compare-chip:has-text('gamma')").click()
    page.locator("#compare-prompt").fill("Cut off")
    page.get_by_role("button", name="Compare", exact=True).last.click()
    expect(page.locator(".compare-column")).to_have_count(3)
    expect(page.get_by_role("button", name="This one is best").first).to_be_visible(timeout=30000)
    expect(page.get_by_role("button", name="Stop")).to_have_count(0)
    cut_text = page.locator(".compare-grid").inner_text()
    check("a cut-off reply says so in plain words and keeps what arrived", "connection closed before the reply was finished" in cut_text and "Gamma starts" in cut_text)
    check("and only the two whole answers can be voted for", page.get_by_role("button", name="This one is best").count() == 2, str(page.get_by_role("button", name="This one is best").count()))

    # stop in the middle: the model that is answering and those waiting are marked stopped
    page.locator("#compare-prompt").fill("Slow one")
    page.get_by_role("button", name="Compare", exact=True).last.click()
    expect(page.get_by_role("button", name="Stop")).to_be_visible()
    page.get_by_role("button", name="Stop").click()
    expect(page.get_by_role("button", name="Compare", exact=True).last).to_be_visible()
    check("stop leaves no column running or waiting", page.locator(".compare-status:has-text('Waiting')").count() == 0 and page.locator(".compare-status:has-text('Loading')").count() == 0)
    page.screenshot(path=str(OUT / "4-stopped.png"))
    browser.close()

check("no page errors", not errors, "; ".join(errors))
failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
