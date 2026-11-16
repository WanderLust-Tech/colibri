"""Browser check of the reply line (C1), the plain-words errors (C3) and the message actions (C4).

Run `npm run build`, then `python3 tests/reply_ui.py [dist-dir [screenshot-dir]]` (needs Playwright and its Chromium; set CHROME_PATH to use another Chromium binary).
A local fixture server stands in for the gateway: it streams a reply with usage and cached tokens, and answers the
message "TOOLONG" with the context-length error. No model or credentials needed. Screenshots go to the temp folder.
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
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.gettempdir()) / "colibri-reply-ui"
OUT.mkdir(exist_ok=True)
requests = []
health = {"status": "ok", "kv_slots": 1, "continue_assistant": True,
          "scheduler": {"active": 0, "capacity": 1, "queued": 0, "max_queue": 8, "completed": 3, "rejected": 0, "timed_out": 0, "cancelled": 0}}
MODEL = "fixture-model-colibri"
CONTEXT = ("This model's maximum context length is 32768 tokens, however your messages resulted in at least 40111 tokens. "
           "Please shorten the conversation, or restart the server with a larger CTX.")


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
            self.reply({"data": [{"id": MODEL, "capabilities": ["chat"], "context_length": 32768}]})
        elif self.path == "/health":
            self.reply(health)
        elif self.path in ("/experts", "/profile"):
            self.reply({})
        else:
            super().do_GET()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append(body)
        last = body["messages"][-1]["content"]
        if last == "TOOLONG":
            self.reply({"error": {"message": CONTEXT}}, 400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        cached = max(0, sum(len(m["content"]) for m in body["messages"][:-1]) // 4)
        for piece in ("Here ", "is ", "an ", "answer."):
            self.wfile.write(("data: " + json.dumps({"model": MODEL, "choices": [{"delta": {"content": piece}}]}) + "\n\n").encode())
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(("data: " + json.dumps({"model": MODEL, "choices": [{"delta": {}, "finish_reason": "stop"}]}) + "\n\n").encode())
        usage = {"prompt_tokens": 120 + cached, "completion_tokens": 4, "total_tokens": 124 + cached,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        self.wfile.write(("data: " + json.dumps({"model": MODEL, "choices": [], "usage": usage}) + "\n\ndata: [DONE]\n\n").encode())


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_port}"
errors = []
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def send(page, text):
    page.locator("#draft").fill(text)
    page.locator("#draft").press("Enter")


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"], executable_path=os.environ.get("CHROME_PATH") or None)
    page = browser.new_page(viewport={"width": 1280, "height": 900}, locale="en-US", reduced_motion="reduce")
    page.set_default_timeout(10000)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url)
    expect(page.locator(".model-button")).to_contain_text("fixture-model")

    send(page, "Hello there")
    expect(page.locator(".message.assistant .message-body").first).to_have_text("Here is an answer.")
    page.wait_for_timeout(300)
    line = page.locator(".reply-stats").first.inner_text()
    check("C1 stats line shows model, tokens and window", "fixture-model" in line and "tok" in line and "32,768" in line, line)
    send(page, "A second question")
    expect(page.locator(".message.assistant .message-body").nth(1)).to_have_text("Here is an answer.")
    page.wait_for_timeout(300)
    line2 = page.locator(".reply-stats").nth(1).inner_text()
    check("C1 second reply says how much was cached", "prompt tokens cached" in line2, line2)
    page.screenshot(path=str(OUT / "1-stats.png"))

    # C4: the buttons
    check("C4 edit button on both user messages", page.locator('.message.user .icon-action').count() == 2)
    check("C4 branch button on both replies", page.locator('.message.assistant .icon-action[aria-label^="Branch"]').count() == 2)
    check("C4 rewrite buttons only on the last reply",
          page.locator('.icon-action[aria-label^="Rewrite shorter"]').count() == 1 and page.locator('.icon-action[aria-label^="Explain simpler"]').count() == 1)
    page.screenshot(path=str(OUT / "2-actions.png"))

    page.locator('.icon-action[aria-label^="Rewrite shorter"]').click()
    expect(page.locator(".message.user")).to_have_count(3)
    page.wait_for_timeout(300)
    check("C4 rewrite sent a normal follow-up", requests[-1]["messages"][-1]["content"].startswith("Rewrite your last answer shorter"),
          requests[-1]["messages"][-1]["content"])
    before = [m["content"] for m in requests[-2]["messages"]]
    after = [m["content"] for m in requests[-1]["messages"]]
    check("C4 the earlier messages were sent unchanged (cache-friendly)", after[:len(before)] == before, f"{len(before)} then {len(after)} messages")

    # C3: a context error
    send(page, "TOOLONG")
    expect(page.locator(".error-banner")).to_be_visible()
    banner = page.locator(".error-banner").inner_text()
    check("C3 plain-words explanation with the numbers", "40,111 tokens against 32,768" in banner, banner[:120])
    check("C3 the server's own message is still shown", "maximum context length" in banner)
    check("C3 offers a new chat", page.locator('.error-banner button:has-text("New chat")').count() == 1)
    page.screenshot(path=str(OUT / "3-error.png"))

    # C4: edit the first user message
    page.locator('.message.user .icon-action').first.click()
    page.wait_for_timeout(200)
    check("C4 edit puts the text back in the composer", page.locator("#draft").input_value() == "Hello there", page.locator("#draft").input_value())
    check("C4 edit cut the conversation there", page.locator(".message").count() == 0)
    page.screenshot(path=str(OUT / "4-edited.png"))
    page.get_by_role("button", name="Conversations").first.click()
    page.wait_for_timeout(300)
    page.screenshot(path=str(OUT / "5-history.png"))
    check("C4 the original is in History", page.locator("text=Hello there").count() >= 1)
    browser.close()

check("no page errors", not errors, "; ".join(errors))
failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
