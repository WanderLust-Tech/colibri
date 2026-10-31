"""Browser check of editing pictures in the image view, against a local, deterministic fixture server.

Run `npm run build`, then `python3 tests/image_edit_ui.py` (requires Playwright and a Chromium: set
COLI_CHROMIUM to a browser executable if Playwright's own is not installed). No model, API credentials or
external server needed. The fixture answers /v1/models like a gateway whose engine edits pictures, takes
the multipart POST /v1/images/edits and the JSON POST /v1/images/generations, and records what it received.

What it holds the page to: the attach button exists only for a model that edits; an attached picture shows
as a thumbnail and turns the size to "Auto"; the request is multipart with the files and no `size`; the
result says what it was edited from; "Again" resends the same pictures with an explicit size; removing the
pictures goes back to plain generation; more pictures than the model takes are refused in the page.
"""
import base64
import json
import os
import struct
import threading
import zlib
from email.parser import BytesParser
from email.policy import HTTP
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def png(width, height, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


PHOTO = png(64, 48, (200, 80, 40))              # 4:3, so an edit of it is 896 x 672 at the engine's default
RESULT = base64.b64encode(png(8, 6, (90, 10, 160))).decode()


def make_server(edits):
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT / "dist"), **kwargs)

        def log_message(self, *_):
            pass

        def reply_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                entry = {"id": "qwen-image-2.1-colibri", "object": "model",
                         "capabilities": ["image_generation", "image_edit"] if edits else ["image_generation"],
                         "image": {"default_width": 768, "default_height": 512, "default_steps": 8,
                                   "min_side": 256, "max_side": 2048, "multiple": 32}}
                if edits:
                    entry["image"]["edit"] = {"max_images": 2, "max_side": 2048, "max_pixels": 4 << 20,
                                              "ref_resolution": {"default": 768, "min": 384, "max": 1024}}
                self.reply_json({"data": [entry]})
            elif self.path == "/health":
                self.reply_json({"status": "ok", "kv_slots": 1})
            else:
                super().do_GET()

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            kind = self.headers["Content-Type"].split(";")[0]
            record = {"path": self.path, "type": kind, "auth": self.headers.get("Authorization")}
            if kind == "multipart/form-data":
                message = BytesParser(policy=HTTP).parsebytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
                record["fields"], record["files"] = {}, []
                for part in message.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    if part.get_filename():
                        record["files"].append((name, part.get_filename(), part.get_payload(decode=True)))
                    else:
                        record["fields"][name] = part.get_payload(decode=True).decode()
                body = record["fields"]
            else:
                body = record["fields"] = json.loads(raw)
            requests.append(record)
            if self.path == "/v1/images/edits":
                size = {"width": 896, "height": 672} if "size" not in body else dict(
                    zip(("width", "height"), map(int, body["size"].split("x"))))
            else:
                size = {"width": 768, "height": 512}
            done = {"created": 1, "data": [{"b64_json": RESULT}],
                    "colibri": {**size, "seed": int(body.get("seed", 1)), "steps": int(body.get("steps", 8))}}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(("event: image_generation.completed\ndata: " + json.dumps(done) +
                              "\n\ndata: [DONE]\n\n").encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


def open_page(browser, server):
    page = browser.new_page(viewport={"width": 1280, "height": 900}, locale="en-US", reduced_motion="reduce")
    page.set_default_timeout(10000)
    page.errors = []
    page.on("pageerror", lambda e: page.errors.append(str(e)))
    page.goto(f"http://127.0.0.1:{server.server_port}")
    expect(page.locator(".model-button")).to_contain_text("qwen-image-2.1-colibri")
    return page


def attach(page, *names):
    page.locator(".composer input[type=file]").set_input_files([
        {"name": name, "mimeType": "image/png", "buffer": PHOTO} for name in names])


def send(page, text):
    page.locator("#draft").fill(text)
    page.get_by_role("button", name="Generate image").click()


with_edits, requests = make_server(edits=True)
without_edits, _ = make_server(edits=False)
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"],
                                    **({"executable_path": os.environ["COLI_CHROMIUM"]} if os.environ.get("COLI_CHROMIUM") else {}))

        # -- a model that cannot edit: no attach button, no "Auto" size ---------------------------------
        page = open_page(browser, without_edits)
        assert page.get_by_role("button", name="Attach a picture to edit").count() == 0
        assert page.locator(".composer input[type=file]").count() == 0
        assert page.get_by_label("Image size").locator("option[value=auto]").count() == 0
        assert not page.errors, page.errors
        page.close()

        # -- a model that edits ------------------------------------------------------------------------
        page = open_page(browser, with_edits)
        expect(page.get_by_role("button", name="Attach a picture to edit")).to_be_visible()
        size = page.get_by_label("Image size")
        assert size.locator("option[value=auto]").count() == 0, "Auto means nothing without a picture"
        expect(page.locator("#draft")).to_have_attribute("placeholder", "Describe the image to generate…")

        attach(page, "photo.png")
        expect(page.locator(".composer-attachments .attachment img")).to_have_count(1)
        expect(size).to_have_value("auto")
        expect(page.locator("#draft")).to_have_attribute("placeholder", "Describe the edit: what should change?")
        page.screenshot(path=str(Path(os.environ.get("TMPDIR", "/tmp")) / "colibri-image-edit-attached.png"))

        send(page, "make the sky purple")
        expect(page.locator(".image-meta")).to_contain_text("896 × 672")
        expect(page.locator(".image-meta")).to_contain_text("edited from 1")
        edit = requests[-1]
        assert edit["path"] == "/v1/images/edits" and edit["type"] == "multipart/form-data", edit["path"]
        assert edit["fields"]["prompt"] == "make the sky purple" and edit["fields"]["stream"] == "true", edit["fields"]
        assert "size" not in edit["fields"], "an automatic size is the engine's to derive"
        assert edit["fields"]["model"] == "qwen-image-2.1-colibri" and edit["fields"]["n"] == "1"
        assert [(n, f, b) for n, f, b in edit["files"]] == [("image", "photo.png", PHOTO)], edit["files"]
        assert edit["auth"] is None
        # the picture stays attached for the next refinement, and the user's turn is in the transcript
        expect(page.locator(".composer-attachments .attachment img")).to_have_count(1)
        expect(page.locator(".message.user")).to_have_count(1)

        # -- Again: the same pictures, the size now explicit (what was drawn) ----------------------------
        page.get_by_role("button", name="Again with a new seed").click()
        expect(page.locator(".image-meta")).to_have_count(2)
        again = requests[-1]
        assert again["path"] == "/v1/images/edits" and again["fields"]["size"] == "896x672", again["fields"]
        assert again["files"][0][2] == PHOTO and again["fields"]["seed"] != edit["fields"]["seed"]

        # -- choosing a size is a choice: it is sent with the pictures ---------------------------------------
        size.select_option("1024x576")
        send(page, "now wider")
        expect(page.locator(".image-meta")).to_have_count(3)
        assert requests[-1]["fields"]["size"] == "1024x576" and requests[-1]["files"], requests[-1]["fields"]
        size.select_option("auto")
        assert size.input_value() == "auto"

        # -- the limit: this model takes two ------------------------------------------------------------------
        attach(page, "second.png")
        expect(page.locator(".composer-attachments .attachment img")).to_have_count(2)
        attach(page, "third.png")
        expect(page.get_by_role("alert")).to_contain_text("only a few pictures")
        expect(page.locator(".composer-attachments .attachment img")).to_have_count(2)

        # -- removing every picture goes back to plain generation ------------------------------------------------
        while page.locator(".composer-attachments .attachment").count():
            page.get_by_role("button", name="Remove this picture").first.click()
        assert size.locator("option[value=auto]").count() == 0
        send(page, "a plain fox")
        expect(page.locator(".image-meta")).to_have_count(4)
        assert requests[-1]["path"] == "/v1/images/generations" and requests[-1]["type"] == "application/json"
        assert requests[-1]["fields"]["size"] != "auto" and "image" not in requests[-1]["fields"]
        page.screenshot(path=str(Path(os.environ.get("TMPDIR", "/tmp")) / "colibri-image-edit.png"))
        assert not page.errors, page.errors
        browser.close()
    print("image edit UI ok:", len(requests), "requests")
finally:
    for server in (with_edits, without_edits):
        server.shutdown()
