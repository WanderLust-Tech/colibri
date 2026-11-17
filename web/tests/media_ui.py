"""Browser check of the settings a finished picture or song carries, against the real gateway with stub engines.

Run `npm run build`, then `python3 tests/media_ui.py` (requires Playwright and a Chromium: set CHROME_PATH to a browser
executable if Playwright's own is not installed). c/openai_server.py is the real gateway, serving web/dist; the engines
behind it are the stubs the gateway's own tests use, so no model and no weights are needed.

What it holds the page to: the PNG the Download button saves, and the song's WAV, carry the settings that made them (read
back with the server's own c/media_meta.py); a fresh page that opens such a file through the settings button gets the
model, prompt, size, steps and seed (a song: length, language and lyrics) back in its controls; a file with no settings
is a message and leaves the controls alone; a chat model has no such button.
"""
import os
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT.parent / "c"
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tests"))

import image_engine  # noqa: E402
import media_meta  # noqa: E402
import music_engine  # noqa: E402
from acestep_fixture import STUB as MUSIC_STUB, make_fake_checkpoint, stub_env as music_env  # noqa: E402
from openai_server import APIServer  # noqa: E402
from qwen_image_fixture import STUB as IMAGE_STUB, make_fake_pipeline, stub_env as image_env  # noqa: E402

IMAGE_MODEL = "qwen-image-colibri"
MUSIC_MODEL = "ace-step-1.5-colibri"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


class Gateway:
    def __init__(self, kind, workdir):
        if kind == "image":
            self.engine = image_engine.ImageEngine(IMAGE_STUB, make_fake_pipeline(Path(workdir) / "qwen-image"),
                                                   env=image_env(0.0), on_log=lambda _m: None)
            model = IMAGE_MODEL
        else:
            self.engine = music_engine.MusicEngine(MUSIC_STUB, make_fake_checkpoint(Path(workdir) / "ace-step"),
                                                   env=music_env(0.0), on_log=lambda _m: None)
            model = MUSIC_MODEL
        self.server = APIServer(("127.0.0.1", 0), self.engine, model, "", 16, max_queue=8, queue_timeout=30)
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/"

    def close(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.engine.close(timeout=5)


def open_page(browser, url):
    context = browser.new_context(locale="en-US", viewport={"width": 1100, "height": 800}, accept_downloads=True)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url)
    expect(page.locator("#draft")).to_be_visible()
    return context, page, errors


def open_settings(page, path):
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Open the settings of a saved picture or song").click()
    chooser.value.set_files(str(path))


def picture(browser, gateway, work):
    context, page, errors = open_page(browser, gateway.url)
    form = page.locator("form.composer")
    form.get_by_label("Denoising steps").fill("2")
    form.get_by_label("Seed", exact=True).fill("5")
    form.get_by_label("Image size").select_option("512x512")
    page.locator("#draft").fill("a lighthouse at dusk, \"quoted\"")
    page.get_by_role("button", name="Generate image").click()
    download = page.get_by_role("button", name="Download PNG")
    expect(download).to_be_visible(timeout=30000)
    with page.expect_download() as saved:
        download.click()
    path = work / "made.png"
    saved.value.save_as(str(path))
    found = media_meta.read_png(path.read_bytes())
    check("the PNG the page saves carries the settings that made it",
          found == {"app": "colibri", "version": 1, "kind": "image", "model": IMAGE_MODEL,
                    "prompt": "a lighthouse at dusk, \"quoted\"", "width": 512, "height": 512, "steps": 2, "seed": 5}, str(found))
    context.close()

    # ---- a fresh page: different controls, then the file
    context, page, errors = open_page(browser, gateway.url)
    form = page.locator("form.composer")
    form.get_by_label("Denoising steps").fill("9")
    form.get_by_label("Seed", exact=True).fill("123")
    form.get_by_label("Image size").select_option("1024x1024")
    page.locator("#draft").fill("something else")
    open_settings(page, path)
    expect(page.locator("#draft")).to_have_value("a lighthouse at dusk, \"quoted\"")
    expect(form.get_by_label("Denoising steps")).to_have_value("2")
    expect(form.get_by_label("Seed", exact=True)).to_have_value("5")
    expect(form.get_by_label("Image size")).to_have_value("512x512")
    check("opening the file puts the prompt, steps, seed and size back", True)
    check("the page raised no error", errors == [], str(errors))

    # ---- a picture from somewhere else
    plain = work / "plain.png"
    plain.write_bytes(image_engine.encode_png(4, 4, bytes(range(64)), 4))
    open_settings(page, plain)
    expect(page.locator(".error-banner")).to_contain_text("carries no Colibri settings")
    expect(page.locator("#draft")).to_have_value("a lighthouse at dusk, \"quoted\"")
    expect(form.get_by_label("Seed", exact=True)).to_have_value("5")
    check("a file with no settings is a message and changes nothing", True)
    context.close()


def song(browser, gateway, work):
    context, page, errors = open_page(browser, gateway.url)
    form = page.locator("form.composer")
    form.get_by_label("Length of the song, in seconds").fill("2")
    form.get_by_label("Sampler steps").fill("3")
    form.get_by_label("Seed", exact=True).fill("7")
    form.get_by_label("Language of the vocals").select_option("en")
    form.get_by_role("button", name="Lyrics").click()
    form.get_by_label("Lyrics", exact=True).fill("la la\nla")
    page.locator("#draft").fill("calm piano, soft strings")
    page.get_by_role("button", name="Make music").click()
    download = page.get_by_role("button", name="Download WAV")
    expect(download).to_be_visible(timeout=60000)
    with page.expect_download() as saved:
        download.click()
    path = work / "made.wav"
    saved.value.save_as(str(path))
    found = media_meta.read_wav(path.read_bytes())
    ok = bool(found) and (found["kind"], found["model"], found["prompt"], found["lyrics"], found["language"], found["seed"]) == (
        "music", MUSIC_MODEL, "calm piano, soft strings", "la la\nla", "en", 7) and found["steps"] == 3
    check("the WAV the page saves carries the settings that made it", ok, str(found))
    context.close()

    context, page, errors = open_page(browser, gateway.url)
    form = page.locator("form.composer")
    form.get_by_label("Length of the song, in seconds").fill("9")
    form.get_by_label("Sampler steps").fill("6")
    form.get_by_label("Seed", exact=True).fill("123")
    page.locator("#draft").fill("something else")
    open_settings(page, path)
    expect(page.locator("#draft")).to_have_value("calm piano, soft strings")
    expect(form.get_by_label("Length of the song, in seconds")).to_have_value("2")
    expect(form.get_by_label("Sampler steps")).to_have_value("3")
    expect(form.get_by_label("Seed", exact=True)).to_have_value("7")
    expect(form.get_by_label("Language of the vocals")).to_have_value("en")
    expect(form.get_by_label("Lyrics", exact=True)).to_have_value("la la\nla")
    check("opening the file puts the prompt, length, steps, seed, language and lyrics back", True)
    check("the page raised no error", errors == [], str(errors))
    context.close()


with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
    work = Path(tmp)
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"],
                                executable_path=os.environ.get("CHROME_PATH") or os.environ.get("COLI_CHROMIUM") or None)
    for kind, run in (("image", picture), ("music", song)):
        gateway = Gateway(kind, work)
        try:
            run(browser, gateway, work)
        finally:
            gateway.close()
    browser.close()

failed = [name for name, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
