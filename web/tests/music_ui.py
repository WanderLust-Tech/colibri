"""Browser check of the music view, against the real gateway with a stub engine behind it.

Run `npm run build`, then `python3 tests/music_ui.py` (requires Playwright and a Chromium: set COLI_CHROMIUM to a
browser executable if Playwright's own is not installed). No model and no weights: c/tools/acestep_stub.py plays the
engine, and c/openai_server.py is the real gateway, serving web/dist and answering POST /v1/audio/music with its
real server-sent events, so what reaches the page is what a real engine's song would be framed as.

What it holds the page to: a music model shows the music composer (length, steps, language, seed, a lyrics switch)
and no picture attachment; the request goes out as the contract's body; progress shows while the engine works and a
cancel stops it and leaves the page ready; the finished song is a player the browser can really play (its length is
read back from the audio element), with the length, steps and seed under it; Download saves a WAV named after the
caption and seed; "New seed" makes another song under it; lyrics and a language travel only when given; an engine
failure is a message, not a stuck page; a text model's conversation never sees any of it.
"""
import json
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

import music_engine  # noqa: E402
from acestep_fixture import STUB, make_fake_checkpoint, stub_env  # noqa: E402
from openai_server import APIServer  # noqa: E402

MODEL_ID = "ace-step-1.5-colibri"
SHOTS = os.environ.get("COLI_SHOTS")


class Gateway:
    def __init__(self, delay=0.0):
        self.tmp = tempfile.TemporaryDirectory()
        model = make_fake_checkpoint(Path(self.tmp.name) / "ace-step")
        self.engine = music_engine.MusicEngine(STUB, model, env=stub_env(delay), on_log=lambda _m: None)
        self.server = APIServer(("127.0.0.1", 0), self.engine, MODEL_ID, "", 16, max_queue=8, queue_timeout=30)
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/"

    def close(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.engine.close(timeout=5)
        self.tmp.cleanup()


def shot(page, name):
    if SHOTS:
        page.screenshot(path=str(Path(SHOTS) / f"music-{name}.png"))


def open_page(browser, url):
    context = browser.new_context(locale="en-US", viewport={"width": 1100, "height": 800}, accept_downloads=True)
    page = context.new_page()
    posts = []

    def on_request(request):
        if request.method == "POST" and request.url.endswith("/v1/audio/music"):
            posts.append(json.loads(request.post_data or "{}"))
    page.on("request", on_request)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url)
    expect(page.locator("#draft")).to_be_visible()
    return context, page, posts, errors


def composer(page):
    return page.locator("form.composer")


def send(page, caption):
    page.locator("#draft").fill(caption)
    page.get_by_role("button", name="Make music").click()


def check_composer(page):
    form = composer(page)
    expect(page.locator("#draft")).to_have_attribute("placeholder", "Describe the music: genre, mood, instruments…")
    for label in ("Length of the song, in seconds", "Sampler steps", "Language of the vocals"):
        expect(form.get_by_label(label)).to_be_visible()
    expect(form.get_by_label("Seed", exact=True)).to_be_visible()
    expect(form.get_by_role("button", name="Lyrics")).to_be_visible()
    expect(form.locator(".attach-chip")).to_have_count(0)               # music has no pictures to attach
    expect(form.locator(".reasoning-chip")).to_have_count(0)            # nor a reasoning switch
    expect(page.get_by_role("button", name="Make music")).to_be_disabled()   # nothing to say yet
    expect(page.locator("#draft")).to_have_attribute("aria-label", "Describe the music: genre, mood, instruments…")


def make_one(page, posts, caption, duration="2", steps="3", seed="7"):
    form = composer(page)
    form.get_by_label("Length of the song, in seconds").fill(duration)
    form.get_by_label("Sampler steps").fill(steps)
    form.get_by_label("Seed", exact=True).fill(seed)
    before = len(posts)
    send(page, caption)
    return before


def run(browser, gateway, slow_gateway):
    # ---- a song, end to end
    context, page, posts, errors = open_page(browser, gateway.url)
    check_composer(page)
    shot(page, "empty")

    before = make_one(page, posts, "calm piano with soft strings")
    request = posts[before]
    assert request == {"model": MODEL_ID, "prompt": "calm piano with soft strings", "duration": 2, "steps": 3, "seed": 7, "stream": True}, request
    expect(page.locator("audio.music-player")).to_have_count(1, timeout=20000)
    player = page.locator("audio.music-player").first
    src = player.get_attribute("src")
    assert src and src.startswith("blob:"), src
    seconds = page.wait_for_function("(el) => (el.readyState >= 1 && Number.isFinite(el.duration)) ? el.duration : false", arg=player.element_handle(), timeout=10000).json_value()
    assert abs(seconds - 2.0) < 0.05, f"the browser reads {seconds}s of audio, expected 2.0"
    meta = page.locator(".image-meta").first
    expect(meta).to_contain_text("0:02")
    expect(meta).to_contain_text("3 steps")
    expect(meta).to_contain_text("seed 7")
    expect(page.locator(".message.user").first).to_contain_text("calm piano with soft strings")
    shot(page, "song")

    # ---- Download: a WAV named after the caption and the seed
    with page.expect_download() as download:
        page.get_by_role("button", name="Download WAV").first.click()
    saved = download.value
    assert saved.suggested_filename == "colibri-calm-piano-with-soft-strings-7.wav", saved.suggested_filename
    data = Path(saved.path()).read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE" and len(data) == 44 + 2 * 8000 * 4, (data[:12], len(data))

    # ---- a new seed makes another song under it, with the same caption, length and steps
    page.get_by_role("button", name="New seed").first.click()
    expect(page.locator("audio.music-player")).to_have_count(2, timeout=20000)
    assert posts[-1]["prompt"] == "calm piano with soft strings" and posts[-1]["duration"] == 2 and posts[-1]["steps"] == 3
    assert posts[-1]["seed"] != 7, posts[-1]
    assert page.locator("audio.music-player").nth(1).get_attribute("src") != src

    # ---- lyrics and a language travel only when given
    form = composer(page)
    form.get_by_role("button", name="Lyrics").click()
    box = page.get_by_role("textbox", name="Lyrics")
    expect(box).to_be_visible()
    box.fill("[Verse]\nwalking down the street")
    expect(page.locator(".music-lyrics small")).to_contain_text("/ 32768 bytes")
    form.get_by_label("Language of the vocals").select_option("en")
    form.get_by_label("Seed", exact=True).fill("11")
    shot(page, "lyrics")
    send(page, "upbeat pop")
    expect(page.locator("audio.music-player")).to_have_count(3, timeout=20000)
    assert posts[-1]["lyrics"] == "[Verse]\nwalking down the street" and posts[-1]["language"] == "en", posts[-1]
    assert "lyrics" not in posts[before] and "language" not in posts[before]
    # lyrics alone are enough
    page.locator("#draft").fill("")
    expect(page.get_by_role("button", name="Make music")).to_be_enabled()
    page.get_by_role("button", name="Make music").click()
    expect(page.locator("audio.music-player")).to_have_count(4, timeout=20000)
    assert posts[-1]["prompt"] == "" and posts[-1]["lyrics"].startswith("[Verse]"), posts[-1]
    expect(page.locator(".message.user").last).to_contain_text("♪ [Verse]")

    # ---- a length the engine would refuse cannot be sent, and a long caption is not the page's to cut
    form.get_by_label("Length of the song, in seconds").fill("9999")
    expect(page.get_by_role("button", name="Make music")).to_be_disabled()
    form.get_by_label("Length of the song, in seconds").blur()
    expect(form.get_by_label("Length of the song, in seconds")).to_have_value("600")
    # an invalid number of steps holds sending back too, and a decimal point can be typed
    form.get_by_label("Sampler steps").fill("99")
    expect(page.get_by_role("button", name="Make music")).to_be_disabled()
    form.get_by_label("Sampler steps").fill("3")
    expect(page.get_by_role("button", name="Make music")).to_be_enabled()
    duration = form.get_by_label("Length of the song, in seconds")
    duration.fill("")
    duration.press_sequentially("2.5")
    expect(duration).to_have_value("2.5")

    # ---- an engine failure is a message, and the page goes on
    form.get_by_label("Length of the song, in seconds").fill("1")
    send(page, "__stub_error__ piano")
    expect(page.get_by_role("alert").first).to_contain_text("asked to fail", timeout=20000)
    expect(page.get_by_role("button", name="Try again").first).to_be_visible()
    send(page, "calm piano again")
    expect(page.locator("audio.music-player")).to_have_count(5, timeout=20000)
    assert not errors, errors
    context.close()

    # ---- cancel: progress shows, Cancel stops it, the page is ready for the next
    context, page, posts, errors = open_page(browser, slow_gateway.url)
    form = composer(page)
    form.get_by_label("Sampler steps").fill("20")
    form.get_by_label("Length of the song, in seconds").fill("1")
    send(page, "a very long piece")
    expect(page.get_by_role("progressbar", name="Music generation progress")).to_be_visible(timeout=10000)
    expect(page.locator(".image-stage")).to_contain_text("Composing, step", timeout=15000)
    shot(page, "progress")
    page.get_by_role("button", name="Cancel").first.click()
    expect(page.get_by_text("Generation cancelled")).to_be_visible(timeout=15000)
    expect(page.locator("audio.music-player")).to_have_count(0)
    form.get_by_label("Sampler steps").fill("1")
    send(page, "a short one")
    expect(page.locator("audio.music-player")).to_have_count(1, timeout=20000)
    assert not errors, errors
    context.close()


def main():
    gateway, slow = Gateway(), Gateway(delay=0.4)
    launch = {"executable_path": os.environ["COLI_CHROMIUM"]} if os.environ.get("COLI_CHROMIUM") else {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch)
            try:
                run(browser, gateway, slow)
            finally:
                browser.close()
    finally:
        gateway.close()
        slow.close()
    print("music UI: ok")


if __name__ == "__main__":
    main()
