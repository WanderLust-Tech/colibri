"""Browser check of the model switcher against `coli router`'s router, with real gateways (stub engines) behind it.

Run `npm run build`, then `python3 tests/router_ui.py` (requires Playwright and a Chromium: set COLI_CHROMIUM to a
browser executable if Playwright's own is not installed). The router (c/router.py) serves web/dist and starts a real
`coli serve` for each model when it is asked for; the engines are the stubs of the music and image tests.

What it holds the page to: with a router, the models list says what is loaded and what is not, with a Load button on
the ones that are not (no confirmation, no restart); Load shows that the model is loading, and when the router answers
the row says Loaded, with its memory and when it unloads; the other model stays as it was; and the page that was open
is still working.
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

import router  # noqa: E402
from acestep_fixture import STUB as MUSIC_STUB, make_fake_checkpoint as make_music, stub_env as music_env  # noqa: E402
from qwen_image_fixture import STUB as IMAGE_STUB, make_fake_pipeline as make_image, stub_env as image_env  # noqa: E402


class Router:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        make_image(root / "models" / "qwen-image")
        make_music(root / "models" / "ace-step")
        catalog = router.Catalog(roots=[str(root / "models")], engine_ready=lambda family: True)

        def worker_env(entry):
            if entry["modality"] == "image":
                return {"COLI_ENGINE": str(IMAGE_STUB), "COLI_IMAGE_DIR": str(root / "images"),
                        **{k: v for k, v in image_env().items() if k.startswith("QWENIMAGE")}}
            return {"COLI_ENGINE": str(MUSIC_STUB), "COLI_MUSIC_DIR": str(root / "music"),
                    **{k: v for k, v in music_env().items() if k.startswith("ACESTEP")}}

        self.pool = router.Pool(catalog, worker_env=worker_env, grace=8.0, state=router.State())
        self.server = router.RouterServer(("127.0.0.1", 0), self.pool, catalog, router.Gate(), None,
                                          ("http://127.0.0.1:1",))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"

    def states(self):
        return {w.entry["name"]: w.state for w in self.pool.snapshot()}

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.pool.close()
        self.tmp.cleanup()


def open_tab(page, name):
    page.get_by_role("button", name="Settings", exact=True).click()
    page.locator(".settings-tabs").get_by_role("button", name=name, exact=True).click()


def run(browser, rig):
    context = browser.new_context(locale="en-US", viewport={"width": 1100, "height": 800})
    page = context.new_page()
    page.set_default_timeout(20000)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    switches = []
    page.on("request", lambda request: switches.append(request.url) if request.url.endswith("/v1/models/switch") else None)
    page.goto(rig.url, wait_until="networkidle")
    page.wait_for_timeout(800)

    open_tab(page, "Connection")
    page.get_by_role("button", name="Probe server", exact=True).click()
    expect(page.locator(".settings-card .connection-state")).to_have_text("Engine reachable")

    open_tab(page, "Model")
    rows = page.locator(".model-row")
    expect(rows).to_have_count(2)
    expect(page.locator(".model-switcher .field-help").first).to_contain_text("loads when you use it")
    assert rig.states() == {}, "nothing is loaded before anything is asked for"
    for index in range(2):
        expect(rows.nth(index).get_by_role("button", name="Load", exact=True)).to_be_enabled()
    expect(page.locator(".model-current")).to_have_count(0)

    music = rows.filter(has_text="ACE-Step")
    music.get_by_role("button", name="Load", exact=True).click()           # no confirmation: nothing restarts
    expect(page.locator(".model-switching")).to_be_visible()
    expect(page.locator(".model-switching")).to_contain_text("Loading")
    expect(music.locator(".model-current")).to_have_text("Loaded", timeout=30000)
    assert len(switches) == 1, switches
    assert rig.states() == {"ACE-Step 1.5": "ready"}, rig.states()
    expect(music.locator("small").first).to_contain_text("unloads in")
    image = rows.filter(has_text="Qwen-Image")
    expect(image.get_by_role("button", name="Load", exact=True)).to_be_enabled()      # the other one is as it was

    image.get_by_role("button", name="Load", exact=True).click()
    expect(image.locator(".model-current")).to_have_text("Loaded", timeout=30000)
    expect(music.locator(".model-current")).to_have_text("Loaded")                    # both fit: both stay
    assert rig.states() == {"ACE-Step 1.5": "ready", "Qwen-Image-2.1": "ready"}, rig.states()
    assert not errors, errors
    context.close()


def main():
    rig = Router()
    launch = {"executable_path": os.environ["COLI_CHROMIUM"]} if os.environ.get("COLI_CHROMIUM") else {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch)
            try:
                run(browser, rig)
            finally:
                browser.close()
    finally:
        rig.close()
    print("router UI: ok")


if __name__ == "__main__":
    main()
