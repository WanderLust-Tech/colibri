"""Browser check of installing and removing models in the dashboard, against `coli router`'s router with a fake hub.

Run `npm run build`, then `python3 tests/install_ui.py` (requires Playwright and a Chromium: set COLI_CHROMIUM to a browser
executable if Playwright's own is not installed). The router (c/router.py) serves web/dist, has a real Installer, and the
"internet" is a local stand-in for the Hugging Face hub (c/tests/fake_hub.py) holding one small picture model.

What it holds the page to:
- Settings -> Model lists what can be installed below the models that are here, with the licence and the size, and an
  Install button that asks (size and licence) before anything is fetched; Cancel in that question fetches nothing;
- a download shows a progress bar and a Stop button; Stop keeps what was fetched and says so; Install again finishes it;
- when it is done the model is in the list above and no longer offered below, and the page says it is installed;
- Remove asks first, deletes the files, says how much was freed, and the model is offered for install again;
- a model that is loaded cannot be removed (the button is off); once unloaded it can.
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT.parent / "c"
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tests"))

import model_installer  # noqa: E402
import router  # noqa: E402
import setup_catalog  # noqa: E402
import setup_download  # noqa: E402
from acestep_fixture import STUB as MUSIC_STUB, make_fake_checkpoint as make_music, stub_env as music_env  # noqa: E402
from fake_hub import FakeHub  # noqa: E402
from qwen_image_fixture import STUB as IMAGE_STUB, make_fake_pipeline as make_image, stub_env as image_env  # noqa: E402

REPO = "someone/tiny-image"


class Rig:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.models = root / "models"
        make_music(self.models / "ace-step")
        make_image(root / "repo")
        self.hub = FakeHub({REPO: str(root / "repo")})
        self.hub.__enter__()
        os.environ["HF_ENDPOINT"] = self.hub.base
        self.slow = True
        real = setup_download.download_repo
        rig = self

        def paced(repo, revision, model_dir, files, **kw):
            inner = kw["progress"]

            def watch(event):
                if rig.slow:
                    time.sleep(0.8)
                inner(event)
            return real(repo, revision, model_dir, files, **{**kw, "progress": watch})

        self.paced = paced
        self.real_download = real
        setup_download.download_repo = paced
        catalog = router.Catalog(roots=[str(self.models)], engine_ready=lambda family: True)

        def worker_env(entry):
            if entry["modality"] == "image":
                return {"COLI_ENGINE": str(IMAGE_STUB), "COLI_IMAGE_DIR": str(root / "images"),
                        **{k: v for k, v in image_env().items() if k.startswith("QWENIMAGE")}}
            return {"COLI_ENGINE": str(MUSIC_STUB), "COLI_MUSIC_DIR": str(root / "music"),
                    **{k: v for k, v in music_env().items() if k.startswith("ACESTEP")}}

        self.pool = router.Pool(catalog, worker_env=worker_env, grace=8.0, state=router.State())
        fetchable = setup_catalog.CatalogModel(
            id="tiny-image", family="qwen_image", name="Tiny image", repo=REPO, disk_gb=0.001, ram_min_gb=0.1,
            ram_good_gb=0.2, dense_gb=0.01, rank=1, size_class="small", summary="fixture", modality="image",
            license_note="Apache-2.0")

        def installed_now():
            out = []
            for found in catalog.entries(refresh=True):
                worker = self.pool.get(found["key"])
                found["state"] = worker.state if worker is not None else "idle"
                out.append(found)
            return out

        installer = model_installer.Installer(roots=lambda: [str(self.models)], installed=installed_now,
                                              refresh=catalog.invalidate, catalog=lambda: (fetchable,),
                                              memory=lambda: 64 * 10 ** 9)
        self.server = router.RouterServer(("127.0.0.1", 0), self.pool, catalog, router.Gate(), None,
                                          ("http://127.0.0.1:1",), installer=installer)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"

    def close(self):
        setup_download.download_repo = self.real_download
        self.server.shutdown()
        self.server.server_close()
        self.pool.close()
        self.hub.__exit__()
        self.tmp.cleanup()


def open_tab(page, name):
    page.get_by_role("button", name="Settings", exact=True).click()
    page.locator(".settings-tabs").get_by_role("button", name=name, exact=True).click()


def run(browser, rig):
    context = browser.new_context(locale="en-US", viewport={"width": 1100, "height": 900})
    page = context.new_page()
    page.set_default_timeout(20000)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: errors.append("policy: " + message.text) if "Content Security Policy" in message.text else None)
    installs = []
    page.on("request", lambda request: installs.append(request.url) if request.url.endswith("/v1/models/install") else None)
    page.goto(rig.url, wait_until="networkidle")
    page.wait_for_timeout(800)
    open_tab(page, "Connection")
    page.get_by_role("button", name="Probe server", exact=True).click()
    expect(page.locator(".settings-card .connection-state")).to_have_text("Engine reachable")

    open_tab(page, "Model")
    here = page.locator(".model-switcher > .model-list > .model-row")
    expect(here).to_have_count(1)                                          # ACE-Step, from the folder
    expect(page.locator(".model-install .section-title")).to_contain_text("Install a model")
    offer = page.locator(".install-row").filter(has_text="Tiny image")
    expect(offer).to_have_count(1)
    expect(offer).to_contain_text("Licence: Apache-2.0")
    expect(offer).to_contain_text("pictures")

    # the question comes first, and "Cancel" fetches nothing
    offer.get_by_role("button", name="Install Tiny image", exact=True).click()
    expect(offer).to_contain_text("accept its licence?")
    offer.get_by_role("button", name="Cancel", exact=True).click()
    expect(offer.get_by_role("button", name="Install Tiny image", exact=True)).to_be_visible()
    assert installs == [], installs
    assert not (rig.models / "tiny-image").exists(), "something was fetched without an answer"

    # a download: progress, then Stop keeps the part
    offer.get_by_role("button", name="Install Tiny image", exact=True).click()
    offer.get_by_role("button", name="Download", exact=True).click()
    expect(offer.get_by_role("progressbar")).to_be_visible(timeout=15000)
    expect(offer.get_by_role("button", name="Stop download", exact=True)).to_be_visible()
    assert len(installs) == 1, installs
    offer.get_by_role("button", name="Stop download", exact=True).click()
    expect(offer).to_contain_text("Stopped. What was downloaded is kept", timeout=20000)
    expect(offer.get_by_role("button", name="Install Tiny image", exact=True)).to_be_visible()

    # asked again, it finishes (no more pacing), and the model is here
    rig.slow = False
    offer.get_by_role("button", name="Install Tiny image", exact=True).click()
    offer.get_by_role("button", name="Download", exact=True).click()
    expect(here).to_have_count(2, timeout=30000)
    expect(page.locator(".model-result")).to_contain_text("Tiny image is installed")
    expect(page.locator(".install-row")).to_have_count(0)
    expect(page.locator(".model-install")).to_contain_text("already here")
    assert (rig.models / "tiny-image" / "model_index.json").exists()

    # remove: asks, deletes, says what was freed, and offers it again
    tiny = here.filter(has_text="Qwen-Image")
    tiny.get_by_role("button", name="Remove Qwen-Image-2.1", exact=True).click()
    expect(tiny).to_contain_text("deletes its files")
    tiny.get_by_role("button", name="Remove files", exact=True).click()
    expect(page.locator(".model-result")).to_contain_text("removed", timeout=20000)
    expect(here).to_have_count(1)
    assert not (rig.models / "tiny-image").exists(), "the folder is still there"
    expect(page.locator(".install-row").filter(has_text="Tiny image")).to_have_count(1)

    # a loaded model cannot be removed; unloaded it can
    music = here.filter(has_text="ACE-Step")
    music.get_by_role("button", name="Load", exact=True).click()
    expect(music.locator(".model-current")).to_have_text("Loaded", timeout=30000)
    expect(music.get_by_role("button", name="Remove ACE-Step 1.5", exact=True)).to_be_disabled()
    music.get_by_role("button", name="Unload", exact=True).click()
    expect(music.get_by_role("button", name="Load", exact=True)).to_be_visible(timeout=30000)
    expect(music.get_by_role("button", name="Remove ACE-Step 1.5", exact=True)).to_be_enabled()
    assert not errors, errors
    context.close()


def main():
    rig = Rig()
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
    print("install UI: ok")


if __name__ == "__main__":
    main()
