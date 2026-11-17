"""Browser check of what the Model page shows about the models on disk, against a real `coli router` over a folder of stub models.

Run `npm run build`, then `python3 tests/catalogue_ui.py` (requires Playwright and a Chromium: set CHROME_PATH or COLI_CHROMIUM to a
browser executable if Playwright's own is not installed). No model is loaded: the page only reads the router's list.

What it holds the page to:
- each model row says what its folder says about itself (architecture, weights, the context limit it was made for) before the model
  has ever run, read from its small files;
- a download that is not finished is listed with how far it got, and cannot be loaded;
- a model whose drive is not connected stays on the list, greyed, with the reason, and cannot be loaded; when the folder is back it is
  an ordinary row again.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT.parent / "c"
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tests"))

import jobs  # noqa: E402
import router  # noqa: E402
import setup_download  # noqa: E402
from acestep_fixture import make_fake_checkpoint as make_music  # noqa: E402
from qwen_image_fixture import make_fake_pipeline as make_image  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


class Router:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = Path(make_image(self.root / "models" / "qwen-image"))
        make_music(self.root / "models" / "ace-step")
        partial = self.root / "models" / "qwen-image-partial"
        shutil.copytree(self.image, partial)
        (partial / setup_download.MANIFEST).write_text(json.dumps({"complete": False, "files": [
            {"path": "model_index.json", "size": (partial / "model_index.json").stat().st_size},
            {"path": "transformer/never-arrived.safetensors", "size": 1000}]}))
        catalog = router.Catalog(roots=[str(self.root / "models")], engine_ready=lambda family: True, ttl=0,
                                 known_path=str(self.root / "state" / "known-models.json"))
        self.pool = router.Pool(catalog, state=router.State(str(self.root / "state.json")))
        self.jobs = jobs.JobManager(jobs.JobStore(str(self.root / "jobs.jsonl")))
        self.server = router.RouterServer(("127.0.0.1", 0), self.pool, catalog, router.Gate(), None, ("http://127.0.0.1:1",), jobs=self.jobs)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"

    def close(self):
        self.jobs.close()
        self.server.shutdown()
        self.server.server_close()
        self.pool.close()
        self.tmp.cleanup()


def open_models(page):
    page.get_by_role("button", name="Settings", exact=True).click()
    page.locator(".settings-tabs").get_by_role("button", name="Model", exact=True).click()


def main(browser, rig):
    context = browser.new_context(locale="en-US", viewport={"width": 1100, "height": 900})
    page = context.new_page()
    page.set_default_timeout(20000)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(rig.url, wait_until="networkidle")
    open_models(page)
    rows = page.locator(".model-row")
    expect(rows).to_have_count(3)
    qwen = rows.filter(has=page.locator("strong", has_text="Qwen-Image")).filter(has_not_text="download incomplete")
    expect(qwen.locator(".model-details")).to_contain_text("QwenImage21Pipeline")
    expect(qwen.get_by_role("button", name="Load", exact=True)).to_be_enabled()
    check("a model's row says its architecture and weights, read from its folder before it ever ran", True)

    partial = rows.filter(has_text="download incomplete")
    expect(partial).to_have_count(1)
    expect(partial).to_contain_text("1 of 2 files here")
    expect(partial.get_by_role("button", name="Load", exact=True)).to_be_disabled()
    check("a download that is not finished is listed with how far it got and cannot be loaded", True)

    # the folder goes (as if its drive were unplugged): the row stays, greyed, with the reason
    original = router.why_missing
    router.why_missing = lambda path, *a, **k: ("the drive E: is not connected", False)
    try:
        shutil.move(str(rig.image), str(rig.root / "unplugged"))
        page.reload(wait_until="networkidle")
        open_models(page)
        gone = page.locator(".model-row").filter(has_text="the drive E: is not connected")
        expect(gone).to_have_count(1, timeout=15000)
        expect(gone.get_by_role("button", name="Load", exact=True)).to_be_disabled()
        check("a model whose drive is not connected stays listed with the reason and cannot be loaded", True)
    finally:
        router.why_missing = original
    shutil.move(str(rig.root / "unplugged"), str(rig.image))
    page.reload(wait_until="networkidle")
    open_models(page)
    back = page.locator(".model-row").filter(has=page.locator("strong", has_text="Qwen-Image")).filter(has_not_text="download incomplete")
    expect(back.locator(".model-details")).to_contain_text("QwenImage21Pipeline", timeout=15000)
    expect(back.get_by_role("button", name="Load", exact=True)).to_be_enabled()
    expect(page.locator(".model-row").filter(has_text="not connected")).to_have_count(0)
    check("with the folder back it is an ordinary row again", True)
    check("the page raised no errors", errors == [], str(errors))
    context.close()


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"],
                                executable_path=os.environ.get("CHROME_PATH") or os.environ.get("COLI_CHROMIUM") or None)
    rig = Router()
    try:
        main(browser, rig)
    finally:
        rig.close()
        browser.close()

failed = [name for name, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
