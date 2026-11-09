"""Browser check of the Presets and Requests tabs against `coli router`'s router, with real gateways (stub engines) behind it.

Run `npm run build`, then `python3 tests/presets_ui.py` (requires Playwright and a Chromium: set COLI_CHROMIUM to a
browser executable if Playwright's own is not installed). The router (c/router.py) serves web/dist, keeps its presets in a
temporary file and writes a request log; the models are the picture and music stubs of router_ui.py.

What it holds the page to: a router with no presets says so; a preset made in the form is in the file, in the model list
(under Presets, with a note saying what it is a preset of) and usable at once; choosing it starts the picture controls from its
settings; Use takes the conversation to it; Edit changes it in the file; a name that is a model's own and a setting the router
refuses are reported without touching the file; Delete asks first and removes it; the Requests tab lists what the router
answered, with the preset's name; a router with no log says how to turn it on.
"""
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT.parent / "c"
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import presets  # noqa: E402
import request_log  # noqa: E402
from router_ui import Router, open_tab  # noqa: E402


def post(rig, path, body):
    request = urllib.request.Request(rig.url.rstrip("/") + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def run(browser, rig, file, log_path):
    context = browser.new_context(locale="en-US", viewport={"width": 1100, "height": 900})
    page = context.new_page()
    page.set_default_timeout(20000)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(rig.url, wait_until="networkidle")
    page.wait_for_timeout(800)
    open_tab(page, "Connection")
    page.get_by_role("button", name="Probe server", exact=True).click()
    expect(page.locator(".settings-card .connection-state")).to_have_text("Engine reachable")

    # ---- an empty router
    open_tab(page, "Presets")
    expect(page.locator(".preset-panel")).to_contain_text("There are no presets yet")
    assert not os.path.exists(file), "looking at presets must not create the file"

    # ---- make one in the form
    page.get_by_role("button", name="New preset", exact=True).click()
    expect(page.locator("#preset-system")).to_have_count(0)                    # a picture model has no system prompt
    page.locator("#preset-model").select_option(label="Qwen-Image-2.1")
    page.locator("#preset-name").fill("Photo-Wide")
    save = page.get_by_role("button", name="Save preset", exact=True)
    expect(save).to_be_enabled()
    page.locator("#preset-name").fill("no spaces")
    expect(save).to_be_disabled()
    expect(page.locator(".preset-form")).to_contain_text("Use lower-case letters")
    page.locator("#preset-name").fill("photo-wide")
    page.locator("#preset-description").fill("wide pictures")
    page.get_by_role("button", name="Add a setting").click()
    page.get_by_label("Setting", exact=True).last.fill("size")
    page.get_by_label("Value", exact=True).last.fill("1024x768")
    page.get_by_role("button", name="Add a setting").click()
    page.get_by_label("Setting", exact=True).last.fill("steps")
    page.get_by_label("Value", exact=True).last.fill("12")
    page.locator("#preset-keep-alive").fill("2m")
    save.click()
    expect(page.locator(".model-result")).to_contain_text("photo-wide was created")
    row = page.locator('[data-preset="photo-wide"]')
    expect(row).to_contain_text("runs qwen-image-2.1-colibri")
    expect(row).to_contain_text("size 1024x768, steps 12")
    expect(row).to_contain_text("wide pictures")
    stored = json.load(open(file, encoding="utf-8"))["presets"]
    assert stored == {"photo-wide": {"model": "qwen-image-2.1-colibri", "description": "wide pictures", "keep_alive": "2m",
                                     "defaults": {"size": "1024x768", "steps": 12}}}, stored       # steps is a number

    # ---- it is a model of its own, and choosing it starts the controls from it
    open_tab(page, "Model")
    expect(page.locator("#model-select optgroup[label='Presets'] option")).to_have_text(["photo-wide"])
    page.locator("#model-select").select_option("photo-wide")
    expect(page.locator(".preset-note")).to_contain_text("A preset of qwen-image-2.1-colibri. wide pictures")

    # ---- Use: the conversation is on the preset, with its size and steps in the picture controls
    open_tab(page, "Presets")
    row = page.locator('[data-preset="photo-wide"]')
    row.get_by_role("button", name="Use", exact=True).click()
    expect(page.locator(".model-button")).to_contain_text("photo-wide")
    expect(page.get_by_label("Denoising steps")).to_have_value("12")
    size = page.get_by_label("Image size")
    expect(size).to_have_count(1)
    if size.input_value() == "custom":                                          # 1024x768 is not one of the listed sizes
        expect(page.get_by_label("Width", exact=True)).to_have_value("1024")
        expect(page.get_by_label("Height", exact=True)).to_have_value("768")
    else:
        assert size.input_value() == "1024x768", size.input_value()

    # ---- a request to it is answered and logged under its name
    answer = post(rig, "/v1/images/generations", {"model": "photo-wide", "prompt": "a cat", "n": 1})
    assert answer.get("data"), answer

    # ---- edit
    open_tab(page, "Presets")
    page.locator('[data-preset="photo-wide"]').get_by_role("button", name="Edit", exact=True).click()
    expect(page.locator("#preset-name")).to_be_disabled()
    expect(page.locator("#preset-description")).to_have_value("wide pictures")
    expect(page.get_by_label("Setting", exact=True)).to_have_count(2)
    page.locator("#preset-description").fill("very wide")
    page.get_by_label("Value", exact=True).last.fill("20")
    page.get_by_role("button", name="Save preset", exact=True).click()
    expect(page.locator(".model-result")).to_contain_text("photo-wide was saved")
    stored = json.load(open(file, encoding="utf-8"))["presets"]["photo-wide"]
    assert (stored["description"], stored["defaults"]) == ("very wide", {"size": "1024x768", "steps": 20}), stored

    # ---- what the router refuses is shown, and the file is not changed
    before = open(file, encoding="utf-8").read()
    page.get_by_role("button", name="New preset", exact=True).click()
    page.locator("#preset-name").fill("qwen-image-2.1")                          # a model's own name
    page.get_by_role("button", name="Save preset", exact=True).click()
    expect(page.locator(".model-result.error")).to_contain_text("name of an installed model")
    page.get_by_role("button", name="Add a setting").click()
    page.get_by_label("Setting", exact=True).last.fill("prompt")                             # what a request says, not a default
    page.get_by_label("Value", exact=True).last.fill("a dog")
    page.locator("#preset-name").fill("bad-one")
    page.get_by_role("button", name="Save preset", exact=True).click()
    expect(page.locator(".model-result.error")).to_contain_text("cannot be a default")
    assert open(file, encoding="utf-8").read() == before
    page.get_by_role("button", name="Cancel", exact=True).click()

    # ---- delete asks first
    row = page.locator('[data-preset="photo-wide"]')
    row.get_by_role("button", name="Delete", exact=True).click()
    expect(row).to_contain_text("Delete photo-wide?")
    row.get_by_role("button", name="Cancel", exact=True).click()
    expect(row.get_by_role("button", name="Use", exact=True)).to_be_visible()
    row.get_by_role("button", name="Delete", exact=True).click()
    expect(row).to_contain_text("Delete photo-wide?")
    row.get_by_role("button", name="Delete", exact=True).click()                  # the confirmation's own button
    expect(page.locator('[data-preset="photo-wide"]')).to_have_count(0)
    expect(page.locator(".model-result")).to_contain_text("photo-wide was deleted")
    assert json.load(open(file, encoding="utf-8"))["presets"] == {}

    # ---- requests
    open_tab(page, "Requests")
    expect(page.locator(".requests-panel")).to_contain_text(log_path.name)
    table = page.locator(".requests-table tbody tr")
    expect(table).to_have_count(1)
    expect(table.first).to_contain_text("images/generations")
    expect(table.first).to_contain_text("photo-wide")                           # the name it was asked for under
    expect(table.first).to_contain_text("200")
    assert not errors, errors
    context.close()


def run_without_log(browser, rig):
    context = browser.new_context(locale="en-US", viewport={"width": 1100, "height": 900})
    page = context.new_page()
    page.set_default_timeout(20000)
    page.goto(rig.url, wait_until="networkidle")
    open_tab(page, "Connection")
    page.get_by_role("button", name="Probe server", exact=True).click()
    expect(page.locator(".settings-card .connection-state")).to_have_text("Engine reachable")
    open_tab(page, "Requests")
    expect(page.locator(".requests-panel")).to_contain_text("keeps no request log")
    expect(page.locator(".requests-table")).to_have_count(0)
    context.close()


def main():
    rig = Router()
    tmp = tempfile.TemporaryDirectory()
    file = os.path.join(tmp.name, "presets.json")
    log_path = Path(tmp.name) / "requests.log"
    rig.server.presets = presets.Presets(file, log=lambda message: None)
    launch = {"executable_path": os.environ["COLI_CHROMIUM"]} if os.environ.get("COLI_CHROMIUM") else {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch)
            try:
                run_without_log(browser, rig)
                rig.server.request_log = request_log.RequestLog(str(log_path))
                run(browser, rig, file, log_path)
            finally:
                browser.close()
    finally:
        rig.close()
        tmp.cleanup()
    print("presets UI: ok")


if __name__ == "__main__":
    main()
