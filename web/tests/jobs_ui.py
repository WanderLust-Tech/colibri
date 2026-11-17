"""Browser check of background jobs and the Library, against a real `coli router` with real gateways (stub engines) behind it.

Run `npm run build`, then `python3 tests/jobs_ui.py` (requires Playwright and a Chromium: set CHROME_PATH or COLI_CHROMIUM to a
browser executable if Playwright's own is not installed). The router (c/router.py) serves web/dist and starts a real `coli serve`
for each model when it is asked for; the engines are the stubs of the picture and music tests, slowed down so a render takes seconds.

What it holds the page to:
- with the cloud switch on, a picture goes out with `Prefer: respond-async`, the router answers 202, the card says it keeps going,
  and the finished picture is the kept file; with it off the request is the ordinary one and no job exists;
- closing the page in the middle of a render does not stop it: the router finishes it and the file is kept;
- Settings > Library lists the kept pictures and songs with their prompt, size and seed, "Make again" puts a card's model, prompt and
  settings back in the composer, and a filter shows only songs;
- the Jobs part shows a running job (with its progress) and a waiting one (with its place in line), and Cancel stops each: the
  waiting one never runs, the running one is interrupted and the engine stops computing;
- a song is a job too.
"""
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT.parent / "c"
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tests"))

import jobs  # noqa: E402
import router  # noqa: E402
from acestep_fixture import STUB as MUSIC_STUB, make_fake_checkpoint as make_music, stub_env as music_env  # noqa: E402
from qwen_image_fixture import STUB as IMAGE_STUB, make_fake_pipeline as make_image, stub_env as image_env  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


class Router:
    def __init__(self, step_delay=0.3):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_image(self.root / "models" / "qwen-image")
        make_music(self.root / "models" / "ace-step")
        self.images, self.music = self.root / "images", self.root / "music"
        # the router reads the kept folders from its own environment, as the workers it starts do (the same machine, the same variables)
        self.saved_env = {name: os.environ.get(name) for name in ("COLI_IMAGE_DIR", "COLI_MUSIC_DIR")}
        os.environ["COLI_IMAGE_DIR"], os.environ["COLI_MUSIC_DIR"] = str(self.images), str(self.music)
        catalog = router.Catalog(roots=[str(self.root / "models")], engine_ready=lambda family: True)

        def worker_env(entry):
            if entry["modality"] == "image":
                return {"COLI_ENGINE": str(IMAGE_STUB), "COLI_IMAGE_DIR": str(self.images),
                        **{k: v for k, v in image_env(step_delay).items() if k.startswith("QWENIMAGE")}}
            return {"COLI_ENGINE": str(MUSIC_STUB), "COLI_MUSIC_DIR": str(self.music),
                    **{k: v for k, v in music_env().items() if k.startswith("ACESTEP")}}

        self.catalog = catalog
        self.pool = router.Pool(catalog, worker_env=worker_env, grace=8.0, state=router.State(str(self.root / "state.json")))
        self.jobs = jobs.JobManager(jobs.JobStore(str(self.root / "jobs.jsonl")))
        self.server = router.RouterServer(("127.0.0.1", 0), self.pool, catalog, router.Gate(), None, ("http://127.0.0.1:1",), jobs=self.jobs)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/"
        entries = {e["modality"]: e["id"] for e in catalog.entries()}
        self.image_id, self.music_id = entries["image"], entries["audio"] if "audio" in entries else entries["music"]

    def api(self, method, path, body=None, headers=None):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method=method, headers={"Content-Type": "application/json", **(headers or {})},
                                         data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())

    def job_list(self):
        return self.api("GET", "/v1/jobs")[1]["data"]

    def wait(self, condition, seconds=60):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            found = condition()
            if found:
                return found
            time.sleep(0.1)
        return None

    def close(self):
        self.jobs.close()
        self.server.shutdown()
        self.server.server_close()
        self.pool.close()
        for name, value in self.saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.tmp.cleanup()


def open_page(browser, rig, model, background=False, errors=None):
    context = browser.new_context(locale="en-US", viewport={"width": 1100, "height": 800})
    context.add_init_script(f"try {{ localStorage.setItem('colibri.model', '{model}'); localStorage.setItem('colibri.background', '{'1' if background else ''}') }} catch (e) {{}}")
    page = context.new_page()
    page.set_default_timeout(30000)
    problems = errors if errors is not None else []
    page.on("pageerror", lambda error: problems.append(str(error)))
    page.on("console", lambda message: problems.append("policy: " + message.text) if "Content Security Policy" in message.text else None)
    page.goto(rig.url, wait_until="networkidle")
    expect(page.locator("#draft")).to_be_visible()
    return context, page


def open_library(page):
    page.get_by_role("button", name="Settings", exact=True).click()
    page.locator(".settings-tabs").get_by_role("button", name="Library", exact=True).click()


def make_picture(page, prompt, steps, seed):
    form = page.locator("form.composer")
    form.get_by_label("Denoising steps").fill(str(steps))
    form.get_by_label("Seed", exact=True).fill(str(seed))
    form.get_by_label("Image size").select_option("512x512")
    page.locator("#draft").fill(prompt)
    page.get_by_role("button", name="Generate image").click()


def main(browser, rig):
    errors = []
    posts = []

    # ---- a picture as a background job
    context, page = open_page(browser, rig, rig.image_id, errors=errors)
    page.on("request", lambda r: posts.append((r.url, r.headers.get("prefer"))) if r.method == "POST" and "/v1/images/" in r.url else None)
    chip = page.get_by_role("button", name="Keep going if I leave")
    expect(chip).to_be_visible()
    expect(chip).to_have_attribute("aria-pressed", "false")
    chip.click()
    expect(chip).to_have_attribute("aria-pressed", "true")
    make_picture(page, "a lighthouse at dusk", 3, 5)
    expect(page.locator(".job-note")).to_contain_text("keeps going if you close the page", timeout=15000)
    expect(page.get_by_role("button", name="Download PNG")).to_be_visible(timeout=60000)
    ok = page.locator(".image-turn img").first.evaluate("img => img.naturalWidth > 0 && img.src.startsWith('blob:')")
    check("a picture made as a background job is shown from the kept file", ok)
    check("it went with Prefer: respond-async", posts and posts[-1][1] == "respond-async", str(posts))
    listed = rig.job_list()
    check("the router has it as a succeeded job with its result", len(listed) == 1 and listed[0]["status"] == "succeeded"
          and listed[0]["result"]["file"].endswith("-a-lighthouse-at-dusk-s5.png"), str(listed))
    check("the file is on the server", (rig.images / listed[0]["result"]["file"]).is_file())

    # ---- switched off: the ordinary request, no job
    chip.click()
    expect(chip).to_have_attribute("aria-pressed", "false")
    before = len(rig.job_list())
    make_picture(page, "a foreground picture", 2, 9)
    expect(page.get_by_role("button", name="Download PNG")).to_have_count(2, timeout=60000)
    check("with the switch off no job is made and the request carries no preference", len(rig.job_list()) == before and posts[-1][1] is None, str(posts[-1]))
    context.close()

    # ---- closing the page does not stop the render
    context, page = open_page(browser, rig, rig.image_id, background=True)
    make_picture(page, "a long render", 8, 11)
    started = rig.wait(lambda: [j for j in rig.job_list() if j["request"].get("prompt") == "a long render" and j["status"] == "running"])
    check("the job is running on the router", bool(started))
    context.close()                                              # the page is gone mid-render
    done = rig.wait(lambda: [j for j in rig.job_list() if j["request"].get("prompt") == "a long render" and j["status"] == "succeeded"], 90)
    check("closing the page did not stop it: the router finished the render", bool(done), str(rig.job_list()[:1]))
    check("and kept the picture", bool(done) and (rig.images / done[0]["result"]["file"]).is_file())

    # ---- and without the switch it does (the contrast this is for)
    context, page = open_page(browser, rig, rig.image_id, background=False)
    make_picture(page, "a lost render", 12, 13)
    busy = rig.wait(lambda: any(w.inflight for w in rig.pool.snapshot()), 30)
    context.close()
    free = rig.wait(lambda: not any(w.inflight for w in rig.pool.snapshot()), 20)
    time.sleep(1.5)
    check("without the switch, closing the page cancels the render and nothing is kept",
          bool(busy) and bool(free) and not any("a-lost-render" in p.name for p in rig.images.iterdir())
          and not any(j["request"].get("prompt") == "a lost render" for j in rig.job_list()))

    # ---- the Library
    context, page = open_page(browser, rig, rig.image_id, errors=errors)
    open_library(page)
    cards = page.locator(".library-card")
    expect(cards).to_have_count(3, timeout=15000)                  # background, foreground, the long render
    long_render = cards.filter(has_text="a long render")
    expect(long_render).to_have_count(1)
    expect(long_render.locator(".library-facts")).to_contain_text("512×512")
    expect(long_render.locator(".library-facts")).to_contain_text("seed 11")
    expect(long_render.locator("img")).to_be_visible(timeout=15000)
    ok = long_render.locator("img").evaluate("img => img.naturalWidth > 0")
    check("the Library shows the kept pictures with their prompt, size and seed", ok)
    expect(page.locator(".library-filter").get_by_role("button", name="Songs")).to_be_visible()
    page.locator(".library-filter").get_by_role("button", name="Songs").click()
    expect(page.locator(".library-empty, .settings-help").filter(has_text="Nothing kept yet")).to_be_visible()
    expect(cards).to_have_count(0)
    check("the Songs filter shows only songs", True)
    page.locator(".library-filter").get_by_role("button", name="All").click()
    expect(cards).to_have_count(3)
    cards.filter(has_text="a long render").get_by_role("button", name="Make again").click()
    expect(page.locator("#draft")).to_have_value("a long render")
    form = page.locator("form.composer")
    expect(form.get_by_label("Denoising steps")).to_have_value("8")
    expect(form.get_by_label("Seed", exact=True)).to_have_value("11")
    check("Make again puts the prompt, steps and seed back in the composer", True)
    context.close()

    # ---- the Jobs part: a running job, a waiting one, and Cancel for both
    _s, running = rig.api("POST", "/v1/images/generations", {"model": rig.image_id, "prompt": "first long", "size": "512x512", "steps": 40, "seed": 1},
                          {"Prefer": "respond-async"})
    _s, waiting = rig.api("POST", "/v1/images/generations", {"model": rig.image_id, "prompt": "second waits", "size": "512x512", "steps": 3, "seed": 2},
                          {"Prefer": "respond-async"})
    context, page = open_page(browser, rig, rig.image_id, errors=errors)
    open_library(page)
    rows = page.locator(".job-row")
    expect(rows).to_have_count(2, timeout=20000)
    first, second = rows.filter(has_text="first long"), rows.filter(has_text="second waits")
    expect(second).to_contain_text("number 1 in line", timeout=20000)
    expect(first.locator("[role=progressbar]")).to_be_visible(timeout=30000)
    check("the Jobs part shows the running job with a progress bar and the waiting one with its place", True)
    second.get_by_role("button", name="Cancel this job").click()
    expect(rows).to_have_count(1, timeout=15000)
    gone = rig.wait(lambda: rig.api("GET", f"/v1/jobs/{waiting['id']}")[1]["status"] == "cancelled", 10)
    check("cancelling the waiting job removes it", bool(gone))
    first.get_by_role("button", name="Cancel this job").click()
    interrupted = rig.wait(lambda: rig.api("GET", f"/v1/jobs/{running['id']}")[1]["status"] == "cancelled", 15)
    check("cancelling the running job interrupts it", bool(interrupted))
    expect(rows).to_have_count(0, timeout=15000)
    check("neither cancelled job left a file", not any("first-long" in p.name or "second-waits" in p.name for p in rig.images.iterdir()))
    free = rig.wait(lambda: not any(w.inflight for w in rig.pool.snapshot()), 10)
    check("and the engine is free again", bool(free))
    context.close()

    # ---- a song
    context, page = open_page(browser, rig, rig.music_id, background=True, errors=errors)
    form = page.locator("form.composer")
    form.get_by_label("Length of the song, in seconds").fill("2")
    form.get_by_label("Sampler steps").fill("3")
    form.get_by_label("Seed", exact=True).fill("7")
    page.locator("#draft").fill("calm piano, soft strings")
    page.get_by_role("button", name="Make music").click()
    expect(page.get_by_role("button", name="Download WAV")).to_be_visible(timeout=90000)
    src = page.locator("audio.music-player").first.get_attribute("src")
    check("a song made as a background job plays from the kept file", bool(src) and src.startswith("blob:"), str(src))
    open_library(page)
    expect(page.locator(".library-card")).to_have_count(4, timeout=15000)
    page.locator(".library-filter").get_by_role("button", name="Songs").click()
    song = page.locator(".library-card")
    expect(song).to_have_count(1)
    expect(song.locator(".library-prompt")).to_have_text("calm piano, soft strings")
    expect(song.get_by_role("button", name="Play")).to_be_visible()
    check("the Library lists the song under Songs", True)
    context.close()

    check("the pages raised no errors and no content-policy violations", errors == [], str(errors))


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
