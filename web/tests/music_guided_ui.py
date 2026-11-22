"""Browser check of the music view's planner and guided-DiT controls, against the real gateway with a stub engine behind it.

Run `npm run build`, then `python3 tests/music_guided_ui.py` (Playwright and a Chromium: COLI_CHROMIUM names a browser executable when
Playwright's own is not installed). The stub engine (c/tools/acestep_stub.py) is told to be a checkpoint with the 5 Hz planner and the
base DiT (ACESTEP_STUB_PLANNER, ACESTEP_STUB_GUIDED), so READY announces them and /v1/models carries them to the page.

What it holds the page to: the planner chip, the auto-length chip and the DiT choice appear only for a checkpoint that has them (a plain
one shows none of them); with the planner on and the length left to it the request has `thinking` and no `duration`, and the song
says what the planner chose; choosing the base DiT sends `dit`, its guidance and its own step count (32, not the turbo's 8); extract,
lego and complete are offered, extract needs its clip and sends its track, lego sends its span, and complete is made of the tracks
that are switched on.
"""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import music_ui  # noqa: E402
from music_ui import MODEL_ID, composer, open_page, send, write_wav  # noqa: E402

import music_engine  # noqa: E402
from acestep_fixture import STUB, make_fake_checkpoint, stub_env  # noqa: E402
from openai_server import APIServer  # noqa: E402


class StubGateway(music_ui.Gateway):
    """music_ui's gateway with the stub told what the checkpoint has."""

    def __init__(self, **stub):
        self.tmp = tempfile.TemporaryDirectory()
        model = make_fake_checkpoint(Path(self.tmp.name) / "ace-step")
        self.engine = music_engine.MusicEngine(STUB, model, env=stub_env(0.0, **stub), on_log=lambda _m: None)
        self.calls = []
        generate = self.engine.generate

        def spy(*args, **kwargs):
            self.calls.append({"caption": args[0] if args else kwargs.get("caption"), **kwargs})
            return generate(*args, **kwargs)
        self.engine.generate = spy
        self.server = APIServer(("127.0.0.1", 0), self.engine, MODEL_ID, "", 16, max_queue=8, queue_timeout=30)
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,), daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/"


def run_plain(browser, gateway):
    context, page, _posts, errors = open_page(browser, gateway.url)
    form = composer(page)
    expect(form.get_by_label("Length of the song, in seconds")).to_be_visible()
    expect(form.get_by_role("button", name="Plan")).to_have_count(0)
    expect(form.get_by_label("DiT", exact=True)).to_have_count(0)
    expect(form.get_by_label("Guidance", exact=True)).to_have_count(0)
    for name in ("Extract", "Add a track", "Complete"):
        expect(form.get_by_role("button", name=name)).to_have_count(0)
    assert not errors, errors
    context.close()


def run_planner(browser, gateway):
    context, page, posts, errors = open_page(browser, gateway.url)
    form = composer(page)
    expect(form.get_by_role("button", name="Plan")).to_be_visible()
    expect(form.get_by_role("button", name="Auto length")).to_have_count(0)          # only once the planner is on
    form.get_by_role("button", name="Plan").click()
    auto = form.get_by_role("button", name="Auto length")
    expect(auto).to_have_attribute("aria-pressed", "true")
    expect(form.get_by_label("Length of the song, in seconds")).to_have_count(0)      # the planner chooses it
    form.get_by_label("Seed", exact=True).fill("5")
    form.get_by_label("Sampler steps").fill("2")
    send(page, "calm piano")
    expect(page.locator("audio.music-player")).to_have_count(1, timeout=30000)
    request = posts[-1]
    assert request["thinking"] is True and "duration" not in request, request
    summary = page.locator("details.music-meta summary")
    expect(summary).to_contain_text("What the planner chose")
    expect(summary).to_contain_text("100 bpm")
    expect(summary).to_contain_text("C major")
    summary.click()
    expect(page.locator("details.music-meta")).to_contain_text("A planned take on: calm piano")
    expect(page.locator(".image-meta").first).to_contain_text("0:12")                 # the length the planner picked (the stub's 12 s)

    # a length of one's own goes with the planner too
    auto.click()
    expect(auto).to_have_attribute("aria-pressed", "false")
    form.get_by_label("Length of the song, in seconds").fill("4")
    send(page, "calm piano")
    expect(page.locator("audio.music-player")).to_have_count(2, timeout=30000)
    request = posts[-1]
    assert request["thinking"] is True and request["duration"] == 4, request
    # and it is off again: a plain song has neither
    form.get_by_role("button", name="Plan").click()
    send(page, "calm piano")
    expect(page.locator("audio.music-player")).to_have_count(3, timeout=30000)
    assert "thinking" not in posts[-1] and posts[-1]["duration"] == 4, posts[-1]
    assert not errors, errors
    context.close()


def run_guided(browser, gateway, work):
    context, page, posts, errors = open_page(browser, gateway.url)
    form = composer(page)
    dit = form.get_by_label("DiT", exact=True)
    expect(dit).to_be_visible()
    expect(form.get_by_label("Guidance", exact=True)).to_have_count(0)                           # the turbo DiT has guidance built in
    group = form.get_by_role("group", name="What to make")
    for name in ("Compose", "Cover", "Repaint", "Extract", "Add a track", "Complete"):
        expect(group.get_by_role("button", name=name)).to_be_visible()

    # the base DiT: its own steps (32), guidance 7, sent as asked
    dit.select_option("base")
    expect(form.get_by_label("Guidance", exact=True)).to_have_value("7")
    expect(form.get_by_label("Sampler steps")).to_have_value("32")
    form.get_by_label("Sampler steps").fill("3")
    form.get_by_label("Length of the song, in seconds").fill("2")
    form.get_by_label("Seed", exact=True).fill("9")
    form.get_by_label("Guidance", exact=True).fill("5")
    send(page, "heavy riffs")
    expect(page.locator("audio.music-player")).to_have_count(1, timeout=30000)
    request = posts[-1]
    assert request["dit"] == "base" and request["guidance_scale"] == 5 and request["steps"] == 3 and request["duration"] == 2, request
    expect(page.locator(".image-meta").first).to_contain_text("base DiT")
    dit.select_option("")
    expect(form.get_by_label("Sampler steps")).to_have_value("8")                    # back on the turbo DiT's range
    expect(form.get_by_label("Guidance", exact=True)).to_have_count(0)

    # extract: needs its clip, takes a track
    clip = write_wav(Path(work) / "band.wav", 3)
    group.get_by_role("button", name="Extract").click()
    expect(page.get_by_role("button", name="Make music")).to_be_disabled()           # no clip yet
    expect(form.get_by_label("DiT", exact=True)).to_have_count(0)                                # the task has its own
    page.locator(".music-clip[data-role=source] input[type=file]").set_input_files(str(clip))
    expect(page.locator(".music-clip[data-role=source] .music-clip-name")).to_have_text("band.wav")
    form.get_by_label("Track", exact=True).select_option("drums")
    form.get_by_label("Sampler steps").fill("2")
    before = len(gateway.calls)
    send(page, "")
    expect(page.locator("audio.music-player")).to_have_count(2, timeout=30000)
    call = gateway.calls[before]
    assert call["task"] == "extract" and call["track"] == "drums" and call["src_audio"] == clip.read_bytes(), call
    expect(page.locator(".music-request").last).to_contain_text("drums out of band.wav")

    # lego: the span and the track
    group.get_by_role("button", name="Add a track").click()
    form.get_by_label("Track", exact=True).select_option("bass")
    form.get_by_label("Repaint from, in seconds").fill("1")
    form.get_by_label("Repaint to, in seconds").fill("2")
    before = len(gateway.calls)
    send(page, "")
    expect(page.locator("audio.music-player")).to_have_count(3, timeout=30000)
    call = gateway.calls[before]
    assert call["task"] == "lego" and call["track"] == "bass" and call["repaint_start"] == 1 and call["repaint_end"] == 2, call
    expect(page.locator(".music-request").last).to_contain_text("bass added to band.wav")

    # complete: made of the tracks switched on, with or without a clip
    group.get_by_role("button", name="Complete").click()
    expect(page.get_by_role("button", name="Make music")).to_be_disabled()           # no track chosen
    tracks = form.get_by_role("group", name="Tracks to complete the clip with")
    tracks.get_by_role("button", name="drums").click()
    tracks.get_by_role("button", name="bass").click()
    expect(page.get_by_role("button", name="Make music")).to_be_enabled()
    before = len(gateway.calls)
    send(page, "")
    expect(page.locator("audio.music-player")).to_have_count(4, timeout=30000)
    call = gateway.calls[before]
    assert call["task"] == "complete" and call["tracks"] == "drums,bass", call
    assert not errors, errors
    context.close()


def main():
    plain, planner, guided = StubGateway(), StubGateway(ACESTEP_STUB_PLANNER=1), StubGateway(ACESTEP_STUB_PLANNER=1, ACESTEP_STUB_GUIDED=1)
    launch = {"executable_path": os.environ["COLI_CHROMIUM"]} if os.environ.get("COLI_CHROMIUM") else {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch)
            try:
                run_plain(browser, plain)
                run_planner(browser, planner)
                with tempfile.TemporaryDirectory() as work:
                    run_guided(browser, guided, work)
            finally:
                browser.close()
    finally:
        for gateway in (plain, planner, guided):
            gateway.close()
    print("music guided UI: ok")


if __name__ == "__main__":
    main()
