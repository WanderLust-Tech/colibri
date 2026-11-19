import { describe, expect, it } from "vitest"

import type { CatalogRow, InstallJob } from "./api"
import { anyInstalling, canInstall, etaText, formatBytes, installable, installPercent, progressLine } from "./installs"

const job = (extra: Partial<InstallJob> = {}): InstallJob => ({
  id: "z-image-turbo", name: "Z-Image-Turbo", state: "downloading", done: 13_500_000_000, total: 32_900_000_000,
  rate: 41_000_000, eta_seconds: 723, file: "x", error: null, ...extra,
})

const row = (extra: Partial<CatalogRow> = {}): CatalogRow => ({
  id: "z-image-turbo", name: "Z-Image-Turbo", kind: "image", download_gb: 32.9, ram_min_gb: 10, ram_good_gb: 14, summary: "",
  license_note: "Apache-2.0", status: "good", fits: true, reason: "fits", installed: null, unfinished: false, job: null, ...extra,
})

describe("install numbers", () => {
  it("formats sizes in decimal units, as the server and the downloads do", () => {
    expect(formatBytes(0)).toBe("0 B")
    expect(formatBytes(512)).toBe("512 B")
    expect(formatBytes(3_000_000)).toBe("3.0 MB")
    expect(formatBytes(32_900_000_000)).toBe("32.9 GB")
    expect(formatBytes(419_300_000_000)).toBe("419 GB")
  })

  it("keeps the percentage inside 0..100 even for a total that is not known yet", () => {
    expect(installPercent({ done: 5, total: 0 })).toBe(0)
    expect(installPercent({ done: 50, total: 100 })).toBe(50)
    expect(installPercent({ done: 150, total: 100 })).toBe(100)
  })

  it("writes the time left the way a person says it", () => {
    expect(etaText(null)).toBe("")
    expect(etaText(45)).toBe("45s")
    expect(etaText(723)).toBe("12m 03s")
    expect(etaText(7500)).toBe("2h 05m")
  })

  it("puts size, speed and time left on one line", () => {
    expect(progressLine(job())).toBe("13.5 GB / 32.9 GB · 41.0 MB/s · 12m 03s")
    expect(progressLine(job({ rate: null, eta_seconds: null }))).toBe("13.5 GB / 32.9 GB")
  })
})

describe("what to offer", () => {
  it("offers what is not here, and not what is", () => {
    const here = row({ id: "a", installed: { key: "k", id: "a", state: "idle" } })
    expect(installable([here, row({ id: "b" }), row({ id: "c", unfinished: true })]).map((r) => r.id)).toEqual(["b", "c"])
  })

  it("lets an install start when it fits, or only RAM is short, and nothing else runs", () => {
    expect(canInstall(row(), false)).toBe(true)
    expect(canInstall(row({ status: "tight" }), false)).toBe(true)
    expect(canInstall(row({ status: "needs-ram", fits: false }), false)).toBe(true)
    expect(canInstall(row({ status: "needs-disk", fits: false }), false)).toBe(false)
    expect(canInstall(row({ status: "unsupported", fits: false }), false)).toBe(false)
    expect(canInstall(row(), true)).toBe(false)
  })

  it("knows when to keep polling", () => {
    expect(anyInstalling([row()])).toBe(false)
    expect(anyInstalling([row({ job: job({ state: "done" }) })])).toBe(false)
    expect(anyInstalling([row({ job: job({ state: "failed" }) })])).toBe(false)
    expect(anyInstalling([row(), row({ job: job({ state: "downloading" }) })])).toBe(true)
    expect(anyInstalling([row({ job: job({ state: "finishing" }) })])).toBe(true)
  })
})
