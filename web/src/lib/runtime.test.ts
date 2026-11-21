import { describe, expect, it } from "vitest"

import type { HealthResponse } from "./api"
import { activeRequests, serverHas, settingsPages, supportsCacheSlots, supportsContinuation } from "./runtime"

const healthWithActive = (active: boolean | number): HealthResponse => ({
  status: "ok",
  scheduler: {
    active,
    queued: 0,
    max_queue: 0,
    queue_timeout_seconds: 0,
    admitted: 0,
    completed: 0,
    rejected: 0,
    timed_out: 0,
    cancelled: 0,
  },
})

describe("runtime capability normalization", () => {
  it.each([
    [true, 1],
    [false, 0],
    [3, 3],
    [0, 0],
  ] as const)("normalizes scheduler.active %s to %d", (active, expected) => {
    expect(activeRequests(healthWithActive(active))).toBe(expected)
  })

  it("treats missing scheduler metrics as idle", () => {
    expect(activeRequests({ status: "ok" })).toBe(0)
    expect(activeRequests(null)).toBe(0)
  })

  it("only enables cache slots when the health response advertises them", () => {
    expect(supportsCacheSlots({ status: "ok", kv_slots: 4 })).toBe(true)
    expect(supportsCacheSlots({ status: "ok" })).toBe(false)
    expect(supportsCacheSlots(null)).toBe(false)
  })

  it("says a server has a feature unless it says it does not", () => {
    expect(serverHas(null, "server_log")).toBe(true)
    expect(serverHas({ status: "ok" }, "server_log")).toBe(true)
    expect(serverHas({ status: "ok", features: { api: 1 } }, "a_name_from_a_newer_client")).toBe(true)
    expect(serverHas({ status: "ok", features: { server_log: false } }, "server_log")).toBe(false)
    expect(serverHas({ status: "ok", features: { model_switch: "restart" } }, "model_switch")).toBe(true)
  })

  it("offers the router's pages only where there is a router", () => {
    const all = ["general", "model", "presets", "voices", "requests", "log", "connection", "system"]
    expect(settingsPages(null)).toEqual(all)
    expect(settingsPages({ status: "ok" })).toEqual(all)
    expect(settingsPages({ status: "ok", features: { router: true, request_log: false, presets: false, server_log: true } })).toEqual(all)
    expect(settingsPages({ status: "ok", features: { router: false, server_log: false } })).toEqual(["general", "model", "connection", "system"])
  })

  it("offers the Library only where the server says it has one: an older server would answer 404", () => {
    expect(settingsPages({ status: "ok", features: { router: true, server_log: true, library: true } }))
      .toEqual(["general", "model", "presets", "voices", "requests", "library", "log", "connection", "system"])
    expect(settingsPages({ status: "ok", features: { router: false, server_log: false, library: true } }))
      .toEqual(["general", "model", "library", "connection", "system"])
    expect(settingsPages({ status: "ok", features: { router: true, library: false } })).not.toContain("library")
    expect(settingsPages({ status: "ok" })).not.toContain("library")
    expect(settingsPages(null)).not.toContain("library")
  })

  it("only offers Continue when the server says it continues a trailing assistant turn", () => {
    expect(supportsContinuation({ status: "ok", continue_assistant: true })).toBe(true)
    expect(supportsContinuation({ status: "ok", continue_assistant: false })).toBe(false)
    expect(supportsContinuation({ status: "ok" })).toBe(false)
    expect(supportsContinuation(null)).toBe(false)
  })
})
