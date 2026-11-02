import { describe, expect, it, vi } from "vitest"

import { persistPublicSettings, stored, takeKeyFromLocation, type StringStorage } from "./storage"

describe("the API key carried by a link", () => {
  const at = (hash: string, search = "") => ({ hash, pathname: "/", search })

  it("reads the key from the fragment and removes it from the address bar", () => {
    const history = { replaceState: vi.fn() }
    expect(takeKeyFromLocation(at("#key=abc-DEF_123"), history)).toBe("abc-DEF_123")
    expect(history.replaceState).toHaveBeenCalledWith(null, "", "/")
  })

  it("keeps the query and any other fragment parameters", () => {
    const history = { replaceState: vi.fn() }
    expect(takeKeyFromLocation(at("#view=music&key=k1", "?a=1"), history)).toBe("k1")
    expect(history.replaceState).toHaveBeenCalledWith(null, "", "/?a=1#view=music")
  })

  it("does nothing without a key", () => {
    const history = { replaceState: vi.fn() }
    expect(takeKeyFromLocation(at(""), history)).toBe("")
    expect(takeKeyFromLocation(at("#view=music"), history)).toBe("")
    expect(history.replaceState).not.toHaveBeenCalled()
  })

  it("still returns the key when the history API refuses", () => {
    const history = { replaceState: () => { throw new Error("blocked") } }
    expect(takeKeyFromLocation(at("#key=k"), history)).toBe("k")
  })
})

function memoryStorage(initial: Record<string, string> = {}): StringStorage & { values: Map<string, string> } {
  const values = new Map(Object.entries(initial))
  return {
    values,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value) },
    removeItem: (key) => { values.delete(key) },
  }
}

describe("browser settings persistence", () => {
  it("persists endpoint and model but removes legacy API credentials", () => {
    const storage = memoryStorage({ "colibri.apiKey": "legacy-secret" })
    persistPublicSettings(storage, "https://localhost/v1", "test-model")
    expect(Object.fromEntries(storage.values)).toEqual({
      "colibri.baseUrl": "https://localhost/v1",
      "colibri.model": "test-model",
    })
  })

  it("does not attempt to write an API key", () => {
    const storage = memoryStorage()
    const setItem = vi.spyOn(storage, "setItem")
    persistPublicSettings(storage, "http://localhost/v1", "test-model")
    expect(setItem).not.toHaveBeenCalledWith("colibri.apiKey", expect.anything())
  })

  it("uses a fallback when storage access is unavailable", () => {
    expect(stored({ getItem: () => { throw new Error("denied") } }, "key", "fallback")).toBe("fallback")
  })
})
