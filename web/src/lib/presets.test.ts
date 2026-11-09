import { afterEach, describe, expect, it, vi } from "vitest"

import { deletePreset, listPresets, listModelInfo, listRequests, presetInfo, savePreset } from "./api"
import { controlDefaults, defaultsOf, nameProblem, parseDefault, rowsOf, suggestedKeys } from "./presets"

afterEach(() => vi.unstubAllGlobals())

describe("controlDefaults", () => {
  it("takes the settings the page has controls for and keeps them in range", () => {
    expect(controlDefaults({ temperature: 0.2, max_tokens: 300, size: "1024x768", steps: 20, seed: 4 })).toEqual({
      temperature: 0.2, maxTokens: 300, size: { width: 1024, height: 768 }, steps: 20,
    })
    expect(controlDefaults({ temperature: 9, max_tokens: 999999 })).toEqual({ temperature: 2, maxTokens: 32768 })
    expect(controlDefaults({ max_completion_tokens: 50 })).toEqual({ maxTokens: 50 })
  })

  it("leaves out what is not a usable number or size", () => {
    expect(controlDefaults({ temperature: "hot", max_tokens: 0, size: "big", steps: -1 })).toEqual({})
    expect(controlDefaults({})).toEqual({})
  })
})

describe("default rows", () => {
  it("shows a default as the text of a field and reads it back as the same value", () => {
    const defaults = { temperature: 0.2, max_tokens: 300, size: "1024x768", thinking: true, stop: ["\n\n"] }
    expect(rowsOf(defaults).map((row) => row.value)).toEqual(["0.2", "300", "1024x768", "true", '["\\n\\n"]'])
    expect(defaultsOf(rowsOf(defaults))).toEqual(defaults)
  })

  it("reads numbers, booleans and JSON, and keeps anything else as text", () => {
    expect(parseDefault(" 20 ")).toBe(20)
    expect(parseDefault("-0.5")).toBe(-0.5)
    expect(parseDefault("false")).toBe(false)
    expect(parseDefault("null")).toBeNull()
    expect(parseDefault('{"a": 1}')).toEqual({ a: 1 })
    expect(parseDefault("{not json")).toBe("{not json")
    expect(parseDefault("1024x768")).toBe("1024x768")
    expect(parseDefault("1e3")).toBe(1000)
  })

  it("drops a row with no key or no value and keeps the last of a repeated key", () => {
    expect(defaultsOf([{ key: "", value: "1" }, { key: "steps", value: "" }, { key: "temperature", value: "0.1" },
      { key: " temperature ", value: "0.3" }])).toEqual({ temperature: 0.3 })
  })

  it("suggests names that fit the kind of model", () => {
    expect(suggestedKeys("text")).toContain("max_tokens")
    expect(suggestedKeys("image")).toContain("steps")
    expect(suggestedKeys("audio")).toContain("duration")
  })
})

describe("nameProblem", () => {
  it.each([["terse", null], ["Photo-1024", null], ["a.b_c-d", null], ["", "empty"], ["  ", "empty"], ["has space", "invalid"],
    ["-lead", "invalid"], ["x".repeat(65), "invalid"]])("%j -> %s", (name, problem) => {
    expect(nameProblem(name)).toBe(problem)
  })
})

describe("presets in /v1/models", () => {
  it("reads preset_of and the preset's own settings, and ignores a malformed one", () => {
    expect(presetInfo("gemma-4-e2b-colibri", { name: "pirate", defaults: { temperature: 0 }, system: "Arr", description: "d", keep_alive: "2m" }))
      .toEqual({ of: "gemma-4-e2b-colibri", defaults: { temperature: 0 }, system: "Arr", description: "d", keepAlive: "2m" })
    expect(presetInfo("m", { defaults: [1] })).toEqual({ of: "m", defaults: {} })
    expect(presetInfo(undefined, { defaults: {} })).toBeUndefined()
    expect(presetInfo("m", null)).toBeUndefined()
  })

  it("keeps the preset on the model it describes, capabilities and all", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ data: [
      { id: "photo", capabilities: ["image_generation"], preset_of: "qwen-image", preset: { defaults: { size: "1024x768" } } },
      { id: "qwen-image", capabilities: ["image_generation"] },
      { id: "terse", preset_of: "m", preset: { defaults: {}, system: "Short." } },
    ] })))
    vi.stubGlobal("fetch", fetchMock)
    const models = await listModelInfo("http://h/v1", "")
    expect(models[0]).toMatchObject({ id: "photo", capabilities: ["image_generation"], preset: { of: "qwen-image", defaults: { size: "1024x768" } } })
    expect(models[1].preset).toBeUndefined()
    expect(models[2]).toMatchObject({ id: "terse", preset: { system: "Short." } })
  })
})

describe("presets and requests API", () => {
  const reply = (body: unknown, status = 200) => vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status }))

  it("lists the presets of the file with what could not be used", async () => {
    const fetchMock = reply({ path: "/p/presets.json", data: [{ name: "terse", model: "m", defaults: {}, available: true, reason: null, resolved: "m-colibri" }],
      problems: [{ name: "Bad", reason: "the name" }], error: null })
    vi.stubGlobal("fetch", fetchMock)
    const list = await listPresets("http://h/v1", "k")
    expect(fetchMock).toHaveBeenCalledWith("http://h/v1/presets", expect.objectContaining({ headers: expect.objectContaining({ Authorization: "Bearer k" }) }))
    expect(list).toMatchObject({ supported: true, path: "/p/presets.json", problems: [{ name: "Bad" }], error: null })
    expect(list.presets[0].name).toBe("terse")
  })

  it("says a server without the route does not support presets or a log, rather than failing", async () => {
    vi.stubGlobal("fetch", reply({ error: { message: "Not found." } }, 404))
    expect(await listPresets("http://h/v1", "")).toMatchObject({ supported: false, presets: [] })
    expect(await listRequests("http://h/v1", "")).toEqual({ enabled: false, path: null, requests: [] })
  })

  it("saves a preset under its name and says what happened", async () => {
    const fetchMock = reply({ status: "created", name: "terse" })
    vi.stubGlobal("fetch", fetchMock)
    expect(await savePreset("http://h/v1", "k", "terse", { model: "m", defaults: { temperature: 0.2 } })).toBe("created")
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("http://h/v1/presets")
    expect(init.method).toBe("POST")
    expect(JSON.parse(init.body)).toEqual({ name: "terse", model: "m", defaults: { temperature: 0.2 } })
  })

  it("shows the router's reason when a preset is refused", async () => {
    vi.stubGlobal("fetch", reply({ error: { message: "`messages` cannot be a default" } }, 400))
    await expect(savePreset("http://h/v1", "", "x", { model: "m" })).rejects.toThrow("`messages` cannot be a default")
    vi.stubGlobal("fetch", reply({ error: { message: "No preset is called `x`." } }, 404))
    await expect(deletePreset("http://h/v1", "", "x")).rejects.toThrow("No preset is called")
  })

  it("deletes by name", async () => {
    const fetchMock = reply({ status: "deleted", name: "terse" })
    vi.stubGlobal("fetch", fetchMock)
    await deletePreset("http://h/v1", "", "terse")
    expect(fetchMock.mock.calls[0][0]).toBe("http://h/v1/presets/delete")
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ name: "terse" })
  })

  it("asks for the last requests and passes the log's path", async () => {
    const fetchMock = reply({ enabled: true, path: "/l/req.log", data: [{ ts: "t", path: "/v1/chat/completions", status: 200, seconds: 1.2 }] })
    vi.stubGlobal("fetch", fetchMock)
    const view = await listRequests("http://h/v1", "", 25)
    expect(fetchMock.mock.calls[0][0]).toBe("http://h/v1/requests?limit=25")
    expect(view).toMatchObject({ enabled: true, path: "/l/req.log" })
    expect(view.requests).toHaveLength(1)
  })
})
