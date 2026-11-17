import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { generateImage, generateMusic } from "./api"
import { cancelJob, followJob, forgetKeptFiles, getJob, jobFraction, jobTitle, keptFileUrl, listJobs, parseJob, runBackground, type Job } from "./jobs"

const wire = (status: string, extra: Record<string, unknown> = {}) => ({
  id: "job_1", object: "job", kind: "image", model: "qwen-image-colibri", status, created: 100, started: null, finished: null,
  request: { prompt: "a red fox", steps: 8, seed: 42 }, ...extra,
})
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
const job = (status: string, extra: Record<string, unknown> = {}) => parseJob(wire(status, extra)) as Job

let fetchMock: ReturnType<typeof vi.fn>
beforeEach(() => {
  fetchMock = vi.fn()
  vi.stubGlobal("fetch", fetchMock)
  forgetKeptFiles()
})
afterEach(() => vi.unstubAllGlobals())

describe("parseJob", () => {
  it("reads a queued, a running and a finished job", () => {
    expect(job("queued", { position: 2 })).toMatchObject({ id: "job_1", kind: "image", status: "queued", position: 2, request: { prompt: "a red fox", steps: 8, seed: 42 } })
    expect(job("running", { started: 101, progress: { stage: "denoise", step: 3, steps: 8, junk: null } }).progress).toEqual({ stage: "denoise", step: 3, steps: 8 })
    const done = job("succeeded", { finished: 120, result: { kind: "images", file: "a.png", url: "/v1/library/images/a.png", width: 768, seed: 42, timings: { denoise: 3.5, x: "no" } } })
    expect(done.result).toEqual({ kind: "images", file: "a.png", url: "/v1/library/images/a.png", width: 768, seed: 42, timings: { denoise: 3.5 } })
    expect(job("failed", { error: { code: "engine_error", message: "out of memory" } }).error).toEqual({ code: "engine_error", message: "out of memory" })
  })

  it("refuses what is not a job and drops what is of the wrong type", () => {
    expect(parseJob(null)).toBeNull()
    expect(parseJob({ id: "x" })).toBeNull()
    expect(parseJob(wire("done"))).toBeNull()
    expect(parseJob(wire("queued", { kind: "video" }))).toBeNull()
    const odd = parseJob(wire("queued", { position: "1", started: "now", request: { prompt: ["x"], steps: 4 }, result: { kind: "images" } }))!
    expect(odd.position).toBeUndefined()
    expect(odd.started).toBeNull()
    expect(odd.request).toEqual({ steps: 4 })
    expect(odd.result).toBeUndefined()
  })
})

describe("the routes", () => {
  it("takes the dashboard's base URL, which already ends in /v1", async () => {
    fetchMock.mockResolvedValue(json({ data: [] }))
    await listJobs("http://h:8000/v1/", "")
    expect(fetchMock.mock.calls[0][0]).toBe("http://h:8000/v1/jobs")
    fetchMock.mockResolvedValue(json(wire("queued")))
    await getJob("http://h:8000/v1", "", "job_1")
    expect(fetchMock.mock.calls[1][0]).toBe("http://h:8000/v1/jobs/job_1")
  })

  it("lists with a query, and sends the key", async () => {
    fetchMock.mockResolvedValue(json({ object: "list", total: 3, has_more: true, data: [wire("running"), { nonsense: 1 }] }))
    const page = await listJobs("http://h/", "k", { status: "running", limit: 2, offset: 1 })
    expect(fetchMock.mock.calls[0][0]).toBe("http://h/v1/jobs?status=running&limit=2&offset=1")
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Authorization: "Bearer k" })
    expect(page.jobs).toHaveLength(1)
    expect(page).toMatchObject({ total: 3, hasMore: true })
  })

  it("says what the server said when it refuses", async () => {
    fetchMock.mockResolvedValue(json({ error: { message: "No job `x`." } }, 404))
    await expect(getJob("http://h", "", "x")).rejects.toThrow("No job `x`.")
  })

  it("cancels by id and tells whether it did", async () => {
    fetchMock.mockResolvedValue(json({ id: "j", cancelled: true, status: "cancelling" }))
    expect(await cancelJob("http://h", "k", "j")).toEqual({ cancelled: true, status: "cancelling" })
    expect(fetchMock.mock.calls[0][0]).toBe("http://h/v1/jobs/cancel")
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ id: "j" })
    fetchMock.mockResolvedValue(json({ id: "j", cancelled: false, status: null }))
    expect(await cancelJob("http://h", "", "j")).toEqual({ cancelled: false, status: null })
  })
})

describe("followJob", () => {
  it("asks until the job is finished and reports each answer", async () => {
    fetchMock.mockResolvedValueOnce(json(wire("running", { progress: { step: 1, steps: 4 } })))
      .mockResolvedValueOnce(json(wire("running", { progress: { step: 3, steps: 4 } })))
      .mockResolvedValueOnce(json(wire("succeeded", { result: { kind: "images", file: "a.png", url: "/v1/library/images/a.png" } })))
    const seen: string[] = []
    const done = await followJob({ baseUrl: "http://h", apiKey: "", job: job("queued"), pollMs: 0, onUpdate: (j) => seen.push(`${j.status}:${jobFraction(j)}`) })
    expect(done.status).toBe("succeeded")
    expect(seen).toEqual(["running:0.25", "running:0.75", "succeeded:null"])
  })

  it("is not put off by a few failed answers, but gives up on many and leaves the job alone", async () => {
    fetchMock.mockRejectedValueOnce(new Error("offline")).mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(json(wire("succeeded", { result: { kind: "images", file: "a.png", url: "/u" } })))
    expect((await followJob({ baseUrl: "http://h", apiKey: "", job: job("running"), pollMs: 0 })).status).toBe("succeeded")

    fetchMock.mockReset()
    fetchMock.mockRejectedValue(new Error("offline"))
    await expect(followJob({ baseUrl: "http://h", apiKey: "", job: job("running"), pollMs: 0 })).rejects.toThrow(/Lost contact.*keeps going.*Library/)
    expect(fetchMock.mock.calls.every(([url]) => !String(url).endsWith("/cancel"))).toBe(true)
  })

  it("cancels the job when asked to stop, then throws AbortError", async () => {
    fetchMock.mockImplementation(async (url: string) => String(url).endsWith("/cancel") ? json({ cancelled: true, status: "cancelling" }) : json(wire("running")))
    const controller = new AbortController()
    const run = followJob({ baseUrl: "http://h", apiKey: "", job: job("running"), pollMs: 20, signal: controller.signal })
    controller.abort()
    await expect(run).rejects.toMatchObject({ name: "AbortError" })
    expect(fetchMock.mock.calls.some(([url]) => String(url) === "http://h/v1/jobs/cancel")).toBe(true)
  })
})

describe("runBackground", () => {
  const options = (send: () => Promise<Response>) => ({
    baseUrl: "http://h", apiKey: "k", signal: new AbortController().signal, send, failed: async (r: Response) => new Error(`refused ${r.status}`), pollMs: 0,
  })

  it("follows the job and returns the kept file as an object URL", async () => {
    fetchMock.mockImplementation(async (url: string) => {
      if (String(url).endsWith("/v1/jobs/job_1")) return json(wire("succeeded", { result: { kind: "images", file: "a.png", url: "/v1/library/images/a.png" } }))
      return new Response(new Blob(["PNG"]), { status: 200 })
    })
    const seen: Job[] = []
    const outcome = await runBackground({ ...options(async () => json(wire("queued"), 202)), onJob: (j) => seen.push(j) })
    expect("job" in outcome && outcome.job.status).toBe("succeeded")
    expect("url" in outcome && outcome.url.startsWith("blob:")).toBe(true)
    expect(seen.map((j) => j.status)).toEqual(["queued"])
    expect(fetchMock.mock.calls.at(-1)![0]).toBe("http://h/v1/library/images/a.png")
    expect(fetchMock.mock.calls.at(-1)![1].headers).toEqual({ Authorization: "Bearer k" })
  })

  it("hands a plain answer back: a server without jobs just made the picture", async () => {
    const answer = json({ data: [{ b64_json: "x" }] })
    const outcome = await runBackground(options(async () => answer))
    expect("plain" in outcome && outcome.plain).toBe(answer)
  })

  it("throws what the server said when the request is refused, or when the job fails or is cancelled", async () => {
    await expect(runBackground(options(async () => json({}, 404)))).rejects.toThrow("refused 404")
    fetchMock.mockResolvedValue(json(wire("failed", { error: { code: "engine_error", message: "out of memory" } })))
    await expect(runBackground(options(async () => json(wire("running"), 202)))).rejects.toThrow("out of memory")
    fetchMock.mockResolvedValue(json(wire("cancelled")))
    await expect(runBackground(options(async () => json(wire("running"), 202)))).rejects.toThrow("cancelled")
    await expect(runBackground(options(async () => json({ nonsense: true }, 202)))).rejects.toThrow("not understood")
  })
})

describe("keptFileUrl", () => {
  it("fetches a file once and gives the same URL again", async () => {
    fetchMock.mockImplementation(async () => new Response(new Blob(["x"])))
    const first = await keptFileUrl("http://h", "", "/v1/library/images/a.png")
    expect(await keptFileUrl("http://h", "", "/v1/library/images/a.png")).toBe(first)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("forgets a failure, so asking again tries again", async () => {
    fetchMock.mockResolvedValueOnce(json({ error: { message: "Not found." } }, 404)).mockResolvedValueOnce(new Response(new Blob(["x"])))
    await expect(keptFileUrl("http://h", "", "/v1/library/images/a.png")).rejects.toThrow("Not found.")
    await expect(keptFileUrl("http://h", "", "/v1/library/images/a.png")).resolves.toMatch(/^blob:/)
  })
})

describe("small helpers", () => {
  it("jobFraction is null until the engine says something", () => {
    expect(jobFraction(job("running"))).toBeNull()
    expect(jobFraction(job("running", { progress: { step: 2, steps: 0 } }))).toBeNull()
    expect(jobFraction(job("running", { progress: { step: 9, steps: 4 } }))).toBe(1)
  })

  it("jobTitle is the prompt, or what kind of thing it is", () => {
    expect(jobTitle(job("queued"))).toBe("a red fox")
    expect(jobTitle(job("queued", { request: {} }))).toBe("Picture")
    expect(jobTitle(job("queued", { kind: "music", request: {} }))).toBe("Song")
  })
})

describe("generateImage and generateMusic in the background", () => {
  const server = (kind: "images" | "music") => async (url: string, init?: RequestInit) => {
    const target = String(url)
    if (init?.method === "POST" && !target.endsWith("/cancel")) return json(wire("queued", { kind: kind === "images" ? "image" : "music" }), 202)
    if (target.includes("/v1/jobs/job_1")) {
      return json(wire("succeeded", { kind: kind === "images" ? "image" : "music", progress: { stage: "denoise", step: 8, steps: 8 },
        result: { kind, file: "a", url: `/v1/library/${kind}/a`, width: 512, height: 256, seed: 77, steps: 4, duration: 12.5, timings: { denoise: 2 } } }))
    }
    return new Response(new Blob(["bytes"]), { status: 200 })
  }

  it("a picture goes with the Prefer header and comes back as the kept file", async () => {
    fetchMock.mockImplementation(server("images"))
    const states: string[] = []
    const picture = await generateImage({ baseUrl: "http://h", apiKey: "k", model: "m", prompt: "a fox", width: 768, height: 512, steps: 8, seed: 1,
      signal: new AbortController().signal, background: true, onJob: (j) => states.push(j.status) })
    expect(fetchMock.mock.calls[0][1].headers).toMatchObject({ Prefer: "respond-async", Authorization: "Bearer k" })
    expect(states).toEqual(["queued", "succeeded"])
    expect(picture).toMatchObject({ prompt: "a fox", width: 512, height: 256, steps: 4, seed: 77, timings: { denoise: 2 } })
    expect(picture.url).toMatch(/^blob:/)
  })

  it("a song, the same way", async () => {
    fetchMock.mockImplementation(server("music"))
    const song = await generateMusic({ baseUrl: "http://h", apiKey: "", model: "m", prompt: "calm", duration: 30, steps: 8, seed: 1,
      signal: new AbortController().signal, background: true })
    expect(fetchMock.mock.calls[0][1].headers).toMatchObject({ Prefer: "respond-async" })
    expect(song).toMatchObject({ prompt: "calm", duration: 12.5, seed: 77 })
    expect(song.url).toMatch(/^blob:/)
  })

  it("a server that does not know jobs answers the request itself, and that answer is used", async () => {
    fetchMock.mockResolvedValue(json({ created: 1, data: [{ b64_json: "QUJD" }], colibri: { width: 64, height: 64, seed: 5, steps: 2 } }))
    const picture = await generateImage({ baseUrl: "http://h", apiKey: "", model: "m", prompt: "a fox", width: 64, height: 64, steps: 2, seed: 5,
      signal: new AbortController().signal, background: true })
    expect(picture.url).toBe("data:image/png;base64,QUJD")
  })

  it("an error before the job exists is the server's own message", async () => {
    fetchMock.mockResolvedValue(json({ error: { message: "The model `x` does not exist." } }, 404))
    await expect(generateImage({ baseUrl: "http://h", apiKey: "", model: "x", prompt: "a", steps: 2, seed: 1, signal: new AbortController().signal, background: true }))
      .rejects.toThrow("does not exist")
  })
})
