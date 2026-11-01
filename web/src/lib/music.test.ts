import { afterEach, describe, expect, it, vi } from "vitest"

import { generateMusic, generatesMusic, listModelInfo, musicLimits, wavUrl, type GenerateMusicOptions, type MusicProgress } from "./api"
import { chatTurns } from "./images"
import {
  DEFAULT_MUSIC_LIMITS, MUSIC_LANGUAGES, clampDuration, clampMusicSteps, lyricBytes, musicFileName, musicProgressFraction,
  songClock, validDuration, validLanguage,
} from "./music"
import en from "../i18n/en"

afterEach(() => vi.unstubAllGlobals())

describe("music limits", () => {
  const limits = DEFAULT_MUSIC_LIMITS

  it("holds a length to the engine's range and to a tenth of a second", () => {
    expect([0, 0.4, 1, 12.34, 600, 601, 9999].map((v) => clampDuration(v, limits))).toEqual([1, 1, 1, 12.3, 600, 600, 600])
    expect(clampDuration(Number.NaN, limits)).toBe(30)
    expect(clampDuration(2, { ...limits, minDuration: 5, maxDuration: 60 })).toBe(5)
  })

  it("says whether a length is acceptable without changing it", () => {
    expect([0.9, 1, 30, 600, 600.1, Number.NaN, Infinity].map((v) => validDuration(v, limits))).toEqual([false, true, true, true, false, false, false])
  })

  it("holds the steps to whole numbers between 1 and the engine's most", () => {
    expect([0, 1, 2.6, 8, 20, 21, 99].map((v) => clampMusicSteps(v, limits))).toEqual([1, 1, 3, 8, 20, 20, 20])
    expect(clampMusicSteps(Number.NaN, limits)).toBe(8)
  })

  it("counts the lyrics' UTF-8 bytes, which is what the engine limits", () => {
    expect([lyricBytes(""), lyricBytes("abc"), lyricBytes("é"), lyricBytes("你好"), lyricBytes("😀")]).toEqual([0, 3, 2, 6, 4])
  })

  it("knows the languages the model sings in, and that none is allowed", () => {
    expect(validLanguage("")).toBe(true)
    for (const code of ["en", "zh", "ja", "yue", "pt"]) expect(validLanguage(code)).toBe(true)
    for (const code of ["xx", "EN", "english", "unknown"]) expect(validLanguage(code)).toBe(false)   // "unknown" is what none means
    expect(new Set(MUSIC_LANGUAGES).size).toBe(MUSIC_LANGUAGES.length)
  })

  it("reads /v1/models' audio block, filling what is missing with the contract's defaults", () => {
    expect(musicLimits(undefined)).toBeUndefined()
    expect(musicLimits("nope")).toBeUndefined()
    expect(musicLimits({})).toEqual(DEFAULT_MUSIC_LIMITS)
    expect(musicLimits({ sample_rate: 44100, min_duration: 5, max_duration: 60, default_duration: 20, default_steps: 4, max_steps: 10, default_shift: 3, max_lyric_bytes: 1000 }))
      .toEqual({ sampleRate: 44100, minDuration: 5, maxDuration: 60, defaultDuration: 20, defaultSteps: 4, maxSteps: 10, defaultShift: 3, maxLyricBytes: 1000 })
    expect(musicLimits({ max_duration: -1, max_steps: "many" })).toEqual(DEFAULT_MUSIC_LIMITS)       // junk is not guessed at
    expect(musicLimits({ min_duration: 90, max_duration: 60 })).toBeUndefined()                      // an empty range is no range
  })
})

describe("music progress", () => {
  const at = (stage: MusicProgress["stage"], step: number, steps: number) => musicProgressFraction({ stage, step, steps })

  it("is empty before the engine says anything and grows through the stages", () => {
    expect(musicProgressFraction(null)).toBe(0)
    const run = [at("text", 0, 1), at("condition", 0, 1), at("denoise", 0, 8), at("denoise", 4, 8), at("denoise", 8, 8), at("decode", 0, 3), at("decode", 3, 3)]
    for (let i = 1; i < run.length; i++) expect(run[i]).toBeGreaterThanOrEqual(run[i - 1])     // never backwards (it rests between two stages)
    expect(new Set(run).size).toBe(run.length - 2)         // two resting points: after the conditioning and after the last step
    expect(run[0]).toBeGreaterThan(0)
    expect(run.at(-1)).toBeCloseTo(1, 10)
  })

  it("survives nonsense without leaving the bar's range", () => {
    for (const progress of [at("denoise", 99, 8), at("denoise", -5, 8), at("decode", 5, 0), at("denoise", Number.NaN, Number.NaN)]) {
      expect(progress).toBeGreaterThanOrEqual(0)
      expect(progress).toBeLessThanOrEqual(1)
    }
    expect(musicProgressFraction({ stage: "unknown" as never, step: 1, steps: 1 })).toBe(0)
  })
})

describe("music names and clocks", () => {
  it("names a file after the caption and its seed", () => {
    expect(musicFileName("Calm piano, with soft strings!", 42)).toBe("colibri-calm-piano-with-soft-strings-42.wav")
    expect(musicFileName("Café déjà vu", 7)).toBe("colibri-cafe-deja-vu-7.wav")
    expect(musicFileName("", 3)).toBe("colibri-music-3.wav")
    expect(musicFileName("日本語", 3)).toBe("colibri-music-3.wav")
    expect(musicFileName("word ".repeat(40), 1).length).toBeLessThan(70)
  })

  it("writes a song's length as minutes and seconds", () => {
    expect([0, 9.4, 59.6, 60, 125, 600].map(songClock)).toEqual(["0:00", "0:09", "1:00", "1:00", "2:05", "10:00"])
  })

  it("keeps music turns out of what a text model is shown", () => {
    const turns = [
      { id: "1", role: "user" as const, content: "hello" }, { id: "2", role: "assistant" as const, content: "hi" },
      { id: "3", role: "user" as const, content: "calm piano", mode: "music" as const }, { id: "4", role: "assistant" as const, content: "", mode: "music" as const },
      { id: "5", role: "user" as const, content: "a fox", mode: "image" as const },
    ]
    expect(chatTurns(turns).map((turn) => turn.id)).toEqual(["1", "2"])
  })
})

describe("listModelInfo and music", () => {
  it("recognises a music model and reads what it accepts", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ data: [
      { id: "ace-step-1.5-colibri", capabilities: ["music_generation"], audio: { max_duration: 120, max_steps: 12 } },
      { id: "no-block", capabilities: ["music_generation"] },
      { id: "chatty", capabilities: ["chat", "systemone"], audio: { max_duration: 5 } },
      { id: "old-server" },
    ] }))))
    const models = await listModelInfo("http://localhost:8000/v1", "")
    expect(models.map(generatesMusic)).toEqual([true, true, false, false])
    expect(models[0].music).toMatchObject({ maxDuration: 120, maxSteps: 12, minDuration: 1, defaultDuration: 30 })
    expect(models[1].music).toBeUndefined()           // the page then uses DEFAULT_MUSIC_LIMITS for it
    expect(models[2].music).toBeUndefined()           // a block without the capability is not a music model
  })
})

describe("generateMusic", () => {
  const encoder = new TextEncoder()
  const sse = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
  /* "RIFF" in base64 is not a WAV, and it does not need to be: the page hands the bytes to an <audio>. */
  const wav = btoa("RIFF....WAVEfmt ")
  const answer = {
    created: 1790000000,
    data: [{ b64_json: wav, format: "wav" }],
    colibri: { sample_rate: 48000, samples: 480000, duration: 10, seed: 1234, steps: 8, shift: 1, truncated: false, timings: { text: 0.7, condition: 1.1, denoise: 12.5, decode: 1.9 } },
  }
  const events = [
    sse("music_generation.progress", { stage: "text", step: 0, steps: 1 }),
    sse("music_generation.progress", { stage: "denoise", step: 3, steps: 8 }),
    sse("music_generation.progress", { stage: "decode", step: 1, steps: 1 }),
    sse("music_generation.completed", answer),
    "data: [DONE]\n\n",
  ].join("")
  const streamed = (text: string, { cut = 41, close = true } = {}) => new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      for (let at = 0; at < text.length; at += cut) controller.enqueue(encoder.encode(text.slice(at, at + cut)))
      if (close) controller.close()
    },
  }), { headers: { "content-type": "text/event-stream" } })
  const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } })
  const options = (extra: Partial<GenerateMusicOptions> = {}): GenerateMusicOptions => ({
    baseUrl: "http://localhost:8000/v1/", apiKey: "secret", model: "ace-step-1.5-colibri", prompt: "calm piano",
    duration: 10, steps: 8, seed: 1234, signal: new AbortController().signal, ...extra,
  })
  const bodyOf = (call: unknown[]) => JSON.parse(String((call[1] as RequestInit).body)) as Record<string, unknown>
  const bytesOf = async (url: string) => new Uint8Array(await (await fetch(url)).arrayBuffer())

  it("posts the contract's body and streams progress and the song", async () => {
    const real = globalThis.fetch
    const fetchMock = vi.fn().mockResolvedValue(streamed(events))
    vi.stubGlobal("fetch", fetchMock)
    const progress: MusicProgress[] = []
    const song = await generateMusic(options({ onProgress: (item) => progress.push(item) }))
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:8000/v1/audio/music")
    expect((fetchMock.mock.calls[0][1] as RequestInit).headers).toMatchObject({ Authorization: "Bearer secret" })
    expect(bodyOf(fetchMock.mock.calls[0])).toEqual({ model: "ace-step-1.5-colibri", prompt: "calm piano", duration: 10, steps: 8, seed: 1234, stream: true })
    expect(progress.map((item) => [item.stage, item.step, item.steps])).toEqual([["text", 0, 1], ["denoise", 3, 8], ["decode", 1, 1]])
    expect(song).toMatchObject({ prompt: "calm piano", duration: 10, steps: 8, seed: 1234, timings: { text: 0.7, condition: 1.1, denoise: 12.5, decode: 1.9 } })
    expect(song.url).toMatch(/^blob:/)
    expect(song.truncated).toBeUndefined()
    vi.stubGlobal("fetch", real)
    expect(Array.from(await bytesOf(song.url!))).toEqual(Array.from(atob(wav), (c) => c.charCodeAt(0)))     // the WAV, byte for byte
  })

  it("sends lyrics and a language only when there are some", async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(streamed(events)))
    vi.stubGlobal("fetch", fetchMock)
    const withBoth = await generateMusic(options({ lyrics: "[Verse]\nla la la", language: "en" }))
    expect(bodyOf(fetchMock.mock.calls[0])).toMatchObject({ lyrics: "[Verse]\nla la la", language: "en" })
    expect(withBoth).toMatchObject({ lyrics: "[Verse]\nla la la", language: "en" })
    await generateMusic(options({ lyrics: "   \n", language: "" }))
    expect(bodyOf(fetchMock.mock.calls[1])).not.toHaveProperty("lyrics")
    expect(bodyOf(fetchMock.mock.calls[1])).not.toHaveProperty("language")
  })

  it("reports what the engine made over what was asked for, and that it cut the text", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamed(
      sse("music_generation.completed", { ...answer, colibri: { ...answer.colibri, seed: 99, duration: 9.6, truncated: true } }) + "data: [DONE]\n\n")))
    await expect(generateMusic(options())).resolves.toMatchObject({ seed: 99, duration: 9.6, truncated: true })
  })

  it("stops reading at [DONE] even when the connection stays open", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamed(events, { close: false })))
    await expect(generateMusic(options())).resolves.toMatchObject({ seed: 1234 })
  })

  it("accepts a server that ignores `stream` and answers with the plain object, without asking twice", async () => {
    const fetchMock = vi.fn().mockResolvedValue(json(answer))
    vi.stubGlobal("fetch", fetchMock)
    await expect(generateMusic(options())).resolves.toMatchObject({ seed: 1234 })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("falls back to one plain request when the stream never starts, and says so", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(null, { headers: { "content-type": "text/event-stream" } }))
      .mockResolvedValueOnce(json(answer))
    vi.stubGlobal("fetch", fetchMock)
    const plain = vi.fn()
    await expect(generateMusic(options({ onPlainRequest: plain }))).resolves.toMatchObject({ seed: 1234 })
    expect(plain).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(bodyOf(fetchMock.mock.calls[0])).toMatchObject({ stream: true })
    expect(bodyOf(fetchMock.mock.calls[1])).toMatchObject({ stream: false })
  })

  it("shows the server's own message for an HTTP 400 and does not retry", async () => {
    const fetchMock = vi.fn().mockResolvedValue(json({ error: { message: "`duration` must be a number of seconds between 1 and 600.", type: "invalid_request_error", param: "duration" } }, 400))
    vi.stubGlobal("fetch", fetchMock)
    await expect(generateMusic(options())).rejects.toThrow("`duration` must be a number of seconds between 1 and 600.")
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("shows a full queue's 503 as the server wrote it", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ error: { message: "The server is busy.", type: "server_error" } }, 503)))
    await expect(generateMusic(options())).rejects.toThrow("The server is busy.")
  })

  it("raises an error event's message, and a stream that ends before the song", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamed(
      sse("music_generation.progress", { stage: "text", step: 0, steps: 1 }) +
      sse("error", { error: { message: "The music engine failed: out of memory", type: "server_error" } }) + "data: [DONE]\n\n")))
    await expect(generateMusic(options())).rejects.toThrow("The music engine failed: out of memory")
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamed(sse("music_generation.progress", { stage: "denoise", step: 1, steps: 8 }))))
    await expect(generateMusic(options())).rejects.toThrow("before the song was finished")
  })

  it("does not start a second song when the caller aborted", async () => {
    const controller = new AbortController()
    const fetchMock = vi.fn().mockImplementation(() => { controller.abort(); return Promise.reject(new DOMException("aborted", "AbortError")) })
    vi.stubGlobal("fetch", fetchMock)
    await expect(generateMusic(options({ signal: controller.signal }))).rejects.toThrow()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("refuses an answer without a song", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ created: 1, data: [] })))
    await expect(generateMusic(options())).rejects.toThrow("without a song")
  })

  it("turns base64 into a playable URL of a WAV blob", async () => {
    const url = wavUrl(btoa("abc"))
    expect(url).toMatch(/^blob:/)
    const blob = await (await fetch(url)).blob()
    expect(blob.type).toBe("audio/wav")
    expect(await blob.text()).toBe("abc")
  })
})

describe("the music strings", () => {
  /* Every source file of the page, as text (the tests themselves left out). */
  const files = import.meta.glob(["../**/*.ts", "../**/*.tsx", "!../**/*.test.ts", "!../**/*.test.tsx"], { query: "?raw", import: "default", eager: true }) as Record<string, string>
  const texts = Object.values(files)

  it("has every key the page asks for, in English", () => {
    const wanted = new Set<string>()
    for (const text of texts) {
      for (const match of text.matchAll(/\bt\(\s*"(music\.[\w.]+)"/g)) wanted.add(match[1])
      for (const match of text.matchAll(/\bt\(`(music\.[\w.]+)\.\$\{/g)) for (const stage of ["text", "condition", "decode"]) wanted.add(`${match[1]}.${stage}`)
      for (const match of text.matchAll(/(?:key|label):\s*"(music\.[\w.]+)"/g)) wanted.add(match[1])
    }
    expect(wanted.size).toBeGreaterThan(20)
    expect([...wanted].filter((key) => !(key in en))).toEqual([])
  })

  it("has no English key that nothing asks for", () => {
    const text = texts.join("\n")
    const unused = Object.keys(en).filter((key) => key.startsWith("music.") && !text.includes(`"${key}"`) && !/music\.stage\.(text|condition|decode)/.test(key))
    expect(unused).toEqual([])
  })
})
