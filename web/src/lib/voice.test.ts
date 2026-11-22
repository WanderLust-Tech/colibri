import { afterEach, describe, expect, it, vi } from "vitest"

import { SilenceDetector, parseEvents, pcmToFloat32, pickVoiceModels, spokenText, streamSpeech, transcribeRecording } from "./voice"

describe("spokenText", () => {
  it("drops markdown, code, link addresses and emoji", () => {
    expect(spokenText("# Title\n\n- **bold** item\n- `code` and [a link](http://x.y/z)\n\n```py\nprint(1)\n```\nDone \u{1F600}.")).toBe("Title bold item code and a link Done .")
  })
  it("drops what follows an unclosed code fence", () => {
    expect(spokenText("Here: ```x = 1\nmore")).toBe("Here:")
  })
})

describe("pickVoiceModels", () => {
  it("takes the first recogniser, the first plain speech model and a cloning one", () => {
    const models = [
      { id: "chat", capabilities: ["chat"] }, { id: "ears", capabilities: ["transcription"] }, { id: "ears2", capabilities: ["transcription"] },
      { id: "qwen3-tts-0.6b-base", capabilities: ["speech"] }, { id: "qwen3-tts-0.6b", capabilities: ["speech"] },
    ]
    expect(pickVoiceModels(models)).toEqual({ asr: "ears", tts: "qwen3-tts-0.6b", clone: "qwen3-tts-0.6b-base" })
  })
  it("falls back to a Base model when it is the only speech model, and says nothing for what is missing", () => {
    expect(pickVoiceModels([{ id: "only-base", capabilities: ["speech"] }])).toEqual({ asr: undefined, tts: "only-base", clone: "only-base" })
    expect(pickVoiceModels([{ id: "chat", capabilities: ["chat"] }, { id: "no-caps" }])).toEqual({ asr: undefined, tts: undefined, clone: undefined })
  })
})

describe("parseEvents", () => {
  it("returns the complete events and keeps the unfinished one", () => {
    const first = parseEvents('data: {"a":1}\n\ndata: {"b":2}\n\ndata: {"c"')
    expect(first.events).toEqual([{ a: 1 }, { b: 2 }])
    expect(first.rest).toBe('data: {"c"')
    expect(parseEvents(first.rest + ":3}\n\n").events).toEqual([{ c: 3 }])
  })
  it("ignores [DONE], comments and what is not JSON", () => {
    expect(parseEvents(": keep-alive\n\ndata: [DONE]\n\ndata: nonsense\n\ndata: {\"ok\":true}\n\n").events).toEqual([{ ok: true }])
  })
})

describe("pcmToFloat32", () => {
  it("reads 16-bit little-endian samples, dropping an odd byte", () => {
    const out = pcmToFloat32(new Uint8Array([0x00, 0x40, 0x00, 0xC0, 0xFF, 0x7F, 0x99]))
    expect(Array.from(out)).toEqual([0.5, -0.5, 32767 / 32768])
  })
  it("respects a view's offset", () => {
    const base = new Uint8Array([9, 9, 0x00, 0x40])
    expect(Array.from(pcmToFloat32(base.subarray(2)))).toEqual([0.5])
  })
})

describe("SilenceDetector", () => {
  const run = (levels: number[], step = 50, options = {}) => {
    const detector = new SilenceDetector(options)
    const seen: string[] = []
    levels.forEach((level, i) => { const outcome = detector.push(level, i * step); if (outcome !== "idle") seen.push(`${outcome}@${i * step}`) })
    return seen
  }
  const flat = (n: number, level: number) => Array(n).fill(level)

  it("hears speech begin and, after a second of quiet, end", () => {
    const seen = run([...flat(10, 0.002), ...flat(20, 0.2), ...flat(30, 0.002)])
    expect(seen[0]).toMatch(/^speech@/)
    expect(seen[1]).toMatch(/^end@/)
    expect(seen).toHaveLength(2)
  })
  it("does not end on a short pause", () => {
    expect(run([...flat(10, 0.002), ...flat(10, 0.2), ...flat(10, 0.002), ...flat(10, 0.2), ...flat(10, 0.002)])).toHaveLength(1)
  })
  it("ignores a click", () => {
    expect(run([...flat(10, 0.002), 0.6, 0.6, ...flat(40, 0.002)])).toEqual([])
  })
  it("raises the bar in a noisy room", () => {
    expect(run([...flat(10, 0.03), ...flat(20, 0.4), ...flat(30, 0.03)])).toHaveLength(2)
    expect(run([...flat(10, 0.03), ...flat(40, 0.05)])).toEqual([])
  })
})

describe("the two calls", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("transcribeRecording posts the file as a form with the key and returns the words", async () => {
    const calls: Array<{ url: string; init: RequestInit }> = []
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      calls.push({ url, init })
      return new Response(JSON.stringify({ text: " hello there " }), { status: 200 })
    }))
    expect(await transcribeRecording("http://h:8000/v1", "k", "ears", new Blob(["x"]))).toBe("hello there")
    expect(calls[0].url).toBe("http://h:8000/v1/audio/transcriptions")
    expect((calls[0].init.headers as Record<string, string>).Authorization).toBe("Bearer k")
    expect((calls[0].init.body as FormData).get("model")).toBe("ears")
    expect((calls[0].init.body as FormData).get("file")).toBeInstanceOf(Blob)
  })

  it("transcribeRecording says what the server said when it fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ error: { message: "the recogniser is down" } }), { status: 500 })))
    await expect(transcribeRecording("http://h/v1", "", "ears", new Blob(["x"]))).rejects.toThrow("the recogniser is down")
  })

  it("streamSpeech hands over each delta as floats and stops at the end, even when an event is split across reads", async () => {
    const pcm = btoa(String.fromCharCode(0x00, 0x40, 0x00, 0xC0))
    const text = `data: {"type":"speech.audio.delta","audio":"${pcm}"}\n\ndata: {"type":"speech.audio.delta","audio":"${pcm}"}\n\ndata: {"type":"speech.audio.done"}\n\n`
    const cut = text.indexOf("audio", 60)
    const encoder = new TextEncoder()
    vi.stubGlobal("fetch", vi.fn(async (_url: string, init: RequestInit) => {
      expect(JSON.parse(init.body as string)).toEqual({ model: "mouth", input: "hi", stream_format: "sse", voice: "ana" })
      return new Response(new ReadableStream({ start(controller) { controller.enqueue(encoder.encode(text.slice(0, cut))); controller.enqueue(encoder.encode(text.slice(cut))); controller.close() } }), { status: 200 })
    }))
    const chunks: number[][] = []
    await streamSpeech("http://h/v1", "", "mouth", "hi", "ana", (samples) => chunks.push(Array.from(samples)))
    expect(chunks).toEqual([[0.5, -0.5], [0.5, -0.5]])
  })

  it("streamSpeech raises an error event and a refusal", async () => {
    const encoder = new TextEncoder()
    vi.stubGlobal("fetch", vi.fn(async () => new Response(new ReadableStream({ start(c) { c.enqueue(encoder.encode('data: {"type":"error","error":{"message":"the card failed"}}\n\n')); c.close() } }), { status: 200 })))
    await expect(streamSpeech("http://h/v1", "", "m", "hi", undefined, () => undefined)).rejects.toThrow("the card failed")
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ error: { message: "no such voice" } }), { status: 400 })))
    await expect(streamSpeech("http://h/v1", "", "m", "hi", "x", () => undefined)).rejects.toThrow("no such voice")
  })
})
