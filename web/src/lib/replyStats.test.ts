import { afterEach, describe, expect, it, vi } from "vitest"

import { listModelInfo, streamChat, type ChatMessage, type TokenUsage } from "./api"
import { buildReplyStats, cachedTokens, contextShare, replyParts, shortModel, type ReplyStats } from "./replyStats"
import { setStats } from "./transcript"

afterEach(() => vi.unstubAllGlobals())

const usage = (prompt: number, completion: number, cached?: number | null): TokenUsage => ({
  prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion,
  ...(cached === undefined ? {} : { prompt_tokens_details: cached === null ? null : { cached_tokens: cached } }),
})

/* The same words the English dictionary has, so the tests read what a person would. */
import en from "../i18n/en"
const t = (key: string, vars?: Record<string, string | number>) =>
  (en[key] ?? key).replace(/\{\{(\w+)\}\}/g, (_, name) => String(vars?.[name] ?? `{{${name}}}`))

describe("cachedTokens", () => {
  it("reads the count the server gives, including a real zero", () => {
    expect(cachedTokens(usage(279, 2, 266))).toBe(266)
    expect(cachedTokens(usage(204, 2, 0))).toBe(0)
  })

  it("is undefined when the server does not say, so a missing count never reads as none cached", () => {
    expect(cachedTokens(usage(279, 2))).toBeUndefined()
    expect(cachedTokens(usage(279, 2, null))).toBeUndefined()
    expect(cachedTokens(null)).toBeUndefined()
  })

  it("ignores a bad figure and never claims more than the prompt held", () => {
    expect(cachedTokens({ ...usage(100, 1), prompt_tokens_details: { cached_tokens: -5 } })).toBeUndefined()
    expect(cachedTokens({ ...usage(100, 1), prompt_tokens_details: { cached_tokens: Number.NaN } })).toBeUndefined()
    expect(cachedTokens(usage(100, 1, 900))).toBe(100)
  })
})

describe("buildReplyStats", () => {
  it("collects what the reply cost", () => {
    expect(buildReplyStats({
      usage: usage(279, 2, 266), model: "gemma-4-e2b-colibri", preset: "terse",
      tokPerSec: 18.04, ttftMs: 1234.4, elapsedMs: 4100.6, contextLength: 32768,
    })).toEqual({
      model: "gemma-4-e2b-colibri", preset: "terse", promptTokens: 279, cachedTokens: 266, completionTokens: 2,
      tokPerSec: 18.04, ttftMs: 1234, elapsedMs: 4101, contextLength: 32768,
    })
  })

  it("is null for a reply with no usage block, which has no tokens to report", () => {
    expect(buildReplyStats({ usage: null, model: "gemma-4-e2b-colibri" })).toBeNull()
  })

  it("leaves out what was not measured", () => {
    expect(buildReplyStats({ usage: usage(10, 3), model: null, tokPerSec: null, ttftMs: null, elapsedMs: 0 }))
      .toEqual({ promptTokens: 10, completionTokens: 3 })
  })
})

describe("contextShare", () => {
  const base: ReplyStats = { promptTokens: 6000, completionTokens: 500, contextLength: 65536 }

  it("is the prompt plus the answer over the window, with a decimal only below 10%", () => {
    expect(contextShare(base)).toEqual({ percent: "9.9", total: 65536 })
    expect(contextShare({ ...base, promptTokens: 30000 })).toEqual({ percent: "47", total: 65536 })
  })

  it("needs a window and a prompt, and never passes 100%", () => {
    expect(contextShare({ promptTokens: 10 })).toBeUndefined()
    expect(contextShare({ contextLength: 100 })).toBeUndefined()
    expect(contextShare({ promptTokens: 500, completionTokens: 10, contextLength: 100 })?.percent).toBe("100")
  })
})

describe("replyParts", () => {
  it("words a continuation, with the model, speed and window", () => {
    const parts = replyParts({
      model: "qwen3-coder-30b-a3b-colibri", promptTokens: 279, cachedTokens: 266, completionTokens: 2,
      tokPerSec: 18.04, ttftMs: 1234, elapsedMs: 4101, contextLength: 65536,
    }, t)
    expect(parts).toEqual([
      "qwen3-coder-30b-a3b", "266 of 279 prompt tokens cached", "2 tokens", "18.0 tok/s", "first token 1.2 s", "4.1 s",
      "0.4% of 65,536 context",
    ])
  })

  it("says none were cached when the engine read the whole prompt, and says nothing of it when the server is silent", () => {
    expect(replyParts({ promptTokens: 204, cachedTokens: 0 }, t)).toEqual(["204 prompt tokens, none cached"])
    expect(replyParts({ promptTokens: 204 }, t)).toEqual(["204 prompt tokens"])
  })

  it("names the preset that routed the request", () => {
    expect(replyParts({ model: "gemma-4-e2b-colibri", preset: "terse", promptTokens: 5 }, t))
      .toEqual(["gemma-4-e2b", "via preset terse", "5 prompt tokens"])
  })

  it("shows whole seconds once a figure is long, and no parts for an empty record", () => {
    expect(replyParts({ elapsedMs: 163000 }, t)).toEqual(["163 s"])
    expect(replyParts({}, t)).toEqual([])
  })
})

describe("shortModel", () => {
  it("drops only the shared suffix", () => {
    expect(shortModel("qwen3-coder-30b-a3b-colibri")).toBe("qwen3-coder-30b-a3b")
    expect(shortModel("colibri-test")).toBe("colibri-test")
    expect(shortModel("terse")).toBe("terse")
  })
})

describe("setStats", () => {
  const messages: ChatMessage[] = [
    { id: "u1", role: "user", content: "hi" },
    { id: "a1", role: "assistant", content: "hello", finish: "stop" },
  ]

  it("attaches the stats to the one bubble and leaves the rest alone", () => {
    const next = setStats(messages, "a1", { promptTokens: 5 })
    expect(next[1].stats).toEqual({ promptTokens: 5 })
    expect(next[0]).toBe(messages[0])
    expect(next[1].content).toBe("hello")
  })

  it("replaces the earlier figures of a continued turn, and does nothing for null", () => {
    const first = setStats(messages, "a1", { promptTokens: 5 })
    expect(setStats(first, "a1", { promptTokens: 9 })[1].stats).toEqual({ promptTokens: 9 })
    expect(setStats(first, "a1", null)).toBe(first)
  })
})

describe("what the stream and the model list give", () => {
  const sse = (...events: unknown[]) => new Response(
    events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("") + "data: [DONE]\n\n",
    { headers: { "content-type": "text/event-stream" } },
  )

  it("returns the model that answered and the cached-token count from the usage chunk", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sse(
      { model: "gemma-4-e2b-colibri", choices: [{ delta: { content: "A" } }], usage: null },
      { model: "gemma-4-e2b-colibri", choices: [{ delta: {}, finish_reason: "stop" }], usage: null },
      { model: "gemma-4-e2b-colibri", choices: [], usage: usage(279, 2, 266) },
    )))
    const result = await streamChat({
      baseUrl: "http://x/v1", apiKey: "", model: "terse", messages: [], temperature: 0, maxTokens: 8,
      enableThinking: false, signal: new AbortController().signal, onDelta: () => undefined,
    })
    expect(result.model).toBe("gemma-4-e2b-colibri")
    expect(cachedTokens(result.usage)).toBe(266)
  })

  it("has no model when the stream does not name one", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sse({ choices: [{ delta: { content: "A" } }] })))
    const result = await streamChat({
      baseUrl: "http://x/v1", apiKey: "", model: "m", messages: [], temperature: 0, maxTokens: 8,
      enableThinking: false, signal: new AbortController().signal, onDelta: () => undefined,
    })
    expect(result.model).toBeNull()
  })

  it("reads a model's window from its card, and only a sensible one", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ data: [
      { id: "qwen3-coder-30b-a3b-colibri", capabilities: ["chat"], context_length: 65536 },
      { id: "gemma-4-e4b-colibri", capabilities: ["chat"] },
      { id: "odd-colibri", context_length: "big" },
      { id: "negative-colibri", capabilities: ["chat"], context_length: -1 },
    ] }), { headers: { "content-type": "application/json" } })))
    const models = await listModelInfo("http://x/v1", "")
    expect(models.map((model) => model.contextLength)).toEqual([65536, undefined, undefined, undefined])
  })
})
