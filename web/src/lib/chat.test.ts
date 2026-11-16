import { afterEach, describe, expect, it, vi } from "vitest"

import { streamChat, type ChatMessage } from "./api"
import { branchAt, editFrom, isPlainTurn, REWRITES, resendFrom } from "./chat"

afterEach(() => vi.unstubAllGlobals())

const user = (content: string, images?: string[]): ChatMessage =>
  images?.length
    ? { id: "u", role: "user", content, images }
    : { id: "u", role: "user", content }

const assistant = (content: string): ChatMessage => ({ id: "a", role: "assistant", content })

describe("resendFrom", () => {
  it("keeps the pictures that arrived with the last user turn", () => {
    const photo = "data:image/png;base64,aaa"
    const messages = [user("what is this?", [photo]), assistant("a cat")]
    expect(resendFrom(messages, 1)).toEqual({
      text: "what is this?",
      previous: [],
      pictures: [photo],
    })
  })

  it("lets an image-only turn be regenerated", () => {
    const photo = "data:image/jpeg;base64,bbb"
    const messages = [user("", [photo]), assistant("a diagram")]
    const retry = resendFrom(messages, 1)
    expect(retry).not.toBeNull()
    expect(retry!.text).toBe("")
    expect(retry!.pictures).toEqual([photo])
  })

  it("uses the nearest user turn and leaves earlier history alone", () => {
    const first: ChatMessage = { id: "u0", role: "user", content: "hi" }
    const reply: ChatMessage = { id: "a0", role: "assistant", content: "hello" }
    const photo = "data:image/png;base64,ccc"
    const second: ChatMessage = { id: "u1", role: "user", content: "and this?", images: [photo] }
    const last: ChatMessage = { id: "a1", role: "assistant", content: "a bird" }
    const messages = [first, reply, second, last]
    expect(resendFrom(messages, 3)).toEqual({
      text: "and this?",
      previous: [first, reply],
      pictures: [photo],
    })
  })

  it("returns null when there is no user turn to resend", () => {
    expect(resendFrom([assistant("orphan")], 0)).toBeNull()
  })

  it("leaves a text-only turn without a pictures field on the wire", () => {
    expect(resendFrom([user("hello"), assistant("hi")], 1)).toEqual({
      text: "hello",
      previous: [],
      pictures: [],
    })
  })
})

describe("regenerate request body", () => {
  it("puts those pictures on the wire as image_url parts", async () => {
    const photo = "data:image/png;base64,ddd"
    const retry = resendFrom([user("describe this", [photo]), assistant("skipped")], 1)
    const fetchMock = vi.fn().mockResolvedValue(new Response("data: [DONE]\n\n", {
      headers: { "content-type": "text/event-stream" },
    }))
    vi.stubGlobal("fetch", fetchMock)
    await streamChat({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      model: "test-model",
      messages: [{ id: "u", role: "user", content: retry!.text, images: retry!.pictures }],
      temperature: 0,
      maxTokens: 8,
      enableThinking: false,
      signal: new AbortController().signal,
      onDelta: () => undefined,
    })
    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string) as {
      messages: Array<{ content: unknown }>
    }
    expect(body.messages[0].content).toEqual([
      { type: "text", text: "describe this" },
      { type: "image_url", image_url: { url: photo } },
    ])
  })

  it("sends an image-only regenerate as image parts with no empty text", async () => {
    const photo = "data:image/png;base64,eee"
    const retry = resendFrom([user("", [photo]), assistant("skipped")], 1)
    const fetchMock = vi.fn().mockResolvedValue(new Response("data: [DONE]\n\n", {
      headers: { "content-type": "text/event-stream" },
    }))
    vi.stubGlobal("fetch", fetchMock)
    await streamChat({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      model: "test-model",
      messages: [{ id: "u", role: "user", content: retry!.text, images: retry!.pictures }],
      temperature: 0,
      maxTokens: 8,
      enableThinking: false,
      signal: new AbortController().signal,
      onDelta: () => undefined,
    })
    const body = JSON.parse(fetchMock.mock.calls[0][1].body as string) as {
      messages: Array<{ content: unknown }>
    }
    expect(body.messages[0].content).toEqual([
      { type: "image_url", image_url: { url: photo } },
    ])
  })
})

describe("editFrom", () => {
  const photo = "data:image/png;base64,ccc"
  const chat: ChatMessage[] = [user("first"), assistant("one"), user("second", [photo]), assistant("two")]

  it("cuts the conversation at the message and hands back its text and pictures", () => {
    const turn = editFrom(chat, 2)
    expect(turn).toEqual({ text: "second", pictures: [photo], previous: chat.slice(0, 2) })
  })

  it("can start the conversation over from its first message", () => {
    expect(editFrom(chat, 0)).toEqual({ text: "first", pictures: [], previous: [] })
  })

  it("does not touch the original array, and refuses what is not a plain user turn", () => {
    editFrom(chat, 2)
    expect(chat).toHaveLength(4)
    expect(editFrom(chat, 1)).toBeNull()
    expect(editFrom(chat, 9)).toBeNull()
    expect(editFrom([{ id: "i", role: "user", content: "a cat", mode: "image" }], 0)).toBeNull()
  })
})

describe("branchAt", () => {
  const chat: ChatMessage[] = [user("first"), assistant("one"), user("second"), assistant("two")]

  it("keeps the reply and everything before it", () => {
    expect(branchAt(chat, 1)).toEqual(chat.slice(0, 2))
    expect(branchAt(chat, 3)).toEqual(chat)
  })

  it("is null for a user message, an empty reply or an image turn", () => {
    expect(branchAt(chat, 0)).toBeNull()
    expect(branchAt([user("q"), assistant("  ")], 1)).toBeNull()
    expect(branchAt([user("q"), { id: "g", role: "assistant", content: "", mode: "image" }], 1)).toBeNull()
  })
})

describe("one-click rewrites", () => {
  it("name dictionary keys, so the wording follows the language", () => {
    expect(REWRITES.shorter).toBe("rewrite.shorterPrompt")
    expect(REWRITES.simpler).toBe("rewrite.simplerPrompt")
  })

  it("plain means no image or music mode", () => {
    expect(isPlainTurn(assistant("x"))).toBe(true)
    expect(isPlainTurn({ id: "m", role: "assistant", content: "", mode: "music" })).toBe(false)
  })
})
