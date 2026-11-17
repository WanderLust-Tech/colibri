import { afterEach, describe, expect, it, vi } from "vitest"

import { listRequests, setClientTags, streamChat, TAGS_HEADER } from "./api"
import { dashboardTags, newChatTag } from "./tags"

describe("the tags a conversation carries", () => {
  it("names a chat c- and six hex digits", () => {
    expect(newChatTag(() => 0)).toBe("c-000000")
    expect(newChatTag(() => 0.999)).toBe("c-ffffff")
    expect(newChatTag()).toMatch(/^c-[0-9a-f]{6}$/)
  })
  it("says the app and the chat", () => {
    expect(dashboardTags("c-1a2b3c")).toEqual({ app: "dashboard", chat: "c-1a2b3c" })
  })
})

describe("what goes out with a request", () => {
  afterEach(() => { setClientTags(null); vi.unstubAllGlobals() })

  const stream = () => new Response(new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode("data: [DONE]\n\n")); c.close() } }), { status: 200, headers: { "content-type": "text/event-stream" } })
  const options = { baseUrl: "http://h/v1", apiKey: "", model: "m", messages: [{ id: "1", role: "user" as const, content: "hi" }], temperature: 0, maxTokens: 8, enableThinking: false, signal: new AbortController().signal, onDelta: () => {} }

  it("puts the tags on a chat, as one JSON header", async () => {
    const fetched = vi.fn(async () => stream())
    vi.stubGlobal("fetch", fetched)
    setClientTags(dashboardTags("c-1a2b3c"))
    await streamChat(options)
    const headers = (fetched.mock.calls[0] as unknown as [string, RequestInit])[1].headers as Record<string, string>
    expect(JSON.parse(headers[TAGS_HEADER])).toEqual({ app: "dashboard", chat: "c-1a2b3c" })
  })

  it("sends none when there are none", async () => {
    const fetched = vi.fn(async () => stream())
    vi.stubGlobal("fetch", fetched)
    await streamChat(options)
    const headers = (fetched.mock.calls[0] as unknown as [string, RequestInit])[1].headers as Record<string, string>
    expect(headers[TAGS_HEADER]).toBeUndefined()
  })

  it("asks the request log for one tag, escaped", async () => {
    const fetched = vi.fn(async () => new Response(JSON.stringify({ enabled: true, path: "x", data: [] }), { status: 200 }))
    vi.stubGlobal("fetch", fetched)
    await listRequests("http://h/v1", "", 50, undefined, "chat:c-1a2b3c")
    expect((fetched.mock.calls[0] as unknown as [string])[0]).toBe("http://h/v1/requests?limit=50&tag=chat%3Ac-1a2b3c")
  })
})
