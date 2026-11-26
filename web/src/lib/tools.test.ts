import { afterEach, describe, expect, it, vi } from "vitest"

import { answerApproval, listTools, streamChat, type ChatMessage } from "./api"
import {
  NO_TOOLS, addText, argumentsLine, argumentsText, clipText, enabledTools, foldToolEvent, hasTools, markSending, parseToolsListing,
  policyFolders, resolveApproval, secondsLeft, settleParts, toolEventOf,
  type ApprovalPart, type CallPart, type ToolEvent, type TurnPart,
} from "./tools"
import { answeredApproval, appendDelta, applyToolEvent, continuable, sendingApproval, settleTurn } from "./transcript"
import { READ_A_FILE, WRITE_ALLOWED, WRITE_ALWAYS, WRITE_DECLINED, colibri, finish, text, toSse, usage } from "./tools.fixtures"

afterEach(() => vi.unstubAllGlobals())

/* What GET /v1/tools answers (builtin_tools.listing + Policy.summary). */
const listing = {
  object: "list", path: "C:\\Users\\me\\tools-policy.json", problem: null,
  policy: { folders: [{ path: "C:\\work\\project", write: false }, { path: "C:\\work\\scratch", write: true }], web: true, allowSecrets: false, approvalSeconds: 120, disabled: [] },
  data: [
    { type: "function", function: { name: "web_fetch", description: "Fetch a page", parameters: { type: "object" } }, colibri: { group: "web", read_only: true, enabled: true } },
    { type: "function", function: { name: "files_read", description: "Read a file", parameters: { type: "object" } }, colibri: { group: "files", read_only: true, enabled: true } },
    { type: "function", function: { name: "files_write", description: "Write a file", parameters: { type: "object" } }, colibri: { group: "write", read_only: false, enabled: false } },
  ],
}

describe("the tools listing", () => {
  it("reads the tools, which are on, and the policy", () => {
    const parsed = parseToolsListing(listing)
    expect(parsed.available).toBe(true)
    expect(parsed.tools.map((tool) => [tool.name, tool.group, tool.readOnly, tool.enabled])).toEqual([
      ["web_fetch", "web", true, true], ["files_read", "files", true, true], ["files_write", "write", false, false],
    ])
    expect(enabledTools(parsed).map((tool) => tool.name)).toEqual(["web_fetch", "files_read"])
    expect(hasTools(parsed)).toBe(true)
    expect(policyFolders(parsed.policy!)).toEqual({ read: ["C:\\work\\project"], write: ["C:\\work\\scratch"] })
    expect(parsed.policy).toMatchObject({ web: true, allowSecrets: false, approvalSeconds: 120 })
  })

  it("offers nothing when no tool is on, or when the answer makes no sense", () => {
    const off = parseToolsListing({ ...listing, data: listing.data.map((row) => ({ ...row, colibri: { ...row.colibri, enabled: false } })) })
    expect(off.available).toBe(true)
    expect(hasTools(off)).toBe(false)
    expect(parseToolsListing("nonsense")).toEqual(NO_TOOLS)
    expect(parseToolsListing(null)).toEqual(NO_TOOLS)
    expect(hasTools(parseToolsListing({ data: [{ nope: 1 }, null, 3] }))).toBe(false)
  })

  it("carries the problem of a policy file that is not valid", () => {
    expect(parseToolsListing({ ...listing, problem: "not json", data: [] }).problem).toBe("not json")
  })

  it("treats 404 (an older router, or coli serve) as no tools and reads a listing otherwise", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}", { status: 404 })))
    await expect(listTools("http://localhost:8000/v1", "k")).resolves.toEqual(NO_TOOLS)
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(listing)))
    vi.stubGlobal("fetch", fetchMock)
    const read = await listTools("http://localhost:8000/v1/", "secret")
    expect(hasTools(read)).toBe(true)
    expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:8000/v1/tools")
    expect(fetchMock.mock.calls[0][1].headers).toMatchObject({ Authorization: "Bearer secret" })
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { message: "bad key" } }), { status: 401 })))
    await expect(listTools("http://localhost:8000/v1", "k")).rejects.toThrow("bad key")
  })
})

describe("events of the stream", () => {
  it("reads the four events as the router writes them", () => {
    const events = [...READ_A_FILE, ...WRITE_ALLOWED].flatMap((chunk) => {
      const event = chunk.colibri === undefined ? null : toolEventOf(chunk.colibri)
      return event ? [event.type] : []
    })
    expect(events).toEqual(["tool_call", "tool_result", "approval", "approval_answered", "tool_call", "tool_result"])
    expect(toolEventOf({ event: "approval", id: "x", tool: "files_write", arguments: {}, summary: "s", expires_in: 120 }))
      .toEqual({ type: "approval", id: "x", tool: "files_write", arguments: {}, summary: "s", expiresIn: 120 })
  })

  it("ignores what it does not know", () => {
    expect(toolEventOf({ event: "something_new" })).toBeNull()
    expect(toolEventOf({ event: "approval" })).toBeNull()
    expect(toolEventOf("x")).toBeNull()
    expect(toolEventOf(null)).toBeNull()
    expect(toolEventOf({ event: "tool_result", name: "web_fetch", ok: false, text: "Error: nope" })).toMatchObject({ ok: false, chars: 11 })
  })
})

const call = (over: Partial<ToolEvent & { type: "tool_call" }> = {}): ToolEvent =>
  ({ type: "tool_call", name: "files_read", arguments: { path: "a" }, readOnly: true, how: "read-only", ...over })
const ask = (id = "q1", expiresIn = 120): ToolEvent => ({ type: "approval", id, tool: "files_write", arguments: { path: "a" }, summary: "write 1 characters to a", expiresIn })

describe("folding events into the parts of a turn", () => {
  it("keeps text and calls in the order they happened", () => {
    let parts: TurnPart[] = addText([], "Let me look. ")
    parts = foldToolEvent(parts, call())
    expect(parts.map((part) => part.kind)).toEqual(["text", "call"])
    expect((parts[1] as CallPart).state).toBe("running")
    parts = foldToolEvent(parts, { type: "tool_result", name: "files_read", ok: true, chars: 19, text: "the notes say hello" })
    parts = addText(parts, "It says hello.")
    expect(parts).toEqual([
      { kind: "text", text: "Let me look. " },
      { kind: "call", name: "files_read", arguments: { path: "a" }, readOnly: true, how: "read-only", state: "ok", chars: 19, result: "the notes say hello" },
      { kind: "text", text: "It says hello." },
    ])
  })

  it("joins consecutive text and puts a failed result on the matching call", () => {
    let parts = addText(addText([], "a"), "b")
    expect(parts).toEqual([{ kind: "text", text: "ab" }])
    parts = foldToolEvent(parts, call({ name: "web_fetch" }))
    parts = foldToolEvent(parts, call({ name: "files_read" }))
    parts = foldToolEvent(parts, { type: "tool_result", name: "web_fetch", ok: false, chars: 11, text: "Error: nope" })
    expect(parts.filter((part): part is CallPart => part.kind === "call").map((part) => [part.name, part.state])).toEqual([["web_fetch", "error"], ["files_read", "running"]])
  })

  it("shows a declined call as declined at once: nothing runs, no result follows", () => {
    const parts = foldToolEvent([], call({ name: "files_write", readOnly: false, how: "declined" }))
    expect((parts[0] as CallPart).state).toBe("declined")
  })

  it("makes a card for a result whose call it never saw", () => {
    const parts = foldToolEvent([], { type: "tool_result", name: "git_log", ok: true, chars: 3, text: "abc" })
    expect(parts).toMatchObject([{ kind: "call", name: "git_log", state: "ok", result: "abc" }])
  })

  it("starts a question with a countdown on this page's clock and ends it when the router says", () => {
    let parts = foldToolEvent([], ask("q1", 120), 1_000_000)
    const question = parts[0] as ApprovalPart
    expect(question).toMatchObject({ kind: "approval", id: "q1", state: "waiting", expiresIn: 120, deadline: 1_120_000 })
    expect(secondsLeft(question.deadline, 1_000_000)).toBe(120)
    expect(secondsLeft(question.deadline, 1_119_001)).toBe(1)
    expect(secondsLeft(question.deadline, 2_000_000)).toBe(0)
    parts = foldToolEvent(parts, { type: "approval_answered", id: "q1", tool: "files_write", allowed: true })
    expect((parts[0] as ApprovalPart).state).toBe("allowed")
    parts = foldToolEvent(parts, { type: "approval_answered", id: "q1", tool: "files_write", allowed: false })
    expect((parts[0] as ApprovalPart).state).toBe("allowed")        // the first answer stands
    parts = foldToolEvent(parts, { type: "approval_answered", id: "other", tool: "files_write", allowed: false })
    expect((parts[0] as ApprovalPart).state).toBe("allowed")
  })

  it("plays the real streams from the Python tests into the right parts", () => {
    const play = (chunks: typeof READ_A_FILE) => {
      let parts: TurnPart[] = []
      for (const chunk of chunks) {
        const content = (chunk.choices as Array<{ delta: { content?: string } }>)[0]?.delta.content
        if (content) parts = addText(parts, content)
        const event = chunk.colibri === undefined ? null : toolEventOf(chunk.colibri)
        if (event) parts = foldToolEvent(parts, event, 0)
      }
      return parts.map((part) => part.kind === "text" ? `text:${part.text}` : part.kind === "call" ? `call:${part.name}:${part.state}` : `approval:${part.state}`)
    }
    expect(play(READ_A_FILE)).toEqual(["text:Let me look. ", "call:files_read:ok", "text:It says hello."])
    expect(play(WRITE_ALLOWED)).toEqual(["approval:allowed", "call:files_write:ok", "text:I made the file."])
    expect(play(WRITE_DECLINED)).toEqual(["approval:declined", "call:files_write:declined", "text:Understood, I will not."])
    expect(play(WRITE_ALWAYS)).toEqual(["approval:allowed", "call:files_write:ok", "call:files_write:ok", "text:both done"])
  })
})

describe("an approval that cannot be answered any more", () => {
  it("becomes inert when the stream ends without approval_answered", () => {
    const open = foldToolEvent(foldToolEvent([], call({ name: "web_fetch" })), ask())
    const settled = settleParts(open)
    expect(settled.map((part) => part.kind === "approval" || part.kind === "call" ? part.state : "text")).toEqual(["stopped", "stopped"])
    expect(settleParts(settled)).toBe(settled)
  })

  it("also settles one whose answer was on its way, and leaves finished ones alone", () => {
    let parts = foldToolEvent([], ask("q1"))
    parts = markSending(parts, "q1", false)
    expect((parts[0] as ApprovalPart).state).toBe("sending")
    expect((settleParts(parts)[0] as ApprovalPart).state).toBe("stopped")
    const done = foldToolEvent(foldToolEvent([], ask("q2")), { type: "approval_answered", id: "q2", tool: "files_write", allowed: false })
    expect(settleParts(done)).toBe(done)
  })

  it("is settled on the turn in the transcript when a stream ends, errors or is stopped", () => {
    const turn: ChatMessage = { id: "t", role: "assistant", content: "" }
    let messages = applyToolEvent([turn], "t", ask())
    expect((messages[0].parts![0] as ApprovalPart).state).toBe("waiting")
    messages = settleTurn(messages, "t")
    expect((messages[0].parts![0] as ApprovalPart).state).toBe("stopped")
    expect(settleTurn([{ id: "u", role: "user", content: "hi" }], "u")).toEqual([{ id: "u", role: "user", content: "hi" }])
  })
})

describe("answering an approval", () => {
  const waiting = () => foldToolEvent([], ask("q1"))

  it("sends id, allow and always, and reads 200 as answered", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: "answered", id: "q1", allow: true })))
    vi.stubGlobal("fetch", fetchMock)
    await expect(answerApproval("http://localhost:8000/v1", "secret", "q1", true, true)).resolves.toBe("answered")
    expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:8000/v1/tools/pending")
    expect(fetchMock.mock.calls[0][1].method).toBe("POST")
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ id: "q1", allow: true, always: true })
    await answerApproval("http://localhost:8000/v1", "", "q2", false, false)
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ id: "q2", allow: false })
  })

  it("reads 404 approval_not_found as gone, not as a failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { code: "approval_not_found", message: "No call is waiting" } }), { status: 404 })))
    await expect(answerApproval("http://localhost:8000/v1", "k", "q1", true, false)).resolves.toBe("gone")
  })

  it("throws for any other failure so the card can be answered again", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { message: "bad key" } }), { status: 401 })))
    await expect(answerApproval("http://localhost:8000/v1", "k", "q1", true, false)).rejects.toThrow("bad key")
  })

  it("keeps the buttons off while sending, then shows the outcome", () => {
    const sending = markSending(waiting(), "q1", true)
    expect(sending[0]).toMatchObject({ state: "sending", always: true })
    expect(markSending(sending, "q1", false)).toEqual(sending)             // no second answer while one is on its way
    expect(resolveApproval(sending, "q1", "answered", true)[0]).toMatchObject({ state: "allowed", always: true })
    expect(resolveApproval(sending, "q1", "answered", false)[0]).toMatchObject({ state: "declined" })
  })

  it("says the question expired on a 404, and offers it again after a failure", () => {
    const sending = markSending(waiting(), "q1", false)
    expect(resolveApproval(sending, "q1", "gone", true)[0]).toMatchObject({ state: "expired" })
    expect(resolveApproval(sending, "q1", "failed", true)[0]).toMatchObject({ state: "waiting", problem: "send-failed" })
  })

  it("lets the stream's own word stand when it arrived before the reply to our answer", () => {
    let parts = markSending(waiting(), "q1", false)
    parts = foldToolEvent(parts, { type: "approval_answered", id: "q1", tool: "files_write", allowed: false })   // timed out at the router
    expect((parts[0] as ApprovalPart).state).toBe("declined")
    expect(resolveApproval(parts, "q1", "gone", true)).toEqual(parts)
  })

  it("works on the turn in the transcript", () => {
    const turn: ChatMessage = { id: "t", role: "assistant", content: "" }
    let messages = applyToolEvent([turn], "t", ask("q1"))
    messages = sendingApproval(messages, "t", "q1", false)
    messages = answeredApproval(messages, "t", "q1", "gone", true)
    expect((messages[0].parts![0] as ApprovalPart).state).toBe("expired")
  })
})

describe("the turn in the transcript", () => {
  it("turns the text so far into the first part, and keeps content whole", () => {
    let messages: ChatMessage[] = [{ id: "t", role: "assistant", content: "" }]
    messages = appendDelta(messages, "t", "content", "Let me look. ")
    expect(messages[0].parts).toBeUndefined()                      // a turn without tools stays a plain turn
    messages = applyToolEvent(messages, "t", call())
    messages = applyToolEvent(messages, "t", { type: "tool_result", name: "files_read", ok: true, chars: 3, text: "abc" })
    messages = appendDelta(messages, "t", "content", "It says hello.")
    messages = appendDelta(messages, "t", "reasoning", "hmm")
    expect(messages[0].content).toBe("Let me look. It says hello.")
    expect(messages[0].parts!.map((part) => part.kind)).toEqual(["text", "call", "text"])
    expect(messages[0].reasoning).toBe("hmm")
  })

  it("does not offer Continue for a turn that has tool calls in it", () => {
    const plain: ChatMessage = { id: "a", role: "assistant", content: "cut off", finish: "length" }
    expect(continuable(plain)).toBe(true)
    expect(continuable({ ...plain, parts: foldToolEvent([], call()) })).toBe(false)
  })
})

describe("a whole stream through streamChat", () => {
  const request = (chunks: Parameters<typeof toSse>[0], over: Partial<Parameters<typeof streamChat>[0]> = {}, sse: Parameters<typeof toSse>[1] = {}) => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(toSse(chunks, sse), { headers: { "content-type": "text/event-stream" } }))
    vi.stubGlobal("fetch", fetchMock)
    let messages: ChatMessage[] = [{ id: "t", role: "assistant", content: "" }]
    const events: ToolEvent[] = []
    const result = streamChat({
      baseUrl: "http://localhost:8000/v1", apiKey: "", model: "text-a-colibri", messages: [{ id: "u", role: "user", content: "hi" }],
      temperature: 0, maxTokens: 8, enableThinking: false, tools: true, signal: new AbortController().signal,
      onDelta: (delta) => { messages = appendDelta(messages, "t", "content", delta) },
      onTool: (event) => { events.push(event); messages = applyToolEvent(messages, "t", event, 0) },
      ...over,
    })
    return { fetchMock, result, events, messages: () => messages }
  }

  it("sends colibri_tools and no tools of its own, and nothing when tools are off", async () => {
    const on = request([text("hi", "assistant"), finish()])
    await on.result
    const body = JSON.parse(on.fetchMock.mock.calls[0][1].body as string) as Record<string, unknown>
    expect(body.colibri_tools).toBe(true)
    expect(body).not.toHaveProperty("tools")
    expect(body).not.toHaveProperty("tool_choice")
    const off = request([text("hi", "assistant"), finish()], { tools: false })
    await off.result
    expect(JSON.parse(off.fetchMock.mock.calls[0][1].body as string)).not.toHaveProperty("colibri_tools")
  })

  it("lays a read out in order and takes the usage summed over the rounds from the last chunk", async () => {
    const run = request(READ_A_FILE)
    const result = await run.result
    expect(run.messages()[0].content).toBe("Let me look. It says hello.")
    expect(run.messages()[0].parts!.map((part) => part.kind === "text" ? part.text : `[${part.kind === "call" ? part.name + ":" + part.state : part.kind}]`))
      .toEqual(["Let me look. ", "[files_read:ok]", "It says hello."])
    expect(result.finishReason).toBe("stop")
    expect(result.usage).toEqual({ prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 })
    expect(result.model).toBe("text-a-colibri")
  })

  it("ignores the router's `: waiting` lines while it waits for a person", async () => {
    const run = request(WRITE_ALLOWED, {}, { heartbeats: true })
    await run.result
    expect(run.events.map((event) => event.type)).toEqual(["approval", "approval_answered", "tool_call", "tool_result"])
    expect(run.messages()[0].content).toBe("I made the file.")
  })

  it("leaves a question that was never answered inert when the stream ends there", async () => {
    const run = request([WRITE_ALLOWED[0]], {}, { done: false })
    await run.result
    expect((run.messages()[0].parts![0] as ApprovalPart).state).toBe("waiting")     // what the stream said
    const settled = settleTurn(run.messages(), "t")                                  // what the page does when it ends
    expect((settled[0].parts![0] as ApprovalPart).state).toBe("stopped")
  })

  it("passes an error frame in the middle of the stream on", async () => {
    const run = request([colibri({ event: "tool_call", name: "files_read", arguments: {}, read_only: true, how: "read-only" }),
      { error: { message: "the model's stream was cut", type: "server_error" } }])
    await expect(run.result).rejects.toThrow("the model's stream was cut")
  })

  it("reads a stream with no tools in it as before", async () => {
    const run = request([text("plain", "assistant"), finish(), usage(3, 4)], { tools: false })
    const result = await run.result
    expect(run.messages()[0]).toMatchObject({ content: "plain" })
    expect(run.messages()[0].parts).toBeUndefined()
    expect(result.usage?.total_tokens).toBe(7)
  })
})

describe("what is shown of a call", () => {
  it("makes the arguments one short line, cutting long values", () => {
    expect(argumentsLine({ path: "C:/work/notes.txt", start_line: 1 })).toBe("path: C:/work/notes.txt, start_line: 1")
    const long = argumentsLine({ path: "a", content: "x".repeat(500) })
    expect(long.length).toBeLessThanOrEqual(141)
    expect(argumentsLine({ text: "two\nlines\t here" })).toBe("text: two lines here")
    expect(argumentsLine(undefined)).toBe("")
    expect(argumentsLine("odd")).toBe("odd")
  })

  it("clips what it prints in full", () => {
    expect(clipText("abcdef", 3)).toBe("abc…")
    expect(argumentsText({ a: 1 })).toBe('{\n  "a": 1\n}')
    expect(argumentsText({ content: "y".repeat(9000) }, 100).length).toBe(101)
    expect(argumentsText(undefined)).toBe("")
  })
})
