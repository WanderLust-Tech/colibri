import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"

import { ToolTurn } from "./ToolParts"
import { foldToolEvent, type ToolEvent, type TurnPart } from "@/lib/tools"

/* Static rendering, no DOM: the same approach as Markdown.test.tsx. Without a provider `t` answers with the key, which is
   enough to see which strings and buttons are drawn. */
const noop = () => undefined
const render = (parts: TurnPart[], live = true) => renderToStaticMarkup(<ToolTurn parts={parts} live={live} onAnswer={noop} />)

const fold = (...events: ToolEvent[]) => events.reduce<TurnPart[]>((parts, event) => foldToolEvent(parts, event, Date.now()), [])
const call: ToolEvent = { type: "tool_call", name: "web_fetch", arguments: { url: "https://example.test/" }, readOnly: true, how: "read-only" }
const ask: ToolEvent = { type: "approval", id: "q1", tool: "files_write", arguments: { path: "C:\\scratch\\a.txt", content: "hello" }, summary: "write 5 characters to C:\\scratch\\a.txt", expiresIn: 120 }

describe("a tool call card", () => {
  it("shows the name, the short arguments and the state, with the result collapsed", () => {
    const html = render(fold(call, { type: "tool_result", name: "web_fetch", ok: true, chars: 11, text: "page text" }))
    expect(html).toContain("web_fetch")
    expect(html).toContain("url: https://example.test/")
    expect(html).toContain("tools.state.ok")
    expect(html).toContain("<details")
    expect(html).not.toContain("<details open")
    expect(html).toContain("page text")
  })

  it("never turns tool output into markup, whatever it contains", () => {
    const hostile = '<script>alert(1)</script><img src=x onerror="alert(2)"> **bold** [x](javascript:alert(3)) <a href="http://evil">link</a>'
    const html = render(fold(
      { ...call, arguments: { url: hostile } },
      { type: "tool_result", name: "web_fetch", ok: true, chars: hostile.length, text: hostile },
    ))
    expect(html).not.toContain("<script")
    expect(html).not.toContain("<img")
    expect(html).not.toContain("<a ")
    expect(html).not.toContain("<strong>bold")
    expect(html).toContain("&lt;script&gt;alert(1)&lt;/script&gt;")
    expect(html).toContain("&lt;img src=x")
    expect(html).toContain("**bold**")                                   // shown as typed, not as markdown
    expect(html).toContain("<pre")
  })

  it("keeps the model's own words as before: markdown is for the model's text only", () => {
    const html = render([{ kind: "text", text: "It **says** hello." }])
    expect(html).toContain("<strong>says</strong>")
  })

  it("clips a very long result", () => {
    const html = render(fold(call, { type: "tool_result", name: "web_fetch", ok: true, chars: 50_000, text: "z".repeat(50_000) }))
    expect(html.length).toBeLessThan(10_000)
  })

  it("shows a failed call as failed and a declined one as declined", () => {
    expect(render(fold(call, { type: "tool_result", name: "web_fetch", ok: false, chars: 7, text: "Error: x" }))).toContain("tools.state.error")
    expect(render(fold({ ...call, how: "declined" }))).toContain("tools.state.declined")
  })
})

describe("an approval card", () => {
  const buttons = (html: string) => (html.match(/<button/g) ?? []).length

  it("shows the summary, the tool, the countdown and three buttons while the reply is streaming", () => {
    const html = render(fold(ask))
    expect(html).toContain("write 5 characters to C:\\scratch\\a.txt")
    expect(html).toContain("files_write")
    expect(html).toContain("tools.countdown")
    expect(buttons(html)).toBe(3)
    expect(html).toContain("tools.allow")
    expect(html).toContain("tools.allowAlways")
    expect(html).toContain("tools.decline")
    expect(html).not.toContain(`disabled=""`)
  })

  it("disables the buttons while the answer is on its way", () => {
    const sending = fold(ask).map((part) => part.kind === "approval" ? { ...part, state: "sending" as const } : part)
    const html = render(sending)
    expect(buttons(html)).toBe(3)
    expect((html.match(/disabled=""/g) ?? []).length).toBe(3)
  })

  it("is inert when the reply is over: no button, and it says it was not answered", () => {
    const html = render(fold(ask), false)
    expect(buttons(html)).toBe(0)
    expect(html).toContain("tools.approval.stopped")
  })

  it("is inert once it was answered, and says how", () => {
    const allowed = render(fold(ask, { type: "approval_answered", id: "q1", tool: "files_write", allowed: true }))
    expect(buttons(allowed)).toBe(0)
    expect(allowed).toContain("tools.approval.allowed")
    const declined = render(fold(ask, { type: "approval_answered", id: "q1", tool: "files_write", allowed: false }))
    expect(buttons(declined)).toBe(0)
    expect(declined).toContain("tools.approval.declined")
  })

  it("says it expired when the router no longer had the question", () => {
    const expired = fold(ask).map((part) => part.kind === "approval" ? { ...part, state: "expired" as const } : part)
    const html = render(expired)
    expect(buttons(html)).toBe(0)
    expect(html).toContain("tools.approval.expired")
  })

  it("does not render the model's summary as markup", () => {
    const html = render(fold({ ...ask, summary: "<script>x</script>" }))
    expect(html).not.toContain("<script>")
    expect(html).toContain("&lt;script&gt;")
  })
})
