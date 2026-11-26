import type { ChatMessage } from "./api"
import type { ReplyStats } from "./replyStats"
import { addText, foldToolEvent, markSending, resolveApproval, settleParts, type AnswerResult, type ToolEvent } from "./tools"

/* Stream a delta into one bubble of the transcript. Both a fresh answer and a
   continued turn go through here; the only difference is which bubble. */
export function appendDelta(messages: ChatMessage[], targetId: string, field: "content" | "reasoning", delta: string): ChatMessage[] {
  return messages.map((item) => {
    if (item.id !== targetId) return item
    const grown = { ...item, [field]: (item[field] ?? "") + delta }
    /* A turn that has used tools keeps its text in order with the calls; `content` still holds all of it. */
    return field === "content" && item.parts ? { ...grown, parts: addText(item.parts, delta) } : grown
  })
}

const edit = (messages: ChatMessage[], targetId: string, change: (item: ChatMessage) => ChatMessage): ChatMessage[] =>
  messages.map((item) => item.id === targetId ? change(item) : item)

/* One thing the router's tools did, into the turn being streamed. The first one turns the text so far into the first part. */
export function applyToolEvent(messages: ChatMessage[], targetId: string, event: ToolEvent, now = Date.now()): ChatMessage[] {
  return edit(messages, targetId, (item) => {
    const before = item.parts ?? (item.content ? addText([], item.content) : [])
    return { ...item, parts: foldToolEvent(before, event, now) }
  })
}

/* The reply is over, however it ended: no call stays "running" and no question stays open. */
export function settleTurn(messages: ChatMessage[], targetId: string): ChatMessage[] {
  return edit(messages, targetId, (item) => item.parts ? { ...item, parts: settleParts(item.parts) } : item)
}

export function sendingApproval(messages: ChatMessage[], targetId: string, approvalId: string, always: boolean): ChatMessage[] {
  return edit(messages, targetId, (item) => item.parts ? { ...item, parts: markSending(item.parts, approvalId, always) } : item)
}

export function answeredApproval(messages: ChatMessage[], targetId: string, approvalId: string, result: AnswerResult, allow: boolean): ChatMessage[] {
  return edit(messages, targetId, (item) => item.parts ? { ...item, parts: resolveApproval(item.parts, approvalId, result, allow) } : item)
}

/* Record how the generation streaming into `targetId` ended. A null finish is
   a stream that closed without a finish_reason: colibri always sends one last,
   so the reply was cut off, and is recorded as "incomplete", not as done. */
export function setFinish(messages: ChatMessage[], targetId: string, finish: string | null): ChatMessage[] {
  return messages.map((item) => item.id === targetId ? { ...item, finish: finish ?? "incomplete" } : item)
}

/* Attach a finished reply's stats (tokens, speed, cache, model) to its bubble. Nothing to attach leaves the bubble
   as it was; a continued turn gets the figures of its latest request, replacing the earlier ones. */
export function setStats(messages: ChatMessage[], targetId: string, stats: ReplyStats | null): ChatMessage[] {
  if (!stats) return messages
  return messages.map((item) => item.id === targetId ? { ...item, stats } : item)
}

/* An assistant turn is open, and worth continuing, when it hit max_tokens or
   was cut off by a stop, an error, or a stream that closed without saying how
   it finished. "stop" means the model closed the turn,
   and continuing it only makes the model re-emit its stop and add nothing. */
export function continuable(message: ChatMessage): boolean {
  /* A turn with tool calls in it is not continued: what would follow is a new answer, not the rest of this text. */
  return message.role === "assistant" && !!message.content.trim() && !message.parts?.length
    && ["length", "aborted", "error", "incomplete"].includes(message.finish ?? "")
}

/* The request that continues the trailing assistant turn instead of opening a
   new one: the transcript ending on that turn, streamed back into the same
   bubble. Trailing whitespace is stripped because the server refuses it (the
   template strips it, so the model would resume from different bytes than were
   sent); the returned history carries those exact bytes, so the bubble shows
   what it resumes from. null when there is no non-empty assistant turn last. */
export function continuation(messages: ChatMessage[]): { history: ChatMessage[]; targetId: string } | null {
  const last = messages[messages.length - 1]
  if (!last || last.role !== "assistant" || !last.content.trim()) return null
  const trimmed = last.content.replace(/\s+$/, "")
  return {
    history: messages.map((item) => item.id === last.id ? { ...item, content: trimmed } : item),
    targetId: last.id,
  }
}
