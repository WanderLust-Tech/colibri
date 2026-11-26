/* Streams as the router writes them, copied from what c/tests/test_router_tools.py asserts (LoopTest and ApprovalTest):
   ordinary chat chunks, and chunks with an empty `choices` and a `colibri` object between them. Used by the tests only. */

export type Chunk = Record<string, unknown>

const base = { id: "chatcmpl-0123456789abcdef01234567", object: "chat.completion.chunk", created: 1_700_000_000, model: "text-a-colibri" }

export const text = (content: string, role?: string): Chunk =>
  ({ ...base, choices: [{ index: 0, delta: { ...(role ? { role } : {}), content }, finish_reason: null }] })
export const finish = (reason = "stop"): Chunk => ({ ...base, choices: [{ index: 0, delta: {}, finish_reason: reason }] })
export const usage = (prompt = 20, completion = 10): Chunk =>
  ({ ...base, choices: [], usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion } })
export const colibri = (fields: Record<string, unknown>): Chunk => ({ ...base, choices: [], colibri: fields })

const NOTES = "C:\\work\\project\\notes.txt"
const MADE = "C:\\work\\scratch\\made.txt"

/* test_a_streamed_answer_carries_the_text_the_events_and_no_tool_calls_of_the_models_own */
export const READ_A_FILE: Chunk[] = [
  text("Let me look. ", "assistant"),
  colibri({ event: "tool_call", name: "files_read", arguments: { path: NOTES }, read_only: true, how: "read-only" }),
  colibri({ event: "tool_result", name: "files_read", ok: true, chars: 17, text: "the notes say hello" }),
  text("It says hello."),
  finish("stop"),
  usage(20, 10),
]

/* test_a_streamed_write_waits_for_a_yes_and_then_goes_on: approval, approval_answered, tool_call, tool_result */
export const WRITE_ALLOWED: Chunk[] = [
  colibri({ event: "approval", id: "a1b2c3d4e5f6", tool: "files_write", arguments: { path: MADE, content: "written by the model" },
    summary: `write 20 characters to ${MADE}`, expires_in: 120 }),
  colibri({ event: "approval_answered", id: "a1b2c3d4e5f6", tool: "files_write", allowed: true }),
  colibri({ event: "tool_call", name: "files_write", arguments: { path: MADE, content: "written by the model" }, read_only: false, how: "approved" }),
  colibri({ event: "tool_result", name: "files_write", ok: true, chars: 30, text: "Wrote 20 characters" }),
  text("I made the file.", "assistant"),
  finish("stop"),
]

/* test_a_no_goes_back_to_the_model_and_the_file_is_not_written: "declined: the call is shown, never run" */
export const WRITE_DECLINED: Chunk[] = [
  colibri({ event: "approval", id: "ffeeddccbbaa", tool: "files_write", arguments: { path: MADE, content: "x" }, summary: `write 1 characters to ${MADE}`, expires_in: 120 }),
  colibri({ event: "approval_answered", id: "ffeeddccbbaa", tool: "files_write", allowed: false }),
  colibri({ event: "tool_call", name: "files_write", arguments: { path: MADE, content: "x" }, read_only: false, how: "declined" }),
  text("Understood, I will not.", "assistant"),
  finish("stop"),
]

/* test_always_stops_the_questions_for_that_tool_in_this_conversation: the second call is not asked about */
export const WRITE_ALWAYS: Chunk[] = [
  colibri({ event: "approval", id: "112233445566", tool: "files_write", arguments: { path: MADE, content: "1" }, summary: "write 1 characters", expires_in: 120 }),
  colibri({ event: "approval_answered", id: "112233445566", tool: "files_write", allowed: true }),
  colibri({ event: "tool_call", name: "files_write", arguments: { path: MADE, content: "1" }, read_only: false, how: "approved" }),
  colibri({ event: "tool_result", name: "files_write", ok: true, chars: 5, text: "Wrote 1 characters" }),
  colibri({ event: "tool_call", name: "files_write", arguments: { path: MADE, content: "2" }, read_only: false, how: "approved for this session" }),
  colibri({ event: "tool_result", name: "files_write", ok: true, chars: 5, text: "Wrote 1 characters" }),
  text("both done", "assistant"),
  finish("stop"),
]

/* The router waits for a person with ": waiting" lines; the page must ignore them. */
export function toSse(chunks: Chunk[], { heartbeats = false, done = true }: { heartbeats?: boolean; done?: boolean } = {}): string {
  const frames: string[] = []
  for (const chunk of chunks) {
    if (heartbeats) frames.push(": waiting\n\n")
    frames.push(`data: ${JSON.stringify(chunk)}\n\n`)
  }
  if (done) frames.push("data: [DONE]\n\n")
  return frames.join("")
}
