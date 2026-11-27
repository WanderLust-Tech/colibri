/* The router's built-in tools (docs/builtin-tools.md): fetch a web page, read files and git, change files after a person says yes.
   They are off until the owner turns them on, on the router's PC; this page can only use what is on, never change the policy.

   Everything here is plain data in, plain data out, so the stream handling can be tested without React: a tool listing is
   parsed, the `colibri` objects of a chat stream are turned into events, and the events are folded into the parts of one
   assistant turn (text, tool calls, approvals) in the order they happened. Tool output is only ever text; nothing in this file
   builds markup. */

export interface ToolInfo {
  name: string
  description: string
  group: string
  readOnly: boolean
  enabled: boolean
}

export interface ToolsPolicy {
  folders: Array<{ path: string; write: boolean }>
  web: boolean
  allowSecrets: boolean
  approvalSeconds: number | null
  disabled: string[]
}

/* GET /v1/tools. `available` is false where there is nothing to ask: a router older than the feature, or a plain `coli serve`
   (both answer 404). */
export interface ToolsListing {
  available: boolean
  tools: ToolInfo[]
  policy: ToolsPolicy | null
  problem: string | null
  /* Where the policy file is, on the router's PC. */
  path: string | null
}

export const NO_TOOLS: ToolsListing = { available: false, tools: [], policy: null, problem: null, path: null }

/* One change to the policy, as POST /v1/tools/policy takes it. `allow` adds a folder or replaces its entry, which is how a
   folder switches between read-only and write; `deny` removes it; `web` turns web fetch on or off. */
export type ToolsPolicyEdit =
  | { action: "allow"; folder: string; write: boolean }
  | { action: "deny"; folder: string }
  | { action: "web"; on: boolean }

const isObject = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value)
const text = (value: unknown, fallback = "") => typeof value === "string" ? value : fallback

/* The `policy` object of GET /v1/tools and of the answer to an edit; null for anything that is not one. */
export function parsePolicy(raw: unknown): ToolsPolicy | null {
  if (!isObject(raw)) return null
  return {
    folders: (Array.isArray(raw.folders) ? raw.folders : []).flatMap((folder) =>
      isObject(folder) && typeof folder.path === "string" && folder.path ? [{ path: folder.path, write: folder.write === true }] : []),
    web: raw.web === true,
    allowSecrets: raw.allowSecrets === true,
    approvalSeconds: typeof raw.approvalSeconds === "number" && Number.isFinite(raw.approvalSeconds) ? raw.approvalSeconds : null,
    disabled: (Array.isArray(raw.disabled) ? raw.disabled : []).filter((name): name is string => typeof name === "string"),
  }
}

export function parseToolsListing(body: unknown): ToolsListing {
  if (!isObject(body)) return NO_TOOLS
  const tools: ToolInfo[] = []
  for (const row of Array.isArray(body.data) ? body.data : []) {
    if (!isObject(row)) continue
    const fn = isObject(row.function) ? row.function : {}
    const info = isObject(row.colibri) ? row.colibri : {}
    const name = text(fn.name)
    if (!name) continue
    tools.push({ name, description: text(fn.description), group: text(info.group), readOnly: info.read_only === true, enabled: info.enabled === true })
  }
  const policy = parsePolicy(body.policy)
  return { available: true, tools, policy, problem: typeof body.problem === "string" && body.problem ? body.problem : null,
    path: typeof body.path === "string" && body.path ? body.path : null }
}

/* The tools the router will run: the switch in the chat is offered only when there is at least one. */
export function enabledTools(listing: ToolsListing): ToolInfo[] {
  return listing.tools.filter((tool) => tool.enabled)
}

export function hasTools(listing: ToolsListing): boolean {
  return listing.available && enabledTools(listing).length > 0
}

/* ---- the events of a chat stream */

export type ToolEvent =
  | { type: "tool_call"; name: string; arguments: unknown; readOnly: boolean; how: string }
  | { type: "tool_result"; name: string; ok: boolean; chars: number; text: string }
  | { type: "approval"; id: string; tool: string; arguments: unknown; summary: string; expiresIn: number }
  | { type: "approval_answered"; id: string; tool: string; allowed: boolean }

/* The `colibri` object of a stream chunk (the chunk's `choices` is empty), or null for anything this page does not know:
   a client that ignores unknown fields sees an ordinary stream, and so does this one. */
export function toolEventOf(colibri: unknown): ToolEvent | null {
  if (!isObject(colibri)) return null
  switch (colibri.event) {
    case "tool_call":
      return { type: "tool_call", name: text(colibri.name, "?"), arguments: colibri.arguments, readOnly: colibri.read_only === true, how: text(colibri.how) }
    case "tool_result":
      return { type: "tool_result", name: text(colibri.name, "?"), ok: colibri.ok === true,
        chars: typeof colibri.chars === "number" ? colibri.chars : text(colibri.text).length, text: text(colibri.text) }
    case "approval": {
      const id = text(colibri.id)
      if (!id) return null
      const seconds = typeof colibri.expires_in === "number" && Number.isFinite(colibri.expires_in) ? Math.max(0, colibri.expires_in) : 0
      return { type: "approval", id, tool: text(colibri.tool, "?"), arguments: colibri.arguments, summary: text(colibri.summary), expiresIn: seconds }
    }
    case "approval_answered": {
      const id = text(colibri.id)
      return id ? { type: "approval_answered", id, tool: text(colibri.tool, "?"), allowed: colibri.allowed === true } : null
    }
    default:
      return null
  }
}

/* ---- the parts of one assistant turn */

/* "stopped": the reply ended (finished, failed or was stopped) before this call came back. */
export type CallState = "running" | "ok" | "error" | "declined" | "stopped"
/* "waiting" is the only live state. "sending": our answer is on its way. "expired": the router no longer knew the question
   (answered already, or it timed out). "stopped": the reply ended with the question open, so it can never be answered. */
export type ApprovalState = "waiting" | "sending" | "allowed" | "declined" | "expired" | "stopped"

export interface TextPart { kind: "text"; text: string }
export interface CallPart {
  kind: "call"
  name: string
  arguments: unknown
  readOnly: boolean
  how: string
  state: CallState
  chars?: number
  /* The result as text, already clipped by the router. Shown in a <pre>, never as markup. */
  result?: string
}
export interface ApprovalPart {
  kind: "approval"
  id: string
  tool: string
  arguments: unknown
  summary: string
  expiresIn: number
  /* When the question lapses, on this page's clock (the moment it arrived plus `expiresIn`). */
  deadline: number
  state: ApprovalState
  /* Set when the answer was "always" (for the rest of this one request). */
  always?: boolean
  /* Why a state the person did not choose: the answer did not get through. */
  problem?: "send-failed"
}
export type TurnPart = TextPart | CallPart | ApprovalPart

export function addText(parts: TurnPart[], delta: string): TurnPart[] {
  if (!delta) return parts
  const last = parts[parts.length - 1]
  if (last?.kind === "text") return [...parts.slice(0, -1), { kind: "text", text: last.text + delta }]
  return [...parts, { kind: "text", text: delta }]
}

const LIVE_APPROVAL: ApprovalState[] = ["waiting", "sending"]

export function foldToolEvent(parts: TurnPart[], event: ToolEvent, now = Date.now()): TurnPart[] {
  switch (event.type) {
    case "tool_call": {
      const call: CallPart = { kind: "call", name: event.name, arguments: event.arguments, readOnly: event.readOnly, how: event.how,
        /* A declined call is shown and never run: no result will follow. */
        state: event.how === "declined" ? "declined" : "running" }
      return [...parts, call]
    }
    case "tool_result": {
      let at = -1
      for (let i = parts.length - 1; i >= 0; i--) {
        const part = parts[i]
        if (part.kind === "call" && part.state === "running" && part.name === event.name) { at = i; break }
      }
      const state: CallState = event.ok ? "ok" : "error"
      if (at < 0) return [...parts, { kind: "call", name: event.name, arguments: undefined, readOnly: true, how: "", state, chars: event.chars, result: event.text }]
      return parts.map((part, i) => i === at && part.kind === "call" ? { ...part, state, chars: event.chars, result: event.text } : part)
    }
    case "approval":
      return [...parts, { kind: "approval", id: event.id, tool: event.tool, arguments: event.arguments, summary: event.summary,
        expiresIn: event.expiresIn, deadline: now + event.expiresIn * 1000, state: "waiting" }]
    case "approval_answered":
      return parts.map((part) => part.kind === "approval" && part.id === event.id && LIVE_APPROVAL.includes(part.state)
        ? { ...part, state: event.allowed ? "allowed" : "declined", problem: undefined } : part)
  }
}

/* The reply is over (the stream ended, broke, or Stop was pressed): whatever was still open can never be answered or finish.
   No part is left in a state that offers a button. */
export function settleParts(parts: TurnPart[]): TurnPart[] {
  if (!parts.some((part) => (part.kind === "approval" && LIVE_APPROVAL.includes(part.state)) || (part.kind === "call" && part.state === "running"))) return parts
  return parts.map((part) => {
    if (part.kind === "approval" && LIVE_APPROVAL.includes(part.state)) return { ...part, state: "stopped" as const }
    if (part.kind === "call" && part.state === "running") return { ...part, state: "stopped" as const }
    return part
  })
}

/* ---- answering an approval (POST /v1/tools/pending) */

/* "answered": 200. "gone": 404, the router no longer has that question (answered already, or timed out). */
export type AnswerResult = "answered" | "gone" | "failed"

/* Our own answer, taking effect when the router said yes (or no) or has forgotten the question. Only a question that is
   still "sending" changes: the stream may already have said how it ended, and that wins. */
export function resolveApproval(parts: TurnPart[], id: string, result: AnswerResult, allow: boolean): TurnPart[] {
  return parts.map((part) => {
    if (part.kind !== "approval" || part.id !== id || part.state !== "sending") return part
    if (result === "answered") return { ...part, state: allow ? "allowed" : "declined", problem: undefined }
    if (result === "gone") return { ...part, state: "expired", problem: undefined }
    return { ...part, state: "waiting", always: undefined, problem: "send-failed" }
  })
}

export function markSending(parts: TurnPart[], id: string, always: boolean): TurnPart[] {
  return parts.map((part) => part.kind === "approval" && part.id === id && part.state === "waiting"
    ? { ...part, state: "sending", always: always || undefined, problem: undefined } : part)
}

/* ---- what is shown */

export function clipText(value: string, limit: number): string {
  return value.length <= limit ? value : value.slice(0, limit) + "…"
}

/* The arguments of a call in one short line: `path: C:/work/notes.txt, start_line: 1`. A long value (the content of a file
   about to be written) is cut. */
export function argumentsLine(args: unknown, limit = 140): string {
  if (args === undefined || args === null) return ""
  if (!isObject(args)) return clipText(typeof args === "string" ? args : JSON.stringify(args) ?? "", limit)
  const pieces = Object.entries(args).map(([key, value]) => {
    const shown = typeof value === "string" ? value.replace(/\s+/g, " ") : JSON.stringify(value)
    return `${key}: ${clipText(shown ?? "", 60)}`
  })
  return clipText(pieces.join(", "), limit)
}

/* The whole arguments, for the details of a call or an approval, clipped. */
export function argumentsText(args: unknown, limit = 4000): string {
  if (args === undefined) return ""
  try { return clipText(JSON.stringify(args, null, 2) ?? "", limit) } catch { return "" }
}

/* Seconds left before a question lapses, never below 0. */
export function secondsLeft(deadline: number, now: number): number {
  return Math.max(0, Math.ceil((deadline - now) / 1000))
}

/* Folders the policy lets the tools read, and the ones it lets them change (those are readable too). */
export function policyFolders(policy: ToolsPolicy): { read: string[]; write: string[] } {
  return { read: policy.folders.filter((folder) => !folder.write).map((folder) => folder.path), write: policy.folders.filter((folder) => folder.write).map((folder) => folder.path) }
}
