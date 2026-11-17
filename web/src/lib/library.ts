/* The pictures and songs the server has kept (c/library.py, GET /v1/library). */
import type { MediaSettings } from "./mediaMeta"
import { serverRoot } from "./jobs"
import { parseRecord } from "./mediaMeta"

export type LibraryKind = "images" | "music"

export interface LibraryEntry {
  kind: LibraryKind
  name: string
  url: string
  bytes: number
  modified: number
  width?: number
  height?: number
  duration?: number
  /* What the server wrote into the file when it made it: the prompt, the seed, the model. A file from somewhere else has none. */
  settings?: MediaSettings
}

export interface LibraryPage {
  entries: LibraryEntry[]
  total: number
  hasMore: boolean
}

function entryOf(value: unknown): LibraryEntry | null {
  if (typeof value !== "object" || value === null) return null
  const found = value as Record<string, unknown>
  if ((found.kind !== "images" && found.kind !== "music") || typeof found.name !== "string" || typeof found.url !== "string") return null
  const entry: LibraryEntry = {
    kind: found.kind, name: found.name, url: found.url,
    bytes: typeof found.bytes === "number" ? found.bytes : 0,
    modified: typeof found.modified === "number" ? found.modified : 0,
  }
  for (const key of ["width", "height", "duration"] as const) {
    const item = found[key]
    if (typeof item === "number" && Number.isFinite(item)) entry[key] = item
  }
  if (typeof found.settings === "object" && found.settings !== null) {
    const settings = parseRecord(JSON.stringify(found.settings))
    if (settings) entry.settings = settings
  }
  return entry
}

export async function listLibrary(baseUrl: string, apiKey: string, query: { kind?: LibraryKind; limit?: number; offset?: number } = {}, signal?: AbortSignal): Promise<LibraryPage> {
  const parts = Object.entries(query).filter(([, value]) => value !== undefined).map(([key, value]) => `${key}=${encodeURIComponent(String(value))}`)
  const response = await fetch(`${serverRoot(baseUrl)}/v1/library${parts.length ? `?${parts.join("&")}` : ""}`, {
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {}, signal,
  })
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`.trim()
    try { message = ((await response.json()) as { error?: { message?: string } }).error?.message || message } catch { /* keep the status line */ }
    throw new Error(message)
  }
  const body = (await response.json()) as { data?: unknown[]; total?: number; has_more?: boolean }
  const entries = (body.data ?? []).map(entryOf).filter((entry): entry is LibraryEntry => entry !== null)
  return { entries, total: typeof body.total === "number" ? body.total : entries.length, hasMore: body.has_more === true }
}

/* "2.4 MB", "310 KB". */
export function fileSize(bytes: number): string {
  if (bytes >= 1 << 20) return `${(bytes / (1 << 20)).toFixed(1)} MB`
  if (bytes >= 1 << 10) return `${Math.round(bytes / (1 << 10))} KB`
  return `${bytes} B`
}

/* "1:05" for a song's length. */
export function clock(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds))
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`
}
