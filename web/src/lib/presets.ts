/* Presets in the dashboard (docs/presets.md): turning a preset's defaults into what the page's own controls show, and the
   text of an editable "key = value" row into a default and back. */

import { parseSize, type ImageSize } from "@/lib/images"

/* What the page's controls start from when a preset is chosen. Only what the page has a control for: the others are left to
   the router, which fills them in itself, because the page does not send them. */
export interface ControlDefaults {
  temperature?: number
  maxTokens?: number
  size?: ImageSize
  steps?: number
}

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value)

export function controlDefaults(defaults: Record<string, unknown>): ControlDefaults {
  const out: ControlDefaults = {}
  if (finite(defaults.temperature)) out.temperature = Math.min(2, Math.max(0, defaults.temperature))
  const max = defaults.max_tokens ?? defaults.max_completion_tokens
  if (finite(max) && max >= 1) out.maxTokens = Math.min(32768, Math.round(max))
  if (typeof defaults.size === "string") {
    const size = parseSize(defaults.size)
    if (size) out.size = size
  }
  if (finite(defaults.steps) && defaults.steps >= 1) out.steps = Math.round(defaults.steps)
  return out
}

/* A default as the text of a field: a string as it is, anything else as JSON (20, 0.2, true, {"a":1}). */
export function defaultText(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value)
}

/* The text of a field as a default: numbers, true, false, null and JSON objects or arrays are those; anything else is text. */
export function parseDefault(text: string): unknown {
  const trimmed = text.trim()
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(trimmed) || trimmed === "true" || trimmed === "false" || trimmed === "null") {
    return JSON.parse(trimmed)
  }
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try { return JSON.parse(trimmed) } catch { /* text after all */ }
  }
  return text
}

export interface DefaultRow { key: string; value: string }

export const rowsOf = (defaults: Record<string, unknown>): DefaultRow[] =>
  Object.entries(defaults).map(([key, value]) => ({ key, value: defaultText(value) }))

/* The rows as the `defaults` of a preset: a row with no key is nothing, a key given twice keeps its last value. */
export function defaultsOf(rows: DefaultRow[]): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const { key, value } of rows) {
    const name = key.trim()
    if (name && value.trim() !== "") out[name] = parseDefault(value)
  }
  return out
}

/* The names the editor offers for a default, by the kind of model the preset is for. Free text is allowed too. */
export function suggestedKeys(modality: string): string[] {
  if (modality === "image") return ["size", "steps", "seed", "n"]
  if (modality === "audio" || modality === "music") return ["duration", "steps", "shift", "seed", "task"]
  return ["temperature", "max_tokens", "top_p", "top_k", "frequency_penalty", "presence_penalty", "stop", "seed"]
}

/* A preset name as the router accepts it (presets.py NAME); an empty text is not an error yet. */
export function nameProblem(name: string): "empty" | "invalid" | null {
  if (!name.trim()) return "empty"
  return /^[a-z0-9][a-z0-9._-]{0,63}$/.test(name.trim().toLowerCase()) ? null : "invalid"
}
