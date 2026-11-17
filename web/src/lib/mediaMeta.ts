/* The settings the server writes inside a finished picture or song (c/media_meta.py): a PNG's iTXt chunk named
   "colibri", a WAV's LIST/INFO comment. Reading one back refills the page's controls. Only the chunk headers are
   read, so a song of a hundred megabytes is not loaded to find a record that sits after the samples. */

import type { MusicLimits } from "./api"
import { MAX_SEED, clampSteps, validSide, type ImageSize } from "./images"
import { DEFAULT_MUSIC_LIMITS, clampDuration, clampMusicSteps, validLanguage } from "./music"

export interface MediaSettings {
  kind: "image" | "music"
  model?: string
  prompt: string
  seed?: number
  steps?: number
  width?: number
  height?: number
  lyrics?: string
  language?: string
  duration?: number
}

const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
/* A record is a prompt and a few numbers; lyrics at the server's limit are 32 KB. A chunk claiming more is not ours. */
const MAX_RECORD = 1 << 20

const text = (bytes: Uint8Array) => new TextDecoder("utf-8").decode(bytes)
const ascii = (bytes: Uint8Array, start: number, length: number) => text(bytes.subarray(start, start + length))

async function bytes(blob: Blob, start: number, end: number) {
  return new Uint8Array(await blob.slice(start, end).arrayBuffer())
}

function number(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

/* The record as the page uses it, or null when it is not a Colibri record (somebody else's JSON, a different app,
   a kind the page has no controls for). Fields of the wrong type are dropped, not trusted. */
export function parseRecord(raw: string): MediaSettings | null {
  let found: unknown
  try { found = JSON.parse(raw) } catch { return null }
  if (typeof found !== "object" || found === null) return null
  const record = found as Record<string, unknown>
  if (record.app !== "colibri" || (record.kind !== "image" && record.kind !== "music")) return null
  if (typeof record.prompt !== "string") return null
  const settings: MediaSettings = { kind: record.kind, prompt: record.prompt }
  if (typeof record.model === "string") settings.model = record.model
  for (const key of ["seed", "steps", "width", "height", "duration"] as const) {
    const value = number(record[key])
    if (value !== undefined) settings[key] = value
  }
  if (typeof record.lyrics === "string") settings.lyrics = record.lyrics
  if (typeof record.language === "string") settings.language = record.language
  return settings
}

async function pngRecord(file: Blob): Promise<string | null> {
  let position = PNG_MAGIC.length
  while (position + 12 <= file.size) {
    const head = await bytes(file, position, position + 8)
    const size = new DataView(head.buffer).getUint32(0)
    const kind = ascii(head, 4, 4)
    if (kind === "IDAT" || kind === "IEND") return null            // the record sits before the pixels
    if (kind === "iTXt" && size <= MAX_RECORD) {
      const body = await bytes(file, position + 8, position + 8 + size)
      const keyword = body.indexOf(0)
      if (keyword > 0 && ascii(body, 0, keyword) === "colibri" && body[keyword + 1] === 0) {   // uncompressed only
        const language = body.indexOf(0, keyword + 3)
        const translated = language < 0 ? -1 : body.indexOf(0, language + 1)
        if (translated >= 0) return text(body.subarray(translated + 1))
      }
    }
    position += 12 + size
  }
  return null
}

async function wavRecord(file: Blob): Promise<string | null> {
  let position = 12
  while (position + 8 <= file.size) {
    const head = await bytes(file, position, position + 8)
    const size = new DataView(head.buffer).getUint32(4, true)
    if (ascii(head, 0, 4) === "LIST" && size >= 4 && size <= MAX_RECORD * 2) {
      const body = await bytes(file, position + 8, position + 8 + size)
      if (ascii(body, 0, 4) === "INFO") {
        let inner = 4
        while (inner + 8 <= body.length) {
          const length = new DataView(body.buffer, body.byteOffset).getUint32(inner + 4, true)
          if (ascii(body, inner, 4) === "ICMT") {
            const comment = body.subarray(inner + 8, inner + 8 + length)
            const end = comment.indexOf(0)
            return text(end < 0 ? comment : comment.subarray(0, end))
          }
          inner += 8 + length + (length & 1)
        }
      }
    }
    position += 8 + size + (size & 1)
  }
  return null
}

/* The settings in a picture or a song, or null when the file has none (a picture from somewhere else, a server told
   not to write them, a format the server does not tag). */
export async function readSettings(file: Blob): Promise<MediaSettings | null> {
  try {
    const head = await bytes(file, 0, 12)
    const raw = PNG_MAGIC.every((byte, at) => head[at] === byte) ? await pngRecord(file)
      : ascii(head, 0, 4) === "RIFF" && ascii(head, 8, 4) === "WAVE" ? await wavRecord(file) : null
    return raw === null ? null : parseRecord(raw)
  } catch {
    return null
  }
}

/* What opening a file does to the page: the model to switch to, the prompt, and each control the record had a valid
   value for (the others are left as they are). The file's own model when this server has it, else the first model
   of that kind; no model of that kind is the one thing it cannot do. */
export interface Refill {
  model: string
  prompt: string
  seed?: string
  steps?: number
  size?: ImageSize
  duration?: number
  language?: string
  lyrics?: string
}

export function refillFrom(
  settings: MediaSettings, imageModels: readonly string[], musicModels: Readonly<Record<string, MusicLimits>>,
): Refill | { error: "media.noModel" } {
  const names = settings.kind === "image" ? [...imageModels] : Object.keys(musicModels)
  const model = settings.model && names.includes(settings.model) ? settings.model : names[0]
  if (!model) return { error: "media.noModel" }
  const plan: Refill = { model, prompt: settings.prompt }
  if (settings.seed !== undefined && Number.isInteger(settings.seed) && settings.seed >= 0 && settings.seed <= MAX_SEED) plan.seed = String(settings.seed)
  if (settings.kind === "image") {
    if (settings.steps !== undefined) plan.steps = clampSteps(settings.steps)
    if (settings.width !== undefined && settings.height !== undefined && validSide(settings.width) && validSide(settings.height)) {
      plan.size = { width: settings.width, height: settings.height }
    }
  } else {
    const limits = musicModels[model] ?? DEFAULT_MUSIC_LIMITS
    if (settings.steps !== undefined) plan.steps = clampMusicSteps(settings.steps, limits)
    if (settings.duration !== undefined) plan.duration = clampDuration(settings.duration, limits)
    if (settings.language !== undefined && validLanguage(settings.language)) plan.language = settings.language
    if (settings.lyrics !== undefined) plan.lyrics = settings.lyrics
  }
  return plan
}
