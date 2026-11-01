import type { ChatMessage, ImageProgress } from "./api"

/* The engine takes sides that are multiples of 32 between 256 and 2048 and
   answers an error otherwise (qwenimage READY: min_side, max_side, multiple). */
export const SIDE = { min: 256, max: 2048, multiple: 32 } as const

export interface ImageSize {
  width: number
  height: number
}

/* Every preset is a multiple of 32 on both sides, so none of them can be
   refused. 16:9 at that granularity exists only as multiples of 512x288.
   The select groups them by ratio, in the order a ratio first appears here,
   and each group runs from the smallest picture to the largest: a bigger
   picture costs more than its pixel count in time, so the order is the cost. */
export const SIZE_PRESETS: ReadonlyArray<ImageSize & { ratio: string }> = [
  { width: 512, height: 512, ratio: "1:1" },
  { width: 768, height: 768, ratio: "1:1" },
  { width: 1024, height: 1024, ratio: "1:1" },
  { width: 1536, height: 1536, ratio: "1:1" },
  { width: 768, height: 512, ratio: "3:2" },
  { width: 1152, height: 768, ratio: "3:2" },
  { width: 1536, height: 1024, ratio: "3:2" },
  { width: 512, height: 768, ratio: "2:3" },
  { width: 768, height: 1152, ratio: "2:3" },
  { width: 1024, height: 1536, ratio: "2:3" },
  { width: 768, height: 576, ratio: "4:3" },
  { width: 1024, height: 768, ratio: "4:3" },
  { width: 576, height: 768, ratio: "3:4" },
  { width: 768, height: 1024, ratio: "3:4" },
  { width: 1024, height: 576, ratio: "16:9" },
  { width: 1536, height: 864, ratio: "16:9" },
  { width: 576, height: 1024, ratio: "9:16" },
  { width: 864, height: 1536, ratio: "9:16" },
]

export const DEFAULT_SIZE: ImageSize = { width: 768, height: 512 }
export const DEFAULT_STEPS = 8
export const MAX_STEPS = 200
/* One step is refused by the engine (its schedule divides by zero there). */
export const MIN_STEPS = 2
/* A seed the UI draws itself fits a signed 32-bit integer, whatever the engine
   keeps it in. The gateway is still sent an explicit seed every time, so a new
   seed is really new even on a server that would fall back to a fixed one. */
export const MAX_RANDOM_SEED = 2 ** 31
/* The largest seed the gateway accepts (image_engine.MAX_SEED). */
export const MAX_SEED = 2 ** 32 - 1

export const sizeKey = ({ width, height }: ImageSize) => `${width}x${height}`

export function presetOf(size: ImageSize) {
  return SIZE_PRESETS.find((preset) => preset.width === size.width && preset.height === size.height)
}

export function validSide(value: number) {
  return Number.isInteger(value) && value >= SIDE.min && value <= SIDE.max && value % SIDE.multiple === 0
}

/* What a custom side becomes when the field is left: the nearest multiple of
   32 inside the range, so a typed 1000 turns into 992 instead of an error. */
export function snapSide(value: number) {
  if (!Number.isFinite(value)) return DEFAULT_SIZE.width
  const snapped = Math.round(value / SIDE.multiple) * SIDE.multiple
  return Math.min(SIDE.max, Math.max(SIDE.min, snapped))
}

/* A picture attached to an edit: the file to upload, an object URL for its
   thumbnail (revoked when it is removed), and its natural size, which gives the
   size the edit will come out at. */
export interface EditPicture {
  id: string
  file: File
  url: string
  width: number
  height: number
}

export function loadPicture(file: File): Promise<EditPicture> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const image = new Image()
    image.onload = () => resolve({
      id: `${file.name}:${file.size}:${file.lastModified}:${Math.random().toString(36).slice(2)}`,
      file, url, width: image.naturalWidth, height: image.naturalHeight,
    })
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error(file.name)) }
    image.src = url
  })
}

/* The side, in pixels, the engine works reference pictures at unless told otherwise (the
   pipeline's own is 1024: this is the memory-friendly default the engine reports). */
export const REF_RESOLUTION = 768

/* The size an edit comes out at when none is chosen: the engine's own rule (the
   pipeline's calculate_dimensions), the last picture's aspect at ref*ref pixels with
   each side rounded to a multiple of 32, ties to even. Used only to shape the
   placeholder; the finished picture says what was really drawn. null when the
   picture is so long and thin that a side rounds to nothing. */
export function derivedSize(width: number, height: number, ref = REF_RESOLUTION): ImageSize | null {
  if (!(width > 0 && height > 0)) return null
  const wide = Math.sqrt(ref * ref * (width / height))
  const roundEven = (value: number) => {
    const floor = Math.floor(value)
    const diff = value - floor
    return (diff < 0.5 ? floor : diff > 0.5 ? floor + 1 : floor % 2 === 0 ? floor : floor + 1) * SIDE.multiple
  }
  const size = { width: roundEven(wide / SIDE.multiple), height: roundEven(wide / (width / height) / SIDE.multiple) }
  return size.width >= SIDE.multiple && size.height >= SIDE.multiple ? size : null
}

export function parseSize(text: string | null): ImageSize | null {
  const match = /^(\d+)x(\d+)$/.exec(text || "")
  if (!match) return null
  const size = { width: Number(match[1]), height: Number(match[2]) }
  return validSide(size.width) && validSide(size.height) ? size : null
}

export function clampSteps(value: number) {
  if (!Number.isFinite(value)) return DEFAULT_STEPS
  return Math.min(MAX_STEPS, Math.max(MIN_STEPS, Math.round(value)))
}

/* An empty field means "draw a new seed"; anything else must be a whole
   number the page can hold exactly. */
export function parseSeed(text: string): number | null | undefined {
  const trimmed = text.trim()
  if (!trimmed) return null
  if (!/^\d+$/.test(trimmed)) return undefined
  const seed = Number(trimmed)
  return Number.isSafeInteger(seed) && seed <= MAX_SEED ? seed : undefined
}

export function randomSeed(random: () => number = Math.random) {
  return Math.floor(random() * MAX_RANDOM_SEED)
}

/* Share of the whole job done, for the bar. Encoding and decoding each cost
   about one denoising step on the engine, so the job is steps + 2 units. The
   encode stage shows half a unit, so the bar is not empty while the prompt is
   read. */
export function progressFraction(progress: ImageProgress | null) {
  if (!progress) return 0
  const steps = Math.max(1, Math.round(progress.steps) || 1)
  const step = Math.min(steps, Math.max(0, Math.round(progress.step) || 0))
  switch (progress.stage) {
    case "encode": return 0.5 / (steps + 2)
    case "denoise": return (1 + step) / (steps + 2)
    case "decode": return (steps + 1) / (steps + 2)
    default: return 0
  }
}

/* The turns a text model is shown: everything but the image exchanges. */
export function chatTurns(messages: ChatMessage[]) {
  return messages.filter((item) => item.mode !== "image")
}

export function elapsedClock(seconds: number) {
  const whole = Math.max(0, Math.floor(seconds))
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`
}

export function duration(seconds: number, locale = "en") {
  if (seconds < 60) {
    return `${seconds.toLocaleString(locale, { maximumFractionDigits: seconds < 10 ? 1 : 0 })} s`
  }
  const whole = Math.round(seconds)
  return `${Math.floor(whole / 60)} min ${whole % 60} s`
}

/* A name that says what the file is and which seed makes it again. */
export function imageFileName(prompt: string, seed: number) {
  const words = prompt.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48).replace(/-+$/, "")
  return `colibri-${words || "image"}-${seed}.png`
}

export function dataUrlBlob(url: string) {
  const comma = url.indexOf(",")
  const type = /^data:([^;,]+)/.exec(url)?.[1] || "image/png"
  const binary = atob(url.slice(comma + 1))
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return new Blob([bytes], { type })
}
