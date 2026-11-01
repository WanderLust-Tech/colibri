import type { MusicLimits, MusicProgress } from "./api"

/* The state of the one song being made. Kept out of the transcript: it changes on every
   step, and only the finished song belongs there. */
export interface MusicRun {
  targetId: string
  started: number
  progress: MusicProgress | null
  /* No stream: one plain request, so no stage will ever be reported. */
  plain?: boolean
}

/* The contract's defaults (music_engine.DEFAULT_INFO), for a server that states less than
   its /v1/models `audio` block should. The block, when there, wins. */
export const DEFAULT_MUSIC_LIMITS: MusicLimits = {
  sampleRate: 48000, minDuration: 1, maxDuration: 600, defaultDuration: 30,
  defaultSteps: 8, maxSteps: 20, defaultShift: 1, maxLyricBytes: 32768,
}

/* The languages the model was trained to sing in (ACE-Step's VALID_LANGUAGES), and "" for
   none given: the engine then says "unknown". Instrumental music needs no language. */
export const MUSIC_LANGUAGES = [
  "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu",
  "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr",
  "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
] as const

export function validLanguage(code: string) {
  return code === "" || (MUSIC_LANGUAGES as readonly string[]).includes(code)
}

/* A clip is a number of seconds the engine accepts; the engine rounds it to a tenth, and a
   clip under 5.12 s is made on a longer one and cut, so a fraction is meaningful. */
export function validDuration(value: number, limits: MusicLimits) {
  return Number.isFinite(value) && value >= limits.minDuration && value <= limits.maxDuration
}

/* What a duration becomes when the field is left: inside the range, to a tenth of a second,
   so a typed 0 turns into the shortest clip instead of an error. */
export function clampDuration(value: number, limits: MusicLimits) {
  if (!Number.isFinite(value)) return limits.defaultDuration
  const clamped = Math.min(limits.maxDuration, Math.max(limits.minDuration, value))
  return Math.round(clamped * 10) / 10
}

export function clampMusicSteps(value: number, limits: MusicLimits) {
  if (!Number.isFinite(value)) return limits.defaultSteps
  return Math.min(limits.maxSteps, Math.max(1, Math.round(value)))
}

/* The lyrics the engine reads are cut at a size it states; the page says so before sending
   rather than after, with the bytes the engine counts (UTF-8), not the characters. */
export function lyricBytes(text: string) {
  return new TextEncoder().encode(text).length
}

/* Share of the whole job done, for the bar. The sampler's steps are nearly all of it (the
   text encoder, the conditioning and the decoder are each a small slice), so the steps carry
   75% of the bar, the decoder's windows the last 20%, and the two short stages a sliver each
   so the bar is not empty while they run. */
export function musicProgressFraction(progress: MusicProgress | null) {
  if (!progress) return 0
  const steps = Math.max(1, Math.round(progress.steps) || 1)
  const step = Math.min(steps, Math.max(0, Math.round(progress.step) || 0))
  switch (progress.stage) {
    case "text": return 0.02
    case "condition": return 0.05
    case "denoise": return 0.05 + 0.75 * (step / steps)
    case "decode": return 0.8 + 0.2 * (step / steps)
    default: return 0
  }
}

/* A name that says what the file is and which seed makes it again. */
export function musicFileName(prompt: string, seed: number) {
  const words = prompt.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48).replace(/-+$/, "")
  return `colibri-${words || "music"}-${seed}.wav`
}

/* "2:05" for a song of 125 seconds. */
export function songClock(seconds: number) {
  const whole = Math.max(0, Math.round(seconds))
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`
}
