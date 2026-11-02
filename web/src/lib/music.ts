import type { MusicLimits, MusicProgress, MusicTask } from "./api"

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
  defaultSteps: 8, maxSteps: 20, defaultShift: 1, maxLyricBytes: 32768, tasks: ["text2music"],
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
   70% of the bar, the decoder's windows the last 20%, reading a clip (a cover, a repaint or a
   reference: its windows) the first 8%, and the two short stages a sliver each so the bar is
   not empty while they run. A song made from text alone never reports the first, and its
   sampler's share starts a little lower, which is the same bar to the eye. */
export function musicProgressFraction(progress: MusicProgress | null) {
  if (!progress) return 0
  const steps = Math.max(1, Math.round(progress.steps) || 1)
  const step = Math.min(steps, Math.max(0, Math.round(progress.step) || 0))
  switch (progress.stage) {          // whole percents, divided at the end, so the stages meet exactly
    case "encode": return (1 + 7 * (step / steps)) / 100
    case "text": return 9 / 100
    case "condition": return 10 / 100
    case "denoise": return (10 + 70 * (step / steps)) / 100
    case "decode": return (80 + 20 * (step / steps)) / 100
    default: return 0
  }
}

/* ---- audio in: a clip to cover or repaint, a reference clip ------------------------------- */

/* The most a clip may weigh: what the gateway takes of one file (music_engine.MAX_CLIP_BYTES). */
export const MAX_CLIP_BYTES = 128 * 1024 * 1024

/* WAV and MP3 are what the engine reads. The page only has the name and the type the browser
   gives, so this says no to the obvious (a picture, an .ogg); the engine looks at the bytes. */
export function acceptedClip(file: { name: string; type: string }) {
  return /\.(wav|mp3)$/i.test(file.name) || /^audio\/(wav|x-wav|wave|vnd\.wave|mpeg|mp3)$/i.test(file.type)
}

export type ClipProblem = "music.clip.type" | "music.clip.large" | "music.clip.long"

/* Why a clip cannot be sent, as the key of the sentence that says it, or undefined. */
export function clipProblem(file: { name: string; type: string; size: number }, seconds: number | undefined, limits: MusicLimits): ClipProblem | undefined {
  if (!acceptedClip(file)) return "music.clip.type"
  if (file.size > MAX_CLIP_BYTES) return "music.clip.large"
  if (seconds !== undefined && limits.audioIn && seconds > limits.audioIn.maxSeconds) return "music.clip.long"
  return undefined
}

/* The two numbers of a repaint, as typed: a start in seconds (before 0 means silence is added ahead
   of the clip) and an end ("" is the clip's end; past it silence is added after). `clip` is the
   clip's length when the browser could read it. The span must run forwards, start inside the clip,
   and leave the whole, with the silence added, within what the engine makes. */
export function repaintValid(from: number, to: number | undefined, clip: number | undefined, limits: MusicLimits) {
  if (!Number.isFinite(from) || (to !== undefined && !Number.isFinite(to))) return false
  const end = to ?? clip
  if (end !== undefined && !(end > from)) return false
  if (clip !== undefined) {
    if (!(from < clip)) return false
    const total = clip + Math.max(0, -from) + Math.max(0, (end ?? clip) - clip)
    if (limits.audioIn && total > limits.audioIn.maxSeconds) return false
  }
  return true
}

/* A number typed into a 0-to-1 field, put inside the range; NaN when it is not a number. */
export function clampUnit(value: number) {
  if (!Number.isFinite(value)) return Number.NaN
  return Math.round(Math.min(1, Math.max(0, value)) * 100) / 100
}

/* Does the model offer this task (an older engine offers composing only)? */
export const offersTask = (limits: MusicLimits | undefined, task: MusicTask) => (limits?.tasks ?? ["text2music"]).includes(task)

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
