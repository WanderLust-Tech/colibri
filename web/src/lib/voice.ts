/* The voice loop in the page (docs/voice-loop.md): what the microphone and the loudspeaker need from the server's models, and the small pieces of
   arithmetic around them. The recording and the playing are in useVoice.ts; this file has no browser audio in it, so it can be tested. */

import { endpoint, type ModelInfo } from "./api"

export const MAX_SPEECH_CHARS = 4096

/* What a voice should say of a chat reply: markdown, code blocks, the addresses of links and emoji dropped, white space made plain. */
export function spokenText(text: string): string {
  return text
    .replace(/```[\s\S]*?(```|$)/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/^\s{0,3}#{1,6}\s*/gm, "")
    .replace(/^\s*(?:[-*+]|\d+[.)])\s+/gm, "")
    .replace(/[*_`~]{1,3}/g, "")
    .replace(/[\u{1F000}-\u{1FFFF}☀-➿️‍]/gu, "")
    .replace(/\s+/g, " ")
    .trim()
}

/* The models a spoken conversation uses, from /v1/models: the first speech recogniser, the first speech model, and a voice-cloning (Base) one for a saved voice. */
export function pickVoiceModels(models: ModelInfo[]): { asr?: string; tts?: string; clone?: string } {
  const speech = models.filter((model) => model.capabilities?.includes("speech"))
  return {
    asr: models.find((model) => model.capabilities?.includes("transcription"))?.id,
    tts: (speech.find((model) => !/base/i.test(model.id)) ?? speech[0])?.id,
    clone: speech.find((model) => /base/i.test(model.id))?.id,
  }
}

/* Splits the text of server-sent events as it arrives: the JSON of each complete `data:` event, and what is left of the last one. */
export function parseEvents(buffer: string): { events: unknown[]; rest: string } {
  const events: unknown[] = []
  const blocks = buffer.split("\n\n")
  const rest = blocks.pop() ?? ""
  for (const block of blocks) {
    for (const line of block.split("\n")) {
      if (!line.startsWith("data:")) continue
      const data = line.slice(5).trim()
      if (!data || data === "[DONE]") continue
      try { events.push(JSON.parse(data)) } catch { /* a line that is not JSON is not ours */ }
    }
  }
  return { events, rest }
}

/* 16-bit little-endian PCM as the floats a Web Audio buffer holds. An odd trailing byte is dropped. */
export function pcmToFloat32(bytes: Uint8Array): Float32Array {
  const count = Math.floor(bytes.length / 2)
  const view = new DataView(bytes.buffer, bytes.byteOffset, count * 2)
  const out = new Float32Array(count)
  for (let i = 0; i < count; i++) out[i] = view.getInt16(i * 2, true) / 32768
  return out
}

export function base64Bytes(text: string): Uint8Array {
  const binary = atob(text)
  const out = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i)
  return out
}

function authorization(apiKey: string): Record<string, string> {
  return apiKey ? { Authorization: `Bearer ${apiKey}` } : {}
}

async function failure(response: Response): Promise<Error> {
  try {
    const body = (await response.json()) as { error?: { message?: string } }
    return new Error(body.error?.message || `${response.status} ${response.statusText}`)
  } catch {
    return new Error(`${response.status} ${response.statusText}`)
  }
}

/* The words in a recording (a blob the browser made: webm, ogg, mp4; the server reads what ffmpeg reads). */
export async function transcribeRecording(baseUrl: string, apiKey: string, model: string, audio: Blob, signal?: AbortSignal): Promise<string> {
  const form = new FormData()
  form.append("model", model)
  form.append("response_format", "json")
  form.append("file", audio, "speech.webm")
  const response = await fetch(endpoint(baseUrl, "audio/transcriptions"), { method: "POST", headers: authorization(apiKey), body: form, signal })
  if (!response.ok) throw await failure(response)
  return String(((await response.json()) as { text?: unknown }).text ?? "").trim()
}

/* Speech for `text` as the server makes it: `onChunk` gets each piece as floats (24 kHz mono) the moment it arrives. */
export async function streamSpeech(
  baseUrl: string, apiKey: string, model: string, text: string, voice: string | undefined,
  onChunk: (samples: Float32Array) => void, signal?: AbortSignal,
): Promise<void> {
  const body = { model, input: text.slice(0, MAX_SPEECH_CHARS), stream_format: "sse", ...(voice ? { voice } : {}) }
  const response = await fetch(endpoint(baseUrl, "audio/speech"), {
    method: "POST", headers: { "Content-Type": "application/json", ...authorization(apiKey) }, body: JSON.stringify(body), signal,
  })
  if (!response.ok) throw await failure(response)
  if (!response.body) throw new Error("the server sent no audio")
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parsed = parseEvents(buffer)
    buffer = parsed.rest
    for (const event of parsed.events as Array<{ type?: string; audio?: string; error?: { message?: string } }>) {
      if (event.type === "speech.audio.delta" && event.audio) onChunk(pcmToFloat32(base64Bytes(event.audio)))
      else if (event.type === "error") throw new Error(event.error?.message || "the speech model failed")
    }
  }
}

/* Decides, from the loudness of the microphone a few times a second, when someone starts talking and when they have stopped: speech is a run of
   `attack` readings above the bar (3 times the noise learned in the first readings, and never under `floor`); it ends after `silenceMs` below it. */
export class SilenceDetector {
  private noise: number | null = null
  private loud = 0
  private quietSince: number | null = null
  private readings = 0
  speaking = false

  constructor(private readonly options: { floor?: number; silenceMs?: number; attack?: number; calibrate?: number } = {}) {}

  /* `level` is an RMS in 0..1, `now` a time in ms. Returns "speech" when speech has just begun, "end" when it has just ended, else "idle". */
  push(level: number, now: number): "idle" | "speech" | "end" {
    const { floor = 0.012, silenceMs = 1000, attack = 3, calibrate = 8 } = this.options
    this.readings++
    if (this.noise === null) this.noise = level
    if (!this.speaking && this.readings <= calibrate) {
      this.noise = this.readings === 1 ? level : Math.min(this.noise, level)
      return "idle"
    }
    const bar = Math.max(floor, 3 * this.noise)
    if (!this.speaking) {
      this.loud = level > bar ? this.loud + 1 : 0
      if (level <= bar) this.noise = 0.95 * this.noise + 0.05 * level
      if (this.loud >= attack) { this.speaking = true; this.quietSince = null; return "speech" }
      return "idle"
    }
    if (level > bar) { this.quietSince = null; return "idle" }
    if (this.quietSince === null) this.quietSince = now
    if (now - this.quietSince >= silenceMs) { this.speaking = false; this.loud = 0; return "end" }
    return "idle"
  }
}
