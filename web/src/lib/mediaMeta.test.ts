import { describe, expect, it } from "vitest"
import { DEFAULT_MUSIC_LIMITS } from "./music"
import { parseRecord, readSettings, refillFrom } from "./mediaMeta"

/* Made by c/media_meta.py itself, so the two sides are tested against each other and not each against its own idea. */
const PNG = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAo2lUWHRjb2xpYnJpAAAAAAB7ImFwcCI6ImNvbGlicmkiLCJ2ZXJzaW9uIjoxLCJraW5kIjoiaW1hZ2UiLCJtb2RlbCI6InF3ZW4taW1hZ2UiLCJwcm9tcHQiOiLml6XmnKzjga7njKsgXCJxdW90ZWRcIlxubGluZSIsIndpZHRoIjo3NjgsImhlaWdodCI6NTEyLCJzdGVwcyI6OCwic2VlZCI6NDJ98NElMwAAABRJREFUeJxjYGBkYmZhZWNn4oACAAKUAF/r/qahAAAAAElFTkSuQmCC"
const WAV = "UklGRgoBAABXQVZFZm10IBAAAAABAAIAgLsAAADuAgAEABAAZGF0YQgAAAABAAIAAwAEAExJU1TWAAAASU5GT0lTRlQIAAAAQ29saWJyaQBJTkFNCwAAAGNhbG0gcGlhbm8AAElDTVSmAAAAeyJhcHAiOiJjb2xpYnJpIiwidmVyc2lvbiI6MSwia2luZCI6Im11c2ljIiwibW9kZWwiOiJhY2Utc3RlcCIsInByb21wdCI6ImNhbG0gcGlhbm8iLCJseXJpY3MiOiJsYSBsYVxubGEiLCJsYW5ndWFnZSI6ImVuIiwiZHVyYXRpb24iOjMwLCJzdGVwcyI6OCwic2hpZnQiOjEsInNlZWQiOjd9AA=="

const raw = (base64: string) => Uint8Array.from(atob(base64), (c) => c.charCodeAt(0))
const blob = (base64: string) => new Blob([raw(base64)])

describe("readSettings", () => {
  it("reads a picture's record, quotes, newline and non-Latin text included", async () => {
    expect(await readSettings(blob(PNG))).toEqual({
      kind: "image", model: "qwen-image", prompt: "日本の猫 \"quoted\"\nline", width: 768, height: 512, steps: 8, seed: 42,
    })
  })

  it("reads a song's record from after the samples", async () => {
    expect(await readSettings(blob(WAV))).toEqual({
      kind: "music", model: "ace-step", prompt: "calm piano", lyrics: "la la\nla", language: "en", duration: 30, steps: 8, seed: 7,
    })
  })

  it("answers null for a file that has none, or that is not a picture or a song", async () => {
    const whole = raw(PNG)
    const header = whole.slice(0, 33)                                  // signature and IHDR
    const end = new Uint8Array([0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130])
    expect(await readSettings(new Blob([header, end]))).toBeNull()
    expect(await readSettings(new Blob(["hello, this is text"]))).toBeNull()
    expect(await readSettings(new Blob([]))).toBeNull()
  })

  it("does not trust a file that is cut off inside a chunk", async () => {
    expect(await readSettings(new Blob([raw(PNG).slice(0, 60)]))).toBeNull()
  })
})

describe("parseRecord", () => {
  const record = (fields: object) => JSON.stringify({ app: "colibri", version: 1, kind: "image", prompt: "a fox", ...fields })

  it("keeps what is right and drops what is of the wrong type", () => {
    expect(parseRecord(record({ seed: 5, steps: "8", width: null, model: 3, lyrics: ["x"] })))
      .toEqual({ kind: "image", prompt: "a fox", seed: 5 })
  })

  it("refuses somebody else's JSON, another kind, no prompt, and text that is not JSON", () => {
    expect(parseRecord(JSON.stringify({ app: "other", kind: "image", prompt: "a" }))).toBeNull()
    expect(parseRecord(record({ kind: "video" }))).toBeNull()
    expect(parseRecord(JSON.stringify({ app: "colibri", kind: "image" }))).toBeNull()
    expect(parseRecord("{oops")).toBeNull()
    expect(parseRecord("[1]")).toBeNull()
    expect(parseRecord("null")).toBeNull()
  })
})

describe("refillFrom", () => {
  const music = { "ace-step": DEFAULT_MUSIC_LIMITS, "other-music": { ...DEFAULT_MUSIC_LIMITS, maxSteps: 4, maxDuration: 60 } }

  it("goes back to the model the picture was made with, with its size, steps and seed", () => {
    expect(refillFrom({ kind: "image", model: "b", prompt: "a fox", width: 768, height: 512, steps: 8, seed: 42 }, ["a", "b"], music))
      .toEqual({ model: "b", prompt: "a fox", seed: "42", steps: 8, size: { width: 768, height: 512 } })
  })

  it("takes the first image model when this server does not have the file's", () => {
    expect(refillFrom({ kind: "image", model: "gone", prompt: "x" }, ["a", "b"], music)).toEqual({ model: "a", prompt: "x" })
    expect(refillFrom({ kind: "image", prompt: "x" }, ["a", "b"], music)).toEqual({ model: "a", prompt: "x" })
  })

  it("says so when there is no model of that kind", () => {
    expect(refillFrom({ kind: "image", prompt: "x" }, [], music)).toEqual({ error: "media.noModel" })
    expect(refillFrom({ kind: "music", prompt: "x" }, ["a"], {})).toEqual({ error: "media.noModel" })
  })

  it("leaves out a size the engine would refuse and a seed that does not fit", () => {
    const plan = refillFrom({ kind: "image", model: "a", prompt: "x", width: 100, height: 512, steps: 8, seed: -1 }, ["a"], music)
    expect(plan).toEqual({ model: "a", prompt: "x", steps: 8 })
    expect(refillFrom({ kind: "image", model: "a", prompt: "x", seed: 2 ** 32 }, ["a"], music)).toEqual({ model: "a", prompt: "x" })
    expect(refillFrom({ kind: "image", model: "a", prompt: "x", seed: 1.5 }, ["a"], music)).toEqual({ model: "a", prompt: "x" })
  })

  it("refills a song: lyrics, language, duration, steps, seed, held to the chosen model's limits", () => {
    expect(refillFrom({ kind: "music", model: "other-music", prompt: "calm", lyrics: "la", language: "en", duration: 600, steps: 20, seed: 7 }, [], music))
      .toEqual({ model: "other-music", prompt: "calm", lyrics: "la", language: "en", duration: 60, steps: 4, seed: "7" })
  })

  it("drops a language the model was not trained on", () => {
    expect(refillFrom({ kind: "music", model: "ace-step", prompt: "x", language: "xx" }, [], music)).toEqual({ model: "ace-step", prompt: "x" })
  })

  it("a song does not pick an image model and a picture does not pick a music one", () => {
    expect(refillFrom({ kind: "music", model: "a", prompt: "x" }, ["a"], music)).toEqual({ model: "ace-step", prompt: "x" })
    expect(refillFrom({ kind: "image", model: "ace-step", prompt: "x" }, ["a"], music)).toEqual({ model: "a", prompt: "x" })
  })
})
