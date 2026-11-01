import { useEffect, useMemo, useState, type MutableRefObject } from "react"

import { generateMusic, type ChatMessage, type GeneratedMusic, type MusicLimits } from "./api"
import { parseSeed, randomSeed } from "./images"
import { DEFAULT_MUSIC_LIMITS, clampDuration, clampMusicSteps, lyricBytes, musicFileName, validDuration, validLanguage, type MusicRun } from "./music"
import { setFinish } from "./transcript"

/* What App hands over: the connection, the one conversation it lives in, and the flags it
   shares with the chat and the picture mode (loading, the abort controller, the error line). */
export interface MusicEnvironment {
  baseUrl: string
  apiKey: string
  model: string
  /* The limits of the selected model when it is a music model, else undefined. */
  limits: MusicLimits | undefined
  draft: string
  setDraft: (text: string) => void
  loading: boolean
  setLoading: (loading: boolean) => void
  abortRef: MutableRefObject<AbortController | null>
  updateMessages: (next: (current: ChatMessage[]) => ChatMessage[]) => void
  setError: (key: string) => void
  setConnected: (connected: boolean) => void
  newMessage: (role: ChatMessage["role"], content: string) => ChatMessage
}

const read = (key: string, fallback: string) => {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}

/* Music mode: the state of the controls, the song being made, and what each button does.
   Length, steps and language are remembered like the endpoint; the seed and the lyrics are
   not, because a remembered seed would silently make the same song again and lyrics belong
   to one song. */
export function useMusic(env: MusicEnvironment) {
  const limits = env.limits ?? DEFAULT_MUSIC_LIMITS
  const mode = env.limits !== undefined
  const [duration, setDuration] = useState(() => Number(read("colibri.musicDuration", "")) || DEFAULT_MUSIC_LIMITS.defaultDuration)
  const [steps, setSteps] = useState(() => Number(read("colibri.musicSteps", "")) || DEFAULT_MUSIC_LIMITS.defaultSteps)
  const [language, setLanguage] = useState(() => { const code = read("colibri.musicLanguage", ""); return validLanguage(code) ? code : "" })
  const [seedText, setSeedText] = useState("")
  const [lyrics, setLyrics] = useState("")
  const [lyricsOpen, setLyricsOpen] = useState(false)
  const [run, setRun] = useState<MusicRun | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const seed = parseSeed(seedText)

  useEffect(() => {
    try {
      if (Number.isFinite(duration)) localStorage.setItem("colibri.musicDuration", String(duration))     // NaN: a field being edited
      if (Number.isFinite(steps)) localStorage.setItem("colibri.musicSteps", String(steps))
      localStorage.setItem("colibri.musicLanguage", language)
    } catch { /* restricted storage mode */ }
  }, [duration, steps, language])

  /* The values the next song is made with: the remembered ones, put inside the range of the
     model that is selected now (another music model may allow less). */
  const effectiveDuration = clampDuration(duration, limits)
  const effectiveSteps = clampMusicSteps(steps, limits)

  const canSend = useMemo(() => mode && Boolean(
    (env.draft.trim() || lyrics.trim()) && env.model && !env.loading && seed !== undefined &&
    validDuration(duration, limits) && Number.isInteger(steps) && steps >= 1 && steps <= limits.maxSteps &&
    lyricBytes(lyrics) <= limits.maxLyricBytes,
  ), [mode, env.draft, env.model, env.loading, lyrics, seed, duration, steps, limits])

  /* Generates into the assistant turn `targetId`, whose `song` already holds the request.
     Progress lives in `run`, outside the transcript; the finished song is written onto the
     turn. Cancel is the same abort controller as the chat's Stop. */
  const make = async (targetId: string, request: GeneratedMusic) => {
    env.setError("")
    env.setLoading(true)
    const controller = new AbortController()
    env.abortRef.current = controller
    const started = performance.now()
    setRun({ targetId, started: Date.now(), progress: null })
    const update = (change: Partial<MusicRun>) => setRun((current) => current && current.targetId === targetId ? { ...current, ...change } : current)
    try {
      const song = await generateMusic({
        baseUrl: env.baseUrl, apiKey: env.apiKey, model: env.model,
        prompt: request.prompt, lyrics: request.lyrics, language: request.language,
        duration: request.duration, steps: request.steps, seed: request.seed,
        signal: controller.signal,
        onProgress: (progress) => update({ progress }),
        onPlainRequest: () => update({ plain: true }),
      })
      const seconds = (performance.now() - started) / 1000
      env.updateMessages((current) => current.map((item) => item.id === targetId
        ? { ...item, finish: "stop", song: { ...song, seconds } } : item))
      env.setConnected(true)
    } catch (cause) {
      const finish = controller.signal.aborted ? "aborted" : "error"
      if (!controller.signal.aborted) env.setError(cause instanceof Error ? cause.message : "status.generationFailed")
      env.updateMessages((current) => setFinish(current, targetId, finish))
    } finally {
      env.abortRef.current = null
      setRun(null)
      env.setLoading(false)
    }
  }

  const turn = (request: GeneratedMusic): ChatMessage => ({ ...env.newMessage("assistant", ""), mode: "music", song: request })

  const send = async () => {
    const prompt = env.draft.trim()
    if (!canSend || seed === undefined) return
    const request: GeneratedMusic = {
      prompt, duration: effectiveDuration, steps: effectiveSteps, seed: seed ?? randomSeed(),
      ...(lyrics.trim() ? { lyrics } : {}), ...(language ? { language } : {}),
    }
    const assistant = turn(request)
    /* the question as it reads in the conversation: the caption, or the first line of the lyrics when there is none */
    const asked = prompt || `♪ ${lyrics.trim().split("\n")[0]}`
    env.setDraft("")
    env.updateMessages((current) => [...current, { ...env.newMessage("user", asked), mode: "music" }, assistant])
    await make(assistant.id, request)
  }

  /* Same caption, lyrics, length and steps, a seed never used before: a new song under the
     old one, so the two can be compared. */
  const again = async (song: GeneratedMusic) => {
    if (env.loading) return
    const { prompt, lyrics: words, language: code, duration: length, steps: count } = song
    const assistant = turn({ prompt, duration: length, steps: count, seed: randomSeed(), ...(words ? { lyrics: words } : {}), ...(code ? { language: code } : {}) })
    env.updateMessages((current) => [...current, assistant])
    await make(assistant.id, assistant.song!)
  }

  const retry = async (item: ChatMessage) => {
    if (env.loading || !item.song) return
    const { url: _url, seconds: _seconds, timings: _timings, truncated: _truncated, ...request } = item.song
    env.updateMessages((current) => current.map((entry) => entry.id === item.id ? { ...entry, finish: undefined, song: request } : entry))
    await make(item.id, request)
  }

  const download = (song: GeneratedMusic) => {
    if (!song.url) return
    const anchor = document.createElement("a")
    anchor.href = song.url
    anchor.download = musicFileName(song.prompt || song.lyrics || "", song.seed)
    anchor.click()
  }

  const copySeed = async (item: ChatMessage) => {
    if (!item.song) return
    try { await navigator.clipboard.writeText(String(item.song.seed)); setCopied(`${item.id}:seed`) } catch { env.setError("ui.copyError") }
  }

  return {
    mode, limits, duration, steps, language, seedText, lyrics, lyricsOpen, run, copied, seed, canSend,
    setDuration, setSteps, setLanguage, setSeedText, setLyrics,
    toggleLyrics: () => setLyricsOpen((open) => !open),
    cancel: () => env.abortRef.current?.abort(),
    send, again, retry, download, copySeed,
  }
}
