import { useEffect, useMemo, useRef, useState, type MutableRefObject } from "react"

import { generateMusic, type ChatMessage, type GeneratedMusic, type MusicLimits, type MusicTask, type RepaintMode } from "./api"
import { parseSeed, randomSeed } from "./images"
import {
  DEFAULT_MUSIC_LIMITS, clampDuration, clampMusicSteps, clampUnit, clipProblem, lyricBytes, musicFileName, offersTask, repaintValid,
  validDuration, validLanguage, type ClipProblem, type MusicRun,
} from "./music"
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

/* A clip the user chose: the file, a URL to play it by, and its length once the browser has read it
   (a browser that cannot decode the format leaves it out: the engine may still read the file). */
export interface Clip {
  file: File
  name: string
  url: string
  seconds?: number
}

const read = (key: string, fallback: string) => {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}

/* How long a clip is, by asking the browser's own decoder; undefined when it cannot say, within a few
   seconds (an MP3 of a long song reports its length from its header, so this is quick). */
function clipSeconds(url: string): Promise<number | undefined> {
  return new Promise((resolve) => {
    const audio = new Audio()
    const finish = (value: number | undefined) => { window.clearTimeout(timer); audio.removeAttribute("src"); audio.load(); resolve(value) }
    const timer = window.setTimeout(() => finish(undefined), 8000)
    audio.preload = "metadata"
    audio.onloadedmetadata = () => finish(Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : undefined)
    audio.onerror = () => finish(undefined)
    audio.src = url
  })
}

const number = (text: string) => (text.trim() === "" ? Number.NaN : Number(text))

/* Music mode: the state of the controls, the song being made, and what each button does.
   Length, steps and language are remembered like the endpoint; the seed and the lyrics are
   not, because a remembered seed would silently make the same song again and lyrics belong
   to one song. The clips are not either: a file chosen once is not offered again unasked. */
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
  /* audio in */
  const [chosenTask, setTask] = useState<MusicTask>("text2music")
  const [source, setSource] = useState<Clip | null>(null)
  const [reference, setReference] = useState<Clip | null>(null)
  const [repaintFrom, setRepaintFrom] = useState("0")
  const [repaintTo, setRepaintTo] = useState("")
  const [repaintMode, setRepaintMode] = useState<RepaintMode>("balanced")
  const [repaintStrength, setRepaintStrength] = useState("0.5")
  const [coverStrength, setCoverStrength] = useState("1")
  const [coverNoise, setCoverNoise] = useState("0")
  /* The files of each turn, by its id, for the buttons that make it again: kept in the page, not in the transcript. */
  const clipsOf = useRef(new Map<string, { source?: File; reference?: File }>())
  const seed = parseSeed(seedText)

  /* A model that does not offer the task chosen (or a model that is not for music) puts it back to composing. */
  const task: MusicTask = mode && offersTask(env.limits, chosenTask) ? chosenTask : "text2music"
  const audioIn = mode && limits.audioIn !== undefined

  useEffect(() => {
    try {
      if (Number.isFinite(duration)) localStorage.setItem("colibri.musicDuration", String(duration))     // NaN: a field being edited
      if (Number.isFinite(steps)) localStorage.setItem("colibri.musicSteps", String(steps))
      localStorage.setItem("colibri.musicLanguage", language)
    } catch { /* restricted storage mode */ }
  }, [duration, steps, language])

  /* Object URLs live until the clip is replaced or the page closes. */
  const urls = useRef(new Set<string>())
  useEffect(() => () => { urls.current.forEach((url) => URL.revokeObjectURL(url)); urls.current.clear() }, [])
  useEffect(() => { if (!mode) { setSource(null); setReference(null) } }, [mode])

  const chooseClip = async (role: "source" | "reference", file: File | undefined) => {
    if (!file) return
    const url = URL.createObjectURL(file)
    urls.current.add(url)
    const set = role === "source" ? setSource : setReference
    set((previous) => { if (previous) { URL.revokeObjectURL(previous.url); urls.current.delete(previous.url) } return { file, name: file.name, url } })
    const seconds = await clipSeconds(url)
    set((current) => current && current.url === url ? { ...current, ...(seconds !== undefined ? { seconds } : {}) } : current)
  }
  const clearClip = (role: "source" | "reference") => {
    const set = role === "source" ? setSource : setReference
    set((previous) => { if (previous) { URL.revokeObjectURL(previous.url); urls.current.delete(previous.url) } return null })
  }

  const sourceProblem: ClipProblem | undefined = source ? clipProblem(source.file, source.seconds, limits) : undefined
  const referenceProblem: ClipProblem | undefined = reference ? clipProblem(reference.file, reference.seconds, limits) : undefined
  const from = number(repaintFrom)
  const to = repaintTo.trim() === "" ? undefined : number(repaintTo)
  const spanOk = task !== "repaint" || repaintValid(from, to, source?.seconds, limits)
  const strengthValue = number(repaintStrength)
  const coverStrengthValue = number(coverStrength)
  const coverNoiseValue = number(coverNoise)
  const unitOk = (value: number) => Number.isFinite(value) && value >= 0 && value <= 1
  const settingsOk = task === "cover" ? unitOk(coverStrengthValue) && unitOk(coverNoiseValue)
    : task === "repaint" ? repaintMode !== "balanced" || unitOk(strengthValue) : true
  const clipsOk = (task === "text2music" || (source !== null && !sourceProblem)) && !referenceProblem

  /* The values the next song is made with: the remembered ones, put inside the range of the
     model that is selected now (another music model may allow less). */
  const effectiveDuration = clampDuration(duration, limits)
  const effectiveSteps = clampMusicSteps(steps, limits)

  const canSend = useMemo(() => mode && Boolean(
    (task !== "text2music" || env.draft.trim() || lyrics.trim()) && env.model && !env.loading && seed !== undefined &&
    (task !== "text2music" || validDuration(duration, limits)) && Number.isInteger(steps) && steps >= 1 && steps <= limits.maxSteps &&
    lyricBytes(lyrics) <= limits.maxLyricBytes && clipsOk && spanOk && settingsOk,
  ), [mode, task, env.draft, env.model, env.loading, lyrics, seed, duration, steps, limits, clipsOk, spanOk, settingsOk])

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
    const files = clipsOf.current.get(targetId)
    try {
      const song = await generateMusic({
        baseUrl: env.baseUrl, apiKey: env.apiKey, model: env.model,
        prompt: request.prompt, lyrics: request.lyrics, language: request.language,
        duration: request.duration, steps: request.steps, seed: request.seed,
        ...(request.task ? { task: request.task } : {}),
        ...(files?.source ? { source: files.source } : {}), ...(files?.reference ? { reference: files.reference } : {}),
        ...(request.repaint ? { repaint: request.repaint } : {}), ...(request.cover ? { cover: request.cover } : {}),
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

  /* What the controls ask for, as the request a turn holds (the clips themselves go to clipsOf). */
  const requestNow = (prompt: string): GeneratedMusic => ({
    prompt, duration: task === "text2music" ? effectiveDuration : (source?.seconds ?? effectiveDuration), steps: effectiveSteps, seed: seed ?? randomSeed(),
    ...(lyrics.trim() ? { lyrics } : {}), ...(language ? { language } : {}),
    ...(task !== "text2music" && source ? { task, source: { name: source.name, ...(source.seconds !== undefined ? { seconds: source.seconds } : {}) } } : {}),
    ...(task === "repaint" ? { repaint: { from, ...(to !== undefined ? { to } : {}), mode: repaintMode, ...(repaintMode === "balanced" ? { strength: clampUnit(strengthValue) } : {}) } } : {}),
    ...(task === "cover" ? { cover: { strength: clampUnit(coverStrengthValue), noise: clampUnit(coverNoiseValue) } } : {}),
    ...(reference ? { reference: { name: reference.name } } : {}),
  })

  const send = async () => {
    const prompt = env.draft.trim()
    if (!canSend || seed === undefined) return
    const request = requestNow(prompt)
    const assistant = turn(request)
    clipsOf.current.set(assistant.id, { ...(task !== "text2music" && source ? { source: source.file } : {}), ...(reference ? { reference: reference.file } : {}) })
    /* the question as it reads in the conversation: the caption, or the first line of the lyrics when there is none, or the clip */
    const asked = prompt || (lyrics.trim() ? `♪ ${lyrics.trim().split("\n")[0]}` : source ? `♪ ${source.name}` : "♪")
    env.setDraft("")
    env.updateMessages((current) => [...current, { ...env.newMessage("user", asked), mode: "music" }, assistant])
    await make(assistant.id, request)
  }

  /* Do the clips a turn was made from still exist? They do for the page's lifetime, not after a reload. */
  const clipsKept = (item: ChatMessage) => {
    const song = item.song
    if (!song || (!song.source && !song.reference)) return true
    const files = clipsOf.current.get(item.id)
    return Boolean(files && (!song.source || files.source) && (!song.reference || files.reference))
  }

  /* Same caption, lyrics, length, steps and clips, a seed never used before: a new song under the
     old one, so the two can be compared. */
  const again = async (item: ChatMessage) => {
    const song = item.song
    if (env.loading || !song) return
    if (!clipsKept(item)) { env.setError("music.clipsLost"); return }
    const { url: _url, seconds: _seconds, timings: _timings, truncated: _truncated, ...request } = song
    const assistant = turn({ ...request, seed: randomSeed() })
    const files = clipsOf.current.get(item.id)
    if (files) clipsOf.current.set(assistant.id, files)
    env.updateMessages((current) => [...current, assistant])
    await make(assistant.id, assistant.song!)
  }

  const retry = async (item: ChatMessage) => {
    if (env.loading || !item.song) return
    if (!clipsKept(item)) { env.setError("music.clipsLost"); return }
    const { url: _url, seconds: _seconds, timings: _timings, truncated: _truncated, ...request } = item.song
    env.updateMessages((current) => current.map((entry) => entry.id === item.id ? { ...entry, finish: undefined, song: request } : entry))
    await make(item.id, request)
  }

  const download = (song: GeneratedMusic) => {
    if (!song.url) return
    const anchor = document.createElement("a")
    anchor.href = song.url
    anchor.download = musicFileName(song.prompt || song.lyrics || song.source?.name || "", song.seed)
    anchor.click()
  }

  const copySeed = async (item: ChatMessage) => {
    if (!item.song) return
    try { await navigator.clipboard.writeText(String(item.song.seed)); setCopied(`${item.id}:seed`) } catch { env.setError("ui.copyError") }
  }

  return {
    mode, limits, duration, steps, language, seedText, lyrics, lyricsOpen, run, copied, seed, canSend,
    task, audioIn, source, reference, sourceProblem, referenceProblem, spanOk,
    repaintFrom, repaintTo, repaintMode, repaintStrength, coverStrength, coverNoise,
    setDuration, setSteps, setLanguage, setSeedText, setLyrics, setTask,
    setRepaintFrom, setRepaintTo, setRepaintMode, setRepaintStrength, setCoverStrength, setCoverNoise,
    chooseClip, clearClip,
    toggleLyrics: () => setLyricsOpen((open) => !open),
    cancel: () => env.abortRef.current?.abort(),
    send, again, retry, download, copySeed,
  }
}
