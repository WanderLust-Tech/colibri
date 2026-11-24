import { useEffect, useMemo, useRef, useState, type MutableRefObject } from "react"

import { generateMusic, type ChatMessage, type GeneratedMusic, type MusicDit, type MusicLimits, type MusicSampler, type MusicTask, type RepaintMode } from "./api"
import { parseSeed, randomSeed } from "./images"
import {
  DEFAULT_MUSIC_LIMITS, clampDuration, clampMusicSteps, clampUnit, clipProblem, guidanceValid, isGuidedTask, lyricBytes, musicFileName, offersTask,
  optionalUnit, optionalValue, VNORM_MAX, repaintValid, runsGuided, stepLimits, validDuration, validLanguage, type ClipProblem, type MusicRun,
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
  /* Make songs as background jobs (`Prefer: respond-async`): the server goes on if the page is closed. */
  background: boolean
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
  /* a flow-edit: what the clip is now, the window of the schedule it edits, the draws averaged a step, and the cover's context */
  const [editCaption, setEditCaption] = useState("")
  const [editLyrics, setEditLyrics] = useState("")
  const [editFrom, setEditFrom] = useState("0")
  const [editTo, setEditTo] = useState("1")
  const [editAvg, setEditAvg] = useState("1")
  const [editCover, setEditCover] = useState(false)
  /* the planner and the guided DiT */
  const [plan, setPlan] = useState(() => read("colibri.musicPlan", "") === "1")
  const [lengthAuto, setLengthAuto] = useState(() => read("colibri.musicLengthAuto", "1") === "1")
  const [chosenDit, setDit] = useState<MusicDit | "">("")
  const [guidedSteps, setGuidedSteps] = useState(() => Number(read("colibri.musicGuidedSteps", "")) || 32)
  const [guidance, setGuidance] = useState("7")
  /* the sampler (any DiT): the step, and the velocity cap and smoothing (empty: off) */
  const [sampler, setSampler] = useState<MusicSampler | "">("")
  const [vnorm, setVnorm] = useState("")
  const [vema, setVema] = useState("")
  const [track, setTrack] = useState("")
  const [completeTracks, setCompleteTracks] = useState<string[]>([])
  /* The files of each turn, by its id, for the buttons that make it again: kept in the page, not in the transcript. */
  const clipsOf = useRef(new Map<string, { source?: File; reference?: File }>())
  const seed = parseSeed(seedText)

  /* A model that does not offer the task chosen (or a model that is not for music) puts it back to composing. */
  const task: MusicTask = mode && offersTask(env.limits, chosenTask) ? chosenTask : "text2music"
  const audioIn = mode && limits.audioIn !== undefined
  /* What the checkpoint has: a planner switch only with a planner, a DiT choice only with a second one. */
  const hasPlanner = mode && limits.planner !== undefined
  const dits = mode && limits.guided ? limits.guided.dits : []
  const dit: MusicDit | "" = chosenDit !== "" && dits.includes(chosenDit) && chosenDit !== "turbo" ? chosenDit : ""
  const guided = mode && limits.guided !== undefined && runsGuided(task, dit)
  const planning = hasPlanner && plan && task === "text2music"
  const auto = planning && lengthAuto
  const activeLimits = stepLimits(limits, guided)
  const activeSteps = guided ? guidedSteps : steps
  const setActiveSteps = guided ? setGuidedSteps : setSteps
  const guidanceValue = number(guidance)
  const hasSampler = mode && limits.sampler === true
  const vnormOk = optionalUnit(vnorm, VNORM_MAX)
  const vemaOk = optionalUnit(vema, 1)
  const samplerOk = !hasSampler || (vnormOk && vemaOk)

  useEffect(() => {
    try {
      if (Number.isFinite(duration)) localStorage.setItem("colibri.musicDuration", String(duration))     // NaN: a field being edited
      if (Number.isFinite(steps)) localStorage.setItem("colibri.musicSteps", String(steps))
      if (Number.isFinite(guidedSteps)) localStorage.setItem("colibri.musicGuidedSteps", String(guidedSteps))
      localStorage.setItem("colibri.musicLanguage", language)
      localStorage.setItem("colibri.musicPlan", plan ? "1" : "")
      localStorage.setItem("colibri.musicLengthAuto", lengthAuto ? "1" : "")
    } catch { /* restricted storage mode */ }
  }, [duration, steps, guidedSteps, language, plan, lengthAuto])

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
  const spanOk = (task !== "repaint" && task !== "lego") || repaintValid(from, to, source?.seconds, limits)
  const strengthValue = number(repaintStrength)
  const coverStrengthValue = number(coverStrength)
  const coverNoiseValue = number(coverNoise)
  const unitOk = (value: number) => Number.isFinite(value) && value >= 0 && value <= 1
  const editFromValue = number(editFrom), editToValue = number(editTo), editAvgValue = number(editAvg)
  const editOk = unitOk(editFromValue) && unitOk(editToValue) && editToValue > 0 && editToValue >= editFromValue &&
    Number.isInteger(editAvgValue) && editAvgValue >= 1 && editAvgValue <= 16
  const hasEditCover = mode && limits.flowEditCover === true
  const settingsOk = task === "cover" ? unitOk(coverStrengthValue) && unitOk(coverNoiseValue)
    : task === "edit" ? editOk
    : task === "repaint" ? repaintMode !== "balanced" || unitOk(strengthValue) : true
  /* A completion may start from the description alone; the other clip tasks need the clip. */
  const clipsOk = (task === "text2music" || task === "complete" ? !source || !sourceProblem : source !== null && !sourceProblem) && !referenceProblem
  const needsLength = (task === "text2music" && !auto) || (task === "complete" && !source)
  const guidedOk = (!guided || (guidanceValid(guidanceValue) && (task !== "complete" || completeTracks.length > 0))) && samplerOk

  /* The values the next song is made with: the remembered ones, put inside the range of the
     model that is selected now (another music model may allow less). */
  const effectiveDuration = clampDuration(duration, limits)
  const effectiveSteps = clampMusicSteps(activeSteps, activeLimits)

  const canSend = useMemo(() => mode && Boolean(
    (task !== "text2music" || env.draft.trim() || lyrics.trim()) && env.model && !env.loading && seed !== undefined &&
    (!needsLength || validDuration(duration, limits)) && Number.isInteger(activeSteps) && activeSteps >= 1 && activeSteps <= activeLimits.maxSteps &&
    lyricBytes(lyrics) <= limits.maxLyricBytes && clipsOk && spanOk && settingsOk && guidedOk,
  ), [mode, task, env.draft, env.model, env.loading, lyrics, seed, duration, activeSteps, activeLimits, limits, clipsOk, spanOk, settingsOk, needsLength, guidedOk])

  /* Generates into the assistant turn `targetId`, whose `song` already holds the request.
     Progress lives in `run`, outside the transcript; the finished song is written onto the
     turn. Cancel is the same abort controller as the chat's Stop. */
  const make = async (targetId: string, request: GeneratedMusic) => {
    env.setError("")
    env.setLoading(true)
    const controller = new AbortController()
    env.abortRef.current = controller
    const started = performance.now()
    setRun({ targetId, started: Date.now(), progress: null, ...(env.background ? { background: true } : {}) })
    const update = (change: Partial<MusicRun>) => setRun((current) => current && current.targetId === targetId ? { ...current, ...change } : current)
    const files = clipsOf.current.get(targetId)
    try {
      const song = await generateMusic({
        baseUrl: env.baseUrl, apiKey: env.apiKey, model: env.model,
        prompt: request.prompt, lyrics: request.lyrics, language: request.language,
        duration: request.duration, steps: request.steps, seed: request.seed,
        ...(request.task ? { task: request.task } : {}),
        ...(request.plan ? { plan: true } : {}), ...(request.auto ? { auto: true } : {}), ...(request.dit ? { dit: request.dit } : {}),
        ...(request.guidance !== undefined ? { guidance: request.guidance } : {}),
        ...(request.sampler ? { sampler: request.sampler } : {}), ...(request.vnorm ? { vnorm: request.vnorm } : {}), ...(request.vema ? { vema: request.vema } : {}),
        ...(request.track ? { track: request.track } : {}), ...(request.tracks?.length ? { tracks: request.tracks } : {}),
        ...(files?.source ? { source: files.source } : {}), ...(files?.reference ? { reference: files.reference } : {}),
        ...(request.repaint ? { repaint: request.repaint } : {}), ...(request.cover ? { cover: request.cover } : {}), ...(request.edit ? { edit: request.edit } : {}),
        signal: controller.signal,
        onProgress: (progress) => update({ progress }),
        onPlainRequest: () => update({ plain: true }),
        background: env.background,
        onJob: (job) => update({ queue: job.status === "queued" ? job.position : undefined }),
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
    prompt, duration: task === "text2music" || (task === "complete" && !source) ? effectiveDuration : (source?.seconds ?? effectiveDuration), steps: effectiveSteps, seed: seed ?? randomSeed(),
    ...(planning ? { plan: true, ...(auto ? { auto: true } : {}) } : {}),
    ...(dit ? { dit } : {}),
    ...(guided ? { guidance: guidanceValue } : {}),
    ...(hasSampler && sampler && task !== "edit" ? { sampler } : {}),            // a flow-edit takes the plain Euler step
    ...(hasSampler && optionalValue(vnorm) > 0 ? { vnorm: optionalValue(vnorm) } : {}),
    ...(hasSampler && optionalValue(vema) > 0 ? { vema: optionalValue(vema) } : {}),
    ...((task === "extract" || task === "lego") && track ? { track } : {}),
    ...(task === "complete" && completeTracks.length ? { tracks: completeTracks } : {}),
    ...(lyrics.trim() ? { lyrics } : {}), ...(language ? { language } : {}),
    ...(task !== "text2music" ? { task } : {}),
    ...(task !== "text2music" && source ? { source: { name: source.name, ...(source.seconds !== undefined ? { seconds: source.seconds } : {}) } } : {}),
    ...(task === "repaint" ? { repaint: { from, ...(to !== undefined ? { to } : {}), mode: repaintMode, ...(repaintMode === "balanced" ? { strength: clampUnit(strengthValue) } : {}) } } : {}),
    ...(task === "lego" ? { repaint: { from, ...(to !== undefined ? { to } : {}), mode: "balanced" as RepaintMode } } : {}),
    ...(task === "cover" ? { cover: { strength: clampUnit(coverStrengthValue), noise: clampUnit(coverNoiseValue) } } : {}),
    ...(task === "edit" ? { edit: {
      caption: editCaption.trim(), ...(editLyrics.trim() ? { lyrics: editLyrics } : {}),
      ...(editFromValue > 0 ? { nMin: editFromValue } : {}), ...(editToValue !== 1 ? { nMax: editToValue } : {}), ...(editAvgValue !== 1 ? { nAvg: editAvgValue } : {}),
      ...(hasEditCover && editCover ? { cover: true } : {}),
    } } : {}),
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
    const { url: _url, seconds: _seconds, timings: _timings, truncated: _truncated, meta: _meta, planCodes: _codes, ranDit: _ran, ...request } = song
    const assistant = turn({ ...request, seed: randomSeed() })
    const files = clipsOf.current.get(item.id)
    if (files) clipsOf.current.set(assistant.id, files)
    env.updateMessages((current) => [...current, assistant])
    await make(assistant.id, assistant.song!)
  }

  const retry = async (item: ChatMessage) => {
    if (env.loading || !item.song) return
    if (!clipsKept(item)) { env.setError("music.clipsLost"); return }
    const { url: _url, seconds: _seconds, timings: _timings, truncated: _truncated, meta: _meta, planCodes: _codes, ranDit: _ran, ...request } = item.song
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
    mode, limits, duration, steps: activeSteps, stepLimits: activeLimits, language, seedText, lyrics, lyricsOpen, run, copied, seed, canSend,
    task, audioIn, source, reference, sourceProblem, referenceProblem, spanOk,
    repaintFrom, repaintTo, repaintMode, repaintStrength, coverStrength, coverNoise,
    editCaption, editLyrics, editFrom, editTo, editAvg, editCover, editOk, hasEditCover,
    setEditCaption, setEditLyrics, setEditFrom, setEditTo, setEditAvg, setEditCover,
    hasPlanner, plan, planning, lengthAuto, auto, dits, dit, guided, guidance, guidanceOk: guidanceValid(guidanceValue), track, completeTracks, needsLength,
    hasSampler, sampler, vnorm, vema, vnormOk, vemaOk, setSampler, setVnorm, setVema,
    setPlan, setLengthAuto, setDit, setGuidance, setTrack, setCompleteTracks,
    setDuration, setSteps: setActiveSteps, setLanguage, setSeedText, setLyrics, setTask,
    setRepaintFrom, setRepaintTo, setRepaintMode, setRepaintStrength, setCoverStrength, setCoverNoise,
    chooseClip, clearClip,
    toggleLyrics: () => setLyricsOpen((open) => !open),
    openLyrics: () => setLyricsOpen(true),
    cancel: () => env.abortRef.current?.abort(),
    send, again, retry, download, copySeed,
  }
}
