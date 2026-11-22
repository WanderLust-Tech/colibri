import { useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  ArrowUp,
  BrainCircuit,
  CircleStop,
  Clock,
  Cpu,
  Database,
  Search, SquarePen, History, Settings2, Sun, Moon, X, Download, Copy, ChevronDown, Code2,
  Gauge,
  Globe,
  HardDrive,
  KeyRound,
  AlertTriangle,
  Layers,
  Link2,
  LoaderCircle,
  MemoryStick,
  MessageSquareText,
  MonitorDot,
  RefreshCw,
  StepForward,
  SlidersHorizontal,
  Timer,
  Trash2,
  Zap, ImagePlus, Pencil, GitBranch, Minimize2, Lightbulb, Camera, Grid3x3, Palette, Music, Waves, Guitar, FileInput, CloudUpload} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { dashboardTags, newChatTag } from "@/lib/tags"
import {
  editsImages, generateImage, generatesImages, generatesMusic, getHealth, listModelInfo, makesEmbeddings, ranksDocuments, setClientTags, streamChat,
  type ChatMessage, type GeneratedImage, type HealthResponse, type ModelInfo, type MusicLimits, type PresetInfo, type StreamChatResult,
} from "@/lib/api"
import { controlDefaults } from "@/lib/presets"
import {
  DEFAULT_SIZE, DEFAULT_STEPS, chatTurns, clampSteps, dataUrlBlob, derivedSize, imageFileName, loadPicture, parseSeed,
  parseSize, presetOf, randomSeed, sizeKey, validSide, type EditPicture, type ImageSize,
} from "@/lib/images"
import { branchAt, editFrom, isPlainTurn, resendFrom, REWRITES, type Rewrite } from "@/lib/chat"
import { DEFAULT_MUSIC_LIMITS } from "@/lib/music"
import { readSettings, refillFrom, type MediaSettings } from "@/lib/mediaMeta"
import { useMusic } from "@/lib/useMusic"
import { activeRequests, settingsPages, supportsCacheSlots, supportsContinuation } from "@/lib/runtime"
import Brio from "./Brio"
import { BrainWorkspace } from "./BrainWorkspace"
import { Brand } from "./components/Brand"
import { ImageControls, ImageProgressCard, ImageResultCard, ImageStoppedCard, type ImageRun } from "./components/ImageTurn"
import { MusicClips, MusicControls, MusicLyrics, MusicProgressCard, MusicResultCard, MusicStoppedCard } from "./components/MusicTurn"
import { Markdown } from "./components/Markdown"
import { NavigationDock, type View } from "./components/NavigationDock"
import { Profiling } from "./Profiling"
import { persistPublicSettings, stored, takeKeyFromLocation } from "@/lib/storage"
import { appendDelta, continuable, continuation, setFinish, setStats } from "@/lib/transcript"
import { buildReplyStats, replyParts } from "@/lib/replyStats"
import { explainError } from "@/lib/explainError"
import { cn } from "@/lib/utils"
import ModelSwitcher from "@/components/ModelSwitcher"
import PresetsPanel from "@/components/PresetsPanel"
import VoicesPanel from "@/components/VoicesPanel"
import VoiceControls from "@/components/VoiceControls"
import { useVoice } from "@/lib/useVoice"
import { pickVoiceModels } from "@/lib/voice"
import LibraryPanel from "@/components/LibraryPanel"
import LogPanel from "@/components/LogPanel"
import RequestsPanel from "@/components/RequestsPanel"
import { Compare } from "@/components/Compare"
import { isChatModel } from "@/lib/compare"
import { useLocale } from "./i18n"

const message = (role: ChatMessage["role"], content: string, images?: string[]): ChatMessage => {
  let id: string
  try { id = crypto.randomUUID() } catch { id = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => { const r = Math.random() * 16 | 0; return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16) }) }
  return images?.length ? { id, role, content, images } : { id, role, content }
}

/* Read once when the page loads, so a development double render cannot lose it. */
const linkKey = typeof window !== "undefined" ? takeKeyFromLocation(window.location, window.history) : ""

export default function App() {
  const { t, locale, setLocale, locales } = useLocale()

  const servedByEngine = typeof window !== "undefined" && window.location.port !== "5173" && window.location.protocol.startsWith("http")
  const defaultBase = servedByEngine ? `${window.location.origin}/v1` : "http://127.0.0.1:8000/v1"
  const [baseUrl, setBaseUrl] = useState(() => {
    const saved = stored(localStorage, "colibri.baseUrl", defaultBase)
    if (servedByEngine && saved === "http://127.0.0.1:8000/v1" && defaultBase !== saved) return defaultBase
    return saved
  })
  const [apiKey, setApiKey] = useState(linkKey)
  const [models, setModels] = useState<string[]>([])
  const [modelInfos, setModelInfos] = useState<ModelInfo[]>([])
  const chatModelInfos = useMemo(() => modelInfos.filter(isChatModel), [modelInfos])
  /* Models the server marks with the image_generation capability. With one of
     them selected the conversation generates pictures instead of text. */
  const [imageModels, setImageModels] = useState<string[]>([])
  /* Of those, the ones whose engine also edits pictures (the image_edit capability),
     with what /v1/images/edits accepts. */
  const [editLimits, setEditLimits] = useState<Record<string, NonNullable<ModelInfo["edit"]>>>({})
  /* Models marked music_generation, with the limits of their /v1/models `audio` block. With one of
     them selected the conversation makes songs instead of text. */
  const [musicModels, setMusicModels] = useState<Record<string, MusicLimits>>({})
  const [model, setModel] = useState(() => stored(localStorage, "colibri.model", "glm-5.2-colibri"))
  /* The models that are presets of `coli router`: what each runs and the settings it fills in (docs/presets.md). */
  const [presetInfos, setPresetInfos] = useState<Record<string, PresetInfo>>({})
  /* Each model's window in tokens, from its card, for the share of it a reply used. */
  const [contextLengths, setContextLengths] = useState<Record<string, number>>({})
  /* The model whose preset settings the controls were last started from: they follow a change of model, not every reconnect. */
  const presetApplied = useRef("")
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(4096)
  const [thinking, setThinking] = useState(false)
  const [cacheSlot, setCacheSlot] = useState(0)
  const [conversations, setConversations] = useState<Record<number, ChatMessage[]>>({ 0: [] })
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState("")
  const [lastRun, setLastRun] = useState<StreamChatResult | null>(null)
  const [draft, setDraft] = useState("")
  const [loading, setLoading] = useState(false)
  const [streamStart, setStreamStart] = useState<number | null>(null)
  const [tokenCount, setTokenCount] = useState(0)
  const [tokPerSec, setTokPerSec] = useState<number | null>(null)
  const [ttft, setTtft] = useState<number | null>(null)
  const [totalTokens, setTotalTokens] = useState({ prompt: 0, completion: 0 })
  const [connecting, setConnecting] = useState(false)
  const [connected, setConnected] = useState(false)
  const [view, setView] = useState<View>("chat")
  const [settingsPage, setSettingsPage] = useState("general")
  /* Pictures and songs as background jobs: the router goes on with them if this page is closed. Offered only where the server says it can. */
  const [background, setBackground] = useState(() => stored(localStorage, "colibri.background", "") === "1")
  const canBackground = health?.features?.jobs === true
  const pages = settingsPages(health)                                   // a plain `coli serve` has no Presets, Requests or Log
  const activePage = pages.includes(settingsPage) ? settingsPage : "general"
  const [historyOpen, setHistoryOpen] = useState(false)
  const [query, setQuery] = useState("")
  const [archives, setArchives] = useState<Array<{ id: string; slot: number; messages: ChatMessage[] }>>([])
  const [theme, setTheme] = useState(() => stored(localStorage, "colibri.theme", "dark"))
  const [copied, setCopied] = useState<string | null>(null)
  const draftRef = useRef<HTMLTextAreaElement>(null)
  /* What this conversation is called to the router's request log and jobs (X-Colibri-Tags): one tag per conversation. */
  const chatTag = useRef(newChatTag())
  useEffect(() => { setClientTags(dashboardTags(chatTag.current)) }, [])
  const fileRef = useRef<HTMLInputElement>(null)
  /* Pictures wait here until the turn is sent, then travel on the message. */
  const [pending, setPending] = useState<string[]>([])
  /* Size and steps are remembered like the endpoint; the seed is not, because
     a remembered seed would silently draw the same picture again. */
  const [imageSize, setImageSize] = useState<ImageSize>(() => parseSize(stored(localStorage, "colibri.imageSize", "")) || DEFAULT_SIZE)
  const [customSize, setCustomSize] = useState(() => !presetOf(parseSize(stored(localStorage, "colibri.imageSize", "")) || DEFAULT_SIZE))
  const [imageSteps, setImageSteps] = useState(() => clampSteps(Number(stored(localStorage, "colibri.imageSteps", String(DEFAULT_STEPS)))))
  const [seedText, setSeedText] = useState("")
  const [imageRun, setImageRun] = useState<ImageRun | null>(null)
  /* Pictures attached to the next image, which then edits them, and whether the
     size follows the last one. Kept after sending: refining an edit is the
     common next step. */
  const [pictures, setPictures] = useState<EditPicture[]>([])
  const [sizeAuto, setSizeAuto] = useState(true)
  const editRef = useRef<HTMLInputElement>(null)
  const settingsRef = useRef<HTMLInputElement>(null)
  /* The files an edit was made from, by the turn that shows its result: Again and
     Try again need them, and nothing a reload could bring back holds them. */
  const editFiles = useRef(new Map<string, File[]>())
  const historyDialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { document.documentElement.dataset.theme = theme; try { localStorage.setItem("colibri.theme", theme) } catch {} }, [theme])
  useEffect(() => { if (historyOpen) historyDialog.current?.showModal(); else historyDialog.current?.close() }, [historyOpen])
  const [error, setError] = useState("")
  const autoConnected = useRef(false)
  const abortRef = useRef<AbortController | null>(null)
  const probeRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const messages = conversations[cacheSlot] || []
  const kvSlots = Math.max(1, health?.kv_slots || 1)
  const active = activeRequests(health)
  const capacity = health?.scheduler?.capacity || kvSlots
  const failures = health?.scheduler ? health.scheduler.rejected + health.scheduler.timed_out + health.scheduler.cancelled : 0
  const imageMode = imageModels.includes(model)
  const editLimit = imageMode ? editLimits[model] : undefined
  const canEdit = Boolean(editLimit)
  const attached = canEdit ? pictures : []
  const seed = parseSeed(seedText)

  const updateMessages = (next: ChatMessage[] | ((current: ChatMessage[]) => ChatMessage[])) =>
    setConversations((current) => ({
      ...current,
      [cacheSlot]: typeof next === "function" ? next(current[cacheSlot] || []) : next,
    }))

  const music = useMusic({
    baseUrl, apiKey, model, limits: musicModels[model], draft, setDraft, loading, setLoading, abortRef,
    updateMessages, setError, setConnected, newMessage: message, background: background && canBackground,
  })

  // EFFECT #1
  useEffect(() => {
    persistPublicSettings(localStorage, baseUrl, model)
  }, [baseUrl, model])
  useEffect(() => {
    try { localStorage.setItem("colibri.background", background ? "1" : "") } catch { /* restricted storage mode */ }
  }, [background])
  useEffect(() => {
    try { localStorage.setItem("colibri.imageSize", sizeKey(imageSize)); localStorage.setItem("colibri.imageSteps", String(imageSteps)) } catch { /* restricted storage mode */ }
  }, [imageSize, imageSteps])

  // EFFECT #2
  useEffect(() => {
    setConnected(false)
    setHealth(null)
    setHealthError("")
  }, [baseUrl, apiKey])

  // EFFECT #3
  useEffect(() => () => {
    probeRef.current?.abort()
    abortRef.current?.abort()
  }, [])

  // EFFECT #4
  useEffect(() => {
    if (!connected) return
    let disposed = false
    const poll = async () => {
      if (document.visibilityState === "hidden") return
      try {
        const result = await getHealth(baseUrl, apiKey)
        if (!disposed) { setHealth(result); setHealthError("") }
      } catch (cause) {
        if (!disposed) setHealthError(cause instanceof Error ? cause.message : "status.runtimeUnavailable")
      }
    }
    const timer = window.setInterval(() => void poll(), 5000)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [apiKey, baseUrl, connected])

  /* Choosing a preset starts the controls from its settings (a request that carries its own value would override the preset's,
     and this page always sends temperature and the length): the temperature and length of a chat, the size and steps of a picture. */
  const startFromPreset = (name: string) => {
    presetApplied.current = name
    const preset = presetInfos[name]
    if (!preset) return
    const start = controlDefaults(preset.defaults)
    if (start.temperature !== undefined) setTemperature(start.temperature)
    if (start.maxTokens !== undefined) setMaxTokens(start.maxTokens)
    if (start.size) { setImageSize(start.size); setCustomSize(!presetOf(start.size)); setSizeAuto(false) }
    if (start.steps !== undefined) setImageSteps(clampSteps(start.steps))
  }
  useEffect(() => {
    if (connected && presetApplied.current !== model) startFromPreset(model)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected, model, presetInfos])

  // EFFECT #5
  useEffect(() => {
    if (cacheSlot >= kvSlots) setCacheSlot(0)
  }, [cacheSlot, kvSlots])

  // EFFECT #6
  useEffect(() => { setLastRun(null) }, [cacheSlot])

  // EFFECT #7
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }) }, [messages])

  const connect = async () => {
    probeRef.current?.abort()
    const controller = new AbortController()
    probeRef.current = controller
    setConnecting(true)
    setError("")
    try {
      const info = await listModelInfo(baseUrl, apiKey, controller.signal)
      const found = info.filter((item) => !makesEmbeddings(item) && !ranksDocuments(item)).map((item) => item.id)
      setModels(found)
      setModelInfos(info)
      setPresetInfos(Object.fromEntries(info.flatMap((item) => item.preset ? [[item.id, item.preset] as const] : [])))
      setContextLengths(Object.fromEntries(info.flatMap((item) => item.contextLength ? [[item.id, item.contextLength] as const] : [])))
      setImageModels(info.filter(generatesImages).map((item) => item.id))
      setEditLimits(Object.fromEntries(info.flatMap((item) => editsImages(item) && item.edit ? [[item.id, item.edit] as const] : [])))
      setMusicModels(Object.fromEntries(info.filter(generatesMusic).map((item) => [item.id, item.music ?? DEFAULT_MUSIC_LIMITS] as const)))
      if (found.length && !found.includes(model)) setModel(found[0])
      setConnected(true)
      try {
        setHealth(await getHealth(baseUrl, apiKey, controller.signal))
        setHealthError("")
      } catch (cause) {
        if (!controller.signal.aborted) {
          setHealth(null)
          setHealthError(cause instanceof Error ? cause.message : "status.runtimeUnavailable")
        }
      }
    } catch (cause) {
      if (controller.signal.aborted) return
      setConnected(false)
      setError(cause instanceof Error ? cause.message : "status.serverError")
    } finally {
      if (probeRef.current === controller) { probeRef.current = null; setConnecting(false) }
    }
  }

  if (servedByEngine && !autoConnected.current && !connected) {
    autoConnected.current = true
    setTimeout(() => connect(), 0)
  }

  /* A picture becomes a data URI here and nowhere else: the API takes one
     directly, so there is no upload endpoint, no temporary file and nothing to
     clean up. Anything that is not an image is ignored rather than refused --
     a dropped PDF should not interrupt what someone was typing. */
  const attach = async (files: FileList | File[] | null) => {
    const pictures = Array.from(files || []).filter(file => file.type.startsWith("image/"))
    if (!pictures.length) return
    const read = (file: File) => new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result))
      reader.onerror = () => reject(reader.error)
      reader.readAsDataURL(file)
    })
    try {
      const urls = await Promise.all(pictures.map(read))
      setPending(current => [...current, ...urls])
    } catch { setError("status.serverError") }
  }

  /* Pictures for an edit stay files: the request is a multipart upload, so there is
     no data URI to make. Anything that is not an image is ignored, as in the chat. */
  const attachPictures = async (files: FileList | File[] | null) => {
    const chosen = Array.from(files || []).filter((file) => file.type.startsWith("image/"))
    if (!chosen.length || !editLimit) return
    if (pictures.length + chosen.length > editLimit.maxImages) { setError("image.tooManyPictures"); return }
    try {
      const loaded = await Promise.all(chosen.map(loadPicture))
      setPictures((current) => [...current, ...loaded])
      setSizeAuto(true)
      setError("")
    } catch { setError("status.serverError") }
  }
  const removePicture = (id: string) => setPictures((current) => {
    const gone = current.find((picture) => picture.id === id)
    if (gone) URL.revokeObjectURL(gone.url)
    return current.filter((picture) => picture.id !== id)
  })

  const canSend = useMemo(() => music.mode ? music.canSend : imageMode
    ? Boolean(draft.trim() && model && !loading && seed !== undefined && ((attached.length && sizeAuto) || (validSide(imageSize.width) && validSide(imageSize.height))))
    : (draft.trim() || pending.length) && model && !loading, [attached.length, draft, imageMode, imageSize, loading, model, music.mode, music.canSend, pending, seed, sizeAuto])

  /* The shared streaming core. POSTs `payload` and appends every delta into the
     message `targetId`, with the same live metrics and abort/error wiring whether
     the turn is a fresh answer (send) or a continued one (continueTurn). The only
     difference between the two callers is what they put in `payload` and which
     bubble they stream into; everything below is identical, so it lives here. */
  const runStream = async (payload: ChatMessage[], targetId: string) => {
    setError("")
    setLoading(true)
    setStreamStart(null)
    setTokenCount(0)
    let decodeStart = 0
    setTokPerSec(null)
    setTtft(null)
    const t0 = performance.now()
    let firstToken = true
    let firstTokenMs: number | null = null
    let count = 0
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const result = await streamChat({
        baseUrl,
        apiKey,
        model,
        messages: chatTurns(payload),
        temperature,
        maxTokens,
        enableThinking: thinking,
        cacheSlot: supportsCacheSlots(health) ? cacheSlot : undefined,
        signal: controller.signal,
        /* Reasoning tokens are tokens: they count toward the rate, and the
           first one is the real time-to-first-token. The answer's first token
           arrives much later on a reasoning model. */
        onReasoning: (delta) => {
          if (firstToken) { firstTokenMs = performance.now() - t0; setTtft(firstTokenMs); setStreamStart(performance.now()); decodeStart = performance.now(); firstToken = false }
          count++
          setTokenCount(count)
          const since = (performance.now() - decodeStart) / 1000
          if (count > 1 && since > 0.2) setTokPerSec((count - 1) / since)
          updateMessages((current) => appendDelta(current, targetId, "reasoning", delta))
        },
        onDelta: (delta) => {
          if (firstToken) { firstTokenMs = performance.now() - t0; setTtft(firstTokenMs); setStreamStart(performance.now()); decodeStart = performance.now(); firstToken = false }
          count++
          setTokenCount(count)
          /* The live rate is DECODE speed: measured from the first token, not
             from the request, because the prefill wait belongs to TTFT and
             averaging it in makes the number crawl upward for the whole turn
             instead of showing what the engine is doing now. */
          const since = (performance.now() - decodeStart) / 1000
          if (count > 1 && since > 0.2) setTokPerSec((count - 1) / since)
          updateMessages((current) => appendDelta(current, targetId, "content", delta))
        },
      })
      /* The turn's own figure keeps the same meaning as the live one, so the
         badge does not jump to a different number the moment it stops moving. */
      const decodeElapsed = (performance.now() - (decodeStart || t0)) / 1000
      if (count > 1 && decodeElapsed > 0) setTokPerSec((count - 1) / decodeElapsed)
      if (result.usage) setTotalTokens(prev => ({
        prompt: prev.prompt + (result.usage?.prompt_tokens || 0),
        completion: prev.completion + (result.usage?.completion_tokens || 0),
      }))
      setLastRun(result)
      /* The line under the reply: what it cost, whether the engine read the prompt again, and which model answered
         (behind a preset it is not the one in the picker). The window comes from the model that answered. */
      const stats = buildReplyStats({
        usage: result.usage,
        model: result.model,
        preset: presetInfos[model] ? model : undefined,
        tokPerSec: count > 1 && decodeElapsed > 0 ? (count - 1) / decodeElapsed : null,
        ttftMs: firstTokenMs,
        elapsedMs: performance.now() - t0,
        contextLength: (result.model ? contextLengths[result.model] : undefined) ?? contextLengths[model],
      })
      updateMessages((current) => setStats(setFinish(current, targetId, result.finishReason), targetId, stats))
      setConnected(true)
    } catch (cause) {
      /* Drop the target bubble only if it is empty -- a first turn that never got
         a token. A continued turn already carries the client's opening, so the
         `|| item.content` keeps it on the screen through an abort or an error, and
         `|| item.reasoning` keeps a turn stopped while it was still thinking. */
      const finish = controller.signal.aborted ? "aborted" : "error"
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "status.generationFailed")
      updateMessages((current) => setFinish(current, targetId, finish)
        .filter((item) => item.id !== targetId || item.content || item.reasoning))
    } finally {
      abortRef.current = null
      setLoading(false)
    }
  }

  /* consumeDraft is false for regenerate so a follow-up already in the composer is not eaten. */
  const send = async (text = draft, previous = messages, pictures = pending, consumeDraft = true) => {
    const content = text.trim()
    if ((!content && !pictures.length) || loading) return
    const user = message("user", content, pictures)
    const assistant = message("assistant", "")
    const history = [...previous, user]
    if (consumeDraft) {
      setDraft("")
      setPending([])
    }
    updateMessages([...history, assistant])
    await runStream(history, assistant.id)
  }

  /* Continue the trailing assistant turn instead of opening a new one. Offered
     only when /health reports continue_assistant: with COLI_CONTINUE_ASSISTANT=0,
     or on a server that predates the field, the turn would be answered fresh. */
  const continueTurn = async () => {
    if (loading) return
    const request = continuation(messages)
    if (!request) return
    updateMessages(request.history)
    await runStream(request.history, request.targetId)
  }

  /* The image counterpart of runStream: generates into the assistant turn
     `targetId`, whose `generated` field already holds the request. Progress and
     previews live in imageRun, outside the transcript; the finished picture is
     written onto the turn. Cancel is the same abortRef as the chat's Stop. */
  const runImage = async (targetId: string, request: GeneratedImage, input?: { files: File[]; auto: boolean }) => {
    const inBackground = background && canBackground
    setError("")
    setLoading(true)
    const controller = new AbortController()
    abortRef.current = controller
    const started = performance.now()
    setImageRun({ targetId, started: Date.now(), progress: null, preview: null, ...(inBackground ? { background: true } : {}) })
    const update = (change: Partial<ImageRun>) => setImageRun((run) => run && run.targetId === targetId ? { ...run, ...change } : run)
    try {
      const image = await generateImage({
        baseUrl, apiKey, model, ...request,
        /* an edit: the pictures go up, and an automatic size is left to the engine */
        ...(input ? { images: input.files, ...(input.auto ? { width: undefined, height: undefined } : {}) } : {}),
        signal: controller.signal,
        onProgress: (progress) => update({ progress }),
        onPartial: (preview) => update({ preview }),
        onPlainRequest: () => update({ plain: true }),
        background: inBackground,
        onJob: (job) => update({ queue: job.status === "queued" ? job.position : undefined }),
      })
      const seconds = (performance.now() - started) / 1000
      updateMessages((current) => current.map((item) => item.id === targetId
        ? { ...item, finish: "stop", generated: { ...image, seconds, ...(request.edit ? { edit: request.edit } : {}) } } : item))
      setConnected(true)
    } catch (cause) {
      const finish = controller.signal.aborted ? "aborted" : "error"
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "status.generationFailed")
      updateMessages((current) => setFinish(current, targetId, finish))
    } finally {
      abortRef.current = null
      setImageRun(null)
      setLoading(false)
    }
  }

  const imageTurn = (request: GeneratedImage): ChatMessage => ({ ...message("assistant", ""), mode: "image", generated: request })

  const sendImage = async () => {
    const prompt = draft.trim()
    if (!prompt || loading || seed === undefined) return
    const files = attached.map((picture) => picture.file)
    /* With pictures attached and the size on auto, the engine picks the size from the last
       one; the placeholder is shaped by the same rule, and falls back to the chosen size
       for a picture too long and thin for it. */
    const last = attached[attached.length - 1]
    const derived = files.length && sizeAuto && last ? derivedSize(last.width, last.height, editLimit?.refResolution?.default) : null
    const guess = derived && validSide(derived.width) && validSide(derived.height) ? derived : null
    const size = guess ?? imageSize
    const request = { prompt, width: size.width, height: size.height, steps: imageSteps, seed: seed ?? randomSeed(), ...(files.length ? { edit: files.length } : {}) }
    const assistant = imageTurn(request)
    setDraft("")
    updateMessages((current) => [...current, { ...message("user", prompt), mode: "image" }, assistant])
    if (files.length) editFiles.current.set(assistant.id, files)
    await runImage(assistant.id, request, files.length ? { files, auto: guess !== null } : undefined)
  }

  /* Same prompt, size and steps, a seed never used before: a new picture
     under the old one, so the two can be compared. */
  const imageAgain = async (image: GeneratedImage, sourceId: string) => {
    if (loading) return
    const files = image.edit ? editFiles.current.get(sourceId) : undefined
    if (image.edit && !files) { setError("image.editInputsGone"); return }
    const { prompt, width, height, steps } = image
    const assistant = imageTurn({ prompt, width, height, steps, seed: randomSeed(), ...(image.edit ? { edit: image.edit } : {}) })
    if (files) editFiles.current.set(assistant.id, files)
    updateMessages((current) => [...current, assistant])
    await runImage(assistant.id, assistant.generated!, files ? { files, auto: false } : undefined)
  }

  const imageRetry = async (item: ChatMessage) => {
    if (loading || !item.generated) return
    const { prompt, width, height, steps, seed: previous, edit } = item.generated
    const files = edit ? editFiles.current.get(item.id) : undefined
    if (edit && !files) { setError("image.editInputsGone"); return }
    const request = { prompt, width, height, steps, seed: previous, ...(edit ? { edit } : {}) }
    updateMessages((current) => current.map((turn) => turn.id === item.id ? { ...turn, finish: undefined, generated: request } : turn))
    await runImage(item.id, request, files ? { files, auto: false } : undefined)
  }

  /* A saved picture or song knows how it was made (the server writes the settings into the file): opening one puts the
     model, prompt and controls back as they were, so the same render can be made again or varied. Dropping the file is
     taken by attaching it, so this is its own button. */
  const openSettingsFile = async (file: File | undefined) => {
    if (!file) return
    const settings = await readSettings(file)
    if (!settings) { setError("media.noSettings"); return }
    applySettings(settings)
  }

  /* The settings of a file, or of a card in the Library, put back in the composer. */
  const applySettings = (settings: MediaSettings) => {
    const plan = refillFrom(settings, imageModels, musicModels)
    if ("error" in plan) { setError(t(plan.error, { kind: t(settings.kind === "image" ? "media.kindImage" : "media.kindSong") })); return }
    setError("")
    setModel(plan.model)
    setDraft(plan.prompt)
    if (settings.kind === "image") {
      setSeedText(plan.seed ?? "")
      if (plan.steps !== undefined) setImageSteps(plan.steps)
      if (plan.size) { setImageSize(plan.size); setCustomSize(!presetOf(plan.size)); setSizeAuto(false) }
    } else {
      music.setSeedText(plan.seed ?? "")
      if (plan.steps !== undefined) music.setSteps(plan.steps)
      if (plan.duration !== undefined) music.setDuration(plan.duration)
      if (plan.language !== undefined) music.setLanguage(plan.language)
      music.setLyrics(plan.lyrics ?? "")
      if (plan.lyrics) music.openLyrics()
    }
    draftRef.current?.focus()
  }

  const downloadImage = (image: GeneratedImage) => {
    if (!image.url) return
    const url = image.url.startsWith("data:") ? URL.createObjectURL(dataUrlBlob(image.url)) : image.url
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = imageFileName(image.prompt, image.seed); anchor.click()
    if (url !== image.url) setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const copySeed = async (item: ChatMessage) => {
    if (!item.generated) return
    try { await navigator.clipboard.writeText(String(item.generated.seed)); setCopied(`${item.id}:seed`) } catch { setError(t("ui.copyError")) }
  }

  /* The voice loop (docs/voice-loop.md): the microphone sends what is said as the next message, replies can be read aloud, and hands-free listens
     again after each. It needs a speech recognition model and a speech model on the server. */
  const voiceModels = useMemo(() => pickVoiceModels(modelInfos), [modelInfos])
  const [speakReplies, setSpeakReplies] = useState(() => stored(localStorage, "colibri.speakReplies", "") === "1")
  const [handsFree, setHandsFree] = useState(() => stored(localStorage, "colibri.handsFree", "") === "1")
  const [replyVoice, setReplyVoice] = useState(() => stored(localStorage, "colibri.replyVoice", ""))
  useEffect(() => {
    try {
      localStorage.setItem("colibri.speakReplies", speakReplies ? "1" : "")
      localStorage.setItem("colibri.handsFree", handsFree ? "1" : "")
      localStorage.setItem("colibri.replyVoice", replyVoice)
    } catch { /* restricted storage mode */ }
  }, [speakReplies, handsFree, replyVoice])
  const sendRef = useRef(send)
  sendRef.current = send
  const voice = useVoice({
    baseUrl, apiKey, asrModel: voiceModels.asr, ttsModel: replyVoice && voiceModels.clone ? voiceModels.clone : voiceModels.tts, voice: replyVoice || undefined,
    onTranscript: (words) => { void sendRef.current(words) }, onNothing: () => setHandsFree(false), onError: setError,
  })
  const voiceRef = useRef(voice)
  voiceRef.current = voice
  const heardLast = useRef(messages.length ? messages[messages.length - 1].id : "")
  const lastTurn = messages.length ? messages[messages.length - 1] : undefined
  useEffect(() => {
    if (loading || !lastTurn || lastTurn.role !== "assistant" || lastTurn.mode || !lastTurn.content || heardLast.current === lastTurn.id) return
    if (lastTurn.finish !== "stop" && lastTurn.finish !== "length") return
    heardLast.current = lastTurn.id
    if (!voiceRef.current.available) return
    void (async () => {
      if (speakReplies) await voiceRef.current.speak(lastTurn.content)
      if (handsFree) void voiceRef.current.listen()
    })()
  }, [loading, lastTurn?.id, lastTurn?.finish]) // eslint-disable-line react-hooks/exhaustive-deps

  const submit = () => { if (music.mode) void music.send(); else if (imageMode) void sendImage(); else void send() }

  const openSettings = (page = "general") => { setSettingsPage(page); setView("settings") }
  const clearChat = () => { chatTag.current = newChatTag(); setClientTags(dashboardTags(chatTag.current)); updateMessages([]); setLastRun(null); setTokPerSec(null); setTtft(null); setTokenCount(0); setTotalTokens({ prompt: 0, completion: 0 }); setError("") }
  const newChat = () => {
    if (loading) return
    if (messages.length) setArchives(items => [{ id: message("system", "").id, slot: cacheSlot, messages: [...messages] }, ...items])
    clearChat(); setDraft(""); setView("chat"); requestAnimationFrame(() => draftRef.current?.focus())
  }
  /* The conversation on screen goes to History first, so editing or branching never loses it. */
  const archiveChat = () => { if (messages.length) setArchives(items => [{ id: message("system", "").id, slot: cacheSlot, messages: [...messages] }, ...items]) }
  const editMessage = (index: number) => {
    if (loading) return
    const turn = editFrom(messages, index)
    if (!turn) return
    archiveChat()
    updateMessages(turn.previous); setDraft(turn.text); setPending(turn.pictures); setError("")
    requestAnimationFrame(() => draftRef.current?.focus())
  }
  const branchFrom = (index: number) => {
    if (loading) return
    const kept = branchAt(messages, index)
    if (!kept) return
    archiveChat()
    updateMessages(kept); setError("")
    requestAnimationFrame(() => draftRef.current?.focus())
  }
  const rewrite = (kind: Rewrite) => { if (!loading) void send(t(REWRITES[kind]), messages, [], false) }
  const exportChat = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ model, messages }, null, 2)], { type: "application/json" }))
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = "colibri-chat.json"; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  const copyMessage = async (item: ChatMessage) => {
    try { await navigator.clipboard.writeText(item.content); setCopied(item.id) } catch { setError(t("ui.copyError")) }
  }
  const restoreArchive = (entry: { id: string; slot: number; messages: ChatMessage[] }) => {
    if (loading) return
    const slot = entry.slot < kvSlots ? entry.slot : cacheSlot
    const currentMessages = conversations[slot] || []
    setArchives(items => [...(currentMessages.length ? [{ id: message("system", "").id, slot, messages: currentMessages }] : []), ...items.filter(item => item.id !== entry.id)])
    setConversations(items => ({ ...items, [slot]: entry.messages })); setCacheSlot(slot); setView("chat"); setHistoryOpen(false); setLastRun(null); setError(""); setDraft("")
  }
  /* The server's own message stays on screen; when it is one we know, a sentence in plain words goes above it. */
  const errorBanner = (() => {
    if (!error) return null
    const shown = t(error)
    const known = explainError(shown)
    if (!known) return <div className="error-banner" role="alert">{shown}</div>
    return <div className="error-banner" role="alert">
      <div className="error-explain">{t(known.key, known.vars)}</div>
      <div className="error-original">{shown}</div>
      {known.action === "newChat" && <Button type="button" variant="secondary" size="sm" onClick={newChat} disabled={loading}>{t("explain.newChat")}</Button>}
    </div>
  })()
  const empty = messages.length === 0
  const connectionControls = (<fieldset disabled={loading}><section className="side-section">
          <div className="section-title"><Link2 className="size-3.5" /> {t("sidebar.connection")}</div>
          <label>{t("sidebar.endpoint")}<Input id="endpoint" value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} /></label>
          <label>{t("sidebar.apiKey")}<div className="relative"><KeyRound className="field-icon" /><Input id="api-key" className="pl-9" type="password" value={apiKey} placeholder={t("sidebar.apiKeyPlaceholder")} onChange={(event) => setApiKey(event.target.value)} /></div><span className="field-help">{t("sidebar.apiKeyHelp")}</span></label>
          <Button type="button" variant="secondary" onClick={connect} disabled={connecting}>
            {connecting ? <LoaderCircle className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
            {t("sidebar.probe")}
          </Button>
          <div className={cn("connection-state", connected && "connected")} aria-live="polite"><span />{connected ? t("status.connected") : t("status.notConnected")}</div>
        </section></fieldset>)
  const inferenceControls = (<fieldset disabled={loading}><section className="side-section">
          <div className="section-title"><SlidersHorizontal className="size-3.5" /> {t("sidebar.inference")}</div>
          <label>{t("sidebar.model")}<select id="model-select" value={model} onChange={(event) => setModel(event.target.value)}>{models.length ? <>{models.filter((id) => !presetInfos[id]).map((id) => <option key={id}>{id}</option>)}{models.some((id) => presetInfos[id]) && <optgroup label={t("sidebar.presets")}>{models.filter((id) => presetInfos[id]).map((id) => <option key={id}>{id}</option>)}</optgroup>}</> : <option>{model}</option>}</select></label>
          {presetInfos[model] && <p className="field-help preset-note">{[t("presets.note", { model: presetInfos[model].of }), presetInfos[model].description].filter(Boolean).join(" ")}</p>}
          {music.mode ? <p className="field-help">{t("music.modelNote")}</p> : imageMode ? <p className="field-help">{t("image.modelNote")}</p> : <>
          {health?.kv_slots && health.kv_slots > 1 ? <label>{t("sidebar.kvSession")}<select id="kv-slot" value={cacheSlot} onChange={(event) => setCacheSlot(Number(event.target.value))} disabled={loading}>
            {Array.from({ length: kvSlots }, (_, slot) => <option key={slot} value={slot}>{t("sidebar.sessionLabel", { slot: slot + 1 })}</option>)}
          </select><span className="field-help">{t("sidebar.kvSessionHelp")}</span></label> : null}
          <label><span className="label-line"><span>{t("sidebar.temperature")}</span><code>{temperature.toFixed(1)}</code></span><input id="temperature" className="range" type="range" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
          <label>{t("sidebar.maxTokens")}<Input id="max-tokens" type="number" min={1} max={32768} value={maxTokens} onChange={(event) => { const value = Number(event.target.value); if (Number.isFinite(value)) setMaxTokens(Math.min(32768, Math.max(1, Math.round(value)))) }} /></label>
          <button type="button" className={cn("toggle-row", thinking && "active")} aria-pressed={thinking} onClick={() => setThinking((value) => !value)}>
            <span><BrainCircuit className="size-4" /> {t("sidebar.reasoning")}</span><i><b /></i>
          </button>
          </>}
        </section></fieldset>)
  const runtimeControls = (<><section className="side-section runtime-section" aria-live="polite">
          <div className="section-title"><Activity className="size-3.5" /> {t("sidebar.runtime")}</div>
          {health?.hwinfo ? <div className="hw-panel">
            {health.hwinfo.cpu ? <div className="hw-row"><Cpu className="size-3.5" /><span>{health.hwinfo.cpu}</span></div> : null}
            {health.hwinfo.gpus > 0 ? <div className="hw-row"><MonitorDot className="size-3.5" /><span>{health.hwinfo.gpus}× GPU<small>{health.hwinfo.vram_total_gb.toFixed(0)} GB VRAM</small></span></div> : null}
            <div className="hw-row"><MemoryStick className="size-3.5" /><span>{health.hwinfo.ram_total_gb.toFixed(0)} GB RAM<small>{health.hwinfo.ram_avail_gb.toFixed(0)} GB free</small></span></div>
            <div className="hw-row"><HardDrive className="size-3.5" /><span>{health.hwinfo.cores} cores</span></div>
          </div> : null}
          {health?.scheduler ? <>
            <div className="runtime-grid">
              <div><span>{t("dashboard.active")}</span><strong>{active}<small> / {capacity}</small></strong></div>
              <div><span>{t("dashboard.queued")}</span><strong>{health.scheduler.queued}<small> / {health.scheduler.max_queue}</small></strong></div>
              <div><span>{t("dashboard.completed")}</span><strong>{health.scheduler.completed}</strong></div>
              <div><span>{t("dashboard.failures")}</span><strong>{failures}</strong></div>
            </div>
            {health.tiers ? (() => {
              const ti = health.tiers
              const total = Math.max(ti.vram + ti.ram + ti.disk, 1)
              return <div className="tier-panel">
                <div className="tier-bar" role="img" aria-label={t("tier.ariaLabel", { vram: ti.vram, ram: ti.ram, disk: ti.disk })}>
                  <span className="tier-vram" style={{ width: `${(100 * ti.vram) / total}%` }} />
                  <span className="tier-ram" style={{ width: `${(100 * ti.ram) / total}%` }} />
                  <span className="tier-disk" style={{ width: `${(100 * ti.disk) / total}%` }} />
                </div>
                <div className="tier-legend">
                  <span><i className="tier-vram" />{t("tier.vram")} <strong>{ti.vram.toLocaleString()}</strong><small>{ti.vram_gb.toFixed(1)} GB</small></span>
                  <span><i className="tier-ram" />{t("tier.ram")} <strong>{ti.ram.toLocaleString()}</strong><small>{ti.ram_gb.toFixed(1)} GB</small></span>
                  <span><i className="tier-disk" />{t("tier.disk")} <strong>{ti.disk.toLocaleString()}</strong></span>
                </div>
              </div>
            })() : null}
            {totalTokens.prompt + totalTokens.completion > 0 ? <div className="session-stats">
              <span><Database className="size-3" /> {t("dashboard.session")} <strong>{totalTokens.prompt.toLocaleString()}</strong> {t("dashboard.prompt")} + <strong>{totalTokens.completion.toLocaleString()}</strong> {t("dashboard.completion")}</span>
            </div> : null}
            <div className="runtime-foot"><span className="runtime-dot" /> {t("sidebar.schedulerOnline")} <code>{kvSlots} KV</code></div>
          </> : <p className="runtime-unavailable">{connected ? (healthError ? t(healthError) : t("status.runtimeUnavailable")) : t("sidebar.runtimeProbe")}</p>}
        </section></>)
  const metrics = (<div className="live-metrics" aria-live="polite">              {loading && tokenCount > 0 ? <Badge className="badge-live"><Zap className="size-3 flash" /> {t("topbar.tokens", { n: tokenCount })}</Badge> : null}
              {tokPerSec != null ? <Badge className={cn("badge-speed", loading && "badge-live")}><Gauge className="size-3" /> {t("topbar.tokPerSec", { n: tokPerSec.toFixed(1) })}</Badge> : null}
              {!loading && ttft != null ? <Badge><Timer className="size-3" /> TTFT {(ttft/1000).toFixed(1)}s</Badge> : null}
              {!loading && lastRun?.usage ? <Badge><Layers className="size-3" /> {lastRun.usage.prompt_tokens}→{lastRun.usage.completion_tokens}</Badge> : null}
              {!loading && lastRun?.finishReason === "length" ? <Badge className="badge-warn" title={t("topbar.truncatedHelp")}><AlertTriangle className="size-3" /> {t("topbar.truncated")}</Badge> : null}
              {lastRun?.queueWaitMs != null ? <Badge><Clock className="size-3" /> queue {Math.round(lastRun.queueWaitMs)}ms</Badge> : null}
              {!imageMode && !music.mode && <Badge><MonitorDot className="size-3" /> {t("topbar.slot", { n: cacheSlot + 1 })}</Badge>}
</div>)

  const imageCard = (item: ChatMessage) => {
    const image = item.generated!
    if (image.url) return <ImageResultCard image={image} copied={copied === `${item.id}:seed`} busy={loading || !imageMode}
      onDownload={() => downloadImage(image)} onCopySeed={() => void copySeed(item)} onAgain={() => void imageAgain(image, item.id)} />
    if (imageRun?.targetId === item.id) return <ImageProgressCard request={image} run={imageRun} onCancel={() => abortRef.current?.abort()} />
    return <ImageStoppedCard request={image} finish={item.finish} busy={loading || !imageMode} onRetry={() => void imageRetry(item)} />
  }
  const musicCard = (item: ChatMessage) => {
    const song = item.song!
    if (song.url) return <MusicResultCard song={song} copied={music.copied === `${item.id}:seed`} busy={loading || !music.mode}
      onDownload={() => music.download(song)} onCopySeed={() => void music.copySeed(item)} onAgain={() => void music.again(item)} />
    if (music.run?.targetId === item.id) return <MusicProgressCard request={song} run={music.run} onCancel={music.cancel} />
    return <MusicStoppedCard request={song} finish={item.finish} busy={loading || !music.mode} onRetry={() => void music.retry(item)} />
  }
  const suggestions = music.mode
    ? [{ key: "music.prompt.piano", Icon: Music, label: "music.suggest.piano" }, { key: "music.prompt.synth", Icon: Waves, label: "music.suggest.synth" }, { key: "music.prompt.folk", Icon: Guitar, label: "music.suggest.folk" }]
    : imageMode
    ? [{ key: "image.prompt.photo", Icon: Camera, label: "image.suggest.photo" }, { key: "image.prompt.pixel", Icon: Grid3x3, label: "image.suggest.pixel" }, { key: "image.prompt.poster", Icon: Palette, label: "image.suggest.poster" }]
    : [{ key: "prompts.routing", Icon: BrainCircuit, label: "ui.idea" }, { key: "prompts.benchmark", Icon: Code2, label: "ui.code" }, { key: "prompts.caching", Icon: Database, label: "ui.analyze" }]

  return <div className="app-shell redesigned">
    <aside className="rail" aria-label={t("ui.sidebar")}>
      <button className="rail-brand" title="colibrì" onClick={() => setView("chat")}><Brand /></button>
      <button className="rail-button" aria-label={t("ui.newChat")} title={t("ui.newChat")} onClick={newChat} disabled={loading}><SquarePen /></button>
      <button className="rail-button" aria-label={t("ui.search")} title={t("ui.search")} onClick={() => { setQuery(""); setHistoryOpen(true) }}><Search /></button>
      <button className="rail-button" aria-label={t("ui.history")} title={t("ui.history")} onClick={() => { setQuery(""); setHistoryOpen(true) }}><History /></button>
      <div className="rail-bottom">
        <button className={cn("rail-button", view === "settings" && "active")} aria-label={t("ui.settings")} title={t("ui.settings")} onClick={() => openSettings()}><Settings2 /></button>
        <button className="rail-button" aria-label={t("ui.theme")} title={t("ui.theme")} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? <Sun /> : <Moon />}</button>
      </div>
    </aside>
    <main className={cn("workspace", view === "brain" && "brain-workspace")}>
      {view !== "brain" && <header className="workspace-topbar">
        <button className="model-button" onClick={() => openSettings("model")}><span>{model}</span><ChevronDown /></button>
        <div className="workspace-actions">
          <button className={cn("connection-state", connected && "connected")} onClick={() => openSettings("connection")}><span />{connected ? t("status.connected") : t("status.notConnected")}</button>
          {view === "chat" && <><button className="icon-action" aria-label={t("ui.export")} title={t("ui.export")} disabled={empty} onClick={exportChat}><Download /></button><button className="icon-action" aria-label={t("topbar.clear")} title={t("topbar.clear")} disabled={empty || loading} onClick={clearChat}><Trash2 /></button></>}
        </div>
      </header>}
      {view === "settings" ? <section className="settings-page">
        <header className="page-heading"><span>COLIBRI</span><h1>{t("ui.settings")}</h1><p>{t("ui.settingsIntro")}</p></header>
        <nav className="settings-tabs" aria-label={t("ui.settings")}>
          {pages.map(id => <button key={id} aria-current={activePage === id ? "page" : undefined} onClick={() => setSettingsPage(id)}>{t(`ui.${id}`)}</button>)}
        </nav>
        {errorBanner}
        <div className="settings-content">
          {activePage === "general" && <section className="settings-card"><h2>{t("ui.appearance")}</h2><div className="theme-choices">{["dark", "light"].map(value => <button key={value} className={`theme-choice ${value}`} aria-pressed={theme === value} onClick={() => setTheme(value)}><span /><b>{t(`ui.${value}`)}</b></button>)}</div><label className="language-field"><Globe />{t("ui.language")}<select value={locale} onChange={e => setLocale(e.target.value)}>{locales.map(l => <option key={l.code} value={l.code}>{l.label}</option>)}</select></label><p className="settings-help">{t("chat.inputHint")}</p></section>}
          {activePage === "presets" && <section className="settings-card">
            <PresetsPanel baseUrl={baseUrl} apiKey={apiKey} connected={connected} onChanged={() => void connect()}
              onUse={(name) => { startFromPreset(name); setModel(name); setView("chat") }} />
          </section>}
          {activePage === "voices" && <section className="settings-card"><VoicesPanel baseUrl={baseUrl} apiKey={apiKey} connected={connected} replyVoice={replyVoice} onReplyVoice={setReplyVoice} /></section>}
          {activePage === "requests" && <section className="settings-card wide"><RequestsPanel baseUrl={baseUrl} apiKey={apiKey} connected={connected} /></section>}
          {activePage === "log" && <section className="settings-card wide"><LogPanel baseUrl={baseUrl} apiKey={apiKey} connected={connected} /></section>}
          {activePage === "library" && <section className="settings-card wide"><LibraryPanel baseUrl={baseUrl} apiKey={apiKey} connected={connected} jobsAvailable={canBackground}
            onMakeAgain={(settings) => { applySettings(settings); setView("chat") }} /></section>}
          {activePage === "connection" && <section className="settings-card">{connectionControls}</section>}
          {activePage === "model" && <section className="settings-card">
            <ModelSwitcher baseUrl={baseUrl} apiKey={apiKey} connected={connected} busy={loading} onSwitched={() => void connect()} />
            {inferenceControls}
          </section>}
          {activePage === "system" && <section className="settings-card">{runtimeControls}{metrics}</section>}
        </div>
      </section> : view === "brain" ? <BrainWorkspace baseUrl={baseUrl} apiKey={apiKey} connected={connected} />
      : view === "compare" ? <Compare baseUrl={baseUrl} apiKey={apiKey} connected={connected} models={chatModelInfos} settings={{ temperature, maxTokens, thinking }} />
      : view === "profiling" ? <section className="profiling-workspace"><header className="page-heading"><span>COLIBRI / ENGINE</span><h1>Profiling</h1></header>{metrics}<Profiling baseUrl={baseUrl} apiKey={apiKey} connected={connected} /></section>
      : <section className={cn("chat-view", empty && "empty")} hidden={view === "brio"}>
        {empty ? <div className="welcome-brand"><Brand word /></div> : <div className="conversation" role="log" aria-label={t("nav.chat")}>
          <div className="message-list">{messages.map((item, index) => <article key={item.id} className={cn("message", item.role)}>
            {item.role !== "user" && <div className="assistant-brand"><Brand /><span>colibrì</span></div>}
            {item.images?.length ? <div className="message-images">{item.images.map((url, at) =>
              <img key={at} src={url} alt={t("ui.attachedImage", { n: at + 1 })} />)}</div> : null}
            {item.role === "assistant" && item.song ? musicCard(item) : item.role === "assistant" && item.generated ? imageCard(item) : <div className="message-body">{item.reasoning
              ? <details className="reasoning" open={!item.content}>
                  <summary>{t("sidebar.reasoning")}</summary>
                  <div className="reasoning-body">{item.reasoning}</div>
                </details>
              : null}{item.content ? (item.role === "assistant" ? <Markdown text={item.content} /> : item.content) : <span className="typing" aria-label={t("ui.generating")}><i /><i /><i /></span>}</div>}
            {item.role === "assistant" && item.content && <div className="message-actions"><button className="icon-action" aria-label={t("ui.copy")} title={t("ui.copy")} onClick={() => void copyMessage(item)}><Copy /></button>{copied === item.id && <span role="status">{t("ui.copied")}</span>}{index === messages.length - 1 && !loading && !imageMode && <button className="icon-action" aria-label={t("ui.regenerate")} title={t("ui.regenerate")} onClick={() => { const retry = resendFrom(messages, index); if (retry) void send(retry.text, retry.previous, retry.pictures, false) }}><RefreshCw /></button>}{index === messages.length - 1 && !loading && !imageMode && supportsContinuation(health) && continuable(item) && <button className="icon-action" aria-label={t("ui.continue")} title={t("ui.continue")} onClick={() => void continueTurn()}><StepForward /></button>}{!loading && !imageMode && isPlainTurn(item) && <button className="icon-action" aria-label={t("ui.branch")} title={t("ui.branch")} onClick={() => branchFrom(index)}><GitBranch /></button>}{index === messages.length - 1 && !loading && !imageMode && isPlainTurn(item) && <><button className="icon-action" aria-label={t("ui.rewriteShorter")} title={t("ui.rewriteShorter")} onClick={() => rewrite("shorter")}><Minimize2 /></button><button className="icon-action" aria-label={t("ui.rewriteSimpler")} title={t("ui.rewriteSimpler")} onClick={() => rewrite("simpler")}><Lightbulb /></button></>}</div>}
            {item.role === "user" && !loading && !imageMode && isPlainTurn(item) && <div className="message-actions"><button className="icon-action" aria-label={t("ui.edit")} title={t("ui.edit")} onClick={() => editMessage(index)}><Pencil /></button></div>}
            {item.role === "assistant" && item.stats && (() => { const parts = replyParts(item.stats, t); return parts.length ? <div className="reply-stats" title={t("reply.help")}>{parts.join(" · ")}</div> : null })()}
          </article>)}<div ref={bottomRef} /></div>
        </div>}
        <div className="composer-wrap">
          {errorBanner}
          <form className={cn("composer", (imageMode || music.mode) && "image-mode")} onSubmit={e => { e.preventDefault(); submit() }}
            onDragOver={e => { if (((music.mode && music.audioIn) || (!music.mode && (!imageMode || canEdit))) && e.dataTransfer.types.includes("Files")) e.preventDefault() }}
            onDrop={e => {
              if (music.mode && music.audioIn && e.dataTransfer.files.length) {
                e.preventDefault()
                const file = e.dataTransfer.files[0]
                void music.chooseClip(music.task !== "text2music" && !music.source ? "source" : "reference", file)
              } else if (!music.mode && (!imageMode || canEdit) && e.dataTransfer.files.length) { e.preventDefault(); void (imageMode ? attachPictures : attach)(e.dataTransfer.files) }
            }}>
            {imageMode && attached.length > 0 && <div className="composer-attachments">
              {attached.map((picture, at) => <span key={picture.id} className="attachment">
                <img src={picture.url} alt={t("image.attachedAlt", { n: at + 1 })} />
                <button type="button" aria-label={t("image.removePicture")} title={t("image.removePicture")}
                  onClick={() => removePicture(picture.id)}><X /></button>
              </span>)}
            </div>}
            {!imageMode && pending.length > 0 && <div className="composer-attachments">
              {pending.map((url, at) => <span key={at} className="attachment">
                <img src={url} alt={t("ui.attachedImage", { n: at + 1 })} />
                <button type="button" aria-label={t("ui.removeImage")} title={t("ui.removeImage")}
                  onClick={() => setPending(list => list.filter((_, index) => index !== at))}><X /></button>
              </span>)}
            </div>}
            {music.mode && music.audioIn && <MusicClips task={music.task} source={music.source} reference={music.reference}
              sourceProblem={music.sourceProblem} referenceProblem={music.referenceProblem} onChoose={(role, file) => void music.chooseClip(role, file)}
              onClear={music.clearClip} />}
            {music.mode && music.lyricsOpen && <MusicLyrics value={music.lyrics} max={music.limits.maxLyricBytes} onChange={music.setLyrics} />}
            <Textarea ref={draftRef} id="draft" aria-label={music.mode ? t("music.placeholder") : imageMode ? t(attached.length ? "image.placeholderEdit" : "image.placeholder") : t("chat.placeholder")} value={draft} onChange={e => setDraft(e.target.value)} placeholder={music.mode ? t("music.placeholder") : imageMode ? t(attached.length ? "image.placeholderEdit" : "image.placeholder") : t("chat.placeholder")} onPaste={event => { if (music.mode || (imageMode && !canEdit)) return; const files = Array.from(event.clipboardData.files || []); if (files.length) { event.preventDefault(); void (imageMode ? attachPictures : attach)(files) } }} onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); submit() } }} />
            <div className="composer-foot">
              {music.mode ? <MusicControls limits={music.limits} task={music.task} onTask={music.setTask}
                cover={{ strength: music.coverStrength, noise: music.coverNoise, onStrength: music.setCoverStrength, onNoise: music.setCoverNoise }}
                repaint={{ from: music.repaintFrom, to: music.repaintTo, mode: music.repaintMode, strength: music.repaintStrength, spanOk: music.spanOk,
                  onFrom: music.setRepaintFrom, onTo: music.setRepaintTo, onMode: music.setRepaintMode, onStrength: music.setRepaintStrength }}
                hasReference={music.reference !== null} onReferenceFile={(file) => void music.chooseClip("reference", file)}
                duration={music.duration} steps={music.steps} seed={music.seedText}
                seedValid={music.seed !== undefined} language={music.language} lyricsOpen={music.lyricsOpen} lyricsFilled={music.lyrics.trim() !== ""}
                onDuration={music.setDuration} onSteps={music.setSteps} onSeed={music.setSeedText} onLanguage={music.setLanguage} onLyrics={music.toggleLyrics} />
              : imageMode ? <>
              {canEdit && <input ref={editRef} type="file" accept="image/*" multiple hidden
                onChange={e => { void attachPictures(e.target.files); e.target.value = "" }} />}
              <ImageControls size={imageSize} custom={customSize} steps={imageSteps} seed={seedText} seedValid={seed !== undefined}
                onSize={(next) => { setSizeAuto(false); setImageSize(next) }} onCustom={(next) => { setSizeAuto(false); setCustomSize(next) }}
                onSteps={setImageSteps} onSeed={setSeedText}
                canEdit={canEdit} auto={attached.length > 0 && sizeAuto} onAuto={attached.length ? () => setSizeAuto(true) : undefined}
                onAttach={() => editRef.current?.click()} />
              </> : <>
              <input ref={fileRef} type="file" accept="image/*" multiple hidden
                onChange={e => { void attach(e.target.files); e.target.value = "" }} />
              <button type="button" className="attach-chip" aria-label={t("ui.attachImage")} title={t("ui.attachImage")}
                onClick={() => fileRef.current?.click()}><ImagePlus /></button>
              <button type="button" className="reasoning-chip" aria-pressed={thinking} onClick={() => setThinking(value => !value)}><BrainCircuit />{t("sidebar.reasoning")}</button>
              {voice.available && <VoiceControls state={voice.state} level={voice.level} replies={speakReplies} onReplies={() => { setSpeakReplies((value) => !value); voice.stopSpeaking() }}
                handsFree={handsFree} onHandsFree={() => { if (handsFree) { setHandsFree(false); voice.cancel() } else { setHandsFree(true); if (voice.state === "idle" && !loading) void voice.listen() } }}
                onListen={() => void voice.listen()} onFinish={voice.finish} onStop={() => { setHandsFree(false); voice.cancel() }} />}
              <button type="button" className="slot-chip" onClick={() => openSettings("model")}><Database />{t("topbar.slot", { n: cacheSlot + 1 })}</button>
              </>}
              {(imageMode || music.mode) && <>
                <input ref={settingsRef} type="file" accept="image/png,audio/wav,audio/x-wav,audio/wave,.png,.wav" hidden
                  onChange={e => { void openSettingsFile(e.target.files?.[0]); e.target.value = "" }} />
                <button type="button" className="settings-chip" aria-label={t("media.open")} title={t("media.open")}
                  onClick={() => settingsRef.current?.click()}><FileInput /></button>
                {canBackground && <button type="button" className="settings-chip" aria-pressed={background} aria-label={t("job.background")} title={t("job.backgroundHelp")}
                  onClick={() => setBackground((value) => !value)}><CloudUpload /></button>}
              </>}
              {loading ? <button type="button" className="send-button" aria-label={music.mode || imageMode ? t("image.cancel") : t("chat.stop")} onClick={() => abortRef.current?.abort()}><CircleStop /></button> : <button type="submit" className="send-button" aria-label={music.mode ? t("music.generate") : imageMode ? t("image.generate") : t("chat.send")} disabled={!canSend}><ArrowUp /></button>}
            </div>
          </form>
          {empty && <div className="suggestions">{suggestions.map(({ key, Icon, label }) => <button key={key} onClick={() => { setDraft(t(key)); draftRef.current?.focus() }}><Icon />{t(label)}</button>)}</div>}
          {!empty && !imageMode && lastRun?.finishReason === "length" && <p className="truncation-note">{t("topbar.truncatedHelp")}</p>}
        </div>
      </section>}
      <section className="brio-workspace" hidden={view !== "brio"}>
        <header className="page-heading"><span>COLIBRI / SYSTEM ONE</span><h1>{t("nav.brio")}</h1></header>
        <Brio baseUrl={baseUrl} apiKey={apiKey} model={model} connected={connected} />
      </section>
    </main>
    <NavigationDock view={view} onNavigate={setView} loading={loading} />
    <dialog ref={historyDialog} className="history-dialog" onCancel={() => setHistoryOpen(false)} onClose={() => setHistoryOpen(false)} onClick={e => { if (e.target === e.currentTarget) { const r = e.currentTarget.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) setHistoryOpen(false) } }}>
      <header><h2>{t("ui.history")}</h2><button className="icon-action" aria-label={t("ui.close")} onClick={() => setHistoryOpen(false)}><X /></button></header>
      <label className="history-search"><Search /><input aria-label={t("ui.search")} placeholder={t("ui.search")} value={query} onChange={e => setQuery(e.target.value)} /></label>
      <div className="history-results">
        {Object.entries(conversations).filter(([slot, items]) => Number(slot) < kvSlots && items.some(m => m.content.toLocaleLowerCase().includes(query.toLocaleLowerCase()))).map(([slot, items]) => <button key={slot} disabled={loading} onClick={() => { setCacheSlot(Number(slot)); setView("chat"); setHistoryOpen(false); setDraft(""); setError("") }}><MessageSquareText /><span>{items.find(m => m.role === "user")?.content.slice(0, 85)}<small>{t("topbar.slot", { n: Number(slot) + 1 })}</small></span></button>)}
        {archives.filter(entry => entry.messages.some(m => m.content.toLocaleLowerCase().includes(query.toLocaleLowerCase()))).map(entry => <button key={entry.id} disabled={loading} onClick={() => restoreArchive(entry)}><History /><span>{entry.messages.find(m => m.role === "user")?.content.slice(0, 85)}<small>{t("ui.archived")}</small></span></button>)}
        <p>{t("ui.historyMemory")}</p>
      </div>
    </dialog>
  </div>
}
