import { PREFER_ASYNC, runBackground, type Job } from "./jobs"
import type { ModelDetails } from "./modelDetails"
import type { LogLine, LogRead } from "./logTail"
import type { ReplyStats } from "./replyStats"
import { NO_TOOLS, parsePolicy, parseToolsListing, toolEventOf, type AnswerResult, type ToolEvent, type ToolsListing, type ToolsPolicy, type ToolsPolicyEdit, type TurnPart } from "./tools"

export type ChatRole = "system" | "user" | "assistant"

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  /* Reasoning models stream their thinking on a separate delta field before
     the answer. Kept apart from `content` so it can be rendered as its own
     block and excluded from what is sent back as conversation history. */
  reasoning?: string
  /* Data URIs of the pictures attached to this turn. Kept on the message rather
     than on the draft, because the transcript is resent on every later turn and
     the model has to keep seeing what it was shown. */
  images?: string[]
  /* How an assistant turn's last generation ended: the server's finish_reason;
     "aborted" / "error" when the client stopped or lost the stream; or
     "incomplete" when the stream closed without a finish_reason, which colibri
     always sends last, so its absence means the reply was cut off. Kept on
     the message so it travels with the transcript through slot switches and
     archives; never sent to the server. */
  finish?: string
  /* What an assistant turn cost and where it ran: tokens, speed, the share of the prompt that was not read again, and
     which model answered. Set when the reply finishes; kept with the transcript, never sent to the server. */
  stats?: ReplyStats
  /* What happened inside an assistant turn that used the router's built-in tools, in order: the text pieces, the calls and
     the questions put to the person (lib/tools.ts). Set only when a tool was used; `content` still holds all of the text, so
     everything that reads a turn as text (copy, history, resend) is unchanged. Never sent to the server. */
  parts?: TurnPart[]
  /* Both turns of a text-to-image exchange, the prompt and its picture, carry
     mode "image". They stay in the conversation on screen but are never sent to
     a text model, which would read a question followed by an empty answer. */
  mode?: "image" | "music"
  /* The picture of an image turn: the request while it runs, then the result.
     `url` appears only once the image is finished. */
  generated?: GeneratedImage
  /* The song of a music turn (mode "music"), the same way: the request while it
     runs, then the result with its `url`. */
  song?: GeneratedMusic
}

interface OpenAIError {
  error?: { message?: string; param?: string | null }
}

export interface SchedulerHealth {
  active: boolean | number
  capacity?: number
  queued: number
  max_queue: number
  queue_timeout_seconds: number
  admitted: number
  completed: number
  rejected: number
  timed_out: number
  cancelled: number
}

export interface TiersHealth {
  vram: number
  ram: number
  disk: number
  vram_gb: number
  ram_gb: number
}

export interface HwinfoHealth {
  cores: number
  ram_total_gb: number
  ram_avail_gb: number
  gpus: number
  vram_total_gb: number
  cpu: string
  gpu: string
}

export interface HealthResponse {
  status: string
  scheduler?: SchedulerHealth
  kv_slots?: number
  tiers?: TiersHealth
  hwinfo?: HwinfoHealth
  /* Whether a message list ending on an assistant turn is continued rather than
     answered fresh (COLI_CONTINUE_ASSISTANT). Absent on older servers. */
  continue_assistant?: boolean
  /* What the server can do (GET /health with the key): `router` is true for `coli router`, `server_log` is the Log page's
     route, and so on; see docs/api.md. Absent on a server that predates it: ask, and read a 404 as "not here". */
  features?: Record<string, boolean | number | string>
}

export interface ProfileTurn {
  wall_s: number
  prompt_tokens: number
  completion_tokens: number
  expert_disk_s: number
  expert_wait_s: number
  expert_matmul_s: number
  attention_s: number
  lm_head_s: number
  forwards: number
}

export interface ProfileResponse {
  seq: number
  turns: ProfileTurn[]
}

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  /* How much of the prompt the engine did not read again (a continuation of the last request). Left out by a server
     or an engine that does not say, which is not the same as 0. */
  prompt_tokens_details?: { cached_tokens?: number } | null
}

export interface StreamChatResult {
  finishReason: string | null
  usage: TokenUsage | null
  /* The model that answered, as the stream names it: behind a preset or a short alias this is the real model. */
  model: string | null
  requestId: string | null
  queueWaitMs: number | null
}

export function endpoint(baseUrl: string, path: string) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`
}

export function serverEndpoint(baseUrl: string, path: string) {
  return endpoint(baseUrl.replace(/\/v1\/?$/, ""), path)
}

function headers(apiKey: string) {
  return {
    "Content-Type": "application/json",
    ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
  }
}

/* Labels the page puts on what it asks a model for (X-Colibri-Tags, docs/api.md "Client tags"), so the router's request log and its
   jobs can be listed by chat. Set by the app when a chat starts; only requests that compute carry them. */
let clientTags: Record<string, string> | null = null
export function setClientTags(tags: Record<string, string> | null) { clientTags = tags && Object.keys(tags).length ? { ...tags } : null }
export const TAGS_HEADER = "X-Colibri-Tags"
function tagHeaders(): Record<string, string> {
  return clientTags ? { [TAGS_HEADER]: JSON.stringify(clientTags) } : {}
}
function computeHeaders(apiKey: string) {
  return { ...headers(apiKey), ...tagHeaders() }
}

async function responseError(response: Response) {
  const fallback = `${response.status} ${response.statusText}`
  try {
    const body = (await response.json()) as OpenAIError
    return body.error?.message || fallback
  } catch {
    return fallback
  }
}

export interface ModelInfo {
  id: string
  /* What a model does. colibri marks an image model with "image_generation"
     (it answers /v1/images/generations and refuses chat), a language model
     with "chat" and "systemone", and a decision model with "systemone" and
     "decision" (it answers only POST /v1/systemone). */
  capabilities?: string[]
  /* What POST /v1/images/edits takes, for a model that edits pictures
     ("image_edit" among its capabilities): how many pictures at most, and the
     range of the side, in pixels, the engine works them at. */
  edit?: { maxImages: number; refResolution?: { default: number; min: number; max: number } }
  /* What a music model ("music_generation") accepts, from the `audio` block of its
     /v1/models entry: the range of a clip in seconds, the sampler's steps. */
  music?: MusicLimits
  /* A preset served by `coli router` (docs/presets.md): the model it runs and the settings it fills in. */
  preset?: PresetInfo
  /* The context window, in tokens, the engine serves this model with (the card's `context_length`). Absent until the
     engine has said, which for some models is only after their first run. */
  contextLength?: number
}

/* A named model with settings baked in. `defaults` are what a request leaves out; the dashboard starts its own
   controls from them. */
export interface PresetInfo {
  of: string
  description?: string
  system?: string
  defaults: Record<string, unknown>
  keepAlive?: string | number
}

const positive = (value: unknown) => typeof value === "number" && Number.isFinite(value) && value > 0

/* /v1/models: `image.edit` is {max_images, ref_resolution: {default, min, max}} on a
   model that edits. Anything else is left out rather than guessed. */
function editLimits(image: unknown): ModelInfo["edit"] {
  const edit = (image as { edit?: { max_images?: unknown; ref_resolution?: Record<string, unknown> } } | undefined)?.edit
  if (!edit || !positive(edit.max_images)) return undefined
  const range = edit.ref_resolution
  const refResolution = range && positive(range.default) && positive(range.min) && positive(range.max)
    ? { default: Number(range.default), min: Number(range.min), max: Number(range.max) } : undefined
  return { maxImages: Number(edit.max_images), ...(refResolution ? { refResolution } : {}) }
}

/* The tasks a /v1/models `audio` block lists: the ones this page knows, in its order, and composing
   always (an engine that says nothing only composes). */
function musicTasks(value: unknown, flowEdit = false): MusicTask[] {
  const listed = Array.isArray(value) ? value : []
  /* "edit" is not a task of the engine: it is its flow-edit overlay, announced on its own (audio.flow_edit) and needing the clip tasks' audio in */
  const known = MUSIC_TASKS.filter((task) => task === "text2music" || (task === "edit" ? flowEdit && listed.includes("cover") : listed.includes(task)))
  return known.length ? known : ["text2music"]
}

/* /v1/models: `audio` on a music model. Every number must be there and positive; a
   server that states less gets the contract's defaults for the rest (the engine's
   own, music_engine.DEFAULT_INFO), and one that states nothing is left out. */
export function musicLimits(audio: unknown): MusicLimits | undefined {
  if (!audio || typeof audio !== "object") return undefined
  const info = audio as Record<string, unknown>
  const number = (value: unknown, fallback: number) => (positive(value) ? Number(value) : fallback)
  const limits: MusicLimits = {
    sampleRate: number(info.sample_rate, 48000),
    minDuration: number(info.min_duration, 1),
    maxDuration: number(info.max_duration, 600),
    defaultDuration: number(info.default_duration, 30),
    defaultSteps: number(info.default_steps, 8),
    maxSteps: number(info.max_steps, 20),
    defaultShift: number(info.default_shift, 1),
    maxLyricBytes: number(info.max_lyric_bytes, 32768),
    tasks: musicTasks(info.tasks, info.flow_edit === true),
  }
  const audioIn = info.audio_in as { max_seconds?: unknown; ref_seconds?: unknown } | undefined
  if (audioIn && positive(audioIn.max_seconds) && positive(audioIn.ref_seconds) && limits.tasks.length > 1) {
    limits.audioIn = { maxSeconds: Number(audioIn.max_seconds), refSeconds: Number(audioIn.ref_seconds) }
  }
  const strings = (value: unknown) => (Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [])
  const planner = info.planner as { available?: unknown; lms?: unknown; default_lm?: unknown } | undefined
  if (planner && planner.available === true) {
    limits.planner = { lms: strings(planner.lms), defaultLm: typeof planner.default_lm === "string" ? planner.default_lm : "1.7B" }
  }
  const guided = info.guided as { available?: unknown; default_steps?: unknown; max_steps?: unknown; default_guidance?: unknown; tracks?: unknown } | undefined
  if (guided && guided.available === true) {
    const dits = strings(info.dits).filter((item): item is MusicDit => (MUSIC_DITS as readonly string[]).includes(item))
    limits.guided = {
      dits, defaultSteps: number(guided.default_steps, 32), maxSteps: number(guided.max_steps, 200),
      defaultGuidance: number(guided.default_guidance, 7), tracks: strings(guided.tracks),
    }
  }
  if (info.sampler === true) limits.sampler = true
  if (info.flow_edit === true && info.flow_edit_cover === true) limits.flowEditCover = true
  return limits.minDuration <= limits.maxDuration ? limits : undefined
}

export async function listModelInfo(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<ModelInfo[]> {
  const response = await fetch(endpoint(baseUrl, "models"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: Array<{ id: string; capabilities?: unknown; image?: unknown; audio?: unknown; preset_of?: unknown; preset?: unknown; context_length?: unknown }> }
  return (body.data || []).map(({ id, capabilities, image, audio, preset_of, preset, context_length }) => {
    const named = presetInfo(preset_of, preset)
    const window = typeof context_length === "number" && Number.isInteger(context_length) && context_length > 0
      ? { contextLength: context_length } : {}
    if (!Array.isArray(capabilities)) return { id, ...(named ? { preset: named } : {}), ...window }
    const list = capabilities.filter((item): item is string => typeof item === "string")
    const edit = list.includes("image_edit") ? editLimits(image) : undefined
    const music = list.includes("music_generation") ? musicLimits(audio) : undefined
    return { id, capabilities: list, ...(edit ? { edit } : {}), ...(music ? { music } : {}), ...(named ? { preset: named } : {}), ...window }
  })
}

/* `preset_of` and `preset` of a /v1/models entry that is a preset; anything malformed is no preset. */
export function presetInfo(of: unknown, preset: unknown): PresetInfo | undefined {
  if (typeof of !== "string" || !preset || typeof preset !== "object") return undefined
  const raw = preset as Record<string, unknown>
  const defaults = raw.defaults && typeof raw.defaults === "object" && !Array.isArray(raw.defaults) ? raw.defaults as Record<string, unknown> : {}
  return {
    of, defaults,
    ...(typeof raw.description === "string" && raw.description ? { description: raw.description } : {}),
    ...(typeof raw.system === "string" && raw.system ? { system: raw.system } : {}),
    ...(typeof raw.keep_alive === "string" || typeof raw.keep_alive === "number" ? { keepAlive: raw.keep_alive } : {}),
  }
}

export async function listModels(baseUrl: string, apiKey: string, signal?: AbortSignal) {
  return (await listModelInfo(baseUrl, apiKey, signal)).map((model) => model.id)
}

/* A model downloaded on the machine running the server, from GET /v1/models/installed.
   `switchable` is the server's own verdict (engine built, KV slots, cluster mode);
   `reason` says why not. The server names a model by `key`, never by path. */
export interface InstalledModel {
  key: string
  id: string
  name: string
  family: string
  modality: string
  size_bytes: number | null
  verified: boolean
  current: boolean
  switchable: boolean
  reason: string | null
  /* Only from `coli router`: where the model is in its life (idle, starting, ready, stopping), what it holds
     in memory, how long until it is unloaded for being idle, and why its last load failed. */
  state?: "idle" | "starting" | "ready" | "stopping"
  resident_bytes?: number
  expires_in?: number | null
  /* Seconds it stays loaded once idle; null means it never unloads (a request or a preload said so). */
  keep_alive?: number | null
  inflight?: number
  last_error?: string
  /* Only from `coli router`: false when the model is turned off, and where the router started it (card or CPU). */
  enabled?: boolean
  gpu?: GpuDecision | null
  /* What the folder says about itself, read from its small files (architecture, weights, the model's own context limit, source). */
  details?: ModelDetails
  /* Seen on an earlier scan and not found now (a drive that is not connected: `reason` says), or a download that is not finished. */
  missing?: boolean
  unfinished?: boolean
  /* Only from `coli router`: how long the model took to load, the slowest of its last three starts, in seconds. */
  reload_seconds?: number
}

export interface InstalledModels {
  /* False when this server cannot restart itself (not started by the coli launcher). */
  switch: boolean
  reason: string | null
  /* A `coli router`: models load when asked for and unload when idle, so nothing restarts. */
  router: boolean
  models: InstalledModel[]
}

export async function listInstalledModels(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<InstalledModels> {
  const response = await fetch(endpoint(baseUrl, "models/installed"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { switch?: boolean; reason?: string | null; router?: boolean; data?: InstalledModel[] }
  return { switch: body.switch === true, reason: body.reason ?? null, router: body.router === true, models: body.data ?? [] }
}

/* Asks the server to restart on another installed model. It answers 202 and goes
   down: follow it with waitForModel. A `coli router` instead loads the model and
   answers 200 when it is ready (or says why it could not start). */
export async function switchModel(baseUrl: string, apiKey: string, key: string, signal?: AbortSignal): Promise<{ id: string; name: string }> {
  const response = await fetch(endpoint(baseUrl, "models/switch"), {
    method: "POST", headers: headers(apiKey), body: JSON.stringify({ key }), signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { id: string; name: string }
  return { id: body.id, name: body.name }
}

/* Asks a `coli router` to unload a model. One that is answering a request is not cut off: the router says
   "unloading when idle" (202) and it goes when the request has finished. */
export async function unloadModel(baseUrl: string, apiKey: string, key: string, signal?: AbortSignal): Promise<{ status: string; id: string }> {
  const response = await fetch(endpoint(baseUrl, "models/unload"), {
    method: "POST", headers: headers(apiKey), body: JSON.stringify({ key }), signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { status: string; id: string }
  return { status: body.status, id: body.id }
}

/* An install of a catalog model (c/model_installer.py): the router fetches its files as a background job. */
export interface InstallJob {
  id: string
  name: string
  state: "starting" | "downloading" | "finishing" | "done" | "failed" | "cancelled"
  done: number
  total: number
  rate: number | null
  eta_seconds: number | null
  file: string | null
  error: string | null
  license_note?: string
}

export function installActive(job: InstallJob | null | undefined): boolean {
  return !!job && (job.state === "starting" || job.state === "downloading" || job.state === "finishing")
}

/* One row of GET /v1/models/catalog: a model that can be fetched, what this machine makes of it, and whether it is here. */
export interface CatalogRow {
  id: string
  name: string
  kind: "text" | "image" | "audio" | "embedding" | "rerank"
  download_gb: number
  ram_min_gb: number
  ram_good_gb: number
  summary: string
  license_note: string
  status: "good" | "tight" | "needs-ram" | "needs-disk" | "unsupported" | "needs-parent"
  fits: boolean
  reason: string
  /* An add-on (the music planner, the base DiT) names the catalog model whose folder it goes into; it has no key of its own. */
  into?: string
  installed: { key: string | null; id: string; state: string; size_bytes?: number | null } | null
  /* A download of it was started and is not finished: installing again continues it. */
  unfinished: boolean
  partial_bytes?: number
  job: InstallJob | null
}

/* A models folder an install can go to, with the room it has (GET /v1/models/catalog `folders`); `preferred` is the one a model
   goes to when nobody chooses. */
export interface ModelFolder {
  path: string
  free_gb: number | null
  preferred: boolean
}

export interface ModelCatalog {
  rows: CatalogRow[]
  roots: string[]
  folders: ModelFolder[]
  freeGb: number | null
}

/* What can be fetched, or null when this server cannot install models (not a router, or an older one). */
export async function getModelCatalog(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<ModelCatalog | null> {
  const response = await fetch(endpoint(baseUrl, "models/catalog"), { headers: headers(apiKey), signal })
  if (response.status === 404) return null
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: CatalogRow[]; roots?: string[]; folders?: ModelFolder[]; free_gb?: number | null }
  return { rows: body.data ?? [], roots: body.roots ?? [], folders: body.folders ?? [], freeGb: body.free_gb ?? null }
}

/* Starts fetching a catalog model. `confirm` is sent because the person was shown its size and licence first; the router
   refuses without it. `root` is one of the catalog's `folders`, to say where it goes (else the router picks the first with room).
   Answers the job (202); follow it through getModelCatalog's `job`. */
export async function installModel(baseUrl: string, apiKey: string, id: string, root?: string | null, signal?: AbortSignal): Promise<InstallJob> {
  const response = await fetch(endpoint(baseUrl, "models/install"), {
    method: "POST", headers: headers(apiKey), body: JSON.stringify(root ? { id, confirm: true, root } : { id, confirm: true }), signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as InstallJob
}

/* Stops an install; what was fetched is kept and installing again continues it. */
export async function cancelInstall(baseUrl: string, apiKey: string, id: string, signal?: AbortSignal): Promise<void> {
  const response = await fetch(endpoint(baseUrl, "models/install/cancel"), {
    method: "POST", headers: headers(apiKey), body: JSON.stringify({ id }), signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
}

/* Removes an installed model's files. The router refuses a model that is loaded. `presets` names the presets that used it. */
export async function uninstallModel(baseUrl: string, apiKey: string, key: string, signal?: AbortSignal): Promise<{ freedBytes: number; presets: string[] }> {
  const response = await fetch(endpoint(baseUrl, "models/uninstall"), {
    method: "POST", headers: headers(apiKey), body: JSON.stringify({ key, confirm: true }), signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { freed_bytes?: number; presets?: string[] }
  return { freedBytes: body.freed_bytes ?? 0, presets: body.presets ?? [] }
}

/* Removes an installed add-on's files by its catalog id. The router refuses while the model it belongs to is loaded. */
export async function uninstallAddon(baseUrl: string, apiKey: string, id: string, signal?: AbortSignal): Promise<{ freedBytes: number }> {
  const response = await fetch(endpoint(baseUrl, "models/uninstall"), {
    method: "POST", headers: headers(apiKey), body: JSON.stringify({ id, confirm: true }), signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { freed_bytes?: number }
  return { freedBytes: body.freed_bytes ?? 0 }
}

/* Where a model runs: the graphics card, the CPU, or whatever the router's policy says ("auto"). */
export type GpuChoice = "auto" | "vulkan" | "cpu"

/* What the router decided about a loaded model's card or CPU, and why. "unchanged": the router did not choose. */
export interface GpuDecision {
  mode: "vulkan" | "cpu" | "unchanged"
  why: string
  free_gb?: number
}

/* The router's own policy (`coli router --gpu`): auto, on, off, or unset, and the card's memory when it can be read. */
export interface GpuPolicy {
  mode: string
  reserve_gb: number
  min_gb: number
  free_gb?: number
  total_gb?: number
}

/* One row of GET /v1/models/settings: a model turned on or off, where it is pinned, and where it runs now. */
export interface ModelSettingRow {
  id: string
  key: string
  enabled: boolean
  gpu: GpuChoice
  /* The model's own context window in tokens, or null for the default. */
  context?: number | null
  /* The most its family supports; 0 for a model with no context window (a picture, a song, speech). */
  context_max?: number
  state: string
  running?: GpuDecision
}

export interface ModelSettings {
  rows: ModelSettingRow[]
  policy: GpuPolicy | null
}

/* The settings of every model, or null when this server has none to offer (not a router, or an older one). */
export async function getModelSettings(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<ModelSettings | null> {
  const response = await fetch(endpoint(baseUrl, "models/settings"), { headers: headers(apiKey), signal })
  if (response.status === 404) return null
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: ModelSettingRow[]; gpu_policy?: GpuPolicy | null }
  return { rows: body.data ?? [], policy: body.gpu_policy ?? null }
}

/* Turns a model off or on, and/or pins where it runs. Turning one off unloads it: at once when it is idle, or when it has
   finished what it is answering (`unload` says "unloading when idle"). */
export async function setModelSettings(
  baseUrl: string, apiKey: string, key: string, change: { enabled?: boolean; gpu?: GpuChoice; context?: number | "auto" }, signal?: AbortSignal,
): Promise<{ enabled: boolean; gpu: GpuChoice; context?: number | null; unload?: string }> {
  const response = await fetch(endpoint(baseUrl, "models/settings"), {
    method: "POST", headers: headers(apiKey), body: JSON.stringify({ key, ...change }), signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as { enabled: boolean; gpu: GpuChoice; context?: number | null; unload?: string }
}

/* One row of GET /v1/presets: a preset of the router's file, whether it can be used, and why not. */
export interface PresetRow {
  name: string
  model: string
  defaults: Record<string, unknown>
  system?: string
  description?: string
  keep_alive?: string | number
  available: boolean
  reason: string | null
  resolved: string | null
}

export interface PresetList {
  /* False on a plain `coli serve`, which has no presets. */
  supported: boolean
  path: string | null
  presets: PresetRow[]
  /* Entries of the file the router cannot use, and a file it cannot read at all. */
  problems: Array<{ name: string; reason: string }>
  error: string | null
}

export async function listPresets(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<PresetList> {
  const response = await fetch(endpoint(baseUrl, "presets"), { headers: headers(apiKey), signal })
  if (response.status === 404) return { supported: false, path: null, presets: [], problems: [], error: null }
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { path?: string | null; data?: PresetRow[]; problems?: PresetList["problems"]; error?: string | null }
  return { supported: true, path: body.path ?? null, presets: body.data ?? [], problems: body.problems ?? [], error: body.error ?? null }
}

/* What a preset is when it is saved: the fields of the file's entry. */
export interface PresetSpec {
  model: string
  system?: string
  description?: string
  keep_alive?: string | number
  defaults?: Record<string, unknown>
}

/* Creates or replaces a preset in the router's file; it takes effect at once. */
export async function savePreset(baseUrl: string, apiKey: string, name: string, spec: PresetSpec, signal?: AbortSignal): Promise<"created" | "replaced"> {
  const response = await fetch(endpoint(baseUrl, "presets"), {
    method: "POST", headers: headers(apiKey), body: JSON.stringify({ name, ...spec }), signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  return ((await response.json()) as { status: "created" | "replaced" }).status
}

export async function deletePreset(baseUrl: string, apiKey: string, name: string, signal?: AbortSignal): Promise<void> {
  const response = await fetch(endpoint(baseUrl, "presets/delete"), {
    method: "POST", headers: headers(apiKey), body: JSON.stringify({ name }), signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
}

/* One saved voice of the cloning speech models (GET /v1/audio/voices): what it says, and whether the clone uses it (in-context) or only the
   speaker embedding (x-vector). */
export interface SavedVoice {
  name: string
  text: string
  seconds: number
  created: number
  mode: "in-context" | "x-vector"
}

export interface VoiceList {
  /* False on a plain `coli serve`: the saved voices are a router's. */
  supported: boolean
  voices: SavedVoice[]
}

export async function listVoices(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<VoiceList> {
  const response = await fetch(endpoint(baseUrl, "audio/voices"), { headers: headers(apiKey), signal })
  if (response.status === 404) return { supported: false, voices: [] }
  if (!response.ok) throw new Error(await responseError(response))
  return { supported: true, voices: ((await response.json()) as { data?: SavedVoice[] }).data ?? [] }
}

/* Saves a voice from a reference clip (the base64 of an audio file) and what it says (may be empty: the speaker embedding alone is used). */
export async function saveVoice(baseUrl: string, apiKey: string, name: string, audio: string, text: string, signal?: AbortSignal): Promise<SavedVoice & { status: "created" | "replaced" }> {
  const response = await fetch(endpoint(baseUrl, "audio/voices"), { method: "POST", headers: headers(apiKey), body: JSON.stringify({ name, audio, text }), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as SavedVoice & { status: "created" | "replaced" }
}

export async function deleteVoice(baseUrl: string, apiKey: string, name: string, signal?: AbortSignal): Promise<void> {
  const response = await fetch(endpoint(baseUrl, "audio/voices/delete"), { method: "POST", headers: headers(apiKey), body: JSON.stringify({ name }), signal })
  if (!response.ok) throw new Error(await responseError(response))
}

/* Speech in a saved voice, as the WAV the server answers with. */
export async function speakInVoice(baseUrl: string, apiKey: string, model: string, voice: string, input: string, signal?: AbortSignal): Promise<Blob> {
  const response = await fetch(endpoint(baseUrl, "audio/speech"), { method: "POST", headers: computeHeaders(apiKey), body: JSON.stringify({ model, voice, input }), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return await response.blob()
}

/* One request of the router's request log (docs/request-log.md): the router's line with the worker's tokens joined on. */
export interface LoggedRequest {
  ts: string
  trace?: string
  path: string
  status: number
  seconds: number
  model?: string
  preset?: string
  resolved?: string
  stream?: boolean
  prompt_tokens?: number
  completion_tokens?: number
  finish?: string
  error?: string
  engine_seconds?: number
  /* What the client labelled it with (X-Colibri-Tags): app, chat... */
  tags?: Record<string, string | number | boolean>
}

export interface RequestLogView {
  /* False when the router was started without a request log. */
  enabled: boolean
  path: string | null
  requests: LoggedRequest[]
}

/* The router's request log, newest first. A server that has no such route (a plain `coli serve`) is "not enabled". */
export async function listRequests(baseUrl: string, apiKey: string, limit = 100, signal?: AbortSignal, tag?: string): Promise<RequestLogView> {
  const filter = tag ? `&tag=${encodeURIComponent(tag)}` : ""
  const response = await fetch(`${endpoint(baseUrl, "requests")}?limit=${limit}${filter}`, { headers: headers(apiKey), signal })
  if (response.status === 404) return { enabled: false, path: null, requests: [] }
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { enabled?: boolean; path?: string | null; data?: LoggedRequest[] }
  return { enabled: body.enabled === true, path: body.path ?? null, requests: body.data ?? [] }
}

/* The lines the router and its models printed after number `since` (GET /v1/logs). A server that has no such route (a plain
   `coli serve`) is "not available". */
export async function listLogs(baseUrl: string, apiKey: string, since = 0, signal?: AbortSignal): Promise<LogRead> {
  const response = await fetch(`${endpoint(baseUrl, "logs")}?since=${since}&limit=500`, { headers: headers(apiKey), signal })
  if (response.status === 404) return { available: false, lines: [], next: since, first: 0 }
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: LogLine[]; next?: number; first?: number }
  return { available: true, lines: body.data ?? [], next: body.next ?? since, first: body.first ?? 0 }
}

function pause(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"))
    const timer = setTimeout(() => { signal?.removeEventListener("abort", onAbort); resolve() }, ms)
    const onAbort = () => { clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")) }
    signal?.addEventListener("abort", onAbort, { once: true })
  })
}

export interface WaitForModelOptions {
  signal?: AbortSignal
  intervalMs?: number
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>
}

/* Polls until the restarted server answers. There is no deadline: Kimi K3 takes
   forty minutes to load, and the user can leave the page. "ready" means `id` is
   being served. "fell-back" means the server came back on another model after
   being down, which is the launcher returning to the previous one because `id`
   did not start. Connection errors while it restarts are expected, not failures. */
export async function waitForModel(
  baseUrl: string, apiKey: string, id: string, options: WaitForModelOptions = {},
): Promise<"ready" | "fell-back"> {
  const { signal, intervalMs = 2000, sleep = pause } = options
  let wentDown = false
  for (;;) {
    await sleep(intervalMs, signal)
    try {
      const ids = await listModels(baseUrl, apiKey, signal)
      if (ids.includes(id)) return "ready"
      if (wentDown && ids.length) return "fell-back"
    } catch (cause) {
      if (signal?.aborted) throw cause
      wentDown = true
    }
  }
}

export function generatesImages(model: ModelInfo | undefined) {
  return !!model?.capabilities?.includes("image_generation")
}

/* A music model: it answers POST /v1/audio/music and refuses chat */
export function generatesMusic(model: ModelInfo | undefined) {
  return !!model?.capabilities?.includes("music_generation")
}

/* An embedding model: it answers POST /v1/embeddings and refuses chat, so it has no place in the conversation's model list */
export function makesEmbeddings(model: ModelInfo | undefined) {
  return !!model?.capabilities?.includes("embeddings")
}

/* A reranking model: it answers POST /v1/rerank and refuses chat, so it has no place in the conversation's model list either */
export function ranksDocuments(model: ModelInfo | undefined) {
  return !!model?.capabilities?.includes("rerank")
}

/* An image model whose engine can also edit pictures: POST /v1/images/edits */
export function editsImages(model: ModelInfo | undefined) {
  return !!model?.capabilities?.includes("image_edit")
}

export async function getHealth(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "health"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as HealthResponse
}

export async function getProfile(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<ProfileResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "profile"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ProfileResponse
}

export function extractSSE(buffer: string) {
  const frames = buffer.split(/\r?\n\r?\n/)
  const rest = frames.pop() || ""
  const data = frames.flatMap((frame) =>
    frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart()),
  )
  return { data, rest }
}

export interface SSEEvent {
  event: string
  data: string
}

/* extractSSE with the event names kept. The image endpoint tells progress,
   previews and the result apart by the `event:` line, not by the payload, so the
   name has to survive parsing. A frame without an event line is a "message" (the
   SSE default), which is how `data: [DONE]` arrives; comment lines such as
   ": keepalive" and frames without data are dropped, as the spec says. */
export function extractSSEEvents(buffer: string): { events: SSEEvent[]; rest: string } {
  const frames = buffer.split(/\r?\n\r?\n/)
  const rest = frames.pop() || ""
  const events: SSEEvent[] = []
  for (const frame of frames) {
    let event = "message"
    const data: string[] = []
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim() || "message"
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""))
    }
    if (data.length) events.push({ event, data: data.join("\n") })
  }
  return { events, rest }
}

export interface StreamChatOptions {
  baseUrl: string
  apiKey: string
  model: string
  messages: ChatMessage[]
  temperature: number
  maxTokens: number
  enableThinking: boolean
  cacheSlot?: number
  /* Let the router run its built-in tools while it answers (`colibri_tools: true`, docs/builtin-tools.md). The model's own
     tool calls are not forwarded; what the router does arrives through onTool. */
  tools?: boolean
  signal: AbortSignal
  onDelta: (text: string) => void
  onReasoning?: (text: string) => void
  onTool?: (event: ToolEvent) => void
}

export async function streamChat(options: StreamChatOptions): Promise<StreamChatResult> {
  const response = await fetch(endpoint(options.baseUrl, "chat/completions"), {
    method: "POST",
    headers: computeHeaders(options.apiKey),
    signal: options.signal,
    body: JSON.stringify({
      model: options.model,
      /* A turn with pictures goes out in the content-array form the API takes;
         a plain turn stays a string, so a text-only server sees exactly what it
         saw before this existed. */
      messages: options.messages.map(({ role, content, images }) => images?.length
        ? { role, content: [
            ...(content ? [{ type: "text", text: content }] : []),
            ...images.map(url => ({ type: "image_url", image_url: { url } })),
          ] }
        : { role, content }),
      temperature: options.temperature,
      max_completion_tokens: options.maxTokens,
      enable_thinking: options.enableThinking,
      ...(options.cacheSlot === undefined ? {} : { cache_slot: options.cacheSlot }),
      ...(options.tools ? { colibri_tools: true } : {}),
      stream: true,
      stream_options: { include_usage: true },
    }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  if (!response.body) throw new Error("The server returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let finishReason: string | null = null
  let usage: TokenUsage | null = null
  let answeredBy: string | null = null

  const consume = (data: string) => {
    if (data === "[DONE]") return
    const event = JSON.parse(data) as {
      model?: unknown
      error?: { message?: unknown } | string
      choices?: Array<{ delta?: { content?: string; reasoning_content?: string }; finish_reason?: string | null }>
      usage?: TokenUsage | null
      colibri?: unknown
    }
    /* An error frame in the middle of the answer is the server saying why it stopped. */
    if (event.error) {
      const said = typeof event.error === "string" ? event.error : event.error.message
      throw new Error(typeof said === "string" && said.trim() ? said : "The server reported an error while answering.")
    }
    if (!answeredBy && typeof event.model === "string" && event.model) answeredBy = event.model
    /* A chunk with no choices and a `colibri` object is the router telling what its tools are doing. */
    const tool = event.colibri === undefined ? null : toolEventOf(event.colibri)
    if (tool) options.onTool?.(tool)
    const choice = event.choices?.[0]
    const text = choice?.delta?.content
    if (text) options.onDelta(text)
    const reasoning = choice?.delta?.reasoning_content
    if (reasoning) options.onReasoning?.(reasoning)
    if (choice?.finish_reason) finishReason = choice.finish_reason
    if (event.usage) usage = event.usage
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const parsed = extractSSE(buffer)
    buffer = parsed.rest
    parsed.data.forEach(consume)
    if (done) break
  }

  const queueWaitHeader = response.headers.get("x-colibri-queue-wait-ms")
  const parsedQueueWait = queueWaitHeader === null ? null : Number(queueWaitHeader)
  return {
    finishReason,
    usage,
    model: answeredBy,
    requestId: response.headers.get("x-request-id"),
    queueWaitMs: parsedQueueWait !== null && Number.isFinite(parsedQueueWait) ? parsedQueueWait : null,
  }
}

/* The router's built-in tools and the policy that turns them on (GET /v1/tools). A router that predates them, and a plain
   `coli serve`, answer 404: no tools. */
export async function listTools(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<ToolsListing> {
  const response = await fetch(endpoint(baseUrl, "tools"), { headers: headers(apiKey), signal })
  if (response.status === 404) return NO_TOOLS
  if (!response.ok) throw new Error(await responseError(response))
  return parseToolsListing(await response.json())
}

/* What the router answers to a change of its tools policy: the folder as it stored it (a normalised path; null for a web
   change) and the whole policy as it is now. */
export interface ToolsPolicyEdited { folder: string | null; policy: ToolsPolicy }

/* Changes the folders the router's tools may use, or web fetch (POST /v1/tools/policy). A change the router refuses (400 a
   folder that is not there, 404 a folder that is not in the list, 401, 403) throws with the router's own message. */
export async function editToolsPolicy(baseUrl: string, apiKey: string, edit: ToolsPolicyEdit, signal?: AbortSignal): Promise<ToolsPolicyEdited> {
  const response = await fetch(endpoint(baseUrl, "tools/policy"), { method: "POST", headers: headers(apiKey), signal, body: JSON.stringify(edit) })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json().catch(() => null)) as { folder?: unknown; policy?: unknown } | null
  const policy = parsePolicy(body?.policy)
  if (!policy) throw new Error("The router's answer did not include its tools policy.")
  return { folder: typeof body?.folder === "string" ? body.folder : null, policy }
}

/* A person's answer to a call the router is waiting on (POST /v1/tools/pending). 404 is not an error to show as one: the
   question was answered already, or timed out, and the router has forgotten it. Anything else that fails throws. */
export async function answerApproval(
  baseUrl: string, apiKey: string, id: string, allow: boolean, always: boolean, signal?: AbortSignal,
): Promise<Exclude<AnswerResult, "failed">> {
  const response = await fetch(endpoint(baseUrl, "tools/pending"), {
    method: "POST", headers: headers(apiKey), signal,
    body: JSON.stringify({ id, allow, ...(always ? { always: true } : {}) }),
  })
  if (response.status === 404) return "gone"
  if (!response.ok) throw new Error(await responseError(response))
  return "answered"
}

/* System One mode: the model does not generate, it gives a probability to
 * every allowed option. The loop (photographing the shared prefix, one read per
 * option, the normalisation) lives in the gateway, behind POST /v1/systemone,
 * the one decision API for language models and decision models alike: the
 * page's question goes as one `choice` whose labels are the options, and the
 * reply is drawn as bars. The entropy, normalised over the options, is
 * computed here from the probabilities. */
export interface DecisionChoice {
  option: string
  p: number
}

export interface DecisionResponse {
  answer: string
  entropy: number
  choices: DecisionChoice[]
  /* input_tokens: the prompt read; output_tokens: the option tokens a language
     model read (a decision model reads everything as input); nothing is generated */
  usage: { input_tokens: number; output_tokens: number }
}

export async function askSystemOne(
  baseUrl: string,
  apiKey: string,
  model: string,
  state: string,
  question: string,
  options: string[],
  signal?: AbortSignal,
): Promise<DecisionResponse> {
  const response = await fetch(endpoint(baseUrl, "systemone"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({
      model, state,
      questions: { q: { type: "choice", instructions: question, criteria: Object.fromEntries(options.map((o) => [o, null])) } },
    }),
    signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as {
    answers?: { q?: { choice?: string; probabilities?: Record<string, number> } }
    usage?: { input_tokens?: number; output_tokens?: number }
  }
  const probabilities = body.answers?.q?.probabilities ?? {}
  const choices = options
    .map((option) => ({ option, p: probabilities[option] ?? 0 }))
    .sort((a, b) => b.p - a.p)
  const entropy = choices.length > 1
    ? -choices.reduce((sum, c) => sum + (c.p > 0 ? c.p * Math.log(c.p) : 0), 0) / Math.log(choices.length)
    : 0
  return {
    answer: body.answers?.q?.choice ?? choices[0]?.option ?? "",
    entropy,
    choices,
    usage: { input_tokens: body.usage?.input_tokens ?? 0, output_tokens: body.usage?.output_tokens ?? 0 },
  }
}

/* ---- text-to-image ----------------------------------------------------------
 * POST /v1/images/generations, in the OpenAI shape plus colibri's `steps`,
 * `seed` and the `colibri` block of the answer. With `stream: true` the gateway
 * sends named SSE events: progress while the engine works, partial_image when
 * it has a preview, completed with the same object the plain request returns. */

export type ImageStage = "encode" | "denoise" | "decode"

export interface ImageProgress {
  stage: ImageStage
  step: number
  steps: number
  elapsed: number
}

export interface ImageTimings {
  encode?: number
  denoise?: number
  decode?: number
}

export interface ImageGenerationResponse {
  created: number
  data: Array<{ b64_json?: string; url?: string; revised_prompt?: string | null }>
  colibri?: { width: number; height: number; seed: number; steps: number; timings?: ImageTimings }
}

/* One picture of the conversation. The request fields are known from the
   start; url, seconds and timings arrive with the result, and seed becomes the
   one the engine reports having used. */
export interface GeneratedImage {
  prompt: string
  width: number
  height: number
  steps: number
  seed: number
  url?: string
  seconds?: number
  timings?: ImageTimings
  /* How many pictures this was edited from, when it was an edit. The pictures
     themselves are not kept with the turn. */
  edit?: number
}

export interface GenerateImageOptions {
  baseUrl: string
  apiKey: string
  model: string
  prompt: string
  /* Left out together, the size is the engine's choice: for an edit it follows
     the last picture. The values here then only say what to expect. */
  width?: number
  height?: number
  steps: number
  seed: number
  /* Pictures to edit (the prompt is then an instruction about them): the same
     request goes to /v1/images/edits as a multipart form, the way OpenAI's SDKs
     send it. */
  images?: File[]
  /* The side, in pixels, the engine works the pictures at; its own default when absent. */
  refResolution?: number
  signal: AbortSignal
  onProgress?: (progress: ImageProgress) => void
  /* A preview as a data URI, small and noisy: the latent of the step so far. */
  onPartial?: (url: string, index: number) => void
  /* Streaming was not available and the picture is being made by one plain
     request: nothing more will arrive until it is done. */
  onPlainRequest?: () => void
  /* Make it in the background (`Prefer: respond-async`): the router goes on with the picture if this page is closed, and the page
     follows the job. A server that does not know jobs makes it the ordinary way. `onJob` hears each state of the job (queued with
     its position, running, finished); Stop cancels the job. */
  background?: boolean
  onJob?: (job: Job) => void
}

/* The stream did not work as a stream (no body, or a close before the first
   event), so the same request is worth making once without it. Not raised for
   an HTTP error, an abort or a stream that broke halfway: those would only
   start the same generation over, or hide the server's own message. */
class StreamUnavailable extends Error {}

const IMAGE_EVENT = {
  progress: "image_generation.progress",
  partial: "image_generation.partial_image",
  completed: "image_generation.completed",
}

const imagePath = (options: GenerateImageOptions) => options.images?.length ? "images/edits" : "images/generations"

function imageRequest(options: GenerateImageOptions, stream: boolean): RequestInit {
  const fields: Record<string, string | number | boolean> = {
    model: options.model,
    prompt: options.prompt,
    n: 1,
    response_format: "b64_json",
    steps: options.steps,
    seed: options.seed,
    stream,
  }
  if (options.width !== undefined && options.height !== undefined) fields.size = `${options.width}x${options.height}`
  if (options.images?.length) {
    /* multipart: the browser sets the boundary, so no Content-Type here */
    if (options.refResolution !== undefined) fields.ref_resolution = options.refResolution
    const form = new FormData()
    for (const [name, value] of Object.entries(fields)) form.append(name, String(value))
    for (const file of options.images) form.append("image", file, file.name)
    const auth: Record<string, string> = { ...(options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}), ...tagHeaders() }
    return { method: "POST", headers: auth, signal: options.signal, body: form }
  }
  return { method: "POST", headers: computeHeaders(options.apiKey), signal: options.signal, body: JSON.stringify(fields) }
}

async function imageError(response: Response) {
  const fallback = `${response.status} ${response.statusText}`
  try {
    const body = (await response.json()) as OpenAIError
    if (response.status === 400 && body.error?.param === "stream") return new StreamUnavailable(body.error.message || fallback)
    return new Error(body.error?.message || fallback)
  } catch {
    return new Error(fallback)
  }
}

function streamError(data: string) {
  try {
    const body = JSON.parse(data) as OpenAIError & { message?: string }
    return new Error(body.error?.message || body.message || data)
  } catch {
    return new Error(data)
  }
}

async function streamImage(options: GenerateImageOptions): Promise<ImageGenerationResponse> {
  const response = await fetch(endpoint(options.baseUrl, imagePath(options)), imageRequest(options, true))
  if (!response.ok) throw await imageError(response)
  /* A server that ignores `stream` answers with the plain object: that is the
     result, not a failure, and asking again would generate a second picture. */
  if ((response.headers.get("content-type") || "").includes("application/json")) {
    return (await response.json()) as ImageGenerationResponse
  }
  if (!response.body) throw new StreamUnavailable("The server returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let seen = false
  let finished = false
  let result: ImageGenerationResponse | null = null

  const consume = ({ event, data }: SSEEvent) => {
    seen = true
    if (data === "[DONE]") { finished = true; return }
    if (event === "error") throw streamError(data)
    const payload = JSON.parse(data) as Record<string, unknown>
    if (payload.error) throw streamError(data)
    if (event === IMAGE_EVENT.progress) options.onProgress?.(payload as unknown as ImageProgress)
    else if (event === IMAGE_EVENT.partial && typeof payload.b64_json === "string") {
      options.onPartial?.(`data:image/png;base64,${payload.b64_json}`, Number(payload.partial_image_index) || 0)
    } else if (event === IMAGE_EVENT.completed) result = payload as unknown as ImageGenerationResponse
  }

  /* [DONE] ends the read even if the connection stays open; an error event
     cancels the body so the gateway sees the client leave. */
  try {
    while (!finished) {
      const { value, done } = await reader.read()
      const before = buffer.length
      buffer += decoder.decode(value, { stream: !done })
      /* A frame ends at a blank line. A chunk that brings none cannot complete
         one, so the pending frame (the finished image is megabytes of base64)
         is not split again: re-splitting it on every chunk was quadratic. The
         three characters before the chunk catch a boundary cut in half. */
      if (done || /\r?\n\r?\n/.test(buffer.slice(Math.max(0, before - 3)))) {
        const parsed = extractSSEEvents(buffer)
        buffer = parsed.rest
        parsed.events.forEach(consume)
      }
      if (done) break
    }
  } catch (cause) {
    reader.cancel().catch(() => undefined)
    throw cause
  }
  if (finished) reader.cancel().catch(() => undefined)
  if (result) return result
  if (!seen) throw new StreamUnavailable("The server closed the stream without sending anything.")
  throw new Error("The server closed the stream before the image was finished.")
}

async function requestImage(options: GenerateImageOptions): Promise<ImageGenerationResponse> {
  const response = await fetch(endpoint(options.baseUrl, imagePath(options)), imageRequest(options, false))
  if (!response.ok) throw await imageError(response)
  return (await response.json()) as ImageGenerationResponse
}

/* The same request with the header that asks for a job. */
function asJob(init: RequestInit): RequestInit {
  return { ...init, headers: { ...(init.headers as Record<string, string>), ...PREFER_ASYNC } }
}

/* Streams when it can, so the page can show the stage and the previews; falls
   back to one plain request when streaming is not available. In the background
   there is no stream to read: the page asks the job how far it is. */
export async function generateImage(options: GenerateImageOptions): Promise<GeneratedImage> {
  let answer: ImageGenerationResponse
  if (options.background) {
    const outcome = await runBackground({
      baseUrl: options.baseUrl, apiKey: options.apiKey, signal: options.signal,
      send: () => fetch(endpoint(options.baseUrl, imagePath(options)), asJob(imageRequest(options, false))),
      failed: imageError,
      onJob: options.onJob,
      onUpdate: (job) => { options.onJob?.(job); if (job.progress) options.onProgress?.(job.progress as unknown as ImageProgress) },
    })
    if ("plain" in outcome) {
      answer = (await outcome.plain.json()) as ImageGenerationResponse
    } else {
      const kept = outcome.job.result!
      answer = {
        created: outcome.job.created, data: [{ url: outcome.url }],
        colibri: { width: kept.width ?? options.width ?? 0, height: kept.height ?? options.height ?? 0, seed: kept.seed ?? options.seed, steps: kept.steps ?? options.steps, ...(kept.timings ? { timings: kept.timings as ImageTimings } : {}) },
      }
    }
  } else {
    try {
      answer = await streamImage(options)
    } catch (cause) {
      if (!(cause instanceof StreamUnavailable) || options.signal.aborted) throw cause
      options.onPlainRequest?.()
      answer = await requestImage(options)
    }
  }
  const picture = answer.data?.[0]
  const url = picture?.b64_json ? `data:image/png;base64,${picture.b64_json}` : picture?.url
  if (!url) throw new Error("The server answered without an image.")
  const info = answer.colibri
  return {
    prompt: options.prompt,
    width: info?.width ?? options.width ?? 0,
    height: info?.height ?? options.height ?? 0,
    steps: info?.steps ?? options.steps,
    seed: info?.seed ?? options.seed,
    url,
    ...(info?.timings ? { timings: info.timings } : {}),
  }
}

/* ---- text-to-music ----------------------------------------------------------
 * POST /v1/audio/music (docs/acestep.md): a caption and optional lyrics in, a WAV
 * out. Like the image route it takes `stream: true`, and then sends named SSE
 * events: progress while the engine works, completed with the same object the
 * plain request returns. The WAV travels as base64 in `data[0].b64_json`. */

export type MusicStage = "meta" | "plan" | "encode" | "text" | "condition" | "denoise" | "decode"

/* What the engine can be asked to make: new music from a description (the default), a cover of a
   clip, or a clip with a span of it repainted. A reference clip, for the style, goes with any. The guided
   DiT (the base one, when installed) adds three: one track taken out of a clip, one track made to go with a
   clip inside a span, and a partial track completed with others. */
export type MusicTask = "text2music" | "cover" | "edit" | "repaint" | "extract" | "lego" | "complete"
export type RepaintMode = "balanced" | "conservative" | "aggressive"
export const MUSIC_TASKS: readonly MusicTask[] = ["text2music", "cover", "edit", "repaint", "extract", "lego", "complete"]
export const GUIDED_TASKS: readonly MusicTask[] = ["extract", "lego", "complete"]
export type MusicDit = "turbo" | "base" | "sft"
export const MUSIC_DITS: readonly MusicDit[] = ["turbo", "base", "sft"]

export interface MusicProgress {
  stage: MusicStage
  step: number
  steps: number
}

export interface MusicTimings {
  encode?: number
  text?: number
  condition?: number
  denoise?: number
  decode?: number
}

export interface MusicLimits {
  sampleRate: number
  minDuration: number
  maxDuration: number
  defaultDuration: number
  defaultSteps: number
  maxSteps: number
  defaultShift: number
  maxLyricBytes: number
  /* The tasks the engine reports (an older one reports none: it only composes) and, when it takes
     audio in, the longest clip and the length a reference is cut to. */
  tasks: MusicTask[]
  audioIn?: { maxSeconds: number; refSeconds: number }
  /* The 5 Hz planner (a bare caption is enough: it writes the tempo, key, length and the song's structure first), when the
     checkpoint has one; and the guided DiTs, with the tracks they work on. */
  planner?: { lms: string[]; defaultLm: string }
  guided?: { dits: MusicDit[]; defaultSteps: number; maxSteps: number; defaultGuidance: number; tracks: string[] }
  /* The engine takes upstream's sampler options (Heun, the SDE step, the velocity clamp and smoothing), for any DiT. */
  sampler?: boolean
  /* The flow-edit also runs on a cover's context (the clip's own structure kept), not only on silence. */
  flowEditCover?: boolean
}

/* A flow-edit of a clip (the "edit" task of this page: upstream's overlay on text to music, or on a cover): what the clip is now (`caption`, `lyrics`),
   the part of the schedule the edit runs over (`nMin` to `nMax`, 0 to 1), the noise draws averaged a step (`nAvg`, 1 to 16), and whether it runs on a cover's context.
   The prompt and the lyrics of the request are what the clip should become. Each is left out at its default. */
export interface MusicEdit { caption: string; lyrics?: string; nMin?: number; nMax?: number; nAvg?: number; cover?: boolean }

/* The sampler: the plain Euler step (not sent), Heun's second-order one, or the stochastic one. `vnorm` caps a step's velocity as a multiple of the
   latents' norm (above 0), `vema` smooths it with the last step's (above 0, up to 1); each is left out when off. */
export type MusicSampler = "heun" | "sde"
export interface MusicSamplerOptions { sampler?: MusicSampler; vnorm?: number; vema?: number }

/* What the planner chose for a song (colibri.meta): the tempo, the key and scale, the time signature, the language, the length
   and the caption it rewrote for the DiT. Any of it may be missing. */
export interface MusicMeta {
  bpm?: number
  duration?: number
  keyscale?: string
  timesignature?: string
  language?: string
  caption?: string
}

export interface MusicGenerationResponse {
  created: number
  data: Array<{ b64_json?: string; format?: string }>
  colibri?: {
    sample_rate?: number; samples?: number; duration?: number; seed?: number; steps?: number; shift?: number
    task?: MusicTask; src_seconds?: number
    truncated?: boolean; timings?: MusicTimings
    dit?: string; lm?: string; plan_codes?: number; meta?: MusicMeta | null
  }
}

/* One song of the conversation. The request fields are known from the start; url,
   seconds and timings arrive with the result, and seed and duration become the ones
   the engine reports having made. */
export interface GeneratedMusic {
  prompt: string
  lyrics?: string
  language?: string
  duration: number
  steps: number
  seed: number
  /* What was asked: absent is a plain composition. The clips are named, not kept: the files live in
     the page (lib/useMusic.ts) for the buttons that make the song again. */
  task?: MusicTask
  source?: { name: string; seconds?: number }
  reference?: { name: string }
  repaint?: { from: number; to?: number; mode: RepaintMode; strength?: number }
  cover?: { strength: number; noise: number }
  edit?: MusicEdit
  /* The planner and the guided DiT, as asked: `auto` is a length left to the planner (`duration` is then the one it chose). */
  plan?: boolean
  auto?: boolean
  dit?: MusicDit
  guidance?: number
  sampler?: MusicSampler
  vnorm?: number
  vema?: number
  track?: string
  tracks?: string[]
  /* What came back: the planner's choices and the codes it wrote, and the DiT that ran. */
  meta?: MusicMeta
  planCodes?: number
  ranDit?: string
  url?: string
  seconds?: number
  timings?: MusicTimings
  /* The caption or the lyrics were longer than the model reads and were cut. */
  truncated?: boolean
}

export interface GenerateMusicOptions {
  baseUrl: string
  apiKey: string
  model: string
  prompt: string
  lyrics?: string
  language?: string
  duration: number
  steps: number
  seed: number
  task?: MusicTask
  /* The clips, as files the user chose: sent as an upload, the way the picture edit sends its pictures. */
  source?: File
  reference?: File
  repaint?: { from: number; to?: number; mode: RepaintMode; strength?: number }
  cover?: { strength: number; noise: number }
  edit?: MusicEdit
  plan?: boolean
  /* With `plan`: no length is sent and the planner picks it. */
  auto?: boolean
  dit?: MusicDit
  guidance?: number
  sampler?: MusicSampler
  vnorm?: number
  vema?: number
  track?: string
  tracks?: string[]
  signal: AbortSignal
  onProgress?: (progress: MusicProgress) => void
  /* Streaming was not available and the song is being made by one plain request:
     nothing more will arrive until it is done. */
  onPlainRequest?: () => void
  /* In the background, as for a picture (GenerateImageOptions.background). */
  background?: boolean
  onJob?: (job: Job) => void
}

const MUSIC_EVENT = {
  progress: "music_generation.progress",
  completed: "music_generation.completed",
}

/* The fields of a request, with upstream's names for the audio ones (docs/acestep.md). A cover or a
   repaint has no `duration`: its length is the clip's. */
export function musicFields(options: GenerateMusicOptions, stream: boolean) {
  const task = options.task ?? "text2music"
  const fields: Record<string, string | number | boolean> = {
    model: options.model,
    prompt: options.prompt,
    steps: options.steps,
    seed: options.seed,
    stream,
  }
  if (task === "text2music") { if (!(options.plan && options.auto)) fields.duration = options.duration }
  else if (task === "edit") {
    /* a flow-edit is the overlay on text to music (no task sent) or on a cover; the clip's length is the song's */
    fields.flow_edit_morph = true
    if (options.edit?.cover) fields.task = "cover"
  }
  else fields.task = task
  if (task === "complete" && !options.source) fields.duration = options.duration      // no clip: made from the description, so it needs a length
  if (options.plan) fields.thinking = true
  if (options.dit) fields.dit = options.dit
  if (options.guidance !== undefined) fields.guidance_scale = options.guidance
  if (options.sampler === "heun") fields.sampler_mode = "heun"
  if (options.sampler === "sde") fields.infer_method = "sde"
  if (options.vnorm) fields.velocity_norm_threshold = options.vnorm
  if (options.vema) fields.velocity_ema_factor = options.vema
  if ((task === "extract" || task === "lego") && options.track) fields.track_name = options.track
  if (task === "complete" && options.tracks?.length) fields.complete_track_classes = options.tracks.join(",")
  if (task === "lego" && options.repaint) {
    fields.repainting_start = options.repaint.from
    if (options.repaint.to !== undefined) fields.repainting_end = options.repaint.to
  }
  if (options.lyrics?.trim()) fields.lyrics = options.lyrics
  if (options.language) fields.language = options.language
  if (task === "repaint" && options.repaint) {
    fields.repainting_start = options.repaint.from
    if (options.repaint.to !== undefined) fields.repainting_end = options.repaint.to
    if (options.repaint.mode !== "balanced") fields.repaint_mode = options.repaint.mode
    if (options.repaint.mode === "balanced" && options.repaint.strength !== undefined) fields.repaint_strength = options.repaint.strength
  }
  if (task === "edit" && options.edit) {
    const edit = options.edit
    if (edit.caption.trim()) fields.flow_edit_source_caption = edit.caption
    if (edit.lyrics?.trim()) fields.flow_edit_source_lyrics = edit.lyrics
    if (edit.nMin) fields.flow_edit_n_min = edit.nMin
    if (edit.nMax !== undefined && edit.nMax !== 1) fields.flow_edit_n_max = edit.nMax
    if (edit.nAvg && edit.nAvg !== 1) fields.flow_edit_n_avg = edit.nAvg
  }
  if (task === "cover" && options.cover) {
    if (options.cover.strength !== 1) fields.audio_cover_strength = options.cover.strength
    if (options.cover.noise !== 0) fields.cover_noise_strength = options.cover.noise
  }
  return fields
}

function musicRequest(options: GenerateMusicOptions, stream: boolean): RequestInit {
  const fields = musicFields(options, stream)
  if (options.source || options.reference) {
    const form = new FormData()
    for (const [name, value] of Object.entries(fields)) form.append(name, String(value))
    if (options.source) form.append("src_audio", options.source, options.source.name)
    if (options.reference) form.append("reference_audio", options.reference, options.reference.name)
    const auth: Record<string, string> = { ...(options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}), ...tagHeaders() }
    return { method: "POST", headers: auth, signal: options.signal, body: form }
  }
  return { method: "POST", headers: computeHeaders(options.apiKey), signal: options.signal, body: JSON.stringify(fields) }
}

async function streamMusic(options: GenerateMusicOptions): Promise<MusicGenerationResponse> {
  const response = await fetch(endpoint(options.baseUrl, "audio/music"), musicRequest(options, true))
  if (!response.ok) throw await imageError(response)
  /* A server that ignores `stream` answers with the plain object: that is the result,
     not a failure, and asking again would make a second song. */
  if ((response.headers.get("content-type") || "").includes("application/json")) {
    return (await response.json()) as MusicGenerationResponse
  }
  if (!response.body) throw new StreamUnavailable("The server returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let seen = false
  let finished = false
  let result: MusicGenerationResponse | null = null

  const consume = ({ event, data }: SSEEvent) => {
    seen = true
    if (data === "[DONE]") { finished = true; return }
    if (event === "error") throw streamError(data)
    const payload = JSON.parse(data) as Record<string, unknown>
    if (payload.error) throw streamError(data)
    if (event === MUSIC_EVENT.progress) options.onProgress?.(payload as unknown as MusicProgress)
    else if (event === MUSIC_EVENT.completed) result = payload as unknown as MusicGenerationResponse
  }

  try {
    while (!finished) {
      const { value, done } = await reader.read()
      const before = buffer.length
      buffer += decoder.decode(value, { stream: !done })
      /* The finished song is megabytes of base64 in one frame: only a chunk that brings a
         blank line can complete a frame, so the pending one is not re-split on every chunk. */
      if (done || /\r?\n\r?\n/.test(buffer.slice(Math.max(0, before - 3)))) {
        const parsed = extractSSEEvents(buffer)
        buffer = parsed.rest
        parsed.events.forEach(consume)
      }
      if (done) break
    }
  } catch (cause) {
    reader.cancel().catch(() => undefined)
    throw cause
  }
  if (finished) reader.cancel().catch(() => undefined)
  if (result) return result
  if (!seen) throw new StreamUnavailable("The server closed the stream without sending anything.")
  throw new Error("The server closed the stream before the song was finished.")
}

async function requestMusic(options: GenerateMusicOptions): Promise<MusicGenerationResponse> {
  const response = await fetch(endpoint(options.baseUrl, "audio/music"), musicRequest(options, false))
  if (!response.ok) throw await imageError(response)
  return (await response.json()) as MusicGenerationResponse
}

/* The WAV as a playable object URL: a data URI of several megabytes is a poor thing to keep
   in a React state and to hand to an <audio>, and a Blob is what the download wants anyway. */
export function wavUrl(b64: string) {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }))
}

/* Streams when it can, so the page can show the stage; falls back to one plain request
   when streaming is not available. In the background the page follows a job instead. */
export async function generateMusic(options: GenerateMusicOptions): Promise<GeneratedMusic> {
  let answer: MusicGenerationResponse
  let keptUrl: string | undefined
  if (options.background) {
    const outcome = await runBackground({
      baseUrl: options.baseUrl, apiKey: options.apiKey, signal: options.signal,
      send: () => fetch(endpoint(options.baseUrl, "audio/music"), asJob(musicRequest(options, false))),
      failed: imageError,
      onJob: options.onJob,
      onUpdate: (job) => { options.onJob?.(job); if (job.progress) options.onProgress?.(job.progress as unknown as MusicProgress) },
    })
    if ("plain" in outcome) {
      answer = (await outcome.plain.json()) as MusicGenerationResponse
    } else {
      const kept = outcome.job.result!
      keptUrl = outcome.url
      answer = {
        created: outcome.job.created, data: [{ format: "wav" }],
        colibri: { ...(kept.duration !== undefined ? { duration: kept.duration } : {}), ...(kept.seed !== undefined ? { seed: kept.seed } : {}), ...(kept.steps !== undefined ? { steps: kept.steps } : {}), ...(kept.timings ? { timings: kept.timings as MusicTimings } : {}) },
      }
    }
  } else {
    try {
      answer = await streamMusic(options)
    } catch (cause) {
      if (!(cause instanceof StreamUnavailable) || options.signal.aborted) throw cause
      options.onPlainRequest?.()
      answer = await requestMusic(options)
    }
  }
  const song = answer.data?.[0]
  if (!keptUrl && !song?.b64_json) throw new Error("The server answered without a song.")
  const info = answer.colibri
  return {
    prompt: options.prompt,
    ...(options.lyrics?.trim() ? { lyrics: options.lyrics } : {}),
    ...(options.language ? { language: options.language } : {}),
    duration: info?.duration ?? options.duration,
    steps: info?.steps ?? options.steps,
    seed: info?.seed ?? options.seed,
    ...(options.task && options.task !== "text2music" ? { task: options.task } : {}),
    ...(options.source ? { source: { name: options.source.name, ...(info?.src_seconds ? { seconds: info.src_seconds } : {}) } } : {}),
    ...(options.reference ? { reference: { name: options.reference.name } } : {}),
    ...((options.task === "repaint" || options.task === "lego") && options.repaint ? { repaint: options.repaint } : {}),
    ...(options.task === "cover" && options.cover ? { cover: options.cover } : {}),
    ...(options.task === "edit" && options.edit ? { edit: options.edit } : {}),
    ...(options.plan ? { plan: true, ...(options.auto ? { auto: true } : {}) } : {}),
    ...(options.dit ? { dit: options.dit } : {}),
    ...(options.guidance !== undefined ? { guidance: options.guidance } : {}),
    ...(options.sampler ? { sampler: options.sampler } : {}),
    ...(options.vnorm ? { vnorm: options.vnorm } : {}),
    ...(options.vema ? { vema: options.vema } : {}),
    ...(options.track ? { track: options.track } : {}),
    ...(options.tracks?.length ? { tracks: options.tracks } : {}),
    ...(info?.meta && typeof info.meta === "object" ? { meta: info.meta } : {}),
    ...(typeof info?.plan_codes === "number" ? { planCodes: info.plan_codes } : {}),
    ...(typeof info?.dit === "string" ? { ranDit: info.dit } : {}),
    url: keptUrl ?? wavUrl(song!.b64_json!),
    ...(info?.timings ? { timings: info.timings } : {}),
    ...(info?.truncated ? { truncated: true } : {}),
  }
}
