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
}

export interface StreamChatResult {
  finishReason: string | null
  usage: TokenUsage | null
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
  }
  return limits.minDuration <= limits.maxDuration ? limits : undefined
}

export async function listModelInfo(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<ModelInfo[]> {
  const response = await fetch(endpoint(baseUrl, "models"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: Array<{ id: string; capabilities?: unknown; image?: unknown; audio?: unknown }> }
  return (body.data || []).map(({ id, capabilities, image, audio }) => {
    if (!Array.isArray(capabilities)) return { id }
    const list = capabilities.filter((item): item is string => typeof item === "string")
    const edit = list.includes("image_edit") ? editLimits(image) : undefined
    const music = list.includes("music_generation") ? musicLimits(audio) : undefined
    return { id, capabilities: list, ...(edit ? { edit } : {}), ...(music ? { music } : {}) }
  })
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
}

export interface InstalledModels {
  /* False when this server cannot restart itself (not started by the coli launcher). */
  switch: boolean
  reason: string | null
  models: InstalledModel[]
}

export async function listInstalledModels(baseUrl: string, apiKey: string, signal?: AbortSignal): Promise<InstalledModels> {
  const response = await fetch(endpoint(baseUrl, "models/installed"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { switch?: boolean; reason?: string | null; data?: InstalledModel[] }
  return { switch: body.switch === true, reason: body.reason ?? null, models: body.data ?? [] }
}

/* Asks the server to restart on another installed model. It answers 202 and goes
   down: follow it with waitForModel. */
export async function switchModel(baseUrl: string, apiKey: string, key: string, signal?: AbortSignal): Promise<{ id: string; name: string }> {
  const response = await fetch(endpoint(baseUrl, "models/switch"), {
    method: "POST", headers: headers(apiKey), body: JSON.stringify({ key }), signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { id: string; name: string }
  return { id: body.id, name: body.name }
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
  signal: AbortSignal
  onDelta: (text: string) => void
  onReasoning?: (text: string) => void
}

export async function streamChat(options: StreamChatOptions): Promise<StreamChatResult> {
  const response = await fetch(endpoint(options.baseUrl, "chat/completions"), {
    method: "POST",
    headers: headers(options.apiKey),
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

  const consume = (data: string) => {
    if (data === "[DONE]") return
    const event = JSON.parse(data) as {
      choices?: Array<{ delta?: { content?: string; reasoning_content?: string }; finish_reason?: string | null }>
      usage?: TokenUsage | null
    }
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
    requestId: response.headers.get("x-request-id"),
    queueWaitMs: parsedQueueWait !== null && Number.isFinite(parsedQueueWait) ? parsedQueueWait : null,
  }
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
    const auth: Record<string, string> = options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}
    return { method: "POST", headers: auth, signal: options.signal, body: form }
  }
  return { method: "POST", headers: headers(options.apiKey), signal: options.signal, body: JSON.stringify(fields) }
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

/* Streams when it can, so the page can show the stage and the previews; falls
   back to one plain request when streaming is not available. */
export async function generateImage(options: GenerateImageOptions): Promise<GeneratedImage> {
  let answer: ImageGenerationResponse
  try {
    answer = await streamImage(options)
  } catch (cause) {
    if (!(cause instanceof StreamUnavailable) || options.signal.aborted) throw cause
    options.onPlainRequest?.()
    answer = await requestImage(options)
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

export type MusicStage = "text" | "condition" | "denoise" | "decode"

export interface MusicProgress {
  stage: MusicStage
  step: number
  steps: number
}

export interface MusicTimings {
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
}

export interface MusicGenerationResponse {
  created: number
  data: Array<{ b64_json?: string; format?: string }>
  colibri?: {
    sample_rate?: number; samples?: number; duration?: number; seed?: number; steps?: number; shift?: number
    truncated?: boolean; timings?: MusicTimings
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
  signal: AbortSignal
  onProgress?: (progress: MusicProgress) => void
  /* Streaming was not available and the song is being made by one plain request:
     nothing more will arrive until it is done. */
  onPlainRequest?: () => void
}

const MUSIC_EVENT = {
  progress: "music_generation.progress",
  completed: "music_generation.completed",
}

function musicRequest(options: GenerateMusicOptions, stream: boolean): RequestInit {
  const fields: Record<string, string | number | boolean> = {
    model: options.model,
    prompt: options.prompt,
    duration: options.duration,
    steps: options.steps,
    seed: options.seed,
    stream,
  }
  if (options.lyrics?.trim()) fields.lyrics = options.lyrics
  if (options.language) fields.language = options.language
  return { method: "POST", headers: headers(options.apiKey), signal: options.signal, body: JSON.stringify(fields) }
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
   when streaming is not available. */
export async function generateMusic(options: GenerateMusicOptions): Promise<GeneratedMusic> {
  let answer: MusicGenerationResponse
  try {
    answer = await streamMusic(options)
  } catch (cause) {
    if (!(cause instanceof StreamUnavailable) || options.signal.aborted) throw cause
    options.onPlainRequest?.()
    answer = await requestMusic(options)
  }
  const song = answer.data?.[0]
  if (!song?.b64_json) throw new Error("The server answered without a song.")
  const info = answer.colibri
  return {
    prompt: options.prompt,
    ...(options.lyrics?.trim() ? { lyrics: options.lyrics } : {}),
    ...(options.language ? { language: options.language } : {}),
    duration: info?.duration ?? options.duration,
    steps: info?.steps ?? options.steps,
    seed: info?.seed ?? options.seed,
    url: wavUrl(song.b64_json),
    ...(info?.timings ? { timings: info.timings } : {}),
    ...(info?.truncated ? { truncated: true } : {}),
  }
}
