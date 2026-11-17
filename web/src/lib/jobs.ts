/* Background jobs and the kept files (c/jobs.py, c/library.py; docs/api.md "Background jobs"). A picture or a song asked for with
   `Prefer: respond-async` is made apart from the connection: the router answers 202 with a job, and this page can follow it, close,
   and come back to it later. Nothing here needs the page to stay open for the job to go on. */

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled"
export const JOB_STATUSES: readonly JobStatus[] = ["queued", "running", "succeeded", "failed", "cancelled"]

export interface JobResult {
  kind: "images" | "music"
  file: string
  url: string
  width?: number
  height?: number
  seed?: number
  steps?: number
  duration?: number
  timings?: Record<string, number>
}

export interface Job {
  id: string
  kind: "image" | "music"
  model: string
  status: JobStatus
  created: number
  started: number | null
  finished: number | null
  /* While queued: 1 is next. */
  position?: number
  /* While running, when the engine reports it. */
  progress?: Record<string, number | string>
  request: Record<string, string | number>
  result?: JobResult
  error?: { code?: string | null; message: string }
}

export const isFinal = (job: Pick<Job, "status">) => job.status === "succeeded" || job.status === "failed" || job.status === "cancelled"
export const isActive = (job: Pick<Job, "status">) => !isFinal(job)

/* The dashboard's base URL is the API's (`http://host:8000/v1`); the routes here are written out in full (`/v1/jobs`), as the server's are. */
export const serverRoot = (baseUrl: string) => baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "")
const base = serverRoot
const auth = (apiKey: string): Record<string, string> => (apiKey ? { Authorization: `Bearer ${apiKey}` } : {})

function simple(value: unknown): Record<string, string | number> {
  const out: Record<string, string | number> = {}
  if (typeof value === "object" && value !== null) {
    for (const [key, item] of Object.entries(value)) if (typeof item === "string" || typeof item === "number") out[key] = item
  }
  return out
}

function numbers(value: unknown): Record<string, number | string> | undefined {
  if (typeof value !== "object" || value === null) return undefined
  const out: Record<string, number | string> = {}
  for (const [key, item] of Object.entries(value)) if (typeof item === "string" || (typeof item === "number" && Number.isFinite(item))) out[key] = item
  return out
}

function parseResult(value: unknown): JobResult | undefined {
  if (typeof value !== "object" || value === null) return undefined
  const found = value as Record<string, unknown>
  if ((found.kind !== "images" && found.kind !== "music") || typeof found.file !== "string" || typeof found.url !== "string") return undefined
  const result: JobResult = { kind: found.kind, file: found.file, url: found.url }
  for (const key of ["width", "height", "seed", "steps", "duration"] as const) {
    const item = found[key]
    if (typeof item === "number" && Number.isFinite(item)) result[key] = item
  }
  if (typeof found.timings === "object" && found.timings !== null) {
    const timings: Record<string, number> = {}
    for (const [key, item] of Object.entries(found.timings)) if (typeof item === "number") timings[key] = item
    result.timings = timings
  }
  return result
}

/* A job as the server sends it, or null when it is not one (a server that answered something else). Fields of the wrong type
   are dropped, not trusted. */
export function parseJob(value: unknown): Job | null {
  if (typeof value !== "object" || value === null) return null
  const found = value as Record<string, unknown>
  if (typeof found.id !== "string" || !JOB_STATUSES.includes(found.status as JobStatus)) return null
  if (found.kind !== "image" && found.kind !== "music") return null
  const job: Job = {
    id: found.id,
    kind: found.kind,
    model: typeof found.model === "string" ? found.model : "",
    status: found.status as JobStatus,
    created: typeof found.created === "number" ? found.created : 0,
    started: typeof found.started === "number" ? found.started : null,
    finished: typeof found.finished === "number" ? found.finished : null,
    request: simple(found.request),
  }
  if (typeof found.position === "number") job.position = found.position
  const progress = numbers(found.progress)
  if (progress) job.progress = progress
  const result = parseResult(found.result)
  if (result) job.result = result
  if (typeof found.error === "object" && found.error !== null) {
    const error = found.error as Record<string, unknown>
    job.error = { code: typeof error.code === "string" ? error.code : null, message: typeof error.message === "string" ? error.message : "The job failed." }
  }
  return job
}

async function failure(response: Response) {
  const fallback = `${response.status} ${response.statusText}`.trim()
  try {
    const body = (await response.json()) as { error?: { message?: string } }
    return new Error(body.error?.message || fallback)
  } catch {
    return new Error(fallback)
  }
}

export interface JobPage {
  jobs: Job[]
  total: number
  hasMore: boolean
}

export async function listJobs(baseUrl: string, apiKey: string, query: { status?: JobStatus; limit?: number; offset?: number } = {}, signal?: AbortSignal): Promise<JobPage> {
  const parts = Object.entries(query).filter(([, value]) => value !== undefined).map(([key, value]) => `${key}=${encodeURIComponent(String(value))}`)
  const response = await fetch(`${base(baseUrl)}/v1/jobs${parts.length ? `?${parts.join("&")}` : ""}`, { headers: auth(apiKey), signal })
  if (!response.ok) throw await failure(response)
  const body = (await response.json()) as { data?: unknown[]; total?: number; has_more?: boolean }
  const jobs = (body.data ?? []).map(parseJob).filter((job): job is Job => job !== null)
  return { jobs, total: typeof body.total === "number" ? body.total : jobs.length, hasMore: body.has_more === true }
}

export async function getJob(baseUrl: string, apiKey: string, id: string, signal?: AbortSignal): Promise<Job> {
  const response = await fetch(`${base(baseUrl)}/v1/jobs/${encodeURIComponent(id)}`, { headers: auth(apiKey), signal })
  if (!response.ok) throw await failure(response)
  const job = parseJob(await response.json())
  if (!job) throw new Error("The server's answer about the job was not understood.")
  return job
}

export async function cancelJob(baseUrl: string, apiKey: string, id: string): Promise<{ cancelled: boolean; status: string | null }> {
  const response = await fetch(`${base(baseUrl)}/v1/jobs/cancel`, {
    method: "POST", headers: { "Content-Type": "application/json", ...auth(apiKey) }, body: JSON.stringify({ id }),
  })
  if (!response.ok) throw await failure(response)
  const body = (await response.json()) as { cancelled?: boolean; status?: string | null }
  return { cancelled: body.cancelled === true, status: typeof body.status === "string" ? body.status : null }
}

const sleepFor = (ms: number, signal?: AbortSignal) => new Promise<void>((resolve) => {
  const timer = setTimeout(done, ms)
  function done() { clearTimeout(timer); signal?.removeEventListener("abort", done); resolve() }
  signal?.addEventListener("abort", done)
})

/* How many answers in a row may fail before the page gives up on following: a sleeping laptop or a restarting router is not the
   end of a job that is running on the server. */
const MAX_FAILED_POLLS = 6

export interface FollowOptions {
  baseUrl: string
  apiKey: string
  job: Job
  /* Stop is the user's: it cancels the job (the page was told to stop making it), then throws AbortError. */
  signal?: AbortSignal
  onUpdate?: (job: Job) => void
  pollMs?: number
}

/* Ask about a job until it is finished. Returns the finished job. When `signal` aborts the job is cancelled first. When the
   server cannot be reached six times in a row this throws and leaves the job alone: it is still going on, and the Library page
   shows it. */
export async function followJob(options: FollowOptions): Promise<Job> {
  const { baseUrl, apiKey, signal, onUpdate, pollMs = 1500 } = options
  let job = options.job
  let failed = 0
  while (!isFinal(job)) {
    await sleepFor(pollMs, signal)
    if (signal?.aborted) {
      await cancelJob(baseUrl, apiKey, job.id).catch(() => undefined)
      throw new DOMException("The job was cancelled.", "AbortError")
    }
    try {
      job = await getJob(baseUrl, apiKey, job.id, signal)
      failed = 0
    } catch (cause) {
      if (signal?.aborted) continue
      if (++failed >= MAX_FAILED_POLLS) throw new Error(`Lost contact with the server; the job keeps going (${cause instanceof Error ? cause.message : cause}). Look in the Library.`)
      continue
    }
    onUpdate?.(job)
  }
  return job
}

export type BackgroundOutcome = { job: Job; url: string } | { plain: Response }

export interface BackgroundOptions {
  baseUrl: string
  apiKey: string
  signal: AbortSignal
  /* Sends the request, with the header that asks for a job already on it. */
  send: () => Promise<Response>
  /* The error for a response that is not ok. */
  failed: (response: Response) => Promise<Error>
  onJob?: (job: Job) => void
  onUpdate?: (job: Job) => void
  pollMs?: number
}

/* Make something in the background and wait for it, the way the dashboard shows a picture or a song: the request goes with
   `Prefer: respond-async`, the 202 names the job, the page follows it, and the finished file is fetched (the key goes in a header,
   which an <img src> cannot carry, so the bytes come back as an object URL). A server that does not know background jobs ignores
   the header and answers the request itself: that response comes back as `plain`, to be read the ordinary way. */
export async function runBackground(options: BackgroundOptions): Promise<BackgroundOutcome> {
  const response = await options.send()
  if (!response.ok) throw await options.failed(response)
  if (response.status !== 202) return { plain: response }
  const submitted = parseJob(await response.json())
  if (!submitted) throw new Error("The server's answer about the job was not understood.")
  options.onJob?.(submitted)
  const job = await followJob({ baseUrl: options.baseUrl, apiKey: options.apiKey, job: submitted, signal: options.signal, onUpdate: options.onUpdate, pollMs: options.pollMs })
  if (job.status === "failed") throw new Error(job.error?.message || "The job failed.")
  if (job.status === "cancelled") throw new Error("The job was cancelled.")
  if (!job.result) throw new Error("The job finished without a result.")
  return { job, url: await keptFileUrl(options.baseUrl, options.apiKey, job.result.url, options.signal) }
}

/* Header value for a request that wants a job. */
export const PREFER_ASYNC = { Prefer: "respond-async" } as const

const objectUrls = new Map<string, Promise<string>>()

/* A kept picture or song as an object URL, fetched once per page. Entries are forgotten when the fetch fails, so asking again tries again. */
export function keptFileUrl(baseUrl: string, apiKey: string, path: string, signal?: AbortSignal): Promise<string> {
  const key = `${base(baseUrl)}${path}`
  const cached = objectUrls.get(key)
  if (cached) return cached
  const pending = (async () => {
    const response = await fetch(key, { headers: auth(apiKey), signal })
    if (!response.ok) throw await failure(response)
    return URL.createObjectURL(await response.blob())
  })()
  objectUrls.set(key, pending)
  pending.catch(() => objectUrls.delete(key))
  return pending
}

/* For tests: forget what was fetched. */
export function forgetKeptFiles() {
  objectUrls.clear()
}

/* How far along a running job is, 0 to 1, or null when the engine has said nothing yet. */
export function jobFraction(job: Job): number | null {
  const step = Number(job.progress?.step)
  const steps = Number(job.progress?.steps)
  if (!Number.isFinite(step) || !Number.isFinite(steps) || steps <= 0) return null
  return Math.max(0, Math.min(1, step / steps))
}

/* The one line that says what a job asked for. */
export function jobTitle(job: Job): string {
  const prompt = typeof job.request.prompt === "string" ? job.request.prompt : ""
  return prompt || (job.kind === "image" ? "Picture" : "Song")
}
