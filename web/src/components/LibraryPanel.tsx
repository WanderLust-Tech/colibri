import { useCallback, useEffect, useRef, useState } from "react"
import { Download, Music, Play, RotateCcw, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useLocale } from "@/i18n"
import { cancelJob, isActive, jobFraction, jobTitle, keptFileUrl, listJobs, type Job } from "@/lib/jobs"
import { clock, fileSize, listLibrary, type LibraryEntry, type LibraryKind } from "@/lib/library"
import type { MediaSettings } from "@/lib/mediaMeta"
import { cn } from "@/lib/utils"

interface LibraryPanelProps {
  baseUrl: string
  apiKey: string
  connected: boolean
  /* Whether the server has background jobs (a router): the Jobs part is there only then. */
  jobsAvailable: boolean
  /* Put a file's settings back in the composer (App.applySettings). */
  onMakeAgain: (settings: MediaSettings) => void
}

const PAGE = 24
const JOBS_REFRESH_MS = 2500

/* The kept pictures and songs, and the jobs that are making more. Nothing here needs the page that asked for a render to still be
   open: close the tab, come back, and the job is in the list with how far it got, and the picture is in the grid when it is done. */
export default function LibraryPanel({ baseUrl, apiKey, connected, jobsAvailable, onMakeAgain }: LibraryPanelProps) {
  const { t } = useLocale()
  const [kind, setKind] = useState<LibraryKind | "all">("all")
  const [entries, setEntries] = useState<LibraryEntry[]>([])
  const [total, setTotal] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [error, setError] = useState("")
  const [jobs, setJobs] = useState<Job[]>([])
  const [jobsError, setJobsError] = useState("")
  const finishedSeen = useRef<Set<string> | null>(null)

  const load = useCallback(async (from: number, signal?: AbortSignal) => {
    try {
      const page = await listLibrary(baseUrl, apiKey, { ...(kind !== "all" ? { kind } : {}), limit: PAGE, offset: from }, signal)
      setError("")
      setEntries((current) => from === 0 ? page.entries : [...current, ...page.entries.filter((entry) => !current.some((known) => known.name === entry.name && known.kind === entry.kind))])
      setTotal(page.total)
      setHasMore(page.hasMore)
    } catch (cause) {
      if (signal?.aborted) return
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }, [apiKey, baseUrl, kind])

  useEffect(() => {
    if (!connected) return
    const controller = new AbortController()
    void load(0, controller.signal)
    return () => controller.abort()
  }, [connected, load])

  /* The jobs, every couple of seconds while the page is in front. When one has finished since the last look, the grid is read again:
     its picture is in the library now. */
  useEffect(() => {
    if (!connected || !jobsAvailable) return
    const controller = new AbortController()
    let timer = 0
    const poll = async () => {
      if (document.visibilityState === "visible") {
        try {
          const page = await listJobs(baseUrl, apiKey, { limit: 20 }, controller.signal)
          setJobs(page.jobs)
          setJobsError("")
          const done = new Set(page.jobs.filter((job) => job.status === "succeeded").map((job) => job.id))
          if (finishedSeen.current && [...done].some((id) => !finishedSeen.current!.has(id))) void load(0, controller.signal)
          finishedSeen.current = done
        } catch (cause) {
          if (!controller.signal.aborted) setJobsError(cause instanceof Error ? cause.message : String(cause))
        }
      }
      if (!controller.signal.aborted) timer = window.setTimeout(() => void poll(), JOBS_REFRESH_MS)
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [apiKey, baseUrl, connected, jobsAvailable, load])

  const stop = async (job: Job) => {
    try { await cancelJob(baseUrl, apiKey, job.id) } catch (cause) { setJobsError(cause instanceof Error ? cause.message : String(cause)) }
  }

  const shownJobs = jobs.filter((job) => isActive(job) || job.status === "failed").slice(0, 8)

  return <div className="library-panel">
    {jobsAvailable && <section className="library-jobs" aria-label={t("job.title")}>
      <h3>{t("job.title")}</h3>
      {jobsError && <p className="library-error" role="alert">{jobsError}</p>}
      {shownJobs.length === 0 && !jobsError && <p className="settings-help">{t("job.empty")}</p>}
      <ul>
        {shownJobs.map((job) => <JobRow key={job.id} job={job} onCancel={() => void stop(job)} />)}
      </ul>
    </section>}

    <section className="library-files" aria-label={t("library.title")}>
      <div className="library-head">
        <h3>{t("library.title")}</h3>
        <div className="library-filter" role="group" aria-label={t("library.filter")}>
          {(["all", "images", "music"] as const).map((value) => <button key={value} type="button" aria-pressed={kind === value} onClick={() => setKind(value)}>
            {t(value === "all" ? "library.all" : value === "images" ? "library.pictures" : "library.songs")}
          </button>)}
        </div>
        <span className="library-count">{t("library.count", { n: total })}</span>
      </div>
      {error && <p className="library-error" role="alert">{error}</p>}
      {!error && entries.length === 0 && <p className="settings-help">{t("library.empty")}</p>}
      <div className="library-grid">
        {entries.map((entry) => <LibraryCard key={`${entry.kind}/${entry.name}`} entry={entry} baseUrl={baseUrl} apiKey={apiKey} onMakeAgain={onMakeAgain} />)}
      </div>
      {hasMore && <Button type="button" variant="secondary" className="library-more" onClick={() => void load(entries.length)}>{t("library.more")}</Button>}
    </section>
  </div>
}

function JobRow({ job, onCancel }: { job: Job; onCancel: () => void }) {
  const { t } = useLocale()
  const fraction = job.status === "running" ? jobFraction(job) : null
  const where = job.status === "queued" ? t("job.queued", { n: job.position ?? 1 })
    : job.status === "running" ? (job.progress?.step && job.progress?.steps ? t("job.step", { step: job.progress.step, steps: job.progress.steps }) : t("job.running"))
    : job.error?.message ?? t(`job.${job.status}`)
  return <li className={cn("job-row", job.status)} data-status={job.status}>
    <span className="job-kind">{job.kind === "image" ? "🖼" : "♪"}</span>
    <span className="job-title" title={jobTitle(job)}>{jobTitle(job)}</span>
    <span className="job-where">{where}</span>
    {isActive(job) && <button type="button" className="job-cancel" onClick={onCancel} aria-label={t("job.cancel")} title={t("job.cancel")}><X /></button>}
    {job.status === "running" && <div className={cn("image-bar", fraction === null && "waiting")} role="progressbar" aria-label={t("image.progress")}
      aria-valuemin={0} aria-valuemax={100} aria-valuenow={fraction === null ? undefined : Math.round(fraction * 100)}>
      <span style={{ width: `${((fraction ?? 0) * 100).toFixed(1)}%` }} />
    </div>}
  </li>
}

/* A picture is fetched when its card comes into view (the key goes in a header, which an <img src> cannot carry); a song when
   Play is pressed, since a song is megabytes. */
function useVisible() {
  const ref = useRef<HTMLDivElement | null>(null)
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    const node = ref.current
    if (!node || seen) return
    if (typeof IntersectionObserver === "undefined") { setSeen(true); return }
    const watcher = new IntersectionObserver((hits) => { if (hits.some((hit) => hit.isIntersecting)) { setSeen(true); watcher.disconnect() } }, { rootMargin: "200px" })
    watcher.observe(node)
    return () => watcher.disconnect()
  }, [seen])
  return { ref, seen }
}

function LibraryCard({ entry, baseUrl, apiKey, onMakeAgain }: { entry: LibraryEntry; baseUrl: string; apiKey: string; onMakeAgain: (settings: MediaSettings) => void }) {
  const { t, locale } = useLocale()
  const { ref, seen } = useVisible()
  const [url, setUrl] = useState<string | null>(null)
  const [problem, setProblem] = useState("")
  const picture = entry.kind === "images"

  const fetchFile = useCallback(async () => {
    try { const found = await keptFileUrl(baseUrl, apiKey, entry.url); setUrl(found); setProblem(""); return found } catch (cause) { setProblem(cause instanceof Error ? cause.message : String(cause)); return null }
  }, [apiKey, baseUrl, entry.url])

  useEffect(() => { if (picture && seen) void fetchFile() }, [fetchFile, picture, seen])

  const download = async () => {
    const found = url ?? await fetchFile()
    if (!found) return
    const anchor = document.createElement("a")
    anchor.href = found
    anchor.download = entry.name
    anchor.click()
  }
  const play = async () => { await fetchFile() }

  const prompt = entry.settings?.prompt ?? entry.name
  const facts = [
    picture && entry.width && entry.height ? `${entry.width}×${entry.height}` : null,
    !picture && entry.duration !== undefined ? clock(entry.duration) : null,
    entry.settings?.seed !== undefined ? `${t("library.seed")} ${entry.settings.seed}` : null,
    fileSize(entry.bytes),
  ].filter(Boolean).join(" · ")
  const when = new Date(entry.modified * 1000).toLocaleString(locale)
  const ratio = picture && entry.width && entry.height ? `${entry.width} / ${entry.height}` : undefined

  return <article className="library-card" data-kind={entry.kind}>
    <div className="library-media" ref={ref} style={picture ? { aspectRatio: ratio ?? "1 / 1" } : undefined}>
      {picture
        ? (url ? <img src={url} alt={prompt} /> : <span className={cn("library-placeholder", problem && "bad")}>{problem || ""}</span>)
        : (url ? <audio controls autoPlay src={url} aria-label={prompt} />
          : <button type="button" className="library-play" onClick={() => void play()} aria-label={t("library.play")}><Play /><Music /></button>)}
    </div>
    <p className="library-prompt" title={prompt}>{prompt}</p>
    <p className="library-facts">{facts}</p>
    <p className="library-when">{when}</p>
    <div className="library-actions">
      {entry.settings && <button type="button" onClick={() => onMakeAgain(entry.settings!)} title={t("library.makeAgainHelp")}><RotateCcw />{t("library.makeAgain")}</button>}
      <button type="button" onClick={() => void download()} aria-label={t("library.download")} title={t("library.download")}><Download />{t("library.download")}</button>
    </div>
  </article>
}
