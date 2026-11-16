import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Pause, Play, Terminal, Trash2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useLocale } from "@/i18n"
import { listLogs } from "@/lib/api"
import { clockTime, lostLines, mergeLines, sourcesOf, type LogLine } from "@/lib/logTail"
import { cn } from "@/lib/utils"

interface LogPanelProps {
  baseUrl: string
  apiKey: string
  connected: boolean
}

const REFRESH_MS = 2000

/* What the router and the models it started printed lately: a model loading, a start that failed, the GPU giving way to the
   CPU. The page asks every couple of seconds for the lines after the last it saw. Nothing is stored; a reload starts from what
   the router still holds in memory. */
export default function LogPanel({ baseUrl, apiKey, connected }: LogPanelProps) {
  const { t } = useLocale()
  const [lines, setLines] = useState<LogLine[]>([])
  const [available, setAvailable] = useState(true)
  const [error, setError] = useState("")
  const [paused, setPaused] = useState(false)
  const [gap, setGap] = useState(false)
  const [source, setSource] = useState("")
  const seen = useRef(0)
  const scroller = useRef<HTMLDivElement | null>(null)
  const stuck = useRef(true)            /* follow the newest line unless the reader scrolled up */

  const read = useCallback(async (signal?: AbortSignal) => {
    try {
      const result = await listLogs(baseUrl, apiKey, seen.current, signal)
      setError("")
      setAvailable(result.available)
      if (!result.available) return
      if (lostLines(result, seen.current)) setGap(true)
      seen.current = Math.max(seen.current, result.next)
      setLines((held) => mergeLines(held, result.lines))
    } catch (cause) {
      if (signal?.aborted) return
      setError(cause instanceof Error ? cause.message : String(cause))
    }
  }, [baseUrl, apiKey])

  useEffect(() => {
    if (!connected || paused) return
    const controller = new AbortController()
    void read(controller.signal)
    const timer = window.setInterval(() => void read(controller.signal), REFRESH_MS)
    return () => { controller.abort(); window.clearInterval(timer) }
  }, [connected, paused, read])

  const shown = useMemo(() => source ? lines.filter((line) => line.source === source) : lines, [lines, source])
  const sources = useMemo(() => sourcesOf(lines), [lines])

  useEffect(() => {
    const box = scroller.current
    if (box && stuck.current) box.scrollTop = box.scrollHeight
  }, [shown])

  const onScroll = () => {
    const box = scroller.current
    if (box) stuck.current = box.scrollHeight - box.scrollTop - box.clientHeight < 24
  }

  return (
    <section className="side-section log-panel">
      <div className="section-title"><Terminal className="size-3.5" /> {t("log.title")}</div>
      <p className="field-help">{t("log.intro")}</p>
      {error && <p className="field-help model-error" role="alert">{t("models.loadError", { message: error })}</p>}
      {!available && <p className="field-help">{t("log.unavailable")}</p>}
      {available && (
        <div className="log-controls">
          <Button type="button" variant="secondary" onClick={() => setPaused((value) => !value)}>
            {paused ? <Play className="size-4" /> : <Pause className="size-4" />} {paused ? t("log.resume") : t("log.pause")}
          </Button>
          <Button type="button" variant="secondary" onClick={() => { setLines([]); setGap(false) }} disabled={!lines.length}>
            <Trash2 className="size-4" /> {t("log.clear")}
          </Button>
          <label className="log-source">
            <span>{t("log.source")}</span>
            <select value={source} onChange={(event) => setSource(event.target.value)}>
              <option value="">{t("log.all")}</option>
              {sources.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
        </div>
      )}
      {available && gap && <p className="field-help">{t("log.dropped")}</p>}
      {available && shown.length === 0 && <p className="field-help">{t("log.empty")}</p>}
      {available && shown.length > 0 && (
        <div className="log-scroll" ref={scroller} onScroll={onScroll} role="log" aria-live="off">
          {shown.map((line) => (
            <div key={line.seq} className={cn("log-line", /fail|error|did not start|cannot|stopped answering/i.test(line.text) && "bad")}>
              <span className="log-time">{clockTime(line.time)}</span>
              <span className="log-source-name">{line.source}</span>
              <span className="log-text">{line.text}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
