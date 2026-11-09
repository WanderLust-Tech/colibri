import { useCallback, useEffect, useState } from "react"
import { LoaderCircle, RefreshCw, ScrollText } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useLocale } from "@/i18n"
import { listRequests, type LoggedRequest, type RequestLogView } from "@/lib/api"
import { cn } from "@/lib/utils"

interface RequestsPanelProps {
  baseUrl: string
  apiKey: string
  connected: boolean
}

const REFRESH_MS = 5000

/* The time of day of an ISO time the log wrote in UTC, in the viewer's own zone. */
function clock(ts: string): string {
  const date = new Date(ts)
  return Number.isNaN(date.getTime()) ? ts : date.toLocaleTimeString()
}

/* "/v1/chat/completions" -> "chat/completions": the part that tells requests apart. */
const shortPath = (path: string) => path.replace(/^\/v1\//, "")

function tokens(request: LoggedRequest): string {
  if (request.prompt_tokens === undefined && request.completion_tokens === undefined) return ""
  return `${request.prompt_tokens ?? 0} → ${request.completion_tokens ?? 0}`
}

/* What the router's request log holds (docs/request-log.md): the last requests it answered, newest first, with the model or
   preset asked for, how it ended and how long it took. Reads again every few seconds while the tab is open. */
export default function RequestsPanel({ baseUrl, apiKey, connected }: RequestsPanelProps) {
  const { t } = useLocale()
  const [view, setView] = useState<RequestLogView | null>(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true)
    try {
      setView(await listRequests(baseUrl, apiKey, 100, signal))
      setError("")
    } catch (cause) {
      if (signal?.aborted) return
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }, [baseUrl, apiKey])

  useEffect(() => {
    if (!connected) return
    const controller = new AbortController()
    void refresh(controller.signal)
    const timer = window.setInterval(() => void refresh(controller.signal), REFRESH_MS)
    return () => { controller.abort(); window.clearInterval(timer) }
  }, [connected, refresh])

  return (
    <section className="side-section requests-panel">
      <div className="section-title"><ScrollText className="size-3.5" /> {t("requests.title")}</div>
      <p className="field-help">{t("requests.intro")}</p>
      {error && <p className="field-help model-error" role="alert">{t("models.loadError", { message: error })}</p>}
      {view && !view.enabled && <p className="field-help">{t("requests.off")}</p>}
      {view?.enabled && <p className="field-help">{t("requests.where", { path: view.path ?? "" })}</p>}
      <Button type="button" variant="secondary" onClick={() => void refresh()} disabled={loading}>
        {loading ? <LoaderCircle className="size-4 animate-spin" /> : <RefreshCw className="size-4" />} {t("requests.refresh")}
      </Button>
      {view?.enabled && view.requests.length === 0 && <p className="field-help">{t("requests.empty")}</p>}
      {view?.enabled && view.requests.length > 0 && (
        <div className="requests-scroll">
          <table className="requests-table">
            <thead><tr>
              <th>{t("requests.time")}</th><th>{t("requests.what")}</th><th>{t("requests.model")}</th>
              <th>{t("requests.status")}</th><th>{t("requests.seconds")}</th><th>{t("requests.tokens")}</th>
            </tr></thead>
            <tbody>
              {view.requests.map((request, index) => (
                <tr key={`${request.trace ?? request.ts}-${index}`} className={cn(request.status >= 400 && "failed")}>
                  <td>{clock(request.ts)}</td>
                  <td>{shortPath(request.path)}</td>
                  <td>{request.preset ? <>{request.preset} <small>({request.resolved ?? request.model})</small></> : (request.resolved ?? request.model ?? "")}</td>
                  <td title={request.error ?? request.finish ?? ""}>{request.status}{request.error ? ` ${request.error}` : ""}</td>
                  <td>{request.seconds.toFixed(request.seconds < 10 ? 2 : 1)}</td>
                  <td>{tokens(request)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
