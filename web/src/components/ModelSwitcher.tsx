import { useCallback, useEffect, useRef, useState } from "react"
import { HardDrive, LoaderCircle } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useLocale } from "@/i18n"
import {
  listInstalledModels, switchModel, waitForModel,
  type InstalledModel, type InstalledModels,
} from "@/lib/api"
import { cn } from "@/lib/utils"

interface ModelSwitcherProps {
  baseUrl: string
  apiKey: string
  connected: boolean
  /* A response is being generated: the server would refuse, so do not offer it. */
  busy: boolean
  /* The server is serving the new model: re-read what it serves. */
  onSwitched: () => void
}

function formatSize(bytes: number | null) {
  if (!bytes) return ""
  const gb = bytes / 1e9
  return gb >= 100 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`
}

/* The models downloaded on the machine running the server, and a way to restart it
   on another one. A server serves one model, so a switch is a restart: the server
   goes away for as long as the new model takes to load, and the launcher brings it
   back, on the previous model if the new one does not start (lib/api.ts, waitForModel). */
export default function ModelSwitcher({ baseUrl, apiKey, connected, busy, onSwitched }: ModelSwitcherProps) {
  const { t } = useLocale()
  const [installed, setInstalled] = useState<InstalledModels | null>(null)
  const [loadError, setLoadError] = useState("")
  const [confirmKey, setConfirmKey] = useState<string | null>(null)
  const [switching, setSwitching] = useState<{ id: string; name: string } | null>(null)
  const [result, setResult] = useState<{ kind: "error" | "fell-back"; text: string } | null>(null)
  const waiting = useRef<AbortController | null>(null)

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      setInstalled(await listInstalledModels(baseUrl, apiKey, signal))
      setLoadError("")
    } catch (cause) {
      if (signal?.aborted) return
      setLoadError(cause instanceof Error ? cause.message : String(cause))
    }
  }, [baseUrl, apiKey])

  useEffect(() => {
    if (!connected) return
    const controller = new AbortController()
    void refresh(controller.signal)
    return () => controller.abort()
  }, [connected, refresh])

  useEffect(() => () => waiting.current?.abort(), [])

  const run = async (entry: InstalledModel) => {
    setConfirmKey(null)
    setResult(null)
    const controller = new AbortController()
    waiting.current = controller
    try {
      const target = await switchModel(baseUrl, apiKey, entry.key, controller.signal)
      setSwitching(target)
      const outcome = await waitForModel(baseUrl, apiKey, target.id, { signal: controller.signal })
      if (outcome === "fell-back") setResult({ kind: "fell-back", text: t("models.fellBack", { name: target.name }) })
      onSwitched()
      await refresh()
    } catch (cause) {
      if (controller.signal.aborted) return
      setResult({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    } finally {
      if (waiting.current === controller) waiting.current = null
      setSwitching(null)
    }
  }

  return (
    <section className="side-section model-switcher">
      <div className="section-title"><HardDrive className="size-3.5" /> {t("models.title")}</div>
      <p className="field-help">{t("models.intro")}</p>
      {loadError && <p className="field-help model-error" role="alert">{t("models.loadError", { message: loadError })}</p>}
      {installed && !installed.switch && (
        <p className="field-help">{t("models.unavailable", { reason: installed.reason || t("models.unavailableDefault") })}</p>
      )}
      {switching && (
        <div className="model-switching" role="status" aria-live="polite">
          <LoaderCircle className="size-4 animate-spin" />
          <div><strong>{t("models.switching", { name: switching.name })}</strong><span>{t("models.switchingHelp", { name: switching.name })}</span></div>
        </div>
      )}
      {result && <p className={cn("field-help", "model-result", result.kind)} role="alert">{result.text}</p>}
      <ul className="model-list">
        {installed?.models.map((model) => (
          <li key={model.key} className={cn("model-row", model.current && "current")}>
            <div className="model-info">
              <strong>{model.name}</strong>
              <small>{[model.family, formatSize(model.size_bytes), model.verified ? "" : t("models.unverified")].filter(Boolean).join(" · ")}</small>
              {!model.current && model.reason && <small className="model-reason">{model.reason}</small>}
            </div>
            {model.current ? <span className="model-current">{t("models.current")}</span>
              : confirmKey === model.key ? (
                <div className="model-confirm">
                  <span>{t("models.confirm", { name: model.name })}</span>
                  <Button type="button" size="sm" onClick={() => void run(model)}>{t("models.confirmYes")}</Button>
                  <Button type="button" size="sm" variant="secondary" onClick={() => setConfirmKey(null)}>{t("models.cancel")}</Button>
                </div>
              ) : (
                <Button type="button" size="sm" variant="secondary" disabled={!model.switchable || busy || !!switching}
                  onClick={() => setConfirmKey(model.key)}>{t("models.switch")}</Button>
              )}
          </li>
        ))}
      </ul>
      {installed && installed.models.length <= 1 && <p className="field-help">{t("models.empty")}</p>}
    </section>
  )
}
