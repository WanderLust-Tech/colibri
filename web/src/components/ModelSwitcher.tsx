import { contextChoices, contextLabel } from "@/lib/contextWindow"
import { useCallback, useEffect, useRef, useState } from "react"
import { HardDrive, LoaderCircle } from "lucide-react"

import { Button } from "@/components/ui/button"
import ModelInstaller from "@/components/ModelInstaller"
import { useLocale } from "@/i18n"
import {
  cancelInstall, getModelCatalog, uninstallAddon, getModelSettings, installActive, installModel, listInstalledModels, setModelSettings,
  switchModel, uninstallModel, unloadModel, waitForModel,
  type CatalogRow, type GpuChoice, type InstalledModel, type InstalledModels, type ModelCatalog, type ModelSettings,
} from "@/lib/api"
import { formatBytes } from "@/lib/installs"
import { detailsLine } from "@/lib/modelDetails"
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

/* The models downloaded on the machine running the server, and a way to use another one.
   A plain server serves one model, so a switch is a restart: the server goes away for as
   long as the new model takes to load, and the launcher brings it back, on the previous
   model if the new one does not start (lib/api.ts, waitForModel). A `coli router` holds
   several: Load starts one (the call answers when it is ready) and nothing restarts. */
export default function ModelSwitcher({ baseUrl, apiKey, connected, busy, onSwitched }: ModelSwitcherProps) {
  const { t, locale } = useLocale()
  const [installed, setInstalled] = useState<InstalledModels | null>(null)
  const [loadError, setLoadError] = useState("")
  const [confirmKey, setConfirmKey] = useState<string | null>(null)
  const [switching, setSwitching] = useState<{ id: string; name: string } | null>(null)
  const [result, setResult] = useState<{ kind: "error" | "fell-back" | "info"; text: string } | null>(null)
  const [unloading, setUnloading] = useState<string | null>(null)
  const [settings, setSettings] = useState<ModelSettings | null>(null)
  const [changing, setChanging] = useState<string | null>(null)
  /* What can be installed, or null when this router cannot (an older one, or not a router). */
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null)
  const [removeKey, setRemoveKey] = useState<string | null>(null)
  const [removing, setRemoving] = useState<string | null>(null)
  const waiting = useRef<AbortController | null>(null)
  const wasInstalling = useRef(false)
  const lastInstalling = useRef("")
  const router = installed?.router === true

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const list = await listInstalledModels(baseUrl, apiKey, signal)
      setInstalled(list)
      /* a router that keeps model settings says which models are off and where each is pinned; any other server has none */
      setSettings(list.router ? await getModelSettings(baseUrl, apiKey, signal).catch(() => null) : null)
      setCatalog(list.router ? await getModelCatalog(baseUrl, apiKey, signal).catch(() => null) : null)
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

  /* While a download runs, ask the router how far it is; when it ends, re-read the models that are here. */
  const installing = !!catalog && catalog.rows.some((row) => installActive(row.job))
  useEffect(() => {
    if (!connected || !installing) return
    const timer = window.setInterval(() => {
      getModelCatalog(baseUrl, apiKey).then((next) => { if (next) setCatalog(next) }).catch(() => undefined)
    }, 1500)
    return () => window.clearInterval(timer)
  }, [connected, installing, baseUrl, apiKey])

  useEffect(() => {
    const active = catalog?.rows.find((row) => installActive(row.job))
    if (active) lastInstalling.current = active.id
    if (wasInstalling.current && !installing) {
      const row = catalog?.rows.find((candidate) => candidate.id === lastInstalling.current)
      if (row?.job?.state === "done") setResult({ kind: "info", text: t("install.done", { name: row.name }) })
      else if (row?.job?.state === "failed") setResult({ kind: "error", text: t("install.failed", { message: row.job.error ?? "" }) })
      void refresh()
      onSwitched()
    }
    wasInstalling.current = installing
  }, [installing, catalog, refresh, onSwitched, t])

  useEffect(() => () => waiting.current?.abort(), [])

  const run = async (entry: InstalledModel) => {
    setConfirmKey(null)
    setResult(null)
    const controller = new AbortController()
    waiting.current = controller
    try {
      if (router) setSwitching({ id: entry.id, name: entry.name })     // the call itself waits for the load
      const target = await switchModel(baseUrl, apiKey, entry.key, controller.signal)
      setSwitching(target)
      const outcome = router ? "ready" : await waitForModel(baseUrl, apiKey, target.id, { signal: controller.signal })
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

  /* Router only. A model that is answering is not cut off: it goes when it has finished. */
  const unload = async (entry: InstalledModel) => {
    setResult(null)
    setUnloading(entry.key)
    try {
      const outcome = await unloadModel(baseUrl, apiKey, entry.key)
      if (outcome.status !== "unloaded") setResult({ kind: "info", text: t("models.unloadBusy", { name: entry.name }) })
      onSwitched()
      await refresh()
    } catch (cause) {
      setResult({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    } finally {
      setUnloading(null)
    }
  }

  /* Router with settings only. Turning a model off unloads it, after the request it is answering. */
  const change = async (entry: InstalledModel, what: { enabled?: boolean; gpu?: GpuChoice; context?: number | "auto" }) => {
    setResult(null)
    setChanging(entry.key)
    try {
      const outcome = await setModelSettings(baseUrl, apiKey, entry.key, what)
      if (outcome.unload === "unloading when idle") setResult({ kind: "info", text: t("models.unloadBusy", { name: entry.name }) })
      onSwitched()
      await refresh()
    } catch (cause) {
      setResult({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    } finally {
      setChanging(null)
    }
  }

  /* Router with install routes only. The confirmation (size and licence) was asked by ModelInstaller. */
  const install = async (row: CatalogRow) => {
    setResult(null)
    try {
      await installModel(baseUrl, apiKey, row.id)
      const next = await getModelCatalog(baseUrl, apiKey)
      if (next) setCatalog(next)
    } catch (cause) {
      setResult({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    }
  }

  const stopInstall = async (row: CatalogRow) => {
    try {
      await cancelInstall(baseUrl, apiKey, row.id)
    } catch (cause) {
      setResult({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    }
  }

  /* Deletes an add-on's files (the planner, the base DiT). The router refuses while the model it belongs to is loaded. */
  const removeAddon = async (row: CatalogRow) => {
    setResult(null)
    try {
      const outcome = await uninstallAddon(baseUrl, apiKey, row.id)
      setResult({ kind: "info", text: t("install.addonRemoved", { name: row.name, size: formatBytes(outcome.freedBytes) }) })
      const next = await getModelCatalog(baseUrl, apiKey)
      if (next) setCatalog(next)
    } catch (cause) {
      setResult({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    }
  }

  /* Deletes the model's files. The router refuses a model that is loaded, and the button is off for one. */
  const remove = async (entry: InstalledModel) => {
    setRemoveKey(null)
    setResult(null)
    setRemoving(entry.key)
    try {
      const outcome = await uninstallModel(baseUrl, apiKey, entry.key)
      const text = t("models.removed", { name: entry.name, size: formatBytes(outcome.freedBytes) })
      setResult({ kind: "info", text: outcome.presets.length ? `${text} ${t("models.removedPresets", { names: outcome.presets.join(", ") })}` : text })
      onSwitched()
      await refresh()
    } catch (cause) {
      setResult({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    } finally {
      setRemoving(null)
    }
  }

  const policy = settings?.policy
  const policyText = !settings ? "" : !policy || policy.mode === "unset" ? t("models.gpuUnset")
    : typeof policy.free_gb === "number" && typeof policy.total_gb === "number"
      ? t("models.gpuPolicy", { mode: policy.mode, free: policy.free_gb, total: policy.total_gb, reserve: policy.reserve_gb })
      : t("models.gpuPolicyNoReading", { mode: policy.mode, reserve: policy.reserve_gb })
  const gpuLabel = (choice: GpuChoice) => t(choice === "vulkan" ? "models.gpuVulkan" : choice === "cpu" ? "models.gpuCpu" : "models.gpuAuto")

  return (
    <section className="side-section model-switcher">
      <div className="section-title"><HardDrive className="size-3.5" /> {t("models.title")}</div>
      <p className="field-help">{t(router ? "models.routerIntro" : "models.intro")}</p>
      {loadError && <p className="field-help model-error" role="alert">{t("models.loadError", { message: loadError })}</p>}
      {installed && !installed.switch && (
        <p className="field-help">{t("models.unavailable", { reason: installed.reason || t("models.unavailableDefault") })}</p>
      )}
      {switching && (
        <div className="model-switching" role="status" aria-live="polite">
          <LoaderCircle className="size-4 animate-spin" />
          <div>
            <strong>{t(router ? "models.loading" : "models.switching", { name: switching.name })}</strong>
            <span>{t(router ? "models.loadingHelp" : "models.switchingHelp", { name: switching.name })}</span>
          </div>
        </div>
      )}
      {result && <p className={cn("field-help", "model-result", result.kind)} role="alert">{result.text}</p>}
      {policyText && <p className="field-help model-gpu-policy">{policyText}</p>}
      <ul className="model-list">
        {installed?.models.map((model) => {
          const setting = settings?.rows.find((row) => row.key === model.key)
          const enabled = setting ? setting.enabled : model.enabled !== false
          return (
          <li key={model.key} className={cn("model-row", model.current && "current", !enabled && "off")}>
            <div className="model-info">
              <strong>{model.name}</strong>
              <small>{[model.family, formatSize(model.size_bytes), model.verified ? "" : t("models.unverified"),
                router && model.current && model.resident_bytes ? t("models.inMemory", { size: formatSize(model.resident_bytes) }) : "",
                router && model.current && typeof model.expires_in === "number" ? t("models.unloadsIn", { minutes: Math.max(1, Math.ceil(model.expires_in / 60)) }) : "",
                router && model.current && model.keep_alive === null && !model.inflight ? t("models.stays") : "",
                router && model.current && model.gpu && model.gpu.mode !== "unchanged" ? t(model.gpu.mode === "vulkan" ? "models.onGpu" : "models.onCpu") : "",
                router && !enabled ? t("models.off") : "",
              ].filter(Boolean).join(" · ")}</small>
              {model.details && detailsLine(model.details, (count) => t("models.contextUpTo", { n: count }), (repo) => t("models.source", { repo }), locale) &&
                <small className="model-details">{detailsLine(model.details, (count) => t("models.contextUpTo", { n: count }), (repo) => t("models.source", { repo }), locale)}</small>}
              {model.details?.problems?.length ? <small className="model-reason">{model.details.problems.join("; ")}</small> : null}
              {!model.current && model.reason && <small className="model-reason">{model.reason}</small>}
              {router && !model.current && model.last_error && <small className="model-reason">{t("models.lastError", { message: model.last_error })}</small>}
            </div>
            {router && (model.state === "starting" || model.state === "stopping") ? <span className="model-current">{t(model.state === "starting" ? "models.starting" : "models.stopping")}</span>
              : model.current && router ? (
                <div className="model-confirm">
                  <span className="model-current">{t("models.loaded")}</span>
                  <Button type="button" size="sm" variant="secondary" disabled={unloading !== null || !!switching}
                    onClick={() => void unload(model)}>{t(unloading === model.key ? "models.unloading" : "models.unload")}</Button>
                </div>
              )
              : model.current ? <span className="model-current">{t("models.current")}</span>
              : router ? (
                <Button type="button" size="sm" variant="secondary" disabled={!model.switchable || !enabled || busy || !!switching}
                  onClick={() => void run(model)}>{t("models.load")}</Button>
              )
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
            {router && (setting || catalog) && (
              <div className="model-controls">
                {setting && (
                  <>
                    <label>
                      <span>{t("models.runsOn")}</span>
                      <select className="model-select" aria-label={`${t("models.runsOn")} ${model.name}`} title={t("models.runsOnHelp")} value={setting.gpu}
                        disabled={changing !== null} onChange={(event) => void change(model, { gpu: event.target.value as GpuChoice })}>
                        {(["auto", "vulkan", "cpu"] as const).map((choice) => <option key={choice} value={choice}>{gpuLabel(choice)}</option>)}
                      </select>
                    </label>
                    {(setting.context_max ?? 0) > 0 && (
                      <label>
                        <span>{t("models.context")}</span>
                        <select className="model-select" aria-label={`${t("models.context")} ${model.name}`} title={t("models.contextHelp")} value={setting.context ?? 0}
                          disabled={changing !== null} onChange={(event) => void change(model, { context: Number(event.target.value) || "auto" })}>
                          {contextChoices(setting.context ?? null, setting.context_max ?? 0).map((tokens) =>
                            <option key={tokens} value={tokens}>{tokens === 0 ? t("models.contextDefault") : contextLabel(tokens)}</option>)}
                        </select>
                      </label>
                    )}
                    <Button type="button" size="sm" variant="secondary" disabled={changing !== null}
                      aria-label={`${t(enabled ? "models.turnOff" : "models.turnOn")} ${model.name}`}
                      onClick={() => void change(model, { enabled: !enabled })}>{t(enabled ? "models.turnOff" : "models.turnOn")}</Button>
                  </>
                )}
                {catalog && (removeKey === model.key ? (
                  <span className="model-confirm">
                    <span>{t("models.removeConfirm", { name: model.name, size: model.size_bytes ? ` (${formatSize(model.size_bytes)})` : "" })}</span>
                    <Button type="button" size="sm" onClick={() => void remove(model)}>{t("models.removeYes")}</Button>
                    <Button type="button" size="sm" variant="secondary" onClick={() => setRemoveKey(null)}>{t("models.cancel")}</Button>
                  </span>
                ) : (
                  <Button type="button" size="sm" variant="secondary"
                    disabled={model.current || model.missing === true || model.state === "starting" || model.state === "stopping" || removing !== null || !!switching}
                    title={model.current ? t("models.removeLoaded") : undefined}
                    aria-label={`${t("models.remove")} ${model.name}`} onClick={() => setRemoveKey(model.key)}>
                    {t(removing === model.key ? "models.removing" : "models.remove")}
                  </Button>
                ))}
              </div>
            )}
          </li>
          )
        })}
      </ul>
      {installed && installed.models.length <= 1 && !catalog && <p className="field-help">{t("models.empty")}</p>}
      {router && catalog && (
        <ModelInstaller catalog={catalog} onInstall={(row) => void install(row)} onStop={(row) => void stopInstall(row)} onRemoveAddon={(row) => void removeAddon(row)} disabled={!!switching || removing !== null} />
      )}
    </section>
  )
}
