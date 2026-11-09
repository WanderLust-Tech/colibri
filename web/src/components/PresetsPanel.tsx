import { useCallback, useEffect, useState, type FormEvent } from "react"
import { Bookmark, LoaderCircle, Pencil, Play, Plus, Trash2, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { useLocale } from "@/i18n"
import {
  deletePreset, listInstalledModels, listPresets, savePreset,
  type InstalledModel, type PresetList, type PresetRow, type PresetSpec,
} from "@/lib/api"
import { defaultText, defaultsOf, nameProblem, parseDefault, rowsOf, suggestedKeys, type DefaultRow } from "@/lib/presets"

interface PresetsPanelProps {
  baseUrl: string
  apiKey: string
  connected: boolean
  /* A preset was saved or removed: the model list the page keeps needs reading again. */
  onChanged: () => void
  /* Chat with this preset: the page selects it and goes to the conversation. */
  onUse: (name: string) => void
}

interface Draft {
  /* The preset being edited, or null for a new one: its name cannot change. */
  original: string | null
  name: string
  model: string
  description: string
  system: string
  keepAlive: string
  rows: DefaultRow[]
}

const blank = (model: string): Draft => ({ original: null, name: "", model, description: "", system: "", keepAlive: "", rows: [] })

const draftOf = (row: PresetRow): Draft => ({
  original: row.name, name: row.name, model: row.model, description: row.description ?? "", system: row.system ?? "",
  keepAlive: row.keep_alive === undefined ? "" : defaultText(row.keep_alive), rows: rowsOf(row.defaults),
})

function specOf(draft: Draft): PresetSpec {
  const defaults = defaultsOf(draft.rows)
  const keep = draft.keepAlive.trim()
  return {
    model: draft.model,
    ...(draft.description.trim() ? { description: draft.description.trim() } : {}),
    ...(draft.system.trim() ? { system: draft.system } : {}),
    ...(keep ? { keep_alive: parseDefault(keep) as string | number } : {}),
    ...(Object.keys(defaults).length ? { defaults } : {}),
  }
}

/* The presets of `coli router`: models with a name, a system prompt and settings of their own. They are in the model list of the
   conversation like any model; here they are made, changed and removed. The router keeps them in its presets file
   (docs/presets.md) and uses a change at once. */
export default function PresetsPanel({ baseUrl, apiKey, connected, onChanged, onUse }: PresetsPanelProps) {
  const { t } = useLocale()
  const [list, setList] = useState<PresetList | null>(null)
  const [models, setModels] = useState<InstalledModel[]>([])
  const [loadError, setLoadError] = useState("")
  const [draft, setDraft] = useState<Draft | null>(null)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ kind: "error" | "ok"; text: string } | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const [presets, installed] = await Promise.all([listPresets(baseUrl, apiKey, signal), listInstalledModels(baseUrl, apiKey, signal)])
      setList(presets)
      setModels(installed.models.filter((model) => model.modality !== "decision"))
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

  const modelOf = (id: string) => models.find((model) => model.id === id || model.key === id || model.name === id)
  const patch = (changes: Partial<Draft>) => setDraft((current) => current && { ...current, ...changes })
  const patchRow = (index: number, changes: Partial<DefaultRow>) =>
    setDraft((current) => current && { ...current, rows: current.rows.map((row, at) => at === index ? { ...row, ...changes } : row) })

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!draft) return
    setMessage(null)
    setSaving(true)
    try {
      const outcome = await savePreset(baseUrl, apiKey, draft.name.trim().toLowerCase(), specOf(draft))
      setMessage({ kind: "ok", text: t(outcome === "created" ? "presets.created" : "presets.replaced", { name: draft.name.trim().toLowerCase() }) })
      setDraft(null)
      onChanged()
      await refresh()
    } catch (cause) {
      setMessage({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    } finally {
      setSaving(false)
    }
  }

  const remove = async (name: string) => {
    setMessage(null)
    setConfirmDelete(null)
    try {
      await deletePreset(baseUrl, apiKey, name)
      setMessage({ kind: "ok", text: t("presets.deleted", { name }) })
      if (draft?.original === name) setDraft(null)
      onChanged()
      await refresh()
    } catch (cause) {
      setMessage({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    }
  }

  const problem = draft && !draft.original ? nameProblem(draft.name) : null
  const modality = draft ? modelOf(draft.model)?.modality ?? "text" : "text"
  const summary = (row: PresetRow) => Object.entries(row.defaults).map(([key, value]) => `${key} ${defaultText(value)}`).join(", ")

  return (
    <section className="side-section preset-panel">
      <div className="section-title"><Bookmark className="size-3.5" /> {t("presets.title")}</div>
      <p className="field-help">{t("presets.intro")}</p>
      {loadError && <p className="field-help model-error" role="alert">{t("models.loadError", { message: loadError })}</p>}
      {list && !list.supported && <p className="field-help">{t("presets.unsupported")}</p>}
      {list?.error && <p className="field-help model-error" role="alert">{t("presets.fileError", { message: list.error })}</p>}
      {list?.problems.map((item) => <p key={item.name} className="field-help model-reason">{t("presets.problem", { name: item.name, reason: item.reason })}</p>)}
      {message && <p className={`field-help model-result ${message.kind === "error" ? "error" : ""}`} role={message.kind === "error" ? "alert" : "status"}>{message.text}</p>}
      {list?.supported && !draft && <>
        {list.presets.length === 0 && <p className="field-help">{t("presets.empty")}</p>}
        <ul className="model-list">
          {list.presets.map((row) => (
            <li key={row.name} className="model-row preset-row" data-preset={row.name}>
              <div className="model-info">
                <strong>{row.name}</strong>
                <small>{[t("presets.runs", { model: row.model }), summary(row)].filter(Boolean).join(" · ")}</small>
                {row.description && <small>{row.description}</small>}
                {row.system && <small className="preset-system">{t("presets.systemLine", { text: row.system.length > 120 ? `${row.system.slice(0, 119)}…` : row.system })}</small>}
                {!row.available && row.reason && <small className="model-reason">{t("presets.unavailable", { reason: row.reason })}</small>}
              </div>
              {confirmDelete === row.name ? (
                <div className="model-confirm">
                  <span>{t("presets.confirmDelete", { name: row.name })}</span>
                  <Button type="button" size="sm" onClick={() => void remove(row.name)}>{t("presets.delete")}</Button>
                  <Button type="button" size="sm" variant="secondary" onClick={() => setConfirmDelete(null)}>{t("models.cancel")}</Button>
                </div>
              ) : (
                <div className="model-confirm">
                  <Button type="button" size="sm" variant="secondary" disabled={!row.available} onClick={() => onUse(row.name)}><Play className="size-3.5" /> {t("presets.use")}</Button>
                  <Button type="button" size="sm" variant="secondary" aria-label={t("presets.edit")} title={t("presets.edit")} onClick={() => { setMessage(null); setDraft(draftOf(row)) }}><Pencil className="size-3.5" /></Button>
                  <Button type="button" size="sm" variant="secondary" aria-label={t("presets.delete")} title={t("presets.delete")} onClick={() => setConfirmDelete(row.name)}><Trash2 className="size-3.5" /></Button>
                </div>
              )}
            </li>
          ))}
        </ul>
        <Button type="button" variant="secondary" disabled={models.length === 0} onClick={() => { setMessage(null); setDraft(blank(models[0]?.id ?? "")) }}>
          <Plus className="size-4" /> {t("presets.new")}
        </Button>
      </>}
      {draft && (
        <form className="preset-form" onSubmit={(event) => void submit(event)}>
          <h3>{t(draft.original ? "presets.editing" : "presets.creating", { name: draft.original ?? "" })}</h3>
          <label>{t("presets.name")}
            <Input id="preset-name" value={draft.name} disabled={draft.original !== null} placeholder="terse" autoComplete="off"
              onChange={(event) => patch({ name: event.target.value })} />
            <span className="field-help">{problem === "invalid" ? t("presets.nameInvalid") : t("presets.nameHelp")}</span>
          </label>
          <label>{t("presets.model")}
            <select id="preset-model" value={draft.model} onChange={(event) => patch({ model: event.target.value })}>
              {!modelOf(draft.model) && <option value={draft.model}>{draft.model}</option>}
              {models.map((model) => <option key={model.key} value={model.id}>{model.name}</option>)}
            </select>
          </label>
          <label>{t("presets.description")}
            <Input id="preset-description" value={draft.description} onChange={(event) => patch({ description: event.target.value })} />
          </label>
          {modality === "text" && <label>{t("presets.system")}
            <Textarea id="preset-system" rows={4} value={draft.system} onChange={(event) => patch({ system: event.target.value })} />
            <span className="field-help">{t("presets.systemHelp")}</span>
          </label>}
          <fieldset className="preset-defaults">
            <legend>{t("presets.defaults")}</legend>
            <span className="field-help">{t("presets.defaultsHelp")}</span>
            <datalist id="preset-keys">{suggestedKeys(modality).map((key) => <option key={key} value={key} />)}</datalist>
            {draft.rows.map((row, index) => (
              <div className="preset-default-row" key={index}>
                <Input aria-label={t("presets.setting")} list="preset-keys" value={row.key} placeholder={suggestedKeys(modality)[0]} onChange={(event) => patchRow(index, { key: event.target.value })} />
                <Input aria-label={t("presets.value")} value={row.value} onChange={(event) => patchRow(index, { value: event.target.value })} />
                <Button type="button" size="sm" variant="secondary" aria-label={t("presets.removeSetting")} title={t("presets.removeSetting")}
                  onClick={() => patch({ rows: draft.rows.filter((_, at) => at !== index) })}><X className="size-3.5" /></Button>
              </div>
            ))}
            <Button type="button" size="sm" variant="secondary" onClick={() => patch({ rows: [...draft.rows, { key: "", value: "" }] })}><Plus className="size-3.5" /> {t("presets.addSetting")}</Button>
          </fieldset>
          <label>{t("presets.keepAlive")}
            <Input id="preset-keep-alive" value={draft.keepAlive} placeholder="10m" onChange={(event) => patch({ keepAlive: event.target.value })} />
            <span className="field-help">{t("presets.keepAliveHelp")}</span>
          </label>
          <div className="preset-actions">
            <Button type="submit" disabled={saving || nameProblem(draft.name) !== null || !draft.model}>
              {saving && <LoaderCircle className="size-4 animate-spin" />} {t("presets.save")}
            </Button>
            <Button type="button" variant="secondary" disabled={saving} onClick={() => setDraft(null)}>{t("models.cancel")}</Button>
          </div>
        </form>
      )}
    </section>
  )
}
