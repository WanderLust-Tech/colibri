import { useState } from "react"
import { Download } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useLocale } from "@/i18n"
import { installActive, type CatalogRow, type ModelCatalog } from "@/lib/api"
import { canInstall, formatBytes, installable, installedAddons, installPercent, progressLine } from "@/lib/installs"

interface ModelInstallerProps {
  catalog: ModelCatalog
  /* Starts the download; the licence and size were shown by the confirmation this component asks for. */
  onInstall: (row: CatalogRow) => void
  onStop: (row: CatalogRow) => void
  /* Deletes an installed add-on's files; the confirmation was asked here. */
  onRemoveAddon: (row: CatalogRow) => void
  disabled: boolean
}

/* The models a router can download, below the ones that are here: size, kind, licence and whether this machine fits it, an
   Install button that first shows the size and the licence and asks, and a progress bar with a stop button while it
   downloads (c/model_installer.py, docs/model-install.md). */
export default function ModelInstaller({ catalog, onInstall, onStop, onRemoveAddon, disabled }: ModelInstallerProps) {
  const { t } = useLocale()
  const [confirmId, setConfirmId] = useState<string | null>(null)
  const [removeId, setRemoveId] = useState<string | null>(null)
  const rows = installable(catalog.rows)
  const addons = installedAddons(catalog.rows)
  const running = catalog.rows.some((row) => installActive(row.job))

  const status = (row: CatalogRow) => {
    if (row.status === "needs-disk") return t("install.noRoom")
    if (row.status === "unsupported") return t("install.unsupported")
    if (row.status === "needs-parent") return t("install.needsParent")
    if (row.status === "needs-ram") return t("install.needsRam")
    return ""
  }

  return (
    <div className="model-install">
      <div className="section-title"><Download className="size-3.5" /> {t("install.title")}</div>
      <p className="field-help">{t("install.intro", { free: catalog.freeGb !== null ? t("install.free", { gb: catalog.freeGb }) : "" })}</p>
      {rows.length === 0 && <p className="field-help">{t("install.allHere")}</p>}
      <ul className="model-list">
        {rows.map((row) => {
          const job = row.job
          const active = installActive(job)
          const flag = status(row)
          return (
            <li key={row.id} className="model-row install-row">
              <div className="model-info">
                <strong>{row.name}</strong>
                <small>
                  <span className="install-kind">{t(`install.kind.${row.kind}`)}</span> · {row.download_gb.toFixed(1)} GB
                  {row.summary ? ` · ${row.summary}` : ""}
                </small>
                <small>{row.license_note ? t("install.licence", { text: row.license_note }) : t("install.noLicence")}</small>
                <small className={row.fits ? undefined : "model-reason"}>{flag ? `${flag}: ` : ""}{row.reason}</small>
                {row.unfinished && !active && <small>{t("install.partial", { size: formatBytes(row.partial_bytes) })}</small>}
                {!active && job?.state === "failed" && <small className="model-reason" role="alert">{t("install.failed", { message: job.error ?? "" })}</small>}
                {!active && job?.state === "cancelled" && <small>{t("install.cancelled")}</small>}
              </div>
              {active ? (
                <Button type="button" size="sm" variant="secondary" onClick={() => onStop(row)}>{t("install.stop")}</Button>
              ) : confirmId === row.id ? (
                <div className="model-confirm">
                  <span>{t("install.confirm", { size: row.download_gb.toFixed(1), name: row.name })}</span>
                  <Button type="button" size="sm" onClick={() => { setConfirmId(null); onInstall(row) }}>{t("install.confirmYes")}</Button>
                  <Button type="button" size="sm" variant="secondary" onClick={() => setConfirmId(null)}>{t("install.cancel")}</Button>
                </div>
              ) : (
                <Button type="button" size="sm" variant="secondary" disabled={disabled || running || !canInstall(row, false)}
                  aria-label={`${t("install.install")} ${row.name}`} onClick={() => setConfirmId(row.id)}>{t("install.install")}</Button>
              )}
              {active && job && (
                <div className="install-progress" role="status" aria-live="polite">
                  <div className="install-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(installPercent(job))}>
                    <span style={{ width: `${installPercent(job)}%` }} />
                  </div>
                  <small>
                    {job.state === "starting" ? t("install.starting") : job.state === "finishing" ? t("install.finishing")
                      : `${Math.floor(installPercent(job))}% · ${progressLine(job)}`}
                  </small>
                </div>
              )}
            </li>
          )
        })}
      </ul>
      {addons.length > 0 && (
        <>
          <div className="section-title">{t("install.addons")}</div>
          <p className="field-help">{t("install.addonsHelp")}</p>
          <ul className="model-list">
            {addons.map((row) => (
              <li key={row.id} className="model-row install-row">
                <div className="model-info">
                  <strong>{row.name}</strong>
                  <small>{formatBytes(row.installed?.size_bytes)}{row.summary ? ` · ${row.summary}` : ""}</small>
                </div>
                {removeId === row.id ? (
                  <div className="model-confirm">
                    <span>{t("install.removeConfirm", { name: row.name, size: formatBytes(row.installed?.size_bytes) })}</span>
                    <Button type="button" size="sm" onClick={() => { setRemoveId(null); onRemoveAddon(row) }}>{t("install.removeYes")}</Button>
                    <Button type="button" size="sm" variant="secondary" onClick={() => setRemoveId(null)}>{t("install.cancel")}</Button>
                  </div>
                ) : (
                  <Button type="button" size="sm" variant="secondary" disabled={disabled || running}
                    aria-label={`${t("install.remove")} ${row.name}`} onClick={() => setRemoveId(row.id)}>{t("install.remove")}</Button>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  )
}
