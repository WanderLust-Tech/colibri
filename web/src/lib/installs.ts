import type { CatalogRow, InstallJob, ModelFolder } from "./api"

/* The numbers an install row shows. Plain digits and unit letters, the same in every language. */

export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes || bytes <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"]
  let value = bytes
  let unit = 0
  while (value >= 1000 && unit < units.length - 1) { value /= 1000; unit += 1 }
  return unit <= 1 ? `${Math.round(value)} ${units[unit]}` : value >= 100 ? `${Math.round(value)} ${units[unit]}` : `${value.toFixed(1)} ${units[unit]}`
}

/* Whether the confirmation offers a choice of folder: more than one models folder, and a model that is not an add-on (it goes into
   its parent's folder) or a part-fetched one (it carries on where it was begun). */
export function offersFolder(folders: ModelFolder[] | undefined, row: Pick<CatalogRow, "into" | "unfinished">): boolean {
  return (folders?.length ?? 0) > 1 && !row.into && !row.unfinished
}

/* The folder the choice starts at: the router's own pick, else the first. */
export function defaultFolder(folders: ModelFolder[]): string {
  return (folders.find((folder) => folder.preferred) ?? folders[0])?.path ?? ""
}

export function installPercent(job: Pick<InstallJob, "done" | "total">): number {
  if (!job.total || job.total <= 0) return 0
  return Math.max(0, Math.min(100, (100 * job.done) / job.total))
}

export function etaText(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0) return ""
  const s = Math.round(seconds)
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`
  if (s >= 60) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`
  return `${s}s`
}

/* "13.5 GB / 32.9 GB · 41 MB/s · 12m 03s" while it downloads. */
export function progressLine(job: InstallJob): string {
  return [
    `${formatBytes(job.done)} / ${formatBytes(job.total)}`,
    job.rate && job.rate > 0 ? `${formatBytes(job.rate)}/s` : "",
    etaText(job.eta_seconds),
  ].filter(Boolean).join(" · ")
}

/* The rows to offer as "install": not here as a whole model, fits or not (a row that does not fit says why and has no button). */
export function installable(rows: CatalogRow[]): CatalogRow[] {
  return rows.filter((row) => !row.installed)
}

/* The add-ons that are here: they are not in the list of installed models (they are files inside one), so they are listed with a Remove. */
export function installedAddons(rows: CatalogRow[]): CatalogRow[] {
  return rows.filter((row) => !!row.into && !!row.installed)
}

/* Whether the page should keep asking the router for progress. */
export function anyInstalling(rows: CatalogRow[]): boolean {
  return rows.some((row) => row.job && (row.job.state === "starting" || row.job.state === "downloading" || row.job.state === "finishing"))
}

/* Whether a row may be installed from here: it fits (or only RAM is short, which the person may accept), and no install is running. */
export function canInstall(row: CatalogRow, busy: boolean): boolean {
  if (busy) return false
  return row.status === "good" || row.status === "tight" || row.status === "needs-ram"
}
