import { useRef, useState } from "react"
import { Wrench } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useLocale } from "@/i18n"
import { editToolsPolicy, listTools } from "@/lib/api"
import { enabledTools, type ToolsListing, type ToolsPolicyEdit } from "@/lib/tools"
import { POLICY_UI_START, createPolicyController, removeEdit, switchEdit, webEdit } from "@/lib/toolsPolicy"

interface ViewProps {
  listing: ToolsListing
  /* A change is on its way to the router: nothing can be pressed. */
  busy: boolean
  /* The router's message for the change it refused. */
  error: string
  folder: string
  write: boolean
  onFolder: (value: string) => void
  onWrite: (value: boolean) => void
  onAdd: () => void
  onEdit: (edit: ToolsPolicyEdit) => void
}

/* What the router's tools may use, with the controls to change it. All state lives in ToolsAccess below; this only draws. */
export function ToolsAccessView({ listing, busy, error, folder, write, onFolder, onWrite, onAdd, onEdit }: ViewProps) {
  const { t } = useLocale()
  if (!listing.available) return null
  const on = enabledTools(listing)
  const folders = listing.policy?.folders ?? []
  /* A policy file that is not valid turns everything off, and the router refuses to edit it. Say so instead of offering buttons
     that can only fail. */
  const locked = busy || !!listing.problem
  return (
    <section className="side-section tools-policy tools-access" aria-busy={busy}>
      <div className="section-title"><Wrench className="size-3.5" /> {t("toolsAccess.title")}</div>
      <p className="field-help">{t("toolsAccess.intro")}</p>
      {listing.problem && (
        <p className="field-help model-error" role="alert">{t("tools.policyProblem", { message: listing.problem })} {t("toolsAccess.problemHint")}</p>
      )}
      <p className="field-help">{on.length ? t("tools.policyOn", { tools: on.map((tool) => tool.name).join(", ") }) : t("tools.policyOff")}</p>

      {folders.length === 0 ? (
        <p className="field-help tools-empty">{t("toolsAccess.empty")}</p>
      ) : (
        <ul className="model-list tools-folders">
          {folders.map((entry) => (
            <li key={entry.path} className="tools-folder">
              <code className="tools-path">{entry.path}</code>
              <span className={entry.write ? "tools-badge write" : "tools-badge"}>{t(entry.write ? "toolsAccess.readWrite" : "toolsAccess.readOnly")}</span>
              <span className="tools-actions">
                <Button type="button" size="sm" variant="secondary" disabled={locked}
                  aria-label={t(entry.write ? "toolsAccess.makeReadOnlyFor" : "toolsAccess.allowWritingFor", { path: entry.path })}
                  onClick={() => onEdit(switchEdit(entry))}>{t(entry.write ? "toolsAccess.makeReadOnly" : "toolsAccess.allowWriting")}</Button>
                <Button type="button" size="sm" variant="secondary" disabled={locked}
                  aria-label={t("toolsAccess.removeFor", { path: entry.path })}
                  onClick={() => onEdit(removeEdit(entry.path))}>{t("toolsAccess.remove")}</Button>
              </span>
            </li>
          ))}
        </ul>
      )}

      <form className="tools-add" onSubmit={(event) => { event.preventDefault(); onAdd() }}>
        <input type="text" className="tools-input" value={folder} disabled={locked} spellCheck={false} autoComplete="off"
          aria-label={t("toolsAccess.folderLabel")} placeholder={t("toolsAccess.folderPlaceholder")}
          onChange={(event) => onFolder(event.target.value)} />
        <label className="tools-write"><input type="checkbox" checked={write} disabled={locked} onChange={(event) => onWrite(event.target.checked)} /> {t("toolsAccess.allowWriting")}</label>
        <Button type="submit" size="sm" disabled={locked || !folder.trim()}>{t("toolsAccess.add")}</Button>
      </form>
      <p className="field-help">{t("toolsAccess.folderHint")}</p>
      {error && <p className="tool-problem" role="alert">{error}</p>}

      <label className="tools-web">
        <input type="checkbox" role="switch" checked={listing.policy?.web === true} disabled={locked} onChange={(event) => onEdit(webEdit(event.target.checked))} />
        <span>{t("toolsAccess.web")}</span>
      </label>
      <p className="field-help">{t("toolsAccess.webHelp")}</p>
      {busy && <p className="field-help" role="status">{t("toolsAccess.saving")}</p>}
      {listing.path && <p className="field-help tools-file">{t("toolsAccess.file")} <code>{listing.path}</code></p>}
    </section>
  )
}

interface Props {
  baseUrl: string
  apiKey: string
  listing: ToolsListing
  /* The page's copy of the tools listing: the chat's Tools switch reads it too, so a change shows there at once. */
  onListing: (listing: ToolsListing) => void
}

export function ToolsAccess(props: Props) {
  const [state, setState] = useState(POLICY_UI_START)
  /* The controller is made once; what it asks for is read from here when a change is made. */
  const latest = useRef(props)
  latest.current = props
  const controller = useRef<ReturnType<typeof createPolicyController>>(undefined)
  if (!controller.current) {
    controller.current = createPolicyController({
      getListing: () => latest.current.listing,
      send: (edit) => editToolsPolicy(latest.current.baseUrl, latest.current.apiKey, edit),
      refresh: () => listTools(latest.current.baseUrl, latest.current.apiKey),
      onListing: (listing) => latest.current.onListing(listing),
      onState: setState,
    })
  }
  const ctl = controller.current
  return <ToolsAccessView listing={props.listing} {...state}
    onFolder={ctl.setFolder} onWrite={ctl.setWrite} onAdd={() => { void ctl.add() }} onEdit={(change) => { void ctl.edit(change) }} />
}
