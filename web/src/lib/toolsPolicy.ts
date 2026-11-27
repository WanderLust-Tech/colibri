import type { ToolsPolicyEdited } from "./api"
import { NO_TOOLS, type ToolsListing, type ToolsPolicyEdit } from "./tools"

/* Editing the router's tools policy from the dashboard (docs/builtin-tools.md). The folder has to exist on the router's PC and a
   browser cannot pick a path, so it is typed. All of the flow is here, without React, so it can be tested: what a typed path
   becomes, what each button asks the router, and what the page shows afterwards. */

/* A path as typed or pasted: spaces around it dropped, and the quotation marks that Explorer's "Copy as path" adds. The router
   normalises the rest (slashes, `..`, drive letters). */
export function cleanFolder(raw: string): string {
  const trimmed = raw.trim()
  const quoted = /^"(.*)"$/.exec(trimmed)
  return (quoted ? quoted[1] : trimmed).trim()
}

export const addEdit = (raw: string, write: boolean): ToolsPolicyEdit | null => {
  const folder = cleanFolder(raw)
  return folder ? { action: "allow", folder, write } : null
}

/* The same folder with the other kind of access: the router replaces an entry that has the same path. */
export const switchEdit = (folder: { path: string; write: boolean }): ToolsPolicyEdit => ({ action: "allow", folder: folder.path, write: !folder.write })
export const removeEdit = (path: string): ToolsPolicyEdit => ({ action: "deny", folder: path })
export const webEdit = (on: boolean): ToolsPolicyEdit => ({ action: "web", on })

export interface PolicyEditOutcome {
  ok: boolean
  /* What the page should show now: the new one after a change, the old one after a refusal. */
  listing: ToolsListing
  /* The router's message when it refused, or the reason it could not be asked. */
  error: string
}

/* Asks the router for one change. After a change the policy shown is the one the router answered with, and the tool list is read
   again, so what the chat offers follows at once; if that second read fails the change still stands. */
export async function runPolicyEdit(
  listing: ToolsListing,
  edit: ToolsPolicyEdit,
  send: (edit: ToolsPolicyEdit) => Promise<ToolsPolicyEdited>,
  refresh: () => Promise<ToolsListing>,
): Promise<PolicyEditOutcome> {
  let edited: ToolsPolicyEdited
  try {
    edited = await send(edit)
  } catch (cause) {
    return { ok: false, listing, error: cause instanceof Error ? cause.message : String(cause) }
  }
  const updated: ToolsListing = { ...(listing.available ? listing : NO_TOOLS), available: true, policy: edited.policy, problem: null }
  try {
    const fresh = await refresh()
    if (fresh.available) return { ok: true, listing: fresh, error: "" }
  } catch { /* the change was made; the list of tools catches up the next time the page asks */ }
  return { ok: true, listing: updated, error: "" }
}

/* ---- the state behind the controls */

export interface PolicyUiState {
  /* A change is on its way to the router: nothing can be pressed. */
  busy: boolean
  /* The router's message for the change it refused. */
  error: string
  /* The folder box and the "Allow writing" tick of the add form. */
  folder: string
  write: boolean
}

export const POLICY_UI_START: PolicyUiState = { busy: false, error: "", folder: "", write: false }

export interface PolicyControllerDeps {
  /* Read at the moment of a change: the page's copy of the listing, which a refresh elsewhere may have replaced. */
  getListing: () => ToolsListing
  send: (edit: ToolsPolicyEdit) => Promise<ToolsPolicyEdited>
  refresh: () => Promise<ToolsListing>
  /* The new listing after a change, for the rest of the page (the chat's Tools switch reads the same one). */
  onListing: (listing: ToolsListing) => void
  onState: (state: PolicyUiState) => void
}

/* One change at a time, and what the controls show before, during and after it. The component only holds the state it is told. */
export function createPolicyController(deps: PolicyControllerDeps) {
  let state = POLICY_UI_START
  let inFlight = false
  const set = (patch: Partial<PolicyUiState>) => { state = { ...state, ...patch }; deps.onState(state) }

  /* Resolves true when the router made the change. A second change while one is on its way is not sent. */
  async function edit(change: ToolsPolicyEdit): Promise<boolean> {
    if (inFlight) return false
    inFlight = true
    set({ busy: true, error: "" })
    try {
      const outcome = await runPolicyEdit(deps.getListing(), change, deps.send, deps.refresh)
      if (outcome.ok) deps.onListing(outcome.listing)
      set({ busy: false, error: outcome.error })
      return outcome.ok
    } finally {
      inFlight = false
      if (state.busy) set({ busy: false })
    }
  }

  return {
    state: () => state,
    edit,
    setFolder: (folder: string) => set({ folder }),
    setWrite: (write: boolean) => set({ write }),
    /* The box is emptied only when the router took the folder: a refused path stays there to be corrected. */
    async add(): Promise<void> {
      const change = addEdit(state.folder, state.write)
      if (change && await edit(change)) set({ folder: "", write: false })
    },
  }
}
