import { isValidElement, type ReactElement, type ReactNode } from "react"
import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it, vi, type Mock } from "vitest"

import { ToolsAccessView } from "./ToolsAccess"
import { NO_TOOLS, type ToolsListing, type ToolsPolicy } from "@/lib/tools"
import de from "@/i18n/de"
import en from "@/i18n/en"
import id from "@/i18n/id"
import it_ from "@/i18n/it"
import zhCN from "@/i18n/zh-CN"
import zhTW from "@/i18n/zh-TW"

/* The real English strings, so what is asserted is what a person reads. */
vi.mock("@/i18n", async () => {
  const dict = (await import("@/i18n/en")).default as Record<string, string>
  return {
    useLocale: () => ({
      t: (key: string, vars?: Record<string, string | number>) =>
        (dict[key] ?? key).replace(/\{\{(\w+)\}\}/g, (_, name: string) => String(vars?.[name] ?? `{{${name}}}`)),
    }),
  }
})

const policy = (over: Partial<ToolsPolicy> = {}): ToolsPolicy => ({ folders: [], web: false, allowSecrets: false, approvalSeconds: 120, disabled: [], ...over })
const listing = (p: ToolsPolicy | null, extra: Partial<ToolsListing> = {}): ToolsListing =>
  ({ available: true, tools: [], policy: p, problem: null, path: "C:\\Users\\me\\tools-policy.json", ...extra })
const FOLDERS = [{ path: "C:\\work\\project", write: false }, { path: "C:\\work\\scratch", write: true }]

function props(over: Partial<Parameters<typeof ToolsAccessView>[0]> = {}) {
  return {
    listing: listing(policy({ folders: FOLDERS })), busy: false, error: "", folder: "", write: false,
    onFolder: vi.fn(), onWrite: vi.fn(), onAdd: vi.fn(), onEdit: vi.fn(), ...over,
  }
}
const html = (over: Parameters<typeof props>[0] = {}) => renderToStaticMarkup(<ToolsAccessView {...props(over)} />)

/* The element tree the view returns, searched without a DOM: the handlers of the controls can be called as a click would. */
type Found = ReactElement<Record<string, any>> // eslint-disable-line @typescript-eslint/no-explicit-any
function findAll(node: ReactNode, match: (element: Found) => boolean, found: Found[] = []): Found[] {
  if (Array.isArray(node)) { node.forEach((child) => findAll(child, match, found)); return found }
  if (!isValidElement(node)) return found
  const element = node as Found
  if (match(element)) found.push(element)
  findAll(element.props.children as ReactNode, match, found)
  return found
}
const byLabel = (tree: ReactNode, label: string) => {
  const hits = findAll(tree, (element) => element.props["aria-label"] === label)
  expect(hits, `a control labelled "${label}"`).toHaveLength(1)
  return hits[0].props
}
const view = (over: Parameters<typeof props>[0] = {}) => {
  const p = props(over)
  return { p, tree: ToolsAccessView(p) as ReactNode }
}

describe("the tools access section", () => {
  it("is not drawn for a router that has no tools endpoint", () => {
    expect(html({ listing: NO_TOOLS })).toBe("")
  })

  it("says that no folder is allowed, in the empty state, and still offers the add form and the web switch", () => {
    const out = html({ listing: listing(policy()) })
    expect(out).toContain("No folder is allowed, so the file and git tools are off.")
    expect(out).not.toContain("tools-folders")
    expect(out).toContain("Folder to allow")
    expect(out).toContain('placeholder="D:/code/project"')
    expect(out).toContain('role="switch"')
  })

  it("treats a listing without a policy like an empty one", () => {
    expect(html({ listing: listing(null) })).toContain("No folder is allowed")
  })

  it("lists every folder with its path and a badge for the kind of access", () => {
    const out = html()
    expect(out).toContain("C:\\work\\project")
    expect(out).toContain("C:\\work\\scratch")
    expect(out).not.toContain("No folder is allowed")
    expect(out).toContain('<span class="tools-badge">Read-only</span>')
    expect(out).toContain('<span class="tools-badge write">Read and write</span>')
  })

  it("puts the right action on each row: allow writing on a read-only folder, make read-only on a writable one", () => {
    const out = html()
    expect(out).toContain('aria-label="Allow writing in C:\\work\\project"')
    expect(out).toContain('aria-label="Make C:\\work\\scratch read-only"')
    expect(out).toContain('aria-label="Remove C:\\work\\project"')
    expect(out).toContain('aria-label="Remove C:\\work\\scratch"')
  })

  it("says that the folder must exist on the router's PC, because a browser cannot pick one", () => {
    expect(html()).toContain("A browser cannot pick a folder")
    expect(html()).toContain("must exist on the router&#x27;s PC")
  })

  it("shows the policy file and which tools are on", () => {
    const out = html({ listing: listing(policy(), { tools: [{ name: "files_read", description: "", group: "files", readOnly: true, enabled: true }] }) })
    expect(out).toContain("C:\\Users\\me\\tools-policy.json")
    expect(out).toContain("Tools on: files_read.")
    expect(html({ listing: listing(policy(), { path: null }) })).not.toContain("Policy file:")
  })

  it("has no confirm step: changing access is a plain click", () => {
    expect(html()).not.toMatch(/confirm/i)
  })

  it("reflects the web switch from the policy", () => {
    expect(html({ listing: listing(policy({ web: true })) })).toMatch(/role="switch"[^>]*checked=""|checked=""[^>]*role="switch"/)
    expect(html({ listing: listing(policy({ web: false })) })).not.toMatch(/role="switch"[^>]*checked=""|checked=""[^>]*role="switch"/)
  })
})

describe("what the controls do", () => {
  it("switches a read-only folder to write and a writable one back, by that folder's own path", () => {
    const { p, tree } = view()
    byLabel(tree, "Allow writing in C:\\work\\project").onClick()
    byLabel(tree, "Make C:\\work\\scratch read-only").onClick()
    expect((p.onEdit as Mock).mock.calls).toEqual([
      [{ action: "allow", folder: "C:\\work\\project", write: true }],
      [{ action: "allow", folder: "C:\\work\\scratch", write: false }],
    ])
  })

  it("removes the folder whose button was pressed, not another", () => {
    const { p, tree } = view()
    byLabel(tree, "Remove C:\\work\\scratch").onClick()
    expect((p.onEdit as Mock).mock.calls).toEqual([[{ action: "deny", folder: "C:\\work\\scratch" }]])
  })

  it("turns web fetch on and off with what the switch says", () => {
    const { p, tree } = view()
    const web = findAll(tree, (element) => element.props.role === "switch")[0].props
    web.onChange({ target: { checked: true } })
    web.onChange({ target: { checked: false } })
    expect((p.onEdit as Mock).mock.calls).toEqual([[{ action: "web", on: true }], [{ action: "web", on: false }]])
  })

  it("passes on the typed folder and the tick, and adds on submit without reloading the page", () => {
    const { p, tree } = view({ folder: "D:/code/x", write: true })
    byLabel(tree, "Folder to allow").onChange({ target: { value: "D:/code/y" } })
    const tick = findAll(tree, (element) => element.type === "input" && element.props.type === "checkbox" && element.props.role !== "switch")[0].props
    expect(tick.checked).toBe(true)
    tick.onChange({ target: { checked: false } })
    const form = findAll(tree, (element) => element.type === "form")[0].props
    const preventDefault = vi.fn()
    form.onSubmit({ preventDefault })
    expect(p.onFolder).toHaveBeenCalledWith("D:/code/y")
    expect(p.onWrite).toHaveBeenCalledWith(false)
    expect(p.onAdd).toHaveBeenCalledTimes(1)
    expect(preventDefault).toHaveBeenCalled()
  })

  it("does not offer Add for an empty box", () => {
    const add = (folder: string) => findAll(view({ folder }).tree, (element) => element.props.type === "submit")[0].props.disabled
    expect(add("")).toBe(true)
    expect(add("   ")).toBe(true)
    expect(add("D:/x")).toBe(false)
  })
})

describe("while a change is on its way", () => {
  const controls = (tree: ReactNode) => findAll(tree, (element) =>
    (typeof element.type === "string" && ["input"].includes(element.type)) || element.props.type === "button" || element.props.type === "submit")

  it("disables every control and says it is saving", () => {
    const { tree } = view({ busy: true, folder: "D:/x" })
    const all = controls(tree)
    expect(all.length).toBe(1 + 1 + 1 + 4 + 1)                 // path box, tick, Add, four row buttons, web switch
    expect(all.filter((element) => !element.props.disabled)).toEqual([])
    expect(html({ busy: true })).toContain("Saving…")
    expect(html({ busy: true })).toContain('aria-busy="true"')
  })

  it("enables them again when it is over", () => {
    const { tree } = view({ busy: false, folder: "D:/x" })
    expect(controls(tree).filter((element) => element.props.disabled)).toEqual([])
    expect(html({ busy: false })).not.toContain("Saving…")
  })
})

describe("a refused change", () => {
  it("shows the router's message inline, as an alert, as text", () => {
    const out = html({ error: "D:/nope is not a folder that exists. <b>x</b>" })
    expect(out).toContain('role="alert"')
    expect(out).toContain("D:/nope is not a folder that exists. &lt;b&gt;x&lt;/b&gt;")
    expect(out).not.toContain("<b>x</b>")
  })

  it("shows nothing when there was no refusal", () => {
    expect(html({ error: "" })).not.toContain("tool-problem")
  })
})

describe("a policy file that is not valid", () => {
  const broken = listing(null, { problem: "tools-policy.json: line 3: expected a comma" })

  it("says why every tool is off and that editing waits for the file to be fixed", () => {
    const out = html({ listing: broken })
    expect(out).toContain("tools-policy.json: line 3: expected a comma")
    expect(out).toContain("every tool is off")
    expect(out).toContain("Editing is off until the file is fixed")
  })

  it("disables the controls, which the router would refuse anyway", () => {
    const { tree } = view({ listing: broken, folder: "D:/x" })
    const all = findAll(tree, (element) => element.type === "input" || element.props.type === "submit")
    expect(all.length).toBeGreaterThanOrEqual(3)
    expect(all.filter((element) => !element.props.disabled)).toEqual([])
  })

  it("leaves the controls alone when the file is fine", () => {
    const { tree } = view({ listing: listing(policy({ folders: FOLDERS })) })
    expect(findAll(tree, (element) => element.props.type === "button").some((element) => element.props.disabled)).toBe(false)
  })
})

describe("the new strings", () => {
  const KEYS = Object.keys(en).filter((key) => key.startsWith("toolsAccess."))
  const DICTS: Record<string, Record<string, string>> = { de, id, it: it_, "zh-CN": zhCN, "zh-TW": zhTW }
  const holes = (text: string) => (text.match(/\{\{\w+\}\}/g) ?? []).sort()

  it("has the same keys, and the same placeholders, in all six languages", () => {
    expect(KEYS.length).toBeGreaterThanOrEqual(20)
    for (const [code, dict] of Object.entries(DICTS)) {
      expect(KEYS.filter((key) => !(key in dict)), code).toEqual([])
      expect(Object.keys(dict).filter((key) => key.startsWith("toolsAccess.") && !(key in en)), code).toEqual([])
      for (const key of KEYS) {
        expect(dict[key].trim(), `${code} ${key}`).not.toBe("")
        expect(holes(dict[key]), `${code} ${key}`).toEqual(holes((en as Record<string, string>)[key]))
      }
    }
  })

  it("leaves no string behind that nothing uses any more", () => {
    for (const dict of [en, ...Object.values(DICTS)] as Record<string, string>[]) {
      for (const gone of ["policyTitle", "policyRead", "policyWrite", "policyNoFolders", "policyWebOn", "policyWebOff", "policyHow"]) {
        expect(dict[`tools.${gone}`]).toBeUndefined()
      }
    }
  })
})
