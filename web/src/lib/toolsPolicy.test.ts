import { afterEach, describe, expect, it, vi } from "vitest"

import { editToolsPolicy, type ToolsPolicyEdited } from "./api"
import { NO_TOOLS, parseToolsListing, type ToolsListing, type ToolsPolicy, type ToolsPolicyEdit } from "./tools"
import {
  POLICY_UI_START, addEdit, cleanFolder, createPolicyController, removeEdit, runPolicyEdit, switchEdit, webEdit, type PolicyUiState,
} from "./toolsPolicy"

afterEach(() => vi.unstubAllGlobals())

const policy = (over: Partial<ToolsPolicy> = {}): ToolsPolicy =>
  ({ folders: [], web: false, allowSecrets: false, approvalSeconds: 120, disabled: [], ...over })
const listingOf = (p: ToolsPolicy, extra: Partial<ToolsListing> = {}): ToolsListing =>
  ({ available: true, tools: [], policy: p, problem: null, path: "C:\\Users\\me\\tools-policy.json", ...extra })

describe("editToolsPolicy", () => {
  const answer = { status: "ok", action: "allow", folder: "D:\\code\\x", policy: { folders: [{ path: "D:\\code\\x", write: true }], web: false, allowSecrets: false, approvalSeconds: 120, disabled: [] } }

  it("posts the change as JSON with the bearer key, to /v1/tools/policy, and returns the folder and the policy", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(answer)))
    vi.stubGlobal("fetch", fetchMock)
    const edit: ToolsPolicyEdit = { action: "allow", folder: "D:/code/x", write: true }
    const done = await editToolsPolicy("http://localhost:8000/v1/", "secret", edit)
    expect(done.folder).toBe("D:\\code\\x")
    expect(done.policy.folders).toEqual([{ path: "D:\\code\\x", write: true }])
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("http://localhost:8000/v1/tools/policy")
    expect(init.method).toBe("POST")
    expect(init.headers).toMatchObject({ Authorization: "Bearer secret", "Content-Type": "application/json" })
    expect(JSON.parse(init.body)).toEqual({ action: "allow", folder: "D:/code/x", write: true })
  })

  it("sends a deny and a web change as they are, and reads a null folder", async () => {
    const fetchMock = vi.fn().mockImplementation(async () => new Response(JSON.stringify({ ...answer, folder: null })))
    vi.stubGlobal("fetch", fetchMock)
    expect((await editToolsPolicy("http://h/v1", "", { action: "web", on: true })).folder).toBeNull()
    await editToolsPolicy("http://h/v1", "", { action: "deny", folder: "D:/code/x" })
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ action: "web", on: true })
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ action: "deny", folder: "D:/code/x" })
    expect(fetchMock.mock.calls[0][1].headers).not.toHaveProperty("Authorization")
  })

  it.each([
    [400, "That folder does not exist on this PC."],
    [404, "That folder is not in the list."],
    [401, "Invalid or missing API key."],
    [403, "Not allowed from this address."],
  ])("throws the router's own message for a %i", async (status, message) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { message, code: "x", param: null } }), { status })))
    await expect(editToolsPolicy("http://h/v1", "k", { action: "deny", folder: "D:/x" })).rejects.toThrow(message)
  })

  it("falls back to the status line when the refusal has no message", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("not json", { status: 400, statusText: "Bad Request" })))
    await expect(editToolsPolicy("http://h/v1", "k", { action: "web", on: true })).rejects.toThrow("400 Bad Request")
  })

  it("does not take an answer without a policy for a success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: "ok" }))))
    await expect(editToolsPolicy("http://h/v1", "k", { action: "web", on: true })).rejects.toThrow(/policy/)
  })
})

describe("what each control asks for", () => {
  it("cleans a typed path: spaces and the quotes of Explorer's Copy as path", () => {
    expect(cleanFolder("  D:/code/project  ")).toBe("D:/code/project")
    expect(cleanFolder('"D:\\My Projects\\a"')).toBe("D:\\My Projects\\a")
    expect(cleanFolder(' "D:\\a" ')).toBe("D:\\a")
    expect(cleanFolder("   ")).toBe("")
  })

  it("adds with the tick as it is, and nothing for an empty box", () => {
    expect(addEdit("D:/x", false)).toEqual({ action: "allow", folder: "D:/x", write: false })
    expect(addEdit("D:/x", true)).toEqual({ action: "allow", folder: "D:/x", write: true })
    expect(addEdit("  ", true)).toBeNull()
  })

  it("switches a folder to the other kind of access and removes the one named", () => {
    expect(switchEdit({ path: "D:\\a", write: false })).toEqual({ action: "allow", folder: "D:\\a", write: true })
    expect(switchEdit({ path: "D:\\a", write: true })).toEqual({ action: "allow", folder: "D:\\a", write: false })
    expect(removeEdit("D:\\b")).toEqual({ action: "deny", folder: "D:\\b" })
    expect(webEdit(true)).toEqual({ action: "web", on: true })
    expect(webEdit(false)).toEqual({ action: "web", on: false })
  })
})

describe("runPolicyEdit", () => {
  const before = listingOf(policy())
  const after = policy({ folders: [{ path: "D:\\x", write: false }] })
  const edit: ToolsPolicyEdit = { action: "allow", folder: "D:/x", write: false }

  it("shows the policy the router answered with, and the tool list read again", async () => {
    const fresh = listingOf(after, { tools: [{ name: "files_read", description: "", group: "files", readOnly: true, enabled: true }] })
    const send = vi.fn().mockResolvedValue({ folder: "D:\\x", policy: after })
    const refresh = vi.fn().mockResolvedValue(fresh)
    const outcome = await runPolicyEdit(before, edit, send, refresh)
    expect(send).toHaveBeenCalledWith(edit)
    expect(refresh).toHaveBeenCalledTimes(1)
    expect(outcome).toEqual({ ok: true, listing: fresh, error: "" })
  })

  it("keeps the change when the second read fails or finds no tools endpoint", async () => {
    const send = vi.fn().mockResolvedValue({ folder: "D:\\x", policy: after })
    const failed = await runPolicyEdit(listingOf(policy(), { problem: "old problem" }), edit, send, () => Promise.reject(new Error("down")))
    expect(failed.ok).toBe(true)
    expect(failed.listing.policy).toEqual(after)
    expect(failed.listing.problem).toBeNull()
    expect(failed.listing.path).toBe(before.path)
    const gone = await runPolicyEdit(before, edit, send, () => Promise.resolve(NO_TOOLS))
    expect(gone.listing.policy).toEqual(after)
    expect(gone.listing.available).toBe(true)
  })

  it("leaves the listing as it was, with the router's message, when the change is refused", async () => {
    const refresh = vi.fn()
    const outcome = await runPolicyEdit(before, edit, () => Promise.reject(new Error("That folder does not exist.")), refresh)
    expect(outcome).toEqual({ ok: false, listing: before, error: "That folder does not exist." })
    expect(refresh).not.toHaveBeenCalled()
  })

  it("reports a failure that is not an Error too", async () => {
    expect((await runPolicyEdit(before, edit, () => Promise.reject("boom"), vi.fn())).error).toBe("boom")
  })
})

describe("the controller behind the controls", () => {
  const empty = listingOf(policy())
  const withX = policy({ folders: [{ path: "D:\\x", write: false }] })

  function setup(send: (edit: ToolsPolicyEdit) => Promise<ToolsPolicyEdited>, refresh: () => Promise<ToolsListing> = () => Promise.resolve(listingOf(withX))) {
    const states: PolicyUiState[] = []
    const listings: ToolsListing[] = []
    const sent: ToolsPolicyEdit[] = []
    const controller = createPolicyController({
      getListing: () => empty,
      send: (edit) => { sent.push(edit); return send(edit) },
      refresh,
      onListing: (listing) => listings.push(listing),
      onState: (state) => states.push(state),
    })
    return { controller, states, listings, sent }
  }
  const took = () => Promise.resolve({ folder: "D:\\x", policy: withX })

  it("is busy while the router thinks, then idle with the new listing handed on", async () => {
    let finish!: (value: ToolsPolicyEdited) => void
    const { controller, states, listings } = setup(() => new Promise((resolve) => { finish = resolve }))
    const done = controller.edit({ action: "web", on: true })
    expect(controller.state().busy).toBe(true)
    expect(listings).toEqual([])
    finish({ folder: null, policy: withX })
    expect(await done).toBe(true)
    expect(controller.state().busy).toBe(false)
    expect(states.map((state) => state.busy)).toEqual([true, false])
    expect(listings).toEqual([listingOf(withX)])
  })

  it("sends nothing for a second change while one is on its way, and takes the next one after", async () => {
    let finish!: (value: ToolsPolicyEdited) => void
    const { controller, sent } = setup(() => new Promise((resolve) => { finish = resolve }))
    const first = controller.edit({ action: "web", on: true })
    expect(await controller.edit({ action: "deny", folder: "D:\\x" })).toBe(false)
    expect(sent).toHaveLength(1)
    finish({ folder: null, policy: withX })
    await first
    const second = controller.edit({ action: "web", on: false })
    expect(sent).toHaveLength(2)
    finish({ folder: null, policy: withX })
    await second
  })

  it("shows the router's refusal, keeps the typed path, and clears the message at the next try", async () => {
    let refuse = true
    const { controller, listings } = setup(() => refuse ? Promise.reject(new Error("D:/nope is not a folder that exists.")) : took())
    controller.setFolder("D:/nope")
    controller.setWrite(true)
    await controller.add()
    expect(controller.state()).toEqual({ busy: false, error: "D:/nope is not a folder that exists.", folder: "D:/nope", write: true })
    expect(listings).toEqual([])
    refuse = false
    await controller.add()
    expect(controller.state()).toEqual(POLICY_UI_START)
    expect(listings).toHaveLength(1)
  })

  it("adds the cleaned path with the tick, then empties the form", async () => {
    const { controller, sent } = setup(took)
    controller.setFolder('  "D:\\code\\x" ')
    controller.setWrite(true)
    await controller.add()
    expect(sent).toEqual([{ action: "allow", folder: "D:\\code\\x", write: true }])
    expect(controller.state()).toMatchObject({ folder: "", write: false })
  })

  it("does not ask the router to add an empty box", async () => {
    const { controller, sent } = setup(took)
    controller.setFolder("   ")
    await controller.add()
    expect(sent).toEqual([])
  })

  it("can be used again after a refusal", async () => {
    const { controller, sent } = setup(() => Promise.reject(new Error("no")))
    await controller.edit(removeEdit("D:\\x"))
    await controller.edit(removeEdit("D:\\x"))
    expect(sent).toHaveLength(2)
    expect(controller.state().busy).toBe(false)
  })
})

describe("the listing carries the policy file", () => {
  it("reads the path of the file and the problem", () => {
    const parsed = parseToolsListing({ path: "C:\\p.json", problem: "bad json", policy: policy(), data: [] })
    expect(parsed.path).toBe("C:\\p.json")
    expect(parsed.problem).toBe("bad json")
    expect(parseToolsListing({ data: [] }).path).toBeNull()
  })
})
