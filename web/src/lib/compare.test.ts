import { describe, expect, it } from "vitest"

import { isChatModel, loadScores, ranked, recordVote, runSequentially, saveScores, shuffle, winPercent, type Scores } from "./compare"

describe("isChatModel", () => {
  it("takes a language model, and a server that says nothing, as a chat model", () => {
    expect(isChatModel({ id: "a", capabilities: ["chat", "systemone"] })).toBe(true)
    expect(isChatModel({ id: "b" })).toBe(true)
    expect(isChatModel({ id: "c", capabilities: [] })).toBe(true)
  })

  it("leaves out image, music and decision models", () => {
    expect(isChatModel({ id: "i", capabilities: ["image_generation"] })).toBe(false)
    expect(isChatModel({ id: "m", capabilities: ["music_generation"] })).toBe(false)
    expect(isChatModel({ id: "d", capabilities: ["systemone", "decision"] })).toBe(false)
  })
})

describe("shuffle", () => {
  it("keeps every item and does not touch the original", () => {
    const items = ["a", "b", "c", "d"]
    const out = shuffle(items, () => 0.3)
    expect([...out].sort()).toEqual(items)
    expect(items).toEqual(["a", "b", "c", "d"])
  })

  it("can be fixed for a test, and moves things when the dice say so", () => {
    expect(shuffle(["a", "b", "c"], () => 0)).toEqual(["b", "c", "a"])
    expect(shuffle([], () => 0.5)).toEqual([])
  })
})

describe("the scoreboard", () => {
  it("counts a round for every model that played and a win for the one chosen", () => {
    const scores = recordVote({}, ["coder", "gemma"], "gemma")
    expect(scores).toEqual({ coder: { wins: 0, rounds: 1 }, gemma: { wins: 1, rounds: 1 } })
    expect(recordVote(scores, ["coder", "gemma", "e2b"], "coder")).toEqual({
      coder: { wins: 1, rounds: 2 }, gemma: { wins: 1, rounds: 2 }, e2b: { wins: 0, rounds: 1 },
    })
  })

  it("ignores a winner who did not play, and leaves the old scores alone", () => {
    const before: Scores = { a: { wins: 1, rounds: 1 } }
    expect(recordVote(before, ["a", "b"], "zzz")).toBe(before)
    recordVote(before, ["a", "b"], "b")
    expect(before).toEqual({ a: { wins: 1, rounds: 1 } })
  })

  it("ranks by share of rounds won, then by rounds played", () => {
    expect(winPercent({ wins: 1, rounds: 3 })).toBe(33)
    expect(winPercent({ wins: 0, rounds: 0 })).toBe(0)
    const order = ranked({ x: { wins: 1, rounds: 2 }, y: { wins: 3, rounds: 4 }, z: { wins: 2, rounds: 4 } }).map((row) => row.model)
    expect(order).toEqual(["y", "z", "x"]) // z and x tie at 50%; z has played more
    expect(ranked({ a: { wins: 1, rounds: 1 }, b: { wins: 3, rounds: 3 } }).map((row) => row.model)).toEqual(["b", "a"])
  })
})

describe("saving the scores", () => {
  const memory = () => {
    const data = new Map<string, string>()
    return { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => void data.set(k, v), removeItem: (k: string) => void data.delete(k) }
  }

  it("round-trips, and an empty board is removed rather than saved", () => {
    const store = memory()
    const scores: Scores = { a: { wins: 2, rounds: 5 } }
    saveScores(store, scores)
    expect(loadScores(store)).toEqual(scores)
    saveScores(store, {})
    expect(store.getItem("colibri.compareScores")).toBeNull()
  })

  it("drops anything damaged instead of breaking the page", () => {
    const store = memory()
    store.setItem("colibri.compareScores", "not json")
    expect(loadScores(store)).toEqual({})
    store.setItem("colibri.compareScores", JSON.stringify({ ok: { wins: 1, rounds: 2 }, odd: { wins: 5, rounds: 2 }, bad: "x", neg: { wins: -1, rounds: 3 } }))
    expect(loadScores(store)).toEqual({ ok: { wins: 1, rounds: 2 } })
    store.setItem("colibri.compareScores", "[1,2]")
    expect(loadScores(store)).toEqual({})
  })

  it("survives storage that throws", () => {
    const broken = { getItem: () => { throw new Error("denied") }, setItem: () => { throw new Error("denied") }, removeItem: () => { throw new Error("denied") } }
    expect(loadScores(broken)).toEqual({})
    expect(() => saveScores(broken, { a: { wins: 1, rounds: 1 } })).not.toThrow()
  })
})

describe("runSequentially", () => {
  it("runs the items in order, one at a time", async () => {
    const log: string[] = []
    await runSequentially(["a", "b", "c"], async (item) => {
      log.push(`start ${item}`)
      await new Promise((resolve) => setTimeout(resolve, 5))
      log.push(`end ${item}`)
    }, new AbortController().signal)
    expect(log).toEqual(["start a", "end a", "start b", "end b", "start c", "end c"])
  })

  it("goes on after one fails", async () => {
    const done: string[] = []
    await runSequentially(["a", "b", "c"], async (item) => {
      if (item === "b") throw new Error("boom")
      done.push(item)
    }, new AbortController().signal)
    expect(done).toEqual(["a", "c"])
  })

  it("does not start the next item once stopped", async () => {
    const controller = new AbortController()
    const done: string[] = []
    await runSequentially(["a", "b", "c"], async (item) => {
      done.push(item)
      if (item === "a") controller.abort()
    }, controller.signal)
    expect(done).toEqual(["a"])
  })
})
