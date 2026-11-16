import { describe, expect, it } from "vitest"

import { KEEP_LINES, lostLines, mergeLines, sourcesOf, type LogLine } from "./logTail"

const line = (seq: number, source = "router", text = `line ${seq}`): LogLine => ({ seq, time: 1000 + seq, source, text })

describe("mergeLines", () => {
  it("adds new lines after the old ones", () => {
    expect(mergeLines([line(1), line(2)], [line(3)]).map((l) => l.seq)).toEqual([1, 2, 3])
  })

  it("does not add a line twice when two reads overlap", () => {
    expect(mergeLines([line(1), line(2)], [line(2), line(3)]).map((l) => l.seq)).toEqual([1, 2, 3])
  })

  it("returns the same list when nothing is new", () => {
    const held = [line(1)]
    expect(mergeLines(held, [])).toBe(held)
    expect(mergeLines(held, [line(1)])).toBe(held)
  })

  it("keeps only the newest lines", () => {
    const held = Array.from({ length: 5 }, (_, i) => line(i + 1))
    expect(mergeLines(held, [line(6), line(7)], 4).map((l) => l.seq)).toEqual([4, 5, 6, 7])
    expect(KEEP_LINES).toBeGreaterThan(100)
  })
})

describe("lostLines", () => {
  it("is true when the server's oldest line is later than the one after the last seen", () => {
    expect(lostLines({ first: 50 }, 10)).toBe(true)
  })

  it("is false on the first read, and when the next line is still held", () => {
    expect(lostLines({ first: 50 }, 0)).toBe(false)
    expect(lostLines({ first: 11 }, 10)).toBe(false)
    expect(lostLines({ first: 1 }, 10)).toBe(false)
  })
})

describe("sourcesOf", () => {
  it("lists the router first and then the models in the order they appeared", () => {
    expect(sourcesOf([line(1, "gemma"), line(2, "router"), line(3, "qwen"), line(4, "gemma")])).toEqual(["router", "gemma", "qwen"])
  })
})
