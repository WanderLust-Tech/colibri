import { describe, expect, it } from "vitest"

import { contextChoices, contextLabel } from "./contextWindow"

describe("contextChoices", () => {
  it("offers the default and the steps up to what the family supports", () => {
    expect(contextChoices(null, 32768)).toEqual([0, 4096, 8192, 16384, 32768])
    expect(contextChoices(null, 262144).at(-1)).toBe(262144)
  })
  it("keeps the model's own odd setting in the list, in order", () => {
    expect(contextChoices(20000, 65536)).toEqual([0, 4096, 8192, 16384, 20000, 32768, 65536])
  })
  it("does not double a setting that is a step", () => {
    expect(contextChoices(16384, 32768).filter((tokens) => tokens === 16384)).toHaveLength(1)
  })
})

describe("contextLabel", () => {
  it("says k for what is a multiple of half a k, and the number otherwise", () => {
    expect(contextLabel(32768)).toBe("32k")
    expect(contextLabel(1048576)).toBe("1024k")
    expect(contextLabel(1536)).toBe("1.5k")
    expect(contextLabel(20000)).toBe("20000")
  })
})
