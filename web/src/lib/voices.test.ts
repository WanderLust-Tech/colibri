import { describe, expect, it } from "vitest"

import { fileToBase64, voiceNameProblem } from "./voices"

describe("saved voice names", () => {
  it("accepts what the router accepts, case-insensitively", () => {
    for (const name of ["ana", "Ana", "a-b_c", "9lives", "x".repeat(64)]) expect(voiceNameProblem(name)).toBeNull()
  })

  it("says empty for nothing and invalid for the rest", () => {
    expect(voiceNameProblem("")).toBe("empty")
    expect(voiceNameProblem("   ")).toBe("empty")
    for (const name of ["-a", "a b", "a/b", "ünï", "x".repeat(65)]) expect(voiceNameProblem(name)).toBe("invalid")
  })
})

describe("fileToBase64", () => {
  it("gives the bytes' base64 without the data: prefix", async () => {
    expect(await fileToBase64(new Blob([new Uint8Array([72, 105, 33])]))).toBe("SGkh")
  })
})
