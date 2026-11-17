import { describe, expect, it } from "vitest"
import { detailsLine } from "./modelDetails"

const tokens = (n: string) => `up to ${n} tokens`
const from = (repo: string) => `from ${repo}`

describe("detailsLine", () => {
  it("puts what is known on one line", () => {
    expect(detailsLine({ architecture: "Qwen3MoeForCausalLM", quantisation: "u8 packed", context_length: 262144, source: "acme/model" }, tokens, from, "en-US"))
      .toBe("Qwen3MoeForCausalLM · u8 packed · up to 262,144 tokens · from acme/model")
  })

  it("leaves out what is not known, and is empty with nothing", () => {
    expect(detailsLine({ architecture: "QwenImage21Pipeline" }, tokens, from)).toBe("QwenImage21Pipeline")
    expect(detailsLine({ quantisation: "bf16", shards: 4 }, tokens, from)).toBe("bf16")
    expect(detailsLine({}, tokens, from)).toBe("")
    expect(detailsLine(undefined, tokens, from)).toBe("")
  })
})
