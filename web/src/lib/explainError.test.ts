import { describe, expect, it } from "vitest"

import en from "../i18n/en"
import { explainError } from "./explainError"

describe("explainError", () => {
  it("reads the window and the prompt out of a context error, and offers a new chat", () => {
    const message = "This model's maximum context length is 65536 tokens, however your messages resulted in at least 70123 tokens. Please shorten the conversation, or restart the server with a larger CTX."
    expect(explainError(message)).toEqual({ key: "explain.context", vars: { limit: "65,536", used: "70,123" }, action: "newChat" })
  })

  it("still recognises a context error whose figures are missing", () => {
    expect(explainError("code: context_length_exceeded")).toEqual({ key: "explain.contextPlain", action: "newChat" })
  })

  it("names an unreachable server, a bad key and a busy queue", () => {
    expect(explainError("Failed to fetch")?.key).toBe("explain.unreachable")
    expect(explainError("status.serverError")).toBeNull()
    expect(explainError("Could not reach the server.")?.key).toBe("explain.unreachable")
    expect(explainError("Invalid or missing API key.")?.key).toBe("explain.key")
    expect(explainError("401 Unauthorized")?.key).toBe("explain.key")
    expect(explainError("The inference queue is full.")?.key).toBe("explain.busy")
    expect(explainError("Timed out waiting for the inference engine.")?.key).toBe("explain.busy")
  })

  it("explains a lost GPU and a lack of memory", () => {
    expect(explainError("[VK] tier qwen36: the device stopped answering, the experts stay on the CPU")?.key).toBe("explain.gpu")
    expect(explainError("std::bad_alloc")?.key).toBe("explain.memory")
    expect(explainError("The colibri engine failed to process the request.")?.key).toBe("explain.engine")
  })

  it("says which model is missing, and when a model cannot chat", () => {
    expect(explainError("The model `nope` does not exist.")).toEqual({ key: "explain.noModel", vars: { model: "nope" } })
    expect(explainError("`flux` is an image generation model and does not chat.")?.key).toBe("explain.wrongKind")
  })

  it("has nothing to say about an error it does not know, or an empty one", () => {
    expect(explainError("`messages` must be a non-empty array.")).toBeNull()
    expect(explainError("")).toBeNull()
  })

  it("only names dictionary keys that exist", () => {
    const samples = [
      "maximum context length is 10 tokens, however your messages resulted in at least 20 tokens", "context_length_exceeded",
      "device stopped answering", "out of memory", "Invalid or missing API key.", "Failed to fetch", "The inference queue is full.",
      "The inference scheduler is shutting down.", "The model `x` does not exist.", "is an image generation model", "The image engine failed: x",
    ]
    for (const sample of samples) {
      const found = explainError(sample)
      expect(found, sample).not.toBeNull()
      expect(en[found!.key], found!.key).toBeTruthy()
    }
  })
})
