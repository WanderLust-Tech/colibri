/* Plain words for the errors a person can run into. The gateway and the engine speak in short technical sentences
   ("This model's maximum context length is 65536 tokens, however your messages resulted in at least 70123 tokens");
   this reads those sentences and says what happened and what to do. The original message always stays on screen
   beside the explanation, so nothing is hidden by a wrong guess. */

export interface ErrorExplanation {
  /* i18n key of the explanation, and the values its text uses. */
  key: string
  vars?: Record<string, string | number>
  /* A button the dashboard can offer for this error. */
  action?: "newChat"
}

const number = (value: string) => Number(value.replace(/[,_]/g, ""))

export function explainError(message: string): ErrorExplanation | null {
  const text = (message || "").trim()
  if (!text) return null

  const context = /maximum context length is\s+([\d,_]+)\s+tokens.*?(?:at least|resulted in)\s+([\d,_]+)\s+tokens/is.exec(text)
  if (context) return { key: "explain.context", vars: { limit: number(context[1]), used: number(context[2]) }, action: "newChat" }
  if (/context_length_exceeded|CONTEXT_EXCEEDED/.test(text)) return { key: "explain.contextPlain", action: "newChat" }

  /* The GPU path says this on the engine's own log; it reaches the page only if a server forwards it. */
  if (/device stopped answering/i.test(text)) return { key: "explain.gpu" }
  if (/out of memory|cannot allocate|not enough (free )?memory|bad_alloc/i.test(text)) return { key: "explain.memory" }

  if (/invalid or missing api key/i.test(text) || /^401\b/.test(text)) return { key: "explain.key" }
  if (/failed to fetch|networkerror|load failed|fetch failed|could not reach the server/i.test(text)) return { key: "explain.unreachable" }

  if (/inference queue is full|timed out waiting for the inference engine/i.test(text)) return { key: "explain.busy" }
  if (/scheduler is shutting down/i.test(text)) return { key: "explain.shutdown" }

  const missing = /the model `([^`]+)` does not exist/i.exec(text)
  if (missing) return { key: "explain.noModel", vars: { model: missing[1] } }
  if (/image generation model|does not generate images|is a decision model/i.test(text)) return { key: "explain.wrongKind" }

  if (/engine (is not running|failed|then reported|exited)|engine failed to process/i.test(text)) return { key: "explain.engine" }
  return null
}
