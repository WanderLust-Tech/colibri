/* What an installed model folder says about itself (c/model_info.py, the `details` of GET /v1/models/installed): read from its
   small files only, so it is there before the model has ever been loaded. */
export interface ModelDetails {
  architecture?: string
  quantisation?: string
  /* The model's own limit, in tokens; the server may serve less (the card's `context_length` is what it serves). */
  context_length?: number
  shards?: number
  bytes?: number
  source?: string
  problems?: string[]
}

/* One line for the model row: "Qwen3MoeForCausalLM · u8 packed · up to 262,144 tokens · from Justvugg/Qwen3-Coder...". */
export function detailsLine(details: ModelDetails | undefined, tokens: (count: string) => string, from: (repo: string) => string, locale?: string): string {
  if (!details) return ""
  return [
    details.architecture,
    details.quantisation,
    details.context_length ? tokens(details.context_length.toLocaleString(locale)) : "",
    details.source ? from(details.source) : "",
  ].filter(Boolean).join(" · ")
}
