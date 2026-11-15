import type { TokenUsage } from "./api"

/* What one assistant reply cost and where it ran. Built when the reply finishes and kept on the message. */
export interface ReplyStats {
  /* The model that answered, as the stream names it. */
  model?: string
  /* The name of the preset the dashboard asked for, when it asked for one: the answer then came from a different
     model than the one named in the picker, and this says why. */
  preset?: string
  promptTokens?: number
  /* Prompt tokens the engine did not read again. Absent when the server does not say; 0 means it said none. */
  cachedTokens?: number
  completionTokens?: number
  tokPerSec?: number
  ttftMs?: number
  elapsedMs?: number
  /* The window the model serves with, for the share of it this exchange used. */
  contextLength?: number
}

export interface ReplyStatsInput {
  usage: TokenUsage | null
  model: string | null
  preset?: string
  tokPerSec?: number | null
  ttftMs?: number | null
  elapsedMs?: number | null
  contextLength?: number
}

const count = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) && value >= 0 ? Math.round(value) : undefined

const positive = (value: unknown): number | undefined => {
  const n = count(value)
  return n !== undefined && n > 0 ? n : undefined
}

/* The cached-token count of a usage block, or undefined when the server did not give a usable one. Never more than
   the prompt: a figure larger than the prompt is a server mistake, not a reason to claim more than was sent. */
export function cachedTokens(usage: TokenUsage | null | undefined): number | undefined {
  const cached = count(usage?.prompt_tokens_details?.cached_tokens)
  if (cached === undefined) return undefined
  const prompt = count(usage?.prompt_tokens)
  return prompt === undefined ? cached : Math.min(cached, prompt)
}

/* The stats of a finished reply, or null when there is nothing to show: a reply that ended without a usage block
   (stopped, or the connection dropped) has no tokens to report, and a bare model name is not worth a line. */
export function buildReplyStats(input: ReplyStatsInput): ReplyStats | null {
  const { usage } = input
  if (!usage) return null
  const stats: ReplyStats = {}
  if (input.model) stats.model = input.model
  if (input.preset) stats.preset = input.preset
  const prompt = count(usage.prompt_tokens)
  if (prompt !== undefined) stats.promptTokens = prompt
  const cached = cachedTokens(usage)
  if (cached !== undefined) stats.cachedTokens = cached
  const completion = count(usage.completion_tokens)
  if (completion !== undefined) stats.completionTokens = completion
  if (typeof input.tokPerSec === "number" && Number.isFinite(input.tokPerSec) && input.tokPerSec > 0) stats.tokPerSec = input.tokPerSec
  const ttft = positive(input.ttftMs)
  if (ttft !== undefined) stats.ttftMs = ttft
  const elapsed = positive(input.elapsedMs)
  if (elapsed !== undefined) stats.elapsedMs = elapsed
  const window = positive(input.contextLength)
  if (window !== undefined) stats.contextLength = window
  return stats.promptTokens === undefined && stats.completionTokens === undefined ? null : stats
}

/* `qwen3-coder-30b-a3b-colibri` -> `qwen3-coder-30b-a3b`: every model here carries the same suffix, which says nothing. */
export function shortModel(id: string): string {
  return id.replace(/-colibri$/, "")
}

/* The share of the window this exchange filled (prompt plus answer), as a percentage with one decimal below 10%
   and none above; undefined without a window. Capped at 100 so a window that grew since does not read as over. */
export function contextShare(stats: ReplyStats): { percent: string; total: number } | undefined {
  const total = stats.contextLength
  if (!total || stats.promptTokens === undefined) return undefined
  const used = stats.promptTokens + (stats.completionTokens ?? 0)
  const percent = Math.min(100, (used / total) * 100)
  return { percent: percent < 10 ? percent.toFixed(1) : String(Math.round(percent)), total }
}

type Translate = (key: string, vars?: Record<string, string | number>) => string

const seconds = (ms: number) => (ms / 1000).toFixed(ms < 10000 ? 1 : 0)

/* The pieces of a reply's line, in the order they are shown, each already worded. Empty when there is nothing to say.
   Cache wording: "X of Y prompt tokens cached" for a continuation, "Y prompt tokens, none cached" for a prompt the
   engine read in full, plain "Y prompt tokens" when the server does not say. */
export function replyParts(stats: ReplyStats, t: Translate): string[] {
  const parts: string[] = []
  if (stats.model) parts.push(shortModel(stats.model))
  if (stats.preset) parts.push(t("reply.viaPreset", { preset: stats.preset }))
  if (stats.promptTokens !== undefined) {
    if (stats.cachedTokens === undefined) parts.push(t("reply.prompt", { prompt: stats.promptTokens }))
    else if (stats.cachedTokens === 0) parts.push(t("reply.promptNoneCached", { prompt: stats.promptTokens }))
    else parts.push(t("reply.promptCached", { cached: stats.cachedTokens, prompt: stats.promptTokens }))
  }
  if (stats.completionTokens !== undefined) parts.push(t("topbar.tokens", { n: stats.completionTokens }))
  if (stats.tokPerSec !== undefined) parts.push(t("topbar.tokPerSec", { n: stats.tokPerSec.toFixed(1) }))
  if (stats.ttftMs !== undefined) parts.push(t("reply.firstToken", { s: seconds(stats.ttftMs) }))
  if (stats.elapsedMs !== undefined) parts.push(t("reply.elapsed", { s: seconds(stats.elapsedMs) }))
  const share = contextShare(stats)
  if (share) parts.push(t("reply.context", { percent: share.percent, total: share.total.toLocaleString("en-US") }))
  return parts
}
