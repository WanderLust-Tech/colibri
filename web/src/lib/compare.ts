/* The compare view: the same prompt to several models, one after another, side by side. The logic that does not need a
   page lives here so it can be tested: which models can be compared, the blind labels, the scoreboard, and the loop that
   runs the models in turn. */

import type { ModelInfo } from "./api"

/* A model a person can chat with. An image or music model refuses chat, and a decision model only answers /v1/systemone.
   A server that does not say what a model does (no capabilities) is taken as a chat model. */
export function isChatModel(info: ModelInfo): boolean {
  const caps = info.capabilities
  if (!caps || caps.length === 0) return true
  return caps.includes("chat")
}

/* Most models in one round: each one loads in turn, and a wider page stops being readable. */
export const MAX_COMPARED = 4

export const BLIND_LABELS = ["A", "B", "C", "D"] as const

/* A copy in random order (Fisher-Yates). `random` is injectable so a test can fix the order. */
export function shuffle<T>(items: readonly T[], random: () => number = Math.random): T[] {
  const out = [...items]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

export interface Tally { wins: number; rounds: number }
export type Scores = Record<string, Tally>

/* A round was judged: every model in it played, the winner won. */
export function recordVote(scores: Scores, played: readonly string[], winner: string): Scores {
  if (!played.includes(winner)) return scores
  const next: Scores = { ...scores }
  for (const model of new Set(played)) {
    const before = next[model] ?? { wins: 0, rounds: 0 }
    next[model] = { wins: before.wins + (model === winner ? 1 : 0), rounds: before.rounds + 1 }
  }
  return next
}

export const winPercent = (tally: Tally): number => (tally.rounds ? Math.round((tally.wins / tally.rounds) * 100) : 0)

/* The scoreboard best first: by share of rounds won, then by rounds played, then by name. */
export function ranked(scores: Scores): { model: string; wins: number; rounds: number; percent: number }[] {
  return Object.entries(scores)
    .map(([model, tally]) => ({ model, ...tally, percent: winPercent(tally) }))
    .sort((a, b) => b.percent - a.percent || b.rounds - a.rounds || a.model.localeCompare(b.model))
}

const SCORES_KEY = "colibri.compareScores"

/* Read what an earlier visit saved. Anything that is not a plain map of two counts is dropped, so a damaged value cannot
   break the page; restricted storage gives an empty scoreboard. */
export function loadScores(storage: Pick<Storage, "getItem">): Scores {
  try {
    const raw = storage.getItem(SCORES_KEY)
    if (!raw) return {}
    const data = JSON.parse(raw) as unknown
    if (!data || typeof data !== "object" || Array.isArray(data)) return {}
    const out: Scores = {}
    for (const [model, tally] of Object.entries(data as Record<string, unknown>)) {
      const t = tally as { wins?: unknown; rounds?: unknown } | null
      if (t && Number.isInteger(t.wins) && Number.isInteger(t.rounds) && (t.wins as number) >= 0 && (t.rounds as number) >= (t.wins as number)) {
        out[model] = { wins: t.wins as number, rounds: t.rounds as number }
      }
    }
    return out
  } catch { return {} }
}

export function saveScores(storage: Pick<Storage, "setItem" | "removeItem">, scores: Scores) {
  try {
    if (Object.keys(scores).length) storage.setItem(SCORES_KEY, JSON.stringify(scores))
    else storage.removeItem(SCORES_KEY)
  } catch { /* restricted storage mode */ }
}

/* Run the items one after another. `runOne` reports its own failure on its own column, so one model failing does not
   stop the others; an abort (the Stop button) ends the loop before the next item starts. */
export async function runSequentially<T>(items: readonly T[], runOne: (item: T, index: number) => Promise<void>, signal: AbortSignal): Promise<void> {
  for (let index = 0; index < items.length; index++) {
    if (signal.aborted) return
    try { await runOne(items[index], index) } catch { /* shown on that column by runOne */ }
  }
}
