import { useEffect, useMemo, useRef, useState } from "react"
import { Eye, Play, RotateCcw, Square, Trophy } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { streamChat, type ModelInfo } from "@/lib/api"
import {
  BLIND_LABELS, MAX_COMPARED, loadScores, ranked, recordVote, runSequentially, saveScores, shuffle, type Scores,
} from "@/lib/compare"
import { explainError } from "@/lib/explainError"
import { controlDefaults } from "@/lib/presets"
import { buildReplyStats, replyParts, type ReplyStats } from "@/lib/replyStats"
import { cn } from "@/lib/utils"
import { useLocale } from "../i18n"
import { Markdown } from "./Markdown"

type Status = "waiting" | "loading" | "running" | "done" | "error" | "stopped"

interface Column {
  key: string
  model: string
  label: string
  status: Status
  text: string
  reasoning: string
  stats: ReplyStats | null
  error: string
  startedAt: number
  firstMs: number | null
}

export interface CompareSettings { temperature: number; maxTokens: number; thinking: boolean }

/* The same prompt to several models, one after another, side by side. The router holds one model at a time for
   computing, so they run in turn; each column says what it is waiting for. */
export function Compare({ baseUrl, apiKey, connected, models, settings }: {
  baseUrl: string
  apiKey: string
  connected: boolean
  models: ModelInfo[]
  settings: CompareSettings
}) {
  const { t } = useLocale()
  const [prompt, setPrompt] = useState("")
  const [chosen, setChosen] = useState<string[]>([])
  const [blind, setBlind] = useState(true)
  const [columns, setColumns] = useState<Column[]>([])
  const [running, setRunning] = useState(false)
  const [revealed, setRevealed] = useState(true)
  const [voted, setVoted] = useState<string | null>(null)
  const [scores, setScores] = useState<Scores>(() => { try { return loadScores(localStorage) } catch { return {} } })
  const [now, setNow] = useState(() => performance.now())
  const abortRef = useRef<AbortController | null>(null)

  const byId = useMemo(() => Object.fromEntries(models.map((info) => [info.id, info])), [models])

  /* Start with the first two models, once. After that the person's choice stays as it is, even when it is empty; only a
     choice that is no longer in the list (a reconnect to another server) is dropped. */
  const started = useRef(false)
  useEffect(() => {
    if (!models.length) return
    if (!started.current) {
      started.current = true
      setChosen(models.slice(0, 2).map((info) => info.id))
      return
    }
    setChosen((current) => {
      const still = current.filter((id) => byId[id])
      return still.length === current.length ? current : still
    })
  }, [models, byId])

  useEffect(() => () => abortRef.current?.abort(), [])

  const waiting = columns.some((column) => column.status === "loading")
  useEffect(() => {
    if (!waiting) return
    const timer = window.setInterval(() => setNow(performance.now()), 500)
    return () => window.clearInterval(timer)
  }, [waiting])

  const toggle = (id: string) => setChosen((current) =>
    current.includes(id) ? current.filter((item) => item !== id) : current.length >= MAX_COMPARED ? current : [...current, id])

  const run = async () => {
    const text = prompt.trim()
    if (!text || running || chosen.length < 2) return
    const order = blind ? shuffle(chosen) : chosen
    const stamp = Date.now()
    const fresh: Column[] = order.map((model, index) => ({
      key: `${stamp}-${index}`, model, label: BLIND_LABELS[index], status: "waiting", text: "", reasoning: "", stats: null, error: "", startedAt: 0, firstMs: null,
    }))
    setColumns(fresh); setRevealed(!blind); setVoted(null); setRunning(true)
    const controller = new AbortController()
    abortRef.current = controller
    const patch = (key: string, change: Partial<Column> | ((column: Column) => Partial<Column>)) =>
      setColumns((current) => current.map((column) => column.key === key ? { ...column, ...(typeof change === "function" ? change(column) : change) } : column))

    await runSequentially(fresh, async (column) => {
      const info = byId[column.model]
      const preset = info?.preset ? controlDefaults(info.preset.defaults) : {}
      const start = performance.now()
      let first: number | null = null
      let count = 0
      patch(column.key, { status: "loading", startedAt: start })
      try {
        const result = await streamChat({
          baseUrl, apiKey, model: column.model,
          messages: [{ id: "compare", role: "user", content: text }],
          temperature: preset.temperature ?? settings.temperature,
          maxTokens: preset.maxTokens ?? settings.maxTokens,
          enableThinking: settings.thinking,
          signal: controller.signal,
          onReasoning: (delta) => {
            first ??= performance.now() - start
            count++
            patch(column.key, (c) => ({ status: "running", reasoning: c.reasoning + delta, firstMs: first }))
          },
          onDelta: (delta) => {
            first ??= performance.now() - start
            count++
            patch(column.key, (c) => ({ status: "running", text: c.text + delta, firstMs: first }))
          },
        })
        const elapsed = performance.now() - start
        const decode = first === null ? 0 : (elapsed - first) / 1000
        patch(column.key, {
          status: "done",
          stats: buildReplyStats({
            usage: result.usage, model: result.model, preset: info?.preset ? column.model : undefined,
            tokPerSec: count > 1 && decode > 0 ? (count - 1) / decode : null, ttftMs: first, elapsedMs: elapsed, contextLength: info?.contextLength,
          }),
        })
      } catch (cause) {
        if (controller.signal.aborted) patch(column.key, { status: "stopped" })
        else patch(column.key, { status: "error", error: cause instanceof Error ? cause.message : String(cause) })
      }
    }, controller.signal)

    setColumns((current) => current.map((column) => column.status === "waiting" || column.status === "loading" || column.status === "running" ? { ...column, status: "stopped" } : column))
    setRunning(false)
    abortRef.current = null
  }

  const stop = () => abortRef.current?.abort()

  const finished = columns.filter((column) => column.status === "done")
  const canVote = !running && voted === null && finished.length >= 2
  const vote = (model: string) => {
    if (!canVote) return
    setScores((current) => {
      const next = recordVote(current, finished.map((column) => column.model), model)
      try { saveScores(localStorage, next) } catch { /* no storage */ }
      return next
    })
    setVoted(model)
    setRevealed(true)
  }
  const resetScores = () => { setScores({}); try { saveScores(localStorage, {}) } catch { /* no storage */ } }

  const seconds = (ms: number) => `${Math.round(ms / 1000)} s`
  const board = ranked(scores)

  return <section className="compare-workspace">
    <header className="page-heading"><span>COLIBRI / COMPARE</span><h1>{t("compare.title")}</h1></header>
    <p className="compare-note">{t("compare.note")}</p>

    <div className="compare-controls">
      <Textarea id="compare-prompt" value={prompt} placeholder={t("compare.placeholder")} disabled={running}
        onChange={(event) => setPrompt(event.target.value)}
        onKeyDown={(event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); void run() } }} />
      <div className="compare-models" role="group" aria-label={t("compare.models")}>
        {models.map((info) => {
          const on = chosen.includes(info.id)
          return <button key={info.id} type="button" className={cn("compare-chip", on && "active")} aria-pressed={on}
            disabled={running || (!on && chosen.length >= MAX_COMPARED)} onClick={() => toggle(info.id)}>
            {info.id}{info.preset ? <small>{t("compare.preset")}</small> : null}
          </button>
        })}
      </div>
      <div className="compare-actions">
        <label className="compare-blind"><input type="checkbox" checked={blind} disabled={running} onChange={(event) => setBlind(event.target.checked)} /> {t("compare.blind")}</label>
        {running
          ? <Button type="button" variant="secondary" onClick={stop}><Square className="size-4" /> {t("compare.stop")}</Button>
          : <Button type="button" onClick={() => void run()} disabled={!connected || !prompt.trim() || chosen.length < 2}><Play className="size-4" /> {t("compare.run")}</Button>}
        <span className="compare-hint">{chosen.length < 2 ? t("compare.pickTwo") : t("compare.count", { n: chosen.length })}</span>
      </div>
    </div>

    {columns.length > 0 && <div className="compare-grid" style={{ gridTemplateColumns: `repeat(${columns.length}, minmax(0, 1fr))` }}>
      {columns.map((column) => {
        const known = column.error ? explainError(column.error) : null
        const title = revealed ? column.model : `${t("compare.answer")} ${column.label}`
        /* The heading names the model; blind, the line must not give the preset away either. */
        const parts = column.stats ? replyParts({ ...column.stats, model: undefined, preset: revealed ? column.stats.preset : undefined }, t) : []
        return <article key={column.key} className={cn("compare-column", voted === column.model && "winner")} aria-label={title}>
          <header><h2>{title}</h2>{voted === column.model && <Trophy className="size-4" aria-label={t("compare.winner")} />}</header>
          <div className="compare-body">
            {column.status === "waiting" && <p className="compare-status">{t("compare.waiting")}</p>}
            {column.status === "loading" && <p className="compare-status" role="status">{t("compare.loading", { s: seconds(Math.max(0, now - column.startedAt)) })}</p>}
            {column.reasoning && <details className="reasoning"><summary>{t("sidebar.reasoning")}</summary><div className="reasoning-body">{column.reasoning}</div></details>}
            {column.text && <Markdown text={column.text} />}
            {column.status === "stopped" && !column.text && <p className="compare-status">{t("compare.stopped")}</p>}
            {column.status === "error" && <div className="error-banner" role="alert">{known && <div className="error-explain">{t(known.key, known.vars)}</div>}<div className="error-original">{column.error}</div></div>}
          </div>
          {parts.length > 0 && <div className="reply-stats">{parts.join(" · ")}</div>}
          {canVote && column.status === "done" && <Button type="button" variant="secondary" size="sm" onClick={() => vote(column.model)}><Trophy className="size-4" /> {t("compare.best")}</Button>}
        </article>
      })}
    </div>}

    {!revealed && !running && finished.length > 0 && <Button type="button" variant="secondary" size="sm" className="compare-reveal" onClick={() => setRevealed(true)}><Eye className="size-4" /> {t("compare.reveal")}</Button>}

    {board.length > 0 && <section className="compare-scores" aria-label={t("compare.scores")}>
      <header><h2>{t("compare.scores")}</h2><button type="button" className="icon-action" aria-label={t("compare.resetScores")} title={t("compare.resetScores")} onClick={resetScores}><RotateCcw /></button></header>
      <table>
        <thead><tr><th>{t("compare.model")}</th><th>{t("compare.wins")}</th><th>{t("compare.rounds")}</th><th>%</th></tr></thead>
        <tbody>{board.map((row) => <tr key={row.model}><td>{row.model}</td><td>{row.wins}</td><td>{row.rounds}</td><td>{row.percent}%</td></tr>)}</tbody>
      </table>
    </section>}
  </section>
}
