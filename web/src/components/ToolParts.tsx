import { useEffect, useState } from "react"
import { CircleAlert, LoaderCircle, ShieldQuestionMark, Wrench } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Markdown } from "@/components/Markdown"
import { useLocale } from "@/i18n"
import {
  argumentsLine, argumentsText, clipText, secondsLeft,
  type ApprovalPart, type CallPart, type TurnPart,
} from "@/lib/tools"
import { cn } from "@/lib/utils"

/* What a tool returned can be anything, including text that looks like markup or markdown. It is shown as text in a <pre>
   and nowhere else: this page never builds markup from tool output. */
const RESULT_SHOWN = 4000

type Answer = (approvalId: string, allow: boolean, always: boolean) => void

function CallCard({ part }: { part: CallPart }) {
  const { t } = useLocale()
  const args = argumentsText(part.arguments)
  return (
    <div className={cn("tool-card", part.state)} role="group" aria-label={t("tools.cardLabel", { name: part.name })}>
      <div className="tool-head">
        {part.state === "running" ? <LoaderCircle className="size-3.5 animate-spin" /> : part.state === "ok" ? <Wrench className="size-3.5" /> : <CircleAlert className="size-3.5" />}
        <strong>{part.name}</strong>
        <code className="tool-args">{argumentsLine(part.arguments)}</code>
        <span className="tool-state">{t(`tools.state.${part.state}`)}</span>
      </div>
      {(args || part.result) && (
        <details className="tool-more">
          <summary>{t("tools.details")}{part.chars !== undefined ? ` · ${t("tools.chars", { n: part.chars.toLocaleString() })}` : ""}</summary>
          {args && <><div className="tool-label">{t("tools.arguments")}</div><pre className="tool-pre">{args}</pre></>}
          {part.result && <><div className="tool-label">{t("tools.result")}</div><pre className="tool-pre">{clipText(part.result, RESULT_SHOWN)}</pre></>}
        </details>
      )}
    </div>
  )
}

function ApprovalCard({ part, live, onAnswer }: { part: ApprovalPart; live: boolean; onAnswer: Answer }) {
  const { t } = useLocale()
  /* Only a reply that is still streaming can be answered. A question left open on a turn that is over is shown as such,
     whatever its stored state says, so a button that could never work is never drawn. */
  const state = !live && (part.state === "waiting" || part.state === "sending") ? "stopped" : part.state
  const open = state === "waiting" || state === "sending"
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!open) return
    setNow(Date.now())
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [open])
  const left = secondsLeft(part.deadline, now)
  const canAnswer = state === "waiting" && left > 0
  const args = argumentsText(part.arguments)
  return (
    <div className={cn("tool-card approval", state)} role="group" aria-label={t("tools.approvalLabel", { name: part.tool })}>
      <div className="tool-head">
        <ShieldQuestionMark className="size-3.5" />
        <strong>{part.tool}</strong>
        <span className="tool-state">{open ? t("tools.approvalTitle") : t(`tools.approval.${state}`)}</span>
      </div>
      <p className="tool-summary">{part.summary}</p>
      {args && (
        <details className="tool-more">
          <summary>{t("tools.details")}</summary>
          <pre className="tool-pre">{args}</pre>
        </details>
      )}
      {open && (
        <>
          <div className="tool-actions">
            <Button type="button" size="sm" disabled={!canAnswer} onClick={() => onAnswer(part.id, true, false)}>{t("tools.allow")}</Button>
            <Button type="button" size="sm" variant="secondary" disabled={!canAnswer} title={t("tools.allowAlwaysHelp")} onClick={() => onAnswer(part.id, true, true)}>{t("tools.allowAlways")}</Button>
            <Button type="button" size="sm" variant="secondary" disabled={!canAnswer} onClick={() => onAnswer(part.id, false, false)}>{t("tools.decline")}</Button>
            <span className="tool-countdown" aria-live="off">{left > 0 ? t("tools.countdown", { n: left }) : t("tools.timeUp")}</span>
          </div>
          {part.problem === "send-failed" && <p className="tool-problem" role="alert">{t("tools.sendFailed")}</p>}
        </>
      )}
      {!open && part.always && state === "allowed" && <p className="tool-note">{t("tools.alwaysNote")}</p>}
    </div>
  )
}

/* The inside of an assistant turn that used tools: text, calls and questions in the order they happened. `live` is true only
   for the reply that is streaming now. */
export function ToolTurn({ parts, live, onAnswer }: { parts: TurnPart[]; live: boolean; onAnswer: Answer }) {
  const { t } = useLocale()
  const last = parts[parts.length - 1]
  /* Between a tool's end and the next words the model is reading what came back. */
  const thinking = live && !!last && ((last.kind === "call" && last.state !== "running")
    || (last.kind === "approval" && (last.state === "allowed" || last.state === "declined")))
  return (
    <>
      {parts.map((part, index) => part.kind === "text"
        ? <Markdown key={index} text={part.text} />
        : part.kind === "call"
          ? <CallCard key={index} part={part} />
          : <ApprovalCard key={index} part={part} live={live} onAnswer={onAnswer} />)}
      {thinking && <span className="typing" aria-label={t("ui.generating")}><i /><i /><i /></span>}
    </>
  )
}
