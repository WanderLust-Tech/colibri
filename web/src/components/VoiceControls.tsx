import { Mic, Repeat, Square, Volume2, VolumeX } from "lucide-react"

import { useLocale } from "@/i18n"
import type { VoiceState } from "@/lib/useVoice"
import { cn } from "@/lib/utils"

interface VoiceControlsProps {
  state: VoiceState
  level: number
  /* Read the replies aloud. */
  replies: boolean
  onReplies: () => void
  /* After each reply, listen again: a conversation with no keys. */
  handsFree: boolean
  onHandsFree: () => void
  onListen: () => void
  /* While listening: send what was said now, without waiting for a pause. */
  onFinish: () => void
  /* While writing it down or speaking: give up. */
  onStop: () => void
}

/* The chat composer's voice buttons: speak to it (the microphone), have the replies read aloud, and listen again after each (hands-free). */
export default function VoiceControls({ state, level, replies, onReplies, handsFree, onHandsFree, onListen, onFinish, onStop }: VoiceControlsProps) {
  const { t } = useLocale()
  const listening = state === "listening"
  const busy = state !== "idle"
  const label = listening ? t("voice.send") : busy ? t("voice.stop") : t("voice.listen")
  const status = state === "listening" ? t("voice.listening") : state === "transcribing" ? t("voice.transcribing") : state === "speaking" ? t("voice.speaking") : ""
  return (
    <>
      <button type="button" className={cn("voice-chip", state === "listening" && "listening")} aria-pressed={state === "listening"}
        aria-label={label} title={label}
        style={state === "listening" ? { ["--voice-level" as string]: String(Math.min(1, level * 8)) } : undefined}
        onClick={listening ? onFinish : busy ? onStop : onListen}>
        {busy && !listening ? <Square /> : <Mic />}
      </button>
      <button type="button" className="voice-chip" aria-pressed={replies} aria-label={t("voice.replies")} title={t("voice.repliesHelp")} onClick={onReplies}>
        {replies ? <Volume2 /> : <VolumeX />}
      </button>
      <button type="button" className="voice-chip" aria-pressed={handsFree} aria-label={t("voice.handsFree")} title={t("voice.handsFreeHelp")} onClick={onHandsFree}>
        <Repeat />
      </button>
      <span className="voice-status" role="status" aria-live="polite">{status}</span>
    </>
  )
}
