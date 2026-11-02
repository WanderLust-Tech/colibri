import { useEffect, useRef, useState } from "react"
import { AudioLines, Clock, Copy, Dices, Download, Footprints, Languages, Mic, Music2, Paintbrush, Repeat2, RefreshCw, RotateCcw, SlidersHorizontal, Upload, X } from "lucide-react"

import type { GeneratedMusic, MusicLimits, MusicTask, RepaintMode } from "@/lib/api"
import { duration as formatSeconds, elapsedClock } from "@/lib/images"
import {
  MAX_CLIP_BYTES, MUSIC_LANGUAGES, clampDuration, clampMusicSteps, lyricBytes, musicProgressFraction, songClock, validDuration,
  type ClipProblem, type MusicRun,
} from "@/lib/music"
import type { Clip } from "@/lib/useMusic"
import { cn } from "@/lib/utils"
import { useLocale } from "../i18n"

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 250)
    return () => window.clearInterval(timer)
  }, [])
  return <time className="image-clock">{elapsedClock((now - since) / 1000)}</time>
}

export function MusicProgressCard({ request, run, onCancel }: { request: GeneratedMusic; run: MusicRun; onCancel: () => void }) {
  const { t } = useLocale()
  const progress = run.progress
  const fraction = musicProgressFraction(progress)
  const stage = !progress ? t(run.plain ? "music.stage.working" : "music.stage.queued")
    : progress.stage === "denoise" ? t("music.stage.denoise", { step: Math.min(progress.steps, progress.step + 1), steps: progress.steps })
    : progress.stage === "encode" || progress.stage === "text" || progress.stage === "condition" || progress.stage === "decode" ? t(`music.stage.${progress.stage}`)
    : t("music.stage.working")
  return <div className="image-turn music-turn">
    <p className="music-request">{requestLine(request, t)}</p>
    <div className="image-status">
      <span className="image-stage" aria-live="polite">{stage}</span>
      <Elapsed since={run.started} />
      <button type="button" className="image-cancel" onClick={onCancel}><X />{t("image.cancel")}</button>
    </div>
    <div className={cn("image-bar", !progress && "waiting")} role="progressbar" aria-label={t("music.progress")}
      aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress ? Math.round(fraction * 100) : undefined}>
      <span style={{ width: `${(fraction * 100).toFixed(1)}%` }} />
    </div>
  </div>
}

/* The sentence that says what is being made: a song of some length, a cover of a clip, a clip repainted,
   and the reference it takes its style from. */
function requestLine(request: GeneratedMusic, t: (key: string, values?: Record<string, string | number>) => string) {
  const parts: string[] = []
  const fmt = (seconds: number) => String(Math.round(seconds * 10) / 10)
  if (request.task === "cover" && request.source) parts.push(t("music.requested.cover", { name: request.source.name }))
  else if (request.task === "repaint" && request.source && request.repaint)
    parts.push(t("music.requested.repaint", { name: request.source.name, from: fmt(request.repaint.from), to: request.repaint.to !== undefined ? fmt(request.repaint.to) : "∞" }))
  else parts.push(t("music.requested", { length: songClock(request.duration) }))
  if (request.reference) parts.push(t("music.result.reference", { name: request.reference.name }))
  return parts.join(" · ")
}

export function MusicResultCard({ song, copied, busy, onDownload, onCopySeed, onAgain }: {
  song: GeneratedMusic
  copied: boolean
  busy: boolean
  onDownload: () => void
  onCopySeed: () => void
  onAgain: () => void
}) {
  const { t, locale } = useLocale()
  const timings = song.timings
  const times = timings && [timings.text, timings.condition, timings.denoise, timings.decode].every((value) => typeof value === "number")
    ? {
      text: formatSeconds(timings.text!, locale), condition: formatSeconds(timings.condition!, locale),
      denoise: formatSeconds(timings.denoise!, locale), decode: formatSeconds(timings.decode!, locale),
    }
    : undefined
  const breakdown = times && (typeof timings!.encode === "number" ? t("music.timingsClip", { ...times, encode: formatSeconds(timings!.encode, locale) }) : t("music.timings", times))
  const fmt = (seconds: number) => String(Math.round(seconds * 10) / 10)
  const origin = song.task === "cover" && song.source ? t("music.result.cover", { name: song.source.name })
    : song.task === "repaint" && song.source && song.repaint
      ? t("music.result.repaint", { name: song.source.name, from: fmt(song.repaint.from), to: song.repaint.to !== undefined ? fmt(song.repaint.to) : songClock(song.duration) })
      : undefined
  return <div className="image-turn music-turn">
    {/* the browser's own player: seeking, volume and keyboard control come with it */}
    <audio className="music-player" controls preload="metadata" src={song.url} aria-label={t("music.alt", { prompt: song.prompt || song.lyrics?.split("\n")[0] || "" })} />
    {(origin || song.reference) && <p className="music-request">{[origin, song.reference && t("music.result.reference", { name: song.reference.name })].filter(Boolean).join(" · ")}</p>}
    <p className="image-meta">
      <span>{songClock(song.duration)}</span>
      <span>{t("image.meta.steps", { n: song.steps })}</span>
      <span>{t("image.meta.seed", { seed: song.seed })}</span>
      {song.seconds != null && <span title={breakdown}>{formatSeconds(song.seconds, locale)}</span>}
      {song.truncated && <span className="music-cut" title={t("music.truncatedHelp")}>{t("music.truncated")}</span>}
    </p>
    <div className="image-actions">
      <button type="button" className="image-action" onClick={onDownload} aria-label={t("music.download")} title={t("music.download")}><Download /><span>{t("music.download")}</span></button>
      <button type="button" className="image-action" onClick={onCopySeed} aria-label={t("image.copySeed")} title={t("image.copySeed")}><Copy /><span>{t("image.copySeed")}</span></button>
      <button type="button" className="image-action primary" onClick={onAgain} disabled={busy} aria-label={t("image.again")} title={t("image.again")}><RefreshCw /><span>{t("image.againShort")}</span></button>
      {copied && <span role="status">{t("image.seedCopied")}</span>}
    </div>
  </div>
}

export function MusicStoppedCard({ request, finish, busy, onRetry }: { request: GeneratedMusic; finish?: string; busy: boolean; onRetry: () => void }) {
  const { t } = useLocale()
  return <div className="image-turn">
    <div className="image-stopped">
      <span>{finish === "aborted" ? t("image.cancelled") : t("image.failed")}</span>
      <small>{requestLine(request, t)} · {t("image.meta.steps", { n: request.steps })}</small>
      <button type="button" className="image-action" onClick={onRetry} disabled={busy}><RotateCcw /><span>{t("image.retry")}</span></button>
    </div>
  </div>
}

/* Length, steps, seed, language and the lyrics switch, in the composer where the image
   controls and the chat's reasoning chip sit. A typed value counts as soon as it is valid, so
   sending straight from a field uses what is on screen; leaving the field puts an invalid one
   inside the range. */
export function MusicControls({ limits, task, onTask, cover, repaint, hasReference, onReferenceFile, duration, steps, seed, seedValid, language, lyricsOpen, lyricsFilled, onDuration, onSteps, onSeed, onLanguage, onLyrics }: {
  limits: MusicLimits
  task: MusicTask
  onTask: (task: MusicTask) => void
  cover: { strength: string; noise: string; onStrength: (text: string) => void; onNoise: (text: string) => void }
  repaint: {
    from: string; to: string; mode: RepaintMode; strength: string; spanOk: boolean
    onFrom: (text: string) => void; onTo: (text: string) => void; onMode: (mode: RepaintMode) => void; onStrength: (text: string) => void
  }
  hasReference: boolean
  onReferenceFile: (file: File | undefined) => void
  duration: number
  steps: number
  seed: string
  seedValid: boolean
  language: string
  lyricsOpen: boolean
  lyricsFilled: boolean
  onDuration: (seconds: number) => void
  onSteps: (steps: number) => void
  onSeed: (seed: string) => void
  onLanguage: (code: string) => void
  onLyrics: () => void
}) {
  const { t } = useLocale()
  const [durationText, setDurationText] = useState(String(duration))
  const [stepsText, setStepsText] = useState(String(steps))
  /* The text follows the value only when they disagree (a preset, another model's range): a half-typed
     "2." already means 2, and an invalid entry is NaN, and neither may be rewritten under the cursor. */
  useEffect(() => { if (Number.isFinite(duration)) setDurationText((text) => Number(text) === duration ? text : String(duration)) }, [duration])
  useEffect(() => { if (Number.isFinite(steps)) setStepsText((text) => Number(text) === steps ? text : String(steps)) }, [steps])
  const commitDuration = (text: string) => {
    const value = clampDuration(Number(text || duration), limits)
    setDurationText(String(value)); onDuration(value)
  }
  const commitSteps = (text: string) => {
    const value = clampMusicSteps(Number(text || steps), limits)
    setStepsText(String(value)); onSteps(value)
  }
  const durationOk = validDuration(Number(durationText), limits)
  const referenceInput = useRef<HTMLInputElement>(null)
  const unit =(text: string) => text.trim() !== "" && Number.isFinite(Number(text)) && Number(text) >= 0 && Number(text) <= 1
  const composing = task === "text2music"
  const TaskIcon = { text2music: Music2, cover: Repeat2, repaint: Paintbrush } as const
  return <div className="image-controls">
    {limits.tasks.length > 1 && <div className="music-tasks" role="group" aria-label={t("music.mode.label")}>
      {limits.tasks.map((item) => {
        const Icon = TaskIcon[item]
        return <button key={item} type="button" className={cn("image-chip", task === item && "on")} aria-pressed={task === item}
          title={t(`music.mode.help.${item}`)} onClick={() => onTask(item)}>
          <Icon aria-hidden="true" /><span>{t(`music.mode.${item}`)}</span>
        </button>
      })}
    </div>}
    <button type="button" className={cn("image-chip music-lyrics-chip", (lyricsOpen || lyricsFilled) && "on")} aria-pressed={lyricsOpen}
      title={t("music.lyrics")} onClick={onLyrics}>
      <Mic aria-hidden="true" /><span>{t("music.lyrics")}</span>
    </button>
    {limits.audioIn && <>
      <input ref={referenceInput} type="file" accept=".wav,.mp3,audio/wav,audio/x-wav,audio/mpeg" hidden aria-label={t("music.reference.add")}
        onChange={(event) => { onReferenceFile(event.target.files?.[0]); event.target.value = "" }} />
      <button type="button" className={cn("image-chip", hasReference && "on")} aria-pressed={hasReference}
        title={t("music.reference.help")} onClick={() => referenceInput.current?.click()}>
        <AudioLines aria-hidden="true" /><span>{t("music.reference.add")}</span>
      </button>
    </>}
    {task === "cover" && <>
      <label className={cn("image-chip", !unit(cover.strength) && "invalid")} title={t("music.cover.strengthHelp")}>
        <SlidersHorizontal aria-hidden="true" />
        <input aria-label={t("music.cover.strength")} aria-invalid={!unit(cover.strength)} inputMode="decimal" className="music-unit" value={cover.strength}
          onChange={(event) => cover.onStrength(event.target.value)} />
      </label>
      <label className={cn("image-chip", !unit(cover.noise) && "invalid")} title={t("music.cover.noiseHelp")}>
        <Upload aria-hidden="true" />
        <input aria-label={t("music.cover.noise")} aria-invalid={!unit(cover.noise)} inputMode="decimal" className="music-unit" value={cover.noise}
          onChange={(event) => cover.onNoise(event.target.value)} />
      </label>
    </>}
    {task === "repaint" && <>
      <label className={cn("image-chip", !repaint.spanOk && "invalid")} title={t("music.repaint.help")}>
        <Clock aria-hidden="true" />
        <input aria-label={t("music.repaint.from")} aria-invalid={!repaint.spanOk} inputMode="decimal" className="music-duration" value={repaint.from}
          onChange={(event) => repaint.onFrom(event.target.value)} />
        <span className="image-unit">–</span>
        <input aria-label={t("music.repaint.to")} aria-invalid={!repaint.spanOk} inputMode="decimal" className="music-duration" value={repaint.to}
          placeholder="∞" onChange={(event) => repaint.onTo(event.target.value)} />
        <span className="image-unit">{t("music.durationUnit")}</span>
      </label>
      <label className="image-chip" title={t("music.repaint.mode")}>
        <SlidersHorizontal aria-hidden="true" />
        <select aria-label={t("music.repaint.mode")} value={repaint.mode} onChange={(event) => repaint.onMode(event.target.value as RepaintMode)}>
          {(["balanced", "conservative", "aggressive"] as const).map((item) => <option key={item} value={item}>{t(`music.repaint.mode.${item}`)}</option>)}
        </select>
      </label>
      {repaint.mode === "balanced" && <label className={cn("image-chip", !unit(repaint.strength) && "invalid")} title={t("music.repaint.strengthHelp")}>
        <Upload aria-hidden="true" />
        <input aria-label={t("music.repaint.strength")} aria-invalid={!unit(repaint.strength)} inputMode="decimal" className="music-unit" value={repaint.strength}
          onChange={(event) => repaint.onStrength(event.target.value)} />
      </label>}
    </>}
    {composing && <label className={cn("image-chip", !durationOk && "invalid")} title={t("music.durationHelp", { min: limits.minDuration, max: limits.maxDuration })}>
      <Clock aria-hidden="true" />
      <input aria-label={t("music.duration")} aria-invalid={!durationOk} inputMode="decimal" className="music-duration" value={durationText}
        onChange={(event) => {
          setDurationText(event.target.value)
          const value = Number(event.target.value)
          /* an invalid length is reported as one (NaN), so sending is held back until it is fixed or the field is left */
          onDuration(event.target.value && validDuration(value, limits) ? value : Number.NaN)
        }}
        onBlur={(event) => commitDuration(event.target.value)}
        onKeyDown={(event) => { if (event.key === "Enter") commitDuration(event.currentTarget.value) }} />
      <span className="image-unit">{t("music.durationUnit")}</span>
    </label>}
    <label className="image-chip" title={t("music.steps")}>
      <Footprints aria-hidden="true" />
      <input aria-label={t("music.steps")} inputMode="numeric" className="image-steps" value={stepsText}
        onChange={(event) => {
          setStepsText(event.target.value)
          const value = Number(event.target.value)
          onSteps(event.target.value && Number.isInteger(value) && value >= 1 && value <= limits.maxSteps ? value : Number.NaN)
        }}
        onBlur={(event) => commitSteps(event.target.value)}
        onKeyDown={(event) => { if (event.key === "Enter") commitSteps(event.currentTarget.value) }} />
      <span className="image-unit">{t("image.stepsUnit")}</span>
    </label>
    <label className="image-chip" title={t("music.language")}>
      <Languages aria-hidden="true" />
      <select aria-label={t("music.language")} value={language} onChange={(event) => onLanguage(event.target.value)}>
        <option value="">{t("music.languageAuto")}</option>
        {MUSIC_LANGUAGES.map((code) => <option key={code} value={code}>{code}</option>)}
      </select>
    </label>
    <label className={cn("image-chip", !seedValid && "invalid")} title={t("music.seedHelp")}>
      <Dices aria-hidden="true" />
      <input aria-label={t("image.seed")} aria-invalid={!seedValid} inputMode="numeric" className="image-seed" value={seed}
        placeholder={t("image.seedRandom")} onChange={(event) => onSeed(event.target.value)} />
      {seed && <button type="button" className="image-clear" aria-label={t("image.seedClear")} title={t("image.seedClear")} onClick={() => onSeed("")}><X /></button>}
    </label>
  </div>
}

/* The lyrics box above the prompt, open on demand. The count is the engine's: UTF-8 bytes. */
export function MusicLyrics({ value, max, onChange }: { value: string; max: number; onChange: (text: string) => void }) {
  const { t } = useLocale()
  const bytes = lyricBytes(value)
  return <div className="music-lyrics">
    <textarea aria-label={t("music.lyrics")} value={value} placeholder={t("music.lyricsPlaceholder")} rows={5}
      className={cn(bytes > max && "invalid")} onChange={(event) => onChange(event.target.value)} />
    <small className={cn(bytes > max && "invalid")}>{t("music.lyricsBytes", { bytes, max })}</small>
  </div>
}

/* The clips of a cover, a repaint or a reference, above the prompt: a button to choose one, and once chosen its name, its length,
   a player and the way to take it away. The file input is the button; nothing is read until a file is picked. */
export function MusicClips({ task, source, reference, sourceProblem, referenceProblem, onChoose, onClear }: {
  task: MusicTask
  source: Clip | null
  reference: Clip | null
  sourceProblem: ClipProblem | undefined
  referenceProblem: ClipProblem | undefined
  onChoose: (role: "source" | "reference", file: File | undefined) => void
  onClear: (role: "source" | "reference") => void
}) {
  const { t } = useLocale()
  const showSource = task !== "text2music"
  const showReference = reference !== null           /* chosen with the Reference chip in the controls */
  if (!showSource && !showReference) return null
  return <div className="music-clips">
    {showSource && <ClipCard role="source" clip={source} problem={sourceProblem} label={t(task === "cover" ? "music.source.add.cover" : "music.source.add.repaint")}
      title={t("music.source.label")} onChoose={onChoose} onClear={onClear} />}
    {showReference && <ClipCard role="reference" clip={reference} problem={referenceProblem} label={t("music.reference.add")}
      title={t("music.reference.label")} onChoose={onChoose} onClear={onClear} />}
  </div>
}

function ClipCard({ role, clip, problem, label, title, onChoose, onClear }: {
  role: "source" | "reference"
  clip: Clip | null
  problem: ClipProblem | undefined
  label: string
  title: string
  onChoose: (role: "source" | "reference", file: File | undefined) => void
  onClear: (role: "source" | "reference") => void
}) {
  const { t } = useLocale()
  const input = useRef<HTMLInputElement>(null)
  return <div className={cn("music-clip", problem && "invalid")} data-role={role}>
    <input ref={input} type="file" accept=".wav,.mp3,audio/wav,audio/x-wav,audio/mpeg" hidden aria-label={`${title}: ${label}`}
      onChange={(event) => { onChoose(role, event.target.files?.[0]); event.target.value = "" }} />
    {clip ? <>
      <div className="music-clip-head">
        <span className="music-clip-role">{title}</span>
        <span className="music-clip-name" title={clip.name}>{clip.name}</span>
        {clip.seconds !== undefined && <time>{songClock(clip.seconds)}</time>}
        <button type="button" className="image-clear" aria-label={t("music.clip.remove")} title={t("music.clip.remove")} onClick={() => onClear(role)}><X /></button>
      </div>
      <audio className="music-clip-player" controls preload="metadata" src={clip.url} aria-label={`${title}: ${clip.name}`} />
      {clip.seconds === undefined && !problem && <small>{t("music.clip.unreadable")}</small>}
      {problem && <small role="alert">{problem === "music.clip.large" ? t(problem, { mb: Math.round(MAX_CLIP_BYTES / 1048576) }) : problem === "music.clip.long" ? t(problem, { minutes: "10" }) : t(problem)}</small>}
    </> : <button type="button" className="image-chip music-clip-add" onClick={() => input.current?.click()}>
      <Upload aria-hidden="true" /><span>{label}</span>
    </button>}
  </div>
}
