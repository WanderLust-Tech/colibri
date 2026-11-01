import { useEffect, useState } from "react"
import { Clock, Copy, Dices, Download, Footprints, Languages, Mic, RefreshCw, RotateCcw, X } from "lucide-react"

import type { GeneratedMusic, MusicLimits } from "@/lib/api"
import { duration as formatSeconds, elapsedClock } from "@/lib/images"
import {
  MUSIC_LANGUAGES, clampDuration, clampMusicSteps, lyricBytes, musicProgressFraction, songClock, validDuration,
  type MusicRun,
} from "@/lib/music"
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
    : progress.stage === "text" || progress.stage === "condition" || progress.stage === "decode" ? t(`music.stage.${progress.stage}`)
    : t("music.stage.working")
  return <div className="image-turn music-turn">
    <p className="music-request">{t("music.requested", { length: songClock(request.duration) })}</p>
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
  const breakdown = timings && [timings.text, timings.condition, timings.denoise, timings.decode].every((value) => typeof value === "number")
    ? t("music.timings", {
      text: formatSeconds(timings.text!, locale), condition: formatSeconds(timings.condition!, locale),
      denoise: formatSeconds(timings.denoise!, locale), decode: formatSeconds(timings.decode!, locale),
    })
    : undefined
  return <div className="image-turn music-turn">
    {/* the browser's own player: seeking, volume and keyboard control come with it */}
    <audio className="music-player" controls preload="metadata" src={song.url} aria-label={t("music.alt", { prompt: song.prompt || song.lyrics?.split("\n")[0] || "" })} />
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
      <small>{songClock(request.duration)} · {t("image.meta.steps", { n: request.steps })}</small>
      <button type="button" className="image-action" onClick={onRetry} disabled={busy}><RotateCcw /><span>{t("image.retry")}</span></button>
    </div>
  </div>
}

/* Length, steps, seed, language and the lyrics switch, in the composer where the image
   controls and the chat's reasoning chip sit. A typed value counts as soon as it is valid, so
   sending straight from a field uses what is on screen; leaving the field puts an invalid one
   inside the range. */
export function MusicControls({ limits, duration, steps, seed, seedValid, language, lyricsOpen, lyricsFilled, onDuration, onSteps, onSeed, onLanguage, onLyrics }: {
  limits: MusicLimits
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
  return <div className="image-controls">
    <button type="button" className={cn("image-chip music-lyrics-chip", (lyricsOpen || lyricsFilled) && "on")} aria-pressed={lyricsOpen}
      title={t("music.lyrics")} onClick={onLyrics}>
      <Mic aria-hidden="true" /><span>{t("music.lyrics")}</span>
    </button>
    <label className={cn("image-chip", !durationOk && "invalid")} title={t("music.durationHelp", { min: limits.minDuration, max: limits.maxDuration })}>
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
    </label>
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
