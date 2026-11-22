import { useCallback, useEffect, useRef, useState, type FormEvent } from "react"
import { AudioLines, LoaderCircle, Play, Plus, Trash2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { useLocale } from "@/i18n"
import { deleteVoice, listInstalledModels, listVoices, saveVoice, speakInVoice, type InstalledModel, type VoiceList } from "@/lib/api"
import { fileToBase64, voiceNameProblem } from "@/lib/voices"

interface VoicesPanelProps {
  baseUrl: string
  apiKey: string
  connected: boolean
  /* The saved voice the page reads its replies in ("" for the speech model's own), and how it is chosen. */
  replyVoice: string
  onReplyVoice: (name: string) => void
}

/* The saved voices of the voice-cloning speech models (Qwen3-TTS Base): a name, a few seconds of reference speech and what it says
   (docs/qwen3-tts.md). The router keeps them in ~/colibri-voices; `voice` in a speech request names one. Here they are made from a clip,
   tried out and removed. */
export default function VoicesPanel({ baseUrl, apiKey, connected, replyVoice, onReplyVoice }: VoicesPanelProps) {
  const { t } = useLocale()
  const [list, setList] = useState<VoiceList | null>(null)
  const [models, setModels] = useState<InstalledModel[]>([])
  const [loadError, setLoadError] = useState("")
  const [adding, setAdding] = useState(false)
  const [name, setName] = useState("")
  const [text, setText] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ kind: "error" | "ok"; text: string } | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [trying, setTrying] = useState<string | null>(null)
  const [phrase, setPhrase] = useState("")
  const [speaking, setSpeaking] = useState(false)
  const [audioUrl, setAudioUrl] = useState("")
  const audio = useRef<HTMLAudioElement | null>(null)

  const refresh = useCallback(async (signal?: AbortSignal) => {
    try {
      const [voices, installed] = await Promise.all([listVoices(baseUrl, apiKey, signal), listInstalledModels(baseUrl, apiKey, signal)])
      setList(voices)
      setModels(installed.models.filter((model) => model.modality === "speech" && /base/i.test(model.id)))
      setLoadError("")
    } catch (cause) {
      if (signal?.aborted) return
      setLoadError(cause instanceof Error ? cause.message : String(cause))
    }
  }, [baseUrl, apiKey])

  useEffect(() => {
    if (!connected) return
    const controller = new AbortController()
    void refresh(controller.signal)
    return () => controller.abort()
  }, [connected, refresh])

  useEffect(() => () => { if (audioUrl) URL.revokeObjectURL(audioUrl) }, [audioUrl])

  const problem = voiceNameProblem(name)

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!file) return
    setMessage(null)
    setSaving(true)
    try {
      const outcome = await saveVoice(baseUrl, apiKey, name.trim().toLowerCase(), await fileToBase64(file), text.trim())
      setMessage({ kind: "ok", text: t(outcome.status === "created" ? "voices.created" : "voices.replaced", { name: outcome.name, mode: t(outcome.mode === "in-context" ? "voices.modeContext" : "voices.modeVector") }) })
      setAdding(false); setName(""); setText(""); setFile(null)
      await refresh()
    } catch (cause) {
      setMessage({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    } finally {
      setSaving(false)
    }
  }

  const remove = async (voice: string) => {
    setMessage(null)
    setConfirmDelete(null)
    try {
      await deleteVoice(baseUrl, apiKey, voice)
      setMessage({ kind: "ok", text: t("voices.deleted", { name: voice }) })
      if (trying === voice) setTrying(null)
      if (replyVoice === voice) onReplyVoice("")
      await refresh()
    } catch (cause) {
      setMessage({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    }
  }

  const speak = async (voice: string) => {
    const model = models[0]
    if (!model || !phrase.trim()) return
    setMessage(null)
    setSpeaking(true)
    try {
      const blob = await speakInVoice(baseUrl, apiKey, model.id, voice, phrase.trim())
      setAudioUrl(URL.createObjectURL(blob))
      window.setTimeout(() => void audio.current?.play().catch(() => undefined), 50)
    } catch (cause) {
      setMessage({ kind: "error", text: cause instanceof Error ? cause.message : String(cause) })
    } finally {
      setSpeaking(false)
    }
  }

  return (
    <section className="side-section preset-panel voices-panel">
      <div className="section-title"><AudioLines className="size-3.5" /> {t("voices.title")}</div>
      <p className="field-help">{t("voices.intro")}</p>
      {loadError && <p className="field-help model-error" role="alert">{t("models.loadError", { message: loadError })}</p>}
      {list && !list.supported && <p className="field-help">{t("voices.unsupported")}</p>}
      {list?.supported && models.length === 0 && <p className="field-help">{t("voices.noModel")}</p>}
      {message && <p className={`field-help model-result ${message.kind === "error" ? "error" : ""}`} role={message.kind === "error" ? "alert" : "status"}>{message.text}</p>}
      {list?.supported && !adding && <>
        {list.voices.length === 0 && <p className="field-help">{t("voices.empty")}</p>}
        <ul className="model-list">
          {list.voices.map((voice) => (
            <li key={voice.name} className="model-row preset-row" data-voice={voice.name}>
              <div className="model-info">
                <strong>{voice.name}</strong>
                <small>{[`${voice.seconds.toFixed(1)} s`, t(voice.mode === "in-context" ? "voices.modeContext" : "voices.modeVector")].join(" · ")}</small>
                {voice.text && <small className="preset-system">{voice.text.length > 140 ? `${voice.text.slice(0, 139)}…` : voice.text}</small>}
                {trying === voice.name && (
                  <div className="voice-try">
                    <Input aria-label={t("voices.phrase")} value={phrase} placeholder={t("voices.phrasePlaceholder")} onChange={(event) => setPhrase(event.target.value)} />
                    <Button type="button" size="sm" disabled={speaking || !phrase.trim() || models.length === 0} onClick={() => void speak(voice.name)}>
                      {speaking ? <LoaderCircle className="size-3.5 animate-spin" /> : <Play className="size-3.5" />} {t("voices.speak")}
                    </Button>
                    {audioUrl && <audio ref={audio} controls src={audioUrl} />}
                  </div>
                )}
              </div>
              {confirmDelete === voice.name ? (
                <div className="model-confirm">
                  <span>{t("voices.confirmDelete", { name: voice.name })}</span>
                  <Button type="button" size="sm" onClick={() => void remove(voice.name)}>{t("voices.delete")}</Button>
                  <Button type="button" size="sm" variant="secondary" onClick={() => setConfirmDelete(null)}>{t("models.cancel")}</Button>
                </div>
              ) : (
                <div className="model-confirm">
                  <Button type="button" size="sm" variant="secondary" disabled={models.length === 0} onClick={() => setTrying(trying === voice.name ? null : voice.name)}><Play className="size-3.5" /> {t("voices.try")}</Button>
                  <Button type="button" size="sm" variant="secondary" aria-pressed={replyVoice === voice.name} disabled={models.length === 0}
                    onClick={() => onReplyVoice(replyVoice === voice.name ? "" : voice.name)}>{t(replyVoice === voice.name ? "voices.replying" : "voices.replyWith")}</Button>
                  <Button type="button" size="sm" variant="secondary" aria-label={t("voices.delete")} title={t("voices.delete")} onClick={() => setConfirmDelete(voice.name)}><Trash2 className="size-3.5" /></Button>
                </div>
              )}
            </li>
          ))}
        </ul>
        <Button type="button" variant="secondary" onClick={() => { setMessage(null); setAdding(true) }}><Plus className="size-4" /> {t("voices.new")}</Button>
      </>}
      {adding && (
        <form className="preset-form" onSubmit={(event) => void submit(event)}>
          <h3>{t("voices.creating")}</h3>
          <label>{t("voices.name")}
            <Input id="voice-name" value={name} placeholder="ana" autoComplete="off" onChange={(event) => setName(event.target.value)} />
            <span className="field-help">{problem === "invalid" ? t("voices.nameInvalid") : t("voices.nameHelp")}</span>
          </label>
          <label>{t("voices.clip")}
            <input id="voice-file" type="file" accept="audio/*" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            <span className="field-help">{t("voices.clipHelp")}</span>
          </label>
          <label>{t("voices.text")}
            <Textarea id="voice-text" rows={3} value={text} onChange={(event) => setText(event.target.value)} />
            <span className="field-help">{t("voices.textHelp")}</span>
          </label>
          <div className="preset-actions">
            <Button type="submit" disabled={saving || problem !== null || !file}>
              {saving && <LoaderCircle className="size-4 animate-spin" />} {t("voices.save")}
            </Button>
            <Button type="button" variant="secondary" disabled={saving} onClick={() => setAdding(false)}>{t("models.cancel")}</Button>
          </div>
        </form>
      )}
    </section>
  )
}
