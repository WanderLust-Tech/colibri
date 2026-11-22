import { useCallback, useEffect, useRef, useState } from "react"

import { SilenceDetector, spokenText, streamSpeech, transcribeRecording } from "./voice"

export type VoiceState = "idle" | "listening" | "transcribing" | "speaking"

interface VoiceOptions {
  baseUrl: string
  apiKey: string
  /* The models: without both the page offers no voice. */
  asrModel?: string
  ttsModel?: string
  /* A speaker, or a saved voice (with a cloning model). */
  voice?: string
  /* Called with the words of what was said, once the recogniser has written them down. */
  onTranscript: (text: string) => void
  /* Called when listening ended with nothing said (silence, or nothing heard). */
  onNothing?: () => void
  onError: (message: string) => void
}

/* The microphone and the loudspeaker of the page: `listen()` records until the speaker pauses (or `cancel()`), has the server's speech recognition model
   write it down and calls `onTranscript`; `speak(text)` plays the speech model's voice as the server makes it. One thing at a time: listening stops
   the voice, and the microphone is shut while the voice speaks, so the page does not hear itself. */
export function useVoice({ baseUrl, apiKey, asrModel, ttsModel, voice, onTranscript, onNothing, onError }: VoiceOptions) {
  const [state, setState] = useState<VoiceState>("idle")
  const [level, setLevel] = useState(0)
  const recorder = useRef<{ stop: () => void; abandon: () => void } | null>(null)
  const playback = useRef<{ context: AudioContext; sources: Set<AudioBufferSourceNode>; abort: AbortController; cancelled: boolean } | null>(null)
  const callbacks = useRef({ onTranscript, onNothing, onError })
  callbacks.current = { onTranscript, onNothing, onError }
  const available = Boolean(asrModel && ttsModel)

  const stopSpeaking = useCallback(() => {
    const current = playback.current
    if (!current) return
    current.cancelled = true
    current.abort.abort()
    for (const source of current.sources) { try { source.stop() } catch { /* already finished */ } }
    void current.context.close().catch(() => undefined)
    playback.current = null
    setState((now) => now === "speaking" ? "idle" : now)
  }, [])

  const cancel = useCallback(() => {
    recorder.current?.abandon()
    recorder.current = null
    stopSpeaking()
    setLevel(0)
    setState("idle")
  }, [stopSpeaking])

  const listen = useCallback(async () => {
    if (!asrModel || recorder.current) return
    stopSpeaking()
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })
    } catch (cause) {
      callbacks.current.onError(cause instanceof DOMException && cause.name === "NotAllowedError" ? "voice.denied" : "voice.noMicrophone")
      return
    }
    const chunks: Blob[] = []
    const media = new MediaRecorder(stream)
    const context = new AudioContext()
    const analyser = context.createAnalyser()
    analyser.fftSize = 1024
    context.createMediaStreamSource(stream).connect(analyser)
    const samples = new Float32Array(analyser.fftSize)
    const detector = new SilenceDetector()
    const started = performance.now()
    let spoke = false
    let abandoned = false
    let finished = false
    const release = () => {
      window.clearInterval(timer)
      stream.getTracks().forEach((track) => track.stop())
      void context.close().catch(() => undefined)
      recorder.current = null
      setLevel(0)
    }
    const stop = () => { if (media.state !== "inactive") media.stop() }
    const timer = window.setInterval(() => {
      analyser.getFloatTimeDomainData(samples)
      let sum = 0
      for (const value of samples) sum += value * value
      const rms = Math.sqrt(sum / samples.length)
      setLevel(rms)
      const now = performance.now()
      const outcome = detector.push(rms, now)
      if (outcome === "speech") spoke = true
      if ((outcome === "end" && spoke) || now - started > 30000 || (!spoke && now - started > 12000)) stop()
    }, 50)
    media.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data) }
    media.onstop = async () => {
      if (finished) return
      finished = true
      release()
      if (abandoned) return
      if (!spoke) { setState("idle"); callbacks.current.onNothing?.(); return }
      setState("transcribing")
      try {
        const text = await transcribeRecording(baseUrl, apiKey, asrModel, new Blob(chunks, { type: media.mimeType || "audio/webm" }))
        setState("idle")
        if (text) callbacks.current.onTranscript(text)
        else callbacks.current.onNothing?.()
      } catch (cause) {
        setState("idle")
        callbacks.current.onError(cause instanceof Error ? cause.message : String(cause))
      }
    }
    recorder.current = { stop, abandon: () => { abandoned = true; stop(); if (media.state === "inactive" && !finished) { finished = true; release() } } }
    media.start(250)
    setState("listening")
  }, [apiKey, asrModel, baseUrl, stopSpeaking])

  /* Ends the recording now, as if the speaker had paused: what was said so far is written down and handed over. */
  const finish = useCallback(() => { recorder.current?.stop() }, [])

  const speak = useCallback(async (text: string) => {
    const words = spokenText(text)
    if (!ttsModel || !words) return
    recorder.current?.abandon()
    stopSpeaking()
    const context = new AudioContext({ sampleRate: 24000 })
    const current = { context, sources: new Set<AudioBufferSourceNode>(), abort: new AbortController(), cancelled: false }
    playback.current = current
    setState("speaking")
    let next = 0
    let last: AudioBufferSourceNode | null = null
    try {
      await streamSpeech(baseUrl, apiKey, ttsModel, words, voice, (samples) => {
        if (current.cancelled || !samples.length) return
        const buffer = context.createBuffer(1, samples.length, 24000)
        buffer.copyToChannel(new Float32Array(samples), 0)
        const source = context.createBufferSource()
        source.buffer = buffer
        source.connect(context.destination)
        next = Math.max(next, context.currentTime + 0.05)
        source.start(next)
        next += buffer.duration
        current.sources.add(source)
        source.onended = () => current.sources.delete(source)
        last = source
      }, current.abort.signal)
      if (!current.cancelled && last) {
        await new Promise<void>((resolve) => { (last as AudioBufferSourceNode).addEventListener("ended", () => resolve()); current.abort.signal.addEventListener("abort", () => resolve()) })
      }
    } catch (cause) {
      if (!current.cancelled) callbacks.current.onError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      if (playback.current === current) {
        playback.current = null
        void context.close().catch(() => undefined)
        setState("idle")
      }
    }
  }, [apiKey, baseUrl, stopSpeaking, ttsModel, voice])

  useEffect(() => () => { recorder.current?.abandon(); playback.current?.abort.abort(); void playback.current?.context.close().catch(() => undefined) }, [])

  return { state, level, available, listen, finish, speak, cancel, stopSpeaking }
}
