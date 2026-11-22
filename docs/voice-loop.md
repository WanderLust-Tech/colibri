# The voice loop: talk to a model and hear it answer

Speech in, a chat model's answer, speech out, through the models of one running server (`coli router`, or any `coli serve` that has them): Qwen3-ASR
writes down what you say, a chat model answers, Qwen3-TTS speaks the answer. Nothing here is a new model or a new engine: it is three routes the
server already has, chained, with the speech started as soon as the first sentence of the answer exists.

```
microphone -> POST /v1/audio/transcriptions   (qwen3-asr-0.6b: 0.25-0.3 s for a sentence, on the card)
           -> POST /v1/chat/completions       (any chat model, streamed; a Gemma E2B/E4B answers at about 100 tokens a second)
           -> POST /v1/audio/speech           (qwen3-tts: stream_format "sse", a request for each sentence as it is written)
           -> loudspeaker
```

Measured on the RTX 5090 with the three models loaded (Gemma 4 E2B, Qwen3-ASR 0.6B, Qwen3-TTS 0.6B), the first spoken sound comes **about one second
after you stop talking** (ASR 0.3 s, the first token 0.6 s, the text done 0.85 s, the first audio 1.04 s), and a two-sentence answer has finished
playing out of the server in about 4 s (10 s of speech made in 3 s). The first turn after a start also loads the three models: 5 to 14 s once.

**What the server has to do:** hold all three at once. The router keeps three models loaded by default (`max_loaded`), which is exactly this; with a
fourth in use the least recently used goes, and the next turn pays its load. The router computes one model at a time unless started with parallel
models, so the speech starts when the text is done (the 0.2 s above); with parallel models the sentences are spoken while the chat model writes the
next. Install what is missing with `coli model install qwen3-asr-0.6b qwen3-tts-0.6b-customvoice gemma-4-e2b`.

## `coli talk`

```sh
coli talk                                   # the microphone: speak, pause, hear the answer, speak again; Ctrl-C ends it
coli talk --voice vivian --language en      # another speaker
coli talk --voice jfk                       # a saved cloned voice (coli voice add jfk clip.wav): the Base model is picked for you
coli talk --system "You are a pirate."      # what the assistant is told it is
coli talk question.wav --out replies/       # no microphone: one turn from the file, the answer saved as replies/reply-1.wav
coli talk --text "Tell me a short joke."    # typed words, spoken answer
coli talk --list-mics                       # the capture devices ffmpeg can read (Windows)
```

It needs `ffmpeg` and `ffplay` (the microphone and the loudspeaker are reached through them; no Python audio library) and numpy (the detector's
arithmetic). The microphone is read as 16 kHz mono; a voice-activity detector learns the noise of the room in the first 0.4 s, calls a run of 60 ms
above three times it (and above `--threshold`, default 0.012) speech, and ends the utterance after `--silence` seconds (default 0.8) below it, with 0.3 s
of lead-in kept; utterances under 0.4 s are ignored and one never runs past 30 s. It is half-duplex: while the answer plays the microphone is not
listened to (what it heard meanwhile is dropped), so the speakers do not trigger it; there is no interrupting an answer by talking over it.
Each turn prints what was heard, the answer as it is written, and the timings (`heard`, `first text`, `first audio`, `done`).

The chat model is the one named with `--model`, else a Gemma E2B or E4B if the server lists one (they answer fastest), else the first chat model. The
answers are asked to be short spoken sentences (the default system prompt), and the text is cleaned for the voice (markdown, code blocks, links'
addresses and emoji are not spoken) and cut into sentences, the first of them from 14 characters so the voice starts early. The last 16 messages are
kept as the conversation's memory.

## In the dashboard

With a speech recognition model and a speech model on the server, the chat composer has three buttons: the **microphone** (speak; it sends what you
said when you pause, or when you press it again), **read replies aloud** (each finished reply is spoken, with the speech model's own voice, or with a
saved voice chosen on **Settings → Voices → Read replies in this voice**), and **hands-free** (after each reply it listens again: a conversation with no
keys; it stops when nothing is said for 12 s or when you press Stop). The browser records with `MediaRecorder` (the server reads the webm through ffmpeg)
and plays the speech's server-sent events through the Web Audio API as they arrive. **A browser allows the microphone only on `localhost` or https:**
open the dashboard as `http://localhost:8000`, not by the PC's network address (a page on another machine needs https in front of the server).

## The same three calls from anywhere

```sh
curl http://127.0.0.1:8000/v1/audio/transcriptions -H "Authorization: Bearer $KEY" -F model=qwen3-asr-0.6b -F file=@question.wav       # {"text": ...}
curl -N http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"model": "gemma-4-e2b-colibri", "stream": true, "messages": [...]}'
curl -N http://127.0.0.1:8000/v1/audio/speech -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"model": "qwen3-tts-0.6b", "input": "The first sentence.", "stream_format": "sse"}'
```

`c/voice_loop.py` is this as a Python module (stdlib only): `VoiceLoop(server, chat, asr, tts).turn(audio=..., emit=...)` calls `emit("heard" | "text" |
"audio" | "reply" | "timing", value)` as things happen, raises `VoiceError(step, message)` when a step fails (the text of the answer still goes on when only
the voice fails), and keeps the history. `coli talk` is that plus `c/talk_cli.py`.

## Tests

`tests/test_voice_loop.py` (31): the text rules (markdown, code, links, emoji; sentences cut as they complete, "Dr. Smith" and "3.5" kept whole), the
detector on synthetic audio (an utterance with its lead-in and tail, two in a row, a click, a pause inside a sentence, a noisy room, the length
limit), whole turns against a fake server that speaks the three routes (the order of the events, the system prompt and history reaching the chat
model, a request a sentence, the key on every call, each step failing by itself and being named, a voice that fails after the first sentence), and
`coli talk` end to end (models picked, a file turn saved as a WAV, typed words, a saved voice). `web/src/lib/voice.test.ts` (16): the text rules, the
model choice, the event parser, the samples, the silence detector, and the two calls against a mocked `fetch`. The page's recording and playing were
run once in headless Edge with a fake microphone (the JFK clip) against a fake server: speak, listen, send, transcript, state by state.

## Not done

Interrupting an answer by speaking (it needs the microphone open during playback and an echo canceller; the browser has one, `coli talk` does not), the
sentence-by-sentence speech overlapping the chat model on a router started without parallel models, a Realtime-API-shaped WebSocket endpoint (the three
routes above are what a client chains), and a language switch by what was heard (the recogniser detects the language; the speech model is told
`--language` or decides from the text).
