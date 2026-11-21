# Text to speech: Qwen3-TTS

`c/qwen3tts` turns text into speech with [Qwen3-TTS-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) (Apache-2.0): ten
languages, in five models:

| Model | `coli model install` | Voice | Style `instructions` | Download | RAM |
|---|---|---|---|---|---|
| Qwen3-TTS-0.6B-CustomVoice | `qwen3-tts-0.6b-customvoice` (API id `qwen3-tts-0.6b`) | nine named speakers | no | 2.5 GB | 4 / 6 GB |
| Qwen3-TTS-1.7B-CustomVoice | `qwen3-tts-1.7b-customvoice` (`qwen3-tts-1.7b`) | the same nine speakers, in a wider model | yes: how to say it ("speak slowly and sadly") | 4.5 GB | 7 / 10 GB |
| Qwen3-TTS-1.7B-VoiceDesign | `qwen3-tts-1.7b-voicedesign` (`qwen3-tts-1.7b-voicedesign`) | none named: **the voice is described in words** ("a warm, deep male voice") | yes: it is the voice | 4.5 GB | 7 / 10 GB |
| Qwen3-TTS-0.6B-Base | `qwen3-tts-0.6b-base` (`qwen3-tts-0.6b-base`) | **cloned from a few seconds of reference audio** ([below](#voice-cloning-the-base-models)) | no | 2.4 GB | 4 / 6 GB |
| Qwen3-TTS-1.7B-Base | `qwen3-tts-1.7b-base` (`qwen3-tts-1.7b-base`) | the same, in the wider model | no | 4.5 GB | 7 / 10 GB |

It answers OpenAI's `POST /v1/audio/speech`, so the clients written for it work with a model name and a base URL changed, and
`coli speak "text"` does the same from the terminal. Plan and background: [qwen3-speech.md](qwen3-speech.md). The other direction, speech to
text, is [qwen3-asr.md](qwen3-asr.md).

```sh
coli model install qwen3-tts-0.6b-customvoice       # 2.5 GB
coli speak "Hello, this is a test of the speech engine."            # speech.wav, in Ryan's voice
coli speak "Guten Morgen." --voice vivian --language de --out morgen.mp3
echo "read this aloud" | coli speak - --out aloud.wav
curl http://127.0.0.1:8000/v1/audio/speech -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"model": "qwen3-tts-0.6b", "input": "Hello there.", "voice": "alloy"}' -o hello.wav
```

## What a request takes and gives

- **`input`:** the text, up to 4096 characters (OpenAI's limit).
- **`voice`:** one of the model's speakers, case-insensitive: `serena vivian uncle_fu ryan aiden ono_anna sohee eric dylan` (`GET /v1/models`
  lists them). Left out, it is `ryan`. OpenAI's names work too and stand for speakers, so a client that sends `alloy` is answered:
  `alloy`=ryan, `echo`=aiden, `fable`=dylan, `onyx`=uncle_fu, `nova`=vivian, `shimmer`=serena, `ash`=eric, `coral`=ono_anna, `sage`=sohee.
  Anything else is a 400 that lists what there is. `eric` and `dylan` are dialect speakers (Sichuan and Beijing); Chinese text, or `auto`, is
  spoken in their dialect.
- **`language`** (an extension): a name or an ISO code: `zh en de it pt es ja ko fr ru` (also `beijing_dialect`, `sichuan_dialect`), or `auto`,
  the default, which lets the model decide from the text.
- **`response_format`:** `wav` (the default; OpenAI's is `mp3`, but a WAV needs nothing on the server to play), `pcm` (raw 16-bit little-endian
  mono at 24 kHz), and `mp3`, `opus`, `aac`, `flac` when the server has ffmpeg (a 400 before any work is done when it has none).
- **`seed`** (an extension): the same text, voice and seed give the same speech; without one each request differs a little, as the model samples
  (top-k 50, temperature 0.9, repetition penalty 1.05, the reference's defaults).
- **`instructions`** (OpenAI's field, for the two 1.7B models; `coli speak --instruct`): on the **1.7B CustomVoice** a style that steers the
  delivery of the named speaker ("Speak slowly, in a calm and sad voice."); on **VoiceDesign** the voice itself ("A warm, deep male voice,
  speaking slowly and calmly."), and `voice` then names no one (it is accepted and ignored, so a client that always sends one still works).
  The words go to the model as a user turn before the text (`<|im_start|>user\n{instructions}<|im_end|>`). The 0.6B refuses it.
- **`reference_audio`, `reference_text`, `x_vector_only`, and a saved `voice`** (extensions, for the Base models only): see
  [Voice cloning](#voice-cloning-the-base-models). On a Base model `voice` names a saved voice and there is no default speaker.
- **Refused, with the field named:** `speed` other than 1 (the model has no control over the pace), `instructions` on the 0.6B (it speaks as a
  named speaker and takes no style description), `stream_format` `sse` (the audio comes in one piece).
- **The answer** is the audio itself, with `x-colibri-audio-seconds`, `x-colibri-sample-rate` and `x-colibri-queue-wait-ms` headers, and
  `x-colibri-truncated: true` when the text was longer than one piece of speech could hold (4096 frames, five and a half minutes).
- `GET /v1/models` marks the model `"speech"` with a `speech` block (`speakers` (empty for VoiceDesign), `languages`, `formats`, `voices`,
  `sample_rate`, `max_input_chars`, `device`, `kind` (`custom_voice`, `voice_design` or `base`), `instructions` (true when it takes them), `clone` (true for
  a Base model, which also lists its `saved_voices`)); `/health` carries the engine's own READY line. The chat routes, pictures, music, embeddings, reranking and
  transcription answer 400 with a pointer to the speech route.

## How it works (the speakers' models)

```
text -> <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n -> Qwen tokenizer
     -> the prompt rows: role, [language, speaker, codec markers] + tts_pad, the text through text_embedding + a two-layer projection, tts_eos
     -> the talker (a 28-layer Qwen3 decoder over 1024) reads them, then writes the first of 16 codes for each 80 ms frame (12.5 frames a second)
     -> the code predictor (5 layers) writes the other 15 codes of the frame from the talker's state, one at a time
     -> the next frame's input is the sum of that frame's 16 code embeddings + tts_pad; the end token stops it
     -> the codec's decoder: the 16 codes' embeddings summed, an 8-layer sliding-window transformer, two upsampling stages and a
        convolutional vocoder (SnakeBeta, residual units): 1920 samples of 24 kHz for every frame
```

The codec runs in pieces of 300 frames with 25 frames of context on each side, so a long speech does not need a long tensor.

## Checked against the reference

Held to the official `qwen-tts` package (transformers, fp32, CPU, greedy) with `tools/qwen3_tts_ref.py`, on "Hello, this is a test of the
speech engine." in Ryan's voice (63 frames, 5.04 s):

| Stage | Result |
|---|---|
| prompt token ids | identical (19) |
| talker prefill input | 1.2e-7 (max abs, on values of about 2e-2) |
| first logits | 2.0e-5 (on values of about 4.6) |
| the 1008 choices of 63 frames, teacher forced | 0 differ from the engine's own argmax |
| the codes, free-running greedy | identical (63 frames) |
| the waveform from the same codes | 1.4e-6 max abs (full scale 1.0) |
| through the gateway, then Qwen3-ASR | the transcription is the input text, word for word |

The **1.7B models** are held to the same reference (`tools/qwen3_tts_1.7b_reference.json`, made with `tools/qwen3_tts_ref.py --instruct`), each
with an instruction (CustomVoice: "Speak slowly, in a calm and sad voice." in Ryan's voice; VoiceDesign: "A warm, deep male voice, speaking
slowly and calmly."), on the CPU and on the card:

| Stage | 1.7B CustomVoice (47 frames) | 1.7B VoiceDesign (40 frames) |
|---|---|---|
| prompt token ids | identical | identical |
| talker prefill input, instruction rows included | 1.2e-7 | 1.2e-7 |
| first logits | 2.5e-5 | 7.6e-6 |
| the choices, teacher forced | 0 of 752 differ | 0 of 640 differ |
| the codes, free-running greedy | identical | identical |
| the waveform from the same codes | 1.8e-6 | 3.4e-6 |

The 1.7B's talker is 2048 wide and its code predictor stays 1024, so the predictor's input rows go through a `small_to_mtp_projection` (with a bias)
first; that, the instruction rows and the missing speaker row are all that differ from the 0.6B. `tests/test_qwen3tts_17_real.py` runs these
(`QWEN3TTS_MODEL_17`, `QWEN3TTS_MODEL_DESIGN`; 14 tests on each of the CPU and the card).

Teacher forcing (the reference's codes drive the generation; at each choice the engine says whether the code is its own argmax) tests the whole
stack without a floating-point tie able to fail it. `tests/test_qwen3tts_real.py` runs these against `tools/qwen3_tts_0.6b_reference.json`
(the prompt's ids, the 63 frames of codes, the loudness of each frame) and the real engine; the gateway's rules run against a stub engine
(`tools/tts_stub.py`, `tests/test_tts_gateway.py`, 45 tests, no weights needed).

## Speed and memory (24-core i9, RTX 5090)

Loading takes 1.6 s and 2.3 GB of RAM (the talker and the predictor as bf16, the codec as f32); the 1.7B models take 2.6 s and 4.3 GB.
**The 1.7B on the card: 21 to 27 ms a frame** (CPU 139 ms), 5 s of speech in about 1.8 s; the 0.6B figures follow. Greedy decoding (the engine's
`--greedy`, a test tool; the gateway always samples) can fail to emit the end token for some texts and then runs to the frame limit; sampled
speech ends normally.

**On the card** (a `VK=1` build with `COLI_VULKAN=1`, which the router sets when it puts the model on the card): a frame, the talker's step and
the code predictor's sixteen, costs **14 to 19 ms** (the CPU: 111 ms), and the codec's vocoder
(the convolutional stack that turns the frames into samples, nine tenths of the codec) on the card too: 0.17 s for 5 s of speech instead of
1.05 s. 5 s of speech takes about 1.25 s (4 times faster than real time); 56 s of speech (680 characters) took 11 s (it took 21 s with the codec
on the CPU). Each step is one
submission of the dense chain (the same one the embedding and Gemma engines use: norms, q/k/v, head norms, RoPE, attention over a KV cache that
lives on the card, o, SwiGLU, down), the matrices held as bf16 and computed in f32, and only the logits and the hidden row come back. The prompt's
rows go through the same chain. The speech is the CPU's: the codes are the same, and the waveform differs from the CPU's by one step of the
16-bit sample on a few thousand of its samples (the card's sums and its sine round differently); the real-engine tests (the codes teacher forced
and free running, the loudness of each frame) pass on the card. Without a card, or when a step fails (the whole request is then
run again on the CPU), the CPU path runs. About 1 GB of the card for the matrices and, while a request runs, 0.25 MB for each position of the speech.
`READY` says `"device":"vulkan"` or `"cpu"`.

The vocoder on the card: each causal convolution is an im2col (`chain_codec`, a shader beside the dense chain's) and a GEMM over the resident
f32 weights, in tiles; SnakeBeta, the transposed convolution's overlap-add, the bias and the last one-channel layer (with the clamp) are
`chain_codec` modes; a chunk of 300 frames is one submission. The quantizer, the pre-conv, the codec's transformer and the first two upsamplings
(about 0.1 s for 5 s of speech) stay on the CPU.

**On the CPU**: a frame costs 111 ms on a quiet machine (the predictor 81 of them: sixteen steps that each read the same five layers; the talker
24) and up to 250 ms when the machine is busy, and the codec 0.2 s a second of speech: **5 s of speech in about 8 s, 1.6 times slower than real
time**. The decode loop uses a smaller thread team than the prefill and the codec (the physical cores, at most six: `--decode-threads N` or
`QT_DECODE_THREADS`; at 32 threads the same loop takes 346 ms a frame): the steps are one row against each weight, and a big team gains nothing
from them. Pinning the team to the performance cores changed nothing.

## The engine on its own

```sh
make qwen3tts
qwen3tts --model DIR --text "Hello." --speaker ryan --language english --out hello.wav [--instruct STYLE] [--greedy] [--seed N] [--max-frames N]
qwen3tts --model DIR --text "..." --ref REFDIR [--teacher]          # compare with tools/qwen3_tts_ref.py's tensors
qwen3tts --codec DIR/speech_tokenizer --decode-codes codes.i32 --out x.wav     # the codec alone
qwen3tts --model DIR --serve                                        # what the gateway runs
```

The serve protocol (lines of a keyword and a JSON object, one request at a time):

```
READY {"model", "task": "speech", "kind": "custom_voice" | "voice_design", "instruct": bool, "sample_rate": 24000, "layers", "speakers": [...], "languages": [...], "device": "cpu" | "vulkan"}
SPEAK {"id", "text", "speaker", "instruct", "language", "out": path of the WAV to write, "seed", "greedy", "max_frames", "temperature"}
AUDIO {"id", "path", "samples", "sample_rate", "frames", "audio_seconds", "seconds", "truncated"}
ERROR {"id", "message"}                                             # the engine goes on
```

## Voice cloning (the Base models)

The two **Base** checkpoints speak in a voice they are given: a few seconds of reference speech, and ideally what it says. They have no named
speakers. Two ways to give the voice:

- **A saved voice**: `coli voice add ana clip.wav --text "what the clip says"` (without `--text` the server's speech recognition model writes the
  words down; `--no-text` skips that), then `voice` names it: `coli speak "Hello." --voice ana`, or `{"voice": "ana"}`. Voices live in
  `~/colibri-voices/<name>/` (`ref.wav` at 24 kHz mono and `voice.json`; `COLI_VOICES` moves them). `coli voice list`, `coli voice remove ana`; over
  HTTP `GET /v1/audio/voices`, `POST /v1/audio/voices` (`{"name", "audio" (base64), "text"}`) and `POST /v1/audio/voices/delete` on the router.
- **A clip with the request**: `reference_audio` (the base64 of an audio file, or a `data:` URL: WAV, MP3, and with ffmpeg more; 1 to 60 seconds,
  3 to 30 is best; the body limit is 4 MB) and `reference_text`; `coli speak "Hello." --reference clip.wav --reference-text "what it says"`.

With the reference's text the voice is cloned **in context**: the model reads the clip's transcript and its codes in front of the new text and
carries on from them (the closer likeness: the clip's delivery, pace and room come with it). Without the text only the clip's **speaker embedding**
is used (`x_vector_only`, which also forces this mode: a looser likeness, and no cost for the codes). A request that names neither a saved voice
nor a clip is a 400 that says what is saved. `instructions` and the named speakers do not apply to a Base model; a model that is not a Base one
refuses `reference_audio`.

```sh
coli model install qwen3-tts-0.6b-base                             # 2.4 GB
coli voice add jfk jfk.wav                                          # the words are written down by qwen3-asr, if it is installed
coli speak "The quick brown fox jumps over the lazy dog." --voice jfk --out fox.wav
coli speak "Hello there." --reference me.wav --reference-text "the words I said in me.wav"
```

How it works, as `Qwen3TTSModel.generate_voice_clone` does it (`c/qwen3tts_clone.h`):

```
reference audio (24 kHz mono)
  -> log-mel (n_fft 1024, hop 256, 128 mels to 12 kHz, Slaney) -> the speaker encoder (ECAPA-TDNN: a TDNN, three SE-Res2Net blocks, a feature
     aggregation, attentive statistics pooling, a linear) -> one vector the width of the talker, put where a named speaker's codec row goes
  -> (in context) the codec's encoder, which is Kyutai's Mimi: a SEANet convolutional stack (strides 4, 5, 6, 8), an 8-layer transformer,
     a stride-2 downsampling to 12.5 Hz, a split residual quantizer (one semantic, fifteen acoustic codebooks) -> 16 codes a frame
prompt = role, codec prefix with the speaker vector, then (in context) [reference text + text + tts_eos] summed with [codec_bos + the reference's frames],
         the text that does not fit riding along one row to each generation step; x-vector only: the first text token, the rest riding along
speech = the generated frames decoded together with the reference's, the reference's share of the samples cut off
```

The speaker embedding and the codes of a clip are made once and kept (the last four clips), so a request that repeats a clip pays for them once:
a clip of 11 s takes 0.1 s for the embedding and 0.7 s for the codes (CPU; the codec encoder is not on the card).

**Checked against the package** (fp32, greedy, the 11 s JFK clip; `tools/qwen3_tts_clone_ref.py`, `tests/test_qwen3tts_clone_real.py`): the log-mel
features 2e-3 (the log of small energies), the speaker embedding 7e-7, the codec encoder's convolutions 4e-4 and transformer 5e-4, the 138 reference
frames' 2208 codes identical (0 differ), the prompt ids and rows 7e-7, every one of the 1424 codes of the package's speech the engine's own argmax,
the generated codes identical and the waveform 6e-6 away, in both modes, on the CPU and on the card (16 ms a frame; the talker and the predictor run on
the chain as for the other models). Two findings: the Mimi config says `sliding_window` 250, but the transformers code the package runs applies a plain
causal mask and so does this (a clip over 10 s differs otherwise); and greedy decoding of an in-context request can fail to end (the 89-frame
reference is the 90-token cap), where sampling, which the gateway always does, ends normally: 55 frames for the fox sentence, read back by Qwen3-ASR
as exactly that sentence.

## Not built yet

Streaming; the codec's first half on the card (the rest of it takes 0.02 s a second of speech) and the codec's *encoder* (cloning, 0.7 s for 11 s of clip
on the CPU). See [qwen3-speech.md](qwen3-speech.md).
