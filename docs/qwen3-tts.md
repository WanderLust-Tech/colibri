# Text to speech: Qwen3-TTS

`c/qwen3tts` turns text into speech with [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
(Apache-2.0): ten languages, nine named speakers. It answers OpenAI's `POST /v1/audio/speech`, so the clients written for it work with a
model name and a base URL changed, and `coli speak "text"` does the same from the terminal. Plan and background: [qwen3-speech.md](qwen3-speech.md).
The other direction, speech to text, is [qwen3-asr.md](qwen3-asr.md).

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
- **Refused, with the field named:** `speed` other than 1 (the model has no control over the pace), `instructions` (this model speaks as a named
  speaker and takes no style description; the 1.7B CustomVoice does and is not built here yet), `stream_format` `sse` (the audio comes in one piece).
- **The answer** is the audio itself, with `x-colibri-audio-seconds`, `x-colibri-sample-rate` and `x-colibri-queue-wait-ms` headers, and
  `x-colibri-truncated: true` when the text was longer than one piece of speech could hold (4096 frames, five and a half minutes).
- `GET /v1/models` marks the model `"speech"` with a `speech` block (`speakers`, `languages`, `formats`, `voices`, `sample_rate`,
  `max_input_chars`, `device`); `/health` carries the engine's own READY line. The chat routes, pictures, music, embeddings, reranking and
  transcription answer 400 with a pointer to the speech route.

## How it works

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

Teacher forcing (the reference's codes drive the generation; at each choice the engine says whether the code is its own argmax) tests the whole
stack without a floating-point tie able to fail it. `tests/test_qwen3tts_real.py` runs these against `tools/qwen3_tts_0.6b_reference.json`
(the prompt's ids, the 63 frames of codes, the loudness of each frame) and the real engine; the gateway's rules run against a stub engine
(`tools/tts_stub.py`, `tests/test_tts_gateway.py`, 45 tests, no weights needed).

## Speed and memory (24-core i9, CPU)

Loading takes 1.6 s and 2.3 GB of RAM (the talker and the predictor as bf16, the codec as f32). A frame costs 130 to 250 ms for the talker and
the predictor together (the first dozen frames run at about half the cost of the rest, on one thread or many; not explained), and the codec 0.3 s for each second of speech: **5 s of speech in about 17 s, about three and a half times slower than real
time**. The decode loop uses a smaller thread team than the prefill and the codec (the physical cores, at most six: `--decode-threads N` or
`QT_DECODE_THREADS`; at 32 threads the same loop takes 346 ms a frame, at six about 250): the steps are one row against each weight, and a big
team gains nothing from them. Real time needs about 80 ms a frame; the way there is the Vulkan path the other engines have (open, see
[qwen3-speech.md](qwen3-speech.md)).

## The engine on its own

```sh
make qwen3tts
qwen3tts --model DIR --text "Hello." --speaker ryan --language english --out hello.wav [--greedy] [--seed N] [--max-frames N]
qwen3tts --model DIR --text "..." --ref REFDIR [--teacher]          # compare with tools/qwen3_tts_ref.py's tensors
qwen3tts --codec DIR/speech_tokenizer --decode-codes codes.i32 --out x.wav     # the codec alone
qwen3tts --model DIR --serve                                        # what the gateway runs
```

The serve protocol (lines of a keyword and a JSON object, one request at a time):

```
READY {"model", "task": "speech", "kind": "custom_voice", "sample_rate": 24000, "layers", "speakers": [...], "languages": [...], "device": "cpu"}
SPEAK {"id", "text", "speaker", "language", "out": path of the WAV to write, "seed", "greedy", "max_frames", "temperature"}
AUDIO {"id", "path", "samples", "sample_rate", "frames", "audio_seconds", "seconds", "truncated"}
ERROR {"id", "message"}                                             # the engine goes on
```

## Not built yet

The 1.7B CustomVoice (style `instructions`), VoiceDesign (a voice described in words), the Base models (a voice cloned from a few seconds of
audio: needs the codec's encoder and a speaker encoder), streaming, and the Vulkan path. See [qwen3-speech.md](qwen3-speech.md).
