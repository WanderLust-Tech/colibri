# Speech to text: Qwen3-ASR

`c/qwen3asr` turns speech into text with [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (0.6B and 1.7B, Apache-2.0): 30 languages
and 22 Chinese dialects, speech and singing. It answers OpenAI's `POST /v1/audio/transcriptions`, so the clients written for Whisper work
unchanged, and `coli transcribe FILE` does the same from the terminal. Plan and background: [qwen3-speech.md](qwen3-speech.md).

```sh
coli model install qwen3-asr-0.6b            # 1.9 GB; qwen3-asr-1.7b (4.7 GB) hears a little better and takes about 1.8 times as long
coli transcribe speech.wav                    # prints the text
coli transcribe a.mp3 b.wav --language en --context "Kennedy, Berlin" --verbose
curl http://127.0.0.1:8000/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
     -F model=qwen3-asr-0.6b -F file=@speech.wav -F response_format=verbose_json
```

## What a request takes and gives

- **File:** WAV (any rate, bits, channels), MP3, and with ffmpeg on the server FLAC, Ogg, M4A and more; mixed to mono and resampled to
  16 kHz by the gateway. Up to 25 MB and 15 minutes a request. A JSON body with `file` as base64 works as well.
- **`language`** is an ISO-639 code and forces the language: the model then writes only the transcription. Without it the model detects the
  language and says which (`verbose_json` carries it, lower case as OpenAI writes it: `"english"`). The codes it knows: `zh en yue ar de fr
  es pt id it ko ru th vi ja tr hi ms nl sv da fi pl cs tl fa el ro hu mk`; any other is a 400 that lists them.
- **`prompt`** is context for the model (names, terms, what the recording is), up to 1000 characters; it goes into the system turn.
- **`response_format`:** `json` (the default, `{"text", "usage": {"type": "duration", "seconds"}}`), `text`, or `verbose_json` (adds `task`,
  `language`, `duration` and one segment over the whole clip). `srt`, `vtt` and word timestamps are refused: this model gives none (Qwen's
  forced aligner, a separate model, does; it is not built here). `temperature` is accepted and ignored: the model is decoded greedily.
- **Silence:** a clip with nothing in it answers an empty `text` without asking the model.
- **Long audio:** the engine takes up to 20 minutes in one piece (the reference's own limit); a longer clip is cut at the quietest 100 ms
  within 5 s of the limit, and the pieces' texts are joined with a space (the reference joins with nothing, which glues two English words
  together at the cut). The gateway's 15 minutes keeps a request within one piece.
- `GET /v1/models` marks the model `"transcription"` with `input_modalities: ["audio"]` and a `transcription` block (`languages`,
  `formats`, `sample_rate`, `piece_seconds`, `max_seconds`, `device`). The chat routes, pictures, music, embeddings and reranking answer 400
  with a pointer to the transcription route.

## How it works

```
16 kHz mono -> log-mel (128 bins, hop 160: 100 frames a second)
            -> audio encoder: 100-frame chunks -> three Conv2d(3x3, stride 2) + GELU -> 13 frames a chunk -> conv_out -> + sinusoidal position
               in the chunk -> 18 (0.6B) or 24 (1.7B) pre-LayerNorm layers, attention inside windows of 104 frames -> ln_post, proj1, GELU, proj2
            -> a Qwen3 decoder (28 layers; plain RoPE, GQA, per-head q/k RMSNorm, SwiGLU) reads
               <|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>[one slot per frame]<|audio_end|><|im_end|>\n<|im_start|>assistant\n
               and writes, greedily, `language English<asr_text>the text<|im_end|>`
            -> the language and the text are split; repetition loops are cut (the reference's `detect_and_fix_repetitions`)
```

Two details worth knowing, both found by comparing against the reference:

- **Windows.** The checkpoint's config says the encoder attends inside windows of 800 mel frames (8 s). The vLLM backend does that; the stock
  transformers path (`sdpa`) ignores the window boundaries and attends over the whole clip. The two agree for audio up to 8 s and differ
  slightly beyond. This engine attends inside the windows (the intended behaviour); `AS_FULL_ATTN=1` in its environment attends over
  everything, which reproduces the stock path to 7e-7.
- **The checkpoint's mixed rotary scheme** (interleaved MRoPE) is ordinary RoPE here: for text and audio all three position streams are the
  same number.

## Checked against the reference

Held to the official `qwen-asr` package (transformers, fp32, CPU) with `tools/qwen3_asr_ref.py`, on the 11 s JFK clip (public domain, from
whisper.cpp's samples), stage by stage:

| | 0.6B | 1.7B |
|---|---|---|
| log-mel features, largest difference | 1.4e-5 | 1.4e-5 |
| after the convolutions, largest difference | 4.8e-6 | 6.8e-6 |
| encoder output, largest difference (windowed reference) | 5.0e-6 | 4.7e-7 |
| encoder output with `AS_FULL_ATTN=1` against stock transformers | 6.6e-7 | |
| prompt token ids | identical (158) | identical (158) |
| generated token ids | identical (30) | identical (30) |

The answer is `And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.` (the 1.7B
punctuates `you. Ask`), the same as the reference's.

## Speed and memory

An i9-14900KF (24 cores), **CPU** only (no graphics-card path yet); the whole request, 11 s clip:

| | load | mel | encoder | decode | total | resident |
|---|---|---|---|---|---|---|
| Qwen3-ASR-0.6B | 1.3 s | 0.01 s | 0.5 s | 30 tokens in 3.75 s (125 ms a token) | 4.3 s | 1.9 GB |
| Qwen3-ASR-1.7B | 1.5 s | 0.01 s | 0.84 s | 30 tokens in 6.75 s (225 ms a token) | 7.6 s | 4.7 GB |

A 33 s clip in three pieces (a test limit of 15 s) took the 0.6B 1.6 s of encoder and 11.7 s of decoding for 90 tokens. The decoder is
most of the time, since each token reads the whole model once; the encoder is a fixed 46 ms a second of audio on the 0.6B.

## Tests

- `tests/test_asr_gateway.py` (32): the request rules, the engine wrapper, and the whole route over HTTP against `tools/asr_stub.py`: the
  upload decoded to 16 kHz mono whatever it was, the language and the prompt reaching the engine, the three formats, silence, the refusals,
  a dead engine, the model card and the refusals on the other routes, and the family and model-id resolution.
- `tests/test_speech_cli.py` (14): `coli transcribe` end to end against a real gateway.
- `tests/test_qwen3asr_real.py` (8, needs the model and the clip): the answer and token count of the stored transformers reference
  (`tools/qwen3_asr_0.6b_reference.json`), a forced language, a context, bad requests, silence, a clip in pieces joined with spaces; with
  `QWEN3ASR_REF` set, the stage-by-stage tensor comparison above.

## Not done

- A graphics-card path (the encoder and the decoder on the Vulkan chain, as `qwen3embed` has): decoding is the cost and it is a good fit.
- Word and segment timestamps (Qwen3-ForcedAligner-0.6B), streaming transcription, and a batch of requests at once.
- Audio that is not 16 kHz is the gateway's job; the engine's own `--wav` reads 16 kHz PCM16 or float WAV only.
