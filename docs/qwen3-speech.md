# Qwen3 speech: Qwen3-ASR (speech to text) and Qwen3-TTS (text to speech)

Plan and notes, 2026-11-21. Two new engines for the fork, both Apache-2.0 checkpoints from Qwen. Nothing here is built yet: this page is
what was found out (from the checkpoints' `config.json` files and the reference packages' code, not from memory) and the order of work.
It is updated as stages land; the status table at the end is the truth.

Where the references and weights are on this machine (outside the repository):

| | |
|---|---|
| Reference environments | `C:\Users\kp-mi\colibri-refs\venv312` (torch 2.15 CPU, transformers 4.57.6, `qwen-asr`), `...\venv_tts` (transformers 4.57.3, `qwen-tts`): the two packages pin different transformers, so they cannot share one |
| Reference code to read | `venv312\Lib\site-packages\qwen_asr\core\transformers_backend\modeling_qwen3_asr.py`; `venv_tts\Lib\site-packages\qwen_tts\core\models\modeling_qwen3_tts.py` and `core\tokenizer_12hz\modeling_qwen3_tts_tokenizer_v2.py` |
| Checkpoints | `C:\Users\kp-mi\colibri-staging\qwen3-asr-0.6b`, `qwen3-tts-12hz-0.6b-customvoice`, `qwen3-tts-12hz-0.6b-base`, `qwen3-tts-tokenizer-12hz` (the 1.7B ones are not downloaded yet) |
| Test clip | `C:\Users\kp-mi\colibri-refs\clips\jfk.wav` (11 s, 16 kHz mono, public domain, from whisper.cpp's samples) |

## Qwen3-ASR

**What it is.** An audio encoder in front of a plain Qwen3 decoder. Model type `qwen3_asr`, 30 languages and 22 Chinese dialects, speech and
singing; the 0.6B and the 1.7B differ only in widths. The reference transcribes the JFK clip correctly on a CPU in 2.5 s (torch, fp32).

**Audio front end** (`WhisperFeatureExtractor`, 128 mel bins, 16 kHz, `n_fft` 400, hop 160: 100 mel frames a second; padded to the longest, no
truncation; the feature length is the number of valid frames).

**Audio encoder** (`Qwen3ASRAudioEncoder`; 0.6B: 18 layers, `d_model` 896, 14 heads, FFN 3584, output 1024; 1.7B: 24 layers, 1024, 16 heads,
FFN 4096, output 2048):

1. The mel is cut into chunks of `n_window * 2 = 100` frames (the last one shorter).
2. Each chunk goes through three `Conv2d(3x3, stride 2, pad 1)` with GELU (1 to 480 channels, then 480 to 480 twice): 100 frames become
   13 (`ceil` three times), 128 mel bins become 16; the 480 x 16 values of a frame go through a bias-free `Linear` to `d_model`.
3. A fixed sinusoidal embedding (`sin` then `cos`, 10000 timescale, `d_model` channels) for the position **inside the chunk** is added.
4. The chunks' frames are joined. Output length: `((((n % 100) - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1 + (n // 100) * 13` for `n` mel frames.
5. Pre-LN transformer layers (`LayerNorm`, multi-head attention with biases and no mask inside a window, GELU FFN); attention runs inside
   windows of `n_window_infer = 800` mel frames (104 encoder frames; a shorter last window).
6. `ln_post`, `proj1`, GELU, `proj2` to the decoder width.

**Decoder** (`Qwen3ASRThinkerTextModel`): a plain Qwen3 (0.6B: 28 layers, hidden 1024, 16 heads, 8 KV heads, head 128, MLP 3072; 1.7B:
hidden 2048, MLP 6144; vocabulary 151936, tied embeddings, RoPE theta 1e6, per-head q/k RMSNorm). The config says interleaved MRoPE with
sections 24/20/20, but for ASR all three position streams are the same number (`cumsum(mask) - 1`), so it is **ordinary RoPE**.

**Prompt.** `<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>` + one `<|audio_pad|>` per encoder frame +
`<|audio_end|><|im_end|>\n<|im_start|>assistant\n`; with a forced language `language {Name}<asr_text>` is appended. The audio-pad
embeddings are replaced by the encoder's output (ids: start 151669, end 151670, pad 151676). Greedy decoding (`do_sample` false); the answer is
`language English<asr_text>the text<|im_end|>`; `parse_asr_output` splits it; `detect_and_fix_repetitions` (threshold 20) cleans loops.

**Long audio.** The reference takes up to 1200 s in one piece, cutting at the quietest point within 5 s of the limit
(`split_audio_into_chunks`), padding pieces under 0.5 s. A forced aligner (`Qwen3-ForcedAligner-0.6B`) gives word timestamps; it is
not part of this plan.

**Plan for the engine** (`c/qwen3asr.c`, family `qwen3_asr`, modality `audio_in`): reuse the Qwen3 layer code and kernels
(`acestep_dit.h`, `qi_gemm.h`, as `qwen3embed.c` does), add a KV-cache decode loop, the mel front end, the encoder, the prompt and the
parser; serve it on the existing `POST /v1/audio/transcriptions` route (the gateway already decodes WAV, MP3 and, with ffmpeg, more, to
16 kHz mono), with `language` and `prompt` (the context) honoured; catalogue entries `qwen3-asr-0.6b` and `qwen3-asr-1.7b`.

## Qwen3-TTS

**What it is** (model type `qwen3_tts`; variants, all Apache-2.0): `CustomVoice` (nine preset voices, `instruct` for style), `VoiceDesign`
(a voice from a description), `Base` (clone a voice from a 3 s reference clip and its transcript), each in 1.7B and 0.6B (the 0.6B has no
VoiceDesign); plus the speech codec `Qwen3-TTS-Tokenizer-12Hz`. 10 languages and some Chinese dialects. Streaming is claimed (first packet
after one character, 97 ms).

**Pieces** (from the 0.6B CustomVoice `config.json`; the code is 2300 + 1000 lines and has not been read yet):

- **Talker**: a Qwen3-style decoder, 28 layers, hidden 1024, 16 heads, 8 KV heads, head 128, MLP 3072, interleaved MRoPE sections 24/20/20,
  vocabulary 3072 (the first codebook plus control tokens: `codec_bos` 2149, `eos` 2150, `pad` 2148, `think` 2154, language ids such as
  English 2050 and Chinese 2055, speaker ids such as Ryan 3061). Its text side has width 2048 (`text_hidden_size`), vocabulary 151936, so a
  text embedding is projected down; TTS control tokens: `tts_bos` 151672, `eos` 151673, `pad` 151671.
- **Code predictor**: 5 layers, hidden 1024, vocabulary 2048, `num_code_groups` 16: the talker gives codebook 0 of each 12.5 Hz frame, this
  small model gives the other 15, one after another.
- **Speech codec** (`Qwen3-TTS-Tokenizer-12Hz`, 682 MB): 12.5 Hz, 16 quantizers of 2048 entries; a lightweight causal ConvNet decoder (no
  DiT) turns the 16 codes of each frame into audio; an encoder side exists for cloning (`Base`), plus a speaker encoder (x-vector).

**Plan for the engine** (`c/qwen3tts.c`, family `qwen3_tts`, modality `speech_out`): start with the 0.6B CustomVoice (a preset voice, no
cloning): talker, code predictor, codec decoder, sampling as in `generation_config.json`, `POST /v1/audio/speech` in OpenAI's shape (`input`,
`voice`, `response_format` wav/mp3, `instructions` mapped to `instruct`); then the 1.7B, VoiceDesign, and cloning (codec encoder, speaker
encoder) as later stages.

## Staging and how each stage is checked

Each stage is held to the reference by comparing tensors, as ACE-Step was: the reference side is a small script run in the matching
environment that writes its intermediate tensors; the C side loads the same input and compares.

| Stage | What | Check |
|---|---|---|
| A0 | reference capture for the 0.6B and the JFK clip: mel features, encoder output, prompt ids, generated ids | the stored numbers |
| A1 | mel front end in C | against the reference features (max abs difference) |
| A2 | audio encoder in C | against the reference encoder output |
| A3 | decoder, prompt, greedy decode, parse | the generated ids equal the reference's |
| A4 | line protocol, gateway route, catalogue, registry, router, long audio, `coli` | tests with a scripted engine, and the real model |
| A5 | the 1.7B; Vulkan (encoder and decode on the chain); speed table | measured |
| T0 | read `modeling_qwen3_tts*.py`; reference capture (text ids, codes, waveform) | stored |
| T1 | codec decoder in C | codes to waveform against the reference |
| T2 | talker and code predictor, sampling | the codes (greedy and seeded) against the reference |
| T3 | line protocol, `/v1/audio/speech`, catalogue, router, `coli speak` | tests, real model |
| T4 | 1.7B, VoiceDesign | measured |
| T5 | cloning (codec encoder, speaker encoder) | against the reference |

## Status

| Stage | State |
|---|---|
| A0 reference capture | done: `tools/qwen3_asr_ref.py` (stock and `--windowed`), the 0.6B and 1.7B references |
| A1 mel front end | done: 1.4e-5 against the reference features |
| A2 audio encoder | done: 5e-6 (windowed oracle), 6.6e-7 (full attention against stock transformers) |
| A3 decoder, prompt, greedy decode, parse | done: the prompt and the 30 generated ids identical, 0.6B and 1.7B |
| A4 protocol, gateway, catalogue, registry, router, `coli`, long audio, tests | done: [qwen3-asr.md](qwen3-asr.md); on the branch `feat/qwen3-speech`, not merged, not pushed, not deployed |
| A5 Vulkan (encoder and decode on the chain) | open: decoding is 125 ms a token on the CPU |
| T0 to T5 (Qwen3-TTS) | not started |

Found on the way: the stock transformers encoder ignores the 800-frame attention windows that the checkpoint's config and the vLLM backend
use; this engine implements the windows and can reproduce the stock behaviour (`AS_FULL_ATTN=1`).
