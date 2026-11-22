# Word timestamps: Qwen3-ForcedAligner

`c/qwen3asr` also runs [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) (Apache-2.0): given an audio file **and the text that is
said in it**, it says when each word starts and ends. It does not transcribe (it is told the words; it places them), so the usual use is two steps: Qwen3-ASR
writes the words down, the aligner times them. That is what gives subtitles (`srt`, `vtt`) and word timestamps, which the recogniser alone refuses.

| | |
|---|---|
| Model | `qwen3-forcedaligner-0.6b` (`coli model install qwen3-forcedaligner-0.6b`), 1.9 GB, RAM 4 / 6 GB; the catalogue entry is in the speech recognition family |
| Languages | Chinese, Cantonese, English, German, Spanish, French, Italian, Portuguese, Russian, Korean, Japanese |
| Audio | up to 300 s a request (the reference's limit); WAV, MP3, and with ffmpeg more |
| Times | in steps of 80 ms (the head's 5000 classes are 0 to 400 s) |
| Speed | an 11 s clip in about 0.13 s on the card once the weights are up (0.55 s the first request, which uploads them), 1.5 s on the CPU; 33 s and 66 words in 0.8 s |

```sh
coli model install qwen3-asr-0.6b qwen3-forcedaligner-0.6b
coli transcribe talk.mp3 --srt                  # hears it, times what it heard: talk.srt
coli transcribe talk.mp3 --words                # 0.32 0.56 And / 0.64 0.96 so / ...
coli align talk.mp3 --text-file transcript.txt --vtt          # you have the transcript already: talk.vtt
coli align clip.wav --text "And so, my fellow Americans" --json
curl http://127.0.0.1:8000/v1/audio/alignments -H "Authorization: Bearer $KEY" \
     -F model=qwen3-forcedaligner-0.6b -F file=@clip.wav -F text="And so, my fellow Americans" -F response_format=verbose_json
```

## The route: `POST /v1/audio/alignments` (an extension)

Multipart form (or JSON with `file` as base64), as `/v1/audio/transcriptions` takes its audio: `file`, `model`, `text` (what is said: required), and
optionally `language` (a name or an ISO code of the aligner's languages; it only matters for how the text is cut into words), `response_format`, `max_chars`
and `max_seconds`.

- **`json`** (default): `{"words": [{"word", "start", "end"}, ...]}`, seconds. **`verbose_json`** adds `task`, `model`, `language`, `duration` and `text`.
- **`text`**: a line for each word, `start end word`. **`srt`** and **`vtt`**: subtitles, the words gathered into lines that end at a sentence's end, at a comma
  once the line is half full, across a pause of over a second, and before they pass `max_chars` (default 42) or `max_seconds` (default 7).
- The text is cut into words as the reference package does it: a unit is a run of letters, digits and apostrophes (`3.5` is `35`, punctuation is dropped), and
  each CJK character is a unit of its own. **Japanese** goes through `nagisa` when it is installed and a character at a time without it; **Korean** is cut at
  spaces (the reference uses a dictionary tokenizer that is not here).
- 400 with the field named for: no `text`, a text with no words in it, an unknown `language`, a bad `response_format`, no `file`, audio that cannot be read,
  audio over 300 s. A recogniser answers this route with 400 and says it transcribes; the aligner answers `/v1/audio/transcriptions` with 400 and says it aligns.
- `GET /v1/models` lists it with the capability `alignment` and an `alignment` block (`max_seconds`, `languages`, `formats`, `device`); the router keeps it
  beside the other models and loads it when it is asked for.

## How it works

The same network as Qwen3-ASR with a different end: the 1.7B's audio encoder (24 layers, 1024 wide), the 0.6B's decoder (28 layers, 1024), and a head of
5000 outputs instead of the vocabulary. There is no generation:

```
<|audio_start|> [one slot for each encoder frame] <|audio_end|> word1 <timestamp><timestamp> word2 <timestamp><timestamp> ...
  -> the audio encoder; the decoder (causal, as in the recogniser) over the whole prompt in one pass
  -> at each <timestamp> position, the head's argmax is a class, times 80 ms: the start, then the end, of the word before it
  -> fix_timestamp: the longest non-decreasing run of times stays, the others are interpolated between their neighbours or snapped to them
```

The engine's `ALIGN {id, wav, words}` answers `ALIGNED {id, starts, ends (ms), audio_seconds, seconds}`; the gateway cuts the text into words and turns the
milliseconds into seconds ([qwen3asr.c](../c/qwen3asr.c), [align_engine.py](../c/align_engine.py), [subtitles.py](../c/subtitles.py)). On the card the encoder's
layers and the decoder's pass are the chain's (the recogniser's, plus one more output: the decoder's hidden state for every row).

**One difference from the recogniser, on purpose.** The recogniser's audio encoder attends inside windows of 800 mel frames, as the checkpoint's config and the
vLLM backend do; stock transformers ignores them. The aligner has no vLLM backend, so its reference is the transformers code, and the engine follows it: the encoder
attends over the whole clip (the same as the windows under 8 s). `AS_FULL_ATTN` is implied for it.

## Checked against the reference

`tools/qwen3_aligner_ref.py` runs the official `qwen-asr` aligner in fp32 on the CPU and stores the prompt's ids, the class at each timestamp, the head's logits and the
fixed times; `qwen3asr --words-file F --wav CLIP --ref DIR` holds the engine to them. On the 11 s JFK clip (22 words, 44 times) and on the clip three times over (33 s,
66 words, 132 times): the prompt's ids identical, **0 of 44 and 0 of 132 classes differ**, the logits within 1.4e-5, every time in milliseconds identical, on the CPU and
on the card. (Before the encoder's attention matched the package's, 2 of 44 classes differed on the 11 s clip and the last word was 80 ms off.)

## Tests

`tests/test_align.py` (33): the word rules (letters, digits, CJK, punctuation, the marks that sentences and clauses end on), the request rules, the subtitles (a line
closed by length, by a sentence, by a pause, by time; SRT and VTT stamps), the whole route against a stub engine (every format, a JSON body, each refusal with its field,
the two models refusing each other's routes, the model card), and `coli align` / `coli transcribe --srt|--words` against a recogniser and an aligner.
`tests/test_qwen3aligner_real.py` (7 + 1, needs the model and the clip): every word's milliseconds against `tools/qwen3_aligner_reference.json`, the 33 s clip, times that
never go backwards for words that are not in the audio, bad requests, and the stage tensors with `QWEN3ALIGNER_REF`; CPU and card.

## Not done

Word times inside `/v1/audio/transcriptions` itself (`timestamp_granularities[]=word`, `verbose_json` words): a recogniser cannot call another model from inside a
request, so the chain is the client's (`coli transcribe --srt` does it; any client can make the two calls). Audio over 300 s (cut it, or align it in parts), the Korean
dictionary tokenizer, and a per-language check that the text is in the language the audio is in (the aligner places whatever words it is given).
