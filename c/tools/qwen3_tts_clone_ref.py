#!/usr/bin/env python3
"""Reference tensors of Qwen3-TTS voice cloning (the Base models) for the C engine to be held to (docs/qwen3-speech.md, stage T5).

    <venv with qwen-tts>\\python tools/qwen3_tts_clone_ref.py MODEL_DIR OUT_DIR REF_WAV24 --ref-text "..." --text "..." [--language English]

REF_WAV24 is a 24 kHz mono 16-bit WAV (the engine's own input: the gateway resamples whatever it is sent). Runs the official `qwen-tts` code in fp32 on
the CPU, greedily, once in each mode of `generate_voice_clone`, and writes raw little-endian files and a `meta.json`:

    mel.f32            [frames][128]      the speaker encoder's input (mel_spectrogram, n_fft 1024, hop 256)
    spk.f32            [enc_dim]          the speaker embedding (ECAPA-TDNN)
    codec_conv.f32     [frames25][512]    the codec encoder's convolutional stack (25 Hz)
    codec_tr.f32       [frames25][512]    ... after its transformer
    codec_emb.f32      [frames][512]      ... after the downsampling (12.5 Hz), the quantizer's input
    ref_codes.i32      [frames][16]       the reference's codes
    xvec/ and icl/     one folder per mode: input_ids.i32 (the text prompt), ref_ids.i32 (icl: the reference text's), prefill.f32, codes.i32,
                       wav.f32 (the speech, the reference's part cut off as the package does)
"""
import argparse
import json
import os
import sys
import wave

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


def read_wav(path):
    with wave.open(path, "rb") as clip:
        assert clip.getframerate() == 24000 and clip.getnchannels() == 1 and clip.getsampwidth() == 2, "24 kHz mono 16-bit WAV please"
        raw = clip.readframes(clip.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def dump(arr, path):
    np.asarray(arr, dtype="<f4").tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("ref_wav")
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--language", default="English")
    ap.add_argument("--max-new", type=int, default=90)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wav = read_wav(a.ref_wav)

    tts = Qwen3TTSModel.from_pretrained(a.model, dtype=torch.float32, device_map="cpu")
    inner = tts.model
    tok = inner.speech_tokenizer.model                    # Qwen3TTSTokenizerV2Model
    seen = {}

    def keep(name):
        def hook(module, args, output):
            out = output.last_hidden_state if hasattr(output, "last_hidden_state") else (output[0] if isinstance(output, tuple) else output)
            seen[name] = out.detach().clone()
        return hook

    mimi = tok.encoder
    h1 = mimi.encoder.register_forward_hook(keep("conv"))                       # [1][512][L25]
    h2 = mimi.encoder_transformer.register_forward_hook(keep("tr"))             # [1][L25][512]
    h3 = mimi.downsample.register_forward_hook(keep("emb"))                     # [1][512][L]

    # the speaker encoder's input, as extract_speaker_embedding makes it
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    mel = mel_spectrogram(torch.from_numpy(wav).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0,
                          fmax=12000).transpose(1, 2)
    dump(mel[0].numpy(), os.path.join(a.out, "mel.f32"))
    spk = inner.extract_speaker_embedding(audio=wav, sr=24000)
    dump(spk.numpy(), os.path.join(a.out, "spk.f32"))
    enc = tts.model.speech_tokenizer.encode(wav, sr=24000)
    codes = enc.audio_codes[0].to(torch.int32)                                  # [frames][16]
    codes.numpy().astype("<i4").tofile(os.path.join(a.out, "ref_codes.i32"))
    dump(seen["conv"][0].transpose(0, 1).numpy(), os.path.join(a.out, "codec_conv.f32"))
    dump(seen["tr"][0].numpy(), os.path.join(a.out, "codec_tr.f32"))
    dump(seen["emb"][0].transpose(0, 1).numpy(), os.path.join(a.out, "codec_emb.f32"))
    for h in (h1, h2, h3):
        h.remove()

    meta = {"model": os.path.basename(os.path.normpath(a.model)), "text": a.text, "ref_text": a.ref_text, "language": a.language,
            "ref_samples": int(wav.shape[0]), "mel_frames": int(mel.shape[1]), "spk_dim": int(spk.numel()), "ref_frames": int(codes.shape[0]),
            "frames25": int(seen["conv"].shape[2])}

    for mode, xvec in (("xvec", True), ("icl", False)):
        out = os.path.join(a.out, mode)
        os.makedirs(out, exist_ok=True)
        talker = inner.talker
        rec = {}
        original_generate = talker.generate

        def recording_generate(*args, _rec=rec, _og=original_generate, **kwargs):
            _rec["prefill"] = kwargs["inputs_embeds"].detach().clone()
            return _og(*args, **kwargs)

        talker.generate = recording_generate
        original = inner.generate

        def keep_codes(*args, _rec=rec, _og=original, **kwargs):
            _rec["ref_ids"] = kwargs.get("ref_ids")
            _rec["input_ids"] = kwargs.get("input_ids")
            result = _og(*args, **kwargs)
            _rec["codes"] = result[0][0].detach().clone()
            return result

        inner.generate = keep_codes
        wavs, sr = tts.generate_voice_clone(text=a.text, language=a.language, ref_audio=(wav, 24000), ref_text=a.ref_text, x_vector_only_mode=xvec,
                                            do_sample=False, subtalker_dosample=False, max_new_tokens=a.max_new)
        talker.generate = original_generate
        inner.generate = original
        ids = rec["input_ids"][0].reshape(-1).to(torch.int32)
        ids.numpy().astype("<i4").tofile(os.path.join(out, "input_ids.i32"))
        if rec["ref_ids"] and rec["ref_ids"][0] is not None:
            rec["ref_ids"][0].reshape(-1).to(torch.int32).numpy().astype("<i4").tofile(os.path.join(out, "ref_ids.i32"))
        dump(rec["prefill"][0].numpy(), os.path.join(out, "prefill.f32"))
        c = rec["codes"].to(torch.int32)
        c.numpy().astype("<i4").tofile(os.path.join(out, "codes.i32"))
        dump(np.asarray(wavs[0], dtype=np.float32), os.path.join(out, "wav.f32"))
        meta[mode] = {"prefill_len": int(rec["prefill"].shape[1]), "frames": int(c.shape[0]), "samples": int(np.asarray(wavs[0]).shape[0]),
                      "ids": int(ids.numel()), "codes": c.tolist()}
        print(mode, json.dumps({k: v for k, v in meta[mode].items() if k != "codes"}))
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("xvec", "icl")}))


if __name__ == "__main__":
    sys.exit(main())
