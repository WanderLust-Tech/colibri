#!/usr/bin/env python3
"""A stand-in for c/qwen3asr, speaking its serve protocol (the header of c/qwen3asr.c), for tests.

    asr_stub.py --model DIR --serve

It "hears" the WAV file it is given by its length: the answer text is `{language} | {seconds} s | {context}` (the last part only when there is a context), where language is the
forced one or "English" (what a model that detected English would say), so a test can see that the file reached the engine
decoded, that the language and the context were passed on, and what the gateway does with the answer. Environment: ASR_STUB_DELAY
(seconds per request), ASR_STUB_DIE_ON (a context that makes the stub exit without answering), ASR_STUB_FAIL_ON (a context it
answers with ERROR), ASR_STUB_LANGUAGE (what it says it detected; default English, "" for a model that does not say).
"""
import json
import os
import sys
import time
import wave

DELAY = float(os.environ.get("ASR_STUB_DELAY", "0"))
DIE_ON = os.environ.get("ASR_STUB_DIE_ON")
FAIL_ON = os.environ.get("ASR_STUB_FAIL_ON")
DETECTED = os.environ.get("ASR_STUB_LANGUAGE", "English")


def send(keyword, body):
    sys.stdout.write(f"{keyword} {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n")
    sys.stdout.flush()


def main():
    if "--serve" not in sys.argv:
        sys.exit("asr_stub: --serve only")
    send("READY", {"model": "asr-stub", "task": "transcribe", "sample_rate": 16000, "piece_seconds": 1200, "layers": 1, "encoder_layers": 1,
                   "device": "cpu"})
    for line in sys.stdin:
        keyword, _, rest = line.rstrip("\r\n").partition(" ")
        if keyword != "TRANSCRIBE":
            send("ERROR", {"id": "", "message": "unknown message"})
            continue
        try:
            request = json.loads(rest)
        except ValueError:
            send("ERROR", {"id": "", "message": "TRANSCRIBE needs a JSON object"})
            continue
        rid = request.get("id", "")
        path = request.get("wav")
        if not isinstance(path, str):
            send("ERROR", {"id": rid, "message": "wav must be the path of a 16 kHz WAV file"})
            continue
        context = request.get("context") or ""
        if DELAY:
            time.sleep(DELAY)
        if DIE_ON is not None and DIE_ON == context:
            sys.exit(3)
        if FAIL_ON is not None and FAIL_ON == context:
            send("ERROR", {"id": rid, "message": "the audio could not be read"})
            continue
        try:
            with wave.open(path, "rb") as clip:
                rate, frames = clip.getframerate(), clip.getnframes()
                channels = clip.getnchannels()
        except (OSError, wave.Error, EOFError) as error:
            send("ERROR", {"id": rid, "message": f"cannot open {path}: {error}"})
            continue
        if rate != 16000 or channels != 1:
            send("ERROR", {"id": rid, "message": f"{path} is {rate} Hz, {channels} channel(s): the engine reads 16000 Hz mono"})
            continue
        seconds = frames / rate
        language = request.get("language") or DETECTED
        text = f"{language} | {seconds:.1f} s" + (f" | {context}" if context else "")
        send("TEXT", {"id": rid, "language": request.get("language") or DETECTED, "text": text, "tokens": 5, "pieces": 1,
                      "audio_seconds": seconds, "seconds": 0.01, "truncated": False})


if __name__ == "__main__":
    main()
