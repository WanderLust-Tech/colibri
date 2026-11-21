#!/usr/bin/env python3
"""A stand-in for c/qwen3tts, speaking its serve protocol (the header of c/qwen3tts.c), for tests.

    tts_stub.py --model DIR --serve

It "speaks" by writing a 24 kHz mono WAV file of 0.05 s a character of the text. So that a test can see what reached the engine, the
first samples of the file are not sound: sample 0 is the length of a string and the samples after it are that string's bytes, the
string being `speaker|language|seed|greedy|max_frames` (the fields as the engine got them). Environment: TTS_STUB_DELAY (seconds a
request takes), TTS_STUB_DIE_ON (a text that makes the stub exit without answering), TTS_STUB_FAIL_ON (a text it answers with ERROR),
TTS_STUB_TRUNCATE (say every answer was cut at the frame limit), TTS_STUB_NOFILE (answer AUDIO without writing the file),
TTS_STUB_INSTRUCT (a model that takes a style instruction: READY says so, and the echo ends with `|instruct`), TTS_STUB_KIND=voice_design
(no speakers: the instruction makes the voice), TTS_STUB_KIND=base (a cloning model: READY says `clone`, there are no speakers, SPEAK needs
`ref_wav` (the stub reads the file) and the echo ends with `|ref=<bytes of the file>|<ref_text>|<x_vector_only as 0 or 1>`).
Streaming (SPEAK "stream": true; READY says `stream`, unless TTS_STUB_NOSTREAM): the samples that would have gone to the file go out as three PART
lines (`seq`, `samples`, `pcm`: base64 of 16-bit little-endian), then AUDIO; TTS_STUB_PART_DELAY (seconds between parts), TTS_STUB_FAIL_AFTER_PARTS
(answer ERROR after that many parts).
"""
import json
import os
import struct
import sys
import time
import wave

DELAY = float(os.environ.get("TTS_STUB_DELAY", "0"))
DIE_ON = os.environ.get("TTS_STUB_DIE_ON")
FAIL_ON = os.environ.get("TTS_STUB_FAIL_ON")
TRUNCATE = bool(os.environ.get("TTS_STUB_TRUNCATE"))
NOFILE = bool(os.environ.get("TTS_STUB_NOFILE"))
INSTRUCT = bool(os.environ.get("TTS_STUB_INSTRUCT"))
KIND = os.environ.get("TTS_STUB_KIND", "custom_voice")
NOSTREAM = bool(os.environ.get("TTS_STUB_NOSTREAM"))
PART_DELAY = float(os.environ.get("TTS_STUB_PART_DELAY", "0"))
FAIL_AFTER_PARTS = int(os.environ.get("TTS_STUB_FAIL_AFTER_PARTS", "0"))
RATE = 24000
SPEAKERS = ["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"]
LANGUAGES = ["chinese", "english", "german", "italian", "portuguese", "spanish", "japanese", "korean", "french", "russian",
             "beijing_dialect", "sichuan_dialect"]


def send(keyword, body):
    sys.stdout.write(f"{keyword} {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n")
    sys.stdout.flush()


def main():
    if "--serve" not in sys.argv:
        sys.exit("tts_stub: --serve only")
    send("READY", {"model": "tts-stub", "task": "speech", "kind": KIND, "clone": KIND == "base", "stream": not NOSTREAM, "instruct": INSTRUCT or KIND == "voice_design",
                   "sample_rate": RATE, "layers": 1, "speakers": [] if KIND in ("voice_design", "base") else SPEAKERS, "languages": LANGUAGES,
                   "device": "cpu"})
    for line in sys.stdin:
        keyword, _, rest = line.rstrip("\r\n").partition(" ")
        if keyword != "SPEAK":
            send("ERROR", {"id": "", "message": "unknown message"})
            continue
        try:
            request = json.loads(rest)
        except ValueError:
            send("ERROR", {"id": "", "message": "SPEAK needs a JSON object"})
            continue
        rid = request.get("id", "")
        text, out = request.get("text"), request.get("out")
        if not isinstance(text, str) or not text:
            send("ERROR", {"id": rid, "message": "text must be a non-empty string"})
            continue
        stream = bool(request.get("stream"))
        if stream and NOSTREAM:
            send("ERROR", {"id": rid, "message": "unknown field stream"})
            continue
        if not stream and (not isinstance(out, str) or not out):
            send("ERROR", {"id": rid, "message": "out must be the path of the WAV file to write"})
            continue
        if DELAY:
            time.sleep(DELAY)
        if DIE_ON is not None and DIE_ON == text:
            sys.exit(3)
        if FAIL_ON is not None and FAIL_ON == text:
            send("ERROR", {"id": rid, "message": "the speaker is not one this model has"})
            continue
        speaker = request.get("speaker") or ""
        ref_wav = request.get("ref_wav")
        if KIND == "base" and not ref_wav:
            send("ERROR", {"id": rid, "message": "this is a Base model: it speaks in a cloned voice and needs reference audio (ref_wav)"})
            continue
        if KIND != "base" and ref_wav:
            send("ERROR", {"id": rid, "message": "this model does not clone voices (a Base checkpoint does)"})
            continue
        if speaker and speaker not in SPEAKERS and KIND != "base":
            send("ERROR", {"id": rid, "message": f"the speaker `{speaker}` is not one this model has"})
            continue
        echo = "|".join(str(x) for x in (speaker, request.get("language", ""), request.get("seed", 0), int(bool(request.get("greedy"))),
                                         request.get("max_frames", 0))).encode()
        if request.get("instruct"):
            echo += b"|" + request["instruct"].encode()
        if ref_wav:
            try:
                size = os.path.getsize(ref_wav)
            except OSError:
                send("ERROR", {"id": rid, "message": f"cannot open the reference audio {ref_wav}"})
                continue
            echo += f"|ref={size}|{request.get('ref_text', '')}|{int(bool(request.get('x_vector_only')))}".encode()
        samples = max(int(len(text) * 0.05 * RATE), len(echo) + 2)
        pcm = [len(echo)] + list(echo) + [0] * (samples - len(echo) - 1)
        if stream:
            import base64
            raw = struct.pack(f"<{len(pcm)}h", *pcm)
            cuts = [0, len(raw) // 3 // 2 * 2, len(raw) * 2 // 3 // 2 * 2, len(raw)]
            for seq in range(3):
                piece = raw[cuts[seq]:cuts[seq + 1]]
                send("PART", {"id": rid, "seq": seq, "samples": len(piece) // 2, "pcm": base64.b64encode(piece).decode()})
                if FAIL_AFTER_PARTS and seq + 1 >= FAIL_AFTER_PARTS:
                    send("ERROR", {"id": rid, "message": "the graphics card failed while the speech was going out: ask again"})
                    break
                if PART_DELAY:
                    time.sleep(PART_DELAY)
            else:
                send("AUDIO", {"id": rid, "path": "", "samples": samples, "sample_rate": RATE, "frames": samples // 1920,
                               "audio_seconds": samples / RATE, "seconds": 0.01, "truncated": TRUNCATE})
            continue
        if not NOFILE:
            with wave.open(out, "wb") as clip:
                clip.setnchannels(1)
                clip.setsampwidth(2)
                clip.setframerate(RATE)
                clip.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
        send("AUDIO", {"id": rid, "path": out, "samples": samples, "sample_rate": RATE, "frames": samples // 1920,
                       "audio_seconds": samples / RATE, "seconds": 0.01, "truncated": TRUNCATE})


if __name__ == "__main__":
    main()
