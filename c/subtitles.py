#!/usr/bin/env python3
"""Subtitles from timed words: the words gathered into lines (cues), and the SRT and WebVTT files for them.

    cues = build_cues(items, marks)        items: [{"word", "start", "end"}], marks: 0 nothing after the word, 1 a comma or the like, 2 a full stop or the like
    srt(cues), vtt(cues)

A line is closed when the next word would make it longer than `max_chars` or longer in time than `max_seconds`, after a word that ended a sentence, after a
comma once the line is half full, and across a pause of more than a second. Stdlib only; used by the gateway's alignment route and by `coli transcribe`.
"""
from align_engine import is_cjk_char

PAUSE = 1.0                  # a silence this long between two words starts a new line


def join_words(words):
    """The words as text: a space between them, except where a CJK character meets another character (those scripts have no spaces)."""
    out = ""
    for word in words:
        if out and not (is_cjk_char(out[-1]) or is_cjk_char(word[0])):
            out += " "
        out += word
    return out


def build_cues(items, marks=None, max_chars=42, max_seconds=7.0):
    """[{"start", "end", "text"}] for the timed words (seconds), in order."""
    marks = list(marks) if marks else [0] * len(items)
    marks += [0] * (len(items) - len(marks))
    cues, current = [], []

    def close():
        nonlocal current
        if current:
            cues.append({"start": current[0]["start"], "end": current[-1]["end"], "text": join_words([w["word"] for w in current])})
            current = []

    for at, item in enumerate(items):
        if current:
            candidate = join_words([w["word"] for w in current] + [item["word"]])
            if (len(candidate) > max_chars or item["end"] - current[0]["start"] > max_seconds or item["start"] - current[-1]["end"] > PAUSE):
                close()
        current.append(item)
        mark = marks[at]
        if mark == 2 or (mark == 1 and len(join_words([w["word"] for w in current])) >= max_chars // 2):
            close()
    close()
    return cues


def stamp(seconds, separator):
    total = max(0, int(round(seconds * 1000)))
    hours, rest = divmod(total, 3600000)
    minutes, rest = divmod(rest, 60000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def srt(cues):
    return "".join(f"{number}\n{stamp(cue['start'], ',')} --> {stamp(cue['end'], ',')}\n{cue['text']}\n\n" for number, cue in enumerate(cues, 1))


def vtt(cues):
    return "WEBVTT\n\n" + "".join(f"{stamp(cue['start'], '.')} --> {stamp(cue['end'], '.')}\n{cue['text']}\n\n" for cue in cues)
