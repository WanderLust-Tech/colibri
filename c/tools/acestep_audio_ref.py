#!/usr/bin/env python3
"""The oracle for acestep_audio.h: upstream's own audio input, run on the same bytes as the C code.

    PYTHONUTF8=1 python acestep_audio_ref.py [--exe PATH] [--cc gcc] [--slow] [--no-robustness]

Fixtures (WAV, MP3 and damaged files) are written here, and for each one:

  load       upstream's `_read_audio_file` (soundfile: libsndfile, and mpg123 for MP3) against aca_load and aca_load_mem,
             sample for sample;
  stereo48k  upstream's `_numpy_to_channels_first` + `_normalize_audio_to_stereo_48k` (torchaudio's Resample, the clamp)
             against aca_stereo48k, sample for sample.

The upstream functions are imported from the file itself (acestep/core/generation/handler/io_audio.py, found through
ACESTEP_UPSTREAM, a clone of https://github.com/ace-step/ACE-Step-1.5), not through the package: nothing of upstream is
installed or modified. Needs the reference venv (tools/requirements-acestep-ref.txt): numpy, torch, torchaudio, soundfile.

Then the robustness suite, which needs no upstream: garbage, an empty file, a header that claims 4 GB, a zero-length data
chunk, an MP3 whose Xing header claims two billion frames, a clip over the cap, a sample rate with an absurd kernel, FLAC and
Ogg by name, and 150 randomly damaged WAV/MP3 files that must end in OK or ERR and never in a crash. Every refusal is
checked for its message, and for the memory it took to refuse.

Tolerances, max abs error on samples in [-1, 1], measured on the cases below (the third column is the margin):
  WAV load       0          exact: libsndfile and the C loader scale an integer by the same power of two
  WAV resampled  6e-7       measured 2.4e-7 (float32 accumulation order against torch's), margin 2.5x; tight on purpose:
                            a filter two taps narrower on each side moves the output by 1.25e-6
  MP3 load       2e-5       measured 1.4e-6 (minimp3 against mpg123, both float), margin 14x
  MP3 resampled  2e-5       measured 1.6e-6, margin 13x
"""
import argparse
import importlib.util
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.join(HERE, "..")

TOL_LOAD_WAV = 0.0
TOL_LOAD_MP3 = 2e-5
TOL_NORM_WAV = 6e-7
TOL_NORM_MP3 = 2e-5
EDGE = 200                                      # samples at the end of a cut-off clip that the resampler's filter sees differently


# ---------------------------------------------------------------------------------------------------- the signals --
def _tone(sr, seconds, ch, seed, peak=0.9):
    """A few partials, noise, a click, and a stretch of full-scale square wave (a resampler overshoots a square wave past 1,
    which is what a clamp is for). peak above 1 scales it past full scale (float files only)."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    out = []
    for c in range(ch):
        x = 0.35 * np.sin(2 * np.pi * (330 + 110 * c) * t) + 0.2 * np.sin(2 * np.pi * (2100 + 300 * c) * t)
        x += 0.04 * rng.standard_normal(n)
        x[n // 5: n // 5 + 40] += 0.5 * (c + 1) / ch
        a, b = n // 2, min(n, n // 2 + int(sr * 0.06))
        x[a:b] = np.where(np.sin(2 * np.pi * 120 * t[a:b]) >= 0, 0.99, -0.99)
        out.append(x * (peak / 0.99) if peak > 1 else np.clip(x, -peak, peak))
    y = np.stack(out, 1) if ch > 1 else out[0]
    return y.astype(np.float32)


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def _riff(body):
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _wav_with_list_chunk(path_in, path_out):
    """The same file with a LIST chunk of ODD size (so a pad byte follows it) between fmt and data."""
    b = _read(path_in)
    i = b.index(b"data")
    isft = b"colibri-test\x00"                                    # 13 bytes
    text = b"INFO" + b"ISFT" + struct.pack("<I", len(isft)) + isft  # 25 bytes
    chunk = b"LIST" + struct.pack("<I", len(text)) + text + b"\x00"
    out = b[:i] + chunk + b[i:]
    _write(path_out, out[:4] + struct.pack("<I", len(out) - 8) + out[8:])


def _wav_odd_data(path, sr, frames):
    """8-bit mono with an odd number of frames: the data chunk is odd, a pad byte follows it, and then another chunk."""
    raw = np.random.default_rng(7).integers(0, 256, frames, dtype=np.uint8).tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, sr, sr, 1, 8)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(raw)) + raw + (b"\x00" if len(raw) & 1 else b"")
    body += b"LIST" + struct.pack("<I", 4) + b"INFO"
    _write(path, _riff(body))


def _wav_raw(path, tag, ch, sr, bits, data, data_size=None, fmt_size=16, riff_size=None):
    """A WAV written by hand, for the files soundfile will not write (bad formats, lying headers)."""
    blk = ch * bits // 8
    fmt = struct.pack("<HHIIHH", tag, ch, sr, (sr * blk) & 0xFFFFFFFF, blk, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", fmt_size) + fmt + b"data" + struct.pack("<I", len(data) if data_size is None else data_size) + data
    out = b"RIFF" + struct.pack("<I", len(body) if riff_size is None else riff_size) + body
    _write(path, out)


def _mp3_frame_size(h):
    """Bytes in the MP3 frame whose header is h (layer III)."""
    ver = (h[1] >> 3) & 3
    br_tab1 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
    br_tab2 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
    sr_tab = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}
    br = (br_tab1 if ver == 3 else br_tab2)[h[2] >> 4] * 1000
    sr = sr_tab[ver][(h[2] >> 2) & 3]
    return (144 if ver == 3 else 72) * br // sr + ((h[2] >> 1) & 1)


def _mp3_frames(b):
    out, i = [], 0
    while i + 4 <= len(b) and b[i] == 0xFF and (b[i + 1] & 0xE0) == 0xE0:
        n = _mp3_frame_size(b[i:i + 4])
        out.append((i, n))
        i += n
    return out


def _mp3_drop_first_frame(path_in, path_out):
    """The same stream without its Xing/Info frame: no gapless information, as an encoder that writes none leaves it."""
    b = _read(path_in)
    _write(path_out, b[_mp3_frame_size(b[:4]):])


def _id3v2(body):
    size = len(body)
    return b"ID3" + b"\x03\x00\x00" + bytes([(size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F]) + body


def _mp3_with_tags(path_in, path_out):
    """An ID3v2.3 tag (a text frame and padding) in front and a 128-byte ID3v1 tag behind."""
    b = _read(path_in)
    text = b"TIT2" + struct.pack(">I", 12) + b"\x00\x00" + b"\x00colibri-test" + b"\x00" * 300
    _write(path_out, _id3v2(text) + b + b"TAG" + b"colibri-test".ljust(30, b"\x00") + b"\x00" * 95)


def _mp3_id3_trap(path_in, path_out):
    """An ID3v2 tag whose body holds twelve real MP3 frames, as cover art or a lyrics frame might: a decoder that does not skip
    the tag by its size finds a valid stream in it and plays it before the song."""
    b = _read(path_in)
    fr = _mp3_frames(b)
    trap = b"".join(b[o:o + n] for o, n in fr[3:15])
    geob = b"GEOB" + struct.pack(">I", len(trap)) + b"\x00\x00" + trap
    _write(path_out, _id3v2(geob) + b)


def _mp3_crc_xing(path_in, path_out):
    """The Xing frame with its protection bit cleared and the two CRC bytes inserted, so the Xing tag sits 2 bytes later
    (the frame keeps its size: its zero tail gives up the 2 bytes)."""
    b = _read(path_in)
    n = _mp3_frame_size(b[:4])
    f = b[:n]
    assert f[-2:] == b"\x00\x00" and f[1] & 1
    g = bytes([f[0], f[1] & 0xFE, f[2], f[3]]) + b"\x00\x00" + f[4:-2]
    _write(path_out, g + b[n:])


def _mp3_xing_claims(path_in, path_out, frames):
    """The Xing frame says the file holds `frames` frames (the audio itself is unchanged)."""
    b = bytearray(_read(path_in))
    i = b.index(b"Xing")
    flags = struct.unpack(">I", b[i + 4:i + 8])[0]
    assert flags & 1
    b[i + 8:i + 12] = struct.pack(">I", frames)
    _write(path_out, bytes(b))


def build_fixtures(d, slow=False):
    """Write every comparison fixture into d; returns a list of (name, path, kind): wav, mp3 or mp3-prefix."""
    import soundfile as sf
    cases = []

    def wav(name, sr, ch, subtype, fmt="WAV", seconds=1.2, peak=0.9, seed=1):
        p = os.path.join(d, name + ".wav")
        sf.write(p, _tone(sr, seconds, ch, seed, peak), sr, subtype=subtype, format=fmt)
        cases.append((name, p, "wav"))
        return p

    wav("wav44k_st_16", 44100, 2, "PCM_16", peak=1.0, seed=1)
    wav("wav48k_mono_24", 48000, 1, "PCM_24", seed=2)
    wav("wav22k_mono_f32", 22050, 1, "FLOAT", peak=1.6, seed=3)       # float samples beyond 1: only the clamp keeps them in
    wav("wav8k_8bit", 8000, 1, "PCM_U8", seed=4)
    wav("wav96k_st_24", 96000, 2, "PCM_24", seconds=0.6, seed=5)
    wav("wav32k_3ch_16", 32000, 3, "PCM_16", seed=6)                  # a third channel: dropped
    wav("wavx44k_st_16", 44100, 2, "PCM_16", fmt="WAVEX", seed=8)     # WAVE_FORMAT_EXTENSIBLE
    wav("wav48k_st_f64", 48000, 2, "DOUBLE", peak=1.3, seed=9)
    wav("wav16k_st_32", 16000, 2, "PCM_32", seed=10)
    wav("wav24k_mono_16", 24000, 1, "PCM_16", seed=11)
    wav("wav48k_st_f32_loud", 48000, 2, "FLOAT", peak=2.0, seed=12)   # no resampling, only the clamp
    wav("wav11k_mono_16", 11025, 1, "PCM_16", seconds=0.5, seed=13)   # a large reduced ratio (147 : 640)
    p = os.path.join(d, "wav_odd_8bit.wav")
    _wav_odd_data(p, 11025, 1001)                                      # 640 * 1001 / 147 is not a whole number: the length rounds up
    cases.append(("wav_odd_8bit", p, "wav"))
    base = os.path.join(d, "wav44k_st_16.wav")
    p = os.path.join(d, "wav_list_chunk.wav")
    _wav_with_list_chunk(base, p)
    cases.append(("wav_list_chunk", p, "wav"))
    p = os.path.join(d, "wav_truncated.wav")
    b = _read(base)
    _write(p, b[:len(b) - 1001])                             # the header claims the full length; the last frame is cut in two
    cases.append(("wav_truncated", p, "wav"))
    if slow:
        # 15 700 001 frames: 160 * 15700001 / 147 is 17088436.7, which torch rounds to float32 (17088436) before it takes the ceiling.
        p = os.path.join(d, "wav44k_mono_8bit_long.wav")
        n = 15_700_001
        x = (127 + 60 * np.sin(2 * np.pi * 440 * np.arange(n) / 44100) + np.random.default_rng(3).integers(-9, 9, n)).astype(np.uint8)
        _wav_raw(p, 1, 1, 44100, 8, x.tobytes())
        cases.append(("wav44k_mono_8bit_long", p, "wav"))

    def mp3(name, sr, ch, seconds=1.7, seed=20, cbr=False):
        p = os.path.join(d, name + ".mp3")
        kw = dict(compression_level=0.3, bitrate_mode="CONSTANT") if cbr else {}      # CBR: LAME writes "Info", not "Xing"
        with sf.SoundFile(p, "w", samplerate=sr, channels=ch, format="MP3", subtype="MPEG_LAYER_III", **kw) as f:
            f.write(_tone(sr, seconds, ch, seed, 0.8))
        cases.append((name, p, "mp3"))
        return p

    m44 = mp3("mp3_44k_st", 44100, 2, seed=20)
    cbr = mp3("mp3_44k_cbr", 44100, 2, seed=24, cbr=True)
    mp3("mp3_32k_st", 32000, 2, seed=21)
    mp3("mp3_22k_mono", 22050, 1, seed=22)                            # MPEG-2: 576 samples a frame, a smaller side-info block
    mp3("mp3_48k_mono", 48000, 1, seed=23)
    p = os.path.join(d, "mp3_44k_id3.mp3")
    _mp3_with_tags(m44, p)
    cases.append(("mp3_44k_id3", p, "mp3"))
    p = os.path.join(d, "mp3_44k_id3_trap.mp3")
    _mp3_id3_trap(cbr, p)
    cases.append(("mp3_44k_id3_trap", p, "mp3"))
    p = os.path.join(d, "mp3_44k_crc_xing.mp3")
    _mp3_crc_xing(m44, p)
    cases.append(("mp3_44k_crc_xing", p, "mp3"))
    p = os.path.join(d, "mp3_44k_notag.mp3")
    _mp3_drop_first_frame(cbr, p)                                      # CBR: libsndfile's length estimate is exact
    cases.append(("mp3_44k_notag", p, "mp3"))
    b = _read(m44)
    p = os.path.join(d, "mp3_44k_cut.mp3")
    _write(p, b[:int(len(b) * 0.6) + 37])                    # cut inside a frame
    cases.append(("mp3_44k_cut", p, "mp3-prefix"))
    return cases


# --------------------------------------------------------------------------------------------- upstream's code --
def load_upstream():
    """io_audio.py from the upstream clone, by file (its package __init__ files are never run)."""
    root = os.environ.get("ACESTEP_UPSTREAM")
    if not root:
        sys.exit("set ACESTEP_UPSTREAM to a clone of https://github.com/ace-step/ACE-Step-1.5")
    path = os.path.join(root, "acestep", "core", "generation", "handler", "io_audio.py")
    spec = importlib.util.spec_from_file_location("acestep_io_audio_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Case:
    """One fixture and what upstream makes of it: interleaved samples [frames, ch], the rate, planar stereo 48 kHz [2, n]."""

    def __init__(self, mod, name, path, kind):
        self.name, self.path, self.kind = name, path, kind
        audio_np, self.sr = mod._read_audio_file(path)
        self.inter = audio_np.reshape(audio_np.shape[0], -1)
        audio = mod.IoAudioMixin._numpy_to_channels_first(audio_np)
        self.norm = mod.IoAudioMixin._normalize_audio_to_stereo_48k(None, audio, self.sr).numpy()


def prepare_cases(d, slow=False):
    mod = load_upstream()
    return [Case(mod, *c) for c in build_fixtures(d, slow)]


# --------------------------------------------------------------------------------------------------- the C side --
def build_tool(cc, work, header_text=None, opt="-O2", extra=()):
    """Compile acestep_audio_tool.c into work (with a modified acestep_audio.h when header_text is given)."""
    import shutil
    shutil.copyfile(os.path.join(C_DIR, "acestep_audio_tool.c"), os.path.join(work, "acestep_audio_tool.c"))
    os.makedirs(os.path.join(work, "third_party"), exist_ok=True)
    shutil.copyfile(os.path.join(C_DIR, "third_party", "minimp3.h"), os.path.join(work, "third_party", "minimp3.h"))
    if header_text is None:
        shutil.copyfile(os.path.join(C_DIR, "acestep_audio.h"), os.path.join(work, "acestep_audio.h"))
    else:
        with open(os.path.join(work, "acestep_audio.h"), "w", encoding="utf-8", newline="") as f:
            f.write(header_text)
    exe = os.path.join(work, "acestep_audio_tool.exe" if os.name == "nt" else "acestep_audio_tool")
    cmd = [cc, opt, "-fopenmp", "-D_GNU_SOURCE", "-D_FILE_OFFSET_BITS=64", "-Wall", "-Wextra", *extra,
           os.path.join(work, "acestep_audio_tool.c"), "-o", exe, "-lm"]
    run = subprocess.run(cmd, capture_output=True, text=True)
    return (exe, "") if run.returncode == 0 else (None, run.stderr[-600:])


def _peak_mb(proc):
    """Peak working set of a finished child, in MB (None where it cannot be read)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(proc._handle)), ctypes.byref(pmc), pmc.cb)
            return pmc.PeakWorkingSetSize / 1e6 if ok else None
        import resource
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e3
    except Exception:
        return None


def run_tool(exe, mode, path, out, max_seconds=None, timeout=600):
    """(exit code, the tool's one line, peak MB)."""
    cmd = [exe, mode, path, out] + ([str(max_seconds)] if max_seconds is not None else [])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        so, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        return -999, "TIMEOUT", None
    line = so.strip().splitlines()[-1] if so.strip() else ""
    return p.returncode, line, _peak_mb(p)


def _fields(line):
    return dict(kv.split("=") for kv in line.split()[1:])


def compare_case(exe, case, tmp):
    """Run the C tool in all three modes on one fixture. Returns (problems, {mode: max abs error}).

    kind wav / mp3: the same format and the same number of frames as upstream, and the samples within the tolerance.
    kind mp3-prefix (cut off mid-stream): both decoders stop where the bytes stop, a frame apart at most, so the lengths may
    differ by one MP3 frame and the samples are compared over the shorter (the resampled clip leaves out its last EDGE samples,
    where the filter sees the cut)."""
    problems, details = [], {}
    mp3 = case.kind.startswith("mp3")
    prefix = case.kind == "mp3-prefix"
    inter, sr, norm = case.inter, case.sr, case.norm
    for mode in ("load", "mem", "stereo48k"):
        out = os.path.join(tmp, case.name + "." + mode + ".f32")
        rc, line, _ = run_tool(exe, mode, case.path, out)
        if rc != 0:
            problems.append(f"{case.name} {mode}: {line or 'exit ' + str(rc)}")
            continue
        f = _fields(line)
        got = np.fromfile(out, dtype=np.float32)
        if mode in ("load", "mem"):
            frames = int(f["frames"])
            same = (int(f["rate"]), int(f["channels"])) == (sr, inter.shape[1])
            if not same or (abs(frames - inter.shape[0]) > 1152 if prefix else frames != inter.shape[0]):
                problems.append(f"{case.name} {mode}: {line} but upstream has rate={sr} channels={inter.shape[1]} frames={inter.shape[0]}")
                continue
            n = min(frames, inter.shape[0])
            err = float(np.max(np.abs(got.reshape(frames, -1)[:n] - inter[:n])))
            tol = TOL_LOAD_MP3 if mp3 else TOL_LOAD_WAV
        else:
            nf = int(f["out_frames"])
            if abs(nf - norm.shape[1]) > 1152 * 2 if prefix else nf != norm.shape[1]:
                problems.append(f"{case.name} stereo48k: {nf} frames but upstream has {norm.shape[1]}")
                continue
            n = min(nf, norm.shape[1]) - (EDGE if prefix else 0)
            err = float(np.max(np.abs(got.reshape(2, nf)[:, :n] - norm[:, :n])))
            tol = TOL_NORM_MP3 if mp3 else TOL_NORM_WAV
        details[mode] = err
        if not err <= tol:
            problems.append(f"{case.name} {mode}: max abs error {err:.3g} > {tol:g}")
    return problems, details


def run_oracle(exe, cases, tmp, verbose=True, stop_early=False, only=None):
    problems = []
    for case in cases:
        if only is not None and case.name not in only:
            continue
        probs, details = compare_case(exe, case, tmp)
        if verbose:
            print(f"{case.name:24s} load {details.get('load', float('nan')):.2e}  stereo48k {details.get('stereo48k', float('nan')):.2e}  "
                  f"{'ok' if not probs else 'FAIL ' + '; '.join(probs)}", flush=True)
        problems += probs
        if problems and stop_early:
            break
    return problems


# -------------------------------------------------------------------------------------------------- robustness --
def _fuzz_inputs(d):
    """Small valid files (at most ~20 KB each) for the fuzzer to damage."""
    import soundfile as sf
    base = []
    for name, sr, ch, sub, fmt in (("a", 22050, 2, "PCM_16", "WAV"), ("b", 8000, 1, "PCM_U8", "WAV"), ("c", 16000, 3, "PCM_24", "WAV")):
        p = os.path.join(d, f"fuzz_{name}.wav")
        sf.write(p, _tone(sr, 0.15, ch, 5, 0.9), sr, subtype=sub, format=fmt)
        base.append(_read(p))
    p = os.path.join(d, "fuzz_m.mp3")
    sf.write(p, _tone(22050, 0.6, 1, 6, 0.8), 22050, format="MP3", subtype="MPEG_LAYER_III")
    mp3 = _read(p)
    base.append(mp3)
    p2 = os.path.join(d, "fuzz_t.mp3")
    _mp3_with_tags(p, p2)
    base.append(_read(p2))
    return base


def run_robustness(exe, tmp, verbose=True, stop_early=False, fuzz_runs=150):
    """Files that must be refused (or survived) cleanly. Returns the problems."""
    import soundfile as sf
    problems = []
    rng = np.random.default_rng(1234)
    MEM_LIMIT = 150.0                                                   # MB: refusing a 4 GB claim must not allocate anything like it

    def path(name):
        return os.path.join(tmp, name)

    def write(name, data):
        with open(path(name), "wb") as f:
            f.write(data)
        return path(name)

    def expect(name, file, want, max_seconds=None, mode="load", mem_limit=MEM_LIMIT, frames=None):
        rc, line, mb = run_tool(exe, mode, file, path("rb.out"), max_seconds, timeout=120)
        if want is None:                                                # accepted
            ok = rc == 0 and line.startswith("OK") and (frames is None or f"frames={frames}" in line.split("out_frames")[0].split("channels")[-1])
        else:                                                           # refused, with this in the message
            ok = rc == 1 and line.startswith("ERR") and want.lower() in line.lower()
        if mb is not None and mb > mem_limit:
            ok = False
            line += f" [peak {mb:.0f} MB]"
        if verbose:
            print(f"robust {name:34s} {'ok' if ok else 'FAIL'}  {line[:100]}", flush=True)
        if not ok:
            problems.append(f"robust {name}: {line or 'exit ' + str(rc)}")
        return ok

    z = np.zeros(4000, dtype=np.uint8).tobytes()
    steps = [
        ("garbage", lambda: expect("garbage", write("g.bin", b"A" + bytes(rng.integers(0, 256, 3000, dtype=np.uint8))), "not an audio file")),
        ("text", lambda: expect("text", write("t.txt", b"this is not audio at all, it is a sentence\n" * 20), "not an audio file")),
        ("empty", lambda: expect("empty", write("e.wav", b""), "empty")),
        ("riff-only", lambda: expect("riff-only", write("r.wav", b"RIFF\x04\x00\x00\x00WAVE"), "no data chunk")),
        ("flac", lambda: expect("flac", write("f.flac", b"fLaC" + b"\x00" * 60), "FLAC")),
        ("ogg", lambda: expect("ogg", write("o.ogg", b"OggS" + b"\x00" * 60), "Ogg")),
        ("aiff", lambda: expect("aiff", write("a.aif", b"FORM" + b"\x00" * 60), "AIFF")),
        ("m4a", lambda: expect("m4a", write("m.m4a", b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 60), "M4A")),
        ("zero-length-data", lambda: expect("zero-length-data", (_wav_raw(path("z.wav"), 1, 2, 44100, 16, b""), path("z.wav"))[1], "no audio")),
        ("data-before-fmt", lambda: expect("data-before-fmt", write("d.wav", _riff(b"WAVE" + b"data" + struct.pack("<I", 4) + b"\x00" * 4)), "before its fmt")),
        ("adpcm", lambda: expect("adpcm", (_wav_raw(path("ad.wav"), 2, 1, 8000, 4, z), path("ad.wav"))[1], "unsupported WAV encoding")),
        ("12-bit", lambda: expect("12-bit", (_wav_raw(path("b12.wav"), 1, 1, 8000, 12, z), path("b12.wav"))[1], "sample size")),
        ("zero-channels", lambda: expect("zero-channels", (_wav_raw(path("c0.wav"), 1, 0, 8000, 16, z), path("c0.wav"))[1], "channel count")),
        ("zero-rate", lambda: expect("zero-rate", (_wav_raw(path("r0.wav"), 1, 1, 0, 16, z), path("r0.wav"))[1], "sample rate")),
        ("huge-rate", lambda: expect("huge-rate", (_wav_raw(path("rh.wav"), 1, 1, 4_000_000_000, 16, z), path("rh.wav"))[1], "sample rate")),
        ("bad-block-align", lambda: expect("bad-block-align", write("ba.wav", _riff(b"WAVE" + b"fmt " + struct.pack("<I", 16) + struct.pack("<HHIIHH", 1, 2, 8000, 32000, 3, 16) + b"data" + struct.pack("<I", 4000) + z)), "block alignment")),
        ("truncated-fmt", lambda: expect("truncated-fmt", write("tf.wav", b"RIFF\x20\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00"), "fmt chunk is truncated")),
    ]
    # a header that claims 4 GB of data in a file that holds 8 KB: what is there is used, nothing is allocated for the claim
    steps.append(("claims-4gb", lambda: expect("claims-4gb", (_wav_raw(path("big.wav"), 1, 2, 44100, 16, z * 2, data_size=0xFFFFFFFF, riff_size=0xFFFFFFFF), path("big.wav"))[1], None, frames=2000)))
    # a clip over the cap is refused from its header, with its data really there
    sf.write(path("cap.wav"), _tone(8000, 3.0, 1, 2), 8000, subtype="PCM_16")
    steps.append(("over-the-cap", lambda: expect("over-the-cap", path("cap.wav"), "longer than", max_seconds=2.0)))
    steps.append(("under-the-cap", lambda: expect("under-the-cap", path("cap.wav"), None, max_seconds=3.5, frames=24000)))
    # a rate whose reduced ratio to 48 kHz would need a multi-gigabyte kernel
    sf.write(path("odd.wav"), _tone(44056, 0.05, 1, 2), 44056, subtype="PCM_16")
    steps.append(("absurd-kernel", lambda: expect("absurd-kernel", path("odd.wav"), "cannot be resampled", mode="stereo48k")))
    # MP3: a Xing header that claims two billion frames, a stream cut inside its first frame, an ID3 tag that runs off the end
    sf.write(path("m.mp3"), _tone(32000, 1.0, 2, 3, 0.8), 32000, format="MP3", subtype="MPEG_LAYER_III")
    _mp3_xing_claims(path("m.mp3"), path("m_claim.mp3"), 2_000_000_000)
    steps.append(("mp3-claims-2e9-frames", lambda: expect("mp3-claims-2e9-frames", path("m_claim.mp3"), "longer than")))
    mb = _read(path("m.mp3"))
    steps.append(("mp3-cut-in-first-frame", lambda: expect("mp3-cut-in-first-frame", write("mc.mp3", mb[:150]), "")))
    steps.append(("mp3-over-the-cap", lambda: expect("mp3-over-the-cap", path("m.mp3"), "longer than", max_seconds=0.5)))
    steps.append(("mp3-id3-runs-off", lambda: expect("mp3-id3-runs-off", write("mi.mp3", b"ID3\x03\x00\x00\x00\x00\x7f\x7f" + b"\x00" * 50), "ID3")))
    # no Xing frame to claim a length, so only the cap applied while decoding can stop it
    _mp3_drop_first_frame(path("m.mp3"), path("m_notag.mp3"))
    steps.append(("mp3-over-the-cap-no-tag", lambda: expect("mp3-over-the-cap-no-tag", path("m_notag.mp3"), "longer than", max_seconds=0.5)))
    # a stream that changes its sample rate part-way (32 kHz frames, then 44.1 kHz frames)
    sf.write(path("m44.mp3"), _tone(44100, 1.0, 2, 4, 0.8), 44100, format="MP3", subtype="MPEG_LAYER_III")
    _mp3_drop_first_frame(path("m44.mp3"), path("m44_notag.mp3"))
    mn, m44n = _read(path("m_notag.mp3")), _read(path("m44_notag.mp3"))
    fa, fb = _mp3_frames(mn), _mp3_frames(m44n)
    mixed = mn[:fa[12][0]] + m44n[fb[2][0]:fb[14][0]] + mn[fa[12][0]:fa[24][0]]
    steps.append(("mp3-changes-rate", lambda: expect("mp3-changes-rate", write("mix.mp3", mixed), "changes its sample rate")))
    steps.append(("riff-but-not-wave", lambda: expect("riff-but-not-wave", write("avi.avi", b"RIFF\x04\x00\x00\x00AVI " + b"\x00" * 40), "not an audio file")))
    steps.append(("300-channels", lambda: expect("300-channels", (_wav_raw(path("c300.wav"), 1, 300, 8000, 8, bytes(3000)), path("c300.wav"))[1], "channel count")))
    for _, step in steps:
        step()
        if problems and stop_early:
            return problems

    # fuzz: random damage to small valid files; the tool must end in OK or ERR, never in a crash or a hang
    base = _fuzz_inputs(tmp)
    for k in range(fuzz_runs):
        b = bytearray(base[k % len(base)])
        op = int(rng.integers(0, 4))
        if op == 0:                                                     # flip bytes, mostly in the first 80 (the headers)
            for _ in range(int(rng.integers(1, 9))):
                b[int(rng.integers(0, min(80, len(b))))] = int(rng.integers(0, 256))
        elif op == 1:                                                   # flip bytes anywhere
            for _ in range(int(rng.integers(1, 20))):
                b[int(rng.integers(0, len(b)))] = int(rng.integers(0, 256))
        elif op == 2:                                                   # cut it anywhere
            b = b[:int(rng.integers(0, len(b)))]
        else:                                                           # junk spliced in
            i = int(rng.integers(0, len(b)))
            b[i:i] = bytes(rng.integers(0, 256, int(rng.integers(1, 400)), dtype=np.uint8))
        f = write("fuzz.bin", bytes(b))
        for mode in ("mem", "stereo48k"):
            rc, line, mbs = run_tool(exe, mode, f, path("fz.out"), 5.0, timeout=60)
            if rc not in (0, 1) or not (line.startswith("OK") or line.startswith("ERR")) or (mbs is not None and mbs > 400):
                problems.append(f"fuzz {k} (op {op}, {mode}): exit {rc}, {line[:80]!r}, {mbs}")
                if verbose:
                    print(f"fuzz {k} FAIL: {problems[-1]}", flush=True)
                if stop_early:
                    return problems
    if verbose:
        print(f"robust fuzz: {fuzz_runs} damaged files x 2 modes: {'ok' if not any(p.startswith('fuzz') for p in problems) else 'FAIL'}", flush=True)
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe", help="a built acestep_audio_tool (default: build one with gcc)")
    ap.add_argument("--cc", default="gcc")
    ap.add_argument("--slow", action="store_true", help="also the 6-minute clip whose length depends on torch's float32 ceil")
    ap.add_argument("--no-robustness", action="store_true")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="acestep-audio-") as work:
        exe = a.exe
        if not exe:
            exe, err = build_tool(a.cc, work)
            if not exe:
                sys.exit("build failed: " + err)
        cases = prepare_cases(work, slow=a.slow)
        problems = run_oracle(exe, cases, work)
        if not a.no_robustness:
            problems += run_robustness(exe, work)
    if problems:
        print("\nFAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("\nthe oracle passes")


if __name__ == "__main__":
    main()
