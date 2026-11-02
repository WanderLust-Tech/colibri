#!/usr/bin/env python3
"""Reference outputs of ACE-Step 1.5 turbo for the audio-INPUT paths (reference audio, cover, repaint), captured from
upstream's own handler, for the C engine's oracles. The text2music captures are tools/acestep_ref.py's.

    acestep_ref_audio.py enc MODEL_DIR OUTDIR WAVDIR     the VAE encoder: acestep --vae-enc-ref OUTDIR/<case>
    acestep_ref_audio.py tasks MODEL_DIR OUTDIR WAVDIR [--cases a,b]
                                                         whole generations with audio in: reference audio, cover, repaint

MODEL_DIR is the checkpoint, WAVDIR a directory of 48 kHz stereo wavs to encode (the audio a text2music capture wrote,
OUTDIR/<case>/audio.wav, will do: real generated music, where a synthetic signal would not exercise the network the way
music does). The handler is upstream's, CPU, float32, as acestep_ref.py runs it, and `tiled_encode` is upstream's own.

The one thing upstream does that cannot be reproduced is the posterior draw: `latent_dist.sample()` takes torch's global
generator, which nothing seeds, so two runs of upstream on one file differ by the posterior's std (about 1e-2 of a latent).
The capture replaces `sample` by `mean + std * randn(generator=G)` with one torch.Generator G seeded per case, drawn from in
window order: exactly what the C engine does with acg_randn's generator, and the posterior mean and scale are recorded too,
so the network and the sampling are each held to upstream on their own.

    OUTDIR/<case>/ref.safetensors   f32 (int64 for the settings)
        enc{k}.wav, enc{k}.params     a clip encoded in one pass: [1, 2, N] in, the posterior's parameters [1, 128, T] out
                                      (the mean in channels 0..63, the scale in 64..127)
        tile{k}.wav, tile{k}.out      a clip through tiled_encode: [1, 2, N] in, the sampled latents [1, 64, F] out
        tile{k}.cfg                   int64 [chunk, overlap, seed]
"""
import argparse
import os
import sys
import tempfile
import shutil
import atexit

import torch


def open_handler(model_dir):
    """Upstream's handler on CPU in float32, initialised the way acestep_ref.py does it."""
    root = os.environ.get("ACESTEP_UPSTREAM")
    if not root or not os.path.isdir(os.path.join(root, "acestep")):
        sys.exit("set ACESTEP_UPSTREAM to a clone of https://github.com/ace-step/ACE-Step-1.5")
    root, model_dir = os.path.abspath(root), os.path.abspath(model_dir)
    caller_cwd, scratch = os.getcwd(), tempfile.mkdtemp(prefix="acestep-ref-")
    os.chdir(scratch)

    def leave_scratch():
        os.chdir(caller_cwd)
        shutil.rmtree(scratch, ignore_errors=True)
    atexit.register(leave_scratch)
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = model_dir
    sys.path.insert(0, root)
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    from acestep.handler import AceStepHandler
    torch.set_num_threads(os.cpu_count() or 8)
    h = AceStepHandler()
    h._ensure_models_present = lambda **kw: None
    h._sync_model_code_if_needed = lambda *args, **kw: None
    msg, ok = h.initialize_service(project_root=root, config_path="acestep-v15-turbo", device="cpu",
                                   use_flash_attention=False, compile_model=False, use_mlx_dit=False)
    if not ok:
        sys.exit(str(msg))
    return h, root


def seeded_posterior(seed):
    """Make OobleckDiagonalGaussianDistribution.sample draw from a generator seeded with `seed`, in call order."""
    from diffusers.models.autoencoders import autoencoder_oobleck as ob
    gen = torch.Generator().manual_seed(seed)

    def sample(self, generator=None):
        eps = torch.randn(self.mean.shape, generator=gen, dtype=self.mean.dtype)
        return self.mean + self.std * eps
    ob.OobleckDiagonalGaussianDistribution.sample = sample


def load_wav(path):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 48000 or a.shape[1] != 2:
        sys.exit(f"{path}: need 48 kHz stereo, got {sr} Hz x {a.shape[1]}")
    return torch.from_numpy(a.T.copy())            # [2, N]


def concat(paths, extra=0):
    w = torch.cat([load_wav(p) for p in paths], dim=-1)
    return w[:, : w.shape[-1] + extra] if extra <= 0 else torch.cat([w, w[:, :extra]], dim=-1)


def capture_enc(a):
    from safetensors.torch import save_file
    h, _ = open_handler(a.model_dir)
    wavs = sorted(os.path.join(a.wavdir, f) for f in os.listdir(a.wavdir) if f.lower().endswith(".wav"))
    if len(wavs) < 2:
        sys.exit(f"{a.wavdir}: need at least two wavs (a long clip is made by joining them)")
    short, long_ = wavs[0], wavs[-1]
    S = 48000
    # (name, kind, audio, chunk, overlap, seed): enc = one pass, parameters recorded; tile = tiled_encode, sampled
    plan = [
        ("single10", "enc", concat([short]), None, None, 0),
        ("single_odd", "enc", concat([short], -777), None, None, 0),                      # not a whole number of hops
        ("single28", "enc", concat([long_])[:, : 28 * S], None, None, 0),                  # a window as long as the tiled encode's first
        ("tile_small", "tile", concat([short, long_])[:, : 25 * S], 8 * S, 1 * S, 7),     # five windows
        ("tile_odd", "tile", concat([short, long_])[:, : 25 * S + 777], 8 * S, 1 * S, 11),  # the last window partial, odd length
        # upstream's own overlap (2 s) and the chunk a GPU with more than 8 GB gets (30 s): with no GPU it picks 15 s, a
        # choice about memory, not about the model; the engine uses 30 s so a 30 s reference clip is one pass
        ("tile_default", "tile", concat([long_, long_])[:, : 40 * S], 30 * S, 2 * S, 3),
        ("tile_short_tail", "tile", concat([short, long_])[:, : 9 * S + 1000], 8 * S, 1 * S, 5),  # one window then a tail under the overlap
        ("tile_exact", "tile", concat([short, long_])[:, : 8 * S], 8 * S, 1 * S, 13),             # exactly one chunk: one pass, not two windows
    ]
    os.makedirs(a.outdir, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    for name, kind, audio, chunk, overlap, seed in plan:
        if only and name not in only:
            continue
        out = {}
        N = audio.shape[-1]
        x = audio.unsqueeze(0)
        if kind == "enc":
            with torch.inference_mode():
                params = h.vae.encode(x).latent_dist.parameters
            out["enc0.wav"], out["enc0.params"] = x.clone().contiguous(), params.clone().contiguous()
            if a.f64:               # the same network in float64: what the float32 results are rounding of
                from diffusers import AutoencoderOobleck
                vae64 = AutoencoderOobleck.from_pretrained(os.path.join(os.path.abspath(a.model_dir), "vae"), torch_dtype=torch.float64).eval()
                with torch.inference_mode():
                    p64 = vae64.encode(x.double()).latent_dist.parameters
                out["enc0.params64"] = p64.float().clone().contiguous()      # the C loader reads f32: 6e-8 relative, far below what it judges
                d = (params.double() - p64).abs().max().item()
                print(f"   torch float32 vs float64: max |diff| {d:.3e}")
            print(f"{name}: {N} samples -> parameters {tuple(params.shape)}")
        else:
            seeded_posterior(seed)
            kw = {}
            if chunk is not None:
                kw = dict(chunk_size=chunk, overlap=overlap)
            lat = h.tiled_encode(audio, offload_latent_to_cpu=True, **kw)
            if lat.dim() == 2:
                lat = lat.unsqueeze(0)
            out["tile0.wav"], out["tile0.out"] = x.clone().contiguous(), lat.clone().contiguous()
            out["tile0.cfg"] = torch.tensor([chunk if chunk is not None else 30 * S, overlap if overlap is not None else 2 * S, seed], dtype=torch.int64)
            print(f"{name}: {N} samples, chunk {out['tile0.cfg'][0].item()} overlap {out['tile0.cfg'][1].item()} -> latents {tuple(lat.shape)}")
        d = os.path.join(a.outdir, name)
        os.makedirs(d, exist_ok=True)
        save_file(out, os.path.join(d, "ref.safetensors"), metadata={"case": name})


# --------------------------------------------------------------------------------------------- whole generations --

TASK_INSTRUCTIONS = {"text2music": "Fill the audio semantic mask based on the given conditions:",
                     "repaint": "Repaint the mask area based on the given conditions:",
                     "cover": "Generate audio semantic tokens based on the given conditions:"}
S48 = 48000
VAE_SEED_XOR = 0x5EED0001          # the seed of the posterior draws is the request's seed xor this (acestep_gen.h ACG_VAE_SEED_XOR)

COMMON = dict(inference_steps=8, use_random_seed=False, seed=42, batch_size=1, dcw_enabled=True)
INSTR = dict(captions="bright ukulele and light percussion, summery", lyrics="[Instrumental]", vocal_language="unknown")
VOCAL = dict(captions="warm indie folk with soft vocals", lyrics="[Verse 1]\nmorning light on the kitchen floor\n[Chorus]\nsing it back to me once more", vocal_language="en")
# name: (generate_music arguments, the audio it needs: {"src": (wav stem, seconds or None), "ref": (...)}, seconds of the clip when text2music)
TASKS = {
    # cover: the source goes through the tokenizer and detokenizer, the instruction is the cover one, the length is the source's
    "cover10": ({**INSTR, "task_type": "cover"}, {"src": ("inst10", None)}),
    "cover_odd": ({**INSTR, "task_type": "cover"}, {"src": ("long30", 9.37)}),               # 234 frames: not a multiple of 5, so the tokenizer pads
    "cover_vocal": ({**VOCAL, "task_type": "cover"}, {"src": ("long30", 12.0)}),
    "cover_half": ({**INSTR, "task_type": "cover", "audio_cover_strength": 0.5}, {"src": ("inst10", None)}),      # the second half of the steps without the cover context
    "cover_noise": ({**INSTR, "task_type": "cover", "cover_noise_strength": 0.3}, {"src": ("inst10", None)}),     # starts from the source renoised, not from noise (level 0.7: the schedule's 0.75)
    "cover_pad4": ({**INSTR, "task_type": "cover"}, {"src": ("long30", 9.24)}),                # 231 frames: the tokenizer's last window is one frame and four of padding
    "cover_short": ({**INSTR, "task_type": "cover"}, {"src": ("long30", 3.0)}),               # 75 frames, under the floor of 128: the source is padded with silence latents
    "cover_ref": ({**VOCAL, "task_type": "cover"}, {"src": ("inst10", None), "ref": ("long30", None)}),   # both clips: the source is encoded first, then the reference
    # reference audio: the timbre encoder gets the clip's latents instead of the silence latent
    "ref30": ({**INSTR, "audio_duration": 10.0}, {"ref": ("long30", None)}),                  # exactly 30 s: three segments, no randomness
    "ref_short": ({**INSTR, "audio_duration": 10.0}, {"ref": ("inst10", None)}),               # 10 s: repeated three times
    "ref_long": ({**INSTR, "audio_duration": 10.0}, {"ref": ("long45", None)}),                # 45 s: a segment inside each third
    # repaint: the span is generated, the rest is the source, spliced back after the decode
    "repaint_mid": ({**INSTR, "task_type": "repaint", "repainting_start": 3.0, "repainting_end": 6.0}, {"src": ("inst10", None)}),
    "repaint_cons": ({**INSTR, "task_type": "repaint", "repainting_start": 3.0, "repainting_end": 6.0, "repaint_mode": "conservative"}, {"src": ("inst10", None)}),
    "repaint_aggr": ({**INSTR, "task_type": "repaint", "repainting_start": 3.0, "repainting_end": 6.0, "repaint_mode": "aggressive"}, {"src": ("inst10", None)}),
    "repaint_str": ({**INSTR, "task_type": "repaint", "repainting_start": 3.0, "repainting_end": 6.0, "repaint_strength": 0.8}, {"src": ("inst10", None)}),
    "repaint_tail": ({**INSTR, "task_type": "repaint", "repainting_start": 6.5, "repainting_end": -1}, {"src": ("inst10", None)}),      # to the end of the clip
    "repaint_extend": ({**INSTR, "task_type": "repaint", "repainting_start": 8.0, "repainting_end": 12.0}, {"src": ("inst10", None)}),  # past the end: the clip is padded with silence
    "repaint_short": ({**INSTR, "task_type": "repaint", "repainting_start": 1.0, "repainting_end": 3.0}, {"src": ("inst10", 4.0)}),    # 100 frames: under the floor
    "repaint_lead": ({**INSTR, "task_type": "repaint", "repainting_start": -2.0, "repainting_end": 3.0}, {"src": ("inst10", None)}),    # before the start
}


def clip_of(wavdir, stem, seconds):
    """The first `seconds` of the wav named `stem`, or the whole of it. long45 is long30 followed by 15 s of inst10 and a bit more."""
    if stem == "long45":
        w = torch.cat([load_wav(os.path.join(wavdir, "b_long30.wav")), load_wav(os.path.join(wavdir, "a_inst10.wav")),
                       load_wav(os.path.join(wavdir, "a_inst10.wav"))[:, : 5 * S48]], dim=-1)
    else:
        w = load_wav(os.path.join(wavdir, {"inst10": "a_inst10.wav", "long30": "b_long30.wav"}[stem]))
    return w if seconds is None else w[:, : int(round(seconds * S48))]


class AudioRecorder:
    """acestep_ref.Recorder, plus the tokenizer, the detokenizer and the VAE encoder's draws."""

    def __init__(self, h, base):
        self.base = base
        self.rec = base.Recorder(h)
        self.t = self.rec.t
        self.calls = 0
        self.first = self.last = {}
        tk, clean = h.model.tokenizer, base.clean
        # the tokenizer and detokenizer run inside every prepare_condition, also the non-cover one (on the silence latent, its
        # result thrown away by torch.where): only the first call's taps are the cover's
        once = lambda key, v: self.rec.t.__setitem__(key, clean(v)) if self.calls == 0 else None
        tk.audio_acoustic_proj.register_forward_pre_hook(lambda m, a: once("tok.x", a[0]))
        tk.attention_pooler.register_forward_hook(lambda m, a, o: once("tok.pooled", o if isinstance(o, torch.Tensor) else o[0]))
        tk.quantizer.project_in.register_forward_hook(lambda m, a, o: once("tok.z", o))
        def quant_hook(m, a, o):               # a hook must return None: a value replaces the module's output
            once("tok.q", o[0])
            once("tok.idx", o[1])
        tk.quantizer.register_forward_hook(quant_hook)
        h.model.detokenizer.register_forward_hook(lambda m, a, o: once("detok.out", o if isinstance(o, torch.Tensor) else o[0]))
        # The handler calls prepare_condition itself (the cover context), then generate_audio calls it again (the same), and a
        # third time when audio_cover_strength < 1 (the context of the steps after the switch). The base recorder keeps the last
        # call: keep the first as `context`, `encoder.*`, and the last of three as noncover.*.
        inner = h.model.prepare_condition

        def run(*args, **kw):
            out = inner(*args, **kw)
            self.calls += 1
            rt = self.rec.t
            snap = {k: rt[k] for k in ("encoder.hidden", "encoder.mask", "context") if k in rt}
            if self.calls == 1:
                self.first = snap
            self.last = snap
            return out
        h.model.prepare_condition = run

    def reset(self):
        self.rec.reset()
        self.t = self.rec.t
        self.calls = 0
        self.first = self.last = {}

    def finish(self):
        """After a generation: the first call's tensors are the cover's; with three calls the last are the non-cover's."""
        rt = self.rec.t
        if self.calls >= 3:
            rt["noncover.hidden"], rt["noncover.mask"], rt["noncover.context"] = self.last["encoder.hidden"], self.last["encoder.mask"], self.last["context"]
        rt.update(self.first)


def capture_tasks(a):
    import json
    import random
    import numpy as np
    import soundfile as sf
    from safetensors.torch import save_file
    h, root = open_handler(a.model_dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import acestep_ref as base
    from acestep.core.generation.handler import vae_encode
    vae_encode.get_gpu_memory_gb = lambda: 16.0       # a GPU with memory to spare picks upstream's 30 s chunk; with none it picks 15 s, a choice about memory
    rec = AudioRecorder(h, base)
    names = list(TASKS) if not a.cases else a.cases.split(",")
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        sys.exit(f"unknown case(s) {unknown}; known: {list(TASKS)}")
    os.makedirs(a.outdir, exist_ok=True)
    real_randint = random.randint
    # upstream picks the 10 s segments of a reference at random (unseeded); the engine takes the middle of each range
    random.randint = lambda lo, hi: lo + (hi - lo) // 2
    for name in names:
        params, audio = TASKS[name]
        d = os.path.join(a.outdir, name)
        os.makedirs(d, exist_ok=True)
        rec.reset()
        seeded_posterior(COMMON["seed"] ^ VAE_SEED_XOR)
        kw = {**COMMON, **params}
        meta = {"case": name, "params": dict(kw)}
        task = kw.get("task_type", "text2music")
        kw["instruction"] = TASK_INSTRUCTIONS[task]
        for role, (stem, seconds) in audio.items():
            w = clip_of(a.wavdir, stem, seconds)
            path = os.path.join(d, f"{role}.wav")
            sf.write(path, np.clip(w.numpy().T, -1.0, 1.0), S48, subtype="PCM_16")
            # what the handler reads back is the 16-bit file as float32: that is the oracle's input too
            kw["src_audio" if role == "src" else "reference_audio"] = path
            meta[role] = {"stem": stem, "seconds": seconds, "samples": int(w.shape[-1])}
        t0 = __import__("time").time()
        res = h.generate_music(**kw)
        if not res.get("success"):
            sys.exit(f"{name}: {res.get('error')}")
        rec.finish()
        wave = base.clean(res["audios"][0]["tensor"]).float()
        tensors = dict(rec.rec.t)
        tensors["audio"] = wave
        save_file({k: (v.to(torch.uint8) if v.dtype == torch.bool else v) for k, v in tensors.items()},
                  os.path.join(d, "ref.safetensors"), metadata={"case": name})
        meta["shapes"] = {k: list(v.shape) for k, v in tensors.items() if not k.startswith("dit") or k.startswith("dit0")}
        meta["recorded"] = dict(rec.rec.meta)
        with open(os.path.join(d, "ref.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, default=str)
        sf.write(os.path.join(d, "audio.wav"), np.clip(wave.numpy().T, -1.0, 1.0), S48, subtype="PCM_16")
        print(f"{name}: {len([k for k in tensors if k.endswith('.xt')])} DiT calls, audio {tuple(wave.shape)} "
              f"{wave.shape[-1] / S48:.2f} s, peak {wave.abs().max():.3f}, tensors {sorted(k for k in tensors if k.startswith(('tok', 'detok', 'in.')))[:6]}... ({__import__('time').time() - t0:.0f} s)")
    random.randint = real_randint


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks")
    t.add_argument("model_dir")
    t.add_argument("outdir")
    t.add_argument("wavdir")
    t.add_argument("--cases", help="comma-separated; default all")
    e = sub.add_parser("enc")
    e.add_argument("model_dir")
    e.add_argument("outdir")
    e.add_argument("wavdir")
    e.add_argument("--f64", action="store_true", help="also record the one-pass parameters from the same network in float64")
    e.add_argument("--only", help="comma-separated case names")
    a = ap.parse_args()
    if a.cmd == "enc":
        capture_enc(a)
    elif a.cmd == "tasks":
        capture_tasks(a)


if __name__ == "__main__":
    main()
