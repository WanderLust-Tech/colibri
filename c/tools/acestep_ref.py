#!/usr/bin/env python3
"""Reference outputs of ACE-Step 1.5 turbo, captured from upstream's own handler, for the C engine's oracles.

    acestep_ref.py capture MODEL_DIR OUTDIR [--cases inst10,odd9_nodcw] [--check-determinism]
    acestep_ref.py cases

MODEL_DIR is a checkpoint in the layout of the Hugging Face repository ACE-Step/Ace-Step1.5 (the real one, or the
tiny fixture of make_acestep_tiny.py). The handler is upstream's, run as upstream runs it (text2music, no planner,
CPU, float32: the bf16 weights are widened exactly, which is what the C engine does), and every stage is observed
through hooks rather than reimplemented, so the numbers cannot drift from the real pipeline:

    OUTDIR/<case>/ref.safetensors   f32 (ids keep their type; masks, bool in torch, are stored as U8: 1 or 0)
        text.ids, text.hidden        caption tokens and Qwen3-Embedding's last_hidden_state
        lyric.ids, lyric.hidden      lyric tokens and their embedding-table rows (the lyric path uses no layers)
        encoder.hidden, encoder.mask condition encoder output: text, lyric and timbre tokens, packed
        context                      src_latents ++ chunk masks, the DiT's context_latents
        noise                        the initial latents prepare_noise drew from the seed
        in.*                         every tensor handed to generate_audio (silence latent, masks, timbre reference)
        dit{i}.xt, .t, .r, .v        the i-th DiT call: latents in, t, r, velocity out
        dit0.rope_cos, .rope_sin     the DiT's RoPE table of the first call, [1, tokens, head_dim]
        dit0.tap.proj_in, .layer{k}  with --taps: the first call's hidden state after the patch embedding and each layer
        text.tap.embed, .layer{k}    with --taps: the text encoder's first hidden state and each of its layers' outputs,
        text.rope_cos, .rope_sin     and its RoPE table, for the caption pass
        dit{i}.temb_t, .proj_t       its timestep embedding of t ([1, D]) and the six modulation vectors ([1, 6, D]);
        dit{i}.temb_r, .proj_r       and the same for t - r (the second embedding, which is of 0 in turbo)
        latents                      generate_audio's target_latents
        vae{k}.in, vae{k}.out        the k-th VAE decode: [1, 64, frames] in, [1, 2, samples] out
        audio                        what the handler returns: [2, samples] f32 at 48 kHz
    OUTDIR/<case>/ref.json          the case's settings, the versions, upstream's commit, the config hash
    OUTDIR/<case>/audio.wav         the audio as 16-bit PCM, to listen to

Each case has its own directory, as the other engines' oracles do, so the C loader (st.h reads a directory)
opens one case without meeting the tensor names of the others.

Reproducibility, measured: on the real checkpoint a second capture in a new process is bit-identical in every
tensor (inst10 and long30 compared, the latter with its six VAE chunks). On the tiny fixture the DiT stages and
the latents are bit-identical too, but the VAE output of a multi-chunk clip moves by up to 1e-4 (the waveform
peaks near 1.6) between processes: a small convolution picks its reduction order by a run's thread timing. So a
tolerance for the tiny VAE oracle below about 2e-4 would fail the reference against itself.

The tool does not change MODEL_DIR. Upstream's handler would download whatever it thinks is missing (it counts the
1.7B planner as part of the "main model") and overwrite the DiT's python files; both are turned off here, and the
directory is fingerprinted before and after so a change is reported. The DiT's python files must already be
upstream's (make_acestep_tiny.py copies them; for the real checkpoint run upstream once, or copy
acestep/models/turbo/*.py into MODEL_DIR/acestep-v15-turbo).

Needs the reference venv (tools/requirements-acestep-ref.txt) and ACESTEP_UPSTREAM, a clone of
https://github.com/ace-step/ACE-Step-1.5. Run with PYTHONUTF8=1 (upstream prints emoji).
"""
import argparse
import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch

# name -> generate_music arguments. dcw_enabled is always explicit: upstream turns DCW on for turbo, so the cases
# without a suffix are the product path; the _nodcw twins isolate the sampler (the shipped modeling file has no
# DCW). Do not read the audio level of a _nodcw case as quality: on the real checkpoint turbo without DCW often
# collapses to the silence latent (shift3_nodcw and odd9_nodcw are 0.001 rms, the floor of a decoded silence), which
# is upstream's behaviour and what the port has to reproduce, but it makes the final waveform a weak check there;
# the per-step velocities (dit{i}.v) are what holds the sampler to account.
COMMON = dict(inference_steps=8, use_random_seed=False, seed=42, batch_size=1, audio_duration=10.0)
INSTRUMENTAL = dict(captions="calm piano with soft strings", lyrics="[Instrumental]", vocal_language="unknown")
VOCAL = dict(captions="upbeat pop with bright synths",
             lyrics="[Verse 1]\nwalking down the empty street\nthe city sleeps beneath my feet\n[Chorus]\nsing it out loud tonight",
             vocal_language="en")
CASES = {}
for _name, _extra in (
        ("inst10", {}),                                      # 250 latent frames, 125 DiT tokens, one VAE chunk
        ("vocal10", VOCAL),                                  # real lyrics: more lyric tokens, a language tag
        ("odd9", {"audio_duration": 9.0}),                   # 225 frames: odd, so the DiT pads and the DWT sees an odd length
        ("shift3", {"shift": 3.0}),                          # the shift-3 schedule table (the default is shift 1)
        ("steps5", {"inference_steps": 5}),                  # a linspace schedule, not a table
):
    for _dcw in (True, False):
        CASES[_name + ("" if _dcw else "_nodcw")] = {**COMMON, **INSTRUMENTAL, **_extra, "dcw_enabled": _dcw}
# 750 latent frames: the VAE decodes it in overlapping chunks (upstream's chunk is 256 frames, overlap 64), and the
# DiT's sliding window of 128 has a long sequence to cut across.
CASES["long30"] = {**COMMON, **INSTRUMENTAL, "audio_duration": 30.0, "dcw_enabled": True}
# What the text side must get right, through the whole pipeline (the tokenizer has its own much longer list of strings,
# tools/make_acestep_tok_ref.py): the metas formatting with every field given, other scripts and an emoji, a prompt cut
# at the handler's 256-token limit (and lyrics at 2048), and no text at all.
CASES["meta10"] = {**COMMON, **INSTRUMENTAL, "bpm": 120, "key_scale": "C Major", "time_signature": "4", "dcw_enabled": True}
CASES["cjk10"] = {**COMMON, "captions": "ミニマルなテクノ、深いベース、暗い雰囲気 🎧",
                  "lyrics": "[Verse]\n你好，世界\n我们一起唱歌\n[Chorus]\nla la la 1 2 3", "vocal_language": "zh", "dcw_enabled": True}
CASES["cut10"] = {**COMMON, "captions": "a slow cinematic piece with " + ", ".join(f"layered texture number {i}" for i in range(90)),
                  "lyrics": "\n".join(f"[Verse {i}]\nline one of verse {i}\nline two of verse {i}" for i in range(400)),
                  "vocal_language": "en", "dcw_enabled": True}
CASES["empty10"] = {**COMMON, "captions": "", "lyrics": "", "vocal_language": "unknown", "dcw_enabled": True}
# 9.6 s is 240 latent frames, and upstream writes "9 seconds" in the metas (int() cuts, it does not round)
CASES["frac96"] = {**COMMON, **INSTRUMENTAL, "audio_duration": 9.6, "dcw_enabled": True}


def fingerprint(root):
    """{relative path: (size, mtime_ns)} for every file below root, to see whether a run changed anything."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
    return out


def upstream_commit(root):
    try:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def clean(t):
    """A snapshot: the sampler updates its latents in place, and safetensors refuses tensors that share storage."""
    return t.detach().cpu().clone().contiguous()


class Recorder:
    """Hooks on one initialised handler that collect every stage's tensors for the current case."""

    def __init__(self, h, taps=False):
        self.h, self.t, self.meta, self.taps = h, {}, {}, taps
        self.dit_calls = self.vae_calls = 0
        self.inside_text = False
        enc = h.text_encoder
        enc.register_forward_pre_hook(self._text_pre, with_kwargs=True)
        enc.register_forward_hook(self._text_post)
        enc.embed_tokens.register_forward_hook(self._embed)
        h.model.decoder.register_forward_hook(self._dit, with_kwargs=True)
        self.temb_calls = {"t": 0, "r": 0}
        h.model.decoder.rotary_emb.register_forward_hook(self._rope)
        h.model.decoder.time_embed.register_forward_hook(lambda m, a, o: self._temb("t", o))
        h.model.decoder.time_embed_r.register_forward_hook(lambda m, a, o: self._temb("r", o))
        if taps:        # the hidden state after the patch embedding and after every DiT layer, of the first call only
            h.model.decoder.proj_in.register_forward_hook(lambda m, a, o: self._tap("proj_in", o))
            for k, layer in enumerate(h.model.decoder.layers):
                layer.register_forward_hook(lambda m, a, o, k=k: self._tap(f"layer{k}", o[0]))
            # and the text encoder's: each layer's output and its RoPE table, of the caption pass
            for k, layer in enumerate(h.text_encoder.layers):
                layer.register_forward_hook(lambda m, a, o, k=k: self._text_tap(f"layer{k}", o))
            h.text_encoder.rotary_emb.register_forward_hook(
                lambda m, a, o: self.t.update({"text.rope_cos": clean(o[0]), "text.rope_sin": clean(o[1])}) if self.inside_text else None)
        self._wrap(h.model, "prepare_condition", self._prepare_condition)
        self._wrap(h.model, "prepare_noise", self._prepare_noise)
        self._wrap(h.model, "generate_audio", self._generate_audio)
        self._wrap(h.vae, "decode", self._vae_decode)

    @staticmethod
    def _wrap(obj, name, make):
        original = getattr(obj, name)
        setattr(obj, name, make(original))

    def reset(self):
        self.t, self.meta = {}, {}
        self.dit_calls = self.vae_calls = 0
        self.temb_calls = {"t": 0, "r": 0}

    def _rope(self, module, args, output):
        if self.dit_calls == 0:                     # the DiT's RoPE table (cos, sin), [1, tokens, head_dim], of the first call
            self.t["dit0.rope_cos"], self.t["dit0.rope_sin"] = clean(output[0]), clean(output[1])

    def _tap(self, name, out):
        if self.dit_calls == 0:                     # the decoder's own hook has not fired for call 0 yet
            self.t[f"dit0.tap.{name}"] = clean(out)

    def _temb(self, which, out):
        """The timestep embedding of call i (temb [1, D], proj [1, 6, D]): `which` is the embedding of t or of t - r."""
        i = self.temb_calls[which]
        self.temb_calls[which] += 1
        self.t[f"dit{i}.temb_{which}"], self.t[f"dit{i}.proj_{which}"] = clean(out[0]), clean(out[1])

    def _text_pre(self, module, args, kwargs):
        self.inside_text = True
        ids = args[0] if args else kwargs.get("input_ids")
        self.t["text.ids"] = clean(ids)

    def _text_post(self, module, args, output):
        self.inside_text = False
        self.t["text.hidden"] = clean(output.last_hidden_state)

    def _embed(self, module, args, output):
        if not self.inside_text:                       # the lyric path calls the table itself, outside the encoder
            self.t["lyric.ids"], self.t["lyric.hidden"] = clean(args[0]), clean(output)
        elif self.taps:                                # inside the text encoder: its first hidden state
            self.t["text.tap.embed"] = clean(output)

    def _text_tap(self, name, out):
        """A text-encoder layer's output (a tensor, or a tuple whose first element is it, by transformers version)."""
        self.t[f"text.tap.{name}"] = clean(out[0] if isinstance(out, tuple) else out)

    def _dit(self, module, args, kwargs, output):
        i = self.dit_calls
        self.dit_calls += 1
        self.t[f"dit{i}.xt"] = clean(kwargs["hidden_states"])
        self.t[f"dit{i}.t"] = clean(kwargs["timestep"]).reshape(-1)
        self.t[f"dit{i}.r"] = clean(kwargs["timestep_r"]).reshape(-1)
        self.t[f"dit{i}.v"] = clean(output[0])

    def _prepare_condition(self, original):
        def run(*a, **kw):
            hidden, mask, context = original(*a, **kw)
            self.t["encoder.hidden"], self.t["encoder.mask"], self.t["context"] = clean(hidden), clean(mask), clean(context)
            return hidden, mask, context
        return run

    def _prepare_noise(self, original):
        def run(*a, **kw):
            noise = original(*a, **kw)
            self.t.setdefault("noise", clean(noise))
            return noise
        return run

    def _generate_audio(self, original):
        def run(*a, **kw):
            for k, v in kw.items():
                if isinstance(v, torch.Tensor):
                    self.t[f"in.{k}"] = clean(v)
                elif isinstance(v, (bool, int, float, str, list, type(None))):
                    self.meta[f"generate_audio.{k}"] = v
            out = original(*a, **kw)
            self.t["latents"] = clean(out["target_latents"])
            return out
        return run

    def _vae_decode(self, original):
        def run(x, *a, **kw):
            k = self.vae_calls
            self.vae_calls += 1
            out = original(x, *a, **kw)
            self.t[f"vae{k}.in"] = clean(x)
            self.t[f"vae{k}.out"] = clean(getattr(out, "sample", out))
            return out
        return run


def write_wav(path, audio, rate):
    import numpy as np
    import soundfile as sf
    pcm = np.clip(audio.numpy().T, -1.0, 1.0)
    sf.write(path, pcm, rate, subtype="PCM_16")


def capture(a):
    root = os.environ.get("ACESTEP_UPSTREAM")
    if not root or not os.path.isdir(os.path.join(root, "acestep")):
        sys.exit("set ACESTEP_UPSTREAM to a clone of https://github.com/ace-step/ACE-Step-1.5")
    root, model_dir, a.outdir = os.path.abspath(root), os.path.abspath(a.model_dir), os.path.abspath(a.outdir)
    # Upstream keeps timing estimates in a .cache directory relative to where it runs: run it from a scratch
    # directory (every path above is absolute by now) so a capture leaves nothing behind in the caller's.
    caller_cwd, scratch = os.getcwd(), tempfile.mkdtemp(prefix="acestep-ref-")
    os.chdir(scratch)

    def leave_scratch():              # Windows cannot remove a directory that is the working directory
        os.chdir(caller_cwd)
        shutil.rmtree(scratch, ignore_errors=True)
    atexit.register(leave_scratch)
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = model_dir
    sys.path.insert(0, root)
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    from safetensors.torch import save_file
    from acestep.handler import AceStepHandler
    from acestep.model_downloader import _check_code_mismatch

    names = list(CASES) if not a.cases else a.cases.split(",")
    unknown = [n for n in names if n not in CASES]
    if unknown:
        sys.exit(f"unknown case(s) {unknown}; known: {list(CASES)}")
    mismatch = _check_code_mismatch("acestep-v15-turbo", model_dir)
    if mismatch:
        sys.exit(f"{model_dir}/acestep-v15-turbo: {mismatch} differ from upstream's acestep/models/turbo; copy them there "
                 "(this tool will not rewrite a model directory)")

    torch.set_num_threads(os.cpu_count() or 8)
    before = fingerprint(model_dir)
    h = AceStepHandler()
    h._ensure_models_present = lambda **kw: None            # never download: upstream counts the planner as required
    h._sync_model_code_if_needed = lambda *args, **kw: None  # never rewrite the checkpoint's python files
    t0 = time.time()
    msg, ok = h.initialize_service(project_root=root, config_path="acestep-v15-turbo", device="cpu",
                                   use_flash_attention=False, compile_model=False, use_mlx_dit=False)
    if not ok:
        sys.exit(str(msg))
    print(f"initialised in {time.time() - t0:.1f} s ({h.dtype}, {getattr(h.config, '_attn_implementation', '?')})")

    # Upstream turns DCW into a silent no-op, with one log line, when pytorch_wavelets cannot be imported (it needs
    # pkg_resources, so setuptools<72): a capture made that way would say "DCW on" and test nothing.
    from acestep.models.common.dcw_loader import WAVELET_CACHE
    wants_dcw = [n for n in names if CASES[n].get("dcw_enabled")]
    if wants_dcw and WAVELET_CACHE.get(torch.device("cpu"), torch.float32, "haar") is None:
        sys.exit(f"cases {wants_dcw} need DCW, but pytorch_wavelets does not import here (pip install pytorch_wavelets "
                 "PyWavelets 'setuptools<72'); upstream would silently skip the correction")

    with open(os.path.join(model_dir, "acestep-v15-turbo", "config.json"), "rb") as f:
        config_hash = hashlib.sha256(f.read()).hexdigest()
    import diffusers
    import transformers
    versions = {"torch": torch.__version__, "transformers": transformers.__version__, "diffusers": diffusers.__version__,
                "upstream_commit": upstream_commit(root), "dit_config_sha256": config_hash}
    os.makedirs(a.outdir, exist_ok=True)
    rec = Recorder(h, taps=a.taps)

    def run(name):
        rec.reset()
        t = time.time()
        res = h.generate_music(**CASES[name])
        if not res.get("success"):
            sys.exit(f"{name}: {res.get('error')}")
        audio = res["audios"][0]
        rate = int(audio.get("sample_rate", 48000))
        wave = clean(audio["tensor"]).float()
        return res, rec.t.copy(), dict(rec.meta), wave, rate, time.time() - t

    for name in names:
        res, tensors, meta, wave, rate, secs = run(name)
        tensors["audio"] = wave
        if a.check_determinism and name == names[0]:
            _, again, _, wave2, _, _ = run(name)
            same = all(torch.equal(tensors[k], again[k]) for k in tensors if k in again and k != "audio") and torch.equal(wave, wave2)
            print(f"  determinism: {'bit-identical on a second run' if same else 'DIFFERENT on a second run'}")
            if not same:
                sys.exit(1)
        steps = sum(1 for k in tensors if k.startswith("dit") and k.endswith(".xt"))
        case_dir = os.path.join(a.outdir, name)
        os.makedirs(case_dir, exist_ok=True)
        # masks are bool in torch; the C loader (st.h) reads the dtypes its engines use, and bool is not one: U8, 1 or 0
        save_file({k: (v.to(torch.uint8) if v.dtype == torch.bool else v) for k, v in tensors.items()},
                  os.path.join(case_dir, "ref.safetensors"),
                  metadata={"case": name})
        with open(os.path.join(case_dir, "ref.json"), "w", encoding="utf-8") as f:
            json.dump({"case": name, "params": CASES[name], "sample_rate": rate, "dit_calls": steps,
                       "vae_calls": sum(1 for k in tensors if k.startswith("vae") and k.endswith(".in")),
                       "shapes": {k: list(v.shape) for k, v in tensors.items() if not k.startswith("dit") or k.startswith("dit0")},
                       "versions": versions, "recorded": meta}, f, indent=1, default=str)
        write_wav(os.path.join(case_dir, "audio.wav"), wave, rate)
        print(f"{name}: {steps} DiT calls, latents {tuple(tensors['latents'].shape)}, audio {tuple(wave.shape)} "
              f"{wave.shape[-1] / rate:.2f} s, rms {wave.pow(2).mean().sqrt():.4f}, peak {wave.abs().max():.3f} ({secs:.1f} s)")

    after = fingerprint(model_dir)
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    print("model directory unchanged" if not changed else f"WARNING: the run changed {len(changed)} file(s) in MODEL_DIR: {changed[:8]}")
    sys.exit(1 if changed else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("model_dir")
    c.add_argument("outdir")
    c.add_argument("--cases", help="comma-separated; default all (see `cases`)")
    c.add_argument("--taps", action="store_true", help="also record the hidden state after the patch embedding and each DiT layer of the first call (for debugging; 25 MB a case)")
    c.add_argument("--check-determinism", action="store_true", help="run the first case twice and require identical tensors")
    sub.add_parser("cases")
    a = ap.parse_args()
    if a.cmd == "cases":
        for name, p in CASES.items():
            print(f"{name:14s} {json.dumps({k: v for k, v in p.items() if k not in COMMON or COMMON[k] != v})}")
        return
    capture(a)


if __name__ == "__main__":
    main()
