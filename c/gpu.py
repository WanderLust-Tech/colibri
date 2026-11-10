"""Where a model runs, the CPU or the graphics card, decided each time the router loads it (docs/gpu-offload.md).

A `VK=1` engine (gemma4, qwen36, ...) runs on the card when COLI_VULKAN=1 is in its environment and on the CPU otherwise, so
one binary does both. The router sets it for each worker it starts, from how much memory the card has free at that moment:

    coli router --gpu auto        free memory decides, each time a model loads (the default is not to touch the environment)
    coli router --gpu on          always the card (a model that does not fit stays partly on the CPU, the engines see to that)
    coli router --gpu off         never

    --gpu-reserve-gb 6            memory left free for everything else, the user's renders and games (default 6)
    --gpu-min-gb 3                the least that is worth starting on the card, beyond the reserve (default 3)

In `auto` a model goes to the card when (free - reserve) >= min, and to the CPU otherwise. Either way the engines are told the
reserve (COLI_VK_TIER_RESERVE_GB for the expert tier, G4_VK_RESERVE_GB for Gemma's dense matrices), so what they place on the card
leaves it free. A model keeps its choice until it unloads: the next load looks again. `model-settings.json` can pin one model to
the card or to the CPU (`coli model gpu NAME vulkan|cpu|auto`).

Free memory is read with nvidia-smi (NVIDIA cards, including on Windows); where there is none, `auto` leaves the environment
alone and says so. COLI_GPU_FAKE_FREE_GB=N stands in for the reading (tests, and trying a policy out).
"""
import os
import subprocess
import threading
import time

MODES = ("off", "auto", "on")
OVERRIDES = ("auto", "vulkan", "cpu")
DEFAULT_RESERVE_GB = 6.0
DEFAULT_MIN_GB = 3.0
CACHE_SECONDS = 1.5


def read_memory():
    """(free_gb, total_gb) of the first NVIDIA card in GiB, or None when there is no reading."""
    fake = os.environ.get("COLI_GPU_FAKE_FREE_GB")
    if fake:
        try:
            free = float(fake)
            return free, max(free, float(os.environ.get("COLI_GPU_FAKE_TOTAL_GB") or free))
        except ValueError:
            return None
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5, check=False,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    try:
        free, total = done.stdout.strip().splitlines()[0].split(",")
        return float(free) / 1024.0, float(total) / 1024.0
    except (ValueError, IndexError):
        return None


class Decision:
    """What a worker is told about the card, and why."""

    def __init__(self, mode, why, free_gb=None, env=None):
        self.mode = mode                         # "vulkan", "cpu", or "unchanged" (the environment is left as it was)
        self.why = why
        self.free_gb = free_gb
        self.env = dict(env or {})

    def summary(self):
        where = {"vulkan": "on the graphics card", "cpu": "on the CPU", "unchanged": "as the environment says"}[self.mode]
        return f"{where} ({self.why})"

    def public(self):
        out = {"mode": self.mode, "why": self.why}
        if self.free_gb is not None:
            out["free_gb"] = round(self.free_gb, 1)
        return out


class GpuPolicy:
    def __init__(self, mode=None, reserve_gb=DEFAULT_RESERVE_GB, min_gb=DEFAULT_MIN_GB, probe=None):
        if mode is not None and mode not in MODES:
            raise ValueError(f"--gpu must be one of {', '.join(MODES)}, not {mode!r}")
        if reserve_gb < 0 or min_gb < 0:
            raise ValueError("the GPU reserve and minimum cannot be negative")
        self.mode, self.reserve_gb, self.min_gb = mode, float(reserve_gb), float(min_gb)
        self._probe = probe or read_memory
        self._seen, self._at = None, 0.0
        self._lock = threading.Lock()

    def memory(self):
        """(free_gb, total_gb) now, read at most every second or so."""
        with self._lock:
            if time.monotonic() - self._at > CACHE_SECONDS:
                self._seen, self._at = self._probe(), time.monotonic()
            return self._seen

    def _vulkan_env(self):
        reserve = f"{self.reserve_gb:g}"
        return {"COLI_VULKAN": "1", "COLI_VK_TIER_RESERVE_GB": reserve, "G4_VK_RESERVE_GB": reserve}

    def decide(self, entry=None, override=None):
        """The Decision for one model about to load. `override` is its pinned setting: vulkan, cpu, auto, or None."""
        if override == "cpu":
            return Decision("cpu", "pinned to the CPU", env={"COLI_VULKAN": "0"})
        if override == "vulkan":
            seen = self.memory()
            return Decision("vulkan", "pinned to the card", seen[0] if seen else None, self._vulkan_env())
        if self.mode is None:
            return Decision("unchanged", "no --gpu policy")
        if self.mode == "off":
            return Decision("cpu", "--gpu off", env={"COLI_VULKAN": "0"})
        if self.mode == "on":
            seen = self.memory()
            return Decision("vulkan", "--gpu on", seen[0] if seen else None, self._vulkan_env())
        seen = self.memory()                      # auto
        if seen is None:
            return Decision("unchanged", "no reading of the card's free memory (needs nvidia-smi)")
        free = seen[0]
        usable = free - self.reserve_gb
        if usable >= self.min_gb:
            return Decision("vulkan", f"{free:.1f} GB free, {self.reserve_gb:g} kept back", free, self._vulkan_env())
        return Decision("cpu", f"only {free:.1f} GB free, {self.reserve_gb:g} kept back and {self.min_gb:g} needed", free,
                        {"COLI_VULKAN": "0"})

    def public(self):
        seen = self.memory()
        out = {"mode": self.mode or "unset", "reserve_gb": self.reserve_gb, "min_gb": self.min_gb}
        if seen:
            out["free_gb"], out["total_gb"] = round(seen[0], 1), round(seen[1], 1)
        return out
