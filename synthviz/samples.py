"""CC0 one-shot samples (Sonic Pi set) used as layers on top of the synthesized drums."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .dsp import SR

ROOT = Path(__file__).resolve().parent / "samples"
DIR = ROOT / "sonicpi"

KICK_PUNCH = ["bd_tek", "drum_heavy_kick", "bd_haus", "bd_zome"]
SNARE_TOP = ["sn_dolf", "sn_zome", "elec_hi_snare", "drum_snare_hard", "sn_dub", "elec_snare"]
HATS = ["hat_zild", "hat_cab", "hat_gem", "hat_star", "hat_hier", "hat_zap", "drum_cymbal_closed"]
OPEN_HATS = ["drum_cymbal_open", "hat_cats", "hat_len"]
CRASH = ["drum_cymbal_hard", "drum_splash_hard"]
TOMS = ["drum_tom_hi_hard", "drum_tom_mid_hard", "drum_tom_lo_hard"]
SNAPS = ["perc_snap", "perc_snap2"]
# TR-808 (Michael Fischer, CC0) and Kickmess kicks (Meadowlark, CC0): names with a folder are paths under samples/
K808_SNARE = ["tr808/sd8/SD" + t + s for t in ("00", "10", "25", "50", "75") for s in ("25", "50", "75")]
K808_CLAP = ["tr808/cp8/CP"]
K808_HAT = ["tr808/ch8/CH"]
K808_OHAT = ["tr808/oh8/OH25", "tr808/oh8/OH50", "tr808/oh8/OH75"]
K808_COWBELL = ["tr808/cb8/CB"]
KICK_LAYER = (["meadowlark/Snappy/Kickmess Snappy %d" % i for i in range(1, 8)]
              + ["meadowlark/Distorted/Kickmess Distorted %d" % i for i in range(1, 4)]
              + ["meadowlark/Deep/Kickmess Deep %d" % i for i in range(1, 8)])


@lru_cache(maxsize=None)
def load(name: str) -> np.ndarray:
    """Mono float64 at the engine sample rate, peak-normalised."""
    from pedalboard.io import AudioFile
    if "/" in name:
        base = ROOT / name
        path = next((base.with_name(base.name + e) for e in (".WAV", ".wav", ".flac") if base.with_name(base.name + e).exists()), None)
        if path is None:
            raise FileNotFoundError(name)
    else:
        path = DIR / f"{name}.flac"
    with AudioFile(str(path)) as f:
        x = f.read(f.frames)
        sr = int(f.samplerate)
    x = x.mean(axis=0).astype(np.float64)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    x = x - np.mean(x[: min(len(x), 64)]) * 0
    pk = np.max(np.abs(x)) + 1e-9
    return x / pk


def available():
    return sorted(p.stem for p in DIR.glob("*.flac"))


def pitched(name: str, ratio: float) -> np.ndarray:
    """Varispeed (tape-style) pitch shift."""
    x = load(name)
    if abs(ratio - 1) < 1e-3:
        return x
    idx = np.arange(0, len(x) - 1, ratio)
    return np.interp(idx, np.arange(len(x)), x)


def trimmed(name: str, length_s: float, fade_s: float = 0.01) -> np.ndarray:
    x = load(name)[: int(length_s * SR)].copy()
    f = min(len(x), int(fade_s * SR))
    if f:
        x[-f:] *= np.linspace(1, 0, f)
    return x
