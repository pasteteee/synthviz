"""Render notes with Surge XT presets from Python (surgepy, GPL-3).

Setup (once per container, Python 3.13):
    bash /mnt/project-files/synthviz/research/surge/install.sh
Usage:
    from surge_render import render_notes, find_patch
    audio = render_notes(find_patch("Leads/Saw Octaves"), [(0.0, 0.5, 69, 110)], total_s=2.0)  # (n, 2) float64
Notes are (start_s, dur_s, midi, velocity). Overlapping notes are fine (poly); presets with portamento glide.
pitch_bend: optional list of (time_s, value -1..1) for bends (range is set by the patch, often 2 semitones).
"""
from __future__ import annotations
import os, sys
from functools import lru_cache
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import surgepy  # noqa: E402

# use the LOCAL copy made by install.sh: on the network mount Surge init takes ~90 s instead of 0.3 s
DATA = Path("/usr/local/share/surge-xt") if Path("/usr/local/share/surge-xt/patches_factory").exists() else HERE / "data"
SR = 48000


def find_patch(name: str) -> str:
    """'Leads/Saw Octaves' -> factory path; '3rd:Vospi/...' style names: pass a relative path under data/."""
    for root in ("patches_factory", "patches_3rdparty", ""):
        p = DATA / root / f"{name}.fxp"
        if p.exists():
            return str(p)
    hits = sorted(DATA.glob(f"patches_*/**/{Path(name).name}.fxp"))
    if not hits:
        raise FileNotFoundError(name)
    return str(hits[0])


def list_patches(category: str | None = None):
    out = []
    for p in sorted(DATA.glob("patches_*/**/*.fxp")):
        rel = str(p.relative_to(DATA))[:-4]
        if category is None or f"/{category}/" in "/" + rel + "/":
            out.append(rel)
    return out


@lru_cache(maxsize=4)
def _synth(sr):
    return surgepy.createSurge(float(sr))


def render_notes(patch: str, notes, total_s: float, sr: int = SR, tail_s: float = 0.0,
                 pitch_bend=None, cc=None, channel: int = 0):
    s = surgepy.createSurge(float(sr))
    s.loadPatch(patch)
    bs = s.getBlockSize()
    n_blocks = int(np.ceil((total_s + tail_s) * sr / bs))
    ev = []
    for (t, d, m, v) in notes:
        ev.append((int(t * sr), 1, int(m), int(v)))
        ev.append((int((t + d) * sr), 0, int(m), 0))
    for (t, val) in (pitch_bend or []):
        ev.append((int(t * sr), 2, int(np.clip((val + 1) * 8192, 0, 16383)), 0))
    for (t, num, val) in (cc or []):
        ev.append((int(t * sr), 3, int(num), int(val)))
    ev.sort(key=lambda e: (e[0], e[1]))   # note-offs before note-ons at the same sample
    out = np.zeros((2, n_blocks * bs), dtype=np.float32)
    # warm-up: some patches need a few blocks after load
    for _ in range(8):
        s.process()
    i = 0
    for b in range(n_blocks):
        end = (b + 1) * bs
        while i < len(ev) and ev[i][0] < end:
            _, kind, a, v = ev[i]
            if kind == 1:
                s.playNote(channel, a, v, 0)
            elif kind == 0:
                s.releaseNote(channel, a, 0)
            elif kind == 2:
                s.pitchBend(channel, a - 8192)
            else:
                s.channelController(channel, a, v)
            i += 1
        s.process()
        out[:, b * bs:end] = s.getOutput()
    return out.T.astype(np.float64)


if __name__ == "__main__":
    print(len(list_patches()), "patches")
    a = render_notes(find_patch("Leads/Saw Octaves"), [(0, 0.5, 69, 110), (0.5, 0.5, 72, 110)], 1.5)
    print("rendered", a.shape, "peak", float(np.abs(a).max()))
