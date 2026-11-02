"""Optional Surge XT layer (GPL-3, via surgepy): thousands of real synth presets for lead, chords, pads,
plucks and arps. If surgepy is not installed the engine silently uses its own synths.

Setup: `bash third_party/surge/install.sh` (or set SYNTHVIZ_SURGE to a folder with surge_render.py,
lib/ and curated_presets.json). Preset data must be on a LOCAL disk (/usr/local/share/surge-xt),
otherwise Surge needs ~90 s to start.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from .dsp import SR

_CANDIDATES = [os.environ.get("SYNTHVIZ_SURGE", ""), str(Path(__file__).resolve().parents[1] / "third_party" / "surge"),
               "/mnt/project-files/synthviz/research/surge"]

# engine style -> curated genre map + preferred pool per role
STYLE_GENRE = {"hardstyle": "hardstyle", "rawstyle": "hardstyle", "heavy_bass": "dubstep_bass_house",
               "dnb_energy": "dnb", "dnb_liquid": "dnb", "future_bass": "future_bass", "synthwave": "synthwave",
               "trance": "trance", "acid": "acid", "ambient_space": "ambient"}
ROLE_POOLS = {
    # role: {genre: [pools]} with a default
    "lead": {"hardstyle": ["lead_supersaw", "lead_supersaw", "lead"], "rawstyle": ["lead_screech"],
             "future_bass": ["lead", "vox"], "_": None},
    "chords": {"future_bass": ["chord_stab", "lead_supersaw", "keys"], "synthwave": ["keys", "brass", "pad"],
               "dnb": ["keys", "pluck"], "_": ["chord_stab", "keys", "pluck"]},
    "pad": {"_": ["pad"]},
    "arp": {"_": ["pluck"]},          # arp_seq presets run their own sequencer (can leave the key)
    "counter": {"_": ["pluck", "keys", "lead"]},
}
USED: set = set()          # presets already used (fill from the channel history to avoid repeats)
_SURGE_SR = 48000


@lru_cache(maxsize=1)
def _mod():
    for c in _CANDIDATES:
        if c and (Path(c) / "surge_render.py").exists() and (Path(c) / "curated_presets.json").exists():
            if not list((Path(c) / "lib").glob("surgepy*.so")):
                continue                     # not installed here (no binary) -> try the next location
            sys.path.insert(0, c)
            try:
                import surge_render  # noqa: F401
                return surge_render, json.loads((Path(c) / "curated_presets.json").read_text())
            except Exception:  # missing libpython / binary mismatch -> fall back to built-in synths
                sys.path.remove(c)
                continue
    return None


def available() -> bool:
    return _mod() is not None


def _pools_for(style, role):
    mod = _mod()
    if not mod:
        return []
    _, cur = mod
    genre = STYLE_GENRE.get(style)
    table = ROLE_POOLS.get(role, {})
    names = table.get(style) or table.get(genre) or table.get("_")
    if names is None and genre:
        names = cur["genres"].get(genre, {}).get(role) or cur["genres"].get(genre, {}).get("lead")
    names = names or ["lead"]
    return [(n, cur["pools"].get(n, [])) for n in names]


def pick(style, role, rng, register=72, sustain_min=None):
    """Choose a preset for `role` (seeded, avoiding USED). Pre-renders a short phrase and rejects
    clipping / silent presets. Returns dict(path, gain_db, name) or None."""
    mod = _mod()
    if not mod:
        return None
    S, _ = mod
    pools = _pools_for(style, role)
    cands = []
    for _, lst in pools:
        cands += [p for p in lst if p["path"] not in USED]
    if not cands:
        return None
    if sustain_min is not None:
        cands = [p for p in cands if p.get("sus", 1) >= sustain_min] or cands
    order = list(range(len(cands)))
    rng.shuffle(order)
    for i in order[:6]:
        p = cands[i]
        try:
            a = S.render_notes(S.find_patch(p["path"]), [(0.0, 0.3, register, 110), (0.3, 0.3, register + 4, 110),
                                                         (0.6, 0.6, register + 7, 110)], 1.4)
        except Exception:
            continue
        g = 10 ** (p["gain_db"] / 20)
        pk, rms = np.abs(a).max() * g, np.sqrt((a ** 2).mean()) * g
        if not np.isfinite(pk) or pk > 1.0 or rms < 10 ** (-40 / 20):
            continue
        USED.add(p["path"])
        return dict(path=p["path"], gain_db=p["gain_db"], name=p["path"].split("/")[-1])
    return None


def render(preset, notes, spb, T, vel_scale=1.0, legato_slides=True):
    """Render engine Notes (beats) with a Surge preset -> (n, 2) at the engine SR."""
    S, _ = _mod()
    ev = []
    ns = sorted(notes, key=lambda x: x.start)
    for i, n in enumerate(ns):
        dur = n.dur
        if legato_slides and i + 1 < len(ns) and ns[i + 1].slide and ns[i + 1].start - n.end() < 0.3:
            dur = ns[i + 1].start - n.start + 0.06     # overlap -> portamento / legato in mono presets
        ev.append((n.start * spb, max(0.03, dur * spb), int(round(n.pitch)), int(np.clip(n.vel * vel_scale * 120, 1, 127))))
    a = S.render_notes(S.find_patch(preset["path"]), ev, T, tail_s=0.0)
    a = resample_poly(a, 147, 160, axis=0) * 10 ** (preset["gain_db"] / 20)
    n = int(T * SR)
    out = np.zeros((n, 2))
    m = min(n, len(a))
    out[:m] = a[:m]
    return np.nan_to_num(out)
