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

# Hand-picked EDM presets (by name) per role and family. Random picks from the whole library gave bells,
# atmospheres and chord-memory presets (one key -> a whole chord) that sound out of key, so only these are used.
F, T3 = "patches_factory/", "patches_3rdparty/"
BIG_SAWS = [F + "Polysynths/Anthemish 1", F + "Polysynths/Anthemish 2", F + "Polysynths/Hugeness", F + "Leads/Saw Octaves",
            F + "Polysynths/Uni Saw FB", T3 + "Argitoth/Leads/Trance Lead 2", T3 + "Inigo Kennedy/Leads/Happy Saws",
            T3 + "Luna/Leads/Alias Supersaw", T3 + "Luna/Leads/96 Osc Supersaw", T3 + "Luna/Leads/JP8K Supersaw",
            T3 + "Lopyt/Polysynths/Supersaw", T3 + "Lopyt/Leads/Hardstyle Lead 1", T3 + "Lopyt/Leads/Hardstyle Lead 2",
            T3 + "Rozzer/Leads/He Never Lost His Hardcore", T3 + "Lopyt/Leads/Saw Lead 1", T3 + "Jacky Ligon/Leads/Saw Swarm",
            T3 + "Xenofish/Leads/Saturn Saw"]
RAW_LEADS = [F + "Leads/Screamer", T3 + "Lopyt/Leads/Hardstyle Lead 1", T3 + "Lopyt/Leads/Hardstyle Lead 2",
             T3 + "Argitoth/Leads/Screaming Saw", T3 + "Dan Maurer/Synths/Chainsaw", T3 + "Inigo Kennedy/Leads/Dirty Hardcore 2",
             T3 + "Lopyt/Leads/Noise Saw", T3 + "Kinsey Dulcet/Leads/Warehouse Lead", F + "Polysynths/Hugeness"]
SYNC_LEADS = [F + "Leads/Sync Lead", F + "Leads/Classic Lead 1", F + "Leads/Classic Lead 2", F + "Leads/Digi Portalead",
              T3 + "Dan Maurer/Leads/Sharp Lead", T3 + "Lopyt/Leads/Vintage Lead", T3 + "Vospi/Leads/Open Retro Lead",
              T3 + "Lopyt/Leads/Saw Lead 2", T3 + "Lopyt/Leads/Leady", T3 + "Kyurumi/Leads/Normal Lead"]
FB_LEADS = [T3 + "Lopyt/Leads/San Holo Lead", T3 + "Lopyt/Plucks/Supersaw Pluck", F + "Plucks/Nice Pluck 3",
            F + "Leads/Vocal Lead", T3 + "Lopyt/Voices/Vocal-like Thing", F + "Polysynths/Anthemish 1", F + "Plucks/Sync Pluck"]
RETRO_LEADS = [F + "Leads/Classic Lead 1", F + "Leads/Classic Lead 2", T3 + "Vospi/Leads/Open Retro Lead",
               T3 + "Lopyt/Leads/Vintage Lead", T3 + "Luna/Leads/Genesis-y Lead", F + "Brass/Synth Brass 1", F + "Brass/Toto Brass",
               F + "Brass/JX-10 Double Brass", T3 + "Cybersoda/Brass/Soft 80s Brass", T3 + "Luna/Brass/Analog Brass"]
PLUCKS = [F + "Plucks/Nice Pluck 1", F + "Plucks/Nice Pluck 2", F + "Plucks/Nice Pluck 5", F + "Plucks/Saw Pluck",
          F + "Plucks/Sync Pluck", T3 + "Damon Armani/Plucks/Trance Pluck", T3 + "Stefan Singer/Plucks/Trance Pluck",
          T3 + "Vincent Zauhar/Plucks/Super Pluck", T3 + "Rozzer/Plucks/JP80x0 Pluck", T3 + "Lopyt/Plucks/Supersaw Pluck",
          T3 + "Lopyt/Plucks/Another Pluck", T3 + "Vospi/Plucks/Popcorn Pluck", T3 + "Lopyt/Polysynths/Synth Keys"]
KEYS = [T3 + "Luna/Keys/Rave Keys", T3 + "Vincent Zauhar/Keys/Viva La Trance", T3 + "Bluelight/Keys/Digi Piano",
        T3 + "Giana Brotherz/Keys/Glassy E-Piano", T3 + "Luna/Keys/FM Electric Piano", T3 + "Aleksey Zhehanov/Keys/Grand Piano",
        T3 + "Lopyt/Polysynths/Synth Keys"]
STABS = [F + "Polysynths/Anthemish 2", F + "Polysynths/Uni Saw FB", T3 + "Lopyt/Plucks/Supersaw Pluck", F + "Plucks/Saw Pluck",
         T3 + "Damon Armani/Plucks/Trance Pluck", T3 + "Luna/Keys/Rave Keys", F + "Brass/Synth Brass 1"]
FLOW_LEADS = [T3 + "Lopyt/Leads/Vintage Lead", T3 + "Vospi/Leads/Open Retro Lead", T3 + "Luna/Leads/Genesis-y Lead",
              T3 + "Cybersoda/Brass/Soft 80s Brass", T3 + "Luna/Brass/Analog Brass", T3 + "Kyurumi/Leads/Normal Lead",
              F + "Leads/Saw Octaves", T3 + "Luna/Leads/Alias Supersaw", T3 + "Luna/Leads/JP8K Supersaw", F + "Leads/Sync Lead",
              F + "Leads/Vocal Lead", T3 + "Lopyt/Voices/Vocal-like Thing", F + "Brass/Synth Brass 1", T3 + "Inigo Kennedy/Leads/Happy Saws"]
AZ = T3 + "Aleksey Zhehanov/"
# smooth, singing brass/wind leads for calm melodic tracks
SOFT_BRASS = [T3 + "Cybersoda/Brass/Soft 80s Brass", T3 + "Luna/Brass/Analog Brass", F + "Brass/Synth Brass 1",
              F + "Brass/Toto Brass", F + "Brass/JX-10 Double Brass", AZ + "Brass/Flugelhorn", AZ + "Brass/French Horn Ensemble",
              AZ + "Brass/Euphonium", AZ + "Brass/Trumpet Ensemble", AZ + "Winds/Alto Sax MW+AT", AZ + "Winds/Lyrical Flute",
              T3 + "Bluelight/Brass/Horns 1", T3 + "Cybersoda/Brass/Synth Trumpet"]
ALLOW = {
    "flowlead": {"_": FLOW_LEADS},
    "brasslead": {"_": SOFT_BRASS},
    "lead": {"hs": BIG_SAWS, "rawstyle": RAW_LEADS, "bass": SYNC_LEADS + FB_LEADS[:3], "dnb": SYNC_LEADS + PLUCKS[:6],
             "dnb_liquid": KEYS + PLUCKS[:6] + FB_LEADS[:2], "fb": FB_LEADS + BIG_SAWS[:4], "sw": RETRO_LEADS,
             "trance": BIG_SAWS, "acid": SYNC_LEADS},
    "big": {"_": BIG_SAWS},
    "chords": {"fb": STABS, "sw": KEYS + RETRO_LEADS[5:], "dnb": KEYS + PLUCKS, "_": STABS + PLUCKS[:5]},
    "arp": {"_": PLUCKS},
    "counter": {"_": PLUCKS + KEYS},
}
FAMILY_OF = {"hardstyle": "hs", "rawstyle": "rawstyle", "heavy_bass": "bass", "dnb_energy": "dnb", "dnb_liquid": "dnb_liquid",
             "future_bass": "fb", "synthwave": "sw", "trance": "trance", "acid": "acid", "ambient_space": "sw"}
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


def _meta():
    _, cur = _mod()
    out = {}
    for lst in cur["pools"].values():
        for p in lst:
            out[p["path"]] = p
    return out


def _allowed(style, role):
    fam = FAMILY_OF.get(style, "sw")
    t = ALLOW.get(role, {})
    names = t.get(style) or t.get(fam) or t.get("_") or []
    meta = _meta()
    out = []
    for n in names:
        p = meta.get(n)
        if p is None and role != "pad":
            p = {"path": n, "gain_db": 0.0}
        if p and p not in out:
            out.append(p)
    if role == "pad":   # pads: sustained, mid-bright, no atmospheres / drones / chord memory
        bad = ("Atmos", "Drone", "FX", "Chord", "Noise", "Texture", "Sweep", "Rhythm", "Sequence")
        cur = _mod()[1]
        out = [p for p in cur["pools"].get("pad", [])
               if not any(b_ in p["path"] for b_ in bad) and 0.6 <= p.get("sus", 0) <= 2.0 and 500 <= p.get("cen", 0) <= 3000]
    return out


def pure_pitch(a, midi, sr=48000):
    """Share of 200-5000 Hz energy that sits on the harmonics of `midi` (1.0 = one clean in-tune note).
    Chord-memory presets, bells and detuned FX score low."""
    x = a.mean(1) if a.ndim == 2 else a
    x = x[int(0.05 * sr): int(0.6 * sr)]
    if len(x) < 2048 or not np.any(x):
        return 0.0
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    band = (f > 200) & (f < 5000)
    h = np.zeros_like(f, dtype=bool)
    for k in range(1, int(5000 / f0) + 1):
        h |= np.abs(f - k * f0) < f0 * 0.03 + 3
    return float(X[band & h].sum() / (X[band].sum() + 1e-12))


def pick(style, role, rng, register=72, sustain_min=None):
    """Choose an allow-listed preset for `role` (seeded, avoiding USED). Pre-renders single notes and rejects
    clipping, silent or impure presets (chord memory, inharmonic, out of tune). Returns dict(path, gain_db, name) or None."""
    mod = _mod()
    if not mod:
        return None
    S, _ = mod
    cands = [p for p in _allowed(style, role) if p["path"] not in USED] or _allowed(style, role)
    order = list(range(len(cands)))
    rng.shuffle(order)
    for i in order[:10]:
        p = cands[i]
        try:
            a = S.render_notes(S.find_patch(p["path"]), [(0.0, 0.7, register, 110)], 1.0)
        except Exception:
            continue
        g = 10 ** (p.get("gain_db", 0.0) / 20)
        pk, rms = np.abs(a).max() * g, np.sqrt((a ** 2).mean()) * g
        if not np.isfinite(pk) or rms < 10 ** (-45 / 20):
            continue
        if p.get("gain_db", 0.0) == 0.0 and rms > 0:      # not in the measured pools: normalise to -18 dBFS RMS
            g = 10 ** (-18 / 20) / rms
            p = dict(p, gain_db=float(20 * np.log10(g)))
        if pure_pitch(a, register) < 0.55:
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
