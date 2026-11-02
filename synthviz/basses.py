"""Bass & kick-bass design v5 (research sections 2.1-2.3).

Every bass = clean mono sine SUB (<100 Hz) + a separate MID layer (100 Hz-6 kHz)
that is distorted in series with EQ between stages, OTT'd and kept LOUDER than
the sub (K-weighted), so the bass is felt on phones.

Mid timbres: reese, growl (FM + talking formant), wobble (synced LFO filter),
neuro (wavefold + comb), square (house/synthwave), reverse (hardstyle offbeat).
Plus a proper 808 (pitch punch, glide, long decay, saturation) and the
3-layer hardstyle / rawstyle kick whose tail follows the chord root.
"""
from __future__ import annotations

import math

import numpy as np
import pedalboard as P

from . import mixing as M
from .dsp import SR

MID_KINDS = ["reese", "growl", "wobble", "neuro", "square"]


def m2f(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _saw(f):
    ph = np.cumsum(f) / SR
    return 2 * (ph % 1.0) - 1


def pitch_track(notes, spb, n, glide_s=0.06, legato_gap=0.03, rel=0.012, att=0.004):
    """notes: objects with .start/.dur (beats), .pitch (midi), optional .slide.
    Returns per-sample frequency (Hz) and gate envelope; overlapping/adjacent
    notes glide instead of retriggering."""
    f = np.zeros(n)
    g = np.zeros(n)
    trig = []
    notes = sorted(notes, key=lambda x: x.start)
    prev_end, prev_hz = -1, None
    for nt in notes:
        a = int(nt.start * spb * SR)
        b = min(n, int((nt.start + nt.dur) * spb * SR))
        if b <= a or a >= n:
            continue
        hz = m2f(nt.pitch)
        tt = np.arange(b - a) / SR
        legato = prev_hz is not None and (a - prev_end) < int(legato_gap * SR)
        if legato or getattr(nt, "slide", False) and prev_hz is not None:
            f[a:b] = hz + (prev_hz - hz) * np.exp(-tt / glide_s)
            g[a:b] = 1.0
            if a > 0:
                g[max(0, a - int(0.002 * SR)):a] = np.maximum(g[max(0, a - int(0.002 * SR)):a], 1.0)
        else:
            f[a:b] = hz
            g[a:b] = np.clip(tt / att, 0, 1)
            trig.append(a)
        rl = int(rel * SR)
        g[max(a, b - rl):b] *= np.linspace(1, 0, b - max(a, b - rl))
        prev_end, prev_hz = b, hz
    # hold frequency between notes so the oscillator never jumps to 0 Hz
    last = 55.0
    nz = f > 0
    if nz.any():
        idx = np.where(nz, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        f = f[idx]
        f[: np.argmax(nz)] = f[np.argmax(nz)]
    else:
        f[:] = last
    return f, g, trig


def _formant(x, t, rate_hz, vowels=((700, 1200), (450, 800), (325, 700), (600, 1000))):
    """Talking-bass formant filter: crossfades through vowels at rate_hz."""
    pos = (t * rate_hz) % len(vowels)
    out = np.zeros_like(x)
    for k, (f1, f2) in enumerate(vowels):
        w = np.clip(1 - np.abs(((pos - k + len(vowels) / 2) % len(vowels)) - len(vowels) / 2), 0, 1)
        if not np.any(w):
            continue
        y = M.run(x, [P.PeakFilter(f1, 14, 4.0), P.PeakFilter(f2, 11, 5.0)])
        out += y * w
    return out


def mid_layer(kind, f, t, spb, rng, drive=1.0):
    n = len(f)
    if kind == "reese":
        det = rng.uniform(10, 25)
        x = sum(_saw(f * 2 ** (c / 1200)) for c in (-det, 0, det * 0.93)) / 3
        x = M.run(x, [P.LadderFilter(P.LadderFilter.Mode.LPF24, rng.uniform(1000, 2500), 0.3, 1.2)])
        x = x * (0.75 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.3, 0.9) * t + rng.uniform(0, 6)))
    elif kind == "growl":
        ratio = rng.choice([1.0, 2.0, 0.5])
        lfo_rate = 1 / (spb * rng.choice([0.5, 0.25, 1.0]))
        idx = rng.uniform(2, 6) * (0.55 + 0.45 * np.sin(2 * np.pi * lfo_rate * t))
        ph = 2 * np.pi * np.cumsum(f) / SR
        x = np.sin(ph + idx * np.sin(ph * ratio))
        x = M.hard(x, 8)
        x = _formant(x, t, lfo_rate * rng.choice([0.5, 1.0]))
    elif kind == "wobble":
        rate = 1 / (spb * rng.choice([1.0, 0.5, 0.25, 1 / 3]))
        x = 0.6 * _saw(f) + 0.4 * np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR))
        lfo = 0.5 - 0.5 * np.cos(2 * np.pi * rate * t)
        cut = 180 * (5500 / 180) ** lfo
        x = dsp_svf_stereo(x, cut, 0.55)
    elif kind == "neuro":
        ph = 2 * np.pi * np.cumsum(f) / SR
        k = 2.0 + 3.5 * (0.5 + 0.5 * np.sin(2 * np.pi * t / (spb * rng.choice([0.5, 1.0]))))
        x = np.sin(k * np.sin(ph))
        # comb filter tuned to the note, feedback 0.7
        d = np.maximum(1, (SR / np.maximum(f, 30)).astype(int))
        y = x.copy()
        for i in range(1, 3):
            sh = np.clip(np.arange(n) - d * i, 0, n - 1)
            y += x[sh] * (0.7 ** i)
        x = y / 2.2
    else:  # square / house
        pw = 0.5 + 0.15 * np.sin(2 * np.pi * 0.3 * t)
        ph = (np.cumsum(f) / SR) % 1.0
        x = np.where(ph < pw, 1.0, -1.0) * 0.7 + 0.3 * _saw(f)
        x = M.run(x, [P.LadderFilter(P.LadderFilter.Mode.LPF24, rng.uniform(900, 2200), 0.35, 1.0)])
    # serial distortion with EQ between stages, everything above 100 Hz
    x = M.lr4(x, 100, "high")
    x = M.hard(x / (np.max(np.abs(x)) + 1e-9), 18 + 10 * drive)
    x = M.run(x, [P.PeakFilter(rng.uniform(350, 550), 4.0, 0.8), P.PeakFilter(2500, -3.0, 1.0)])
    x = M.hard(x, 8 + 6 * drive, asym=0.1)
    x = M.run(x, [P.LowpassFilter(rng.uniform(5000, 7500)), P.HighpassFilter(100)])
    x = M.ott(x, 0.5 + 0.3 * min(1, drive))
    return x / (np.max(np.abs(x)) + 1e-9)


def dsp_svf_stereo(x, cut, res):
    from .dsp import svf
    return svf(np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(cut, dtype=np.float64), res, 0)


def render_bass(notes, spb, T, rng, kind="reese", drive=1.0, sub_level=0.55, width=0.3, mid_level=1.0):
    """Sub + driven mid layer -> stereo (n,2). Mid is louder than sub in K-weighting."""
    n = int(T * SR)
    if not notes:
        return np.zeros((n, 2))
    f, g, _ = pitch_track(notes, spb, n, glide_s=0.05)
    t = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(f) / SR
    sub = M.lr4(np.tanh(np.sin(ph) * 1.3), 100, "low")
    mid = mid_layer(kind, f, t, spb, rng, drive)
    sub = sub / (np.max(np.abs(sub)) + 1e-9) * sub_level
    y = (sub + mid * mid_level) * g
    st = np.stack([y, y], axis=1)
    if width:
        hi = M.lr4(st, 300, "high")
        wide = M.run(hi, [P.Chorus(rate_hz=0.4, depth=0.15, centre_delay_ms=8, feedback=0.0, mix=0.5)])
        st = st - hi + hi * (1 - width) + wide * width
    return M.mono_low(st, 110)


def render_808(notes, spb, T, rng, drive=1.0, decay=0.9, punch=7.0):
    """808: sine, +7 st pitch punch on each new hit, glide on legato notes, long
    decay, asymmetric saturation that puts 2f..8f into the 100-500 Hz range."""
    n = int(T * SR)
    if not notes:
        return np.zeros((n, 2))
    f = np.zeros(n); amp = np.zeros(n)
    prev = None
    for nt in sorted(notes, key=lambda x: x.start):
        a, b = int(nt.start * spb * SR), min(n, int((nt.start + nt.dur) * spb * SR))
        if b <= a:
            continue
        hz = m2f(nt.pitch)
        tt = np.arange(b - a) / SR
        if prev is not None and abs(prev[1] - a) < int(0.02 * SR) or getattr(nt, "slide", False) and prev is not None:
            f[a:b] = hz + (prev[0] - hz) * np.exp(-tt / rng.uniform(0.06, 0.12))
            env = amp[a - 1] * np.exp(-tt / decay) if a > 0 else np.exp(-tt / decay)
        else:
            f[a:b] = hz * 2 ** (punch / 12 * np.exp(-tt / 0.012))
            env = np.exp(-tt / decay) * np.clip(tt / 0.002, 0, 1)
        env = env * np.clip(((b - a) / SR - tt) / 0.015, 0, 1)
        amp[a:b] = env
        prev = (hz, b)
    nz = f > 0
    idx = np.where(nz, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    f = f[idx]; f[: np.argmax(nz)] = f[np.argmax(nz)] if nz.any() else 50
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * amp
    grit = M.hard(x, 10 + 10 * drive, asym=0.2)
    grit = M.run(grit, [P.HighpassFilter(110), P.PeakFilter(250, 3, 0.8), P.LowpassFilter(5000)])
    y = M.lr4(x, 120, "low") * 0.8 + grit * (0.7 + 0.4 * drive)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
    return np.stack([y, y], axis=1)


def render_reverse_bass(notes, spb, T, rng):
    """Hardstyle reverse bass on the offbeat: swells to the end of the half-beat, cut before the kick."""
    n = int(T * SR)
    y = np.zeros(n)
    for nt in notes:
        a, b = int(nt.start * spb * SR), min(n, int((nt.start + nt.dur) * spb * SR))
        if b <= a:
            continue
        L = b - a
        tt = np.arange(L) / SR
        hz = m2f(nt.pitch)
        x = sum(_saw(np.full(L, hz * 2 ** (c / 1200))) for c in (-12, 0, 11)) / 3
        sub = np.sin(2 * np.pi * hz * tt)
        env = (tt / (L / SR)) ** 2.2 * np.clip((L / SR - tt) / 0.006, 0, 1)
        y[a:b] += (M.run(x, [P.LowpassFilter(1500)]) * 0.8 + sub * 0.6) * env
    y = M.hard(y, 10) * 0.9
    return np.stack([y, y], axis=1)


# ------------------------------------------------------------------ hardstyle / rawstyle kick
FLAVOURS = {
    # v2 recipe (research/hardstyle_kicks.py): pitch start multiple, glide tau, pre/mid/zaag EQ peaks
    "euphoric": dict(p_start=2.0, p_tau=0.10, p_end=1.0, drive=(24, 14), pre=(250, 8, 0.7), mid=(1100, 5, 1.0),
                     zaag=(2800, 7, 1.2), lp=13000, rev=0.18, tail_env="decay", punch=1.0, raw_dip=0.0),
    "classic": dict(p_start=2.4, p_tau=0.16, p_end=1.0, drive=(28, 18), pre=(300, 9, 0.6), mid=(800, 6, 1.4),
                    zaag=(1600, 7, 4.0), lp=10000, rev=0.12, tail_env="hold", punch=0.8, raw_dip=0.0),
    "raw": dict(p_start=3.0, p_tau=0.05, p_end=0.94, drive=(32, 20), pre=(400, 10, 0.8), mid=(1400, 6, 1.5),
                zaag=(3000, 8, 1.5), lp=14000, rev=0.10, tail_env="swell", punch=1.1, raw_dip=0.10),
    "modern_punchy": dict(p_start=2.0, p_tau=0.07, p_end=1.0, drive=(22, 12), pre=(200, 7, 0.7), mid=(900, 4, 0.9),
                          zaag=(2600, 6, 1.2), lp=13000, rev=0.15, tail_env="decay", punch=1.4, raw_dip=0.0),
}
FLAVOURS["pounder"] = dict(FLAVOURS["euphoric"], pounder=True)   # rave pounder (user reference)
KICK_FLAVOUR = None   # force one flavour for every track (set when the user picks one)


class KickCharacter:
    """One v2 flavour per track, lightly jittered so tracks differ."""

    def __init__(self, rng, raw=False, flavour=None):
        self.raw = raw
        name = flavour or KICK_FLAVOUR or ("raw" if raw else str(rng.choice(["pounder", "pounder", "pounder", "euphoric", "classic", "modern_punchy"])))
        fl = dict(FLAVOURS[name])
        j = lambda v, r: v * rng.uniform(1 - r, 1 + r)
        fl["p_tau"] = j(fl["p_tau"], 0.15)
        fl["pre"] = (j(fl["pre"][0], 0.15),) + fl["pre"][1:]
        fl["mid"] = (j(fl["mid"][0], 0.15),) + fl["mid"][1:]
        fl["zaag"] = (j(fl["zaag"][0], 0.12),) + fl["zaag"][1:]
        self.name, self.fl = name, fl
        self.drive = rng.uniform(3.0, 4.2)
        self.crunch = rng.uniform(0.85, 1.15)
        self.peaks = tuple(f * rng.uniform(0.9, 1.1) for f in (1110, 1400, 2000, 2800))


OS = 4


def clip_os(x, drive_db, kind="hard", asym=0.0):
    """4x oversampled clipper (no aliasing fizz)."""
    from scipy.signal import resample_poly
    u = resample_poly(x, OS, 1) * M.db(drive_db) + asym
    u = np.clip(u, -1, 1) if kind == "hard" else np.tanh(u)
    u = u - (np.clip(asym, -1, 1) if kind == "hard" else np.tanh(asym))
    return resample_poly(u, 1, OS)[: len(x)]


def hardstyle_kick(root_hz, length_s, ch: KickCharacter, seed=0):
    """Hardstyle kick v2: punch + EQ->hard clip->EQ->clip tail gliding 2-3x root -> root, zaag EQ, room, clean sub."""
    fl = ch.fl
    if fl.get("pounder"):
        return kick_pounder(root_hz, length_s, ch.drive, ch.crunch, 1.0, seed, ch.peaks)
    rng = np.random.default_rng(seed)
    n = max(64, int(length_s * SR)); t = np.arange(n) / SR
    fp = 60 + 240 * np.exp(-t / 0.012) + 2500 * np.exp(-t / 0.0015)
    punch = np.sin(2 * np.pi * np.cumsum(fp) / SR) * np.exp(-t / 0.06)
    punch += M.lr4(rng.standard_normal(n), 3000, "high") * np.exp(-t / 0.002) * 0.4
    punch = clip_os(punch, 6, "soft")
    punch = M.run(punch, [P.PeakFilter(120, 3, 1.0), P.PeakFilter(4000, 3, 1.0)])
    body = 200 * np.exp(-t / 0.018)
    glide = root_hz * (fl["p_end"] + (fl["p_start"] - fl["p_end"]) * np.exp(-t / fl["p_tau"]))
    f = glide + body
    if fl["raw_dip"]:
        f = f * (1 - fl["raw_dip"] * np.sin(np.pi * np.clip(t / length_s, 0, 1)) ** 2)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = M.run(np.sin(ph), [P.PeakFilter(*fl["pre"]), P.HighpassFilter(40)])
    x = clip_os(x, fl["drive"][0], "hard")
    x = M.run(x, [P.HighpassFilter(140), P.HighpassFilter(140), P.PeakFilter(*fl["mid"])])
    x = clip_os(x, fl["drive"][1], "hard", asym=0.1)
    x = M.run(x, [P.PeakFilter(*fl["zaag"]), P.HighShelfFilter(3500, 6, 0.7), P.PeakFilter(300, -3, 0.8)])
    x = clip_os(x * 0.35, 6, "soft")
    x = M.run(x, [P.LowpassFilter(fl["lp"]), P.HighpassFilter(30)])
    rise = np.clip((t - 0.012) / 0.02, 0, 1)
    if fl["tail_env"] == "decay":
        shape = 0.55 + 0.45 * np.exp(-t / (length_s * 0.6))
    elif fl["tail_env"] == "swell":
        shape = 0.8 + 0.2 * (t / length_s)
    else:
        shape = np.ones(n)
    cut = np.clip((length_s - t) / 0.010, 0, 1)
    tail = x * rise * shape * cut
    if fl["rev"]:
        wet = M.run(np.stack([tail, tail], 1), [P.Reverb(room_size=0.25, damping=0.6, wet_level=1.0, dry_level=0.0, width=1.0),
                                                P.HighpassFilter(300), P.LowpassFilter(7000)])
        tail = tail + wet.mean(axis=1) * fl["rev"]
    tail = tail - M.lr4(tail, 90, "low")
    sub = M.lr4(np.tanh(1.3 * np.sin(ph)), 90, "low") * cut * np.clip(t / 0.005, 0, 1)
    k = tail + punch * 0.45 * fl["punch"] + sub * 0.6
    return k / (np.abs(k).max() + 1e-9) * 0.95


def kick_pounder(root_hz=41.0, length_s=0.40, drive=3.5, crunch=1.0, width=1.0, seed=0, peaks=(1110, 1400, 2000, 2800)):
    """Rave 'pounder' hardstyle kick (research/hardstyle_kicks.py): flat hard-clipped sine body on the root
    plus a wide resonant 'tok' attack (0-140 ms). Stereo (n, 2)."""
    rng = np.random.default_rng(seed)
    length_s = max(0.08, length_s - 0.006)
    n = int(length_s * SR); t = np.arange(n) / SR
    fb = root_hz * (1 + 0.25 * np.exp(-t / 0.03))
    body = clip_os(np.sin(2 * np.pi * np.cumsum(fb) / SR), 20 * np.log10(drive), "hard")
    grit = clip_os(body * 0.9 + 0.02 * rng.standard_normal(n), 6, "hard")
    body = body + 0.16 * M.run(grit, [P.HighpassFilter(600)])
    body = M.run(body, [P.LowpassFilter(7000)])
    body = body * np.clip((t - 0.06) / 0.035, 0, 1) ** 1.5 * np.clip((length_s - t) / 0.006, 0, 1)
    fa = 100 + 220 * np.exp(-t / 0.012)
    tok = np.sin(2 * np.pi * np.cumsum(fa) / SR) + 0.25 * M.lr4(rng.standard_normal(n), 1500, "high") * np.exp(-t / 0.01)
    tok = clip_os(tok, 18, "hard")
    tok = M.run(tok, [P.HighpassFilter(350), P.HighpassFilter(350)] + [P.PeakFilter(f, 9 * crunch, 3.0) for f in peaks]
                + [P.HighShelfFilter(3000, 10, 0.7), P.LowpassFilter(13000)])
    tok = clip_os(tok * 0.6, 8, "soft")
    aenv = np.exp(-t / 0.06) * (1 - 0.35 * np.exp(-((t - 0.022) / 0.008) ** 2))
    aenv *= np.clip((0.14 - t) / 0.04, 0, 1)
    tok = tok * aenv
    dl, dr_ = int(0.0007 * SR), int(0.0023 * SR)
    L = tok.copy(); R = np.zeros(n); R[dr_:] = tok[:-dr_]; L[dl:] = 0.7 * L[dl:] + 0.5 * tok[:-dl]
    st = np.stack([L, R], 1)
    st = st + M.run(st, [P.Reverb(0.2, 0.4, 1.0, 0.0, 1.0), P.HighpassFilter(400)]) * 0.25 * width
    mid = st.mean(1, keepdims=True); side = st - mid
    st = mid + side * width
    st = np.tanh(st / (np.abs(st).max() + 1e-9) * 5.0) * 1.35 * crunch
    k = body[:, None] + st
    return k / (np.abs(k).max() + 1e-9) * 0.97


def kick_root_hz(root_pc):
    """Chord root -> tail frequency in E1..D#2 (41-78 Hz)."""
    m = 28 + (root_pc - 28) % 12
    return m2f(m), m
