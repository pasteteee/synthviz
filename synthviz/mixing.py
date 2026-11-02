"""Mixing & mastering v5 (see research/SOUND_RESEARCH.md, sections 4 and 6).

- per-stem EQ map (no mud at 300-600 Hz, presence at 2-5 kHz), peak clippers on drums
- loudness-based (LUFS-S) stem levels instead of RMS
- per-bus sidechain duck shapes (LFO-Tool style)
- OTT (multiband up/down compression), tape saturation, parallel crunch on kick/snare
- master: glue 2:1 -> soft clip from 0.75 -> LSP limiter (oversampled) at -1 dBTP
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import pedalboard as P
import pyloudnorm as pyln
from numba import njit
from scipy.signal import butter, sosfilt

from . import dsp
from .dsp import SR

LSP = "/usr/lib/vst3/lsp-plugins.vst3"


def run(x, plugins):
    """Apply a list of pedalboard plugins to (n,) or (n,2) float arrays."""
    mono = x.ndim == 1
    a = (x[None, :] if mono else x.T).astype(np.float32)
    y = P.Pedalboard(plugins)(a, SR)
    y = y[0] if mono else y.T
    return y.astype(np.float64)


def db(x):
    return 10 ** (x / 20)


def lr4(x, fc, kind):
    sos = butter(2, fc, btype=kind, fs=SR, output="sos")
    return sosfilt(sos, sosfilt(sos, x, axis=0), axis=0)


_lr4 = lr4


def split3(x, f1=120.0, f2=2500.0):
    lo = lr4(x, f1, "low")
    rest = x - lo
    mid = lr4(rest, f2, "low")
    return lo, mid, rest - mid


def mono_low(x, fc=120.0):
    if x.ndim == 1:
        return x
    lo = lr4(x, fc, "low")
    return x - lo + lo.mean(axis=1, keepdims=True)


def hard(x, drive_db, asym=0.0):
    return np.tanh(x * db(drive_db) + asym) - np.tanh(asym)


def tape(x, g=2.0):
    """Tape/tube style saturation: tanh(x*g)/tanh(g)."""
    return np.tanh(x * g) / np.tanh(g)


def soft_clip(x, thr=0.75):
    a = np.abs(x)
    return np.where(a < thr, x, np.sign(x) * (thr + (1 - thr) * np.tanh((a - thr) / (1 - thr))))


@contextmanager
def _quiet():
    """LSP plugins print a lot of [WRN] lines on stderr when loaded."""
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    nul = os.open(os.devnull, os.O_WRONLY)
    os.dup2(nul, fd)
    try:
        yield
    finally:
        os.dup2(saved, fd)
        os.close(nul); os.close(saved)


@lru_cache(maxsize=None)
def _lsp(name):
    if not os.path.isdir(LSP):
        return None
    try:
        with _quiet():
            return P.load_plugin(LSP, plugin_name=name)
    except Exception:
        return None


# ------------------------------------------------------------------ envelopes / dynamics
@njit(cache=True)
def _env(a, ka, kr):
    e = np.empty_like(a)
    s = 0.0
    for i in range(a.shape[0]):
        v = a[i]
        k = ka if v > s else kr
        s = k * s + (1 - k) * v
        e[i] = s
    return e


def envelope(x, att=0.003, rel=0.15):
    a = np.abs(x) if x.ndim == 1 else np.abs(x).max(axis=1)
    return _env(a.astype(np.float64), np.exp(-1 / (att * SR)), np.exp(-1 / (rel * SR)))


def ott(x, depth=0.3, f1=88.0, f2=2500.0, down_thr=-30.0, up_thr=-40.0, down_ratio=10.0, up_ratio=3.0,
        time=0.5, gains_db=(0.0, 0.0, 2.0)):
    """Multiband upward+downward compression ('OTT'): dense, bright, detailed."""
    if depth <= 0 or not np.any(x):
        return x
    pk = np.max(np.abs(x)) + 1e-9
    xx = x / pk
    lo = lr4(xx, f1, "low"); rest = xx - lo
    mid = lr4(rest, f2, "low"); hi = rest - mid
    out = 0
    for band, g in zip((lo, mid, hi), gains_db):
        e = 20 * np.log10(envelope(band, 0.003 * time * 2, 0.15 * time * 2) + 1e-9)
        gr = np.where(e > down_thr, (down_thr - e) * (1 - 1 / down_ratio), 0.0)
        gr += np.where(e < up_thr, np.minimum((up_thr - e) * (1 - 1 / up_ratio), 24.0), 0.0)
        gr = db(gr + g)
        out = out + band * (gr if band.ndim == 1 else gr[:, None])
    out = out / (np.max(np.abs(out)) + 1e-9)
    return (xx * (1 - depth) + out * depth) * pk


def duck_curve(n, hits_s, beat_s, depth=1.0, hold=0.0, recover=0.5, shape=2.5, att_ms=2.0):
    """Kickstart/LFO-Tool style volume shape on every kick."""
    g = np.ones(n)
    if depth <= 0:
        return g
    att = max(1, int(att_ms / 1000 * SR)); hl = int(hold * beat_s * SR); rec = max(2, int(recover * beat_s * SR))
    seg = np.concatenate([np.linspace(1, 1 - depth, att), np.full(hl, 1 - depth),
                          1 - depth + depth * np.linspace(0, 1, rec) ** shape])
    for h in hits_s:
        a = int(h * SR)
        if a >= n or a < 0:
            continue
        b = min(n, a + len(seg))
        g[a:b] = np.minimum(g[a:b], seg[: b - a])
    return g


# ------------------------------------------------------------------ loudness
_meter = None


def lufs(x):
    global _meter
    if _meter is None:
        _meter = pyln.Meter(SR)
    return _meter.integrated_loudness(x if x.ndim == 2 else x[:, None])


def lufs_s(x, a=0, b=None, win=3.0, pct=90):
    """Short-term loudness (3 s windows, 1 s hop): percentile over [a, b)."""
    seg = x[a:b]
    m = pyln.Meter(SR, block_size=0.4)
    w, hop = int(win * SR), int(1.0 * SR)
    vals = []
    for s in range(0, max(1, len(seg) - w + 1), hop):
        c = seg[s:s + w]
        if len(c) < int(0.5 * SR) or not np.any(c):
            continue
        try:
            v = m.integrated_loudness(c if c.ndim == 2 else c[:, None])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, pct)) if vals else -np.inf


def peak_clip(x, a, b, headroom_db=3.0, hit_win=0.03):
    """Clip a percussive stem 3 dB above its average hit peak in [a,b): no single
    hit can jump out of the mix."""
    seg = x[a:b]
    if not np.any(seg):
        return x
    env = np.abs(seg).max(axis=1) if seg.ndim == 2 else np.abs(seg)
    w = int(hit_win * SR)
    pk = np.array([env[i:i + w].max() for i in range(0, len(env) - w, w)])
    pk = pk[pk > pk.max() * 0.2]
    if not len(pk):
        return x
    thr = float(np.mean(pk)) * db(headroom_db)
    return soft_clip(x / thr, 0.85) * thr


# ------------------------------------------------------------------ stem chains (EQ map 4.4)
STEM_CHAINS = {
    "kick":    lambda: [P.HighpassFilter(25)],
    "snare":   lambda: [P.HighpassFilter(150), P.PeakFilter(450, -3.0, 0.9), P.PeakFilter(3500, 2.5, 0.8),
                        P.HighShelfFilter(8000, 2.0, 0.7)],
    "hats":    lambda: [P.HighpassFilter(500), P.HighShelfFilter(9000, 1.5, 0.7)],
    "perc":    lambda: [P.HighpassFilter(250), P.PeakFilter(450, -3.0, 0.9)],
    "lead":    lambda: [P.HighpassFilter(180), P.PeakFilter(400, -2.5, 0.8), P.PeakFilter(3000, 2.5, 0.8),
                        P.Compressor(-20, 3.0, 8, 100)],
    "counter": lambda: [P.HighpassFilter(250), P.PeakFilter(450, -4.0, 0.8)],
    "chords":  lambda: [P.HighpassFilter(220), P.PeakFilter(420, -4.0, 0.8)],
    "arp":     lambda: [P.HighpassFilter(300), P.PeakFilter(450, -3.0, 0.8)],
    "pad":     lambda: [P.HighpassFilter(250), P.PeakFilter(400, -4.5, 0.7), P.Compressor(-24, 2.0, 30, 250)],
    "bass":    lambda: [P.HighpassFilter(28), P.PeakFilter(170, 3.0, 0.9)],
}


def process_stems(stems):
    for name, chain in STEM_CHAINS.items():
        if name in stems and np.any(stems[name]):
            stems[name] = run(stems[name], chain())
    if "bass" in stems:
        stems["bass"] = mono_low(stems["bass"], 120.0)
    if "kick" in stems:
        stems["kick"] = mono_low(stems["kick"], 150.0)
    return stems


def crunch(x, drive_db=16.0, hp=800.0, level_db=-15.0):
    """Parallel distortion layer (kick / snare only)."""
    if not np.any(x):
        return x
    pk = np.max(np.abs(x)) + 1e-9
    c = hard(x / pk, drive_db)
    c = lr4(c, hp, "high")
    return x + c * pk * db(level_db) * 4


def drum_bus(x):
    """Glue 4:1, slow attack lets transients through; clip what pokes out."""
    y = run(x, [P.Compressor(-16, 4.0, 20, 120)])
    return soft_clip(y, 0.9) if np.max(np.abs(y)) > 0.9 else y


def music_bus(x, ott_depth=0.25, sat=1.6):
    y = ott(x, ott_depth)
    pk = np.max(np.abs(y)) + 1e-9
    y = tape(y / pk, sat) * pk
    return run(y, [P.PeakFilter(3000, 1.0, 0.7), P.HighShelfFilter(9000, 2.0, 0.7)])


def returns_eq(x):
    """Reverb/delay returns: no lows, no mud, no fizz."""
    return run(x, [P.HighpassFilter(500), P.PeakFilter(450, -3.0, 0.8), P.LowpassFilter(7500)])


def overdrive_bass(x, drive=1.0, width=0.35):
    """Legacy helper for synth basses: clean mono sub + loud driven mids."""
    if not np.any(x):
        return x
    pk = np.max(np.abs(x)) + 1e-9
    x = x / pk
    lo = lr4(x, 100, "low")
    hi = x - lo
    hi = hard(hi * 2.0, 18 + 10 * drive)
    hi = run(hi, [P.PeakFilter(400, 4.0, 0.8), P.PeakFilter(2500, -3.0, 1.0)])
    hi = hard(hi, 8 + 6 * drive, asym=0.1)
    hi = run(hi, [P.LowpassFilter(6000), P.HighpassFilter(100)])
    hi = ott(hi, 0.5)
    hi = hi / (np.max(np.abs(hi)) + 1e-9)
    if hi.ndim == 2 and width:
        hi = run(hi, [P.Chorus(rate_hz=0.5, depth=0.12, centre_delay_ms=7, feedback=0.0, mix=width)])
    y = lo * 0.55 / (np.max(np.abs(lo)) + 1e-9) + hi * 1.0
    return mono_low(y, 100) * pk


def felt_report(x):
    """Sub (<100 Hz) vs audible bass harmonics (100 Hz-2.5 kHz), K-weighted."""
    xx = x if x.ndim == 2 else x[:, None]
    lo = lr4(xx, 100, "low"); mid = lr4(xx - lo, 2500, "low")
    a, b = lufs(lo), lufs(mid)
    return {"sub_lufs": round(a, 1), "harm_lufs": round(b, 1), "diff_db": round(b - a, 1)}


# ------------------------------------------------------------------ master (section 6)
def limit(y, ceiling_db=-1.0):
    lim = _lsp("Limiter Stereo")
    if lim is not None and y.ndim == 2:
        try:
            lim.threshold_db = ceiling_db
            lim.lookahead_ms = 5.0
            lim.release_time_ms = 50.0
            lim.gain_boost = False
            try:
                lim.oversampling = lim.parameters["oversampling"].valid_values[-1]
            except Exception:
                pass
            lim.reset()
            # pad the end so lookahead does not swallow the last ms
            pad = np.zeros((int(0.05 * SR), 2))
            z = lim(np.concatenate([y, pad]).T.astype(np.float32), SR).T.astype(np.float64)
            lat = len(z) - len(y)
            z = z[: len(y)] if lat <= 0 else z[: len(y)]
            return z
        except Exception:
            pass
    return dsp.limiter(y, ceiling_db - 0.1, release=0.05)


def master(mix, target_lufs=-8.0, ceiling_db=-1.0, brightness=1.0):
    y = run(mix, [P.HighpassFilter(25), P.LowShelfFilter(100, 0.5, 0.7), P.PeakFilter(380, -1.5, 0.8),
                  P.HighShelfFilter(6000, 2.5 * brightness, 0.7)])
    # glue, 1-2 dB of gain reduction only
    pk_l = lufs(y)
    thr = (pk_l if np.isfinite(pk_l) else -14) + 4.0
    y = run(y, [P.Compressor(thr, 2.0, 30, 150)])
    g = db(target_lufs - lufs(y) + 1.0)
    for _ in range(3):
        z = limit(soft_clip(y * g, 0.75), ceiling_db)
        cur = lufs(z)
        if not np.isfinite(cur) or abs(cur - target_lufs) < 0.2:
            break
        g *= db(target_lufs - cur)
    return np.clip(z, -db(ceiling_db), db(ceiling_db))
