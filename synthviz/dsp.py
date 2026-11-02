"""Low-level DSP: band-limited oscillators, filters, envelopes, effects.
All signals are float64 numpy arrays at SR. Hot loops are numba-compiled."""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.signal import fftconvolve, butter, sosfilt

SR = 44100


# ---------------------------------------------------------------- oscillators
def phase_from_freq(freq, phase0=0.0):
    """freq: array (Hz per sample) -> phase in [0,1)."""
    ph = np.cumsum(freq / SR) + phase0
    return ph - np.floor(ph)


def _polyblep(t, dt):
    out = np.zeros_like(t)
    dt = np.maximum(dt, 1e-9)
    m = t < dt
    x = t[m] / dt[m] if np.ndim(dt) else t[m] / dt
    out[m] = x + x - x * x - 1.0
    m2 = t > 1.0 - dt
    x = (t[m2] - 1.0) / (dt[m2] if np.ndim(dt) else dt)
    out[m2] = x * x + x + x + 1.0
    return out


def osc_saw(ph, freq):
    dt = freq / SR
    return 2.0 * ph - 1.0 - _polyblep(ph, dt)


def osc_pulse(ph, freq, pw=0.5):
    dt = freq / SR
    p2 = ph + pw
    p2 = p2 - np.floor(p2)
    return (osc_saw(ph, freq) - (2.0 * p2 - 1.0 - _polyblep(p2, dt))) * 0.5 + (2 * pw - 1) * 0.0


def osc_tri(ph, freq=None):
    return 4.0 * np.abs(ph - 0.5) - 1.0


def osc_sine(ph, freq=None):
    return np.sin(2 * np.pi * ph)


def osc(kind, freq, phase0=0.0, pw=0.5):
    ph = phase_from_freq(freq, phase0)
    if kind == "saw":
        return osc_saw(ph, freq)
    if kind in ("square", "pulse"):
        return osc_pulse(ph, freq, pw)
    if kind == "tri":
        return osc_tri(ph)
    if kind == "sine":
        return osc_sine(ph)
    raise ValueError(kind)


def noise(n, rng=None, color="white"):
    rng = rng or np.random.default_rng()
    w = rng.standard_normal(n)
    if color == "white":
        return w
    if color == "pink":
        # Paul Kellet economy filter
        return _pink(w)
    raise ValueError(color)


@njit(cache=True)
def _pink(w):
    b0 = b1 = b2 = 0.0
    out = np.empty_like(w)
    for i in range(w.shape[0]):
        x = w[i]
        b0 = 0.99765 * b0 + x * 0.0990460
        b1 = 0.96300 * b1 + x * 0.2965164
        b2 = 0.57000 * b2 + x * 1.0526913
        out[i] = (b0 + b1 + b2 + x * 0.1848) * 0.2
    return out


def fm_op(freq, ratio, index, phase_mod=None, feedback=0.0):
    """Sine operator; phase_mod = modulator output (radians/2pi units)."""
    ph = phase_from_freq(freq * ratio)
    if phase_mod is None and feedback == 0:
        return np.sin(2 * np.pi * ph)
    pm = np.zeros_like(ph) if phase_mod is None else phase_mod
    if feedback:
        return _fb_sine(ph, pm, feedback) * 1.0
    return np.sin(2 * np.pi * ph + index * pm)


@njit(cache=True)
def _fb_sine(ph, pm, fb):
    out = np.empty_like(ph)
    y1 = 0.0
    y2 = 0.0
    for i in range(ph.shape[0]):
        y = np.sin(2 * np.pi * ph[i] + pm[i] + fb * 0.5 * (y1 + y2))
        y2 = y1
        y1 = y
        out[i] = y
    return out


# --------------------------------------------------------------------- filters
@njit(cache=True)
def svf(x, cutoff, res, mode):
    """Cytomic TPT state-variable filter. cutoff: per-sample Hz array.
    res 0..1 (self-osc near 1). mode 0=LP 1=BP 2=HP 3=notch."""
    n = x.shape[0]
    out = np.empty(n)
    ic1 = 0.0
    ic2 = 0.0
    k = 2.0 - 2.0 * min(res, 0.995)
    for i in range(n):
        fc = min(max(cutoff[i], 20.0), SR * 0.45)
        g = np.tan(np.pi * fc / SR)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            out[i] = v2
        elif mode == 1:
            out[i] = v1
        elif mode == 2:
            out[i] = x[i] - k * v1 - v2
        else:
            out[i] = x[i] - k * v1
    return out


@njit(cache=True)
def ladder(x, cutoff, res, drive):
    """Zero-delay-feedback 4-pole ladder with tanh input saturation.
    res 0..1 (k up to ~3.9)."""
    n = x.shape[0]
    out = np.empty(n)
    s1 = s2 = s3 = s4 = 0.0
    k = 3.9 * min(max(res, 0.0), 1.0)
    for i in range(n):
        fc = min(max(cutoff[i], 20.0), SR * 0.42)
        g = np.tan(np.pi * fc / SR)
        G = g / (1.0 + g)
        inv = 1.0 / (1.0 + g)
        S = G * G * G * s1 * inv + G * G * s2 * inv + G * s3 * inv + s4 * inv
        G4 = G * G * G * G
        u = x[i] * drive
        y4 = (G4 * u + S) / (1.0 + k * G4)
        v = np.tanh(u - k * y4)
        # stage 1
        t = (v - s1) * G
        y1 = t + s1
        s1 = y1 + t
        t = (y1 - s2) * G
        y2 = t + s2
        s2 = y2 + t
        t = (y2 - s3) * G
        y3 = t + s3
        s3 = y3 + t
        t = (y3 - s4) * G
        y = t + s4
        s4 = y + t
        out[i] = y * (1.0 + 0.5 * k) / max(drive, 1e-3)
    return out


@njit(cache=True)
def onepole_lp(x, fc):
    out = np.empty_like(x)
    a = np.exp(-2.0 * np.pi * fc / SR)
    y = 0.0
    for i in range(x.shape[0]):
        y = (1 - a) * x[i] + a * y
        out[i] = y
    return out


def butter_filter(x, fc, kind="low", order=2):
    sos = butter(order, fc, btype=kind, fs=SR, output="sos")
    return sosfilt(sos, x, axis=0)


@njit(cache=True)
def biquad(x, b0, b1, b2, a1, a2):
    out = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(x.shape[0]):
        y = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = x[i]
        y2 = y1
        y1 = y
        out[i] = y
    return out


def peaking_eq(x, fc, gain_db, q=1.0):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * fc / SR
    al = np.sin(w) / (2 * q)
    a0 = 1 + al / A
    return biquad(x, (1 + al * A) / a0, -2 * np.cos(w) / a0, (1 - al * A) / a0,
                  -2 * np.cos(w) / a0, (1 - al / A) / a0)


def shelf(x, fc, gain_db, high=True):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * fc / SR
    cs, sn = np.cos(w), np.sin(w)
    al = sn / 2 * np.sqrt(2)
    sA = 2 * np.sqrt(A) * al
    if high:
        b0 = A * ((A + 1) + (A - 1) * cs + sA); b1 = -2 * A * ((A - 1) + (A + 1) * cs)
        b2 = A * ((A + 1) + (A - 1) * cs - sA); a0 = (A + 1) - (A - 1) * cs + sA
        a1 = 2 * ((A - 1) - (A + 1) * cs); a2 = (A + 1) - (A - 1) * cs - sA
    else:
        b0 = A * ((A + 1) - (A - 1) * cs + sA); b1 = 2 * A * ((A - 1) - (A + 1) * cs)
        b2 = A * ((A + 1) - (A - 1) * cs - sA); a0 = (A + 1) + (A - 1) * cs + sA
        a1 = -2 * ((A - 1) + (A + 1) * cs); a2 = (A + 1) + (A - 1) * cs - sA
    if x.ndim == 2:
        from scipy.signal import lfilter
        return lfilter([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0], x, axis=0)
    return biquad(x, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


# ------------------------------------------------------------------- envelopes
def adsr(n_gate, n_total, a, d, s, r, curve=4.0):
    """Exponential-ish ADSR. Times in seconds. n_gate: samples key held."""
    env = np.zeros(n_total)
    A = max(1, int(a * SR)); D = max(1, int(d * SR)); R = max(1, int(r * SR))
    t = np.arange(n_total)
    # attack (slightly convex)
    att = 1 - np.exp(-curve * t / A) if curve > 0 else t / A
    att = att / (1 - np.exp(-curve)) if curve > 0 else att
    dec = s + (1 - s) * np.exp(-5.0 * (t - A) / D)
    e = np.where(t < A, np.minimum(att, 1.0), dec)
    g = min(n_gate, n_total)
    env[:g] = e[:g]
    lvl = env[g - 1] if g > 0 else 0.0
    rel = lvl * np.exp(-6.9 * (t[g:] - g) / R)
    env[g:] = rel
    return env


def exp_decay(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


# --------------------------------------------------------------------- effects
def make_ir(seconds=2.5, damp=0.6, predelay=0.02, seed=7, stereo=True, brightness=1.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    chans = []
    for c in range(2 if stereo else 1):
        w = rng.standard_normal(n)
        env = np.exp(-6.9 * t / seconds)
        # early reflections
        er = np.zeros(n)
        for k in range(14):
            pos = int(rng.uniform(0.003, 0.08) * SR)
            er[pos] += rng.uniform(-1, 1) * 0.7
        sig = w * env
        # frequency-dependent decay: progressively lowpassed segments
        lp = onepole_lp(sig, 9000 * brightness)
        lp2 = onepole_lp(sig, 2500 * brightness)
        mix = np.clip(t / seconds * 2 * damp, 0, 1)
        sig = lp * (1 - mix) + lp2 * mix + er * 0.6
        pd = int(predelay * SR)
        sig = np.concatenate([np.zeros(pd), sig])
        chans.append(sig / np.sqrt(np.sum(sig ** 2)))
    return np.stack(chans, axis=1)


_IR_CACHE = {}


def reverb(x, seconds=2.5, mix=0.25, damp=0.6, predelay=0.02, lowcut=200.0, brightness=1.0):
    """Convolution reverb with synthetic IR. x: (n,2). Returns wet+dry."""
    key = (round(seconds, 2), round(damp, 2), round(predelay, 3), round(brightness, 2))
    if key not in _IR_CACHE:
        _IR_CACHE[key] = make_ir(seconds, damp, predelay, brightness=brightness)
    ir = _IR_CACHE[key]
    src = butter_filter(x, lowcut, "high", 2) if lowcut else x
    wetL = fftconvolve(src[:, 0], ir[:, 0])[: len(x)]
    wetR = fftconvolve(src[:, 1], ir[:, 1])[: len(x)]
    wet = np.stack([wetL, wetR], axis=1) * 2.5
    return x * (1 - mix * 0.5) + wet * mix


def wet_reverb(x, **kw):
    """Only the wet signal (for send busses)."""
    kw = dict(kw)
    mix = kw.pop("mix", 1.0)
    out = reverb(x, mix=1.0, **kw)
    return (out - x * 0.5) * mix


def delay(x, time_s, feedback=0.4, mix=0.3, pingpong=True, lp=4500.0, n_echo=8):
    n = len(x)
    d = int(time_s * SR)
    out = np.zeros_like(x)
    cur = x.copy()
    g = 1.0
    for k in range(1, n_echo + 1):
        g *= feedback
        if g < 0.01 or k * d >= n:
            break
        cur = np.stack([onepole_lp(cur[:, 0], lp), onepole_lp(cur[:, 1], lp)], axis=1)
        sh = np.zeros_like(x)
        sh[k * d:] = cur[: n - k * d]
        if pingpong and k % 2 == 1:
            sh = sh[:, ::-1]
        out += sh * g
    return x + out * mix


def chorus(x, rate=0.6, depth_ms=3.5, base_ms=12.0, mix=0.5):
    """Stereo chorus (quadrature LFOs). x: mono (n,) or stereo (n,2)."""
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    n = len(x)
    t = np.arange(n) / SR
    out = np.empty_like(x)
    idx = np.arange(n, dtype=np.float64)
    for c, ph in ((0, 0.0), (1, np.pi / 2)):
        dly = (base_ms + depth_ms * np.sin(2 * np.pi * rate * t + ph)) * SR / 1000
        out[:, c] = np.interp(idx - dly, idx, x[:, c], left=0.0)
    return x * (1 - mix) + out * mix


def unison_stereo(voices, spread=0.8):
    """voices: list of mono arrays -> stereo array with voices panned across."""
    n = len(voices)
    L = np.zeros_like(voices[0]); R = np.zeros_like(voices[0])
    for i, v in enumerate(voices):
        p = 0.0 if n == 1 else (i / (n - 1) * 2 - 1) * spread
        L += v * np.sqrt((1 - p) / 2); R += v * np.sqrt((1 + p) / 2)
    return np.stack([L, R], axis=1)


def pan(x, p):
    p = float(np.clip(p, -1, 1))
    if x.ndim == 1:
        return np.stack([x * np.sqrt((1 - p) / 2), x * np.sqrt((1 + p) / 2)], axis=1) * np.sqrt(2)
    return np.stack([x[:, 0] * min(1, 1 - p), x[:, 1] * min(1, 1 + p)], axis=1)


def drive(x, amount=2.0, kind="tanh"):
    if kind == "tanh":
        return np.tanh(x * amount) / np.tanh(amount)
    if kind == "hard":
        return np.clip(x * amount, -1, 1)
    if kind == "fold":
        y = x * amount
        return np.sin(y * np.pi / 2)
    if kind == "asym":
        y = x * amount
        return np.where(y > 0, np.tanh(y), np.tanh(y * 0.6) / 0.6 * 0.8)
    raise ValueError(kind)


def bitcrush(x, bits=8, down=4):
    q = 2 ** (bits - 1)
    y = np.round(x * q) / q
    if down > 1:
        y = np.repeat(y[::down], down, axis=0)[: len(x)]
    return y


@njit(cache=True)
def _env_follow(x, att, rel):
    out = np.empty_like(x)
    e = 0.0
    a = np.exp(-1.0 / (att * SR))
    r = np.exp(-1.0 / (rel * SR))
    for i in range(x.shape[0]):
        v = abs(x[i])
        c = a if v > e else r
        e = c * e + (1 - c) * v
        out[i] = e
    return out


def compressor(x, threshold_db=-18, ratio=4.0, attack=0.005, release=0.12, makeup_db=0.0, knee_db=6.0):
    mono = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    env = _env_follow(mono, attack, release)
    lvl = 20 * np.log10(env + 1e-9)
    over = lvl - threshold_db
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    g = 10 ** ((-gr + makeup_db) / 20)
    return x * (g[:, None] if x.ndim == 2 else g)


def limiter(x, ceiling_db=-0.8, release=0.08, lookahead=0.004):
    ceil = 10 ** (ceiling_db / 20)
    la = int(lookahead * SR)
    peak = np.max(np.abs(x), axis=1)
    # lookahead peak hold
    from scipy.ndimage import maximum_filter1d
    held = maximum_filter1d(peak, size=2 * la + 1, origin=0)
    need = np.minimum(1.0, ceil / np.maximum(held, 1e-9))
    # smooth gain: instant down, release up
    g = _gain_smooth(need, release)
    y = x * g[:, None]
    return np.clip(y, -ceil, ceil)


@njit(cache=True)
def _gain_smooth(need, rel):
    out = np.empty_like(need)
    r = np.exp(-1.0 / (rel * SR))
    g = 1.0
    for i in range(need.shape[0]):
        if need[i] < g:
            g = need[i]
        else:
            g = r * g + (1 - r) * need[i]
        out[i] = g
    return out


def sidechain_curve(n, hit_times_s, depth=0.7, attack=0.004, release=0.18, shape=2.0):
    """Gain curve ducking after each hit (kick)."""
    g = np.ones(n)
    rel_n = int(release * SR)
    att_n = max(1, int(attack * SR))
    tt = np.arange(rel_n) / rel_n
    duck = 1 - depth * (1 - tt) ** shape
    att = np.linspace(1, 1 - depth, att_n)
    for t in hit_times_s:
        i = int(t * SR) - att_n
        if i < 0:
            i = 0
        a_end = min(n, i + att_n)
        g[i:a_end] = np.minimum(g[i:a_end], att[: a_end - i])
        j = a_end
        e = min(n, j + rel_n)
        g[j:e] = np.minimum(g[j:e], duck[: e - j])
    return g


def stereo_width(x, width=1.3):
    m = (x[:, 0] + x[:, 1]) * 0.5
    s = (x[:, 0] - x[:, 1]) * 0.5 * width
    return np.stack([m + s, m - s], axis=1)


def place(buf, sig, start_s, gain=1.0):
    """Mix sig (n,) or (n,2) into stereo buf at start time."""
    i = int(round(start_s * SR))
    if i >= len(buf):
        return
    if i < 0:
        sig = sig[-i:]
        i = 0
    n = min(len(sig), len(buf) - i)
    if sig.ndim == 1:
        buf[i:i + n] += sig[:n, None] * gain
    else:
        buf[i:i + n] += sig[:n] * gain
