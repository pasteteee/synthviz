"""Reference recipes for synthviz v5 sound (numpy + scipy + pedalboard).
Every function is standalone and returns float64 arrays at SR. Not wired into the engine:
the engine thread ports what it needs. Run `python recipes.py out_dir` to render test sounds.
"""
from __future__ import annotations
import sys
import numpy as np
import pedalboard as P
from scipy.signal import butter, sosfilt

SR = 48000


# ---------------------------------------------------------------- helpers
def run(x, plugins):
    mono = x.ndim == 1
    a = (x[None, :] if mono else x.T).astype(np.float32)
    y = P.Pedalboard(plugins)(a, SR)
    return (y[0] if mono else y.T).astype(np.float64)


def lr4(x, fc, kind):
    sos = butter(2, fc, btype=kind, fs=SR, output="sos")
    return sosfilt(sos, sosfilt(sos, x, axis=0), axis=0)


def split(x, f):
    lo = lr4(x, f, "low")
    return lo, x - lo


def sine_sweep(f_start, f_end, tau, n, f_curve=None):
    """Exponential pitch glide f_start -> f_end with time constant tau (s)."""
    t = np.arange(n) / SR
    f = f_end + (f_start - f_end) * np.exp(-t / tau) if f_curve is None else f_curve
    return np.sin(2 * np.pi * np.cumsum(f) / SR), f


def env_exp(n, tau):
    return np.exp(-np.arange(n) / SR / tau)


def db(x):
    return 10 ** (x / 20)


def hard(x, drive_db, asym=0.0):
    """Drive into a hard-ish clipper (tanh with knee) – the core of hardstyle / 808 grit."""
    return np.tanh(x * db(drive_db) + asym) - np.tanh(asym)


# ---------------------------------------------------------------- 1. hardstyle / rawstyle kick
def hardstyle_kick(root_hz=43.65, length_s=0.42, raw=False, seed=0):
    """Three layers: punch (click+body), distorted tonal tail, clean sub under 90 Hz.
    root_hz = tail note (follow the chord root!). length = time to the next kick minus ~10 ms."""
    rng = np.random.default_rng(seed)
    n = int(length_s * SR)
    t = np.arange(n) / SR
    # --- punch: sine dropping 3.5 kHz -> 55 Hz very fast + 3 ms noise click
    click, _ = sine_sweep(3500, 55, 0.004, n)
    click *= env_exp(n, 0.035)
    nz = lr4(rng.standard_normal(n), 2500, "high") * env_exp(n, 0.0025) * 0.5
    punch = hard(click + nz, 6)
    # --- tail: pitch 170 Hz -> root over ~120 ms; raw style adds a pitch dip + bounce
    f = root_hz + (170 - root_hz) * np.exp(-t / 0.045)
    if raw:
        f = f * (1 - 0.06 * np.sin(2 * np.pi * 1.0 / length_s * t) ** 2)   # "bwoa" dip
    body, _ = sine_sweep(0, 0, 1, n, f_curve=f)
    body = body + 0.35 * np.sin(2 * np.pi * np.cumsum(2 * f) / SR)          # 2nd harmonic seed
    # distortion chain: EQ -> clip -> resonant peak -> clip -> LP  (EQ before every stage)
    x = run(body, [P.PeakFilter(700, 9, 0.8), P.HighpassFilter(60)])
    x = hard(x, 22 if not raw else 30)
    x = run(x, [P.PeakFilter(950 if not raw else 1400, 6, 1.2), P.LowShelfFilter(150, -6)])
    x = hard(x, 12, asym=0.15)
    x = run(x, [P.LowpassFilter(11000 if raw else 9000), P.PeakFilter(2200, 3, 0.9)])
    # tail amplitude: rises after 25 ms, holds, short 12 ms cut before next kick
    a = np.clip(t / 0.025, 0, 1) * np.clip((length_s - t) / 0.012, 0, 1)
    a *= 0.85 + 0.15 * (t / length_s)                                            # swells slightly
    tail = x * a
    _, tail = split(tail, 90)                                                    # tail owns > 90 Hz
    # --- clean sub: same pitch curve, pure sine, low-passed, owns < 90 Hz
    sub, _ = sine_sweep(0, 0, 1, n, f_curve=f)
    sub = lr4(np.tanh(sub * 1.5), 90, "low") * np.clip((length_s - t) / 0.012, 0, 1)
    k = punch * 0.6 + tail * 1.0 + sub * 0.8
    return k / (np.max(np.abs(k)) + 1e-9) * 0.95


# ---------------------------------------------------------------- 2. 808 with glide + grit
def bass_808(notes, total_s, drive=1.0):
    """notes = [(t_start, dur, midi)]. Mono pitch track with 70 ms glide between overlapping notes,
    +7 semitone pitch punch on each new note, 8 ms click, long decay.
    Saturation puts harmonics at 2f..8f so the bass is audible on phones (100–500 Hz)."""
    n = int(total_s * SR)
    f = np.zeros(n); amp = np.zeros(n)
    prev = None
    for (ts, d, m) in notes:
        a, b = int(ts * SR), min(n, int((ts + d) * SR))
        hz = 440 * 2 ** ((m - 69) / 12)
        tt = np.arange(b - a) / SR
        if prev is not None and abs(prev[1] - a) < int(0.02 * SR):      # legato -> glide
            seg = hz + (prev[0] - hz) * np.exp(-tt / 0.07)
            env = np.ones(b - a)
        else:                                                           # new hit -> pitch punch
            seg = hz * 2 ** (7 / 12 * np.exp(-tt / 0.012))
            env = np.exp(-tt / 0.9) * np.clip(tt / 0.002, 0, 1)
        env *= np.clip((d - tt) / 0.015, 0, 1)
        f[a:b] = seg; amp[a:b] = env
        prev = (hz, b)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * amp
    lo, hi = split(x, 120)
    grit = hard(x, 10 + 10 * drive, asym=0.2)                           # asym -> even harmonics
    grit = run(grit, [P.HighpassFilter(110), P.PeakFilter(250, 3, 0.8), P.LowpassFilter(5000)])
    y = lr4(x, 120, "low") * 0.8 + grit * (0.7 + 0.4 * drive)
    return y / (np.max(np.abs(y)) + 1e-9) * 0.95


# ---------------------------------------------------------------- 3. heavy overdriven bass (reese / mid-bass)
def saw(f_curve):
    ph = np.cumsum(f_curve) / SR
    return 2 * (ph % 1.0) - 1   # naive saw is fine because everything gets distorted & low-passed


def heavy_bass(midi, dur_s, drive=1.0, seed=0):
    """Clean mono sine sub < 100 Hz + 3 detuned saws -> multiband distortion above 100 Hz.
    The 'felt' part is 120 Hz–2 kHz: that band is driven, compressed and kept LOUD."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * SR); t = np.arange(n) / SR
    hz = 440 * 2 ** ((midi - 69) / 12)
    sub = np.sin(2 * np.pi * hz * t)
    mid = sum(saw(np.full(n, hz * 2 ** (c / 1200))) for c in (-14, 0, 13)) / 3
    mid = run(mid, [P.LadderFilter(P.LadderFilter.Mode.LPF24, 900 + 600 * drive, 0.35, 1.0)])
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.6 * t + rng.uniform(0, 6))     # slow movement
    mid = mid * (0.7 + 0.3 * lfo)
    # serial distortion with EQ between stages (multiband drive on the band > 100 Hz)
    _, mid = split(mid, 100)
    mid = hard(mid, 18 + 10 * drive)
    mid = run(mid, [P.PeakFilter(400, 4, 0.8), P.PeakFilter(2500, -3, 1.0)])
    mid = hard(mid, 8 + 6 * drive, asym=0.1)
    mid = run(mid, [P.LowpassFilter(6000), P.HighpassFilter(100)])
    mid = ott(mid, depth=0.5)
    mid = mid / (np.max(np.abs(mid)) + 1e-9)
    y = lr4(sub, 100, "low") * 0.55 + mid * 1.0      # mids LOUDER than the sub in K-weighted loudness
    return y / (np.max(np.abs(y)) + 1e-9) * 0.95


# ---------------------------------------------------------------- 4. OTT-style 3-band up/down compressor
def _env(x, att, rel):
    from scipy.signal import lfilter
    a = np.abs(x) if x.ndim == 1 else np.abs(x).max(axis=1)
    # attack/release one-pole on the rectified signal (cheap, numba-able)
    ka, kr = np.exp(-1 / (att * SR)), np.exp(-1 / (rel * SR))
    e = np.empty_like(a); s = 0.0
    for i, v in enumerate(a):
        k = ka if v > s else kr
        s = k * s + (1 - k) * v
        e[i] = s
    return e


def ott(x, depth=0.3, f1=88.0, f2=2500.0, down_thr=-30.0, up_thr=-40.0, down_ratio=10.0, up_ratio=3.0,
        time=0.5, gains_db=(0.0, 0.0, 2.0)):
    """Multiband upward+downward compression (the 'OTT' trick): quiet detail lifted, peaks pushed down,
    sound gets dense and bright. depth = dry/wet. Use 20–40 % on synths, 50–100 % on bass mids/FX.
    Loop in _env is slow in pure python: port it to numba in the engine."""
    lo = lr4(x, f1, "low"); rest = x - lo
    mid = lr4(rest, f2, "low"); hi = rest - mid
    out = 0
    for band, g in zip((lo, mid, hi), gains_db):
        e = 20 * np.log10(_env(band, 0.003 * time * 2, 0.15 * time * 2) + 1e-9)
        gr = np.where(e > down_thr, (down_thr - e) * (1 - 1 / down_ratio), 0.0)
        gr += np.where(e < up_thr, np.minimum((up_thr - e) * (1 - 1 / up_ratio), 24.0), 0.0)
        gr = db(gr + g)
        out = out + band * (gr if band.ndim == 1 else gr[:, None])
    out = out / (np.max(np.abs(out)) + 1e-9) * np.max(np.abs(x))
    return x * (1 - depth) + out * depth


# ---------------------------------------------------------------- 5. sidechain shapes (LFO-Tool style)
def duck_curve(n, hits_s, beat_s, depth=1.0, hold=0.0, recover=0.5, shape=2.5):
    """Volume shape triggered on every kick: drops to (1-depth) in 2 ms, holds, recovers over
    `recover` beats with curve exponent `shape` (>1 = stays down longer = more audible pump).
    bass: depth 1.0 recover 0.35–0.5; chords: 0.8, 0.5; lead: 0.4, 0.3; reverb returns: 1.0, 0.6."""
    g = np.ones(n)
    att = int(0.002 * SR); hl = int(hold * beat_s * SR); rec = int(recover * beat_s * SR)
    seg = np.concatenate([np.linspace(1, 1 - depth, att), np.full(hl, 1 - depth),
                          1 - depth + depth * np.linspace(0, 1, rec) ** shape])
    for h in hits_s:
        a = int(h * SR); b = min(n, a + len(seg))
        if a < n:
            g[a:b] = np.minimum(g[a:b], seg[: b - a])
    return g


# ---------------------------------------------------------------- 6. master chain (pedalboard + LSP VST3 if present)
def master(mix, target_lufs=-7.0, lsp="/usr/lib/vst3/lsp-plugins.vst3"):
    import pyloudnorm as pyln
    y = run(mix, [P.HighpassFilter(25),
                  P.Compressor(threshold_db=-10, ratio=2.0, attack_ms=30, release_ms=150)])   # 1–2 dB glue only
    meter = pyln.Meter(SR)
    g = db(target_lufs - meter.integrated_loudness(y) + 1.0)   # +1 dB: limiter/clipper eat ~1 dB
    y = y * g
    # clip the transients first (kick/snare peaks), then the limiter only catches what is left
    y = np.where(np.abs(y) < 0.7, y, np.sign(y) * (0.7 + 0.3 * np.tanh((np.abs(y) - 0.7) / 0.3)))
    try:
        lim = P.load_plugin(lsp, plugin_name="Limiter Stereo")
        lim.threshold_db = -1.0
        lim.lookahead_ms = 5.0
        lim.release_time_ms = 50.0
        lim.gain_boost = False        # LSP "boost" adds makeup gain; we set loudness ourselves
        y = lim(y.T.astype(np.float32), SR).T.astype(np.float64)
    except Exception:
        y = run(y, [P.Limiter(threshold_db=-1.0, release_ms=60)])
    return np.clip(y, -db(-1.0), db(-1.0))


# ---------------------------------------------------------------- 7. "is the bass felt?" metric
def felt_report(x):
    """Run on the BASS STEM only (not the full mix). K-weighted loudness of the sub (<100 Hz) vs the
    audible bass harmonics (100 Hz–2.5 kHz). Phones/laptops reproduce almost nothing below ~100 Hz,
    so for a heavy/overdriven bass aim for diff_db >= +5, for a clean 808 around +3..+8."""
    import pyloudnorm as pyln
    m = pyln.Meter(SR)
    xx = x if x.ndim == 2 else x[:, None]
    lo = lr4(xx, 100, "low"); mid = lr4(xx - lo, 2500, "low")
    return {"sub_lufs": round(m.integrated_loudness(lo), 1), "harm_lufs": round(m.integrated_loudness(mid), 1),
            "diff_db": round(m.integrated_loudness(mid) - m.integrated_loudness(lo), 1)}


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    from pedalboard.io import AudioFile
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    bpm = 150; beat = 60 / bpm
    def save(name, x):
        x = np.atleast_2d(x.T).T if x.ndim == 1 else x
        x = np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x
        with AudioFile(f"{out}/{name}.wav", "w", SR, 2) as f:
            f.write(x.T.astype(np.float32))
    # hardstyle kick loop following roots F, Db, Ab, Eb (2 bars each)
    roots = [43.65, 34.65, 51.91, 38.89]
    loop = np.zeros(int(beat * 32 * SR))
    for i in range(32):
        r = roots[i // 8]
        k = hardstyle_kick(r, beat - 0.01, raw=(i // 8) % 2 == 1, seed=i)
        a = int(i * beat * SR); loop[a:a + len(k)] += k
    save("kick_hardstyle_raw_loop", loop * 0.9)
    # 808 pattern with glides
    b140 = 60 / 140
    notes = [(0, b140 * 1.5, 29), (b140 * 1.5, b140 * 0.5, 29), (b140 * 2, b140 * 1.0, 36), (b140 * 3, b140, 34),
             (b140 * 4, b140 * 1.5, 31), (b140 * 5.5, b140 * 2.5, 29)]
    save("bass_808_glide", bass_808(notes * 1, b140 * 8))
    hb = heavy_bass(33, 4.0, drive=1.2)
    save("bass_heavy_overdrive", hb)
    for nm, x in (("kick", loop), ("808", bass_808(notes, b140 * 8)), ("heavy", hb)):
        print(nm, felt_report(x))
    # quick mix check: kick loop + sidechained heavy bass through the master
    n = len(loop); bass = np.resize(hb, n)
    sc = duck_curve(n, [i * beat for i in range(32)], beat, depth=1.0, recover=0.45)
    mix = np.stack([loop + bass * sc * 0.7] * 2, axis=1)
    m = master(mix, -7.0)
    import pyloudnorm as pyln
    print("master LUFS", round(pyln.Meter(SR).integrated_loudness(m), 2), "peak dBFS", round(20 * np.log10(np.abs(m).max()), 2))
    save("mix_kick_plus_sidechained_bass", m)
