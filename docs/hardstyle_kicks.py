"""Hardstyle kick v2: four flavours + a context loop (clap, hats, supersaw lead) for A/B listening.
Differences from recipes.hardstyle_kick: 4x oversampled HARD clipping (no aliasing fizz),
tail pitch glides from ~2x root down to the root over the whole beat (the 'tok-wooo'),
EQ->clip->EQ->clip chain with a resonant 'zaag' peak, short room reverb on the tail,
separate punch layer with its own click. Run: python hardstyle_kicks.py out_dir
"""
from __future__ import annotations
import sys
import numpy as np
import pedalboard as P
from scipy.signal import resample_poly
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from recipes import SR, run, lr4, db, duck_curve, master, ott

OS = 4


def clip_os(x, drive_db, kind="hard", asym=0.0):
    """Oversampled clipper. hard = brickwall, soft = tanh. Oversampling kills aliasing fizz."""
    u = resample_poly(x, OS, 1)
    u = u * db(drive_db) + asym
    u = np.clip(u, -1, 1) if kind == "hard" else np.tanh(u)
    u = u - (np.clip(asym, -1, 1) if kind == "hard" else np.tanh(asym))
    return resample_poly(u, 1, OS)


FLAVOURS = {
    # pitch: start multiple of root, glide time constant; eq peaks before each clip; zaag = resonant peak
    "euphoric": dict(p_start=2.0, p_tau=0.10, p_end=1.0, drive=(24, 14), pre=(250, 8, 0.7), mid=(1100, 5, 1.0),
                     zaag=(2800, 7, 1.2), lp=13000, rev=0.18, tail_env="decay", punch=1.0, raw_dip=0.0),
    "classic": dict(p_start=2.4, p_tau=0.16, p_end=1.0, drive=(28, 18), pre=(300, 9, 0.6), mid=(800, 6, 1.4),
                    zaag=(1600, 7, 4.0), lp=10000, rev=0.12, tail_env="hold", punch=0.8, raw_dip=0.0),
    "raw": dict(p_start=3.0, p_tau=0.05, p_end=0.94, drive=(32, 20), pre=(400, 10, 0.8), mid=(1400, 6, 1.5),
                zaag=(3000, 8, 1.5), lp=14000, rev=0.10, tail_env="swell", punch=1.1, raw_dip=0.10),
    "modern_punchy": dict(p_start=2.0, p_tau=0.07, p_end=1.0, drive=(22, 12), pre=(200, 7, 0.7), mid=(900, 4, 0.9),
                          zaag=(2600, 6, 1.2), lp=13000, rev=0.15, tail_env="decay", punch=1.4, raw_dip=0.0),
}


def kick(root_hz, length_s, fl, seed=0):
    rng = np.random.default_rng(seed)
    n = int(length_s * SR); t = np.arange(n) / SR
    # ---- punch: 300 Hz -> 60 Hz in ~30 ms, plus 2 ms click
    fp = 60 + 240 * np.exp(-t / 0.012) + 2500 * np.exp(-t / 0.0015)
    punch = np.sin(2 * np.pi * np.cumsum(fp) / SR) * np.exp(-t / 0.06)
    punch += lr4(rng.standard_normal(n), 3000, "high") * np.exp(-t / 0.002) * 0.4
    punch = clip_os(punch, 6, "soft")
    punch = run(punch, [P.PeakFilter(120, 3, 1.0), P.PeakFilter(4000, 3, 1.0)])
    # ---- tail pitch: fast 'body' drop then slow glide p_start*root -> p_end*root over the beat
    body = 200 * np.exp(-t / 0.018)
    glide = root_hz * (fl["p_end"] + (fl["p_start"] - fl["p_end"]) * np.exp(-t / fl["p_tau"]))
    f = glide + body
    if fl["raw_dip"]:
        f = f * (1 - fl["raw_dip"] * np.sin(np.pi * np.clip(t / length_s, 0, 1)) ** 2)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph)
    # ---- distortion chain: EQ -> HARD clip -> EQ (mid + zaag) -> clip -> LP
    x = run(x, [P.PeakFilter(*fl["pre"]), P.HighpassFilter(40)])
    x = clip_os(x, fl["drive"][0], "hard")
    # cut the lows hard BEFORE the 2nd clip: the 2nd stage then barks in the mids (sub comes back clean below)
    x = run(x, [P.HighpassFilter(140), P.HighpassFilter(140), P.PeakFilter(*fl["mid"])])
    x = clip_os(x, fl["drive"][1], "hard", asym=0.1)
    # post-EQ AFTER the last hard clip: a fully clipped wave is a square whose top is fixed,
    # so the 'zaag' crunch and the air must be shaped here, then glued by a light soft clip
    x = run(x, [P.PeakFilter(*fl["zaag"]), P.HighShelfFilter(3500, 6, 0.7), P.PeakFilter(300, -3, 0.8)])
    x = clip_os(x * 0.35, 6, "soft")
    x = run(x, [P.LowpassFilter(fl["lp"]), P.HighpassFilter(30)])
    # ---- tail volume shape
    rise = np.clip((t - 0.012) / 0.02, 0, 1)
    if fl["tail_env"] == "decay":
        shape = 0.55 + 0.45 * np.exp(-t / (length_s * 0.6))
    elif fl["tail_env"] == "swell":
        shape = 0.8 + 0.2 * (t / length_s)
    else:
        shape = np.ones(n)
    cut = np.clip((length_s - t) / 0.010, 0, 1)
    tail = x * rise * shape * cut
    # short room on the tail only (keeps it 'big' without washing the punch)
    if fl["rev"]:
        wet = run(np.stack([tail, tail], 1), [P.Reverb(room_size=0.25, damping=0.6, wet_level=1.0, dry_level=0.0, width=1.0),
                                              P.HighpassFilter(300), P.LowpassFilter(7000)])
        tail2 = np.stack([tail, tail], 1) + wet * fl["rev"]
    else:
        tail2 = np.stack([tail, tail], 1)
    # ---- sub: clean sine on the same pitch curve under 90 Hz
    sub = lr4(np.tanh(1.3 * np.sin(ph)), 90, "low") * cut * np.clip(t / 0.005, 0, 1)
    lo, hi = lr4(tail2, 90, "low"), None
    tail2 = tail2 - lo                          # tail owns > 90 Hz, sub owns < 90 Hz
    k = tail2 * 1.0 + (punch * 0.45 * fl["punch"] + sub * 0.6)[:, None]
    return k / (np.abs(k).max() + 1e-9) * 0.95


def supersaw(f, n, voices=7, detune=0.22, rng=None):
    rng = rng or np.random.default_rng(0)
    cents = np.linspace(-1, 1, voices) ** 3 * 40 * detune / 0.22
    L = np.zeros(n); R = np.zeros(n)
    for i, c in enumerate(cents):
        ph = (np.cumsum(np.full(n, f * 2 ** (c / 1200))) / SR + rng.uniform()) % 1
        s = 2 * ph - 1
        p = (i / (voices - 1)) * 2 - 1
        g = 1.0 if c == 0 else 0.7
        L += s * g * (1 - p) / 2; R += s * g * (1 + p) / 2
    return np.stack([L, R], 1) / voices


def context_loop(flavour, bpm=150, bars=8, seed=1):
    beat = 60 / bpm; n = int(bars * 4 * beat * SR)
    fl = FLAVOURS[flavour]
    roots = [43.65, 34.65 * 2 / 2, 51.91, 38.89]                   # F1, Db1, Ab1, Eb1 : i-VI-III-VII in F minor
    kick_buf = np.zeros((n, 2)); hits = []
    for i in range(bars * 4):
        r = roots[(i // 8) % 4]
        k = kick(r if r >= 38 else r * 2, beat - 0.008, fl, seed=i)   # keep tail roots >= 38 Hz
        a = int(i * beat * SR); kick_buf[a:a + len(k)] += k; hits.append(i * beat)
    # clap on 2 & 4 (3 noise bursts + tail), offbeat open hat
    rng = np.random.default_rng(seed)
    def clap():
        m = int(0.25 * SR); t = np.arange(m) / SR; e = np.zeros(m)
        for d in (0, 0.009, 0.019):
            e += (t >= d) * np.exp(-(t - d) / 0.006) * (t >= d)
        e += (t >= 0.028) * np.exp(-(np.maximum(t - 0.028, 0)) / 0.09) * 0.7
        x = run(rng.standard_normal(m) * e, [P.HighpassFilter(900), P.PeakFilter(1600, 5, 1.2), P.LowpassFilter(9000)])
        return x / np.abs(x).max()
    def hat():
        m = int(0.12 * SR); t = np.arange(m) / SR
        x = run(rng.standard_normal(m) * np.exp(-t / 0.035), [P.HighpassFilter(7000), P.PeakFilter(10000, 4, 1)])
        return x / np.abs(x).max()
    perc = np.zeros((n, 2))
    for i in range(bars * 4):
        if i % 2 == 1:
            c = clap(); a = int(i * beat * SR); perc[a:a + len(c)] += np.stack([c, c], 1) * 0.55
        h = hat(); a = int((i + 0.5) * beat * SR)
        if a + len(h) <= n: perc[a:a + len(h)] += np.stack([h * 0.9, h], 1) * 0.18
    wet = run(perc, [P.Reverb(0.5, 0.5, 1.0, 0.0, 1.0), P.HighpassFilter(500)])
    perc = perc + wet * 0.15
    # euphoric lead: long notes, i-VI-III-VII melody in F minor, octave doubled supersaw
    mel = [(0, 3, 77), (3, 3, 80), (6, 2, 79), (8, 4, 77), (12, 4, 73), (16, 3, 75), (19, 3, 77), (22, 2, 80),
           (24, 6, 79), (30, 2, 75)]
    lead = np.zeros((n, 2))
    for rep in range(bars // 8):
        for (st, d, m) in mel:
            a = int((rep * 32 + st) * beat * SR); L = int(d * beat * SR)
            f = 440 * 2 ** ((m - 69) / 12)
            s = supersaw(f, L, 9, rng=rng) + supersaw(f * 2, L, 7, rng=rng) * 0.4
            env = np.minimum(1, np.arange(L) / (0.005 * SR)) * np.clip((L - np.arange(L)) / (0.12 * SR), 0, 1)
            lead[a:a + L] += s * env[:, None]
    lead = run(lead, [P.HighpassFilter(250), P.LowpassFilter(9000), P.PeakFilter(400, -3, 0.8), P.Distortion(4)])
    lead = ott(lead, depth=0.3)
    lrev = run(lead, [P.Reverb(0.8, 0.5, 1.0, 0.0, 1.0), P.HighpassFilter(500), P.LowpassFilter(7000)])
    sc_lead = duck_curve(n, hits, beat, depth=0.5, recover=0.3, shape=1.5)
    sc_rev = duck_curve(n, hits, beat, depth=1.0, recover=0.6, shape=1.5)
    def norm(x, peak): return x / (np.abs(x).max() + 1e-9) * peak
    mix = norm(kick_buf, 1.0) + norm(perc, 0.45) + norm(lead, 0.32) * sc_lead[:, None] + norm(lrev, 0.22) * sc_rev[:, None]
    return master(mix, -6.0), kick_buf


if __name__ == "__main__":
    from pedalboard.io import AudioFile
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for name in FLAVOURS:
        m, k = context_loop(name)
        for tag, x in (("loop", m), ("solo", k[: int(8 * 0.4 * SR)] * 0.9)):
            with AudioFile(f"{out}/hs_{name}_{tag}.wav", "w", SR, 2) as f:
                f.write(x.T.astype(np.float32))
        print(name, "ok")
