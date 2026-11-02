"""Camera rig: zoom / pan curves precomputed for the whole video.

v3 kicked the zoom and shake per frame (exp decay with an instant attack and
sin(t*91) shake), which reads as jitter. Here every response is a smooth
'alpha' pulse (rise ~50 ms, no instant jump), motion only grows with the
arrangement (push-in during builds, release on the drop) and shake is used
only on impacts as one damped low-frequency wobble."""
from __future__ import annotations

import math

import numpy as np


def _alpha_pulse(n, fps, times, amp, tau):
    """Sum of x/tau*exp(1-x/tau) pulses (peak = amp at x = tau)."""
    out = np.zeros(n, np.float64)
    L = int(tau * 8 * fps) + 2
    x = np.arange(L) / fps
    k = (x / tau) * np.exp(1 - x / tau)
    for t in times:
        i0 = int(math.ceil(t * fps))
        if i0 >= n:
            continue
        frac = i0 / fps - t
        kk = ((x + frac) / tau) * np.exp(1 - (x + frac) / tau)
        e = min(n, i0 + L)
        out[i0:e] += amp * kk[: e - i0]
    return out


def _ease_in(x):
    return x * x * x


def _ease_out(x):
    return 1 - (1 - x) ** 3


class CameraRig:
    def __init__(self, F, cuts):
        """cuts: list of (t, kind) where kind in {'dive', 'cross', 'flash'}."""
        n, fps = F.n_frames, F.fps
        t = np.arange(n) / fps
        spb = F.ev["spb"]
        zoom = np.ones(n)
        for s in F.sections:
            a, b = s["t"], s["t"] + s["dur"]
            m = (t >= a) & (t < b)
            u = (t[m] - a) / max(1e-3, b - a)
            if s["name"] == "build":
                zoom[m] += 0.07 * _ease_in(u)                  # anticipation push-in
            elif s["name"] in ("drop", "drop2"):
                zoom[m] += 0.02 + 0.01 * np.sin(2 * np.pi * (t[m] - a) / (16 * spb))
            elif s["name"] == "break":
                zoom[m] += 0.03 * u                            # slow drift in
            elif s["name"] == "outro":
                zoom[m] += 0.02 * (1 - u)
        in_drop = np.zeros(n, bool)
        for s in F.sections:
            if s["name"] in ("drop", "drop2"):
                in_drop |= (t >= s["t"]) & (t < s["t"] + s["dur"])
        kicks = np.array(F.kicks)
        kd = [k for k in kicks if in_drop[min(n - 1, int(k * fps))]]
        ko = [k for k in kicks if not in_drop[min(n - 1, int(k * fps))]]
        zoom += _alpha_pulse(n, fps, kd, 0.014, 0.05) + _alpha_pulse(n, fps, ko, 0.006, 0.05)
        zoom += _alpha_pulse(n, fps, F.impacts, 0.05, 0.07)
        # sub hits: a deeper, slower push than a kick
        for bd in F.ev.get("bass_drops") or []:
            zoom += _alpha_pulse(n, fps, [bd["t"]], 0.02 * bd.get("strength", 1.0), 0.09)
        # cuts: 'dive' = accelerate into the centre and arrive pulled-in
        for tc, kind in cuts:
            if kind == "dive":
                m = (t >= tc - 0.4) & (t < tc)
                zoom[m] += 0.22 * _ease_in((t[m] - (tc - 0.4)) / 0.4)
                m = (t >= tc) & (t < tc + 0.7)
                zoom[m] += 0.12 * (1 - _ease_out((t[m] - tc) / 0.7))
        for e in F.fx:
            if e["kind"] == "gap":
                m = (t >= e["t"]) & (t < e["t"] + e["dur"])
                zoom[m] += 0.05 * ((t[m] - e["t"]) / e["dur"]) ** 2
        # slow pan sway (8 bars) + damped wobble on impacts only
        sway = 5.0 * np.sin(2 * np.pi * t / (32 * spb))
        wob_x = np.zeros(n); wob_y = np.zeros(n)
        for ti in F.impacts:
            m = t >= ti
            x = t[m] - ti
            env = np.exp(-x / 0.22)
            wob_x[m] += 9 * env * np.sin(2 * np.pi * 6.0 * x)
            wob_y[m] += 6 * env * np.sin(2 * np.pi * 4.5 * x + 1.0)
        self.zoom = zoom.astype(np.float32)
        self.sx = (sway + wob_x).astype(np.float32)
        self.sy = wob_y.astype(np.float32)

    def at(self, i):
        i = min(i, len(self.zoom) - 1)
        return float(self.zoom[i]), float(self.sx[i]), float(self.sy[i])
