"""Compact neon 'glass card' HUD drawn into the linear image (so it gets bloom):
left column holds the chord (text is drawn later by the overlay), right side a
scrolling note graph where glides read as curves, and below it a 4x16 drum
step grid of glowing dots."""
from __future__ import annotations

import colorsys
import numpy as np

from . import kernels as K
from . import kernels2 as K2


def _c(h, s=0.6, v=1.0):
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


class HUD:
    PAST = 2.2
    FUTURE = 0.8
    # card geometry (inside the Shorts safe zone)
    CX0, CX1, CY0, CY1 = 150.0, 930.0, 1190.0, 1400.0
    COL = 330.0          # end of the chord column

    def __init__(self, F, W=1080, H=1920):
        self.F, self.W, self.H = F, W, H
        self.x0, self.x1 = self.COL + 26, self.CX1 - 26
        self.y0, self.y1 = self.CY0 + 22, self.CY0 + 118      # note graph
        self.d0, self.d1 = self.CY0 + 140, self.CY1 - 18      # drum dots
        self.xp = self.x0 + (self.x1 - self.x0) * self.PAST / (self.PAST + self.FUTURE)
        self.chord_xy = ((self.CX0 + self.COL) / 2, self.CY0 + 92)
        self.roman_xy = ((self.CX0 + self.COL) / 2, self.CY0 + 162)

    def _y(self, p):
        F = self.F
        return self.y1 - (p - F.p_lo) / max(1, F.p_hi - F.p_lo) * (self.y1 - self.y0)

    def draw(self, img, i, t, hue, alpha=1.0):
        if alpha <= 0.01:
            return
        F = self.F
        kick = F.pulse("kick", t, 0.10)
        # glass card
        K2.darken_rrect(img, self.CX0, self.CY0, self.CX1, self.CY1, 26.0, 0.86 * alpha)
        er, eg, eb = _c(hue + 0.55, 0.55)
        K2.glow_rect(img, self.CX0, self.CY0, self.CX1, self.CY1, er, eg, eb, 0.0, (0.10 + 0.10 * kick) * alpha, 26.0)
        # column divider
        dr, dg, db = _c(hue, 0.3)
        K2.glow_rect(img, self.COL, self.CY0 + 26, self.COL + 1, self.CY1 - 26, dr, dg, db, 0.05 * alpha, 0.05 * alpha, 0.5)

        CR = F.CR
        a = int((t - self.PAST) * CR); b = int((t + self.FUTURE) * CR)
        ip = int(t * CR)
        styles = {"pad": (hue + 0.0, 0.55, 0.8, 1.3), "bass": (hue + 0.62, 0.7, 1.1, 1.8),
                  "harm": (hue + 0.1, 0.45, 1.0, 1.3), "counter": (hue + 0.33, 0.6, 1.2, 1.3),
                  "lead": (hue + 0.05, 0.12, 2.2, 2.0)}
        span = self.x1 - self.x0
        for kind, cv in F.curves:
            h, s_, amp, th = styles[kind]
            r, g, b_ = _c(h, s_)
            lo, hi = max(0, a), min(len(cv), b)
            if hi - lo < 2:
                continue
            seg = cv[lo:hi]
            idx = np.arange(lo, hi)
            xs = self.x0 + (idx / CR - (t - self.PAST)) / (self.PAST + self.FUTURE) * span
            ys = self._y(seg)
            valid = ~np.isnan(seg)
            edges = np.flatnonzero(np.diff(np.concatenate([[0], valid.astype(np.int8), [0]])))
            for s0, s1 in zip(edges[::2], edges[1::2]):
                if s1 - s0 < 2:
                    continue
                xx = xs[s0:s1]; yy = ys[s0:s1]
                past = xx <= self.xp
                fade = np.clip((xx - self.x0) / (span * 0.3), 0, 1)
                if past.any():
                    m = past
                    K.polyline(img, xx[m].astype(np.float64), yy[m].astype(np.float64), r, g, b_,
                               amp * alpha * float(fade[m].mean() * 0.7 + 0.3), th)
                if (~past).any():
                    m = ~past
                    m[max(0, np.argmax(m) - 1)] = True
                    K.polyline(img, xx[m].astype(np.float64), yy[m].astype(np.float64), r, g, b_, amp * 0.14 * alpha, th * 0.7)
            if 0 <= ip < len(cv) and not np.isnan(cv[ip]):
                y = float(self._y(cv[ip]))
                K2.glow_dot(img, self.xp, y, 4.0 if kind == "lead" else 2.5, r, g, b_, 1.2 * alpha)
        pr, pg, pb = _c(hue + 0.5, 0.3)
        K2.glow_rect(img, self.xp - 0.5, self.y0 - 6, self.xp + 0.5, self.y1 + 6, pr, pg, pb, 0.2 * alpha, 0.12 * alpha, 0.5)

        # drum dots
        spb = F.ev["spb"]
        bar = int(t / (4 * spb))
        step_now = int((t - bar * 4 * spb) / spb * 4) % 16
        lanes = ["kick", "snare", "hat", "perc"]
        lh = (self.d1 - self.d0) / 4
        cw = span / 16
        for li, lane in enumerate(lanes):
            pats = F.drum_bars.get(lane)
            steps = pats[bar] if pats and bar < len(pats) else set()
            pulse = F.lane_pulse(lane, t, 0.10 if lane != "kick" else 0.14)
            r, g, b_ = _c(hue + [0.0, 0.14, 0.5, 0.3][li], 0.6)
            y = self.d0 + (li + 0.5) * lh
            for sidx in range(16):
                x = self.x0 + (sidx + 0.5) * cw
                on = sidx in steps
                amp = 0.07 + (0.05 if sidx % 4 == 0 else 0.0)
                rad = 1.6
                if on:
                    amp = 0.35
                    rad = 2.6
                    if sidx == step_now:
                        amp += 1.6 * pulse
                        rad += 1.8 * pulse
                elif sidx == step_now:
                    amp = 0.25
                K2.glow_dot(img, x, y, rad, r, g, b_, amp * alpha)
