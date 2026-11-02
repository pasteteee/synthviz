"""Note ribbon: one continuous timeline at the bottom of the frame where melody,
bass, chords and drums all scroll at the same speed toward one playhead.

Replaces the v3 'glass card'. Design rules (from the visual research):
  * one time axis for everything, so a kick, a note and a chord change that
    happen together line up vertically and visibly belong together;
  * nothing teleports: the pitch range follows the music with a smoothed
    (zero-phase filtered) window, labels slide in from the right and stick
    to the left edge like section headers;
  * a note is drawn dim before it plays and fills with light as the playhead
    passes, so the viewer can 'read ahead' (the Synthesia effect)."""
from __future__ import annotations

import bisect
import colorsys
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import kernels as K
from . import kernels2 as K2


def _c(h, s=0.6, v=1.0):
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


class NoteRibbon:
    XL, XR = 64.0, 940.0
    PLAY_X = 300.0
    PPS = 210.0                     # pixels per second (all lanes)
    YT, YB = 1212.0, 1356.0         # melodic lanes
    BASS_Y = 1378.0
    DRUM_Y = 1418.0
    LABEL_Y = 1180.0

    def __init__(self, F, W=1080, H=1920):
        self.F, self.W, self.H = F, W, H
        ev = F.ev
        self.past = (self.PLAY_X - self.XL) / self.PPS
        self.future = (self.XR - self.PLAY_X) / self.PPS
        self.lines = []
        for kind in ("harm", "counter", "lead"):
            ns = sorted(ev.get(kind) or [], key=lambda n: n["t"])
            if ns:
                self.lines.append((kind, ns, [n["t"] for n in ns]))
        self.bass = sorted(ev.get("bass") or [], key=lambda n: n["t"])
        self.bass_t = [n["t"] for n in self.bass]
        self.pads = [(sorted(l, key=lambda n: n["t"])) for l in ev.get("pad_lines", []) if l]
        # ---- smoothed pitch window (no jumps): per-frame min/max of what's on screen
        n = F.n_frames
        lo = np.full(n, np.nan); hi = np.full(n, np.nan)
        allm = [x for _, ns, _ in self.lines for x in ns]
        allm.sort(key=lambda x: x["t"])
        ts = np.array([x["t"] for x in allm]) if allm else np.zeros(0)
        ps = np.array([x["p"] for x in allm]) if allm else np.zeros(0)
        for i in range(n):
            t = i / F.fps
            a, b = np.searchsorted(ts, [t - self.past, t + self.future])
            if b > a:
                lo[i], hi[i] = ps[a:b].min(), ps[a:b].max()
        def fill(x, default):
            if np.all(np.isnan(x)):
                return np.full(n, default, float)
            idx = np.arange(n); m = ~np.isnan(x)
            return np.interp(idx, idx[m], x[m])
        lo = fill(lo, 60); hi = fill(hi, 72)
        mid = (lo + hi) / 2; span = np.maximum(hi - lo, 12)
        sig = F.fps * 0.9
        self.mid = gaussian_filter1d(mid, sig, mode="nearest")
        self.span = gaussian_filter1d(span, sig, mode="nearest") + 4
        bp = [x["p"] for x in self.bass] or [36]
        self.b_lo, self.b_hi = min(bp) - 1, max(bp) + 1

    def x_of(self, tt, t):
        return self.PLAY_X + (tt - t) * self.PPS

    def y_of(self, p, i):
        i = min(i, len(self.mid) - 1)
        return self.YB - ((p - self.mid[i]) / self.span[i] + 0.5) * (self.YB - self.YT)

    # ------------------------------------------------------------------
    def draw(self, img, i, t, hue, alpha=1.0):
        if alpha <= 0.01:
            return
        F = self.F
        K2.darken_band(img, self.LABEL_Y - 60, self.DRUM_Y + 50, 0.62 * alpha)
        t_a, t_b = t - self.past, t + self.future
        kick = F.pulse("kick", t, 0.12)
        # pads: faint glide lines (harmony bed)
        pr, pg, pb = _c(hue, 0.45, 0.8)
        for line in self.pads:
            for nn in line:
                if nn["t"] + nn["dur"] < t_a or nn["t"] > t_b:
                    continue
                x0 = max(self.XL, self.x_of(nn["t"], t)); x1 = min(self.XR, self.x_of(nn["t"] + nn["dur"], t))
                if x1 - x0 < 2:
                    continue
                y = self.y_of(nn["p"], i)
                if not (self.YT - 20 < y < self.YB + 20):
                    continue
                K.polyline(img, np.array([x0, x1]), np.array([y, y]), pr, pg, pb, 0.10 * alpha, 1.2)
        # arp: tiny sparks that light up as they pass the playhead (keeps intros alive)
        arp = F.arp
        if arp:
            ar, ag, ab = _c(hue + 0.25, 0.5)
            a = bisect.bisect_left(F.arp_t, t_a); bidx = bisect.bisect_right(F.arp_t, t_b)
            for nn in arp[a:bidx]:
                x = self.x_of(nn["t"], t)
                y = self.y_of(nn["p"], i)
                if not (self.YT - 10 < y < self.YB + 10):
                    continue
                age = t - nn["t"]
                fl = math.exp(-age / 0.1) if age >= 0 else 0.0
                fade = min(1.0, (x - self.XL) / 50.0, (self.XR - x) / 80.0)
                K2.glow_dot(img, x, y, 1.5 + 2 * fl, ar, ag, ab, (0.12 + 0.9 * fl) * alpha * fade)
        # melodic lanes as capsules that fill with light when played
        styles = {"lead": (hue + 0.02, 0.18, 14.0), "counter": (hue + 0.33, 0.55, 10.0), "harm": (hue + 0.12, 0.45, 8.0)}
        for kind, ns, ts in self.lines:
            h_, s_, th = styles[kind]
            r, g, b = _c(h_, s_)
            a = max(0, bisect.bisect_left(ts, t_a - 4.0))
            bidx = bisect.bisect_right(ts, t_b)
            for nn in ns[a:bidx]:
                if nn["t"] + nn["dur"] < t_a:
                    continue
                x0 = self.x_of(nn["t"], t); x1 = self.x_of(nn["t"] + nn["dur"], t) - 2
                x1 = max(x1, x0 + th * 0.8)
                xa, xb = max(self.XL, x0), min(self.XR, x1)
                if xb <= xa:
                    continue
                y = self.y_of(nn["p"], i)
                if not (self.YT - 30 < y < self.YB + 30):
                    continue
                edge_fade = min(1.0, (xb - self.XL) / 60.0, (self.XR - xa) / 80.0)
                rad = th / 2
                # outline (future)
                K2.glow_rect(img, xa, y - rad, xb, y + rad, r, g, b, 0.05 * alpha * edge_fade,
                             0.22 * alpha * edge_fade, rad)
                # played part
                if x0 <= self.PLAY_X:
                    xp = min(xb, self.PLAY_X)
                    age = t - nn["t"]
                    hot = math.exp(-age / 0.25)
                    playing = nn["t"] <= t <= nn["t"] + nn["dur"]
                    amp = (0.35 + (0.55 if playing else 0.0) + 0.9 * hot) * alpha * edge_fade
                    if xp - xa > 1:
                        K2.glow_rect(img, xa, y - rad, xp, y + rad, r, g, b, amp, 0.3 * amp, rad)
                    if age < 0.35 and kind == "lead":
                        K2.glow_dot(img, self.PLAY_X, y, 3 + 10 * (1 - hot), r, g, b, 0.9 * hot * alpha)
        # bass lane: thick low capsules, height follows pitch a little
        r, g, b = _c(hue + 0.62, 0.65)
        a = max(0, bisect.bisect_left(self.bass_t, t_a - 2.0))
        bidx = bisect.bisect_right(self.bass_t, t_b)
        for nn in self.bass[a:bidx]:
            x0 = self.x_of(nn["t"], t); x1 = self.x_of(nn["t"] + nn["dur"], t) - 3
            xa, xb = max(self.XL, x0), min(self.XR, x1)
            if xb - xa < 3:
                continue
            y = self.BASS_Y + 6 - 12 * (nn["p"] - self.b_lo) / max(1, self.b_hi - self.b_lo)
            played = x0 <= self.PLAY_X
            amp = (0.55 + 0.9 * math.exp(-(t - nn["t"]) / 0.2) if played else 0.12) * alpha
            K2.glow_rect(img, xa, y - 5, xb, y + 5, r, g, b, amp, 0.25 * amp + 0.06 * alpha, 5.0)
        # drum lane: kicks = blocks, snares = big dots, hats = ticks; flash on the playhead
        lanes = (("kick", 0.0, 9.0), ("snare", 0.14, 6.0), ("hat", 0.5, 2.0), ("perc", 0.3, 3.0))
        # drum fills (events["fills"]): the drum lane swells while a fill plays
        fill_k = 0.0
        for fl_ in F.ev.get("fills") or []:
            if fl_["t"] - 0.05 <= t < fl_.get("t1", fl_["t"] + 0.5) + 0.15:
                fill_k = 1.0 if fl_.get("big") else 0.6
        for lane, dh, size in lanes:
            ts = F.lane_t.get(lane) or []
            if not ts:
                continue
            rr, gg, bb = _c(hue + dh, 0.55)
            a = bisect.bisect_left(ts, t_a); bidx = bisect.bisect_right(ts, t_b)
            for th_ in ts[a:bidx]:
                x = self.x_of(th_, t)
                age = t - th_
                fl = math.exp(-age / 0.12) if age >= 0 else 0.0
                fade = min(1.0, (x - self.XL) / 50.0, (self.XR - x) / 80.0)
                amp = ((0.5 if age >= 0 else 0.22) + (2.0 + 1.5 * fill_k) * fl) * alpha * fade
                size_ = size * (1 + 0.5 * fill_k)
                yoff = {"kick": 0, "snare": 0, "hat": -14, "perc": 14}[lane]
                if lane == "kick":
                    s = size_ * (1 + 0.4 * fl)
                    K2.glow_rect(img, x - s, self.DRUM_Y - s * 0.6, x + s, self.DRUM_Y + s * 0.6, rr, gg, bb,
                                 amp * 0.7, amp * 0.3, 3.0)
                else:
                    K2.glow_dot(img, x, self.DRUM_Y + yoff, size_ * (1 + 0.6 * fl), rr, gg, bb, amp)
        # playhead
        lr, lg, lb = _c(hue + 0.5, 0.25)
        K2.glow_rect(img, self.PLAY_X - 1, self.YT - 22, self.PLAY_X + 1, self.DRUM_Y + 18, lr, lg, lb,
                     (0.35 + 0.4 * kick) * alpha, 0.12 * alpha, 1.0)

    def chord_labels(self, t):
        """[(x, chord, state)] for the overlay: state 0 future, 1 current (sticky), 2 leaving."""
        F = self.F
        out = []
        idx = max(0, bisect.bisect_right(F.chord_t, t) - 1)
        for j in range(max(0, idx - 1), min(len(F.chords), idx + 4)):
            ch = F.chords[j]
            x = self.x_of(ch["t"], t)
            if x > self.XR - 40:
                continue
            nxt = self.x_of(F.chords[j + 1]["t"], t) if j + 1 < len(F.chords) else 1e9
            if j <= idx:
                x = max(self.XL + 6, x) if j == idx else x
                x = min(x, nxt - 150)  # pushed out by the next one
            out.append((x, ch, 1 if j == idx else (2 if j < idx else 0)))
        return out
