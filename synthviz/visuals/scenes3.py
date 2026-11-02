"""Third wave of worlds: aurora over a mirror lake, a lava lamp, underwater caustics, a morphing wire knot,
and a field of falling petals under a giant moon. Each kernel fills the whole frame (no stale pixels)."""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

from .kernels import hsv, fbm, smooth, hash2, vnoise3, splat, polyline
from .scenes import Scene, _hsv, upsample

FM = dict(parallel=True, fastmath=True, cache=True)


# ------------------------------------------------------------------ aurora over a mirror lake
@njit(**FM)
def k_aurora(img, t, hue, energy, kick, lead, bass):
    H, W = img.shape[0], img.shape[1]
    hor = H * 0.64
    for y in prange(H):
        for x in range(W):
            yy = float(y)
            refl = 0.0
            if yy > hor:                                         # lake: mirror the sky, rippled
                d = yy - hor
                yy = hor - d * 1.2 + 4.0 * math.sin(d * 0.12 - t * 2.0) * (d / H) * 8.0
                refl = 1.0
            u = x / W; v = yy / H
            r0, g0, b0 = hsv(hue + 0.62, 0.8, 0.06 + 0.10 * v)
            r, g, b = r0, g0, b0
            hh = hash2(float(x), math.floor(yy) + 0.71)
            if hh > 0.996 and yy < hor:
                s = (hh - 0.996) * 220.0 * (0.6 + 0.4 * math.sin(t * 3.0 + hh * 900.0))
                r += s; g += s; b += s
            # two curtains: a wavy glowing lower edge, rays fading upwards, rays flicker along the fold
            for k in range(2):
                edge = 0.30 + 0.12 * k + 0.07 * math.sin(u * (5.0 + 2.0 * k) + t * (0.35 + 0.1 * k) + k * 2.0) \
                       + 0.05 * fbm(u * 3.0 + k * 7.0, t * 0.1, k, 2) - 0.04 * lead
                if v < edge + 0.01:
                    up = (edge - v) / (0.20 + 0.10 * k + 0.08 * bass)
                    a = math.exp(-up * 2.2) * smooth(-0.01, 0.012, edge - v + 0.01)
                    ray = 0.35 + 0.65 * fbm(u * 26.0 + k * 13.0, t * 0.25, 1.0 + k, 2)
                    w = a * ray * (0.75 + 0.35 * energy + 0.35 * bass + 0.25 * kick) * (1.0 - 0.3 * k)
                    ar, ag, ab = hsv(hue + 0.33 + 0.10 * k + 0.35 * min(1.0, up), 0.7, 1.0)
                    r += ar * w; g += ag * w; b += ab * w
            m = hor - H * (0.04 + 0.05 * fbm(u * 4.0, 7.0, 0.0, 3))
            if yy > m and yy <= hor:
                r = r0 * 0.2; g = g0 * 0.2; b = b0 * 0.25
            if refl > 0.0:
                r *= 0.5; g *= 0.55; b *= 0.65
            img[y, x, 0] = r; img[y, x, 1] = g; img[y, x, 2] = b


# ------------------------------------------------------------------ lava lamp (metaballs)
@njit(**FM)
def k_lava(img, t, hue, energy, kick, bx, by, br):
    H, W = img.shape[0], img.shape[1]
    nb = bx.shape[0]
    for y in prange(H):
        for x in range(W):
            f = 0.0
            for i in range(nb):
                dx = x - bx[i]; dy = y - by[i]
                f += br[i] * br[i] / (dx * dx + dy * dy + 1.0)
            v = y / H
            r0, g0, b0 = hsv(hue + 0.75, 0.7, 0.05 + 0.06 * v)
            if f > 1.0:
                e = min(1.0, (f - 1.0) * 2.0)
                c1r, c1g, c1b = hsv(hue + 0.02 + 0.08 * v, 0.85, 1.0)
                c2r, c2g, c2b = hsv(hue + 0.12, 0.6, 1.0)
                rim = 1.0 - e
                k = 0.38 + 0.22 * kick + 0.15 * energy
                img[y, x, 0] = (c1r * e + c2r * rim * 1.1) * k
                img[y, x, 1] = (c1g * e + c2g * rim * 1.1) * k
                img[y, x, 2] = (c1b * e + c2b * rim * 1.1) * k
            else:
                glow = f * f * 0.25
                gr, gg, gb = hsv(hue + 0.05, 0.8, 1.0)
                img[y, x, 0] = r0 + gr * glow; img[y, x, 1] = g0 + gg * glow; img[y, x, 2] = b0 + gb * glow


# ------------------------------------------------------------------ underwater caustics + god rays
@njit(**FM)
def k_ocean(img, t, hue, energy, kick, snare):
    H, W = img.shape[0], img.shape[1]
    for y in prange(H):
        for x in range(W):
            u = x / H * 3.0; v = y / H * 3.0
            # caustic network: abs of warped noise ridges
            w = fbm(u * 0.8, v * 0.8, t * 0.15, 2)
            n1 = vnoise3(u * 2.0 + w * 2.0, v * 2.0, t * 0.6)
            n2 = vnoise3(u * 2.0 - 3.0, v * 2.0 + w * 2.0, t * 0.6 + 4.0)
            c = 1.0 - abs(n1 - n2) * 4.0
            c = c * c * c * c if c > 0.0 else 0.0
            depth = y / H
            r, g, b = hsv(hue + 0.52 + 0.05 * depth, 0.8, 0.30 * (1.0 - depth) + 0.04)
            cr, cg, cb = hsv(hue + 0.45, 0.35, 1.0)
            ci = c * (0.35 + 0.3 * energy + 0.5 * kick) * (1.0 - depth * 0.7)
            r += cr * ci; g += cg * ci; b += cb * ci
            # god rays from the surface
            ray = 0.5 + 0.5 * math.sin((x - y * 0.35) / H * 22.0 + math.sin(t * 0.3) * 2.0)
            ray = ray ** 8 * (1.0 - depth) * (0.25 + 0.35 * snare + 0.15 * energy)
            r += ray * 0.6; g += ray * 0.8; b += ray
            # bubbles
            col = math.floor(x / 36.0)
            hs = hash2(col, 3.3)
            if hs > 0.82:
                by = H - ((t * (120.0 + 200.0 * hs) + hs * 9000.0) % (H + 100.0))
                bx = (col + 0.5) * 36.0 + 8.0 * math.sin(t * 2.0 + hs * 50.0)
                d = math.sqrt((x - bx) ** 2 + (y - by) ** 2)
                rad = 5.0 + 9.0 * hs
                if d < rad:
                    e = smooth(rad - 2.5, rad, d) * 0.9 + 0.08
                    r += e; g += e; b += e
            img[y, x, 0] = r; img[y, x, 1] = g; img[y, x, 2] = b


@njit(**FM)
def k_clear(img, hue, top, bot):
    H, W = img.shape[0], img.shape[1]
    for y in prange(H):
        v = y / H
        r0, g0, b0 = hsv(hue, 0.8, top * (1 - v) + bot * v)
        for x in range(W):
            img[y, x, 0] = r0; img[y, x, 1] = g0; img[y, x, 2] = b0


# ------------------------------------------------------------------ moon + petals
@njit(**FM)
def k_moon(img, t, hue, energy, kick, cx, cy, R):
    H, W = img.shape[0], img.shape[1]
    for y in prange(H):
        v = y / H
        r0, g0, b0 = hsv(hue + 0.70, 0.55, 0.05 + 0.14 * v)
        for x in range(W):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            r = r0; g = g0; b = b0
            if d < R:
                n = fbm(x / R * 2.0, y / R * 2.0, 1.0, 4)
                s = 0.85 + 0.25 * n - 0.15 * smooth(R * 0.6, R, d)
                mr, mg, mb = hsv(hue + 0.08, 0.18, s * (0.9 + 0.2 * kick))
                r = mr; g = mg; b = mb
            else:
                halo = math.exp(-(d - R) / (R * 0.35)) * (0.35 + 0.2 * energy + 0.2 * kick)
                hr, hg, hb = hsv(hue + 0.05, 0.4, 1.0)
                r += hr * halo; g += hg * halo; b += hb * halo
            # rolling hills silhouette
            hill = H * (0.80 + 0.04 * math.sin(x / W * 5.0 + 1.0) + 0.03 * math.sin(x / W * 13.0))
            if y > hill:
                r = r0 * 0.15; g = g0 * 0.15; b = b0 * 0.22
            img[y, x, 0] = r; img[y, x, 1] = g; img[y, x, 2] = b


# ------------------------------------------------------------------ scenes
class Aurora(Scene):
    name = "aurora"
    hue_range = 0.25

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.base_hue = self.rng.random()
        self.lead = 0.0

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        lead = float(self.F.env["lead"][i]); bass = float(self.F.env["bass"][i])
        self.lead += (lead - self.lead) * min(1.0, dt * 6)
        if not hasattr(self, "lo"):
            self.lo = np.zeros((self.H // 2, self.W // 2, 3), np.float32)
        k_aurora(self.lo, t, hue, c["energy"], c["kick"], self.lead, bass)
        upsample(self.lo, img)


class Lava(Scene):
    name = "lava"
    hue_range = 0.2

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.base_hue = self.rng.random()
        n = 12
        r = np.random.default_rng(int(self.rng.random() * 1e6))
        self.ph = r.uniform(0, 2 * np.pi, (n, 2)); self.sp = r.uniform(0.08, 0.22, (n, 2))
        self.rad = r.uniform(0.05, 0.10, n)

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        W, H = self.W, self.H
        bx = W * (0.5 + 0.36 * np.sin(self.ph[:, 0] + t * self.sp[:, 0]))
        by = H * (0.5 + 0.42 * np.sin(self.ph[:, 1] + t * self.sp[:, 1] * 1.3))
        br = W * self.rad * (1.0 + 0.18 * c["kick"] + 0.1 * c["energy"])
        k_lava(img, t, hue, c["energy"], c["kick"], bx, by, br)


class Ocean(Scene):
    name = "ocean"
    hue_range = 0.2

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.base_hue = self.rng.random()

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        if not hasattr(self, "lo"):
            self.lo = np.zeros((self.H // 2, self.W // 2, 3), np.float32)
        k_ocean(self.lo, t, hue, c["energy"], c["kick"], c["snare"])
        upsample(self.lo, img)


class Knot(Scene):
    """A glowing wire torus knot that morphs (p, q) on every section and spins with the energy."""
    name = "knot"
    hue_range = 0.4

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.base_hue = self.rng.random()
        self.rot = 0.0
        self.s = np.linspace(0, 2 * np.pi, 1600)

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        k_clear(img, hue + 0.7, 0.03, 0.08)
        sec = self.F.section(t)["name"]
        p, q = {"intro": (2, 3), "build": (3, 4), "drop": (3, 5), "break": (2, 5), "drop2": (4, 7), "outro": (2, 3)}.get(sec, (2, 3))
        self.rot += dt * (0.3 + 1.2 * c["energy"])
        s = self.s
        R = 1.0 + 0.18 * c["kick"]
        rr = 0.42 + 0.06 * math.sin(t * 0.7)
        x = (R + rr * np.cos(q * s)) * np.cos(p * s)
        y = (R + rr * np.cos(q * s)) * np.sin(p * s)
        z = rr * np.sin(q * s)
        a, b = self.rot, self.rot * 0.63
        y2 = y * math.cos(a) - z * math.sin(a); z2 = y * math.sin(a) + z * math.cos(a)
        x3 = x * math.cos(b) + z2 * math.sin(b); z3 = -x * math.sin(b) + z2 * math.cos(b)
        pers = 1.0 / (3.2 - z3)
        sx = (self.W * 0.5 + x3 * pers * self.W * 1.05).astype(np.float64)
        sy = (self.H * 0.42 + y2 * pers * self.W * 1.05).astype(np.float64)
        for k, (dh, th, amp) in enumerate(((0.0, 14.0, 0.10), (0.0, 6.0, 0.35), (0.0, 3.0, 1.0), (0.5, 1.4, 0.8))):
            r, g, b_ = _hsv(hue + dh + 0.1 * k, 0.6 if k < 2 else 0.3, 1.0)
            polyline(img, sx, sy, r, g, b_, amp * (0.6 + 0.6 * c["energy"] + 0.5 * c["kick"]), th)


class Petals(Scene):
    """A huge moon over dark hills; petals swirl in the wind and burst outward on every kick."""
    name = "petals"
    hue_range = 0.15

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.base_hue = self.rng.random()
        r = np.random.default_rng(int(self.rng.random() * 1e6))
        n = 1400
        self.px = r.uniform(0, self.W, n); self.py = r.uniform(0, self.H, n)
        self.vx = r.uniform(-30, 30, n); self.vy = r.uniform(40, 120, n)
        self.ph = r.uniform(0, 2 * np.pi, n); self.sz = r.uniform(0.5, 1.4, n)

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        W, H = self.W, self.H
        cx, cy, R = W * 0.5, H * 0.36, W * 0.33 * (1 + 0.03 * c["kick"])
        k_moon(img, t, hue, c["energy"], c["kick"], cx, cy, R)
        wind = 60 + 160 * c["energy"]
        dx, dy = self.px - cx, self.py - cy
        dd = np.sqrt(dx * dx + dy * dy) + 1
        push = 900 * c["kick"] * dt
        self.px += (self.vx + wind + np.sin(t * 1.3 + self.ph) * 40) * dt + dx / dd * push
        self.py += (self.vy + np.cos(t * 0.9 + self.ph) * 30) * dt + dy / dd * push
        self.px %= W; self.py %= H
        col = np.array(_hsv(hue + 0.92, 0.35, 1.0))
        cols = np.tile(col, (len(self.px), 1)) * (0.6 + 0.4 * np.sin(self.ph + t * 3)[:, None] ** 2)
        amp = self.sz * (1.2 + 0.8 * c["energy"])
        for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (2, 1), (1, 2), (-1, -1)):
            splat(img, self.px + ox, self.py + oy, cols, amp)


SCENES3 = {c.name: c for c in (Aurora, Lava, Ocean, Knot, Petals)}
