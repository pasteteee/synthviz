"""Scenes: stateful wrappers around kernels, driven by Features."""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

from . import kernels as K
from ..theory import JUST_RATIOS


def chord_hue(root_pc):
    """Circle-of-fifths position -> hue (neighbouring keys get neighbouring colours)."""
    return ((root_pc * 7) % 12) / 12.0


@njit(parallel=True, fastmath=True, cache=True)
def upsample(src, dst):
    H, W = dst.shape[0], dst.shape[1]
    sh, sw = src.shape[0], src.shape[1]
    fx = sw / W; fy = sh / H
    for y in prange(H):
        for x in range(W):
            sx = (x + 0.5) * fx - 0.5; sy = (y + 0.5) * fy - 0.5
            sx = min(max(sx, 0.0), sw - 1.001); sy = min(max(sy, 0.0), sh - 1.001)
            ix = int(sx); iy = int(sy); ax = sx - ix; ay = sy - iy
            for c in range(3):
                dst[y, x, c] = (src[iy, ix, c] * (1 - ax) * (1 - ay) + src[iy, ix + 1, c] * ax * (1 - ay) +
                                src[iy + 1, ix, c] * (1 - ax) * ay + src[iy + 1, ix + 1, c] * ax * ay)


class Scene:
    hue_range = 1.0
    base_hue = 0.0

    def __init__(self, F, W, H, rng):
        self.F, self.W, self.H, self.rng = F, W, H, rng
        self.hue = None

    def update_hue(self, t, dt):
        _, ch, _ = self.F.chord(t)
        target = self.base_hue + chord_hue(ch["root"]) * self.hue_range
        if self.hue is None:
            self.hue = target
        d = (target - self.hue + 0.5) % 1.0 - 0.5
        self.hue += d * min(1.0, dt * 2.5)
        return self.hue

    def common(self, i, t):
        F = self.F
        return dict(kick=F.pulse("kick", t, 0.11), snare=F.pulse("snare", t, 0.15),
                    impact=F.pulse("impact", t, 0.5), energy=float(F.energy[i]),
                    spec=F.spec[i].astype(np.float64), beat=F.beat(t), travel=float(F.travel[i]))

    def scope_line(self, img, t, y0, amp, hue, inten=1.0, src=None, thickness=1.6):
        w = self.F.wave(t, 1400, src=src)
        xs = np.linspace(self.W * 0.04, self.W * 0.96, len(w)).astype(np.float64)
        ys = (y0 - w / (np.max(np.abs(w)) + 0.15) * amp).astype(np.float64)
        r, g, b = _hsv(hue, 0.5, 1.0)
        K.polyline(img, xs[::2], ys[::2], r, g, b, inten, thickness)


def _hsv(h, s, v):
    import colorsys
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


# --------------------------------------------------------------------------
class Outrun(Scene):
    name = "outrun"
    hue_range = 0.35

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        K.outrun(img, t, c["beat"], c["kick"], hue, c["energy"], c["travel"], c["spec"], c["snare"])
        lead = float(self.F.env["lead"][i]) + float(self.F.env["arp"][i]) * 0.5



class Tunnel(Scene):
    name = "tunnel"
    hue_range = 0.6

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pattern_by_section = {}
        self.base_hue = self.rng.random()

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        sec = self.F.section(t)["name"]
        pattern = {"intro": 2, "build": 1, "drop": 0, "break": 2, "drop2": 1, "outro": 2}.get(sec, 0)
        cx = self.W * 0.5 + self.W * 0.10 * math.sin(t * 0.31)
        cy = self.H * 0.45 + self.H * 0.06 * math.cos(t * 0.23)
        twist = 0.35 * math.sin(t * 0.12)
        K.tunnel(img, t, c["travel"], c["kick"], c["snare"], hue, c["energy"], 12.0, twist, cx, cy, c["spec"], pattern)


class Kaleido(Scene):
    name = "kaleido"
    hue_range = 0.5

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rot = 0.0
        self.zoom = 1.6
        self.base_hue = self.rng.random()

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        sec = self.F.section(t)["name"]
        nfold = {"intro": 6, "build": 8, "drop": 12, "break": 5, "drop2": 10, "outro": 6}.get(sec, 8)
        bass = float(self.F.env["bass"][i])
        self.rot += dt * (0.06 + 0.25 * c["energy"]) + 0.04 * c["snare"] * dt * 10
        zt = 1.7 - 0.25 * c["energy"] - 0.18 * c["kick"]
        self.zoom += (zt - self.zoom) * min(1, dt * 8)
        c1 = 0.62 + 0.10 * math.sin(t * 0.13) + 0.04 * bass
        c2 = 0.55 + 0.10 * math.cos(t * 0.097)
        K.kaleido(img, t, self.zoom, self.rot, float(nfold), hue, c["kick"], c["energy"], c1, c2, self.H * 0.45)


class Ridges(Scene):
    name = "ridges"
    hue_range = 0.3
    M = 44
    P = 160

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lines = np.zeros((self.M, self.P))
        self.acc = 0.0
        self.base_hue = 0.55
        self.noise_seed = self.rng.random() * 100

    def _make_line(self, i, t):
        spec = self.F.spec[i]
        nb = len(spec)
        half = self.P // 2
        x = np.linspace(0, 1, half)
        s = np.interp(x * (nb - 12), np.arange(nb), spec)
        line = np.concatenate([s[::-1], s])
        jitter = np.random.default_rng(int(t * 1000)).random(self.P) * 0.18 * (0.3 + float(self.F.energy[i]))
        return line * (0.9 + 0.3 * float(self.F.env["kick"][i])) + jitter

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        rate = 12.0  # new lines per second
        self.acc += dt * rate
        while self.acc >= 1.0:
            self.lines[:-1] = self.lines[1:]
            self.lines[-1] = self._make_line(i, t)
            self.acc -= 1.0
        # newest line continuously updated so the front ridge is live
        self.lines[-1] = self._make_line(i, t)
        K.ridgelines(img, self.lines, self.acc - 1.0, hue, c["kick"], c["energy"], t)


class Nebula(Scene):
    name = "nebula"
    hue_range = 0.4
    N_STARS = 900

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.small = np.zeros((self.H // 3, self.W // 3, 3), np.float32)
        r = self.rng
        self.sx = np.array([r.uniform(-1.5, 1.5) for _ in range(self.N_STARS)])
        self.sy = np.array([r.uniform(-2.5, 2.5) for _ in range(self.N_STARS)])
        self.sz = np.array([r.uniform(0.05, 1.0) for _ in range(self.N_STARS)])
        self.base_hue = r.random()
        self.drift = 0.0
        self.rot = 0.0

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        self.drift += dt * (0.02 + 0.06 * c["energy"])
        K.nebula(self.small, t, hue, c["energy"], c["kick"], self.drift)
        upsample(self.small, img)
        # warp stars
        speed = (0.08 + 0.55 * c["energy"] ** 2 + 0.25 * c["impact"]) * dt
        z0 = self.sz.copy()
        self.sz -= speed
        dead = self.sz < 0.03
        self.sz[dead] += 1.0
        z0[dead] = self.sz[dead]
        cx, cy = self.W * 0.5, self.H * 0.40
        f = self.W * 0.35
        n_s = 6
        xs, ys, amp = [], [], []
        for k_ in range(n_s):
            z = z0 + (self.sz - z0) * k_ / (n_s - 1)
            xs.append(cx + self.sx / z * f); ys.append(cy + self.sy / z * f)
            amp.append(np.clip(0.06 / z, 0, 2.5) / n_s * (1 + 2 * speed / dt * 0.2))
        xs = np.concatenate(xs); ys = np.concatenate(ys); amp = np.concatenate(amp)
        cols = np.ones((len(xs), 3)) * np.array(_hsv(hue + 0.55, 0.25, 1.0))
        K.splat(img, xs, ys, cols, amp)
        self.rot += dt * 0.15
        R0 = self.W * 0.22 * (1 + 0.03 * c["kick"])
        K.planet_ring(img, t, hue, c["energy"], c["kick"], c["spec"], cx, cy, R0, self.rot)


class Scope(Scene):
    """Harmonograph / Lissajous oscilloscope. The curve's frequency ratio is the
    *just-intonation ratio of the interval* between the chord root and the
    current melody note: a fifth draws 3:2, a major third 5:4, octave 2:1..."""
    name = "scope"
    hue_range = 0.25

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.phos = np.zeros((self.H, self.W, 3), np.float32)
        self.a = 1.0; self.b = 1.0; self.c = 2.0
        self.ph = 0.0
        self.yaw = 0.0
        self.base_hue = 0.33 + self.rng.choice([0.0, 0.17, 0.5])
        self.bg = np.zeros_like(self.phos)

    def render(self, img, i, t, dt):
        F = self.F
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        _, ch, _ = F.chord(t)
        note, since = F.note_at(F.lead, F.lead_t, t)
        interval = 7
        if note is not None and since < note["dur"] + 0.4:
            interval = (note["p"] - ch["root"]) % 12
        num, den = JUST_RATIOS[interval]
        while num > 9 or den > 9:  # keep figures readable
            num, den = max(1, round(num / 2)), max(1, round(den / 2))
        third = ch["intervals"][1] % 12 if len(ch["intervals"]) > 1 else 7
        c3 = JUST_RATIOS[third][0] / JUST_RATIOS[third][1]
        k = min(1.0, dt * 3.0)
        self.a += (num - self.a) * k
        self.b += (den - self.b) * k
        self.c += (c3 * 2 - self.c) * k
        self.ph += dt * (0.35 + 0.8 * c["energy"])
        self.yaw += dt * (0.25 + 0.4 * c["energy"])
        self.phos *= 0.80
        lead = float(F.env["lead"][i])
        n = 9000
        s = np.linspace(0, 2 * np.pi * 2, n)
        x3 = np.sin(self.a * s + self.ph)
        y3 = np.sin(self.b * s)
        z3 = np.sin(self.c * s + self.ph * 0.7)
        cy_, sy_ = math.cos(self.yaw), math.sin(self.yaw)
        xr = x3 * cy_ + z3 * sy_
        zr = -x3 * sy_ + z3 * cy_
        pitch = 0.35 * math.sin(t * 0.21)
        yr = y3 * math.cos(pitch) - zr * math.sin(pitch)
        zr2 = y3 * math.sin(pitch) + zr * math.cos(pitch)
        persp = 1.0 / (2.6 - zr2)
        R = self.W * 1.35 * (0.75 + 0.18 * lead + 0.06 * c["kick"])
        xs = self.W * 0.5 + xr * persp * R
        ys = self.H * 0.42 + yr * persp * R
        depth = (zr2 + 1) * 0.5
        amp = (0.25 + 0.6 * depth) * (0.6 + 0.7 * c["energy"] + 0.6 * c["kick"]) * 1.1
        col = np.array(_hsv(hue, 0.55, 1.0))
        col2 = np.array(_hsv(hue + 0.12, 0.8, 1.0))
        cols = col[None, :] * (1 - depth[:, None]) + col2[None, :] * depth[:, None]
        K.splat(self.phos, xs, ys, cols, amp * np.ones(n))
        # second figure: bass vs root (octaves / fifths), small, counter-rotating
        bnote, bs = F.note_at(F.bass, F.bass_t, t)
        bint = 0 if bnote is None else (bnote["p"] - ch["root"]) % 12
        bn, bd = JUST_RATIOS[bint]
        while bn > 6 or bd > 6:
            bn, bd = max(1, round(bn / 2)), max(1, round(bd / 2))
        s2 = np.linspace(0, 2 * np.pi, 3000)
        bx = np.sin(bn * s2 + self.ph * 1.3) * self.W * 0.12 * (1 + 0.3 * float(F.env["bass"][i]))
        by = np.sin(bd * 2 * s2) * self.W * 0.12
        cr = math.cos(-self.yaw * 0.7); sr = math.sin(-self.yaw * 0.7)
        bxs = self.W * 0.5 + bx * cr - by * sr
        bys = self.H * 0.42 + bx * sr + by * cr
        bcol = np.array(_hsv(hue + 0.5, 0.6, 1.0))
        K.splat(self.phos, bxs, bys, np.tile(bcol, (len(s2), 1)), np.full(len(s2), 0.35 + 0.6 * float(F.env["bass"][i])))
        K.graticule(img, hue, 0.8)
        img += self.phos



SCENES = {cls.name: cls for cls in (Outrun, Tunnel, Kaleido, Ridges, Nebula, Scope)}
