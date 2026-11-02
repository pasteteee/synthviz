"""Finale: particles that assemble into a SUBSCRIBE button and ♥ LIKE with a different
choreography every video (converge, rain, vortex, pixel blocks, laser scan, warp)."""
from __future__ import annotations

import colorsys
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import kernels as K

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
EFFECTS = ["converge", "rain", "vortex", "blocks", "scan", "warp"]


def _ease_out(x):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** 3


def _ease_back(x, s=1.7):
    x = np.clip(x, 0, 1) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


class CTA:
    def __init__(self, F, W, H, rng, t_start, line1="SUBSCRIBE", line2="LIKE", effect=None):
        self.F, self.W, self.H = F, W, H
        self.t0 = t_start
        self.rng = np.random.default_rng(rng.randrange(10 ** 6))
        self.effect = effect or rng.choice(EFFECTS)
        pts, kind = self._targets(line1, line2)
        self.tx, self.ty = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
        self.kind = kind  # 0 text, 1 heart, 2 line2
        n = len(self.tx)
        r = self.rng
        cx, cy = W / 2, H * 0.42
        if self.effect == "converge":
            ang = r.uniform(0, 2 * np.pi, n); rad = r.uniform(0.6, 1.4, n) * H
            self.sx, self.sy = cx + np.cos(ang) * rad, cy + np.sin(ang) * rad
        elif self.effect == "rain":
            self.sx = self.tx + r.normal(0, 30, n); self.sy = -r.uniform(50, H * 0.9, n)
        elif self.effect == "vortex":
            self.a0 = r.uniform(0, 2 * np.pi, n); self.r0 = r.uniform(0.25, 0.6, n) * W
        elif self.effect == "blocks":
            q = 14
            self.tx = np.round(self.tx / q) * q; self.ty = np.round(self.ty / q) * q
            side = r.integers(0, 4, n)
            self.sx = np.where(side == 0, -100, np.where(side == 1, W + 100, self.tx + r.normal(0, 200, n)))
            self.sy = np.where(side == 2, -100, np.where(side == 3, H + 100, self.ty + r.normal(0, 200, n)))
        elif self.effect == "warp":
            self.z0 = r.uniform(4, 12, n)
        self.delay = r.uniform(0, 0.35, n)
        if self.effect in ("rain", "scan"):
            self.delay = (self.tx - self.tx.min()) / (np.ptp(self.tx) + 1) * 0.55 + r.uniform(0, 0.08, n)
        self.jit = r.uniform(0, 2 * np.pi, n)

    def _targets(self, line1, line2):
        W, H = self.W, self.H
        img = Image.new("L", (W, H), 0)
        pill = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        dp = ImageDraw.Draw(pill)
        sp1, sp2 = 14, 18

        def spaced(text, font, sp):
            ws = [d.textlength(ch, font=font) for ch in text]
            return ws, sum(ws) + sp * (len(text) - 1)

        px = 150
        f1 = ImageFont.truetype(str(FONT_DIR / "Oswald.ttf"), px)
        ws, w1 = spaced(line1, f1, sp1)
        while w1 > W * 0.74:
            px -= 6
            f1 = ImageFont.truetype(str(FONT_DIR / "Oswald.ttf"), px)
            ws, w1 = spaced(line1, f1, sp1)
        y1 = H * 0.255
        x = (W - w1) / 2
        for ch, cw in zip(line1, ws):
            d.text((x, y1), ch, font=f1, fill=255)
            x += cw + sp1
        # pill-shaped "button" around line 1
        asc, desc = f1.getmetrics()
        bx0, bx1 = (W - w1) / 2 - 48, (W + w1) / 2 + 48
        by0, by1 = y1 + asc * 0.18, y1 + asc + desc * 0.35 + 30
        r_ = (by1 - by0) / 2
        dp.rounded_rectangle((bx0, by0, bx1, by1), radius=int(r_), outline=255, width=5)
        # line 2: heart + LIKE
        f2 = ImageFont.truetype(str(FONT_DIR / "Oswald.ttf"), 120)
        ws2, w2 = spaced(line2, f2, sp2)
        heart_w, gap = 130, 44
        x2 = (W - (w2 + heart_w + gap)) / 2
        y2 = by1 + 70
        x = x2 + heart_w + gap
        for ch, cw in zip(line2, ws2):
            d.text((x, y2), ch, font=f2, fill=255)
            x += cw + sp2
        a = np.asarray(img)
        ys, xs = np.nonzero(a[::2, ::2] > 128)
        pts = np.stack([xs * 2, ys * 2], axis=1).astype(np.float64)
        pts += self.rng.uniform(-1.0, 1.0, pts.shape)
        kind = np.where(pts[:, 1] < by1, 0, 2)
        pa = np.asarray(pill)
        ys, xs = np.nonzero(pa[::2, ::2] > 128)
        pp = np.stack([xs * 2, ys * 2], axis=1).astype(np.float64) + self.rng.uniform(-0.8, 0.8, (len(xs), 2))
        # heart (filled, parametric)
        n_h = 4200
        tt = self.rng.uniform(0, 2 * np.pi, n_h)
        rr = np.sqrt(self.rng.uniform(0, 1, n_h))
        hx = 16 * np.sin(tt) ** 3
        hy = -(13 * np.cos(tt) - 5 * np.cos(2 * tt) - 2 * np.cos(3 * tt) - np.cos(4 * tt))
        hcx = x2 + heart_w / 2
        _, top2 = f2.getbbox(line2)[:2]
        hcy = y2 + top2 + (f2.getbbox(line2)[3] - top2) / 2
        hp = np.stack([hcx + hx * rr * 3.9, hcy + hy * rr * 3.9], axis=1)
        self.hcx, self.hcy = hcx, hcy
        pts = np.concatenate([pts, hp, pp])
        kind = np.concatenate([kind, np.ones(n_h, int), np.full(len(pp), 3)])
        return pts, kind

    def progress(self, t):
        return (t - self.t0) / 2.4

    def draw(self, img, t, hue, kick):
        p = self.progress(t)
        if p < 0:
            return 0.0
        e_raw = np.clip((p - self.delay) / 0.75, 0, 1)
        e = _ease_out(e_raw)
        W, H = self.W, self.H
        cx, cy = W / 2, H * 0.42
        tx, ty = self.tx, self.ty
        # heart beat after assembly
        beat = 1 + 0.08 * kick * (p > 1.0)
        hmask = self.kind == 1
        tx = np.where(hmask, self.hcx + (tx - self.hcx) * beat, tx)
        ty = np.where(hmask, self.hcy + (ty - self.hcy) * beat, ty)
        amp = np.ones_like(tx)
        if self.effect == "converge":
            sw = (1 - e) * 2.5
            dx, dy = self.sx - tx, self.sy - ty
            x = tx + (dx * np.cos(sw) - dy * np.sin(sw)) * (1 - e)
            y = ty + (dx * np.sin(sw) + dy * np.cos(sw)) * (1 - e)
        elif self.effect == "rain":
            fall = np.clip((p - self.delay) / 0.6, 0, 1)
            y = self.sy + (ty - self.sy) * _ease_back(fall, 1.2)
            x = self.sx + (tx - self.sx) * fall
            amp = 0.4 + 0.6 * fall
        elif self.effect == "vortex":
            ang = self.a0 + (1 - e) * 9.0
            rad = self.r0 * (1 - e)
            x = tx * e + (cx + np.cos(ang) * rad) * (1 - e)
            y = ty * e + (cy + np.sin(ang) * rad) * (1 - e)
        elif self.effect == "blocks":
            eb = _ease_back(e_raw)
            x = self.sx + (tx - self.sx) * eb
            y = self.sy + (ty - self.sy) * eb
            amp = np.ones_like(tx) * 1.6
        elif self.effect == "scan":
            beam = cx - W * 0.55 + p * W * 1.1 / 0.75
            vis = (tx < beam).astype(float)
            ei = np.clip((beam - tx) / 120.0, 0, 1)
            x = tx + (1 - ei) * self.rng.normal(0, 6, len(tx)) * vis
            y = ty - (1 - ei) * 40 * vis
            amp = vis * (0.6 + 1.5 * (1 - ei))
            if beam < W + 20:
                bx = np.full(200, beam); by = np.linspace(H * 0.2, H * 0.6, 200)
                r_, g_, b_ = colorsys.hsv_to_rgb(hue % 1, 0.4, 1.0)
                K.polyline(img, bx, by, r_, g_, b_, 2.0, 2.0)
        else:  # warp: fly in from depth
            z = 1 + (self.z0 - 1) * (1 - e)
            x = cx + (tx - cx) / z
            y = cy + (ty - cy) / z
            amp = 0.3 + 0.7 * e
        # shimmer when assembled
        done = np.clip(p - 1.0, 0, 1)
        x = x + np.sin(t * 3 + self.jit) * 0.8 * done
        y = y + np.cos(t * 2.7 + self.jit) * 0.8 * done
        n = len(x)
        cols = np.empty((n, 3))
        hh = (hue + (tx / W) * 0.25 + 0.45 * (self.kind == 3)) % 1.0
        for k_, (h0, s0) in enumerate([(None, 0.45), (0.97, 0.8), (None, 0.55), (None, 0.85)]):
            m = self.kind == k_
            if not m.any():
                continue
            if h0 is None:
                rgb = np.array([colorsys.hsv_to_rgb(h, s0, 1.0) for h in hh[m]])
            else:
                rgb = np.tile(colorsys.hsv_to_rgb(h0, s0, 1.0), (m.sum(), 1))
            cols[m] = rgb
        a = amp * (3.8 + 1.2 * kick * done) * np.where(e_raw > 0, 1.0, 0.35)
        K.splat(img, x.astype(np.float64), y.astype(np.float64), cols, a.astype(np.float64))
        return float(np.clip(p, 0, 1))
