"""Second wave of scenes: camera flights, pixel art, silk waves, galaxy."""
from __future__ import annotations

import math
import numpy as np

from . import kernels as K
from . import kernels2 as K2
from .scenes import Scene, _hsv


class Terrain(Scene):
    """Camera flight through a synthwave canyon; mountains breathe with the spectrum."""
    name = "terrain"
    hue_range = 0.3

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.base_hue = self.rng.random() * 0.2
        self.z = 0.0

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        self.z += dt * (5.0 + 14.0 * c["energy"] + 10 * c["impact"])
        cam_x = K2.path_x(self.z + 4.0)
        yaw = (K2.path_x(self.z + 12.0) - K2.path_x(self.z)) * 0.04
        cam_h = 3.2 + 0.8 * math.sin(t * 0.4) - 0.6 * c["kick"]
        hor = self.H * (0.47 + 0.02 * math.sin(t * 0.33))
        K2.voxel_terrain(img, cam_x, self.z, cam_h, yaw, hor, hue, t, c["kick"], c["energy"], c["spec"], self.H * 0.55, 70.0)


class City(Scene):
    """Flight down a neon avenue; skyscrapers are spectrum bars."""
    name = "city"
    hue_range = 0.4

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.base_hue = self.rng.random()
        self.z = 0.0

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        self.z += dt * (6.0 + 16.0 * c["energy"] + 12 * c["impact"])
        yaw = 0.12 * math.sin(t * 0.25)
        cam_x = 0.9 * math.sin(t * 0.37)
        cam_h = 2.2 + 1.6 * (0.5 + 0.5 * math.sin(t * 0.21)) - 0.4 * c["kick"]
        hor = self.H * 0.45
        K2.voxel_city(img, cam_x, self.z, cam_h, yaw, hor, hue, t, c["kick"], c["energy"], c["spec"], self.H * 0.6, 60.0)


class Pixel(Scene):
    """8-bit world: dithered sky, pixel sun, parallax hills, equalizer."""
    name = "pixel"
    hue_range = 0.5
    K = 6

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.small = np.zeros((self.H // self.K, self.W // self.K, 3), np.float32)
        self.scroll = 0.0
        self.peaks = np.zeros(16)
        self.base_hue = self.rng.random()

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        self.scroll += dt * (12 + 40 * c["energy"])
        spec = c["spec"]
        bars = np.array([spec[int(b * (len(spec) - 8) / 16) + 2] for b in range(16)])
        self.peaks = np.maximum(bars, self.peaks - dt * 0.5)
        hue_q = round(hue * 12) / 12.0   # palette steps, very 8-bit
        K2.pixel_world(self.small, t, self.scroll, hue_q, c["kick"], c["energy"], spec, self.peaks, K2.BAYER, 5.0)
        K2.upscale_nearest(self.small, img, self.K)
        img *= 0.85


class Waves(Scene):
    """Silk: dozens of glowing wave ribbons in perspective, each fed by a band."""
    name = "waves"
    hue_range = 0.5
    N = 40

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.base_hue = self.rng.random()
        self.ph = 0.0

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        img[:] = 0
        H, W = self.H, self.W
        # background glow
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        br, bg_, bb = _hsv(hue + 0.6, 0.8, 0.10)
        img[:, :, 0] += br * yy * 0.6; img[:, :, 1] += bg_ * yy * 0.6; img[:, :, 2] += bb * yy
        self.ph += dt * (0.8 + 2.5 * c["energy"])
        spec = c["spec"]
        x = np.linspace(-1.3, 1.3, 220)
        hor = H * 0.30
        for j in range(self.N):
            z = 1.0 + j * 0.11                  # depth: front first
            band = spec[int(2 + j * (len(spec) - 10) / self.N)]
            amp = (0.25 + 0.9 * band * (0.4 + 0.8 * c["energy"]) + 0.25 * c["kick"]) / z
            yv = (np.sin(x * 3.1 + self.ph + j * 0.35) * 0.6 + np.sin(x * 7.3 - self.ph * 1.7 + j * 0.2) * 0.25 * band
                  + np.sin(x * 1.3 + self.ph * 0.5) * 0.4) * amp * np.exp(-x * x * 0.6)
            sx = W * 0.5 + x * W * 0.62 / z
            sy = hor + (H * 0.42 + yv * H * 0.35) / z
            r, g, b = _hsv(hue + j * 0.012 + band * 0.1, 0.65, 1.0)
            K.polyline(img, sx, sy, r, g, b, (1.0 + 1.6 * band + 0.6 * c["kick"]) / z ** 0.5, 1.8)


class Galaxy(Scene):
    """Spiral galaxy seen from a slowly diving camera; density waves on kicks."""
    name = "galaxy"
    hue_range = 0.35
    N = 26000

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        r = np.random.default_rng(self.rng.randrange(10 ** 6))
        n = self.N
        arms = r.integers(0, 3, n)
        rad = r.exponential(0.5, n).clip(0.03, 1.8)
        self.rad = rad
        self.theta = arms * (2 * np.pi / 3) + np.log(rad + 0.05) * 2.6 + r.normal(0, 0.16, n) / (rad + 0.4)
        self.yz = r.normal(0, 0.03, n)
        self.col_t = r.random(n)
        self.base_hue = self.rng.random()
        self.rot = 0.0
        self.waves = []

    def render(self, img, i, t, dt):
        c = self.common(i, t)
        hue = self.update_hue(t, dt)
        img[:] = 0
        H, W = self.H, self.W
        self.rot += dt * (0.12 + 0.35 * c["energy"])
        if c["kick"] > 0.95:
            self.waves.append(t)
        self.waves = [w for w in self.waves if t - w < 2.0]
        th = self.theta + self.rot / np.sqrt(self.rad + 0.15)
        x = self.rad * np.cos(th)
        z = self.rad * np.sin(th)
        y = self.yz
        tilt = 1.05 + 0.3 * math.sin(t * 0.15)
        yr = y * math.cos(tilt) - z * math.sin(tilt)
        zr = y * math.sin(tilt) + z * math.cos(tilt)
        cam = 2.2 - 0.4 * c["energy"]
        persp = 1.0 / (cam - zr)
        sx = W * 0.5 + x * persp * W * 1.7
        sy = H * 0.40 + yr * persp * W * 1.7
        bright = np.exp(-self.rad * 1.2) * 3.0 + 1.4
        for w in self.waves:
            R = (t - w) * 0.9
            bright = bright + 0.9 * np.exp(-((self.rad - R) / 0.05) ** 2) * (1 - (t - w) / 2.0)
        c1 = np.array(_hsv(hue + 0.08, 0.45, 1.0)); c2 = np.array(_hsv(hue + 0.6, 0.7, 1.0))
        mix = np.clip(self.rad * 1.3, 0, 1)[:, None]
        cols = c1 * (1 - mix) + c2 * mix
        cols = cols * (0.8 + 0.4 * self.col_t[:, None])
        K.splat(img, sx, sy, cols, bright * persp * 1.2 * (0.8 + 0.6 * c["energy"] + 0.4 * c["kick"]))
        # core glow
        cy_, cx_ = H * 0.40, W * 0.5
        ys = slice(int(cy_ - 250), int(cy_ + 250)); xs = slice(int(cx_ - 250), int(cx_ + 250))
        yy, xx = np.mgrid[-250:250, -250:250]
        g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (40 + 20 * c["kick"]) ** 2)).astype(np.float32) * (0.35 + 0.4 * c["kick"])
        img[ys, xs, 0] += g * c1[0]; img[ys, xs, 1] += g * c1[1]; img[ys, xs, 2] += g * c1[2]
