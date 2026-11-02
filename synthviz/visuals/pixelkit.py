"""Tiny pixel-art toolkit for the arcade inserts: a float RGB canvas, sprites from
ASCII art, a 3x5 digit font, palettes from one hue and stateless particle bursts.

Everything is a pure function of time so any frame can be rendered on its own
(previews, seeking) and the result is identical to a full render."""
from __future__ import annotations

import colorsys
import math

import numpy as np


def hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, s, v), np.float32)


def make_palette(hue):
    """Named colours for an insert, all derived from the video's hue so the game
    always matches the world behind it."""
    return dict(
        sky0=hsv(hue + 0.92, 0.80, 0.05), sky1=hsv(hue + 0.02, 0.75, 0.30),
        far=hsv(hue + 0.95, 0.60, 0.18), near=hsv(hue + 0.90, 0.65, 0.10),
        ground=hsv(hue + 0.93, 0.60, 0.22), ground_hi=hsv(hue + 0.50, 0.50, 0.70),
        accent=hsv(hue + 0.10, 0.75, 1.0), accent2=hsv(hue + 0.45, 0.70, 1.0),
        hero=hsv(hue + 0.50, 0.25, 1.0), hero2=hsv(hue + 0.98, 0.80, 1.0),
        coin=hsv(0.13, 0.75, 1.0), white=np.array([1.0, 1.0, 1.0], np.float32),
        dark=hsv(hue + 0.9, 0.6, 0.03),
    )


def pitch_color(p, hue, s=0.7, v=1.0):
    """Pitch class -> colour on the circle of fifths, rotated to the video hue."""
    return hsv(hue + ((int(p) * 7) % 12) / 12.0 * 0.6, s, v)


def sprite(rows):
    """ASCII art -> (h, w) int array; '.' is empty, other chars are colour keys."""
    h = len(rows); w = max(len(r) for r in rows)
    out = np.full((h, w), -1, np.int16)
    keys = {}
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch != ".":
                keys.setdefault(ch, len(keys))
                out[y, x] = ord(ch)
    return out


class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.a = np.zeros((h, w, 3), np.float32)
        self.mask = np.zeros((h, w), np.float32)  # what is opaque (used for compositing)

    def clear(self, col=None):
        self.a[:] = 0 if col is None else col
        self.mask[:] = 1.0

    def vgrad(self, c0, c1, y0=0, y1=None):
        y1 = self.h if y1 is None else y1
        k = np.linspace(0, 1, max(1, y1 - y0))[:, None, None]
        self.a[y0:y1] = c0 * (1 - k) + c1 * k

    def rect(self, x0, y0, x1, y1, col, add=False):
        x0, y0 = int(round(x0)), int(round(y0))
        x1, y1 = int(round(x1)), int(round(y1))
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(self.w, x1), min(self.h, y1)
        if xa >= xb or ya >= yb:
            return
        if add:
            self.a[ya:yb, xa:xb] += col
        else:
            self.a[ya:yb, xa:xb] = col

    def px(self, x, y, col, add=False):
        x, y = int(round(x)), int(round(y))
        if 0 <= x < self.w and 0 <= y < self.h:
            if add:
                self.a[y, x] += col
            else:
                self.a[y, x] = col

    def blit(self, spr, x, y, colors, flip=False, sx=1.0, sy=1.0):
        """Draw sprite with its bottom-centre at (x, y); sx/sy squash-stretch."""
        h, w = spr.shape
        if flip:
            spr = spr[:, ::-1]
        hh, ww = max(1, int(round(h * sy))), max(1, int(round(w * sx)))
        ys = np.minimum(h - 1, (np.arange(hh) / sy).astype(int))
        xs = np.minimum(w - 1, (np.arange(ww) / sx).astype(int))
        s = spr[ys][:, xs]
        x0 = int(round(x - ww / 2)); y0 = int(round(y - hh))
        for key, col in colors.items():
            yy, xx = np.nonzero(s == ord(key))
            yy = yy + y0; xx = xx + x0
            m = (yy >= 0) & (yy < self.h) & (xx >= 0) & (xx < self.w)
            self.a[yy[m], xx[m]] = col

    def text(self, s, x, y, col, scale=1):
        """3x5 digits/letters, top-left at (x, y)."""
        cx = x
        for ch in s:
            g = FONT.get(ch.upper())
            if g is None:
                cx += 4 * scale
                continue
            for j, row in enumerate(g):
                for i, b in enumerate(row):
                    if b == "#":
                        self.rect(cx + i * scale, y + j * scale, cx + (i + 1) * scale, y + (j + 1) * scale, col)
            cx += 4 * scale

    def burst(self, x, y, age, seed, col, n=14, speed=60.0, life=0.45, grav=60.0):
        """Stateless particle burst: same (seed, age) -> same picture."""
        if age < 0 or age > life:
            return
        r = np.random.default_rng(seed)
        ang = r.uniform(0, 2 * np.pi, n); sp = r.uniform(0.4, 1.0, n) * speed
        px = x + np.cos(ang) * sp * age
        py = y + np.sin(ang) * sp * age + 0.5 * grav * age * age
        k = 1.0 - age / life
        for a_, b_ in zip(px, py):
            self.px(a_, b_, col * (0.6 + 0.8 * k), add=True)


FONT = {
    "0": ["###", "#.#", "#.#", "#.#", "###"], "1": [".#.", "##.", ".#.", ".#.", "###"],
    "2": ["###", "..#", "###", "#..", "###"], "3": ["###", "..#", ".##", "..#", "###"],
    "4": ["#.#", "#.#", "###", "..#", "..#"], "5": ["###", "#..", "###", "..#", "###"],
    "6": ["###", "#..", "###", "#.#", "###"], "7": ["###", "..#", ".#.", ".#.", ".#."],
    "8": ["###", "#.#", "###", "#.#", "###"], "9": ["###", "#.#", "###", "..#", "###"],
    "X": ["#.#", "#.#", ".#.", "#.#", "#.#"], "L": ["#..", "#..", "#..", "#..", "###"],
    "V": ["#.#", "#.#", "#.#", "#.#", ".#."], "E": ["###", "#..", "##.", "#..", "###"],
    "S": ["###", "#..", "###", "..#", "###"], "C": ["###", "#..", "#..", "#..", "###"],
    "O": ["###", "#.#", "#.#", "#.#", "###"], "R": ["##.", "#.#", "##.", "#.#", "#.#"],
    "P": ["###", "#.#", "###", "#..", "#.."], "U": ["#.#", "#.#", "#.#", "#.#", "###"],
    "W": ["#.#", "#.#", "#.#", "###", "#.#"], "A": [".#.", "#.#", "###", "#.#", "#.#"],
    "Y": ["#.#", "#.#", ".#.", ".#.", ".#."], "B": ["##.", "#.#", "##.", "#.#", "##."],
    "T": ["###", ".#.", ".#.", ".#.", ".#."], "N": ["#.#", "###", "###", "#.#", "#.#"],
    "M": ["#.#", "###", "#.#", "#.#", "#.#"], "I": ["###", ".#.", ".#.", ".#.", "###"],
    "G": ["###", "#..", "#.#", "#.#", "###"], "H": ["#.#", "#.#", "###", "#.#", "#.#"],
    "D": ["##.", "#.#", "#.#", "#.#", "##."], "K": ["#.#", "##.", "#..", "##.", "#.#"],
    "F": ["###", "#..", "##.", "#..", "#.."], "!": [".#.", ".#.", ".#.", "...", ".#."],
    ":": ["...", ".#.", "...", ".#.", "..."], "-": ["...", "...", "###", "...", "..."],
}


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def onsets(notes, t0, t1, min_gap=0.12):
    """Note onsets in [t0, t1) thinned so that no two are closer than min_gap."""
    out = []
    for n in sorted(notes, key=lambda n: n["t"]):
        if t0 <= n["t"] < t1 and (not out or n["t"] - out[-1]["t"] >= min_gap):
            out.append(n)
    return out
