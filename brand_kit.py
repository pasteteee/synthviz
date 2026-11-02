"""RAVE.EXE brand kit: pixel logo, avatar, banner and Shorts covers.

Concept: the channel is a program that runs a rave. The logo is a chunky 3D pixel word, the banner is a
retro OS window "RAVE.EXE" floating over a laser stage, and the mascot Bit (the robot that dances in the
videos) lives inside the window. Palette: hot magenta, electric cyan, acid yellow on deep violet night.

usage:
  python brand_kit.py kit <out_dir>                         # avatar.png, banner.jpg, logo.png, previews
  python brand_kit.py cover <video.mp4> <t_sec> <title> <genre_line> <out.jpg>
"""
from __future__ import annotations

import colorsys
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

FONTS = Path(__file__).resolve().parent / "synthviz" / "fonts"
INTER = str(FONTS / "Inter-SemiBold.otf")

NAME = "RAVE.EXE"
TAGLINE = "SYNTH MUSIC YOU CAN SEE"


def hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1, s, v), np.float32)


MAG = np.array([1.00, 0.17, 0.84], np.float32)
CYA = np.array([0.13, 0.94, 1.00], np.float32)
YEL = np.array([1.00, 0.89, 0.24], np.float32)
VIO = np.array([0.43, 0.16, 1.00], np.float32)
NIGHT0 = np.array([0.045, 0.016, 0.095], np.float32)
NIGHT1 = np.array([0.11, 0.025, 0.20], np.float32)


def glow(a, sig=(3, 12, 40), g=(0.9, 0.6, 0.45)):
    out = a.copy()
    for s, k in zip(sig, g):
        out += gaussian_filter(a, (s, s, 0)) * k
    return out


def tonemap(a):
    return (np.clip(1 - np.exp(-np.maximum(a, 0) * 1.15), 0, 1) ** (1 / 1.05) * 255).astype(np.uint8)


def over(base, layer, alpha):
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    return base * (1 - a) + layer * a


# ---------------------------------------------------------------- 5x7 pixel font
_F = {
    "A": ".###.|#...#|#...#|#####|#...#|#...#|#...#", "B": "####.|#...#|#...#|####.|#...#|#...#|####.",
    "C": ".####|#....|#....|#....|#....|#....|.####", "D": "####.|#...#|#...#|#...#|#...#|#...#|####.",
    "E": "#####|#....|#....|####.|#....|#....|#####", "F": "#####|#....|#....|####.|#....|#....|#....",
    "G": ".####|#....|#....|#.###|#...#|#...#|.###.", "H": "#...#|#...#|#...#|#####|#...#|#...#|#...#",
    "I": "#####|..#..|..#..|..#..|..#..|..#..|#####", "J": "..###|...#.|...#.|...#.|#..#.|#..#.|.##..",
    "K": "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#", "L": "#....|#....|#....|#....|#....|#....|#####",
    "M": "#...#|##.##|#.#.#|#.#.#|#...#|#...#|#...#", "N": "#...#|##..#|#.#.#|#..##|#...#|#...#|#...#",
    "O": ".###.|#...#|#...#|#...#|#...#|#...#|.###.", "P": "####.|#...#|#...#|####.|#....|#....|#....",
    "Q": ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#", "R": "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
    "S": ".####|#....|#....|.###.|....#|....#|####.", "T": "#####|..#..|..#..|..#..|..#..|..#..|..#..",
    "U": "#...#|#...#|#...#|#...#|#...#|#...#|.###.", "V": "#...#|#...#|#...#|#...#|.#.#.|.#.#.|..#..",
    "W": "#...#|#...#|#...#|#.#.#|#.#.#|##.##|#...#", "X": "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#",
    "Y": "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..", "Z": "#####|....#|...#.|..#..|.#...|#....|#####",
    "0": ".###.|#..##|#.#.#|#.#.#|##..#|#...#|.###.", "1": "..#..|.##..|..#..|..#..|..#..|..#..|.###.",
    "2": ".###.|#...#|....#|...#.|..#..|.#...|#####", "3": "####.|....#|....#|.###.|....#|....#|####.",
    "4": "#..#.|#..#.|#..#.|#####|...#.|...#.|...#.", "5": "#####|#....|####.|....#|....#|#...#|.###.",
    "6": ".###.|#....|#....|####.|#...#|#...#|.###.", "7": "#####|....#|...#.|..#..|..#..|..#..|..#..",
    "8": ".###.|#...#|#...#|.###.|#...#|#...#|.###.", "9": ".###.|#...#|#...#|.####|....#|....#|.###.",
    ".": ".|.|.|.|.|.|#", "-": "...|...|...|###|...|...|...", "!": "#|#|#|#|#|.|#",
    "?": ".###.|#...#|....#|...#.|..#..|.....|..#..", "&": ".##..|#..#.|#.#..|.#...|#.#.#|#..#.|.##.#",
    "'": "#|#|.|.|.|.|.", ":": ".|#|.|.|.|#|.", "/": "....#|...#.|...#.|..#..|.#...|.#...|#....",
    "_": "#####|.....|.....|.....|.....|.....|.....",  # unused placeholder
    " ": "...|...|...|...|...|...|...",
}
GLYPH = {k: [r for r in v.split("|")] for k, v in _F.items()}


def pixel_mask(text, gap=1):
    """Return a bool grid (7 x N) for the text in the 5x7 font."""
    cols = []
    for i, ch in enumerate(text.upper()):
        g = GLYPH.get(ch, GLYPH["?"])
        w = len(g[0])
        for x in range(w):
            cols.append([g[y][x] == "#" for y in range(7)])
        if i < len(text) - 1:
            cols.extend([[False] * 7] * gap)
    return np.array(cols, bool).T  # (7, N)


def pixel_word(text, cell, fill_fn, depth=0.55, side=None, gap=1, bevel=True):
    """Render a chunky 3D pixel word. fill_fn(u, v) -> rgb for face (u,v in 0..1 over the word).
    Returns (rgb, alpha) arrays with padding for the extrusion."""
    m = pixel_mask(text, gap)
    gh, gw = m.shape
    ext = max(2, int(cell * depth))
    H, W = gh * cell + ext + 4, gw * cell + ext + 4
    rgb = np.zeros((H, W, 3), np.float32)
    alp = np.zeros((H, W), np.float32)
    side = side if side is not None else VIO * 0.35
    # extrusion: stack shifted copies, darker towards the back
    for k in range(ext, 0, -1):
        shade = side * (0.45 + 0.55 * (1 - k / ext))
        for y, x in zip(*np.nonzero(m)):
            y0, x0 = y * cell + k, x * cell + k
            rgb[y0:y0 + cell, x0:x0 + cell] = shade
            alp[y0:y0 + cell, x0:x0 + cell] = 1
    yy, xx = np.mgrid[0:cell, 0:cell].astype(np.float32) / cell
    for y, x in zip(*np.nonzero(m)):
        y0, x0 = y * cell, x * cell
        u, v = (x + 0.5) / gw, (y + 0.5) / gh
        c = fill_fn(u, v)
        face = np.broadcast_to(c, (cell, cell, 3)).copy()
        if bevel:
            b = max(1, cell // 7)
            face[:b] = np.minimum(c * 1.6 + 0.25, 2.2)       # top highlight
            face[:, :b] = np.minimum(c * 1.3 + 0.12, 2.0)
            face[-b:] *= 0.62
            face[:, -b:] *= 0.75
        # subtle scanlines on the face
        face *= (0.88 + 0.12 * (np.sin(yy * np.pi * 6) > -0.3))[..., None]
        rgb[y0:y0 + cell, x0:x0 + cell] = face
        alp[y0:y0 + cell, x0:x0 + cell] = 1
    return rgb, alp


def paste(canvas, alpha_acc, rgb, alp, x, y):
    H, W = canvas.shape[:2]
    h, w = alp.shape
    xa, ya, xb, yb = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if xa >= xb or ya >= yb:
        return
    sub = rgb[ya - y:yb - y, xa - x:xb - x]
    sa = alp[ya - y:yb - y, xa - x:xb - x]
    canvas[ya:yb, xa:xb] = over(canvas[ya:yb, xa:xb], sub, sa)
    if alpha_acc is not None:
        alpha_acc[ya:yb, xa:xb] = np.maximum(alpha_acc[ya:yb, xa:xb], sa)


def logo_fill(split):
    """RAVE in magenta->yellow, .EXE in cyan."""
    def f(u, v):
        if u < split:
            t = u / split
            return (MAG * (1 - v) * (1 - 0.3 * t) + (MAG * 0.45 + YEL * 0.75) * v) * 1.25
        return (CYA * (1.15 - 0.35 * v) + VIO * 0.25 * v)
    return f


def logo(cell, depth=0.6):
    text = NAME
    split = len(pixel_mask("RAVE", 1)[0]) / len(pixel_mask(text, 1)[0])
    return pixel_word(text, cell, logo_fill(split), depth=depth)


# ---------------------------------------------------------------- Bit
BIT_HEAD = [
    "...........aa...........",
    "...........aa...........",
    "..........a..a..........",
    "...........aa...........",
    "...........bb...........",
    "....tttttttttttttttt....",
    "...tttttttttttttttttt...",
    "..hhbbbbbbbbbbbbbbbbhh..",
    ".hhhbbbbbbbbbbbbbbbbhhh.",
    ".hhhbbddddddddddddbbhhh.",
    ".hhhbdddddddddddddddbhhh",
    ".hhhbddeeeddddeeeddbhhh.",
    ".hhhbddeeeddddeeeddbhhh.",
    ".hhhbdddddddddddddddbhhh",
    ".hhhbbddddddddddddbbhhh.",
    "..hhbbbbbbbbbbbbbbbbhh..",
    "....bbbbbbmmmmbbbbbb....",
    "....bbbbbbbbbbbbbbbb....",
    ".....bbbbbbbbbbbbbb.....",
    ".......cccccccccc.......",
]
BIT_COLS = {"a": YEL * 1.7, "b": np.array([0.88, 0.95, 0.92], np.float32) * 1.1, "t": MAG * 1.3,
            "h": MAG * 1.15, "d": np.array([0.03, 0.01, 0.06], np.float32), "e": CYA * 2.3,
            "m": MAG * 1.3, "c": np.array([0.55, 0.62, 0.60], np.float32)}


def bit_head(cell, eyes="open"):
    rows = list(BIT_HEAD)
    if eyes == "happy":  # ^ ^ eyes
        rows[11] = ".hhhbdddeddddddedddbhhh."
        rows[12] = ".hhhbddedeedddedeedbhhh."
        rows[11] = ".hhhbddeddeddeddeddbhhh."
        rows[12] = ".hhhbdedddeddedddedbhhh."
    gh, gw = len(rows), len(rows[0])
    H, W = gh * cell, gw * cell
    rgb = np.zeros((H, W, 3), np.float32)
    alp = np.zeros((H, W), np.float32)
    g = max(1, cell // 9)
    for j, r in enumerate(rows):
        for i, ch in enumerate(r):
            if ch in BIT_COLS:
                c = BIT_COLS[ch]
                y0, x0 = j * cell, i * cell
                rgb[y0:y0 + cell, x0:x0 + cell] = c * 0.8
                rgb[y0 + g:y0 + cell - g, x0 + g:x0 + cell - g] = c
                rgb[y0 + g:y0 + 2 * g, x0 + g:x0 + cell - g] = c * 1.25 + 0.05
                alp[y0:y0 + cell, x0:x0 + cell] = 1
    return rgb, alp


# ---------------------------------------------------------------- backgrounds
def night(W, H):
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    return (NIGHT1 * (1 - yy) * 0.8 + NIGHT0 * (0.3 + yy)) * np.ones((1, W, 1), np.float32)


def lasers(W, H, origins, n=7, spread=1.1, cols=(MAG, CYA, YEL), width=0.006, seed=0, power=1.0):
    """Laser beams fanning upwards from points on the floor."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.zeros((H, W, 3), np.float32)
    for (ox, oy), base in origins:
        dx, dy = x - ox, oy - y
        ang = np.arctan2(dx, np.maximum(dy, 1e-3))
        dist = np.hypot(dx, dy) / max(W, H)
        for k in range(n):
            a = base + (k - (n - 1) / 2) / max(n - 1, 1) * spread + rng.uniform(-0.04, 0.04)
            c = cols[k % len(cols)]
            d = np.abs(ang - a)
            beam = np.exp(-(d / width) ** 2) * (dy > 0) * np.exp(-dist * 1.6)
            haze = np.exp(-(d / (width * 7)) ** 2) * (dy > 0) * np.exp(-dist * 2.2) * 0.12
            out += c[None, None] * (beam + haze)[..., None] * power
    return out


def floor_grid(W, H, horizon, col=MAG, cx=None, density=1.0, power=1.0):
    cx = W / 2 if cx is None else cx
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    below = y > horizon + 1
    z = 1.0 / np.maximum((y - horizon) / H, 1e-3)
    gx = ((x - cx) / H * z * 0.55 * density) % 1
    gz = (z * 0.9 * density) % 1
    lx = np.exp(-(np.minimum(gx, 1 - gx) / (0.018 * z ** 0.6)) ** 2)
    lz = np.exp(-(np.minimum(gz, 1 - gz) / (0.035 * z ** 0.25)) ** 2)
    fade = np.clip((y - horizon) / (H - horizon), 0, 1) ** 0.8
    g = np.clip(lx + lz, 0, 1.2) * below * fade
    return col[None, None] * g[..., None] * power


def eq_bars(W, H, y_base, n, max_h, seed=1, cols=(MAG, VIO, CYA), gap=0.25, power=1.0):
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W, 3), np.float32)
    bw = W / n
    lv = np.clip(0.25 + 0.75 * np.abs(np.sin(np.arange(n) * 0.37 + 1.3)) * rng.uniform(0.5, 1.0, n), 0, 1)
    seg = max(4, int(bw * 0.5))
    for i in range(n):
        h = int(lv[i] * max_h)
        x0, x1 = int(i * bw + bw * gap / 2), int((i + 1) * bw - bw * gap / 2)
        t = i / max(n - 1, 1)
        c = cols[0] * (1 - t) + cols[-1] * t if len(cols) == 2 else (
            cols[0] * max(0, 1 - 2 * t) + cols[1] * (1 - abs(2 * t - 1)) + cols[2] * max(0, 2 * t - 1))
        for yy in range(y_base - h, y_base, seg):
            k = (y_base - yy) / max_h
            out[max(0, yy):max(0, yy + seg - 3), x0:x1] = c * (0.6 + 0.8 * k) * power
    return out


def stars(W, H, n, seed=3, ymax=1.0):
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W, 3), np.float32)
    for _ in range(n):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H * ymax))
        s = rng.uniform(0.3, 1.4)
        out[y:y + 2, x:x + 2] = s * (0.7 + 0.3 * hsv(rng.uniform(0.5, 0.95), 0.4, 1.0))
    return out


def text_mask(W, H, items):
    """items: (text, px, (x, y), anchor[, tracking])"""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for it in items:
        text, px, xy, anchor = it[:4]
        track = it[4] if len(it) > 4 else 0
        f = ImageFont.truetype(INTER, px)
        if track:
            widths = [d.textlength(ch, font=f) for ch in text]
            total = sum(widths) + track * (len(text) - 1)
            x = xy[0] - total / 2 if anchor[0] == "m" else (xy[0] - total if anchor[0] == "r" else xy[0])
            for ch, w in zip(text, widths):
                d.text((x, xy[1]), ch, font=f, anchor="l" + anchor[1], fill=255)
                x += w + track
        else:
            d.text(xy, text, font=f, anchor=anchor, fill=255)
    return np.asarray(img).astype(np.float32) / 255


def window(canvas, x0, y0, x1, y1, title, bar=56, border=4, accent=CYA, fill_alpha=0.82):
    """A retro OS window: title bar with the program name and three buttons, glowing border."""
    H, W = canvas.shape[:2]
    lay = np.zeros_like(canvas)
    a = np.zeros((H, W), np.float32)
    # body
    a[y0:y1, x0:x1] = fill_alpha
    yy = np.linspace(0, 1, y1 - y0, dtype=np.float32)[:, None, None]
    lay[y0:y1, x0:x1] = NIGHT0 * 0.6 + NIGHT1 * 0.4 * (1 - yy)
    # title bar gradient
    xx = np.linspace(0, 1, x1 - x0, dtype=np.float32)[None, :, None]
    lay[y0:y0 + bar, x0:x1] = (MAG * (1 - xx) + VIO * xx) * 0.95
    a[y0:y0 + bar, x0:x1] = 1
    # stripes in the bar (classic program window)
    canvas[:] = over(canvas, lay, a)
    edge = np.zeros_like(canvas)
    edge[y0:y0 + border, x0:x1] = accent
    edge[y1 - border:y1, x0:x1] = accent
    edge[y0:y1, x0:x0 + border] = accent
    edge[y0:y1, x1 - border:x1] = accent
    edge[y0 + bar:y0 + bar + border, x0:x1] = accent * 0.8
    # buttons
    bs = int(bar * 0.52)
    by = y0 + (bar - bs) // 2
    for k, c in enumerate((YEL, CYA, MAG * 1.2)):
        bx = x1 - (k + 1) * (bs + int(bar * 0.28)) - int(bar * 0.1)
        edge[by:by + bs, bx:bx + bs] = c * 0.25
        edge[by:by + border, bx:bx + bs] = c
        edge[by + bs - border:by + bs, bx:bx + bs] = c
        edge[by:by + bs, bx:bx + border] = c
        edge[by:by + bs, bx + bs - border:bx + bs] = c
        if k == 0:  # close X
            for i in range(bs - 2 * border * 2):
                p = by + 2 * border + i
                q1, q2 = bx + 2 * border + i, bx + bs - 2 * border - i - 1
                edge[p:p + border - 1, q1:q1 + border - 1] = c
                edge[p:p + border - 1, q2:q2 + border - 1] = c
    canvas += glow(edge, (3, 14), (0.8, 0.45))
    canvas[:] = over(canvas, edge * 1.4, (edge.sum(2) > 0).astype(np.float32))
    if not title:
        return
    tm = text_mask(W, H, [(title, int(bar * 0.48), (x0 + int(bar * 0.45), y0 + bar // 2 + 1), "lm", 2)])
    canvas[:] = over(canvas, np.ones_like(canvas) * 1.6, tm)


def progress_bar(canvas, x0, y0, x1, h, frac, label_left, label_right):
    H, W = canvas.shape[:2]
    lay = np.zeros_like(canvas)
    lay[y0:y0 + h, x0:x1] = NIGHT1 * 1.5
    n = 34
    cw = (x1 - x0 - 8) / n
    for i in range(int(n * frac)):
        t = i / n
        c = MAG * (1 - t) + CYA * t
        lay[y0 + 4:y0 + h - 4, int(x0 + 4 + i * cw):int(x0 + 4 + (i + 1) * cw) - 4] = c * 1.3
    edge = np.zeros_like(canvas)
    b = 3
    edge[y0:y0 + b, x0:x1] = CYA; edge[y0 + h - b:y0 + h, x0:x1] = CYA
    edge[y0:y0 + h, x0:x0 + b] = CYA; edge[y0:y0 + h, x1 - b:x1] = CYA
    canvas[:] = canvas + glow(lay * 0.4, (4, 14), (0.6, 0.3))
    a = (lay.sum(2) > 0).astype(np.float32)
    canvas[:] = over(canvas, lay, a)
    canvas[:] = canvas + edge * 1.2 + glow(edge, (3, 10), (0.4, 0.2))
    px = int(h * 0.62)
    tm = text_mask(W, H, [(label_left, px, (x0, y0 + h + int(px * 1.3)), "lm", 3),
                          (label_right, px, (x1, y0 + h + int(px * 1.3)), "rm", 3)])
    canvas[:] = over(canvas, np.ones_like(canvas) * np.array([0.85, 0.9, 1.1], np.float32), tm * 0.9)


def scanlines(img, period=4, depth=0.10):
    H = img.shape[0]
    s = 1 - depth * (np.arange(H) % period == 0)
    return img * s[:, None, None].astype(np.float32)


def chroma(img, shift=3):
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], shift, axis=1)
    out[..., 2] = np.roll(img[..., 2], -shift, axis=1)
    return out


# ---------------------------------------------------------------- avatar
def avatar(path, S=1600):
    img = night(S, S) * 1.4
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    cx, cy = S / 2, S / 2
    r = np.hypot(x - cx, y - cy) / S
    ang = np.arctan2(y - cy, x - cx)
    img += VIO * 0.35 * np.exp(-r * 3.2)[..., None]
    # light rays
    rays = (0.5 + 0.5 * np.cos(ang * 10)) ** 8 * np.exp(-r * 2.0)
    img += (MAG * 0.25)[None, None] * rays[..., None]
    # floor grid + lasers at the bottom
    img += floor_grid(S, S, S * 0.70, MAG, density=0.9, power=0.55)
    img += lasers(S, S, [((S * 0.5, S * 0.74), 0.0)], n=9, spread=2.4, width=0.010, seed=4, power=0.55)
    # EQ ring (round, follows the YouTube crop)
    nb = 56
    rng = np.random.default_rng(11)
    lv = 0.45 + 0.55 * np.abs(np.sin(np.arange(nb) * 0.61)) * rng.uniform(0.6, 1.0, nb)
    k = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    frac = ((ang + np.pi) / (2 * np.pi) * nb) % 1
    r0 = 0.355
    ring = ((r > r0) & (r < r0 + 0.075 * lv[k]) & (frac > 0.2) & (frac < 0.8)).astype(np.float32)
    t = np.clip((ang + np.pi) / (2 * np.pi), 0, 1)[..., None]
    rc = MAG * (1 - np.abs(2 * t - 1)) + CYA * np.abs(2 * t - 1)
    img += glow(ring[..., None] * rc * 1.3, (5, 20, 50), (1.0, 0.6, 0.35))
    # Bit, big and happy
    cell = int(S * 0.0305)
    hr, ha = bit_head(cell)
    hh, hw = ha.shape
    hx, hy = int(cx - hw / 2), int(cy - hh / 2 + S * 0.005)
    halo = gaussian_filter(np.pad(ha, 0), cell * 1.4)
    full_halo = np.zeros((S, S), np.float32); full_halo[hy:hy + hh, hx:hx + hw] = halo
    img *= (1 - 0.85 * np.clip(full_halo * 2.5, 0, 1))[..., None]
    lay = np.zeros_like(img); al = np.zeros((S, S), np.float32)
    paste(lay, al, hr, ha, hx, hy)
    img = over(img, lay, al) + glow(lay * 0.22, (8, 30), (0.6, 0.4))
    img *= np.clip(1.35 - r * 1.6, 0, 1)[..., None] ** 0.6
    img = scanlines(img, 6, 0.06)
    Image.fromarray(tonemap(img)).resize((800, 800), Image.LANCZOS).save(path)


# ---------------------------------------------------------------- banner
def banner(path):
    W, H = 2560, 1440
    img = night(W, H)
    img += stars(W, H, 500, ymax=0.6) * 0.6
    hz = int(H * 0.66)
    # big glow on the horizon
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img += (MAG * 0.55)[None, None] * np.exp(-((y - hz) / (H * 0.10)) ** 2)[..., None] * np.exp(-((x - W / 2) / (W * 0.45)) ** 2)[..., None]
    img += floor_grid(W, H, hz, MAG, density=0.8, power=0.9)
    img += eq_bars(W, H, hz, 64, int(H * 0.16), seed=7, power=0.55) * (np.abs(np.linspace(-1, 1, W)) > 0.0)[None, :, None]
    img += lasers(W, H, [((W * 0.12, hz), 0.18), ((W * 0.88, hz), -0.18), ((W * 0.5, hz), 0.0)],
                  n=7, spread=1.25, width=0.0045, seed=2, power=0.85)
    img = glow(img * 0.85, (6, 24), (0.25, 0.2))
    # the program window in the safe area (1546 x 423 centred)
    sx0, sy0 = (W - 1546) // 2, (H - 423) // 2
    x0, x1 = sx0 - 40, sx0 + 1546 + 40
    y0, y1 = sy0 - 70, sy0 + 423 + 40
    # drop shadow
    sh = np.zeros((H, W), np.float32); sh[y0 + 24:y1 + 24, x0 + 24:x1 + 24] = 1
    img *= (1 - 0.7 * gaussian_filter(sh, 30))[..., None]
    window(img, x0, y0, x1, y1, "C:\\RAVE.EXE", bar=62)
    # Bit inside the window on the left
    cell = 15
    hr, ha = bit_head(cell)
    hh, hw = ha.shape
    bx, by = sx0 + 30, y0 + 62 + (y1 - y0 - 62 - hh) // 2 - 6
    lay = np.zeros_like(img); al = np.zeros((H, W), np.float32)
    paste(lay, al, hr, ha, bx, by)
    img = over(img, lay, al) + glow(lay * 0.18, (8, 26), (0.6, 0.35))
    # logo
    lr, la = logo(23, depth=0.55)
    lh, lw = la.shape
    lx = bx + hw + 60
    ly = y0 + 62 + 52
    lay = np.zeros_like(img); al = np.zeros((H, W), np.float32)
    paste(lay, al, lr, la, lx, ly)
    img = img + glow(lay * 0.35, (8, 30, 70), (0.7, 0.45, 0.25)) * (1 - al)[..., None]
    img = over(img, lay, al)
    # tagline + "now playing" bar
    right = x1 - 60
    tm = text_mask(W, H, [(TAGLINE, 40, (lx + 4, ly + lh + 52), "lm", 6)])
    img = over(img, np.ones_like(img) * np.array([1.0, 0.95, 1.15], np.float32) * 1.4, tm)
    progress_bar(img, lx + 4, ly + lh + 98, right, 36, 0.72,
                  "LOADING DROP...", "HARDSTYLE · DNB · SYNTHWAVE · TRANCE")
    img = scanlines(img, 4, 0.07)
    img = chroma(img, 2)
    Image.fromarray(tonemap(img)).save(path, quality=94)


# ---------------------------------------------------------------- covers (9:16)
def grab(video, t):
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], check=True, capture_output=True).stdout
    from io import BytesIO
    return Image.open(BytesIO(out)).convert("RGB")


def fit_word(text, max_w, max_cell, gap=1):
    gw = pixel_mask(text, gap).shape[1]
    return max(6, min(max_cell, int(max_w / (gw + 0.6))))


def cover(video, t, title, genre_line, out, crop_top=0.0, crop_h=1.0):
    W, H = 1080, 1920
    fr = grab(video, t).resize((W, H), Image.LANCZOS)
    a = np.asarray(fr).astype(np.float32) / 255
    lin = -np.log(np.clip(1 - a * 0.985, 1e-3, 1))   # roughly back to linear for the glow
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    # darken top band (header) and lower part (title area) for legibility
    lin *= 1 - 0.80 * np.clip((0.15 - yy) / 0.09, 0, 1)
    lin *= 1 - 0.97 * np.clip((yy - 0.575) / 0.035, 0, 1)
    lin += floor_grid(W, H, int(H * 0.80), MAG, density=0.9, power=0.45)
    lin += lasers(W, H, [((W * 0.5, H * 0.80), 0.0)], n=7, spread=2.6, width=0.006, seed=9, power=0.25) * (yy > 0.60)
    # header: RAVE.EXE logo, small
    lr, la = logo(14, depth=0.5)
    lh, lw = la.shape
    lay = np.zeros_like(lin); al = np.zeros((H, W), np.float32)
    paste(lay, al, lr, la, (W - lw) // 2, 70)
    lin = lin + glow(lay * 0.3, (6, 20), (0.6, 0.3)) * (1 - al)[..., None]
    lin = over(lin, lay, al)
    # title as a pixel word, one or two lines
    words = title.upper().split()
    lines = [title.upper()]
    if len(words) > 1 and pixel_mask(title, 1).shape[1] > 34:
        best = None
        for i in range(1, len(words)):
            l1, l2 = " ".join(words[:i]), " ".join(words[i:])
            m = max(pixel_mask(l1).shape[1], pixel_mask(l2).shape[1])
            if best is None or m < best[0]:
                best = (m, [l1, l2])
        lines = best[1]
    cell = min(fit_word(l, W - 110, 34) for l in lines)
    ty = int(H * 0.625)
    fill = lambda u, v: (YEL * (1 - v) + MAG * v) * 1.3
    for j, l in enumerate(lines):
        r_, a_ = pixel_word(l, cell, fill, depth=0.6, side=MAG * 0.3)
        h_, w_ = a_.shape
        lay = np.zeros_like(lin); al = np.zeros((H, W), np.float32)
        paste(lay, al, r_, a_, (W - w_) // 2, ty)
        lin = lin + glow(lay * 0.35, (8, 28, 70), (0.7, 0.45, 0.25)) * (1 - al)[..., None]
        lin = over(lin, lay, al)
        ty += h_ + int(cell * 1.4)
    # genre chip (window-styled)
    gy = ty + 40
    gm = text_mask(W, H, [(genre_line, 46, (W // 2, gy + 44), "mm", 5)])
    cols = np.nonzero(gm.max(0) > 0.3)[0]
    gx0, gx1 = cols.min() - 40, cols.max() + 40
    window(lin, gx0, gy, gx1, gy + 88, "", bar=0, border=4, accent=CYA, fill_alpha=0.75)
    lin = over(lin, np.ones_like(lin) * np.array([0.9, 1.25, 1.4], np.float32) * 1.3, gm)
    lin = scanlines(lin, 4, 0.06)
    Image.fromarray(tonemap(lin)).save(out, quality=94)


def previews(kit):
    """Avatar in a circle at real sizes and the banner with the device crops marked."""
    av = Image.open(kit / "avatar.png").convert("RGB")
    sheet = Image.new("RGB", (1180, 440), (24, 24, 28))
    x = 30
    for s in (360, 176, 88, 48):
        im = av.resize((s, s), Image.LANCZOS)
        m = Image.new("L", (s, s), 0); ImageDraw.Draw(m).ellipse((0, 0, s - 1, s - 1), fill=255)
        sheet.paste(im, (x, (440 - s) // 2), m)
        x += s + 60
    sheet.save(kit / "preview_avatar_sizes.jpg", quality=92)
    b = Image.open(kit / "banner.jpg").convert("RGB")
    d = ImageDraw.Draw(b)
    W, H = b.size
    for (w, h, c) in ((2560, 423, (255, 230, 60)), (1546, 423, (40, 240, 255))):
        d.rectangle(((W - w) // 2, (H - h) // 2, (W + w) // 2, (H + h) // 2), outline=c, width=5)
    b.resize((1280, 720), Image.LANCZOS).save(kit / "preview_banner_safe_zones.jpg", quality=90)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "kit":
        kit = Path(sys.argv[2]); kit.mkdir(parents=True, exist_ok=True)
        avatar(kit / "avatar.png")
        banner(kit / "banner.jpg")
        lr, la = logo(40, depth=0.6)
        rgba = np.dstack([tonemap(lr * 1.2), (la * 255).astype(np.uint8)])
        Image.fromarray(rgba, "RGBA").save(kit / "logo.png")
        previews(kit)
    elif cmd == "cover":
        cover(sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
