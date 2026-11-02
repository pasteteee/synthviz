"""More numba scenes: voxel-space terrain & neon-city flyovers, pixel-art world,
plus HUD primitives (glow rectangles)."""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

from .kernels import hash2, hash3, hsv, smooth, fract

FM = dict(parallel=True, fastmath=True, cache=True)


@njit(inline="always", fastmath=True, cache=True)
def vnoise2(x, y):
    ix = math.floor(x); iy = math.floor(y)
    fx = x - ix; fy = y - iy
    ux = fx * fx * (3 - 2 * fx); uy = fy * fy * (3 - 2 * fy)
    a = hash2(ix, iy); b = hash2(ix + 1, iy); c = hash2(ix, iy + 1); d = hash2(ix + 1, iy + 1)
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy


@njit(inline="always", fastmath=True, cache=True)
def ridged(x, y):
    s = 0.0; amp = 0.55; f = 1.0
    for _ in range(4):
        n = 1.0 - abs(vnoise2(x * f, y * f) * 2 - 1)
        s += n * n * amp
        f *= 2.1; amp *= 0.5
    return s


@njit(inline="always", fastmath=True, cache=True)
def path_x(z):
    return math.sin(z * 0.045) * 6.0 + math.sin(z * 0.017 + 1.3) * 9.0


# ------------------------------------------------------------------ sky helper
@njit(fastmath=True, cache=True)
def _sky(y, x, H, W, hor, hue, t, sun_x, sun_y, sun_r, kick):
    v = min(1.0, max(0.0, y / hor))
    tr, tg, tb = hsv(0.70 + hue * 0.15, 0.85, 0.08)
    hr, hg, hb = hsv(0.92 + hue * 0.15, 0.75, 0.75)
    w = v ** 1.8
    r = tr + (hr - tr) * w; g = tg + (hg - tg) * w; b = tb + (hb - tb) * w
    hh = hash2(float(x // 4), float(y // 4))
    if hh > 0.994 and v < 0.7:
        s = (0.5 + 0.5 * math.sin(t * 3.0 + hh * 500.0)) * 1.2 * (1 - v)
        r += s; g += s; b += s
    dx = x - sun_x; dy = y - sun_y
    d = math.sqrt(dx * dx + dy * dy) / sun_r
    sr, sg, sb = hsv(0.10 - 0.12 * max(0.0, dy / sun_r) + hue * 0.05, 0.85, 1.05)
    if d < 1.0:
        rel = dy / sun_r
        cut = 1.0
        if rel > 0.0:
            cut = smooth(0.0, 0.05, fract(rel * 5.0 - t * 0.4) - (0.1 + 0.5 * rel) * 0.5)
        a = smooth(1.0, 0.97, d) * cut
        r = r * (1 - a) + sr * a; g = g * (1 - a) + sg * a; b = b * (1 - a) + sb * a
    else:
        gl = math.exp(-(d - 1.0) * 4.0) * (0.25 + 0.3 * kick)
        r += sr * gl; g += sg * gl * 0.6; b += sb * gl * 0.8
    return r, g, b


# ------------------------------------------------------------------ voxel terrain flyover
@njit(**FM)
def voxel_terrain(img, cam_x, cam_z, cam_h, yaw, hor, hue, t, kick, energy, spec, scale, zmax):
    H, W = img.shape[0], img.shape[1]
    nb = spec.shape[0]
    fov = 1.1
    sun_x = W * 0.5 + yaw * -400.0
    for x in prange(W):
        a = yaw + (x / W - 0.5) * fov
        sa = math.sin(a); ca = math.cos(a)
        ybuf = H
        for y in range(H):
            if y < hor + 4:
                r, g, b = _sky(y, x, H, W, hor, hue, t, sun_x, hor - W * 0.20, W * 0.30, kick)
            else:
                r, g, b = 0.0, 0.0, 0.0
            img[y, x, 0] = r; img[y, x, 1] = g; img[y, x, 2] = b
        z = 0.6
        dz = 0.05
        lr, lg, lb = hsv(0.83 + hue * 0.2, 0.9, 1.0)
        fr_, fg_, fb_ = hsv(0.90 + hue * 0.15, 0.7, 0.6)
        while z < zmax:
            px = cam_x + sa * z
            pz = cam_z + ca * z
            off = abs(px - path_x(pz))
            canyon = smooth(1.5, 7.0, off)
            band = int(min(nb - 1, off * 2.0 + 2))
            hgt = ridged(px * 0.09, pz * 0.09) * 9.0 * canyon * (0.7 + 0.9 * spec[band] * energy)
            sy = hor + (cam_h - hgt) / z * scale
            if sy < ybuf:
                gx = abs(fract(px * 0.5) - 0.5)
                gz = abs(fract(pz * 0.5) - 0.5)
                fw = z * 0.012 + 0.01
                line = math.exp(-((0.5 - gx) / fw) ** 2) + math.exp(-((0.5 - gz) / fw) ** 2)
                fog = math.exp(-z / (zmax * 0.35))
                shade = 0.04 + 0.10 * hgt / 9.0
                ir = (shade * 0.6 + lr * line * (0.7 + 1.2 * kick)) * fog
                ig = (shade * 0.1 + lg * line * (0.7 + 1.2 * kick)) * fog
                ib = (shade * 1.0 + lb * line * (0.7 + 1.2 * kick)) * fog
                ir += fr_ * (1 - fog) * 0.35; ig += fg_ * (1 - fog) * 0.25; ib += fb_ * (1 - fog) * 0.35
                y0 = int(max(sy, 0.0))
                for y in range(y0, int(ybuf)):
                    img[y, x, 0] = ir; img[y, x, 1] = ig; img[y, x, 2] = ib
                # ridge edge glow
                if y0 < H:
                    e = 0.6 * fog * (0.5 + kick)
                    img[y0, x, 0] += lr * e; img[y0, x, 1] += lg * e; img[y0, x, 2] += lb * e
                ybuf = sy
                if ybuf <= 0:
                    break
            z += dz
            dz *= 1.012


# ------------------------------------------------------------------ voxel neon city
@njit(**FM)
def voxel_city(img, cam_x, cam_z, cam_h, yaw, hor, hue, t, kick, energy, spec, scale, zmax):
    H, W = img.shape[0], img.shape[1]
    nb = spec.shape[0]
    fov = 1.15
    for x in prange(W):
        a = yaw + (x / W - 0.5) * fov
        sa = math.sin(a); ca = math.cos(a)
        for y in range(H):
            v = y / hor
            if y < hor:
                r, g, b = hsv(0.72 + hue * 0.2 + v * 0.05, 0.8, 0.04 + 0.18 * v ** 3)
                hh = hash2(float(x // 3), float(y // 3))
                if hh > 0.996:
                    r += 0.8; g += 0.8; b += 0.8
            else:
                r, g, b = 0.0, 0.0, 0.0
            img[y, x, 0] = r; img[y, x, 1] = g; img[y, x, 2] = b
        ybuf = float(H)
        z = 0.3
        dz = 0.03
        while z < zmax:
            px = cam_x + sa * z
            pz = cam_z + ca * z
            ci = math.floor(px / 2.0); cj = math.floor(pz / 2.0)
            street = abs(px) < 3.0
            hgt = 0.0
            if not street:
                hb_ = hash2(ci * 1.0, cj * 1.0)
                band = int(hash2(ci + 7.0, cj) * (nb - 6)) + 2
                hgt = (1.5 + hb_ * hb_ * 14.0) * (0.75 + 0.6 * spec[band] * energy)
                # gaps between blocks
                lx = fract(px / 2.0); lz = fract(pz / 2.0)
                if lx < 0.12 or lz < 0.12:
                    hgt = 0.0
            sy = hor + (cam_h - hgt) / z * scale
            if sy < ybuf:
                fog = math.exp(-z / (zmax * 0.4))
                y0 = int(max(sy, 0.0))
                bh, bs, bv = hash2(ci + 3.0, cj + 1.0), 0.8, 1.0
                wr, wg, wb = hsv(hue + 0.08 + bh * 0.25, 0.6, 1.0)
                er, eg, eb = hsv(hue + 0.5 + bh * 0.1, 0.85, 1.0)
                for y in range(y0, int(ybuf)):
                    if hgt > 0.0:
                        wy = cam_h - (y - hor) * z / scale     # world height of this pixel
                        fl = fract(wy * 1.3)
                        col = fract((px + pz) * 2.0)
                        lit = hash3(ci, cj, math.floor(wy * 1.3)) > 0.45 and fl > 0.35 and col > 0.3
                        base = 0.015
                        if lit:
                            fl2 = 0.55 + 0.45 * math.sin(t * 2.0 + hash2(ci, cj) * 9.0)
                            img[y, x, 0] = (base + wr * 0.9 * fl2) * fog
                            img[y, x, 1] = (base + wg * 0.9 * fl2) * fog
                            img[y, x, 2] = (base + wb * 0.9 * fl2) * fog
                        else:
                            img[y, x, 0] = base * fog; img[y, x, 1] = base * fog; img[y, x, 2] = (base + 0.02) * fog
                    else:
                        # street with neon lane lines
                        lane = math.exp(-((abs(px) - 2.6) / (0.05 + z * 0.004)) ** 2)
                        dash = math.exp(-(px / (0.04 + z * 0.003)) ** 2) * (fract(pz * 0.25) < 0.5)
                        refl = 0.05 * (1 + kick)
                        img[y, x, 0] = (er * (lane + dash) * (0.6 + kick) + refl * 0.5) * fog
                        img[y, x, 1] = (eg * (lane + dash) * (0.6 + kick) + refl * 0.2) * fog
                        img[y, x, 2] = (eb * (lane + dash) * (0.6 + kick) + refl) * fog
                    # fog tint
                    img[y, x, 0] += (1 - fog) * 0.10; img[y, x, 1] += (1 - fog) * 0.02; img[y, x, 2] += (1 - fog) * 0.16
                if hgt > 0.0 and y0 < H:
                    e = (0.9 + 1.5 * kick) * fog
                    img[y0, x, 0] += er * e; img[y0, x, 1] += eg * e; img[y0, x, 2] += eb * e
                ybuf = sy
                if ybuf <= 0:
                    break
            z += dz
            dz *= 1.01


# ------------------------------------------------------------------ pixel-art world (low-res)
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float64) / 16.0


@njit(**FM)
def pixel_world(img, t, scroll, hue, kick, energy, spec, peaks, bayer, levels):
    H, W = img.shape[0], img.shape[1]
    nb = spec.shape[0]
    hor = int(H * 0.55)
    for y in prange(H):
        for x in range(W):
            d = bayer[y & 3, x & 3] - 0.5
            if y < hor:
                v = y / hor
                r, g, b = hsv(0.72 + hue * 0.2 + v * 0.12, 0.8, 0.12 + 0.7 * v ** 2)
                # pixel sun
                dx = x - W * 0.5; dy = y - hor * 0.62
                dd = math.sqrt(dx * dx + dy * dy)
                R = W * 0.26 * (1 + 0.05 * kick)
                if dd < R and not (dy > 0 and (int(dy + scroll * 2) % 6) < 1 + int(dy / R * 4)):
                    r, g, b = hsv(0.12 - 0.1 * (dy / R + 1) * 0.5 + hue * 0.05, 0.9, 1.0)
                # stars
                if hash2(float(x), float(y)) > 0.985 and v < 0.6:
                    tw = 0.5 + 0.5 * math.sin(t * 4.0 + x * 1.7 + y)
                    r += tw; g += tw; b += tw
                # parallax mountains (3 layers)
                for L in range(3):
                    sp = scroll * (0.3 + 0.5 * L)
                    hm = (math.sin((x + sp) * 0.05 + L * 2.0) * 0.5 + 0.5) * (0.5 + 0.5 * math.sin((x + sp) * 0.013 + L))
                    hm = hor - (8 + L * 6) - hm * (18 - L * 4)
                    if y > hm:
                        r, g, b = hsv(0.75 + hue * 0.2 - L * 0.04, 0.7, 0.18 + 0.12 * L)
            else:
                # checker ground scrolling toward the viewer
                dy = y - hor + 1
                z = 40.0 / dy
                wx = (x - W * 0.5) * z * 0.05
                wz = z + scroll * 0.25
                chk = (int(math.floor(wx)) + int(math.floor(wz))) & 1
                r, g, b = hsv(0.82 + hue * 0.2 + chk * 0.08, 0.8, (0.25 + 0.35 * chk) * min(1.0, dy / 12.0) * (1 + kick * 0.6))
            # equalizer skyline standing on the horizon
            bw = W // 16
            bi = x // bw
            if bi < 16 and (x % bw) < bw - 1:
                si = int(bi * (nb - 8) / 16) + 2
                bh = int(spec[si] * H * 0.17 * (0.5 + energy * 0.7))
                base = hor
                if base - bh <= y < base and ((base - y) % 3) != 0:
                    r, g, b = hsv(hue + 0.3 - (base - y) / (H * 0.17) * 0.35, 0.85, 1.0)
                pk = int(peaks[bi] * H * 0.17 * (0.5 + energy * 0.7))
                if y == base - pk - 2:
                    r, g, b = 1.0, 1.0, 1.0
            # quantise with ordered dithering
            r = math.floor(min(1.0, r) * levels + d) / levels
            g = math.floor(min(1.0, g) * levels + d) / levels
            b = math.floor(min(1.0, b) * levels + d) / levels
            img[y, x, 0] = max(r, 0.0); img[y, x, 1] = max(g, 0.0); img[y, x, 2] = max(b, 0.0)


@njit(**FM)
def upscale_nearest(src, dst, k):
    H, W = dst.shape[0], dst.shape[1]
    for y in prange(H):
        sy = min(src.shape[0] - 1, y // k)
        for x in range(W):
            sx = min(src.shape[1] - 1, x // k)
            for c in range(3):
                dst[y, x, c] = src[sy, sx, c]


# ------------------------------------------------------------------ HUD primitives
@njit(fastmath=True, cache=True)
def glow_rect(img, x0, y0, x1, y1, r, g, b, fill, edge, rad):
    """Rounded rect: interior fill + glowing outline (additive)."""
    H, W = img.shape[0], img.shape[1]
    pad = 6
    for y in range(max(0, int(y0) - pad), min(H, int(y1) + pad)):
        for x in range(max(0, int(x0) - pad), min(W, int(x1) + pad)):
            cx = min(max(x, x0 + rad), x1 - rad)
            cy = min(max(y, y0 + rad), y1 - rad)
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - rad
            inside = 1.0 if d < 0 else 0.0
            e = math.exp(-abs(d) * 0.9) * edge + math.exp(-max(d, 0.0) * 0.25) * edge * 0.15
            w = inside * fill + e
            img[y, x, 0] += r * w; img[y, x, 1] += g * w; img[y, x, 2] += b * w


@njit(fastmath=True, cache=True)
def darken_band(img, y0, y1, amt):
    H, W = img.shape[0], img.shape[1]
    for y in range(max(0, int(y0)), min(H, int(y1))):
        fy = min(1.0, min(y - y0, y1 - y) / 40.0)
        k = 1.0 - amt * fy
        for x in range(W):
            img[y, x, 0] *= k; img[y, x, 1] *= k; img[y, x, 2] *= k


@njit(fastmath=True, cache=True)
def darken_rrect(img, x0, y0, x1, y1, rad, amt):
    """Multiply a rounded rectangle by (1-amt) with a soft 3px edge (glass card)."""
    H, W = img.shape[0], img.shape[1]
    for y in range(max(0, int(y0) - 2), min(H, int(y1) + 2)):
        for x in range(max(0, int(x0) - 2), min(W, int(x1) + 2)):
            cx = min(max(x, x0 + rad), x1 - rad)
            cy = min(max(y, y0 + rad), y1 - rad)
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - rad
            m = min(1.0, max(0.0, 0.5 - d / 3.0))
            k = 1.0 - amt * m
            img[y, x, 0] *= k; img[y, x, 1] *= k; img[y, x, 2] *= k


@njit(fastmath=True, cache=True)
def glow_dot(img, x, y, rad, r, g, b, amp):
    H, W = img.shape[0], img.shape[1]
    R = int(rad * 3) + 2
    for yy in range(max(0, int(y) - R), min(H, int(y) + R + 1)):
        for xx in range(max(0, int(x) - R), min(W, int(x) + R + 1)):
            d = math.sqrt((xx - x) ** 2 + (yy - y) ** 2)
            w = (1.0 if d < rad else math.exp(-(d - rad) * 0.6)) * amp
            img[yy, xx, 0] += r * w; img[yy, xx, 1] += g * w; img[yy, xx, 2] += b * w
