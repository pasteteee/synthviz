"""Numba 'shaders': each scene writes a float32 HxWx3 linear-light image."""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

FM = dict(parallel=True, fastmath=True, cache=True)


# ------------------------------------------------------------------ helpers
@njit(inline="always", fastmath=True, cache=True)
def fract(x):
    return x - math.floor(x)


@njit(inline="always", fastmath=True, cache=True)
def smooth(e0, e1, x):
    t = (x - e0) / (e1 - e0)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


@njit(inline="always", fastmath=True, cache=True)
def hash2(x, y):
    h = math.sin(x * 127.1 + y * 311.7) * 43758.5453
    return h - math.floor(h)


@njit(inline="always", fastmath=True, cache=True)
def hash3(x, y, z):
    h = math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
    return h - math.floor(h)


@njit(inline="always", fastmath=True, cache=True)
def hsv(h, s, v):
    h = h - math.floor(h)
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


@njit(inline="always", fastmath=True, cache=True)
def vnoise3(x, y, z):
    ix = math.floor(x); iy = math.floor(y); iz = math.floor(z)
    fx = x - ix; fy = y - iy; fz = z - iz
    ux = fx * fx * (3 - 2 * fx); uy = fy * fy * (3 - 2 * fy); uz = fz * fz * (3 - 2 * fz)
    a = hash3(ix, iy, iz); b = hash3(ix + 1, iy, iz)
    c = hash3(ix, iy + 1, iz); d = hash3(ix + 1, iy + 1, iz)
    e = hash3(ix, iy, iz + 1); f = hash3(ix + 1, iy, iz + 1)
    g = hash3(ix, iy + 1, iz + 1); h = hash3(ix + 1, iy + 1, iz + 1)
    k0 = a + (b - a) * ux; k1 = c + (d - c) * ux
    k2 = e + (f - e) * ux; k3 = g + (h - g) * ux
    l0 = k0 + (k1 - k0) * uy; l1 = k2 + (k3 - k2) * uy
    return l0 + (l1 - l0) * uz


@njit(inline="always", fastmath=True, cache=True)
def fbm(x, y, z, octaves):
    s = 0.0; a = 0.5
    for _ in range(octaves):
        s += a * vnoise3(x, y, z)
        x = x * 2.03 + 1.7; y = y * 2.03 - 3.1; z = z * 1.7
        a *= 0.5
    return s


# ------------------------------------------------------------------ point splatting
@njit(fastmath=True, cache=True)
def splat(img, xs, ys, cols, amp):
    """Additive bilinear splat of points. cols: (n,3)."""
    H, W = img.shape[0], img.shape[1]
    for i in range(xs.shape[0]):
        x = xs[i]; y = ys[i]
        if x < 0 or y < 0 or x >= W - 1 or y >= H - 1:
            continue
        ix = int(x); iy = int(y)
        fx = x - ix; fy = y - iy
        w00 = (1 - fx) * (1 - fy) * amp[i]; w10 = fx * (1 - fy) * amp[i]
        w01 = (1 - fx) * fy * amp[i]; w11 = fx * fy * amp[i]
        for c in range(3):
            v = cols[i, c]
            img[iy, ix, c] += v * w00
            img[iy, ix + 1, c] += v * w10
            img[iy + 1, ix, c] += v * w01
            img[iy + 1, ix + 1, c] += v * w11


@njit(fastmath=True, cache=True)
def polyline(img, xs, ys, r, g, b, amp, thickness):
    """Draw a polyline by dense splatting (spacing 0.5px), thickness in px."""
    H, W = img.shape[0], img.shape[1]
    for i in range(xs.shape[0] - 1):
        x0 = xs[i]; y0 = ys[i]; x1 = xs[i + 1]; y1 = ys[i + 1]
        L = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        n = int(L * 2) + 1
        for k in range(n):
            t = k / n
            x = x0 + (x1 - x0) * t; y = y0 + (y1 - y0) * t
            rad = int(thickness) + 1
            for dy in range(-rad, rad + 1):
                yy = int(y) + dy
                if yy < 0 or yy >= H:
                    continue
                for dx in range(-rad, rad + 1):
                    xx = int(x) + dx
                    if xx < 0 or xx >= W:
                        continue
                    d2 = (xx - x) ** 2 + (yy - y) ** 2
                    w = math.exp(-d2 / (thickness * thickness * 0.5 + 0.2)) * amp / n * (L + 0.5) * 0.5
                    img[yy, xx, 0] += r * w
                    img[yy, xx, 1] += g * w
                    img[yy, xx, 2] += b * w


# ------------------------------------------------------------------ 1. OUTRUN
@njit(**FM)
def outrun(img, t, beat, kick, hue, energy, travel, spec, snare):
    H, W = img.shape[0], img.shape[1]
    hor = H * 0.56
    cx = W * 0.5
    R = W * 0.33 * (1.0 + 0.035 * kick)
    sy = hor - R * 0.62
    nb = spec.shape[0]
    tr, tg, tb = hsv(0.72 + hue * 0.12, 0.85, 0.10)
    hr, hg, hb = hsv(0.93 + hue * 0.12, 0.75, 0.95)
    lr, lg, lb = hsv(0.83 + hue * 0.15, 0.9, 1.0)       # grid line colour
    for y in prange(H):
        for x in range(W):
            r = 0.0; g = 0.0; b = 0.0
            if y < hor:
                v = (y / hor) ** 1.6
                r = tr + (hr - tr) * v; g = tg + (hg - tg) * v; b = tb + (hb - tb) * v
                r *= 0.55; g *= 0.55; b *= 0.55
                # stars
                gx = x // 5; gy = y // 5
                hh = hash2(gx * 1.0, gy * 1.0)
                if hh > 0.993 and y < hor * 0.75:
                    tw = 0.5 + 0.5 * math.sin(t * 3.0 + hh * 900.0)
                    ddx = x - gx * 5 - 2.5; ddy = y - gy * 5 - 2.5
                    s = math.exp(-(ddx * ddx + ddy * ddy) * 0.6) * tw * 1.5 * (1 - v)
                    r += s; g += s; b += s
                # sun
                dx = x - cx; dy = y - sy
                d = math.sqrt(dx * dx + dy * dy) / R
                vy = (y - (sy - R)) / (2 * R)
                sr, sg, sb = hsv(0.13 - 0.16 * vy + hue * 0.03, 0.85 + 0.1 * vy, 1.05)
                if d < 1.0:
                    cut = 1.0
                    rel = (y - sy) / R
                    if rel > -0.15:
                        gap = 0.05 + 0.75 * (rel + 0.15)
                        f = fract(rel * 5.0 - beat * 0.25)
                        cut = smooth(0.0, 0.03, f - gap * 0.5)
                    a = smooth(1.0, 0.985, d) * cut
                    r = r * (1 - a) + sr * a; g = g * (1 - a) + sg * a; b = b * (1 - a) + sb * a
                if d > 0.95:
                    gl = math.exp(-(d - 1.0) * 5.0) * (0.25 + 0.4 * kick)
                    r += sr * gl * 0.6; g += sg * gl * 0.4; b += sb * gl * 0.6
                # spectrum mountains
                ax = abs(x - cx) / (W * 0.5)
                fi = ax * (nb - 6) * 0.7 + 2
                i0 = int(fi); fr = fi - i0
                sv = spec[i0] * (1 - fr) + spec[min(i0 + 1, nb - 1)] * fr
                ridge = 0.5 + 0.5 * math.sin(x * 0.045 + 1.3) * math.sin(x * 0.013)
                mh = H * (0.03 + 0.17 * sv * (0.35 + 0.65 * ax) + 0.04 * ridge * ax) * (0.6 + 0.4 * energy)
                h_ = hor - y
                if h_ < mh:
                    fade = h_ / max(mh, 1.0)
                    r = 0.05 + 0.10 * fade; g = 0.0 + 0.02 * fade; b = 0.12 + 0.18 * fade
                    # wireframe stripes on mountains
                    wl = math.exp(-abs(fract(h_ / 14.0) - 0.5) * 30.0) * 0.25 * fade
                    r += lr * wl * 0.5; g += lg * wl * 0.5; b += lb * wl * 0.5
                el = math.exp(-abs(h_ - mh) * 0.7)
                r += 0.2 * el * (1 + kick); g += 1.0 * el * (1 + kick); b += 1.2 * el * (1 + kick)
            else:
                dy = (y - hor) + 1.0
                z = H * 0.14 / dy
                wx = (x - cx) / W * z * 9.0
                wz = z * 3.0 + travel * 2.0
                fwz = H * 0.14 * 3.0 / (dy * dy)
                fwx = z * 9.0 / W
                lx = abs(fract(wx) - 0.5); lx = 0.5 - lx
                lz = abs(fract(wz) - 0.5); lz = 0.5 - lz
                ax_ = lx / max(fwx, 1e-4); az_ = lz / max(fwz, 1e-4)
                line = math.exp(-ax_ * ax_ * 0.35) + math.exp(-az_ * az_ * 0.35)
                glow = math.exp(-ax_ * 0.25) * 0.25 + math.exp(-az_ * 0.25) * 0.25
                fog = 1.0 - math.exp(-dy / (H * 0.08))
                inten = (line + glow) * (0.9 + 1.4 * kick) * fog
                r = 0.02 + lr * inten; g = 0.0 + lg * inten * 0.8; b = 0.06 + lb * inten
                # sun reflection
                ref = math.exp(-((x - cx) / (W * 0.18)) ** 2) * math.exp(-dy / (H * 0.12)) * 0.5
                sr, sg, sb = hsv(0.0 + hue * 0.03, 0.8, 1.0)
                r += sr * ref; g += sg * ref * 0.6; b += sb * ref * 0.7
                hz = math.exp(-dy / (H * 0.02))
                r += hr * hz * 0.6; g += hg * hz * 0.6; b += hb * hz * 0.6
            img[y, x, 0] = r; img[y, x, 1] = g; img[y, x, 2] = b


# ------------------------------------------------------------------ 2. TUNNEL
@njit(**FM)
def tunnel(img, t, travel, kick, snare, hue, energy, nseg, twist, cx, cy, spec, pattern):
    H, W = img.shape[0], img.shape[1]
    nb = spec.shape[0]
    for y in prange(H):
        for x in range(W):
            dx = (x - cx) / H; dy = (y - cy) / H
            r = math.sqrt(dx * dx + dy * dy) + 1e-4
            a = math.atan2(dy, dx)
            an = abs(a) / math.pi
            si = int(an * (nb - 8)) + 4
            sv = spec[min(si, nb - 1)]
            rr = r * (1.0 - 0.18 * sv * energy)
            v = 0.32 / rr
            u = (a / (2 * math.pi) + 0.5) * nseg + v * twist
            vv = v + travel * 3.0
            cu = fract(u); cv = fract(vv)
            fw = 0.007 + 0.011 * v   # crude AA width
            eu = math.exp(-min(cu, 1 - cu) / fw * 0.9)
            ev = math.exp(-min(cv, 1 - cv) / (fw * 1.3) * 0.9)
            chk = (int(math.floor(u)) + int(math.floor(vv))) & 1
            hh = hue + vv * 0.035 + chk * 0.08
            if pattern == 1:   # diamonds / retro rhombs
                q = abs(fract(u + vv) - 0.5) + abs(fract(u - vv) - 0.5)
                eu = math.exp(-abs(q - 0.5) / fw * 1.2)
                ev = 0.0
            elif pattern == 2:  # concentric hex-ish rings + spokes
                eu = math.exp(-min(cu, 1 - cu) / fw * 2.0) * 0.6
                ev = math.exp(-abs(math.sin(vv * math.pi * 2.0 + math.sin(u * 2.094) * 0.8)) / (fw * 6.0))
            br, bg, bb = hsv(hh, 0.75, 0.08 + 0.10 * chk)
            nr, ng, nbb = hsv(hh + 0.5 * (pattern == 2), 0.65, 1.0)
            e = (eu + ev) * (0.35 + 1.1 * sv) * (0.7 + 1.3 * kick)
            fog = smooth(0.02, 0.45, r)
            cr = (br + nr * e) * fog; cg = (bg + ng * e) * fog; cb = (bb + nbb * e) * fog
            # light at the end of the tunnel
            c = math.exp(-r * 32.0) * (0.9 + 1.6 * kick + 0.6 * snare) + math.exp(-r * 7.0) * 0.06
            lr_, lg_, lb_ = hsv(hue + 0.12, 0.35, 1.0)
            cr += lr_ * c; cg += lg_ * c; cb += lb_ * c
            # travelling ring on each beat
            img[y, x, 0] = cr; img[y, x, 1] = cg; img[y, x, 2] = cb


# ------------------------------------------------------------------ 3. KALEIDOSCOPE
@njit(**FM)
def kaleido(img, t, zoom, rot, nfold, hue, kick, energy, c1, c2, cy):
    H, W = img.shape[0], img.shape[1]
    sector = 2 * math.pi / nfold
    cr_, sr_ = math.cos(rot), math.sin(rot)
    for y in prange(H):
        for x in range(W):
            px = (x - W * 0.5) / H * zoom
            py = (y - cy) / H * zoom
            qx = px * cr_ - py * sr_; qy = px * sr_ + py * cr_
            r = math.sqrt(qx * qx + qy * qy)
            a = math.atan2(qy, qx) + math.pi
            a = a - sector * math.floor(a / sector)
            a = abs(a - sector * 0.5)
            qx = r * math.cos(a); qy = r * math.sin(a)
            t1 = 1e9; t2 = 1e9; t3 = 0.0
            for i in range(8):
                d = qx * qx + qy * qy + 1e-5
                qx = abs(qx) / d - c1
                qy = abs(qy) / d - c2
                m = math.sqrt(qx * qx + qy * qy)
                t1 = min(t1, abs(qx * qy))
                t2 = min(t2, abs(m - 0.7))
                t3 += math.exp(-m * 2.5)
            g1 = math.exp(-t1 * 30.0)
            g2 = math.exp(-t2 * 16.0)
            c_r, c_g, c_b = hsv(hue + t2 * 0.5 + r * 0.15, 0.75, 1.0)
            d_r, d_g, d_b = hsv(hue + 0.45 + t1 * 2.0, 0.6, 1.0)
            k = 0.28 + 0.32 * energy + 0.45 * kick
            vig = smooth(1.25, 0.15, r / zoom * 1.6)
            cr = (c_r * g2 * 0.9 + d_r * g1 * 0.8 + t3 * 0.010) * k * vig
            cg = (c_g * g2 * 0.9 + d_g * g1 * 0.8 + t3 * 0.006) * k * vig
            cb = (c_b * g2 * 0.9 + d_b * g1 * 0.8 + t3 * 0.018) * k * vig
            img[y, x, 0] = cr; img[y, x, 1] = cg; img[y, x, 2] = cb


# ------------------------------------------------------------------ 4. RIDGELINES
@njit(**FM)
def ridgelines(img, lines, offset, hue, kick, energy, t):
    """lines: (M, P) newest last. Painter's algorithm per column."""
    H, W = img.shape[0], img.shape[1]
    M, Pn = lines.shape[0], lines.shape[1]
    for x in prange(W):
        # background
        for y in range(H):
            v = y / H
            br, bg, bb = hsv(0.70 + v * 0.08 + (hue - 0.55) * 0.3, 0.75, 0.03 + 0.07 * v)
            hh = hash2(float(x), float(y))
            if hh > 0.998 and v < 0.6:
                st = (hh - 0.998) * 300 * (0.6 + 0.4 * math.sin(t * 2.0 + hh * 1e3))
                br += st; bg += st; bb += st
            img[y, x, 0] = br; img[y, x, 1] = bg; img[y, x, 2] = bb
        for j in range(M):
            depth = (j + offset) / (M - 1)          # 0 = far, 1 = near
            if depth < 0.0:
                continue
            persp = depth * depth * 0.6 + depth * 0.4
            base = H * (0.24 + 0.52 * persp)
            half = W * (0.22 + 0.33 * persp)
            xw = (x - W * 0.5) / half
            if abs(xw) > 1.05:
                continue
            amp = H * (0.03 + 0.11 * persp) * (0.6 + 0.6 * energy)
            fi = (xw * 0.5 + 0.5) * (Pn - 1)
            fi = min(max(fi, 0.0), Pn - 1.001)
            i0 = int(fi); fr = fi - i0
            hv = lines[j, i0] * (1 - fr) + lines[j, i0 + 1] * fr
            win = math.exp(-xw * xw * 2.2)
            yl = base - amp * hv * win - amp * 0.04
            # previous column for vertical continuity
            xw2 = (x - 1 - W * 0.5) / half
            fi2 = min(max((xw2 * 0.5 + 0.5) * (Pn - 1), 0.0), Pn - 1.001)
            k0 = int(fi2); fr2 = fi2 - k0
            hv2 = lines[j, k0] * (1 - fr2) + lines[j, k0 + 1] * fr2
            yl2 = base - amp * hv2 * math.exp(-xw2 * xw2 * 2.2) - amp * 0.04
            top = int(min(yl, yl2)) - 2
            bot = int(base + amp * 0.25)
            fade = smooth(1.05, 0.85, abs(xw))
            lr, lg, lb = hsv(hue + 0.08 * depth + hv * 0.15, 0.55 - 0.3 * hv, 1.0)
            inten = (0.25 + 0.95 * depth) * (0.8 + 0.6 * hv + 0.8 * kick * depth) * fade
            for y in range(max(0, top), min(H, bot)):
                # occlude below the line
                if y > yl + 1.5:
                    v = y / H
                    br, bg, bb = hsv(0.70 + v * 0.08 + (hue - 0.55) * 0.3, 0.75, 0.02 + 0.05 * v)
                    img[y, x, 0] = br; img[y, x, 1] = bg; img[y, x, 2] = bb
                lo = min(yl, yl2); hi = max(yl, yl2)
                dd = 0.0
                if y < lo:
                    dd = lo - y
                elif y > hi:
                    dd = y - hi
                w = math.exp(-dd * dd * 0.9) * inten
                if w > 0.003:
                    img[y, x, 0] += lr * w; img[y, x, 1] += lg * w; img[y, x, 2] += lb * w


# ------------------------------------------------------------------ 5. NEBULA (half-res background)
@njit(**FM)
def nebula(img, t, hue, energy, kick, drift):
    H, W = img.shape[0], img.shape[1]
    for y in prange(H):
        for x in range(W):
            u = x / H * 2.2; v = y / H * 2.2 + drift
            w1 = fbm(u * 1.3 + 3.0, v * 1.3, t * 0.05, 4)
            w2 = fbm(u * 1.3 - 2.0, v * 1.3 + 5.0, t * 0.05, 4)
            n = fbm(u + 1.6 * w1, v + 1.6 * w2, t * 0.03, 5)
            d = smooth(0.35, 0.85, n)
            c1r, c1g, c1b = hsv(hue + 0.6, 0.8, 1.0)
            c2r, c2g, c2b = hsv(hue + 0.9 + w1 * 0.2, 0.7, 1.0)
            m = w2
            br = (c1r * (1 - m) + c2r * m) * d * d * (0.35 + 0.25 * energy)
            bg = (c1g * (1 - m) + c2g * m) * d * d * (0.35 + 0.25 * energy)
            bb = (c1b * (1 - m) + c2b * m) * d * d * (0.35 + 0.25 * energy)
            dust = smooth(0.55, 0.9, fbm(u * 4.0, v * 4.0, t * 0.02, 3)) * 0.12
            img[y, x, 0] = br + dust * 0.8; img[y, x, 1] = bg + dust * 0.7; img[y, x, 2] = bb + dust
            # tiny stars
            hh = hash2(float(x), float(y) + 0.37)
            if hh > 0.9975:
                s = (hh - 0.9975) * 400 * (0.6 + 0.4 * math.sin(t * 2 + hh * 1e3))
                img[y, x, 0] += s; img[y, x, 1] += s; img[y, x, 2] += s


@njit(**FM)
def planet_ring(img, t, hue, energy, kick, spec, cx, cy, R0, rot):
    H, W = img.shape[0], img.shape[1]
    nb = spec.shape[0]
    Rp = R0 * 0.72
    for y in prange(H):
        for x in range(W):
            dx = x - cx; dy = y - cy
            r = math.sqrt(dx * dx + dy * dy)
            a = math.atan2(dx, -dy)  # 0 at top
            an = abs(a) / math.pi    # mirrored 0..1
            cr = 0.0; cg = 0.0; cb = 0.0
            # planet
            if r < Rp + 2:
                nx = dx / Rp; ny = dy / Rp
                nz2 = 1 - nx * nx - ny * ny
                if nz2 > 0:
                    nz = math.sqrt(nz2)
                    lx, ly, lz = -0.55, -0.45, 0.70
                    dif = max(0.0, nx * lx + ny * ly + nz * lz)
                    lon = math.atan2(nx, nz) + rot
                    band = 0.5 + 0.5 * math.sin(ny * 13.0 + 1.6 * fbm(lon * 1.5, ny * 3.0, t * 0.1, 3))
                    pr, pg, pb = hsv(hue + 0.05 + band * 0.08, 0.6, 0.25 + 0.75 * band)
                    edge = smooth(1.0, 0.985, math.sqrt(nx * nx + ny * ny))
                    sh = dif * 0.95 + 0.04
                    cr = pr * sh * edge; cg = pg * sh * edge; cb = pb * sh * edge
                # atmosphere
                rim = math.exp(-abs(r - Rp) * 0.08) * 0.6
                ar, ag, ab = hsv(hue + 0.55, 0.6, 1.0)
                cr += ar * rim; cg += ag * rim; cb += ab * rim
            else:
                rim = math.exp(-(r - Rp) * 0.03) * (0.25 + 0.4 * kick)
                ar, ag, ab = hsv(hue + 0.55, 0.6, 1.0)
                cr += ar * rim; cg += ag * rim; cb += ab * rim
            # spectrum ring (radial bars, mirrored)
            fi = an * (nb - 10) + 2
            bi = int(fi)
            frb = fi - bi
            sv = spec[bi] * (1 - frb) + spec[min(bi + 1, nb - 1)] * frb
            L = R0 * (0.06 + 0.55 * sv * sv * (0.6 + 0.5 * energy)) * (1 + 0.15 * kick)
            seg = fract(an * 72.0)
            if r > R0 and r < R0 + L + 2:
                gap = smooth(0.0, 0.08, seg) * smooth(1.0, 0.75, seg)
                al = smooth(R0 + L + 2, R0 + L - 1, r) * gap
                g = (r - R0) / max(L, 1.0)
                br, bg, bb = hsv(hue + 0.5 + g * 0.25 + an * 0.15, 0.7, 1.4)
                cr += br * al; cg += bg * al; cb += bb * al
            # thin base circle
            c = math.exp(-abs(r - R0 + 4) * 0.6) * (0.8 + kick)
            cr += c * 0.9; cg += c * 0.9; cb += c
            img[y, x, 0] += cr; img[y, x, 1] += cg; img[y, x, 2] += cb


# ------------------------------------------------------------------ scope background (CRT graticule)
@njit(**FM)
def graticule(img, hue, inten):
    H, W = img.shape[0], img.shape[1]
    step = W / 10.0
    cyy = H * 0.45
    for y in prange(H):
        for x in range(W):
            gx = abs(fract((x - W * 0.5) / step + 0.5) - 0.5) * step
            gy = abs(fract((y - cyy) / step + 0.5) - 0.5) * step
            l = math.exp(-gx * gx * 0.5) + math.exp(-gy * gy * 0.5)
            dots = math.exp(-((fract((x - W * 0.5) / (step / 5) + 0.5) - 0.5) ** 2 * 400 + gy * gy * 0.3))
            r_, g_, b_ = hsv(hue, 0.6, 1.0)
            v = (l * 0.05 + dots * 0.06) * inten
            img[y, x, 0] = r_ * v + 0.004; img[y, x, 1] = g_ * v + 0.008; img[y, x, 2] = b_ * v + 0.01


# ------------------------------------------------------------------ post
@njit(**FM)
def downsample4(src, dst):
    H, W = dst.shape[0], dst.shape[1]
    for y in prange(H):
        for x in range(W):
            for c in range(3):
                s = 0.0
                for j in range(4):
                    for i in range(4):
                        s += src[y * 4 + j, x * 4 + i, c]
                dst[y, x, c] = s / 16.0


@njit(inline="always", fastmath=True, cache=True)
def _bil(img, x, y, c):
    H, W = img.shape[0], img.shape[1]
    x = min(max(x, 0.0), W - 1.001); y = min(max(y, 0.0), H - 1.001)
    ix = int(x); iy = int(y); fx = x - ix; fy = y - iy
    return (img[iy, ix, c] * (1 - fx) * (1 - fy) + img[iy, ix + 1, c] * fx * (1 - fy) +
            img[iy + 1, ix, c] * (1 - fx) * fy + img[iy + 1, ix + 1, c] * fx * fy)


@njit(**FM)
def finalize(img, bloom1, bloom2, out, zoom, aberr, bloom_amt, scan, vign, grain, seed, exposure, shake_x, shake_y, sat=1.0):
    H, W = img.shape[0], img.shape[1]
    bh, bw = bloom1.shape[0], bloom1.shape[1]
    cx = W * 0.5; cy = H * 0.5
    for y in prange(H):
        for x in range(W):
            dx = (x - cx) / zoom + shake_x; dy = (y - cy) / zoom + shake_y
            rr = math.sqrt(dx * dx + dy * dy) / H
            k = aberr * rr
            c0 = 0.0; c1 = 0.0; c2 = 0.0
            for c in range(3):
                s = 1.0 + k * (c - 1)
                sx = cx + dx * s; sy = cy + dy * s
                v = _bil(img, sx, sy, c)
                bx = sx / 4.0 - 0.5; by = sy / 4.0 - 0.5
                v += (_bil(bloom1, bx, by, c) * 0.35 + _bil(bloom2, bx, by, c) * 0.45) * bloom_amt
                if c == 0:
                    c0 = v
                elif c == 1:
                    c1 = v
                else:
                    c2 = v
            vg = 1.0 - vign * ((x - cx) ** 2 / (cx * cx) * 0.6 + (y - cy) ** 2 / (cy * cy) * 0.4)
            sl = 1.0 - scan * (0.5 + 0.5 * math.cos(y * math.pi * 2.0 / 3.0))
            n = (hash2(x * 1.0 + seed, y * 1.0 - seed) - 0.5) * grain
            lum = 0.3 * c0 + 0.55 * c1 + 0.15 * c2
            c0 = max(0.0, lum + (c0 - lum) * sat); c1 = max(0.0, lum + (c1 - lum) * sat); c2 = max(0.0, lum + (c2 - lum) * sat)
            for c in range(3):
                v = (c0 if c == 0 else (c1 if c == 1 else c2)) * exposure * vg * sl
                v = 1.0 - math.exp(-v)           # filmic-ish tonemap
                v = v + n
                out[y, x, c] = np.uint8(min(255.0, max(0.0, v * 255.0)))
