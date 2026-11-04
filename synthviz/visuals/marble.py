"""MARBLE: a bright paper-marbling (ebru) world played by the music.

Not a scene inside the dark neon pipeline, but its own renderer: a sheet of
paper where every musical event is a marbling operation, computed exactly with
the classic ink-drop math (a drop of radius r at c pushes every older point p
out to c + (p - c) * sqrt(1 + r^2 / |p - c|^2)), so the picture is the whole
history of the track up to this frame:

    kick in a drop   -> a big ink drop near the centre: bullseye rings pulse out
    snare            -> a comb stroke (tines) drags the rings into feathers
    lead note        -> a small drop; x from pitch, colour from pitch class
    build            -> drops spiral in, smaller and faster
    impact (drop on) -> one huge drop floods the sheet in the new palette
    bass             -> the paper breathes (wave warp), kicks punch the zoom

Flat, saturated pigments on paper: nothing is dimmed by bloom or a dark grade.
Each video gets its own palette pair, layout and stroke directions from the seed.
"""
from __future__ import annotations

import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter, maximum_filter
from PIL import Image, ImageDraw, ImageFont

from ..composer import save_wav
from .features import Features
from .notes import NoteRibbon
from .captions import Captions

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


# paper, ink line, 5 pigments
PALETTES = {
    "candy":    ("#FFF3DC", "#1A1033", ["#FF2E88", "#FFD000", "#00B7FF", "#7B2CFF", "#FF6A00"]),
    "acid":     ("#1E00FF", "#0A0030", ["#D4FF00", "#FF2BD6", "#00F0FF", "#FFFFFF", "#FF7A00"]),
    "tropical": ("#00C9A7", "#04332B", ["#FFE600", "#FF3D7F", "#FFFFFF", "#6A00F4", "#FF8C00"]),
    "sunset":   ("#FF4F1F", "#2B0505", ["#FFE14D", "#2BD9FE", "#FFFFFF", "#B5179E", "#3A0CA3"]),
    "bubblegum": ("#FF8FD8", "#2D0034", ["#00E5FF", "#FFF200", "#7000FF", "#FFFFFF", "#00D084"]),
    "lime":     ("#C6FF00", "#102000", ["#FF0080", "#0033FF", "#FFFFFF", "#FF6D00", "#00C2D1"]),
    "ultra":    ("#FFFFFF", "#0B0B2A", ["#0047FF", "#FF1F4B", "#FFC400", "#00D27A", "#9B00FF"]),
}

DROP, COMB, WAVE = 0, 1, 2


@njit(parallel=True, fastmath=True, cache=True)
def _paint(out, W, H, ops, n_ops, cols, paper, zoom, wave_a, wave_ph, sway):
    """ops rows: kind, cx, cy, r(anim), outline_w, dirx, diry, spacing, z, u, col_idx, line_r,g,b"""
    cx0 = W * 0.5
    cy0 = H * 0.5
    for y in prange(H):
        for x in range(W):
            # camera: kick zoom + bass breathing
            px = (x - cx0) / zoom + cx0
            py = (y - cy0) / zoom + cy0
            px += wave_a * math.sin(py * 0.0105 + wave_ph) + sway * math.sin(py * 0.0031 - wave_ph * 0.3)
            py += wave_a * 0.6 * math.sin(px * 0.0087 - wave_ph * 0.7)
            ar = 0.0; ag = 0.0; ab = 0.0; aa = 0.0
            k = n_ops - 1
            while k >= 0 and aa < 0.995:
                kind = int(ops[k, 0])
                if kind == 0:
                    cx = ops[k, 1]; cy = ops[k, 2]; r = ops[k, 3]
                    if r > 0.0:
                        dx = px - cx; dy = py - cy
                        d = math.sqrt(dx * dx + dy * dy) + 1e-6
                        if d < r + 1.0:
                            w = r - d + 0.5
                            w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
                            ci = int(ops[k, 10])
                            cr = cols[ci, 0]; cg = cols[ci, 1]; cb = cols[ci, 2]
                            ow = ops[k, 4]
                            if ow > 0.0:
                                e = (d - (r - ow)) / 1.2 + 0.5
                                e = 0.0 if e < 0.0 else (1.0 if e > 1.0 else e)
                                cr = cr + (ops[k, 11] - cr) * e
                                cg = cg + (ops[k, 12] - cg) * e
                                cb = cb + (ops[k, 13] - cb) * e
                            f = (1.0 - aa) * w
                            ar += f * cr; ag += f * cg; ab += f * cb; aa += f
                        if d > r:
                            s = math.sqrt(1.0 - (r * r) / (d * d))
                            px = cx + dx * s
                            py = cy + dy * s
                        else:
                            px = cx; py = cy
                elif kind == 1:
                    # comb: tines perpendicular to (dirx, diry), drag along it
                    dirx = ops[k, 5]; diry = ops[k, 6]
                    sp = ops[k, 7]; z = ops[k, 8]; u = ops[k, 9]
                    v = (px - ops[k, 1]) * (-diry) + (py - ops[k, 2]) * dirx
                    dd = abs(v - sp * math.floor(v / sp + 0.5))
                    m = z * u / (dd + u)
                    px -= dirx * m
                    py -= diry * m
                else:
                    # sine wave stroke
                    dirx = ops[k, 5]; diry = ops[k, 6]
                    sp = ops[k, 7]; z = ops[k, 8]
                    v = (px - ops[k, 1]) * (-diry) + (py - ops[k, 2]) * dirx
                    m = z * math.sin(v * 6.2831853 / sp)
                    px -= dirx * m
                    py -= diry * m
                k -= 1
            f = 1.0 - aa
            r_ = ar + f * paper[0]; g_ = ag + f * paper[1]; b_ = ab + f * paper[2]
            out[y, x, 0] = np.uint8(min(255.0, max(0.0, r_)))
            out[y, x, 1] = np.uint8(min(255.0, max(0.0, g_)))
            out[y, x, 2] = np.uint8(min(255.0, max(0.0, b_)))


def _ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


class MarbleScore:
    """Turns song events into a timed list of marbling operations."""

    def __init__(self, F, W, H, rng, palettes=None):
        self.F, self.W, self.H = F, W, H
        names = palettes or rng.sample(list(PALETTES), 2)
        self.palette_names = names
        self.pals = [PALETTES[n] for n in names]
        # colour table: per palette 5 pigments + paper
        cols = []
        for paper, line, pig in self.pals:
            cols += [_hex(c) for c in pig] + [_hex(paper)]
        self.cols = np.array(cols, np.float64)
        self.ops = []          # (t, anim_dur, kind, params...)
        ev = F.ev
        spb = ev["spb"]
        secs = F.sections
        sec_at = lambda t: F.section(t)
        drops = [s for s in secs if s["name"] in ("drop", "drop2")]
        pal_of = lambda t: 1 if any(s["name"] == "drop2" and s["t"] <= t + 1e-6 for s in secs) else 0
        Wc, Hc = W / 2, H / 2
        diag = math.hypot(W, H)
        self.paper0 = _hex(self.pals[0][0])
        # opening: a few concentric stones so frame 0 is already a picture
        for j, r in enumerate([520, 380, 250, 140, 60]):
            self._drop(-1.0 + j * 0.01, 0.0, Wc, Hc * 0.92, r, j % 5, 0, outline=3.0)
        self._wave(-0.5, 0.0, Wc, Hc, rng.uniform(0, math.pi), 260, 34)

        # lead notes -> small drops on a path that drifts with the melody
        p_lo, p_hi = F.p_lo, F.p_hi
        for n in ev["lead"]:
            s = sec_at(n["t"])
            if s["name"] in ("drop", "drop2"):
                continue                          # drops belong to the kick
            xr = (n["p"] - p_lo) / max(1, p_hi - p_lo)
            x = W * (0.15 + 0.7 * xr) + rng.uniform(-40, 40)
            y = H * (0.5 + 0.28 * math.sin(n["t"] * 0.9 + xr * 3)) + rng.uniform(-60, 60)
            r = 40 + 90 * min(1.0, n["dur"] / (spb * 1.5))
            self._drop(n["t"], 0.18, x, y, r, n["p"] % 5, pal_of(n["t"]))
        # build: hats / snares spiral in, smaller and faster
        for s in secs:
            if s["name"] != "build":
                continue
            hits = [h for h in F.snares + F.hats if s["t"] <= h < s["t"] + s["dur"]]
            hits = sorted(hits)[::2] if len(hits) > 48 else sorted(hits)
            for j, h in enumerate(hits):
                u = (h - s["t"]) / s["dur"]
                ang = j * 2.39996
                rad = (1 - u) * W * 0.42 + 30
                self._drop(h, 0.12, Wc + math.cos(ang) * rad, Hc + math.sin(ang) * rad * 1.3,
                           26 + 50 * (1 - u), j % 5, pal_of(h), outline=2.0)
        # drops: impact floods, kicks make bullseye rings, snares comb
        for di, s in enumerate(drops):
            a, b = s["t"], s["t"] + s["dur"]
            pal = pal_of(a + 0.01)
            cx, cy = Wc + rng.uniform(-120, 120), Hc + rng.uniform(-200, 200)
            self._drop(a, 0.35, cx, cy, diag * 0.62, rng.randrange(5), pal, outline=6.0)
            kicks = [k for k in F.kicks if a + 0.05 < k < b]
            centres = [(cx, cy)]
            for j, k in enumerate(kicks):
                if j % 16 == 15:                  # move the stone every 4 bars-ish
                    centres.append((Wc + rng.uniform(-260, 260), Hc + rng.uniform(-520, 520)))
                c = centres[-1]
                r = 150 + 120 * rng.random()
                self._drop(k, 0.16, c[0], c[1], r, j % 5, pal, outline=4.0)
            snares = [x for x in F.snares if a + 0.05 < x < b]
            last = -9
            for j, x in enumerate(snares):
                if x - last < spb * 0.9:
                    continue
                last = x
                ang = (j % 4) * math.pi / 2 + (math.pi / 4 if di % 2 else 0.0) + rng.uniform(-0.25, 0.25)
                self._comb(x, 0.22, Wc, Hc, ang, rng.choice([110, 140, 180]), rng.uniform(55, 95), 26)
            # lead notes in the drop: confetti drops along the top/bottom thirds
            for n in ev["lead"]:
                if a <= n["t"] < b and rng.random() < 0.6:
                    xr = (n["p"] - p_lo) / max(1, p_hi - p_lo)
                    yy = H * (0.18 if (n["p"] % 2) else 0.82) + rng.uniform(-90, 90)
                    self._drop(n["t"], 0.12, W * (0.1 + 0.8 * xr), yy, 28 + 30 * rng.random(), (n["p"] + 2) % 5, pal)
        # breaks / outro: slow sine strokes on chord changes
        for c in F.chords:
            s = sec_at(c["t"])
            if s["name"] in ("break", "outro", "intro") and c["t"] > 0.2:
                self._wave(c["t"], c["dur"] * 0.9, Wc, Hc, rng.uniform(0, math.pi), rng.choice([300, 420, 560]), 22)
        self.ops.sort(key=lambda o: o[0])
        self.t = np.array([o[0] for o in self.ops])

    # op = (t, anim, kind, cx, cy, r, outline, dirx, diry, spacing, z, u, col_idx, line_rgb)
    def _line(self, pal):
        return _hex(self.pals[pal][1])

    def _drop(self, t, anim, x, y, r, ci, pal, outline=0.0):
        self.ops.append((t, anim, DROP, x, y, r, outline, 0, 0, 1, 0, 1, pal * 6 + ci, self._line(pal)))

    def _comb(self, t, anim, x, y, ang, spacing, z, u):
        self.ops.append((t, anim, COMB, x, y, 0, 0, math.cos(ang), math.sin(ang), spacing, z, u, 0, [0, 0, 0]))

    def _wave(self, t, anim, x, y, ang, spacing, z):
        self.ops.append((t, anim, WAVE, x, y, 0, 0, math.cos(ang), math.sin(ang), spacing, z, 1, 0, [0, 0, 0]))

    def at(self, t):
        n = int(np.searchsorted(self.t, t, side="right"))
        arr = np.zeros((max(1, n), 14), np.float64)
        for i in range(n):
            o = self.ops[i]
            a = 1.0 if o[1] <= 0 else _ease_out((t - o[0]) / o[1])
            arr[i, 0] = o[2]; arr[i, 1] = o[3]; arr[i, 2] = o[4]
            arr[i, 3] = o[5] * math.sqrt(a) if o[2] == DROP else 0.0
            arr[i, 4] = o[6]; arr[i, 5] = o[7]; arr[i, 6] = o[8]; arr[i, 7] = o[9]
            arr[i, 8] = o[10] * a; arr[i, 9] = o[11]; arr[i, 10] = o[12]
            arr[i, 11:14] = o[13]
        # a drop that covers the whole sheet hides everything older: cut the history there
        cut = 0
        diag = math.hypot(self.W, self.H)
        for i in range(n - 1, -1, -1):
            if arr[i, 0] == DROP:
                far = max(math.hypot(arr[i, 1] - cx, arr[i, 2] - cy) for cx in (0, self.W) for cy in (0, self.H))
                if arr[i, 3] > far + 80:
                    cut = i
                    break
        return arr[cut:n], n - cut

    def paper(self, t):
        return np.array(self.paper0, np.float64)


class TextLayer:
    """Title card + DROP slams, pre-rendered RGBA, composited only where alpha > 0."""

    def __init__(self, W, H, title, sub, ink, accent):
        self.W, self.H = W, H
        self.cards = {}
        self.cards["title"] = self._card([(title, 118, ink, "Oswald.ttf"), (sub, 40, ink, "Inter-SemiBold.otf")],
                                         y=170, accent=accent)
        self.cards["drop"] = self._card([("DROP", 260, ink, "Oswald.ttf")], y=H // 2 - 170, accent=accent)
        self.cards["drop2"] = self._card([("DROP 2", 230, ink, "Oswald.ttf")], y=H // 2 - 160, accent=accent)
        self.cards["tag"] = self._card([(title, 44, ink, "Oswald.ttf")], y=90, accent=None)

    def _card(self, lines, y, accent):
        im = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        yy = y
        for text, size, col, font in lines:
            f = ImageFont.truetype(str(FONT_DIR / font), size)
            w = d.textlength(text, font=f)
            x = (self.W - w) / 2
            if accent is not None:
                d.text((x + size * 0.06, yy + size * 0.06), text, font=f, fill=tuple(accent) + (255,))
            d.text((x, yy), text, font=f, fill=tuple(col) + (255,), stroke_width=max(2, size // 40),
                   stroke_fill=(255, 255, 255, 255))
            yy += int(size * 1.15)
        a = np.asarray(im)
        ys, xs = np.nonzero(a[..., 3])
        if not len(ys):
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return (y0, x0, a[y0:y1, x0:x1, :3].astype(np.float32), a[y0:y1, x0:x1, 3:].astype(np.float32) / 255)

    def draw(self, frame, name, alpha, scale=1.0):
        c = self.cards.get(name)
        if c is None or alpha <= 0:
            return
        y0, x0, rgb, a = c
        if scale != 1.0:
            h, w = rgb.shape[:2]
            nh, nw = int(h * scale), int(w * scale)
            im = Image.fromarray(np.concatenate([rgb, a * 255], 2).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
            arr = np.asarray(im).astype(np.float32)
            rgb, a = arr[..., :3], arr[..., 3:] / 255
            y0 = int(y0 + (h - nh) / 2); x0 = int(x0 + (w - nw) / 2)
        h, w = rgb.shape[:2]
        ya, xa = max(0, y0), max(0, x0)
        yb, xb = min(self.H, y0 + h), min(self.W, x0 + w)
        if yb <= ya or xb <= xa:
            return
        sub = frame[ya:yb, xa:xb].astype(np.float32)
        aa = a[ya - y0: yb - y0, xa - x0: xb - x0] * alpha
        frame[ya:yb, xa:xb] = (sub * (1 - aa) + rgb[ya - y0: yb - y0, xa - x0: xb - x0] * aa).astype(np.uint8)


def render_marble(song, out, seed=0, fps=30, W=1080, H=1920, crf=20, palettes=None, preview_times=None, log=print):
    t_start = time.time()
    rng = random.Random(seed)
    F = Features(song, fps)
    score = MarbleScore(F, W, H, rng, palettes)
    log(f"[marble] palettes={score.palette_names} ops={len(score.ops)}")
    ink = _hex(score.pals[0][1])
    accent = _hex(score.pals[0][2][0])
    ev = song.events
    sub = f"{ev.get('style', '')} · {ev['bpm']:.0f} bpm · {ev['key'].lower()}"
    text = TextLayer(W, H, song.title, sub, ink, accent)
    # the v4 note ribbon (melody, bass, chords, drum lanes) + kinetic captions on a dark ink panel
    import colorsys
    hue = colorsys.rgb_to_hsv(*[c / 255 for c in accent])[0]
    ribbon = NoteRibbon(F, W, H)
    caps = Captions(F, W, H, song.title, hue, ribbon=ribbon, outro=rng.choice(["question", "rate", "replay"]),
                    subtitle=sub)
    by0, by1 = int(ribbon.LABEL_Y - 70), int(ribbon.DRUM_Y + 46)
    fy = np.minimum(1.0, np.minimum(np.arange(by0, by1) - by0, by1 - np.arange(by0, by1)) / 18.0)[:, None, None]
    ink_f = np.array(ink, np.float32)[None, None, :]
    buf = np.zeros((H, W, 3), np.float32)
    cap_l = np.zeros((H, W, 3), np.uint8)
    drops = [s for s in F.sections if s["name"] in ("drop", "drop2")]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((H, W, 3), np.uint8)
    frames = list(range(F.n_frames)) if preview_times is None else [min(F.n_frames - 1, int(t * fps)) for t in preview_times]
    proc = None
    if preview_times is None:
        wav = out.with_suffix(".wav")
        save_wav(wav, song.audio)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(fps), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    previews = []
    ph = 0.0
    for n_, i in enumerate(frames):
        t = i / fps
        ops, n = score.at(t)
        kick = F.pulse("kick", t, 0.09)
        impact = F.pulse("impact", t, 0.3)
        bass = float(F.env.get("bass", np.zeros(F.n_frames))[i])
        energy = float(F.energy[i])
        zoom = 1.0 + 0.028 * kick + 0.06 * impact
        ph += (0.6 + 2.2 * energy) / fps
        _paint(frame, W, H, ops, n, score.cols, score.paper(t), zoom, 4 + 16 * bass * energy, ph * 2.0,
               10 * math.sin(t * 0.35))
        # note ribbon on an ink panel
        ui = min(1.0, t / 0.8) * (1.0 - _ease_out((t - caps.outro_t) / 0.6))
        if ui > 0.01:
            band = frame[by0:by1].astype(np.float32)
            band = band + (ink_f - band) * (0.9 * fy * ui)
            buf[by0 - 40:by1 + 40] = 0
            ribbon.draw(buf, i, t, hue, ui)
            light = 255.0 * (1.0 - np.exp(-2.6 * buf[by0:by1]))
            frame[by0:by1] = np.clip(band + light, 0, 255).astype(np.uint8)
        # captions are glow-on-dark: draw them on black, then lay an ink shadow under them on the paper
        cap_l[:] = 0
        caps.draw(cap_l, i, t, hue, 1.0)
        a4 = cap_l[::4, ::4].max(axis=2).astype(np.float32) / 255.0
        if a4.max() > 0.02:
            sh = gaussian_filter(maximum_filter(a4, size=5), 2.0)
            sh = np.clip(sh * 2.2, 0, 0.82)
            sh = np.repeat(np.repeat(sh, 4, 0), 4, 1)[:H, :W, None]
            f_ = frame.astype(np.float32)
            f_ = f_ + (ink_f - f_) * sh
            frame[:] = np.clip(f_ + cap_l, 0, 255).astype(np.uint8)
        for s in drops:
            age = t - s["t"]
            if 0 <= age < 0.9:
                text.draw(frame, s["name"], 1.0 - _ease_out((age - 0.5) / 0.4), 1.35 - 0.35 * _ease_out(age / 0.25))
        if proc:
            proc.stdin.write(frame.tobytes())
        else:
            previews.append(frame.copy())
        if n_ % 150 == 0 and n_:
            el = time.time() - t_start
            log(f"  frame {n_}/{len(frames)}  {el / n_:.3f}s/frame  ops={n}")
    if proc:
        proc.stdin.close(); proc.wait()
        log(f"[done] {out}  {time.time() - t_start:.1f}s")
        return out
    return previews
