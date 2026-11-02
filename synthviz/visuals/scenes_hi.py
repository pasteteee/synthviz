"""Hi-res inserts: little 'screens' drawn with PIL at full resolution (940 x 750) instead of pixel art.

  chat   a messenger conversation: messages pop in on melody notes, the other side is "typing..." on the
         hi-hats, and the story ends on the drop with a big "i love u" and a burst of hearts.
  tape   a cassette deck: reels spin with the tempo, VU needles jump with the kick, the counter runs,
         the label carries the track title.
  keys   a synth keyboard with falling notes (piano-roll waterfall): every bar lands on its key exactly
         when the note sounds; lead and bass in different colours.

They plug into ArcadeLayer like the pixel games (hires = True -> draw_hi(t) returns an (H, W, 3) image)."""
from __future__ import annotations

import bisect
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pixelkit import hsv, smoothstep, pitch_color
from .inserts import Insert, CW, CH, SCALE

W, H = CW * SCALE, CH * SCALE
FONTS = Path(__file__).resolve().parents[1] / "fonts"
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"


@lru_cache(maxsize=16)
def font(name, px):
    return ImageFont.truetype(str(FONTS / name), px)


@lru_cache(maxsize=64)
def emoji(ch, px):
    """Colour emoji as an RGBA image of height px (None if the emoji font is missing)."""
    try:
        f = ImageFont.truetype(EMOJI_FONT, 109)
    except OSError:
        return None
    im = Image.new("RGBA", (160, 136), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((0, 0), ch, font=f, embedded_color=True)
    bb = im.getbbox()
    if not bb:
        return None
    im = im.crop(bb)
    return im.resize((max(1, int(im.width * px / im.height)), px), Image.LANCZOS)


def to_arr(im):
    return np.asarray(im.convert("RGB")).astype(np.float32) / 255.0


def rgb(c, k=1.0):
    c = np.clip(np.asarray(c, np.float32) * k, 0, 1)
    return tuple(int(v * 255) for v in c)


class HiInsert(Insert):
    hires = True

    def draw(self, cv, t):        # pixel path unused
        pass

    def draw_hi(self, t):
        raise NotImplementedError


# ---------------------------------------------------------------- chat
CHAT_STORIES = [
    [("in", "hey"), ("in", "u up?"), ("out", "yeah, can't sleep"), ("in", "what are u listening to"),
     ("out", "this track on repeat", "🎧"), ("in", "send it"), ("out", "wait for the drop"), ("in", "ok i love u", "💜")],
    [("out", "where are u??"), ("in", "front row"), ("in", "the bass is shaking my phone", "📳"), ("out", "record it"),
     ("in", "too busy dancing", "🕺"), ("out", "save me a spot"), ("in", "always"), ("in", "i love u", "❤️")],
    [("in", "did u make this?"), ("out", "maybe", "👀"), ("in", "it's stuck in my head"), ("out", "good"),
     ("in", "play it again"), ("out", "turn it up first", "🔊"), ("in", "ok i love u", "💖")],
    [("out", "miss u"), ("in", "miss u more"), ("out", "listen to this"), ("in", "omg the melody", "😭"),
     ("out", "it's about u"), ("in", "stop", "🥺"), ("in", "i love u", "💗")],
]


class Chat(HiInsert):
    name = "chat"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        rng = np.random.default_rng(self.seed + 17)
        self.story = CHAT_STORIES[int(rng.integers(len(CHAT_STORIES)))]
        # one message per melody onset, at least one beat apart; the last (love) lands on a downbeat near the end
        gap = self.spb * 1.0
        ts, last = [], -9.0
        for n in self.notes:
            if self.t0 + 0.4 <= n["t"] < self.t1 - 1.0 and n["t"] - last >= gap:
                ts.append(n["t"]); last = n["t"]
        k_ = len(self.story)
        if len(ts) >= k_:
            idx = np.linspace(0, len(ts) - 1, k_).round().astype(int)
            ts = [ts[i] for i in idx]
        else:
            ts = list(np.linspace(self.t0 + 0.5, self.t1 - 1.2, k_))
        kicks = [x for x in self.F.kicks if self.t0 < x < self.t1 - 0.4]
        if kicks:
            ts[-1] = max(ts[-2] + self.spb * 0.5 if k_ > 1 else ts[-1], min(kicks, key=lambda x: abs(x - ts[-1])))
        self.times = ts
        self.hue = self.cfg.hue
        self.bubbles = [self._bubble(m) for m in self.story]

    def _bubble(self, m):
        side, text = m[0], m[1]
        em = m[2] if len(m) > 2 else None
        big = "love" in text
        f = font("Inter-SemiBold.otf", 46 if big else 36)
        tw = int(ImageDraw.Draw(Image.new("L", (1, 1))).textlength(text, font=f))
        e_im = emoji(em, 46 if big else 38) if em else None
        ew = (e_im.width + 12) if e_im is not None else 0
        pw, ph = tw + ew + 52, (78 if big else 66)
        im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        if side == "out":
            c0 = rgb(hsv(self.cfg.hue + 0.55, 0.65, 0.95))
        else:
            c0 = rgb(hsv(self.cfg.hue + 0.95, 0.15, 0.30)) if not big else rgb(hsv(self.cfg.hue + 0.92, 0.7, 0.95))
        d.rounded_rectangle((0, 0, pw - 1, ph - 1), radius=ph // 2, fill=c0 + (255,))
        d.text((26, ph / 2), text, font=f, anchor="lm", fill=(255, 255, 255, 255))
        if e_im is not None:
            im.alpha_composite(e_im, (26 + tw + 12, (ph - e_im.height) // 2))
        return side, im

    def draw_hi(self, t):
        F = self.F
        hat = F.pulse("hat", t, 0.08)
        kick = F.pulse("kick", t, 0.12)
        img = Image.new("RGB", (W, H), rgb(hsv(self.hue + 0.75, 0.5, 0.07)))
        d = ImageDraw.Draw(img, "RGBA")
        # subtle wallpaper dots
        for y in range(120, H, 46):
            for x in range(20 + (y // 46 % 2) * 23, W, 46):
                d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=rgb(hsv(self.hue + 0.75, 0.4, 0.13)) + (255,))
        # header
        d.rectangle((0, 0, W, 104), fill=rgb(hsv(self.hue + 0.75, 0.45, 0.12)))
        d.ellipse((30, 20, 94, 84), fill=rgb(hsv(self.hue + 0.92, 0.7, 0.9)))
        d.text((62, 52), "B", font=font("Oswald.ttf", 38), anchor="mm", fill=(255, 255, 255))
        d.text((114, 34), "bit", font=font("Inter-SemiBold.otf", 32), anchor="lm", fill=(255, 255, 255))
        i_next = bisect.bisect_right(self.times, t)
        typing = i_next < len(self.times) and self.story[i_next][0] == "in" and self.times[i_next] - t < self.spb * 1.6
        d.text((114, 70), "typing..." if typing else "online", font=font("Inter-Medium.otf", 24), anchor="lm",
               fill=rgb(hsv(self.hue + 0.45, 0.6, 1.0)))
        # stack of messages from the bottom
        shown = [(i, self.times[i]) for i in range(len(self.times)) if self.times[i] <= t]
        y = H - 40
        if typing:
            y -= 70
            bx = 40
            d.rounded_rectangle((bx, y, bx + 120, y + 58), radius=29, fill=rgb(hsv(self.hue + 0.95, 0.15, 0.30)))
            for j in range(3):
                ph = (t / (self.spb / 2) + j * 0.33) % 1
                dy = -8 * math.sin(math.pi * ph) * (0.5 + hat)
                cx = bx + 32 + j * 28
                d.ellipse((cx - 8, y + 29 - 8 + dy, cx + 8, y + 29 + 8 + dy), fill=(230, 230, 240))
            y -= 16
        for i, ts in reversed(shown):
            side, bim = self.bubbles[i]
            age = t - ts
            s = 1.0 + 0.18 * math.exp(-age / 0.09) * math.sin(age * 40) if age < 0.4 else 1.0
            s = s * smoothstep(age / 0.08) if age < 0.08 else s
            if s <= 0.02:
                continue
            bw, bh = max(1, int(bim.width * s)), max(1, int(bim.height * s))
            b2 = bim.resize((bw, bh), Image.BILINEAR) if (bw, bh) != bim.size else bim
            y -= bim.height
            x = W - 40 - bim.width if side == "out" else 40
            ox, oy = x + (bim.width - bw) // 2, y + (bim.height - bh) // 2
            if y + bim.height < 110:
                break
            img.paste(b2, (ox, oy), b2)
            if side == "out" and age > self.spb:       # read receipt
                d.text((x + bim.width - 4, y + bim.height + 4), "read", font=font("Inter-Medium.otf", 18), anchor="ra",
                       fill=rgb(hsv(self.hue + 0.5, 0.3, 0.75)))
            y -= 22
        arr = to_arr(img)
        # hearts burst on the last message
        tl = self.times[-1]
        if t >= tl:
            arr = self._hearts(arr, t - tl)
        return arr * (1.0 + 0.06 * kick)

    def _hearts(self, arr, age):
        rng = np.random.default_rng(self.seed + 5)
        n = 22
        hearts = [h for h in [emoji(c, 54) for c in ("💜", "❤️", "💖", "💗")] if h is not None]
        if not hearts or age > 2.5:
            return arr
        im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
        for j in range(n):
            ang = rng.uniform(-math.pi * 0.9, -math.pi * 0.1)
            sp = rng.uniform(250, 600)
            x = W / 2 + math.cos(ang) * sp * age
            y = H * 0.7 + math.sin(ang) * sp * age + 300 * age * age
            hi = hearts[j % len(hearts)]
            s = rng.uniform(0.6, 1.2) * (1 - 0.3 * age)
            hh = hi.resize((max(1, int(hi.width * s)), max(1, int(hi.height * s))))
            if age > 1.8:
                a = np.asarray(hh).copy(); a[..., 3] = (a[..., 3] * max(0.0, (2.5 - age) / 0.7)).astype(np.uint8)
                hh = Image.fromarray(a)
            im.alpha_composite(hh, (int(x - hh.width / 2), int(y - hh.height / 2)))
        return to_arr(im)


# ---------------------------------------------------------------- tape deck
class Tape(HiInsert):
    name = "tape"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.title = (self.F.ev.get("title") or "MIXTAPE").upper()
        self.ret = 0.0

    def draw_hi(self, t):
        F = self.F
        kick = F.pulse("kick", t, 0.10)
        snare = F.pulse("snare", t, 0.10)
        en = float(F.energy[min(len(F.energy) - 1, int(t * F.fps))])
        hue = self.cfg.hue
        img = Image.new("RGB", (W, H), rgb(hsv(hue + 0.05, 0.35, 0.10)))
        d = ImageDraw.Draw(img, "RGBA")
        # deck body (brushed metal stripes)
        for y in range(0, H, 6):
            d.rectangle((0, y, W, y + 2), fill=rgb(hsv(hue + 0.05, 0.2, 0.13)))
        # cassette shell
        x0, y0, x1, y1 = 120, 70, W - 120, 470
        d.rounded_rectangle((x0, y0, x1, y1), radius=26, fill=rgb(hsv(hue + 0.1, 0.25, 0.18)), outline=rgb(hsv(hue, 0.2, 0.35)), width=4)
        # label
        lab = rgb(hsv(hue + 0.12, 0.35, 0.95))
        d.rounded_rectangle((x0 + 34, y0 + 30, x1 - 34, y0 + 230), radius=14, fill=lab)
        stripe = [hsv(hue + 0.92, 0.75, 0.95), hsv(hue + 0.02, 0.75, 1.0), hsv(hue + 0.10, 0.7, 1.0)]
        for j, c in enumerate(stripe):
            d.rectangle((x0 + 34, y0 + 150 + j * 18, x1 - 34, y0 + 166 + j * 18), fill=rgb(c))
        f = font("Oswald.ttf", 54)
        d.text(((x0 + x1) / 2, y0 + 82), self.title, font=f, anchor="mm", fill=rgb(hsv(hue + 0.75, 0.6, 0.25)))
        d.text((x0 + 58, y0 + 48), "A", font=font("Oswald.ttf", 40), anchor="lm", fill=rgb(hsv(hue + 0.75, 0.6, 0.25)))
        d.text((x1 - 58, y0 + 48), f"{int(self.F.ev.get('bpm', 120))} BPM", font=font("Inter-SemiBold.otf", 22),
               anchor="rm", fill=rgb(hsv(hue + 0.75, 0.6, 0.30)))
        # window + reels
        wy = y0 + 250
        d.rounded_rectangle((x0 + 150, wy, x1 - 150, wy + 120), radius=50, fill=(18, 14, 20))
        ang = (t - self.t0) * 2 * math.pi / (self.spb * 2)
        prog = (t - self.t0) / max(1e-3, self.t1 - self.t0)
        for side, rad in ((-1, 54 - 18 * prog), (1, 36 + 18 * prog)):
            cx = (x0 + x1) / 2 + side * 165
            cy = wy + 60
            d.ellipse((cx - rad - 8, cy - rad - 8, cx + rad + 8, cy + rad + 8), fill=(55, 38, 30))     # tape pack
            d.ellipse((cx - 30, cy - 30, cx + 30, cy + 30), fill=(235, 232, 225))
            for s in range(6):
                a = ang * side + s * math.pi / 3
                d.line((cx, cy, cx + math.cos(a) * 26, cy + math.sin(a) * 26), fill=(40, 40, 46), width=6)
            d.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill=(40, 40, 46))
        # screws
        for sx, sy in ((x0 + 22, y0 + 22), (x1 - 22, y0 + 22), (x0 + 22, y1 - 22), (x1 - 22, y1 - 22)):
            d.ellipse((sx - 7, sy - 7, sx + 7, sy + 7), fill=(120, 120, 128))
        # VU meters
        for k_, cx in enumerate((W / 2 - 200, W / 2 + 200)):
            cy = 690
            d.rounded_rectangle((cx - 170, 520, cx + 170, 735), radius=14, fill=rgb(hsv(0.12, 0.35, 0.92)))
            for j in range(11):
                a = math.radians(-60 + j * 12)
                r0, r1 = 120, 140 if j % 2 == 0 else 132
                col = (190, 40, 40) if j >= 8 else (40, 36, 40)
                d.line((cx + math.sin(a) * r0, cy - math.cos(a) * r0, cx + math.sin(a) * r1, cy - math.cos(a) * r1), fill=col, width=4)
            d.text((cx, 610), "VU", font=font("Oswald.ttf", 30), anchor="mm", fill=(40, 36, 40))
            lvl = min(1.0, 0.25 + 0.45 * en + 0.35 * (kick if k_ == 0 else max(kick * 0.8, snare)))
            a = math.radians(-60 + 120 * lvl)
            d.line((cx, cy, cx + math.sin(a) * 150, cy - math.cos(a) * 150), fill=(20, 18, 22), width=5)
            d.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=(30, 28, 32))
            # peak LED
            on = lvl > 0.85
            d.ellipse((cx + 130, 540, cx + 152, 562), fill=(255, 60, 60) if on else (90, 30, 30))
        # tape counter
        cnt = f"{int((t - self.t0) * 10) % 1000:03d}"
        d.rounded_rectangle((W / 2 - 62, 560, W / 2 + 62, 620), radius=8, fill=(14, 12, 16))
        d.text((W / 2, 590), cnt, font=font("Oswald.ttf", 42), anchor="mm", fill=rgb(hsv(hue + 0.45, 0.6, 1.0)))
        arr = to_arr(img)
        return arr * (1.0 + 0.05 * kick)


# ---------------------------------------------------------------- synth keys with falling notes
class Keys(HiInsert):
    name = "keys"
    FALL = 1.6          # seconds a note bar travels before it hits its key

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        ev = self.F.ev
        self.lead = [n for n in ev.get("lead", []) if self.t0 - 2 <= n["t"] < self.t1 + 2]
        self.bass = [n for n in ev.get("bass", []) if self.t0 - 2 <= n["t"] < self.t1 + 2]
        ps = [n["p"] for n in self.lead] or [60]
        top = int(max(ps)) + 1
        lo = min(int(min(ps)), top - 24)
        self.lo = lo - (lo % 12)                    # start on a C
        self.hi = max(top, self.lo + 24)
        whites = [p for p in range(self.lo, self.hi + 1) if p % 12 in (0, 2, 4, 5, 7, 9, 11)]
        self.whites = whites
        self.ww = W / len(whites)
        self.KY = 560

    def key_x(self, p):
        """Centre x and width of a key for pitch p (bass notes folded into the keyboard range)."""
        while p < self.lo:
            p += 12
        while p > self.hi:
            p -= 12
        p = int(round(p))
        if p % 12 in (1, 3, 6, 8, 10):
            j = self.whites.index(p - 1) if (p - 1) in self.whites else 0
            return (j + 1) * self.ww, self.ww * 0.6, True
        j = self.whites.index(p) if p in self.whites else 0
        return (j + 0.5) * self.ww, self.ww, False

    def draw_hi(self, t):
        F = self.F
        kick = F.pulse("kick", t, 0.12)
        hue = self.cfg.hue
        img = Image.new("RGB", (W, H), rgb(hsv(hue + 0.7, 0.6, 0.06)))
        d = ImageDraw.Draw(img, "RGBA")
        # beat lines in the waterfall
        b0 = math.floor((t - self.FALL) / self.spb)
        for b in range(b0, b0 + int(self.FALL / self.spb) + 3):
            tb = b * self.spb
            y = self.KY - (tb - t) / self.FALL * self.KY
            if 0 <= y <= self.KY:
                d.line((0, y, W, y), fill=rgb(hsv(hue + 0.7, 0.4, 0.22 if b % 4 == 0 else 0.12)), width=2)
        # falling bars
        active = {}
        for lane, notes, col_f in (("bass", self.bass, lambda p: hsv(hue + 0.55, 0.7, 0.9)),
                                   ("lead", self.lead, lambda p: pitch_color(p, hue, 0.65, 1.0))):
            for n in notes:
                y_hit = self.KY - (n["t"] - t) / self.FALL * self.KY
                y_end = self.KY - (n["t"] + n["dur"] - t) / self.FALL * self.KY
                if y_hit < 0 or y_end > self.KY:
                    continue
                cx, kw, black = self.key_x(n["p"])
                c = col_f(n["p"])
                w2 = kw * (0.36 if lane == "bass" else 0.42)
                d.rounded_rectangle((cx - w2, max(0, y_end), cx + w2, min(self.KY, y_hit)), radius=int(w2),
                                    fill=rgb(c, 0.95), outline=rgb(c, 1.4), width=2)
                if n["t"] <= t < n["t"] + max(0.12, n["dur"]):
                    active[int(round(n["p"]))] = c
        # glow line at the keys
        d.rectangle((0, self.KY - 4, W, self.KY), fill=rgb(hsv(hue + 0.5, 0.5, 0.6 + 0.4 * kick)))
        # keyboard
        def lit_col(p):
            for q, c in active.items():
                qq = q
                while qq < self.lo:
                    qq += 12
                while qq > self.hi:
                    qq -= 12
                if qq == p:
                    return c
            return None
        for j, p in enumerate(self.whites):
            x = j * self.ww
            c = lit_col(p)
            fill = rgb(c, 1.0) if c is not None else (236, 234, 240)
            d.rounded_rectangle((x + 2, self.KY + 4, x + self.ww - 2, H - 6), radius=8, fill=fill)
        for p in range(self.lo, self.hi + 1):
            if p % 12 in (1, 3, 6, 8, 10):
                cx, kw, _ = self.key_x(p)
                c = lit_col(p)
                fill = rgb(c, 0.85) if c is not None else (22, 20, 26)
                d.rounded_rectangle((cx - kw / 2, self.KY + 4, cx + kw / 2, self.KY + 4 + (H - self.KY) * 0.6), radius=6, fill=fill)
        arr = to_arr(img)
        # soft glow above lit keys
        for q, c in active.items():
            cx, kw, _ = self.key_x(q)
            xa, xb = int(max(0, cx - kw)), int(min(W, cx + kw))
            ys = np.arange(self.KY - 120, self.KY)
            g = np.exp(-((self.KY - ys) / 60.0) ** 2)[:, None, None] * 0.5
            arr[self.KY - 120:self.KY, xa:xb] += g * np.asarray(c, np.float32)[None, None]
        return arr


HI_INSERTS = {c.name: c for c in (Chat, Tape, Keys)}
