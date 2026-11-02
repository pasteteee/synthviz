"""Text / UI overlay: title, chord name + roman numeral, progression strip and
a mini piano showing the live chord voicing and melody note.
Layout respects YouTube Shorts safe zones (UI covers ~top 120px, bottom ~420px,
right ~140px)."""
from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"


def _font(name, size):
    return ImageFont.truetype(str(FONT_DIR / name), size)


def _rgb(h, s=0.6, v=1.0):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _text_sprite(text, font, fill=(255, 255, 255), glow=(255, 80, 200), glow_r=10, spacing=0, pad=30):
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    if spacing:
        widths = [tmp.textlength(ch, font=font) for ch in text]
        w = int(sum(widths) + spacing * (len(text) - 1)) + 2 * pad
    else:
        w = int(tmp.textlength(text, font=font)) + 2 * pad
    asc, desc = font.getmetrics()
    h = asc + desc + 2 * pad
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)
    if spacing:
        x = pad
        for ch, cw in zip(text, widths):
            d.text((x, pad), ch, font=font, fill=fill + (255,))
            x += cw + spacing
    else:
        d.text((pad, pad), text, font=font, fill=fill + (255,))
    if glow_r:
        g = Image.new("RGBA", (w, h), glow + (0,))
        a = base.split()[3].filter(ImageFilter.GaussianBlur(glow_r))
        g.putalpha(a.point(lambda v: min(255, int(v * 1.6))))
        out = Image.alpha_composite(g, base)
    else:
        out = base
    arr = np.asarray(out).astype(np.float32) / 255.0
    return arr


def blit(frame, spr, cx, cy, alpha=1.0, anchor="center"):
    h, w = spr.shape[:2]
    H, W = frame.shape[:2]
    x0 = int(cx - w / 2) if anchor == "center" else int(cx)
    y0 = int(cy - h / 2)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    s = spr[ya - y0: yb - y0, xa - x0: xb - x0]
    a = s[..., 3:4] * alpha
    reg = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (reg * (1 - a) + s[..., :3] * 255 * a).astype(np.uint8)


ROMAN_BY_SEMI = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV", 6: "#IV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII"}


def roman_of(chord, tonic):
    semi = (chord["root"] - tonic) % 12
    rn = ROMAN_BY_SEMI[semi]
    q = chord["name"]
    minor = len(chord["intervals"]) > 1 and chord["intervals"][1] == 3
    if minor:
        rn = rn.lower() if not rn.startswith("b") and not rn.startswith("#") else rn[0] + rn[1:].lower()
    return rn


class Overlay:
    """Crisp text drawn after post-processing: current chord + roman numeral,
    and the finale's small caption."""

    def __init__(self, F, W=1080, H=1920, caption=None):
        self.F, self.W, self.H = F, W, H
        self.f_chord = _font("Oswald.ttf", 78)
        self.f_roman = _font("Inter-SemiBold.otf", 30)
        self.f_cap = _font("Inter-SemiBold.otf", 34)
        from .hud import HUD
        self.chord_xy = ((HUD.CX0 + HUD.COL) / 2, HUD.CY0 + 88)
        self.roman_xy = ((HUD.CX0 + HUD.COL) / 2, HUD.CY0 + 158)
        self.cache = {}
        self.tonic = F.ev["tonic"]
        self.caption = caption

    def _spr(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def draw(self, frame, i, t, hue, hud_alpha=1.0, cta_p=0.0):
        F = self.F
        W = self.W
        idx, ch, since = F.chord(t)
        glow = _rgb(hue, 0.7, 1.0)
        if hud_alpha > 0.02:
            name = ch["name"]
            rn = roman_of(ch, self.tonic)
            spr = self._spr(("ch", name, glow), lambda: _text_sprite(name, self.f_chord, glow=glow, glow_r=14))
            rs = self._spr(("rn", rn, glow), lambda: _text_sprite(rn, self.f_roman, fill=(220, 220, 255), glow=glow, glow_r=8))
            a = min(1.0, since / 0.10, t / 0.5) * hud_alpha
            pop = 1.0 - min(1.0, since / 0.12)
            blit(frame, spr, self.chord_xy[0], self.chord_xy[1] - 6 * pop, a)
            blit(frame, rs, self.roman_xy[0], self.roman_xy[1], a * 0.8)
        if self.caption and cta_p > 0.75:
            cap = self._spr(("cap",), lambda: _text_sprite(self.caption, self.f_cap, fill=(235, 235, 255),
                                                            glow=_rgb(hue, 0.6, 1.0), glow_r=10, spacing=7))
            blit(frame, cap, W / 2, self.H * 0.585, min(1.0, (cta_p - 0.75) / 0.25))
