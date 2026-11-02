"""Kinetic text layer (drawn on the final 8-bit frame, crisp, after post FX).

  * TITLE   letters of the track title appear on the first notes / hats, then the
            title glides into a small corner tag that pulses with the music;
  * COUNTDOWN 'DROP IN 4-3-2-1' over the build (and before drop 2): digits roll
            on the bar line, a thin bar fills beat by beat; it zooms out with the dive;
  * CHORDS  chord names ride the note ribbon and stick to its left edge;
  * OUTRO   a short, human line instead of a SUBSCRIBE banner (see OUTRO_LINES):
            words land on beats, then the frame melts into frame 0 so the short loops.
"""
from __future__ import annotations

import colorsys
import math

import numpy as np
from PIL import Image

from .overlay import _font, _text_sprite, blit, roman_of


def _rgb(h, s=0.6, v=1.0):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _ease_back(x, s=1.6):
    x = min(1.0, max(0.0, x)) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


# What the end says. Short, lowercase, gives a reason to comment or come back.
# Research notes: Shorts already show a native Subscribe button; a big banner at
# the end reads as begging and kills the loop. Questions drive comments, a
# promise ('tomorrow') drives follows, and a seamless loop drives rewatches.
OUTRO_LINES = {
    "question": (["drop 1", "or", "drop 2?"], "tell me in the comments"),
    "rate":     (["rate", "this", "drop"], "1 to 10, comments"),
    "tomorrow": (["new track", "tomorrow"], "follow so you don't miss it"),
    "replay":   (["wait for", "the switch"], "watch it again"),
    "loop":     ([], ""),
}


class Captions:
    def __init__(self, F, W, H, title, plan_hue, ribbon=None, outro="question", subtitle=""):
        self.F, self.W, self.H = F, W, H
        self.title = title.upper()
        self.subtitle = subtitle
        self.ribbon = ribbon
        self.hue = plan_hue
        self.cache = {}
        self.f_title = _font("Oswald.ttf", 118)
        self.f_sub = _font("Inter-SemiBold.otf", 30)
        self.f_count = _font("Oswald.ttf", 170)
        self.f_lbl = _font("Inter-SemiBold.otf", 34)
        self.f_chord = _font("Oswald.ttf", 46)
        self.f_roman = _font("Inter-SemiBold.otf", 22)
        self.f_word = _font("Oswald.ttf", 132)
        self.f_small = _font("Inter-SemiBold.otf", 34)
        spb = F.ev["spb"]
        self.spb = spb
        # title letters land on early note/hat onsets
        evs = sorted([n["t"] for k in ("arp", "lead", "counter") for n in F.ev.get(k, [])] + list(F.hats))
        ts, last = [], 0.12
        for e in evs:
            if e >= last + 0.07:
                ts.append(e); last = e
            if len(ts) >= len(self.title):
                break
        while len(ts) < len(self.title):
            ts.append(last + 0.08 * (len(ts) + 1))
        self.letter_t = ts
        secs = F.sections
        first_drop = next((s for s in secs if s["name"] == "drop"), None)
        self.tag_t = first_drop["t"] - 4 * 4 * spb if first_drop else 4.0      # title -> corner at build start
        self.tag_t = max(self.tag_t, ts[-1] + 1.2)
        # countdowns: last 4 bars before each drop
        self.countdowns = []
        for s in secs:
            if s["name"] in ("drop", "drop2"):
                n_bars = 4
                self.countdowns.append((s["t"] - n_bars * 4 * spb, s["t"], "DROP" if s["name"] == "drop" else "DROP 2"))
        out = secs[-1]
        self.outro_t = out["t"]
        self.end_t = F.dur
        words, small = OUTRO_LINES.get(outro, OUTRO_LINES["question"])
        self.outro = outro
        self.words, self.small = words, small
        # words land on beats in the second half of the outro (first half = the world breathes)
        start = out["t"] + max(0.35, round((out["dur"] / 2 - 1.0) / spb) * spb)
        beats = [start + k * spb for k in range(len(words))]
        self.word_t = beats

    def _spr(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def _scaled(self, key, spr, s):
        q = round(s * 40) / 40
        if abs(q - 1) < 1e-3:
            return spr
        k = (key, q)
        if k not in self.cache:
            im = Image.fromarray((np.clip(spr, 0, 1) * 255).astype(np.uint8), "RGBA")
            w, h = max(1, int(im.width * q)), max(1, int(im.height * q))
            self.cache[k] = np.asarray(im.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255
        return self.cache[k]

    # ------------------------------------------------------------------
    def draw(self, frame, i, t, hue, ui_alpha=1.0):
        F = self.F
        W = self.W
        glow = _rgb(hue, 0.75, 1.0)
        glow2 = _rgb(hue + 0.5, 0.6, 1.0)
        lvl = float(F.level[min(i, len(F.level) - 1)])
        # ---------------- title: typed on notes, then glides to the corner tag
        n_vis = sum(1 for x in self.letter_t if x <= t)
        if n_vis and t < self.outro_t + 0.4:
            k = _ease_out((t - self.tag_t) / 0.7) if t >= self.tag_t else 0.0
            if k < 1:
                txt = self.title[:n_vis]
                spr = self._spr(("title", txt, glow), lambda: _text_sprite(txt, self.f_title, glow=glow, glow_r=16, spacing=6))
                pop = 1 + 0.06 * math.exp(-(t - self.letter_t[n_vis - 1]) / 0.08)
                s = pop * (1 - 0.68 * k)
                sp = self._scaled(("title", txt, glow), spr, s)
                full_w = self._spr(("titlew",), lambda: _text_sprite(self.title, self.f_title, glow_r=0, spacing=6).shape[1])
                # keep the text left edge fixed while typing (no re-centering jumps)
                x_left0 = (W - full_w) / 2
                x_left1 = 70 - 30 * 0.32
                xl = x_left0 + (x_left1 - x_left0) * k
                y = 330 + (232 - 330) * k
                blit(frame, sp, xl, y, ui_alpha * (1 - 0.0 * k), anchor="left")
                if self.subtitle and k < 0.5 and n_vis == len(self.title):
                    a = min(1.0, (t - self.letter_t[-1]) / 0.4) * (1 - 2 * k)
                    ss = self._spr(("sub",), lambda: _text_sprite(self.subtitle, self.f_sub, fill=(220, 225, 255),
                                                                  glow=glow2, glow_r=6, spacing=3))
                    blit(frame, ss, W / 2, 420, a * ui_alpha)
            else:
                tag = self._spr(("tag", glow), lambda: _text_sprite(self.title, self.f_title, glow=glow, glow_r=16, spacing=6))
                sp = self._scaled(("tag", glow), tag, 0.32)
                blit(frame, sp, 70 - 30 * 0.32, 232, ui_alpha * 0.95, anchor="left")
                # live level underline
                x0 = 72; w = int(sp.shape[1] - 30 * 0.32 * 2 - 10)
                ww = int(w * min(1.0, 0.15 + 0.85 * lvl))
                frame[262:265, x0:x0 + ww] = np.maximum(frame[262:265, x0:x0 + ww],
                                                        np.array(glow2, np.uint8)[None, None, :])
        # ---------------- countdown to the drop
        for a, b, label in self.countdowns:
            if a - 0.2 <= t < b + 0.35:
                self._countdown(frame, t, a, b, label, glow, glow2, ui_alpha)
        # ---------------- chord labels riding the ribbon
        if self.ribbon is not None and ui_alpha > 0.02:
            tonic = F.ev["tonic"]
            for x, ch, state in self.ribbon.chord_labels(t):
                name = ch["name"]
                sp = self._spr(("ch", name, state == 1, glow), lambda: _text_sprite(
                    name, self.f_chord, fill=(255, 255, 255) if state == 1 else (190, 195, 230),
                    glow=glow, glow_r=10 if state == 1 else 4))
                fade = min(1.0, (self.ribbon.XR - x) / 120.0, max(0.0, (x + 100) / 120.0))
                a_ = (1.0 if state == 1 else 0.45) * fade * ui_alpha
                blit(frame, sp, x - 30, self.ribbon.LABEL_Y, a_, anchor="left")
                if state == 1:
                    rn = roman_of(ch, tonic)
                    rs = self._spr(("rn", rn, glow), lambda: _text_sprite(rn, self.f_roman, fill=(210, 215, 255), glow=glow, glow_r=5))
                    blit(frame, rs, x - 30 + sp.shape[1] - 50, self.ribbon.LABEL_Y + 6, a_ * 0.8, anchor="left")
        # ---------------- outro line
        if self.words and t >= self.outro_t:
            self._outro(frame, t, glow, glow2)

    def _countdown(self, frame, t, a, b, label, glow, glow2, ui_alpha):
        W = self.W
        bar = 4 * self.spb
        intro = _ease_out((t - a + 0.2) / 0.35)
        out = 1.0 - _ease_out((t - b) / 0.3) if t >= b - 1e-6 else 1.0
        # dive: grows and fades as the camera dives into the drop
        dive = max(0.0, (t - (b - 0.35)) / 0.35) if t >= b - 0.35 else 0.0
        alpha = ui_alpha * intro * out * (1 - 0.6 * dive)
        if alpha <= 0.01:
            return
        lab = self._spr(("cdl", label, glow2), lambda: _text_sprite(label + " IN", self.f_lbl, fill=(230, 232, 255),
                                                                    glow=glow2, glow_r=6, spacing=10))
        blit(frame, lab, W / 2, 470, alpha)
        left = b - t
        n = max(1, min(4, int(math.ceil(left / bar - 1e-6))))
        # roll: old digit slides up and out, new digit rises in (0.18 s)
        roll = _ease_out((bar - (left - (n - 1) * bar)) / 0.18) if left > 0 else 1.0
        y_new = 600 + 60 * (1 - roll)
        sc = 1 + 0.9 * dive
        d = self._spr(("cd", n, glow), lambda: _text_sprite(str(n), self.f_count, glow=glow, glow_r=22))
        blit(frame, self._scaled(("cd", n, glow), d, sc), W / 2, y_new, alpha * roll)
        if n < 4 and roll < 1:
            o = self._spr(("cd", n + 1, glow), lambda: _text_sprite(str(n + 1), self.f_count, glow=glow, glow_r=22))
            blit(frame, o, W / 2, 600 - 70 * roll, alpha * (1 - roll))
        # beat bar: fills during the current bar
        frac = 1 - (left - (n - 1) * bar) / bar
        bw = 260
        x0 = int(W / 2 - bw / 2); y0 = 700
        seg = bw / 4
        for k in range(4):
            on = frac >= (k + 1) / 4 - 1e-3 or frac >= k / 4
            fill = min(1.0, max(0.0, (frac - k / 4) * 4))
            xa = int(x0 + k * seg + 4); xb = int(x0 + (k + 1) * seg - 4)
            col = np.array(glow, np.float32) if on else np.array((90, 90, 120), np.float32)
            reg = frame[y0:y0 + 8, xa:xb].astype(np.float32)
            frame[y0:y0 + 8, xa:xb] = (reg * (1 - alpha * (0.35 + 0.65 * fill)) + col * alpha * (0.35 + 0.65 * fill)).astype(np.uint8)

    def _outro(self, frame, t, glow, glow2):
        W = self.W
        fade_out = 1.0 - _ease_out((t - (self.end_t - 0.55)) / 0.45)
        y0 = 640
        for j, (w, tw) in enumerate(zip(self.words, self.word_t)):
            if t < tw:
                continue
            k = _ease_back((t - tw) / 0.28)
            sp = self._spr(("w", w, glow), lambda: _text_sprite(w, self.f_word, glow=glow, glow_r=18, spacing=2))
            y = y0 + j * 150 - 40 * (1 - k)
            blit(frame, self._scaled(("w", w, glow), sp, 0.9 + 0.1 * k), W / 2, y, min(1.0, (t - tw) / 0.12) * fade_out)
        if self.small and self.word_t and t >= self.word_t[-1] + self.spb:
            a = min(1.0, (t - self.word_t[-1] - self.spb) / 0.3) * fade_out
            ss = self._spr(("small", glow2), lambda: _text_sprite(self.small, self.f_small, fill=(225, 228, 255),
                                                                  glow=glow2, glow_r=6, spacing=2))
            blit(frame, ss, W / 2, y0 + len(self.words) * 150 - 10, a)
