"""Arcade inserts: little games that play the music.

Each game is a pure function of time, precomputed from the song's events so
that every jump, bounce, brick and explosion lands exactly on a note or drum hit:

  runner    original hero "Bit" (a small robot in headphones) jumps from note to note;
            platforms are the melody (height = pitch), coins are hi-hats.
  bounce    the "satisfying square": every lead note is a bounce off a wall that
            appears exactly where the square hits it.
  breakout  the paddle hits on kicks, every melody note breaks a brick.
  shooter   the ship fires so that each enemy explodes exactly on a note,
            the formation flinches on snares.

All games share InsertCfg so they can be tuned without touching code:
palette hue, speed, how thin the note stream is, hero colours, which lane drives it.
ArcadeLayer composites a game into the frame as a CRT screen that powers on/off
on bar lines."""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

from . import kernels2 as K2
from .pixelkit import Canvas, make_palette, pitch_color, sprite, smoothstep, onsets, hsv

CW, CH, SCALE = 188, 150, 5


@dataclass
class InsertCfg:
    hue: float = 0.0             # palette base (taken from the video's world)
    speed: float = 1.0           # tempo of motion (1 = default)
    lane: str = "lead"           # which note line drives the game: lead | counter | arp | bass
    min_gap: float = 0.0         # thin notes closer than this (0 = auto from tempo)
    hero_hue: float | None = None
    title: str = ""              # "LEVEL 1" style card at start ('' = auto)
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- sprites (original designs)
BIT_RUN = [sprite(r) for r in (
    ["....aa....", "....a.....", "..bbbbbb..", ".hbccccbh.", ".hbcdcdbh.", ".hbccccbh.", "..bbbbbb..",
     "...bbbb...", "..bbbbbb..", "..bb..bb..", "..b....b..", ".bb....bb."],
    ["....aa....", "....a.....", "..bbbbbb..", ".hbccccbh.", ".hbcdcdbh.", ".hbccccbh.", "..bbbbbb..",
     "...bbbb...", "..bbbbbb..", "...bbbb...", "...b..b...", "..bb..bb.."],
)]
BIT_JUMP = sprite(["....aa....", "....a.....", "..bbbbbb..", ".hbccccbh.", ".hbcdcdbh.", ".hbccccbh.",
                   "b.bbbbbb.b", ".bbbbbbbb.", "..bbbbbb..", "..b....b..", ".b......b.", ".........."])
SHIP = sprite(["....a....", "...aaa...", "...aba...", "..aabaa..", ".aaaaaaa.", "aa.aaa.aa", "a...c...a"])
BLOB = [sprite(r) for r in (
    ["..aaaa..", ".aaaaaa.", "aabaabaa", "aaaaaaaa", ".a.aa.a.", "a......a"],
    ["..aaaa..", ".aaaaaa.", "aabaabaa", "aaaaaaaa", "a.a..a.a", ".a....a."],
)]
COIN = [sprite(r) for r in ([".aa.", "abba", "abba", ".aa."], [".a.", "aba", "aba", ".a."], ["a", "b", "b", "a"])]


def _lane_notes(F, cfg):
    ev = F.ev
    notes = list(ev.get(cfg.lane, []) or [])
    if cfg.lane == "lead" and ev.get("counter"):
        notes += ev["counter"]
    return notes


class Insert:
    name = "base"

    def __init__(self, F, t0, t1, cfg: InsertCfg, seed=0):
        self.F, self.t0, self.t1, self.cfg = F, t0, t1, cfg
        self.pal = make_palette(cfg.hue)
        if cfg.hero_hue is not None:
            self.pal["hero2"] = hsv(cfg.hero_hue, 0.8, 1.0)
        self.seed = seed
        self.spb = F.ev["spb"]
        gap = cfg.min_gap or max(0.11, self.spb / 4 * 0.95)
        self.notes = onsets(_lane_notes(F, cfg), t0 - 0.05, t1 + 2.0, gap)
        if len(self.notes) < 4:  # fall back to kicks so a game never stands still
            self.notes = [dict(t=k, p=60 + (i * 5) % 12, dur=0.2) for i, k in enumerate(F.kicks) if t0 <= k < t1 + 2]
        ps = [n["p"] for n in self.notes] or [60]
        self.p_lo, self.p_hi = min(ps), max(ps) + 1e-3
        self.nt = [n["t"] for n in self.notes]

    def pnorm(self, p):
        return (p - self.p_lo) / max(1.0, self.p_hi - self.p_lo)

    def hits_before(self, times, t):
        return bisect.bisect_right(times, t)

    def title_card(self, cv, t, text):
        age = t - self.t0
        if 0.15 < age < 1.6 and int(age * 6) % 2 == 0 or 0.15 < age < 0.9:
            w = len(text) * 8
            cv.rect((CW - w) / 2 - 4, 54, (CW + w) / 2 + 2, 70, self.pal["dark"])
            cv.text(text, (CW - w) / 2, 57, self.pal["white"], 2)

    def score(self, cv, n):
        cv.text(f"{n:05d}", 4, 4, self.pal["white"])


# ---------------------------------------------------------------- 1. runner
class Runner(Insert):
    name = "runner"
    HX = 56          # hero x on screen
    GROUND = 128

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.v = 64.0 * self.cfg.speed
        self.ys = [self.GROUND - 14 - self.pnorm(n["p"]) * 44 for n in self.notes]
        self.xs = [self.v * (n["t"] - self.t0) for n in self.notes]
        self.ws = [float(np.clip(self.v * n.get("dur", 0.2) * 0.9, 8, 26)) for n in self.notes]
        hats = [h for h in self.F.hats if self.t0 <= h < self.t1 + 2]
        self.coins = [h for i, h in enumerate(hats) if i % 2 == 0 and
                      all(abs(h - x) > 0.06 for x in self.nt[max(0, bisect.bisect_left(self.nt, h) - 1):][:2])]
        self.stars = np.random.default_rng(self.seed).uniform(0, 1, (40, 3))

    def hero_y(self, t):
        i = bisect.bisect_right(self.nt, t) - 1
        if i < 0:
            # run along the ground, then hop onto the first note
            if not self.nt:
                return self.GROUND, False
            dt = self.nt[0] - self.t0 + 0.4
            s = (t - (self.nt[0] - dt)) / dt
            if s < 0.5:
                return self.GROUND, False
            s = (s - 0.5) * 2
            return self.GROUND + (self.ys[0] - self.GROUND) * s - 4 * 26 * s * (1 - s), True
        if i >= len(self.nt) - 1:
            return self.ys[-1], False
        dt = self.nt[i + 1] - self.nt[i]
        s = (t - self.nt[i]) / dt
        h = float(np.clip(dt * 28, 4, 16))
        return self.ys[i] + (self.ys[i + 1] - self.ys[i]) * s - 4 * h * s * (1 - s), dt > 0.09

    def draw(self, cv, t):
        P = self.pal
        F = self.F
        cam = self.v * (t - self.t0) - self.HX
        kick = F.pulse("kick", t, 0.12)
        hat = F.pulse("hat", t, 0.06)
        cv.vgrad(P["sky0"], P["sky1"], 0, self.GROUND)
        # stars twinkle on hats
        for sx, sy, ph in self.stars:
            x = (sx * CW * 2 - cam * 0.05) % CW
            b = 0.35 + 0.5 * hat * (ph > 0.5)
            cv.px(x, sy * self.GROUND * 0.6, P["white"] * b)
        # pixel moon
        mx, my = CW * 0.78, 26
        for dy in range(-9, 10):
            w = int(math.sqrt(max(0, 81 - dy * dy)))
            cv.rect(mx - w, my + dy, mx + w, my + dy + 1, P["accent"] * (0.9 + 0.3 * kick))
        # parallax hills (breathe on kick)
        xs = np.arange(CW)
        far = (self.GROUND - 34 - 10 * np.sin((xs + cam * 0.2) * 0.045) - 6 * np.sin((xs + cam * 0.2) * 0.11)
               - 3 * kick).astype(int)
        near = (self.GROUND - 16 - 8 * np.sin((xs + cam * 0.5) * 0.07 + 1.3) - 2 * kick).astype(int)
        for x in range(CW):
            cv.a[far[x]:self.GROUND, x] = P["far"]
            cv.a[near[x]:self.GROUND, x] = P["near"]
        # ground tiles
        cv.rect(0, self.GROUND, CW, CH, P["ground"])
        off = int(cam) % 12
        for x in range(-off, CW, 12):
            cv.rect(x, self.GROUND, x + 11, self.GROUND + 1, P["ground_hi"])
            cv.rect(x + 6, self.GROUND + 6, x + 7, CH, P["dark"])
        # note platforms (pillars of bricks, top = pitch)
        score = 0
        for i, (n, x, y, w) in enumerate(zip(self.notes, self.xs, self.ys, self.ws)):
            sx = x - cam
            if sx < -30 or sx > CW + 30:
                if n["t"] <= t:
                    score += 1
                continue
            age = t - n["t"]
            lit = math.exp(-max(0.0, age) / 0.25) if age >= 0 else 0.0
            col = pitch_color(n["p"], self.cfg.hue)
            bump = -2 * math.exp(-max(0.0, age) / 0.06) if age >= 0 else 0.0
            cv.rect(sx - w / 2, y + bump, sx + w / 2, self.GROUND, col * 0.35)
            for yy in range(int(y + 4), self.GROUND, 4):
                cv.rect(sx - w / 2, yy, sx + w / 2, yy + 1, col * 0.2)
            cv.rect(sx - w / 2, y + bump, sx + w / 2, y + 3 + bump, col * (0.8 + 1.4 * lit))
            if age >= 0:
                score += 1
                cv.burst(sx, y, age, self.seed * 1000 + i, col, n=10, speed=40, life=0.35)
        # coins on hi-hats
        for j, h in enumerate(self.coins):
            x = self.v * (h - self.t0) - cam
            if x < -5 or x > CW + 5:
                if h <= t:
                    score += 1
                continue
            hy, _ = self.hero_y(h)
            cy = hy - 12
            if t < h:
                fr = COIN[int((t * 8 + j) % 3)]
                cv.blit(fr, x, cy + 2, {"a": P["coin"], "b": P["white"]})
            else:
                score += 1
                cv.burst(x, cy, t - h, 77 + j, P["coin"], n=6, speed=30, life=0.25, grav=-40)
        # hero
        y, air = self.hero_y(t)
        i = bisect.bisect_right(self.nt, t) - 1
        land = t - self.nt[i] if i >= 0 else 9
        sq = math.exp(-land / 0.07) if land < 0.3 else 0.0
        spr = BIT_JUMP if air and land > 0.05 else BIT_RUN[int(t * 10) % 2]
        cols = {"a": P["accent"], "b": P["hero"], "c": P["dark"], "d": P["accent2"] * (1 + 0.8 * hat), "h": P["hero2"]}
        cv.blit(spr, self.HX, y, cols, sx=2 * (1 + 0.35 * sq), sy=2 * (1 - 0.3 * sq))
        self.score(cv, score * 10)
        self.title_card(cv, t, self.cfg.title or "LEVEL 1")


# ---------------------------------------------------------------- 2. bounce
def _fold(p, lo, hi):
    """Reflect an unbounded coordinate into [lo, hi]; returns (pos, direction sign)."""
    L = hi - lo
    u = (p - lo) % (2 * L)
    return (lo + u, 1.0) if u <= L else (hi - (u - L), -1.0)


class Bounce(Insert):
    name = "bounce"
    SZ = 6
    LO, HI = (10, 12), (CW - 10, CH - 10)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        nt = [x for x in self.nt if self.t0 + 0.3 <= x < self.t1]
        self.ev = nt
        gaps = np.diff(nt) if len(nt) > 1 else np.array([0.4])
        g = float(np.percentile(np.clip(gaps, 0.05, 1.2), 85))
        self.v = float(np.clip(48.0 / g, 60, 220)) * self.cfg.speed
        lo = (self.LO[0] + self.SZ / 2, self.LO[1] + self.SZ / 2)
        hi = (self.HI[0] - self.SZ / 2, self.HI[1] - self.SZ / 2)
        self.lo, self.hi = lo, hi
        pos = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2])
        d = np.array([0.8, -0.6])
        self.seg = []      # (t_start, pos, dir)
        self.walls = []    # (t, x0, y0, x1, y1, pitch)
        tprev = self.t0
        rng = np.random.default_rng(self.seed)
        notes_by_t = {n["t"]: n for n in self.notes}
        for k_, te in enumerate(nt + [self.t1 + 3.0]):
            self.seg.append((tprev, pos.copy(), d.copy()))
            p_end, d_end = self._advance(pos, d, (te - tprev) * self.v)
            pos, d = p_end, d_end
            if k_ >= len(nt):
                break
            nxt = (nt[k_ + 1] if k_ + 1 < len(nt) else te + 0.5) - te
            best = None
            for axis in (0, 1):
                dd = d.copy(); dd[axis] *= -1
                # keep the run straight: no outer-wall folds during the next segment
                pe = pos + dd * nxt * self.v
                inside = lo[0] <= pe[0] <= hi[0] and lo[1] <= pe[1] <= hi[1]
                cdist = np.hypot(pe[0] - (lo[0] + hi[0]) / 2, pe[1] - (lo[1] + hi[1]) / 2)
                sc = (0 if inside else 1, cdist + rng.uniform(0, 8))
                if best is None or sc < best[0]:
                    best = (sc, axis, dd)
            _, axis, dd = best
            n = notes_by_t.get(te, {"p": 60})
            s = self.SZ / 2 + 0.5
            if axis == 0:
                wx = pos[0] + np.sign(d[0]) * s
                self.walls.append((te, wx, pos[1] - 7, wx + np.sign(d[0]) * 2, pos[1] + 7, n["p"]))
            else:
                wy = pos[1] + np.sign(d[1]) * s
                self.walls.append((te, pos[0] - 7, wy, pos[0] + 7, wy + np.sign(d[1]) * 2, n["p"]))
            d = dd
            tprev = te
        self.seg_t = [s_[0] for s_ in self.seg]

    def _advance(self, pos, d, L):
        x, sx = _fold(pos[0] + d[0] * L, self.lo[0], self.hi[0])
        y, sy = _fold(pos[1] + d[1] * L, self.lo[1], self.hi[1])
        return np.array([x, y]), np.array([abs(d[0]) * sx * np.sign(d[0] or 1), abs(d[1]) * sy * np.sign(d[1] or 1)])

    def pos(self, t):
        i = max(0, bisect.bisect_right(self.seg_t, t) - 1)
        t_s, p, d = self.seg[i]
        return self._advance(p, d, max(0.0, t - t_s) * self.v)[0]

    def draw(self, cv, t):
        P = self.pal
        F = self.F
        kick = F.pulse("kick", t, 0.12)
        cv.vgrad(P["sky0"] * 0.6, P["near"] * 0.6)
        # arena frame breathes on kick
        fc = P["accent2"] * (0.35 + 0.6 * kick)
        cv.rect(self.LO[0] - 2, self.LO[1] - 2, self.HI[0] + 2, self.LO[1], fc)
        cv.rect(self.LO[0] - 2, self.HI[1], self.HI[0] + 2, self.HI[1] + 2, fc)
        cv.rect(self.LO[0] - 2, self.LO[1], self.LO[0], self.HI[1], fc)
        cv.rect(self.HI[0], self.LO[1], self.HI[0] + 2, self.HI[1], fc)
        hits = 0
        for j, (tw, x0, y0, x1, y1, p) in enumerate(self.walls):
            age = t - tw
            if age < -0.35:
                continue
            col = pitch_color(p, self.cfg.hue)
            if age < 0:
                a = 0.15 * (1 + age / 0.35)
            else:
                hits += 1
                a = 0.45 + 1.6 * math.exp(-age / 0.18)
                cv.burst((x0 + x1) / 2, (y0 + y1) / 2, age, 500 + j, col, n=8, speed=35, life=0.3, grav=0)
            cv.rect(min(x0, x1), min(y0, y1), max(x0, x1) + 1, max(y0, y1) + 1, col * a)
        # trail
        for k in range(14, 0, -1):
            q = self.pos(t - k * 0.018)
            cv.rect(q[0] - 2, q[1] - 2, q[0] + 2, q[1] + 2, P["accent"] * (0.5 * (1 - k / 15)), add=True)
        q = self.pos(t)
        s = self.SZ / 2 * (1 + 0.25 * kick)
        cv.rect(q[0] - s, q[1] - s, q[0] + s, q[1] + s, P["white"])
        cv.rect(q[0] - s + 1, q[1] - s + 1, q[0] + s - 1, q[1] + s - 1, P["accent"] * 1.3)
        cv.text(f"{hits:03d}", CW / 2 - 6, 3, P["white"] * 0.8)


# ---------------------------------------------------------------- 3. breakout
class Breakout(Insert):
    name = "breakout"
    ROWS, COLS = 6, 12
    BW, BH, BY0 = 15, 5, 16
    PY = 136

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        F = self.F
        kicks = [x for x in F.kicks if self.t0 + 0.2 <= x < self.t1 + 1.0]
        if len(kicks) < 4:
            kicks = list(np.arange(self.t0 + 0.2, self.t1 + 1.0, self.spb))
        # paddle hits on kicks, but never faster than ~0.7 s per round trip
        step = max(1, int(round(0.7 / max(0.2, np.median(np.diff(kicks)) if len(kicks) > 1 else 0.5))))
        ph = kicks[::step]
        self.x0b = (CW - self.COLS * (self.BW + 1)) / 2
        alive = np.ones((self.ROWS, self.COLS), bool)
        self.kills = []       # (t, r, c)
        self.respawns = [self.t0 - 1]
        pts = []              # (t, x, y, kind)
        rng = np.random.default_rng(self.seed)
        last_x = CW / 2
        for a_, b_ in zip(ph[:-1], ph[1:]):
            mid = (a_ + b_) / 2
            cand = [n for n in self.notes if a_ + 0.25 * (b_ - a_) < n["t"] < b_ - 0.25 * (b_ - a_)]
            if not alive.any():
                alive[:] = True
                self.respawns.append(a_)
            if cand:
                n = min(cand, key=lambda n: abs(n["t"] - mid))
                c = (int(n["p"]) * 7) % self.COLS
                cols = [cc for cc in sorted(range(self.COLS), key=lambda cc: abs(cc - c)) if alive[:, cc].any()]
                c = cols[0]
                r = max(np.nonzero(alive[:, c])[0])
                alive[r, c] = False
                self.kills.append((n["t"], r, c, n["p"]))
                bx = self.x0b + c * (self.BW + 1) + self.BW / 2
                by = self.BY0 + r * (self.BH + 1) + self.BH + 2
                pts.append((n["t"], bx, by, 1))
            else:
                bx = float(np.clip(last_x + rng.uniform(-50, 50), 10, CW - 10))
                pts.append((mid, bx, 6, 2))
            last_x = bx
        # paddle hit x: between the brick before and after, so the angles look right
        out = []
        for j, tp in enumerate(ph):
            prev_x = pts[j - 1][1] if j - 1 >= 0 and j - 1 < len(pts) else CW / 2
            next_x = pts[j][1] if j < len(pts) else prev_x
            out.append((tp, float(np.clip((prev_x + next_x) / 2 + rng.uniform(-14, 14), 16, CW - 16)), self.PY - 3, 0))
        allp = sorted(out + pts)
        self.path_t = [p_[0] for p_ in allp]
        self.path = allp
        self.pad = [(p_[0], p_[1]) for p_ in out]
        self.pad_t = [p_[0] for p_ in out]

    def ball(self, t):
        i = bisect.bisect_right(self.path_t, t) - 1
        if i < 0:
            a = self.path[0]
            return a[1], a[2] - (a[0] - t) * 60
        if i >= len(self.path) - 1:
            return self.path[-1][1], self.path[-1][2]
        a, b = self.path[i], self.path[i + 1]
        s = (t - a[0]) / (b[0] - a[0])
        return a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s

    def paddle_x(self, t):
        i = bisect.bisect_right(self.pad_t, t) - 1
        if i < 0:
            return self.pad[0][1]
        if i >= len(self.pad) - 1:
            return self.pad[-1][1]
        (ta, xa), (tb, xb) = self.pad[i], self.pad[i + 1]
        return xa + (xb - xa) * smoothstep((t - ta) / (tb - ta) * 1.25)

    def draw(self, cv, t):
        P = self.pal
        F = self.F
        kick = F.pulse("kick", t, 0.1)
        snare = F.pulse("snare", t, 0.1)
        cv.vgrad(P["sky0"] * 0.7, P["near"] * 0.8)
        gen = bisect.bisect_right(self.respawns, t) - 1
        g0 = self.respawns[gen]
        g1 = self.respawns[gen + 1] if gen + 1 < len(self.respawns) else 1e9
        dead = {(r, c): tk for tk, r, c, _ in self.kills if g0 <= tk < g1}
        drop_in = smoothstep((t - g0) / 0.4) if gen > 0 else 1.0
        score = sum(1 for k_ in self.kills if k_[0] <= t)
        for r in range(self.ROWS):
            rc = hsv(self.cfg.hue + r / self.ROWS * 0.5, 0.7, 1.0)
            for c in range(self.COLS):
                x = self.x0b + c * (self.BW + 1)
                y = self.BY0 + r * (self.BH + 1) - (1 - drop_in) * (60 + r * 6)
                tk = dead.get((r, c))
                if tk is not None and t >= tk:
                    cv.burst(x + self.BW / 2, y + 2, t - tk, r * 31 + c + gen * 997, rc, n=14, speed=45, life=0.4)
                    continue
                cv.rect(x, y, x + self.BW, y + self.BH, rc * (0.55 + 0.25 * snare))
                cv.rect(x, y, x + self.BW, y + 1, rc * 1.2)
        # ball + trail
        for k in range(8, 0, -1):
            bx, by = self.ball(t - k * 0.015)
            cv.rect(bx - 1, by - 1, bx + 2, by + 2, P["accent"] * (0.4 * (1 - k / 9)), add=True)
        bx, by = self.ball(t)
        cv.rect(bx - 2, by - 2, bx + 2, by + 2, P["white"])
        # paddle squashes on the kick
        px = self.paddle_x(t)
        w = 26 + 6 * kick
        cv.rect(px - w / 2, self.PY - 1 * kick, px + w / 2, self.PY + 4, P["accent2"] * (0.8 + 0.8 * kick))
        cv.rect(px - w / 2, self.PY - 1 * kick, px + w / 2, self.PY, P["white"])
        self.score(cv, score * 50)
        self.title_card(cv, t, self.cfg.title or "STAGE 1")


# ---------------------------------------------------------------- 4. shooter
class Shooter(Insert):
    name = "shooter"
    ROWS, COLS = 4, 8
    SY = 140
    FLIGHT = 0.22

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        hits = [n for n in self.notes if self.t0 + 0.6 <= n["t"] < self.t1]
        alive = np.ones((self.ROWS, self.COLS), bool)
        self.waves = [self.t0 - 0.0]
        self.kills = []
        for n in hits:
            if not alive.any():
                alive[:] = True
                self.waves.append(n["t"] - 0.05)
            c = (int(n["p"]) * 5) % self.COLS
            cols = [cc for cc in sorted(range(self.COLS), key=lambda cc: abs(cc - c)) if alive[:, cc].any()]
            c = cols[0]
            r = max(np.nonzero(alive[:, c])[0])
            alive[r, c] = False
            self.kills.append((n["t"], r, c, n["p"]))
        self.fire = [(tk - self.FLIGHT, self.enemy_xy(tk, r, c)[0]) for tk, r, c, _ in self.kills]
        self.fire_t = [f[0] for f in self.fire]
        self.stars = np.random.default_rng(self.seed).uniform(0, 1, (70, 3))

    def formation(self, t):
        per = 8 * self.spb
        return 22 + 16 * math.sin(2 * math.pi * (t - self.t0) / per), 18

    def enemy_xy(self, t, r, c):
        fx, fy = self.formation(t)
        gen = bisect.bisect_right(self.waves, t) - 1
        g0 = self.waves[max(0, gen)]
        fly = 1 - smoothstep((t - g0) / 0.5) if gen > 0 else 1 - smoothstep((t - self.t0) / 0.6)
        return fx + c * 19, fy + r * 14 - fly * 90

    def ship_x(self, t):
        i = bisect.bisect_right(self.fire_t, t) - 1
        if i < 0:
            return self.fire[0][1] if self.fire else CW / 2
        if i >= len(self.fire) - 1:
            return self.fire[-1][1]
        (ta, xa), (tb, xb) = self.fire[i], self.fire[i + 1]
        return xa + (xb - xa) * smoothstep((t - ta) / max(1e-3, tb - ta) * 1.4)

    def draw(self, cv, t):
        P = self.pal
        F = self.F
        kick = F.pulse("kick", t, 0.12)
        snare = F.pulse("snare", t, 0.09)
        cv.vgrad(P["dark"], P["sky0"])
        sp = 40 + 50 * float(F.energy[min(len(F.energy) - 1, int(t * F.fps))])
        for sx, sy, z in self.stars:
            y = (sy * CH + t * sp * (0.3 + z)) % CH
            ln = 1 + 3 * kick * z
            cv.rect(sx * CW, y, sx * CW + 1, y + ln, P["white"] * (0.2 + 0.5 * z))
        gen = bisect.bisect_right(self.waves, t) - 1
        g0 = self.waves[max(0, gen)]
        g1 = self.waves[gen + 1] if gen + 1 < len(self.waves) else 1e9
        dead = {(r, c): (tk, p) for tk, r, c, p in self.kills if g0 <= tk < g1}
        frame = int(t / self.spb) % 2
        score = sum(1 for k_ in self.kills if k_[0] <= t)
        for r in range(self.ROWS):
            ec = hsv(self.cfg.hue + 0.3 + r * 0.12, 0.7, 1.0)
            for c in range(self.COLS):
                x, y = self.enemy_xy(t, r, c)
                d = dead.get((r, c))
                if d is not None and t >= d[0]:
                    cv.burst(x, y - 3, t - d[0], r * 17 + c + gen * 331, pitch_color(d[1], self.cfg.hue),
                             n=16, speed=50, life=0.45, grav=20)
                    continue
                flinch = -2 * snare
                cv.blit(BLOB[frame], x, y + flinch, {"a": ec * (0.8 + 0.6 * snare), "b": P["dark"]})
        # bullets
        for (tk, r, c, p) in self.kills:
            tf = tk - self.FLIGHT
            if tf <= t < tk:
                s = (t - tf) / self.FLIGHT
                ex, ey = self.enemy_xy(tk, r, c)
                by = self.SY - 8 + (ey - (self.SY - 8)) * s
                cv.rect(ex - 0.5, by, ex + 1.5, by + 5, pitch_color(p, self.cfg.hue) * 1.6)
        sx = self.ship_x(t)
        last_fire = t - self.fire_t[bisect.bisect_right(self.fire_t, t) - 1] if self.fire_t and t >= self.fire_t[0] else 9
        recoil = 2 * math.exp(-last_fire / 0.05)
        cv.blit(SHIP, sx, self.SY + recoil + 4, {"a": P["hero"], "b": P["accent"], "c": P["accent2"] * (1 + kick)}, sx=1.5, sy=1.5)
        self.score(cv, score * 100)
        self.title_card(cv, t, self.cfg.title or "WAVE 1")


# ---------------------------------------------------------------- 5. dancer
# Poses: absolute limb angles in degrees (0 = hanging down, 90 = out to the side, 180 = straight up),
# for upper arm / forearm / thigh / shin of each side, plus crouch (px), torso lean (px) and head tilt.
_P = lambda ua, fa, ub, fb, ta=8, sa=0, tb=8, sb=0, cr=0.0, lean=0.0, tilt=0.0: dict(
    uaL=ua, faL=fa, uaR=ub, faR=fb, thL=ta, shL=sa, thR=tb, shR=sb, cr=cr, lean=lean, tilt=tilt)
DANCE_MOVES = {
    # one key pose per half beat; the dancer snaps into each pose right on the hit
    "pump": [_P(150, 175, 150, 175, cr=0), _P(60, 150, 60, 150, 14, -10, 14, -10, cr=5)],
    "running_man": [_P(40, 110, -30, 60, 45, -20, -5, 0, cr=2), _P(-30, 60, 40, 110, -5, 0, 45, -20, cr=2),
                    _P(40, 110, -30, 60, 10, 5, 10, 5, cr=4), _P(-30, 60, 40, 110, 10, 5, 10, 5, cr=4)],
    "point": [_P(160, 165, 20, 40, 15, 0, 5, 0, lean=-3, tilt=-8), _P(20, 40, 20, 40, 10, 0, 10, 0, cr=3),
              _P(20, 40, 160, 165, 5, 0, 15, 0, lean=3, tilt=8), _P(20, 40, 20, 40, 10, 0, 10, 0, cr=3)],
    "wave": [_P(95, 140, 90, 50, cr=1, tilt=-5), _P(90, 50, 95, 140, cr=1, tilt=5)],
    "jumping_jack": [_P(160, 170, 160, 170, 30, 30, 30, 30, cr=-6), _P(30, 20, 30, 20, 6, 0, 6, 0, cr=4)],
    "shuffle": [_P(70, 120, 70, 120, 35, -30, 5, 0, cr=3, lean=-2), _P(70, 120, 70, 120, 5, 0, 35, -30, cr=3, lean=2)],
    "hands_up": [_P(170, 178, 170, 178, 10, 0, 10, 0, cr=0, tilt=-4), _P(150, 185, 150, 185, 14, -8, 14, -8, cr=5, tilt=4)],
}


class Dancer(Insert):
    """An original pixel robot ("Bit", the runner hero) dancing on a disco floor. Every half beat it snaps
    into a new pose, it squashes on kicks, the move changes every bar (big moves in the drops), the floor
    tiles light up with the melody notes and the crowd in front bounces on the kick."""
    name = "dancer"
    GROUND = 132

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        rng = np.random.default_rng(self.seed + 5)
        names = list(DANCE_MOVES)
        bars = int((self.t1 - self.t0) / (4 * self.spb)) + 3
        seq, prev = [], None
        for _ in range(bars):
            m = str(rng.choice([n for n in names if n != prev]))
            seq.append(m); prev = m
        self.seq = seq
        self.crowd = rng.uniform(0, 1, (16, 3))
        self.tiles = rng.integers(0, 1000, (6, 12))

    # ---- skeleton
    def pose(self, t):
        hb = self.spb / 2
        u = (t - self.t0) / hb
        i = int(math.floor(u)); ph = u - i
        bar = max(0, int((t - self.t0) / (4 * self.spb)))
        keys = DANCE_MOVES[self.seq[bar % len(self.seq)]]
        a, b = keys[(i - 1) % len(keys)], keys[i % len(keys)]
        e = smoothstep(min(1.0, ph * 3.5))          # snap in the first ~30 % of the half beat, then hold
        return {k_: a[k_] + (b[k_] - a[k_]) * e for k_ in a}

    @staticmethod
    def _limb(cv, x, y, ang, side, ln, w, col):
        r = math.radians(ang)
        dx, dy = math.sin(r) * side, math.cos(r)
        steps = int(ln * 1.5) + 1
        for j in range(steps + 1):
            px, py = x + dx * ln * j / steps, y + dy * ln * j / steps
            cv.rect(px - w / 2, py - w / 2, px + w / 2, py + w / 2, col)
        return x + dx * ln, y + dy * ln

    def draw_bit(self, cv, t, cx, P, S=1.45):
        F = self.F
        kick = F.pulse("kick", t, 0.10)
        hat = F.pulse("hat", t, 0.06)
        q = self.pose(t)
        sq = 3.0 * kick
        hip_y = self.GROUND - 27 * S + q["cr"] * S + sq
        hx = cx + q["lean"] * S
        body, dark, glow, trim = P["hero"], P["dark"], P["accent2"] * (1 + 0.7 * hat), P["hero2"]
        legc = body * 0.75
        R = lambda x0, y0, x1, y1, c, add=False: cv.rect(x0, y0, x1, y1, c, add)
        for side, th, sh in ((-1, q["thL"], q["shL"]), (1, q["thR"], q["shR"])):
            kx, ky = self._limb(cv, hx + 3 * S * side, hip_y, th, side, 13 * S, 4 * S, legc)
            fx, fy = self._limb(cv, kx, ky, th + sh, side, 13 * S, 4 * S, legc)
            R(fx - 3 * S + 2 * S * side, fy - S, fx + 3 * S + 2 * S * side, fy + 2 * S, trim)
        top = hip_y - 21 * S
        R(hx - 7 * S, top, hx + 7 * S, hip_y + S, body)
        R(hx - 7 * S, top + 13 * S, hx + 7 * S, top + 15 * S, trim)
        R(hx - 3 * S, top + 4 * S, hx + 3 * S, top + 10 * S, glow * (0.5 + 0.8 * kick))
        for side, ua, fa in ((-1, q["uaL"], q["faL"]), (1, q["uaR"], q["faR"])):
            ex, ey = self._limb(cv, hx + 7 * S * side, top + 2 * S, ua, side, 11 * S, 4 * S, body * 0.9)
            wx, wy = self._limb(cv, ex, ey, fa, side, 10 * S, 3 * S, body * 0.9)
            R(wx - 2 * S, wy - 2 * S, wx + 2 * S, wy + 2 * S, trim)
        tl = q["tilt"] * 0.12 * S
        hy = top - 2 * S
        h = lambda x0, y0, x1, y1, c: R(hx + x0 * S + tl, hy + y0 * S, hx + x1 * S + tl, hy + y1 * S, c)
        h(-8, -15, 8, 0, body)
        h(-6, -11, 6, -5, dark)
        h(-5, -9, -1, -7, glow)
        h(1, -9, 5, -7, glow)
        h(-10, -11, -7, -4, trim)
        h(7, -11, 10, -4, trim)
        h(-8, -18, 8, -15, trim)
        h(-1, -22, 1, -18, P["accent"])
        R(hx + tl - S, hy - 24 * S, hx + tl + S, hy - 22 * S, P["accent"] * (1 + 2 * kick), True)

    def draw(self, cv, t):
        P = self.pal
        F = self.F
        kick = F.pulse("kick", t, 0.12)
        snare = F.pulse("snare", t, 0.10)
        en = float(F.energy[min(len(F.energy) - 1, int(t * F.fps))])
        cv.vgrad(P["dark"], P["sky0"] * 1.6, 0, 90)
        # spotlights sweeping with the bar
        for j in range(3):
            ang = math.sin((t - self.t0) / (2 * self.spb) * math.pi + j * 2.1) * 0.5
            col = hsv(self.cfg.hue + 0.15 * j, 0.6, 1.0) * (0.06 + 0.10 * en + 0.10 * snare)
            x0 = 30 + 64 * j
            for y in range(0, self.GROUND, 2):
                w = 2 + y * 0.18
                xc = x0 + ang * y
                cv.rect(xc - w, y, xc + w, y + 2, col, add=True)
        # disco floor in perspective: tiles flash with the melody notes
        hor = 84
        i_n = bisect.bisect_right(self.nt, t) - 1
        last = self.notes[i_n] if i_n >= 0 else None
        age = t - last["t"] if last else 9
        for r in range(6):
            y0 = hor + (r ** 1.6) * 4.2; y1 = hor + ((r + 1) ** 1.6) * 4.2
            wrow = 40 + r * 26
            for c in range(12):
                xa = CW / 2 - wrow + c * 2 * wrow / 12
                xb = xa + 2 * wrow / 12 - 1
                lit = (self.tiles[r, c] + (i_n if i_n >= 0 else 0) * 7) % 5 == 0
                col = pitch_color(last["p"] if last else 60, self.cfg.hue) if lit else P["ground"]
                k_ = (0.25 + 1.2 * math.exp(-age / 0.25)) if lit else 0.45 + 0.3 * kick
                cv.rect(xa, y0, xb, y1 - 1, col * k_)
        # dancer + reflection-free shadow
        cx = CW / 2
        cv.rect(cx - 20, self.GROUND + 1, cx + 20, self.GROUND + 3, P["dark"])
        self.draw_bit(cv, t, cx, P)
        # crowd silhouettes in front, bouncing on the kick
        for x, ph, s in self.crowd:
            bx = x * CW
            by = CH - 6 - 4 * kick * (0.5 + s) - 2 * math.sin(t * 6 + ph * 6)
            cv.rect(bx - 6, by, bx + 6, CH, P["dark"])
            cv.rect(bx - 4, by - 7, bx + 4, by, P["dark"])
            if s > 0.7:     # some hands in the air
                cv.rect(bx + 4, by - 16 - 3 * kick, bx + 6, by - 2, P["dark"])
        # confetti on snares in the drops
        if en > 0.6:
            for j, sn in enumerate(F.snares[max(0, bisect.bisect_right(F.snares, t) - 2):bisect.bisect_right(F.snares, t)]):
                cv.burst(CW / 2, 40, t - sn, int(sn * 1000) + j, P["accent"], n=18, speed=70, life=0.6, grav=50)
        self.title_card(cv, t, self.cfg.title or "DANCE!")


INSERTS = {c.name: c for c in (Runner, Bounce, Breakout, Shooter, Dancer)}
from .scenes_hi import HI_INSERTS  # noqa: E402  (hi-res screens: chat, tape, keys)
INSERTS.update(HI_INSERTS)


# ---------------------------------------------------------------- compositing
class ArcadeLayer:
    """Shows scheduled games as a CRT screen in the middle of the frame."""
    X0 = (1080 - CW * SCALE) // 2
    Y0 = 330

    def __init__(self, F, schedule, W=1080, H=1920):
        """schedule: list of (insert_instance, t_on, t_off)."""
        self.F, self.W, self.H = F, W, H
        self.items = schedule
        self.cv = Canvas(CW, CH)
        h, w = CH * SCALE, CW * SCALE
        yy, xx = np.mgrid[0:h, 0:w]
        rad = 34
        cx = np.clip(xx, rad, w - rad); cy = np.clip(yy, rad, h - rad)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - rad
        self.round = np.clip(0.5 - d, 0, 1).astype(np.float32)
        scan = np.ones(h, np.float32); scan[SCALE - 1::SCALE] = 0.72
        self.scan = scan[:, None, None]

    def active(self, t):
        for ins, a, b in self.items:
            if a - 0.05 <= t < b + 0.05:
                return ins, a, b
        return None

    def presence(self, t):
        """0..1 how much of the screen is up (used to fade the note ribbon)."""
        x = self.active(t)
        if not x:
            return 0.0
        _, a, b = x
        return smoothstep((t - a) / 0.3) * smoothstep((b - t) / 0.3)

    def draw(self, img, t, hue):
        x = self.active(t)
        if not x:
            return 0.0
        ins, a, b = x
        on = smoothstep((t - a) / 0.28)
        off = smoothstep((b - t) / 0.22)
        k = min(on, off)
        if k <= 0:
            return 0.0
        if getattr(ins, "hires", False):
            up = ins.draw_hi(t).astype(np.float32)
        else:
            self.cv.clear()
            ins.draw(self.cv, t)
            up = np.repeat(np.repeat(self.cv.a, SCALE, 0), SCALE, 1) * self.scan
        h, w = up.shape[:2]
        # CRT power on/off: a bright line that opens vertically
        vis_h = max(4.0, h * k ** 1.5)
        y_c = h / 2
        ys = np.arange(h)
        vm = np.clip((vis_h / 2 - np.abs(ys - y_c)) / 3.0, 0, 1).astype(np.float32)[:, None]
        alpha = (self.round * vm)[..., None]
        line = math.exp(-((k - 0.15) / 0.12) ** 2) if k < 0.6 else 0.0
        reg = img[self.Y0:self.Y0 + h, self.X0:self.X0 + w]
        kick = self.F.pulse("kick", t, 0.12)
        reg[:] = reg * (1 - 0.94 * alpha) + up * 1.05 * alpha + line * 3.0 * alpha * np.array([0.8, 0.9, 1.0], np.float32)
        r, g, b_ = [float(v) for v in hsv(hue + 0.5, 0.5, 1.0)]
        K2.glow_rect(img, self.X0 - 3, self.Y0 + y_c - vis_h / 2 - 3, self.X0 + w + 3, self.Y0 + y_c + vis_h / 2 + 3,
                     r, g, b_, 0.0, (0.25 + 0.35 * kick) * k, 34.0)
        return k
