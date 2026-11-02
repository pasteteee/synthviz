"""Director: turns the arrangement into a coherent edit.

v3 cut to a random scene every 4 bars with a glitch, recoloured every scene by
the chord on the full colour wheel, so a video looked like 6 unrelated clips.
The v4 rules:

  * one WORLD per video: a palette hue + a family of 2-4 related scenes;
  * cuts only where the music turns a page (section starts, the drop-2 switch),
    each with a transition that matches the moment: 'dive' into a drop,
    'cross' (half-bar dissolve) into calm parts, 'flash' on the switch;
  * chords only nudge the hue inside the world (+-0.08) instead of jumping;
  * ARCADE INSERTS show up where the music 'runs': the game is chosen from the
    section's note density, note length and drum style (see pick_insert)."""
from __future__ import annotations

import numpy as np

from .inserts import INSERTS, InsertCfg

# world -> (hue, calm scenes, energetic scenes)
WORLDS = {
    "neon_drive": (0.86, ["ridges", "waves"], ["outrun", "city", "terrain"]),
    "deep_space": (0.62, ["nebula", "galaxy"], ["tunnel", "galaxy", "nebula"]),
    "arcade":     (0.08, ["pixel"], ["pixel", "outrun", "tunnel"]),
    "liquid":     (0.50, ["waves", "nebula"], ["waves", "terrain", "galaxy"]),
    "rave":       (0.78, ["scope", "waves"], ["kaleido", "tunnel", "city"]),
    "canyon":     (0.98, ["ridges", "waves"], ["terrain", "outrun", "city"]),
}
STYLE_WORLDS = {
    "synthwave": ["neon_drive", "canyon"],
    "dnb_liquid": ["liquid", "deep_space"],
    "dnb_energy": ["rave", "deep_space", "neon_drive"],
    "hardstyle": ["rave", "canyon", "neon_drive"],
    "trance": ["deep_space", "liquid", "rave"],
    "acid": ["rave", "arcade"],
    "chiptune": ["arcade"],
    "ambient_space": ["deep_space", "liquid"],
    "future_bass": ["liquid", "neon_drive", "rave"],
    "heavy_bass": ["rave", "deep_space", "canyon"],
    "rawstyle": ["rave", "canyon"],
}
STYLE_INSERTS = {
    "chiptune": ["runner", "shooter", "breakout", "bounce"],
    "ambient_space": ["bounce"],
    "dnb_energy": ["runner", "shooter", "bounce"],
    "dnb_liquid": ["runner", "bounce"],
}
ALL_INSERTS = list(INSERTS)


def section_stats(F, s):
    a, b = s["t"], s["t"] + s["dur"]
    lead = [n for n in F.ev.get("lead", []) if a <= n["t"] < b]
    dens = len(lead) / max(1e-3, s["dur"])
    mdur = float(np.median([n["dur"] for n in lead])) if lead else 0.3
    kicks = [k for k in F.kicks if a <= k < b]
    snares = [k for k in F.snares if a <= k < b]
    four = len(kicks) / max(1e-3, s["dur"] / F.ev["spb"])   # kicks per beat
    return dict(dens=dens, mdur=mdur, four=four, snare=len(snares) / max(1e-3, s["dur"]))


def pick_insert(F, s, pool, used, rng):
    """Music -> game. A 'run' of short notes is a runner, long sustained notes
    bounce, a straight four-on-the-floor plays breakout, busy breakbeats shoot."""
    st = section_stats(F, s)
    score = {
        "runner": 1.0 + 0.5 * min(4.0, st["dens"]) - 1.5 * max(0.0, st["mdur"] - 0.3),
        "bounce": 1.0 + 3.0 * min(0.6, st["mdur"]) - 0.25 * st["dens"],
        "breakout": 1.0 + 2.0 * (st["four"] > 0.85) + 0.2 * st["dens"],
        "shooter": 1.0 + 0.4 * st["snare"] + 0.3 * st["dens"] - (1.0 if st["four"] > 0.85 else 0.0),
        "dancer": 1.5 + 2.5 * (st["four"] > 0.85),          # Bit dances wherever there is a straight beat
        "chat": 1.5 + 1.5 * min(0.6, st["mdur"]),           # slow melodies read like a conversation
        "tape": 1.6,
        "keys": 1.5 + 0.3 * min(4.0, st["dens"]),
    }
    cand = [g for g in pool if g not in used] or pool
    for g in cand:
        score[g] = score.get(g, 0) + rng.uniform(0, 0.6)
    return max(cand, key=lambda g: score.get(g, 0))


class Plan:
    def __init__(self, F, style, rng, world=None, inserts=None, insert_mode="drops", force_scene=None):
        self.F = F
        spb = F.ev["spb"]
        bar = 4 * spb
        self.world = world or rng.choice(STYLE_WORLDS.get(style, list(WORLDS)))
        hue, calm, hot = WORLDS[self.world]
        self.hue = (hue + rng.uniform(-0.04, 0.04)) % 1.0
        if force_scene:
            calm = hot = [force_scene]
        hot_cycle = rng.sample(hot, len(hot))
        hi = 0
        shots = []      # (t, scene, transition)
        for s in F.sections:
            nm = s["name"]
            if nm in ("intro", "break"):
                scene = rng.choice(calm)
                tr = "cross"
            elif nm == "outro":
                scene = shots[0][1] if shots else calm[0]     # come back home -> loopable
                tr = "cross"
            else:
                scene = hot_cycle[hi % len(hot_cycle)]; hi += 1
                tr = "dive" if nm in ("drop", "drop2") else "cross"
            if shots and shots[-1][1] == scene and nm not in ("outro",):
                alt = [x for x in (hot if nm not in ("intro", "break") else calm) if x != scene]
                if alt:
                    scene = rng.choice(alt)
            shots.append((s["t"], scene, tr))
        # the drop-2 switch reveals a world we have not seen yet (same palette)
        for e in F.fx:
            if e["kind"] == "switch":
                used = {sc for _, sc, _ in shots}
                from ..video import SCENES as _ALL
                pool = [x for x in hot + calm if x not in used] or [x for x in _ALL if x not in used] or hot
                shots.append((e["t"], rng.choice(pool), "flash"))
        shots.sort(key=lambda x: x[0])
        self.shots = shots
        self.cuts = [(t, tr) for t, _, tr in shots[1:]]
        # ---------------- arcade inserts
        self.inserts = []
        if insert_mode != "none":
            pool = inserts or STYLE_INSERTS.get(style, ALL_INSERTS)
            used = []
            sw = [e["t"] for e in F.fx if e["kind"] == "switch"]
            targets = ("drop", "drop2") if insert_mode == "drops" else ("build", "drop", "break", "drop2")
            n_level = 0
            for s in F.sections:
                if s["name"] not in targets:
                    continue
                a = s["t"] + (bar * 2 if s["name"] in ("drop", "drop2") and s["dur"] >= bar * 6 else 0.0)
                b = s["t"] + s["dur"] - 0.05
                inner = [x for x in sw if s["t"] < x < b]
                if inner:
                    # the drop-2 switch is the reveal: the screen powers on into a game
                    a = inner[0] + 0.02
                if b - a < bar * 2:
                    continue
                g = pick_insert(F, s, pool, used, rng)
                used.append(g)
                n_level += 1
                title = {"runner": "LEVEL", "breakout": "STAGE", "shooter": "WAVE", "bounce": "", "dancer": ""}.get(g, "")
                cfg = InsertCfg(hue=self.hue, title=f"{title} {n_level}" if title else "",
                                hero_hue=(self.hue + 0.5) % 1.0)
                self.inserts.append((g, a, b, cfg))

    def scene_at(self, t):
        cur = self.shots[0]
        for sh in self.shots:
            if sh[0] <= t + 1e-6:
                cur = sh
        return cur

    def build_inserts(self, seed=0):
        out = []
        for j, (g, a, b, cfg) in enumerate(self.inserts):
            out.append((INSERTS[g](self.F, a, b, cfg, seed=seed + j), a, b))
        return out

    def describe(self):
        return dict(world=self.world, hue=round(self.hue, 3),
                    shots=[(round(t, 2), sc, tr) for t, sc, tr in self.shots],
                    inserts=[(g, round(a, 2), round(b, 2)) for g, a, b, _ in self.inserts])
