"""Batch variety: give N videos in a row different looks.

plan_batch(styles, seed) -> list of kwargs for render_video (world, inserts,
outro), so that inside a batch:
  * no two neighbouring videos share a world, and worlds rotate before repeating;
  * the pair of arcade games changes every video (each game appears evenly);
  * the outro line rotates (question / rate / tomorrow / replay).
The style still limits which worlds fit (director.STYLE_WORLDS), but if a style's
own worlds are used up, any unused world is taken instead of repeating."""
from __future__ import annotations

import itertools
import random

from .director import WORLDS, STYLE_WORLDS
from .inserts import INSERTS

OUTROS = ["question", "rate", "tomorrow", "replay"]


def plan_batch(styles, seed=0, history=None):
    """styles: list of style ids, one per video. history: kwargs of earlier
    videos (e.g. read back from the .json files) to avoid as well."""
    rng = random.Random(seed)
    games = list(INSERTS)
    pairs = [list(p) for p in itertools.permutations(games, 2)]
    rng.shuffle(pairs)
    used_worlds = [h.get("world") for h in (history or []) if h.get("world")]
    used_pairs = [tuple(x[0] if isinstance(x, (list, tuple)) else x for x in (h.get("inserts") or ()))
                  for h in (history or [])]
    off = rng.randrange(len(OUTROS))
    out = []
    for k, style in enumerate(styles):
        fit = STYLE_WORLDS.get(style, list(WORLDS))
        recent = used_worlds[-max(1, len(WORLDS) - 1):]
        cand = [w for w in fit if w not in recent] or [w for w in WORLDS if w not in recent] or list(WORLDS)
        # least-used first, random tie-break
        world = min(cand, key=lambda w: (used_worlds.count(w), rng.random()))
        used_worlds.append(world)
        pair = next((p for p in pairs if tuple(p) not in used_pairs and
                     (not out or p[0] not in out[-1]["inserts"])), pairs[k % len(pairs)])
        used_pairs.append(tuple(pair))
        out.append(dict(world=world, inserts=pair, outro=OUTROS[(k + off) % len(OUTROS)]))
    return out
