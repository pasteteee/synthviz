"""SynthViz release: render N channel-ready Shorts that never repeat earlier ones.

    python release.py --n 10 --out release/batch1 --history channel/history.json

For every video it
  * picks a style (no two neighbours alike, least-used styles first),
  * composes a track and rejects it if its sound fingerprint (style + lead + bass + kick + key + progression)
    or its hook already exists in the history, or if the melody is out of key,
  * picks a visual look (world, game pair, outro) that differs from neighbours and history,
  * renders the mp4 (+ .json), a cover, and English title / description / tags / pinned comment.
Then it writes the channel kit (avatar, banner, channel texts) and appends everything to the history file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from synthviz.composer import STYLES, compose  # noqa: E402
from synthviz import harmony  # noqa: E402

# ------------------------------------------------------------------ texts (international, English)
GENRE = {"synthwave": "Synthwave", "dnb_liquid": "Liquid DnB", "dnb_energy": "Drum & Bass", "hardstyle": "Hardstyle",
         "rawstyle": "Rawstyle", "trance": "Trance", "acid": "Acid Techno", "chiptune": "Chiptune",
         "ambient_space": "Space Ambient", "future_bass": "Future Bass", "heavy_bass": "Bass House", "jerk": "Jerk"}
EMOJI = {"synthwave": "🌆", "dnb_liquid": "🌊", "dnb_energy": "⚡", "hardstyle": "🔥", "rawstyle": "💀", "trance": "✨",
         "acid": "🧪", "chiptune": "👾", "ambient_space": "🪐", "future_bass": "💜", "heavy_bass": "🔊", "jerk": "💎"}
TAGS = {
    "synthwave": ["synthwave", "retrowave", "outrun", "80s", "neon"],
    "dnb_liquid": ["dnb", "liquiddnb", "drumandbass", "chill", "liquid"],
    "dnb_energy": ["dnb", "drumandbass", "neurofunk", "bass", "reese"],
    "hardstyle": ["hardstyle", "euphoric", "hardstylekick", "edm", "festival"],
    "rawstyle": ["rawstyle", "hardstyle", "rawkick", "hardcore", "screech"],
    "trance": ["trance", "uplifting", "edm", "euphoria", "supersaw"],
    "acid": ["acid", "303", "techno", "rave", "acidtechno"],
    "chiptune": ["chiptune", "8bit", "pixelart", "retrogaming", "gameboy"],
    "ambient_space": ["ambient", "space", "relax", "cosmos", "chillout"],
    "future_bass": ["futurebass", "808", "edm", "melodic", "chords"],
    "heavy_bass": ["basshouse", "bass", "growl", "edm", "dubstep"],
    "jerk": ["jerk", "jerkbeat", "hyperpop", "808", "typebeat"],
}
COMMON_TAGS = ["electronicmusic", "musicvisualizer", "synth", "producer", "shorts"]
HOOKS = {
    "synthwave": ["for night drives", "straight out of 1986", "for the neon city", "with a chorus you'll hum all day"],
    "dnb_liquid": ["that flows like water", "for rainy windows", "smooth as glass"],
    "dnb_energy": ["at full speed", "with a reese bass that bites", "174 BPM of pure energy"],
    "hardstyle": ["with a kick you can feel", "for the main stage", "wait for the drop"],
    "rawstyle": ["with the rawest kick", "pure distortion", "do not play this quietly"],
    "trance": ["for the sunrise", "hands-in-the-air melody", "pure euphoria"],
    "acid": ["that melts your brain", "303 squelch therapy", "warehouse at 4am"],
    "chiptune": ["in a pixel world", "boss fight energy", "final level music"],
    "ambient_space": ["to drift through space", "zero gravity", "for deep focus"],
    "future_bass": ["that hits different at 2am", "with chords that bend", "808 + supersaws"],
    "heavy_bass": ["with a bass you can feel", "filthy growl drop", "turn your speakers up"],
    "jerk": ["that makes you bounce", "sparkly bells + 808 slides", "pure jerk energy"],
}
QUESTIONS = ["Drop 1 or drop 2? 👇", "Rate the drop 1–10 👇", "Which part got stuck in your head? 👇",
             "What genre should I make next? 👇", "Did you catch the switch in the second drop? 👇",
             "Headphones or speakers for this one? 👇", "Would you play this at a party? 👇"]
NOTE_TEXT = {"hardstyle": "The kick is tuned to every chord.", "rawstyle": "The kick is tuned to every chord.",
             "future_bass": "The 808 glides between notes.", "heavy_bass": "Clean sub + distorted mid bass.",
             "dnb_energy": "Reese bass, half-time switch in the second drop.",
             "dnb_liquid": "Liquid rollers, warm reese and soft keys."}


def texts(meta, rng):
    style = meta["style_id"]
    title = meta.get("title", "UNTITLED").upper()
    g = GENRE.get(style, style.title())
    em = EMOJI.get(style, "🎹")
    hook = rng.choice(HOOKS.get(style, ["to zone out to"]))
    prog = meta.get("progression", "")
    if isinstance(prog, list):
        prog = " – ".join(prog)
    tags = TAGS.get(style, []) + COMMON_TAGS
    yt_title = rng.choice([f"{title} · {g} {hook} {em} #shorts",
                           f"{g} {hook} {em} | {title} #shorts",
                           f"{title} ({g}) {em} {hook} #shorts"])[:100]
    desc = "\n".join([
        f"{title} · {g} · {meta.get('key', '')} · {meta.get('bpm', '')} BPM",
        f"Chords: {prog}",
        "",
        "Every sound, chord and melody here is generated in code, and the visuals react to the real notes, drums and bass.",
        NOTE_TEXT.get(style, "Watch the note lane: every light is a real note of the track."),
        "",
        "Subscribe for more drops 💜",
        "",
        " ".join("#" + t for t in tags[:8]),
    ])
    return dict(title=yt_title, description=desc, tags=tags, pinned=rng.choice(QUESTIONS),
                hashtags=" ".join("#" + t for t in tags[:3]))


# ------------------------------------------------------------------ uniqueness
TITLE_A = ["NEON", "VOLT", "CRIMSON", "ORBIT", "VELVET", "FERAL", "GOLDEN", "STATIC", "MIDNIGHT", "HOLLOW", "SOLAR",
           "LUNAR", "PHANTOM", "CHROME", "WILD", "SILENT", "ELECTRIC", "SAVAGE", "CRYSTAL", "ATOMIC", "INFINITE", "SHADOW",
           "SONIC", "ULTRA", "NOVA", "TITAN", "COBALT", "RUBY", "ARCTIC", "MOLTEN"]
TITLE_B = ["DRIFT", "RUSH", "SKYLINE", "EMPIRE", "PROTOCOL", "MIRAGE", "VORTEX", "HORIZON", "OVERDRIVE", "RIOT", "SPIRAL",
           "RITUAL", "COMET", "PARADOX", "STORM", "FRONTIER", "AFTERGLOW", "SURGE", "VERTIGO", "CIRCUIT", "LEGACY", "FEVER",
           "ZENITH", "TEMPLE", "MACHINE", "ANTHEM", "CASCADE", "IGNITION", "VELOCITY", "DYNASTY"]


def fresh_title(title, history_titles, rng):
    """Keep the engine's title unless one of its words was used in the last 30 titles."""
    recent = {w for t in history_titles[-30:] for w in str(t).split()}
    if title and not (set(title.split()) & recent):
        return title
    for _ in range(200):
        t = f"{rng.choice(TITLE_A)} {rng.choice(TITLE_B)}"
        if not (set(t.split()) & recent):
            return t
    return title


def fingerprint(ev):
    return "|".join(str(ev.get(k)) for k in ("style_id", "lead_patch", "bass_patch", "kick", "key", "prog_name"))


def hook_sig(ev):
    """Pitch-interval + rhythm signature of the first 16 lead notes (transposition-free)."""
    ln = ev.get("lead", [])[:16]
    if len(ln) < 4:
        return ""
    t0 = ln[0]["t"]
    s = ",".join(f"{round((n['t'] - t0) / ev['spb'] * 4)}:{n['p'] - ln[0]['p']}" for n in ln)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def in_key_ratio(song):
    """Share of lead/counter/bass notes that sit in the chord-scale of their chord (musical sanity)."""
    ev = song.events
    from synthviz.theory import Key
    k = song.key if isinstance(song.key, Key) else None
    ok = tot = 0
    for part in ("lead", "counter", "bass"):
        for n in ev.get(part, []):
            beat = n["t"] / ev["spb"]
            ce = harmony.chord_at(song.chords, beat)
            pcs = set(harmony.chord_scale(k, ce.chord)) if k else set(ce.chord.pcs)
            tot += 1
            ok += (n["p"] % 12) in pcs
    return ok / max(1, tot)


def pick_styles(n, history, rng, allowed):
    used = [h.get("style_id") for h in history]
    out = []
    for _ in range(n):
        prev = out[-1] if out else (used[-1] if used else None)
        cand = [s for s in allowed if s != prev]
        s = min(cand, key=lambda s: ((used + out).count(s), rng.random()))
        out.append(s)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="release/batch")
    ap.add_argument("--history", default="channel/history.json")
    ap.add_argument("--styles", help="comma list to restrict styles")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--audio-only", action="store_true", help="only compose + check (fast dry run)")
    ap.add_argument("--no-kit", action="store_true", help="skip avatar/banner/covers")
    a = ap.parse_args()

    rng = random.Random(a.seed if a.seed is not None else time.time_ns())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hist_path = Path(a.history)
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    allowed = a.styles.split(",") if a.styles else [s for s in STYLES if s != "ambient_space"]
    styles = pick_styles(a.n, history, rng, allowed)
    from synthviz.visuals.variety import plan_batch
    looks = plan_batch(styles, seed=rng.randrange(10 ** 6), history=history)
    from synthviz import surge
    for h in history:
        surge.USED.update((h.get("surge") or {}).values())
    print("surge:", "on" if surge.available() else "off (built-in synths only)", flush=True)
    fps_seen = {h.get("fingerprint") for h in history}
    hooks_seen = {h.get("hook") for h in history}
    titles_seen = {h.get("title") for h in history}
    made = []
    base_n = len(history)
    for j, style in enumerate(styles):
        for attempt in range(8):
            seed = rng.randrange(10 ** 6)
            song = compose(style, seed=seed)
            ev = dict(song.events, style_id=style)
            fp, hs, ik = fingerprint(ev), hook_sig(ev), in_key_ratio(song)
            if fp in fps_seen or (hs and hs in hooks_seen) or song.title in titles_seen or ik < 0.9:
                print(f"  reject {style} s{seed}: fp_dup={fp in fps_seen} hook_dup={hs in hooks_seen} in_key={ik:.2f}")
                continue
            break
        song.title = fresh_title(song.title, [h.get("title") for h in history] + [m["title"] for m in made], rng)
        song.events["title"] = song.title
        fps_seen.add(fp); hooks_seen.add(hs); titles_seen.add(song.title)
        name = f"{base_n + j + 1:03d}_{style}_{seed}"
        path = out / f"{name}.mp4"
        lk = looks[j]
        print(f"[{j + 1}/{a.n}] {name} | {song.title} | {ev['key']} | lead={ev['lead_patch']} bass={ev['bass_patch']} "
              f"| world={lk['world']} games={lk['inserts']} outro={lk['outro']} | in-key {ik:.2f}", flush=True)
        if a.audio_only:
            from synthviz.composer import save_wav
            save_wav(path.with_suffix(".wav"), song.audio)
            meta = dict(seed=seed, style_id=style, **{k: ev[k] for k in ("title", "bpm", "key", "progression", "prog_name",
                                                                          "lead_patch", "bass_patch", "kick")}, **lk)
        else:
            from synthviz.video import render_video
            render_video(style, seed=seed, out=path, fps=a.fps, world=lk["world"], inserts=lk["inserts"],
                         outro=lk["outro"], song=song)
            meta = json.loads(path.with_suffix(".json").read_text())
        meta.update(fingerprint=fp, hook=hs, in_key=round(ik, 3), file=path.name, surge=ev.get("surge", {}), **lk)
        meta["texts"] = texts(meta, rng)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        made.append(meta)
        history.append({k: meta.get(k) for k in ("file", "style_id", "seed", "title", "key", "lead_patch", "bass_patch",
                                                 "kick", "world", "inserts", "outro", "fingerprint", "hook", "surge")})
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_text(json.dumps(history, indent=1, ensure_ascii=False))

    # ---------------- YouTube texts for the batch
    md = ["# YouTube upload sheet\n", "Copy each block as is. The cover with the same name is in `covers/`.\n"]
    for m in made:
        t = m["texts"]
        md += [f"## {m['file']}\n", f"**Title:**\n```\n{t['title']}\n```", f"**Description:**\n```\n{t['description']}\n```",
               f"**Tags:**\n```\n{', '.join(t['tags'])}\n```", f"**Pinned comment:**\n```\n{t['pinned']}\n```\n"]
    (out / "UPLOAD.md").write_text("\n".join(md))
    if not a.no_kit and not a.audio_only:
        import channel_kit as CK
        CK.build(out, out / "kit", made)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
