"""CLI:  python -m synthviz --style dnb_liquid --seed 7 --out out/liquid.mp4

    --list                 show styles, scenes, patches, scales, progressions
    --style NAME           synthwave | dnb_liquid | dnb_energy | hardstyle | trance |
                           acid | chiptune | ambient_space   (or 'random')
    --scene / --alt-scene  visual scene for drops / for intro-build-break
    --key A --scale dorian --prog liquid_dnb --bpm 174 --duration 45 --title "..."
    --audio-only           only render the .wav
    --no-overlay           no title / chords / piano
    --batch N              render N random shorts
"""
import argparse
import random
import sys
from pathlib import Path

from . import theory
from .composer import STYLES, compose, save_wav
from .instruments import PATCHES


def main(argv=None):
    ap = argparse.ArgumentParser(prog="synthviz")
    ap.add_argument("--style", default="random")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="out/short.mp4")
    ap.add_argument("--scene")
    ap.add_argument("--alt-scene")
    ap.add_argument("--key")
    ap.add_argument("--scale")
    ap.add_argument("--prog")
    ap.add_argument("--bpm", type=float)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--title")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=20, help="x264 quality (lower = bigger/better)")
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    # visuals v4
    ap.add_argument("--world", help="neon_drive | deep_space | arcade | liquid | rave | canyon")
    ap.add_argument("--inserts", help="comma list: runner,bounce,breakout,shooter")
    ap.add_argument("--insert-mode", default="drops", help="drops | all | none")
    ap.add_argument("--outro", help="question | rate | replay | loop")
    ap.add_argument("--segments", help="preview ranges, e.g. 8.5-18.5,31-41")
    # external track (e.g. made by Lyria via tools/gemini_music.py): visuals follow its analysed events
    ap.add_argument("--audio", help="render visuals for this audio file instead of composing a track")
    ap.add_argument("--window", help="cut of --audio in seconds, e.g. 31.5-90.6")
    ap.add_argument("--look", default="v4", help="v4 (neon worlds) | marble (bright paper-marbling renderer)")
    ap.add_argument("--palettes", help="marble: two palette names, e.g. acid,bubblegum")
    a = ap.parse_args(argv)

    if a.list:
        from .visuals.scenes import SCENES
        print("styles:      ", ", ".join(STYLES))
        print("scenes:      ", ", ".join(SCENES))
        from .visuals.director import WORLDS
        from .visuals.inserts import INSERTS
        from .visuals.captions import OUTRO_LINES
        print("worlds:      ", ", ".join(WORLDS))
        print("inserts:     ", ", ".join(INSERTS))
        print("outros:      ", ", ".join(OUTRO_LINES))
        print("patches:     ", ", ".join(PATCHES))
        print("scales:      ", ", ".join(theory.SCALES))
        print("chords:      ", ", ".join(q or "maj" for q in theory.CHORD_TYPES))
        print("progressions:")
        for k, (r, tags) in theory.PROGRESSIONS.items():
            print(f"   {k:16s} {' - '.join(r):34s} {', '.join(sorted(tags))}")
        return

    n = max(1, a.batch)
    styles_b = [a.style if a.style != "random" else random.choice(list(STYLES)) for _ in range(n)]
    looks = None
    if n > 1:
        from .visuals.variety import plan_batch
        looks = plan_batch(styles_b, seed=a.seed or random.randrange(10 ** 6))
    for j in range(n):
        style = styles_b[j]
        seed = a.seed if (a.seed is not None and n == 1) else random.randrange(10 ** 6)
        out = Path(a.out)
        if n > 1:
            out = out.with_name(f"{out.stem}_{j + 1:02d}_{style}_{seed}{out.suffix}")
        kw = dict(key=a.key, scale=a.scale, progression=a.prog, bpm=a.bpm, title=a.title, duration=a.duration)
        kw = {k: v for k, v in kw.items() if v is not None}
        if a.audio_only:
            song = compose(style, seed=seed, **kw)
            out.parent.mkdir(parents=True, exist_ok=True)
            save_wav(out.with_suffix(".wav"), song.audio)
            print(f"{song.title} | {song.events['key']} | {song.bpm} BPM | {' - '.join(song.events['progression'])} -> {out.with_suffix('.wav')}")
            continue
        from .video import render_video
        ext = None
        if a.audio:
            from .external import song_from_audio
            w0, w1 = (map(float, a.window.split("-")) if a.window else (0.0, None))
            ext = song_from_audio(a.audio, t0=w0, t1=w1, title=a.title or Path(a.audio).stem.upper(), style=style,
                                  bpm=a.bpm)
        if a.look == "marble":
            from .composer import compose as _compose
            from .visuals.marble import render_marble
            song = ext or _compose(style, seed=seed, **kw)
            render_marble(song, out, seed=seed, fps=a.fps, crf=a.crf,
                          palettes=a.palettes.split(",") if a.palettes else None)
            continue
        render_video(style, seed=seed, song=ext, out=out, scene=a.scene, alt_scene=a.alt_scene, fps=a.fps,
                     overlay=not a.no_overlay, song_kw=kw, crf=a.crf, world=a.world or (looks[j]["world"] if looks else None),
                     inserts=a.inserts.split(",") if a.inserts else (looks[j]["inserts"] if looks else None),
                     insert_mode=a.insert_mode,
                     outro=a.outro or (looks[j]["outro"] if looks else None),
                     segments=[tuple(map(float, x.split("-"))) for x in a.segments.split(",")] if a.segments else None)
        if a.audio:
            break


if __name__ == "__main__":
    sys.exit(main())
