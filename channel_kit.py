"""Channel kit: avatar, banner, per-video covers and videos.md from rendered demos.

usage: python channel_kit.py <demos_dir> <out_dir>
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter

FONTS = Path(__file__).resolve().parent / "synthviz" / "fonts"
OSWALD = str(FONTS / "Oswald.ttf")
INTER = str(FONTS / "Inter-SemiBold.otf")

GENRE_RU = {
    "synthwave": "SYNTHWAVE", "dnb_liquid": "LIQUID DNB", "dnb_energy": "DRUM & BASS",
    "hardstyle": "HARDSTYLE", "trance": "TRANCE", "acid": "ACID", "chiptune": "CHIPTUNE",
    "ambient_space": "SPACE AMBIENT", "future_bass": "FUTURE BASS", "rawstyle": "RAWSTYLE", "heavy_bass": "BASS HOUSE",
}
TAGS = {
    "synthwave": "#synthwave #retrowave #outrun #80s #neon",
    "dnb_liquid": "#dnb #liquiddnb #drumandbass #chill #neon",
    "dnb_energy": "#dnb #drumandbass #neurofunk #bass #energy",
    "hardstyle": "#hardstyle #rawstyle #edm #kick #festival",
    "trance": "#trance #uplifting #edm #euphoria #neon",
    "acid": "#acid #303 #techno #rave #neon",
    "chiptune": "#chiptune #8bit #pixelart #retrogaming #gameboy",
    "ambient_space": "#ambient #space #relax #cosmos #chillout",
    "future_bass": "#futurebass #edm #kawaii #chords #melodic",
}
EMOJI = {"synthwave": "🌆", "dnb_liquid": "🌊", "dnb_energy": "⚡", "hardstyle": "🔥", "trance": "✨",
         "acid": "🧪", "chiptune": "👾", "ambient_space": "🪐", "future_bass": "💜"}
NOTE_RU = {"C": "до", "C#": "до-диез", "Db": "ре-бемоль", "D": "ре", "D#": "ре-диез", "Eb": "ми-бемоль",
           "E": "ми", "F": "фа", "F#": "фа-диез", "Gb": "соль-бемоль", "G": "соль", "G#": "соль-диез",
           "Ab": "ля-бемоль", "A": "ля", "A#": "ля-диез", "Bb": "си-бемоль", "B": "си"}
MODE_RU = {"major": "мажор", "minor": "минор", "natural_minor": "минор", "harmonic_minor": "гармонический минор",
           "dorian": "дорийский", "phrygian": "фригийский", "lydian": "лидийский", "mixolydian": "миксолидийский",
           "aeolian": "минор", "melodic_minor": "мелодический минор", "pentatonic_minor": "минорная пентатоника",
           "pentatonic_major": "мажорная пентатоника"}


HOOKS = {"synthwave": "for night drives", "dnb_liquid": "that flows like water", "dnb_energy": "at full speed",
         "hardstyle": "with a kick you can feel", "trance": "for the sunrise", "acid": "that melts your brain",
         "chiptune": "in a pixel world", "ambient_space": "to drift through space", "future_bass": "that hits different at 2am"}
GENRE_TITLE = {"synthwave": "Synthwave", "dnb_liquid": "Liquid DnB", "dnb_energy": "Drum & Bass", "hardstyle": "Hardstyle",
               "trance": "Trance", "acid": "Acid", "chiptune": "Chiptune", "ambient_space": "Space Ambient",
               "future_bass": "Future Bass"}
PINNED = {"hardstyle": "Did the kick at the drop hit you too? Rate it 1–10 👇",
          "synthwave": "Where are you driving to with this one? 🌆👇",
          "chiptune": "Which game would this be the soundtrack of? 👾👇",
          "ambient_space": "Which planet are you on right now? 🪐👇"}


# ---------------------------------------------------------------- helpers
def glow(arr, sigmas=(3, 12, 40), gains=(0.9, 0.6, 0.45)):
    out = arr.copy()
    for s, g in zip(sigmas, gains):
        out += gaussian_filter(arr, (s, s, 0)) * g
    return out


def tonemap(a):
    return (np.clip(1 - np.exp(-a * 1.1), 0, 1) * 255).astype(np.uint8)


def text_layer(size, items):
    """items: (text, font_path, px, xy|None(center), anchor, fill(float rgb), stroke)"""
    img = Image.new("RGB", size, 0)
    d = ImageDraw.Draw(img)
    for text, fp, px, xy, anchor, col in items:
        f = ImageFont.truetype(fp, px)
        d.text(xy, text, font=f, anchor=anchor, fill=tuple(int(c * 255) for c in col))
    return np.asarray(img).astype(np.float32) / 255.0


def hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1, s, v), np.float32)


def grab(video, t, out):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(out)],
                   check=True)
    return Image.open(out).convert("RGB")


def best_frame(video, dur, tmp, prev_sigs):
    """Most vivid frame of the drops that does not look like covers picked before
    (so the set of covers shows different worlds) and is not an invert flash."""
    cands = []
    for k, frac in enumerate(np.linspace(0.26, 0.80, 14)):
        im = grab(video, dur * frac, tmp / f"cand{k}.png")
        a = np.asarray(im).astype(np.float32) / 255
        top = a[:1100]
        if top.mean() > 0.55:          # invert flash / washed frame
            continue
        mx, mn = top.max(2), top.min(2)
        sat = (mx - mn) / (mx + 1e-3)
        score = float((sat * mx).mean()) + 0.5 * float(mx.mean())
        sig = np.asarray(Image.fromarray((top * 255).astype(np.uint8)).resize((6, 10))).astype(np.float32).ravel() / 255
        sim = max([1 - np.abs(sig - p).mean() * 4 for p in prev_sigs], default=0.0)
        cands.append((score - 0.9 * max(0.0, sim), im, sig))
    cands.sort(key=lambda c: -c[0])
    _, im, sig = cands[0]
    prev_sigs.append(sig)
    return im


# ---------------------------------------------------------------- avatar
def avatar(path, size=1024):
    S = size
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    cx, cy = S / 2, S / 2
    img = np.zeros((S, S, 3), np.float32)
    # deep space background
    r = np.hypot(x - cx, y - cy) / S
    img += hsv(0.72, 0.8, 0.10) * np.exp(-r * 2.5)[..., None]
    rng = np.random.default_rng(3)
    for _ in range(260):
        sx, sy = rng.uniform(0, S, 2)
        img[int(sy), int(sx)] += rng.uniform(0.3, 1.2)
    # retro sun with slits, in a circle
    R = S * 0.30
    sun_c = (cx, cy - S * 0.02)
    d = np.hypot(x - sun_c[0], y - sun_c[1])
    inside = d < R
    tgrad = np.clip((y - (sun_c[1] - R)) / (2 * R), 0, 1)
    col = (hsv(0.13, 0.85, 1.0)[None, None] * (1 - tgrad[..., None]) + hsv(0.92, 0.85, 1.0)[None, None] * tgrad[..., None])
    rel = (y - sun_c[1]) / R
    slit = np.ones_like(y)
    for k in range(6):
        y0 = 0.10 + k * 0.155
        h = 0.025 + k * 0.012
        slit *= ~((rel > y0) & (rel < y0 + h))
    img += col * (inside * slit)[..., None] * 0.9
    # horizon grid below
    hor = cy + R * 0.95
    below = y > hor
    zz = 1.0 / np.maximum((y - hor) / S, 1e-3)
    gx = np.abs(((x - cx) / S * zz * 0.35) % 1 - 0.5)
    gz = np.abs((zz * 0.08) % 1 - 0.5)
    grid = (np.exp(-((0.5 - gx) * S / zz * 0.07) ** 2) + np.exp(-((0.5 - gz) / 0.02) ** 2)) * below
    img += hsv(0.85, 0.7, 1.0) * (grid * np.exp(-(y - hor) / S * 1.2))[..., None] * 0.7
    # lissajous ring around the sun (3:4)
    t = np.linspace(0, 2 * np.pi, 6000)
    lx = cx + np.sin(3 * t + 0.6) * S * 0.40
    ly = cy + np.sin(4 * t) * S * 0.40
    line = np.zeros((S, S), np.float32)
    np.add.at(line, (np.clip(ly.astype(int), 0, S - 1), np.clip(lx.astype(int), 0, S - 1)), 0.6)
    line = gaussian_filter(line, 1.2) * 6
    lcol = hsv(0.52, 0.75, 1.0)
    img += lcol * line[..., None] * (1 - inside * 0.85)[..., None]
    img = glow(img)
    # monogram NT
    tl = text_layer((S, S), [("NT", OSWALD, int(S * 0.24), (cx, cy + S * 0.31), "mm", (1, 1, 1))])
    shadow = np.clip(gaussian_filter(tl[..., 0] if tl.ndim == 3 else tl, S * 0.018) * 2.2, 0, 1)
    img = img * (1 - 0.85 * shadow)[..., None]                       # dark halo so the letters read on the sun
    img = img + glow(tl * hsv(0.52, 0.8, 1.0)[None, None], (5, 18, 45), (1.2, 0.8, 0.5))   # cyan neon
    img = img * (1 - tl) + tl * 2.2
    # circular vignette (youtube crops a circle)
    img *= np.clip(1.25 - r * 1.6, 0, 1)[..., None] ** 0.6
    Image.fromarray(tonemap(img)).resize((800, 800), Image.LANCZOS).save(path)


# ---------------------------------------------------------------- banner
def banner(path, frames):
    W, H = 2560, 1440
    canvas = np.zeros((H, W, 3), np.float32)
    n = len(frames)
    pw = W // n
    for k, fr in enumerate(frames):
        # crop a central 1080 x (1080*H/pw) window of the scene area (top part, no HUD)
        a = np.asarray(fr).astype(np.float32) / 255
        ch = 1150
        cw = int(ch * pw / H)
        x0 = (1080 - cw) // 2
        crop = Image.fromarray((a[0:ch, x0:x0 + cw] * 255).astype(np.uint8)).resize((pw, H), Image.LANCZOS)
        canvas[:, k * pw:(k + 1) * pw] = np.asarray(crop).astype(np.float32) / 255
    # tone down, unify colours
    canvas = canvas ** 1.2 * 0.75
    # dark centre band for the safe area (1546 x 423)
    yy = np.linspace(-1, 1, H)[:, None]
    xx = np.linspace(-1, 1, W)[None, :]
    band = np.exp(-(yy / 0.32) ** 4) * np.exp(-(xx / 0.75) ** 6)
    canvas *= (1 - 0.8 * band)[..., None]
    # neon dividers
    lin = canvas.copy() * 0
    for k in range(1, n):
        lin[:, k * pw - 2:k * pw + 2] += hsv(0.85 + k * 0.1, 0.6, 1.0) * 0.8
    tl = text_layer((W, H), [
        ("NEON THEORY", OSWALD, 210, (W / 2, H / 2 - 40), "mm", (1, 1, 1)),
        ("SYNTH MUSIC YOU CAN SEE  •  NEW TRACK EVERY DAY", INTER, 44, (W / 2, H / 2 + 120), "mm", (1, 1, 1)),
    ])
    title_col = np.zeros_like(canvas)
    xx01 = np.linspace(0, 1, W)[None, :, None]
    title_col[:] = hsv(0.92, 0.75, 1.0) * (1 - xx01) + hsv(0.53, 0.75, 1.0) * xx01
    canvas = canvas + glow(lin, (4, 18), (1, 0.5))
    canvas = canvas * (1 - tl) + tl * (0.6 + 0.6 * title_col) + glow(tl * title_col, (5, 18, 50), (0.9, 0.6, 0.35))
    Image.fromarray(tonemap(canvas * 1.1)).save(path, quality=93)


# ---------------------------------------------------------------- cover
def cover(path, frame, title, genre, hue):
    W, H = 1080, 1920
    a = np.asarray(frame.resize((W, H))).astype(np.float32) / 255
    # crop out HUD region visually: darken bottom, keep scene
    yy = np.linspace(0, 1, H)[:, None, None]
    a = a * (1 - 0.93 * np.clip((yy - 0.52) / 0.08, 0, 1) ** 1.5)
    lin = -np.log(np.clip(1 - a * 0.98, 1e-3, 1))  # back to linear-ish for glow
    col = hsv(hue, 0.7, 1.0)
    col2 = hsv(hue + 0.5, 0.6, 1.0)
    f = ImageFont.truetype(OSWALD, 200)
    tw = ImageDraw.Draw(Image.new("L", (1, 1))).textlength(title, font=f)
    px = int(200 * min(1.0, 980 / max(tw, 1)))
    tl = text_layer((W, H), [(title, OSWALD, px, (W / 2, 1270), "mm", (1, 1, 1))])
    gl = text_layer((W, H), [(genre, OSWALD, 96, (W / 2, 1440), "mm", (1, 1, 1))])
    sl = text_layer((W, H), [("NEON THEORY", INTER, 40, (W / 2, 1560), "mm", (1, 1, 1))])
    lin = lin * (1 - tl) + tl * (0.8 + col) + glow(tl * col[None, None], (6, 22, 60), (1.0, 0.7, 0.4))
    lin = lin * (1 - gl) + gl * (0.5 + col2) + glow(gl * col2[None, None], (5, 18), (0.9, 0.5))
    lin = lin + sl * 0.9
    Image.fromarray(tonemap(lin)).save(path, quality=93)


def key_ru(key):
    parts = key.split()
    tonic = parts[0]
    mode = "_".join(parts[1:]) if len(parts) > 1 else "minor"
    return f"{NOTE_RU.get(tonic, tonic)} {MODE_RU.get(mode, mode.replace('_', ' '))}"


def main():
    demos, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    (out / "covers").mkdir(exist_ok=True)
    tmp = out / ".tmp"; tmp.mkdir(exist_ok=True)
    avatar(out / "avatar.png")
    print("avatar ok")
    vids = sorted(demos.glob("*.mp4"))
    frames = []
    md = ["# Texts for the demo Shorts\n",
          "> Для тебя: копируй блоки как есть. Обложка с тем же номером лежит в `covers/`.\n"]
    sigs = []
    for v in vids:
        meta = json.loads(v.with_suffix(".json").read_text())
        style = meta.get("style_id", meta.get("style", "synthwave"))
        title = meta.get("title", v.stem).upper()
        dur = float(meta.get("duration", 50))
        fr = best_frame(v, dur, tmp, sigs)
        frames.append(fr)
        hue = (hash(title) % 1000) / 1000
        genre = GENRE_RU.get(style, style.upper())
        key = meta.get("key", "A minor").replace("_", " ")
        prog = meta.get("progression", "")
        if isinstance(prog, list):
            prog = " – ".join(prog)
        em = EMOJI.get(style, "🎹")
        hook = HOOKS.get(style, "to zone out to")
        g_title = GENRE_TITLE.get(style, genre.title())
        cover(out / "covers" / f"{v.stem}.jpg", fr, title, genre, hue)
        md += [f"## {v.stem}\n",
               f"**Title:** `{title} · {g_title} {hook} {em} #shorts`\n",
               "**Description:**\n", "```",
               f"{title} · {g_title} · {key} · {meta.get('bpm')} BPM",
               f"Chords: {prog}",
               "",
               "Built entirely in code: synths, drums and harmony, no samples.",
               "The card at the bottom shows the live chord, the notes of the melody and bass, and the drum pattern.",
               "",
               "Subscribe for a new track every day 💜",
               "",
               f"{TAGS.get(style, '')} #electronicmusic #musicvisualizer #shorts", "```",
               "**Pinned comment:**", "```",
               PINNED.get(style, "Drop or ending, which part got you? Which genre should we make next? 👇"), "```\n"]
        print("cover", v.stem)
    picks = [frames[i] for i in (0, 2, 4, 6) if i < len(frames)] or frames
    banner(out / "banner.jpg", picks)
    print("banner ok")
    (out / "videos.md").write_text("\n".join(md))
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()


CHANNEL_TEXTS = """# NEON THEORY: channel kit (international)

## Name and handles
**NEON THEORY** · YouTube `@NeonTheory` (backups `@NeonTheoryMusic`, `@NeonTheoryFM`) · TikTok / Instagram `@neontheory`

## Channel description (YouTube → Customise channel → Basic info)
```
Synth music you can see.

Every track on this channel is generated in code: synths, 808s, hardstyle kicks, reese and growl basses, chords and melodies. The visuals are driven by the real notes. Kicks move the camera, the melody runs across the neon note lane, chords change the colours, and every drop throws you into a new world: a neon highway, deep space, an arcade game or a rave.

Hardstyle · Rawstyle · Drum & Bass · Liquid DnB · Future Bass · Bass House · Synthwave · Trance · Acid · Chiptune

New Shorts every day. Tell me in the comments which genre should drop next.
```

## Short bio (TikTok / Instagram, max 80 chars)
```
Synth music you can see 🎹 hardstyle · dnb · synthwave · new drop daily
```

## Channel keywords (Settings → Channel → Keywords)
```
synth music, music visualizer, electronic music, hardstyle, rawstyle, drum and bass, liquid dnb, future bass, bass house, synthwave, retrowave, trance, acid techno, chiptune, 8bit music, edm shorts, satisfying music, neon, 808, producer
```

## Channel settings checklist
- Avatar: `avatar.png` (800×800). Banner: `banner.jpg` (2560×1440, text sits in the safe area).
- Default upload settings: category **Music**, language **English**, "Altered or synthetic content" = **Yes** (the music is generated).
- Playlists: Hardstyle & Rawstyle Kicks · Drum & Bass 174 · Future Bass & Bass House · Synthwave Nights · 8-Bit World.
- Upload 1–2 Shorts a day at 17:00–19:00 US Eastern; pin the comment from UPLOAD.md under each Short and answer the first comments.
"""


def build(videos_dir, out, metas=None):
    """Avatar, banner, covers and channel texts for a rendered batch (used by release.py)."""
    videos_dir, out = Path(videos_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "covers").mkdir(exist_ok=True)
    tmp = out / ".tmp"; tmp.mkdir(exist_ok=True)
    avatar(out / "avatar.png")
    frames, sigs = [], []
    metas = metas or [json.loads(p.read_text()) for p in sorted(videos_dir.glob("*.json"))]
    for m in metas:
        v = videos_dir / m["file"]
        fr = best_frame(v, float(m.get("duration", 50)), tmp, sigs)
        frames.append(fr)
        title = m.get("title", v.stem).upper()
        hue = (int(hashlib.md5(title.encode()).hexdigest(), 16) % 1000) / 1000
        cover(out / "covers" / f"{v.stem}.jpg", fr, title, GENRE_RU.get(m["style_id"], m["style_id"].upper()), hue)
    picks = frames[::max(1, len(frames) // 4)][:4] or frames
    banner(out / "banner.jpg", picks)
    (out / "CHANNEL.md").write_text(CHANNEL_TEXTS)
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
