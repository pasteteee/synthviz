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
    "ambient_space": "SPACE AMBIENT", "future_bass": "FUTURE BASS", "rawstyle": "RAWSTYLE", "heavy_bass": "BASS HOUSE", "jerk": "JERK",
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


# ---------------------------------------------------------------- brand
CHANNEL_NAME = "RAVE.EXE"
TAGLINE = "SYNTH MUSIC YOU CAN SEE  •  HARDSTYLE · DNB · SYNTHWAVE"

# the mascot "Bit" (same robot that runs and dances in the videos), 24x24 pixel head
BIT_HEAD = [
    "...........aa...........",
    "...........aa...........",
    "..........a..a..........",
    "...........aa...........",
    "...........bb...........",
    "....tttttttttttttttt....",
    "...tttttttttttttttttt...",
    "..hhbbbbbbbbbbbbbbbbhh..",
    ".hhhbbbbbbbbbbbbbbbbhhh.",
    ".hhhbbddddddddddddbbhhh.",
    ".hhhbdddddddddddddddbhhh",
    ".hhhbddeeeddddeeeddbhhh.",
    ".hhhbddeeeddddeeeddbhhh.",
    ".hhhbdddddddddddddddbhhh",
    ".hhhbbddddddddddddbbhhh.",
    "..hhbbbbbbbbbbbbbbbbhh..",
    "....bbbbbbmmmmbbbbbb....",
    "....bbbbbbbbbbbbbbbb....",
    ".....bbbbbbbbbbbbbb.....",
    ".......cccccccccc.......",
    "......cccccccccccc......",
]


def _pixels(rows, cols, cell, W, H, ox, oy):
    """Draw ASCII pixel art (cols: char -> rgb) into a float image of size (H, W)."""
    img = np.zeros((H, W, 3), np.float32)
    for j, r in enumerate(rows):
        for i, ch in enumerate(r):
            if ch in cols:
                x0, y0 = int(ox + i * cell), int(oy + j * cell)
                img[max(0, y0):max(0, y0 + cell - 2), max(0, x0):max(0, x0 + cell - 2)] = cols[ch]
    return img


def avatar(path, size=1024):
    """Bit's head with headphones on a neon stage: readable even at 48 px in a circle."""
    S = size
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    cx, cy = S / 2, S / 2
    r = np.hypot(x - cx, y - cy) / S
    ang = np.arctan2(y - cy, x - cx)
    img = np.zeros((S, S, 3), np.float32)
    img += hsv(0.78, 0.85, 0.22) * np.exp(-r * 3.0)[..., None]
    # sun-burst rays behind the head (rave lights)
    rays = (0.5 + 0.5 * np.cos(ang * 12)) ** 6 * np.exp(-r * 2.2)
    img += (hsv(0.88, 0.7, 1.0) * 0.30)[None, None] * rays[..., None]
    # equaliser ring
    ring = np.zeros((S, S), np.float32)
    rng = np.random.default_rng(11)
    nb = 48
    lv = 0.5 + 0.5 * np.abs(np.sin(np.arange(nb) * 0.7)) * rng.uniform(0.6, 1.0, nb)
    k = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    frac = ((ang + np.pi) / (2 * np.pi) * nb) % 1
    r0 = 0.37
    ring = ((r > r0) & (r < r0 + 0.07 * lv[k]) & (frac > 0.18) & (frac < 0.82)).astype(np.float32)
    rc = hsv(0.52, 0.8, 1.0)[None, None] * (1 - (r[..., None] - r0) * 6) + hsv(0.88, 0.8, 1.0)[None, None] * (r[..., None] - r0) * 6
    img += glow(ring[..., None] * rc, (4, 16, 40), (1.0, 0.6, 0.4)) * 0.9
    # the mascot
    cell = int(S * 0.026)
    cols = {"a": hsv(0.13, 0.8, 1.0) * 1.6, "b": hsv(0.40, 0.18, 1.0) * 1.15, "t": hsv(0.85, 0.8, 1.0) * 1.3,
            "h": hsv(0.85, 0.75, 0.95) * 1.2, "d": hsv(0.75, 0.6, 0.06), "e": hsv(0.30, 0.85, 1.0) * 2.4,
            "m": hsv(0.85, 0.8, 1.0) * 1.2, "c": hsv(0.40, 0.18, 0.75)}
    w_ = len(BIT_HEAD[0]) * cell; h_ = len(BIT_HEAD) * cell
    head = _pixels(BIT_HEAD, cols, cell, S, S, (S - w_) / 2, (S - h_) / 2 - S * 0.01)
    m = (head.sum(2) > 0)[..., None]
    halo = gaussian_filter(m[..., 0].astype(np.float32), S * 0.03)
    img = img * (1 - 0.8 * np.clip(halo * 2, 0, 1))[..., None]
    img = img * (1 - m) + head + glow(head * 0.25, (6, 24), (0.6, 0.4))
    img *= np.clip(1.3 - r * 1.7, 0, 1)[..., None] ** 0.5
    Image.fromarray(tonemap(img)).resize((800, 800), Image.LANCZOS).save(path)


def _dancer_frame(t, hue, seed):
    """One frame of the Dancer insert (same character as in the videos) as an RGB float array."""
    from synthviz.visuals.inserts import Dancer, InsertCfg, CW, CH
    from synthviz.visuals.pixelkit import Canvas

    class _F:
        pass
    f = _F(); spb = 0.4
    f.ev = dict(spb=spb, lead=[dict(t=i * 0.2, p=60 + (i * 5) % 12, dur=0.2) for i in range(200)])
    f.kicks = [i * spb for i in range(100)]; f.snares = [i * spb * 2 + spb for i in range(50)]
    f.hats = [i * spb + spb / 2 for i in range(100)]; f.fps = 30; f.energy = np.ones(3000, np.float32)

    def pulse(kind, tt, tau=0.12):
        d = [tt - x for x in {"kick": f.kicks, "snare": f.snares, "hat": f.hats}[kind] if x <= tt]
        return float(np.exp(-min(d) / tau)) if d else 0.0
    f.pulse = pulse
    d = Dancer(f, 0, 20, InsertCfg(hue=hue, title=" "), seed=seed)
    cv = Canvas(CW, CH); cv.clear(); d.draw(cv, t)
    return np.clip(cv.a, 0, 1.5)


def banner(path, frames=None):
    """Stage with Bit dancing left and right of the name; text stays inside the 1546x423 safe area."""
    W, H = 2560, 1440
    yy0 = np.linspace(0, 1, H)[:, None, None]
    canvas = (hsv(0.78, 0.8, 0.10) * (1 - yy0) + hsv(0.86, 0.8, 0.04) * yy0) * np.ones((1, W, 1), np.float32)
    PW = 980                                   # each dancer panel; centred dancers sit outside the text area
    ph = int(PW * 150 / 188)
    for k, (t, hue, x0) in enumerate([(2.35, 0.80, -40), (3.15, 0.55, W - PW + 40)]):
        fr = _dancer_frame(t, hue, 3 + k)
        up = np.asarray(Image.fromarray((np.clip(fr / 1.5, 0, 1) * 255).astype(np.uint8)).resize((PW, ph), Image.NEAREST)).astype(np.float32) / 255 * 1.5
        xs = np.linspace(-1, 1, PW)
        fade = np.clip((1 - np.abs(xs)) * 3.0, 0, 1)[None, :, None] * np.clip(np.linspace(0, 1, ph) * 4, 0, 1)[:, None, None]
        y0 = H - ph
        xa, xb = max(0, x0), min(W, x0 + PW)
        reg = canvas[y0:, xa:xb]
        sub, fsub = up[:, xa - x0:xb - x0], fade[:, xa - x0:xb - x0]
        canvas[y0:, xa:xb] = reg * (1 - fsub) + sub * fsub
    yy = np.linspace(-1, 1, H)[:, None]; xx = np.linspace(-1, 1, W)[None, :]
    band = np.exp(-(yy / 0.30) ** 4) * np.exp(-(xx / 0.48) ** 8)
    canvas *= (1 - 0.6 * band)[..., None]
    tl = text_layer((W, H), [(CHANNEL_NAME, OSWALD, 250, (W / 2, H / 2 - 40), "mm", (1, 1, 1))])
    sl = text_layer((W, H), [(TAGLINE, INTER, 42, (W / 2, H / 2 + 125), "mm", (1, 1, 1))])
    xx01 = np.linspace(0, 1, W)[None, :, None]
    col = hsv(0.88, 0.8, 1.0) * (1 - xx01) + hsv(0.52, 0.8, 1.0) * xx01
    canvas = canvas * (1 - tl) + tl * (0.7 + 0.6 * col) + glow(tl * col, (5, 20, 60), (1.0, 0.7, 0.45))
    canvas = canvas * (1 - sl) + sl * 0.95
    Image.fromarray(tonemap(canvas * 1.05)).save(path, quality=93)


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
    sl = text_layer((W, H), [(CHANNEL_NAME, INTER, 40, (W / 2, 1560), "mm", (1, 1, 1))])
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
               "Subscribe for more drops 💜",
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


CHANNEL_TEXTS = """# RAVE.EXE: channel kit (international)

## Name and handles
**RAVE.EXE** · YouTube `@RaveExe` (backups `@RaveExeMusic`, `@rave.exe.music`) · TikTok / Instagram `@rave.exe`

Mascot: **Bit**, the little robot in headphones who runs, shoots and dances through every video.

## Channel description (YouTube → Customise channel → Basic info)
```
Synth music you can see. Hosted by Bit, a tiny robot who can't stop dancing.

Every track on this channel is generated in code: synths, 808s, hardstyle kicks, reese and growl basses, chords and melodies. The visuals are driven by the real notes. Kicks move the camera, the melody runs across the neon note lane, chords change the colours, and every drop throws you into a new world: a neon highway, deep space, an arcade game or a rave where Bit dances to the drop.

Hardstyle · Rawstyle · Drum & Bass · Liquid DnB · Future Bass · Bass House · Synthwave · Trance · Acid · Chiptune

Tell me in the comments which genre should drop next.
```

## Short bio (TikTok / Instagram, max 80 chars)
```
Synth music you can see 🎹 hardstyle · dnb · synthwave · made in code
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
