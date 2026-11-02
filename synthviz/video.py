"""Render pipeline v4: song -> features -> director (one world, cuts on section
lines, arcade inserts where the music runs) -> note ribbon -> camera rig ->
post -> musical frame FX -> kinetic captions -> loop back to frame 0 -> ffmpeg.

See README 'Visuals v4' for the music -> picture map."""
from __future__ import annotations

import json
import math
import pickle
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .composer import compose, save_wav
from .visuals import kernels as K
from .visuals.features import Features
from .visuals.scenes import SCENES as S1
from .visuals.scenes2 import Terrain, City, Pixel, Waves, Galaxy
from .visuals.director import Plan
from .visuals.notes import NoteRibbon
from .visuals.camera import CameraRig
from .visuals.inserts import ArcadeLayer
from .visuals.captions import Captions, OUTRO_LINES

SCENES = dict(S1)
SCENES.update({c.name: c for c in (Terrain, City, Pixel, Waves, Galaxy)})
from .visuals.scenes3 import SCENES3  # noqa: E402  aurora, lava, ocean, knot, petals
SCENES.update(SCENES3)
PIXELATE = {"chiptune": 6}
HUE_RANGE = 0.14          # how far chords may push the world's hue
STYLE_NAMES = {"dnb_liquid": "liquid dnb", "dnb_energy": "drum & bass", "ambient_space": "space ambient",
               "future_bass": "future bass"}


def pixelate(frame, block, levels):
    H, W = frame.shape[:2]
    hh, ww = H // block, W // block
    sm = frame[: hh * block, : ww * block].reshape(hh, block, ww, block, 3).mean(axis=(1, 3))
    sm = np.round(sm / 255 * (levels - 1)) / (levels - 1) * 255
    frame[: hh * block, : ww * block] = np.repeat(np.repeat(sm, block, 0), block, 1).astype(np.uint8)


def wave_warp(frame, t, amp):
    H, W = frame.shape[:2]
    y = np.arange(H)
    off = (amp * np.sin(y / 38.0 + t * 9.0) + amp * 0.5 * np.sin(y / 11.0 - t * 5.0)).astype(np.int64)
    cols = (np.arange(W)[None, :] + off[:, None]) % W
    frame[:] = frame[y[:, None], cols]


def _ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def render_video(style="synthwave", seed=None, out="out.mp4", scene=None, alt_scene=None, fps=30,
                 W=1080, H=1920, overlay=True, preview_times=None, song_kw=None, crf=20, log=print,
                 cta_effect=None, hud=True, world=None, inserts=None, insert_mode="drops", outro=None,
                 segments=None, song=None):
    """segments: [(t0, t1), ...] -> render only those ranges, each to its own
    <out>_segN.mp4 with audio (fast previews before a long render)."""
    t0 = time.time()
    seed = seed if seed is not None else random.randrange(10 ** 6)
    if song is None:
        song = compose(style, seed=seed, **(song_kw or {}))
    ev = song.events
    log(f"[song] {song.title} | {ev['style']} | {ev['key']} | {song.bpm} BPM | {' - '.join(ev['progression'])} | "
        f"{song.duration:.1f}s [{time.time() - t0:.1f}s]")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wav = out.with_suffix(".wav")
    save_wav(wav, song.audio)
    F = Features(song, fps)
    rng = random.Random(seed)
    plan = Plan(F, style, rng, world=world, inserts=inserts, insert_mode=insert_mode,
                force_scene=scene)
    outro = outro or cta_effect or rng.choice(["question", "rate", "replay", "question"])
    if outro not in OUTRO_LINES:
        outro = "question"
    log(f"[plan] world={plan.world} hue={plan.hue:.2f} shots={[(round(a, 1), s, tr) for a, s, tr in plan.shots]}")
    log(f"[plan] inserts={[(g, round(a, 1), round(b, 1)) for g, a, b, _ in plan.inserts]} outro={outro}")
    names = sorted({s for _, s, _ in plan.shots})
    scenes = {}
    for n in names:
        sc = SCENES[n](F, W, H, rng)
        sc.base_hue = (plan.hue - HUE_RANGE / 2) % 1.0
        sc.hue_range = HUE_RANGE
        scenes[n] = sc
    ribbon = NoteRibbon(F, W, H) if hud else None
    arcade = ArcadeLayer(F, plan.build_inserts(seed), W, H)
    cam = CameraRig(F, plan.cuts)
    sub = f"{STYLE_NAMES.get(style, style.replace('_', ' '))}  ·  {song.bpm} bpm  ·  {ev['key'].lower()}"
    caps = Captions(F, W, H, song.title, plan.hue, ribbon=ribbon, outro=outro, subtitle=sub) if overlay else None

    img = np.zeros((H, W, 3), np.float32)
    img2 = np.zeros_like(img)
    small = np.zeros((H // 4, W // 4, 3), np.float32)
    frame = np.zeros((H, W, 3), np.uint8)
    pix = PIXELATE.get(style, 0)
    spb = F.ev["spb"]
    cross_d = 2 * spb
    import colorsys
    tint = np.array(colorsys.hsv_to_rgb(plan.hue, 0.55, 1.0), np.float32)
    tint /= float(tint @ np.array([0.3, 0.55, 0.15], np.float32))
    grade = 0.42
    bass_drops = sorted(ev.get("bass_drops") or [], key=lambda b: b["t"])                       # half-bar dissolve
    cuts = plan.cuts

    # ------------------------------------------------ which frames
    capture = set()
    seg_of = {}
    procs = {}
    if preview_times is not None:
        frames = [0]
        for tt in preview_times:
            e = min(F.n_frames - 1, int(tt * fps))
            frames += list(range(max(1, e - int(1.2 * fps)), e + 1))
            capture.add(e)
        frames = sorted(set(frames))
    elif segments:
        frames = [0]
        for k, (a, b) in enumerate(segments):
            ia, ib = int(a * fps), min(F.n_frames, int(b * fps))
            frames += list(range(max(1, ia - int(1.2 * fps)), ib))
            for j in range(ia, ib):
                seg_of[j] = k
        frames = sorted(set(frames))
        for k, (a, b) in enumerate(segments):
            p = out.with_name(f"{out.stem}_seg{k + 1}.mp4")
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                   "-r", str(fps), "-i", "-", "-ss", f"{int(a * fps) / fps:.4f}", "-t", f"{(int(b * fps) - int(a * fps)) / fps:.4f}",
                   "-i", str(wav), "-c:v", "libx264", "-preset", "fast", "-crf", str(crf + 2), "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(p)]
            procs[k] = (subprocess.Popen(cmd, stdin=subprocess.PIPE), p)
    else:
        frames = list(range(F.n_frames))
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(fps), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out)]
        procs[0] = (subprocess.Popen(cmd, stdin=subprocess.PIPE), out)

    dt = 1.0 / fps
    previews = []
    tr0 = time.time()
    frame0 = None
    echo = None
    loop_d = 0.5
    prev_i = -1
    for n_, i in enumerate(frames):
        t = i / fps
        step = dt * max(1, i - prev_i) if prev_i >= 0 else dt
        prev_i = i
        sec = F.section(t)["name"]
        in_drop = sec in ("drop", "drop2")
        # ---------------- world: current shot (+ dissolve into the next one)
        st, sname, _ = plan.scene_at(t)
        sc = scenes[sname]
        sc.render(img, i, t, step)
        hue = sc.hue or plan.hue
        for tc, kind in cuts:
            if kind == "cross" and tc - cross_d / 2 <= t < tc + cross_d / 2:
                other = plan.scene_at(tc - 1e-3 if t >= tc else tc + 1e-3)[1]
                if other != sname:
                    scenes[other].render(img2, i, t, step)
                    x = _ease((t - (tc - cross_d / 2)) / cross_d)
                    w_new = x if t >= tc else x       # weight of the shot after the cut
                    if t >= tc:
                        img[:] = img * w_new + img2 * (1 - w_new)
                    else:
                        img[:] = img * (1 - w_new) + img2 * w_new
        # ---------------- world grade: pull every scene toward the world's palette
        if grade > 0:
            lum = img[..., 0] * 0.3 + img[..., 1] * 0.55 + img[..., 2] * 0.15
            img *= (1 - grade)
            img += lum[..., None] * (tint * grade)
        # ---------------- sub hits (events["bass_drops"]): a shockwave rises from the floor
        for bd in bass_drops:
            age = t - bd["t"]
            if 0 <= age < 0.7:
                u = age / 0.7
                R = 80 + H * 0.75 * (1 - (1 - u) ** 2)
                ang = np.linspace(math.pi * 1.05, math.pi * 1.95, 160)
                xs = (W / 2 + np.cos(ang) * R * 1.25).astype(np.float64)
                ys = (H + 60 + np.sin(ang) * R).astype(np.float64)
                cr, cg, cb = colorsys.hsv_to_rgb((hue + 0.5) % 1.0, 0.5, 1.0)
                K.polyline(img, xs, ys, cr, cg, cb, 2.2 * bd.get("strength", 1.0) * (1 - u) ** 1.5, 3.0 + 6 * u)
        # ---------------- notes + arcade (both live in the linear image -> get bloom)
        cta_fade = 1.0 - _ease((t - caps.outro_t) / 0.6) if caps else 1.0
        if caps is None and t >= F.sections[-1]["t"]:
            cta_fade = 1.0 - _ease((t - F.sections[-1]["t"]) / 0.6)
        arc_p = arcade.presence(t)
        if ribbon:
            ribbon.draw(img, i, t, hue, min(1.0, t / 0.8) * cta_fade * (1 - 0.3 * arc_p))
        arcade.draw(img, t, hue)
        # ---------------- post
        kick = F.pulse("kick", t, 0.10)
        snare = F.pulse("snare", t, 0.10)
        impact = F.pulse("impact", t, 0.35)
        energy = float(F.energy[i])
        K.downsample4(img, small)
        b1 = gaussian_filter(small, sigma=(3, 3, 0))
        b2 = gaussian_filter(small, sigma=(12, 12, 0))
        zoom, sx, sy = cam.at(i)
        expo = 1.45 + 0.8 * impact + (0.3 * snare if in_drop else 0.0) + 0.25 * energy
        gap = F.fx_active(t, "gap")
        if gap is not None:
            expo *= 0.3 + 0.7 * (1 - gap) ** 2
        sw = F.fx_active(t, "switch")
        for tc, kind in cuts:
            if kind == "flash" and 0 <= t - tc < 0.3:
                expo *= 1 + 5 * (1 - (t - tc) / 0.3) ** 2
        K.finalize(img, b1, b2, frame, zoom, 0.006 + 0.012 * kick + 0.03 * impact, 0.55 + 0.45 * energy,
                   0.06, 0.36, 0.03, float(i % 97), expo, sx, sy, 1.12 + 0.25 * energy)
        # ---------------- musical moments -> frame effects (kept, but no jitter)
        cr = F.fx_active(t, "crush")
        if cr is not None:
            pixelate(frame, 12 if cr < 0.5 else 8, 5)
        dv = F.fx_active(t, "dive")
        if dv is not None:
            wave_warp(frame, t, 3 + 24 * math.sin(math.pi * dv))
        hp = F.fx_active(t, "hpdive")
        if hp is not None:
            # high-pass dive: the bottom (bass) of the frame drains away, image floats up
            k_ = math.sin(math.pi * hp)
            ramp = np.linspace(1.0, 1.0 - 0.75 * k_, H, dtype=np.float32)[:, None, None]
            lift = int(18 * k_)
            if lift:
                frame[:-lift] = frame[lift:].copy()
            frame[:] = (frame * ramp).astype(np.uint8)
        stt = F.fx_active(t, "stutter")
        if stt is not None:
            e_ = next(e for e in F.fx if e["kind"] == "stutter" and e["t"] <= t < e["t"] + e["dur"])
            third = min(2, int(stt * 3))
            sl = spb / [2, 4, 8][third]
            k0 = e_["t"] + third * e_["dur"] / 3
            ph = ((t - k0) % sl) / sl
            if echo is None or ph < dt / sl:
                echo = frame.copy()
            # echo trails: the slice start keeps ghosting under the live frame
            frame[:] = (frame * 0.6 + echo * 0.4 * (1 - ph) + frame * 0.4 * ph).astype(np.uint8)
        else:
            echo = None
        ts_ = F.fx_active(t, "tapestop")
        if ts_ is not None:
            g_ = frame.mean(axis=2, keepdims=True)
            frame[:] = (frame * (1 - ts_) + g_ * ts_).astype(np.uint8)
            droop = int(ts_ ** 2 * H * 0.06)
            if droop:
                frame[droop:] = frame[:-droop].copy(); frame[:droop] = 0
        if pix and sname != "pixel":
            hh, ww = H // pix, W // pix
            smf = frame[: hh * pix, : ww * pix].reshape(hh, pix, ww, pix, 3).mean(axis=(1, 3))
            frame[: hh * pix, : ww * pix] = np.repeat(np.repeat(smf, pix, 0), pix, 1).astype(np.uint8)
        if caps:
            caps.draw(frame, i, t, hue, min(1.0, cta_fade + 0.0))
        # ---------------- seamless loop: the last half second melts into frame 0
        if i == 0:
            frame0 = frame.copy()
        if frame0 is not None and t > F.dur - loop_d:
            x = _ease((t - (F.dur - loop_d)) / loop_d)
            frame[:] = (frame * (1 - x) + frame0 * x).astype(np.uint8)
        # ---------------- output
        if preview_times is not None:
            if i in capture:
                previews.append(frame.copy())
        elif segments:
            k = seg_of.get(i)
            if k is not None:
                procs[k][0].stdin.write(frame.tobytes())
                if seg_of.get(i + 1) != k:      # segment finished -> close it now
                    procs[k][0].stdin.close()
        else:
            procs[0][0].stdin.write(frame.tobytes())
        if n_ % 150 == 0 and n_:
            el = time.time() - tr0
            log(f"  frame {n_}/{len(frames)}  {el / n_:.3f}s/frame")
    outs = []
    for k, (p, path) in procs.items():
        if not p.stdin.closed:
            p.stdin.close()
        p.wait(); outs.append(path)
    meta = dict(seed=seed, style_id=style, outro=outro, **plan.describe(),
                **{k: v for k, v in ev.items() if k in ("title", "bpm", "key", "progression", "prog_name", "lead_patch",
                                                         "bass_patch", "pad_patch", "counter_patch", "chords_patch",
                                                         "arp_patch", "kick", "melody_char", "glide_chords", "duration",
                                                         "style")})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"[done] {', '.join(map(str, outs)) or 'previews'}  total {time.time() - t0:.1f}s")
    if preview_times is not None:
        return previews
    return outs if segments else out
