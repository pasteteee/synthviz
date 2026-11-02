---
name: synthviz-shorts
description: Generate unique electronic-music YouTube Shorts with the SynthViz engine (pasteteee/synthviz) — new sound, new hook, new visual world every time, plus English titles, descriptions, tags, covers and channel kit.
---

# SynthViz Shorts

Makes channel-ready vertical music videos (1080×1920, 30 fps, ~40–60 s) from scratch: the track is composed and
mixed in code, the visuals are driven by its real notes, and every video is checked against the channel history
so sound, hook and look never repeat. All on-screen text and all YouTube texts are in **English** (international channel).

## 1. Setup (once per machine)
```bash
git clone https://github.com/pasteteee/synthviz && cd synthviz
bash install_deps.sh            # apt: ffmpeg, LSP plugins; pip: requirements.txt
bash third_party/surge/install.sh   # Surge XT presets (recommended; or SURGE_PREBUILT=<dir with lib/ and data/>)
export PYTHONPATH=.
```
The channel history lives in `channel/history.json` inside the repo. Pull before rendering and commit it after,
otherwise the uniqueness check cannot see earlier videos.

## 2. Make N videos
```bash
python release.py --n 10 --out release/$(date +%F) --history channel/history.json
```
What it guarantees per video:
- **Different sound:** Surge XT presets for lead/chords/pads/plucks (curated pools, never reused on the channel); style rotates (no two neighbours alike, least-used first); per-track random patch character
  (detune, filter, envelopes, octave layers), random hardstyle-kick flavour, random drum-kit layers (TR-808, Kickmess).
  A fingerprint (style + lead + bass + kick + key + progression) must be new.
- **Different melody:** a 1–2 bar hook from genre rhythm cells, repeated with a changed ending (ABAB′ / AAAB),
  slides, scoops and delay throws; its interval+rhythm signature must be new.
- **In tune:** chord-scale aware writing + `fix_harmony`; videos with < 90 % in-chord-scale notes are re-rolled.
- **Different look:** `visuals.variety.plan_batch` gives each video a new world (neon_drive, deep_space, arcade,
  liquid, rave, canyon), a new pair of mini-games and a different outro, avoiding neighbours and the history.

Fast dry run (audio only, ~40 s per track): add `--audio-only`. One video takes ~9 min on 4 cores.

## 3. Outputs (in `--out`)
- `NNN_style_seed.mp4` + `.json` (all musical and visual choices, texts)
- `UPLOAD.md`: per video — title, description, tags, pinned comment (copy-paste)
- `kit/avatar.png`, `kit/banner.jpg`, `kit/covers/*.jpg`, `kit/CHANNEL.md` (channel description, keywords, settings)

## 4. Before delivering
1. Grab 2–3 frames per video (`ffmpeg -ss T -i v.mp4 -frames:v 1 f.png`) and look at them: no black frames, text readable.
2. Check the log line per video: LUFS around −7…−9, `in-key ≥ 0.9`.
3. Commit `channel/history.json` so the next batch stays unique.
4. Say honestly that the audio was checked by measurement, not by listening.

## 5. Tuning knobs
- `--styles hardstyle,dnb_energy,future_bass` restricts genres; `--seed` makes a batch reproducible.
- Single video: `python -m synthviz --style hardstyle --seed 21 --out out/hs.mp4 --world rave --inserts runner,bounce`.
- Mix levels per genre: `composer.LEVELS`, loudness: `composer.LUFS_TARGET`; texts: `release.py` (HOOKS, TAGS, QUESTIONS).
