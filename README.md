# SynthViz

Generative electronic music + music-driven vertical video for YouTube Shorts, TikTok and Reels.
One command composes a track (harmony, hooks, basses, drums, mix and master), renders a 1080×1920 video whose
visuals follow the real notes, and writes English titles, descriptions, tags and covers for the channel.

```bash
bash install_deps.sh && export PYTHONPATH=.
python release.py --n 10 --out release/batch1          # 10 unique Shorts + UPLOAD.md + channel kit
python -m synthviz --style hardstyle --seed 21 --out out/hs.mp4 --world rave --inserts runner,bounce
python -m synthviz --style future_bass --audio-only --out out/fb.wav
python -m synthviz --list                               # styles, patches, scales, progressions
```

## Sound
- **Styles:** hardstyle, rawstyle, heavy_bass (bass house), dnb_energy, dnb_liquid, future_bass, synthwave, trance, acid, chiptune, ambient_space.
- **Harmony:** keys and modes, genre progressions, chord-scale aware melodies, `fix_harmony` safety pass.
- **Hooks:** 1–2 bar motifs from genre rhythm cells, repeated with a changed ending (ABAB′/AAAB), slides, scoops, delay throws; counter-melody an octave below.
- **Bass:** clean sub + separately distorted mid layer (reese, growl, wobble, neuro, square), 808 with glide, hardstyle kick v2 (euphoric / classic / raw / modern punchy) tuned to every chord.
- **Drums:** synthesized kits layered with CC0 samples (Sonic Pi, TR-808, Kickmess); fills, build rolls, impacts.
- **Mix:** LUFS-S auto-gain per genre, per-bus sidechain curves, OTT, tape saturation, parallel crunch, EQ map, LSP limiter master (−1 dBTP).
- **Variety:** every track randomizes patch character, kick flavour and drum layers.

## Surge XT layer (optional, recommended)
With Surge XT installed, lead, alt lead, chords, pads, plucks and counter-melodies are played by one of ~2,000 curated
Surge presets (`third_party/surge/curated_presets.json`, pools per role and genre), picked per track and never repeated
across the channel. Install: `bash third_party/surge/install.sh` (builds `surgepy` from source, ~20–30 min) or
`SURGE_PREBUILT=/path/with/lib_and_data bash third_party/surge/install.sh`. Without it the built-in synths are used.
Surge XT is GPL-3; its binaries and data are not stored in this repo.

## Visuals
One "world" per video (neon_drive, deep_space, arcade, liquid, rave, canyon), mini-games in the drops (runner, bounce,
breakout, shooter), a note lane that shows every part, a camera driven by kicks, sub drops and fills, and a seamless loop
(audio and picture). See `docs/VISUALS_V4.md`.

## Uniqueness
`release.py` keeps `channel/history.json` and rejects any track whose sound fingerprint or hook signature was used
before; `visuals/variety.py` avoids repeating worlds, game pairs and outros.

## Docs
- `docs/SOUND_RESEARCH.md`: instruments, mixing, distortion, plugins and kits research.
- `docs/VISUALS_V4.md`: visual research and the music → image catalogue.
- `skill/SKILL.md`: the Claude skill that runs the whole pipeline.

Samples are CC0 (see `synthviz/samples/CREDITS.md`).

## External tracks (Gemini / Lyria)
Visuals can also follow a track made elsewhere: `synthviz/external.py` estimates the beat grid, kicks/snares/hats,
bass and lead lines, chords, key and sections from the audio, so the whole visual pipeline runs unchanged.
```bash
GEMINI_API_KEY=... python tools/gemini_music.py --model lyria-3.5 --out music/track \
    --prompt "Energetic instrumental future bass, F minor, big supersaw drop, no vocals"
python -m synthviz --audio music/track_0.mp3 --window 31.5-90.6 --style future_bass --title "SKYLINE PULSE" --out out/v.mp4
```
