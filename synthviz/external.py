"""External audio -> Song: lets the SynthViz visuals run on a track made elsewhere
(e.g. Google Lyria via the Gemini API, see tools/gemini_music.py).

The composer knows every note it wrote; for an external mix we estimate the same
events from the audio: beat grid (tempo + phase), kick / snare / hat onsets from
the percussive part, bass and lead pitch lines (pYIN on band-limited harmonic
part), arp hits, chords per half bar (chroma templates), key (Krumhansl), and
sections (intro / build / drop / break / drop2 / outro) from bar energy.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .dsp import SR
from .composer import Song

NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _band(x, lo=None, hi=None, order=4):
    if lo and hi:
        sos = butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    elif lo:
        sos = butter(order, lo, btype="high", fs=SR, output="sos")
    else:
        sos = butter(order, hi, btype="low", fs=SR, output="sos")
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


def _onsets(lb, x, sr, delta, wait_s=0.08):
    env = lb.onset.onset_strength(y=x, sr=sr, hop_length=256)
    env = env / (env.max() + 1e-9)
    fr = lb.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=256, delta=delta,
                               wait=max(1, int(wait_s * sr / 256)), units="frames")
    t = lb.frames_to_time(fr, sr=sr, hop_length=256)
    return [float(a) for a in t], [float(env[f]) for f in fr]


def _notes_from_pitch(f0, voiced, times, min_dur, vel):
    midi = np.where(voiced, np.round(12 * np.log2(np.maximum(f0, 1e-3) / 440.0) + 69), np.nan)
    notes, cur, start = [], None, 0.0
    dt = times[1] - times[0]
    for i, m in enumerate(list(midi) + [np.nan]):
        if cur is not None and (np.isnan(m) or m != cur):
            t1 = times[i] if i < len(times) else times[-1] + dt
            if t1 - start >= min_dur:
                notes.append(dict(t=float(start), dur=float(t1 - start), p=int(cur), v=vel,
                                  slide=bool(notes and abs(notes[-1]["t"] + notes[-1]["dur"] - start) < 2 * dt),
                                  bend=0.0))
            cur = None
        if cur is None and not np.isnan(m):
            cur, start = m, times[i]
    return notes


def _salient_line(lb, sig, sr, lo_midi=60, n_oct=3, hop=256, min_dur=0.07):
    """Top-line melody from a band-limited harmonic signal: per frame, the
    strongest CQT semitone when it clearly sticks out of the frame (salience),
    held while it stays; pYIN is unreliable on a full mix, this is robust."""
    s22 = lb.resample(sig, orig_sr=SR, target_sr=sr)
    C = np.abs(lb.cqt(s22, sr=sr, hop_length=hop, fmin=lb.midi_to_hz(lo_midi), n_bins=12 * n_oct,
                      bins_per_octave=12))
    C = lb.decompose.nn_filter(C, aggregate=np.median, width=3) if C.shape[1] > 9 else C
    tt = lb.times_like(C, sr=sr, hop_length=hop)
    loud = C.max(axis=0)
    gate = np.percentile(loud, 35)
    top = C.argmax(axis=0)
    sal = C.max(axis=0) / (np.median(C, axis=0) + 1e-9)
    ok = (loud > gate) & (sal > 3.0)
    f0 = np.where(ok, lb.midi_to_hz(lo_midi + top), np.nan)
    notes = _notes_from_pitch(np.nan_to_num(f0, nan=1.0), ok, tt, min_dur, 0.6)
    for nn in notes:
        nn["v"] = float(np.clip(0.4 + 0.6 * loud[int(nn["t"] * sr / hop)] / (loud.max() + 1e-9), 0, 1))
    return notes


def _key(chroma_mean):
    best = None
    for r in range(12):
        for name, prof in (("major", _MAJ), ("minor", _MIN)):
            c = np.corrcoef(np.roll(prof, r), chroma_mean)[0, 1]
            if best is None or c > best[0]:
                best = (c, r, name)
    return best[1], best[2]


def _chord(ch):
    best = None
    for r in range(12):
        for q, iv in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            tpl = np.zeros(12); tpl[[(r + i) % 12 for i in iv]] = 1
            s = float(ch @ tpl) - 0.35 * float(ch @ (1 - tpl))
            if best is None or s > best[0]:
                best = (s, r, q, iv)
    _, r, q, iv = best
    pcs = sorted((r + i) % 12 for i in iv)
    base = 48 + r if 48 + r >= 53 else 60 + r          # root around F3..E4
    voicing = [base + iv[0], base + iv[1], base + iv[2], base + 12]
    return dict(name=NOTE_NAMES[r] + q, root=r, pcs=pcs, intervals=list(iv) + [12], voicing=voicing)


def song_from_audio(path, t0=0.0, t1=None, title="UNTITLED", style="future_bass", style_name=None,
                    bpm=None, fade_in=0.15, fade_out=1.2, log=print) -> Song:
    import librosa as lb

    y, _ = lb.load(path, sr=SR, mono=False, offset=t0, duration=None if t1 is None else t1 - t0)
    if y.ndim == 1:
        y = np.stack([y, y])
    x = y.T.astype(np.float32).copy()
    n = len(x)
    if fade_in:
        k = int(fade_in * SR); x[:k] *= np.linspace(0, 1, k)[:, None]
    if fade_out:
        k = int(fade_out * SR); x[-k:] *= np.linspace(1, 0, k)[:, None] ** 1.5
    peak = np.abs(x).max()
    if peak > 0.98:
        x *= 0.98 / peak
    dur = n / SR
    mono = x.mean(axis=1)

    # ---------- analysis at 22.05k
    sr = 22050
    m22 = lb.resample(mono, orig_sr=SR, target_sr=sr)
    tempo, beats = lb.beat.beat_track(y=m22, sr=sr, hop_length=256, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    if bpm is None:
        bpm = tempo
        while bpm < 85:
            bpm *= 2
        while bpm > 180:
            bpm /= 2
    spb = 60.0 / bpm
    # beat phase: robust median of beat times modulo spb
    ph = np.angle(np.exp(2j * np.pi * np.asarray(beats) / spb).mean()) / (2 * np.pi) * spb % spb
    log(f"[ext] {dur:.1f}s  tempo {tempo:.1f} -> {bpm:.1f} BPM  phase {ph:.3f}s")

    H, P = lb.effects.hpss(m22, margin=(1.0, 3.0))
    harm = lb.resample(H, orig_sr=sr, target_sr=SR)[:n]
    perc = lb.resample(P, orig_sr=sr, target_sr=SR)[:n]
    harm = np.pad(harm, (0, n - len(harm)))
    perc = np.pad(perc, (0, n - len(perc)))
    side = (x[:, 0] - x[:, 1]) * 0.5

    # ---------- drums
    kick_sig = _band(perc, hi=140)
    snare_sig = _band(perc, lo=180, hi=3000)
    hat_sig = _band(perc, lo=7000)
    kicks, kv = _onsets(lb, lb.resample(kick_sig, orig_sr=SR, target_sr=sr), sr, 0.12, 0.16)
    snares, sv = _onsets(lb, lb.resample(snare_sig, orig_sr=SR, target_sr=sr), sr, 0.14, 0.12)
    hats, hv = _onsets(lb, lb.resample(hat_sig, orig_sr=SR, target_sr=sr), sr, 0.10, 0.06)
    # keep strong ones only
    def keep(ts, vs, q):
        if not ts:
            return [], []
        th = np.percentile(vs, q)
        z = [(a, b) for a, b in zip(ts, vs) if b >= th]
        return [a for a, _ in z], [b for _, b in z]
    kicks, kv = keep(kicks, kv, 25)
    snares, sv = keep(snares, sv, 40)
    hats, hv = keep(hats, hv, 30)

    # ---------- bar energy -> sections
    bar = 4 * spb
    n_bars = int(np.ceil((dur - ph) / bar))
    rms = np.array([np.sqrt(np.mean(mono[int((ph + b * bar) * SR): int((ph + (b + 1) * bar) * SR)] ** 2) + 1e-12)
                    for b in range(n_bars)])
    low = _band(mono, hi=120)
    lrms = np.array([np.sqrt(np.mean(low[int((ph + b * bar) * SR): int((ph + (b + 1) * bar) * SR)] ** 2) + 1e-12)
                     for b in range(n_bars)])
    e = 0.5 * rms / rms.max() + 0.5 * lrms / lrms.max()
    hi_th = np.percentile(e, 55)
    lo_th = np.percentile(e, 30)
    lab = ["drop" if v >= hi_th else ("break" if v < lo_th else "mid") for v in e]
    # smooth labels in 2-bar blocks
    for b in range(1, n_bars - 1):
        if lab[b - 1] == lab[b + 1] != lab[b]:
            lab[b] = lab[b - 1]
    secs = []
    for b, l in enumerate(lab):
        if secs and secs[-1][0] == l:
            secs[-1][2] += 1
        else:
            secs.append([l, b, 1])
    out = []
    drops = 0
    for i, (l, b, nb) in enumerate(secs):
        nxt = secs[i + 1][0] if i + 1 < len(secs) else None
        if l == "drop":
            name = "drop" if drops == 0 else "drop2"; drops += 1
            en = 1.0
        elif nxt == "drop":
            name, en = "build", 0.55
        elif i == 0:
            name, en = "intro", 0.25
        elif i == len(secs) - 1:
            name, en = "outro", 0.2
        else:
            name, en = "break", 0.35
        t_ = 0.0 if not out else ph + b * bar
        out.append(dict(name=name, t=float(t_), dur=float(ph + (b + nb) * bar - t_), energy=en))
    # short in-between sections (< 2 bars) are fills, not sections: fold into the previous one
    for i in range(len(out) - 2, 0, -1):
        if out[i]["dur"] < 3 * bar - 1e-6 and out[i]["name"] in ("build", "break"):
            out[i - 1]["dur"] += out[i]["dur"]
            del out[i]
            if out[i]["name"] == out[i - 1]["name"] or (out[i]["name"] == "drop2" and out[i - 1]["name"] in ("drop", "drop2")):
                out[i - 1]["dur"] += out[i]["dur"]
                del out[i]
    # merge neighbours with the same name
    sections = []
    for s in out:
        if sections and sections[-1]["name"] == s["name"]:
            sections[-1]["dur"] += s["dur"]
        else:
            sections.append(s)
    sections[-1]["dur"] = dur - sections[-1]["t"]
    log("[ext] sections " + ", ".join(f"{s['name']}@{s['t']:.1f}" for s in sections))
    drop_starts = [s["t"] for s in sections if s["name"] in ("drop", "drop2")]

    # ---------- chords per half bar, key
    hop = 512
    chroma = lb.feature.chroma_cqt(y=lb.resample(harm, orig_sr=SR, target_sr=sr), sr=sr, hop_length=hop)
    ct = lb.times_like(chroma, sr=sr, hop_length=hop)
    tonic, mode = _key(chroma.mean(axis=1))
    chords = []
    step = bar / 2
    t_ = ph - step if ph > 0.05 else 0.0
    starts = [0.0] + list(np.arange(ph if ph > 0.05 else step, dur, step))
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else dur
        m = (ct >= a) & (ct < b)
        ch = chroma[:, m].mean(axis=1) if m.any() else np.ones(12)
        c = _chord(ch / (ch.max() + 1e-9))
        if chords and chords[-1]["name"] == c["name"]:
            chords[-1]["dur"] = float(b - chords[-1]["t"])
            continue
        c.update(t=float(a), dur=float(b - a))
        chords.append(c)
    chords = [{k: c[k] for k in ("t", "dur", "name", "root", "pcs", "intervals", "voicing")} for c in chords]
    prog = []
    for c in chords:
        if c["name"] not in prog:
            prog.append(c["name"])
        if len(prog) == 4:
            break
    key = f"{NOTE_NAMES[tonic]} {mode}"
    log(f"[ext] key {key}  chords {len(chords)}  prog {' - '.join(prog)}")

    # ---------- pitch lines
    def pitched(sig, fmin, fmax, min_dur, vel, hop_=512):
        s22 = lb.resample(sig, orig_sr=SR, target_sr=sr)
        f0, vf, vp = lb.pyin(s22, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_, frame_length=2048 if fmin > 60 else 4096)
        tt = lb.times_like(f0, sr=sr, hop_length=hop_)
        return _notes_from_pitch(f0, vf & (vp > 0.35), tt, min_dur, vel)

    bass_sig = _band(harm, lo=30, hi=220)
    lead_sig = _band(harm, lo=350, hi=4000)
    bass = pitched(bass_sig, 32.0, 200.0, 0.09, 0.9)
    for nn in bass:                       # octave errors -> fold into the sub register
        while nn["p"] > 47:
            nn["p"] -= 12
    lead = _salient_line(lb, lead_sig, sr)
    log(f"[ext] notes: lead {len(lead)}  bass {len(bass)}")

    # arp: bright harmonic onsets -> top chroma pitch at that moment
    arp_t, arp_v = _onsets(lb, lb.resample(_band(harm, lo=1500), orig_sr=SR, target_sr=sr), sr, 0.12, 0.06)
    arp = []
    for a, v in zip(arp_t, arp_v):
        j = min(chroma.shape[1] - 1, int(np.searchsorted(ct, a)))
        pc = int(np.argmax(chroma[:, j]))
        arp.append(dict(t=a, dur=0.06, p=72 + pc, v=float(min(1.0, 0.4 + v)), slide=False, bend=0.0))

    pad_lines = [[dict(t=c["t"], dur=c["dur"], p=c["voicing"][k], v=0.7, slide=False, bend=0.0) for c in chords]
                 for k in range(4)]

    # ---------- impacts / fx / fills / bass drops
    impacts = [t for t in drop_starts if t > 0.5]
    fx = [dict(kind="gap", t=max(0.0, t - spb), dur=spb * 0.9) for t in impacts]
    fills = []
    for s in sections:
        if s["name"] == "build" and s["dur"] >= 2 * spb:
            fills.append(dict(t=s["t"] + s["dur"] - 4 * spb, t1=s["t"] + s["dur"], type="buildroll", big=True,
                              section="build"))
    bass_drops = [dict(t=t, strength=1.0, p=None) for t in impacts]
    for nn in bass:
        sec = next((s for s in reversed(sections) if s["t"] <= nn["t"]), sections[0])
        if sec["name"] in ("drop", "drop2") and nn["dur"] > 0.25:
            bass_drops.append(dict(t=nn["t"], strength=0.6, p=nn["p"]))
    bass_drops.sort(key=lambda d: d["t"])
    thin = []
    for d in bass_drops:            # at most one sub shockwave per beat
        if not thin or d["t"] - thin[-1]["t"] >= spb * 0.95 or d["strength"] > thin[-1]["strength"]:
            thin.append(d)
    bass_drops = thin

    lanes = dict(kick=[[a, float(min(1.0, 0.5 + b))] for a, b in zip(kicks, kv)],
                 snare=[[a, float(min(1.0, 0.5 + b))] for a, b in zip(snares, sv)],
                 hat=[[a, float(min(1.0, 0.3 + b))] for a, b in zip(hats, hv)],
                 perc=[])

    # ---------- stems (approximate, used for envelopes and the oscilloscope)
    st2 = lambda s: np.stack([s, s], axis=1).astype(np.float32)
    stems = dict(lead=st2(lead_sig), counter=st2(lead_sig * 0.4), bass=st2(bass_sig),
                 pad=st2(_band(harm, lo=200, hi=2000) * 0.7), chords=st2(_band(harm, lo=200, hi=2000)),
                 arp=st2(_band(harm, lo=1500)), kick=st2(kick_sig), snare=st2(snare_sig), hats=st2(hat_sig),
                 perc=st2(_band(perc, lo=3000, hi=7000)), fx=st2(side), music=st2(harm))

    events = dict(bpm=round(bpm, 1), spb=spb, key=key, style=style_name or style.replace("_", " ").title(),
                  style_id=style, title=title, scale=mode, tonic=tonic, progression=prog, prog_name="detected",
                  lead_patch="external", alt_lead="external", bass_patch="external", pad_patch="external",
                  counter_patch="external", chords_patch="external", arp_patch="external", kick="external",
                  melody_char="external", glide_chords=False, kicks=kicks, snares=snares, hats=hats,
                  impacts=impacts, lanes=lanes, fx=fx, sections=sections, chords=chords, lead=lead, harm=[],
                  counter=[], bass=bass, arp=arp, pad_lines=pad_lines, glide_s=0.0, lead_glide=0.03,
                  duration=dur, loop_s=0.5, fills=fills, bass_drops=bass_drops, source=str(path),
                  source_window=[t0, t1])
    song = Song(style, title, round(bpm, 1), key, sections, chords, dur)
    song.audio = x
    song.stems = stems
    song.events = events
    return song
