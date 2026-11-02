"""Synth patches (subtractive / FM / supersaw / formant), mono-glide & poly
voices, and a fully synthetic drum machine + FX sounds."""
from __future__ import annotations

import copy
from dataclasses import replace as dc_replace
import numpy as np
from numba import njit

from . import dsp
from .dsp import SR
from .theory import Note


# ============================================================ patch definitions
def P(**kw):
    base = dict(
        oscs=[dict(wave="saw", level=1.0, semi=0, fine=0, voices=1, detune=0, pw=0.5, pwm=0.0, pwm_rate=0.5)],
        sub=0.0, noise=0.0,
        fm=None,              # dict(ratio, index, decay, feedback, sustain)
        formant=None,         # list of vowels for choir pads
        filter=dict(type="ladder", cutoff=2000, res=0.2, env=2.0, keytrack=0.5, drive=1.0,
                    lfo_rate=0.0, lfo_depth=0.0, lfo_sync=None),
        amp=(0.005, 0.2, 0.8, 0.2),
        fenv=(0.002, 0.25, 0.3, 0.2),
        glide=0.0, mono=False,
        vibrato=(5.5, 0.0, 0.3),
        pitch_env=(0.0, 0.05),
        drive=None, chorus=0.0, width=1.0, gain=1.0, pan=0.0,
        accent=0.0,           # extra filter env on accented notes (303)
        velo_filter=0.5,
    )
    if kw.get("fm") is not None and "oscs" not in kw:
        base["oscs"] = []
    for k, v in kw.items():
        if k == "filter":
            f = dict(base["filter"]); f.update(v); base["filter"] = f
        else:
            base[k] = v
    return base


def O(wave="saw", level=1.0, semi=0, fine=0, voices=1, detune=0, pw=0.5, pwm=0.0, pwm_rate=0.5, **extra):
    return dict(wave=wave, level=level, semi=semi, fine=fine, voices=voices, detune=detune, pw=pw, pwm=pwm, pwm_rate=pwm_rate, **extra)


PATCHES = {
    # ---------------------------------------------------------------- leads
    "supersaw_lead": P(oscs=[O("saw", 1.0, voices=7, detune=28), O("saw", 0.35, semi=12, voices=3, detune=18)],
                       filter=dict(type="ladder", cutoff=3800, res=0.15, env=1.5, keytrack=0.4),
                       amp=(0.004, 0.3, 0.85, 0.25), fenv=(0.001, 0.4, 0.4, 0.3), chorus=0.25, width=1.0, gain=0.55),
    "glide_lead": P(oscs=[O("saw", 0.8, voices=2, detune=9), O("square", 0.5, semi=-12, pw=0.45)],
                    filter=dict(type="ladder", cutoff=2600, res=0.35, env=1.8, keytrack=0.6),
                    amp=(0.01, 0.4, 0.8, 0.25), fenv=(0.005, 0.5, 0.45, 0.3),
                    glide=0.075, mono=True, vibrato=(5.2, 0.22, 0.35), gain=0.6),
    "smooth_lead": P(oscs=[O("tri", 0.8), O("saw", 0.35, voices=2, detune=7), O("sine", 0.4, semi=12)],
                     filter=dict(type="svf_lp", cutoff=2200, res=0.15, env=1.0, keytrack=0.6),
                     amp=(0.03, 0.4, 0.85, 0.4), glide=0.12, mono=True, vibrato=(4.8, 0.25, 0.4), gain=0.75),
    "chip_lead": P(oscs=[O("square", 1.0, pw=0.25, pwm=0.15, pwm_rate=3.0)], filter=dict(type=None),
                   amp=(0.001, 0.12, 0.6, 0.06), vibrato=(6.5, 0.18, 0.15), glide=0.025, mono=True, gain=0.32),
    "screech": P(oscs=[O("saw", 1.0, voices=5, detune=22), O("square", 0.6, semi=7, pw=0.3)],
                 filter=dict(type="svf_bp", cutoff=1800, res=0.55, env=2.2, keytrack=0.7,
                             lfo_sync=0.25, lfo_depth=0.8),
                 amp=(0.003, 0.1, 0.9, 0.12), fenv=(0.001, 0.15, 0.6, 0.1),
                 drive=(5.0, "asym"), glide=0.04, mono=True, gain=0.35),
    "hoover": P(oscs=[O("pulse", 1.0, voices=5, detune=35, pw=0.3, pwm=0.25, pwm_rate=0.7), O("saw", 0.5, semi=-12, voices=3, detune=25)],
                filter=dict(type="ladder", cutoff=2400, res=0.2, env=1.0), amp=(0.02, 0.3, 0.85, 0.3),
                pitch_env=(-7.0, 0.09), glide=0.09, mono=True, chorus=0.5, gain=0.4),
    "fm_lead": P(fm=dict(ratio=2.0, index=2.2, decay=0.25, sustain=0.4, feedback=0.0), filter=dict(type=None),
                 amp=(0.003, 0.3, 0.75, 0.3), glide=0.05, mono=True, vibrato=(5.5, 0.15, 0.3), gain=0.55),
    # ---------------------------------------------------------------- brass
    "synth_brass": P(oscs=[O("saw", 1.0, voices=3, detune=12), O("saw", 0.6, semi=-12, voices=2, detune=8)],
                     filter=dict(type="ladder", cutoff=700, res=0.1, env=2.6, keytrack=0.7),
                     amp=(0.04, 0.4, 0.85, 0.35), fenv=(0.07, 0.5, 0.55, 0.4), chorus=0.3, gain=0.45),
    "stab_brass": P(oscs=[O("saw", 1.0, voices=4, detune=15), O("pulse", 0.5, pw=0.35)],
                    filter=dict(type="ladder", cutoff=900, res=0.15, env=3.0, keytrack=0.6),
                    amp=(0.005, 0.25, 0.0, 0.15), fenv=(0.004, 0.18, 0.0, 0.12), chorus=0.2, gain=0.55),
    "poly_brass_80s": P(oscs=[O("pulse", 1.0, voices=2, detune=10, pw=0.4, pwm=0.2, pwm_rate=0.4), O("saw", 0.7, voices=2, detune=6)],
                        filter=dict(type="ladder", cutoff=600, res=0.2, env=2.8, keytrack=0.7),
                        amp=(0.03, 0.6, 0.8, 0.5), fenv=(0.05, 0.6, 0.5, 0.5), chorus=0.55, gain=0.45),
    # ---------------------------------------------------------------- pads
    "warm_pad": P(oscs=[O("saw", 1.0, voices=6, detune=22), O("tri", 0.6, semi=12, voices=2, detune=10)],
                  filter=dict(type="ladder", cutoff=1300, res=0.1, env=0.8, keytrack=0.3, lfo_rate=0.15, lfo_depth=0.35),
                  amp=(0.7, 1.0, 0.85, 1.6), fenv=(1.0, 1.5, 0.6, 1.5), chorus=0.6, gain=0.32),
    "glass_pad": P(fm=dict(ratio=3.0, index=1.2, decay=1.8, sustain=0.35, feedback=0.0),
                   oscs=[O("tri", 0.5, semi=12)], filter=dict(type="svf_lp", cutoff=6000, res=0.0, env=0.0),
                   amp=(0.5, 1.2, 0.8, 1.8), chorus=0.7, gain=0.32),
    "choir_pad": P(oscs=[O("saw", 1.0, voices=5, detune=16)], formant=["a", "o"],
                   filter=dict(type=None), amp=(0.6, 1.0, 0.9, 1.5), vibrato=(4.5, 0.08, 0.5), chorus=0.6, gain=0.55),
    "string_ens": P(oscs=[O("saw", 1.0, voices=4, detune=14), O("saw", 0.6, semi=12, voices=2, detune=10)],
                    filter=dict(type="svf_lp", cutoff=3200, res=0.05, env=0.3), amp=(0.35, 0.8, 0.85, 0.9),
                    vibrato=(5.0, 0.07, 0.4), chorus=0.8, gain=0.3),
    "space_drone": P(oscs=[O("saw", 0.7, voices=4, detune=30), O("sine", 0.8, semi=-12)],
                     filter=dict(type="ladder", cutoff=500, res=0.45, env=0.0, keytrack=0.2, lfo_rate=0.07, lfo_depth=1.2),
                     amp=(2.0, 2.0, 1.0, 3.0), chorus=0.7, gain=0.35),
    # ---------------------------------------------------------------- plucks / keys
    "pluck": P(oscs=[O("saw", 1.0, voices=3, detune=10), O("square", 0.4, semi=12)],
               filter=dict(type="ladder", cutoff=400, res=0.25, env=4.2, keytrack=0.6),
               amp=(0.001, 0.35, 0.0, 0.25), fenv=(0.001, 0.12, 0.0, 0.1), chorus=0.3, gain=0.5),
    "arp_pluck": P(oscs=[O("pulse", 1.0, pw=0.3), O("saw", 0.5, fine=8)],
                   filter=dict(type="ladder", cutoff=500, res=0.4, env=3.5, keytrack=0.5, lfo_rate=0.1, lfo_depth=0.6),
                   amp=(0.001, 0.18, 0.0, 0.12), fenv=(0.001, 0.09, 0.0, 0.08), gain=0.42),
    "bell_fm": P(fm=dict(ratio=3.5, index=3.0, decay=0.6, sustain=0.0, feedback=0.0), filter=dict(type=None),
                 amp=(0.001, 1.6, 0.0, 1.2), gain=0.42),
    "epiano_fm": P(fm=dict(ratio=1.0, index=1.6, decay=0.35, sustain=0.15, feedback=0.15), oscs=[O("sine", 0.3, semi=12)],
                   filter=dict(type=None), amp=(0.002, 1.4, 0.25, 0.6), chorus=0.45, gain=0.5),
    "marimba": P(fm=dict(ratio=4.0, index=1.0, decay=0.05, sustain=0.0, feedback=0.0), filter=dict(type=None),
                 amp=(0.001, 0.35, 0.0, 0.3), gain=0.6),
    # ---------------------------------------------------------------- bass
    "sub_bass": P(oscs=[O("sine", 1.0), O("tri", 0.15)], filter=dict(type=None), amp=(0.005, 0.2, 1.0, 0.08),
                  glide=0.06, mono=True, gain=0.9),
    "reese": P(oscs=[O("saw", 1.0, voices=2, detune=18), O("saw", 0.8, semi=-12, fine=-7), O("sine", 0.0)],
               sub=0.6, filter=dict(type="ladder", cutoff=600, res=0.25, env=0.6, keytrack=0.2, drive=1.8,
                                    lfo_sync=2.0, lfo_depth=1.1),
               amp=(0.01, 0.3, 0.95, 0.12), glide=0.09, mono=True, drive=(2.0, "tanh"), width=0.6, gain=0.42),
    "acid_303": P(oscs=[O("saw", 1.0)], filter=dict(type="ladder", cutoff=320, res=0.82, env=3.6, keytrack=0.4, drive=2.2),
                  amp=(0.002, 0.25, 0.75, 0.04), fenv=(0.001, 0.22, 0.0, 0.05), glide=0.055, mono=True,
                  accent=1.6, drive=(2.2, "tanh"), gain=0.42),
    "offbeat_bass": P(oscs=[O("saw", 1.0, voices=2, detune=8)], sub=0.5,
                      filter=dict(type="ladder", cutoff=500, res=0.2, env=2.5, keytrack=0.3),
                      amp=(0.002, 0.15, 0.5, 0.06), fenv=(0.001, 0.12, 0.1, 0.06), gain=0.55),
    "synthwave_bass": P(oscs=[O("saw", 1.0), O("square", 0.5, semi=-12, pw=0.5)],
                        filter=dict(type="ladder", cutoff=380, res=0.3, env=2.5, keytrack=0.4),
                        amp=(0.002, 0.18, 0.55, 0.06), fenv=(0.001, 0.13, 0.15, 0.08), gain=0.6),
    "wobble": P(oscs=[O("saw", 1.0, voices=3, detune=14), O("square", 0.6, semi=-12)], sub=0.6,
                filter=dict(type="ladder", cutoff=300, res=0.45, env=0.0, keytrack=0.2, drive=2.0, lfo_sync=0.5, lfo_depth=3.2),
                amp=(0.005, 0.2, 1.0, 0.08), glide=0.03, mono=True, drive=(2.5, "tanh"), gain=0.42),
    "fm_bass": P(fm=dict(ratio=1.0, index=2.5, decay=0.12, sustain=0.2, feedback=0.3), sub=0.5, filter=dict(type="ladder", cutoff=1500, res=0.1, env=0.5),
                 amp=(0.002, 0.25, 0.7, 0.06), mono=True, glide=0.03, gain=0.6),
}

VOWELS = {  # formant frequencies, gains, bandwidth-ish resonance
    "a": [(800, 1.0), (1150, 0.5), (2900, 0.25)],
    "e": [(400, 1.0), (1600, 0.35), (2700, 0.2)],
    "i": [(270, 1.0), (2100, 0.3), (3000, 0.2)],
    "o": [(450, 1.0), (800, 0.6), (2830, 0.15)],
    "u": [(325, 1.0), (700, 0.3), (2530, 0.1)],
}


# ================================================================== voice core
@njit(cache=True)
def _glide_curve(target, tau_samples):
    out = np.empty_like(target)
    y = target[0]
    a = np.exp(-1.0 / max(tau_samples, 1.0))
    for i in range(target.shape[0]):
        y = a * y + (1 - a) * target[i]
        out[i] = y
    return out


def _lfo(n, rate_hz, t0=0.0, shape="sine"):
    t = np.arange(n) / SR + t0
    ph = rate_hz * t
    if shape == "sine":
        return np.sin(2 * np.pi * ph)
    if shape == "tri":
        return 4 * np.abs(ph - np.floor(ph) - 0.5) - 1
    if shape == "saw":
        return 1 - 2 * (ph - np.floor(ph))
    raise ValueError(shape)


class Synth:
    """Renders Note lists with a patch. bpm needed for tempo-synced LFOs."""

    def __init__(self, patch, bpm=120.0, seed=0, **overrides):
        if isinstance(patch, str):
            patch = PATCHES[patch]
        self.p = copy.deepcopy(patch)
        for k, v in overrides.items():
            if k == "filter":
                self.p["filter"].update(v)
            else:
                self.p[k] = v
        self.bpm = bpm
        self.rng = np.random.default_rng(seed)

    # group notes: mono+slide notes are joined into legato groups
    def _groups(self, notes):
        notes = sorted(notes, key=lambda n: n.start)
        if not self.p["mono"]:
            return [[n] for n in notes]
        groups = []
        for n in notes:
            if groups and n.slide and n.start <= groups[-1][-1].end() + 0.26:
                groups[-1].append(n)
            else:
                # mono: cut previous group if overlapping
                if groups and groups[-1][-1].end() > n.start:
                    last = groups[-1][-1]
                    groups[-1][-1] = dc_replace(last, dur=max(0.05, n.start - last.start))
                groups.append([n])
        return groups

    def render(self, notes, total_s, transpose=0):
        out = np.zeros((int(total_s * SR) + SR, 2))
        spb = 60.0 / self.bpm
        for g in self._groups(notes):
            t0 = g[0].start * spb
            gate = (g[-1].end() - g[0].start) * spb
            sig = self._voice(g, t0, gate, spb, transpose)
            dsp.place(out, sig, t0, 1.0)
        out = out[: int(total_s * SR)]
        p = self.p
        if p["chorus"]:
            out = dsp.chorus(out, rate=0.45, depth_ms=3.0, base_ms=11, mix=p["chorus"])
        if p["width"] != 1.0:
            out = dsp.stereo_width(out, p["width"])
        if p["pan"]:
            out = dsp.pan(out, p["pan"])
        return out * p["gain"]

    def _voice(self, group, t0, gate_s, spb, transpose):
        p = self.p
        a, d, s, r = p["amp"]
        n_gate = max(1, int(gate_s * SR))
        n = n_gate + int(r * SR * 1.2) + 64
        # ---- pitch curve (semitones)
        target = np.full(n, float(group[0].pitch + transpose))
        for note in group[1:]:
            i = int((note.start - group[0].start) * spb * SR)
            target[i:] = note.pitch + transpose
        if len(group) > 1 and p["glide"] > 0:
            pitch = _glide_curve(target, p["glide"] * SR)
        else:
            pitch = target.copy()
        # per-note expression: scoops into the note, dives / rises at its tail
        for note in group:
            i0 = int((note.start - group[0].start) * spb * SR)
            i1 = min(n, i0 + int(note.dur * spb * SR))
            if note.scoop:
                L = min(int(0.07 * SR), i1 - i0)
                if L > 0:
                    pitch[i0:i0 + L] += note.scoop * (1 - np.linspace(0, 1, L)) ** 2
            if note.bend:
                L = max(1, int((i1 - i0) * 0.45))
                a_ = i1 - L
                pitch[a_:i1] += note.bend * np.linspace(0, 1, L) ** 2
                pitch[i1:] += note.bend if note is group[-1] else 0.0
        t = np.arange(n) / SR
        pl = p.get("pitch_lfo")
        if pl:
            rate, depth, synced = pl
            rate = 1.0 / (rate * spb) if synced else rate
            pitch = pitch + depth * _lfo(n, rate, t0)
        vr, vd, vdel = p["vibrato"]
        if vd:
            pitch = pitch + vd * np.clip((t - vdel) / 0.3, 0, 1) * np.sin(2 * np.pi * vr * t)
        pe, pdec = p["pitch_env"]
        if pe:
            pitch = pitch + pe * np.exp(-t / pdec)
        freq = 440.0 * 2 ** ((pitch - 69) / 12)
        vel = group[0].vel
        accent = any(nn.accent for nn in group[:1]) and p["accent"] > 0
        # ---- sources
        voices = []
        if p["fm"] is not None:
            fm = p["fm"]
            idx_env = fm["sustain"] + (1 - fm["sustain"]) * np.exp(-t / max(fm["decay"], 1e-3))
            mod = dsp.fm_op(freq, fm["ratio"], 0, feedback=fm.get("feedback", 0.0)) if fm.get("feedback") else \
                np.sin(2 * np.pi * dsp.phase_from_freq(freq * fm["ratio"]))
            if fm.get("lfo_sync"):   # growl: index wobble locked to tempo
                idx_env = idx_env * (0.3 + 0.7 * (0.5 + 0.5 * _lfo(n, 1.0 / (fm["lfo_sync"] * spb), t0)))
            if fm.get("ratio2"):
                mod = mod + fm.get("index2", 1.0) * np.sin(2 * np.pi * dsp.phase_from_freq(freq * fm["ratio2"])) * np.exp(-t / fm.get("decay2", 0.3))
            car = np.sin(2 * np.pi * dsp.phase_from_freq(freq) + fm["index"] * idx_env * vel * mod)
            voices.append(car)
        stereo_parts = []
        for o in p["oscs"]:
            nv = o["voices"]
            vs = []
            for k in range(nv):
                det = 0.0 if nv == 1 else (k / (nv - 1) - 0.5) * 2 * o["detune"]
                f = freq * 2 ** ((o["semi"] + (o["fine"] + det) / 100.0) / 12)
                ph0 = self.rng.random() if nv > 1 else 0.0
                if o["wave"] in ("pulse", "square"):
                    pw = o["pw"]
                    if o["pwm"]:
                        pw = np.clip(o["pw"] + o["pwm"] * np.sin(2 * np.pi * o["pwm_rate"] * (t + t0) + k), 0.05, 0.95)
                    ph = dsp.phase_from_freq(f, ph0)
                    v = dsp.osc_pulse(ph, f, pw)
                elif o["wave"] == "sync":
                    ratio = o.get("sync", 2.0) + o.get("sync_env", 0.0) * np.exp(-t / o.get("sync_decay", 0.3))
                    if o.get("sync_lfo"):
                        ratio = ratio + o["sync_lfo"] * (0.5 + 0.5 * _lfo(n, o.get("sync_lfo_rate", 0.3), t0))
                    pm = dsp.phase_from_freq(f, ph0)
                    sp = ratio * pm
                    sp = sp - np.floor(sp)
                    v = 2 * sp - 1
                elif o["wave"] == "fold":
                    ph = dsp.phase_from_freq(f, ph0)
                    amt = o.get("fold", 2.0) * (0.6 + 0.4 * np.exp(-t / 0.4))
                    v = np.sin(np.pi / 2 * amt * np.sin(2 * np.pi * ph))
                elif o["wave"] == "organ":
                    ph = dsp.phase_from_freq(f, ph0)
                    bars = o.get("drawbars", (0.6, 1.0, 0.7, 0.5, 0.3, 0.25))
                    v = sum(b * np.sin(2 * np.pi * ph * m) for b, m in zip(bars, (0.5, 1, 1.5, 2, 3, 4)) if b)
                    v = v / sum(bars)
                elif o["wave"] == "noise":
                    v = self.rng.standard_normal(n) * 0.5
                else:
                    v = dsp.osc(o["wave"], f, ph0)
                vs.append(v * o["level"] / np.sqrt(nv))
            stereo_parts.append(dsp.unison_stereo(vs, 0.9) if nv > 1 else np.stack([sum(vs)] * 2, axis=1) * 0.7071)
        mono_src = sum(voices) if voices else 0.0
        src = np.zeros((n, 2))
        for sp in stereo_parts:
            src += sp
        if voices:
            src += np.stack([mono_src, mono_src], axis=1) * 0.7071
        if p["sub"]:
            sub = np.sin(2 * np.pi * dsp.phase_from_freq(freq * 0.5)) * p["sub"] * 0.7
            src += sub[:, None]
        if p["noise"]:
            src += self.rng.standard_normal((n, 2)) * p["noise"] * 0.3
        # ---- formant (choir / vocal lead / talking bass)
        if p["formant"]:
            vow = p["formant"]
            fr = p.get("formant_rate", 0.11)
            if p.get("formant_sync"):
                fr = 1.0 / (p["formant_sync"] * spb)
            morph = 0.5 + 0.5 * np.sin(2 * np.pi * fr * (t + t0))
            outf = np.zeros_like(src)
            for (f1, g1), (f2, g2) in zip(VOWELS[vow[0]], VOWELS[vow[-1]]):
                fc = f1 * (1 - morph) + f2 * morph
                gg = g1 * (1 - morph) + g2 * morph
                for c in range(2):
                    outf[:, c] += dsp.svf(src[:, c], fc, 0.88, 1) * gg
            src = outf * 1.4
        # ---- filter
        flt = p["filter"]
        if flt.get("type"):
            fa, fd, fs, fr = p["fenv"]
            fenv = dsp.adsr(n_gate, n, fa, fd, fs, fr)
            amt = flt["env"] * (1 - p["velo_filter"] + p["velo_filter"] * vel)
            if accent:
                amt += p["accent"]
            octs = amt * fenv + flt["keytrack"] * (pitch - 60) / 12
            if flt.get("lfo_sync"):
                rate = 1.0 / (flt["lfo_sync"] * spb)
                octs = octs + flt["lfo_depth"] * (0.5 + 0.5 * _lfo(n, rate, t0, "sine")) - flt["lfo_depth"] * 0.5
            elif flt.get("lfo_rate"):
                octs = octs + flt["lfo_depth"] * _lfo(n, flt["lfo_rate"], t0)
            cutoff = flt["cutoff"] * 2 ** octs
            res = flt["res"] + (0.08 if accent else 0)
            for c in range(2):
                x = np.ascontiguousarray(src[:, c])
                if flt["type"] == "ladder":
                    src[:, c] = dsp.ladder(x, cutoff, res, flt.get("drive", 1.0))
                else:
                    mode = {"svf_lp": 0, "svf_bp": 1, "svf_hp": 2}[flt["type"]]
                    src[:, c] = dsp.svf(x, cutoff, res, mode) * (1.8 if mode == 1 else 1.0)
        # ---- amp
        if p.get("amp_curve") == "reverse":
            # swells up through the note and stops dead (hardstyle reverse bass)
            x = np.clip(t / max(gate_s, 1e-3), 0, 1)
            env = (x ** 2.2) * (t < gate_s) + 0.0
            env = env * np.minimum(1, (gate_s - t).clip(0) / 0.006 + (t >= gate_s))
            env[t >= gate_s] = 0.0
        else:
            env = dsp.adsr(n_gate, n, a, d, s, r)
        if accent:
            env = env * 1.25
        y = src * (env * (0.35 + 0.65 * vel))[:, None]
        if p["drive"]:
            amt, kind = p["drive"]
            y = dsp.drive(y, amt, kind) * 0.7
        return y


# ================================================================== drums
class Drums:
    """Synthesised drum kit. Every hit is generated from scratch (with small
    random variation per hit so loops breathe)."""

    def __init__(self, seed=0, tune_hz=55.0):
        self.rng = np.random.default_rng(seed)
        self.tune = tune_hz
        self._cache = {}

    # ---------------------------------------------------------------- kicks
    def kick(self, kind="909", vel=1.0):
        rng = self.rng
        if kind == "909":
            n = int(0.45 * SR); t = np.arange(n) / SR
            f = 52 + 180 * np.exp(-t / 0.03) + 60 * np.exp(-t / 0.004)
            body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22)
            click = dsp.svf(rng.standard_normal(n), np.full(n, 3500.0), 0.2, 1) * np.exp(-t / 0.004) * 0.6
            y = dsp.drive(body + click, 1.6)
        elif kind == "808":
            n = int(1.2 * SR); t = np.arange(n) / SR
            f = self.tune + 90 * np.exp(-t / 0.04)
            y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.55)
            y = dsp.drive(y, 1.3)
        elif kind == "punch":   # dnb / breaks
            n = int(0.32 * SR); t = np.arange(n) / SR
            f = 58 + 240 * np.exp(-t / 0.018)
            body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.13)
            click = dsp.svf(rng.standard_normal(n), np.full(n, 5000.0), 0.3, 1) * np.exp(-t / 0.003)
            y = dsp.drive(body * 1.2 + click * 0.5, 2.2)
        elif kind == "hard":    # hardstyle: pitched distorted tail tuned to key
            n = int(0.62 * SR); t = np.arange(n) / SR
            f = self.tune + 900 * np.exp(-t / 0.006) + 180 * np.exp(-t / 0.035)
            ph = np.cumsum(f) / SR
            body = np.sin(2 * np.pi * ph) * (np.exp(-t / 0.5) * (t < 0.5) + np.exp(-(t - 0.5) / 0.03) * (t >= 0.5))
            y = dsp.drive(body, 9.0, "asym")
            y = dsp.svf(y, 2200 * np.exp(-t / 0.25) + 300, 0.3, 0)
            y = dsp.drive(y * 2.5, 2.0)
            click = rng.standard_normal(n) * np.exp(-t / 0.002) * 0.5
            y = y + click
        elif kind == "soft":
            n = int(0.35 * SR); t = np.arange(n) / SR
            f = 50 + 70 * np.exp(-t / 0.03)
            y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18)
        else:
            raise ValueError(kind)
        y = y / (np.max(np.abs(y)) + 1e-9)
        fade = np.minimum(1, (len(y) - np.arange(len(y))) / (0.01 * SR))
        return y * fade * vel

    # ---------------------------------------------------------------- snares
    def snare(self, kind="909", vel=1.0):
        rng = self.rng
        if kind == "909":
            n = int(0.35 * SR); t = np.arange(n) / SR
            tone = (np.sin(2 * np.pi * np.cumsum(185 + 60 * np.exp(-t / 0.01)) / SR) * 0.6 +
                    np.sin(2 * np.pi * np.cumsum(330 + 40 * np.exp(-t / 0.01)) / SR) * 0.4) * np.exp(-t / 0.07)
            nz = dsp.svf(rng.standard_normal(n), np.full(n, 6000.0), 0.1, 2) * np.exp(-t / 0.13)
            y = tone * 0.8 + nz * 0.9
        elif kind == "dnb":
            n = int(0.3 * SR); t = np.arange(n) / SR
            tone = np.sin(2 * np.pi * np.cumsum(240 + 140 * np.exp(-t / 0.008)) / SR) * np.exp(-t / 0.05)
            nz = rng.standard_normal(n)
            nz = dsp.svf(nz, np.full(n, 4500.0), 0.3, 1) * 0.8 + dsp.svf(nz, np.full(n, 8000.0), 0.0, 2) * 0.5
            y = dsp.drive(tone * 1.0 + nz * np.exp(-t / 0.09), 2.5)
        elif kind == "gated":   # 80s snare; gated reverb added at mix
            n = int(0.4 * SR); t = np.arange(n) / SR
            tone = np.sin(2 * np.pi * np.cumsum(200 + 80 * np.exp(-t / 0.01)) / SR) * np.exp(-t / 0.08)
            nz = dsp.svf(rng.standard_normal(n), np.full(n, 3000.0), 0.1, 2) * np.exp(-t / 0.18)
            y = tone * 0.8 + nz
        elif kind == "rim":
            n = int(0.08 * SR); t = np.arange(n) / SR
            y = (np.sin(2 * np.pi * 1700 * t) + dsp.svf(rng.standard_normal(n), np.full(n, 3200.0), 0.6, 1)) * np.exp(-t / 0.012)
        else:
            raise ValueError(kind)
        y = y / (np.max(np.abs(y)) + 1e-9)
        return y * vel

    def clap(self, vel=1.0):
        rng = self.rng
        n = int(0.4 * SR); t = np.arange(n) / SR
        nz = dsp.svf(rng.standard_normal(n), np.full(n, 1300.0), 0.45, 1)
        env = np.zeros(n)
        for k, off in enumerate([0.0, 0.009, 0.019, 0.028]):
            i = int(off * SR)
            env[i:] += np.exp(-(t[i:] - off) / (0.008 if k < 3 else 0.14)) * (0.8 if k < 3 else 1.0)
        y = nz * env
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    # ---------------------------------------------------------------- metals
    def _metal(self, n):
        t = np.arange(n) / SR
        freqs = [205.3, 304.4, 369.6, 522.7, 540.0, 800.0]
        y = np.zeros(n)
        for f in freqs:
            y += np.sign(np.sin(2 * np.pi * f * 1.7 * t + self.rng.random() * 6))
        return y

    def hat(self, open_=False, vel=1.0, kind="808"):
        dec = 0.32 if open_ else 0.035 + self.rng.uniform(-0.005, 0.01)
        n = int((dec * 4 + 0.02) * SR); t = np.arange(n) / SR
        if kind == "808":
            src = self._metal(n) * 0.6 + self.rng.standard_normal(n) * 0.5
        else:
            src = self.rng.standard_normal(n)
        y = dsp.svf(src, np.full(n, 9500.0), 0.25, 1) * 0.8 + dsp.svf(src, np.full(n, 7000.0), 0.0, 2) * 0.5
        y = y * np.exp(-t / dec)
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    def ride(self, vel=1.0):
        n = int(1.2 * SR); t = np.arange(n) / SR
        src = self._metal(n) + self.rng.standard_normal(n) * 0.3
        y = dsp.svf(src, np.full(n, 6000.0), 0.4, 1) * np.exp(-t / 0.45)
        y += np.sin(2 * np.pi * 3100 * t) * 0.15 * np.exp(-t / 0.3)
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    def crash(self, vel=1.0):
        n = int(2.4 * SR); t = np.arange(n) / SR
        src = self._metal(n) * 0.4 + self.rng.standard_normal(n)
        y = dsp.svf(src, np.full(n, 5000.0), 0.1, 2) * (np.exp(-t / 0.9) * 0.8 + np.exp(-t / 0.05) * 0.5)
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    def tom(self, pitch_hz=110, vel=1.0):
        n = int(0.5 * SR); t = np.arange(n) / SR
        f = pitch_hz * (1 + 0.6 * np.exp(-t / 0.03))
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22)
        y += dsp.svf(self.rng.standard_normal(n), np.full(n, 2000.0), 0.2, 1) * np.exp(-t / 0.01) * 0.3
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    def shaker(self, vel=1.0):
        n = int(0.09 * SR); t = np.arange(n) / SR
        env = (t / 0.02).clip(0, 1) * np.exp(-np.maximum(t - 0.02, 0) / 0.025)
        y = dsp.svf(self.rng.standard_normal(n), np.full(n, 8000.0), 0.3, 2) * env
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    def cowbell(self, vel=1.0):
        n = int(0.3 * SR); t = np.arange(n) / SR
        y = dsp.osc_pulse(dsp.phase_from_freq(np.full(n, 540.0)), np.full(n, 540.0)) + dsp.osc_pulse(dsp.phase_from_freq(np.full(n, 800.0)), np.full(n, 800.0))
        y = dsp.svf(y, np.full(n, 900.0), 0.5, 1) * np.exp(-t / 0.08)
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    def zap(self, vel=1.0, f0=3000, f1=80, dur=0.25):
        n = int(dur * SR); t = np.arange(n) / SR
        f = f1 + (f0 - f1) * np.exp(-t / (dur / 5))
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (dur / 2.5))
        return y * vel

    # ---------------------------------------------------------------- FX
    def riser(self, seconds, vel=1.0, tonal_hz=None):
        n = int(seconds * SR); t = np.arange(n) / SR
        x = t / seconds
        nz = self.rng.standard_normal(n)
        cut = 300 * (40 ** (x ** 1.6))
        y = dsp.svf(nz, cut, 0.55, 1) * (x ** 1.5)
        if tonal_hz:
            f = tonal_hz * 2 ** (x * 2)
            y += dsp.osc_saw(dsp.phase_from_freq(f), f) * 0.15 * x ** 2
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    def downlifter(self, seconds=2.0, vel=1.0):
        n = int(seconds * SR); t = np.arange(n) / SR
        x = t / seconds
        cut = 9000 * (0.02 ** x)
        y = dsp.svf(self.rng.standard_normal(n), cut, 0.5, 1) * (1 - x) ** 1.5
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    def impact(self, vel=1.0):
        n = int(3.0 * SR); t = np.arange(n) / SR
        boom = np.sin(2 * np.pi * np.cumsum(38 + 80 * np.exp(-t / 0.06)) / SR) * np.exp(-t / 0.9)
        nz = dsp.svf(self.rng.standard_normal(n), 4000 * np.exp(-t / 0.4) + 200, 0.2, 0) * np.exp(-t / 0.5)
        y = dsp.drive(boom * 1.3 + nz * 0.5, 1.8)
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    def reverse_cymbal(self, seconds=1.5, vel=1.0):
        c = self.crash()[: int(seconds * SR)][::-1]
        return c * vel

    def vinyl(self, seconds):
        n = int(seconds * SR)
        y = self.rng.standard_normal(n) * 0.02
        pops = self.rng.random(n) < 0.0004
        y[pops] += self.rng.uniform(-0.6, 0.6, pops.sum())
        return dsp.svf(y, np.full(n, 5000.0), 0.0, 0)
