"""Extended sound library: ~50 extra synth patches and a bigger drum kit
(hardstyle / rawstyle / gabber kicks with tuned distorted tails, trap, lo-fi,
techno, breakbeat, latin percussion, vocal hits)."""
from __future__ import annotations

import math
import numpy as np

from . import dsp
from .dsp import SR
from .instruments import PATCHES, Drums, P, O

NEW = {
    # ------------------------------------------------------------ leads
    "sync_lead": P(oscs=[O("sync", 1.0), O("saw", 0.4, fine=6)], filter=dict(type="ladder", cutoff=3500, res=0.2, env=1.2),
                   amp=(0.003, 0.3, 0.85, 0.2), glide=0.06, mono=True, vibrato=(5.5, 0.18, 0.3), gain=0.42),
    "laser_lead": P(oscs=[dict(O("sync", 1.0), sync=1.5, sync_env=5.0, sync_decay=0.18), O("square", 0.35, semi=-12)],
                    filter=dict(type="ladder", cutoff=4200, res=0.25, env=1.0), amp=(0.002, 0.3, 0.8, 0.15),
                    glide=0.05, mono=True, gain=0.38),
    "pwm_lead": P(oscs=[O("pulse", 1.0, voices=2, detune=8, pw=0.35, pwm=0.25, pwm_rate=4.0)],
                  filter=dict(type="ladder", cutoff=3000, res=0.3, env=1.4), amp=(0.005, 0.3, 0.8, 0.2), glide=0.07,
                  mono=True, vibrato=(5.0, 0.2, 0.3), gain=0.5),
    "whistle": P(oscs=[O("sine", 1.0), O("tri", 0.15, semi=12)], noise=0.04, filter=dict(type="svf_bp", cutoff=2500, res=0.2, env=0.5, keytrack=1.0),
                 amp=(0.04, 0.3, 0.9, 0.25), glide=0.09, mono=True, vibrato=(5.8, 0.3, 0.25), gain=0.7),
    "vox_lead": P(oscs=[O("saw", 1.0, voices=3, detune=10)], formant=["a", "i"], formant_rate=0.6,
                  filter=dict(type=None), amp=(0.02, 0.3, 0.85, 0.25), glide=0.08, mono=True, vibrato=(5.2, 0.25, 0.25), gain=0.55),
    "trance_lead": P(oscs=[O("saw", 1.0, voices=7, detune=34), O("saw", 0.4, semi=12, voices=3, detune=20)],
                     filter=dict(type="ladder", cutoff=2600, res=0.2, env=2.2), amp=(0.002, 0.22, 0.55, 0.3),
                     fenv=(0.001, 0.2, 0.3, 0.3), chorus=0.3, gain=0.5),
    "future_lead": P(oscs=[O("saw", 1.0, voices=5, detune=20), O("square", 0.4, voices=2, detune=10)],
                     filter=dict(type="ladder", cutoff=3200, res=0.15, env=1.0), amp=(0.005, 0.3, 0.8, 0.2),
                     pitch_env=(-2.0, 0.05), glide=0.06, mono=True, chorus=0.3, gain=0.42),
    "mini_lead": P(oscs=[O("saw", 1.0), O("saw", 0.8, fine=-7), O("square", 0.6, semi=-12)],
                   filter=dict(type="ladder", cutoff=1600, res=0.45, env=2.2, drive=1.6), amp=(0.003, 0.3, 0.8, 0.2),
                   glide=0.1, mono=True, vibrato=(5.0, 0.15, 0.4), gain=0.5),
    "gliss_lead": P(oscs=[O("saw", 0.9, voices=3, detune=12), O("tri", 0.6, semi=12)],
                    filter=dict(type="ladder", cutoff=2400, res=0.25, env=1.2), amp=(0.01, 0.3, 0.9, 0.3),
                    glide=0.22, mono=True, vibrato=(4.5, 0.3, 0.35), chorus=0.35, gain=0.5),
    "acid_lead": P(oscs=[O("saw", 1.0)], filter=dict(type="ladder", cutoff=700, res=0.75, env=3.0, drive=2.0),
                   amp=(0.002, 0.2, 0.7, 0.05), fenv=(0.001, 0.18, 0.1, 0.05), glide=0.05, mono=True, accent=1.4,
                   drive=(1.8, "tanh"), gain=0.42),
    "sine_lead": P(oscs=[O("sine", 1.0), O("sine", 0.2, semi=19)], filter=dict(type=None), amp=(0.01, 0.2, 0.9, 0.3),
                   glide=0.08, mono=True, vibrato=(5.5, 0.2, 0.3), drive=(1.5, "tanh"), gain=0.7),
    "raw_screech": P(oscs=[O("saw", 1.0, voices=4, detune=30), O("sync", 0.7)],
                     filter=dict(type="svf_bp", cutoff=1400, res=0.65, env=2.0, keytrack=0.6, lfo_sync=0.125, lfo_depth=1.4),
                     amp=(0.002, 0.1, 0.95, 0.08), fenv=(0.001, 0.12, 0.5, 0.1), drive=(8.0, "asym"),
                     pitch_lfo=(0.25, 0.4, True), glide=0.03, mono=True, gain=0.3),
    "chip_tri": P(oscs=[O("tri", 1.0)], filter=dict(type=None), amp=(0.001, 0.1, 0.8, 0.04), vibrato=(6.0, 0.15, 0.2),
                  glide=0.02, mono=True, gain=0.6),
    "chip_pulse12": P(oscs=[O("pulse", 1.0, pw=0.125)], filter=dict(type=None), amp=(0.001, 0.08, 0.7, 0.04),
                      vibrato=(7.0, 0.2, 0.12), glide=0.015, mono=True, gain=0.28),
    # ------------------------------------------------------------ brass
    "horn_section": P(oscs=[O("saw", 1.0, voices=4, detune=14), O("pulse", 0.4, pw=0.4)],
                      filter=dict(type="ladder", cutoff=500, res=0.1, env=2.4, keytrack=0.8),
                      amp=(0.06, 0.4, 0.9, 0.3), fenv=(0.09, 0.4, 0.6, 0.3), vibrato=(5.0, 0.12, 0.35), chorus=0.4, gain=0.42),
    "trumpet_synth": P(oscs=[O("pulse", 1.0, pw=0.3), O("saw", 0.6, fine=5)],
                       filter=dict(type="ladder", cutoff=600, res=0.15, env=3.0, keytrack=0.8),
                       amp=(0.02, 0.3, 0.85, 0.2), fenv=(0.04, 0.3, 0.55, 0.2), pitch_env=(-1.0, 0.04),
                       vibrato=(5.5, 0.2, 0.3), glide=0.04, mono=True, gain=0.5),
    "brass_stab_hard": P(oscs=[O("saw", 1.0, voices=5, detune=22), O("square", 0.5, semi=-12)],
                         filter=dict(type="ladder", cutoff=1200, res=0.2, env=2.5), amp=(0.002, 0.2, 0.0, 0.12),
                         fenv=(0.002, 0.15, 0.0, 0.1), drive=(3.0, "tanh"), chorus=0.2, gain=0.45),
    "jump_brass": P(oscs=[O("pulse", 1.0, voices=3, detune=12, pw=0.25), O("saw", 0.6, voices=2, detune=8)],
                    filter=dict(type="ladder", cutoff=1500, res=0.2, env=1.6), amp=(0.003, 0.25, 0.4, 0.12),
                    fenv=(0.002, 0.2, 0.2, 0.1), chorus=0.3, gain=0.45),
    "obx_brass": P(oscs=[O("saw", 1.0, voices=2, detune=9), O("saw", 0.9, semi=7, voices=2, detune=7)],
                   filter=dict(type="ladder", cutoff=700, res=0.25, env=2.4), amp=(0.03, 0.5, 0.75, 0.4),
                   fenv=(0.05, 0.5, 0.45, 0.4), chorus=0.5, gain=0.38),
    # ------------------------------------------------------------ pads
    "glide_pad": P(oscs=[O("saw", 1.0, voices=5, detune=18), O("tri", 0.7, semi=12, voices=2, detune=8)],
                   filter=dict(type="ladder", cutoff=1700, res=0.15, env=0.4, keytrack=0.4, lfo_rate=0.12, lfo_depth=0.3),
                   amp=(0.25, 0.8, 0.9, 0.9), glide=0.28, mono=True, vibrato=(4.2, 0.08, 0.6), chorus=0.6, gain=0.32),
    "shimmer_pad": P(oscs=[O("saw", 0.8, voices=4, detune=16), O("saw", 0.5, semi=12, voices=3, detune=20), O("sine", 0.4, semi=24)],
                     filter=dict(type="svf_lp", cutoff=4500, res=0.1, env=0.3, lfo_rate=0.2, lfo_depth=0.4),
                     amp=(1.0, 1.0, 0.9, 2.0), chorus=0.8, gain=0.26),
    "dark_pad": P(oscs=[O("saw", 1.0, voices=4, detune=25), O("square", 0.5, semi=-12, voices=2, detune=12)],
                  filter=dict(type="ladder", cutoff=450, res=0.35, env=0.6, lfo_rate=0.08, lfo_depth=0.8),
                  amp=(0.8, 1.0, 0.9, 1.5), chorus=0.5, gain=0.4),
    "pwm_strings": P(oscs=[O("pulse", 1.0, voices=3, detune=10, pw=0.4, pwm=0.3, pwm_rate=0.8)],
                     filter=dict(type="svf_lp", cutoff=3500, res=0.05, env=0.3), amp=(0.3, 0.8, 0.85, 0.8),
                     chorus=0.9, gain=0.38),
    "choir_ooh": P(oscs=[O("saw", 1.0, voices=5, detune=14)], formant=["o", "u"], formant_rate=0.07,
                   filter=dict(type=None), amp=(0.5, 1.0, 0.9, 1.4), vibrato=(4.8, 0.1, 0.5), chorus=0.6, gain=0.55),
    "noise_pad": P(oscs=[O("noise", 1.0), O("sine", 0.3)], filter=dict(type="svf_bp", cutoff=1200, res=0.85, env=0.0, keytrack=1.0,
                   lfo_rate=0.1, lfo_depth=0.6), amp=(1.0, 1.0, 1.0, 1.8), chorus=0.5, gain=0.6),
    "crystal_pad": P(fm=dict(ratio=5.0, index=0.9, decay=2.5, sustain=0.25, feedback=0.0), oscs=[O("sine", 0.4, semi=12)],
                     filter=dict(type=None), amp=(0.3, 1.5, 0.7, 2.0), chorus=0.8, gain=0.3),
    "evolving_pad": P(oscs=[O("pulse", 0.8, voices=3, detune=14, pw=0.3, pwm=0.25, pwm_rate=0.25), O("saw", 0.6, voices=3, detune=18)],
                      filter=dict(type="ladder", cutoff=900, res=0.3, env=0.6, lfo_rate=0.06, lfo_depth=1.3),
                      amp=(1.2, 1.0, 0.95, 2.0), chorus=0.7, gain=0.32),
    # ------------------------------------------------------------ keys / plucks / mallets
    "rhodes": P(fm=dict(ratio=1.0, index=1.2, decay=0.6, sustain=0.1, feedback=0.0, ratio2=14.0, index2=0.6, decay2=0.03),
                filter=dict(type=None), amp=(0.002, 1.8, 0.2, 0.5), chorus=0.5, gain=0.5),
    "organ": P(oscs=[O("organ", 1.0)], filter=dict(type=None), amp=(0.01, 0.1, 1.0, 0.08), chorus=0.6,
               pitch_lfo=(6.5, 0.04, False), gain=0.45),
    "clav": P(oscs=[O("pulse", 1.0, pw=0.15)], filter=dict(type="svf_bp", cutoff=900, res=0.4, env=2.0),
              amp=(0.001, 0.25, 0.0, 0.08), fenv=(0.001, 0.12, 0.0, 0.06), gain=0.6),
    "kalimba": P(fm=dict(ratio=7.0, index=0.8, decay=0.04, sustain=0.0, feedback=0.0), oscs=[O("sine", 0.3, semi=12)],
                 filter=dict(type=None), amp=(0.001, 0.7, 0.0, 0.5), gain=0.55),
    "harp_pluck": P(oscs=[O("tri", 1.0), O("saw", 0.3, semi=12)], filter=dict(type="svf_lp", cutoff=1200, res=0.1, env=2.5),
                    amp=(0.001, 1.0, 0.0, 0.8), fenv=(0.001, 0.15, 0.0, 0.2), gain=0.55),
    "music_box": P(oscs=[O("sine", 1.0), O("sine", 0.25, semi=24), O("sine", 0.12, semi=31)], filter=dict(type=None),
                   amp=(0.001, 1.2, 0.0, 1.0), gain=0.5),
    "future_pluck": P(oscs=[O("saw", 1.0, voices=5, detune=24)], filter=dict(type="ladder", cutoff=600, res=0.2, env=3.8),
                      amp=(0.001, 0.3, 0.0, 0.3), fenv=(0.001, 0.15, 0.0, 0.12), chorus=0.4, gain=0.42),
    "steel_pluck": P(fm=dict(ratio=2.0, index=2.0, decay=0.08, sustain=0.0, feedback=0.2), filter=dict(type=None),
                     amp=(0.001, 0.5, 0.0, 0.3), gain=0.5),
    "house_piano": P(fm=dict(ratio=1.0, index=1.8, decay=0.25, sustain=0.05, feedback=0.1, ratio2=4.0, index2=0.5, decay2=0.05),
                     oscs=[O("tri", 0.2, semi=12)], filter=dict(type=None), amp=(0.001, 0.9, 0.15, 0.3), gain=0.5),
    "vox_chop": P(oscs=[O("saw", 1.0, voices=2, detune=6)], formant=["a", "o"], formant_rate=4.0, filter=dict(type=None),
                  amp=(0.003, 0.12, 0.3, 0.06), gain=0.7),
    # ------------------------------------------------------------ bass
    "reverse_bass": P(oscs=[O("saw", 1.0, voices=2, detune=12), O("sine", 0.9, semi=-12)],
                      filter=dict(type="ladder", cutoff=900, res=0.15, env=0.0, keytrack=0.3, drive=2.5),
                      amp_curve="reverse", amp=(0.001, 0.1, 1.0, 0.005), drive=(6.0, "asym"), gain=0.5),
    "moog_bass": P(oscs=[O("saw", 1.0), O("saw", 0.9, fine=-6), O("square", 0.7, semi=-12)],
                   filter=dict(type="ladder", cutoff=280, res=0.35, env=2.4, drive=1.8), amp=(0.002, 0.25, 0.7, 0.08),
                   fenv=(0.001, 0.18, 0.15, 0.1), glide=0.04, mono=True, gain=0.6),
    "pluck_bass": P(oscs=[O("square", 1.0, pw=0.4), O("sine", 0.8)], filter=dict(type="ladder", cutoff=250, res=0.3, env=3.0),
                    amp=(0.001, 0.22, 0.0, 0.05), fenv=(0.001, 0.08, 0.0, 0.05), gain=0.65),
    "808_bass": P(oscs=[O("sine", 1.0), O("tri", 0.1)], filter=dict(type=None), pitch_env=(4.0, 0.03),
                  amp=(0.002, 1.2, 0.6, 0.2), glide=0.08, mono=True, drive=(2.2, "tanh"), gain=0.75),
    "growl_bass": P(fm=dict(ratio=1.5, index=4.0, decay=10.0, sustain=1.0, feedback=0.3, lfo_sync=0.5), sub=0.6,
                    formant=["a", "o"], formant_sync=0.5, filter=dict(type=None), amp=(0.005, 0.2, 1.0, 0.06),
                    glide=0.04, mono=True, drive=(3.0, "tanh"), gain=0.4),
    "neuro_bass": P(oscs=[dict(O("sync", 1.0), sync=2.0, sync_lfo=3.0, sync_lfo_rate=2.0), O("fold", 0.6)], sub=0.7,
                    filter=dict(type="ladder", cutoff=900, res=0.4, env=0.5, drive=2.5, lfo_sync=0.25, lfo_depth=2.0),
                    amp=(0.003, 0.2, 1.0, 0.05), glide=0.03, mono=True, drive=(3.5, "tanh"), width=0.5, gain=0.35),
    "dubstep_wobble": P(oscs=[O("saw", 1.0, voices=4, detune=18), O("square", 0.7, semi=-12)], sub=0.7,
                        filter=dict(type="ladder", cutoff=200, res=0.55, env=0.0, drive=2.4, lfo_sync=0.25, lfo_depth=4.0),
                        amp=(0.003, 0.2, 1.0, 0.06), glide=0.03, mono=True, drive=(3.0, "tanh"), gain=0.38),
    "house_bass": P(oscs=[O("organ", 1.0, drawbars=(0.8, 1.0, 0, 0.3, 0, 0))], filter=dict(type="ladder", cutoff=600, res=0.1, env=1.0),
                    amp=(0.002, 0.25, 0.5, 0.06), gain=0.8),
    "slap_fm_bass": P(fm=dict(ratio=1.0, index=3.5, decay=0.06, sustain=0.1, feedback=0.4), sub=0.4, filter=dict(type=None),
                      amp=(0.001, 0.3, 0.4, 0.06), mono=True, glide=0.02, gain=0.55),
    "rumble_bass": P(oscs=[O("sine", 1.0), O("saw", 0.3)], filter=dict(type="ladder", cutoff=180, res=0.3, env=0.5, drive=3.0),
                     amp=(0.002, 0.4, 0.4, 0.3), drive=(2.5, "tanh"), gain=0.8),
    # ------------------------------------------------------------ fx synths
    "siren": P(oscs=[O("saw", 1.0, voices=3, detune=20)], filter=dict(type="svf_lp", cutoff=4000, res=0.2, env=0.0),
               pitch_lfo=(1.0, 3.0, True), amp=(0.05, 0.3, 0.9, 0.3), gain=0.3),
    "zap_synth": P(oscs=[O("square", 1.0)], pitch_env=(36.0, 0.04), filter=dict(type=None), amp=(0.001, 0.12, 0.0, 0.05), gain=0.35),
}
PATCHES.update(NEW)

CATEGORIES = {
    "lead": ["supersaw_lead", "glide_lead", "smooth_lead", "chip_lead", "screech", "hoover", "fm_lead", "sync_lead", "laser_lead",
             "pwm_lead", "whistle", "vox_lead", "trance_lead", "future_lead", "mini_lead", "gliss_lead", "acid_lead",
             "sine_lead", "raw_screech", "chip_tri", "chip_pulse12"],
    "brass": ["synth_brass", "stab_brass", "poly_brass_80s", "horn_section", "trumpet_synth", "brass_stab_hard", "jump_brass", "obx_brass"],
    "pad": ["warm_pad", "glass_pad", "choir_pad", "string_ens", "space_drone", "glide_pad", "shimmer_pad", "dark_pad",
            "pwm_strings", "choir_ooh", "noise_pad", "crystal_pad", "evolving_pad"],
    "keys": ["pluck", "arp_pluck", "bell_fm", "epiano_fm", "marimba", "rhodes", "organ", "clav", "kalimba", "harp_pluck",
             "music_box", "future_pluck", "steel_pluck", "house_piano", "vox_chop"],
    "bass": ["sub_bass", "reese", "acid_303", "offbeat_bass", "synthwave_bass", "wobble", "fm_bass", "reverse_bass", "moog_bass",
             "pluck_bass", "808_bass", "growl_bass", "neuro_bass", "dubstep_wobble", "house_bass", "slap_fm_bass", "rumble_bass"],
    "fx": ["siren", "zap_synth"],
}


class DrumKit(Drums):
    """Drums + many more kicks, snares and percussion."""

    kick_len = 0.38

    def tail_root(self):
        return self.tune * 2.0 ** round(math.log2(52.0 / self.tune))

    def tail_midi(self):
        return int(round(69 + 12 * math.log2(self.tail_root() / 440.0)))

    def _thunder(self, vel):
        """Hardstyle/hardcore kick-bass ('thunder'): clean pitch-dropping body that
        lands on the key root, then a distorted, wide, swelling tail that is the
        actual bass of the track. Modelled on a reference: f0 ~167 Hz -> root in
        ~200 ms, dip at ~200 ms, crescendo to the end of the beat, hard cut."""
        rng = self.rng
        L = self.kick_len
        n = int(L * SR)
        t = np.arange(n) / SR
        root = self.tail_root()
        f = root + (172.0 - root) * np.exp(-t / 0.053) + 900 * np.exp(-t / 0.003)
        ph = 2 * np.pi * np.cumsum(f) / SR
        # body: near-sine, softly driven
        body = np.tanh(np.sin(ph) * 2.6 + 0.25 * np.sin(2 * ph) * np.exp(-t / 0.03))
        body = body + 0.05 * dsp.peaking_eq(np.tanh(np.sin(ph) * 14), 2000, 6.0, 0.8)
        click = dsp.svf(rng.standard_normal(n), np.full(n, 3500.0), 0.3, 1) * np.exp(-t / 0.002) * 0.7
        body_env = np.clip(1.0 - (t - 0.11) / 0.07, 0, 1)
        # tail: heavily distorted, slightly detuned per channel -> wide
        x0 = 0.13
        tail_env = np.where(t < 0.20, np.clip((t - x0) / 0.06, 0, 1) * 0.18,
                            0.18 + 0.55 * np.clip((t - 0.20) / (L - 0.035 - 0.20), 0, 1) ** 1.2)
        tail_env *= np.clip((L - 0.012 - t) / 0.012, 0, 1)
        chans = []
        for c, (cents, fold) in enumerate([(+14.0, 1.5), (-14.0, 2.3)]):
            fc = f * 2 ** (cents / 1200)
            pc = 2 * np.pi * np.cumsum(fc) / SR + c * 0.4
            src = np.sin(pc) + 0.6 * np.sin(2 * pc + 0.3) + 0.35 * np.sin(3.02 * pc)
            dist = np.tanh(src * 9.0)
            dist = np.sin(dist * fold * np.pi / 2)            # wavefold -> screechy upper partials
            dist = dsp.drive(dist + 0.15 * np.sign(src), 3.0, "asym")
            hi = dsp.peaking_eq(dist, 2400, 9.0, 1.1)
            hi = dsp.peaking_eq(hi, 900, 4.0, 1.4)
            hi = dsp.svf(hi, np.full(n, 9000.0), 0.0, 0)
            low = dsp.svf(np.tanh(np.sin(pc) * 3.0), np.full(n, 160.0), 0.1, 0)
            tail = 0.75 * hi + 0.45 * low
            mono_body = (body * 1.0 + click) * body_env
            chans.append(np.tanh((mono_body + tail * tail_env) * 1.4))
        y = np.stack(chans, axis=1)
        y /= np.max(np.abs(y)) + 1e-9
        return y * vel

    def _classic_hs(self, vel, euphoric=True):
        """Classic (euphoric) hardstyle kick: sampled punch + sine tail tuned to
        the key root, driven through a distortion -> EQ -> distortion chain, so
        the tail is a loud, bright, pitched 'tok' that IS the bassline."""
        import pedalboard as Pb
        from . import mixing, samples
        L = min(self.kick_len, 0.42)
        n = int(L * SR)
        t = np.arange(n) / SR
        root = self.tail_root()
        if root < 46:
            root *= 2 ** (5 / 12) if root * 2 > 100 else 2
        # tail oscillator: fast pitch drop into the root, held
        f = root + root * 3.0 * np.exp(-t / 0.012) + 240 * np.exp(-t / 0.035)
        ph = 2 * np.pi * np.cumsum(f) / SR
        src = np.sin(ph)
        amp = np.where(t < L - 0.09, 1.0, np.clip((L - 0.012 - t) / 0.078, 0, 1) ** 1.6)
        amp *= 1.0 - 0.15 * np.clip((t - 0.05) / (L - 0.1), 0, 1)
        tail = mixing.run(src * amp, [Pb.Gain(22), Pb.Distortion(24), Pb.HighpassFilter(35),
                                      Pb.PeakFilter(115, 5.0, 0.9), Pb.PeakFilter(1300 if euphoric else 2200, 7.0, 1.1),
                                      Pb.LowpassFilter(7000), Pb.Distortion(14), Pb.LowpassFilter(9500),
                                      Pb.Compressor(-12, 8.0, 0.5, 40)])
        tail = tail / (np.max(np.abs(tail)) + 1e-9)
        tail *= np.clip((L - 0.004 - t) / 0.006, 0, 1)
        # punch layer: sampled click + short body sweep
        click = samples.trimmed("bd_tek", 0.06, 0.02)
        punch = np.zeros(n)
        punch[: len(click)] += click[:n] * 0.8
        body = np.sin(2 * np.pi * np.cumsum(55 + 260 * np.exp(-t / 0.02)) / SR) * np.exp(-t / 0.05)
        punch += body * 0.9
        punch = np.tanh(punch * 2.0)
        y = 1.0 * tail * np.clip(t / 0.004, 0, 1) + 0.75 * punch
        y = np.tanh(y * 1.6)
        y /= np.max(np.abs(y)) + 1e-9
        return y * vel

    def kick(self, kind="909", vel=1.0):
        rng = self.rng
        if kind == "thunder":
            return self._thunder(vel)
        if kind in ("classic", "euphoric"):
            return self._classic_hs(vel, True)
        if kind == "raw_classic":
            return self._classic_hs(vel, False)
        if kind in ("hardstyle", "rawstyle", "gabber"):
            n = int((0.42 if kind != "gabber" else 0.3) * SR)
            t = np.arange(n) / SR
            f0 = self.tune
            # punch: fast pitch drop + click
            fpunch = f0 + 1400 * np.exp(-t / 0.004) + 260 * np.exp(-t / 0.022)
            # tail: tuned to key root, slight pitch fall ("zaag")
            ftail = f0 * (1 + 0.25 * np.exp(-t / 0.05)) * (1 - 0.06 * t / t[-1])
            mix = np.clip(t / 0.03, 0, 1)
            f = fpunch * (1 - mix) + ftail * mix
            ph = np.cumsum(f) / SR
            body = np.sin(2 * np.pi * ph) + 0.35 * np.sin(4 * np.pi * ph)
            env = np.exp(-t / 0.9) * np.clip((t[-1] - t) / 0.02, 0, 1)
            y = body * env
            drive_amt = {"hardstyle": 10.0, "rawstyle": 22.0, "gabber": 30.0}[kind]
            y = np.tanh(y * drive_amt)
            y = dsp.drive(y, 1.5, "asym")
            # crunch / screech formant of the tail
            y = dsp.peaking_eq(y, 1600 if kind != "rawstyle" else 2300, 7.0 if kind != "hardstyle" else 4.0, 1.4)
            y = dsp.peaking_eq(y, 120, 3.0, 0.9)
            y = dsp.svf(y, np.full(n, 7500.0 if kind != "gabber" else 9000.0), 0.0, 0)
            click = dsp.svf(rng.standard_normal(n), np.full(n, 4000.0), 0.2, 2) * np.exp(-t / 0.0025) * 0.6
            y = np.tanh((y + click) * 1.3)
            # post-distortion contour: punch, slight dip, sustained tail, tight fade
            post = np.where(t < 0.03, 1.0, 0.82 * np.exp(-(t - 0.03) / 0.9))
            post *= np.clip((t[-1] - t) / 0.05, 0, 1)
            y = y * post
        elif kind == "trap":
            n = int(0.5 * SR); t = np.arange(n) / SR
            f = 48 + 160 * np.exp(-t / 0.02)
            y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.3)
            y += dsp.svf(rng.standard_normal(n), np.full(n, 6000.0), 0.2, 2) * np.exp(-t / 0.002) * 0.5
            y = dsp.drive(y, 2.0)
        elif kind == "lofi":
            n = int(0.35 * SR); t = np.arange(n) / SR
            f = 55 + 100 * np.exp(-t / 0.025)
            y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.14)
            y = dsp.svf(dsp.bitcrush(y, 10, 2), np.full(n, 2500.0), 0.1, 0)
        elif kind == "techno":
            n = int(0.6 * SR); t = np.arange(n) / SR
            f = 50 + 200 * np.exp(-t / 0.015)
            y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18)
            rumble = dsp.svf(np.tanh(np.sin(2 * np.pi * self.tune * t) * 4), np.full(n, 200.0), 0.4, 0) * np.exp(-t / 0.3) * 0.6
            y = dsp.drive(y * 1.2 + rumble * (t > 0.06), 2.0)
        elif kind == "house":
            n = int(0.4 * SR); t = np.arange(n) / SR
            f = 54 + 140 * np.exp(-t / 0.025)
            y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.2)
            y += np.sin(2 * np.pi * 2200 * t) * np.exp(-t / 0.003) * 0.3
            y = dsp.drive(y, 1.4)
        else:
            return super().kick(kind, vel)
        y = y / (np.max(np.abs(y)) + 1e-9)
        return y * vel

    def snare(self, kind="909", vel=1.0):
        rng = self.rng
        if kind == "trap":
            n = int(0.25 * SR); t = np.arange(n) / SR
            nz = dsp.svf(rng.standard_normal(n), np.full(n, 5500.0), 0.2, 2) * np.exp(-t / 0.06)
            tone = np.sin(2 * np.pi * np.cumsum(220 + 120 * np.exp(-t / 0.01)) / SR) * np.exp(-t / 0.04)
            y = nz + tone * 0.6 + self.clap(1.0)[:n] * 0.6
        elif kind == "electro":
            n = int(0.3 * SR); t = np.arange(n) / SR
            y = np.sin(2 * np.pi * np.cumsum(230 + 300 * np.exp(-t / 0.006)) / SR) * np.exp(-t / 0.05)
            y += dsp.bitcrush(rng.standard_normal(n), 6, 3) * np.exp(-t / 0.1) * 0.6
        elif kind == "lofi":
            n = int(0.3 * SR); t = np.arange(n) / SR
            y = super().snare("909", 1.0)[:n]
            y = dsp.svf(dsp.bitcrush(y, 8, 3), np.full(n, 3500.0), 0.1, 0)
        elif kind == "breakbeat":
            n = int(0.35 * SR); t = np.arange(n) / SR
            tone = np.sin(2 * np.pi * np.cumsum(190 + 90 * np.exp(-t / 0.012)) / SR) * np.exp(-t / 0.09)
            ring = np.sin(2 * np.pi * 410 * t) * np.exp(-t / 0.12) * 0.25
            nz = dsp.svf(rng.standard_normal(n), np.full(n, 3800.0), 0.35, 1) * np.exp(-t / 0.15)
            y = dsp.drive(tone + ring + nz, 2.0)
        elif kind == "hardstyle":
            y = self.clap(1.0)
            n = len(y); t = np.arange(n) / SR
            y = y + dsp.svf(rng.standard_normal(n), np.full(n, 2000.0), 0.4, 1) * np.exp(-t / 0.2) * 0.5
        else:
            return super().snare(kind, vel)
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    def perc(self, kind="conga", vel=1.0, pitch=1.0):
        rng = self.rng
        if kind in ("conga", "bongo"):
            f0 = (210 if kind == "conga" else 380) * pitch
            n = int(0.35 * SR); t = np.arange(n) / SR
            y = np.sin(2 * np.pi * np.cumsum(f0 * (1 + 0.3 * np.exp(-t / 0.01))) / SR) * np.exp(-t / (0.12 if kind == "conga" else 0.07))
            y += dsp.svf(rng.standard_normal(n), np.full(n, 3000.0), 0.4, 1) * np.exp(-t / 0.004) * 0.3
        elif kind == "clave":
            n = int(0.12 * SR); t = np.arange(n) / SR
            y = np.sin(2 * np.pi * 2500 * pitch * t) * np.exp(-t / 0.02)
        elif kind == "woodblock":
            n = int(0.1 * SR); t = np.arange(n) / SR
            y = dsp.svf(rng.standard_normal(n), np.full(n, 1100.0 * pitch), 0.93, 1) * np.exp(-t / 0.03)
        elif kind == "snap":
            n = int(0.1 * SR); t = np.arange(n) / SR
            y = dsp.svf(rng.standard_normal(n), np.full(n, 2600.0), 0.55, 1) * np.exp(-t / 0.012)
        elif kind == "tambourine":
            n = int(0.25 * SR); t = np.arange(n) / SR
            y = dsp.svf(self._metal(n) * 0.5 + rng.standard_normal(n), np.full(n, 9000.0), 0.3, 1) * np.exp(-t / 0.07)
        elif kind == "vox_hey":
            n = int(0.25 * SR); t = np.arange(n) / SR
            f = 220 * pitch * (1 + 0.1 * np.exp(-t / 0.05))
            src = dsp.osc_saw(dsp.phase_from_freq(f), f) + rng.standard_normal(n) * 0.2
            y = sum(dsp.svf(src, np.full(n, fc), 0.9, 1) * g for fc, g in ((700, 1.0), (1700, 0.6), (2600, 0.3)))
            y *= np.clip(t / 0.01, 0, 1) * np.exp(-t / 0.09)
        elif kind == "laser":
            y = self.zap(1.0, 4000, 200, 0.18)
        elif kind == "sub_drop":
            n = int(1.2 * SR); t = np.arange(n) / SR
            f = 25 + 80 * np.exp(-t / 0.35)
            y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.7)
        else:
            raise ValueError(kind)
        return y / (np.max(np.abs(y)) + 1e-9) * vel

    def hat_roll(self, n_hits, spacing_s, vel=0.6):
        """Trap-style hat roll as one buffer."""
        L = int((n_hits * spacing_s + 0.1) * SR)
        out = np.zeros(L)
        for i in range(n_hits):
            h = self.hat(False, vel * (0.6 + 0.4 * i / max(1, n_hits - 1)))
            a = int(i * spacing_s * SR)
            out[a:a + len(h)] += h[: L - a]
        return out


def jitter_patch(name, rng, amount=1.0):
    """Per-track character: the same patch name with randomised detune, cutoff, envelopes, drive,
    chorus and sometimes an extra octave layer, so two tracks never share the exact sound."""
    import copy as _c
    p = _c.deepcopy(PATCHES[name])
    u = lambda lo_, hi_: lo_ + (hi_ - lo_) * rng.random()
    a = amount
    for o in p.get("oscs") or []:
        if o.get("detune"):
            o["detune"] = o["detune"] * u(1 - 0.35 * a, 1 + 0.45 * a)
        if o.get("voices", 1) >= 3 and rng.random() < 0.3 * a:
            o["voices"] = int(min(9, o["voices"] + 2))
    f = p.get("filter") or {}
    if f.get("type"):
        f["cutoff"] = f.get("cutoff", 2000) * u(1 - 0.25 * a, 1 + 0.5 * a)
        f["res"] = min(0.85, max(0.0, f.get("res", 0.2) + u(-0.1, 0.15) * a))
        f["env"] = f.get("env", 1.0) * u(0.75, 1.3)
    at, d, s_, r = p["amp"]
    p["amp"] = (at, d * u(0.7, 1.5), min(1.0, s_ * u(0.85, 1.15)), r * u(0.7, 1.6))
    fa, fd, fs, fr = p["fenv"]
    p["fenv"] = (fa, fd * u(0.6, 1.6), fs, fr)
    p["chorus"] = min(0.6, p.get("chorus", 0.0) + u(-0.1, 0.25) * a) if rng.random() < 0.7 else p.get("chorus", 0.0)
    if p.get("oscs") and rng.random() < 0.3 * a and len(p["oscs"]) < 3 and not p.get("formant"):
        semi = 12 if rng.random() < 0.6 else -12
        p["oscs"] = list(p["oscs"]) + [dict(p["oscs"][0], level=0.3, semi=p["oscs"][0].get("semi", 0) + semi)]
    vr, vd, vdl = p["vibrato"]
    p["vibrato"] = (vr * u(0.85, 1.15), vd * u(0.6, 1.5), vdl)
    return p
