"""Composer v2: genre styles, arrangement with development & interruptions,
stem rendering, mixing and the event log consumed by the visualiser."""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

import numpy as np

from . import dsp, theory, harmony
from .dsp import SR
from . import mixing, basses, surge
from .instruments import Synth, P, O, PATCHES
from . import library
from .library import DrumKit, CATEGORIES  # noqa: F401  (registers the extended library)
from .theory import Key, Note, ChordEvent

PATCHES.setdefault("chip_bass", P(oscs=[O("pulse", 1.0, pw=0.5)], filter=dict(type=None),
                                   amp=(0.001, 0.1, 0.8, 0.03), gain=0.4))
PATCHES.setdefault("chip_arp", P(oscs=[O("pulse", 1.0, pw=0.125)], filter=dict(type=None),
                                  amp=(0.001, 0.06, 0.4, 0.03), gain=0.22))

# ---------------------------------------------------------------- styles
# lead/pad/... entries are pools: every song picks its own instruments.
STYLES = {
    "heavy_bass": dict(
        label="Bass House", bpm=(124, 128), scales=["minor", "phrygian"],
        progs=["dark_descent", "minor_plagal", "andalusian"], bars_per_chord=2,
        lead=["vox_chop", "future_pluck", "brass_stab_hard", "laser_lead"], lead_char=["playful", "wild"],
        counter=["zap_synth", "vox_chop"],
        bass=[("@growl", "house"), ("@wobble", "drop_long"), ("@neuro", "house"), ("@square", "house"), ("@reese", "drop_long")], sub=False,
        pad=["dark_pad"], glide_chords=0.0, chords_inst=["brass_stab_hard", None],
        arp=(["pluck"], ["up"], 0.25),
        drums="four", kick="techno", snare="909", perc=["clave", "vox_hey", "snap"], reverb=(1.5, 0.18), sidechain=0.75,
        mix={"bass": -12.5, "lead": -18.0, "kick": -10.5}, bass_drive=1.2, drop_layer=None, drum_crunch=1.2,
        gate=0.0, swing=0.04, target_s=44,
        words=("GRIME", "FURNACE", "MAGMA", "WRECK", "BRUTE", "GRIT", "TOXIC", "DIESEL"),
    ),
    "rawstyle": dict(
        label="Rawstyle", bpm=(150, 154), scales=["minor", "phrygian", "harmonic_minor"],
        progs=["dark_descent", "minor_plagal", "axis_minor"], bars_per_chord=1,
        lead=["raw_screech", "screech", "hoover"], lead_char=["wild"],
        counter=["brass_stab_hard", "zap_synth"],
        bass=[("reverse_bass", "reverse")], sub=False,
        pad=["dark_pad", "choir_pad"], glide_chords=0.2, chords_inst=["brass_stab_hard", None],
        arp=(["pluck"], ["up"], 0.25),
        drums="hardstyle", kick=["hs_raw"], kick_bass=True, snare="hardstyle", perc=["vox_hey"],
        mix={"kick": -7.0, "lead": -16.0, "snare": -18.0}, reverb=(2.2, 0.25), sidechain=0.9,
        drum_crunch=0.5, drop_layer=None,
        gate=0.3, swing=0.0, target_s=46,
        words=("CHAOS", "WARHEAD", "BERSERK", "VENOM", "RIOT", "DEMON", "OBLIVION", "HAVOC"),
    ),
    "synthwave": dict(
        label="Synthwave", bpm=(98, 112), scales=["minor", "dorian"],
        progs=["axis_minor", "outrun", "dark_descent", "dorian_vamp"], bars_per_chord=1,
        lead=["glide_lead", "synth_brass", "supersaw_lead", "mini_lead", "gliss_lead", "trumpet_synth", "pwm_lead"],
        lead_char=["flowing", "epic", "smooth"], counter=["arp_pluck", "bell_fm", "clav"],
        bass=[("synthwave_bass", "octave8"), ("moog_bass", "octave8")], sub=False,
        pad=["warm_pad", "string_ens", "pwm_strings", "obx_brass"], glide_chords=0.6,
        chords_inst=["poly_brass_80s", "horn_section", None], arp=(["arp_pluck", "pluck"], ["up", "updown", "pinky"], 0.25),
        drums="synthwave", kick="909", snare="gated", perc=["tambourine"], reverb=(2.8, 0.32), sidechain=0.55,
        gate=0.0, swing=0.0, target_s=46,
        words=("NEON", "MIDNIGHT", "CHROME", "SUNSET", "VHS", "DRIVE", "ARCADE", "LASER"),
    ),
    "dnb_liquid": dict(
        label="Liquid Drum & Bass", bpm=(172, 176), scales=["minor", "dorian"],
        progs=["liquid_dnb", "liquid_dnb_2", "space_suspended"], bars_per_chord=2,
        lead=["smooth_lead", "whistle", "glide_lead", "vox_lead", "gliss_lead", "rhodes"], lead_char=["smooth", "flowing"],
        counter=["rhodes", "kalimba", "bell_fm", "harp_pluck"],
        bass=[("@reese", "reese"), ("@square", "reese")], sub=False, bass_drive=0.35,
        pad=["glide_pad", "warm_pad", "choir_ooh", "shimmer_pad", "string_ens"], glide_chords=0.85,
        chords_inst=["rhodes", "epiano_fm", None], arp=(["bell_fm", "kalimba", "music_box"], ["up", "converge"], 0.5),
        drums="dnb", kick="punch", snare="breakbeat", perc=["shaker", "snap"], mix={"snare": -14.5},
        reverb=(3.2, 0.35), sidechain=0.35, gate=0.0, swing=0.0, target_s=46,
        words=("LIQUID", "RAIN", "AURORA", "DEEP", "FLOW", "BLUE", "TIDE", "SILK"),
    ),
    "dnb_energy": dict(
        label="Drum & Bass", bpm=(174, 176), scales=["minor", "phrygian", "harmonic_minor"],
        progs=["dark_descent", "andalusian", "minor_plagal"], bars_per_chord=2,
        lead=["hoover", "screech", "laser_lead", "sync_lead", "future_lead"], lead_char=["wild", "playful"],
        counter=["vox_chop", "brass_stab_hard", "zap_synth"],
        bass=[("@reese", "rolling"), ("@neuro", "rolling"), ("@growl", "rolling"), ("@wobble", "reese")], sub=False,
        pad=["dark_pad", "warm_pad"], glide_chords=0.3, chords_inst=["brass_stab_hard", "stab_brass"],
        arp=(["pluck", "future_pluck"], ["up", "pinky"], 0.25),
        drums="dnb", kick="punch", snare="dnb", perc=["vox_hey", "laser"], reverb=(1.8, 0.22), sidechain=0.4,
        bass_drive=0.9, drop_layer=None,
        gate=0.0, swing=0.0, target_s=44,
        words=("VOLT", "RAZOR", "SHOCK", "NOVA", "KINETIC", "FURY", "CIRCUIT", "ION"),
    ),
    "hardstyle": dict(
        label="Hardstyle", bpm=(148, 152), scales=["minor", "harmonic_minor"],
        progs=["heroic", "euphoric", "axis_minor"], bars_per_chord=1,
        lead=["supersaw_lead", "raw_screech", "screech", "trance_lead"], lead_char=["epic", "wild"],
        counter=["jump_brass", "vox_chop"],
        bass=[("reverse_bass", "reverse")], sub=False,
        pad=["warm_pad", "choir_pad", "shimmer_pad"], glide_chords=0.4, chords_inst=["supersaw_lead", "jump_brass"],
        arp=(["pluck", "future_pluck"], ["up"], 0.25),
        drums="hardstyle", kick=["hs_euphoric"], kick_bass=True, snare="hardstyle", perc=["vox_hey"],
        mix={"kick": -7.5, "bass": -16.0, "lead": -15.5, "snare": -17.5}, reverb=(2.5, 0.3), sidechain=0.9,
        drum_crunch=0.6,
        gate=0.5, swing=0.0, target_s=46,
        words=("TITAN", "INFERNO", "LEGION", "VORTEX", "RAGE", "EUPHORIA", "THUNDER", "NEXUS"),
    ),
    "trance": dict(
        label="Trance", bpm=(136, 140), scales=["minor"],
        progs=["axis_minor", "euphoric", "andalusian"], bars_per_chord=1,
        lead=["trance_lead", "supersaw_lead", "pwm_lead", "laser_lead"], lead_char=["epic", "flowing"],
        counter=["pluck", "bell_fm", "kalimba"],
        bass=[("offbeat_bass", "offbeat")], sub=True, pad=["glide_pad", "warm_pad", "shimmer_pad", "string_ens"], glide_chords=0.6,
        chords_inst=[None, "trance_lead"], arp=(["pluck", "future_pluck"], ["up", "updown", "converge"], 0.25),
        drums="four", kick="909", snare="909", perc=["clave", "tambourine"], reverb=(3.5, 0.35), sidechain=0.7,
        gate=0.6, swing=0.0, target_s=46,
        words=("SKYLINE", "ETERNAL", "HORIZON", "ASCEND", "ORBIT", "STELLAR", "SOLAR", "ZENITH"),
    ),
    "acid": dict(
        label="Acid Techno", bpm=(128, 134), scales=["phrygian", "minor", "dorian"],
        progs=["dark_descent", "minor_plagal"], bars_per_chord=2,
        lead=["acid_lead", "bell_fm", "sync_lead", "organ"], lead_char=["playful"], counter=["vox_chop", "zap_synth", "clav"],
        bass=[("acid_303", "acid")], sub=True, pad=["space_drone", "noise_pad", "dark_pad"], glide_chords=0.3,
        chords_inst=[None, "organ"], arp=(["arp_pluck", "clav"], ["up", "pinky"], 0.25),
        drums="four", kick="techno", snare="909", perc=["clave", "woodblock", "conga"],
        mix={"bass": -14.0, "lead": -20.0}, reverb=(2.0, 0.22), sidechain=0.45, gate=0.0, swing=0.08, target_s=44,
        words=("ACID", "303", "WAREHOUSE", "SQUELCH", "LYSERGIC", "TOXIC", "MOLECULE", "ROBOT"),
    ),
    "chiptune": dict(
        label="Chiptune", bpm=(140, 150), scales=["major", "mixolydian", "major_pentatonic"],
        progs=["chip_happy", "mixo_drive", "axis_major"], bars_per_chord=1,
        lead=["chip_lead", "chip_pulse12", "chip_tri"], lead_char=["playful", "wild"], counter=["chip_pulse12", "chip_tri"],
        bass=[("chip_bass", "octave8"), ("chip_tri", "octave8")], sub=False, pad=[None], glide_chords=0.0,
        chords_inst=[None], arp=(["chip_arp"], ["up", "updown"], 0.125),
        drums="chip", kick="909", snare="909", perc=[], reverb=(1.0, 0.12), sidechain=0.0, gate=0.0,
        swing=0.0, target_s=42,
        words=("PIXEL", "8-BIT", "COIN", "LEVEL", "TURBO", "BONUS", "JOYPAD", "SPRITE"),
    ),
    "ambient_space": dict(
        label="Space Ambient", bpm=(78, 88), scales=["lydian", "dorian", "major_pentatonic", "minor"],
        progs=["lydian_float", "dreamy_maj7", "space_suspended", "royal_road"], bars_per_chord=1,
        lead=["smooth_lead", "whistle", "sine_lead", "crystal_pad", "gliss_lead"], lead_char=["smooth"],
        counter=["kalimba", "music_box", "bell_fm", "harp_pluck"],
        bass=[("sub_bass", "root")], sub=False, pad=["glide_pad", "glass_pad", "choir_ooh", "evolving_pad", "shimmer_pad"],
        glide_chords=0.9, chords_inst=["crystal_pad", None], arp=(["bell_fm", "kalimba", "music_box"], ["up", "converge"], 0.5),
        drums="halftime", kick="soft", snare="rim", perc=["shaker"],
        mix={"kick": -17.0, "snare": -20.0, "pad": -17.0, "hats": -27.0},
        reverb=(5.5, 0.5), sidechain=0.15, gate=0.0, swing=0.0, target_s=48,
        words=("NEBULA", "COSMOS", "DRIFT", "ORBITAL", "STARDUST", "VOID", "LUNAR", "QUASAR"),
    ),
    "future_bass": dict(
        label="Future Bass", bpm=(145, 155), scales=["major", "lydian", "minor"],
        progs=["royal_road", "dreamy_maj7", "axis_major", "sad_axis"], bars_per_chord=1,
        lead=["future_lead", "vox_lead", "whistle", "sync_lead"], lead_char=["playful", "flowing"],
        counter=["vox_chop", "future_pluck", "kalimba"],
        bass=[("@808", "trap808")], sub=False, bass_drive=0.8,
        pad=["glide_pad", "shimmer_pad"], glide_chords=0.9, chords_inst=["trance_lead", "future_pluck"],
        arp=(["future_pluck", "kalimba"], ["up", "converge"], 0.25),
        drums="future", kick="trap", snare="trap", perc=["snap", "vox_hey"], reverb=(2.8, 0.32), sidechain=0.8,
        gate=0.0, swing=0.0, target_s=44,
        words=("SUGAR", "CANDY", "BLOOM", "PRISM", "HALO", "PASTEL", "KAWAII", "CLOUD"),
    ),
    "jerk": dict(
        label="Jerk", bpm=(150, 165), scales=["major", "minor", "dorian", "lydian"],
        progs=["royal_road", "sad_axis", "dreamy_maj7", "axis_major"], bars_per_chord=1,
        lead=["bell_fm", "kalimba", "music_box", "future_pluck", "chip_pulse12"], lead_char=["playful", "wild"],
        counter=["kalimba", "bell_fm", "music_box"],
        bass=[("@808", "trap808")], sub=False, bass_drive=0.9,
        pad=["shimmer_pad", "glass_pad", None], glide_chords=0.6, chords_inst=["future_pluck", None],
        arp=(["music_box", "kalimba", "bell_fm"], ["up", "updown", "converge"], 0.25),
        drums="jerk", kick="trap", snare="trap", perc=["snap"], reverb=(2.4, 0.3), sidechain=0.6,
        hook="jerk", gate=0.0, swing=0.0, target_s=44,
        words=("JERK", "BOUNCE", "GLITTER", "SPARKLE", "HYPER", "CANDY", "DIAMOND", "SWAG"),
    ),
}

# stem levels in LUFS-S relative to the kick, by genre family (research 4.2)
LEVELS = {
    # LUFS-S relative to the kick in the drop. K-weighting under-reads lows, so the music sits well below the kick
    # (measured against reference hardstyle / bass music: 20-80 Hz must dominate the drop).
    "hs":   {"kick": 0, "bass": -6, "snare": -15, "lead": -11, "chords": -16, "pad": -17, "hats": -19, "arp": -17, "counter": -15, "perc": -19},
    "bass": {"kick": 0, "bass": 1.5, "snare": -3, "lead": -8, "chords": -14, "pad": -15, "hats": -12, "arp": -14, "counter": -11, "perc": -14},
    "dnb":  {"kick": 0, "bass": 0.5, "snare": -3, "lead": -8, "chords": -13, "pad": -13, "hats": -13, "arp": -14, "counter": -11, "perc": -14},
    "fb":   {"kick": 0, "bass": 0, "snare": -4, "lead": -7, "chords": -6, "pad": -11, "hats": -14, "arp": -12, "counter": -10, "perc": -15},
    "sw":   {"kick": 0, "bass": -3, "snare": -4, "lead": -5, "chords": -10, "pad": -10, "hats": -15, "arp": -11, "counter": -10, "perc": -15},
}
FAMILY = {"hardstyle": "hs", "rawstyle": "hs", "heavy_bass": "bass", "dnb_energy": "dnb", "dnb_liquid": "dnb",
          "future_bass": "fb", "jerk": "fb", "synthwave": "sw", "trance": "sw", "acid": "sw", "chiptune": "sw", "ambient_space": "sw"}
# master high shelf (+2.5 dB x this at 6 kHz); hardstyle is kept darker so the kick carries the energy
BRIGHT = {"hs": 0.0, "bass": 0.6, "dnb": 0.8, "fb": 1.0, "sw": 1.0}
LEAD_ROLE = None   # force a Surge lead role, e.g. "brasslead" for calm melodic tracks
LUFS_TARGET = {"hs": -6.5, "bass": -6.5, "dnb": -7.0, "fb": -7.5, "sw": -9.0}

_TEMPLATE = [("intro", 2), ("build", 2), ("drop", 4), ("break", 2), ("drop2", 4), ("outro", 1)]
_SECTION_ROLES = {
    "intro": {"pad", "arp_f", "fx", "counter"},
    "build": {"pad", "arp", "hats", "kick_light", "roll", "fx", "lead_f"},   # no bass: the drop must feel bigger
    "drop": {"pad", "arp", "bass", "lead", "chords", "kick", "snare", "hats", "perc", "fx", "counter"},
    "break": {"pad", "lead_soft", "arp_f", "hats_soft", "counter"},
    "drop2": {"pad", "arp", "bass", "lead", "chords", "kick", "snare", "hats", "perc", "fx", "counter", "harm"},
    "outro": {"pad", "lead_soft"},
}
_ENERGY = {"intro": 0.25, "build": 0.55, "drop": 1.0, "break": 0.35, "drop2": 1.0, "outro": 0.2}
OUTRO_SECONDS = 5.5  # room for the subscribe animation
MELODY_MODE = None   # force "flow" or a family name for auditions
LOOP_S = 0.5         # tail crossfaded into the start for a seamless loop


@dataclass
class Section:
    name: str
    start_bar: int
    bars: int
    energy: float
    roles: set


@dataclass
class Song:
    style: str
    title: str
    bpm: float
    key: Key
    sections: list
    chords: list
    duration: float
    audio: np.ndarray = None
    stems: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)

    @property
    def spb(self):
        return 60.0 / self.bpm


def _pick(rng, x):
    return rng.choice(x) if isinstance(x, (list, tuple)) else x


def _arrangement(style, bpm, rng):
    bar_s = 4 * 60 / bpm
    body = max(8, round(style["target_s"] / bar_s))
    weights = sum(w for n, w in _TEMPLATE if n != "outro")
    bars = []
    for name, w in _TEMPLATE[:-1]:
        b = max(2, round(body * w / weights))
        b -= b % 2
        bars.append([name, b])
    outro = max(2, int(np.ceil(OUTRO_SECONDS / bar_s)))
    bars.append(["outro", outro])
    out, start = [], 0
    for name, b in bars:
        out.append(Section(name, start, b, _ENERGY[name], set(_SECTION_ROLES[name])))
        start += b
    return out


def _sec_at(sections, bar):
    for s in sections:
        if s.start_bar <= bar < s.start_bar + s.bars:
            return s
    return sections[-1]


def _filter_notes(notes, sections, roles):
    return [n for n in notes if roles & _sec_at(sections, int(n.start // 4)).roles]


# ---------------------------------------------------------------- drums
def _drum_bar(kind, bis, sec, rng):
    """(instrument, step16, vel) list for one bar, with fills every 4 bars."""
    hits = []
    roles = sec.roles
    _drum_bar.last_fill = None

    def add(inst, steps, vel=1.0, jitter=0.08):
        for s in steps:
            hits.append((inst, s, max(0.05, min(1.0, vel + rng.uniform(-jitter, jitter)))))

    full = "kick" in roles
    light = "kick_light" in roles
    fill = full and bis % 4 == 3
    big_fill = full and bis % 8 == 7
    if kind in ("four", "hardstyle"):
        if full or light:
            ks = [0, 4, 8, 12]
            if fill and kind == "hardstyle":
                ks = [0, 4, 8, 10, 12, 14]
            add("kick", ks, 1.0 if full else 0.8, 0.0)
        if full:
            add("clap", [4, 12], 0.9)
            add("hat_o", [2, 6, 10, 14], 0.55)
            add("hat", [1, 3, 5, 7, 9, 11, 13, 15], 0.35)
            add("perc", [s for s, on in enumerate(theory.euclid(5, 16, 2 + bis % 3)) if on], 0.45)
        elif light or "hats" in roles:
            add("hat", [2, 6, 10, 14], 0.45)
    elif kind == "synthwave":
        if full or light:
            add("kick", [0, 8] if light else [0, 4, 8, 12], 0.95, 0.0)
        if full:
            add("snare", [4, 12], 1.0, 0.0)
            add("hat", list(range(0, 16, 2)), 0.45)
            add("hat", [s for s in range(1, 16, 2) if rng.random() < 0.3], 0.25)
            add("perc", [2, 6, 10, 14], 0.3)
        elif light or "hats" in roles:
            add("hat", list(range(0, 16, 2)), 0.35)
    elif kind == "dnb":
        variants = [([0, 10], [4, 12]), ([0, 10, 11], [4, 12]), ([0, 2, 10], [4, 12]), ([0, 6, 10], [4, 12, 15])]
        k, s = variants[bis % len(variants)] if full else ([0, 10], [4, 12])
        if full:
            add("kick", k, 1.0, 0.0)
            add("snare", s, 1.0, 0.03)
            add("ghost", [st for st in (7, 9, 14) if rng.random() < 0.5], 0.28)
            add("hat", list(range(0, 16, 2)), 0.5)
            add("hat", [st for st in range(1, 16, 2) if rng.random() < 0.55], 0.22)
            add("perc", [st for st in (3, 11) if rng.random() < 0.6], 0.4)
            if bis % 2 == 1:
                add("hat_o", [6], 0.4)
        elif light:
            add("kick", [0], 0.85)
            add("hat", list(range(0, 16, 2)), 0.4)
        elif "hats" in roles or "hats_soft" in roles:
            add("hat", [2, 6, 10, 14], 0.3)
    elif kind == "future":   # half-time with trap hat rolls
        if full or light:
            add("kick", [0, 7, 10] if full else [0], 0.95, 0.0)
        if full:
            add("snare", [8], 1.0, 0.0)
            add("hat", [0, 2, 4, 6, 10, 12], 0.4)
            add("hat_roll", [14] if bis % 2 else [6, 14], 0.45)
            add("perc", [3, 11], 0.4)
        elif light or "hats" in roles:
            add("hat", list(range(0, 16, 4)), 0.35)
    elif kind == "jerk":     # jerk: bouncy syncopated 808 kicks, claps on 2 and 4, skittering 16th hats + rolls
        if full or light:
            pats = [[0, 3, 6, 10], [0, 3, 8, 11, 14], [0, 6, 7, 10], [0, 3, 10, 13]]
            add("kick", pats[bis % len(pats)] if full else [0], 0.95, 0.0)
        if full:
            add("clap", [4, 12], 1.0, 0.0)
            add("hat", [s_ for s_ in range(16) if s_ % 2 == 0 or rng.random() < 0.55], 0.38, 0.15)
            add("hat_roll", [6, 14] if bis % 2 else [14], 0.5)
            add("hat_o", [10] if bis % 2 == 0 else [2], 0.35)
            add("perc", [s_ for s_ in (5, 13) if rng.random() < 0.5], 0.4)
        elif light or "hats" in roles:
            add("hat", list(range(0, 16, 2)), 0.35)
    elif kind == "halftime_hard":   # DnB groove switch: heavy half-time
        add("kick", [0, 3, 10] if bis % 2 == 0 else [0, 6, 10, 11], 1.0, 0.0)
        add("snare", [8], 1.0, 0.0)
        add("hat", list(range(0, 16, 2)), 0.45)
        add("perc", [5, 13], 0.4)
    elif kind == "halftime":
        if full or light:
            add("kick", [0, 10] if full else [0], 0.8, 0.0)
            add("snare", [8], 0.8)
            add("shaker", list(range(0, 16, 2)), 0.35)
        elif "hats" in roles or "hats_soft" in roles:
            add("shaker", [4, 12], 0.25)
    elif kind == "chip":
        if full or light:
            add("kick", [0, 4, 8, 12] if full else [0, 8], 0.9, 0.0)
        if full:
            add("noise_snare", [4, 12], 0.8)
            add("noise_hat", list(range(0, 16, 2)), 0.4)
        elif light or "hats" in roles:
            add("noise_hat", list(range(0, 16, 4)), 0.35)
    # fills: replace the last beat with a roll / toms (big fill every 8 bars)
    if fill and kind not in ("halftime",):
        hits = [h for h in hits if h[1] < (8 if big_fill else 12) or h[0] in ("hat", "kick" if kind in ("four", "hardstyle") else "_")]
        four = kind in ("four", "hardstyle", "synthwave", "chip")
        if big_fill:
            v = rng.choice(["toms", "accel", "kickroll" if four else "toms", "flams"])
            _drum_bar.last_fill = (v, True)
            if v == "toms":
                add("tom", [8, 10, 11, 12, 13, 14, 15], 0.9)
                add("snare_roll", [9, 15], 0.7)
            elif v == "accel":
                add("snare_roll", [8, 10, 12, 13, 14, 14.5, 15, 15.5], 0.8)
            elif v == "kickroll":
                add("kick", [8, 9, 10, 11, 12, 13, 14, 15], 0.95, 0.0)
                add("snare_roll", [12, 14, 15, 15.5], 0.8)
            else:
                add("snare", [8, 11, 14], 1.0, 0.0)
                add("flam", [8, 11, 14], 0.6)
                add("tom", [12, 13, 15], 0.85)
            add("drumstutter", [15], 1.0)
        else:
            v = rng.choice(["roll", "trip", "flam", "kick16" if four else "roll", "snap"])
            _drum_bar.last_fill = (v, False)
            if v == "roll":
                add("snare_roll", [12, 13, 14, 15], 0.75)
            elif v == "trip":
                add("snare_roll", [12, 12 + 4 / 6, 12 + 8 / 6, 14, 14 + 2 / 3, 14 + 4 / 3], 0.7)
                add("hat_roll", [12, 14], 0.5)
            elif v == "flam":
                add("snare", [12, 14.5], 0.9, 0.0)
                add("flam", [12, 14.5], 0.5)
                add("snare_roll", [15, 15.5], 0.7)
            elif v == "kick16":
                add("kick", [12, 13, 14, 15], 0.9, 0.0)
                add("snare_roll", [14, 15], 0.7)
            else:
                add("snap", [12, 13.5, 15], 0.8)
                add("snare_roll", [14, 15, 15.5], 0.7)
    if "roll" in roles and kind == "hardstyle":
        left = sec.bars - bis
        hits = [h for h in hits if h[0] != "kick"]
        if left == 2:
            add("kick", list(range(0, 16, 2)), 0.9, 0.0)
        elif left == 1:
            add("kick", list(range(0, 12)), 0.95, 0.0)
        else:
            add("kick", [0, 4, 8, 12], 0.85, 0.0)
    if "roll" in roles:
        left = sec.bars - bis
        if left > 2:
            add("snare_roll", [12, 14], 0.35)
        elif left == 2:
            add("snare_roll", [x * 2 for x in range(8)], 0.55)            # 8ths
        elif left == 1:                                                  # 16ths -> 32nds, last beat = gap
            add("snare_roll", list(range(0, 8)) + [8 + x * 0.5 for x in range(8)], 0.8)
    return hits


# ---------------------------------------------------------------- audio fx on time ranges
def _gain_range(n, ranges, floor=0.0, ramp=0.004):
    g = np.ones(n)
    r = int(ramp * SR)
    for a, b in ranges:
        ia, ib = int(a * SR), int(b * SR)
        g[ia:ib] = floor
        if r and ia - r > 0:
            g[ia - r:ia] = np.minimum(g[ia - r:ia], np.linspace(1, floor, r))
        if r and ib + r < n:
            g[ib:ib + r] = np.minimum(g[ib:ib + r], np.linspace(floor, 1, r))
    return g


def stutter(x, t0, t1, spb):
    """Beat-repeat roll: repeats slices that get shorter (1/8 -> 1/16 -> 1/32)."""
    a, b = int(t0 * SR), int(t1 * SR)
    seg = x[a:b].copy()
    L = b - a
    out = np.zeros_like(seg)
    pos = 0
    sizes = [spb / 2, spb / 4, spb / 8]
    for k, sz in enumerate(sizes):
        span = L // len(sizes)
        sl = int(sz * SR)
        src = x[a + pos: a + pos + sl]
        if len(src) < 8:
            break
        fade = np.minimum(1, np.minimum(np.arange(sl), sl - np.arange(sl)) / 64)[:, None]
        for j in range(pos, min(L, pos + span), sl):
            m = min(sl, L - j)
            out[j:j + m] = src[:m] * fade[:m]
        pos += span
    x[a:b] = out


def tape_stop(x, t0, t1):
    """Slow the audio down to a halt between t0 and t1."""
    a, b = int(t0 * SR), int(t1 * SR)
    L = b - a
    speed = np.linspace(1.0, 0.0, L) ** 1.5
    idx = a + np.cumsum(speed)
    idx = np.clip(idx, 0, len(x) - 1)
    for c in range(x.shape[1]):
        x[a:b, c] = np.interp(idx, np.arange(len(x)), x[:, c]) * np.linspace(1, 0.3, L)


def crush(x, t0, t1, bits=6, down=10):
    a, b = int(t0 * SR), int(t1 * SR)
    seg = x[a:b]
    idx = (np.arange(len(seg)) // down) * down
    q = 2 ** (bits - 1)
    x[a:b] = np.round(seg[idx] * q) / q


def filter_dive(x, t0, t1, f0=16000.0, f1=280.0):
    """Low-pass dive on a range (the music sinks underwater, then snaps back)."""
    a, b = int(t0 * SR), int(t1 * SR)
    sweep = f0 * (f1 / f0) ** (np.linspace(0, 1, b - a) ** 0.7)
    for c in range(x.shape[1]):
        x[a:b, c] = dsp.svf(np.ascontiguousarray(x[a:b, c]), sweep, 0.55, 0)


def excite(x, fc=3500.0, amt=0.18):
    """Harmonic exciter: saturated top end added back -> brighter, more present."""
    hp = dsp.butter_filter(x, fc, "high", 2)
    return x + dsp.butter_filter(np.tanh(hp * 4.0), fc, "high", 2) * amt


def trance_gate(n, sections, spb, pattern, names=("drop", "drop2")):
    g = np.ones(n)
    step = spb / 4
    for s in sections:
        if s.name not in names:
            continue
        t = s.start_bar * 4 * spb
        end = (s.start_bar + s.bars) * 4 * spb
        k = 0
        while t < end:
            if not pattern[k % len(pattern)]:
                a, b = int(t * SR), int((t + step) * SR)
                g[a:b] = 0.15
            t += step
            k += 1
    return dsp.onepole_lp(g, 300.0)


# ---------------------------------------------------------------- main
def compose(style_name="synthwave", seed=None, bpm=None, key=None, scale=None, progression=None,
            title=None, duration=None, glide_chords=None) -> Song:
    style = dict(STYLES[style_name])
    style.setdefault("family", FAMILY.get(style_name, "sw"))
    style.setdefault("lufs", LUFS_TARGET[style["family"]])
    if duration:
        style["target_s"] = duration
    rng = random.Random(seed)
    bpm = bpm or rng.randint(*style["bpm"])
    scale = scale or rng.choice(style["scales"])
    tonic = key if key is not None else rng.choice(["A", "D", "E", "F", "C", "G", "B", "F#", "C#"])
    k = Key(tonic, scale)
    sections = _arrangement(style, bpm, rng)
    total_bars = sections[-1].start_bar + sections[-1].bars
    prog_name = progression or rng.choice(style["progs"])
    try:
        chords = theory.progression(k, prog_name)
    except (ValueError, KeyError):
        chords = theory.generate_progression(k, 4, rng)
    bpc = style["bars_per_chord"]
    cevents = theory.chord_track(chords, bpc, total_bars, lo=50, hi=74, n_voices=4)
    # the last chord of the song rings on the tonic
    last = cevents[-1]
    tonic_ch = chords[0]
    cevents[-1] = ChordEvent(last.start, last.dur, tonic_ch, theory.voice_chord(tonic_ch, cevents[-2].voicing, 50, 74, 4))
    spb = 60.0 / bpm
    dur = total_bars * 4 * spb + 2.0
    w = style["words"]
    auto_title = f"{rng.choice(w)} {rng.choice(('DREAM', 'PULSE', 'ECHO', 'RUNNER', 'WAVES', 'SIGNAL', 'CITY', 'HEART', 'GLOW', 'MOTION'))}"
    title = title or auto_title   # rng is consumed either way, so a custom title never changes the music
    song = Song(style_name, title, bpm, k, sections, cevents, dur)
    sec_by = {s.name: s for s in sections}
    beat_t = lambda b: b * spb

    # ---------------- instruments for this song
    char = rng.choice(style["lead_char"])
    lead_patch = rng.choice(style["lead"])
    counter_patch = rng.choice(style["counter"])
    bass_patch, bass_pat = rng.choice(style["bass"])
    pad_patch = rng.choice(style["pad"])
    ch_patch = rng.choice(style["chords_inst"])
    arp_pool, arp_modes, arp_rate = style["arp"]
    arp_patch = rng.choice(arp_pool)
    kick_kind = _pick(rng, style["kick"])
    alt_pool = [p_ for p_ in style["lead"] if p_ != lead_patch]
    alt_lead = rng.choice(alt_pool) if alt_pool else lead_patch
    use_glide = (rng.random() < style["glide_chords"]) if glide_chords is None else glide_chords
    prng = random.Random(f"patch-{style_name}-{seed}-{lead_patch}")  # own stream: keeps the other picks stable
    lead_p = library.jitter_patch(lead_patch, prng)
    alt_p = library.jitter_patch(alt_lead, prng)
    lead_lo = 62 if lead_patch in ("synth_brass", "hoover", "screech", "raw_screech", "trumpet_synth", "horn_section") else 66

    # ---------------- lead melody with development
    cycle = len(chords) * bpc
    mel_bars = max(8 if bpm > 120 else 4, cycle)
    mel_seed = rng.randrange(10 ** 9)
    phrase_form = rng.choice(["AABA", "ABAC", "AAAB", "ABAB"])
    mel_cache = {}
    mrng = random.Random(mel_seed)
    use_hook = mrng.random() < 0.9
    # melody mode: genre hook or a slow flowing lead (long gliding notes)
    p_flow = {"jerk": 0.0, "dnb_liquid": 0.6, "synthwave": 0.45, "trance": 0.4, "future_bass": 0.35, "ambient_space": 0.8}.get(style_name, 0.2)
    mel_family = MELODY_MODE or ("flow" if mrng.random() < p_flow else style.get("hook", style["family"]))
    hook_lo = 62 if lead_lo == 62 else 69   # drop lead around C5-C6

    def melody_for(start_bar, variant):
        key_ = (start_bar % cycle, variant)
        if key_ not in mel_cache:
            sub = [ChordEvent(ce.start - start_bar * 4, ce.dur, ce.chord, ce.voicing) for ce in cevents
                   if ce.start + ce.dur > start_bar * 4 and ce.start < (start_bar + mel_bars) * 4]
            r = random.Random(mel_seed + key_[0] * 101 + variant * 7)
            if variant in (0, 1) and use_hook:
                # the hook: same motif in break and drops so the ear learns it
                r = random.Random(mel_seed + key_[0] * 101)
                mel_cache[key_] = harmony.make_hook(k, sub, mel_bars, mel_family, lo=hook_lo, hi=hook_lo + 18, rng=r)
            else:
                mel_cache[key_] = harmony.make_melody(k, sub, mel_bars, char, lo=lead_lo, hi=lead_lo + 19, rng=r,
                                                      phrase_form=phrase_form)
        return mel_cache[key_]

    lead_notes = []
    for s in sections:
        if not ({"lead", "lead_soft", "lead_f"} & s.roles):
            continue
        b = 0
        while b < s.bars:
            variant = 1 if (s.name == "break" or (s.name == "drop2" and b == 0)) else 0
            if s.name == "outro":
                variant = 2
            for n in melody_for(s.start_bar + b, variant):
                bb = n.start / 4
                if bb >= mel_bars or b + bb >= s.bars:
                    continue
                soft = "lead_soft" in s.roles or "lead_f" in s.roles
                nn = replace(n, start=(s.start_bar + b) * 4 + n.start, vel=n.vel * (0.6 if soft else 1.0))
                if s.name == "outro" and b + bb >= 1.0:
                    continue  # only a short tag in the outro
                lead_notes.append(nn)
            b += mel_bars
    # never let a lead note ring into the pre-drop gaps
    gaps = []
    for name in ("drop", "drop2"):
        if name in sec_by:
            d0 = sec_by[name].start_bar * 4
            gaps.append((d0 - 1.0, d0))
    switch_b = None   # beat where drop2 flips to a new lead sound / groove
    if "drop2" in sec_by and sec_by["drop2"].bars >= 4:
        s2_ = sec_by["drop2"]
        switch_b = (s2_.start_bar + s2_.bars // 2) * 4
        gaps.append((switch_b - 0.5, switch_b))
    lead_notes = [n for n in lead_notes if not any(a <= n.start < b for a, b in gaps)]
    def _trim_gap(nn):
        for ga, _ in gaps:
            if nn.start < ga < nn.end():
                return replace(nn, dur=max(0.12, ga - nn.start))
        return nn
    lead_notes = [_trim_gap(nn) for nn in lead_notes]
    lead_notes = harmony.fix_harmony(lead_notes, k, cevents)
    # drop2: harmony in thirds + delay throws
    harm_notes = []
    if "drop2" in sec_by:
        s2 = sec_by["drop2"]
        in2 = [n for n in lead_notes if s2.start_bar * 4 <= n.start < (s2.start_bar + s2.bars) * 4]
        harm_notes = harmony.harmonize_thirds(in2, k, cevents, below=True)
    throw_notes = [n for n in lead_notes if n.throw]
    # counter-melody answers in the lead's gaps ("перебивки")
    counter_notes = []
    for s in sections:
        if "counter" not in s.roles:
            continue
        a_, b_ = s.start_bar * 4, (s.start_bar + s.bars) * 4
        ln = [n for n in lead_notes if a_ <= n.start < b_]
        if s.name == "intro":
            ln = [Note(a_ + 1.5, 1.0, 60)] if s.bars >= 2 else []
        fills = harmony.response_fills(ln + [Note(b_, 0.1, 60)], k, cevents, b_, rng, lo=lead_lo - 9, hi=lead_lo + 7)  # an octave under the hook
        counter_notes += [f for f in fills if a_ <= f.start < b_ - 1.0]
    counter_notes = harmony.fix_harmony(counter_notes, k, cevents)

    # ---------------- bass
    bass_all = theory.make_bass(cevents, bass_pat, octave_root=(33 if "chip" not in bass_patch else 45) - (12 if bass_patch == "@808" else 0) + (0 if bass_patch == "@808" else 0), rng=rng, key=k)
    bass_notes = harmony.fix_harmony(_filter_notes(bass_all, sections, {"bass"}), k, cevents, strict_len=0.25)
    bass_notes = [n for n in bass_notes if not any(a <= n.start < b for a, b in gaps)]
    if style.get("kick_bass"):
        bass_notes = [n for n in bass_notes if not ({"kick", "kick_light"} & _sec_at(sections, int(n.start // 4)).roles)]
    sub_notes = []
    if style["sub"]:
        sub_notes = [replace(n, pitch=n.pitch - (12 if n.pitch >= 40 else 0)) for n in
                     _filter_notes(theory.make_bass(cevents, "root" if bass_pat not in ("reese",) else "reese", 33, rng), sections, {"bass"})]
        sub_notes = harmony.fix_harmony(sub_notes, k, cevents, strict_len=0.25)
        sub_notes = [n for n in sub_notes if not any(a <= n.start < b for a, b in gaps)]

    # ---------------- pads (optionally gliding voices) + chords + arp
    pad_lines = []
    pad_notes = []
    if pad_patch:
        ok = lambda beat: "pad" in _sec_at(sections, int(beat // 4)).roles
        if use_glide:
            pad_lines = harmony.glide_voices(cevents, ok)
        else:
            for ce in cevents:
                if ok(ce.start):
                    pad_notes += [Note(ce.start, ce.dur * 0.98, p, 0.7) for p in ce.voicing]
    chord_notes = []
    if ch_patch:
        stab = ch_patch in ("stab_brass", "brass_stab_hard", "supersaw_lead", "trance_lead", "jump_brass", "future_pluck")
        for ce in cevents:
            s = _sec_at(sections, int(ce.start // 4))
            if "chords" not in s.roles:
                continue
            if stab:
                rhythm = {"hardstyle": [0.5, 1.5, 2.5, 3.5], "future_bass": [0, 0.75, 1.5, 2.5, 3.0]}.get(style_name, [0, 1.5, 3.0])
                for bar in range(int(ce.dur // 4)):
                    for r in rhythm:
                        t = ce.start + bar * 4 + r
                        if any(a <= t < b for a, b in gaps):
                            continue
                        for p in ce.voicing:
                            chord_notes.append(Note(t, 0.35 if style_name != "future_bass" else 0.7,
                                                    p + (12 if "lead" in ch_patch else 0), 0.7,
                                                    scoop=-2.0 if style_name == "future_bass" else 0.0))
            else:
                for p in ce.voicing:
                    chord_notes.append(Note(ce.start, ce.dur * 0.9, p + 12 * (ch_patch == "crystal_pad"), 0.6))
    arp_all = theory.make_arp(cevents, rng.choice(arp_modes), arp_rate, octaves=2, gate=0.55)
    for nn in arp_all:
        nn.pitch += 12
    arp_notes = [n for n in _filter_notes(arp_all, sections, {"arp", "arp_f"}) if not any(a <= n.start < b for a, b in gaps)]

    # ---------------- render stems
    T = dur
    n = int(T * SR)
    Z = lambda: np.zeros((n, 2))
    stems = {}
    # Surge XT presets (optional): own seeded stream so the arrangement stays the same with or without Surge
    srng = random.Random(f"surge-{style_name}-{seed}")
    sx = {}
    if surge.available() and style.get("drums") != "chip":
        reg = hook_lo + 7
        for role, prob, kw in (("lead", 0.85, dict(register=reg)), ("alt", 0.8, dict(register=reg)),
                               ("big", 0.0 if style_name in ("dnb_liquid", "synthwave", "chiptune", "jerk") else 0.85, dict(register=reg)),
                               ("chords", 0.7, dict(register=64)), ("pad", 0.6, dict(register=60, sustain_min=0.6)),
                               ("arp", 0.6, dict(register=72)), ("counter", 0.5, dict(register=lead_lo))):
            if srng.random() < prob:
                srole = "lead" if role == "alt" else role
                if srole == "lead" and mel_family == "flow":
                    srole = "flowlead"
                if srole in ("lead", "flowlead") and LEAD_ROLE:
                    srole = LEAD_ROLE
                pr = surge.pick(style_name, srole, srng, **kw)
                if pr:
                    sx[role] = pr

    def LR(notes, which, seed_, gain=1.0):
        pr = sx.get(which)
        if pr:
            return surge.render(pr, notes, spb, T) * gain
        return Synth(lead_p if which == "lead" else alt_p, bpm, seed=seed_,
                     **({"gain": PATCHES[lead_patch]["gain"] * gain} if gain != 1.0 else {})).render(notes, T)

    if switch_b is not None and alt_lead != lead_patch:
        pre = [x for x in lead_notes if x.start < switch_b]
        post = [x for x in lead_notes if x.start >= switch_b]
        stems["lead"] = LR(pre, "lead", 1)
        if post:
            stems["lead"] += LR(post, "alt", 21)
    else:
        stems["lead"] = LR(lead_notes, "lead", 1)
    if harm_notes:
        stems["lead"] += LR(harm_notes, "lead", 11, 0.55)
    throw_buf = LR(throw_notes, "lead", 12) if throw_notes else Z()
    # drops: the hook doubled by a big supersaw preset (same octave), lead pushed up vs build/break
    drop_lead = [x for x in lead_notes if _sec_at(sections, int(x.start // 4)).name in ("drop", "drop2")]
    if "big" in sx and drop_lead:
        stems["lead"] += surge.render(sx["big"], drop_lead, spb, T) * 0.75
    lenv = np.full(n, 0.8)
    for s_ in sections:
        if s_.name in ("drop", "drop2"):
            lenv[int(s_.start_bar * 4 * spb * SR):int((s_.start_bar + s_.bars) * 4 * spb * SR)] = 1.25
    stems["lead"] *= dsp.onepole_lp(lenv, 30.0)[:, None]
    # drops: fat octave-down layer under the lead
    dl = style.get("drop_layer", "supersaw_lead")
    if dl:
        dnotes = [replace(x, pitch=x.pitch - 12, vel=x.vel * 0.9) for x in lead_notes
                  if _sec_at(sections, int(x.start // 4)).name in ("drop", "drop2")]
        if dnotes:
            stems["lead"] += Synth(dl, bpm, seed=31, gain=PATCHES[dl]["gain"] * 0.5).render(dnotes, T)
    if not counter_notes:
        stems["counter"] = Z()
    elif "counter" in sx:
        stems["counter"] = surge.render(sx["counter"], counter_notes, spb, T)
    else:
        stems["counter"] = Synth(library.jitter_patch(counter_patch, prng), bpm, seed=13).render(counter_notes, T)
    if bass_patch.startswith("@"):
        bk = bass_patch[1:]
        brng = np.random.default_rng(rng.randrange(10 ** 6))
        if bk == "808":
            stems["bass"] = basses.render_808(bass_notes, spb, T, brng, drive=style.get("bass_drive", 1.0),
                                              decay=0.5 if style_name == "future_bass" else 0.9)
        else:
            stems["bass"] = basses.render_bass(bass_notes, spb, T, brng, bk, drive=style.get("bass_drive", 1.0))
    elif bass_pat == "reverse" and bass_notes:
        stems["bass"] = basses.render_reverse_bass(bass_notes, spb, T, np.random.default_rng(2))
    else:
        stems["bass"] = Synth(bass_patch, bpm, seed=2).render(bass_notes, T) if bass_notes else Z()
    if sub_notes:
        stems["bass"] += Synth("sub_bass", bpm, seed=3, gain=0.6).render(sub_notes, T)
    if pad_lines:
        gp = dict(PATCHES[pad_patch]) if PATCHES[pad_patch]["fm"] is None else dict(PATCHES["glide_pad"])
        pad = Z()
        for i, line in enumerate(pad_lines):
            pad += Synth(gp, bpm, seed=40 + i, mono=True, glide=max(0.12, min(0.4, spb * 0.45)), chorus=0.0).render(line, T)
        stems["pad"] = dsp.chorus(pad, rate=0.4, depth_ms=3.0, base_ms=11, mix=0.6)
    elif pad_notes and "pad" in sx:
        stems["pad"] = surge.render(sx["pad"], pad_notes, spb, T, legato_slides=False)
    elif pad_notes:
        stems["pad"] = Synth(pad_patch, bpm, seed=4).render(pad_notes, T)
    else:
        stems["pad"] = Z()
    if ch_patch and "chords" in sx and chord_notes:
        stems["chords"] = surge.render(sx["chords"], chord_notes, spb, T, legato_slides=False) * 0.7
    else:
        stems["chords"] = Synth(ch_patch, bpm, seed=5, gain=PATCHES[ch_patch]["gain"] * 0.7).render(chord_notes, T) if ch_patch else Z()
    if "arp" in sx and arp_notes:
        stems["arp"] = surge.render(sx["arp"], arp_notes, spb, T, legato_slides=False)
    else:
        stems["arp"] = Synth(library.jitter_patch(arp_patch, prng, 0.7), bpm, seed=6).render(arp_notes, T)

    # filter automation: intro opens, build sweeps up, break darkens, outro closes
    sweep = np.full(n, 18000.0)
    for s in sections:
        a, b = int(s.start_bar * 4 * spb * SR), min(n, int((s.start_bar + s.bars) * 4 * spb * SR))
        if b <= a:
            continue
        x = np.linspace(0, 1, b - a)
        if s.name == "intro":
            sweep[a:b] = 350 * (5000 / 350) ** x
        elif s.name == "build":
            sweep[a:b] = 1200 * (18000 / 1200) ** (x ** 2.2)
        elif s.name == "break":
            sweep[a:b] = 1800
        elif s.name == "outro":
            sweep[a:b] = 2500 * (300 / 2500) ** x
    sweep[int(sections[-1].start_bar * 4 * spb * SR):] = np.minimum(sweep[int(sections[-1].start_bar * 4 * spb * SR):], 3000)
    for st in ("arp", "pad", "lead", "counter"):
        for c in range(2):
            stems[st][:, c] = dsp.svf(np.ascontiguousarray(stems[st][:, c]), sweep, 0.25, 0)

    # trance gate on pads in the drops
    if style["gate"] and rng.random() < style["gate"]:
        pat = rng.choice([[1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 0, 1, 0] * 2, [1, 0] * 8])
        stems["pad"] *= trance_gate(n, sections, spb, pat)[:, None]

    # ---------------- drums
    d = DrumKit(seed=rng.randint(0, 9999), tune_hz=theory.midi_to_hz(33 + (k.tonic - 33) % 12))
    kick_buf, sn_buf, hat_buf, perc_buf, fx_buf = Z(), Z(), Z(), Z(), Z()
    lanes = {"kick": [], "snare": [], "hat": [], "perc": []}
    drum_stutters = []
    hs_cache = {}
    hs_char = basses.KickCharacter(np.random.default_rng(rng.randrange(10 ** 6)), raw=kick_kind == "hs_raw")
    kick_disp = []
    from . import samples as SMP
    sm_snare = rng.choice(SMP.SNARE_TOP)
    sm_hat = rng.choice(SMP.HATS)
    sm_ohat = rng.choice(SMP.OPEN_HATS)
    krng = random.Random(f"kit-{style_name}-{seed}")       # own stream: kit choice doesn't shift the arrangement
    p808 = {"fb": 0.9, "sw": 0.5}.get(style["family"], 0.3)
    if krng.random() < p808:
        sm_snare = krng.choice(SMP.K808_SNARE + SMP.K808_CLAP * 4)
    if krng.random() < p808:
        sm_hat = krng.choice(SMP.K808_HAT)
        sm_ohat = krng.choice(SMP.K808_OHAT)
    kick_layer = krng.choice(SMP.KICK_LAYER) if krng.random() < 0.7 else None
    use_samples = style["drums"] != "chip"
    roll_t = []
    fills_ev = []
    perc_pool = style["perc"] or []
    swing = style["swing"]
    for bar in range(total_bars):
        sec = _sec_at(sections, bar)
        bis = bar - sec.start_bar
        dk = style["drums"]
        if switch_b is not None and dk in ("dnb",) and switch_b <= bar * 4 < switch_b + 8:
            dk = "halftime_hard"
        hits_ = _drum_bar(dk, bis, sec, rng)
        if _drum_bar.last_fill:
            fv, fbig = _drum_bar.last_fill
            f0 = (bar * 4 + (2 if fbig else 3)) * spb
            fills_ev.append(dict(t=f0, t1=(bar + 1) * 4 * spb, type=fv, big=fbig, section=sec.name))
        if "roll" in sec.roles and bis == sec.bars - 1:
            fills_ev.append(dict(t=bar * 4 * spb, t1=(bar + 1) * 4 * spb, type="buildroll", big=True, section=sec.name))
        ksteps = sorted(st_ for inst_, st_, _ in hits_ if inst_ == "kick") + [16]
        for inst, step, vel in hits_:
            sw = swing * spb / 2 if step % 2 == 1 else 0.0
            t = (bar * 4 + step / 4) * spb + sw + (rng.uniform(-0.002, 0.002) if inst != "kick" else 0)
            if any(a * spb <= t < b * spb for a, b in gaps):
                continue
            if inst == "kick":
                if kick_kind in ("thunder", "classic", "euphoric", "raw_classic", "hs_euphoric", "hs_raw"):
                    nxt = min(st_ for st_ in ksteps if st_ > step)
                    d.kick_len = min(0.95 * (nxt - step) / 4 * spb, 0.42)
                    if bar + 1 < total_bars and nxt == 16 and not ({"kick", "kick_light"} & _sec_at(sections, bar + 1).roles):
                        d.kick_len = min(0.95 * spb, 0.42)
                    if kick_kind.startswith("hs_"):
                        rc = harmony.chord_at(cevents, t / spb + 0.01).chord
                        hz_, mid_ = basses.kick_root_hz(rc.bass if rc.bass is not None else rc.root)
                        if "roll" in sec.roles:   # build kick roll: tail pitch climbs
                            prog_ = 1 - min(1.0, ((sec.start_bar + sec.bars) * 4 - t / spb) / 8)
                            hz_ *= 2 ** (prog_ * 7 / 12)
                        key_ = (round(hz_, 2), round(d.kick_len, 3))
                        if key_ not in hs_cache:
                            hs_cache[key_] = basses.hardstyle_kick(hz_, d.kick_len, hs_char, seed=len(hs_cache))
                        kick_disp.append(Note(t / spb, d.kick_len / spb, mid_, vel))
                        dsp.place(kick_buf, hs_cache[key_] * vel, t, 0.9); lanes["kick"].append((t, vel))
                        continue
                    kick_disp.append(Note(t / spb, d.kick_len / spb, d.tail_midi(), vel))
                dsp.place(kick_buf, d.kick(kick_kind, vel), t, 0.9); lanes["kick"].append((t, vel))
                if kick_layer is not None and kick_kind not in ("thunder", "classic", "euphoric", "raw_classic"):
                    dsp.place(kick_buf, SMP.trimmed(kick_layer, min(0.45, spb * 0.9), 0.05) * vel, t, 0.3)
            elif inst in ("snare", "snare_roll", "ghost"):
                g = {"snare": 0.75, "snare_roll": 0.55, "ghost": 0.25}[inst]
                if inst == "snare_roll" and "roll" in sec.roles:
                    # build roll: pitch rises and gets louder towards the drop
                    left_b = (sec.start_bar + sec.bars) * 4 - (bar * 4 + step / 4)
                    prog = 1 - min(1.0, left_b / (2 * 4))
                    vel = 0.35 + 0.6 * prog
                    if use_samples:
                        dsp.place(sn_buf, SMP.pitched(sm_snare, 0.9 + 0.6 * prog), t, 0.55 * vel)
                    dsp.place(sn_buf, d.snare(style["snare"], vel), t, 0.4 * vel)
                    roll_t.append(t)
                else:
                    dsp.place(sn_buf, d.snare(style["snare"], vel), t, g)
                    if use_samples and inst != "ghost":
                        dsp.place(sn_buf, SMP.pitched(sm_snare, 1.0 + rng.uniform(-0.02, 0.02)), t, g * 0.55 * vel)
                    if inst == "snare" and style["drums"] not in ("synthwave",):
                        dsp.place(sn_buf, d.clap(vel), t, 0.3)
                lanes["snare"].append((t, vel * (0.4 if inst == "ghost" else 1)))
            elif inst == "flam":
                dsp.place(sn_buf, d.snare(style["snare"], vel), t - 0.025, 0.35)
            elif inst == "snap":
                dsp.place(perc_buf, dsp.pan(SMP.load(rng.choice(SMP.SNAPS)), rng.uniform(-0.4, 0.4)), t, 0.45 * vel)
                lanes["perc"].append((t, vel))
            elif inst == "drumstutter":
                drum_stutters.append(t)
            elif inst == "clap":
                dsp.place(sn_buf, d.snare(style["snare"], vel) if style["snare"] == "hardstyle" else d.clap(vel), t, 0.6)
                lanes["snare"].append((t, vel))
            elif inst == "noise_snare":
                y = d.hat(False, vel, "noise")
                dsp.place(sn_buf, np.repeat(y[::4], 4)[: len(y)], t, 0.6); lanes["snare"].append((t, vel))
            elif inst in ("hat", "noise_hat"):
                if inst == "hat" and use_samples:
                    h = SMP.pitched(sm_hat, 1.0 + rng.uniform(-0.03, 0.03)) * vel
                    h = h * 0.7
                    dsp.place(hat_buf, dsp.pan(h, rng.uniform(-0.3, 0.3)), t, 0.3)
                else:
                    h = d.hat(False, vel, "808" if inst == "hat" else "noise")
                    dsp.place(hat_buf, dsp.pan(h, rng.uniform(-0.3, 0.3)), t, 0.35)
                lanes["hat"].append((t, vel))
            elif inst == "hat_o":
                if use_samples:
                    dsp.place(hat_buf, dsp.pan(SMP.trimmed(sm_ohat, spb * 0.45, 0.04) * vel, 0.2), t, 0.25)
                else:
                    dsp.place(hat_buf, dsp.pan(d.hat(True, vel), 0.2), t, 0.25)
                lanes["hat"].append((t, vel))
            elif inst == "hat_roll":
                nh = rng.choice([3, 4, 6])
                sp = spb / 4 * 2 / nh
                dsp.place(hat_buf, dsp.pan(d.hat_roll(nh, sp, vel), 0.25), t, 0.3)
                lanes["hat"] += [(t + j * sp, vel) for j in range(nh)]
            elif inst == "perc" and perc_pool:
                kind = perc_pool[(bar + step) % len(perc_pool)]
                if kind == "shaker":
                    y = d.shaker(vel)
                elif kind == "tambourine":
                    y = d.perc("tambourine", vel)
                else:
                    y = d.perc(kind, vel, pitch=1.0 + 0.12 * ((step // 4) % 3))
                dsp.place(perc_buf, dsp.pan(y, rng.uniform(-0.6, 0.6)), t, 0.3); lanes["perc"].append((t, vel))
            elif inst == "shaker":
                dsp.place(hat_buf, dsp.pan(d.shaker(vel), 0.3), t, 0.4); lanes["hat"].append((t, vel))
            elif inst == "tom":
                if use_samples:
                    tn = SMP.TOMS[min(2, int((step - 8) / 3))] if step >= 8 else SMP.TOMS[0]
                    dsp.place(sn_buf, dsp.pan(SMP.trimmed(tn, 0.35, 0.08), 0.3 - 0.1 * (step % 6)), t, 0.4 * vel)
                dsp.place(sn_buf, d.tom(95 * (1 + 0.18 * (15 - step)), vel), t, 0.4); lanes["perc"].append((t, vel))
    fx_events = []
    impacts = []
    for i, s in enumerate(sections):
        t0 = s.start_bar * 4 * spb
        if s.name in ("drop", "drop2"):
            dsp.place(fx_buf, d.impact(0.8), t0, 0.45)
            dsp.place(fx_buf, d.crash(0.8), t0, 0.35)
            dsp.place(fx_buf, dsp.pan(SMP.trimmed(rng.choice(SMP.CRASH), 2.2, 0.8), 0.15), t0, 0.35)
            dsp.place(fx_buf, SMP.trimmed("perc_impact1", 1.1, 0.3), t0, 0.35)
            if i and len(roll_t):
                # reverse snare swell into the drop
                sn_ = SMP.load(sm_snare)
                pad_ = np.zeros(int(2.0 * SR)); sn_ = sn_[: len(pad_)]; pad_[: len(sn_)] = sn_
                wet_ = dsp.wet_reverb(np.stack([pad_, pad_], 1), seconds=1.6, mix=1.0)[::-1]
                wet_ = wet_[-int(min(1.5, 2 * spb) * SR):]
                dsp.place(fx_buf, wet_ / (np.max(np.abs(wet_)) + 1e-9), t0 - len(wet_) / SR, 0.3)
            dsp.place(fx_buf, d.perc("sub_drop", 0.9), t0, 0.35)
            impacts.append(t0)
            pre = sections[i - 1] if i else None
            if pre is not None:
                rl = min(pre.bars, 2 if pre.name != "build" else pre.bars) * 4 * spb - spb
                r = d.riser(rl, 0.8, tonal_hz=theory.midi_to_hz(48 + k.tonic))
                dsp.place(fx_buf, np.stack([r, np.roll(r, 300)], axis=1), t0 - spb - rl, 0.22)
                dsp.place(fx_buf, d.reverse_cymbal(min(1.5, rl)), t0 - spb - min(1.5, rl), 0.2)
                dsp.place(fx_buf, d.perc("laser", 0.7), t0 - spb * 0.5, 0.15)
                fx_events.append(dict(kind="gap", t=t0 - spb, dur=spb))
        elif s.name == "break":
            dsp.place(fx_buf, np.stack([d.downlifter(2.5)] * 2, axis=1), t0, 0.25)
        if s.name == "drop2" and switch_b is not None:
            ts_ = switch_b * spb
            dsp.place(fx_buf, d.reverse_cymbal(1.2), ts_ - 0.5 * spb - 1.2, 0.3)
            dsp.place(fx_buf, d.impact(0.9), ts_, 0.45)
            dsp.place(fx_buf, d.perc("sub_drop", 0.9), ts_, 0.3)
            impacts.append(ts_)
            fx_events.append(dict(kind="gap", t=ts_ - 0.5 * spb, dur=0.5 * spb))
            fx_events.append(dict(kind="switch", t=ts_, dur=spb))
        elif s.name == "outro":
            dsp.place(fx_buf, d.impact(0.5), t0, 0.25)
            impacts.append(t0)
    stems.update(kick=kick_buf, snare=sn_buf, hats=hat_buf, perc=perc_buf, fx=fx_buf)
    # drum chops at the end of big fills
    for ts_ in drum_stutters:
        for st_ in ("kick", "snare", "hats", "perc"):
            stutter(stems[st_], ts_ - 0.75 * spb, ts_ + 0.25 * spb, spb)
        fx_events.append(dict(kind="stutter", t=ts_ - 0.75 * spb, dur=spb))
    # heavy overdriven bass + per-stem mixing chains (EQ, peak control, mono lows)
    if style.get("bass_drive") and not bass_patch.startswith("@"):
        stems["bass"] = mixing.overdrive_bass(stems["bass"], style["bass_drive"])
    stems = mixing.process_stems(stems)

    # ---------------- levels: short-term loudness (LUFS-S) relative to the kick (research 4.2)
    fam = style.get("family", "sw")
    REL = LEVELS[fam]
    drop = sec_by.get("drop", sections[0])
    da, db = int(drop.start_bar * 4 * spb * SR), int((drop.start_bar + drop.bars) * 4 * spb * SR)
    KICK_REF = -11.0
    for name in ("kick", "bass", "snare", "lead", "chords", "pad", "hats", "arp", "counter", "perc"):
        if name not in stems or not np.any(stems[name]):
            continue
        if name in ("snare", "hats", "perc"):
            stems[name] = mixing.peak_clip(stems[name], da, db, 3.0)
        cur = mixing.lufs_s(stems[name], da, db)
        if not np.isfinite(cur):
            cur = mixing.lufs_s(stems[name])
        if not np.isfinite(cur):
            continue
        tgt = KICK_REF + REL.get(name, -10.0) + style.get("mix_rel", {}).get(name, 0.0)
        gain = float(np.clip(10 ** ((tgt - cur) / 20), 0.02, 30))
        stems[name] = stems[name] * gain
        if name == "lead":
            throw_buf = throw_buf * gain
    # parallel crunch on kick (not on hardstyle kicks: already distorted) and snare
    if not kick_kind.startswith("hs_") and kick_kind != "thunder":
        stems["kick"] = mixing.crunch(stems["kick"], 14.0, 900.0, -16.0 + 3 * style.get("drum_crunch", 1.0))
    stems["snare"] = mixing.crunch(stems["snare"], 16.0, 800.0, -15.0)

    # ---------------- interruptions on the music bus
    music = stems["lead"] + stems["arp"] + stems["chords"] + stems["counter"]
    if "drop" in sec_by:
        s = sec_by["drop"]
        end = (s.start_bar + s.bars) * 4 * spb
        stutter(music, end - spb, end, spb)
        fx_events.append(dict(kind="stutter", t=end - spb, dur=spb))
        mid = (s.start_bar + s.bars // 2) * 4 * spb
        if s.bars >= 8:
            stutter(music, mid - spb * 0.5, mid, spb)
            fx_events.append(dict(kind="stutter", t=mid - spb * 0.5, dur=spb * 0.5))
        # bit-crush hit at the end of the 2nd bar, low-pass dive into the final stutter
        if s.bars >= 4:
            tc = (s.start_bar + 2) * 4 * spb - spb
            for buf in (music, stems["bass"]):
                crush(buf, tc, tc + spb)
            fx_events.append(dict(kind="crush", t=tc, dur=spb))
            for buf in (music, stems["bass"], stems["pad"]):
                filter_dive(buf, end - 5 * spb, end - spb)
            fx_events.append(dict(kind="dive", t=end - 5 * spb, dur=4 * spb))
    if "drop2" in sec_by:
        s = sec_by["drop2"]
        # whole-band beat repeat at the end of the 2nd bar
        tb = (s.start_bar + 2) * 4 * spb
        for st_ in ("kick", "snare", "hats", "perc", "bass"):
            stutter(stems[st_], tb - spb, tb, spb)
        stutter(music, tb - spb, tb, spb)
        fx_events.append(dict(kind="stutter", t=tb - spb, dur=spb))
    if "break" in sec_by:
        s = sec_by["break"]
        end = (s.start_bar + s.bars) * 4 * spb
        for st in ("pad",):
            tape_stop(stems[st], end - spb * 2, end - spb)
        tape_stop(music, end - spb * 2, end - spb)
        fx_events.append(dict(kind="tapestop", t=end - spb * 2, dur=spb))

    # ---------------- mix: buses, per-bus sidechain shapes, returns, master (research 4.1-4.6, 6)
    kicks_t = [t for t, _ in lanes["kick"]]
    rv_len, rv_mix = style["reverb"]
    snare_t = [t for t, v in lanes["snare"] if v > 0.5]
    if style["snare"] == "gated":
        gated = dsp.wet_reverb(stems["snare"], seconds=1.2, mix=1.0, damp=0.2)
        gate = dsp.sidechain_curve(n, [t + 0.28 for t in snare_t], depth=1.0, attack=0.02, release=spb * 3, shape=0.3)
        stems["snare"] = stems["snare"] + gated * 1.4 * np.minimum(1, gate)[:, None]
    scl = min(1.0, style["sidechain"] / 0.7)
    duck = lambda depth, rec, shape, hold=0.0: mixing.duck_curve(n, kicks_t, spb, depth, hold, rec, shape)[:, None]
    d_bass = duck(1.0 if style["sidechain"] >= 0.5 else 0.7, 0.42, 2.5, 0.05)
    d_chords = duck((0.95 if fam == "fb" else 0.8) * scl, 0.5, 2.0)
    d_pad = duck(0.85 * scl, 0.5, 1.8)
    d_lead = duck(0.45 * scl, 0.3, 1.5)
    d_ret = duck(1.0 * scl, 0.6, 1.5)
    d_hat = duck(0.2 * scl, 0.2, 1.0)
    # future bass: rhythmic 1/8 volume wobble on the chords in the drops
    if fam == "fb" and np.any(stems["chords"]):
        tt_ = np.arange(n) / SR
        wob = 0.55 + 0.45 * (1 - ((tt_ / (spb / 2)) % 1.0)) ** 1.5
        inside = np.zeros(n)
        for s_ in sections:
            if s_.name in ("drop", "drop2"):
                inside[int(s_.start_bar * 4 * spb * SR):int((s_.start_bar + s_.bars) * 4 * spb * SR)] = 1
        stems["chords"] = stems["chords"] * (1 - inside * (1 - wob))[:, None]
    # pre-drop: high-pass dive on everything musical over the last 2 bars of the build
    if "build" in sec_by:
        sb = sec_by["build"]
        e_ = (sb.start_bar + sb.bars) * 4 * spb
        a_ = int(max(sb.start_bar * 4 * spb, e_ - 8 * spb) * SR); b_ = int(e_ * SR)
        if b_ > a_ + 100:
            hp_ = 20 * (450 / 20) ** np.linspace(0, 1, b_ - a_) ** 1.5
            for st_ in ("bass", "pad", "kick"):
                for c in range(2):
                    stems[st_][a_:b_, c] = dsp.svf(np.ascontiguousarray(stems[st_][a_:b_, c]), hp_, 0.1, 2)
            for c in range(2):
                music[a_:b_, c] = dsp.svf(np.ascontiguousarray(music[a_:b_, c]), hp_, 0.1, 2)
            fx_events.append(dict(kind="hpdive", t=a_ / SR, dur=(b_ - a_) / SR))
    music = mixing.music_bus(music, style.get("ott", 0.3), 1.6)
    if fam == "fb":
        stems["chords"] = mixing.ott(stems["chords"], 0.5)
    send = stems["pad"] * 0.5 + music * 0.4 + stems["snare"] * 0.15
    wet = mixing.returns_eq(dsp.wet_reverb(send, seconds=rv_len, mix=1.0, damp=0.5, predelay=0.03))
    lead_d = mixing.returns_eq(dsp.delay(music, spb * 0.75, 0.33, 0.25) - music)
    throw = dsp.delay(throw_buf, spb * 0.75, 0.62, 1.0, n_echo=10) - throw_buf
    drums = mixing.drum_bus(stems["kick"] + stems["snare"] + stems["hats"] * d_hat + stems["perc"] * d_hat)
    mix = (drums + stems["fx"] * 0.7 + stems["bass"] * d_bass + stems["pad"] * d_pad
           + music * (d_lead * 0.5 + d_chords * 0.5) + (lead_d + throw * 0.45 + wet * rv_mix * 1.1) * d_ret)
    # pre-drop silence gaps (fx keep running)
    gap_g = _gain_range(n, [(e["t"], e["t"] + e["dur"]) for e in fx_events if e["kind"] == "gap"], floor=0.0)
    mix = (mix - stems["fx"] * 0.7) * gap_g[:, None] + stems["fx"] * 0.7
    # vinyl brake of the whole band at the end of drop 2
    if "drop2" in sec_by:
        s = sec_by["drop2"]
        e2 = (s.start_bar + s.bars) * 4 * spb
        tape_stop(mix, e2 - spb, e2)
        fx_events.append(dict(kind="tapestop", t=e2 - spb, dur=spb))
    # macro dynamics: intro/break sit lower, build climbs, drops hit at full level
    sg = np.ones(n)
    for s_ in sections:
        a_, b_ = int(s_.start_bar * 4 * spb * SR), min(n, int((s_.start_bar + s_.bars) * 4 * spb * SR))
        if b_ <= a_:
            continue
        db_ = {"intro": -6.0, "break": -6.5, "outro": -4.0}.get(s_.name, 0.0)
        if s_.name == "build":
            sg[a_:b_] = 10 ** (np.linspace(-10.0, -4.0, b_ - a_) / 20)
        else:
            sg[a_:b_] = 10 ** (db_ / 20)
    sg = dsp.onepole_lp(sg, 20.0)
    mix = mix * sg[:, None]
    mix = mixing.master(mix, target_lufs=style.get("lufs", -8.0), brightness=BRIGHT.get(fam, 1.0))
    # seamless Shorts loop: the outro sinks to ~25% and its last LOOP_S crossfade into the first beat,
    # so when the player jumps back to 0 the sound continues without a gap
    fo = int(1.8 * SR); mix[-fo:] *= (0.25 + 0.75 * np.linspace(1, 0, fo) ** 1.5)[:, None]
    L = int(LOOP_S * SR)
    w = np.linspace(0, np.pi / 2, L)[:, None]
    tail = mix[-L:].copy()
    mix = mix[:-L].copy()
    mix[:L] = mix[:L] * np.sin(w) + tail * np.cos(w)
    fi = int(0.003 * SR); mix[:fi] *= np.linspace(0, 1, fi)[:, None]
    song.audio = mix.astype(np.float32)
    stems["lead"] = stems["lead"] + throw * 0.3
    stems["music"] = music
    song.stems = {k_: v[: len(song.audio)].astype(np.float32) for k_, v in stems.items()}
    dur = len(song.audio) / SR
    song.duration = dur
    # sub hits for visuals: drop downbeats + long / new-pitch bass notes inside drops
    drop_spans = [(s_.start_bar * 4, (s_.start_bar + s_.bars) * 4) for s_ in sections if s_.name in ("drop", "drop2")]
    bass_drops, last_b, last_p = [], -9.0, None
    for a_, _ in drop_spans:
        bass_drops.append(dict(t=beat_t(a_), strength=1.0, p=None))
    for n_ in sorted(bass_notes + kick_disp, key=lambda x: x.start):
        if not any(a_ <= n_.start < b_ for a_, b_ in drop_spans):
            continue
        big = n_.dur >= 0.75 or (last_p is not None and n_.pitch != last_p)
        last_p = n_.pitch
        if big and n_.start - last_b >= 1.0 and not any(abs(beat_t(n_.start) - d["t"]) < spb * 0.9 for d in bass_drops):
            bass_drops.append(dict(t=beat_t(n_.start), strength=round(0.5 + 0.4 * min(1.0, n_.dur / 2), 2), p=n_.pitch))
            last_b = n_.start
    bass_drops.sort(key=lambda d: d["t"])

    # ---------------- events for visuals
    flats = lambda ch: k.prefers_flats() or ch.roman.startswith("b")
    nd = lambda lst: [dict(t=beat_t(x.start), dur=beat_t(x.dur), p=x.pitch, v=x.vel, slide=x.slide, bend=x.bend) for x in lst]
    song.events = dict(
        bpm=bpm, spb=spb, key=k.name(), style=style["label"], style_id=style_name, title=title, scale=scale, tonic=k.tonic,
        progression=[c.name(flats(c)) for c in chords], prog_name=prog_name if isinstance(prog_name, str) else "custom",
        lead_patch=("surge:" + sx["lead"]["name"]) if "lead" in sx else lead_patch, alt_lead=alt_lead if switch_b is not None else None,
        bass_patch=bass_patch if not style.get("kick_bass") else "kick tail (" + kick_kind + ")", pad_patch=pad_patch, counter_patch=counter_patch,
        chords_patch=ch_patch, arp_patch=arp_patch, kick=kick_kind, melody_char=char, glide_chords=bool(pad_lines),
        kicks=kicks_t, snares=snare_t, hats=[t for t, _ in lanes["hat"]], impacts=impacts,
        lanes={k_: [list(x) for x in v] for k_, v in lanes.items()}, fx=fx_events,
        sections=[dict(name=s.name, t=beat_t(s.start_bar * 4), dur=beat_t(s.bars * 4), energy=s.energy) for s in sections],
        chords=[dict(t=beat_t(c.start), dur=beat_t(c.dur), name=c.chord.name(flats(c.chord)), root=c.chord.root,
                     pcs=c.chord.pcs, intervals=list(c.chord.intervals), voicing=c.voicing) for c in cevents],
        lead=nd(lead_notes), harm=nd(harm_notes), counter=nd(counter_notes), bass=nd(bass_notes + kick_disp), arp=nd(arp_notes),
        pad_lines=[nd(l) for l in pad_lines] if pad_lines else [nd([x for x in pad_notes if x.pitch == x.pitch])],
        glide_s=max(0.12, min(0.4, spb * 0.45)), lead_glide=PATCHES[lead_patch]["glide"],
        duration=dur, loop_s=LOOP_S, kick_flavour=hs_char.name if hs_char is not None else None, melody_mode=mel_family if use_hook else "classic", surge={k_: v["path"] for k_, v in sx.items()},
        fills=fills_ev, bass_drops=bass_drops,
    )
    return song


def save_wav(path, audio):
    import wave
    a = np.clip(audio, -1, 1)
    w = wave.open(str(path), "wb")
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((a * 32767).astype("<i2").tobytes())
    w.close()
