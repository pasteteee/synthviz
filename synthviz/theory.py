"""Music theory core: notes, intervals, scales/modes, chords, roman numerals,
functional harmony, voice leading, melody / bass / arp generation.

Everything is expressed in MIDI note numbers and beats, so the composer and the
synth engine share one language.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field, replace

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

INTERVALS = {
    "P1": 0, "m2": 1, "M2": 2, "m3": 3, "M3": 4, "P4": 5, "TT": 6, "P5": 7,
    "m6": 8, "M6": 9, "m7": 10, "M7": 11, "P8": 12, "m9": 13, "M9": 14,
    "P11": 17, "#11": 18, "M13": 21,
}

# Just-intonation ratios of intervals (used by visuals: Lissajous shapes)
JUST_RATIOS = {0: (1, 1), 1: (16, 15), 2: (9, 8), 3: (6, 5), 4: (5, 4), 5: (4, 3),
               6: (45, 32), 7: (3, 2), 8: (8, 5), 9: (5, 3), 10: (9, 5), 11: (15, 8)}

SCALES = {
    # diatonic modes
    "major": [0, 2, 4, 5, 7, 9, 11],
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    # minor variants
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    # pentatonic / blues
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    # colours
    "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],
    "lydian_dominant": [0, 2, 4, 6, 7, 9, 10],
    "hungarian_minor": [0, 2, 3, 6, 7, 8, 11],
    "double_harmonic": [0, 1, 4, 5, 7, 8, 11],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "in_sen": [0, 1, 5, 7, 10],
    "hirajoshi": [0, 2, 3, 7, 8],
}

CHORD_TYPES = {
    "5": [0, 7],
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "mMaj7": [0, 3, 7, 11],
    "m7b5": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "7sus4": [0, 5, 7, 10],
    "add9": [0, 4, 7, 14],
    "madd9": [0, 3, 7, 14],
    "9": [0, 4, 7, 10, 14],
    "maj9": [0, 4, 7, 11, 14],
    "m9": [0, 3, 7, 10, 14],
    "m11": [0, 3, 7, 10, 14, 17],
    "maj7#11": [0, 4, 7, 11, 18],
    "13": [0, 4, 7, 10, 14, 21],
    "7b9": [0, 4, 7, 10, 13],
}

# roman numeral degree -> semitones above the tonic (major reference)
_ROMAN = {"I": 0, "II": 2, "III": 4, "IV": 5, "V": 7, "VI": 9, "VII": 11}


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def note_name(m: int, flats: bool = False) -> str:
    names = FLAT_NAMES if flats else NOTE_NAMES
    return f"{names[m % 12]}{m // 12 - 1}"


def pc_name(pc: int, flats: bool = False) -> str:
    return (FLAT_NAMES if flats else NOTE_NAMES)[pc % 12]


def parse_note(s: str) -> int:
    """'A4' -> 69, 'C#3' -> 49, 'Eb' -> pitch class 3 (octave 4 assumed)."""
    s = s.strip()
    name = s[0].upper()
    rest = s[1:]
    acc = 0
    while rest and rest[0] in "#b":
        acc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    octv = int(rest) if rest else 4
    return 12 * (octv + 1) + NOTE_NAMES.index(name) + acc


# --------------------------------------------------------------------------
# Chords
# --------------------------------------------------------------------------
@dataclass
class Chord:
    root: int                 # pitch class 0..11
    quality: str              # key into CHORD_TYPES
    bass: int | None = None   # slash-chord bass pitch class
    roman: str = ""

    @property
    def intervals(self):
        return CHORD_TYPES[self.quality]

    @property
    def pcs(self):
        return sorted({(self.root + i) % 12 for i in self.intervals})

    def tones(self, octave_root_midi: int):
        return [octave_root_midi + i for i in self.intervals]

    def name(self, flats=True):
        n = pc_name(self.root, flats) + self.quality
        if self.bass is not None and self.bass != self.root:
            n += "/" + pc_name(self.bass, flats)
        return n


def chord_from_name(name: str) -> Chord:
    """'Am9', 'F#m7b5', 'Bbmaj7', 'C/E'."""
    bass = None
    if "/" in name:
        name, b = name.split("/")
        bass = parse_note(b) % 12
    root = parse_note(name[0] + ("#" if name[1:2] == "#" else "b" if name[1:2] == "b" else "")) % 12
    q = name[1:].lstrip("#b") if len(name) > 1 else ""
    if q not in CHORD_TYPES:
        raise ValueError(f"unknown chord quality {q!r}")
    return Chord(root, q, bass)


def stack_thirds(scale_pcs, degree, n_notes=3):
    """Diatonic chord built by stacking scale thirds on a 0-based degree."""
    L = len(scale_pcs)
    out = []
    for k in range(n_notes):
        idx = degree + 2 * k
        pc = scale_pcs[idx % L] + 12 * (idx // L)
        out.append(pc)
    root = out[0]
    return root % 12, [p - root for p in out]


def identify(intervals) -> str:
    iv = sorted(set(intervals))
    for q, ref in CHORD_TYPES.items():
        if sorted(set(ref)) == iv:
            return q
    # reduce compound to simple and retry
    red = sorted({i % 12 for i in iv})
    for q, ref in CHORD_TYPES.items():
        if sorted({i % 12 for i in ref}) == red:
            return q
    return ""


class Key:
    def __init__(self, tonic: int | str, scale: str = "minor"):
        self.tonic = parse_note(tonic) % 12 if isinstance(tonic, str) else tonic % 12
        self.scale = scale
        self.intervals = SCALES[scale]

    @property
    def pcs(self):
        return [(self.tonic + i) % 12 for i in self.intervals]

    def name(self):
        nice = self.scale.replace("_", " ")
        return f"{pc_name(self.tonic, flats=self.prefers_flats())} {nice}"

    _MODE_OFFSET = {"major": 0, "ionian": 0, "dorian": 2, "phrygian": 4, "lydian": 5, "mixolydian": 7,
                    "minor": 9, "aeolian": 9, "locrian": 11, "harmonic_minor": 9, "melodic_minor": 9,
                    "minor_pentatonic": 9, "blues": 9, "major_pentatonic": 0}

    def prefers_flats(self):
        """Spell like the key signature of the parent major key."""
        parent = (self.tonic - self._MODE_OFFSET.get(self.scale, 0)) % 12
        return parent in (5, 10, 3, 8, 1)

    def degree_chord(self, degree: int, n_notes=3) -> Chord:
        """1-based diatonic degree -> Chord (triad / 7th / 9th by n_notes)."""
        root, ivs = stack_thirds(self.intervals, degree - 1, n_notes)
        return Chord((self.tonic + root) % 12, identify(ivs))

    def in_scale(self, midi: int) -> bool:
        return (midi - self.tonic) % 12 in self.intervals

    def scale_notes(self, lo: int, hi: int):
        return [m for m in range(lo, hi + 1) if self.in_scale(m)]

    def snap(self, midi: int) -> int:
        """Nearest scale tone."""
        for d in (0, -1, 1, -2, 2):
            if self.in_scale(midi + d):
                return midi + d
        return midi

    def roman(self, rn: str) -> Chord:
        """Roman numeral relative to the tonic, e.g. 'i', 'bVI', 'IVmaj7',
        'V7', 'ii9', 'bVIIadd9', 'iv/1' (slash: bass on degree 1).
        Case gives triad quality, suffix replaces/extends it."""
        s = rn
        bass_deg = None
        if "/" in s:
            s, b = s.split("/")
            bass_deg = int(b)
        acc = 0
        while s and s[0] in "b#":
            acc += -1 if s[0] == "b" else 1
            s = s[1:]
        mm = re.match(r"(VII|VI|V|IV|III|II|I)", s, re.I)
        if not mm:
            raise ValueError(f"cannot parse roman numeral {rn!r}")
        num = mm.group(1)
        suffix = s[len(num):]
        minor = num.islower()
        semis = _ROMAN[num.upper()] + acc
        root = (self.tonic + semis) % 12
        if suffix in CHORD_TYPES:
            q = suffix
            if minor and suffix in ("7", "9", "11", "6", "add9", "maj7"):
                q = {"7": "m7", "9": "m9", "11": "m11", "6": "m6", "add9": "madd9", "maj7": "mMaj7"}[suffix]
            elif not minor and suffix == "":
                q = ""
        elif suffix == "o" or suffix == "°":
            q = "dim"
        elif suffix == "ø":
            q = "m7b5"
        else:
            raise ValueError(f"cannot parse roman numeral {rn!r}")
        if minor and q == "":
            q = "m"
        bass = None
        if bass_deg is not None:
            bass = (self.tonic + self.intervals[(bass_deg - 1) % len(self.intervals)]) % 12
        return Chord(root, q, bass, roman=rn)


# --------------------------------------------------------------------------
# Progressions
# --------------------------------------------------------------------------
PROGRESSIONS = {
    # name: (romans, mood tags)
    "axis_minor": (["i", "bVI", "bIII", "bVII"], {"epic", "uplifting", "synthwave", "hardstyle", "trance"}),
    "axis_major": (["I", "V", "vi", "IV"], {"uplifting", "pop"}),
    "sad_axis": (["vi", "IV", "I", "V"], {"epic", "emotional"}),
    "andalusian": (["i", "bVII", "bVI", "V"], {"dark", "dramatic", "trance"}),
    "dark_descent": (["i", "bVI", "bVII", "i"], {"dark", "synthwave", "dnb"}),
    "minor_plagal": (["i", "iv", "bVI", "V"], {"dark", "dramatic"}),
    "liquid_dnb": (["im9", "bVImaj7", "bIIImaj7", "bVII"], {"dreamy", "dnb", "liquid"}),
    "liquid_dnb_2": (["ivm9", "bVII9", "bIIImaj7", "bVImaj7"], {"dreamy", "dnb", "liquid"}),
    "royal_road": (["IVmaj7", "V7", "iii7", "vi"], {"dreamy", "pop", "chill"}),
    "dreamy_maj7": (["Imaj7", "vi9", "IVmaj7", "V7sus4"], {"dreamy", "chill", "ambient"}),
    "lydian_float": (["Imaj7", "IIadd9", "Imaj7", "viim7"], {"dreamy", "ambient", "space"}),
    "circle": (["vi7", "ii7", "V7", "Imaj7"], {"jazzy", "chill"}),
    "dorian_vamp": (["i7", "IV9", "i7", "IV9"], {"funky", "synthwave", "groove"}),
    "outrun": (["i", "bVII", "bVI", "bVII"], {"synthwave", "drive"}),
    "heroic": (["bVI", "bVII", "i", "i"], {"epic", "hardstyle", "trance"}),
    "euphoric": (["bVI", "bIII", "bVII", "iv"], {"uplifting", "hardstyle", "trance"}),
    "mixo_drive": (["I", "bVII", "IV", "I"], {"rock", "drive", "playful"}),
    "chip_happy": (["I", "vi", "IV", "V"], {"playful", "chip", "pop"}),
    "space_suspended": (["isus2", "bVIImaj7", "bVIadd9", "ivm9"], {"space", "ambient", "dreamy"}),
}


def progression(key: Key, name_or_romans) -> list[Chord]:
    romans = PROGRESSIONS[name_or_romans][0] if isinstance(name_or_romans, str) else name_or_romans
    return [key.roman(r) for r in romans]


# functional harmony Markov chain (for generative progressions)
_FUNC_NEXT = {
    "T": [("PD", 0.45), ("D", 0.25), ("T", 0.30)],
    "PD": [("D", 0.6), ("T", 0.2), ("PD", 0.2)],
    "D": [("T", 0.75), ("PD", 0.1), ("D", 0.15)],
}
_FUNC_MINOR = {"T": ["i", "bIII", "bVI"], "PD": ["iv", "iiø", "bVI", "iv7"], "D": ["V", "bVII", "V7", "vii°"]}
_FUNC_MAJOR = {"T": ["I", "vi", "iii"], "PD": ["IV", "ii", "ii7", "IVmaj7"], "D": ["V", "V7", "vii°", "V7sus4"]}


def generate_progression(key: Key, length=4, rng: random.Random | None = None) -> list[Chord]:
    rng = rng or random.Random()
    minorish = key.intervals[2] == 3
    table = _FUNC_MINOR if minorish else _FUNC_MAJOR
    func = "T"
    out = [table["T"][0]]
    for _ in range(length - 1):
        opts, w = zip(*_FUNC_NEXT[func])
        func = rng.choices(opts, w)[0]
        cand = [c for c in table[func] if c != out[-1]]
        out.append(rng.choice(cand))
    # make the loop turn around: last chord dominant-ish if possible
    return [key.roman(r) for r in out]


# --------------------------------------------------------------------------
# Voice leading
# --------------------------------------------------------------------------
def voice_chord(chord: Chord, prev: list[int] | None, lo=52, hi=76, n_voices=4) -> list[int]:
    """Pick a voicing of `chord` within [lo,hi] minimising movement from prev."""
    ivs = chord.intervals
    pcs = [(chord.root + i) % 12 for i in ivs]
    # choose which tones to keep for n_voices (drop 5th first in big chords)
    if len(pcs) > n_voices:
        prio = sorted(range(len(pcs)), key=lambda k: (ivs[k] % 12 == 7, ivs[k] == 0))
        keep = sorted(prio[:n_voices])
        pcs = [pcs[k] for k in keep]
    while len(pcs) < n_voices:
        pcs = pcs + [pcs[len(pcs) % len(pcs)]]
    # candidate notes for each pc
    cands = []
    for pc in pcs:
        cands.append([m for m in range(lo, hi + 1) if m % 12 == pc])
    best, best_cost = None, 1e9
    target = prev or [int((lo + hi) / 2) + d for d in (-7, -3, 2, 6, 9)][:n_voices]
    centre = sum(target) / len(target)

    def rec(i, chosen):
        nonlocal best, best_cost
        if i == len(cands):
            v = sorted(chosen)
            if len(set(v)) < len(v):
                return
            span = v[-1] - v[0]
            if span > 19:
                return
            t = sorted(target)
            cost = sum(abs(a - b) for a, b in zip(v, t)) + 0.3 * abs(sum(v) / len(v) - centre)
            # avoid clusters at the bottom (low interval limit)
            if v[1] - v[0] < 3 and v[0] < 55:
                cost += 6
            if cost < best_cost:
                best, best_cost = v, cost
            return
        for m in cands[i]:
            rec(i + 1, chosen + [m])

    rec(0, [])
    return best or sorted(m for m in [lo + (pc - lo) % 12 for pc in pcs])


# --------------------------------------------------------------------------
# Note events
# --------------------------------------------------------------------------
@dataclass
class Note:
    start: float       # beats
    dur: float         # beats
    pitch: int         # midi
    vel: float = 0.8
    slide: bool = False   # glide into this note (mono synths)
    accent: bool = False
    bend: float = 0.0     # semitones of pitch dive/rise over the note's tail
    scoop: float = 0.0    # semitones the note scoops up from at its start
    throw: bool = False   # send to the long delay ("delay throw")

    def end(self):
        return self.start + self.dur


@dataclass
class ChordEvent:
    start: float
    dur: float
    chord: Chord
    voicing: list = field(default_factory=list)


def euclid(k: int, n: int, rot: int = 0) -> list[int]:
    """Bjorklund / euclidean rhythm, list of 0/1 length n."""
    if k <= 0:
        return [0] * n
    pattern = [1 if ((i * k) % n) < k else 0 for i in range(n)]
    return pattern[-rot:] + pattern[:-rot] if rot else pattern


# --------------------------------------------------------------------------
# Melody generation
# --------------------------------------------------------------------------
MELODY_CHARACTER = {
    #           density(16ths) leap_prob  rest_prob  long_note  legato  range
    "smooth":   dict(density=0.30, leap=0.12, rest=0.15, long=0.55, legato=0.9, span=12, run=0.0, octave=0.0),
    "flowing":  dict(density=0.45, leap=0.18, rest=0.12, long=0.35, legato=0.8, span=14, run=0.1, octave=0.05),
    "playful":  dict(density=0.60, leap=0.35, rest=0.25, long=0.10, legato=0.35, span=15, run=0.15, octave=0.1),
    "wild":     dict(density=0.80, leap=0.30, rest=0.08, long=0.08, legato=0.5, span=19, run=0.35, octave=0.2),
    "epic":     dict(density=0.35, leap=0.25, rest=0.10, long=0.50, legato=0.85, span=14, run=0.05, octave=0.1),
}

_RHYTHM_CELLS = {
    # cells of 16th-steps within one beat (4 steps): onsets
    "smooth": [[0], [0], [0, 2], [0], [0, 3]],
    "flowing": [[0, 2], [0], [0, 3], [0, 2, 3], [2]],
    "playful": [[0, 2], [0, 1, 2], [0, 3], [1, 3], [0, 2, 3], [0, 1, 3]],
    "wild": [[0, 1, 2, 3], [0, 2, 3], [0, 1, 2, 3], [0, 1, 3], [0, 2]],
    "epic": [[0], [0], [0, 2], [0, 3], [0]],
}


def _motif_rhythm(char: str, beats: int, rng: random.Random):
    """Return onsets (in 16th steps) for a motif of `beats` beats."""
    cells = _RHYTHM_CELLS[char]
    onsets = []
    p = MELODY_CHARACTER[char]
    for b in range(beats):
        if rng.random() < p["long"] * 0.6 and b > 0:
            continue  # held note across this beat
        cell = rng.choice(cells)
        onsets.extend(4 * b + c for c in cell)
    if not onsets or onsets[0] != 0:
        onsets = [0] + onsets
    return sorted(set(onsets))


def _contour(n, rng, char):
    """Melodic contour shape (relative scale-step directions)."""
    shape = rng.choice(["arch", "rise", "fall", "wave", "valley"])
    steps = []
    for i in range(n):
        x = i / max(1, n - 1)
        if shape == "arch":
            d = 1 if x < 0.5 else -1
        elif shape == "rise":
            d = 1 if rng.random() < 0.7 else -1
        elif shape == "fall":
            d = -1 if rng.random() < 0.7 else 1
        elif shape == "valley":
            d = -1 if x < 0.5 else 1
        else:
            d = 1 if math.sin(x * math.pi * 3) > 0 else -1
        steps.append(d)
    return steps


def make_melody(key: Key, chords: list[ChordEvent], bars: int, char="flowing",
                lo=67, hi=86, rng: random.Random | None = None, beats_per_bar=4,
                phrase_form="AABA") -> list[Note]:
    """Motif-based melody that follows harmony.
    * strong beats land on chord tones, weak positions use scale steps
    * leaps are followed by a step back (gap-fill)
    * motifs repeat with variation in a phrase form (AABA / ABAC / AAAB)
    * phrase ends resolve to chord root/third
    """
    rng = rng or random.Random()
    P = MELODY_CHARACTER[char]
    scale = key.scale_notes(lo - 12, hi + 12)
    motif_beats = beats_per_bar * (2 if bars >= 8 else 1)
    n_motifs = max(1, bars * beats_per_bar // motif_beats)
    letters = (phrase_form * 8)[:n_motifs]
    motifs = {}
    notes: list[Note] = []
    prev = rng.choice([m for m in scale if lo + 3 <= m <= lo + 10])

    def chord_at(beat):
        for ce in chords:
            if ce.start <= beat < ce.start + ce.dur:
                return ce.chord
        return chords[-1].chord

    def nearest(cands, target):
        return min(cands, key=lambda m: (abs(m - target), rng.random()))

    for mi, letter in enumerate(letters):
        base = mi * motif_beats
        if letter not in motifs or rng.random() < 0.15:
            on = _motif_rhythm(char, motif_beats, rng)
            motifs[letter] = (on, _contour(len(on), rng, char))
        on, cont = motifs[letter]
        # variation: A' -> shift contour on last quarter
        if letters[:mi].count(letter) >= 1 and rng.random() < 0.5:
            cont = cont[: len(cont) * 3 // 4] + [-c for c in cont[len(cont) * 3 // 4:]]
        last_motif = mi == len(letters) - 1
        for k, step in enumerate(on):
            beat = base + step / 4.0
            nxt = on[k + 1] if k + 1 < len(on) else motif_beats * 4
            length = (nxt - step) / 4.0
            if rng.random() < P["rest"] and k > 0 and length >= 0.5:
                length *= 0.5  # leave a breath
            ch = chord_at(beat)
            ctones = [m for m in range(lo - 5, hi + 1) if m % 12 in ch.pcs]
            strong = step % 4 == 0
            direction = cont[k]
            if strong or rng.random() < 0.25:
                # chord tone near the contour direction
                tgt = prev + direction * rng.choice([2, 3, 4])
                if rng.random() < P["leap"]:
                    tgt = prev + direction * rng.choice([5, 7, 8])
                if rng.random() < P["octave"]:
                    tgt = prev + direction * 12
                pitch = nearest(ctones, tgt)
            else:
                idx = min(range(len(scale)), key=lambda i: abs(scale[i] - prev))
                jump = 1 if rng.random() > P["leap"] else rng.choice([2, 3])
                idx = max(0, min(len(scale) - 1, idx + direction * jump))
                pitch = scale[idx]
            # gap-fill: after a big leap step back
            if notes and abs(notes[-1].pitch - prev) > 4:
                back = -1 if prev > notes[-1].pitch else 1
                idx = min(range(len(scale)), key=lambda i: abs(scale[i] - prev))
                pitch = scale[max(0, min(len(scale) - 1, idx + back))]
            # keep in range
            while pitch > hi:
                pitch -= 12
            while pitch < lo:
                pitch += 12
            # fast runs ornament
            if rng.random() < P["run"] and length >= 0.5:
                run_dir = rng.choice([-1, 1])
                idx = min(range(len(scale)), key=lambda i: abs(scale[i] - pitch))
                for r in range(int(length * 4)):
                    rp = scale[max(0, min(len(scale) - 1, idx + run_dir * r))]
                    rp = min(max(rp, lo), hi)
                    notes.append(Note(beat + r * 0.25, 0.25, rp, vel=0.7 + 0.2 * (r == 0), slide=False))
                prev = notes[-1].pitch
                continue
            # phrase ending: resolve
            if last_motif and k == len(on) - 1:
                pitch = nearest([m for m in ctones if m % 12 in (ch.root, (ch.root + ch.intervals[1]) % 12)] or ctones, prev)
                length = max(length, 1.0)
            dur = length * (P["legato"] if length < 1 else 0.95)
            slide = bool(notes) and rng.random() < P["legato"] * 0.5 and abs(pitch - prev) <= 7
            vel = 0.75 + (0.15 if strong else 0) + rng.uniform(-0.05, 0.05)
            notes.append(Note(beat, max(0.12, dur), pitch, vel=vel, slide=slide, accent=strong))
            prev = pitch
    return notes


# --------------------------------------------------------------------------
# Bass & arps
# --------------------------------------------------------------------------
def make_bass(chords: list[ChordEvent], pattern: str, octave_root=33, rng=None, key: Key | None = None) -> list[Note]:
    """pattern: 'root' | 'octave8' (synthwave) | 'offbeat' (trance) | 'reese' (long,
    slides) | 'acid' (303 16ths) | 'rolling' (dnb 1/8 syncopated) | 'halftime'."""
    rng = rng or random.Random()
    out = []
    for ce in chords:
        bass_pc = ce.chord.bass if ce.chord.bass is not None else ce.chord.root
        r = octave_root + (bass_pc - octave_root) % 12
        fifth = r + 7
        b0, L = ce.start, ce.dur
        if pattern == "root":
            out.append(Note(b0, L * 0.95, r, 0.9))
        elif pattern == "octave8":
            for i in range(int(L * 2)):
                out.append(Note(b0 + i * 0.5, 0.42, r + (12 if i % 2 else 0), 0.85 + 0.1 * (i % 2 == 0)))
        elif pattern == "offbeat":
            for i in range(int(L)):
                out.append(Note(b0 + i + 0.5, 0.4, r, 0.9))
        elif pattern == "rolling16":
            for i in range(int(L * 4)):
                if i % 4 == 0:
                    continue
                out.append(Note(b0 + i * 0.25, 0.2, r + (12 if i % 4 == 3 and rng.random() < 0.3 else 0), 0.8))
        elif pattern == "reese":
            out.append(Note(b0, L * 0.6, r, 0.9))
            nxt = r + rng.choice([0, 7, 12, ce.chord.intervals[1] if len(ce.chord.intervals) > 2 else 7])
            out.append(Note(b0 + L * 0.625, L * 0.35, nxt, 0.85, slide=True))
        elif pattern == "rolling":
            # dnb: 1 . . x . . x . | syncopated
            steps = [0, 3, 6, 10, 11, 14] if rng.random() < 0.5 else [0, 3, 8, 10, 14]
            for bar in range(int(L // 4) or 1):
                for s in steps:
                    p = r if s in (0, 8) else rng.choice([r, r, fifth, r + 12])
                    out.append(Note(b0 + bar * 4 + s / 4, 0.3 if s not in (0, 8) else 0.7, p, 0.85, slide=rng.random() < 0.2))
        elif pattern == "reverse":
            # hardstyle reverse bass: swells between the kicks
            for i in range(int(L)):
                out.append(Note(b0 + i + 0.22, 0.74, r + 12, 0.95))
        elif pattern == "acid":
            scale = key.scale_notes(r - 2, r + 15) if key else [r, r + 3, r + 7, r + 10, r + 12]
            for i in range(int(L * 4)):
                if rng.random() < 0.18:
                    continue
                p = r if i % 4 == 0 else rng.choice(scale)
                if rng.random() < 0.15:
                    p += 12
                out.append(Note(b0 + i * 0.25, 0.22 if rng.random() < 0.6 else 0.3, p, 0.8,
                                slide=rng.random() < 0.22, accent=rng.random() < 0.3))
        elif pattern == "trap808":
            # 808: long root, pickups, slides up to the 5th/octave and back
            for bar in range(max(1, int(L // 4))):
                o = b0 + bar * 4
                v = rng.random()
                if v < 0.34:
                    seq = [(0, 1.5, r, False), (1.5, 0.5, r, False), (2.5, 1.0, r + 12, True), (3.5, 0.5, r + 7, True)]
                elif v < 0.67:
                    seq = [(0, 0.75, r, False), (0.75, 0.75, r, False), (1.5, 1.5, r + 7, True), (3.0, 1.0, r, True)]
                else:
                    seq = [(0, 2.5, r, False), (2.5, 0.5, r + 3 if key and (r + 3) % 12 in key.pcs else r + 4, True), (3.0, 1.0, r + 12, True)]
                for st, d, p, sl in seq:
                    out.append(Note(o + st, d, p, 0.9, slide=sl))
        elif pattern == "house":
            # bass house: offbeat stabs + 16th pickups, octave jumps
            for i in range(int(L)):
                out.append(Note(b0 + i + 0.5, 0.3, r + (12 if i % 2 else 0), 0.95))
                if rng.random() < 0.45:
                    out.append(Note(b0 + i + 0.75, 0.2, r + rng.choice([0, 12, 7]), 0.8))
        elif pattern == "drop_long":
            # dubstep / growl: long notes with call-and-response every 2 beats
            for i in range(0, int(L), 2):
                out.append(Note(b0 + i, 1.4, r if i % 4 == 0 else r + rng.choice([0, 12, 7, 3]), 0.95))
                out.append(Note(b0 + i + 1.5, 0.45, r + rng.choice([12, 7, 0]), 0.85, slide=rng.random() < 0.5))
        elif pattern == "halftime":
            out.append(Note(b0, 1.5, r, 0.9))
            out.append(Note(b0 + 2.5, 1.0, r, 0.8))
            if L > 4:
                out.append(Note(b0 + 4, L - 4.2, r + (rng.choice([0, 12, 7])), 0.85, slide=True))
        else:
            raise ValueError(pattern)
    return out


def make_arp(chords: list[ChordEvent], mode="up", rate=0.25, octaves=2, gate=0.6, lo=60) -> list[Note]:
    out = []
    for ce in chords:
        base = sorted(ce.voicing) if ce.voicing else ce.chord.tones(lo + (ce.chord.root - lo) % 12)
        seq = []
        for o in range(octaves):
            seq += [n + 12 * o for n in base]
        if mode == "down":
            seq = seq[::-1]
        elif mode == "updown":
            seq = seq + seq[-2:0:-1]
        elif mode == "pinky":       # alternating with top note
            top = seq[-1]
            seq = [x for n in seq[:-1] for x in (n, top)]
        elif mode == "converge":
            s2 = []
            i, j = 0, len(seq) - 1
            while i <= j:
                s2.append(seq[i])
                if i != j:
                    s2.append(seq[j])
                i += 1
                j -= 1
            seq = s2
        n_steps = int(round(ce.dur / rate))
        for i in range(n_steps):
            out.append(Note(ce.start + i * rate, rate * gate, seq[i % len(seq)], 0.6 + 0.25 * (i % 4 == 0)))
    return out


def chord_track(chords: list[Chord], bars_per_chord: float, total_bars: int, beats_per_bar=4,
                lo=52, hi=76, n_voices=4) -> list[ChordEvent]:
    evs = []
    prev = None
    t = 0.0
    i = 0
    while t < total_bars * beats_per_bar - 1e-6:
        ch = chords[i % len(chords)]
        v = voice_chord(ch, prev, lo, hi, n_voices)
        evs.append(ChordEvent(t, bars_per_chord * beats_per_bar, ch, v))
        prev = v
        t += bars_per_chord * beats_per_bar
        i += 1
    return evs
