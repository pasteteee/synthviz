"""Harmony-aware melody writing (v2).

* chord-scales: the key's scale bent to fit the current chord (borrowed chords
  like bVI / iv in a major mode get their own chromatic notes, so passing tones
  never rub against the chord)
* avoid-notes: a half step above a chord tone is never held
* motif development: sequence, inversion, rhythmic diminution, octave lift,
  call & response, diatonic thirds harmonisation
* expressive marks: scoops, end-of-phrase dives, delay throws, grace runs
* fix_harmony(): last safety pass over any note list
"""
from __future__ import annotations

import random
from dataclasses import replace

from .theory import Key, Note, ChordEvent, Chord, MELODY_CHARACTER, _RHYTHM_CELLS


# ------------------------------------------------------------------ chord scales
def chord_scale(key: Key, chord: Chord) -> list[int]:
    pcs = set(key.pcs)
    ctones = set(chord.pcs)
    for c in ctones:
        if c not in pcs:
            # chromatic chord tone: replace scale neighbours a half step away
            pcs.discard((c + 1) % 12)
            pcs.discard((c - 1) % 12)
            pcs.add(c)
    # remove scale notes a half step from a chord tone only if they'd create b9 vs root
    pcs.discard((chord.root + 1) % 12)
    for c in ctones:
        pcs.add(c)
    return sorted(pcs)


def avoid_pcs(chord: Chord) -> set:
    """Pitch classes a half step above a chord tone (minor 9th clash) that are
    not themselves chord tones."""
    ct = set(chord.pcs)
    return {(c + 1) % 12 for c in ct} - ct


def stable_pcs(chord: Chord) -> set:
    """Chord tones + safe tensions (9th on non-dominant chords)."""
    s = set(chord.pcs)
    ivs = set(i % 12 for i in chord.intervals)
    dominant = 4 in ivs and 10 in ivs
    if (3 in ivs or 4 in ivs) and not dominant:
        if (chord.root + 2) % 12 not in avoid_pcs(chord):
            s.add((chord.root + 2) % 12)
    return s


def chord_at(chords: list[ChordEvent], beat: float) -> ChordEvent:
    for ce in chords:
        if ce.start <= beat + 1e-6 < ce.start + ce.dur:
            return ce
    return chords[-1] if beat >= chords[-1].start else chords[0]


def _notes_in(pcs, lo, hi):
    return [m for m in range(lo, hi + 1) if m % 12 in pcs]


# ------------------------------------------------------------------ melody v2
def make_melody(key: Key, chords: list[ChordEvent], bars: int, char="flowing", lo=67, hi=86,
                rng: random.Random | None = None, phrase_form="AABA", beats_per_bar=4) -> list[Note]:
    rng = rng or random.Random()
    P = MELODY_CHARACTER[char]
    motif_beats = beats_per_bar * (2 if bars >= 8 else 1)
    n_motifs = max(1, bars * beats_per_bar // motif_beats)
    letters = (phrase_form * 8)[:n_motifs]
    motifs = {}
    out: list[Note] = []
    prev = rng.choice(_notes_in(set(chord_at(chords, 0).chord.pcs), lo + 2, lo + 11) or [lo + 5])

    def pick(cands, target):
        return min(cands, key=lambda m: (abs(m - target), rng.random()))

    for mi, letter in enumerate(letters):
        base = mi * motif_beats
        seen_before = letter in motifs
        if not seen_before:
            onsets = []
            cells = _RHYTHM_CELLS[char]
            for b in range(motif_beats):
                if b > 0 and rng.random() < P["long"] * 0.6:
                    continue
                onsets += [4 * b + c for c in rng.choice(cells)]
            onsets = sorted(set([0] + onsets))
            # breath at the end of a motif
            if len(onsets) > 3 and rng.random() < 0.6:
                onsets = [o for o in onsets if o < motif_beats * 4 - 3]
            shape = [rng.choice([-2, -1, -1, 1, 1, 2]) for _ in onsets]
            motifs[letter] = (onsets, shape)
        onsets, shape = motifs[letter]
        # development of repeated motifs
        dev = None
        if seen_before:
            dev = rng.choice(["sequence", "invert", "diminish", "same", "octave", "sequence"])
        if dev == "invert":
            shape = [-x for x in shape]
        if dev == "diminish":
            extra = [o + 2 for o in onsets if o + 2 not in onsets and (o + 2) % 4 == 2 and rng.random() < 0.6]
            onsets = sorted(set(onsets + extra))
            shape = shape + [rng.choice([-1, 1]) for _ in extra]
        seq_shift = rng.choice([2, 3, -2]) if dev == "sequence" else 0
        octave = 12 if (dev == "octave" and prev + 12 <= hi + 5) else 0
        last_motif = mi == len(letters) - 1
        cur = prev + seq_shift
        for k_, step in enumerate(onsets):
            beat = base + step / 4.0
            nxt = onsets[k_ + 1] if k_ + 1 < len(onsets) else motif_beats * 4
            length = (nxt - step) / 4.0
            ce = chord_at(chords, beat)
            ch = ce.chord
            cs = chord_scale(key, ch)
            # does the note outlast the chord?
            ends_in_next = beat + length > ce.start + ce.dur + 0.24
            strong = step % 4 == 0
            long_note = length >= 0.75
            d = shape[k_ % len(shape)]
            if strong or long_note or ends_in_next:
                pool = stable_pcs(ch) if not ends_in_next else set(ch.pcs)
                if ends_in_next:
                    nch = chord_at(chords, ce.start + ce.dur).chord
                    common = set(ch.pcs) & set(nch.pcs)
                    if common:
                        pool = common
                    else:
                        length = ce.start + ce.dur - beat  # cut at the chord change
                tgt = cur + d * rng.choice([2, 3, 4])
                if rng.random() < P["leap"]:
                    tgt = cur + d * rng.choice([5, 7])
                pitch = pick(_notes_in(pool, lo - 2, hi), tgt)
            else:
                scale = _notes_in(set(cs) - avoid_pcs(ch) if length >= 0.5 else set(cs), lo - 2, hi)
                idx = min(range(len(scale)), key=lambda i: abs(scale[i] - cur))
                jump = 1 if rng.random() > P["leap"] else 2
                pitch = scale[max(0, min(len(scale) - 1, idx + (1 if d > 0 else -1) * jump))]
            # gap fill: after a leap, step back
            if out and abs(out[-1].pitch - cur) > 4 and not strong:
                back = -1 if cur > out[-1].pitch else 1
                scale = _notes_in(set(cs), lo - 2, hi)
                idx = min(range(len(scale)), key=lambda i: abs(scale[i] - cur))
                pitch = scale[max(0, min(len(scale) - 1, idx + back))]
            while pitch > hi:
                pitch -= 12
            while pitch < lo - 2:
                pitch += 12
            # resolution at the end of the phrase
            if last_motif and k_ == len(onsets) - 1:
                pitch = pick(_notes_in({ch.root, (ch.root + ch.intervals[1]) % 12}, lo, hi), cur)
                length = max(length, 1.0)
            dur = length * (P["legato"] if length < 1 else 0.95)
            slide = bool(out) and rng.random() < P["legato"] * 0.45 and abs(pitch - cur) <= 7
            n = Note(beat, max(0.12, dur), pitch + octave, vel=0.75 + 0.15 * strong + rng.uniform(-0.05, 0.05),
                     slide=slide, accent=strong)
            # expressive marks
            if long_note and rng.random() < 0.35:
                n.scoop = -rng.choice([1.0, 2.0])
            if k_ == len(onsets) - 1 and length >= 1.0:
                if rng.random() < 0.35 and not last_motif:
                    n.bend = -rng.choice([2.0, 5.0, 12.0])
                n.throw = True
            # grace run into a strong note
            if strong and k_ > 0 and rng.random() < P["run"] and out:
                sc = _notes_in(set(cs), lo - 2, hi + 2)
                gi = min(range(len(sc)), key=lambda i: abs(sc[i] - n.pitch))
                for g in (2, 1):
                    gp = sc[max(0, min(len(sc) - 1, gi - g * (1 if n.pitch > out[-1].pitch else -1)))]
                    out.append(Note(beat - g * 0.125, 0.11, gp, 0.6))
            out.append(n)
            cur = pitch
        prev = cur
    out.sort(key=lambda n: n.start)
    return [n for n in out if n.start >= 0]


# ------------------------------------------------------------------ development tools
def harmonize_thirds(notes: list[Note], key: Key, chords: list[ChordEvent], below=True) -> list[Note]:
    """Diatonic (chord-scale) third below/above each note; resolves clashes to chord tones."""
    out = []
    for n in notes:
        ch = chord_at(chords, n.start).chord
        cs = _notes_in(set(chord_scale(key, ch)), n.pitch - 14, n.pitch + 14)
        if n.pitch not in cs:
            continue
        i = cs.index(n.pitch)
        j = i - 2 if below else i + 2
        if not 0 <= j < len(cs):
            continue
        p = cs[j]
        if n.dur >= 0.5 and p % 12 in avoid_pcs(ch):
            ct = _notes_in(set(ch.pcs), p - 4, p + 4)
            p = min(ct, key=lambda m: abs(m - p)) if ct else p
        out.append(replace(n, pitch=p, vel=n.vel * 0.75, throw=False))
    return out


def response_fills(lead: list[Note], key: Key, chords: list[ChordEvent], total_beats: float, rng,
                   lo=72, hi=91, min_gap=1.5) -> list[Note]:
    """Counter-melody 'перебивки': short answers played in the gaps of the lead."""
    out = []
    ln = sorted(lead, key=lambda n: n.start)
    gaps = []
    prev_end = 0.0
    for n in ln:
        if n.start - prev_end >= min_gap:
            gaps.append((prev_end, n.start))
        prev_end = max(prev_end, n.end())
    for a, b in gaps:
        start = a + 0.25 if (a * 4) % 1 == 0 else (int(a * 2) + 1) / 2
        ce = chord_at(chords, start)
        ct = _notes_in(set(ce.chord.pcs), lo, hi)
        if not ct:
            continue
        pat = rng.choice([[0, 0.5, 1.0], [0, 0.25, 0.5, 0.75], [0, 0.75], [0, 0.5, 0.75, 1.0]])
        p = rng.choice(ct)
        up = rng.random() < 0.5
        for k, o in enumerate(pat):
            t = start + o
            if t + 0.2 > b - 0.25:
                break
            cands = _notes_in(set(chord_at(chords, t).chord.pcs), lo, hi)
            if not cands:
                break
            idx = min(range(len(cands)), key=lambda i: abs(cands[i] - p))
            idx = max(0, min(len(cands) - 1, idx + (1 if up else -1) * (k > 0)))
            p = cands[idx]
            out.append(Note(t, 0.2, p, 0.65 - 0.05 * k))
    return out


def fix_harmony(notes: list[Note], key: Key, chords: list[ChordEvent], strict_len=0.5) -> list[Note]:
    """Safety pass: snap out-of-chord-scale notes, never hold avoid-notes, and
    cut / re-target notes that ring into a chord they clash with."""
    out = []
    for n in notes:
        ce = chord_at(chords, n.start)
        ch = ce.chord
        cs = set(chord_scale(key, ch))
        p = n.pitch
        if p % 12 not in cs:
            p = min((m for m in range(p - 2, p + 3) if m % 12 in cs), key=lambda m: abs(m - p), default=p)
        if n.dur >= 0.75 and p % 12 not in stable_pcs(ch):
            st = stable_pcs(ch)
            p = min((m for m in range(p - 3, p + 4) if m % 12 in st), key=lambda m: (abs(m - p), m % 12 not in ch.pcs), default=p)
        if n.dur >= 0.5 and p % 12 not in stable_pcs(ch) and any((p - c) % 12 in (1, 11) for c in ch.pcs):
            st = stable_pcs(ch)
            p = min((m for m in range(p - 3, p + 4) if m % 12 in st), key=lambda m: abs(m - p), default=p)
        if n.dur >= strict_len and p % 12 in avoid_pcs(ch):
            p = min((m for m in range(p - 2, p + 3) if m % 12 in ch.pcs), key=lambda m: abs(m - p), default=p)
        dur = n.dur
        boundary = ce.start + ce.dur
        if n.start + dur > boundary + 0.2 and boundary < chords[-1].start + chords[-1].dur:
            nch = chord_at(chords, boundary).chord
            if p % 12 in avoid_pcs(nch) or p % 12 not in set(chord_scale(key, nch)):
                dur = max(0.12, boundary - n.start - 0.02)
        out.append(replace(n, pitch=p, dur=dur))
    return out


def glide_voices(chords: list[ChordEvent], sections_ok, extra_octave=0) -> list[list[Note]]:
    """Each voice of the (voice-led) chord progression as one legato line that
    glides into the next chord's corresponding voice."""
    n_v = max(len(ce.voicing) for ce in chords)
    lines = [[] for _ in range(n_v)]
    prev_ok = False
    for ce in chords:
        ok = sections_ok(ce.start)
        for v in range(n_v):
            p = sorted(ce.voicing)[min(v, len(ce.voicing) - 1)] + extra_octave
            lines[v].append(Note(ce.start, ce.dur, p, 0.7, slide=prev_ok and ok))
        prev_ok = ok
    return [[n for n in line if sections_ok(n.start)] for line in lines]


# ------------------------------------------------------------------ hooks (v5)
# One-bar rhythm cells in 16th steps, by genre family. Hooks reuse ONE cell so the ear can learn it.
HOOK_RHYTHMS = {
    "hs": [[0, 6, 12], [0, 6, 12, 14], [0, 4, 6, 8, 12], [0, 3, 6, 10, 12], [0, 8, 10, 12, 14], [0, 4, 8, 10, 12]],
    "fb": [[0, 3, 6, 8, 11, 14], [0, 2, 3, 6, 8, 10, 11, 14], [0, 2, 4, 6, 7, 10, 12, 14], [0, 3, 6, 10, 12, 13]],
    "dnb": [[0, 3, 6], [0, 2, 6, 8], [0, 3, 6, 8], [0, 2, 4, 10]],
    "bass": [[0, 3, 6, 10, 12], [2, 6, 10, 14], [0, 3, 6, 8, 11, 14], [0, 2, 3, 6, 10]],
    "sw": [[0, 4, 6, 10, 12], [0, 8, 10, 12], [0, 3, 6, 8, 12], [0, 6, 8, 12, 14]],
}


def _fit(pitch, ch, key, strong, lo, hi, rng):
    """Keep the hook's pitch if it still fits the new chord, else move to the nearest safe tone."""
    pool = stable_pcs(ch) if strong else set(chord_scale(key, ch)) - avoid_pcs(ch)
    if pitch % 12 in pool:
        return pitch
    cands = _notes_in(pool, lo - 2, hi)
    return min(cands, key=lambda m: (abs(m - pitch), m > pitch, rng.random())) if cands else pitch


def _motif(key, chords, base, rhythm, lo, hi, rng, start_pitch, span=9):
    """Build one motif (pitches + lengths) over `rhythm` (16th onsets across the motif)."""
    notes = []
    cur = start_pitch
    total = (rhythm[-1] // 16 + 1) * 16
    leap_at = rng.randrange(len(rhythm)) if len(rhythm) > 2 else -1
    direction = rng.choice([-1, 1])
    for i, st in enumerate(rhythm):
        beat = base + st / 4.0
        nxt = rhythm[i + 1] if i + 1 < len(rhythm) else total
        length = (nxt - st) / 4.0
        ch = chord_at(chords, beat).chord
        strong = st % 4 == 0 or length >= 0.75
        if i == 0:
            p = cur
        else:
            step = rng.choice([1, 1, 2, 2, 3]) * direction
            if i == leap_at:
                step = rng.choice([4, 5, 7]) * direction
            sc = _notes_in(set(chord_scale(key, ch)), lo - 2, hi + 2)
            idx = min(range(len(sc)), key=lambda j: abs(sc[j] - cur))
            if abs(step) < 4:
                p = sc[max(0, min(len(sc) - 1, idx + step))]
            else:
                ct = _notes_in(set(ch.pcs), lo - 2, hi + 2)
                p = min(ct, key=lambda m: abs(m - (cur + step))) if ct else cur
            if abs(p - start_pitch) > span or p > hi or p < lo:
                direction = -direction
                p = sc[max(0, min(len(sc) - 1, idx + direction))]
            if rng.random() < 0.3:
                direction = -direction
        p = _fit(p, ch, key, strong, lo, hi, rng)
        notes.append([st, length, p])
        cur = p
    return notes


def make_hook(key: Key, chords: list[ChordEvent], bars: int, family="sw", lo=70, hi=88,
              rng: random.Random | None = None, form=None) -> list[Note]:
    """Catchy hook: a 1-2 bar motif repeated 2-4 times with only the ending changed (A A' / A B A B').
    Rhythm comes from genre cells, pitches stay put across chords when they fit (EDM pedal-melody feel)."""
    rng = rng or random.Random()
    cells = HOOK_RHYTHMS.get(family, HOOK_RHYTHMS["sw"])
    motif_bars = 2 if bars >= 8 else 1
    n_motifs = max(1, bars // motif_bars)
    form = form or rng.choice(["ABAB'", "AAAB", "AA'AB", "ABAC"])
    letters = [form.replace("'", "")[i % len(form.replace("'", ""))] for i in range(n_motifs)]
    # a 2-bar motif = cell + answering cell (often the same one shortened -> call/response)
    def rhythm():
        c1 = rng.choice(cells)
        if motif_bars == 1:
            return c1
        c2 = rng.choice([c1, c1, rng.choice(cells), [0, 4, 8]])
        c2 = [s for s in c2 if s < 12] or [0]    # breath at the end of the motif
        return c1 + [16 + s for s in c2]
    first = chord_at(chords, 0).chord
    start = rng.choice(_notes_in(set(first.pcs), lo + 2, lo + 10) or [lo + 5])
    motifs = {}
    out = []
    for mi, L in enumerate(letters):
        base = mi * motif_bars * 4
        last = mi == n_motifs - 1
        if L not in motifs:
            r = rhythm() if L == "A" or "A" not in motifs else (
                motifs["A"][0] if rng.random() < 0.5 else rhythm())
            sp = start if L == "A" else start + rng.choice([-3, 2, 4, 5])
            motifs[L] = (r, _motif(key, chords, base, r, lo, hi, rng, sp))
        r, m = motifs[L]
        for i, (st, length, p) in enumerate(m):
            beat = base + st / 4.0
            ch = chord_at(chords, beat).chord
            strong = st % 4 == 0 or length >= 0.75
            p = _fit(p, ch, key, strong, lo, hi, rng)
            is_tail = i >= len(m) - 2
            # A' / B': same rhythm, new ending: up for tension mid-phrase, resolve to root at the end
            if is_tail and (last or (mi % 2 == 1 and rng.random() < 0.6)):
                if last and i == len(m) - 1:
                    p = min(_notes_in({ch.root}, lo, hi) or [p], key=lambda x: abs(x - p))
                    length = max(length, 1.0)
                elif not last:
                    up = _notes_in(stable_pcs(ch), p + 1, min(hi, p + 7))
                    p = up[0] if up else p
            dur = length * (0.85 if length < 1 else 0.95)
            n = Note(beat, max(0.12, dur), p, vel=0.8 + 0.12 * (st % 4 == 0) + rng.uniform(-0.04, 0.04),
                     slide=bool(out) and length < 0.5 and rng.random() < 0.25 and abs(p - out[-1].pitch) <= 5,
                     accent=st % 4 == 0)
            if length >= 1.0 and rng.random() < 0.3:
                n.scoop = -1.0
            if i == len(m) - 1 and length >= 1.0:
                n.throw = True
            out.append(n)
    return [n for n in out if n.start < bars * 4]
