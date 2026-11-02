"""Per-frame audio features + exact musical events -> what the visuals react to."""
from __future__ import annotations

import bisect
import numpy as np

from ..dsp import SR

N_BANDS = 64


def _band_edges(n=N_BANDS, lo=35.0, hi=14000.0, nfft=4096):
    f = np.geomspace(lo, hi, n + 1)
    bins = np.clip(np.round(f / SR * nfft).astype(int), 1, nfft // 2)
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1
    return bins


class Features:
    def __init__(self, song, fps=30):
        self.song = song
        self.fps = fps
        ev = song.events
        self.ev = ev
        self.dur = song.duration
        self.n_frames = int(self.dur * fps)
        mono = song.audio.mean(axis=1).astype(np.float64)
        self.mono = mono
        self.stereo = song.audio
        st = song.stems
        self.melodic = (st["lead"] + st["bass"] * 0.6 + st["arp"] * 0.5).mean(axis=1).astype(np.float64)
        self.n_frames_cta = 0
        # ---------- spectrum
        nfft = 4096
        win = np.hanning(nfft)
        edges = _band_edges(N_BANDS, nfft=nfft)
        spec = np.zeros((self.n_frames, N_BANDS), np.float32)
        pad = np.concatenate([np.zeros(nfft), mono, np.zeros(nfft)])
        for i in range(self.n_frames):
            c = int(i / fps * SR) + nfft
            seg = pad[c - nfft // 2: c + nfft // 2] * win
            mag = np.abs(np.fft.rfft(seg))
            cs = np.concatenate([[0], np.cumsum(mag)])
            band = (cs[edges[1:]] - cs[edges[:-1]]) / np.maximum(1, edges[1:] - edges[:-1])
            spec[i] = 20 * np.log10(band + 1e-6)
        # tilt compensation (+3dB/oct) so highs show up
        spec += np.linspace(0, 18, N_BANDS)[None, :]
        lo_, hi_ = np.percentile(spec, 20), np.percentile(spec, 99.5)
        spec = np.clip((spec - lo_) / (hi_ - lo_), 0, 1)
        sm = np.zeros_like(spec)
        cur = np.zeros(N_BANDS, np.float32)
        for i in range(self.n_frames):
            a = np.where(spec[i] > cur, 0.65, 0.18)
            cur = cur + (spec[i] - cur) * a
            sm[i] = cur
        self.spec = sm
        # ---------- stem envelopes
        self.env = {}
        hop = SR // fps
        for name, st in song.stems.items():
            m = np.abs(st).mean(axis=1)
            n = self.n_frames
            m = m[: n * hop]
            if len(m) < n * hop:
                m = np.concatenate([m, np.zeros(n * hop - len(m))])
            r = np.sqrt((m.reshape(n, hop) ** 2).mean(axis=1))
            ref = np.percentile(r, 98) + 1e-6
            r = np.clip(r / ref, 0, 1.5)
            out = np.zeros(n, np.float32)
            c = 0.0
            for i in range(n):
                c = c + (r[i] - c) * (0.7 if r[i] > c else 0.25)
                out[i] = c
            self.env[name] = out
        lvl = np.sqrt((mono[: self.n_frames * hop].reshape(-1, hop) ** 2).mean(axis=1))
        self.level = np.clip(lvl / (np.percentile(lvl, 98) + 1e-6), 0, 1.3).astype(np.float32)
        # ---------- event times
        self.kicks = sorted(ev["kicks"])
        self.snares = sorted(ev["snares"])
        self.hats = sorted(ev["hats"])
        self.impacts = sorted(ev["impacts"])
        self.chords = ev["chords"]
        self.chord_t = [c["t"] for c in self.chords]
        self.lead = ev["lead"]
        self.lead_t = [n["t"] for n in self.lead]
        self.bass = ev["bass"]
        self.bass_t = [n["t"] for n in self.bass]
        self.arp = ev["arp"]
        self.arp_t = [n["t"] for n in self.arp]
        self.sections = ev["sections"]
        # ---------- pitch curves for the HUD graph (glides become curves)
        self.CR = 120
        NC = int(self.dur * self.CR) + 2
        def curve(notes, glide):
            arr = np.full(NC, np.nan)
            for n in notes:
                a = int(n["t"] * self.CR); b = max(a + 1, int((n["t"] + n["dur"]) * self.CR))
                seg = np.full(b - a, float(n["p"]))
                if n.get("bend"):
                    L = max(1, int((b - a) * 0.45))
                    seg[-L:] += n["bend"] * np.linspace(0, 1, L) ** 2
                arr[a:b] = seg[: max(0, min(b, NC) - a)]
            if glide > 0:
                k = 1 - np.exp(-1.0 / (glide * self.CR))
                y = np.nan
                for i in range(NC):
                    v = arr[i]
                    if np.isnan(v):
                        y = np.nan
                        continue
                    y = v if np.isnan(y) else y + (v - y) * k
                    arr[i] = y
            return arr
        self.curves = []
        g = ev.get("glide_s", 0.2) if ev.get("glide_chords") else 0.0
        for line in ev.get("pad_lines", []):
            if line:
                self.curves.append(("pad", curve(line, g)))
        self.curves.append(("bass", curve(ev["bass"], 0.03)))
        if ev.get("harm"):
            self.curves.append(("harm", curve(ev["harm"], ev.get("lead_glide", 0.0))))
        if ev.get("counter"):
            self.curves.append(("counter", curve(ev["counter"], 0.0)))
        self.curves.append(("lead", curve(ev["lead"], max(0.02, ev.get("lead_glide", 0.0)))))
        allp = [n["p"] for k_ in ("lead", "counter", "harm") for n in ev.get(k_, [])] + \
               [n["p"] for line in ev.get("pad_lines", []) for n in line]
        self.p_lo = (min(allp) if allp else 48) - 3
        self.p_hi = (max(allp) if allp else 84) + 3
        # ---------- drum lanes: per-bar 16-step patterns + hit times
        spb = ev["spb"]
        self.lanes = {k_: sorted(v) for k_, v in ev.get("lanes", {}).items()}
        self.lane_t = {k_: [x[0] for x in v] for k_, v in self.lanes.items()}
        n_bars = int(self.dur / (4 * spb)) + 2
        self.drum_bars = {k_: [set() for _ in range(n_bars)] for k_ in self.lanes}
        for k_, hits in self.lanes.items():
            for (tt, v) in hits:
                bar = int(tt / (4 * spb) + 1e-6)
                step = int(round((tt - bar * 4 * spb) / spb * 4)) % 16
                if bar < n_bars:
                    self.drum_bars[k_][bar].add(step)
        self.fx = ev.get("fx", [])
        # smoothed energy curve (section energy blended with level)
        en = np.zeros(self.n_frames, np.float32)
        for i in range(self.n_frames):
            en[i] = self.section(i / fps)["energy"]
        k = int(fps * 0.6)
        en = np.convolve(np.pad(en, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
        self.energy = en.astype(np.float32)
        # integrated travel (for tunnels / grids): speed follows energy
        self.travel = np.cumsum(0.4 + 1.6 * self.energy) / fps

    # -------------------------------------------------------------- queries
    @staticmethod
    def _since(times, t):
        i = bisect.bisect_right(times, t) - 1
        return t - times[i] if i >= 0 else 1e9

    def pulse(self, kind, t, tau=0.12):
        times = {"kick": self.kicks, "snare": self.snares, "hat": self.hats, "impact": self.impacts,
                 "perc": self.lane_t.get("perc", [])}[kind]
        return float(np.exp(-self._since(times, t) / tau))

    def section(self, t):
        cur = self.sections[0]
        for s in self.sections:
            if s["t"] <= t + 1e-6:
                cur = s
        return cur

    def chord(self, t):
        i = max(0, bisect.bisect_right(self.chord_t, t) - 1)
        return i, self.chords[i], t - self.chords[i]["t"]

    def note_at(self, notes, times, t):
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            return None, 1e9
        n = notes[i]
        return n, t - n["t"]

    def active_notes(self, notes, times, t, lookback=4.0):
        i = bisect.bisect_right(times, t)
        out = []
        j = i - 1
        while j >= 0 and times[j] > t - lookback:
            n = notes[j]
            if n["t"] + n["dur"] >= t:
                out.append(n)
            j -= 1
        return out

    def beat(self, t):
        return t / self.ev["spb"]

    def wave(self, t, n=2048, channel=None, src=None):
        """Waveform window around t, triggered at rising zero-crossing."""
        x = self.mono if src is None else src
        c = int(t * SR)
        a = max(0, c - n)
        seg = x[a: a + 2 * n]
        if len(seg) < 2 * n:
            seg = np.concatenate([seg, np.zeros(2 * n - len(seg))])
        zc = np.where((seg[:-1] < 0) & (seg[1:] >= 0))[0]
        zc = zc[zc < n]
        s = zc[-1] if len(zc) else 0
        return seg[s: s + n]

    def fx_active(self, t, kind):
        for e in self.fx:
            if e["kind"] == kind and e["t"] <= t < e["t"] + e["dur"]:
                return (t - e["t"]) / e["dur"]
        return None

    def lane_pulse(self, lane, t, tau=0.12):
        times = self.lane_t.get(lane, [])
        i = bisect.bisect_right(times, t) - 1
        if i < 0:
            return 0.0
        return float(np.exp(-(t - times[i]) / tau))
