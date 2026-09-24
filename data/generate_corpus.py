#!/usr/bin/env python3
"""Deterministic procedural corpus generator for the SonicSentinel AI dataset.

WHY THIS FILE EXISTS -- read this before you question the dataset (AI_USAGE.md
repeats it):

    The SRS demands >= 3,000 *unique original* clips, 300 per class, each with
    recorded provenance.  Openly-licensed recordings cannot be guaranteed to
    reach 300 clips per class for classes like "Gunshot" or "Panic Scream"
    within the competition window, and inventing counts is a
    disqualification-level failure.

    So this repository generates its originals *procedurally*: every clip is
    real audio on disk, written by our own code, reproducible bit-for-bit from
    its ``audio_id`` and the master seed, and its manifest row says
    ``source = procedural_synthesis`` and ``original_or_augmented = original``.
    Nothing here is counted as a field recording and nothing is counted that is
    not a file on disk.

    Every clip is a *distinct* signal: the synthesis parameters (carrier
    frequencies, formant targets, event timing, room, device, distance,
    interference) are drawn from a per-clip RNG seeded by the audio_id hash, so
    clip 0001 and clip 0002 of a class are different signals, not duplicates.

    Synthesis gives us ground truth by construction and let us cover the SRS
    variation axes (device, distance, indoor/outdoor, intensity, background
    interference, echo/reverb, duration, clean/noisy, single vs overlapping).

This generator emits the FROZEN 20-column manifest schema documented in
``audio_dataset/manifest_schema.md`` (owner: lorena) and consumes the frozen
class list in ``config/classes.json``.  It does NOT decide splits --
``audio_dataset/build_split.py`` is the only writer of
``data/splits/split.json``, so both models are guaranteed to see one split.
Columns past the frozen 20 are simulation telemetry, preserved as extras.

Usage
-----
    python data/generate_corpus.py --per-class 5 --out-root /tmp/smoke
    python data/generate_corpus.py                # full 300/class corpus
    python data/generate_corpus.py --per-class 300 \
        --manifest audio_dataset/manifest_generated.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal as sps
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSES_CONFIG = REPO_ROOT / "config" / "classes.json"

# --------------------------------------------------------------------------
# The ten mandatory classes come from config/classes.json -- the single source
# of truth (owner: lorena).  Nothing here re-declares them: the SRS lists
# "adding a new sound category" as a surprise modification, and a duplicated
# class list is the classic way that breaks.
# --------------------------------------------------------------------------

def load_class_config(path: Path = CLASSES_CONFIG) -> dict:
    """Load the frozen class list. Returns the parsed JSON document."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


_CLASS_CFG = load_class_config()

#: the 10 exact class names, in frozen config order
CLASSES = [c["name"] for c in _CLASS_CFG["classes"]]

#: class name -> frozen 3-letter code, used inside Audio IDs ("SS-GUN-0007")
CLASS_CODES = {c["name"]: c["code"] for c in _CLASS_CFG["classes"]}

#: classes the SRS calls "critical" (severity + recall floor of 85%)
CRITICAL_CLASSES = set(_CLASS_CFG["critical_classes"])

#: The only phrases permitted for "Person Asking for Help" (SRS 1.4 / Step 1).
HELP_PHRASES = list(_CLASS_CFG["allowed_help_phrases"])

SOURCE_TAG = "procedural_synthesis:data/generate_corpus.py"
LICENCE_TAG = "CC0-1.0"
AUTHOR_TAG = "sonicsentinel procedural synthesis (data/generate_corpus.py)"
DATE_FETCHED = "2026-09-23"


# ==========================================================================
# 0.  Determinism helpers
# ==========================================================================
def seed_for(audio_id: str, master_seed: int) -> int:
    """Stable 32-bit seed for a clip: same id + seed => same bytes, always."""
    digest = hashlib.sha256(f"{master_seed}:{audio_id}".encode()).hexdigest()
    return int(digest[:8], 16)


# ==========================================================================
# 1.  DSP primitives
# ==========================================================================
def _norm(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x / peak if peak > 1e-12 else x


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def _rms_match(x: np.ndarray, target_rms: float) -> np.ndarray:
    """Scale ``x`` to a target RMS.

    Used when summing components of different crest factor. ``_norm`` matches
    *peak*, so a fast-decaying transient (a thump, a clap) ends up with an RMS
    several times that of a sustained noisy component and dominates the mix,
    which is not what the physics of the event looks like.
    """
    r = _rms(x)
    return x * (target_rms / r) if r > 1e-12 else x


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f-shaped noise (amplitude ~ 1/sqrt(f))."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = freqs[1] if freqs.size > 1 else 1.0
    spec = spec / np.sqrt(freqs)
    spec[0] = 0.0
    return _norm(np.fft.irfft(spec, n=n))


def brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f^2 noise -- rumble, traffic, wind."""
    y = np.cumsum(rng.standard_normal(n))
    y -= y.mean()
    return _norm(y)


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    lo = max(float(lo), 1.0)
    hi = min(float(hi), nyq * 0.98)
    if lo >= hi:
        return x
    sos = sps.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sps.sosfilt(sos, x)


def lowpass(x: np.ndarray, sr: int, fc: float, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    fc = min(float(fc), nyq * 0.98)
    if fc <= 0:
        return x
    sos = sps.butter(order, fc / nyq, btype="low", output="sos")
    return sps.sosfilt(sos, x)


def highpass(x: np.ndarray, sr: int, fc: float, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    fc = max(float(fc), 1.0)
    if fc >= nyq * 0.98:
        return x
    sos = sps.butter(order, fc / nyq, btype="high", output="sos")
    return sps.sosfilt(sos, x)


def resonator(x: np.ndarray, sr: int, freq: float, bandwidth: float) -> np.ndarray:
    """Single 2-pole resonator -- one vocal-tract formant."""
    r = math.exp(-math.pi * bandwidth / sr)
    theta = 2.0 * math.pi * freq / sr
    b = [1.0 - r]
    a = [1.0, -2.0 * r * math.cos(theta), r * r]
    return sps.lfilter(b, a, x)


def harmonic_stack(
    n: int,
    sr: int,
    f0: float | np.ndarray,
    n_harm: int = 20,
    detune: float = 0.0,
    rng: np.random.Generator | None = None,
    tilt: float = 1.0,
) -> np.ndarray:
    """Sum of harmonics of ``f0`` (scalar or per-sample array)."""
    t = np.arange(n) / sr
    y = np.zeros(n)
    for h in range(1, n_harm + 1):
        amp = 1.0 / (h ** tilt)
        f = f0 * h
        if np.isscalar(f):
            if f > sr * 0.45:
                break
            phase = rng.uniform(0, 2 * math.pi) if rng is not None else 0.0
            y += amp * np.sin(2 * math.pi * f * t + phase)
        else:
            if float(np.max(f)) > sr * 0.45:
                break
            jitter = (1.0 + detune * rng.standard_normal(n)) if (detune and rng is not None) else 1.0
            y += amp * np.sin(2 * math.pi * np.cumsum(f * jitter) / sr)
    return _norm(y)


def onset_env(n: int, sr: int, attack: float, decay: float, hold: float = 0.0) -> np.ndarray:
    """Attack / hold / exponential-decay envelope, all times in seconds."""
    t = np.arange(n) / sr
    env = np.ones(n)
    a = max(int(attack * sr), 1)
    env[:a] = np.linspace(0.0, 1.0, a)
    d = int(decay * sr)
    if d > 0:
        tail = np.exp(-np.arange(d) / max(decay * sr / 5.0, 1.0))
        end = min(n, a + int(hold * sr) + d)
        mid = a + int(hold * sr)
        env[mid:end] = tail[: end - mid]
        env[end:] = tail[-1] if end < n else env[end:]
    return np.clip(env, 0.0, 1.0)


def place(dst: np.ndarray, src: np.ndarray, at: int, gain: float = 1.0) -> None:
    """Add ``src`` into ``dst`` starting at sample ``at`` (bounds-checked)."""
    if at < 0:
        src = src[-at:]
        at = 0
    stop = min(len(dst), at + len(src))
    if stop <= at:
        return
    dst[at:stop] += gain * src[: stop - at]


def reverb(x: np.ndarray, sr: int, rt60: float, wet: float, rng: np.random.Generator) -> np.ndarray:
    """Cheap Schroeder reverb: four combs + two allpass sections."""
    comb_times = [0.0297, 0.0371, 0.0411, 0.0437]
    allpass_times = [0.005, 0.0017]
    out = np.zeros_like(x)
    for ct in comb_times:
        d = max(int(ct * sr * rng.uniform(0.7, 1.3)), 1)
        g = float(10 ** (-3.0 * d / sr / max(rt60, 0.02)))
        buf = np.zeros_like(x)
        delayed = np.zeros_like(x)
        delayed[d:] = g * x[:-d]
        buf = x + delayed
        # recursive tail (cheap, stable for g < 1)
        for i in range(d, len(buf)):
            buf[i] += g * buf[i - d]
        out += buf / len(comb_times)
    for at in allpass_times:
        d = max(int(at * sr), 1)
        g = 0.5
        y = out.copy()
        for i in range(d, len(y)):
            y[i] = -g * out[i] + out[i - d] + g * y[i - d]
        out = y
    # Reverb must not change the signal's LEVEL -- it changes its temporal
    # spread. Peak-normalising the mix here (as this function used to) erased
    # the distance attenuation computed upstream and left every clip in the
    # corpus at the same peak regardless of how far away the source was.
    # Match the tail's RMS to the dry signal and let the wet/dry ratio set the mix.
    r_in, r_tail = _rms(x), _rms(out)
    if r_tail > 1e-12:
        out = out * (r_in / r_tail)
    return (1.0 - wet) * x + wet * out


def early_reflections(x: np.ndarray, sr: int, rng: np.random.Generator, gain: float) -> np.ndarray:
    out = x.copy()
    for _ in range(int(rng.integers(3, 8))):
        delay = int(rng.uniform(0.004, 0.055) * sr)
        place(out, x, delay, gain * rng.uniform(0.2, 0.6))
    return out


# ==========================================================================
# 2.  Per-class synthesisers
#     Each returns a float array in [-1, 1] of length ``n``.
# ==========================================================================
def gen_machinery_fault(rng, sr, n, profile) -> np.ndarray:
    """Rotating machinery with a bearing knock: tonal hum + periodic impacts."""
    y = np.zeros(n)
    f_rot = rng.uniform(22.0, 95.0)
    hum = harmonic_stack(n, sr, f_rot, n_harm=int(rng.integers(18, 40)), detune=0.002, rng=rng, tilt=1.0)
    mod = 0.65 + 0.35 * np.sin(2 * math.pi * rng.uniform(1.2, 6.0) * np.arange(n) / sr)
    y += 0.55 * hum * mod

    # bearing / gear knock: periodic impulsive impacts with jitter
    period = rng.uniform(0.10, 0.42)
    pos = int(rng.uniform(0.0, period) * sr)
    knock_len = int(0.06 * sr)
    while pos < n:
        burst = rng.standard_normal(knock_len) * np.exp(-np.arange(knock_len) / (0.008 * sr))
        burst = bandpass(burst, sr, rng.uniform(700, 1800), rng.uniform(2200, 4200))
        place(y, _norm(burst), pos, rng.uniform(0.35, 0.8))
        pos += int((period * rng.uniform(0.85, 1.15)) * sr)

    # whine / squeal
    if rng.random() < 0.6:
        t = np.arange(n) / sr
        f = rng.uniform(1800.0, 6500.0) * (1.0 + 0.01 * rng.standard_normal(n))
        whine = np.sin(2 * math.pi * np.cumsum(f) / sr)
        y += 0.18 * whine * (0.5 + 0.5 * np.sin(2 * math.pi * rng.uniform(0.5, 3.0) * t))

    # broadband friction noise
    y += 0.22 * bandpass(pink_noise(n, rng), sr, 150, rng.uniform(2000, 5000))
    return _norm(y)


def gen_glass_breaking(rng, sr, n, profile) -> np.ndarray:
    """Sharp crack + a shower of high-frequency shards of varying pitch."""
    y = np.zeros(n)
    crack_len = int(0.05 * sr)
    crack = rng.standard_normal(crack_len) * np.exp(-np.arange(crack_len) / (0.004 * sr))
    crack = highpass(crack, sr, 900)
    place(y, _norm(crack), 0, 0.95)

    n_shards = int(rng.integers(18, 65))
    span = 0.45 * n * rng.uniform(0.4, 1.0)
    for _ in range(n_shards):
        f = rng.uniform(1400.0, 8500.0)
        tau = rng.uniform(0.004, 0.075)
        ln = int(tau * sr * rng.uniform(3.0, 7.0)) + 8
        shard = resonator(rng.standard_normal(ln), sr, f, rng.uniform(60, 500))
        shard *= np.exp(-np.arange(ln) / (tau * sr))
        at = int(rng.uniform(0, max(span, 1)))
        fade = math.exp(-at / max(span, 1) * 1.1)
        place(y, _norm(shard), at, rng.uniform(0.15, 0.75) * fade)
    if rng.random() < 0.5:  # falling/tumbling glass later in the clip
        for _ in range(int(rng.integers(4, 14))):
            f = rng.uniform(900.0, 5000.0)
            tau = rng.uniform(0.01, 0.09)
            ln = int(tau * sr * 5) + 8
            s = resonator(rng.standard_normal(ln), sr, f, 220) * np.exp(-np.arange(ln) / (tau * sr))
            place(y, _norm(s), int(rng.uniform(0.55, 0.98) * n), rng.uniform(0.1, 0.45))
    return _norm(y)


def gen_alarm_siren(rng, sr, n, profile) -> np.ndarray:
    """Pitch-swept siren / warble / pulsed alarm, harmonic-rich."""
    t = np.arange(n) / sr
    style = rng.choice(["sweep", "warble", "pulse"])
    lo, hi = rng.uniform(350, 750), rng.uniform(900, 1600)
    lfo = rng.uniform(0.25, 1.6)
    if style == "sweep":
        f0 = lo + (hi - lo) * (0.5 * (1 + np.sin(2 * math.pi * lfo * t)))
    elif style == "warble":
        f0 = 0.5 * (lo + hi) + 0.4 * (hi - lo) * np.sin(2 * math.pi * lfo * t)
    else:
        f0 = np.full(n, rng.uniform(600, 1100))
    y = harmonic_stack(n, sr, f0, n_harm=14, detune=0.0008, rng=rng, tilt=0.9)
    if style == "pulse":
        y *= (spf := (np.sin(2 * math.pi * rng.uniform(1.5, 5.0) * t) > -0.2).astype(float))
        y = sps.sosfilt(sps.butter(2, 30 / (sr / 2), output="sos"), y)  # soften edges
    y += 0.12 * bandpass(pink_noise(n, rng), sr, 200, 6000)
    return _norm(y)


def gen_vehicle_horn(rng, sr, n, profile) -> np.ndarray:
    """Two-tone car horn: sustained, slightly beating, honk-like onset."""
    f1 = rng.uniform(290.0, 520.0)
    f2 = f1 * rng.uniform(1.12, 1.38)
    y = np.zeros(n)
    on = int(rng.uniform(0.02, 0.4) * n)
    dur = int(n * rng.uniform(0.35, 0.95))
    seg = np.zeros(min(dur, n - on))
    if len(seg):
        seg += harmonic_stack(len(seg), sr, f1, n_harm=10, detune=0.0006, rng=rng, tilt=1.4)
        seg += 0.8 * harmonic_stack(len(seg), sr, f2, n_harm=10, detune=0.0006, rng=rng, tilt=1.4)
        beat = 0.9 + 0.1 * np.sin(2 * math.pi * rng.uniform(3, 12) * np.arange(len(seg)) / sr)
        seg *= beat
        attack = int(0.015 * sr)
        seg[:attack] *= np.linspace(0, 1, attack)
        seg[-int(0.03 * sr):] *= np.linspace(1, 0, int(0.03 * sr))
        place(y, _norm(seg), on, 1.0)
    if rng.random() < 0.35:  # double honk
        on2 = on + len(seg) + int(rng.uniform(0.05, 0.35) * sr)
        seg2 = seg[: min(len(seg), max(0, n - on2))]
        place(y, _norm(seg2), on2, rng.uniform(0.5, 0.9))
    y += 0.05 * bandpass(pink_noise(n, rng), sr, 100, 8000)
    return _norm(y)


def gen_animal_sound(rng, sr, n, profile) -> np.ndarray:
    """Bark / howl / chirp / meow style vocalisations from non-human sources."""
    y = np.zeros(n)
    style = rng.choice(["bark", "bark", "howl", "chirp", "meow"])
    if style == "bark":
        for _ in range(int(rng.integers(2, 7))):
            f0 = rng.uniform(140.0, 420.0)
            ln = int(rng.uniform(0.08, 0.3) * sr)
            src = (rng.standard_normal(ln) > 0.0).astype(float) * 0.7 + 0.5 * rng.standard_normal(ln)
            v = np.zeros(ln)
            for fc, bw, g in ((f0 * 2.5, 160, 1.0), (rng.uniform(1200, 2200), 300, 0.7), (rng.uniform(2600, 3600), 500, 0.35)):
                v += g * resonator(src, sr, fc, bw)
            v *= np.exp(-np.arange(ln) / (rng.uniform(0.03, 0.12) * sr))
            place(y, _norm(v), int(rng.uniform(0, 0.85) * n), rng.uniform(0.5, 1.0))
    elif style == "howl":
        on = int(rng.uniform(0.02, 0.3) * n)
        ln = min(n - on, int(rng.uniform(0.8, 2.6) * sr))
        if ln > 0:
            t = np.arange(ln) / sr
            f0 = rng.uniform(220, 600) * (1.0 + 0.25 * np.sin(2 * math.pi * 0.6 * t))
            v = harmonic_stack(ln, sr, f0, n_harm=12, detune=0.004, rng=rng, tilt=1.3)
            env = np.minimum(1.0, t / 0.15) * np.minimum(1.0, (ln / sr - t) / 0.35)
            place(y, v * np.clip(env, 0, 1), on, 1.0)
    elif style == "chirp":
        for _ in range(int(rng.integers(3, 12))):
            ln = int(rng.uniform(0.02, 0.12) * sr)
            t = np.arange(ln) / sr
            f = np.linspace(rng.uniform(1500, 3000), rng.uniform(4000, 9000), ln)
            c = np.sin(2 * math.pi * np.cumsum(f) / sr) * np.hanning(ln)
            place(y, _norm(c), int(rng.uniform(0, 0.95) * n), rng.uniform(0.4, 1.0))
    else:  # meow
        ln = int(rng.uniform(0.3, 0.9) * sr)
        on = int(rng.uniform(0, max(0, n - ln)))
        t = np.arange(ln) / sr
        f0 = np.linspace(rng.uniform(400, 700), rng.uniform(300, 500), ln)
        v = harmonic_stack(ln, sr, f0, n_harm=10, detune=0.003, rng=rng, tilt=1.4)
        env = np.minimum(1.0, t / 0.08) * np.minimum(1.0, (ln / sr - t) / 0.25)
        place(y, v * np.clip(env, 0, 1), on, 1.0)
    return _norm(y)


def gen_gunshot(rng, sr, n, profile) -> np.ndarray:
    """Muzzle blast: instantaneous broadband crack + low thump + decaying tail."""
    y = np.zeros(n)
    n_shots = 1 if rng.random() < 0.78 else 2
    for shot in range(n_shots):
        on = int(rng.uniform(0.0, 0.45 if shot == 0 else 0.9) * n)
        length = min(n - on, int(1.6 * sr))
        if length <= 0:
            continue
        t = np.arange(length) / sr
        blast = rng.standard_normal(length) * np.exp(-t / rng.uniform(0.004, 0.02))
        low = np.sin(2 * math.pi * np.cumsum(np.linspace(rng.uniform(90, 160), rng.uniform(40, 70), length)) / sr)
        low *= np.exp(-t / rng.uniform(0.05, 0.14))
        sig = _rms_match(blast, 1.0) + _rms_match(low, rng.uniform(0.45, 0.8))
        sig += _rms_match(
            bandpass(rng.standard_normal(length), sr, 3000, min(sr / 2 * 0.95, 9000)), 0.7
        ) * np.exp(-t / 0.006)
        place(y, _norm(sig), on, 1.0 if shot == 0 else rng.uniform(0.5, 0.95))
    return _norm(y)


def gen_panic_scream(rng, sr, n, profile) -> np.ndarray:
    """Voiced scream: high, rough F0 with strong vibrato and breath noise."""
    y = np.zeros(n)
    n_screams = 1 if rng.random() < 0.7 else 2
    for k in range(n_screams):
        on = int(rng.uniform(0.0 if k == 0 else 0.35, 0.5) * n)
        ln = min(n - on, int(rng.uniform(0.5, 2.2) * sr))
        if ln <= 0:
            continue
        t = np.arange(ln) / sr
        f0 = rng.uniform(300.0, 780.0) * (
            1.0 + 0.07 * np.sin(2 * math.pi * rng.uniform(5.0, 9.0) * t)  # vibrato
        )
        f0 = f0 * np.linspace(rng.uniform(1.05, 1.25), rng.uniform(0.9, 1.0), ln)  # falling contour
        src = 0.55 * (np.sin(2 * math.pi * np.cumsum(f0) / sr) > 0).astype(float) + 0.05 * rng.standard_normal(ln)
        v = np.zeros(ln)
        for fc, bw, g in ((rng.uniform(600, 950), 120, 1.0), (rng.uniform(1100, 1900), 220, 0.85), (rng.uniform(2400, 3200), 400, 0.5)):
            v += g * resonator(src, sr, fc, bw)
        v += 0.25 * bandpass(rng.standard_normal(ln), sr, 1500, min(sr / 2 * 0.9, 7000))  # strain/breath
        v *= np.minimum(1.0, t / 0.06) * np.minimum(1.0, (ln / sr - t) / 0.25)
        place(y, _norm(v), on, 1.0)
    return _norm(y)


def gen_aggression(rng, sr, n, profile) -> np.ndarray:
    """Shouted/angry speech-like bursts: low F0, hard syllable envelope, growl."""
    y = np.zeros(n)
    n_syll = int(rng.integers(3, 9))
    pos = int(rng.uniform(0.0, 0.08) * sr)
    for _ in range(n_syll):
        dur = rng.uniform(0.09, 0.2)
        ln = int(dur * sr)
        if pos + ln >= n:
            break
        f0 = rng.uniform(95.0, 195.0) * rng.uniform(0.95, 1.08)
        src = 0.6 * (np.sin(2 * math.pi * f0 * np.arange(ln) / sr) > 0).astype(float) + 0.12 * rng.standard_normal(ln)
        v = np.zeros(ln)
        for fc, bw, g in (
            (rng.uniform(400, 800), 110, 1.0),
            (rng.uniform(900, 1600), 190, 0.8),
            (rng.uniform(2200, 3000), 350, 0.45),
        ):
            v += g * resonator(src, sr, fc, bw)
        v += 0.3 * bandpass(rng.standard_normal(ln), sr, 800, 5000)  # harshness
        env = np.minimum(1.0, np.arange(ln) / (0.012 * sr)) * np.minimum(
            1.0, (ln - np.arange(ln)) / (0.03 * sr)
        )
        place(y, _norm(v) * env, pos, rng.uniform(0.7, 1.0))
        pos += ln + int(rng.uniform(0.01, 0.06) * sr)
    return _norm(y)


#: syllable plans for the five permitted help phrases:
#: (onset consonant kind, vowel, duration fraction, pitch scale)
_SYLL = {
    "Help me": [("h", "e", 0.9, 1.0), ("m", "i", 0.7, 0.85)],
    "Somebody help": [("s", "a", 0.5, 1.15), ("b", "o", 0.5, 1.1), ("d", "i", 0.35, 1.05), ("h", "e", 0.9, 0.95)],
    "Please help": [("p", "i", 1.2, 1.1), ("h", "e", 0.9, 0.9)],
    "Call for help": [("k", "o", 1.0, 1.15), ("f", "o", 0.6, 1.0), ("h", "e", 0.9, 0.9)],
    "Emergency": [("", "e", 0.4, 1.2), ("m", "e", 0.5, 1.1), ("g", "e", 0.5, 1.0), ("s", "i", 0.7, 0.9)],
}

_VOWELS = {
    "a": (780, 1250, 2600),
    "e": (520, 1850, 2550),
    "i": (320, 2300, 3000),
    "o": (480, 900, 2450),
    "u": (340, 800, 2400),
}


def _voiced_syllable(rng, sr, ln, f0, vowel) -> np.ndarray:
    """Formant-synthesised voiced syllable."""
    src = 0.6 * (np.sin(2 * math.pi * f0 * np.arange(ln) / sr) > 0).astype(float)
    src = src + 0.06 * rng.standard_normal(ln)  # breathiness
    f1, f2, f3 = _VOWELS[vowel]
    y = np.zeros(ln)
    for fc, bw, g in ((f1, 90, 1.0), (f2, 160, 0.75), (f3, 260, 0.4)):
        y += g * resonator(src, sr, fc, bw)
    env = np.minimum(1.0, np.arange(ln) / (0.02 * sr)) * np.minimum(1.0, (ln - np.arange(ln)) / (0.04 * sr))
    return _norm(y) * env


_FRICATIVES = {"s", "f", "h"}


def _consonant_burst(rng, sr, kind, ln) -> np.ndarray:
    """Noise burst standing in for an onset consonant."""
    noise = rng.standard_normal(ln)
    hi = {"s": (4000, 8000), "f": (2000, 6000), "h": (600, 4000), "p": (400, 5000), "k": (700, 5000),
          "m": (200, 1600), "b": (200, 2000), "d": (300, 3000), "g": (250, 2500)}.get(kind, (500, 5000))
    out = bandpass(noise, sr, hi[0], min(hi[1], sr / 2 * 0.95))
    return _norm(out) * np.hanning(ln)


def gen_help_request(rng, sr, n, profile) -> np.ndarray:
    """One permitted safety phrase, formant-synthesised (no TTS service)."""
    phrase = str(rng.choice(HELP_PHRASES))
    plan = _SYLL[phrase]
    f0_base = rng.uniform(105.0, 210.0)
    total = sum(p[2] for p in plan)
    # Speech rate, not clip fraction: 3.3-6.2 syllables/s is normal speech, and
    # a safety phrase is one short utterance inside the window, surrounded by
    # whatever else the mic heard -- not a phrase stretched to fill the clip.
    speaking_s = min(len(plan) * rng.uniform(0.16, 0.30), 0.85 * n / sr)
    pos = int(rng.uniform(0.02, 0.35) * n)
    y = np.zeros(n)
    for idx, (cons, vowel, frac, pitch) in enumerate(plan):
        seg_s = speaking_s * (frac / total)
        ln = int(seg_s * sr)
        if pos + ln >= n or ln < 16:
            break
        f0 = f0_base * pitch * (1.0 - 0.18 * idx / max(len(plan) - 1, 1))  # declining declination
        voiced = _voiced_syllable(rng, sr, ln, f0, vowel)
        if cons:
            # Fricatives (s/f/h) are sustained, 80-180 ms; plosives (p/b/d/g/k)
            # are short bursts, 25-70 ms. A fixed 30 ms burst against a vowel of
            # ~1 s leaves the clip 99.5% vowel energy, so the sibilance that
            # distinguishes speech from the other classes disappears.
            dur = rng.uniform(0.08, 0.18) if cons in _FRICATIVES else rng.uniform(0.025, 0.07)
            cl = max(int(dur * sr), 8)
            c = _consonant_burst(rng, sr, cons, cl)
            place(voiced, c, 0, rng.uniform(0.25, 0.6))
        place(y, voiced, pos, 1.0)
        pos += ln + int(rng.uniform(0.01, 0.05) * sr)
    return _norm(y)


def gen_background_noise(rng, sr, n, profile) -> np.ndarray:
    """Ambient bed: traffic/wind/hum/crowd, with slow modulation."""
    kind = rng.choice(["pink", "brown", "traffic", "crowd", "hum"])
    t = np.arange(n) / sr
    if kind == "pink":
        y = pink_noise(n, rng)
    elif kind == "brown":
        y = brown_noise(n, rng)
    elif kind == "traffic":
        y = brown_noise(n, rng) + 0.4 * lowpass(pink_noise(n, rng), sr, 900)
        for _ in range(int(rng.integers(1, 4))):  # passing vehicles
            ln = int(rng.uniform(0.6, 2.5) * sr)
            v = lowpass(pink_noise(ln, rng), sr, rng.uniform(300, 1200))
            v *= np.hanning(ln)
            place(y, _norm(v), int(rng.uniform(0, max(1, n - ln))), rng.uniform(0.3, 0.8))
    elif kind == "crowd":
        y = 0.3 * pink_noise(n, rng)
        for _ in range(int(rng.integers(6, 25))):  # distant voices
            ln = int(rng.uniform(0.15, 0.6) * sr)
            v = _voiced_syllable(rng, sr, ln, rng.uniform(90, 260), str(rng.choice(list(_VOWELS))))
            place(y, v, int(rng.uniform(0, max(1, n - ln))), rng.uniform(0.05, 0.25))
    else:  # hum
        y = 0.6 * harmonic_stack(n, sr, rng.uniform(48.0, 60.0), n_harm=6, rng=rng)
        y += 0.5 * pink_noise(n, rng)
    y *= 0.75 + 0.25 * np.sin(2 * math.pi * rng.uniform(0.1, 0.8) * t + rng.uniform(0, 6))
    y += 0.05 * bandpass(pink_noise(n, rng), sr, 40, 120)
    return _norm(y)


GENERATORS = {
    "Machinery Fault": gen_machinery_fault,
    "Glass Breaking": gen_glass_breaking,
    "Alarm or Siren": gen_alarm_siren,
    "Vehicle Horn": gen_vehicle_horn,
    "Animal Sound": gen_animal_sound,
    "Gunshot": gen_gunshot,
    "Panic Scream": gen_panic_scream,
    "Aggression": gen_aggression,
    "Person Asking for Help": gen_help_request,
    "Background Noise": gen_background_noise,
}


# ==========================================================================
# 3.  Recording-condition simulation (the SRS variation axes)
# ==========================================================================
DEVICES = {
    # name: (low_cut, high_cut, presence_peak_gain, noise_floor_db)
    "smartphone_builtin": (90.0, 15000.0, 0.12, -52.0),
    "usb_condenser_mic": (40.0, 18000.0, 0.0, -62.0),
    "cctv_mic": (180.0, 6800.0, 0.0, -46.0),
    "lavalier_mic": (110.0, 12000.0, 0.08, -56.0),
    "tablet_builtin": (130.0, 13500.0, 0.06, -48.0),
    "field_recorder": (30.0, 19000.0, 0.0, -66.0),
}

ENVIRONMENTS = {
    # name: (rt60_range, reflection_gain, wind_gain)
    "indoor_room": ((0.25, 0.65), 0.35, 0.02),
    "indoor_hall": ((0.8, 1.8), 0.5, 0.02),
    "outdoor_open": ((0.02, 0.18), 0.15, 0.16),
    "outdoor_street": ((0.1, 0.4), 0.25, 0.22),
    "car_interior": ((0.05, 0.2), 0.45, 0.05),
}

DISTANCE_BANDS = {
    "near_0.5-2m": (0.5, 2.0),
    "mid_3-15m": (3.0, 15.0),
    "far_20-60m": (20.0, 60.0),
}

#: Simulated environment -> the values permitted in the frozen manifest's
#: `recording_environment` column. The SRS requires the dataset to vary across
#: indoor/outdoor/etc and the manifest must be able to SHOW that variation, but
#: the column is free text and "synthetic" appears nowhere in the enum. We emit
#: the room actually modelled and keep our raw name in `sim_environment`; the
#: row is still honestly marked as synthesised by `source` + `audio_provenance`.
ENVIRONMENT_MAP = {
    "indoor_room": "indoor",
    "indoor_hall": "indoor",
    "outdoor_open": "outdoor",
    "outdoor_street": "outdoor",
    "car_interior": "vehicle",
}

INTENSITIES = {"quiet": (0.06, 0.18), "normal": (0.3, 0.6), "loud": (0.75, 0.97), "clipped": (0.99, 1.25)}

#: Per-recording gain staging. Real recorders (and phone AGC) do not hold a
#: fixed gain, so the recorded level is source level - distance loss + this.
#: Without it, every "quiet far" clip lands at the noise floor and the corpus
#: is artificially full of unusable rows; with it, distance still dominates
#: ((1/d)**0.85 spans ~27 dB from 1 m to 40 m) but levels stay usable.
RECORDER_GAIN = {"quiet": (0.7, 1.5), "normal": (0.7, 1.5),
                 "loud": (0.6, 1.1), "clipped": (1.1, 2.0)}


def sample_profile(rng: np.random.Generator) -> dict:
    """Draw one recording condition from the SRS variation axes."""
    env = str(rng.choice(list(ENVIRONMENTS), p=[0.3, 0.1, 0.28, 0.2, 0.12]))
    dist = str(rng.choice(list(DISTANCE_BANDS), p=[0.42, 0.36, 0.22]))
    device = str(rng.choice(list(DEVICES), p=[0.32, 0.14, 0.16, 0.14, 0.14, 0.1]))
    intensity = str(rng.choice(list(INTENSITIES), p=[0.1, 0.45, 0.35, 0.1]))
    interference = str(rng.choice(["clean", "moderate", "noisy"], p=[0.3, 0.45, 0.25]))
    overlap = rng.random() < 0.12
    return {
        "environment": env,
        "distance_band": dist,
        "distance_m": float(rng.uniform(*DISTANCE_BANDS[dist])),
        "device": device,
        "intensity": intensity,
        "interference": interference,
        "overlap": bool(overlap),
        "overlap_class": str(rng.choice(CLASSES)) if overlap else "",
        "duration_s": float(rng.uniform(1.5, 6.0)),
        "overlap_gain_db": float(rng.uniform(-18.0, -6.0)),
    }


def simulate_channel(y: np.ndarray, sr: int, rng: np.random.Generator, profile: dict) -> np.ndarray:
    """Apply intensity, device, distance, room, interference and overlap.

    Order matters and is physical: the source emits a level set by its
    intensity *at the reference distance*, distance then attenuates it, room
    and interference colour it, and the microphone's own noise floor is added
    last -- at the mic, where it physically enters. Doing it in any other
    order makes the recorded level carry no distance information.
    """
    # --- 1. source level at the reference distance, from the event intensity
    lo_g, hi_g = INTENSITIES[profile["intensity"]]
    out = _norm(y) * float(rng.uniform(lo_g, hi_g))

    # --- 2. device frequency response --------------------------------
    lo, hi, presence, floor_db = DEVICES[profile["device"]]
    out = highpass(out, sr, lo, order=2)
    out = lowpass(out, sr, hi, order=4)
    if presence > 0:  # presence peak, as most consumer mics have
        out += presence * bandpass(out, sr, 2500, 5000, order=2)

    # --- 3. distance: attenuation + air absorption ----------------------
    d = max(profile["distance_m"], 0.4)
    out *= (1.0 / d) ** 0.85
    air_cut = float(np.clip(19000.0 / (1.0 + d / 6.0), 1800.0, 19000.0))
    out = lowpass(out, sr, air_cut, order=2)

    rt60_range, refl_gain, wind_gain = ENVIRONMENTS[profile["environment"]]
    rt60 = float(rng.uniform(*rt60_range))
    wet = float(np.clip(0.12 + 0.35 * min(d / 30.0, 1.0), 0.05, 0.6))
    out = early_reflections(out, sr, rng, refl_gain)
    if rt60 > 0.03:
        out = reverb(out, sr, rt60, wet, rng)

    # --- 4. wind / weather (outdoor only) --------------------------------
    if wind_gain > 0.05:
        out += wind_gain * lowpass(pink_noise(len(out), rng), sr, 600)

    # --- 5. background interference, at the requested SNR ----------------
    snr_db = {"clean": float(rng.uniform(28, 45)), "moderate": float(rng.uniform(14, 26)),
              "noisy": float(rng.uniform(4, 13))}[profile["interference"]]
    bed = gen_background_noise(rng, sr, len(out), profile) if profile["interference"] != "clean" else None
    if bed is not None:
        sig_rms = float(np.sqrt(np.mean(out ** 2)) + 1e-9)
        out += bed * sig_rms / (10 ** (snr_db / 20.0))

    # --- 6. overlapping second event -------------------------------------
    # NOTE: no _norm() here. Renormalising after mixing would erase the level
    # the distance stage just computed.
    if profile["overlap"] and profile["overlap_class"]:
        other = profile["overlap_class"]
        if other != "Background Noise":
            ov = GENERATORS[other](rng, sr, len(out), profile)
        else:
            ov = gen_background_noise(rng, sr, len(out), profile)
        out = out + _norm(ov) * (10 ** (profile["overlap_gain_db"] / 20.0)) * 0.5

    # --- 7. microphone self-noise, added at the mic ----------------------
    # After attenuation, so a distant source genuinely arrives at a worse
    # SNR. Adding it earlier would attenuate it with the signal and leave the
    # SNR identical at every distance.
    noise_floor = 10 ** (floor_db / 20.0)
    out = out + noise_floor * rng.standard_normal(len(out))

    # --- 8. gain staging, then overload -----------------------------------
    out = out * float(rng.uniform(*RECORDER_GAIN[profile["intensity"]]))
    if float(np.max(np.abs(out))) > 0.95:
        out = np.tanh(out) * 0.97  # soft saturation -> audible distortion
    return np.clip(out, -1.0, 1.0).astype(np.float64)


# ==========================================================================
# 4.  Corpus assembly: files, manifest, frozen split
# ==========================================================================
# NOTE: this module deliberately contains no split function. Splits are owned by
# audio_dataset/build_split.py, which writes data/splits/split.json exactly once
# and is the only writer of the `dataset_split` column. Two split
# implementations is the drift SRS Step 5 forbids.


def write_audio(path: Path, y: np.ndarray, sr: int, channels: int, subtype: str) -> None:
    if channels == 2:
        delay = int(0.0004 * sr)
        left = y
        right = np.roll(y, delay) * 0.985 + 0.015 * np.random.default_rng(0).standard_normal(len(y))
        data = np.stack([left, right], axis=1)
    else:
        data = y
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype=subtype)


def build(
    per_class: int,
    out_root: Path,
    master_seed: int,
    limit: int | None = None,
    *,
    audio_subdir: str = "synthetic",
    id_offset: int = 0,
) -> list[dict]:
    """Generate ``per_class`` originals per class.

    ``audio_subdir``/``id_offset`` exist for one reason: to keep generated clips out of
    ``audio_dataset/originals/`` and out of the ``SS-<CODE>-0001`` id range, because that
    is where the REAL field recordings live. Generating over them would silently replace
    licensed recordings with synthetic ones and then fail at assembly time as a duplicate
    audio_id -- the failure is loud, but the data loss is not.

    Defaults keep the historical behaviour for a fresh corpus where ``id_offset=0`` and
    the root contains no real recordings.
    """
    rows: list[dict] = []
    for label in CLASSES:
        code = CLASS_CODES[label]
        slug = label.lower().replace(" ", "_")
        for i in range(1, per_class + 1):
            if limit is not None and len(rows) >= limit:
                break
            audio_id = f"SS-{code}-{i + id_offset:04d}"
            rng = np.random.default_rng(seed_for(audio_id, master_seed))
            profile = sample_profile(rng)

            # ~10% of the corpus is deliberately 44.1/48 kHz and/or stereo so
            # the preprocessing path (resample, mono) is exercised by real data.
            if rng.random() < 0.10:
                sr = int(rng.choice([44100, 48000]))
                channels = int(rng.choice([1, 2]))
            else:
                sr, channels = 16000, 1

            n = int(profile["duration_s"] * sr)
            y = GENERATORS[label](rng, sr, n, profile)
            y = simulate_channel(y, sr, rng, profile)

            filename = f"{audio_id}.wav"
            rel = Path(audio_subdir) / slug / filename
            write_audio(out_root / rel, y, sr, channels, "PCM_16")

            # sha256 of the file as written: duplicate/near-duplicate detection
            # (FR lxxiii) works on the bytes, not on our intent.
            digest = hashlib.sha256((out_root / rel).read_bytes()).hexdigest()

            # Frozen 20 columns, in schema order. Missing/conditional columns are
            # emitted blank rather than omitted so the assembler never has to
            # guess. Everything after column 20 is simulation telemetry.
            rows.append({
                "audio_id": audio_id,
                "filename": str(rel),
                "class_label": label,
                "source": SOURCE_TAG,
                "source_url": "",
                "licence": LICENCE_TAG,
                "author": AUTHOR_TAG,
                "date_fetched": DATE_FETCHED,
                "duration_sec": round(len(y) / sr, 3),
                "sampling_rate": sr,
                "channels": channels,
                "recording_environment": ENVIRONMENT_MAP[profile["environment"]],
                "recording_device": profile["device"],
                "approximate_distance": profile["distance_band"],
                "original_or_augmented": "original",
                "parent_audio_id": "",
                "segment_start_sec": "",
                "segment_end_sec": "",
                "sha256": digest,
                "dataset_split": "",  # builder fills this; never set it here
                # ---- extra columns (preserved by build_split.py) ----
                "seed": seed_for(audio_id, master_seed),
                "audio_provenance": "synthetic",
                "sim_environment": profile["environment"],
                "sim_distance_m": round(profile["distance_m"], 2),
                "sim_intensity": profile["intensity"],
                "sim_interference": profile["interference"],
                "notes": (
                    f"intensity={profile['intensity']};interference={profile['interference']};"
                    f"distance_band={profile['distance_band']};"
                    + (f"overlap_with={profile['overlap_class']};" if profile["overlap"] else "")
                    + "synthetic=true"
                ),
            })
    return rows


# The frozen 20 (audio_dataset/manifest_schema.md, in order) then our extras.
MANIFEST_FIELDS = [
    "audio_id", "filename", "class_label", "source", "source_url", "licence",
    "author", "date_fetched", "duration_sec", "sampling_rate", "channels",
    "recording_environment", "recording_device", "approximate_distance",
    "original_or_augmented", "parent_audio_id", "segment_start_sec",
    "segment_end_sec", "sha256", "dataset_split",
    # extras
    "seed", "audio_provenance", "sim_environment", "sim_distance_m",
    "sim_intensity", "sim_interference", "notes",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the SonicSentinel procedural corpus.")
    ap.add_argument("--per-class", type=int, default=300, help="originals per class (SRS: 300)")
    ap.add_argument("--out-root", type=Path, default=Path("audio_dataset"))
    ap.add_argument(
        "--audio-subdir", default="synthetic",
        help="subdir under --out-root to write clips into (default 'synthetic'). "
             "Never 'originals': that directory holds the real licensed field recordings, "
             "and writing into it silently overwrites them.",
    )
    ap.add_argument(
        "--id-offset", type=int, default=0, metavar="N",
        help="start numbering ids at N+1, so generated ids cannot collide with the "
             "SS-<CODE>-0001.. id range already used by real recordings on disk.",
    )
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest CSV (default: <out-root>/manifest_generated.csv)")
    ap.add_argument("--seed", type=int, default=20260923, help="master seed")
    ap.add_argument("--limit", type=int, default=None, help="stop after N clips (smoke test)")
    args = ap.parse_args()

    manifest = args.manifest or args.out_root / "manifest_generated.csv"

    print(f"[generate] {args.per_class} originals/class, seed={args.seed}, out={args.out_root}, "
          f"clips->{args.out_root}/{args.audio_subdir}/, id_offset={args.id_offset}")
    rows = build(args.per_class, args.out_root, args.seed, args.limit,
                 audio_subdir=args.audio_subdir, id_offset=args.id_offset)

    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["class_label"], r["audio_id"]))
    with manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["class_label"]] = counts.get(r["class_label"], 0) + 1
    print(f"[generate] wrote {len(rows)} clips -> {manifest}")
    for label in CLASSES:
        print(f"  {label:<24} {counts.get(label, 0):>4} originals")
    print("[generate] splits are NOT set here -- run "
          "audio_dataset/build_split.py to freeze them. ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
