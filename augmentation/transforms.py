"""Audio augmentations (SRS FR xix), also used to degrade clips for robustness tests.

Each function takes a mono float32 waveform and sample rate and returns a new waveform.
Randomness comes from a numpy Generator passed in, so any copy can be regenerated from
its seed.

Used for training copies (train split only, tagged as augmented with a parent ID) and
for robustness probes on test clips (scored, then discarded). Levels are kept moderate;
very low SNRs mostly taught the model to raise false alarms.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64)) + _EPS))


def _peak_safe(y: np.ndarray, limit: float = 0.999) -> np.ndarray:
    # Scale down if needed rather than clip.
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return (y * (limit / peak) if peak > limit else y).astype(np.float32)


def coloured_noise(n: int, rng: np.random.Generator, colour: str = "pink") -> np.ndarray:
    """White, pink (1/f) or brown (1/f^2) noise, unit RMS."""
    white = rng.standard_normal(n)
    if colour == "white":
        noise = white
    else:
        spectrum = np.fft.rfft(white)
        freqs = np.maximum(np.fft.rfftfreq(n), 1.0 / n)
        exponent = 0.5 if colour == "pink" else 1.0
        noise = np.fft.irfft(spectrum / freqs ** exponent, n)
    return (noise / (_rms(noise))).astype(np.float32)


def add_noise(y: np.ndarray, sr: int, rng: np.random.Generator, *, snr_db: float,
              noise: np.ndarray | None = None, colour: str = "pink") -> np.ndarray:
    """Mix in noise at a target SNR.

    ``noise`` can be a real recording; it is looped or cropped at a random offset.
    """
    if noise is None or noise.size == 0:
        bed = coloured_noise(y.size, rng, colour)
    else:
        reps = int(np.ceil(y.size / noise.size)) + 1
        looped = np.tile(noise.astype(np.float32), reps)
        offset = int(rng.integers(0, max(1, looped.size - y.size)))
        bed = looped[offset:offset + y.size]
        bed = bed / _rms(bed)
    gain = _rms(y) / (10 ** (snr_db / 20.0))
    return _peak_safe(y + gain * bed)


def time_shift(y: np.ndarray, sr: int, rng: np.random.Generator, *,
               max_shift_sec: float = 0.5) -> np.ndarray:
    """Shift the audio in time, filling with zeros (not wrapping around)."""
    limit = max(0, y.size - 1)
    shift = int(np.clip(rng.uniform(-max_shift_sec, max_shift_sec) * sr, -limit, limit))
    out = np.zeros_like(y)
    if shift >= 0:
        out[shift:] = y[: y.size - shift]
    else:
        out[:shift] = y[-shift:]
    return out


def pitch_shift(y: np.ndarray, sr: int, rng: np.random.Generator, *,
                max_semitones: float = 2.0) -> np.ndarray:
    import librosa

    steps = float(rng.uniform(-max_semitones, max_semitones))
    return _peak_safe(librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=steps))


def time_stretch(y: np.ndarray, sr: int, rng: np.random.Generator, *,
                 low: float = 0.85, high: float = 1.15) -> np.ndarray:
    import librosa

    rate = float(rng.uniform(low, high))
    return _peak_safe(librosa.effects.time_stretch(y.astype(np.float32), rate=rate))


def volume(y: np.ndarray, sr: int, rng: np.random.Generator, *,
           low_db: float = -12.0, high_db: float = 6.0) -> np.ndarray:
    return _peak_safe(y * 10 ** (rng.uniform(low_db, high_db) / 20.0))


def room_impulse(sr: int, rng: np.random.Generator, *, rt60: float) -> np.ndarray:
    """Synthetic room response: a few early reflections plus a decaying tail with the given RT60."""
    length = int(sr * min(1.5, rt60 * 1.2))
    t = np.arange(length) / sr
    tail = rng.standard_normal(length) * np.exp(-6.9078 * t / rt60)  # ln(1000) = 6.9078
    ir = 0.3 * tail
    ir[0] = 1.0
    for _ in range(6):  # early reflections in the first 50 ms
        k = int(rng.uniform(0.003, 0.05) * sr)
        if k < length:
            ir[k] += rng.uniform(0.2, 0.6) * rng.choice([-1, 1])
    return (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)


def reverberate(y: np.ndarray, sr: int, rng: np.random.Generator, *,
                rt60: float = 0.4, wet: float = 0.35) -> np.ndarray:
    from scipy.signal import fftconvolve

    wet_signal = fftconvolve(y, room_impulse(sr, rng, rt60=rt60))[: y.size]
    wet_signal *= _rms(y) / _rms(wet_signal)
    return _peak_safe((1 - wet) * y + wet * wet_signal)


def distance(y: np.ndarray, sr: int, rng: np.random.Generator, *,
             metres: float = 10.0) -> np.ndarray:
    """Rough distance simulation: level drop, a gentle low-pass and more reverb."""
    from scipy.signal import butter, sosfilt

    level = 1.0 / max(1.0, metres)
    cutoff = float(np.clip(12000.0 / (1 + metres / 20.0), 2500.0, 0.45 * sr))
    sos = butter(2, cutoff, btype="low", fs=sr, output="sos")
    far = sosfilt(sos, y).astype(np.float32) * level
    return reverberate(far, sr, rng, rt60=0.3 + 0.02 * metres,
                       wet=float(np.clip(0.15 + metres / 60.0, 0.15, 0.7)))


DEVICE_BANDS = {
    # (low Hz, high Hz, soft-clip drive), roughly like real devices.
    "phone": (300.0, 3400.0, 1.5),
    "laptop": (150.0, 7000.0, 1.2),
    "cctv": (200.0, 5000.0, 2.0),
}


def device(y: np.ndarray, sr: int, rng: np.random.Generator, *,
           kind: str | None = None) -> np.ndarray:
    """Device simulation: band-limit plus light saturation."""
    from scipy.signal import butter, sosfilt

    kind = kind or str(rng.choice(sorted(DEVICE_BANDS)))
    low, high, drive = DEVICE_BANDS[kind]
    high = min(high, 0.45 * sr)
    sos = butter(4, [low, high], btype="band", fs=sr, output="sos")
    banded = sosfilt(sos, y)
    return _peak_safe(np.tanh(drive * banded) / np.tanh(drive))


def overlay(y: np.ndarray, other: np.ndarray, rng: np.random.Generator, *,
            ratio_db: float = -6.0) -> np.ndarray:
    """Mix in a second sound at ``ratio_db`` relative to ``y``."""
    other = np.resize(other.astype(np.float32), y.size)
    return _peak_safe(y + other * (_rms(y) / _rms(other)) * 10 ** (ratio_db / 20.0))


def partial(y: np.ndarray, sr: int, rng: np.random.Generator, *,
            keep: float = 0.5) -> np.ndarray:
    """Keep only a random part of the clip."""
    n = max(int(sr * 0.5), int(y.size * keep))
    if n >= y.size:
        return y
    start = int(rng.integers(0, y.size - n))
    return y[start:start + n].copy()


# name -> callable(y, sr, rng), used by augment_dataset.
TRAINING_RECIPES = {
    "noise": lambda y, sr, rng: add_noise(y, sr, rng, snr_db=float(rng.uniform(8, 20))),
    "shift": lambda y, sr, rng: time_shift(y, sr, rng),
    "pitch": lambda y, sr, rng: pitch_shift(y, sr, rng),
    "stretch": lambda y, sr, rng: time_stretch(y, sr, rng),
    "volume": lambda y, sr, rng: volume(y, sr, rng),
    "reverb": lambda y, sr, rng: reverberate(y, sr, rng, rt60=float(rng.uniform(0.2, 0.8))),
    "distance": lambda y, sr, rng: distance(y, sr, rng, metres=float(rng.uniform(5, 25))),
    "device": lambda y, sr, rng: device(y, sr, rng),
}
