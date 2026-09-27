"""Tests for the 254-feature extractor (SRS Step 6, FR xx).

Checks the column schema, that there is never NaN or inf, and that features behave as
expected on synthetic signals (e.g. a 440 Hz tone has its centroid near 440 Hz).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from feature_extraction import (
    FEATURE_SCHEMA_VERSION,
    FeatureExtractor,
    extract_features,
    extract_features_from_file,
    feature_columns,
    n_features,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SR = 16000

# A tone well inside the 3 s live window.
LIVE_WINDOW_SECONDS = 3.0


# test signals

def _time(seconds: float, sr: int = SR) -> np.ndarray:
    return np.arange(int(round(seconds * sr)), dtype=np.float64) / sr


def sine(seconds: float, freq: float = 440.0, amplitude: float = 0.5, sr: int = SR) -> np.ndarray:
    """A pure tone."""
    return (amplitude * np.sin(2 * np.pi * freq * _time(seconds, sr))).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(seconds * sr)), dtype=np.float32)


def noise(seconds: float, amplitude: float = 0.3, seed: int = 0, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(round(seconds * sr)))).astype(np.float32)


def mix(*signals: np.ndarray) -> np.ndarray:
    """Average of the given signals."""
    n = min(len(s) for s in signals)
    out = np.zeros(n, dtype=np.float64)
    for s in signals:
        out += s[:n].astype(np.float64)
    return (out / len(signals)).astype(np.float32)


# schema


def test_the_schema_is_exactly_254_columns_and_no_wider():
    """Exactly 254 features."""
    assert n_features() == 254
    assert len(feature_columns()) == 254
    vec = extract_features(sine(LIVE_WINDOW_SECONDS), SR)
    assert vec.shape == (254,)


def test_columns_are_unique_and_the_version_is_declared():
    """Column names are unique and the schema has a version."""
    cols = feature_columns()
    assert len(cols) == len(set(cols))
    assert FEATURE_SCHEMA_VERSION
    assert isinstance(FEATURE_SCHEMA_VERSION, str)


def test_extractor_returns_one_row_in_the_same_column_order():
    """extract_matrix gives one row in feature_columns() order."""
    extractor = FeatureExtractor()
    row = extractor.extract_matrix(sine(LIVE_WINDOW_SECONDS), SR)
    assert row.shape == (1, 254)
    # identical, not just close
    assert np.array_equal(row[0], extract_features(sine(LIVE_WINDOW_SECONDS), SR))

    desc = extractor.describe()
    assert desc["n_features"] == 254
    assert desc.get("feature_version") == FEATURE_SCHEMA_VERSION
    assert len(feature_columns()) == desc["n_features"]
    assert list(desc["columns"]) == list(feature_columns())


# no NaN or inf

HOSTILE_SIGNALS = [
    ("digital silence", silence(1.0)),
    ("single sample", np.array([0.5], dtype=np.float32)),
    ("two samples", np.array([0.5, -0.5], dtype=np.float32)),
    ("shorter than one frame", sine(0.01)),
    ("0.3 s window", sine(0.3)),
    ("very quiet tone", sine(1.0, amplitude=1e-5)),
    ("3 s live window", sine(LIVE_WINDOW_SECONDS)),
    ("dc offset", np.full(int(1.0 * SR), 0.3, dtype=np.float32)),
    ("lone impulse", np.concatenate([np.array([0.9], dtype=np.float32), silence(1.0)])),
    ("broadband noise", noise(1.0)),
]


@pytest.mark.parametrize("name,signal", HOSTILE_SIGNALS, ids=[c[0] for c in HOSTILE_SIGNALS])
def test_no_nan_or_inf_on_any_input_the_pipeline_might_see(name, signal):
    """Short, quiet or tiny inputs don't produce NaN, inf or a crash."""
    vec = extract_features(signal, SR)
    assert vec.shape == (254,)
    assert bool(np.isfinite(vec).all()), f"{name}: {int(np.isnan(vec).sum())} NaN, {int(np.isinf(vec).sum())} inf"


def test_silence_is_finite_and_not_minus_infinity():
    """Silence gives finite values, not -inf."""
    vec = extract_features(silence(1.0), SR)
    assert bool(np.isfinite(vec).all())
    assert float(np.min(vec)) > -np.inf


def test_a_real_clip_on_disk_extracts_finitely():
    """A real 44.1 kHz recording works too."""
    clip = REPO_ROOT / "audio_dataset" / "originals" / "gunshot" / "SS-GUN-0001.wav"
    if not clip.exists():
        pytest.skip(f"{clip} not present in this checkout")
    vec = extract_features_from_file(str(clip))
    assert vec.shape == (254,)
    assert bool(np.isfinite(vec).all())


# feature values


def _column(name: str) -> int:
    return feature_columns().index(name)


def test_a_440_hz_tone_reports_a_centroid_near_440_hz():
    """A 440 Hz tone has its centroid near 440 Hz."""
    vec = extract_features(sine(1.0, freq=440.0), SR)
    centroid_hz = float(vec[_column("centroid_mean")])
    assert centroid_hz == pytest.approx(440.0, abs=80.0)


def test_the_centroid_rises_with_frequency():
    """A higher tone has a higher centroid."""
    low = float(extract_features(sine(1.0, freq=300.0), SR)[_column("centroid_mean")])
    high = float(extract_features(sine(1.0, freq=3000.0), SR)[_column("centroid_mean")])
    assert high > low * 2


def test_a_pure_tone_has_a_lower_centroid_than_broadband_noise():
    """A tone's centroid is far below white noise's (about 4 kHz)."""
    tone = float(extract_features(sine(1.0, freq=880.0), SR)[_column("centroid_mean")])
    broadband = float(extract_features(noise(1.0), SR)[_column("centroid_mean")])
    assert tone < 0.5 * broadband


def test_zero_crossing_rate_tracks_frequency():
    """ZCR goes up with frequency."""
    low = float(extract_features(sine(1.0, freq=500.0), SR)[_column("zcr_mean")])
    high = float(extract_features(sine(1.0, freq=4000.0), SR)[_column("zcr_mean")])
    assert high > low * 2


def test_a_louder_tone_raises_the_log_mel_energy():
    """10x the amplitude reads as +10 dB in the tone's mel band (band 18 for 440 Hz)."""
    quiet = extract_features(sine(1.0, amplitude=0.01), SR)
    mid = extract_features(sine(1.0, amplitude=0.1), SR)
    loud = extract_features(sine(1.0, amplitude=1.0), SR)
    band = _column("melband_018_mean")
    assert float(mid[band]) - float(quiet[band]) == pytest.approx(20.0, abs=3.0)
    assert float(loud[band]) - float(mid[band]) == pytest.approx(20.0, abs=3.0)

    # the step is the same across the range
    for amp in (0.01, 0.03, 0.1, 0.3):
        lower = float(extract_features(sine(1.0, amplitude=amp / 3.0), SR)[band])
        upper = float(extract_features(sine(1.0, amplitude=amp), SR)[band])
        assert upper - lower == pytest.approx(20.0 * np.log10(3.0), abs=3.0)


def test_two_tones_a_known_interval_apart_are_resolvable():
    """Two tones 2 kHz apart show up in two separate mel bands."""
    lo_freq, hi_freq = 500.0, 2500.0
    vec = extract_features(mix(sine(1.0, freq=lo_freq), sine(1.0, freq=hi_freq)), SR)
    bands = np.array([float(vec[_column(f"melband_{i:03d}_mean")]) for i in range(128)])
    lo_band = int(np.argmax(bands))
    top = bands[lo_band]
    # a second peak above 1.5 kHz, at least a third of the first
    assert bands[lo_band + 10 :].max() > 0.33 * top
    # the 500 Hz peak is in a low band
    assert lo_band < 45


def test_mfccs_distinguish_a_tone_from_noise():
    """A tone and white noise give different MFCCs."""
    tone = extract_features(sine(1.0), SR)
    broadband = extract_features(noise(1.0), SR)
    mfcc_slice = slice(_column("mfcc_00_mean"), _column("mfcc_00_mean") + 13)
    distance = float(np.linalg.norm(tone[mfcc_slice] - broadband[mfcc_slice]))
    assert distance > 1.0


def test_the_extractor_is_deterministic():
    """Same input, same vector."""
    a = extract_features(sine(1.0), SR)
    b = extract_features(sine(1.0), SR)
    assert np.array_equal(a, b)


# live budget


def test_warm_extraction_fits_inside_the_3s_live_budget():
    """After a warm-up call, extraction is well inside the 3 s live budget."""
    signal = sine(LIVE_WINDOW_SECONDS)
    extract_features(signal, SR)  # warm-up

    timings = []
    for _ in range(10):
        start = time.perf_counter()
        extract_features(signal, SR)
        timings.append(time.perf_counter() - start)

    median_ms = 1000.0 * float(np.median(timings))
    # Allow one retry in case the CPU is busy.
    if median_ms >= 300.0:
        timings = []
        for _ in range(10):
            start = time.perf_counter()
            extract_features(signal, SR)
            timings.append(time.perf_counter() - start)
        median_ms = 1000.0 * float(np.median(timings))
    assert median_ms < 300.0, f"warm extraction took {median_ms:.1f} ms"


def test_extraction_cost_is_stable_across_windows():
    """Extraction doesn't get slower over repeated calls."""
    signal = sine(LIVE_WINDOW_SECONDS)
    extract_features(signal, SR)  # warm-up

    first = []
    for _ in range(5):
        start = time.perf_counter()
        extract_features(signal, SR)
        first.append(time.perf_counter() - start)

    for _ in range(40):
        extract_features(signal, SR)

    later = []
    for _ in range(5):
        start = time.perf_counter()
        extract_features(signal, SR)
        later.append(time.perf_counter() - start)

    assert float(np.median(later)) <= float(np.median(first)) * 3.0
