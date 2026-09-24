"""Correctness tests for the 254-dim acoustic feature extractor (SRS Step 6, FR xx).

Owner: taha (Audio DSP & feature engineering).

These tests are the guard on criterion 4 ("feature matrix complete with no NaN"): if the
extractor ever changes width, order, or starts emitting NaN/inf, a model trained on the
old schema becomes silently invalid, so the schema is asserted as a hard contract.

Every signal here is synthesised, so the expected answers are analytic -- a 440 Hz tone
must report a centroid near 440 Hz, not at DC -- rather than measured against a corpus.
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


# ----------------------------------------------------------------------------------------------
# signals with analytically known answers
# ----------------------------------------------------------------------------------------------

def _time(seconds: float, sr: int = SR) -> np.ndarray:
    return np.arange(int(round(seconds * sr)), dtype=np.float64) / sr


def sine(seconds: float, freq: float = 440.0, amplitude: float = 0.5, sr: int = SR) -> np.ndarray:
    """A pure tone; every spectral statistic of this is known in closed form."""
    return (amplitude * np.sin(2 * np.pi * freq * _time(seconds, sr))).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(seconds * sr)), dtype=np.float32)


def noise(seconds: float, amplitude: float = 0.3, seed: int = 0, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(int(round(seconds * sr)))).astype(np.float32)


def mix(*signals: np.ndarray) -> np.ndarray:
    """Sum equal-weight signals; the result stays float32 and clips nowhere by construction."""
    n = min(len(s) for s in signals)
    out = np.zeros(n, dtype=np.float64)
    for s in signals:
        out += s[:n].astype(np.float64)
    return (out / len(signals)).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# schema contract -- the frozen 254 columns
# ----------------------------------------------------------------------------------------------


def test_the_schema_is_exactly_254_columns_and_no_wider():
    """254 features, no more, no fewer. Width changes silently invalidate trained models."""
    assert n_features() == 254
    assert len(feature_columns()) == 254
    vec = extract_features(sine(LIVE_WINDOW_SECONDS), SR)
    assert vec.shape == (254,)


def test_columns_are_unique_and_the_version_is_declared():
    """A duplicated column would be dead weight; the version pins the schema for consumers."""
    cols = feature_columns()
    assert len(cols) == len(set(cols))
    assert FEATURE_SCHEMA_VERSION
    assert isinstance(FEATURE_SCHEMA_VERSION, str)


def test_extractor_returns_one_row_in_the_same_column_order():
    """sklearn consumes (n_samples, n_features); the row must line up with feature_columns()."""
    extractor = FeatureExtractor()
    row = extractor.extract_matrix(sine(LIVE_WINDOW_SECONDS), SR)
    assert row.shape == (1, 254)
    # bit-identical, not merely close: the row and the vector are the same computation
    assert np.array_equal(row[0], extract_features(sine(LIVE_WINDOW_SECONDS), SR))

    desc = extractor.describe()
    assert desc["n_features"] == 254
    assert desc.get("feature_version") == FEATURE_SCHEMA_VERSION
    assert len(feature_columns()) == desc["n_features"]
    assert list(desc["columns"]) == list(feature_columns())


# ----------------------------------------------------------------------------------------------
# criterion 4: complete matrix, never NaN or inf
# ----------------------------------------------------------------------------------------------

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
    """A short upload or a truncated live window must not produce NaN, inf, or a crash.

    The quiet and the sub-frame cases used to die in the MFCC delta: librosa's default
    9-frame Savitzky-Golay window needs at least 9 analysis frames.
    """
    vec = extract_features(signal, SR)
    assert vec.shape == (254,)
    assert bool(np.isfinite(vec).all()), f"{name}: {int(np.isnan(vec).sum())} NaN, {int(np.isinf(vec).sum())} inf"


def test_silence_is_finite_and_not_minus_infinity():
    """Silence has no spectral content; the dB conversion must still floor it, not emit -inf."""
    vec = extract_features(silence(1.0), SR)
    assert bool(np.isfinite(vec).all())
    assert float(np.min(vec)) > -np.inf


def test_a_real_clip_on_disk_extracts_finitely():
    """A real 44.1 kHz recording must round-trip the same schema as the synthesised ones."""
    clip = REPO_ROOT / "audio_dataset" / "originals" / "gunshot" / "SS-GUN-0001.wav"
    if not clip.exists():
        pytest.skip(f"{clip} not present in this checkout")
    vec = extract_features_from_file(str(clip))
    assert vec.shape == (254,)
    assert bool(np.isfinite(vec).all())


# ----------------------------------------------------------------------------------------------
# analytic correctness -- the numbers mean what they say
# ----------------------------------------------------------------------------------------------


def _column(name: str) -> int:
    return feature_columns().index(name)


def test_a_440_hz_tone_reports_a_centroid_near_440_hz():
    """Spectral centroid is the feature's headline number; it must track real frequency."""
    vec = extract_features(sine(1.0, freq=440.0), SR)
    centroid_hz = float(vec[_column("centroid_mean")])
    assert centroid_hz == pytest.approx(440.0, abs=80.0)


def test_the_centroid_rises_with_frequency():
    """A higher tone must report a higher centroid -- a sign inversion here would be silent."""
    low = float(extract_features(sine(1.0, freq=300.0), SR)[_column("centroid_mean")])
    high = float(extract_features(sine(1.0, freq=3000.0), SR)[_column("centroid_mean")])
    assert high > low * 2


def test_a_pure_tone_has_a_lower_centroid_than_broadband_noise():
    """A tone concentrates its energy at one spot; white noise spreads it flat across the band.

    Uniform power over 0..8 kHz has its centroid at 4 kHz, so the tone must come out far
    *below* the noise -- an inversion here would be a silent and very expensive bug.
    """
    tone = float(extract_features(sine(1.0, freq=880.0), SR)[_column("centroid_mean")])
    broadband = float(extract_features(noise(1.0), SR)[_column("centroid_mean")])
    assert tone < 0.5 * broadband


def test_zero_crossing_rate_tracks_frequency():
    """ZCR is a cheap frequency proxy and must move the right way."""
    low = float(extract_features(sine(1.0, freq=500.0), SR)[_column("zcr_mean")])
    high = float(extract_features(sine(1.0, freq=4000.0), SR)[_column("zcr_mean")])
    assert high > low * 2


def test_a_louder_tone_raises_the_log_mel_energy():
    """The Mel bands are dB-scaled; 10x the input must read as +10 dB in the tone's own band.

    The assertion runs on the band that actually carries the tone (band 18, centred on
    427.5 Hz for a 440 Hz tone) -- a band an octave away is dominated by the filter's
    skirt, not the tone, and does not move by the full amount.
    """
    quiet = extract_features(sine(1.0, amplitude=0.01), SR)
    mid = extract_features(sine(1.0, amplitude=0.1), SR)
    loud = extract_features(sine(1.0, amplitude=1.0), SR)
    band = _column("melband_018_mean")
    assert float(mid[band]) - float(quiet[band]) == pytest.approx(20.0, abs=3.0)
    assert float(loud[band]) - float(mid[band]) == pytest.approx(20.0, abs=3.0)

    # and the dB step must be flat across the amplitude range, not compressed
    for amp in (0.01, 0.03, 0.1, 0.3):
        lower = float(extract_features(sine(1.0, amplitude=amp / 3.0), SR)[band])
        upper = float(extract_features(sine(1.0, amplitude=amp), SR)[band])
        assert upper - lower == pytest.approx(20.0 * np.log10(3.0), abs=3.0)


def test_two_tones_a_known_interval_apart_are_resolvable():
    """Two tones 2 kHz apart must put energy in two distinct Mel bands, not smear into one."""
    lo_freq, hi_freq = 500.0, 2500.0
    vec = extract_features(mix(sine(1.0, freq=lo_freq), sine(1.0, freq=hi_freq)), SR)
    bands = np.array([float(vec[_column(f"melband_{i:03d}_mean")]) for i in range(128)])
    lo_band = int(np.argmax(bands))
    top = bands[lo_band]
    # a second, separate peak above 1.5 kHz worth at least a third of the first
    upper = bands[80:]
    assert bands[lo_band + 10 :].max() > 0.33 * top
    # the low peak must actually sit low in the band array for a 500 Hz tone
    assert lo_band < 45


def test_mfccs_distinguish_a_tone_from_noise():
    """MFCCs carry timbre; a pure tone and white noise must not map to the same vector."""
    tone = extract_features(sine(1.0), SR)
    broadband = extract_features(noise(1.0), SR)
    mfcc_slice = slice(_column("mfcc_00_mean"), _column("mfcc_00_mean") + 13)
    distance = float(np.linalg.norm(tone[mfcc_slice] - broadband[mfcc_slice]))
    assert distance > 1.0


def test_the_extractor_is_deterministic():
    """Same input, same vector: caching and resampling must not introduce state."""
    a = extract_features(sine(1.0), SR)
    b = extract_features(sine(1.0), SR)
    assert np.array_equal(a, b)


# ----------------------------------------------------------------------------------------------
# the live budget -- warm extraction must fit inside the window
# ----------------------------------------------------------------------------------------------


def test_warm_extraction_fits_inside_the_3s_live_budget():
    """The live window is 3 s; extraction must leave room for the model and the verdict.

    The first call after import pays ~1.3 s of librosa/numba one-off cost.  That is an
    import artefact, not the cost a user pays, so the extractor is primed first and the
    cold run is discarded -- the warm number is the one the live path actually sees.
    """
    signal = sine(LIVE_WINDOW_SECONDS)
    extract_features(signal, SR)  # prime: numba JIT + librosa caches

    timings = []
    for _ in range(10):
        start = time.perf_counter()
        extract_features(signal, SR)
        timings.append(time.perf_counter() - start)

    median_ms = 1000.0 * float(np.median(timings))
    # 3000 ms budget for the whole window; extraction claims well under a tenth of it.
    assert median_ms < 300.0, f"warm extraction took {median_ms:.1f} ms"


def test_extraction_cost_is_stable_across_windows():
    """The cost must not grow with how long the app has been running (a cache leak)."""
    signal = sine(LIVE_WINDOW_SECONDS)
    extract_features(signal, SR)  # prime

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
