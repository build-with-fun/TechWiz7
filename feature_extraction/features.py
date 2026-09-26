"""Hand-made acoustic features (SRS Step 6, FR xx): 254 numbers per recording.

The column order is locked by FEATURE_SCHEMA_VERSION and feature_columns(); a saved model
is bundled with that list, so a reordered column is a load-time error rather than a
silently wrong prediction (tests/test_feature_extraction.py checks they cannot drift).
The STFT is computed once per signal and MFCC, chroma, centroid, bandwidth, roll-off and
flatness are all derived from it.

These features feed the classical baseline (HistGradientBoosting). The served Python model
uses CNN14 embeddings instead (embeddings.py). What each column means, since any of them
may come up in a viva:

* melband_*_mean  -- average log-energy in each of 128 mel bands: the spectral envelope.
* mfcc_*_mean/std -- that envelope compressed into 20 decorrelated numbers (timbre).
* dmfcc_*         -- how fast the MFCCs change: separates a steady siren from a gunshot.
* chroma_*        -- energy per pitch class: tonal machine hum versus broadband noise.
* zcr_*           -- zero-crossing rate: high for noise and fricatives, low for tones.
* rms_*_db        -- loudness over time in dBFS.
* centroid_*      -- spectral brightness in Hz (glass is bright, rumble is dark).
* bandwidth_*     -- spread of the spectrum around the centroid.
* rolloff_*       -- frequency below which 85% of the energy lies.
* flatness_*      -- 0 for a pure tone, 1 for white noise.
* onset_*         -- strength and rate of energy onsets (impulsiveness).
* crest_factor_db -- peak minus RMS: large for an impulse, small for a sustained alarm.
* tempo_bpm       -- periodicity of the onset envelope, for rhythmic sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np

from audio_preprocessing import config as cfg_mod
from audio_preprocessing.exceptions import AudioRejected, UNREADABLE_FORMAT
from audio_preprocessing.transforms import peak_dbfs, rms_dbfs, to_mono

FEATURE_SCHEMA_VERSION = "audiofeat-1.0.0"

# Mean and spread of every frame-wise series. Higher moments added noise, not information, on 3 s
# windows.
FRAME_STATS = ("mean", "std")


def _fc(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Feature config reconciled with the audio block, so the segment length cannot diverge.

    A config carrying its own ``_audio_block`` (from config.feature_config(config_dir=...))
    takes precedence, so a pipeline built against another config directory segments at
    that directory's length.
    """
    conf = dict(cfg) if cfg is not None else cfg_mod.feature_config()
    audio = conf.get("_audio_block")
    if not isinstance(audio, dict):
        audio = cfg_mod.audio_config()
    conf["sample_rate"] = int(audio["target_sample_rate"])
    conf["segment_duration_sec"] = float(audio["segment_duration_sec"])
    return conf


def mel_tensor_shape(cfg: dict[str, Any] | None = None) -> tuple[int, int]:
    """(n_mels, n_frames) for one segment; n_frames = 1 + n_samples // hop (94 at 3 s, 16 kHz, hop
    512).
    """
    conf = _fc(cfg)
    n_samples = int(round(float(conf["segment_duration_sec"]) * int(conf["sample_rate"])))
    n_frames = 1 + n_samples // int(conf["hop_length"])
    return int(conf["n_mels"]), int(n_frames)


# Column names -- the locked schema

def feature_columns(cfg: dict[str, Any] | None = None) -> list[str]:
    """Column names in the exact order the extractor emits them; bundle them with a saved model."""
    conf = _fc(cfg)
    n_mfcc = int(conf["n_mfcc"])
    n_chroma = int(conf["n_chroma"])
    n_mels = int(conf["n_mels"])
    names: list[str] = []
    for stat in FRAME_STATS:
        names += [f"mfcc_{i:02d}_{stat}" for i in range(n_mfcc)]
    if bool(conf.get("include_delta_mfcc", True)):
        for stat in FRAME_STATS:
            names += [f"dmfcc_{i:02d}_{stat}" for i in range(n_mfcc)]
    for stat in FRAME_STATS:
        names += [f"chroma_{i:02d}_{stat}" for i in range(n_chroma)]
    names += ["zcr_mean", "zcr_std", "zcr_p10", "zcr_p90"]
    names += ["rms_mean_db", "rms_std_db", "rms_max_db", "rms_p10_db", "rms_p90_db"]
    names += ["centroid_mean", "centroid_std"]
    names += ["bandwidth_mean", "bandwidth_std"]
    names += ["rolloff_mean", "rolloff_std"]
    names += ["flatness_mean", "flatness_std"]
    names += ["onset_mean", "onset_max", "onset_rate"]
    names += ["crest_factor_db"]
    names += [f"melband_{i:03d}_mean" for i in range(n_mels)]
    names += ["tempo_bpm"]
    return names


def n_features(cfg: dict[str, Any] | None = None) -> int:
    """Width of the feature vector -- derived from the schema, never written down twice."""
    return len(feature_columns(cfg))


# The extractor

class FeatureExtractor:
    """Turns a PreprocessedAudio (or a raw array) into the locked feature vector.

    extract() gives one row per recording, extract_matrix() the (1, n) shape sklearn wants,
    extract_segments() one row per segment.
    """

    def __init__(self, *, feature_config: dict[str, Any] | None = None, extractor_version: str | None = None) -> None:
        self.config = _fc(feature_config)
        self.feature_version = str(self.config.get("feature_version", FEATURE_SCHEMA_VERSION))
        self.extractor_version = extractor_version or FEATURE_SCHEMA_VERSION
        self.columns = feature_columns(self.config)

    def describe(self) -> dict[str, Any]:
        """What a saved model bundle needs to reproduce this extractor's output."""
        return {
            "feature_version": self.feature_version,
            "extractor_version": self.extractor_version,
            "n_features": len(self.columns),
            "columns": list(self.columns),
            "n_fft": int(self.config["n_fft"]),
            "hop_length": int(self.config["hop_length"]),
            "n_mels": int(self.config["n_mels"]),
            "n_mfcc": int(self.config["n_mfcc"]),
            "sample_rate": int(self.config["sample_rate"]),
            "segment_duration_sec": float(self.config["segment_duration_sec"]),
            "mel_tensor_shape": list(mel_tensor_shape(self.config)),
            "config_source": self.config.get("_source"),
        }

    def extract(self, preprocessed_or_samples: Any, sample_rate: int | None = None) -> np.ndarray:
        """One feature row for a whole recording: the mean over its segments.

        Averaging per-segment statistics keeps a 30 s upload on the same scale as a 3 s training
        clip. Tempo is the exception, estimated over the whole signal because 3 s is too short
        for a stable estimate.
        """
        y, sr, segments = _as_signal(preprocessed_or_samples, sample_rate)
        if y.size == 0:
            raise AudioRejected(UNREADABLE_FORMAT, "no samples to extract features from")

        conf = self.config
        seg_seconds = float(conf["segment_duration_sec"])

        from audio_preprocessing.transforms import segment_bounds

        bounds = segment_bounds(y.size, sr, seg_seconds, mode="cover")
        rows = [self._vector(y[s:e], sr, conf, tempo_bpm=None) for s, e in bounds]
        matrix = np.vstack(rows)
        mean_vec = matrix.mean(axis=0)

        tempo = estimate_tempo(y, sr, conf)
        mean_vec[int(self.columns.index("tempo_bpm"))] = tempo
        return np.ascontiguousarray(mean_vec, dtype=np.float32)

    def extract_matrix(self, preprocessed_or_samples: Any, sample_rate: int | None = None) -> np.ndarray:
        """extract() as a (1, n_features) matrix."""
        return self.extract(preprocessed_or_samples, sample_rate).reshape(1, -1)

    def extract_segments(self, preprocessed_or_samples: Any, sample_rate: int | None = None) -> np.ndarray:
        """One feature row per segment, in segment order."""
        y, sr, _ = _as_signal(preprocessed_or_samples, sample_rate)
        if y.size == 0:
            raise AudioRejected(UNREADABLE_FORMAT, "no samples to extract features from")
        conf = self.config
        from audio_preprocessing.transforms import segment_bounds

        bounds = segment_bounds(y.size, sr, float(conf["segment_duration_sec"]), mode="cover")
        rows = [self._vector(y[s:e], sr, conf, tempo_bpm=None) for s, e in bounds]
        tempo = estimate_tempo(y, sr, conf)
        idx = int(self.columns.index("tempo_bpm"))
        matrix = np.vstack(rows)
        matrix[:, idx] = tempo
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def extract_from_file(self, path: str | Path, **preprocess_kwargs: Any) -> np.ndarray:
        """File in, feature row out, with config-driven preprocessing."""
        from audio_preprocessing.pipeline import AudioPipeline, _import_contract

        _, PreprocessedAudio = _import_contract()
        pre = AudioPipeline().preprocess_file(path)
        if getattr(pre, "rejected", False):
            raise AudioRejected(UNREADABLE_FORMAT, pre.rejection_reason or "audio rejected", None)
        return self.extract(pre)

    def _vector(self, y: np.ndarray, sr: int, conf: dict[str, Any], *, tempo_bpm: float | None) -> np.ndarray:
        blocks = _frame_blocks(y, sr, conf)
        values: list[float] = []

        mfcc = blocks["mfcc"]
        for i in range(mfcc.shape[0]):
            stats = _statistics(mfcc[i])
            for stat in FRAME_STATS:
                values.append(stats[stat])
        if bool(conf.get("include_delta_mfcc", True)):
            d = blocks["delta_mfcc"]
            for i in range(d.shape[0]):
                stats = _statistics(d[i])
                for stat in FRAME_STATS:
                    values.append(stats[stat])
        chroma = blocks["chroma"]
        for i in range(chroma.shape[0]):
            stats = _statistics(chroma[i])
            for stat in FRAME_STATS:
                values.append(stats[stat])

        zcr = blocks["zcr"]
        values += [zcr.mean(), zcr.std(), float(np.percentile(zcr, 10)), float(np.percentile(zcr, 90))]
        rms_db = blocks["rms_db"]
        values += [
            float(rms_db.mean()), float(rms_db.std()), float(rms_db.max()),
            float(np.percentile(rms_db, 10)), float(np.percentile(rms_db, 90)),
        ]
        for key in ("centroid", "bandwidth", "rolloff", "flatness"):
            series = blocks[key]
            values += [float(series.mean()), float(series.std())]
        onset = blocks["onset"]
        duration = max(1e-6, y.size / float(sr))
        values += [float(onset.mean()), float(onset.max()), float(onset.size and blocks["n_onsets"]) / duration]
        # Crest factor (peak minus RMS, dB): about 3 dB for a square wave, ~20 dB for an impulse.
        # One of the few features separating a gunshot from a sustained alarm. On digital silence
        # both terms are -inf, so it is floored to 0.0 rather than letting a NaN reject the row.
        crest_db = float(peak_dbfs(y) - rms_dbfs(y))
        if not np.isfinite(crest_db):
            crest_db = 0.0
        values.append(crest_db)
        mel_mean = blocks["logmel"].mean(axis=1)
        values += [float(v) for v in mel_mean]
        values.append(float(tempo_bpm) if tempo_bpm is not None else estimate_tempo(y, sr, conf))

        vec = np.asarray(values, dtype=np.float32)
        if vec.size != len(self.columns):
            raise AssertionError(
                f"feature schema drift: produced {vec.size} values for {len(self.columns)} columns"
            )
        if not np.isfinite(vec).all():
            bad = [self.columns[i] for i in np.flatnonzero(~np.isfinite(vec))]
            raise AssertionError(f"non-finite feature values in columns: {bad[:10]}")
        return vec


# Frame-level computation -- one STFT, everything else derived

def _frame_blocks(y: np.ndarray, sr: int, conf: dict[str, Any]) -> dict[str, np.ndarray]:
    """All frame-wise series for one signal, from a single STFT."""
    import librosa

    arr = to_mono(y)
    n_fft = int(conf["n_fft"])
    hop = int(conf["hop_length"])
    n_mels = int(conf["n_mels"])
    n_mfcc = int(conf["n_mfcc"])
    fmin = float(conf["fmin"])
    fmax = float(conf["fmax"])

    if arr.size < n_fft:
        arr = np.pad(arr, (0, n_fft - arr.size))

    stft = librosa.stft(arr, n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann", center=True)
    mag = np.abs(stft)
    power = mag ** 2

    mel = librosa.feature.melspectrogram(
        S=power, sr=sr, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0,
    )
    # Absolute dB (ref=1.0, no top_db) keeps band energies comparable between recordings.
    logmel = librosa.power_to_db(mel, ref=1.0, top_db=None)

    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=n_mfcc)
    # librosa's delta uses a 9-frame window and raises on shorter input (a < 0.5 s window
    # would crash the pipeline). Use the largest odd width the frame count allows; below 2
    # frames the deltas are zero, which is the honest value.
    if bool(conf.get("include_delta_mfcc", True)):
        frames = int(mfcc.shape[1])
        width = min(9, frames if frames % 2 == 1 else max(0, frames - 1))
        delta = (
            librosa.feature.delta(mfcc, width=width)
            if width >= 3
            else np.zeros_like(mfcc, dtype=np.float32)
        )
    else:
        delta = np.zeros((1, mfcc.shape[1]), dtype=np.float32)

    # On a signal with no tonal content librosa warns and returns zero chroma, which is right.
    # The warning is silenced for this call only so real tuning warnings stay visible.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*empty frequency set.*")
        chroma = librosa.feature.chroma_stft(
            S=power, sr=sr, n_fft=n_fft, hop_length=hop, win_length=n_fft, n_chroma=int(conf["n_chroma"]),
        )
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(S=mag, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(
        S=mag, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=float(conf.get("rolloff_percent", 0.85))
    )
    flatness = librosa.feature.spectral_flatness(S=power, n_fft=n_fft, hop_length=hop)
    # Derived from y, not from S: exact frame RMS with the same framing as the STFT.
    rms = librosa.feature.rms(y=arr, frame_length=n_fft, hop_length=hop, center=True)
    zcr = librosa.feature.zero_crossing_rate(arr, frame_length=n_fft, hop_length=hop, center=True)

    onset_env = librosa.onset.onset_strength(S=logmel, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames")

    n_frames = mag.shape[1]
    return {
        "stft": stft,
        "mag": mag,
        "power": power,
        "mel": mel,
        "logmel": logmel,
        "mfcc": _fit(mfcc, n_frames),
        "delta_mfcc": _fit(delta, n_frames),
        "chroma": _fit(chroma, n_frames),
        "centroid": _flat(centroid),
        "bandwidth": _flat(bandwidth),
        "rolloff": _flat(rolloff),
        "flatness": _flat(flatness),
        "rms": _flat(rms),
        "rms_db": _to_db(_flat(rms)),
        "zcr": _flat(zcr),
        "onset": _flat(onset_env),
        "n_onsets": float(len(onsets)),
    }


def _fit(series: np.ndarray, n_frames: int) -> np.ndarray:
    """Make a (n_coeff, T) block exactly ``n_frames`` wide, so blocks can be stacked."""
    arr = np.atleast_2d(series)
    if arr.shape[1] == n_frames:
        return arr
    if arr.shape[1] > n_frames:
        return arr[:, :n_frames]
    return np.pad(arr, ((0, 0), (0, n_frames - arr.shape[1])), mode="edge")


def _flat(series: np.ndarray) -> np.ndarray:
    """Reduce a librosa feature to a 1-D frame series."""
    arr = np.asarray(series)
    return arr.reshape(-1) if arr.ndim == 1 else arr[0]


def _to_db(series: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Amplitude series to dBFS with a floor, so a silent frame is finite and comparable."""
    arr = np.maximum(np.asarray(series, dtype=np.float64), floor)
    return 20.0 * np.log10(arr)


def _statistics(series: np.ndarray) -> dict[str, float]:
    """Mean and standard deviation, over finite frames only."""
    arr = np.asarray(series, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(finite.mean()), "std": float(finite.std())}


def estimate_tempo(y: np.ndarray, sr: int, conf: dict[str, Any] | None = None) -> float:
    """Tempo in BPM from the onset envelope; 0.0 means no periodicity was found."""
    import librosa

    conf = conf or _fc()
    arr = to_mono(y)
    if arr.size < int(conf["n_fft"]) * 2:
        return 0.0
    onset_env = librosa.onset.onset_strength(y=arr, sr=sr, hop_length=int(conf["hop_length"]))
    if not np.any(onset_env > 0):
        return 0.0
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=int(conf["hop_length"]))
    value = float(np.atleast_1d(tempo)[0])
    return value if np.isfinite(value) else 0.0


# Log-mel tensor for the deep models and the Teachable Machine frontend

def segment_logmel(y: np.ndarray, sr: int, conf: dict[str, Any] | None = None) -> np.ndarray:
    """Log-mel (dB) of one segment, shaped exactly ``(n_mels, n_frames)`` from config."""
    import librosa
    from audio_preprocessing.transforms import pad_or_truncate

    conf = conf or _fc()
    n_mels, n_frames = mel_tensor_shape(conf)
    n_fft = int(conf["n_fft"])
    hop = int(conf["hop_length"])
    n_samples = int(round(float(conf["segment_duration_sec"]) * int(conf["sample_rate"])))

    arr = pad_or_truncate(to_mono(y), n_samples, mode="center")
    stft = librosa.stft(arr, n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann", center=True)
    mel = librosa.feature.melspectrogram(
        S=np.abs(stft) ** 2, sr=sr, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        n_mels=n_mels, fmin=float(conf["fmin"]), fmax=float(conf["fmax"]), power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=1.0, top_db=None)
    if logmel.shape[1] != n_frames:
        # Only reachable if the config's hop and rate are edited inconsistently; fixing the
        # shape here keeps a model's input contract intact rather than failing at predict time.
        if logmel.shape[1] > n_frames:
            logmel = logmel[:, :n_frames]
        else:
            logmel = np.pad(logmel, ((0, 0), (0, n_frames - logmel.shape[1])), mode="edge")
    return np.ascontiguousarray(logmel, dtype=np.float32)


def extract_mel_tensor(preprocessed_or_samples: Any, sample_rate: int | None = None, *,
                       segment_index: int | None = None,
                       feature_config: dict[str, Any] | None = None) -> np.ndarray:
    """Log-mel tensor (dB, not normalised) for one segment: (n_mels, n_frames), 128 x 94 at 3 s.

    ``segment_index=None`` takes the loudest segment (lowest index on ties), so a clip whose
    event starts two seconds in is not scored on room tone.
    """
    y, sr, _ = _as_signal(preprocessed_or_samples, sample_rate)
    if y.size == 0:
        raise AudioRejected(UNREADABLE_FORMAT, "no samples to build a mel tensor from")
    conf = _fc(feature_config)
    idx = _pick_segment(y, sr, conf, segment_index)
    s, e = idx
    return segment_logmel(y[s:e], sr, conf)


def extract_mel_segments(preprocessed_or_samples: Any, sample_rate: int | None = None, *,
                         feature_config: dict[str, Any] | None = None) -> np.ndarray:
    """Log-mel tensors for every segment; row i matches PreprocessedAudio.segments[i]."""
    y, sr, segments = _as_signal(preprocessed_or_samples, sample_rate)
    if y.size == 0:
        raise AudioRejected(UNREADABLE_FORMAT, "no samples to build mel tensors from")
    conf = _fc(feature_config)
    from audio_preprocessing.transforms import segment_bounds

    bounds = segment_bounds(y.size, sr, float(conf["segment_duration_sec"]), mode="cover")
    tensors = [segment_logmel(y[s:e], sr, conf) for s, e in bounds]
    return np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32)


def _pick_segment(y: np.ndarray, sr: int, conf: dict[str, Any], segment_index: int | None) -> tuple[int, int]:
    from audio_preprocessing.transforms import segment_bounds

    bounds = segment_bounds(y.size, sr, float(conf["segment_duration_sec"]), mode="cover")
    if not bounds:
        return 0, y.size
    if segment_index is not None:
        idx = int(np.clip(segment_index, 0, len(bounds) - 1))
        return bounds[idx]
    best_i, best_energy = 0, -1.0
    for i, (s, e) in enumerate(bounds):
        energy = float(np.mean(np.square(y[s:e], dtype=np.float64))) if e > s else 0.0
        if energy > best_energy:
            best_i, best_energy = i, energy
    return bounds[best_i]


# Data for the UI panels

def waveform_envelope(y: np.ndarray, sample_rate: int | None = None, n_points: int = 400) -> dict[str, list[float]]:
    """Min/max envelope for the waveform panel.

    Peak-preserving rather than averaged: averaging hides exactly the impulses (a gunshot, a
    glass break) the panel exists to show.
    """
    arr = to_mono(y)
    if arr.size == 0:
        return {"min": [], "max": [], "times": []}
    n_points = max(1, int(n_points))
    chunk = max(1, int(np.ceil(arr.size / n_points)))
    n_buckets = int(np.ceil(arr.size / chunk))
    padded = np.pad(arr, (0, n_buckets * chunk - arr.size))
    block = padded.reshape(n_buckets, chunk)
    scale = 1.0 / float(sample_rate) if sample_rate else 1.0
    return {
        "min": [float(v) for v in block.min(axis=1)],
        "max": [float(v) for v in block.max(axis=1)],
        "times": [float(i * chunk * scale) for i in range(n_buckets)],
    }


def spectrogram_db(y: np.ndarray, sr: int, *, max_frames: int = 600,
                   conf: dict[str, Any] | None = None) -> dict[str, Any]:
    """Log-mel spectrogram in dB for the spectrogram panel, downsampled if long."""
    conf = conf or _fc()
    n_fft = int(conf["n_fft"])
    hop = int(conf["hop_length"])
    arr = to_mono(y)
    if arr.size < n_fft:
        arr = np.pad(arr, (0, n_fft - arr.size))
    import librosa

    mel = librosa.feature.melspectrogram(
        y=arr, sr=sr, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        n_mels=int(conf["n_mels"]), fmin=float(conf["fmin"]), fmax=float(conf["fmax"]), power=2.0,
    )
    db = librosa.power_to_db(mel, ref=1.0, top_db=None)
    if db.shape[1] > max_frames:
        step = int(np.ceil(db.shape[1] / max_frames))
        db = db[:, ::step]
    return {
        "data": [[float(v) for v in row] for row in db],
        "n_mels": int(db.shape[0]),
        "n_frames": int(db.shape[1]),
        "hop_length": hop,
        "frame_seconds": hop / float(sr),
        "fmax": float(conf["fmax"]),
        "unit": "dBFS-relative (power_to_db ref=1.0)",
    }


# Helpers

def _as_signal(source: Any, sample_rate: int | None = None) -> tuple[np.ndarray, int, list[tuple[float, float]]]:
    """Accept a PreprocessedAudio, a bare array, or anything with ``.samples``.

    A scalar or empty array raises instead of becoming a one-sample signal: passing a count
    where samples belong used to return a plausible-looking 63-microsecond segment.
    """
    if sample_rate is not None:
        arr = np.asarray(source, dtype=np.float32)
        if arr.ndim == 0:
            raise AudioRejected(
                UNREADABLE_FORMAT,
                f"expected a 1-D or 2-D array of samples, got a scalar ({source!r}); "
                "a sample count is not a signal",
            )
        if arr.size == 0:
            raise AudioRejected(UNREADABLE_FORMAT, "the signal holds no samples")
        return to_mono(arr), int(sample_rate), []
    samples = getattr(source, "samples", None)
    if samples is not None:
        if getattr(source, "rejected", False):
            raise AudioRejected(UNREADABLE_FORMAT, getattr(source, "rejection_reason", None) or "audio was rejected")
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 0 or arr.size == 0:
            raise AudioRejected(UNREADABLE_FORMAT, "the preprocessed signal holds no samples")
        return to_mono(arr), int(source.sample_rate), list(getattr(source, "segments", []) or [])
    raise AudioRejected(
        UNREADABLE_FORMAT,
        f"expected a PreprocessedAudio or a bare array with sample_rate, got {type(source).__name__}",
    )


# Module-level convenience

def extract_features(preprocessed_or_samples: Any, sample_rate: int | None = None,
                     *, feature_config: dict[str, Any] | None = None) -> np.ndarray:
    return FeatureExtractor(feature_config=feature_config).extract(preprocessed_or_samples, sample_rate)


def extract_feature_matrix(preprocessed_or_samples: Any, sample_rate: int | None = None,
                           *, feature_config: dict[str, Any] | None = None) -> np.ndarray:
    return FeatureExtractor(feature_config=feature_config).extract_matrix(preprocessed_or_samples, sample_rate)


def extract_features_from_file(path: str | Path, **kwargs: Any) -> np.ndarray:
    return FeatureExtractor(**kwargs).extract_from_file(path)
