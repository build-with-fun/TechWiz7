"""Acoustic feature extraction (SRS Step 6, FR xx).

Bump ``FEATURE_SCHEMA_VERSION`` whenever a column's name, order or meaning changes.
"""

from __future__ import annotations

from .cache import (
    MEL_INDEX_COLUMNS,
    MelCache,
    MelCacheEntry,
    MelNormalizer,
    audio_id_for,
    enumerate_segments,
)
from .features import (
    FEATURE_SCHEMA_VERSION,
    FRAME_STATS,
    FeatureExtractor,
    estimate_tempo,
    extract_feature_matrix,
    extract_features,
    extract_features_from_file,
    extract_mel_segments,
    extract_mel_tensor,
    feature_columns,
    mel_tensor_shape,
    n_features,
    segment_logmel,
    spectrogram_db,
    waveform_envelope,
)

__all__ = [
    "FeatureExtractor",
    "feature_columns",
    "n_features",
    "FEATURE_SCHEMA_VERSION",
    "FRAME_STATS",
    "extract_features",
    "extract_feature_matrix",
    "extract_features_from_file",
    "extract_mel_tensor",
    "extract_mel_segments",
    "segment_logmel",
    "mel_tensor_shape",
    "estimate_tempo",
    "waveform_envelope",
    "spectrogram_db",
    # segments, cache, normalisation
    "enumerate_segments",
    "audio_id_for",
    "MelCache",
    "MelCacheEntry",
    "MelNormalizer",
    "MEL_INDEX_COLUMNS",
]

__version__ = FEATURE_SCHEMA_VERSION
