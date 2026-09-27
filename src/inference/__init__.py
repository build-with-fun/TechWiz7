"""Inference layer shared by both input modes and both models.

Usage:

    from src.inference import (
        AudioSource, PreprocessedAudio, PredictionResult,
        PythonModelPredictor, GtmModelPredictor,
        classify_consistency, requires_manual_review,
    )
"""

from .consistency import (
    ACCEPTABLE_MATCH,
    CONSISTENCY_STATUSES,
    MODEL_DISAGREEMENT,
    STRONG_MATCH,
    UNCERTAIN_RESULT,
    WEAK_MATCH,
    ComparisonResult,
    classify_consistency,
    requires_manual_review,
)
from .contract import (
    AudioSource,
    ModelLoadError,
    PredictionResult,
    PreprocessedAudio,
    Preprocessor,
    audio_fingerprint,
    class_names,
    critical_classes,
    load_class_config,
    load_thresholds,
    normalise_confidences,
)
from .predictor import ModelBundle, PythonModelPredictor, save_bundle

__all__ = [
    "ACCEPTABLE_MATCH", "CONSISTENCY_STATUSES", "MODEL_DISAGREEMENT", "STRONG_MATCH",
    "UNCERTAIN_RESULT", "WEAK_MATCH", "ComparisonResult", "classify_consistency",
    "requires_manual_review", "AudioSource", "ModelLoadError", "PredictionResult",
    "PreprocessedAudio", "Preprocessor", "audio_fingerprint", "class_names",
    "critical_classes", "load_class_config", "load_thresholds", "normalise_confidences",
    "ModelBundle", "PythonModelPredictor", "save_bundle",
]


def load_gtm_predictor(*args, **kwargs):
    """Lazy import so the web app doesn't load TensorFlow unless needed."""
    from .gtm_predictor import GtmModelPredictor

    return GtmModelPredictor.load(*args, **kwargs)
