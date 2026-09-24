"""
The inference layer: the single contract both input modes and both models go through.

Owner: lorena.  Public surface other agents build against:

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
    """Lazy accessor — keeps tensorflow out of the import path of the web app.

    The app must start even if the GTM network cannot be loaded, and report that fact in
    the UI rather than refusing to boot.
    """
    from .gtm_predictor import GtmModelPredictor

    return GtmModelPredictor.load(*args, **kwargs)
