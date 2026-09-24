"""Audio pre-processing -- SRS Step 3, Step 4, Step 13.

Owner: taha (Audio DSP & feature engineering).

Public surface, in the order the pipeline uses it::

    from audio_preprocessing import (
        AudioPipeline,            # the Preprocessor lorena's contract expects
        load_audio,               # decode anything to float32 mono
        validate_file,            # SRS Step 3 file validation, never raises
        analyze_quality,          # SRS Step 13 verdict
        segment_timestamps,       # FR xv start/end timestamps
    )

Nothing in this package holds a literal for an audio parameter.  Sample rate, segment
duration, supported formats, quality thresholds and performance budgets all come from
``config/thresholds.json`` (lorena) and ``config/features.json`` via
:mod:`audio_preprocessing.config`, so an evaluator can change one during the demo and the
running application follows on the next call.
"""

from __future__ import annotations

from .config import (
    audio_config,
    clear_cache,
    config_dir_report,
    feature_config,
    ffmpeg_binary,
    performance_budget,
    quality_config,
    thresholds,
)
from .exceptions import (
    AudioDecodeError,
    AudioRejected,
    RejectionInfo,
    message_for,
)
from .io import (
    convert_format,
    detect_format,
    ffprobe_metadata,
    load_audio,
    load_audio_bytes,
    sha256_file,
    validate_file,
    validate_samples,
)
from .pipeline import (
    PREPROCESSING_VERSION,
    AudioPipeline,
    preprocessing_version,
    preprocess,
    preprocess_file,
    preprocess_samples,
    quality_meets,
    warm_up,
)
from .quality import (
    ACCEPTABLE,
    GOOD,
    POOR,
    UNUSABLE,
    analyze_quality,
    clipping_stats,
    estimate_snr_db,
    frame_dynamic_range_db,
    meets_min_quality,
    quality_summary,
)
from .transforms import (
    amplitude_to_db,
    apply_highpass,
    crop_or_pad_seconds,
    db_to_amplitude,
    ensure_min_amplitude,
    iter_segments,
    normalize_amplitude,
    pad_or_truncate,
    peak_dbfs,
    preemphasis,
    reduce_noise,
    resample,
    rms_dbfs,
    segment_bounds,
    segment_timestamps,
    silence_mask,
    to_mono,
    trim_silence,
)

__all__ = [
    # config
    "audio_config", "quality_config", "feature_config", "performance_budget", "thresholds",
    "clear_cache", "config_dir_report", "ffmpeg_binary",
    # exceptions
    "AudioRejected", "AudioDecodeError", "RejectionInfo", "message_for",
    # io
    "load_audio", "load_audio_bytes", "validate_file", "validate_samples",
    "detect_format", "convert_format", "sha256_file", "ffprobe_metadata",
    # transforms
    "to_mono", "resample", "normalize_amplitude", "trim_silence", "silence_mask",
    "reduce_noise", "apply_highpass", "preemphasis", "pad_or_truncate",
    "crop_or_pad_seconds", "segment_bounds", "segment_timestamps", "iter_segments",
    "peak_dbfs", "rms_dbfs", "db_to_amplitude", "amplitude_to_db", "ensure_min_amplitude",
    # quality
    "analyze_quality", "quality_summary", "meets_min_quality", "estimate_snr_db",
    "frame_dynamic_range_db", "clipping_stats", "GOOD", "ACCEPTABLE", "POOR", "UNUSABLE",
    # pipeline
    "AudioPipeline", "preprocess", "preprocess_file", "preprocess_samples",
    "preprocessing_version", "PREPROCESSING_VERSION", "quality_meets", "warm_up",
]

__version__ = PREPROCESSING_VERSION
