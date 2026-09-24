"""Configuration access for the audio pipeline.

Owner: taha.  SRS Step 4, Step 13, FR xi, xv, lxxii-xxxvii; integrity rule 5.

WHY THIS FILE EXISTS
--------------------
SRS 1.8 rule 5 says evaluators may demand a *changed segment duration* or a *new audio
format* live, during the demonstration.  If the segment duration were the literal ``3.0``
inside :mod:`audio_preprocessing.transforms`, satisfying that demand would mean editing
code in front of them.  So every audio parameter the pipeline uses is read from
``config/thresholds.json`` (lorena's file, already frozen) with the extractor-specific
parameters living in ``config/features.json`` (this module's file).

The cache is keyed on the file's *modification time*, not just its path, so a threshold
edited by the admin UI during a session takes effect on the next call without restarting
the application.  That is the behaviour the ``configurable threshold`` requirement needs.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"

THRESHOLDS_FILE = "thresholds.json"
FEATURES_FILE = "features.json"
CLASSES_FILE = "classes.json"

# --------------------------------------------------------------------------------------
# Fallbacks
# --------------------------------------------------------------------------------------
# These exist so the pipeline can still import and run if a config file is missing --
# NOT so that they can be used silently.  A missing config file raises in `require_config_file`;
# `audio_config()` merely degrades to these values and the caller can see `_source`.

AUDIO_DEFAULTS: dict[str, Any] = {
    "target_sample_rate": 16000,
    "target_channels": 1,
    "segment_duration_sec": 3.0,
    "live_window_sec": 2.0,
    "max_upload_mb": 50,
    "supported_formats": ["wav", "mp3", "flac", "ogg", "m4a"],
    "target_peak_dbfs": -3.0,
    "trim_silence": True,
    "silence_trim_top_db": 30.0,
    "noise_reduction": True,
    "noise_reduction_strength": 0.75,
    "preemphasis_coef": 0.0,
    "apply_highpass": True,
    "highpass_hz": 50.0,
    "max_duration_sec": 300.0,
    "min_duration_sec": 0.5,
}

QUALITY_DEFAULTS: dict[str, Any] = {
    "silence_rms_dbfs_max": -50.0,
    "clipping_ratio_max": 0.01,
    "min_duration_sec": 0.5,
    "max_duration_sec": 300.0,
    "min_snr_db": 5.0,
    "good_snr_db": 20.0,
    "acceptable_snr_db": 12.0,
    "poor_snr_db": 6.0,
    # Not in config/thresholds.json today; the default is stated here so the value is
    # visible and overridable rather than buried in the analyser.
    "low_signal_peak_dbfs": -40.0,
}

PERFORMANCE_DEFAULTS: dict[str, Any] = {
    "max_upload_seconds_budget": 8.0,
    "max_live_window_budget": 3.0,
    "clip_budget_seconds": 30.0,
}

FEATURE_DEFAULTS: dict[str, Any] = {
    "feature_version": "audiofeat-1.0.0",
    "sample_rate": 16000,
    "segment_duration_sec": 3.0,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 128,
    "n_mfcc": 20,
    "n_chroma": 12,
    "fmin": 0.0,
    "fmax": 8000.0,
    "rolloff_percent": 0.85,
    "include_delta_mfcc": True,
    "onset_top_db": 30.0,
}

# --------------------------------------------------------------------------------------
# mtime-keyed JSON cache
# --------------------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, int, dict[str, Any]]] = {}


def _load_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON config file, cached on (mtime_ns, size).

    Returns ``None`` when the file does not exist so callers can distinguish "absent"
    from "empty".  A malformed file raises -- a broken config is a real failure and
    must not be silently replaced by defaults.
    """
    if not path.exists():
        return None
    stat = path.stat()
    key = str(path)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == stat.st_mtime_ns and hit[1] == stat.st_size:
        return copy.deepcopy(hit[2])

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    _CACHE[key] = (stat.st_mtime_ns, stat.st_size, copy.deepcopy(data))
    return copy.deepcopy(data)


def clear_cache() -> None:
    """Drop the config cache.  Used by tests that rewrite a config file in place."""
    _CACHE.clear()


def require_config_file(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Read a config file that MUST exist.

    The web app and the training scripts call this at startup so a missing config is a
    loud error rather than a run with invented defaults.
    """
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    data = _load_json(directory / name)
    if data is None:
        raise FileNotFoundError(
            f"required configuration file missing: {directory / name}. "
            f"The pipeline refuses to run on invented defaults."
        )
    return data


def _merge(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# --------------------------------------------------------------------------------------
# Public accessors
# --------------------------------------------------------------------------------------

def thresholds(config_dir: Path | None = None) -> dict[str, Any]:
    """The full frozen thresholds document (config/thresholds.json)."""
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    data = _load_json(directory / THRESHOLDS_FILE)
    return data if data is not None else {}


def audio_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Audio parameters: sample rate, segment duration, formats, normalisation targets."""
    doc = thresholds(config_dir)
    override = doc.get("audio", {}) if isinstance(doc.get("audio"), dict) else {}
    merged = _merge(AUDIO_DEFAULTS, override)
    source = (Path(config_dir) if config_dir is not None else CONFIG_DIR) / THRESHOLDS_FILE
    merged["_source"] = str(source)
    merged["_config_version"] = doc.get("version", "unknown")
    return merged


def quality_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Audio-quality thresholds used by the Good/Acceptable/Poor/Unusable verdict."""
    doc = thresholds(config_dir)
    override = doc.get("audio_quality", {}) if isinstance(doc.get("audio_quality"), dict) else {}
    merged = _merge(QUALITY_DEFAULTS, override)
    audio = audio_config(config_dir)
    # Duration limits are stated in both blocks; the audio block wins if it is explicit.
    for key in ("min_duration_sec", "max_duration_sec"):
        if key in audio:
            merged.setdefault(key, audio[key])
        merged[key] = override.get(key, audio.get(key, merged[key]))
    merged["_source"] = audio.get("_source", str(CONFIG_DIR / THRESHOLDS_FILE))
    return merged


def performance_budget(config_dir: Path | None = None) -> dict[str, Any]:
    """The SRS performance budgets (30 s clip <= 8 s, live window <= 3 s)."""
    doc = thresholds(config_dir)
    override = doc.get("performance", {}) if isinstance(doc.get("performance"), dict) else {}
    merged = _merge(PERFORMANCE_DEFAULTS, override)
    merged["_source"] = str(
        (Path(config_dir) if config_dir is not None else CONFIG_DIR) / THRESHOLDS_FILE
    )
    return merged


def feature_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Extractor parameters.  ``config/features.json`` is owned by feature_extraction."""
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    data = _load_json(directory / FEATURES_FILE)
    if data is None:
        # Absent file: fall back to the same numbers the thresholds file states, so the
        # extractor and the segmenter can never disagree about segment length.
        audio = audio_config(config_dir)
        merged = copy.deepcopy(FEATURE_DEFAULTS)
        merged["sample_rate"] = audio["target_sample_rate"]
        merged["segment_duration_sec"] = audio["segment_duration_sec"]
        merged["_audio_block"] = {
            "target_sample_rate": audio["target_sample_rate"],
            "segment_duration_sec": audio["segment_duration_sec"],
        }
        merged["_source"] = "built-in defaults (config/features.json absent)"
        return merged
    merged = _merge(FEATURE_DEFAULTS, data)
    merged["_source"] = str(directory / FEATURES_FILE)
    # The audio block is the single source of truth for rate and segment length.  features.json
    # also carries these two keys, so overwrite them here as well: a caller that reads the
    # returned dict directly must not see a segment length that contradicts the audio block.
    audio = audio_config(config_dir)
    merged["_audio_block"] = {
        "target_sample_rate": audio["target_sample_rate"],
        "segment_duration_sec": audio["segment_duration_sec"],
    }
    merged["sample_rate"] = int(audio["target_sample_rate"])
    merged["segment_duration_sec"] = float(audio["segment_duration_sec"])
    return merged


def ffmpeg_binary() -> str:
    """Path to ffmpeg, overridable for a machine where it is not on PATH."""
    return os.environ.get("SONICSENTINEL_FFMPEG", "ffmpeg")


def config_dir_report(config_dir: Path | None = None) -> dict[str, Any]:
    """Which config files this process is actually reading, and their versions.

    Printed by the diagnostics page and pasted into the report's appendix, so an evaluator
    can see the running application is reading the file they just edited rather than a
    bundled copy.
    """
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    thresholds_doc = thresholds(config_dir) or {}
    features_doc = _load_json(directory / FEATURES_FILE) or {}
    classes_doc = _load_json(directory / CLASSES_FILE) or {}
    classes_list = classes_doc.get("classes", []) if isinstance(classes_doc, dict) else []

    def _file_report(name: str, doc: dict[str, Any], version_key: str, fallback: str) -> dict[str, Any]:
        path = directory / name
        return {
            "path": str(path),
            "exists": path.exists(),
            "version": doc.get(version_key, fallback) if isinstance(doc, dict) else fallback,
        }

    files = {
        THRESHOLDS_FILE: _file_report(THRESHOLDS_FILE, thresholds_doc, "version", "unknown"),
        FEATURES_FILE: _file_report(FEATURES_FILE, features_doc, "feature_version", FEATURE_DEFAULTS["feature_version"]),
        CLASSES_FILE: _file_report(CLASSES_FILE, classes_doc, "version", "unknown"),
    }
    return {
        "directory": str(directory),
        "config_dir": str(directory),
        "files": files,
        "classes": len(classes_list),
        "class_names": list(classes_list),
        "thresholds_file": files[THRESHOLDS_FILE]["path"],
        "thresholds_present": files[THRESHOLDS_FILE]["exists"],
        "thresholds_version": files[THRESHOLDS_FILE]["version"],
        "features_file": files[FEATURES_FILE]["path"],
        "features_present": files[FEATURES_FILE]["exists"],
        "feature_version": files[FEATURES_FILE]["version"],
        "audio": {
            k: v for k, v in audio_config(config_dir).items() if not k.startswith("_")
        },
    }


def ffprobe_binary() -> str:
    """Path to ffprobe, overridable for the same reason."""
    return os.environ.get("SONICSENTINEL_FFPROBE", "ffprobe")
