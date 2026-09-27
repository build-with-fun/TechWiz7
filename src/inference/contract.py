"""Data types shared by both input modes and both models.

Uploads and live windows both become an AudioSource and go through the same preprocessing
and predictors. No ML framework is imported here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

REPO_ROOT = Path(__file__).resolve().parents[2]

# Class list, loaded from config


def load_class_config(config_path: Path | None = None) -> dict[str, Any]:
    """Read the class list from config/classes.json."""
    path = config_path or REPO_ROOT / "config" / "classes.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def class_names(config: Mapping[str, Any] | None = None) -> list[str]:
    cfg = config if config is not None else load_class_config()
    return [c["name"] for c in cfg["classes"]]


def critical_classes(config: Mapping[str, Any] | None = None) -> set[str]:
    cfg = config if config is not None else load_class_config()
    return set(cfg.get("critical_classes", []))


def load_thresholds(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or REPO_ROOT / "config" / "thresholds.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# Input

@dataclass
class AudioSource:
    """Input audio: either ``path`` (upload) or ``samples`` (live window)."""

    path: Path | None = None
    samples: Any | None = None            # numpy float32 mono, nominally -1..1
    sample_rate: int | None = None
    origin: str = "upload"                # "upload" | "live"

    def __post_init__(self) -> None:
        if (self.path is None) == (self.samples is None):
            raise ValueError("AudioSource needs exactly one of path or samples")
        if self.path is not None:
            self.path = Path(self.path)
        if self.samples is not None and self.sample_rate is None:
            raise ValueError("in-memory samples require a sample_rate")

    @classmethod
    def from_path(cls, path: str | Path, origin: str = "upload") -> "AudioSource":
        return cls(path=Path(path), origin=origin)

    @classmethod
    def from_samples(cls, samples: Any, sample_rate: int, origin: str = "live") -> "AudioSource":
        return cls(samples=samples, sample_rate=sample_rate, origin=origin)

    @property
    def is_live(self) -> bool:
        return self.origin == "live"


# Preprocessing handoff

@dataclass
class PreprocessedAudio:
    """Preprocessed audio, given to both models."""

    samples: Any                                     # numpy float32 mono, target sample rate
    sample_rate: int
    duration_sec: float
    source_path: str | None = None
    segments: list[tuple[float, float]] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)   # verdict, rms_dbfs, snr_db, ...
    preprocessing: dict[str, Any] = field(default_factory=dict)  # what was applied
    rejected: bool = False
    rejection_reason: str | None = None


@runtime_checkable
class Preprocessor(Protocol):
    """Preprocessing used by both input modes (share one instance)."""

    def __call__(self, source: AudioSource) -> PreprocessedAudio: ...


# Output

@dataclass
class PredictionResult:
    """One model's prediction. ``confidences`` must include every class."""

    model_name: str
    model_version: str
    predicted_class: str
    confidence: float
    confidences: dict[str, float]
    latency_sec: float = 0.0
    feature_version: str = ""
    source_origin: str = "upload"
    extra: dict[str, Any] = field(default_factory=dict)

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """Top k by confidence, ties broken alphabetically."""
        ranked = sorted(self.confidences.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "confidences": {k: round(v, 6) for k, v in self.confidences.items()},
            "top3": [{"class": c, "confidence": round(v, 6)} for c, v in self.top_k(3)],
            "latency_sec": round(self.latency_sec, 4),
            "feature_version": self.feature_version,
            "source_origin": self.source_origin,
            **({"extra": self.extra} if self.extra else {}),
        }


class ModelLoadError(RuntimeError):
    """Raised when a saved model or its sidecar artifacts cannot be loaded."""


def audio_fingerprint(source: AudioSource) -> str:
    """SHA-256 of a file, or of a live window's samples and rate."""
    if source.path is not None:
        digest = hashlib.sha256()
        with source.path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    import numpy as np

    arr = np.ascontiguousarray(np.asarray(source.samples, dtype="float32"))
    digest = hashlib.sha256()
    digest.update(str(source.sample_rate).encode())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def normalise_confidences(raw: Sequence[float], labels: Sequence[str]) -> dict[str, float]:
    """Raw scores -> {class: confidence} summing to 1. NaN and negatives become 0."""
    import math

    if len(raw) != len(labels):
        raise ValueError(f"got {len(raw)} scores for {len(labels)} labels")

    cleaned: list[float] = []
    for value in raw:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            v = 0.0
        cleaned.append(max(v, 0.0))

    total = sum(cleaned)
    if total <= 0.0:
        # All zeros: fall back to uniform.
        uniform = 1.0 / len(labels)
        return {label: uniform for label in labels}
    return {label: value / total for label, value in zip(labels, cleaned, strict=True)}
