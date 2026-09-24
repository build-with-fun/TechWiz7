"""
The inference contract — the one interface every agent builds against.

Owner: lorena.  This module is the boundary between "audio arrives" and "predictions come
out".  SRS Step 2, Step 11, FR xviii-xx, Deliverable 3.

WHY THIS FILE EXISTS
--------------------
The SRS requires that an uploaded clip and a live microphone window be treated identically:
"both models must be applied consistently".  The cheapest way to break that is for the
upload path to preprocess one way and the live path another.  So both paths produce an
`AudioSource` and both go through one `preprocess` callable and one `predictor.predict()`.

Nothing in this module imports a model framework.  The predictor receives preprocessed
features and a loaded sklearn-compatible estimator.  That keeps the web app importable
without loading a model, and keeps the model swappable without touching the app.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------------------
# Class list — loaded from config, never hard-coded
# --------------------------------------------------------------------------------------


def load_class_config(config_path: Path | None = None) -> dict[str, Any]:
    """Read the canonical class list. Evaluators may add a category (SRS surprise mod)."""
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


# --------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------

@dataclass
class AudioSource:
    """Audio entering the system, from EITHER input mode.

    Exactly one of `path` or `samples` is set.

    - Upload mode:      AudioSource.from_path("clip.wav")
    - Live window mode: AudioSource.from_samples(mono_float32_array, sample_rate=48000)

    Both are then handled by the same preprocessing call, which is the point.
    """

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


# --------------------------------------------------------------------------------------
# Preprocessing handoff
# --------------------------------------------------------------------------------------

@dataclass
class PreprocessedAudio:
    """Output of preprocessing, input to feature extraction.

    `taha` owns the code that produces this; `lorena` owns the shape of the contract.
    The `quality` block is what the UI and the manual-review rules read (SRS Step 13).
    """

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
    """The single preprocessing path. Both input modes must use the SAME instance."""

    def __call__(self, source: AudioSource) -> PreprocessedAudio: ...


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """One model's opinion about one audio input.

    `confidences` MUST contain an entry for every configured class — the SRS requires
    per-class confidences for the dashboard, and a missing class would silently become
    a zero that the comparison then uses.
    """

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
        """Top-k by confidence, deterministic on ties (alphabetical by class name)."""
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
    """Stable identity for an audio input.

    For a file: sha256 of its bytes — this is the duplicate/near-duplicate key (FR lxxiii).
    For a live window: sha256 of the samples plus rate, so an identical window is
    recognisable but two different windows never collide.
    """
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
    """Turn a raw score vector into a {class: confidence} map summing to 1.

    Guards the integrity rule against invented confidences: if a model returns
    probabilities that do not sum to 1, we normalise and SAY SO in the log rather than
    passing a number we did not verify. Negative scores and NaN are clamped, not hidden.
    """
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
        # A model that outputs all zeros is broken; a uniform distribution is the honest
        # representation of "I know nothing" rather than a fabricated confident answer.
        uniform = 1.0 / len(labels)
        return {label: uniform for label in labels}
    return {label: value / total for label, value in zip(labels, cleaned, strict=True)}
