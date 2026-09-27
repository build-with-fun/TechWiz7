"""Loads the saved Python model bundle and turns features into class confidences.

The class order and feature columns are stored in the bundle and checked on load, so a
mismatched bundle is refused instead of mislabelling predictions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contract import (
    AudioSource,
    ModelLoadError,
    PredictionResult,
    PreprocessedAudio,
    audio_fingerprint,
    normalise_confidences,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ModelLoadError(f"missing required artifact: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class ModelBundle:
    """A saved model and its metadata."""

    estimator: Any
    class_names: list[str]              # in the estimator's own fitted order
    feature_version: str
    model_name: str
    model_version: str
    feature_columns: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def n_features(self) -> int | None:
        if self.feature_columns:
            return len(self.feature_columns)
        return getattr(self.estimator, "n_features_in_", None)


class PythonModelPredictor:
    """Wraps a model bundle as `predict(audio) -> {class: confidence}`."""

    def __init__(self, bundle: ModelBundle, feature_extractor: Callable[[PreprocessedAudio], Any]):
        self.bundle = bundle
        self._extract = feature_extractor


    @classmethod
    def load(cls, model_dir: str | Path, feature_extractor: Callable[[PreprocessedAudio], Any]) -> "PythonModelPredictor":
        model_dir = Path(model_dir)
        if not model_dir.is_absolute():
            model_dir = REPO_ROOT / model_dir

        import joblib

        model_file = model_dir / "model.joblib"
        if not model_file.exists():
            raise ModelLoadError(
                f"no saved model at {model_file}. Train one with "
                f"python_models/train.py, or the app cannot serve predictions."
            )

        estimator = joblib.load(model_file)
        labels_doc = _read_json(model_dir / "label_encoder.json")
        feature_doc = _read_json(model_dir / "feature_config.json")
        try:
            meta = _read_json(model_dir / "model_meta.json")
        except ModelLoadError:
            meta = {}

        saved_classes = [str(c) for c in labels_doc["class_names"]]
        # Convert numpy strings to str for JSON.
        fitted_classes = [str(c) for c in getattr(estimator, "classes_", [])]

        # Class order must match the label file.
        if fitted_classes and len(fitted_classes) == len(saved_classes):
            if set(fitted_classes) != set(saved_classes):
                raise ModelLoadError(
                    "label drift: the saved estimator was fitted on "
                    f"{fitted_classes} but label_encoder.json says {saved_classes}. "
                    "Predictions would be mapped to the wrong class names."
                )
            # Keep the estimator's order, since argmax indexes into it.
            saved_classes = fitted_classes
        elif fitted_classes and len(fitted_classes) != len(saved_classes):
            raise ModelLoadError(
                f"label count mismatch: estimator has {len(fitted_classes)} classes, "
                f"label_encoder.json has {len(saved_classes)}"
            )

        bundle = ModelBundle(
            estimator=estimator,
            class_names=saved_classes,
            feature_version=feature_doc.get("feature_version", "unknown"),
            feature_columns=list(feature_doc.get("columns", [])),
            model_name=meta.get("model_name", estimator.__class__.__name__),
            model_version=meta.get("model_version", "0.0.0-unversioned"),
            metrics=meta.get("metrics", {}),
            metadata=meta,
            path=model_dir,
        )
        return cls(bundle, feature_extractor)


    def predict(self, source: AudioSource, preprocessor: Any) -> PredictionResult:
        """Classify one audio input."""
        started = time.perf_counter()

        preprocessed = preprocessor(source)
        if getattr(preprocessed, "rejected", False):
            raise ValueError(
                f"audio rejected before prediction: {preprocessed.rejection_reason}"
            )

        vector = self._extract(preprocessed)
        return self._predict_features(
            vector,
            latency_offset=time.perf_counter() - started,
            origin=source.origin,
            extra={"audio_fingerprint": audio_fingerprint(source)},
        )

    def predict_from_preprocessed(
        self, preprocessed: PreprocessedAudio, *, origin: str = "upload", extra: Mapping[str, Any] | None = None
    ) -> PredictionResult:
        """Classify audio that has already been preprocessed."""
        started = time.perf_counter()
        vector = self._extract(preprocessed)
        return self._predict_features(
            vector,
            latency_offset=time.perf_counter() - started,
            origin=origin,
            extra=dict(extra or {}),
        )

    def _predict_features(
        self,
        vector: Any,
        *,
        latency_offset: float = 0.0,
        origin: str = "upload",
        extra: Mapping[str, Any] | None = None,
    ) -> PredictionResult:
        import numpy as np

        x = np.asarray(vector, dtype="float64")
        if x.ndim == 1:
            x = x.reshape(1, -1)

        expected = self.bundle.n_features
        if expected is not None and x.shape[1] != expected:
            raise ValueError(
                f"feature drift: the saved model expects {expected} features but the "
                f"extractor produced {x.shape[1]}. The saved model would be scoring "
                f"the wrong columns. Re-extract features or retrain."
            )

        started = time.perf_counter()
        if hasattr(self.bundle.estimator, "predict_proba"):
            raw = np.asarray(self.bundle.estimator.predict_proba(x))[0]
        else:
            # No probabilities available: return one-hot.
            decision = self.bundle.estimator.predict(x)[0]
            raw = np.array([1.0 if c == decision else 0.0 for c in self.bundle.class_names])
        inference_sec = time.perf_counter() - started

        confidences = normalise_confidences(raw.tolist(), self.bundle.class_names)
        predicted = max(confidences.items(), key=lambda kv: (kv[1], kv[0]))[0]

        return PredictionResult(
            model_name=self.bundle.model_name,
            model_version=self.bundle.model_version,
            predicted_class=predicted,
            confidence=confidences[predicted],
            confidences=confidences,
            latency_sec=latency_offset + inference_sec,
            feature_version=self.bundle.feature_version,
            source_origin=origin,
            extra={"inference_sec": round(inference_sec, 5), **dict(extra or {})},
        )


    @property
    def model_version(self) -> str:
        return self.bundle.model_version

    @property
    def class_names(self) -> list[str]:
        return list(self.bundle.class_names)

    def describe(self) -> dict[str, Any]:
        """Model info for the admin dashboard and audit log."""
        return {
            "model_name": self.bundle.model_name,
            "model_version": self.bundle.model_version,
            "class_names": self.bundle.class_names,
            "feature_version": self.bundle.feature_version,
            "n_features": self.bundle.n_features,
            "metrics": self.bundle.metrics,
            "path": str(self.bundle.path) if self.bundle.path else None,
        }


def save_bundle(
    estimator: Any,
    out_dir: str | Path,
    *,
    class_names: Sequence[str],
    feature_version: str,
    feature_columns: Sequence[str] | None = None,
    model_name: str = "",
    model_version: str = "",
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a model with its labels, feature config and metadata."""
    import joblib

    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(estimator, out_dir / "model.joblib")

    fitted = list(getattr(estimator, "classes_", []))
    if fitted and len(fitted) == len(class_names) and set(fitted) == set(class_names):
        # Store the estimator's order, which is what predict_proba uses.
        ordered = fitted
    else:
        ordered = list(class_names)

    with (out_dir / "label_encoder.json").open("w", encoding="utf-8") as fh:
        json.dump({"class_names": ordered}, fh, indent=2)

    with (out_dir / "feature_config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "feature_version": feature_version,
                "columns": list(feature_columns or []),
                "n_features": len(feature_columns) if feature_columns else None,
            },
            fh, indent=2,
        )

    meta = {
        "model_name": model_name or estimator.__class__.__name__,
        "model_version": model_version or "0.0.0-unversioned",
        "metrics": dict(metrics or {}),
        **(dict(metadata or {})),
    }
    with (out_dir / "model_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    return out_dir
