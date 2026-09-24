"""
Google Teachable Machine audio model adapter — the INDEPENDENT second model.

Owner: lorena.  SRS Step 9, Step 10, Step 11, Deliverable 3 and 4.

INDEPENDENCE IS THE WHOLE POINT
-------------------------------
The SRS requires two separately trained models and requires that the Python model's
prediction or confidence is never an input to the GTM model. Read this module's public
surface: `predict()` accepts an AudioSource and a preprocessor. There is no parameter
anywhere on this path that accepts a class label, a confidence, a top-k list, or a
PredictionResult. That is not an accident and it must stay that way — it is the structural
guarantee that the independence claim is true rather than asserted.
`tests/test_model_independence.py` proves it by inspection.

WHY THE FRONTEND IS CONFIGURED, NOT GUESSED
-------------------------------------------
GTM's exported TF.js model is the CNN only; the spectrogram that feeds it is computed by
GTM's own JavaScript frontend outside the model. To run that model server-side we must
reproduce that frontend exactly. A single wrong parameter (mel bands, frame count, sample
rate, normalisation) yields a model that still runs, still returns confident numbers, and
is completely wrong — the worst kind of failure.

So every frontend parameter lives in `gtm_model/frontend_config.json`, captured from the
real export, and `verify_frontend.py` checks our server-side predictions against
predictions recorded from the browser on the same clips. Until that agreement check has
run, the server-side GTM path reports itself as UNVERIFIED.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import (
    AudioSource,
    ModelLoadError,
    PredictionResult,
    PreprocessedAudio,
    normalise_confidences,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GTM_DIR = REPO_ROOT / "gtm_model"


@dataclass
class GtmFrontendConfig:
    """The audio frontend GTM applies before its CNN. Captured from the real export."""

    sample_rate: int
    window_sec: float
    n_mels: int
    n_frames: int
    fmin: float = 0.0
    fmax: float | None = None
    normalize_mode: str = "none"          # "none" | "div100" | "log" | "max"
    model_input_shape: list[int] | None = None
    frontend_id: str = "unverified"
    source: str = "unknown"

    @classmethod
    def load(cls, path: Path) -> "GtmFrontendConfig":
        if not path.exists():
            raise ModelLoadError(
                f"GTM frontend config missing: {path}\n"
                "This file records the spectrogram parameters GTM's exported model was "
                "trained with. Without it, server-side GTM inference would run with "
                "guessed parameters and return confident nonsense. Capture it from the "
                "GTM export (documentation/research/gtm_audio_export_internals.md "
                "explains where) before enabling the server-side GTM path."
            )
        with path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls(
            sample_rate=int(raw["sample_rate"]),
            window_sec=float(raw["window_sec"]),
            n_mels=int(raw["n_mels"]),
            n_frames=int(raw["n_frames"]),
            fmin=float(raw.get("fmin", 0.0)),
            fmax=raw.get("fmax"),
            normalize_mode=str(raw.get("normalize_mode", "none")),
            model_input_shape=raw.get("model_input_shape"),
            frontend_id=str(raw.get("frontend_id", "unverified")),
            source=str(raw.get("source", "unknown")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "window_sec": self.window_sec,
            "n_mels": self.n_mels,
            "n_frames": self.n_frames,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "normalize_mode": self.normalize_mode,
            "model_input_shape": self.model_input_shape,
            "frontend_id": self.frontend_id,
            "source": self.source,
        }


def compute_spectrogram(samples: Any, config: GtmFrontendConfig) -> Any:
    """Recompute GTM's input spectrogram from a mono waveform.

    Implemented with numpy + librosa so it is testable and inspectable. The parameters are
    the config's, never literals, so a corrected frontend means editing JSON, not code.
    """
    import librosa
    import numpy as np

    y = np.asarray(samples, dtype="float32").ravel()

    # GTM analyses a fixed window; pad short input, truncate long input.
    n_expected = int(round(config.window_sec * config.sample_rate))
    if y.size < n_expected:
        y = np.pad(y, (0, n_expected - y.size))
    elif y.size > n_expected:
        y = y[:n_expected]

    # Power spectrogram on the mel scale, matching GTM's frontend shape.
    hop = max(1, n_expected // config.n_frames)
    mel = librosa.feature.melspectrogram(
        y=y.astype("float64"),
        sr=config.sample_rate,
        n_fft=1024,
        hop_length=hop,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
    )

    # Fix the frame count to the exact value the model expects.
    if mel.shape[1] < config.n_frames:
        mel = np.pad(mel, ((0, 0), (0, config.n_frames - mel.shape[1])))
    else:
        mel = mel[:, : config.n_frames]

    mel = np.abs(mel)
    if config.normalize_mode == "div100":
        mel = mel / 100.0
    elif config.normalize_mode == "log":
        mel = np.log(mel + 1e-6)
    elif config.normalize_mode == "max":
        peak = float(mel.max()) or 1.0
        mel = mel / peak

    return mel.astype("float32")


class GtmModelPredictor:
    """Serves the exported Teachable Machine audio model.

    Backends: the converted Keras model if it can be loaded. Loading failure is loud — a
    silent fallback to a dummy would fabricate the second model's opinion, which is an
    explicit disqualification in the SRS integrity rules.
    """

    def __init__(
        self,
        model: Any,
        frontend: GtmFrontendConfig,
        class_names: list[str],
        model_version: str,
        *,
        backend: str,
        model_dir: Path | None = None,
        verified: bool = False,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.frontend = frontend
        self.class_names = list(class_names)
        self.model_version = model_version
        self.backend = backend
        self.model_dir = model_dir
        self.verified = verified
        self.metrics = metrics or {}

    # -- loading ---------------------------------------------------------------------

    @classmethod
    def load(cls, gtm_dir: str | Path = DEFAULT_GTM_DIR) -> "GtmModelPredictor":
        gtm_dir = Path(gtm_dir)
        if not gtm_dir.is_absolute():
            gtm_dir = REPO_ROOT / gtm_dir

        metadata_path = gtm_dir / "metadata.json"
        if not metadata_path.exists():
            raise ModelLoadError(
                f"GTM metadata.json not found in {gtm_dir}. The exported model from the "
                "Teachable Machine audio project must be placed here "
                "(metadata.json, model.json, weights.bin) before the GTM path can run."
            )
        with metadata_path.open(encoding="utf-8") as fh:
            metadata = json.load(fh)

        labels = metadata.get("wordLabels") or metadata.get("labels")
        if not labels:
            raise ModelLoadError(
                f"{metadata_path} has no 'wordLabels'. It does not look like a Teachable "
                "Machine export."
            )

        frontend = GtmFrontendConfig.load(gtm_dir / "frontend_config.json")

        model, backend = cls._load_backend(gtm_dir)

        verification_path = gtm_dir / "frontend_verification.json"
        verified = False
        if verification_path.exists():
            with verification_path.open(encoding="utf-8") as fh:
                verified = bool(json.load(fh).get("passed", False))

        metrics = {}
        metrics_path = gtm_dir / "gtm_metrics.json"
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as fh:
                metrics = json.load(fh)

        version = str(metadata.get("modelVersion")
                      or metadata.get("version")
                      or f"gtm-{frontend.frontend_id}")

        return cls(model, frontend, list(labels), version, backend=backend,
                   model_dir=gtm_dir, verified=verified, metrics=metrics)

    @staticmethod
    def _load_backend(gtm_dir: Path) -> tuple[Any, str]:
        """Load the exported network. Tries the converted Keras model first."""
        keras_path = gtm_dir / "gtm_model.h5"
        if keras_path.exists():
            try:
                import tensorflow as tf

                return tf.keras.models.load_model(str(keras_path), compile=False), "keras-h5"
            except Exception as exc:  # noqa: BLE001 - surface the real reason
                raise ModelLoadError(f"failed to load {keras_path}: {exc}") from exc

        saved_model_dir = gtm_dir / "gtm_saved_model"
        if saved_model_dir.exists():
            try:
                import tensorflow as tf

                return tf.saved_model.load(str(saved_model_dir)), "tf-saved-model"
            except Exception as exc:  # noqa: BLE001
                raise ModelLoadError(f"failed to load {saved_model_dir}: {exc}") from exc

        raise ModelLoadError(
            f"no loadable GTM network in {gtm_dir}. Convert the exported TF.js model:\n"
            f"  tensorflowjs_converter --input_format tfjs_layers_model "
            f"--output_format keras {gtm_dir}/model.json {gtm_dir}/gtm_model.h5\n"
            "See documentation/research/gtm_audio_export_internals.md for the exact "
            "procedure and its known failure modes."
        )

    # -- prediction ------------------------------------------------------------------

    def predict(self, source: AudioSource, preprocessor: Any) -> PredictionResult:
        """Classify audio with the GTM model. Audio in, confidences out — nothing else.

        Note the signature: there is deliberately no way to pass the Python model's
        opinion in here.
        """
        import time

        started = time.perf_counter()
        preprocessed = preprocessor(source)
        if getattr(preprocessed, "rejected", False):
            raise ValueError(f"audio rejected before prediction: {preprocessed.rejection_reason}")
        return self.predict_from_preprocessed(preprocessed, origin=source.origin,
                                              latency_offset=time.perf_counter() - started)

    def predict_from_preprocessed(
        self, preprocessed: PreprocessedAudio, *, origin: str = "upload",
        latency_offset: float = 0.0,
    ) -> PredictionResult:
        import numpy as np
        import time

        window = self._select_window(preprocessed)
        spectrogram = compute_spectrogram(window, self.frontend)

        model_input = spectrogram
        shape = self.frontend.model_input_shape
        if shape:
            model_input = spectrogram.reshape(shape) if len(shape) == 3 else spectrogram[None, ...]
        else:
            model_input = spectrogram[None, ..., None]

        started = time.perf_counter()
        output = self.model.predict(np.asarray(model_input, dtype="float32"), verbose=0)
        inference_sec = time.perf_counter() - started

        if isinstance(output, (list, tuple)):
            output = output[0]
        raw = np.asarray(output).ravel()

        if raw.size != len(self.class_names):
            raise ModelLoadError(
                f"GTM model returned {raw.size} scores for {len(self.class_names)} labels "
                f"({self.class_names}); the export and metadata.json disagree."
            )

        confidences = normalise_confidences(raw.tolist(), self.class_names)
        predicted = max(confidences.items(), key=lambda kv: (kv[1], kv[0]))[0]

        return PredictionResult(
            model_name="Google Teachable Machine",
            model_version=self.model_version,
            predicted_class=predicted,
            confidence=confidences[predicted],
            confidences=confidences,
            latency_sec=latency_offset + inference_sec,
            feature_version=f"gtm-frontend/{self.frontend.frontend_id}",
            source_origin=origin,
            extra={
                "backend": self.backend,
                "frontend_verified": self.verified,
                "spectrogram_shape": list(spectrogram.shape),
            },
        )

    def _select_window(self, preprocessed: PreprocessedAudio) -> Any:
        """GTM was trained on short fixed windows; pick a matching slice deterministically.

        The loudest window is chosen rather than the first, because a clip whose event
        starts two seconds in would otherwise be scored on silence.
        """
        import numpy as np

        y = np.asarray(preprocessed.samples, dtype="float32").ravel()
        n = int(round(self.frontend.window_sec * self.frontend.sample_rate))
        if y.size <= n:
            return y

        # resample to the frontend rate before windowing, if preprocessing used another
        if preprocessed.sample_rate != self.frontend.sample_rate:
            import librosa

            y = librosa.resample(y, orig_sr=preprocessed.sample_rate,
                                 target_sr=self.frontend.sample_rate).astype("float32")

        hop = max(1, n // 2)
        best_start, best_energy = 0, -1.0
        for start in range(0, y.size - n + 1, hop):
            energy = float(np.sum(np.square(y[start:start + n])))
            if energy > best_energy:
                best_energy, best_start = energy, start
        return y[best_start:best_start + n]

    def describe(self) -> dict[str, Any]:
        return {
            "model_name": "Google Teachable Machine",
            "model_version": self.model_version,
            "backend": self.backend,
            "class_names": self.class_names,
            "frontend": self.frontend.to_dict(),
            "frontend_verified": self.verified,
            "metrics": self.metrics,
        }


def compare_frontend_agreement(
    recorded: list[dict[str, Any]],
    predictor: "GtmModelPredictor",
    preprocessor: Any,
    *,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Prove the server-side GTM frontend reproduces the browser's.

    `recorded` is captured from the GTM project page in the browser — for each clip, GTM's
    own predicted class and per-class confidences. We then run the same clip through this
    module and require the argmax to match and the confidences to stay within tolerance.

    Without this check, "we integrated the exported GTM model" is an unverified claim.
    """
    from .contract import AudioSource

    rows: list[dict[str, Any]] = []
    class_matches = 0
    max_deviation = 0.0

    for item in recorded:
        source = AudioSource.from_path(item["path"])
        ours = predictor.predict(source, preprocessor)

        browser_class = item["gtm_predicted_class"]
        browser_confidences = item.get("gtm_confidences", {})
        agrees = ours.predicted_class == browser_class
        class_matches += int(agrees)

        deviation = 0.0
        if browser_confidences:
            deviation = max(
                abs(ours.confidences.get(name, 0.0) - float(value))
                for name, value in browser_confidences.items()
            )
        max_deviation = max(max_deviation, deviation)

        rows.append({
            "path": item["path"],
            "browser_class": browser_class,
            "server_class": ours.predicted_class,
            "class_agrees": agrees,
            "max_confidence_deviation": round(deviation, 6),
        })

    total = len(rows) or 1
    passed = (class_matches / total) >= 0.95 and max_deviation <= tolerance
    return {
        "n_clips": len(rows),
        "class_agreement": round(class_matches / total, 4),
        "max_confidence_deviation": round(max_deviation, 6),
        "tolerance": tolerance,
        "passed": passed,
        "rows": rows,
    }
