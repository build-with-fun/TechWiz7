"""Runs the Teachable Machine audio model on the server.

It only gets audio, never the Python model's result. The network expects the same
spectrogram as TM's browser frontend; a slightly different one gives confident wrong
answers. ``frontend_verified`` stays false until checked against browser predictions.
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
    """Spectrogram settings TM uses before its CNN, taken from the export."""

    sample_rate: int
    window_sec: float
    n_mels: int
    n_frames: int
    fmin: float = 0.0
    fmax: float | None = None
    normalize_mode: str = "none"          # browser FFT uses "zscore"
    model_input_shape: list[int] | None = None
    frontend_id: str = "unverified"
    source: str = "unknown"
    frontend_kind: str = "mel"
    # "loudest": score only the loudest window. "energy_weighted": score windows at half
    # hops and average them weighted by energy. "energy_weighted_log": same, but averaging
    # log-scores (a weighted geometric mean).
    window_aggregation: str = "loudest"

    @classmethod
    def load(cls, path: Path) -> "GtmFrontendConfig":
        if not path.exists():
            raise ModelLoadError(
                f"GTM frontend config missing: {path}\n"
                "This file records the spectrogram parameters GTM's exported model was "
                "trained with. Without it, server-side GTM inference would run with "
                "guessed parameters and return confident nonsense. Capture it from the "
                "GTM export (gtm_model/upload_package/README.md "
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
            frontend_kind=str(raw.get("frontend_kind", "mel")),
            window_aggregation=str(raw.get("window_aggregation", "loudest")),
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
            "frontend_kind": self.frontend_kind,
            "window_aggregation": self.window_aggregation,
        }


def compute_spectrogram(samples: Any, config: GtmFrontendConfig) -> Any:
    """TM input spectrogram for a mono waveform, using frontend_config.json."""
    import numpy as np

    if config.frontend_kind == "browser_fft":
        y = np.asarray(samples, dtype="float32").ravel()
        expected = int(round(config.window_sec * config.sample_rate))
        y = np.pad(y[:expected], (0, max(0, expected - len(y))))
        fft_size, hop = 2048, 1024
        padded = np.pad(y, (fft_size - hop, fft_size))
        window = np.blackman(fft_size).astype(np.float32)
        blocks = np.stack([
            padded[start:start + fft_size] * window
            for start in np.arange(config.n_frames) * hop
        ])
        magnitudes = np.abs(np.fft.rfft(blocks, axis=1)[:, :config.n_mels]) / fft_size
        result = np.clip(20 * np.log10(np.maximum(magnitudes, 1e-16)), -320, 0)
        if config.normalize_mode == "zscore":
            mean = result.mean()
            std = result.std()
            result = (result - mean) / (std + 1e-7)
        return result.astype("float32")

    import librosa

    y = np.asarray(samples, dtype="float32").ravel()

    # Fixed window length: pad or cut.
    n_expected = int(round(config.window_sec * config.sample_rate))
    if y.size < n_expected:
        y = np.pad(y, (0, n_expected - y.size))
    elif y.size > n_expected:
        y = y[:n_expected]

    # Mel power spectrogram, like TM's frontend.
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

    # Frame count the model expects.
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


def select_gtm_window(samples: Any, sample_rate: int, config: GtmFrontendConfig) -> Any:
    """Resample to the TM rate and return the loudest ``window_sec`` slice.

    make_gtm_imports.py uses the same function for the training clips.
    """
    import numpy as np

    y = np.asarray(samples, dtype="float32").ravel()
    if sample_rate != config.sample_rate:
        import librosa

        y = librosa.resample(y, orig_sr=sample_rate,
                             target_sr=config.sample_rate).astype("float32")
    n = int(round(config.window_sec * config.sample_rate))
    if y.size <= n:
        return y
    start = window_starts(y, n)[0]
    return y[start:start + n]


def window_starts(y: Any, n: int) -> list[int]:
    """Start offsets of ``n``-sample windows at half-window hops, loudest first (ties: earlier)."""
    import numpy as np

    if y.size <= n:
        return [0]
    starts = list(range(0, y.size - n + 1, max(1, n // 2)))
    energy = [float(np.sum(np.square(y[s:s + n], dtype=np.float64))) for s in starts]
    order = sorted(range(len(starts)), key=lambda i: (-energy[i], i))
    return [starts[i] for i in order]


class GtmModelPredictor:
    """The exported Teachable Machine model, converted to Keras. Raises if it can't load."""

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

        # Version = export timestamp plus aggregation mode (which changes the predictions).
        stamp = "".join(ch for ch in str(metadata.get("timeStamp", ""))[:16] if ch.isalnum())
        version = str(metadata.get("modelVersion")
                      or metadata.get("version")
                      or (f"tm-{stamp}" + {"energy_weighted": "-ew", "energy_weighted_log": "-ewlog"}
                                          .get(frontend.window_aggregation, ""))
                      if stamp else f"gtm-{frontend.frontend_id}")

        return cls(model, frontend, list(labels), version, backend=backend,
                   model_dir=gtm_dir, verified=verified, metrics=metrics)

    @staticmethod
    def _load_backend(gtm_dir: Path) -> tuple[Any, str]:
        """Load the network, trying the converted Keras model first."""
        keras_path = gtm_dir / "gtm_model.h5"
        if keras_path.exists():
            try:
                import tf_keras

                return tf_keras.models.load_model(str(keras_path), compile=False), "keras-h5"
            except Exception as exc:  # noqa: BLE001
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
            "See gtm_model/upload_package/README.md for the exact "
            "procedure and its known failure modes."
        )


    def predict(self, source: AudioSource, preprocessor: Any) -> PredictionResult:
        """Classify audio with the TM model."""
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

        windows, weights = self._windows(preprocessed)
        spectrogram = compute_spectrogram(windows[0], self.frontend)
        model_input = np.stack([spectrogram] + [compute_spectrogram(w, self.frontend)
                                                for w in windows[1:]])[..., None]
        shape = self.frontend.model_input_shape
        if shape and list(model_input.shape[1:]) != [value for value in shape if value is not None][-3:]:
            raise ModelLoadError(
                f"GTM frontend produced {model_input.shape}, expected {shape}"
            )

        started = time.perf_counter()
        output = self.model.predict(np.asarray(model_input, dtype="float32"), verbose=0)
        inference_sec = time.perf_counter() - started

        if isinstance(output, (list, tuple)):
            output = output[0]
        scores = np.asarray(output).reshape(len(windows), -1)
        share = (weights / weights.sum())[:, None]
        if self.frontend.window_aggregation == "energy_weighted_log":
            log_mean = (np.log(np.clip(scores, 1e-9, 1.0)) * share).sum(axis=0)
            raw = np.exp(log_mean - log_mean.max())
            raw = raw / raw.sum()
        else:
            raw = (scores * share).sum(axis=0)

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
                "windows_scored": len(windows),
                "window_aggregation": self.frontend.window_aggregation,
            },
        )

    def _select_window(self, preprocessed: PreprocessedAudio) -> Any:
        return select_gtm_window(preprocessed.samples, int(preprocessed.sample_rate),
                                 self.frontend)

    def _windows(self, preprocessed: PreprocessedAudio) -> tuple[list[Any], Any]:
        """Windows to score (loudest first) and their weights."""
        import numpy as np

        if not self.frontend.window_aggregation.startswith("energy_weighted"):
            return [self._select_window(preprocessed)], np.ones(1)
        y = np.asarray(preprocessed.samples, dtype="float32").ravel()
        if int(preprocessed.sample_rate) != self.frontend.sample_rate:
            import librosa

            y = librosa.resample(y, orig_sr=int(preprocessed.sample_rate),
                                 target_sr=self.frontend.sample_rate).astype("float32")
        n = int(round(self.frontend.window_sec * self.frontend.sample_rate))
        if y.size <= n:
            return [y], np.ones(1)
        starts = window_starts(y, n)
        windows = [y[s:s + n] for s in starts]
        energy = np.array([float(np.sum(np.square(w, dtype=np.float64))) for w in windows])
        return windows, (energy if energy.sum() > 0 else np.ones(len(windows)))

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
    """Compare server predictions with ones recorded in the browser for the same clips.

    The top class must match and confidences must be within tolerance.
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
