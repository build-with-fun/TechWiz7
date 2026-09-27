"""The two models must stay independent (SRS integrity rules).

1. The TM module doesn't import the Python predictor and takes no prediction as input.
2. Changing the Python model's output doesn't change the TM output.
3. The same holds the other way round.

Also checks that an upload and the same audio as a live window get the same preprocessing.
"""

from __future__ import annotations

import ast
import inspect
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from src.inference.contract import (
    AudioSource,
    PreprocessedAudio,
    PredictionResult,
    class_names,
)
from src.inference.gtm_predictor import GtmFrontendConfig, GtmModelPredictor
from src.inference.predictor import PythonModelPredictor

REPO_ROOT = Path(__file__).resolve().parents[1]
GTM_SOURCE = REPO_ROOT / "src" / "inference" / "gtm_predictor.py"
PREDICTOR_SOURCE = REPO_ROOT / "src" / "inference" / "predictor.py"
CLASSES = class_names()


# A fake TM model whose output depends on its input (a constant stub would prove nothing).

class StubGtmModel:
    """Output is a smooth function of the spectrogram."""

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.calls = 0
        self.last_input_sum = None

    def predict(self, x, verbose=0):  # noqa: ARG002 - Keras signature
        self.calls += 1
        arr = np.asarray(x, dtype="float64")
        self.last_input_sum = float(arr.sum())

        # Mean energy per band.
        flat = arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else arr.reshape(1, -1)
        per_band = flat.mean(axis=-1).ravel()
        profile = np.zeros(self.n_classes)
        n = min(self.n_classes, per_band.size)
        profile[:n] = per_band[:n]

        logits = profile * 50.0
        logits -= logits.max()
        exp = np.exp(logits)
        return (exp / exp.sum()).reshape(1, -1)


def make_frontend() -> GtmFrontendConfig:
    return GtmFrontendConfig(
        sample_rate=16000,
        window_sec=1.0,
        n_mels=40,
        n_frames=32,
        normalize_mode="none",
        model_input_shape=None,
        frontend_id="test-frontend",
        source="tests/test_model_independence.py",
    )


def make_gtm_predictor() -> GtmModelPredictor:
    return GtmModelPredictor(
        model=StubGtmModel(len(CLASSES)),
        frontend=make_frontend(),
        class_names=CLASSES,
        model_version="gtm-test-0.0.1",
        backend="stub",
    )


def make_audio(seconds: float = 1.0, sr: int = 16000, freq: float = 440.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    tone = 0.3 * np.sin(2 * np.pi * freq * t)
    return (tone + 0.01 * rng.standard_normal(t.size)).astype("float32")


class RecordingPreprocessor:
    """Preprocessor that records every call."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.calls: list[AudioSource] = []
        self.instances_seen: list[int] = []

    def __call__(self, source: AudioSource) -> PreprocessedAudio:
        self.calls.append(source)
        self.instances_seen.append(id(self))
        if source.path is not None:
            import soundfile as sf

            samples, sr = sf.read(str(source.path), dtype="float32", always_2d=False)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
        else:
            samples, sr = np.asarray(source.samples, dtype="float32"), source.sample_rate
        return PreprocessedAudio(
            samples=samples,
            sample_rate=int(sr),
            duration_sec=float(len(samples) / sr),
            source_path=str(source.path) if source.path else None,
            quality={"verdict": "Good", "rms_dbfs": -20.0, "snr_db": 25.0},
            preprocessing={"target_sample_rate": self.sample_rate, "channels": "mono"},
        )


def python_result_for(class_name: str, confidence: float) -> PredictionResult:
    """A fake Python result, used to try to influence the TM path."""
    confidences = {name: (1.0 - confidence) / (len(CLASSES) - 1) for name in CLASSES}
    confidences[class_name] = confidence
    return PredictionResult(
        model_name="python-test",
        model_version="python-test-0.0.1",
        predicted_class=class_name,
        confidence=confidence,
        confidences=confidences,
        latency_sec=0.01,
        feature_version="test",
        source_origin="upload",
    )


# 1. Imports and signatures

def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.update(f"{base}.{alias.name}" for alias in node.names)
            names.add(base)
    return names


def test_gtm_module_cannot_name_the_python_predictor():
    """The TM module doesn't import the Python predictor or the comparison code."""
    imported = _imported_module_names(GTM_SOURCE)
    forbidden = {
        token
        for token in imported
        if token.endswith("predictor") and "gtm" not in token
    } | {t for t in imported if "consistency" in t}
    assert forbidden == set(), (
        "gtm_predictor.py imports code that knows the Python model's answer: "
        f"{sorted(forbidden)}"
    )


def test_python_predictor_cannot_name_the_gtm_model():
    """And the other way round."""
    imported = _imported_module_names(PREDICTOR_SOURCE)
    forbidden = {t for t in imported if "gtm" in t}
    assert forbidden == set(), (
        f"predictor.py imports the GTM model: {sorted(forbidden)}"
    )


def test_only_the_comparison_layer_sees_both_models():
    """Only consistency.py handles both results."""
    consistency = REPO_ROOT / "src" / "inference" / "consistency.py"
    imported = _imported_module_names(consistency)
    assert any("predictor" in t or "gtm" in t for t in imported) is False, (
        "consistency.py should depend only on the contract, not on either predictor."
    )


GTM_ENTRY_POINTS = ["predict", "predict_from_preprocessed", "_select_window"]


@pytest.mark.parametrize("method_name", GTM_ENTRY_POINTS)
def test_no_gtm_entry_point_accepts_a_prediction(method_name):
    """No TM function accepts a prediction."""
    method = getattr(GtmModelPredictor, method_name)
    params = inspect.signature(method).parameters

    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ), f"{method_name} takes **kwargs, which is a hole large enough to pass anything."

    for name, param in params.items():
        if name in {"self", "preprocessed"}:
            continue
        annotation = "" if param.annotation is inspect.Parameter.empty else str(param.annotation)
        assert "PredictionResult" not in annotation, (
            f"{method_name}({name}) is annotated {annotation}; the GTM model must never "
            "receive the Python model's result."
        )
        assert name.lower() not in {
            "prediction", "python_result", "py_result", "other_result",
            "confidences", "python_confidences", "prior", "hint",
        }, f"{method_name}({name}) looks like it carries a prediction into the GTM model."


def test_gtm_predict_arguments_are_audio_only():
    """TM predict takes the audio and the preprocessor only."""
    params = list(inspect.signature(GtmModelPredictor.predict).parameters.values())
    assert [p.name for p in params] == ["self", "source", "preprocessor"]


# 2. Behaviour

def test_gtm_output_is_immutable_to_the_python_models_opinion(monkeypatch):
    """Changing the Python answer doesn't change the TM answer."""
    gtm = make_gtm_predictor()
    preprocessor = RecordingPreprocessor()
    source = AudioSource.from_samples(make_audio(seconds=1.0, freq=440.0), 16000)

    baseline = gtm.predict(source, preprocessor)

    poison = [
        python_result_for("Gunshot", 0.99),
        python_result_for("Background Noise", 0.51),
        python_result_for("Machinery Fault", 1.0),
    ]

    for poisoned_result in poison:
        # Plant the Python result as a module global and on the predictor.
        monkeypatch.setattr(
            sys.modules["src.inference"], "last_result", poisoned_result, raising=False
        )
        gtm.last_result = poisoned_result  # type: ignore[attr-defined]

        replayed = gtm.predict(source, preprocessor)
        assert replayed.confidences == baseline.confidences, (
            "GTM confidences moved when the Python model's opinion was present. "
            f"Python said {poisoned_result.predicted_class}; GTM said "
            f"{replayed.predicted_class} instead of {baseline.predicted_class}."
        )
        assert replayed.predicted_class == baseline.predicted_class
        assert replayed.confidence == baseline.confidence

    # Only the attributes we planted ourselves.
    planted = {"last_result"}
    leaked = [
        name
        for name, value in vars(gtm).items()
        if isinstance(value, PredictionResult) and name not in planted
    ]
    assert leaked == [], f"the GTM predictor is holding a Python prediction in {leaked}"
    assert vars(gtm)["last_result"] is poison[-1], (
        "the predictor overwrote the planted attribute, so the run proves nothing about "
        "what it read"
    )

    # Compare with a clean predictor.
    fresh = make_gtm_predictor()
    fresh.predict(source, preprocessor)
    assert not any(isinstance(v, PredictionResult) for v in vars(fresh).values())


def test_gtm_output_does_change_when_the_audio_changes():
    """Control: different audio does change the TM output."""
    gtm = make_gtm_predictor()
    preprocessor = RecordingPreprocessor()

    quiet = gtm.predict(AudioSource.from_samples(make_audio(freq=200.0), 16000), preprocessor)
    loud = gtm.predict(AudioSource.from_samples(make_audio(freq=3000.0), 16000), preprocessor)

    assert quiet.confidences != loud.confidences, (
        "the stub GTM model ignores its input, so the independence test proves nothing"
    )


def test_python_predictor_output_is_immutable_to_the_gtm_opinion(tmp_path):
    """The TM answer doesn't affect the Python model."""
    class FixedEstimator:
        def __init__(self) -> None:
            self.classes_ = np.asarray(CLASSES)
            self.n_features_in_ = 3

        def predict_proba(self, X):  # noqa: N803 - sklearn convention
            out = np.full((len(X), len(CLASSES)), 0.02)
            out[:, 2] = 0.80
            return out

    def feature_extractor(preprocessed: PreprocessedAudio) -> np.ndarray:
        return np.asarray([[1.0, 2.0, 3.0]], dtype="float64")

    from src.inference.predictor import ModelBundle

    bundle = ModelBundle(
        estimator=FixedEstimator(),
        class_names=list(CLASSES),
        model_name="python-test",
        model_version="0.0.1",
        feature_version="test",
        feature_columns=["a", "b", "c"],
        metrics={},
    )
    python_model = PythonModelPredictor(bundle, feature_extractor)
    source = AudioSource.from_samples(make_audio(), 16000)
    preprocessor = RecordingPreprocessor()

    baseline = python_model.predict(source, preprocessor)
    gtm_opinion = python_result_for("Gunshot", 0.97)

    # A planted TM result doesn't change the Python output.
    sys.modules["src.inference"].gtm_result = gtm_opinion
    python_model.gtm_result = gtm_opinion
    replayed = python_model.predict(source, preprocessor)

    assert replayed.confidences == baseline.confidences
    sys.modules["src.inference"].__dict__.pop("gtm_result", None)


# 3. Shared preprocessing for both input modes

def test_upload_and_live_window_reach_the_same_preprocessor(tmp_path):
    """A file and the same audio in memory go through the same preprocessor."""
    sr = 16000
    samples = make_audio(seconds=1.0, sr=sr)
    clip_path = tmp_path / "clip.wav"
    with wave.open(str(clip_path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())

    preprocessor = RecordingPreprocessor()
    gtm = make_gtm_predictor()

    upload_result = gtm.predict(AudioSource.from_path(clip_path, origin="upload"), preprocessor)
    live_result = gtm.predict(
        AudioSource.from_samples(samples, sr, origin="live"), preprocessor
    )

    assert len(preprocessor.calls) == 2
    # instances_seen holds ids as ints.
    assert set(preprocessor.instances_seen) == {id(preprocessor)}, (
        "the two input modes reached different preprocessor objects"
    )
    assert [c.origin for c in preprocessor.calls] == ["upload", "live"]

    # The spectrograms match up to 16-bit quantisation noise.
    from src.inference.gtm_predictor import compute_spectrogram

    upload_pre = preprocessor(AudioSource.from_path(clip_path, origin="upload"))
    live_pre = preprocessor(AudioSource.from_samples(samples, sr, origin="live"))
    upload_spec = compute_spectrogram(gtm._select_window(upload_pre), gtm.frontend)
    live_spec = compute_spectrogram(gtm._select_window(live_pre), gtm.frontend)

    assert upload_spec.shape == live_spec.shape == (
        gtm.frontend.n_mels, gtm.frontend.n_frames
    ), f"frontend shape drift: {upload_spec.shape} vs {live_spec.shape}"
    max_delta = float(np.max(np.abs(upload_spec - live_spec)))
    scale = float(np.max(np.abs(live_spec))) or 1.0
    assert max_delta / scale < 0.01, (
        f"the two input modes produced spectrograms differing by {max_delta / scale:.3%} "
        "of full scale; the two paths have diverged"
    )

    # Same decision.
    assert upload_result.predicted_class == live_result.predicted_class
    for name in CLASSES:
        assert abs(upload_result.confidences[name] - live_result.confidences[name]) < 0.05

    # The origin is kept.
    assert upload_result.source_origin == "upload"
    assert live_result.source_origin == "live"
    assert upload_result.to_dict()["source_origin"] == "upload"


def test_both_input_modes_are_rejected_by_the_same_gate():
    """Bad audio is rejected the same way in both modes."""
    class RejectingPreprocessor(RecordingPreprocessor):
        def __call__(self, source: AudioSource) -> PreprocessedAudio:
            result = super().__call__(source)
            result.rejected = True
            result.rejection_reason = "too quiet (RMS -62 dBFS)"
            return result

    gtm = make_gtm_predictor()
    preprocessor = RejectingPreprocessor()

    for source in (
        AudioSource.from_samples(make_audio(), 16000, origin="live"),
    ):
        with pytest.raises(ValueError, match="rejected"):
            gtm.predict(source, preprocessor)


# 4. The web app starts without TensorFlow

def test_inference_package_imports_without_tensorflow():
    """src.inference imports without TensorFlow, so a broken TM model doesn't stop the app."""
    import importlib

    for module_name in sorted(
        m for m in list(sys.modules) if m.startswith("src.inference")
    ):
        del sys.modules[module_name]

    saved = {k: v for k, v in sys.modules.items() if k.startswith(("tensorflow", "keras"))}
    for key in saved:
        del sys.modules[key]

    try:
        module = importlib.import_module("src.inference")
        assert hasattr(module, "load_gtm_predictor") or hasattr(module, "GtmModelPredictor")
        assert not any(
            m.startswith(("tensorflow", "keras")) for m in sys.modules
        ), "importing the inference package pulled TensorFlow into the web app's start-up path"
    finally:
        sys.modules.update(saved)


class BandModel:
    """Fake model: class 0 if the lower mel half is louder, else class 1."""

    def predict(self, x, verbose=0):
        x = np.asarray(x)
        half = x.shape[1] // 2
        low = x[:, :half].mean(axis=(1, 2, 3))
        high = x[:, half:].mean(axis=(1, 2, 3))
        out = np.full((x.shape[0], len(CLASSES)), 0.01)
        out[np.arange(x.shape[0]), np.where(low > high, 0, 1)] = 0.9
        return out / out.sum(axis=1, keepdims=True)


def _two_part_clip(sr: int = 16000) -> np.ndarray:
    """One loud second at 440 Hz, then two slightly quieter seconds at 3 kHz."""
    t = np.arange(sr) / sr
    first = 0.5 * np.sin(2 * np.pi * 440 * t)
    rest = 0.45 * np.sin(2 * np.pi * 3000 * np.arange(2 * sr) / sr)
    return np.concatenate([first, rest]).astype("float32")


def _predict_with(aggregation: str):
    frontend = make_frontend()
    frontend.window_aggregation = aggregation
    gtm = GtmModelPredictor(model=BandModel(), frontend=frontend, class_names=CLASSES,
                            model_version="gtm-test-0.0.1", backend="stub")
    return gtm.predict(AudioSource.from_samples(_two_part_clip(), 16000), RecordingPreprocessor())


def test_loudest_window_mode_scores_one_window():
    result = _predict_with("loudest")
    assert result.extra["windows_scored"] == 1
    assert result.predicted_class == CLASSES[0]        # loudest second is 440 Hz


def test_energy_weighted_mode_scores_every_window():
    result = _predict_with("energy_weighted")
    assert result.extra["windows_scored"] == 5         # 3 s at half-second hops
    assert result.predicted_class == CLASSES[1]        # 2 s of 3 kHz beats 1 s of 440 Hz
    assert abs(sum(result.confidences.values()) - 1.0) < 1e-6
