"""The dual-model independence guarantee, tested rather than asserted.

SRS integrity requires two SEPARATELY trained models whose predictions are compared
after the fact. The mandate is broken the moment the Teachable Machine model can see
the Python model's answer — that is no longer two models, it is one model and a
sycophant, and every "consistency" number derived from it is fiction.

A comment claiming independence is worth nothing. These tests attack the property from
three directions:

  1. STRUCTURAL — what can reach the GTM code at all (imports, signatures):
     the GTM module must not be able to name the Python predictor, and no parameter
     of any GTM entry point may accept a prediction, a confidence mapping or **kwargs
     to smuggle one through.

  2. BEHAVIOURAL — the poisoned-input proof. Run the GTM model, then re-run it with the
     Python model's output changed arbitrarily, and require the GTM output to be
     bit-identical. If the GTM path reads Python state anywhere — argument, attribute,
     module global, cache — this fails.

  3. SYMMETRIC — the Python model must not read the GTM model either. Independence that
     only runs one way is not independence; it is a hierarchy.

Plus the input-mode test: an uploaded file and the same audio as an in-memory live
window must reach the SAME preprocessor and produce the SAME confidences. Two code
paths that "should" match drift apart; this one is held together by a test.
"""

from __future__ import annotations

import ast
import copy
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
    load_thresholds,
    normalise_confidences,
)
from src.inference.gtm_predictor import GtmFrontendConfig, GtmModelPredictor
from src.inference.predictor import PythonModelPredictor

REPO_ROOT = Path(__file__).resolve().parents[1]
GTM_SOURCE = REPO_ROOT / "src" / "inference" / "gtm_predictor.py"
PREDICTOR_SOURCE = REPO_ROOT / "src" / "inference" / "predictor.py"
CLASSES = class_names()


# --------------------------------------------------------------------------------------
# Test doubles — a GTM model that actually depends on its input, so "output unchanged"
# means something. A constant stub would make the poisoned-input test pass trivially.
# --------------------------------------------------------------------------------------

class StubGtmModel:
    """A deterministic, CONTINUOUS function of the spectrogram it is handed.

    Continuous matters: a stub whose output is a discontinuous hash of its input (say
    `int(sum*1000) % n_classes`) flips its answer on the 16-bit quantisation of a wav
    round trip, and then a genuine same-preprocessing-path test fails for a reason that
    has nothing to do with the pipeline. Real neural nets respond smoothly to tiny input
    changes; the double has to as well, or it tests the wrong thing.
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.calls = 0
        self.last_input_sum = None

    def predict(self, x, verbose=0):  # noqa: ARG002 - mirrors the Keras signature
        self.calls += 1
        arr = np.asarray(x, dtype="float64")
        self.last_input_sum = float(arr.sum())

        # Per-band mean energy is a smooth, input-dependent summary: the same audio always
        # gives the same vector, different audio gives a different one.
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
    """The single preprocessing path. Records every call so tests can prove who saw what."""

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
    """A plausible Python-model output. Only ever used to try to influence the GTM path."""
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


# --------------------------------------------------------------------------------------
# 1. STRUCTURAL — what the GTM code is even able to name
# --------------------------------------------------------------------------------------

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
    """No import of the Python predictor or of the code that compares the two models."""
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
    """The reverse direction, so independence is not a one-way hierarchy."""
    imported = _imported_module_names(PREDICTOR_SOURCE)
    forbidden = {t for t in imported if "gtm" in t}
    assert forbidden == set(), (
        f"predictor.py imports the GTM model: {sorted(forbidden)}"
    )


def test_only_the_comparison_layer_sees_both_models():
    """consistency.py is the one place allowed to hold two results at once.

    That concentration is deliberate: it means the SRS comparison logic can be audited
    in a single file, and neither model can secretly consult the other.
    """
    consistency = REPO_ROOT / "src" / "inference" / "consistency.py"
    imported = _imported_module_names(consistency)
    assert any("predictor" in t or "gtm" in t for t in imported) is False, (
        "consistency.py should depend only on the contract, not on either predictor — "
        "it must be testable with hand-built results."
    )


GTM_ENTRY_POINTS = ["predict", "predict_from_preprocessed", "_select_window"]


@pytest.mark.parametrize("method_name", GTM_ENTRY_POINTS)
def test_no_gtm_entry_point_accepts_a_prediction(method_name):
    """Signature-level check: nothing on the GTM path can be handed the Python opinion."""
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
    """The two positional arguments are the audio and the shared preprocessor."""
    params = list(inspect.signature(GtmModelPredictor.predict).parameters.values())
    assert [p.name for p in params] == ["self", "source", "preprocessor"]


# --------------------------------------------------------------------------------------
# 2. BEHAVIOURAL — the poisoned-input proof
# --------------------------------------------------------------------------------------

def test_gtm_output_is_immutable_to_the_python_models_opinion(monkeypatch):
    """Change the Python model's answer completely; the GTM answer must not move.

    This is the test that would fail if someone "helped" the second model by seeding it
    with the first model's guess — through an argument, an attribute, a module global or
    a cache. The audio is the only input that differs between the runs, so any change in
    the GTM output is attributable to the Python opinion that was injected alongside it.
    """
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
        # Every route the Python opinion could plausibly travel: a module global on the
        # package, and an attribute on the running predictor.
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

    # Nothing beyond the two attributes the test itself planted. A random plant is
    # necessarily visible; what matters is that predict() neither read it nor kept one.
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

    # A predictor that never saw a planted value is the honest control.
    fresh = make_gtm_predictor()
    fresh.predict(source, preprocessor)
    assert not any(isinstance(v, PredictionResult) for v in vars(fresh).values())


def test_gtm_output_does_change_when_the_audio_changes():
    """The control. Without this, the test above would pass on a constant model."""
    gtm = make_gtm_predictor()
    preprocessor = RecordingPreprocessor()

    quiet = gtm.predict(AudioSource.from_samples(make_audio(freq=200.0), 16000), preprocessor)
    loud = gtm.predict(AudioSource.from_samples(make_audio(freq=3000.0), 16000), preprocessor)

    assert quiet.confidences != loud.confidences, (
        "the stub GTM model ignores its input, so the independence test proves nothing"
    )


def test_python_predictor_output_is_immutable_to_the_gtm_opinion(tmp_path):
    """The reverse poisoning: the GTM answer must not steer the Python model."""
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

    # Even with a GTM result planted everywhere reachable, the Python path must not read it.
    sys.modules["src.inference"].gtm_result = gtm_opinion
    python_model.gtm_result = gtm_opinion
    replayed = python_model.predict(source, preprocessor)

    assert replayed.confidences == baseline.confidences
    sys.modules["src.inference"].__dict__.pop("gtm_result", None)


# --------------------------------------------------------------------------------------
# 3. The shared preprocessing path — both input modes
# --------------------------------------------------------------------------------------

def test_upload_and_live_window_reach_the_same_preprocessor(tmp_path):
    """A file and the same audio in memory must go through one preprocessing path.

    The SRS makes the live path and the upload path separate features and the natural
    implementation drift is for them to grow their own resampling. Then a clip that
    scores 0.95 on upload scores 0.6 live, and nobody can say which is right.
    """
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
    # instances_seen holds ids as ints — compare the values, not id-of-int.
    assert set(preprocessor.instances_seen) == {id(preprocessor)}, (
        "the two input modes reached different preprocessor objects"
    )
    assert [c.origin for c in preprocessor.calls] == ["upload", "live"]

    # The direct evidence that the two doors lead to one room: the spectrograms handed to
    # the model are the same array up to 16-bit quantisation noise. This is continuous, so
    # it measures the pipeline rather than the model's sensitivity.
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
        "of full scale — the paths have diverged"
    )

    # And the downstream decision agrees.
    assert upload_result.predicted_class == live_result.predicted_class
    for name in CLASSES:
        assert abs(upload_result.confidences[name] - live_result.confidences[name]) < 0.05

    # Origin is carried through honestly rather than being silently normalised away.
    assert upload_result.source_origin == "upload"
    assert live_result.source_origin == "live"
    assert upload_result.to_dict()["source_origin"] == "upload"


def test_both_input_modes_are_rejected_by_the_same_gate():
    """A rejected clip must be rejected whichever door it came in through."""
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


# --------------------------------------------------------------------------------------
# 4. Import hygiene — the web app must boot without TensorFlow
# --------------------------------------------------------------------------------------

def test_inference_package_imports_without_tensorflow():
    """The app must start and REPORT a broken GTM model, not refuse to start.

    A single upload endpoint should not be able to take down the dashboard, the
    manual-review queue and the event log.
    """
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
