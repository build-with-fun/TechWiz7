"""The saved-model contract: the two silent failures that ruin a classifier.

A model that is loaded wrong does not crash. It returns confident, well-formed, WRONG
answers — a valid-looking confusion matrix built on shuffled labels is the single most
expensive bug in a project like this, because nothing downstream can detect it.

Two ways that happens here:

  LABEL DRIFT   `predict_proba` returns columns in the ESTIMATOR's fitted order. If the
                code maps those columns onto a class-name list stored in a different
                order, every prediction is attached to the wrong class name. Accuracy
                looks plausible; the whole report is fiction.

  FEATURE DRIFT A saved model scores a fixed number of columns. If the feature extractor
                is later changed, the model keeps scoring happily — on the wrong columns.
                Nothing raises. The model simply becomes bad.

`save_bundle`/`load` exist to make both of those impossible to do quietly, and these
tests are what keeps those guards from being removed for convenience.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.inference.contract import (
    AudioSource,
    PreprocessedAudio,
    class_names,
    normalise_confidences,
)
from src.inference.predictor import (
    ModelBundle,
    PythonModelPredictor,
    save_bundle,
)
from src.inference.contract import ModelLoadError

CLASSES = class_names()          # the 10 canonical names, in config order
N_FEATURES = 24


# --------------------------------------------------------------------------------------
# Fixtures — a real sklearn estimator, because a stub cannot have real drift bugs
# --------------------------------------------------------------------------------------

def _training_data(n_per_class: int = 12, seed: int = 7):
    """Linearly separable-ish data so a real estimator behaves like a real estimator."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, name in enumerate(CLASSES):
        centre = np.zeros(N_FEATURES)
        centre[i % N_FEATURES] = 3.0
        centre[(i + 5) % N_FEATURES] = 1.5
        X.append(centre + 0.3 * rng.standard_normal((n_per_class, N_FEATURES)))
        y.extend([name] * n_per_class)
    return np.vstack(X), np.asarray(y)


@pytest.fixture(scope="module")
def fitted_estimator():
    from sklearn.linear_model import LogisticRegression

    X, y = _training_data()
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    return model


def feature_extractor_factory(width: int = N_FEATURES):
    def extract(preprocessed: PreprocessedAudio) -> np.ndarray:
        # Deterministic, input-dependent, right width.
        seed = int(abs(float(np.asarray(preprocessed.samples).sum())) * 1e6) % (2**31)
        rng = np.random.default_rng(seed)
        return rng.standard_normal((1, width))
    return extract


class PassthroughPreprocessor:
    def __call__(self, source: AudioSource) -> PreprocessedAudio:
        samples = (
            np.asarray(source.samples, dtype="float32")
            if source.samples is not None
            else np.zeros(16000, dtype="float32")
        )
        return PreprocessedAudio(
            samples=samples,
            sample_rate=16000,
            duration_sec=len(samples) / 16000,
            quality={"verdict": "Good"},
            preprocessing={},
        )


def build_bundle(estimator, *, class_names_override=None, n_features=N_FEATURES) -> ModelBundle:
    return ModelBundle(
        estimator=estimator,
        class_names=list(class_names_override or estimator.classes_),
        feature_version="test-v1",
        feature_columns=[f"f{i}" for i in range(n_features)],
        model_name="logreg-test",
        model_version="1.0.0",
        metrics={"accuracy": 0.9},
    )


# --------------------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------------------

def test_save_bundle_then_load_reproduces_the_same_predictions(tmp_path, fitted_estimator):
    """The round trip is the promise every training script relies on."""
    save_bundle(
        fitted_estimator,
        tmp_path,
        class_names=CLASSES,
        feature_version="mfcc-v1",
        feature_columns=[f"f{i}" for i in range(N_FEATURES)],
        model_name="logreg-test",
        model_version="1.2.3",
        metrics={"accuracy": 0.91, "macro_f1": 0.90},
    )

    for expected in ("model.joblib", "label_encoder.json", "feature_config.json", "model_meta.json"):
        assert (tmp_path / expected).exists(), f"save_bundle did not write {expected}"

    extractor = feature_extractor_factory()
    in_memory = PythonModelPredictor(build_bundle(fitted_estimator), extractor)
    reloaded = PythonModelPredictor.load(tmp_path, extractor)

    assert reloaded.model_version == "1.2.3"
    source = AudioSource.from_samples(np.ones(16000, dtype="float32"), 16000)

    before = in_memory.predict(source, PassthroughPreprocessor())
    after = reloaded.predict(source, PassthroughPreprocessor())
    assert before.confidences == after.confidences
    assert before.predicted_class == after.predicted_class
    assert reloaded.describe()["metrics"]["macro_f1"] == 0.90


def test_saved_label_order_is_the_estimators_fitted_order(tmp_path, fitted_estimator):
    """The stored order must be the order predict_proba's columns come back in.

    Storing the config's class order instead is the exact bug that renames every
    prediction, so this asserts the file agrees with `classes_`.
    """
    save_bundle(
        fitted_estimator,
        tmp_path,
        class_names=CLASSES,
        feature_version="mfcc-v1",
    )
    saved = json.loads((tmp_path / "label_encoder.json").read_text())["class_names"]
    assert saved == list(fitted_estimator.classes_)


def test_load_reorders_when_the_json_order_differs_from_the_fitted_order(tmp_path, fitted_estimator):
    """Same set of names, different order in the sidecar: predictions must still be right.

    This is the drift guard earning its keep. The estimator's argmax indexes its own
    fitted order; whichever order the JSON happens to list, the names attached to the
    scores must be the fitted ones.
    """
    save_bundle(
        fitted_estimator,
        tmp_path,
        class_names=CLASSES,
        feature_version="mfcc-v1",
        feature_columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    # Scramble the sidecar's order, keeping the same set of names.
    sidecar = tmp_path / "label_encoder.json"
    saved = json.loads(sidecar.read_text())["class_names"]
    sidecar.write_text(json.dumps({"class_names": list(reversed(saved))}))

    extractor = feature_extractor_factory()
    reloaded = PythonModelPredictor.load(tmp_path, extractor)
    assert reloaded.class_names == list(fitted_estimator.classes_), (
        "load() trusted the sidecar's order over the estimator's fitted order"
    )

    source = AudioSource.from_samples(np.ones(16000, dtype="float32"), 16000)
    correct = PythonModelPredictor(build_bundle(fitted_estimator), extractor).predict(
        source, PassthroughPreprocessor()
    )
    assert reloaded.predict(source, PassthroughPreprocessor()).predicted_class == correct.predicted_class


# --------------------------------------------------------------------------------------
# Label drift — must raise, loudly, at load time
# --------------------------------------------------------------------------------------

def test_label_drift_raises_at_load(tmp_path, fitted_estimator):
    """A model fitted on one class set, a sidecar naming another: refuse to serve it."""
    save_bundle(
        fitted_estimator,
        tmp_path,
        class_names=CLASSES,
        feature_version="mfcc-v1",
        feature_columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    wrong = list(CLASSES)
    wrong[0], wrong[1] = "Explosion", "Siren"
    (tmp_path / "label_encoder.json").write_text(json.dumps({"class_names": wrong}))

    with pytest.raises(ModelLoadError, match="label drift"):
        PythonModelPredictor.load(tmp_path, feature_extractor_factory())


def test_label_count_mismatch_raises_at_load(tmp_path, fitted_estimator):
    save_bundle(
        fitted_estimator,
        tmp_path,
        class_names=CLASSES,
        feature_version="mfcc-v1",
        feature_columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    (tmp_path / "label_encoder.json").write_text(
        json.dumps({"class_names": CLASSES[:9]})
    )
    with pytest.raises(ModelLoadError, match="mismatch|drift"):
        PythonModelPredictor.load(tmp_path, feature_extractor_factory())


def test_the_error_message_names_both_class_lists(tmp_path, fitted_estimator):
    """An operator woken at 3am needs to know WHICH list is wrong."""
    save_bundle(
        fitted_estimator, tmp_path, class_names=CLASSES,
        feature_version="v1", feature_columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    (tmp_path / "label_encoder.json").write_text(
        json.dumps({"class_names": ["A", "B"] + CLASSES[2:]})
    )
    with pytest.raises(ModelLoadError) as excinfo:
        PythonModelPredictor.load(tmp_path, feature_extractor_factory())
    message = str(excinfo.value)
    assert "A" in message and "Gunshot" in message


# --------------------------------------------------------------------------------------
# Missing artefacts — refused, never guessed
# --------------------------------------------------------------------------------------

def test_missing_model_file_raises(tmp_path):
    with pytest.raises(ModelLoadError, match="no saved model"):
        PythonModelPredictor.load(tmp_path, feature_extractor_factory())


def test_missing_label_encoder_raises(tmp_path, fitted_estimator):
    import joblib

    joblib.dump(fitted_estimator, tmp_path / "model.joblib")
    with pytest.raises(ModelLoadError):
        PythonModelPredictor.load(tmp_path, feature_extractor_factory())


def test_missing_feature_config_raises(tmp_path, fitted_estimator):
    import joblib

    joblib.dump(fitted_estimator, tmp_path / "model.joblib")
    (tmp_path / "label_encoder.json").write_text(
        json.dumps({"class_names": CLASSES})
    )
    with pytest.raises(ModelLoadError):
        PythonModelPredictor.load(tmp_path, feature_extractor_factory())


# --------------------------------------------------------------------------------------
# Feature drift — must raise at predict time
# --------------------------------------------------------------------------------------

def test_feature_drift_raises_at_predict(fitted_estimator):
    """A changed feature extractor must not silently score the wrong columns."""
    bundle = build_bundle(fitted_estimator, n_features=N_FEATURES)
    predictor = PythonModelPredictor(bundle, feature_extractor_factory(width=N_FEATURES + 6))

    with pytest.raises(ValueError, match="feature drift"):
        predictor.predict(
            AudioSource.from_samples(np.ones(16000, dtype="float32"), 16000),
            PassthroughPreprocessor(),
        )


def test_feature_drift_message_states_both_widths(fitted_estimator):
    bundle = build_bundle(fitted_estimator, n_features=N_FEATURES)
    predictor = PythonModelPredictor(bundle, feature_extractor_factory(width=17))
    with pytest.raises(ValueError) as excinfo:
        predictor.predict(
            AudioSource.from_samples(np.ones(16000, dtype="float32"), 16000),
            PassthroughPreprocessor(),
        )
    message = str(excinfo.value)
    assert str(N_FEATURES) in message and "17" in message


def test_no_width_check_when_the_bundle_does_not_declare_one(fitted_estimator):
    """An undeclared width cannot be checked; the code must not invent a limit."""
    class WideEstimator:
        classes_ = np.asarray(CLASSES)

        def predict_proba(self, X):  # noqa: N803
            out = np.full((len(X), len(CLASSES)), 0.1)
            out[:, 0] = 0.1
            return out / out.sum(axis=1, keepdims=True)

    bundle = ModelBundle(
        estimator=WideEstimator(),
        class_names=list(CLASSES),
        feature_version="v1",
        model_name="wide",
        model_version="1",
    )
    assert bundle.n_features is None
    predictor = PythonModelPredictor(bundle, feature_extractor_factory(width=99))
    result = predictor.predict(
        AudioSource.from_samples(np.ones(16000, dtype="float32"), 16000),
        PassthroughPreprocessor(),
    )
    assert set(result.confidences) == set(CLASSES)


# --------------------------------------------------------------------------------------
# Honest output — no fabricated confidences
# --------------------------------------------------------------------------------------

def test_a_model_without_predict_proba_reports_a_one_hot_not_a_fake_distribution():
    """If the model only says a class, report that — do not manufacture a softmax."""
    class HardEstimator:
        classes_ = np.asarray(CLASSES)
        n_features_in_ = N_FEATURES

        def predict(self, X):  # noqa: N803
            return np.asarray([CLASSES[3]] * len(X))

    bundle = ModelBundle(
        estimator=HardEstimator(),
        class_names=list(CLASSES),
        feature_version="v1",
        model_name="hard",
        model_version="1",
        feature_columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    result = PythonModelPredictor(bundle, feature_extractor_factory()).predict(
        AudioSource.from_samples(np.ones(16000, dtype="float32"), 16000),
        PassthroughPreprocessor(),
    )
    assert result.predicted_class == CLASSES[3]
    assert result.confidence == 1.0
    assert sum(result.confidences.values()) == pytest.approx(1.0)
    # Exactly one class non-zero: spread confidence would be invented information.
    assert sum(1 for v in result.confidences.values() if v > 0) == 1


def test_predictions_cover_every_class_and_sum_to_one(fitted_estimator):
    """The UI, the comparison table and the TM independence rule all need the full row."""
    predictor = PythonModelPredictor(build_bundle(fitted_estimator), feature_extractor_factory())
    result = predictor.predict(
        AudioSource.from_samples(np.ones(16000, dtype="float32"), 16000),
        PassthroughPreprocessor(),
    )
    assert set(result.confidences) == set(CLASSES)
    assert len(result.confidences) == 10
    assert sum(result.confidences.values()) == pytest.approx(1.0)
    assert 0.0 <= result.confidence <= 1.0


def test_rejected_audio_is_never_scored(fitted_estimator):
    class Rejecting:
        def __call__(self, source: AudioSource) -> PreprocessedAudio:
            return PreprocessedAudio(
                samples=np.zeros(16000, dtype="float32"), sample_rate=16000,
                duration_sec=1.0, rejected=True,
                rejection_reason="silence (RMS -70 dBFS)",
            )

    predictor = PythonModelPredictor(build_bundle(fitted_estimator), feature_extractor_factory())
    with pytest.raises(ValueError, match="rejected"):
        predictor.predict(
            AudioSource.from_samples(np.zeros(16000, dtype="float32"), 16000), Rejecting()
        )


# --------------------------------------------------------------------------------------
# The fingerprint — duplicate detection (FR lxxiii)
# --------------------------------------------------------------------------------------

def test_same_audio_gives_the_same_fingerprint_and_different_audio_does_not(tmp_path):
    from src.inference.contract import audio_fingerprint

    samples = np.sin(np.linspace(0, 40, 16000)).astype("float32")
    a = audio_fingerprint(AudioSource.from_samples(samples, 16000))
    b = audio_fingerprint(AudioSource.from_samples(samples.copy(), 16000))
    c = audio_fingerprint(AudioSource.from_samples(samples * 0.5, 16000))
    assert a == b, "identical audio must hash identically or duplicate detection misses it"
    assert a != c


# --------------------------------------------------------------------------------------
# describe() — the audit trail
# --------------------------------------------------------------------------------------

def test_describe_reports_everything_the_audit_trail_needs(tmp_path, fitted_estimator):
    save_bundle(
        fitted_estimator, tmp_path, class_names=CLASSES, feature_version="mfcc-v3",
        feature_columns=[f"f{i}" for i in range(N_FEATURES)],
        model_name="svm-rbf", model_version="2.0.0", metrics={"macro_f1": 0.83},
    )
    described = PythonModelPredictor.load(tmp_path, feature_extractor_factory()).describe()
    assert described["feature_version"] == "mfcc-v3"
    assert described["n_features"] == N_FEATURES
    assert described["model_name"] == "svm-rbf"
    assert described["metrics"] == {"macro_f1": 0.83}
    assert set(described["class_names"]) == set(CLASSES)
