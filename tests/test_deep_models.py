"""Deep model zoo contract tests -- SRS Step 7.

Owner: nadia.

These are CONTRACT tests, not accuracy tests.  Their job is to guarantee that whatever the
training script picks as the winner, the app can actually load it and predict with it:

    * every registered candidate builds, fits and predicts over the locked feature vector;
    * ``predict_proba`` returns ``(n, n_classes)`` with rows summing to 1 in the config class
      order, so the comparison table and the consistency verdict are never silently misaligned;
    * the saved bundle reloads through ``PythonModelPredictor`` with no label or feature
      drift -- the exact path the web app's pipeline takes at load time;
    * the spatial slice reads the melband columns the extractor actually produces.

Accuracy on real audio is asserted by the training script against the untouched test split,
not here: a smoke corpus is too small to say anything about generalisation, and asserting a
floor against generated noise would be a test that cannot fail honestly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT,):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from typing import Any  # noqa: E402

from feature_extraction.features import FeatureExtractor, feature_columns  # noqa: E402
from python_models import deep  # noqa: E402
from src.inference.predictor import PythonModelPredictor, save_bundle  # noqa: E402


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

N_CLASSES = 10
N_ROWS = 72


def _load_classes() -> tuple[list[str], list[str]]:
    config = json.loads((REPO_ROOT / "config" / "classes.json").read_text(encoding="utf-8"))
    names = [str(c["name"]) for c in config["classes"]]
    critical = [str(c) for c in config.get("critical_classes", [])]
    return names, critical


CLASS_NAMES, CRITICAL = _load_classes()


def _synthetic_matrix(seed: int) -> tuple[np.ndarray, list[str]]:
    """A matrix with genuine class structure in the melband block.

    The classes are given distinct mel profiles, so a model that reads the spatial block can
    actually learn something -- which distinguishes 'the model is broken' from 'the model has
    nothing to learn' when a test checks prediction shape only.
    """
    rng = np.random.default_rng(seed)
    cols = feature_columns()
    idx = deep.melband_columns(cols)
    X = rng.normal(size=(N_ROWS, len(cols))).astype(np.float32)
    labels: list[str] = []
    for i in range(N_ROWS):
        c = i % N_CLASSES
        profile = np.linspace(0.0, 1.0, len(idx)) + c * 0.7
        X[i, idx] = 3.0 * profile + rng.normal(0, 0.25, len(idx))
        labels.append(CLASS_NAMES[c])
    return X, labels


@pytest.fixture(scope="module")
def synthetic() -> tuple[np.ndarray, list[str]]:
    return _synthetic_matrix(0)


@pytest.fixture(scope="module")
def fitted_models(synthetic) -> dict[str, Any]:
    """Every candidate, fitted on the synthetic matrix once and reused across the module.

    Built with small epoch counts on purpose: these tests are about the contract, and a
    contract test that takes minutes to run gets skipped.
    """
    X, y = synthetic
    fitted: dict[str, Any] = {}
    for name, spec in deep.DEEP_CANDIDATES.items():
        params = {"epochs": 4, "channels": 16, "hidden": 32, "head_units": 24}
        estimator = deep.build_estimator(spec, params, seed=0)
        deep.fit_estimator(estimator, X, y)
        fitted[name] = estimator
    return fitted


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def test_registry_has_at_least_three_deep_candidates():
    """The SRS requires at least three models trained and compared. Fewer than three means the
    requirement is not met by construction, so this is checked before anything is fitted."""
    assert len(deep.DEEP_CANDIDATES) >= 3, (
        "SRS Step 7 requires >=3 models; the deep zoo has " f"{len(deep.DEEP_CANDIDATES)}"
    )


@pytest.mark.parametrize("name", sorted(deep.DEEP_CANDIDATES))
def test_every_candidate_is_registered_and_documented(name):
    spec = deep.get_candidate(name)
    assert spec.name == name
    assert spec.backend in {"torch", "keras"}, f"{name}: unknown backend {spec.backend!r}"
    # A candidate with no notes cannot be explained to an evaluator who asks what it tests.
    assert spec.notes, f"{name}: the zoo must document the hypothesis each model tests"


def test_get_candidate_rejects_unknown_names():
    """An unknown name must raise, not return a half-built model."""
    with pytest.raises(KeyError, match="unknown deep candidate"):
        deep.get_candidate("definitely_not_a_model")


# --------------------------------------------------------------------------------------
# Feature-vector contract
# --------------------------------------------------------------------------------------


def test_feature_vector_is_locked_width():
    """The extractor's width is what the app feeds an estimator; a change here is a contract
    break between features and every trained model, so it is pinned."""
    assert len(feature_columns()) == 254


def test_melband_block_is_present_and_contiguous():
    """The spatial models slice the melband columns out of the locked vector and treat them as
    a spectrum. A non-contiguous or missing block means the slice reads the wrong columns."""
    idx = deep.melband_columns(feature_columns())
    assert len(idx) == 128, "the CNN/CRNN frequency axis is the 128-band mel block"
    assert idx == list(range(idx[0], idx[0] + len(idx))), "the mel block must be contiguous"


def test_melband_columns_raises_on_missing_block():
    with pytest.raises(ValueError, match="melband_"):
        deep.melband_columns(["mfcc_00_mean", "zcr_mean"])


# --------------------------------------------------------------------------------------
# Fit / predict contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(deep.DEEP_CANDIDATES))
def test_fit_then_predict_proba_shape_and_order(fitted_models, name):
    """``predict_proba`` must give one probability per class in the config order.

    ``TuningProtocol`` indexes confidence columns positionally, and the app's consistency
    verdict subtracts the GTM model's confidence from a specific class's -- a silent column
    reorder between the two would produce a wrong verdict, not an error.
    """
    X, _ = _synthetic_matrix(1)
    estimator = fitted_models[name]
    proba = estimator.predict_proba(X)
    assert proba.shape == (X.shape[0], N_CLASSES)
    assert list(estimator.classes_) == sorted(CLASS_NAMES)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(proba).all(), "a NaN/inf confidence would poison the consistency verdict"


@pytest.mark.parametrize("name", sorted(deep.DEEP_CANDIDATES))
def test_predict_agrees_with_predict_proba(fitted_models, name):
    X, _ = _synthetic_matrix(2)
    estimator = fitted_models[name]
    proba = estimator.predict_proba(X)
    labels = list(estimator.predict(X))
    assert len(labels) == X.shape[0]
    expected = [str(estimator.classes_[i]) for i in proba.argmax(axis=1)]
    assert labels == expected


@pytest.mark.parametrize("name", sorted(deep.DEEP_CANDIDATES))
def test_fit_is_deterministic_under_a_fixed_seed(synthetic, name):
    """Two estimators of the same candidate on the same seed must agree exactly.

    Seeding is what makes the frozen split reproducible and the comparison report
    re-generable; a model that ignores its seed cannot be re-run for an evaluator.
    """
    X, y = synthetic
    params = {"epochs": 3, "channels": 16, "hidden": 32, "head_units": 24}
    a = deep.build_estimator(deep.get_candidate(name), params, seed=7)
    deep.fit_estimator(a, X, y)
    b = deep.build_estimator(deep.get_candidate(name), params, seed=7)
    deep.fit_estimator(b, X, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-5)


@pytest.mark.parametrize("name", sorted(deep.DEEP_CANDIDATES))
def test_sample_weight_changes_the_fit(synthetic, name):
    """Critical-class weighting is a first-class knob in the protocol, so it must actually be
    wired through -- a fit that ignores ``sample_weight`` would silently discard the boost."""
    X, y = synthetic
    params = {"epochs": 4, "channels": 16, "hidden": 32, "head_units": 24}
    plain = deep.build_estimator(deep.get_candidate(name), params, seed=0)
    deep.fit_estimator(plain, X, y)
    boosted = deep.build_estimator(deep.get_candidate(name), params, seed=0)
    weights = {c: 3.0 for c in CRITICAL}
    deep.fit_estimator(boosted, X, y, class_weights=weights)
    assert not np.allclose(plain.predict_proba(X), boosted.predict_proba(X), atol=1e-6), (
        f"{name}: sample_weight had no effect on the fit"
    )


@pytest.mark.parametrize("name", sorted(deep.DEEP_CANDIDATES))
def test_feature_drift_is_rejected(fitted_models, name):
    """The app raises on a width mismatch; a deep model must raise the same way rather than
    silently misreading the vector."""
    X, _ = _synthetic_matrix(3)
    truncated = X[:, :100]
    estimator = fitted_models[name]
    with pytest.raises(ValueError, match="drift|columns"):
        estimator.predict_proba(truncated)


# --------------------------------------------------------------------------------------
# The save/load round-trip the web app performs at startup
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(deep.DEEP_CANDIDATES))
def test_save_and_reload_through_the_app_predictor(tmp_path, fitted_models, name):
    """The bundle must reload through ``PythonModelPredictor`` -- the app's only load path.

    This is the test that catches the failure mode nothing else does: a model that trains and
    scores fine but whose saved bundle the app cannot serve.  Both backends go through
    ``save_bundle`` joblib-dump, and the Keras wrapper carries its weights as bytes and
    rebuilds the architecture on load, so this also proves that path end to end.
    """
    X, y = _synthetic_matrix(4)
    estimator = fitted_models[name]
    model_dir = tmp_path / f"bundle_{name}"
    out_dir = save_bundle(
        estimator,
        model_dir,
        class_names=CLASS_NAMES,
        feature_version=FeatureExtractor().feature_version,
        feature_columns=feature_columns(),
        model_name=f"deep_{name}",
        model_version="1.0.0",
        metrics={},
        metadata={"algorithm": name},
    )
    assert (Path(out_dir) / "model.joblib").exists()

    predictor = PythonModelPredictor.load(Path(out_dir), FeatureExtractor())

    # ``_predict_features`` is the exact path ``predict`` takes after feature extraction, so
    # feeding it the vector directly isolates the round-trip from the audio pipeline: if the
    # reloaded estimator disagrees with the one that was saved, this is where it shows.
    result = predictor._predict_features(X[0], origin="test")
    assert set(result.confidences) == set(CLASS_NAMES)
    total = sum(result.confidences.values())
    assert 0.999 <= total <= 1.001

    saved_order = list(estimator.classes_)
    fresh = estimator.predict_proba(X[:1])[0]
    reloaded = np.asarray(
        [result.confidences[c] for c in saved_order], dtype=np.float64
    )
    np.testing.assert_allclose(reloaded, fresh, atol=5e-4)


def test_predictor_class_order_matches_the_config(tmp_path, fitted_models):
    """The bundle's class order must match the estimator's fitted order, which the loader
    preserves and the app verifies against config/classes.json at load time."""
    estimator = fitted_models["cnn1d"]
    out_dir = save_bundle(
        estimator,
        tmp_path / "bundle_order",
        class_names=CLASS_NAMES,
        feature_version=FeatureExtractor().feature_version,
        feature_columns=feature_columns(),
        model_name="deep_cnn1d",
        model_version="1.0.0",
    )
    predictor = PythonModelPredictor.load(Path(out_dir), FeatureExtractor())
    assert list(predictor.bundle.class_names) == list(estimator.classes_)
    assert set(predictor.bundle.class_names) == set(CLASS_NAMES)
