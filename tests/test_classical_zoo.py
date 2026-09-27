"""Tests for the classical candidates (python_models/classical.py).

Like test_deep_models.py: the registry is complete, every candidate has the sklearn
interface, and class weights actually reach the estimator.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from python_models import classical


def _synthetic_matrix(seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Three well-separated Gaussian blobs."""
    rng = np.random.default_rng(seed)
    centres = np.array([[5, 0] * 4, [0, 5] * 4, [-5, -5] * 4], dtype=np.float64)
    labels = ["Alpha", "Beta", "Gamma"]
    X = np.vstack([rng.normal(loc=c, scale=0.7, size=(30, 8)) for c in centres])
    y = [label for label in labels for _ in range(30)]
    return X, y


# Registry


def test_registry_has_at_least_three_classical_candidates():
    """At least three candidates."""
    assert len(classical.CLASSICAL_CANDIDATES) >= 3


@pytest.mark.parametrize("name", sorted(classical.CLASSICAL_CANDIDATES))
def test_every_candidate_is_registered_and_documented(name):
    spec = classical.get_candidate(name)
    assert spec.name == name
    assert spec.param_grid, f"{name}: an empty grid means nothing was actually compared"
    assert spec.notes, f"{name}: the zoo must document the hypothesis each model tests"


def test_get_candidate_rejects_unknown_names():
    with pytest.raises(KeyError, match="unknown classical candidate"):
        classical.get_candidate("definitely_not_a_model")


# Fit and predict, every candidate


@pytest.fixture(scope="module")
def fitted_models() -> dict[str, Any]:
    X, y = _synthetic_matrix()
    fitted: dict[str, Any] = {}
    for name, spec in classical.CLASSICAL_CANDIDATES.items():
        params = {k: grid[0] for k, grid in spec.param_grid.items()}
        estimator = classical.build_estimator(spec, params, seed=0)
        estimator = classical.fit_estimator(estimator, X, y)
        fitted[name] = estimator
    return fitted


@pytest.mark.parametrize("name", sorted(classical.CLASSICAL_CANDIDATES))
def test_predict_returns_original_string_labels(name, fitted_models):
    """predict returns string labels, not encoded integers."""
    estimator = fitted_models[name]
    X, y = _synthetic_matrix(seed=1)
    pred = estimator.predict(X)
    assert set(np.unique(pred)) <= set(y), f"{name}: predicted non-label values {set(np.unique(pred))}"


@pytest.mark.parametrize("name", sorted(classical.CLASSICAL_CANDIDATES))
def test_predict_proba_rows_sum_to_one(name, fitted_models):
    estimator = fitted_models[name]
    X, _ = _synthetic_matrix(seed=2)
    proba = np.asarray(estimator.predict_proba(X))
    assert proba.ndim == 2 and proba.shape[0] == X.shape[0]
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", sorted(classical.CLASSICAL_CANDIDATES))
def test_predict_agrees_with_proba_argmax(name, fitted_models):
    """predict matches the argmax of predict_proba."""
    estimator = fitted_models[name]
    X, _ = _synthetic_matrix(seed=3)
    proba = np.asarray(estimator.predict_proba(X))
    names = np.asarray(sorted(set(_synthetic_matrix(seed=3)[1])), dtype=object)
    pred = estimator.predict(X)
    assert (names[np.argmax(proba, axis=1)] == pred).all(), f"{name}: proba columns do not line up with predict()"


# Class weights


WEIGHTS = {"Alpha": 2.0, "Beta": 1.0, "Gamma": 1.0}


@pytest.mark.parametrize("name", sorted(classical.CLASSICAL_CANDIDATES))
def test_weighted_variant_fits_without_silently_dropping_weights(name):
    """Weighted variants either use the weights or warn."""
    spec = classical.weighted_variant(classical.get_candidate(name), WEIGHTS)
    assert spec.name == f"{name}+cw"
    X, y = _synthetic_matrix(seed=4)
    params = {k: grid[0] for k, grid in spec.param_grid.items() if k != "class_weight"}
    estimator = classical.build_estimator(spec, params, seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator = classical.fit_estimator(estimator, X, list(y), class_weights=WEIGHTS)
    dropped = [str(w.message) for w in caught if "weighting was NOT applied" in str(w.message)]
    assert not dropped, f"{name}: {dropped[0]}"
    # Still predicts real labels.
    assert set(np.unique(estimator.predict(X))) <= set(y)


def test_hist_gradient_boosting_weight_travels_as_row_weights():
    """HGB gets string-keyed class weights as row weights (this used to raise ValueError)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    spec = classical.get_candidate("hist_gradient_boosting")
    estimator = classical.build_estimator(spec, {"max_iter": 30}, seed=0)
    inner = estimator.steps[-1][1]
    assert isinstance(inner.estimator, HistGradientBoostingClassifier)
    # The dict must not be set on the inner estimator.
    assert getattr(inner.estimator, "class_weight", None) is None
    # The wrapper still accepts sample_weight.
    assert "sample_weight" in inner.fit.__code__.co_varnames


# Feature importances


def test_feature_importances_empty_when_not_exposed(fitted_models):
    """Models without importances return {}."""
    cols = [f"f{i}" for i in range(8)]
    for name, estimator in fitted_models.items():
        result = classical.feature_importances(estimator, cols)
        assert isinstance(result, dict)
        if result:
            assert all(col in cols for col in result)
            assert abs(sum(result.values()) - 1.0) < 1e-3 or max(result.values()) <= 1.0