"""Classical candidates for the Python model comparison (SRS Step 7, FR xxiii-xxvi).

SVM (RBF), random forest, extra trees, gradient boosting, HistGradientBoosting and XGBoost,
all on the same 254 hand-made features and the same split, built through build_estimator
so the only thing that differs between candidates is the estimator.

Critical-class weighting is a candidate dimension (weighted_variant), not a global setting:
the SRS asks for 85% recall on five critical classes, which competes with overall accuracy,
so every model is trained both ways and the comparison table shows what the weighting cost.
The best of these (HistGradientBoosting) is the baseline; the served model is CNN14 + MLP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np



@dataclass
class CandidateSpec:
    """One comparable model: how to build it, what to search, and what it costs.

    Scaling lives inside the estimator pipeline, so it is fitted on training rows only and
    saved with the model.
    """

    name: str
    builder: Callable[[Mapping[str, Any], int, np.random.Generator | None], Any]
    param_grid: dict[str, Sequence[Any]] = field(default_factory=dict)
    preprocess: str = "none"
    notes: str = ""
    n_jobs: int = 1

    def build(self, params: Mapping[str, Any] | None = None, seed: int = 0) -> Any:
        return self.builder(dict(params or {}), seed, None)


def _svm(params: Mapping[str, Any], seed: int, _rng: Any) -> Any:
    """RBF SVM with cross-validated (Platt) probability calibration.

    The comparison subtracts the two models' top confidences, so an uncalibrated SVM that says
    0.99 when it is right 70% of the time would distort every consistency status.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import SVC

    base = SVC(
        kernel="rbf",
        C=float(params.get("C", 10.0)),
        gamma=params.get("gamma", "scale"),
        class_weight=params.get("class_weight"),
        random_state=seed,
        cache_size=512,
    )
    # Sigmoid (Platt) rather than isotonic: ~200 rows per class is too few for an isotonic map.
    return CalibratedClassifierCV(
        base,
        method=str(params.get("calibration", "sigmoid")),
        cv=int(params.get("calibration_cv", 3)),
        ensemble=False,
    )


def _random_forest(params: Mapping[str, Any], seed: int, _rng: Any) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 500)),
        max_depth=params.get("max_depth"),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        max_features=params.get("max_features", "sqrt"),
        class_weight=params.get("class_weight"),
        random_state=seed,
        n_jobs=-1,
    )


def _gradient_boosting(params: Mapping[str, Any], seed: int, _rng: Any) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier

    # No class_weight parameter here; the weights go in as sample_weight (see fit_estimator).
    return GradientBoostingClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        max_depth=int(params.get("max_depth", 3)),
        subsample=float(params.get("subsample", 1.0)),
        random_state=seed,
    )


def _hist_gradient_boosting(params: Mapping[str, Any], seed: int, _rng: Any) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return _HistGBShim(
        HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 300)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 63)),
            max_depth=params.get("max_depth"),
            l2_regularization=float(params.get("l2_regularization", 0.0)),
            early_stopping=False,
            random_state=seed,
        )
    )


class _HistGBShim(BaseEstimator, ClassifierMixin):
    """Wrapper that lets HistGradientBoosting take string-keyed class weights.

    HGB validates a class_weight dict against its internal integer labels, so the zoo's
    string-keyed weights raise. The weights travel as sample_weight instead, and get_params is
    deliberately shallow so build_estimator cannot bake the string dict into the inner model.
    """

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"estimator": self.estimator}

    def fit(self, X, y, sample_weight=None, **fit_params):
        labels = sorted({str(v) for v in y})
        self._label_to_int_ = {label: i for i, label in enumerate(labels)}
        self.classes_ = np.asarray(labels, dtype=object)
        encoded = np.asarray([self._label_to_int_[str(v)] for v in y], dtype=np.int64)
        self.estimator.fit(X, encoded, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.classes_[np.asarray(self.estimator.predict(X))]

    def predict_proba(self, X):
        # Inner classes_ are sorted ints, i.e. the same order as the sorted string labels.
        return self.estimator.predict_proba(X)


class _LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Fit an integer-only estimator (XGBoost) on string class labels.

    Labels are encoded as sorted(unique(y)), the order sklearn itself uses, so classes_ and
    the predict_proba columns stay in string-label order for the rest of the pipeline.
    """

    def __init__(self, estimator: Any):
        self.estimator = estimator

    # sample_weight is named explicitly: fit_estimator reads this signature to decide how to
    # pass class weights, and without it the weighting would be dropped silently.
    def fit(self, X, y, sample_weight=None, **fit_params):
        labels = sorted({str(v) for v in y})
        self._label_to_int_ = {label: i for i, label in enumerate(labels)}
        self.classes_ = np.asarray(labels, dtype=object)

        encoded = np.asarray([self._label_to_int_[str(v)] for v in y], dtype=np.int64)
        if sample_weight is not None:
            self.estimator.fit(X, encoded, sample_weight=sample_weight, **fit_params)
        else:
            self.estimator.fit(X, encoded, **fit_params)
        return self

    def predict(self, X):
        ints = np.asarray(self.estimator.predict(X))
        return self.classes_[ints]

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    # Fitted state (feature_importances_ etc.) lives on the inner estimator.
    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_

    def __getattr__(self, name):
        # Only reached for attributes the wrapper does not define itself.
        return getattr(self.estimator, name)


def _xgboost(params: Mapping[str, Any], seed: int, _rng: Any) -> Any:
    from xgboost import XGBClassifier

    return _LabelEncodedClassifier(
        XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 6)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            eval_metric="mlogloss",
        )
    )


def _extra_trees(params: Mapping[str, Any], seed: int, _rng: Any) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(
        n_estimators=int(params.get("n_estimators", 500)),
        max_depth=params.get("max_depth"),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        class_weight=params.get("class_weight"),
        random_state=seed,
        n_jobs=-1,
    )



#: The classical candidates, cheapest first. Two or three values per axis keeps the search
#: within a CPU budget; the question is which family wins, not the fourth decimal.
CLASSICAL_CANDIDATES: dict[str, CandidateSpec] = {
    "svm_rbf": CandidateSpec(
        name="svm_rbf",
        builder=_svm,
        preprocess="scale",
        param_grid={"C": [1.0, 10.0, 100.0], "gamma": ["scale"]},
        notes=(
            "RBF-kernel SVM with Platt-scaled probabilities. Strongest cheap model expected "
            "here; its confidence is the least calibrated of the four, which matters because "
            "the comparison rests on confidence differences rather than on labels."
        ),
    ),
    "random_forest": CandidateSpec(
        name="random_forest",
        builder=_random_forest,
        preprocess="none",
        param_grid={
            "n_estimators": [500],
            "max_depth": [None, 20],
            "min_samples_leaf": [1, 2],
        },
        notes="Bagged trees; scale-free, gives feature_importances_ we can defend line by line.",
    ),
    "extra_trees": CandidateSpec(
        name="extra_trees",
        builder=_extra_trees,
        preprocess="none",
        param_grid={"n_estimators": [500], "min_samples_leaf": [1, 2]},
        notes="Extremely randomised trees -- a cheap variance check on the forest result.",
    ),
    "gradient_boosting": CandidateSpec(
        name="gradient_boosting",
        builder=_gradient_boosting,
        preprocess="none",
        param_grid={
            "n_estimators": [200],
            "learning_rate": [0.1],
            "max_depth": [3, 5],
        },
        notes="sklearn boosted trees; critical-class weighting via sample_weight.",
    ),
    "xgboost": CandidateSpec(
        name="xgboost",
        builder=_xgboost,
        preprocess="none",
        param_grid={
            "n_estimators": [300],
            "learning_rate": [0.1],
            "max_depth": [4, 6],
        },
        notes="Regularised boosted trees; native sample_weight for critical-class weighting.",
    ),
    "hist_gradient_boosting": CandidateSpec(
        name="hist_gradient_boosting",
        builder=_hist_gradient_boosting,
        preprocess="none",
        param_grid={
            "max_iter": [400],
            "learning_rate": [0.1],
            "max_leaf_nodes": [31],
            "max_depth": [8],
        },
        notes=(
            "Histogram gradient boosting (sklearn's LightGBM): the strongest classical "
            "family on this feature set. Critical-class weighting travels as row weights "
            "via the shim, because HGB validates class_weight against its encoded labels."
        ),
    ),
}


def get_candidate(name: str) -> CandidateSpec:
    try:
        return CLASSICAL_CANDIDATES[name]
    except KeyError:
        raise KeyError(
            f"unknown classical candidate {name!r}; available: {sorted(CLASSICAL_CANDIDATES)}"
        ) from None




def critical_class_weights(
    class_names: Sequence[str],
    critical_classes: Sequence[str],
    *,
    boost: float = 2.0,
) -> dict[str, float]:
    """Class weights with the critical classes multiplied by ``boost``; the rest stay at 1.0."""
    critical = set(critical_classes)
    unknown = critical - set(class_names)
    if unknown:
        raise ValueError(f"critical classes not in the class list: {sorted(unknown)}")
    if boost < 1.0:
        raise ValueError(f"boost must be >= 1.0, got {boost}")
    return {name: (float(boost) if name in critical else 1.0) for name in class_names}


def _class_weight_key(estimator: Any) -> str | None:
    """Parameter path that carries class_weight (nested ones included), or None."""
    try:
        params = estimator.get_params()
    except AttributeError:
        return None
    for key in ("class_weight", *(k for k in params if k.endswith("__class_weight"))):
        if key in params:
            return key
    return None


def supports_class_weight(estimator: Any) -> bool:
    """True when class_weight can be set somewhere on this estimator."""
    return _class_weight_key(estimator) is not None


def sample_weights_for(
    y: Sequence[str],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Per-row weights for estimators that take sample_weight instead of class_weight."""
    missing = sorted({str(label) for label in y} - set(weights))
    if missing:
        raise ValueError(f"no weight defined for label(s): {missing}")
    return np.asarray([float(weights[str(label)]) for label in y], dtype=np.float64)


def weighted_variant(spec: CandidateSpec, weights: Mapping[str, float] | None) -> CandidateSpec:
    """``spec`` with the class weights added to its parameter grid."""
    if not weights:
        return spec
    grid = {k: list(v) for k, v in spec.param_grid.items()}
    grid["class_weight"] = [dict(weights)]
    return CandidateSpec(
        name=f"{spec.name}+cw",
        builder=spec.builder,
        param_grid=grid,
        preprocess=spec.preprocess,
        notes=f"{spec.notes} [critical-class weights applied]",
        n_jobs=spec.n_jobs,
    )




def build_estimator(spec: CandidateSpec, params: Mapping[str, Any] | None = None, seed: int = 0) -> Any:
    """A ready-to-fit estimator with the spec's preprocessing in front.

    class_weight is stripped for estimators that do not accept it; fit_estimator then passes
    the weights as sample_weight.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    resolved = dict(params or {})
    seed = int(resolved.pop("_seed", seed))
    raw = spec.builder({k: v for k, v in resolved.items() if k != "class_weight"}, seed, None)

    key = _class_weight_key(raw) if "class_weight" in resolved else None
    if key is not None:
        raw.set_params(**{key: resolved["class_weight"]})

    steps: list[tuple[str, Any]] = []
    if spec.preprocess == "scale":
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", raw))
    return Pipeline(steps)


def fit_estimator(
    estimator: Any,
    X: np.ndarray,
    y: Sequence[str],
    *,
    class_weights: Mapping[str, float] | None = None,
) -> Any:
    """Fit, routing class weights to whatever the estimator understands.

    A wrong route is invisible (the model trains and ignores the weights), so an estimator
    that takes neither class_weight nor sample_weight gets a warning.
    """
    import warnings

    target = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator

    if class_weights:
        if supports_class_weight(target):
            # Already in the estimator; sample_weight as well would double-count the boost.
            return estimator.fit(X, list(y))
        if _accepts_sample_weight(target):
            estimator.fit(X, list(y), estimator__sample_weight=sample_weights_for(y, class_weights))
            return estimator
        warnings.warn(
            f"{type(target).__name__} accepts neither class_weight nor sample_weight; "
            "critical-class weighting was NOT applied for this candidate.",
            RuntimeWarning,
            stacklevel=2,
        )
    return estimator.fit(X, list(y))


def _accepts_sample_weight(estimator: Any) -> bool:
    import inspect

    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


def estimator_coefficients(estimator: Any) -> Mapping[str, Any]:
    """The fitted inner estimator, for importances or support vectors."""
    return getattr(estimator, "named_steps", {}).get("estimator", estimator)


def feature_importances(estimator: Any, columns: Sequence[str]) -> dict[str, float]:
    """Ranked feature importances, or {} for models that have none (no substitute is invented)."""
    inner = estimator_coefficients(estimator)
    importances = getattr(inner, "feature_importances_", None)
    if importances is None:
        return {}
    pairs = sorted(zip(columns, (float(v) for v in importances)), key=lambda kv: -kv[1])
    return dict(pairs)


def describe_zoo() -> list[dict[str, Any]]:
    """The classical candidates, for the comparison report."""
    return [
        {
            "name": spec.name,
            "preprocess": spec.preprocess,
            "param_grid": {k: list(v) for k, v in spec.param_grid.items()},
            "notes": spec.notes,
        }
        for spec in CLASSICAL_CANDIDATES.values()
    ]
