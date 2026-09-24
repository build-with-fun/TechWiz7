"""Classical model zoo for SonicSentinel AI -- SRS Step 7, FR xxiii-xxvi.

Owner: bilal (classical ML + tuning).

WHAT THIS IS
------------
The SRS requires "at least three models trained and compared". This module defines the
classical half of that comparison: a linear/RBF SVM, a Random Forest, a Gradient Boosting
machine and an XGBoost model, all consuming the SAME locked feature vector from
``feature_extraction.feature_columns()`` on the SAME frozen split.

WHY THESE FOUR
--------------
Each one probes a different hypothesis about this problem, which is the only reason to
compare models at all:

* ``svm_rbf``  -- with 254 standardised features and 2,100 training rows, a kernel SVM is
  the strongest cheap model available. It handles the class-to-class boundaries in this
  feature space smoothly and needs no feature engineering. It is also the model whose
  confidence output is worst calibrated (Platt scaling), which the report must say.
* ``random_forest`` -- a bagged tree ensemble: robust to feature scaling, gives an honest
  out-of-bag view, and its ``feature_importances_`` let us defend columns to an evaluator.
* ``gradient_boosting`` -- sklearn's boosted trees: sequential error correction, usually
  the best classical model on tabular features of this size.
* ``xgboost`` -- a faster, regularised boosted-tree implementation with explicit
  ``sample_weight`` support, which is how critical-class weighting is applied here.

Every one of them is a legitimately trainable model on 2,100 x 254. None is a wrapper
around an external API -- the SRS forbids an external generative-AI API for the final
classification, and nothing here calls one.

THE TWO THINGS THAT BREAK COMPARISONS
-------------------------------------
1. Different preprocessing per model. Every candidate here is built by
   :func:`build_estimator`, which composes from the same ``preprocess`` step, so the only
   thing that differs between candidates is the estimator itself.
2. Different feature columns per model. The column list is passed in from the extractor
   and stored in the saved bundle; ``save_bundle`` then makes a mismatch a load-time
   error rather than a silently permuted prediction.

CLASS WEIGHTING, HONESTLY
-------------------------
The SRS demands >= 85% recall on five critical classes and only >= 85% accuracy overall.
Those two goals conflict: the critical classes are 5 of 10, so a model that never
predicts ``Gunshot`` still scores 90% accuracy if the other nine are easy. Weighting the
critical classes trades overall accuracy for critical recall.

That trade is MEASURED, not assumed: :func:`weighted_variant` returns the same model with
the class weights applied, :mod:`tuning` scores both on validation, and the report records
the accuracy each one gave up for the recall it bought. If weighting does not earn its
cost on this dataset, the numbers will say so and the unweighted model wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# Candidate specification
# --------------------------------------------------------------------------------------


@dataclass
class CandidateSpec:
    """One comparable model: how to build it, what to search, and what it costs.

    ``preprocess`` is ``"scale"`` or ``"none"``. Scaling is part of the model pipeline, not
    a separate step, so a saved bundle always carries the exact transform that produced its
    training features -- the classic failure where a scaler fitted on the whole dataset
    leaks test statistics into training is impossible here because the scaler lives inside
    the estimator and is therefore fitted by ``fit(X_train)`` alone.
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
    """RBF SVM with explicitly calibrated probabilities.

    ``SVC(probability=True)`` is deprecated in scikit-learn 1.9 and, more importantly, its
    Platt scaling is fitted on the same data the support vectors came from. This module uses
    ``CalibratedClassifierCV(..., ensemble=False)``, which calibrates the SVC's decision
    function with cross-validated folds instead.

    That is not a detail: the SRS's consistency taxonomy is computed from
    ``|python_top_confidence - gtm_top_confidence|``, so a model whose confidences are not
    probabilities makes that subtraction meaningless. A model that says 0.99 when it is
    right 70% of the time would inflate every "Strong Match" it appears in, and every
    "Model Disagreement" it fails to trigger.
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
    # sigmoid == Platt scaling, the standard choice for SVM; isotonic needs far more
    # calibration data than 10 classes x ~200 rows can supply without overfitting the map.
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

    # sklearn's GradientBoostingClassifier has no class_weight parameter, so critical-class
    # weighting is applied through sample_weight at fit time instead (see fit_estimator).
    return GradientBoostingClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        max_depth=int(params.get("max_depth", 3)),
        subsample=float(params.get("subsample", 1.0)),
        random_state=seed,
    )


def _xgboost(params: Mapping[str, Any], seed: int, _rng: Any) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
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


# --------------------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------------------

#: The comparable classical candidates, cheapest first. The grids are deliberately small:
#: two or three values per axis keeps the search inside the SRS budget on a CPU box, and a
#: wide coarse grid beats a narrow fine one when the real question is "which family wins".
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
}


def get_candidate(name: str) -> CandidateSpec:
    try:
        return CLASSICAL_CANDIDATES[name]
    except KeyError:
        raise KeyError(
            f"unknown classical candidate {name!r}; available: {sorted(CLASSICAL_CANDIDATES)}"
        ) from None


# --------------------------------------------------------------------------------------
# Class weighting for the critical classes
# --------------------------------------------------------------------------------------


def critical_class_weights(
    class_names: Sequence[str],
    critical_classes: Sequence[str],
    *,
    boost: float = 2.0,
) -> dict[str, float]:
    """Weights that make a critical-class miss cost more than a non-critical-class miss.

    ``boost`` multiplies the critical classes' weight. It is a parameter rather than a
    constant because how far to push recall above precision is a product decision the SRS
    sets a floor for (>= 85% critical recall) and does not otherwise specify -- and because
    the tuning harness measures whether the boost actually bought the recall it was
    supposed to, rather than trusting that it does.

    Non-critical classes stay at 1.0, so the weights are interpretable to an evaluator:
    "a Gunshot training clip counted twice as much as a Background Noise clip."
    """
    critical = set(critical_classes)
    unknown = critical - set(class_names)
    if unknown:
        raise ValueError(f"critical classes not in the class list: {sorted(unknown)}")
    if boost < 1.0:
        raise ValueError(f"boost must be >= 1.0, got {boost}")
    return {name: (float(boost) if name in critical else 1.0) for name in class_names}


def _class_weight_key(estimator: Any) -> str | None:
    """The fit param name that carries ``class_weight``, or None if there is none.

    Searches nested params too, because some estimators wrap the classifier that actually
    takes the weights (``CalibratedClassifierCV`` puts it at ``estimator__class_weight``).
    Returning the *path* rather than a bool means the caller can set it, and means a model
    that cannot be weighted is detected rather than silently trained unweighted.
    """
    try:
        params = estimator.get_params()
    except AttributeError:
        return None
    for key in ("class_weight", *(k for k in params if k.endswith("__class_weight"))):
        if key in params:
            return key
    return None


def supports_class_weight(estimator: Any) -> bool:
    """True when ``class_weight`` can be set somewhere on this estimator."""
    return _class_weight_key(estimator) is not None


def sample_weights_for(
    y: Sequence[str],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Per-row weights for the estimators that take ``sample_weight`` instead."""
    missing = sorted({str(label) for label in y} - set(weights))
    if missing:
        raise ValueError(f"no weight defined for label(s): {missing}")
    return np.asarray([float(weights[str(label)]) for label in y], dtype=np.float64)


def weighted_variant(spec: CandidateSpec, weights: Mapping[str, float] | None) -> CandidateSpec:
    """``spec`` with the class weights baked into its parameter grid.

    Weighting is a *candidate dimension*, not a global setting, precisely so the report can
    say what weighting cost and bought: the harness trains the weighted and unweighted
    version of every model and both appear in the comparison table.
    """
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


# --------------------------------------------------------------------------------------
# Building the estimator
# --------------------------------------------------------------------------------------


def build_estimator(spec: CandidateSpec, params: Mapping[str, Any] | None = None, seed: int = 0) -> Any:
    """A ready-to-fit estimator, with the spec's preprocessing composed in front when needed.

    ``class_weight`` is stripped for estimators that do not accept it, so a weighted grid
    can be applied uniformly and the weighting silently travels via ``sample_weight`` in
    :func:`fit_estimator` instead. That keeps ``weighted_variant`` usable for every
    candidate rather than only the ones whose API happens to match.
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
    """Fit, routing class weights to whichever mechanism the estimator understands.

    Three cases, and getting this wrong is invisible -- the model trains happily and just
    ignores the weighting:

    * a Pipeline whose final step takes ``class_weight`` -> already applied by
      :func:`build_estimator`; nothing to do here.
    * an estimator that takes ``sample_weight`` (GBM, XGBoost) -> pass row weights.
    * anything else -> warn loudly rather than pretend the weighting happened.
    """
    import warnings

    target = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator

    if class_weights:
        if supports_class_weight(target):
            # Already baked in by build_estimator; re-applying via sample_weight would
            # double-count the boost.
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
    """The fitted inner estimator, for introspection (importances, support vectors)."""
    return getattr(estimator, "named_steps", {}).get("estimator", estimator)


def feature_importances(estimator: Any, columns: Sequence[str]) -> dict[str, float]:
    """Ranked feature importances, or an empty dict for models that do not expose them.

    An empty dict is returned rather than fabricated values -- an SVM has no
    ``feature_importances_`` and inventing one from coefficient magnitudes would be a
    different quantity wearing the same name.
    """
    inner = estimator_coefficients(estimator)
    importances = getattr(inner, "feature_importances_", None)
    if importances is None:
        return {}
    pairs = sorted(zip(columns, (float(v) for v in importances)), key=lambda kv: -kv[1])
    return dict(pairs)


def describe_zoo() -> list[dict[str, Any]]:
    """What the comparison report lists as 'the models we trained and compared'."""
    return [
        {
            "name": spec.name,
            "preprocess": spec.preprocess,
            "param_grid": {k: list(v) for k, v in spec.param_grid.items()},
            "notes": spec.notes,
        }
        for spec in CLASSICAL_CANDIDATES.values()
    ]
