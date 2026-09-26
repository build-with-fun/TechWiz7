"""Label-order contract of TuningProtocol._predictions.

Regression for a real bug: predict_proba columns follow the estimator's own
``classes_`` (alphabetical for sklearn), but the protocol indexed them positionally
against ``class_names`` (config order). Every prediction was relabelled and whole
candidate sweeps scored at chance level -- silently, with no error.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from python_models.tuning import TuningProtocol

CONFIG_ORDER = [
    "Machinery Fault",      # config/classes.json order, deliberately NOT alphabetical
    "Glass Breaking",
    "Alarm or Siren",
    "Vehicle Horn",
]
ALPHABETICAL = sorted(CONFIG_ORDER)


def _dataset(alphabetical_labels: bool):
    rng = np.random.default_rng(0)
    n_per, width = 40, 6
    X, y, idx = [], [], []
    for i, name in enumerate(CONFIG_ORDER):
        # separable clusters so a healthy score is a known constant
        centre = np.full(width, float(i) * 10.0)
        X.append(centre + rng.normal(0, 0.1, (n_per, width)))
        y.extend([name] * n_per)
        idx.extend([f"{name[:3]}-{j:04d}" for j in range(n_per)])
    X = np.vstack(X)
    if alphabetical_labels:
        y = [str(cls) for cls in y]
    # interleave so train/val both see every class (rows are currently grouped by class)
    order = rng.permutation(len(y))
    return X[order], [y[i] for i in order], [idx[i] for i in order]


def _predict_proba(estimator, X):
    return np.asarray(estimator.predict_proba(X))


def _protocol(alphabetical_labels: bool) -> TuningProtocol:
    names = ALPHABETICAL if alphabetical_labels else CONFIG_ORDER
    return TuningProtocol(
        class_names=names,
        critical_classes=("Glass Breaking",),
        fit=_fit(0),
        predict_proba_of=_predict_proba,
        selection_metric="macro_f1",
        critical_recall_floor=None,  # label-order test, not the floor check
    )


def _fit(random_state: int):
    def _fit_impl(X, y, params, seed, class_weights):
        est = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)
        est.fit(X, y)
        return est

    return _fit_impl


def _attach(protocol: TuningProtocol, alphabetical_labels: bool):
    X, y, idx = _dataset(alphabetical_labels)
    cut = int(len(y) * 0.75)
    return protocol.attach_data(
        X_train=X[:cut],
        y_train=y[:cut],
        train_ids=idx[:cut],
        X_val=X[cut:],
        y_val=y[cut:],
        val_ids=idx[cut:],
    )


def test_columns_are_resolved_through_the_estimators_classes():
    """Scores come back in class_names order: argmax is resolved through the estimator's classes_.
    """
    protocol = _attach(_protocol(alphabetical_labels=False), alphabetical_labels=False)
    selection = protocol.select(
        [("rf", {"_candidate": "unused"})],
        class_weights=None,
        split_used_for_selection="val",
    )
    trial = selection.trials[0]
    assert trial.val_accuracy > 0.9, (
        f"positional label scrambling suspected: acc={trial.val_accuracy:.3f} "
        f"(chance for 4 classes is 0.25)"
    )
    proba = np.asarray(trial.predictions_proba) if hasattr(trial, "predictions_proba") else None
    if proba is not None:
        assert proba.shape[1] == len(CONFIG_ORDER)


def test_alphabetical_class_list_still_scores_perfectly():
    """The same dataset with an already-alphabetical class list must be unaffected."""
    protocol = _attach(_protocol(alphabetical_labels=True), alphabetical_labels=True)
    selection = protocol.select(
        [("rf", {"_candidate": "unused"})],
        class_weights=None,
        split_used_for_selection="val",
    )
    assert selection.trials[0].val_accuracy > 0.9


def test_scrambled_estimates_fail_loudly_not_silently():
    """If the mapping is ever wrong again, accuracy lands at chance and the test fails --
    documenting what the bug looked like (0.25 acc with no error raised)."""
    protocol = _attach(_protocol(alphabetical_labels=False), alphabetical_labels=False)
    fit = _fit(0)
    est = fit(protocol.X_train, protocol.y_train, {}, 0, None)
    predicted, proba = protocol._predictions(est, protocol.X_val)
    truth = np.asarray(protocol.y_val)
    acc = float((predicted == truth).mean())
    assert proba.shape[1] == len(CONFIG_ORDER)
    assert acc > 0.9
