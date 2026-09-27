"""Tuning and comparison code shared by every candidate model.

Every model gets the same rows, split, preprocessing, metrics and class list.

- Selection uses train and validation only; ``finalize`` scores the winner on test once.
- The winner has the best validation macro-F1 among those meeting the critical-recall floor.
- Each trial's parameters, seed, metrics and fit time go to a JSONL log.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from src.training.evaluation import (
    EvaluationResult,
    compute_metrics,
)


class TuningError(RuntimeError):
    """Raised when a tuning run is set up wrong (e.g. test data in selection)."""


# Data containers


@dataclass
class Trial:
    """One candidate scored on validation."""

    candidate: str
    params: dict[str, Any]
    seed: int
    val_macro_f1: float
    val_accuracy: float
    val_critical_recall: float
    fit_seconds: float
    score_seconds: float
    predictions: np.ndarray | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "params": _jsonable(self.params),
            "seed": self.seed,
            "val_macro_f1": self.val_macro_f1,
            "val_accuracy": self.val_accuracy,
            "val_critical_recall": self.val_critical_recall,
            "fit_seconds": round(self.fit_seconds, 3),
            "score_seconds": round(self.score_seconds, 3),
            "note": self.note,
        }


@dataclass
class Selection:
    """Tuning result: the winner and all the trials."""

    winner: str
    params: dict[str, Any]
    seed: int
    criterion: str
    trials: list[Trial] = field(default_factory=list)
    rationale: str = ""

    @property
    def ranked(self) -> list[Trial]:
        return sorted(
            self.trials,
            key=lambda t: (-t.val_macro_f1, -t.val_accuracy, t.fit_seconds),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "params": _jsonable(self.params),
            "seed": self.seed,
            "criterion": self.criterion,
            "rationale": self.rationale,
            "n_trials": len(self.trials),
            "trials": [t.to_dict() for t in self.trials],
        }


# Leakage checks


def assert_no_test_peeking(split_used_for_selection: str) -> None:
    """Raise if selection would use anything but train/validation."""
    normalised = str(split_used_for_selection).strip().lower()
    if normalised not in ("val", "validation", "train"):
        raise TuningError(
            f"refusing to select a model using the {split_used_for_selection!r} split. "
            "Selection must use train/validation only; the test split is scored once, at "
            "the end, or the reported accuracy is not a held-out estimate."
        )


def assert_disjoint_train_val(
    train_ids: Iterable[str], val_ids: Iterable[str], context: str = ""
) -> None:
    """Assert the training and validation row sets do not overlap."""
    train_set, val_set = set(train_ids), set(val_ids)
    overlap = train_set & val_set
    if overlap:
        where = f" ({context})" if context else ""
        raise TuningError(
            f"{len(overlap)} audio_id(s) appear in both train and validation{where}, "
            f"e.g. {sorted(overlap)[:5]}. That is leakage inside the tuning loop."
        )


# Feature matrix assembly


def build_feature_matrix(
    records: Sequence[Mapping[str, Any]],
    extractor: Any,
    *,
    cache_path: str | Path | None = None,
    key_field: str = "audio_id",
    progress: bool = False,
    unusable: list[dict[str, Any]] | None = None,
    on_unusable: str = "skip",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Features and labels for some records, cached by audio_id and feature version.

    Returns ``(X, y, audio_ids)``. Rejected recordings are added to ``unusable`` with the
    reason (or raise with ``on_unusable="raise"``).
    """
    from python_models.dataset import resolve_audio_path

    if on_unusable not in ("skip", "raise"):
        raise TuningError(f"on_unusable must be 'skip' or 'raise', got {on_unusable!r}")

    cache: dict[str, list[float]] = {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("feature_version") != getattr(extractor, "feature_version", None):
            raise TuningError(
                f"feature cache {cache_file} was built with feature_version "
                f"{cached.get('feature_version')!r} but the extractor is "
                f"{getattr(extractor, 'feature_version', None)!r}. Rebuild the cache."
            )
        cache = {k: list(v) for k, v in cached.get("vectors", {}).items()}

    rows: list[np.ndarray] = []
    labels: list[str] = []
    ids: list[str] = []
    dirty = False

    for i, record in enumerate(records):
        audio_id = str(record[key_field])
        label = str(record["class_label"])
        vector = cache.get(audio_id)
        if vector is None:
            path = resolve_audio_path(record)
            try:
                vector = [float(v) for v in extractor.extract_from_file(path)]
            except Exception as exc:  # AudioRejected or anything else
                if on_unusable == "raise":
                    raise
                reason = getattr(exc, "reason", None) or str(exc)
                if unusable is not None:
                    unusable.append(
                        {
                            "audio_id": audio_id,
                            "class_label": label,
                            "split": record.get("dataset_split", ""),
                            "reason": str(reason),
                        }
                    )
                continue
            cache[audio_id] = vector
            dirty = True
        rows.append(np.asarray(vector, dtype=np.float32))
        labels.append(label)
        ids.append(audio_id)
        if progress and (i + 1) % 50 == 0:
            print(f"  features: {i + 1}/{len(records)}", flush=True)

    if not rows:
        raise TuningError("no records to extract features from")

    if unusable is not None and len(unusable):
        lost_classes = sorted({u["class_label"] for u in unusable})
        present = set(labels)
        emptied = [c for c in lost_classes if c not in present]
        if emptied:
            raise TuningError(
                "the quality gate rejected every recording of "
                + ", ".join(emptied)
                + "; a class cannot be trained on nothing. Fix the corpus rather than "
                "training on a class-less split."
            )

    expected = getattr(extractor, "columns", None) or []
    if expected and len(expected) != rows[0].size:
        raise TuningError(
            f"extractor declared {len(expected)} columns but produced "
            f"{rows[0].size} values per row"
        )

    if cache_file and dirty:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {
                    "feature_version": getattr(extractor, "feature_version", None),
                    "columns": list(getattr(extractor, "columns", []) or []),
                    "vectors": cache,
                }
            ),
            encoding="utf-8",
        )

    return np.vstack(rows), np.asarray(labels, dtype=object), ids


# Protocol


@dataclass
class TuningProtocol:
    """Fit on train, select on validation, score on test once.

    ``fit(X, y, params, seed, class_weights)`` returns an estimator and
    ``predict_proba_of(est, X)`` an ``(n_rows, n_classes)`` array.
    """

    class_names: Sequence[str]
    critical_classes: Sequence[str] = ()
    fit: Callable[..., Any] | None = None
    predict_proba_of: Callable[[Any, np.ndarray], np.ndarray] | None = None
    selection_metric: str = "macro_f1"
    critical_recall_floor: float | None = 0.85
    X_train: np.ndarray | None = None
    y_train: Sequence[str] | None = None
    train_ids: Sequence[str] | None = None
    X_val: np.ndarray | None = None
    y_val: Sequence[str] | None = None
    val_ids: Sequence[str] | None = None
    seeds: Sequence[int] = (0,)
    log_path: str | Path | None = None
    trials: list[Trial] = field(default_factory=list)


    def __post_init__(self) -> None:
        if self.X_train is not None and self.train_ids is not None and self.val_ids is not None:
            assert_disjoint_train_val(self.train_ids, self.val_ids, context="tuning protocol")

    def attach_data(
        self,
        *,
        X_train: np.ndarray,
        y_train: Sequence[str],
        train_ids: Sequence[str],
        X_val: np.ndarray,
        y_val: Sequence[str],
        val_ids: Sequence[str],
    ) -> "TuningProtocol":
        """Attach the feature matrices."""
        assert_disjoint_train_val(train_ids, val_ids, context="tuning protocol")
        if X_train.shape[1] != X_val.shape[1]:
            raise TuningError(
                f"train features have {X_train.shape[1]} columns but validation has "
                f"{X_val.shape[1]}; they were extracted with different settings"
            )
        self.X_train, self.y_train, self.train_ids = X_train, list(y_train), list(train_ids)
        self.X_val, self.y_val, self.val_ids = X_val, list(y_val), list(val_ids)
        if len(self.y_train) != X_train.shape[0] or len(self.y_val) != X_val.shape[0]:
            raise TuningError("label count does not match feature row count")
        return self


    def _predictions(self, estimator: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted classes and the confidence matrix, in ``class_names`` order.

        predict_proba columns follow the estimator's ``classes_``, not the config order.
        """
        proba = np.asarray(self.predict_proba_of(estimator, X))
        if proba.ndim != 2 or proba.shape[1] != len(self.class_names):
            raise TuningError(
                f"predict_proba_of returned shape {proba.shape}; expected "
                f"(n_rows, {len(self.class_names)}) matching the configured class list"
            )
        learned = list(getattr(estimator, "classes_", []))
        if learned and len(learned) == proba.shape[1]:
            # Reorder columns to class_names.
            order = [learned.index(str(name)) for name in self.class_names]
            proba = proba[:, order]
        # No classes_: assume class_names order.
        idx = np.argmax(proba, axis=1)
        names = np.asarray(self.class_names, dtype=object)
        return names[idx], proba

    def evaluate_on_val(self, estimator: Any, trial: Trial) -> Trial:
        """Score a fitted estimator on validation."""
        started = time.perf_counter()
        predicted, _ = self._predictions(estimator, self.X_val)
        trial.score_seconds = time.perf_counter() - started

        result = compute_metrics(
            list(self.y_val),
            [str(p) for p in predicted],
            self.class_names,
            self.critical_classes,
            model_name=trial.candidate,
            model_version="tuning",
            split="val",
        )
        trial.val_macro_f1 = result.macro_f1
        trial.val_accuracy = result.accuracy
        trial.val_critical_recall = result.critical_recall
        trial.predictions = predicted
        return trial


    def select(
        self,
        candidates: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        class_weights: Mapping[str, float] | None = None,
        split_used_for_selection: str = "val",
    ) -> Selection:
        """Score each ``(name, params)`` candidate on validation and pick the winner."""
        assert_no_test_peeking(split_used_for_selection)
        if self.fit is None or self.predict_proba_of is None:
            raise TuningError("protocol has no fit/predict_proba_of callables")
        if self.X_train is None or self.X_val is None:
            raise TuningError("protocol has no data; call attach_data() first")

        self.trials = []
        for seed in self.seeds:
            for name, params in candidates:
                trial = Trial(
                    candidate=name,
                    params=dict(params),
                    seed=int(seed),
                    val_macro_f1=float("nan"),
                    val_accuracy=float("nan"),
                    val_critical_recall=float("nan"),
                    fit_seconds=0.0,
                    score_seconds=0.0,
                )
                started = time.perf_counter()
                estimator = self.fit(
                    self.X_train, list(self.y_train), dict(params), int(seed), class_weights
                )
                trial.fit_seconds = time.perf_counter() - started
                self.evaluate_on_val(estimator, trial)
                self.trials.append(trial)
                self._log(trial)
                print(
                    f"  {name:<22} seed={seed}  val_macro_f1={trial.val_macro_f1:.4f}  "
                    f"acc={trial.val_accuracy:.4f}  crit_recall={trial.val_critical_recall:.4f}  "
                    f"fit={trial.fit_seconds:.1f}s",
                    flush=True,
                )

        ranked = [
            t
            for t in sorted(
                self.trials,
                key=lambda t: (-getattr(t, f"val_{self.selection_metric}", t.val_macro_f1), t.fit_seconds),
            )
        ]
        eligible = (
            [t for t in ranked if t.val_critical_recall + 1e-12 >= self.critical_recall_floor]
            if self.critical_recall_floor is not None
            else ranked
        )
        note = ""
        if not eligible:
            note = (
                f"no candidate met the critical-recall floor of {self.critical_recall_floor}; "
                "selecting the best available on macro-F1 and flagging this in the report"
            )
            eligible = ranked

        best = eligible[0]
        rationale = (
            f"selected {best.candidate} on validation {self.selection_metric}="
            f"{getattr(best, f'val_{self.selection_metric}', best.val_macro_f1):.4f} "
            f"(val accuracy {best.val_accuracy:.4f}, critical recall "
            f"{best.val_critical_recall:.4f}) out of {len(self.trials)} trials; "
            f"test split not used for selection"
        )
        if note:
            rationale += f". {note}"

        return Selection(
            winner=best.candidate,
            params=dict(best.params),
            seed=int(best.seed),
            criterion=self.selection_metric,
            trials=list(self.trials),
            rationale=rationale,
        )

    def _log(self, trial: Trial) -> None:
        if not self.log_path:
            return
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trial.to_dict()) + "\n")


# Test evaluation


@dataclass
class FinalModel:
    """The refitted winner and its test result."""

    estimator: Any
    selection: Selection
    result: EvaluationResult
    rows: list[dict[str, Any]]
    wall_clock_sec: float
    fitted_on: str = "train"


def finalize(
    protocol: TuningProtocol,
    selection: Selection,
    estimator: Any,
    *,
    X_test: np.ndarray,
    y_test: Sequence[str],
    test_ids: Sequence[str],
    test_records: Sequence[Mapping[str, Any]] | None = None,
    model_version: str = "",
    fitted_on: str = "train",
) -> FinalModel:
    """Score the selected model on the test split.

    ``fitted_on`` records whether it was trained on train or train+val.
    """
    if fitted_on not in ("train", "train+val"):
        raise TuningError(f"unknown fitted_on {fitted_on!r}; expected 'train' or 'train+val'")
    if X_test.shape[0] != len(y_test) or X_test.shape[0] != len(test_ids):
        raise TuningError(
            f"test matrix has {X_test.shape[0]} rows but there are {len(y_test)} labels and "
            f"{len(test_ids)} audio ids; they should be the same length"
        )
    if test_records is not None:
        # Only frozen test rows, no augmented audio.
        from src.training.evaluation import assert_reportable_split

        assert_reportable_split(test_records, "test", context=f"finalize({selection.winner})")

    started = time.perf_counter()
    predicted, proba = protocol._predictions(estimator, X_test)
    order = np.argsort(-proba, axis=1)
    top2 = [[str(np.asarray(protocol.class_names, dtype=object)[j]) for j in row[:2]] for row in order]

    result = compute_metrics(
        list(y_test),
        [str(p) for p in predicted],
        protocol.class_names,
        protocol.critical_classes,
        model_name=selection.winner,
        model_version=model_version,
        split="test",
        top2=top2,
        audio_ids=list(test_ids),
        config_snapshot={
            "selection": selection.to_dict(),
            "fitted_on": fitted_on,
            "critical_recall_floor": protocol.critical_recall_floor,
        },
    )

    rows: list[dict[str, Any]] = []
    for i, audio_id in enumerate(test_ids):
        confidences = {
            str(name): float(proba[i, j]) for j, name in enumerate(protocol.class_names)
        }
        rows.append(
            {
                "audio_id": str(audio_id),
                "actual_class": str(y_test[i]),
                "predicted_class": str(predicted[i]),
                "confidence": float(np.max(proba[i])),
                "confidences": confidences,
                "correct": str(predicted[i]) == str(y_test[i]),
                "split": "test",
            }
        )

    return FinalModel(
        estimator=estimator,
        selection=selection,
        result=result,
        rows=rows,
        wall_clock_sec=time.perf_counter() - started,
        fitted_on=fitted_on,
    )


# Refit on train+val


def refit_on_train_and_val(
    protocol: TuningProtocol,
    selection: Selection,
    class_weights: Mapping[str, float] | None = None,
) -> Any:
    """Refit the selected configuration on train+val (after ``select``)."""
    if protocol.X_train is None or protocol.X_val is None:
        raise TuningError("protocol has no data to refit on")
    X = np.vstack([protocol.X_train, protocol.X_val])
    y = list(protocol.y_train) + list(protocol.y_val)
    return protocol.fit(X, y, dict(selection.params), int(selection.seed), class_weights)


# Cross-validation of the winner


def cross_validate_selected(
    protocol: TuningProtocol,
    selection: Selection,
    *,
    folds: int = 5,
    class_weights: Mapping[str, float] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Stratified k-fold CV of the winner on the training rows, to see the variance."""
    from sklearn.model_selection import StratifiedKFold

    if protocol.X_train is None:
        raise TuningError("protocol has no training data")
    X, y = protocol.X_train, list(protocol.y_train)
    if len(set(y)) < 2:
        raise TuningError("cross-validation needs at least two classes in the training set")

    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=seed)
    scores: list[float] = []
    accs: list[float] = []
    crits: list[float] = []

    for fold, (tr, va) in enumerate(splitter.split(X, y), start=1):
        estimator = protocol.fit(X[tr], [y[i] for i in tr], dict(selection.params), int(selection.seed), class_weights)
        predicted, _ = protocol._predictions(estimator, X[va])
        result = compute_metrics(
            [y[i] for i in va],
            [str(p) for p in predicted],
            protocol.class_names,
            protocol.critical_classes,
            model_name=f"{selection.winner}-cv{fold}",
            model_version="cv",
            split="train",
        )
        scores.append(result.macro_f1)
        accs.append(result.accuracy)
        crits.append(result.critical_recall)
        print(
            f"  cv fold {fold}/{folds}: macro_f1={result.macro_f1:.4f} acc={result.accuracy:.4f}",
            flush=True,
        )

    arr = np.asarray(scores)
    return {
        "folds": int(folds),
        "macro_f1_mean": float(arr.mean()),
        "macro_f1_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "macro_f1_scores": [float(v) for v in scores],
        "accuracy_mean": float(np.mean(accs)),
        "critical_recall_mean": float(np.mean(crits)),
        "seed": int(seed),
    }


# Helpers


def _jsonable(value: Any) -> Any:
    """Make params JSON-safe."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def comparable_table(results: Sequence[EvaluationResult]) -> list[dict[str, Any]]:
    """One row per model with all the reported columns."""
    table: list[dict[str, Any]] = []
    for result in results:
        table.append(
            {
                "model": result.model_name,
                "version": result.model_version,
                "split": result.split,
                "n": result.n_records,
                "accuracy": result.accuracy,
                "macro_precision": result.macro_precision,
                "macro_recall": result.macro_recall,
                "macro_f1": result.macro_f1,
                "weighted_f1": result.weighted_f1,
                "critical_recall": result.critical_recall,
                "top2_accuracy": result.top2_accuracy,
                "worst_class": (result.worst_classes(1) or [{}])[0].get("class_name", ""),
            }
        )
    return table


def assert_comparable(results: Sequence[EvaluationResult], *, split: str = "test") -> None:
    """Check that all results used the same split and row count."""
    if not results:
        raise TuningError("nothing to compare")
    splits = {r.split for r in results}
    if splits != {split}:
        raise TuningError(
            f"comparison mixes splits {sorted(splits)}; all rows must come from {split!r}"
        )
    counts = {r.n_records for r in results}
    if len(counts) != 1:
        raise TuningError(
            f"comparison mixes row counts {sorted(counts)}; every model must be scored on "
            "the identical set of recordings"
        )
    classes = {tuple(r.class_names) for r in results}
    if len(classes) != 1:
        raise TuningError("comparison mixes class lists; the models are not on equal terms")
