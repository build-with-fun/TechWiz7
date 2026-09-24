"""The one tuning and comparison harness -- every model, classical and deep, is judged here.

Owner: bilal (tuning harness), joint with lorena (selection).

WHY A SINGLE HARNESS
--------------------
Two models can only be compared if everything except the model is held constant: the same
rows, the same split, the same preprocessing, the same metric code, the same class list.
If `nadia` tunes her CNN on one protocol and I tune my SVM on another, the resulting table
is a comparison of two protocols that happen to have models attached. So selection lives
here, once, and both of us call it.

THE THREE WAYS TUNING LIES, AND WHAT THIS MODULE DOES ABOUT THEM
----------------------------------------------------------------
1. **Tuning against test.** The most expensive mistake in the project: it produces a
   headline number that is optimistic by an unknown amount and can never be recovered. So
   this module CANNOT see the test set during selection. :class:`TuningProtocol` is
   constructed from a *validation* evaluator only; :meth:`TuningProtocol.select` scores
   candidates on validation, and :func:`finalize` takes the winner to test exactly once.
   ``assert_no_test_peeking`` is called by ``select`` and refuses a protocol whose
   selection metric came from the test split -- including one assembled by accident by
   evaluating on a superset of the splits.

2. **Selecting on the metric you are judged on, then reporting it.** Selection uses
   validation macro-F1 (and the critical-recall floor as a constraint); the test number is
   reported once, as a held-out estimate. Both are stored, so the report can show
   validation and test side by side and a gap that is large is visible rather than hidden.

3. **Averaging over classes that never appear.** All averaging uses the full configured
   class list, so a class the model stopped predicting scores zero and drags macro-F1 down
   instead of vanishing. That is enforced in ``src.training.evaluation``; this module
   simply must not bypass it.

WHAT A TRIAL RECORDS
--------------------
Every trial keeps its parameters, seed, validation metrics and wall-clock fit time, and is
appended to a JSONL log. That log is the evidence for "we tuned it" and the input to the
report's tuning table -- and it means a surprising result can be traced to the exact run
that produced it rather than argued about.

DEEP MODELS USE THE SAME HARNESS
--------------------------------
``TuningProtocol`` accepts any callable ``fit(X, y) -> estimator`` and any callable
``score(estimator, X, y) -> class predictions``. A CNN training loop wraps in two lambdas
and gets the identical protocol, split guard and metric code as a Random Forest. That is
the whole point.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from src.training.evaluation import (
    EvaluationError,
    EvaluationResult,
    compute_metrics,
)


class TuningError(RuntimeError):
    """Raised when a tuning run would produce an untrustworthy result."""


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------


@dataclass
class Trial:
    """One candidate evaluated on the validation split. The unit of tuning evidence."""

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
    """The outcome of tuning: the winner, everything it beat, and why."""

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


# --------------------------------------------------------------------------------------
# The leakage guard for selection
# --------------------------------------------------------------------------------------


def assert_no_test_peeking(split_used_for_selection: str) -> None:
    """Refuse to select a model on anything but train/validation.

    Called at the top of :meth:`TuningProtocol.select`. The failure it prevents is silent
    and fatal to the project's credibility, so it is an exception rather than a warning.
    """
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
    """Assert the training and validation row sets do not overlap.

    A row in both is memorised and then scored as if it were generalisation, inflating
    validation and steering selection to the model that overfits hardest.
    """
    train_set, val_set = set(train_ids), set(val_ids)
    overlap = train_set & val_set
    if overlap:
        where = f" ({context})" if context else ""
        raise TuningError(
            f"{len(overlap)} audio_id(s) appear in both train and validation{where}, "
            f"e.g. {sorted(overlap)[:5]}. That is leakage inside the tuning loop."
        )


# --------------------------------------------------------------------------------------
# Feature matrix assembly
# --------------------------------------------------------------------------------------


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
    """Features + labels for a record set, with an on-disk cache keyed by audio_id.

    The cache is not an optimisation only: extraction is deterministic and the cache is
    keyed by the frozen ``audio_id`` plus the extractor's ``feature_version``, so the
    matrix a model was trained on and the matrix it is evaluated on are provably the same
    columns in the same order. A stale cache is detected and refused rather than silently
    reusing features from an older feature version.

    Returns ``(X, y, audio_ids)`` with rows in ``records`` order, so the caller's labels
    and the matrix cannot drift apart.

    Recordings the quality gate rejects (silence, excessive noise) are **not** silently
    dropped. Each one is appended to ``unusable`` as
    ``{"audio_id", "class_label", "reason", "split"}`` so the run can report exactly which
    frozen-split recordings did not make it into the matrix -- a number the report has to
    be able to state. ``on_unusable="raise"`` turns the first rejection into an error
    instead, for a caller that would rather fail loudly.
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
            except Exception as exc:  # AudioRejected and anything else from the pipeline
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


# --------------------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------------------


@dataclass
class TuningProtocol:
    """Fit-on-train / select-on-validation / report-on-test-once.

    ``fit`` takes ``(X_train, y_train, params, seed, class_weights)`` and returns a fitted
    estimator. ``predict_proba_of`` takes ``(estimator, X)`` and returns an
    ``(n_rows, n_classes)`` array -- so a deep model that outputs logits or a torch tensor
    is adapted in the caller's two lambdas and then gets the identical treatment.
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

    # -- setup ------------------------------------------------------------------------

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
        """Attach the pre-extracted matrices. Kept separate from construction so a protocol
        can be described (and its guard tested) before any features exist."""
        assert_disjoint_train_val(train_ids, val_ids, context="tuning protocol")
        if X_train.shape[1] != X_val.shape[1]:
            raise TuningError(
                f"train features have {X_train.shape[1]} columns but validation has "
                f"{X_val.shape[1]} -- the two sets were extracted with different settings"
            )
        self.X_train, self.y_train, self.train_ids = X_train, list(y_train), list(train_ids)
        self.X_val, self.y_val, self.val_ids = X_val, list(y_val), list(val_ids)
        if len(self.y_train) != X_train.shape[0] or len(self.y_val) != X_val.shape[0]:
            raise TuningError("label count does not match feature row count")
        return self

    # -- scoring ----------------------------------------------------------------------

    def _predictions(self, estimator: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Class predictions and the full confidence matrix, in ``class_names`` order."""
        proba = np.asarray(self.predict_proba_of(estimator, X))
        if proba.ndim != 2 or proba.shape[1] != len(self.class_names):
            raise TuningError(
                f"predict_proba_of returned shape {proba.shape}; expected "
                f"(n_rows, {len(self.class_names)}) matching the configured class list"
            )
        idx = np.argmax(proba, axis=1)
        names = np.asarray(self.class_names, dtype=object)
        return names[idx], proba

    def evaluate_on_val(self, estimator: Any, trial: Trial) -> Trial:
        """Score one fitted estimator on validation and fill in the trial record."""
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

    # -- the loop ---------------------------------------------------------------------

    def select(
        self,
        candidates: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        class_weights: Mapping[str, float] | None = None,
        split_used_for_selection: str = "val",
    ) -> Selection:
        """Evaluate every ``(name, params)`` candidate on validation and pick a winner.

        The winner maximises the selection metric subject to the critical-recall floor: a
        model that edges ahead on macro-F1 while missing a critical class is not a winner,
        because the SRS's floor on critical recall is a safety requirement, not a
        preference to be traded against an average.
        """
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


# --------------------------------------------------------------------------------------
# Taking the winner to test -- once
# --------------------------------------------------------------------------------------


@dataclass
class FinalModel:
    """A refitted winner plus its one held-out measurement."""

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
    """Score the selected model on the test split. The one and only time.

    ``fitted_on`` is recorded because two honest strategies exist and the report must say
    which was used: fit on train only (test stays fully held out from every tuning
    decision) or refit on train+val after selection (more data, at the cost of validation
    no longer being an independent estimate). It is never allowed to be silent.

    The row count here is also checked against ``len(y_test)`` before anything is scored --
    a mismatch between the label vector and the matrix rows shifts every prediction by one
    and produces a plausible-but-wrong confusion matrix.
    """
    if fitted_on not in ("train", "train+val"):
        raise TuningError(f"unknown fitted_on {fitted_on!r}; expected 'train' or 'train+val'")
    if X_test.shape[0] != len(y_test) or X_test.shape[0] != len(test_ids):
        raise TuningError(
            f"test matrix has {X_test.shape[0]} rows but there are {len(y_test)} labels and "
            f"{len(test_ids)} audio ids -- these must describe the same recordings"
        )
    if test_records is not None:
        # Lorena's guard, applied on the test path specifically. It refuses any record not
        # in the frozen test split and refuses augmented audio, so "unseen data" in the
        # report means unseen data.
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


# --------------------------------------------------------------------------------------
# Refit on train+val
# --------------------------------------------------------------------------------------


def refit_on_train_and_val(
    protocol: TuningProtocol,
    selection: Selection,
    class_weights: Mapping[str, float] | None = None,
) -> Any:
    """Refit the selected configuration on train+val after selection is already made.

    Only legal *after* ``select`` has finished, because the validation split has then
    served its purpose. Using train+val is strictly better for the deployed model, and the
    report records that test is then no longer comparable to validation for this model.
    """
    if protocol.X_train is None or protocol.X_val is None:
        raise TuningError("protocol has no data to refit on")
    X = np.vstack([protocol.X_train, protocol.X_val])
    y = list(protocol.y_train) + list(protocol.y_val)
    return protocol.fit(X, y, dict(selection.params), int(selection.seed), class_weights)


# --------------------------------------------------------------------------------------
# Cross-validation, for a variance estimate on the winner only
# --------------------------------------------------------------------------------------


def cross_validate_selected(
    protocol: TuningProtocol,
    selection: Selection,
    *,
    folds: int = 5,
    class_weights: Mapping[str, float] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Stratified k-fold CV of the winning configuration over *training* rows only.

    Purpose is a variance estimate -- "is this model's advantage over the runner-up bigger
    than its fold-to-fold spread?" -- computed without touching validation or test. The
    fold split is stratified so every fold contains every class; with 2,100 rows and 10
    classes an unstratified fold can leave a class out entirely and produce a meaningless
    zero.
    """
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


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Params out of a search space, JSON-safe (numpy scalars, class-weight dicts, None)."""
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
    """One row per model, every mandated column, straight from the metric objects.

    This is what the model-comparison report's main table is built from, and it is built
    from ``EvaluationResult`` rather than re-parsed from text so the table cannot disagree
    with the artefacts it summarises.
    """
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
    """Assert every result in a comparison table came from the same split and row count.

    A table mixing a model scored on 450 test rows with one scored on 200 is not a
    comparison, and the mixed table looks perfectly normal on the page.
    """
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
