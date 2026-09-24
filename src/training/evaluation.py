"""Evaluation core: every number in the report comes from here, or it does not go in.

Three rules this module exists to enforce.

1. ONE SOURCE OF TRUTH FOR THE SPLIT. Metrics are computed only over records whose
   `dataset_split` matches the split being reported. `assert_reportable_split` refuses
   anything else, so "test accuracy" cannot be quietly computed over val, train, or a
   mixture — the mistake that makes a project look finished and be worthless.

2. PROBES ARE NOT RESULTS. A noise-robustness run scores DEGRADED audio. Those numbers
   are real and worth having, and they are NOT test accuracy. Every result carries
   `probe` / `probe_label`, and `meets_floors` refuses to certify a probe. The headline
   figure is always clean, frozen, unseen data.

3. NO HARD-CODED METRIC. Accuracy, macro-F1, critical recall and the floors they are
   tested against all derive from the confusion matrix and from config. Nothing here
   knows what a good model looks like.

The confusion matrix is computed here rather than imported, with the label ordering made
explicit, because "which axis was which class" is the single most common way a
classification report ends up backwards. `test_evaluation_metrics.py` asserts this
implementation agrees with scikit-learn's on identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

SPLIT_NAMES = ("train", "val", "test")


class EvaluationError(RuntimeError):
    """Raised when an evaluation would produce a number that cannot be trusted."""


# --------------------------------------------------------------------------------------
# The leakage guard
# --------------------------------------------------------------------------------------

def assert_reportable_split(
    records: Iterable[Mapping[str, Any]],
    split: str,
    *,
    allow_augmented: bool = False,
    context: str = "",
) -> int:
    """Refuse to evaluate anything but the frozen split, clean.

    Called before every evaluation. The failure it prevents is silent: training on the
    test set, or reporting augmented clips as unseen performance, both produce a good
    number and a worthless project.
    """
    if split not in SPLIT_NAMES:
        raise EvaluationError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")

    where = f" ({context})" if context else ""
    bad_split: list[str] = []
    augmented: list[str] = []
    n = 0

    for record in records:
        n += 1
        actual = str(record.get("dataset_split", "")).strip()
        if actual != split:
            bad_split.append(f"{record.get('audio_id')}={actual or '<empty>'}")

        status = str(record.get("original_or_augmented", "original")).strip().lower()
        if status.startswith("aug") and not allow_augmented:
            augmented.append(str(record.get("audio_id")))

    if bad_split:
        raise EvaluationError(
            f"refusing to report {split} metrics{where}: {len(bad_split)} of {n} records are "
            f"not in the {split} split, e.g. {bad_split[:5]}. Metrics computed over the wrong "
            "split are the most expensive kind of wrong — they look like progress."
        )
    if augmented:
        raise EvaluationError(
            f"refusing to report {split} metrics{where}: {len(augmented)} record(s) are "
            f"augmented, e.g. {augmented[:5]}. Augmented audio is not unseen data. "
            "Use a probe (probe=True) if you want a robustness figure, and label it as one."
        )
    if n == 0:
        raise EvaluationError(f"refusing to report {split} metrics{where}: no records")
    return n


# --------------------------------------------------------------------------------------
# Confusion matrix — written out so the axis order is not a convention we inherited
# --------------------------------------------------------------------------------------

def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    """Rows = actual, columns = predicted, in exactly `labels` order.

    `labels` is required and ordered by the caller. Nothing here guesses an ordering from
    the data, because a matrix whose axes were sorted alphabetically by accident produces
    per-class numbers attached to the wrong classes — and still sums correctly, so it
    survives inspection.
    """
    index = {name: i for i, name in enumerate(labels)}
    if len(index) != len(list(labels)):
        raise EvaluationError("labels contains duplicates")
    matrix = np.zeros((len(index), len(index)), dtype=np.int64)

    for actual, predicted in zip(y_true, y_pred):
        if actual not in index:
            raise EvaluationError(
                f"actual label {actual!r} is not one of the {len(index)} configured classes"
            )
        if predicted not in index:
            raise EvaluationError(
                f"predicted label {predicted!r} is not one of the {len(index)} configured classes"
            )
        matrix[index[actual], index[predicted]] += 1
    return matrix


def per_class_scores(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Precision / recall / F1 / support from the matrix, zero-safe.

    A class with no predictions has undefined precision; it is reported as 0.0, not 1.0
    and not skipped. Skipping it would hide a class the model has stopped predicting,
    which is exactly the failure a 10-class mandate needs to catch.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    true_positive = np.diag(matrix)
    predicted_total = matrix.sum(axis=0)   # column sums
    actual_total = matrix.sum(axis=1)      # row sums

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted_total > 0, true_positive / predicted_total, 0.0)
        recall = np.where(actual_total > 0, true_positive / actual_total, 0.0)
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual_total.astype(np.int64),
        "predicted": predicted_total.astype(np.int64),
        "true_positive": true_positive.astype(np.int64),
    }


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    model_name: str
    model_version: str
    split: str
    n_records: int
    class_names: list[str]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: list[dict[str, Any]]
    confusion_matrix: list[list[int]]
    critical_classes: list[str]
    critical_recall: float
    critical_recall_by_class: dict[str, float]
    top2_accuracy: float | None = None
    severe_errors: list[dict[str, Any]] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    probe: bool = False
    probe_label: str = ""

    # ---- floors -----------------------------------------------------------------------

    def meets_floors(self, floors: Mapping[str, float]) -> tuple[bool, list[str]]:
        """Check the SRS floors. A probe can never be certified.

        A robustness probe measures how gracefully the model degrades; it says nothing
        about the accuracy floor, which is defined on unseen clean data.
        """
        if self.probe:
            return False, [
                f"this is a probe ({self.probe_label}), not a test-set measurement; "
                "the accuracy and F1 floors are defined on clean unseen audio"
            ]

        failures: list[str] = []
        checks = (
            ("accuracy", self.accuracy, floors.get("accuracy")),
            ("macro_f1", self.macro_f1, floors.get("macro_f1")),
            ("critical_recall", self.critical_recall, floors.get("critical_recall")),
        )
        for name, actual, required in checks:
            if required is None:
                continue
            if actual + 1e-12 < float(required):
                failures.append(
                    f"{name}: {actual:.4f} < {float(required):.4f} required"
                )
        return not failures, failures

    def worst_classes(self, k: int = 3) -> list[dict[str, Any]]:
        """The classes dragging macro-F1 down, worst first."""
        return sorted(self.per_class, key=lambda c: (c["f1"], c["support"]))[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "split": self.split,
            "n_records": self.n_records,
            "probe": self.probe,
            "probe_label": self.probe_label,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "critical_classes": list(self.critical_classes),
            "critical_recall": self.critical_recall,
            "critical_recall_by_class": dict(self.critical_recall_by_class),
            "top2_accuracy": self.top2_accuracy,
            "per_class": [dict(c) for c in self.per_class],
            "confusion_matrix": self.confusion_matrix,
            "class_names": list(self.class_names),
            "severe_errors": [dict(e) for e in self.severe_errors],
            "config_snapshot": dict(self.config_snapshot),
        }


# --------------------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------------------

def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_names: Sequence[str],
    critical_classes: Sequence[str] = (),
    *,
    model_name: str = "model",
    model_version: str = "",
    split: str = "test",
    top2: Sequence[Sequence[str]] | None = None,
    audio_ids: Sequence[str] | None = None,
    config_snapshot: Mapping[str, Any] | None = None,
    probe: bool = False,
    probe_label: str = "",
) -> EvaluationResult:
    """Turn two label sequences into every number the SRS asks a model to be judged by.

    `class_names` is the full configured class list and is used for ALL averaging, so a
    class the model never predicts still counts against macro-F1 with a recall of zero.
    Averaging only over classes that happen to appear would let a model that ignores a
    hard class report a flattering macro-F1.
    """
    if len(y_true) != len(y_pred):
        raise EvaluationError(
            f"got {len(y_true)} actual labels but {len(y_pred)} predictions"
        )
    if not y_true:
        raise EvaluationError("no predictions to evaluate")

    labels = list(class_names)
    matrix = confusion_matrix(y_true, y_pred, labels)
    scores = per_class_scores(matrix)

    n = len(y_true)
    accuracy = float(np.trace(matrix) / n)
    macro_precision = float(np.mean(scores["precision"]))
    macro_recall = float(np.mean(scores["recall"]))
    macro_f1 = float(np.mean(scores["f1"]))

    support = scores["support"].astype(np.float64)
    total_support = float(support.sum()) or 1.0
    weighted_f1 = float(np.sum(scores["f1"] * support) / total_support)

    per_class = [
        {
            "class_name": name,
            "index": i,
            "support": int(scores["support"][i]),
            "predicted": int(scores["predicted"][i]),
            "true_positive": int(scores["true_positive"][i]),
            "precision": float(scores["precision"][i]),
            "recall": float(scores["recall"][i]),
            "f1": float(scores["f1"][i]),
            "is_critical": name in set(critical_classes),
        }
        for i, name in enumerate(labels)
    ]

    # Critical-event recall: the one number that decides whether this system is worth
    # deploying. A missed gunshot is not a slightly worse gunshot prediction.
    critical_recall_by_class: dict[str, float] = {}
    for entry in per_class:
        if entry["is_critical"]:
            critical_recall_by_class[entry["class_name"]] = entry["recall"]
    critical_recall = (
        float(np.mean(list(critical_recall_by_class.values())))
        if critical_recall_by_class
        else 0.0
    )

    top2_accuracy = None
    if top2 is not None:
        if len(top2) != n:
            raise EvaluationError(
                f"top2 must align with y_true ({n} rows), got {len(top2)}"
            )
        hits = sum(
            1
            for actual, row in zip(y_true, top2)
            if row and actual in list(row)[:2]
        )
        top2_accuracy = float(hits / n)

    # Severe errors: a critical event the system failed to flag as critical.
    # These are ranked by consequence, not by count: a gunshot called "Vehicle Horn" is a
    # false alarm, a gunshot called "Background Noise" is a missed emergency. The report
    # must distinguish them, because only one of those is survivable in deployment.
    critical_set = set(critical_classes)
    severe_errors: list[dict[str, Any]] = []
    if critical_set:
        ids = list(audio_ids) if audio_ids is not None else [f"row-{i}" for i in range(n)]
        for i, (actual, predicted) in enumerate(zip(y_true, y_pred)):
            if actual in critical_set and predicted != actual:
                still_critical = predicted in critical_set
                severe_errors.append(
                    {
                        "audio_id": str(ids[i]),
                        "actual": actual,
                        "predicted": predicted,
                        # True when the system would have raised no alert at all.
                        "silent_miss": not still_critical,
                        # True when it alerted, but with the wrong label: a false alarm.
                        "misattributed_alert": still_critical,
                    }
                )
        # Silent misses first — those are the ones that cost lives.
        severe_errors.sort(key=lambda e: (not e["silent_miss"], e["actual"]))

    return EvaluationResult(
        model_name=model_name,
        model_version=model_version,
        split=split,
        n_records=n,
        class_names=labels,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class=per_class,
        confusion_matrix=matrix.tolist(),
        critical_classes=list(critical_classes),
        critical_recall=critical_recall,
        critical_recall_by_class=critical_recall_by_class,
        top2_accuracy=top2_accuracy,
        severe_errors=severe_errors,
        config_snapshot=dict(config_snapshot or {}),
        probe=probe,
        probe_label=probe_label,
    )


def evaluate_predictions(
    records: Sequence[Mapping[str, Any]],
    predict: Callable[[Mapping[str, Any]], Any],
    class_names: Sequence[str],
    critical_classes: Sequence[str] = (),
    *,
    label_field: str = "class_label",
    split: str = "test",
    model_name: str = "model",
    model_version: str = "",
    config_snapshot: Mapping[str, Any] | None = None,
    require_frozen_split: bool = True,
    allow_augmented: bool = False,
    probe: bool = False,
    probe_label: str = "",
) -> tuple[EvaluationResult, list[dict[str, Any]]]:
    """Run `predict` over `records` and compute the metrics.

    `predict` returns either a class name, or a Mapping with `predicted_class` and
    `confidences` — the shape of `PredictionResult.to_dict()`, so the same harness drives
    the Python model, the GTM model, a baseline and a probe without special cases.

    Returns the metrics and the per-record predictions, because the comparison report
    needs both the summary and the row-level evidence behind it.
    """
    if require_frozen_split:
        assert_reportable_split(
            records,
            split,
            allow_augmented=allow_augmented,
            context=f"evaluating {model_name}",
        )

    y_true: list[str] = []
    y_pred: list[str] = []
    top2: list[list[str]] = []
    audio_ids: list[str] = []
    rows: list[dict[str, Any]] = []

    for record in records:
        actual = str(record.get(label_field, "")).strip()
        if not actual:
            raise EvaluationError(
                f"record {record.get('audio_id')} has no {label_field!r}"
            )
        outcome = predict(record)

        if isinstance(outcome, str):
            predicted, confidences = outcome, {}
        elif isinstance(outcome, Mapping):
            predicted = str(outcome.get("predicted_class", "")).strip()
            confidences = dict(outcome.get("confidences", {}) or {})
        else:
            predicted = str(getattr(outcome, "predicted_class", "")).strip()
            confidences = dict(getattr(outcome, "confidences", {}) or {})

        if not predicted:
            raise EvaluationError(
                f"predict returned nothing for {record.get('audio_id')}"
            )
        if predicted not in set(class_names):
            raise EvaluationError(
                f"predicted class {predicted!r} is not a configured class"
            )

        ranking = (
            [c for c, _ in sorted(confidences.items(), key=lambda kv: -kv[1])]
            if confidences
            else [predicted]
        )

        y_true.append(actual)
        y_pred.append(predicted)
        top2.append(ranking[:2])
        audio_ids.append(str(record.get("audio_id", "")))
        rows.append(
            {
                "audio_id": str(record.get("audio_id", "")),
                "filename": str(record.get("filename", "")),
                "actual_class": actual,
                "predicted_class": predicted,
                "confidence": float(confidences.get(predicted, float("nan")))
                if confidences
                else None,
                "confidences": confidences,
                "correct": predicted == actual,
                "split": str(record.get("dataset_split", split)),
            }
        )

    result = compute_metrics(
        y_true,
        y_pred,
        class_names,
        critical_classes,
        model_name=model_name,
        model_version=model_version,
        split=split,
        top2=top2,
        audio_ids=audio_ids,
        config_snapshot=config_snapshot,
        probe=probe,
        probe_label=probe_label,
    )
    return result, rows


def load_split_records(
    manifest_path: str,
    split: str,
) -> list[dict[str, str]]:
    """Read the manifest and return only the rows of one split.

    Prefers `manifest_with_split.csv` (the builder's output, which has the authoritative
    assignment). Raises rather than filtering silently, so a wrong path is not mistaken
    for an empty split.
    """
    import csv
    from pathlib import Path

    path = Path(manifest_path)
    if not path.exists():
        raise EvaluationError(f"manifest not found: {path}")

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise EvaluationError(f"{path} has no header row")
        if "dataset_split" not in reader.fieldnames:
            raise EvaluationError(
                f"{path} has no dataset_split column. The split is assigned by "
                "audio_dataset/build_split.py — run it first; do not filter by hand."
            )
        every = list(reader)

    missing = [r.get("audio_id") for r in every if not str(r.get("dataset_split", "")).strip()]
    if missing:
        raise EvaluationError(
            f"{len(missing)} record(s) in {path} have an empty dataset_split, "
            f"e.g. {missing[:5]}. Run audio_dataset/build_split.py to assign it."
        )

    records = [r for r in every if str(r["dataset_split"]).strip() == split]
    if not records:
        raise EvaluationError(f"no records in split {split!r} of {path}")
    return records
