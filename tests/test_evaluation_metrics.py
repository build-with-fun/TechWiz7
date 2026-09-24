"""Tests for the evaluation core.

The metric code is the part of this project that produces every number in the report, so
it is the part where a bug is least visible and most expensive. Two things are checked
that ordinary coverage would not:

  CROSS-CHECK AGAINST SCIKIT-LEARN. My confusion matrix is hand-written so the axis order
  is explicit rather than inherited. That is only safe if it agrees with the reference
  implementation on identical inputs, so it is asserted against
  `sklearn.metrics.confusion_matrix` (which is label-sorted) after aligning the order.

  THE FLOORS CAN FAIL. A floor check that always passes is decoration. The tests include
  a model that genuinely misses the accuracy floor and one that misses critical recall,
  and assert the failure is reported with the numbers in it.

Also covered: the leakage guards, because `assert_reportable_split` is the only thing
standing between a stray val row and a meaningless "test accuracy".
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.contract import critical_classes as config_critical
from src.inference.contract import class_names as config_class_names
from src.training.evaluation import (
    EvaluationError,
    assert_reportable_split,
    compute_metrics,
    confusion_matrix,
    evaluate_predictions,
    per_class_scores,
)

CLASSES = config_class_names()
CRITICAL = config_critical()
N = len(CLASSES)                      # 10
FLOORS = {"accuracy": 0.85, "macro_f1": 0.80, "critical_recall": 0.85}


def perfect_predictions(n_per_class: int = 10):
    y_true, y_pred = [], []
    for name in CLASSES:
        y_true.extend([name] * n_per_class)
        y_pred.extend([name] * n_per_class)
    return y_true, y_pred


# --------------------------------------------------------------------------------------
# The confusion matrix
# --------------------------------------------------------------------------------------

def test_confusion_matrix_axes_are_actual_rows_predicted_columns():
    """The classic transposition bug, pinned down by an asymmetric example."""
    labels = ["A", "B", "C"]
    matrix = confusion_matrix(["A", "A", "B"], ["B", "A", "B"], labels)

    assert matrix.shape == (3, 3)
    assert matrix[0].tolist() == [1, 1, 0]   # actual A: one B, one A
    assert matrix[1].tolist() == [0, 1, 0]   # actual B: one B
    assert matrix[2].tolist() == [0, 0, 0]   # actual C: none
    assert int(matrix.sum()) == 3, "every record must land exactly once"
    assert int(np.trace(matrix)) == 2, "two correct"


def test_confusion_matrix_agrees_with_scikit_learn():
    """Cross-check against the reference implementation, order-aligned."""
    from sklearn.metrics import confusion_matrix as sklearn_matrix

    rng = np.random.default_rng(11)
    y_true = rng.choice(CLASSES, size=400).tolist()
    y_pred = rng.choice(CLASSES, size=400).tolist()

    mine = confusion_matrix(y_true, y_pred, CLASSES)
    theirs = sklearn_matrix(y_true, y_pred, labels=CLASSES)
    assert np.array_equal(mine, theirs), (
        "the hand-written confusion matrix disagrees with scikit-learn"
    )


def test_confusion_matrix_rejects_a_label_outside_the_configured_classes():
    with pytest.raises(EvaluationError, match="not one of the 3 configured classes"):
        confusion_matrix(["A"], ["Explosion"], ["A", "B", "C"])


def test_confusion_matrix_rejects_duplicate_labels():
    with pytest.raises(EvaluationError, match="duplicates"):
        confusion_matrix(["A"], ["A"], ["A", "A"])


def test_per_class_scores_are_zero_not_perfect_when_a_class_is_never_predicted():
    """A never-predicted class must score 0, not be skipped.

    Skipping it would let a model that abandons a hard class report a high macro-F1.
    """
    labels = ["A", "B", "C"]
    matrix = np.asarray([[5, 0, 0], [0, 5, 0], [5, 0, 0]])  # C never predicted
    scores = per_class_scores(matrix)

    assert scores["recall"][2] == 0.0, "class C is never recalled"
    assert scores["precision"][2] == 0.0
    assert scores["f1"][2] == 0.0
    assert scores["support"][2] == 5


# --------------------------------------------------------------------------------------
# The core metrics
# --------------------------------------------------------------------------------------

def test_a_perfect_model_scores_one_everywhere():
    y_true, y_pred = perfect_predictions()
    result = compute_metrics(y_true, y_pred, CLASSES, CRITICAL)

    assert result.accuracy == pytest.approx(1.0)
    assert result.macro_f1 == pytest.approx(1.0)
    assert result.critical_recall == pytest.approx(1.0)
    assert result.weighted_f1 == pytest.approx(1.0)
    assert result.n_records == 100
    assert len(result.per_class) == 10
    assert sum(sum(row) for row in result.confusion_matrix) == 100


def test_accuracy_and_macro_f1_are_computed_by_hand():
    """Two labels, one right and one wrong, checked arithmetically."""
    # 2 classes for arithmetic clarity; macro over 2 classes.
    labels = ["A", "B"]
    # actual: A A B B -> predicted: A B B B
    result = compute_metrics(["A", "A", "B", "B"], ["A", "B", "B", "B"], labels)
    assert result.accuracy == pytest.approx(0.75)
    # A: precision 1.0, recall 0.5 -> f1 0.666..  B: precision 0.666.., recall 1.0 -> 0.8
    assert result.macro_f1 == pytest.approx((2 / 3 + 0.8) / 2)
    assert result.macro_precision == pytest.approx((1.0 + 2 / 3) / 2)
    assert result.macro_recall == pytest.approx((0.5 + 1.0) / 2)


def test_macro_f1_averages_over_all_configured_classes_not_just_observed_ones():
    """The honest averaging choice for a 10-class mandate.

    A model that predicts only two classes perfectly must not score 1.0: the eight it
    never predicts have recall 0 and belong in the average.
    """
    y_true = ["Machinery Fault"] * 5 + ["Glass Breaking"] * 5
    y_pred = list(y_true)   # perfect on the two classes it was given
    result = compute_metrics(y_true, y_pred, CLASSES, CRITICAL)

    assert result.accuracy == pytest.approx(1.0)
    # 2 perfect classes + 8 classes with f1 0 -> mean over 10 = 0.2
    assert result.macro_f1 == pytest.approx(0.2)
    assert result.macro_f1 < 0.80, "this must fail the macro-F1 floor"


def test_critical_recall_uses_only_the_critical_classes():
    """Critical recall must not be diluted by Background Noise."""

    def outcome(actual: str) -> str:
        # Every non-critical class is perfect; every critical class is missed.
        return "Background Noise" if actual in CRITICAL else actual

    y_true = [c for c in CLASSES for _ in range(4)]
    y_pred = [outcome(a) for a in y_true]
    result = compute_metrics(y_true, y_pred, CLASSES, CRITICAL)

    assert result.critical_recall == pytest.approx(0.0)
    assert set(result.critical_recall_by_class) == set(CRITICAL)
    assert len(result.critical_recall_by_class) == len(CRITICAL)
    # Overall accuracy is poor but not zero — the point is the two numbers differ.
    assert result.accuracy > 0.0


def test_critical_recall_is_the_mean_over_critical_classes():
    y_true, y_pred = [], []
    for name in CLASSES:
        if name in CRITICAL:
            # 4 records: 3 correct, 1 missed
            y_true.extend([name] * 4)
            y_pred.extend([name] * 3 + ["Background Noise"])
        else:
            y_true.extend([name] * 4)
            y_pred.extend([name] * 4)
    result = compute_metrics(y_true, y_pred, CLASSES, CRITICAL)
    assert result.critical_recall == pytest.approx(0.75)
    for value in result.critical_recall_by_class.values():
        assert value == pytest.approx(0.75)


def test_top2_accuracy_is_computed_when_a_ranking_is_supplied():
    y_true = ["Gunshot", "Gunshot", "Gunshot", "Gunshot"]
    y_pred = ["Gunshot", "Glass Breaking", "Vehicle Horn", "Gunshot"]
    top2 = [
        ["Gunshot", "Glass Breaking"],   # hit at rank 1
        ["Glass Breaking", "Gunshot"],   # hit at rank 2
        ["Vehicle Horn", "Animal Sound"],  # miss
        ["Gunshot", "Panic Scream"],     # hit at rank 1
    ]
    result = compute_metrics(y_true, y_pred, CLASSES, CRITICAL, top2=top2)
    assert result.top2_accuracy == pytest.approx(0.75)
    # Top-2 can only ever be at least as good as top-1.
    assert result.top2_accuracy >= result.accuracy


def test_top2_accuracy_is_none_when_no_ranking_is_given():
    y_true, y_pred = perfect_predictions(n_per_class=2)
    assert compute_metrics(y_true, y_pred, CLASSES, CRITICAL).top2_accuracy is None


def test_top2_must_align_with_the_labels():
    with pytest.raises(EvaluationError, match="top2 must align"):
        compute_metrics(
            ["Gunshot", "Gunshot"], ["Gunshot", "Gunshot"], CLASSES, CRITICAL,
            top2=[["Gunshot"]],
        )


def test_mismatched_lengths_are_refused():
    with pytest.raises(EvaluationError, match="actual labels but"):
        compute_metrics(["A", "B"], ["A"], ["A", "B"])


def test_empty_input_is_refused():
    with pytest.raises(EvaluationError, match="no predictions"):
        compute_metrics([], [], CLASSES, CRITICAL)


# --------------------------------------------------------------------------------------
# Severe errors — ranked by consequence
# --------------------------------------------------------------------------------------

def test_severe_errors_separate_silent_misses_from_misattributed_alerts():
    """A gunshot called 'Vehicle Horn' alerts someone. Called 'Background Noise', nobody
    is told. Both are wrong; only one is dangerous."""
    # "Aggression" is critical, "Background Noise" and "Vehicle Horn" are not — so the
    # first is a wrong alert, the other two are wrong silences.
    y_true = ["Gunshot", "Gunshot", "Panic Scream"]
    y_pred = ["Aggression", "Background Noise", "Vehicle Horn"]
    result = compute_metrics(
        y_true, y_pred, CLASSES, CRITICAL, audio_ids=["a", "b", "c"]
    )

    assert len(result.severe_errors) == 3
    by_id = {e["audio_id"]: e for e in result.severe_errors}
    assert by_id["a"]["silent_miss"] is False
    assert by_id["a"]["misattributed_alert"] is True
    assert by_id["b"]["silent_miss"] is True
    assert by_id["b"]["misattributed_alert"] is False

    # Sorted worst-first: the silent miss leads regardless of insertion order.
    assert result.severe_errors[0]["silent_miss"] is True


def test_a_correct_critical_prediction_is_not_a_severe_error():
    y_true = ["Gunshot", "Vehicle Horn"]
    y_pred = ["Gunshot", "Vehicle Horn"]
    result = compute_metrics(y_true, y_pred, CLASSES, CRITICAL)
    assert result.severe_errors == []


def test_severe_errors_is_empty_when_no_critical_classes_are_configured():
    result = compute_metrics(["A"], ["B"], ["A", "B"], [])
    assert result.severe_errors == []
    assert result.critical_recall == 0.0


# --------------------------------------------------------------------------------------
# The floors
# --------------------------------------------------------------------------------------

def test_a_strong_model_meets_the_floors():
    y_true, y_pred = perfect_predictions()
    result = compute_metrics(y_true, y_pred, CLASSES, CRITICAL)
    ok, failures = result.meets_floors(FLOORS)
    assert ok is True
    assert failures == []


def test_a_model_below_the_accuracy_floor_fails_with_the_numbers_stated():
    """A floor check that cannot fail is decoration."""
    y_true, y_pred = [], []
    for i, name in enumerate(CLASSES):
        y_true.extend([name] * 10)
        # 6 right, 4 wrong per class, misattributed to a rotating OTHER class so no
        # class accidentally scores its own mistakes -> accuracy exactly 0.6.
        y_pred.extend([name] * 6 + [CLASSES[(i + 1) % len(CLASSES)]] * 4)
    result = compute_metrics(y_true, y_pred, CLASSES, CRITICAL)

    assert result.accuracy == pytest.approx(0.6)
    ok, failures = result.meets_floors(FLOORS)
    assert ok is False
    joined = " ".join(failures)
    assert "accuracy" in joined and "0.85" in joined
    assert "0.600" in joined, "the failure must state the measured value"


def test_a_model_with_high_accuracy_but_poor_critical_recall_still_fails():
    """This is the case the floor exists for: a model that looks fine and is not."""
    y_true, y_pred = [], []
    for name in CLASSES:
        if name in CRITICAL:
            y_true.extend([name] * 10)
            y_pred.extend([name] * 6 + ["Background Noise"] * 4)   # recall 0.6
        else:
            y_true.extend([name] * 2)
            y_pred.extend([name] * 2)
    result = compute_metrics(y_true, y_pred, CLASSES, CRITICAL)

    assert result.critical_recall == pytest.approx(0.6)
    ok, failures = result.meets_floors(FLOORS)
    assert ok is False
    assert any("critical_recall" in f for f in failures)


def test_a_probe_can_never_be_certified_against_the_floors():
    """Robustness numbers are worth having and are not test accuracy."""
    y_true, y_pred = perfect_predictions()
    result = compute_metrics(
        y_true, y_pred, CLASSES, CRITICAL,
        probe=True, probe_label="noise robustness @ 5 dB SNR",
    )
    ok, failures = result.meets_floors(FLOORS)
    assert ok is False
    assert any("probe" in f for f in failures)


def test_a_missing_floor_is_skipped_not_treated_as_zero():
    y_true, y_pred = perfect_predictions()
    result = compute_metrics(y_true, y_pred, CLASSES, CRITICAL)
    ok, failures = result.meets_floors({"accuracy": 0.85})   # no macro_f1, no critical
    assert ok is True
    assert failures == []


# --------------------------------------------------------------------------------------
# The leakage guards
# --------------------------------------------------------------------------------------

def _records(split: str, n: int = 3, status: str = "original"):
    return [
        {"audio_id": f"r{i}", "dataset_split": split, "original_or_augmented": status,
         "class_label": CLASSES[0], "filename": f"{i}.wav"}
        for i in range(n)
    ]


def test_the_guard_accepts_a_clean_test_split():
    assert assert_reportable_split(_records("test"), "test") == 3


def test_the_guard_refuses_a_mixed_split():
    mixed = _records("test", 2) + _records("train", 1)
    with pytest.raises(EvaluationError, match="not in the test split"):
        assert_reportable_split(mixed, "test")


def test_the_guard_refuses_an_augmented_record():
    """The rule I will halt other work over: augmented audio is not unseen data."""
    with pytest.raises(EvaluationError, match="augmented"):
        assert_reportable_split(
            _records("test", 2, status="augmented"), "test"
        )


def test_the_guard_allows_augmented_records_only_when_explicitly_permitted():
    n = assert_reportable_split(
        _records("test", 2, status="augmented"), "test", allow_augmented=True
    )
    assert n == 2


def test_the_guard_refuses_an_empty_split():
    with pytest.raises(EvaluationError, match="no records"):
        assert_reportable_split([], "test")


def test_the_guard_refuses_an_empty_split_value():
    with pytest.raises(EvaluationError, match="<empty>"):
        assert_reportable_split(_records(""), "test")


def test_the_guard_rejects_a_split_name_that_is_not_one_of_the_three():
    with pytest.raises(EvaluationError, match="unknown split"):
        assert_reportable_split(_records("test"), "testing")


def test_the_guard_message_names_the_evaluated_model():
    with pytest.raises(EvaluationError, match="evaluating svm-rbf"):
        assert_reportable_split(
            _records("val"), "test", context="evaluating svm-rbf"
        )


# --------------------------------------------------------------------------------------
# Driving a model through the harness
# --------------------------------------------------------------------------------------

def test_evaluate_predictions_accepts_a_bare_class_name():
    records = _records("test", 4)
    result, rows = evaluate_predictions(
        records, lambda r: CLASSES[0], CLASSES, CRITICAL, model_name="stub"
    )
    assert result.accuracy == pytest.approx(1.0)
    assert len(rows) == 4
    assert all(row["correct"] for row in rows)


def test_evaluate_predictions_accepts_a_prediction_result_shape():
    records = _records("test", 3)

    def predict(record):
        return {
            "predicted_class": CLASSES[0],
            "confidences": {name: (0.9 if name == CLASSES[0] else 0.1 / 9) for name in CLASSES},
        }

    result, rows = evaluate_predictions(records, predict, CLASSES, CRITICAL)
    assert result.accuracy == pytest.approx(1.0)
    assert rows[0]["confidence"] == pytest.approx(0.9)
    assert rows[0]["confidences"][CLASSES[0]] == pytest.approx(0.9)
    # Top-2 comes from the confidences, so a ranking is available without extra plumbing.
    assert result.top2_accuracy == pytest.approx(1.0)


def test_evaluate_predictions_refuses_a_split_that_is_not_frozen():
    records = _records("train", 3)
    with pytest.raises(EvaluationError, match="not in the test split"):
        evaluate_predictions(records, lambda r: CLASSES[0], CLASSES, CRITICAL)


def test_evaluate_predictions_refuses_a_prediction_outside_the_class_list():
    records = _records("test", 2)
    with pytest.raises(EvaluationError, match="not a configured class"):
        evaluate_predictions(records, lambda r: "Explosion", CLASSES, CRITICAL)


def test_evaluate_predictions_refuses_a_record_with_no_label():
    records = _records("test", 2)
    records[0]["class_label"] = ""
    with pytest.raises(EvaluationError, match="has no 'class_label'"):
        evaluate_predictions(records, lambda r: CLASSES[0], CLASSES, CRITICAL)


def test_the_split_can_be_bypassed_only_deliberately():
    """Training code needs to score the train split; it must say so explicitly."""
    records = _records("train", 3)
    result, _ = evaluate_predictions(
        records, lambda r: CLASSES[0], CLASSES, CRITICAL,
        split="train", require_frozen_split=False,
    )
    assert result.split == "train"
    assert result.accuracy == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Serialisation — the report reads this
# --------------------------------------------------------------------------------------

def test_to_dict_round_trips_through_json():
    import json

    y_true, y_pred = perfect_predictions()
    payload = compute_metrics(y_true, y_pred, CLASSES, CRITICAL).to_dict()
    restored = json.loads(json.dumps(payload))

    assert restored["accuracy"] == pytest.approx(1.0)
    assert len(restored["confusion_matrix"]) == 10
    assert len(restored["confusion_matrix"][0]) == 10
    assert restored["per_class"][0]["class_name"] in CLASSES
    assert restored["class_names"] == CLASSES
    assert "probe" in restored


def test_worst_classes_names_the_classes_with_the_lowest_f1():
    y_true, y_pred = [], []
    for i, name in enumerate(CLASSES):
        y_true.extend([name] * 10)
        # Every class is perfect except the last, which is never recognised at all.
        correct = 10 if i != len(CLASSES) - 1 else 0
        y_pred.extend([name] * correct + ["Machinery Fault"] * (10 - correct))
    result = compute_metrics(y_true, y_pred, CLASSES, CRITICAL)
    worst = result.worst_classes(3)
    assert worst[0]["class_name"] == CLASSES[-1]
    assert worst[0]["f1"] == pytest.approx(0.0)


def test_config_snapshot_is_carried_into_the_result():
    """Every metric must be traceable to the thresholds that produced it."""
    snapshot = {"min_confidence": 0.6, "segment_duration_sec": 3.0}
    result = compute_metrics(
        ["Gunshot"], ["Gunshot"], CLASSES, CRITICAL, config_snapshot=snapshot
    )
    assert result.config_snapshot == snapshot
    assert result.to_dict()["config_snapshot"] == snapshot
