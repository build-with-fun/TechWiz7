"""Comparison report format and integrity (SRS Deliverable 3, Step 11, Step 14)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.inference.contract import PredictionResult
from src.inference.consistency import (
    MODEL_DISAGREEMENT,
    STRONG_MATCH,
    classify_consistency,
)
from src.training.comparison_report import (
    CSV_COLUMNS,
    ComparisonReport,
    ComparisonRow,
)


# fixtures


def make_result(predicted: str, conf: float, classes: tuple[str, ...]) -> PredictionResult:
    """PredictionResult with confidences for every class."""
    confidences = {c: 0.0 for c in classes}
    confidences[predicted] = conf
    # spread the rest so it sums to 1
    rest = 1.0 - conf
    others = [c for c in classes if c != predicted]
    if others:
        share = rest / len(others)
        for c in others:
            confidences[c] = round(share, 6)
        # fix rounding on the predicted class
        confidences[predicted] = round(1.0 - sum(v for k, v in confidences.items() if k != predicted), 6)
    return PredictionResult(
        model_name="test",
        model_version="1.0",
        predicted_class=predicted,
        confidence=conf,
        confidences=confidences,
    )


CLASSES = (
    "Aggression", "Alarm or Siren", "Animal Sound", "Background Noise",
    "Glass Breaking", "Gunshot", "Machinery Fault", "Panic Scream",
    "Person Asking for Help", "Vehicle Horn",
)

THRESHOLDS = {
    "confidence": {"min_confidence": 0.60, "low_confidence_band": 0.75,
                   "top_two_margin_min": 0.15, "overlap_secondary_confidence": 0.25},
    "consistency": {"strong_match_max_diff": 0.05, "acceptable_match_max_diff": 0.15,
                    "weak_match_max_diff": 0.30},
}


def make_row(audio_id: str, true_class: str, py: str, py_conf: float,
             gtm: str, gtm_conf: float) -> ComparisonRow:
    py_res = make_result(py, py_conf, CLASSES)
    gtm_res = make_result(gtm, gtm_conf, CLASSES)
    comp = classify_consistency(py_res, gtm_res, THRESHOLDS)
    return ComparisonRow(audio_id=audio_id, true_class=true_class, comparison=comp)


def make_report(n_per_class: int = 10) -> ComparisonReport:
    rows = []
    for cls in CLASSES:
        for i in range(n_per_class):
            # both models agree, high confidence -> Strong Match
            rows.append(make_row(f"SS-{cls[:3].upper()}-{i:04d}", cls, cls, 0.95, cls, 0.93))
    return ComparisonReport(
        rows=rows, split_name="test", python_model_name="py-artifact",
        gtm_model_name="gtm-artifact", threshold_snapshot=THRESHOLDS,
        classes=CLASSES,
    )


# integrity


def test_gtm_result_is_built_independently_of_python():
    """The TM result is a separate object from the Python one."""
    py = make_result("Gunshot", 0.9, CLASSES)
    gtm = make_result("Gunshot", 0.9, CLASSES)
    assert py is not gtm
    comp = classify_consistency(py, gtm, THRESHOLDS)
    # each confidence comes from its own model
    assert comp.python_confidence == 0.9
    assert comp.gtm_confidence == 0.9


def test_no_hardcoded_predictions_in_rows():
    """Row classes come from the prediction objects."""
    r = make_row("SS-GUN-0001", "Gunshot", "Gunshot", 0.9, "Gunshot", 0.9)
    assert r.comparison.python_class == "Gunshot"
    assert r.comparison.gtm_class == "Gunshot"
    assert r.comparison.python_confidence == 0.9
    assert r.comparison.gtm_confidence == 0.9


def test_confidences_are_never_nan_or_infinite():
    """No NaN or infinite confidences."""
    r = make_row("SS-GUN-0001", "Gunshot", "Gunshot", 0.9, "Gunshot", 0.9)
    import math
    for v in (r.comparison.python_confidence, r.comparison.gtm_confidence,
              r.comparison.confidence_difference):
        assert math.isfinite(v)


# format


def test_csv_columns_are_stable_and_complete():
    """CSV columns are fixed."""
    assert CSV_COLUMNS == (
        "audio_id", "true_class", "python_predicted_class", "python_confidence",
        "gtm_predicted_class", "gtm_confidence", "classes_agree",
        "confidence_difference", "consistency_status", "python_correct",
        "gtm_correct", "reason",
    )


def test_row_csv_has_every_column():
    r = make_row("SS-GUN-0001", "Gunshot", "Gunshot", 0.9, "Gunshot", 0.88)
    row = r.to_csv_row()
    assert set(row.keys()) == set(CSV_COLUMNS)


def test_report_meets_floor_at_10_per_class():
    """At least 100 clips and 10 per class."""
    rep = make_report(n_per_class=10)
    assert rep.n_clips == 100
    ok, problems = rep.meets_report_floor()
    assert ok, problems
    assert problems == []


def test_report_fails_floor_when_a_class_is_missing():
    rep = make_report(n_per_class=10)
    # drop every Gunshot row
    rep.rows = [r for r in rep.rows if r.true_class != "Gunshot"]
    ok, problems = rep.meets_report_floor()
    assert not ok
    assert any("Gunshot" in p for p in problems)


def test_report_fails_floor_below_100_clips():
    rep = make_report(n_per_class=5)
    ok, problems = rep.meets_report_floor()
    assert not ok
    assert any("50 recordings" in p for p in problems)


def test_agreement_and_accuracy():
    """Each model is scored against the true label."""
    rep = ComparisonReport(classes=CLASSES)
    # agree and both right
    rep.rows.append(make_row("a", "Gunshot", "Gunshot", 0.9, "Gunshot", 0.9))
    # disagree, python right
    rep.rows.append(make_row("b", "Gunshot", "Gunshot", 0.9, "Siren", 0.7))
    assert rep.agreement_rate() == 0.5
    assert rep.accuracy("python") == 1.0
    assert rep.accuracy("gtm") == 0.5


def test_accuracy_rejects_unknown_model_name():
    rep = ComparisonReport(classes=CLASSES)
    with pytest.raises(ValueError):
        rep.accuracy("unknown-model")


# round trip


def test_write_and_read_back(tmp_path: Path):
    """CSV and JSON read back the same."""
    rep = make_report(n_per_class=10)
    out_csv = tmp_path / "comparison.csv"
    out_json = tmp_path / "comparison_summary.json"
    rep.write(out_csv, out_json)

    with open(out_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == CSV_COLUMNS
        records = list(reader)
    assert len(records) == 100
    assert records[0]["consistency_status"] == STRONG_MATCH
    assert records[0]["python_correct"] == "true"
    assert records[0]["gtm_correct"] == "true"

    summary = json.loads(out_json.read_text())
    assert summary["n_clips"] == 100
    assert summary["meets_report_floor"] is True
    assert summary["consistency_breakdown"][STRONG_MATCH] == 100
    assert summary["per_class_counts"]["Gunshot"] == 10


def test_summary_records_provenance(tmp_path: Path):
    """The summary names the split and the models."""
    rep = make_report(n_per_class=10)
    out_csv = tmp_path / "c.csv"
    out_json = tmp_path / "c.json"
    rep.write(out_csv, out_json)
    summary = json.loads(out_json.read_text())
    assert summary["split"] == "test"
    assert summary["python_model"] == "py-artifact"
    assert summary["gtm_model"] == "gtm-artifact"
    assert summary["threshold_snapshot"] is not None


def test_consistency_status_is_from_the_srs_taxonomy():
    """Only the five known statuses appear."""
    rep = make_report(n_per_class=1)
    statuses = {r.comparison.consistency_status for r in rep.rows}
    # any of the five
    assert statuses.issubset({
        "Strong Match", "Acceptable Match", "Weak Match",
        "Model Disagreement", "Uncertain Result",
    })


def test_disagreement_row_is_labelled():
    """Disagreement rows are labelled "Model Disagreement"."""
    r = make_row("SS-GUN-0002", "Gunshot", "Gunshot", 0.9, "Glass Breaking", 0.8)
    row = r.to_csv_row()
    assert row["classes_agree"] == "false"
    assert row["consistency_status"] == MODEL_DISAGREEMENT
    assert row["python_correct"] == "true"
    assert row["gtm_correct"] == "false"
