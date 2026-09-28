"""Consistency statuses (SRS Step 11, Step 14, FR xxiii-xxx, lii).

Thresholds are passed in explicitly, to check that the result follows the config.
"""

from __future__ import annotations

import pytest

from src.inference.consistency import (
    ACCEPTABLE_MATCH,
    CONSISTENCY_STATUSES,
    MODEL_DISAGREEMENT,
    STRONG_MATCH,
    UNCERTAIN_RESULT,
    WEAK_MATCH,
    classify_consistency,
    requires_manual_review,
)
from src.inference.contract import PredictionResult, load_class_config, load_thresholds

CLASSES = [c["name"] for c in load_class_config()["classes"]]
CONFIG = load_thresholds()
# The logic tests pin their own values, so re-calibrating config/thresholds.json cannot
# change what they check; test_real_config_* tests cover the live file.
THRESHOLDS = {
    **CONFIG,
    "confidence": dict(CONFIG["confidence"], min_confidence=0.6, low_confidence_band=0.75,
                       top_two_margin_min=0.15, overlap_secondary_confidence=0.25),
    "consistency": dict(CONFIG["consistency"], strong_match_max_diff=0.05,
                        acceptable_match_max_diff=0.15, weak_match_max_diff=0.3,
                        confident_agreement_min=0.8),
}


def make_result(model: str, predicted: str, confidence: float, runner_up: str | None = None,
                runner_up_confidence: float | None = None) -> PredictionResult:
    """PredictionResult with a full 10-class distribution that sums to 1."""
    rest = [c for c in CLASSES if c not in (predicted, runner_up)]
    if runner_up is None:
        share = (1.0 - confidence) / len(rest)
        confidences = {c: share for c in rest}
    else:
        share = (1.0 - confidence - float(runner_up_confidence)) / len(rest)
        confidences = {c: share for c in rest}
        confidences[runner_up] = float(runner_up_confidence)
    confidences[predicted] = confidence
    return PredictionResult(
        model_name=model, model_version="test", predicted_class=predicted,
        confidence=confidence, confidences=confidences,
    )


# Every status can be reached

def test_strong_match():
    """Same class, small confidence gap."""
    py = make_result("python", "Gunshot", 0.92)
    gtm = make_result("gtm", "Gunshot", 0.90)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.consistency_status == STRONG_MATCH
    assert result.classes_agree is True
    assert result.confidence_difference == pytest.approx(0.02, abs=1e-9)


def test_acceptable_match():
    py = make_result("python", "Glass Breaking", 0.90)
    gtm = make_result("gtm", "Glass Breaking", 0.80)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.consistency_status == ACCEPTABLE_MATCH
    assert result.confidence_difference == pytest.approx(0.10, abs=1e-9)


def test_weak_match():
    """Same class, wide gap, and GTM below the "both confident" level."""
    py = make_result("python", "Vehicle Horn", 0.95)
    gtm = make_result("gtm", "Vehicle Horn", 0.72)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.consistency_status == WEAK_MATCH
    assert result.confidence_difference == pytest.approx(0.23, abs=1e-9)


def test_weak_match_when_gtm_agrees_below_the_floor():
    """Python is sure and GTM picks the same class without reaching the floor."""
    py = make_result("python", "Gunshot", 0.93)
    gtm = make_result("gtm", "Gunshot", 0.41)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.consistency_status == WEAK_MATCH
    assert "GTM" in result.reason


def test_confident_agreement_is_acceptable_despite_a_wide_gap():
    """SRS Step 11: the models scale scores differently, so both confident is enough."""
    py = make_result("python", "Glass Breaking", 0.99)
    gtm = make_result("gtm", "Glass Breaking", 0.81)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.confidence_difference == pytest.approx(0.18, abs=1e-9)
    assert result.consistency_status == ACCEPTABLE_MATCH
    assert "both are confident" in result.reason
    needed, reason = requires_manual_review(result, THRESHOLDS)
    assert needed is False and reason == ""


def test_without_confident_agreement_min_the_gap_decides():
    """Older configs without the key keep grading by the difference alone."""
    legacy = {
        "confidence": THRESHOLDS["confidence"],
        "consistency": {k: v for k, v in THRESHOLDS["consistency"].items()
                        if k != "confident_agreement_min"},
    }
    py = make_result("python", "Glass Breaking", 0.99)
    gtm = make_result("gtm", "Glass Breaking", 0.81)
    assert classify_consistency(py, gtm, legacy).consistency_status == WEAK_MATCH


def test_model_disagreement():
    """Different classes means disagreement, whatever the confidences."""
    py = make_result("python", "Gunshot", 0.95)
    gtm = make_result("gtm", "Glass Breaking", 0.94)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.consistency_status == MODEL_DISAGREEMENT
    assert result.classes_agree is False
    assert "Gunshot" in result.reason and "Glass Breaking" in result.reason


def test_uncertain_result_both_models_unsure():
    py = make_result("python", "Animal Sound", 0.30)
    gtm = make_result("gtm", "Animal Sound", 0.28)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.consistency_status == UNCERTAIN_RESULT


def test_uncertain_result_when_the_python_result_is_unsure():
    """Agreement doesn't make the reported (Python) result certain."""
    py = make_result("python", "Alarm or Siren", 0.20)
    gtm = make_result("gtm", "Alarm or Siren", 0.91)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.consistency_status == UNCERTAIN_RESULT
    assert "Python" in result.reason


def test_disagreement_is_named_even_when_one_model_is_unsure():
    py = make_result("python", "Gunshot", 0.90)
    gtm = make_result("gtm", "Glass Breaking", 0.35)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.consistency_status == MODEL_DISAGREEMENT
    assert "below" in result.reason


def test_all_five_statuses_are_producible():
    """All five statuses can actually occur."""
    produced = {
        classify_consistency(*pair, THRESHOLDS).consistency_status
        for pair in [
            (make_result("p", "Gunshot", 0.92), make_result("g", "Gunshot", 0.90)),
            (make_result("p", "Gunshot", 0.90), make_result("g", "Gunshot", 0.80)),
            (make_result("p", "Gunshot", 0.95), make_result("g", "Gunshot", 0.72)),
            (make_result("p", "Gunshot", 0.95), make_result("g", "Glass Breaking", 0.94)),
            (make_result("p", "Gunshot", 0.10), make_result("g", "Gunshot", 0.12)),
        ]
    }
    assert produced == set(CONSISTENCY_STATUSES)


# Boundaries

@pytest.mark.parametrize("strong_max,acceptable_max,weak_max,difference,expected", [
    # exact binary fractions, so float rounding doesn't matter
    (0.25, 0.50, 0.75, 0.125, STRONG_MATCH),      # inside strong
    (0.25, 0.50, 0.75, 0.250, STRONG_MATCH),      # on the strong bound (inclusive)
    (0.25, 0.50, 0.75, 0.500, ACCEPTABLE_MATCH),  # past strong, on the acceptable bound
    (0.25, 0.50, 0.75, 0.750, WEAK_MATCH),        # on the weak bound
    (0.25, 0.50, 0.75, 0.875, WEAK_MATCH),        # past the weak bound: still weak
    (0.50, 0.75, 0.875, 0.250, STRONG_MATCH),     # same difference, looser config
    (0.125, 0.25, 0.50, 0.250, ACCEPTABLE_MATCH), # same difference, stricter config
])
def test_match_grading_boundaries(strong_max, acceptable_max, weak_max, difference, expected):
    """Thresholds are inclusive upper bounds."""
    thresholds = {
        # min_confidence is set low so only the grading thresholds matter here.
        "confidence": dict(THRESHOLDS["confidence"], min_confidence=0.0),
        "consistency": {
            "strong_match_max_diff": strong_max,
            "acceptable_match_max_diff": acceptable_max,
            "weak_match_max_diff": weak_max,
        },
    }
    py = make_result("python", "Gunshot", 0.875)
    gtm = make_result("gtm", "Gunshot", 0.875 - difference)
    result = classify_consistency(py, gtm, thresholds)
    assert result.consistency_status == expected
    assert result.confidence_difference == pytest.approx(difference, abs=1e-12)


def test_real_config_thresholds_are_honoured():
    """The live config values are used."""
    cons = CONFIG["consistency"]
    floor = CONFIG["confidence"]["min_confidence"]
    pairs = [
        (0.95 - cons["strong_match_max_diff"] / 2, STRONG_MATCH),
        (0.95 - (cons["strong_match_max_diff"] + cons["acceptable_match_max_diff"]) / 2,
         ACCEPTABLE_MATCH),
        (floor - 0.01, WEAK_MATCH),
    ]
    for gtm_confidence, expected in pairs:
        py = make_result("python", "Gunshot", 0.95)
        gtm = make_result("gtm", "Gunshot", gtm_confidence)
        assert classify_consistency(py, gtm, CONFIG).consistency_status == expected


def test_confident_agreement_never_counts_a_model_below_the_floor():
    """Raising only the floor (a typical surprise change) keeps "both confident" above it."""
    raised = {
        "confidence": dict(THRESHOLDS["confidence"], min_confidence=0.9),
        "consistency": dict(THRESHOLDS["consistency"], confident_agreement_min=0.5),
    }
    py = make_result("python", "Gunshot", 0.95)
    gtm = make_result("gtm", "Gunshot", 0.85)
    result = classify_consistency(py, gtm, raised)
    assert result.threshold_snapshot["confident_agreement_min"] == 0.9
    assert result.consistency_status == WEAK_MATCH      # GTM is below the raised floor


def test_min_confidence_boundary_is_inclusive():
    """A confidence exactly at the minimum is accepted."""
    threshold = THRESHOLDS["confidence"]["min_confidence"]
    py = make_result("python", "Gunshot", threshold)
    gtm = make_result("gtm", "Gunshot", threshold)
    assert classify_consistency(py, gtm, THRESHOLDS).consistency_status != UNCERTAIN_RESULT


def test_just_below_min_confidence_is_uncertain():
    threshold = THRESHOLDS["confidence"]["min_confidence"]
    py = make_result("python", "Gunshot", threshold - 0.001)
    gtm = make_result("gtm", "Gunshot", threshold - 0.002)
    assert classify_consistency(py, gtm, THRESHOLDS).consistency_status == UNCERTAIN_RESULT


# Config changes

def test_taxonomy_responds_to_changed_thresholds():
    """Changing the thresholds changes the result."""
    py = make_result("python", "Gunshot", 0.95)
    gtm = make_result("gtm", "Gunshot", 0.80)   # difference 0.15 -> Acceptable by default

    assert classify_consistency(py, gtm, THRESHOLDS).consistency_status == ACCEPTABLE_MATCH

    stricter = {
        "confidence": dict(THRESHOLDS["confidence"]),
        "consistency": dict(THRESHOLDS["consistency"], acceptable_match_max_diff=0.05,
                            confident_agreement_min=0.9),
    }
    assert classify_consistency(py, gtm, stricter).consistency_status == WEAK_MATCH


def test_raising_min_confidence_turns_agreement_into_uncertain():
    py = make_result("python", "Gunshot", 0.70)
    gtm = make_result("gtm", "Gunshot", 0.70)
    assert classify_consistency(py, gtm, THRESHOLDS).consistency_status != UNCERTAIN_RESULT

    strict = {
        "confidence": dict(THRESHOLDS["confidence"], min_confidence=0.85),
        "consistency": dict(THRESHOLDS["consistency"]),
    }
    assert classify_consistency(py, gtm, strict).consistency_status == UNCERTAIN_RESULT


def test_threshold_snapshot_is_recorded_with_every_result():
    """Each result records the thresholds used."""
    result = classify_consistency(
        make_result("python", "Gunshot", 0.9), make_result("gtm", "Gunshot", 0.9), THRESHOLDS
    )
    snapshot = result.threshold_snapshot
    assert set(snapshot) == {
        "min_confidence", "strong_match_max_diff", "acceptable_match_max_diff", "weak_match_max_diff",
        "confident_agreement_min",
    }
    assert all(isinstance(v, float) for v in snapshot.values())


# Confidence difference

def test_confidence_difference_is_absolute():
    """|python - gtm| == |gtm - python|"""
    a = make_result("python", "Gunshot", 0.95)
    b = make_result("gtm", "Gunshot", 0.70)
    forward = classify_consistency(a, b, THRESHOLDS)
    backward = classify_consistency(b, a, THRESHOLDS)
    assert forward.confidence_difference == pytest.approx(0.25, abs=1e-9)
    assert forward.confidence_difference == pytest.approx(backward.confidence_difference, abs=1e-9)
    assert forward.consistency_status == backward.consistency_status


def test_top_two_margin_reported_for_both_models():
    py = make_result("python", "Gunshot", 0.70, runner_up="Glass Breaking", runner_up_confidence=0.20)
    gtm = make_result("gtm", "Gunshot", 0.68)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.top_two_margin_python == pytest.approx(0.50, abs=1e-6)
    assert 0.0 <= result.top_two_margin_gtm <= 1.0


def test_overlap_detected_when_runner_up_is_strong():
    """A strong second class is flagged as an overlapping sound."""
    py = make_result("python", "Gunshot", 0.62, runner_up="Panic Scream", runner_up_confidence=0.30)
    gtm = make_result("gtm", "Gunshot", 0.60)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.overlapping is True
    assert result.secondary_detection == "Panic Scream"
    assert "overlapping" in result.reason.lower()


def test_no_overlap_flagged_when_models_disagree():
    """No overlap flag when the models disagree."""
    py = make_result("python", "Gunshot", 0.62, runner_up="Panic Scream", runner_up_confidence=0.30)
    gtm = make_result("gtm", "Glass Breaking", 0.61)
    assert classify_consistency(py, gtm, THRESHOLDS).overlapping is False


# Manual-review routing

def test_disagreement_goes_to_review():
    result = classify_consistency(
        make_result("python", "Gunshot", 0.95), make_result("gtm", "Vehicle Horn", 0.93), THRESHOLDS
    )
    needed, reason = requires_manual_review(result, THRESHOLDS)
    assert needed is True
    assert MODEL_DISAGREEMENT in reason


def test_uncertain_goes_to_review():
    result = classify_consistency(
        make_result("python", "Gunshot", 0.30), make_result("gtm", "Gunshot", 0.31), THRESHOLDS
    )
    needed, _ = requires_manual_review(result, THRESHOLDS)
    assert needed is True


def test_confident_strong_match_does_not_go_to_review():
    """A confident strong match stays out of the review queue."""
    result = classify_consistency(
        make_result("python", "Gunshot", 0.95), make_result("gtm", "Gunshot", 0.94), THRESHOLDS
    )
    needed, reason = requires_manual_review(result, THRESHOLDS)
    assert needed is False
    assert reason == ""


def test_weak_match_goes_to_review():
    result = classify_consistency(
        make_result("python", "Gunshot", 0.95), make_result("gtm", "Gunshot", 0.70), THRESHOLDS
    )
    assert result.consistency_status == WEAK_MATCH
    needed, reason = requires_manual_review(result, THRESHOLDS)
    assert needed is True
    assert "Weak model agreement" in reason


def test_review_reason_names_every_trigger():
    """The review reason lists every trigger."""
    py = make_result("python", "Gunshot", 0.62, runner_up="Panic Scream", runner_up_confidence=0.30)
    gtm = make_result("gtm", "Gunshot", 0.90)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.consistency_status == WEAK_MATCH

    needed, reason = requires_manual_review(result, THRESHOLDS)
    assert needed is True
    assert "Weak model agreement" in reason
    assert "Low Python confidence" in reason
    assert "overlapping" in reason.lower()


def test_disagreement_and_low_confidence_both_named():
    py = make_result("python", "Gunshot", 0.65)
    gtm = make_result("gtm", "Glass Breaking", 0.72)
    result = classify_consistency(py, gtm, THRESHOLDS)
    needed, reason = requires_manual_review(result, THRESHOLDS)
    assert needed is True
    assert "Model Disagreement" in reason
    assert "Low Python confidence" in reason


def test_strong_runner_up_implies_low_confidence_and_routes_to_review():
    """A runner-up at 0.25 caps the top class at 0.75, so overlaps go to review."""
    py = make_result("python", "Gunshot", 0.75, runner_up="Panic Scream", runner_up_confidence=0.25)
    gtm = make_result("gtm", "Gunshot", 0.85)
    result = classify_consistency(py, gtm, THRESHOLDS)

    assert result.overlapping is True
    assert result.secondary_detection == "Panic Scream"
    assert result.consistency_status == ACCEPTABLE_MATCH   # same class, difference 0.10

    needed, reason = requires_manual_review(result, THRESHOLDS)
    assert needed is True
    assert "overlapping" in reason.lower()


def test_a_weak_runner_up_is_not_an_overlap():
    """A weak runner-up is not an overlap."""
    py = make_result("python", "Gunshot", 0.90, runner_up="Panic Scream", runner_up_confidence=0.02)
    gtm = make_result("gtm", "Gunshot", 0.89)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.overlapping is False
    assert result.secondary_detection is None
    needed, reason = requires_manual_review(result, THRESHOLDS)
    assert needed is False
    assert reason == ""


def test_comparison_serialises_for_the_api():
    """to_dict has the fields the UI and reports use."""
    result = classify_consistency(
        make_result("python", "Gunshot", 0.9), make_result("gtm", "Gunshot", 0.88), THRESHOLDS
    )
    payload = result.to_dict()
    assert set(payload) >= {
        "python", "gtm", "classes_agree", "confidence_difference",
        "consistency_status", "reason", "threshold_snapshot",
    }
    assert len(payload["python"]["top3"]) == 3
    assert len(payload["gtm"]["top3"]) == 3
    for entry in payload["python"]["top3"] + payload["gtm"]["top3"]:
        assert set(entry) == {"class", "confidence"}
