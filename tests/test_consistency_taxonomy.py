"""
The SRS consistency taxonomy must be exactly right — it is what the whole comparison view,
the manual-review queue and the alert engine are built on.

Owner: lorena.  SRS Step 11, Step 14, FR xxiii-xxx and lii.

Every test here injects thresholds rather than importing defaults, because one of the
things under test is that the taxonomy RESPONDS to config — the SRS says an evaluator may
change a threshold live, and a threshold baked into code would silently ignore them.
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
THRESHOLDS = load_thresholds()


def make_result(model: str, predicted: str, confidence: float, runner_up: str | None = None,
                runner_up_confidence: float | None = None) -> PredictionResult:
    """Build a PredictionResult with a full 10-class distribution.

    The remaining mass is spread evenly, so confidences still sum to 1 — a distribution
    that does not sum to 1 is exactly the fabricated-confidence pattern we refuse.
    """
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


# --------------------------------------------------------------------------------------
# Each status must be reachable
# --------------------------------------------------------------------------------------

def test_strong_match():
    """Agreement with a tiny confidence gap."""
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
    py = make_result("python", "Vehicle Horn", 0.95)
    gtm = make_result("gtm", "Vehicle Horn", 0.72)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.consistency_status == WEAK_MATCH
    assert result.confidence_difference == pytest.approx(0.23, abs=1e-9)


def test_model_disagreement():
    """Different classes is disagreement, regardless of how confident each model is."""
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


def test_uncertain_result_when_only_one_model_is_unsure():
    """Two models agreeing is not evidence if one of them is guessing."""
    py = make_result("python", "Alarm or Siren", 0.91)
    gtm = make_result("gtm", "Alarm or Siren", 0.20)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.consistency_status == UNCERTAIN_RESULT
    assert "GTM" in result.reason


def test_all_five_statuses_are_producible():
    """Guard against a status that is documented but unreachable."""
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


# --------------------------------------------------------------------------------------
# Boundary behaviour — the off-by-one a code reviewer will find
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("strong_max,acceptable_max,weak_max,difference,expected", [
    # threshold values and differences are all exact binary fractions, so this test
    # exercises the comparison operators and NOT floating-point representation
    (0.25, 0.50, 0.75, 0.125, STRONG_MATCH),      # comfortably inside strong
    (0.25, 0.50, 0.75, 0.250, STRONG_MATCH),      # exactly on the strong bound: inclusive
    (0.25, 0.50, 0.75, 0.500, ACCEPTABLE_MATCH),  # past strong, on the acceptable bound
    (0.25, 0.50, 0.75, 0.750, WEAK_MATCH),        # on the weak bound
    (0.25, 0.50, 0.75, 0.875, WEAK_MATCH),        # past the weak band: still a weak match
    (0.50, 0.75, 0.875, 0.250, STRONG_MATCH),     # same difference, looser config
    (0.125, 0.25, 0.50, 0.250, ACCEPTABLE_MATCH), # same difference, stricter config
])
def test_match_grading_boundaries(strong_max, acceptable_max, weak_max, difference, expected):
    """Thresholds are upper bounds and INCLUSIVE.

    A strict `<` would misgrade the boundary case, and the boundary case is precisely what
    a change of threshold produces — which is the surprise modification the SRS warns about.
    """
    thresholds = {
        # min_confidence is neutralised here so this test isolates the GRADING thresholds;
        # the min-confidence gate is exercised separately below.
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
    """The shipped config values must actually drive the verdict, not just the injected ones."""
    cons = THRESHOLDS["consistency"]
    pairs = [
        (cons["strong_match_max_diff"] / 2, STRONG_MATCH),
        (cons["acceptable_match_max_diff"] / 2, ACCEPTABLE_MATCH),
        (cons["weak_match_max_diff"] / 2, WEAK_MATCH),
    ]
    for difference, expected in pairs:
        py = make_result("python", "Gunshot", 0.95)
        gtm = make_result("gtm", "Gunshot", 0.95 - difference)
        assert classify_consistency(py, gtm, THRESHOLDS).consistency_status == expected


def test_min_confidence_boundary_is_inclusive():
    """A model exactly at the floor is accepted, not rejected."""
    threshold = THRESHOLDS["confidence"]["min_confidence"]
    py = make_result("python", "Gunshot", threshold)
    gtm = make_result("gtm", "Gunshot", threshold)
    assert classify_consistency(py, gtm, THRESHOLDS).consistency_status != UNCERTAIN_RESULT


def test_just_below_min_confidence_is_uncertain():
    threshold = THRESHOLDS["confidence"]["min_confidence"]
    py = make_result("python", "Gunshot", threshold - 0.001)
    gtm = make_result("gtm", "Gunshot", threshold - 0.002)
    assert classify_consistency(py, gtm, THRESHOLDS).consistency_status == UNCERTAIN_RESULT


# --------------------------------------------------------------------------------------
# Configurability — the SRS surprise modification
# --------------------------------------------------------------------------------------

def test_taxonomy_responds_to_changed_thresholds():
    """Change the config and the verdict must change. This is what 'not hard-coded' means."""
    py = make_result("python", "Gunshot", 0.95)
    gtm = make_result("gtm", "Gunshot", 0.80)   # difference 0.15 -> Acceptable by default

    assert classify_consistency(py, gtm, THRESHOLDS).consistency_status == ACCEPTABLE_MATCH

    stricter = {
        "confidence": dict(THRESHOLDS["confidence"]),
        "consistency": dict(THRESHOLDS["consistency"], acceptable_match_max_diff=0.05),
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
    """The audit trail must show which thresholds produced a verdict, not just the verdict."""
    result = classify_consistency(
        make_result("python", "Gunshot", 0.9), make_result("gtm", "Gunshot", 0.9), THRESHOLDS
    )
    snapshot = result.threshold_snapshot
    assert set(snapshot) == {
        "min_confidence", "strong_match_max_diff", "acceptable_match_max_diff", "weak_match_max_diff",
    }
    assert all(isinstance(v, float) for v in snapshot.values())


# --------------------------------------------------------------------------------------
# Confidence difference — the SRS's headline number
# --------------------------------------------------------------------------------------

def test_confidence_difference_is_absolute():
    """Order must not matter: |python - gtm| == |gtm - python|."""
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
    """SRS 'overlapping sounds': a strong second class means a second simultaneous event."""
    py = make_result("python", "Gunshot", 0.62, runner_up="Panic Scream", runner_up_confidence=0.30)
    gtm = make_result("gtm", "Gunshot", 0.60)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.overlapping is True
    assert result.secondary_detection == "Panic Scream"
    assert "overlapping" in result.reason.lower()


def test_no_overlap_flagged_when_models_disagree():
    """An overlap claim on a disagreement would double-count an already-confused signal."""
    py = make_result("python", "Gunshot", 0.62, runner_up="Panic Scream", runner_up_confidence=0.30)
    gtm = make_result("gtm", "Glass Breaking", 0.61)
    assert classify_consistency(py, gtm, THRESHOLDS).overlapping is False


# --------------------------------------------------------------------------------------
# Manual-review routing
# --------------------------------------------------------------------------------------

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
    """Otherwise the queue fills with clean results and reviewers stop reading it."""
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
    """A queue entry with no explanation is a usability defect — reviewers must know why."""
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
    """With 10 classes, a strong runner-up mathematically caps the top confidence.

    top + runner_up <= 1, so a runner-up at the 0.25 overlap threshold forces the top
    class to at most 0.75 — which is exactly the low-confidence band. A genuine
    overlapping detection is therefore always a low-confidence case, and routing it to a
    human is correct rather than over-cautious: the recorded label may simply be
    incomplete, and a machine should not silently choose between two real events.
    """
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
    """Below the overlap threshold the second class is just the tail of the distribution."""
    py = make_result("python", "Gunshot", 0.90, runner_up="Panic Scream", runner_up_confidence=0.02)
    gtm = make_result("gtm", "Gunshot", 0.89)
    result = classify_consistency(py, gtm, THRESHOLDS)
    assert result.overlapping is False
    assert result.secondary_detection is None
    needed, reason = requires_manual_review(result, THRESHOLDS)
    assert needed is False
    assert reason == ""


def test_comparison_serialises_for_the_api():
    """The UI, the audit record and the report all read this dict — shape is a contract."""
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
