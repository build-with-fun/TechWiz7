"""Compares the two models' outputs and assigns a consistency status (SRS Step 11).

    difference = |python_top_confidence - gtm_top_confidence|

Different classes: Model Disagreement. Same class: graded by the difference, but if both
models are confident it is at least an Acceptable Match, since the two models don't score
on the same scale (SRS Step 11). TM unsure on the same class: Weak Match. Python unsure:
Uncertain Result. Thresholds come from config/thresholds.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contract import PredictionResult

# Consistency statuses (SRS Step 11, FR xxv).
STRONG_MATCH = "Strong Match"
ACCEPTABLE_MATCH = "Acceptable Match"
WEAK_MATCH = "Weak Match"
MODEL_DISAGREEMENT = "Model Disagreement"
UNCERTAIN_RESULT = "Uncertain Result"

CONSISTENCY_STATUSES = (
    STRONG_MATCH, ACCEPTABLE_MATCH, WEAK_MATCH, MODEL_DISAGREEMENT, UNCERTAIN_RESULT,
)


@dataclass
class ComparisonResult:
    """Result of comparing the two models."""

    python_class: str
    python_confidence: float
    python_top3: list[dict[str, Any]]

    gtm_class: str
    gtm_confidence: float
    gtm_top3: list[dict[str, Any]]

    classes_agree: bool
    confidence_difference: float
    top_two_margin_python: float
    top_two_margin_gtm: float

    consistency_status: str
    reason: str
    threshold_snapshot: dict[str, float]

    secondary_detection: str | None = None
    overlapping: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": {
                "predicted_class": self.python_class,
                "confidence": round(self.python_confidence, 6),
                "top3": self.python_top3,
                "top_two_margin": round(self.top_two_margin_python, 6),
            },
            "gtm": {
                "predicted_class": self.gtm_class,
                "confidence": round(self.gtm_confidence, 6),
                "top3": self.gtm_top3,
                "top_two_margin": round(self.top_two_margin_gtm, 6),
            },
            "classes_agree": self.classes_agree,
            "confidence_difference": round(self.confidence_difference, 6),
            "consistency_status": self.consistency_status,
            "reason": self.reason,
            "secondary_detection": self.secondary_detection,
            "overlapping": self.overlapping,
            "threshold_snapshot": self.threshold_snapshot,
        }


def _top_two_margin(result: PredictionResult) -> float:
    """Gap between the top two classes."""
    top = result.top_k(2)
    if len(top) < 2:
        return float(top[0][1]) if top else 0.0
    return float(top[0][1] - top[1][1])


def classify_consistency(
    python_result: PredictionResult,
    gtm_result: PredictionResult,
    thresholds: Mapping[str, Any],
) -> ComparisonResult:
    """Compare two independent model outputs and return the consistency status."""
    conf = thresholds["confidence"]
    cons = thresholds["consistency"]

    min_confidence = float(conf["min_confidence"])
    overlap_conf = float(conf.get("overlap_secondary_confidence", 0.25))

    py_top3 = [{"class": c, "confidence": round(v, 6)} for c, v in python_result.top_k(3)]
    gtm_top3 = [{"class": c, "confidence": round(v, 6)} for c, v in gtm_result.top_k(3)]

    py_conf = float(python_result.confidence)
    gtm_conf = float(gtm_result.confidence)
    difference = abs(py_conf - gtm_conf)
    agree = python_result.predicted_class == gtm_result.predicted_class

    snapshot = {
        "min_confidence": min_confidence,
        "strong_match_max_diff": float(cons["strong_match_max_diff"]),
        "acceptable_match_max_diff": float(cons["acceptable_match_max_diff"]),
        "weak_match_max_diff": float(cons["weak_match_max_diff"]),
        # Never below the floor; without the key only the difference counts.
        "confident_agreement_min": max(float(cons.get("confident_agreement_min", 1.01)),
                                       min_confidence),
    }

    # If both agree, a strong Python runner-up suggests a second, overlapping event.
    secondary = None
    overlapping = False
    if agree and len(py_top3) >= 2 and float(py_top3[1]["confidence"]) >= overlap_conf:
        secondary = py_top3[1]["class"]
        overlapping = True

    # Checked in order:
    # 1. Both unsure -> Uncertain, even if they agree.
    if py_conf < min_confidence and gtm_conf < min_confidence:
        status = UNCERTAIN_RESULT
        reason = (
            f"Neither model reached the {min_confidence:.2f} confidence floor "
            f"(Python {py_conf:.3f}, GTM {gtm_conf:.3f}). Routed to manual review."
        )
    # 2. Different classes -> Disagreement (even if one model is also unsure).
    elif not agree:
        status = MODEL_DISAGREEMENT
        reason = (
            f"Python says '{python_result.predicted_class}' ({py_conf:.3f}) but GTM says "
            f"'{gtm_result.predicted_class}' ({gtm_conf:.3f}); difference {difference:.3f}."
        )
        if min(py_conf, gtm_conf) < min_confidence:
            weaker = "Python" if py_conf < min_confidence else "GTM"
            reason += f" The {weaker} model is also below the {min_confidence:.2f} confidence floor."
    # 3. Same class, Python unsure -> Uncertain.
    elif py_conf < min_confidence:
        status = UNCERTAIN_RESULT
        reason = (
            f"Both models chose '{python_result.predicted_class}', but the Python model is "
            f"below the {min_confidence:.2f} confidence floor (Python {py_conf:.3f}, "
            f"GTM {gtm_conf:.3f}). Routed to manual review."
        )
    # 4. Same class, TM unsure -> Weak Match.
    elif gtm_conf < min_confidence:
        status = WEAK_MATCH
        reason = (
            f"Both models chose '{python_result.predicted_class}', but the GTM model's "
            f"support is below the {min_confidence:.2f} confidence floor (Python "
            f"{py_conf:.3f}, GTM {gtm_conf:.3f})."
        )
    # 5. Same class, both confident -> grade by the confidence difference.
    else:
        both_sure = min(py_conf, gtm_conf) >= snapshot["confident_agreement_min"]
        if difference <= snapshot["strong_match_max_diff"]:
            status = STRONG_MATCH
            reason = (
                f"Both models agree on '{python_result.predicted_class}' and their "
                f"confidences differ by only {difference:.3f}."
            )
        elif difference <= snapshot["acceptable_match_max_diff"]:
            status = ACCEPTABLE_MATCH
            reason = (
                f"Both models agree on '{python_result.predicted_class}'; confidence "
                f"difference {difference:.3f} is within the acceptable band."
            )
        elif both_sure:
            status = ACCEPTABLE_MATCH
            reason = (
                f"Both models agree on '{python_result.predicted_class}' and both are "
                f"confident (Python {py_conf:.3f}, GTM {gtm_conf:.3f}); the models score "
                f"on different scales, so the {difference:.3f} gap is not held against them."
            )
        elif difference <= snapshot["weak_match_max_diff"]:
            status = WEAK_MATCH
            reason = (
                f"Both models agree on '{python_result.predicted_class}' but the "
                f"confidence difference {difference:.3f} is wide."
            )
        else:
            status = WEAK_MATCH
            reason = (
                f"Both models agree on '{python_result.predicted_class}' yet the "
                f"confidence difference {difference:.3f} exceeds the weak-match band; "
                f"the agreement is fragile and is flagged for review."
            )

    if overlapping:
        reason += f" Possible overlapping event: '{secondary}' also detected."

    return ComparisonResult(
        python_class=python_result.predicted_class,
        python_confidence=py_conf,
        python_top3=py_top3,
        gtm_class=gtm_result.predicted_class,
        gtm_confidence=gtm_conf,
        gtm_top3=gtm_top3,
        classes_agree=agree,
        confidence_difference=difference,
        top_two_margin_python=_top_two_margin(python_result),
        top_two_margin_gtm=_top_two_margin(gtm_result),
        consistency_status=status,
        reason=reason,
        threshold_snapshot=snapshot,
        secondary_detection=secondary,
        overlapping=overlapping,
    )


def requires_manual_review(comparison: ComparisonResult, thresholds: Mapping[str, Any]) -> tuple[bool, str]:
    """Whether this comparison needs manual review, and why."""
    reasons: list[str] = []

    if comparison.consistency_status in (MODEL_DISAGREEMENT, UNCERTAIN_RESULT):
        reasons.append(comparison.consistency_status)
    if comparison.consistency_status == WEAK_MATCH:
        reasons.append("Weak model agreement")
    if comparison.python_confidence < float(thresholds["confidence"]["low_confidence_band"]):
        reasons.append("Low Python confidence")
    if comparison.gtm_confidence < float(thresholds["confidence"]["low_confidence_band"]):
        reasons.append("Low GTM confidence")
    if comparison.overlapping:
        reasons.append("Possible overlapping sounds")

    return (bool(reasons), "; ".join(reasons) if reasons else "")
