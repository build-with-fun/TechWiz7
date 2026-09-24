"""
Cross-model comparison and consistency classification.

Owner: lorena.  SRS Step 11, Step 14, FR xxiii-xxx, lii, Deliverable 3.

THE TAXONOMY (SRS Step 11)
--------------------------
    difference = |python_top_confidence - gtm_top_confidence|

    Strong Match          both models agree on the class, difference is small
    Acceptable Match      both agree, difference is moderate
    Weak Match            both agree, but the confidence gap is wide
    Model Disagreement    the two models name different classes
    Uncertain Result      neither model is confident enough to be relied on

Ordering of the checks matters and is argued in the module docstrings below, because the
taxonomy is not fully specified and the order is where the judgement lives.

Every threshold is injected from config/thresholds.json. None is a literal here — the SRS
says an evaluator may demand a changed threshold, and a threshold buried in this file
would require a code change to satisfy them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contract import PredictionResult

# Consistency statuses — the exact vocabulary required by SRS Step 11 / FR xxv.
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
    """The full cross-model verdict, as the UI, the audit trail and the report need it."""

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
    """Gap between the best and second-best class.

    A large margin means the model is decisive; a tiny margin means two classes are
    nearly tied and the "prediction" is close to a coin toss. The SRS asks for this
    explicitly, and it is the honest way to explain a low-confidence result.
    """
    top = result.top_k(2)
    if len(top) < 2:
        return float(top[0][1]) if top else 0.0
    return float(top[0][1] - top[1][1])


def classify_consistency(
    python_result: PredictionResult,
    gtm_result: PredictionResult,
    thresholds: Mapping[str, Any],
) -> ComparisonResult:
    """Apply the SRS consistency taxonomy to two independent model outputs.

    Precondition the caller must uphold: `python_result` and `gtm_result` come from
    models that have never seen each other's output. The GTM model receives audio only.
    If anyone ever feeds a Python confidence into the GTM path, this function's
    output becomes meaningless — and the integrity rules would be broken.
    """
    conf = thresholds["confidence"]
    cons = thresholds["consistency"]

    min_confidence = float(conf["min_confidence"])
    low_band = float(conf.get("low_confidence_band", min_confidence))
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
    }

    # --- Overlap / secondary event detection -----------------------------------------
    # Only meaningful when BOTH models agree on the primary class; then a strong runner-up
    # in the Python model suggests a second simultaneous event (SRS "overlapping sounds").
    secondary = None
    overlapping = False
    if agree and len(py_top3) >= 2 and float(py_top3[1]["confidence"]) >= overlap_conf:
        secondary = py_top3[1]["class"]
        overlapping = True

    # --- Decision order -----------------------------------------------------------------
    # 1. Both models unsure -> Uncertain Result, even if they happen to agree. Two models
    #    agreeing on a guess is not evidence; the SRS lists "Uncertain Result" precisely so
    #    that a weak agreement is not dressed up as a match.
    if py_conf < min_confidence and gtm_conf < min_confidence:
        status = UNCERTAIN_RESULT
        reason = (
            f"Neither model reached the {min_confidence:.2f} confidence floor "
            f"(Python {py_conf:.3f}, GTM {gtm_conf:.3f}). Routed to manual review."
        )
    # 2. One model confident, the other not: too weak to act on, regardless of class.
    elif py_conf < min_confidence or gtm_conf < min_confidence:
        weaker = "Python" if py_conf < min_confidence else "GTM"
        status = UNCERTAIN_RESULT
        reason = (
            f"{weaker} model below the {min_confidence:.2f} confidence floor "
            f"(Python {py_conf:.3f}, GTM {gtm_conf:.3f}). Routed to manual review."
        )
    # 3. Different classes: the models contradict each other. Report the disagreement
    #    rather than picking a winner — the SRS wants that conflict visible to a reviewer.
    elif not agree:
        status = MODEL_DISAGREEMENT
        reason = (
            f"Python says '{python_result.predicted_class}' ({py_conf:.3f}) but GTM says "
            f"'{gtm_result.predicted_class}' ({gtm_conf:.3f}); difference {difference:.3f}."
        )
    # 4. Same class and both confident: grade the agreement by how close the confidences are.
    else:
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
    """SRS Step 19 / FR lvii: which cases go to a human.

    Returns (needs_review, reason) so the queue can show WHY an item is there; an
    unexplained queue entry is a usability defect the reviewers will (rightly) flag.
    """
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
