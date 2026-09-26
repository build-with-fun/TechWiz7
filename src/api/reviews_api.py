"""Manual-review API (FR lvii-lxi).

The original model outputs, copied onto the review row when it was queued, are never
rewritten by a decision (FR lxi). An event can be decided once; a second decision is a 409.
Overrides and rejections need a comment. Decisions can be addressed by review id (the
console's forms) or by event id; both use the same decision code.
"""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, redirect, request, url_for
from sqlalchemy import func, select

from src.auth import capability_required, client_ip, current_user
from src.db import record_audit, session_scope
from src.errors import ApiError, current_request_id, not_found, validation_error
from src.models import Event, Review, utcnow
from src.services.config import get_store

bp = Blueprint("reviews_api", __name__)

_LOGGER = logging.getLogger(__name__)

_DECISIONS = ("confirm", "override", "reject")
_DEFAULT_PAGE_SIZE = 50
_MAX_PAGE_SIZE = 200




def _payload() -> dict:
    if request.is_json:
        body = request.get_json(silent=True)
        return body if isinstance(body, dict) else {}
    return {key: value for key, value in request.form.items() if value != ""}


def _wants_html() -> bool:
    if request.is_json:
        return False
    accept = request.accept_mimetypes
    return not (accept["application/json"] > accept["text/html"])


def _back_href() -> str:
    candidate = request.form.get("next") or request.headers.get("Referer")
    if candidate and candidate.startswith("/"):
        return candidate
    try:
        return url_for("main.reviews")
    except Exception:  # pragma: no cover
        return candidate or "/reviews"


def _review_to_dict(review: Review, *, with_context: bool = False) -> dict:
    data = {
        "id": review.id,
        "event_id": review.event_id,
        "status": review.status,
        "priority": review.priority,
        "condition_ids": review.condition_ids or [],
        "reason_text": review.reason_text,
        "recommended_action": review.recommended_action,
        "queued_at": review.queued_at.isoformat() + "Z" if review.queued_at else None,
        "assigned_to_id": review.assigned_to_id,
        "decision": review.decision,
        "final_class": review.final_class,
        "final_severity": review.final_severity,
        "comments": review.comments,
        "false_alarm": bool(review.false_alarm),
        "decided_by_id": review.decided_by_id,
        "decided_at": review.decided_at.isoformat() + "Z" if review.decided_at else None,
        "original": {
            "python": {"class": review.original_python_class,
                       "confidence": review.original_python_confidence,
                       "model_version": review.original_python_model_version},
            "gtm": {"class": review.original_gtm_class,
                    "confidence": review.original_gtm_confidence,
                    "model_version": review.original_gtm_model_version},
        },
        "original_severity": review.original_severity,
    }
    if with_context and review.event is not None:
        event = review.event
        audio = event.audio_file
        data["event"] = {
            "id": event.id,
            "predicted_class": event.predicted_class,
            "severity": event.severity,
            "consistency_status": event.consistency_status,
            "confidence_difference": event.confidence_difference,
            "quality_verdict": event.quality_verdict,
            "location": event.location,
            "filename": audio.filename if audio else None,
            "created_at": event.created_at.isoformat() + "Z" if event.created_at else None,
        }
    return data


def _load_review(session, review_id: int) -> Review:
    review = session.get(Review, review_id)
    if review is None:
        raise not_found("review item")
    return review


def _load_review_for_event(session, event_id: int) -> Review:
    """The decision target for an event: its newest open review, else its decided one."""
    review = session.execute(
        select(Review)
        .where(Review.event_id == event_id)
        .order_by(Review.queued_at.desc(), Review.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    if review is None:
        raise not_found("review item")
    return review


def _audit_decision(session, review: Review, before: dict, event: Event | None) -> None:
    record_audit(
        session,
        action="review_decided",
        actor=current_user._get_current_object(),
        target_type="review",
        target_id=str(review.id),
        outcome="success",
        detail=(
            f"Review #{review.id} on event #{review.event_id}: {review.decision}"
            + (f" -> {review.final_class}" if review.final_class else "")
        ),
        before=before,
        after={
            "status": review.status,
            "decision": review.decision,
            "final_class": review.final_class,
            "final_severity": review.final_severity,
            "false_alarm": bool(review.false_alarm),
            "decided_by_id": review.decided_by_id,
        },
        ip_address=client_ip(),
        user_agent=request.headers.get("User-Agent"),
        request_id=current_request_id(),
    )


def _apply_decision(review: Review, body: dict) -> dict:
    """Validate and apply a decision to ``review`` inside the caller's session, so write and audit
    commit together.
    """
    decision = (body.get("decision") or "").strip().lower()
    if decision not in _DECISIONS:
        raise validation_error(
            "decision must be one of confirm, override, reject.",
            decision={"given": decision, "accepted": list(_DECISIONS)},
        )
    if review.status == "Reviewed":
        raise ApiError(
            "invalid_state_transition",
            f"Review #{review.id} was already decided"
            f" ({review.decision} at {review.decided_at}); decisions are not overwritten.",
        )

    comments = (body.get("comments") or body.get("comment") or "").strip()
    if decision in {"override", "reject"} and not comments:
        raise validation_error(
            "comments are required when overriding or rejecting.",
            comments="required",
        )

    final_class = (body.get("final_class") or "").strip() or None
    final_severity = (body.get("final_severity") or "").strip() or None
    if decision == "override":
        if not final_class:
            raise validation_error(
                "final_class is required when overriding.", final_class="required"
            )
        valid_classes = list(get_store().class_names())
        if final_class not in valid_classes:
            raise validation_error(
                "Unknown class name.",
                final_class={"given": final_class, "accepted": valid_classes},
            )
    elif decision == "confirm":
        # A confirmation adopts the models' agreed answer; a reject is a non-event.
        final_class = review.original_python_class
    if final_severity is not None:
        valid_severities = list(get_store().severity_scale())
        if final_severity not in valid_severities:
            raise validation_error(
                "Unknown severity.",
                final_severity={"given": final_severity, "accepted": valid_severities},
            )

    review.status = "Reviewed"
    review.decision = decision
    review.final_class = final_class
    review.final_severity = final_severity
    review.comments = comments or None
    review.false_alarm = decision == "reject" or (
        str(body.get("false_alarm", "")).lower() == "true"
    )
    review.decided_by_id = current_user._get_current_object().id
    review.decided_at = utcnow()
    return {"decision": decision, "final_class": review.final_class,
            "final_severity": review.final_severity}




@bp.get("/reviews/queue")
@capability_required("review_queue")
def queue():
    """FR lvii: the prioritised queue, each item carrying its reason."""
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = int(request.args.get("per_page", _DEFAULT_PAGE_SIZE))
    except ValueError:
        raise validation_error("page and per_page must be integers.")
    per_page = max(1, min(per_page, _MAX_PAGE_SIZE))
    status = request.args.get("status") or "Pending Review"

    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        statement = (
            select(Review)
            .order_by(Review.priority.asc().nullslast(), Review.queued_at.asc())
        )
        if status != "all":
            statement = statement.where(Review.status == status)
        total = session.execute(
            select(func.count()).select_from(statement.subquery())
        ).scalar() or 0
        rows = session.execute(
            statement.limit(per_page).offset((page - 1) * per_page)
        ).scalars().all()

    return jsonify(
        {
            "data": [_review_to_dict(row) for row in rows],
            "meta": {
                "total": total, "page": page, "per_page": per_page,
                "pages": max(1, -(-total // per_page)), "filters": {"status": status},
            },
        }
    )


@bp.get("/reviews/history")
@capability_required("review_queue")
def history():
    """FR lxi: every decision, who made it and when -- the override audit view."""
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        rows = session.execute(
            select(Review)
            .where(Review.status == "Reviewed")
            .order_by(Review.decided_at.desc(), Review.id.desc())
            .limit(500)
        ).scalars().all()
        outcomes = session.execute(
            select(Review.decision, func.count(Review.id))
            .where(Review.status == "Reviewed")
            .group_by(Review.decision)
        ).all()

    return jsonify(
        {
            "data": [_review_to_dict(row) for row in rows],
            "meta": {"outcome_counts": {name or "pending": count for name, count in outcomes}},
        }
    )


@bp.get("/reviews/event/<int:event_id>")
@capability_required("review_queue")
def event_context(event_id: int):
    """FR lviii: full context for a decision -- the event, the originals, the reason."""
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        review = _load_review_for_event(session, event_id)
        data = _review_to_dict(review, with_context=True)
    return jsonify({"data": data})




@bp.post("/reviews/<int:review_id>/decision")
@capability_required("review_decide")
def decide_by_review(review_id: int):
    """The console's form target: decide by review row id, then defer to the shared path."""
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        review = _load_review(session, review_id)
        event_id = review.event_id
    return _decide(event_id)


@bp.post("/reviews/event/<int:event_id>/decision")
@capability_required("review_decide")
def decide_by_event(event_id: int):
    """The contract's address (FR lix-lxi): decide the newest review item for an event."""
    return _decide(event_id)


def _decide(event_id: int):
    body = _payload()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        review = _load_review_for_event(session, event_id)
        before = {
            "status": review.status,
            "decision": review.decision,
            "final_class": review.final_class,
            "original_python_class": review.original_python_class,
            "original_gtm_class": review.original_gtm_class,
        }
        outcome = _apply_decision(review, body)
        event = session.get(Event, event_id)
        if event is not None and outcome["decision"] != "pending":
            event.requires_manual_review = False
        _audit_decision(session, review, before, event)
        data = _review_to_dict(review, with_context=True)

    if _wants_html():
        return redirect(_back_href(), code=303)
    return jsonify({"data": data, "meta": {"outcome": outcome}})