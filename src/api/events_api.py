"""Events API: list and search, detail, evidence, audio stream, flag and delete.

A viewer without view_all_events sees only their own uploads, enforced inside the query.
Another user's event is a 404, not a 403. An unknown filter value is a 422 naming the
parameter, not silently ignored.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.auth import capability_required, client_ip, current_user
from src.db import session_scope
from src.errors import ApiError, current_request_id, validation_error
from src.models import Event
from src.services.config import get_store
from src.services.search import parse_filters, run_search

bp = Blueprint("events_api", __name__)

_LOGGER = logging.getLogger(__name__)

# Contract §3.2's sort names, mapped to the ones the search service knows.
_SORT_BY_NAME = {
    "created_at": "newest",
    "confidence": "confidence",
    "severity": "severity",
}


def _viewer_scope() -> tuple[int | None, bool]:
    """(viewer_id, sees_all) for the current caller."""
    user = current_user._get_current_object()
    if user is None or not getattr(user, "is_authenticated", False):
        return None, False
    sees_all = user.can("view_all_events")
    return user.id, bool(sees_all)


def _actor_dict(store) -> dict:
    """Who the pipeline acted for, when the record does not yet carry it.

    A freshly uploaded clip is persisted with ``created_by_id`` before the response is built,
    but the pipeline record itself never embeds the actor — so the caller would otherwise see
    ``null`` for an event that clearly belongs to somebody. Falls back to nulls for anonymous
    live windows, which are allowed by FR lxii.
    """
    actor = getattr(store, "actor", None)
    if actor is None or not getattr(actor, "id", None):
        return {"id": None, "username": None, "role": None}
    return {"id": actor.id, "username": actor.username, "role": actor.role}


def event_to_dict(record: dict, store) -> dict:
    """Shape a pipeline record as the API's Event; values are copied from the record, never derived
    here.
    """
    audio = record.get("audio") or {}
    quality = record.get("quality") or {}
    predictions = record.get("predictions") or {}
    comparison = record.get("comparison") or {}
    severity = record.get("severity") or {}
    alert = record.get("alert") or {}
    review = record.get("review") or {}
    decision = record.get("decision") or {}
    duplicate = record.get("duplicate") or {}
    models = record.get("model_versions") or {}
    meta = record.get("meta") or {}

    def _model_block(key: str) -> dict:
        pred = predictions.get(key) or {}
        version = (models.get(key) or {}).get("version") or pred.get("model_version")
        return {
            "name": pred.get("model_name") or key,
            "version": version,
            "predicted_class": pred.get("predicted_class"),
            "confidence": pred.get("confidence"),
            # The full distribution, so the frontend never guesses at runner-up classes.
            "confidences": pred.get("confidences"),
            "latency_sec": pred.get("latency_sec"),
        }

    alert_id = record.get("alert_id") or alert.get("id")
    raised = bool(alert.get("raised"))

    return {
        "id": record.get("event_id"),
        "audio_id": record.get("audio_id") or audio.get("audio_id"),
        "filename": meta.get("filename") or audio.get("source_path"),
        "source": "microphone" if record.get("origin") == "live" else (
            meta.get("source") or record.get("origin") or "upload"),
        "created_at": record.get("created_at"),
        "created_by": record.get("created_by") or _actor_dict(store),
        "duration_sec": audio.get("duration_sec"),
        "sample_rate": audio.get("sample_rate"),
        "sha256": audio.get("sha256"),
        "near_duplicate_of": duplicate.get("near_duplicate_of"),
        "status": record.get("event_status"),
        "quality": {
            "verdict": quality.get("verdict"),
            "score": quality.get("score"),
            "detail": quality.get("summary") or quality.get("notes"),
            "problems": quality.get("problems"),
        },
        "predicted_class": decision.get("final_class") or (
            (predictions.get("python") or {}).get("predicted_class")
        ),
        "severity": severity.get("severity"),
        "severity_display": severity.get("severity_display") or decision.get("severity_display"),
        "critical_class": severity.get("critical_class"),
        "consistency_status": comparison.get("consistency_status"),
        "confidence_difference": comparison.get("confidence_difference"),
        "requires_manual_review": bool(review.get("required")),
        "review_reason": review.get("reason_text"),
        "review_priority": review.get("priority"),
        "alert": {
            "id": alert_id,
            "status": "Open" if raised else None,
            "severity": severity.get("severity") if raised else None,
        } if raised else None,
        "location": meta.get("location"),
        "models": {
            "python": _model_block("python"),
            "gtm": _model_block("gtm"),
        },
        "timing": {
            "elapsed_ms": record.get("elapsed_ms"),
            "within_budget": record.get("within_budget"),
            "budget_sec": record.get("budget_sec"),
        },
        "config_snapshot": record.get("config_snapshot"),
        "request_id": record.get("request_id") or current_request_id(),
    }


def _event_row_to_dict(event: Event, store) -> dict:
    """Shape a stored Event row the same way, so a stored row and a fresh analysis look identical to
    the frontend.
    """
    audio = event.audio_file
    alert = event.alerts[0] if event.alerts else None
    py = event.python_model_version
    gtm = event.gtm_model_version
    creator = event.created_by

    def _score_block(model_name: str, version) -> dict:
        scores = sorted(
            (score for score in event.confidence_scores if score.model_name == model_name),
            key=lambda score: score.rank,
        )
        top = next((score for score in scores if score.is_top), scores[0] if scores else None)
        return {
            "name": model_name,
            "version": version.version if version else None,
            "predicted_class": top.class_name if top else None,
            "confidence": top.confidence if top else None,
            "confidences": {score.class_name: score.confidence for score in scores},
            "top3": [{"class": score.class_name, "confidence": score.confidence}
                     for score in scores[:3]],
        }

    return {
        "id": event.id,
        "audio_id": audio.audio_id if audio else None,
        "filename": audio.filename if audio else None,
        "source": audio.source if audio else None,
        "created_at": event.created_at.isoformat() + "Z" if event.created_at else None,
        "created_by": (
            {"id": creator.id, "username": creator.username, "role": creator.role}
            if creator else None
        ),
        "duration_sec": audio.duration_sec if audio else None,
        "sample_rate": audio.sample_rate if audio else None,
        "channels": audio.channels if audio else None,
        "bit_depth": audio.bit_depth if audio else None,
        "size_bytes": audio.size_bytes if audio else None,
        "format": (audio.container_format or audio.original_format) if audio else None,
        "sha256": audio.sha256 if audio else None,
        "near_duplicate_of": audio.near_duplicate_of_id,
        "status": event.status,
        "quality": {
            "verdict": event.quality_verdict,
            "score": event.quality_score,
            "detail": event.quality_detail,
        },
        "predicted_class": event.predicted_class,
        "severity": event.severity,
        "severity_display": event.severity,
        "critical_class": event.is_critical,
        "consistency_status": event.consistency_status,
        "confidence_difference": event.confidence_difference,
        "requires_manual_review": bool(event.requires_manual_review),
        "review_reason": event.review_reason,
        "review_priority": event.reviews[0].priority if event.reviews else None,
        "alert": (
            {"id": alert.id, "status": alert.status, "severity": alert.severity}
            if alert else None
        ),
        "location": event.location,
        "models": {
            "python": _score_block("python", py),
            "gtm": _score_block("gtm", gtm),
        },
        "timing": None,
        "config_snapshot": event.config_snapshot,
        "request_id": None,
    }


@bp.get("/events")
@capability_required("view_own_events")
def list_events():
    """Search and filter events, FR lxvii. The full validated filter set is accepted."""
    store = get_store()
    viewer_id, sees_all = _viewer_scope()

    # Sort names differ between the API and the search service; unknown values are a 422.
    sort = (request.args.get("sort") or "created_at").strip().lower()
    if sort not in _SORT_BY_NAME:
        raise validation_error(
            f"sort must be one of: {', '.join(sorted(_SORT_BY_NAME))}",
            sort=sort,
        )
    args = dict(request.args)
    args["sort"] = _SORT_BY_NAME[sort]

    filters = parse_filters(args, store, viewer=current_user._get_current_object(),
                            default_page_size=25, max_page_size=200)
    if filters.problems:
        # An unknown value in a validated parameter is a 422 naming it.
        raise validation_error(
            "Some filters were not understood",
            problems=filters.problems,
        )

    factory = current_app.config["SST_SESSION_FACTORY"]
    with session_scope(factory) as session:
        events, meta = run_search(session, filters, store)

    return jsonify({
        "data": [_event_row_to_dict(event, store) for event in events],
        "meta": {
            "total": meta["total"],
            "page": meta["page"],
            "per_page": meta["page_size"],
            "pages": meta["page_count"],
            "shown": meta["shown"],
            "filters": meta["filters"],
            "generated_at": _now_iso(),
        },
    })


@bp.get("/events/<int:event_id>")
@capability_required("view_own_events")
def get_event(event_id: int):
    """One event, with its models' versions and its evidence pointer."""
    store = get_store()
    event = _visible_event(event_id)
    return jsonify({"data": _event_row_to_dict(event, store)})


@bp.get("/events/<int:event_id>/evidence")
@capability_required("view_own_events")
def event_evidence(event_id: int):
    """Evidence for a reviewer: both models' full distributions and the quality measurements, as
    stored for this event.
    """
    event = _visible_event(event_id)
    scores: dict[str, list] = {"python": [], "gtm": []}
    for score in sorted(event.confidence_scores, key=lambda s: (s.model_name, s.rank)):
        scores.setdefault(score.model_name, []).append({
            "class": score.class_name,
            "confidence": score.confidence,
            "rank": score.rank,
            "is_top": score.is_top,
        })
    audio = event.audio_file
    return jsonify({
        "data": {
            "event_id": event.id,
            "audio_id": audio.audio_id if audio else None,
            "sha256": audio.sha256 if audio else None,
            "models": {
                name: {
                    "version": (event.python_model_version if name == "python"
                                else event.gtm_model_version).version
                    if (event.python_model_version if name == "python"
                        else event.gtm_model_version) else None,
                    "predicted_class": event.predicted_class,
                    "confidences": scores.get(name, []),
                }
                for name in ("python", "gtm")
            },
            "quality": {
                "verdict": event.quality_verdict,
                "score": event.quality_score,
                "detail": event.quality_detail,
            },
            "severity": {
                "severity": event.severity,
                "display": event.severity,
                "critical_class": event.is_critical,
            },
            "comparison": {
                "consistency_status": event.consistency_status,
                "confidence_difference": event.confidence_difference,
            },
            "fingerprint": audio.perceptual_fingerprint if audio else None,
            "stored_path": audio.stored_path if audio else None,
        },
    })


@bp.get("/events/<int:event_id>/audio")
@capability_required("view_own_events")
def event_audio(event_id: int):
    """Stream the stored audio. The owner, or anyone who can see all events."""
    event = _visible_event(event_id)
    audio = event.audio_file
    if audio is None:
        raise ApiError("not_found", "This event has no stored audio.")
    if not audio.stored_path:
        raise ApiError("no_audio", "This recording has reached its retention limit.")
    storage = current_app.config["SST_STORAGE"]
    path = storage.resolve(audio.stored_path)
    if not path.is_file():
        raise ApiError(
            "storage_error",
            "The audio for this event is no longer on disk.",
            details={"audio_id": audio.audio_id},
        )
    return send_file(
        str(path),
        mimetype=audio.content_type or "application/octet-stream",
        as_attachment=True,
        download_name=f"{audio.audio_id}{Path(audio.stored_path).suffix or '.wav'}",
    )


@bp.post("/events/<int:event_id>/flag")
@capability_required("view_own_events")
def flag_event(event_id: int):
    """Flag an event for a reviewer's attention (audited). Does not change the classification."""
    from src.db import record_audit

    event = _visible_event(event_id)
    note = (request.form.get("note") or request.json.get("note", "")
            if request.is_json else request.form.get("note", "")).strip() if request.data else ""
    factory = current_app.config["SST_SESSION_FACTORY"]
    with session_scope(factory) as session:
        stored = session.execute(
            select(Event).where(Event.id == event.id)
        ).scalar_one()
        record_audit(
            session,
            action="event_flagged",
            actor=current_user._get_current_object(),
            target_type="event",
            target_id=str(stored.id),
            outcome="success",
            detail=f"flagged by {current_user.username} for reviewer attention",
            after={"note": note, "status": stored.status},
            ip_address=client_ip(),
            user_agent=request.headers.get("User-Agent"),
            request_id=current_request_id(),
        )
    return jsonify({"data": {"event_id": event.id, "flagged": True, "note": note}})


@bp.delete("/events/<int:event_id>")
@capability_required("view_all_events")
def delete_event(event_id: int):
    """Delete an event and its audio (reviewer and above), audited with a before-image."""
    from src.db import record_audit

    event = _visible_event(event_id)
    audio = event.audio_file
    before = {
        "event_id": event.id,
        "audio_id": audio.audio_id if audio else None,
        "sha256": audio.sha256 if audio else None,
        "predicted_class": event.predicted_class,
        "status": event.status,
    }
    factory = current_app.config["SST_SESSION_FACTORY"]
    with session_scope(factory) as session:
        stored = session.execute(select(Event).where(Event.id == event.id)).scalar_one()
        session.delete(stored)
        record_audit(
            session,
            action="event_deleted",
            actor=current_user._get_current_object(),
            target_type="event",
            target_id=str(event.id),
            outcome="success",
            detail=f"deleted by {current_user.username}",
            before=before,
            ip_address=client_ip(),
            user_agent=request.headers.get("User-Agent"),
            request_id=current_request_id(),
        )
    _LOGGER.info("event %s deleted by %s", event.id, current_user.username)
    return jsonify({"data": {"deleted": True, "event_id": event.id}})


# Internals

def _now_iso() -> str:
    from src.models import utcnow

    return utcnow().isoformat() + "Z"


def _visible_event(event_id: int) -> Event:
    """The event if this caller may see it, else 404 (single-resource routes bypass the search
    scoping).
    """
    viewer_id, sees_all = _viewer_scope()
    factory = current_app.config["SST_SESSION_FACTORY"]
    with session_scope(factory) as session:
        statement = (
            select(Event)
            .where(Event.id == event_id)
            .options(
                selectinload(Event.audio_file),
                selectinload(Event.created_by),
                selectinload(Event.python_model_version),
                selectinload(Event.gtm_model_version),
                selectinload(Event.confidence_scores),
                selectinload(Event.alerts),
                selectinload(Event.reviews),
            )
        )
        if not sees_all:
            statement = statement.where(Event.created_by_id == viewer_id)
        event = session.execute(statement).scalar_one_or_none()
        if event is None:
            raise ApiError("not_found", "No event has that id.")
        # Relationships are loaded while the session is open, so serialising after it closes is
        # safe.
        session.expunge(event)
    return event
