"""``/api/events`` -- the central resource of the console.

Owner: sara.

Read paths are list + detail + evidence + the binary stream; the write paths are the flag
and the delete. The interesting part is not any single one of them, it is the two rules
they all share:

* **Scoping is not optional.** A viewer without ``view_all_events`` sees only their own
  uploads, applied as a hard constraint inside the search service rather than as a filter
  the caller can drop. A request for another user's event is a 404, not a 403 -- the
  resource does not exist for them, and telling them it does leaks the count.
* **A bad filter is a 422.** Silently ignoring an unknown ``severity`` shows the operator a
  result set that quietly excludes what they asked for, which reads as a search bug. The
  search service collects those problems; this layer turns them into the structured error.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from sqlalchemy import select

from src.auth import capability_required, client_ip, current_user
from src.db import session_scope
from src.errors import ApiError, current_request_id, validation_error
from src.models import AudioFile, Event
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
    """What this caller is allowed to see.

    Returns ``(viewer_id, sees_all)``. ``viewer_id`` is None only for an anonymous caller,
    which the decorator on the route already prevents; the search service still defends.
    """
    user = current_user._get_current_object()
    if user is None or not getattr(user, "is_authenticated", False):
        return None, False
    sees_all = user.can("view_all_events")
    return user.id, bool(sees_all)


def event_to_dict(record: dict, store) -> dict:
    """Render a pipeline record as the contract's ``Event`` (§3.1).

    Accepts the record straight out of ``analyse()`` -- the audio, quality, predictions,
    comparison, severity, alert, review and decision blocks are all present on it, so this
    is a reshaping, not a re-derivation. Nothing here may invent a number: every value comes
    from the record the models actually produced.
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
            # The per-class distribution is what the console uses for the bar chart; it is
            # included so the frontend never has to guess at the runner-up classes.
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
        "created_by": record.get("created_by") or {"id": None, "username": None, "role": None},
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
    """Render a stored ``Event`` row as the same shape.

    The console reads the same resource from the database (list, detail) and from the
    pipeline (the 201 of an upload). Keeping one shape means a row and a fresh analysis are
    interchangeable to the frontend, which is what the contract's §3.1 example is.
    """
    audio = event.audio_file
    alert = event.alerts[0] if event.alerts else None
    py = event.python_model_version
    gtm = event.gtm_model_version
    creator = event.created_by

    def _score_block(model_name: str, version, predicted: str | None) -> dict:
        return {
            "name": model_name,
            "version": version.version if version else None,
            "predicted_class": predicted,
            "confidence": event.top_confidence if predicted is not None else None,
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
        "sha256": audio.sha256 if audio else None,
        "near_duplicate_of": audio.near_duplicate_of.audio_id
            if audio and audio.near_duplicate_of else None,
        "status": event.status,
        "quality": {
            "verdict": event.quality_verdict,
            "score": event.quality_score,
            "detail": event.quality_detail,
        },
        "predicted_class": event.predicted_class,
        "severity": event.severity,
        "severity_display": event.severity_display,
        "critical_class": event.critical_class,
        "consistency_status": event.consistency_status,
        "confidence_difference": event.confidence_difference,
        "requires_manual_review": bool(event.requires_manual_review),
        "review_reason": event.review_reason,
        "review_priority": event.review_priority,
        "alert": (
            {"id": alert.id, "status": alert.status, "severity": alert.severity}
            if alert else None
        ),
        "location": event.location,
        "models": {
            "python": _score_block("python", py, event.predicted_class),
            "gtm": _score_block("gtm", gtm, event.predicted_class),
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

    # The contract's names are the public ones; the search service works in its own. Sort
    # is the one that genuinely differs, so it is translated and unknown values are a 422.
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
        # Contract §3.2: an unknown value in a validated param is a 422 naming the param and
        # the accepted values. Ignoring it would show a result set that silently omits what
        # the operator asked for.
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
    """The evidence bundle for an event: both models' full distributions and the quality
    measurements, so a reviewer can see what the models saw before deciding.

    The per-class confidences are stored per event and per model (``ConfidenceScore``), so
    a reviewer who overrides a decision is arguing with the record as it stood, not with
    the current active model.
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
                "display": event.severity_display,
                "critical_class": event.critical_class,
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
    """Mark an event for a reviewer's attention. Any viewer of the event may flag it.

    Flagging is not a decision: it does not change the classification, the severity or the
    alert. It is the operator's way of saying "someone should look at this", and it is
    audited so the reviewer can see who flagged it and when.
    """
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
    """Delete an event and its audio. Reviewer and above.

    Destructive and therefore audited with a before-image: the deletion record carries what
    was there, so the audit trail can answer "what did we delete and why".
    """
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


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------

def _now_iso() -> str:
    from src.models import utcnow

    return utcnow().isoformat() + "Z"


def _visible_event(event_id: int) -> Event:
    """The event if this caller may see it, else a 404.

    Scoping is enforced here rather than in the search service because the single-resource
    paths bypass the query builder. A 404 rather than a 403: whether the row exists at all
    is not information an unauthorised caller should get.
    """
    viewer_id, sees_all = _viewer_scope()
    factory = current_app.config["SST_SESSION_FACTORY"]
    with session_scope(factory) as session:
        statement = select(Event).where(Event.id == event_id)
        if not sees_all:
            statement = statement.where(Event.created_by_id == viewer_id)
        event = session.execute(statement).scalar_one_or_none()
        if event is None:
            raise ApiError("not_found", "No event has that id.")
        # Detach a fully loaded copy: the relationships the serializer needs are loaded here
        # while the session is open, and the response is built after it closes.
        session.expunge(event)
    return event
