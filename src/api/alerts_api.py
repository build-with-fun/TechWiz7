"""Alert console API (FR liii-lvi): list, detail, history, acknowledge, dismiss, escalate.

Transitions are explicit: acknowledging does not resolve, dismissing records why
(resolution_note, is_false_alarm), escalating records the new severity. Every transition is
audited with before and after. Form posts get a redirect back to the console, JSON callers
the JSON envelope; the rules are the same.
"""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, flash, jsonify, redirect, request, url_for
from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from src.auth import capability_required, client_ip, current_user
from src.db import record_audit, session_scope
from src.errors import ApiError, current_request_id, not_found, validation_error
from src.models import ALERT_STATUSES, Alert, utcnow
from src.services.config import get_store

bp = Blueprint("alerts_api", __name__)

_LOGGER = logging.getLogger(__name__)

_DEFAULT_PAGE_SIZE = 50
_MAX_PAGE_SIZE = 200




def _payload() -> dict:
    """Read a write payload from JSON or a form post; both go through the same validation."""
    if request.is_json:
        body = request.get_json(silent=True)
        return body if isinstance(body, dict) else {}
    return {key: value for key, value in request.form.items() if value != ""}


def _wants_html() -> bool:
    """True for the console's plain form posts: they should land back on the page."""
    if request.is_json:
        return False
    accept = request.accept_mimetypes
    return not (accept["application/json"] > accept["text/html"])


def _back_href(default_endpoint: str, **params) -> str:
    candidate = request.form.get("next") or request.headers.get("Referer")
    if candidate and candidate.startswith("/"):
        return candidate
    try:
        return url_for(default_endpoint, **params)
    except Exception:  # pragma: no cover - only when pages slice is absent
        return candidate or "/alerts"


def _alert_to_dict(alert: Alert, *, with_evidence: bool = False) -> dict:
    event = alert.event
    rule = {}
    if alert.rule_snapshot:
        rule = alert.rule_snapshot if isinstance(alert.rule_snapshot, dict) else {}
    data = {
        "id": alert.id,
        "event_id": alert.event_id,
        "severity": alert.severity,
        "status": alert.status,
        "rule_class": alert.rule_class,
        "rule_snapshot": rule,
        "recommended_action": alert.recommended_action,
        "message": alert.message,
        "escalated_to_severity": alert.escalated_to_severity,
        "dedup_key": alert.dedup_key,
        "created_at": alert.created_at.isoformat() + "Z" if alert.created_at else None,
        "acknowledged_by_id": alert.acknowledged_by_id,
        "acknowledged_at": (
            alert.acknowledged_at.isoformat() + "Z" if alert.acknowledged_at else None
        ),
        "resolved_by_id": alert.resolved_by_id,
        "resolved_at": alert.resolved_at.isoformat() + "Z" if alert.resolved_at else None,
        "resolution_note": alert.resolution_note,
        "is_false_alarm": bool(alert.is_false_alarm),
        "notified_channels": alert.notified_channels or [],
    }
    if with_evidence and alert.event is not None:
        event = alert.event
        data["event"] = {
            "id": event.id,
            "predicted_class": event.predicted_class,
            "severity": event.severity,
            "consistency_status": event.consistency_status,
            "confidence_difference": event.confidence_difference,
            "quality_verdict": event.quality_verdict,
            "location": event.location,
            "created_at": event.created_at.isoformat() + "Z" if event.created_at else None,
        }
    return data


def _load_alert(session, alert_id: int) -> Alert:
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise not_found("alert")
    return alert


def _audit_transition(
    session, *, action: str, alert: Alert, before: dict, after: dict, detail: str
) -> None:
    record_audit(
        session,
        action=action,
        actor=current_user._get_current_object(),
        target_type="alert",
        target_id=str(alert.id),
        outcome="success",
        detail=detail,
        before=before,
        after=after,
        ip_address=client_ip(),
        user_agent=request.headers.get("User-Agent"),
        request_id=current_request_id(),
    )




@bp.get("/alerts")
@capability_required("view_alerts")
def list_alerts():
    """FR liii: the filterable alert list. ``status``/``severity`` are validated."""
    store = get_store()
    valid_severities = list(store.severity_scale())
    statuses = list(ALERT_STATUSES) + ["all"]

    status = request.args.get("status") or "all"
    if status not in statuses:
        raise validation_error(
            "Unknown alert status.", status={"given": status, "accepted": statuses}
        )
    severities = [value for value in request.args.getlist("severity") if value]
    unknown = [value for value in severities if value not in valid_severities]
    if unknown:
        raise validation_error(
            "Unknown severity.",
            severity={"given": unknown, "accepted": valid_severities},
        )

    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = int(request.args.get("per_page", _DEFAULT_PAGE_SIZE))
    except ValueError:
        raise validation_error("page and per_page must be integers.")
    per_page = max(1, min(per_page, _MAX_PAGE_SIZE))

    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        statement = select(Alert).order_by(Alert.created_at.desc(), Alert.id.desc())
        if status != "all":
            statement = statement.where(Alert.status == status)
        if severities:
            statement = statement.where(Alert.severity.in_(severities))
        total = session.execute(
            select(func.count()).select_from(statement.subquery())
        ).scalar() or 0
        rows = session.execute(
            statement.limit(per_page).offset((page - 1) * per_page)
        ).scalars().all()
        data = [_alert_to_dict(row) for row in rows]

    return jsonify(
        {
            "data": data,
            "meta": {
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": max(1, -(-total // per_page)),
                "filters": {"status": status, "severity": severities},
            },
        }
    )


@bp.get("/alerts/history")
@capability_required("view_alerts")
def alert_history():
    """FR lvi: every past alert and its outcome -- acknowledged, dismissed, escalated."""
    store = get_store()
    valid_severities = list(store.severity_scale())
    severities = [value for value in request.args.getlist("severity") if value]
    unknown = [value for value in severities if value not in valid_severities]
    if unknown:
        raise validation_error(
            "Unknown severity.",
            severity={"given": unknown, "accepted": valid_severities},
        )
    try:
        limit = min(500, max(1, int(request.args.get("limit", 200))))
    except ValueError:
        raise validation_error("limit must be an integer.")

    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        statement = (
            select(Alert)
            # _alert_to_dict reads alert.event after the session closes, so load it now.
            .options(joinedload(Alert.event))
            .where(Alert.status != "Open")
            .order_by(Alert.created_at.desc(), Alert.id.desc())
            .limit(limit)
        )
        if severities:
            statement = statement.where(Alert.severity.in_(severities))
        rows = session.execute(statement).scalars().all()
        by_severity = session.execute(
            select(Alert.severity, func.count(Alert.id)).group_by(Alert.severity)
        ).all()
        false_alarms = session.execute(
            select(func.count(Alert.id)).where(Alert.is_false_alarm.is_(True))
        ).scalar() or 0

    return jsonify(
        {
            "data": [_alert_to_dict(row) for row in rows],
            "meta": {
                "outcome_counts": {name: count for name, count in by_severity},
                "false_alarm_count": false_alarms,
            },
        }
    )


@bp.get("/alerts/<int:alert_id>")
@capability_required("view_alerts")
def get_alert(alert_id: int):
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        alert = _load_alert(session, alert_id)
        data = _alert_to_dict(alert, with_evidence=True)
    return jsonify({"data": data})




@bp.post("/alerts/<int:alert_id>/acknowledge")
@capability_required("acknowledge_alerts")
def acknowledge(alert_id: int):
    """FR lv: acknowledge an open alert. Idempotent: re-acknowledging is a no-op."""
    return _transition(alert_id, "acknowledge")


@bp.post("/alerts/<int:alert_id>/dismiss")
@capability_required("acknowledge_alerts")
def dismiss(alert_id: int):
    """FR lv: dismiss as a false alarm -- the reason is mandatory, and stored."""
    body = _payload()
    reason = (body.get("reason") or body.get("note") or "").strip()
    if not reason:
        raise validation_error("A dismissal reason is required.", reason="required")
    return _transition(
        alert_id,
        "dismiss",
        resolution_note=reason,
        is_false_alarm=str(body.get("false_alarm", "true")).lower() != "false",
    )


@bp.post("/alerts/<int:alert_id>/escalate")
@capability_required("acknowledge_alerts")
def escalate(alert_id: int):
    """FR lvi: escalate -- the target severity must exist on the active scale."""
    store = get_store()
    valid_severities = list(store.severity_scale())
    body = _payload()
    note = (body.get("note") or body.get("reason") or "").strip()
    target = body.get("severity") or body.get("to")
    if target not in valid_severities:
        raise validation_error(
            "Unknown escalation severity.",
            severity={"given": target, "accepted": valid_severities},
        )
    return _transition(alert_id, "escalate", escalated_to_severity=target, note=note)


def _redirect_with_error(alert_id: int, error: ApiError):
    """Send a failed form post back to the console with a flash message instead of a raw 409."""
    flash(f"Alert #{alert_id}: {error.message}", "error")
    return redirect(_back_href("main.alerts", status="Open"), code=303)


def _transition(alert_id: int, action: str, **changes) -> "jsonify | redirect":
    """Apply one transition with its audit row. Dismissed and Closed are final; anything else
    may still be acknowledged, escalated or dismissed.
    """
    if action not in {"acknowledge", "dismiss", "escalate"}:  # pragma: no cover
        raise ApiError("bad_request", f"Unknown alert action {action!r}.")
    now = utcnow()
    user = current_user._get_current_object()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        alert = _load_alert(session, alert_id)
        before = {
            "status": alert.status,
            "acknowledged_by_id": alert.acknowledged_by_id,
            "resolved_by_id": alert.resolved_by_id,
            "escalated_to_severity": alert.escalated_to_severity,
            "is_false_alarm": bool(alert.is_false_alarm),
        }
        if alert.status in {"Dismissed", "Closed"}:
            error = ApiError(
                "invalid_state_transition",
                f"Alert #{alert.id} is {alert.status}; it cannot be {action}ed.",
            )
            if _wants_html():
                return _redirect_with_error(alert.id, error)
            raise error
        # Until 26 Sep only Open alerts changed; later actions returned 200 and did nothing.
        if action == "acknowledge":
            if alert.acknowledged_by_id is None:
                alert.acknowledged_by_id = user.id
                alert.acknowledged_at = now
            if alert.status == "Open":
                alert.status = "Acknowledged"
        elif action == "dismiss":
            alert.status = "Dismissed"
            alert.resolved_by_id = user.id
            alert.resolved_at = now
            alert.resolution_note = changes["resolution_note"]
            alert.is_false_alarm = changes["is_false_alarm"]
        else:
            alert.status = "Escalated"
            alert.escalated_to_severity = changes["escalated_to_severity"]
            if changes.get("note"):
                alert.resolution_note = changes["note"]
        after = {
            "status": alert.status,
            "acknowledged_by_id": alert.acknowledged_by_id,
            "resolved_by_id": alert.resolved_by_id,
            "escalated_to_severity": alert.escalated_to_severity,
            "is_false_alarm": bool(alert.is_false_alarm),
        }
        _audit_transition(
            session,
            action=f"alert_{action}",
            alert=alert,
            before=before,
            after=after,
            detail=f"Alert #{alert.id} ({alert.severity}) -> {alert.status}",
        )
        data = _alert_to_dict(alert)
        outcome_status = alert.status

    if _wants_html():
        flash(f"Alert #{alert_id} {outcome_status.lower()}.", "success")
        return redirect(_back_href("main.alerts", status="Open"), code=303)
    return jsonify({"data": data, "meta": {"outcome": outcome_status}})