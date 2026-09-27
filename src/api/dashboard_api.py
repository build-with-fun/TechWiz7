"""Dashboard and analytics JSON (FR lxiv-lxviii).

The window is ``hours`` (1-8760, default one week).
"""

from __future__ import annotations

import datetime as _dt
import logging
import typing as t

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import case, func, select

from src.auth import capability_required
from src.db import session_scope
from src.errors import validation_error
from src.models import Alert, Event, Review, utcnow
from src.services.config import get_store

bp = Blueprint("dashboard_api", __name__)

_LOGGER = logging.getLogger(__name__)


def _window() -> tuple[int, "t.Any"]:
    """Parse ``hours``; out of range is a 422."""
    raw = request.args.get("hours")
    if raw is None:
        hours = 24 * 7
    else:
        try:
            hours = int(raw)
        except ValueError:
            raise validation_error("hours must be an integer.", hours={"given": raw})
        if hours < 1 or hours > 24 * 365:
            raise validation_error(
                "hours must be between 1 and 8760.", hours={"given": raw}
            )
    since = utcnow() - _dt.timedelta(hours=hours)
    return hours, since


def _counts(session, column, since=None) -> list[dict]:
    statement = select(column, func.count(Event.id)).group_by(column)
    if since is not None:
        statement = statement.where(Event.created_at >= since)
    return [{str(key): count, "count": count} for key, count in
            session.execute(statement).all()]




@bp.get("/dashboard/summary")
@capability_required("view_dashboards")
def summary():
    """FR lxiv: totals for the dashboard cards."""
    hours_window, since = _window()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        totals = {
            "events_total": session.execute(select(func.count(Event.id))).scalar() or 0,
            "events_window": session.execute(
                select(func.count(Event.id)).where(Event.created_at >= since)
            ).scalar() or 0,
            "open_alerts": session.execute(
                select(func.count(Alert.id)).where(Alert.status == "Open")
            ).scalar() or 0,
            "acknowledged_alerts": session.execute(
                select(func.count(Alert.id)).where(Alert.status == "Acknowledged")
            ).scalar() or 0,
            "queue_depth": session.execute(
                select(func.count(Review.id)).where(Review.status == "Pending Review")
            ).scalar() or 0,
            "awaiting_review": session.execute(
                select(func.count(Event.id)).where(Event.requires_manual_review.is_(True))
            ).scalar() or 0,
            "critical_window": session.execute(
                select(func.count(Event.id)).where(
                    Event.created_at >= since, Event.severity == "Critical"
                )
            ).scalar() or 0,
        }
        by_severity = _counts(session, Event.severity, since)
        by_status = _counts(session, Event.status, since)
        by_consistency = _counts(session, Event.consistency_status, since)
    return jsonify(
        {
            "data": {
                "totals": totals,
                "by_severity": by_severity,
                "by_status": by_status,
                "by_consistency": by_consistency,
            },
            "meta": {"window_hours": hours_window},
        }
    )


@bp.get("/dashboard/timeline")
@capability_required("view_dashboards")
def timeline():
    """Events per day for the trend chart."""
    hours, since = _window()
    bucket = request.args.get("bucket") or "hour"
    if bucket not in {"hour", "day"}:
        raise validation_error(
            "bucket must be hour or day.", bucket={"given": bucket, "accepted": ["hour", "day"]}
        )
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        rows = session.execute(
            select(Event.created_at).where(Event.created_at >= since).order_by(Event.created_at)
        ).scalars().all()
    buckets: dict[str, int] = {}
    for created_at in rows:
        if created_at is None:
            continue
        if bucket == "hour":
            key = created_at.strftime("%Y-%m-%dT%H:00Z")
        else:
            key = created_at.strftime("%Y-%m-%d")
        buckets[key] = buckets.get(key, 0) + 1
    series = [{"bucket": key, "count": count} for key, count in sorted(buckets.items())]
    return jsonify(
        {"data": series, "meta": {"hours": hours, "bucket": bucket, "total": sum(buckets.values())}}
    )




@bp.get("/analytics/classes")
@capability_required("view_analytics")
def analytics_classes():
    """FR lxviii: counts and average confidences per class."""
    hours, since = _window()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        rows = session.execute(
            select(
                Event.predicted_class,
                func.count(Event.id),
                func.avg(Event.top_confidence),
                func.avg(Event.confidence_difference),
            )
            .where(Event.created_at >= since)
            .group_by(Event.predicted_class)
        ).all()
    data = [
        {
            "class": name,
            "count": count,
            "mean_confidence": round(float(avg_conf or 0), 4),
            "mean_confidence_difference": round(float(avg_diff or 0), 4),
        }
        for name, count, avg_conf, avg_diff in rows
    ]
    return jsonify({"data": data, "meta": {"hours": hours}})


@bp.get("/analytics/model-comparison")
@capability_required("view_analytics")
def analytics_model_comparison():
    """FR lxix: agreement rate, disagreements and confidence differences."""
    hours, since = _window()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        total = session.execute(
            select(func.count(Event.id)).where(Event.created_at >= since)
        ).scalar() or 0
        agreed = session.execute(
            select(func.count(Event.id)).where(
                Event.created_at >= since,
                Event.consistency_status.in_(["Strong Match", "Acceptable Match", "Weak Match"]),
            )
        ).scalar() or 0
        differences = session.execute(
            select(Event.confidence_difference)
            .where(Event.created_at >= since, Event.confidence_difference.is_not(None))
        ).scalars().all()
        bands = {"0.00-0.05": 0, "0.05-0.10": 0, "0.10-0.20": 0, "0.20+": 0}
        for diff in differences:
            value = float(diff)
            key = (
                "0.00-0.05" if value < 0.05
                else "0.05-0.10" if value < 0.10
                else "0.10-0.20" if value < 0.20
                else "0.20+"
            )
            bands[key] += 1
    data = {
        "total": total,
        "class_agreements": agreed,
        "disagreements": total - agreed,
        "agreement_rate": round(agreed / total, 4) if total else None,
        "confidence_difference_distribution": bands,
        "mean_confidence_difference": (
            round(sum(float(d) for d in differences) / len(differences), 4)
            if differences else None
        ),
    }
    return jsonify({"data": data, "meta": {"hours": hours}})


@bp.get("/analytics/consistency")
@capability_required("view_analytics")
def analytics_consistency():
    """FR xxxiii: counts per consistency status."""
    hours, since = _window()
    store = get_store()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        counts = dict(
            session.execute(
                select(Event.consistency_status, func.count(Event.id))
                .where(Event.created_at >= since)
                .group_by(Event.consistency_status)
            ).all()
        )
    ordering = (
        store.consistency_ordering()
        if hasattr(store, "consistency_ordering")
        else list(counts)
    )
    data = [
        {"status": status, "count": counts.get(status, 0)}
        for status in dict.fromkeys([*ordering, *counts])
    ]
    return jsonify({"data": data, "meta": {"hours": hours}})


@bp.get("/analytics/alerts")
@capability_required("view_analytics")
def analytics_alerts():
    """FR lxx: alerts by severity and status, and the false-alarm rate."""
    hours, since = _window()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        by_severity = session.execute(
            select(Alert.severity, func.count(Alert.id))
            .where(Alert.created_at >= since)
            .group_by(Alert.severity)
        ).all()
        by_status = session.execute(
            select(Alert.status, func.count(Alert.id))
            .where(Alert.created_at >= since)
            .group_by(Alert.status)
        ).all()
        total = session.execute(
            select(func.count(Alert.id)).where(Alert.created_at >= since)
        ).scalar() or 0
        false_alarms = session.execute(
            select(func.count(Alert.id)).where(
                Alert.created_at >= since, Alert.is_false_alarm.is_(True)
            )
        ).scalar() or 0
        escalated = session.execute(
            select(func.count(Alert.id)).where(
                Alert.created_at >= since, Alert.status == "Escalated"
            )
        ).scalar() or 0
    data = {
        "by_severity": {name: count for name, count in by_severity},
        "by_status": {name: count for name, count in by_status},
        "total": total,
        "false_alarm_count": false_alarms,
        "false_alarm_rate": round(false_alarms / total, 4) if total else None,
        "escalated_count": escalated,
    }
    return jsonify({"data": data, "meta": {"hours": hours}})


@bp.get("/analytics/quality")
@capability_required("view_analytics")
def analytics_quality():
    """FR lxx: audio quality and how it affects agreement."""
    hours, since = _window()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        rows = session.execute(
            select(
                Event.quality_verdict,
                func.count(Event.id),
                func.avg(case((Event.consistency_status == "Model Disagreement", 0), else_=1)),
                func.avg(Event.confidence_difference),
            )
            .where(Event.created_at >= since)
            .group_by(Event.quality_verdict)
        ).all()
    data = [
        {
            "verdict": verdict,
            "count": count,
            "agreement_rate": round(float(agreement or 0), 4),
            "mean_confidence_difference": round(float(avg_diff or 0), 4),
        }
        for verdict, count, agreement, avg_diff in rows
    ]
    return jsonify({"data": data, "meta": {"hours": hours}})


@bp.get("/analytics/reviews")
@capability_required("view_analytics")
def analytics_reviews():
    """FR lxx: review outcomes and decisions per reviewer."""
    hours, since = _window()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        outcomes = session.execute(
            select(Review.decision, func.count(Review.id))
            .where(Review.status == "Reviewed", Review.decided_at >= since)
            .group_by(Review.decision)
        ).all()
        per_reviewer = session.execute(
            select(Review.decided_by_id, Review.decision, func.count(Review.id))
            .where(Review.status == "Reviewed", Review.decided_at >= since)
            .group_by(Review.decided_by_id, Review.decision)
        ).all()
        queue_depth = session.execute(
            select(func.count(Review.id)).where(Review.status == "Pending Review")
        ).scalar() or 0
    decided = sum(count for _, count in outcomes)
    overrides = sum(count for name, count in outcomes if name == "override")
    reviewers: dict[int, dict[str, int]] = {}
    for reviewer_id, decision, count in per_reviewer:
        if reviewer_id is None:
            continue
        reviewers.setdefault(reviewer_id, {})[decision or "pending"] = count
    data = {
        "outcomes": {name or "pending": count for name, count in outcomes},
        "decided": decided,
        "override_count": overrides,
        "override_rate": round(overrides / decided, 4) if decided else None,
        "queue_depth": queue_depth,
        "per_reviewer": [
            {"reviewer_id": reviewer_id, "decisions": decisions}
            for reviewer_id, decisions in sorted(reviewers.items())
        ],
    }
    return jsonify({"data": data, "meta": {"hours": hours}})