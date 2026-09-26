"""Operational anomaly checks for administrators (SRS FR lxxviii).

The SRS lists six things an administrator should be alerted about. Each check below
reads rows the app already writes (audit records, events, alerts), so it needs no
extra monitoring service and still works with the network down.

    repeated failed uploads        audit: audio_upload with outcome=failure
    model failures                 audit: model_failure
    unusual low-confidence spike   events: share below the confidence floor, last hour
                                   versus the previous 7 days
    excessive critical alerts      alerts: severity Critical, last hour
    duplicate files                audit: audio_duplicate_rejected / _allowed
    failed login attempts          audit: login_failure, plus accounts currently locked

Thresholds live in config/thresholds.json under "monitoring" so an administrator can
tune them without a code change. The defaults are deliberately low for a demo: with a
handful of users, ten failed logins in an hour is already unusual.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

from sqlalchemy import func, select

DEFAULTS: dict[str, float] = {
    "window_minutes": 60,
    "failed_uploads": 5,
    "model_failures": 1,
    "low_confidence_share": 0.5,
    "low_confidence_min_events": 10,
    "low_confidence_vs_baseline": 2.0,
    "critical_alerts": 10,
    "duplicate_files": 5,
    "failed_logins": 10,
    "error_rate": 0.20,
    "review_queue_depth": 100,
}


def settings(thresholds: Mapping[str, Any] | None) -> dict[str, float]:
    configured = dict((thresholds or {}).get("monitoring") or {})
    return {k: float(configured.get(k, v)) for k, v in DEFAULTS.items()}


def compute_anomalies(session, thresholds: Mapping[str, Any] | None, *,
                      now: dt.datetime | None = None) -> dict[str, Any]:
    from src.models import Alert, AuditRecord, Event, Review, User

    cfg = settings(thresholds)
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    since = now - dt.timedelta(minutes=cfg["window_minutes"])
    week_ago = now - dt.timedelta(days=7)
    floor = float(((thresholds or {}).get("confidence") or {}).get("min_confidence", 0.6))

    def audit_count(*actions: str, outcome: str | None = None) -> int:
        query = select(func.count(AuditRecord.id)).where(
            AuditRecord.timestamp >= since, AuditRecord.action.in_(actions))
        if outcome:
            query = query.where(AuditRecord.outcome == outcome)
        return int(session.execute(query).scalar() or 0)

    def low_share(start: dt.datetime, end: dt.datetime) -> tuple[int, float]:
        total = session.execute(select(func.count(Event.id)).where(
            Event.created_at >= start, Event.created_at < end,
            Event.top_confidence.is_not(None))).scalar() or 0
        low = session.execute(select(func.count(Event.id)).where(
            Event.created_at >= start, Event.created_at < end,
            Event.top_confidence < floor)).scalar() or 0
        return int(total), (low / total if total else 0.0)

    observed = {
        "failed_uploads": audit_count("audio_upload", outcome="failure"),
        "model_failures": audit_count("model_failure"),
        "duplicate_files": audit_count("audio_duplicate_rejected", "audio_duplicate_allowed"),
        "failed_logins": audit_count("login_failure", "login_blocked"),
        "critical_alerts": int(session.execute(select(func.count(Alert.id)).where(
            Alert.created_at >= since, Alert.severity == "Critical")).scalar() or 0),
        "locked_accounts": int(session.execute(select(func.count(User.id)).where(
            User.locked_until.is_not(None), User.locked_until > now)).scalar() or 0),
        "review_queue_depth": int(session.execute(select(func.count(Review.id)).where(
            Review.status == "Pending Review")).scalar() or 0),
    }
    recent_n, recent_share = low_share(since, now + dt.timedelta(seconds=1))
    _, baseline_share = low_share(week_ago, since)
    observed.update({"events_in_window": recent_n, "low_confidence_share": round(recent_share, 4),
                     "low_confidence_baseline": round(baseline_share, 4)})
    actions = int(session.execute(select(func.count(AuditRecord.id)).where(
        AuditRecord.timestamp >= since)).scalar() or 0)
    failures = int(session.execute(select(func.count(AuditRecord.id)).where(
        AuditRecord.timestamp >= since, AuditRecord.outcome == "failure")).scalar() or 0)
    observed["error_rate"] = round(failures / actions, 4) if actions else 0.0

    minutes = int(cfg["window_minutes"])
    found: list[dict[str, Any]] = []

    def flag(kind: str, severity: str, value: Any, threshold: Any, message: str) -> None:
        found.append({"kind": kind, "severity": severity, "value": value,
                      "threshold": threshold, "message": message})

    if observed["failed_uploads"] >= cfg["failed_uploads"]:
        flag("failed_uploads", "Medium", observed["failed_uploads"], cfg["failed_uploads"],
             f"{observed['failed_uploads']} uploads failed in the last {minutes} min.")
    if observed["model_failures"] >= cfg["model_failures"]:
        flag("model_failures", "High", observed["model_failures"], cfg["model_failures"],
             f"A model failed to analyse audio {observed['model_failures']} time(s) "
             f"in the last {minutes} min. Check /api/health/ready.")
    # A spike needs enough events to mean anything, a high share, and a clear jump from
    # the usual level. The last condition stops a site that is always noisy from
    # alerting every hour.
    if (recent_n >= cfg["low_confidence_min_events"]
            and recent_share >= cfg["low_confidence_share"]
            and recent_share >= cfg["low_confidence_vs_baseline"] * max(baseline_share, 0.05)):
        flag("low_confidence_spike", "Medium", round(recent_share, 3),
             cfg["low_confidence_share"],
             f"{recent_share:.0%} of the last {recent_n} events were below the "
             f"{floor:.2f} confidence floor (usual level {baseline_share:.0%}). "
             "A microphone may be failing or the site may be unusually noisy.")
    if observed["critical_alerts"] >= cfg["critical_alerts"]:
        flag("excessive_critical_alerts", "High", observed["critical_alerts"], cfg["critical_alerts"],
             f"{observed['critical_alerts']} critical alerts in the last {minutes} min. "
             "Check for a real incident first, then for a misbehaving rule or device.")
    if observed["duplicate_files"] >= cfg["duplicate_files"]:
        flag("duplicate_files", "Low", observed["duplicate_files"], cfg["duplicate_files"],
             f"{observed['duplicate_files']} duplicate uploads in the last {minutes} min.")
    if observed["failed_logins"] >= cfg["failed_logins"] or observed["locked_accounts"]:
        flag("failed_logins", "High", observed["failed_logins"], cfg["failed_logins"],
             f"{observed['failed_logins']} failed sign-ins in the last {minutes} min; "
             f"{observed['locked_accounts']} account(s) currently locked.")
    if actions and observed["error_rate"] > cfg["error_rate"]:
        flag("error_rate", "Medium", observed["error_rate"], cfg["error_rate"],
             f"{failures}/{actions} audited actions failed in the last {minutes} min.")
    if observed["review_queue_depth"] > cfg["review_queue_depth"]:
        flag("review_queue_depth", "Medium", observed["review_queue_depth"],
             cfg["review_queue_depth"],
             f"{observed['review_queue_depth']} events are waiting for manual review.")
    return {"anomalies": found, "observed": observed, "settings": cfg,
            "window_start": since.isoformat(), "computed_at": now.isoformat()}
