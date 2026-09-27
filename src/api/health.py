"""Health endpoints.

/api/health: liveness, no database or inference. /api/health/ready: 200 only when the
database and both models work. /api/health/detail (signed in): full status.
"""

from __future__ import annotations

import datetime as dt
import os
import time

from flask import Blueprint, current_app, jsonify
from sqlalchemy import func, select, text

from src.auth import capability_required
from src.db import REPO_ROOT, session_scope
from src.models import Alert, Event, ModelVersion, User, utcnow

bp = Blueprint("health", __name__)

_STARTED_MONOTONIC = time.monotonic()


def _folder_report() -> dict[str, dict]:
    """Which of the SRS 1.10 folders exist."""
    names = (
        "src", "templates", "static", "config", "alert_rules", "database", "data",
        "tests", "audio_dataset", "sample_audio", "documentation", "python_models",
        "gtm_model", "feature_extraction", "audio_preprocessing", "augmentation",
        "notebooks", "reports", "screenshots",
    )
    report: dict[str, dict] = {}
    for name in names:
        path = REPO_ROOT / name
        entry: dict = {"present": path.exists()}
        if path.is_dir():
            # Count only, no filenames.
            try:
                entry["entries"] = sum(1 for _ in os.scandir(path))
            except OSError:
                entry["entries"] = None
        report[name] = entry
    return report


@bp.get("/api/health")
def health():
    """Liveness and whether analysis can run. Doesn't touch the database."""
    from src.services.pipeline import pipeline_status

    status = pipeline_status()
    payload = {
        "status": "ok",
        "app": "SonicSentinel AI",
        "time": utcnow().isoformat(),
        "uptime_seconds": round(time.monotonic() - _STARTED_MONOTONIC, 1),
        "analysis_ready": bool(status.get("ready")),
        "checks": {
            "database": "not checked by this endpoint",
            "models": "ready" if status.get("ready") else "unavailable",
        },
    }
    # Still 200 without a model: sign-in, dashboards and review work.
    return jsonify(payload)


@bp.get("/api/health/ready")
def ready():
    """Return 503 when the database or dual-model analysis is unavailable."""
    try:
        factory = current_app.config["SST_SESSION_FACTORY"]
        with session_scope(factory) as session:
            session.execute(text("SELECT 1")).scalar()
    except Exception as exc:
        return jsonify({
            "status": "unavailable",
            "reason": "database unreachable",
            "detail": type(exc).__name__,
        }), 503
    if current_app.config.get("SST_PIPELINE") is None:
        return jsonify({"status": "unavailable", "reason": "analysis models unavailable"}), 503
    return jsonify({"status": "ready", "analysis_ready": True})


@bp.get("/api/health/detail")
@capability_required("view_dashboards")
def health_detail():
    """Full status for operators (FR lxxviii)."""
    from src.services.pipeline import pipeline_status
    from src.services.config import get_store

    store = get_store()
    factory = current_app.config["SST_SESSION_FACTORY"]
    now = utcnow()
    day_ago = now - dt.timedelta(hours=24)

    db: dict = {"ok": False}
    try:
        with session_scope(factory) as session:
            counts = {
                "users": session.execute(select(func.count(User.id))).scalar(),
                "events": session.execute(select(func.count(Event.id))).scalar(),
                "open_alerts": session.execute(
                    select(func.count(Alert.id)).where(Alert.status == "Open")
                ).scalar(),
                "events_24h": session.execute(
                    select(func.count(Event.id)).where(Event.created_at >= day_ago)
                ).scalar(),
            }
            versions = session.execute(
                select(ModelVersion).order_by(ModelVersion.model_name, ModelVersion.id)
            ).scalars().all()
            db = {
                "ok": True,
                "counts": counts,
                "model_versions": [
                    {
                        "model": v.model_name,
                        "version": v.version,
                        "active": bool(v.is_active),
                        "labels": v.label,
                        "algorithm": v.algorithm,
                        # metrics may be empty.
                        "test_accuracy": (v.metrics or {}).get("accuracy"),
                        "macro_f1": (v.metrics or {}).get("macro_f1"),
                        "trained_at": v.trained_at.isoformat() if v.trained_at else None,
                    }
                    for v in versions
                ],
            }
    except Exception as exc:
        db = {"ok": False, "error": type(exc).__name__}

    pipeline = pipeline_status()
    storage = current_app.config["SST_STORAGE"]

    # FR lxxviii anomaly counters.
    anomalies: list[dict] = []
    if not pipeline.get("ready"):
        anomalies.append({
            "kind": "model_unavailable",
            "severity": "high",
            "message": pipeline.get("reason", "the analysis pipeline is not initialised"),
        })
    if not db.get("ok"):
        anomalies.append({
            "kind": "database_unreachable",
            "severity": "critical",
            "message": "the database could not be read",
        })
    if os.environ.get("SST_SECRET_KEY") is None and not current_app.config.get("TESTING"):
        anomalies.append({
            "kind": "insecure_configuration",
            "severity": "medium",
            "message": "SST_SECRET_KEY is not set, so a development session key is in use",
        })

    return jsonify({
        "status": "ok" if not any(a["severity"] == "critical" for a in anomalies) else "degraded",
        "instance": {
            "id": current_app.config.get("SST_INSTANCE_ID"),
            "started_at": current_app.config.get("SST_STARTED_AT"),
            "uptime_seconds": round(time.monotonic() - _STARTED_MONOTONIC, 1),
            "version": current_app.config.get("SST_APP_VERSION", "1.0.0"),
        },
        "database": db,
        "analysis": {
            "ready": pipeline.get("ready", False),
            "reason": pipeline.get("reason"),
            "models": pipeline.get("models"),
            "warmed": pipeline.get("warmed"),
        },
        "storage": {
            "root": str(storage.root),
            "audio_dir": str(storage.audio_dir),
            "writable": os.access(storage.root, os.W_OK) if storage.root.exists() else False,
        },
        "configuration": {
            "config_dir": str(store.config_dir),
            "alert_rules_dir": str(store.alert_rules_dir),
            "class_count": len(store.class_names()),
            "critical_classes": sorted(store.critical_classes()),
            "severity_scale": list(store.severity_scale()),
            "review_conditions": list(store.review_condition_ids()),
            "hashes": store.snapshot().to_dict().get("hashes"),
        },
        "deliverables": _folder_report(),
        "anomalies": anomalies,
    })


@bp.get("/api/version")
def version():
    """App name and version. No login needed."""
    from src.app import APP_NAME, APP_VERSION

    return jsonify({
        "app": APP_NAME,
        "version": APP_VERSION,
        "started_at": current_app.config.get("SST_STARTED_AT"),
    })
