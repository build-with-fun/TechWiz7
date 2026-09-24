"""Health, readiness and monitoring.  SRS FR lxxviii, and the >= 99% uptime NFR.

Owner: sara.

Two endpoints, serving two different consumers:

* ``/api/health`` -- for the load balancer and for monitoring. Fast, no database, no model
  inference. It must answer while the database is down, because that 503 *is* the signal.
* ``/api/health/detail`` -- for an administrator, and for the demo. It reports the state of
  the pipeline, the database, the storage tree, the configuration, the model versions and
  the anomaly counters. It is deliberately readable: an evaluator asking "how do you know
  it is up?" gets a page, not a shrug.
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
    """The SRS §1.10 deliverable folders, and whether each is actually there.

    Reported rather than assumed: "we built it" should be checkable by the person marking
    it, on the running instance.
    """
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
            # A count, not a listing: enough to show it is populated, cheap enough for a
            # health check, and it cannot leak a filename.
            try:
                entry["entries"] = sum(1 for _ in os.scandir(path))
            except OSError:
                entry["entries"] = None
        report[name] = entry
    return report


@bp.get("/api/health")
def health():
    """Liveness plus a one-word answer about whether analysis can run.

    Deliberately does **not** touch the database: a health check that fails because the
    database is briefly busy would restart a working application.
    """
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
            "models": "ready" if status.get("ready") else status.get("reason", "unavailable"),
        },
    }
    # 200 while the console is usable. A missing model is reported in the body and by the
    # detail endpoint rather than by a non-200 here, because the sign-in page, the
    # dashboards, the review queue and the audit trail all still work -- and a load
    # balancer taking the instance out of rotation for that would be wrong.
    return jsonify(payload)


@bp.get("/api/health/ready")
def ready():
    """Readiness: is everything this instance needs actually available?

    503 when the database cannot be reached, because that is the one dependency without
    which no request is meaningful.
    """
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
    return jsonify({"status": "ready"})


@bp.get("/api/health/detail")
@capability_required("view_dashboards")
def health_detail():
    """Everything an operator needs to see, in one place. FR lxxviii."""
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
                        # ``metrics`` is one JSON column, not fixed columns, so an evaluator's
                        # model_meta.json can carry whatever it measured without a migration.
                        # Read defensively: a version registered without metrics is normal.
                        "test_accuracy": (v.metrics or {}).get("accuracy"),
                        "macro_f1": (v.metrics or {}).get("macro_f1"),
                        "trained_at": v.trained_at.isoformat() if v.trained_at else None,
                    }
                    for v in versions
                ],
                "path": current_app.config["SST_DB_PATH"],
            }
    except Exception as exc:
        db = {"ok": False, "error": type(exc).__name__}

    pipeline = pipeline_status()
    storage = current_app.config["SST_STORAGE"]

    # FR lxxviii's anomaly list, reported as live counters.
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
    """Which build is running. Unauthenticated on purpose -- it exposes nothing but a name."""
    from src.app import APP_NAME, APP_VERSION

    return jsonify({
        "app": APP_NAME,
        "version": APP_VERSION,
        "started_at": current_app.config.get("SST_STARTED_AT"),
    })
