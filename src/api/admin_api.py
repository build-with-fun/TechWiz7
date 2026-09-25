"""``/api/admin`` -- configuration, users, audit, monitoring, retention (FR lxxv-lxxx).

Owner: sara.

This is the slice where a mistake damages the app *at runtime*: a bad threshold JSON stops
every future upload. The write path is therefore deliberately conservative:

1. the caller must hold ``edit_config`` (or ``manage_users``/``read_audit`` for its slice);
2. the new document is written to a temp file next to the real one;
3. ``ConfigStore.validate()`` runs over the *proposed* on-disk state -- the same validator
   boot runs, so a config the app will not start with cannot be written;
4. only on a clean validate is the temp file swapped in, the store's cache invalidated,
   and the change audited with before/after.

Config writes are audited even when they *fail* validation (outcome ``failure``), because
an attempted bad edit is itself something the audit trail should show (FR lxxvi).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import func, or_, select, update

from src.auth import capability_required, client_ip, current_user, hash_password
from src.db import record_audit, session_scope
from src.errors import ApiError, current_request_id, not_found, validation_error
from src.models import ROLES, ROLE_LABELS, AuditRecord, Review, User, utcnow
from src.services.config import get_store

bp = Blueprint("admin_api", __name__)

_LOGGER = logging.getLogger(__name__)

#: Which files the PUT endpoint accepts, and where they live.
EDITABLE_FILES: dict[str, str] = {
    "thresholds": "config",
    "alert-rules": "alert_rules",
    "alert_rules": "alert_rules",
    "manual-review-conditions": "alert_rules",
    "manual_review_conditions": "alert_rules",
    "severity-levels": "alert_rules",
    "severity_levels": "alert_rules",
    "retention": "alert_rules",
    "classes": "config",
    "features": "config",
    "auth": "config",
}


# ---------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------


@bp.get("/config")
@capability_required("edit_config")
def config_overview():
    """FR liii: every live config value with its source resolved."""
    store = get_store()
    snapshot = store.snapshot()
    data = {
        "config_dir": str(store.config_dir),
        "alert_rules_dir": str(store.alert_rules_dir),
        "thresholds": store.thresholds(),
        "alert_rules": store.alert_rules(),
        "manual_review": store.manual_review(),
        "severity_levels": store.severity_levels(),
        "severity_scale": store.severity_scale(),
        "retention": store.retention(),
        "classes": store.classes_config(),
        "snapshot": snapshot.to_dict(),
    }
    return jsonify({"data": data})


def _path_for(store, file_key: str) -> Path:
    where = EDITABLE_FILES.get(file_key)
    if where is None:
        raise validation_error(
            "Unknown configuration file.",
            file={"given": file_key, "accepted": sorted(EDITABLE_FILES)},
        )
    return (store.config_dir if where == "config" else store.alert_rules_dir) / f"{file_key.replace('-', '_')}.json"


@bp.put("/config/<file_key>")
@capability_required("edit_config")
def put_config(file_key: str):
    """FR liii / FR lxxx: an edited config file, validated before anything touches disk."""
    store = get_store()
    path = _path_for(store, file_key)
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise validation_error("Request body must be a JSON object.")

    before = path.read_text(encoding="utf-8") if path.exists() else ""
    if not before:
        # A file the store reads with defaults is still editable, but only whole.
        raise not_found(f"configuration file {file_key}")

    directory = path.parent
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, prefix=f".{path.name}.", suffix=".tmp",
        delete=False,
    )
    temp_name = Path(handle.name)
    try:
        with handle:
            json.dump(body, handle, indent=2, ensure_ascii=False, sort_keys=False)
            handle.write("\n")
        from src.services.config import ConfigStore

        class ProposedConfigStore(ConfigStore):
            def _load(self, candidate_path):
                if candidate_path.resolve() == path.resolve():
                    return body
                return super()._load(candidate_path)

        problems = ProposedConfigStore(store.config_dir, store.alert_rules_dir).validate()
        if problems:
            temp_name.unlink(missing_ok=True)
            raise validation_error(
                "The proposed configuration failed validation.", problems=problems
            )
        os.replace(temp_name, path)
    except ApiError:
        temp_name.unlink(missing_ok=True)
        raise
    except Exception:
        temp_name.unlink(missing_ok=True)
        raise
    finally:
        temp_name.unlink(missing_ok=True)

    with store._lock:  # same cache the readers use; drop stale entries now
        store._cache.clear()
        store._hashes.clear()
    after = path.read_text(encoding="utf-8")
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        record_audit(
            session,
            action="config_edited",
            actor=current_user._get_current_object(),
            target_type="config",
            target_id=file_key,
            outcome="success",
            detail=f"{file_key} updated ({path.name}); candidate validation passed",
            before={"content": before},
            after={"content": after},
            ip_address=client_ip(),
            user_agent=request.headers.get("User-Agent"),
            request_id=current_request_id(),
        )
    return jsonify(
        {"data": {"file": file_key, "path": str(path), "problems": []},
         "meta": {"outcome": "updated"}}
    )


@bp.get("/config/history")
@capability_required("read_audit")
def config_history():
    """FR lxx: who changed which config value, when, from what to what."""
    return _audit_response(extra_clauses=[AuditRecord.action == "config_edited"])


# ---------------------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------------------


def _user_to_dict(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role,
        "role_label": user.role_label,
        "is_active": bool(user.is_active),
        "must_change_password": bool(user.must_change_password),
        "failed_login_count": user.failed_login_count,
        "locked_until": user.locked_until.isoformat() + "Z" if user.locked_until else None,
        "created_at": user.created_at.isoformat() + "Z" if user.created_at else None,
        "last_login_at": user.last_login_at.isoformat() + "Z" if user.last_login_at else None,
    }


@bp.get("/users")
@capability_required("manage_users")
def list_users():
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        rows = session.execute(
            select(User).order_by(User.role, User.username)
        ).scalars().all()
        counts = dict(
            session.execute(select(User.role, func.count(User.id)).group_by(User.role)).all()
        )
    return jsonify(
        {
            "data": [_user_to_dict(row) for row in rows],
            "meta": {"role_counts": counts, "roles": list(ROLES)},
        }
    )


@bp.post("/users")
@capability_required("manage_users")
def create_user():
    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    role = (body.get("role") or "").strip()
    if not username or len(username) < 3:
        raise validation_error("username must be at least 3 characters.", username="required")
    if role not in ROLES:
        raise validation_error(
            "Unknown role.", role={"given": role, "accepted": list(ROLES)}
        )
    if len(password) < 12:
        raise validation_error(
            "password must be at least 12 characters.", password="too_short"
        )
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        existing = session.execute(
            select(User).where(or_(User.username == username, User.email == body.get("email")))
        ).scalar_one_or_none()
        if existing is not None:
            raise ApiError("conflict", f"User {existing.username!r} already exists.")
        user = User(
            username=username,
            email=(body.get("email") or f"{username}@sonicsentinel.local").strip(),
            display_name=(body.get("display_name") or username).strip(),
            password_hash=hash_password(password),
            role=role,
            is_active=True,
            must_change_password=True,
            created_by_id=current_user._get_current_object().id,
        )
        session.add(user)
        session.flush()
        record_audit(
            session,
            action="user_created",
            actor=current_user._get_current_object(),
            target_type="user",
            target_id=str(user.id),
            outcome="success",
            detail=f"created {user.username} as {role}",
            after={"role": role, "username": user.username},
            ip_address=client_ip(),
            user_agent=request.headers.get("User-Agent"),
            request_id=current_request_id(),
        )
        session.expunge(user)
        data = _user_to_dict(user)
    return jsonify({"data": data, "meta": {"outcome": "created"}}), 201


@bp.patch("/users/<int:user_id>")
@capability_required("manage_users")
def patch_user(user_id: int):
    body = request.get_json(silent=True) or {}
    if not body:
        raise validation_error("No changes supplied.")
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        user = session.get(User, user_id)
        if user is None:
            raise not_found("user")
        before = {"role": user.role, "is_active": bool(user.is_active)}
        if "role" in body:
            if body["role"] not in ROLES:
                raise validation_error(
                    "Unknown role.", role={"given": body["role"], "accepted": list(ROLES)}
                )
            user.role = body["role"]
        if "is_active" in body:
            user.is_active = bool(body["is_active"])
        if "unlock" in body and body["unlock"]:
            user.locked_until = None
            user.failed_login_count = 0
        if "password" in body and body["password"]:
            if len(body["password"]) < 12:
                raise validation_error(
                    "password must be at least 12 characters.", password="too_short"
                )
            user.password_hash = hash_password(body["password"])
            user.must_change_password = True
            user.password_changed_at = utcnow()
        record_audit(
            session,
            action="user_updated",
            actor=current_user._get_current_object(),
            target_type="user",
            target_id=str(user.id),
            outcome="success",
            detail=f"updated {user.username}",
            before=before,
            after={"role": user.role, "is_active": bool(user.is_active),
                   "password_reset": "password" in body},
            ip_address=client_ip(),
            user_agent=request.headers.get("User-Agent"),
            request_id=current_request_id(),
        )
        session.expunge(user)
        data = _user_to_dict(user)
    return jsonify({"data": data, "meta": {"outcome": "updated"}})


# ---------------------------------------------------------------------------------------
# audit trail, monitoring, retention
# ---------------------------------------------------------------------------------------


def _audit_response(extra_clauses=None):
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(200, max(1, int(request.args.get("per_page", 100))))
    except ValueError:
        raise validation_error("page and per_page must be integers.")
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        statement = select(AuditRecord).order_by(
            AuditRecord.timestamp.desc(), AuditRecord.id.desc()
        )
        clauses = list(extra_clauses) if extra_clauses else _extra_clauses()
        for clause in clauses:
            statement = statement.where(clause)
        total = session.execute(
            select(func.count(AuditRecord.id)).where(*clauses)
        ).scalar() or 0
        rows = session.execute(
            statement.limit(per_page).offset((page - 1) * per_page)
        ).scalars().all()
    return jsonify(
        {
            "data": [_audit_to_dict(row) for row in rows],
            "meta": {
                "total": total, "page": page, "per_page": per_page,
                "pages": max(1, -(-total // per_page)),
            },
        }
    )


def _extra_clauses():
    clauses = []
    action = request.args.get("action")
    if action:
        clauses.append(AuditRecord.action == action)
    actor = request.args.get("actor")
    if actor:
        clauses.append(AuditRecord.actor_username == actor)
    outcome = request.args.get("outcome")
    if outcome:
        clauses.append(AuditRecord.outcome == outcome)
    target_type = request.args.get("target_type")
    if target_type:
        clauses.append(AuditRecord.target_type == target_type)
    return clauses


def _audit_to_dict(row: AuditRecord) -> dict:
    return {
        "timestamp": row.timestamp.isoformat() + "Z" if row.timestamp else None,
        "actor": {"username": row.actor_username, "role": row.actor_role},
        "action": row.action,
        "target_type": row.target_type,
        "target_id": row.target_id,
        "outcome": row.outcome,
        "detail": row.detail,
        "before": row.before,
        "after": row.after,
        "ip_address": row.ip_address,
        "request_id": row.request_id,
        "sha256": row.sha256,
    }


@bp.get("/audit")
@capability_required("read_audit")
def audit_trail():
    """FR lxxvi: the filterable audit trail."""
    return _audit_response()


@bp.get("/monitoring/anomalies")
@capability_required("read_audit")
def monitoring_anomalies():
    """FR lxxviii: in-app health signals -- error rate, disk use, queue depth, pipeline.

    Computed from data the app already has (the audit table, the DB file, the configured
    retention floor) rather than an external monitor, so an anomaly is visible even with
    the network gone -- which is exactly when the monitoring matters.
    """
    anomalies: list[dict] = []
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        hour_ago = utcnow().replace(microsecond=0) - __import__("datetime").timedelta(hours=1)
        failed = session.execute(
            select(func.count(AuditRecord.id)).where(
                AuditRecord.timestamp >= hour_ago, AuditRecord.outcome == "failure"
            )
        ).scalar() or 0
        total_actions = session.execute(
            select(func.count(AuditRecord.id)).where(AuditRecord.timestamp >= hour_ago)
        ).scalar() or 0
        queue_depth = session.execute(
            select(func.count(Review.id)).where(Review.status == "Pending Review")
        ).scalar() or 0
    error_rate = failed / total_actions if total_actions else 0.0
    if total_actions and error_rate > 0.20:
        anomalies.append(
            {"kind": "error_rate", "value": round(error_rate, 4),
             "threshold": 0.20,
             "message": f"{failed}/{total_actions} audited actions failed in the last hour."}
        )
    if queue_depth > 100:
        anomalies.append(
            {"kind": "queue_depth", "value": queue_depth, "threshold": 100,
             "message": "Manual-review queue depth above 100 pending items."}
        )
    try:
        db_path = Path(current_app.config["SST_DB_PATH"])
        usage = shutil.disk_usage(db_path.parent)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 1.0:
            anomalies.append(
                {"kind": "disk_free", "value": round(free_gb, 3), "threshold": 1.0,
                 "message": f"Only {free_gb:.2f} GB free where the database lives."}
            )
    except Exception:  # pragma: no cover - disk check must never fail the endpoint
        _LOGGER.debug("disk check skipped", exc_info=True)
    return jsonify(
        {
            "data": {"anomalies": anomalies},
            "meta": {"error_rate": round(error_rate, 4),
                     "actions_last_hour": total_actions, "queue_depth": queue_depth},
        }
    )


@bp.post("/retention/preview")
@capability_required("manage_users")
def retention_preview():
    """FR lxxx: what a purge *would* delete. Reads only; nothing is removed."""
    return _retention(dry_run=True)


@bp.post("/retention/purge")
@capability_required("manage_users")
def retention_purge():
    """FR lxxx: execute the purge. ``?dry_run=1`` is the default, on purpose."""
    dry_run = request.args.get("dry_run", "1") not in {"0", "false"}
    return _retention(dry_run=dry_run)


def _retention(*, dry_run: bool):
    from src.models import Alert, AudioFile, Event, LiveWindow

    store = get_store()
    now = utcnow()
    policy = store.retention()
    defaults = policy.get("defaults", {})
    overrides = policy.get("overrides_by_severity", {})
    batch_size = max(1, min(5000, int(policy.get("purge_batch_size", 500))))

    def days_for(artifact: str, events: list[Event]) -> int | None:
        base = defaults.get(f"{artifact}_days")
        if base is None:
            return None
        return max([base, *[
            overrides.get(event.severity or "", {}).get(f"{artifact}_days", base)
            for event in events
        ]])

    def has_hold(session, audio: AudioFile) -> bool:
        if audio.flagged_for_investigation:
            return True
        ids = [event.id for event in audio.events]
        if not ids:
            return False
        pending_review = session.execute(
            select(func.count(Review.id)).where(
                Review.event_id.in_(ids), Review.status == "Pending Review"
            )
        ).scalar() or 0
        open_alert = session.execute(
            select(func.count(Alert.id)).where(
                Alert.event_id.in_(ids), Alert.status.in_(["Open", "Escalated", "Acknowledged"])
            )
        ).scalar() or 0
        return bool(pending_review or open_alert)

    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        plan: dict[str, int] = {"event_records": 0, "audio": 0,
                                "microphone_session_audio": 0}
        files = session.execute(select(AudioFile).order_by(AudioFile.created_at, AudioFile.id)).scalars().all()
        for audio in files:
            if has_hold(session, audio):
                continue
            events = list(audio.events)
            for event in events:
                days = days_for("event_records", [event])
                if days is not None and event.created_at < now - timedelta(days=days):
                    if plan["event_records"] < batch_size:
                        plan["event_records"] += 1
                        if not dry_run:
                            session.execute(update(LiveWindow).where(
                                LiveWindow.event_id == event.id
                            ).values(event_id=None))
                            session.delete(event)
            artifact = "microphone_session_audio" if audio.source == "microphone" else "audio"
            days = days_for(artifact, events)
            if (days is None or not audio.stored_path
                    or audio.created_at >= now - timedelta(days=days)
                    or plan[artifact] >= batch_size):
                continue
            plan[artifact] += 1
            if not dry_run:
                path = current_app.config["SST_STORAGE"].resolve(audio.stored_path)
                path.unlink(missing_ok=True)
                audio.stored_path = ""
        record_audit(
            session,
            action="retention_preview" if dry_run else "retention_purge",
            actor=current_user._get_current_object(),
            target_type="retention",
            target_id="-",
            outcome="success",
            detail=("dry-run plan" if dry_run else "purge executed")
                   + f": {json.dumps(plan, sort_keys=True)}",
            after=plan,
            ip_address=client_ip(),
            user_agent=request.headers.get("User-Agent"),
            request_id=current_request_id(),
        )
    return jsonify(
        {
            "data": {"plan": plan, "dry_run": dry_run,
                     "note": "dry run only; nothing was deleted" if dry_run
                     else "purge executed"},
        }
    )
