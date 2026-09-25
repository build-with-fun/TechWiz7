"""``/api/live`` -- the rolling live-session slice (FR xxxvi, FR lxxix).

Owner: sara (contract §3.4, §6.1: rolling POST, not a WebSocket).

One session per console page-load, addressed by its uuid key:

* ``POST /sessions`` opens it -- **the consent gate is not bypassable**: without
  ``consent_acknowledged`` the session is refused (FR lxxix), and the consent event is
  audited because it is a privacy action, not a UI checkbox.
* ``POST /sessions/<sid>/windows`` carries one 1-3 s window as base64 audio, runs it
  through the same pipeline as an upload (independent models, comparison, severity), keeps
  the repeated-detection state on the session row so the streak survives reloads and
  restarts, and persists a ``LiveWindow`` row per window -- the live panel's data is a
  resource, not scrollback.
* ``POST /sessions/<sid>/stop`` closes and rolls up; ``GET /sessions/<sid>`` reads the
  state. A session whose windows stop arriving is expired lazily.

The endpoints that write are owner-only (a session is its opener's data); reads too --
another user's session is a 404, per the contract's scoping rule.
"""

from __future__ import annotations

import base64
import binascii
import datetime as _dt
import logging
import uuid

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import select

from src.auth import capability_required, client_ip, current_user
from src.db import record_audit, session_scope
from src.errors import ApiError, current_request_id, not_found, validation_error
from src.models import LiveSession, LiveWindow, utcnow
from src.services.config import get_store
from src.services.pipeline import get_pipeline

bp = Blueprint("live_api", __name__)

_LOGGER = logging.getLogger(__name__)

#: How long a session may sit without a window before reads start reporting it expired.
#: Shorter than the retention window on purpose: a silent microphone should stop counting
#: as "live" long before its audio ages out.
_IDLE_TIMEOUT = _dt.timedelta(minutes=10)


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------


def _session_to_dict(session_row: LiveSession) -> dict:
    return {
        "id": session_row.id,
        "user_id": session_row.user_id,
        "device_label": session_row.device_label,
        "location": session_row.location,
        "status": session_row.status,
        "started_at": (
            session_row.started_at.isoformat() + "Z" if session_row.started_at else None
        ),
        "ended_at": session_row.ended_at.isoformat() + "Z" if session_row.ended_at else None,
        "consent_acknowledged": bool(session_row.consent_acknowledged),
        "consent_recorded_at": (
            session_row.consent_recorded_at.isoformat() + "Z"
            if session_row.consent_recorded_at else None
        ),
        "window_count": session_row.window_count,
        "detection_count": session_row.detection_count,
        "alert_count": session_row.alert_count,
        "consecutive_state": session_row.consecutive_state or {},
    }


def _load_owned_session(session, session_id: str) -> LiveSession:
    """Load the session and enforce ownership: another user's session is a 404."""
    row = session.get(LiveSession, session_id)
    if row is None:
        raise not_found("live session")
    user = current_user._get_current_object()
    if row.user_id != user.id and not user.can("view_all_events"):
        # Same rule as events: not "exists but is not yours" -- simply not found.
        raise not_found("live session")
    return row


def _audit(session, *, action: str, row: LiveSession, detail: str, after=None) -> None:
    record_audit(
        session,
        action=action,
        actor=current_user._get_current_object(),
        target_type="live_session",
        target_id=str(row.id),
        outcome="success",
        detail=detail,
        after=after,
        ip_address=client_ip(),
        user_agent=request.headers.get("User-Agent"),
        request_id=current_request_id(),
    )


# ---------------------------------------------------------------------------------------
# session lifecycle
# ---------------------------------------------------------------------------------------


@bp.get("/sessions")
@capability_required("live_session")
def list_sessions():
    """This caller's sessions, newest first -- the console's session picker."""
    user = current_user._get_current_object()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        statement = (
            select(LiveSession)
            .where(LiveSession.user_id == user.id)
            .order_by(LiveSession.started_at.desc())
            .limit(50)
        )
        rows = session.execute(statement).scalars().all()
    return jsonify({"data": [_session_to_dict(row) for row in rows]})


@bp.post("/sessions")
@capability_required("live_session")
def start_session():
    """FR lxxix: open a microphone session behind the consent gate."""
    body = request.get_json(silent=True) or {}
    consent = body.get("consent_ack") if "consent_ack" in body else body.get("consent_acknowledged")
    if consent not in (True, "true", "True", 1, "1"):
        raise validation_error(
            "Live sessions require an explicit consent acknowledgement (FR lxxix).",
            consent_ack="required",
        )
    user = current_user._get_current_object()
    now = utcnow()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        row = LiveSession(
            # The PK is a client-facing UUID4 (contract §3.4); it is assigned here,
            # never auto-generated, so a session id leaks nothing about volume.
            id=str(uuid.uuid4()),
            user_id=user.id,
            device_label=(body.get("device_label") or "browser").strip()[:80],
            location=(body.get("location") or "").strip()[:200] or None,
            status="active",
            started_at=now,
            consent_acknowledged=True,
            consent_recorded_at=now,
            window_count=0,
            detection_count=0,
            alert_count=0,
        )
        session.add(row)
        session.flush()
        _audit(
            session, action="live_session_started", row=row,
            detail=f"session {row.id} opened by {user.username} (consent recorded)",
            after={"consent": True, "location": row.location},
        )
        data = _session_to_dict(row)
    return jsonify({"data": data}), 201


@bp.get("/sessions/<session_id>")
@capability_required("live_session")
def get_session(session_id: str):
    """Session state plus its window roll -- what the panel shows on reload."""
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        row = _load_owned_session(session, session_id)
        if row.status == "active" and row.started_at is not None:
            stale_for = utcnow() - row.started_at
            windows = session.execute(
                select(LiveWindow).where(LiveWindow.session_id == row.id)
                .order_by(LiveWindow.seq.desc()).limit(1)
            ).scalar_one_or_none()
            if windows is not None and windows.captured_at is not None:
                stale_for = utcnow() - windows.captured_at
            if stale_for > _IDLE_TIMEOUT:
                row.status = "expired"
                row.ended_at = utcnow()
        windows = session.execute(
            select(LiveWindow).where(LiveWindow.session_id == row.id)
            .order_by(LiveWindow.seq.desc()).limit(50)
        ).scalars().all()
        window_rows = [
            {
                "seq": w.seq,
                "captured_at": w.captured_at.isoformat() + "Z" if w.captured_at else None,
                "duration_sec": w.duration_sec,
                "predicted_class": w.predicted_class,
                "python_confidence": w.python_confidence,
                "gtm_class": w.gtm_class,
                "gtm_confidence": w.gtm_confidence,
                "confidence_difference": w.confidence_difference,
                "consistency_status": w.consistency_status,
                "quality_verdict": w.quality_verdict,
                "severity": w.severity,
                "confirmed": bool(w.confirmed),
                "consecutive": w.consecutive,
                "needed": w.needed,
                "latency_ms": w.latency_ms,
                "event_id": w.event_id,
            }
            for w in reversed(windows)
        ]
        data = _session_to_dict(row)
        data["windows"] = window_rows
    return jsonify({"data": data})


@bp.post("/sessions/<session_id>/stop")
@capability_required("live_session")
def stop_session(session_id: str):
    """Close the session and roll up its counters; the rows stay readable (FR lxxvi)."""
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        row = _load_owned_session(session, session_id)
        if row.status != "active":
            raise ApiError(
                "invalid_state_transition",
                f"Session {session_id} is already {row.status}.",
            )
        row.status = "stopped"
        row.ended_at = utcnow()
        _audit(
            session, action="live_session_stopped", row=row,
            detail=(f"session {row.id} stopped after {row.window_count} windows, "
                    f"{row.detection_count} detections, {row.alert_count} alerts"),
            after={"window_count": row.window_count, "alert_count": row.alert_count},
        )
        data = _session_to_dict(row)
    return jsonify({"data": data})


# ---------------------------------------------------------------------------------------
# the rolling window
# ---------------------------------------------------------------------------------------


@bp.post("/sessions/<session_id>/windows")
@capability_required("live_session")
def push_window(session_id: str):
    """Contract §3.4: one 1-3 s window in, one verdict out -- inside the 3 s budget.

    The window runs through the *same* pipeline as an upload: independent models, the
    comparison, severity and the repeated-detection tracker. The tracker's state is stored
    on the session row after each window, so a streak survives a page reload or a server
    restart -- the live panel's "2 of 3 confirming windows" is then always true.
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise validation_error("Request body must be a JSON object.")
    try:
        seq = int(body.get("seq"))
    except (TypeError, ValueError):
        raise validation_error("seq must be an integer.", seq={"given": body.get("seq")})
    audio_b64 = body.get("audio_b64") or ""
    if not audio_b64:
        raise validation_error("audio_b64 is required.", audio_b64="required")
    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except (binascii.Error, ValueError):
        raise validation_error("audio_b64 is not valid base64.", audio_b64="invalid")
    if not audio_bytes:
        raise validation_error("audio_b64 decodes to no audio.", audio_b64="empty")

    captured_at = body.get("captured_at")
    duration_sec = body.get("duration_sec")
    location = body.get("location")

    pipeline = current_app.config.get("SST_PIPELINE")
    if pipeline is None:
        try:
            pipeline = get_pipeline()
        except Exception:
            pipeline = None
    if pipeline is None:
        raise ApiError(
            "pipeline_unavailable",
            "The analysis pipeline is not ready; live windows cannot be classified yet.",
        )

    from src.services.persistence import make_persistence_callback

    persist = make_persistence_callback(
        storage=current_app.config["SST_STORAGE"],
        actor=current_user._get_current_object(),
        request_id=current_request_id(),
    )
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        row = _load_owned_session(session, session_id)
        if row.status != "active":
            raise ApiError(
                "invalid_state_transition",
                f"Session {session_id} is {row.status}; open a new session to continue.",
            )
        # Duplicate protection: a retried window must not double-count a streak.
        existing = session.execute(
            select(LiveWindow).where(
                LiveWindow.session_id == row.id, LiveWindow.seq == seq
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise ApiError(
                "duplicate_window",
                f"Window {seq} was already processed for this session.",
            )

        consent = bool(row.consent_acknowledged)
        sample_rate = _validated_sample_rate(body)
        meta = {
            "session_id": row.id,
            "seq": seq,
            "device_label": row.device_label,
            "location": location or row.location,
            "consent_acknowledged": consent,
            "source": "microphone",
            "client_ip": client_ip(),
            "captured_at": captured_at,
            "duration_sec": duration_sec,
            "request_id": current_request_id(),
        }
        if audio_bytes[:4] == b"RIFF":
            # The contract's example ("UklGR…") is a WAV: hand the encoded container to
            # the same decode path an upload takes, so a live window and an uploaded clip
            # cannot disagree about decoding.
            record = pipeline.analyse_bytes(
                audio_bytes, filename=f"window_{seq}.wav", origin="live",
                persist=persist, **meta
            )
        else:
            # Raw little-endian float32 PCM from a client that streams without a header.
            record = pipeline.analyse_samples(audio_bytes, sample_rate, origin="live",
                                              persist=persist, **meta)

        if (record.get("stored") or {}).get("error") or (
            record.get("status") == "analysed" and not record.get("event_id")
        ):
            _LOGGER.error("live window could not be stored for request %s: %s",
                          current_request_id(), (record.get("stored") or {}).get("error"))
            raise ApiError("storage_error", "This window was analysed but could not be saved. Try again.")

        predictions = record.get("predictions") or {}
        comparison = record.get("comparison") or {}
        severity_block = record.get("severity") or {}
        alert_block = record.get("alert") or {}
        quality = record.get("quality") or {}
        repeat = record.get("repeat") or alert_block
        python_block = predictions.get("python") or {}
        gtm_block = predictions.get("gtm") or {}
        timings = record.get("timings_ms") or {}
        latency_ms = int(timings.get("total") or record.get("elapsed_ms") or 0)

        window_row = LiveWindow(
            session_id=row.id,
            seq=seq,
            captured_at=_parse_iso(captured_at),
            received_at=utcnow(),
            duration_sec=float(duration_sec) if duration_sec is not None else None,
            predicted_class=python_block.get("predicted_class"),
            python_class=python_block.get("predicted_class"),
            python_confidence=python_block.get("confidence"),
            gtm_class=gtm_block.get("predicted_class"),
            gtm_confidence=gtm_block.get("confidence"),
            confidence_difference=comparison.get("confidence_difference"),
            consistency_status=comparison.get("status"),
            quality_verdict=quality.get("verdict"),
            severity=severity_block.get("severity"),
            confirmed=bool(repeat.get("confirmed")),
            consecutive=int(repeat.get("consecutive") or 0),
            needed=int(repeat.get("needed") or 0),
            latency_ms=latency_ms,
            event_id=record.get("event_id"),
        )
        session.add(window_row)
        row.window_count = (row.window_count or 0) + 1
        if python_block.get("predicted_class"):
            row.detection_count = (row.detection_count or 0) + 1
        if alert_block.get("raised"):
            row.alert_count = (row.alert_count or 0) + 1
        row.consecutive_state = {
            "class": repeat.get("class"),
            "consecutive": repeat.get("consecutive"),
            "needed": repeat.get("needed"),
            "confirmed": bool(repeat.get("confirmed")),
            "window_seconds": repeat.get("window_seconds"),
        }
        payload = {
            "seq": seq,
            "prediction": {
                "class": python_block.get("predicted_class"),
                "confidence": python_block.get("confidence"),
                "model_version": python_block.get("model_version"),
            },
            "gtm": {"class": gtm_block.get("predicted_class"),
                    "confidence": gtm_block.get("confidence")},
            "confidence_difference": comparison.get("confidence_difference"),
            "consistency_status": comparison.get("status"),
            "quality": quality.get("verdict"),
            "severity": severity_block.get("severity"),
            "severity_display": severity_block.get("severity_display") or severity_block.get("severity"),
            "confirmed": bool(repeat.get("confirmed")),
            "consecutive": int(repeat.get("consecutive") or 0),
            "needed": int(repeat.get("needed") or 0),
            "alert": alert_block if alert_block.get("raised") else None,
            "event_id": record.get("event_id"),
            "latency_ms": latency_ms,
            "within_budget": bool(record.get("within_budget", True)),
            "requeue_hint_ms": 500,
        }
        data = _session_to_dict(row)
        data["latest_window"] = payload  # the row is committed with the session counters

    return jsonify({"data": payload, "meta": {"session": data}})


def _validated_sample_rate(body: dict) -> int:
    """The window's sample rate, defaulting to the browser's capture rate (16 kHz)."""
    try:
        rate = int(body.get("sample_rate") or 16000)
    except (TypeError, ValueError):
        raise validation_error("sample_rate must be an integer.")
    if rate < 8000 or rate > 96000:
        raise validation_error(
            "sample_rate out of range.", sample_rate={"given": rate, "accepted": "8000..96000"}
        )
    return rate


def _parse_iso(value: str | None):
    if not value:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return None
